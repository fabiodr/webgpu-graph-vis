// GPU force-directed layout (App module, order 0): d3-force-3d semantics on WebGPU compute.
//
// One tick = separate compute passes, so no pass reads what it writes:
//   1. repulsion    Barnes–Hut over the clustering hierarchy (+ virtual leaf splitting, see
//                   forceKernels.js), stackless preorder walk -> nodeForce
//   2. velocity     v = (v + α·(repulsion + links + gravity))·(1 − velocityDecay), blended with the
//                   sphere-layout spring by uniforms.layoutBlend -> nodeVel (gather, no atomics)
//   3. position     x += clamp(v) − centering; pinned (nodePos.w = 1) and dragged nodes -> nodePos
//   4. aggregation  bottom-up per height: centroids + conservative radii -> clusterPos, bhCluster
//   5. flatten      tree-ordered copy of positions/centroids for the next repulsion pass
// Aggregation closes the tick, so clusterPos always matches nodePos after the layout's frame work
// (the edge renderer reads it). Commands (pins, drops) are small uniforms applied by a separate
// pass before the first tick of a frame; nodeState easing runs every frame, even when cooled.
//
// Each tick is one renderer.compute() call (one command buffer), so per-tick uniforms (alpha)
// change between ticks of the same frame. Ticks per frame adapt: ~120 ticks/s for graphs up to
// 2k nodes (settle in ~2.5 s), 60/s from 8k nodes, capped by an estimate of the Barnes–Hut work
// (CPU mirror on a sample) and by frame-time feedback (one tick less below ~45 fps).
// Virtual Barnes–Hut clusters are re-partitioned from app.positions every `resortTicks` ticks
// while the layout is hot, amortised over frames (≤ `resortBudgetMs` per frame).
//
// Simulation control (d3): alpha starts at 1, alphaMin 0.001, alphaDecay = 1 − alphaMin^(1/300),
// alphaTarget 0, velocityDecay 0.4; the layout stops when alpha < alphaMin and nothing is dragged.
// Defaults: charge −30, distanceMin 1, linkDistance = graph.linkDistance, link strength
// 1/min(deg) with d3's degree bias, theta 0.8 (on the bounding radius: accept r < θ·d), a weak
// centering (translates the centroid toward the origin) and an automatic gravity that keeps the
// layout near graph.layoutRadius (large graphs otherwise keep expanding under 1/d repulsion).
//
// Layout modes: 'force' and 'sphere' (the paper's layout). Sphere targets are
//   nodeTarget.xyz · sphereRadius · mix(mix(1, 0.75, state.y), 0.5, state.x)
// (selected node and its neighbours move toward the viewer at the sphere centre). The blend
// uniform animates over ~1 s with smoothstep; forces fade with (1 − blend) while a discrete
// critically damped spring (double eigenvalue λ = e^(−ω·h), framerate independent) fades in.
// Full sphere mode skips repulsion, runs one tick per frame and sleeps once converged (~2.5 s
// after the last change of its targets: nodeStateTarget version, selection, radius, drag, pin).
// Pinned nodes stay fixed in both modes.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//   new ForceLayout( { charge, linkDistance, linkStrength, gravity, centering, theta,
//     velocityDecay, distanceMin, maxIterations, ticksPerFrame, leafSize, mode, autoStart, … } )
//   setGraph( buffers, graph, app ), update( dt, app )             (App module hooks)
//   reheat( alpha = 1 ), pause(), resume(), running, paused, alpha, alphaTarget, stats
//   setLayoutMode( 'force' | 'sphere' ), layoutMode
//   startDrag( index ), drag( index, worldPos: Vector3 ), endDrag( index, { pin = false } )
//   setPinned( index, bool ), isPinned( index ), unpinAll(), pinnedCount
//   resetPositions()                 re-uploads graph.initPos (clears pins) and reheats
//   properties: charge, linkDistance, linkStrength, gravity (null = auto; autoGravity is the
//     automatic value, e.g. gravity = autoGravity * k for a UI multiplier), centering, theta,
//     velocityDecay, distanceMin, ticksPerFrame (0 = adaptive), maxIterations, leafSize,
//     alphaMin, alphaDecay
//   debug: benchmark( ticks ) -> wall-clock ms per tick, readForces( { theta } ) -> nodeForce,
//     estimateVisits( positions ) -> Barnes–Hut walk steps per node (CPU mirror)
//
// Pausing freezes the simulation (force and sphere); dragging while paused moves only the
// dragged node (and its edges). reheat() and drags do not resume a paused layout;
// setLayoutMode() does. While dragging, alphaTarget is 0.3 (alpha ≥ 0.3), as in d3 examples.

import { uniform, uniformArray } from 'three/tsl';
import { Vector3, Vector4 } from 'three/webgpu';
import {
	buildStructure, resortStructure, resortTasks, runResortTask, createPrivateBuffers, aggregateCPU, repulsionCPU,
	aggregateRealKernel, aggregateVirtualKernel, flattenKernel, repulsionKernel, velocityKernel, positionKernel,
	commandKernel, stateKernel, DEFAULT_LEAF_SIZE, DEFAULT_FANOUT,
} from './forceKernels.js';

const MAX_COMMANDS = 16;
const WORK_BUDGET = 3e7; // BH interactions per frame we allow ourselves on a desktop GPU

export class ForceLayout {

	constructor( options = {} ) {

		this.order = 0;
		const o = {
			charge: - 30,
			linkDistance: null, // null = graph.linkDistance
			linkStrength: 1,
			gravity: null, // null = automatic (keeps the layout near graph.layoutRadius)
			centering: 0.3,
			theta: 0.8,
			velocityDecay: 0.4,
			distanceMin: 1,
			maxIterations: 8192,
			ticksPerFrame: 0,
			leafSize: DEFAULT_LEAF_SIZE,
			fanout: DEFAULT_FANOUT,
			alphaMin: 0.001,
			alphaDecay: null,
			sphereOmega: 8, // rad/s of the sphere-mode spring
			blendDuration: 1, // s
			stateTau: 0.06, // s, nodeState easing time constant (~95% after 180 ms)
			resortTicks: 20, // ticks between virtual-cluster re-partition cycles (× (1 − alpha))
			resortBudgetMs: 1.5, // main-thread budget per frame for re-partitioning
			mode: 'force',
			autoStart: true,
			...options,
		};
		this.options = o;

		this.U = {
			alpha: uniform( 1 ),
			charge: uniform( o.charge ),
			distanceMin2: uniform( o.distanceMin * o.distanceMin ),
			theta2: uniform( o.theta * o.theta ),
			maxIterations: uniform( o.maxIterations >>> 0, 'uint' ),
			softIterations: uniform( o.maxIterations >>> 1, 'uint' ),
			linkDistance: uniform( 30 ),
			linkStrength: uniform( o.linkStrength ),
			gravity: uniform( 0 ),
			centering: uniform( o.centering ),
			velocityDecay: uniform( o.velocityDecay ),
			velRescale: uniform( 1 ),
			maxDisp: uniform( 60 ),
			lam2: uniform( 1 ),
			springK: uniform( 0 ),
			dragIndex: uniform( - 1, 'int' ),
			dragHasTarget: uniform( 0 ),
			dragTarget: uniform( new Vector3() ),
			moveIndex: uniform( - 1, 'int' ),
			unpinAll: uniform( 0 ),
			cmdCount: uniform( 0, 'uint' ),
			cmds: uniformArray( Array.from( { length: MAX_COMMANDS }, () => new Vector4() ), 'vec4' ),
			stateEase: uniform( 0.2 ),
		};

		this._charge = o.charge;
		this._linkDistance = o.linkDistance;
		this._gravity = o.gravity;
		this._theta = o.theta;
		this._distanceMin = o.distanceMin;
		this._maxIterations = o.maxIterations >>> 0;
		this._ticksPerFrame = o.ticksPerFrame | 0;
		this._leafSize = o.leafSize;
		this.alphaMin = o.alphaMin;
		this.alphaDecay = o.alphaDecay ?? 1 - Math.pow( o.alphaMin, 1 / 300 );

		this._alpha = 1;
		this._alphaTarget = 0;
		this._paused = ! o.autoStart;
		this._mode = o.mode === 'sphere' ? 'sphere' : 'force';
		this._blendT = this._mode === 'sphere' ? 1 : 0; // linear animation parameter
		this._blend = this._blendT;

		this._dragIndex = - 1;
		this._dragHasTarget = false;
		this._dragDirty = false;
		this._pinned = new Set();
		this._commands = [];
		this._unpinAll = false;
		this._moveIndex = - 1;

		this._dtSmooth = 1 / 60;
		this._K = 1;
		this._kChangeTime = - Infinity;
		this._awakeUntil = 0;
		this._sphereSig = '';
		this._kDyn = 8;
		this._slowTime = 0;
		this._fastTime = 0;
		this._hPrev = 1 / 60;
		this._time = 0;
		this._resortQueue = [];
		this._resortTick = 0;
		this._resortVersion = - 1;
		this._resortPos = null;
		this._resortBusy = false;

		this.ticks = 0;
		this._lastFrameTicks = 0;
		this._needsAggregate = true;
		this.buffers = null;
		this.graph = null;
		this.kernels = null;

	}

	init( app ) {

		this.app = app;

	}

	// ─── graph ────────────────────────────────────────────────────────────────────────────

	setGraph( buffers, graph, app ) {

		this._disposeKernels();
		this.app = app;
		this.buffers = buffers;
		this.graph = graph;
		this.ticks = 0;
		this._buildStructure();

		this._pinned.clear();
		this._commands.length = 0;
		this._unpinAll = false;
		this._moveIndex = - 1;
		this._dragIndex = - 1;
		this._dragHasTarget = false;
		this.U.dragIndex.value = - 1;
		this.U.moveIndex.value = - 1;
		this._alpha = 1;
		this._alphaTarget = 0;
		this._K = 1;
		this._kChangeTime = - Infinity;
		this._kDyn = 8;
		this._sphereSig = '';
		this._wake();
		this._resortVersion = app.positionsVersion;
		buffers.uniforms.layoutBlend.value = this._blend;
		this._applyGraphParams();

	}

	_buildStructure() {

		const graph = this.graph, buffers = this.buffers;
		const t0 = performance.now();
		const struct = buildStructure( graph, { leafSize: this._leafSize, fanout: this.options.fanout } );
		const pos = this.app?.positions?.length >= 4 * graph.N ? this.app.positions : graph.initPos;
		resortStructure( struct, pos );
		this.struct = struct;
		this.priv = createPrivateBuffers( struct );
		const { N, C, TT } = struct;
		const ctx = { buffers, priv: this.priv, U: this.U, N, TT, C, root: graph.root };
		const aggregate = [];
		for ( const r of struct.vHeightRanges ) aggregate.push( aggregateVirtualKernel( { ...ctx, start: r.start, count: r.count, height: r.height } ) );
		for ( const r of graph.heightRanges ) aggregate.push( aggregateRealKernel( { ...ctx, start: r.start, count: r.count, nodeChildren: r.height === 1 } ) );
		this.kernels = {
			aggregate,
			flatten: flattenKernel( ctx ), // tree-ordered copy for the next repulsion pass
			repulsion: repulsionKernel( ctx ),
			velocity: velocityKernel( ctx ),
			position: positionKernel( ctx ),
			command: commandKernel( ctx ),
			state: stateKernel( ctx ),
		};
		this._needsAggregate = true;
		this._resortQueue = []; // no cycle in progress
		this._resortTick = this.ticks;
		this._resortPos = null;

		// work estimate: CPU mirror of the BH walk on a sample of slots at the seeded positions
		this._visitsPerNode = this.estimateVisits( graph.initPos, 192 ).mean;
		this._updateStaticBudget();
		this._structMs = performance.now() - t0;

	}

	_updateStaticBudget() {

		const N = this.graph.N;
		const work = N * this._visitsPerNode + 4 * this.graph.E + 8 * N;
		this._kStatic = Math.max( 1, Math.min( 8, Math.floor( WORK_BUDGET / work ) ) );

	}

	/**
	 * CPU estimate of Barnes–Hut loop iterations per node at positions `pos4` (Float32Array 4N),
	 * from the CPU mirror of the GPU walk on `samples` evenly spaced slots.
	 * @returns {{ mean, p95, max }}
	 */
	estimateVisits( pos4, samples = 256 ) {

		const struct = this.struct, N = struct.N;
		const bh = aggregateCPU( this.graph, struct, pos4 );
		const S = Math.min( N, samples );
		const slots = [];
		for ( let k = 0; k < S; k ++ ) slots.push( Math.floor( ( ( k + 0.5 ) * N ) / S ) );
		const { visits } = repulsionCPU( struct, pos4, bh, { theta: this._theta, charge: this._charge, distanceMin: this._distanceMin, maxIterations: this._maxIterations, slots } );
		const v = slots.map( ( t ) => visits[ struct.nodeOrder[ t ] ] ).sort( ( a, b ) => a - b );
		return { mean: v.reduce( ( a, b ) => a + b, 0 ) / Math.max( 1, S ), p95: v[ Math.floor( 0.95 * ( S - 1 ) ) ], max: v[ S - 1 ] };

	}

	_applyGraphParams() {

		const g = this.graph;
		if ( ! g ) return;
		const L = this._linkDistance ?? g.linkDistance ?? 30;
		this.U.linkDistance.value = L;
		this.U.maxDisp.value = Math.max( 2 * L, 0.05 * g.layoutRadius );
		this.U.gravity.value = this._gravity ?? this._autoGravity();

	}

	/** The automatic gravity for the current graph and charge (what `gravity = null` uses). */
	get autoGravity() {

		return this.graph ? this._autoGravity() : 0;

	}

	/**
	 * Gravity that roughly balances the far-field repulsion at the radius prepare.js expects
	 * (graph.layoutRadius): repulsion on an outer node ≈ |charge|·N / R, gravity = g·R. The
	 * (N/1000)^¼ factor (measured) compensates the growing far field of large graphs
	 * (converged radius ≈ 0.85–1.2 × layoutRadius from 77 to 20k nodes).
	 */
	_autoGravity() {

		const g = this.graph;
		const R = g.layoutRadius;
		const large = Math.max( 1, Math.pow( g.N / 1000, 0.25 ) );
		return Math.min( 0.5, ( 0.35 * large * Math.abs( this._charge ) * g.N ) / ( R * R ) );

	}

	// ─── frame ────────────────────────────────────────────────────────────────────────────

	update( dt, app ) {

		if ( ! this.buffers || ! this.kernels ) return;
		const renderer = app.renderer;
		this._time += dt;
		this._dtSmooth += ( dt - this._dtSmooth ) * 0.1;
		const U = this.U;

		// layout-mode blend (smoothstep over blendDuration)
		const goal = this._mode === 'sphere' ? 1 : 0;
		if ( this._blendT !== goal ) {

			const step = dt / Math.max( 1e-3, this.options.blendDuration );
			this._blendT = goal > this._blendT ? Math.min( goal, this._blendT + step ) : Math.max( goal, this._blendT - step );
			const t = this._blendT;
			this._blend = t * t * ( 3 - 2 * t );
			this.buffers.uniforms.layoutBlend.value = this._blend;
			this._wake();

		}

		const b = this._blend;
		U.stateEase.value = 1 - Math.exp( - dt / Math.max( 1e-3, this.options.stateTau ) );

		// full sphere mode sleeps once converged; wake on CPU-visible changes of its targets
		if ( b >= 1 ) {

			const u = this.buffers.uniforms;
			const sig = `${u.sphereRadius.value}|${u.selectedIndex.value}|${this.buffers.nodeStateTarget.value.version}`;
			if ( sig !== this._sphereSig ) {

				this._sphereSig = sig;
				this._wake();

			}

		}

		const ticking = this._isTicking();
		const dragging = this._dragIndex >= 0;

		// re-partition virtual clusters first: a changed nodeOrder needs re-aggregation (and a
		// fresh tree-ordered copy) before the next repulsion pass
		if ( ticking ) this._maybeResort( app, b );

		// commands (pins, drops; explicit drag moves while not ticking)
		const pre = [];
		if ( ! ticking && dragging && this._dragHasTarget && this._dragDirty ) this._moveIndex = this._dragIndex;
		this._dragDirty = false;
		if ( this._commands.length || this._unpinAll || this._moveIndex >= 0 ) {

			const n = Math.min( MAX_COMMANDS, this._commands.length );
			for ( let k = 0; k < n; k ++ ) {

				const [ index, value ] = this._commands[ k ];
				U.cmds.array[ k ].set( index, value, 0, 0 );

			}

			this._commands.splice( 0, n );
			U.cmdCount.value = n;
			U.unpinAll.value = this._unpinAll ? 1 : 0;
			U.moveIndex.value = this._moveIndex;
			if ( this._moveIndex >= 0 ) this._needsAggregate = true; // a node moved
			this._unpinAll = false;
			this._moveIndex = - 1;
			pre.push( this.kernels.command );

		}

		if ( this._needsAggregate ) {

			pre.push( ...this.kernels.aggregate, this.kernels.flatten );
			this._needsAggregate = false;

		}

		let K = 0;
		if ( ticking ) {

			K = this._ticksForFrame( dt, b );
			// sphere spring: per-tick decay of a critically damped system, h = seconds per tick
			const h = this._dtSmooth / K;
			const lam = Math.exp( - this.options.sphereOmega * h );
			U.lam2.value = lam * lam;
			U.springK.value = ( 1 - lam ) * ( 1 - lam );
			// keep the spring's velocity (per tick) continuous when the tick length changes
			const ratio = Math.min( 4, Math.max( 0.25, h / this._hPrev ) );
			this._hPrev = h;
			U.dragIndex.value = dragging ? this._dragIndex : - 1;
			U.dragHasTarget.value = this._dragHasTarget ? 1 : 0;
			const tickList = b < 1
				? [ this.kernels.repulsion, this.kernels.velocity, this.kernels.position, ...this.kernels.aggregate, this.kernels.flatten ]
				: [ this.kernels.velocity, this.kernels.position, ...this.kernels.aggregate ];
			for ( let k = 0; k < K; k ++ ) {

				this._alpha += ( this._alphaTarget - this._alpha ) * this.alphaDecay;
				U.alpha.value = this._alpha;
				U.velRescale.value = k === 0 ? 1 + ( ratio - 1 ) * b : 1;
				renderer.compute( k === 0 && pre.length ? [ ...pre, ...tickList ] : tickList );
				if ( k === 0 && pre.length ) this._clearCommandUniforms(); // commands are one-shot

			}

			this.ticks += K;

		} else if ( pre.length ) {

			renderer.compute( pre );
			this._clearCommandUniforms();

		}

		this._lastFrameTicks = K;
		renderer.compute( this.kernels.state );

	}

	/** Whether this frame advances the simulation (see `running`). */
	_isTicking() {

		if ( this._paused ) return false;
		const b = this._blend;
		const dragging = this._dragIndex >= 0;
		if ( b >= 1 ) return dragging || this._time < this._awakeUntil;
		if ( b > 0 || this._mode === 'sphere' ) return true;
		return this._alpha >= this.alphaMin || dragging;

	}

	/** Keeps full sphere mode stepping for a while (its spring converges within ~2 s). */
	_wake( seconds = 2.5 ) {

		this._awakeUntil = Math.max( this._awakeUntil, this._time + seconds );

	}

	_clearCommandUniforms() {

		const U = this.U;
		U.cmdCount.value = 0;
		U.unpinAll.value = 0;
		U.moveIndex.value = - 1;

	}

	_ticksForFrame( dt, b ) {

		if ( this._ticksPerFrame > 0 ) return this._ticksPerFrame;
		if ( b >= 1 ) return 1;
		// frame-time feedback: drop a tick when the frame rate falls below ~45 fps, add one back
		// after 3 s above ~55 fps
		if ( this._dtSmooth > 1 / 45 ) {

			this._slowTime += dt; this._fastTime = 0;

		} else if ( this._dtSmooth < 1 / 55 ) {

			this._fastTime += dt; this._slowTime = 0;

		} else {

			this._slowTime = 0; this._fastTime = 0;

		}

		if ( this._slowTime > 0.5 && this._kDyn > 1 ) {

			this._kDyn = Math.max( 1, Math.min( this._kDyn, this._K ) - 1 );
			this._slowTime = 0;

		} else if ( this._fastTime > 3 && this._kDyn < 8 ) {

			this._kDyn ++;
			this._fastTime = 0;

		}

		const N = this.graph.N;
		const tps = N <= 2000 ? 120 : N >= 8000 ? 60 : 120 - ( 60 * ( N - 2000 ) ) / 6000;
		const want = Math.max( 1, Math.round( tps * this._dtSmooth ) );
		const K = Math.max( 1, Math.min( want, this._kStatic, this._kDyn ) );
		if ( K !== this._K && this._time - this._kChangeTime > 0.5 ) {

			this._K = K;
			this._kChangeTime = this._time;

		}

		return Math.max( 1, Math.min( this._K, this._kStatic, this._kDyn ) );

	}

	/**
	 * Re-partitions virtual Barnes–Hut clusters from recent position readbacks. A cycle (one
	 * coarse-to-fine pass over all split clusters) starts every resortTicks·(1 − alpha) ticks
	 * (continuously while hot, when nodes move most); within a cycle each frame spends at most
	 * `resortBudgetMs` of main-thread time, and only the touched slot range of nodeOrder is
	 * uploaded. Stale groups never break correctness (radii are recomputed every tick), they
	 * only make the walk longer.
	 */
	_maybeResort( app, b ) {

		const struct = this.struct;
		if ( ! struct.groups.length || b >= 1 ) return;
		if ( ! this._resortQueue.length ) {

			const cadence = Math.max( 1, Math.round( this.options.resortTicks * ( 1 - this._alpha ) ) );
			if ( this.ticks - this._resortTick < cadence ) return;
			if ( this._alpha < 0.005 && this._dragIndex < 0 ) return;
			if ( app.options?.readback !== false ) {

				if ( ! app.positions || app.positionsVersion === this._resortVersion ) return;
				this._resortVersion = app.positionsVersion;
				this._resortPos = app.positions; // updated in place by later readbacks: fine

			} else {

				// no shared readback: fetch positions ourselves, start the cycle when they arrive
				if ( this._resortBusy ) return;
				this._resortBusy = true;
				const buffers = this.buffers;
				app.renderer.getArrayBufferAsync( buffers.nodePos.value ).then( ( ab ) => {

					if ( buffers === this.buffers && ! this._resortQueue.length ) {

						this._resortPos = new Float32Array( ab );
						this._resortQueue = resortTasks( this.struct );
						this._resortTick = this.ticks;

					}

				} ).catch( () => {} ).finally( () => {

					this._resortBusy = false;

				} );
				return;

			}

			this._resortQueue = resortTasks( struct );
			this._resortTick = this.ticks;

		}

		const pos = this._resortPos;
		if ( ! pos || pos.length < 4 * struct.N ) {

			this._resortQueue = [];
			return;

		}

		const t0 = performance.now();
		let lo = Infinity, hi = - Infinity;
		while ( this._resortQueue.length ) {

			const r = runResortTask( struct, pos, this._resortQueue.shift(), this._resortQueue );
			if ( r.changed ) {

				lo = Math.min( lo, r.lo );
				hi = Math.max( hi, r.hi );

			}

			// twice the budget while hot: nodes move fastest then and stale groups cost the most
			if ( performance.now() - t0 > this.options.resortBudgetMs * ( this._alpha > 0.25 ? 2 : 1 ) ) break;

		}

		if ( hi > lo ) this._uploadOrder( lo, hi );

	}

	_uploadOrder( lo = 0, hi = this.struct.N ) {

		const attr = this.priv.nodeOrder.value;
		attr.array.set( this.struct.nodeOrder.subarray( lo, hi ), lo );
		if ( lo > 0 || hi < this.struct.N ) attr.addUpdateRange( lo, hi - lo );
		else attr.clearUpdateRanges();
		attr.needsUpdate = true;
		this._needsAggregate = true;

	}

	// ─── simulation control ───────────────────────────────────────────────────────────────

	get running() {

		return !! this.buffers && this._isTicking();

	}

	get paused() {

		return this._paused;

	}

	get alpha() {

		return this._alpha;

	}

	set alpha( v ) {

		this._alpha = Math.max( 0, Math.min( 1, v ) );

	}

	get alphaTarget() {

		return this._alphaTarget;

	}

	set alphaTarget( v ) {

		this._alphaTarget = Math.max( 0, Math.min( 1, v ) );

	}

	reheat( alpha = 1 ) {

		this._alpha = Math.max( 0, Math.min( 1, alpha ) );
		this._wake();
		return this;

	}

	pause() {

		this._paused = true;
		return this;

	}

	resume() {

		this._paused = false;
		this._wake();
		return this;

	}

	get layoutMode() {

		return this._mode;

	}

	setLayoutMode( mode ) {

		const m = mode === 'sphere' ? 'sphere' : 'force';
		if ( m === this._mode ) return this;
		this._mode = m;
		if ( m === 'force' ) {

			this.reheat( 1 );
			this._needsAggregate = true; // full sphere mode skips the flatten pass

		}

		this.resume();
		return this;

	}

	resetPositions() {

		if ( ! this.buffers ) return this;
		this.buffers.setPositions( this.graph.initPos );
		this._pinned.clear();
		this._commands.length = 0;
		this._unpinAll = false;
		this._needsAggregate = true;
		resortStructure( this.struct, this.graph.initPos );
		this._uploadOrder();
		this._resortQueue = [];
		this._resortTick = this.ticks;
		this._wake();
		this.reheat( 1 );
		return this;

	}

	// ─── drag & pin ───────────────────────────────────────────────────────────────────────

	startDrag( index ) {

		if ( ! this._valid( index ) ) return;
		this._dragIndex = index;
		this._dragHasTarget = false; // held in place (zero velocity) until the first drag()
		this._wake();
		this.U.dragIndex.value = - 1;
		this._alphaTarget = 0.3;
		this._alpha = Math.max( this._alpha, 0.3 );

	}

	drag( index, worldPos ) {

		if ( ! this._valid( index ) ) return;
		if ( index !== this._dragIndex ) this.startDrag( index );
		this.U.dragTarget.value.copy( worldPos );
		this._dragHasTarget = true;
		this._dragDirty = true;
		this._wake();

	}

	endDrag( index, { pin = false } = {} ) {

		if ( this._dragIndex >= 0 && index !== undefined && index !== null && index !== this._dragIndex ) return; // not the dragged node
		const i = this._dragIndex >= 0 ? this._dragIndex : index;
		if ( this._dragIndex >= 0 && this._dragHasTarget ) this._moveIndex = i; // final placement, even without a tick
		this._dragIndex = - 1;
		this._dragHasTarget = false;
		this.U.dragIndex.value = - 1;
		this._alphaTarget = 0;
		this._wake();
		if ( pin && this._valid( i ) ) this.setPinned( i, true );

	}

	setPinned( index, pinned ) {

		if ( ! this._valid( index ) ) return;
		if ( pinned ) this._pinned.add( index );
		else {

			this._pinned.delete( index );
			this._alpha = Math.max( this._alpha, 0.1 ); // let it relax

		}

		this._commands.push( [ index, pinned ? 1 : 0 ] );
		this._wake();

	}

	isPinned( index ) {

		return this._pinned.has( index );

	}

	get pinnedCount() {

		return this._pinned.size;

	}

	unpinAll() {

		this._commands.length = 0;
		this._unpinAll = true;
		if ( this._pinned.size ) this._alpha = Math.max( this._alpha, 0.1 );
		this._pinned.clear();
		this._wake();

	}

	_valid( index ) {

		return this.graph && Number.isInteger( index ) && index >= 0 && index < this.graph.N;

	}

	// ─── parameters ───────────────────────────────────────────────────────────────────────

	get charge() {

		return this._charge;

	}

	set charge( v ) {

		this._charge = v;
		this.U.charge.value = v;
		if ( this._gravity === null && this.graph ) this.U.gravity.value = this._autoGravity();

	}

	get linkDistance() {

		return this.U.linkDistance.value;

	}

	set linkDistance( v ) {

		this._linkDistance = v;
		this._applyGraphParams();

	}

	get linkStrength() {

		return this.U.linkStrength.value;

	}

	set linkStrength( v ) {

		this.U.linkStrength.value = v;

	}

	/** Gravity toward the origin (d3 forceX/Y/Z strength); null restores the automatic value. */
	get gravity() {

		return this.U.gravity.value;

	}

	set gravity( v ) {

		this._gravity = v;
		this._applyGraphParams();

	}

	get centering() {

		return this.U.centering.value;

	}

	set centering( v ) {

		this.U.centering.value = v;

	}

	get theta() {

		return this._theta;

	}

	set theta( v ) {

		this._theta = Math.max( 0, v );
		this.U.theta2.value = this._theta * this._theta;

	}

	get velocityDecay() {

		return this.U.velocityDecay.value;

	}

	set velocityDecay( v ) {

		this.U.velocityDecay.value = Math.max( 0, Math.min( 1, v ) );

	}

	get distanceMin() {

		return this._distanceMin;

	}

	set distanceMin( v ) {

		this._distanceMin = Math.max( 1e-3, v );
		this.U.distanceMin2.value = this._distanceMin * this._distanceMin;

	}

	get ticksPerFrame() {

		return this._ticksPerFrame;

	}

	/** 0 = adaptive; otherwise a fixed number of ticks per frame. */
	set ticksPerFrame( v ) {

		this._ticksPerFrame = Math.max( 0, v | 0 );

	}

	get maxIterations() {

		return this._maxIterations;

	}

	set maxIterations( v ) {

		this._maxIterations = Math.max( 16, v >>> 0 );
		this.U.maxIterations.value = this._maxIterations;
		this.U.softIterations.value = this._maxIterations >>> 1;

	}

	get leafSize() {

		return this._leafSize;

	}

	/** Max nodes per virtual Barnes–Hut leaf (rebuilds the structure and kernels). */
	set leafSize( v ) {

		v = Math.max( 2, v | 0 );
		if ( v === this._leafSize ) return;
		this._leafSize = v;
		if ( this.graph ) {

			this._disposeKernels();
			this._buildStructure();

		}

	}

	get stats() {

		const s = this.struct;
		return {
			running: this.running,
			paused: this._paused,
			alpha: this._alpha,
			alphaTarget: this._alphaTarget,
			mode: this._mode,
			blend: this._blend,
			ticks: this.ticks,
			ticksPerFrame: this._lastFrameTicks,
			kStatic: this._kStatic,
			kDynamic: this._kDyn,
			dragging: this._dragIndex,
			pinned: this._pinned.size,
			gravity: this.U.gravity.value,
			bh: s ? {
				treeSize: s.TT, virtualClusters: s.V, virtualHeights: s.vHeightRanges.length, leafSize: s.leafSize,
				splitClusters: s.groups.length, visitsPerNode: this._visitsPerNode, buildMs: this._structMs,
			} : null,
		};

	}

	// ─── debug / measurement ──────────────────────────────────────────────────────────────

	/**
	 * Runs `ticks` full ticks back to back (state unchanged otherwise) and returns wall-clock ms
	 * per tick after the GPU finished, plus repulsion-only ms per pass.
	 */
	async benchmark( ticks = 20, { repulsionOnly = false } = {} ) {

		const renderer = this.app.renderer;
		const device = renderer.backend.device;
		const k = this.kernels;
		const list = repulsionOnly ? [ k.repulsion ] : [ k.repulsion, k.velocity, k.position, ...k.aggregate, k.flatten ];
		renderer.compute( list ); // warm-up (pipeline compile)
		await device.queue.onSubmittedWorkDone();
		const t0 = performance.now();
		for ( let k = 0; k < ticks; k ++ ) renderer.compute( list );
		await device.queue.onSubmittedWorkDone();
		return ( performance.now() - t0 ) / ticks;

	}

	/** Runs aggregation + repulsion once with the given theta and reads nodeForce back. */
	async readForces( { theta = this._theta } = {} ) {

		const renderer = this.app.renderer;
		const prev = this.U.theta2.value;
		this.U.theta2.value = theta * theta;
		renderer.compute( [ ...this.kernels.aggregate, this.kernels.flatten, this.kernels.repulsion ] );
		this.U.theta2.value = prev;
		const ab = await renderer.getArrayBufferAsync( this.buffers.nodeForce.value );
		return new Float32Array( ab.slice( 0 ) );

	}

	// ─── lifecycle ────────────────────────────────────────────────────────────────────────

	_disposeKernels() {

		if ( ! this.kernels ) return;
		const k = this.kernels;
		for ( const node of [ ...k.aggregate, k.flatten, k.repulsion, k.velocity, k.position, k.command, k.state ] ) node.dispose?.();
		if ( this.priv ) for ( const b of Object.values( this.priv ) ) b.value?.dispose?.();
		this.kernels = null;
		this.priv = null;

	}

	dispose() {

		this._disposeKernels();
		this.buffers = null;
		this.graph = null;

	}

}
