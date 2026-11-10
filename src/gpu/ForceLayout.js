// GPU graph layout (App module, order 0): a hierarchy-aware force model on WebGPU compute, plus
// target layouts (sphere, radial, stress, custom) that share one mechanism with it.
//
// ─── Model ("H": bounded repulsion, linear attraction, Jacobi-preconditioned) ────────────────
//   F_i = − Σ_j β·m_i·m_j·(x_j − x_i) / (1 + |x_j − x_i|²/u²)²        bounded short-range repulsion
//         + Σ_{j ∈ N(i)} λ^lca(i,j)·(x_j − x_i)                       attraction, rest length 0
//         + κ·(c_A(i) − x_i)                                          cohesion to the finest cluster
//         − g·x_i                                                     gravity
//   m_i = (1 + deg_i/avgDeg)^p      masses (hubs repel more)
//   M_i = 1 + κ + g + Σ_j λ^lca + ρ·S_i,  S_i = Σ_j β·m_i·m_j/(1 + …)²   diagonal stiffness
//   v_i ← (v_i + α_i·F_i/M_i)·(1 − damping),  |v_i| ≤ maxDisp,  x_i ← x_i + v_i
// lca(i,j) is the hierarchy level at which i and j first share a cluster (0 = same finest
// cluster), so links between clusters pull less than links inside them. The repulsion has no
// singularity and a finite range u, the attraction no rest length: the forces are smooth, every
// node has a well conditioned local problem, and F/M is close to a Newton step, so α = 1 is
// stable for any degree. The model is homogeneous: scaling positions and u together gives the
// same layout, which the *auto-scale* uses instead of a hand-tuned gravity: every tick positions,
// velocities and u are multiplied by f = clamp(ext^−0.06, 0.985, 1.015), ext = rms radius /
// target rms, so the layout fills a ball of radius graph.layoutRadius·layoutSize (extent 1.0).
// The rms radius comes from the position readback (app.positions) and is corrected for the scale
// applied since that readback was requested. Pinned nodes keep their place: they are neither
// scaled nor measured, so the free nodes fill the ball and layoutSize / repulsion keep working
// with pins (CPU mirror, hsbm-500, 1–125 of 500 nodes pinned: extent of the free nodes
// 1.497–1.505 at layoutSize 1.5, 1.000–1.004 back at 1, pins exact, stress unchanged). The scale
// is frozen while a node is dragged, while more than a quarter of the nodes is pinned (the free
// rest could only follow by distorting the layout), in target layouts and once alpha < 0.002.
// Scaling needs ticks, not heat: while the extent is still more than 4% off, the schedule waits
// at alpha 0.02 instead of cooling to rest (at most 150 ticks per reheat), so the extent is on
// target when the layout settles, also after a parameter change that moves the natural size
// several times (measured: hub spacing 2 -> 0 on a star, kernel unit × 4.8).
//
// Schedule: α = 1 for the first 30% of the tick budget (default 200 ticks), then exponential
// cooling to alphaMin = 0.001; the layout stops ticking when it has cooled and nothing is
// dragged. Dragging does not reheat the graph: the dragged node is a heat source (heat 1) that
// spreads over links (× 0.6 per hop, × 0.92 per tick), and α_i = max(α, α_drag·heat_i), so only
// the neighbourhood of the node moves. α_drag is 1 while the pointer moves, fades (× 0.985 per
// tick) while it rests, and after the release the heat tail runs out within ~120 ticks.
//
// ─── One tick (separate compute passes: no pass reads what it writes; all are gathers) ───────
//   1. repulsion    Barnes–Hut walk over the clustering hierarchy with virtual leaf clusters
//                   (forceKernels.js) -> nodeForce = (F_rep, S)
//   2. links        attraction, cohesion, gravity, preconditioner, heat diffusion
//                   -> nodeForce = (F/M, heat)
//   3. velocity     force step, blended with the critically damped target spring -> nodeVel
//   4. position     clamp, integrate, centering, auto-scale, pins, drag, flat lock -> nodePos
//   5. aggregation  bottom-up per height: mass-weighted centres + conservative radii
//                   -> clusterPos, bhCluster
//   6. flatten      tree-ordered copy of positions / centres for the next repulsion pass
// Aggregation closes the tick, so clusterPos always matches nodePos after the layout's frame
// work. Commands (pins, drops) are small uniforms applied by a separate pass before the first
// tick of a frame; the nodeState / nodeColor easing pass runs every frame, even at rest.
// Each tick is one renderer.compute() call, so per-tick uniforms (alpha, scale) change between
// ticks of a frame. Ticks per frame adapt: ~120 ticks/s up to 2k nodes, 60/s from 8k nodes,
// capped by an estimate of the Barnes–Hut work and by frame-time feedback. Virtual clusters are
// re-partitioned from position readbacks every `resortTicks` ticks while hot (amortised,
// ≤ resortBudgetMs per frame); their exact masses are re-uploaded with the changed ranges.
//
// ─── Layout modes ────────────────────────────────────────────────────────────────────────────
//  One mechanism: buffers.nodeTarget holds an ABSOLUTE target per node (xyz, w = weight) and
//  uniforms.targetBlend (eased 0..1) mixes the force step with a critically damped spring to the
//  target (double eigenvalue λ = e^(−ω·h): no overshoot, frame-rate independent). Changing mode
//  rewrites the targets and/or eases the blend, so every switch animates from where the nodes are.
//    'force'   the model above in 3D
//    'flat'    the same physics in the z = 0 plane: z is not simulated but locked to
//              zMem·(1 − flat), zMem = the z of the node when the mode was entered (the layout is
//              squashed into the plane and unfolds again when the mode is left). The auto-scale
//              targets a disc of radius flatRadius(R, N) = R·max(1.3, 0.5·N^(1/6)) (1.3·R up to
//              ~300 nodes, 1.8·R at 2k, 2.1·R at 5k, 2.6·R at 20k): a plane has less room, and
//              node sizes are fixed in world units. The extent alone does not keep nodes apart:
//              with the 3D parameters the nodes of a dense community sit inside the linear core
//              of the bounded repulsion and pairs overlap. From 300 nodes on the parameters are
//              therefore adapted with the mode's blend (flatTuning in forceKernels.js; full
//              strength from 2000 nodes: β × 8, λ → λ², κ × 2; `repulsion`, `clusterSeparation`
//              and `cohesion` keep reporting the user's values, stats.beta the effective β).
//              Measured (CPU mirror, from the seed, three runs per graph): stress 0.128 / 0.177 /
//              0.202 on lesmis / hsbm-500 / lfr-2k (3D: 0.071 / 0.097 / 0.126), median distance
//              to the nearest node 3.6–7 node diameters. Overlapping node pairs (node radii of
//              the renderer): 0 in every run on lfr-2k, ws-2k, lattice-12, rgg-3k, hsbm-5k and
//              ba-5k (without the adaptation 2–5 on lfr-2k, 15–22 on hsbm-5k, 13–23 on ba-5k);
//              known limit: 3–9 pairs of 10000 nodes on lfr-10k (150–167 without).
//    'sphere'  the paper's layout: targets graph.sphereDir·sphereRadius; the selected node and its
//              neighbours move toward the centre (× 0.5 / × 0.75). uniforms.layoutBlend
//              ("sphere-ness") is 1 only here.
//    'radial'  BFS shells around a root (graph/layouts.js radialLayout, main thread, a few ms).
//              setLayoutMode('radial', { root, dim = 3 }); without `root` the selected node (else
//              the node of highest degree) is used and the layout follows the selection. The
//              outermost shell has the layout radius (recomputed when layoutSize changes).
//    'stress'  sparse stress layout (sgd2Layout in the data worker: 200 pivots up to 2000 nodes,
//              50 above); asynchronous: the previous layout stays until the result arrives
//              (`pending`), results are cached per graph and options and rescaled (no new
//              request) when layoutSize changes.
//    'custom'  entered by setTargets( positions4, { blend, weights } ).
//  In a pure target layout (blend 1, all weights 1) repulsion and links are skipped, one tick runs
//  per frame and the module sleeps ~2.5 s after the last change of its targets. Returning to
//  'force' or 'flat' reheats (α = 1) from the current positions. Pinned nodes stay fixed in every
//  mode. A 'layout' event ({ mode, settled, pending }) is dispatched on app.events when the mode
//  changes and whenever `settled` flips.
//
// ─── Parameters (live; the measured effect is from the CPU mirror on lesmis / hsbm-500 / lfr-2k,
//     defaults: stress 0.071 / 0.097 / 0.126, silhouette 0.615 / 0.560 / 0.467) ────────────────
//   clusterSeparation λ  0.7   (0, 1]. Weight of a link per hierarchy level it crosses. 1 = plain
//                              graph layout. 1 → 0.3: silhouette 0.56/0.48/0.36 → 0.70/0.66/0.64,
//                              stress 0.062/0.090/0.122 → 0.099/0.126/0.140.
//   cohesion κ           0.5   ≥ 0. Pull of every node to the centre of its finest cluster.
//                              0 → 2: silhouette +0.07..0.09, stress +1..15%.
//   hubSpacing p         1     0..2. Mass exponent: free space around hubs. Distance of the top-1%
//                              degree nodes to their 5 nearest nodes relative to the median node:
//                              p 0: 1.05/0.77/0.64, p 1: 1.13/0.95/1.02, p 2: 1.18/1.09/1.33.
//   layoutSize s         1     > 0. Radius of the layout in units of graph.layoutRadius: the
//                              auto-scale target of 'force' and 'flat', and the radius radial and
//                              stress targets are (re)written for. Not applied to 'sphere'
//                              (uniforms.sphereRadius) and to custom targets (the caller's units).
//   damping              0.3   0..1 velocity decay per tick. 0.15–0.4: metrics within ±4%;
//                              0.6: stress +4..9% (too slow for the tick budget).
//   repulsion β          8     > 0. Because of the auto-scale β does not change the size but the
//                              range of the repulsion relative to the layout (u/linkDistance 1.1–1.4
//                              at β 2, 0.5–0.7 at β 32): small β = tighter clusters (silhouette
//                              +0.03 at β 2, −0.01 at β 32), stress within ±3%.
//   gravity g            null  null = 0.03. Numeric values are used as g. 0 … 0.12: all metrics
//                              within ±3% on connected graphs; it mainly holds components together.
//   theta                0.8   Barnes–Hut accuracy (accept a cluster when radius ≤ θ·distance)
//   tickBudget           200   ticks of a full run (30% plateau, 70% cooling)
//   setPreset( name )          'default' (λ 0.7, κ 0.5), 'graph-only' (λ 1, κ 0: lowest stress
//                              0.062/0.090/0.122, silhouette 0.51/0.46/0.33), 'communities'
//                              (λ 0.4, κ 1.5: silhouette 0.70/0.66/0.62, stress 0.097/0.121/0.136)
//  Setting one of these reheats to α ≥ 0.3 (layoutSize: ≥ 0.5, damping: not at all;
//  options.autoReheat = false disables that).
//
//  Parameters of the previous d3-force model are still accepted:
//   charge          default −30. Scales the repulsion: β_eff = repulsion·|charge|/30 (≥ 0.25).
//   linkDistance    null = graph.linkDistance. Links have no rest length any more; the value
//                   scales the layout: radius = graph.layoutRadius·linkDistance/graph.linkDistance
//                   ·layoutSize (in the same modes as layoutSize).
//   linkStrength    1. Multiplies all link weights.
//   gravity         see above; autoGravity is the automatic value (0.03), so
//                   gravity = autoGravity·k still works for a UI multiplier.
//   velocityDecay   alias of damping.
//   distanceMin     accepted and ignored (the repulsion is bounded).
//   alpha, alphaTarget, alphaMin, alphaDecay   as in d3 (alphaDecay follows tickBudget unless set)
//
// ─── Public API ──────────────────────────────────────────────────────────────────────────────
//   new ForceLayout( { data?: DataClient (for 'stress'; a private one is created when missing),
//     mode = 'force', autoStart = true, autoReheat = true, any parameter above, maxIterations,
//     ticksPerFrame, leafSize, fanout, springOmega = 8, blendDuration = 1, … } )
//   setGraph( buffers, graph, app ), update( dt, app ), dispose()            App module hooks
//   reheat( alpha = 1 ) -> this      restart the schedule at alpha (plateau ∝ alpha)
//   pause() / resume() -> this, paused, running (advancing this frame), settled (at rest in the
//     current mode: cooled, nothing dragged, no blend in progress), alpha, ticks, stats
//   setLayoutMode( mode: 'force'|'flat'|'sphere'|'radial'|'stress', options? ) -> this
//     options: radial { root: node index, dim: 2|3, follow: boolean }, stress { pivots, iters,
//     seed, dim }. Resumes a paused layout. layoutMode, pending (stress result outstanding),
//     radialRoot. LAYOUT_MODES: [ { id, label, description } ] for the UI.
//   setTargets( positions4: Float32Array( 4N ) (xyz world units), { blend = 1, weights = null:
//     Float32Array( N ) 0..1 } ) -> this      mode 'custom'. blend < 1 or weights < 1 keep the
//     forces running: the targets then act as a soft pull while the simulation is hot, and
//     as it cools nodes with weight > 0 end on their targets. setTargets( null ) = 'force'.
//     Custom targets are used as given: layoutSize / linkDistance do not rescale them.
//   setPreset( name ) -> this, preset ('custom' when the parameters match none), LAYOUT_PRESETS
//   startDrag( index ), drag( index, worldPos: Vector3 ), endDrag( index?, { pin = false } )
//   setPinned( index, bool ), isPinned( index ), unpinAll(), pinnedCount
//   resetPositions( positions4 = graph.initPos )   re-uploads positions (xyz; w is ignored: all
//     pins are cleared, also the pin flags a position readback carries in w), reheats
//   debug: benchmark( ticks ) -> ms per tick; readForces( { theta, links } ) -> Float32Array( 4N )
//     (F_rep, S) or, with links, (F/M, heat); estimateVisits( positions4 ) -> { mean, p95, max }
//
// Pausing freezes the simulation in every mode; dragging while paused moves only the dragged
// node. reheat() and drags do not resume a paused layout; setLayoutMode() and setTargets() do.

import { uniform, uniformArray } from 'three/tsl';
import { Vector3, Vector4 } from 'three/webgpu';
import {
	buildStructure, resortStructure, resortTasks, runResortTask, setMasses, createPrivateBuffers, aggregateCPU, repulsionCPU,
	aggregateRealKernel, aggregateVirtualKernel, flattenKernel, repulsionKernel, linksKernel, velocityKernel, positionKernel,
	commandKernel, stateKernel, zSnapshotKernel, AlphaSchedule, ScaleController, rmsRadius, flatRadius, flatTuning,
	DEFAULT_LEAF_SIZE, DEFAULT_FANOUT, MASS_ONE, LEVEL_MAX, MODEL,
} from './forceKernels.js';
import { radialLayout } from '../graph/layouts.js';
import { DataClient } from '../data/client.js';

const MAX_COMMANDS = 16;
const WORK_BUDGET = 3e7; // BH interactions per frame we allow ourselves on a desktop GPU
const DEFAULT_GRAVITY = 0.03;

/** Layout modes for the UI (setLayoutMode ids; 'custom' is entered through setTargets). */
export const LAYOUT_MODES = [
	{ id: 'force', label: 'Force 3D', description: 'Hierarchy-aware force layout in space' },
	{ id: 'flat', label: 'Force 2D', description: 'The same force layout in a plane' },
	{ id: 'sphere', label: 'Sphere', description: 'Clusters on a sphere; the selection moves inward' },
	{ id: 'radial', label: 'Radial', description: 'Shells by distance from the selected node' },
	{ id: 'stress', label: 'Stress', description: 'Distances match graph distances (meshes, rings, trees)' },
];

/** Parameter presets (setPreset). */
export const LAYOUT_PRESETS = {
	default: { clusterSeparation: 0.7, cohesion: 0.5 },
	'graph-only': { clusterSeparation: 1, cohesion: 0 },
	communities: { clusterSeparation: 0.4, cohesion: 1.5 },
};

const FORCE_MODES = new Set( [ 'force', 'flat' ] );
const clamp01 = ( v ) => Math.max( 0, Math.min( 1, v ) );
const clampHub = ( v ) => ( Number.isFinite( v ) ? Math.max( 0, Math.min( 2, v ) ) : 1 ); // measured range of p

// smoothstep animation of one blend value
class Ease {

	constructor( value = 0 ) {

		this.snap( value );

	}

	snap( value ) {

		this.value = this.from = this.to = value;
		this.t = 1;

	}

	set( to ) {

		if ( to === this.to ) return;
		this.from = this.value;
		this.to = to;
		this.t = 0;

	}

	get active() {

		return this.t < 1;

	}

	/** Advances by `dt` (fraction of the duration); returns whether the value changed. */
	step( dt ) {

		if ( this.t >= 1 ) return false;
		this.t = Math.min( 1, this.t + dt );
		const s = this.t * this.t * ( 3 - 2 * this.t );
		this.value = this.t >= 1 ? this.to : this.from + ( this.to - this.from ) * s;
		return true;

	}

}

export class ForceLayout {

	constructor( options = {} ) {

		this.order = 0;
		const o = {
			clusterSeparation: 0.7,
			cohesion: 0.5,
			hubSpacing: 1,
			layoutSize: 1,
			damping: options.velocityDecay ?? 0.3,
			repulsion: 8,
			gravity: null, // null = automatic (0.03)
			centering: 0.3,
			theta: 0.8,
			tickBudget: 200,
			charge: - 30, // legacy: scales the repulsion
			linkDistance: null, // legacy: null = graph.linkDistance; scales the layout
			linkStrength: 1,
			distanceMin: 1, // legacy: ignored
			maxIterations: 8192,
			ticksPerFrame: 0,
			leafSize: DEFAULT_LEAF_SIZE,
			fanout: DEFAULT_FANOUT,
			alphaMin: MODEL.alphaMin,
			alphaDecay: null,
			autoScale: true,
			autoReheat: true,
			springOmega: options.sphereOmega ?? 8, // rad/s of the target spring
			blendDuration: 1, // s
			stateTau: 0.06, // s, nodeState easing time constant (~95% after 180 ms)
			colorTau: 0.12, // s, nodeColor (color + radius factor) easing time constant
			resortTicks: 20, // ticks between virtual-cluster re-partition cycles (× (1 − alpha))
			resortBudgetMs: 1.5, // main-thread budget per frame for re-partitioning
			mode: 'force',
			autoStart: true,
			data: null,
			...options,
		};
		this.options = o;
		this.data = o.data || null;

		this.U = {
			alpha: uniform( 1 ),
			alphaDrag: uniform( 0 ),
			betaScaled: uniform( 8 / ( MASS_ONE * MASS_ONE ) ),
			invUnit2: uniform( 1 / 900 ),
			jitterLength: uniform( MODEL.jitter * 30 ),
			theta2: uniform( o.theta * o.theta ),
			maxIterations: uniform( o.maxIterations >>> 0, 'uint' ),
			softIterations: uniform( o.maxIterations >>> 1, 'uint' ),
			lcaWeight: uniformArray( new Array( LEVEL_MAX + 1 ).fill( 1 ), 'float' ),
			kappa: uniform( 0 ),
			gravity: uniform( DEFAULT_GRAVITY ),
			rho: uniform( MODEL.rho ),
			heatActive: uniform( 0 ),
			heatIndex: uniform( - 1, 'int' ),
			heatDecay: uniform( MODEL.heatDecay ),
			heatSpread: uniform( MODEL.heatSpread ),
			hasForces: uniform( 1 ),
			centering: uniform( o.centering ),
			velocityDecay: uniform( o.damping ),
			velRescale: uniform( 1 ),
			maxDisp: uniform( 60 ),
			scaleF: uniform( 1 ),
			zKick: uniform( 0 ),
			flat: uniform( 0 ),
			flatActive: uniform( 0 ),
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
			colorEase: uniform( 0.1 ),
		};

		this._lambda = o.clusterSeparation;
		this._kappa = o.cohesion;
		this._hubSpacing = clampHub( o.hubSpacing );
		this._layoutSize = o.layoutSize;
		this._repulsion = o.repulsion;
		this._charge = o.charge;
		this._linkDistance = o.linkDistance;
		this._linkStrength = o.linkStrength;
		this._gravity = o.gravity;
		this._theta = o.theta;
		this._distanceMin = o.distanceMin;
		this._maxIterations = o.maxIterations >>> 0;
		this._ticksPerFrame = o.ticksPerFrame | 0;
		this._leafSize = o.leafSize;

		this.schedule = new AlphaSchedule( { budget: o.tickBudget, alphaMin: o.alphaMin, alphaDecay: o.alphaDecay } );
		this.scale = new ScaleController( 30 );
		this._paused = ! o.autoStart;

		this._mode = LAYOUT_MODES.some( ( m ) => m.id === o.mode ) ? o.mode : 'force';
		this._modeOptions = {};
		this._tb = new Ease( 0 ); // -> uniforms.targetBlend
		this._sph = new Ease( 0 ); // -> uniforms.layoutBlend
		this._flat = new Ease( 0 ); // -> U.flat
		this._targetsPartial = false; // some node keeps its forces at full blend
		this._targetKind = 'sphere'; // what nodeTarget currently holds
		this._pending = null; // outstanding stress request
		this._stressCache = null;
		this._radialRoot = - 1;
		this._radialDim = 3;
		this._radialFollow = false;
		this._holdHot = false;
		this._zKick = false;
		this._zSnapshot = false;
		this._forcesPrev = true;

		this._dragIndex = - 1;
		this._dragHasTarget = false;
		this._dragDirty = false;
		this._dragMoved = false;
		this._alphaDrag = 0;
		this._heatTicks = 0;
		this._pinned = new Set();
		this._commands = [];
		this._unpinAll = false;
		this._moveIndex = - 1;

		this._dtSmooth = 1 / 60;
		this._K = 1;
		this._kChangeTime = - Infinity;
		this._awakeUntil = 0;
		this._targetSig = '';
		this._kDyn = 8;
		this._slowTime = 0;
		this._fastTime = 0;
		this._hPrev = 1 / 60;
		this._time = 0;
		this._resortQueue = [];
		this._resortTick = 0;
		this._resortVersion = - 1;
		this._resortPos = null;
		this._scaleVersion = - 1;
		this._scaleSkip = 0;
		this._stampLog = 0;
		this._own = { busy: false, pos: null, version: 0, stamp: 0, frame: 0 };
		this._settledPrev = null;

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
		this._alphaDrag = 0;
		this._heatTicks = 0;
		this.U.dragIndex.value = - 1;
		this.U.moveIndex.value = - 1;
		this._K = 1;
		this._kChangeTime = - Infinity;
		this._kDyn = 8;
		this._targetSig = '';
		this._targetKind = 'sphere'; // GraphBuffers fills nodeTarget with the sphere layout
		this._targetsPartial = false;
		this._pending = null;
		this._stressCache = null;
		this._radialRoot = - 1;
		this._own = { busy: false, pos: null, version: 0, stamp: 0, frame: 0 };
		this._resortVersion = app.positionsVersion;
		this._applyParams();
		this._resetScale( graph.initPos, app );
		this.schedule.reheat( 1 );
		this._wake();
		this._enterMode( this._mode === 'custom' ? 'force' : this._mode, this._modeOptions, true );
		this._settledPrev = null;

	}

	_buildStructure() {

		const graph = this.graph, buffers = this.buffers;
		const t0 = performance.now();
		const struct = buildStructure( graph, { leafSize: this._leafSize, fanout: this.options.fanout, massPower: this._hubSpacing } );
		const pos = this.app?.positions?.length >= 4 * graph.N ? this.app.positions : graph.initPos;
		resortStructure( struct, pos );
		this.struct = struct;
		this.priv = createPrivateBuffers( struct, graph );
		this._uploadNodeMasses();
		const { N, C, TT } = struct;
		const ctx = { buffers, priv: this.priv, U: this.U, N, TT, C, D: graph.D, root: graph.root };
		const aggregate = [];
		for ( const r of struct.vHeightRanges ) aggregate.push( aggregateVirtualKernel( { ...ctx, start: r.start, count: r.count, height: r.height } ) );
		for ( const r of graph.heightRanges ) aggregate.push( aggregateRealKernel( { ...ctx, start: r.start, count: r.count, nodeChildren: r.height === 1 } ) );
		this.kernels = {
			aggregate,
			flatten: flattenKernel( ctx ), // tree-ordered copy for the next repulsion pass
			repulsion: repulsionKernel( ctx ),
			links: linksKernel( ctx ),
			velocity: velocityKernel( ctx ),
			position: positionKernel( ctx ),
			command: commandKernel( ctx ),
			zSnapshot: zSnapshotKernel( ctx ),
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

	// nodeInfo.w = fixed-point mass of every node (the aggregation kernels weight with it)
	_uploadNodeMasses() {

		const attr = this.buffers.nodeInfo.value, mass = this.struct.nodeMass;
		let changed = false;
		for ( let i = 0; i < this.struct.N; i ++ ) {

			if ( attr.array[ 4 * i + 3 ] !== mass[ i ] ) {

				attr.array[ 4 * i + 3 ] = mass[ i ];
				changed = true;

			}

		}

		if ( changed ) attr.needsUpdate = true;

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
		const { visits } = repulsionCPU( struct, pos4, bh, { theta: this._theta, maxIterations: this._maxIterations, slots } );
		const v = slots.map( ( t ) => visits[ struct.nodeOrder[ t ] ] ).sort( ( a, b ) => a - b );
		return { mean: v.reduce( ( a, b ) => a + b, 0 ) / Math.max( 1, S ), p95: v[ Math.floor( 0.95 * ( S - 1 ) ) ] ?? 0, max: v[ S - 1 ] ?? 0 };

	}

	// ─── derived quantities ───────────────────────────────────────────────────────────────

	/** Size of the layout relative to graph.layoutRadius (legacy linkDistance × layoutSize). */
	_sizeFactor() {

		const g = this.graph;
		const base = g?.linkDistance ?? 30;
		return ( ( this._linkDistance ?? base ) / base ) * this._layoutSize;

	}

	/** Radius the 3D layout fills (world units). */
	_radius() {

		return ( this.graph?.layoutRadius ?? 100 ) * this._sizeFactor();

	}

	/** rms radius the auto-scale steers to (ball of radius R, or the flat disc). */
	_targetRms() {

		const R = this._radius(), F = this._flat.value;
		return ( 1 - F ) * Math.sqrt( 0.6 ) * R + F * Math.SQRT1_2 * flatRadius( R, this.graph.N );

	}

	/** Flat-mode adaptation of β, λ and κ (forceKernels.js flatTuning), eased with the flat blend. */
	_flatTuning() {

		return flatTuning( this.graph?.N ?? 0, this._flat.value );

	}

	/** The β the kernels use: user value × legacy charge × flat-mode factor. */
	get _beta() {

		return Math.max( 0.25, ( this._repulsion * Math.abs( this._charge ) ) / 30 ) * this._flatTuning().beta;

	}

	_applyParams() {

		const g = this.graph, U = this.U;
		const ft = this._flatTuning();
		U.betaScaled.value = this._beta / ( MASS_ONE * MASS_ONE );
		U.gravity.value = this._gravity ?? DEFAULT_GRAVITY;
		for ( let l = 0; l <= LEVEL_MAX; l ++ ) U.lcaWeight.array[ l ] = this._linkStrength * Math.pow( this._lambda, l * ft.lambdaPower );
		if ( ! g ) return;
		U.kappa.value = g.D > 0 ? this._kappa * ft.kappa : 0;
		const k = this._sizeFactor();
		U.maxDisp.value = Math.max( 2 * ( g.linkDistance ?? 30 ), 0.05 * g.layoutRadius ) * k;

	}

	// the kernel unit restarts at the link distance; the first measurement comes from `pos4`
	_resetScale( pos4, app, sameBuffers = false ) {

		const g = this.graph;
		this.scale.reset( ( g.linkDistance ?? 30 ) * this._sizeFactor() );
		this.scale.measure( rmsRadius( pos4, g.N, true ), this._targetRms(), 0 );
		this._stampLog = 0;
		this._scaleVersion = app ? app.positionsVersion : - 1;
		// a readback of the positions we just replaced may be in flight: ignore it
		this._scaleSkip = sameBuffers && app && app._readbackBusy === true ? 1 : 0;
		this._own.stamp = 0;
		this._own.pos = null;

	}

	// ─── frame ────────────────────────────────────────────────────────────────────────────

	update( dt, app ) {

		if ( ! this.buffers || ! this.kernels ) return;
		const renderer = app.renderer;
		const U = this.U, uniforms = this.buffers.uniforms, k = this.kernels;
		this._time += dt;
		this._dtSmooth += ( dt - this._dtSmooth ) * 0.1;

		this._followTargets();
		this._stepBlends( dt );
		this._measureScale( app );
		U.stateEase.value = 1 - Math.exp( - dt / Math.max( 1e-3, this.options.stateTau ) );
		U.colorEase.value = 1 - Math.exp( - dt / Math.max( 1e-3, this.options.colorTau ) );

		const tb = uniforms.targetBlend.value;
		const forces = this._forcesOn();
		if ( forces && ! this._forcesPrev ) this._needsAggregate = true; // target-only ticks skip the flatten pass
		this._forcesPrev = forces;
		const ticking = this._isTicking();
		const dragging = this._dragIndex >= 0;

		// re-partition virtual clusters first: a changed nodeOrder needs re-aggregation (and a
		// fresh tree-ordered copy) before the next repulsion pass
		if ( ticking && forces ) this._maybeResort( app );

		// commands (pins, drops; explicit drag moves while not ticking)
		const pre = [];
		if ( this._zSnapshot ) {

			pre.push( k.zSnapshot );
			this._zSnapshot = false;

		}

		if ( ! ticking && dragging && this._dragHasTarget && this._dragDirty ) this._moveIndex = this._dragIndex;
		this._dragDirty = false;
		if ( this._commands.length || this._unpinAll || this._moveIndex >= 0 ) {

			const n = Math.min( MAX_COMMANDS, this._commands.length );
			for ( let c = 0; c < n; c ++ ) {

				const [ index, value, heat ] = this._commands[ c ];
				U.cmds.array[ c ].set( index, value, heat, 0 );

			}

			this._commands.splice( 0, n );
			U.cmdCount.value = n;
			U.unpinAll.value = this._unpinAll ? 1 : 0;
			U.moveIndex.value = this._moveIndex;
			if ( this._moveIndex >= 0 ) this._needsAggregate = true; // a node moved
			this._unpinAll = false;
			this._moveIndex = - 1;
			pre.push( k.command );

		}

		if ( this._needsAggregate ) {

			pre.push( ...k.aggregate, k.flatten );
			this._needsAggregate = false;

		}

		let K = 0;
		if ( ticking ) {

			K = this._ticksForFrame( dt, forces );
			// target spring: per-tick decay of a critically damped system, h = seconds per tick
			const h = this._dtSmooth / K;
			const lam = Math.exp( - this.options.springOmega * h );
			U.lam2.value = lam * lam;
			U.springK.value = ( 1 - lam ) * ( 1 - lam );
			// keep the spring's velocity (per tick) continuous when the tick length changes
			const ratio = Math.min( 4, Math.max( 0.25, h / this._hPrev ) );
			this._hPrev = h;
			U.dragIndex.value = dragging ? this._dragIndex : - 1;
			U.dragHasTarget.value = this._dragHasTarget ? 1 : 0;
			U.hasForces.value = forces ? 1 : 0;
			const moved = this._dragMoved;
			const tickList = forces
				? [ k.repulsion, k.links, k.velocity, k.position, ...k.aggregate, k.flatten ]
				: [ k.velocity, k.position, ...k.aggregate ];
			for ( let t = 0; t < K; t ++ ) {

				if ( forces ) this._stepForceUniforms( dragging, moved );
				else {

					// nothing to cool in a pure target layout: `alpha` reads as at rest
					this.schedule.alpha = 0;
					this.schedule.hold = 0;
					U.scaleF.value = 1;
					U.zKick.value = 0;

				}

				U.velRescale.value = t === 0 ? 1 + ( ratio - 1 ) * tb : 1;
				renderer.compute( t === 0 && pre.length ? [ ...pre, ...tickList ] : tickList );
				if ( t === 0 && pre.length ) this._clearCommandUniforms(); // commands are one-shot

			}

			this.ticks += K;
			// soft targets: the spring keeps converging for a while after the forces have cooled
			if ( forces && tb > 0 && ! this.schedule.cooled ) this._wake();

		} else if ( pre.length ) {

			renderer.compute( pre );
			this._clearCommandUniforms();

		}

		this._dragMoved = false;
		if ( K > 0 || pre.length ) this.buffers.moveFrame = app.frame; // positions changed this frame
		else if ( this.buffers.moveFrame === Infinity ) this.buffers.moveFrame = app.frame;
		this._lastFrameTicks = K;
		renderer.compute( k.state );

		// scale bookkeeping for the readback the App requests after this frame's render
		if ( app.options?.readback === false ) this._ownReadback( app, ticking && forces );
		else if ( app._readbackBusy !== true ) this._stampLog = this.scale.logScale;

		const settled = this.settled;
		if ( settled !== this._settledPrev ) {

			this._settledPrev = settled;
			this._emit();

		}

	}

	// per-tick uniforms of a force tick: alpha schedule, auto-scale, drag heat
	_stepForceUniforms( dragging, moved ) {

		const U = this.U;
		let alpha = this.schedule.step();

		// pinned nodes are not scaled (positionKernel) and not measured (rmsRadius, free nodes only)
		const scaling = this.options.autoScale && alpha > MODEL.scaleMinAlpha && ! dragging
			&& this._pinned.size <= MODEL.scaleMaxPinned * this.graph.N && this._tb.value === 0 && this._tb.to === 0;
		// the extent is still off: the schedule waits (at alpha 0.02) instead of cooling to rest
		if ( scaling && this.scale.off ) alpha = this.schedule.wait();
		U.alpha.value = alpha;
		const f = this.scale.step( scaling );
		const unit = this.scale.unit / f; // the passes of this tick see the positions before scaling
		U.scaleF.value = f;
		U.invUnit2.value = 1 / ( unit * unit );
		U.jitterLength.value = MODEL.jitter * unit;
		U.zKick.value = this._zKick ? MODEL.jitter * unit : 0; // one tick only
		this._zKick = false;

		if ( dragging ) this._alphaDrag = moved ? 1 : this._alphaDrag * MODEL.idleDecay;
		else if ( this._heatTicks > 0 ) this._heatTicks --;
		const heatOn = dragging || this._heatTicks > 0;
		U.heatActive.value = heatOn ? 1 : 0;
		U.heatIndex.value = dragging ? this._dragIndex : - 1;
		U.alphaDrag.value = heatOn ? this._alphaDrag : 0;

	}

	// advances the three blends and mirrors them into the uniforms
	_stepBlends( dt ) {

		const uniforms = this.buffers.uniforms, U = this.U;
		const step = dt / Math.max( 1e-3, this.options.blendDuration );
		let changed = false;
		if ( this._tb.step( step ) ) changed = true;
		if ( this._sph.step( step ) ) changed = true;
		if ( this._flat.step( step ) ) {

			changed = true;
			this._applyParams(); // β, λ, κ follow the flat blend (flatTuning)

		}

		uniforms.targetBlend.value = this._tb.value;
		uniforms.layoutBlend.value = this._sph.value;
		U.flat.value = this._flat.value;
		U.flatActive.value = this._flat.value > 0 || this._flat.to > 0 ? 1 : 0;
		if ( changed ) this._wake();
		// the plateau of a (re)started force layout begins when its blend has finished
		if ( this._holdHot ) {

			if ( this._tb.active || this._flat.active ) this.schedule.reheat( 1 );
			else {

				this._holdHot = false;
				this._zKick = this._mode === 'force'; // a planar start would stay planar in 3D

			}

		}

	}

	// keeps the targets of the current mode up to date (sphere radius, selection)
	_followTargets() {

		const uniforms = this.buffers.uniforms;
		if ( this._mode === 'sphere' && this._targetKind === 'sphere' ) {

			if ( uniforms.sphereRadius.value !== this._sphereRadius ) this._writeSphereTargets();
			// the selected node and its neighbours move inward: wake the sleeping spring
			const sig = `${uniforms.selectedIndex.value}|${this.buffers.nodeStateTarget.value.version}`;
			if ( sig !== this._targetSig ) {

				this._targetSig = sig;
				this._wake();

			}

		} else if ( this._mode === 'radial' && this._radialFollow ) {

			const sel = uniforms.selectedIndex.value;
			if ( sel >= 0 && sel < this.graph.N && sel !== this._radialRoot ) this._writeRadialTargets( sel );

		}

	}

	// auto-scale: a new measurement whenever a position readback has arrived
	_measureScale( app ) {

		const N = this.graph.N;
		let pos, stamp;
		if ( app.options?.readback === false ) {

			if ( this._own.version === this._scaleVersion || ! this._own.pos ) return;
			this._scaleVersion = this._own.version;
			pos = this._own.pos;
			stamp = this._own.stamp;

		} else {

			if ( app.positionsVersion === this._scaleVersion || ! app.positions ) return;
			this._scaleVersion = app.positionsVersion;
			if ( this._scaleSkip > 0 ) {

				this._scaleSkip --;
				return;

			}

			pos = app.positions;
			stamp = this._stampLog;

		}

		if ( pos.length < 4 * N || N === 0 ) return;
		this.scale.measure( rmsRadius( pos, N, true ), this._targetRms(), stamp ); // free nodes (w = pin flag)

	}

	// without the App's shared readback the layout fetches positions itself (every 4th frame)
	_ownReadback( app, wanted ) {

		const own = this._own;
		if ( ! wanted || own.busy || app.frame - own.frame < 4 ) return;
		own.busy = true;
		own.frame = app.frame;
		const buffers = this.buffers, stamp = this.scale.logScale;
		app.renderer.getArrayBufferAsync( buffers.nodePos.value ).then( ( ab ) => {

			if ( buffers !== this.buffers || own !== this._own ) return;
			own.pos = new Float32Array( ab.slice( 0 ) );
			own.stamp = stamp;
			own.version ++;

		} ).catch( () => {} ).finally( () => {

			own.busy = false;

		} );

	}

	/** Whether forces are computed (false in a pure target layout). */
	_forcesOn() {

		return this._tb.value < 1 || this._tb.to < 1 || this._targetsPartial;

	}

	/** Whether the layout still has work to do in its current mode (ignores pause). */
	_wantsTicks() {

		if ( this._tb.active || this._sph.active || this._flat.active ) return true;
		const dragging = this._dragIndex >= 0;
		if ( this._forcesOn() ) {

			if ( ! this.schedule.cooled ) return true;
			if ( dragging ) return this._alphaDrag > 1e-3; // a held node lets its neighbourhood come to rest
			if ( this._heatTicks > 0 ) return true;
			return this._tb.value > 0 && this._time < this._awakeUntil;

		}

		return dragging || this._time < this._awakeUntil;

	}

	/** Whether this frame advances the simulation (see `running`). */
	_isTicking() {

		return ! this._paused && this._wantsTicks();

	}

	/** Keeps a target layout stepping for a while (its spring converges within ~2 s). */
	_wake( seconds = 2.5 ) {

		this._awakeUntil = Math.max( this._awakeUntil, this._time + seconds );

	}

	_clearCommandUniforms() {

		const U = this.U;
		U.cmdCount.value = 0;
		U.unpinAll.value = 0;
		U.moveIndex.value = - 1;

	}

	_ticksForFrame( dt, forces ) {

		if ( this._ticksPerFrame > 0 ) return this._ticksPerFrame;
		if ( ! forces ) return 1;
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
	 * `resortBudgetMs` of main-thread time, and only the touched ranges of nodeOrder, the tree
	 * (masses) and vInfo are uploaded. Stale groups never break correctness (radii and centres are
	 * recomputed every tick with the masses that match the uploaded order), they only make the
	 * walk longer.
	 */
	_maybeResort( app ) {

		const struct = this.struct;
		if ( ! struct.groups.length ) return;
		const alpha = this.schedule.alpha;
		if ( ! this._resortQueue.length ) {

			const cadence = Math.max( 1, Math.round( this.options.resortTicks * ( 1 - alpha ) ) );
			if ( this.ticks - this._resortTick < cadence ) return;
			if ( alpha < 0.005 && this._dragIndex < 0 && this._heatTicks === 0 ) return;
			const own = app.options?.readback === false;
			const pos = own ? this._own.pos : app.positions;
			const version = own ? this._own.version : app.positionsVersion;
			if ( ! pos || version === this._resortVersion ) return;
			this._resortVersion = version;
			this._resortPos = pos; // app.positions is updated in place by later readbacks: fine
			this._resortQueue = resortTasks( struct );
			this._resortTick = this.ticks;

		}

		const pos = this._resortPos;
		if ( ! pos || pos.length < 4 * struct.N ) {

			this._resortQueue = [];
			return;

		}

		const t0 = performance.now();
		const r = { lo: Infinity, hi: - Infinity, t0: Infinity, t1: - Infinity, vLo: Infinity, vHi: - Infinity };
		while ( this._resortQueue.length ) {

			const c = runResortTask( struct, pos, this._resortQueue.shift(), this._resortQueue );
			if ( c.changed ) {

				r.lo = Math.min( r.lo, c.lo ); r.hi = Math.max( r.hi, c.hi );
				r.t0 = Math.min( r.t0, c.t0 ); r.t1 = Math.max( r.t1, c.t1 );
				r.vLo = Math.min( r.vLo, c.vLo ); r.vHi = Math.max( r.vHi, c.vHi );

			}

			// twice the budget while hot: nodes move fastest then and stale groups cost the most
			if ( performance.now() - t0 > this.options.resortBudgetMs * ( alpha > 0.25 ? 2 : 1 ) ) break;

		}

		if ( r.hi > r.lo ) this._uploadStructure( r );

	}

	/**
	 * Uploads the slot range [lo, hi) of nodeOrder, the tree range [t0, t1) (masses) and the
	 * virtual clusters [vLo, vHi) (masses) after a resort; without a range, everything.
	 */
	_uploadStructure( r = null ) {

		const struct = this.struct, priv = this.priv;
		const range = ( attr, start, count, full ) => {

			if ( full ) attr.clearUpdateRanges();
			else attr.addUpdateRange( start, count );
			attr.needsUpdate = true;

		};

		const order = priv.nodeOrder.value;
		const lo = r ? r.lo : 0, hi = r ? r.hi : struct.N;
		order.array.set( struct.nodeOrder.subarray( lo, hi ), lo );
		range( order, lo, hi - lo, ! r || ( lo <= 0 && hi >= struct.N ) );
		// tree and vInfo share their arrays with the structure
		const t0 = r ? r.t0 : 0, t1 = r ? r.t1 : struct.TT;
		if ( t1 > t0 ) range( priv.tree.value, 4 * t0, 4 * ( t1 - t0 ), ! r || ( t0 <= 0 && t1 >= struct.TT ) );
		const vLo = r ? r.vLo : 0, vHi = r ? r.vHi : struct.V;
		if ( vHi > vLo ) range( priv.vInfo.value, 4 * vLo, 4 * ( vHi - vLo ), ! r || ( vLo <= 0 && vHi >= struct.V ) );
		if ( ! r ) {

			priv.rInfo.value.clearUpdateRanges();
			priv.rInfo.value.needsUpdate = true;

		}

		this._needsAggregate = true;

	}

	// ─── simulation control ───────────────────────────────────────────────────────────────

	get running() {

		return !! this.buffers && this._isTicking();

	}

	get paused() {

		return this._paused;

	}

	/** At rest in the current mode: cooled, nothing dragged, no blend or request in progress. */
	get settled() {

		return !! this.buffers && ! this._pending && ! this._wantsTicks();

	}

	get alpha() {

		return this.schedule.alpha;

	}

	set alpha( v ) {

		this.schedule.alpha = clamp01( v );
		this.schedule.hold = 0;

	}

	get alphaTarget() {

		return this.schedule.target;

	}

	set alphaTarget( v ) {

		this.schedule.target = clamp01( v );

	}

	get alphaMin() {

		return this.schedule.alphaMin;

	}

	set alphaMin( v ) {

		this.schedule.alphaMin = Math.max( 1e-6, v );
		this.schedule.budget = this.schedule.budget;

	}

	get alphaDecay() {

		return this.schedule.decay;

	}

	set alphaDecay( v ) {

		this.schedule._decayOverride = v;
		this.schedule.budget = this.schedule.budget;

	}

	/** Ticks of a full run from alpha 1 (30% plateau, 70% cooling). */
	get tickBudget() {

		return this.schedule.budget;

	}

	set tickBudget( v ) {

		this.schedule.budget = v;

	}

	reheat( alpha = 1 ) {

		this.schedule.reheat( alpha );
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

	// ─── layout modes ─────────────────────────────────────────────────────────────────────

	get layoutMode() {

		return this._mode;

	}

	/** True while the result of an asynchronous layout ('stress') is outstanding. */
	get pending() {

		return !! this._pending;

	}

	/** Root of the radial layout (−1 when it has not been computed). */
	get radialRoot() {

		return this._radialRoot;

	}

	setLayoutMode( mode, options = {} ) {

		const m = LAYOUT_MODES.some( ( x ) => x.id === mode ) ? mode : 'force';
		options = options || {};
		const same = m === this._mode && ! this._pending;
		if ( same && ( FORCE_MODES.has( m ) || m === 'sphere' ) ) return this;
		if ( same && m === 'stress' && JSON.stringify( options ) === JSON.stringify( this._modeOptions ) ) return this;
		this._modeOptions = { ...options };
		if ( this.buffers ) this._enterMode( m, this._modeOptions, false );
		else this._mode = m;
		this.resume();
		return this;

	}

	/**
	 * Pulls nodes to absolute target positions (mode 'custom'). positions4: Float32Array( 4N ),
	 * xyz in world units. blend 0..1 is the strength of the pull (uniforms.targetBlend), weights
	 * an optional per-node factor 0..1. With blend 1 and no weights below 1 forces are skipped.
	 */
	setTargets( positions4, { blend = 1, weights = null } = {} ) {

		if ( ! positions4 ) return this.setLayoutMode( 'force' );
		if ( ! this.buffers ) return this;
		if ( positions4.length < 4 * this.graph.N ) throw new Error( 'setTargets: positions need 4 floats per node' );
		const prev = this._mode;
		this._pending = null;
		this._mode = 'custom';
		this._writeTargets( positions4, weights, 'custom' );
		blend = clamp01( blend );
		this._setGoals( blend, 0, 0 );
		// soft targets work through the running simulation
		if ( ( blend < 1 || this._targetsPartial ) && ( prev !== 'custom' || this.schedule.cooled ) ) this.reheat( 1 );
		this.resume();
		this._emit();
		return this;

	}

	// switches to `mode`; `initial` (new graph) snaps the blends instead of animating them
	_enterMode( mode, options, initial ) {

		const prev = this._mode;
		const prevPending = this._pending;
		this._pending = null;
		this._mode = mode;
		this._radialFollow = false;
		if ( mode === 'sphere' ) {

			this._writeSphereTargets();
			this._setGoals( 1, 1, 0 );

		} else if ( mode === 'radial' ) {

			const sel = this.buffers.uniforms.selectedIndex.value;
			const N = this.graph.N;
			let root = Number.isInteger( options.root ) && options.root >= 0 && options.root < N ? options.root : - 1;
			this._radialFollow = options.follow ?? root < 0;
			if ( root < 0 ) root = sel >= 0 && sel < N ? sel : this._hubNode();
			this._writeRadialTargets( root );
			this._setGoals( 1, 0, 0 );

		} else if ( mode === 'stress' ) {

			const key = JSON.stringify( options );
			const cached = this._stressCache;
			if ( cached && cached.graph === this.graph && cached.key === key ) {

				this._writeStressTargets();
				this._setGoals( 1, 0, 0 );

			} else {

				// keep moving as before until the positions arrive
				if ( initial ) this._setGoals( 0, 0, 0 );
				this._requestStress( options, key, prevPending ? prevPending.fallback : prev );

			}

		} else {

			const flat = mode === 'flat';
			// entering the plane: remember the z of every node (unless a flat blend is still in progress)
			if ( flat && this._flat.value === 0 && this._flat.to === 0 ) this._zSnapshot = true;
			this._setGoals( 0, 0, flat ? 1 : 0 );
			if ( ! initial ) {

				this.reheat( 1 );
				this._holdHot = true;

			}

		}

		if ( initial ) {

			this._tb.snap( this._tb.to );
			this._sph.snap( this._sph.to );
			this._flat.snap( this._flat.to );
			const uniforms = this.buffers.uniforms;
			uniforms.targetBlend.value = this._tb.value;
			uniforms.layoutBlend.value = this._sph.value;
			this.U.flat.value = this._flat.value;
			this.U.flatActive.value = this._flat.value > 0 ? 1 : 0;
			this._applyParams(); // flat-mode parameters
			if ( this._flat.value > 0 ) this._zSnapshot = true; // z of the seed, then flat from the first tick
			this._holdHot = false;
			this._forcesPrev = true;

		}

		this._wake();
		this._emit();

	}

	_setGoals( target, sphere, flat ) {

		this._tb.set( target );
		this._sph.set( sphere );
		this._flat.set( flat );

	}

	_emit() {

		const events = this.app?.events;
		if ( ! events || typeof CustomEvent === 'undefined' ) return;
		events.dispatchEvent( new CustomEvent( 'layout', { detail: { mode: this._mode, settled: this.settled, pending: !! this._pending } } ) );

	}

	// node of highest degree (default root of the radial layout)
	_hubNode() {

		const deg = this.graph.degree;
		let best = 0;
		for ( let i = 1; i < this.graph.N; i ++ ) if ( deg[ i ] > deg[ best ] ) best = i;
		return best;

	}

	/** Writes nodeTarget (xyz from positions4, w = weights or 1). */
	_writeTargets( positions4, weights, kind ) {

		const attr = this.buffers.nodeTarget.value, a = attr.array, N = this.graph.N;
		let partial = false;
		for ( let i = 0; i < N; i ++ ) {

			a[ 4 * i ] = positions4[ 4 * i ];
			a[ 4 * i + 1 ] = positions4[ 4 * i + 1 ];
			a[ 4 * i + 2 ] = positions4[ 4 * i + 2 ];
			const w = weights ? clamp01( weights[ i ] ) : 1;
			a[ 4 * i + 3 ] = w;
			if ( w < 1 ) partial = true;

		}

		attr.needsUpdate = true;
		this._targetsPartial = partial;
		this._targetKind = kind;
		this._wake();

	}

	_writeSphereTargets() {

		const g = this.graph, attr = this.buffers.nodeTarget.value, a = attr.array;
		const R = this._sphereRadius = this.buffers.uniforms.sphereRadius.value;
		for ( let i = 0; i < g.N; i ++ ) {

			a[ 4 * i ] = g.sphereDir[ 4 * i ] * R;
			a[ 4 * i + 1 ] = g.sphereDir[ 4 * i + 1 ] * R;
			a[ 4 * i + 2 ] = g.sphereDir[ 4 * i + 2 ] * R;
			a[ 4 * i + 3 ] = 1;

		}

		attr.needsUpdate = true;
		this._targetsPartial = false;
		this._targetKind = 'sphere';
		this._wake();

	}

	_writeRadialTargets( root ) {

		const dim = this._modeOptions.dim === 2 ? 2 : 3;
		const r = radialLayout( this.graph, root, { dim, radius: this._radius() } );
		this._radialRoot = root;
		this._radialDim = dim;
		this.radialInfo = r.info;
		this._writeTargets( r.positions, null, 'radial' );

	}

	// writes the cached stress layout, scaled from the radius it was computed for to the current one
	_writeStressTargets() {

		const c = this._stressCache;
		const k = c.radius > 0 ? this._radius() / c.radius : 1;
		let positions = c.positions;
		if ( k !== 1 ) {

			// the result is centred on the origin, so another radius is a pure scaling
			positions = new Float32Array( c.positions.length );
			for ( let i = 0; i < positions.length; i ++ ) positions[ i ] = c.positions[ i ] * k;

		}

		this._writeTargets( positions, null, 'stress' );

	}

	// the size of the layout changed (layoutSize, legacy linkDistance): radial and stress targets
	// follow (the force modes follow through the auto-scale); the sphere has its own radius
	// (uniforms.sphereRadius) and custom targets are the caller's
	_resizeTargets() {

		if ( ! this.buffers || this._tb.to <= 0 ) return; // targets not in use: rewritten when their mode is entered
		if ( this._targetKind === 'radial' && this._radialRoot >= 0 ) {

			const r = radialLayout( this.graph, this._radialRoot, { dim: this._radialDim, radius: this._radius() } );
			this.radialInfo = r.info;
			this._writeTargets( r.positions, null, 'radial' );

		} else if ( this._targetKind === 'stress' && this._stressCache && this._stressCache.graph === this.graph ) this._writeStressTargets();

	}

	// the DataClient the stress layout is requested from (a private one unless options.data was given)
	_dataClient() {

		if ( this.data ) return this.data;
		if ( ! this._ownData ) this._ownData = new DataClient();
		return this._ownData;

	}

	// asks the data worker for the stress layout; the mode takes effect when it arrives
	_requestStress( options, key, fallback ) {

		const graph = this.graph;
		const pending = this._pending = { key, fallback: fallback === 'stress' || fallback === 'custom' ? 'force' : fallback };
		const current = () => this._pending === pending && this.graph === graph;
		// computed for the radius of this moment; a size change while it is outstanding (or later,
		// from the cache) only rescales the result
		const radius = this._radius();
		Promise.resolve().then( () => this._dataClient().layout( 'stress', { graph, ...options, radius } ) ).then( ( r ) => {

			if ( ! current() ) return;
			this._pending = null;
			if ( r.cancelled ) return this._enterMode( pending.fallback, {}, false );
			this._stressCache = { graph, key, positions: r.positions, radius };
			this.stressInfo = { ...r.info, timings: r.timings };
			this._writeStressTargets();
			this._setGoals( 1, 0, 0 );
			this._emit();

		} ).catch( ( e ) => {

			console.warn( 'stress layout failed', e );
			if ( ! current() ) return;
			this._pending = null;
			this._enterMode( pending.fallback, {}, false );

		} );

	}

	// ─── presets ──────────────────────────────────────────────────────────────────────────

	setPreset( name ) {

		const p = LAYOUT_PRESETS[ name ];
		if ( ! p ) throw new Error( `setPreset: unknown preset "${name}" (${Object.keys( LAYOUT_PRESETS ).join( ', ' )})` );
		this._lambda = Math.max( 1e-3, Math.min( 1, p.clusterSeparation ) );
		this._kappa = Math.max( 0, p.cohesion );
		this._applyParams();
		this._changed();
		return this;

	}

	get preset() {

		for ( const [ name, p ] of Object.entries( LAYOUT_PRESETS ) ) if ( p.clusterSeparation === this._lambda && p.cohesion === this._kappa ) return name;
		return 'custom';

	}

	// a model parameter changed: let the layout adapt
	_changed( alpha = 0.3 ) {

		if ( this.options.autoReheat && this.buffers ) this.reheat( Math.max( this.schedule.alpha, alpha ) );

	}

	resetPositions( positions4 = this.graph?.initPos ) {

		if ( ! this.buffers ) return this;
		const N = this.graph.N;
		if ( ! positions4 ) positions4 = this.graph.initPos;
		if ( positions4.length < 4 * N ) throw new Error( 'resetPositions: positions need 4 floats per node' );
		// w is the pin flag on the GPU: positions from a readback (app.positions) carry the pins of
		// that moment, which this call clears
		let flagged = false;
		for ( let i = 0; i < N && ! flagged; i ++ ) flagged = positions4[ 4 * i + 3 ] !== 0;
		if ( flagged ) {

			positions4 = new Float32Array( positions4 ); // a copy: the caller's array stays as it is
			for ( let i = 0; i < N; i ++ ) positions4[ 4 * i + 3 ] = 0;

		}

		this.buffers.setPositions( positions4 );
		this._pinned.clear();
		this._commands.length = 0;
		this._unpinAll = false;
		this._heatTicks = 0;
		this._alphaDrag = 0;
		resortStructure( this.struct, positions4 );
		this._uploadStructure();
		this._resortQueue = [];
		this._resortTick = this.ticks;
		this._resetScale( positions4, this.app, true );
		if ( this._flat.value > 0 || this._flat.to > 0 ) this._zSnapshot = true;
		this._wake();
		this.reheat( 1 );
		return this;

	}

	// ─── drag & pin ───────────────────────────────────────────────────────────────────────

	startDrag( index ) {

		if ( ! this._valid( index ) ) return;
		this._dragIndex = index;
		this._dragHasTarget = false; // held in place (zero velocity) until the first drag()
		this._dragMoved = true;
		this._alphaDrag = 1; // the global alpha is left alone: only the neighbourhood heats up
		this._heatTicks = MODEL.heatTail;
		this.U.dragIndex.value = - 1;
		this._wake();

	}

	drag( index, worldPos ) {

		if ( ! this._valid( index ) ) return;
		if ( index !== this._dragIndex ) this.startDrag( index );
		const t = this.U.dragTarget.value;
		// in the plane of the flat layout the node cannot leave it
		const z = this._flat.value >= 1 && this._flat.to >= 1 && this._tb.value === 0 ? 0 : worldPos.z;
		if ( ! this._dragHasTarget || t.x !== worldPos.x || t.y !== worldPos.y || t.z !== z ) {

			this._dragMoved = true;
			this._alphaDrag = 1;

		}

		t.set( worldPos.x, worldPos.y, z );
		this._dragHasTarget = true;
		this._dragDirty = true;
		this._wake();

	}

	endDrag( index, { pin = false } = {} ) {

		if ( this._dragIndex >= 0 && index !== undefined && index !== null && index !== this._dragIndex ) return; // not the dragged node
		const i = this._dragIndex >= 0 ? this._dragIndex : index;
		const wasDragging = this._dragIndex >= 0;
		if ( wasDragging && this._dragHasTarget ) this._moveIndex = i; // final placement, even without a tick
		this._dragIndex = - 1;
		this._dragHasTarget = false;
		this.U.dragIndex.value = - 1;
		this._wake();
		if ( ! this._valid( i ) ) return;
		if ( pin ) this._pinned.add( i );
		if ( wasDragging ) {

			// the released node and its neighbourhood relax while the heat runs out; the heat is
			// seeded here because a drag while paused never ran a tick that could set it
			this._alphaDrag = 1;
			this._heatTicks = MODEL.heatTail;
			this._commands.push( [ i, this._pinned.has( i ) ? 1 : 0, 1 ] );

		} else if ( pin ) this._commands.push( [ i, 1, 0 ] );

	}

	setPinned( index, pinned ) {

		if ( ! this._valid( index ) ) return;
		let heat = 0;
		if ( pinned ) this._pinned.add( index );
		else if ( this._pinned.delete( index ) ) {

			// let it relax: local heat instead of a global reheat
			heat = 1;
			this._alphaDrag = 1;
			this._heatTicks = MODEL.heatTail;

		}

		this._commands.push( [ index, pinned ? 1 : 0, heat ] );
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
		if ( this._pinned.size ) this.schedule.reheat( Math.max( this.schedule.alpha, 0.1 ) );
		this._pinned.clear();
		this._wake();

	}

	_valid( index ) {

		return this.graph && Number.isInteger( index ) && index >= 0 && index < this.graph.N;

	}

	// ─── parameters ───────────────────────────────────────────────────────────────────────

	/** λ in (0, 1]: weight of a link per hierarchy level it crosses (1 = plain graph layout). */
	get clusterSeparation() {

		return this._lambda;

	}

	set clusterSeparation( v ) {

		this._lambda = Math.max( 1e-3, Math.min( 1, v ) );
		this._applyParams();
		this._changed();

	}

	/** κ ≥ 0: pull of every node to the centre of its finest cluster. */
	get cohesion() {

		return this._kappa;

	}

	set cohesion( v ) {

		this._kappa = Math.max( 0, v );
		this._applyParams();
		this._changed();

	}

	/** p in [0, 2]: mass exponent, m = (1 + deg/avgDeg)^p (free space around hubs). */
	get hubSpacing() {

		return this._hubSpacing;

	}

	set hubSpacing( v ) {

		v = clampHub( v );
		if ( v === this._hubSpacing ) return;
		this._hubSpacing = v;
		if ( this.struct ) {

			setMasses( this.struct, this.graph, v );
			this._uploadNodeMasses();
			this._uploadStructure();

		}

		// the masses set the natural size (the kernel unit moves up to 5× on a star between p = 0
		// and 2); the schedule waits for the auto-scale when a reheat to 0.3 is too short for that
		this._changed();

	}

	/** s > 0: radius of the layout in units of graph.layoutRadius. */
	get layoutSize() {

		return this._layoutSize;

	}

	set layoutSize( v ) {

		v = Math.max( 0.05, v );
		if ( ! Number.isFinite( v ) ) return;
		const changed = v !== this._layoutSize;
		this._layoutSize = v;
		this._applyParams();
		if ( changed ) this._resizeTargets();
		this._changed( 0.5 ); // ±1.5% per tick: a larger change needs more ticks

	}

	/** Velocity decay per tick, 0..1. */
	get damping() {

		return this.U.velocityDecay.value;

	}

	set damping( v ) {

		this.U.velocityDecay.value = clamp01( v );

	}

	/** β > 0: strength of the repulsion (range of the kernel relative to the layout). */
	get repulsion() {

		return this._repulsion;

	}

	set repulsion( v ) {

		this._repulsion = Math.max( 1e-3, v );
		this._applyParams();
		this._changed();

	}

	/** Legacy (d3 many-body strength): scales the repulsion, β_eff = repulsion·|charge|/30. */
	get charge() {

		return this._charge;

	}

	set charge( v ) {

		this._charge = v;
		this._applyParams();

	}

	/** Legacy: scales the layout (radius ∝ linkDistance / graph.linkDistance). null = graph value. */
	get linkDistance() {

		return this._linkDistance ?? this.graph?.linkDistance ?? 30;

	}

	set linkDistance( v ) {

		const before = this._sizeFactor();
		this._linkDistance = v === null || v === undefined ? null : Math.max( 1e-3, v );
		this._applyParams();
		if ( this._sizeFactor() !== before ) this._resizeTargets();

	}

	/** Multiplies all link weights. */
	get linkStrength() {

		return this._linkStrength;

	}

	set linkStrength( v ) {

		this._linkStrength = Math.max( 0, v );
		this._applyParams();

	}

	/** Gravity toward the origin (g of the model); null restores the automatic value. */
	get gravity() {

		return this.U.gravity.value;

	}

	set gravity( v ) {

		this._gravity = v === null || v === undefined ? null : Math.max( 0, v );
		this._applyParams();

	}

	/** The automatic gravity (what `gravity = null` uses). */
	get autoGravity() {

		return DEFAULT_GRAVITY;

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

	/** Alias of `damping` (d3 name). */
	get velocityDecay() {

		return this.damping;

	}

	set velocityDecay( v ) {

		this.damping = v;

	}

	/** Legacy: accepted and ignored (the repulsion kernel is bounded). */
	get distanceMin() {

		return this._distanceMin;

	}

	set distanceMin( v ) {

		this._distanceMin = v;

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
			if ( this._flat.value > 0 || this._flat.to > 0 ) this._zSnapshot = true; // the z memory was rebuilt

		}

	}

	get stats() {

		const s = this.struct;
		return {
			running: this.running,
			paused: this._paused,
			settled: this.settled,
			alpha: this.schedule.alpha,
			alphaTarget: this.schedule.target,
			mode: this._mode,
			pending: !! this._pending,
			blend: this._sph.value,
			targetBlend: this._tb.value,
			flat: this._flat.value,
			preset: this.preset,
			ticks: this.ticks,
			ticksPerFrame: this._lastFrameTicks,
			kStatic: this._kStatic,
			kDynamic: this._kDyn,
			dragging: this._dragIndex,
			alphaDrag: this._alphaDrag,
			heatTicks: this._heatTicks,
			pinned: this._pinned.size,
			gravity: this.U.gravity.value,
			beta: this._beta,
			unit: this.scale.unit,
			extent: this.scale.ext, // rms radius of the free nodes / target (the auto-scale's estimate)
			scale: this.scale.f,
			scaleWaited: this.schedule.waited, // ticks the schedule waited for the auto-scale since the last reheat
			radius: this.graph ? this._radius() : 0,
			bh: s ? {
				treeSize: s.TT, virtualClusters: s.V, virtualHeights: s.vHeightRanges.length, leafSize: s.leafSize,
				splitClusters: s.groups.length, visitsPerNode: this._visitsPerNode, buildMs: this._structMs,
			} : null,
		};

	}

	// ─── debug / measurement ──────────────────────────────────────────────────────────────

	/**
	 * Runs `ticks` full force ticks back to back (the simulation advances at the current alpha,
	 * without auto-scale) and returns wall-clock ms per tick after the GPU finished.
	 */
	async benchmark( ticks = 20, { repulsionOnly = false } = {} ) {

		const renderer = this.app.renderer;
		const device = renderer.backend.device;
		const k = this.kernels, U = this.U;
		const list = repulsionOnly ? [ k.repulsion ] : [ k.repulsion, k.links, k.velocity, k.position, ...k.aggregate, k.flatten ];
		U.scaleF.value = 1;
		U.zKick.value = 0;
		U.hasForces.value = 1;
		renderer.compute( [ ...k.aggregate, k.flatten, ...list ] ); // warm-up (pipeline compile)
		await device.queue.onSubmittedWorkDone();
		const t0 = performance.now();
		for ( let t = 0; t < ticks; t ++ ) renderer.compute( list );
		await device.queue.onSubmittedWorkDone();
		this._needsAggregate = true;
		return ( performance.now() - t0 ) / ticks;

	}

	/**
	 * Runs aggregation + repulsion (+ the links pass with `links`) once on the current positions
	 * and reads nodeForce back: (F_rep, S), or (F/M, heat) with `links`. Positions do not change.
	 */
	async readForces( { theta = this._theta, links = false } = {} ) {

		const renderer = this.app.renderer;
		const U = this.U, k = this.kernels;
		const prev = U.theta2.value;
		U.theta2.value = theta * theta;
		const unit = this.scale.unit;
		U.invUnit2.value = 1 / ( unit * unit );
		U.jitterLength.value = MODEL.jitter * unit;
		renderer.compute( [ ...k.aggregate, k.flatten, k.repulsion, ...( links ? [ k.links ] : [] ) ] );
		U.theta2.value = prev;
		const ab = await renderer.getArrayBufferAsync( this.buffers.nodeForce.value );
		return new Float32Array( ab.slice( 0 ) );

	}

	// ─── lifecycle ────────────────────────────────────────────────────────────────────────

	_disposeKernels() {

		if ( ! this.kernels ) return;
		const k = this.kernels;
		for ( const node of [ ...k.aggregate, k.flatten, k.repulsion, k.links, k.velocity, k.position, k.command, k.zSnapshot, k.state ] ) node.dispose?.();
		if ( this.priv ) for ( const b of Object.values( this.priv ) ) b.value?.dispose?.();
		this.kernels = null;
		this.priv = null;

	}

	dispose() {

		this._disposeKernels();
		this._pending = null;
		if ( this._ownData ) this._ownData.dispose();
		this._ownData = null;
		this.buffers = null;
		this.graph = null;

	}

}
