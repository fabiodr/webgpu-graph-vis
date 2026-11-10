// Application core: WebGPU renderer, scene, camera, render graph, frame loop, module system
// and the shared node-position readback used by picking, labels and camera fitting.
//
// Rendering is owned by `app.gfx` (src/render/gfx/RenderGraph.js): the scene pass, the quality
// tiers, the post-processing chain and the label overlay. `app.pipeline` is its RenderPipeline,
// `app.scenePass` the current scene PassNode (replaced whenever the graph is rebuilt, so read it
// when needed instead of keeping it), `app.setOutput( fn )` the legacy hook that replaces the
// chain with fn( scenePass, app ). The renderer itself has no MSAA (antialias: false): the scene
// pass has the samples of the active tier, see gfxState.samples.
//
// Options: maxPixelRatio (2), orbitControls (true), readback (true), idleFps ('auto'), idleAfter
// (2.5 s), gfx: { quality, overrides, governor, values, timestamps } for the RenderGraph. Without
// `gfx` the page URL decides (parseGfxParams in Quality.js: ?quality= ?governor= ?ao= ?dof=
// ?shadows= ?traa= ?scale= ?debug= ?gputime=), so every page that creates an App understands the
// same switches (also ?noreadback and ?idle=0 | 1 | <fps>).
//
// Frame order: pending resize → time uniform → snapshotPrev (when the scene pass has a velocity
// attachment) → modules.update → controls.update → gfx.update (governor, clip planes, focus,
// overlay camera) → updateMotionUniforms → pipeline.render → commitMotionUniforms →
// gfx.afterRender (GPU timing) → modules.afterRender → readback → 'frame'.
//
// Drawing buffer: the renderer pixel ratio is min( devicePixelRatio, maxPixelRatio ) and follows
// the display: resize() re-reads it, and a resolution media query calls for a resize when the
// window moves to another monitor or the page is zoomed (which does not always fire 'resize').
// Window resize events are applied once per animation frame (every resize reallocates the render
// targets of the chain); app.resize() itself is synchronous.
//
// Rest: the position readback stops a few frames after the node positions last changed
// (buffers.moveFrame) and starts again with them, so `positionsVersion` rests with the layout and
// everything keyed on it does too (app.positionsCurrent is true then; app.requestReadback() asks
// for one more). With `idleFps` the whole loop slows to that rate once nothing has happened for
// `idleAfter` seconds: no input, no app event, the camera and the layout still, no module that
// reports `animating`. 'auto' = 20 fps on touch devices (a phone that renders the full chain at
// the display rate for a still picture heats up until the system throttles it), off elsewhere;
// app.wake() ends it from outside. The ambient motion (dust, grain, pulses) keeps running at that
// rate; the governor does not measure idle frames.
//
// Compatibility mode (app.compat): a device without 'core-features-and-limits' puts three into
// WebGPU compatibility mode, where only the 'low' chain of the render graph is valid (no depth
// texture gather, no multisampled float targets, no per-attachment MRT blending). The render
// graph resolves every quality setting to 'low' there (gfxState.compat, gfx.config.compat).
//
// Modules are plain objects/classes with any of these optional members:
//   order                 number, update order (lower first). Suggested: layout 0, visuals 10,
//                         labels 20, interaction 30, ui 40.
//   init(app)             once, when added with app.use()
//   setGraph(buffers, graph, app)   whenever a new graph is loaded (GraphBuffers, PreparedGraph)
//   update(dt, app)       every frame before rendering (dt in seconds, clamped)
//   animating             optional boolean (getter): true keeps the loop at the display rate
//   afterRender(app)      every frame after rendering
//   resize(width, height, app)
//   dispose()
//
// Events (app.events is an EventTarget): 'graph' (detail: {buffers, graph}), 'frame',
// 'positions' (new readback available), 'gfx' (detail: gfxState, the pipeline was rebuilt or its
// tier / samples / velocity / shadows changed), plus anything modules dispatch ('select', …).

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GraphBuffers } from '../gpu/GraphBuffers.js';
import { gfxState, gfxUniforms, updateMotionUniforms, commitMotionUniforms, resetMotionUniforms } from '../render/gfx/shared.js';
import { RenderGraph } from '../render/gfx/RenderGraph.js';
import { parseGfxParams } from './Quality.js';

/** Frames after the node positions last changed for which the readback keeps running. */
const READBACK_REST = 8;

/** Window events that count as activity (see "Rest" in the header). */
const INPUT_EVENTS = [ 'pointerdown', 'pointermove', 'pointerup', 'wheel', 'keydown', 'touchstart', 'touchmove' ];

export async function checkWebGPU() {

	if ( typeof navigator === 'undefined' || ! navigator.gpu ) return { ok: false, reason: 'This browser does not expose WebGPU (navigator.gpu is missing).' };
	try {

		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
		if ( ! adapter ) return { ok: false, reason: 'No WebGPU adapter is available on this device.' };
		return { ok: true, adapter };

	} catch ( e ) {

		return { ok: false, reason: 'WebGPU adapter request failed: ' + e.message };

	}

}

/** Device limits we ask for: as many storage buffers per stage as the adapter allows (≤ 16). */
function pickLimits( adapter ) {

	const L = adapter.limits;
	const want = {};
	const set = ( k, v ) => {

		if ( k in L && v !== undefined ) want[ k ] = v;

	};

	set( 'maxStorageBuffersPerShaderStage', Math.min( L.maxStorageBuffersPerShaderStage, 16 ) );
	if ( 'maxStorageBuffersInVertexStage' in L ) set( 'maxStorageBuffersInVertexStage', Math.min( L.maxStorageBuffersInVertexStage, 16 ) );
	if ( 'maxStorageBuffersInFragmentStage' in L ) set( 'maxStorageBuffersInFragmentStage', Math.min( L.maxStorageBuffersInFragmentStage, 16 ) );
	set( 'maxStorageBufferBindingSize', L.maxStorageBufferBindingSize );
	set( 'maxBufferSize', L.maxBufferSize );
	set( 'maxComputeWorkgroupStorageSize', L.maxComputeWorkgroupStorageSize );
	set( 'maxComputeInvocationsPerWorkgroup', L.maxComputeInvocationsPerWorkgroup );
	set( 'maxComputeWorkgroupSizeX', L.maxComputeWorkgroupSizeX );
	return want;

}

export class App {

	constructor( container = document.body, options = {} ) {

		this.container = container;
		this.options = { maxPixelRatio: 2, orbitControls: true, readback: true, idleFps: 'auto', idleAfter: 2.5, ...options };
		if ( typeof location !== 'undefined' ) {

			const params = new URLSearchParams( location.search );
			if ( params.has( 'noreadback' ) ) this.options.readback = false;
			const idle = params.get( 'idle' );
			if ( idle !== null && idle !== '' ) this.options.idleFps = idle === '0' ? 0 : ( + idle > 1 ? + idle : 20 );

		}

		this.events = new EventTarget();
		this.modules = [];
		this.buffers = null;
		this.graph = null;
		this.time = 0;
		this.frame = 0;
		this.fps = 0;
		this.frameMs = 0;
		this.paused = false;

		// latest node positions read back from the GPU (Float32Array, 4 per node) + version counter
		this.positions = null;
		this.positionsVersion = 0;
		this._readbackBusy = false;
		this._readbackBuffer = null;
		this._readbackFrame = - 1; // frame in which the latest completed readback was requested
		this._readbackWanted = false; // requestReadback()

		// rest (see the header)
		this.compat = false;
		this.idle = false; // the loop runs at idleFps
		this._idleFps = 0;
		this._activeAt = 0; // performance.now() of the last activity
		this._resizePending = false;
		this._idleViewProj = new THREE.Matrix4();

	}

	/** The renderer pixel ratio for the display the window is on. */
	pixelRatio() {

		return Math.min( ( typeof window !== 'undefined' && window.devicePixelRatio ) || 1, this.options.maxPixelRatio );

	}

	/** Something happened (or is about to): the loop runs at the display rate for at least idleAfter seconds. */
	wake() {

		this._activeAt = performance.now();

	}

	/** True while `positions` holds the node positions as they are on the GPU (the layout rests, the readback has caught up). */
	get positionsCurrent() {

		return ! this._readbackDue();

	}

	/** Asks for one more position readback although the layout rests (after writing nodePos from outside). */
	requestReadback() {

		this._readbackWanted = true;

	}

	async init() {

		const gpu = await checkWebGPU();
		if ( ! gpu.ok ) {

			const err = new Error( gpu.reason );
			err.code = 'NO_WEBGPU';
			throw err;

		}

		this.adapterInfo = gpu.adapter.info || {};
		const gfxOptions = this.options.gfx ?? ( typeof location !== 'undefined' ? parseGfxParams( new URLSearchParams( location.search ) ) : {} );
		const renderer = new THREE.WebGPURenderer( {
			antialias: false, // the scene pass of the render graph owns the samples
			// GPU pass times for app.gfx.stats and the governor's headroom test (where the adapter has them)
			trackTimestamp: gfxOptions.timestamps !== false && gpu.adapter.features.has( 'timestamp-query' ),
			powerPreference: 'high-performance',
			requiredLimits: pickLimits( gpu.adapter ),
		} );
		await renderer.init();
		if ( ! renderer.backend.isWebGPUBackend ) {

			const err = new Error( 'WebGPU backend could not be initialized (fell back to WebGL).' );
			err.code = 'NO_WEBGPU';
			throw err;

		}

		this.renderer = renderer;
		// a device without 'core-features-and-limits': the render graph stays in its 'low' chain
		this.compat = renderer.backend.compatibilityMode === true;
		if ( this.compat ) console.info( 'WebGPU compatibility mode: rendering with the "low" quality chain only.' );
		renderer.setPixelRatio( this.pixelRatio() );
		renderer.toneMapping = THREE.NeutralToneMapping;
		renderer.toneMappingExposure = 1;
		renderer.domElement.classList.add( 'gl' );
		renderer.domElement.style.touchAction = 'none';
		this.container.appendChild( renderer.domElement );

		this.scene = new THREE.Scene();
		this.scene.background = new THREE.Color( 0x000000 );
		this.camera = new THREE.PerspectiveCamera( 45, 1, 0.5, 20000 );
		this.camera.position.set( 0, 0, 400 );

		if ( this.options.orbitControls ) {

			this.controls = new OrbitControls( this.camera, renderer.domElement );
			this.controls.enableDamping = true;
			this.controls.dampingFactor = 0.08;

		}

		// the render graph needs the final drawing-buffer size for its pixel budget
		this._measure();
		renderer.setSize( this.width, this.height );
		this.gfx = new RenderGraph( this, gfxOptions );
		this.pipeline = this.gfx.pipeline;

		// resize events are applied by the next frame, once (see the header)
		this._onResize = () => {

			this._resizePending = true;
			this.wake();

		};

		window.addEventListener( 'resize', this._onResize );
		this._watchPixelRatio();
		this.resize();

		// rest: input and app events wake the loop; a tab that comes back is measured afresh
		const idleFps = this.options.idleFps;
		const coarse = typeof matchMedia === 'function' && matchMedia( '(pointer:coarse)' ).matches;
		this._idleFps = idleFps === 'auto' ? ( coarse ? 20 : 0 ) : Math.max( 0, + idleFps || 0 );
		this._onInput = () => this.wake();
		for ( const type of INPUT_EVENTS ) window.addEventListener( type, this._onInput, { passive: true, capture: true } );
		const dispatch = this.events.dispatchEvent.bind( this.events );
		this.events.dispatchEvent = ( event ) => {

			if ( event.type !== 'frame' && event.type !== 'positions' ) this.wake();
			return dispatch( event );

		};

		this._onVisibility = () => {

			if ( document.visibilityState !== 'visible' ) return;
			this._last = performance.now();
			this.wake();
			this.gfx.governor.settle();

		};

		document.addEventListener( 'visibilitychange', this._onVisibility );

		this._last = performance.now();
		this.wake();
		this._fpsAcc = 0; this._fpsFrames = 0;
		renderer.setAnimationLoop( () => this._tick() );
		return this;

	}

	/** The scene PassNode of the render graph. Replaced on every rebuild: do not keep it. */
	get scenePass() {

		return this.gfx ? this.gfx.scenePass : null;

	}

	/**
	 * Legacy hook: replace the post-processing chain with fn( scenePass, app ) => outputNode (the
	 * pipeline then tone maps and encodes as it did before the render graph). null restores the chain.
	 */
	setOutput( fn ) {

		this.gfx.setOutput( fn );

	}

	use( module ) {

		this.modules.push( module );
		this.modules.sort( ( a, b ) => ( a.order ?? 50 ) - ( b.order ?? 50 ) );
		module.init?.( this );
		if ( this.buffers ) module.setGraph?.( this.buffers, this.graph, this );
		module.resize?.( this.width, this.height, this );
		return module;

	}

	/** Load a prepared graph (see graph/prepare.js). Disposes the previous GPU buffers. */
	setGraph( graph ) {

		const old = this.buffers;
		this.graph = graph;
		this.buffers = new GraphBuffers( graph );
		resetMotionUniforms();
		// before the modules: tiers depend on the graph size and modules read gfxState in setGraph
		this.gfx?.setGraph( this.buffers, graph );
		this.positions = new Float32Array( graph.initPos );
		this.positionsVersion ++;
		this._readbackFrame = - 1;
		if ( this._readbackBuffer ) {

			// a mapAsync may still be in flight: dispose once it settles
			const rb = this._readbackBuffer;
			this._readbackBuffer = null;
			if ( this._readbackBusy ) this._staleReadbacks = [ ...( this._staleReadbacks || [] ), rb ];
			else rb.dispose();

		}

		for ( const m of this.modules ) m.setGraph?.( this.buffers, graph, this );
		this.events.dispatchEvent( new CustomEvent( 'graph', { detail: { buffers: this.buffers, graph } } ) );
		// dispose after modules switched over to the new buffers
		if ( old ) setTimeout( () => old.dispose(), 0 );
		return this.buffers;

	}

	_measure() {

		this.width = this.container.clientWidth || window.innerWidth;
		this.height = this.container.clientHeight || window.innerHeight;

	}

	/**
	 * Calls for a resize when devicePixelRatio changes: a `resolution` media query matches one value
	 * only, so it is armed again after every change.
	 */
	_watchPixelRatio() {

		if ( typeof matchMedia !== 'function' ) return;
		this._dprQuery?.removeEventListener( 'change', this._onDpr );
		this._onDpr = () => {

			this._watchPixelRatio();
			this._resizePending = true;
			this.wake();

		};

		this._dprQuery = matchMedia( `(resolution: ${window.devicePixelRatio || 1}dppx)` );
		this._dprQuery.addEventListener( 'change', this._onDpr );

	}

	resize() {

		this._resizePending = false;
		this._measure();
		const w = this.width, h = this.height;
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		// the display may have changed (another monitor, page zoom); a capture owns the ratio while it runs
		const ratio = this.pixelRatio();
		if ( ! this.gfx.capturing && ratio !== this.renderer.getPixelRatio() ) this.renderer.setPixelRatio( ratio );
		this.renderer.setSize( w, h );
		this.gfx.resize(); // pixel budget → render scale → gfxUniforms.pxScale
		for ( const m of this.modules ) m.resize?.( w, h, this );

	}

	/** Whether anything keeps the loop at the display rate (see "Rest" in the header). */
	_busy( now ) {

		const buffers = this.buffers;
		let moving = ! buffers || this.frame - buffers.moveFrame <= 2 || this.gfx.capturing || this._resizePending ||
			! gfxUniforms.viewProj.value.equals( this._idleViewProj ); // the camera moved in the last frame
		if ( ! moving ) for ( const m of this.modules ) if ( m.animating === true ) moving = true;
		if ( moving ) this._activeAt = now; // … and the loop stays awake for idleAfter seconds after the last motion
		return now - this._activeAt < this.options.idleAfter * 1000;

	}

	_tick() {

		const now = performance.now();
		// rest: at idleFps while nothing happens (the frames in between are not rendered at all)
		let idle = false;
		if ( this._idleFps > 0 ) {

			idle = ! this._busy( now );
			if ( idle && now - this._last < 1000 / this._idleFps - 2 ) return;
			if ( this.idle && ! idle ) this.gfx.governor.settle();
			this._idleViewProj.copy( gfxUniforms.viewProj.value );

		}

		this.idle = idle;
		if ( this._resizePending ) this.resize();
		const rawDt = ( now - this._last ) / 1000;
		this._last = now;
		const dt = Math.min( rawDt, 1 / 15 );
		this.time += dt;
		this.frame ++;
		this._fpsAcc += rawDt; this._fpsFrames ++;
		if ( this._fpsAcc >= 0.5 ) {

			this.fps = this._fpsFrames / this._fpsAcc;
			this.frameMs = ( 1000 * this._fpsAcc ) / this._fpsFrames;
			this._fpsAcc = 0; this._fpsFrames = 0;

		}

		if ( this.buffers ) this.buffers.uniforms.time.value = this.time;
		// motion vectors: nodePrev = positions of the previous rendered frame (before the layout moves)
		if ( this.buffers && gfxState.velocity ) this.buffers.snapshotPrev( this.renderer );
		for ( const m of this.modules ) m.update?.( dt, this );
		this.controls?.update( dt );
		this.gfx.update( dt, idle ? 0 : rawDt * 1000 ); // idle frames are not the device's pace
		updateMotionUniforms( this.camera );
		this.pipeline.render();
		commitMotionUniforms();
		this.gfx.afterRender();
		for ( const m of this.modules ) m.afterRender?.( this );
		if ( this.options.readback && this.buffers && this._readbackDue() ) this._readPositions();
		this.events.dispatchEvent( new Event( 'frame' ) );

	}

	/** True while a position readback is in flight (`positions` then lags the GPU by that request). */
	get readbackPending() {

		return this._readbackBusy === true;

	}

	/**
	 * Whether `positions` may be behind the GPU: the positions changed within the last
	 * READBACK_REST frames (buffers.moveFrame; Infinity = unknown = moving), or no readback has been
	 * requested since they came to rest, or one was asked for.
	 */
	_readbackDue() {

		const buffers = this.buffers;
		if ( ! buffers ) return false;
		const moved = buffers.moveFrame;
		return this._readbackWanted || ! ( this.frame - moved > READBACK_REST ) || this._readbackFrame < moved + READBACK_REST;

	}

	/** Asynchronously copies nodePos to `this.positions` (one readback in flight at a time). */
	_readPositions() {

		if ( this._readbackBusy ) return;
		const buffers = this.buffers;
		const frame = this.frame;
		this._readbackWanted = false;
		const attr = buffers.nodePos.value;
		const bytes = attr.array.byteLength;
		if ( ! this._readbackBuffer ) {

			this._readbackBuffer = new THREE.ReadbackBuffer( bytes );
			this._readbackBuffer.name = 'nodePosReadback';

		}

		const rb = this._readbackBuffer;
		this._readbackBusy = true;
		this.renderer.getArrayBufferAsync( attr, rb ).then( () => {

			if ( buffers === this.buffers && rb === this._readbackBuffer && rb.buffer ) {

				const src = new Float32Array( rb.buffer, 0, attr.array.length );
				if ( ! this.positions || this.positions.length !== src.length ) this.positions = new Float32Array( src.length );
				this.positions.set( src );
				this.positionsVersion ++;
				this._readbackFrame = frame;
				this.events.dispatchEvent( new Event( 'positions' ) );

			}

			if ( rb._mapped ) rb.release();

		} ).catch( ( e ) => {

			if ( e && e.name !== 'AbortError' ) console.warn( 'readback failed', e );

		} ).finally( () => {

			this._readbackBusy = false;
			if ( this._staleReadbacks ) {

				for ( const old of this._staleReadbacks ) old.dispose();
				this._staleReadbacks = null;

			}

		} );

	}

	/** Bounding sphere of the latest positions: {center: Vector3, radius}. */
	boundingSphere( target = new THREE.Sphere() ) {

		const p = this.positions;
		const n = this.graph ? this.graph.N : 0;
		if ( ! p || n === 0 ) return target.set( new THREE.Vector3(), 100 );
		let x = 0, y = 0, z = 0;
		for ( let i = 0; i < n; i ++ ) {

			x += p[ 4 * i ]; y += p[ 4 * i + 1 ]; z += p[ 4 * i + 2 ];

		}

		x /= n; y /= n; z /= n;
		// robust radius: 98th percentile distance, so a few outliers do not shrink the view
		const d = new Float32Array( n );
		for ( let i = 0; i < n; i ++ ) d[ i ] = Math.hypot( p[ 4 * i ] - x, p[ 4 * i + 1 ] - y, p[ 4 * i + 2 ] - z );
		d.sort();
		const r = d[ Math.min( n - 1, Math.floor( n * 0.98 ) ) ];
		return target.set( new THREE.Vector3( x, y, z ), Math.max( r, 1 ) );

	}

	dispose() {

		this.renderer.setAnimationLoop( null );
		window.removeEventListener( 'resize', this._onResize );
		this._dprQuery?.removeEventListener( 'change', this._onDpr );
		for ( const type of INPUT_EVENTS ) window.removeEventListener( type, this._onInput, { capture: true } );
		document.removeEventListener( 'visibilitychange', this._onVisibility );
		for ( const m of this.modules ) m.dispose?.();
		this.buffers?.dispose();
		this.controls?.dispose();
		this.gfx?.dispose();
		this.renderer.dispose();

	}

}

export async function createApp( container, options ) {

	const app = new App( container, options );
	await app.init();
	return app;

}
