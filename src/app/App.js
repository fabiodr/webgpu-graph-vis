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
// Options: maxPixelRatio (2), orbitControls (true), readback (true), gfx: { quality, overrides,
// governor, values, timestamps } for the RenderGraph. Without `gfx` the page URL decides
// (parseGfxParams in Quality.js: ?quality= ?governor= ?ao= ?dof= ?shadows= ?traa= ?scale=
// ?debug= ?gputime=), so every page that creates an App understands the same switches.
//
// Frame order: time uniform → snapshotPrev (when the scene pass has a velocity attachment) →
// modules.update → controls.update → gfx.update (governor, clip planes, focus, overlay camera) →
// updateMotionUniforms → pipeline.render → commitMotionUniforms → gfx.afterRender (GPU timing) →
// modules.afterRender → readback → 'frame'.
//
// Modules are plain objects/classes with any of these optional members:
//   order                 number, update order (lower first). Suggested: layout 0, visuals 10,
//                         labels 20, interaction 30, ui 40.
//   init(app)             once, when added with app.use()
//   setGraph(buffers, graph, app)   whenever a new graph is loaded (GraphBuffers, PreparedGraph)
//   update(dt, app)       every frame before rendering (dt in seconds, clamped)
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
import { gfxState, updateMotionUniforms, commitMotionUniforms, resetMotionUniforms } from '../render/gfx/shared.js';
import { RenderGraph } from '../render/gfx/RenderGraph.js';
import { parseGfxParams } from './Quality.js';

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
		this.options = { maxPixelRatio: 2, orbitControls: true, readback: true, ...options };
		if ( typeof location !== "undefined" && new URLSearchParams( location.search ).has( "noreadback" ) ) this.options.readback = false;
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
		renderer.setPixelRatio( Math.min( window.devicePixelRatio || 1, this.options.maxPixelRatio ) );
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

		this._onResize = () => this.resize();
		window.addEventListener( 'resize', this._onResize );
		this.resize();

		this._last = performance.now();
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

	resize() {

		this._measure();
		const w = this.width, h = this.height;
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		this.renderer.setSize( w, h );
		this.gfx.resize(); // pixel budget → render scale → gfxUniforms.pxScale
		for ( const m of this.modules ) m.resize?.( w, h, this );

	}

	_tick() {

		const now = performance.now();
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
		this.gfx.update( dt, rawDt * 1000 );
		updateMotionUniforms( this.camera );
		this.pipeline.render();
		commitMotionUniforms();
		this.gfx.afterRender();
		for ( const m of this.modules ) m.afterRender?.( this );
		if ( this.options.readback && this.buffers ) this._readPositions();
		this.events.dispatchEvent( new Event( 'frame' ) );

	}

	/** Asynchronously copies nodePos to `this.positions` (one readback in flight at a time). */
	_readPositions() {

		if ( this._readbackBusy ) return;
		const buffers = this.buffers;
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
