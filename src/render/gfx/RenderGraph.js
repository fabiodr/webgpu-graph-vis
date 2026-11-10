// Render graph: owns the RenderPipeline, the scene pass and the whole post-processing chain, and
// rebuilds them per quality tier. Created by the App as `app.gfx`.
//
// ─── Chain (each stage exists only where the tier asks for it; see src/app/Quality.js) ────────
//
//  scene pass   HDR RGBA16F + depth at the scene scale. Samples per tier (MSAA 4× in medium).
//               TRAA tiers add an MRT { output, velocity }: RGBA16F, normal blending on the
//               velocity attachment, cleared to 0, pass default vec4( 0 ), so a material without
//               mrtNode leaves the velocity alone (shared.js: velocityOutput / skipVelocity).
//   ├─ GTAO     from depth with reconstructed normals (no normal attachment), R8 at the AO scale;
//   │           rotating noise when TRAA accumulates it, else a 3×3 joint bilateral upsample
//   └─► lit     beauty · mix( 1, AO, strength · protect ), protect = 1 − smoothstep( 0.9, 2.5, luma )
//               so emissive cores and bright bundles are not dimmed
//  ─► TRAA      (own lit texture + depth + velocity; maxVelocityLength 16)   | SMAA | MSAA | —
//  ─► graphDof  scatter-as-gather bokeh on the shared DoF uniforms, background in focus
//  ─► bloom     luminance threshold, 5 mips, at bloomScale of the capped resolution
//  ─► final HDR composite: [chromatic aberration, 3 taps of the sharp layer] + bloom
//               [+ anamorphic streak: highlights above streakThreshold only, drawn out sideways
//               at a quarter of the resolution], vignette, exposure
//  ─► renderOutput( Neutral, sRGB )   (pipeline.outputColorTransform = false)
//  ─► [RCAS, 5 taps inline] | [FXAA on an 8-bit target, low tier]
//  ─► zero-mean grain ─► dither ─► label overlay ( base · ( 1 − a ) + rgb ) ─► canvas
//
//  The overlay is its own scene (app.gfx.overlayScene) rendered after tone mapping at full
//  resolution into an RGBA8 target with a camera of its own: TRAA jitters the main camera inside
//  pipeline.render(), so the overlay camera is a copy made every frame before the render. The
//  labels are cut against the scene depth; in TRAA tiers against a copy of it that stands still
//  while the picture does (see _refreshLabelDepth), so the jitter does not make text shimmer.
//
// ─── Resolution ─────────────────────────────────────────────────────────────────────────────
//  renderScale = cap (pixel budget of the tier, or the fixed `scale`) × dynamic factor (governor).
//  It is scenePass.setResolutionScale(): the scene pass, AO, lit, TRAA, DoF and SMAA run at it,
//  bloom and streak follow the cap only (so the glow does not breathe with the governor), the final
//  composite, RCAS, grain, dither and the overlay stay at the drawing-buffer resolution.
//  gfxUniforms.pxScale = renderer pixel ratio × renderScale at all times.
//
// ─── r187 rules this file obeys (recipes post.md / aa.md) ───────────────────────────────────
//  · every node that owns render targets is tracked and disposed on rebuild; rtt() always with
//    depthBuffer: false; addons only ever get texture nodes (no hidden convertToTexture() RTTs)
//  · the velocity MRT node is created once and reused (render contexts are keyed by its id)
//  · rebuild = build the new graph, assign pipeline.outputNode, needsUpdate, dispose the old one
//  · GTAO, bloom, SMAA size themselves from the drawing buffer, not from their input: their
//    scales are set relative to it here
//  · TRAANode is subclassed (StableTRAANode): jitter in scene-pass pixels on top of an existing
//    camera view offset (the camera rig centres the graph beside the panel with one), which is
//    restored afterwards instead of cleared; and the history depth copy also works when the
//    scene pass is smaller than the drawing buffer
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//  app.gfx.config                 resolved configuration (see Quality.js), read only. config.compat:
//                                 the device runs three's compatibility mode, where only the 'low'
//                                 chain is valid: every setting resolves to it (config.requested is
//                                 what was asked for; gfxState.compat for the UI)
//  app.gfx.tier / .setting        'low' | 'medium' | 'high' | 'ultra'  /  the setting incl. 'auto'
//  app.gfx.setQuality( setting )  'auto' | tier name
//  app.gfx.set( key, value )      effect switches and strengths; null = back to the tier default
//      structural (rebuild when the node graph changes): post (false = no effects at all, only
//        anti-aliasing, tone mapping and dither), ao, dof, traa, sharpen, streak, grain,
//        vignette, chromaticAberration (booleans or strengths), shadows (materials react to
//        'gfx'), scale (fixed render scale, null = pixel cap), debug ('ao' | 'depth' |
//        'velocity' | 'coc' | null), toneMapping ('neutral' | 'agx' | 'aces' | 'none')
//      live: aoStrength (0..1), aoRadius (× node radius, default 3.5), dofStrength (0..1),
//        dofMaxBlur (CSS px), bloom (0..2, the settings.bloom scale), bloomThreshold, bloomRadius,
//        exposure, sharpness (RCAS, 0 = max … 2 = off), dynamicResolution (bool), governor (bool)
//  app.gfx.get( key )             the value last set, or undefined
//  app.gfx.setFocus( { distance, range } | null )   override the automatic focus (world units
//                                 along the view axis); null = selected node (range 0.75 R), else
//                                 the orbit target (5 R in an overview, 2.4 R close to the graph)
//  app.gfx.setRenderScale( k )    dynamic factor 0.25..1 on top of the cap (what the governor sets)
//  app.gfx.attachLabels( labels ) puts a TextLabels module into the overlay (srgb output, no depth
//                                 test, soft occlusion from the scene depth, kept across rebuilds)
//  app.gfx.overlayScene           THREE.Scene for screen-space UI meshes (premultiplied, sRGB encoded)
//  app.gfx.overlayCamera          the unjittered copy of app.camera used for it
//  app.gfx.scenePass / .pipeline  the current scene PassNode (replaced on rebuild) / RenderPipeline
//  app.gfx.uniforms               live TSL uniforms of the chain; the shared ones (DoF, fog, motion,
//                                 pxScale) are gfxUniforms of shared.js
//  app.gfx.stats                  { tier, setting, renderScale, capScale, width, height (scene pass),
//                                   passes, targets (of the chain, counted when it is built), fps,
//                                   frameMs, displayFps (display rate measured by the governor, 0
//                                   while it is off), gpuMs (render + compute passes of a frame,
//                                   smoothed, from timestamp queries; null where the adapter has
//                                   none), gpuRenderMs, gpuComputeMs, textures, textureBytes, builds }
//  app.gfx.governor               the Governor (Quality.js); .enabled
//  app.gfx.capture( { scale, maxSize, frames, type, quality } )   Promise<Blob>: photo mode, the
//                                 image rendered with `scale` × the canvas pixel ratio (default 2,
//                                 longer side ≤ maxSize = 4096 px), all effects and labels included
//  app.gfx.setOutput( fn | null ) legacy: fn( scenePass, app ) → output node replaces the chain (the
//                                 pipeline then tone maps and encodes itself, unless fn turns
//                                 pipeline.outputColorTransform off; the overlay is not composited)
//  Event on app.events: 'gfx' (detail: gfxState) after every (re)build and whenever the tier,
//  samples, velocity, shadows (on / map size / radius), lit or dof state changed. gfxState is
//  current before the event; renderScale alone changes without an event.
//  Called by the App: update( dt, frameMs ), afterRender(), resize(), setGraph( buffers, graph ),
//  dispose().
import * as THREE from 'three/webgpu';
import {
	pass, mrt, output, rtt, uv, Fn, vec2, vec3, vec4, float, uniform, mix, max, min, exp, exp2, abs, smoothstep, luminance,
	length, screenUV, screenSize, screenCoordinate, renderOutput, time, interleavedGradientNoise, rand, fract, textureSize,
	perspectiveDepthToViewZ, texture,
} from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import TRAANode from 'three/addons/tsl/display/TRAANode.js';
import { computeHaltonOffsets } from 'three/addons/tsl/utils/TAAUtils.js';
import SMAANode from 'three/addons/tsl/display/SMAANode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { chromaticAberration } from 'three/addons/tsl/display/ChromaticAberrationNode.js';
import { gfxUniforms, gfxState } from './shared.js';
import { graphDof, normalizedDistance, sceneCoc } from './graphDof.js';
import { TIER_ORDER, resolveQuality, structureKey, autoTier, autoCeiling, gpuClass, isSoftwareAdapter, normalizeSetting, Governor } from '../../app/Quality.js';

/** settings.bloom → BloomNode strength. */
const BLOOM_SCALE = 0.9;

export const TONE_MAPPINGS = {
	neutral: THREE.NeutralToneMapping,
	agx: THREE.AgXToneMapping,
	aces: THREE.ACESFilmicToneMapping,
	none: THREE.NoToneMapping,
};

/** app.events after which nodes may ease to another size or place: the label depth follows every frame for a while. */
const LIVE_EVENTS = [ 'graph', 'select', 'hover', 'path', 'lens', 'ripple', 'layout', 'immersive', 'gfx' ];
const STRUCTURAL_KEYS = [ 'post', 'ao', 'dof', 'traa', 'sharpen', 'streak', 'grain', 'vignette', 'chromaticAberration', 'shadows', 'scale', 'debug', 'dynamicResolution' ];
const HALTON = /*@__PURE__*/ computeHaltonOffsets( 32 );
/** The jitter offset nearest the pixel centre: the frame whose depth the labels keep while the picture rests. */
const REST_JITTER = /*@__PURE__*/ HALTON.reduce( ( best, j, i ) => ( Math.hypot( j[ 0 ] - 0.5, j[ 1 ] - 0.5 ) < Math.hypot( HALTON[ best ][ 0 ] - 0.5, HALTON[ best ][ 1 ] - 0.5 ) ? i : best ), 0 );
const _size = /*@__PURE__*/ new THREE.Vector2();
const _v = /*@__PURE__*/ new THREE.Vector3();
const _dir = /*@__PURE__*/ new THREE.Vector3();
const _color = /*@__PURE__*/ new THREE.Color();

// Created once: render contexts are keyed by the id of the MRT node, a fresh one per rebuild
// would leave render objects behind for every material (aa recipe, section 8).
let _velocityMRT = null;
function velocityMRT() {

	if ( _velocityMRT === null ) {

		_velocityMRT = mrt( { output, velocity: vec4( 0 ) } );
		_velocityMRT.setBlendMode( 'velocity', new THREE.BlendMode( THREE.NormalBlending ) );
		_velocityMRT.setClearColor( 'velocity', new THREE.Color( 0, 0, 0 ), 0 );

	}

	return _velocityMRT;

}

/**
 * TRAA with three fixes for this app (r187 TRAANode):
 *  · the jitter is added to an existing camera view offset and that offset is restored after the
 *    frame (the stock node replaces and then clears it);
 *  · the jitter is ± half a pixel of the scene pass, not of the drawing buffer;
 *  · the history depth is copied whenever depth and history have the same size (the stock node
 *    only does it when that size is the drawing buffer's, i.e. never with a scaled scene pass).
 */
class StableTRAANode extends TRAANode {

	static get type() {

		return 'StableTRAANode';

	}

	constructor( beautyNode, depthNode, velocityNode, camera ) {

		super( beautyNode, depthNode, velocityNode, camera );
		this._jittered = false;
		this._view = { enabled: false, fullWidth: 1, fullHeight: 1, offsetX: 0, offsetY: 0, width: 1, height: 1 };

	}

	get beautyRenderTarget() {

		return this.beautyNode.isRTTNode ? this.beautyNode.renderTarget : this.beautyNode.passNode.renderTarget;

	}

	/** Index of the jitter offset (of the 32 Halton offsets) the current frame is rendered with. */
	get jitterIndex() {

		return this._jitterIndex;

	}

	setViewOffset() {

		const camera = this.camera, target = this.beautyRenderTarget;
		const w = target.width, h = target.height;
		if ( w < 2 || h < 2 ) return; // not rendered yet
		camera.updateProjectionMatrix();
		this._originalProjectionMatrix.copy( camera.projectionMatrix );
		this._velocityNode.setProjectionMatrix( this._originalProjectionMatrix );
		const j = HALTON[ this._jitterIndex ], saved = this._view, view = camera.view;
		saved.enabled = view !== null && view.enabled === true;
		if ( saved.enabled ) {

			Object.assign( saved, view );
			// one scene-pass pixel in the units of the existing view
			camera.setViewOffset( view.fullWidth, view.fullHeight, saved.offsetX + ( j[ 0 ] - 0.5 ) * saved.width / w, saved.offsetY + ( j[ 1 ] - 0.5 ) * saved.height / h, view.width, view.height );

		} else {

			camera.setViewOffset( w, h, j[ 0 ] - 0.5, j[ 1 ] - 0.5, w, h );

		}

		this._jittered = true;

	}

	clearViewOffset() {

		// the after-callback also runs in the frame that built the node, where nothing was set
		if ( this._jittered === false ) return;
		this._jittered = false;
		const camera = this.camera, saved = this._view;
		if ( saved.enabled ) camera.setViewOffset( saved.fullWidth, saved.fullHeight, saved.offsetX, saved.offsetY, saved.width, saved.height );
		else camera.clearViewOffset();
		this._velocityNode.setProjectionMatrix( null );
		this._jitterIndex = ( this._jitterIndex + 1 ) % HALTON.length;

	}

	updateBefore( frame ) {

		super.updateBefore( frame );
		const { renderer } = frame;
		const history = this._historyRenderTarget, scene = this.depthNode.passNode.renderTarget;
		renderer.getDrawingBufferSize( _size );
		if ( ( history.width !== _size.width || history.height !== _size.height ) && history.width === scene.width && history.height === scene.height ) {

			renderer.copyTextureToTexture( this.depthNode.value, history.depthTexture );

		}

	}

}

/** SMAA sized from its input texture (the stock node always uses the drawing-buffer size). */
class InputSizedSMAANode extends SMAANode {

	static get type() {

		return 'InputSizedSMAANode';

	}

	setSize() {

		const image = this.textureNode.value.image;
		super.setSize( Math.max( 1, image.width ), Math.max( 1, image.height ) );

	}

}

/**
 * The overlay pass: RGBA8 (PassNode.setup() forces the renderer's half-float output type),
 * cleared to transparent whatever the renderer's clear colour is, and skipped altogether while
 * the overlay scene has nothing visible and the target already holds a clear frame.
 */
class OverlayPassNode extends THREE.PassNode {

	static get type() {

		return 'OverlayPassNode';

	}

	constructor( scene, camera ) {

		super( THREE.PassNode.COLOR, scene, camera, { samples: 0, depthBuffer: false } );
		this.name = 'overlay';
		this._clean = false;
		this.renders = 0;
		this.beforeRender = null; // () => void, right before the overlay scene is drawn (the scene pass of the frame is done)

	}

	setup( builder ) {

		const node = super.setup( builder );
		this.renderTarget.samples = 0;
		this.renderTarget.texture.type = THREE.UnsignedByteType;
		return node;

	}

	updateBefore( frame ) {

		let active = false;
		for ( const child of this.scene.children ) if ( child.visible ) active = true;
		if ( ! active && this._clean ) return;
		const { renderer } = frame;
		if ( active && this.beforeRender ) this.beforeRender();
		renderer.getClearColor( _color );
		const alpha = renderer.getClearAlpha();
		renderer.setClearColor( 0x000000, 0 );
		super.updateBefore( frame );
		renderer.setClearColor( _color, alpha );
		this._clean = ! active;
		this.renders ++;

	}

	dispose() {

		super.dispose();
		this._clean = false;

	}

}

/** AMD FidelityFX RCAS (robust contrast-adaptive sharpening), 5 taps of a display-referred texture. */
const rcas = ( tex, sharpness ) => Fn( () => {

	const p = uv();
	const texel = vec2( 1 ).div( vec2( textureSize( tex ) ) );
	const tap = ( x, y ) => tex.sample( p.add( vec2( x, y ).mul( texel ) ) ).rgb.toVar();
	const e = tap( 0, 0 ), b = tap( 0, - 1 ), d = tap( - 1, 0 ), f = tap( 1, 0 ), h = tap( 0, 1 );
	const mn4 = min( min( b, d ), min( f, h ) ).toVar(), mx4 = max( max( b, d ), max( f, h ) ).toVar();
	// how much sharpening the local contrast tolerates before it clips (guards: flat black / white)
	const hitMin = min( mn4, e ).div( max( mx4.mul( 4 ), vec3( 1e-5 ) ) );
	const hitMax = vec3( 1 ).sub( max( mx4, e ) ).div( min( mn4.mul( 4 ).sub( 4 ), vec3( - 1e-5 ) ) );
	const lobeRGB = max( hitMin.negate(), hitMax );
	const lobe = max( float( - 0.1875 ), min( max( lobeRGB.r, max( lobeRGB.g, lobeRGB.b ) ), float( 0 ) ) ).mul( exp2( sharpness.negate() ) ).toVar();
	return vec4( b.add( d ).add( f ).add( h ).mul( lobe ).add( e ).div( lobe.mul( 4 ).add( 1 ) ), 1 );

} )();

export class RenderGraph {

	/**
	 * @param {import('../../app/App.js').App} app renderer, scene and camera must exist
	 * @param {object} [options] { quality: setting, overrides: { key: value } as in set() for the
	 *   structural keys, values: { key: value } for the live keys of set(), governor: bool | Governor
	 *   options (default: on, off under browser automation) }
	 */
	constructor( app, options = {} ) {

		this.isRenderGraph = true;
		this.app = app;
		this.renderer = app.renderer;
		this.scene = app.scene;
		this.camera = app.camera;
		this.options = options;

		this.pipeline = new THREE.RenderPipeline( this.renderer );
		this.pipeline.outputColorTransform = false; // the chain tone maps and encodes itself
		this.renderer.toneMappingExposure = 1; // exposure is a uniform of the chain

		this.uniforms = {
			aoStrength: uniform( 1 ),
			aoRadius: uniform( 1 ), // world units (set every frame from the node size)
			aoThickness: uniform( 1 ), // world units
			aoScale: uniform( 2 ), // exponent on the AO term
			bloomStrength: uniform( BLOOM_SCALE * 0.8 ),
			bloomRadius: uniform( 0.4 ),
			bloomThreshold: uniform( 0.24 ),
			streak: uniform( 0.35 ),
			streakTint: uniform( new THREE.Color( 0.55, 0.75, 1 ) ),
			streakThreshold: uniform( 1 ), // luminance above which a pixel feeds the streak (full weight at twice that)
			chromaticAberration: uniform( 0.15 ),
			vignette: uniform( 0.22 ),
			exposure: uniform( 1 ),
			sharpness: uniform( 0.4 ), // RCAS: 0 = maximum, 2 = off
			grain: uniform( 0.035 ),
			dither: uniform( 1 / 255 ),
			cameraNear: uniform( 0.5 ), // of the scene camera (cameraNear / cameraFar of TSL are the quad camera's in post passes)
			cameraFar: uniform( 20000 ),
			velocityGain: uniform( 40 ), // debug view
		};
		this.sharedUniforms = gfxUniforms;

		// label / UI overlay
		this.overlayScene = new THREE.Scene();
		this.overlayScene.name = 'overlay';
		this.overlayCamera = new THREE.PerspectiveCamera();
		this.overlayCamera.matrixAutoUpdate = false;
		this.overlayCamera.matrixWorldAutoUpdate = false;
		this.overlayPass = new OverlayPassNode( this.overlayScene, this.overlayCamera );
		this.overlayPass.beforeRender = () => this._refreshLabelDepth();
		this._labels = null;
		// things that move or resize nodes without moving the camera or the layout (selection, lens, …)
		this._onLive = () => ( this._liveUntil = Math.max( this._liveUntil, app.frame + 90 ) );
		for ( const name of LIVE_EVENTS ) app.events.addEventListener( name, this._onLive );

		// quality
		this._setting = normalizeSetting( options.quality );
		this._overrides = { ...( options.overrides || {} ) };
		this._autoTier = null;
		this._values = { bloom: 0.8, aoStrength: 1, aoRadius: 3.5, bloomRadius: 0.4, toneMapping: 'neutral', ...( options.values || {} ) };
		this._dynamic = 1; // governor factor on top of the cap
		this._ceiling = 'high'; // the highest tier the governor may lift 'auto' to (autoCeiling)
		this._demotedLoad = 0; // graph size ( N + E ) under which the governor last stepped the tier down
		this._graphTime = - 1e9; // this._time at the last setGraph
		const automated = typeof navigator !== 'undefined' && navigator.webdriver === true; // screenshots must not adapt
		this.governor = new Governor( typeof options.governor === 'object' && options.governor ? options.governor : {} );
		this.governor.enabled = options.governor === undefined || options.governor === null ? ! automated : !! options.governor;

		this.config = null;
		this.chain = null;
		this.renderScale = 1;
		this._customOutput = null;
		this._focusOverride = null;
		this._focus = { distance: 0, range: 0, blur: 0, valid: false };
		this._sphere = { center: new THREE.Vector3(), radius: 100, version: - 1, time: - 1e9, graph: null, valid: false, speed: 0, latest: - 1, latestFrame: 0, live: true };
		this._liveUntil = 0; // app.frame until which the label depth is refreshed every frame (see _refreshLabelDepth)
		this._time = 0;
		this.stats = { tier: '', setting: '', renderScale: 1, capScale: 1, width: 0, height: 0, passes: 0, targets: 0, fps: 0, frameMs: 0, displayFps: 0, gpuMs: null, gpuRenderMs: null, gpuComputeMs: null, textures: 0, textureBytes: 0, builds: 0 };
		this._gpuPending = false;
		this._capture = null;

		this.build();
		for ( const key in options.values || {} ) this.set( key, options.values[ key ] );

	}

	// ── quality ─────────────────────────────────────────────────────────────────────────────

	get tier() {

		return this.config ? this.config.tier : gfxState.tier;

	}

	get setting() {

		return this._setting;

	}

	get scenePass() {

		return this.chain ? this.chain.scenePass : null;

	}

	setQuality( setting ) {

		setting = normalizeSetting( setting );
		this._setting = setting;
		this._autoTier = null; // 'auto' starts from the device again
		this._demotedLoad = 0;
		this._dynamic = 1;
		this.governor.reset();
		this.refresh();
		return this.config;

	}

	/** Effect switches and strengths, see the header. null / undefined = tier default. */
	set( key, value ) {

		if ( value === undefined ) value = null;
		const U = this.uniforms, V = this._values;
		V[ key ] = value;
		if ( STRUCTURAL_KEYS.includes( key ) ) {

			if ( key === 'debug' && ( value === 'none' || value === '' ) ) value = null;
			this._overrides[ key ] = value;
			if ( key === 'scale' ) this._dynamic = 1;
			this.refresh();
			return this;

		}

		switch ( key ) {

			case 'aoStrength': U.aoStrength.value = value === null ? 1 : Math.max( 0, + value ); break;
			case 'aoRadius': V.aoRadius = value === null ? 3.5 : Math.max( 0.1, + value ); break;
			case 'dofStrength': case 'dofMaxBlur': break; // read by the focus update
			case 'bloom':
				V.bloom = value === null ? 0.8 : Math.max( 0, + value || 0 );
				U.bloomStrength.value = BLOOM_SCALE * V.bloom;
				break;
			case 'bloomThreshold': U.bloomThreshold.value = value === null ? 0.24 : + value; break;
			case 'bloomRadius':
				V.bloomRadius = value === null ? 0.4 : + value;
				this._applyLive();
				break;
			case 'exposure': U.exposure.value = value === null ? 1 : Math.max( 0, + value ); break;
			case 'sharpness': U.sharpness.value = value === null ? 0.4 : + value; break;
			case 'governor': this.governor.enabled = !! value; this._configureGovernor(); break;
			case 'toneMapping':
				V.toneMapping = value in TONE_MAPPINGS ? value : 'neutral';
				this.build();
				break;
			default: console.warn( `RenderGraph.set: unknown key "${key}"` );

		}

		return this;

	}

	get( key ) {

		return this._values[ key ];

	}

	/** Legacy hook (App.setOutput): fn( scenePass, app ) → output node replaces the chain; null restores it. */
	setOutput( fn ) {

		this._customOutput = typeof fn === 'function' ? fn : null;
		this.build();

	}

	_env() {

		const app = this.app;
		const hasWindow = typeof window !== 'undefined';
		const w = app.width || ( hasWindow ? window.innerWidth : 1280 ), h = app.height || ( hasWindow ? window.innerHeight : 800 );
		return {
			coarsePointer: typeof matchMedia === 'function' && matchMedia( '(pointer:coarse)' ).matches,
			smallScreen: Math.min( w, h ) < 600,
			softwareAdapter: isSoftwareAdapter( app.adapterInfo ),
			gpuClass: gpuClass( app.adapterInfo ),
			// three's compatibility mode (a device without 'core-features-and-limits'): only 'low' is valid
			compat: this.renderer.backend.compatibilityMode === true,
			edges: app.graph ? app.graph.E : 0,
			dpr: this.renderer.getPixelRatio(),
			deviceDpr: hasWindow && window.devicePixelRatio > 0 ? window.devicePixelRatio : this.renderer.getPixelRatio(),
			cssWidth: w,
			cssHeight: h,
		};

	}

	_resolve() {

		const env = this._env();
		// 'auto' is decided once (and by the governor afterwards), not on every resize
		if ( this._setting === 'auto' && this._autoTier === null ) this._autoTier = autoTier( env );
		this._ceiling = autoCeiling( env );
		return resolveQuality( this._setting, env, { ...this._overrides, tier: this._autoTier } );

	}

	/** Re-resolves the configuration; rebuilds only when the node graph has to change. */
	refresh() {

		const config = this._resolve();
		if ( this.config === null || structureKey( config ) !== structureKey( this.config ) ) this.build( config );
		else this._applyLive( config );
		return this.config;

	}

	// ── build ───────────────────────────────────────────────────────────────────────────────

	/** (Re)builds the node graph for `config` (default: the current settings) without leaking the old one. */
	build( config = this._resolve() ) {

		const old = this.chain;
		this.config = config;
		this.renderScale = this._scaleFor( config );
		this.chain = this._buildChain( config );
		this.pipeline.outputColorTransform = false;
		if ( this._customOutput ) {

			// legacy contract: the pipeline applies tone mapping and the colour space unless fn turns that off
			this.pipeline.outputColorTransform = true;
			this.pipeline.outputNode = this._customOutput( this.chain.scenePass, this.app );

		} else {

			this.pipeline.outputNode = this.chain.output;

		}

		this.pipeline.needsUpdate = true;
		if ( old ) old.dispose();
		if ( this._labels ) this._labels.setSceneDepth( this.chain.labelDepth );
		this.stats.builds ++;
		this._applyLive( config, true );

	}

	_scaleFor( config ) {

		const k = config.dynamicResolution ? this._dynamic : 1;
		return Math.max( 0.05, config.renderScale * k );

	}

	/** Everything that is a uniform or a resolution scale; also keeps gfxState and the governor current. */
	_applyLive( config = this.config, built = false ) {

		const U = this.uniforms, V = this._values;
		this.config = config;
		if ( ! config.dynamicResolution ) this._dynamic = 1;
		const scale = this._scaleFor( config );
		const cap = Math.min( 1, config.renderScale );
		this.renderScale = scale;
		this.chain.setScale( scale, cap );

		if ( config.grain > 0 ) U.grain.value = config.grain;
		if ( config.chromaticAberration > 0 ) U.chromaticAberration.value = config.chromaticAberration;
		U.vignette.value = typeof V.vignette === 'number' ? V.vignette : 0.22;
		if ( typeof V.streak === 'number' && V.streak > 0 ) U.streak.value = V.streak;
		// a quarter-resolution bloom has mips twice as wide: half the radius keeps the look
		U.bloomRadius.value = V.bloomRadius * ( config.bloomScale <= 0.25 ? 0.5 : 1 );

		gfxUniforms.pxScale.value = this.renderer.getPixelRatio() * scale;
		const changed = gfxState.tier !== config.tier || gfxState.samples !== config.samples || gfxState.velocity !== config.velocity ||
			gfxState.shadows !== config.shadows || gfxState.shadowMapSize !== config.shadowMapSize || gfxState.lit !== config.lit ||
			gfxState.shadowRadius !== config.shadowRadius || gfxState.dof !== config.dof;
		gfxState.tier = config.tier;
		gfxState.samples = config.samples;
		gfxState.velocity = config.velocity;
		gfxState.shadows = config.shadows;
		gfxState.shadowMapSize = config.shadowMapSize;
		gfxState.lit = config.lit;
		gfxState.renderScale = scale;
		gfxState.aa = config.aa;
		gfxState.clearcoat = config.clearcoat;
		gfxState.shadowRadius = config.shadowRadius;
		gfxState.dof = config.dof;
		gfxState.edgeLit = config.edges.lit;
		gfxState.edgeTubes = config.edges.tubes;
		gfxState.edgeDefocus = config.edges.defocus;
		gfxState.compat = config.compat;

		const S = this.stats;
		S.tier = config.tier; S.setting = config.setting; S.renderScale = scale; S.capScale = config.capScale;
		S.passes = this.chain.passes; S.targets = this.chain.targets;
		this._configureGovernor();
		if ( built || changed ) this.app.events.dispatchEvent( new CustomEvent( 'gfx', { detail: gfxState } ) );

	}

	_configureGovernor() {

		const c = this.config;
		// compatibility mode: 'low' is not a choice there, so nothing may be stepped
		this.governor.configure( { tier: c.tier, explicit: c.explicit || c.compat, ceiling: this._ceiling, scale: this.renderScale, maxScale: c.renderScale, dynamicResolution: c.dynamicResolution } );

	}

	/** Dynamic resolution: factor (0.25 … 1) on top of the tier's pixel cap. No rebuild. */
	setRenderScale( factor ) {

		this._dynamic = Math.min( 1, Math.max( 0.25, + factor || 1 ) );
		this._applyLive();
		return this.renderScale;

	}

	_buildChain( cfg ) {

		const { scene, camera } = this;
		const U = this.uniforms;
		const owned = [];
		const own = ( node ) => ( owned.push( node ), node );
		const scale = this.renderScale;
		const sceneScaled = []; // [ rtt node, factor ]: follow the scene scale
		const capScaled = []; // … follow the capped scale only
		let passes = 1, targets = 2; // scene colour + depth
		// PITFALL: rtt() allocates a depth texture unless told otherwise
		const RT = ( node, factor = 1, options = {}, list = sceneScaled ) => {

			const n = own( rtt( node, null, null, { depthBuffer: false, ...options, resolutionScale: factor > 0 ? factor * scale : 1 } ) );
			if ( factor > 0 ) list.push( [ n, factor ] );
			passes ++; targets ++;
			return n;

		};

		// ── scene pass ──────────────────────────────────────────────────────────────────────
		const samples = cfg.samples > 1 ? cfg.samples : 0;
		const passOptions = { samples };
		// the multisampled colour is only needed for the resolve; the depth is sampled afterwards
		if ( samples > 1 ) passOptions.storeMultisampledColorBuffer = false;
		const scenePass = own( pass( scene, camera, passOptions ) );
		scenePass.name = 'scene';
		scenePass.setResolutionScale( scale );
		if ( cfg.velocity ) scenePass.setMRT( velocityMRT() );
		const beauty = scenePass.getTextureNode( 'output' );
		const depth = scenePass.getTextureNode( 'depth' );
		const velocity = cfg.velocity ? scenePass.getTextureNode( 'velocity' ) : null;
		if ( velocity ) targets ++;
		const distanceAt = ( q ) => perspectiveDepthToViewZ( depth.sample( q ).r, U.cameraNear, U.cameraFar ).negate();

		// ── ambient occlusion ───────────────────────────────────────────────────────────────
		let aoPass = null, aoTex = null;
		if ( cfg.ao ) {

			aoPass = own( ao( depth, null, camera ) ); // normals reconstructed from depth
			aoPass.resolutionScale = cfg.aoScale * scale; // relative to the drawing buffer, read every frame
			aoPass.samples.value = cfg.aoSamples; // baked into the shader
			aoPass.radius = U.aoRadius;
			aoPass.thickness = U.aoThickness;
			aoPass.scale = U.aoScale;
			aoPass.useTemporalFiltering = cfg.aoTemporal; // rotating noise, accumulated by TRAA
			aoTex = aoPass.getTextureNode();
			passes ++; targets ++;
			if ( cfg.aoDenoise === 'bilateral' ) {

				// 3×3 joint bilateral upsample: every tap of the low-resolution AO is weighted by a
				// Gaussian and by the view-distance difference to the full-resolution centre
				const raw = aoTex;
				aoTex = RT( Fn( () => {

					const p = uv();
					const texel = vec2( 1 ).div( vec2( textureSize( raw ) ) );
					const zc = distanceAt( p ).toVar();
					const phi = U.aoRadius.mul( 0.1 ); // world units
					const sum = float( 0 ).toVar(), wsum = float( 0 ).toVar();
					for ( let j = - 1; j <= 1; j ++ ) for ( let i = - 1; i <= 1; i ++ ) {

						const q = p.add( vec2( i, j ).mul( texel ) );
						const w = exp( abs( distanceAt( q ).sub( zc ) ).div( phi ).negate() ).mul( Math.exp( - ( i * i + j * j ) / 2 ) );
						sum.addAssign( raw.sample( q ).r.mul( w ) );
						wsum.addAssign( w );

					}

					return vec4( vec3( sum.div( max( wsum, 1e-4 ) ) ), 1 );

				} )(), 1, { format: THREE.RedFormat, type: THREE.UnsignedByteType } );

			}

		}

		// ── lit composite ───────────────────────────────────────────────────────────────────
		// `hdr` is either a texture node or an inline node; effects that sample at other UVs (or that
		// would call convertToTexture() themselves) get a texture through asTexture()
		let hdr = beauty, isTexture = true;
		const asTexture = () => {

			if ( ! isTexture ) {

				hdr = RT( hdr );
				isTexture = true;

			}

			return hdr;

		};

		if ( aoTex ) {

			hdr = Fn( () => {

				const c = beauty.rgb.toVar();
				// emissive cores and bright bundles are light sources: occlusion only belongs on
				// reflected light, and luminance is the proxy for that without an emissive buffer
				const protect = smoothstep( 0.9, 2.5, luminance( c ) ).oneMinus();
				c.mulAssign( mix( float( 1 ), aoTex.r, U.aoStrength.mul( protect ) ) );
				return vec4( c, 1 );

			} )();
			isTexture = false;

		}

		// ── anti-aliasing on the HDR image ──────────────────────────────────────────────────
		let traaNode = null;
		if ( cfg.aa === 'traa' ) {

			traaNode = own( new StableTRAANode( asTexture(), depth, velocity, camera ) );
			traaNode.maxVelocityLength = 16; // px of motion at which the history is dropped
			hdr = traaNode.getTextureNode();
			passes ++; targets += 3; // resolve, history colour + depth

		} else if ( cfg.aa === 'smaa' ) {

			hdr = own( new InputSizedSMAANode( asTexture() ) ).getTextureNode();
			passes += 3; targets += 3;

		}

		// ── depth for the label overlay: the live depth, or a copy that rests with the picture ──
		let labelDepth = depth, labelSnapshot = null;
		if ( traaNode ) {

			labelSnapshot = new THREE.DepthTexture( 1, 1 ); // sized and filled by _refreshLabelDepth()
			labelSnapshot.name = 'labels.depth';
			owned.push( labelSnapshot );
			labelDepth = texture( labelSnapshot );
			targets ++;

		}

		// ── depth of field ──────────────────────────────────────────────────────────────────
		let dofPass = null;
		if ( cfg.dof ) {

			dofPass = own( graphDof( asTexture(), depth, { cameraNear: U.cameraNear, cameraFar: U.cameraFar, taps: cfg.dofTaps, sceneScale: scale } ) );
			hdr = RT( dofPass.output );
			passes += 2; targets += 2;

		}

		// ── bloom and streak ────────────────────────────────────────────────────────────────
		let bloomPass = null, bloomTex = null, streakTex = null;
		const streakSource = cfg.bloom && cfg.streak ? asTexture() : null; // before the bloom takes `hdr`
		if ( cfg.bloom ) {

			bloomPass = own( bloom( hdr, U.bloomStrength, U.bloomRadius, U.bloomThreshold ) );
			bloomPass.smoothWidth.value = 0.5;
			bloomTex = bloomPass.getTextureNode();
			passes += 12; targets += 11;

		}

		if ( streakSource ) {

			// Anamorphic streak: only what is clearly brighter than a lit surface (emissive cores, the
			// densest bundles) is drawn out sideways. It is fed from the image, not from the bloom: the
			// bloom is a wide glow already, and its streak was a haze across the whole frame. Two targets
			// at a quarter of the capped resolution: highlights (16 bilinear taps, an 8×8 block of the
			// source, so small highlights do not flicker through the downsample), then one horizontal
			// Gaussian of 33 taps (σ = 9 texels).
			const bright = RT( Fn( () => {

				const p = uv();
				const texel = vec2( 1 ).div( vec2( textureSize( streakSource ) ) );
				const sum = vec3( 0 ).toVar();
				for ( let j = - 3; j <= 3; j += 2 ) for ( let i = - 3; i <= 3; i += 2 ) {

					const c = streakSource.sample( p.add( vec2( i, j ).mul( texel ) ) ).rgb.toVar();
					sum.addAssign( c.mul( smoothstep( U.streakThreshold, U.streakThreshold.mul( 2 ), luminance( c ) ) ) );

				}

				return vec4( sum.div( 16 ), 1 );

			} )(), 0.5, {}, capScaled );
			streakTex = RT( Fn( () => {

				const p = uv();
				const step = vec2( 1.5, 0 ).div( vec2( textureSize( bright ) ) );
				const sum = vec3( 0 ).toVar();
				let norm = 0;
				for ( let i = - 16; i <= 16; i ++ ) {

					const w = Math.exp( - i * i / 72 ); // σ = 6 taps of 1.5 texels
					sum.addAssign( bright.sample( p.add( step.mul( i ) ) ).rgb.mul( w ) );
					norm += w;

				}

				return vec4( sum.div( norm ), 1 );

			} )(), 0.5, {}, capScaled );

		}

		// ── final HDR composite ─────────────────────────────────────────────────────────────
		const useCA = cfg.chromaticAberration > 0;
		const sharp = useCA ? chromaticAberration( asTexture(), U.chromaticAberration, vec2( 0.5, 0.5 ), float( 1.1 ) ) : hdr;
		const hdrOut = Fn( () => {

			const c = vec4( sharp ).rgb.toVar();
			if ( bloomTex ) c.addAssign( bloomTex.rgb );
			if ( streakTex ) c.addAssign( streakTex.rgb.mul( U.streakTint ).mul( U.streak ) );
			if ( cfg.vignette ) {

				// aspect-correct, in linear light
				const aspect = screenSize.x.div( max( screenSize.y, 1 ) );
				const p = screenUV.sub( 0.5 ).mul( vec2( aspect, 1 ) );
				const maxR = length( vec2( aspect.mul( 0.5 ), 0.5 ) );
				c.mulAssign( float( 1 ).sub( U.vignette.mul( smoothstep( 0.3, 1.0, length( p ).div( maxR ) ) ) ) );

			}

			return vec4( c.mul( U.exposure ), 1 );

		} )();

		const toneMapping = TONE_MAPPINGS[ this._values.toneMapping ] ?? THREE.NeutralToneMapping;
		const ldr = renderOutput( hdrOut, toneMapping, THREE.SRGBColorSpace );

		// triangular-ish dither (two IGN taps), ± 1 LSB, always the last thing before an 8-bit store
		const dither = ( c ) => {

			const n1 = interleavedGradientNoise( screenCoordinate.xy );
			const n2 = interleavedGradientNoise( screenCoordinate.xy.add( vec2( 17.3, 41.7 ) ) );
			return c.add( vec3( n1.add( n2 ).sub( 1 ).mul( U.dither ) ) );

		};

		// ── display-referred stages ─────────────────────────────────────────────────────────
		let display = ldr, dithered = false;
		if ( cfg.debug ) {

			display = this._debugNode( cfg.debug, { depth, velocity, aoTex } );
			dithered = true;

		} else if ( cfg.sharpen ) {

			// RCAS after tone mapping and encoding (its math assumes [0, 1]), at the output resolution
			display = rcas( RT( ldr, 0 ), U.sharpness );

		} else if ( cfg.aa === 'fxaa' ) {

			// FXAA wants tone mapped sRGB input; the 8-bit target is dithered when it is written
			const ldrTex = RT( Fn( () => vec4( dither( vec4( ldr ).rgb ), 1 ) )(), 1, { type: THREE.UnsignedByteType } );
			display = own( fxaa( ldrTex ) );
			dithered = true;

		}

		const overlay = this.overlayPass.getTextureNode( 'output' );
		passes += 2; targets ++; // overlay pass + output quad
		const outputNode = Fn( () => {

			const c = vec4( display ).rgb.toVar();
			if ( cfg.grain > 0 && ! cfg.debug ) {

				// zero-mean, strongest in the mid tones, animated
				const n = rand( fract( screenUV.add( fract( time ) ) ) ).sub( 0.5 );
				const l = luminance( c );
				c.addAssign( n.mul( U.grain ).mul( l.mul( float( 1 ).sub( l ) ).mul( 4 ).add( 0.15 ) ) );

			}

			if ( ! dithered ) c.assign( dither( c ) );
			// labels / UI: premultiplied, display-referred
			c.assign( c.mul( overlay.a.oneMinus() ).add( overlay.rgb ) );
			return vec4( c, 1 );

		} )();

		return {
			config: cfg, output: outputNode, scenePass, beauty, depth, velocity, aoPass, traaNode, dofPass, bloomPass, passes, targets,
			labelDepth, labelSnapshot, labelDepthFrame: - 1,
			setScale( s, cap ) {

				scenePass.setResolutionScale( s );
				if ( aoPass ) aoPass.resolutionScale = cfg.aoScale * s;
				if ( dofPass ) dofPass.setSceneScale( s );
				for ( const [ node, factor ] of sceneScaled ) node.setResolutionScale( factor * s );
				if ( bloomPass ) bloomPass.setResolutionScale( cfg.bloomScale * cap );
				for ( const [ node, factor ] of capScaled ) node.setResolutionScale( factor * cfg.bloomScale * cap );

			},
			dispose() {

				for ( const node of owned ) node.dispose();
				owned.length = 0;

			},
		};

	}

	_debugNode( view, { depth, velocity, aoTex } ) {

		const U = this.uniforms;
		return Fn( () => {

			const nd = normalizedDistance( depth.sample( uv() ).r, U.cameraNear, U.cameraFar );
			if ( view === 'ao' ) return vec4( vec3( aoTex ? aoTex.r : 1 ), 1 );
			if ( view === 'velocity' ) return velocity ? vec4( velocity.rg.mul( U.velocityGain ).add( 0.5 ), 0.5, 1 ) : vec4( 0.5, 0.5, 0.5, 1 );
			if ( view === 'coc' ) {

				// circle of confusion relative to the maximum radius; background (in focus) is black
				const maxBlur = max( gfxUniforms.dofMaxBlur.mul( gfxUniforms.dofAmount ).mul( gfxUniforms.pxScale ), 1e-3 );
				return vec4( vec3( sceneCoc( nd, U.cameraFar ).div( maxBlur ) ), 1 );

			}

			// depth: white at the near plane, black at the far plane and where there is no geometry
			const n = U.cameraNear.div( U.cameraFar );
			return vec4( vec3( nd.greaterThan( 0.999 ).select( float( 0 ), float( 1 ).sub( nd.sub( n ).div( float( 1 ).sub( n ) ) ) ) ), 1 );

		} )();

	}

	// ── overlay ─────────────────────────────────────────────────────────────────────────────

	/** Wires a TextLabels module into the overlay pass (kept across rebuilds). */
	attachLabels( labels ) {

		this._labels = labels;
		labels.depthTest = false;
		labels.output = 'srgb';
		this.overlayScene.add( labels.mesh );
		labels.setSceneDepth( this.chain.labelDepth );
		return labels;

	}

	/**
	 * The depth the labels are cut against in TRAA tiers. TRAA renders the scene with another
	 * sub-pixel offset every frame, so text that is tested against the live depth shimmers along
	 * every silhouette that crosses it, and the overlay is never temporally filtered. Shifting the
	 * lookup by the jitter cannot repair that: the depth is point sampled, and a shift of less than
	 * a texel selects the same texel. Instead the labels read a COPY of the scene depth
	 * (chain.labelSnapshot) that is refreshed every frame while something moves (the camera, the
	 * layout, or for a moment after an event that resizes nodes: there the shimmer drowns in the
	 * motion) and, while the picture rests, only in the frame of ONE jitter offset, the one nearest
	 * the pixel centre: a still picture gets a still mask, at most 32 frames behind anything this
	 * test does not know about (nodes that change size without one of LIVE_EVENTS). One depth texture
	 * of the scene-pass size, one texture copy per refresh.
	 * Called by the overlay pass right before it draws (the scene pass of the frame is done by then).
	 */
	_refreshLabelDepth() {

		const chain = this.chain, snapshot = chain ? chain.labelSnapshot : null;
		if ( ! snapshot || ! this._labels ) return;
		const app = this.app, source = chain.scenePass.renderTarget.depthTexture;
		const w = source.image.width, h = source.image.height;
		let refresh = false;
		if ( snapshot.image.width !== w || snapshot.image.height !== h || snapshot.format !== source.format || snapshot.type !== source.type ) {

			snapshot.image.width = w; snapshot.image.height = h;
			snapshot.format = source.format; snapshot.type = source.type;
			snapshot.needsUpdate = true; // recreated in the new size by the copy below
			refresh = true;

		}

		const G = gfxUniforms, buffers = app.buffers;
		// buffers.moveFrame: the frame in which node positions last changed (Infinity = unknown = moving)
		const moving = ! G.viewProj.value.equals( G.prevViewProj.value ) || ! buffers || app.frame - buffers.moveFrame <= 1 || app.frame <= this._liveUntil;
		// at rest the chosen offset comes round every 32 frames; the frame count covers a chain whose TRAA
		// is not running (debug views), where there is no jitter either
		if ( refresh || moving || chain.traaNode.jitterIndex === REST_JITTER || app.frame - chain.labelDepthFrame >= HALTON.length ) {

			this.renderer.copyTextureToTexture( source, snapshot );
			chain.labelDepthFrame = app.frame;

		}

	}

	_syncOverlayCamera() {

		const cam = this.camera, o = this.overlayCamera;
		cam.updateMatrixWorld();
		o.matrixWorld.copy( cam.matrixWorld );
		o.matrixWorldInverse.copy( cam.matrixWorldInverse );
		o.projectionMatrix.copy( cam.projectionMatrix );
		o.projectionMatrixInverse.copy( cam.projectionMatrixInverse );
		o.near = cam.near; o.far = cam.far; o.fov = cam.fov; o.aspect = cam.aspect; o.zoom = cam.zoom;
		o.layers.mask = cam.layers.mask;

	}

	// ── per frame ───────────────────────────────────────────────────────────────────────────

	/** Called by the App after the modules and the controls, right before the render. */
	update( dt, frameMs = dt * 1000 ) {

		this._time += dt;
		const S = this.stats, app = this.app;
		S.fps = app.fps; S.frameMs = app.frameMs; S.displayFps = this.governor.displayFps;
		if ( this._capture && ! this._capture.started ) this._beginCapture( this._capture );

		// A load that will go away is not charged to the tier: the layout is moving nodes (its compute
		// passes run inside the frame), or a graph was loaded a moment ago (pipelines compile, the
		// layout starts hot). With GPU timings the headroom test looks at the render passes alone.
		const buffers = app.buffers;
		const transient = !! buffers && ( app.frame - buffers.moveFrame <= 2 || this._time - this._graphTime < 5 );
		const action = this.governor.sample( frameMs, performance.now(), S.gpuMs, { transient, renderMs: S.gpuRenderMs } );
		if ( action ) this._applyGovernor( action );

		this._fitClipPlanes();
		this._updateFocus( dt );

		const U = this.uniforms, cam = this.camera;
		U.cameraNear.value = cam.near;
		U.cameraFar.value = cam.far;
		const nodeRadius = this._nodeRadius();
		U.aoRadius.value = this._values.aoRadius * nodeRadius;
		U.aoThickness.value = 3 * nodeRadius;
		this._syncOverlayCamera();

		this.renderer.getDrawingBufferSize( _size );
		S.width = Math.floor( _size.x * this.renderScale ); S.height = Math.floor( _size.y * this.renderScale );
		const mem = this.renderer.info.memory;
		S.textures = mem.textures; S.textureBytes = mem.texturesSize;

	}

	/** Called by the App after the render: GPU time of the frame from timestamp queries (when available). */
	afterRender() {

		const renderer = this.renderer;
		const job = this._capture;
		if ( job && job.started && -- job.left <= 0 ) this._finishCapture( job );
		if ( ! renderer.backend.trackTimestamp || this._gpuPending || ( this.app.frame & 3 ) !== 0 ) return;
		this._gpuPending = true;
		const S = this.stats;
		// render passes of the last frame + the compute passes of the layout
		Promise.all( [ renderer.resolveTimestampsAsync( 'render' ), renderer.resolveTimestampsAsync( 'compute' ) ] ).then( ( [ render, compute ] ) => {

			const ms = ( render || 0 ) + ( compute || 0 );
			if ( ! ( ms > 0 ) ) return;
			S.gpuRenderMs = render || 0;
			S.gpuComputeMs = compute || 0;
			S.gpuMs = S.gpuMs === null ? ms : S.gpuMs + ( ms - S.gpuMs ) * 0.25;

		} ).catch( () => {} ).finally( () => {

			this._gpuPending = false;

		} );

	}

	_applyGovernor( action ) {

		if ( action.type === 'scale' ) {

			this._dynamic = Math.min( 1, Math.max( 0.05, action.scale / Math.max( 1e-6, this.config.renderScale ) ) );
			this._applyLive();

		} else if ( action.type === 'tier' && this._setting === 'auto' ) {

			// the load a step down was measured with: a much smaller graph starts from the device again
			if ( TIER_ORDER.indexOf( action.tier ) < TIER_ORDER.indexOf( this.config.tier ) ) this._demotedLoad = this._load();
			this._autoTier = action.tier;
			this._dynamic = 1;
			this.refresh();

		}

	}

	/** World radius of a typical node (AO radius and thickness are tied to it). */
	_nodeRadius() {

		const app = this.app;
		let nodes = this._nodes;
		if ( ! nodes || ! app.modules.includes( nodes ) ) nodes = this._nodes = app.modules.find( ( m ) => m && m.isNodeRenderer ) || null;
		if ( nodes && nodes.baseSize > 0 ) return nodes.baseSize * ( nodes.size ?? 1 );
		const g = app.graph;
		return g ? 0.065 * ( g.layoutRadius || 100 ) / Math.cbrt( Math.max( 1, g.N ) ) : 1;

	}

	/**
	 * Bounding sphere of the latest positions (centroid + largest distance; two passes, no sort).
	 * Followed with every readback up to 50k nodes and four times a second above; `speed` is how
	 * fast it grew or drifted between the last two refreshes, `live` whether a readback feeds it.
	 */
	_boundingSphere() {

		const app = this.app, s = this._sphere;
		const g = app.graph, p = app.positions;
		if ( ! g || ! p || g.N === 0 ) {

			s.valid = false;
			return s;

		}

		const fresh = s.graph !== g;
		if ( fresh || s.latest !== app.positionsVersion ) {

			s.latest = app.positionsVersion;
			s.latestFrame = app.frame; // a new graph or a new readback

		}

		const interval = g.N > 50000 ? 0.25 : 0;
		if ( fresh || ( s.version !== app.positionsVersion && this._time - s.time >= interval ) ) {

			const n = Math.min( g.N, Math.floor( p.length / 4 ) );
			let x = 0, y = 0, z = 0;
			for ( let i = 0; i < n; i ++ ) {

				x += p[ 4 * i ]; y += p[ 4 * i + 1 ]; z += p[ 4 * i + 2 ];

			}

			x /= n; y /= n; z /= n;
			let r2 = 0;
			for ( let i = 0; i < n; i ++ ) {

				const dx = p[ 4 * i ] - x, dy = p[ 4 * i + 1 ] - y, dz = p[ 4 * i + 2 ] - z;
				const d2 = dx * dx + dy * dy + dz * dz;
				if ( d2 > r2 ) r2 = d2;

			}

			const radius = Math.max( Math.sqrt( r2 ), 1e-3 );
			const dt = this._time - s.time;
			s.speed = fresh || ! ( dt > 1e-4 ) || ! s.valid ? 0 : ( Math.max( 0, radius - s.radius ) + Math.hypot( x - s.center.x, y - s.center.y, z - s.center.z ) ) / dt;
			s.center.set( x, y, z );
			s.radius = radius;
			s.version = app.positionsVersion;
			s.time = this._time;
			s.graph = g;
			s.valid = Number.isFinite( radius ) && Number.isFinite( x + y + z );
			if ( ! Number.isFinite( s.speed ) ) s.speed = 0;

		}

		// The readback follows the GPU by a few frames. Where there is none (App option readback:
		// false) or it stopped coming, the sphere is that of the seed positions and says nothing
		// about where the layout has gone since.
		s.live = app.options.readback !== false && ( app.frame - s.latestFrame <= 60 || app.positionsCurrent === true ); // (the readback rests with the layout)
		return s;

	}

	/**
	 * Fits camera.near / camera.far to the graph every frame, so the depth buffer spends its
	 * precision on the graph at any zoom (AO turns into speckle otherwise, post recipe). The fit is
	 * generous, because the sphere is a few frames old and a layout can grow fast: near at half the
	 * distance to the sphere, far at 1.5 radii behind its centre (a 24-bit depth step of ≈ 1e-6 R at
	 * the far side, against node radii of ≈ 1e-2 R), more when the sphere grows. Without a live
	 * readback the nominal layout radius is used with the wide margins of the camera rig. Handles
	 * the camera inside the sphere (immersive view) and the guide sphere of the sphere layout.
	 */
	_fitClipPlanes() {

		const cam = this.camera, app = this.app, s = this._boundingSphere();
		if ( ! s.valid ) return;
		const d = cam.position.distanceTo( s.center );
		// nodes at the rim have a radius, selected ones grow, halos and curves bulge a little
		let R, nearest, farthest, minNear;
		if ( s.live ) {

			// what the sphere may have gained since it was measured and until it is measured again
			const lag = this._time - s.time + 0.25;
			R = s.radius + 4 * this._nodeRadius() + Math.min( s.speed * lag, s.radius );
			nearest = d - R;
			farthest = d + 1.5 * R;
			minNear = 0.004 * R;

		} else {

			R = Math.max( s.radius, app.graph.layoutRadius || 0 ) + 4 * this._nodeRadius();
			nearest = d - 1.6 * R;
			farthest = d + 4 * R;
			minNear = 0.002 * R;

		}

		const buffers = app.buffers;
		if ( buffers && buffers.uniforms.layoutBlend && buffers.uniforms.layoutBlend.value > 0.001 ) {

			// the lat / long guide sphere of the sphere layout
			const sr = buffers.uniforms.sphereRadius.value * 1.02;
			const dg = cam.position.distanceTo( buffers.uniforms.graphCenter.value );
			nearest = Math.min( nearest, dg - sr );
			farthest = Math.max( farthest, dg + sr );

		}

		const far = Math.max( farthest, R * 0.01 );
		// inside (or almost inside) the graph the near plane cannot follow the geometry: a fixed
		// fraction of its size keeps far / near below ≈ 1000, plenty for a 24-bit depth buffer
		const near = Math.max( nearest * 0.5, minNear, far * 1e-5 );
		// a little hysteresis keeps the projection (and with it the TRAA history) still at rest
		if ( Math.abs( near - cam.near ) > 0.005 * cam.near || Math.abs( far - cam.far ) > 0.005 * cam.far ) {

			cam.near = near;
			cam.far = far;
			cam.updateProjectionMatrix();

		}

	}

	/** Focus override: { distance, range } in world units along the view axis, or null for automatic. */
	setFocus( focus ) {

		this._focusOverride = focus && Number.isFinite( focus.distance ) ? { distance: + focus.distance, range: Number.isFinite( focus.range ) ? + focus.range : null } : null;

	}

	/**
	 * Eased focus distance and range into the shared DoF uniforms: the selected node when there is
	 * one, else the orbit target; the range is relative to the graph radius and tighter while a
	 * node is selected. With DoF off the amount is 0 (also for the strands' analytic defocus).
	 */
	_updateFocus( dt ) {

		const G = gfxUniforms, cfg = this.config, cam = this.camera, app = this.app, f = this._focus;
		const V = this._values;
		const amount = cfg.dof ? ( typeof V.dofStrength === 'number' ? Math.min( 1, Math.max( 0, V.dofStrength ) ) : cfg.dofAmount ) : 0;
		G.dofAmount.value = amount;

		const s = this._sphere;
		const R = s.valid ? s.radius : 100;
		cam.getWorldDirection( _dir );
		let distance, range;
		const sel = app.buffers ? app.buffers.uniforms.selectedIndex.value | 0 : - 1;
		const p = app.positions;
		if ( sel >= 0 && p && 4 * sel + 2 < p.length ) {

			distance = _v.set( p[ 4 * sel ], p[ 4 * sel + 1 ], p[ 4 * sel + 2 ] ).sub( cam.position ).dot( _dir );
			range = R * 0.75;

		} else {

			const target = app.controls && app.controls.target ? app.controls.target : ( s.valid ? s.center : null );
			distance = target ? _v.copy( target ).sub( cam.position ).dot( _dir ) : G.dofFocus.value;
			// An overview stays readable: with a range of 5 R the front and the back of the graph keep a
			// circle of confusion below one CSS pixel. Close to the graph or inside it the range
			// tightens and what lies far behind the focus falls off.
			const close = s.valid ? 1 - THREE.MathUtils.smoothstep( cam.position.distanceTo( s.center ), 1.2 * R, 2.2 * R ) : 0;
			range = R * ( 5 - 2.6 * close );

		}

		const o = this._focusOverride;
		if ( o ) {

			distance = o.distance;
			if ( o.range !== null ) range = o.range;

		}

		distance = Math.max( distance, cam.near );
		range = Math.max( range, 1e-3 );
		// the blur radius follows the on-screen size of the graph, so a zoomed-out graph is not mush
		const graphPx = R * cam.projectionMatrix.elements[ 5 ] * ( app.height || 800 ) * 0.5 / Math.max( distance, 1e-3 );
		const maxBlur = typeof V.dofMaxBlur === 'number' ? V.dofMaxBlur : cfg.dofMaxBlur;
		const blur = Math.min( maxBlur, 0.035 * graphPx );

		if ( ! f.valid || ! ( dt > 0 ) ) {

			f.distance = distance; f.range = range; f.blur = blur; f.valid = true;

		} else {

			const k = 1 - Math.exp( - dt / 0.28 ); // focus pull, ≈ 0.65 s to 90 %
			f.distance += ( distance - f.distance ) * k;
			f.range += ( range - f.range ) * k;
			f.blur += ( blur - f.blur ) * k;

		}

		G.dofFocus.value = f.distance;
		G.dofRange.value = f.range;
		G.dofMaxBlur.value = f.blur;

	}

	// ── app hooks ───────────────────────────────────────────────────────────────────────────

	resize() {

		if ( this.config ) this.refresh(); // the pixel cap depends on the canvas size

	}

	/** Size of the loaded graph as the governor sees it (nodes + edges). */
	_load() {

		const g = this.app.graph;
		return g ? g.N + g.E : 0;

	}

	setGraph() {

		this._sphere.graph = null;
		this._focus.valid = false;
		this._liveUntil = this.app.frame + 90;
		this._graphTime = this._time;
		if ( this._demotedLoad > 0 && this._load() < 0.5 * this._demotedLoad ) {

			// 'auto' was stepped down under a graph more than twice this size: that says nothing about
			// this one. The device's own tier again, at full resolution, with a clean record
			this._demotedLoad = 0;
			this._autoTier = null;
			this._dynamic = 1;
			this.governor.reset();

		} else this.governor.reset( false ); // another load: what was measured does not apply (the tier history does)

		this.refresh(); // the medium tier depends on the edge count

	}

	/**
	 * Photo mode: renders a few frames with the drawing buffer enlarged by `scale` (the whole chain
	 * and the labels follow; the longer side is limited to `maxSize` pixels and to what the device
	 * allows), and resolves with the image. The capture runs inside the frame loop: nodes update
	 * once per animation frame, and a WebGPU canvas only holds its image during the task that drew it.
	 * @param {object} [options] { scale = 2, maxSize = 4096, frames (default: 16 with TRAA so the
	 *   history converges, else 2), type = 'image/png', quality }
	 * @returns {Promise<Blob>}
	 */
	/** True while a capture is pending or running (the drawing buffer is the capture's then). */
	get capturing() {

		return this._capture !== null;

	}

	capture( { scale = 2, maxSize = 4096, frames = null, type = 'image/png', quality } = {} ) {

		if ( this._capture ) return this._capture.promise;
		const job = { scale, maxSize, frames, type, quality, started: false, left: 0 };
		job.promise = new Promise( ( resolve, reject ) => {

			job.resolve = resolve; job.reject = reject;

		} );
		this._capture = job;
		this.app.wake?.();
		return job.promise;

	}

	_setDrawingBuffer( pixelRatio ) {

		const app = this.app;
		this.renderer.setPixelRatio( pixelRatio );
		this.renderer.setSize( app.width, app.height, false );
		this.refresh();
		for ( const m of app.modules ) m.resize?.( app.width, app.height, app ); // pixel-ratio dependent uniforms

	}

	_beginCapture( job ) {

		const renderer = this.renderer, app = this.app;
		const limit = Math.min( job.maxSize, renderer.backend.device ? renderer.backend.device.limits.maxTextureDimension2D : 8192 );
		job.ratio = renderer.getPixelRatio();
		job.saved = { scale: this._overrides.scale, dynamic: this._dynamic, governor: this.governor.enabled };
		const ratio = Math.min( job.ratio * job.scale, limit / Math.max( app.width, app.height ) );
		this.governor.enabled = false;
		this._overrides.scale = 1; // the full resolution of the enlarged drawing buffer, no pixel cap
		this._setDrawingBuffer( ratio );
		job.left = job.frames ?? ( this.config.aa === 'traa' ? 16 : 2 );
		job.started = true;

	}

	_finishCapture( job ) {

		const app = this.app;
		this._capture = null;
		// same task as the render: the canvas still holds the frame
		this.renderer.domElement.toBlob( ( blob ) => ( blob ? job.resolve( blob ) : job.reject( new Error( 'capture failed' ) ) ), job.type, job.quality );
		this._overrides.scale = job.saved.scale;
		this._dynamic = job.saved.dynamic;
		this.governor.enabled = job.saved.governor;
		// the display's ratio as it is now (the window may have changed monitors meanwhile)
		this._setDrawingBuffer( typeof app.pixelRatio === 'function' ? app.pixelRatio() : job.ratio );
		// the resized canvas is empty: put the picture back before this frame is presented (the passes
		// have run for this frame, so this only draws the output again, from the enlarged targets)
		this.pipeline.render();

	}

	dispose() {

		for ( const name of LIVE_EVENTS ) this.app.events.removeEventListener( name, this._onLive );
		if ( this.chain ) this.chain.dispose();
		this.chain = null;
		this.overlayPass.dispose();
		this.pipeline.dispose();

	}

}
