// GPU node labels: every glyph of every shown label is one instance of ONE instanced draw of SDF
// quads, positioned in the vertex shader straight from the live `nodePos` storage buffer. Labels
// therefore never trail their nodes (layout ticks, drags and camera moves are followed in the same
// frame), they can be occluded by nodes in front of them, and thousands of them cost one draw.
// Replaces the HTML overlay of src/render/Labels.js (same selection rules, same API where sensible).
//
// ─── Division of labour ─────────────────────────────────────────────────────────────────────
//  GPU, every frame   position (nodePos), on-screen node radius and emphasis (nodeColor.w, nodeState),
//                     projection, pixel snapping, alpha easing, depth fade, context dimming.
//  CPU, ~8 Hz         WHICH labels show (from the readback app.positions, which may lag): selected,
//                     hovered, neighbours of the selection by degree, then context labels by degree
//                     (mode 'all', or 'auto' when N ≤ autoMaxNodes), greedy overlap culling on a
//                     LabelGrid with `hysteresis` px of stickiness (a shown label is tested with a
//                     shrunken box, a hidden one with an inflated box, so labels do not flicker at
//                     the overlap threshold). A decision runs every `interval` seconds while the
//                     camera, the canvas size or the CONTENT of app.positions changes (a checksum:
//                     the App reads back every frame, so positionsVersion alone never rests), and at
//                     once when the selection / hover / a setting changes. It only writes per-label
//                     alpha end points; nothing is written or uploaded while nothing changes.
//                     Text work is lazy: a candidate is first only MEASURED (segmentation + advance
//                     widths, at most `measureBudget` new labels per decision, cached per node; a
//                     decision that ran out of budget is continued `interval` later even when
//                     nothing moves); its glyphs are appended only if it passes the overlap test.
//
// ─── Buffers ────────────────────────────────────────────────────────────────────────────────
//  labelState  storage vec4 × N        ( alpha0, alpha1, t0, class ): the shader eases
//                                       alpha0 → alpha1 with smoothstep( ( time − t0 ) / fadeTime ),
//                                       the CPU evaluates the same curve when it retargets a label.
//                                       class: 0 context, 1 neighbour, 2 hovered, 3 selected.
//  glyphs      storage vec4 × 2 per glyph  A = ( node index, x, y, 0 ): the atlas cell's left edge
//                                       relative to the label origin and its top edge above the
//                                       baseline, in em; B = ( x, y, w, h ) cell in atlas texels.
//                                       Append-only: a label is laid out the first time it is
//                                       shown; the buffer doubles when full (new storage node →
//                                       one material rebuild) and, at `maxGlyphs`, is compacted to
//                                       the labels of the current decision (plus the ones still
//                                       fading out when they fit in half the buffer). Glyphs of
//                                       labels that are gone are dropped once they make up 3/4 of
//                                       a buffer larger than `trimGlyphs`, so the vertex work stays
//                                       within 4× the glyphs of the labels on screen.
//  atlas       R8 texture              runtime SDF atlas (src/render/sdfAtlas.js), sampled only in
//                                       the fragment stage.
//  Vertex stage bindings: nodePos, nodeColor, nodeState + labelState, glyphs (5 storage buffers,
//  0 textures). Fragment stage: the atlas (+ the optional scene depth texture), 0 storage buffers.
//
// ─── Draw ───────────────────────────────────────────────────────────────────────────────────
//  mesh.count = 2 × glyphCount. Instances [0, G) draw the dark halo (outline + soft shadow) of
//  every glyph, instances [G, 2G) the fill. Primitives of one draw are blended in order, so all
//  halos lie under all fills: the halo of a glyph can never eat into its neighbours (the classic
//  artefact of single-quad outlined SDF text). Fill quads are trimmed to the ink box. Instances of
//  hidden labels (alpha ≈ 0) and of nodes behind the camera are collapsed in the vertex shader.
//
// ─── Typography ─────────────────────────────────────────────────────────────────────────────
//  Constant on-screen size: fontSize CSS px × ( target height / CSS height ), so the text follows
//  the DPR and the resolution of whatever target the mesh is rendered into. The label origin and
//  the baseline are snapped to the pixel grid of that target; glyphs are stretched vertically by a
//  few percent so the x-height is a whole number of pixels (vertical hinting: baseline and x-line
//  of lower-case text fall on pixel boundaries and stay crisp at 10–14 px). The edge is
//  anti-aliased over one pixel from the screen-space derivative of the atlas coordinate.
//  The label sits to the right of its node at ( on-screen node radius + padding ), or outside the
//  halo ring for selected / hovered / neighbour nodes, vertically centred on the node — the same
//  rule as the HTML labels, but evaluated per frame from nodeState, so it glides with the node's
//  growth instead of jumping. Emphasis (weight, size, white → node-hue tint) is continuous too.
//
// ─── Occlusion ──────────────────────────────────────────────────────────────────────────────
//  The quads are written at the depth of the node's front-most point. (a) Rendered inside the
//  scene pass, the ordinary depth test hides text behind nearer nodes (`depthTest`, default true).
//  (b) Rendered in an overlay pass without the scene depth buffer, setSceneDepth( depthTextureNode )
//  compares against the scene depth in the fragment shader with a soft threshold. Selected and
//  hovered labels are always on top (`onTop`).
//
// ─── Color pipeline ─────────────────────────────────────────────────────────────────────────
//  WebGPURenderer applies tone mapping and the output color space only in the final output quad,
//  never per material, so the mesh must emit what its target expects:
//   output = 'linear'  (default) linear working-space values × `gain`: for the HDR scene pass or any
//                      target that is tone mapped and encoded afterwards. Blending is linear-light.
//   output = 'srgb'    display-referred, sRGB-encoded values: for an overlay target composited after
//                      tone mapping and encoding (blending in display space, like HTML text).
//  The material blends premultiplied ( ONE, ONE_MINUS_SRC_ALPHA ), so an overlay target ends up
//  premultiplied: composite it as base × ( 1 − a ) + rgb.
//  Light text blended in linear light looks bolder and its dark halo thinner than the same coverage
//  blended in display space (what browsers do for HTML text). `uniforms.blendGamma` compensates by
//  remapping the coverage: fill a → a^g, halo a → 1 − ( 1 − a )^g. g = 1.8 for 'linear' (matched by
//  eye against the HTML labels through this app's tone mapping, not derived), 1 for 'srgb'.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//  const labels = app.use( new TextLabels( { mode: 'auto' } ) );   // App module, order 20
//  someScene.add( labels.mesh );      // the module never adds it itself (or pass { scene })
//  labels.mode = 'auto' | 'all' | 'off';   labels.maxLabels / maxNeighbors / autoMaxNodes
//  labels.fontSize (CSS px, null = 11, 12 on coarse pointers) / fontFamily / fontWeight
//  labels.depthTest, labels.setSceneDepth( node | null ), labels.output, labels.gain
//  labels.fadeTime, labels.interval, labels.hysteresis, labels.rebuildFont()
//  labels.visible, labels.refresh(), labels.text( i ), labels.alphaOf( i ), labels.stats, labels.uniforms
//  labels.setColors()                  no-op: the tint is read from buffers.nodeColor on the GPU
//  labels.setGraph( buffers, graph, app ), labels.update( dt, app ), labels.dispose()
//  uniforms worth tuning: color, emphasisColor, opacity, tint, haloColor, outline, outlineOpacity,
//  shadowBlur, shadowOpacity, shadowOffset, weight, weightPx, embolden, selScale, pad, snap, hint,
//  depthFade, contextFade, onTop, depthBias, occlusionSoftness, gain, blendGamma.
//  Not supported: orthographic cameras, logarithmic depth, more than one canvas per instance.
//  Tests / Node: pass { source: { raster, measure, metrics } } instead of the canvas glyph source.

import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, positionGeometry, uniform, instancedArray, texture, varyingProperty, float, uint, vec2, vec3, vec4,
	select, mix, clamp, max, min, floor, sqrt, pow, dot, step, smoothstep, fwidth, length,
	cameraViewMatrix, cameraProjectionMatrix, cameraNear, cameraFar, screenSize, screenUV,
	perspectiveDepthToViewZ, sRGBTransferOETF,
} from 'three/tsl';
import { LabelGrid } from './Labels.js';
import { GlyphAtlas, createCanvasRaster, segmentText, layoutUnits, makeKerning, SYSTEM_UI_FONT } from './sdfAtlas.js';

export const CTX = 0, NB = 1, HOV = 2, SEL = 3;

const LINEAR_BLEND_GAMMA = 1.8; // default coverage remap for output = 'linear' (see uniforms.blendGamma)
const REBASE_SECONDS = 2048; // keep the float32 time uniform precise: shift the time base now and then
const smooth = ( k ) => ( k <= 0 ? 0 : k >= 1 ? 1 : k * k * ( 3 - 2 * k ) );
const pow2 = ( n ) => 2 ** Math.ceil( Math.log2( Math.max( 1, n ) ) );

function quadGeometry() {

	// corners in [0, 1]²; y = 0 is the top of the glyph cell
	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0 ] ), 3 ) );
	g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	return g;

}

export class TextLabels {

	constructor( options = {} ) {

		const o = {
			mode: 'auto',
			maxLabels: 300,
			maxNeighbors: 40,
			autoMaxNodes: 150,
			fontSize: null, // CSS px; null = 11 (12 on coarse pointers), like the HTML labels
			fontFamily: SYSTEM_UI_FONT,
			fontWeight: 500,
			letterSpacing: 0.01, // em
			maxChars: 48, // longer labels end with an ellipsis
			interval: 0.12, // s between visibility decisions while something moves
			fadeTime: 0.2, // s
			hysteresis: 2, // CSS px a label must clear / overlap before it appears / disappears
			depthTest: true,
			output: 'linear', // 'linear' | 'srgb', see header
			gain: 1,
			shadow: true, // second atlas tap in the halo layer for the offset soft shadow
			emphasis: 'state', // 'state': from nodeState (eased on the GPU) | 'class': from labelState.w
			measureBudget: 500, // labels measured for the first time per decision (~12 µs each; spreads the cost on 20k graphs)
			glyphCapacity: 8192, // glyphs the buffer starts with (doubles on demand)
			maxGlyphs: 1 << 16, // hard limit of the glyph buffer: 2 MB, ~4000 labels on screen at once
			trimGlyphs: 4096, // above this, drop the glyphs of gone labels once they are 3/4 of the buffer
			sdf: {}, // GlyphAtlas options (size, buffer, radius, cutoff, width, height, maxSize)
			source: null, // { raster, measure, metrics } instead of createCanvasRaster()
			scene: null, // convenience: add the mesh to this scene in init()
			renderOrder: 20,
			...options,
		};
		this.options = o;
		this.order = 20;
		this.isTextLabels = true;
		this.app = null;
		this.graph = null;
		this.buffers = null;
		this.atlas = null;

		this._mode = o.mode;
		this._maxLabels = o.maxLabels;
		this._maxNeighbors = o.maxNeighbors;
		this._autoMaxNodes = o.autoMaxNodes;
		this._fontSize = o.fontSize;
		this._fontFamily = o.fontFamily;
		this._fontWeight = o.fontWeight;
		this._depthTest = o.depthTest;
		this._output = o.output;
		this._depthNode = null;
		this._visible = true;
		this.interval = o.interval;
		this.hysteresis = o.hysteresis;

		this.uniforms = {
			time: uniform( 0 ),
			fade: uniform( o.fadeTime ),
			glyphCount: uniform( 0, 'uint' ),
			cssHeight: uniform( 800 ), // CSS px height of the canvas: target px per CSS px = screenSize.y / cssHeight
			fontSize: uniform( 11 ), // CSS px per em
			atlasInv: uniform( new THREE.Vector2( 1, 1 ) ),
			xHeight: uniform( 0.52 ), // em
			baseline: uniform( 0.35 ), // em: baseline below the node center (vertically centers the text)
			nodeBase: uniform( 1 ), // world radius of a radius-factor-1 node (NodeRenderer.baseSize × size)
			minPx: uniform( 1.5 ), // CSS px, NodeRenderer.minPixelRadius
			pad: uniform( 3 ), // CSS px between node (or halo ring) and text
			selScale: uniform( 1.14 ), // font scale of the selected label (12.5 / 11, as in the HTML labels)
			snap: uniform( 1 ), // 1: snap origin and baseline to target pixels
			hint: uniform( 1 ), // 1: integer x-height below ~32 px
			weight: uniform( 0 ), // em added to every stroke edge (font weight offset)
			weightPx: uniform( 0 ), // target px added to every stroke edge (blend-space compensation)
			embolden: uniform( 0.012 ), // em added for selected / hovered labels (500 → ~600)
			color: uniform( new THREE.Color( 0xd6e0f4 ) ), // context labels (HTML: rgba(214,224,244,.84))
			emphasisColor: uniform( new THREE.Color( 0xffffff ) ), // neighbours / hovered / selected
			opacity: uniform( 0.86 ), // context label opacity
			tint: uniform( 0.3 ), // selected / hovered: mix toward the node hue
			haloColor: uniform( new THREE.Color( 0x02040a ) ),
			haloTint: uniform( 0.5 ), // selected / hovered: halo toward a dark shade of the node color
			outline: uniform( 1.25 ), // CSS px of solid dark outline around the ink
			outlineOpacity: uniform( 0.8 ),
			shadowBlur: uniform( 3 ), // CSS px, limited by the SDF range (0.25 em)
			shadowOpacity: uniform( 0.7 ),
			shadowOffset: uniform( new THREE.Vector2( 0, 0.75 ) ), // CSS px, y down
			depthRange: uniform( new THREE.Vector2( 0, 0 ) ), // ( near, 1 / ( far − near ) ) of the graph along the view axis
			depthFade: uniform( 0.65 ), // far labels keep 1 − depthFade of their opacity
			contextFade: uniform( 0.8 ), // context labels lose this much while a selection exists
			onTop: uniform( 1 ), // selected / hovered labels ignore occlusion
			depthBias: uniform( 1 ), // label depth = node center + depthBias × radius toward the camera
			occlusionSoftness: uniform( 1 ), // soft depth comparison width in node radii (setSceneDepth)
			gain: uniform( o.gain ), // 'linear' output only: multiplies the emitted color
			blendGamma: uniform( o.output === 'srgb' ? 1 : LINEAR_BLEND_GAMMA ), // coverage remap, see the fragment stage
		};

		this._geometry = quadGeometry();
		this._placeholder = new THREE.NodeMaterial();
		this.mesh = new THREE.Mesh( this._geometry, this._placeholder );
		this.mesh.name = 'TextLabels';
		this.mesh.frustumCulled = false;
		this.mesh.renderOrder = o.renderOrder;
		this.mesh.count = 0;
		this.mesh.visible = false;
		this.mesh.raycast = () => {};

		this.stats = { labels: 0, glyphs: 0, capacity: 0, atlas: [ 0, 0 ], atlasGlyphs: 0, decisions: 0, decideMs: 0, uploads: 0, compactions: 0 };

		this._grid = new LabelGrid();
		this._stateNode = null;
		this._glyphNode = null;
		this._atlasNode = null;
		this._atlasTex = null;
		this._atlasVersion = - 1;
		this._atlasSizeVersion = - 1;
		this._source = null;
		this._kerning = null;
		this._capacity = 0;
		this._G = 0;
		this._active = [];
		this._gen = 0;
		this._dirty = true;
		this._starved = false;
		this._lastDecision = - 1e9;
		this._lastSel = - 2;
		this._lastHov = - 2;
		this._lastPositions = - 1;
		this._camKey = new Float64Array( 34 );
		this._neighbors = [];
		this._neighborsOf = - 2;
		this._bounds = { x: 0, y: 0, z: 0, r: 1, version: - 1, t: - 1e9 };
		this._stateRange = [ Infinity, - 1 ];
		this._glyphRange = [ Infinity, - 1 ];
		this._nodeBase = 1;
		this._minPx = 1.5;
		this._time0 = 0;
		this._m = new Float64Array( 16 );

	}

	// ── settings ────────────────────────────────────────────────────────────────────────────

	get mode() {

		return this._mode;

	}

	set mode( v ) {

		this._mode = v;
		this._dirty = true;

	}

	get maxLabels() {

		return this._maxLabels;

	}

	set maxLabels( v ) {

		this._maxLabels = Math.max( 0, v | 0 );
		this._dirty = true;

	}

	get maxNeighbors() {

		return this._maxNeighbors;

	}

	set maxNeighbors( v ) {

		this._maxNeighbors = Math.max( 0, v | 0 );
		this._neighborsOf = - 2;
		this._dirty = true;

	}

	get autoMaxNodes() {

		return this._autoMaxNodes;

	}

	set autoMaxNodes( v ) {

		this._autoMaxNodes = v;
		this._dirty = true;

	}

	/** Font size in CSS px. */
	get fontSize() {

		if ( this._fontSize !== null && this._fontSize !== undefined ) return this._fontSize;
		if ( this._autoFont === undefined ) this._autoFont = typeof matchMedia !== 'undefined' && matchMedia( '(pointer:coarse)' ).matches ? 12 : 11;
		return this._autoFont;

	}

	set fontSize( v ) {

		this._fontSize = v === null || v === undefined ? null : Math.max( 4, + v );
		this._dirty = true;

	}

	get fontFamily() {

		return this._fontFamily;

	}

	set fontFamily( v ) {

		if ( v === this._fontFamily ) return;
		this._fontFamily = v;
		this.rebuildFont();

	}

	get fontWeight() {

		return this._fontWeight;

	}

	set fontWeight( v ) {

		if ( v === this._fontWeight ) return;
		this._fontWeight = v;
		this.rebuildFont();

	}

	get fadeTime() {

		return this.uniforms.fade.value;

	}

	set fadeTime( v ) {

		this.uniforms.fade.value = Math.max( 1e-3, v );

	}

	get visible() {

		return this._visible;

	}

	set visible( v ) {

		this._visible = !! v;
		this.mesh.visible = this._visible && this._G > 0 && this.mesh.material !== this._placeholder;

	}

	/** Ordinary depth test against the depth buffer of the pass the mesh is rendered in. */
	get depthTest() {

		return this._depthTest;

	}

	set depthTest( v ) {

		this._depthTest = !! v;
		if ( this.mesh.material !== this._placeholder ) {

			this.mesh.material.depthTest = this._depthTest;
			this.mesh.material.needsUpdate = true;

		}

	}

	/** 'linear' (scene-referred, tone mapped later) or 'srgb' (display-referred overlay). */
	get output() {

		return this._output;

	}

	set output( v ) {

		if ( v === this._output ) return;
		this._output = v;
		this.uniforms.blendGamma.value = v === 'srgb' ? 1 : LINEAR_BLEND_GAMMA;
		this._rebuildMaterial();

	}

	get gain() {

		return this.uniforms.gain.value;

	}

	set gain( v ) {

		this.uniforms.gain.value = v;

	}

	/**
	 * Soft occlusion for overlay passes: a depth texture node of the scene (for example
	 * scenePass.getTextureNode( 'depth' )), sampled at screenUV and converted with the camera's
	 * near / far (perspective, standard or reversed depth). null turns it off. Rebuilds the material.
	 */
	setSceneDepth( depthTextureNode ) {

		if ( depthTextureNode === this._depthNode ) return;
		this._depthNode = depthTextureNode || null;
		this._rebuildMaterial();

	}

	/** Kept for parity with Labels: the tint comes from buffers.nodeColor on the GPU. */
	setColors() {}

	/** Forces a visibility decision on the next update(). */
	refresh() {

		this._dirty = true;

	}

	text( i ) {

		const t = this.graph && this.graph.labels ? this.graph.labels[ i ] : null;
		return t === null || t === undefined ? String( i ) : String( t );

	}

	// ── module hooks ────────────────────────────────────────────────────────────────────────

	init( app ) {

		this.app = app;
		if ( this.options.scene ) this.options.scene.add( this.mesh );

	}

	setGraph( buffers, graph, app ) {

		if ( app ) this.app = app;
		this._disposeGraph();
		this.buffers = buffers;
		this.graph = graph;
		const N = graph.N;

		const order = new Uint32Array( N );
		for ( let i = 0; i < N; i ++ ) order[ i ] = i;
		order.sort( ( a, b ) => graph.degree[ b ] - graph.degree[ a ] || a - b );
		this._byDegree = order;

		this._stateNode = instancedArray( Math.max( 1, N ), 'vec4' ).setName( 'labelState' );
		this._L = this._stateNode.value.array;
		for ( let i = 0; i < N; i ++ ) this._L[ 4 * i + 2 ] = - 1e3;
		this._glyphStart = new Int32Array( N ).fill( - 1 );
		this._glyphCount = new Uint16Array( N );
		this._widthEm = new Float32Array( N ).fill( - 1 ); // −1: not measured yet
		this._stamp = new Uint32Array( N );
		this._inActive = new Uint8Array( N );
		this._active = [];
		this._gen = 0;
		this._G = 0;
		this._capacity = Math.min( this.options.maxGlyphs, pow2( this.options.glyphCapacity ) );
		this._glyphNode = instancedArray( 2 * this._capacity, 'vec4' ).setName( 'labelGlyphs' );
		this._A = this._glyphNode.value.array;

		this._ensureAtlas();
		this._neighborsOf = - 2;
		this._bounds.version = - 1;
		this._lastSel = this._lastHov = - 2;
		this._lastDecision = - 1e9;
		this._time0 = performance.now() / 1000;
		this._dirty = true;
		this._stateRange = [ 0, Math.max( 1, N ) - 1 ];
		this._glyphRange = [ Infinity, - 1 ];
		this.uniforms.glyphCount.value = 0;
		this.mesh.count = 0;
		this._rebuildMaterial();

	}

	update( dt, app = this.app ) {

		if ( ! this.graph || ! app ) return;
		const U = this.uniforms;
		let now = performance.now() / 1000 - this._time0;
		if ( now > REBASE_SECONDS ) now = this._rebase( now );
		U.time.value = now;
		U.cssHeight.value = Math.max( 1, app.height || 1 );
		U.fontSize.value = this.fontSize;
		this._syncNodeSizing( app );
		this._updateDepthRange( app );

		const BU = this.buffers.uniforms;
		const sel = BU.selectedIndex.value | 0, hov = BU.hoverIndex.value | 0;
		let need = this._dirty || sel !== this._lastSel || hov !== this._lastHov;
		if ( ! need && now - this._lastDecision >= this.interval ) need = this._sceneChanged( app ) || this._starved;
		if ( need ) {

			const t0 = performance.now();
			this._sceneChanged( app ); // store the camera key of this decision
			this._decide( app, now, sel, hov );
			this._dirty = false;
			this._lastSel = sel;
			this._lastHov = hov;
			this._lastDecision = now;
			this.stats.decisions ++;
			this.stats.decideMs = performance.now() - t0;

		}

		this._flush();

	}

	resize() {

		this._dirty = true;

	}

	dispose() {

		this._disposeGraph();
		this.mesh.removeFromParent();
		this._geometry.dispose();
		this._placeholder.dispose();
		this._atlasTex?.dispose();
		this._atlasTex = null;
		this._atlasNode = null;
		this.atlas = null;
		this.graph = null;
		this.buffers = null;

	}

	// ── font / atlas ────────────────────────────────────────────────────────────────────────

	/** Re-rasterizes everything (after a font change, or once a web font has finished loading). */
	rebuildFont() {

		this.atlas = null;
		if ( ! this.graph ) return;
		this._ensureAtlas();
		this._glyphStart.fill( - 1 );
		this._glyphCount.fill( 0 );
		this._widthEm.fill( - 1 );
		this._G = 0;
		this.uniforms.glyphCount.value = 0;
		this.mesh.count = 0;
		this._dirty = true;
		this._rebuildMaterial();

	}

	_ensureAtlas() {

		if ( this.atlas ) return;
		const sdf = this.options.sdf || {};
		const src = this.options.source || createCanvasRaster( { fontFamily: this._fontFamily, fontWeight: this._fontWeight, size: sdf.size } );
		this._source = src;
		this.atlas = new GlyphAtlas( { ...sdf, raster: src.raster } );
		this._kerning = src.measure ? makeKerning( src.measure, this.atlas ) : null;
		const m = src.metrics || {};
		const ascent = m.ascent ?? 0.93, descent = m.descent ?? 0.24;
		this.uniforms.xHeight.value = m.xHeight ?? 0.52;
		// the HTML labels center the line box on the node: baseline = center + ( ascent − descent ) / 2
		this.uniforms.baseline.value = ( ascent - descent ) / 2;
		this._atlasSizeVersion = - 1;
		this._syncAtlas();

	}

	_syncAtlas() {

		const atlas = this.atlas;
		if ( ! this._atlasTex || this._atlasSizeVersion !== atlas.sizeVersion ) {

			const old = this._atlasTex;
			const tex = new THREE.DataTexture( atlas.data, atlas.width, atlas.height, THREE.RedFormat, THREE.UnsignedByteType );
			tex.name = 'TextLabels.atlas';
			tex.minFilter = tex.magFilter = THREE.LinearFilter;
			tex.generateMipmaps = false;
			tex.colorSpace = THREE.NoColorSpace;
			tex.needsUpdate = true;
			this._atlasTex = tex;
			if ( this._atlasNode ) this._atlasNode.value = tex;
			else this._atlasNode = texture( tex );
			this.uniforms.atlasInv.value.set( 1 / atlas.width, 1 / atlas.height );
			if ( old ) old.dispose();
			this._atlasSizeVersion = atlas.sizeVersion;
			this._atlasVersion = atlas.version;
			this.stats.uploads ++;

		} else if ( this._atlasVersion !== atlas.version ) {

			this._atlasTex.needsUpdate = true;
			this._atlasVersion = atlas.version;
			this.stats.uploads ++;

		}

		this.stats.atlas = [ atlas.width, atlas.height ];
		this.stats.atlasGlyphs = atlas.glyphs.size;

	}

	// ── text layout into the glyph buffer ───────────────────────────────────────────────────

	_layout( i ) {

		const units = segmentText( this.text( i ), { maxUnits: this.options.maxChars } );
		return layoutUnits( units, this.atlas, { kerning: this._kerning, letterSpacing: this.options.letterSpacing * this.atlas.size } );

	}

	/** Advance width of label i in em (0 when it has no visible glyph). Rasterizes missing glyphs. */
	_measure( i ) {

		let w = this._widthEm[ i ];
		if ( w >= 0 ) return w;
		const lay = this._layout( i );
		w = lay.glyphs.length > 0 ? lay.width / this.atlas.size : 0;
		this._widthEm[ i ] = w;
		return w;

	}

	/** Lays label i out (once) and appends its glyphs. False when there is no room left. */
	_ensureLayout( i ) {

		if ( this._glyphStart[ i ] >= 0 ) return true;
		const atlas = this.atlas, S = atlas.size;
		const lay = this._layout( i );
		const n = lay.glyphs.length;
		if ( this._G + n > this._capacity && ! this._makeRoom( n ) ) return false;
		const A = this._A;
		let o = 8 * this._G;
		for ( let k = 0; k < n; k ++ ) {

			const g = lay.glyphs[ k ];
			A[ o ] = i;
			A[ o + 1 ] = ( lay.x[ k ] + g.left ) / S;
			A[ o + 2 ] = g.top / S;
			A[ o + 3 ] = 0;
			A[ o + 4 ] = g.x;
			A[ o + 5 ] = g.y;
			A[ o + 6 ] = g.w;
			A[ o + 7 ] = g.h;
			o += 8;

		}

		this._glyphStart[ i ] = this._G;
		this._glyphCount[ i ] = n;
		this._widthEm[ i ] = n > 0 ? lay.width / S : 0;
		if ( n > 0 ) {

			this._glyphRange[ 0 ] = Math.min( this._glyphRange[ 0 ], 8 * this._G );
			this._glyphRange[ 1 ] = Math.max( this._glyphRange[ 1 ], o - 1 );

		}

		this._G += n;
		return true;

	}

	/** Grows the glyph buffer (new storage node → new material) or compacts it at maxGlyphs. */
	_makeRoom( n ) {

		const maxGlyphs = this.options.maxGlyphs;
		if ( this._capacity < maxGlyphs && this._G + n <= maxGlyphs ) {

			const cap = Math.min( maxGlyphs, pow2( Math.max( 2 * this._capacity, this._G + n ) ) );
			const old = this._glyphNode;
			const node = instancedArray( 2 * cap, 'vec4' ).setName( 'labelGlyphs' );
			node.value.array.set( this._A.subarray( 0, 8 * this._G ) );
			this._glyphNode = node;
			this._A = node.value.array;
			this._capacity = cap;
			this._glyphRange = [ Infinity, - 1 ]; // a new buffer is uploaded whole
			this._rebuildMaterial();
			old.value.dispose();
			return true;

		}

		// at the limit: keep the labels placed by the decision in progress; the other active ones
		// (shown before, or fading out) stay only while the buffer is less than half full
		const start = this._glyphStart, count = this._glyphCount, stamp = this._stamp, gen = this._gen;
		const current = [], others = [];
		let total = n;
		for ( const i of this._active ) {

			if ( start[ i ] < 0 ) continue;
			( stamp[ i ] === gen ? current : others ).push( i );
			total += count[ i ];

		}

		this._compact( total <= this._capacity / 2 ? current.concat( others ) : current );
		return this._G + n <= this._capacity;

	}

	/** Rewrites the glyph buffer with the glyphs of the (laid out) labels in `keep` only. */
	_compact( keep ) {

		const A = this._A, start = this._glyphStart, count = this._glyphCount;
		keep.sort( ( a, b ) => start[ a ] - start[ b ] );
		const fresh = new Int32Array( start.length ).fill( - 1 );
		let G = 0;
		for ( const i of keep ) {

			A.copyWithin( 8 * G, 8 * start[ i ], 8 * ( start[ i ] + count[ i ] ) ); // moves down only
			fresh[ i ] = G;
			G += count[ i ];

		}

		this._glyphStart = fresh;
		this._G = G;
		this._glyphRange = [ 0, Math.max( 0, 8 * G - 1 ) ];
		this.stats.compactions ++;

	}

	// ── visibility decision (CPU, low rate) ─────────────────────────────────────────────────

	_decide( app, now, sel, hov ) {

		const g = this.graph, N = g.N;
		const L = this._L, stamp = this._stamp;
		const gen = ++ this._gen;
		let shown = 0, starved = false;

		if ( this._mode !== 'off' && N > 0 ) {

			const pos = app.positions && app.positions.length >= 4 * N ? app.positions : this.buffers.nodePos.value.array;
			const cam = app.camera;
			const W = app.width, H = app.height;
			const m = this._viewProj( cam );
			const P11 = cam.projectionMatrix.elements[ 5 ];
			const grid = this._grid;
			grid.reset( W, H );
			const fontPx = this.fontSize;
			const selScale = this.uniforms.selScale.value;
			const lineH = Math.round( fontPx * 1.27 );
			const hyst = this.hysteresis;
			let budget = this.options.measureBudget;

			const place = ( i, kind, force ) => {

				if ( stamp[ i ] === gen ) return false;
				const x = pos[ 4 * i ], y = pos[ 4 * i + 1 ], z = pos[ 4 * i + 2 ];
				const cw = m[ 3 ] * x + m[ 7 ] * y + m[ 11 ] * z + m[ 15 ];
				if ( ! ( cw > cam.near ) ) return false;
				const sx = ( ( m[ 0 ] * x + m[ 4 ] * y + m[ 8 ] * z + m[ 12 ] ) / cw * 0.5 + 0.5 ) * W;
				const sy = ( 0.5 - ( m[ 1 ] * x + m[ 5 ] * y + m[ 9 ] * z + m[ 13 ] ) / cw * 0.5 ) * H;
				if ( sx < - 40 || sy < - 20 || sx > W + 4 || sy > H + 20 ) return false;
				let wEm = this._widthEm[ i ];
				if ( wEm < 0 ) {

					// first look at this label: measure it (bounded per decision; the rest waits its turn)
					if ( ! force && budget <= 0 ) {

						starved = true;
						return false;

					}

					budget --;
					wEm = this._measure( i );

				}

				if ( ! ( wEm > 0 ) ) return false;
				const off = this._offset( i, kind, cw, P11, H );
				const scale = kind === SEL ? selScale : 1;
				const x0 = sx + off, x1 = x0 + wEm * fontPx * scale + 2;
				const hh = ( kind >= HOV ? lineH * scale + 4 : lineH ) * 0.5;
				const y0 = sy - hh, y1 = sy + hh;
				if ( ! force ) {

					// hysteresis: a shown label has to be overlapped clearly before it goes, a hidden one
					// has to be clear of its neighbours before it comes back (no flicker at the threshold)
					const h = L[ 4 * i + 1 ] > 0 ? - hyst : hyst;
					if ( ! grid.free( x0 - 1 - h, y0 - h * 0.5, x1 + 1 + h, y1 + h * 0.5 ) ) return false;

				}

				if ( ! this._ensureLayout( i ) ) return false; // glyphs go to the GPU only for labels that show
				grid.insert( x0 - 1, y0, x1 + 1, y1 );
				if ( kind >= HOV ) {

					// keep other labels off the selected / hovered node and its halo
					const ring = off - 5;
					grid.insert( sx - ring, sy - ring, sx + ring, sy + ring );

				}

				stamp[ i ] = gen;
				this._setTarget( i, 1, kind, now );
				shown ++;
				return true;

			};

			const hasSel = sel >= 0 && sel < N;
			if ( hasSel ) place( sel, SEL, true );
			if ( hov >= 0 && hov < N && hov !== sel ) place( hov, HOV, true );
			if ( hasSel ) {

				const nb = this._neighborsSorted( sel );
				for ( let k = 0; k < nb.length; k ++ ) place( nb[ k ], NB, false );

			}

			if ( this._mode === 'all' || ( this._mode === 'auto' && N <= this._autoMaxNodes ) ) {

				const order = this._byDegree, cap = this._maxLabels;
				for ( let k = 0; k < N && shown < cap; k ++ ) place( order[ k ], CTX, false );

			}

		}

		// labels that were not placed this time fade out; fully faded ones leave the active list
		const active = this._active, fade = this.uniforms.fade.value;
		let w = 0;
		for ( let k = 0; k < active.length; k ++ ) {

			const i = active[ k ];
			if ( stamp[ i ] !== gen ) this._setTarget( i, 0, L[ 4 * i + 3 ], now );
			if ( L[ 4 * i + 1 ] === 0 && now - L[ 4 * i + 2 ] > fade ) {

				this._inActive[ i ] = 0;
				continue;

			}

			active[ w ++ ] = i;

		}

		active.length = w;
		this.stats.labels = shown;
		this._starved = starved; // candidates are waiting to be measured: decide again although nothing moves

		// every glyph in the buffer is vertex work, shown or not: once the glyphs of labels that are
		// gone make up 3/4 of it, rewrite it with the active labels (shown or still fading) only
		if ( this._G > this.options.trimGlyphs ) {

			const start = this._glyphStart, count = this._glyphCount;
			const keep = [];
			let live = 0;
			for ( let k = 0; k < w; k ++ ) {

				const i = active[ k ];
				if ( start[ i ] < 0 ) continue;
				keep.push( i );
				live += count[ i ];

			}

			if ( this._G > 4 * live ) this._compact( keep );

		}

	}

	/** Retargets label i: the ease restarts from its current alpha (same curve as the shader). */
	_setTarget( i, alpha, kind, now ) {

		const L = this._L, o = 4 * i;
		if ( L[ o + 1 ] === alpha && L[ o + 3 ] === kind ) return;
		if ( L[ o + 1 ] !== alpha ) {

			let cur = L[ o ] + ( L[ o + 1 ] - L[ o ] ) * smooth( ( now - L[ o + 2 ] ) / this.uniforms.fade.value );
			if ( alpha > 0 && kind >= HOV ) cur = Math.max( cur, 0.6 * alpha ); // hovered / selected appear fast
			L[ o ] = cur;
			L[ o + 1 ] = alpha;
			L[ o + 2 ] = now;

		}

		L[ o + 3 ] = kind;
		if ( i < this._stateRange[ 0 ] ) this._stateRange[ 0 ] = i;
		if ( i > this._stateRange[ 1 ] ) this._stateRange[ 1 ] = i;
		if ( alpha > 0 && ! this._inActive[ i ] ) {

			this._inActive[ i ] = 1;
			this._active.push( i );

		}

	}

	/** Current eased alpha of label i (what the shader computes, before depth fade). */
	alphaOf( i, now = performance.now() / 1000 - this._time0 ) {

		const L = this._L, o = 4 * i;
		return L[ o ] + ( L[ o + 1 ] - L[ o ] ) * smooth( ( now - L[ o + 2 ] ) / this.uniforms.fade.value );

	}

	_neighborsSorted( sel ) {

		if ( this._neighborsOf === sel ) return this._neighbors;
		const g = this.graph;
		const s = g.adjStart[ sel ], n = g.degree[ sel ];
		const list = Array.from( g.adjList.subarray( s, s + n ) );
		list.sort( ( a, b ) => g.degree[ b ] - g.degree[ a ] || a - b );
		this._neighbors = list.slice( 0, this._maxNeighbors );
		this._neighborsOf = sel;
		return this._neighbors;

	}

	/**
	 * Horizontal distance (CSS px) from the node center to the label origin; the CPU twin of the
	 * vertex shader's offset (which uses the eased nodeState instead of the discrete kind).
	 */
	_offset( i, kind, cw, P11, H ) {

		const pad = this.uniforms.pad.value;
		const emph = kind === SEL ? 1.5 : kind === HOV ? 1.25 : 1;
		const rWorld = this._nodeBase * ( this.graph.radii ? this.graph.radii[ i ] : 1 ) * emph;
		const rPx = Math.max( this._minPx, ( rWorld * P11 * H * 0.5 ) / cw );
		if ( kind === CTX ) return rPx + pad;
		const ring = rPx * 1.3 + 3 + 1.9 * ( kind === SEL ? 1 : kind === HOV ? 0.5 : 0 );
		return ring + pad + ( kind >= HOV ? 2 : 0 );

	}

	_viewProj( cam ) {

		// column-major P · V
		const p = cam.projectionMatrix.elements, v = cam.matrixWorldInverse.elements, out = this._m;
		for ( let c = 0; c < 4; c ++ ) {

			for ( let r = 0; r < 4; r ++ ) {

				out[ c * 4 + r ] = p[ r ] * v[ c * 4 ] + p[ 4 + r ] * v[ c * 4 + 1 ] + p[ 8 + r ] * v[ c * 4 + 2 ] + p[ 12 + r ] * v[ c * 4 + 3 ];

			}

		}

		return out;

	}

	/** True when the camera, the viewport or the readback changed since the last call. */
	_sceneChanged( app ) {

		const key = this._camKey, cam = app.camera;
		const a = cam.matrixWorld.elements, b = cam.projectionMatrix.elements;
		let changed = false;
		if ( this._lastPositions !== app.positionsVersion ) {

			// the readback runs every frame even when the layout is asleep: compare the content
			this._lastPositions = app.positionsVersion;
			const p = app.positions;
			let sum = 0;
			if ( p ) for ( let i = 0, n = p.length; i < n; i += 4 ) sum += p[ i ] + 1.7 * p[ i + 1 ] + 2.9 * p[ i + 2 ];
			changed = sum !== this._positionsSum;
			this._positionsSum = sum;

		}

		for ( let k = 0; k < 16; k ++ ) {

			if ( key[ k ] !== a[ k ] || key[ 16 + k ] !== b[ k ] ) changed = true;
			key[ k ] = a[ k ];
			key[ 16 + k ] = b[ k ];

		}

		if ( key[ 32 ] !== app.width || key[ 33 ] !== app.height ) changed = true;
		key[ 32 ] = app.width;
		key[ 33 ] = app.height;
		return changed;

	}

	// ── per-frame uniforms ──────────────────────────────────────────────────────────────────

	_syncNodeSizing( app ) {

		let nodes = this._nodes;
		if ( ! nodes || ! app.modules || ! app.modules.includes( nodes ) ) nodes = this._nodes = ( app.modules && app.modules.find( ( m ) => m && m.isNodeRenderer ) ) || null;
		if ( nodes ) {

			this._nodeBase = nodes.baseSize * ( nodes.size ?? 1 );
			this._minPx = nodes.minPixelRadius ?? 1.5;

		} else {

			// NodeRenderer's formula, for apps without one
			this._nodeBase = 0.065 * ( this.graph.layoutRadius || 100 ) / Math.cbrt( Math.max( 1, this.graph.N ) );
			this._minPx = 1.5;

		}

		this.uniforms.nodeBase.value = this._nodeBase;
		this.uniforms.minPx.value = this._minPx;

	}

	/** Depth range of the graph along the view axis, for the depth fade (bounds from the readback). */
	_updateDepthRange( app ) {

		const b = this._bounds, now = performance.now();
		const p = app.positions, n = this.graph.N;
		if ( p && n > 0 && p.length >= 4 * n && b.version !== app.positionsVersion && ( b.version < 0 || now - b.t >= 250 ) ) {

			let x = 0, y = 0, z = 0;
			for ( let i = 0; i < n; i ++ ) {

				x += p[ 4 * i ]; y += p[ 4 * i + 1 ]; z += p[ 4 * i + 2 ];

			}

			x /= n; y /= n; z /= n;
			let s = 0;
			for ( let i = 0; i < n; i ++ ) s += ( p[ 4 * i ] - x ) ** 2 + ( p[ 4 * i + 1 ] - y ) ** 2 + ( p[ 4 * i + 2 ] - z ) ** 2;
			b.x = x; b.y = y; b.z = z;
			b.r = Math.max( 1, 1.6 * Math.sqrt( s / n ) );
			b.version = app.positionsVersion;
			b.t = now;

		}

		const c = app.camera.position;
		const dc = Math.hypot( c.x - b.x, c.y - b.y, c.z - b.z );
		const near = Math.max( 1e-3, dc - b.r ), far = dc + b.r;
		this.uniforms.depthRange.value.set( near, 1 / Math.max( 1e-3, far - near ) );

	}

	_rebase( now ) {

		const shift = now - 8, L = this._L;
		for ( let o = 2; o < L.length; o += 4 ) L[ o ] = Math.max( - 1e3, L[ o ] - shift );
		this._time0 += shift;
		this._lastDecision -= shift;
		this._stateRange = [ 0, L.length / 4 - 1 ];
		return now - shift;

	}

	/** Uploads what changed: label states, new glyphs, new atlas texels. */
	_flush() {

		const sr = this._stateRange;
		if ( sr[ 1 ] >= sr[ 0 ] ) {

			const attr = this._stateNode.value;
			attr.addUpdateRange( 4 * sr[ 0 ], 4 * ( sr[ 1 ] - sr[ 0 ] + 1 ) );
			attr.needsUpdate = true;
			this._stateRange = [ Infinity, - 1 ];
			this.stats.uploads ++;

		}

		const gr = this._glyphRange;
		if ( gr[ 1 ] >= gr[ 0 ] ) {

			const attr = this._glyphNode.value;
			attr.addUpdateRange( gr[ 0 ], gr[ 1 ] - gr[ 0 ] + 1 );
			attr.needsUpdate = true;
			this._glyphRange = [ Infinity, - 1 ];
			this.stats.uploads ++;

		}

		this._syncAtlas();
		this.uniforms.glyphCount.value = this._G;
		this.mesh.count = 2 * this._G;
		this.mesh.visible = this._visible && this._G > 0 && this.mesh.material !== this._placeholder;
		this.stats.glyphs = this._G;
		this.stats.capacity = this._capacity;

	}

	// ── material ────────────────────────────────────────────────────────────────────────────

	_rebuildMaterial() {

		if ( ! this.buffers || ! this._glyphNode || ! this.atlas ) return;
		const old = this.mesh.material;
		this.mesh.material = this._buildMaterial();
		if ( old && old !== this._placeholder ) old.dispose();

	}

	_buildMaterial() {

		const U = this.uniforms, buffers = this.buffers;
		const glyphs = this._glyphNode, labelState = this._stateNode, atlasTex = this._atlasNode;
		const focus = buffers.uniforms.focus;
		const atlas = this.atlas;
		const S = atlas.size, BUF = atlas.buffer, EDGE = atlas.edge, RANGE = atlas.radius;
		const REACH = EDGE * RANGE; // texels of field outside the outline (= BUF for the defaults)
		const srgb = this._output === 'srgb';
		const fromState = this.options.emphasis !== 'class';
		const shadow = !! this.options.shadow;
		const depthNode = this._depthNode;

		const vUv = varyingProperty( 'vec2', 'vLabelUv' ); // atlas texel coordinates
		const vCell = varyingProperty( 'vec4', 'vLabelCell' ); // cell rect in texels ( min, max ), for the shadow tap
		const vColor = varyingProperty( 'vec4', 'vLabelColor' ); // rgb in the output encoding, alpha
		const vParams = varyingProperty( 'vec4', 'vLabelParams' ); // grow px, layer (0 halo, 1 fill), target px per CSS px, on-top
		const vDepth = varyingProperty( 'vec3', 'vLabelDepth' ); // label distance, softness (view units), strong emphasis

		const material = new THREE.NodeMaterial();
		material.name = 'TextLabels';
		material.lights = false;
		material.fog = false;
		material.transparent = true;
		material.depthWrite = false;
		material.depthTest = this._depthTest;
		material.premultipliedAlpha = true;
		material.side = THREE.DoubleSide;

		const encode = ( c ) => ( srgb ? sRGBTransferOETF( c ) : c.mul( U.gain ) );

		material.vertexNode = Fn( () => {

			// instance → layer + glyph record → label record → node
			const G = U.glyphCount;
			const isFill = instanceIndex.greaterThanEqual( G );
			const gi = instanceIndex.sub( select( isFill, G, uint( 0 ) ) ).toVar( 'lGlyph' );
			const gA = glyphs.element( gi.mul( uint( 2 ) ) ).toVar( 'lGA' );
			const gB = glyphs.element( gi.mul( uint( 2 ) ).add( uint( 1 ) ) ).toVar( 'lGB' );
			const node = uint( gA.x.add( 0.5 ) ).toVar( 'lNode' );
			const lab = labelState.element( node ).toVar( 'lState' );
			const pos = buffers.nodePos.element( node );
			const col = buffers.nodeColor.element( node ).toVar( 'lNodeColor' );
			const st = buffers.nodeState.element( node ).toVar( 'lNodeState' );
			const layer = select( isFill, float( 1 ), float( 0 ) );

			// emphasis: eased nodeState (same signal the nodes use) or the discrete label class
			let sel, hov, nb, dim;
			if ( fromState ) {

				sel = st.x; hov = st.w; nb = st.y;
				dim = min( st.z, focus ).mul( hov.oneMinus() );

			} else {

				const cls = lab.w;
				sel = step( 2.5, cls );
				hov = step( 1.5, cls ).mul( sel.oneMinus() );
				nb = step( 0.5, cls ).mul( step( cls, 1.5 ) );
				dim = step( cls, 0.5 ).mul( focus );

			}

			const strong = max( sel, hov ).toVar( 'lStrong' ); // selected / hovered
			const lit = max( strong, nb ).toVar( 'lLit' ); // any emphasis

			// alpha: analytic ease between the CPU-written end points
			const k = clamp( U.time.sub( lab.z ).div( U.fade ), 0, 1 );
			const present = mix( lab.x, lab.y, k.mul( k ).mul( k.mul( - 2 ).add( 3 ) ) );

			const center = cameraViewMatrix.mul( vec4( pos.xyz, 1 ) ).xyz.toVar( 'lCenter' );
			const dist = center.z.negate().toVar( 'lDist' );
			const depthT = clamp( dist.sub( U.depthRange.x ).mul( U.depthRange.y ), 0, 1 );
			const alpha = present
				.mul( float( 1 ).sub( U.depthFade.mul( depthT ).mul( strong.oneMinus() ) ) )
				.mul( float( 1 ).sub( U.contextFade.mul( dim ) ) ).toVar( 'lAlpha' );

			// pixel scales of the current target
			const css = screenSize.y.div( U.cssHeight ).toVar( 'lCss' ); // target px per CSS px
			const P11 = cameraProjectionMatrix.element( 1 ).element( 1 );
			const pxScale = P11.mul( screenSize.y ).mul( 0.5 ).div( max( dist, cameraNear ) ).toVar( 'lPxScale' );

			// on-screen node radius, as NodeRenderer draws it (emphasis growth, minimum size, silhouette)
			const R0 = U.nodeBase.mul( col.w ).mul( sel.mul( 0.5 ).add( hov.mul( 0.25 ) ).add( 1 ) );
			const R = max( R0, U.minPx.mul( css ).div( pxScale ) ).toVar( 'lR' );
			const d2 = dot( center, center );
			const sil = min( sqrt( d2.div( max( d2.sub( R.mul( R ) ), d2.mul( 0.25 ) ) ) ), 2 );
			const rPx = R.mul( sil ).mul( pxScale ).toVar( 'lRPx' );

			// origin: right of the node, or outside its halo ring when it has one
			const ring = rPx.mul( 1.3 ).add( css.mul( max( sel, hov.mul( 0.5 ) ).mul( 1.9 ).add( 3 ) ) );
			const offset = mix( rPx.add( css.mul( U.pad ) ), ring.add( css.mul( strong.mul( 2 ).add( U.pad ) ) ), lit );

			const F = U.fontSize.mul( css ).mul( sel.mul( U.selScale.sub( 1 ) ).add( 1 ) ).toVar( 'lF' ); // target px per em
			const xh = U.xHeight.mul( F );
			const hintY = mix( float( 1 ), max( floor( xh.add( 0.5 ) ), 1 ).div( xh ), U.hint.mul( smoothstep( 24, 40, F ).oneMinus() ) );

			const clipC = cameraProjectionMatrix.mul( vec4( center, 1 ) ).toVar( 'lClip' );
			const sp = clipC.xy.div( clipC.w ).mul( 0.5 ).add( 0.5 ).mul( screenSize ); // px, y up
			const ox = sp.x.add( offset ).toVar( 'lOx' );
			const oy = sp.y.sub( U.baseline.mul( F ) ).toVar( 'lOy' );
			const origin = vec2( mix( ox, floor( ox.add( 0.5 ) ), U.snap ), mix( oy, floor( oy.add( 0.5 ) ), U.snap ) );

			// glyph quad: the whole cell for the halo layer, trimmed to the ink for the fill layer
			const tpp = float( S ).div( F ).toVar( 'lTpp' ); // atlas texels per target px
			const reachPx = float( REACH ).div( tpp );
			const growFill = U.weightPx.add( F.mul( U.weight.add( U.embolden.mul( strong ) ) ) );
			const growHalo = min( U.outline.add( strong.mul( 0.5 ) ).mul( css ), max( reachPx.sub( 1.25 ), 0 ) );
			const grow = select( isFill, growFill, growHalo ).toVar( 'lGrow' );
			const inset = select( isFill, clamp( float( BUF ).sub( growFill.add( 1 ).mul( tpp ).add( 1 ) ), 0, BUF ), float( 0 ) );
			const t = positionGeometry.xy.mul( gB.zw.sub( inset.mul( 2 ) ) ).add( inset ).toVar( 'lT' ); // texels in the cell
			const em = vec2( gA.y.add( t.x.div( S ) ), gA.z.sub( t.y.div( S ) ) );
			const P = origin.add( em.mul( vec2( F, F.mul( hintY ) ) ) );
			const ndc = P.div( screenSize ).mul( 2 ).sub( 1 );

			// depth: the node's front-most point; selected / hovered labels go to the near plane
			const zNear = cameraNear.negate().mul( 1.001 );
			const zLabel = min( center.z.add( R.mul( U.depthBias ) ), zNear );
			const top = step( 0.5, strong ).mul( U.onTop );
			const zv = mix( zLabel, zNear, top );
			const cz = cameraProjectionMatrix.mul( vec4( 0, 0, zv, 1 ) );

			// colors (per quad, so the output encoding is applied here)
			const hue = col.xyz.div( max( max( col.x, max( col.y, col.z ) ), 1e-3 ) );
			const fillC = mix( mix( U.color, U.emphasisColor, lit ), mix( hue, vec3( 1 ), 0.25 ), U.tint.mul( strong ) );
			const haloC = mix( U.haloColor, col.xyz.mul( 0.22 ), U.haloTint.mul( strong ) );
			const fillA = mix( U.opacity, float( 1 ), lit );

			vUv.assign( gB.xy.add( t ) );
			vCell.assign( vec4( gB.xy, gB.xy.add( gB.zw ) ) );
			vColor.assign( vec4( encode( select( isFill, fillC, haloC ) ), alpha.mul( select( isFill, fillA, float( 1 ) ) ) ) );
			vParams.assign( vec4( grow, layer, css, top ) );
			vDepth.assign( vec3( zLabel.negate(), max( R.mul( U.occlusionSoftness ), 1e-4 ), strong ) );

			const visible = alpha.greaterThan( 1 / 255 ).and( clipC.w.greaterThan( cameraNear ) ).and( gB.z.greaterThan( 0 ) );
			return select( visible, vec4( ndc, cz.z.div( cz.w ), 1 ), vec4( 0, 0, 2, 1 ) );

		} )();

		material.colorNode = Fn( () => {

			// screen px per atlas texel from the derivative of the texel coordinate (x and y differ
			// by the hinting stretch; the mean is exact to a few percent)
			const fw = max( fwidth( vUv ), vec2( 1e-5 ) );
			const pxPerTexel = float( 2 ).div( fw.x.add( fw.y ) ).toVar( 'lPxPerTexel' );
			const sd = atlasTex.sample( vUv.mul( U.atlasInv ) ).r.sub( EDGE ).mul( RANGE ).mul( pxPerTexel ); // px, > 0 inside
			const body = clamp( sd.add( vParams.x ).add( 0.5 ), 0, 1 ); // the ink (fill) or the solid outline (halo)

			// soft shadow / glow under the outline, reaching 0 before the field runs out
			const css = vParams.z;
			const offPx = U.shadowOffset.mul( css );
			const reach = float( REACH ).mul( pxPerTexel );
			const soft = max( min( U.shadowBlur.mul( css ), reach.sub( length( offPx ) ).mul( 0.95 ) ), 0.5 );
			let sdShadow = sd;
			if ( shadow ) {

				const uvS = clamp( vUv.sub( offPx.div( pxPerTexel ) ), vCell.xy.add( 0.5 ), vCell.zw.sub( 0.5 ) );
				sdShadow = atlasTex.sample( uvS.mul( U.atlasInv ) ).r.sub( EDGE ).mul( RANGE ).mul( pxPerTexel );

			}

			const glow = smoothstep( soft.negate(), 0, sdShadow );
			// selected / hovered labels get a denser halo (they sit on the brightest bundles)
			const halo = max( body.mul( mix( U.outlineOpacity, float( 0.95 ), vDepth.z ) ), glow.mul( glow ).mul( mix( U.shadowOpacity, float( 0.9 ), vDepth.z ) ) );
			// blend-space compensation (blendGamma > 1 when the target blends in linear light): light
			// ink is thinned as a^g, the dark halo strengthened as 1 − (1 − a)^g, which reproduces
			// display-space blending exactly over black and approximately elsewhere
			const aFill = pow( body.mul( vColor.a ), U.blendGamma );
			const aHalo = float( 1 ).sub( pow( float( 1 ).sub( halo.mul( vColor.a ) ), U.blendGamma ) );
			let a = mix( aHalo, aFill, vParams.y );

			if ( depthNode ) {

				// overlay pass: soft comparison against the scene depth
				const sceneDist = perspectiveDepthToViewZ( depthNode.sample( screenUV ).r, cameraNear, cameraFar ).negate();
				const occluded = smoothstep( 0, vDepth.y, vDepth.x.sub( sceneDist ) );
				a = a.mul( float( 1 ).sub( occluded.mul( float( 1 ).sub( vParams.w ) ) ) );

			}

			return vec4( vColor.xyz, a );

		} )();

		return material;

	}

	_disposeGraph() {

		const m = this.mesh.material;
		if ( m && m !== this._placeholder ) {

			this.mesh.material = this._placeholder;
			m.dispose();

		}

		this.mesh.visible = false;
		this.mesh.count = 0;
		this._stateNode?.value.dispose();
		this._glyphNode?.value.dispose();
		this._stateNode = null;
		this._glyphNode = null;
		this._G = 0;
		this._active = [];

	}

}
