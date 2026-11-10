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
//                     a readback can deliver the same positions again, e.g. the last ones before the
//                     App lets the readback rest, so positionsVersion alone says too much), and at
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
// ─── Community labels and semantic zoom ─────────────────────────────────────────────────────
//  At overview distance a node label would cover a dozen nodes; what can be named there is the
//  community. `clusterLabels` ('auto' | 'on' | 'off') adds one name per community of one hierarchy
//  level (`clusterLevel`: 'auto' = the level of the app's community field, the module with
//  isClusterField, followed when it changes; chooseFieldLevel() without one), at most
//  `maxClusterLabels`, largest first, communities of at least `clusterMinSize` members:
//   text     the data's name when it has one ( graph.clusterNames[ cluster ] ), else derived: the
//            label of the member with the highest degree, followed by the member count in a lighter
//            weight and opacity: "Valjean · 17" (communityNames() in Labels.js)
//   place    ABOVE its community, centred over the LIVE centroid (buffers.clusterPos, read in the
//            vertex stage; pushed out onto the sphere in the sphere layout): the text sits
//            `clusterPad` px above the top of the community's projected ball (the centroid is the
//            densest spot of a community: a name printed there covers its hub and collides with
//            the labels around it). The lift is the projected radius, computed in the vertex stage
//            from the world radius of the last decision (1.29 rms radii), at most `clusterLiftMax`
//            of the view's height and not past the top edge of the view. Always on top: a name belongs to a region, not to a point in depth
//   look     its size follows the community on screen: `clusterNameSize` (0.22) × the projected
//            radius, between `clusterNameMin` (11 CSS px) and the size it had before (clusterScale ×
//            1…1.4 with the community's member count, × the node font), so the names of a graph
//            that is small on screen are small captions instead of a layer of headlines. Wider
//            letter spacing, a light tint of the community's colour (the mean colour of its
//            members, so it follows a colour lens), calmer opacity; fades back while a selection
//            exists. A community whose ball is below `clusterMinRadius` = [ 10, 18 ] CSS px fades
//            out with its size (not with clusterLabels 'on')
//   zoom     every community cross-fades on its own between its name and the labels of its nodes,
//            driven by its projected size: t = communityFade(), the mean on-screen distance between
//            neighbouring members (measured inside the finest communities, where the members
//            actually sit) mapped from `clusterFade` = [ 28, 44 ] CSS px to 0…1: below, a node
//            label would cover its neighbours, above, most labels fit. communityWeights( t ) gives
//            the opacity of the name and, in mode 'auto', of the context labels of the members
//            (a cross-fade with a dip, so the two never show at full strength together). Mode
//            'all' keeps every node label at full strength, 'on' keeps every name.
//            Graphs larger than autoMaxNodes get node labels this way too (the camera has to come
//            close), but as a map names towns: the nodes of the resolved communities in degree
//            order, at most `maxZoomLabels` at once (null = zoomLabelBudget(): one per 20000 CSS
//            px² of the view, 16…80) with `zoomSpacingLarge` px of air around each. Without that
//            limit the mid view of a 2000-node graph was 290 labels, a page of text. Graphs whose
//            nodes have no names get none of these labels (an index says nothing; the selection,
//            the hovered node and the neighbours still show theirs).
//            A community whose members are spread through the whole graph (compactness() of the
//            community field) is no place and gets no name.
//   fit      a name sits on what it names: it is placed while its community is at least as large
//            on screen as the name ( nameFits(): width ≤ `clusterFit` × projected diameter, the
//            diameter being 2.58 rms radii ). Of the names that do not fit (a small projection of
//            the graph: the far view, a phone) only those of the largest communities are placed:
//            in order of size while together they cover no more than a fifth of the graph's
//            projected disc ( looseNameArea() ), at most `maxLooseNames` (4). No name is wider than
//            `clusterMaxWidth` (0.3) of the view: its font shrinks, down to 0.85 of the smallest
//            name's ( nameScale() ). A view carries at most `maxNamesInView` names (null =
//            nameBudget(): one per 80000 CSS px², 6…24), the largest communities in it first.
//            With clusterLabels 'on' every name may be placed.
//   overlap  one LabelGrid for everything, in priority order: selected, hovered, neighbours, the
//            community names that dominate their cross-fade (t ≤ ½, largest first), context
//            node labels, then the receding names, which only take what is free. All with the
//            same hysteresis; a name additionally keeps `clusterSpacing` (half its font size) of
//            air to whatever is placed already.
//  A second small instanced draw (a child of `mesh`, drawn before the node labels): 4 vertex storage
//  buffers (clusterPos, the label records, the radii, the glyphs), the shared atlas, the same fragment stage.
//  All names are laid out once per graph (≤ 48 names, a few hundred glyphs); the draw is skipped
//  while no name is visible.
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
//  labels.clusterLabels = 'auto' | 'on' | 'off'   community names: 'auto' cross-fades them with the
//                      node labels (and hides them with mode 'off'), 'on' always shows them
//  labels.clusterLevel = 'auto' | hierarchy height;   labels.maxClusterLabels (48)
//  labels.maxZoomLabels = null | n     context labels the semantic zoom brings up in graphs beyond
//                      autoMaxNodes (null = from the size of the view, see "zoom")
//  labels.clusterText( cluster ) → the name shown for a community of the active level, or null
//  labels.clusterBox( cluster ) → { x0, y0, x1, y1, font } (CSS px) of that name as placed, or null
//  labels.clusterMesh  the second draw (a child of labels.mesh)
//  labels.fontSize (CSS px, null = 11, 12 on coarse pointers) / fontFamily / fontWeight
//  labels.depthTest, labels.setSceneDepth( node | null ), labels.output, labels.gain
//  labels.fadeTime, labels.interval, labels.hysteresis, labels.rebuildFont()
//  labels.visible, labels.refresh(), labels.text( i ), labels.alphaOf( i ), labels.stats, labels.uniforms
//  labels.setColors()                  no-op: the tint is read from buffers.nodeColor on the GPU
//  labels.setGraph( buffers, graph, app ), labels.update( dt, app ), labels.dispose()
//  uniforms worth tuning: color, emphasisColor, opacity, tint, haloColor, outline, outlineOpacity,
//  shadowBlur, shadowOpacity, shadowOffset, weight, weightPx, embolden, selScale, pad, snap, hint,
//  depthFade, contextFade, onTop, depthBias, occlusionSoftness, gain, blendGamma; community names:
//  clusterScale, clusterColor, clusterOpacity, clusterTint, clusterCount, clusterWeight,
//  clusterFocusFade, clusterDepthFade, clusterNameSize, clusterNameMin, clusterPad, clusterLiftMax.
//  labels.stats: { labels, clusters (names shown), zoomBudget (0 = not limited by it), glyphs,
//                  clusterGlyphs, capacity, atlas, … }
//  Not supported: orthographic cameras, logarithmic depth, more than one canvas per instance.
//  Tests / Node: pass { source: { raster, measure, metrics } } instead of the canvas glyph source.

import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, positionGeometry, uniform, instancedArray, texture, varyingProperty, float, uint, vec2, vec3, vec4,
	select, mix, clamp, max, min, abs, floor, sqrt, pow, dot, step, smoothstep, fwidth, length,
	cameraViewMatrix, cameraProjectionMatrix, cameraNear, cameraFar, screenSize, screenUV,
	perspectiveDepthToViewZ, sRGBTransferOETF,
} from 'three/tsl';
import {
	LabelGrid, communityNames, communityFade, communityWeights, communityScale, communitySpacing, nameFits, looseNameArea, nameScale, zoomLabelBudget, nameBudget,
} from './Labels.js';
import { chooseFieldLevel, levelCounts, clusterStats, clusterColors, compactness } from './ClusterField.js';
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
			clusterLabels: 'auto', // 'auto' | 'on' | 'off': community names, see the header
			clusterLevel: 'auto', // hierarchy height of the named communities
			maxClusterLabels: 48,
			clusterMinSize: 3, // communities with fewer members get no name
			clusterMaxChars: 22, // longer names are shortened
			clusterLetterSpacing: 0.03, // em
			clusterFade: [ 28, 44 ], // CSS px of member spacing over which a name hands over to node labels
			clusterFit: 1, // widest name in projected diameters of its community (0 = any); wider ones are "loose"
			maxLooseNames: 4, // names placed although they do not fit (the largest communities, see looseNameArea())
			maxNamesInView: null, // names shown at once (null = by the size of the view, nameBudget())
			clusterSpacing: 0.5, // free space a name keeps to what is placed already, in units of its font size
			clusterMinRadius: [ 10, 18 ], // projected radius (CSS px) over which a community's name fades in
			clusterMaxWidth: 0.3, // widest name as a share of the view's width (its font shrinks)
			maxZoomLabels: null, // context labels the zoom brings up in graphs beyond autoMaxNodes (null = by the size of the view)
			zoomSpacing: 5, // CSS px of free space a context label of the semantic zoom needs around it …
			zoomSpacingLarge: 12, // … and in graphs beyond autoMaxNodes, where only the larger nodes are named
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
		this._clusterLabels = o.clusterLabels;
		this._clusterLevel = o.clusterLevel;
		this._maxClusterLabels = o.maxClusterLabels;
		this._maxZoomLabels = o.maxZoomLabels;
		this._named = false; // the nodes of the graph have names (graph.labels)
		this._field = null; // the app's community field (the level of the names follows it)
		this._cl = null; // community names of the graph, built on first use (see _ensureClusters)
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
			// community names
			clusterGlyphCount: uniform( 0, 'uint' ),
			clusterScale: uniform( 1.3 ), // font size of the smallest community name relative to node labels
			clusterColor: uniform( new THREE.Color( 0xeef3ff ) ),
			clusterOpacity: uniform( 0.82 ),
			clusterTint: uniform( 0.4 ), // toward the community's colour
			clusterCount: uniform( 0.6 ), // opacity of the member count relative to the name
			clusterWeight: uniform( 0.007 ), // em added to the strokes of the name and taken from the count
			clusterFocusFade: uniform( 0.6 ), // names lose this much while a selection exists
			clusterDepthFade: uniform( 0.35 ), // far names keep 1 − clusterDepthFade of their opacity
			clusterNameSize: uniform( 0.22 ), // font size of a name in projected radii of its community …
			clusterNameMin: uniform( 11 ), // … at least this (CSS px), at most clusterScale × its scale × the node font
			clusterPad: uniform( 4 ), // CSS px between the top of the community's ball and the name
			clusterLiftMax: uniform( 0.28 ), // most lift of a name above its centroid, in view heights
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

		// community names: a second draw, a child of the node labels' mesh so that it goes wherever
		// that is put (the overlay scene); drawn first, so node labels are never under a name
		this.clusterMesh = new THREE.Mesh( this._geometry, this._placeholder );
		this.clusterMesh.name = 'TextLabels.communities';
		this.clusterMesh.frustumCulled = false;
		this.clusterMesh.renderOrder = o.renderOrder - 1;
		this.clusterMesh.count = 0;
		this.clusterMesh.visible = false;
		this.clusterMesh.raycast = () => {};
		this.mesh.add( this.clusterMesh );

		this.stats = { labels: 0, clusters: 0, zoomBudget: 0, glyphs: 0, clusterGlyphs: 0, capacity: 0, atlas: [ 0, 0 ], atlasGlyphs: 0, decisions: 0, decideMs: 0, uploads: 0, compactions: 0 };

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

	/** Community names: 'auto' (cross-fading with the node labels) | 'on' (always) | 'off'. */
	get clusterLabels() {

		return this._clusterLabels;

	}

	set clusterLabels( v ) {

		this._clusterLabels = v === 'on' || v === true ? 'on' : v === 'off' || v === false ? 'off' : 'auto';
		this._dirty = true;

	}

	/** Hierarchy height of the named communities: 'auto' or 1 (finest) … D. */
	get clusterLevel() {

		return this._clusterLevel;

	}

	set clusterLevel( v ) {

		if ( v === this._clusterLevel ) return;
		this._clusterLevel = v;
		this._disposeClusters();
		this._dirty = true;

	}

	get maxClusterLabels() {

		return this._maxClusterLabels;

	}

	set maxClusterLabels( v ) {

		v = Math.max( 0, v | 0 );
		if ( v === this._maxClusterLabels ) return;
		this._maxClusterLabels = v;
		this._disposeClusters();
		this._dirty = true;

	}

	/** Context labels the semantic zoom may bring up in graphs beyond autoMaxNodes (null = by the size of the view). */
	get maxZoomLabels() {

		return this._maxZoomLabels;

	}

	set maxZoomLabels( v ) {

		this._maxZoomLabels = v === null || v === undefined ? null : Math.max( 0, v | 0 );
		this._dirty = true;

	}

	/** The name shown for community `cluster` (an index of the active level), or null. */
	clusterText( cluster ) {

		const cl = this._cl || ( this.graph ? this._ensureClusters( true ) : null );
		const slot = cl && cl.K > 0 ? cl.slots.find( ( s ) => s.cluster === cluster ) : null;
		return slot ? slot.text : null;

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
		this._syncVisibility();

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

		if ( this.clusterMesh.material !== this._placeholder ) {

			this.clusterMesh.material.depthTest = this._depthTest;
			this.clusterMesh.material.needsUpdate = true;

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
		this._named = Array.isArray( graph.labels ) && graph.labels.some( ( t ) => t !== null && t !== undefined && t !== '' );

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

		// names on 'auto' sit on the volumes of the community field: follow its level
		const cl = this._cl;
		if ( cl && cl.auto ) {

			const height = this._fieldHeight();
			if ( height > 0 && height !== cl.height ) {

				this._disposeClusters();
				this._dirty = true;

			}

		}

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

		this._syncClusterColors();
		this._flush();

	}

	resize() {

		this._dirty = true;

	}

	dispose() {

		this._disposeGraph();
		this.clusterMesh.removeFromParent();
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
		this._disposeClusters(); // laid out again with the new font on the next decision
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
		let shown = 0, starved = false, named = 0, zoomBudget = 0;
		const showNodes = this._mode !== 'off' && N > 0;
		const cmode = this._clusterLabels;
		const cl = N > 0 && ( cmode === 'on' || ( cmode === 'auto' && showNodes ) ) ? this._ensureClusters() : null;

		if ( showNodes || cl ) {

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

			const place = ( i, kind, force, alpha = 1, margin = 0 ) => {

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
					const h = ( L[ 4 * i + 1 ] > 0 ? - hyst : hyst ) + margin;
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
				this._setTarget( i, alpha, kind, now );
				shown ++;
				return true;

			};

			const hasSel = showNodes && sel >= 0 && sel < N;
			if ( hasSel ) place( sel, SEL, true );
			if ( showNodes && hov >= 0 && hov < N && hov !== sel ) place( hov, HOV, true );
			if ( hasSel ) {

				const nb = this._neighborsSorted( sel );
				for ( let k = 0; k < nb.length; k ++ ) place( nb[ k ], NB, false );

			}

			// community names: the ones that dominate their cross-fade claim their place before the
			// context labels, the receding ones take what is left afterwards
			if ( cl ) {

				this._measureClusters( cl, pos, m, cam, W, H, P11, fontPx );
				named += this._placeClusters( cl, now, grid, fontPx, hyst, W, H, true );

			}

			if ( showNodes ) {

				const all = this._mode === 'all' || ( this._mode === 'auto' && N <= this._autoMaxNodes );
				// semantic zoom ('auto' only): the context labels of a community show with its cross-fade t;
				// graphs beyond autoMaxNodes get the labels of the communities the camera has resolved
				const zoom = cl !== null && this._mode === 'auto' && ( cmode === 'auto' || ! all );
				// … those labels exist only because of the zoom: they have a small budget of their own
				// (the larger nodes, like towns on a map), and there are none when the nodes have no
				// names: a few hundred indices say nothing
				const zoomOnly = zoom && ! all;
				zoomBudget = zoomOnly ? Math.min( this._maxLabels, this._maxZoomLabels ?? zoomLabelBudget( W, H ) ) : 0;
				if ( all || ( zoomOnly && cl.resolved && this._named ) ) {

					const order = this._byDegree, cap = all ? this._maxLabels : zoomBudget;
					const anc = g.anc, D = g.D, l = zoom ? cl.height - 1 : 0, offset = zoom ? cl.offset : 0, fade = zoom ? cl.nodes : null;
					// labels that the zoom brings up keep a distance from what is placed already: a map
					// names the larger towns first and leaves air between them
					const air = zoomOnly ? this.options.zoomSpacingLarge : this.options.zoomSpacing;
					// mode 'all' / small graphs: the limit counts every label; the zoom's budget only its own
					for ( let k = 0, context = 0; k < N && ( all ? shown : context ) < cap; k ++ ) {

						const i = order[ k ];
						if ( ! zoom ) place( i, CTX, false );
						else {

							const a = fade[ anc[ i * D + l ] - offset ];
							if ( a >= 0.06 && place( i, CTX, false, a, air ) ) context ++;

						}

					}

				}

			}

			if ( cl ) named += this._placeClusters( cl, now, grid, fontPx, hyst, W, H, false );

		}

		this._retireClusters( now );
		this.stats.clusters = named;
		this.stats.zoomBudget = zoomBudget;

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

	// ── community names ─────────────────────────────────────────────────────────────────────

	/**
	 * Names, glyphs, buffers and material of the community labels, built the first time they are
	 * needed for a graph. Returns the state, or null when the graph has no communities to name.
	 * @param {boolean} [namesOnly] only resolve the level and the texts (no GPU objects)
	 */
	_ensureClusters( namesOnly = false ) {

		let cl = this._cl;
		if ( cl && ( cl.built || namesOnly ) ) return cl.K > 0 ? cl : null;
		const g = this.graph, o = this.options;
		if ( ! cl ) {

			const level = this._clusterLevel;
			const hierarchy = g.D > 0 && g.anc && g.levelOffset;
			// 'auto': the level whose volumes the community field draws, so a name sits on its volume
			const auto = level === 'auto' || level === null || level === undefined;
			let height = ! hierarchy ? 0 : auto ? this._fieldHeight() || chooseFieldLevel( levelCounts( g ) ) : Math.round( + level ) || 0;
			if ( height !== 0 ) height = Math.min( g.D, Math.max( 1, height ) );
			cl = this._cl = { height, auto, offset: 0, count: 0, K: 0, built: false, slots: [], resolved: false, live: 0, stateDirty: false, colorVersion: - 1, loose: 0, looseArea: 0, looseRoom: Infinity, looseStop: false, placed: 0, graphPx: Infinity };
			if ( height === 0 || this._maxClusterLabels === 0 ) return null;
			cl.offset = g.levelOffset[ height - 1 ];
			cl.count = g.levelOffset[ height ] - cl.offset; // communities of the level
			const picked = communityNames( g, height, { maxChars: o.clusterMaxChars } )
				.filter( ( r ) => r.count >= o.clusterMinSize )
				.sort( ( a, b ) => b.count - a.count || a.cluster - b.cluster )
				.slice( 0, this._maxClusterLabels );
			const largest = picked.length > 0 ? picked[ 0 ].count : 1;
			cl.slots = picked.map( ( r ) => ( { cluster: r.cluster, k: r.cluster - cl.offset, count: r.count, name: r.name, tail: r.tail, text: r.text, scale: communityScale( r.count, largest ), shown: 0, widthEm: 0, gen: 0 } ) );
			cl.K = cl.slots.length;
			// per community of the level (not only the named ones), refreshed by every decision
			cl.sx = new Float32Array( cl.count );
			cl.sy = new Float32Array( cl.count );
			cl.fade = new Float32Array( cl.count ); // position in the semantic zoom: 0 = a name, 1 = node labels
			cl.name = new Float32Array( cl.count ); // opacity of the name …
			cl.nodes = new Float32Array( cl.count ); // … and of the members' labels (communityWeights)
			cl.weight = new Float32Array( cl.count ); // 0 = not a place (dispersed) or behind the camera
			cl.spacing = new Float32Array( cl.count ); // world units: rms radius / √members of the finest communities
			cl.radius = new Float32Array( cl.count ); // projected radius of the community, CSS px (1.29 × rms: the ball)
			cl.world = new Float32Array( cl.count ); // … and in world units
			cl.stats = null;
			// the finest communities inside each community of the level (their member spacing is the one that counts)
			cl.fineParent = null;
			if ( height > 1 && g.clusterParent && g.clusterHeight ) {

				const K1 = g.levelOffset[ 1 ] - g.levelOffset[ 0 ];
				cl.fineParent = new Int32Array( K1 );
				for ( let j = 0; j < K1; j ++ ) {

					let c = j;
					while ( c >= 0 && g.clusterHeight[ c ] < height ) c = g.clusterParent[ c ];
					cl.fineParent[ j ] = c - cl.offset;

				}

				cl.acc = new Float64Array( 2 * cl.count );
				cl.fineStats = null;

			}

		}

		if ( cl.K === 0 ) return null;
		if ( namesOnly ) return cl;

		// lay every name out once: the name, then the count in the second style
		this._ensureAtlas();
		const atlas = this.atlas, S = atlas.size;
		const spacing = { kerning: this._kerning, letterSpacing: o.clusterLetterSpacing * S };
		const records = [];
		for ( let j = 0; j < cl.K; j ++ ) {

			const slot = cl.slots[ j ];
			const a = layoutUnits( segmentText( slot.name, { maxUnits: o.clusterMaxChars + 1 } ), atlas, spacing );
			const b = layoutUnits( segmentText( slot.tail ), atlas, spacing );
			const width = a.width + ( a.width > 0 ? spacing.letterSpacing : 0 ) + b.width;
			slot.widthEm = width / S;
			const half = width / 2;
			for ( let k = 0; k < a.glyphs.length; k ++ ) records.push( [ j, a.x[ k ] - half, a.glyphs[ k ], 0 ] );
			for ( let k = 0; k < b.glyphs.length; k ++ ) records.push( [ j, width - b.width + b.x[ k ] - half, b.glyphs[ k ], slot.name ? 1 : 0 ] );

		}

		const G = records.length;
		cl.G = G;
		cl.glyphNode = instancedArray( 2 * Math.max( 1, G ), 'vec4' ).setName( 'clusterLabelGlyphs' );
		const A = cl.glyphNode.value.array;
		for ( let r = 0; r < G; r ++ ) {

			const [ j, x, glyph, style ] = records[ r ];
			// A = ( label, x of the cell's left edge from the CENTRE of the line, top above the baseline, style ), em
			A.set( [ j, ( x + glyph.left ) / S, glyph.top / S, style, glyph.x, glyph.y, glyph.w, glyph.h ], 8 * r );

		}

		// two records per name: ( alpha0, alpha1, t0, font scale ), ( cluster index, rgb of the community )
		cl.stateNode = instancedArray( 2 * cl.K, 'vec4' ).setName( 'clusterLabelState' );
		const T = cl.S = cl.stateNode.value.array;
		for ( let j = 0; j < cl.K; j ++ ) {

			cl.slots[ j ].shown = cl.slots[ j ].scale; // the scale on screen: limited on narrow views (nameScale())
			T.set( [ 0, 0, - 1e3, cl.slots[ j ].scale, cl.slots[ j ].cluster, 1, 1, 1 ], 8 * j );

		}

		// one record per name: ( world radius of the community, –, –, – ): the lift and the size of the
		// name are projected from it in the vertex stage, so they follow the camera every frame
		cl.geomNode = instancedArray( Math.max( 1, cl.K ), 'vec4' ).setName( 'clusterLabelGeom' );
		cl.geom = cl.geomNode.value.array;
		cl.geomDirty = false;

		cl.built = true;
		cl.colorVersion = - 1;
		this._syncClusterColors();
		this.uniforms.clusterGlyphCount.value = G;
		this.clusterMesh.count = 2 * G;
		this.stats.clusterGlyphs = G;
		this._rebuildClusterMaterial();
		return cl;

	}

	/**
	 * Screen position, cross-fade and weight of every community of the level for this decision,
	 * from the readback (the GPU draws the names at the live centroid; the two differ by what the
	 * layout moved in a frame or two, which the collision boxes tolerate).
	 */
	_measureClusters( cl, pos, m, cam, W, H, P11, fontPx = this.fontSize ) {

		const g = this.graph, BU = this.buffers.uniforms;
		const s = cl.stats = clusterStats( g, pos, cl.height, cl.stats );
		const [ lo, hi ] = this.options.clusterFade;

		// no name is wider than a share of the view: its font shrinks (nameScale(); a resize, rarely more)
		const size = fontPx * this.uniforms.clusterScale.value, share = this.options.clusterMaxWidth;
		for ( let j = 0; j < cl.K; j ++ ) {

			const slot = cl.slots[ j ];
			const scale = share > 0 ? nameScale( slot.scale, slot.widthEm * size, W, { share } ) : slot.scale;
			if ( scale === slot.shown ) continue;
			slot.shown = scale;
			cl.S[ 8 * j + 3 ] = scale;
			cl.stateDirty = true;

		}

		// world radius of every named community for the vertex stage (steps below 2 % are not uploaded)
		for ( let j = 0; j < cl.K; j ++ ) {

			const R = 1.29 * s.rms[ cl.slots[ j ].k ], was = cl.geom[ 4 * j ];
			if ( Math.abs( R - was ) <= 0.02 * was ) continue;
			cl.geom[ 4 * j ] = R;
			cl.geomDirty = true;

		}

		// projected radius of the whole graph (for the names that do not fit their community)
		let gx = 0, gy = 0, gz = 0;
		for ( let k = 0; k < cl.count; k ++ ) {

			const n = s.size[ k ];
			gx += n * s.center[ 3 * k ]; gy += n * s.center[ 3 * k + 1 ]; gz += n * s.center[ 3 * k + 2 ];

		}

		const total = Math.max( 1, s.total );
		const gw = ( m[ 3 ] * gx + m[ 7 ] * gy + m[ 11 ] * gz ) / total + m[ 15 ];
		cl.graphPx = gw > cam.near ? 1.29 * s.graphRms * P11 * H * 0.5 / gw : Infinity; // the camera is inside or past it: no limit
		cl.looseRoom = looseNameArea( cl.graphPx );
		cl.loose = cl.looseArea = cl.placed = 0;
		cl.looseStop = false;

		const sphere = BU.layoutBlend ? Math.min( 1, Math.max( 0, BU.layoutBlend.value ) ) : 0;
		const gc = BU.graphCenter ? BU.graphCenter.value : null, sr = BU.sphereRadius ? BU.sphereRadius.value : 0;
		// member spacing in world units: the community's own or that of the finest communities inside
		// it, whichever is smaller (communitySpacing())
		const sp = cl.spacing, acc = cl.acc;
		if ( cl.fineParent ) {

			const f = cl.fineStats = clusterStats( g, pos, 1, cl.fineStats );
			acc.fill( 0 );
			for ( let j = 0; j < f.count; j ++ ) {

				const n = f.size[ j ], k = cl.fineParent[ j ];
				if ( n < 3 || k < 0 ) continue;
				acc[ 2 * k ] += Math.sqrt( n ) * f.rms[ j ]; // n · rms / √n
				acc[ 2 * k + 1 ] += n;

			}

		}

		for ( let k = 0; k < cl.count; k ++ ) sp[ k ] = communitySpacing( s.rms[ k ], s.size[ k ], acc ? acc[ 2 * k ] : 0, acc ? acc[ 2 * k + 1 ] : 0 );

		// communities of one or two nodes have no size of their own: they take the spacing of the
		// typical (median) community, so their nodes get labels when their surroundings do
		const known = Array.from( sp ).filter( ( v ) => v > 0 ).sort( ( a, b ) => a - b );
		const typical = known.length > 0 ? known[ known.length >> 1 ] : s.graphRms / Math.sqrt( Math.max( 1, s.total ) );
		let resolved = false;
		for ( let k = 0; k < cl.count; k ++ ) {

			let x = s.center[ 3 * k ], y = s.center[ 3 * k + 1 ], z = s.center[ 3 * k + 2 ];
			if ( sphere > 0 && gc ) {

				// the centroid of a cap lies inside the sphere: the name goes out onto it, like the glow
				const dx = x - gc.x, dy = y - gc.y, dz = z - gc.z, d = Math.max( 1e-4, Math.hypot( dx, dy, dz ) );
				x += ( gc.x + dx / d * sr - x ) * sphere; y += ( gc.y + dy / d * sr - y ) * sphere; z += ( gc.z + dz / d * sr - z ) * sphere;

			}

			const cw = m[ 3 ] * x + m[ 7 ] * y + m[ 11 ] * z + m[ 15 ];
			if ( ! ( cw > cam.near ) || s.size[ k ] === 0 ) {

				cl.weight[ k ] = 0;
				cl.fade[ k ] = cl.name[ k ] = cl.nodes[ k ] = cl.radius[ k ] = cl.world[ k ] = 0;
				continue;

			}

			cl.sx[ k ] = ( ( m[ 0 ] * x + m[ 4 ] * y + m[ 8 ] * z + m[ 12 ] ) / cw * 0.5 + 0.5 ) * W;
			cl.sy[ k ] = ( 0.5 - ( m[ 1 ] * x + m[ 5 ] * y + m[ 9 ] * z + m[ 13 ] ) / cw * 0.5 ) * H;
			cl.radius[ k ] = 1.29 * s.rms[ k ] * P11 * H * 0.5 / cw;
			cl.world[ k ] = P11 * H * 0.5 / cw; // CSS px per world unit at the centroid
			// projected spacing = 1.29 · ( rms / √n ) · √π · px per unit (a uniform ball of radius r has an rms radius of 0.775 r)
			const rPx = 1.29 * ( sp[ k ] > 0 ? sp[ k ] : typical ) * P11 * H * 0.5 / cw;
			const t = Math.round( communityFade( rPx, 1, { lo, hi } ) * 16 ) / 16;
			const w = communityWeights( t );
			cl.fade[ k ] = t;
			cl.name[ k ] = w.name;
			cl.nodes[ k ] = Math.round( w.nodes * 16 ) / 16;
			cl.weight[ k ] = compactness( s.rms[ k ], s.size[ k ], s.graphRms, s.total );
			if ( cl.nodes[ k ] >= 0.06 ) resolved = true;

		}

		cl.resolved = resolved;

	}

	/**
	 * Places the community names of one priority class: `dominant` = the names of communities in
	 * the first half of their cross-fade (t ≤ ½). Returns how many were placed.
	 */
	_placeClusters( cl, now, grid, fontPx, hyst, W, H, dominant ) {

		const gen = this._gen, always = this._clusterLabels === 'on';
		const size = fontPx * this.uniforms.clusterScale.value;
		const fit = always ? 0 : this.options.clusterFit, air = this.options.clusterSpacing;
		const most = always ? Infinity : this.options.maxNamesInView ?? nameBudget( W, H );
		const U = this.uniforms, [ r0, r1 ] = this.options.clusterMinRadius;
		const nameK = U.clusterNameSize.value, nameMin = U.clusterNameMin.value, pad = U.clusterPad.value, liftMax = U.clusterLiftMax.value * H;
		let placed = 0;
		for ( let j = 0; j < cl.K && cl.placed < most; j ++ ) {

			const slot = cl.slots[ j ], k = slot.k;
			if ( slot.gen === gen ) continue;
			// the vertex stage works with the world radius of the last upload: the same value here
			const rPx = cl.built ? cl.geom[ 4 * j ] * cl.world[ k ] : cl.radius[ k ];
			// a community that is a dot on screen is not named (its name would be larger than itself)
			const big = always || ! ( r1 > r0 ) ? 1 : smooth( ( rPx - r0 ) / ( r1 - r0 ) );
			const alpha = Math.round( ( always ? 1 : cl.name[ k ] ) * cl.weight[ k ] * big * 16 ) / 16;
			if ( alpha < 0.06 || ( always || cl.fade[ k ] <= 0.5 ) !== dominant ) continue;
			// size and place as the vertex stage computes them (_buildClusterMaterial)
			const F = Math.min( size * slot.shown, Math.max( nameMin, nameK * rPx ) );
			const width = slot.widthEm * F;
			const hw = width * 0.5 + 3, hh = F * 0.66;
			// … but not out of the view's top edge while the centroid is inside it
			const sx = cl.sx[ k ], sy = Math.max( cl.sy[ k ] - ( Math.min( rPx, liftMax ) + pad + 0.5 * F ), Math.min( cl.sy[ k ], 1.1 * F ) );
			if ( sx + hw < 0 || sx - hw > W || sy + hh < 0 || sy - hh > H ) continue;
			const was = cl.S[ 8 * j + 1 ] > 0;
			// a name sits on what it names; of those that are wider than their community only the
			// largest few are placed (the slots are in order of size). A shown name keeps its place
			// a little longer than a hidden one needs to get it (no flicker at the threshold)
			const loose = fit > 0 && ! nameFits( width, cl.radius[ k ], was ? fit * 1.1 : fit );
			if ( loose && ( cl.looseStop || cl.loose >= this.options.maxLooseNames || cl.looseArea + width * 2 * hh > cl.looseRoom ) ) {

				cl.looseStop = true; // the following names belong to smaller communities
				continue;

			}

			const h = ( was ? - hyst : hyst ) + air * F;
			if ( ! grid.free( sx - hw - h, sy - hh - h * 0.5, sx + hw + h, sy + hh + h * 0.5 ) ) continue;
			grid.insert( sx - hw, sy - hh, sx + hw, sy + hh );
			slot.box = [ sx - hw, sy - hh, sx + hw, sy + hh, F ];
			if ( loose ) {

				cl.loose ++;
				cl.looseArea += width * 2 * hh;

			}

			slot.gen = gen;
			this._setClusterTarget( cl, j, alpha, now );
			placed ++;
			cl.placed ++;

		}

		return placed;

	}

	/** Hierarchy height of the app's community field for this graph (0: there is none, or it has no level). */
	_fieldHeight() {

		const app = this.app, modules = app && app.modules;
		let field = this._field;
		if ( ! field || ! modules || ! modules.includes( field ) ) field = this._field = ( modules && modules.find( ( m ) => m && m.isClusterField ) ) || null;
		return field && field.graph === this.graph && field.height > 0 ? field.height : 0;

	}

	/** Names that were not placed by the decision in progress fade out. */
	_retireClusters( now ) {

		const cl = this._cl;
		if ( ! cl || ! cl.built ) return;
		for ( let j = 0; j < cl.K; j ++ ) if ( cl.slots[ j ].gen !== this._gen ) this._setClusterTarget( cl, j, 0, now );

	}

	_setClusterTarget( cl, j, alpha, now ) {

		const T = cl.S, o = 8 * j;
		if ( T[ o + 1 ] === alpha ) return;
		T[ o ] = T[ o ] + ( T[ o + 1 ] - T[ o ] ) * smooth( ( now - T[ o + 2 ] ) / this.uniforms.fade.value );
		T[ o + 1 ] = alpha;
		T[ o + 2 ] = now;
		cl.stateDirty = true;

	}

	/**
	 * Where the last decision placed the name of community `cluster` (an index of the active level):
	 * { x0, y0, x1, y1 } in CSS px (y down) and its font size, or null while it is not shown.
	 */
	clusterBox( cluster ) {

		const cl = this._cl;
		const j = cl && cl.built ? cl.slots.findIndex( ( s ) => s.cluster === cluster ) : - 1;
		if ( j < 0 || ! ( cl.S[ 8 * j + 1 ] > 0 ) || ! cl.slots[ j ].box ) return null;
		const [ x0, y0, x1, y1, font ] = cl.slots[ j ].box;
		return { x0, y0, x1, y1, font };

	}

	/** Current eased alpha of the name of slot j (what the shader computes, before the fades). */
	clusterAlphaOf( j, now = performance.now() / 1000 - this._time0 ) {

		const cl = this._cl;
		if ( ! cl || ! cl.built || ! ( j >= 0 && j < cl.K ) ) return 0;
		const T = cl.S, o = 8 * j;
		return T[ o ] + ( T[ o + 1 ] - T[ o ] ) * smooth( ( now - T[ o + 2 ] ) / this.uniforms.fade.value );

	}

	/** Tint of every name: the mean colour of the community's members (palette or colour lens). */
	_syncClusterColors() {

		const cl = this._cl;
		if ( ! cl || ! cl.built ) return;
		const target = this.buffers.nodeColorTarget;
		const colors = target && target.value;
		if ( ! colors || cl.colorVersion === colors.version ) return;
		cl.colorVersion = colors.version;
		const mean = cl.mean = clusterColors( this.graph, colors.array, cl.height, cl.mean );
		const T = cl.S;
		for ( let j = 0; j < cl.K; j ++ ) {

			const k = cl.slots[ j ].k;
			const r = mean[ 3 * k ], g = mean[ 3 * k + 1 ], b = mean[ 3 * k + 2 ];
			const top = Math.max( r, g, b );
			if ( top > 1e-4 ) T.set( [ r / top, g / top, b / top ], 8 * j + 5 );
			else T.set( [ 1, 1, 1 ], 8 * j + 5 );

		}

		cl.stateDirty = true;

	}

	_disposeClusters() {

		const cl = this._cl;
		this._cl = null;
		const m = this.clusterMesh.material;
		if ( m && m !== this._placeholder ) {

			this.clusterMesh.material = this._placeholder;
			m.dispose();

		}

		this.clusterMesh.visible = false;
		this.clusterMesh.count = 0;
		this.uniforms.clusterGlyphCount.value = 0;
		this.stats.clusters = 0;
		this.stats.clusterGlyphs = 0;
		if ( cl && cl.built ) {

			cl.glyphNode.value.dispose();
			cl.stateNode.value.dispose();
			cl.geomNode.value.dispose();

		}

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

			// a readback can deliver unchanged positions ( a sleeping layout ): compare the content
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
		const cl = this._cl;
		if ( cl && cl.built ) {

			for ( let o = 2; o < cl.S.length; o += 8 ) cl.S[ o ] = Math.max( - 1e3, cl.S[ o ] - shift );
			cl.stateDirty = true;

		}

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

		const cl = this._cl;
		if ( cl && cl.built ) {

			if ( cl.stateDirty ) {

				cl.stateNode.value.needsUpdate = true;
				cl.stateDirty = false;
				this.stats.uploads ++;

			}

			if ( cl.geomDirty ) {

				cl.geomNode.value.needsUpdate = true;
				cl.geomDirty = false;
				this.stats.uploads ++;

			}

			// names that show or are still fading out
			const T = cl.S, now = this.uniforms.time.value, fade = this.uniforms.fade.value;
			let live = 0;
			for ( let o = 0; o < T.length; o += 8 ) if ( T[ o + 1 ] > 0 || now - T[ o + 2 ] <= fade ) live ++;
			cl.live = live;

		}

		this._syncAtlas();
		this.uniforms.glyphCount.value = this._G;
		this.mesh.count = 2 * this._G;
		this._syncVisibility();
		this.stats.glyphs = this._G;
		this.stats.capacity = this._capacity;

	}

	/** The node labels' mesh is the parent of the community names: it is visible while either draws. */
	_syncVisibility() {

		const cl = this._cl;
		const names = this._visible && !! cl && cl.built && cl.live > 0 && cl.G > 0 && this.clusterMesh.material !== this._placeholder;
		this.clusterMesh.visible = names;
		this.mesh.visible = this._visible && ( this._G > 0 || names ) && this.mesh.material !== this._placeholder;

	}

	// ── material ────────────────────────────────────────────────────────────────────────────

	_rebuildMaterial() {

		if ( ! this.buffers || ! this._glyphNode || ! this.atlas ) return;
		const old = this.mesh.material;
		this.mesh.material = this._buildMaterial();
		if ( old && old !== this._placeholder ) old.dispose();
		this._rebuildClusterMaterial();

	}

	_rebuildClusterMaterial() {

		const cl = this._cl;
		if ( ! cl || ! cl.built || ! this.buffers || ! this.atlas ) return;
		const old = this.clusterMesh.material;
		this.clusterMesh.material = this._buildClusterMaterial( cl );
		if ( old && old !== this._placeholder ) old.dispose();

	}

	_buildMaterial() {

		const U = this.uniforms, buffers = this.buffers;
		const glyphs = this._glyphNode, labelState = this._stateNode;
		const focus = buffers.uniforms.focus;
		const atlas = this.atlas;
		const S = atlas.size, BUF = atlas.buffer;
		const REACH = atlas.edge * atlas.radius; // texels of field outside the outline (= BUF for the defaults)
		const srgb = this._output === 'srgb';
		const fromState = this.options.emphasis !== 'class';
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

		material.colorNode = this._inkNode( { vUv, vCell, vColor, vParams, vDepth }, depthNode );
		return material;

	}

	/**
	 * Fragment stage shared by the node labels and the community names: SDF ink or halo of one
	 * glyph quad. Varyings: vUv (atlas texels), vCell (cell rect), vColor (encoded rgb, alpha),
	 * vParams ( grow px, layer 0 halo / 1 fill, target px per CSS px, on-top ), vDepth ( label
	 * distance, softness, halo density 0..1 ). depthNode: scene depth for the soft occlusion or null.
	 */
	_inkNode( { vUv, vCell, vColor, vParams, vDepth }, depthNode ) {

		const U = this.uniforms, atlasTex = this._atlasNode;
		const atlas = this.atlas;
		const EDGE = atlas.edge, RANGE = atlas.radius;
		const REACH = EDGE * RANGE; // texels of field outside the outline
		const shadow = !! this.options.shadow;

		return Fn( () => {

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

	}

	/** The community names: the node labels' vertex stage without a node (centred text above clusterPos). */
	_buildClusterMaterial( cl ) {

		const U = this.uniforms, buffers = this.buffers, BU = buffers.uniforms;
		const glyphs = cl.glyphNode, records = cl.stateNode, geom = cl.geomNode;
		const atlas = this.atlas;
		const S = atlas.size, BUF = atlas.buffer, REACH = atlas.edge * atlas.radius;
		const srgb = this._output === 'srgb';

		const vUv = varyingProperty( 'vec2', 'vNameUv' );
		const vCell = varyingProperty( 'vec4', 'vNameCell' );
		const vColor = varyingProperty( 'vec4', 'vNameColor' );
		const vParams = varyingProperty( 'vec4', 'vNameParams' );
		const vDepth = varyingProperty( 'vec3', 'vNameDepth' );

		const material = new THREE.NodeMaterial();
		material.name = 'TextLabels.communities';
		material.lights = false;
		material.fog = false;
		material.transparent = true;
		material.depthWrite = false;
		material.depthTest = this._depthTest;
		material.premultipliedAlpha = true;
		material.side = THREE.DoubleSide;

		const encode = ( c ) => ( srgb ? sRGBTransferOETF( c ) : c.mul( U.gain ) );

		material.vertexNode = Fn( () => {

			const G = U.clusterGlyphCount;
			const isFill = instanceIndex.greaterThanEqual( G );
			const gi = instanceIndex.sub( select( isFill, G, uint( 0 ) ) ).toVar( 'nGlyph' );
			const gA = glyphs.element( gi.mul( uint( 2 ) ) ).toVar( 'nGA' );
			const gB = glyphs.element( gi.mul( uint( 2 ) ).add( uint( 1 ) ) ).toVar( 'nGB' );
			const slot = uint( gA.x.add( 0.5 ) ).toVar( 'nSlot' );
			const lab = records.element( slot.mul( uint( 2 ) ) ).toVar( 'nState' ); // alpha0, alpha1, t0, font scale
			const info = records.element( slot.mul( uint( 2 ) ).add( uint( 1 ) ) ).toVar( 'nInfo' ); // cluster, rgb
			const layer = select( isFill, float( 1 ), float( 0 ) );
			const count = gA.w; // 1: the member count (lighter)

			// the live centroid; in the sphere layout the communities lie on the sphere
			const cp = buffers.clusterPos.element( uint( info.x.add( 0.5 ) ) ).xyz.toVar( 'nCentroid' );
			const rel = cp.sub( BU.graphCenter ).toVar( 'nRel' );
			const onSphere = BU.graphCenter.add( rel.div( max( length( rel ), 1e-4 ) ).mul( BU.sphereRadius ) );
			const world = mix( cp, onSphere, clamp( BU.layoutBlend, 0, 1 ) );

			const k = clamp( U.time.sub( lab.z ).div( U.fade ), 0, 1 );
			const present = mix( lab.x, lab.y, k.mul( k ).mul( k.mul( - 2 ).add( 3 ) ) );
			const center = cameraViewMatrix.mul( vec4( world, 1 ) ).xyz.toVar( 'nCenter' );
			const dist = center.z.negate();
			const depthT = clamp( dist.sub( U.depthRange.x ).mul( U.depthRange.y ), 0, 1 );
			const alpha = present
				.mul( float( 1 ).sub( U.clusterDepthFade.mul( depthT ) ) )
				.mul( float( 1 ).sub( U.clusterFocusFade.mul( BU.focus ) ) ).toVar( 'nAlpha' );

			const css = screenSize.y.div( U.cssHeight ).toVar( 'nCss' );
			const clipC = cameraProjectionMatrix.mul( vec4( center, 1 ) ).toVar( 'nClip' );
			// the community's ball on screen (CSS px): the name is sized by it and sits above it
			// (the CPU twin of both is in _placeClusters)
			const rCss = geom.element( slot ).x.mul( abs( cameraProjectionMatrix.element( 1 ).element( 1 ) ) ).mul( U.cssHeight ).mul( 0.5 ).div( max( clipC.w, 1e-4 ) ).toVar( 'nBall' );
			const Fcss = min( U.fontSize.mul( U.clusterScale ).mul( lab.w ), max( U.clusterNameMin, rCss.mul( U.clusterNameSize ) ) ).toVar( 'nFcss' );
			const F = Fcss.mul( css ).toVar( 'nF' ); // target px per em
			const xh = U.xHeight.mul( F );
			const hintY = mix( float( 1 ), max( floor( xh.add( 0.5 ) ), 1 ).div( xh ), U.hint.mul( smoothstep( 24, 40, F ).oneMinus() ) );

			const sp = clipC.xy.div( clipC.w ).mul( 0.5 ).add( 0.5 ).mul( screenSize );
			const lift = min( rCss, U.cssHeight.mul( U.clusterLiftMax ) ).add( U.clusterPad ).add( Fcss.mul( 0.5 ) ).mul( css );
			// (kept inside the top edge of the view while the centroid is)
			const oy = min( sp.y.add( lift ), max( sp.y, screenSize.y.sub( F.mul( 1.1 ) ) ) ).sub( U.baseline.mul( F ) );
			const origin = vec2( mix( sp.x, floor( sp.x.add( 0.5 ) ), U.snap ), mix( oy, floor( oy.add( 0.5 ) ), U.snap ) );

			const tpp = float( S ).div( F ).toVar( 'nTpp' );
			const reachPx = float( REACH ).div( tpp );
			const growFill = U.weightPx.add( F.mul( U.weight.add( U.clusterWeight.mul( float( 1 ).sub( count.mul( 2 ) ) ) ) ) );
			const growHalo = min( U.outline.mul( 1.2 ).mul( css ), max( reachPx.sub( 1.25 ), 0 ) );
			const grow = select( isFill, growFill, growHalo ).toVar( 'nGrow' );
			const inset = select( isFill, clamp( float( BUF ).sub( growFill.add( 1 ).mul( tpp ).add( 1 ) ), 0, BUF ), float( 0 ) );
			const t = positionGeometry.xy.mul( gB.zw.sub( inset.mul( 2 ) ) ).add( inset ).toVar( 'nT' );
			const em = vec2( gA.y.add( t.x.div( S ) ), gA.z.sub( t.y.div( S ) ) );
			const P = origin.add( em.mul( vec2( F, F.mul( hintY ) ) ) );
			const ndc = P.div( screenSize ).mul( 2 ).sub( 1 );

			// a name belongs to a region, not to a depth: always in front
			const zNear = cameraNear.negate().mul( 1.001 );
			const cz = cameraProjectionMatrix.mul( vec4( 0, 0, zNear, 1 ) );

			const fillC = mix( U.clusterColor, info.yzw, U.clusterTint );
			const fillA = U.clusterOpacity.mul( mix( float( 1 ), U.clusterCount, count ) );

			vUv.assign( gB.xy.add( t ) );
			vCell.assign( vec4( gB.xy, gB.xy.add( gB.zw ) ) );
			vColor.assign( vec4( encode( select( isFill, fillC, U.haloColor ) ), alpha.mul( select( isFill, fillA, mix( float( 1 ), float( 0.8 ), count ) ) ) ) );
			vParams.assign( vec4( grow, layer, css, 1 ) );
			vDepth.assign( vec3( dist, 1, 0.5 ) );

			const visible = alpha.greaterThan( 1 / 255 ).and( clipC.w.greaterThan( cameraNear ) ).and( gB.z.greaterThan( 0 ) );
			return select( visible, vec4( ndc, cz.z.div( cz.w ), 1 ), vec4( 0, 0, 2, 1 ) );

		} )();

		material.colorNode = this._inkNode( { vUv, vCell, vColor, vParams, vDepth }, null );
		return material;

	}

	_disposeGraph() {

		const m = this.mesh.material;
		if ( m && m !== this._placeholder ) {

			this.mesh.material = this._placeholder;
			m.dispose();

		}

		this._disposeClusters();
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
