// Bundled edge renderer, version 2: the signature look of ImsvGraphVis (Kwon et al., TVCG 2016) —
// every edge a bundled cubic B-spline, coloured from the source to the target node hue — drawn
// as luminous silk strands, with solid lit tubes for the edges that are opaque anyway.
//
// ─── Curves ─────────────────────────────────────────────────────────────────────────────────
//  The curve of every edge lives in a GPU cache (src/gpu/EdgeCurves.js, `edges.curves`): one
//  compute kernel evaluates explicit control paths (hierarchical bundling by default, any path set
//  of src/graph/bundling.js through curves.setPaths) into S + 1 samples per edge, and only when
//  positions, the bundling strength, the path set or the sphere-layout parameters changed. The
//  draws below only read samples: nothing per frame is O(E) on the CPU, and a graph at rest costs
//  no curve evaluation. S comes from autoSegments( E, longest path, sphere scale ), capped by the
//  cache's memory budget (`memoryBudget`: 64 MiB; 384k edges: S = 8, 55 MB; 24 MiB on touch
//  devices, where three's CPU copy of the buffer, as large again, is the largest single allocation
//  of the page and a memory spike closes the tab: S = 3 there for 384k edges). Small graphs
//  reserve the room for the sphere layout's finer sampling up front (below 8 MiB), so the first
//  morph rebuilds nothing.
//
// ─── Strand layer (every edge, the default look) ────────────────────────────────────────────
//  ONE instanced draw: a ribbon strip per edge ( attribute position = ( vertex index, side ±1,
//  0 ) ), expanded in screen space perpendicular to the projected tangent. The strip is finer than
//  the cache: `strandSub` vertex pairs per cached segment (about STRAND_SEGMENTS per edge, at most
//  4 per segment, fewer when the draw would pass strandVertexLimit; 1 for very large graphs), put
//  on the Catmull-Rom spline through the samples in the vertex stage, the same curve the tubes
//  follow. A close-up therefore shows curves, not chords, at any cache resolution, and the cache
//  (memory, compute) stays as small as it was. Alpha-blended (normal or additive), no depth write,
//  forceSinglePass. Vertex-stage storage reads: curve samples 5 (the four spline points and the
//  last sample for the curve length), edgeInfo 1, edgeAttr 1, nodeColor 2, nodeState 2 = 5
//  bindings (7 with strandVelocity 'true': + nodePos, nodePrev), none in the fragment stage.
//   · Width: world half width (scales with graph.layoutRadius) × the per-edge factors, clamped to
//     [minPixelWidth, maxPixelWidth] CSS px × gfxUniforms.pxScale (highlighted and hovered strands
//     up to 1.25 × that). Strands are silk: near the camera they stay thin lines (8 px at most)
//     instead of growing into flat tapes; what should have body there is a tube. Below 1 pass
//     pixel the ribbon keeps 1 px and its alpha is scaled by the coverage ratio; the strip is
//     padded by 0.5 px per side for the analytic edge profile (1 px box filter) in the fragment stage.
//   · Colour: OKLab gradient source → target (cube-root LMS blend) with the chroma restored: a
//     straight line through OKLab between two hues far apart passes near the grey axis, so the
//     blend is pushed back out to the mix of the two end chromas (gradientColor() in
//     gfx/strandSurface.js; hue and lightness of the blend are kept). HDR on purpose (bloom).
//   · Light (gfxState.lit): the analytic thin-strand model of gfx/strandSurface.js per fragment —
//     Lambert averaged over the visible half cylinder and a Kajiya-Kay highlight around the
//     tangent from gfxUniforms.keyDir / keyColor, hemisphere ambient — on top of an emissive
//     base, with a per-strand shift of the highlight: bundles read as silk. The highlight takes
//     the strand's colour (look.sheenTint), which keeps dense bundles saturated. No texture taps.
//     gfxState.lit === false: the flat shading of the first renderer (same picture as before).
//   · Fog: fogTransmittance( view depth ); normal blending fades toward gfxUniforms.fogColor,
//     additive fades out.
//   · Depth of field: the ribbon is widened by cocPixels( view depth ), its alpha scaled by the
//     width ratio and its cross profile blended to a bell, so a defocused strand is blurred, not
//     fat; it also grows past its two end points by the circle of confusion and fades there.
//     dofAmount = 0 leaves the picture unchanged. The circle is capped at `defocusLimit` CSS px
//     (auto: 10 px up to 5k edges, falling with log E to 0 at 200k), because every pixel of
//     width is blended fill and dense graphs already overdraw the picture many times.
//   · Emphasis (eased nodeState, as before): edges of the selected node bright and wide, edges
//     of the hovered node boosted, the rest dimmed to contextOpacity with uniforms.focus. A
//     highlighted set (a path) dims the rest the same way, by look.setContext of that amount.
//   · Motion vectors (gfxState.velocity): 'camera' writes the camera motion of the curve point,
//     'true' adds the end points' motion interpolated along the edge (nodePrev), 'skip' leaves
//     the attachment alone (edges.strandVelocity).
//
// ─── Tube layer (edges that are opaque anyway) ──────────────────────────────────────────────
//  A second instanced draw over an index list ( instance → edge id ): the edges of the selected
//  node, the highlighted set (the 'path' event, setHighlightedEdges) and, with style 'solid',
//  all edges. Opaque, depth-writing ribbons shaded as cylinders: per-fragment tube frame + depth
//  and applySurface() into three's MeshStandardNodeMaterial (key light, environment, shadows
//  received; cast while the tube count is at most tubeShadowLimit), emissive in the gradient
//  colours so they glow into the bloom. Variants follow gfxState: alpha-to-coverage with
//  samples > 1, a hard cut otherwise; velocityOutput when velocity; an analytic Blinn model
//  without three's lights when lit is false. The hand-over is eased: the tube radius grows with
//  the eased highlight while the strand of the same edge fades, and shrinks again before the
//  edge leaves the list (it lingers there for `tubeLinger` seconds). Vertex-stage storage
//  bindings: tube index, curve samples, edgeInfo, edgeAttr, nodeColor, nodeState = 6.
//   · Ends: a tube tapers to look.tubeTaper of its radius between three node radii from an end
//     point and that node's surface (arc length from the cache, node radius = the node renderer's
//     base size × the node's radius factor), and its emission falls with it: the many tubes of a
//     hub arrive as fine lines and the node, its ring and its label stay readable. The emission
//     of highlighted edges is also scaled by 1 / √( edges of the selection / 8 ) (at least
//     look.crowdFloor), so a hub's bundle does not add up to a white blob in the bloom.
//   · Width: world radius, at least tubeMinPixelWidth and at most tubeMaxPixelWidth CSS px.
//   · Smooth: the tube strip is finer than the cache (about 96 segments per tube: 2 to 8 vertex
//     pairs per cached segment, fewer when the draw would pass tubeVertexLimit) and interpolates
//     the samples with a Catmull-Rom spline in the vertex stage (4 sample reads, C1). Strands
//     follow the same spline (see above), so tube and sheath coincide; only where the strand
//     layer cannot afford the subdivision (strandSub = 1, very large graphs) the strand is a chain
//     of chords, and its sheath fades where it leaves the tube by about a tube radius.
//   · Shadows on demand: the shadow maps of the scene's lights (shadow.autoUpdate = false) are
//     asked for a new frame whenever the tube list changes and in every frame while tubes ease
//     in or out.
//   · Nothing is built at a click: the index list is allocated once per graph for the most tubes
//     there can be, and the tube mesh stays in the scene with zero instances, so the renderer
//     compiles its pipelines (main and shadow pass) together with the rest of the scene.
//
// ─── Flow ───────────────────────────────────────────────────────────────────────────────────
//  Energy pulses travel from source to target at an even speed (the cache stores the arc
//  length): edges.flow = 'off' | 'selection' (highlighted edges) | 'all'.
//
// ─── Per-edge attributes (buffers.edgeAttr, vec4) ───────────────────────────────────────────
//  x width factor (setEdgeWidths), y highlight weight (setHighlightedEdges / 'path'), z normalized
//  metric value (setEdgeMetric), w random seed.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//  new EdgeRenderer( { beta, opacity, width, minPixelWidth, maxPixelWidth, blendMode, segments,
//                      colorMode, visible, contextOpacity, highlightGlow, highlightLevel,
//                      hoverBoost, brightness, sphereSegmentScale, sphereAxis, renderOrder,
//                      style, flow, flowSpeed, flowSpacing, tubes, tubeLimit, tubeShadowLimit,
//                      solidLimit, tubeLinger, tubeVertexLimit, strandVertexLimit, defocusLimit,
//                      strandVelocity, memoryBudget (bytes of the curve cache; null = auto, see
//                      "Curves"), look } )
//  App module (order 10): init( app ), setGraph( buffers, graph, app ), update( dt, app ), dispose()
//  setBundling( beta ) / .bundling   Holten bundling strength in [0,1] (default 0.85; paper 0.9)
//  .opacity                       multiplier on the automatic opacity (default 1)
//  .width                         multiplier on the world-space width (default 1)
//  .minPixelWidth / .maxPixelWidth   full strand width clamp, CSS pixels (defaults 0.75 / 8)
//  .blendMode                     'normal' | 'additive' (glow; good for dense graphs)
//  .segments                      null = auto from E, or an integer >= 1 (clamped to the budget)
//  .sphereSegmentScale            auto segment multiplier while layoutBlend > 0 (default 2)
//  .colorMode                     'gradient' | 'source'
//  .visible
//  .contextOpacity / .highlightGlow / .highlightLevel / .hoverBoost / .brightness
//  .sphereAxis                    THREE.Vector3, direction the sphere layout faces (default +Z)
//  .style                         'luminous' (strands; default) | 'solid' (every edge a tube, for
//                                 graphs up to solidLimit edges) | 'auto' (= luminous)
//  .tubes                         true (default): highlighted edges become tubes; false: they stay
//                                 bright, wide strands as in the first renderer
//  .tubeLimit / .tubeShadowLimit / .solidLimit / .tubeLinger
//  .tubeVertexLimit               most vertices of the tube draw (default 2 M): bounds the tubes'
//                                 subdivision, never their number
//  .strandVertexLimit             most vertices of the strand draw that subdivision may lead to
//                                 (default 3 M; the cache resolution itself is never reduced by it)
//  .animating                     true while the flow runs (the App keeps its full frame rate)
//  .defocusLimit                  most strand defocus, CSS px of circle of confusion; null = auto
//  .flow / .flowSpeed / .flowSpacing   'off' | 'selection' | 'all'; fractions of layoutRadius per
//                                 second and between pulses (defaults 'off', 0.3, 0.5)
//  .strandVelocity                'camera' (default) | 'true' | 'skip'
//  setHighlightedEdges( ids | null, weight = 1 )   edge ids (array / typed array / Set) shown as
//                                 highlighted tubes; cross-faded when replaced
//  setEdgeMetric( values | null, { opacity: [ at 0, at 1 ], width: [ at 0, at 1 ] } )
//                                 normalized per-edge values in [0,1] scale opacity and width
//  setEdgeWidths( values | null ) per-edge width factors (1 = unchanged)
//  setPaths( paths | null )       = curves.setPaths (control paths of src/graph/bundling.js)
//  setLook( { … } ) / .look       strand / tube material parameters, see LOOK_DEFAULTS
//  .curves                        the EdgeCurves cache
//  .stats                         { edges, segments (of the cache), strandSegments (of the strip),
//                                   vertices, triangles, autoOpacity, tubes, tubeSegments,
//                                   tubeShadows, style, curveUpdates, curveSkipped, curveBytes, variant }
//  .mesh / .tubeMesh              the THREE.Mesh of the strand layer (renderOrder −1: drawn before
//                                 the nodes) and of the tube layer; world space, their own
//                                 transform is ignored; raycast disabled
//
//  Also exported: autoSegments( E, maxPath, sphereScale ), autoOpacity( E ), autoDefocusLimit( E ),
//  autoCurveMemory( coarsePointer ), strandSubdivision( S, E, vertexLimit ),
//  EDGE_DEFAULTS, LOOK_DEFAULTS and evalEdgeCurveCPU( … ) (re-exported from gpu/EdgeCurves.js).

import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, instancedArray, positionGeometry, uniform, float, uint, vec2, vec3, vec4, select, mix, clamp, max, min,
	abs, sqrt, pow, length, dot, normalize, smoothstep, varyingProperty, cameraProjectionMatrix, cameraViewMatrix,
	cameraWorldMatrix, cameraPosition, viewport, mrt, output,
} from 'three/tsl';
import { EdgeCurves, evalEdgeCurveCPU } from '../gpu/EdgeCurves.js';
import { buildEdgeIndex } from '../graph/analytics.js';
import { gfxUniforms, gfxState, motionVector, velocityOutput, skipVelocity, fogTransmittance, cocPixels } from './gfx/shared.js';
import { isOrthographic, pixelsPerUnit, viewToDepth, viewToWorld, applySurface } from './gfx/impostor.js';
import { gradientColor, strandLight, tubeFrame, flowPulse } from './gfx/strandSurface.js';

export { evalEdgeCurveCPU };

export const EDGE_DEFAULTS = {
	beta: 0.85,
	opacity: 1,
	width: 1,
	minPixelWidth: 0.75, // CSS px (full width); thinner edges fade by coverage below 1 pass px
	maxPixelWidth: 8, // CSS px (full width): strands stay lines near the camera (tubes have their own limit)
	blendMode: 'normal',
	segments: null,
	colorMode: 'gradient',
	visible: true,
	contextOpacity: 0.12, // opacity factor of non-highlighted edges while a selection exists
	highlightGlow: 1.6, // extra HDR brightness of highlighted edges (bloom halo)
	highlightOpacity: 0.9,
	highlightLevel: 0.75, // sphere mode: level the cluster points of highlighted edges move to
	hoverBoost: 0.6,
	brightness: 1.25,
	sphereSegmentScale: 2,
	renderOrder: - 1,
	style: 'luminous',
	tubes: true,
	tubeLimit: 30000, // most highlight tubes; a selection with more edges stays strands
	tubeShadowLimit: 20000, // tubes cast shadows while there are at most this many
	solidLimit: 50000, // style 'solid' applies to graphs with at most this many edges
	tubeLinger: 1.2, // seconds an edge stays in the tube list after it left the highlight (ease-out)
	tubeVertexLimit: 2e6, // most vertices of the tube draw: bounds its subdivision (see TUBE_SEGMENTS)
	strandVertexLimit: 3e6, // most vertices of the strand draw that its subdivision may lead to (see STRAND_SEGMENTS)
	defocusLimit: null, // most strand defocus (circle of confusion, CSS px); null = auto from E
	flow: 'off',
	flowSpeed: 0.3, // layout radii per second
	flowSpacing: 0.5, // layout radii between pulses
	strandVelocity: 'camera',
	memoryBudget: null, // bytes of the curve cache; null = 64 MiB, 24 MiB on touch devices
};

/** Curve cache budget for a device (bytes): the buffer exists twice, on the GPU and as three's CPU copy. */
export function autoCurveMemory( coarsePointer = typeof matchMedia === 'function' && matchMedia( '(pointer:coarse)' ).matches ) {

	return ( coarsePointer ? 24 : 64 ) * 1024 * 1024;

}

/** Material parameters of the strand and tube layers (edges.setLook). */
export const LOOK_DEFAULTS = {
	// strands (gfxState.lit)
	emissive: 0.37, // self-luminous part of the strand colour
	diffuse: 0.63, // weight of the key light's cylinder-averaged diffuse term (and of the ambient)
	sheen: 0.18, // strength of the Kajiya-Kay highlight
	gloss: 40, // its exponent (larger = narrower band)
	sheenTint: 1, // 0 = highlight in the light colour, 1 = in the strand colour (keeps bundles saturated)
	shift: 0.22, // per-strand shift of the highlight along the strand
	sheath: 0.3, // part of a highlighted strand that stays as a soft halo around its tube
	setContext: 0.6, // how much of the selection's context fade a highlighted set (a path) applies
	// tubes
	tubeRadius: 2.1, // tube radius in strand half widths
	tubeMinPixelWidth: 2.4, // CSS px (full width)
	tubeMaxPixelWidth: 24, // CSS px (full width)
	tubeTaper: 0.35, // share of its radius (and 0.3 + 0.7 × that of its emission) a tube keeps where it meets a node
	crowdFloor: 0.35, // least emission factor of the tubes of a hub (1 / √( edges / 8 ) above it)
	tubeRoughness: 0.34,
	tubeMetalness: 0,
	tubeAlbedo: 0.6, // diffuse colour = gradient × this
	tubeGlow: 0.7, // emissive of highlighted tubes (× gradient colour × brightness)
	solidGlow: 0.55, // emissive of the tubes of style 'solid' …
	solidGlowThin: 0.9, // … rising to this as a tube gets thinner than 3 CSS px (distant tubes stay luminous, close ones lit)
	chroma: 0.85, // 0..1: how far the gradient's chroma is restored to the mix of the two end chromas
	contextDim: 0.3, // style 'solid': colour factor of the context while a selection exists
	// flow
	flowGain: 1.6, // extra emission at a pulse head
	flowTail: 5, // decay rate of the pulse tail per period
	flowJitter: 1, // 0 = all edges pulse in step, 1 = random phase per edge
};

// Tube strips are finer than the cache: about this many segments per tube (an integer number of
// vertices per cached segment, at most 8), interpolated through the samples in the vertex stage.
const TUBE_SEGMENTS = 96;
// … and so are the strands: about this many segments per edge, at most STRAND_SUB_MAX vertices per
// cached segment, bounded by strandVertexLimit.
const STRAND_SEGMENTS = 96;
const STRAND_SUB_MAX = 4;

/**
 * Ribbon segments per edge. Base by E (<=2k: 32, <=20k: 20, <=100k: 12, else 8), scaled up for
 * hierarchies deeper than 2 levels (more B-spline spans per edge) and by `sphereScale` while the
 * sphere layout is shown (its arcs are longer and cross the sphere silhouette).
 */
export function autoSegments( E, maxPath = 6, sphereScale = 1 ) {

	const base = E <= 2000 ? 32 : E <= 20000 ? 20 : E <= 100000 ? 12 : 8;
	return Math.min( 96, Math.round( base * Math.max( 1, ( maxPath + 1 ) / 7 ) * sphereScale ) );

}

/**
 * Vertex pairs per cached curve segment of the strand strip: about STRAND_SEGMENTS segments per
 * edge (at most STRAND_SUB_MAX per cached segment), fewer while E edges would otherwise pass
 * `vertexLimit` vertices, never less than 1 (the cache resolution itself is not reduced).
 */
export function strandSubdivision( S, E, vertexLimit = EDGE_DEFAULTS.strandVertexLimit ) {

	const wanted = Math.max( 1, Math.min( STRAND_SUB_MAX, Math.round( STRAND_SEGMENTS / Math.max( 1, S ) ) ) );
	const fits = Math.floor( ( vertexLimit / ( 2 * Math.max( 1, E ) ) - 1 ) / Math.max( 1, S ) );
	return Math.max( 1, Math.min( wanted, fits ) );

}

/** Base edge opacity for E edges: dense graphs need lower alpha so bundles do not saturate. */
export function autoOpacity( E ) {

	return Math.min( 0.8, Math.max( 0.035, 0.85 * Math.pow( 400 / Math.max( 1, E ), 0.42 ) ) );

}

/**
 * Most strand defocus (circle-of-confusion radius, CSS px) for E edges: the fill of the blended
 * strand layer grows with the widened ribbons, so dense graphs get less of it. 10 px up to 5k
 * edges, falling with log E to 0 at 200k.
 */
export function autoDefocusLimit( E ) {

	return 10 * Math.min( 1, Math.max( 0, 1 - Math.log( Math.max( 1, E ) / 5000 ) / Math.log( 40 ) ) );

}

/** World half width factor for E edges (fraction of graph.layoutRadius). */
function autoHalfWidth( E ) {

	return 0.0022 * Math.min( 1, Math.max( 0.45, Math.pow( 1500 / Math.max( 1, E ), 0.2 ) ) );

}

/** Ribbon strip of S segments: position = ( sample index 0 … S, side ±1, 0 ). */
function ribbonGeometry( S ) {

	const geometry = new THREE.InstancedBufferGeometry();
	const pos = new Float32Array( ( S + 1 ) * 2 * 3 );
	for ( let i = 0; i <= S; i ++ ) pos.set( [ i, - 1, 0, i, 1, 0 ], i * 6 );
	const index = [];
	for ( let i = 0; i < S; i ++ ) {

		const a = 2 * i;
		index.push( a, a + 1, a + 2, a + 1, a + 3, a + 2 );

	}

	geometry.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
	geometry.setIndex( index );
	geometry.boundingSphere = new THREE.Sphere( new THREE.Vector3(), Infinity );
	return geometry;

}

/** MeshStandardNodeMaterial whose lit result is fogged in the material (the scene has no fog node). */
class TubeNodeMaterial extends THREE.MeshStandardNodeMaterial {

	setupOutput( builder, outputNode ) {

		const t = this.fogTransmittanceNode;
		if ( t ) outputNode = vec4( outputNode.rgb.mul( t ).add( gfxUniforms.fogColor.mul( t.oneMinus() ) ), outputNode.a );
		return super.setupOutput( builder, outputNode );

	}

}

const toIds = ( ids, E ) => {

	if ( ids == null ) return null;
	const out = [];
	for ( const e of ids ) if ( Number.isInteger( e ) && e >= 0 && e < E ) out.push( e );
	return out.length ? Uint32Array.from( new Set( out ) ) : null;

};

export class EdgeRenderer {

	constructor( options = {} ) {

		const o = { ...EDGE_DEFAULTS, ...options };
		this.order = 10;
		this.isEdgeRenderer = true;
		this.app = null;
		this.buffers = null;
		this.graph = null;
		this.mesh = null;
		this.tubeMesh = null;
		this.material = null;
		this.geometry = null;
		this.tubeGeometry = null;
		this._S = 0;
		this._E = 0;
		this._auto = 1;
		this._opacity = o.opacity;
		this._width = o.width;
		this._segments = o.segments;
		this._blendMode = o.blendMode === 'additive' ? 'additive' : 'normal';
		this._colorMode = o.colorMode;
		this._visible = o.visible;
		this._sphereSegments = false;
		this._style = o.style;
		this._tubes = !! o.tubes;
		this._flow = o.flow;
		this._flowSpeed = o.flowSpeed;
		this._flowSpacing = o.flowSpacing;
		this._strandVelocity = o.strandVelocity;
		this.renderOrder = o.renderOrder;
		this.sphereSegmentScale = o.sphereSegmentScale; // auto segments multiplier in sphere mode
		this.tubeLimit = o.tubeLimit;
		this.tubeShadowLimit = o.tubeShadowLimit;
		this.solidLimit = o.solidLimit;
		this.tubeLinger = o.tubeLinger;
		this.tubeVertexLimit = o.tubeVertexLimit;
		this.strandVertexLimit = o.strandVertexLimit;
		this._defocusLimit = o.defocusLimit;
		this.look = { ...LOOK_DEFAULTS, ...( o.look || {} ) };

		/** The curve cache. It survives graph changes (its buffers do not). */
		this.curves = new EdgeCurves( { beta: o.beta, highlightLevel: o.highlightLevel, sphereAxis: o.sphereAxis, memoryBudget: o.memoryBudget ?? autoCurveMemory() } );
		this.curves.onRealloc = () => this._disposeMaterials();

		// uniforms owned by the renderer (they survive graph changes)
		const L = this.look;
		this.u = {
			beta: this.curves.uniforms.beta,
			highlightLevel: this.curves.uniforms.highlightLevel,
			sphereAxis: this.curves.uniforms.sphereAxis,
			hlFade: this.curves.uniforms.highlightFade, // eased weight of edgeAttr.y
			opacity: uniform( 1 ), // auto(E) * opacity multiplier
			halfWidth: uniform( 1 ), // world-space half width
			minPx: uniform( o.minPixelWidth ),
			maxPx: uniform( o.maxPixelWidth ),
			sourceColor: uniform( o.colorMode === 'source' ? 1 : 0 ),
			contextOpacity: uniform( o.contextOpacity ),
			highlightGlow: uniform( o.highlightGlow ),
			highlightOpacity: uniform( o.highlightOpacity ),
			hoverBoost: uniform( o.hoverBoost ),
			brightness: uniform( o.brightness ),
			additive: uniform( this._blendMode === 'additive' ? 1 : 0 ),
			handover: uniform( 0 ), // 1 while highlighted edges are drawn as tubes
			solid: uniform( 0 ), // eased 0..1: style 'solid'
			metricOpacity: uniform( new THREE.Vector2( 1, 1 ) ),
			metricWidth: uniform( new THREE.Vector2( 1, 1 ) ),
			flowSel: uniform( 0 ),
			flowAll: uniform( 0 ),
			flowSpeed: uniform( 1 ), // world units / s
			flowSpacing: uniform( 1 ), // world units
			defocusMax: uniform( 10 ), // most strand circle of confusion, CSS px
			tubeSub: uniform( 1, 'uint' ), // tube vertices per cached curve segment
			strandSub: uniform( 1, 'uint' ), // strand vertices per cached curve segment
			sheathFit: uniform( 0 ), // 1 while the tubes are smoother than the strands (tubeSub > 1, strandSub = 1)
			nodeRadius: uniform( 1 ), // world radius of a node of radius factor 1 (tube ends taper toward the node surface)
			crowd: uniform( 1 ), // emission factor of highlighted edges: falls with the size of the selection's bundle
			// look
			emissive: uniform( L.emissive ), diffuse: uniform( L.diffuse ), sheen: uniform( L.sheen ), gloss: uniform( L.gloss ),
			sheenTint: uniform( L.sheenTint ), shift: uniform( L.shift ), sheath: uniform( L.sheath ), setContext: uniform( L.setContext ),
			tubeRadius: uniform( L.tubeRadius ), tubeMinPixelWidth: uniform( L.tubeMinPixelWidth ), tubeRoughness: uniform( L.tubeRoughness ),
			tubeMaxPixelWidth: uniform( L.tubeMaxPixelWidth ), tubeTaper: uniform( L.tubeTaper ), chroma: uniform( L.chroma ),
			tubeMetalness: uniform( L.tubeMetalness ), tubeAlbedo: uniform( L.tubeAlbedo ), tubeGlow: uniform( L.tubeGlow ),
			solidGlow: uniform( L.solidGlow ), solidGlowThin: uniform( L.solidGlowThin ), contextDim: uniform( L.contextDim ),
			flowGain: uniform( L.flowGain ), flowTail: uniform( L.flowTail ), flowJitter: uniform( L.flowJitter ),
		};

		// tube list state
		this.tubeIndex = null; // storage node: instance → edge id
		this._tubeCapacity = 0;
		this._tubeCount = 0;
		this._tubeSegments = 0;
		this._strandSegments = 0;
		this._tubeDirty = true;
		this._time = 0;
		this._shadowUntil = - Infinity; // tubes are easing: the shadow map follows until then …
		this._shadowFrames = 0; // … and for at least this many frames
		this._selEdges = null; // Uint32Array: edges of the selected node(s)
		this._linger = []; // [ { edges, until, frames } ]: edges easing out
		this._hl = { edges: null, weight: 1, pending: undefined, pendingWeight: 1, fade: 0 };
		this._solidOn = false;
		this._selOverLimit = false;
		this._shadowsInvolved = false;
		this._stateVersion = - 1;
		this._selIndexSeen = - 2;
		this._adjEdge = null;
		this._pathsSeen = null;
		this._strandMaterials = new Map();
		this._tubeMaterials = new Map();
		this._variant = '';
		this._onGfx = () => this._applyVariant();
		this._onPath = ( ev ) => this.setHighlightedEdges( ev.detail ? ev.detail.edges : null );

	}

	init( app ) {

		this.app = app;
		app.events.addEventListener( 'gfx', this._onGfx );
		app.events.addEventListener( 'path', this._onPath );

	}

	// ── API ────────────────────────────────────────────────────────────────────────────────

	setBundling( beta ) {

		this.u.beta.value = Math.min( 1, Math.max( 0, beta ) );

	}

	get bundling() {

		return this.u.beta.value;

	}

	get opacity() {

		return this._opacity;

	}

	set opacity( v ) {

		this._opacity = Math.max( 0, v );
		this._updateUniforms();

	}

	get width() {

		return this._width;

	}

	set width( v ) {

		this._width = Math.max( 0, v );
		this._updateUniforms();

	}

	get minPixelWidth() {

		return this.u.minPx.value;

	}

	set minPixelWidth( v ) {

		this.u.minPx.value = Math.max( 0, v );

	}

	get maxPixelWidth() {

		return this.u.maxPx.value;

	}

	set maxPixelWidth( v ) {

		this.u.maxPx.value = Math.max( 0.1, v );

	}

	get blendMode() {

		return this._blendMode;

	}

	set blendMode( v ) {

		this._blendMode = v === 'additive' ? 'additive' : 'normal';
		this.u.additive.value = this._blendMode === 'additive' ? 1 : 0;
		const blending = this._blendMode === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending;
		for ( const m of this._strandMaterials.values() ) {

			if ( m.blending !== blending ) {

				m.blending = blending;
				m.needsUpdate = true;

			}

		}

	}

	get segments() {

		return this._segments;

	}

	set segments( v ) {

		this._segments = v == null ? null : Math.max( 1, Math.min( 256, Math.round( v ) ) );
		this._updateGeometry();

	}

	get colorMode() {

		return this._colorMode;

	}

	set colorMode( v ) {

		this._colorMode = v === 'source' ? 'source' : 'gradient';
		this.u.sourceColor.value = this._colorMode === 'source' ? 1 : 0;

	}

	get visible() {

		return this._visible;

	}

	set visible( v ) {

		this._visible = !! v;
		this._updateVisibility();

	}

	get contextOpacity() {

		return this.u.contextOpacity.value;

	}

	set contextOpacity( v ) {

		this.u.contextOpacity.value = v;

	}

	get highlightGlow() {

		return this.u.highlightGlow.value;

	}

	set highlightGlow( v ) {

		this.u.highlightGlow.value = v;

	}

	get highlightLevel() {

		return this.u.highlightLevel.value;

	}

	set highlightLevel( v ) {

		this.u.highlightLevel.value = v;

	}

	get hoverBoost() {

		return this.u.hoverBoost.value;

	}

	set hoverBoost( v ) {

		this.u.hoverBoost.value = v;

	}

	get sphereAxis() {

		return this.u.sphereAxis.value;

	}

	set sphereAxis( v ) {

		this.u.sphereAxis.value.copy( v ).normalize();

	}

	get brightness() {

		return this.u.brightness.value;

	}

	set brightness( v ) {

		this.u.brightness.value = v;

	}

	/** 'luminous' | 'solid' | 'auto'. */
	get style() {

		return this._style;

	}

	set style( v ) {

		this._style = v === 'solid' || v === 'auto' ? v : 'luminous';
		this._tubeDirty = true;

	}

	/** Whether highlighted edges are drawn as tubes (true) or stay bright strands (false). */
	get tubes() {

		return this._tubes;

	}

	set tubes( v ) {

		this._tubes = !! v;
		this._tubeDirty = true;

	}

	/** 'off' | 'selection' | 'all'. */
	get flow() {

		return this._flow;

	}

	set flow( v ) {

		this._flow = v === 'selection' || v === 'all' ? v : 'off';
		this._updateUniforms();
		this._applyVariant();

	}

	/** True while pulses travel along the edges: the App keeps rendering at the display rate. */
	get animating() {

		return this._visible && this._flow !== 'off' && ( this._flow === 'all' || this._tubeCount > 0 || this._selEdges !== null || this._hl.edges !== null );

	}

	get flowSpeed() {

		return this._flowSpeed;

	}

	set flowSpeed( v ) {

		this._flowSpeed = + v || 0;
		this._updateUniforms();

	}

	get flowSpacing() {

		return this._flowSpacing;

	}

	set flowSpacing( v ) {

		this._flowSpacing = Math.max( 1e-3, + v || 0 );
		this._updateUniforms();

	}

	/** 'camera' | 'true' | 'skip': what the strands write into the velocity attachment. */
	get strandVelocity() {

		return this._strandVelocity;

	}

	set strandVelocity( v ) {

		this._strandVelocity = v === 'skip' || v === 'true' ? v : 'camera';
		this._applyVariant();

	}

	/** Most strand defocus (circle-of-confusion radius, CSS px); null = auto from the edge count. */
	get defocusLimit() {

		return this._defocusLimit;

	}

	set defocusLimit( v ) {

		this._defocusLimit = v == null ? null : Math.max( 0, + v || 0 );
		this._updateUniforms();

	}

	/** Strand / tube material parameters; see LOOK_DEFAULTS for the keys. */
	setLook( look = {} ) {

		for ( const key of Object.keys( look ) ) {

			if ( ! ( key in LOOK_DEFAULTS ) ) continue;
			this.look[ key ] = look[ key ];
			if ( this.u[ key ] ) this.u[ key ].value = look[ key ];

		}

		return this.look;

	}

	/** Swap the control paths (see gpu/EdgeCurves.js); null = hierarchical bundling. */
	setPaths( paths = null ) {

		this.curves.setPaths( paths );
		this._pathsSeen = this.curves.paths;
		this._updateGeometry();

	}

	/**
	 * Edges drawn as highlighted (tubes, or bright strands when tubes are off): a path, a tool's
	 * result. `ids`: iterable of edge ids, or null. A set that replaces another fades in after the
	 * old one faded out. Also called by the app event 'path' ( detail.edges ).
	 */
	setHighlightedEdges( ids, weight = 1 ) {

		const h = this._hl;
		const next = toIds( ids, this._E );
		if ( h.edges === null && h.pending === undefined ) this._writeHighlight( next, weight );
		else {

			h.pending = next;
			h.pendingWeight = weight;

		}

	}

	/**
	 * Per-edge metric: `values` (length E, normalized to [0,1]) scale the opacity and the width of
	 * every edge between the factors given for 0 and for 1. null removes the metric.
	 */
	setEdgeMetric( values, { opacity = [ 0.3, 1.5 ], width = [ 0.7, 2 ] } = {} ) {

		const b = this.buffers;
		if ( ! b ) return;
		const a = b.edgeAttr.value.array;
		for ( let e = 0; e < this._E; e ++ ) a[ 4 * e + 2 ] = values ? Math.min( 1, Math.max( 0, values[ e ] || 0 ) ) : 0;
		b.edgeAttr.value.clearUpdateRanges();
		b.edgeAttr.value.needsUpdate = true;
		const pair = ( v ) => ( Array.isArray( v ) ? v : [ 1, v ] );
		this.u.metricOpacity.value.fromArray( values ? pair( opacity ) : [ 1, 1 ] );
		this.u.metricWidth.value.fromArray( values ? pair( width ) : [ 1, 1 ] );

	}

	/** Per-edge width factors (length E; 1 = unchanged), e.g. the load of a carrying edge. null resets. */
	setEdgeWidths( values ) {

		const b = this.buffers;
		if ( ! b ) return;
		const a = b.edgeAttr.value.array;
		for ( let e = 0; e < this._E; e ++ ) a[ 4 * e ] = values ? Math.max( 0, values[ e ] ) : 1;
		b.edgeAttr.value.clearUpdateRanges();
		b.edgeAttr.value.needsUpdate = true;

	}

	get stats() {

		const c = this.curves.stats;
		return {
			edges: this._E, segments: this._S, strandSegments: this._strandSegments, vertices: this._E * ( this._strandSegments + 1 ) * 2, triangles: this._E * this._strandSegments * 2, autoOpacity: this._auto,
			tubes: this._tubeCount, tubeSegments: this._tubeSegments, tubeShadows: this._tubesCast(), style: this._solidOn ? 'solid' : 'luminous',
			curveUpdates: c.updates, curveSkipped: c.skipped, curveBytes: c.bytes, variant: this._variant,
		};

	}

	// ── module hooks ───────────────────────────────────────────────────────────────────────

	setGraph( buffers, graph, app ) {

		this.app = app || this.app;
		this._disposeGPU();
		this.buffers = buffers;
		this.graph = graph;
		this._E = graph.E;
		this._auto = autoOpacity( graph.E );
		this._adjEdge = null;
		this._selEdges = null;
		this._linger = [];
		this._hl = { edges: null, weight: 1, pending: undefined, pendingWeight: 1, fade: 0 };
		this.u.hlFade.value = 0;
		this.u.metricOpacity.value.set( 1, 1 );
		this.u.metricWidth.value.set( 1, 1 );
		this._stateVersion = - 1;
		this._selIndexSeen = - 2;
		this._selOverLimit = false;
		this._solidOn = false;
		this.u.solid.value = 0;
		this._updateUniforms();

		const sphere = buffers.uniforms.layoutBlend.value > 0;
		this._sphereSegments = sphere;
		// room for the sphere layout's finer sampling from the start (while that is cheap): the first
		// morph then changes the stride only, no buffer and no shader is rebuilt in its first frame
		const reserve = this._segments == null ? this._wantedSegments( graph.maxPath, true ) : 0;
		this.curves.setGraph( buffers, graph, { segments: this._wantedSegments( graph.maxPath, sphere ), reserve } );
		this._pathsSeen = this.curves.paths;
		this._S = 0;
		this._tubeSegments = 0;
		this._strandSegments = 0;
		this._shadowUntil = - Infinity;
		this._shadowFrames = 0;
		this._shadowsInvolved = false;
		// the tube index list, allocated once for the most tubes there can be (at most 200 kB): the
		// tube materials bind it, so a list that grew with the selection would rebuild them at a click
		this._tubeCapacity = Math.max( 64, Math.min( graph.E, Math.max( this.tubeLimit, this.solidLimit ) ) );
		this.tubeIndex = instancedArray( this._tubeCapacity, 'uint' ).setName( 'edgeTubeIndex' );
		this._updateGeometry();

		const mesh = new THREE.Mesh( this.geometry, null );
		mesh.name = 'edges';
		mesh.frustumCulled = false;
		mesh.raycast = () => {}; // the geometry holds ( sample, side ), not positions: never raycast it
		mesh.renderOrder = this.renderOrder;
		this.mesh = mesh;

		const tubes = new THREE.Mesh( this.tubeGeometry, null );
		tubes.name = 'edgeTubes';
		tubes.frustumCulled = false; // also for the shadow camera
		tubes.raycast = () => {};
		tubes.receiveShadow = true;
		// visible and shadow casting also while the list is empty (zero instances draw nothing): the
		// renderer then compiles the tube pipelines with the rest of the scene, not at the first click
		tubes.castShadow = true;
		this.tubeMesh = tubes;

		this._tubeDirty = true;
		this._applyVariant();
		this._updateVisibility();
		this.app.scene.add( mesh, tubes );

	}

	update( dt, app ) {

		const buffers = this.buffers;
		if ( ! buffers ) return;
		app = app || this.app;
		const curves = this.curves;
		const time = app.time ?? 0;
		dt = Math.min( Math.max( dt || 0, 0 ), 0.1 );

		// O(1) bookkeeping: denser auto sampling while the sphere layout is (partly) shown, and a
		// path set that was swapped behind our back (curves.setPaths)
		const sphere = buffers.uniforms.layoutBlend.value > 0;
		if ( curves.paths !== this._pathsSeen || ( this._segments == null && sphere !== this._sphereSegments ) ) {

			this._pathsSeen = curves.paths;
			this._updateGeometry();

		}

		// highlight cross-fade (edgeAttr.y × hlFade)
		const h = this._hl;
		const fadeTarget = h.edges !== null && h.pending === undefined ? 1 : 0;
		if ( h.fade !== fadeTarget ) {

			h.fade = fadeTarget > h.fade ? Math.min( 1, h.fade + dt / 0.3 ) : Math.max( 0, h.fade - dt / 0.18 );
			this.u.hlFade.value = h.fade * h.fade * ( 3 - 2 * h.fade );
			this._requestShadowUpdate();

		}

		if ( h.fade === 0 && h.pending !== undefined ) {

			const next = h.pending;
			h.pending = undefined;
			this._writeHighlight( next, h.pendingWeight );

		}

		// style: 'solid' eases every edge into a tube
		const solidWanted = this._style === 'solid' && this._E > 0 && this._E <= this.solidLimit;
		const s = this.u.solid.value;
		if ( solidWanted !== this._solidOn && ( solidWanted || s === 0 ) ) {

			this._solidOn = solidWanted;
			this._tubeDirty = true;

		}

		const solidTarget = solidWanted ? 1 : 0;
		if ( s !== solidTarget ) {

			this.u.solid.value = solidTarget > s ? Math.min( 1, s + dt / 0.5 ) : Math.max( 0, s - dt / 0.5 );
			this._updateVisibility();
			this._requestShadowUpdate();

		}

		// tube ends taper toward the node surface: the node renderer's base radius (as it sizes them)
		if ( this._nodes === undefined || ( this._nodes && ! app.modules.includes( this._nodes ) ) ) this._nodes = ( app.modules && app.modules.find( ( m ) => m && m.isNodeRenderer ) ) || null;
		const g = this.graph;
		this.u.nodeRadius.value = this._nodes && this._nodes.baseSize > 0 ? this._nodes.baseSize * ( this._nodes.size ?? 1 ) : 0.065 * g.layoutRadius / Math.cbrt( Math.max( 1, g.N ) );

		this._time = time;
		this._updateSelection( time );
		if ( this._tubeDirty ) this._rebuildTubeList();

		// tubes that ease in or out change their shadows every frame
		if ( time < this._shadowUntil || this._shadowFrames > 0 ) {

			this._shadowFrames --;
			this._requestShadowUpdate();

		}

		curves.update( app );

	}

	dispose() {

		this._disposeGPU();
		this.curves.dispose();
		if ( this.app ) {

			this.app.events.removeEventListener( 'gfx', this._onGfx );
			this.app.events.removeEventListener( 'path', this._onPath );

		}

		this.buffers = null;
		this.graph = null;

	}

	// ── internals: uniforms, geometry, visibility ─────────────────────────────────────────

	_updateUniforms() {

		this.u.opacity.value = this._auto * this._opacity;
		this.u.flowSel.value = this._flow === 'selection' ? 1 : 0;
		this.u.flowAll.value = this._flow === 'all' ? 1 : 0;
		if ( this.graph ) {

			const R = this.graph.layoutRadius;
			this.u.halfWidth.value = R * autoHalfWidth( this.graph.E ) * this._width;
			this.u.flowSpeed.value = R * this._flowSpeed;
			this.u.flowSpacing.value = R * this._flowSpacing;
			this.u.defocusMax.value = this._defocusLimit ?? autoDefocusLimit( this.graph.E );

		}

	}

	_wantedSegments( maxPath, sphere ) {

		return this._segments ?? autoSegments( this._E, maxPath, sphere ? this.sphereSegmentScale : 1 );

	}

	_updateGeometry() {

		if ( ! this.graph ) return;
		const sphere = this.buffers.uniforms.layoutBlend.value > 0;
		this._sphereSegments = sphere;
		const paths = this.curves.paths;
		const maxPath = paths ? paths.maxInterior + 2 : this.graph.maxPath;
		const S = this.curves.setSegments( this._wantedSegments( maxPath, sphere ) );
		// the strip is finer than the cache (see the header): vertex pairs per cached segment
		const sub = strandSubdivision( S, this.graph.E, this.strandVertexLimit );
		if ( S * sub !== this._strandSegments || ! this.geometry ) {

			const old = this.geometry;
			this.geometry = ribbonGeometry( S * sub );
			this._strandSegments = S * sub;
			if ( this.mesh ) this.mesh.geometry = this.geometry;
			if ( old ) old.dispose();

		}

		this._S = S;
		this.u.strandSub.value = sub;
		this.geometry.instanceCount = this.graph.E;
		this._updateTubeGeometry();

	}

	/**
	 * The tube strip: `sub` vertices per cached segment (about TUBE_SEGMENTS per tube, at most 8 per
	 * segment), fewer when the tube count would push the draw past tubeVertexLimit.
	 */
	_updateTubeGeometry() {

		const S = this._S, n = this._tubeCount;
		if ( ! S ) return;
		const wanted = Math.max( 1, Math.min( 8, Math.round( TUBE_SEGMENTS / S ) ) );
		const fits = Math.floor( ( this.tubeVertexLimit / ( 2 * Math.max( 1, n ) ) - 1 ) / S );
		const sub = Math.max( 1, Math.min( wanted, fits ) );
		if ( S * sub !== this._tubeSegments || ! this.tubeGeometry ) {

			const old = this.tubeGeometry;
			this.tubeGeometry = ribbonGeometry( S * sub );
			this._tubeSegments = S * sub;
			if ( this.tubeMesh ) this.tubeMesh.geometry = this.tubeGeometry;
			if ( old ) old.dispose();

		}

		this.u.tubeSub.value = sub;
		this.u.sheathFit.value = sub > 1 && this.u.strandSub.value === 1 ? 1 : 0;
		this.tubeGeometry.instanceCount = n;

	}

	_updateVisibility() {

		if ( this.mesh ) this.mesh.visible = this._visible && this._E > 0 && this.u.solid.value < 1;
		// the tube mesh stays in the scene without instances (see setGraph)
		if ( this.tubeMesh ) this.tubeMesh.visible = this._visible && this._E > 0;

	}

	/** Whether tubes are in the shadow map. */
	_tubesCast() {

		return !! this.tubeMesh && this.tubeMesh.castShadow && this._tubeCount > 0;

	}

	// ── internals: tube list ───────────────────────────────────────────────────────────────

	/** Edge ids incident to the nodes whose selection target is set (nodeStateTarget.x). */
	_updateSelection( time ) {

		const b = this.buffers;
		const target = b.nodeStateTarget.value;
		const sel = b.uniforms.selectedIndex.value;
		const linger = this._linger;

		// drop the edges that finished easing out
		for ( let i = linger.length - 1; i >= 0; i -- ) {

			const l = linger[ i ];
			l.frames --;
			if ( time >= l.until && l.frames <= 0 ) {

				linger.splice( i, 1 );
				this._tubeDirty = true;

			}

		}

		if ( linger.length ) this._requestShadowUpdate();
		if ( target.version === this._stateVersion && sel === this._selIndexSeen ) return;
		this._stateVersion = target.version;
		this._selIndexSeen = sel;

		const g = this.graph, N = g.N, t = target.array;
		const nodes = [];
		let count = 0;
		for ( let i = 0; i < N; i ++ ) if ( t[ 4 * i ] > 0.5 ) {

			nodes.push( i );
			count += g.degree[ i ];

		}

		let edges = null;
		if ( nodes.length && count <= this.tubeLimit ) {

			if ( ! this._adjEdge ) this._adjEdge = g.adjEdge || buildEdgeIndex( g );
			const adjEdge = this._adjEdge;
			const set = new Set();
			for ( const i of nodes ) for ( let k = g.adjStart[ i ], end = k + g.degree[ i ]; k < end; k ++ ) set.add( adjEdge[ k ] );
			edges = Uint32Array.from( set );

		}

		const prev = this._selEdges;
		const same = prev === edges || ( prev && edges && prev.length === edges.length && prev.every( ( e, i ) => e === edges[ i ] ) );
		if ( same ) return;
		if ( prev ) linger.push( { edges: prev, until: time + this.tubeLinger, frames: 24 } );
		this._selEdges = edges;
		this._selOverLimit = nodes.length > 0 && edges === null;
		// a hub's bundle converges on one node: its emission is shared out, so it does not sum to white
		// (kept while the old tubes ease out: the value of the last selection stays until the next one)
		if ( edges ) this.u.crowd.value = Math.max( this.look.crowdFloor, 1 / Math.sqrt( Math.max( 1, edges.length / 8 ) ) );
		this._tubeDirty = true;
		this._easeShadows(); // the new tubes grow with the eased state, the old ones shrink

	}

	_rebuildTubeList() {

		this._tubeDirty = false;
		if ( ! this.graph ) return;
		const E = this._E;
		let list;
		if ( this._solidOn ) {

			list = new Uint32Array( E );
			for ( let e = 0; e < E; e ++ ) list[ e ] = e;

		} else {

			const set = new Set();
			const h = this._hl;
			if ( this._tubes ) {

				if ( this._selEdges ) for ( const e of this._selEdges ) set.add( e );
				for ( const l of this._linger ) for ( const e of l.edges ) set.add( e );
				if ( h.edges ) for ( const e of h.edges ) set.add( e );

			}

			// too many for the tube layer: the highlight stays on the strands
			const over = this._selOverLimit || set.size > this.tubeLimit;
			list = over ? new Uint32Array( 0 ) : Uint32Array.from( set );
			// highlighted strands hand over to their tubes only when those are really drawn
			this.u.handover.value = this._tubes && ! over ? 1 : 0;

		}

		const n = list.length;
		if ( n > this._tubeCapacity ) {

			// only after tubeLimit / solidLimit were raised past the size chosen in setGraph: a larger
			// index buffer; the tube materials bind it, so they are rebuilt
			const old = this.tubeIndex;
			this._tubeCapacity = Math.max( 64, Math.min( E, Math.max( n, this.tubeLimit, this.solidLimit ) ) );
			this.tubeIndex = instancedArray( this._tubeCapacity, 'uint' ).setName( 'edgeTubeIndex' );
			for ( const m of this._tubeMaterials.values() ) m.dispose();
			this._tubeMaterials.clear();
			if ( old ) old.value.dispose();
			this._applyVariant();

		}

		if ( n > 0 ) {

			this.tubeIndex.value.array.set( list );
			this.tubeIndex.value.needsUpdate = true;

		}

		const wasCast = this._tubesCast();
		this._tubeCount = n;
		this._updateTubeGeometry();
		// without tubes the mesh keeps casting (nothing is drawn; see setGraph)
		if ( this.tubeMesh ) this.tubeMesh.castShadow = n <= this.tubeShadowLimit;
		this._updateVisibility();
		// the list changed: the shadow map needs the tubes that came or went
		this._shadowsInvolved = wasCast || this._tubesCast();
		this._requestShadowUpdate();
		this._shadowsInvolved = this._tubesCast();

	}

	/** edgeAttr.y of the old set → 0, of the new set → weight; the fade uniform restarts at 0. */
	_writeHighlight( edges, weight ) {

		const b = this.buffers;
		const h = this._hl;
		if ( b ) {

			const attr = b.edgeAttr.value, a = attr.array;
			const touched = ( h.edges ? h.edges.length : 0 ) + ( edges ? edges.length : 0 );
			const ranged = touched <= 512 && attr.updateRanges.length + touched <= 1024;
			const write = ( list, v ) => {

				if ( ! list ) return;
				for ( const e of list ) {

					a[ 4 * e + 1 ] = v;
					if ( ranged ) attr.addUpdateRange( 4 * e, 4 );

				}

			};

			write( h.edges, 0 );
			write( edges, Math.min( 1, Math.max( 0, weight ) ) );
			if ( ! ranged ) attr.clearUpdateRanges();
			if ( touched ) attr.needsUpdate = true;

		}

		h.edges = edges;
		h.weight = weight;
		h.fade = 0;
		this.u.hlFade.value = 0;
		this._tubeDirty = true;
		this._easeShadows();

	}

	/**
	 * Tubes are about to ease in or out (their radius follows the eased highlight on the GPU): keep
	 * the shadow map following for as long as that takes. The state eases within a few hundred
	 * milliseconds of app time; the frame count covers frame rates where app time runs slow.
	 */
	_easeShadows() {

		this._shadowUntil = Math.max( this._shadowUntil, this._time + Math.max( this.tubeLinger, 0.6 ) );
		this._shadowFrames = 24;

	}

	/** Tubes cast shadows: ask shadow maps that update on demand for a new frame. */
	_requestShadowUpdate() {

		const app = this.app;
		if ( ! app || ! app.scene || ! this._shadowsInvolved ) return;
		// the Environment owns the key light and its cached shadow map; without one (dev pages with
		// their own light) ask the scene's shadow-casting lights directly
		const env = app.modules && app.modules.find( ( m ) => m.isEnvironment );
		if ( env && env.updateShadow ) return void env.updateShadow();
		for ( const child of app.scene.children ) if ( child.isLight && child.castShadow && child.shadow ) child.shadow.needsUpdate = true;

	}

	// ── internals: materials ───────────────────────────────────────────────────────────────

	/** Picks (and caches) the material variants for the active pipeline configuration (gfxState). */
	_applyVariant() {

		if ( ! this.mesh || ! this.graph ) return;
		const lit = gfxState.lit !== false;
		const velocity = gfxState.velocity ? this._strandVelocity : 'off';
		const flow = this._flow === 'all' || ( this._flow === 'selection' && ! this._tubes );
		const sKey = `${lit ? 'lit' : 'flat'}|${velocity}|${flow ? 'flow' : 'still'}`;
		let strand = this._strandMaterials.get( sKey );
		if ( ! strand ) {

			strand = this._createStrandMaterial( { lit, velocity, flow } );
			this._strandMaterials.set( sKey, strand );

		}

		this.mesh.material = strand;
		this.material = strand;

		const a2c = gfxState.samples > 1;
		const tKey = `${lit ? 'lit' : 'flat'}|${a2c ? 'a2c' : 'cut'}|${gfxState.velocity ? 'vel' : 'novel'}`;
		if ( this.tubeIndex ) {

			let tube = this._tubeMaterials.get( tKey );
			if ( ! tube ) {

				tube = this._createTubeMaterial( { lit, a2c, velocity: !! gfxState.velocity } );
				this._tubeMaterials.set( tKey, tube );

			}

			this.tubeMesh.material = tube;

		}

		this._variant = `strand ${sKey} · tube ${tKey}`;

	}

	_disposeMaterials() {

		for ( const m of this._strandMaterials.values() ) m.dispose();
		for ( const m of this._tubeMaterials.values() ) m.dispose();
		this._strandMaterials.clear();
		this._tubeMaterials.clear();
		this.material = null;
		this._applyVariant();

	}

	_disposeGPU() {

		for ( const mesh of [ this.mesh, this.tubeMesh ] ) if ( mesh ) mesh.removeFromParent();
		this.mesh = null;
		this.tubeMesh = null;
		for ( const m of this._strandMaterials.values() ) m.dispose();
		for ( const m of this._tubeMaterials.values() ) m.dispose();
		this._strandMaterials.clear();
		this._tubeMaterials.clear();
		this.material = null;
		for ( const g of [ this.geometry, this.tubeGeometry ] ) if ( g ) g.dispose();
		this.geometry = null;
		this.tubeGeometry = null;
		this._S = 0;
		this._strandSegments = 0;
		if ( this.tubeIndex ) this.tubeIndex.value.dispose();
		this.tubeIndex = null;
		this._tubeCapacity = 0;
		this._tubeCount = 0;

	}

	/**
	 * Shared vertex-stage reads of both layers: the curve point of this vertex, the tangent there
	 * and the per-edge state. Call inside a Fn(). `sub` (uint node) vertex pairs per cached segment
	 * lie on the Catmull-Rom spline through the samples k−1 … k+2 (C1: no corners where the cached
	 * chords meet), the tangent is its derivative. The spline passes through the samples, and with
	 * sub = 1 the vertices are the samples and the tangent their central difference. Five sample
	 * reads (the four spline points and the last sample for the curve length).
	 */
	_edgeVertex( e, sub ) {

		const { buffers, curves } = this;
		const u = this.u;
		const samples = curves.samples, stride = curves.uniforms.stride;
		e = uint( e ).toVar();
		const i = uint( positionGeometry.x ).toVar();
		const S = stride.sub( uint( 1 ) ).toVar();
		const base = e.mul( stride ).toVar();
		const at = ( k ) => samples.element( base.add( k ) );

		// segment k and the fraction f along it, in integers (f = 1 only at the last vertex)
		const k = min( i.div( sub ), S.sub( uint( 1 ) ) ).toVar();
		const f = float( i.sub( k.mul( sub ) ) ).div( float( sub ) ).toVar();
		const s1 = at( k ).toVar(), s2 = at( k.add( uint( 1 ) ) ).toVar();
		// relative to sample k (precision); the neighbours are clamped at the two ends
		const d0 = at( max( k, uint( 1 ) ).sub( uint( 1 ) ) ).xyz.sub( s1.xyz ).toVar();
		const d2 = s2.xyz.sub( s1.xyz ).toVar();
		const d3 = at( min( k.add( uint( 2 ) ), S ) ).xyz.sub( s1.xyz ).toVar();
		const a = d2.sub( d0 ).toVar();
		const b = d0.mul( 2 ).add( d2.mul( 4 ) ).sub( d3 ).toVar();
		const c = d3.sub( d0 ).sub( d2.mul( 3 ) ).toVar();
		const position = s1.xyz.add( a.add( b.add( c.mul( f ) ).mul( f ) ).mul( f ).mul( 0.5 ) ).toVar();
		const tangent = a.add( b.mul( f.mul( 2 ) ) ).add( c.mul( f.mul( f ).mul( 3 ) ) ).toVar(); // 2 × d position / d f
		const t = float( k ).add( f ).div( float( S ) ).toVar();
		const arc = mix( s1.w, s2.w, f );
		// second difference at sample k: 8 × the gap between a chord and the smooth curve (used where sub = 1)
		const bend = d0.add( d2 );
		const first = i.equal( uint( 0 ) ), last = i.greaterThanEqual( S.mul( sub ) );

		const total = at( S ).w; // arc length of the whole curve

		const edge = buffers.edgeInfo.element( e ).toVar();
		const attr = buffers.edgeAttr.element( e ).toVar();
		const stS = buffers.nodeState.element( edge.x ).toVar();
		const stT = buffers.nodeState.element( edge.y ).toVar();
		// highlight: selected node (eased on the GPU) or the highlighted set (eased uniform)
		const hl = clamp( max( max( stS.x, stT.x ), attr.y.mul( u.hlFade ) ), 0, 1 ).toVar();
		const hov = max( stS.w, stT.w ).toVar();

		// colour: OKLab gradient with restored chroma, or the source colour
		const nS = buffers.nodeColor.element( edge.x ).toVar();
		const nT = buffers.nodeColor.element( edge.y ).toVar();
		const cS = nS.xyz;
		const grad = gradientColor( cS, nT.xyz, t, u.chroma );
		const color = mix( grad, cS, u.sourceColor ).toVar();
		// world radii of the two end nodes as the node renderer draws them (selected ones are larger)
		const radii = vec2( nS.w.mul( stS.x.mul( 0.5 ).add( 1 ) ), nT.w.mul( stT.x.mul( 0.5 ).add( 1 ) ) ).mul( u.nodeRadius );

		const widthFactor = attr.x.mul( mix( u.metricWidth.x, u.metricWidth.y, attr.z ) ).toVar();
		return { i, S, t, position, tangent, bend, arc, total, edge, attr, hl, hov, color, widthFactor, first, last, radii };

	}

	_createStrandMaterial( { lit, velocity, flow } ) {

		const { buffers } = this;
		const u = this.u;
		const U = buffers.uniforms;
		const G = gfxUniforms;
		const vColor = varyingProperty( 'vec4', 'vEdgeColor' ); // rgb × brightness, alpha
		const vProfile = varyingProperty( 'vec4', 'vEdgeProfile' ); // signed px, half width px, highlight, defocus softness
		const vTan = varyingProperty( 'vec4', 'vEdgeTan' ); // world tangent, seed
		const vView = varyingProperty( 'vec4', 'vEdgeView' ); // world vector to the camera, arc length
		const vFx = varyingProperty( 'vec4', 'vEdgeFx' ); // fog transmittance, flow gain, flow phase, px past the nearer end
		const vVel = varyingProperty( 'vec2', 'vEdgeVel' );

		const material = new THREE.MeshBasicNodeMaterial( {
			transparent: true,
			depthWrite: false,
			depthTest: true,
			side: THREE.DoubleSide,
			blending: this._blendMode === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending,
		} );
		material.name = 'EdgeRenderer.strand';
		material.fog = false;
		material.forceSinglePass = true; // transparent + DoubleSide is otherwise drawn twice

		material.vertexNode = Fn( () => {

			const side = positionGeometry.y;
			const v = this._edgeVertex( instanceIndex, u.strandSub );
			const { position, tangent, hl, hov } = v;

			// opacity and brightness (focus fades the context, highlight / hover boost, metric)
			// (a highlighted set, e.g. a path, dims the rest like a selection does, by look.setContext)
			const focus = clamp( max( U.focus, u.hlFade.mul( u.setContext ) ), 0, 1 );
			const ctx = mix( float( 1 ), u.contextOpacity, focus.mul( hl.oneMinus() ) );
			const a0 = u.opacity.mul( ctx ).mul( mix( u.metricOpacity.x, u.metricOpacity.y, v.attr.z ) );
			const a1 = mix( a0, max( a0, u.highlightOpacity ), hl );
			const alpha = min( a1.mul( hov.mul( u.hoverBoost ).add( 1 ) ), 1 ).toVar();
			// (the glow of a selection's bundle is shared out between its edges, see uniforms.crowd)
			const bright = u.brightness.mul( hl.mul( u.highlightGlow ).mul( u.crowd ).add( hov.mul( u.hoverBoost ).mul( 0.5 ) ).add( 1 ) );

			// projection and screen-space expansion (the buffers are world space; the mesh transform
			// is ignored, which also saves a per-vertex mat4 × mat4)
			const clip = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( position, 1 ) ) ).toVar();
			const dclip = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( tangent, 0 ) ) ).toVar();
			const vp = viewport.zw;
			const sT = dclip.xy.mul( clip.w ).sub( clip.xy.mul( dclip.w ) ).mul( vp ).toVar();
			const sLen = length( sT );
			const dir = select( sLen.greaterThan( 1e-30 ), sT.div( max( sLen, 1e-30 ) ), vec2( 1, 0 ) );
			const normal = vec2( dir.y.negate(), dir.x );

			const projScale = abs( cameraProjectionMatrix.element( 1 ).element( 1 ) ).mul( vp.y ).mul( 0.5 );
			const projected = u.halfWidth.mul( v.widthFactor ).mul( projScale ).div( max( clip.w, 1e-5 ) );
			const px = G.pxScale;
			const widthBoost = hl.mul( 0.5 ).add( hov.mul( 0.25 ) ).add( 1 );
			// emphasis widens a strand, but only a quarter past the limit: near the camera strands stay lines
			const maxHalf = u.maxPx.mul( px ).mul( 0.5 );
			const half0 = min( clamp( projected, u.minPx.mul( px ).mul( 0.5 ), maxHalf ).mul( widthBoost ), maxHalf.mul( max( hl, hov ).mul( 0.25 ).add( 1 ) ) ).toVar();
			const coverage = clamp( half0.mul( 2 ), 0, 1 );
			const halfSharp = max( half0, 0.5 ).toVar();
			const { first, last } = v;

			// hand-over: where the edge is drawn as a tube (highlight, style 'solid') the strand fades;
			// a highlighted strand keeps a soft sheath around its tube. Both follow the same spline;
			// only a strand that cannot be subdivided (sheathFit) is a chain of chords, and where those
			// leave the tube by about a tube radius the sheath would show as a second line: it goes there
			const tubeHalf = max( projected.mul( u.tubeRadius ), u.tubeMinPixelWidth.mul( px ).mul( 0.5 ) );
			const gap = select( first.or( last ), float( 0 ), length( v.bend ).mul( 0.125 ) ).mul( projScale ).div( max( clip.w, 1e-5 ) );
			const sheath = u.sheath.mul( smoothstep( 0.4, 1.2, gap.div( tubeHalf ) ).mul( u.sheathFit ).oneMinus() );
			const grown = smoothstep( 0, 1, hl );
			const keep = max( grown.mul( u.handover ).mul( sheath.oneMinus() ), u.solid ).oneMinus();

			// analytic depth of field: wider by the circle of confusion, fainter by the width ratio.
			// The circle is capped (defocusLimit): every pixel of width is blended fill
			const coc = min( cocPixels( clip.w ), u.defocusMax.mul( px ) ).toVar();
			const halfPx = halfSharp.add( coc ).toVar();
			const geomHalf = halfPx.add( 0.5 ); // AA pad: the profile reaches 0 at halfPx + 0.5
			// a defocused strand also grows past its two end points by the circle of confusion and
			// fades there (soft ends instead of square stubs); distance from the nearer end in px
			const along = dir.mul( select( first, coc.negate(), select( last, coc, float( 0 ) ) ) );
			const toEnd = min( v.arc, v.total.sub( v.arc ) ).mul( projScale ).div( max( clip.w, 1e-5 ) );
			const offset = normal.mul( side.mul( geomHalf ) ).add( along ).mul( 2 ).div( vp );

			vColor.assign( vec4( v.color.mul( bright ), alpha.mul( coverage ).mul( halfSharp.div( halfPx ) ).mul( keep ) ) );
			vProfile.assign( vec4( side.mul( geomHalf ), halfPx, hl, coc.div( halfPx ) ) );
			const tLen = length( tangent );
			vTan.assign( vec4( tangent.div( max( tLen, 1e-12 ) ), v.attr.w ) );
			vView.assign( vec4( normalize( cameraPosition.sub( position ) ), v.arc ) );
			vFx.assign( vec4( fogTransmittance( clip.w ), max( u.flowAll, u.flowSel.mul( hl ) ), v.attr.w.mul( u.flowJitter ), select( first.or( last ), coc, toEnd.negate() ) ) );

			if ( velocity === 'camera' ) vVel.assign( motionVector( position, position ) );
			else if ( velocity === 'true' ) {

				// end points' motion since the previous frame, interpolated along the edge
				const dS = buffers.nodePos.element( v.edge.x ).xyz.sub( buffers.nodePrev.element( v.edge.x ).xyz );
				const dT = buffers.nodePos.element( v.edge.y ).xyz.sub( buffers.nodePrev.element( v.edge.y ).xyz );
				vVel.assign( motionVector( position, position.sub( mix( dS, dT, v.t ) ) ) );

			}

			// max( w, 0 ): a ribbon crossing the camera plane tapers instead of twisting
			return vec4( clip.xy.add( offset.mul( max( clip.w, 0 ) ) ), clip.z, clip.w );

		} )();

		// ── fragment ──
		const d = abs( vProfile.x );
		const halfPx = vProfile.y;
		const soft = vProfile.w;
		const r = min( d.div( halfPx ), 1 );
		// cross profile: 1 px box filter of the edge; a bell of the same energy when defocused
		const sharp = clamp( halfPx.add( 0.5 ).sub( d ), 0, 1 );
		const x = min( d.div( halfPx.add( 0.5 ) ), 1 );
		const x2 = x.mul( x ).oneMinus();
		const bell = x2.mul( x2 ).mul( 1.875 ).mul( halfPx.div( halfPx.add( 0.5 ) ) );
		// along the strand: full inside, half at the end point, gone one circle of confusion past it
		const ends = smoothstep( 0, 1, vFx.w.div( max( soft.mul( halfPx ).mul( 2 ), 1 ) ).oneMinus().sub( 0.5 ) );
		const profile = mix( sharp, bell, soft ).mul( ends );
		// wide (highlighted) strands are shaded as tubes, with a hot core
		const tube = mix( float( 0.55 ), float( 1 ), sqrt( max( r.mul( r ).oneMinus(), 0 ) ) );
		const shade = mix( float( 1 ), tube, smoothstep( 0.6, 1.8, halfPx ).mul( soft.oneMinus() ) );
		const q = r.oneMinus();
		const core = q.mul( q ).mul( q ).mul( vProfile.z ).mul( u.highlightGlow ).mul( 0.6 );
		const base = vColor.xyz.mul( shade.add( core ) );

		let rgb;
		if ( lit ) {

			const T = normalize( vTan.xyz );
			const V = normalize( vView.xyz );
			const light = strandLight( T, V, { gloss: u.gloss, shift: vTan.w.sub( 0.5 ).mul( u.shift ) } );
			const peak = max( max( vColor.x, vColor.y ), max( vColor.z, 1e-4 ) );
			const tint = mix( vec3( 1 ), vColor.xyz.div( peak ), u.sheenTint );
			rgb = base.mul( u.emissive.add( light.diffuse.mul( u.diffuse ) ) ).add( light.specular.mul( tint ).mul( u.sheen ) );

		} else rgb = base;

		let alpha = vColor.w.mul( profile );
		if ( flow ) {

			const pulse = flowPulse( vView.w, vFx.z, { time: U.time, speed: u.flowSpeed, spacing: u.flowSpacing, tail: u.flowTail } ).mul( vFx.y );
			rgb = rgb.mul( pulse.mul( u.flowGain ).add( 1 ) );
			alpha = min( alpha.mul( pulse.mul( 0.75 ).add( 1 ) ), 1 );

		}

		// fog: normal blending fades toward the fog colour, additive fades out
		const fogT = vFx.x;
		material.colorNode = rgb.mul( fogT ).add( G.fogColor.mul( fogT.oneMinus() ).mul( u.additive.oneMinus() ) );
		material.opacityNode = alpha;

		if ( velocity === 'camera' || velocity === 'true' ) material.mrtNode = mrt( { output, velocity: vec4( vVel, 0, 1 ) } );
		else if ( velocity === 'skip' ) material.mrtNode = skipVelocity();
		return material;

	}

	_createTubeMaterial( { lit, a2c, velocity } ) {

		const { buffers } = this;
		const u = this.u;
		const U = buffers.uniforms;
		const G = gfxUniforms;
		const tubeIndex = this.tubeIndex;
		const vViewPos = varyingProperty( 'vec3', 'vTubeViewPos' ); // ribbon point, view space
		const vTangent = varyingProperty( 'vec3', 'vTubeTangent' ); // view space
		const vProfile = varyingProperty( 'vec4', 'vTubeProfile' ); // signed px, half width px, world radius as drawn, sub-pixel coverage
		const vColor = varyingProperty( 'vec4', 'vTubeColor' ); // gradient colour, emission factor
		const vFx = varyingProperty( 'vec4', 'vTubeFx' ); // arc length, flow gain, flow phase, colour factor

		const material = lit ? new TubeNodeMaterial() : new THREE.MeshBasicNodeMaterial();
		material.name = 'EdgeRenderer.tube';
		material.fog = false;
		material.transparent = false;
		material.side = THREE.DoubleSide; // the winding depends on the screen direction of the curve
		material.shadowSide = THREE.DoubleSide;
		material.alphaToCoverage = a2c;

		material.positionNode = Fn( () => {

			const side = positionGeometry.y;
			const v = this._edgeVertex( tubeIndex.element( instanceIndex ), u.tubeSub );
			const { position, tangent, hl, hov } = v;

			// how much of a tube this edge is: eased highlight, or the style
			const grow = smoothstep( 0, 1, max( hl, u.solid ) ).toVar();

			const C = cameraViewMatrix.mul( vec4( position, 1 ) ).xyz.toVar();
			const Tv = cameraViewMatrix.mul( vec4( tangent, 0 ) ).xyz.toVar();
			// screen-space direction of the curve (exact derivative of the projection)
			const clip = cameraProjectionMatrix.mul( vec4( C, 1 ) ).toVar();
			const dclip = cameraProjectionMatrix.mul( vec4( Tv, 0 ) ).toVar();
			const sT = dclip.xy.mul( clip.w ).sub( clip.xy.mul( dclip.w ) ).mul( viewport.zw ).toVar();
			const sLen = length( sT );
			const dir = select( sLen.greaterThan( 1e-30 ), sT.div( max( sLen, 1e-30 ) ), vec2( 1, 0 ) );
			const across = vec2( dir.y.negate(), dir.x );

			// radius in px: world radius, clamped (no pixel minimum in the shadow pass); grows with
			// the highlight; +0.5 px per side for the analytic silhouette
			const ppu = pixelsPerUnit( C.z ).toVar();
			const px = G.pxScale;
			const radius = u.halfWidth.mul( v.widthFactor ).mul( u.tubeRadius ).mul( hov.mul( 0.25 ).add( 1 ) );
			const minHalf = select( isOrthographic(), float( 0 ), u.tubeMinPixelWidth.mul( px ).mul( 0.5 ) );
			// ends: from three node radii out down to the node's surface the tube thins to tubeTaper of
			// its radius, so the tubes of a hub arrive as fine lines instead of covering the node
			const nearS = smoothstep( v.radii.x, v.radii.x.mul( 3 ), v.arc );
			const nearT = smoothstep( v.radii.y, v.radii.y.mul( 3 ), v.total.sub( v.arc ) );
			const ends = min( nearS, nearT ).toVar();
			const taper = mix( u.tubeTaper, float( 1 ), ends );
			const full = clamp( radius.mul( ppu ), minHalf, max( u.tubeMaxPixelWidth.mul( px ).mul( 0.5 ), minHalf ) ).toVar();
			const half0 = full.mul( taper ).mul( grow ).toVar();
			const halfPx = max( half0, 0.5 );
			// an edge that is no tube (yet) collapses to nothing
			const geomHalf = select( grow.greaterThan( 0.01 ), halfPx.add( 0.5 ), float( 0 ) );
			// a pixel offset at constant depth is a view-space xy offset of px / ppu (any camera)
			const viewPos = C.add( vec3( across, 0 ).mul( side.mul( geomHalf ).div( ppu ) ) ).toVar();

			// colours: context dimming (style 'solid' with a selection), emission by highlight
			const focus = clamp( max( U.focus, u.hlFade.mul( u.setContext ) ), 0, 1 );
			const dim = mix( float( 1 ), u.contextDim, focus.mul( hl.oneMinus() ) );
			// emission: highlighted tubes share the glow of their bundle (crowd) and lose most of it at
			// their ends; a 'solid' tube glows more the thinner it is on screen, so an overview keeps the
			// luminous look and a close-up shows lit surfaces
			const thin = clamp( float( 1 ).sub( full.mul( 2 ).div( px ).div( 3 ) ), 0, 1 );
			const glowHl = u.tubeGlow.mul( u.crowd ).mul( mix( u.tubeTaper.mul( 0.7 ).add( 0.3 ), float( 1 ), ends ) );
			const glow = mix( mix( u.solidGlow, u.solidGlowThin, thin ), glowHl, hl ).mul( hov.mul( u.hoverBoost ).add( 1 ) );

			vViewPos.assign( viewPos );
			vTangent.assign( Tv );
			vProfile.assign( vec4( side.mul( geomHalf ), halfPx, halfPx.div( ppu ), clamp( half0.mul( 2 ), 0, 1 ) ) );
			vColor.assign( vec4( v.color, glow ) );
			vFx.assign( vec4( v.arc, max( u.flowAll, u.flowSel.mul( hl ) ), v.attr.w.mul( u.flowJitter ), dim ) );
			return viewToWorld( viewPos );

		} )();

		// ── fragment: cylinder frame ──
		const frame = tubeFrame( vViewPos, vTangent, vProfile.x, vProfile.y, vProfile.z );
		const coverage = frame.edge.mul( vProfile.w ).toVar( 'tubeCoverage' );
		const dim = vFx.w;
		const gradient = vColor.xyz.mul( dim );
		const pulse = flowPulse( vFx.x, vFx.z, { time: U.time, speed: u.flowSpeed, spacing: u.flowSpacing, tail: u.flowTail } ).mul( vFx.y );
		const emission = gradient.mul( u.brightness ).mul( vColor.w.add( pulse.mul( u.flowGain ) ) );
		const viewDepth = frame.hit.z.negate();

		// alpha-to-coverage turns the coverage into sub-sample coverage; otherwise a hard cut at 50 %
		// (of the coverage, not of the silhouette ramp: a tube that eases in is not drawn as an
		// opaque 1 px line before it covers half a pixel, by which time its strand is bright)
		material.maskNode = coverage.greaterThan( a2c ? 0.002 : 0.5 );
		// the shadow pass keeps the silhouette ramp: tubes thinner than a shadow texel still cast
		material.maskShadowNode = frame.edge.greaterThan( 0.5 );
		if ( a2c ) material.opacityNode = coverage;

		let hitWorld;
		if ( lit ) {

			hitWorld = applySurface( material, { hit: frame.hit, normal: frame.normal, viewDir: frame.V } );
			material.colorNode = vec4( gradient.mul( u.tubeAlbedo ), 1 );
			material.roughnessNode = u.tubeRoughness;
			material.metalnessNode = u.tubeMetalness;
			material.emissiveNode = emission;
			material.fogTransmittanceNode = fogTransmittance( viewDepth );

		} else {

			// no lights, no taps: Blinn-Phong on the cylinder normal from the shared key light
			hitWorld = viewToWorld( frame.hit ).toVar( 'impHitWorld' );
			material.depthNode = viewToDepth( frame.hit );
			const N = frame.normal;
			const Lv = normalize( cameraViewMatrix.mul( vec4( G.keyDir, 0 ) ).xyz );
			const H = normalize( Lv.add( frame.V ) );
			const ambient = mix( G.ambientGround, G.ambientSky, cameraWorldMatrix.mul( vec4( N, 0 ) ).y.mul( 0.5 ).add( 0.5 ) );
			const diffuse = G.keyColor.mul( max( dot( N, Lv ), 0 ) ).add( ambient ).mul( 1 / Math.PI );
			const specular = G.keyColor.mul( pow( max( dot( N, H ), 0 ), 48 ).mul( 0.12 ) );
			const rgb = gradient.mul( u.tubeAlbedo ).mul( diffuse ).add( specular ).add( emission );
			const fogT = fogTransmittance( viewDepth );
			material.colorNode = vec4( rgb.mul( fogT ).add( G.fogColor.mul( fogT.oneMinus() ) ), 1 );

		}

		if ( velocity ) material.mrtNode = velocityOutput( motionVector( hitWorld, hitWorld ) );
		return material;

	}

}
