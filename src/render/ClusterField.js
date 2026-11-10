// Community field (App module, order 13): one soft glow volume per community of one hierarchy
// level. At overview distance a graph is a few pixels of colour per node; the volumes turn the
// community structure, which the layout and the bundling are built around, into shape and space.
//
// ─── Technique ──────────────────────────────────────────────────────────────────────────────
//  One instanced quad per community, drawn in the scene pass (HDR, before tone mapping).
//  Vertex stage: the quad is the tangent billboard of the bounding sphere (impostor.js geometry);
//  when the camera is inside or close to the sphere it becomes a full-screen quad, so flying
//  through a community never pops.
//  Fragment stage: the sphere holds an emissive medium with the smooth radial density
//      ρ( r ) = ( 1 − r² / R² )²    ( r < R, zero outside; C¹ at the border )
//  and the fragment integrates it ANALYTICALLY along its view ray. With the closest approach of
//  the ray to the centre at distance b, a = 1 − b² / R², and s the position along the ray
//  measured from that closest point ( x = s / R ):
//      ∫ ρ ds = R · [ a² x − ⅔ a x³ + ⅕ x⁵ ]  between the entry and the exit of the ray,
//  which is ( 16 / 15 ) · R · a^(5/2) for a ray that crosses the whole sphere, and less when the
//  camera sits inside (the integral then starts at the camera). τ = that / ( 16 R / 15 ) in 0..1 is
//  the share of the densest possible ray. The result is a perspective-correct volume: its profile
//  follows the true chord lengths, it thins out as the camera enters it, and two volumes that
//  overlap in depth add up like media, not like stickers.
//  Colour and blending: premultiplied ( ONE, ONE_MINUS_SRC_ALPHA, set as custom blending because
//  three multiplies the colour of a `premultipliedAlpha` material by its alpha itself ), rgb =
//  colour · intensity · ( τ + core · τ⁴ ), alpha = absorb · τ. Every volume therefore adds at most
//  `intensity` ( FIELD_RADIANCE · strength · tier factor ) of its (luminance-normalised) colour and
//  takes `absorb` of what is behind it away: a stack of volumes converges to
//  intensity / absorb instead of growing without bound, so overlaps stay luminous but can never
//  wash out to white (the overdraw budget below keeps real stacks at a few volumes anyway). `core`
//  adds a narrow HDR centre that reaches the bloom threshold: the field feeds the bloom a little,
//  on purpose (uniforms.core, 0 = none).
//  Depth: the fragment's depth is the point where its ray LEAVES the sphere, tested against the
//  scene depth (no depth write). Nodes inside or in front of a volume therefore hide the glow at
//  their own pixels and keep their full contrast; only what lies behind a volume is veiled by it.
//  That veil has an edge in depth (a bead just behind the exit surface is veiled, one just inside
//  is not), which is why the absorption is small (0.15) and recedes with the light: in close-ups
//  and while the field steps back (below), so nothing darkens or pops when the volumes go.
//  The mesh is drawn before the edges (renderOrder −5), so strands blend over the glow.
//  Close-ups: a volume that fills the view (camera inside it or right in front) would be a
//  coloured haze over the very nodes one came to look at; its light and its absorption fall to
//  1 − closeFade as the ratio radius / distance goes from 0.3 to 1.1.
//  Fog: the shared fog (gfx/shared.js) at the centre's distance. Motion vectors: skipVelocity().
//
// ─── An overview feature ────────────────────────────────────────────────────────────────────
//  The volumes say "these are the communities" while a community is a SHAPE in the view. Two
//  situations in which they say nothing, and in which the field therefore steps back by itself
//  (`adaptive`, on by default):
//   inside   Once the volumes one looks at are hazes that fill the view (the camera is inside the
//            graph or close to a community), they only lift the blacks between the strands. The
//            measure is the size of the volume behind a typical field pixel, Σ c² / Σ c with c the
//            share of the view a volume's quad covers (typicalCoverage(); measured at 1280 × 800
//            on lesmis, hsbm-500 and lfr-2k: 0.01…0.02 from afar, 0.04…0.09 with the graph fitted
//            to the view, 0.33…0.42 halfway into the graph, 0.8…0.9 in front of one community;
//            it does not grow with the number of volumes). The light of ALL volumes eases out
//            between 0.12 and 0.32 (overviewFactor()); at 0 the mesh is not drawn and the kernel
//            does not run, so inside views cost nothing.
//   size     From about 5000 nodes on the point sprites of the nodes form the coloured cloud
//            themselves and a volume around it adds a fringe at best: the default strength falls
//            to 0 between 5000 and 20000 nodes (fieldGraphFactor(); it also fades the field in
//            between 16 and 48 nodes: tiny graphs have nothing to group). The volume of a focused
//            or selected community is exempt (FIELD_FOCUS_KEEP of the neutral strength), because
//            there the volume answers a question: "where is its community?".
//
// ─── Data ───────────────────────────────────────────────────────────────────────────────────
//  centre   buffers.clusterPos (the live mass-weighted centroid on the GPU), low-pass filtered by
//           a tiny compute kernel (K elements; time constant `smoothing`), so the volumes follow a
//           running layout without lag in the long run and without jitter in the short one
//  radius   radius · ( robust radius of the members ), the RMS distance to the centroid measured
//           on the CPU from the position readback (≤ 8 Hz, only while positions change) and eased
//           by the same kernel; without a readback: 0.62 · clusterPos.w (the bounding radius)
//  colour   the mean of the members' CURRENT node colour targets, luminance-normalised. Without a
//           lens that is the palette colour at the middle of the cluster's rank range; with a
//           colour lens it is the community's mean lens colour. A volume in the palette hue under
//           nodes coloured by a metric would be a second, competing colour code, while the mean
//           lens colour reads as the aggregate ("this community is high") and keeps the picture
//           in one key. With a lens the saturation and strength are reduced (×0.7 / ×0.75), so
//           the bright end of a ramp does not turn its community into a lamp.
//  emphasis the community of the selected node and a focused community ( focus(), the 'cluster'
//           event ) glow stronger (×1.6), the others recede (×0.4); eased by the kernel
//  level    'auto' = chooseFieldLevel(): the hierarchy level with a readable number of volumes
//           (4…40, nearest to 10), else the level nearest to that range. At most `maxVolumes`
//           volumes are drawn (the largest communities of the level).
//  sphere   in the sphere layout (uniforms.layoutBlend) the centres are pushed out onto the sphere,
//           where the clusters are
//
// ─── Cost ───────────────────────────────────────────────────────────────────────────────────
//  K quads (≤ maxVolumes: 128 in high / ultra, 48 in medium; 'auto' picks a level with 4…40),
//  one draw, no texture taps, about 45 ALU per fragment, blended into the HDR target (and, in the
//  velocity tiers, a blended no-op write to the velocity attachment).
//  Covered pixels: volumes are translucent and overlap in depth, so the quads of K communities
//  that fill a ball cover the graph's screen area several times (measured in overviews at
//  1280 × 800: 0.3 views for 12 volumes, 0.6 for 9, 1.4 for 42), and a camera on its way into the
//  graph pays one whole view for every volume it is in or right in front of until the field has
//  stepped back (see above). That is bounded by an OVERDRAW BUDGET in views (FIELD_TIERS:
//  medium 2, high 3, ultra 3; maxOverdraw), applied on the CPU every frame from the readback
//  centres (fieldBudget(), K ≤ 128):
//   1. above the budget all volumes first get tighter, down to FIELD_MIN_SHRINK of their radius
//      (eased; halves the covered area, nothing disappears: the case of a dense overview);
//   2. then volumes are drawn in the order emphasised, smallest on screen first, while their sum
//      stays in the budget. Far volumes are the shapes that carry the structure and cost little;
//      what is left out are the largest hazes around the camera, of which three say as much as
//      twenty (their light eases out, then the quad collapses in the vertex stage).
//  field.stats reports quads, covered pixels, overdraw, the budget and what it left out.
//  strength 0, an invisible field, the low tier, a view from inside or a graph that is too large
//  for it: the mesh is not drawn and the kernel does not run (the CPU keeps the K coverages of
//  the view up to date, nothing else). Storage buffers: kernel 5, vertex stage 2, fragment stage 0.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//   const field = app.use( new ClusterField( { strength = 0.5, level = 'auto' } ) )
//   field.strength       0..1, 0 = off (not drawn). The tier and the graph scale it (see TIERS)
//   field.level          'auto' | hierarchy height (1 = finest communities … D = coarsest)
//   field.visible
//   field.focus( cluster | null )   emphasise one community: an index in the prepared graph's
//                        cluster space (any level: a finer cluster lights the volume it belongs to,
//                        a coarser one all volumes inside it). Also driven by the 'cluster' event
//                        { cluster: index | −1 } on app.events.
//   field.radius         size of a volume in robust (RMS) radii of its members (default 1.5: the
//                        glow is at half its peak 0.74 rms radii from the centre)
//   field.core           0..1 HDR centre that feeds the bloom (default 0.35)
//   field.absorb         0..1 share of the light behind a volume that it takes away (default 0.15)
//   field.saturation     0..1.5 (default 0.8)
//   field.closeFade      0..1 share of its light a volume loses while it fills the view (default 0.6)
//   field.smoothing      time constant of the response to a moving layout, seconds (default 0.18)
//   field.maxVolumes     upper bound of drawn volumes (null = tier default)
//   field.maxOverdraw    overdraw budget in views (null = tier default, see "Cost")
//   field.adaptive       true (default): the field steps back inside the graph and on large graphs
//                        (see "An overview feature"); false: strength × tier only
//   field.lowTier        draw in the low tier as well (default false)
//   field.height         resolved hierarchy height (0 = nothing to draw), read only
//   field.clusters       Uint32Array of the cluster indices that have a volume, read only
//   field.snap()         jump to the current targets (no easing), e.g. before a screenshot
//   field.stats          { height, count, quads, coveredPixels, overdraw (views), intensity, budget,
//                          culled, shrink, typical (size of the typical volume, share of the view),
//                          overview (0..1, the eased factor of the view), graph (factor of the
//                          graph size) } for the view of the last update
//   field.volumes()      [ { cluster, size, radius, compact, emphasis, coverage, drawn } ] per volume
//   field.setGraph( buffers, graph, app ), field.update( dt, app ), field.dispose()
//
// Pure helpers (tested in test/atmosphere-field.test.js): chooseFieldLevel, levelCounts,
// densityIntegral, opticalDepth, clusterStats, clusterColors, fieldColor, compactness,
// emphasisTargets, quadCoverage, fieldBudget, typicalCoverage, overviewFactor, fieldGraphFactor.
import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, instancedArray, uniform, varyingProperty, positionGeometry, float, vec3, vec4, uint,
	cameraViewMatrix, cameraProjectionMatrix, cameraNear, normalize, dot, cross, sqrt, max, min, clamp, mix, select, length, abs, smoothstep,
} from 'three/tsl';
import { gfxUniforms, gfxState, skipVelocity, fogTransmittance } from './gfx/shared.js';
import { viewToDepth, quadGeometry } from './gfx/impostor.js';
import { readOnly } from '../gpu/forceKernels.js';

/**
 * Per quality tier: strength factor, the largest number of volumes, and the overdraw budget in
 * views (the sum of the quads drawn, in units of the scene target's area; see fieldBudget()).
 */
export const FIELD_TIERS = {
	low: { factor: 0.8, max: 24, overdraw: 1.5 }, // only when field.lowTier is set
	medium: { factor: 0.85, max: 48, overdraw: 2 },
	high: { factor: 1, max: 128, overdraw: 3 },
	ultra: { factor: 1, max: 128, overdraw: 3 },
};
/** Peak radiance (linear, scene referred) of a neutral volume at strength 1 in the high tier, without the core. */
export const FIELD_RADIANCE = 0.08;
/** When the volumes exceed the budget they first shrink, down to this share of their radius. */
export const FIELD_MIN_SHRINK = 0.75;
/** Share of the neutral strength that the volume of a focused / selected community keeps on graphs too large for the field. */
export const FIELD_FOCUS_KEEP = 0.6;

// ── pure helpers ─────────────────────────────────────────────────────────────────────────────

/** Number of clusters per hierarchy height: [ K(height 1), …, K(height D) ]. */
export function levelCounts( graph ) {

	const out = [];
	for ( let l = 0; l < graph.D; l ++ ) out.push( graph.levelOffset[ l + 1 ] - graph.levelOffset[ l ] );
	return out;

}

/**
 * The hierarchy height (1-based; 0 = none) whose number of communities reads best as volumes or
 * names: inside [ min, max ] and nearest to `ideal` on a log scale; when no level is in the range,
 * the one nearest to it. Levels with fewer than 2 communities say nothing and are never chosen.
 * @param {number[]} counts clusters per height, finest first (levelCounts())
 */
export function chooseFieldLevel( counts, { min = 4, max = 40, ideal = 10 } = {} ) {

	let best = 0, bestCost = Infinity;
	for ( let l = 0; l < counts.length; l ++ ) {

		const k = counts[ l ];
		if ( ! ( k >= 2 ) ) continue;
		// distance to the range counts ten times the distance to the ideal inside it
		const outside = k < min ? Math.log( min / k ) : k > max ? Math.log( k / max ) : 0;
		const cost = outside * 10 + Math.abs( Math.log( k / ideal ) );
		if ( cost < bestCost - 1e-9 ) {

			bestCost = cost;
			best = l + 1;

		}

	}

	return best;

}

/**
 * ∫ ρ ds of the density ρ( r ) = ( 1 − r² / R² )² along a line that passes the centre at distance
 * `impact`, between the positions s0 and s1 along the line (s = 0 at the closest point), clipped to
 * the sphere. World units (density 1 at the centre). The whole chord gives 16/15 · R · a^(5/2).
 */
export function densityIntegral( impact, R, s0 = - Infinity, s1 = Infinity ) {

	const a = 1 - ( impact * impact ) / ( R * R );
	if ( ! ( a > 0 ) || ! ( R > 0 ) ) return 0;
	const h = R * Math.sqrt( a );
	const lo = Math.max( - h, s0 ), hi = Math.min( h, s1 );
	if ( ! ( hi > lo ) ) return 0;
	const F = ( s ) => {

		const x = s / R, x2 = x * x;
		return R * x * ( a * a - ( 2 / 3 ) * a * x2 + 0.2 * x2 * x2 );

	};

	return F( hi ) - F( lo );

}

/**
 * What the fragment shader computes: τ in 0..1 for a camera at the origin, a unit ray direction
 * `d` and a sphere ( centre C, radius R ) in the same space. 1 = a ray through the centre of a
 * sphere that lies wholly in front of the camera.
 */
export function opticalDepth( d, C, R ) {

	const tc = d[ 0 ] * C[ 0 ] + d[ 1 ] * C[ 1 ] + d[ 2 ] * C[ 2 ];
	const b2 = Math.max( 0, C[ 0 ] * C[ 0 ] + C[ 1 ] * C[ 1 ] + C[ 2 ] * C[ 2 ] - tc * tc );
	return densityIntegral( Math.sqrt( b2 ), R, - tc, Infinity ) / ( ( 16 / 15 ) * R );

}

/**
 * Per-community statistics of one hierarchy height from node positions (vec4 array).
 * @return {{ count, offset, size: Uint32Array, center: Float32Array( 3K ), rms: Float32Array( K ), graphRms, total }}
 *   indexed by k = cluster index − offset. rms = root-mean-square distance of the members to their
 *   (unweighted) centroid: the robust radius (a uniform ball of radius r has rms 0.775 r).
 *   graphRms is the same measure for all nodes together.
 */
export function clusterStats( graph, positions, height, out = null ) {

	const { N, D } = graph;
	const offset = graph.levelOffset[ height - 1 ];
	const K = graph.levelOffset[ height ] - offset;
	const o = out && out.count === K ? out : { count: K, offset, size: new Uint32Array( K ), center: new Float32Array( 3 * K ), rms: new Float32Array( K ), _sum: new Float64Array( 4 * K ) };
	o.offset = offset;
	const sum = o._sum, anc = graph.anc, l = height - 1;
	sum.fill( 0 );
	o.size.fill( 0 );
	for ( let i = 0; i < N; i ++ ) {

		const k = anc[ i * D + l ] - offset;
		const x = positions[ 4 * i ], y = positions[ 4 * i + 1 ], z = positions[ 4 * i + 2 ];
		if ( ! ( x === x && y === y && z === z ) ) continue; // NaN
		sum[ 4 * k ] += x; sum[ 4 * k + 1 ] += y; sum[ 4 * k + 2 ] += z; sum[ 4 * k + 3 ] += x * x + y * y + z * z;
		o.size[ k ] ++;

	}

	for ( let k = 0; k < K; k ++ ) {

		const n = Math.max( 1, o.size[ k ] );
		const x = sum[ 4 * k ] / n, y = sum[ 4 * k + 1 ] / n, z = sum[ 4 * k + 2 ] / n;
		o.center[ 3 * k ] = x; o.center[ 3 * k + 1 ] = y; o.center[ 3 * k + 2 ] = z;
		o.rms[ k ] = Math.sqrt( Math.max( 0, sum[ 4 * k + 3 ] / n - ( x * x + y * y + z * z ) ) );

	}

	// the whole graph: rms distance of all nodes to their centroid
	let gx = 0, gy = 0, gz = 0, g2 = 0, gn = 0;
	for ( let k = 0; k < K; k ++ ) {

		gx += sum[ 4 * k ]; gy += sum[ 4 * k + 1 ]; gz += sum[ 4 * k + 2 ]; g2 += sum[ 4 * k + 3 ]; gn += o.size[ k ];

	}

	gn = Math.max( 1, gn );
	o.total = gn;
	o.graphRms = Math.sqrt( Math.max( 0, g2 / gn - ( gx * gx + gy * gy + gz * gz ) / ( gn * gn ) ) );
	return o;

}

/** Mean linear RGB of the members of every community of a height: Float32Array( 3K ). */
export function clusterColors( graph, colors4, height, out = null ) {

	const { N, D } = graph;
	const offset = graph.levelOffset[ height - 1 ];
	const K = graph.levelOffset[ height ] - offset;
	const o = out && out.length === 3 * K ? out : new Float32Array( 3 * K );
	const n = new Uint32Array( K ), anc = graph.anc, l = height - 1;
	o.fill( 0 );
	for ( let i = 0; i < N; i ++ ) {

		const k = anc[ i * D + l ] - offset;
		o[ 3 * k ] += colors4[ 4 * i ]; o[ 3 * k + 1 ] += colors4[ 4 * i + 1 ]; o[ 3 * k + 2 ] += colors4[ 4 * i + 2 ];
		n[ k ] ++;

	}

	for ( let k = 0; k < K; k ++ ) {

		const m = Math.max( 1, n[ k ] );
		o[ 3 * k ] /= m; o[ 3 * k + 1 ] /= m; o[ 3 * k + 2 ] /= m;

	}

	return o;

}

/**
 * Glow colour of a community from the mean colour of its members: scaled to luminance 1 (so no hue
 * glows brighter than another), the largest channel limited to `maxChannel` (deep blue cannot reach
 * luminance 1 inside the gamut), then mixed toward grey by 1 − saturation.
 * @return {number[]} linear RGB
 */
export function fieldColor( r, g, b, saturation = 1, maxChannel = 5 ) {

	const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
	if ( ! ( lum > 1e-5 ) ) return [ 1, 1, 1 ];
	let k = 1 / lum;
	const top = Math.max( r, g, b ) * k;
	if ( top > maxChannel ) k *= maxChannel / top;
	const c = [ r * k, g * k, b * k ];
	const y = 0.2126 * c[ 0 ] + 0.7152 * c[ 1 ] + 0.0722 * c[ 2 ];
	return c.map( ( v ) => Math.max( 0, y + ( v - y ) * saturation ) );

}

/**
 * How much a community is a PLACE: 1 while its members sit together, falling to 0 when they are
 * spread through the graph (a volume around those would be a haze over everything that says
 * nothing). The measure is the community's rms radius relative to the radius it would have if it
 * filled exactly its share of the graph's volume, graphRms · ( n / N )^(1/3): communities that tile
 * the space give 1, the dense clumps of a good layout less, a community in two distant parts or
 * dispersed among the others clearly more.
 * @return {number} 0..1 (1 up to a ratio of `full`, 0 from `none`)
 */
export function compactness( rms, n, graphRms, N, { full = 1.25, none = 2.1 } = {} ) {

	if ( ! ( graphRms > 0 ) || ! ( n > 0 ) || ! ( N > 0 ) ) return 1;
	const ratio = rms / ( graphRms * Math.cbrt( n / N ) );
	const t = Math.min( 1, Math.max( 0, ( ratio - full ) / ( none - full ) ) );
	return 1 - t * t * ( 3 - 2 * t );

}

/**
 * Emphasis per volume: 1 neutral; with a focused community or a selection the volumes that contain
 * it (or lie inside it) get `strong`, the others `weak`.
 * @param {object} graph prepared graph
 * @param {Uint32Array} clusters cluster index per volume (all of one height)
 * @param {number} focus cluster index (any height) or −1
 * @param {number} selected node index or −1
 */
export function emphasisTargets( graph, clusters, focus, selected, { strong = 1.6, weak = 0.4 } = {}, out = new Float32Array( clusters.length ) ) {

	const K = clusters.length;
	out.fill( 1 );
	if ( K === 0 ) return out;
	const height = graph.clusterHeight[ clusters[ 0 ] ];
	let target = - 1, inside = - 1; // a cluster of the field's height, or a coarser cluster that contains volumes
	if ( focus >= 0 && focus < graph.C ) {

		let c = focus;
		while ( c >= 0 && graph.clusterHeight[ c ] < height ) c = graph.clusterParent[ c ];
		if ( c >= 0 && graph.clusterHeight[ c ] === height ) target = c;
		else if ( c >= 0 && c !== graph.root ) inside = c;

	} else if ( selected >= 0 && selected < graph.N && graph.D > 0 ) {

		target = graph.anc[ selected * graph.D + height - 1 ];

	}

	if ( target < 0 && inside < 0 ) return out;
	for ( let k = 0; k < K; k ++ ) {

		let hit = clusters[ k ] === target;
		if ( inside >= 0 ) {

			let c = clusters[ k ];
			while ( c >= 0 && c !== inside && graph.clusterHeight[ c ] < graph.clusterHeight[ inside ] ) c = graph.clusterParent[ c ];
			hit = c === inside;

		}

		out[ k ] = hit ? strong : weak;

	}

	return out;

}

/**
 * Share of the view (0..1) that the quad of one volume covers: the CPU mirror of the vertex stage.
 * @param {number} x view-space centre
 * @param {number} y
 * @param {number} z (negative in front of the camera)
 * @param {number} R volume radius, world units
 * @param {number} P00 projection matrix element [0][0]
 * @param {number} P11 projection matrix element [1][1]
 */
export function quadCoverage( x, y, z, R, P00, P11 ) {

	if ( ! ( R > 0 ) ) return 0;
	const d2 = x * x + y * y + z * z;
	if ( d2 < 1.21 * R * R ) return 1; // camera inside or right in front: the full-screen quad
	if ( ! ( z < 0 ) ) return 0; // the billboard lies behind the camera
	// tangent billboard: a square of half side h through the centre, facing the eye
	const h = R * Math.sqrt( d2 / Math.max( d2 - R * R, 0.2 * R * R ) );
	const cx = x * P00 / - z, cy = y * P11 / - z, hx = h * P00 / - z, hy = h * P11 / - z;
	const w = Math.min( 1, cx + hx ) - Math.max( - 1, cx - hx ), t = Math.min( 1, cy + hy ) - Math.max( - 1, cy - hy );
	return w > 0 && t > 0 ? w * t / 4 : 0;

}

/**
 * Overdraw budget of the field: which volumes are drawn so that the sum of their quads stays
 * within `budget` views. Small (far) volumes are the shapes that show the structure and cost
 * little; volumes that fill the view are hazes, of which a few say as much as twenty. So the
 * volumes are taken in the order emphasised first, then smallest coverage first (ties: the one
 * the camera is deepest inside), while the running sum stays within the budget; the others get
 * weight 0 and are not drawn. A volume that was drawn before keeps its place up to
 * budget × ( 1 + hysteresis ), so a slow camera does not toggle the one at the border.
 * @param {ArrayLike<number>} coverage share of the view per volume (quadCoverage())
 * @param {ArrayLike<number>} priority per volume: > 1 = emphasised, 0 = not drawn anyway
 * @param {number} budget views
 * @param {object} [options] { depth: tie breaker per volume (smaller first), previous: weights of
 *   the last call, hysteresis = 0.15 }
 * @param {Float32Array} [out]
 * @return {{ weights: Float32Array, total: number, drawn: number, culled: number }} weights 1 / 0,
 *   total = Σ coverage of the drawn volumes
 */
export function fieldBudget( coverage, priority, budget, { depth = null, previous = null, hysteresis = 0.15 } = {}, out = new Float32Array( coverage.length ) ) {

	const K = coverage.length;
	const order = [];
	for ( let j = 0; j < K; j ++ ) {

		out[ j ] = 0;
		if ( priority[ j ] > 1e-4 && coverage[ j ] > 0 ) order.push( j );

	}

	order.sort( ( a, b ) => ( priority[ b ] > 1 ) - ( priority[ a ] > 1 ) || coverage[ a ] - coverage[ b ] || ( depth ? depth[ a ] - depth[ b ] : 0 ) || a - b );
	let total = 0, drawn = 0;
	for ( const j of order ) {

		const limit = previous && previous[ j ] > 0 ? budget * ( 1 + hysteresis ) : budget;
		// the first volume is always drawn: a budget below one view still shows where the camera is
		if ( drawn > 0 && total + coverage[ j ] > limit + 1e-9 ) continue;
		out[ j ] = 1;
		total += coverage[ j ];
		drawn ++;

	}

	return { weights: out, total, drawn, culled: order.length - drawn };

}

/**
 * Size of the volume behind a typical field pixel, as a share of the view: Σ w·c² / Σ w·c over the
 * volumes (c = quadCoverage(), w = their weight). Unlike the sum of the coverages it does not grow
 * with the number of volumes: many small shapes give a small value, a few hazes that fill the view
 * a value near 1, whatever lies beyond them.
 * @param {ArrayLike<number>} coverage share of the view per volume
 * @param {ArrayLike<number>} [weights] per volume (0 = not drawn anyway); default 1
 * @return {number} 0..1 (0 when nothing is covered)
 */
export function typicalCoverage( coverage, weights = null ) {

	let sum = 0, sq = 0;
	for ( let j = 0; j < coverage.length; j ++ ) {

		const w = weights ? weights[ j ] : 1, c = coverage[ j ];
		if ( ! ( w > 0 ) || ! ( c > 0 ) ) continue;
		sum += w * c;
		sq += w * c * c;

	}

	return sum > 0 ? sq / sum : 0;

}

/**
 * How much a view is an overview for the field: 1 while the typical volume (typicalCoverage()) is a
 * shape of at most `full` of the view, 0 once it covers `none` of it (the camera is inside the
 * graph or close to a community), smooth between.
 */
export function overviewFactor( typical, { full = 0.12, none = 0.32 } = {} ) {

	const t = Math.min( 1, Math.max( 0, ( typical - full ) / Math.max( 1e-6, none - full ) ) );
	return 1 - t * t * ( 3 - 2 * t );

}

/**
 * Default strength of the field for a graph of N nodes: fades in between small[ 0 ] and small[ 1 ]
 * nodes (a tiny graph has nothing to group) and out between large[ 0 ] and large[ 1 ] (the nodes
 * of a large graph form the coloured cloud themselves). `large: null` keeps large graphs at 1.
 * @return {number} 0..1
 */
export function fieldGraphFactor( N, { small = [ 16, 48 ], large = [ 5000, 20000 ] } = {} ) {

	const up = Math.min( 1, Math.max( 0, ( N - small[ 0 ] ) / Math.max( 1, small[ 1 ] - small[ 0 ] ) ) );
	if ( ! large ) return up;
	const t = Math.min( 1, Math.max( 0, ( N - large[ 0 ] ) / Math.max( 1, large[ 1 ] - large[ 0 ] ) ) );
	return up * ( 1 - t * t * ( 3 - 2 * t ) );

}

// ── the module ───────────────────────────────────────────────────────────────────────────────

const _v = /*@__PURE__*/ new THREE.Vector3();
const _w = /*@__PURE__*/ new THREE.Vector3();

export class ClusterField {

	constructor( options = {} ) {

		const o = { strength: 0.5, level: 'auto', radius: 1.5, core: 0.35, absorb: 0.15, saturation: 0.8, closeFade: 0.6, smoothing: 0.18, maxVolumes: null, maxOverdraw: null, adaptive: true, lowTier: false, renderOrder: - 5, ...options };
		this.order = 13;
		this.isClusterField = true;
		this.app = null;
		this.graph = null;
		this.buffers = null;
		this.mesh = null;
		this.height = 0;
		this.clusters = new Uint32Array( 0 );
		this.smoothing = o.smoothing;
		this.lowTier = !! o.lowTier;
		this.adaptive = o.adaptive !== false; // step back inside the graph and on large graphs (see the header)
		this.renderOrder = o.renderOrder;
		this._strength = Math.min( 1, Math.max( 0, + o.strength || 0 ) );
		this._level = o.level;
		this._visible = true;
		this._saturation = o.saturation;
		this._maxVolumes = o.maxVolumes;
		this._maxOverdraw = o.maxOverdraw;
		this._focus = - 1;
		this._selected = - 1;
		this._lens = false;

		this.uniforms = {
			intensity: uniform( 0 ), // peak radiance of a neutral volume (strength × tier × graph)
			radius: uniform( o.radius ), // volume radius in robust radii
			core: uniform( o.core ),
			absorb: uniform( o.absorb ),
			veil: uniform( 1 ), // the overview factor of the view on the absorption (the light has it in `intensity`)
			closeFade: uniform( o.closeFade ), // share of its light a volume loses while it fills the view
			shrink: uniform( 1 ), // radius factor of the overdraw budget (FIELD_MIN_SHRINK … 1), eased
			// kernel
			easePos: uniform( 1 ), easeRadius: uniform( 1 ), easeColor: uniform( 1 ),
			gpuRadius: uniform( 0 ), // 1: no readback, take 0.62 × clusterPos.w
		};

		this._geometry = quadGeometry();
		this._materials = new Map();
		this._kernel = null;
		this._state = null; // { pos, color, target, aux } storage nodes
		this._snap = true;
		this._stats = null;
		this._emph = new Float32Array( 0 );
		this._compact = new Float32Array( 0 );
		this._cover = new Float32Array( 0 ); // share of the view per volume (quadCoverage), this frame
		this._near = new Float32Array( 0 ); // distance / radius per volume (tie breaker of the budget)
		this._keep = new Float32Array( 0 ); // 1 / 0: inside the overdraw budget
		this._weight = new Float32Array( 0 ); // emphasis × compactness, what the budget ranks (0 = no light anyway)
		this._gain = new Float32Array( 0 ); // factor of the graph size per volume (the focused ones are exempt)
		this._budget = { total: 0, drawn: 0, culled: 0, limit: 0 };
		this._graph = { small: 1, factor: 1, focus: 1 }; // fieldGraphFactor() of the graph, and what a focused volume keeps
		this._typical = 0; // typicalCoverage() of the view
		this._overview = 1; // eased overviewFactor()
		this._snapView = true; // the overview factor jumps to its target in the next update
		this._snapLight = true; // … and so does the light of the volumes (snap(), a new graph)
		this._linger = 0; // seconds the field stays on after its last volume lost its light
		this._dirtyAux = false;
		this._seen = { positions: - 1, colors: - 1, t: - 1e9, sum: NaN, focus: - 2, selected: - 2, max: - 1, tier: '', lens: null, sat: NaN };
		this._onCluster = ( e ) => this.focus( e.detail && e.detail.cluster >= 0 ? e.detail.cluster : null );
		this._onSelect = ( e ) => {

			this._selected = e.detail && e.detail.index >= 0 ? e.detail.index : - 1;

		};

		this._onLens = ( e ) => {

			this._lens = !! ( e.detail && e.detail.color );

		};

	}

	// ── settings ────────────────────────────────────────────────────────────────────────────

	get strength() {

		return this._strength;

	}

	set strength( v ) {

		this._strength = Math.min( 1, Math.max( 0, + v || 0 ) );

	}

	get level() {

		return this._level;

	}

	set level( v ) {

		if ( v === this._level ) return;
		this._level = v;
		if ( this.graph ) this._build();

	}

	get visible() {

		return this._visible;

	}

	set visible( v ) {

		this._visible = !! v;
		if ( this.mesh && ! this._visible ) this.mesh.visible = false;

	}

	get radius() {

		return this.uniforms.radius.value;

	}

	set radius( v ) {

		this.uniforms.radius.value = Math.max( 0.1, + v || 0 );

	}

	get core() {

		return this.uniforms.core.value;

	}

	set core( v ) {

		this.uniforms.core.value = Math.max( 0, + v || 0 );

	}

	get absorb() {

		return this.uniforms.absorb.value;

	}

	set absorb( v ) {

		this.uniforms.absorb.value = Math.min( 1, Math.max( 0, + v || 0 ) );

	}

	get closeFade() {

		return this.uniforms.closeFade.value;

	}

	set closeFade( v ) {

		this.uniforms.closeFade.value = Math.min( 1, Math.max( 0, + v || 0 ) );

	}

	get saturation() {

		return this._saturation;

	}

	set saturation( v ) {

		this._saturation = Math.max( 0, + v || 0 );

	}

	get maxVolumes() {

		return this._maxVolumes;

	}

	set maxVolumes( v ) {

		this._maxVolumes = v === null || v === undefined ? null : Math.max( 1, v | 0 );

	}

	/** Overdraw budget in views (null = the tier's, FIELD_TIERS). */
	get maxOverdraw() {

		return this._maxOverdraw;

	}

	set maxOverdraw( v ) {

		this._maxOverdraw = v === null || v === undefined ? null : Math.max( 0.25, + v || 0 );

	}

	/** Emphasise one community (cluster index of any level in the prepared graph's cluster space), null = none. */
	focus( cluster ) {

		this._focus = cluster === null || cluster === undefined || ! ( cluster >= 0 ) ? - 1 : cluster | 0;
		return this;

	}

	/** Jump to the current targets in the next frame (no easing). */
	snap() {

		this._snap = this._snapView = this._snapLight = true;
		this._seen.t = - 1e9;
		return this;

	}

	/** Strength factor of the active tier and graph (0 = the field is off there). */
	get tierFactor() {

		const tier = FIELD_TIERS[ gfxState.tier ] || FIELD_TIERS.high;
		if ( gfxState.tier === 'low' && ! this.lowTier ) return 0;
		return tier.factor;

	}

	/**
	 * Volumes drawn, the pixels they cover in this view and what that means as overdraw (the state
	 * of the last update(): budget = the limit in views, culled = volumes the budget left out,
	 * shrink = the radius factor it applied).
	 */
	get stats() {

		const out = {
			height: this.height, count: this.clusters.length, quads: 0, coveredPixels: 0, overdraw: 0, intensity: this.uniforms.intensity.value,
			budget: this._budget.limit, culled: 0, shrink: this.uniforms.shrink.value,
			typical: this._typical, overview: this.adaptive ? this._overview : 1, graph: this._graph.factor,
		};
		const app = this.app;
		if ( ! app || ! this._stats || ! this.mesh || ! this.mesh.visible ) return out;
		const scale = gfxUniforms.pxScale.value;
		out.quads = this._budget.drawn;
		out.culled = this._budget.culled;
		out.overdraw = this._budget.total;
		out.coveredPixels = Math.round( this._budget.total * app.width * scale * app.height * scale );
		return out;

	}

	/** Per volume: { cluster, size, radius (robust, world units), compact 0..1, emphasis } (CPU side). */
	volumes() {

		const s = this._stats, out = [];
		if ( ! s ) return out;
		for ( let j = 0; j < this.clusters.length; j ++ ) {

			const k = this.clusters[ j ] - s.offset;
			out.push( { cluster: this.clusters[ j ], size: s.size[ k ], radius: this._radiusOf( k ), compact: this._compact[ j ], emphasis: this._emph[ j ], coverage: this._cover[ j ], drawn: this._keep[ j ] > 0 } );

		}

		return out;

	}

	// ── App module ──────────────────────────────────────────────────────────────────────────

	init( app ) {

		this.app = app;
		app.events.addEventListener( 'cluster', this._onCluster );
		app.events.addEventListener( 'select', this._onSelect );
		app.events.addEventListener( 'lens', this._onLens );

	}

	setGraph( buffers, graph, app ) {

		if ( app ) this.app = app;
		this._disposeGraph();
		this.buffers = buffers;
		this.graph = graph;
		this._focus = - 1;
		this._selected = - 1;
		this._build();

	}

	update( dt, app = this.app ) {

		if ( ! this.graph || ! app ) return;
		// the budget follows the tier
		const tier = FIELD_TIERS[ gfxState.tier ] || FIELD_TIERS.high;
		const budget = this._maxVolumes ?? tier.max;
		if ( budget !== this._seen.max ) this._build();

		const K = this.clusters.length;
		const U = this.uniforms;
		// the graph: tiny ones have nothing to group, in large ones the nodes are the cloud; a focused
		// or selected community keeps its volume there
		const G = this._graph;
		G.small = fieldGraphFactor( this.graph.N, { large: null } );
		G.factor = this.adaptive ? fieldGraphFactor( this.graph.N ) : G.small;
		G.focus = Math.max( G.factor, G.small * FIELD_FOCUS_KEEP );
		const selected = this._selected >= 0 ? this._selected : this.buffers.uniforms.selectedIndex.value | 0;
		const reach = this._focus >= 0 || selected >= 0 ? G.focus : G.factor; // the largest factor a volume can have
		// a focus that ends on a graph without neutral volumes: stay until its light has eased out
		this._linger = reach > 0 ? 1.2 : Math.max( 0, this._linger - dt );
		const base = K > 0 && this._visible && this._linger > 0 ? FIELD_RADIANCE * this._strength * this.tierFactor * ( this._lens ? 0.75 : 1 ) : 0;
		if ( ! ( base > 1e-4 ) ) {

			// off: nothing is measured, drawn or computed
			U.intensity.value = 0;
			if ( this.mesh ) this.mesh.visible = false;
			this._snap = this._snapView = true; // the state goes stale while nothing runs
			return;

		}

		// the view: an overview, or hazes around the camera (CPU, K coverages)
		this._updateTargets( app, selected );
		this._updateBudget( app, dt );
		U.veil.value = this.adaptive ? this._overview : 1;
		U.intensity.value = base * U.veil.value;
		const active = U.intensity.value > 1e-4;
		if ( this.mesh ) this.mesh.visible = active;
		if ( ! active ) {

			this._snap = true; // the smoothed state on the GPU goes stale while nothing is drawn
			return;

		}

		this._applyVariant();
		if ( this._dirtyAux ) {

			// emphasis × compactness × graph size × budget: what the kernel eases the light of a volume toward
			const aux = this._state.aux.value.array;
			for ( let j = 0; j < K; j ++ ) aux[ 4 * j + 1 ] = this._emph[ j ] * this._compact[ j ] * this._gain[ j ] * this._keep[ j ];
			this._state.aux.value.needsUpdate = true;
			this._dirtyAux = false;

		}

		// back after a pause (hidden, off, another tier): places and sizes jump to where the
		// communities are now, the light starts from nothing and eases in. snap() / a new graph: all jumps
		if ( this._snap && ! this._snapLight ) {

			const color = this._state.color.value, c = color.array, t = this._state.target.value.array;
			for ( let j = 0; j < K; j ++ ) {

				c[ 4 * j ] = t[ 4 * j ]; c[ 4 * j + 1 ] = t[ 4 * j + 1 ]; c[ 4 * j + 2 ] = t[ 4 * j + 2 ];
				c[ 4 * j + 3 ] = 0;

			}

			color.needsUpdate = true;

		}

		const tau = Math.max( 1e-3, this.smoothing );
		const ease = ( t, jump ) => ( jump ? 1 : 1 - Math.exp( - dt / t ) );
		U.easePos.value = ease( tau, this._snap );
		U.easeRadius.value = ease( tau * 2, this._snap );
		U.easeColor.value = ease( 0.3, this._snapLight );
		U.gpuRadius.value = app.options && app.options.readback === false ? 1 : 0;
		this._snap = this._snapLight = false;
		app.renderer.compute( this._kernel );

	}

	dispose() {

		this._disposeGraph();
		this._geometry.dispose();
		const app = this.app;
		if ( app ) {

			app.events.removeEventListener( 'cluster', this._onCluster );
			app.events.removeEventListener( 'select', this._onSelect );
			app.events.removeEventListener( 'lens', this._onLens );

		}

	}

	// ── structure ───────────────────────────────────────────────────────────────────────────

	/** Resolves the level, picks the communities and (re)creates buffers, kernel and mesh. */
	_build() {

		const graph = this.graph, buffers = this.buffers, app = this.app;
		this._disposeGraph();
		const counts = levelCounts( graph );
		let height = this._level === 'auto' || this._level === null || this._level === undefined ? chooseFieldLevel( counts ) : Math.round( + this._level ) || 0;
		if ( graph.D === 0 ) height = 0;
		else if ( height !== 0 ) height = Math.min( graph.D, Math.max( 1, height ) );
		this.height = height;
		const tier = FIELD_TIERS[ gfxState.tier ] || FIELD_TIERS.high;
		const budget = this._maxVolumes ?? tier.max;
		this._seen.max = budget;
		this.clusters = new Uint32Array( 0 );
		if ( height === 0 ) return;

		// the communities of the level, largest first, at most `budget` of them
		const offset = graph.levelOffset[ height - 1 ], count = graph.levelOffset[ height ] - offset;
		const ids = Array.from( { length: count }, ( _, k ) => offset + k ).filter( ( c ) => graph.clusterMass[ c ] >= 2 );
		ids.sort( ( a, b ) => graph.clusterMass[ b ] - graph.clusterMass[ a ] || a - b );
		ids.length = Math.min( ids.length, budget );
		const K = ids.length;
		this.clusters = Uint32Array.from( ids );
		if ( K === 0 ) return;

		const S = this._state = {
			pos: instancedArray( K, 'vec4' ).setName( 'fieldPos' ), // smoothed centre, robust radius
			color: instancedArray( K, 'vec4' ).setName( 'fieldColor' ), // smoothed rgb, emphasis
			target: instancedArray( K, 'vec4' ).setName( 'fieldTarget' ), // CPU: rgb, robust radius
			aux: instancedArray( K, 'vec4' ).setName( 'fieldAux' ), // CPU: cluster index, emphasis
		};
		const aux = S.aux.value.array;
		for ( let j = 0; j < K; j ++ ) {

			aux[ 4 * j ] = ids[ j ];
			aux[ 4 * j + 1 ] = 1;

		}

		this._emph = new Float32Array( K ).fill( 1 );
		this._compact = new Float32Array( K ).fill( 1 );
		this._cover = new Float32Array( K );
		this._near = new Float32Array( K );
		this._keep = new Float32Array( K ).fill( 1 );
		this._weight = new Float32Array( K ).fill( 1 );
		this._gain = new Float32Array( K ).fill( 1 );
		this._budget = { total: 0, drawn: 0, culled: 0, limit: 0 };
		this._typical = 0;
		this._overview = 1;
		this._snapView = this._snapLight = true;
		this._seen.gain = NaN;
		this._dirtyAux = true;
		this._stats = null;
		this._seen.positions = - 1;
		this._seen.colors = - 1;
		this._seen.t = - 1e9;
		this._seen.focus = this._seen.selected = - 2;
		this._snap = true;

		// the low-pass kernel
		const U = this.uniforms;
		const clusterPos = readOnly( buffers.clusterPos );
		const target = readOnly( S.target ), auxRO = readOnly( S.aux );
		this._kernel = Fn( () => {

			const a = auxRO.element( instanceIndex ).toVar();
			const t = target.element( instanceIndex ).toVar();
			const cp = clusterPos.element( uint( a.x.add( 0.5 ) ) ).toVar();
			const p = S.pos.element( instanceIndex ).toVar();
			const c = S.color.element( instanceIndex ).toVar();
			const radius = select( U.gpuRadius.greaterThan( 0.5 ), cp.w.mul( 0.62 ), t.w );
			S.pos.element( instanceIndex ).assign( vec4( mix( p.xyz, cp.xyz, U.easePos ), mix( p.w, radius, U.easeRadius ) ) );
			S.color.element( instanceIndex ).assign( mix( c, vec4( t.xyz, a.y ), U.easeColor ) );

		} )().compute( K ).setName( 'field.smooth' );

		this.mesh = new THREE.Mesh( this._geometry, null );
		this.mesh.name = 'ClusterField';
		this.mesh.count = K;
		this.mesh.frustumCulled = false;
		this.mesh.renderOrder = this.renderOrder;
		this.mesh.castShadow = this.mesh.receiveShadow = false;
		this.mesh.visible = false;
		this.mesh.raycast = () => {};
		this._applyVariant();
		app.scene.add( this.mesh );

	}

	_applyVariant() {

		if ( ! this.mesh ) return;
		const key = gfxState.velocity ? 'mv' : 'still';
		let material = this._materials.get( key );
		if ( ! material ) {

			material = this._material( { velocity: !! gfxState.velocity } );
			material.name = 'ClusterField.' + key;
			this._materials.set( key, material );

		}

		if ( this.mesh.material !== material ) this.mesh.material = material;

	}

	/** Robust radius (world units) of community k of the field's level, with a floor for tiny ones. */
	_radiusOf( k ) {

		const s = this._stats;
		return Math.max( s.rms[ k ], s.floor );

	}

	/** CPU targets: robust radii (from the readback, while it changes), colours, emphasis. */
	_updateTargets( app, selected = - 1 ) {

		const graph = this.graph, buffers = this.buffers, S = this._state, seen = this._seen;
		const K = this.clusters.length;
		const tgt = S.target.value.array;
		let dirtyTarget = false, dirtyAux = false;
		const now = performance.now();

		// radii: at most 8 times a second, and only when the content of the readback changed
		const positions = app.positions && app.positions.length >= 4 * graph.N ? app.positions : graph.initPos;
		if ( ! this._stats || ( seen.positions !== app.positionsVersion && now - seen.t >= 120 ) ) {

			seen.positions = app.positionsVersion;
			seen.t = now;
			let sum = 0;
			const step = Math.max( 4, 4 * Math.floor( graph.N / 512 ) );
			for ( let i = 0, n = 4 * graph.N; i < n; i += step ) sum += positions[ i ] + 1.7 * positions[ i + 1 ] + 2.9 * positions[ i + 2 ];
			if ( ! this._stats || sum !== seen.sum ) {

				seen.sum = sum;
				const s = this._stats = clusterStats( graph, positions, this.height, this._stats );
				// floor: a community of two or three nodes on one spot still gets a visible volume
				const sorted = Float32Array.from( s.rms ).sort();
				s.floor = 0.35 * ( sorted[ sorted.length >> 1 ] || 0 ) + 1e-3 * graph.layoutRadius;
				for ( let j = 0; j < K; j ++ ) {

					const k = this.clusters[ j ] - s.offset;
					tgt[ 4 * j + 3 ] = this._radiusOf( k );
					// a community that is spread through the graph is no place: its volume fades out
					const w = compactness( s.rms[ k ], s.size[ k ], s.graphRms, s.total );
					if ( Math.abs( w - this._compact[ j ] ) > 0.02 || ( w === 0 ) !== ( this._compact[ j ] === 0 ) ) {

						this._compact[ j ] = w;
						dirtyAux = true;

					}

				}

				dirtyTarget = true;

			}

		}

		// colours: the members' current colour targets (palette, or a colour lens)
		const colors = buffers.nodeColorTarget.value;
		const sat = this._saturation * ( this._lens ? 0.7 : 1 );
		if ( seen.colors !== colors.version || seen.sat !== sat ) {

			seen.colors = colors.version;
			seen.sat = sat;
			const mean = clusterColors( graph, colors.array, this.height, this._mean );
			this._mean = mean;
			const offset = graph.levelOffset[ this.height - 1 ];
			for ( let j = 0; j < K; j ++ ) {

				const k = this.clusters[ j ] - offset;
				const c = fieldColor( mean[ 3 * k ], mean[ 3 * k + 1 ], mean[ 3 * k + 2 ], sat );
				tgt[ 4 * j ] = c[ 0 ]; tgt[ 4 * j + 1 ] = c[ 1 ]; tgt[ 4 * j + 2 ] = c[ 2 ];

			}

			dirtyTarget = true;

		}

		// emphasis: a focused community, else the community of the selection; the graph-size factor
		// per volume (an emphasised volume keeps its light on graphs that are too large for the field)
		const G = this._graph, gainKey = G.factor + 2 * G.focus;
		if ( seen.focus !== this._focus || seen.selected !== selected || seen.gain !== gainKey ) {

			seen.focus = this._focus;
			seen.selected = selected;
			seen.gain = gainKey;
			emphasisTargets( graph, this.clusters, this._focus, selected, undefined, this._emph );
			for ( let j = 0; j < K; j ++ ) this._gain[ j ] = this._emph[ j ] > 1 ? G.focus : G.factor;
			dirtyAux = true;

		}

		if ( dirtyTarget ) S.target.value.needsUpdate = true;
		if ( dirtyAux ) this._dirtyAux = true;

	}

	/**
	 * The view (CPU, K ≤ 128 volumes): the coverage of every quad from the readback centres; whether
	 * that is an overview (typicalCoverage() → the eased overview factor); and the overdraw budget:
	 * the radius factor that brings the sum toward the budget and which volumes are drawn
	 * (fieldBudget()).
	 */
	_updateBudget( app, dt ) {

		const s = this._stats, K = this.clusters.length, U = this.uniforms, B = this.buffers.uniforms;
		if ( ! s ) return;
		const tier = FIELD_TIERS[ gfxState.tier ] || FIELD_TIERS.high;
		const limit = this._maxOverdraw ?? tier.overdraw;
		const cam = app.camera, m = cam.matrixWorldInverse;
		const P00 = cam.projectionMatrix.elements[ 0 ], P11 = cam.projectionMatrix.elements[ 5 ];
		const blend = Math.min( 1, Math.max( 0, B.layoutBlend.value ) );
		const gc = B.graphCenter.value, sr = B.sphereRadius.value;
		const cover = this._cover, near = this._near, weight = this._weight;
		const radius = U.radius.value;

		// view-space centres (pushed onto the sphere in the sphere layout, like the vertex stage)
		const view = this._view && this._view.length === 3 * K ? this._view : ( this._view = new Float32Array( 3 * K ) );
		let total = 0;
		for ( let j = 0; j < K; j ++ ) {

			const k = this.clusters[ j ] - s.offset;
			_v.set( s.center[ 3 * k ], s.center[ 3 * k + 1 ], s.center[ 3 * k + 2 ] );
			if ( blend > 0 ) {

				_w.copy( _v ).sub( gc );
				_w.multiplyScalar( sr / Math.max( _w.length(), 1e-4 ) ).add( gc );
				_v.lerp( _w, blend );

			}

			_v.applyMatrix4( m );
			view[ 3 * j ] = _v.x; view[ 3 * j + 1 ] = _v.y; view[ 3 * j + 2 ] = _v.z;
			weight[ j ] = this._gain[ j ] > 0 ? this._emph[ j ] * this._compact[ j ] : 0;
			const R = this._radiusOf( k ) * radius;
			near[ j ] = _v.length() / Math.max( R, 1e-6 );
			cover[ j ] = weight[ j ] > 1e-4 ? quadCoverage( _v.x, _v.y, _v.z, R, P00, P11 ) : 0;
			total += cover[ j ];

		}

		// an overview, or hazes around the camera: all volumes ease out together (see the header)
		this._typical = typicalCoverage( cover, weight );
		const overview = overviewFactor( this._typical );
		this._overview = this._snapView ? overview : this._overview + ( overview - this._overview ) * ( 1 - Math.exp( - dt / 0.35 ) );
		if ( Math.abs( this._overview - overview ) < 2e-3 ) this._overview = overview;
		this._snapView = false;

		// over the budget: the volumes first get tighter (eased), then the largest hazes go
		const shrink = total > limit ? Math.max( FIELD_MIN_SHRINK, Math.sqrt( limit / total ) ) : 1;
		U.shrink.value = this._snap ? shrink : U.shrink.value + ( shrink - U.shrink.value ) * ( 1 - Math.exp( - dt / 0.5 ) );
		for ( let j = 0; j < K; j ++ ) {

			const R = this._radiusOf( this.clusters[ j ] - s.offset ) * radius * U.shrink.value;
			cover[ j ] = quadCoverage( view[ 3 * j ], view[ 3 * j + 1 ], view[ 3 * j + 2 ], R, P00, P11 );

		}

		const previous = this._previous && this._previous.length === K ? this._previous : ( this._previous = new Float32Array( K ).fill( 1 ) );
		previous.set( this._keep );
		const r = fieldBudget( cover, weight, limit, { depth: near, previous }, this._keep );
		for ( let j = 0; j < K; j ++ ) if ( previous[ j ] !== this._keep[ j ] ) this._dirtyAux = true;
		this._budget = { total: r.total, drawn: r.drawn, culled: r.culled, limit };

	}

	// ── material ────────────────────────────────────────────────────────────────────────────

	_material( { velocity = false } = {} ) {

		const U = this.uniforms, S = this._state, B = this.buffers.uniforms;
		const vCenter = varyingProperty( 'vec4', 'vFieldCenter' ); // view-space centre, radius
		const vPoint = varyingProperty( 'vec3', 'vFieldPoint' ); // view-space point of the quad under the fragment
		const vColor = varyingProperty( 'vec4', 'vFieldColor' ); // rgb × intensity × emphasis × fog × close-up, absorption

		const material = new THREE.NodeMaterial();
		material.lights = false;
		material.fog = false;
		material.transparent = true;
		// rgb + dst · ( 1 − a ) with rgb and a given independently. Not material.premultipliedAlpha: three
		// then multiplies the colour by the alpha once more (the light would scale with the absorption)
		material.blending = THREE.CustomBlending;
		material.blendEquation = THREE.AddEquation;
		material.blendSrc = material.blendSrcAlpha = THREE.OneFactor;
		material.blendDst = material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
		material.depthWrite = false;
		material.depthTest = true;
		material.side = THREE.DoubleSide;
		if ( velocity ) material.mrtNode = skipVelocity();

		material.vertexNode = Fn( () => {

			const st = S.pos.element( instanceIndex ).toVar( 'fState' );
			const col = S.color.element( instanceIndex ).toVar( 'fColor' );
			// sphere layout: the clusters lie on the sphere, their centroid a little inside it
			const rel = st.xyz.sub( B.graphCenter ).toVar( 'fRel' );
			const onSphere = B.graphCenter.add( rel.div( max( length( rel ), 1e-4 ) ).mul( B.sphereRadius ) );
			const world = mix( st.xyz, onSphere, clamp( B.layoutBlend, 0, 1 ) );
			const R = max( st.w.mul( U.radius ).mul( U.shrink ), 1e-4 ).toVar( 'fR' );
			const center = cameraViewMatrix.mul( vec4( world, 1 ) ).xyz.toVar( 'fCenter' );
			const d2 = dot( center, center ).toVar( 'fD2' );
			const d = sqrt( d2 );
			const r2 = R.mul( R );
			const q = positionGeometry.xy;

			// far: the tangent billboard (plane through the centre, facing the eye; the silhouette cone
			// cuts it in a circle of radius h). Near or inside: a quad that covers the screen
			const close = d2.lessThan( r2.mul( 1.21 ) );
			const e = center.negate().div( max( d, 1e-6 ) ).toVar( 'fEye' );
			const upRef = select( abs( e.y ).greaterThan( 0.999 ), vec3( 1, 0, 0 ), vec3( 0, 1, 0 ) );
			const right = normalize( cross( upRef, e ) ).toVar( 'fRight' );
			const up = cross( e, right );
			const h = R.mul( d ).div( sqrt( max( d2.sub( r2 ), r2.mul( 0.2 ) ) ) );
			const pFar = center.add( right.mul( q.x ).add( up.mul( q.y ) ).mul( h ) );
			const P = cameraProjectionMatrix;
			const dq = cameraNear.mul( 2 );
			const pNear = vec3(
				q.x.mul( 1.02 ).add( P.element( 2 ).element( 0 ) ).mul( dq ).div( P.element( 0 ).element( 0 ) ),
				q.y.mul( 1.02 ).add( P.element( 2 ).element( 1 ) ).mul( dq ).div( P.element( 1 ).element( 1 ) ),
				dq.negate(),
			);
			const p = select( close, pNear, pFar ).toVar( 'fPoint' );

			// a volume that fills the view has done its work: seen from inside (or from right in front)
			// it would only be a coloured haze over the nodes one came to look at, so it recedes
			const span = R.div( max( d, 1e-4 ) ); // 1 = the camera is on the volume's surface
			const closeUp = float( 1 ).sub( U.closeFade.mul( smoothstep( 0.3, 1.1, span ) ) ).toVar( 'fCloseUp' );
			const gain = U.intensity.mul( col.w ).mul( fogTransmittance( d ) ).mul( closeUp ).toVar( 'fGain' );
			vCenter.assign( vec4( center, R ) );
			vPoint.assign( p );
			// the veil recedes with the light: a haze that has faded must not darken what is behind it
			vColor.assign( vec4( col.xyz.mul( gain ), U.absorb.mul( U.veil ).mul( min( col.w, 1 ) ).mul( closeUp ) ) );
			// behind the camera and out of reach, or without light: collapse
			const off = gain.lessThan( 1e-5 ).or( center.z.sub( R ).greaterThan( 0 ) );
			return select( off, vec4( 0, 0, 2, 1 ), P.mul( vec4( p, 1 ) ) );

		} )();

		// ray through the fragment (camera at the origin of view space)
		const ray = normalize( vPoint ).toVar( 'fieldRay' );
		const C = vCenter.xyz, R = vCenter.w;
		const tc = dot( C, ray ).toVar( 'fieldTc' ); // distance along the ray to the closest approach
		const a = clamp( float( 1 ).sub( dot( C, C ).sub( tc.mul( tc ) ).div( R.mul( R ) ) ), 0, 1 ).toVar( 'fieldA' );
		const hx = sqrt( a ).toVar( 'fieldH' ); // half chord in units of R

		material.colorNode = Fn( () => {

			// ∫ ρ ds = R · [ a² x − ⅔ a x³ + ⅕ x⁵ ] from the entry (or the camera) to the exit
			const F = ( x ) => {

				const x2 = x.mul( x );
				return x.mul( a.mul( a ).sub( a.mul( 2 / 3 ).mul( x2 ) ).add( x2.mul( x2 ).mul( 0.2 ) ) );

			};

			const x0 = clamp( tc.negate().div( R ), hx.negate(), hx );
			const tau = clamp( F( hx ).sub( F( x0 ) ).mul( 15 / 16 ), 0, 1 ).toVar( 'fieldTau' );
			const t2 = tau.mul( tau );
			const glow = tau.add( U.core.mul( t2 ).mul( t2 ) );
			return vec4( vColor.xyz.mul( glow ), vColor.w.mul( tau ) );

		} )();

		// depth test at the point where the ray leaves the volume: nodes inside it stay in front
		material.depthNode = viewToDepth( ray.mul( max( tc.add( hx.mul( R ) ), cameraNear.mul( 1.01 ) ) ) );

		return material;

	}

	_disposeGraph() {

		if ( this.mesh ) {

			this.mesh.removeFromParent();
			this.mesh = null;

		}

		for ( const material of this._materials.values() ) material.dispose();
		this._materials.clear();
		this._kernel?.dispose?.();
		this._kernel = null;
		if ( this._state ) for ( const key in this._state ) this._state[ key ].value.dispose?.();
		this._state = null;
		this._stats = null;
		this._mean = null;
		this.clusters = new Uint32Array( 0 );
		this.height = 0;

	}

}
