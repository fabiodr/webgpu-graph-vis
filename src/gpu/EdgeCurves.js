// GPU curve cache for the edge renderer: every edge's bundled curve, evaluated by ONE compute
// kernel into a storage buffer of samples, and only when something the curves depend on changed.
// The renderers (strands, tubes, picking, …) read the samples; a graph at rest costs no curve
// evaluation at all (stats.updates stops counting).
//
// ─── Control paths ──────────────────────────────────────────────────────────────────────────
//  The control polygon of edge e is [ s, ...interior, t ] with the interior points given as an
//  explicit path set (src/graph/bundling.js):
//    pathStart Uint32Array( E + 1 ), pathRefs Uint32Array( total ), maxInterior
//  ref < N = the live position of node `ref`; ref >= N = the live centroid of cluster ref − N.
//  The default set is hierarchicalPaths( graph ) (Holten / ImsvGraphVis: the centroids of the end
//  points' ancestors up to the LCA level), computed in setGraph; setPaths( paths ) swaps the set at
//  run time (edge-path bundling results, mixed sets, straight lines). On the GPU the set is
//    buffers.edgeInfo[e].w   pathStart[e] (element E holds the total)
//    curves.pathRefs         uint per interior point: bit 31 = cluster, bits 27..30 = the cluster's
//                            height (sphere-mode level), bits 0..26 = node / cluster index
//
// ─── Curve ──────────────────────────────────────────────────────────────────────────────────
//  Holten straightening P'_k = mix( lerp( s, t, k / (L−1) ), P_k, beta ) over the L = interior + 2
//  polygon points, then a uniform cubic B-spline with tripled end points (L + 1 spans; the curve
//  starts and ends exactly at the node centres). Sample i of S segments sits at x = i (L+1) / S:
//  span j = min( floor( x ), L ), u = x − j (exact integer arithmetic on the GPU), control points
//  Q[ j .. j+3 ] with Q[ m ] = P'[ clamp( m − 2, 0, L − 1 ) ]: a 4-point window, no loop over the
//  path. With hierarchical paths and unhook = 0 this is exactly the curve of the first edge renderer.
//
//  Unhooking (force layout; uniforms.unhook, default 1, 0 = off): a cluster centroid that lies
//  behind the source or beyond the target of its edge makes the curve leave the node backwards
//  and hook around it; a hub at the rim of its cluster then has the star of its edges beside it,
//  on the centroid. With d = the unit chord ( t − s ) / |t − s|, q = P_k − s, along = q · d,
//  over = along − clamp( along, 0, |t − s| ) and side = q − d · along, the curve uses
//    P_k − ( d · over + side · |over| / ( |side| + |over| ) ) · unhook
//  in place of such a centroid: it is moved into the plane through the end point it lies behind,
//  and toward that end point the further behind it lies (straight behind: the end point itself).
//  Every control point then projects into the chord, so no curve passes behind either of its
//  end points (a B-spline stays in the convex hull of its control points), the pull to the side
//  that makes the bundles stays, and the correction follows the live positions continuously.
//  Centroids that project into the chord are not touched, nor are the nodes of a path
//  (edge-path bundling: places the path really visits) or the sphere layout.
//
//  Sphere layout (buffers.uniforms.layoutBlend > 0, the paper's look): every control point is split
//  into a direction from the sphere centre and a radial level. Nodes (the end points and node refs
//  of a path) keep their own level |p − c| / sphereRadius; cluster points use
//  level( h ) = 1 + levelOffset + levelScale · ( h − 1 )^levelExponent with h = the cluster's height,
//  so bundles arc outward behind the nodes. Directions are blended in the log map of the sphere
//  around `sphereAxis`, levels separately, then recombined as c + exp( V ) · R · level; the result
//  is mixed with the force-layout curve by layoutBlend. Highlighted edges (selected node:
//  nodeState.x of an end point; edgeAttr.y · highlightFade) pull their cluster levels toward
//  `highlightLevel` (inside the sphere, in front of the context).
//
// ─── Cache ──────────────────────────────────────────────────────────────────────────────────
//  curves.samples  vec4 × E × capacity; sample i of edge e at index e · stride + i, i = 0 … S,
//                  stride = S + 1 = uniforms.stride (<= capacity). xyz = world position, w = arc
//                  length from the source in world units (sum of the chords; the last sample holds
//                  the curve's length).
//  One invocation per edge loops over its S + 1 samples (a fixed count: the loop carries the
//  running arc length and hoists the per-edge reads). Storage bindings of the kernel: edgeInfo,
//  pathRefs, nodePos, clusterPos, nodeState, edgeAttr, samples = 7.
//  Memory: 16 bytes · E · ( S + 1 ) ON THE GPU AND AGAIN ON THE CPU: three keeps the zero-filled
//  Float32Array it uploads once as attribute.array for the life of the buffer, although only the
//  compute kernel ever writes it (r187 derives the element count from that array, so it cannot be
//  swapped for a small one; the same holds for the other GPU-written buffers of GraphBuffers,
//  which are small). Both copies are capped by `memoryBudget` (default 64 MiB per copy; the edge
//  renderer passes 24 MiB on touch devices, see autoCurveMemory there): S is clamped to
//  maxSegments(). 384,444 edges: S = 8 → 55.4 MB (force layout); the sphere layout asks for 16 and
//  gets 9 → 61.5 MB. The buffer only grows (a larger S re-allocates it and calls onRealloc so
//  the renderers rebuild their materials). setGraph( …, { reserve } ) allocates room for a larger
//  segment count up front while that stays below `reserveBudget` (8 MiB), so that the first switch
//  to the sphere layout of a small graph only changes `stride` (no re-allocation, no new shaders).
//
//  The kernel runs in update() when
//    · node positions moved: buffers.moveFrame >= the frame of the last evaluation (Infinity =
//      unknown = moved; GraphBuffers.setPositions and the layout module maintain it),
//    · beta, unhook, the sample count, layoutBlend, sphereRadius, the level parameters, graphCenter,
//      sphereAxis or highlightLevel changed (compared by value every frame),
//    · the path set changed (setPaths), or invalidate() was called,
//    · in sphere mode only: the highlight state is easing (a CPU write that changed what the
//      kernel reads — the selection channel x of nodeState / nodeStateTarget, the highlight weight
//      y of edgeAttr — was seen less than `highlightSeconds` ago) or highlightFade moved. Writes
//      that only touch other channels (hover, dimming, metric, width) cost no evaluation: on a
//      version change the channel is hashed (O(N) or O(E), only then) and compared.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//  const curves = new EdgeCurves( { beta, highlightLevel, sphereAxis, memoryBudget } );
//  curves.setGraph( buffers, graph, { segments, paths, reserve } )   allocates, uploads the paths
//                                      (default hierarchicalPaths( graph )); reserve = a segment
//                                      count to keep room for (see Cache)
//  curves.setPaths( paths | null )     swap the control paths (null = hierarchical); throws a
//                                      RangeError / TypeError when the arrays do not fit the graph
//  curves.setSegments( S ) → S         segments per edge (samples = S + 1), clamped to the budget
//  curves.maxSegments()                largest S the memory budget allows for this graph
//  curves.update( app ) → boolean      run the kernel if needed (app.renderer, app.frame, app.time)
//  curves.invalidate( seconds = 0 )    force an evaluation now (and for `seconds` more)
//  curves.readback( renderer ) → Promise<Float32Array>   the samples (tests, CPU picking)
//  curves.samples / .pathRefs          storage nodes (null before setGraph / when E = 0)
//  curves.uniforms                     { beta, unhook, highlightLevel, sphereAxis, highlightFade, stride }
//  curves.segments / .capacity / .paths / .memoryBudget / .reserveBudget / .onRealloc
//  curves.stats                        { updates, skipped, lastUpdateFrame, segments, samples,
//                                        capacity, bytes, pathRefs, maxInterior, reallocations }
//  curves.dispose()
//  As an App module (order 9) it also has update( dt, app ); the EdgeRenderer owns one and drives
//  it from its own update, so pages need not register it.
//
//  Also exported: evalEdgeCurveCPU( graph, e, t, nodePos4, clusterPos4, options ), the pure-JS
//  mirror of the kernel (tests, CPU picking), maxSegmentsForBudget( E, bytes ), encodePathRefs().

import { Fn, If, Loop, instanceIndex, instancedArray, storage, uniform, float, int, uint, vec3, vec4, select, mix, clamp, max, min, pow, length, dot, sin, cos, atan, normalize, abs } from 'three/tsl';
import { Vector3 } from 'three/webgpu';
import { hierarchicalPaths } from '../graph/bundling.js';

export const CURVE_DEFAULTS = {
	beta: 0.85,
	unhook: 1, // 0..1: how far a centroid behind an end point of its edge is pulled back to it (see Curve)
	highlightLevel: 0.75, // sphere mode: level the cluster points of highlighted edges move to
	memoryBudget: 64 * 1024 * 1024, // bytes for the sample buffer
	reserveBudget: 8 * 1024 * 1024, // setGraph( { reserve } ): most bytes allocated ahead of need
	highlightSeconds: 1.2, // sphere mode: how long the kernel keeps running after a highlight change
};

const REF_CLUSTER = 0x80000000;
const REF_HEIGHT_SHIFT = 27;
const REF_INDEX_MASK = 0x07ffffff;

/** Largest segment count whose E · ( S + 1 ) vec4 samples fit into `bytes`. */
export function maxSegmentsForBudget( E, bytes = CURVE_DEFAULTS.memoryBudget ) {

	return Math.max( 1, Math.floor( bytes / ( 16 * Math.max( 1, E ) ) ) - 1 );

}

/**
 * GPU encoding of a path set's refs: bit 31 = cluster, bits 27..30 = cluster height (clamped to
 * 15), bits 0..26 = node or cluster index.
 * @param {object} graph N, [clusterHeight]
 * @param {Uint32Array} pathRefs
 * @param {number} count refs to encode
 * @param {Uint32Array} out at least `count` long
 */
export function encodePathRefs( graph, pathRefs, count, out ) {

	const N = graph.N, height = graph.clusterHeight;
	for ( let k = 0; k < count; k ++ ) {

		const ref = pathRefs[ k ];
		if ( ref < N ) out[ k ] = ref;
		else {

			const c = ref - N;
			const h = Math.min( 15, height ? height[ c ] : 1 );
			out[ k ] = ( REF_CLUSTER | ( h << REF_HEIGHT_SHIFT ) | c ) >>> 0;

		}

	}

	return out;

}

// ── CPU mirror ──────────────────────────────────────────────────────────────────────────────

/** Uniform cubic B-spline basis (b) and derivative (d) at u, as JS numbers. */
function basisCPU( u ) {

	const v = 1 - u, u2 = u * u, u3 = u2 * u;
	return {
		b: [ ( v * v * v ) / 6, ( 3 * u3 - 6 * u2 + 4 ) / 6, ( - 3 * u3 + 3 * u2 + 3 * u + 1 ) / 6, u3 / 6 ],
		d: [ - 0.5 * v * v, 1.5 * u2 - 2 * u, - 1.5 * u2 + u + 0.5, 0.5 * u2 ],
	};

}

const defaultPaths = new WeakMap();

/** hierarchicalPaths( g ), computed once per graph object. */
function hierarchicalPathsOf( g ) {

	let p = defaultPaths.get( g );
	if ( ! p ) {

		p = hierarchicalPaths( g );
		defaultPaths.set( g, p );

	}

	return p;

}

/**
 * Pure-JS mirror of the curve kernel (tests, CPU picking).
 * @param {object} g PreparedGraph
 * @param {number} e edge index
 * @param {number} t curve parameter in [0,1]
 * @param {Float32Array} nodePos4 node positions (vec4 per node)
 * @param {Float32Array} clusterPos4 cluster centroids (vec4 per cluster)
 * @param {object} [o] { paths (default: hierarchicalPaths( g )), beta, unhook, layoutBlend, sphereRadius,
 *                       center:[x,y,z], sphereAxis:[x,y,z], levelOffset, levelScale, levelExponent,
 *                       highlight, highlightLevel }
 * @returns {{ position: number[], tangent: number[], L: number, controlPoints: number[][] }}
 *   tangent = d position / d( span parameter ) (plus 1e-4 of the span chord, which only matters
 *   at the two ends where the tripled knots make the derivative vanish)
 */
export function evalEdgeCurveCPU( g, e, t, nodePos4, clusterPos4, o = {} ) {

	const paths = o.paths ?? hierarchicalPathsOf( g );
	const beta = o.beta ?? CURVE_DEFAULTS.beta;
	const unhook = o.unhook ?? CURVE_DEFAULTS.unhook;
	const blend = o.layoutBlend ?? 0;
	const R = o.sphereRadius ?? g.layoutRadius;
	const c = o.center ?? [ 0, 0, 0 ];
	const lvOff = o.levelOffset ?? 0.1, lvScale = o.levelScale ?? 0.35, lvExp = o.levelExponent ?? 2;
	const hl = o.highlight ?? 0, hlLevel = o.highlightLevel ?? CURVE_DEFAULTS.highlightLevel;
	const N = g.N;
	const s = g.edges[ 2 * e ], tg = g.edges[ 2 * e + 1 ];
	const start = paths.pathStart[ e ];
	const L = paths.pathStart[ e + 1 ] - start + 2;
	const pS = [ nodePos4[ 4 * s ], nodePos4[ 4 * s + 1 ], nodePos4[ 4 * s + 2 ] ];
	const pT = [ nodePos4[ 4 * tg ], nodePos4[ 4 * tg + 1 ], nodePos4[ 4 * tg + 2 ] ];
	const sub = ( a, b ) => [ a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] ];
	const lerp3 = ( a, b, k ) => [ a[ 0 ] + ( b[ 0 ] - a[ 0 ] ) * k, a[ 1 ] + ( b[ 1 ] - a[ 1 ] ) * k, a[ 2 ] + ( b[ 2 ] - a[ 2 ] ) * k ];
	const len = ( a ) => Math.hypot( a[ 0 ], a[ 1 ], a[ 2 ] );
	const nrm = ( a ) => {

		const l = Math.max( len( a ), 1e-6 );
		return [ a[ 0 ] / l, a[ 1 ] / l, a[ 2 ] / l ];

	};

	const ax = o.sphereAxis ?? [ 0, 0, 1 ];
	const dot3 = ( a, b ) => a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];
	const logMap = ( r ) => {

		const dn = nrm( r );
		const cs = dot3( dn, ax );
		const perp = [ dn[ 0 ] - ax[ 0 ] * cs, dn[ 1 ] - ax[ 1 ] * cs, dn[ 2 ] - ax[ 2 ] * cs ];
		const sn = len( perp );
		const k = sn > 1e-7 ? Math.atan2( sn, cs ) / sn : 1;
		return [ perp[ 0 ] * k, perp[ 1 ] * k, perp[ 2 ] * k ];

	};

	const chord = sub( pT, pS ), chordLen = len( chord ), chordDir = chord.map( ( v ) => v / Math.max( chordLen, 1e-6 ) );
	const rS = sub( pS, c ), rT = sub( pT, c );
	const levS = len( rS ) / R, levT = len( rT ) / R, vS = logMap( rS ), vT = logMap( rT );
	const point = ( k ) => {

		if ( k <= 0 ) return { p: pS, f: pS, lev: levS };
		if ( k >= L - 1 ) return { p: pT, f: pT, lev: levT };
		const ref = paths.pathRefs[ start + k - 1 ];
		if ( ref < N ) {

			// a node on the path keeps its own radial level
			const p = [ nodePos4[ 4 * ref ], nodePos4[ 4 * ref + 1 ], nodePos4[ 4 * ref + 2 ] ];
			return { p, f: p, lev: len( sub( p, c ) ) / R };

		}

		const cl = ref - N;
		const h = g.clusterHeight ? g.clusterHeight[ cl ] : 1;
		const level = 1 + lvOff + lvScale * ( h > 1 ? Math.pow( h - 1, lvExp ) : 0 );
		const p = [ clusterPos4[ 4 * cl ], clusterPos4[ 4 * cl + 1 ], clusterPos4[ 4 * cl + 2 ] ];
		// force layout: a centroid behind the source or beyond the target is pulled back to that
		// end point (see Curve), so that the curve does not hook around the node
		const q = sub( p, pS ), along = dot3( q, chordDir );
		const over = along - Math.min( Math.max( along, 0 ), chordLen );
		const side = [ q[ 0 ] - chordDir[ 0 ] * along, q[ 1 ] - chordDir[ 1 ] * along, q[ 2 ] - chordDir[ 2 ] * along ];
		const fade = Math.abs( over ) / Math.max( len( side ) + Math.abs( over ), 1e-6 );
		const f = [ 0, 1, 2 ].map( ( i ) => p[ i ] - ( chordDir[ i ] * over + side[ i ] * fade ) * unhook );
		return { p, f, lev: level + ( hlLevel - level ) * hl };

	};

	const controlPoints = [];
	for ( let k = 0; k < L; k ++ ) controlPoints.push( point( k ).p );

	const x = Math.min( Math.max( t, 0 ), 1 ) * ( L + 1 );
	const j = Math.min( Math.floor( x ), L );
	const u = x - j;
	const { b, d } = basisCPU( u );
	const cF = [ 0, 0, 0 ], tF = [ 0, 0, 0 ], V = [ 0, 0, 0 ], dV = [ 0, 0, 0 ];
	let cL = 0, dL = 0;
	const F = [], Sv = [], Sl = [];
	for ( let i = 0; i < 4; i ++ ) {

		const k = Math.min( Math.max( j + i - 2, 0 ), L - 1 );
		const a = k / ( L - 1 );
		const { p, f: pf, lev } = point( k );
		const f = sub( lerp3( lerp3( pS, pT, a ), pf, beta ), pS );
		const v = lerp3( lerp3( vS, vT, a ), logMap( sub( p, c ) ), beta );
		const lv = ( levS + ( levT - levS ) * a ) * ( 1 - beta ) + lev * beta;
		F.push( f ); Sv.push( v ); Sl.push( lv );
		for ( let q = 0; q < 3; q ++ ) {

			cF[ q ] += b[ i ] * f[ q ]; tF[ q ] += d[ i ] * f[ q ];
			V[ q ] += b[ i ] * v[ q ]; dV[ q ] += d[ i ] * v[ q ];

		}

		cL += b[ i ] * lv; dL += d[ i ] * lv;

	}

	for ( let q = 0; q < 3; q ++ ) {

		tF[ q ] += 1e-4 * ( F[ 3 ][ q ] - F[ 0 ][ q ] );
		dV[ q ] += 1e-4 * ( Sv[ 3 ][ q ] - Sv[ 0 ][ q ] );

	}

	dL += 1e-4 * ( Sl[ 3 ] - Sl[ 0 ] );
	const posF = [ pS[ 0 ] + cF[ 0 ], pS[ 1 ] + cF[ 1 ], pS[ 2 ] + cF[ 2 ] ];
	const r = len( V );
	const sinc = r > 1e-4 ? Math.sin( r ) / r : 1 - ( r * r ) / 6;
	const cr = Math.cos( r );
	const Ex = [ 0, 1, 2 ].map( ( q ) => ax[ q ] * cr + V[ q ] * sinc );
	const n = V.map( ( v ) => v / Math.max( r, 1e-12 ) );
	const nd = dot3( n, dV );
	const dE = [ 0, 1, 2 ].map( ( q ) => - ax[ q ] * r * sinc * nd + n[ q ] * cr * nd + ( dV[ q ] - n[ q ] * nd ) * sinc );
	const posS = [ 0, 1, 2 ].map( ( q ) => c[ q ] + Ex[ q ] * R * cL );
	const tanS = [ 0, 1, 2 ].map( ( q ) => R * ( Ex[ q ] * dL + dE[ q ] * cL ) );
	const position = lerp3( posF, posS, blend );
	const tangent = lerp3( tF, tanS, blend );
	return { position, tangent, L, controlPoints };

}

// ── GPU cache ───────────────────────────────────────────────────────────────────────────────

const roViews = new WeakMap();

/** Read-only binding of a shared storage node (the kernel never writes the graph buffers). */
function ro( node ) {

	let v = roViews.get( node );
	if ( ! v ) {

		v = storage( node.value, node.bufferType, node.bufferCount ).setName( ( node.name || 'buffer' ) + 'CurveRO' ).toReadOnly();
		roViews.set( node, v );

	}

	return v;

}

/**
 * Hash (FNV-1a over the float bits) of one channel of a vec4 attribute array: tells whether a CPU
 * write changed that channel.
 */
function channelHash( array, channel ) {

	const bits = new Uint32Array( array.buffer, array.byteOffset, array.length );
	let h = 0x811c9dc5;
	for ( let i = channel, n = bits.length; i < n; i += 4 ) h = Math.imul( h ^ bits[ i ], 0x01000193 );
	return h >>> 0;

}

function checkPaths( graph, paths ) {

	const { N, E, C } = graph;
	if ( ! paths || ! ( paths.pathStart instanceof Uint32Array ) || ! ( paths.pathRefs instanceof Uint32Array ) ) throw new TypeError( 'EdgeCurves.setPaths: pathStart and pathRefs must be Uint32Arrays' );
	const { pathStart, pathRefs } = paths;
	if ( pathStart.length !== E + 1 ) throw new RangeError( `EdgeCurves.setPaths: pathStart has ${pathStart.length} entries, expected E + 1 = ${E + 1}` );
	if ( pathStart[ 0 ] !== 0 ) throw new RangeError( 'EdgeCurves.setPaths: pathStart[0] must be 0' );
	const total = pathStart[ E ];
	if ( total > pathRefs.length ) throw new RangeError( `EdgeCurves.setPaths: pathStart[E] = ${total} exceeds pathRefs.length = ${pathRefs.length}` );
	let maxInterior = 0;
	for ( let e = 0; e < E; e ++ ) {

		const n = pathStart[ e + 1 ] - pathStart[ e ];
		if ( pathStart[ e + 1 ] < pathStart[ e ] ) throw new RangeError( `EdgeCurves.setPaths: pathStart decreases at edge ${e}` );
		if ( n > maxInterior ) maxInterior = n;

	}

	const limit = N + C;
	if ( limit > REF_INDEX_MASK ) throw new RangeError( 'EdgeCurves.setPaths: graph too large for the packed ref format' );
	for ( let k = 0; k < total; k ++ ) if ( pathRefs[ k ] >= limit ) throw new RangeError( `EdgeCurves.setPaths: pathRefs[${k}] = ${pathRefs[ k ]} is neither a node (< ${N}) nor a cluster (< ${limit})` );
	return { total, maxInterior };

}

export class EdgeCurves {

	constructor( options = {} ) {

		const o = { ...CURVE_DEFAULTS, ...options };
		this.order = 9;
		this.isEdgeCurves = true;
		this.buffers = null;
		this.graph = null;
		this.paths = null;
		this.samples = null;
		this.pathRefs = null;
		this.kernel = null;
		this.segments = 0;
		this.capacity = 0; // samples per edge the buffer can hold
		this.memoryBudget = o.memoryBudget;
		this.reserveBudget = o.reserveBudget;
		this.highlightSeconds = o.highlightSeconds;
		/** Called with ( curves ) after the sample buffer was re-allocated: materials that bound `samples` must be rebuilt. */
		this.onRealloc = null;

		this.uniforms = {
			beta: uniform( o.beta ),
			unhook: uniform( o.unhook ),
			highlightLevel: uniform( o.highlightLevel ),
			sphereAxis: uniform( new Vector3( ...( o.sphereAxis || [ 0, 0, 1 ] ) ).normalize() ),
			highlightFade: uniform( 1 ), // weight of edgeAttr.y in the sphere-mode highlight level
			stride: uniform( 2, 'uint' ), // samples per edge in use (S + 1)
		};

		this._refCapacity = 0;
		this._dirty = true;
		this._evalFrame = - 1;
		this._signature = [];
		this._versions = [ - 1, - 1, - 1 ];
		this._hashes = [ 0, 0, 0 ];
		this._hlUntil = - Infinity;
		this._hlFrames = 0;
		this._forceUntil = - Infinity;
		this._forceSeconds = 0;
		this._stats = { updates: 0, skipped: 0, lastUpdateFrame: - 1, reallocations: 0 };

	}

	get stats() {

		const E = this.graph ? this.graph.E : 0;
		return {
			...this._stats,
			segments: this.segments,
			samples: this.segments + 1,
			capacity: this.capacity,
			bytes: 16 * Math.max( 1, E * this.capacity ),
			pathRefs: this.paths ? this.paths.pathStart[ E ] : 0,
			maxInterior: this.paths ? this.paths.maxInterior : 0,
		};

	}

	/** Largest S the memory budget allows for the current graph. */
	maxSegments() {

		return maxSegmentsForBudget( this.graph ? this.graph.E : 0, this.memoryBudget );

	}

	// ── set up ─────────────────────────────────────────────────────────────────────────────

	/**
	 * @param {import('./GraphBuffers.js').GraphBuffers} buffers
	 * @param {object} graph PreparedGraph
	 * @param {object} [o] { segments: initial S (default 8), paths: initial path set, reserve: a
	 *   segment count the buffer should have room for from the start (e.g. the sphere layout's);
	 *   honoured while 16 · E · ( reserve + 1 ) stays within `reserveBudget` }
	 */
	setGraph( buffers, graph, { segments = 8, paths = null, reserve = 0 } = {} ) {

		this._disposeGPU();
		this.buffers = buffers;
		this.graph = graph;
		this.capacity = 0;
		this.segments = 0;
		this._refCapacity = 0;
		this._evalFrame = - 1;
		this._signature = [];
		// the highlight channels as they are now: only a later change starts the easing window
		this._versions = [ buffers.nodeStateTarget.value.version, buffers.nodeState.value.version, buffers.edgeAttr.value.version ];
		this._hashes = [ channelHash( buffers.nodeStateTarget.value.array, 0 ), channelHash( buffers.nodeState.value.array, 0 ), channelHash( buffers.edgeAttr.value.array, 1 ) ];
		this._hlUntil = - Infinity;
		this._hlFrames = 0;
		this._stats = { updates: 0, skipped: 0, lastUpdateFrame: - 1, reallocations: 0 };
		this.setPaths( paths );
		reserve = Math.min( this.maxSegments(), Math.round( reserve ) || 0 );
		if ( reserve > segments && 16 * graph.E * ( reserve + 1 ) <= this.reserveBudget ) this.setSegments( reserve );
		this.setSegments( segments );

	}

	/**
	 * Swap the control paths. `paths` = { pathStart, pathRefs, maxInterior } of src/graph/bundling.js
	 * for this graph, or null for hierarchicalPaths( graph ). Throws when the arrays do not fit.
	 */
	setPaths( paths = null ) {

		const g = this.graph;
		if ( ! g ) throw new Error( 'EdgeCurves.setPaths: no graph' );
		const p = paths || hierarchicalPathsOf( g );
		const { total, maxInterior } = checkPaths( g, p );
		const E = g.E;

		// pathStart goes into edgeInfo.w (element E = total)
		const info = this.buffers.edgeInfo.value;
		for ( let e = 0; e <= E; e ++ ) info.array[ 4 * e + 3 ] = p.pathStart[ e ];
		info.needsUpdate = true;

		if ( ! this.pathRefs || total > this._refCapacity ) {

			// a new buffer: the kernel binds it, so it is rebuilt on the next update
			const old = this.pathRefs;
			this._refCapacity = Math.max( 1, total );
			this.pathRefs = instancedArray( this._refCapacity, 'uint' ).setName( 'edgePathRefs' ).toReadOnly();
			this._disposeKernel();
			if ( old ) old.value.dispose();

		}

		encodePathRefs( g, p.pathRefs, total, this.pathRefs.value.array );
		this.pathRefs.value.needsUpdate = true;
		this.paths = { pathStart: p.pathStart, pathRefs: p.pathRefs, maxInterior: p.maxInterior ?? maxInterior };
		this._dirty = true;

	}

	/** Segments per edge (samples = S + 1). Clamped to [1, maxSegments()]. Returns the value in use. */
	setSegments( S ) {

		const g = this.graph;
		if ( ! g ) return 0;
		S = Math.max( 1, Math.min( this.maxSegments(), Math.round( S ) || 1 ) );
		if ( S + 1 > this.capacity ) {

			const old = this.samples;
			this.capacity = S + 1;
			this.samples = instancedArray( Math.max( 1, g.E * this.capacity ), 'vec4' ).setName( 'edgeCurveSamples' );
			this._disposeKernel();
			if ( old ) {

				this._stats.reallocations ++;
				this.segments = S;
				this.uniforms.stride.value = S + 1;
				this.onRealloc?.( this );
				old.value.dispose();

			}

		}

		if ( S !== this.segments ) this._dirty = true;
		this.segments = S;
		this.uniforms.stride.value = S + 1;
		return S;

	}

	/** Evaluate on the next update, and on every update for `seconds` after it. */
	invalidate( seconds = 0 ) {

		this._dirty = true;
		if ( seconds > 0 ) this._forceSeconds = Math.max( this._forceSeconds || 0, seconds );

	}

	// ── per frame ──────────────────────────────────────────────────────────────────────────

	/**
	 * Runs the kernel when something the curves depend on changed. Call after the layout moved the
	 * nodes and before rendering. `app` needs { renderer, frame, time }. Also accepts ( dt, app ).
	 * @returns {boolean} whether the kernel ran
	 */
	update( a, b ) {

		const app = b || a;
		const g = this.graph;
		if ( ! g || g.E === 0 || ! app || ! app.renderer ) return false;
		const buffers = this.buffers, U = buffers.uniforms, u = this.uniforms;
		const frame = app.frame ?? 0, time = app.time ?? 0;
		let dirty = this._dirty;

		if ( this._forceSeconds ) {

			this._forceUntil = time + this._forceSeconds;
			this._forceSeconds = 0;

		}

		if ( time < this._forceUntil ) dirty = true;

		// node positions (and with them the cluster centroids)
		const mf = buffers.moveFrame;
		if ( ! ( mf < this._evalFrame ) ) dirty = true;

		// parameters, by value
		const sphere = U.layoutBlend.value > 0;
		const c = U.graphCenter.value, ax = u.sphereAxis.value;
		const sig = [
			u.beta.value, u.unhook.value, u.stride.value, U.layoutBlend.value, U.sphereRadius.value, U.levelOffset.value, U.levelScale.value,
			U.levelExponent.value, c.x, c.y, c.z, ax.x, ax.y, ax.z, u.highlightLevel.value, sphere ? u.highlightFade.value : 0,
		];
		const old = this._signature;
		for ( let i = 0; i < sig.length; i ++ ) if ( sig[ i ] !== old[ i ] ) dirty = true;
		this._signature = sig;

		// highlight easing only bends the curves in sphere mode. CPU writes to the state buffers
		// are visible as attribute versions; the GPU then eases for a few hundred milliseconds.
		// Only the channels the kernel reads count (selection x, highlight weight y): a hover or a
		// metric write bumps the version too, and must not cost a second of evaluations
		const v = this._versions, hashes = this._hashes;
		const watched = [ buffers.nodeStateTarget.value, buffers.nodeState.value, buffers.edgeAttr.value ];
		for ( let k = 0; k < 3; k ++ ) {

			const attribute = watched[ k ];
			if ( attribute.version === v[ k ] ) continue;
			v[ k ] = attribute.version;
			const hash = channelHash( attribute.array, k === 2 ? 1 : 0 );
			if ( hash === hashes[ k ] ) continue;
			hashes[ k ] = hash;
			this._hlUntil = time + this.highlightSeconds;
			this._hlFrames = 20;

		}

		if ( sphere && ( time < this._hlUntil || this._hlFrames > 0 ) ) dirty = true;
		if ( this._hlFrames > 0 ) this._hlFrames --;

		if ( ! dirty ) {

			this._stats.skipped ++;
			return false;

		}

		if ( ! this.kernel ) this._buildKernel();
		app.renderer.compute( this.kernel );
		this._dirty = false;
		this._evalFrame = frame;
		this._stats.updates ++;
		this._stats.lastUpdateFrame = frame;
		return true;

	}

	/** Reads the sample buffer back: vec4 per sample, sample i of edge e at 4 · ( e · ( segments + 1 ) + i ). */
	async readback( renderer ) {

		if ( ! this.samples ) return new Float32Array( 0 );
		return new Float32Array( await renderer.getArrayBufferAsync( this.samples.value ) );

	}

	dispose() {

		this._disposeGPU();
		this.buffers = null;
		this.graph = null;
		this.paths = null;

	}

	// ── internals ──────────────────────────────────────────────────────────────────────────

	_disposeKernel() {

		if ( this.kernel ) {

			this.kernel.dispose();
			this.kernel = null;

		}

		this._dirty = true;

	}

	_disposeGPU() {

		this._disposeKernel();
		if ( this.samples ) this.samples.value.dispose();
		if ( this.pathRefs ) this.pathRefs.value.dispose();
		this.samples = null;
		this.pathRefs = null;

	}

	_buildKernel() {

		const { buffers, graph } = this;
		const U = buffers.uniforms, u = this.uniforms;
		const nodePos = ro( buffers.nodePos ), clusterPos = ro( buffers.clusterPos ), nodeState = ro( buffers.nodeState );
		const edgeInfo = ro( buffers.edgeInfo ), edgeAttr = ro( buffers.edgeAttr );
		const refs = this.pathRefs, out = this.samples;

		this.kernel = Fn( () => {

			const e = instanceIndex;
			const info = edgeInfo.element( e ).toVar();
			const start = info.w.toVar();
			const L = int( edgeInfo.element( e.add( uint( 1 ) ) ).w.sub( start ) ).add( int( 2 ) ).toVar();
			const Lm1 = L.sub( int( 1 ) ).toVar();
			const pS = nodePos.element( info.x ).xyz.toVar();
			const pT = nodePos.element( info.y ).xyz.toVar();
			const beta = clamp( u.beta, 0, 1 ).toVar();
			const unhook = clamp( u.unhook, 0, 1 ).toVar();
			const chordLen = length( pT.sub( pS ) ).toVar();
			const chordDir = pT.sub( pS ).div( max( chordLen, 1e-6 ) ).toVar();
			const blend = clamp( U.layoutBlend, 0, 1 ).toVar();
			const center = U.graphCenter;
			const R = max( U.sphereRadius, 1e-6 ).toVar();
			const axis = u.sphereAxis;

			// sphere mode: log map around the axis (azimuthal-equidistant tangent plane: the planar
			// treemap space of prepare.js), which keeps the angular speed nearly uniform on long hops
			const logMap = ( r ) => {

				const dn = r.div( max( length( r ), 1e-12 ) ).toVar();
				const cs = dot( dn, axis ).toVar();
				const perp = dn.sub( axis.mul( cs ) ).toVar();
				const sn = length( perp ).toVar();
				return perp.mul( select( sn.greaterThan( 1e-7 ), atan( sn, cs ).div( max( sn, 1e-7 ) ), float( 1 ) ) );

			};

			const levS = float( 1 ).toVar(), levT = float( 1 ).toVar(), hl = float( 0 ).toVar();
			const vS = vec3( 0 ).toVar(), vT = vec3( 0 ).toVar();
			If( blend.greaterThan( 0 ), () => {

				const rS = pS.sub( center ), rT = pT.sub( center );
				levS.assign( length( rS ).div( R ) );
				levT.assign( length( rT ).div( R ) );
				vS.assign( logMap( rS ) );
				vT.assign( logMap( rT ) );
				hl.assign( clamp( max( max( nodeState.element( info.x ).x, nodeState.element( info.y ).x ), edgeAttr.element( e ).y.mul( u.highlightFade ) ), 0, 1 ) );

			} );

			const S = int( u.stride ).sub( int( 1 ) ).toVar();
			const base = e.mul( u.stride ).toVar();
			const prev = vec3( pS ).toVar();
			const acc = float( 0 ).toVar();

			Loop( { start: int( 0 ), end: int( u.stride ), type: 'int', condition: '<' }, ( { i } ) => {

				// span j and local parameter uu of x = i (L+1) / S, in integers (exact at the knots)
				const q = i.mul( L.add( int( 1 ) ) ).toVar();
				const j = min( q.div( S ), L ).toVar();
				const uu = float( q.sub( j.mul( S ) ) ).div( float( S ) ).toVar();
				const w = float( 1 ).sub( uu );
				const u2 = uu.mul( uu ), u3 = u2.mul( uu );
				const bs = [
					w.mul( w ).mul( w ).div( 6 ),
					u3.mul( 3 ).sub( u2.mul( 6 ) ).add( 4 ).div( 6 ),
					u3.mul( - 3 ).add( u2.mul( 3 ) ).add( uu.mul( 3 ) ).add( 1 ).div( 6 ),
					u3.div( 6 ),
				].map( ( n ) => n.toVar() );

				// the span's 4 control points: position, and the radial level for sphere mode
				const P = [], PF = [], LV = [], A = [];
				for ( let m = 0; m < 4; m ++ ) {

					const k = clamp( j.add( int( m - 2 ) ), int( 0 ), Lm1 ).toVar();
					const pk = vec3( 0 ).toVar();
					const lev = float( 1 ).toVar();
					const pf = vec3( 0 ).toVar(); // the point the force-layout curve uses
					If( k.lessThanEqual( int( 0 ) ), () => {

						pk.assign( pS ); pf.assign( pS ); lev.assign( levS );

					} ).ElseIf( k.greaterThanEqual( Lm1 ), () => {

						pk.assign( pT ); pf.assign( pT ); lev.assign( levT );

					} ).Else( () => {

						const ref = refs.element( start.add( uint( k ) ).sub( uint( 1 ) ) ).toVar();
						If( ref.lessThan( uint( REF_CLUSTER ) ), () => {

							// a node on the path: its own radial level
							pk.assign( nodePos.element( ref ).xyz );
							pf.assign( pk );
							If( blend.greaterThan( 0 ), () => {

								lev.assign( length( pk.sub( center ) ).div( R ) );

							} );

						} ).Else( () => {

							pk.assign( clusterPos.element( ref.bitAnd( uint( REF_INDEX_MASK ) ) ).xyz );
							// a centroid behind the source or beyond the target: pulled back to that end
							// point, so that the curve does not hook around the node (see Curve)
							const q = pk.sub( pS ).toVar();
							const along = dot( q, chordDir ).toVar();
							const over = along.sub( clamp( along, 0, chordLen ) ).toVar();
							const side = q.sub( chordDir.mul( along ) ).toVar();
							const fade = abs( over ).div( max( length( side ).add( abs( over ) ), 1e-6 ) );
							pf.assign( pk.sub( chordDir.mul( over ).add( side.mul( fade ) ).mul( unhook ) ) );
							If( blend.greaterThan( 0 ), () => {

								const hm1 = float( ref.shiftRight( uint( REF_HEIGHT_SHIFT ) ).bitAnd( uint( 15 ) ) ).sub( 1 ).toVar();
								const level = float( 1 ).add( U.levelOffset ).add( U.levelScale.mul( select( hm1.greaterThan( 0.5 ), pow( max( hm1, 1e-6 ), U.levelExponent ), float( 0 ) ) ) );
								lev.assign( mix( level, u.highlightLevel, hl ) );

							} );

						} );

					} );

					P.push( pk ); PF.push( pf ); LV.push( lev ); A.push( float( k ).div( float( Lm1 ) ).toVar() );

				}

				// force layout: Euclidean B-spline over the straightened points, relative to the source
				const F = PF.map( ( pf, m ) => mix( mix( pS, pT, A[ m ] ), pf, beta ).sub( pS ) );
				const position = pS.add( F[ 0 ].mul( bs[ 0 ] ) ).add( F[ 1 ].mul( bs[ 1 ] ) ).add( F[ 2 ].mul( bs[ 2 ] ) ).add( F[ 3 ].mul( bs[ 3 ] ) ).toVar();

				If( blend.greaterThan( 0 ), () => {

					const Sv = P.map( ( pk, m ) => mix( mix( vS, vT, A[ m ] ), logMap( pk.sub( center ) ), beta ) );
					const Sl = LV.map( ( lev, m ) => mix( mix( levS, levT, A[ m ] ), lev, beta ) );
					const V = Sv[ 0 ].mul( bs[ 0 ] ).add( Sv[ 1 ].mul( bs[ 1 ] ) ).add( Sv[ 2 ].mul( bs[ 2 ] ) ).add( Sv[ 3 ].mul( bs[ 3 ] ) ).toVar();
					const cL = Sl[ 0 ].mul( bs[ 0 ] ).add( Sl[ 1 ].mul( bs[ 1 ] ) ).add( Sl[ 2 ].mul( bs[ 2 ] ) ).add( Sl[ 3 ].mul( bs[ 3 ] ) );
					// exponential map: E = axis cos r + V sin( r ) / r, a unit vector because V is
					// perpendicular to the axis; normalised because WGSL sin / cos are only about 1e-4
					// accurate on some implementations, which would show as a radial error
					const r = length( V ).toVar();
					const sinc = select( r.greaterThan( 1e-4 ), sin( r ).div( max( r, 1e-4 ) ), r.mul( r ).div( - 6 ).add( 1 ) );
					const dir = normalize( axis.mul( cos( r ) ).add( V.mul( sinc ) ) );
					position.assign( mix( position, center.add( dir.mul( R.mul( cL ) ) ), blend ) );

				} );

				// the ends are the node centres by construction; snap them so the sphere-mode trig
				// (WGSL sin / cos / atan are only about 1e-4 accurate) cannot leave an offset
				If( i.equal( int( 0 ) ), () => {

					position.assign( pS );

				} ).ElseIf( i.equal( S ), () => {

					position.assign( pT );

				} );

				acc.addAssign( length( position.sub( prev ) ) );
				prev.assign( position );
				out.element( base.add( uint( i ) ) ).assign( vec4( position, acc ) );

			} );

		} )().compute( graph.E, [ 64 ] ).setName( 'edges.curves' );

	}

}
