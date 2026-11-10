// Pluggable edge bundling: explicit per-edge control paths for the edge renderer, and the
// algorithms that produce them. Pure JS, typed arrays in and out, no three.js and no imports:
// runs on the main thread, in a Web Worker, or in Node (tests). Deterministic.
//
// ─── Control paths ──────────────────────────────────────────────────────────────────────────
//  Every bundling algorithm returns the same structure:
//    pathStart   Uint32Array( E + 1 )  prefix offsets into pathRefs
//    pathRefs    Uint32Array( total )  the INTERIOR control points of edge e are
//                                      pathRefs[ pathStart[e] .. pathStart[e+1] ), ordered from
//                                      the source edges[2e] to the target edges[2e+1]
//    maxInterior number                largest interior count over all edges
//  A ref < N is the live position of node `ref`; a ref >= N is the live centroid of cluster
//  ref - N (cluster ids as in prepare.js). The full control polygon of edge e is
//  [ s, ...interior, t ]; the renderer applies Holten straightening (beta) and a uniform cubic
//  B-spline with tripled end points to it. An edge without interior points is a straight line.
//  Because the refs are indices, not coordinates, a path set stays valid while the layout
//  moves, nodes are dragged, or the view morphs to the sphere layout. Only the *choice* of
//  paths of the position-dependent algorithms goes stale; recompute it when the layout settles.
//
// ─── API ────────────────────────────────────────────────────────────────────────────────────
//  straightPaths( E )                          no interior points
//  hierarchicalPaths( graph )                  Holten 2006 / ImsvGraphVis: the cluster centroids
//                                              of the endpoints' ancestors up to the LCA level;
//                                              exactly the polygon the renderer used implicitly
//  edgePathBundling( graph, positions, options )      Edge-Path Bundling, run to completion or
//                                              to options.timeBudget
//  edgePathBundlingTask( graph, positions, options )  the same as a resumable task
//                                              { total, completed, done, phase, step( n ),
//                                                result(), hurry() }, for a worker that reports
//                                              progress and can be cancelled (same shape as the
//                                              analytics.js tasks: runTask / runTaskAsync drive it)
//  mixedPaths( graph, primary, fallback, options )    primary paths, with the fallback's paths for
//                                              the edges the primary left straight
//  validatePaths( graph, paths, options )      structural checks, optionally "every path is a
//                                              real path of the graph" and a distortion bound
//  pathStats( graph, paths, positions, options )      bundled count, distortion, ink ratio
//  pathPolygon( graph, paths, e )              [ s, ...interior, t ] as refs
//  clusterCentroids( graph, positions )        CPU mirror of the GPU centroid pass (vec4 per cluster)
//
// ─── Edge-Path Bundling ─────────────────────────────────────────────────────────────────────
//  Wallinger, Archambault, Auber, Nöllenburg, Peltonen: "Edge-Path Bundling: A Less Ambiguous
//  Edge Bundling Approach", IEEE TVCG 28(1), 2022. An edge is routed along a path of OTHER
//  EDGES OF THE GRAPH, so a bundle never suggests a connection that does not exist, and every
//  edge that carries a bundle stays a straight line. w(e) = |e|^exponent, |e| = Euclidean
//  length in the given layout; B = maxDistortion * |e| is the longest path an edge may take.
//
//  method 'epb' (the 2022 paper's algorithm), one phase 'bundle':
//    Edges in order of decreasing length. A locked edge is passed over. Otherwise the edge is
//    skipped (taken out of the search graph) and the lightest path from s to t over the
//    non-skipped edges is searched. If its Euclidean length is at most B the edge is bundled:
//    the path's interior nodes become its control points, the path's edges are locked (never
//    bundled themselves), and the edge stays skipped. Otherwise it is un-skipped and straight.
//
//  method 'spanner' (the same authors' faster variant: "Faster Edge-Path Bundling through
//  Graph Spanners", Computer Graphics Forum 42(6), 2023), two phases:
//    'spanner'  the greedy t-spanner H of the layout, t = maxDistortion. Edges in order of
//               increasing length; an edge joins H unless H already connects its end points
//               by a path of Euclidean length <= B. Spanner edges stay straight.
//    'route'    every other edge is routed along the lightest path in H, longest edge first.
//  Both phases search a sparse graph and the second one has no sequential dependency, which
//  makes this the faster method; it is the default.
//
//  options.exact = true runs the papers' rules to the letter (the tests compare both methods
//  with literal re-implementations of the pseudocode). The default deviates on purpose:
//   1. The lightest path is searched among the paths of length <= B (and <= maxInterior
//      interior nodes), not among all paths. Where the papers leave an edge straight because
//      its lightest path is too long, the default bundles it along a slightly heavier path
//      that fits. Where both rules bundle an edge they choose the same path for 98-100% of
//      those edges (measured); the default bundles up to about 20% more edges.
//   2. A hop limit (maxInterior), so the cost per edge in the shader stays bounded.
//   3. An effort limit per search (options.effort node expansions). A spanner test that runs
//      out says "no path" (the edge stays straight, which is always safe); a lightest-path
//      search that runs out keeps the lightest path it has.
//   4. 'spanner': an edge outside H keeps the path its spanner test found until, and unless,
//      phase 'route' finds a lighter one. So a result is complete as soon as phase 'spanner'
//      is done, and 'route' is a refinement that can be cut short.
//   5. 'epb': a short path is searched first (A*); without one the edge stays straight, with
//      one its weight bounds the lightest-path search.
//
// ─── Searches ───────────────────────────────────────────────────────────────────────────────
//  The search graph is a CSR whose per-node slot lists have an *active prefix*: bundled edges
//  are swapped out of it ('epb'), spanner edges into it ('spanner'), O(1) per change, so a
//  search only ever touches usable edges. Two routines run on it, both on a binary heap:
//   - shortest (Euclidean) path: A* with the straight distance to the target as potential.
//     That potential is consistent because the weight of an edge is its length. It is
//     weighted by options.greed: only *a* path within B is needed, and a greedier search
//     finds one sooner. Labels with length + distance to the target > B are dropped (exact).
//   - lightest path (w = |e|^k): bidirectional Dijkstra. A* is of no use here: for k > 1 a
//     cheap path is a chain of short hops, whose cost per unit of distance goes to zero with
//     the hop length, so the only consistent distance-based potential is lmin^(k-1) * distance
//     (lmin = the shortest edge), which steers nothing. Two balls that meet in the middle
//     need no potential and are correct for any non-negative weights: the search stops when
//     the two smallest keys add up to at least the lightest meeting found so far.
//     Sound bounds: only paths lighter than a known one are wanted (the path of the spanner
//     test or of the A* pass; in exact 'epb', B^k, since a path of length <= B weighs at most
//     B^k for k >= 1). Bounds of the default mode: a label is dropped when its Euclidean
//     length plus the straight distance to the target exceeds B, when it has used up the hop
//     limit, or when even the cheapest completion within the remaining hops (distance^k /
//     hops^(k-1)) cannot beat the best path so far. A node keeps only its lightest label, so
//     with these bounds the search returns the lightest path it can see, not always the
//     lightest that exists. Every path returned satisfies the length and hop limits.
//
// ─── Options ────────────────────────────────────────────────────────────────────────────────
//  method         'spanner' | 'epb'                                      default 'spanner'
//  exponent       k of w(e) = |e|^k (papers: 2; 1..3 are sensible; larger = paths prefer short
//                 edges more, bundles get stronger and paths longer)     default 2
//  maxDistortion  path length <= maxDistortion * |e| (papers: 2; 1.5..3) default 2
//  maxInterior    hop limit: at most this many interior control points per edge
//                 (Infinity = none)                                      default 8
//  effort         node expansions per search (Infinity = no limit)       default 256
//  greed          weight of the A* potential, >= 1                       default 1.5
//  exact          the papers' rules: no pruning by length or hops, no effort limit, greed 1,
//                 no provisional paths; maxInterior is then only checked on the final path.
//                 About 3x slower at 5,000 nodes / 40,000 edges          default false
//  timeBudget     ms, for edgePathBundling(): it returns when the time is used up. 'epb' has
//                 then bundled the longest edges and left the rest straight. 'spanner' cannot
//                 work longest-first (the spanner grows from the short edges), so when its
//                 first phase would not finish in time at the recent pace it switches to
//                 quick tests (task.hurry()), and phase 'route' refines what time is left.
//                 A result cut by the clock depends on the machine.     default Infinity
//  maxWork        the same cut as a number of edge relaxations: deterministic  default Infinity
//  stride         floats per node in `positions`                         default 4
//
// ─── Result ─────────────────────────────────────────────────────────────────────────────────
//  { pathStart, pathRefs, maxInterior,
//    state: Uint8Array( E )   EDGE_FREE (0) straight, EDGE_BUNDLED (1), EDGE_CARRYING (2):
//                             straight and part of at least one bundled path
//    load:  Uint32Array( E )  number of bundled edges routed over each edge (a width or
//                             brightness channel for the carrying edges)
//    stats: { method, edges, bundled, carrying, free, bundledRatio, spannerEdges,
//             meanDistortion, maxDistortion, meanInterior, maxInterior, inkRatio,
//             searches, relaxations, timeMs, complete, hurried, phase } }
//  Invariants: all refs are nodes; the polygon of a bundled edge is a simple path of the graph
//  whose edges are all EDGE_CARRYING (never bundled themselves); its length is at most
//  maxDistortion * |e|; it has at most maxInterior interior nodes.
//  inkRatio = ( total length of the straight edges ) / ( total length of all edges ): bundled
//  edges add no ink, they run over edges that are drawn anyway.
//  Edges with a non-finite or zero length and self loops are never bundled; the first and the
//  last kind are never part of a path either.
//
// ─── Complexity ─────────────────────────────────────────────────────────────────────────────
//  hierarchicalPaths O( E * D ). Edge-path bundling: at most two searches per edge. With the
//  default limits a search costs at most `effort` expansions, so the total is
//  O( E * effort * ( degree + log ) ); exact mode has the papers' worst case
//  O( E * ( E + N log N ) ). Memory O( N + E ) plus the paths. Measured timings are in the
//  recipe document of the overhaul (bundling.md).

export const EDGE_FREE = 0;
export const EDGE_BUNDLED = 1;
export const EDGE_CARRYING = 2;

export const BUNDLING_DEFAULTS = {
	method: 'spanner',
	exponent: 2,
	maxDistortion: 2,
	maxInterior: 8,
	exact: false,
	timeBudget: Infinity,
	maxWork: Infinity,
	stride: 4,
	effort: 256,
	greed: 1.5,
};

const NO_LIMIT = 0x7fffffff;

// spanner tests after task.hurry(): node expansions per test, and the weight of the A* potential
const HURRY_EFFORT = 24;
const HURRY_GREED = 3;

const now = typeof performance !== 'undefined' && performance.now ? () => performance.now() : () => Date.now();

// ─── Simple path sets ────────────────────────────────────────────────────────────────────────

/**
 * Paths without interior points: every edge is a straight line.
 * @param {number} E edge count
 * @returns {{ pathStart: Uint32Array, pathRefs: Uint32Array, maxInterior: number }}
 */
export function straightPaths( E ) {

	return { pathStart: new Uint32Array( ( E >>> 0 ) + 1 ), pathRefs: new Uint32Array( 0 ), maxInterior: 0 };

}

/**
 * Hierarchical edge bundling (Holten 2006), the polygon of prepare.js made explicit: for an
 * edge (s, t) with LCA level lca the interior control points are the clusters
 *   anc[s][0], …, anc[s][lca-1], [ anc[s][lca] if lca < D ], anc[t][lca-1], …, anc[t][0]
 * as refs N + cluster. 2*lca + ( lca < D ? 1 : 0 ) points per edge; the root is never used.
 * Time and memory O( E * D ).
 *
 * @param {object} graph PreparedGraph (N, E, D, edges, edgeLca, anc)
 * @returns {{ pathStart: Uint32Array, pathRefs: Uint32Array, maxInterior: number }}
 */
export function hierarchicalPaths( graph ) {

	const { N, E, D, edges, edgeLca, anc } = graph;
	const pathStart = new Uint32Array( E + 1 );
	let maxInterior = 0;
	for ( let e = 0; e < E; e ++ ) {

		const lca = Math.min( edgeLca[ e ], D );
		const n = 2 * lca + ( lca < D ? 1 : 0 );
		pathStart[ e + 1 ] = pathStart[ e ] + n;
		if ( n > maxInterior ) maxInterior = n;

	}

	const pathRefs = new Uint32Array( pathStart[ E ] );
	for ( let e = 0; e < E; e ++ ) {

		const s = edges[ 2 * e ], t = edges[ 2 * e + 1 ];
		const lca = Math.min( edgeLca[ e ], D );
		let o = pathStart[ e ];
		for ( let l = 0; l < lca; l ++ ) pathRefs[ o ++ ] = N + anc[ s * D + l ];
		if ( lca < D ) pathRefs[ o ++ ] = N + anc[ s * D + lca ];
		for ( let l = lca - 1; l >= 0; l -- ) pathRefs[ o ++ ] = N + anc[ t * D + l ];

	}

	return { pathStart, pathRefs, maxInterior };

}

/**
 * The control polygon of one edge as refs: [ s, ...interior, t ].
 * @returns {number[]}
 */
export function pathPolygon( graph, paths, e ) {

	const out = [ graph.edges[ 2 * e ] ];
	for ( let k = paths.pathStart[ e ]; k < paths.pathStart[ e + 1 ]; k ++ ) out.push( paths.pathRefs[ k ] );
	out.push( graph.edges[ 2 * e + 1 ] );
	return out;

}

/**
 * Combines two path sets: an edge keeps its path of `primary` when it has one, and gets its path
 * of `fallback` otherwise. Meant for edge-path bundling with hierarchical paths for the edges
 * it left straight.
 *
 * options.only (when primary carries a `state` array, i.e. is an edgePathBundling result):
 *   'free'      (default) only EDGE_FREE edges take the fallback. The carrying edges stay
 *               straight, so the bundles still run along lines that are drawn.
 *   'straight'  every edge without a primary path takes the fallback.
 *
 * @param {object} graph anything with E
 * @param {object} primary path set, optionally with `state`
 * @param {object} fallback path set of the same graph
 * @param {object} [options]
 * @returns {{ pathStart: Uint32Array, pathRefs: Uint32Array, maxInterior: number, source: Uint8Array }}
 *   source[e]: 0 straight, 1 from primary, 2 from fallback
 */
export function mixedPaths( graph, primary, fallback, options = {} ) {

	const E = graph.E;
	const only = options.only || 'free';
	const state = primary.state || null;
	const source = new Uint8Array( E );
	const pathStart = new Uint32Array( E + 1 );
	let maxInterior = 0;
	for ( let e = 0; e < E; e ++ ) {

		let n = primary.pathStart[ e + 1 ] - primary.pathStart[ e ];
		if ( n > 0 ) source[ e ] = 1;
		else if ( ! ( only === 'free' && state && state[ e ] === EDGE_CARRYING ) ) {

			n = fallback.pathStart[ e + 1 ] - fallback.pathStart[ e ];
			if ( n > 0 ) source[ e ] = 2;

		}

		pathStart[ e + 1 ] = pathStart[ e ] + n;
		if ( n > maxInterior ) maxInterior = n;

	}

	const pathRefs = new Uint32Array( pathStart[ E ] );
	for ( let e = 0; e < E; e ++ ) {

		if ( source[ e ] === 0 ) continue;
		const src = source[ e ] === 1 ? primary : fallback;
		const from = src.pathStart[ e ];
		pathRefs.set( src.pathRefs.subarray( from, from + pathStart[ e + 1 ] - pathStart[ e ] ), pathStart[ e ] );

	}

	return { pathStart, pathRefs, maxInterior, source };

}

// ─── Edge-Path Bundling ──────────────────────────────────────────────────────────────────────

/**
 * Edge ids sorted by ( length ascending, id ascending ). LSD radix sort on the float32 bit
 * patterns (non-negative floats order like their bits), then one insertion pass with the exact
 * doubles, which only has to fix the order inside runs of equal float32 keys. O( n ).
 */
function orderByLength( len, ids ) {

	const n = ids.length;
	const key = new Float32Array( n );
	for ( let i = 0; i < n; i ++ ) key[ i ] = len[ ids[ i ] ];
	const bits = new Uint32Array( key.buffer );
	let a = new Uint32Array( n ), b = new Uint32Array( n );
	for ( let i = 0; i < n; i ++ ) a[ i ] = i;
	const count = new Uint32Array( 2048 );
	for ( let shift = 0; shift < 32; shift += 11 ) {

		count.fill( 0 );
		for ( let i = 0; i < n; i ++ ) count[ ( bits[ i ] >>> shift ) & 2047 ] ++;
		let sum = 0;
		for ( let d = 0; d < 2048; d ++ ) {

			const c = count[ d ];
			count[ d ] = sum;
			sum += c;

		}

		for ( let i = 0; i < n; i ++ ) {

			const x = a[ i ];
			b[ count[ ( bits[ x ] >>> shift ) & 2047 ] ++ ] = x;

		}

		const tmp = a; a = b; b = tmp;

	}

	const out = b;
	for ( let i = 0; i < n; i ++ ) out[ i ] = ids[ a[ i ] ];
	for ( let i = 1; i < n; i ++ ) {

		const x = out[ i ], lx = len[ x ];
		let j = i - 1;
		while ( j >= 0 && ( len[ out[ j ] ] > lx || ( len[ out[ j ] ] === lx && out[ j ] > x ) ) ) {

			out[ j + 1 ] = out[ j ];
			j --;

		}

		out[ j + 1 ] = x;

	}

	return out;

}

// binary min-heap on parallel arrays ( key, node ); entries are never updated in place: a lighter
// label is pushed again and the old entry is recognised as settled when it surfaces
function heapPush( hk, hn, i, key, node ) {

	while ( i > 0 ) {

		const p = ( i - 1 ) >> 1;
		if ( hk[ p ] <= key ) break;
		hk[ i ] = hk[ p ];
		hn[ i ] = hn[ p ];
		i = p;

	}

	hk[ i ] = key;
	hn[ i ] = node;

}

// removes the root; `size` is the size after the removal
function heapPop( hk, hn, size ) {

	const key = hk[ size ], node = hn[ size ];
	let i = 0;
	for ( ;; ) {

		let c = 2 * i + 1;
		if ( c >= size ) break;
		if ( c + 1 < size && hk[ c + 1 ] < hk[ c ] ) c ++;
		if ( hk[ c ] >= key ) break;
		hk[ i ] = hk[ c ];
		hn[ i ] = hn[ c ];
		i = c;

	}

	hk[ i ] = key;
	hn[ i ] = node;

}

/**
 * The search graph and the two searches (see "Searches" in the header).
 * Slots of node u are [ adjStart[u], adjStart[u+1] ); the active ones are the prefix up to
 * adjEnd[u]. Edges with a non-finite length and self loops are left out entirely.
 * Memory is laid out for the inner loops: a relaxation touches one cache line of positions and
 * one or two of labels (both searches' labels of a node sit side by side).
 */
function createRouter( N, E, edges, positions, stride, exponent, allActive ) {

	const P = new Float64Array( 4 * Math.max( 1, N ) ); // x, y, z, - per node
	for ( let i = 0; i < N; i ++ ) {

		P[ 4 * i ] = positions[ i * stride ];
		P[ 4 * i + 1 ] = positions[ i * stride + 1 ];
		P[ 4 * i + 2 ] = positions[ i * stride + 2 ];

	}

	const len = new Float64Array( Math.max( 1, E ) );
	const usable = new Uint8Array( Math.max( 1, E ) );
	const adjStart = new Uint32Array( N + 1 );
	let usableCount = 0;
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		const dx = P[ 4 * a ] - P[ 4 * b ], dy = P[ 4 * a + 1 ] - P[ 4 * b + 1 ], dz = P[ 4 * a + 2 ] - P[ 4 * b + 2 ];
		const l = Math.sqrt( dx * dx + dy * dy + dz * dz );
		if ( a !== b && l < Infinity ) { // false for NaN too

			len[ e ] = l;
			usable[ e ] = 1;
			usableCount ++;
			adjStart[ a + 1 ] ++;
			adjStart[ b + 1 ] ++;

		}

	}

	for ( let i = 0; i < N; i ++ ) adjStart[ i + 1 ] += adjStart[ i ];
	// slots: AI[ 2k ] = neighbour, AI[ 2k + 1 ] = edge id, AF[ 2k ] = length, AF[ 2k + 1 ] = weight
	const S = Math.max( 1, 2 * usableCount );
	const AI = new Uint32Array( 2 * S ), AF = new Float64Array( 2 * S );
	const slotA = new Uint32Array( Math.max( 1, E ) ), slotB = new Uint32Array( Math.max( 1, E ) );
	const adjEnd = new Uint32Array( Math.max( 1, N ) );
	{

		const fill = adjStart.slice( 0, N );
		for ( let e = 0; e < E; e ++ ) {

			if ( ! usable[ e ] ) continue;
			const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
			const l = len[ e ];
			const w = exponent === 2 ? l * l : exponent === 1 ? l : Math.pow( l, exponent );
			let k = fill[ a ] ++;
			AI[ 2 * k ] = b; AI[ 2 * k + 1 ] = e; AF[ 2 * k ] = l; AF[ 2 * k + 1 ] = w; slotA[ e ] = k;
			k = fill[ b ] ++;
			AI[ 2 * k ] = a; AI[ 2 * k + 1 ] = e; AF[ 2 * k ] = l; AF[ 2 * k + 1 ] = w; slotB[ e ] = k;

		}

		for ( let i = 0; i < N; i ++ ) adjEnd[ i ] = allActive ? adjStart[ i + 1 ] : adjStart[ i ];

	}

	// swaps two slots of the same node and keeps the slot index of both edges current
	function moveSlot( node, k, dst ) {

		if ( k === dst ) return;
		const k2 = 2 * k, d2 = 2 * dst;
		const nk = AI[ k2 ], ek = AI[ k2 + 1 ], lk = AF[ k2 ], wk = AF[ k2 + 1 ];
		const ed = AI[ d2 + 1 ];
		AI[ k2 ] = AI[ d2 ]; AI[ k2 + 1 ] = ed; AF[ k2 ] = AF[ d2 ]; AF[ k2 + 1 ] = AF[ d2 + 1 ];
		AI[ d2 ] = nk; AI[ d2 + 1 ] = ek; AF[ d2 ] = lk; AF[ d2 + 1 ] = wk;
		if ( edges[ 2 * ek ] === node ) slotA[ ek ] = dst; else slotB[ ek ] = dst;
		if ( edges[ 2 * ed ] === node ) slotA[ ed ] = k; else slotB[ ed ] = k;

	}

	/** Removes edge e from the search graph. */
	function deactivate( e ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		moveSlot( a, slotA[ e ], -- adjEnd[ a ] );
		moveSlot( b, slotB[ e ], -- adjEnd[ b ] );

	}

	/** Adds edge e to the search graph. */
	function activate( e ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		moveSlot( a, slotA[ e ], adjEnd[ a ] ++ );
		moveSlot( b, slotB[ e ], adjEnd[ b ] ++ );

	}

	// Labels. Node v has two: side 0 (the search from s) and side 1 (the search from t).
	//   LF[ 4v + 2*side + 0 ]  weight of the label's path      LF[ … + 1 ]  its Euclidean length
	//   LI[ 8v + 4*side + 0 ]  search id that set the label     LI[ … + 1 ]  search id that settled it
	//   LI[ 8v + 4*side + 2 ]  edge count                       LI[ … + 3 ]  slot of its last edge
	const LF = new Float64Array( 4 * Math.max( 1, N ) );
	const LI = new Int32Array( 8 * Math.max( 1, N ) );
	let searchId = 0;
	const nextId = () => {

		if ( searchId === 0x7fffffff ) {

			LI.fill( 0 );
			searchId = 0;

		}

		return ++ searchId;

	};

	// one heap per side; grown when a scan could overflow it
	let cap = Math.max( 64, 2 * N );
	let hk0 = new Float64Array( cap ), hn0 = new Uint32Array( cap );
	let hk1 = new Float64Array( cap ), hn1 = new Uint32Array( cap );
	function grow( need ) {

		while ( cap < need ) cap *= 2;
		const copyF = ( a ) => {

			const b = new Float64Array( cap );
			b.set( a );
			return b;

		};

		const copyU = ( a ) => {

			const b = new Uint32Array( cap );
			b.set( a );
			return b;

		};

		hk0 = copyF( hk0 ); hn0 = copyU( hn0 ); hk1 = copyF( hk1 ); hn1 = copyU( hn1 );

	}

	const out = {
		nodes: new Uint32Array( 16 ), // interior nodes, source to target
		edges: new Uint32Array( 17 ), // edge ids, source to target
		interior: 0,
		length: 0, // Euclidean
		weight: 0,
		relaxations: 0,
		searches: 0,
	};

	function reserve( interior ) {

		if ( interior + 1 > out.edges.length ) {

			const size = Math.max( interior + 1, 2 * out.edges.length );
			out.nodes = new Uint32Array( size );
			out.edges = new Uint32Array( size + 1 );

		}

	}

	// least weight of any path that covers a distance d (given d^2) with at most `hops` edges:
	// d^k / hops^(k-1) for k >= 1 (equal hops are cheapest), d^k for k < 1 (one hop is)
	const lowerBound = exponent === 2 ? ( d2, hops ) => d2 / hops
		: exponent >= 1 ? ( d2, hops ) => Math.pow( d2, 0.5 * exponent ) / Math.pow( hops, exponent - 1 )
			: ( d2 ) => Math.pow( d2, 0.5 * exponent );

	/**
	 * Minimum-weight path between s and t over the active edges, without edge `forbidden`:
	 * bidirectional Dijkstra.
	 *
	 * @param {number} bound Euclidean length limit B
	 * @param {number} wBound only paths lighter than this are wanted (Infinity: any)
	 * @param {number} maxHops edge count limit
	 * @param {boolean} prune drop labels that cannot be completed within B / maxHops (or, given
	 *   maxHops, not lighter than the best path so far) and accept only meetings within both
	 *   limits. With prune = false the lightest path is returned whatever its length and hop
	 *   count (the caller checks them).
	 * @param {number} maxPops node expansions after which the search settles for the best path
	 *   it has
	 * @returns {boolean} true when a path was found; it is then in `out` (else `out` is untouched)
	 */
	function search( s, t, bound, wBound, maxHops, prune, forbidden, maxPops ) {

		const id = nextId();
		out.searches ++;
		LI[ 8 * s ] = id; LI[ 8 * s + 1 ] = id; LI[ 8 * s + 2 ] = 0; LI[ 8 * s + 3 ] = - 1;
		LF[ 4 * s ] = 0; LF[ 4 * s + 1 ] = 0;
		LI[ 8 * t + 4 ] = id; LI[ 8 * t + 5 ] = id; LI[ 8 * t + 6 ] = 0; LI[ 8 * t + 7 ] = - 1;
		LF[ 4 * t + 2 ] = 0; LF[ 4 * t + 3 ] = 0;
		let size0 = 0, size1 = 0;
		let mu = wBound, meetSlot = - 1, meetSide = 0, meetNode = 0;
		let relax = 0, pops = 0;
		// both origins are settled up front: scan s, then t, then whatever the heaps yield
		let u = s, side = 0, opening = true;
		for ( ;; ) {

			const si = side << 2, oi = 4 - si; // int offsets of this side's and the other side's label
			const sf = side << 1, of = 2 - sf; // float offsets
			const o4 = 4 * ( side === 0 ? t : s ); // where this side is heading
			const ox = P[ o4 ], oy = P[ o4 + 1 ], oz = P[ o4 + 2 ];
			const gu = LF[ 4 * u + sf ], eu = LF[ 4 * u + sf + 1 ], hu = LI[ 8 * u + si + 2 ] + 1;
			const start = adjStart[ u ], end = adjEnd[ u ];
			relax += end - start;
			let size = side === 0 ? size0 : size1;
			if ( size + end - start > cap ) grow( size + end - start );
			const hk = side === 0 ? hk0 : hk1, hn = side === 0 ? hn0 : hn1;
			for ( let k = 2 * start, k2 = 2 * end; k < k2; k += 2 ) {

				if ( AI[ k + 1 ] === forbidden ) continue;
				const ng = gu + AF[ k + 1 ];
				if ( ! ( ng < mu ) ) continue;
				const v = AI[ k ];
				const ne = eu + AF[ k ];
				const b8 = 8 * v, b4 = 4 * v;
				if ( LI[ b8 + oi + 1 ] === id ) {

					// v is settled by the other search: a complete path
					const total = ng + LF[ b4 + of ];
					if ( total < mu && ( ! prune || ( ne + LF[ b4 + of + 1 ] <= bound && hu + LI[ b8 + oi + 2 ] <= maxHops ) ) ) {

						mu = total; meetSlot = k >> 1; meetSide = side; meetNode = u;
						continue; // nothing through v can beat this meeting

					}

				}

				const iv = b8 + si;
				if ( LI[ iv ] === id ) {

					if ( LI[ iv + 1 ] === id || ! ( ng < LF[ b4 + sf ] ) ) continue; // settled, or not lighter

				}

				if ( prune ) {

					// the rest of the path is at least d long and has at most maxHops - hu edges,
					// so it weighs at least d^2 / ( maxHops - hu ) (for exponent 2)
					if ( hu >= maxHops ) continue;
					const rem = bound - ne;
					const dx = P[ b4 ] - ox, dy = P[ b4 + 1 ] - oy, dz = P[ b4 + 2 ] - oz;
					const d2 = dx * dx + dy * dy + dz * dz;
					if ( ! ( d2 <= rem * rem ) ) continue;
					if ( ng + lowerBound( d2, maxHops - hu ) >= mu ) continue;

				}

				LI[ iv ] = id; LI[ iv + 2 ] = hu; LI[ iv + 3 ] = k >> 1;
				LF[ b4 + sf ] = ng; LF[ b4 + sf + 1 ] = ne;
				heapPush( hk, hn, size ++, ng, v );

			}

			if ( side === 0 ) size0 = size; else size1 = size;

			if ( opening ) {

				opening = false;
				u = t;
				side = 1;
				continue;

			}

			// entries of nodes that were settled through a lighter entry are dropped here
			while ( size0 > 0 && LI[ 8 * hn0[ 0 ] + 1 ] === id ) heapPop( hk0, hn0, -- size0 );
			while ( size1 > 0 && LI[ 8 * hn1[ 0 ] + 5 ] === id ) heapPop( hk1, hn1, -- size1 );
			if ( size0 === 0 || size1 === 0 || pops >= maxPops ) break;
			const k0 = hk0[ 0 ], k1 = hk1[ 0 ];
			if ( k0 + k1 >= mu ) break;
			pops ++;
			if ( k0 <= k1 ) {

				side = 0;
				u = hn0[ 0 ];
				heapPop( hk0, hn0, -- size0 );
				LI[ 8 * u + 1 ] = id;

			} else {

				side = 1;
				u = hn1[ 0 ];
				heapPop( hk1, hn1, -- size1 );
				LI[ 8 * u + 5 ] = id;

			}

		}

		out.relaxations += relax;
		if ( meetSlot < 0 ) return false;

		// path: s … x, then the meeting edge, then y … t
		const far = AI[ 2 * meetSlot ];
		const x = meetSide === 0 ? meetNode : far, y = meetSide === 0 ? far : meetNode;
		const hf = LI[ 8 * x + 2 ], hb = LI[ 8 * y + 6 ];
		const interior = hf + hb;
		reserve( interior );
		const nodes = out.nodes, pe = out.edges;
		let v = x;
		for ( let i = hf - 1; i >= 0; i -- ) {

			nodes[ i ] = v;
			const eid = AI[ 2 * LI[ 8 * v + 3 ] + 1 ];
			pe[ i ] = eid;
			v = edges[ 2 * eid ] === v ? edges[ 2 * eid + 1 ] : edges[ 2 * eid ];

		}

		pe[ hf ] = AI[ 2 * meetSlot + 1 ];
		v = y;
		for ( let i = hf; i < interior; i ++ ) {

			nodes[ i ] = v;
			const eid = AI[ 2 * LI[ 8 * v + 7 ] + 1 ];
			pe[ i + 1 ] = eid;
			v = edges[ 2 * eid ] === v ? edges[ 2 * eid + 1 ] : edges[ 2 * eid ];

		}

		out.interior = interior;
		out.length = LF[ 4 * x + 1 ] + AF[ 2 * meetSlot ] + LF[ 4 * y + 3 ];
		out.weight = mu;
		return true;

	}

	/**
	 * A short Euclidean path between s and t over the active edges, without edge `forbidden`:
	 * A* with the straight distance to t as potential (consistent for Euclidean weights),
	 * weighted by `greed`. greed = 1 gives the shortest path; greed > 1 heads for the target
	 * more eagerly and returns a path at most `greed` times longer. Labels with length +
	 * distance to t > bound are dropped, which loses nothing, and a node is expanded again
	 * whenever a shorter label reaches it, so "no path within the bound" is an exact answer
	 * for any greed as long as maxHops and maxPops do not cut in. A label that has used up
	 * maxHops is dropped too; then the answer is a heuristic (a node keeps its shortest label).
	 *
	 * @param {number} maxPops node expansions after which the search gives up (returns false)
	 * @returns {boolean} true when a path of length <= bound was found; it is then in `out`
	 */
	function shortest( s, t, bound, maxHops, forbidden, greed, maxPops ) {

		const id = nextId();
		out.searches ++;
		const tx = P[ 4 * t ], ty = P[ 4 * t + 1 ], tz = P[ 4 * t + 2 ];
		LI[ 8 * s ] = id; LI[ 8 * s + 2 ] = 0; LI[ 8 * s + 3 ] = - 1;
		LF[ 4 * s ] = - 1; LF[ 4 * s + 1 ] = 0;
		let size = 0, relax = 0, pops = 0, u = s;
		for ( ;; ) {

			const gu = LF[ 4 * u + 1 ], hu = LI[ 8 * u + 2 ] + 1;
			const start = adjStart[ u ], end = adjEnd[ u ];
			relax += end - start;
			if ( size + end - start > cap ) grow( size + end - start );
			const hk = hk0, hn = hn0;
			for ( let k = 2 * start, k2 = 2 * end; k < k2; k += 2 ) {

				if ( AI[ k + 1 ] === forbidden ) continue;
				const v = AI[ k ];
				const ng = gu + AF[ k ];
				const b8 = 8 * v, b4 = 4 * v;
				if ( LI[ b8 ] === id && ! ( ng < LF[ b4 + 1 ] ) ) continue; // not shorter
				const rem = bound - ng;
				const dx = P[ b4 ] - tx, dy = P[ b4 + 1 ] - ty, dz = P[ b4 + 2 ] - tz;
				const d2 = dx * dx + dy * dy + dz * dz;
				if ( ! ( rem >= 0 && d2 <= rem * rem ) || ( hu >= maxHops && v !== t ) ) continue;
				const f = ng + greed * Math.sqrt( d2 );
				LI[ b8 ] = id; LI[ b8 + 2 ] = hu; LI[ b8 + 3 ] = k >> 1;
				LF[ b4 ] = f; LF[ b4 + 1 ] = ng;
				heapPush( hk, hn, size ++, f, v );

			}

			// next node: the smallest key that is still its node's current label
			let key;
			do {

				if ( size === 0 || pops >= maxPops ) {

					out.relaxations += relax;
					return false;

				}

				u = hn0[ 0 ];
				key = hk0[ 0 ];
				heapPop( hk0, hn0, -- size );

			} while ( key !== LF[ 4 * u ] );

			if ( u === t ) break;
			LF[ 4 * u ] = - 1; // expanded; a shorter label re-opens the node
			pops ++;

		}

		out.relaxations += relax;
		// Walk the last edges back to s. A node on the way may have been re-opened (given a
		// shorter label with another hop count) after t's label was derived from it, so the
		// hop count is taken from the chain itself.
		let hops = 0;
		for ( let v = t; v !== s; hops ++ ) {

			const eid = AI[ 2 * LI[ 8 * v + 3 ] + 1 ];
			v = edges[ 2 * eid ] === v ? edges[ 2 * eid + 1 ] : edges[ 2 * eid ];

		}

		if ( hops > maxHops ) return false;
		const interior = hops - 1;
		reserve( interior );
		const nodes = out.nodes, pe = out.edges;
		let v = t, w = 0, l = 0;
		for ( let i = interior; i >= 0; i -- ) {

			const k = 2 * LI[ 8 * v + 3 ], eid = AI[ k + 1 ];
			pe[ i ] = eid;
			l += AF[ k ];
			w += AF[ k + 1 ];
			v = edges[ 2 * eid ] === v ? edges[ 2 * eid + 1 ] : edges[ 2 * eid ];
			if ( i > 0 ) nodes[ i - 1 ] = v;

		}

		out.interior = interior;
		out.length = l;
		out.weight = w;
		return true;

	}

	return { len, usable, usableCount, activate, deactivate, search, shortest, out };

}

function resolveOptions( options ) {

	const o = { ...BUNDLING_DEFAULTS };
	for ( const key in options ) if ( options[ key ] !== undefined ) o[ key ] = options[ key ];
	if ( o.method !== 'epb' && o.method !== 'spanner' ) throw new Error( `edgePathBundling: unknown method '${o.method}'` );
	o.exponent = + o.exponent;
	o.maxDistortion = + o.maxDistortion;
	if ( ! ( o.exponent > 0 ) ) throw new Error( 'edgePathBundling: exponent must be > 0' );
	if ( ! ( o.maxDistortion >= 1 ) ) throw new Error( 'edgePathBundling: maxDistortion must be >= 1' );
	o.maxInterior = o.maxInterior === null || ! ( o.maxInterior < NO_LIMIT ) ? NO_LIMIT - 1 : Math.max( 1, Math.floor( o.maxInterior ) );
	o.stride = Math.max( 3, o.stride | 0 );
	o.exact = !! o.exact;
	o.effort = o.effort === null || ! ( o.effort < Infinity ) ? Infinity : Math.max( 1, Math.floor( o.effort ) );
	o.greed = Math.max( 1, + o.greed || 1 );
	return o;

}

/**
 * Edge-Path Bundling as a resumable task (see the header for the algorithm and the options).
 *
 * The task has the shape the analytics.js runners expect:
 *   total      work units: one per usable edge and phase ('epb' has one phase, 'spanner' two)
 *   completed  units done
 *   done       true when finished, or when options.maxWork is used up
 *   complete   true when every phase ran to its end
 *   phase      'bundle' ('epb'), or 'spanner' then 'route'
 *   step( count = 1 )   processes `count` units, returns done
 *   result()   may be called at ANY time: the paths found so far (edges not reached yet are
 *              straight). A new object each call; the arrays own their buffers (transferable).
 *              O( E + path nodes ), so call it per progress report, not per step.
 *   hurry()    see below
 * The result depends only on the input and the options, never on how the work was chunked.
 * options.timeBudget is not read here: the caller decides when to stop calling step().
 *
 * @param {object} graph anything with N, E and edges (Uint32Array of 2E node indices)
 * @param {ArrayLike<number>} positions x, y, z of node i at positions[ i * stride + 0..2 ]; read
 *   once, when the task is created
 * @param {object} [options] see BUNDLING_DEFAULTS
 */
export function edgePathBundlingTask( graph, positions, options = {} ) {

	const o = resolveOptions( options );
	const { N, E, edges } = graph;
	const tSetup = now();
	const spanner = o.method === 'spanner';
	const router = createRouter( N, E, edges, positions, o.stride, o.exponent, ! spanner );
	const { len, usable, search, shortest, out } = router;
	const maxHops = o.maxInterior + 1;
	const prune = ! o.exact;
	const effort = o.effort, greed = o.greed;
	const K = o.maxDistortion, expo = o.exponent;
	// a path of Euclidean length <= B weighs at most B^k when k >= 1 (the 1e-9 absorbs rounding)
	const weightOf = ( l ) => ( expo === 2 ? l * l : expo === 1 ? l : Math.pow( l, expo ) );
	const weightBound = ( B ) => ( expo >= 1 ? weightOf( B ) * ( 1 + 1e-9 ) : Infinity );

	// usable edges with a positive length, shortest first
	let order;
	{

		let n = 0;
		const ids = new Uint32Array( router.usableCount );
		for ( let e = 0; e < E; e ++ ) if ( usable[ e ] ) ids[ n ++ ] = e;
		order = orderByLength( len, ids );

	}

	const M = order.length;

	// paths: for an edge with c interior nodes, pool holds c nodes, then its c + 1 edge ids
	const pathAt = new Int32Array( Math.max( 1, E ) ).fill( - 1 );
	const pathCount = new Uint32Array( Math.max( 1, E ) );
	let pool = new Uint32Array( 1024 ), poolSize = 0;
	const locked = new Uint8Array( Math.max( 1, E ) ); // 'epb': on a bundled path
	// 'spanner': 0 = in H or not reached, 1 = provisional path from phase 'spanner', 2 = routed,
	// 3 = left straight by the exact rule
	const routed = new Uint8Array( Math.max( 1, E ) );
	const shown = ( e ) => ( ! spanner ? true : prune ? routed[ e ] !== 0 : routed[ e ] === 2 );
	let spannerEdges = 0;

	function storePath( e ) {

		const c = out.interior, need = 2 * c + 1;
		let at = pathAt[ e ];
		if ( at < 0 || c > pathCount[ e ] ) {

			// a new slot (a replaced path keeps its slot unless it grows)
			if ( poolSize + need > pool.length ) {

				const grown = new Uint32Array( Math.max( 2 * pool.length, poolSize + need ) );
				grown.set( pool.subarray( 0, poolSize ) );
				pool = grown;

			}

			at = poolSize;
			poolSize += need;

		}

		pool.set( out.nodes.subarray( 0, c ), at );
		pool.set( out.edges.subarray( 0, c + 1 ), at + c );
		pathAt[ e ] = at;
		pathCount[ e ] = c;

	}

	let cursor = 0; // units done within the phase
	let hurried = false, windowWork = 0;
	let timeMs = now() - tSetup;

	const task = {
		total: spanner ? 2 * M : M,
		completed: 0,
		done: M === 0,
		phase: spanner ? 'spanner' : 'bundle',
		complete: M === 0,

		step( count = 1 ) {

			if ( task.done ) return true;
			const t0 = now();
			let left = count;
			while ( left > 0 && ! task.done ) {

				if ( out.relaxations >= o.maxWork ) {

					task.done = true;
					break;

				}

				// every 256 units of phase 'spanner': at the recent cost per unit, will it finish?
				if ( o.maxWork < Infinity && task.phase === 'spanner' && ! hurried && ( cursor & 255 ) === 0 ) {

					if ( ( M - cursor ) * ( out.relaxations - windowWork ) / 256 > o.maxWork - out.relaxations && cursor > 0 ) task.hurry();
					windowWork = out.relaxations;

				}

				left --;
				task.completed ++;
				if ( task.phase === 'bundle' ) {

					// the paper's loop: longest edge first
					const e = order[ M - 1 - cursor ];
					if ( ++ cursor === M ) task.done = task.complete = true;
					const l = len[ e ];
					if ( locked[ e ] || ! ( l > 0 ) ) continue;
					const s = edges[ 2 * e ], t = edges[ 2 * e + 1 ], B = K * l;
					if ( prune ) {

						// a short path without e first: without one the edge stays straight; with
						// one, only lighter paths are of interest, which bounds the second search
						if ( ! shortest( s, t, B, maxHops, e, greed, effort ) || out.interior < 1 ) continue;
						search( s, t, B, out.weight * ( 1 + 1e-9 ), maxHops, true, e, effort );
						if ( out.interior < 1 ) continue; // a parallel edge is not a bundle

					} else {

						// the lightest path whatever its length, then the paper's test
						if ( ! search( s, t, B, weightBound( B ), NO_LIMIT, false, e, Infinity ) ) continue;
						if ( out.interior < 1 || out.length > B || out.interior + 1 > maxHops ) continue;

					}

					storePath( e );
					router.deactivate( e ); // stays skipped
					for ( let i = 0, pe = out.edges; i <= out.interior; i ++ ) locked[ pe[ i ] ] = 1;

				} else if ( task.phase === 'spanner' ) {

					// greedy t-spanner: shortest edge first
					const e = order[ cursor ];
					if ( ++ cursor === M ) {

						cursor = 0;
						task.phase = 'route';

					}

					const l = len[ e ];
					// a search cut short by the hop limit or the effort limit answers "no path",
					// which can only add edges to H
					if ( l > 0 && shortest( edges[ 2 * e ], edges[ 2 * e + 1 ], K * l, prune ? maxHops : NO_LIMIT, - 1, prune ? ( hurried ? HURRY_GREED : greed ) : 1, prune ? ( hurried ? Math.min( effort, HURRY_EFFORT ) : effort ) : Infinity ) && out.interior >= 1 ) {

						routed[ e ] = 1;
						storePath( e ); // provisional: phase 'route' looks for a lighter one

					} else {

						router.activate( e );
						spannerEdges ++;

					}

				} else {

					// route the edges outside the spanner over it, longest first
					const e = order[ M - 1 - cursor ];
					if ( ++ cursor === M ) task.done = task.complete = true;
					if ( routed[ e ] !== 1 ) continue;
					const at = pathAt[ e ], c = pathCount[ e ];
					let w0 = 0;
					for ( let i = at + c; i <= at + 2 * c; i ++ ) w0 += weightOf( len[ pool[ i ] ] );
					const B = K * len[ e ];
					// only paths lighter than the provisional one are of interest
					routed[ e ] = 2;
					if ( search( edges[ 2 * e ], edges[ 2 * e + 1 ], B, w0 * ( 1 + 1e-9 ), maxHops, prune, - 1, prune ? effort : Infinity ) ) {

						if ( out.interior < 1 ) continue; // a parallel edge is not a bundle
						if ( prune || ( out.length <= B && out.interior + 1 <= maxHops ) ) storePath( e );
						else routed[ e ] = 3; // exact: the lightest path is too long, the edge stays straight

					} else if ( ! prune && c + 1 > maxHops ) routed[ e ] = 3;

				}

			}

			timeMs += now() - t0;
			return task.done;

		},

		/**
		 * Phase 'spanner' only, and not in exact mode: finish the phase with quick, eager tests.
		 * An unfinished spanner leaves the LONGEST edges straight, which is the worst place to
		 * stop; after hurry() the rest of the phase costs little, at the price of more (long)
		 * edges that end up in the spanner. Called by edgePathBundling() when the phase would
		 * not finish within the time budget at its recent pace, and by step() likewise for
		 * options.maxWork.
		 */
		hurry() {

			if ( prune ) hurried = true;

		},

		result() {

			const pathStart = new Uint32Array( E + 1 );
			let maxInterior = 0, bundled = 0, interiorSum = 0;
			for ( let e = 0; e < E; e ++ ) {

				const c = shown( e ) ? pathCount[ e ] : 0;
				pathStart[ e + 1 ] = pathStart[ e ] + c;
				if ( c > 0 ) {

					bundled ++;
					interiorSum += c;
					if ( c > maxInterior ) maxInterior = c;

				}

			}

			const pathRefs = new Uint32Array( pathStart[ E ] );
			const state = new Uint8Array( E ), load = new Uint32Array( E );
			let total = 0, straight = 0, distSum = 0, distMax = 0;
			for ( let e = 0; e < E; e ++ ) {

				const l = usable[ e ] ? len[ e ] : 0;
				total += l;
				const c = pathStart[ e + 1 ] - pathStart[ e ];
				if ( c === 0 ) {

					straight += l;
					continue;

				}

				const at = pathAt[ e ];
				pathRefs.set( pool.subarray( at, at + c ), pathStart[ e ] );
				state[ e ] = EDGE_BUNDLED;
				let pl = 0;
				for ( let i = at + c; i <= at + 2 * c; i ++ ) {

					load[ pool[ i ] ] ++;
					pl += len[ pool[ i ] ];

				}

				const d = pl / l;
				distSum += d;
				if ( d > distMax ) distMax = d;

			}

			let carrying = 0;
			for ( let e = 0; e < E; e ++ ) {

				if ( state[ e ] === EDGE_FREE && load[ e ] > 0 ) {

					state[ e ] = EDGE_CARRYING;
					carrying ++;

				}

			}

			return {
				pathStart, pathRefs, maxInterior, state, load,
				stats: {
					method: o.method,
					edges: E,
					bundled,
					carrying,
					free: E - bundled - carrying,
					bundledRatio: E ? bundled / E : 0,
					spannerEdges: spanner ? spannerEdges : null,
					meanDistortion: bundled ? distSum / bundled : 1,
					maxDistortion: bundled ? distMax : 1,
					meanInterior: bundled ? interiorSum / bundled : 0,
					maxInterior,
					inkRatio: total > 0 ? straight / total : 1,
					searches: out.searches,
					relaxations: out.relaxations,
					timeMs,
					complete: task.complete,
					hurried,
					phase: task.phase,
				},
			};

		},
	};

	return task;

}

/**
 * Edge-Path Bundling, run on the calling thread until it is done or options.timeBudget (ms,
 * setup included) is used up. See the header for the algorithm, the options and the result.
 *
 * @param {object} graph anything with N, E and edges
 * @param {ArrayLike<number>} positions x, y, z per node (see options.stride)
 * @param {object} [options]
 */
export function edgePathBundling( graph, positions, options = {} ) {

	const t0 = now();
	const task = edgePathBundlingTask( graph, positions, options );
	const budget = options.timeBudget === undefined || options.timeBudget === null ? Infinity : + options.timeBudget;
	if ( ! ( budget < Infinity ) ) task.step( Infinity );
	else {

		// pace = ms per unit, smoothed over the last few hundred units
		let pace = 0, last = now(), units = 0;
		while ( ! task.done ) {

			const used = last - t0;
			if ( used >= budget ) break;
			if ( task.phase === 'spanner' && ( task.total / 2 - task.completed ) * pace > budget - used ) task.hurry();
			task.step( 16 );
			const t = now();
			const p = ( t - last ) / Math.max( 1, task.completed - units );
			pace += ( p - pace ) * ( pace > 0 ? 0.05 : 1 );
			last = t;
			units = task.completed;

		}

	}

	const r = task.result();
	r.stats.timeMs = now() - t0;
	return r;

}

// ─── Checks and statistics ───────────────────────────────────────────────────────────────────

/**
 * Mass-weighted cluster centroids, level by level: the CPU mirror of the GPU centroid pass
 * (GraphBuffers.computeClusterCentroidsCPU without the radii).
 *
 * @param {object} graph PreparedGraph
 * @param {ArrayLike<number>} positions vec4 per node
 * @returns {Float32Array} vec4 per cluster (w = 0)
 */
export function clusterCentroids( graph, positions ) {

	const g = graph;
	const cp = new Float32Array( 4 * Math.max( 1, g.C ) );
	for ( const { height, start, count } of g.heightRanges ) {

		const nodeChildren = height === 1 || g.D === 0;
		for ( let c = start; c < start + count; c ++ ) {

			let x = 0, y = 0, z = 0, m = 0;
			const s = g.clusterChildStart[ c ], n = g.clusterChildCount[ c ];
			for ( let k = 0; k < n; k ++ ) {

				const ch = g.childList[ s + k ];
				if ( nodeChildren ) {

					x += positions[ 4 * ch ]; y += positions[ 4 * ch + 1 ]; z += positions[ 4 * ch + 2 ]; m += 1;

				} else {

					const w = g.clusterMass[ ch ];
					x += cp[ 4 * ch ] * w; y += cp[ 4 * ch + 1 ] * w; z += cp[ 4 * ch + 2 ] * w; m += w;

				}

			}

			m = Math.max( m, 1e-9 );
			cp[ 4 * c ] = x / m; cp[ 4 * c + 1 ] = y / m; cp[ 4 * c + 2 ] = z / m;

		}

	}

	return cp;

}

// sorted neighbour lists, for adjacency tests
function sortedAdjacency( N, E, edges ) {

	const start = new Uint32Array( N + 1 );
	for ( let k = 0; k < 2 * E; k ++ ) start[ edges[ k ] + 1 ] ++;
	for ( let i = 0; i < N; i ++ ) start[ i + 1 ] += start[ i ];
	const list = new Uint32Array( Math.max( 1, 2 * E ) );
	const fill = start.slice( 0, N );
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		list[ fill[ a ] ++ ] = b;
		list[ fill[ b ] ++ ] = a;

	}

	for ( let i = 0; i < N; i ++ ) list.subarray( start[ i ], start[ i + 1 ] ).sort();
	return ( a, b ) => {

		let lo = start[ a ], hi = start[ a + 1 ] - 1;
		while ( lo <= hi ) {

			const mid = ( lo + hi ) >> 1, v = list[ mid ];
			if ( v === b ) return true;
			if ( v < b ) lo = mid + 1; else hi = mid - 1;

		}

		return false;

	};

}

/**
 * Checks a path set against its graph.
 *
 * Always: pathStart is a Uint32Array of E + 1 non-decreasing offsets starting at 0, pathRefs is a
 * Uint32Array that holds at least pathStart[E] entries, every ref is below N + C (C = graph.C,
 * 0 when the graph has no clusters), and maxInterior is the largest interior count.
 *
 * options.realPaths = true (edge-path bundling): every ref is a node, and the polygon
 * [ s, ...interior, t ] of every edge with interior points is a simple path of the graph
 * (consecutive nodes are adjacent, no node twice; so it never uses the edge itself).
 * options.state (with realPaths): no edge of a bundled path is itself bundled, and state
 * matches the paths (EDGE_BUNDLED exactly for the edges with interior points).
 * options.positions + options.maxDistortion: the polygon of every edge with interior points is
 * at most maxDistortion times as long as the straight edge (relative tolerance 1e-9).
 * options.maxInterior: no edge has more interior points.
 *
 * @param {object} graph N, E, edges, [C]
 * @param {object} paths { pathStart, pathRefs, maxInterior }
 * @param {object} [options]
 * @returns {{ valid: boolean, errors: string[] }} at most 20 messages
 */
export function validatePaths( graph, paths, options = {} ) {

	const { N, E, edges } = graph;
	const errors = [];
	const fail = ( msg ) => {

		if ( errors.length < 20 ) errors.push( msg );

	};

	const done = () => ( { valid: errors.length === 0, errors } );
	if ( ! paths || ! ( paths.pathStart instanceof Uint32Array ) || ! ( paths.pathRefs instanceof Uint32Array ) ) {

		fail( 'pathStart and pathRefs must be Uint32Arrays' );
		return done();

	}

	const { pathStart, pathRefs } = paths;
	if ( pathStart.length !== E + 1 ) {

		fail( `pathStart has ${pathStart.length} entries, expected E + 1 = ${E + 1}` );
		return done();

	}

	if ( pathStart[ 0 ] !== 0 ) fail( 'pathStart[0] must be 0' );
	if ( pathStart[ E ] > pathRefs.length ) fail( `pathStart[E] = ${pathStart[ E ]} exceeds pathRefs.length = ${pathRefs.length}` );
	let maxInterior = 0;
	for ( let e = 0; e < E; e ++ ) {

		if ( pathStart[ e + 1 ] < pathStart[ e ] ) {

			fail( `pathStart decreases at edge ${e}` );
			return done();

		}

		maxInterior = Math.max( maxInterior, pathStart[ e + 1 ] - pathStart[ e ] );

	}

	if ( errors.length ) return done();
	if ( paths.maxInterior !== maxInterior ) fail( `maxInterior is ${paths.maxInterior}, the paths say ${maxInterior}` );
	if ( options.maxInterior !== undefined && maxInterior > options.maxInterior ) fail( `an edge has ${maxInterior} interior points, the limit is ${options.maxInterior}` );
	const refLimit = options.realPaths ? N : N + ( graph.C || 0 );
	for ( let k = 0, n = pathStart[ E ]; k < n; k ++ ) {

		if ( pathRefs[ k ] >= refLimit ) {

			fail( `ref ${pathRefs[ k ]} at ${k} is out of range (limit ${refLimit})` );
			return done();

		}

	}

	if ( options.realPaths ) {

		const adjacent = sortedAdjacency( N, E, edges );
		const seen = new Int32Array( Math.max( 1, N ) ).fill( - 1 );
		for ( let e = 0; e < E; e ++ ) {

			const from = pathStart[ e ], to = pathStart[ e + 1 ];
			if ( to === from ) continue;
			const s = edges[ 2 * e ], t = edges[ 2 * e + 1 ];
			seen[ s ] = e; seen[ t ] = e;
			let prev = s;
			for ( let k = from; k <= to; k ++ ) {

				const v = k < to ? pathRefs[ k ] : t;
				if ( k < to ) {

					if ( seen[ v ] === e ) fail( `edge ${e}: node ${v} appears twice on its path` );
					seen[ v ] = e;

				}

				if ( ! adjacent( prev, v ) ) fail( `edge ${e}: ${prev} and ${v} are consecutive on its path but not adjacent` );
				prev = v;

			}

		}

		if ( options.state ) {

			const state = options.state;
			const edgeOf = edgeLookup( N, E, edges );
			for ( let e = 0; e < E; e ++ ) {

				const from = pathStart[ e ], to = pathStart[ e + 1 ];
				if ( ( state[ e ] === EDGE_BUNDLED ) !== ( to > from ) ) fail( `edge ${e}: state ${state[ e ]} does not match its ${to - from} interior points` );
				if ( to === from ) continue;
				let prev = edges[ 2 * e ];
				for ( let k = from; k <= to; k ++ ) {

					const v = k < to ? pathRefs[ k ] : edges[ 2 * e + 1 ];
					const m = edgeOf( prev, v );
					if ( m >= 0 && state[ m ] !== EDGE_CARRYING ) fail( `edge ${e}: its path uses edge ${m}, whose state is ${state[ m ]} (expected ${EDGE_CARRYING})` );
					prev = v;

				}

			}

		}

	}

	if ( options.positions && options.maxDistortion !== undefined ) {

		const pos = options.positions, stride = options.stride || 4;
		const cp = pathStart[ E ] > 0 && ! options.realPaths && graph.C ? ( options.clusterPositions || clusterCentroids( graph, pos ) ) : null;
		const point = refPoint( N, pos, stride, cp );
		for ( let e = 0; e < E; e ++ ) {

			const from = pathStart[ e ], to = pathStart[ e + 1 ];
			if ( to === from ) continue;
			const s = edges[ 2 * e ], t = edges[ 2 * e + 1 ];
			let l = 0, prev = s;
			for ( let k = from; k <= to; k ++ ) {

				const v = k < to ? pathRefs[ k ] : t;
				l += point.distance( prev, v );
				prev = v;

			}

			const straight = point.distance( s, t );
			if ( l > options.maxDistortion * straight * ( 1 + 1e-9 ) ) fail( `edge ${e}: path length ${l} exceeds ${options.maxDistortion} x ${straight}` );

		}

	}

	return done();

}

// edge id of a node pair (-1 when not adjacent); for parallel edges, the lowest id
function edgeLookup( N, E, edges ) {

	const start = new Uint32Array( N + 1 );
	for ( let k = 0; k < 2 * E; k ++ ) start[ edges[ k ] + 1 ] ++;
	for ( let i = 0; i < N; i ++ ) start[ i + 1 ] += start[ i ];
	const node = new Uint32Array( Math.max( 1, 2 * E ) ), id = new Uint32Array( Math.max( 1, 2 * E ) );
	const fill = start.slice( 0, N );
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		node[ fill[ a ] ] = b; id[ fill[ a ] ++ ] = e;
		node[ fill[ b ] ] = a; id[ fill[ b ] ++ ] = e;

	}

	return ( a, b ) => {

		for ( let k = start[ a ]; k < start[ a + 1 ]; k ++ ) if ( node[ k ] === b ) return id[ k ];
		return - 1;

	};

}

// position of a ref: node positions below N, cluster centroids (vec4) from N on
function refPoint( N, pos, stride, clusterPos ) {

	const x = ( r ) => ( r < N ? pos[ r * stride ] : clusterPos[ 4 * ( r - N ) ] );
	const y = ( r ) => ( r < N ? pos[ r * stride + 1 ] : clusterPos[ 4 * ( r - N ) + 1 ] );
	const z = ( r ) => ( r < N ? pos[ r * stride + 2 ] : clusterPos[ 4 * ( r - N ) + 2 ] );
	return {
		distance( a, b ) {

			const dx = x( a ) - x( b ), dy = y( a ) - y( b ), dz = z( a ) - z( b );
			return Math.sqrt( dx * dx + dy * dy + dz * dz );

		},
	};

}

/**
 * Statistics of a path set in a given layout. All lengths are those of the control polygons
 * (bundling strength 1, before the B-spline smooths them), so they are upper bounds for what
 * is drawn.
 *
 * @param {object} graph N, E, edges; the cluster tables of a PreparedGraph when refs >= N occur
 * @param {object} paths { pathStart, pathRefs }
 * @param {ArrayLike<number>} positions x, y, z per node
 * @param {object} [options]
 * @param {number} [options.stride=4]
 * @param {ArrayLike<number>} [options.clusterPositions] vec4 per cluster; default
 *   clusterCentroids( graph, positions )
 * @returns {{ edges: number, bundled: number, bundledRatio: number, refs: number,
 *   maxInterior: number, meanInterior: number, straightLength: number, pathLength: number,
 *   lengthRatio: number, meanDistortion: number, maxDistortion: number, inkLength: number,
 *   inkRatio: number }}
 *   bundled: edges with interior points. meanInterior, meanDistortion, maxDistortion: over the
 *   bundled edges, distortion = polygon length / straight length. lengthRatio = pathLength /
 *   straightLength over all edges. inkLength: total length of the DISTINCT polygon segments
 *   (a segment shared by many edges counts once); inkRatio = inkLength / straightLength, the
 *   polygon version of the ink ratio of the bundling literature (lower = more bundling).
 */
export function pathStats( graph, paths, positions, options = {} ) {

	const { N, E, edges } = graph;
	const { pathStart, pathRefs } = paths;
	const stride = options.stride || 4;
	let needClusters = false;
	for ( let k = 0, n = pathStart[ E ]; k < n && ! needClusters; k ++ ) needClusters = pathRefs[ k ] >= N;
	const cp = needClusters ? ( options.clusterPositions || clusterCentroids( graph, positions ) ) : null;
	const point = refPoint( N, positions, stride, cp );

	// distinct segments: open addressing on ( min ref, max ref )
	let cap = 16;
	while ( cap < 2 * ( E + pathStart[ E ] ) ) cap *= 2;
	const keyA = new Uint32Array( cap ), keyB = new Uint32Array( cap ); // keyA holds min + 1 (0 = empty)
	const mask = cap - 1;
	const isNew = ( a, b ) => {

		const lo = a < b ? a : b, hi = a < b ? b : a;
		let h = ( Math.imul( lo, 0x9e3779b1 ) ^ Math.imul( hi, 0x85ebca6b ) ) >>> 0;
		h = ( h ^ ( h >>> 15 ) ) & mask;
		for ( ;; ) {

			if ( keyA[ h ] === 0 ) {

				keyA[ h ] = lo + 1;
				keyB[ h ] = hi;
				return true;

			}

			if ( keyA[ h ] === lo + 1 && keyB[ h ] === hi ) return false;
			h = ( h + 1 ) & mask;

		}

	};

	let bundled = 0, maxInterior = 0, interiorSum = 0;
	let straightLength = 0, pathLength = 0, inkLength = 0, distSum = 0, distMax = 0;
	for ( let e = 0; e < E; e ++ ) {

		const s = edges[ 2 * e ], t = edges[ 2 * e + 1 ];
		const from = pathStart[ e ], to = pathStart[ e + 1 ];
		const straight = point.distance( s, t );
		if ( ! ( straight < Infinity ) ) continue; // NaN / infinite positions
		straightLength += straight;
		let l = 0, prev = s;
		for ( let k = from; k <= to; k ++ ) {

			const v = k < to ? pathRefs[ k ] : t;
			const d = point.distance( prev, v );
			l += d;
			if ( prev !== v && isNew( prev, v ) ) inkLength += d;
			prev = v;

		}

		pathLength += l;
		if ( to > from ) {

			bundled ++;
			interiorSum += to - from;
			if ( to - from > maxInterior ) maxInterior = to - from;
			if ( straight > 0 ) {

				const d = l / straight;
				distSum += d;
				if ( d > distMax ) distMax = d;

			}

		}

	}

	return {
		edges: E,
		bundled,
		bundledRatio: E ? bundled / E : 0,
		refs: pathStart[ E ],
		maxInterior,
		meanInterior: bundled ? interiorSum / bundled : 0,
		straightLength,
		pathLength,
		lengthRatio: straightLength > 0 ? pathLength / straightLength : 1,
		meanDistortion: bundled ? distSum / bundled : 1,
		maxDistortion: bundled ? distMax : 1,
		inkLength,
		inkRatio: straightLength > 0 ? inkLength / straightLength : 1,
	};

}
