// Graph analytics on the PreparedGraph CSR arrays (see prepare.js): traversal, shortest paths,
// components, cores, centralities, clustering, distance statistics, and helpers that map a
// metric to [0,1] for a visual channel (node size / colour / glow, edge width). Pure JS, typed
// arrays in and out, no three.js: runs in a Web Worker, on the main thread, or in Node (tests).
//
// ─── Graph input ────────────────────────────────────────────────────────────────────────────
//  Every function takes `g`: any object with the PreparedGraph CSR fields
//    N, E       node / edge counts
//    edges      Uint32Array( 2E )  undirected pairs, deduplicated, no self loops
//    adjStart   Uint32Array( N )   first adjacency slot of each node
//    degree     Uint32Array( N )
//    adjList    Uint32Array( 2E )  node v owns slots [ adjStart[v], adjStart[v] + degree[v] )
//  Nothing in `g` is ever modified. Edge ids in results are indices into `edges` (0..E-1). They
//  come from an edge index (adjEdge, parallel to adjList): pass options.adjEdge, or set g.adjEdge,
//  or let the module build it once and cache it (WeakMap keyed on g.adjList, about 8E bytes).
//  Results are freshly allocated typed arrays that own their buffers (transferable from a
//  worker), except where a function says it returns a view.
//
// ─── API ────────────────────────────────────────────────────────────────────────────────────
//  graphFromEdges( N, edges )                 CSR + adjEdge from a clean edge list
//  buildEdgeIndex( g )                        adjEdge: edge id of every adjacency slot
//  bfs( g, sources, { maxDepth, out } )       dist / parent / order / levelOffsets (ripple)
//  khopNeighborhood( g, sources, k )          nodes within k hops, with hop counts
//  shortestPath( g, s, t )                    one fewest-hops path (bidirectional BFS)
//  shortestPathWeighted( g, s, t, positions ) shortest Euclidean path (A*, or Dijkstra)
//  shortestPathDag( g, s, t )                 all geodesics: nodes, edges and path counts
//  connectedComponents( g )                   labels sorted by size (0 = largest)
//  kCore( g )                                 core numbers, degeneracy, peeling order
//  pageRank( g, options )                     also personalised (random walk with restart)
//  eigenvectorCentrality( g, options )        per component by default, safe on bipartite graphs
//  betweenness( g, options )                  Brandes: exact, or pivot sampling
//  edgeBetweenness( g, options )
//  betweennessTask( g, options )              the same as a resumable task (see Tasks)
//  closeness( g, options )                    closeness + harmonic + distance distribution
//  closenessTask( g, options )
//  averagePathLength( g, options )            mean distance, hop plot, effective diameter
//  estimateDiameter( g, { maxBfs } )          eccentricity bounds; exact when the bounds meet
//  clusteringCoefficient( g, { edges } )      local / global / average, triangles per node (and edge)
//  density( g ), degreeHistogram( g ), degreeAssortativity( g ), graphSummary( g )
//  normalize( values, { mode } )              linear | log | rank | robust  ->  Float32Array in [0,1]
//  quantiles( values, qs )
//  NODE_METRICS, nodeMetric( g, id, options ) metric catalogue for the UI and one dispatcher
//  runTask( task, options ), runTaskAsync( task, options ), defaultSamples( g )
//
// ─── Tasks (progress and cancellation) ──────────────────────────────────────────────────────
//  The two expensive computations (betweenness, closeness) are sums over BFS pivots. A task is
//    { total, completed, done, exact, step( count = 1 ) -> done, result() }
//  step() processes `count` pivots. result() may be called at ANY time: it returns the estimate
//  from the pivots processed so far, correctly scaled, so a view can refine progressively. The
//  result depends only on the options (seed included), never on how the work was chunked.
//
// ─── Sampling ───────────────────────────────────────────────────────────────────────────────
//  Pivots are drawn without replacement, stratified by connected component: a component of two
//  or more nodes gets round( samples * size / N ) pivots, but at least min( 32, samples ) and at
//  most its size. A component whose quota equals its size is computed exactly. So with the
//  default settings every component of up to 32 nodes is exact (small components are where a
//  wrong value is easy to spot by eye), and `samples >= N` is the exact algorithm. The pivots
//  are processed in the order of one seeded permutation, so components interleave.
//
// ─── Conventions ────────────────────────────────────────────────────────────────────────────
//  Distances are hop counts unless stated. -1 means unreachable. Isolated nodes get 0 in every
//  centrality except PageRank (where they keep their teleport share). All randomness comes
//  from rng( seed ) (prepare.js): same input, same output. No recursion anywhere. Every
//  function documents its time and memory complexity.

import { rng } from './prepare.js';

/**
 * @typedef {object} Graph
 * @property {number} N
 * @property {number} E
 * @property {Uint32Array} edges
 * @property {Uint32Array} adjStart
 * @property {Uint32Array} degree
 * @property {Uint32Array} adjList
 * @property {Uint32Array} [adjEdge]
 */

/** Work budget, in pivots * ( N + 2E ), that defaultSamples() may spend above its 256 pivot floor. */
const SAMPLE_WORK_BUDGET = 1e8;

/** Pivots every component gets at least (when the sample and the component are that large). */
const MIN_COMPONENT_PIVOTS = 32;

// ─── Construction ────────────────────────────────────────────────────────────────────────────

/**
 * CSR adjacency (and the edge index) of a clean edge list, in the exact layout of prepareGraph.
 * The edge list must already be undirected, deduplicated and free of self loops (normalizeGraph
 * output). Time and memory O( N + E ).
 *
 * @param {number} nodeCount
 * @param {ArrayLike<number>} edges flat pairs
 * @returns {Graph}
 */
export function graphFromEdges( nodeCount, edges ) {

	const N = nodeCount >>> 0;
	const list = edges instanceof Uint32Array ? edges : Uint32Array.from( edges );
	const E = list.length >> 1;
	const degree = new Uint32Array( N );
	for ( let k = 0; k < 2 * E; k ++ ) degree[ list[ k ] ] ++;
	const adjStart = new Uint32Array( N );
	for ( let i = 1; i < N; i ++ ) adjStart[ i ] = adjStart[ i - 1 ] + degree[ i - 1 ];
	const adjList = new Uint32Array( Math.max( 1, 2 * E ) );
	const adjEdge = new Uint32Array( Math.max( 1, 2 * E ) );
	const fill = adjStart.slice();
	for ( let e = 0; e < E; e ++ ) {

		const a = list[ 2 * e ], b = list[ 2 * e + 1 ];
		adjEdge[ fill[ a ] ] = e; adjList[ fill[ a ] ++ ] = b;
		adjEdge[ fill[ b ] ] = e; adjList[ fill[ b ] ++ ] = a;

	}

	return { N, E, edges: list, degree, adjStart, adjList, adjEdge };

}

/**
 * Edge id of every adjacency slot: adjEdge[ k ] is the index (into g.edges) of the edge between
 * the owner of slot k and adjList[ k ]. Needed to turn a node path into edge ids.
 * O( E ) when adjList is in prepareGraph's order (edge order), otherwise O( E ) expected through
 * a hash map. Throws when adjList and edges describe different graphs.
 *
 * @param {Graph} g
 * @returns {Uint32Array} length max( 1, 2E )
 */
export function buildEdgeIndex( g ) {

	const { N, E, edges, adjStart, degree, adjList } = g;
	const adjEdge = new Uint32Array( Math.max( 1, 2 * E ) );
	const fill = adjStart.slice();
	let inEdgeOrder = true;
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		const sa = fill[ a ] ++, sb = fill[ b ] ++;
		if ( adjList[ sa ] !== b || adjList[ sb ] !== a ) {

			inEdgeOrder = false;
			break;

		}

		adjEdge[ sa ] = e;
		adjEdge[ sb ] = e;

	}

	if ( inEdgeOrder ) return adjEdge;

	// any other adjacency order (e.g. sorted neighbour lists)
	const idOf = new Map();
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		idOf.set( a < b ? a * N + b : b * N + a, e );

	}

	for ( let v = 0; v < N; v ++ ) {

		for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) {

			const w = adjList[ k ];
			const e = idOf.get( v < w ? v * N + w : w * N + v );
			if ( e === undefined ) throw new Error( `buildEdgeIndex: adjacency ${v}-${w} is not in the edge list` );
			adjEdge[ k ] = e;

		}

	}

	return adjEdge;

}

const edgeIndexCache = new WeakMap();

function edgeIndexOf( g, options ) {

	if ( options && options.adjEdge ) return options.adjEdge;
	if ( g.adjEdge ) return g.adjEdge;
	let adjEdge = edgeIndexCache.get( g.adjList );
	if ( ! adjEdge ) {

		adjEdge = buildEdgeIndex( g );
		edgeIndexCache.set( g.adjList, adjEdge );

	}

	return adjEdge;

}

function checkNode( g, v, name ) {

	if ( ! Number.isInteger( v ) || v < 0 || v >= g.N ) throw new RangeError( `analytics: ${name} ${v} is not a node index (0..${g.N - 1})` );

}

// Scratch for the interactive point queries, reused between calls so that a query costs only
// what it touches (no O( N ) clear): `seen` holds a stamp that changes on every call.
let scratch = null;

function getScratch( N ) {

	if ( ! scratch || scratch.N !== N ) {

		scratch = {
			N, stamp: 0,
			seen: new Uint32Array( N ),
			pred: new Int32Array( N ),
			slot: new Int32Array( N ),
			queueA: new Uint32Array( N ),
			queueB: new Uint32Array( N ),
			heap: null,
		};

	}

	if ( scratch.stamp > 0xfffffff0 ) {

		scratch.seen.fill( 0 );
		scratch.stamp = 0;

	}

	scratch.stamp += 2; // stamp and stamp + 1 are both fresh
	return scratch;

}

// ─── Traversal ───────────────────────────────────────────────────────────────────────────────

/**
 * Breadth-first search from one node or from several (all at distance 0).
 *
 * The result drives the "ripple" animation: `dist` is the per-node arrival step (upload it as a
 * node attribute), `maxDist` the number of steps, and order.subarray( levelOffsets[ d ],
 * levelOffsets[ d + 1 ] ) the wavefront at step d.
 *
 * Time O( N + E ) (O( N ) for the clears plus the edges of what is reached), memory O( N ).
 *
 * @param {Graph} g
 * @param {number|ArrayLike<number>} sources
 * @param {object} [options]
 * @param {number} [options.maxDepth=Infinity] nodes farther than this stay at -1
 * @param {object} [options.out] a previous result for the same N, whose arrays are reused
 * @returns {{ dist: Int32Array, parent: Int32Array, order: Uint32Array, levelOffsets: Uint32Array, maxDist: number, reached: number }}
 *   dist: hops from the nearest source (-1 unreachable); parent: BFS tree (-1 for sources and
 *   unreached nodes); order: reached nodes in visiting order (a view of length `reached`).
 */
export function bfs( g, sources, options = {} ) {

	const { N, adjStart, degree, adjList } = g;
	const maxDepth = options.maxDepth === undefined ? Infinity : options.maxDepth;
	const out = options.out;
	const reuse = out && out.dist && out.dist.length === N && out.order && out.order.buffer.byteLength === 4 * N;
	const dist = reuse ? out.dist : new Int32Array( N );
	const parent = reuse ? out.parent : new Int32Array( N );
	const queue = reuse ? new Uint32Array( out.order.buffer ) : new Uint32Array( N );
	dist.fill( - 1 );
	parent.fill( - 1 );

	let tail = 0;
	if ( typeof sources === 'number' ) {

		checkNode( g, sources, 'source' );
		dist[ sources ] = 0;
		queue[ tail ++ ] = sources;

	} else {

		for ( let i = 0; i < sources.length; i ++ ) {

			const s = sources[ i ];
			checkNode( g, s, 'source' );
			if ( dist[ s ] === 0 ) continue;
			dist[ s ] = 0;
			queue[ tail ++ ] = s;

		}

	}

	const levels = [ 0 ];
	let levelStart = 0, depth = 0;
	while ( levelStart < tail && depth < maxDepth ) {

		const levelEnd = tail;
		for ( let i = levelStart; i < levelEnd; i ++ ) {

			const v = queue[ i ];
			for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) {

				const w = adjList[ k ];
				if ( dist[ w ] < 0 ) {

					dist[ w ] = depth + 1;
					parent[ w ] = v;
					queue[ tail ++ ] = w;

				}

			}

		}

		levelStart = levelEnd;
		if ( tail > levelEnd ) {

			depth ++;
			levels.push( levelEnd );

		}

	}

	levels.push( tail );
	return { dist, parent, order: queue.subarray( 0, tail ), levelOffsets: Uint32Array.from( levels ), maxDist: depth, reached: tail };

}

/**
 * The nodes within k hops of a node (or of a set of nodes), nearest first.
 * Time O( N + edges of the neighbourhood ), memory O( N ).
 *
 * @param {Graph} g
 * @param {number|ArrayLike<number>} sources
 * @param {number} k
 * @returns {{ nodes: Uint32Array, hops: Uint32Array, levelOffsets: Uint32Array, dist: Int32Array, maxHop: number }}
 *   nodes (sources first) and their hop counts in parallel; nodes with hop h are
 *   nodes.subarray( levelOffsets[ h ], levelOffsets[ h + 1 ] ); dist is the same per node
 *   (-1 outside the neighbourhood), ready to upload as a node attribute.
 */
export function khopNeighborhood( g, sources, k ) {

	const r = bfs( g, sources, { maxDepth: Math.max( 0, k ) } );
	const nodes = r.order.slice();
	const hops = new Uint32Array( nodes.length );
	for ( let i = 0; i < nodes.length; i ++ ) hops[ i ] = r.dist[ nodes[ i ] ];
	return { nodes, hops, levelOffsets: r.levelOffsets, dist: r.dist, maxHop: r.maxDist };

}

// ─── Shortest paths ──────────────────────────────────────────────────────────────────────────

/**
 * One shortest (fewest hops) path between two nodes, by bidirectional BFS: both ends grow one
 * level at a time, always the side with the smaller frontier, until an edge joins them.
 *
 * Time O( edges touched ): about O( b^(L/2) ) for branching factor b and path length L, where a
 * plain BFS costs O( b^L ); worst case O( N + E ). No per-call O( N ) work (stamped scratch).
 * Among equally short paths the choice is deterministic (adjacency order).
 *
 * @param {Graph} g
 * @param {number} s
 * @param {number} t
 * @param {object} [options]
 * @param {Uint32Array} [options.adjEdge] edge index (see buildEdgeIndex)
 * @returns {{ nodes: Uint32Array, edges: Uint32Array, length: number }|null} nodes from s to t,
 *   the edge ids between consecutive nodes, length = hops; null when t is unreachable.
 */
export function shortestPath( g, s, t, options = {} ) {

	const { N, adjStart, degree, adjList } = g;
	checkNode( g, s, 'source' );
	checkNode( g, t, 'target' );
	if ( s === t ) return { nodes: Uint32Array.of( s ), edges: new Uint32Array( 0 ), length: 0 };

	const sc = getScratch( N );
	const A = sc.stamp, B = A + 1;
	const { seen, pred, slot, queueA, queueB } = sc;
	seen[ s ] = A;
	seen[ t ] = B;
	queueA[ 0 ] = s;
	queueB[ 0 ] = t;
	let a0 = 0, a1 = 1, b0 = 0, b1 = 1;
	let meetA = - 1, meetB = - 1, meetSlot = - 1;

	// Invariant: both sides hold complete BFS levels and no edge joins them, so the first joining
	// edge found while growing one level closes a shortest path.
	search: while ( a0 < a1 && b0 < b1 ) {

		if ( a1 - a0 <= b1 - b0 ) {

			const levelEnd = a1;
			for ( let i = a0; i < levelEnd; i ++ ) {

				const u = queueA[ i ];
				for ( let k = adjStart[ u ], end = k + degree[ u ]; k < end; k ++ ) {

					const w = adjList[ k ], mark = seen[ w ];
					if ( mark === A ) continue;
					if ( mark === B ) {

						meetA = u; meetB = w; meetSlot = k;
						break search;

					}

					seen[ w ] = A;
					pred[ w ] = u;
					slot[ w ] = k;
					queueA[ a1 ++ ] = w;

				}

			}

			a0 = levelEnd;

		} else {

			const levelEnd = b1;
			for ( let i = b0; i < levelEnd; i ++ ) {

				const u = queueB[ i ];
				for ( let k = adjStart[ u ], end = k + degree[ u ]; k < end; k ++ ) {

					const w = adjList[ k ], mark = seen[ w ];
					if ( mark === B ) continue;
					if ( mark === A ) {

						meetA = w; meetB = u; meetSlot = k;
						break search;

					}

					seen[ w ] = B;
					pred[ w ] = u;
					slot[ w ] = k;
					queueB[ b1 ++ ] = w;

				}

			}

			b0 = levelEnd;

		}

	}

	if ( meetSlot < 0 ) return null;

	const adjEdge = edgeIndexOf( g, options );
	let la = 0, lb = 0;
	for ( let v = meetA; v !== s; v = pred[ v ] ) la ++;
	for ( let v = meetB; v !== t; v = pred[ v ] ) lb ++;
	const length = la + lb + 1;
	const nodes = new Uint32Array( length + 1 ), edges = new Uint32Array( length );
	for ( let v = meetA, i = la; ; v = pred[ v ], i -- ) {

		nodes[ i ] = v;
		if ( v === s ) break;
		edges[ i - 1 ] = adjEdge[ slot[ v ] ];

	}

	edges[ la ] = adjEdge[ meetSlot ];
	for ( let v = meetB, i = la + 1; ; v = pred[ v ], i ++ ) {

		nodes[ i ] = v;
		if ( v === t ) break;
		edges[ i ] = adjEdge[ slot[ v ] ];

	}

	return { nodes, edges, length };

}

/**
 * Shortest path by Euclidean edge length, the length of an edge being the distance between its
 * endpoints in `positions` (e.g. the layout readback): the path that looks shortest on screen.
 * A* with the straight-line distance to the target as heuristic and an indexed binary heap;
 * options.heuristic = false gives plain Dijkstra (same result, more nodes expanded).
 *
 * Time O( ( n + m ) log n ) for the n nodes and m edges expanded, worst case
 * O( ( N + E ) log N ); memory O( N ), reused between calls. Edges with a non-finite length
 * (NaN positions) are ignored.
 *
 * @param {Graph} g
 * @param {number} s
 * @param {number} t
 * @param {ArrayLike<number>} positions x, y, z of node i at positions[ i * stride + 0..2 ]
 * @param {object} [options]
 * @param {number} [options.stride=4]
 * @param {boolean} [options.heuristic=true]
 * @param {Uint32Array} [options.adjEdge]
 * @returns {{ nodes: Uint32Array, edges: Uint32Array, length: number }|null} length is the
 *   Euclidean length of the path (hops = edges.length); null when t is unreachable.
 */
export function shortestPathWeighted( g, s, t, positions, options = {} ) {

	const { N, adjStart, degree, adjList } = g;
	checkNode( g, s, 'source' );
	checkNode( g, t, 'target' );
	if ( s === t ) return { nodes: Uint32Array.of( s ), edges: new Uint32Array( 0 ), length: 0 };
	const stride = options.stride || 4;
	const useHeuristic = options.heuristic !== false;

	const sc = getScratch( N );
	if ( ! sc.heap ) {

		sc.heap = new Uint32Array( N );
		sc.heapPos = new Int32Array( N );
		sc.gScore = new Float64Array( N );
		sc.fScore = new Float64Array( N );

	}

	const stamp = sc.stamp;
	const { seen, pred, slot, heap, heapPos, gScore, fScore } = sc;
	const tx = positions[ t * stride ], ty = positions[ t * stride + 1 ], tz = positions[ t * stride + 2 ];

	seen[ s ] = stamp;
	gScore[ s ] = 0;
	fScore[ s ] = 0;
	heap[ 0 ] = s;
	heapPos[ s ] = 0;
	let size = 1, found = false;

	while ( size > 0 ) {

		// pop the node with the smallest f
		const u = heap[ 0 ];
		heapPos[ u ] = - 1;
		size --;
		if ( size > 0 ) {

			const last = heap[ size ], fLast = fScore[ last ];
			let i = 0;
			for ( ;; ) {

				let c = 2 * i + 1;
				if ( c >= size ) break;
				if ( c + 1 < size && fScore[ heap[ c + 1 ] ] < fScore[ heap[ c ] ] ) c ++;
				if ( fScore[ heap[ c ] ] >= fLast ) break;
				heap[ i ] = heap[ c ];
				heapPos[ heap[ i ] ] = i;
				i = c;

			}

			heap[ i ] = last;
			heapPos[ last ] = i;

		}

		if ( u === t ) {

			found = true;
			break;

		}

		const gu = gScore[ u ];
		const ux = positions[ u * stride ], uy = positions[ u * stride + 1 ], uz = positions[ u * stride + 2 ];
		for ( let k = adjStart[ u ], end = k + degree[ u ]; k < end; k ++ ) {

			const w = adjList[ k ];
			const wx = positions[ w * stride ], wy = positions[ w * stride + 1 ], wz = positions[ w * stride + 2 ];
			const dx = wx - ux, dy = wy - uy, dz = wz - uz;
			const cand = gu + Math.sqrt( dx * dx + dy * dy + dz * dz );
			const known = seen[ w ] === stamp;
			if ( ! ( cand < ( known ? gScore[ w ] : Infinity ) ) ) continue; // also rejects NaN
			if ( ! known ) {

				seen[ w ] = stamp;
				heapPos[ w ] = - 1;

			}

			gScore[ w ] = cand;
			pred[ w ] = u;
			slot[ w ] = k;
			let f = cand;
			if ( useHeuristic ) {

				const hx = tx - wx, hy = ty - wy, hz = tz - wz;
				f += Math.sqrt( hx * hx + hy * hy + hz * hz );

			}

			fScore[ w ] = f;
			// insert, or decrease the key of a queued node (a settled node is re-opened, which
			// keeps the result optimal even if rounding breaks the heuristic's consistency)
			let i = heapPos[ w ];
			if ( i < 0 ) i = size ++;
			while ( i > 0 ) {

				const p = ( i - 1 ) >> 1, pn = heap[ p ];
				if ( fScore[ pn ] <= f ) break;
				heap[ i ] = pn;
				heapPos[ pn ] = i;
				i = p;

			}

			heap[ i ] = w;
			heapPos[ w ] = i;

		}

	}

	if ( ! found ) return null;

	const adjEdge = edgeIndexOf( g, options );
	let hops = 0;
	for ( let v = t; v !== s; v = pred[ v ] ) hops ++;
	const nodes = new Uint32Array( hops + 1 ), edges = new Uint32Array( hops );
	for ( let v = t, i = hops; ; v = pred[ v ], i -- ) {

		nodes[ i ] = v;
		if ( v === s ) break;
		edges[ i - 1 ] = adjEdge[ slot[ v ] ];

	}

	return { nodes, edges, length: gScore[ t ] };

}

/**
 * All geodesics between two nodes: every node and edge that lies on at least one shortest
 * path, with the number of shortest paths through each. Useful to draw the whole "bundle" of
 * equally short routes, with width or opacity by share ( nodePaths / pathCount ).
 *
 * Forward BFS from s that counts paths and stops after t's level, then a backward sweep from t
 * along the shortest-path DAG. Time O( N + E ), memory O( N ). Counts are doubles: exact up to
 * 2^53, rounded beyond.
 *
 * @param {Graph} g
 * @param {number} s
 * @param {number} t
 * @param {object} [options]
 * @param {Uint32Array} [options.adjEdge]
 * @returns {{ length: number, pathCount: number, nodes: Uint32Array, depth: Uint32Array, nodePaths: Float64Array, edges: Uint32Array, edgePaths: Float64Array }|null}
 *   nodes sorted by depth (hops from s; s first, t last), nodePaths[ i ] = geodesics through
 *   nodes[ i ], edges = edge ids of the DAG, edgePaths likewise; null when t is unreachable.
 */
export function shortestPathDag( g, s, t, options = {} ) {

	const { N, adjStart, degree, adjList } = g;
	checkNode( g, s, 'source' );
	checkNode( g, t, 'target' );
	if ( s === t ) {

		return {
			length: 0, pathCount: 1, nodes: Uint32Array.of( s ), depth: new Uint32Array( 1 ),
			nodePaths: Float64Array.of( 1 ), edges: new Uint32Array( 0 ), edgePaths: new Float64Array( 0 ),
		};

	}

	const dist = new Int32Array( N ).fill( - 1 );
	const sigma = new Float64Array( N ); // shortest paths from s
	const queue = new Uint32Array( N );
	dist[ s ] = 0;
	sigma[ s ] = 1;
	queue[ 0 ] = s;
	let head = 0, tail = 1;
	while ( head < tail && dist[ t ] < 0 ) {

		// one full level, so that sigma is final for every node of the next level
		const levelEnd = tail;
		for ( ; head < levelEnd; head ++ ) {

			const v = queue[ head ];
			const next = dist[ v ] + 1, sv = sigma[ v ];
			for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) {

				const w = adjList[ k ], dw = dist[ w ];
				if ( dw < 0 ) {

					dist[ w ] = next;
					sigma[ w ] = sv;
					queue[ tail ++ ] = w;

				} else if ( dw === next ) sigma[ w ] += sv;

			}

		}

	}

	if ( dist[ t ] < 0 ) return null;

	const adjEdge = edgeIndexOf( g, options );
	const length = dist[ t ];
	const tau = new Float64Array( N ); // shortest paths to t, for nodes on a geodesic
	const back = queue; // the forward queue is no longer needed
	const edgeIds = [], edgeCounts = [];
	tau[ t ] = 1;
	back[ 0 ] = t;
	head = 0; tail = 1;
	while ( head < tail ) {

		// FIFO from t: every node of depth d + 1 is handled before any node of depth d, so tau
		// is final when a node is taken
		const w = back[ head ++ ];
		const prev = dist[ w ] - 1, tw = tau[ w ];
		if ( prev < 0 ) continue;
		for ( let k = adjStart[ w ], end = k + degree[ w ]; k < end; k ++ ) {

			const v = adjList[ k ];
			if ( dist[ v ] !== prev ) continue;
			if ( tau[ v ] === 0 ) back[ tail ++ ] = v;
			tau[ v ] += tw;
			edgeIds.push( adjEdge[ k ] );
			edgeCounts.push( sigma[ v ] * tw );

		}

	}

	const count = tail;
	const nodes = new Uint32Array( count ), depth = new Uint32Array( count ), nodePaths = new Float64Array( count );
	for ( let i = 0; i < count; i ++ ) {

		const v = back[ count - 1 - i ];
		nodes[ i ] = v;
		depth[ i ] = dist[ v ];
		nodePaths[ i ] = sigma[ v ] * tau[ v ];

	}

	return { length, pathCount: sigma[ t ], nodes, depth, nodePaths, edges: Uint32Array.from( edgeIds ), edgePaths: Float64Array.from( edgeCounts ) };

}

// ─── Structure ───────────────────────────────────────────────────────────────────────────────

/**
 * Connected components by BFS. Labels are sorted by size: component 0 is the largest (ties:
 * the one containing the smallest node index first). Time O( N + E ), memory O( N ).
 *
 * @param {Graph} g
 * @returns {{ component: Uint32Array, count: number, sizes: Uint32Array }}
 */
export function connectedComponents( g ) {

	const { N, adjStart, degree, adjList } = g;
	const label = new Int32Array( N ).fill( - 1 );
	const queue = new Uint32Array( N );
	const found = [];
	for ( let s = 0; s < N; s ++ ) {

		if ( label[ s ] >= 0 ) continue;
		const c = found.length;
		let head = 0, tail = 0;
		label[ s ] = c;
		queue[ tail ++ ] = s;
		while ( head < tail ) {

			const v = queue[ head ++ ];
			for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) {

				const w = adjList[ k ];
				if ( label[ w ] < 0 ) {

					label[ w ] = c;
					queue[ tail ++ ] = w;

				}

			}

		}

		found.push( tail );

	}

	const count = found.length;
	const bySize = Array.from( found.keys() ).sort( ( a, b ) => found[ b ] - found[ a ] || a - b );
	const remap = new Uint32Array( count ), sizes = new Uint32Array( count );
	for ( let i = 0; i < count; i ++ ) {

		remap[ bySize[ i ] ] = i;
		sizes[ i ] = found[ bySize[ i ] ];

	}

	const component = new Uint32Array( N );
	for ( let v = 0; v < N; v ++ ) component[ v ] = remap[ label[ v ] ];
	return { component, count, sizes };

}

/**
 * k-core decomposition (Batagelj and Zaversnik, 2003): core[ v ] is the largest k such that v
 * belongs to a subgraph in which every node has degree >= k. Bucket sort by degree, then
 * peeling in place. Time O( N + E ), memory O( N + max degree ).
 *
 * @param {Graph} g
 * @returns {{ core: Uint32Array, degeneracy: number, order: Uint32Array }} degeneracy = largest
 *   core number; order = the peeling order (a degeneracy ordering: every node has at most
 *   `degeneracy` neighbours later in it).
 */
export function kCore( g ) {

	const { N, adjStart, degree, adjList } = g;
	const deg = Uint32Array.from( degree );
	let maxDeg = 0;
	for ( let v = 0; v < N; v ++ ) if ( deg[ v ] > maxDeg ) maxDeg = deg[ v ];
	const bin = new Uint32Array( maxDeg + 2 ); // bin[ d ] = first position of degree d in `order`
	for ( let v = 0; v < N; v ++ ) bin[ deg[ v ] + 1 ] ++;
	for ( let d = 1; d <= maxDeg + 1; d ++ ) bin[ d ] += bin[ d - 1 ];
	const order = new Uint32Array( N ), pos = new Uint32Array( N );
	{

		const fill = bin.slice();
		for ( let v = 0; v < N; v ++ ) {

			pos[ v ] = fill[ deg[ v ] ] ++;
			order[ pos[ v ] ] = v;

		}

	}

	let degeneracy = 0;
	for ( let i = 0; i < N; i ++ ) {

		const v = order[ i ], dv = deg[ v ]; // deg[ v ] is now final: the core number
		if ( dv > degeneracy ) degeneracy = dv;
		for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) {

			const u = adjList[ k ], du = deg[ u ];
			if ( du <= dv ) continue;
			// move u to the front of its bin, then shrink the bin from the left
			const pu = pos[ u ], pw = bin[ du ], w = order[ pw ];
			if ( u !== w ) {

				order[ pu ] = w; pos[ w ] = pu;
				order[ pw ] = u; pos[ u ] = pw;

			}

			bin[ du ] ++;
			deg[ u ] = du - 1;

		}

	}

	return { core: deg, degeneracy, order };

}

/**
 * Triangles and clustering coefficients. Every edge is oriented from its lower to its higher
 * endpoint in ( degree, index ) order; each triangle u < v < w is then found exactly once, from
 * its lowest node u, by marking the out-neighbours of u and scanning the out-list of each of
 * them (the degree-ordered orientation of the "forward" family of triangle algorithms, with
 * neighbour marking in place of sorted-list intersection).
 *
 * Time O( N + E + sum over nodes of outdegree^2 ) <= O( E * sqrt( E ) ): the orientation bounds
 * every out-degree by sqrt( 2E ). Memory O( N + E ).
 *
 * @param {Graph} g
 * @param {object} [options]
 * @param {boolean} [options.edges=false] also count the triangles through every edge
 * @returns {{ local: Float32Array, global: number, average: number, triangles: Uint32Array, triangleCount: number, edgeTriangles: Uint32Array|null }}
 *   local[ v ] = triangles[ v ] / ( d ( d - 1 ) / 2 ), 0 when d < 2; global = transitivity =
 *   3 * triangleCount / connected triples; average = mean of local over all nodes;
 *   edgeTriangles[ e ] = common neighbours of the edge's endpoints (its "embeddedness").
 */
export function clusteringCoefficient( g, options = {} ) {

	const { N, E, edges, degree } = g;
	const outStart = new Uint32Array( N + 1 );
	const lowEnd = new Uint8Array( E ); // 1 when the second endpoint is the lower one
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		const da = degree[ a ], db = degree[ b ];
		const bLower = db < da || ( db === da && b < a );
		lowEnd[ e ] = bLower ? 1 : 0;
		outStart[ ( bLower ? b : a ) + 1 ] ++;

	}

	for ( let v = 0; v < N; v ++ ) outStart[ v + 1 ] += outStart[ v ];
	const outList = new Uint32Array( Math.max( 1, E ) ), outEdge = new Uint32Array( Math.max( 1, E ) );
	{

		const fill = outStart.slice( 0, N );
		for ( let e = 0; e < E; e ++ ) {

			const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
			const from = lowEnd[ e ] ? b : a, to = lowEnd[ e ] ? a : b;
			outEdge[ fill[ from ] ] = e;
			outList[ fill[ from ] ++ ] = to;

		}

	}

	// The inner loop has no data-dependent branch: `mark` is 0 or 1 and is added, not tested
	// (a test would be mispredicted too often; this form is about 1.6 times faster in V8).
	// Counts are kept per out-slot and per node of the current out-list, and distributed after.
	const triangles = new Uint32Array( N );
	const slotTriangles = options.edges ? new Uint32Array( Math.max( 1, E ) ) : null;
	const mark = new Uint8Array( N ); // 1 for the out-neighbours of the current u
	const closing = new Uint32Array( N ); // triangles of the current u closed by the arc u -> w
	let triangleCount = 0;
	for ( let u = 0; u < N; u ++ ) {

		const uStart = outStart[ u ], uEnd = outStart[ u + 1 ];
		if ( uEnd - uStart < 2 ) continue;
		for ( let k = uStart; k < uEnd; k ++ ) mark[ outList[ k ] ] = 1;
		let atU = 0;
		for ( let k = uStart; k < uEnd; k ++ ) {

			const v = outList[ k ];
			let atV = 0;
			if ( slotTriangles ) {

				for ( let j = outStart[ v ], jEnd = outStart[ v + 1 ]; j < jEnd; j ++ ) {

					const w = outList[ j ], hit = mark[ w ]; // triangle u, v, w when w is marked
					atV += hit;
					closing[ w ] += hit;
					slotTriangles[ j ] += hit;

				}

				slotTriangles[ k ] += atV;

			} else {

				for ( let j = outStart[ v ], jEnd = outStart[ v + 1 ]; j < jEnd; j ++ ) {

					const w = outList[ j ], hit = mark[ w ];
					atV += hit;
					closing[ w ] += hit;

				}

			}

			triangles[ v ] += atV;
			atU += atV;

		}

		triangles[ u ] += atU;
		triangleCount += atU;
		for ( let k = uStart; k < uEnd; k ++ ) {

			const w = outList[ k ], n = closing[ w ];
			triangles[ w ] += n;
			if ( slotTriangles ) slotTriangles[ k ] += n;
			closing[ w ] = 0;
			mark[ w ] = 0;

		}

	}

	let edgeTriangles = null;
	if ( slotTriangles ) {

		edgeTriangles = new Uint32Array( E );
		for ( let k = 0; k < E; k ++ ) edgeTriangles[ outEdge[ k ] ] = slotTriangles[ k ];

	}

	const local = new Float32Array( N );
	let triples = 0, sum = 0;
	for ( let v = 0; v < N; v ++ ) {

		const d = degree[ v ];
		if ( d < 2 ) continue;
		const pairs = d * ( d - 1 ) / 2;
		const c = triangles[ v ] / pairs;
		local[ v ] = c;
		triples += pairs;
		sum += c;

	}

	return {
		local,
		global: triples > 0 ? 3 * triangleCount / triples : 0,
		average: N > 0 ? sum / N : 0,
		triangles, triangleCount, edgeTriangles,
	};

}

// ─── Spectral centralities ───────────────────────────────────────────────────────────────────

/**
 * PageRank by power iteration (pull form). Isolated nodes are "dangling": the mass that sits on
 * them is redistributed along the teleport vector, so the result always sums to 1. Starts from
 * the degree distribution within each component, which is already the answer for damping 1.
 *
 * With `personalization` the teleport returns to the given nodes only (random walk with
 * restart): a one-hot vector gives a "relevance to the selected node" score, which is exactly
 * 0 for every node the selected one cannot reach.
 *
 * Time O( iterations * ( N + E ) ), memory O( N ). The L1 error shrinks by at least the factor
 * `damping` per iteration, so iterations <= log( tolerance ) / log( damping ) (85 by default),
 * usually far fewer.
 *
 * @param {Graph} g
 * @param {object} [options]
 * @param {number} [options.damping=0.85]
 * @param {number} [options.tolerance=1e-6] stop when the L1 change of one iteration is below it
 * @param {number} [options.maxIterations=100]
 * @param {ArrayLike<number>} [options.personalization] non-negative weights per node (any scale)
 * @param {object} [options.components] a connectedComponents( g ) result, to avoid recomputing it
 * @param {object} [options.stats] if given, receives { iterations, error, converged }
 * @returns {Float32Array} probabilities per node, sum 1
 */
export function pageRank( g, options = {} ) {

	const { N, adjStart, degree, adjList } = g;
	const damping = options.damping === undefined ? 0.85 : options.damping;
	const tolerance = options.tolerance === undefined ? 1e-6 : options.tolerance;
	const maxIterations = options.maxIterations === undefined ? 100 : options.maxIterations;
	const result = new Float32Array( N );
	if ( N === 0 ) return result;

	// teleport vector (null = uniform)
	let teleport = null;
	if ( options.personalization ) {

		let total = 0;
		teleport = new Float64Array( N );
		for ( let v = 0; v < N; v ++ ) {

			const w = + options.personalization[ v ];
			if ( w > 0 && Number.isFinite( w ) ) {

				teleport[ v ] = w;
				total += w;

			}

		}

		if ( total > 0 ) for ( let v = 0; v < N; v ++ ) teleport[ v ] /= total;
		else teleport = null;

	}

	// Start: in the limit every component holds exactly the teleport mass that lands in it;
	// spread the damped part of that mass by degree (the fixed point for damping 1). A component
	// without teleport mass starts, and therefore stays, at exactly 0.
	const uniform = 1 / N;
	const comp = options.components || connectedComponents( g );
	const landing = new Float64Array( comp.count ), degreeSum = new Float64Array( comp.count );
	for ( let v = 0; v < N; v ++ ) {

		const c = comp.component[ v ];
		landing[ c ] += teleport ? teleport[ v ] : uniform;
		degreeSum[ c ] += degree[ v ];

	}

	let p = new Float64Array( N ), next = new Float64Array( N );
	const share = new Float64Array( N ); // p[ v ] / degree[ v ]
	const invDegree = new Float64Array( N );
	for ( let v = 0; v < N; v ++ ) {

		const c = comp.component[ v ], tv = teleport ? teleport[ v ] : uniform;
		invDegree[ v ] = degree[ v ] > 0 ? 1 / degree[ v ] : 0;
		p[ v ] = degree[ v ] > 0 ? ( 1 - damping ) * tv + damping * landing[ c ] * degree[ v ] / degreeSum[ c ] : tv;

	}

	let iterations = 0, error = Infinity;
	while ( iterations < maxIterations && error >= tolerance ) {

		let dangling = 0;
		for ( let v = 0; v < N; v ++ ) {

			share[ v ] = p[ v ] * invDegree[ v ];
			if ( invDegree[ v ] === 0 ) dangling += p[ v ];

		}

		const base = 1 - damping + damping * dangling;
		error = 0;
		for ( let v = 0; v < N; v ++ ) {

			let sum = 0;
			for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) sum += share[ adjList[ k ] ];
			const value = base * ( teleport ? teleport[ v ] : uniform ) + damping * sum;
			error += Math.abs( value - p[ v ] );
			next[ v ] = value;

		}

		const swap = p; p = next; next = swap;
		iterations ++;

	}

	let total = 0;
	for ( let v = 0; v < N; v ++ ) total += p[ v ];
	for ( let v = 0; v < N; v ++ ) result[ v ] = p[ v ] / total;
	if ( options.stats ) Object.assign( options.stats, { iterations, error, converged: error < tolerance } );
	return result;

}

/**
 * Eigenvector centrality: the principal eigenvector of the adjacency matrix, by power iteration
 * on A + I. The shift leaves the eigenvectors unchanged and removes the oscillation that plain
 * power iteration shows on bipartite graphs (trees, grids, stars), whose spectrum is symmetric.
 *
 * Disconnected graphs: the global eigenvector is non-zero only on the component with the
 * largest eigenvalue, which makes it useless as a visual metric. By default (perComponent) each
 * component gets its own principal eigenvector, scaled so that every component has the same
 * root-mean-square value; a value is then "centrality relative to the typical node of its own
 * component". On a connected graph both modes return the usual unit-length vector.
 * perComponent = false gives the plain global iteration (smaller components decay to 0).
 * Isolated nodes get 0.
 *
 * Time O( iterations * ( N + E ) ), memory O( N ). The rate is ( 1 + l2 ) / ( 1 + l1 ) for the
 * two largest eigenvalue magnitudes of a component: slow when communities are nearly
 * disconnected or the component is bipartite with a large l1; check options.stats.converged.
 *
 * @param {Graph} g
 * @param {object} [options]
 * @param {number} [options.tolerance=1e-6] stop when the mean absolute change per node is below it
 * @param {number} [options.maxIterations=500]
 * @param {boolean} [options.perComponent=true]
 * @param {object} [options.components] a connectedComponents( g ) result, to avoid recomputing it
 * @param {object} [options.stats] if given, receives { iterations, error, converged }
 * @returns {Float32Array} non-negative, unit Euclidean length (all 0 for a graph without edges)
 */
export function eigenvectorCentrality( g, options = {} ) {

	const { N, adjStart, degree, adjList } = g;
	const tolerance = options.tolerance === undefined ? 1e-6 : options.tolerance;
	const maxIterations = options.maxIterations === undefined ? 500 : options.maxIterations;
	const perComponent = options.perComponent !== false;
	const result = new Float32Array( N );

	// groups: the components (or one group for the global mode); isolated nodes stay at 0
	let group = null, groupCount = 1;
	if ( perComponent ) {

		const comp = options.components || connectedComponents( g );
		group = comp.component;
		groupCount = comp.count;

	}

	const groupSize = new Float64Array( groupCount ), norm = new Float64Array( groupCount );
	let x = new Float64Array( N ), y = new Float64Array( N );
	let active = 0;
	for ( let v = 0; v < N; v ++ ) {

		if ( degree[ v ] === 0 ) continue;
		x[ v ] = 1;
		active ++;
		groupSize[ group ? group[ v ] : 0 ] ++;

	}

	let iterations = 0, error = active > 0 ? Infinity : 0;
	while ( iterations < maxIterations && error >= tolerance * active ) {

		norm.fill( 0 );
		for ( let v = 0; v < N; v ++ ) {

			let sum = x[ v ];
			for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) sum += x[ adjList[ k ] ];
			y[ v ] = sum;
			norm[ group ? group[ v ] : 0 ] += sum * sum;

		}

		// rescale every group to root-mean-square 1
		for ( let c = 0; c < groupCount; c ++ ) norm[ c ] = norm[ c ] > 0 ? Math.sqrt( groupSize[ c ] / norm[ c ] ) : 0;
		error = 0;
		for ( let v = 0; v < N; v ++ ) {

			const value = y[ v ] * norm[ group ? group[ v ] : 0 ];
			error += Math.abs( value - x[ v ] );
			y[ v ] = value;

		}

		const swap = x; x = y; y = swap;
		iterations ++;

	}

	const scale = active > 0 ? 1 / Math.sqrt( active ) : 0;
	for ( let v = 0; v < N; v ++ ) result[ v ] = x[ v ] * scale;
	if ( options.stats ) Object.assign( options.stats, { iterations, error: active > 0 ? error / active : 0, converged: error < tolerance * active || active === 0 } );
	return result;

}

// ─── Pivot sampling and tasks ────────────────────────────────────────────────────────────────

/**
 * Default pivot count of the sampled algorithms: as many as fit a fixed work budget
 * ( pivots * ( N + 2E ) <= 1e8: measured 0.30 s of betweenness at 5k nodes / 40k edges with
 * 1,181 pivots ), but at least 256 (0.35 s at 20k / 384k) and at most N. So a graph is computed
 * exactly when N ( N + 2E ) <= 1e8 (e.g. 2,000 nodes / 24,000 edges). O( 1 ).
 *
 * @param {Graph} g
 * @returns {number}
 */
export function defaultSamples( g ) {

	const work = Math.max( 1, g.N + 2 * g.E );
	return Math.min( g.N, Math.max( 256, Math.floor( SAMPLE_WORK_BUDGET / work ) ) );

}

// Pivots in processing order plus the quota of each component (see "Sampling" in the header).
function planPivots( g, comp, samples, seed ) {

	const { N } = g;
	const { component, count, sizes } = comp;
	const wanted = samples === undefined || samples === null || samples === 'auto' || Number.isNaN( + samples ) ? defaultSamples( g ) : + samples;
	const k = Math.max( 1, Math.min( N, Math.floor( wanted ) ) );
	const quota = new Uint32Array( count );
	// at least 2: a pivot's own value is extrapolated from the other pivots of its component
	const least = Math.max( 2, Math.min( MIN_COMPONENT_PIVOTS, k ) );
	let total = 0, exact = true;
	for ( let c = 0; c < count; c ++ ) {

		const n = sizes[ c ];
		if ( n < 2 ) continue;
		const q = k >= N ? n : Math.min( n, Math.max( least, Math.round( k * n / N ) ) );
		quota[ c ] = q;
		total += q;
		if ( q < n ) exact = false;

	}

	// one seeded permutation; each component takes the first `quota` of its nodes in that order
	const rand = rng( seed === undefined ? 1 : seed );
	const perm = new Uint32Array( N );
	for ( let i = 0; i < N; i ++ ) perm[ i ] = i;
	for ( let i = N - 1; i > 0; i -- ) {

		const j = Math.floor( rand() * ( i + 1 ) );
		const tmp = perm[ i ]; perm[ i ] = perm[ j ]; perm[ j ] = tmp;

	}

	const pivots = new Uint32Array( total ), taken = new Uint32Array( count );
	let m = 0;
	for ( let i = 0; i < N && m < total; i ++ ) {

		const v = perm[ i ], c = component[ v ];
		if ( taken[ c ] < quota[ c ] ) {

			taken[ c ] ++;
			pivots[ m ++ ] = v;

		}

	}

	return { pivots, quota, exact };

}

/**
 * Runs a task to completion on the calling thread.
 *
 * @param {object} task
 * @param {object} [options]
 * @param {function(number, number): (boolean|void)} [options.onProgress] called with
 *   ( completed, total ) after every chunk; return false to stop early (the task keeps a valid
 *   partial result)
 * @param {number} [options.chunk=4] pivots per chunk
 * @returns {boolean} true when the task finished, false when it was stopped
 */
export function runTask( task, options = {} ) {

	const { onProgress } = options;
	const chunk = Math.max( 1, options.chunk || 4 );
	if ( ! onProgress ) return task.step( task.total );
	while ( ! task.done ) {

		task.step( chunk );
		if ( onProgress( task.completed, task.total ) === false ) return task.done;

	}

	return true;

}

/**
 * Runs a task in time slices, yielding to the event loop between them, so that a worker can
 * post progress and receive a cancel message (or a page can stay responsive).
 *
 * @param {object} task
 * @param {object} [options]
 * @param {function(number, number): (boolean|void)} [options.onProgress] as in runTask
 * @param {number} [options.budgetMs=30] length of one slice
 * @param {AbortSignal} [options.signal] checked after every slice
 * @param {function(): Promise} [options.wait] how to yield; default setTimeout( 0 ) (which
 *   browsers clamp to 4 ms when nested, hence the 30 ms slices)
 * @returns {Promise<boolean>} true when the task finished, false when it was cancelled
 */
export async function runTaskAsync( task, options = {} ) {

	const { onProgress, signal } = options;
	const budgetMs = options.budgetMs === undefined ? 30 : options.budgetMs;
	const wait = options.wait || ( () => new Promise( ( resolve ) => setTimeout( resolve, 0 ) ) );
	const now = typeof performance !== 'undefined' ? () => performance.now() : () => Date.now();
	if ( signal && signal.aborted ) return false;
	while ( ! task.done ) {

		const t0 = now();
		do task.step( 1 ); while ( ! task.done && now() - t0 < budgetMs );
		if ( onProgress && onProgress( task.completed, task.total ) === false ) return task.done;
		if ( task.done ) break;
		await wait();
		if ( signal && signal.aborted ) return false;

	}

	return true;

}

// ─── Betweenness ─────────────────────────────────────────────────────────────────────────────

// The pivot loops below run a direction-optimising BFS (Beamer, Asanović and Patterson, 2012).
// A level is normally expanded top-down (scan the edges of the frontier). When the nodes not
// yet reached have fewer edges in total than the frontier has, it is cheaper to go bottom-up:
// scan those nodes and look for neighbours in the frontier. On small-world graphs, where one
// or two levels hold most of the nodes, this skips most of the 2E edge visits of a plain BFS;
// by construction a level never costs more than it would top-down (plus one pass over the
// component, once per BFS, to list the unreached nodes).
//
// BRANCH-FREE: in the middle levels of such a BFS "is this neighbour new?" is close to a coin
// flip, and the mispredicted branches cost more than the memory accesses. The top-down loops
// therefore write unconditionally and advance their cursors by 0 or 1 ( queue[ tail ] = w;
// tail += fresh ), which is why the queue and pair arrays have one spare slot at the end.
// Measured in V8 (Node 22) on the 20k node / 384k edge LFR preset: about 1.4 times faster per
// Brandes pivot and about 1.2 times for the plain distance BFS than the branching loops.

// Nodes grouped by component, and the degree sum of each component.
function componentIndex( g, comp ) {

	const { N, degree } = g;
	const { component, count, sizes } = comp;
	const offset = new Uint32Array( count + 1 );
	for ( let c = 0; c < count; c ++ ) offset[ c + 1 ] = offset[ c ] + sizes[ c ];
	const nodes = new Uint32Array( N ), degreeSum = new Float64Array( count );
	const fill = offset.slice( 0, count );
	for ( let v = 0; v < N; v ++ ) {

		const c = component[ v ];
		nodes[ fill[ c ] ++ ] = v;
		degreeSum[ c ] += degree[ v ];

	}

	return { nodes, offset, degreeSum };

}

// Distance-only BFS with reusable scratch. run( s ) returns the eccentricity L of s and leaves
// dist filled for the reached nodes, which are queue[ 0 .. reached ) in level order, level d
// being queue[ levelStart[ d ] .. levelStart[ d + 1 ] ). reset() puts dist back to -1.
function createSweeper( g, comp ) {

	const { N, adjStart, degree, adjList } = g;
	const index = componentIndex( g, comp );
	const dist = new Int32Array( N ).fill( - 1 );
	const queue = new Uint32Array( N + 1 ), rest = new Uint32Array( N );
	const levelStart = new Uint32Array( N + 2 );
	const sweeper = {
		dist, queue, levelStart, reached: 0,

		run( s ) {

			const c = comp.component[ s ];
			let frontierDegree = degree[ s ], unreachedDegree = index.degreeSum[ c ] - frontierDegree;
			let restCount = - 1; // unreached nodes of the component, listed on first use
			dist[ s ] = 0;
			queue[ 0 ] = s;
			levelStart[ 0 ] = 0;
			let levelBegin = 0, tail = 1, depth = 0;
			while ( levelBegin < tail ) {

				const levelEnd = tail, next = depth + 1;
				let found = 0; // degree sum of the next level
				if ( unreachedDegree < frontierDegree ) {

					// bottom-up
					if ( restCount < 0 ) {

						restCount = 0;
						for ( let i = index.offset[ c ], end = index.offset[ c + 1 ]; i < end; i ++ ) {

							const w = index.nodes[ i ];
							if ( dist[ w ] < 0 ) rest[ restCount ++ ] = w;

						}

					}

					let keep = 0;
					for ( let i = 0; i < restCount; i ++ ) {

						const w = rest[ i ];
						if ( dist[ w ] >= 0 ) continue; // reached by a top-down level since it was listed
						let hit = false;
						for ( let k = adjStart[ w ], end = k + degree[ w ]; k < end; k ++ ) {

							if ( dist[ adjList[ k ] ] === depth ) {

								hit = true;
								break;

							}

						}

						if ( hit ) {

							dist[ w ] = next;
							queue[ tail ++ ] = w;
							found += degree[ w ];

						} else rest[ keep ++ ] = w;

					}

					restCount = keep;

				} else {

					// top-down, without data-dependent branches (see BRANCH-FREE above)
					for ( let i = levelBegin; i < levelEnd; i ++ ) {

						const v = queue[ i ];
						for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) {

							const w = adjList[ k ], dw = dist[ w ];
							const fresh = dw >>> 31; // 1 when w is unreached (-1)
							dist[ w ] = dw + fresh * ( next + 1 );
							queue[ tail ] = w;
							tail += fresh;

						}

					}

					for ( let i = levelEnd; i < tail; i ++ ) found += degree[ queue[ i ] ];

				}

				levelStart[ next ] = levelEnd;
				levelBegin = levelEnd;
				depth = next;
				unreachedDegree -= found;
				frontierDegree = found;

			}

			// the last pass found nothing: depth is one past the deepest level
			sweeper.reached = tail;
			return depth - 1;

		},

		reset() {

			for ( let i = 0; i < sweeper.reached; i ++ ) dist[ queue[ i ] ] = - 1;
			sweeper.reached = 0;

		},
	};

	return sweeper;

}

// Brandes' algorithm for one pivot at a time, with reusable scratch.
//
// Forward: the BFS above, counting shortest paths (sigma) and recording every edge of the
// shortest-path DAG as a pair ( from, to ), level by level. A bottom-up level scans the whole
// adjacency of each unreached node, so it sees every predecessor: the counts stay exact.
//
// Backward: with c[ w ] = ( 1 + delta[ w ] ) / sigma[ w ], the dependency recurrence
//   delta[ v ] = sum over DAG successors w of sigma[ v ] / sigma[ w ] * ( 1 + delta[ w ] )
// becomes  delta[ v ] = sigma[ v ] * t[ v ]  with  t[ v ] = sum of c[ w ],  c[ v ] = t[ v ] + 1 / sigma[ v ].
// So the pass is, from the deepest level up: turn t into c for the level's nodes, then one
// branch-free loop over the level's pairs, c[ from ] += c[ to ]. One array holds t, then c.
function createBrandes( g, comp, adjEdge ) {

	const { N, E, adjStart, degree, adjList } = g;
	const index = componentIndex( g, comp );
	const dist = new Int32Array( N ).fill( - 1 );
	const sigma = new Float64Array( N ), coef = new Float64Array( N );
	const queue = new Uint32Array( N + 1 ), rest = new Uint32Array( N );
	const levelStart = new Uint32Array( N + 2 ), pairStart = new Uint32Array( N + 2 );
	// every undirected edge is a DAG edge at most once per pivot (+ 1: see BRANCH-FREE above)
	const pairFrom = new Uint32Array( E + 1 ), pairTo = new Uint32Array( E + 1 );
	const pairEdge = adjEdge ? new Uint32Array( E + 1 ) : null;

	// adds the dependencies of pivot s to accNode (and accEdge, when there is an edge index)
	return function run( s, accNode, accEdge ) {

		const c = comp.component[ s ];
		let frontierDegree = degree[ s ], unreachedDegree = index.degreeSum[ c ] - frontierDegree;
		let restCount = - 1;
		dist[ s ] = 0;
		sigma[ s ] = 1;
		queue[ 0 ] = s;
		levelStart[ 0 ] = 0;
		let levelBegin = 0, tail = 1, depth = 0, m = 0;
		while ( levelBegin < tail ) {

			const levelEnd = tail, next = depth + 1;
			let found = 0;
			pairStart[ depth ] = m;
			if ( unreachedDegree < frontierDegree ) {

				// bottom-up: all predecessors of w are its neighbours at the current depth
				if ( restCount < 0 ) {

					restCount = 0;
					for ( let i = index.offset[ c ], end = index.offset[ c + 1 ]; i < end; i ++ ) {

						const w = index.nodes[ i ];
						if ( dist[ w ] < 0 ) rest[ restCount ++ ] = w;

					}

				}

				let keep = 0;
				for ( let i = 0; i < restCount; i ++ ) {

					const w = rest[ i ];
					if ( dist[ w ] >= 0 ) continue;
					let paths = 0;
					for ( let k = adjStart[ w ], end = k + degree[ w ]; k < end; k ++ ) {

						const v = adjList[ k ];
						const diff = dist[ v ] ^ depth;
						const hit = ( ( diff | - diff ) >>> 31 ) ^ 1; // 1 when v is in the frontier
						paths += hit * sigma[ v ];
						pairFrom[ m ] = v;
						pairTo[ m ] = w;
						if ( pairEdge ) pairEdge[ m ] = adjEdge[ k ];
						m += hit;

					}

					if ( paths > 0 ) {

						dist[ w ] = next;
						sigma[ w ] = paths;
						queue[ tail ++ ] = w;
						found += degree[ w ];

					} else rest[ keep ++ ] = w;

				}

				restCount = keep;

			} else {

				// top-down, without data-dependent branches (see BRANCH-FREE above)
				for ( let i = levelBegin; i < levelEnd; i ++ ) {

					const v = queue[ i ], sv = sigma[ v ];
					for ( let k = adjStart[ v ], end = k + degree[ v ]; k < end; k ++ ) {

						const w = adjList[ k ], dw = dist[ w ];
						const fresh = dw >>> 31; // 1 when w is unreached (-1)
						const dNew = dw + fresh * ( next + 1 );
						const diff = dNew ^ next;
						const onDag = ( ( diff | - diff ) >>> 31 ) ^ 1; // 1 when w is on the next level
						dist[ w ] = dNew;
						queue[ tail ] = w;
						tail += fresh;
						sigma[ w ] += onDag * sv;
						pairFrom[ m ] = v;
						pairTo[ m ] = w;
						if ( pairEdge ) pairEdge[ m ] = adjEdge[ k ];
						m += onDag;

					}

				}

				for ( let i = levelEnd; i < tail; i ++ ) found += degree[ queue[ i ] ];

			}

			levelStart[ next ] = levelEnd;
			levelBegin = levelEnd;
			depth = next;
			unreachedDegree -= found;
			frontierDegree = found;

		}

		// levels 0 .. depth - 1; the pairs from level l to l + 1 are pairStart[ l ] .. pairStart[ l + 1 ]
		for ( let level = depth - 1; level >= 1; level -- ) {

			for ( let i = levelStart[ level ], end = levelStart[ level + 1 ]; i < end; i ++ ) {

				const w = queue[ i ], sw = sigma[ w ], t = coef[ w ];
				accNode[ w ] += sw * t; // delta[ w ]: exactly 0 for a node without successors
				coef[ w ] = t + 1 / sw;

			}

			const first = pairStart[ level - 1 ], end = pairStart[ level ];
			if ( accEdge ) {

				for ( let j = first; j < end; j ++ ) {

					const v = pairFrom[ j ], cw = coef[ pairTo[ j ] ];
					coef[ v ] += cw;
					accEdge[ pairEdge[ j ] ] += sigma[ v ] * cw;

				}

			} else {

				for ( let j = first; j < end; j ++ ) coef[ pairFrom[ j ] ] += coef[ pairTo[ j ] ];

			}

		}

		for ( let i = 0; i < tail; i ++ ) {

			const v = queue[ i ];
			dist[ v ] = - 1;
			sigma[ v ] = 0;
			coef[ v ] = 0;

		}

	};

}

/**
 * Betweenness centrality as a resumable task (see "Tasks" and "Sampling" in the file header).
 * Brandes' algorithm (2001): per pivot, one BFS that counts shortest paths and one reverse
 * accumulation over the shortest-path DAG; the BFS is direction-optimising.
 *
 * Values count unordered pairs: node[ v ] = sum over pairs { s, t } (both different from v) of
 * the fraction of shortest s-t paths through v; edge[ e ] likewise for paths using e (endpoints
 * included, so every edge has at least 1). With fewer pivots than nodes the sums over sources
 * are extrapolated per component: a node that was not a pivot is scaled by ( n - 1 ) / k, a node
 * that was a pivot by ( n - 1 ) / ( k - 1 ) (as a source it contributes nothing to itself), an
 * edge by n / k, for k pivots done in a component of n nodes. These are unbiased, and exact at
 * k = n.
 *
 * Time O( pivots * ( N + E ) ): O( N * E ) exact. Memory O( N + E ).
 *
 * @param {Graph} g
 * @param {object} [options]
 * @param {number} [options.samples] pivot count; default defaultSamples( g ); >= N means exact
 * @param {number} [options.seed=1]
 * @param {boolean} [options.edges=false] also compute edge betweenness
 * @param {boolean} [options.normalized=false] divide node values by ( N - 1 )( N - 2 ) / 2 and
 *   edge values by N ( N - 1 ) / 2, the number of pairs, giving [0,1]
 * @param {Uint32Array} [options.adjEdge]
 * @param {object} [options.components] a connectedComponents( g ) result
 * @returns {{ total: number, completed: number, done: boolean, exact: boolean, step: function(number=): boolean, result: function(): { node: Float32Array, edge: Float32Array|null, pivots: number, exact: boolean } }}
 */
export function betweennessTask( g, options = {} ) {

	const { N, E, edges } = g;
	const comp = options.components || connectedComponents( g );
	const plan = planPivots( g, comp, options.samples, options.seed );
	const { pivots } = plan;
	const brandes = createBrandes( g, comp, options.edges ? edgeIndexOf( g, options ) : null );
	const accNode = new Float64Array( N );
	const accEdge = options.edges ? new Float64Array( Math.max( 1, E ) ) : null;
	const doneIn = new Uint32Array( comp.count ); // pivots processed per component
	const wasPivot = new Uint8Array( N );

	const task = {
		total: pivots.length,
		completed: 0,
		done: pivots.length === 0,
		exact: plan.exact,

		step( count = 1 ) {

			const stop = Math.min( pivots.length, task.completed + Math.max( 1, count ) );
			for ( let i = task.completed; i < stop; i ++ ) {

				const s = pivots[ i ];
				brandes( s, accNode, accEdge );
				wasPivot[ s ] = 1;
				doneIn[ comp.component[ s ] ] ++;

			}

			task.completed = stop;
			task.done = stop === pivots.length;
			return task.done;

		},

		result() {

			// every unordered pair is seen from both of its ends, hence the factor 1/2
			const nodeNorm = options.normalized && N > 2 ? 2 / ( ( N - 1 ) * ( N - 2 ) ) : 1;
			const edgeNorm = options.normalized && N > 1 ? 2 / ( N * ( N - 1 ) ) : 1;
			const node = new Float32Array( N );
			for ( let v = 0; v < N; v ++ ) {

				const c = comp.component[ v ];
				const k = doneIn[ c ] - wasPivot[ v ];
				if ( k > 0 ) node[ v ] = 0.5 * nodeNorm * accNode[ v ] * ( comp.sizes[ c ] - 1 ) / k;

			}

			let edge = null;
			if ( accEdge ) {

				edge = new Float32Array( E );
				for ( let e = 0; e < E; e ++ ) {

					const c = comp.component[ edges[ 2 * e ] ];
					if ( doneIn[ c ] > 0 ) edge[ e ] = 0.5 * edgeNorm * accEdge[ e ] * comp.sizes[ c ] / doneIn[ c ];

				}

			}

			return { node, edge, pivots: task.completed, exact: plan.exact && task.done };

		},
	};

	return task;

}

/**
 * Node betweenness centrality (see betweennessTask for definition, scaling and options).
 * Time O( pivots * ( N + E ) ).
 *
 * @param {Graph} g
 * @param {object} [options] betweennessTask options, plus
 * @param {function(number, number): (boolean|void)} [options.onProgress] ( completed, total );
 *   return false to stop early and get the estimate from the pivots done so far
 * @param {number} [options.chunk=4] pivots between two onProgress calls
 * @returns {Float32Array}
 */
export function betweenness( g, options = {} ) {

	const task = betweennessTask( g, { ...options, edges: false } );
	runTask( task, options );
	return task.result().node;

}

/**
 * Edge betweenness centrality, e.g. for edge thickness: bridges between communities carry many
 * shortest paths. Same options as betweenness(). Time O( pivots * ( N + E ) ), memory O( N + E ).
 *
 * @param {Graph} g
 * @param {object} [options]
 * @returns {Float32Array} per edge id
 */
export function edgeBetweenness( g, options = {} ) {

	const task = betweennessTask( g, { ...options, edges: true } );
	runTask( task, options );
	return task.result().edge;

}

// ─── Closeness, harmonic centrality, distance distribution ───────────────────────────────────

/**
 * Closeness and harmonic centrality, plus the distance distribution, as a resumable task (see
 * "Tasks" and "Sampling" in the file header). One direction-optimising BFS per pivot.
 *
 *   closeness[ v ] = ( r - 1 ) / ( N - 1 ) * ( r - 1 ) / ( sum of distances from v ), r = size of
 *                    v's component (Wasserman and Faust's form: the inverse mean distance,
 *                    scaled by the fraction of the graph v can reach, so that nodes of small
 *                    components do not score high). In a connected graph: 1 / mean distance.
 *   harmonic[ v ]  = mean over all other nodes u of 1 / distance( v, u ), with 1 / infinity = 0.
 *                    Needs no correction for disconnected graphs.
 *
 * Both are in [0,1] and equal 1 exactly for a node adjacent to every other node. A pivot knows
 * its own values exactly; any other node is estimated from its distances to the pivots of its
 * component (Eppstein and Wang, 2004), unbiased for the sums involved, exact when every node is
 * a pivot.
 *
 * Time O( pivots * ( N + E ) ), memory O( N ).
 *
 * @param {Graph} g
 * @param {object} [options]
 * @param {number} [options.samples] pivot count; default defaultSamples( g ); >= N means exact
 * @param {number} [options.seed=1]
 * @param {object} [options.components] a connectedComponents( g ) result
 * @returns {{ total: number, completed: number, done: boolean, exact: boolean, step: function(number=): boolean, result: function(): object }}
 *   result(): { closeness: Float32Array, harmonic: Float32Array, averagePathLength,
 *   distanceDistribution: Float64Array, effectiveDiameter, maxDistance, pivots, exact } where
 *   distanceDistribution[ d ] = fraction of connected pairs at distance d (index 0 unused),
 *   averagePathLength = its mean (0 for a graph without edges), effectiveDiameter = its 90th
 *   percentile (interpolated), maxDistance = the largest distance seen (a lower bound of the
 *   diameter).
 */
export function closenessTask( g, options = {} ) {

	const { N } = g;
	const comp = options.components || connectedComponents( g );
	const plan = planPivots( g, comp, options.samples, options.seed );
	const { pivots, quota } = plan;
	const sweeper = createSweeper( g, comp );
	const { queue, levelStart } = sweeper;
	const sumDist = new Float64Array( N ), sumInv = new Float64Array( N ); // over the pivots
	const ownDist = new Float64Array( N ), ownInv = new Float64Array( N ); // exact, pivots only
	const wasPivot = new Uint8Array( N );
	const doneIn = new Uint32Array( comp.count );
	const histogram = []; // pair counts per distance; a pivot's pairs stand for size / quota pivots' worth
	let maxDistance = 0;

	const task = {
		total: pivots.length,
		completed: 0,
		done: pivots.length === 0,
		exact: plan.exact,

		step( count = 1 ) {

			const stop = Math.min( pivots.length, task.completed + Math.max( 1, count ) );
			for ( let i = task.completed; i < stop; i ++ ) {

				const p = pivots[ i ], c = comp.component[ p ];
				const weight = comp.sizes[ c ] / quota[ c ]; // pairs represented by one pair of this pivot
				const depth = sweeper.run( p );
				let far = 0, near = 0;
				for ( let d = 1; d <= depth; d ++ ) {

					const first = levelStart[ d ], end = levelStart[ d + 1 ], inv = 1 / d;
					for ( let k = first; k < end; k ++ ) {

						const w = queue[ k ];
						sumDist[ w ] += d;
						sumInv[ w ] += inv;

					}

					far += ( end - first ) * d;
					near += ( end - first ) * inv;
					histogram[ d ] = ( histogram[ d ] || 0 ) + weight * ( end - first );

				}

				sweeper.reset();
				if ( depth > maxDistance ) maxDistance = depth;
				ownDist[ p ] = far;
				ownInv[ p ] = near;
				wasPivot[ p ] = 1;
				doneIn[ c ] ++;

			}

			task.completed = stop;
			task.done = stop === pivots.length;
			return task.done;

		},

		result() {

			const closeness = new Float32Array( N ), harmonic = new Float32Array( N );
			for ( let v = 0; v < N; v ++ ) {

				const c = comp.component[ v ], n = comp.sizes[ c ];
				if ( n < 2 ) continue;
				let far, near;
				if ( wasPivot[ v ] ) {

					far = ownDist[ v ];
					near = ownInv[ v ];

				} else {

					const k = doneIn[ c ];
					if ( k === 0 ) continue; // not estimated yet (partial result)
					far = sumDist[ v ] * ( n - 1 ) / k;
					near = sumInv[ v ] * ( n - 1 ) / k;

				}

				closeness[ v ] = ( n - 1 ) / far * ( n - 1 ) / ( N - 1 );
				harmonic[ v ] = near / ( N - 1 );

			}

			const distanceDistribution = new Float64Array( maxDistance + 1 );
			let pairs = 0, total = 0;
			for ( let d = 1; d <= maxDistance; d ++ ) {

				const h = histogram[ d ] || 0;
				pairs += h;
				total += h * d;

			}

			let effectiveDiameter = 0, cumulative = 0;
			for ( let d = 1; d <= maxDistance; d ++ ) {

				const f = pairs > 0 ? ( histogram[ d ] || 0 ) / pairs : 0;
				distanceDistribution[ d ] = f;
				if ( cumulative < 0.9 && cumulative + f >= 0.9 ) effectiveDiameter = d - 1 + ( 0.9 - cumulative ) / f;
				cumulative += f;

			}

			return {
				closeness, harmonic,
				averagePathLength: pairs > 0 ? total / pairs : 0,
				distanceDistribution, effectiveDiameter, maxDistance,
				pivots: task.completed, exact: plan.exact && task.done,
			};

		},
	};

	return task;

}

/**
 * Closeness and harmonic centrality with the distance statistics that come for free (see
 * closenessTask for definitions and options). Time O( pivots * ( N + E ) ).
 *
 * @param {Graph} g
 * @param {object} [options] closenessTask options, plus onProgress / chunk as in betweenness()
 * @returns {{ closeness: Float32Array, harmonic: Float32Array, averagePathLength: number, distanceDistribution: Float64Array, effectiveDiameter: number, maxDistance: number, pivots: number, exact: boolean }}
 */
export function closeness( g, options = {} ) {

	const task = closenessTask( g, options );
	runTask( task, options );
	return task.result();

}

/**
 * Average shortest path length over connected pairs, with the distance distribution ("hop
 * plot") and the effective diameter, from a sample of BFS sources (64 by default; pass
 * samples >= N for the exact value). Time O( samples * ( N + E ) ).
 *
 * @param {Graph} g
 * @param {object} [options]
 * @param {number} [options.samples=64]
 * @param {number} [options.seed=1]
 * @returns {{ averagePathLength: number, distanceDistribution: Float64Array, effectiveDiameter: number, maxDistance: number, pivots: number, exact: boolean }}
 */
export function averagePathLength( g, options = {} ) {

	const r = closeness( g, { ...options, samples: options.samples === undefined ? Math.min( 64, g.N ) : options.samples } );
	return {
		averagePathLength: r.averagePathLength, distanceDistribution: r.distanceDistribution,
		effectiveDiameter: r.effectiveDiameter, maxDistance: r.maxDistance, pivots: r.pivots, exact: r.exact,
	};

}

/**
 * Diameter and eccentricity bounds from a few BFS (Takes and Kosters' BoundingDiameters, 2011).
 * Every BFS from v gives ecc( v ) exactly and, for each node w at distance d of v,
 *   max( d, ecc( v ) - d ) <= ecc( w ) <= ecc( v ) + d.
 * Sources alternate between the node with the largest upper bound and the one with the smallest
 * lower bound; the first two are the classic double sweep (a hub of the largest component, then
 * the node farthest from it). The search stops when no node can still raise the diameter (then
 * it is exact) or after maxBfs sources.
 *
 * How soon the bounds meet depends on the graph. It is quick when central nodes have an
 * eccentricity near half the diameter (3 BFS on the karate club; Takes and Kosters report a
 * handful on many real networks). It takes thousands of BFS when almost every node has an
 * eccentricity of diameter - 1, as in the synthetic presets: measured 281 BFS on the 5k hSBM,
 * 3,896 on a 20k Barabási-Albert graph and 16,756 on the 20k LFR graph. With the default 32 the
 * result is then an interval (4..6 on the LFR graph, true value 4), and the lower bound itself
 * can still be short of the truth (6 after 2,048 BFS on the BA graph, true value 7). So check
 * `exact`, and show "d to dUpper" otherwise.
 *
 * For a disconnected graph the diameter is the largest finite distance, i.e. the largest
 * diameter of a component; eccentricities are within the node's own component.
 *
 * Time O( bfsCount * ( N + E ) ), memory O( N ).
 *
 * @param {Graph} g
 * @param {object} [options]
 * @param {number} [options.maxBfs=32]
 * @param {object} [options.components] a connectedComponents( g ) result
 * @returns {{ diameter: number, diameterUpper: number, exact: boolean, endpoints: Int32Array, eccLower: Int32Array, eccUpper: Int32Array, bfsCount: number }}
 *   diameter is a lower bound that is attained (the distance between `endpoints`, which are -1
 *   in a graph without edges) and diameterUpper an upper bound; they are equal when exact.
 */
export function estimateDiameter( g, options = {} ) {

	const { N, E, edges, degree } = g;
	const maxBfs = options.maxBfs === undefined ? 32 : options.maxBfs;
	const comp = options.components || connectedComponents( g );
	const lower = new Int32Array( N ), upper = new Int32Array( N );
	for ( let v = 0; v < N; v ++ ) {

		const n = comp.sizes[ comp.component[ v ] ];
		upper[ v ] = n - 1;
		lower[ v ] = n > 1 ? 1 : 0;

	}

	const endpoints = new Int32Array( 2 ).fill( - 1 );
	let diameter = 0;
	if ( E > 0 ) {

		diameter = 1;
		endpoints[ 0 ] = edges[ 0 ];
		endpoints[ 1 ] = edges[ 1 ];

	}

	const sweeper = createSweeper( g, comp );
	const { dist, queue } = sweeper;
	let bfsCount = 0, exact = false, wantUpper = true;
	for ( ;; ) {

		// next source among the nodes that could still raise the diameter
		let source = - 1;
		for ( let v = 0; v < N; v ++ ) {

			if ( upper[ v ] <= diameter || lower[ v ] === upper[ v ] ) continue;
			if ( source < 0 ) source = v;
			else if ( wantUpper ) {

				// largest upper bound; ties: a hub for the very first source, the periphery later
				if ( upper[ v ] > upper[ source ] ) source = v;
				else if ( upper[ v ] === upper[ source ] && ( bfsCount === 0 ? degree[ v ] > degree[ source ] : degree[ v ] < degree[ source ] ) ) source = v;

			} else if ( lower[ v ] < lower[ source ] || ( lower[ v ] === lower[ source ] && degree[ v ] > degree[ source ] ) ) source = v;

		}

		if ( source < 0 ) {

			exact = true;
			break;

		}

		if ( bfsCount >= maxBfs ) break;

		const ecc = sweeper.run( source );
		const reached = sweeper.reached, far = queue[ reached - 1 ];
		bfsCount ++;
		if ( ecc > diameter ) {

			diameter = ecc;
			endpoints[ 0 ] = source;
			endpoints[ 1 ] = far;

		}

		for ( let i = 0; i < reached; i ++ ) {

			const w = queue[ i ], d = dist[ w ];
			const lo = d > ecc - d ? d : ecc - d, up = ecc + d;
			if ( lo > lower[ w ] ) lower[ w ] = lo;
			if ( up < upper[ w ] ) upper[ w ] = up;

		}

		sweeper.reset();
		wantUpper = ! wantUpper;
		if ( bfsCount === 1 ) wantUpper = true; // second source: the farthest node (double sweep)

	}

	let diameterUpper = diameter;
	for ( let v = 0; v < N; v ++ ) if ( upper[ v ] > diameterUpper ) diameterUpper = upper[ v ];
	return { diameter, diameterUpper, exact, endpoints, eccLower: lower, eccUpper: upper, bfsCount };

}

// ─── Cheap statistics ────────────────────────────────────────────────────────────────────────

/**
 * Edge density 2E / ( N ( N - 1 ) ): the fraction of node pairs that are linked. O( 1 ).
 *
 * @param {Graph} g
 * @returns {number}
 */
export function density( g ) {

	return g.N > 1 ? 2 * g.E / ( g.N * ( g.N - 1 ) ) : 0;

}

/**
 * Degree histogram: result[ d ] = number of nodes of degree d, for d = 0..max degree. O( N ).
 *
 * @param {Graph} g
 * @returns {Uint32Array}
 */
export function degreeHistogram( g ) {

	const { N, degree } = g;
	let maxDegree = 0;
	for ( let v = 0; v < N; v ++ ) if ( degree[ v ] > maxDegree ) maxDegree = degree[ v ];
	const histogram = new Uint32Array( maxDegree + 1 );
	for ( let v = 0; v < N; v ++ ) histogram[ degree[ v ] ] ++;
	return histogram;

}

/**
 * Degree assortativity (Newman, 2002): the Pearson correlation of the degrees at the two ends
 * of an edge. Positive when hubs link to hubs (social networks), negative when hubs link to
 * low-degree nodes (stars, technological and biological networks). O( E ).
 *
 * @param {Graph} g
 * @returns {number} in [-1, 1]; NaN when every node with an edge has the same degree (regular
 *   graphs: the correlation is undefined) or there are no edges
 */
export function degreeAssortativity( g ) {

	const { E, edges, degree } = g;
	if ( E === 0 ) return NaN;
	let product = 0, sum = 0, squares = 0;
	for ( let e = 0; e < E; e ++ ) {

		const j = degree[ edges[ 2 * e ] ], k = degree[ edges[ 2 * e + 1 ] ];
		product += j * k;
		sum += j + k;
		squares += j * j + k * k;

	}

	const mean = sum / ( 2 * E );
	const variance = squares / ( 2 * E ) - mean * mean;
	if ( ! ( variance > 1e-12 * mean * mean ) ) return NaN;
	return ( product / E - mean * mean ) / variance;

}

/**
 * The cheap statistics in one object, for a stats panel. O( N + E ).
 *
 * @param {Graph} g
 * @returns {{ nodes: number, edges: number, density: number, minDegree: number, maxDegree: number, meanDegree: number, medianDegree: number, isolated: number, degreeHistogram: Uint32Array, assortativity: number }}
 */
export function graphSummary( g ) {

	const { N, E } = g;
	const histogram = degreeHistogram( g );
	let minDegree = 0;
	while ( minDegree < histogram.length - 1 && histogram[ minDegree ] === 0 ) minDegree ++;
	// median from the histogram (mean of the two middle values when N is even)
	let medianDegree = 0;
	if ( N > 0 ) {

		const lowRank = ( N - 1 ) >> 1, highRank = N >> 1;
		let seen = 0, low = - 1;
		for ( let d = 0; d < histogram.length; d ++ ) {

			seen += histogram[ d ];
			if ( low < 0 && seen > lowRank ) low = d;
			if ( seen > highRank ) {

				medianDegree = ( low + d ) / 2;
				break;

			}

		}

	}

	return {
		nodes: N, edges: E, density: density( g ),
		minDegree: N > 0 ? minDegree : 0, maxDegree: histogram.length - 1,
		meanDegree: N > 0 ? 2 * E / N : 0, medianDegree,
		isolated: N > 0 ? histogram[ 0 ] : 0,
		degreeHistogram: histogram,
		assortativity: degreeAssortativity( g ),
	};

}

// ─── Mapping a metric to a visual channel ────────────────────────────────────────────────────

function sortedFinite( values ) {

	const sorted = new Float64Array( values.length );
	let n = 0;
	for ( let i = 0; i < values.length; i ++ ) if ( Number.isFinite( values[ i ] ) ) sorted[ n ++ ] = values[ i ];
	return sorted.subarray( 0, n ).sort();

}

function quantileOfSorted( sorted, q ) {

	const n = sorted.length;
	if ( n === 0 ) return NaN;
	const x = Math.min( 1, Math.max( 0, q ) ) * ( n - 1 );
	const i = Math.floor( x ), f = x - i;
	return i + 1 < n ? sorted[ i ] + f * ( sorted[ i + 1 ] - sorted[ i ] ) : sorted[ n - 1 ];

}

/**
 * Quantiles of the finite values (linear interpolation between order statistics, the common
 * "type 7" definition). O( n log n ) for the sort.
 *
 * @param {ArrayLike<number>} values
 * @param {number[]} qs fractions in [0,1], e.g. [ 0.05, 0.5, 0.95 ]
 * @returns {number[]} NaN when there is no finite value
 */
export function quantiles( values, qs ) {

	const sorted = sortedFinite( values );
	return Array.from( qs, ( q ) => quantileOfSorted( sorted, q ) );

}

/**
 * Maps a metric to [0,1] for a visual channel (radius, colour ramp position, glow, width).
 *
 *   linear   ( v - min ) / ( max - min ). Keeps ratios; a single outlier flattens everything else.
 *   log      logarithmic, for heavy tails (degree, betweenness, PageRank). All values positive:
 *            log( v / lo ) / log( max / lo ) with lo = max( min, max / logRange ). Otherwise
 *            (zeros present): log1p( x / f ) / log1p( xmax / f ) with x = v - min and
 *            f = max( smallest positive x, xmax / logRange ), so 0 maps to 0. logRange (1e4 by
 *            default, four decades) is the dynamic range shown; what lies below it collapses to 0.
 *   rank     histogram equalisation: the fraction of the values above the minimum that are
 *            <= v, i.e. ( count( <= v ) - count( min ) ) / ( n - count( min ) ). Without ties this
 *            is rank / ( n - 1 ). Uniform use of the channel whatever the distribution; equal
 *            values get equal output; the minimum maps to 0 and the maximum to 1 even when
 *            many values tie there (e.g. the zeros of betweenness). Loses magnitudes.
 *   robust   linear between two percentiles (2% and 98% by default), clamped: linear without
 *            the outlier problem. Falls back to min / max when the two percentiles coincide.
 *
 * `gamma` is applied last ( t ^ gamma ): 0.5 makes a disc's AREA proportional to the metric when
 * the result drives a radius. Non-finite inputs map to `invalid`; when all finite values are
 * equal every one maps to `flat`.
 *
 * Time O( n ) for linear and log, O( n log n ) for rank and robust (one native sort). Memory
 * O( n ) for those two.
 *
 * @param {ArrayLike<number>} values
 * @param {object} [options]
 * @param {'linear'|'log'|'rank'|'robust'} [options.mode='linear']
 * @param {number[]} [options.percentiles=[0.02,0.98]] robust mode
 * @param {number} [options.logRange=1e4] log mode
 * @param {number} [options.gamma=1]
 * @param {number} [options.flat=0.5]
 * @param {number} [options.invalid=0]
 * @param {Float32Array} [options.out] written and returned instead of a new array (it may be
 *   `values` itself)
 * @returns {Float32Array}
 */
export function normalize( values, options = {} ) {

	const n = values.length;
	const mode = options.mode || 'linear';
	if ( mode !== 'linear' && mode !== 'log' && mode !== 'rank' && mode !== 'robust' ) throw new Error( `normalize: unknown mode "${mode}" (linear, log, rank, robust)` );
	const gamma = options.gamma === undefined ? 1 : options.gamma;
	const flat = options.flat === undefined ? 0.5 : options.flat;
	const invalid = options.invalid === undefined ? 0 : options.invalid;
	const out = options.out || new Float32Array( n );

	let min = Infinity, max = - Infinity, finite = 0;
	for ( let i = 0; i < n; i ++ ) {

		const v = values[ i ];
		if ( ! Number.isFinite( v ) ) continue;
		finite ++;
		if ( v < min ) min = v;
		if ( v > max ) max = v;

	}

	// positions that must not go through gamma (kept aside because `out` may be `values`)
	let skip = null;
	if ( finite < n && gamma !== 1 ) {

		skip = new Uint8Array( n );
		for ( let i = 0; i < n; i ++ ) if ( ! Number.isFinite( values[ i ] ) ) skip[ i ] = 1;

	}

	if ( ! ( max > min ) ) {

		// no finite value, or all equal
		for ( let i = 0; i < n; i ++ ) out[ i ] = Number.isFinite( values[ i ] ) ? flat : invalid;
		return out;

	}

	if ( mode === 'rank' && values instanceof Float32Array && n <= RANK_PACK_LIMIT ) {

		rankFloat32( values, out, invalid );

	} else if ( mode === 'rank' ) {

		const sorted = sortedFinite( values );
		const m = sorted.length;
		let atMin = 1; // number of values equal to the minimum
		while ( sorted[ atMin ] === sorted[ 0 ] ) atMin ++;
		for ( let i = 0; i < n; i ++ ) {

			const v = values[ i ];
			if ( ! Number.isFinite( v ) ) {

				out[ i ] = invalid;
				continue;

			}

			// number of values <= v
			let lo = 0, hi = m;
			while ( lo < hi ) {

				const mid = ( lo + hi ) >> 1;
				if ( sorted[ mid ] <= v ) lo = mid + 1; else hi = mid;

			}

			out[ i ] = ( lo - atMin ) / ( m - atMin );

		}

	} else if ( mode === 'log' ) {

		const logRange = options.logRange === undefined ? 1e4 : options.logRange;
		if ( min > 0 ) {

			const lo = Math.max( min, max / logRange );
			const scale = 1 / Math.log( max / lo );
			for ( let i = 0; i < n; i ++ ) {

				const v = values[ i ];
				out[ i ] = Number.isFinite( v ) ? ( v > lo ? Math.min( 1, Math.log( v / lo ) * scale ) : 0 ) : invalid;

			}

		} else {

			let smallest = Infinity;
			for ( let i = 0; i < n; i ++ ) {

				const x = values[ i ] - min;
				if ( x > 0 && x < smallest ) smallest = x;

			}

			const floor = Math.max( smallest, ( max - min ) / logRange );
			const scale = 1 / Math.log1p( ( max - min ) / floor );
			for ( let i = 0; i < n; i ++ ) {

				const v = values[ i ];
				out[ i ] = Number.isFinite( v ) ? Math.min( 1, Math.log1p( ( v - min ) / floor ) * scale ) : invalid;

			}

		}

	} else {

		let lo = min, hi = max;
		if ( mode === 'robust' ) {

			const [ qLo, qHi ] = options.percentiles || [ 0.02, 0.98 ];
			const sorted = sortedFinite( values );
			const a = quantileOfSorted( sorted, qLo ), b = quantileOfSorted( sorted, qHi );
			if ( b > a ) {

				lo = a;
				hi = b;

			}

		}

		const scale = 1 / ( hi - lo );
		for ( let i = 0; i < n; i ++ ) {

			const v = values[ i ];
			out[ i ] = Number.isFinite( v ) ? Math.min( 1, Math.max( 0, ( v - lo ) * scale ) ) : invalid;

		}

	}

	if ( gamma !== 1 ) {

		for ( let i = 0; i < n; i ++ ) if ( ! skip || skip[ i ] === 0 ) out[ i ] = Math.pow( out[ i ], gamma );

	}

	return out;

}

const RANK_PACK_LIMIT = 1 << 21;

// Rank normalisation of a Float32Array in one native sort: each finite value becomes the double
// key * 2^21 + index, where key is the value's bit pattern remapped so that unsigned order is
// numeric order (32 + 21 bits fit a double exactly). Sorting those sorts the values and carries
// the indices along, which a comparator sort or a search per element does several times slower.
function rankFloat32( values, out, invalid ) {

	const n = values.length;
	const bits = new Uint32Array( values.buffer, values.byteOffset, n );
	const packed = new Float64Array( n );
	let m = 0;
	for ( let i = 0; i < n; i ++ ) {

		const v = values[ i ];
		if ( ! Number.isFinite( v ) ) {

			out[ i ] = invalid;
			continue;

		}

		const b = v === 0 ? 0 : bits[ i ]; // -0 ranks with +0
		const key = b >= 0x80000000 ? ( ~ b ) >>> 0 : b + 0x80000000;
		packed[ m ++ ] = key * RANK_PACK_LIMIT + i;

	}

	const sorted = packed.subarray( 0, m ).sort();
	let atMin = 0; // number of values equal to the minimum (the first run)
	for ( let first = 0; first < m; ) {

		const key = Math.floor( sorted[ first ] / RANK_PACK_LIMIT );
		let last = first + 1;
		while ( last < m && Math.floor( sorted[ last ] / RANK_PACK_LIMIT ) === key ) last ++;
		if ( first === 0 ) atMin = last;
		const t = ( last - atMin ) / ( m - atMin ); // `last` values are <= this one
		for ( let j = first; j < last; j ++ ) out[ sorted[ j ] - key * RANK_PACK_LIMIT ] = t;
		first = last;

	}

}

// ─── Metric catalogue ────────────────────────────────────────────────────────────────────────

/**
 * Node metrics a UI can offer, with the normalisation that suits each ( `scale`, a normalize()
 * mode ) and a cost class, as measured at 20k nodes / 384k edges: 'instant' = one near-linear
 * pass, under 10 ms (clustering and triangles: about 30 ms); 'fast' = iterative, 20 to 80 ms;
 * 'slow' = pivot sampling, 0.1 to 0.4 s with the default pivot count and far more when asked
 * for more pivots: run it in a worker, as a task.
 */
export const NODE_METRICS = [
	{ id: 'degree', label: 'Degree', scale: 'log', cost: 'instant', description: 'Number of neighbours.' },
	{ id: 'core', label: 'k-core', scale: 'linear', cost: 'instant', description: 'Depth in the dense core of the graph.' },
	{ id: 'pagerank', label: 'PageRank', scale: 'log', cost: 'fast', description: 'Share of a random walk that visits the node.' },
	{ id: 'eigenvector', label: 'Eigenvector', scale: 'log', cost: 'fast', description: 'Being linked to well-linked nodes.' },
	{ id: 'betweenness', label: 'Betweenness', scale: 'log', cost: 'slow', description: 'Share of shortest paths through the node: bridges and brokers.' },
	{ id: 'closeness', label: 'Closeness', scale: 'robust', cost: 'slow', description: 'Inverse mean distance to the other nodes.' },
	{ id: 'harmonic', label: 'Harmonic', scale: 'robust', cost: 'slow', description: 'Mean inverse distance to the other nodes.' },
	{ id: 'clustering', label: 'Clustering', scale: 'linear', cost: 'instant', description: 'Fraction of a node’s neighbours that are linked to each other.' },
	{ id: 'triangles', label: 'Triangles', scale: 'log', cost: 'instant', description: 'Number of triangles through the node.' },
	{ id: 'component', label: 'Component', scale: 'rank', cost: 'instant', description: 'Connected component (0 = largest).' },
];

/**
 * One node metric by id (see NODE_METRICS), as a Float32Array of raw values: pass it through
 * normalize() with the metric's `scale` to drive a visual channel. `options` go to the
 * underlying function (samples, seed, onProgress, damping, ...). For the 'slow' metrics in a
 * worker prefer betweennessTask / closenessTask with runTaskAsync.
 *
 * @param {Graph} g
 * @param {string} id
 * @param {object} [options]
 * @returns {Float32Array}
 */
export function nodeMetric( g, id, options = {} ) {

	switch ( id ) {

		case 'degree': return Float32Array.from( g.degree );
		case 'core': return Float32Array.from( kCore( g ).core );
		case 'pagerank': return pageRank( g, options );
		case 'eigenvector': return eigenvectorCentrality( g, options );
		case 'betweenness': return betweenness( g, options );
		case 'closeness': return closeness( g, options ).closeness;
		case 'harmonic': return closeness( g, options ).harmonic;
		case 'clustering': return clusteringCoefficient( g ).local;
		case 'triangles': return Float32Array.from( clusteringCoefficient( g ).triangles );
		case 'component': return Float32Array.from( connectedComponents( g ).component );
		default: throw new Error( `nodeMetric: unknown metric "${id}" (${NODE_METRICS.map( ( m ) => m.id ).join( ', ' )})` );

	}

}
