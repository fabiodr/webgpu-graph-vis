// Static graph layouts and the connectivity-aware seed of the force layout. Pure JS on typed
// arrays (PreparedGraph fields in, positions out), no three.js and no recursion: runs in the data
// worker, on the main thread or in Node (tests). Deterministic: all randomness comes from
// rng( seed ) of prepare.js.
//
// Positions are returned as Float32Array( 4N ) in the layout of graph.initPos / app.positions
// (xyz, w = 0), so they can be handed to ForceLayout.setTargets() or GraphBuffers.setPositions().
//
// ─── stressSeed( g, { maxClusters = 400, fill = 0.9, seed = 1, iters = 30 } ) ────────────────
//  Initial positions for the force layout (prepare.js uses it for graph.initPos). The previous
//  seed spread sibling clusters on a Fibonacci sphere by mass, ignoring which clusters are
//  connected; a cooling force layout then does not finish untangling meshes and rings.
//   1. finest hierarchy level with at most `maxClusters` clusters; its quotient graph (clusters as
//      nodes, an edge when any graph edge joins two clusters);
//   2. cluster ball radius r_c = R·∛(mass_c / N)·fill, quotient edge length r_a + r_b (touching
//      balls), all-pairs shortest paths (Dijkstra), exact SGD² stress minimisation in 3D;
//   3. the centres are rescaled so that the node cloud has the reference extent (rms radius of a
//      uniform ball of radius R = g.layoutRadius) and overlapping balls are pushed apart;
//   4. below a solved cluster: sub-clusters on a Fibonacci sphere, nodes uniform in the ball of
//      their finest cluster.
//  Returns null when the graph has no hierarchy level with ≤ maxClusters clusters (D = 0 included);
//  the caller then keeps its own seed. Cost: 2–60 ms up to 2k nodes, 80–220 ms at 5k–20k nodes.
//   g: { N, D, E, edges, anc, levelOffset, clusterMass, clusterChildStart, clusterChildCount,
//        childRefs, layoutRadius }
//
// ─── sgd2Layout( g, { pivots = 'auto', iters = 30, eps = 0.1, seed = 1, dim = 3, init, radius } ) ─
//  Stress layout by stochastic gradient descent ("SGD²", Zheng, Pawar, Goodman, TVCG 2019) with
//  the sparse pivot approximation of Ortmann, Klimenta, Brandes (JGAA 2017). Terms: every edge
//  (d = 1, w = 1) and every (node, pivot) pair with BFS distance d and weight s/d², s = number of
//  nodes of the pivot's Voronoi region within d/2 of the pivot. Update for a term (i, j, d) with
//  step η: r = (|x_i − x_j| − d)/2 · unit vector, x_i −= min(1, w_ij·η)·r, x_j += min(1, w_ji·η)·r;
//  η decays exponentially from 1/w_min to eps/w_max; terms are reshuffled every sweep.
//  pivots: 'auto' = 200 up to 2000 nodes, 50 above; 0 or ≥ N = exact all-pairs stress.
//  The result is centred and scaled to the reference extent (rms radius √(3/5)·radius, radius
//  defaults to g.layoutRadius). Lowest stress of all layouts on meshes, rings, trees and small
//  graphs; a hairball on community graphs. Cost (one thread): 200 pivots 0.1–1.3 s up to 2k nodes;
//  50 pivots 0.6 s at 5k, ~4 s at 20k nodes. Meant for the data worker.
//   g: { N, E, edges, adjStart, degree, adjList, layoutRadius }
//   -> { positions: Float32Array( 4N ), info: { pivots, terms, exact, scale, ms } }
//
// ─── radialLayout( g, root, { shellGap, dim = 3, pack = 0.9, minSpacing, radius } ) ───────────
//  Focus layout: concentric shells by BFS depth around `root` ("cone tree on shells"). Every node
//  owns a spherical cap around its direction; its BFS children get sub-caps with a solid angle
//  proportional to their subtree size, placed on a Fibonacci spiral inside the parent's cap
//  (largest subtree at the cap centre). The BFS parent of a node is its neighbour of the previous
//  depth with the highest degree. Unreachable nodes sit on an extra outer shell. Shell radii are
//  at least `shellGap` (1.5·linkDistance) apart and grow when a shell is too crowded for
//  `minSpacing` (0.35·linkDistance); the whole layout is then scaled so that the outermost shell
//  has radius `radius` (default g.layoutRadius; null keeps the natural size).
//  dim = 2 gives rings in the z = 0 plane. O(N + E) plus a sort of every child list: < 1 ms for
//  small graphs, 17–41 ms at 5k and 53–62 ms at 20k nodes: cheap enough to recompute per click.
//   g: { N, adjStart, degree, adjList, linkDistance, layoutRadius }
//   -> { positions: Float32Array( 4N ), depth: Int32Array( N ) (−1 = unreachable), info }
//
// Helpers: fitToRadius( pos, N, R, stride = 3 ), quotientStress(), pickLevel(), toPositions4().

import { rng } from './prepare.js';

const GOLDEN = Math.PI * ( 3 - Math.sqrt( 5 ) );
const UNREACHED = 65535;

function fibonacci( k, m, out ) {

	const y = m === 1 ? 0 : 1 - ( 2 * ( k + 0.5 ) ) / m;
	const r = Math.sqrt( Math.max( 0, 1 - y * y ) );
	const phi = k * GOLDEN;
	out[ 0 ] = Math.cos( phi ) * r; out[ 1 ] = y; out[ 2 ] = Math.sin( phi ) * r;

}

/** Float64Array( 3N ) -> Float32Array( 4N ) with w = 0 (the layout of graph.initPos). */
export function toPositions4( pos3, N ) {

	const out = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) {

		out[ 4 * i ] = pos3[ 3 * i ]; out[ 4 * i + 1 ] = pos3[ 3 * i + 1 ]; out[ 4 * i + 2 ] = pos3[ 3 * i + 2 ];

	}

	return out;

}

/**
 * Centres a layout (in place) and rescales it so that its rms radius is √(3/5)·R, the rms radius
 * of a uniform ball of radius R. Returns the applied scale factor.
 */
export function fitToRadius( pos, N, R, stride = 3 ) {

	let cx = 0, cy = 0, cz = 0;
	for ( let i = 0; i < N; i ++ ) {

		cx += pos[ stride * i ]; cy += pos[ stride * i + 1 ]; cz += pos[ stride * i + 2 ];

	}

	const n = Math.max( 1, N );
	cx /= n; cy /= n; cz /= n;
	let q = 0;
	for ( let i = 0; i < N; i ++ ) {

		const o = stride * i;
		pos[ o ] -= cx; pos[ o + 1 ] -= cy; pos[ o + 2 ] -= cz;
		q += pos[ o ] * pos[ o ] + pos[ o + 1 ] * pos[ o + 1 ] + pos[ o + 2 ] * pos[ o + 2 ];

	}

	const s = ( Math.sqrt( 0.6 ) * R ) / Math.max( 1e-12, Math.sqrt( q / n ) );
	for ( let i = 0; i < N; i ++ ) {

		const o = stride * i;
		pos[ o ] *= s; pos[ o + 1 ] *= s; pos[ o + 2 ] *= s;

	}

	return s;

}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Stress seed
// ─────────────────────────────────────────────────────────────────────────────────────────────

/** Exact weighted SGD² on a small graph given a full distance matrix (Float32Array K·K). */
function sgdDense( K, dist, { iters = 30, eps = 0.1, seed = 1 } = {} ) {

	const rand = rng( seed );
	const pos = new Float64Array( 3 * K );
	for ( let k = 0; k < 3 * K; k ++ ) pos[ k ] = rand();
	const T = ( K * ( K - 1 ) ) / 2;
	if ( T === 0 ) return pos;
	const ti = new Int32Array( T ), tj = new Int32Array( T );
	let t = 0, dMin = Infinity, dMax = 0;
	for ( let i = 0; i < K; i ++ ) for ( let j = i + 1; j < K; j ++ ) {

		ti[ t ] = i; tj[ t ] = j; t ++;
		const d = dist[ i * K + j ];
		if ( d < dMin ) dMin = d;
		if ( d > dMax ) dMax = d;

	}

	const etaMax = dMax * dMax, etaMin = eps * dMin * dMin;
	const lambda = Math.log( etaMax / etaMin ) / Math.max( 1, iters - 1 );
	const order = new Int32Array( T );
	for ( let k = 0; k < T; k ++ ) order[ k ] = k;
	for ( let it = 0; it < iters; it ++ ) {

		const eta = etaMax * Math.exp( - lambda * it );
		for ( let k = T - 1; k > 0; k -- ) {

			const r = Math.floor( rand() * ( k + 1 ) );
			const q = order[ k ]; order[ k ] = order[ r ]; order[ r ] = q;

		}

		for ( let k = 0; k < T; k ++ ) {

			const q = order[ k ];
			const a = ti[ q ], b = tj[ q ], i = 3 * a, j = 3 * b;
			const d = dist[ a * K + b ];
			const dx = pos[ i ] - pos[ j ], dy = pos[ i + 1 ] - pos[ j + 1 ], dz = pos[ i + 2 ] - pos[ j + 2 ];
			const mag = Math.sqrt( dx * dx + dy * dy + dz * dz ) + 1e-12;
			let mu = eta / ( d * d );
			if ( mu > 1 ) mu = 1;
			const r = ( mu * ( mag - d ) ) / ( 2 * mag );
			pos[ i ] -= r * dx; pos[ i + 1 ] -= r * dy; pos[ i + 2 ] -= r * dz;
			pos[ j ] += r * dx; pos[ j + 1 ] += r * dy; pos[ j + 2 ] += r * dz;

		}

	}

	return pos;

}

/** All-pairs shortest paths on a small weighted graph (CSR), Dijkstra with a binary heap. */
function allPairs( K, start, nbr, len ) {

	const dist = new Float32Array( K * K ).fill( Infinity );
	const heapN = new Int32Array( 4 * nbr.length + K + 16 ), heapD = new Float64Array( heapN.length );
	for ( let s = 0; s < K; s ++ ) {

		const d = dist.subarray( s * K, ( s + 1 ) * K );
		let size = 0;
		d[ s ] = 0;
		heapN[ size ] = s; heapD[ size ++ ] = 0;
		while ( size > 0 ) {

			// pop the minimum
			const u = heapN[ 0 ], du = heapD[ 0 ];
			size --;
			if ( size > 0 ) {

				const ln = heapN[ size ], lv = heapD[ size ];
				let i = 0;
				for ( ;; ) {

					let c = 2 * i + 1;
					if ( c >= size ) break;
					if ( c + 1 < size && heapD[ c + 1 ] < heapD[ c ] ) c ++;
					if ( heapD[ c ] >= lv ) break;
					heapN[ i ] = heapN[ c ]; heapD[ i ] = heapD[ c ];
					i = c;

				}

				heapN[ i ] = ln; heapD[ i ] = lv;

			}

			if ( du > d[ u ] ) continue;
			for ( let k = start[ u ]; k < start[ u + 1 ]; k ++ ) {

				const v = nbr[ k ], nd = du + len[ k ];
				if ( nd < d[ v ] ) {

					d[ v ] = nd;
					// push
					let i = size ++;
					while ( i > 0 ) {

						const p = ( i - 1 ) >> 1;
						if ( heapD[ p ] <= nd ) break;
						heapN[ i ] = heapN[ p ]; heapD[ i ] = heapD[ p ];
						i = p;

					}

					heapN[ i ] = v; heapD[ i ] = nd;

				}

			}

		}

	}

	return dist;

}

/** Finest hierarchy level with at most maxClusters clusters (−1: none). */
export function pickLevel( g, maxClusters = 400 ) {

	for ( let l = 0; l < g.D; l ++ ) if ( g.levelOffset[ l + 1 ] - g.levelOffset[ l ] <= maxClusters ) return l;
	return - 1;

}

/**
 * Stress layout of the quotient graph of hierarchy level `lvl` (clusters as balls).
 * `separation` scales the quotient edge lengths (1 = touching balls, > 1 leaves gaps).
 * @returns {{ K, off, cpos: Float64Array, rad: Float64Array, quotientEdges }} K clusters with ids
 *   off..off+K−1, centres (3K, mass-weighted centroid at the origin) and ball radii.
 */
export function quotientStress( g, lvl, { fill = 0.9, seed = 1, iters = 30, separation = 1, relax = 30, overlap = 0.8 } = {} ) {

	const { N, D, E, edges, anc, levelOffset, clusterMass } = g;
	const R = g.layoutRadius;
	const off = levelOffset[ lvl ], K = levelOffset[ lvl + 1 ] - off;
	const pairs = new Map();
	for ( let e = 0; e < E; e ++ ) {

		const a = anc[ edges[ 2 * e ] * D + lvl ] - off, b = anc[ edges[ 2 * e + 1 ] * D + lvl ] - off;
		if ( a === b ) continue;
		const key = a < b ? a * K + b : b * K + a;
		pairs.set( key, ( pairs.get( key ) || 0 ) + 1 );

	}

	const rad = new Float64Array( K );
	for ( let c = 0; c < K; c ++ ) rad[ c ] = R * Math.cbrt( clusterMass[ off + c ] / N ) * fill;
	const start = new Int32Array( K + 1 );
	for ( const key of pairs.keys() ) {

		start[ Math.floor( key / K ) + 1 ] ++; start[ ( key % K ) + 1 ] ++;

	}

	for ( let c = 0; c < K; c ++ ) start[ c + 1 ] += start[ c ];
	const nbr = new Int32Array( 2 * pairs.size ), len = new Float32Array( 2 * pairs.size );
	const fillAt = start.slice();
	for ( const key of pairs.keys() ) {

		const a = Math.floor( key / K ), b = key % K;
		const l = ( rad[ a ] + rad[ b ] ) * separation;
		nbr[ fillAt[ a ] ] = b; len[ fillAt[ a ] ++ ] = l;
		nbr[ fillAt[ b ] ] = a; len[ fillAt[ b ] ++ ] = l;

	}

	const dist = allPairs( K, start, nbr, len );
	let dMax = 0;
	for ( let k = 0; k < dist.length; k ++ ) if ( dist[ k ] !== Infinity && dist[ k ] > dMax ) dMax = dist[ k ];
	// disconnected clusters: one step beyond the largest finite distance
	for ( let a = 0; a < K; a ++ ) for ( let b = 0; b < K; b ++ ) if ( a !== b && dist[ a * K + b ] === Infinity ) dist[ a * K + b ] = dMax + ( rad[ a ] + rad[ b ] ) * separation;
	const cpos = K > 1 ? sgdDense( K, dist, { iters, seed } ) : new Float64Array( 3 );

	// centres the centres (mass weighted) and returns their rms radius
	const centre = () => {

		let cx = 0, cy = 0, cz = 0;
		for ( let c = 0; c < K; c ++ ) {

			const m = clusterMass[ off + c ];
			cx += cpos[ 3 * c ] * m; cy += cpos[ 3 * c + 1 ] * m; cz += cpos[ 3 * c + 2 ] * m;

		}

		cx /= N; cy /= N; cz /= N;
		let q = 0;
		for ( let c = 0; c < K; c ++ ) {

			cpos[ 3 * c ] -= cx; cpos[ 3 * c + 1 ] -= cy; cpos[ 3 * c + 2 ] -= cz;
			q += clusterMass[ off + c ] * ( cpos[ 3 * c ] * cpos[ 3 * c ] + cpos[ 3 * c + 1 ] * cpos[ 3 * c + 1 ] + cpos[ 3 * c + 2 ] * cpos[ 3 * c + 2 ] );

		}

		return Math.sqrt( q / N );

	};

	centre();
	if ( K > 1 ) {

		// Dense quotient graphs (every community touches every other) have near-equal target
		// distances, which no 3D layout can realise: the stress solution then packs the centres far
		// closer than r_a + r_b. Keep its arrangement but (1) rescale the centres so that the node
		// cloud gets the reference extent and (2) push overlapping balls apart.
		// rms² of the nodes = rms² of the centres + Σ m_c·(3/5)·r_c² / N
		let inner = 0;
		for ( let c = 0; c < K; c ++ ) inner += clusterMass[ off + c ] * 0.6 * rad[ c ] * rad[ c ];
		inner /= N;
		const target = Math.sqrt( Math.max( 0.05 * R * R, 0.6 * R * R * separation * separation - inner ) );
		const rescale = () => {

			const rms = centre();
			if ( rms > 1e-9 ) for ( let k = 0; k < 3 * K; k ++ ) cpos[ k ] *= target / rms;

		};

		rescale();
		for ( let it = 0; it < relax; it ++ ) {

			let moved = 0;
			for ( let a = 0; a < K; a ++ ) for ( let b = a + 1; b < K; b ++ ) {

				let dx = cpos[ 3 * b ] - cpos[ 3 * a ], dy = cpos[ 3 * b + 1 ] - cpos[ 3 * a + 1 ], dz = cpos[ 3 * b + 2 ] - cpos[ 3 * a + 2 ];
				let l = Math.sqrt( dx * dx + dy * dy + dz * dz );
				const want = ( rad[ a ] + rad[ b ] ) * overlap;
				if ( l >= want ) continue;
				if ( l < 1e-9 ) {

					dx = 1e-3 * ( ( a * 7 + b * 3 ) % 5 - 2 ); dy = 1e-3 * ( ( a * 5 + b * 11 ) % 7 - 3 ); dz = 1e-3;
					l = Math.sqrt( dx * dx + dy * dy + dz * dz );

				}

				const ma = clusterMass[ off + a ], mb = clusterMass[ off + b ];
				const push = ( 0.5 * ( want - l ) ) / l;
				const fa = mb / ( ma + mb ), fb = ma / ( ma + mb );
				cpos[ 3 * a ] -= dx * push * fa; cpos[ 3 * a + 1 ] -= dy * push * fa; cpos[ 3 * a + 2 ] -= dz * push * fa;
				cpos[ 3 * b ] += dx * push * fb; cpos[ 3 * b + 1 ] += dy * push * fb; cpos[ 3 * b + 2 ] += dz * push * fb;
				moved ++;

			}

			if ( ! moved ) break;

		}

		rescale();

	}

	return { K, off, cpos, rad, quotientEdges: pairs.size };

}

/** Connectivity-aware initial positions (see the header). Float32Array( 4N ) or null. */
export function stressSeed( g, { maxClusters = 400, fill = 0.9, seed = 1, iters = 30 } = {} ) {

	const { N, D } = g;
	if ( ! D || ! N ) return null;
	const lvl = pickLevel( g, maxClusters );
	if ( lvl < 0 ) return null;
	const rand = rng( seed * 7919 + 13 );
	const { K, off, cpos, rad } = quotientStress( g, lvl, { fill, seed, iters } );

	// seed the subtrees: sub-clusters on a Fibonacci sphere, nodes uniform in their cluster's ball
	const pos = new Float32Array( 4 * N );
	const dir = [ 0, 0, 0 ];
	// rotate each direction randomly so the Fibonacci pattern does not align across levels
	const randomRotate = ( v ) => {

		const a = rand() * Math.PI * 2, b = Math.acos( 2 * rand() - 1 );
		const ca = Math.cos( a ), sa = Math.sin( a ), cb = Math.cos( b ), sb = Math.sin( b );
		const x = v[ 0 ] * ca - v[ 2 ] * sa, z = v[ 0 ] * sa + v[ 2 ] * ca;
		const y = v[ 1 ] * cb - z * sb, z2 = v[ 1 ] * sb + z * cb;
		v[ 0 ] = x; v[ 1 ] = y; v[ 2 ] = z2;

	};

	const { clusterChildStart, clusterChildCount, childRefs, clusterMass } = g;
	const stack = [];
	for ( let c = 0; c < K; c ++ ) stack.push( [ N + off + c, cpos[ 3 * c ], cpos[ 3 * c + 1 ], cpos[ 3 * c + 2 ], rad[ c ] ] );
	while ( stack.length ) {

		const [ ref, x, y, z, r ] = stack.pop();
		if ( ref < N ) {

			pos[ 4 * ref ] = x; pos[ 4 * ref + 1 ] = y; pos[ 4 * ref + 2 ] = z;
			continue;

		}

		const c = ref - N;
		const m = clusterChildCount[ c ], mass = clusterMass[ c ] || 1;
		for ( let k = 0; k < m; k ++ ) {

			const child = childRefs[ clusterChildStart[ c ] + k ];
			if ( child < N ) {

				const u = Math.cbrt( rand() ) * r;
				fibonacci( Math.floor( rand() * 4096 ), 4096, dir );
				randomRotate( dir );
				stack.push( [ child, x + dir[ 0 ] * u, y + dir[ 1 ] * u, z + dir[ 2 ] * u, 0 ] );

			} else {

				const cr = r * Math.cbrt( clusterMass[ child - N ] / mass ) * 0.9;
				const spread = m === 1 ? 0 : r - cr * 0.6;
				fibonacci( k, m, dir );
				stack.push( [ child, x + dir[ 0 ] * spread, y + dir[ 1 ] * spread, z + dir[ 2 ] * spread, cr ] );

			}

		}

	}

	return pos;

}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Stress layout (sparse SGD²)
// ─────────────────────────────────────────────────────────────────────────────────────────────

function bfsInto( g, src, out, queue ) {

	const { adjStart, degree, adjList } = g;
	out.fill( UNREACHED );
	let head = 0, tail = 0;
	queue[ tail ++ ] = src;
	out[ src ] = 0;
	while ( head < tail ) {

		const u = queue[ head ++ ], du = out[ u ] + 1;
		for ( let k = adjStart[ u ], e = k + degree[ u ]; k < e; k ++ ) {

			const v = adjList[ k ];
			if ( out[ v ] === UNREACHED ) {

				out[ v ] = du;
				queue[ tail ++ ] = v;

			}

		}

	}

}

/** Default pivot count of the stress layout: 200 up to 2000 nodes, 50 above. */
export function autoPivots( N ) {

	return N <= 2000 ? 200 : 50;

}

/** Sparse stress layout (see the header). */
export function sgd2Layout( g, { dim = 3, pivots = 'auto', iters = 30, eps = 0.1, seed = 1, init = null, radius = undefined } = {} ) {

	const t0 = now();
	const N = g.N, E = g.E;
	if ( N === 0 ) return { positions: new Float32Array( 0 ), info: { pivots: 0, terms: 0, exact: true, scale: 1, ms: 0 } };
	const rand = rng( seed );
	const want = pivots === 'auto' || pivots === undefined || pivots === null ? autoPivots( N ) : pivots;
	const P = want <= 0 || want >= N ? N : want | 0;
	const exact = P === N;
	const queue = new Int32Array( N );

	// ── pivots (max–min / farthest first) and BFS distances ──
	const piv = new Int32Array( P );
	const dist = new Uint16Array( P * N );
	const minDist = new Uint16Array( N ).fill( UNREACHED );
	const region = new Int32Array( N ).fill( - 1 ); // closest pivot (index into piv)
	if ( exact ) {

		for ( let p = 0; p < N; p ++ ) {

			piv[ p ] = p;
			bfsInto( g, p, dist.subarray( p * N, ( p + 1 ) * N ), queue );

		}

	} else {

		// start from the highest-degree node
		let next = 0;
		for ( let i = 1; i < N; i ++ ) if ( g.degree[ i ] > g.degree[ next ] ) next = i;
		for ( let p = 0; p < P; p ++ ) {

			piv[ p ] = next;
			const d = dist.subarray( p * N, ( p + 1 ) * N );
			bfsInto( g, next, d, queue );
			let best = - 1, bestD = - 1;
			for ( let i = 0; i < N; i ++ ) {

				if ( d[ i ] < minDist[ i ] ) {

					minDist[ i ] = d[ i ]; region[ i ] = p;

				}

				// unreachable nodes (65535) are picked first, so every component gets a pivot
				if ( minDist[ i ] > bestD ) {

					bestD = minDist[ i ]; best = i;

				}

			}

			next = best;

		}

	}

	let maxFinite = 1;
	for ( let k = 0; k < dist.length; k ++ ) if ( dist[ k ] !== UNREACHED && dist[ k ] > maxFinite ) maxFinite = dist[ k ];
	const far = maxFinite + 1; // distance of unreachable pairs

	// ── terms ──
	const isPivot = new Int32Array( N ).fill( - 1 );
	for ( let p = 0; p < P; p ++ ) isPivot[ piv[ p ] ] = p;
	let T = 0;
	const cap = exact ? ( N * ( N - 1 ) ) / 2 : E + P * N;
	const ti = new Int32Array( cap ), tj = new Int32Array( cap );
	const td = new Float32Array( cap ), wij = new Float32Array( cap ), wji = new Float32Array( cap );
	let wMin = Infinity, wMax = 0;
	const push = ( i, j, d, a, b ) => {

		ti[ T ] = i; tj[ T ] = j; td[ T ] = d; wij[ T ] = a; wji[ T ] = b; T ++;
		if ( a > 0 ) {

			if ( a < wMin ) wMin = a;
			if ( a > wMax ) wMax = a;

		}

		if ( b > 0 ) {

			if ( b < wMin ) wMin = b;
			if ( b > wMax ) wMax = b;

		}

	};

	if ( exact ) {

		for ( let i = 0; i < N; i ++ ) for ( let j = i + 1; j < N; j ++ ) {

			let d = dist[ i * N + j ];
			if ( d === UNREACHED ) d = far;
			const w = 1 / ( d * d );
			push( i, j, d, w, w );

		}

	} else {

		for ( let e = 0; e < E; e ++ ) push( g.edges[ 2 * e ], g.edges[ 2 * e + 1 ], 1, 1, 1 );
		// region members (CSR, ascending node index)
		const mStart = new Int32Array( P + 1 );
		for ( let i = 0; i < N; i ++ ) if ( region[ i ] >= 0 ) mStart[ region[ i ] + 1 ] ++;
		for ( let p = 0; p < P; p ++ ) mStart[ p + 1 ] += mStart[ p ];
		const members = new Int32Array( Math.max( 1, mStart[ P ] ) );
		{

			const fillAt = mStart.slice( 0, P );
			for ( let i = 0; i < N; i ++ ) if ( region[ i ] >= 0 ) members[ fillAt[ region[ i ] ] ++ ] = i;

		}

		// s_ip from per-pivot cumulative histograms of the region members' distances
		const hist = new Int32Array( far + 2 );
		const neighbourMark = new Int32Array( N ).fill( - 1 );
		for ( let p = 0; p < P; p ++ ) {

			const d = dist.subarray( p * N, ( p + 1 ) * N );
			hist.fill( 0 );
			for ( let m = mStart[ p ]; m < mStart[ p + 1 ]; m ++ ) hist[ Math.min( far, d[ members[ m ] ] ) ] ++;
			for ( let k = 1; k < hist.length; k ++ ) hist[ k ] += hist[ k - 1 ];
			const pn = piv[ p ];
			for ( let k = g.adjStart[ pn ], e = k + g.degree[ pn ]; k < e; k ++ ) neighbourMark[ g.adjList[ k ] ] = p;
			for ( let i = 0; i < N; i ++ ) {

				if ( i === pn || neighbourMark[ i ] === p ) continue; // edges are terms already
				const q = isPivot[ i ];
				if ( q >= 0 && q < p ) continue; // pivot–pivot pairs once (from the lower index)
				let dd = d[ i ];
				if ( dd === UNREACHED ) dd = far;
				const s = Math.max( 1, hist[ Math.floor( dd / 2 ) ] );
				const w = s / ( dd * dd );
				let wr = 0;
				if ( q >= 0 ) {

					// reverse weight: nodes of i's region within dd/2 of i
					let c = 0;
					const dq = dist.subarray( q * N, ( q + 1 ) * N );
					for ( let m = mStart[ q ]; m < mStart[ q + 1 ]; m ++ ) if ( dq[ members[ m ] ] <= dd / 2 ) c ++;
					wr = Math.max( 1, c ) / ( dd * dd );

				}

				push( i, pn, dd, w, wr );

			}

		}

	}

	// ── positions ──
	const pos = new Float64Array( 3 * N );
	if ( init ) {

		const stride = init.length === 4 * N ? 4 : 3;
		for ( let i = 0; i < N; i ++ ) for ( let a = 0; a < dim; a ++ ) pos[ 3 * i + a ] = init[ stride * i + a ];

	} else for ( let i = 0; i < N; i ++ ) for ( let a = 0; a < dim; a ++ ) pos[ 3 * i + a ] = rand();

	// ── SGD ──
	if ( T > 0 && wMax > 0 ) {

		const etaMax = 1 / wMin, etaMin = eps / wMax;
		const lambda = Math.log( etaMax / etaMin ) / Math.max( 1, iters - 1 );
		const order = new Int32Array( T );
		for ( let k = 0; k < T; k ++ ) order[ k ] = k;
		for ( let it = 0; it < iters; it ++ ) {

			const eta = etaMax * Math.exp( - lambda * it );
			for ( let k = T - 1; k > 0; k -- ) {

				const r = Math.floor( rand() * ( k + 1 ) );
				const t = order[ k ]; order[ k ] = order[ r ]; order[ r ] = t;

			}

			for ( let k = 0; k < T; k ++ ) {

				const t = order[ k ];
				const i = 3 * ti[ t ], j = 3 * tj[ t ];
				const dx = pos[ i ] - pos[ j ], dy = pos[ i + 1 ] - pos[ j + 1 ], dz = pos[ i + 2 ] - pos[ j + 2 ];
				const mag = Math.sqrt( dx * dx + dy * dy + dz * dz ) + 1e-12;
				const r = ( mag - td[ t ] ) / ( 2 * mag );
				let mi = wij[ t ] * eta, mj = wji[ t ] * eta;
				if ( mi > 1 ) mi = 1;
				if ( mj > 1 ) mj = 1;
				pos[ i ] -= mi * r * dx; pos[ i + 1 ] -= mi * r * dy; pos[ i + 2 ] -= mi * r * dz;
				pos[ j ] += mj * r * dx; pos[ j + 1 ] += mj * r * dy; pos[ j + 2 ] += mj * r * dz;

			}

		}

	}

	const R = radius === undefined ? g.layoutRadius : radius;
	const scale = R ? fitToRadius( pos, N, R ) : 1;
	return { positions: toPositions4( pos, N ), info: { pivots: P, terms: T, exact, scale, ms: now() - t0 } };

}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Radial focus layout
// ─────────────────────────────────────────────────────────────────────────────────────────────

/** BFS shells around `root` (see the header). */
export function radialLayout( g, root = 0, { shellGap = null, dim = 3, pack = 0.9, minSpacing = null, radius = undefined } = {} ) {

	const t0 = now();
	const { N, adjStart, degree, adjList } = g;
	const depth = new Int32Array( N ).fill( - 1 );
	if ( N === 0 ) return { positions: new Float32Array( 0 ), depth, info: { root: - 1, maxDepth: 0, reached: 0, outerRadius: 0, scale: 1, ms: 0 } };
	root = Math.min( N - 1, Math.max( 0, root | 0 ) );
	const L = g.linkDistance ?? 30;
	const gap = shellGap ?? 1.5 * L;
	const spacing = minSpacing ?? 0.35 * L;
	const parent = new Int32Array( N ).fill( - 1 );
	const order = new Int32Array( N );
	let head = 0, tail = 0;
	order[ tail ++ ] = root;
	depth[ root ] = 0;
	while ( head < tail ) {

		const u = order[ head ++ ];
		for ( let k = adjStart[ u ], e = k + degree[ u ]; k < e; k ++ ) {

			const v = adjList[ k ];
			if ( depth[ v ] < 0 ) {

				depth[ v ] = depth[ u ] + 1;
				parent[ v ] = u;
				order[ tail ++ ] = v;

			} else if ( depth[ v ] === depth[ u ] + 1 && degree[ u ] > degree[ parent[ v ] ] ) parent[ v ] = u;

		}

	}

	const reached = tail;
	const maxDepth = depth[ order[ reached - 1 ] ]; // BFS order is by depth

	// subtree sizes (reverse BFS order) and child lists (CSR)
	const size = new Float64Array( N ).fill( 1 );
	const childStart = new Int32Array( N + 1 );
	for ( let k = reached - 1; k > 0; k -- ) {

		const v = order[ k ];
		size[ parent[ v ] ] += size[ v ];
		childStart[ parent[ v ] + 1 ] ++;

	}

	for ( let i = 0; i < N; i ++ ) childStart[ i + 1 ] += childStart[ i ];
	const fillAt = childStart.slice( 0, N );
	const children = new Int32Array( Math.max( 1, reached - 1 ) );
	for ( let k = 1; k < reached; k ++ ) children[ fillAt[ parent[ order[ k ] ] ] ++ ] = order[ k ];

	// shell radii: at least `gap` apart, and wide enough for the shell's population
	const shellCount = new Int32Array( maxDepth + 2 );
	for ( let k = 0; k < reached; k ++ ) shellCount[ depth[ order[ k ] ] ] ++;
	const shellRadius = new Float64Array( maxDepth + 2 );
	for ( let d = 1; d <= maxDepth + 1; d ++ ) {

		const n = d <= maxDepth ? shellCount[ d ] : N - reached;
		// area per node >= spacing² (sphere: 4πr²), or arc per node >= spacing (circle: 2πr)
		const need = dim === 3 ? Math.sqrt( ( n * spacing * spacing ) / ( 4 * Math.PI ) ) : ( n * spacing ) / ( 2 * Math.PI );
		shellRadius[ d ] = Math.max( shellRadius[ d - 1 ] + gap, need );

	}

	const outer = shellRadius[ reached < N ? maxDepth + 1 : maxDepth ];
	const R = radius === undefined ? g.layoutRadius : radius;
	const scale = R && outer > 0 ? R / outer : 1;

	const pos = new Float32Array( 4 * N );
	// per node: direction (unit vector) and cap half-angle (3D) or sector half-width (2D)
	const dir = new Float64Array( 3 * N ), cap = new Float64Array( N );
	dir[ 3 * root + 1 ] = 1;
	cap[ root ] = Math.PI;
	const bySize = ( a, b ) => size[ b ] - size[ a ] || a - b;
	for ( let k = 0; k < reached; k ++ ) {

		const u = order[ k ];
		const s = childStart[ u ], e = childStart[ u + 1 ];
		const m = e - s;
		if ( m === 0 ) continue;
		const kids = children.subarray( s, e );
		if ( m > 1 ) kids.sort( bySize );
		let total = 0;
		for ( let q = 0; q < m; q ++ ) total += size[ kids[ q ] ];
		const ux = dir[ 3 * u ], uy = dir[ 3 * u + 1 ], uz = dir[ 3 * u + 2 ];
		const theta = cap[ u ];
		const share = u === root ? 1 : pack;
		if ( dim === 2 ) {

			// sector [−theta, theta] around the parent's angle, split proportionally to subtree size
			const a0 = Math.atan2( uy, ux );
			let acc = - theta * share;
			const span = 2 * theta * share;
			for ( let q = 0; q < m; q ++ ) {

				const c = kids[ q ];
				const w = ( span * size[ c ] ) / total;
				const a = a0 + acc + w / 2;
				acc += w;
				dir[ 3 * c ] = Math.cos( a ); dir[ 3 * c + 1 ] = Math.sin( a ); dir[ 3 * c + 2 ] = 0;
				cap[ c ] = w / 2;

			}

			continue;

		}

		// orthonormal frame (b, c) around u's direction
		let ax = 0, ay = 0;
		if ( Math.abs( ux ) < 0.9 ) ax = 1; else ay = 1;
		let bx = - uz * ay, by = uz * ax, bz = ux * ay - uy * ax;
		const bl = Math.hypot( bx, by, bz );
		bx /= bl; by /= bl; bz /= bl;
		const cx = uy * bz - uz * by, cy = uz * bx - ux * bz, cz = ux * by - uy * bx;
		const omega = 2 * Math.PI * ( 1 - Math.cos( theta ) ) * share; // solid angle to share
		const cosT = 1 - omega / ( 2 * Math.PI );
		let acc = 0;
		for ( let q = 0; q < m; q ++ ) {

			const c = kids[ q ];
			const w = size[ c ] / total;
			// area-proportional position on the cap: cumulative solid angle at the middle of the share
			const f = acc + w / 2;
			acc += w;
			const cosP = 1 - f * ( 1 - cosT );
			const sinP = Math.sqrt( Math.max( 0, 1 - cosP * cosP ) );
			const az = q * GOLDEN + 0.37 * u;
			const sx = Math.cos( az ) * sinP, sy = Math.sin( az ) * sinP;
			dir[ 3 * c ] = ux * cosP + bx * sx + cx * sy;
			dir[ 3 * c + 1 ] = uy * cosP + by * sx + cy * sy;
			dir[ 3 * c + 2 ] = uz * cosP + bz * sx + cz * sy;
			cap[ c ] = Math.acos( Math.max( - 1, 1 - ( omega * w ) / ( 2 * Math.PI ) ) );

		}

	}

	for ( let k = 0; k < reached; k ++ ) {

		const u = order[ k ], r = shellRadius[ depth[ u ] ] * scale;
		pos[ 4 * u ] = dir[ 3 * u ] * r; pos[ 4 * u + 1 ] = dir[ 3 * u + 1 ] * r; pos[ 4 * u + 2 ] = dir[ 3 * u + 2 ] * r;

	}

	// unreachable nodes: outer shell
	if ( reached < N ) {

		const r = shellRadius[ maxDepth + 1 ] * scale, m = N - reached;
		let q = 0;
		for ( let i = 0; i < N; i ++ ) {

			if ( depth[ i ] >= 0 ) continue;
			if ( dim === 2 ) {

				const a = ( 2 * Math.PI * q ) / m;
				pos[ 4 * i ] = Math.cos( a ) * r; pos[ 4 * i + 1 ] = Math.sin( a ) * r;

			} else {

				const y = 1 - ( 2 * ( q + 0.5 ) ) / m, rr = Math.sqrt( Math.max( 0, 1 - y * y ) ), phi = q * GOLDEN;
				pos[ 4 * i ] = Math.cos( phi ) * rr * r; pos[ 4 * i + 1 ] = y * r; pos[ 4 * i + 2 ] = Math.sin( phi ) * rr * r;

			}

			q ++;

		}

	}

	return { positions: pos, depth, info: { root, maxDepth, reached, outerRadius: outer * scale, scale, ms: now() - t0 } };

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}
