// Louvain community detection (Blondel et al., J. Stat. Mech. 2008) producing a multi-level
// dendrogram like python-louvain's generate_dendrogram + partition_at_level, which is what
// ImsvGraphVis used to build its clustering hierarchy. Pure JS, typed arrays, no dependencies
// besides the seeded PRNG: runs in the main thread, a worker, or Node.
//
// ─── Algorithm ──────────────────────────────────────────────────────────────────────────────
//  Level loop: local moving on the current (induced) graph, renumber communities densely,
//  aggregate communities into super-nodes (internal weight becomes a self loop), repeat while
//  modularity improves by more than minGain (python-louvain's __MIN = 1e-7). The first level is
//  always kept, as in python-louvain.
//  Local moving: passes over all nodes in a seeded random order (reshuffled every pass); a node
//  leaves its community and joins the neighbouring community with the best strictly positive
//  gain  k_i,in(C) − γ · tot(C) · k_i / 2m  over staying. Passes stop when a pass moves nothing
//  or improves modularity by less than minGain.
//  Modularity: Q = Σ_c [ in_c / m − γ (tot_c / 2m)² ]  (in_c = internal edge weight counted once,
//  tot_c = sum of degrees, γ = resolution). Equal to networkx/python-louvain for γ = 1.
//
// ─── Output ─────────────────────────────────────────────────────────────────────────────────
//  levels[l][i] = community of original node i at level l, dense 0..K_l−1, levels[0] finest,
//  every level nested in the next (the canonical hierarchy of src/data/format.js). modularity[l]
//  is the modularity of levels[l] on the input graph.
//
// ─── Refinement (options.refine = { maxLeafSize }) ──────────────────────────────────────────
//  Finest communities larger than maxLeafSize are split by running Louvain on their induced
//  subgraph (taking the coarsest sub-level whose parts all fit; if none does, the coarsest real
//  split, recursing into parts that are still too large; as a last resort, e.g. for cliques that
//  Louvain will not split, cutting BFS-ordered chunks).
//  The result is inserted as a new, finer level 0; communities that were small enough become a
//  single sub-cluster, so the hierarchy keeps uniform depth. Large leaf clusters hurt both the
//  bundled-edge look (one huge star per cluster) and the hierarchy-based Barnes–Hut layout.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//  louvain( { nodeCount, edges, weights? }, options ) -> { levels, modularity, levelSizes, refined, passes, ms }
//      options: seed (1), resolution (1), minGain (1e-7), maxLevels (32), maxPasses (0 = no cap),
//               refine ({ maxLeafSize, maxDepth = 6 } or null), onProgress({ phase, level, pass,
//               communities, modularity })
//  modularity( { nodeCount, edges, weights? }, partition, resolution = 1 ) -> Q
//  louvainHierarchy( g, options ) -> { method, levels, modularity, resolution, seed, refined, maxLeafSize, dropped,
//                                      communities (size of the final Louvain partition) }
//      canonical hierarchy for a canonical graph; levels pass format.js normalizeLevels unchanged.
//      options.maxLeafSize defaults to DEFAULT_MAX_LEAF_SIZE (200; 0 disables refinement).
//      options.minLevelRatio (DEFAULT_MIN_LEVEL_RATIO = 1.2; 1 keeps every level): a level that
//      does not reduce the cluster count by this factor (Louvain's last levels often merge a
//      handful of clusters) adds bundling control points and Barnes–Hut depth without adding
//      structure, so it replaces the level below it (never the finest level, which defines the
//      leaves). The top level stays the final Louvain partition.
//  ensureHierarchy( g, options ) -> g itself when it already has a hierarchy (unless options.force),
//      otherwise a shallow copy with `hierarchy` set and meta.hierarchy = { method, modularity, … }.

import { rng } from '../graph/prepare.js';

export const DEFAULT_MAX_LEAF_SIZE = 200;
export const DEFAULT_MIN_LEVEL_RATIO = 1.2;
const MIN_GAIN = 1e-7;

// ── weighted CSR graph (self loops kept separately) ─────────────────────────────────────────

function csrFromEdges( n, edges, weights ) {

	const E = edges.length >> 1;
	const count = new Int32Array( n + 1 );
	const loop = new Float64Array( n );
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		if ( a !== b ) {

			count[ a + 1 ] ++;
			count[ b + 1 ] ++;

		}

	}

	for ( let i = 0; i < n; i ++ ) count[ i + 1 ] += count[ i ];
	const start = count;
	const adj = new Int32Array( start[ n ] );
	const w = new Float64Array( start[ n ] );
	const fill = start.slice( 0, n );
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		const we = weights ? weights[ e ] : 1;
		if ( a === b ) {

			loop[ a ] += we;
			continue;

		}

		adj[ fill[ a ] ] = b; w[ fill[ a ] ++ ] = we;
		adj[ fill[ b ] ] = a; w[ fill[ b ] ++ ] = we;

	}

	return finishCsr( n, start, adj, w, loop );

}

function finishCsr( n, start, adj, w, loop ) {

	const k = new Float64Array( n );
	let m2 = 0;
	for ( let i = 0; i < n; i ++ ) {

		let s = 2 * loop[ i ];
		for ( let e = start[ i ]; e < start[ i + 1 ]; e ++ ) s += w[ e ];
		k[ i ] = s;
		m2 += s;

	}

	return { n, start, adj, w, loop, k, m2 };

}

/** Aggregates communities (dense ids 0..K-1) of g into a new CSR graph. */
function induce( g, comm, K ) {

	const { n, start, adj, w, loop } = g;
	const cstart = new Int32Array( K + 1 );
	for ( let i = 0; i < n; i ++ ) cstart[ comm[ i ] + 1 ] ++;
	for ( let c = 0; c < K; c ++ ) cstart[ c + 1 ] += cstart[ c ];
	const members = new Int32Array( n );
	{

		const fill = cstart.slice( 0, K );
		for ( let i = 0; i < n; i ++ ) members[ fill[ comm[ i ] ] ++ ] = i;

	}

	const nstart = new Int32Array( K + 1 );
	const nadj = new Int32Array( adj.length );
	const nw = new Float64Array( adj.length );
	const nloop = new Float64Array( K );
	const acc = new Float64Array( K ).fill( - 1 );
	const touched = new Int32Array( K );
	let pos = 0;
	for ( let c = 0; c < K; c ++ ) {

		let cnt = 0, internal = 0, loops = 0;
		for ( let q = cstart[ c ]; q < cstart[ c + 1 ]; q ++ ) {

			const i = members[ q ];
			loops += loop[ i ];
			for ( let e = start[ i ]; e < start[ i + 1 ]; e ++ ) {

				const cj = comm[ adj[ e ] ];
				if ( cj === c ) {

					internal += w[ e ];
					continue;

				}

				if ( acc[ cj ] < 0 ) {

					acc[ cj ] = 0;
					touched[ cnt ++ ] = cj;

				}

				acc[ cj ] += w[ e ];

			}

		}

		nloop[ c ] = loops + internal / 2; // every internal edge was seen from both ends
		for ( let q = 0; q < cnt; q ++ ) {

			const cj = touched[ q ];
			nadj[ pos ] = cj;
			nw[ pos ++ ] = acc[ cj ];
			acc[ cj ] = - 1;

		}

		nstart[ c + 1 ] = pos;

	}

	return finishCsr( K, nstart, nadj.subarray( 0, pos ), nw.subarray( 0, pos ), nloop );

}

/** Induced subgraph of g on `members` (local ids = position in members), as CSR. */
function subgraph( g, members, localOf ) {

	const { start, adj, w } = g;
	const m = members.length;
	const nstart = new Int32Array( m + 1 );
	let total = 0;
	for ( let q = 0; q < m; q ++ ) {

		const i = members[ q ];
		for ( let e = start[ i ]; e < start[ i + 1 ]; e ++ ) if ( localOf[ adj[ e ] ] >= 0 ) total ++;
		nstart[ q + 1 ] = total;

	}

	const nadj = new Int32Array( total );
	const nw = new Float64Array( total );
	const nloop = new Float64Array( m );
	let pos = 0;
	for ( let q = 0; q < m; q ++ ) {

		const i = members[ q ];
		nloop[ q ] = g.loop[ i ];
		for ( let e = start[ i ]; e < start[ i + 1 ]; e ++ ) {

			const l = localOf[ adj[ e ] ];
			if ( l >= 0 ) {

				nadj[ pos ] = l;
				nw[ pos ++ ] = w[ e ];

			}

		}

	}

	return finishCsr( m, nstart, nadj, nw, nloop );

}

// ── local moving ─────────────────────────────────────────────────────────────────────────────

function shuffle( arr, rand ) {

	for ( let i = arr.length - 1; i > 0; i -- ) {

		const j = Math.floor( rand() * ( i + 1 ) );
		const t = arr[ i ]; arr[ i ] = arr[ j ]; arr[ j ] = t;

	}

}

function stateModularity( n, tot, inner, m2, resolution ) {

	if ( m2 <= 0 ) return 0;
	const m = m2 / 2;
	let q = 0;
	for ( let c = 0; c < n; c ++ ) {

		if ( tot[ c ] > 0 ) {

			const t = tot[ c ] / m2;
			q += inner[ c ] / m - resolution * t * t;

		}

	}

	return q;

}

function oneLevel( g, comm, resolution, rand, minGain, maxPasses, scratch, report ) {

	const { n, start, adj, w, loop, k, m2 } = g;
	const tot = new Float64Array( n );
	const inner = new Float64Array( n );
	for ( let i = 0; i < n; i ++ ) {

		comm[ i ] = i;
		tot[ i ] = k[ i ];
		inner[ i ] = loop[ i ];

	}

	const neighW = scratch.neighW.subarray( 0, n ).fill( - 1 );
	const neighCom = scratch.neighCom;
	const order = scratch.order.subarray( 0, n );
	for ( let i = 0; i < n; i ++ ) order[ i ] = i;

	let mod = stateModularity( n, tot, inner, m2, resolution );
	let passes = 0;
	if ( m2 <= 0 ) return { modularity: mod, passes };
	for ( ;; ) {

		passes ++;
		shuffle( order, rand );
		let moved = 0;
		for ( let t = 0; t < n; t ++ ) {

			const i = order[ t ];
			const ci = comm[ i ];
			const ki = k[ i ];
			let cnt = 1;
			neighCom[ 0 ] = ci;
			neighW[ ci ] = 0;
			for ( let e = start[ i ]; e < start[ i + 1 ]; e ++ ) {

				const cj = comm[ adj[ e ] ];
				if ( neighW[ cj ] < 0 ) {

					neighW[ cj ] = 0;
					neighCom[ cnt ++ ] = cj;

				}

				neighW[ cj ] += w[ e ];

			}

			// remove i from its community, then pick the best one (staying wins ties)
			tot[ ci ] -= ki;
			inner[ ci ] -= neighW[ ci ] + loop[ i ];
			const f = ( resolution * ki ) / m2;
			let best = ci;
			let bestGain = neighW[ ci ] - tot[ ci ] * f;
			for ( let q = 1; q < cnt; q ++ ) {

				const c = neighCom[ q ];
				const gain = neighW[ c ] - tot[ c ] * f;
				if ( gain > bestGain ) {

					bestGain = gain;
					best = c;

				}

			}

			tot[ best ] += ki;
			inner[ best ] += neighW[ best ] + loop[ i ];
			comm[ i ] = best;
			if ( best !== ci ) moved ++;
			for ( let q = 0; q < cnt; q ++ ) neighW[ neighCom[ q ] ] = - 1;

		}

		const newMod = stateModularity( n, tot, inner, m2, resolution );
		report?.( passes, newMod );
		const gain = newMod - mod;
		mod = newMod;
		if ( moved === 0 || gain < minGain || passes === maxPasses ) break;

	}

	return { modularity: mod, passes };

}

/** Renumbers comm in place to dense ids by first appearance; returns the community count. */
function renumber( comm ) {

	const n = comm.length;
	const map = new Int32Array( n ).fill( - 1 );
	let K = 0;
	for ( let i = 0; i < n; i ++ ) {

		const c = comm[ i ];
		if ( map[ c ] < 0 ) map[ c ] = K ++;
		comm[ i ] = map[ c ];

	}

	return K;

}

/** Core multi-level Louvain on a CSR graph. levels are composed onto the CSR's nodes. */
function louvainCsr( g0, { seed = 1, resolution = 1, minGain = MIN_GAIN, maxLevels = 32, maxPasses = 0, onProgress = null } = {} ) {

	const n0 = g0.n;
	const rand = rng( seed );
	const scratch = { neighW: new Float64Array( n0 ), neighCom: new Int32Array( n0 ), order: new Int32Array( n0 ) };
	const levels = [];
	const modularity = [];
	const levelSizes = [];
	let passesTotal = 0;
	const map = new Int32Array( n0 );
	for ( let i = 0; i < n0; i ++ ) map[ i ] = i;
	let g = g0;
	let mod = - Infinity;
	for ( let level = 0; level < maxLevels && g.n > 0; level ++ ) {

		const comm = new Int32Array( g.n );
		const report = onProgress ? ( pass, q ) => onProgress( { phase: 'louvain', level, pass, communities: g.n, modularity: q } ) : null;
		const res = oneLevel( g, comm, resolution, rand, minGain, maxPasses, scratch, report );
		passesTotal += res.passes;
		if ( level > 0 && res.modularity - mod < minGain ) break;
		const K = renumber( comm );
		for ( let i = 0; i < n0; i ++ ) map[ i ] = comm[ map[ i ] ];
		levels.push( map.slice() );
		modularity.push( res.modularity );
		levelSizes.push( K );
		mod = res.modularity;
		if ( K === g.n || K <= 1 ) break; // nothing merged / everything merged
		g = induce( g, comm, K );

	}

	return { levels, modularity, levelSizes, passes: passesTotal };

}

// ── refinement of oversized leaf communities ──────────────────────────────────────────────────

function partSizes( part, K ) {

	const s = new Int32Array( K );
	for ( let i = 0; i < part.length; i ++ ) s[ part[ i ] ] ++;
	return s;

}

/** Splits the nodes of sub (local CSR) into parts of at most maxSize; returns local part ids. */
function splitCommunity( sub, maxSize, seed, resolution, minGain, depth, maxDepth ) {

	const m = sub.n;
	const out = new Int32Array( m );
	let chosen = null, chosenK = 1;
	if ( depth < maxDepth && sub.m2 > 0 ) {

		const r = louvainCsr( sub, { seed, resolution, minGain } );
		for ( let l = r.levels.length - 1; l >= 0; l -- ) {

			// coarsest level whose parts all fit
			const K = r.levelSizes[ l ];
			if ( K <= 1 ) continue;
			const sizes = partSizes( r.levels[ l ], K );
			let max = 0;
			for ( let c = 0; c < K; c ++ ) if ( sizes[ c ] > max ) max = sizes[ c ];
			if ( max <= maxSize ) {

				chosen = r.levels[ l ]; chosenK = K;
				break;

			}

		}

		// none fits: take the coarsest real split and recurse into the parts that are too large
		// (taking the finest level instead over-fragments graphs without community structure)
		for ( let l = r.levels.length - 1; l >= 0 && ! chosen; l -- ) {

			if ( r.levelSizes[ l ] > 1 ) {

				chosen = r.levels[ l ]; chosenK = r.levelSizes[ l ];

			}

		}

	}

	// no split, or one that leaves the nodes alone (a star at a high resolution): chunks
	if ( ! chosen || 2 * chosenK > m ) return bfsChunks( sub, maxSize );

	// recurse into parts that are still too large
	const sizes = partSizes( chosen, chosenK );
	const members = Array.from( { length: chosenK }, () => [] );
	for ( let i = 0; i < m; i ++ ) members[ chosen[ i ] ].push( i );
	let next = 0;
	const localOf = new Int32Array( m ).fill( - 1 );
	for ( let c = 0; c < chosenK; c ++ ) {

		const mem = members[ c ];
		if ( sizes[ c ] <= maxSize ) {

			for ( const i of mem ) out[ i ] = next;
			next ++;
			continue;

		}

		for ( let q = 0; q < mem.length; q ++ ) localOf[ mem[ q ] ] = q;
		const inner = subgraph( sub, mem, localOf );
		for ( let q = 0; q < mem.length; q ++ ) localOf[ mem[ q ] ] = - 1;
		const parts = splitCommunity( inner, maxSize, ( seed * 31 + c + 1 ) >>> 0, resolution, minGain, depth + 1, maxDepth );
		let K = 0;
		for ( let q = 0; q < mem.length; q ++ ) {

			out[ mem[ q ] ] = next + parts[ q ];
			if ( parts[ q ] + 1 > K ) K = parts[ q ] + 1;

		}

		next += K;

	}

	return out;

}

/** Fallback split: BFS order (connected pieces stay together) cut into near-equal chunks. */
function bfsChunks( sub, maxSize ) {

	const m = sub.n;
	const chunks = Math.ceil( m / maxSize );
	const size = Math.ceil( m / chunks );
	const out = new Int32Array( m );
	const seen = new Uint8Array( m );
	const queue = new Int32Array( m );
	let head = 0, tail = 0, rank = 0;
	for ( let s = 0; s < m; s ++ ) {

		if ( seen[ s ] ) continue;
		seen[ s ] = 1;
		queue[ tail ++ ] = s;
		while ( head < tail ) {

			const i = queue[ head ++ ];
			out[ i ] = Math.floor( rank ++ / size );
			for ( let e = sub.start[ i ]; e < sub.start[ i + 1 ]; e ++ ) {

				const j = sub.adj[ e ];
				if ( ! seen[ j ] ) {

					seen[ j ] = 1;
					queue[ tail ++ ] = j;

				}

			}

		}

	}

	return out;

}

function refineLeaves( g0, leaf, K, { maxLeafSize, maxDepth = 6 }, seed, resolution, minGain, onProgress ) {

	const n = g0.n;
	const sizes = partSizes( leaf, K );
	let oversized = 0;
	for ( let c = 0; c < K; c ++ ) if ( sizes[ c ] > maxLeafSize ) oversized ++;
	if ( ! oversized ) return null;

	const cstart = new Int32Array( K + 1 );
	for ( let c = 0; c < K; c ++ ) cstart[ c + 1 ] = cstart[ c ] + sizes[ c ];
	const members = new Int32Array( n );
	{

		const fill = cstart.slice( 0, K );
		for ( let i = 0; i < n; i ++ ) members[ fill[ leaf[ i ] ] ++ ] = i;

	}

	const fine = new Int32Array( n );
	const localOf = new Int32Array( n ).fill( - 1 );
	let next = 0, done = 0;
	for ( let c = 0; c < K; c ++ ) {

		const mem = members.subarray( cstart[ c ], cstart[ c + 1 ] );
		if ( sizes[ c ] <= maxLeafSize ) {

			for ( let q = 0; q < mem.length; q ++ ) fine[ mem[ q ] ] = next;
			next ++;
			continue;

		}

		for ( let q = 0; q < mem.length; q ++ ) localOf[ mem[ q ] ] = q;
		const sub = subgraph( g0, mem, localOf );
		for ( let q = 0; q < mem.length; q ++ ) localOf[ mem[ q ] ] = - 1;
		const parts = splitCommunity( sub, maxLeafSize, ( seed + Math.imul( c + 1, 0x9e3779b1 ) ) >>> 0, resolution, minGain, 0, maxDepth );
		let k = 0;
		for ( let q = 0; q < mem.length; q ++ ) {

			fine[ mem[ q ] ] = next + parts[ q ];
			if ( parts[ q ] + 1 > k ) k = parts[ q ] + 1;

		}

		next += k;
		done ++;
		onProgress?.( { phase: 'refine', level: 0, pass: done, communities: oversized, modularity: NaN } );

	}

	const KF = renumber( fine );
	return KF > K ? { level: fine, K: KF } : null;

}

// ── public API ───────────────────────────────────────────────────────────────────────────────

/**
 * Multi-level Louvain. Input: { nodeCount, edges (flat pairs), weights? (per edge) }.
 * @returns {{ levels: Int32Array[], modularity: number[], levelSizes: number[], refined: boolean, passes: number, ms: number }}
 */
export function louvain( graph, options = {} ) {

	const t0 = now();
	const { seed = 1, resolution = 1, minGain = MIN_GAIN, refine = null, onProgress = null } = options;
	const n = graph.nodeCount | 0;
	const g0 = csrFromEdges( n, graph.edges, graph.weights );
	const r = louvainCsr( g0, { ...options, seed, resolution, minGain } );
	let refined = false;
	if ( refine && refine.maxLeafSize > 0 && r.levels.length ) {

		const fine = refineLeaves( g0, r.levels[ 0 ], r.levelSizes[ 0 ], refine, seed, resolution, minGain, onProgress );
		if ( fine ) {

			r.levels.unshift( fine.level );
			r.levelSizes.unshift( fine.K );
			r.modularity.unshift( csrModularity( g0, fine.level, resolution ) );
			refined = true;

		}

	}

	return { levels: r.levels, modularity: r.modularity, levelSizes: r.levelSizes, refined, passes: r.passes, ms: now() - t0 };

}

function csrModularity( g, part, resolution ) {

	const { n, start, adj, w, loop, m2 } = g;
	if ( m2 <= 0 ) return 0;
	let K = 0;
	for ( let i = 0; i < n; i ++ ) if ( part[ i ] + 1 > K ) K = part[ i ] + 1;
	const tot = new Float64Array( K );
	let inside = 0;
	for ( let i = 0; i < n; i ++ ) {

		const c = part[ i ];
		tot[ c ] += g.k[ i ];
		inside += 2 * loop[ i ];
		for ( let e = start[ i ]; e < start[ i + 1 ]; e ++ ) if ( part[ adj[ e ] ] === c ) inside += w[ e ];

	}

	let q = inside / m2;
	for ( let c = 0; c < K; c ++ ) q -= resolution * ( tot[ c ] / m2 ) ** 2;
	return q;

}

/** Modularity of a partition (array of community ids per node) of { nodeCount, edges, weights? }. */
export function modularity( graph, partition, resolution = 1 ) {

	return csrModularity( csrFromEdges( graph.nodeCount | 0, graph.edges, graph.weights ), partition, resolution );

}

/**
 * Canonical hierarchy for a canonical graph (see format.js): Louvain levels with oversized leaf
 * communities refined, filtered like normalizeLevels (no all-singleton level, no single-cluster
 * level, no level that merges nothing) with modularity kept aligned.
 */
export function louvainHierarchy( g, options = {} ) {

	const { maxLeafSize = DEFAULT_MAX_LEAF_SIZE, minLevelRatio = DEFAULT_MIN_LEVEL_RATIO, seed = 1, resolution = 1 } = options;
	const r = louvain( g, { ...options, seed, resolution, refine: maxLeafSize > 0 ? { maxLeafSize, maxDepth: options.maxDepth ?? 6 } : null } );
	const n = g.nodeCount;
	const kept = []; // level indices
	let prevK = n;
	for ( let l = 0; l < r.levels.length; l ++ ) {

		const K = r.levelSizes[ l ];
		if ( K <= 1 ) break;
		if ( K >= prevK ) continue;
		if ( kept.length && K * minLevelRatio > prevK ) {

			// barely merges anything: the coarser level replaces the previous one, except the
			// finest (it defines the leaf clusters, possibly refined)
			if ( kept.length > 1 ) {

				kept[ kept.length - 1 ] = l;
				prevK = K;

			}

			continue;

		}

		kept.push( l );
		prevK = K;

	}

	return {
		method: r.refined ? `louvain (refined, max leaf ${maxLeafSize})` : 'louvain',
		levels: kept.map( ( l ) => Array.from( r.levels[ l ] ) ),
		modularity: kept.map( ( l ) => r.modularity[ l ] ),
		resolution,
		seed,
		refined: r.refined,
		maxLeafSize,
		dropped: r.levels.length - kept.length,
		communities: r.levelSizes.length ? r.levelSizes[ r.levelSizes.length - 1 ] : 0, // of the final partition (1 = no split)
		ms: r.ms,
	};

}

/** Returns g when it already has a hierarchy (unless options.force); otherwise a copy with one. */
export function ensureHierarchy( g, options = {} ) {

	if ( g.hierarchy && g.hierarchy.levels && g.hierarchy.levels.length && ! options.force ) return g;
	const h = louvainHierarchy( g, options );
	return {
		...g,
		hierarchy: h.levels.length ? { method: h.method, levels: h.levels } : null,
		meta: { ...( g.meta || {} ), hierarchy: { method: h.method, modularity: h.modularity, resolution: h.resolution, seed: h.seed, refined: h.refined, maxLeafSize: h.maxLeafSize } },
	};

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}
