// Leiden community detection (V. A. Traag, L. Waltman & N. J. van Eck, "From Louvain to Leiden:
// guaranteeing well-connected communities", Sci. Rep. 9, 5233, 2019) and the nested clustering
// hierarchy built from it. Drop-in alternative to src/data/louvain.js: the same exports
// (ensureHierarchy, modularity, DEFAULT_MAX_LEAF_SIZE, DEFAULT_MIN_LEVEL_RATIO) and result shapes,
// with leiden / leidenHierarchy in place of louvain / louvainHierarchy. Pure JS, typed arrays, no
// dependencies besides the seeded PRNG: runs in the main thread, a worker, or Node.
//
// ─── Leiden (leidenCsr) ─────────────────────────────────────────────────────────────────────
//  One iteration, on the current (aggregate) graph, starting from singletons:
//   1. Fast local moving (moveNodesFast): a queue of all nodes in seeded random order; a node
//      moves to the community with the best strictly positive gain (an empty community counts),
//      and only its neighbours outside the new community are queued again.
//   2. Refinement (refinePartition): inside every community the nodes start as singletons again
//      and merge, in random order and only once, into a refined community of the same community
//      that they and it are well connected to, picked at random ∝ exp( gain / randomness ).
//      Merges follow edges, so refined communities are connected: this is what Louvain lacks.
//   3. Aggregation on the REFINED partition; the non-refined communities seed the assignment of
//      the aggregate nodes. Repeat from 1 until every community is a single aggregate node.
//  Further iterations restart from the result on the original graph and stop when nothing moves
//  (options.iterations caps them; graphs with clear communities settle in 2–3).
//  Quality: modularity  Q = Σ_c [ in_c / m − γ (tot_c / 2m)² ]  (the same definition as
//  louvain.js), or CPM  H = Σ_c [ in_c − γ n_c² / 2 ]  with options.quality = 'cpm'. Both use the
//  gain  w(v, C) − γ' · s_v · S_C  (modularity: s = degree, γ' = γ / 2m; CPM: s = node size, γ' = γ).
//
// ─── Hierarchy (options.strategy) ───────────────────────────────────────────────────────────
//  'recursive' (default): a tree of significant splits, turned into uniform-depth levels.
//   · The root is split by Leiden on the whole graph (always accepted). Every part is then
//     clustered again on its OWN induced subgraph: with its own null model the resolution limit
//     of the level above does not apply. A split is kept only when it is significant: its
//     modularity must be at least options.significance (1.3) times what a random graph of that
//     size and degree reaches (randomGraphModularity). This stops the recursion where the
//     structure stops (an unstructured community splits into parts too, just not better than
//     chance), down to options.levelsBelow levels.
//   · Coarser groups: after each split, Leiden runs on the graph of the connections BETWEEN the
//     parts (self loops dropped). A significant grouping (modularity ≥ coarsenMinModularity, 0.3)
//     is inserted above the parts, repeated up to options.levelsAbove times. At the root this
//     adds the levels above the modularity optimum, which no Louvain / Leiden run produces.
//   · Balancing (balanceTree): modularity merges small clusters, so some tree nodes are unions
//     of clusters of their siblings' kind and make their branch one level too deep. Such nodes
//     are dissolved; the remaining depth differences are aligned by cluster size (treeLevels).
//   · Leaves above maxLeafSize are cut into connected pieces (Leiden parts packed up to the
//     limit; BFS-grown chunks when Leiden will not split, e.g. a clique) as a new finest level.
//  'aggregate': the refined partitions of the aggregation levels of the run (finest first, the
//   last one is the final partition). Measured to recover nested structure worse than both the
//   recursive strategy and Louvain's levels (the finest levels are small random fragments).
//  'flat': the Leiden partition only.
//  Every cluster of every level induces a connected subgraph, for all strategies.
//
// ─── Output ─────────────────────────────────────────────────────────────────────────────────
//  levels[l][i] = cluster of original node i at level l, dense 0..K_l−1, levels[0] finest, every
//  level nested in the next (the canonical hierarchy of src/data/format.js). modularity[l] is the
//  modularity of levels[l] on the input graph. Unlike Louvain's dendrogram it is NOT monotone:
//  the levels above the modularity optimum have a lower modularity than `partition`.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//  leiden( { nodeCount, edges, weights? }, options ) -> { levels, modularity, levelSizes,
//          partition, partitionModularity, communities, refined, passes, iterations, stable, ms }
//      partition: the Leiden partition of the whole graph (the modularity optimum; it is one of
//      the levels unless balancing dissolved some of its communities).
//      options: seed (1), resolution γ (1), quality ('modularity' | 'cpm'), nodeWeights (CPM node
//               sizes, default 1), randomness θ (0.01), iterations (4; ≤ 0 = until stable, at most 1000),
//               strategy ('recursive' | 'aggregate' | 'flat'), significance (1.3; 0 = keep every
//               split), minSplitSize (12), levelsBelow (3), levelsAbove (3), coarsenResolution
//               (1.5), coarsenMinModularity (0.3), balance (true),
//               refine ({ maxLeafSize, maxDepth = 6 } or null),
//               onProgress({ phase: 'leiden', level, pass: iteration, communities, modularity }
//                          | { phase: 'refine', level, pass: nodes done, communities: nodeCount })
//  modularity( { nodeCount, edges, weights? }, partition, resolution = 1 ) -> Q
//  randomGraphModularity( n, k ) -> expected maximum modularity of a random graph
//  leidenHierarchy( g, options ) -> { method, levels, modularity, resolution, seed, refined, maxLeafSize, dropped, ms }
//      canonical hierarchy for a canonical graph; levels pass format.js normalizeLevels unchanged.
//      options.maxLeafSize defaults to DEFAULT_MAX_LEAF_SIZE (200; 0 disables the size limit).
//      options.minLevelRatio (DEFAULT_MIN_LEVEL_RATIO = 1.2; 1 keeps every level): a level that
//      does not reduce the cluster count of the kept level below it by this factor is dropped
//      (louvain.js keeps the coarser of the two instead; here the finer one stays, because the
//      finest level defines the leaves and the others are validated splits).
//      options.maxLevels (DEFAULT_MAX_LEVELS = 6): more levels than this lose the least distinct ones.
//  ensureHierarchy( g, options ) -> g itself when it already has a hierarchy (unless options.force),
//      otherwise a shallow copy with `hierarchy` set and meta.hierarchy = { method, modularity, … }.
//
//  __internals: moveNodesFast / refinePartition and their CSR helpers, exported for the unit tests only.
//
//  Weights must be non-negative. Deterministic for a given seed.

import { rng } from '../graph/prepare.js';

export const DEFAULT_MAX_LEAF_SIZE = 200;
export const DEFAULT_MIN_LEVEL_RATIO = 1.2;
export const DEFAULT_MAX_LEVELS = 6;
export const DEFAULT_RANDOMNESS = 0.01;
export const DEFAULT_ITERATIONS = 4;
export const DEFAULT_SIGNIFICANCE = 1.3;
export const DEFAULT_COARSEN_MIN_MODULARITY = 0.3;
export const DEFAULT_COARSEN_RESOLUTION = 1.5;
export const DEFAULT_MIN_SPLIT_SIZE = 12;
export const DEFAULT_LEVELS_BELOW = 3;
export const DEFAULT_LEVELS_ABOVE = 3;
const BALANCE_MIN_SCALE = 0.25;
const EARLY_REJECT = 0.85;
const PACK_MAX_PARTS = 384;
const MAX_ITERATIONS = 1000; // safety net of "until stable" (the slowest graph seen needed 108)

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

/** Adds weighted degrees k (a self loop counts twice) and the total 2m. */
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

/**
 * Aggregates the parts (dense ids 0..K-1) of g into a new CSR graph: one node per part, edge
 * weights summed, internal weight kept as a self loop. With keepLoops = false the self loops are
 * dropped (the graph of the connections BETWEEN the parts only).
 */
function aggregate( g, part, K, keepLoops = true, S = null ) {

	const { n, start, adj, w, loop } = g;
	const cstart = new Int32Array( K + 1 );
	for ( let i = 0; i < n; i ++ ) cstart[ part[ i ] + 1 ] ++;
	for ( let c = 0; c < K; c ++ ) cstart[ c + 1 ] += cstart[ c ];
	const members = new Int32Array( n );
	{

		const fill = cstart.slice( 0, K );
		for ( let i = 0; i < n; i ++ ) members[ fill[ part[ i ] ] ++ ] = i;

	}

	const nstart = new Int32Array( K + 1 );
	// worst case as many entries as g has; with scratch buffers only the used part is allocated
	const pooled = S !== null && S.edgeAdj.length >= adj.length;
	const nadj = pooled ? S.edgeAdj : new Int32Array( adj.length );
	const nw = pooled ? S.edgeW : new Float64Array( adj.length );
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

				const cj = part[ adj[ e ] ];
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

		if ( keepLoops ) nloop[ c ] = loops + internal / 2; // every internal edge was seen from both ends
		for ( let q = 0; q < cnt; q ++ ) {

			const cj = touched[ q ];
			nadj[ pos ] = cj;
			nw[ pos ++ ] = acc[ cj ];
			acc[ cj ] = - 1;

		}

		nstart[ c + 1 ] = pos;

	}

	return finishCsr( K, nstart, pooled ? nadj.slice( 0, pos ) : nadj.subarray( 0, pos ), pooled ? nw.slice( 0, pos ) : nw.subarray( 0, pos ), nloop );

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

// ── scratch memory ───────────────────────────────────────────────────────────────────────────

/** Work arrays for graphs of up to n nodes; reused by every (sub-)run so recursion does not allocate. */
function createScratch( n, entries = 0 ) {

	return {
		n,
		edgeAdj: new Int32Array( entries ), // aggregate() builds into these, then copies the used part
		edgeW: new Float64Array( entries ),
		tot: new Float64Array( n ), // node weight per community
		csize: new Int32Array( n ), // nodes per community
		neighW: new Float64Array( n ).fill( - 1 ), // edge weight to each neighbouring community (-1: untouched)
		neighCom: new Int32Array( n + 1 ),
		queue: new Int32Array( n ),
		inQueue: new Uint8Array( n ),
		empty: new Int32Array( n ), // stack of unused community ids
		rtot: new Float64Array( n ), // node weight per refined community
		ext: new Float64Array( n ), // weight between a refined community and the rest of its community
		merged: new Uint8Array( n ), // refined community has more than one node
		cand: new Float64Array( n + 1 ),
	};

}

function shuffle( arr, n, rand ) {

	for ( let i = n - 1; i > 0; i -- ) {

		const j = Math.floor( rand() * ( i + 1 ) );
		const t = arr[ i ]; arr[ i ] = arr[ j ]; arr[ j ] = t;

	}

}

/** Renumbers part in place to dense ids by first appearance; returns the part count. */
function renumber( part, map ) {

	const n = part.length;
	map.fill( - 1, 0, n );
	let K = 0;
	for ( let i = 0; i < n; i ++ ) {

		const c = part[ i ];
		if ( map[ c ] < 0 ) map[ c ] = K ++;
		part[ i ] = map[ c ];

	}

	return K;

}

// ── fast local moving ────────────────────────────────────────────────────────────────────────

/**
 * Queue-based local moving (the paper's MoveNodesFast). comm holds the starting assignment
 * (ids < n) and is updated in place. A node moves to the community with the best strictly
 * positive gain over staying (an empty community is always a candidate); only the neighbours
 * that are not in its new community are queued again. Returns the number of moves.
 * Gain of joining C:  w(v, C) − γ' · nw_v · tot(C)  (γ' = γ / 2m and nw = degree for modularity,
 * γ' = γ and nw = node size for CPM).
 */
function moveNodesFast( g, nw, comm, gamma, rand, S ) {

	const { n, start, adj, w } = g;
	const { tot, csize, neighW, neighCom, queue, inQueue, empty, ext } = S;
	tot.fill( 0, 0, n );
	csize.fill( 0, 0, n );
	for ( let i = 0; i < n; i ++ ) {

		tot[ comm[ i ] ] += nw[ i ];
		csize[ comm[ i ] ] ++;

	}

	let emptyCount = 0;
	for ( let c = n - 1; c >= 0; c -- ) if ( csize[ c ] === 0 ) empty[ emptyCount ++ ] = c;
	for ( let i = 0; i < n; i ++ ) queue[ i ] = i;
	shuffle( queue, n, rand );
	inQueue.fill( 1, 0, n );
	let head = 0, tail = 0, count = n, moves = 0;
	while ( count > 0 ) {

		const v = queue[ head ];
		head = head + 1 === n ? 0 : head + 1;
		count --;
		inQueue[ v ] = 0;
		const cv = comm[ v ];
		const kv = nw[ v ];
		const s0 = start[ v ], s1 = start[ v + 1 ];
		let cnt = 1;
		neighCom[ 0 ] = cv;
		neighW[ cv ] = 0;
		for ( let e = s0; e < s1; e ++ ) {

			const cj = comm[ adj[ e ] ];
			if ( neighW[ cj ] < 0 ) {

				neighW[ cj ] = 0;
				neighCom[ cnt ++ ] = cj;

			}

			neighW[ cj ] += w[ e ];

		}

		// take v out of its community, then pick the best one (staying wins ties)
		tot[ cv ] -= kv;
		csize[ cv ] --;
		const f = gamma * kv;
		let best = cv;
		let bestGain = neighW[ cv ] - tot[ cv ] * f;
		for ( let q = 1; q < cnt; q ++ ) {

			const c = neighCom[ q ];
			const gain = neighW[ c ] - tot[ c ] * f;
			if ( gain > bestGain ) {

				bestGain = gain;
				best = c;

			}

		}

		// an empty community (gain 0) beats every option with a negative gain
		let toBest = neighW[ best ];
		if ( bestGain < 0 && csize[ cv ] > 0 ) {

			best = empty[ -- emptyCount ];
			toBest = 0;

		}

		for ( let q = 0; q < cnt; q ++ ) neighW[ neighCom[ q ] ] = - 1;
		tot[ best ] += kv;
		csize[ best ] ++;
		ext[ v ] = toBest; // weight between v and the rest of its community, kept current for the refinement
		if ( best !== cv ) {

			comm[ v ] = best;
			moves ++;
			if ( csize[ cv ] === 0 ) empty[ emptyCount ++ ] = cv;
			for ( let e = s0; e < s1; e ++ ) {

				const u = adj[ e ];
				if ( comm[ u ] === best ) ext[ u ] += w[ e ];
				else if ( inQueue[ u ] === 0 ) {

					// (a neighbour left behind in the old community is queued here, so its ext is recomputed)
					inQueue[ u ] = 1;
					queue[ tail ] = u;
					tail = tail + 1 === n ? 0 : tail + 1;
					count ++;

				}

			}

		}

	}

	return moves;

}

// ── refinement ───────────────────────────────────────────────────────────────────────────────

/**
 * The paper's RefinePartition / MergeNodesSubset, following the authors' reference
 * implementation (CWTSLeiden networkanalysis, LocalMergingAlgorithm). Every node starts as its
 * own refined community. Nodes are visited once in random order; a node that is still alone and
 * well connected to its community S (w(v, S − v) ≥ γ' nw_v (tot_S − nw_v)) merges into a refined
 * community C ⊆ S that is itself well connected to S, chosen at random with probability
 * ∝ exp( ΔH / θ ) among the candidates with ΔH = w(v, C) − γ' nw_v tot_C ≥ 0 (staying alone has
 * ΔH = 0). Merges only follow edges, so refined communities are connected.
 * Needs S.tot / S.csize / S.ext as left by moveNodesFast. Writes dense ids into `refined`; returns their count.
 */
function refinePartition( g, nw, comm, gamma, theta, rand, S, refined ) {

	const { n, start, adj, w } = g;
	const { tot, csize, neighW, neighCom, rtot, ext, merged, cand } = S;
	const order = S.queue;
	for ( let v = 0; v < n; v ++ ) {

		rtot[ v ] = nw[ v ]; // ext[ v ] = w( v, S − v ) comes from moveNodesFast
		refined[ v ] = v;
		order[ v ] = v;

	}

	merged.fill( 0, 0, n );
	shuffle( order, n, rand );
	const invTheta = 1 / theta;
	let K = n;
	for ( let t = 0; t < n; t ++ ) {

		const v = order[ t ];
		if ( merged[ v ] === 1 ) continue;
		const cv = comm[ v ];
		if ( csize[ cv ] === 1 ) continue;
		const kv = nw[ v ];
		const totS = tot[ cv ];
		if ( ext[ v ] < gamma * kv * ( totS - kv ) ) continue; // not well connected: stays alone
		const s0 = start[ v ], s1 = start[ v + 1 ];
		let cnt = 0;
		for ( let e = s0; e < s1; e ++ ) {

			const u = adj[ e ];
			if ( comm[ u ] !== cv ) continue;
			const r = refined[ u ];
			if ( neighW[ r ] < 0 ) {

				neighW[ r ] = 0;
				neighCom[ cnt ++ ] = r;

			}

			neighW[ r ] += w[ e ];

		}

		// candidates: well-connected refined communities with a non-negative gain; staying alone has gain 0
		let maxDiff = 0;
		for ( let q = 0; q < cnt; q ++ ) {

			const r = neighCom[ q ];
			const rt = rtot[ r ];
			let diff = - 1;
			if ( ext[ r ] >= gamma * rt * ( totS - rt ) ) {

				diff = neighW[ r ] - gamma * kv * rt;
				if ( diff > maxDiff ) maxDiff = diff;

			}

			cand[ q ] = diff;
			neighW[ r ] = - 1;

		}

		// sample ∝ exp( diff / θ ), computed relative to the best candidate so it cannot overflow
		let total = Math.exp( - maxDiff * invTheta ); // staying alone
		const stay = total;
		for ( let q = 0; q < cnt; q ++ ) {

			const diff = cand[ q ];
			if ( diff >= 0 ) {

				const x = ( diff - maxDiff ) * invTheta;
				if ( x > - 40 ) total += Math.exp( x );

			}

			cand[ q ] = total;

		}

		const pick = rand() * total;
		if ( pick < stay ) continue;
		let q = 0;
		while ( q < cnt - 1 && cand[ q ] <= pick ) q ++;
		const chosen = neighCom[ q ];
		refined[ v ] = chosen;
		rtot[ chosen ] += kv;
		merged[ chosen ] = 1;
		K --;
		let x = ext[ chosen ];
		for ( let e = s0; e < s1; e ++ ) {

			const u = adj[ e ];
			if ( comm[ u ] !== cv ) continue;
			if ( refined[ u ] === chosen ) x -= w[ e ];
			else x += w[ e ];

		}

		ext[ chosen ] = x;

	}

	if ( K < n ) renumber( refined, S.empty );
	return K;

}

/** Splits every community of comm into its connected components on g (in place, dense ids); returns the count. */
function splitComponents( g, comm ) {

	const { n, start, adj } = g;
	const out = new Int32Array( n ).fill( - 1 );
	const stack = new Int32Array( n );
	let K = 0;
	for ( let s = 0; s < n; s ++ ) {

		if ( out[ s ] >= 0 ) continue;
		let top = 0;
		stack[ top ++ ] = s;
		out[ s ] = K;
		const c = comm[ s ];
		while ( top > 0 ) {

			const i = stack[ -- top ];
			for ( let e = start[ i ]; e < start[ i + 1 ]; e ++ ) {

				const j = adj[ e ];
				if ( out[ j ] < 0 && comm[ j ] === c ) {

					out[ j ] = K;
					stack[ top ++ ] = j;

				}

			}

		}

		K ++;

	}

	comm.set( out );
	return K;

}

// ── core ─────────────────────────────────────────────────────────────────────────────────────

/**
 * Leiden on a CSR graph.
 * @returns {{ partition: Int32Array, K: number, chain: Int32Array[], chainSizes: number[], iterations: number, stable: boolean, moves: number }}
 *   partition: final communities (dense) of the CSR's nodes; chain: the refined partitions of the
 *   aggregation levels of the last iteration (finest first, nested, the last one is `partition`).
 */
function leidenCsr( g0, nw0, gamma, { seed = 1, randomness = DEFAULT_RANDOMNESS, iterations = 2, maxLevels = 64, initial = null, report = null } = {}, S = null ) {

	const n0 = g0.n;
	if ( ! S || S.n < n0 ) S = createScratch( n0, g0.adj.length );
	const rand = rng( seed );
	const maxIter = iterations > 0 ? iterations : MAX_ITERATIONS;
	let partition = new Int32Array( n0 );
	if ( initial ) partition.set( initial );
	else for ( let i = 0; i < n0; i ++ ) partition[ i ] = i;
	let K = n0, chain = [], chainSizes = [], stable = false, iter = 0, totalMoves = 0;
	for ( ; iter < maxIter && ! stable; iter ++ ) {

		let g = g0, nw = nw0;
		let comm = partition.slice();
		const flat = new Int32Array( n0 ); // original node -> node of the current aggregate graph
		for ( let i = 0; i < n0; i ++ ) flat[ i ] = i;
		const levels = [], sizes = [];
		let moves = 0, split = false;
		for ( let level = 0; level < maxLevels; level ++ ) {

			moves += moveNodesFast( g, nw, comm, gamma, rand, S );
			let Kc = 0;
			for ( let c = 0; c < g.n; c ++ ) if ( S.csize[ c ] > 0 ) Kc ++;
			report?.( iter, level, g, comm, Kc );
			if ( Kc === g.n ) break; // every community is a single node: done
			const refined = new Int32Array( g.n );
			const Kr = refinePartition( g, nw, comm, gamma, randomness, rand, S, refined );
			if ( Kr === g.n ) {

				// nothing merged although communities have several nodes (only possible through the
				// random "stay alone" choice at zero gain): keep the communities, made connected
				const Kf = splitComponents( g, comm );
				for ( let i = 0; i < n0; i ++ ) flat[ i ] = comm[ flat[ i ] ];
				levels.push( flat.slice() );
				sizes.push( Kf );
				split = true;
				break;

			}

			for ( let i = 0; i < n0; i ++ ) flat[ i ] = refined[ flat[ i ] ];
			levels.push( flat.slice() );
			sizes.push( Kr );
			// aggregate on the refined partition; the non-refined one seeds the next level
			renumber( comm, S.empty );
			const seedComm = new Int32Array( Kr );
			const nwAgg = new Float64Array( Kr );
			for ( let v = 0; v < g.n; v ++ ) {

				seedComm[ refined[ v ] ] = comm[ v ];
				nwAgg[ refined[ v ] ] += nw[ v ];

			}

			g = aggregate( g, refined, Kr, true, S );
			nw = nwAgg;
			comm = seedComm;

		}

		totalMoves += moves;
		chain = levels;
		chainSizes = sizes;
		// final partition of this iteration = the nodes of the last aggregate graph
		if ( levels.length ) {

			partition = levels[ levels.length - 1 ].slice();
			K = sizes[ sizes.length - 1 ];

		} else {

			for ( let i = 0; i < n0; i ++ ) partition[ i ] = i;
			K = n0;

		}

		if ( moves === 0 && ! split ) stable = true; // nothing moved at any level

	}

	return { partition, K, chain, chainSizes, iterations: iter, stable, moves: totalMoves };

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

// ── significance of a split ──────────────────────────────────────────────────────────────────

/**
 * Expected maximum modularity of a random graph with n nodes and mean degree k: Guimerà,
 * Sales-Pardo & Amaral, "Modularity from fluctuations in random graphs and complex networks",
 * Phys. Rev. E 70, 025101 (2004), Q ≈ (1 − 2/√n) (2/k)^(2/3). Even a graph without any community
 * structure can be cut into parts with this much modularity, so a split is only evidence of real
 * structure when it clearly exceeds it. Checked against this module on random graphs with n = 12
 * to 1000 and k = 3 to 20: within 0.02 of the median Leiden modularity for n ≥ 80 and k ≥ 8, and
 * above it (a stricter test) by up to 0.06 for sparser graphs and up to 0.12 for n ≤ 20.
 */
export function randomGraphModularity( n, k ) {

	if ( ! ( n > 4 ) || ! ( k > 0 ) ) return 1;
	return Math.min( 1, ( 1 - 2 / Math.sqrt( n ) ) * Math.pow( 2 / k, 2 / 3 ) );

}

function partSizes( part, K ) {

	const s = new Int32Array( K );
	for ( let i = 0; i < part.length; i ++ ) s[ part[ i ] ] ++;
	return s;

}

/** CSR of the members of every part: members[ cstart[c] .. cstart[c+1] ). */
function partMembers( part, K ) {

	const n = part.length;
	const cstart = new Int32Array( K + 1 );
	for ( let i = 0; i < n; i ++ ) cstart[ part[ i ] + 1 ] ++;
	for ( let c = 0; c < K; c ++ ) cstart[ c + 1 ] += cstart[ c ];
	const members = new Int32Array( n );
	const fill = cstart.slice( 0, K );
	for ( let i = 0; i < n; i ++ ) members[ fill[ part[ i ] ] ++ ] = i;
	return { cstart, members };

}

function subSeed( seed, a, b ) {

	return ( ( seed ^ Math.imul( a + 1, 0x9e3779b1 ) ^ Math.imul( b + 1, 0x85ebca6b ) ) >>> 0 ) || 1;

}

/** Modularity a split of sub must reach to count as real structure (see randomGraphModularity). */
function significanceThreshold( sub, ctx, floor = 0 ) {

	if ( ctx.significance <= 0 ) return floor;
	let n = 0; // nodes that have edges (an aggregate graph can contain isolated ones)
	for ( let i = 0; i < sub.n; i ++ ) if ( sub.start[ i + 1 ] > sub.start[ i ] ) n ++;
	if ( n === 0 ) return Infinity;
	return Math.max( floor, ctx.significance * randomGraphModularity( n, sub.adj.length / n ) ); // structural mean degree

}

function isSignificant( sub, K, q, threshold ) {

	return K >= 2 && K < sub.n && q > 0 && q >= threshold;

}

/**
 * Modularity-based Leiden on a (sub)graph; returns { partition, K, q } with q = modularity at
 * resolution 1. The first iteration already comes within a few percent of the final modularity,
 * so the remaining iterations only run when the split can still reach `threshold`.
 */
function splitGraph( sub, ctx, seed, threshold ) {

	if ( sub.m2 <= 0 || sub.n < 2 ) return { partition: null, K: 1, q: 0 };
	const gamma = ctx.subResolution / sub.m2;
	let r = leidenCsr( sub, sub.k, gamma, { seed, randomness: ctx.randomness, iterations: 1 }, ctx.S );
	let q = r.K > 1 ? csrModularity( sub, r.partition, 1 ) : 0;
	if ( r.K > 1 && r.K < sub.n && ! r.stable && ctx.iterations !== 1 && q >= EARLY_REJECT * threshold ) {

		r = leidenCsr( sub, sub.k, gamma, { seed: subSeed( seed, 7, 7 ), randomness: ctx.randomness, iterations: ctx.iterations > 0 ? ctx.iterations - 1 : ctx.iterations, initial: r.partition }, ctx.S );
		q = r.K > 1 ? csrModularity( sub, r.partition, 1 ) : 0;

	}

	return { partition: r.partition, K: r.K, q };

}

/**
 * Fallback split of a graph that Leiden will not split (cliques, stars): regions grown by BFS
 * over the still unassigned nodes, so every chunk is connected and has at most maxSize nodes.
 */
function connectedChunks( sub, maxSize ) {

	const m = sub.n;
	const size = Math.ceil( m / Math.ceil( m / maxSize ) );
	const out = new Int32Array( m ).fill( - 1 );
	const queue = new Int32Array( m );
	let id = 0;
	for ( let s = 0; s < m; s ++ ) {

		if ( out[ s ] >= 0 ) continue;
		let head = 0, tail = 0, cnt = 1;
		queue[ tail ++ ] = s;
		out[ s ] = id;
		while ( head < tail && cnt < size ) {

			const i = queue[ head ++ ];
			for ( let e = sub.start[ i ]; e < sub.start[ i + 1 ] && cnt < size; e ++ ) {

				const j = sub.adj[ e ];
				if ( out[ j ] < 0 ) {

					out[ j ] = id;
					queue[ tail ++ ] = j;
					cnt ++;

				}

			}

		}

		id ++;

	}

	return { partition: out, K: id };

}

/**
 * Merges the parts of a partition of sub into as few connected groups of at most maxSize nodes
 * as a greedy agglomeration finds: repeatedly the adjacent pair with the best modularity gain
 * that still fits. Used for forced splits, where the parts carry no significant structure and
 * fewer, fuller pieces invent less.
 */
function packParts( sub, part, K, maxSize ) {

	if ( K < 3 || K > PACK_MAX_PARTS ) return { partition: part, K };
	const agg = aggregate( sub, part, K );
	const size = partSizes( part, K );
	const W = new Float64Array( K * K );
	for ( let a = 0; a < K; a ++ ) for ( let e = agg.start[ a ]; e < agg.start[ a + 1 ]; e ++ ) W[ a * K + agg.adj[ e ] ] = agg.w[ e ];
	const k = agg.k.slice();
	const alive = new Uint8Array( K ).fill( 1 );
	const group = new Int32Array( K );
	for ( let a = 0; a < K; a ++ ) group[ a ] = a;
	for ( ;; ) {

		let bestA = - 1, bestB = - 1, best = - Infinity;
		for ( let a = 0; a < K; a ++ ) {

			if ( ! alive[ a ] ) continue;
			for ( let b = a + 1; b < K; b ++ ) {

				if ( ! alive[ b ] || W[ a * K + b ] <= 0 || size[ a ] + size[ b ] > maxSize ) continue;
				const gain = W[ a * K + b ] - ( k[ a ] * k[ b ] ) / agg.m2;
				if ( gain > best ) {

					best = gain; bestA = a; bestB = b;

				}

			}

		}

		if ( bestA < 0 ) break;
		// merge b into a
		alive[ bestB ] = 0;
		size[ bestA ] += size[ bestB ];
		k[ bestA ] += k[ bestB ];
		for ( let c = 0; c < K; c ++ ) {

			if ( c === bestA || c === bestB ) continue;
			W[ bestA * K + c ] += W[ bestB * K + c ];
			W[ c * K + bestA ] = W[ bestA * K + c ];

		}

		for ( let c = 0; c < K; c ++ ) if ( group[ c ] === bestB ) group[ c ] = bestA;

	}

	const out = new Int32Array( sub.n );
	for ( let i = 0; i < sub.n; i ++ ) out[ i ] = group[ part[ i ] ];
	const Kp = renumber( out, new Int32Array( K ) );
	return { partition: out, K: Kp };

}

/**
 * Splits sub into connected parts of at most maxSize nodes: the Leiden parts `r` of sub,
 * recursing into the parts that are still too large, all flattened into one partition and
 * packed (packParts); BFS-grown chunks where Leiden does not split.
 */
function forcedSplit( sub, maxSize, r, ctx, seed, depth ) {

	if ( ! r || r.K < 2 || r.K >= sub.n ) return connectedChunks( sub, maxSize );
	const sizes = partSizes( r.partition, r.K );
	let over = false;
	for ( let c = 0; c < r.K; c ++ ) if ( sizes[ c ] > maxSize ) over = true;
	if ( ! over ) return packParts( sub, r.partition, r.K, maxSize );
	const m = sub.n;
	const out = new Int32Array( m );
	const { cstart, members } = partMembers( r.partition, r.K );
	const localOf = new Int32Array( m ).fill( - 1 );
	let next = 0;
	for ( let c = 0; c < r.K; c ++ ) {

		const mem = members.subarray( cstart[ c ], cstart[ c + 1 ] );
		if ( sizes[ c ] <= maxSize ) {

			for ( let q = 0; q < mem.length; q ++ ) out[ mem[ q ] ] = next;
			next ++;
			continue;

		}

		for ( let q = 0; q < mem.length; q ++ ) localOf[ mem[ q ] ] = q;
		const inner = subgraph( sub, mem, localOf );
		for ( let q = 0; q < mem.length; q ++ ) localOf[ mem[ q ] ] = - 1;
		const innerSeed = subSeed( seed, depth, c );
		const parts = forcedSplit( inner, maxSize, depth + 1 < ctx.forceDepth ? splitGraph( inner, ctx, innerSeed, Infinity ) : null, ctx, innerSeed, depth + 1 );
		for ( let q = 0; q < mem.length; q ++ ) out[ mem[ q ] ] = next + parts.partition[ q ];
		next += parts.K;

	}

	return packParts( sub, out, next, maxSize );

}

// ── hierarchy: a tree of significant splits ──────────────────────────────────────────────────

/**
 * One coarsening step: Leiden on the graph of the connections BETWEEN the parts (self loops
 * dropped, so the null model only sees the inter-part weight). Run at resolution
 * ctx.coarsenResolution > 1 so that a part only joins a group it is clearly tied to (without the
 * margin a part that has no partner still drifts into some group, because a lone node cannot hold
 * itself without a self loop). Kept when it is significant and reaches ctx.coarsenMinModularity.
 * Returns { partition (per node of sub), K } or null.
 */
function coarsenPartition( sub, part, K, ctx, seed ) {

	if ( K < 4 ) return null;
	const agg = aggregate( sub, part, K, false );
	if ( agg.m2 <= 0 ) return null;
	let best = null, bestQuality = - Infinity;
	const tries = K <= 2000 ? 4 : 1; // tiny graphs: a few restarts cost nothing
	for ( let t = 0; t < tries; t ++ ) {

		const r = leidenCsr( agg, agg.k, ctx.coarsenResolution / agg.m2, { seed: subSeed( seed, 977, t ), randomness: ctx.randomness, iterations: ctx.iterations }, ctx.S );
		const quality = csrModularity( agg, r.partition, ctx.coarsenResolution );
		if ( quality > bestQuality ) {

			best = r;
			bestQuality = quality;

		}

	}

	const q = best.K > 1 ? csrModularity( agg, best.partition, 1 ) : 0;
	if ( ! isSignificant( agg, best.K, q, significanceThreshold( agg, ctx, ctx.coarsenMinModularity ) ) ) return null;
	const out = new Int32Array( sub.n );
	for ( let i = 0; i < sub.n; i ++ ) out[ i ] = best.partition[ part[ i ] ];
	return { partition: out, K: best.K };

}

function makeNode( members, parent ) {

	return { members, size: members ? members.length : 0, parent, children: null, height: 0, depth: 0, lev: 0, split: null };

}

/**
 * Grows the tree under `node`: Leiden on the node's own induced subgraph (its own null model, so
 * the resolution limit of the level above does not apply). A significant split becomes the
 * node's children; when the parts themselves group significantly (coarsenPartition, repeated) the
 * groups are inserted between the node and the parts. Then every part is grown in turn.
 * `first`: an already computed split of sub (the Leiden partition of the whole graph at the root,
 * which is always accepted).
 */
function grow( g0, node, sub, first, splitDepth, ctx ) {

	const size = node.size;
	const done = () => {

		ctx.covered += size;
		ctx.onProgress?.( { phase: 'refine', level: splitDepth, pass: ctx.covered, communities: g0.n, modularity: NaN } );

	};

	const tooLarge = ctx.maxLeafSize > 0 && size > ctx.maxLeafSize;
	const mayGrow = ctx.strategy === 'recursive' && splitDepth <= ctx.levelsBelow && size >= ctx.minSplitSize;
	if ( ! first && ! mayGrow && ! tooLarge ) return done();
	const seed = subSeed( ctx.seed, splitDepth, node.members.length ? node.members[ 0 ] : 0 );
	if ( ! sub ) sub = subgraphOf( g0, node.members, ctx.localOf );
	const threshold = mayGrow ? significanceThreshold( sub, ctx ) : Infinity;
	const r = first || splitGraph( sub, ctx, seed, threshold );
	if ( ! first && ! ( mayGrow && isSignificant( sub, r.K, r.q, threshold ) ) ) {

		// a leaf; one above maxLeafSize is cut into connected pieces now, while its subgraph exists
		if ( tooLarge ) node.split = forcedSplit( sub, ctx.maxLeafSize, r, ctx, seed, 0 );
		return done();

	}

	// the split and its own coarser groupings, finest first
	const chain = [ r.partition ], chainK = [ r.K ];
	if ( ctx.strategy === 'recursive' ) {

		for ( let up = 0; up < ctx.levelsAbove; up ++ ) {

			const c = coarsenPartition( sub, chain[ chain.length - 1 ], chainK[ chainK.length - 1 ], ctx, subSeed( seed, 31, up ) );
			if ( ! c ) break;
			chain.push( c.partition );
			chainK.push( c.K );

		}

	}

	// nodes of the parts, then the groups from the finest grouping upwards (a group of one is that one)
	const { cstart, members } = partMembers( chain[ 0 ], chainK[ 0 ] );
	for ( let q = 0; q < members.length; q ++ ) members[ q ] = node.members[ members[ q ] ];
	let nodes = [];
	const first0 = new Int32Array( chainK[ 0 ] ); // a member (local id) of every part
	for ( let i = sub.n - 1; i >= 0; i -- ) first0[ chain[ 0 ][ i ] ] = i;
	for ( let c = 0; c < chainK[ 0 ]; c ++ ) nodes.push( makeNode( members.subarray( cstart[ c ], cstart[ c + 1 ] ), node ) );
	const parts = nodes;
	let rep = first0;
	for ( let j = 1; j < chain.length; j ++ ) {

		const groups = Array.from( { length: chainK[ j ] }, () => [] );
		const nextRep = new Int32Array( chainK[ j ] );
		for ( let c = 0; c < nodes.length; c ++ ) {

			const gid = chain[ j ][ rep[ c ] ];
			groups[ gid ].push( nodes[ c ] );
			nextRep[ gid ] = rep[ c ];

		}

		nodes = groups.map( ( list ) => {

			if ( list.length === 1 ) return list[ 0 ];
			const group = makeNode( null, node );
			group.children = list;
			for ( const child of list ) {

				child.parent = group;
				group.size += child.size;

			}

			return group;

		} );
		rep = nextRep;

	}

	node.children = nodes;
	for ( const p of parts ) grow( g0, p, null, null, splitDepth + 1, ctx );

}

function subgraphOf( g0, members, localOf ) {

	for ( let q = 0; q < members.length; q ++ ) localOf[ members[ q ] ] = q;
	const sub = subgraph( g0, members, localOf );
	for ( let q = 0; q < members.length; q ++ ) localOf[ members[ q ] ] = - 1;
	return sub;

}

/**
 * Sets height and depth of every tree node and, with dissolve = true, balances the tree
 * bottom-up. Modularity merges small clusters (the resolution limit), so some nodes of the tree
 * are unions of several clusters of the kind their siblings are: they show up as children that
 * are taller than the others. For every node the target height of its children is the
 * size-weighted median of their heights; a taller child is dissolved (its children take its
 * place) until it fits, provided its children are of the siblings' scale (their mean log size is
 * at least BALANCE_MIN_SCALE of the siblings'). Shorter children stay as they are.
 */
function balanceTree( root, dissolve ) {

	const pre = [];
	const stack = [ root ];
	while ( stack.length ) {

		const v = stack.pop();
		pre.push( v );
		if ( v.children ) for ( const c of v.children ) stack.push( c );

	}

	for ( let k = pre.length - 1; k >= 0; k -- ) {

		const v = pre[ k ];
		v.height = 0;
		if ( ! v.children ) continue;
		if ( dissolve ) v.children = dissolveTallChildren( v );
		for ( const c of v.children ) if ( c.height + 1 > v.height ) v.height = c.height + 1;

	}

	root.depth = 0;
	stack.push( root );
	while ( stack.length ) {

		const v = stack.pop();
		if ( v.children ) for ( const c of v.children ) {

			c.depth = v.depth + 1;
			stack.push( c );

		}

	}

}

function dissolveTallChildren( v ) {

	// target height: the size-weighted median of the children's heights
	const sorted = v.children.slice().sort( ( a, b ) => a.height - b.height );
	let half = 0, target = 0;
	for ( const c of sorted ) half += c.size;
	half /= 2;
	for ( let q = 0, acc = 0; q < sorted.length; q ++ ) {

		acc += sorted[ q ].size;
		target = sorted[ q ].height;
		if ( acc >= half ) break;

	}

	// typical (log) size of the children that have the target height
	let sumLog = 0, weight = 0;
	for ( const c of v.children ) {

		if ( c.height === target ) {

			sumLog += c.size * Math.log( c.size );
			weight += c.size;

		}

	}

	const minLogSize = sumLog / weight + Math.log( BALANCE_MIN_SCALE );
	let list = v.children;
	for ( let again = true; again; ) {

		again = false;
		const next = [];
		for ( const c of list ) {

			let dissolve = false;
			if ( c.height > target ) {

				// only when its children are of the siblings' kind: not much smaller than them
				let cl = 0, cw = 0;
				for ( const cc of c.children ) {

					cl += cc.size * Math.log( cc.size );
					cw += cc.size;

				}

				dissolve = cl / cw >= minLogSize;

			}

			if ( dissolve ) {

				for ( const cc of c.children ) {

					cc.parent = v;
					next.push( cc );

				}

				again = true;

			} else next.push( c );

		}

		list = next;

	}

	return list;

}

/**
 * Turns the tree into uniform-depth levels (every node of the graph has one cluster per level).
 * A tree node v is a cluster at the levels lev(v) … lev(parent) − 1; leaves have lev 0 and
 * lev(v) > lev(child). Where a branch is shorter than the tallest one, lev is chosen by scale:
 * the level whose clusters on the tallest branches are closest to v in (log) size, so clusters
 * of one level have comparable sizes whichever branch they are in.
 */
function treeLevels( root, n, balance ) {

	balanceTree( root, balance );
	const order = []; // preorder, children in order
	const stack = [ root ];
	while ( stack.length ) {

		const v = stack.pop();
		order.push( v );
		if ( v.children ) for ( let k = v.children.length - 1; k >= 0; k -- ) stack.push( v.children[ k ] );

	}

	const H = root.height;
	if ( H === 0 ) return { levels: [], levelSizes: [] };
	// reference cluster size per level, from the nodes on the tallest branches
	const sumLog = new Float64Array( H ), weight = new Float64Array( H );
	for ( const v of order ) {

		if ( v !== root && v.depth + v.height === H ) {

			sumLog[ v.height ] += v.size * Math.log( v.size );
			weight[ v.height ] += v.size;

		}

	}

	root.lev = H;
	for ( const v of order ) {

		if ( v === root ) continue;
		v.lev = v.height;
		const hi = v.parent.lev - 1;
		if ( v.children && hi > v.height ) {

			const logSize = Math.log( v.size );
			let best = Infinity;
			for ( let l = v.height; l <= hi; l ++ ) {

				const d = Math.abs( logSize - sumLog[ l ] / weight[ l ] );
				if ( d < best ) {

					best = d;
					v.lev = l;

				}

			}

		}

	}

	const levels = Array.from( { length: H }, () => new Int32Array( n ) );
	const levelSizes = new Array( H ).fill( 0 );
	const cur = new Int32Array( H );
	for ( const v of order ) {

		if ( v === root ) continue;
		for ( let l = v.lev; l < v.parent.lev; l ++ ) cur[ l ] = levelSizes[ l ] ++;
		if ( ! v.children ) {

			const mem = v.members;
			for ( let l = 0; l < H; l ++ ) {

				const id = cur[ l ], level = levels[ l ];
				for ( let q = 0; q < mem.length; q ++ ) level[ mem[ q ] ] = id;

			}

		}

	}

	return { levels, levelSizes };

}

/** The level below the leaves: leaves above maxLeafSize cut into their pieces (null when there are none). */
function forcedLevel( root, n ) {

	const level = new Int32Array( n );
	let K = 0, any = false;
	const stack = [ root ];
	while ( stack.length ) {

		const v = stack.pop();
		if ( v.children ) {

			for ( let k = v.children.length - 1; k >= 0; k -- ) stack.push( v.children[ k ] );
			continue;

		}

		const mem = v.members;
		if ( v.split && v.split.K > 1 ) {

			for ( let q = 0; q < mem.length; q ++ ) level[ mem[ q ] ] = K + v.split.partition[ q ];
			K += v.split.K;
			any = true;

		} else {

			for ( let q = 0; q < mem.length; q ++ ) level[ mem[ q ] ] = K;
			K ++;

		}

	}

	return any ? { level, K } : null;

}

/**
 * Leiden on the whole graph, then the nested levels (finest first) around its partition.
 * @returns {{ levels: Int32Array[], levelSizes: number[], partition: Int32Array, K: number, refined: boolean, iterations: number, stable: boolean }}
 */
function buildLevels( g0, ctx ) {

	const n = g0.n;
	const nw0 = ctx.quality === 'cpm' ? ( ctx.nodeWeights || new Float64Array( n ).fill( 1 ) ) : g0.k;
	const gamma0 = ctx.quality === 'cpm' ? ctx.resolution : ( g0.m2 > 0 ? ctx.resolution / g0.m2 : 0 );
	const report = ctx.onProgress ? ( iter, level, g, comm, Kc ) => {

		const dense = comm.slice();
		renumber( dense, new Int32Array( g.n ) );
		const q = csrModularity( g, dense, ctx.quality === 'cpm' ? 1 : ctx.resolution );
		ctx.onProgress( { phase: 'leiden', level, pass: iter + 1, communities: Kc, modularity: q } );

	} : null;
	const top = leidenCsr( g0, nw0, gamma0, { seed: ctx.seed, randomness: ctx.randomness, iterations: ctx.iterations, report }, ctx.S );
	const result = { levels: [ top.partition ], levelSizes: [ top.K ], partition: top.partition, K: top.K, refined: false, iterations: top.iterations, stable: top.stable };
	// nothing to nest when no two nodes merged; one community or the flat strategy only need the size limit
	const sizeLimit = ctx.maxLeafSize > 0;
	const nest = top.K < 2 ? sizeLimit && n > ctx.maxLeafSize : top.K < n && ( ctx.strategy !== 'flat' || sizeLimit );
	if ( nest ) nestedLevels( g0, top, ctx, result );
	// canonical numbering: cluster ids by first appearance in node order (as format.js normalizeLevels)
	const map = new Int32Array( n );
	renumber( result.partition, map );
	for ( const level of result.levels ) renumber( level, map );
	return result;

}

function nestedLevels( g0, top, ctx, result ) {

	const n = g0.n;
	const members = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) members[ i ] = i;
	const root = makeNode( members, null );
	ctx.covered = 0;
	const aggregateLevels = ctx.strategy === 'aggregate' && top.chain.length > 0;
	if ( top.K < 2 ) grow( g0, root, g0, null, ctx.levelsBelow + 1, ctx ); // one community: only cut to size
	else if ( aggregateLevels ) grow( g0, root, g0, { partition: top.chain[ 0 ], K: top.chainSizes[ 0 ], q: 0 }, 0, ctx );
	else grow( g0, root, g0, { partition: top.partition, K: top.K, q: 0 }, 0, ctx );
	if ( aggregateLevels ) {

		// the refined partitions of the aggregation levels of the run (the last one is the partition)
		result.levels = top.chain.slice();
		result.levelSizes = top.chainSizes.slice();

	} else if ( root.children ) {

		const t = treeLevels( root, n, ctx.balance );
		result.levels = t.levels;
		result.levelSizes = t.levelSizes;

	}

	const forced = forcedLevel( root, n );
	if ( forced ) {

		// the leaves above maxLeafSize, cut into connected pieces, as a new finest level
		if ( ! aggregateLevels && ! root.children ) {

			result.levels = [];
			result.levelSizes = [];

		}

		result.levels.unshift( forced.level );
		result.levelSizes.unshift( forced.K );
		result.refined = true;

	}

}

// ── public API ───────────────────────────────────────────────────────────────────────────────

function makeContext( n, options ) {

	const {
		seed = 1, resolution = 1, quality = 'modularity', randomness = DEFAULT_RANDOMNESS, iterations = DEFAULT_ITERATIONS,
		strategy = 'recursive', significance = DEFAULT_SIGNIFICANCE, coarsenMinModularity = DEFAULT_COARSEN_MIN_MODULARITY, coarsenResolution = DEFAULT_COARSEN_RESOLUTION,
		minSplitSize = DEFAULT_MIN_SPLIT_SIZE, levelsBelow = DEFAULT_LEVELS_BELOW, levelsAbove = DEFAULT_LEVELS_ABOVE,
		balance = true, refine = null, nodeWeights = null, onProgress = null,
	} = options;
	if ( quality !== 'modularity' && quality !== 'cpm' ) throw new Error( `leiden: unknown quality "${quality}" (modularity | cpm)` );
	if ( strategy !== 'recursive' && strategy !== 'aggregate' && strategy !== 'flat' ) throw new Error( `leiden: unknown strategy "${strategy}" (recursive | aggregate | flat)` );
	return {
		seed: ( seed >>> 0 ) || 1, resolution, quality, randomness: Math.max( 1e-6, randomness ), iterations,
		strategy, significance, coarsenMinModularity, coarsenResolution, minSplitSize: Math.max( 2, minSplitSize ), levelsBelow, levelsAbove, balance,
		maxLeafSize: refine && refine.maxLeafSize > 0 ? refine.maxLeafSize : 0,
		forceDepth: ( refine && refine.maxDepth ) ?? 6,
		subResolution: quality === 'cpm' ? 1 : resolution,
		nodeWeights, onProgress,
		S: createScratch( n ),
		localOf: new Int32Array( n ).fill( - 1 ),
	};

}

/**
 * Leiden with its nested levels. Input: { nodeCount, edges (flat pairs), weights? (per edge) }.
 * @returns {{ levels: Int32Array[], modularity: number[], levelSizes: number[], partition: Int32Array, partitionModularity: number, communities: number, refined: boolean, passes: number, iterations: number, stable: boolean, ms: number }}
 */
export function leiden( graph, options = {} ) {

	const t0 = now();
	const n = graph.nodeCount | 0;
	const ctx = makeContext( n, options );
	// no nodes: no levels (as louvain.js)
	if ( n === 0 ) return { levels: [], modularity: [], levelSizes: [], partition: new Int32Array( 0 ), partitionModularity: 0, communities: 0, refined: false, passes: 0, iterations: 0, stable: true, ms: now() - t0 };
	const g0 = csrFromEdges( n, graph.edges, graph.weights );
	const r = buildLevels( g0, ctx );
	const resolution = ctx.quality === 'cpm' ? 1 : ctx.resolution;
	return {
		levels: r.levels,
		modularity: r.levels.map( ( l ) => csrModularity( g0, l, resolution ) ),
		levelSizes: r.levelSizes,
		partition: r.partition,
		partitionModularity: csrModularity( g0, r.partition, resolution ),
		communities: r.K,
		refined: r.refined,
		passes: r.iterations,
		iterations: r.iterations,
		stable: r.stable,
		ms: now() - t0,
	};

}

/** Modularity of a partition (array of community ids per node) of { nodeCount, edges, weights? }. */
export function modularity( graph, partition, resolution = 1 ) {

	return csrModularity( csrFromEdges( graph.nodeCount | 0, graph.edges, graph.weights ), partition, resolution );

}

/**
 * Indices of the levels to keep: no all-singleton level, no single-cluster level (the implicit
 * root), no level that reduces the cluster count of the kept level below it by less than
 * minLevelRatio (the finer one stays), and at most maxLevels (the level closest in cluster count
 * to the one below it goes first; never the finest, which defines the leaves).
 */
function pruneLevels( levelSizes, n, minLevelRatio, maxLevels ) {

	const kept = [];
	let prevK = n;
	for ( let l = 0; l < levelSizes.length; l ++ ) {

		const K = levelSizes[ l ];
		if ( K <= 1 ) break;
		if ( K >= prevK ) continue;
		if ( kept.length && K * minLevelRatio > prevK ) continue;
		kept.push( l );
		prevK = K;

	}

	while ( maxLevels > 0 && kept.length > maxLevels && kept.length > 1 ) {

		let drop = 1, best = Infinity;
		for ( let q = 1; q < kept.length; q ++ ) {

			const ratio = levelSizes[ kept[ q - 1 ] ] / levelSizes[ kept[ q ] ];
			if ( ratio < best ) {

				best = ratio;
				drop = q;

			}

		}

		kept.splice( drop, 1 );

	}

	return kept;

}

/**
 * Canonical hierarchy for a canonical graph (see format.js): the Leiden levels with oversized
 * leaf communities split, filtered like normalizeLevels (no all-singleton level, no
 * single-cluster level, no level that merges nothing) with modularity kept aligned.
 */
export function leidenHierarchy( g, options = {} ) {

	const { maxLeafSize = DEFAULT_MAX_LEAF_SIZE, minLevelRatio = DEFAULT_MIN_LEVEL_RATIO, maxLevels = DEFAULT_MAX_LEVELS, seed = 1, resolution = 1 } = options;
	const r = leiden( g, { ...options, seed, resolution, refine: maxLeafSize > 0 ? { maxLeafSize, maxDepth: options.maxDepth ?? 6 } : null } );
	const n = g.nodeCount;
	const kept = pruneLevels( r.levelSizes, n, minLevelRatio, maxLevels );
	return {
		method: r.refined ? `leiden (refined, max leaf ${maxLeafSize})` : 'leiden',
		levels: kept.map( ( l ) => Array.from( r.levels[ l ] ) ),
		modularity: kept.map( ( l ) => r.modularity[ l ] ),
		resolution,
		seed,
		refined: r.refined,
		maxLeafSize,
		dropped: r.levels.length - kept.length,
		ms: r.ms,
	};

}

/** Returns g when it already has a hierarchy (unless options.force); otherwise a copy with one. */
export function ensureHierarchy( g, options = {} ) {

	if ( g.hierarchy && g.hierarchy.levels && g.hierarchy.levels.length && ! options.force ) return g;
	const h = leidenHierarchy( g, options );
	return {
		...g,
		hierarchy: h.levels.length ? { method: h.method, levels: h.levels } : null,
		meta: { ...( g.meta || {} ), hierarchy: { method: h.method, modularity: h.modularity, resolution: h.resolution, seed: h.seed, refined: h.refined, maxLeafSize: h.maxLeafSize } },
	};

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}

/** The two phases of one Leiden level on a CSR graph, for the unit tests only (not part of the API). */
export const __internals = { csrFromEdges, createScratch, moveNodesFast, refinePartition };
