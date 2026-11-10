// Clustering front end of the data pipeline: one entry point for every way a canonical graph
// gets its cluster hierarchy, and the "ladder" that shapes a hierarchy for the picture. Pure JS:
// runs in the data worker, on the main thread, or in Node (scripts/generate.mjs, tests).
//
// The hierarchy drives three things at once: the layout (cohesion per cluster, the Barnes–Hut
// tree, the stress seed), hierarchical edge bundling (one control point per ancestor) and the
// colours (hue ranges per subtree). All three want the same shape: 2 to 4 levels, a handful of
// top-level clusters, a branching of roughly 3 to 16 between levels, leaves of at most a few
// hundred nodes. A community detection run gives what the graph has, which is often something
// else: one level of 48 communities under the root (LFR benchmarks), thousands of isolated
// nodes as clusters of their own, a hub whose leaves cannot be cut into connected pieces.
//
// ─── clusterGraph( g, options ) ─────────────────────────────────────────────────────────────
//  1. Dust. Components of one or two nodes (isolated nodes, lone pairs) are taken out first:
//     every community detection method leaves them as clusters of their own at every level.
//     They come back at the end as a tree of "dust" clusters next to the real ones (6-ary, 32
//     nodes per leaf; wider, up to 24 children, when there is so much dust that the leaves
//     would exceed maxLeafSize, and then with several dust clusters at the top).
//  2. Communities. options.method: 'leiden' (default; src/data/leiden.js: connected communities,
//     nested levels that pass a significance test, groupings above the modularity optimum) or
//     'louvain' (src/data/louvain.js: the dendrogram of the passes). Edge weights (g.weights)
//     are used by both (not finite counts 1, negative by its size, all zero means unweighted).
//     options.resolution is the knob of a slider: 1 = the default, larger values give more and
//     smaller communities. It moves in quarter-octave steps (2^( k / 4 ): … 0.5, 0.59, 0.71,
//     0.84, 1, 1.19, 1.41, 1.68, 2 …, from 1/16 to 16; a request is computed at the step next
//     to it on the side of 1). Every resolution is built on ONE hierarchy, the method's own at
//     resolution 1, and on its ANCHOR: the level that carries the method's partition.
//      · below 1 the communities of the anchor are MERGED: modularity at the grid resolutions
//        down to the request on their quotient graph. The finer levels of resolution 1 stay
//        nested inside ("coarse communities at low resolution with the finer ones inside").
//      · above 1 they are SPLIT, each inside itself: where resolution 1 already found finer
//        clusters in a community, by grouping those; elsewhere by the method on the
//        community's subgraph, at a resolution that follows the request. A community keeps the
//        split with the most parts of all steps up to the request, and the split is one level.
//     So the number of communities never falls when the resolution rises (by construction, not
//     by luck of a run), two resolutions can be compared (their communities are unions or
//     parts of the same anchor), there are levels on both sides of the main one wherever the
//     graph has them, and a slider costs one hierarchy plus a small step each (the hierarchy
//     of resolution 1 and the steps computed so far are kept per graph object: anchorState).
//     The leaves stay within maxLeafSize at every resolution. A graph without communities at
//     resolution 1 (a star, a clique) has none at any: its hierarchy stays the cut to size.
//  3. The ladder (options.ladder, default true): levels are ADDED around what step 2 found,
//     never changed.
//      · Too wide at the top (more than maxTop = 16 clusters under the root; when all
//        maxLevels are taken, the level closest to the one below it or a finest level of pairs
//        and triples makes room): the clusters are
//        grouped on their quotient graph (the weight between two clusters, self loops dropped)
//        by modularity, with the resolution searched so that the group count hits the geometric
//        step toward a branching of about 8; where modularity has nothing to hold on to (no
//        connections, a star of clusters) by a size-capped agglomeration: the smallest group
//        joins its strongest neighbour that still fits. Repeated until the top is narrow.
//        These are coarse communities at low resolution with the finer ones nested inside.
//      · A gap between two levels (more than maxBranch = 24 children per parent on average):
//        the same grouping inside every parent.
//      · Still fewer than minDepth = 2 levels (at any resolution): leaves of at least
//        minSplit = 8 nodes are split by Leiden on their own subgraph (parts of fewer than 3
//        nodes join their neighbour) as a new finest level, when that gives at least 1.25 times
//        as many clusters; below resolution 1, where that gives nothing, the communities of
//        resolution 1 are that level.
//      · More than maxLevels = 4 levels: of the two levels closest in cluster count, the
//        coarser one is dropped (never the finest, never the one that carries the communities).
//     Added levels are tagged in info.kinds ('group' / 'split'); they are an aid for the layout
//     and the bundling and claim no statistical significance.
//  Planted partitions (options.planted = { levels, complete?, source? }): a complete hierarchy
//  (generator ground truth) is used as it is. A partial one (a group attribute of the file,
//  subgraphs) becomes the top: the computed levels are intersected with it and kept below.
//
//  Returns { levels, info }:
//    levels   canonical hierarchy levels (plain arrays, finest first, dense ids by first
//             appearance, nested; [] when the graph has no usable structure)
//    info     { method: 'leiden' | 'louvain' | 'planted', detail, resolution (as requested),
//               resolutionUsed (the grid resolution the main level belongs to; below 1 the one
//               its grouping was found at; 1 for a graph without communities), seed,
//               maxLeafSize, ladder: boolean, levels: D, clustersPerLevel: number[],
//               kinds: string[] per level ('community' | 'group' | 'split' | 'planted'),
//               mainLevel (index of the level that carries the communities: the one to show,
//               colour by and count; the coarsest level of a graph without communities; −1
//               without levels),
//               communities (the number that follows the resolution monotonically; ALWAYS
//               clustersPerLevel[ mainLevel ] − dustClusters[ mainLevel ]),
//               dustClusters: number[] (dust clusters per level, 0 without dust),
//               modularity (of the main level, γ = 1), quality (the same level at
//               resolutionUsed), modularityPerLevel: number[] (γ = 1), weighted, dust (nodes),
//               capped, ms }
//
// ─── Other exports ──────────────────────────────────────────────────────────────────────────
//  CLUSTER_DEFAULTS, LADDER_DEFAULTS, RESOLUTION_STEPS (4)
//  resolutionSteps( resolution ) -> signed number of grid steps from 1 a request is computed at
//  mergeCommunities( graph, part, K, steps, seed ) -> { group, count, gamma } | null   the
//      grouping of the clusters of a partition that a resolution below 1 uses
//  applyClustering( g, options ) -> g' : a shallow copy of the canonical graph with `hierarchy`
//      ({ method: info.detail, levels } or null), `clustering` (info) and meta.hierarchy set;
//      what the worker and scripts/generate.mjs call.
//  describeHierarchy( g, levels, base ) -> info for a hierarchy that came with the file
//  shapeLevels( graph, levels, options ) -> { levels, kinds, main }   the ladder alone
//  levelModularity( n, edges, weights, level, gamma = 1 ) -> Q
//  levelSizes( level ) -> cluster count (ids must be dense)

import { leiden, leidenHierarchy, cutToSize, modularity as partitionModularity } from './leiden.js';
import { louvain, louvainHierarchy } from './louvain.js';
import { normalizeLevels } from './format.js';

export const CLUSTER_DEFAULTS = { method: 'leiden', resolution: 1, seed: 1, maxLeafSize: 200, ladder: true };
export const LADDER_DEFAULTS = { maxTop: 16, branch: 8, maxBranch: 24, minDepth: 2, minSplit: 8, maxLevels: 4 };

/** Grid points per doubling of the resolution, and how far the grid goes (2^±4 = 1/16 … 16). */
export const RESOLUTION_STEPS = 4;
const RESOLUTION_MAX_STEPS = 16;
/** A community is not split into parts of fewer nodes than this (resolutions above 1). */
const MIN_PART = 3;
/** Tuning options of leiden.js that clusterGraph passes on (they change the hierarchy of resolution 1). */
const LEIDEN_KEYS = [ 'significance', 'iterations', 'minSplitSize', 'levelsBelow', 'levelsAbove', 'coarsenMinModularity', 'coarsenResolution', 'balance', 'queueCap', 'strategy', 'randomness', 'minLevelRatio', 'maxLevels' ];

const DUST_BIN = 32; // nodes per dust cluster at the finest level
const DUST_FAN = 6;
const MAX_DUST_FAN = 24; // children per dust cluster when there is more dust than the 6-ary tree holds

// ── small helpers ────────────────────────────────────────────────────────────────────────────

export function levelSizes( level ) {

	let m = - 1;
	for ( let i = 0; i < level.length; i ++ ) if ( level[ i ] > m ) m = level[ i ];
	return m + 1;

}

function densify( level ) {

	const map = new Map();
	const out = new Int32Array( level.length );
	for ( let i = 0; i < level.length; i ++ ) {

		let id = map.get( level[ i ] );
		if ( id === undefined ) {

			id = map.size;
			map.set( level[ i ], id );

		}

		out[ i ] = id;

	}

	return { level: out, K: map.size };

}

/** Modularity Q = Σ_c [ in_c / m − γ ( tot_c / 2m )² ] of a dense partition, in one pass over the edges. */
export function levelModularity( n, edges, weights, level, gamma = 1 ) {

	const K = levelSizes( level );
	const tot = new Float64Array( Math.max( 1, K ) );
	let inside = 0, m2 = 0;
	for ( let e = 0, E = edges.length >> 1; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		const w = weights ? weights[ e ] : 1;
		if ( a === b ) {

			// a self loop counts twice in the degree and is always internal
			tot[ level[ a ] ] += 2 * w;
			inside += 2 * w;
			m2 += 2 * w;
			continue;

		}

		tot[ level[ a ] ] += w;
		tot[ level[ b ] ] += w;
		m2 += 2 * w;
		if ( level[ a ] === level[ b ] ) inside += 2 * w;

	}

	if ( ! ( m2 > 0 ) ) return 0;
	let q = inside / m2;
	for ( let c = 0; c < K; c ++ ) q -= gamma * ( tot[ c ] / m2 ) ** 2;
	return q;

}

/** Weighted quotient graph of a dense partition: the connections between its clusters. */
function quotient( edges, weights, level, K, parent = null ) {

	const acc = new Map();
	for ( let e = 0, E = edges.length >> 1; e < E; e ++ ) {

		let a = level[ edges[ 2 * e ] ], b = level[ edges[ 2 * e + 1 ] ];
		if ( a === b ) continue;
		if ( parent && parent[ edges[ 2 * e ] ] !== parent[ edges[ 2 * e + 1 ] ] ) continue;
		if ( a > b ) {

			const t = a; a = b; b = t;

		}

		const key = a * K + b;
		acc.set( key, ( acc.get( key ) || 0 ) + ( weights ? weights[ e ] : 1 ) );

	}

	const qEdges = new Uint32Array( 2 * acc.size ), qWeights = new Float64Array( acc.size );
	let k = 0;
	for ( const [ key, w ] of acc ) {

		qEdges[ 2 * k ] = Math.floor( key / K );
		qEdges[ 2 * k + 1 ] = key % K;
		qWeights[ k ++ ] = w;

	}

	return { edges: qEdges, weights: qWeights };

}

// ── grouping clusters ────────────────────────────────────────────────────────────────────────

/** Binary min-heap of ( key, id ) pairs with lazy deletion left to the caller. */
class MinHeap {

	constructor() {

		this.keys = [];
		this.ids = [];

	}

	get size() {

		return this.ids.length;

	}

	push( key, id ) {

		const keys = this.keys, ids = this.ids;
		let i = ids.length;
		keys.push( key );
		ids.push( id );
		while ( i > 0 ) {

			const p = ( i - 1 ) >> 1;
			if ( keys[ p ] < key || ( keys[ p ] === key && ids[ p ] <= id ) ) break;
			keys[ i ] = keys[ p ]; ids[ i ] = ids[ p ];
			i = p;

		}

		keys[ i ] = key; ids[ i ] = id;

	}

	pop() {

		const keys = this.keys, ids = this.ids;
		const top = [ keys[ 0 ], ids[ 0 ] ];
		const key = keys.pop(), id = ids.pop();
		const n = ids.length;
		if ( n > 0 ) {

			let i = 0;
			for ( ;; ) {

				let c = 2 * i + 1;
				if ( c >= n ) break;
				if ( c + 1 < n && ( keys[ c + 1 ] < keys[ c ] || ( keys[ c + 1 ] === keys[ c ] && ids[ c + 1 ] < ids[ c ] ) ) ) c ++;
				if ( key < keys[ c ] || ( key === keys[ c ] && id <= ids[ c ] ) ) break;
				keys[ i ] = keys[ c ]; ids[ i ] = ids[ c ];
				i = c;

			}

			keys[ i ] = key; ids[ i ] = id;

		}

		return top;

	}

}

/**
 * Size-capped agglomeration of the K nodes of a quotient graph, starting from `group` (dense
 * ids, k groups): the smallest group joins the neighbour it is most strongly connected to among
 * those that keep the union within the cap (1.5 times the mean target size); without such a
 * neighbour, the smallest other group that fits; without that, its strongest neighbour or the
 * smallest group. Until at most `target` groups are left. Deterministic.
 */
function mergeDown( K, qEdges, qWeights, sizes, group, k, target ) {

	if ( k <= target ) return { group, count: k };
	const gsize = new Float64Array( k );
	let total = 0;
	for ( let c = 0; c < K; c ++ ) {

		gsize[ group[ c ] ] += sizes[ c ];
		total += sizes[ c ];

	}

	const adj = Array.from( { length: k }, () => new Map() );
	for ( let e = 0; e < qWeights.length; e ++ ) {

		const a = group[ qEdges[ 2 * e ] ], b = group[ qEdges[ 2 * e + 1 ] ];
		if ( a === b ) continue;
		adj[ a ].set( b, ( adj[ a ].get( b ) || 0 ) + qWeights[ e ] );
		adj[ b ].set( a, ( adj[ b ].get( a ) || 0 ) + qWeights[ e ] );

	}

	const cap = Math.ceil( ( 1.5 * total ) / target );
	const alive = new Uint8Array( k ).fill( 1 );
	const into = new Int32Array( k );
	for ( let g = 0; g < k; g ++ ) into[ g ] = g;
	const heap = new MinHeap();
	for ( let g = 0; g < k; g ++ ) heap.push( gsize[ g ], g );
	const popValid = () => {

		while ( heap.size ) {

			const [ key, g ] = heap.pop();
			if ( alive[ g ] && key === gsize[ g ] ) return g;

		}

		return - 1;

	};

	let count = k;
	while ( count > target ) {

		const g = popValid();
		if ( g < 0 ) break;
		let best = - 1, bestW = 0, strongest = - 1, strongestW = 0;
		for ( const [ h, w ] of adj[ g ] ) {

			if ( w > strongestW || ( w === strongestW && h < strongest ) ) {

				strongest = h; strongestW = w;

			}

			if ( gsize[ g ] + gsize[ h ] > cap ) continue;
			if ( w > bestW || ( w === bestW && ( best < 0 || gsize[ h ] < gsize[ best ] || ( gsize[ h ] === gsize[ best ] && h < best ) ) ) ) {

				best = h; bestW = w;

			}

		}

		if ( best < 0 ) {

			const other = popValid(); // the next smallest group
			if ( other < 0 ) {

				heap.push( gsize[ g ], g );
				break;

			}

			if ( gsize[ g ] + gsize[ other ] <= cap || strongest < 0 ) best = other;
			else {

				best = strongest;
				heap.push( gsize[ other ], other );

			}

		}

		// merge g into best
		alive[ g ] = 0;
		into[ g ] = best;
		gsize[ best ] += gsize[ g ];
		const mine = adj[ g ], theirs = adj[ best ];
		theirs.delete( g );
		for ( const [ h, w ] of mine ) {

			if ( h === best ) continue;
			theirs.set( h, ( theirs.get( h ) || 0 ) + w );
			const back = adj[ h ];
			back.delete( g );
			back.set( best, ( back.get( best ) || 0 ) + w );

		}

		adj[ g ] = null;
		heap.push( gsize[ best ], best );
		count --;

	}

	const find = ( g ) => {

		while ( into[ g ] !== g ) g = into[ g ];
		return g;

	};

	const out = new Int32Array( K );
	for ( let c = 0; c < K; c ++ ) out[ c ] = find( group[ c ] );
	const d = densify( out );
	return { group: d.level, count: d.K };

}

/**
 * Groups the K clusters of a quotient graph into about `target` groups: by modularity on the
 * quotient graph where that works (γ = 1 first, then a search on log γ for the group count),
 * finished or replaced by the size-capped agglomeration (mergeDown).
 * @returns {{ group: Int32Array, count: number, how: 'modularity' | 'agglomeration' | 'none' }}
 */
function groupClusters( K, qEdges, qWeights, sizes, target, seed ) {

	target = Math.max( 2, Math.round( target ) );
	const identity = new Int32Array( K );
	for ( let c = 0; c < K; c ++ ) identity[ c ] = c;
	if ( K <= target ) return { group: identity, count: K, how: 'none' };
	const lo = target / 1.6, hi = target * 1.6;
	let best = null;
	if ( qWeights.length > 0 ) {

		// clusters without connections never join a modularity group: they are not counted here
		// and are packed by the agglomeration below
		const connected = new Uint8Array( K );
		for ( let k = 0; k < qEdges.length; k ++ ) connected[ qEdges[ k ] ] = 1;
		let lone = 0;
		for ( let c = 0; c < K; c ++ ) if ( ! connected[ c ] ) lone ++;
		if ( K - lone > lo ) {

			const q = { nodeCount: K, edges: qEdges, weights: qWeights };
			let a = Math.log( 1 / 64 ), b = Math.log( 64 ), x = 0;
			for ( let it = 0; it < 9; it ++ ) {

				const r = leiden( q, { strategy: 'flat', resolution: Math.exp( x ), seed, iterations: 4 } );
				const count = r.communities - lone;
				const score = Math.abs( Math.log( Math.max( 1, count ) / target ) );
				if ( ! best || score < best.score ) best = { count, score, part: r.partition, total: r.communities };
				// the natural grouping (γ = 1) is taken when it is anywhere near; a searched one
				// has to come close, so the count does not wander from seed to seed
				if ( it === 0 ? count >= lo && count <= hi : score < Math.log( 1.25 ) ) break;
				if ( count > target ) b = x;
				else a = x;
				x = ( a + b ) / 2;

			}

		}

	}

	if ( best && best.count >= 2 && best.count >= lo / 1.5 && best.count <= hi * 2 ) {

		const d = densify( best.part );
		if ( d.K <= hi ) return { group: d.level, count: d.K, how: 'modularity' };
		const r = mergeDown( K, qEdges, qWeights, sizes, d.level, d.K, Math.max( 2, Math.round( Math.min( hi, target * 1.3 ) ) ) );
		return { group: r.group, count: r.count, how: 'modularity' };

	}

	const r = mergeDown( K, qEdges, qWeights, sizes, identity, K, target );
	return { group: r.group, count: r.count, how: 'agglomeration' };

}

/** Cluster count the next coarser level should have, given how many levels may still be added. */
function coarserTarget( K, levelsLeft, branch ) {

	const steps = Math.max( 1, Math.min( levelsLeft, Math.ceil( Math.log( K ) / Math.log( branch ) - 1e-9 ) - 1 ) );
	return Math.max( 2, Math.pow( K, steps / ( steps + 1 ) ) );

}

function sizesOf( level, K ) {

	const s = new Float64Array( K );
	for ( let i = 0; i < level.length; i ++ ) s[ level[ i ] ] ++;
	return s;

}

/** A coarser level above `level` (dense, K clusters) with about `target` clusters, or null. */
function groupLevel( graph, level, K, target, seed ) {

	const q = quotient( graph.edges, graph.weights, level, K );
	const r = groupClusters( K, q.edges, q.weights, sizesOf( level, K ), target, seed );
	if ( r.count < 2 || r.count * 1.5 > K ) return null;
	const out = new Int32Array( level.length );
	for ( let i = 0; i < out.length; i ++ ) out[ i ] = r.group[ level[ i ] ];
	return { level: out, K: r.count };

}

/** A level between `fine` and `coarse`: the fine clusters grouped inside every coarse cluster. */
function groupBetween( graph, fine, Kf, coarse, Kc, opts, seed ) {

	const parentOf = new Int32Array( Kf ).fill( - 1 );
	for ( let i = 0; i < fine.length; i ++ ) parentOf[ fine[ i ] ] = coarse ? coarse[ i ] : 0;
	const kids = Array.from( { length: Kc }, () => [] );
	for ( let c = 0; c < Kf; c ++ ) kids[ parentOf[ c ] ].push( c );
	const sizes = sizesOf( fine, Kf );
	const q = quotient( graph.edges, graph.weights, fine, Kf, coarse );
	// quotient edges bucketed by parent
	const edgesOf = Array.from( { length: Kc }, () => [] );
	for ( let e = 0; e < q.weights.length; e ++ ) edgesOf[ parentOf[ q.edges[ 2 * e ] ] ].push( e );
	const local = new Int32Array( Kf );
	const groupOf = new Int32Array( Kf );
	let next = 0;
	for ( let p = 0; p < Kc; p ++ ) {

		const list = kids[ p ];
		if ( list.length <= opts.maxTop ) {

			for ( const c of list ) groupOf[ c ] = next ++;
			continue;

		}

		list.forEach( ( c, k ) => ( local[ c ] = k ) );
		const es = edgesOf[ p ];
		const le = new Uint32Array( 2 * es.length ), lw = new Float64Array( es.length );
		es.forEach( ( e, k ) => {

			le[ 2 * k ] = local[ q.edges[ 2 * e ] ];
			le[ 2 * k + 1 ] = local[ q.edges[ 2 * e + 1 ] ];
			lw[ k ] = q.weights[ e ];

		} );
		const ls = Float64Array.from( list, ( c ) => sizes[ c ] );
		const r = groupClusters( list.length, le, lw, ls, Math.sqrt( list.length ), ( seed + 31 * p ) >>> 0 );
		for ( let k = 0; k < list.length; k ++ ) groupOf[ list[ k ] ] = next + r.group[ k ];
		next += r.count;

	}

	if ( next * 1.5 > Kf || next < Kc * 1.5 ) return null;
	const out = new Int32Array( fine.length );
	for ( let i = 0; i < out.length; i ++ ) out[ i ] = groupOf[ fine[ i ] ];
	return { level: out, K: next };

}

/** A finer level below `level`: every leaf of at least minSplit nodes split by Leiden on its own subgraph (its own null model, resolution 1). */
function splitLeaves( graph, level, K, opts, seed ) {

	const n = level.length, E = graph.edges.length >> 1;
	const size = new Int32Array( K );
	for ( let i = 0; i < n; i ++ ) size[ level[ i ] ] ++;
	// members and internal edges per cluster (CSR)
	const mstart = new Int32Array( K + 1 ), estart = new Int32Array( K + 1 );
	for ( let c = 0; c < K; c ++ ) mstart[ c + 1 ] = mstart[ c ] + size[ c ];
	const members = new Int32Array( n ), localOf = new Int32Array( n );
	{

		const fill = mstart.slice( 0, K );
		for ( let i = 0; i < n; i ++ ) {

			localOf[ i ] = fill[ level[ i ] ] - mstart[ level[ i ] ];
			members[ fill[ level[ i ] ] ++ ] = i;

		}

	}

	for ( let e = 0; e < E; e ++ ) {

		const c = level[ graph.edges[ 2 * e ] ];
		if ( c === level[ graph.edges[ 2 * e + 1 ] ] && graph.edges[ 2 * e ] !== graph.edges[ 2 * e + 1 ] ) estart[ c + 1 ] ++;

	}

	for ( let c = 0; c < K; c ++ ) estart[ c + 1 ] += estart[ c ];
	const inner = new Int32Array( estart[ K ] );
	{

		const fill = estart.slice( 0, K );
		for ( let e = 0; e < E; e ++ ) {

			const c = level[ graph.edges[ 2 * e ] ];
			if ( c === level[ graph.edges[ 2 * e + 1 ] ] && graph.edges[ 2 * e ] !== graph.edges[ 2 * e + 1 ] ) inner[ fill[ c ] ++ ] = e;

		}

	}

	const out = new Int32Array( n );
	let next = 0;
	for ( let c = 0; c < K; c ++ ) {

		const m = size[ c ], me = estart[ c + 1 ] - estart[ c ];
		let parts = null, count = 1;
		if ( m >= opts.minSplit && me > 0 ) {

			const edges = new Uint32Array( 2 * me );
			const weights = graph.weights ? new Float64Array( me ) : null;
			for ( let k = 0; k < me; k ++ ) {

				const e = inner[ estart[ c ] + k ];
				edges[ 2 * k ] = localOf[ graph.edges[ 2 * e ] ];
				edges[ 2 * k + 1 ] = localOf[ graph.edges[ 2 * e + 1 ] ];
				if ( weights ) weights[ k ] = graph.weights[ e ];

			}

			const r = leiden( { nodeCount: m, edges, weights }, { strategy: 'flat', resolution: 1, seed: ( seed + 131 * c ) >>> 0 } );
			if ( r.communities >= 2 && r.communities < m ) {

				parts = r.partition.slice();
				count = absorbSmall( m, edges, weights, parts, r.communities, 3 );

			}

		}

		for ( let k = 0; k < m; k ++ ) out[ members[ mstart[ c ] + k ] ] = next + ( parts && count > 1 ? parts[ k ] : 0 );
		next += parts && count > 1 ? count : 1;

	}

	return next >= 1.25 * K ? { level: out, K: next } : null;

}

/** Parts with fewer than minSize nodes join the part they are most connected to; returns the new count (dense ids). */
function absorbSmall( m, edges, weights, parts, K, minSize ) {

	for ( let round = 0; round < 4; round ++ ) {

		const size = new Int32Array( K );
		for ( let i = 0; i < m; i ++ ) size[ parts[ i ] ] ++;
		let small = false;
		for ( let c = 0; c < K; c ++ ) if ( size[ c ] > 0 && size[ c ] < minSize ) small = true;
		if ( ! small ) break;
		const link = new Map(); // small part * K + other part -> weight
		for ( let e = 0; e < edges.length >> 1; e ++ ) {

			const a = parts[ edges[ 2 * e ] ], b = parts[ edges[ 2 * e + 1 ] ];
			if ( a === b ) continue;
			const w = weights ? weights[ e ] : 1;
			if ( size[ a ] < minSize ) link.set( a * K + b, ( link.get( a * K + b ) || 0 ) + w );
			if ( size[ b ] < minSize ) link.set( b * K + a, ( link.get( b * K + a ) || 0 ) + w );

		}

		const target = new Int32Array( K ).fill( - 1 ), bestW = new Float64Array( K );
		for ( const [ key, w ] of link ) {

			const a = Math.floor( key / K ), b = key % K;
			if ( w > bestW[ a ] || ( w === bestW[ a ] && b < target[ a ] ) ) {

				bestW[ a ] = w;
				target[ a ] = b;

			}

		}

		// two small parts that point at each other merge once (the lower id wins)
		let changed = false;
		for ( let i = 0; i < m; i ++ ) {

			const a = parts[ i ], t = target[ a ];
			if ( t < 0 ) continue;
			if ( target[ t ] === a && t > a ) continue;
			parts[ i ] = t;
			changed = true;

		}

		if ( ! changed ) break;

	}

	const d = densify( parts );
	parts.set( d.level );
	return d.K;

}

// ── the ladder ───────────────────────────────────────────────────────────────────────────────

/**
 * Adds levels around `levels` (finest first, nested, dense ids) so the hierarchy has the shape
 * the layout and the bundling want (see the header). The given levels are kept as they are,
 * except when more than maxLevels are left at the end.
 *
 * @param {{ nodeCount: number, edges: ArrayLike<number>, weights?: ArrayLike<number>|null }} graph
 * @param {ArrayLike<number>[]} levels
 * @param {object} [options] LADDER_DEFAULTS, seed, kinds (tags of the given levels),
 *   maxLeafSize (a finest level that keeps the leaves within it is never dropped),
 *   main (index of a level that is never dropped: the one that carries the communities),
 *   fallback (a level nested in levels[0]: the finest level when the hierarchy would otherwise
 *   have fewer than minDepth levels and the leaves cannot be split)
 * @returns {{ levels: Int32Array[], kinds: string[], main: number }}
 */
export function shapeLevels( graph, levels, options = {} ) {

	const opts = { ...LADDER_DEFAULTS, ...options };
	const seed = ( options.seed >>> 0 ) || 1;
	const n = graph.nodeCount;
	const lv = levels.map( ( l ) => Int32Array.from( l ) );
	const Ks = lv.map( levelSizes );
	const kinds = options.kinds ? options.kinds.slice() : lv.map( () => 'community' );
	let main = Number.isInteger( options.main ) && options.main >= 0 && options.main < lv.length ? options.main : - 1;
	if ( lv.length === 0 || n === 0 ) return { levels: lv, kinds, main };

	const removeAt = ( l ) => {

		lv.splice( l, 1 ); Ks.splice( l, 1 ); kinds.splice( l, 1 );
		if ( l < main ) main --;

	};

	const insertAt = ( l, level, K, kind ) => {

		lv.splice( l, 0, level ); Ks.splice( l, 0, K ); kinds.splice( l, 0, kind );
		if ( main >= l ) main ++;

	};

	// Of the two levels closest in cluster count (a step below `limit`), the one that can go:
	// the coarser one, or the finer one when the coarser carries the communities; never the
	// finest (it defines the leaves) and never the main level.
	const weakest = ( limit ) => {

		let drop = - 1, best = limit;
		for ( let l = 1; l < lv.length; l ++ ) {

			const ratio = Ks[ l - 1 ] / Ks[ l ];
			const which = l !== main ? l : ( l - 1 > 0 ? l - 1 : - 1 );
			if ( which >= 0 && ratio < best ) {

				best = ratio;
				drop = which;

			}

		}

		return drop;

	};

	const dropWeakest = () => {

		// not a step of 2 or more (that is structure, not a near-duplicate)
		const drop = weakest( 2 );
		if ( drop < 0 ) return false;
		removeAt( drop );
		return true;

	};

	const dropTinyLeaves = () => {

		// a finest level of pairs and triples (Louvain's first pass on a sparse graph) carries
		// nothing the picture needs: it makes room for a level at the top
		// (unless it is what keeps the leaves within the size limit)
		if ( lv.length < 2 || main === 0 || n / Ks[ 0 ] >= opts.minSplit / 2 ) return false;
		if ( opts.maxLeafSize > 0 ) {

			const size = new Int32Array( Ks[ 1 ] );
			for ( let i = 0; i < n; i ++ ) if ( ++ size[ lv[ 1 ][ i ] ] > opts.maxLeafSize ) return false;

		}

		removeAt( 0 );
		return true;

	};

	// 1. the top: at most maxTop clusters under the root
	for ( let guard = 0; guard < 8 && Ks[ Ks.length - 1 ] > opts.maxTop; guard ++ ) {

		if ( lv.length >= opts.maxLevels && ! dropWeakest() && ! dropTinyLeaves() ) break;
		const top = lv.length - 1;
		const g = groupLevel( graph, lv[ top ], Ks[ top ], coarserTarget( Ks[ top ], opts.maxLevels - lv.length, opts.branch ), seed + 7 * guard );
		if ( ! g ) break;
		lv.push( g.level ); Ks.push( g.K ); kinds.push( 'group' );

	}

	// 2. gaps between levels, from the top down
	for ( let l = lv.length - 2; l >= 0; l -- ) {

		for ( let guard = 0; guard < 3 && Ks[ l ] / Ks[ l + 1 ] > opts.maxBranch && lv.length < opts.maxLevels; guard ++ ) {

			const g = groupBetween( graph, lv[ l ], Ks[ l ], lv[ l + 1 ], Ks[ l + 1 ], opts, seed + 13 * l + guard );
			if ( ! g ) break;
			insertAt( l + 1, g.level, g.K, 'group' );

		}

	}

	// 3. depth: a finer level when there are fewer than minDepth (at every resolution: one level
	//    means one control point per bundled edge and a flat Barnes-Hut tree)
	if ( lv.length < opts.minDepth && lv.length < opts.maxLevels ) {

		const s = splitLeaves( graph, lv[ 0 ], Ks[ 0 ], opts, seed + 3 );
		const f = options.fallback || null;
		if ( s ) insertAt( 0, s.level, s.K, 'split' );
		else if ( f && f.length === n && levelSizes( f ) >= 1.2 * Ks[ 0 ] && levelSizes( f ) < n ) insertAt( 0, Int32Array.from( f ), levelSizes( f ), options.fallbackKind || 'community' );

	}

	// 4. at most maxLevels (the given levels may exceed it)
	while ( lv.length > opts.maxLevels ) {

		const drop = weakest( Infinity );
		if ( drop < 0 ) break;
		removeAt( drop );

	}

	return { levels: lv, kinds, main };

}

// ── dust ─────────────────────────────────────────────────────────────────────────────────────

/** Nodes in components of one or two nodes: Uint8Array flag per node, and their count. */
function findDust( n, edges ) {

	const degree = new Uint32Array( n ), other = new Int32Array( n ).fill( - 1 );
	for ( let k = 0; k < edges.length; k += 2 ) {

		const a = edges[ k ], b = edges[ k + 1 ];
		if ( a === b ) continue;
		degree[ a ] ++; degree[ b ] ++;
		other[ a ] = b; other[ b ] = a;

	}

	const dust = new Uint8Array( n );
	let count = 0;
	for ( let i = 0; i < n; i ++ ) {

		if ( degree[ i ] === 0 || ( degree[ i ] === 1 && degree[ other[ i ] ] === 1 ) ) {

			dust[ i ] = 1;
			count ++;

		}

	}

	return { dust, count, other, degree };

}

/**
 * Dust clusters for D levels: the dust nodes in node order (a pair stays together) cut into
 * runs, as a tree of their own next to the real clusters.
 *  · The finest level has ceil( count / DUST_BIN ) runs of about DUST_BIN nodes while a 6-ary
 *    tree with one top cluster holds them ( DUST_FAN^( D − 1 ) runs ); beyond that the runs grow,
 *    up to maxLeafSize nodes; beyond that there are more runs: no dust leaf is larger than
 *    maxLeafSize (0 = no limit), however many isolated nodes a file brings. The hierarchy is the
 *    Barnes-Hut tree of the layout: a leaf of ten thousand nodes would be ten thousand squared
 *    pair interactions per step.
 *  · Every coarser level joins `fan` runs: DUST_FAN while that ends in one top cluster, else
 *    the D-th root of the number of runs (a balanced tree, at most MAX_DUST_FAN children; with
 *    more runs than that holds, the top level has several dust clusters).
 * Returns per level an Int32Array( n ) with the run index of a dust node, −1 elsewhere, and
 * the number of runs per level.
 */
function dustLevels( n, D, info, maxLeafSize ) {

	const { dust, count, other, degree } = info;
	// order: node order, the partner of a pair right behind its first member
	const order = new Int32Array( count );
	const seen = new Uint8Array( n );
	let k = 0;
	for ( let i = 0; i < n; i ++ ) {

		if ( ! dust[ i ] || seen[ i ] ) continue;
		seen[ i ] = 1;
		order[ k ++ ] = i;
		if ( degree[ i ] === 1 && ! seen[ other[ i ] ] ) {

			seen[ other[ i ] ] = 1;
			order[ k ++ ] = other[ i ];

		}

	}

	const out = [], counts = [];
	const tree = Math.pow( DUST_FAN, D - 1 ); // finest runs of the 6-ary tree with one top cluster
	let runs = Math.max( 1, Math.min( Math.ceil( count / DUST_BIN ), tree ) );
	// (a run boundary moves by one node to keep a pair together: one node of slack)
	if ( maxLeafSize > 1 && count / runs > maxLeafSize - 1 ) runs = Math.ceil( count / ( maxLeafSize - 1 ) );
	const fan = runs <= tree ? DUST_FAN : Math.min( MAX_DUST_FAN, Math.max( DUST_FAN, Math.ceil( Math.pow( runs, 1 / D ) ) ) );
	// run boundaries of the finest level never split a pair; coarser runs are unions of finer ones
	let bounds = new Int32Array( runs + 1 );
	for ( let r = 1; r <= runs; r ++ ) {

		let at = Math.round( ( r * count ) / runs );
		if ( at < count && at > 0 && degree[ order[ at ] ] === 1 && other[ order[ at ] ] === order[ at - 1 ] ) at ++;
		bounds[ r ] = Math.min( count, Math.max( at, bounds[ r - 1 ] ) );

	}

	bounds[ runs ] = count;
	for ( let l = 0; l < D; l ++ ) {

		const level = new Int32Array( n ).fill( - 1 );
		for ( let r = 0; r < runs; r ++ ) for ( let q = bounds[ r ]; q < bounds[ r + 1 ]; q ++ ) level[ order[ q ] ] = r;
		out.push( level );
		counts.push( runs );
		if ( l + 1 < D ) {

			// (the 6-ary tree keeps its old shape: at most DUST_FAN^( levels above ) runs)
			const next = Math.max( 1, fan === DUST_FAN && runs <= Math.pow( DUST_FAN, D - 1 - l ) ? Math.min( Math.ceil( runs / DUST_FAN ), Math.pow( DUST_FAN, D - 2 - l ) ) : Math.ceil( runs / fan ) );
			const nb = new Int32Array( next + 1 );
			for ( let r = 1; r <= next; r ++ ) nb[ r ] = bounds[ Math.min( runs, Math.round( ( r * runs ) / next ) ) ];
			nb[ next ] = count;
			bounds = nb;
			runs = next;

		}

	}

	return { levels: out, counts };

}

// ── front end ────────────────────────────────────────────────────────────────────────────────

/**
 * The resolution a request is computed at: the grid point 2^( k / RESOLUTION_STEPS ) next to it
 * on the side of 1 (a tenth of a step of tolerance, so 0.35, 0.7, 1.4 and 2.8 mean 2^±0.5 and
 * 2^±1.5). Returns the signed number of steps k (0 = the default resolution).
 */
export function resolutionSteps( resolution ) {

	const x = RESOLUTION_STEPS * Math.log2( resolution );
	const k = x >= 0 ? Math.floor( x + 0.1 ) : - Math.floor( - x + 0.1 );
	return Math.max( - RESOLUTION_MAX_STEPS, Math.min( RESOLUTION_MAX_STEPS, k ) ) || 0;

}

const gridResolution = ( k ) => Math.pow( 2, k / RESOLUTION_STEPS );

/** The method's own partition of a graph at resolution γ: { count, partition }. */
function methodPartition( graph, opts, gamma, seed ) {

	if ( opts.method === 'louvain' ) {

		const r = louvain( graph, { resolution: gamma, seed } );
		return r.levels.length ? { count: r.levelSizes[ r.levelSizes.length - 1 ], partition: r.levels[ r.levels.length - 1 ] } : { count: Math.min( 1, graph.nodeCount ), partition: new Int32Array( graph.nodeCount ) };

	}

	const r = leiden( graph, { strategy: 'flat', resolution: gamma, seed, queueCap: opts.queueCap } );
	return { count: r.communities, partition: r.partition };

}

/** The method's own hierarchy at resolution 1 (leiden.js / louvain.js as they are). */
function baseAt( graph, opts ) {

	const common = { seed: opts.seed, resolution: 1, maxLeafSize: opts.maxLeafSize > 0 ? Math.round( opts.maxLeafSize ) : 0, onProgress: opts.onProgress };
	if ( opts.method === 'louvain' ) {

		const h = louvainHierarchy( graph, common );
		return { levels: h.levels, detail: h.method, mainCount: h.communities, capped: 0, refined: !! h.refined };

	}

	if ( opts.method !== 'leiden' ) throw new Error( `Clustering: unknown method "${opts.method}" (leiden | louvain)` );
	const extra = {};
	for ( const key of LEIDEN_KEYS ) if ( opts[ key ] !== undefined ) extra[ key ] = opts[ key ];
	const h = leidenHierarchy( graph, { ...common, ...extra } );
	return { levels: h.levels, detail: h.method, mainCount: h.communities, capped: h.capped, refined: !! h.refined };

}

/** Edge weights the clustering can use: finite and not negative; null when there are none or all are 0. */
function usableWeights( weights, E ) {

	if ( ! weights ) return null;
	let bad = false, positive = false;
	for ( let e = 0; e < E; e ++ ) {

		const w = weights[ e ];
		if ( ! ( w >= 0 ) || w === Infinity ) bad = true;
		else if ( w > 0 ) positive = true;

	}

	if ( ! bad ) return positive ? weights : null;
	// as cleanEdges of format.js: a weight that is not a number counts 1, a negative one by its size
	const out = new Float64Array( E );
	for ( let e = 0; e < E; e ++ ) {

		const w = Math.abs( weights[ e ] );
		out[ e ] = w < Infinity ? w : 1; // (NaN fails the comparison too)

	}

	return out;

}

/**
 * Everything about a graph that does not depend on the resolution, kept for the next request
 * (a slider sends many): the dust, the graph without it, and the hierarchy of resolution 1
 * with its ANCHOR, the level that carries the method's own partition (for Leiden the level of
 * the balanced tree closest to it in cluster count). Every other resolution is built on the
 * anchor: below 1 its communities are merged, above 1 they are split, so the community counts
 * of two resolutions can be compared and a level always shows them.
 * The memo is keyed by the graph object, which must not be changed in place between calls.
 */
const anchorMemo = new WeakMap();

function anchorState( g, opts ) {

	const key = [ 'method', 'seed', 'maxLeafSize', ...LEIDEN_KEYS ].map( ( k ) => `${k}=${opts[ k ]}` ).join( '|' );
	const kept = anchorMemo.get( g );
	if ( kept && kept.key === key && kept.edges === g.edges && kept.weights === g.weights && kept.n === g.nodeCount ) return kept.state;

	const n = g.nodeCount, E = g.edges.length >> 1;
	const weights = usableWeights( g.weights, E );
	const whole = weights === ( g.weights || null ) ? g : { nodeCount: n, edges: g.edges, weights };
	// 1. dust: components of one or two nodes do not take part in the clustering
	const dustInfo = findDust( n, g.edges );
	const useDust = dustInfo.count >= 3 && dustInfo.count < n;
	let work = whole, map = null;
	if ( useDust ) {

		const keep = new Uint8Array( n );
		for ( let i = 0; i < n; i ++ ) keep[ i ] = dustInfo.dust[ i ] ? 0 : 1;
		( { graph: work, map } = inducedGraph( whole, keep, n - dustInfo.count ) );

	}

	// 2. the hierarchy of resolution 1 and its anchor
	const base = baseAt( work, opts );
	const levels = base.levels.map( ( l ) => Int32Array.from( l ) );
	const Ks = levels.map( levelSizes );
	let anchor = - 1;
	if ( base.mainCount >= 2 ) {

		let best = Infinity;
		Ks.forEach( ( K, l ) => {

			const d = Math.abs( Math.log( K / base.mainCount ) );
			if ( d < best ) {

				best = d;
				anchor = l;

			}

		} );

	}

	// (without communities the levels are cuts to size: a star, a clique)
	const kinds = levels.map( () => ( anchor >= 0 ? 'community' : 'split' ) );
	// `forced`: the finest level is the method's cut of oversized leaves, not structure it found
	const state = { whole, dustInfo, useDust, work, map, levels, Ks, kinds, anchor, forced: base.refined && levels.length > 0, detail: base.detail, capped: base.capped || 0, up: null };
	anchorMemo.set( g, { key, edges: g.edges, weights: g.weights, n, state } );
	return state;

}

/**
 * The method's hierarchy at the requested resolution, built on the anchor (see anchorState).
 * The resolution moves on a grid of RESOLUTION_STEPS (4) points per doubling; the number of
 * communities never falls when it rises, by construction, and the communities are always one
 * level of the result (`main`).
 *
 *  γ = 1   the method as it is (leiden.js / louvain.js defaults); main = the anchor.
 *  γ < 1   the communities of the anchor are MERGED, never cut again (mergedHierarchy).
 *  γ > 1   the communities of the anchor are SPLIT, each inside itself (splitHierarchy).
 *  A graph without communities at resolution 1 (a star, a clique: fewer than two) has none at
 *  any resolution: its hierarchy stays the one of resolution 1.
 *
 * Returns { levels, kinds, main, detail, gamma (the grid resolution the main level belongs to;
 * below 1 the one its grouping was found at; 1 for a graph without communities), fallback (a
 * finer level for the ladder, or null) }.
 */
function baseHierarchy( st, opts, maxTop = Infinity ) {

	const steps = resolutionSteps( opts.resolution );
	const same = { levels: st.levels, kinds: st.kinds, main: st.anchor, detail: st.detail, gamma: 1, fallback: null };
	if ( steps === 0 || st.anchor < 0 ) return same;
	// (nothing merges, nothing splits: the communities of resolution 1 are the answer at this one too)
	return ( steps < 0 ? mergedHierarchy( st, opts, - steps ) : splitHierarchy( st, opts, steps, maxTop ) ) || { ...same, gamma: gridResolution( steps ) };

}

/**
 * Quotient graph of a dense partition with the internal weight of every cluster as a self loop.
 * The modularity (any γ) of a grouping of its nodes equals the modularity of the merged
 * partition on the graph.
 */
function quotientWithLoops( edges, weights, level, K ) {

	const acc = new Map();
	for ( let e = 0, E = edges.length >> 1; e < E; e ++ ) {

		let a = level[ edges[ 2 * e ] ], b = level[ edges[ 2 * e + 1 ] ];
		if ( a > b ) {

			const t = a; a = b; b = t;

		}

		const key = a * K + b;
		acc.set( key, ( acc.get( key ) || 0 ) + ( weights ? weights[ e ] : 1 ) );

	}

	const qEdges = new Uint32Array( 2 * acc.size ), qWeights = new Float64Array( acc.size );
	let k = 0;
	for ( const [ key, w ] of acc ) {

		qEdges[ 2 * k ] = Math.floor( key / K );
		qEdges[ 2 * k + 1 ] = key % K;
		qWeights[ k ++ ] = w;

	}

	return { nodeCount: K, edges: qEdges, weights: qWeights };

}

/**
 * Groups the K clusters of the dense partition `part` by modularity on their quotient graph, at
 * the grid resolutions 2^( −j / RESOLUTION_STEPS ), j = 1 … steps (best of 4 seeds each, one for
 * more than 512 clusters): the grouping with the fewest groups, at least two ("fewer
 * communities" should not mean "no communities"); among equals the one at the lowest
 * resolution. Null when nothing merges. The count never rises with `steps`.
 * @returns {{ group: Int32Array( K ), count: number, gamma: number } | null}
 */
export function mergeCommunities( graph, part, K, steps, seed = 1 ) {

	if ( K < 3 || steps < 1 ) return null;
	const q = quotientWithLoops( graph.edges, graph.weights, part, K );
	const tries = K <= 512 ? 4 : 1;
	let best = null;
	for ( let j = 1; j <= steps; j ++ ) {

		const gamma = gridResolution( - j );
		let pick = null;
		for ( let t = 0; t < tries; t ++ ) {

			const r = leiden( q, { strategy: 'flat', resolution: gamma, seed: ( seed + 977 * j + 7919 * t ) >>> 0 || 1 } );
			const quality = partitionModularity( q, r.partition, gamma );
			if ( ! pick || quality > pick.quality + 1e-12 ) pick = { quality, count: r.communities, group: r.partition };

		}

		if ( pick.count < 2 ) break; // below this resolution everything is one community
		if ( pick.count < K && ( ! best || pick.count <= best.count ) ) best = { group: pick.group, count: pick.count, gamma };

	}

	return best;

}

function nestedIn( fine, coarse, Kf ) {

	const parent = new Int32Array( Kf ).fill( - 1 );
	for ( let i = 0; i < fine.length; i ++ ) {

		const p = parent[ fine[ i ] ];
		if ( p < 0 ) parent[ fine[ i ] ] = coarse[ i ];
		else if ( p !== coarse[ i ] ) return false;

	}

	return true;

}

function largestCluster( level, K ) {

	const size = new Int32Array( K );
	let largest = 0;
	for ( let i = 0; i < level.length; i ++ ) if ( ++ size[ level[ i ] ] > largest ) largest = size[ level[ i ] ];
	return largest;

}

/**
 * The hierarchy below resolution 1: the communities of the anchor merged. Modularity at the
 * grid resolutions from 1 down to the request is optimised on their quotient graph (the
 * internal weight of a community is a self loop there, so this is the modularity of the merged
 * partition on the graph itself) and the grouping with the fewest groups is the main level.
 *  · Below it: the levels of resolution 1 from the anchor down that are a real step finer (at
 *    least twice as many clusters as the level kept above them), and the finest one whenever
 *    the leaves would otherwise exceed maxLeafSize. "Coarse communities at low resolution with
 *    the finer ones nested inside."
 *  · Above it: the levels of resolution 1 that the merged communities are nested in.
 *  · When nothing stays below (the merged communities are less than twice as coarse as the
 *    anchor), the ladder splits the leaves or, where that gives nothing, takes the anchor
 *    back (`fallback`), so the hierarchy keeps two levels.
 */
function mergedHierarchy( st, opts, steps ) {

	const { work: graph, levels: H, Ks, anchor: a } = st;
	const merged = mergeCommunities( graph, H[ a ], Ks[ a ], steps, opts.seed );
	if ( ! merged ) return null;
	const n = graph.nodeCount;
	const coarse = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) coarse[ i ] = merged.group[ H[ a ][ i ] ];
	const top = densify( coarse );

	const below = [];
	let last = top.K;
	for ( let l = a; l >= 0; l -- ) {

		if ( Ks[ l ] >= 2 * last ) {

			below.unshift( l );
			last = Ks[ l ];

		}

	}

	// the leaf size: the finest level of resolution 1 has it (in place of a level it is close to)
	const cap = opts.maxLeafSize > 0 ? opts.maxLeafSize : Infinity;
	if ( below[ 0 ] !== 0 && ( below.length ? largestCluster( H[ below[ 0 ] ], Ks[ below[ 0 ] ] ) : largestCluster( top.level, top.K ) ) > cap ) {

		if ( below.length && Ks[ 0 ] < 2 * Ks[ below[ 0 ] ] ) below[ 0 ] = 0;
		else below.unshift( 0 );

	}

	const levels = below.map( ( l ) => H[ l ] ), kinds = below.map( ( l ) => st.kinds[ l ] );
	const main = levels.length;
	levels.push( top.level );
	kinds.push( 'community' );
	last = top.K;
	for ( let l = a + 1; l < H.length; l ++ ) {

		if ( Ks[ l ] >= 2 && last >= 1.5 * Ks[ l ] && nestedIn( top.level, H[ l ], top.K ) ) {

			levels.push( H[ l ] );
			kinds.push( st.kinds[ l ] );
			last = Ks[ l ];

		}

	}

	return { levels, kinds, main, detail: `${st.detail}, merged at resolution ${+ merged.gamma.toFixed( 3 )}`, gamma: merged.gamma, fallback: below.length ? null : H[ a ] };

}

// ── splitting communities (resolutions above 1) ──────────────────────────────────────────────

/** Members, internal edges, degree sums and finer clusters of every community of a level. */
function communityIndex( graph, A, K, fineLevel ) {

	const n = graph.nodeCount, edges = graph.edges, weights = graph.weights, E = edges.length >> 1;
	const size = new Int32Array( K );
	for ( let i = 0; i < n; i ++ ) size[ A[ i ] ] ++;
	const mstart = new Int32Array( K + 1 );
	for ( let c = 0; c < K; c ++ ) mstart[ c + 1 ] = mstart[ c ] + size[ c ];
	const members = new Int32Array( n ), localOf = new Int32Array( n );
	{

		const fill = mstart.slice( 0, K );
		for ( let i = 0; i < n; i ++ ) {

			localOf[ i ] = fill[ A[ i ] ] - mstart[ A[ i ] ];
			members[ fill[ A[ i ] ] ++ ] = i;

		}

	}

	// tot: sum of the degrees of a community in the whole graph; inner2: twice its internal weight
	const estart = new Int32Array( K + 1 ), tot = new Float64Array( K ), inner2 = new Float64Array( K );
	let m2 = 0;
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		if ( a === b ) continue;
		const w = weights ? weights[ e ] : 1;
		m2 += 2 * w;
		tot[ A[ a ] ] += w;
		tot[ A[ b ] ] += w;
		if ( A[ a ] === A[ b ] ) {

			estart[ A[ a ] + 1 ] ++;
			inner2[ A[ a ] ] += 2 * w;

		}

	}

	for ( let c = 0; c < K; c ++ ) estart[ c + 1 ] += estart[ c ];
	const inner = new Int32Array( estart[ K ] );
	{

		const fill = estart.slice( 0, K );
		for ( let e = 0; e < E; e ++ ) {

			const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
			if ( a !== b && A[ a ] === A[ b ] ) inner[ fill[ A[ a ] ] ++ ] = e;

		}

	}

	// the clusters of the finest level of resolution 1 inside every community: local ids, count
	const fine = new Int32Array( n ), fineCount = new Int32Array( K ).fill( 1 );
	if ( fineLevel ) {

		fineCount.fill( 0 );
		const id = new Int32Array( levelSizes( fineLevel ) ).fill( - 1 );
		for ( let i = 0; i < n; i ++ ) {

			let f = id[ fineLevel[ i ] ];
			if ( f < 0 ) f = id[ fineLevel[ i ] ] = fineCount[ A[ i ] ] ++;
			fine[ i ] = f;

		}

	}

	return { size, mstart, members, localOf, estart, inner, tot, inner2, m2, fine, fineCount };

}

/**
 * The graph a community is split on: its own subgraph, or, when resolution 1 already found
 * finer clusters inside it, their quotient graph (internal weights as self loops), so that the
 * split is a grouping of those clusters and both stay levels of one hierarchy.
 */
function communityGraph( graph, ix, c ) {

	const from = ix.estart[ c ], me = ix.estart[ c + 1 ] - from;
	const r = ix.fineCount[ c ];
	if ( r >= 2 ) {

		const acc = new Map();
		for ( let k = 0; k < me; k ++ ) {

			const e = ix.inner[ from + k ];
			let a = ix.fine[ graph.edges[ 2 * e ] ], b = ix.fine[ graph.edges[ 2 * e + 1 ] ];
			if ( a > b ) {

				const t = a; a = b; b = t;

			}

			acc.set( a * r + b, ( acc.get( a * r + b ) || 0 ) + ( graph.weights ? graph.weights[ e ] : 1 ) );

		}

		const edges = new Uint32Array( 2 * acc.size ), weights = new Float64Array( acc.size );
		let k = 0;
		for ( const [ key, w ] of acc ) {

			edges[ 2 * k ] = Math.floor( key / r );
			edges[ 2 * k + 1 ] = key % r;
			weights[ k ++ ] = w;

		}

		return { nodeCount: r, edges, weights };

	}

	const edges = new Uint32Array( 2 * me );
	const weights = graph.weights ? new Float64Array( me ) : null;
	for ( let k = 0; k < me; k ++ ) {

		const e = ix.inner[ from + k ];
		edges[ 2 * k ] = ix.localOf[ graph.edges[ 2 * e ] ];
		edges[ 2 * k + 1 ] = ix.localOf[ graph.edges[ 2 * e + 1 ] ];
		if ( weights ) weights[ k ] = graph.weights[ e ];

	}

	return { nodeCount: ix.size[ c ], edges, weights };

}

/**
 * The resolution a community is split at, on its own subgraph, for grid step j (γ = 2^( j / 4 )):
 * the larger of
 *  · γ · tot² / ( 2m · 2m_c ): what modularity at γ on the WHOLE graph asks of this community
 *    (tot = its degree sum, 2m_c = twice its internal weight). Large communities feel it first;
 *  · log2( γ ) / 2: the community's OWN null model, phased in from nothing at γ = 1 to its
 *    plain modularity at γ = 4. Without it a graph with many small, clear communities (a
 *    planted partition) would not react to the slider at all: the whole graph's modularity
 *    only splits them at resolutions in the tens.
 */
function splitResolution( ix, c, j ) {

	const own = j / ( 2 * RESOLUTION_STEPS );
	const global = ix.inner2[ c ] > 0 ? ( gridResolution( j ) * ix.tot[ c ] * ix.tot[ c ] ) / ( ix.m2 * ix.inner2[ c ] ) : 0;
	return Math.max( own, global );

}

/** One community at one grid step: { count, parts (per node of `sub`) } or null (no usable split). */
function splitCommunity( sub, quotient, lambda, seed, opts ) {

	if ( quotient ) {

		const r = leiden( sub, { strategy: 'flat', resolution: lambda, seed } );
		return r.communities >= 2 ? { count: r.communities, parts: r.partition } : null;

	}

	const p = methodPartition( sub, opts, lambda, seed );
	const m = sub.nodeCount;
	if ( p.count < 2 || p.count >= m ) return null;
	// beyond the structure: a quarter of the nodes in parts of one or two (the nodes of a clique,
	// the leaves of a star) is no partition to show
	const size = new Int32Array( p.count );
	for ( let i = 0; i < m; i ++ ) size[ p.partition[ i ] ] ++;
	let small = 0;
	for ( let k = 0; k < p.count; k ++ ) if ( size[ k ] < MIN_PART ) small += size[ k ];
	if ( 4 * small > m ) return null;
	const parts = Int32Array.from( p.partition );
	const count = small > 0 ? absorbSmall( m, sub.edges, sub.weights, parts, p.count, MIN_PART ) : p.count;
	return count >= 2 ? { count, parts } : null;

}

/**
 * The anchor's communities split for the grid steps 1 … steps above resolution 1. Every
 * community is split on its own (communityGraph) at every step (splitResolution, the method of
 * the request; parts of fewer than MIN_PART nodes join their strongest neighbour); at step j it
 * has the split with the most parts among the steps 1 … j. So the number of communities never
 * falls when the resolution rises, whatever the noise of a single run, and the result for a
 * step does not depend on which steps were asked for before (the work is kept in the state: a
 * slider that moves up computes every step once).
 * @returns {{ level: Int32Array( n ), K: number }} the partition at `steps`
 */
function splitCommunities( st, opts, steps ) {

	const graph = st.work, n = graph.nodeCount, a = st.anchor;
	const A = st.levels[ a ], K = st.Ks[ a ];
	if ( ! st.up ) {

		// the finest level of resolution 1 that is structure (not the cut to size), when it is below the anchor
		const real = st.forced ? 1 : 0;
		const ix = communityIndex( graph, A, K, real < a ? st.levels[ real ] : null );
		ix.fineLevel = real < a ? st.levels[ real ] : null;
		st.up = { ix, step: 0, results: [], best: new Int32Array( n ), count: new Int32Array( K ).fill( 1 ), done: new Uint8Array( K ) };
		for ( let c = 0; c < K; c ++ ) if ( ix.inner2[ c ] <= 0 || ( ix.fineCount[ c ] < 2 && ix.size[ c ] < 2 * MIN_PART ) ) st.up.done[ c ] = 1;

	}

	const up = st.up, ix = up.ix;
	if ( steps > up.step ) {

		const first = up.step + 1;
		const partAt = [], countAt = [];
		for ( let s = first; s <= steps; s ++ ) {

			partAt.push( new Int32Array( n ) );
			countAt.push( new Int32Array( K ) );

		}

		let covered = 0;
		for ( let c = 0; c < K; c ++ ) {

			const from = ix.mstart[ c ], m = ix.size[ c ];
			const quotient = ix.fineCount[ c ] >= 2;
			const sub = up.done[ c ] ? null : communityGraph( graph, ix, c );
			for ( let s = first; s <= steps; s ++ ) {

				if ( ! up.done[ c ] ) {

					const r = splitCommunity( sub, quotient, splitResolution( ix, c, s ), ( opts.seed + 7919 * s + 131 * c ) >>> 0 || 1, opts );
					if ( r && r.count > up.count[ c ] ) {

						up.count[ c ] = r.count;
						for ( let q = 0; q < m; q ++ ) {

							const i = ix.members[ from + q ];
							up.best[ i ] = r.parts[ quotient ? ix.fine[ i ] : q ];

						}

						// (a grouping that keeps every finer cluster apart is the finest there is)
						if ( quotient && r.count === ix.fineCount[ c ] ) up.done[ c ] = 1;

					}

				}

				const part = partAt[ s - first ];
				for ( let q = 0; q < m; q ++ ) part[ ix.members[ from + q ] ] = up.best[ ix.members[ from + q ] ];
				countAt[ s - first ][ c ] = up.count[ c ];

			}

			covered += m;
			opts.onProgress?.( { phase: 'refine', level: 0, pass: covered, communities: n, modularity: NaN } );

		}

		for ( let s = first; s <= steps; s ++ ) {

			const part = partAt[ s - first ], counts = countAt[ s - first ];
			const offset = new Int32Array( K + 1 );
			for ( let c = 0; c < K; c ++ ) offset[ c + 1 ] = offset[ c ] + counts[ c ];
			for ( let i = 0; i < n; i ++ ) part[ i ] += offset[ A[ i ] ];
			up.results[ s ] = { level: part, K: offset[ K ] };

		}

		up.step = steps;

	}

	return up.results[ steps ];

}

/**
 * A finer level that keeps the leaves within `cap` nodes, or null when `level` does. A cluster
 * above the cap that is a union of clusters of `forced` (the cut to size of resolution 1) takes
 * that cut, so an unsplit community keeps its leaves from one resolution to the next; any other
 * is cut into connected pieces by cutToSize of leiden.js (the same forced split).
 */
function capLeaves( graph, level, K, cap, forced, seed ) {

	const n = graph.nodeCount;
	const size = new Int32Array( K );
	for ( let i = 0; i < n; i ++ ) size[ level[ i ] ] ++;
	let over = 0;
	for ( let c = 0; c < K; c ++ ) if ( size[ c ] > cap ) over ++;
	if ( ! over ) return null;
	const Kf = forced ? levelSizes( forced ) : 0;
	const whole = new Uint8Array( K ).fill( forced ? 1 : 0 );
	if ( forced ) {

		const home = new Int32Array( Kf ).fill( - 1 );
		for ( let i = 0; i < n; i ++ ) {

			const h = home[ forced[ i ] ];
			if ( h < 0 ) home[ forced[ i ] ] = level[ i ];
			else if ( h !== level[ i ] ) whole[ level[ i ] ] = whole[ h ] = 0;

		}

		// (and the cut must fit: its clusters are within the cap by construction of resolution 1)

	}

	// members and internal edges of the clusters that need a cut of their own
	const local = new Int32Array( n ).fill( - 1 );
	const members = new Map();
	for ( let i = 0; i < n; i ++ ) {

		const c = level[ i ];
		if ( size[ c ] <= cap || whole[ c ] ) continue;
		let list = members.get( c );
		if ( ! list ) members.set( c, list = [] );
		local[ i ] = list.length;
		list.push( i );

	}

	const inner = new Map();
	if ( members.size ) {

		for ( let e = 0, E = graph.edges.length >> 1; e < E; e ++ ) {

			const a = graph.edges[ 2 * e ], b = graph.edges[ 2 * e + 1 ];
			if ( a === b || local[ a ] < 0 || level[ a ] !== level[ b ] ) continue;
			let list = inner.get( level[ a ] );
			if ( ! list ) inner.set( level[ a ], list = [] );
			list.push( e );

		}

	}

	const key = new Float64Array( n );
	for ( let i = 0; i < n; i ++ ) key[ i ] = size[ level[ i ] ] <= cap ? level[ i ] : whole[ level[ i ] ] ? K + forced[ i ] : - 1;
	let next = K + Kf;
	for ( const [ c, list ] of members ) {

		const es = inner.get( c ) || [];
		const edges = new Uint32Array( 2 * es.length );
		const weights = graph.weights ? new Float64Array( es.length ) : null;
		es.forEach( ( e, k ) => {

			edges[ 2 * k ] = local[ graph.edges[ 2 * e ] ];
			edges[ 2 * k + 1 ] = local[ graph.edges[ 2 * e + 1 ] ];
			if ( weights ) weights[ k ] = graph.weights[ e ];

		} );
		const r = cutToSize( { nodeCount: list.length, edges, weights }, cap, { seed: ( seed + 131 * c ) >>> 0 || 1 } );
		for ( let q = 0; q < list.length; q ++ ) key[ list[ q ] ] = next + r.partition[ q ];
		next += r.K;

	}

	const d = densify( key );
	return d.K > K ? { level: d.level, K: d.K } : null;

}

/**
 * The hierarchy above resolution 1: the communities of the anchor split (splitCommunities) as
 * the main level.
 *  · Below it: where resolution 1 found finer clusters inside a community, they stay the
 *    leaves (the split is a grouping of them), so the leaves follow the structure and not the
 *    slider; elsewhere the split communities are the leaves themselves. No level is invented
 *    below the communities that were asked for (leaves of a handful of nodes make a busy
 *    picture), except the cut to size where a leaf is above maxLeafSize (capLeaves).
 *  · Above it: the anchor when the split at least doubled it, or when nothing else would be
 *    above and the split communities are few enough to hang under the root (`maxTop`; more get
 *    groups from the ladder instead); then the levels of resolution 1 above the anchor.
 */
function splitHierarchy( st, opts, steps, maxTop ) {

	const a = st.anchor, H = st.levels, Ks = st.Ks, graph = st.work, n = graph.nodeCount;
	const split = splitCommunities( st, opts, steps );
	if ( split.K <= Ks[ a ] ) return null;
	// the finest structure: the finer clusters of resolution 1 where a community has them
	const ix = st.up.ix;
	let S = split, F = split;
	if ( ix.fineLevel ) {

		const A = H[ a ], K0 = levelSizes( ix.fineLevel );
		const key = new Float64Array( n );
		for ( let i = 0; i < n; i ++ ) key[ i ] = ix.fineCount[ A[ i ] ] >= 2 ? ix.fineLevel[ i ] : K0 + split.level[ i ];
		F = densify( key );
		// Within a fifth of it, the finest structure IS the answer: the level the default
		// hierarchy shows, not a grouping that merges a few of its clusters. (Once this holds it
		// holds at every higher resolution, so the count still never falls.)
		if ( F.K < 1.2 * split.K ) S = F;

	}

	const below = [];
	if ( F !== S ) below.push( { level: F.level, K: F.K, kind: 'community' } );
	const cut = opts.maxLeafSize > 0 ? capLeaves( graph, F.level, F.K, opts.maxLeafSize, st.forced ? H[ 0 ] : null, opts.seed + 17 ) : null;
	if ( cut ) {

		// (in place of a level it is close to: the leaves need the cut)
		if ( below.length && cut.K < 1.2 * below[ 0 ].K ) below[ 0 ] = { ...cut, kind: 'split' };
		else below.unshift( { ...cut, kind: 'split' } );

	}

	const levels = below.map( ( b ) => b.level ), kinds = below.map( ( b ) => b.kind );
	const main = levels.length;
	levels.push( S.level );
	kinds.push( 'community' );
	if ( S.K >= 2 * Ks[ a ] || ( a + 1 >= H.length && ! below.length && S.K <= maxTop ) ) {

		levels.push( H[ a ] );
		kinds.push( st.kinds[ a ] );

	}

	for ( let l = a + 1; l < H.length; l ++ ) {

		levels.push( H[ l ] );
		kinds.push( st.kinds[ l ] );

	}

	const gamma = gridResolution( steps );
	return { levels, kinds, main, detail: `${st.detail}, split at resolution ${+ gamma.toFixed( 3 )}`, gamma, fallback: null };

}

/** The induced subgraph on the nodes with keep[i] = 1 (ids renumbered in node order). */
function inducedGraph( g, keep, kept ) {

	const map = new Int32Array( g.nodeCount ).fill( - 1 );
	for ( let i = 0, k = 0; i < g.nodeCount; i ++ ) if ( keep[ i ] ) map[ i ] = k ++;
	const E = g.edges.length >> 1;
	let m = 0;
	for ( let e = 0; e < E; e ++ ) if ( map[ g.edges[ 2 * e ] ] >= 0 && map[ g.edges[ 2 * e + 1 ] ] >= 0 ) m ++;
	const edges = new Uint32Array( 2 * m );
	const weights = g.weights ? new Float32Array( m ) : null;
	for ( let e = 0, k = 0; e < E; e ++ ) {

		const a = map[ g.edges[ 2 * e ] ], b = map[ g.edges[ 2 * e + 1 ] ];
		if ( a < 0 || b < 0 ) continue;
		edges[ 2 * k ] = a; edges[ 2 * k + 1 ] = b;
		if ( weights ) weights[ k ] = g.weights[ e ];
		k ++;

	}

	return { graph: { nodeCount: kept, edges, weights }, map };

}

/**
 * The info record of a hierarchy. base.mainLevel names the level that carries the communities;
 * without it (a hierarchy that came with a file) it is the 'community' or 'planted' level
 * closest to base.mainCount clusters, or the level with the highest modularity.
 */
function finishInfo( g, levels, kinds, base ) {

	const n = g.nodeCount;
	const clustersPerLevel = levels.map( levelSizes );
	const modularityPerLevel = levels.map( ( l ) => levelModularity( n, g.edges, g.weights, l, 1 ) );
	const dustClusters = Array.isArray( base.dustClusters ) && base.dustClusters.length === levels.length ? base.dustClusters.slice() : levels.map( () => 0 );
	let mainLevel = - 1;
	if ( levels.length ) {

		if ( Number.isInteger( base.mainLevel ) && base.mainLevel >= 0 && base.mainLevel < levels.length ) mainLevel = base.mainLevel;
		else if ( base.mainCount > 0 ) {

			let best = Infinity;
			levels.forEach( ( l, k ) => {

				if ( kinds[ k ] !== 'community' && kinds[ k ] !== 'planted' ) return;
				const d = Math.abs( Math.log( ( clustersPerLevel[ k ] - dustClusters[ k ] ) / base.mainCount ) );
				if ( d < best ) {

					best = d;
					mainLevel = k;

				}

			} );

		}

		if ( mainLevel < 0 ) mainLevel = modularityPerLevel.indexOf( Math.max( ...modularityPerLevel ) );

	}

	const gamma = base.resolutionUsed ?? base.resolution ?? 1;
	return {
		method: base.method,
		detail: base.detail,
		resolution: base.resolution ?? 1,
		resolutionUsed: gamma,
		seed: base.seed ?? null,
		maxLeafSize: base.maxLeafSize ?? null,
		ladder: !! base.ladder,
		levels: levels.length,
		clustersPerLevel,
		kinds,
		mainLevel,
		communities: mainLevel >= 0 ? clustersPerLevel[ mainLevel ] - dustClusters[ mainLevel ] : 0,
		modularity: mainLevel >= 0 ? modularityPerLevel[ mainLevel ] : 0,
		quality: mainLevel >= 0 ? ( gamma === 1 ? modularityPerLevel[ mainLevel ] : levelModularity( n, g.edges, g.weights, levels[ mainLevel ], gamma ) ) : 0,
		modularityPerLevel,
		weighted: !! g.weights,
		dust: base.dust || 0,
		dustClusters,
		capped: base.capped || 0,
		ms: base.ms ?? 0,
	};

}

/**
 * Cluster hierarchy of a canonical graph (see the header).
 * @param {{ nodeCount: number, edges: Uint32Array, weights?: Float32Array|null }} g
 * @param {object} [options] method, resolution, seed, maxLeafSize, ladder (boolean or
 *   LADDER_DEFAULTS overrides), planted, onProgress, and the tuning options of leiden.js
 * @returns {{ levels: number[][], info: object }}
 */
export function clusterGraph( g, options = {} ) {

	const t0 = now();
	const opts = { ...CLUSTER_DEFAULTS };
	for ( const key in options ) if ( options[ key ] !== undefined && options[ key ] !== null ) opts[ key ] = options[ key ];
	opts.resolution = Number.isFinite( + opts.resolution ) && + opts.resolution > 0 ? + opts.resolution : 1;
	opts.seed = ( opts.seed >>> 0 ) || 1;
	const n = g.nodeCount;
	const ladder = opts.ladder === false ? null : { ...LADDER_DEFAULTS, ...( typeof opts.ladder === 'object' ? opts.ladder : {} ) };
	const planted = options.planted && options.planted.levels && options.planted.levels.length ? options.planted : null;
	const base = { method: opts.method, resolution: opts.resolution, seed: opts.seed, maxLeafSize: opts.maxLeafSize, ladder: !! ladder };
	if ( opts.method !== 'leiden' && opts.method !== 'louvain' ) throw new Error( `Clustering: unknown method "${opts.method}" (leiden | louvain)` );

	if ( planted && planted.complete ) {

		const levels = normalizeLevels( planted.levels.map( ( l ) => Array.from( l ) ), n );
		const weights = usableWeights( g.weights, g.edges.length >> 1 );
		// (the main level of a generator's truth: the one with the highest modularity)
		return { levels, info: finishInfo( weights === ( g.weights || null ) ? g : { nodeCount: n, edges: g.edges, weights }, levels, levels.map( () => 'planted' ), { ...base, method: 'planted', detail: 'planted', ladder: false, resolution: 1, seed: null, maxLeafSize: null, ms: now() - t0 } ) };

	}

	// 1. dust taken out, 2. the hierarchy of resolution 1 with its anchor (kept between calls)
	const st = anchorState( g, opts );
	const { work, map, useDust, dustInfo } = st;

	// 2. communities at the requested resolution, built on the anchor
	const h = baseHierarchy( st, opts, ladder ? ladder.maxTop : Infinity );
	let levels = h.levels.slice();
	let kinds = h.kinds.slice();
	let detail = h.detail;
	let main = h.main;
	let fallback = h.fallback;

	// a partial planted partition (a group attribute of the file) goes on top: the computed
	// levels are intersected with it, so every cluster lies inside one planted group
	if ( planted ) {

		const tops = normalizeLevels( planted.levels.map( ( l ) => Array.from( l ) ), n ).map( ( l ) => ( map ? compact( l, map, work.nodeCount ) : Int32Array.from( l ) ) ).map( ( l ) => densify( l ).level );
		if ( tops.length ) {

			const finest = tops[ 0 ], Kp = levelSizes( finest );
			const below = [];
			let mainBelow = - 1;
			levels.forEach( ( level, l ) => {

				const K = levelSizes( level );
				const cut = new Float64Array( level.length );
				for ( let i = 0; i < cut.length; i ++ ) cut[ i ] = level[ i ] * Kp + finest[ i ];
				const d = densify( cut );
				if ( d.K > Kp * 1.2 && ( ! below.length || levelSizes( below[ below.length - 1 ] ) > d.K * 1.2 ) && K > 1 ) {

					below.push( d.level );
					if ( l === h.main ) mainBelow = below.length - 1;

				}

			} );

			levels = [ ...below, ...tops ];
			kinds = [ ...below.map( () => 'community' ), ...tops.map( () => 'planted' ) ];
			detail = `${planted.source || 'planted'} + ${h.detail}`;
			base.method = 'planted';
			// the communities inside the groups of the file; the groups themselves where the
			// communities add nothing to them
			main = mainBelow >= 0 ? mainBelow : below.length;
			fallback = null;

		}

	}

	// 3. the ladder
	if ( ladder && levels.length ) {

		const before = kinds.join();
		const shaped = shapeLevels( work, levels, { ...ladder, seed: opts.seed, kinds, main, fallback, maxLeafSize: opts.maxLeafSize > 0 ? opts.maxLeafSize : 0 } );
		levels = shaped.levels;
		kinds = shaped.kinds;
		main = shaped.main;
		if ( kinds.join() !== before ) detail += ' + ladder';
		opts.onProgress?.( { phase: 'ladder', level: levels.length, pass: 1, communities: levels.length ? levelSizes( levels[ 0 ] ) : 0, modularity: NaN } );

	}

	// without communities (a star, a clique) the coarsest level stands in for them
	if ( main < 0 && levels.length ) main = levels.length - 1;

	// 4. back to all nodes, with the dust tree
	let dustClusters = null;
	if ( useDust && levels.length ) {

		const D = levels.length;
		const dl = dustLevels( n, D, dustInfo, opts.maxLeafSize > 0 ? opts.maxLeafSize : 0 );
		dustClusters = dl.counts;
		levels = levels.map( ( level, l ) => {

			const K = levelSizes( level );
			const out = new Int32Array( n );
			for ( let i = 0; i < n; i ++ ) out[ i ] = map[ i ] >= 0 ? level[ map[ i ] ] : K + dl.levels[ l ][ i ];
			return out;

		} );

	}

	const canonical = levels.length ? normalizeLevels( levels.map( ( l ) => Array.from( l ) ), n ) : [];
	// normalizeLevels may drop a degenerate level: keep the tags aligned by cluster count
	let tags = kinds;
	if ( canonical.length !== levels.length ) {

		const counts = levels.map( levelSizes );
		const at = canonical.map( ( l ) => counts.indexOf( levelSizes( l ) ) );
		tags = at.map( ( k ) => kinds[ k ] || 'community' );
		if ( dustClusters ) dustClusters = at.map( ( k ) => dustClusters[ k ] || 0 );
		main = canonical.length ? Math.max( 0, at.indexOf( main ) ) : - 1;

	}

	return { levels: canonical, info: finishInfo( st.whole, canonical, tags, { ...base, resolutionUsed: h.gamma, detail, mainLevel: main, dust: useDust ? dustInfo.count : 0, dustClusters, capped: st.capped, ms: now() - t0 } ) };

}

function compact( level, map, kept ) {

	const out = new Int32Array( kept );
	for ( let i = 0; i < level.length; i ++ ) if ( map[ i ] >= 0 ) out[ map[ i ] ] = level[ i ];
	return out;

}

/**
 * Info record (the same shape as clusterGraph's, plus stored: true) for a hierarchy that came
 * with the graph. base: what the file says about it ({ method, detail, resolution, seed,
 * maxLeafSize, ladder, kinds, mainLevel or mainCount }); method is 'file' when it says nothing.
 */
export function describeHierarchy( g, levels, base = {} ) {

	const lv = levels || [];
	const kinds = Array.isArray( base.kinds ) && base.kinds.length === lv.length ? base.kinds.slice() : lv.map( () => base.kind || 'community' );
	const info = finishInfo( g, lv, kinds, { method: 'file', detail: 'file', resolution: 1, ladder: false, mainCount: 0, ...base } );
	info.stored = true;
	return info;

}

/**
 * A shallow copy of the canonical graph g with its hierarchy computed: `hierarchy` =
 * { method: info.detail, levels } (null when there is none) and `clustering` = info. What the
 * data worker and scripts/generate.mjs call, so stored presets match what the worker computes.
 */
export function applyClustering( g, options = {} ) {

	const { levels, info } = clusterGraph( g, options );
	return {
		...g,
		hierarchy: levels.length ? { method: info.detail, levels } : null,
		clustering: info,
		meta: { ...( g.meta || {} ), hierarchy: { method: info.detail, algorithm: info.method, modularity: info.modularityPerLevel, resolution: info.resolution, seed: info.seed, maxLeafSize: info.maxLeafSize, ladder: info.ladder, kinds: info.kinds, communities: info.communities, mainLevel: info.mainLevel } },
	};

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}
