// Graph file formats.
//
// Canonical in-memory/on-disk format ("wgv-graph", version 1). JSON, compact enough for ~20k
// nodes / 400k edges:
//
// {
//   "format": "wgv-graph", "version": 1,
//   "name": "Les Misérables",
//   "description": "...",                // optional
//   "source": "...",                     // optional attribution
//   "nodeCount": 77,
//   "labels": ["Myriel", ...],           // optional, length nodeCount
//   "edges": [s0, t0, s1, t1, ...],      // flat undirected pairs of node indices
//   "weights": [w0, w1, ...],            // optional, one non-negative weight per edge
//   "attributes": {                      // optional node attributes (at most 32)
//     "age":  { "type": "number", "values": [31, null, ...] },              // null = missing
//     "kind": { "type": "category", "values": [0, 2, -1, ...], "categories": ["a", "b", "c"] }
//   },
//   "hierarchy": {                       // optional (computed in the data worker when missing)
//     "method": "leiden",
//     "levels": [[...], [...]]           // levels[0] = finest partition; levels[l][i] = cluster
//   },                                   //   id of node i at level l, dense 0..K_l-1, nested
//   "meta": { ... },                     // optional: generator parameters, stats
//   "positions": [x0, y0, z0, ...]       // optional saved layout (export.js), used as the start
// }
//
// Hierarchy semantics follow ImsvGraphVis/python-louvain: every node has one ancestor per level
// (uniform depth), level 0 is the finest, and an implicit root sits above the last level.
// Nesting: nodes sharing a level-l cluster share their level-(l+1) cluster.
//
// Importers for the ImsvGraphVis preprocessed format (*.igv.json) and the common d3 / networkx
// node-link format ({nodes:[{id}], links:[{source,target}]}) are provided too.

export const FORMAT = 'wgv-graph';
export const VERSION = 1;

/** At most this many node attributes are kept per graph (see normalizeAttributes). */
export const MAX_ATTRIBUTES = 32;
/** A text attribute with more distinct values than this is not a category (it is dropped). */
export const MAX_CATEGORIES = 64;
/** The largest weight an edge can carry (weights are stored as 32-bit floats). */
export const MAX_WEIGHT = 3.4028234663852886e38;

/**
 * Validates and cleans a canonical graph: drops self loops and invalid edges, merges duplicate
 * edges (either direction; their weights are summed), renumbers hierarchy levels densely,
 * repairs nesting violations and drops degenerate levels. Returns a new object; never mutates
 * the input. The result carries `dropped: { selfLoops, duplicates, invalid }` and, when weights
 * were limited or ignored, `weightIssues` (see cleanEdges; neither is serialized).
 */
export function normalizeGraph( raw ) {

	if ( ! raw || typeof raw !== 'object' ) throw new Error( 'Graph: expected an object' );
	const n = raw.nodeCount | 0;
	if ( ! ( n > 0 ) ) throw new Error( 'Graph: nodeCount must be > 0' );

	const cleaned = cleanEdges( n, raw.edges || [], raw.weights || null );
	const g = {
		format: FORMAT,
		version: VERSION,
		name: raw.name || 'Untitled graph',
		description: raw.description || '',
		source: raw.source || '',
		nodeCount: n,
		labels: Array.isArray( raw.labels ) && raw.labels.length === n ? raw.labels.map( String ) : null,
		edges: cleaned.edges,
		weights: cleaned.weights,
		attributes: normalizeAttributes( raw.attributes, n ),
		hierarchy: null,
		meta: raw.meta || {},
		dropped: cleaned.dropped,
	};
	if ( cleaned.mirrored ) g.mirrored = true;
	if ( cleaned.weightIssues.limited || cleaned.weightIssues.allZero ) g.weightIssues = cleaned.weightIssues;
	// saved layout (written by export.js): the loader starts the layout from it
	if ( raw.positions && raw.positions.length === 3 * n ) g.positions = Float32Array.from( raw.positions );
	if ( raw.directed ) g.directed = true;

	if ( raw.hierarchy && Array.isArray( raw.hierarchy.levels ) && raw.hierarchy.levels.length > 0 ) {

		g.hierarchy = {
			method: raw.hierarchy.method || 'unknown',
			levels: normalizeLevels( raw.hierarchy.levels, n ),
		};
		if ( g.hierarchy.levels.length === 0 ) g.hierarchy = null;

	}

	return g;

}

/**
 * Undirected simple edges from flat pairs: every pair is ordered (a < b), self loops and pairs
 * with an end point outside 0..n-1 are dropped, duplicates are merged into their first occurrence
 * (weights summed: a duplicate without weights counts 1), and the survivors keep the order of
 * their first occurrence. Two stable counting sorts, typed arrays only: O( E + n ) time and about
 * 17 bytes per input edge, so a multi-million edge file does not need a hash set.
 * weights: null when no weights were given and nothing was merged, or when every merged weight
 * is 1; non-finite weights count 1, negative ones by their absolute value. A sum beyond the
 * range of a 32-bit float is limited to MAX_WEIGHT (counted in weightIssues.limited), and when
 * every weight is 0 (or too small for a 32-bit float) the graph is unweighted
 * (weightIssues.allZero): weights that carry no information must not switch the clustering off.
 * mirrored: every edge came once per direction with the same weight (an undirected graph stored
 * as arcs, or a full symmetric matrix): such an edge keeps that weight instead of twice it.
 *
 * @param {number} n node count
 * @param {ArrayLike<number>} pairs flat node index pairs
 * @param {ArrayLike<number>|null} weights one per pair
 * @returns {{ edges: Uint32Array, weights: Float32Array|null, dropped: { selfLoops: number, duplicates: number, invalid: number }, mirrored: boolean, weightIssues: { limited: number, allZero: boolean } }}
 */
export function cleanEdges( n, pairs, weights = null ) {

	const m = pairs.length >> 1;
	const A = new Uint32Array( m ), B = new Uint32Array( m );
	const valid = new Uint32Array( m ); // indices of the usable pairs
	let v = 0, selfLoops = 0, invalid = 0;
	for ( let e = 0; e < m; e ++ ) {

		const x = pairs[ 2 * e ], y = pairs[ 2 * e + 1 ];
		const a = x | 0, b = y | 0;
		if ( a !== x || b !== y || a < 0 || b < 0 || a >= n || b >= n ) {

			// (non-integers are accepted when they truncate into range, as before)
			if ( ! ( x >= 0 && y >= 0 && x < n && y < n ) ) {

				invalid ++;
				continue;

			}

		}

		if ( a === b ) {

			selfLoops ++;
			continue;

		}

		A[ e ] = a < b ? a : b;
		B[ e ] = a < b ? b : a;
		valid[ v ++ ] = e;

	}

	// stable counting sort of the usable pairs by b, then by a
	const count = new Uint32Array( n + 1 );
	const byB = new Uint32Array( v ), order = new Uint32Array( v );
	for ( let k = 0; k < v; k ++ ) count[ B[ valid[ k ] ] + 1 ] ++;
	for ( let i = 0; i < n; i ++ ) count[ i + 1 ] += count[ i ];
	for ( let k = 0; k < v; k ++ ) byB[ count[ B[ valid[ k ] ] ] ++ ] = valid[ k ];
	count.fill( 0 );
	for ( let k = 0; k < v; k ++ ) count[ A[ byB[ k ] ] + 1 ] ++;
	for ( let i = 0; i < n; i ++ ) count[ i + 1 ] += count[ i ];
	for ( let k = 0; k < v; k ++ ) order[ count[ A[ byB[ k ] ] ] ++ ] = byB[ k ];

	// runs of equal ( a, b ): the first (lowest index, the sorts are stable) keeps the edge
	const weightOf = ( e ) => {

		if ( ! weights ) return 1;
		const w = + weights[ e ];
		return Number.isFinite( w ) ? Math.abs( w ) : 1;

	};

	// A run sums its weights. When every edge is listed once per direction with the same total
	// (an undirected graph stored as arcs, a full symmetric matrix) the edge has that weight, not
	// twice it: `mirrored`. Otherwise both directions add up (A + Aᵀ).
	const sum = new Float64Array( v ? m : 0 );
	const keep = new Uint8Array( m );
	let unique = 0, mirrored = v > 0;
	for ( let k = 0; k < v; ) {

		const first = order[ k ];
		let f = 0, r = 0;
		let j = k;
		while ( j < v && A[ order[ j ] ] === A[ first ] && B[ order[ j ] ] === B[ first ] ) {

			const e = order[ j ++ ];
			if ( pairs[ 2 * e ] < pairs[ 2 * e + 1 ] ) f += weightOf( e );
			else r += weightOf( e );

		}

		if ( mirrored && ! ( f > 0 && Math.abs( f - r ) <= 1e-6 * ( f + r ) ) ) mirrored = false;
		keep[ first ] = 1;
		sum[ first ] = f + r;
		unique ++;
		k = j;

	}

	const scale = mirrored ? 0.5 : 1;
	let weighted = false;
	for ( let e = 0; e < m && ! weighted; e ++ ) if ( keep[ e ] && sum[ e ] * scale !== 1 ) weighted = true;
	const edges = new Uint32Array( 2 * unique );
	let out = weighted ? new Float32Array( unique ) : null;
	let limited = 0, positive = false;
	for ( let e = 0, k = 0; e < m; e ++ ) {

		if ( ! keep[ e ] ) continue;
		edges[ 2 * k ] = A[ e ];
		edges[ 2 * k + 1 ] = B[ e ];
		if ( out ) {

			let w = sum[ e ] * scale;
			if ( w > MAX_WEIGHT ) {

				w = MAX_WEIGHT;
				limited ++;

			}

			out[ k ] = w;
			if ( out[ k ] > 0 ) positive = true;

		}

		k ++;

	}

	// every weight 0 (as stored: 1e-50 is 0 in a 32-bit float): no information, an unweighted graph
	const allZero = !! out && unique > 0 && ! positive;
	if ( allZero ) out = null;
	return { edges, weights: out, dropped: { selfLoops, duplicates: v - unique, invalid }, mirrored, weightIssues: { limited, allZero } };

}

/**
 * Node attributes of a canonical graph: { [name]: { type: 'number', values: Float32Array( n ) (NaN =
 * missing) } | { type: 'category', values: Int32Array( n ) (-1 = missing), categories: string[] } }.
 * Accepts plain arrays (JSON: null = missing) and drops malformed entries; at most
 * MAX_ATTRIBUTES are kept. Returns {} when there are none.
 */
export function normalizeAttributes( attributes, n ) {

	const out = {};
	if ( ! attributes || typeof attributes !== 'object' ) return out;
	let kept = 0;
	for ( const name of Object.keys( attributes ) ) {

		const a = attributes[ name ];
		if ( ! a || ! a.values || a.values.length !== n ) continue;
		if ( kept >= MAX_ATTRIBUTES ) break;
		if ( a.type === 'category' && Array.isArray( a.categories ) ) {

			const values = new Int32Array( n );
			const K = a.categories.length;
			for ( let i = 0; i < n; i ++ ) {

				const c = a.values[ i ];
				values[ i ] = Number.isInteger( c ) && c >= 0 && c < K ? c : - 1;

			}

			out[ name ] = { type: 'category', values, categories: a.categories.map( String ) };

		} else if ( a.type === 'number' ) {

			const values = new Float32Array( n );
			for ( let i = 0; i < n; i ++ ) {

				const x = a.values[ i ];
				values[ i ] = x === null || x === undefined || x === '' ? NaN : + x;

			}

			out[ name ] = { type: 'number', values };

		} else continue;

		kept ++;

	}

	return out;

}

/**
 * Makes hierarchy levels dense (0..K-1), enforces nesting (a finer cluster that maps to several
 * coarser clusters is assigned to the coarser cluster of its first member) and removes levels
 * that do not merge anything (identical to the previous one) or that are a single cluster
 * (that is the implicit root).
 */
export function normalizeLevels( levels, n ) {

	const result = [];
	let prev = null; // Int32Array of previous (finer) level

	for ( const lvl of levels ) {

		if ( ! lvl || lvl.length !== n ) throw new Error( 'Graph: hierarchy level length must equal nodeCount' );
		const ids = new Int32Array( n );
		const remap = new Map();
		for ( let i = 0; i < n; i ++ ) {

			const key = lvl[ i ];
			let id = remap.get( key );
			if ( id === undefined ) {

				id = remap.size;
				remap.set( key, id );

			}

			ids[ i ] = id;

		}

		if ( prev ) {

			// enforce nesting: each finer cluster takes the coarse id of its first member
			const coarseOfFine = new Map();
			for ( let i = 0; i < n; i ++ ) {

				const f = prev[ i ];
				const c = coarseOfFine.get( f );
				if ( c === undefined ) coarseOfFine.set( f, ids[ i ] );
				else ids[ i ] = c;

			}

			// densify again after repairs
			const remap2 = new Map();
			for ( let i = 0; i < n; i ++ ) {

				let id = remap2.get( ids[ i ] );
				if ( id === undefined ) {

					id = remap2.size;
					remap2.set( ids[ i ], id );

				}

				ids[ i ] = id;

			}

		}

		const count = countDistinct( ids );
		if ( count <= 1 ) break; // a single cluster is the root
		if ( prev && count === countDistinct( prev ) ) continue; // no merge: skip
		if ( ! prev && count === n ) continue; // all singletons: no information
		result.push( Array.from( ids ) );
		prev = ids;

	}

	return result;

}

function countDistinct( ids ) {

	let max = - 1;
	for ( let i = 0; i < ids.length; i ++ ) if ( ids[ i ] > max ) max = ids[ i ];
	return max + 1; // ids are dense

}

/** Imports the ImsvGraphVis preprocessed format (Saved/Data/Graph/*.igv.json). */
export function fromIgv( json, name = 'Untitled graph' ) {

	const nodes = json.nodes;
	const n = nodes.length;
	const byIdx = new Map( nodes.map( ( node, i ) => [ node.idx, i ] ) );
	const edges = [];
	for ( const l of json.links ) edges.push( byIdx.get( l.sourceIdx ), byIdx.get( l.targetIdx ) );

	const depth = nodes[ 0 ].ancIdxs.length;
	const levels = [];
	for ( let lv = 0; lv < depth; lv ++ ) levels.push( nodes.map( ( node ) => node.ancIdxs[ lv ] ) );

	return normalizeGraph( {
		name,
		nodeCount: n,
		labels: nodes.map( ( node ) => String( node.label ?? node.idx ) ),
		edges,
		hierarchy: { method: 'louvain (python-louvain 0.8, ImsvGraphVis preprocessing)', levels },
	} );

}

/** Imports d3 / networkx node-link JSON: {nodes:[{id,...}], links|edges:[{source,target}]}. */
export function fromNodeLink( json, name = 'Untitled graph' ) {

	const nodes = json.nodes || [];
	const links = json.links || json.edges || [];
	const index = new Map();
	nodes.forEach( ( node, i ) => index.set( typeof node === 'object' ? ( node.id ?? i ) : node, i ) );
	const edges = [];
	const resolve = ( v ) => {

		if ( v && typeof v === 'object' ) v = v.id;
		if ( index.has( v ) ) return index.get( v );
		if ( typeof v === 'number' && v >= 0 && v < nodes.length ) return v;
		return - 1;

	};

	for ( const l of links ) {

		const a = resolve( l.source ), b = resolve( l.target );
		if ( a >= 0 && b >= 0 ) edges.push( a, b );

	}

	const labels = nodes.map( ( node, i ) => String( ( typeof node === 'object' ? ( node.label ?? node.name ?? node.id ) : node ) ?? i ) );
	return normalizeGraph( { name: json.name || name, nodeCount: nodes.length, labels, edges } );

}

/** Detects the format of a parsed JSON value and returns a normalized canonical graph. */
export function parseGraphJson( json, name ) {

	if ( json && json.format === FORMAT ) return normalizeGraph( json );
	if ( json && Array.isArray( json.nodes ) && Array.isArray( json.links ) && json.clusters && json.nodes[ 0 ] && 'ancIdxs' in json.nodes[ 0 ] ) return fromIgv( json, name );
	if ( json && Array.isArray( json.nodes ) ) return fromNodeLink( json, name );
	throw new Error( 'Unrecognized graph JSON (expected wgv-graph, ImsvGraphVis .igv.json or node-link {nodes, links})' );

}

/** Serializes a canonical graph to a JSON string (typed arrays become plain arrays). */
export function serializeGraph( g ) {

	let attributes;
	if ( g.attributes && Object.keys( g.attributes ).length ) {

		attributes = {};
		for ( const [ name, a ] of Object.entries( g.attributes ) ) {

			if ( a.type === 'category' ) attributes[ name ] = { type: 'category', values: Array.from( a.values ), categories: a.categories };
			else attributes[ name ] = { type: 'number', values: Array.from( a.values, ( x ) => ( Number.isFinite( x ) ? x : null ) ) };

		}

	}

	return JSON.stringify( {
		format: FORMAT,
		version: VERSION,
		name: g.name,
		description: g.description || undefined,
		source: g.source || undefined,
		nodeCount: g.nodeCount,
		labels: g.labels || undefined,
		edges: Array.from( g.edges ),
		weights: g.weights ? Array.from( g.weights ) : undefined,
		attributes,
		hierarchy: g.hierarchy ? { method: g.hierarchy.method, levels: g.hierarchy.levels.map( ( l ) => Array.from( l ) ) } : undefined,
		meta: g.meta && Object.keys( g.meta ).length ? g.meta : undefined,
	} );

}
