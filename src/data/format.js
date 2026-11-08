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
//   "hierarchy": {                       // optional (computed with Louvain when missing)
//     "method": "louvain",
//     "levels": [[...], [...]]           // levels[0] = finest partition; levels[l][i] = cluster
//   },                                   //   id of node i at level l, dense 0..K_l-1, nested
//   "meta": { ... }                      // optional: generator parameters, stats
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

/**
 * Validates and cleans a canonical graph: drops self loops and duplicate/invalid edges,
 * renumbers hierarchy levels densely, repairs nesting violations and drops degenerate levels.
 * Returns a new object; never mutates the input.
 */
export function normalizeGraph( raw ) {

	if ( ! raw || typeof raw !== 'object' ) throw new Error( 'Graph: expected an object' );
	const n = raw.nodeCount | 0;
	if ( ! ( n > 0 ) ) throw new Error( 'Graph: nodeCount must be > 0' );

	const src = raw.edges || [];
	const seen = new Set();
	const out = [];
	for ( let k = 0; k + 1 < src.length; k += 2 ) {

		let a = src[ k ] | 0, b = src[ k + 1 ] | 0;
		if ( a === b || a < 0 || b < 0 || a >= n || b >= n ) continue;
		if ( a > b ) [ a, b ] = [ b, a ];
		const key = a * n + b;
		if ( seen.has( key ) ) continue;
		seen.add( key );
		out.push( a, b );

	}

	const g = {
		format: FORMAT,
		version: VERSION,
		name: raw.name || 'Untitled graph',
		description: raw.description || '',
		source: raw.source || '',
		nodeCount: n,
		labels: Array.isArray( raw.labels ) && raw.labels.length === n ? raw.labels.map( String ) : null,
		edges: Uint32Array.from( out ),
		hierarchy: null,
		meta: raw.meta || {},
	};

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

	return JSON.stringify( {
		format: FORMAT,
		version: VERSION,
		name: g.name,
		description: g.description || undefined,
		source: g.source || undefined,
		nodeCount: g.nodeCount,
		labels: g.labels || undefined,
		edges: Array.from( g.edges ),
		hierarchy: g.hierarchy ? { method: g.hierarchy.method, levels: g.hierarchy.levels.map( ( l ) => Array.from( l ) ) } : undefined,
		meta: g.meta && Object.keys( g.meta ).length ? g.meta : undefined,
	} );

}
