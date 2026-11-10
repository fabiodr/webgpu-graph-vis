// GraphBuilder: what every file parser writes into, and the one place where a parsed file
// becomes a canonical graph (format.js). Pure JS; runs in the data worker, on the main thread
// and in Node.
//
// A parser declares nodes by their id in the file, sets labels and raw attribute values, adds
// edges between node indices and may say that one node (or a named group) contains another.
// finish() then does the work that is the same for every format:
//  · node order: the order of first appearance, or numeric order when every id is a
//    non-negative integer and the parser asks for it (edge lists);
//  · edges: undirected and simple (format.js cleanEdges): arcs in both directions and parallel
//    edges are merged with their weights summed, self loops and edges with an unknown end
//    point are dropped, and the counts are reported;
//  · attributes: a column whose values are all numbers becomes { type: 'number' } (missing =
//    NaN); a text column with at most MAX_CATEGORIES distinct values becomes
//    { type: 'category' }; other text columns (free text, a different value per node), empty
//    and constant columns are dropped and listed in the report. At most MAX_ATTRIBUTES columns are kept, in file order;
//  · labels: explicit labels, else the first column named label / name / title (any case),
//    else the ids;
//  · groups: containment (GraphML nested graphs, GEXF pid, Cytoscape parent, GML gid, DOT
//    subgraphs) or, without it, a column named group / community / cluster / modularity_class /
//    module / block / partition / class / club / faction gives `planted` = { levels, complete:
//    false, source }: a partial hierarchy that the clustering uses as its top level when
//    options.cluster.mode is 'planted' (hierarchy.js). Container nodes without edges of their
//    own are removed (they are boxes in the source, not nodes);
//  · limits: more than limits.maxNodes nodes or limits.maxEdges edges (after merging) throws a
//    GraphError with code 'too-large' and a message meant for the person who dropped the file.
//
// API
//   new GraphBuilder( { name, limits?, onProgress? } )
//   node( id ) -> index                   declares the node on first use
//   has( id ) -> boolean
//   label( index, text )
//   attr( index, name, value )            raw value: number, boolean or string
//   attrDefault( name, value )            value for the nodes that have none
//   groupColumn( name )                   the column holds class ids (a Pajek partition)
//   edge( a, b, weight = 1 )              node indices; any order, repeats allowed. Weights
//                                         beyond a 32-bit float are limited to it, and a file
//                                         whose weights are all 0 is unweighted (both reported)
//   invalidEdge()                         counts an edge whose end points could not be resolved
//   parent( child, parentIndex )          containment between two nodes
//   group( name, label?, parentGroup? ) -> index of a synthetic container (DOT subgraph)
//   warn( message | () => message )       kept in the report (the first 20; all are counted;
//                                         a function is called only for those that are kept)
//   progress( fraction )                  throttled; forwarded to onProgress
//   directed = true                       the file's edges are arcs (reported)
//   finish( { format, numericOrder?, meta? } ) -> canonical graph
//       .meta = { format, ...meta, report }   report: { format, nodes, edges, directed, mirrored
//         (every edge was listed once per direction: not summed), weighted,
//         dropped: { selfLoops, duplicates, invalid }, labels: 'label' | 'attribute "name"' |
//         'ids', attributes: [ { name, type, categories? } ], droppedAttributes: [ { name,
//         reason } ], groups: { source, count, levels } | null, containers (removed), warnings:
//         string[], warningCount }
//       .planted (not enumerable) when the file has groups
//   DEFAULT_LIMITS, GraphError( message, code ), tooLarge( what, count, limits ),
//   weightWarnings( issues, limitedBefore? ) -> string[]

import { cleanEdges, FORMAT, VERSION, MAX_ATTRIBUTES, MAX_CATEGORIES, MAX_WEIGHT } from './format.js';

/** Largest graph the pipeline accepts from a file or a generator (override per request). */
export const DEFAULT_LIMITS = { maxNodes: 250000, maxEdges: 1000000 };

const MAX_WARNINGS = 20;
const MAX_COLUMNS = 4 * MAX_ATTRIBUTES; // raw columns kept while parsing
const MAX_GROUP_LEVELS = 3;

const LABEL_KEYS = [ 'label', 'name', 'title' ];
const GROUP_KEYS = [ 'group', 'groups', 'community', 'communities', 'cluster', 'clusters', 'modularity_class', 'modularity class', 'modularityclass', 'module', 'block', 'partition', 'class', 'club', 'faction' ];
const MISSING = new Set( [ '', 'nan', 'null', 'none', 'na', 'n/a', 'undefined' ] );
const NUMBER_RE = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;
const INT_RE = /^\d+$/;

/** An error whose message is meant for the user; code: 'parse' | 'too-large' | 'empty' | 'format'. */
export class GraphError extends Error {

	constructor( message, code = 'parse' ) {

		super( message );
		this.name = 'GraphError';
		this.code = code;

	}

}

const fmt = ( x ) => Number( x ).toLocaleString( 'en-US' );

export function tooLarge( what, count, limits ) {

	return new GraphError( `This graph is too large for the viewer: it has ${count === null ? 'more than ' + fmt( what === 'nodes' ? limits.maxNodes : limits.maxEdges ) : fmt( count )} ${what}. The limit is ${fmt( limits.maxNodes )} nodes and ${fmt( limits.maxEdges )} edges.`, 'too-large' );

}

function isMissing( v ) {

	return v === undefined || v === null || ( typeof v === 'number' && Number.isNaN( v ) ) || ( typeof v === 'string' && MISSING.has( v.trim().toLowerCase() ) );

}

/** Sentences about edge weights that could not be used as they are (cleanEdges weightIssues). */
export function weightWarnings( issues, limitedBefore = 0 ) {

	const out = [];
	const limited = ( issues ? issues.limited : 0 ) + limitedBefore;
	if ( limited > 0 ) out.push( `${limited} edge weight${limited === 1 ? ' is' : 's are'} beyond the range of a 32-bit float and ${limited === 1 ? 'was' : 'were'} limited to it` );
	if ( issues && issues.allZero ) out.push( 'every edge weight is 0 (or too small to store): the weights are ignored' );
	return out;

}

export class GraphBuilder {

	constructor( options = {} ) {

		this.name = options.name || 'Untitled graph';
		this.description = '';
		this.limits = { ...DEFAULT_LIMITS, ...( options.limits || {} ) };
		this.onProgress = options.onProgress || null;
		this.ids = new Map(); // id (string) -> index
		this.names = []; // ids in order of first appearance
		this.labels = []; // sparse
		this.hasLabels = false;
		this.columns = new Map(); // name -> sparse array of raw values
		this.defaults = new Map();
		this.overflowColumns = new Set();
		this.groupColumns = new Set();
		this.edges = new Uint32Array( 2048 );
		this.weights = null;
		this.edgeCount = 0;
		this.invalid = 0;
		this.directed = false;
		this.parents = null; // sparse: child index -> parent index
		this.synthetic = null; // sparse: 1 for containers that are not nodes of the file
		this.warnings = [];
		this.warningCount = 0;
		this.limitedWeights = 0;
		this.lastProgress = 0;
		this.rawEdgeLimit = Math.max( 16, 4 * this.limits.maxEdges );

	}

	get nodeCount() {

		return this.names.length;

	}

	has( id ) {

		return this.ids.has( typeof id === 'string' ? id : String( id ) );

	}

	node( id ) {

		const key = typeof id === 'string' ? id : String( id );
		let i = this.ids.get( key );
		if ( i === undefined ) {

			i = this.names.length;
			// containers do not count, hence the slack; finish() checks the final number
			if ( i >= this.limits.maxNodes * 1.25 + 16 ) throw tooLarge( 'nodes', null, this.limits );
			this.ids.set( key, i );
			this.names.push( key );

		}

		return i;

	}

	label( index, text ) {

		if ( text === undefined || text === null ) return;
		this.labels[ index ] = String( text );
		this.hasLabels = true;

	}

	attr( index, name, value ) {

		if ( value === undefined || value === null || typeof value === 'object' ) return;
		let col = this.columns.get( name );
		if ( ! col ) {

			if ( this.columns.size >= MAX_COLUMNS ) {

				this.overflowColumns.add( name );
				return;

			}

			col = [];
			this.columns.set( name, col );

		}

		col[ index ] = value;

	}

	/** Says that a column holds class ids: a category, and a candidate for the planted groups. */
	groupColumn( name ) {

		this.groupColumns.add( name );

	}

	attrDefault( name, value ) {

		if ( value !== undefined && value !== null && value !== '' ) this.defaults.set( name, value );

	}

	edge( a, b, weight = 1 ) {

		const k = this.edgeCount;
		if ( k >= this.rawEdgeLimit ) throw tooLarge( 'edges', null, this.limits );
		if ( 2 * k + 2 > this.edges.length ) {

			const grown = new Uint32Array( this.edges.length * 2 );
			grown.set( this.edges );
			this.edges = grown;
			if ( this.weights ) {

				const w = new Float32Array( grown.length >> 1 );
				w.set( this.weights );
				this.weights = w;

			}

		}

		this.edges[ 2 * k ] = a;
		this.edges[ 2 * k + 1 ] = b;
		if ( weight !== 1 && weight !== undefined && weight !== null && ! Number.isNaN( weight ) ) {

			if ( ! this.weights ) this.weights = new Float32Array( this.edges.length >> 1 ).fill( 1, 0, k );
			// a finite weight beyond a 32-bit float would be stored as Infinity
			if ( ( weight > MAX_WEIGHT || weight < - MAX_WEIGHT ) && Number.isFinite( weight ) ) {

				weight = weight > 0 ? MAX_WEIGHT : - MAX_WEIGHT;
				this.limitedWeights ++;

			}

			this.weights[ k ] = weight;

		} else if ( this.weights ) this.weights[ k ] = 1;

		this.edgeCount = k + 1;

	}

	invalidEdge() {

		this.invalid ++;

	}

	parent( child, parentIndex ) {

		if ( child === parentIndex || parentIndex < 0 ) return;
		if ( ! this.parents ) this.parents = [];
		this.parents[ child ] = parentIndex;

	}

	group( name, label = null, parentGroup = - 1 ) {

		const id = '\u0000group:' + name;
		const fresh = ! this.ids.has( id );
		const i = this.node( id );
		if ( fresh ) {

			if ( ! this.synthetic ) this.synthetic = [];
			this.synthetic[ i ] = 1;
			this.labels[ i ] = String( label ?? name );
			if ( parentGroup >= 0 ) this.parent( i, parentGroup );

		} else if ( label !== null && label !== undefined ) this.labels[ i ] = String( label );

		return i;

	}

	warn( message ) {

		// a function is only called for the warnings that are kept (a line number that takes a
		// scan of the text must not be computed a hundred thousand times)
		this.warningCount ++;
		if ( this.warnings.length < MAX_WARNINGS ) this.warnings.push( typeof message === 'function' ? message() : message );

	}

	progress( fraction ) {

		if ( ! this.onProgress || fraction - this.lastProgress < 0.02 ) return;
		this.lastProgress = fraction;
		this.onProgress( fraction );

	}

	/**
	 * @param {{ format: string, numericOrder?: boolean, meta?: object }} options
	 * @returns canonical graph (format.js) with meta.report and, with groups, `planted`
	 */
	finish( options = {} ) {

		const format = options.format || 'graph';
		const total = this.names.length;
		const raw = this.edges, m = this.edgeCount;

		// 1. containers: parents of other nodes. Synthetic ones (subgraphs) and those without an
		// edge of their own are not nodes of the graph.
		const parents = this.parents;
		let remove = null, removed = 0;
		if ( parents || this.synthetic ) {

			const isParent = new Uint8Array( total );
			if ( parents ) for ( const k in parents ) isParent[ parents[ k ] ] = 1;
			const touched = new Uint8Array( total );
			for ( let e = 0; e < m; e ++ ) {

				if ( raw[ 2 * e ] !== raw[ 2 * e + 1 ] ) touched[ raw[ 2 * e ] ] = touched[ raw[ 2 * e + 1 ] ] = 1;

			}

			remove = new Uint8Array( total );
			for ( let i = 0; i < total; i ++ ) {

				if ( ( this.synthetic && this.synthetic[ i ] ) || ( isParent[ i ] && ! touched[ i ] ) ) {

					remove[ i ] = 1;
					removed ++;

				}

			}

			if ( removed === 0 ) remove = null;

		}

		const n = total - removed;
		if ( n <= 0 ) throw new GraphError( `${this.name}: no nodes found (${format})`, 'empty' );
		if ( n > this.limits.maxNodes ) throw tooLarge( 'nodes', n, this.limits );

		// 2. node order: old index -> new index (-1 = removed)
		let map = null;
		const numeric = !! options.numericOrder && ! remove && this.names.every( ( s ) => INT_RE.test( s ) );
		if ( numeric ) {

			const names = this.names;
			const perm = Array.from( names.keys() ).sort( ( x, y ) => Number( names[ x ] ) - Number( names[ y ] ) || ( names[ x ] < names[ y ] ? - 1 : names[ x ] > names[ y ] ? 1 : 0 ) );
			let sorted = true;
			for ( let k = 0; k < total && sorted; k ++ ) sorted = perm[ k ] === k;
			if ( ! sorted ) {

				map = new Int32Array( total );
				perm.forEach( ( p, k ) => ( map[ p ] = k ) );

			}

		} else if ( remove ) {

			map = new Int32Array( total );
			for ( let i = 0, k = 0; i < total; i ++ ) map[ i ] = remove[ i ] ? - 1 : k ++;

		}

		const at = map ? ( i ) => map[ i ] : ( i ) => i;

		// 3. edges
		let pairs = raw.subarray( 0, 2 * m ), lost = 0;
		if ( map ) {

			const out = new Float64Array( 2 * m ); // -1 survives (an invalid end point)
			for ( let k = 0; k < 2 * m; k ++ ) out[ k ] = map[ raw[ k ] ];
			pairs = out;
			if ( remove ) for ( let e = 0; e < m; e ++ ) if ( out[ 2 * e ] < 0 || out[ 2 * e + 1 ] < 0 ) lost ++;

		}

		const cleaned = cleanEdges( n, pairs, this.weights ? this.weights.subarray( 0, m ) : null );
		if ( cleaned.edges.length / 2 > this.limits.maxEdges ) throw tooLarge( 'edges', cleaned.edges.length / 2, this.limits );
		for ( const w of weightWarnings( cleaned.weightIssues, this.limitedWeights ) ) this.warn( w );
		const dropped = { selfLoops: cleaned.dropped.selfLoops, duplicates: cleaned.dropped.duplicates, invalid: cleaned.dropped.invalid - lost + this.invalid };

		// 4. groups from containment
		let planted = null, groupColumn = null;
		if ( parents ) {

			const made = this._containment( total, n, at, remove );
			if ( made ) {

				planted = made.planted;
				groupColumn = made.column;

			}

		}

		// 5. attributes and labels
		const report = {
			format, nodes: n, edges: cleaned.edges.length / 2,
			directed: !! this.directed, mirrored: cleaned.mirrored, weighted: !! cleaned.weights,
			dropped, labels: 'ids', attributes: [], droppedAttributes: [], groups: null,
			containers: removed, warnings: this.warnings, warningCount: this.warningCount,
		};
		const columnAt = ( col, dflt ) => {

			// raw values in the final node order
			const out = new Array( n );
			for ( let i = 0; i < total; i ++ ) {

				const k = at( i );
				if ( k < 0 ) continue;
				const v = col[ i ];
				out[ k ] = isMissing( v ) ? dflt : v;

			}

			return out;

		};

		let labels = null;
		if ( this.hasLabels ) {

			labels = new Array( n );
			for ( let i = 0; i < total; i ++ ) {

				const k = at( i );
				if ( k >= 0 ) labels[ k ] = this.labels[ i ] !== undefined && this.labels[ i ] !== '' ? this.labels[ i ] : this.names[ i ];

			}

			report.labels = 'label';

		}

		const lower = new Map();
		for ( const name of this.columns.keys() ) if ( ! lower.has( name.toLowerCase() ) ) lower.set( name.toLowerCase(), name );
		let labelColumn = null;
		if ( ! labels ) {

			for ( const key of LABEL_KEYS ) {

				const name = lower.get( key );
				if ( name === undefined ) continue;
				const values = columnAt( this.columns.get( name ), this.defaults.get( name ) );
				let filled = 0;
				for ( let i = 0; i < n; i ++ ) if ( ! isMissing( values[ i ] ) ) filled ++;
				if ( filled * 2 < n ) continue;
				labels = new Array( n );
				for ( let i = 0; i < total; i ++ ) {

					const k = at( i );
					if ( k >= 0 ) labels[ k ] = isMissing( values[ k ] ) ? this.names[ i ] : String( values[ k ] );

				}

				labelColumn = name;
				report.labels = `attribute "${name}"`;
				break;

			}

		}

		if ( ! labels ) {

			labels = new Array( n );
			for ( let i = 0; i < total; i ++ ) {

				const k = at( i );
				if ( k >= 0 ) labels[ k ] = this.names[ i ];

			}

		}

		const attributes = {};
		let kept = 0;
		const groupKeyOf = ( name ) => ( this.groupColumns.has( name ) ? 0 : GROUP_KEYS.indexOf( name.toLowerCase() ) );
		let groupLevel = null, groupSource = null, groupRank = Infinity;
		const add = ( name, typed ) => {

			if ( kept >= MAX_ATTRIBUTES ) {

				report.droppedAttributes.push( { name, reason: `more than ${MAX_ATTRIBUTES} attributes` } );
				return;

			}

			attributes[ name ] = typed;
			kept ++;
			report.attributes.push( typed.type === 'category' ? { name, type: 'category', categories: typed.categories.length } : { name, type: 'number' } );

		};

		if ( groupColumn ) add( this.columns.has( 'group' ) ? 'parent' : 'group', groupColumn );
		for ( const [ name, col ] of this.columns ) {

			const values = columnAt( col, this.defaults.get( name ) );
			const typed = typeColumn( values, n, groupKeyOf( name ) >= 0 );
			// a group column gives the planted partition even when it has too many values to be
			// kept as a category
			const rank = groupKeyOf( name );
			if ( rank >= 0 && rank < groupRank && ! planted && typed.kind !== 'empty' && typed.kind !== 'constant' ) {

				const d = distinctIds( values, n );
				if ( d.count >= 2 && d.count <= n / 2 ) {

					groupLevel = d.ids;
					groupSource = `attribute "${name}"`;
					groupRank = rank;

				}

			}

			if ( name === labelColumn ) continue; // used as the labels
			if ( typed.kind === 'empty' ) report.droppedAttributes.push( { name, reason: 'no values' } );
			else if ( typed.kind === 'constant' ) report.droppedAttributes.push( { name, reason: 'the same value for every node' } );
			else if ( typed.kind === 'text' ) report.droppedAttributes.push( { name, reason: `text with ${fmt( typed.distinct )} distinct values` } );
			else add( name, typed.kind === 'number' ? { type: 'number', values: typed.values } : { type: 'category', values: typed.values, categories: typed.categories } );

		}

		for ( const name of this.overflowColumns ) report.droppedAttributes.push( { name, reason: `more than ${MAX_COLUMNS} columns` } );

		if ( ! planted && groupLevel ) planted = { levels: [ groupLevel ], complete: false, source: groupSource };
		if ( planted ) report.groups = { source: planted.source, count: countDistinct( planted.levels[ 0 ] ), levels: planted.levels.length };

		const g = {
			format: FORMAT,
			version: VERSION,
			name: this.name,
			description: this.description || '',
			source: '',
			nodeCount: n,
			labels,
			edges: cleaned.edges,
			weights: cleaned.weights,
			attributes,
			hierarchy: null,
			meta: { format, ...( options.meta || {} ), report },
			dropped,
		};
		if ( this.directed ) g.directed = true;
		if ( planted ) Object.defineProperty( g, 'planted', { value: planted, enumerable: false, writable: true } );
		return g;

	}

	/** Group levels (finest first) and a category column from the parent pointers. */
	_containment( total, n, at, remove ) {

		const parents = this.parents;
		const chainOf = ( i ) => {

			// a container that stays a node is a member of its own group
			const chain = [];
			let p = this._isContainer( i ) ? i : parents[ i ];
			while ( p !== undefined && chain.length < 64 && ! chain.includes( p ) ) {

				chain.push( p );
				p = parents[ p ];

			}

			return chain;

		};

		let depth = 0;
		const chains = new Array( total );
		for ( let i = 0; i < total; i ++ ) {

			if ( at( i ) < 0 ) continue;
			const c = chainOf( i );
			chains[ i ] = c;
			if ( c.length > depth ) depth = c.length;

		}

		if ( depth === 0 ) return null;
		const L = Math.min( depth, MAX_GROUP_LEVELS );
		const levels = [];
		for ( let l = 0; l < L; l ++ ) {

			const level = new Int32Array( n );
			const ids = new Map();
			for ( let i = 0; i < total; i ++ ) {

				const k = at( i );
				if ( k < 0 ) continue;
				const c = chains[ i ];
				// level L-1 is the outermost container; shallower chains repeat their innermost one
				const key = c.length ? c[ Math.max( 0, c.length - 1 - ( L - 1 - l ) ) ] : - 1;
				let id = ids.get( key );
				if ( id === undefined ) {

					id = ids.size;
					ids.set( key, id );

				}

				level[ k ] = id;

			}

			levels.push( level );

		}

		// the innermost group as a category (when there are few enough to tell apart)
		let column = null;
		const names = new Map();
		const values = new Int32Array( n ).fill( - 1 );
		for ( let i = 0; i < total; i ++ ) {

			const k = at( i );
			if ( k < 0 || ! chains[ i ].length ) continue;
			const p = chains[ i ][ 0 ];
			let id = names.get( p );
			if ( id === undefined ) {

				id = names.size;
				names.set( p, id );

			}

			values[ k ] = id;

		}

		if ( names.size >= 2 && names.size <= MAX_CATEGORIES ) column = { type: 'category', values, categories: [ ...names.keys() ].map( ( p ) => this._labelOf( p ) ) };
		return { planted: { levels, complete: false, source: 'groups of the file' }, column };

	}

	/** The label a node would get: its explicit label, else its label / name / title value, else its id. */
	_labelOf( i ) {

		if ( this.labels[ i ] !== undefined && this.labels[ i ] !== '' ) return this.labels[ i ];
		for ( const [ name, col ] of this.columns ) {

			if ( LABEL_KEYS.includes( name.toLowerCase() ) && ! isMissing( col[ i ] ) ) return String( col[ i ] );

		}

		return this.names[ i ];

	}

	_isContainer( i ) {

		if ( ! this._containers ) {

			this._containers = new Set();
			for ( const k in this.parents ) this._containers.add( this.parents[ k ] );

		}

		return this._containers.has( i );

	}

}

function countDistinct( level ) {

	return new Set( level ).size;

}

function distinctIds( values, n ) {

	const ids = new Map();
	const out = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) {

		const key = isMissing( values[ i ] ) ? '\u0000missing' : String( values[ i ] ).trim();
		let id = ids.get( key );
		if ( id === undefined ) {

			id = ids.size;
			ids.set( key, id );

		}

		out[ i ] = id;

	}

	return { ids: out, count: ids.size };

}

/**
 * Types one raw column: { kind: 'number', values: Float32Array } | { kind: 'category', values:
 * Int32Array, categories } | { kind: 'text' | 'empty' | 'constant' }, each with `distinct` (the
 * number of distinct values; Infinity for numbers with more than MAX_CATEGORIES). With
 * preferCategory an integer column with few values is a category (class ids).
 */
export function typeColumn( values, n, preferCategory = false ) {

	let count = 0, numeric = true, integer = true;
	for ( let i = 0; i < n; i ++ ) {

		const v = values[ i ];
		if ( isMissing( v ) ) continue;
		count ++;
		if ( typeof v === 'number' ) {

			if ( ! Number.isFinite( v ) ) numeric = false;
			else if ( integer && ! Number.isInteger( v ) ) integer = false;

		} else if ( typeof v === 'string' && NUMBER_RE.test( v.trim() ) ) {

			if ( integer && ! /^[-+]?\d+$/.test( v.trim() ) ) integer = false;

		} else numeric = false;

	}

	if ( count === 0 ) return { kind: 'empty', distinct: 0 };

	if ( numeric ) {

		const out = new Float32Array( n );
		let min = Infinity, max = - Infinity;
		const seen = new Set();
		for ( let i = 0; i < n; i ++ ) {

			const v = values[ i ];
			const x = isMissing( v ) ? NaN : + v;
			out[ i ] = x;
			if ( x === x ) {

				if ( x < min ) min = x;
				if ( x > max ) max = x;
				if ( seen.size <= MAX_CATEGORIES ) seen.add( x );

			}

		}

		if ( min === max && count === n ) return { kind: 'constant', distinct: 1 };
		if ( preferCategory && integer && seen.size <= MAX_CATEGORIES ) {

			// a class id: categories in numeric order
			const sorted = [ ...seen ].sort( ( a, b ) => a - b );
			const index = new Map( sorted.map( ( x, k ) => [ x, k ] ) );
			const cat = new Int32Array( n );
			for ( let i = 0; i < n; i ++ ) cat[ i ] = out[ i ] === out[ i ] ? index.get( out[ i ] ) : - 1;
			return { kind: 'category', values: cat, categories: sorted.map( String ), distinct: sorted.length };

		}

		return { kind: 'number', values: out, distinct: seen.size > MAX_CATEGORIES ? Infinity : seen.size };

	}

	const index = new Map();
	const cat = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) {

		const v = values[ i ];
		if ( isMissing( v ) ) {

			cat[ i ] = - 1;
			continue;

		}

		const key = String( v ).trim();
		let id = index.get( key );
		if ( id === undefined ) {

			id = index.size;
			index.set( key, id );

		}

		cat[ i ] = id;

	}

	if ( index.size === 1 && count === n ) return { kind: 'constant', distinct: 1 };
	// more values than a legend can show, or (beyond a handful of nodes) a different one per node
	if ( index.size > MAX_CATEGORIES || ( index.size === count && count > 8 ) ) return { kind: 'text', distinct: index.size };
	return { kind: 'category', values: cat, categories: [ ...index.keys() ], distinct: index.size };

}
