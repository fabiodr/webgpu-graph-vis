// Text edge-list parsing into a canonical graph (see format.js).
//
// Accepted input: one edge per line, "a b", "a,b", "a;b" or "a<TAB>b"; extra columns (weights,
// timestamps) are ignored; a line with a single id declares an isolated node. Ids are arbitrary
// strings; quoted CSV fields ("Jean Valjean","Fantine") are supported. Comment lines start with
// '#', '%' or '//'. A header line such as "source,target" / "from to" / "Source;Target;Weight"
// is skipped. Matrix Market coordinate files (%%MatrixMarket … then "rows cols nnz") work too.
//
// The delimiter is detected from the first data lines: tab, then comma, then semicolon, else
// runs of whitespace. With an explicit delimiter, spaces inside fields are kept (names).
// Node order: when every id is a non-negative integer, nodes are sorted numerically (so node
// index order follows the file's numbering); otherwise by first appearance. Labels are the ids.
// Duplicate edges, reversed duplicates and self loops are dropped (undirected graph).
//
// API
//   parseEdgeList( text, { name = 'Edge list', delimiter = 'auto' } ) -> canonical graph
//       meta: { format: 'edgelist', delimiter, lines, edgeLines, skipped, header, matrixMarket }
//   looksLikeEdgeList( text ) -> boolean   quick sniff used by the worker

import { normalizeGraph } from './format.js';

const HEADER_WORDS = new Set( [ 'source', 'target', 'from', 'to', 'src', 'dst', 'dest', 'destination', 'node1', 'node2', 'node_1', 'node_2', 'id1', 'id2', 'head', 'tail', 'nodea', 'nodeb', 'weight', 'value', 'source_id', 'target_id', 'from_id', 'to_id', 'sourceid', 'targetid' ] );

function isComment( line ) {

	const c = line.charCodeAt( 0 );
	return c === 35 /* # */ || c === 37 /* % */ || ( c === 47 /* / */ && line.charCodeAt( 1 ) === 47 );

}

/** Iterates trimmed, non-empty lines without building one huge array of all lines. */
function* lines( text ) {

	let pos = 0;
	const n = text.length;
	while ( pos < n ) {

		let end = text.indexOf( '\n', pos );
		if ( end < 0 ) end = n;
		let line = text.slice( pos, end );
		pos = end + 1;
		if ( line.length && line.charCodeAt( line.length - 1 ) === 13 ) line = line.slice( 0, - 1 );
		line = line.trim();
		if ( line.length ) yield line;

	}

}

function detectDelimiter( text ) {

	let tab = 0, comma = 0, semi = 0, seen = 0;
	for ( const line of lines( text ) ) {

		if ( isComment( line ) ) continue;
		if ( line.includes( '\t' ) ) tab ++;
		if ( line.includes( ',' ) ) comma ++;
		if ( line.includes( ';' ) ) semi ++;
		if ( ++ seen >= 20 ) break;

	}

	if ( ! seen ) return 'whitespace';
	const half = seen / 2;
	if ( tab >= half ) return '\t';
	if ( comma >= half ) return ',';
	if ( semi >= half ) return ';';
	return 'whitespace';

}

/** Splits a line into fields; handles double-quoted fields with "" escapes. */
function split( line, delimiter, out ) {

	out.length = 0;
	if ( delimiter === 'whitespace' && line.indexOf( '"' ) < 0 ) {

		const parts = line.split( /\s+/ );
		for ( const p of parts ) if ( p.length ) out.push( p );
		return out;

	}

	const n = line.length;
	let i = 0;
	const isSep = delimiter === 'whitespace' ? ( c ) => c === ' ' || c === '\t' : ( c ) => c === delimiter;
	while ( i < n ) {

		// skip leading spaces (and, for whitespace mode, runs of separators)
		while ( i < n && ( line[ i ] === ' ' || ( delimiter === 'whitespace' && isSep( line[ i ] ) ) ) ) i ++;
		if ( i >= n ) break;
		let field = '';
		if ( line[ i ] === '"' ) {

			i ++;
			while ( i < n ) {

				if ( line[ i ] === '"' ) {

					if ( line[ i + 1 ] === '"' ) {

						field += '"';
						i += 2;
						continue;

					}

					i ++;
					break;

				}

				field += line[ i ++ ];

			}

			while ( i < n && ! isSep( line[ i ] ) ) i ++;

		} else {

			const s = i;
			while ( i < n && ! isSep( line[ i ] ) ) i ++;
			field = line.slice( s, i ).trim();

		}

		out.push( field );
		if ( i < n && isSep( line[ i ] ) ) {

			i ++;
			// "a,,b": keep the empty field for explicit delimiters
			if ( delimiter !== 'whitespace' && i >= n ) out.push( '' );

		}

	}

	return out;

}

const INT_RE = /^\d+$/;

/**
 * The first data line is a header when both of its first fields are column names (source,
 * target, from, to, …), or when they are not integers while the next line's ids are.
 */
function detectHeader( text, delimiter, matrixMarket ) {

	const rows = [];
	let skipSize = matrixMarket;
	for ( const line of lines( text ) ) {

		if ( isComment( line ) ) continue;
		if ( skipSize ) {

			skipSize = false;
			continue;

		}

		rows.push( split( line, delimiter, [] ) );
		if ( rows.length === 2 ) break;

	}

	if ( ! rows.length ) return - 1;
	const [ a, b ] = rows[ 0 ];
	const word = ( s ) => HEADER_WORDS.has( ( s || '' ).toLowerCase().replace( /\s+/g, '_' ) );
	if ( word( a ) && ( b === undefined || word( b ) ) ) return 0;
	if ( rows.length === 2 && b !== undefined && ! INT_RE.test( a ) && ! INT_RE.test( b ) && INT_RE.test( rows[ 1 ][ 0 ] || '' ) && INT_RE.test( rows[ 1 ][ 1 ] || '' ) ) return 0;
	return - 1;

}

/**
 * @param {string} text
 * @param {{ name?: string, delimiter?: 'auto'|'whitespace'|','|';'|'\t' }} [options]
 */
export function parseEdgeList( text, options = {} ) {

	const name = options.name || 'Edge list';
	if ( typeof text !== 'string' ) throw new Error( 'Edge list: expected text' );
	if ( text.charCodeAt( 0 ) === 0xfeff ) text = text.slice( 1 );
	const matrixMarket = /^%%MatrixMarket/i.test( text );
	const delimiter = ! options.delimiter || options.delimiter === 'auto' ? detectDelimiter( text ) : options.delimiter;

	const ids = new Map(); // id string -> provisional index (first appearance)
	const names = [];
	const raw = []; // provisional index pairs
	let allInt = true;
	let lineCount = 0, edgeLines = 0, skipped = 0, header = null;
	let mmSizeLine = matrixMarket;
	const fields = [];
	const headerAt = detectHeader( text, delimiter, matrixMarket ); // index of the header among data lines, or -1
	let dataLine = - 1;
	const idOf = ( s ) => {

		let k = ids.get( s );
		if ( k === undefined ) {

			k = names.length;
			ids.set( s, k );
			names.push( s );
			if ( allInt && ! INT_RE.test( s ) ) allInt = false;

		}

		return k;

	};

	for ( const line of lines( text ) ) {

		lineCount ++;
		if ( isComment( line ) ) continue;
		if ( mmSizeLine ) {

			mmSizeLine = false; // "rows cols nnz"
			continue;

		}

		split( line, delimiter, fields );
		const a = fields[ 0 ], b = fields[ 1 ];
		if ( ++ dataLine === headerAt ) {

			header = fields.slice();
			continue;

		}

		if ( ! a ) {

			skipped ++;
			continue;

		}

		if ( b === undefined || b === '' ) {

			idOf( a );
			continue;

		}

		raw.push( idOf( a ), idOf( b ) );
		edgeLines ++;

	}

	const n = names.length;
	if ( n === 0 ) throw new Error( 'Edge list: no edges found' );

	// final node order
	let order = null; // provisional index -> final index
	let labels = names;
	if ( allInt ) {

		const perm = Array.from( names.keys() ).sort( ( x, y ) => Number( names[ x ] ) - Number( names[ y ] ) || ( names[ x ] < names[ y ] ? - 1 : 1 ) );
		order = new Int32Array( n );
		labels = new Array( n );
		perm.forEach( ( p, k ) => {

			order[ p ] = k;
			labels[ k ] = names[ p ];

		} );

	}

	const edges = new Uint32Array( raw.length );
	for ( let k = 0; k < raw.length; k ++ ) edges[ k ] = order ? order[ raw[ k ] ] : raw[ k ];

	return normalizeGraph( {
		name,
		nodeCount: n,
		labels,
		edges,
		meta: {
			format: 'edgelist',
			delimiter: delimiter === '\t' ? 'tab' : delimiter,
			lines: lineCount,
			edgeLines,
			skipped,
			header: header || undefined,
			matrixMarket: matrixMarket || undefined,
		},
	} );

}

/** True when the text does not look like JSON (first non-space character is not { or [). */
export function looksLikeEdgeList( text ) {

	const m = /\S/.exec( text.slice( 0, 4096 ) );
	return ! m || ( m[ 0 ] !== '{' && m[ 0 ] !== '[' );

}
