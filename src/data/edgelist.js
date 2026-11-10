// Text edge-list parsing into a canonical graph (see format.js).
//
// Accepted input: one edge per line, "a b", "a,b", "a;b" or "a<TAB>b"; a line with a single id
// declares an isolated node. Ids are arbitrary strings; quoted CSV fields ("Jean Valjean",
// "Fantine") are supported. Comment lines start with '#', '%' or '//'. A header line such as
// "source,target" / "from to" / "Source;Target;Weight" is skipped.
//
// Weights: with a header, the column named weight / value / count / strength / w / freq. Without
// one, the third column when it is a number in every one of the first 50 rows, unless those are
// all integers of at least 10^8 (time stamps). Other columns are ignored.
//
// The delimiter is detected from the first data lines: tab, then comma, then semicolon, else
// runs of whitespace. With an explicit delimiter, spaces inside fields are kept (names).
// Node order: when every id is a non-negative integer, nodes are sorted numerically (so node
// index order follows the file's numbering); otherwise by first appearance. Labels are the ids.
// Duplicate edges and reversed duplicates are merged (weights summed) and self loops dropped
// (undirected graph); the counts are in meta.report (builder.js).
//
// API
//   parseEdgeList( text, { name = 'Edge list', delimiter = 'auto', limits, onProgress } ) -> canonical graph
//       meta: { format: 'edgelist', delimiter, lines, edgeLines, skipped, header, weightColumn,
//               matrixMarket, report }
//   looksLikeEdgeList( text ) -> boolean   quick sniff: not JSON

import { GraphBuilder, GraphError } from './builder.js';

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

const MAX_LINE = 65536; // characters of a line that are looked at

/** Splits a line into fields; handles double-quoted fields with "" escapes. */
function split( line, delimiter, out ) {

	out.length = 0;
	// an edge is in the first few fields: a file that is one line of 20 MB must not cost seconds
	if ( line.length > MAX_LINE ) line = line.slice( 0, MAX_LINE );
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
 * target, from, to, …), when they are not integers while the next line's ids are, or when a
 * later field is named like a weight column and the line below has a number there.
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
	// "a,b,weight" over "x,y,2": a later column is named like a weight and holds numbers below
	if ( rows.length === 2 ) for ( let k = 2; k < rows[ 0 ].length; k ++ ) if ( WEIGHT_NAMES.test( rows[ 0 ][ k ].trim() ) && NUMBER_RE.test( rows[ 1 ][ k ] || '' ) ) return 0;
	return - 1;

}

const WEIGHT_NAMES = /^(weights?|w|value|count|strength|freq|frequency)$/i;
const NUMBER_RE = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;

/**
 * Which column holds the edge weight (-1 = none): a header names it; without a header the third
 * column when it is a number in every one of the first rows and does not look like a timestamp.
 */
function detectWeightColumn( text, delimiter, matrixMarket, headerAt ) {

	const rows = [];
	let header = null, skipSize = matrixMarket, dataLine = - 1;
	for ( const line of lines( text ) ) {

		if ( isComment( line ) ) continue;
		if ( skipSize ) {

			skipSize = false;
			continue;

		}

		const fields = split( line, delimiter, [] );
		if ( ++ dataLine === headerAt ) header = fields;
		else rows.push( fields );
		if ( rows.length === 50 ) break;

	}

	if ( header ) {

		for ( let k = 2; k < header.length; k ++ ) if ( WEIGHT_NAMES.test( header[ k ].trim() ) ) return k;
		return - 1;

	}

	const edgeRows = rows.filter( ( r ) => r.length >= 2 && r[ 1 ] !== '' );
	if ( ! edgeRows.length ) return - 1;
	let stamps = true;
	for ( const r of edgeRows ) {

		if ( r.length < 3 || ! NUMBER_RE.test( r[ 2 ] ) ) return - 1;
		if ( stamps && ! ( /^\d+$/.test( r[ 2 ] ) && Number( r[ 2 ] ) >= 1e8 ) ) stamps = false;

	}

	return stamps ? - 1 : 2;

}

/**
 * @param {string} text
 * @param {{ name?: string, delimiter?: 'auto'|'whitespace'|','|';'|'\t', limits?: object, onProgress?: Function }} [options]
 */
export function parseEdgeList( text, options = {} ) {

	const name = options.name || 'Edge list';
	if ( typeof text !== 'string' ) throw new GraphError( 'Edge list: expected text', 'format' );
	if ( text.charCodeAt( 0 ) === 0xfeff ) text = text.slice( 1 );
	const matrixMarket = /^%%MatrixMarket/i.test( text );
	const delimiter = ! options.delimiter || options.delimiter === 'auto' ? detectDelimiter( text ) : options.delimiter;

	const b = new GraphBuilder( { name, limits: options.limits, onProgress: options.onProgress } );
	let lineCount = 0, edgeLines = 0, skipped = 0, header = null, badWeights = 0;
	let mmSizeLine = matrixMarket;
	const fields = [];
	const headerAt = detectHeader( text, delimiter, matrixMarket ); // index of the header among data lines, or -1
	const weightAt = detectWeightColumn( text, delimiter, matrixMarket, headerAt );
	let dataLine = - 1;
	const total = Math.max( 1, text.length );
	let chars = 0;

	for ( const line of lines( text ) ) {

		lineCount ++;
		chars += line.length + 1;
		if ( ( lineCount & 8191 ) === 0 ) b.progress( Math.min( 1, chars / total ) );
		if ( isComment( line ) ) continue;
		if ( mmSizeLine ) {

			mmSizeLine = false; // "rows cols nnz"
			continue;

		}

		split( line, delimiter, fields );
		const a = fields[ 0 ], c = fields[ 1 ];
		if ( ++ dataLine === headerAt ) {

			header = fields.slice();
			continue;

		}

		if ( ! a ) {

			skipped ++;
			continue;

		}

		if ( c === undefined || c === '' ) {

			b.node( a );
			continue;

		}

		let w = 1;
		if ( weightAt >= 0 ) {

			const f = fields[ weightAt ];
			w = f === undefined || f === '' ? NaN : Number( f );
			if ( ! Number.isFinite( w ) ) {

				if ( badWeights ++ === 0 ) b.warn( `line ${lineCount}: "${f === undefined ? '' : f}" is not a weight (counted as 1)` );
				w = 1;

			}

		}

		b.edge( b.node( a ), b.node( c ), w );
		edgeLines ++;

	}

	if ( b.nodeCount === 0 ) throw new GraphError( 'Edge list: no edges found', 'empty' );
	if ( badWeights > 1 ) b.warn( `${badWeights} weights are not numbers (counted as 1)` );

	return b.finish( {
		format: 'edgelist',
		numericOrder: true,
		meta: {
			delimiter: delimiter === '\t' ? 'tab' : delimiter,
			lines: lineCount,
			edgeLines,
			skipped,
			header: header || undefined,
			weightColumn: weightAt >= 0 ? ( header ? header[ weightAt ] : weightAt + 1 ) : undefined,
			matrixMarket: matrixMarket || undefined,
		},
	} );

}

/** True when the text does not look like JSON (first non-space character is not { or [). */
export function looksLikeEdgeList( text ) {

	const m = /\S/.exec( text.slice( 0, 4096 ) );
	return ! m || ( m[ 0 ] !== '{' && m[ 0 ] !== '[' );

}
