// File formats of the data pipeline: the registry behind data.formats(), detection, and the one
// entry point that turns the text of a file into a canonical graph. Pure JS; runs in the data
// worker, on the main thread and in Node.
//
// Detection looks at the content first and at the file name second, because dropped files are
// often misnamed (.txt for everything, .xml for GraphML, no extension at all):
//   starts with <              GraphML or GEXF, by the root element
//   starts with { or [         JSON; the shape is recognised after parsing (parse-json.js)
//   %%MatrixMarket             Matrix Market
//   *Vertices, *Network, …     Pajek
//   [strict] (di)graph … {     DOT
//   graph [                    GML
//   anything else              the format of the extension when that names one, else an edge list
// (comment lines of the respective format are skipped before the test). A file that starts
// with binary data is rejected with a message that says so.
//
// API
//   FORMATS                    [ { id, name, extensions: [ '.graphml', … ], description } ]
//   listFormats() -> copies of the above (what data.formats() returns; every format may also be
//                    gzipped, e.g. graph.graphml.gz)
//   acceptExtensions() -> string[]    all extensions and '.gz', for <input accept>
//   acceptsName( name ) -> boolean    for a drop filter: a registered extension (also under
//                    '.gz', in any case) or no extension at all. loadFile itself reads any file
//                    (the content decides, the extension only breaks ties), so a UI may as well
//                    take every file and show the error of the one that is not a graph.
//   detectFormat( text, name ) -> { id, by: 'content' | 'extension' | 'default' }
//   parseGraph( text, name = 'graph', context = {} ) -> { graph, format, id }
//       graph   canonical graph (format.js) with meta.report (builder.js) and, when the file has
//               groups, `planted`
//       format  what was recognised, in detail: 'graphml', 'gexf', 'gml', 'dot', 'pajek', 'mtx',
//               'edgelist', 'wgv-graph', 'igv', 'node-link', 'cytoscape', 'graphology',
//               'adjacency', 'edge-array'
//       id      the FORMATS entry
//       context { limits?: { maxNodes, maxEdges }, onProgress?( fraction ), name? }
//   Errors are GraphError (builder.js): message for the user, code 'parse' | 'format' | 'empty' |
//   'too-large'.

import { normalizeGraph, fromIgv } from './format.js';
import { GraphError, DEFAULT_LIMITS, tooLarge, weightWarnings } from './builder.js';
import { parseEdgeList } from './edgelist.js';
import { parseGraphML, parseGexf } from './parse-xml.js';
import { parseGml, parseDot, parsePajek, parseMtx } from './parse-text.js';
import { detectJsonShape, parseJsonShape, JSON_SHAPES } from './parse-json.js';

export const FORMATS = [
	{ id: 'graphml', name: 'GraphML', extensions: [ '.graphml', '.xml' ], description: 'GraphML, including yEd files: node attributes, edge weights, nested graphs as groups' },
	{ id: 'gexf', name: 'GEXF (Gephi)', extensions: [ '.gexf' ], description: 'GEXF 1.1 to 1.3: attributes, weights, modularity_class, parent nodes as groups' },
	{ id: 'gml', name: 'GML', extensions: [ '.gml' ], description: 'Graph Modelling Language: node attributes, value / weight on edges' },
	{ id: 'dot', name: 'Graphviz DOT', extensions: [ '.dot', '.gv' ], description: 'The common subset of DOT; subgraphs become groups' },
	{ id: 'pajek', name: 'Pajek', extensions: [ '.net', '.paj' ], description: 'Pajek networks: vertices, arcs, edges, lists, matrix, partitions' },
	{ id: 'mtx', name: 'Matrix Market', extensions: [ '.mtx' ], description: 'Sparse or dense adjacency matrix; a rectangular matrix is a bipartite graph' },
	{ id: 'edgelist', name: 'Edge list (CSV, TSV, text)', extensions: [ '.csv', '.tsv', '.txt', '.edges', '.edgelist', '.el', '.ssv', '.tab', '.ncol', '.dat' ], description: 'One edge per line; optional header, weight column, quoted names, comments' },
	{ id: 'json', name: 'JSON', extensions: [ '.json' ], description: 'wgv-graph, d3 / networkx node-link, Cytoscape.js, graphology / sigma, adjacency lists' },
];

const BY_EXTENSION = new Map();
for ( const f of FORMATS ) for ( const ext of f.extensions ) BY_EXTENSION.set( ext, f.id );

export function listFormats() {

	return FORMATS.map( ( f ) => ( { id: f.id, name: f.name, extensions: f.extensions.slice(), description: f.description } ) );

}

export function acceptExtensions() {

	const out = [];
	for ( const f of FORMATS ) out.push( ...f.extensions );
	out.push( '.gz' );
	return out;

}

export function acceptsName( name ) {

	const base = String( name || '' ).toLowerCase().replace( /\.gz$/, '' );
	const dot = base.lastIndexOf( '.' );
	if ( dot <= 0 ) return true; // no extension: detection reads the content
	return BY_EXTENSION.has( base.slice( dot ) );

}

function cleanName( name ) {

	return String( name || '' ).split( /[\\/]/ ).pop().split( '?' )[ 0 ].split( '#' )[ 0 ];

}

export function baseName( name ) {

	const file = cleanName( name || 'graph' );
	return file.replace( /\.gz$/i, '' ).replace( /\.igv\.json$/i, '' ).replace( /\.[a-z0-9]+$/i, '' ) || 'graph';

}

function extensionOf( name ) {

	const m = /(\.[a-z0-9]+)$/i.exec( cleanName( name ).replace( /\.gz$/i, '' ) );
	return m ? m[ 1 ].toLowerCase() : '';

}

/** The head of the text without leading blank and comment lines (per comment style). */
function skipComments( head, style ) {

	let s = head;
	for ( ;; ) {

		s = s.replace( /^\s+/, '' );
		if ( style === 'c' && s.startsWith( '/*' ) ) {

			const end = s.indexOf( '*/' );
			if ( end < 0 ) return '';
			s = s.slice( end + 2 );

		} else if ( ( style === 'c' && ( s.startsWith( '//' ) || s.startsWith( '#' ) ) ) || ( style === '#' && s.startsWith( '#' ) ) || ( style === '%' && s.startsWith( '%' ) ) ) {

			const end = s.indexOf( '\n' );
			if ( end < 0 ) return '';
			s = s.slice( end + 1 );

		} else return s;

	}

}

export function detectFormat( text, name = '' ) {

	const start = text.charCodeAt( 0 ) === 0xfeff ? 1 : 0;
	const head = text.slice( start, start + 8192 );
	if ( head.indexOf( '\u0000' ) >= 0 ) throw new GraphError( `${cleanName( name ) || 'This file'} is not a text file (it contains binary data). Supported: ${FORMATS.map( ( f ) => f.name ).join( ', ' )}.`, 'format' );
	const ext = extensionOf( name );
	const byExt = BY_EXTENSION.get( ext ) || null;
	const first = /\S/.exec( head );
	if ( ! first ) throw new GraphError( `${cleanName( name ) || 'The file'} is empty`, 'empty' );

	if ( first[ 0 ] === '<' ) {

		if ( /<(?:[\w.-]+:)?graphml[\s>]/i.test( head ) ) return { id: 'graphml', by: 'content' };
		if ( /<(?:[\w.-]+:)?gexf[\s>]/i.test( head ) ) return { id: 'gexf', by: 'content' };
		if ( byExt === 'graphml' || byExt === 'gexf' ) return { id: byExt, by: 'extension' };
		const root = /<([A-Za-z_][\w:.-]*)/.exec( head.replace( /<\?[\s\S]*?\?>|<!--[\s\S]*?-->|<![^>]*>/g, '' ) );
		throw new GraphError( `${cleanName( name ) || 'This file'} is XML with the root element <${root ? root[ 1 ] : '?'}>; the XML formats read here are GraphML (<graphml>) and GEXF (<gexf>)`, 'format' );

	}

	if ( first[ 0 ] === '{' || first[ 0 ] === '[' ) return { id: 'json', by: 'content' };
	if ( /^%%MatrixMarket\b/i.test( head ) ) return { id: 'mtx', by: 'content' };
	if ( /^\*\s*(vertices|network|arcs|edges|arcslist|edgeslist|matrix)\b/i.test( skipComments( head, '%' ) ) ) return { id: 'pajek', by: 'content' };
	if ( /^(strict\s+)?(di)?graph\b\s*("(?:[^"\\]|\\.)*"|[\w.\-\u0080-￿]+)?\s*\{/i.test( skipComments( head, 'c' ) ) ) return { id: 'dot', by: 'content' };
	const gml = skipComments( head, '#' );
	if ( /^graph\s*\[/i.test( gml ) || /^(?:[A-Za-z_]\w*[ \t]+(?:"[^"]*"|[^\s[\]]+)\s+){1,8}graph\s*\[/.test( gml ) ) return { id: 'gml', by: 'content' };
	if ( byExt && byExt !== 'edgelist' && byExt !== 'json' ) return { id: byExt, by: 'extension' };
	if ( byExt === 'json' ) return { id: 'json', by: 'extension' };
	return { id: 'edgelist', by: byExt ? 'extension' : 'default' };

}

function checkLimits( g, limits ) {

	const lim = { ...DEFAULT_LIMITS, ...( limits || {} ) };
	if ( g.nodeCount > lim.maxNodes ) throw tooLarge( 'nodes', g.nodeCount, lim );
	if ( g.edges.length / 2 > lim.maxEdges ) throw tooLarge( 'edges', g.edges.length / 2, lim );

}

/** The report (builder.js) for a graph that came in the canonical or the igv shape. */
function reportOf( g, format, raw, warnings = [] ) {

	const all = [ ...warnings, ...weightWarnings( g.weightIssues ) ];
	return {
		format, nodes: g.nodeCount, edges: g.edges.length / 2,
		directed: !! ( raw && raw.directed ), mirrored: !! g.mirrored, weighted: !! g.weights,
		dropped: g.dropped || { selfLoops: 0, duplicates: 0, invalid: 0 },
		labels: g.labels ? 'label' : 'ids',
		attributes: Object.entries( g.attributes || {} ).map( ( [ name, a ] ) => ( a.type === 'category' ? { name, type: 'category', categories: a.categories.length } : { name, type: 'number' } ) ),
		droppedAttributes: [], groups: null, containers: 0, warnings: all, warningCount: all.length,
	};

}

/** True when a stored hierarchy can be read: an array of levels, each an array with one entry per node. */
function usableHierarchy( h, n ) {

	if ( ! h || ! Array.isArray( h.levels ) ) return false;
	for ( const level of h.levels ) if ( ! ( Array.isArray( level ) || ArrayBuffer.isView( level ) ) || level.length !== n ) return false;
	return true;

}

function parseJson( text, name, context ) {

	let json;
	try {

		json = JSON.parse( text );

	} catch ( e ) {

		throw new GraphError( `${baseName( name )}: invalid JSON (${e.message})`, 'parse' );

	}

	const shape = detectJsonShape( json );
	if ( shape === 'wgv-graph' || shape === 'igv' ) {

		// cheap size check before the arrays are copied
		const lim = { ...DEFAULT_LIMITS, ...( context.limits || {} ) };
		const count = shape === 'igv' ? json.nodes.length : json.nodeCount | 0;
		if ( count > lim.maxNodes ) throw tooLarge( 'nodes', count, lim );
		if ( ! ( count > 0 ) ) throw new GraphError( `${baseName( name )}: the graph has no nodes`, 'empty' );
		const warnings = [];
		let g;
		try {

			if ( shape === 'igv' ) g = fromIgv( json, context.name );
			else {

				// a stored hierarchy that does not fit the nodes is dropped, not fatal: a new one is computed
				let raw = json;
				if ( json.hierarchy !== undefined && json.hierarchy !== null && ! usableHierarchy( json.hierarchy, count ) ) {

					raw = { ...json, hierarchy: null };
					warnings.push( 'the stored hierarchy does not have one entry per node in every level and was ignored' );

				}

				g = normalizeGraph( raw );

			}

		} catch ( e ) {

			if ( e instanceof GraphError ) throw e;
			throw new GraphError( `${baseName( name )}: looks like ${shape === 'igv' ? 'an ImsvGraphVis .igv.json file' : 'a wgv-graph file'} but cannot be read (${String( ( e && e.message ) || e ).replace( /^Graph: /, '' )})`, 'parse' );

		}

		checkLimits( g, context.limits );
		g.meta = { ...( g.meta || {} ), report: reportOf( g, shape, json, warnings ) };
		return { graph: g, format: shape };

	}

	if ( ! shape ) throw new GraphError( `${baseName( name )}: this JSON is not a graph in a known shape (expected ${JSON_SHAPES})`, 'format' );
	const graph = parseJsonShape( json, shape, context );
	return { graph, format: shape };

}

const PARSERS = {
	graphml: parseGraphML,
	gexf: parseGexf,
	gml: parseGml,
	dot: parseDot,
	pajek: parsePajek,
	mtx: parseMtx,
	edgelist: ( text, context ) => parseEdgeList( text, context ),
};

/**
 * Text of a graph file -> canonical graph (see the header).
 * @param {string} text
 * @param {string} [name] file name or URL (extension as a hint, base name as the graph's name)
 * @param {{ limits?: object, onProgress?: Function, name?: string }} [context]
 */
export function parseGraph( text, name = 'graph', context = {} ) {

	if ( typeof text !== 'string' ) throw new GraphError( 'parseGraph: expected text', 'format' );
	if ( text.charCodeAt( 0 ) === 0xfeff ) text = text.slice( 1 ); // a byte order mark (files and downloads lose it in the decoder)
	const ctx = { ...context, name: context.name || baseName( name ) };
	const { id } = detectFormat( text, name );
	if ( id === 'json' ) {

		const { graph, format } = parseJson( text, name, ctx );
		return { graph, format, id };

	}

	const graph = PARSERS[ id ]( text, ctx );
	return { graph, format: graph.meta.format, id };

}
