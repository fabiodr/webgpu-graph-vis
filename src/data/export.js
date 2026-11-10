// Writers: a canonical graph (format.js) as text in one of the formats the pipeline also reads,
// so what is on screen can be saved and loaded again. Pure JS; runs in the data worker.
//
//   exportGraph( g, format, { positions? } ) -> { text, mime, extension }
//     'json'     the canonical wgv-graph JSON (labels, weights, attributes, hierarchy); with
//                positions (Float32Array( 4N ), xyz per node) also "positions": [ x0, y0, z0, … ],
//                which the loader uses as start positions
//     'graphml'  nodes with label, the attributes, one cluster id per hierarchy level (level0 =
//                finest) and x / y / z when positions are given; edges with weight when weighted
//     'csv'      the edge list: source,target[,weight] with a header. Ids are the labels when
//                those are unique, else the node indices. Isolated nodes follow as single ids.
//   Reading the result back gives the same nodes, edges, weights and labels (tested); the
//   hierarchy survives only in 'json' (as the hierarchy) and in 'graphml' (as attributes).

import { serializeGraph } from './format.js';

function xml( s ) {

	return String( s ).replace( /[<>&"]/g, ( c ) => ( c === '<' ? '&lt;' : c === '>' ? '&gt;' : c === '&' ? '&amp;' : '&quot;' ) ).replace( /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, '' );

}

function round( x ) {

	return Math.round( x * 1000 ) / 1000;

}

function toGraphML( g, positions ) {

	const n = g.nodeCount, E = g.edges.length / 2;
	const out = [];
	out.push( '<?xml version="1.0" encoding="UTF-8"?>' );
	out.push( '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">' );
	const keys = []; // { id, write( i ) -> string | null }
	const addKey = ( name, type, write ) => {

		const id = `d${keys.length}`;
		out.push( `  <key id="${id}" for="node" attr.name="${xml( name )}" attr.type="${type}"/>` );
		keys.push( { id, write } );

	};

	if ( g.labels ) addKey( 'label', 'string', ( i ) => g.labels[ i ] );
	for ( const [ name, a ] of Object.entries( g.attributes || {} ) ) {

		if ( a.type === 'category' ) addKey( name, 'string', ( i ) => ( a.values[ i ] >= 0 ? a.categories[ a.values[ i ] ] : null ) );
		else addKey( name, 'double', ( i ) => ( Number.isFinite( a.values[ i ] ) ? a.values[ i ] : null ) );

	}

	const levels = g.hierarchy && g.hierarchy.levels ? g.hierarchy.levels : [];
	levels.forEach( ( level, l ) => addKey( `level${l}`, 'int', ( i ) => level[ i ] ) );
	if ( positions ) [ 'x', 'y', 'z' ].forEach( ( axis, k ) => addKey( axis, 'double', ( i ) => round( positions[ 4 * i + k ] ) ) );
	if ( g.weights ) out.push( '  <key id="w" for="edge" attr.name="weight" attr.type="double"/>' );
	out.push( `  <graph id="${xml( g.name || 'G' )}" edgedefault="undirected">` );
	for ( let i = 0; i < n; i ++ ) {

		let data = '';
		for ( const key of keys ) {

			const v = key.write( i );
			if ( v !== null && v !== undefined ) data += `<data key="${key.id}">${xml( v )}</data>`;

		}

		out.push( data ? `    <node id="n${i}">${data}</node>` : `    <node id="n${i}"/>` );

	}

	for ( let e = 0; e < E; e ++ ) {

		const s = g.edges[ 2 * e ], t = g.edges[ 2 * e + 1 ];
		out.push( g.weights ? `    <edge source="n${s}" target="n${t}"><data key="w">${g.weights[ e ]}</data></edge>` : `    <edge source="n${s}" target="n${t}"/>` );

	}

	out.push( '  </graph>' );
	out.push( '</graphml>' );
	return out.join( '\n' ) + '\n';

}

function csvField( s ) {

	const v = String( s );
	return /[",\n\r]/.test( v ) || v !== v.trim() || v === '' ? '"' + v.replace( /"/g, '""' ) + '"' : v;

}

function toCsv( g ) {

	const n = g.nodeCount, E = g.edges.length / 2;
	// labels serve as ids when they are unique and cannot be mistaken for a comment line
	let ids = null;
	if ( g.labels && new Set( g.labels ).size === n && g.labels.every( ( s ) => s !== '' && ! /^(#|%|\/\/)/.test( s ) ) ) ids = g.labels;
	const id = ( i ) => ( ids ? csvField( ids[ i ] ) : String( i ) );
	const out = [ g.weights ? 'source,target,weight' : 'source,target' ];
	const touched = new Uint8Array( n );
	for ( let e = 0; e < E; e ++ ) {

		const s = g.edges[ 2 * e ], t = g.edges[ 2 * e + 1 ];
		touched[ s ] = touched[ t ] = 1;
		out.push( g.weights ? `${id( s )},${id( t )},${g.weights[ e ]}` : `${id( s )},${id( t )}` );

	}

	for ( let i = 0; i < n; i ++ ) if ( ! touched[ i ] ) out.push( id( i ) );
	return out.join( '\n' ) + '\n';

}

export function exportGraph( g, format = 'json', options = {} ) {

	// (exactly 4N floats: a longer array is the readback of another graph)
	const positions = options.positions && options.positions.length === 4 * g.nodeCount ? options.positions : null;
	if ( format === 'json' ) {

		let text = serializeGraph( g );
		if ( positions ) {

			const xyz = new Array( 3 * g.nodeCount );
			for ( let i = 0; i < g.nodeCount; i ++ ) for ( let k = 0; k < 3; k ++ ) xyz[ 3 * i + k ] = round( positions[ 4 * i + k ] );
			text = text.slice( 0, - 1 ) + ',"positions":' + JSON.stringify( xyz ) + '}';

		}

		return { text, mime: 'application/json', extension: '.json' };

	}

	if ( format === 'graphml' ) return { text: toGraphML( g, positions ), mime: 'application/graphml+xml', extension: '.graphml' };
	if ( format === 'csv' ) return { text: toCsv( g ), mime: 'text/csv', extension: '.csv' };
	throw new Error( `export: unknown format "${format}" (json | graphml | csv)` );

}
