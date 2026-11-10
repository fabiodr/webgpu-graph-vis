// JSON graph shapes other than the canonical one. Each function takes the parsed JSON value and
// a context ({ name, limits, onProgress }) and writes into a GraphBuilder (builder.js), so
// attributes, labels, groups, weights and the "what was dropped" report work as for the text
// formats. Pure JS.
//
//  node-link   d3-force / networkx node_link_data / sigma v1:
//              { nodes: [ { id, …attributes } | id ], links | edges: [ { source, target, weight |
//              value } ], directed? }. source / target are ids, node objects, or (when no node
//              has that id) indices into nodes. Layout fields (x, y, vx, fx, index, …) are skipped.
//  cytoscape   Cytoscape.js: { elements: { nodes: [ { data: { id, parent, … } } ], edges: [ { data:
//              { source, target, weight } } ] } }, { elements: [ … ] } or the bare array; an
//              element is an edge when its data has source and target. `parent` (compound nodes)
//              gives groups.
//  graphology  graphology / sigma v2 export: { options: { type }, attributes: { name }, nodes:
//              [ { key, attributes } ], edges: [ { source, target, attributes: { weight },
//              undirected? } ] }
//  adjacency   { "a": [ "b", "c" ], … }, { "a": { "b": 1.5 } | { "b": { weight: 1.5 } }, … }
//              (networkx dict of dicts), [ [ 1, 2 ], [ 0 ], [ 0 ] ] (array of neighbour index
//              lists), networkx adjacency_data { nodes: [ { id } ], adjacency: [ [ { id, weight } ] ] }
//  edge-array  [ [ a, b ], [ a, c, weight ], … ] or [ { source, target, weight }, … ]
//
// API
//   detectJsonShape( json ) -> 'wgv-graph' | 'igv' | 'graphology' | 'cytoscape' | 'adjacency' |
//                              'node-link' | 'edge-array' | null
//   parseJsonShape( json, shape, context ) -> canonical graph   (not for wgv-graph / igv: format.js)
//   JSON_SHAPES: the list above as text, for error messages

import { GraphBuilder, GraphError } from './builder.js';

export const JSON_SHAPES = 'wgv-graph, ImsvGraphVis .igv.json, node-link { nodes, links }, Cytoscape.js { elements }, graphology { nodes: [ { key } ], edges }, adjacency lists, or an array of edges';

const SKIP_NODE_KEYS = new Set( [ 'id', 'key', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'index', 'parent', 'position', 'selected', 'locked', 'grabbable', 'selectable', 'classes' ] );

function isObject( v ) {

	return v !== null && typeof v === 'object' && ! Array.isArray( v );

}

function isScalar( v ) {

	return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean';

}

function weightOf( o ) {

	if ( ! isObject( o ) ) return 1;
	const w = o.weight !== undefined ? o.weight : o.value;
	const x = typeof w === 'string' ? Number( w ) : w;
	return typeof x === 'number' && Number.isFinite( x ) ? x : 1;

}

function nodeAttributes( b, i, o, skip = SKIP_NODE_KEYS ) {

	for ( const key in o ) {

		if ( skip.has( key ) ) continue;
		const v = o[ key ];
		if ( isScalar( v ) ) b.attr( i, key, v );

	}

}

export function detectJsonShape( json ) {

	if ( isObject( json ) ) {

		if ( json.format === 'wgv-graph' ) return 'wgv-graph';
		if ( Array.isArray( json.nodes ) && Array.isArray( json.links ) && json.clusters && json.nodes[ 0 ] && typeof json.nodes[ 0 ] === 'object' && 'ancIdxs' in json.nodes[ 0 ] ) return 'igv';
		if ( json.elements && ( Array.isArray( json.elements ) || isObject( json.elements ) ) ) return 'cytoscape';
		if ( Array.isArray( json.nodes ) && Array.isArray( json.adjacency ) ) return 'adjacency';
		if ( Array.isArray( json.nodes ) ) {

			const first = json.nodes[ 0 ];
			if ( isObject( first ) && 'key' in first && ! ( 'id' in first ) ) return 'graphology';
			if ( isObject( first ) && isObject( first.data ) ) return 'cytoscape';
			return 'node-link';

		}

		const values = Object.values( json );
		if ( values.length && ( values.every( Array.isArray ) || values.every( isObject ) ) ) return 'adjacency';
		return null;

	}

	if ( Array.isArray( json ) && json.length ) {

		const first = json[ 0 ];
		if ( isObject( first ) ) {

			if ( isObject( first.data ) || first.group === 'nodes' || first.group === 'edges' ) return 'cytoscape';
			if ( 'source' in first && 'target' in first ) return 'edge-array';
			return null;

		}

		if ( Array.isArray( first ) ) {

			// neighbour lists (every entry an index below the length, symmetric) or edge pairs
			const n = json.length;
			let lists = true;
			for ( let i = 0; i < n && lists; i ++ ) {

				const row = json[ i ];
				if ( ! Array.isArray( row ) ) return null;
				for ( const v of row ) {

					if ( ! Number.isInteger( v ) || v < 0 || v >= n || ! Array.isArray( json[ v ] ) || ! json[ v ].includes( i ) ) {

						lists = false;
						break;

					}

				}

			}

			if ( lists ) return 'adjacency';
			return json.every( ( row ) => Array.isArray( row ) && ( row.length === 2 || row.length === 3 ) ) ? 'edge-array' : null;

		}

	}

	return null;

}

function nodeLink( json, context ) {

	const b = new GraphBuilder( context );
	if ( typeof json.name === 'string' && json.name ) b.name = json.name;
	else if ( isObject( json.graph ) && typeof json.graph.name === 'string' && json.graph.name ) b.name = json.graph.name;
	const nodes = json.nodes;
	const links = Array.isArray( json.links ) ? json.links : Array.isArray( json.edges ) ? json.edges : [];
	const n = nodes.length;
	for ( let i = 0; i < n; i ++ ) {

		const node = nodes[ i ];
		if ( isObject( node ) ) {

			const idx = b.node( node.id !== undefined && node.id !== null ? node.id : i );
			if ( idx !== i ) b.warn( `node ${i}: the id "${node.id}" is used twice` );
			nodeAttributes( b, idx, node );

		} else b.node( node !== undefined && node !== null ? node : i );

	}

	const declared = b.nodeCount;
	const resolve = ( v ) => {

		if ( isObject( v ) ) v = v.id;
		if ( v === undefined || v === null ) return - 1;
		if ( b.has( v ) ) return b.node( v );
		if ( Number.isInteger( v ) && v >= 0 && v < declared ) return v;
		return - 1;

	};

	const directed = json.directed === true;
	b.directed = directed;
	let bad = 0;
	for ( let k = 0; k < links.length; k ++ ) {

		const l = links[ k ];
		let a = - 1, c = - 1, w = 1;
		if ( Array.isArray( l ) ) {

			a = resolve( l[ 0 ] );
			c = resolve( l[ 1 ] );
			if ( typeof l[ 2 ] === 'number' ) w = l[ 2 ];

		} else if ( isObject( l ) ) {

			a = resolve( l.source !== undefined ? l.source : l.from );
			c = resolve( l.target !== undefined ? l.target : l.to );
			w = weightOf( l );

		}

		if ( a < 0 || c < 0 ) {

			if ( bad ++ === 0 ) b.warn( `link ${k}: source or target is not a node` );
			b.invalidEdge();

		} else b.edge( a, c, w );

		if ( ( k & 16383 ) === 0 ) b.progress( k / links.length );

	}

	if ( bad > 1 ) b.warn( `${bad} links have a source or target that is not a node` );
	return b.finish( { format: 'node-link' } );

}

function cytoscape( json, context ) {

	const b = new GraphBuilder( context );
	if ( isObject( json ) && isObject( json.data ) && typeof json.data.name === 'string' && json.data.name ) b.name = json.data.name;
	let elements = Array.isArray( json ) ? json : json.elements !== undefined ? json.elements : json;
	if ( isObject( elements ) ) elements = [ ...( Array.isArray( elements.nodes ) ? elements.nodes : [] ), ...( Array.isArray( elements.edges ) ? elements.edges : [] ) ];
	if ( ! Array.isArray( elements ) ) throw new GraphError( 'Cytoscape JSON: "elements" is neither an array nor { nodes, edges }', 'format' );
	const edges = [];
	const skip = new Set( [ ...SKIP_NODE_KEYS, 'source', 'target' ] );
	let bad = 0;
	elements.forEach( ( el, k ) => {

		const d = isObject( el ) && isObject( el.data ) ? el.data : null;
		if ( ! d ) {

			if ( bad ++ === 0 ) b.warn( `element ${k} has no "data" object` );
			return;

		}

		if ( d.source !== undefined && d.target !== undefined ) {

			edges.push( d );
			return;

		}

		if ( d.id === undefined || d.id === null ) {

			if ( bad ++ === 0 ) b.warn( `element ${k}: a node without data.id` );
			return;

		}

		const i = b.node( d.id );
		nodeAttributes( b, i, d, skip );
		if ( d.parent !== undefined && d.parent !== null && d.parent !== '' ) b.parent( i, b.node( d.parent ) );

	} );
	if ( bad > 1 ) b.warn( `${bad} elements were skipped` );
	let missing = 0;
	for ( const d of edges ) {

		if ( ! b.has( d.source ) || ! b.has( d.target ) ) {

			missing ++;
			b.invalidEdge();
			continue;

		}

		b.edge( b.node( d.source ), b.node( d.target ), weightOf( d ) );

	}

	if ( missing ) b.warn( `${missing} edge${missing === 1 ? ' refers' : 's refer'} to nodes that do not exist` );
	return b.finish( { format: 'cytoscape' } );

}

function graphology( json, context ) {

	const b = new GraphBuilder( context );
	if ( isObject( json.attributes ) && typeof json.attributes.name === 'string' && json.attributes.name ) b.name = json.attributes.name;
	const type = isObject( json.options ) ? json.options.type : undefined;
	json.nodes.forEach( ( node, k ) => {

		if ( ! isObject( node ) || node.key === undefined || node.key === null ) {

			b.warn( `node ${k} has no key` );
			return;

		}

		const i = b.node( node.key );
		if ( isObject( node.attributes ) ) nodeAttributes( b, i, node.attributes );

	} );
	const edges = Array.isArray( json.edges ) ? json.edges : [];
	let missing = 0, directed = false;
	for ( const e of edges ) {

		if ( ! isObject( e ) || ! b.has( e.source ) || ! b.has( e.target ) ) {

			missing ++;
			b.invalidEdge();
			continue;

		}

		if ( e.undirected !== true && type !== 'undirected' ) directed = true;
		b.edge( b.node( e.source ), b.node( e.target ), weightOf( e.attributes ) );

	}

	b.directed = directed;
	if ( missing ) b.warn( `${missing} edge${missing === 1 ? ' refers' : 's refer'} to nodes that do not exist` );
	return b.finish( { format: 'graphology' } );

}

function adjacency( json, context ) {

	const b = new GraphBuilder( context );
	if ( Array.isArray( json ) ) {

		// neighbour index lists
		for ( let i = 0; i < json.length; i ++ ) b.node( i );
		json.forEach( ( row, i ) => row.forEach( ( j ) => b.edge( i, j ) ) );
		return b.finish( { format: 'adjacency' } );

	}

	if ( Array.isArray( json.nodes ) && Array.isArray( json.adjacency ) ) {

		// networkx adjacency_data
		if ( isObject( json.graph ) && typeof json.graph.name === 'string' && json.graph.name ) b.name = json.graph.name;
		b.directed = json.directed === true;
		json.nodes.forEach( ( node, i ) => {

			const idx = b.node( isObject( node ) ? ( node.id !== undefined ? node.id : i ) : node );
			if ( isObject( node ) ) nodeAttributes( b, idx, node );

		} );
		const declared = b.nodeCount;
		json.adjacency.forEach( ( row, i ) => {

			if ( ! Array.isArray( row ) || i >= declared ) return;
			for ( const nb of row ) {

				const id = isObject( nb ) ? nb.id : nb;
				if ( id === undefined || id === null || ! b.has( id ) ) b.invalidEdge();
				else b.edge( i, b.node( id ), weightOf( nb ) );

			}

		} );
		return b.finish( { format: 'adjacency' } );

	}

	// { node: [ neighbours ] } or { node: { neighbour: weight | { weight } } }
	for ( const key of Object.keys( json ) ) b.node( key );
	for ( const key of Object.keys( json ) ) {

		const i = b.node( key ), row = json[ key ];
		if ( Array.isArray( row ) ) {

			for ( const nb of row ) {

				if ( isScalar( nb ) ) b.edge( i, b.node( nb ) );
				else if ( isObject( nb ) && nb.id !== undefined ) b.edge( i, b.node( nb.id ), weightOf( nb ) );
				else b.invalidEdge();

			}

		} else if ( isObject( row ) ) {

			for ( const nb of Object.keys( row ) ) {

				const w = row[ nb ];
				b.edge( i, b.node( nb ), typeof w === 'number' && Number.isFinite( w ) ? w : weightOf( w ) );

			}

		}

	}

	return b.finish( { format: 'adjacency' } );

}

function edgeArray( json, context ) {

	const b = new GraphBuilder( context );
	let bad = 0;
	json.forEach( ( e, k ) => {

		let a, c, w = 1;
		if ( Array.isArray( e ) ) {

			[ a, c ] = e;
			if ( typeof e[ 2 ] === 'number' ) w = e[ 2 ];

		} else if ( isObject( e ) ) {

			a = e.source;
			c = e.target;
			w = weightOf( e );

		}

		if ( ! isScalar( a ) || ! isScalar( c ) ) {

			if ( bad ++ === 0 ) b.warn( `edge ${k} has no usable source and target` );
			b.invalidEdge();
			return;

		}

		b.edge( b.node( a ), b.node( c ), w );

	} );
	if ( bad > 1 ) b.warn( `${bad} edges have no usable source and target` );
	return b.finish( { format: 'edge-array', numericOrder: true } );

}

export function parseJsonShape( json, shape, context = {} ) {

	switch ( shape ) {

		case 'node-link': return nodeLink( json, context );
		case 'cytoscape': return cytoscape( json, context );
		case 'graphology': return graphology( json, context );
		case 'adjacency': return adjacency( json, context );
		case 'edge-array': return edgeArray( json, context );
		default: throw new GraphError( `Unrecognized graph JSON (expected ${JSON_SHAPES})`, 'format' );

	}

}
