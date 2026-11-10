// XML graph formats: GraphML and GEXF, on a small XML scanner of our own (DOMParser does not
// exist in workers, and a DOM of a 50 MB file would not fit anyway). Pure JS.
//
// ─── scanXml( text, handler ) ───────────────────────────────────────────────────────────────
//  One pass over the text, no tree. handler.open( name, attrs, qname, pos ), handler.close(
//  name, pos ), handler.text( string ) (character data and CDATA, entities decoded; only called
//  while handler.wantsText is true) and handler.progress( fraction ). `name` is the local name
//  (prefix removed), `qname` the name as written (y:NodeLabel). Comments, processing
//  instructions and DOCTYPE are skipped. Not validating: it checks what it needs to stay in
//  step, i.e. that tags are terminated, attribute values are quoted and end tags match, and
//  reports those errors with the line number.
//
// ─── GraphML (graphml.graphdrawing.org) ─────────────────────────────────────────────────────
//  <key id for attr.name attr.type> with <default>; <graph edgedefault>; <node id> with <data
//  key>; <edge source target directed> with <data key>. Node data become attributes, the edge
//  key named "weight" the edge weight (its <default> is the weight of an edge without that
//  <data>). yEd files: the text of <y:NodeLabel> is the label.
//  Nested graphs (<node><graph>…</graph></node>) are groups. Hyperedges and ports are ignored
//  (reported). An edge to an undeclared node declares it.
//
// ─── GEXF 1.1 to 1.3 (gexf.net) ─────────────────────────────────────────────────────────────
//  <graph defaultedgetype>; <attributes class="node|edge"><attribute id title type> with
//  <default>; <node id label pid> with <attvalues><attvalue for value>, <viz:size value>,
//  nested <nodes> or pid as groups; <edge source target weight type>. Dynamic graphs load as
//  the union of all times (reported).
//
// API
//   scanXml( text, handler )
//   parseGraphML( text, context ) -> canonical graph      context: { name, limits, onProgress }
//   parseGexf( text, context ) -> canonical graph

import { GraphBuilder, GraphError } from './builder.js';

const ENTITIES = { lt: '<', gt: '>', amp: '&', quot: '"', apos: "'" };

export function decodeEntities( s ) {

	if ( s.indexOf( '&' ) < 0 ) return s;
	return s.replace( /&(#x[0-9a-fA-F]+|#\d+|[a-zA-Z]+);/g, ( m, code ) => {

		if ( code[ 0 ] === '#' ) {

			const n = code[ 1 ] === 'x' || code[ 1 ] === 'X' ? parseInt( code.slice( 2 ), 16 ) : parseInt( code.slice( 1 ), 10 );
			return Number.isFinite( n ) && n > 0 && n <= 0x10ffff ? String.fromCodePoint( n ) : m;

		}

		return ENTITIES[ code ] !== undefined ? ENTITIES[ code ] : m;

	} );

}

export function lineAt( text, pos ) {

	let line = 1;
	for ( let i = text.indexOf( '\n' ); i >= 0 && i < pos; i = text.indexOf( '\n', i + 1 ) ) line ++;
	return line;

}

function isSpace( c ) {

	return c === 32 || c === 10 || c === 9 || c === 13;

}

export function scanXml( text, h ) {

	const n = text.length;
	const fail = ( message, pos ) => {

		throw new GraphError( `XML: ${message} (line ${lineAt( text, pos )})`, 'parse' );

	};

	const stack = [], opened = [];
	let i = 0, steps = 0;
	while ( i < n ) {

		const lt = text.indexOf( '<', i );
		if ( lt < 0 ) break;
		if ( lt > i && h.wantsText ) h.text( decodeEntities( text.slice( i, lt ) ) );
		if ( ( ++ steps & 4095 ) === 0 && h.progress ) h.progress( lt / n );
		const c = text.charCodeAt( lt + 1 );

		if ( c === 33 /* ! */ ) {

			if ( text.startsWith( '<!--', lt ) ) {

				const end = text.indexOf( '-->', lt + 4 );
				if ( end < 0 ) fail( 'a comment is not closed', lt );
				i = end + 3;

			} else if ( text.startsWith( '<![CDATA[', lt ) ) {

				const end = text.indexOf( ']]>', lt + 9 );
				if ( end < 0 ) fail( 'a CDATA section is not closed', lt );
				if ( h.wantsText ) h.text( text.slice( lt + 9, end ) );
				i = end + 3;

			} else {

				// <!DOCTYPE …>, possibly with an internal subset in [ ]
				let depth = 0, p = lt + 2;
				for ( ; p < n; p ++ ) {

					const ch = text.charCodeAt( p );
					if ( ch === 91 ) depth ++;
					else if ( ch === 93 ) depth --;
					else if ( ch === 62 && depth <= 0 ) break;

				}

				if ( p >= n ) fail( 'a <! declaration is not closed', lt );
				i = p + 1;

			}

			continue;

		}

		if ( c === 63 /* ? */ ) {

			const end = text.indexOf( '?>', lt + 2 );
			if ( end < 0 ) fail( 'a processing instruction is not closed', lt );
			i = end + 2;
			continue;

		}

		if ( c === 47 /* / */ ) {

			const end = text.indexOf( '>', lt + 2 );
			if ( end < 0 ) fail( 'an end tag is not closed', lt );
			const qname = text.slice( lt + 2, end ).trim();
			if ( ! stack.length ) fail( `</${qname}> has no start tag`, lt );
			const top = stack.pop(), at = opened.pop();
			if ( top !== qname ) fail( `</${qname}> closes <${top}>, which was opened in line ${lineAt( text, at )}`, lt );
			const colon = qname.indexOf( ':' );
			h.close( colon < 0 ? qname : qname.slice( colon + 1 ), lt );
			i = end + 1;
			continue;

		}

		// start tag
		let p = lt + 1;
		while ( p < n ) {

			const ch = text.charCodeAt( p );
			if ( isSpace( ch ) || ch === 62 || ch === 47 ) break;
			p ++;

		}

		const qname = text.slice( lt + 1, p );
		if ( ! qname.length || p >= n ) fail( p >= n ? `<${qname.slice( 0, 40 )} is not closed` : 'a "<" that starts no tag (write &lt; in text)', lt );
		const attrs = {};
		let selfClosing = false;
		for ( ;; ) {

			while ( p < n && isSpace( text.charCodeAt( p ) ) ) p ++;
			if ( p >= n ) fail( `<${qname}> is not closed`, lt );
			const ch = text.charCodeAt( p );
			if ( ch === 62 ) {

				p ++;
				break;

			}

			if ( ch === 47 ) {

				if ( text.charCodeAt( p + 1 ) !== 62 ) fail( `unexpected "/" in <${qname}>`, p );
				selfClosing = true;
				p += 2;
				break;

			}

			const start = p;
			while ( p < n ) {

				const k = text.charCodeAt( p );
				if ( k === 61 || isSpace( k ) || k === 62 || k === 47 ) break;
				p ++;

			}

			const key = text.slice( start, p );
			while ( p < n && isSpace( text.charCodeAt( p ) ) ) p ++;
			if ( text.charCodeAt( p ) !== 61 ) fail( `attribute "${key}" of <${qname}> has no value`, start );
			p ++;
			while ( p < n && isSpace( text.charCodeAt( p ) ) ) p ++;
			const quote = text.charCodeAt( p );
			if ( quote !== 34 && quote !== 39 ) fail( `the value of attribute "${key}" of <${qname}> is not quoted`, p );
			const end = text.indexOf( quote === 34 ? '"' : "'", p + 1 );
			if ( end < 0 ) fail( `the value of attribute "${key}" of <${qname}> is not closed`, p );
			attrs[ key ] = decodeEntities( text.slice( p + 1, end ) );
			p = end + 1;

		}

		const colon = qname.indexOf( ':' );
		const name = colon < 0 ? qname : qname.slice( colon + 1 );
		h.open( name, attrs, qname, lt );
		if ( selfClosing ) h.close( name, lt );
		else {

			stack.push( qname );
			opened.push( lt );

		}

		i = p;

	}

	if ( stack.length ) fail( `<${stack[ stack.length - 1 ]}> is never closed (the file may be cut off)`, opened[ opened.length - 1 ] );

}

const WEIGHT_RE = /^(weight|w)$/i;

function toWeight( s ) {

	const w = Number( s );
	return Number.isFinite( w ) ? w : 1;

}

// ── GraphML ──────────────────────────────────────────────────────────────────────────────────

export function parseGraphML( text, context = {} ) {

	const b = new GraphBuilder( context );
	const keys = new Map(); // id -> { domain, name }
	const nodeStack = []; // open <node> elements (nested graphs)
	let sawRoot = false, graphs = 0, directedDefault = false;
	let key = null, inDefault = false;
	let edge = null, edgeLine = 0;
	let data = null, buffer = '', dataChildren = 0, dataDepth = 0, label = null, inLabel = false;
	let hyper = 0, anonymous = 0, undeclared = 0;
	let defaultWeight = 1; // <default> of the edge key named weight: what an edge without <data> weighs

	const h = {
		wantsText: false,
		progress: ( f ) => b.progress( f ),
		text( s ) {

			buffer += s;

		},
		open( name, attrs, qname, pos ) {

			if ( ! sawRoot ) {

				if ( name !== 'graphml' ) throw new GraphError( `GraphML: the root element is <${qname}>, expected <graphml>`, 'format' );
				sawRoot = true;
				return;

			}

			if ( data ) {

				// structured content of a <data> element (yEd): only the node label is read
				dataDepth ++;
				dataChildren ++;
				if ( name === 'NodeLabel' && label === null ) {

					inLabel = true;
					buffer = '';
					h.wantsText = true;

				}

				return;

			}

			switch ( name ) {

				case 'key':
					key = { domain: attrs.for || 'all', name: attrs[ 'attr.name' ] || attrs.id || '' };
					if ( attrs.id !== undefined ) keys.set( attrs.id, key );
					break;
				case 'default':
					if ( key ) {

						inDefault = true;
						buffer = '';
						h.wantsText = true;

					}

					break;
				case 'graph':
					graphs ++;
					if ( graphs === 1 ) directedDefault = attrs.edgedefault === 'directed';
					break;
				case 'node': {

					let id = attrs.id;
					if ( id === undefined ) {

						id = `(node ${ ++ anonymous })`;
						b.warn( () => `line ${lineAt( text, pos )}: <node> without an id` );

					}

					const i = b.node( id );
					if ( nodeStack.length ) b.parent( i, nodeStack[ nodeStack.length - 1 ] );
					nodeStack.push( i );
					break;

				}

				case 'edge':
					edge = { s: attrs.source, t: attrs.target, directed: attrs.directed === undefined ? directedDefault : attrs.directed === 'true', w: defaultWeight };
					edgeLine = pos;
					break;
				case 'data':
					data = { key: attrs.key };
					buffer = '';
					dataChildren = 0;
					dataDepth = 0;
					label = null;
					h.wantsText = true;
					break;
				case 'hyperedge':
					hyper ++;
					break;
				default:

			}

		},
		close( name ) {

			if ( data ) {

				if ( dataDepth > 0 ) {

					dataDepth --;
					if ( inLabel && name === 'NodeLabel' ) {

						label = buffer.trim();
						inLabel = false;
						h.wantsText = false;

					}

					return;

				}

				h.wantsText = false;
				const k = keys.get( data.key );
				const attrName = k ? k.name : data.key;
				const value = dataChildren > 0 ? '' : buffer.trim(); // text between child elements is not a value
				if ( edge ) {

					if ( WEIGHT_RE.test( attrName || '' ) && value !== '' ) edge.w = toWeight( value );

				} else if ( nodeStack.length ) {

					const i = nodeStack[ nodeStack.length - 1 ];
					if ( label !== null && label !== '' ) b.label( i, label );
					else if ( value !== '' && attrName ) b.attr( i, attrName, value );

				}

				data = null;
				buffer = '';
				return;

			}

			switch ( name ) {

				case 'default':
					if ( inDefault && key ) {

						if ( key.domain === 'node' || key.domain === 'all' ) b.attrDefault( key.name, buffer.trim() );
						key.default = buffer.trim();
						if ( ( key.domain === 'edge' || key.domain === 'all' ) && WEIGHT_RE.test( key.name ) && key.default !== '' ) defaultWeight = toWeight( key.default );
						inDefault = false;
						h.wantsText = false;
						buffer = '';

					}

					break;
				case 'key':
					key = null;
					break;
				case 'node':
					nodeStack.pop();
					break;
				case 'edge':
					if ( ! edge ) break;
					if ( edge.s === undefined || edge.t === undefined ) {

						b.warn( () => `line ${lineAt( text, edgeLine )}: <edge> without source or target` );
						b.invalidEdge();

					} else {

						if ( ! b.has( edge.s ) || ! b.has( edge.t ) ) undeclared ++;
						if ( edge.directed ) b.directed = true;
						b.edge( b.node( edge.s ), b.node( edge.t ), edge.w );

					}

					edge = null;
					break;
				default:

			}

		},
	};
	scanXml( text, h );
	if ( ! sawRoot ) throw new GraphError( 'GraphML: no <graphml> element found', 'format' );
	if ( graphs === 0 ) throw new GraphError( 'GraphML: the file has no <graph> element', 'format' );
	if ( hyper ) b.warn( `${hyper} hyperedge${hyper === 1 ? '' : 's'} ignored` );
	if ( undeclared ) b.warn( `${undeclared} edge${undeclared === 1 ? ' refers' : 's refer'} to nodes that are declared later or not at all` );
	return b.finish( { format: 'graphml' } );

}

// ── GEXF ─────────────────────────────────────────────────────────────────────────────────────

export function parseGexf( text, context = {} ) {

	const b = new GraphBuilder( context );
	const nodeAttrs = new Map(), edgeAttrs = new Map(); // id -> title
	const nodeStack = [];
	let sawRoot = false, sawGraph = false, directedDefault = false, dynamic = false;
	let attrClass = null, attribute = null, inDefault = false, buffer = '';
	let edge = null, edgeLine = 0, anonymous = 0, undeclared = 0;

	const h = {
		wantsText: false,
		progress: ( f ) => b.progress( f ),
		text( s ) {

			buffer += s;

		},
		open( name, attrs, qname, pos ) {

			if ( ! sawRoot ) {

				if ( name !== 'gexf' ) throw new GraphError( `GEXF: the root element is <${qname}>, expected <gexf>`, 'format' );
				sawRoot = true;
				return;

			}

			switch ( name ) {

				case 'graph':
					sawGraph = true;
					directedDefault = attrs.defaultedgetype === 'directed';
					if ( attrs.mode === 'dynamic' ) dynamic = true;
					break;
				case 'description':
				case 'default':
					if ( name === 'default' && ! attribute ) break;
					inDefault = name === 'default';
					buffer = '';
					h.wantsText = true;
					break;
				case 'attributes':
					attrClass = attrs.class === 'edge' ? 'edge' : 'node';
					if ( attrs.mode === 'dynamic' ) dynamic = true;
					break;
				case 'attribute':
					attribute = { id: attrs.id, title: attrs.title || attrs.id || '' };
					if ( attrs.id !== undefined ) ( attrClass === 'edge' ? edgeAttrs : nodeAttrs ).set( attrs.id, attribute.title );
					break;
				case 'node': {

					let id = attrs.id;
					if ( id === undefined ) {

						id = `(node ${ ++ anonymous })`;
						b.warn( () => `line ${lineAt( text, pos )}: <node> without an id` );

					}

					const i = b.node( id );
					if ( attrs.label !== undefined && attrs.label !== '' ) b.label( i, attrs.label );
					if ( attrs.pid !== undefined && attrs.pid !== '' && attrs.pid !== id ) b.parent( i, b.node( attrs.pid ) );
					else if ( nodeStack.length ) b.parent( i, nodeStack[ nodeStack.length - 1 ] );
					nodeStack.push( i );
					break;

				}

				case 'parent':
					if ( nodeStack.length && attrs.for !== undefined && ! ( b.parents && b.parents[ nodeStack[ nodeStack.length - 1 ] ] !== undefined ) ) b.parent( nodeStack[ nodeStack.length - 1 ], b.node( attrs.for ) );
					break;
				case 'attvalue': {

					const ref = attrs.for !== undefined ? attrs.for : attrs.id;
					if ( edge ) {

						const title = edgeAttrs.get( ref ) || ref || '';
						if ( WEIGHT_RE.test( title ) && attrs.value !== undefined && edge.w === null ) edge.w = toWeight( attrs.value );

					} else if ( nodeStack.length && attrs.value !== undefined ) {

						b.attr( nodeStack[ nodeStack.length - 1 ], nodeAttrs.get( ref ) || String( ref ), attrs.value );

					}

					break;

				}

				case 'size':
					if ( qname !== name && nodeStack.length && ! edge && attrs.value !== undefined ) b.attr( nodeStack[ nodeStack.length - 1 ], 'size', attrs.value );
					break;
				case 'edge':
					edge = { s: attrs.source, t: attrs.target, w: attrs.weight !== undefined ? toWeight( attrs.weight ) : null, directed: attrs.type === undefined ? directedDefault : attrs.type === 'directed' };
					edgeLine = pos;
					break;
				default:

			}

		},
		close( name ) {

			switch ( name ) {

				case 'description':
					if ( h.wantsText && ! inDefault ) b.description = buffer.trim().slice( 0, 400 );
					h.wantsText = false;
					buffer = '';
					break;
				case 'default':
					if ( inDefault && attribute ) {

						if ( attrClass !== 'edge' ) b.attrDefault( attribute.title, buffer.trim() );
						inDefault = false;
						h.wantsText = false;
						buffer = '';

					}

					break;
				case 'attribute':
					attribute = null;
					break;
				case 'node':
					nodeStack.pop();
					break;
				case 'edge':
					if ( ! edge ) break;
					if ( edge.s === undefined || edge.t === undefined ) {

						b.warn( () => `line ${lineAt( text, edgeLine )}: <edge> without source or target` );
						b.invalidEdge();

					} else {

						if ( ! b.has( edge.s ) || ! b.has( edge.t ) ) undeclared ++;
						if ( edge.directed ) b.directed = true;
						b.edge( b.node( edge.s ), b.node( edge.t ), edge.w === null ? 1 : edge.w );

					}

					edge = null;
					break;
				default:

			}

		},
	};
	scanXml( text, h );
	if ( ! sawRoot ) throw new GraphError( 'GEXF: no <gexf> element found', 'format' );
	if ( ! sawGraph ) throw new GraphError( 'GEXF: the file has no <graph> element', 'format' );
	if ( dynamic ) b.warn( 'dynamic graph: the time information is ignored, all nodes and edges are shown' );
	if ( undeclared ) b.warn( `${undeclared} edge${undeclared === 1 ? ' refers' : 's refer'} to nodes that are declared later or not at all` );
	return b.finish( { format: 'gexf' } );

}
