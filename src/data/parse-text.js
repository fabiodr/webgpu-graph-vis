// Text graph formats: GML, Graphviz DOT, Pajek and Matrix Market. Pure JS, one pass over the
// text each, no intermediate tree; every parser writes into a GraphBuilder (builder.js), which
// makes the canonical graph. Errors are GraphError with the line number.
//
// ─── GML ────────────────────────────────────────────────────────────────────────────────────
//  graph [ directed 0|1  node [ id … label "…" … ]  edge [ source … target … value|weight … ] ]
//  Node keys other than id and label become attributes (nested lists such as graphics [ ] are
//  skipped). Edge weight: `weight`, else a numeric `value` (the convention of the Newman
//  collection). gid (yEd) puts a node into the group node with that id. Comments: # to the end
//  of the line. Strings may span lines and use XML entities.
//
// ─── DOT (the common subset) ────────────────────────────────────────────────────────────────
//  [strict] graph|digraph [name] { … } with node statements, edge statements (chains a -- b --
//  c, groups { a b } -- c, ports are ignored), attribute lists, default attribute statements
//  (node [ … ] / edge [ weight=… ], scoped), name = value, comments (//, /* */, #), quoted
//  strings with escapes and + concatenation, HTML strings (tags removed). Named subgraphs are
//  groups (nested ones give nested groups); a node belongs to the first named subgraph it
//  appears in. Node attributes are kept except the purely visual ones (shape, style, fonts,
//  sizes, positions); the edge attribute `weight` is the edge weight.
//
// ─── Pajek .net / .paj ──────────────────────────────────────────────────────────────────────
//  *Vertices n (number, optional "label", the rest ignored), *Arcs, *Edges (u v [weight]),
//  *Arcslist, *Edgeslist (u v1 v2 …), *Matrix (n rows of n numbers), *Network, and in project
//  files *Partition / *Vector followed by *Vertices n and one value per line (an attribute).
//  % starts a comment line. Section names are not case sensitive.
//
// ─── Matrix Market .mtx ─────────────────────────────────────────────────────────────────────
//  %%MatrixMarket matrix coordinate|array real|integer|complex|pattern general|symmetric|
//  skew-symmetric|hermitian. A square matrix is an adjacency matrix (entry = edge weight, by
//  magnitude; explicit zeros are no edges; general = directed, A + Aᵀ). A rectangular one is a
//  bipartite graph of rows and columns (attribute "side").
//
// API ( context: { name, limits, onProgress } )
//   parseGml( text, context ), parseDot( text, context ), parsePajek( text, context ),
//   parseMtx( text, context ) -> canonical graph

import { GraphBuilder, GraphError, tooLarge } from './builder.js';
import { decodeEntities } from './parse-xml.js';

// ── GML ──────────────────────────────────────────────────────────────────────────────────────

export function parseGml( text, context = {} ) {

	const b = new GraphBuilder( context );
	const n = text.length;
	let i = 0, line = 1, steps = 0;
	const T = { type: 'eof', value: null, line: 1 };

	const fail = ( message, at = T.line ) => {

		throw new GraphError( `GML: line ${at}: ${message}`, 'parse' );

	};

	function next() {

		for ( ;; ) {

			if ( i >= n ) {

				T.type = 'eof';
				T.value = null;
				T.line = line;
				return T;

			}

			const c = text.charCodeAt( i );
			if ( c === 10 ) {

				line ++;
				i ++;

			} else if ( c === 32 || c === 9 || c === 13 || c === 0xfeff ) i ++;
			else if ( c === 35 /* # */ ) {

				const end = text.indexOf( '\n', i );
				i = end < 0 ? n : end;

			} else break;

		}

		if ( ( ++ steps & 8191 ) === 0 ) b.progress( i / n );
		T.line = line;
		const c = text.charCodeAt( i );
		if ( c === 91 || c === 93 ) {

			T.type = c === 91 ? '[' : ']';
			T.value = null;
			i ++;
			return T;

		}

		if ( c === 34 /* " */ ) {

			const end = text.indexOf( '"', i + 1 );
			if ( end < 0 ) fail( 'a string is not closed', line );
			const raw = text.slice( i + 1, end );
			for ( let k = raw.indexOf( '\n' ); k >= 0; k = raw.indexOf( '\n', k + 1 ) ) line ++;
			T.type = 'str';
			T.value = decodeEntities( raw );
			i = end + 1;
			return T;

		}

		const start = i;
		while ( i < n ) {

			const k = text.charCodeAt( i );
			if ( k === 32 || k === 9 || k === 10 || k === 13 || k === 91 || k === 93 || k === 34 ) break;
			i ++;

		}

		const word = text.slice( start, i );
		const num = Number( word );
		if ( word.length && Number.isFinite( num ) && /^[-+.\d]/.test( word ) ) {

			T.type = 'num';
			T.value = num;

		} else {

			T.type = 'key';
			T.value = word;

		}

		return T;

	}

	/** Skips a list whose '[' was just read. */
	function skipList( openLine ) {

		let depth = 1;
		while ( depth > 0 ) {

			next();
			if ( T.type === 'eof' ) fail( 'a "[" is never closed (the file may be cut off)', openLine );
			if ( T.type === '[' ) depth ++;
			else if ( T.type === ']' ) depth --;

		}

	}

	/** Reads the scalar entries of a list whose '[' was just read (nested lists are skipped). */
	function readRecord( openLine, what ) {

		const rec = {};
		for ( ;; ) {

			next();
			if ( T.type === ']' ) return rec;
			if ( T.type === 'eof' ) fail( `the ${what} that starts here is never closed (the file may be cut off)`, openLine );
			if ( T.type !== 'key' ) fail( `expected a key inside ${what}, found ${describe( T )}` );
			const key = T.value;
			next();
			if ( T.type === '[' ) skipList( T.line );
			else if ( T.type === 'eof' || T.type === ']' ) fail( `the key "${key}" has no value` );
			else if ( ! ( key in rec ) ) rec[ key ] = T.value;

		}

	}

	const describe = ( t ) => ( t.type === 'eof' ? 'the end of the file' : t.type === 'str' ? 'a string' : t.type === 'num' ? `the number ${t.value}` : `"${t.type === 'key' ? t.value : t.type}"` );

	let graphs = 0, anonymous = 0;
	for ( ;; ) {

		next();
		if ( T.type === 'eof' ) break;
		if ( T.type !== 'key' ) fail( `expected a key, found ${describe( T )}` );
		const key = T.value.toLowerCase();
		next();
		if ( T.type !== '[' ) continue; // Creator "…", Version 1
		const openLine = T.line;
		if ( key !== 'graph' || graphs > 0 ) {

			if ( key === 'graph' ) b.warn( `line ${openLine}: a second graph block is ignored` );
			skipList( openLine );
			continue;

		}

		graphs ++;
		for ( ;; ) {

			next();
			if ( T.type === ']' ) break;
			if ( T.type === 'eof' ) fail( 'the graph block is never closed (the file may be cut off)', openLine );
			if ( T.type !== 'key' ) fail( `expected a key inside graph [ ], found ${describe( T )}` );
			const k = T.value.toLowerCase();
			const at = T.line;
			next();
			if ( T.type === 'eof' || T.type === ']' ) fail( `the key "${k}" has no value`, at );
			if ( T.type !== '[' ) {

				if ( k === 'directed' ) b.directed = Number( T.value ) === 1;
				else if ( ( k === 'label' || k === 'name' ) && typeof T.value === 'string' && T.value && ! context.name ) b.name = T.value;
				continue;

			}

			if ( k === 'node' ) {

				const rec = readRecord( at, 'node' );
				let id = rec.id;
				if ( id === undefined ) {

					b.warn( `line ${at}: node without an id` );
					id = `(node ${ ++ anonymous })`;

				}

				const idx = b.node( id );
				for ( const name in rec ) {

					if ( name === 'id' ) continue;
					if ( name === 'label' ) b.label( idx, rec.label );
					else if ( name === 'gid' ) b.parent( idx, b.node( rec.gid ) );
					else if ( name !== 'isGroup' ) b.attr( idx, name, rec[ name ] );

				}

			} else if ( k === 'edge' ) {

				const rec = readRecord( at, 'edge' );
				if ( rec.source === undefined || rec.target === undefined ) {

					b.warn( `line ${at}: edge without source or target` );
					b.invalidEdge();
					continue;

				}

				let w = 1;
				if ( typeof rec.weight === 'number' ) w = rec.weight;
				else if ( typeof rec.value === 'number' ) w = rec.value;
				b.edge( b.node( rec.source ), b.node( rec.target ), w );

			} else skipList( at );

		}

	}

	if ( graphs === 0 ) throw new GraphError( 'GML: no "graph [ … ]" block found', 'format' );
	return b.finish( { format: 'gml' } );

}

// ── DOT ──────────────────────────────────────────────────────────────────────────────────────

const DOT_VISUAL = new Set( [ 'shape', 'style', 'fontname', 'fontsize', 'fontcolor', 'width', 'height', 'pos', 'penwidth', 'fixedsize', 'margin', 'peripheries', 'url', 'href', 'tooltip', 'image', 'labelloc', 'xlabel', 'fillcolor', 'color', 'bgcolor', 'rank', 'rankdir', 'ordering', 'regular', 'sides', 'distortion', 'orientation', 'skew', 'pin', 'rects', 'vertices', 'z', 'xlp', 'lp', 'target', 'layer', 'gradientangle', 'colorscheme', 'labeljust', 'nojustify', 'imagescale', 'imagepos', 'samplepoints', 'shapefile', 'showboxes', 'sortv', 'root', 'id', 'class', 'comment' ] );

const DOT_ID = /(?:-?(?:\.\d+|\d+(?:\.\d*)?)|[A-Za-z_\u0080-￿][A-Za-z0-9_\u0080-￿]*)/y;

const DOT_MAX_NESTING = 200;

export function parseDot( text, context = {} ) {

	const b = new GraphBuilder( context );
	const n = text.length;
	let i = 0, line = 1, steps = 0, nesting = 0;
	// current token: t = 'id' | 'op' | 'eof'; v = text; q = quoted or HTML
	let t = 'eof', v = '', q = false, tl = 1;

	const fail = ( message, at = tl ) => {

		throw new GraphError( `DOT: line ${at}: ${message}`, 'parse' );

	};

	const found = () => ( t === 'eof' ? 'the end of the file' : `"${v.length > 24 ? v.slice( 0, 24 ) + '…' : v}"` );

	function advance() {

		for ( ;; ) {

			if ( i >= n ) {

				t = 'eof';
				v = '';
				q = false;
				tl = line;
				return;

			}

			const c = text.charCodeAt( i );
			if ( c === 10 ) {

				line ++;
				i ++;

			} else if ( c === 32 || c === 9 || c === 13 || c === 0xfeff ) i ++;
			else if ( c === 35 /* # */ || ( c === 47 && text.charCodeAt( i + 1 ) === 47 ) ) {

				const end = text.indexOf( '\n', i );
				i = end < 0 ? n : end;

			} else if ( c === 47 && text.charCodeAt( i + 1 ) === 42 ) {

				const end = text.indexOf( '*/', i + 2 );
				if ( end < 0 ) fail( 'a /* comment is not closed', line );
				for ( let k = text.indexOf( '\n', i ); k >= 0 && k < end; k = text.indexOf( '\n', k + 1 ) ) line ++;
				i = end + 2;

			} else break;

		}

		if ( ( ++ steps & 8191 ) === 0 ) b.progress( i / n );
		tl = line;
		q = false;
		const c = text.charCodeAt( i );
		if ( c === 34 /* " */ ) {

			let out = '';
			let p = i + 1;
			for ( ;; ) {

				if ( p >= n ) fail( 'a string is not closed', tl );
				const ch = text[ p ];
				if ( ch === '"' ) break;
				if ( ch === '\\' && p + 1 < n ) {

					const e = text[ p + 1 ];
					if ( e === '"' || e === '\\' ) out += e;
					else if ( e === '\n' ) line ++;
					else if ( e === '\r' && text[ p + 2 ] === '\n' ) {

						line ++;
						p ++;

					} else out += ch + e;
					p += 2;
					continue;

				}

				if ( ch === '\n' ) line ++;
				out += ch;
				p ++;

			}

			i = p + 1;
			t = 'id';
			v = out;
			q = true;
			return;

		}

		if ( c === 60 /* < */ ) {

			let depth = 0, p = i;
			for ( ; p < n; p ++ ) {

				const ch = text.charCodeAt( p );
				if ( ch === 60 ) depth ++;
				else if ( ch === 62 && -- depth === 0 ) break;
				else if ( ch === 10 ) line ++;

			}

			if ( p >= n ) fail( 'an HTML string <…> is not closed', tl );
			t = 'id';
			v = decodeEntities( text.slice( i + 1, p ).replace( /<[^>]*>/g, ' ' ) ).replace( /\s+/g, ' ' ).trim();
			q = true;
			i = p + 1;
			return;

		}

		if ( c === 45 /* - */ && ( text.charCodeAt( i + 1 ) === 45 || text.charCodeAt( i + 1 ) === 62 ) ) {

			t = 'op';
			v = text.slice( i, i + 2 );
			i += 2;
			return;

		}

		if ( c === 123 || c === 125 || c === 91 || c === 93 || c === 61 || c === 59 || c === 44 || c === 58 || c === 43 ) {

			t = 'op';
			v = text[ i ];
			i ++;
			return;

		}

		DOT_ID.lastIndex = i;
		const m = DOT_ID.exec( text );
		if ( ! m ) fail( `unexpected character "${text[ i ]}"`, line );
		t = 'id';
		v = m[ 0 ];
		i += v.length;

	}

	const isOp = ( s ) => t === 'op' && v === s;
	const isEdgeOp = () => t === 'op' && ( v === '--' || v === '->' );
	const keyword = ( s ) => t === 'id' && ! q && v.toLowerCase() === s;

	function readId( what ) {

		if ( t !== 'id' ) fail( `expected ${what}, found ${found()}` );
		let s = v;
		const quoted = q;
		advance();
		while ( quoted && isOp( '+' ) ) {

			advance();
			if ( t !== 'id' ) fail( `expected a string after "+", found ${found()}` );
			s += v;
			advance();

		}

		return s;

	}

	function attrLists() {

		let out = null;
		while ( isOp( '[' ) ) {

			const open = tl;
			advance();
			out = out || {};
			for ( ;; ) {

				if ( isOp( ']' ) ) break;
				if ( t === 'eof' ) fail( 'a "[" is never closed', open );
				if ( isOp( ';' ) || isOp( ',' ) ) {

					advance();
					continue;

				}

				const key = readId( 'an attribute name or "]"' );
				let value = 'true';
				if ( isOp( '=' ) ) {

					advance();
					value = readId( `a value for "${key}"` );

				}

				out[ key ] = value;

			}

			advance();

		}

		return out;

	}

	const cleanLabel = ( s ) => s.replace( /\\[nlr]/g, ' ' ).replace( /\s+/g, ' ' ).trim();

	function applyNodeAttrs( idx, attrs ) {

		for ( const key in attrs ) {

			const lower = key.toLowerCase();
			if ( lower === 'label' ) {

				const s = cleanLabel( attrs[ key ] );
				if ( s && s !== '\\N' ) b.label( idx, s );

			} else if ( ! DOT_VISUAL.has( lower ) ) b.attr( idx, key, attrs[ key ] );

		}

	}

	const named = new Map(); // subgraph name -> node list
	let digraph = false;

	function nodeRef( name, scope ) {

		const fresh = ! b.has( name );
		const idx = b.node( name );
		if ( fresh && scope.nodeDefaults ) applyNodeAttrs( idx, scope.nodeDefaults );
		if ( scope.group >= 0 && ! ( b.parents && b.parents[ idx ] !== undefined ) ) b.parent( idx, scope.group );
		scope.nodes.push( idx );
		return idx;

	}

	function skipPort() {

		while ( isOp( ':' ) ) {

			advance();
			readId( 'a port name' );

		}

	}

	function subgraph( scope ) {

		let name = null;
		if ( keyword( 'subgraph' ) ) {

			advance();
			if ( t === 'id' ) name = readId( 'a subgraph name' );

		}

		if ( ! isOp( '{' ) ) {

			if ( name !== null && named.has( name ) ) return named.get( name ); // a reference to an earlier subgraph
			fail( `expected "{", found ${found()}` );

		}

		const open = tl;
		// the parser recurses once per nesting level: a file of 100,000 "{" must not end the
		// worker with a stack overflow
		if ( ++ nesting > DOT_MAX_NESTING ) fail( `subgraphs are nested more than ${DOT_MAX_NESTING} deep` );
		advance();
		const inner = {
			group: name !== null ? b.group( name, null, scope.group ) : scope.group,
			own: name !== null,
			nodes: [],
			nodeDefaults: scope.nodeDefaults,
			edgeWeight: scope.edgeWeight,
		};
		statements( inner, open );
		nesting --;
		advance(); // }
		if ( name !== null ) named.set( name, inner.nodes );
		for ( const idx of inner.nodes ) scope.nodes.push( idx );
		return inner.nodes;

	}

	function operand( scope ) {

		if ( isOp( '{' ) || keyword( 'subgraph' ) ) return subgraph( scope );
		const name = readId( 'a node name' );
		const idx = nodeRef( name, scope );
		skipPort();
		return [ idx ];

	}

	function statements( scope, openLine ) {

		for ( ;; ) {

			if ( isOp( '}' ) ) return;
			if ( t === 'eof' ) fail( 'a "{" is never closed (the file may be cut off)', openLine );
			if ( isOp( ';' ) ) {

				advance();
				continue;

			}

			statement( scope );

		}

	}

	function statement( scope ) {

		if ( t === 'id' && ! q ) {

			const lower = v.toLowerCase();
			if ( lower === 'graph' || lower === 'node' || lower === 'edge' ) {

				advance();
				const attrs = attrLists() || {};
				if ( lower === 'node' ) scope.nodeDefaults = { ...( scope.nodeDefaults || {} ), ...attrs };
				else if ( lower === 'edge' ) {

					if ( attrs.weight !== undefined && Number.isFinite( Number( attrs.weight ) ) ) scope.edgeWeight = Number( attrs.weight );

				} else if ( attrs.label !== undefined && scope.own ) b.labels[ scope.group ] = cleanLabel( attrs.label );
				return;

			}

		}

		let first;
		let single = - 1;
		if ( isOp( '{' ) || keyword( 'subgraph' ) ) first = subgraph( scope );
		else if ( t === 'id' ) {

			const name = readId( 'a statement' );
			if ( isOp( '=' ) ) {

				advance();
				const value = readId( `a value for "${name}"` );
				if ( name.toLowerCase() === 'label' && scope.own ) b.labels[ scope.group ] = cleanLabel( value );
				return;

			}

			single = nodeRef( name, scope );
			skipPort();
			first = [ single ];

		} else fail( `expected a statement, found ${found()}` );

		if ( isEdgeOp() ) {

			const chain = [ first ];
			while ( isEdgeOp() ) {

				if ( v === '->' ) digraph = true;
				advance();
				chain.push( operand( scope ) );

			}

			const attrs = attrLists();
			let w = scope.edgeWeight ?? 1;
			if ( attrs && attrs.weight !== undefined && Number.isFinite( Number( attrs.weight ) ) ) w = Number( attrs.weight );
			for ( let k = 0; k + 1 < chain.length; k ++ ) for ( const a of chain[ k ] ) for ( const c of chain[ k + 1 ] ) b.edge( a, c, w );

		} else if ( single >= 0 ) {

			const attrs = attrLists();
			if ( attrs ) applyNodeAttrs( single, attrs );

		}

	}

	advance();
	if ( keyword( 'strict' ) ) advance();
	if ( keyword( 'digraph' ) ) digraph = true;
	else if ( ! keyword( 'graph' ) ) throw new GraphError( `DOT: line ${tl}: expected "graph" or "digraph", found ${found()}`, 'format' );
	advance();
	if ( t === 'id' ) {

		const name = readId( 'a graph name' );
		if ( name && ! context.name ) b.name = name;

	}

	if ( ! isOp( '{' ) ) fail( `expected "{", found ${found()}` );
	const open = tl;
	advance();
	statements( { group: - 1, own: false, nodes: [], nodeDefaults: null, edgeWeight: undefined }, open );
	b.directed = digraph;
	return b.finish( { format: 'dot' } );

}

// ── line reader shared by Pajek and Matrix Market ────────────────────────────────────────────

function eachLine( text, visit, progress ) {

	const n = text.length;
	let pos = text.charCodeAt( 0 ) === 0xfeff ? 1 : 0, line = 0;
	while ( pos < n ) {

		let end = text.indexOf( '\n', pos );
		if ( end < 0 ) end = n;
		line ++;
		let s = text.slice( pos, end );
		pos = end + 1;
		if ( s.length && s.charCodeAt( s.length - 1 ) === 13 ) s = s.slice( 0, - 1 );
		s = s.trim();
		if ( ( line & 8191 ) === 0 && progress ) progress( pos / n );
		if ( s.length && visit( s, line ) === false ) return;

	}

}

// ── Pajek ────────────────────────────────────────────────────────────────────────────────────

export function parsePajek( text, context = {} ) {

	const b = new GraphBuilder( context );
	const fail = ( line, message ) => {

		throw new GraphError( `Pajek: line ${line}: ${message}`, 'parse' );

	};

	let section = null; // vertices | arcs | edges | arcslist | edgeslist | matrix | values | skip
	let sawNetwork = false, declared = 0, row = 0;
	let pending = null; // { kind: 'partition' | 'vector', name }
	let valueIndex = 0, valueName = '';
	const names = { partition: 0, vector: 0 };

	const vertex = ( token, line ) => {

		if ( ! /^\d+$/.test( token ) ) fail( line, `expected a vertex number, found "${token}"` );
		if ( declared && ( + token < 1 || + token > declared ) ) {

			b.warn( `line ${line}: vertex ${token} is outside 1..${declared}` );
			return - 1;

		}

		return b.node( String( + token ) );

	};

	eachLine( text, ( s, line ) => {

		if ( s.charCodeAt( 0 ) === 37 /* % */ ) return;
		if ( s.charCodeAt( 0 ) === 42 /* * */ ) {

			const m = /^\*\s*([A-Za-z]+)\s*(.*)$/.exec( s );
			const word = m ? m[ 1 ].toLowerCase() : '';
			const rest = m ? m[ 2 ].trim() : '';
			if ( word === 'vertices' ) {

				const count = parseInt( rest, 10 );
				if ( ! ( count >= 0 ) ) fail( line, `"*Vertices" needs a count, found "${rest}"` );
				if ( pending ) {

					section = 'values';
					valueIndex = 0;
					valueName = pending.name;
					if ( pending.kind === 'partition' ) b.groupColumn( valueName );
					pending = null;

				} else {

					if ( count > b.limits.maxNodes ) throw tooLarge( 'nodes', count, b.limits );
					section = 'vertices';
					sawNetwork = true;
					declared = count;
					for ( let k = 1; k <= count; k ++ ) b.node( String( k ) );

				}

			} else if ( word === 'arcs' || word === 'edges' || word === 'arcslist' || word === 'edgeslist' || word === 'matrix' ) {

				section = word;
				row = 0;
				if ( word === 'arcs' || word === 'arcslist' || word === 'matrix' ) b.directed = true;

			} else if ( word === 'network' ) {

				if ( rest && ! context.name ) b.name = rest.replace( /^"|"$/g, '' );
				section = null;

			} else if ( word === 'partition' || word === 'vector' ) {

				names[ word ] ++;
				pending = { kind: word, name: rest.replace( /^"|"$/g, '' ).replace( /\.(clu|vec)$/i, '' ) || ( word === 'partition' ? ( names.partition > 1 ? `partition ${names.partition}` : 'partition' ) : ( names.vector > 1 ? `vector ${names.vector}` : 'vector' ) ) };
				section = 'skip';

			} else {

				b.warn( `line ${line}: section "${s.slice( 0, 30 )}" is not supported and was skipped` );
				section = 'skip';

			}

			return;

		}

		if ( section === 'vertices' ) {

			const m = /^(\S+)\s*(?:"((?:[^"]|"")*)"|(\S+))?/.exec( s );
			const idx = vertex( m[ 1 ], line );
			const label = m[ 2 ] !== undefined ? m[ 2 ] : m[ 3 ];
			if ( idx >= 0 && label !== undefined && label !== '' ) b.label( idx, label );

		} else if ( section === 'arcs' || section === 'edges' ) {

			const f = s.split( /\s+/ );
			if ( f.length < 2 ) fail( line, `expected "from to [weight]", found "${s.slice( 0, 30 )}"` );
			const a = vertex( f[ 0 ], line ), c = vertex( f[ 1 ], line );
			if ( a < 0 || c < 0 ) {

				b.invalidEdge();
				return;

			}

			const w = f.length > 2 ? Number( f[ 2 ] ) : 1;
			b.edge( a, c, Number.isFinite( w ) ? w : 1 );

		} else if ( section === 'arcslist' || section === 'edgeslist' ) {

			const f = s.split( /\s+/ );
			const a = vertex( f[ 0 ], line );
			for ( let k = 1; k < f.length; k ++ ) {

				const c = vertex( f[ k ], line );
				if ( a < 0 || c < 0 ) b.invalidEdge();
				else b.edge( a, c );

			}

		} else if ( section === 'matrix' ) {

			if ( ! declared ) fail( line, '"*Matrix" before "*Vertices"' );
			const f = s.split( /\s+/ );
			if ( row >= declared ) {

				b.warn( `line ${line}: more matrix rows than vertices` );
				return;

			}

			if ( f.length !== declared ) b.warn( `line ${line}: matrix row ${row + 1} has ${f.length} entries, expected ${declared}` );
			for ( let c = 0; c < f.length && c < declared; c ++ ) {

				const w = Number( f[ c ] );
				if ( w !== 0 && Number.isFinite( w ) ) b.edge( row, c, Math.abs( w ) );

			}

			row ++;

		} else if ( section === 'values' ) {

			if ( valueIndex < b.nodeCount ) b.attr( valueIndex, valueName, s.split( /\s+/ )[ 0 ] );
			valueIndex ++;

		} else if ( section === null && ! sawNetwork ) fail( line, `expected a section such as "*Vertices 34", found "${s.slice( 0, 30 )}"` );

	}, ( f ) => b.progress( f ) );

	if ( b.nodeCount === 0 ) throw new GraphError( 'Pajek: no "*Vertices" section and no edges found', 'format' );
	return b.finish( { format: 'pajek', numericOrder: true } );

}

// ── Matrix Market ────────────────────────────────────────────────────────────────────────────

export function parseMtx( text, context = {} ) {

	const b = new GraphBuilder( context );
	const fail = ( line, message ) => {

		throw new GraphError( `Matrix Market: line ${line}: ${message}`, 'parse' );

	};

	let header = null, size = null, bipartite = false;
	let entries = 0, zeros = 0, outOfRange = 0;
	let col = 0, rowInCol = 0; // array format cursor

	eachLine( text, ( s, line ) => {

		if ( ! header ) {

			const f = s.toLowerCase().split( /\s+/ );
			if ( f[ 0 ] !== '%%matrixmarket' ) fail( line, 'the file does not start with "%%MatrixMarket"' );
			if ( f[ 1 ] !== 'matrix' ) fail( line, `object "${f[ 1 ] || ''}" is not supported (only "matrix")` );
			if ( f[ 2 ] !== 'coordinate' && f[ 2 ] !== 'array' ) fail( line, `format "${f[ 2 ] || ''}" is not supported (coordinate or array)` );
			header = { layout: f[ 2 ], field: f[ 3 ] || 'real', symmetry: f[ 4 ] || 'general' };
			if ( ! [ 'real', 'integer', 'complex', 'pattern', 'double' ].includes( header.field ) ) fail( line, `field "${header.field}" is not supported` );
			if ( ! [ 'general', 'symmetric', 'skew-symmetric', 'hermitian' ].includes( header.symmetry ) ) fail( line, `symmetry "${header.symmetry}" is not supported` );
			return;

		}

		if ( s.charCodeAt( 0 ) === 37 ) return;
		const f = s.split( /\s+/ );
		if ( ! size ) {

			const rows = Number( f[ 0 ] ), cols = Number( f[ 1 ] );
			if ( ! Number.isInteger( rows ) || ! Number.isInteger( cols ) || rows < 1 || cols < 1 || ( header.layout === 'coordinate' && f.length < 3 ) ) fail( line, `expected the size line "rows columns${header.layout === 'coordinate' ? ' entries' : ''}", found "${s.slice( 0, 40 )}"` );
			size = { rows, cols, nnz: header.layout === 'coordinate' ? Number( f[ 2 ] ) : rows * cols };
			bipartite = rows !== cols;
			if ( bipartite && header.symmetry !== 'general' ) fail( line, `a ${header.symmetry} matrix must be square (${rows} x ${cols})` );
			const count = bipartite ? rows + cols : rows;
			if ( count > b.limits.maxNodes ) throw tooLarge( 'nodes', count, b.limits );
			if ( bipartite ) {

				for ( let k = 1; k <= rows; k ++ ) b.attr( b.node( 'r' + k ), 'side', 'row' );
				for ( let k = 1; k <= cols; k ++ ) b.attr( b.node( 'c' + k ), 'side', 'column' );

			} else for ( let k = 1; k <= rows; k ++ ) b.node( String( k ) );

			b.directed = ! bipartite && header.symmetry === 'general';
			if ( header.symmetry === 'skew-symmetric' ) rowInCol = 1; // no diagonal
			return;

		}

		let r, c, value;
		if ( header.layout === 'coordinate' ) {

			r = Number( f[ 0 ] );
			c = Number( f[ 1 ] );
			if ( f.length < 2 || ! Number.isInteger( r ) || ! Number.isInteger( c ) ) fail( line, `expected "row column${header.field === 'pattern' ? '' : ' value'}", found "${s.slice( 0, 40 )}"` );
			value = header.field === 'pattern' || f.length < 3 ? 1 : header.field === 'complex' ? Math.hypot( Number( f[ 2 ] ), Number( f[ 3 ] || 0 ) ) : Number( f[ 2 ] );

		} else {

			// column-major; the symmetric kinds store the lower triangle only
			if ( col >= size.cols ) {

				outOfRange ++;
				return;

			}

			r = rowInCol + 1;
			c = col + 1;
			value = header.field === 'complex' ? Math.hypot( Number( f[ 0 ] ), Number( f[ 1 ] || 0 ) ) : Number( f[ 0 ] );
			if ( ++ rowInCol >= size.rows ) {

				col ++;
				rowInCol = header.symmetry === 'general' ? 0 : col + ( header.symmetry === 'skew-symmetric' ? 1 : 0 );

			}

		}

		entries ++;
		if ( r < 1 || c < 1 || r > size.rows || c > size.cols ) {

			if ( outOfRange ++ === 0 ) b.warn( `line ${line}: entry (${r}, ${c}) is outside the ${size.rows} x ${size.cols} matrix` );
			b.invalidEdge();
			return;

		}

		if ( ! Number.isFinite( value ) ) fail( line, `"${f[ 2 ]}" is not a number` );
		if ( value === 0 ) {

			zeros ++;
			return;

		}

		b.edge( r - 1, bipartite ? size.rows + c - 1 : c - 1, Math.abs( value ) );

	}, ( f ) => b.progress( f ) );

	if ( ! header ) throw new GraphError( 'Matrix Market: the file is empty', 'empty' );
	if ( ! size ) throw new GraphError( 'Matrix Market: the size line is missing', 'parse' );
	if ( header.layout === 'coordinate' && Number.isFinite( size.nnz ) && entries !== size.nnz ) b.warn( `the header announces ${size.nnz} entries, the file has ${entries}` );
	if ( outOfRange > 1 ) b.warn( `${outOfRange} entries are outside the matrix` );
	return b.finish( { format: 'mtx', meta: { matrix: { rows: size.rows, columns: size.cols, field: header.field, symmetry: header.symmetry, layout: header.layout, zeros, bipartite } } } );

}
