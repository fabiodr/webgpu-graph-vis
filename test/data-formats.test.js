import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseGraph, detectFormat, listFormats, acceptExtensions, acceptsName, FORMATS } from '../src/data/formats.js';
import { GraphBuilder, GraphError, DEFAULT_LIMITS, typeColumn, weightWarnings } from '../src/data/builder.js';
import { scanXml, parseGraphML, parseGexf } from '../src/data/parse-xml.js';
import { parseGml, parseDot, parsePajek, parseMtx } from '../src/data/parse-text.js';
import { detectJsonShape } from '../src/data/parse-json.js';
import { parseEdgeList } from '../src/data/edgelist.js';
import { exportGraph } from '../src/data/export.js';
import { cleanEdges, normalizeGraph, serializeGraph, MAX_ATTRIBUTES, MAX_CATEGORIES, MAX_WEIGHT } from '../src/data/format.js';
import { clusterGraph } from '../src/data/hierarchy.js';
import { DataClient } from '../src/data/client.js';
import { runRequest } from '../src/data/worker.js';
import { generate } from '../src/data/generators.js';

// edges as sorted "label-label" strings, weights by the same key
const pairs = ( g ) => {

	const out = [];
	for ( let k = 0; k < g.edges.length; k += 2 ) out.push( [ g.labels[ g.edges[ k ] ], g.labels[ g.edges[ k + 1 ] ] ].sort().join( '~' ) );
	return out.sort();

};

const weightMap = ( g ) => {

	const out = {};
	for ( let e = 0; e < g.edges.length / 2; e ++ ) out[ [ g.labels[ g.edges[ 2 * e ] ], g.labels[ g.edges[ 2 * e + 1 ] ] ].sort().join( '~' ) ] = g.weights ? g.weights[ e ] : 1;
	return out;

};

const fails = ( fn, pattern, code ) => {

	assert.throws( fn, ( e ) => {

		assert.ok( e instanceof GraphError, `GraphError expected, got ${e && e.name}: ${e && e.message}` );
		assert.match( e.message, pattern );
		if ( code ) assert.equal( e.code, code );
		return true;

	} );

};

// ── the same small graph in every format ─────────────────────────────────────────────────────
// nodes A B C D (kind: x x y y, score: 1 2.5 3 missing), edges A-B (2), B-C (1), C-D (0.5), a
// self loop on D, and A-B once more in the other direction (weight 3): A-B must come out as 5.

const SAMPLES = {
	graphml: `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="k0" for="node" attr.name="label" attr.type="string"/>
  <key id="k1" for="node" attr.name="kind" attr.type="string"/>
  <key id="k2" for="node" attr.name="score" attr.type="double"/>
  <key id="k3" for="edge" attr.name="weight" attr.type="double"/>
  <graph id="G" edgedefault="directed">
    <node id="n0"><data key="k0">A</data><data key="k1">x</data><data key="k2">1</data></node>
    <node id="n1"><data key="k0">B</data><data key="k1">x</data><data key="k2">2.5</data></node>
    <node id="n2"><data key="k0">C</data><data key="k1">y</data><data key="k2">3</data></node>
    <node id="n3"><data key="k0">D</data><data key="k1">y</data></node>
    <edge source="n0" target="n1"><data key="k3">2</data></edge>
    <edge source="n1" target="n2"/>
    <edge source="n2" target="n3"><data key="k3">0.5</data></edge>
    <edge source="n3" target="n3"/>
    <edge source="n1" target="n0"><data key="k3">3</data></edge>
  </graph>
</graphml>`,
	gexf: `<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" version="1.3">
  <graph mode="static" defaultedgetype="directed">
    <attributes class="node">
      <attribute id="0" title="kind" type="string"/>
      <attribute id="1" title="score" type="float"/>
    </attributes>
    <nodes>
      <node id="a" label="A"><attvalues><attvalue for="0" value="x"/><attvalue for="1" value="1"/></attvalues></node>
      <node id="b" label="B"><attvalues><attvalue for="0" value="x"/><attvalue for="1" value="2.5"/></attvalues></node>
      <node id="c" label="C"><attvalues><attvalue for="0" value="y"/><attvalue for="1" value="3"/></attvalues></node>
      <node id="d" label="D"><attvalues><attvalue for="0" value="y"/></attvalues></node>
    </nodes>
    <edges>
      <edge id="0" source="a" target="b" weight="2"/>
      <edge id="1" source="b" target="c"/>
      <edge id="2" source="c" target="d" weight="0.5"/>
      <edge id="3" source="d" target="d"/>
      <edge id="4" source="b" target="a" weight="3"/>
    </edges>
  </graph>
</gexf>`,
	gml: `Creator "test"
graph [
  directed 1
  node [ id 1 label "A" kind "x" score 1 ]
  node [ id 2 label "B" kind "x" score 2.5 ]
  node [ id 3 label "C" kind "y" score 3 graphics [ x 1 y 2 ] ]
  node [ id 4 label "D" kind "y" ]
  edge [ source 1 target 2 weight 2 ]
  edge [ source 2 target 3 ]
  edge [ source 3 target 4 value 0.5 ]
  edge [ source 4 target 4 ]
  edge [ source 2 target 1 weight 3 ]
]`,
	dot: `digraph G {
  // nodes
  n0 [label="A", kind=x, score=1];
  n1 [label="B", kind=x, score=2.5];
  n2 [label="C", kind=y, score=3];
  n3 [label="D", kind=y];
  n0 -> n1 [weight=2];
  n1 -> n2;
  n2 -> n3 [weight=0.5];
  n3 -> n3;
  n1 -> n0 [weight=3]
}`,
	pajek: `*Vertices 4
1 "A"
2 "B"
3 "C"
4 "D"
*Arcs
1 2 2
2 3 1
3 4 0.5
4 4 1
2 1 3
`,
	mtx: `%%MatrixMarket matrix coordinate real general
% a comment
4 4 5
1 2 2
2 3 1
3 4 0.5
4 4 1
2 1 3
`,
	edgelist: `source,target,weight
A,B,2
B,C,1
C,D,0.5
D,D,1
B,A,3
`,
	'node-link': JSON.stringify( {
		directed: true,
		nodes: [ { id: 'a', name: 'A', kind: 'x', score: 1 }, { id: 'b', name: 'B', kind: 'x', score: 2.5 }, { id: 'c', name: 'C', kind: 'y', score: 3 }, { id: 'd', name: 'D', kind: 'y' } ],
		links: [ { source: 'a', target: 'b', weight: 2 }, { source: 'b', target: 'c' }, { source: 'c', target: 'd', value: 0.5 }, { source: 'd', target: 'd' }, { source: 'b', target: 'a', weight: 3 } ],
	} ),
	cytoscape: JSON.stringify( { elements: {
		nodes: [ { data: { id: 'a', name: 'A', kind: 'x', score: 1 } }, { data: { id: 'b', name: 'B', kind: 'x', score: 2.5 } }, { data: { id: 'c', name: 'C', kind: 'y', score: 3 } }, { data: { id: 'd', name: 'D', kind: 'y' } } ],
		edges: [ { data: { id: 'e0', source: 'a', target: 'b', weight: 2 } }, { data: { source: 'b', target: 'c' } }, { data: { source: 'c', target: 'd', weight: 0.5 } }, { data: { source: 'd', target: 'd' } }, { data: { source: 'b', target: 'a', weight: 3 } } ],
	} } ),
	graphology: JSON.stringify( {
		options: { type: 'directed', multi: false },
		nodes: [ { key: 'a', attributes: { label: 'A', kind: 'x', score: 1 } }, { key: 'b', attributes: { label: 'B', kind: 'x', score: 2.5 } }, { key: 'c', attributes: { label: 'C', kind: 'y', score: 3 } }, { key: 'd', attributes: { label: 'D', kind: 'y' } } ],
		edges: [ { source: 'a', target: 'b', attributes: { weight: 2 } }, { source: 'b', target: 'c' }, { source: 'c', target: 'd', attributes: { weight: 0.5 } }, { source: 'd', target: 'd' }, { source: 'b', target: 'a', attributes: { weight: 3 } } ],
	} ),
	adjacency: JSON.stringify( { A: { B: 2 }, B: { C: 1, A: 3 }, C: { D: 0.5 }, D: { D: 1 } } ),
};

const FILE_NAMES = { graphml: 'g.graphml', gexf: 'g.gexf', gml: 'g.gml', dot: 'g.dot', pajek: 'g.net', mtx: 'g.mtx', edgelist: 'g.csv', 'node-link': 'g.json', cytoscape: 'g.cyjs', graphology: 'g.json', adjacency: 'g.json' };
const WITH_ATTRIBUTES = [ 'graphml', 'gexf', 'gml', 'dot', 'node-link', 'cytoscape', 'graphology' ];

test( 'every format: the same graph, weights summed over both directions, self loop dropped, report', () => {

	for ( const [ format, text ] of Object.entries( SAMPLES ) ) {

		// content detection must not depend on the file name
		for ( const name of [ FILE_NAMES[ format ], 'download', 'renamed.txt' ] ) {

			const r = parseGraph( text, name );
			assert.equal( r.format, format, `${format} as ${name}` );
			const g = r.graph;
			assert.equal( g.nodeCount, 4, format );
			assert.ok( g.edges instanceof Uint32Array );
			const labels = format === 'mtx' ? [ '1', '2', '3', '4' ] : [ 'A', 'B', 'C', 'D' ];
			assert.deepEqual( g.labels, labels, format );
			const [ a, b, c, d ] = labels;
			assert.deepEqual( pairs( g ), [ `${a}~${b}`, `${b}~${c}`, `${c}~${d}` ].sort(), format );
			assert.deepEqual( weightMap( g ), { [ `${a}~${b}` ]: 5, [ `${b}~${c}` ]: 1, [ `${c}~${d}` ]: 0.5 }, format );
			assert.ok( g.weights instanceof Float32Array );
			const rep = g.meta.report;
			assert.equal( rep.format, format );
			assert.deepEqual( rep.dropped, { selfLoops: 1, duplicates: 1, invalid: 0 }, format );
			assert.equal( rep.weighted, true );
			assert.equal( rep.nodes, 4 );
			assert.equal( rep.edges, 3 );
			if ( ! [ 'edgelist', 'adjacency', 'cytoscape' ].includes( format ) ) assert.equal( rep.directed, true, format ); // these say nothing about direction
			assert.equal( g.hierarchy, null );

		}

	}

} );

test( 'attributes: numbers and categories, missing values, label attribute', () => {

	for ( const format of WITH_ATTRIBUTES ) {

		const g = parseGraph( SAMPLES[ format ], FILE_NAMES[ format ] ).graph;
		assert.deepEqual( Object.keys( g.attributes ).sort(), [ 'kind', 'score' ], format );
		const kind = g.attributes.kind, score = g.attributes.score;
		assert.equal( kind.type, 'category' );
		assert.ok( kind.values instanceof Int32Array );
		assert.deepEqual( kind.categories, [ 'x', 'y' ] );
		assert.deepEqual( Array.from( kind.values ), [ 0, 0, 1, 1 ] );
		assert.equal( score.type, 'number' );
		assert.ok( score.values instanceof Float32Array );
		assert.deepEqual( Array.from( score.values.subarray( 0, 3 ) ), [ 1, 2.5, 3 ] );
		assert.ok( Number.isNaN( score.values[ 3 ] ), `${format}: missing number is NaN` );
		assert.deepEqual( g.meta.report.attributes, [ { name: 'kind', type: 'category', categories: 2 }, { name: 'score', type: 'number' } ] );
		// the name / label column became the labels and is not repeated as an attribute
		assert.match( g.meta.report.labels, /label|name/ );

	}

} );

test( 'builder: typing rules, caps, label choice, group column, numeric order', () => {

	// more than MAX_CATEGORIES distinct strings is free text; all-different strings are ids
	const many = Array.from( { length: 200 }, ( _, i ) => `v${i % ( MAX_CATEGORIES + 1 )}` );
	assert.equal( typeColumn( many, 200 ).kind, 'text' );
	assert.equal( typeColumn( many.map( ( v, i ) => `v${i % MAX_CATEGORIES}` ), 200 ).kind, 'category' );
	assert.equal( typeColumn( Array.from( { length: 50 }, ( _, i ) => `id${i}` ), 50 ).kind, 'text' );
	assert.equal( typeColumn( [ '1', '2', ' 3.5 ', 'NaN', null, '' ], 6 ).kind, 'number' );
	assert.equal( typeColumn( [ '1', 'two', '3' ], 3 ).kind, 'category' );
	assert.equal( typeColumn( [ 7, 7, 7 ], 3 ).kind, 'constant' );
	assert.equal( typeColumn( [ undefined, null, '' ], 3 ).kind, 'empty' );
	assert.equal( typeColumn( [ true, false, true ], 3 ).kind, 'category' );
	const cls = typeColumn( [ 3, 1, 3, 2 ], 4, true );
	assert.equal( cls.kind, 'category' );
	assert.deepEqual( cls.categories, [ '1', '2', '3' ] ); // class ids in numeric order
	assert.deepEqual( Array.from( cls.values ), [ 2, 0, 2, 1 ] );

	const b = new GraphBuilder( { name: 'caps' } );
	const n = 40;
	for ( let i = 0; i < n; i ++ ) {

		const idx = b.node( `n${i}` );
		for ( let k = 0; k < MAX_ATTRIBUTES + 5; k ++ ) b.attr( idx, `a${k}`, i * ( k + 1 ) );
		b.attr( idx, 'title', `Node ${i}` );
		b.attr( idx, 'community', i % 4 );
		b.attr( idx, 'bio', `long text ${i}` );
		b.attr( idx, 'same', 'x' );
		if ( i ) b.edge( idx, idx - 1 );

	}

	const g = b.finish( { format: 'test' } );
	assert.equal( Object.keys( g.attributes ).length, MAX_ATTRIBUTES );
	assert.deepEqual( g.labels.slice( 0, 2 ), [ 'Node 0', 'Node 1' ] );
	assert.equal( g.meta.report.labels, 'attribute "title"' );
	const droppedNames = g.meta.report.droppedAttributes.map( ( d ) => d.name );
	assert.ok( droppedNames.includes( 'bio' ) && droppedNames.includes( 'same' ) );
	assert.ok( g.meta.report.droppedAttributes.some( ( d ) => /more than 32 attributes/.test( d.reason ) ) );
	assert.match( g.meta.report.droppedAttributes.find( ( d ) => d.name === 'bio' ).reason, /40 distinct values/ );
	// the community column is the planted top level, whether or not it was kept as an attribute
	assert.ok( g.planted && g.planted.complete === false );
	assert.deepEqual( Array.from( g.planted.levels[ 0 ].slice( 0, 6 ) ), [ 0, 1, 2, 3, 0, 1 ] );
	assert.deepEqual( g.meta.report.groups, { source: 'attribute "community"', count: 4, levels: 1 } );
	assert.ok( ! Object.keys( g ).includes( 'planted' ), 'planted is not enumerable (not serialized)' );

	// numeric ids: numeric node order on request
	const c = new GraphBuilder();
	c.edge( c.node( '10' ), c.node( '9' ) );
	c.edge( c.node( '9' ), c.node( '100' ) );
	assert.deepEqual( c.finish( { format: 'x', numericOrder: true } ).labels, [ '9', '10', '100' ] );
	const d = new GraphBuilder();
	d.edge( d.node( '10' ), d.node( '9' ) );
	assert.deepEqual( d.finish( { format: 'x' } ).labels, [ '10', '9' ] );

} );

test( 'cleanEdges: order of first occurrence, summed weights, mirrored arcs count once', () => {

	const r = cleanEdges( 5, [ 3, 4, 0, 1, 1, 0, 2, 2, 1, 2, 9, 1, 4, 3 ], [ 1, 2, 4, 1, 1.5, 1, 1 ] );
	assert.deepEqual( Array.from( r.edges ), [ 3, 4, 0, 1, 1, 2 ] );
	assert.deepEqual( Array.from( r.weights ), [ 2, 6, 1.5 ] );
	assert.deepEqual( r.dropped, { selfLoops: 1, duplicates: 2, invalid: 1 } );
	assert.equal( r.mirrored, false );
	// every edge once per direction with equal weights: an undirected graph stored as arcs
	const m = cleanEdges( 3, [ 0, 1, 1, 0, 1, 2, 2, 1 ], null );
	assert.equal( m.mirrored, true );
	assert.equal( m.weights, null );
	const mw = cleanEdges( 3, [ 0, 1, 1, 0, 1, 2, 2, 1 ], [ 2, 2, 3, 3 ] );
	assert.deepEqual( Array.from( mw.weights ), [ 2, 3 ] );
	// unweighted duplicates count: weight 2
	assert.deepEqual( Array.from( cleanEdges( 2, [ 0, 1, 0, 1 ], null ).weights ), [ 2 ] );
	// negative and broken weights
	assert.deepEqual( Array.from( cleanEdges( 3, [ 0, 1, 1, 2 ], [ - 2, NaN ] ).weights ), [ 2, 1 ] );
	assert.equal( cleanEdges( 3, [], null ).edges.length, 0 );

} );

test( 'detection: content first, extension second, friendly rejections', () => {

	assert.deepEqual( detectFormat( '  <?xml version="1.0"?>\n<!-- c -->\n<graphml>', 'x.txt' ), { id: 'graphml', by: 'content' } );
	assert.deepEqual( detectFormat( '<ns:gexf version="1.2">', 'x' ), { id: 'gexf', by: 'content' } );
	assert.deepEqual( detectFormat( '﻿{"nodes":[]}', 'x.csv' ), { id: 'json', by: 'content' } );
	assert.deepEqual( detectFormat( '%%MatrixMarket matrix coordinate pattern general\n', 'x.txt' ), { id: 'mtx', by: 'content' } );
	assert.deepEqual( detectFormat( '% c\n*Vertices 3\n', 'x.txt' ), { id: 'pajek', by: 'content' } );
	assert.deepEqual( detectFormat( '/* c */\n// d\nstrict digraph "a b" {', 'x.txt' ), { id: 'dot', by: 'content' } );
	assert.deepEqual( detectFormat( 'graph\n{\n a -- b }', 'x' ), { id: 'dot', by: 'content' } );
	assert.deepEqual( detectFormat( '# c\nCreator "yEd"\nVersion "2.14"\ngraph\n[', 'x.txt' ), { id: 'gml', by: 'content' } );
	assert.deepEqual( detectFormat( '1 2\n2 3\n', 'x.unknown' ), { id: 'edgelist', by: 'default' } );
	assert.deepEqual( detectFormat( '1 2\n2 3\n', 'x.tsv' ), { id: 'edgelist', by: 'extension' } );
	// content says nothing: the extension decides (here a GML file whose graph starts late)
	assert.deepEqual( detectFormat( '# ' + 'x'.repeat( 9000 ) + '\ngraph [ ]', 'late.gml' ), { id: 'gml', by: 'extension' } );
	assert.deepEqual( detectFormat( 'a b\n', 'edges.gz' ), { id: 'edgelist', by: 'default' } );
	assert.deepEqual( detectFormat( '<graphml/>', 'https://example.org/data/g.graphml.gz?x=1' ).id, 'graphml' );
	fails( () => detectFormat( '\u0000\u0001\u0002PNG', 'pic.png' ), /pic\.png is not a text file/, 'format' );
	fails( () => detectFormat( '<svg xmlns="x">', 'pic.svg' ), /root element <svg>.*GraphML.*GEXF/, 'format' );
	fails( () => detectFormat( ' \n\t ', 'empty.txt' ), /empty/, 'empty' );
	// a graph called "graph" in an edge list is not DOT
	assert.equal( detectFormat( 'graph b\nb c\n', 'x' ).id, 'edgelist' );

} );

test( 'formats(): the registry behind data.formats()', () => {

	const list = listFormats();
	assert.deepEqual( list.map( ( f ) => f.id ), [ 'graphml', 'gexf', 'gml', 'dot', 'pajek', 'mtx', 'edgelist', 'json' ] );
	for ( const f of list ) {

		assert.ok( f.name && Array.isArray( f.extensions ) && f.extensions.length && f.extensions.every( ( e ) => /^\.[a-z0-9]+$/.test( e ) ), f.id );
		assert.equal( typeof f.description, 'string' );

	}

	list[ 0 ].extensions.push( '.oops' ); // copies: the registry itself is untouched
	assert.ok( ! FORMATS[ 0 ].extensions.includes( '.oops' ) );
	const all = acceptExtensions();
	for ( const ext of [ '.graphml', '.gexf', '.gml', '.dot', '.gv', '.net', '.mtx', '.csv', '.tsv', '.txt', '.json', '.gz' ] ) assert.ok( all.includes( ext ), ext );
	assert.equal( new Set( all ).size, all.length, 'no extension is claimed twice' );
	const client = new DataClient( { inline: true } );
	assert.deepEqual( client.formats(), listFormats() );
	// what a file input and a drop filter need: every format the pipeline reads, not three of them
	assert.deepEqual( client.acceptExtensions(), all );
	for ( const name of [ 'a.graphml', 'A.GEXF', 'karate.gml', 'g.dot', 'g.gv', 'x.net', 'm.mtx', 'e.csv', 'e.tsv', 'e.txt', 'g.json', 'big.graphml.gz', 'edges.csv.GZ', 'yed.xml', 'README', 'edges' ] ) {

		assert.ok( acceptsName( name ), name );
		assert.ok( client.accepts( name ), name );

	}

	for ( const name of [ 'photo.png', 'paper.pdf', 'archive.zip', 'photo.png.gz', '.hidden.jpg' ] ) assert.ok( ! client.accepts( name ), name );
	for ( const f of list ) for ( const ext of FORMATS.find( ( x ) => x.id === f.id ).extensions ) assert.ok( acceptsName( 'file' + ext ) && acceptsName( 'file' + ext + '.gz' ), ext );
	client.dispose();

} );

// ── per format: details and malformed input ──────────────────────────────────────────────────

test( 'XML scanner: entities, CDATA, comments, DOCTYPE, namespaces; errors name the line', () => {

	const events = [];
	const h = {
		wantsText: true,
		open: ( name, attrs, qname ) => events.push( [ 'open', name, qname, attrs ] ),
		close: ( name ) => events.push( [ 'close', name ] ),
		text: ( s ) => {

			if ( s.trim() ) events.push( [ 'text', s.trim() ] );

		},
	};
	scanXml( `<?xml version="1.0"?>
<!DOCTYPE a [ <!ENTITY x "y"> ]>
<!-- <not a="tag"> -->
<a x="1 &lt; 2" y='say &quot;hi&quot; &#65;&#x42;'>
  <ns:b/> text &amp; more <![CDATA[<raw & stuff>]]>
</a>`, h );
	assert.deepEqual( events, [
		[ 'open', 'a', 'a', { x: '1 < 2', y: 'say "hi" AB' } ],
		[ 'open', 'b', 'ns:b', {} ],
		[ 'close', 'b' ],
		[ 'text', 'text & more' ],
		[ 'text', '<raw & stuff>' ],
		[ 'close', 'a' ],
	] );
	const none = { open() {}, close() {} };
	fails( () => scanXml( '<a>\n<b>\n</a>', none ), /<\/a> closes <b>, which was opened in line 2 \(line 3\)/ );
	fails( () => scanXml( '<a>\n<b x=1/>\n</a>', none ), /attribute "x" of <b> is not quoted \(line 2\)/ );
	fails( () => scanXml( '<a>\n\n<b>', none ), /<b> is never closed.*\(line 3\)/ );
	fails( () => scanXml( '<a><!-- never closed', none ), /comment is not closed/ );
	fails( () => scanXml( '<a x="1></a>', none ), /not closed/ );
	fails( () => scanXml( '</a>', none ), /<\/a> has no start tag/ );

} );

test( 'GraphML: defaults, yEd labels, nested graphs as groups, undeclared nodes, hyperedges', () => {

	const g = parseGraphML( `<graphml xmlns:y="http://www.yworks.com/xml/graphml">
  <key id="d0" for="node" yfiles.type="nodegraphics"/>
  <key id="d1" for="node" attr.name="color" attr.type="string"><default>grey</default></key>
  <key id="d2" for="edge" attr.name="Weight" attr.type="double"/>
  <graph edgedefault="undirected">
    <node id="g1">
      <data key="d0"><y:ProxyAutoBoundsNode><y:Realizers><y:GroupNode><y:NodeLabel>Team One</y:NodeLabel></y:GroupNode></y:Realizers></y:ProxyAutoBoundsNode></data>
      <graph id="g1:">
        <node id="a"><data key="d0"><y:ShapeNode><y:Geometry x="1"/><y:NodeLabel>Ann &amp; Co</y:NodeLabel></y:ShapeNode></data><data key="d1">red</data></node>
        <node id="b"><data key="d0"><y:ShapeNode><y:NodeLabel>Bob</y:NodeLabel></y:ShapeNode></data></node>
      </graph>
    </node>
    <node id="g2"><graph><node id="c"><data key="d1">red</data></node><node id="d"/></graph></node>
    <edge source="a" target="b"><data key="d2">2.5</data></edge>
    <edge source="b" target="c"/>
    <edge source="c" target="d"/>
    <edge source="d" target="zz"/>
    <hyperedge><endpoint node="a"/><endpoint node="b"/></hyperedge>
    <edge target="a"/>
  </graph>
</graphml>`, { name: 'yed' } );
	assert.deepEqual( g.labels, [ 'Ann & Co', 'Bob', 'c', 'd', 'zz' ] ); // the two group nodes are gone
	assert.equal( g.meta.report.containers, 2 );
	assert.deepEqual( weightMap( g ), { 'Ann & Co~Bob': 2.5, 'Bob~c': 1, 'c~d': 1, 'd~zz': 1 } );
	assert.deepEqual( Array.from( g.attributes.color.values ).map( ( v ) => g.attributes.color.categories[ v ] ), [ 'red', 'grey', 'red', 'grey', 'grey' ] );
	assert.deepEqual( g.attributes.group.categories, [ 'Team One', 'g2' ] );
	assert.deepEqual( Array.from( g.attributes.group.values ), [ 0, 0, 1, 1, - 1 ] );
	assert.deepEqual( Array.from( g.planted.levels[ 0 ] ), [ 0, 0, 1, 1, 2 ] );
	assert.equal( g.meta.report.dropped.invalid, 1 );
	assert.ok( g.meta.report.warnings.some( ( w ) => /hyperedge/.test( w ) ) );
	assert.ok( g.meta.report.warnings.some( ( w ) => /line 19: <edge> without source or target/.test( w ) ) );
	assert.ok( g.meta.report.warnings.some( ( w ) => /declared later or not at all/.test( w ) ) );

	fails( () => parseGraphML( '<graphml><graph><node id="a"></graph></graphml>' ), /closes <node>/, 'parse' );
	fails( () => parseGraphML( '<graphml>\n<graph>\n<node id="a"/>\n' ), /<graph> is never closed.*line 2/, 'parse' );
	fails( () => parseGraphML( '<html><body/></html>' ), /root element is <html>, expected <graphml>/, 'format' );
	fails( () => parseGraphML( '<graphml></graphml>' ), /no <graph> element/, 'format' );
	fails( () => parseGraphML( '<graphml><graph></graph></graphml>', { name: 'e' } ), /no nodes/, 'empty' );

} );

test( 'GEXF: modularity_class as groups, pid hierarchy, viz:size, dynamic mode, old attvalue ids', () => {

	const g = parseGexf( `<gexf xmlns:viz="http://www.gexf.net/1.2draft/viz">
  <meta><description>Two teams</description></meta>
  <graph mode="dynamic" defaultedgetype="undirected">
    <attributes class="node"><attribute id="mc" title="modularity_class" type="integer"><default>9</default></attribute></attributes>
    <attributes class="edge"><attribute id="w2" title="weight" type="float"/></attributes>
    <nodes>
      <node id="1" label="one"><attvalues><attvalue for="mc" value="0"/></attvalues><viz:size value="10"/></node>
      <node id="2" label="two"><attvalues><attvalue id="mc" value="0"/></attvalues><viz:size value="20"/></node>
      <node id="3" label="three"><attvalues><attvalue for="mc" value="1"/></attvalues><viz:size value="30"/></node>
      <node id="4" label="four"/>
    </nodes>
    <edges>
      <edge source="1" target="2"><attvalues><attvalue for="w2" value="4"/></attvalues></edge>
      <edge source="2" target="3" weight="2"/>
      <edge source="3" target="4" type="directed"/>
    </edges>
  </graph>
</gexf>`, { name: 'gephi' } );
	assert.deepEqual( g.labels, [ 'one', 'two', 'three', 'four' ] );
	assert.equal( g.description, 'Two teams' );
	assert.deepEqual( weightMap( g ), { 'one~two': 4, 'three~two': 2, 'four~three': 1 } );
	assert.deepEqual( g.attributes.modularity_class.categories, [ '0', '1', '9' ] ); // the default fills node 4
	assert.deepEqual( Array.from( g.attributes.modularity_class.values ), [ 0, 0, 1, 2 ] );
	assert.deepEqual( Array.from( g.attributes.size.values ).slice( 0, 3 ), [ 10, 20, 30 ] );
	assert.equal( g.meta.report.directed, true ); // one directed edge
	assert.ok( g.meta.report.warnings.some( ( w ) => /dynamic/.test( w ) ) );

	const h = parseGexf( `<gexf><graph>
    <nodes>
      <node id="top" label="Top"><nodes><node id="a"/><node id="b"/></nodes></node>
      <node id="c" pid="other"/><node id="d" pid="other"/><node id="other" label="Other"/>
    </nodes>
    <edges><edge source="a" target="b"/><edge source="c" target="d"/><edge source="b" target="c"/></edges>
  </graph></gexf>` );
	assert.deepEqual( h.labels, [ 'a', 'b', 'c', 'd' ] );
	assert.deepEqual( Array.from( h.planted.levels[ 0 ] ), [ 0, 0, 1, 1 ] );
	assert.deepEqual( h.attributes.group.categories, [ 'Top', 'Other' ] );

	fails( () => parseGexf( '<gexf><graph><nodes><node id="a"></nodes></graph></gexf>' ), /closes <node>/ );
	fails( () => parseGexf( '<gexf></gexf>' ), /no <graph>/, 'format' );
	fails( () => parseGexf( '<graphml/>' ), /expected <gexf>/, 'format' );

} );

test( 'GML: value as weight, attributes, comments, multi-line strings, yEd groups; malformed input', () => {

	const g = parseGml( `# Newman style
graph
[
  node [ id 0 label "zero
one" value 2 ]
  node [ id 1 label "A &amp; B" value 5 ]
  node [ id 2 value 7 gid 9 ]
  node [ id 3 value 8 gid 9 ]
  node [ id 9 label "Box" isGroup 1 ]
  edge [ source 0 target 1 value 3 ]
  edge [ source 1 target 2 label "x" ]
  edge [ source 2 target 3 ]
  edge [ source 2 ]
]` );
	assert.deepEqual( g.labels, [ 'zero\none', 'A & B', '2', '3' ] );
	assert.deepEqual( Array.from( g.edges ), [ 0, 1, 1, 2, 2, 3 ] );
	assert.deepEqual( Array.from( g.weights ), [ 3, 1, 1 ] );
	assert.deepEqual( Array.from( g.attributes.value.values ), [ 2, 5, 7, 8 ] );
	assert.deepEqual( Array.from( g.planted.levels[ 0 ] ), [ 0, 0, 1, 1 ] );
	assert.equal( g.meta.report.dropped.invalid, 1 );
	assert.match( g.meta.report.warnings[ 0 ], /line 13: edge without source or target/ );
	assert.equal( g.directed, undefined );

	fails( () => parseGml( 'graph [\n node [ id 1 ]\n edge [ source 1\n' ), /line 3: the edge that starts here is never closed/, 'parse' );
	fails( () => parseGml( 'graph [ node [ id 1 label "x ] ]' ), /line 1: a string is not closed/, 'parse' );
	fails( () => parseGml( 'graph [\n node [ id ]\n]' ), /line 2: the key "id" has no value/ );
	fails( () => parseGml( 'graph [\n node [ 5 5 ] ]' ), /line 2: expected a key inside node/ );
	fails( () => parseGml( 'just words here' ), /no "graph \[ … \]" block/, 'format' );
	fails( () => parseGml( 'graph [ directed 1 ]', { name: 'e' } ), /no nodes/, 'empty' );

} );

test( 'DOT: chains, node groups, subgraphs as nested groups, defaults, ports, quoting, comments', () => {

	const g = parseDot( `/* header */ strict graph "My graph" {
  rankdir=LR; node [shape=box, team=red]; edge [weight=2];
  a [label="Alpha\\nOne" kind=x];
  b [label=<<b>Beta</b> &amp; co>, kind=y]; "c d" [label="C" + "D"];
  subgraph cluster_0 { label="Zero"; e; f [team=blue]; e -- f [weight=2.5]
    subgraph inner { g } }
  subgraph cluster_1 { graph [label="One"]; h -- i }
  a -- b -- "c d";      // a chain
  {a b} -- h:port:n     # a group on the left, a port on the right
  i -- g [weight=1]; i -- g;
  { rank=same; a; b }
}` );
	assert.deepEqual( g.labels, [ 'Alpha One', 'Beta & co', 'CD', 'e', 'f', 'g', 'h', 'i' ] );
	assert.deepEqual( weightMap( g ), { 'Alpha One~Beta & co': 2, 'Beta & co~CD': 2, 'e~f': 2.5, 'h~i': 2, 'Alpha One~h': 2, 'Beta & co~h': 2, 'g~i': 3 } );
	assert.equal( g.meta.report.directed, false );
	assert.deepEqual( g.attributes.team.categories, [ 'red', 'blue' ] );
	assert.deepEqual( Array.from( g.attributes.team.values ), [ 0, 0, 0, 0, 1, 0, 0, 0 ] );
	assert.ok( ! ( 'shape' in g.attributes ) && ! ( 'rank' in g.attributes ), 'visual attributes are not kept' );
	// groups: innermost (cluster_0 / inner / cluster_1) and outermost (inner lies in cluster_0)
	assert.deepEqual( g.attributes.group.categories, [ 'Zero', 'inner', 'One' ] );
	assert.deepEqual( Array.from( g.attributes.group.values ), [ - 1, - 1, - 1, 0, 0, 1, 2, 2 ] );
	assert.equal( g.planted.levels.length, 2 );
	assert.deepEqual( Array.from( g.planted.levels[ 0 ] ), [ 0, 0, 0, 1, 1, 2, 3, 3 ] );
	assert.deepEqual( Array.from( g.planted.levels[ 1 ] ), [ 0, 0, 0, 1, 1, 1, 2, 2 ] );
	assert.equal( g.meta.report.containers, 3 );

	const d = parseDot( 'digraph { a -> b; b -> a; b -> c }' );
	assert.equal( d.meta.report.directed, true );
	assert.deepEqual( weightMap( d ), { 'a~b': 2, 'b~c': 1 } );

	fails( () => parseDot( 'graph {\n a -- b [weight=2 }' ), /line 2: expected an attribute name or "\]", found "}"/, 'parse' );
	fails( () => parseDot( 'graph {\n a -- b\n  c -- ' ), /line 3: expected a node name, found the end of the file/ );
	fails( () => parseDot( 'graph {\n a -- b\n' ), /line 1: a "{" is never closed/ );
	fails( () => parseDot( 'graph { a [label="x }' ), /a string is not closed/ );
	fails( () => parseDot( 'graph {\n a ~ b }' ), /line 2: unexpected character "~"/ );
	fails( () => parseDot( 'nodes { }' ), /expected "graph" or "digraph", found "nodes"/, 'format' );
	fails( () => parseDot( 'graph { }', { name: 'e' } ), /no nodes/, 'empty' );

} );

test( 'Pajek: vertices with labels, arcs, edges, lists, matrix, partitions and vectors', () => {

	const g = parsePajek( `*Network toy
% a comment
*Vertices 6
1 "Ann Lee" 0.1 0.2 0.5 ellipse
2 Bob
3 "Cy"
*Arcs
1 2 2.5
*Edges
2 3
3 4 0.5
*Edgeslist
4 5 6
*Partition clubs.clu
*Vertices 6
1
1
1
2
2
2
*Vector age
*Vertices 6
31
32
33
34
35
36
` );
	assert.deepEqual( g.labels, [ 'Ann Lee', 'Bob', 'Cy', '4', '5', '6' ] );
	assert.deepEqual( weightMap( g ), { 'Ann Lee~Bob': 2.5, 'Bob~Cy': 1, '4~Cy': 0.5, '4~5': 1, '4~6': 1 } );
	assert.equal( g.attributes.clubs.type, 'category' );
	assert.deepEqual( Array.from( g.attributes.clubs.values ), [ 0, 0, 0, 1, 1, 1 ] );
	assert.deepEqual( Array.from( g.attributes.age.values ), [ 31, 32, 33, 34, 35, 36 ] );
	assert.deepEqual( Array.from( g.planted.levels[ 0 ] ), [ 0, 0, 0, 1, 1, 1 ] );
	assert.equal( g.meta.report.groups.source, 'attribute "clubs"' );

	// a symmetric matrix is an undirected graph: weights are not doubled
	const m = parsePajek( '*vertices 3\n*matrix\n0 1 2\n1 0 0\n2 0 0\n' );
	assert.deepEqual( Array.from( m.edges ), [ 0, 1, 0, 2 ] );
	assert.deepEqual( Array.from( m.weights ), [ 1, 2 ] );
	assert.equal( m.meta.report.mirrored, true );

	// edges without a vertex section; ids in numeric order
	assert.deepEqual( parsePajek( '*Edges\n10 9\n9 100\n' ).labels, [ '9', '10', '100' ] );
	const w = parsePajek( '*Vertices 2\n*Edges\n1 2\n1 7\n*Weird stuff\nabc\n' );
	assert.equal( w.meta.report.dropped.invalid, 1 );
	assert.match( w.meta.report.warnings[ 0 ], /line 4: vertex 7 is outside 1\.\.2/ );
	assert.match( w.meta.report.warnings[ 1 ], /line 5: section "\*Weird stuff" is not supported/ );

	fails( () => parsePajek( 'hello\n1 2\n' ), /line 1: expected a section such as "\*Vertices 34", found "hello"/, 'parse' );
	fails( () => parsePajek( '*Vertices many\n' ), /line 1: "\*Vertices" needs a count/ );
	fails( () => parsePajek( '*Vertices 2\n*Edges\n1 x\n' ), /line 3: expected a vertex number, found "x"/ );
	fails( () => parsePajek( '*Vertices 2\n*Edges\n1\n' ), /line 3: expected "from to \[weight\]"/ );
	fails( () => parsePajek( '*Vertices 9999999\n' ), /too large/, 'too-large' );

} );

test( 'Matrix Market: symmetric, general, pattern, array, complex, rectangular; malformed input', () => {

	const sym = parseMtx( '%%MatrixMarket matrix coordinate real symmetric\n% c\n4 4 4\n2 1 1.5\n3 1 -2\n4 3 0\n4 4 9\n' );
	assert.deepEqual( sym.labels, [ '1', '2', '3', '4' ] );
	assert.deepEqual( Array.from( sym.edges ), [ 0, 1, 0, 2 ] ); // the explicit zero is no edge
	assert.deepEqual( Array.from( sym.weights ), [ 1.5, 2 ] ); // by magnitude
	assert.equal( sym.meta.matrix.zeros, 1 );
	assert.equal( sym.meta.report.dropped.selfLoops, 1 );
	assert.equal( sym.meta.report.directed, false );

	// general = directed: A + Aᵀ, except when the matrix is symmetric in full
	const gen = parseMtx( '%%MatrixMarket matrix coordinate integer general\n3 3 3\n1 2 2\n2 1 3\n2 3 4\n' );
	assert.deepEqual( Array.from( gen.weights ), [ 5, 4 ] );
	assert.equal( gen.meta.report.directed, true );
	const full = parseMtx( '%%MatrixMarket matrix coordinate integer general\n3 3 4\n1 2 2\n2 1 2\n2 3 4\n3 2 4\n' );
	assert.deepEqual( Array.from( full.weights ), [ 2, 4 ] );
	assert.equal( full.meta.report.mirrored, true );

	const pat = parseMtx( '%%MatrixMarket matrix coordinate pattern symmetric\n4 4 3\n2 1\n3 1\n4 3\n' );
	assert.equal( pat.weights, null );
	assert.equal( pat.edges.length / 2, 3 );

	const arr = parseMtx( '%%MatrixMarket matrix array real symmetric\n3 3\n0\n1\n2\n0\n3\n0\n' ); // lower triangle, by columns
	assert.deepEqual( Array.from( arr.edges ), [ 0, 1, 0, 2, 1, 2 ] );
	assert.deepEqual( Array.from( arr.weights ), [ 1, 2, 3 ] );
	const dense = parseMtx( '%%MatrixMarket matrix array integer general\n2 2\n0\n5\n5\n0\n' );
	assert.deepEqual( Array.from( dense.weights ), [ 5 ] );

	const cx = parseMtx( '%%MatrixMarket matrix coordinate complex hermitian\n2 2 1\n2 1 3 4\n' );
	assert.deepEqual( Array.from( cx.weights ), [ 5 ] );

	const rect = parseMtx( '%%MatrixMarket matrix coordinate integer general\n2 3 3\n1 1 1\n2 3 2\n1 3 5\n' );
	assert.deepEqual( rect.labels, [ 'r1', 'r2', 'c1', 'c2', 'c3' ] );
	assert.deepEqual( Array.from( rect.edges ), [ 0, 2, 1, 4, 0, 4 ] );
	assert.deepEqual( rect.attributes.side.categories, [ 'row', 'column' ] );
	assert.equal( rect.meta.matrix.bipartite, true );

	const off = parseMtx( '%%MatrixMarket matrix coordinate pattern general\n2 2 3\n1 2\n1 5\n' );
	assert.equal( off.meta.report.dropped.invalid, 1 );
	assert.ok( off.meta.report.warnings.some( ( w ) => /line 4: entry \(1, 5\) is outside the 2 x 2 matrix/.test( w ) ) );
	assert.ok( off.meta.report.warnings.some( ( w ) => /announces 3 entries, the file has 2/.test( w ) ) );

	fails( () => parseMtx( '%%MatrixMarket matrix coordinate real general\n3 3\n1 2 1\n' ), /line 2: expected the size line "rows columns entries", found "3 3"/, 'parse' );
	fails( () => parseMtx( '%%MatrixMarket matrix coordinate real general\n3 3 1\n1 x 1\n' ), /line 3: expected "row column value", found "1 x 1"/ );
	fails( () => parseMtx( '%%MatrixMarket matrix coordinate real general\n3 3 1\n1 2 abc\n' ), /line 3: "abc" is not a number/ );
	fails( () => parseMtx( '%%MatrixMarket vector coordinate real general\n' ), /object "vector" is not supported/ );
	fails( () => parseMtx( '%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 2 1\n' ), /symmetric matrix must be square/ );
	fails( () => parseMtx( '%%MatrixMarket matrix coordinate real general\n% only comments\n' ), /size line is missing/ );
	fails( () => parseMtx( '%%MatrixMarket matrix coordinate real general\n9999999 9999999 1\n1 2 1\n' ), /too large/, 'too-large' );

} );

test( 'edge lists: weight column by header or by content, time stamps, quoted names, bad weights', () => {

	const byHeader = parseEdgeList( 'from;to;year;Weight\n"Jean Valjean";Fantine;1815;3\nFantine;"Marius ""M"" P.";1832;0.5\n' );
	assert.deepEqual( byHeader.labels, [ 'Jean Valjean', 'Fantine', 'Marius "M" P.' ] );
	assert.deepEqual( Array.from( byHeader.weights ), [ 3, 0.5 ] );
	assert.equal( byHeader.meta.weightColumn, 'Weight' );
	assert.equal( parseEdgeList( 'source,target,year\na,b,1999\nb,c,2001\n' ).weights, null ); // a header without a weight column

	const byContent = parseEdgeList( '# c\n1 2 0.5\n2 3 2\n3 1 1e1\n' );
	assert.deepEqual( Array.from( byContent.weights ), [ 0.5, 2, 10 ] );
	assert.equal( byContent.meta.weightColumn, 3 );
	assert.equal( parseEdgeList( '1 2 1200000000\n2 3 1300000000\n' ).weights, null ); // time stamps
	assert.equal( parseEdgeList( '1 2 5\n2 3\n' ).weights, null ); // not in every row
	assert.equal( parseEdgeList( 'x y # note\ny z\n' ).weights, null );

	const bad = parseEdgeList( 'a,b,weight\nx,y,2\ny,z,heavy\nz,x,\n' );
	assert.deepEqual( Array.from( bad.weights ), [ 2, 1, 1 ] );
	assert.match( bad.meta.report.warnings[ 0 ], /line 3: "heavy" is not a weight/ );
	assert.match( bad.meta.report.warnings[ 1 ], /2 weights are not numbers/ );

	// directed pairs listed both ways: one edge each, not weight 2
	const both = parseEdgeList( '1 2\n2 1\n2 3\n3 2\n' );
	assert.equal( both.weights, null );
	assert.equal( both.meta.report.mirrored, true );
	assert.deepEqual( both.meta.report.dropped, { selfLoops: 0, duplicates: 2, invalid: 0 } );
	fails( () => parseEdgeList( '# nothing\n\n' ), /no edges/, 'empty' );

} );

test( 'JSON shapes: detection and details', () => {

	assert.equal( detectJsonShape( { format: 'wgv-graph', nodeCount: 1, edges: [] } ), 'wgv-graph' );
	assert.equal( detectJsonShape( { nodes: [ { idx: 0, ancIdxs: [ 0 ] } ], links: [], clusters: [ {} ] } ), 'igv' );
	assert.equal( detectJsonShape( { nodes: [ { id: 1 } ], links: [] } ), 'node-link' );
	assert.equal( detectJsonShape( { nodes: [ 'a', 'b' ], edges: [ [ 'a', 'b' ] ] } ), 'node-link' );
	assert.equal( detectJsonShape( { nodes: [ { key: 'a' } ], edges: [] } ), 'graphology' );
	assert.equal( detectJsonShape( { elements: [] } ), 'cytoscape' );
	assert.equal( detectJsonShape( [ { data: { id: 'a' } } ] ), 'cytoscape' );
	assert.equal( detectJsonShape( { nodes: [ { data: { id: 'a' } } ], edges: [] } ), 'cytoscape' );
	assert.equal( detectJsonShape( { a: [ 'b' ], b: [ 'a' ] } ), 'adjacency' );
	assert.equal( detectJsonShape( { nodes: [ { id: 0 } ], adjacency: [ [] ] } ), 'adjacency' );
	assert.equal( detectJsonShape( [ [ 1, 2 ], [ 0 ], [ 0 ] ] ), 'adjacency' );
	assert.equal( detectJsonShape( [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ] ] ), 'edge-array' );
	assert.equal( detectJsonShape( [ { source: 1, target: 2 } ] ), 'edge-array' );
	assert.equal( detectJsonShape( { hello: 1 } ), null );
	assert.equal( detectJsonShape( [ 1, 2, 3 ] ), null );
	assert.equal( detectJsonShape( 'text' ), null );

	// node-link: links by id, by node object, by index; d3 group; unknown end points reported
	const nl = parseGraph( JSON.stringify( {
		nodes: Array.from( { length: 8 }, ( _, i ) => ( { id: `n${i}`, group: i < 4 ? 1 : 2, x: i, y: 0, index: i } ) ),
		links: [ { source: 'n0', target: 'n1', value: 3 }, { source: { id: 'n1' }, target: { id: 'n2' } }, { source: 4, target: 5 }, { source: 'n0', target: 'ghost' } ],
	} ), 'mis.json' ).graph;
	assert.deepEqual( Array.from( nl.edges ), [ 0, 1, 1, 2, 4, 5 ] );
	assert.deepEqual( Array.from( nl.weights ), [ 3, 1, 1 ] );
	assert.deepEqual( Object.keys( nl.attributes ), [ 'group' ] ); // layout fields are not attributes
	assert.deepEqual( nl.attributes.group.categories, [ '1', '2' ] );
	assert.deepEqual( Array.from( nl.planted.levels[ 0 ] ), [ 0, 0, 0, 0, 1, 1, 1, 1 ] );
	assert.equal( nl.meta.report.dropped.invalid, 1 );
	assert.match( nl.meta.report.warnings[ 0 ], /link 3: source or target is not a node/ );

	// Cytoscape compound nodes
	const cy = parseGraph( JSON.stringify( [
		{ data: { id: 'p', name: 'Parent' } }, { data: { id: 'q' } },
		{ data: { id: 'a', parent: 'p' } }, { data: { id: 'b', parent: 'p' } }, { data: { id: 'c', parent: 'q' } }, { data: { id: 'd', parent: 'q' } },
		{ data: { source: 'a', target: 'b' } }, { data: { source: 'b', target: 'c' } }, { data: { source: 'c', target: 'd' } }, { data: { source: 'd', target: 'nope' } }, { nothing: 1 },
	] ), 'net.cyjs' ).graph;
	assert.deepEqual( cy.labels, [ 'a', 'b', 'c', 'd' ] );
	assert.deepEqual( Array.from( cy.planted.levels[ 0 ] ), [ 0, 0, 1, 1 ] );
	assert.deepEqual( cy.attributes.group.categories, [ 'Parent', 'q' ] );
	assert.equal( cy.meta.report.dropped.invalid, 1 );

	// graphology: undirected type, graph name
	const go = parseGraph( JSON.stringify( { attributes: { name: 'G' }, options: { type: 'undirected' }, nodes: [ { key: 1 }, { key: 2 } ], edges: [ { source: 1, target: 2, attributes: { weight: 2 } } ] } ), 'x.json' );
	assert.equal( go.format, 'graphology' );
	assert.equal( go.graph.name, 'G' );
	assert.equal( go.graph.meta.report.directed, false );
	assert.deepEqual( Array.from( go.graph.weights ), [ 2 ] );

	// adjacency lists: each edge listed from both sides is one edge
	const adj = parseGraph( JSON.stringify( { a: [ 'b', 'c' ], b: [ 'a' ], c: [ 'a' ], d: [] } ), 'adj.json' ).graph;
	assert.equal( adj.nodeCount, 4 );
	assert.equal( adj.weights, null );
	assert.deepEqual( Array.from( adj.edges ), [ 0, 1, 0, 2 ] );
	const nx = parseGraph( JSON.stringify( { directed: false, multigraph: false, graph: { name: 'NX' }, nodes: [ { id: 'a', club: 'x' }, { id: 'b', club: 'y' } ], adjacency: [ [ { id: 'b', weight: 2 } ], [ { id: 'a', weight: 2 } ] ] } ), 'nx.json' ).graph;
	assert.equal( nx.name, 'NX' );
	assert.deepEqual( Array.from( nx.weights ), [ 2 ] );
	const idx = parseGraph( '[[1,2],[0],[0]]', 'lists.json' );
	assert.equal( idx.format, 'adjacency' );
	assert.deepEqual( Array.from( idx.graph.edges ), [ 0, 1, 0, 2 ] );
	const ea = parseGraph( '[[5,6],[6,7,2.5]]', 'pairs.json' );
	assert.equal( ea.format, 'edge-array' );
	assert.deepEqual( ea.graph.labels, [ '5', '6', '7' ] );
	assert.deepEqual( Array.from( ea.graph.weights ), [ 1, 2.5 ] );

	fails( () => parseGraph( '{ "hello": 1 }', 'x.json' ), /not a graph in a known shape \(expected wgv-graph.*node-link.*Cytoscape.*graphology.*adjacency/, 'format' );
	fails( () => parseGraph( '{ nope', 'x.json' ), /x: invalid JSON \(.*\)/, 'parse' );
	fails( () => parseGraph( '{"nodes":[],"links":[]}', 'x.json' ), /no nodes/, 'empty' );

} );

test( 'canonical JSON: weights, attributes and saved positions survive; report for file graphs', () => {

	const raw = {
		format: 'wgv-graph', version: 1, name: 'W', nodeCount: 4, labels: [ 'a', 'b', 'c', 'd' ],
		edges: [ 0, 1, 1, 2, 2, 3, 1, 0, 3, 3 ], weights: [ 1, 2, 3, 4, 5 ],
		attributes: { age: { type: 'number', values: [ 1, null, 3, 4 ] }, kind: { type: 'category', values: [ 0, 1, - 1, 7 ], categories: [ 'x', 'y' ] }, bad: { type: 'number', values: [ 1 ] } },
		positions: [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 ],
	};
	const r = parseGraph( JSON.stringify( raw ), 'w.json' );
	assert.equal( r.format, 'wgv-graph' );
	const g = r.graph;
	assert.deepEqual( Array.from( g.edges ), [ 0, 1, 1, 2, 2, 3 ] );
	assert.deepEqual( Array.from( g.weights ), [ 5, 2, 3 ] );
	assert.deepEqual( Object.keys( g.attributes ), [ 'age', 'kind' ] );
	assert.ok( Number.isNaN( g.attributes.age.values[ 1 ] ) );
	assert.deepEqual( Array.from( g.attributes.kind.values ), [ 0, 1, - 1, - 1 ] );
	assert.deepEqual( Array.from( g.positions ), raw.positions );
	assert.deepEqual( g.meta.report.dropped, { selfLoops: 1, duplicates: 1, invalid: 0 } );
	const back = normalizeGraph( JSON.parse( serializeGraph( g ) ) );
	assert.deepEqual( Array.from( back.weights ), [ 5, 2, 3 ] );
	assert.deepEqual( Array.from( back.attributes.kind.values ), [ 0, 1, - 1, - 1 ] );
	assert.ok( Number.isNaN( back.attributes.age.values[ 1 ] ) );
	// existing files still load
	const lesmis = parseGraph( readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ), 'lesmis.json' );
	assert.equal( lesmis.graph.nodeCount, 77 );
	assert.equal( lesmis.graph.hierarchy.levels.length, 2 );
	assert.deepEqual( lesmis.graph.attributes, {} );
	assert.equal( lesmis.graph.weights, null );

} );

// ── round trips ──────────────────────────────────────────────────────────────────────────────

test( 'round trips: export json / graphml / csv and read back', () => {

	const base = parseGraph( SAMPLES.graphml, 'g.graphml' ).graph;
	base.hierarchy = { method: 'test', levels: [ [ 0, 0, 1, 1 ] ] };
	const positions = Float32Array.from( [ 1, 2, 3, 0, - 4.5, 5, 6, 0, 7, 8, 9.25, 0, 10, 11, 12, 0 ] );

	const json = exportGraph( base, 'json', { positions } );
	assert.equal( json.extension, '.json' );
	assert.equal( json.mime, 'application/json' );
	const j = parseGraph( json.text, 'x.json' ).graph;
	assert.deepEqual( j.labels, base.labels );
	assert.deepEqual( Array.from( j.edges ), Array.from( base.edges ) );
	assert.deepEqual( Array.from( j.weights ), Array.from( base.weights ) );
	assert.deepEqual( Array.from( j.attributes.kind.values ), Array.from( base.attributes.kind.values ) );
	assert.deepEqual( j.hierarchy.levels, [ [ 0, 0, 1, 1 ] ] );
	assert.deepEqual( Array.from( j.positions ), [ 1, 2, 3, - 4.5, 5, 6, 7, 8, 9.25, 10, 11, 12 ] );

	const gml = exportGraph( base, 'graphml', { positions } );
	assert.equal( gml.extension, '.graphml' );
	const m = parseGraph( gml.text, 'x.txt' );
	assert.equal( m.format, 'graphml' );
	assert.deepEqual( m.graph.labels, base.labels );
	assert.deepEqual( Array.from( m.graph.edges ), Array.from( base.edges ) );
	assert.deepEqual( Array.from( m.graph.weights ), Array.from( base.weights ) );
	assert.deepEqual( m.graph.attributes.kind.categories, [ 'x', 'y' ] );
	assert.deepEqual( Array.from( m.graph.attributes.score.values.subarray( 0, 3 ) ), [ 1, 2.5, 3 ] );
	assert.ok( Number.isNaN( m.graph.attributes.score.values[ 3 ] ) );
	assert.deepEqual( Array.from( m.graph.attributes.level0.values ), [ 0, 0, 1, 1 ] );
	assert.deepEqual( Array.from( m.graph.attributes.x.values ), [ 1, - 4.5, 7, 10 ] );
	assert.equal( m.graph.meta.report.directed, false );

	const csv = exportGraph( base, 'csv' );
	assert.equal( csv.text.split( '\n' )[ 0 ], 'source,target,weight' );
	const c = parseGraph( csv.text, 'x.csv' ).graph;
	assert.deepEqual( c.labels, base.labels );
	assert.deepEqual( weightMap( c ), weightMap( base ) );

	// names that need quoting, an isolated node, no weights
	const tricky = normalizeGraph( { nodeCount: 4, labels: [ 'a, b', 'say "hi"', ' padded ', 'alone' ], edges: [ 0, 1, 1, 2 ] } );
	const t = parseGraph( exportGraph( tricky, 'csv' ).text, 't.csv' ).graph;
	assert.deepEqual( t.labels, tricky.labels ); // quoted: commas, quotes and surrounding spaces survive
	assert.deepEqual( Array.from( t.edges ), [ 0, 1, 1, 2 ] );
	const x = parseGraph( exportGraph( tricky, 'graphml' ).text, 't.graphml' ).graph;
	assert.deepEqual( x.labels, [ 'a, b', 'say "hi"', 'padded', 'alone' ] ); // XML text is trimmed
	// duplicate labels: indices are the ids
	const dup = normalizeGraph( { nodeCount: 3, labels: [ 'same', 'same', 'other' ], edges: [ 0, 1, 1, 2 ] } );
	assert.equal( exportGraph( dup, 'csv' ).text, 'source,target\n0,1\n1,2\n' );
	assert.throws( () => exportGraph( base, 'pdf' ), /unknown format/ );

	// a generated graph through every writer
	const gen = generate( { model: 'hsbm', params: { n: 300, avgDegree: 8 }, seed: 4 } );
	for ( const format of [ 'json', 'graphml', 'csv' ] ) {

		const back = parseGraph( exportGraph( gen, format ).text, 'gen' ).graph;
		assert.equal( back.nodeCount, gen.nodeCount, format );
		assert.equal( back.edges.length, gen.edges.length, format );
		assert.deepEqual( pairs( back ), pairs( gen ), format );

	}

} );

// ── limits ───────────────────────────────────────────────────────────────────────────────────

test( 'limits: a friendly error with code too-large, for every way in', async () => {

	assert.deepEqual( DEFAULT_LIMITS, { maxNodes: 250000, maxEdges: 1000000 } );
	const limits = { maxNodes: 10, maxEdges: 12 };
	const ring = ( n ) => Array.from( { length: n }, ( _, i ) => `${i} ${( i + 1 ) % n}` ).join( '\n' );
	fails( () => parseGraph( ring( 30 ), 'big.txt', { limits } ), /This graph is too large for the viewer: it has (30|more than 10) nodes\. The limit is 10 nodes and 12 edges\./, 'too-large' );
	// few nodes, many edges
	const dense = [];
	for ( let i = 0; i < 8; i ++ ) for ( let j = i + 1; j < 8; j ++ ) dense.push( `${i} ${j}` );
	fails( () => parseGraph( dense.join( '\n' ), 'dense.txt', { limits } ), /it has 28 edges/, 'too-large' );
	assert.equal( parseGraph( ring( 10 ), 'ok.txt', { limits } ).graph.nodeCount, 10 );
	fails( () => parseGraph( JSON.stringify( { format: 'wgv-graph', nodeCount: 11, edges: [ 0, 1 ] } ), 'c.json', { limits } ), /11 nodes/, 'too-large' );
	fails( () => parseGraph( JSON.stringify( { nodes: Array.from( { length: 40 }, ( _, i ) => ( { id: i } ) ), links: [] } ), 'n.json', { limits } ), /too large/, 'too-large' );
	fails( () => parseGraph( `<graphml><graph>${Array.from( { length: 40 }, ( _, i ) => `<node id="${i}"/>` ).join( '' )}</graph></graphml>`, 'g.graphml', { limits } ), /too large/, 'too-large' );

	// through the worker pipeline: files and generators; the message keeps its code
	await assert.rejects( runRequest( { source: { kind: 'text', text: ring( 30 ), name: 'big.txt' }, limits } ), ( e ) => e.code === 'too-large' && /too large/.test( e.message ) );
	await assert.rejects( runRequest( { source: { kind: 'generate', spec: { model: 'ws', params: { n: 500, k: 4 } } }, limits } ), ( e ) => e.code === 'too-large' && /500 nodes/.test( e.message ) );
	const ok = await runRequest( { source: { kind: 'text', text: ring( 10 ), name: 'ok.txt' }, limits } );
	assert.deepEqual( ok.meta.limits, limits );
	const client = new DataClient( { inline: true, limits } );
	await assert.rejects( client.loadText( ring( 30 ), 'big.txt' ), ( e ) => e.code === 'too-large' );
	const fine = await client.loadText( ring( 30 ), 'big.txt', {}, { limits: { maxNodes: 100, maxEdges: 100 } } ); // per request
	assert.equal( fine.graph.N, 30 );
	client.dispose();

} );

test( 'meta: what was simplified, notes, timings, attributes on the prepared graph', async () => {

	const r = await runRequest( { source: { kind: 'text', text: SAMPLES.graphml, name: 'sample.graphml' } } );
	const m = r.meta;
	assert.equal( m.format, 'graphml' );
	assert.equal( m.simplified, true );
	assert.equal( m.weighted, true );
	assert.equal( m.file.directed, true );
	assert.deepEqual( m.file.dropped, { selfLoops: 1, duplicates: 1, invalid: 0 } );
	assert.ok( m.notes.some( ( s ) => /Directed graph, shown undirected: 1 edges that run both ways or repeat were merged/.test( s ) ) );
	assert.ok( m.notes.some( ( s ) => /1 self loop dropped/.test( s ) ) );
	assert.deepEqual( m.attributes, [ { name: 'kind', type: 'category', categories: 2 }, { name: 'score', type: 'number' } ] );
	for ( const key of [ 'parse', 'prepare', 'total' ] ) assert.ok( m.timings[ key ] >= 0, key );
	assert.equal( m.file.hierarchy, undefined );
	// the prepared graph: attributes in node order, weights in edge order
	const g = r.graph;
	assert.deepEqual( Object.keys( g.attributes ), [ 'kind', 'score' ] );
	assert.ok( g.attributes.kind.values instanceof Int32Array && g.attributes.kind.values.length === g.N );
	assert.ok( g.edgeWeight instanceof Float32Array && g.edgeWeight.length === g.E );
	const byPair = {};
	for ( let e = 0; e < g.E; e ++ ) byPair[ [ g.labels[ g.edges[ 2 * e ] ], g.labels[ g.edges[ 2 * e + 1 ] ] ].sort().join( '~' ) ] = g.edgeWeight[ e ];
	assert.deepEqual( byPair, { 'A~B': 5, 'B~C': 1, 'C~D': 0.5 } );

	const plain = await runRequest( { source: { kind: 'text', text: '1 2\n2 3\n3 1\n', name: 'tri.txt' } } );
	assert.equal( plain.meta.simplified, false );
	assert.deepEqual( plain.meta.notes, [] );
	assert.equal( plain.graph.edgeWeight, null );
	assert.deepEqual( plain.graph.attributes, {} );
	const gen = await runRequest( { source: { kind: 'generate', spec: { model: 'caveman', params: { cliques: 4, cliqueSize: 5 } } } } );
	assert.equal( gen.meta.file, null );
	assert.equal( gen.meta.simplified, false );

} );

test( 'a 12 MB edge list: parsed with progress in a few seconds', () => {

	const n = 200000, m = 900000;
	const parts = [];
	let seed = 12345;
	const rand = () => ( seed = ( Math.imul( seed, 1664525 ) + 1013904223 ) >>> 0 ) / 4294967296;
	for ( let i = 0; i < m; i ++ ) parts.push( `${Math.floor( rand() * n )}\t${Math.floor( rand() * n )}` );
	const text = parts.join( '\n' );
	assert.ok( text.length > 11e6 );
	const fractions = [];
	const t0 = performance.now();
	const { graph } = parseGraph( text, 'big.tsv', { onProgress: ( f ) => fractions.push( f ) } );
	const ms = performance.now() - t0;
	assert.ok( graph.nodeCount > 190000 && graph.nodeCount <= n );
	assert.ok( graph.edges.length / 2 > 890000 );
	assert.ok( fractions.length >= 10, `${fractions.length} progress calls` );
	for ( let k = 1; k < fractions.length; k ++ ) assert.ok( fractions[ k ] > fractions[ k - 1 ] && fractions[ k ] <= 1 );
	assert.ok( ms < 15000, `${ms.toFixed( 0 )} ms` );

} );

test( 'sample files: every file of public/data/samples is the same graph', () => {

	const dir = 'public/data/samples/';
	const index = JSON.parse( readFileSync( dir + 'index.json', 'utf8' ) );
	assert.ok( index.files.length >= 12 );
	const ids = new Set();
	let reference = null, degreesOfAll = null;
	for ( const name of index.files ) {

		const text = readFileSync( dir + name, 'utf8' );
		// under its own name, and misnamed: the content decides
		for ( const as of [ name, 'download', 'graph.txt' ] ) {

			const { graph: g, id, format } = parseGraph( text, as );
			ids.add( id );
			assert.equal( g.nodeCount, index.nodes, `${name} as ${as}: nodes` );
			assert.equal( g.edges.length / 2, index.edges, `${name} as ${as}: edges` );
			if ( format !== 'adjacency' ) {

				assert.ok( g.weights, `${name}: weights` );
				assert.equal( Math.round( g.weights.reduce( ( a, w ) => a + w, 0 ) ), index.weightSum, `${name} as ${as}: weight sum` );

			}

			// the same structure everywhere; the same names where the format has names
			const degree = new Array( g.nodeCount ).fill( 0 );
			for ( const v of g.edges ) degree[ v ] ++;
			const degrees = degree.sort( ( a, b ) => a - b ).join( ',' );
			if ( degreesOfAll === null ) degreesOfAll = degrees;
			assert.equal( degrees, degreesOfAll, `${name} as ${as}: degree sequence` );
			if ( format === 'mtx' ) continue; // a matrix has row numbers only
			assert.equal( g.labels.length, index.nodes );
			assert.ok( g.labels.includes( 'Valjean' ), `${name}: labels are the names` );
			const key = pairs( g ).join( '|' );
			if ( reference === null ) reference = key;
			assert.equal( key, reference, `${name} as ${as}: the same edges` );

		}

	}

	// one file for every entry of the registry
	assert.deepEqual( [ ...ids ].sort(), FORMATS.map( ( f ) => f.id ).sort() );

} );

test( 'hostile files: every outcome is a graph or a GraphError, in bounded time', () => {

	const outcome = ( text, name ) => {

		const t0 = performance.now();
		try {

			const g = parseGraph( text, name ).graph;
			assert.ok( Number.isInteger( g.nodeCount ) && g.edges.length % 2 === 0, `${name}: a canonical graph` );
			for ( let k = 0; k < g.edges.length; k ++ ) assert.ok( g.edges[ k ] >= 0 && g.edges[ k ] < g.nodeCount, `${name}: edge ends are node indices` );
			if ( g.weights ) for ( const w of g.weights ) assert.ok( Number.isFinite( w ) && w >= 0, `${name}: weights are finite` );
			return { graph: g, ms: performance.now() - t0 };

		} catch ( e ) {

			assert.ok( e instanceof GraphError, `${name}: GraphError expected, got ${e && e.constructor && e.constructor.name}: ${e && e.message}` );
			assert.ok( [ 'parse', 'format', 'empty', 'too-large' ].includes( e.code ), `${name}: code ${e.code}` );
			assert.ok( e.message.length > 5 );
			return { error: e, ms: performance.now() - t0 };

		}

	};

	// 1. mutations of the sample files: cut off, pieces removed or doubled, junk inserted
	const dir = 'public/data/samples/';
	const index = JSON.parse( readFileSync( dir + 'index.json', 'utf8' ) );
	let seed = 99;
	const rand = () => ( seed = ( Math.imul( seed, 1664525 ) + 1013904223 ) >>> 0 ) / 4294967296;
	const pick = ( n ) => Math.floor( rand() * n );
	const junk = [ '�', '<', '>', '"', '{', '}', '[', ']', ';', ',', '\n', '&', '&#xZZ;', '<!--', '<![CDATA[', '%%', '*', '/*', '#', '->', '1e999', 'NaN', '9'.repeat( 400 ), 'x'.repeat( 3000 ), '\\' ];
	let graphs = 0, errors = 0;
	for ( const name of index.files ) {

		const text = readFileSync( dir + name, 'utf8' );
		for ( let round = 0; round < 40; round ++ ) {

			let t = text;
			for ( let q = 0, k = 1 + pick( 3 ); q < k; q ++ ) {

				const a = pick( t.length + 1 );
				const kind = pick( 5 );
				if ( kind === 0 ) t = t.slice( 0, a );
				else if ( kind === 1 ) t = t.slice( 0, a ) + t.slice( a + 1 + pick( 200 ) );
				else if ( kind === 2 ) t = t.slice( 0, a ) + junk[ pick( junk.length ) ] + t.slice( a );
				else if ( kind === 3 ) t = t.slice( 0, a + 150 ) + t.slice( a, a + 150 ) + t.slice( a + 150 );
				else t = t.slice( 0, a ) + t.slice( a ).replace( /\d+/, [ '-1', '0', '99999999999', '1.5', 'abc' ][ pick( 5 ) ] );

			}

			const r = outcome( t, round % 2 ? name : 'x.txt' );
			if ( r.graph ) graphs ++; else errors ++;
			assert.ok( r.ms < 2000, `${name} round ${round}: ${r.ms.toFixed( 0 )} ms` );

		}

	}

	assert.ok( graphs > 50 && errors > 50, `${graphs} graphs, ${errors} errors` );

	// 2. shapes that would cost memory, time or the stack
	const rep = ( s, n ) => s.repeat( n );
	const cases = {
		'deep.dot': [ 'graph g { ' + rep( 'subgraph { ', 50000 ) + 'a -- b ' + rep( '} ', 50000 ) + '}', /nested more than 200 deep/ ],
		'open.dot': [ 'graph g { ' + rep( '{ ', 50000 ), /nested more than 200 deep/ ],
		'deep.gml': [ 'graph [ ' + rep( 'node [ ', 50000 ), /never closed/ ],
		'deep.graphml': [ '<graphml><graph>' + rep( '<node id="a"><graph>', 50000 ), /never closed/ ],
		'deep.json': [ rep( '[', 50000 ) + rep( ']', 50000 ), /not a graph in a known shape/ ],
		'huge.net': [ '*Vertices 1000000000\n1 "a"\n2 "b"\n*Edges\n1 2\n', /too large/ ],
		'huge.mtx': [ '%%MatrixMarket matrix coordinate real symmetric\n1000000000 1000000000 2\n1 2 1.0\n999999999 5 2.0\n', /too large/ ],
		'picture.graphml': [ '\x89PNG\r\n\x1a\n\0\0\0\rIHDR' + rep( '\0\x01\x02', 300 ), /not a text file/ ],
		'page.graphml': [ '<!DOCTYPE html><html><body><p>Not found</p></body></html>', /root element is <html>/ ],
		'comment.graphml': [ '<graphml><!--' + rep( 'x', 2e6 ), /comment is not closed/ ],
		'string.dot': [ 'graph g { a [label="' + rep( 'x', 2e6 ), /string is not closed/ ],
	};
	for ( const [ name, [ text, pattern ] ] of Object.entries( cases ) ) {

		const r = outcome( text, name );
		assert.ok( r.error, `${name}: rejected` );
		assert.match( r.error.message, pattern, name );
		assert.ok( r.ms < 5000, `${name}: ${r.ms.toFixed( 0 )} ms` );

	}

	// accepted, cheaply: a file that is one line of 8 MB, 100,000 unknown elements, a 100,000 x
	// 100,000 dense matrix with two entries, references to nodes that do not exist
	const fine = {
		'line.txt': [ rep( 'a b ', 2e6 ), 2, 1 ],
		'unknown.graphml': [ '<graphml><graph>' + rep( '<bogus x="1"/>\n', 100000 ) + '<node id="a"/><node id="b"/><edge source="a" target="b"/></graph></graphml>', 2, 1 ],
		'bomb.graphml': [ '<?xml version="1.0"?><!DOCTYPE z [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;"><!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]><graphml><graph><node id="&c;"/><node id="b"/><edge source="&c;" target="b"/></graph></graphml>', 2, 1 ],
		'chain.dot': [ 'graph g { ' + Array.from( { length: 100000 }, ( _, i ) => 'n' + ( i % 1000 ) ).join( ' -- ' ) + ' }', 1000, 1000 ],
	};
	for ( const [ name, [ text, nodes, edges ] ] of Object.entries( fine ) ) {

		const r = outcome( text, name );
		assert.ok( r.graph, `${name}: read (${r.error && r.error.message})` );
		assert.equal( r.graph.nodeCount, nodes, name );
		assert.equal( r.graph.edges.length / 2, edges, name );
		assert.ok( r.ms < 5000, `${name}: ${r.ms.toFixed( 0 )} ms` );

	}

} );

test( 'files through the worker: a File is read as a stream with progress; gzip is unpacked', async () => {

	const { gzipSync } = await import( 'node:zlib' );
	const text = readFileSync( 'public/data/samples/lesmis.graphml', 'utf8' );
	const stages = [];
	const plain = await runRequest( { source: { kind: 'file', file: new File( [ text ], 'lesmis.graphml' ) } }, ( p ) => stages.push( p.stage ) );
	assert.equal( plain.graph.N, 77 );
	assert.equal( plain.graph.E, 254 );
	assert.equal( plain.meta.format, 'graphml' );
	for ( const stage of [ 'read', 'parse', 'cluster', 'prepare' ] ) assert.ok( stages.includes( stage ), `progress: ${stage}` );
	for ( const key of [ 'read', 'parse', 'cluster', 'prepare', 'total' ] ) assert.ok( plain.timings[ key ] >= 0, `timings.${key}` );

	// the same file gzipped, under a name that says nothing about the format inside
	for ( const name of [ 'lesmis.graphml.gz', 'download.gz' ] ) {

		const packed = await runRequest( { source: { kind: 'file', file: new File( [ gzipSync( text ) ], name ) } } );
		assert.equal( packed.graph.N, 77, name );
		assert.equal( packed.graph.E, 254, name );
		assert.equal( packed.meta.format, 'graphml', name );
		assert.deepEqual( Array.from( packed.graph.edges ), Array.from( plain.graph.edges ), name );

	}

	// gzip is recognised by its content: no .gz in the name; and .gz on a plain file is no harm
	const unnamed = await runRequest( { source: { kind: 'file', file: new File( [ gzipSync( text ) ], 'lesmis.graphml' ) } } );
	assert.equal( unnamed.graph.E, 254 );
	const misnamed = await runRequest( { source: { kind: 'file', file: new File( [ text ], 'plain.graphml.gz' ) } } );
	assert.equal( misnamed.graph.E, 254 );

	// a broken archive is a readable error, not a crash
	const broken = gzipSync( text ).subarray( 0, 400 );
	await assert.rejects( runRequest( { source: { kind: 'file', file: new File( [ broken ], 'cut.graphml.gz' ) } } ), ( e ) => e instanceof GraphError && e.code === 'parse' && /cut\.graphml\.gz cannot be unpacked/.test( e.message ) );
	// an archive that unpacks to more than the viewer reads stops at the limit
	const bomb = gzipSync( Buffer.alloc( 8 * 1024 * 1024, 0x20 ) );
	assert.ok( bomb.length < 64 * 1024 );
	await assert.rejects( runRequest( { source: { kind: 'file', file: new File( [ bomb ], 'bomb.csv.gz' ) }, limits: { maxBytes: 1024 * 1024 } } ), ( e ) => e instanceof GraphError && e.code === 'too-large' && /unpacks to more than 1(\.0)? MB/.test( e.message ) );
	await assert.rejects( runRequest( { source: { kind: 'file', file: new File( [ 'a b\n'.repeat( 400000 ) ], 'long.txt' ) }, limits: { maxBytes: 1024 * 1024 } } ), ( e ) => e instanceof GraphError && e.code === 'too-large' && /long\.txt is 1\.5 MB/.test( e.message ) );

} );

// ── weights at the edges of the number range, defaults, broken stored data ───────────────────

test( 'GraphML: the <default> of the weight key is the weight of an edge without a value', () => {

	const text = `<?xml version="1.0"?><graphml>
<key id="d1" for="edge" attr.name="weight" attr.type="double"><default>2.5</default></key>
<graph edgedefault="undirected"><node id="a"/><node id="b"/><node id="c"/><node id="d"/>
<edge source="a" target="b"/><edge source="b" target="a"><data key="d1">1.5</data></edge>
<edge source="b" target="c"/><edge source="c" target="d"><data key="d1">7</data></edge></graph></graphml>`;
	const g = parseGraph( text, 'w.graphml' ).graph;
	// a-b: the default (2.5) plus the explicit 1.5; b-c: the default; c-d: its own value
	assert.deepEqual( weightMap( g ), { 'a~b': 4, 'b~c': 2.5, 'c~d': 7 } );
	// for="all" counts for edges too; a default on a node key or on another edge key does not
	const all = parseGraph( text.replace( 'for="edge"', 'for="all"' ), 'w.graphml' ).graph;
	assert.deepEqual( weightMap( all ), { 'a~b': 4, 'b~c': 2.5, 'c~d': 7 } );
	// (the two a-b edges are merged: 1 + 1)
	const node = parseGraph( text.replace( 'for="edge"', 'for="node"' ).replaceAll( '<data key="d1">1.5</data>', '' ).replaceAll( '<data key="d1">7</data>', '' ), 'w.graphml' ).graph;
	assert.deepEqual( weightMap( node ), { 'a~b': 2, 'b~c': 1, 'c~d': 1 }, 'a default of a node key is not an edge weight' );
	const other = parseGraph( text.replace( 'attr.name="weight"', 'attr.name="since"' ), 'w.graphml' ).graph;
	assert.deepEqual( weightMap( other ), { 'a~b': 2, 'b~c': 1, 'c~d': 1 }, 'a default of another attribute is not a weight' );
	// no default: 1, as before
	const plain = parseGraph( text.replace( '<default>2.5</default>', '' ), 'w.graphml' ).graph;
	assert.deepEqual( weightMap( plain ), { 'a~b': 2.5, 'b~c': 1, 'c~d': 7 } );
	// an empty or unreadable default is no default
	for ( const bad of [ '<default></default>', '<default>heavy</default>' ] ) assert.deepEqual( weightMap( parseGraph( text.replace( '<default>2.5</default>', bad ), 'w.graphml' ).graph ), { 'a~b': 2.5, 'b~c': 1, 'c~d': 7 }, bad );

} );

test( 'weights beyond a 32-bit float are limited, all-zero weights are dropped; both say so and cluster', async () => {

	assert.equal( MAX_WEIGHT, 3.4028234663852886e38 );
	assert.equal( Math.fround( MAX_WEIGHT ), MAX_WEIGHT );
	// two huge weights on one pair (their sum overflows a float), one beyond the range on its own
	const big = parseGraph( 'a b 3e38\nb a 3e38\nb c 1\nc d 2\nd a 1e39\nd e 5\ne f 5\nf d 5\n', 'big.txt' ).graph;
	assert.ok( big.weights instanceof Float32Array );
	assert.ok( Array.from( big.weights ).every( ( w ) => Number.isFinite( w ) && w > 0 ), 'no Infinity reaches the clustering or the buffers' );
	assert.equal( weightMap( big )[ 'a~b' ], MAX_WEIGHT );
	assert.equal( weightMap( big )[ 'a~d' ], MAX_WEIGHT );
	assert.equal( weightMap( big )[ 'b~c' ], 1 );
	assert.deepEqual( big.meta.report.warnings, [ '2 edge weights are beyond the range of a 32-bit float and were limited to it' ] );
	for ( const method of [ 'leiden', 'louvain' ] ) {

		const r = clusterGraph( big, { method } );
		assert.ok( r.info.modularityPerLevel.every( Number.isFinite ), `${method}: finite modularity` );
		assert.ok( r.levels.every( ( level ) => level.length === 6 ) );

	}

	// every weight 0 (or too small for a float): modularity would divide by a total of 0
	let lines = '';
	for ( let i = 0; i < 40; i ++ ) lines += `n${i} n${( i + 1 ) % 40} 0\nn${i} n${( i + 5 ) % 40} 1e-50\n`;
	const zero = parseGraph( lines, 'zero.txt' ).graph;
	assert.equal( zero.weights, null );
	assert.equal( zero.edges.length, 160 );
	assert.deepEqual( zero.meta.report.warnings, [ 'every edge weight is 0 (or too small to store): the weights are ignored' ] );
	for ( const method of [ 'leiden', 'louvain' ] ) {

		const r = clusterGraph( zero, { method } );
		assert.ok( r.levels.length >= 1, `${method}: the ring has communities` );
		assert.equal( r.info.weighted, false );

	}

	// some zero weights among others are weights like any other
	assert.deepEqual( Array.from( parseGraph( 'a b 0\nb c 2\n', 'some.txt' ).graph.weights ), [ 0, 2 ] );

	// the same rules for every way in: cleanEdges reports, the canonical JSON reader warns
	const c = cleanEdges( 3, [ 0, 1, 1, 0, 1, 2 ], [ 3e38, 3e38, 1e-50 ] );
	assert.deepEqual( c.weightIssues, { limited: 1, allZero: false } );
	assert.equal( c.weights[ 0 ], MAX_WEIGHT );
	assert.deepEqual( cleanEdges( 3, [ 0, 1, 1, 2 ], [ 0, 0 ] ).weightIssues, { limited: 0, allZero: true } );
	assert.equal( cleanEdges( 3, [ 0, 1, 1, 2 ], [ 0, 0 ] ).weights, null );
	assert.deepEqual( cleanEdges( 3, [ 0, 1, 1, 2 ], [ 1, 2 ] ).weightIssues, { limited: 0, allZero: false } );
	assert.deepEqual( weightWarnings( { limited: 1, allZero: false } ), [ '1 edge weight is beyond the range of a 32-bit float and was limited to it' ] );
	assert.deepEqual( weightWarnings( null ), [] );
	assert.deepEqual( weightWarnings( { limited: 0, allZero: true }, 3 ).length, 2 );
	const stored = parseGraph( JSON.stringify( { format: 'wgv-graph', nodeCount: 3, edges: [ 0, 1, 1, 2 ], weights: [ 0, 0 ] } ), 'g.json' ).graph;
	assert.equal( stored.weights, null );
	assert.deepEqual( stored.meta.report.warnings, [ 'every edge weight is 0 (or too small to store): the weights are ignored' ] );
	// a clustering asked for with weights it cannot use (a caller's own arrays): no NaN either
	const raw = { nodeCount: 6, edges: Uint32Array.of( 0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 2, 3 ), weights: Float32Array.of( Infinity, 1, 1, NaN, 1, 1, - 2 ) };
	for ( const method of [ 'leiden', 'louvain' ] ) assert.ok( clusterGraph( raw, { method } ).info.modularityPerLevel.every( Number.isFinite ), method );
	// and through the worker the user is told
	const viaWorker = await runRequest( { source: { kind: 'text', text: lines, name: 'zero.txt' } } );
	assert.ok( viaWorker.meta.notes.some( ( note ) => /every edge weight is 0/i.test( note ) ), viaWorker.meta.notes.join( ' | ' ) );
	assert.equal( viaWorker.graph.edgeWeight, null );

} );

test( 'canonical JSON: errors carry a code, a broken stored hierarchy is dropped, a BOM is skipped', async () => {

	const doc = ( extra ) => JSON.stringify( { format: 'wgv-graph', nodeCount: 4, edges: [ 0, 1, 1, 2, 2, 3 ], ...extra } );
	// a level that is too short, too long or not an array; levels that are not an array
	for ( const levels of [ [ [ 0, 0, 1 ] ], [ [ 0, 0, 1, 1 ], [ 0, 0 ] ], [ [ 0, 0, 1, 1, 1 ] ], [ 'abcd' ], [ null ], [ { length: 4 } ], 'none', 4 ] ) {

		const r = parseGraph( doc( { hierarchy: { levels } } ), 'g.json' );
		assert.equal( r.graph.nodeCount, 4, JSON.stringify( levels ) );
		assert.equal( r.graph.hierarchy, null, JSON.stringify( levels ) );
		assert.deepEqual( r.graph.meta.report.warnings, [ 'the stored hierarchy does not have one entry per node in every level and was ignored' ] );

	}

	// a good one is kept, without a warning
	const good = parseGraph( doc( { hierarchy: { levels: [ [ 0, 0, 1, 1 ] ] } } ), 'g.json' ).graph;
	assert.deepEqual( good.hierarchy.levels, [ [ 0, 0, 1, 1 ] ] );
	assert.deepEqual( good.meta.report.warnings, [] );
	// through the worker: the graph loads and gets a computed hierarchy instead
	const loaded = await runRequest( { source: { kind: 'text', text: doc( { hierarchy: { levels: [ [ 0, 0, 1 ] ] } } ), name: 'g.json' } } );
	assert.equal( loaded.graph.N, 4 );
	assert.ok( loaded.meta.notes.some( ( note ) => /stored hierarchy .* was ignored/.test( note ) ) );

	// errors of the canonical reader are GraphErrors with a code, like those of every other reader
	const fails = ( text, code, pattern ) => assert.throws( () => parseGraph( text, 'g.json' ), ( e ) => e instanceof GraphError && e.code === code && pattern.test( e.message ), `${code}: ${text.slice( 0, 60 )}` );
	fails( JSON.stringify( { format: 'wgv-graph', nodeCount: 0, edges: [] } ), 'empty', /^g: the graph has no nodes$/ );
	fails( JSON.stringify( { format: 'wgv-graph', nodeCount: - 3, edges: [ 0, 1 ] } ), 'empty', /no nodes/ );
	fails( JSON.stringify( { nodes: [ { ancIdxs: [ 0 ] }, null ], links: [ {} ], clusters: [ null ] } ), 'parse', /^g: looks like an ImsvGraphVis \.igv\.json file but cannot be read \(/ );
	fails( '{"format":"wgv-graph","nodeCount":3,', 'parse', /invalid JSON/ );
	fails( '{"some":"thing"}', 'format', /not a graph in a known shape/ );
	await assert.rejects( runRequest( { source: { kind: 'text', text: JSON.stringify( { format: 'wgv-graph', nodeCount: 0, edges: [] } ), name: 'g.json' } } ), ( e ) => e.code === 'empty' );

	// a byte order mark (an editor or an export tool put it there) is not part of the file
	const bom = '﻿';
	assert.equal( parseGraph( bom + doc( {} ), 'g.json' ).graph.nodeCount, 4 );
	assert.equal( parseGraph( bom + doc( {} ), 'g.json' ).format, parseGraph( doc( {} ), 'g.json' ).format );
	assert.deepEqual( parseGraph( bom + 'a b\nb c\n', 'e.txt' ).graph.labels, [ 'a', 'b', 'c' ] );
	assert.equal( parseGraph( bom + '{"nodes":[{"id":"a"},{"id":"b"}],"links":[{"source":"a","target":"b"}]}', 'd3.json' ).graph.nodeCount, 2 );
	assert.equal( parseGraph( bom + '<graphml><graph><node id="a"/><node id="b"/><edge source="a" target="b"/></graph></graphml>', 'x.graphml' ).graph.nodeCount, 2 );
	assert.deepEqual( detectFormat( bom + doc( {} ), '' ), detectFormat( doc( {} ), '' ) );
	const client = new DataClient( { inline: true } );
	const viaClient = await client.loadText( bom + doc( {} ), 'bom.json' );
	assert.equal( viaClient.graph.N, 4 );
	client.dispose();

} );

test( 'malformed XML: a warning per element does not rescan the file (120,000 warnings in well under a second)', () => {

	// the line of a warning is only worked out for the 20 warnings that are kept
	let graphml = '<graphml><graph>';
	for ( let i = 0; i < 60000; i ++ ) graphml += '<node/>\n';
	for ( let i = 0; i < 60000; i ++ ) graphml += '<edge/>\n';
	graphml += '<node id="a"/><node id="b"/><edge source="a" target="b"/></graph></graphml>';
	let gexf = '<gexf><graph><nodes>';
	for ( let i = 0; i < 60000; i ++ ) gexf += '<node/>\n';
	gexf += '<node id="a"/><node id="b"/></nodes><edges>';
	for ( let i = 0; i < 60000; i ++ ) gexf += '<edge/>\n';
	gexf += '<edge source="a" target="b"/></edges></graph></gexf>';
	for ( const [ name, text ] of [ [ 'bad.graphml', graphml ], [ 'bad.gexf', gexf ] ] ) {

		const t0 = performance.now();
		const g = parseGraph( text, name ).graph;
		const ms = performance.now() - t0;
		assert.equal( g.meta.report.warningCount, 120000, name );
		assert.equal( g.meta.report.warnings.length, 20, 'the first 20 are kept' );
		assert.match( g.meta.report.warnings[ 0 ], /^line 1: / );
		assert.match( g.meta.report.warnings[ 19 ], /^line 20: / );
		assert.ok( g.nodeCount >= 2 );
		// (it was 80 s: a scan of the file from its start for every warning)
		assert.ok( ms < 5000, `${name}: ${ms.toFixed( 0 )} ms` );

	}

	// (while here: positions of another graph are not written into an export)
	const small = parseGraph( 'a b\nb c\n', 'e.txt' ).graph;
	assert.ok( /"positions"/.test( exportGraph( small, 'json', { positions: new Float32Array( 12 ) } ).text ) );
	for ( const length of [ 8, 16 ] ) assert.ok( ! /"positions"/.test( exportGraph( small, 'json', { positions: new Float32Array( length ) } ).text ), `${length} floats for 3 nodes` );

	// a warning given as a function is evaluated only when it is kept
	const b = new GraphBuilder();
	let calls = 0;
	for ( let i = 0; i < 100; i ++ ) b.warn( () => `warning ${++ calls}` );
	b.warn( 'a plain string still works' );
	assert.equal( calls, 20 );
	assert.equal( b.warnings.length, 20 );
	assert.equal( b.warningCount, 101 );

} );
