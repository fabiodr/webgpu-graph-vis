// The highlight model: how selection, community focus, k-hop neighbourhood and shortest path
// merge into nodeStateTarget ( x selected, y neighbour, z dimmed, w hovered ), which layer wins,
// what Esc removes next, and that nothing is left behind.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { HighlightModel, selectionWeights, clusterWeights, clusterMembers, clusterLevel, LAYERS, PEEL_ORDER, PATH_END } from '../src/explore/highlight.js';
import { Tools } from '../src/analysis/Tools.js';
import { parseGraphJson } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';

const load = ( name ) => prepareGraph( parseGraphJson( JSON.parse( readFileSync( `public/data/${name}.json`, 'utf8' ) ), name ) );

/** A path 0–1–2–3–4–5 with a triangle 2–6–7 hanging off it, as the parts of a PreparedGraph the model reads. */
function smallGraph() {

	const edges = [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ], [ 4, 5 ], [ 2, 6 ], [ 6, 7 ], [ 7, 2 ] ];
	const N = 8;
	const adj = Array.from( { length: N }, () => [] );
	for ( const [ a, b ] of edges ) {

		adj[ a ].push( b );
		adj[ b ].push( a );

	}

	const degree = Uint32Array.from( adj, ( a ) => a.length );
	const adjStart = new Uint32Array( N + 1 );
	for ( let i = 0; i < N; i ++ ) adjStart[ i + 1 ] = adjStart[ i ] + degree[ i ];
	return { N, E: edges.length, degree, adjStart, adjList: Uint32Array.from( adj.flat() ) };

}

const column = ( t, k, N ) => Array.from( { length: N }, ( _, i ) => Math.round( t[ 4 * i + k ] * 100 ) / 100 );
const weights = ( N, entries ) => {

	const w = new Float32Array( N );
	for ( const [ i, v ] of Object.entries( entries ) ) w[ i ] = v;
	return w;

};

test( 'no layer: nothing is marked and nothing is dimmed', () => {

	const m = new HighlightModel();
	m.reset( 8 );
	assert.equal( m.active, false );
	assert.equal( m.top(), null );
	assert.deepEqual( m.visible(), [] );
	assert.deepEqual( Array.from( m.compose() ), new Array( 32 ).fill( 0 ) );
	assert.deepEqual( LAYERS, [ 'selection', 'cluster', 'khop', 'path' ] );
	assert.deepEqual( PEEL_ORDER, [ 'path', 'khop', 'cluster', 'selection' ] );

} );

test( 'selection: the node, its neighbours, the rest dimmed', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.selected = 2;
	m.set( 'selection', selectionWeights( g, 2 ) );
	const t = m.compose();
	assert.deepEqual( column( t, 0, g.N ), [ 0, 0, 1, 0, 0, 0, 0, 0 ] );
	assert.deepEqual( column( t, 1, g.N ), [ 0, 1, 0, 1, 0, 0, 1, 1 ] );
	assert.deepEqual( column( t, 2, g.N ), [ 1, 0, 0, 0, 1, 1, 0, 0 ] );
	assert.deepEqual( column( t, 3, g.N ), new Array( 8 ).fill( 0 ) );
	assert.equal( m.top(), 'selection' );
	assert.deepEqual( m.visible(), [ 'selection' ] );

} );

test( 'hover and search preview only write w, with or without a layer', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.hover = 5;
	m.preview = 0;
	let t = m.compose();
	assert.deepEqual( column( t, 3, g.N ), [ 1, 0, 0, 0, 0, 1, 0, 0 ] );
	assert.deepEqual( column( t, 2, g.N ), new Array( 8 ).fill( 0 ), 'a hover alone dims nothing' );
	assert.equal( m.active, false );
	m.selected = 2;
	m.set( 'selection', selectionWeights( g, 2 ) );
	t = m.compose();
	assert.deepEqual( column( t, 3, g.N ), [ 1, 0, 0, 0, 0, 1, 0, 0 ] );
	assert.equal( t[ 4 * 5 + 2 ], 1, 'the hovered node keeps its dimming target: the shader lifts it' );
	assert.equal( t[ 4 * 5 ], 0 );

} );

test( 'community focus and selection add up', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.set( 'cluster', weights( g.N, { 3: 0.6, 4: 0.6, 5: 0.6 } ), { cluster: 9 } );
	let t = m.compose();
	assert.deepEqual( column( t, 0, g.N ), new Array( 8 ).fill( 0 ), 'a focused community selects nothing' );
	assert.deepEqual( column( t, 1, g.N ), [ 0, 0, 0, 0.5, 0.5, 0.5, 0, 0 ] );
	assert.deepEqual( column( t, 2, g.N ), [ 1, 1, 1, 0, 0, 0, 1, 1 ] );
	assert.equal( m.get( 'cluster' ).meta.cluster, 9 );

	m.selected = 2; // outside the community, with one neighbour ( 3 ) inside
	m.set( 'selection', selectionWeights( g, 2 ) );
	t = m.compose();
	assert.deepEqual( column( t, 0, g.N ), [ 0, 0, 1, 0, 0, 0, 0, 0 ] );
	assert.deepEqual( column( t, 1, g.N ), [ 0, 1, 0, 1, 0.5, 0.5, 1, 1 ], 'neighbours stand out, also inside the community' );
	assert.deepEqual( column( t, 2, g.N ), [ 1, 0, 0, 0, 0, 0, 0, 0 ] );
	assert.deepEqual( m.visible(), [ 'selection', 'cluster' ] );
	assert.equal( m.top(), 'cluster', 'Esc leaves the community first, then the selection' );

} );

test( 'a neighbourhood replaces the picture below it and fades with the distance', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.selected = 0;
	m.set( 'selection', selectionWeights( g, 0 ) );
	m.set( 'cluster', weights( g.N, { 5: 0.6 } ) );
	// rings as Tools.highlightSet( 'khop' ) gives them: 1 at the source, 0.85 … 0.35 by distance
	m.set( 'khop', weights( g.N, { 0: 1, 1: 0.85, 2: 0.65, 3: 0.35, 6: 0.35, 7: 0.35 } ), { k: 3 } );
	const t = m.compose();
	assert.deepEqual( column( t, 0, g.N ), [ 1, 0, 0, 0, 0, 0, 0, 0 ] );
	assert.deepEqual( column( t, 1, g.N ), [ 0, 1, 0.78, 0.45, 0, 0, 0.45, 0.45 ], 'first ring 1, last ring 0.45' );
	assert.deepEqual( column( t, 2, g.N ), [ 0, 0, 0.14, 0.35, 1, 1, 0.35, 0.35 ], 'the far rings recede a little, the rest is dimmed' );
	assert.equal( t[ 4 * 5 + 1 ], 0, 'the community below does not show through' );
	assert.deepEqual( m.visible(), [ 'khop' ] );
	assert.equal( m.top(), 'khop' );

} );

test( 'a path is read alone: ends marked without their stars, interior as neighbours', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.selected = 0;
	m.set( 'selection', selectionWeights( g, 0 ) );
	m.set( 'khop', weights( g.N, { 0: 1, 1: 0.85, 2: 0.35 } ) );
	m.set( 'path', weights( g.N, { 0: 1, 1: 0.85, 2: 0.85, 3: 0.85, 4: 1 } ), { source: 0, target: 4 } );
	const t = m.compose();
	assert.equal( PATH_END, 0.4 );
	assert.deepEqual( column( t, 0, g.N ), [ 0.4, 0, 0, 0, 0.4, 0, 0, 0 ], 'x stays under the 0.5 at which the renderers light every edge of a node' );
	assert.deepEqual( column( t, 1, g.N ), [ 1, 1, 1, 1, 1, 0, 0, 0 ] );
	assert.deepEqual( column( t, 2, g.N ), [ 0, 0, 0, 0, 0, 1, 1, 1 ] );
	assert.deepEqual( m.visible(), [ 'path' ], 'path wins over the neighbourhood' );

	// with every shortest path shown, the nodes of the other routes are fainter
	m.set( 'path', weights( g.N, { 0: 1, 1: 0.85, 6: 0.5, 4: 1 } ) );
	assert.deepEqual( column( m.compose(), 1, g.N ), [ 1, 1, 0, 0, 1, 0, 0.65, 0 ] );

} );

test( 'the selected node is never dimmed, whatever is on top', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.selected = 7;
	m.set( 'selection', selectionWeights( g, 7 ) );
	m.set( 'path', weights( g.N, { 0: 1, 1: 0.85, 2: 1 } ) ); // a path that does not touch node 7
	let t = m.compose();
	assert.equal( t[ 4 * 7 + 2 ], 0 );
	assert.equal( Math.round( t[ 4 * 7 ] * 100 ) / 100, 0.4, 'marked like a path end: its star would bury the path' );
	m.set( 'path', null );
	m.set( 'khop', weights( g.N, { 0: 1, 1: 0.85 } ) );
	t = m.compose();
	assert.equal( t[ 4 * 7 ], 1 );
	assert.equal( t[ 4 * 7 + 2 ], 0 );
	m.set( 'khop', null );
	m.set( 'selection', null );
	m.set( 'cluster', weights( g.N, { 0: 0.6 } ) );
	t = m.compose();
	assert.equal( t[ 4 * 7 ], 1, 'also while only a community is focused' );

} );

test( 'Esc order: path, neighbourhood, community, selection; the lower layers come back', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.selected = 2;
	m.set( 'selection', selectionWeights( g, 2 ) );
	const selectionOnly = Array.from( m.compose() );
	m.set( 'cluster', weights( g.N, { 4: 0.6, 5: 0.6 } ) );
	const withCluster = Array.from( m.compose() );
	m.set( 'khop', weights( g.N, { 2: 1, 1: 0.85, 3: 0.85 } ) );
	m.set( 'path', weights( g.N, { 2: 1, 3: 0.85, 4: 1 } ) );
	const peeled = [];
	const pictures = [];
	for ( let name = m.top(); name; name = m.top() ) {

		peeled.push( name );
		m.set( name, null );
		if ( name === 'selection' ) m.selected = - 1;
		pictures.push( Array.from( m.compose() ) );

	}

	assert.deepEqual( peeled, [ 'path', 'khop', 'cluster', 'selection' ] );
	assert.deepEqual( pictures[ 1 ], withCluster, 'under the neighbourhood: selection and community, untouched' );
	assert.deepEqual( pictures[ 2 ], selectionOnly );
	assert.deepEqual( pictures[ 3 ], new Array( 32 ).fill( 0 ), 'nothing sticks' );
	assert.equal( m.active, false );

} );

test( 'reset (a new graph) forgets every layer, hover and selection', () => {

	const g = smallGraph(), m = new HighlightModel();
	m.reset( g.N );
	m.selected = 1;
	m.hover = 2;
	m.preview = 3;
	for ( const name of LAYERS ) m.set( name, weights( g.N, { 1: 1 } ) );
	m.reset( 3 );
	assert.equal( m.N, 3 );
	assert.equal( m.active, false );
	assert.deepEqual( [ m.selected, m.hover, m.preview ], [ - 1, - 1, - 1 ] );
	assert.deepEqual( Array.from( m.compose() ), new Array( 12 ).fill( 0 ) );

} );

test( 'compose writes into the given array and checks its input', () => {

	const m = new HighlightModel();
	m.reset( 4 );
	const out = new Float32Array( 16 ).fill( 9 );
	assert.equal( m.compose( out ), out );
	assert.deepEqual( Array.from( out ), new Array( 16 ).fill( 0 ) );
	assert.throws( () => m.set( 'glow', new Float32Array( 4 ) ), RangeError );
	assert.throws( () => m.set( 'path', new Float32Array( 2 ) ), RangeError );
	m.hover = 99; // out of range indices are ignored
	m.selected = 99;
	assert.deepEqual( Array.from( m.compose() ), new Array( 16 ).fill( 0 ) );

} );

test( 'the sets of the analysis tools merge by the same rules', () => {

	const g = load( 'imsv/lesmis' );
	const app = { events: new EventTarget() };
	const tools = new Tools();
	tools.init( app );
	const fx = new Float32Array( 4 * g.N );
	tools.setGraph( { N: g.N, graph: g, nodeFx: { value: { array: fx, needsUpdate: false } }, uniforms: { rippleFront: { value: 0 }, rippleWidth: { value: 1 }, rippleGain: { value: 0 } }, setFx: ( a ) => fx.set( a ) }, g );
	const valjean = g.labels.indexOf( 'Valjean' ), myriel = g.labels.indexOf( 'Myriel' ), marius = g.labels.indexOf( 'Marius' );
	const m = new HighlightModel();
	m.reset( g.N );
	m.selected = myriel;
	m.set( 'selection', selectionWeights( g, myriel ) );

	tools.khop( myriel, 2 );
	m.set( 'khop', tools.highlightSet( 'khop' ) );
	let t = m.compose();
	assert.equal( t[ 4 * myriel ], 1 );
	assert.equal( t[ 4 * valjean + 1 ], 1, 'Valjean is a direct neighbour of Myriel: first ring' );
	assert.ok( t[ 4 * marius + 1 ] > 0.4 && t[ 4 * marius + 1 ] < 1, 'Marius is two hops away: a fainter ring' );
	const inside = column( t, 2, g.N ).filter( ( z ) => z < 1 ).length;
	assert.equal( inside, tools.state.khop.count, 'exactly the neighbourhood is in the picture' );

	const path = tools.path( myriel, marius );
	assert.deepEqual( Array.from( path.nodes ), [ myriel, valjean, marius ] );
	assert.ok( tools.highlightSet( 'khop' ), 'the tools keep the neighbourhood beside the path' );
	m.set( 'path', tools.highlightSet( 'path' ) ); // both layers set: the path alone decides the picture
	t = m.compose();
	assert.deepEqual( [ myriel, valjean, marius ].map( ( i ) => Math.round( t[ 4 * i ] * 10 ) / 10 ), [ 0.4, 0, 0.4 ] );
	assert.deepEqual( [ myriel, valjean, marius ].map( ( i ) => t[ 4 * i + 1 ] ), [ 1, 1, 1 ] );
	assert.equal( column( t, 2, g.N ).filter( ( z ) => z === 0 ).length, 3, 'three nodes in the picture, all others dimmed' );

	tools.clear();
	assert.equal( tools.highlightSet( 'path' ), null );

} );

test( 'cluster helpers: members, level and weights of a community', () => {

	const g = load( 'imsv/lesmis' );
	const valjean = g.labels.indexOf( 'Valjean' );
	const fine = g.anc[ valjean * g.D ], coarse = g.anc[ valjean * g.D + g.D - 1 ];
	assert.equal( clusterLevel( g, fine ), 0 );
	assert.equal( clusterLevel( g, coarse ), g.D - 1 );
	assert.equal( clusterLevel( g, g.root ), - 1 );
	assert.equal( clusterLevel( g, - 1 ), - 1 );
	const members = clusterMembers( g, coarse );
	assert.equal( members.length, g.clusterMass[ coarse ] );
	assert.ok( Array.from( members ).includes( valjean ) );
	assert.ok( Array.from( members ).every( ( i ) => g.anc[ i * g.D + g.D - 1 ] === coarse ) );
	assert.equal( clusterMembers( g, g.root ).length, 0, 'the root is not a community' );
	const w = clusterWeights( g, coarse );
	assert.equal( Array.from( w.subarray( 0, g.N ) ).filter( ( v ) => v > 0 ).length, members.length );
	assert.ok( Math.abs( w[ valjean ] - 0.6 ) < 1e-6 );
	const sel = selectionWeights( g, valjean );
	assert.equal( sel[ valjean ], 1 );
	assert.equal( Array.from( sel.subarray( 0, g.N ) ).filter( ( v ) => v > 0 && v < 1 ).length, g.degree[ valjean ] );
	assert.equal( selectionWeights( g, - 1 ).some( ( v ) => v > 0 ), false );

} );
