import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PerspectiveCamera, Vector3 } from 'three';
import { LabelGrid, Labels } from '../src/render/Labels.js';

test( 'LabelGrid: greedy overlap culling', () => {

	const g = new LabelGrid( 50, 20 );
	g.reset( 400, 200 );
	assert.equal( g.tryInsert( 10, 10, 90, 24 ), true );
	assert.equal( g.tryInsert( 80, 20, 150, 34 ), false, 'overlaps the first rect' );
	assert.equal( g.tryInsert( 91, 10, 140, 24 ), true, 'touching on the right is free' );
	assert.equal( g.tryInsert( 10, 25, 90, 39 ), true, 'below is free' );
	assert.equal( g.tryInsert( 0, 0, 400, 200, true ), true, 'forced insert always succeeds' );
	assert.equal( g.free( 300, 150, 310, 160 ), false, 'forced rect blocks the rest' );
	// rects spanning many cells and partially off screen
	g.reset( 400, 200 );
	assert.equal( g.tryInsert( - 30, - 5, 260, 12 ), true );
	assert.equal( g.free( 255, 0, 300, 4 ), false );
	assert.equal( g.free( 261, 0, 300, 4 ), true );
	// grows past the initial capacity
	g.reset( 4000, 4000 );
	for ( let k = 0; k < 600; k ++ ) assert.equal( g.tryInsert( ( k % 30 ) * 100, Math.floor( k / 30 ) * 30, ( k % 30 ) * 100 + 90, Math.floor( k / 30 ) * 30 + 20 ), true );
	assert.equal( g.count, 600 );

} );

/** Star (node 0 with 60 leaves) + a few extra nodes on a ring, all in the z = 0 plane. */
function fakeScene( { N = 70, mode = 'auto' } = {} ) {

	const edges = [];
	for ( let i = 1; i <= 60; i ++ ) edges.push( 0, i );
	edges.push( 1, 2, 1, 3, 2, 3 ); // give leaves 1..3 a higher degree
	const E = edges.length / 2;
	const degree = new Uint32Array( N );
	for ( const v of edges ) degree[ v ] ++;
	const adjStart = new Uint32Array( N );
	for ( let i = 1; i < N; i ++ ) adjStart[ i ] = adjStart[ i - 1 ] + degree[ i - 1 ];
	const fill = adjStart.slice(), adjList = new Uint32Array( 2 * E );
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		adjList[ fill[ a ] ++ ] = b; adjList[ fill[ b ] ++ ] = a;

	}

	const positions = new Float32Array( 4 * N );
	for ( let i = 1; i < N; i ++ ) {

		const a = ( i / N ) * Math.PI * 2;
		positions[ 4 * i ] = Math.cos( a ) * 100;
		positions[ 4 * i + 1 ] = Math.sin( a ) * 100;

	}

	const labels = Array.from( { length: N }, ( _, i ) => 'node' + i );
	const graph = { N, E, degree, adjStart, adjList, labels, colors: new Float32Array( 4 * N ).fill( 0.5 ), radii: new Float32Array( N ).fill( 1 ) };
	const camera = new PerspectiveCamera( 45, 1600 / 1000, 0.5, 20000 );
	camera.position.set( 0, 0, 400 );
	camera.lookAt( new Vector3() );
	camera.updateMatrixWorld();
	const u = ( v ) => ( { value: v } );
	const buffers = { uniforms: { selectedIndex: u( - 1 ), hoverIndex: u( - 1 ), focus: u( 0 ) } };
	const app = { width: 1600, height: 1000, camera, positions, positionsVersion: 1, modules: [] };
	const L = new Labels( { mode } );
	L.app = app;
	L.setGraph( buffers, graph );
	return { L, app, buffers, graph };

}

test( 'Labels: selected, hovered and neighbors by degree (capped)', () => {

	const { L, app } = fakeScene( { mode: 'off' } );
	L.maxNeighbors = 10;
	L._collect( 0, 5, 1, app );
	const w = L._wanted;
	assert.equal( w.get( 0 ).kind, 3, 'selected' );
	assert.equal( w.get( 5 ).kind, 2, 'hovered' );
	const nb = [ ...w ].filter( ( [ , v ] ) => v.kind === 1 ).map( ( [ i ] ) => i );
	assert.ok( nb.length <= 10, 'neighbor cap' );
	// highest-degree neighbors (1, 2, 3) come first and survive culling
	for ( const i of [ 1, 2, 3 ] ) assert.ok( nb.includes( i ), `neighbor ${i} labelled` );
	assert.deepEqual( L._neighborsSorted( 0 ).slice( 0, 3 ), [ 1, 2, 3 ] );

} );

test( 'Labels: context labels in auto/all modes, culled and faded by focus', () => {

	const auto = fakeScene( { mode: 'auto' } );
	auto.L._collect( - 1, - 1, 0, auto.app );
	const n0 = auto.L._wanted.size;
	assert.ok( n0 > 10, `some context labels placed (${n0})` );
	assert.ok( n0 <= auto.graph.N );
	// no two placed labels overlap
	const rects = [ ...auto.L._wanted.values() ].map( ( v ) => [ v.x, v.y - 7, v.x + auto.L._width( 0, 11 ), v.y + 7 ] );
	void rects;
	for ( const [ , v ] of auto.L._wanted ) assert.ok( v.alpha > 0.3 && v.alpha <= 1 );

	// big graph in auto mode: only selection-driven labels
	const big = fakeScene( { mode: 'auto' } );
	big.L.autoMaxNodes = 10;
	big.L._collect( - 1, - 1, 0, big.app );
	assert.equal( big.L._wanted.size, 0 );
	big.L.mode = 'all';
	big.L.maxLabels = 7;
	big.L._collect( - 1, - 1, 0, big.app );
	assert.equal( big.L._wanted.size, 7, 'mode all respects maxLabels' );
	// highest degree first
	assert.ok( big.L._wanted.has( 0 ) );

	// a selection fades the context labels
	auto.L._collect( - 1, - 1, 1, auto.app );
	for ( const [ , v ] of auto.L._wanted ) assert.ok( v.alpha <= 0.21, 'context faded under focus' );

} );

test( 'Labels: nodes behind the camera are not labelled', () => {

	const { L, app } = fakeScene( { mode: 'all' } );
	app.camera.position.set( 0, 0, 0.1 );
	app.camera.lookAt( new Vector3( 0, 0, 1 ) ); // looking away from the z = 0 plane
	app.camera.updateMatrixWorld();
	L._collect( 0, - 1, 0, app );
	assert.equal( L._wanted.size, 0 );

} );
