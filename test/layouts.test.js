// Tests for the static layouts and the stress seed (src/graph/layouts.js), the layout metrics
// (src/graph/layoutMetrics.js) and the static layout requests of the data worker / DataClient.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';
import { runRequest, runLayout, handleMessage } from '../src/data/worker.js';
import { DataClient } from '../src/data/client.js';
import { stressSeed, sgd2Layout, radialLayout, autoPivots, pickLevel, fitToRadius } from '../src/graph/layouts.js';
import { prepareEval, layoutMetrics, bfsDistances, extent, countOverlaps } from '../src/graph/layoutMetrics.js';

const loadPipeline = async ( file, cluster = { mode: 'auto' } ) => ( await runRequest( { source: { kind: 'text', text: readFileSync( `public/data/${file}`, 'utf8' ), name: file.split( '/' ).pop() }, cluster } ) ).graph;
const raw = ( nodeCount, edges ) => prepareGraph( normalizeGraph( { nodeCount, edges: edges.flat() } ) );

function grid( w, h ) {

	const edges = [];
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

		if ( x + 1 < w ) edges.push( [ y * w + x, y * w + x + 1 ] );
		if ( y + 1 < h ) edges.push( [ y * w + x, ( y + 1 ) * w + x ] );

	}

	return raw( w * h, edges );

}

const ring = ( n ) => raw( n, Array.from( { length: n }, ( _, i ) => [ i, ( i + 1 ) % n ] ) );
const dist3 = ( p, i, j ) => Math.hypot( p[ 4 * i ] - p[ 4 * j ], p[ 4 * i + 1 ] - p[ 4 * j + 1 ], p[ 4 * i + 2 ] - p[ 4 * j + 2 ] );

test( 'stressSeed: deterministic, reference extent, connected clusters end up close', async () => {

	const g = await loadPipeline( 'generated/caveman-24.json' ); // 24 cliques of 8 joined in a ring
	assert.ok( g.D >= 1 );
	const a = stressSeed( g ), b = stressSeed( g );
	assert.ok( a instanceof Float32Array && a.length === 4 * g.N );
	assert.deepEqual( a, b );
	assert.deepEqual( a, g.initPos, 'prepareGraph seeds with it' );
	assert.ok( a.every( Number.isFinite ) );
	assert.ok( Math.abs( extent( g, a ).ext - 1 ) < 0.1, `extent ${extent( g, a ).ext}` );
	const other = stressSeed( g, { seed: 2 } );
	assert.notDeepEqual( a, other );
	// centroids of the solved level: clusters joined by an edge are closer than the average pair
	const lvl = pickLevel( g );
	const lo = g.levelOffset[ lvl ], n = g.levelOffset[ lvl + 1 ] - lo;
	const c = new Float64Array( 3 * n ), m = new Float64Array( n );
	for ( let i = 0; i < g.N; i ++ ) {

		const k = g.anc[ i * g.D + lvl ] - lo;
		for ( let q = 0; q < 3; q ++ ) c[ 3 * k + q ] += a[ 4 * i + q ];
		m[ k ] ++;

	}

	for ( let k = 0; k < n; k ++ ) for ( let q = 0; q < 3; q ++ ) c[ 3 * k + q ] /= m[ k ];
	const cd = ( x, y ) => Math.hypot( c[ 3 * x ] - c[ 3 * y ], c[ 3 * x + 1 ] - c[ 3 * y + 1 ], c[ 3 * x + 2 ] - c[ 3 * y + 2 ] );
	const joined = new Set();
	for ( let e = 0; e < g.E; e ++ ) {

		const x = g.anc[ g.edges[ 2 * e ] * g.D + lvl ] - lo, y = g.anc[ g.edges[ 2 * e + 1 ] * g.D + lvl ] - lo;
		if ( x !== y ) joined.add( Math.min( x, y ) * n + Math.max( x, y ) );

	}

	let near = 0, all = 0, pairs = 0;
	for ( let x = 0; x < n; x ++ ) for ( let y = x + 1; y < n; y ++ ) {

		all += cd( x, y ); pairs ++;

	}

	for ( const key of joined ) near += cd( Math.floor( key / n ), key % n );
	assert.ok( joined.size > 0 && near / joined.size < 0.6 * ( all / pairs ), `joined ${near / joined.size} vs all ${all / pairs}` );

} );

test( 'stressSeed: no usable hierarchy level -> null, prepareGraph keeps the previous seed', () => {

	const g = prepareGraph( { ...normalizeGraph( { nodeCount: 40, edges: Array.from( { length: 39 }, ( _, i ) => [ i, i + 1 ] ).flat() } ), hierarchy: null } );
	assert.equal( g.D, 0 );
	assert.equal( stressSeed( g ), null );
	assert.equal( pickLevel( g ), - 1 );
	assert.equal( g.initPos.length, 4 * g.N );
	assert.ok( g.initPos.every( Number.isFinite ) );
	let spread = 0;
	for ( let i = 0; i < g.N; i ++ ) spread = Math.max( spread, Math.hypot( g.initPos[ 4 * i ], g.initPos[ 4 * i + 1 ], g.initPos[ 4 * i + 2 ] ) );
	assert.ok( spread > 0 );

} );

test( 'sgd2Layout: exact stress recovers a grid and a ring; deterministic; 2D option', () => {

	const g = grid( 8, 8 );
	const a = sgd2Layout( g, { pivots: 0 } );
	assert.equal( a.info.exact, true );
	assert.equal( a.info.pivots, g.N );
	assert.equal( a.positions.length, 4 * g.N );
	assert.ok( a.positions.every( Number.isFinite ) );
	assert.deepEqual( sgd2Layout( g, { pivots: 0 } ).positions, a.positions );
	const ev = prepareEval( g );
	const m = layoutMetrics( ev, a.positions, 4 );
	assert.ok( m.stress < 0.02, `grid stress ${m.stress}` );
	assert.ok( Math.abs( m.ext - 1 ) < 1e-3, 'fitted to the reference extent' );
	// all grid edges have nearly the same length
	assert.ok( m.edgeCV < 0.15, `edge length CV ${m.edgeCV}` );
	// a custom radius scales the result
	const half = sgd2Layout( g, { pivots: 0, radius: g.layoutRadius / 2 } );
	assert.ok( Math.abs( extent( g, half.positions ).ext - 0.5 ) < 1e-3 );

	const r = ring( 60 );
	const flat = sgd2Layout( r, { pivots: 0, dim: 2 } );
	for ( let i = 0; i < r.N; i ++ ) assert.equal( flat.positions[ 4 * i + 2 ], 0 );
	const mr = layoutMetrics( prepareEval( r ), flat.positions, 4 );
	assert.ok( mr.stress < 0.06, `ring stress ${mr.stress}` ); // a circle: chords are shorter than arcs
	// opposite nodes of the ring are far apart, neighbours close
	assert.ok( dist3( flat.positions, 0, 30 ) > 8 * dist3( flat.positions, 0, 1 ) );

} );

test( 'sgd2Layout: sparse pivots approximate the exact layout', async () => {

	assert.equal( autoPivots( 2000 ), 200 );
	assert.equal( autoPivots( 2001 ), 50 );
	const g = grid( 20, 20 );
	const ev = prepareEval( g );
	const exact = layoutMetrics( ev, sgd2Layout( g, { pivots: 0 } ).positions, 4 ).stress;
	const sparse = sgd2Layout( g, { pivots: 40 } );
	assert.equal( sparse.info.exact, false );
	assert.equal( sparse.info.pivots, 40 );
	assert.ok( sparse.info.terms < g.N * 45 + g.E );
	const s = layoutMetrics( ev, sparse.positions, 4 ).stress;
	assert.ok( s < exact + 0.02, `sparse ${s} vs exact ${exact}` );
	// beats the force layout's seed on a mesh by a wide margin
	const lat = await loadPipeline( 'generated/caveman-24.json' );
	const st = sgd2Layout( lat );
	assert.ok( layoutMetrics( prepareEval( lat ), st.positions, 4 ).stress < 0.1 );

} );

test( 'radialLayout: shells by BFS depth, root in the centre, unreachable nodes outside', () => {

	// a tree plus a ring plus two isolated nodes
	const edges = [ [ 0, 1 ], [ 0, 2 ], [ 0, 3 ], [ 1, 4 ], [ 1, 5 ], [ 2, 6 ], [ 6, 7 ], [ 7, 8 ], [ 3, 9 ], [ 9, 10 ], [ 10, 3 ] ];
	const g = raw( 13, edges );
	const r = radialLayout( g, 0 );
	const depth = bfsDistances( g, 0 );
	assert.equal( r.info.root, 0 );
	assert.equal( r.info.reached, 11 );
	const rad = ( i ) => Math.hypot( r.positions[ 4 * i ], r.positions[ 4 * i + 1 ], r.positions[ 4 * i + 2 ] );
	assert.ok( rad( 0 ) < 1e-9, 'root at the origin' );
	const shell = new Map();
	for ( let i = 0; i < g.N; i ++ ) {

		const d = depth[ i ] === 65535 ? - 1 : depth[ i ];
		assert.equal( r.depth[ i ], d );
		if ( ! shell.has( d ) ) shell.set( d, rad( i ) );
		assert.ok( Math.abs( rad( i ) - shell.get( d ) ) < 1e-3 * g.layoutRadius, 'same depth, same radius' );

	}

	for ( let d = 1; d <= r.info.maxDepth; d ++ ) assert.ok( shell.get( d ) > shell.get( d - 1 ) );
	assert.ok( shell.get( - 1 ) > shell.get( r.info.maxDepth ), 'unreachable nodes on an outer shell' );
	assert.ok( Math.abs( shell.get( - 1 ) - g.layoutRadius ) < 1e-3 * g.layoutRadius, 'outermost shell = layoutRadius' );
	assert.deepEqual( radialLayout( g, 0 ).positions, r.positions );
	// children stay near their parent: a child is closer to its own parent than to another branch
	assert.ok( dist3( r.positions, 4, 1 ) < dist3( r.positions, 4, 2 ) );
	// another root, in the plane
	const p = radialLayout( g, 7, { dim: 2 } );
	for ( let i = 0; i < g.N; i ++ ) assert.equal( p.positions[ 4 * i + 2 ], 0 );
	assert.ok( Math.hypot( p.positions[ 28 ], p.positions[ 29 ] ) < 1e-9 );
	// natural size
	const nat = radialLayout( g, 0, { radius: null } );
	assert.ok( Math.abs( nat.info.scale - 1 ) < 1e-12 );

} );

test( 'radialLayout: real graph, every root works without recursion limits', async () => {

	const g = await loadPipeline( 'imsv/lesmis.json' );
	for ( const root of [ 0, 11, g.N - 1 ] ) {

		const r = radialLayout( g, root );
		assert.ok( r.positions.every( Number.isFinite ) );
		assert.equal( r.info.reached, g.N );
		const depth = bfsDistances( g, root );
		for ( let i = 0; i < g.N; i ++ ) assert.equal( r.depth[ i ], depth[ i ] );
		let outer = 0;
		for ( let i = 0; i < g.N; i ++ ) outer = Math.max( outer, Math.hypot( r.positions[ 4 * i ], r.positions[ 4 * i + 1 ], r.positions[ 4 * i + 2 ] ) );
		assert.ok( Math.abs( outer - g.layoutRadius ) < 1e-3 * g.layoutRadius );

	}

	// a long path: depth 2999 (no recursion anywhere)
	const path = raw( 3000, Array.from( { length: 2999 }, ( _, i ) => [ i, i + 1 ] ) );
	const r = radialLayout( path, 0 );
	assert.equal( r.info.maxDepth, 2999 );
	assert.ok( r.positions.every( Number.isFinite ) );

} );

test( 'countOverlaps: every overlapping pair once, in every octant', () => {

	// pseudo-random points in a cube around the origin (negative coordinates included)
	let seed = 12345;
	const rnd = () => ( ( seed = ( Math.imul( seed, 1664525 ) + 1013904223 ) >>> 0 ) / 4294967296 );
	const N = 1500;
	for ( const offset of [ 0, - 5000, 5000 ] ) {

		const pos = new Float32Array( 4 * N ), radii = new Float32Array( N );
		for ( let i = 0; i < N; i ++ ) {

			for ( let a = 0; a < 3; a ++ ) pos[ 4 * i + a ] = ( rnd() - 0.5 ) * 200 + offset;
			radii[ i ] = 2 + 3 * rnd();

		}

		for ( const scale of [ 1, 0.5 ] ) {

			let brute = 0;
			for ( let i = 0; i < N; i ++ ) for ( let j = i + 1; j < N; j ++ ) if ( dist3( pos, i, j ) < ( radii[ i ] + radii[ j ] ) / scale ) brute ++;
			assert.ok( brute > 20 );
			assert.equal( countOverlaps( pos, N, radii, scale, 4 ), brute, `offset ${offset}, scale ${scale}` );

		}

	}

	assert.equal( countOverlaps( new Float32Array( 0 ), 0, new Float32Array( 0 ) ), 0 );

} );

test( 'fitToRadius: centres and scales to the rms radius of a uniform ball', () => {

	const pos = new Float64Array( [ 10, 0, 0, 14, 0, 0, 12, 3, 0, 12, - 3, 0 ] );
	fitToRadius( pos, 4, 100 );
	let cx = 0, q = 0;
	for ( let i = 0; i < 4; i ++ ) cx += pos[ 3 * i ];
	assert.ok( Math.abs( cx ) < 1e-9 );
	for ( let i = 0; i < 4; i ++ ) q += pos[ 3 * i ] ** 2 + pos[ 3 * i + 1 ] ** 2 + pos[ 3 * i + 2 ] ** 2;
	assert.ok( Math.abs( Math.sqrt( q / 4 ) - Math.sqrt( 0.6 ) * 100 ) < 1e-6 );

} );

test( 'worker: static layout requests', async () => {

	const g = await loadPipeline( 'imsv/karate.json' );
	const stress = runLayout( { kind: 'stress', graph: g, options: { pivots: 0 } } );
	assert.equal( stress.kind, 'stress' );
	assert.equal( stress.positions.length, 4 * g.N );
	assert.deepEqual( stress.positions, sgd2Layout( g, { pivots: 0 } ).positions );
	const radial = runLayout( { kind: 'radial', graph: g, options: { root: 33 } } );
	assert.deepEqual( radial.positions, radialLayout( g, 33 ).positions );
	assert.equal( radial.depth.length, g.N );
	assert.throws( () => runLayout( { kind: 'spiral', graph: g } ), /unknown kind/ );
	assert.throws( () => runLayout( { kind: 'stress', graph: { N: 3 } } ), /graph needs/ );
	// through the message protocol
	const posted = [];
	await handleMessage( { id: 7, type: 'layout', kind: 'stress', graph: g, options: {} }, ( data, transfer ) => posted.push( { data, transfer } ) );
	assert.equal( posted.length, 1 );
	assert.equal( posted[ 0 ].data.type, 'layout-result' );
	assert.equal( posted[ 0 ].data.id, 7 );
	assert.equal( posted[ 0 ].transfer[ 0 ], posted[ 0 ].data.positions.buffer );
	await handleMessage( { id: 8, type: 'layout', kind: 'nope', graph: g }, ( data ) => posted.push( { data } ) );
	assert.equal( posted[ 1 ].data.type, 'error' );

} );

test( 'DataClient.layout: result, default graph, supersede, errors, cancelled by a load', async () => {

	const client = new DataClient( { inline: true } );
	await assert.rejects( client.layout( 'stress' ), /no graph/ );
	const text = readFileSync( 'public/data/imsv/karate.json', 'utf8' );
	const loaded = await client.loadText( text, 'karate.json' );
	const g = loaded.graph;
	const r = await client.layout( 'stress', { pivots: 0 } ); // the last loaded graph
	assert.equal( r.cancelled, false );
	assert.equal( r.kind, 'stress' );
	assert.deepEqual( r.positions, sgd2Layout( g, { pivots: 0 } ).positions );
	assert.ok( r.info.exact && r.timings.compute >= 0 && r.timings.roundTrip >= 0 );
	// the main thread keeps its graph (the request copies, it does not transfer)
	assert.equal( g.adjList.length, 2 * g.E );
	const radial = await client.layout( 'radial', { graph: g, root: 0 } );
	assert.equal( radial.depth[ 0 ], 0 );
	// a newer layout request supersedes the older one
	const first = client.layout( 'stress', { graph: g } );
	const second = client.layout( 'stress', { graph: g, seed: 2 } );
	assert.deepEqual( await first, { cancelled: true } );
	assert.equal( ( await second ).cancelled, false );
	// loads and layouts queue behind each other in the worker; neither cancels the other
	const pendingLoad = client.loadText( text, 'karate.json' );
	const lay = client.layout( 'stress', { graph: g } );
	assert.equal( ( await pendingLoad ).cancelled, false );
	assert.equal( ( await lay ).cancelled, false );
	const lay2 = client.layout( 'stress', { graph: g } );
	const load2 = client.loadText( text, 'karate.json' );
	assert.equal( ( await lay2 ).cancelled, false );
	assert.equal( ( await load2 ).cancelled, false );
	// … but a layout dies with the worker: a load superseding a load terminates it
	const loadA = client.loadText( text, 'karate.json' );
	const lay3 = client.layout( 'stress', { graph: g } );
	const loadB = client.loadText( text, 'karate.json' );
	assert.deepEqual( await loadA, { cancelled: true } );
	assert.deepEqual( await lay3, { cancelled: true } );
	assert.equal( ( await loadB ).cancelled, false );
	await assert.rejects( client.layout( 'spiral', { graph: g } ), /unknown kind/ );
	const lay4 = client.layout( 'stress', { graph: g } );
	client.dispose();
	assert.deepEqual( await lay4, { cancelled: true } );

} );
