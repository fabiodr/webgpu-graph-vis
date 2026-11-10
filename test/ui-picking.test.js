import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PerspectiveCamera } from 'three/webgpu';
import { Picking, resolveClick } from '../src/interact/Picking.js';

const W = 1000, H = 800;

function camera( z = 100 ) {

	const cam = new PerspectiveCamera( 45, W / H, 0.5, 5000 );
	cam.position.set( 0, 0, z );
	cam.lookAt( 0, 0, 0 );
	cam.updateMatrixWorld();
	cam.updateProjectionMatrix();
	return cam;

}

const pos = ( ...pts ) => Float32Array.from( pts.flatMap( ( p ) => [ p[ 0 ], p[ 1 ], p[ 2 ], 0 ] ) );

test( 'project: origin maps to the screen centre, radius in px follows the focal length', () => {

	const pk = new Picking( { radius: () => 2 } );
	const cam = camera();
	const p = pk.project( pos( [ 0, 0, 0 ] ), cam, W, H, 0 );
	assert.ok( Math.abs( p.x - W / 2 ) < 1e-6 && Math.abs( p.y - H / 2 ) < 1e-6 );
	const focal = ( H / 2 ) / Math.tan( ( 45 * Math.PI ) / 360 );
	assert.ok( Math.abs( p.radius - ( 2 * focal ) / 100 ) < 1e-6 );
	assert.ok( Math.abs( p.depth - 100 ) < 1e-6 );

} );

test( 'pick hits the node under the pointer and respects slop', () => {

	const pk = new Picking( { radius: () => 1 } ); // ~9.7 px at depth 100
	const cam = camera();
	const P = pos( [ 0, 0, 0 ], [ 30, 0, 0 ] );
	const s1 = pk.project( P, cam, W, H, 1 );
	assert.equal( pk.pick( P, 2, cam, W, H, s1.x + 2, s1.y - 2, 8 ), 1 );
	// just outside the disc but inside an 18 px touch slop
	assert.equal( pk.pick( P, 2, cam, W, H, s1.x + 15, s1.y, 8 ), - 1 );
	assert.equal( pk.pick( P, 2, cam, W, H, s1.x + 15, s1.y, 18 ), 1 );
	assert.equal( pk.result.index, 1 );
	assert.ok( pk.result.distance > 14 && pk.result.distance < 16 );
	// empty space
	assert.equal( pk.pick( P, 2, cam, W, H, 50, 50, 18 ), - 1 );
	assert.equal( pk.result.index, - 1 );

} );

test( 'overlapping discs: the node nearest to the camera wins', () => {

	const pk = new Picking( { radius: () => 3 } );
	const cam = camera();
	const P = pos( [ 0, 0, - 40 ], [ 0.5, 0, 20 ], [ 0, 0.4, 0 ] );
	assert.equal( pk.pick( P, 3, cam, W, H, W / 2 + 1, H / 2, 8 ), 1 );

} );

test( 'a disc under the pointer beats a nearer node that is only within slop', () => {

	const pk = new Picking( { radius: ( i ) => ( i === 0 ? 0.2 : 2 ) } );
	const cam = camera();
	// node 0: tiny, close to the camera, 12 px left of the pointer; node 1: big, far, under it
	const P = pos( [ - 1.2, 0, 40 ], [ 0, 0, - 50 ] );
	const s0 = pk.project( P, cam, W, H, 0 );
	assert.ok( s0.radius < 4 );
	assert.equal( pk.pick( P, 2, cam, W, H, W / 2, H / 2, 18 ), 1 );

} );

test( 'near misses: the smallest gap to the disc edge wins', () => {

	const pk = new Picking( { radius: () => 0.1 } );
	const cam = camera();
	const P = pos( [ 1, 0, 0 ], [ - 0.8, 0, 0 ] ); // ~9.7 px and ~7.7 px from the centre
	assert.equal( pk.pick( P, 2, cam, W, H, W / 2, H / 2, 18 ), 1 );

} );

test( 'nodes behind the camera are ignored', () => {

	const pk = new Picking( { radius: () => 50 } );
	const cam = camera();
	const P = pos( [ 0, 0, 150 ] ); // behind (camera at z=100 looking at -z)
	assert.equal( pk.pick( P, 1, cam, W, H, W / 2, H / 2, 18 ), - 1 );
	assert.equal( pk.project( P, cam, W, H, 0 ), null );

} );

test( 'default radius: radii[i] * layoutRadius / (6 cbrt N), scaled by radiusScale', () => {

	const graph = { N: 8, layoutRadius: 120, radii: new Float32Array( 8 ).fill( 1.5 ) };
	const pk = new Picking();
	pk.setGraph( graph );
	assert.ok( Math.abs( pk.radiusOf( 3 ) - ( 1.5 * 120 ) / ( 6 * 2 ) ) < 1e-6 );
	const cam = camera( 300 );
	const P = new Float32Array( 32 );
	const r1 = pk.project( P, cam, W, H, 0 ).radius;
	pk.radiusScale = 2;
	assert.ok( Math.abs( pk.project( P, cam, W, H, 0 ).radius - 2 * r1 ) < 1e-6 );

} );

test( 'pick on 20k nodes takes a few ms at most', () => {

	const N = 20000;
	const P = new Float32Array( 4 * N );
	let s = 12345;
	const rnd = () => ( ( s = ( s * 1103515245 + 12345 ) >>> 0 ) / 4294967296 ) * 2 - 1;
	for ( let i = 0; i < N; i ++ ) {

		P[ 4 * i ] = rnd() * 400; P[ 4 * i + 1 ] = rnd() * 400; P[ 4 * i + 2 ] = rnd() * 400;

	}

	const graph = { N, layoutRadius: 1100, radii: new Float32Array( N ).fill( 1 ) };
	const pk = new Picking();
	pk.setGraph( graph );
	const cam = camera( 1600 );
	for ( let k = 0; k < 20; k ++ ) pk.pick( P, N, cam, W, H, 500, 400, 8 ); // warm up
	const runs = 100;
	const t0 = performance.now();
	let hits = 0;
	for ( let k = 0; k < runs; k ++ ) if ( pk.pick( P, N, cam, W, H, 300 + k * 4, 400, 8 ) >= 0 ) hits ++;
	const ms = ( performance.now() - t0 ) / runs;
	console.log( `  pick 20k nodes: ${ms.toFixed( 3 )} ms/pick, ${hits}/${runs} hits` );
	assert.ok( ms < 4, `pick took ${ms} ms` );
	assert.ok( hits > 0 );

} );

test( 'resolveClick: a double click means the node of the first click', () => {

	const first = { t: 1000, x: 200, y: 300, index: 7 };
	// no click before, or one too long ago, or too far away: a single click on what is under it
	assert.deepEqual( resolveClick( null, { x: 200, y: 300, index: 7 }, 1000 ), { index: 7, double: false } );
	assert.deepEqual( resolveClick( first, { x: 201, y: 301, index: 7 }, 1400 ), { index: 7, double: false } );
	assert.deepEqual( resolveClick( first, { x: 220, y: 300, index: 9 }, 1100 ), { index: 9, double: false } );
	// the same node again
	assert.deepEqual( resolveClick( first, { x: 202, y: 299, index: 7 }, 1150 ), { index: 7, double: true } );
	// the view moved under the pointer: another node, or nothing, lies there now
	assert.deepEqual( resolveClick( first, { x: 202, y: 299, index: 12 }, 1150 ), { index: 7, double: true } );
	assert.deepEqual( resolveClick( first, { x: 202, y: 299, index: - 1 }, 1150 ), { index: 7, double: true } );
	// two clicks on empty space stay a double click on empty space (fit), whatever slid under the second
	assert.deepEqual( resolveClick( { t: 1000, x: 50, y: 50, index: - 1 }, { x: 52, y: 51, index: 3 }, 1200 ), { index: - 1, double: true } );
	// a finger is less exact than a mouse
	assert.equal( resolveClick( first, { x: 220, y: 310, index: 7 }, 1150, { tolerance: 32 } ).double, true );
	assert.equal( resolveClick( first, { x: 220, y: 310, index: 7 }, 1150, { doubleMs: 100, tolerance: 32 } ).double, false );

} );
