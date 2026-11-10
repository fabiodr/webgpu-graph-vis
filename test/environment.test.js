import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	ENV_PRESETS, DEFAULT_KEY_DIRECTION, nebulaPattern, skyRadiance, skyIrradiance, quantizeRadius,
	fitShadowBox, shadowResolve, fogParameters, boundsOfPositions,
} from '../src/render/gfx/Environment.js';
import { NODE_LOOK, nodeVariant, NodeRenderer } from '../src/render/NodeRenderer.js';

const dir = DEFAULT_KEY_DIRECTION.toArray();
const keyOf = ( p ) => ( { dir, radiance: p.key.color.map( ( c ) => c * p.key.box ), size: p.key.size } );
const close = ( a, b, eps, msg ) => assert.ok( Math.abs( a - b ) <= eps, `${msg || ''} ${a} vs ${b}` );

test( 'presets: complete, finite, with node look hints that exist in NODE_LOOK', () => {

	assert.deepEqual( Object.keys( ENV_PRESETS ), [ 'aurora', 'studio', 'void' ] );
	for ( const [ name, p ] of Object.entries( ENV_PRESETS ) ) {

		for ( const group of [ p.backdrop, p.sky, p.key ] ) {

			for ( const v of Object.values( group ) ) assert.ok( [ v ].flat().every( Number.isFinite ), name );

		}

		assert.ok( p.key.light > 0 && p.envIntensity > 0 && p.shadow >= 0 && p.shadow <= 1 && p.fog >= 0 && p.fog <= 1, name );
		for ( const key of Object.keys( p.nodes ) ) assert.ok( key in NODE_LOOK, `${name}: ${key}` );

	}

} );

test( 'nebulaPattern: bounded and deterministic', () => {

	for ( let i = 0; i < 500; i ++ ) {

		const x = Math.sin( i * 12.9 ), y = Math.cos( i * 7.3 ), z = Math.sin( i * 3.1 + 1 );
		const [ n, hue ] = nebulaPattern( x, y, z );
		assert.ok( n >= - 1 && n <= 1 && hue >= 0 && hue <= 1 );
		assert.deepEqual( nebulaPattern( x, y, z ), [ n, hue ] );

	}

} );

test( 'skyRadiance: the soft box sits on the key direction, the fill opposite to it', () => {

	const p = ENV_PRESETS.aurora, key = keyOf( p );
	const toward = skyRadiance( dir, p.sky, key );
	const side = skyRadiance( [ dir[ 2 ], 0, - dir[ 0 ] ].map( ( v ) => v / Math.hypot( dir[ 0 ], dir[ 2 ] ) ), p.sky, key );
	const away = skyRadiance( dir.map( ( v ) => - v ), p.sky, key );
	for ( let c = 0; c < 3; c ++ ) assert.ok( toward[ c ] >= key.radiance[ c ], 'box radiance present' );
	assert.ok( toward[ 0 ] > 20 * side[ 0 ] && toward[ 0 ] > 5 * away[ 0 ] );
	// the cool fill: blue dominates opposite the key
	assert.ok( away[ 2 ] > away[ 0 ] && away[ 2 ] > p.sky.fillIntensity * p.sky.fill[ 2 ] * 0.9 );
	for ( const L of [ toward, side, away ] ) assert.ok( L.every( ( v ) => v >= 0 && Number.isFinite( v ) ) );

} );

test( 'skyIrradiance: a constant sky gives π·L, the key side receives more than the far side', () => {

	const flat = { zenith: [ 0.5, 0.25, 1 ], nadir: [ 0.5, 0.25, 1 ], fill: [ 0, 0, 0 ], fillIntensity: 0, fillSize: [ 0.35, 0.45 ], rim: [ 0, 0, 0 ], rimIntensity: 0, nebulaA: [ 0, 0, 0 ], nebulaB: [ 0, 0, 0 ], nebula: 0 };
	const none = { dir, radiance: [ 0, 0, 0 ], size: [ 0.97, 0.02 ] };
	const E = skyIrradiance( [ 0, 1, 0 ], flat, none );
	close( E[ 0 ], Math.PI * 0.5, 0.01 ); close( E[ 1 ], Math.PI * 0.25, 0.01 ); close( E[ 2 ], Math.PI, 0.02 );

	const p = ENV_PRESETS.studio, key = keyOf( p );
	const lit = skyIrradiance( dir, p.sky, key ), back = skyIrradiance( dir.map( ( v ) => - v ), p.sky, key );
	assert.ok( lit[ 0 ] > back[ 0 ] && lit[ 1 ] > back[ 1 ] );
	// the soft box alone: radiance × solid angle of the cap (cos ≥ size[0]) within a factor
	const cap = 2 * Math.PI * ( 1 - p.key.size[ 0 ] ) * key.radiance[ 0 ];
	const boxOnly = skyIrradiance( dir, { ...flat, zenith: [ 0, 0, 0 ], nadir: [ 0, 0, 0 ] }, key, 20000 );
	assert.ok( boxOnly[ 0 ] > 0.6 * cap && boxOnly[ 0 ] < 1.4 * cap, `${boxOnly[ 0 ]} vs ${cap}` );

} );

test( 'quantizeRadius: powers of 1.25 above r, with hysteresis', () => {

	for ( const r of [ 0.3, 1, 7, 99.9, 100, 212.2, 5000 ] ) {

		const q = quantizeRadius( r );
		assert.ok( q >= r * ( 1 - 1e-9 ) && q < r * 1.2501, `${r} → ${q}` );
		const e = Math.log( q ) / Math.log( 1.25 );
		close( e, Math.round( e ), 1e-6, 'power of 1.25' );

	}

	const q = quantizeRadius( 100 );
	assert.equal( quantizeRadius( 100 * 0.9, q ), q, 'kept while shrinking a little' );
	assert.equal( quantizeRadius( q / 1.49, q ), q );
	assert.ok( quantizeRadius( q / 1.6, q ) < q, 'released when much smaller' );
	assert.ok( quantizeRadius( q * 1.01, q ) > q, 'grows as soon as the bounds leave the box' );
	assert.ok( quantizeRadius( 0 ) > 0 && Number.isFinite( quantizeRadius( 0 ) ) );

} );

test( 'fitShadowBox: box contains the sphere, centre snapped to texels, stable under sub-texel moves', () => {

	const R = quantizeRadius( 212 ), size = 2048;
	const c = [ 13.37, - 4.2, 7.77 ];
	const fit = fitShadowBox( dir, c, R, size );
	close( fit.texel, 2 * R / size, 1e-12 );
	assert.equal( fit.radius, R );
	// the light looks along −dir from 2R away: depth range [ near, far ] covers the sphere
	const toLight = fit.position.map( ( v, k ) => v - fit.target[ k ] );
	close( Math.hypot( ...toLight ), 2 * R, 1e-6 );
	for ( let k = 0; k < 3; k ++ ) close( toLight[ k ] / ( 2 * R ), dir[ k ], 1e-9 );
	assert.ok( fit.near <= 2 * R - 212 && fit.far >= 2 * R + 212 );
	// the snapped centre is within a texel of the true centre across the light, exact along it
	const off = fit.target.map( ( v, k ) => v - c[ k ] );
	const along = off[ 0 ] * dir[ 0 ] + off[ 1 ] * dir[ 1 ] + off[ 2 ] * dir[ 2 ];
	close( along, 0, 1e-6 );
	assert.ok( Math.hypot( ...off ) <= fit.texel * Math.SQRT1_2 + 1e-9 );
	// a move much smaller than a texel (not across a rounding boundary) leaves the box where it is
	let same = 0;
	for ( let i = 0; i < 50; i ++ ) {

		const d = [ Math.sin( i ) * fit.texel * 0.02, Math.cos( i * 2 ) * fit.texel * 0.02, 0 ];
		const f2 = fitShadowBox( dir, c.map( ( v, k ) => v + d[ k ] ), R, size );
		const lateral = f2.target.map( ( v, k ) => v - fit.target[ k ] );
		const al = lateral[ 0 ] * dir[ 0 ] + lateral[ 1 ] * dir[ 1 ] + lateral[ 2 ] * dir[ 2 ];
		const across = Math.hypot( ...lateral.map( ( v, k ) => v - dir[ k ] * al ) );
		// any change across the light is a whole number of texels
		close( across / fit.texel, Math.round( across / fit.texel ), 1e-6 );
		if ( across < 1e-9 ) same ++;

	}

	assert.ok( same >= 40, `${same} of 50 unchanged` );
	// a vertical light direction still has a valid basis
	const top = fitShadowBox( [ 0, 1, 0 ], c, R, size );
	assert.ok( top.target.every( Number.isFinite ) && top.position.every( Number.isFinite ) );

} );

test( 'shadowResolve: no shadows from spheres of a few texels, fade in, hysteresis', () => {

	// well resolved: full strength
	assert.deepEqual( shadowResolve( 13 ), { cast: true, strength: 1 } );
	assert.deepEqual( shadowResolve( 4.5 ), { cast: true, strength: 1 } );
	// unresolved: the light stops casting (no shadow pass, no PCF taps)
	assert.deepEqual( shadowResolve( 1.66 ), { cast: false, strength: 0 } );
	assert.deepEqual( shadowResolve( 0 ), { cast: false, strength: 0 } );
	assert.deepEqual( shadowResolve( NaN ), { cast: false, strength: 0 } );
	// in between the strength fades in monotonically from 0
	let last = - 1;
	for ( let t = 2.5; t <= 4.5; t += 0.125 ) {

		const r = shadowResolve( t );
		assert.equal( r.cast, true );
		assert.ok( r.strength >= last && r.strength >= 0 && r.strength <= 1, `${t}: ${r.strength}` );
		last = r.strength;

	}

	close( shadowResolve( 2.5 ).strength, 0, 1e-12 );
	close( shadowResolve( 3.5 ).strength, 0.5, 1e-12 );
	// hysteresis: a light that stopped casting comes back only above 2.9 texels
	assert.equal( shadowResolve( 2.7, true ).cast, true );
	assert.equal( shadowResolve( 2.7, false ).cast, false );
	assert.equal( shadowResolve( 2.95, false ).cast, true );
	// the measured cases: 20k nodes at 2048² (1.66 texels) off, at 4096² (3.3) weak, 77 nodes full
	const texels = ( N, mapSize, boxOverLayout = 1.25 ) => 0.065 / Math.cbrt( N ) / ( 2 * boxOverLayout / mapSize );
	assert.equal( shadowResolve( texels( 20000, 2048 ) ).cast, false );
	assert.ok( shadowResolve( texels( 20000, 4096 ) ).strength > 0 && shadowResolve( texels( 20000, 4096 ) ).strength < 1 );
	assert.equal( shadowResolve( texels( 77, 2048 ) ).strength, 1 );
	assert.equal( shadowResolve( texels( 500, 2048 ) ).strength, 1 );

} );

test( 'fogParameters: off at strength 0, the far side keeps 1 − 0.62·strength', () => {

	assert.deepEqual( fogParameters( 0, 500, 200 ), { start: 0, density: 0 } );
	assert.deepEqual( fogParameters( 0.5, 500, 0 ), { start: 0, density: 0 } );
	for ( const s of [ 0.2, 0.45, 1 ] ) {

		const { start, density } = fogParameters( s, 500, 200 );
		close( start, 500 - 0.35 * 200, 1e-9 );
		const T = ( d ) => Math.exp( - ( ( Math.max( d - start, 0 ) * density ) ** 2 ) );
		close( T( 700 ), 1 - 0.62 * s, 1e-9, 'far side' );
		assert.equal( T( 300 ), 1, 'near side untouched' );
		assert.ok( T( 500 ) > T( 700 ) );

	}

	// the camera inside the graph: fog starts at the camera
	assert.equal( fogParameters( 0.5, 10, 200 ).start, 0 );
	assert.equal( fogParameters( 7, 500, 200 ).density, fogParameters( 1, 500, 200 ).density, 'strength is clamped' );

} );

test( 'boundsOfPositions: box centre and largest distance, ignores non-finite entries and the tail', () => {

	const p = new Float32Array( [ - 10, 0, 0, 1, 10, 0, 0, 1, 0, 4, 0, 1, NaN, 0, 0, 1, 500, 500, 500, 1 ] );
	const b = boundsOfPositions( p, 4 );
	assert.deepEqual( b.center, [ 0, 2, 0 ] );
	close( b.radius, Math.hypot( 10, 2 ), 1e-6 );
	assert.deepEqual( boundsOfPositions( p, 0 ), { center: [ 0, 0, 0 ], radius: 0 } );
	assert.ok( boundsOfPositions( p, 5 ).radius > 300 );

} );

test( 'nodeVariant: one key per pipeline configuration', () => {

	const base = { tier: 'high', samples: 4, velocity: false, lit: true };
	assert.equal( nodeVariant( base ), 'std|a2c|still' );
	assert.equal( nodeVariant( { ...base, samples: 1, velocity: true } ), 'std|cut|mv' );
	assert.equal( nodeVariant( { ...base, tier: 'ultra', samples: 1, velocity: true } ), 'phys|cut|mv' );
	assert.equal( nodeVariant( { ...base, tier: 'ultra', lit: false } ), 'unlit|a2c|still' );
	assert.equal( nodeVariant( { ...base, tier: 'medium' } ), 'std|a2c|still' );

} );

test( 'NodeRenderer: size contract and look parameters without a graph', () => {

	const nodes = new NodeRenderer( { size: 2, look: { roughness: 0.5, bogus: 3 } } );
	assert.equal( nodes.isNodeRenderer, true );
	assert.equal( nodes.size, 2 );
	assert.equal( nodes.look.roughness, 0.5 );
	assert.ok( ! ( 'bogus' in nodes.look ) );
	assert.deepEqual( Object.keys( nodes.look ), Object.keys( NODE_LOOK ) );
	nodes.setLook( { emissive: 0.4, rim: 'x' } );
	assert.equal( nodes.look.emissive, 0.4 );
	assert.equal( nodes.look.rim, NODE_LOOK.rim, 'non-numeric values are ignored' );
	// radius = baseSize · radius factor · size
	nodes.baseSize = 3;
	nodes.graph = { radii: Float32Array.from( [ 1, 1.5 ] ) };
	assert.equal( nodes.worldRadius( 1 ), 3 * 1.5 * 2 );
	// the halo ring lies outside the sphere and grows with the selection
	assert.ok( nodes.haloRadius( 10, 0 ) > 10 && nodes.haloRadius( 10, 1 ) > nodes.haloRadius( 10, 0 ) );
	nodes.minPixelRadius = - 1;
	assert.equal( nodes.minPixelRadius, 0 );
	nodes.dispose();

} );
