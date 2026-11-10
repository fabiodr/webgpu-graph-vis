// Depth motes (src/render/gfx/Environment.js) and the node point response
// (src/render/NodeRenderer.js): the pure parts and the settings. The shaders are checked in
// dev/atmosphere.html.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ENV_PRESETS, ENV_MOTES, MOTE_VOLUME, MOTE_OCTAVES, MOTE_SIZE, moteResponse, Environment } from '../src/render/gfx/Environment.js';
import { NODE_LOOK, POINT_GLOW_TIER, POINT_SIGMA, pointSprite, NodeRenderer } from '../src/render/NodeRenderer.js';

const close = ( a, b, eps, msg ) => assert.ok( Math.abs( a - b ) <= eps, `${msg || ''} ${a} vs ${b}` );

test( 'moteResponse: the light a mote sends does not depend on how large it is drawn', () => {

	// radiance × drawn area = the true disc's area, whatever the minimum size and the blur
	for ( const truePx of [ 0.05, 0.3, 0.9, 1.1, 2.5, 8 ] ) {

		for ( const coc of [ 0, 0.5, 3, 12 ] ) {

			const { radius, gain } = moteResponse( truePx, 1.1, coc );
			assert.ok( radius >= 1.1 - 1e-12 && radius >= truePx - 1e-12 && radius >= coc - 1e-12 );
			assert.ok( gain > 0 && gain <= 1 );
			close( gain * radius * radius, truePx * truePx, 1e-9, `energy at ${truePx}px, coc ${coc}` );

		}

	}

	// a mote larger than the minimum and in focus is drawn as it is
	assert.deepEqual( moteResponse( 3, 1.1, 0 ), { radius: 3, gain: 1 } );
	// continuous through the minimum size: nothing pops when a mote crosses it
	let last = moteResponse( 0.5, 1.1 ).gain;
	for ( let r = 0.5; r <= 2; r += 0.01 ) {

		const g = moteResponse( r, 1.1 ).gain;
		assert.ok( g >= last - 1e-12 && g - last < 0.03 );
		last = g;

	}

	assert.equal( last, 1 );
	// far away the light goes out smoothly instead of the sprite shrinking below a pixel
	assert.ok( moteResponse( 0.05, 1.1 ).gain < 0.003 );
	// blur spreads the light: the radiance falls with the square of the circle of confusion
	close( moteResponse( 2, 1.1, 8 ).gain / moteResponse( 2, 1.1, 16 ).gain, ( 4 + 256 ) / ( 4 + 64 ), 1e-9 );

} );

test( 'mote budget and presets: nothing in the low tier, subtle in aurora, off in studio', () => {

	assert.equal( ENV_MOTES.low, 0 );
	assert.ok( ENV_MOTES.medium > 0 && ENV_MOTES.medium <= ENV_MOTES.high && ENV_MOTES.high <= ENV_MOTES.ultra );
	assert.ok( ENV_MOTES.ultra <= 2000, 'a sparse field' );
	assert.ok( ENV_PRESETS.aurora.dust > 0 && ENV_PRESETS.aurora.dust <= 0.5 );
	assert.equal( ENV_PRESETS.studio.dust, 0 );
	assert.ok( ENV_PRESETS.void.dust >= 0 && ENV_PRESETS.void.dust < ENV_PRESETS.aurora.dust );
	assert.ok( MOTE_VOLUME > 1 && MOTE_OCTAVES >= 2 && MOTE_SIZE > 0 );
	// a mote half a cell away (where the lattice fades out) is about a pixel in a 800 px high view:
	// world radius × px per unit, with px per unit = cot( 22.5° ) · 400 / distance
	const px = MOTE_SIZE * ( 1 / Math.tan( Math.PI / 8 ) ) * 400 / 0.5;
	assert.ok( px > 0.6 && px < 1.6, `${px} px` );

} );

test( 'Environment.dust: clamped, follows the preset, read back', () => {

	const env = new Environment();
	assert.equal( env.dust, ENV_PRESETS.aurora.dust );
	env.dust = 3; assert.equal( env.dust, 1 );
	env.dust = - 1; assert.equal( env.dust, 0 );
	env.dust = 'x'; assert.equal( env.dust, 0 );
	env.dust = 0.25; assert.equal( env.dust, 0.25 );
	assert.equal( env.stats.motes, 0, 'nothing is drawn before a graph and a renderer exist' );
	assert.equal( env.moteMesh, null );
	const studio = new Environment( { preset: 'studio' } );
	assert.equal( studio.dust, 0 );

} );

test( 'node point response: look parameters exist, the low tier has no point glow', () => {

	assert.equal( NODE_LOOK.point, 1, 'energy conserving by default' );
	assert.ok( NODE_LOOK.pointGlow > 0 && NODE_LOOK.pointGlow <= 1 );
	assert.equal( POINT_GLOW_TIER.low, 0 );
	assert.ok( POINT_GLOW_TIER.medium > 0 && POINT_GLOW_TIER.medium <= POINT_GLOW_TIER.high );
	const nodes = new NodeRenderer();
	assert.equal( nodes.look.point, 1 );
	nodes.setLook( { point: 0, pointGlow: 0.2, nonsense: 5 } );
	assert.equal( nodes.look.point, 0 );
	assert.equal( nodes.look.pointGlow, 0.2 );
	assert.ok( ! ( 'nonsense' in nodes.look ) );
	// the fade the vertex stage computes: f · mix( 1, f, point ) for a node at a fraction f of the minimum radius
	const fade = ( f, point ) => f * ( 1 + ( f - 1 ) * point );
	close( fade( 0.5, 1 ), 0.25, 1e-12, 'area' );
	close( fade( 0.5, 0 ), 0.5, 1e-12, 'radius' );
	assert.equal( fade( 1, 1 ), 1 );

} );

test( 'point sprite: the Gaussian holds the true disc of the node at every size and defocus', () => {

	const minPx = 2;
	for ( const truePx of [ 0.05, 0.3, 0.8, 1.5 ] ) {

		for ( const coc of [ 0, 3, 12 ] ) {

			const s = pointSprite( truePx, minPx, coc );
			// integral of cover · exp( − r² / 2σ² ) over the plane
			const area = 2 * Math.PI * s.sigma * s.sigma * s.cover;
			assert.ok( Math.abs( area - Math.PI * truePx * truePx ) < 1e-9, `${truePx} px, coc ${coc}` );
			// numeric check of the same integral on rings
			let sum = 0;
			const dr = s.sigma / 200;
			for ( let r = dr / 2; r < s.sigma * 8; r += dr ) sum += s.cover * Math.exp( - r * r / ( 2 * s.sigma * s.sigma ) ) * 2 * Math.PI * r * dr;
			assert.ok( Math.abs( sum - Math.PI * truePx * truePx ) < 1e-3 * Math.PI * truePx * truePx + 1e-9 );

		}

	}

	// sharp sprite: σ in minimum radii, wider when out of focus, never negative
	assert.ok( Math.abs( pointSprite( 1, 2 ).sigma - POINT_SIGMA * 2 ) < 1e-12 );
	assert.ok( pointSprite( 1, 2, 8 ).sigma > pointSprite( 1, 2, 0 ).sigma );
	assert.equal( pointSprite( 0, 2 ).cover, 0 );
	// monotonic in the size, continuous to the disc: at the minimum size the centre is covered ( ≥ 1 )
	let last = - 1;
	for ( let t = 0; t <= 2; t += 0.1 ) {

		const c = pointSprite( t, 2 ).cover;
		assert.ok( c >= last );
		last = c;

	}

	assert.ok( pointSprite( 2, 2 ).cover >= 1 );
	// look.point = 0: linear fade ( more light for small nodes than their disc )
	assert.ok( pointSprite( 0.5, 2, 0, 0 ).cover > pointSprite( 0.5, 2, 0, 1 ).cover );

} );
