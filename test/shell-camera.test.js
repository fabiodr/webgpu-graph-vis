// Pure helpers of the camera rig: damping, pose interpolation, flight arc, fitting, wheel zoom.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	smoothDamp, easeInOutCubic, shortestAngle, orbitFromPose, orbitToPosition, lerpPose, arcDistance, fitDistanceFor,
	wheelZoomFactor, driftRates, DRIFT, CameraRig, enclosingSphere,
} from '../src/interact/CameraRig.js';

const close = ( a, b, eps = 1e-6, msg = '' ) => assert.ok( Math.abs( a - b ) <= eps, `${msg} ${a} vs ${b}` );

test( 'smoothDamp: converges without overshoot at 60 fps', () => {

	let x = 0, v = 0, max = 0;
	const dt = 1 / 60, time = 0.1;
	const trace = [];
	for ( let i = 0; i < 120; i ++ ) {

		[ x, v ] = smoothDamp( x, v, 1, time, dt );
		max = Math.max( max, x );
		trace.push( x );

	}

	assert.ok( max <= 1 + 1e-12, 'no overshoot' );
	close( x, 1, 1e-6, 'arrived after 2 s' );
	for ( let i = 1; i < trace.length; i ++ ) assert.ok( trace[ i ] >= trace[ i - 1 ] - 1e-12, 'monotone' );
	// "reaches it in about smoothTime": most of the way after one smoothTime, nearly there after two
	assert.ok( trace[ 5 ] > 0.55 && trace[ 5 ] < 0.75, `after one smoothTime: ${trace[ 5 ]}` );
	assert.ok( trace[ 11 ] > 0.9, `after two: ${trace[ 11 ]}` );

} );

test( 'smoothDamp: stable for huge frame times and frame-rate independent', () => {

	// 2 fps (software rendering): no explosion, no overshoot
	let x = 0, v = 0;
	for ( let i = 0; i < 6; i ++ ) {

		[ x, v ] = smoothDamp( x, v, 10, 0.085, 0.5 );
		assert.ok( x <= 10 && x >= 0 && Number.isFinite( v ) );

	}

	close( x, 10, 1e-3 );
	// the same half second in 30 or 240 steps ends at nearly the same place
	const run = ( steps ) => {

		let a = 0, b = 0;
		for ( let i = 0; i < steps; i ++ ) [ a, b ] = smoothDamp( a, b, 1, 0.3, 0.25 / steps );
		return a;

	};

	close( run( 8 ), run( 60 ), 0.01, 'step-size independence' );

} );

test( 'smoothDamp: an initial velocity toward the target is absorbed, away from it turned around', () => {

	let x = 0, v = 50; // fast toward 1
	let max = 0;
	for ( let i = 0; i < 200; i ++ ) {

		[ x, v ] = smoothDamp( x, v, 1, 0.2, 1 / 60 );
		max = Math.max( max, x );

	}

	assert.ok( max <= 1 + 1e-9, `clamped at the target (${max})` );
	close( x, 1, 1e-6 );
	x = 0; v = - 3;
	let min = 0;
	for ( let i = 0; i < 300; i ++ ) {

		[ x, v ] = smoothDamp( x, v, 1, 0.2, 1 / 60 );
		min = Math.min( min, x );

	}

	assert.ok( min < 0, 'first keeps going' );
	close( x, 1, 1e-6 );
	// degenerate arguments
	assert.deepEqual( smoothDamp( 3, 2, 9, 0.1, 0 ), [ 3, 2 ] );
	assert.deepEqual( smoothDamp( 3, 2, 9, 0, 0.016 ), [ 9, 0 ] );

} );

test( 'easing and angles', () => {

	assert.equal( easeInOutCubic( 0 ), 0 );
	assert.equal( easeInOutCubic( 1 ), 1 );
	close( easeInOutCubic( 0.5 ), 0.5 );
	close( easeInOutCubic( 0.25 ) + easeInOutCubic( 0.75 ), 1, 1e-12, 'symmetric' );
	close( shortestAngle( 3 * Math.PI ), Math.PI );
	close( shortestAngle( - 3 * Math.PI ), Math.PI );
	close( shortestAngle( 0.2 ), 0.2 );
	close( shortestAngle( 2 * Math.PI - 0.2 ), - 0.2 );
	close( shortestAngle( - 2 * Math.PI + 0.2 ), 0.2 );
	for ( let a = - 20; a < 20; a += 0.37 ) {

		const s = shortestAngle( a );
		assert.ok( s > - Math.PI - 1e-12 && s <= Math.PI + 1e-12 );
		close( Math.sin( s ), Math.sin( a ), 1e-9 );
		close( Math.cos( s ), Math.cos( a ), 1e-9 );

	}

} );

test( 'orbit parameters round trip', () => {

	const target = [ 3, - 2, 5 ];
	for ( const [ yaw, pitch, d ] of [ [ 0, 0, 400 ], [ 1.2, 0.5, 10 ], [ - 2.9, - 1.1, 0.02 ], [ 3.1, 1.5, 7 ] ] ) {

		const pos = orbitToPosition( target, yaw, pitch, d );
		const o = orbitFromPose( pos, target );
		close( o.distance, d, d * 1e-9 );
		close( shortestAngle( o.yaw - yaw ), 0, 1e-9 );
		close( o.pitch, pitch, 1e-9 );

	}

	// yaw 0, pitch 0 is the +Z axis: the default camera of the app
	assert.deepEqual( orbitToPosition( [ 0, 0, 0 ], 0, 0, 400 ), [ 0, 0, 400 ] );
	// straight above: clamped just short of the pole, still a valid distance
	const top = orbitFromPose( [ 0, 10, 0 ], [ 0, 0, 0 ] );
	assert.ok( top.pitch < Math.PI / 2 && top.pitch > 1.5 );
	close( top.distance, 10 );
	// Vector3-likes work too
	close( orbitFromPose( { x: 0, y: 0, z: 5 }, { x: 0, y: 0, z: 1 } ).distance, 4 );

} );

test( 'lerpPose: end points, geometric distance, short way round, linear target and fov', () => {

	const a = { position: orbitToPosition( [ 0, 0, 0 ], 3.0, 0.2, 100 ), target: [ 0, 0, 0 ], fov: 45 };
	const b = { position: orbitToPosition( [ 10, 20, 30 ], - 3.0, - 0.4, 400 ), target: [ 10, 20, 30 ], fov: 75 };
	const p0 = lerpPose( a, b, 0 ), p1 = lerpPose( a, b, 1 );
	for ( let k = 0; k < 3; k ++ ) {

		close( p0.position[ k ], a.position[ k ], 1e-9 );
		close( p1.position[ k ], b.position[ k ], 1e-7 );

	}

	const mid = lerpPose( a, b, 0.5 );
	assert.deepEqual( mid.target, [ 5, 10, 15 ] );
	assert.equal( mid.fov, 60 );
	const o = orbitFromPose( mid.position, mid.target );
	close( o.distance, 200, 1e-6, 'geometric mean of 100 and 400' );
	close( o.pitch, - 0.1, 1e-9 );
	// from yaw 3.0 to −3.0 the short way crosses π (0.283 rad), not zero (6 rad)
	close( Math.abs( shortestAngle( o.yaw - Math.PI ) ), 0, 1e-9, 'passes through π' );
	// writes into the object it is given
	const out = { position: [ 0, 0, 0 ], target: [ 0, 0, 0 ], fov: 0 };
	assert.equal( lerpPose( a, b, 0.25, out ), out );
	close( orbitFromPose( out.position, out.target ).distance, 100 * Math.SQRT2, 1e-6 );

} );

test( 'arcDistance: geometric without travel, keeps a far destination in the frame, continuous', () => {

	close( arcDistance( 0, 100, 400 ), 100 );
	close( arcDistance( 1, 100, 400 ), 400 );
	close( arcDistance( 0.5, 100, 400 ), 200 );
	// a destination 10 distances to the side
	const d0 = 10, d1 = 10, travel = 100, tanHalf = 0.4, margin = 0.8;
	close( arcDistance( 0, d0, d1, travel, tanHalf, margin ), d0, 1e-9, 'starts where the camera is' );
	close( arcDistance( 1, d0, d1, travel, tanHalf, margin ), d1, 1e-9, 'ends at the requested distance' );
	let prev = arcDistance( 0, d0, d1, travel, tanHalf, margin ), peak = 0;
	for ( let i = 1; i <= 200; i ++ ) {

		const e = i / 200;
		const d = arcDistance( e, d0, d1, travel, tanHalf, margin );
		assert.ok( Math.abs( d - prev ) < 0.06 * Math.max( d, prev ) + 2.5, `no jump at e = ${e}` );
		// once the pull-back is complete the destination is inside the frame
		if ( e >= 0.3 ) assert.ok( ( 1 - e ) * travel <= d * tanHalf * margin + 1e-9, `in frame at e = ${e}` );
		peak = Math.max( peak, d );
		prev = d;

	}

	assert.ok( peak > 5 * d0, 'the camera hops out' );
	// a near destination needs no hop
	for ( let i = 0; i <= 20; i ++ ) close( arcDistance( i / 20, 100, 50, 5, 0.4 ), Math.exp( Math.log( 100 ) + ( Math.log( 50 ) - Math.log( 100 ) ) * i / 20 ), 1e-9 );

} );

test( 'fitDistanceFor: the sphere touches the tighter side of the free area', () => {

	// square view, 90° fov: half angle 45°, r / sin 45°
	close( fitDistanceFor( 1, 90, 1 ), Math.SQRT2, 1e-12 );
	// landscape: the vertical fov limits; portrait: the horizontal one
	close( fitDistanceFor( 1, 90, 2 ), Math.SQRT2, 1e-12 );
	close( fitDistanceFor( 1, 90, 0.5 ), 1 / Math.sin( Math.atan( 0.5 ) ), 1e-12 );
	// a panel that covers a quarter of the width
	const full = fitDistanceFor( 100, 45, 1.6 ), part = fitDistanceFor( 100, 45, 1.6, 0.4, 1 );
	assert.ok( part > full, 'less room: farther away' );
	close( fitDistanceFor( 200, 45, 1.6 ), 2 * full, 1e-9, 'linear in the radius' );

} );

test( 'wheelZoomFactor: notches, lines, pinch and limits', () => {

	assert.ok( wheelZoomFactor( 100 ) > 1, 'scrolling down zooms out' );
	close( wheelZoomFactor( 100 ) * wheelZoomFactor( - 100 ), 1, 1e-12 );
	close( wheelZoomFactor( 3, 1 ), wheelZoomFactor( 48, 0 ), 1e-12, 'line mode' );
	close( wheelZoomFactor( 5, 0, true ), wheelZoomFactor( 40 ), 1e-12, 'pinch deltas are small' );
	close( wheelZoomFactor( 1e6 ), wheelZoomFactor( 240 ), 1e-12, 'one event cannot jump' );
	// ten small trackpad deltas equal one notch of the same total
	close( Math.pow( wheelZoomFactor( 10 ), 10 ), wheelZoomFactor( 100 ), 1e-9 );

} );

test( 'rig: constructs without a DOM and reports its state', () => {

	const rig = new CameraRig( { autoRotate: false } );
	assert.equal( rig.order, 31 );
	assert.equal( rig.autoRotate, false );
	assert.equal( rig.immersive, false );
	assert.equal( rig.busy, false );
	assert.equal( rig.userCamera, false );
	for ( const name of [ 'fitToGraph', 'flyTo', 'focusNode', 'pose', 'setPose', 'orbit', 'zoomBy', 'keepFramed', 'setAutoRotate', 'setImmersive', 'notifyInteraction', 'setInputLocked', 'setInsets', 'fitDistance' ] ) {

		assert.equal( typeof rig[ name ], 'function', name );

	}

} );

test( 'driftRates: the cinematic idle starts level, stays within its excursions and never jumps', () => {

	const r0 = driftRates( 0 );
	assert.ok( r0.pitch > 0, 'begins by rising' );
	close( r0.zoom, 0, 1e-12, 'no zoom rate at the start' );
	close( r0.focus, 0, 1e-12, 'focus pull starts on the target' );
	let pitch = 0, logDist = 0, maxPitch = 0, minDist = 0, maxDist = 0, maxRate = 0;
	const dt = 1 / 60, out = {};
	for ( let t = 0; t < 900; t += dt ) {

		const r = driftRates( t, out );
		assert.equal( r, out ); // no allocation per frame
		pitch += r.pitch * dt;
		logDist += r.zoom * dt;
		maxPitch = Math.max( maxPitch, Math.abs( pitch ) );
		minDist = Math.min( minDist, logDist );
		maxDist = Math.max( maxDist, logDist );
		maxRate = Math.max( maxRate, Math.abs( r.pitch ), Math.abs( r.zoom ) );
		assert.ok( Math.abs( r.focus ) <= 1 );

	}

	assert.ok( maxPitch <= DRIFT.pitch * 1.02, `pitch excursion ${maxPitch}` );
	assert.ok( maxPitch > DRIFT.pitch * 0.9 );
	assert.ok( minDist >= - 2 * DRIFT.zoom * 1.02 && maxDist <= 1e-3, `distance ${minDist} … ${maxDist}` ); // in by at most 20 %, never out
	assert.ok( Math.exp( minDist ) > 0.78 );
	assert.ok( maxRate < 0.02, 'slow: under 1.2° per second' );

} );

// ── the sphere the camera frames ────────────────────────────────────────────────────────────

const cloud = ( points ) => {

	const a = new Float32Array( points.length * 4 );
	points.forEach( ( p, i ) => a.set( [ p[ 0 ], p[ 1 ], p[ 2 ], 1 ], 4 * i ) );
	return a;

};

/** Deterministic points in a ball of radius r around c. */
const ball = ( n, c, r, seed = 1 ) => {

	let s = seed;
	const rnd = () => ( ( s = ( s * 1664525 + 1013904223 ) >>> 0 ) / 4294967296 );
	const out = [];
	while ( out.length < n ) {

		const x = 2 * rnd() - 1, y = 2 * rnd() - 1, z = 2 * rnd() - 1;
		if ( x * x + y * y + z * z <= 1 ) out.push( [ c[ 0 ] + r * x, c[ 1 ] + r * y, c[ 2 ] + r * z ] );

	}

	return out;

};

test( 'enclosingSphere: a symmetric cloud is framed around its middle', () => {

	const s = enclosingSphere( cloud( ball( 2000, [ 10, - 20, 30 ], 100 ) ) );
	close( s.center[ 0 ], 10, 6 ); close( s.center[ 1 ], - 20, 6 ); close( s.center[ 2 ], 30, 6 );
	assert.ok( s.radius > 85 && s.radius <= 100, `radius ${s.radius}` );

} );

test( 'enclosingSphere: a lopsided graph is framed around its extent, not around its mass', () => {

	// 900 nodes in one ball, 100 in a small far cluster: the centroid sits near the big ball
	const points = [ ...ball( 900, [ 0, 0, 0 ], 50, 2 ), ...ball( 100, [ 400, 0, 0 ], 20, 3 ) ];
	const s = enclosingSphere( cloud( points ) );
	// the smallest ball around both spans x = −50 … 420: centre near 185, radius near 235
	close( s.center[ 0 ], 185, 20, 'centre x' );
	assert.ok( s.radius > 215 && s.radius < 250, `radius ${s.radius}` );
	// the centroid sphere would be far larger (centroid x = 40: 380 to the far side)
	let cx = 0;
	for ( const p of points ) cx += p[ 0 ] / points.length;
	assert.ok( s.radius < 0.75 * ( 420 - cx ), 'much tighter than the centroid sphere' );
	// every node (but the trimmed strays) is inside
	let outside = 0;
	for ( const p of points ) if ( Math.hypot( p[ 0 ] - s.center[ 0 ], p[ 1 ] - s.center[ 1 ], p[ 2 ] - s.center[ 2 ] ) > s.radius * 1.0001 ) outside ++;
	assert.ok( outside <= 0.02 * points.length + 1, `${outside} nodes outside` );

} );

test( 'enclosingSphere: strays are trimmed, broken positions skipped, empty input is a point', () => {

	const points = ball( 500, [ 0, 0, 0 ], 10, 5 );
	const tight = enclosingSphere( cloud( points ) ).radius;
	const withStray = enclosingSphere( cloud( [ ...points, [ 5000, 0, 0 ] ] ) );
	assert.ok( withStray.radius < 12, `one stray in 501 does not set the frame: ${withStray.radius}` );
	assert.ok( Math.abs( withStray.center[ 0 ] ) < 12 );
	const all = enclosingSphere( cloud( [ ...points, [ 5000, 0, 0 ] ] ), undefined, { trim: 0 } );
	assert.ok( all.radius > 2400, 'trim 0 keeps every node' );
	const broken = enclosingSphere( cloud( [ ...points, [ NaN, 0, 0 ], [ 0, Infinity, 0 ] ] ) );
	assert.ok( Number.isFinite( broken.radius ) && Math.abs( broken.radius - tight ) < 1.5, `${broken.radius} vs ${tight}` );
	assert.deepEqual( enclosingSphere( new Float32Array( 0 ) ), { center: [ 0, 0, 0 ], radius: 0 } );
	assert.deepEqual( enclosingSphere( cloud( [ [ NaN, NaN, NaN ] ] ) ), { center: [ 0, 0, 0 ], radius: 0 } );
	const one = enclosingSphere( cloud( [ [ 3, 4, 5 ] ] ) );
	assert.deepEqual( [ one.center, one.radius ], [ [ 3, 4, 5 ], 0 ] );

} );

test( 'enclosingSphere: count limits the nodes read, large graphs are sampled, the result is reused', () => {

	const points = [ ...ball( 300, [ 0, 0, 0 ], 10, 7 ), ...ball( 300, [ 900, 0, 0 ], 10, 8 ) ];
	const a = cloud( points );
	assert.ok( enclosingSphere( a, 300 ).radius < 11, 'only the first 300' );
	assert.ok( enclosingSphere( a ).radius > 440 );
	const big = cloud( ball( 40000, [ 0, 0, 0 ], 100, 9 ) );
	const sampled = enclosingSphere( big, undefined, { maxSamples: 2000 } ), full = enclosingSphere( big, undefined, { maxSamples: 1e9 } );
	assert.ok( Math.abs( sampled.radius - full.radius ) < 4, `${sampled.radius} vs ${full.radius}` );
	const out = { center: [ 0, 0, 0 ], radius: 0 };
	assert.equal( enclosingSphere( a, 300, {}, out ), out );

} );

// ── the rig on a stand-in app: immersive transitions, poses ─────────────────────────────────

/** A rig on a stand-in app: 200 nodes on a sphere of radius 100 around the origin, the sphere layout. */
async function stubRig( { sphere = true } = {} ) {

	const { PerspectiveCamera, Vector3 } = await import( 'three/webgpu' );
	// the rig listens on window for pointer moves and keys
	globalThis.window ??= { addEventListener() {}, removeEventListener() {} };
	const N = 200, R = 100;
	const positions = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) {

		// a Fibonacci sphere
		const y = 1 - ( 2 * ( i + 0.5 ) ) / N, r = Math.sqrt( 1 - y * y ), a = i * 2.399963;
		positions.set( [ R * r * Math.cos( a ), R * y, R * r * Math.sin( a ), 1 ], 4 * i );

	}

	const camera = new PerspectiveCamera( 45, 1.6, 0.1, 10000 );
	camera.position.set( 0, 0, 400 );
	const canvas = { style: {}, addEventListener() {}, removeEventListener() {}, getBoundingClientRect: () => ( { left: 0, top: 0, width: 1280, height: 800 } ), setPointerCapture() {} };
	const app = {
		camera, renderer: { domElement: canvas }, events: new EventTarget(), controls: null,
		graph: { N }, positions, positionsVersion: 1, width: 1280, height: 800,
		boundingSphere: ( s ) => ( s.center.set( 0, 0, 0 ), s.radius = R, s ),
	};
	const buffers = { uniforms: { graphCenter: { value: new Vector3() }, sphereRadius: { value: R }, layoutBlend: { value: sphere ? 1 : 0 } } };
	const rig = new CameraRig( { autoRotate: false } );
	rig.init( app );
	if ( sphere ) app.events.dispatchEvent( new CustomEvent( 'layout', { detail: { mode: 'sphere', settled: true } } ) );
	rig.setGraph( buffers, app.graph, app );
	const dt = 1 / 60;
	/** Steps frames; returns the largest move of the camera between two frames, the largest change of its fov and of its view direction. */
	const run = ( frames ) => {

		let move = 0, dfov = 0, turn = 0;
		const p = new Vector3(), d = new Vector3(), d1 = new Vector3();
		for ( let i = 0; i < frames; i ++ ) {

			p.copy( camera.position );
			camera.getWorldDirection( d );
			const f = camera.fov;
			rig.update( dt );
			move = Math.max( move, p.distanceTo( camera.position ) );
			dfov = Math.max( dfov, Math.abs( camera.fov - f ) );
			turn = Math.max( turn, d.angleTo( camera.getWorldDirection( d1 ) ) );

		}

		return { move, dfov, turn };

	};

	const settle = () => {

		let n = 0;
		while ( rig.busy && n ++ < 2000 ) rig.update( dt );
		rig.update( dt );
		return n;

	};

	return { rig, app, camera, R, run, settle, Vector3 };

}

test( 'rig: fits the stand-in graph, and zoomBy without animation writes the camera at once', async () => {

	const { rig, camera } = await stubRig();
	const fit = camera.position.length();
	close( fit, rig.fitDistance( 100 * 1.03 * 1.4 ), 1e-6, 'sphere layout: guide sphere with room for the arcs' );
	assert.equal( rig.pose().fov, 45 );
	// no frame in between: pose() must already be the new one
	rig.zoomBy( 0.5, { animate: false } );
	close( Math.hypot( ...rig.pose().position ), fit * 0.5, 1e-6 );
	assert.equal( rig.userCamera, true, 'a zoom takes the camera' );
	await rig.orbit( { yaw: 0.7, pitch: 0.3, animate: false } );
	rig.zoomBy( 2, { animate: false } );
	const o = orbitFromPose( rig.pose().position, rig.pose().target );
	close( o.distance, fit, 1e-6 );
	close( o.yaw, 0.7, 1e-9 );
	close( o.pitch, 0.3, 1e-9 );

} );

test( 'rig: an uninterrupted visit of the immersive view ends at the fit, with the orbit\'s field of view', async () => {

	const { rig, camera, run, settle } = await stubRig();
	const fit = camera.position.length();
	rig.setImmersive( true );
	const inward = run( 96 + 2 ); // 1.6 s
	assert.equal( rig.busy, false );
	close( camera.position.length(), 0, 1e-9, 'at the centre' );
	assert.ok( camera.fov >= 75, `immersive field of view ${camera.fov}` );
	// the camera swings half way round the centre while it moves in: at most 3× the mean speed (cubic ease)
	assert.ok( inward.move < 0.08 * fit && inward.dfov < 1.5, `no jump on the way in (${inward.move}, fov ${inward.dfov})` );
	const pose = rig.pose();
	assert.equal( pose.immersive, true );
	rig.setImmersive( false );
	const outward = run( 75 + 2 ); // 1.25 s
	settle();
	assert.equal( rig.immersive, false );
	assert.ok( outward.move < 0.08 * fit && outward.dfov < 1.5, `no jump on the way out (${outward.move}, fov ${outward.dfov})` );
	close( camera.fov, 45, 1e-9 );
	assert.equal( rig.pose().fov, 45 );
	close( camera.position.length(), fit, 1e-6, 'the leave ends where a fit would stand' );
	assert.equal( rig.userCamera, false );

} );

test( 'rig: leaving the immersive view half way in returns to the orbit, without a jump', async () => {

	const { rig, camera, run, settle, Vector3 } = await stubRig();
	await rig.orbit( { yaw: 0.9, pitch: 0.4, animate: false } );
	rig.zoomBy( 0.8, { animate: false } );
	const start = camera.position.clone(), dir = camera.getWorldDirection( new Vector3() );
	const d = start.length();
	rig.setImmersive( true );
	const a = run( 8 );
	const mid = camera.position.clone(), fovMid = camera.fov;
	assert.ok( fovMid > 45 && fovMid < 60, `the field of view has begun to open (${fovMid})` );
	rig.setImmersive( false );
	// the first frame after the reversal moves no further than the frames before it did
	const first = run( 1 );
	assert.ok( first.move <= a.move + 1e-9 && mid.distanceTo( camera.position ) < 0.01 * d, `no teleport: ${first.move} after ${a.move}` );
	const b = run( 200 );
	settle();
	assert.ok( b.move < 0.02 * d && b.dfov < 1 && b.turn < 0.05, `continuous: move ${b.move}, fov ${b.dfov}, turn ${b.turn}` );
	assert.ok( start.distanceTo( camera.position ) < 1e-6 * d, 'back where it started' );
	assert.ok( dir.angleTo( camera.getWorldDirection( new Vector3() ) ) < 1e-6 );
	close( camera.fov, 45, 1e-9 );
	assert.equal( rig.immersive, false );
	assert.equal( rig.userCamera, true, 'the camera is still the user\'s' );

} );

test( 'rig: re-entering half way out and leaving again keeps the orbit\'s field of view', async () => {

	const { rig, camera, run, settle, Vector3 } = await stubRig();
	const fit = camera.position.length();
	// an orbit the person placed, closer than the fit
	await rig.orbit( { yaw: - 0.5, pitch: 0.2, animate: false } );
	rig.zoomBy( 0.7, { animate: false } );
	rig.setImmersive( true, { animate: false } );
	// the person looks somewhere else
	rig.orbit( { yaw: 0.6, pitch: - 0.2 } );
	run( 2 );
	const look = camera.getWorldDirection( new Vector3() ), fovIn = camera.fov;
	rig.setImmersive( false );
	const a = run( 8 );
	assert.ok( camera.fov < fovIn && camera.fov > 45 );
	rig.setImmersive( true );
	const b = run( 8 );
	rig.setImmersive( false );
	const c = run( 300 );
	settle();
	for ( const r of [ a, b, c ] ) assert.ok( r.move < 0.08 * fit && r.dfov < 1.5 && r.turn < 0.12, `continuous: move ${r.move}, fov ${r.dfov}, turn ${r.turn}` );
	close( camera.fov, 45, 1e-9, 'not a value from the middle of a transition' );
	assert.equal( rig.pose().fov, 45 );
	// where the first leave was heading: straight back along the look, at the distance of a fit
	close( camera.position.length(), rig.fitDistance( 100 * 1.03 * 1.4 ), 1e-6 );
	assert.ok( look.angleTo( camera.getWorldDirection( new Vector3() ) ) < 1e-6, 'along the look it left' );
	assert.deepEqual( rig.pose().target, [ 0, 0, 0 ] );

	// a leave reversed half way goes back to the look it left
	rig.setImmersive( true, { animate: false } );
	rig.orbit( { yaw: - 1.1, pitch: 0.3 } );
	run( 2 );
	look.copy( camera.getWorldDirection( new Vector3() ) );
	const fov = camera.fov;
	rig.setImmersive( false );
	run( 10 );
	rig.setImmersive( true );
	const back = run( 120 );
	assert.ok( back.move < 0.08 * fit );
	close( camera.position.length(), 0, 1e-9 );
	assert.ok( look.angleTo( camera.getWorldDirection( new Vector3() ) ) < 1e-6, 'the same look' );
	close( camera.fov, fov, 1e-6 );

} );

test( 'rig: a pose of the immersive view comes back as the same look', async () => {

	const { rig, camera, run, settle, Vector3 } = await stubRig();
	rig.setImmersive( true, { animate: false } );
	rig.orbit( { yaw: 1.2, pitch: 0.35 } );
	rig.zoomBy( 1.2 );
	run( 90 );
	const pose = rig.pose();
	const look = camera.getWorldDirection( new Vector3() );
	assert.equal( pose.immersive, true );
	assert.deepEqual( pose.position, [ 0, 0, 0 ] );
	close( pose.fov, camera.fov, 1e-3 );
	const want = new Vector3().fromArray( pose.target ).normalize();
	assert.ok( want.angleTo( look ) < 1e-6, 'the target lies along the look' );

	// an orbit pose leaves the view; the immersive pose enters it again, at once when not animated
	const orbitPose = { position: [ 0, 50, 500 ], target: [ 0, 0, 0 ], fov: 45 };
	assert.equal( await rig.setPose( orbitPose, { animate: false } ), true );
	assert.equal( rig.immersive, false );
	close( camera.fov, 45, 1e-9 );
	assert.equal( await rig.setPose( pose, { animate: false } ), true );
	assert.equal( rig.immersive, true );
	assert.equal( rig.busy, false );
	assert.ok( look.angleTo( camera.getWorldDirection( new Vector3() ) ) < 1e-6 );
	close( camera.fov, pose.fov, 1e-9 );
	close( camera.position.length(), 0, 1e-9 );

	// animated, from the orbit: resolves on arrival
	await rig.setPose( orbitPose, { animate: false } );
	let arrived = null;
	rig.setPose( pose ).then( ( ok ) => ( arrived = ok ) );
	run( 30 );
	await null;
	assert.equal( arrived, null, 'still on the way' );
	settle();
	await new Promise( ( r ) => setTimeout( r, 0 ) );
	assert.equal( arrived, true );
	assert.ok( look.angleTo( camera.getWorldDirection( new Vector3() ) ) < 1e-6 );
	close( camera.fov, pose.fov, 1e-6 );

	// while immersive: another look, as a turn; leaving in the middle of it resolves false
	const other = { position: [ 0, 0, 0 ], target: [ - 100, 20, 0 ], fov: 60, immersive: true };
	let ok = null;
	rig.setPose( other ).then( ( v ) => ( ok = v ) );
	run( 5 );
	rig.setImmersive( false );
	await new Promise( ( r ) => setTimeout( r, 0 ) );
	assert.equal( ok, false );
	settle();
	close( camera.fov, 45, 1e-9 );

} );
