// Camera control (App module, order 31): orbit, pan and zoom with inertia, fit / fly-to / focus /
// pose animations, framing that follows the layout, idle auto-rotate, keyboard control and the
// immersive "inside the sphere" view of the paper's spherical layout.
//
// Design
//  • The rig owns the orbit itself (no OrbitControls): the camera is ( target, yaw, pitch,
//    distance ) around the world up axis. Input never moves the camera directly: it moves a GOAL
//    state, and the camera follows the goal through a critically damped spring (smoothDamp: no
//    overshoot, stable at any frame time), so mouse, trackpad, touch and keys all get the same
//    eased motion. Releasing a rotate drag in motion keeps the goal turning with friction
//    (inertia). Zoom acts on log( distance ) and moves the goal target so that the point under
//    the cursor (wheel) or between the fingers (pinch) stays where it is.
//      mouse     left drag rotate, right drag (or ctrl / shift + left) pan, wheel zoom to cursor,
//                middle drag zoom
//      trackpad  two-finger scroll zooms like a wheel, pinch (ctrl + wheel) zooms faster
//      touch     one finger rotates, two fingers pinch-zoom about their centre and pan with it
//      keys      arrows orbit, + / − zoom, Home fits (only while no control has the focus; F
//                fits too, through Interaction, which owns Esc, F, Space, R, L and ?)
//    `rig.controls` is a small stand-in for the OrbitControls other code expects: `target`
//    (Vector3), `enabled`, `enableRotate`, `enablePan`, `enableZoom`, `minDistance`,
//    `maxDistance`, `update()`. Code that sets camera.position / controls.target by hand keeps
//    working: the rig notices a pose it did not write and adopts it. An existing `app.controls`
//    (the App's OrbitControls) is disposed and replaced.
//  • Animations (fitToGraph, flyTo, focusNode, setPose, orbit) are flights of the same state:
//    target linear, angles the short way round, distance geometric, ease-in-out. A flight is an
//    arc: when the destination is far to the side, the camera first pulls back just enough to
//    bring it into the frame and keeps it there while it travels (arcDistance). Flights return
//    a Promise ( true = arrived, false = interrupted ); any camera gesture interrupts them.
//    focusNode( i ) keeps tracking the node during the flight (it may move under the layout).
//  • Framing uses the real extent: a tight sphere around the latest positions (enclosingSphere:
//    strays left out, centred on the extent instead of the mass; the guide sphere in the sphere
//    layout), so flat, radial, resized and lopsided layouts are framed correctly.
//    keepFramed( s ) follows that sphere softly for s seconds. On a 'layout' event the view is
//    refitted softly: a new mode starts a follow (the flat layout also turns the camera to face
//    its plane), a settle corrects what is left. None of this happens once the user has TAKEN the
//    camera (any orbit / pan / zoom gesture, flyTo, focusNode, setPose, orbit); fitToGraph() and
//    a new graph hand it back.
//  • Clip planes are not touched here: app.gfx fits near / far every frame.
//  • Auto-rotate: slow turn that starts `idleDelay` seconds after the last input, ramped in
//    smoothly; any input (pointer, wheel, key, camera gesture) eases it out. Off by default when
//    the user prefers reduced motion (`rig.reducedMotion`); an explicit setAutoRotate( true )
//    still wins. Reduced motion also shortens all animations and drops the inertia.
//  • Immersive mode (paper's VR view adapted to screens): the camera sits at the sphere centre
//    (buffers.uniforms.graphCenter) and looks around: drag = yaw / pitch with inertia (pitch
//    clamped to ±80°), wheel / pinch = field of view (30–115°), arrows and + / − likewise.
//    Entering swings the camera around behind the centre while dollying in; leaving pulls
//    straight back along the current view direction to the distance of a fit. Every transition
//    starts from the camera as it stands, and one that is reversed half way returns to where it
//    came from (the orbit it left, or the look it left), so toggling quickly never jumps and the
//    orbit keeps its own field of view.
//  • View insets: UI chrome (the settings panel, the phone bottom sheet) covers part of the
//    canvas. setInsets( { left, right, top, bottom } ) shifts the projection centre (camera view
//    offset, eased) so the graph is centred in the free area, and fitting uses the free area.
//    When the free area changes (insets, window size) and the rig still owns the framing, the
//    fit follows softly; a camera the user placed keeps its distance and only shifts. A fit never
//    assumes less than half of the height (a quarter of the width) to be free: a tall sheet, like
//    the node card of a phone, moves the view centre but does not push the camera far out.
//  • Interaction (order 30) calls setInputLocked( true ) while a press that started on a node is
//    in progress: rotation / pan / look are blocked but pointers stay tracked, so when a second
//    finger lands the lock is released and the pinch goes straight to the camera.
//
// API
//   const rig = app.use( new CameraRig( { autoRotate = true, autoRotateSpeed = 0.45, idleDelay = 8,
//                                          followSeconds = 8, keys = true } ) )
//   rig.controls                       { target, enabled, enableRotate, enablePan, enableZoom,
//                                        minDistance, maxDistance, update() }
//   rig.fitToGraph( { animate = true, padding (1.12; 1.4 in the sphere layout, whose bundles arc
//                     outside the nodes), direction, duration } ) -> Promise
//                                      direction = view direction (Vector3, toward the graph)
//   rig.flyTo( point, distance?, { duration, direction, track, animate } ) -> Promise
//   rig.focusNode( index, { animate = true, distance } ) -> Promise
//   rig.pose() -> { position: [ x, y, z ], target: [ x, y, z ], fov }      world units, fov in °
//                                      (in the immersive view: the centre, a point one radius
//                                      along the look, the look's fov, and `immersive: true`)
//   rig.setPose( pose, { duration = 1.2 (s), animate = true } ) -> Promise
//                                      an orbit pose leaves the immersive view; a pose with
//                                      `immersive: true` enters it (or stays) and turns the look
//   rig.orbit( { yaw = 0, pitch = 0 (radians, relative, positive yaw = camera to the right,
//                positive pitch = camera up), duration = 1.5 (s) } ) -> Promise
//   rig.zoomBy( factor, { animate = true } )   distance × factor about the view centre; without
//                                      animation the camera is written at once, like setPose,
//                                      orbit and fitToGraph (pose() right after it is current)
//   rig.keepFramed( seconds )
//   rig.keepPoseOnNextGraph()          the next setGraph does not refit (re-clustering)
//   rig.setAutoRotate( on ) / rig.autoRotate / rig.autoRotating (currently turning)
//   rig.drifting / rig.driftFocus      cinematic idle: after options.cinematicDelay (45 s) of
//                                      turning, the camera also rises, sinks and moves in and
//                                      out on a slow path (driftRates) until any input; driftFocus
//                                      { distance, range } is its focus pull for app.gfx.setFocus
//   rig.setImmersive( on, { animate = true, direction, fov } ) / rig.toggleImmersive() /
//   rig.immersive                      direction (unit vector) and fov (°): the look to arrive at
//   rig.notifyInteraction()            resets the idle timer (pauses auto-rotate)
//   rig.setInputLocked( on )           used by Interaction during node presses / drags
//   rig.setInsets( { left, right, top, bottom }, { animate = true } )   CSS px covered by UI
//   rig.fitDistance( radius )          distance at which a sphere fills the free area
//   rig.busy                           true while a camera animation runs
//   rig.userCamera                     true once the user has taken the camera (see above)
//   rig.reducedMotion                  prefers-reduced-motion: reduce
// Events on app.events: 'immersive' { immersive }, 'autorotate' { active }, 'cinematic' { active }.
// Pure helpers (tested in test/shell-camera.test.js): smoothDamp, easeInOutCubic, shortestAngle,
// orbitFromPose, orbitToPosition, lerpPose, arcDistance, fitDistanceFor, wheelZoomFactor, driftRates,
// enclosingSphere.

import { Vector3, Quaternion, Matrix4, MathUtils, Sphere } from 'three/webgpu';

const UP = new Vector3( 0, 1, 0 );
const _v = new Vector3();
const _v2 = new Vector3();
const _v3 = new Vector3();
const _axis = new Vector3();
const _q = new Quaternion();
const _m = new Matrix4();
const ZERO = new Vector3();
const _sd = [ 0, 0 ];
const _free = { w: 1, h: 1 };

const MAX_PITCH = 1.553; // 89°: the up vector stays well defined
// smallest share of the view's width / height that a fit assumes to be free (see _freeFraction)
const FREE_MIN_W = 0.25, FREE_MIN_H = 0.5;
const KEY_TARGETS = 'input, select, textarea, button, a, summary, [role], [contenteditable=""], [contenteditable="true"], [tabindex]';

// ── pure helpers ───────────────────────────────────────────────────────────────────────────

export const easeInOutCubic = ( t ) => ( t < 0.5 ? 4 * t * t * t : 1 - Math.pow( - 2 * t + 2, 3 ) / 2 );
const smooth = ( t ) => t * t * ( 3 - 2 * t );
const clamp = ( v, a, b ) => Math.min( b, Math.max( a, v ) );
const lerp = ( a, b, t ) => a + ( b - a ) * t;

/**
 * One step of a critically damped spring from x (velocity v) toward `target`: reaches it in about
 * `smoothTime` seconds without overshoot and is stable for any dt (closed form, after Game
 * Programming Gems 4, 1.10). Returns out = [ x', v' ].
 */
export function smoothDamp( x, v, target, smoothTime, dt, out = [ 0, 0 ] ) {

	if ( ! ( dt > 0 ) ) return ( out[ 0 ] = x, out[ 1 ] = v, out );
	if ( ! ( smoothTime > 1e-4 ) ) return ( out[ 0 ] = target, out[ 1 ] = 0, out );
	const omega = 2 / smoothTime;
	const k = omega * dt;
	const decay = 1 / ( 1 + k + 0.48 * k * k + 0.235 * k * k * k );
	const change = x - target;
	const temp = ( v + omega * change ) * dt;
	let nv = ( v - omega * temp ) * decay;
	let nx = target + ( change + temp ) * decay;
	// never pass the target (large steps with a high initial velocity)
	if ( ( target - x > 0 ) === ( nx > target ) && nx !== target ) {

		nx = target;
		nv = 0;

	}

	out[ 0 ] = nx; out[ 1 ] = nv;
	return out;

}

/** The angle a wrapped into ( −π, π ]. */
export function shortestAngle( a ) {

	a = ( a + Math.PI ) % ( 2 * Math.PI );
	if ( a <= 0 ) a += 2 * Math.PI;
	return a - Math.PI;

}

/** Orbit parameters of a camera at `position` looking at `target` (arrays or Vector3-likes). */
export function orbitFromPose( position, target ) {

	const px = position.x ?? position[ 0 ], py = position.y ?? position[ 1 ], pz = position.z ?? position[ 2 ];
	const tx = target.x ?? target[ 0 ], ty = target.y ?? target[ 1 ], tz = target.z ?? target[ 2 ];
	const x = px - tx, y = py - ty, z = pz - tz;
	const distance = Math.max( 1e-9, Math.hypot( x, y, z ) );
	return { yaw: Math.atan2( x, z ), pitch: clamp( Math.asin( clamp( y / distance, - 1, 1 ) ), - MAX_PITCH, MAX_PITCH ), distance };

}

/** Camera position of an orbit: out = target + distance · ( sin yaw cos pitch, sin pitch, cos yaw cos pitch ). */
export function orbitToPosition( target, yaw, pitch, distance, out = [ 0, 0, 0 ] ) {

	const tx = target.x ?? target[ 0 ], ty = target.y ?? target[ 1 ], tz = target.z ?? target[ 2 ];
	const c = Math.cos( pitch );
	const x = tx + distance * Math.sin( yaw ) * c, y = ty + distance * Math.sin( pitch ), z = tz + distance * Math.cos( yaw ) * c;
	if ( out.isVector3 ) return out.set( x, y, z );
	out[ 0 ] = x; out[ 1 ] = y; out[ 2 ] = z;
	return out;

}

/**
 * Pose between a and b ( { position: [3], target: [3], fov } ) at t in [0,1]: the target moves
 * linearly, the orbit angles the short way round, the distance geometrically, the fov linearly.
 */
export function lerpPose( a, b, t, out = { position: [ 0, 0, 0 ], target: [ 0, 0, 0 ], fov: 45 } ) {

	const oa = orbitFromPose( a.position, a.target ), ob = orbitFromPose( b.position, b.target );
	for ( let k = 0; k < 3; k ++ ) out.target[ k ] = lerp( a.target[ k ], b.target[ k ], t );
	const yaw = oa.yaw + shortestAngle( ob.yaw - oa.yaw ) * t;
	const pitch = lerp( oa.pitch, ob.pitch, t );
	const distance = Math.exp( lerp( Math.log( oa.distance ), Math.log( ob.distance ), t ) );
	orbitToPosition( out.target, yaw, pitch, distance, out.position );
	out.fov = lerp( a.fov ?? 45, b.fov ?? 45, t );
	return out;

}

/**
 * Camera distance of a flight at eased progress e: geometric from d0 to d1, but never so close
 * that the destination (still ( 1 − e ) · travel to the side of the look-at point) leaves the
 * frame; the pull-back is eased in over the first third so nothing jumps. tanHalf = tangent of
 * the smaller half field of view, margin = part of the half frame the destination may use.
 */
export function arcDistance( e, d0, d1, travel = 0, tanHalf = 0.41, margin = 0.8 ) {

	const geo = Math.exp( lerp( Math.log( d0 ), Math.log( d1 ), e ) );
	if ( ! ( travel > 0 ) ) return geo;
	const need = ( ( 1 - e ) * travel ) / ( tanHalf * margin );
	const w = smooth( clamp( e / 0.3, 0, 1 ) );
	return geo + w * Math.max( 0, need - geo );

}

/** Distance at which a sphere of radius r fills the part ( fracW × fracH ) of a view with the given vertical fov (degrees) and aspect. */
export function fitDistanceFor( r, fovDeg, aspect, fracW = 1, fracH = 1 ) {

	const t = Math.tan( ( fovDeg * Math.PI ) / 360 );
	const half = Math.min( Math.atan( t * fracH ), Math.atan( t * aspect * fracW ) );
	return r / Math.sin( half );

}

/** Zoom factor ( distance multiplier ) of a wheel event: notched wheels, trackpad scroll and pinch (ctrl) feel alike. */
export function wheelZoomFactor( deltaY, deltaMode = 0, ctrlKey = false ) {

	let dy = deltaY * ( deltaMode === 1 ? 16 : deltaMode === 2 ? 400 : 1 );
	if ( ctrlKey ) dy *= 8; // trackpad pinch: small deltas
	return Math.exp( clamp( dy, - 240, 240 ) * 0.0016 );

}

/**
 * A tight sphere around node positions ( Float32Array, 4 floats per node ). The outermost `trim`
 * of the nodes (by distance from the centroid) are left out, so a few strays do not shrink the
 * picture; the centre is that of the smallest ball around the rest (Bădoiu–Clarkson iteration:
 * step toward the farthest point with a shrinking step), not the centroid, so a graph with one
 * far cluster is framed around its extent instead of around its mass. Never larger than the
 * centroid sphere of the same nodes. Large graphs are sampled ( at most `maxSamples` nodes ).
 * @returns {{ center: number[], radius: number }}
 */
export function enclosingSphere( positions, count = positions.length >> 2, { trim = 0.02, maxSamples = 16384, iterations = 24 } = {}, out = { center: [ 0, 0, 0 ], radius: 0 } ) {

	const n = Math.min( count, positions.length >> 2 );
	const stride = Math.max( 1, Math.ceil( n / maxSamples ) );
	let cx = 0, cy = 0, cz = 0, k = 0;
	for ( let i = 0; i < n; i += stride ) {

		const x = positions[ 4 * i ], y = positions[ 4 * i + 1 ], z = positions[ 4 * i + 2 ];
		if ( x - x !== 0 || y - y !== 0 || z - z !== 0 ) continue; // NaN or infinite: not a place
		cx += x; cy += y; cz += z; k ++;

	}

	if ( k === 0 ) {

		out.center[ 0 ] = out.center[ 1 ] = out.center[ 2 ] = 0;
		out.radius = 0;
		return out;

	}

	cx /= k; cy /= k; cz /= k;
	// squared distances from the centroid decide who is a stray
	const d2 = new Float32Array( k );
	let j = 0;
	for ( let i = 0; i < n; i += stride ) {

		const x = positions[ 4 * i ] - cx, y = positions[ 4 * i + 1 ] - cy, z = positions[ 4 * i + 2 ] - cz;
		if ( x - x !== 0 || y - y !== 0 || z - z !== 0 ) continue;
		d2[ j ++ ] = x * x + y * y + z * z;

	}

	const limit = d2.slice().sort()[ Math.min( k - 1, Math.floor( k * ( 1 - trim ) ) ) ];
	let bx = cx, by = cy, bz = cz, best = limit; // the centroid sphere is the fallback
	let x0 = cx, y0 = cy, z0 = cz;
	for ( let it = 1; it <= iterations; it ++ ) {

		let far = - 1, fx = 0, fy = 0, fz = 0;
		j = 0;
		for ( let i = 0; i < n; i += stride ) {

			const px = positions[ 4 * i ], py = positions[ 4 * i + 1 ], pz = positions[ 4 * i + 2 ];
			if ( px - px !== 0 || py - py !== 0 || pz - pz !== 0 ) continue;
			if ( d2[ j ++ ] > limit ) continue;
			const x = px - x0, y = py - y0, z = pz - z0;
			const q = x * x + y * y + z * z;
			if ( q > far ) {

				far = q; fx = px; fy = py; fz = pz;

			}

		}

		if ( far < 0 ) break;
		if ( far < best ) {

			best = far; bx = x0; by = y0; bz = z0;

		}

		const w = 1 / ( it + 1 );
		x0 += ( fx - x0 ) * w; y0 += ( fy - y0 ) * w; z0 += ( fz - z0 ) * w;

	}

	out.center[ 0 ] = bx; out.center[ 1 ] = by; out.center[ 2 ] = bz;
	out.radius = Math.sqrt( Math.max( 0, best ) );
	return out;

}

/** Spherical interpolation of unit vectors; antipodal pairs rotate about the world up axis. */
function slerpUnit( a, b, t, out ) {

	const dot = MathUtils.clamp( a.dot( b ), - 1, 1 );
	if ( dot > 0.9999 ) return out.copy( a ).lerp( b, t ).normalize();
	if ( dot < - 0.9999 ) {

		_axis.copy( UP ).addScaledVector( a, - UP.dot( a ) );
		if ( _axis.lengthSq() < 1e-8 ) _axis.set( 1, 0, 0 );
		_axis.normalize();

	} else {

		_axis.crossVectors( a, b ).normalize();

	}

	_q.setFromAxisAngle( _axis, Math.acos( dot ) * t );
	return out.copy( a ).applyQuaternion( _q ).normalize();

}

/**
 * Path of the cinematic idle, as rates at path time t (s): the camera rises and sinks (pitch,
 * rad/s) and moves in and out (zoom, 1/s on the logarithm of the distance), each on its own slow
 * period so the path does not repeat for a long time. Integrated from t = 0 the pitch is
 * A·sin( … ) around where it started and the distance goes in by up to 20 % and back; `focus`
 * (−1 … 1) is the phase of the focus pull. Rates rather than offsets: the path has no anchor, so
 * stopping it leaves the camera where it is.
 */
export const DRIFT = { pitch: 0.2, pitchPeriod: 71, zoom: 0.11, zoomPeriod: 97, focusPeriod: 43 };

export function driftRates( t, out = { pitch: 0, zoom: 0, focus: 0 } ) {

	const wp = ( 2 * Math.PI ) / DRIFT.pitchPeriod, wz = ( 2 * Math.PI ) / DRIFT.zoomPeriod;
	out.pitch = DRIFT.pitch * wp * Math.cos( wp * t );
	out.zoom = - DRIFT.zoom * wz * Math.sin( wz * t );
	out.focus = Math.sin( ( 2 * Math.PI * t ) / DRIFT.focusPeriod );
	return out;

}

/** Camera quaternion looking along unit direction d (world up = +Y). */
function lookQuaternion( d, out ) {

	_m.lookAt( ZERO, d, Math.abs( d.y ) > 0.999 ? _v2.set( 0, 0, 1 ) : UP );
	return out.setFromRotationMatrix( _m );

}

export function prefersReducedMotion() {

	return typeof matchMedia === 'function' && matchMedia( '(prefers-reduced-motion: reduce)' ).matches;

}

/** What the rest of the app sees where OrbitControls used to be. */
class RigControls {

	constructor( rig, target ) {

		this.isRigControls = true;
		this.rig = rig;
		this.target = target;
		this.enabled = true;
		this.enableRotate = true;
		this.enablePan = true;
		this.enableZoom = true;
		this.minDistance = 0;
		this.maxDistance = Infinity;

	}

	/** Adopts a camera pose set from outside (camera.position / target written by hand). */
	update() {

		return this.rig._adoptExternal();

	}

	dispose() {}

}

export class CameraRig {

	constructor( options = {} ) {

		this.order = 31;
		this.options = {
			orbitControls: null,
			autoRotate: true,
			autoRotateSpeed: 0.45, // 2π/60 rad/s per unit → ~2.2 min per turn
			autoRotateRamp: 2.5, // seconds to reach full speed
			idleDelay: 8,
			startDelay: 2, // auto-rotate starts this long after a graph loads
			cinematic: true, // after a long idle the turn becomes a slow drift (see driftRates)
			cinematicDelay: 45, // seconds of turning before the drift begins
			cinematicRamp: 8, // seconds to ease into it
			followSeconds: 8,
			minFov: 30,
			maxFov: 115,
			keys: true,
			rotateSpeed: 0.6, // turns of π·2 per canvas height dragged (mouse); touch uses 0.9
			smoothTime: 0.085, // s, orbit / pan follow
			zoomSmoothTime: 0.13, // s
			friction: 4.5, // 1/s, decay of the rotate inertia
			...options,
		};
		// known at construction so the app can sync settings.autoRotate before applying them
		this.reducedMotion = prefersReducedMotion();
		this.autoRotate = this.options.autoRotate && ! this.reducedMotion;
		this.idleDelay = this.options.idleDelay;
		this.autoRotateSpeed = this.options.autoRotateSpeed;
		this.immersive = false;
		this.autoRotating = false;
		this.app = null;
		this.controls = null;

		this.drifting = false; // the cinematic idle is moving the camera
		this.driftFocus = { distance: 0, range: 0 }; // its focus pull, for app.gfx.setFocus (world units)

		this._time = 0;
		this._idle = 0;
		this._autoAmount = 0;
		this._driftAmount = 0;
		this._driftTime = 0;
		this._driftRates = { pitch: 0, zoom: 0, focus: 0 };
		this._inputLocked = false;
		this._taken = false;
		this._center = new Vector3();
		this._radius = 100;
		this._sphere = new Sphere();
		this._ball = { center: [ 0, 0, 0 ], radius: 0 };
		this._sphereTime = - Infinity;
		this._sphereVersion = - 1;
		this._orbitFov = 45;
		this._layoutMode = null;
		this._layoutSettled = false;

		// orbit: current state, goal state, spring velocities, inertia of a released rotate drag
		this._o = { yaw: 0, pitch: 0, dist: 400, target: new Vector3() };
		this._g = { yaw: 0, pitch: 0, logDist: Math.log( 400 ), target: new Vector3() };
		this._vel = { yaw: 0, pitch: 0, logDist: 0, x: 0, y: 0, z: 0 };
		this._fling = { yaw: 0, pitch: 0 };
		this._wPos = new Vector3( NaN, NaN, NaN ); // what the rig wrote last (see _adoptExternal)
		this._wTarget = new Vector3( NaN, NaN, NaN );
		this._dirty = true;

		// flight
		this._flight = null;
		this._fl = { t: 0, dur: 1, track: - 1, t0: new Vector3(), t1: new Vector3(), y0: 0, y1: 0, p0: 0, p1: 0, d0: 1, d1: 1, f0: 45, f1: 45, arc: true, resolve: null, after: 0 };
		// framing follow
		this._follow = null;
		this._followCenter = new Vector3();
		this._followDist = 0;
		this._followNext = 0;
		// immersive
		this._look = { yaw: 0, pitch: 0, vyaw: 0, vpitch: 0, fov: 75, fovTarget: 75 };
		this._lookTween = null;
		this._transition = null;
		this._lookArrive = null; // resolve function of a look that is still turning (flyTo / setPose in the immersive view)
		// p0 / q0 / f0: the camera when the transition began; 'in' swings u0 → u1 around the centre from
		// distance d0, 'out' moves p0 → p1. home / homeQ: the orbit side of the transitions under way
		// (the orbit an entry left, homeOrbit, or the place a leave is heading for), where a reversed
		// entry goes back to; keepTarget: the leave ends in the orbit it came from, around its target
		this._tr = { t: 0, dur: 1.4, dir: 'in', p0: new Vector3(), p1: new Vector3(), u0: new Vector3(), u1: new Vector3(), d0: 1, q0: new Quaternion(), q1: new Quaternion(), f0: 45, f1: 75, home: new Vector3(), homeQ: new Quaternion(), homeOrbit: true, keepTarget: false };
		// pointers
		this._pointers = new Map(); // pointerId -> { x, y, type }
		this._gesture = null; // { mode: 'rotate' | 'pan' | 'dolly' | 'pinch', … }
		this._lookDrag = null;
		this._pinch = null;
		this._rect = { left: 0, top: 0, width: 1, height: 1 };
		this._dir = new Vector3();
		this._camCenter = new Vector3();
		this._insets = { left: 0, right: 0, top: 0, bottom: 0 };
		this._insetCur = { left: 0, right: 0, top: 0, bottom: 0 };
		this._insetsDirty = false;

	}

	// ── module hooks ──────────────────────────────────────────────────────────────────────

	init( app ) {

		this.app = app;
		this.camera = app.camera;
		this.canvas = app.renderer.domElement;
		this._orbitFov = this.camera.fov;

		// take over from OrbitControls where the App (or the caller) made one
		const old = this.options.orbitControls || app.controls;
		const target = new Vector3();
		if ( old && old.target ) target.copy( old.target );
		if ( old && ! old.isRigControls ) old.dispose?.();
		this.controls = new RigControls( this, target );
		app.controls = this.controls;
		this._o.target = target;
		this._adoptCamera();

		if ( typeof matchMedia === 'function' ) {

			this._mq = matchMedia( '(prefers-reduced-motion: reduce)' );
			this._onMq = () => {

				this.reducedMotion = this._mq.matches;

			};

			this._mq.addEventListener?.( 'change', this._onMq );

		}

		const c = this.canvas;
		// touches on the picture belong to the camera: without this the browser pans and pinch-zooms
		// the page instead (OrbitControls sets it too, and takes it back when it is disposed)
		this._touchAction = c.style.touchAction;
		c.style.touchAction = 'none';
		this._onDown = ( e ) => this._pointerDown( e );
		this._onMove = ( e ) => this._pointerMove( e );
		this._onUp = ( e ) => this._pointerUp( e );
		this._onWheel = ( e ) => this._wheel( e );
		this._onHover = ( e ) => {

			if ( e.pointerType === 'mouse' ) this.notifyInteraction();

		};

		this._onKey = ( e ) => this._keyDown( e );
		this._onMenu = ( e ) => e.preventDefault();
		this._onLayout = ( e ) => this._layoutEvent( e.detail || {} );
		c.addEventListener( 'pointerdown', this._onDown );
		c.addEventListener( 'pointermove', this._onHover );
		c.addEventListener( 'contextmenu', this._onMenu );
		window.addEventListener( 'pointermove', this._onMove );
		window.addEventListener( 'pointerup', this._onUp );
		window.addEventListener( 'pointercancel', this._onUp );
		c.addEventListener( 'wheel', this._onWheel, { passive: false } );
		window.addEventListener( 'keydown', this._onKey );
		app.events.addEventListener( 'layout', this._onLayout );
		this._updateRect();

	}

	setGraph( buffers, graph, app ) {

		this.buffers = buffers;
		this.graph = graph;
		this._sphereVersion = - 1;
		this._refreshSphere( true );
		this._layoutSettled = false;
		if ( this.immersive ) return;
		if ( this._keepPose ) {

			// the same nodes under a new hierarchy (re-clustering): the view stays where it is
			this._keepPose = false;
			if ( ! this._taken ) this.keepFramed( this.options.followSeconds );
			return;

		}

		this._endFlight( false );
		this._taken = false;
		this.fitToGraph( { animate: false } );
		this.keepFramed( this.options.followSeconds );
		this._idle = Math.max( 0, this.idleDelay - this.options.startDelay );
		void app;

	}

	update( dt ) {

		this._time += dt;
		this._idle += dt;
		this._refreshSphere( false );

		if ( this._transition ) this._stepTransition( dt );
		else if ( this.immersive ) this._stepImmersive( dt );
		else this._adoptExternal();
		this._stepAutoRotate( dt );
		if ( ! this._transition && ! this.immersive ) {

			if ( this._flight ) this._stepFlight( dt );
			else {

				if ( this._follow && ! this._inputLocked ) this._stepFollow( dt );
				this._stepInertia( dt );
				this._stepOrbit( dt );

			}

			this._writeCamera();

		}

		this._stepInsets( dt );

	}

	resize() {

		this._insetsDirty = true;
		this._applyViewOffset();
		this._updateRect();
		this._reframe();

	}

	dispose() {

		const c = this.canvas;
		if ( ! c ) return;
		c.removeEventListener( 'pointerdown', this._onDown );
		c.removeEventListener( 'pointermove', this._onHover );
		c.removeEventListener( 'contextmenu', this._onMenu );
		window.removeEventListener( 'pointermove', this._onMove );
		window.removeEventListener( 'pointerup', this._onUp );
		window.removeEventListener( 'pointercancel', this._onUp );
		c.removeEventListener( 'wheel', this._onWheel );
		c.style.touchAction = this._touchAction || '';
		window.removeEventListener( 'keydown', this._onKey );
		this.app?.events.removeEventListener( 'layout', this._onLayout );
		this._mq?.removeEventListener?.( 'change', this._onMq );
		this._endFlight( false );

	}

	// ── public API ────────────────────────────────────────────────────────────────────────

	get busy() {

		return !! ( this._flight || this._transition || this._lookTween );

	}

	/** True once the user (or a caller of flyTo / focusNode / setPose / orbit) has taken the camera. */
	get userCamera() {

		return this._taken;

	}

	notifyInteraction() {

		this._idle = 0;

	}

	setAutoRotate( on ) {

		this.autoRotate = !! on;
		if ( on ) this._idle = Math.max( this._idle, this.idleDelay - 0.5 );

	}

	setInputLocked( on ) {

		this._inputLocked = !! on;
		const c = this.controls;
		if ( c ) {

			c.enableRotate = ! on;
			c.enablePan = ! on;

		}

		if ( on ) {

			this._gesture = null;
			this._lookDrag = null;
			this._look.vyaw = 0; this._look.vpitch = 0;
			this._fling.yaw = this._fling.pitch = 0;
			this.notifyInteraction();

		}

	}

	/** Distance at which a sphere of radius r fills the free (non-inset) part of the view. */
	fitDistance( r ) {

		const f = this._freeFraction();
		return fitDistanceFor( r, this._orbitFov, this.camera.aspect, f.w, f.h );

	}

	/**
	 * The share of the view that counts as free when a distance is fitted. Covered height counts
	 * up to half of the view only: a sheet that takes more than that (the node card of a phone)
	 * shifts the view centre like every inset, but does not push the camera far out for the
	 * moment it is open (FREE_MIN_H, FREE_MIN_W).
	 */
	_freeFraction() {

		const ins = this._insets;
		const w = Math.max( 1, this.app?.width || 1 ), h = Math.max( 1, this.app?.height || 1 );
		_free.w = Math.max( FREE_MIN_W * w, w - ins.left - ins.right ) / w;
		_free.h = Math.max( FREE_MIN_H * h, h - ins.top - ins.bottom ) / h;
		return _free;

	}

	setInsets( { left = 0, right = 0, top = 0, bottom = 0 } = {}, { animate = true } = {} ) {

		const ins = this._insets;
		const changed = ins.left !== left || ins.right !== right || ins.top !== top || ins.bottom !== bottom;
		ins.left = left; ins.right = right; ins.top = top; ins.bottom = bottom;
		if ( ! animate ) Object.assign( this._insetCur, ins );
		this._insetsDirty = true;
		if ( ! animate && this.camera ) this._applyViewOffset();
		if ( changed ) this._reframe();

	}

	/**
	 * The free part of the view changed (a surface opened or closed, the window was resized): while
	 * the rig still owns the framing the fit follows softly; a camera the user placed only shifts.
	 */
	_reframe( seconds = 2.5 ) {

		if ( this.immersive || this._transition || this._taken || this._flight || ! this.graph || ! this.camera ) return;
		const until = this._time + seconds;
		if ( this._follow ) this._follow.until = Math.max( this._follow.until, until );
		else this.keepFramed( seconds );

	}

	fitToGraph( { animate = true, padding = null, direction = null, duration = null } = {} ) {

		this._refreshSphere( true );
		if ( this.immersive ) {

			this._layoutDirection( _v );
			return this._startLookTween( _v, this._immersiveFov(), animate );

		}

		this._taken = false; // an explicit fit hands the framing back to the rig
		return this._fit( { animate, padding, direction, duration } );

	}

	flyTo( point, distance = null, { duration = null, direction = null, track = - 1, animate = true } = {} ) {

		if ( this.immersive ) {

			_v.copy( point ).sub( this._immersiveCenter( _v2 ) );
			if ( _v.lengthSq() > 1e-12 ) return this._startLookTween( _v.normalize(), this._look.fovTarget, animate );
			return Promise.resolve( true );

		}

		this._taken = true;
		return this._fly( point, distance, { duration, direction, track, animate } );

	}

	focusNode( index, { animate = true, distance = null } = {} ) {

		const app = this.app, p = app.positions, g = app.graph;
		if ( ! p || ! g || index < 0 || index >= g.N ) return Promise.resolve( false );
		const x = p[ 4 * index ], y = p[ 4 * index + 1 ], z = p[ 4 * index + 2 ];
		_v3.set( x, y, z );
		if ( this.immersive ) return this.flyTo( _v3, null, { animate } );

		let dist = distance;
		if ( ! dist ) {

			// frame the node's neighbourhood (sampled), within sane bounds of the graph size
			const s = g.adjStart[ index ], n = Math.min( g.degree[ index ], 256 );
			let r = 0;
			for ( let k = 0; k < n; k ++ ) {

				const j = g.adjList[ s + k ];
				r = Math.max( r, Math.hypot( p[ 4 * j ] - x, p[ 4 * j + 1 ] - y, p[ 4 * j + 2 ] - z ) );

			}

			r = MathUtils.clamp( r * 1.15, this._radius * 0.08, this._radius * 0.75 );
			dist = this.fitDistance( r );

		}

		return this.flyTo( _v3, dist, { animate, track: index } );

	}

	/** The current view: { position, target (world units), fov (vertical, degrees) }. */
	pose() {

		const cam = this.camera, t = this.controls.target;
		if ( this.immersive ) {

			// a look from the centre has no orbit target: a point one radius along the look, and the
			// mode, so that setPose( pose() ) comes back to this look
			const position = this._immersiveCenter( _v2 ).toArray();
			_v2.addScaledVector( this._lookDirection( _v ), Math.max( 1, this._radius ) );
			return { position, target: _v2.toArray(), fov: this._look.fovTarget, immersive: true };

		}

		if ( this._transition ) {

			// on the way out of the immersive view: the camera as it stands
			cam.getWorldDirection( _v );
			_v2.copy( cam.position ).addScaledVector( _v, Math.max( 1, this._radius ) );
			return { position: cam.position.toArray(), target: _v2.toArray(), fov: cam.fov };

		}

		return { position: cam.position.toArray(), target: t.toArray(), fov: cam.fov };

	}

	/** Moves to a pose of pose(). Resolves true on arrival, false when interrupted. */
	setPose( pose, { duration = 1.2, animate = true } = {} ) {

		if ( ! pose || ! pose.position || ! pose.target ) return Promise.resolve( false );
		const nums = [ ...pose.position, ...pose.target ];
		if ( nums.length !== 6 || ! nums.every( Number.isFinite ) ) return Promise.resolve( false );
		if ( pose.immersive ) return this._setLookPose( pose, animate );
		if ( this.immersive ) this.setImmersive( false, { animate: false } );
		const o = orbitFromPose( pose.position, pose.target );
		this._taken = true;
		_v3.fromArray( pose.target );
		return this._fly( _v3, o.distance, { duration, animate, yaw: o.yaw, pitch: o.pitch, fov: Number.isFinite( pose.fov ) ? clamp( pose.fov, 10, 120 ) : this._orbitFov, arc: false } );

	}

	/** Turns the camera around the target by yaw / pitch (radians, relative). */
	orbit( { yaw = 0, pitch = 0, duration = 1.5, animate = true } = {} ) {

		if ( this.immersive || this._transition ) {

			const L = this._look;
			L.yaw -= yaw;
			L.pitch = clamp( L.pitch - pitch, - 1.396, 1.396 );
			return Promise.resolve( true );

		}

		this._taken = true;
		const g = this._g;
		return this._fly( g.target, Math.exp( g.logDist ), { duration, animate, yaw: g.yaw + yaw, pitch: clamp( g.pitch + pitch, - MAX_PITCH, MAX_PITCH ), wrap: false, arc: false } );

	}

	/** Distance × factor about the view centre (keyboard zoom). */
	zoomBy( factor, { animate = true } = {} ) {

		if ( this.immersive ) {

			const L = this._look;
			L.fovTarget = clamp( L.fovTarget * factor, this.options.minFov, this.options.maxFov );
			return;

		}

		this._userGesture();
		this._zoomGoal( factor, null );
		if ( ! animate ) {

			this._snapToGoal();
			this._writeCamera(); // pose() reads the camera: it must not wait for the next frame

		}

	}

	/** The next setGraph keeps the view (the same nodes under a new hierarchy, e.g. re-clustering). */
	keepPoseOnNextGraph() {

		this._keepPose = true;

	}

	keepFramed( seconds = this.options.followSeconds ) {

		if ( this.immersive ) return;
		this._follow = seconds > 0 ? { until: this._time + seconds } : null;
		this._followNext = 0;

	}

	toggleImmersive() {

		this.setImmersive( ! this.immersive );

	}

	/**
	 * Enters or leaves the immersive view. `direction` (unit vector) and `fov` (degrees) name the look
	 * to arrive at when entering (default: toward the middle of the layout, or the look that an
	 * interrupted leave came from).
	 */
	setImmersive( on, { animate = true, direction = null, fov = null } = {} ) {

		on = !! on;
		if ( on === this.immersive || ! this.camera ) return;
		const cam = this.camera, tr = this._tr, look = this._look;
		// the opposite transition is still under way: this one reverses it
		const reversing = !! this._transition;
		if ( on && direction ) _v.copy( direction ).normalize();
		this._endFlight( false );
		this._follow = null;
		this._dropLookTween();
		this._gesture = null;
		this._clearInertia();
		const C = this._immersiveCenter( this._camCenter );
		// from the camera as it stands, so that a transition reversed half way does not jump
		tr.p0.copy( cam.position );
		tr.q0.copy( cam.quaternion );
		tr.f0 = cam.fov;
		tr.t = 0;
		if ( on ) {

			if ( ! reversing ) {

				// the orbit this entry leaves: its field of view comes back on the way out (never a
				// value from the middle of a transition), and a reversed entry returns to its pose
				this._orbitFov = cam.fov;
				tr.home.copy( cam.position );
				tr.homeQ.copy( cam.quaternion );
				tr.homeOrbit = true;

			}

			// final look: as asked, else back to the look a leave under way came from, else the layout
			if ( ! direction && reversing ) this._lookDirection( _v );
			else if ( ! direction ) this._layoutDirection( _v );
			look.yaw = Math.atan2( _v.x, _v.z );
			look.pitch = MathUtils.clamp( Math.asin( MathUtils.clamp( _v.y, - 1, 1 ) ), - 1.396, 1.396 );
			look.vyaw = look.vpitch = 0;
			look.fovTarget = clamp( fov ?? ( reversing ? look.fovTarget : this._immersiveFov() ), this.options.minFov, this.options.maxFov );
			look.fov = look.fovTarget;
			this._lookDirection( _v );
			tr.dir = 'in';
			tr.u0.copy( cam.position ).sub( C );
			tr.d0 = tr.u0.length();
			if ( tr.d0 < 1e-6 ) tr.u0.copy( _v ).negate();
			tr.u0.normalize();
			tr.u1.copy( _v ).negate();
			tr.q1.copy( lookQuaternion( _v, _q ) );
			tr.f1 = look.fov;
			tr.dur = ( this.reducedMotion ? 0.5 : 1.6 );

		} else {

			tr.dir = 'out';
			if ( reversing ) {

				// entering was under way: back to where it came from (the orbit it left, or the place
				// the leave before it was heading for)
				tr.p1.copy( tr.home );
				tr.q1.copy( tr.homeQ );
				tr.keepTarget = tr.homeOrbit;

			} else {

				// straight back along the look, to where a fit of the graph would stand
				this._lookDirection( this._dir );
				this._refreshSphere( true );
				tr.p1.copy( C ).addScaledVector( this._dir, - this.fitDistance( this._sphere.radius * this._padding() ) );
				tr.q1.copy( lookQuaternion( this._dir, _q ) );
				tr.home.copy( tr.p1 );
				tr.homeQ.copy( tr.q1 );
				tr.homeOrbit = false;
				tr.keepTarget = false;

			}

			tr.f1 = this._orbitFov;
			tr.dur = ( this.reducedMotion ? 0.45 : 1.25 );

		}

		this.immersive = on;
		this._transition = tr;
		if ( ! animate ) this._finishTransition();
		this.app.events.dispatchEvent( new CustomEvent( 'immersive', { detail: { immersive: on } } ) );

	}

	/** Brings a running immersive transition to its end at once. */
	_finishTransition() {

		if ( ! this._transition ) return;
		this._tr.t = 1;
		this._stepTransition( 0 );

	}

	/** A pose of the immersive view (pose() while immersive): its look direction and field of view; enters the view when needed. */
	_setLookPose( pose, animate ) {

		_v3.fromArray( pose.target ).sub( _v.fromArray( pose.position ) );
		if ( _v3.lengthSq() < 1e-12 ) return Promise.resolve( false );
		_v3.normalize();
		const fov = clamp( Number.isFinite( pose.fov ) ? pose.fov : this._immersiveFov(), this.options.minFov, this.options.maxFov );
		if ( this.immersive ) {

			if ( ! animate ) this._finishTransition();
			return this._startLookTween( _v3, fov, animate );

		}

		this.setImmersive( true, { animate, direction: _v3, fov } );
		if ( ! this._transition ) return Promise.resolve( true );
		return new Promise( ( resolve ) => {

			this._lookArrive = resolve; // resolved when the transition ends (see _stepTransition)

		} );

	}

	// ── internals: orbit state ───────────────────────────────────────────────────────────

	/** Orbit state from the camera as it stands (pose written from outside, end of a transition). */
	_adoptCamera() {

		const o = this._o, g = this._g, v = this._vel, cam = this.camera;
		const p = orbitFromPose( cam.position, o.target );
		o.yaw = p.yaw; o.pitch = p.pitch; o.dist = p.distance;
		g.yaw = o.yaw; g.pitch = o.pitch; g.logDist = Math.log( o.dist ); g.target.copy( o.target );
		v.yaw = v.pitch = v.logDist = v.x = v.y = v.z = 0;
		this._fling.yaw = this._fling.pitch = 0;
		this._wPos.copy( cam.position );
		this._wTarget.copy( o.target );
		this._dirty = true;

	}

	_adoptExternal() {

		if ( this.immersive || this._transition || ! this.camera ) return false;
		if ( this._wPos.equals( this.camera.position ) && this._wTarget.equals( this._o.target ) ) return false;
		this._endFlight( false );
		this._follow = null;
		this._adoptCamera();
		this._writeCamera();
		return true;

	}

	_snapToGoal() {

		const o = this._o, g = this._g, v = this._vel;
		o.yaw = g.yaw; o.pitch = g.pitch; o.dist = Math.exp( g.logDist ); o.target.copy( g.target );
		v.yaw = v.pitch = v.logDist = v.x = v.y = v.z = 0;
		this._dirty = true;

	}

	_clearInertia() {

		const v = this._vel, g = this._g, o = this._o;
		this._fling.yaw = this._fling.pitch = 0;
		v.yaw = v.pitch = v.logDist = v.x = v.y = v.z = 0;
		g.yaw = o.yaw; g.pitch = o.pitch; g.logDist = Math.log( o.dist ); g.target.copy( o.target );

	}

	/** The camera follows the goal: critically damped, snapped once the rest is invisible. */
	_stepOrbit( dt ) {

		const o = this._o, g = this._g, v = this._vel;
		const quick = this.reducedMotion ? 0.4 : 1;
		const st = this.options.smoothTime * quick, zt = this.options.zoomSmoothTime * quick;
		g.pitch = clamp( g.pitch, - MAX_PITCH, MAX_PITCH );
		const c = this.controls;
		g.logDist = clamp( g.logDist, Math.log( Math.max( 1e-6, c.minDistance ) ), Math.log( Math.max( 1e-5, c.maxDistance ) ) );
		const logD = Math.log( o.dist );
		const eps = 1e-5;
		let moving = false;
		const step = ( cur, vel, goal, time, tol ) => {

			if ( Math.abs( goal - cur ) <= tol && Math.abs( vel ) <= tol * 4 ) return ( _sd[ 0 ] = goal, _sd[ 1 ] = 0, false );
			smoothDamp( cur, vel, goal, time, dt, _sd );
			return true;

		};

		if ( step( o.yaw, v.yaw, g.yaw, st, eps ) ) moving = true;
		const yaw = _sd[ 0 ]; v.yaw = _sd[ 1 ];
		if ( step( o.pitch, v.pitch, g.pitch, st, eps ) ) moving = true;
		const pitch = _sd[ 0 ]; v.pitch = _sd[ 1 ];
		if ( step( logD, v.logDist, g.logDist, zt, eps ) ) moving = true;
		const ld = _sd[ 0 ]; v.logDist = _sd[ 1 ];
		const tol = o.dist * 1e-5;
		// the target follows with the slower of the two times, so a zoom to the cursor stays put
		if ( step( o.target.x, v.x, g.target.x, zt, tol ) ) moving = true;
		const tx = _sd[ 0 ]; v.x = _sd[ 1 ];
		if ( step( o.target.y, v.y, g.target.y, zt, tol ) ) moving = true;
		const ty = _sd[ 0 ]; v.y = _sd[ 1 ];
		if ( step( o.target.z, v.z, g.target.z, zt, tol ) ) moving = true;
		const tz = _sd[ 0 ]; v.z = _sd[ 1 ];
		if ( yaw !== o.yaw || pitch !== o.pitch || ld !== logD || tx !== o.target.x || ty !== o.target.y || tz !== o.target.z ) {

			o.yaw = yaw; o.pitch = pitch; o.dist = Math.exp( ld ); o.target.set( tx, ty, tz );
			this._dirty = true;

		}

		return moving;

	}

	_stepInertia( dt ) {

		const f = this._fling, g = this._g;
		if ( f.yaw === 0 && f.pitch === 0 ) return;
		g.yaw += f.yaw * dt;
		g.pitch += f.pitch * dt;
		const k = Math.exp( - dt * this.options.friction );
		f.yaw *= k; f.pitch *= k;
		if ( Math.abs( f.yaw ) < 2e-3 ) f.yaw = 0;
		if ( Math.abs( f.pitch ) < 2e-3 ) f.pitch = 0;

	}

	_writeCamera() {

		const o = this._o, cam = this.camera;
		if ( ! this._dirty ) return;
		this._dirty = false;
		orbitToPosition( o.target, o.yaw, o.pitch, o.dist, cam.position );
		cam.lookAt( o.target );
		this._wPos.copy( cam.position );
		this._wTarget.copy( o.target );

	}

	/** A camera gesture of the user: the rig stops framing by itself, flights end. */
	_userGesture() {

		this._taken = true;
		this._follow = null;
		this._endFlight( false );
		this.notifyInteraction();

	}

	// ── internals: framing, flight, follow ───────────────────────────────────────────────

	_refreshSphere( force ) {

		const app = this.app;
		if ( ! app || ! app.graph ) return;
		if ( ! force && ( this._time - this._sphereTime < 1 || app.positionsVersion === this._sphereVersion ) ) return;
		this._sphereTime = this._time;
		this._sphereVersion = app.positionsVersion;
		const p = app.positions;
		if ( p && app.graph.N > 0 ) {

			// tighter than app.boundingSphere() (centroid + percentile) for lopsided graphs
			const ball = enclosingSphere( p, app.graph.N, undefined, this._ball );
			this._sphere.center.fromArray( ball.center );
			this._sphere.radius = Math.max( 1, ball.radius );

		} else app.boundingSphere( this._sphere );

		const u = this.buffers ? this.buffers.uniforms : null;
		if ( u && ( this._layoutMode === 'sphere' || u.layoutBlend.value > 0.5 ) ) {

			// the paper's layout: the guide sphere is the extent, wherever the nodes are on their way
			this._sphere.center.copy( u.graphCenter.value );
			this._sphere.radius = Math.max( 1, u.sphereRadius.value * 1.03 );

		}

		this._center.copy( this._sphere.center );
		this._radius = Math.max( 1e-3, this._sphere.radius );

	}

	_limits() {

		const c = this.controls, r = this._radius;
		c.minDistance = r * 0.01;
		c.maxDistance = Math.max( c.maxDistance === Infinity ? 0 : c.maxDistance, r * 30 );
		if ( c.maxDistance > r * 300 ) c.maxDistance = r * 30;

	}

	/** Room around the bounding sphere of the nodes. In the sphere layout the bundles arc outward behind the nodes: they belong in the picture. */
	_padding() {

		const u = this.buffers ? this.buffers.uniforms : null;
		return this._layoutMode === 'sphere' || ( u && u.layoutBlend.value > 0.5 ) ? 1.4 : 1.12;

	}

	_fit( { animate = true, padding = null, direction = null, duration = null, after = 0 } = {} ) {

		const s = this._sphere;
		const dist = this.fitDistance( s.radius * ( padding ?? this._padding() ) );
		this._limits();
		return this._fly( s.center, dist, { animate, direction, duration, after } );

	}

	/**
	 * Flight to ( point, distance ) and optionally a new direction ( view direction vector, or
	 * yaw / pitch ) and fov. Does not change who owns the framing (see flyTo for the public one).
	 */
	_fly( point, distance, { duration = null, direction = null, track = - 1, animate = true, yaw = null, pitch = null, fov = null, wrap = true, arc = true, after = 0 } = {} ) {

		const o = this._o, f = this._fl;
		// a flight starts from the orbit: a leave of the immersive view that is under way ends first
		if ( this._transition && ! this.immersive ) this._finishTransition();
		this._endFlight( false );
		f.t0.copy( o.target );
		f.t1.copy( point );
		f.y0 = o.yaw; f.p0 = o.pitch; f.d0 = Math.max( 1e-6, o.dist );
		f.d1 = distance && distance > 0 ? distance : f.d0;
		if ( direction ) {

			_v.copy( direction ).negate().normalize(); // from the target to the camera
			yaw = Math.atan2( _v.x, _v.z );
			pitch = clamp( Math.asin( clamp( _v.y, - 1, 1 ) ), - MAX_PITCH, MAX_PITCH );

		}

		f.y1 = yaw === null ? f.y0 : ( wrap ? f.y0 + shortestAngle( yaw - f.y0 ) : yaw );
		f.p1 = pitch === null ? f.p0 : pitch;
		f.f0 = this.camera.fov;
		f.f1 = fov === null ? f.f0 : fov;
		f.track = track;
		f.arc = arc && ! this.reducedMotion;
		f.after = after;
		const travel = f.t0.distanceTo( f.t1 );
		const ref = Math.max( f.d0, f.d1 );
		const turn = Math.abs( f.y1 - f.y0 ) + Math.abs( f.p1 - f.p0 );
		const auto = MathUtils.clamp( 0.75 + 0.22 * Math.log2( 1 + travel / ref + Math.abs( Math.log2( f.d1 / f.d0 ) ) * 0.5 + turn * 0.6 ), 0.75, 1.8 );
		f.dur = ( duration ?? auto ) * ( this.reducedMotion ? 0.35 : 1 );
		f.t = 0;
		this._follow = null;
		this._fling.yaw = this._fling.pitch = 0;
		const promise = new Promise( ( resolve ) => {

			f.resolve = resolve;

		} );
		this._flight = f;
		if ( ! animate || f.dur <= 0.01 ) {

			f.t = 1;
			this._applyFlight();
			this._writeCamera();
			this._endFlight( true );

		}

		return promise;

	}

	_endFlight( completed ) {

		const f = this._flight;
		if ( ! f ) return;
		this._flight = null;
		const resolve = f.resolve;
		f.resolve = null;
		this._clearInertia();
		if ( completed && f.after > 0 ) this.keepFramed( f.after );
		resolve?.( completed );

	}

	_stepFlight( dt ) {

		const f = this._fl;
		f.t = Math.min( 1, f.t + dt / f.dur );
		if ( f.track >= 0 ) {

			const p = this.app.positions;
			if ( p && 4 * f.track + 2 < p.length ) f.t1.set( p[ 4 * f.track ], p[ 4 * f.track + 1 ], p[ 4 * f.track + 2 ] );

		}

		this._applyFlight();
		if ( f.t >= 1 ) this._endFlight( true );

	}

	_tanHalfFree() {

		const cam = this.camera, f = this._freeFraction();
		const t = Math.tan( MathUtils.degToRad( this._orbitFov ) / 2 );
		return Math.min( t * f.h, t * cam.aspect * f.w );

	}

	_applyFlight() {

		const f = this._fl, o = this._o, cam = this.camera;
		const e = easeInOutCubic( f.t );
		o.target.lerpVectors( f.t0, f.t1, e );
		o.yaw = lerp( f.y0, f.y1, e );
		o.pitch = lerp( f.p0, f.p1, e );
		o.dist = arcDistance( e, f.d0, f.d1, f.arc ? f.t0.distanceTo( f.t1 ) : 0, this._tanHalfFree() );
		if ( f.f1 !== f.f0 ) {

			cam.fov = lerp( f.f0, f.f1, e );
			this._orbitFov = cam.fov;
			cam.updateProjectionMatrix();

		}

		this._clearInertia(); // the goal rides along
		this._dirty = true;

	}

	_stepFollow( dt ) {

		const fo = this._follow, o = this._o, g = this._g;
		if ( this._time > fo.until ) {

			this._follow = null;
			return;

		}

		if ( this._time >= this._followNext ) {

			this._followNext = this._time + 0.3;
			this._refreshSphere( true );
			this._followCenter.copy( this._sphere.center );
			this._followDist = this.fitDistance( this._sphere.radius * this._padding() );
			this._limits();

		}

		const k = 1 - Math.exp( - dt * 2.2 );
		g.target.lerp( this._followCenter, k );
		g.logDist += ( Math.log( this._followDist ) - g.logDist ) * k;
		// the follow is already an eased motion: the camera takes it without a second lag
		o.target.copy( g.target );
		o.dist = Math.exp( g.logDist );
		this._vel.logDist = this._vel.x = this._vel.y = this._vel.z = 0;
		this._dirty = true;

	}

	/** 'layout' event: refit softly on a new mode and on settle, unless the user has the camera. */
	_layoutEvent( { mode, settled } ) {

		const modeChanged = mode !== undefined && mode !== this._layoutMode && this._layoutMode !== null;
		const settledNow = !! settled && ! this._layoutSettled;
		if ( mode !== undefined ) this._layoutMode = mode;
		this._layoutSettled = !! settled;
		if ( this.immersive || this._transition || this._taken || ! this.graph ) return;
		if ( modeChanged ) {

			this._sphereVersion = - 1;
			if ( mode === 'flat' || mode === 'sphere' ) {

				// face the plane of the flat layout / the front of the sphere, then follow the nodes
				this._refreshSphere( true );
				this._fit( { direction: _v2.set( 0, 0, - 1 ), duration: 1.5, after: 5 } );

			} else {

				this.keepFramed( 6 );

			}

		} else if ( settledNow && ! this._flight ) {

			this._softRefit();

		}

	}

	/** Corrects the framing when it is visibly off (after the layout came to rest). */
	_softRefit() {

		this._refreshSphere( true );
		const s = this._sphere, g = this._g;
		const want = this.fitDistance( s.radius * this._padding() );
		const off = g.target.distanceTo( s.center ) / Math.max( 1e-6, s.radius );
		if ( Math.abs( Math.log( want / Math.exp( g.logDist ) ) ) < 0.06 && off < 0.08 ) return;
		this._limits();
		this._fly( s.center, want, { duration: 1.1, arc: false } );

	}

	_stepAutoRotate( dt ) {

		const want = this.autoRotate && ! this.immersive && ! this._transition && ! this._flight && ! this._inputLocked &&
			this._idle >= this.idleDelay && this._pointers.size === 0 && !! this.app.graph;
		// in: a slow ramp; out: a quick ease, so an input never meets a camera that snaps still
		this._autoAmount = want ? Math.min( 1, this._autoAmount + dt / this.options.autoRotateRamp ) : Math.max( 0, this._autoAmount - dt * 4 );
		if ( this._autoAmount > 0 && ! this.immersive && ! this._transition && ! this._flight ) {

			const rate = this.autoRotateSpeed * ( ( 2 * Math.PI ) / 60 ) * smooth( this._autoAmount );
			this._g.yaw += rate * dt;
			this._o.yaw += rate * dt;
			this._dirty = true;

		}

		const active = want && this._autoAmount > 0;
		if ( active !== this.autoRotating ) {

			this.autoRotating = active;
			this.app.events.dispatchEvent( new CustomEvent( 'autorotate', { detail: { active } } ) );

		}

		// cinematic idle: when nobody has touched anything for a long time, the turn becomes a slow
		// path on which the camera also rises, sinks and moves in and out, with a focus pull for the lens
		const O = this.options;
		const drift = want && O.cinematic && ! this.reducedMotion && ! this._follow && this._idle >= this.idleDelay + O.cinematicDelay;
		this._driftAmount = drift ? Math.min( 1, this._driftAmount + dt / O.cinematicRamp ) : Math.max( 0, this._driftAmount - dt * 4 );
		if ( this._driftAmount > 0 && this._autoAmount > 0 && ! this.immersive && ! this._transition && ! this._flight ) {

			const g = this._g, o = this._o, c = this.controls;
			const w = smooth( this._driftAmount ), r = driftRates( ( this._driftTime += dt ), this._driftRates );
			let dp = r.pitch * w * dt, dz = r.zoom * w * dt;
			// never push further out of a comfortable range than the person had gone
			if ( Math.abs( g.pitch + dp ) > 1.1 && Math.abs( g.pitch + dp ) > Math.abs( g.pitch ) ) dp = 0;
			if ( g.logDist + dz < Math.log( c.minDistance * 4 ) || g.logDist + dz > Math.log( c.maxDistance ) ) dz = 0;
			g.pitch += dp;
			o.pitch += dp;
			g.logDist += dz;
			o.dist *= Math.exp( dz );
			this.driftFocus.distance = Math.max( 1e-3, o.dist + r.focus * w * 0.6 * this._radius );
			this.driftFocus.range = 1.3 * this._radius;

		} else if ( this._driftAmount === 0 ) this._driftTime = 0;

		const drifting = drift && this._driftAmount > 0;
		if ( drifting !== this.drifting ) {

			this.drifting = drifting;
			this.app.events.dispatchEvent( new CustomEvent( 'cinematic', { detail: { active: drifting } } ) );

		}

	}

	_stepInsets( dt ) {

		const a = this._insets, c = this._insetCur;
		let moving = false;
		const k = 1 - Math.exp( - dt * ( this.reducedMotion ? 30 : 7 ) );
		for ( const key of [ 'left', 'right', 'top', 'bottom' ] ) {

			const d = a[ key ] - c[ key ];
			if ( Math.abs( d ) < 0.25 ) {

				if ( d !== 0 ) moving = true;
				c[ key ] = a[ key ];

			} else {

				c[ key ] += d * k;
				moving = true;

			}

		}

		if ( moving || this._insetsDirty ) this._applyViewOffset();

	}

	_applyViewOffset() {

		const cam = this.camera, app = this.app;
		if ( ! cam || ! app || ! app.width ) return;
		this._insetsDirty = false;
		const c = this._insetCur;
		const ox = ( c.left - c.right ) / 2, oy = ( c.top - c.bottom ) / 2;
		if ( Math.abs( ox ) < 0.25 && Math.abs( oy ) < 0.25 ) {

			if ( cam.view && cam.view.enabled ) cam.clearViewOffset();
			return;

		}

		cam.setViewOffset( app.width, app.height, - ox, - oy, app.width, app.height );

	}

	// ── internals: immersive ─────────────────────────────────────────────────────────────

	_immersiveCenter( out ) {

		return this.buffers ? out.copy( this.buffers.uniforms.graphCenter.value ) : out.set( 0, 0, 0 );

	}

	/** Direction from the sphere centre toward the middle of the layout (node centroid), else +Z. */
	_layoutDirection( out ) {

		this._immersiveCenter( _v2 );
		out.copy( this._center ).sub( _v2 );
		if ( out.lengthSq() < ( this._radius * 0.05 ) ** 2 ) out.set( 0, 0, 1 );
		return out.normalize();

	}

	_lookDirection( out ) {

		const { yaw, pitch } = this._look;
		return out.set( Math.sin( yaw ) * Math.cos( pitch ), Math.sin( pitch ), Math.cos( yaw ) * Math.cos( pitch ) );

	}

	_immersiveFov() {

		const a = this.camera.aspect || 1;
		const v = MathUtils.radToDeg( 2 * Math.atan( Math.tan( Math.PI / 4 ) / a ) ); // ≥ 90° horizontally
		return MathUtils.clamp( Math.max( 75, v ), 75, 110 );

	}

	/** Turns the immersive look toward `dir` with field of view `fov`. Resolves true on arrival, false when an input or another request took over. */
	_startLookTween( dir, fov, animate = true ) {

		const L = this._look;
		const yaw1 = Math.atan2( dir.x, dir.z );
		const pitch1 = MathUtils.clamp( Math.asin( MathUtils.clamp( dir.y, - 1, 1 ) ), - 1.396, 1.396 );
		const dy = shortestAngle( yaw1 - L.yaw );
		this._dropLookTween();
		L.vyaw = L.vpitch = 0;
		if ( ! animate ) {

			L.yaw += dy; L.pitch = pitch1; L.fov = L.fovTarget = fov;
			if ( ! this._transition ) this._writeLook(); // at once, like the orbit's poses
			return Promise.resolve( true );

		}

		this._lookTween = { t: 0, dur: this.reducedMotion ? 0.3 : 0.9, y0: L.yaw, y1: L.yaw + dy, p0: L.pitch, p1: pitch1, f0: L.fovTarget, f1: fov };
		return new Promise( ( resolve ) => {

			this._lookArrive = resolve;

		} );

	}

	/** Ends a look that an input or a new request replaces; whoever awaits it hears false. */
	_dropLookTween( arrived = false ) {

		this._lookTween = null;
		const resolve = this._lookArrive;
		this._lookArrive = null;
		resolve?.( arrived );

	}

	/** The camera of the immersive view: at the centre, along the look. */
	_writeLook() {

		const L = this._look, cam = this.camera;
		this._immersiveCenter( cam.position );
		cam.quaternion.copy( lookQuaternion( this._lookDirection( this._dir ), _q ) );
		if ( Math.abs( cam.fov - L.fov ) > 1e-3 ) {

			cam.fov = L.fov;
			cam.updateProjectionMatrix();

		}

	}

	_stepImmersive( dt ) {

		const L = this._look;
		const tw = this._lookTween;
		if ( tw ) {

			tw.t = Math.min( 1, tw.t + dt / tw.dur );
			const e = easeInOutCubic( tw.t );
			L.yaw = MathUtils.lerp( tw.y0, tw.y1, e );
			L.pitch = MathUtils.lerp( tw.p0, tw.p1, e );
			L.fovTarget = MathUtils.lerp( tw.f0, tw.f1, e );
			if ( tw.t >= 1 ) this._dropLookTween( true );

		} else if ( ! this._lookDrag ) {

			// inertia
			L.yaw += L.vyaw * dt;
			L.pitch += L.vpitch * dt;
			const k = Math.exp( - dt * 4.5 );
			L.vyaw *= k; L.vpitch *= k;
			if ( Math.abs( L.vyaw ) < 1e-4 ) L.vyaw = 0;
			if ( Math.abs( L.vpitch ) < 1e-4 ) L.vpitch = 0;

		}

		L.pitch = MathUtils.clamp( L.pitch, - 1.396, 1.396 ); // ±80°
		L.fov += ( L.fovTarget - L.fov ) * ( 1 - Math.exp( - dt * 12 ) );
		this._writeLook();

	}

	_stepTransition( dt ) {

		const tr = this._tr, cam = this.camera;
		tr.t = Math.min( 1, tr.t + dt / tr.dur );
		const e = easeInOutCubic( tr.t );
		const C = this._immersiveCenter( this._camCenter );
		if ( tr.dir === 'in' ) {

			slerpUnit( tr.u0, tr.u1, e, _v );
			cam.position.copy( C ).addScaledVector( _v, tr.d0 * ( 1 - e ) );

		} else {

			cam.position.lerpVectors( tr.p0, tr.p1, e );

		}

		cam.quaternion.slerpQuaternions( tr.q0, tr.q1, e );
		cam.fov = MathUtils.lerp( tr.f0, tr.f1, e );
		cam.updateProjectionMatrix();
		if ( tr.t < 1 ) return;

		this._transition = null;
		if ( tr.dir === 'out' ) {

			// back in the orbit: around the centre, or around the target of the orbit a reversed entry left
			if ( ! tr.keepTarget ) this._o.target.copy( C );
			this._adoptCamera();
			this._limits();
			this._idle = 0;
			// the leave ends at the distance of a fit; where the centre of the nodes is not the centre
			// of the sphere, the framing finishes the rest softly
			if ( ! tr.keepTarget && ! this._taken ) this.keepFramed( 2.5 );

		} else if ( ! this._lookTween ) this._dropLookTween( true ); // arrived at the look that was asked for

	}

	// ── internals: pointer input ─────────────────────────────────────────────────────────

	_updateRect() {

		if ( ! this.canvas || ! this.canvas.getBoundingClientRect ) return;
		const r = this.canvas.getBoundingClientRect();
		this._rect.left = r.left; this._rect.top = r.top;
		this._rect.width = Math.max( 1, r.width ); this._rect.height = Math.max( 1, r.height );

	}

	/** World-space ray direction through a client point (honours the view offset of the insets). */
	_screenRay( clientX, clientY, out ) {

		const cam = this.camera, r = this._rect;
		cam.updateMatrixWorld();
		out.set( ( ( clientX - r.left ) / r.width ) * 2 - 1, - ( ( clientY - r.top ) / r.height ) * 2 + 1, 0.5 ).unproject( cam );
		return out.sub( cam.position ).normalize();

	}

	/**
	 * Multiplies the goal distance by `factor`. With a client point, the goal target moves so
	 * that what is under that point stays there (zoom to cursor / pinch centre).
	 */
	_zoomGoal( factor, point ) {

		const g = this._g, c = this.controls, cam = this.camera;
		if ( ! c.enableZoom ) return;
		const dOld = Math.exp( g.logDist );
		const dNew = clamp( dOld * factor, Math.max( 1e-6, c.minDistance ), c.maxDistance );
		const s = dNew / dOld;
		if ( s === 1 ) return;
		if ( point ) {

			const ray = this._screenRay( point.x, point.y, _v );
			cam.getWorldDirection( _v2 );
			const cos = ray.dot( _v2 );
			if ( cos > 1e-3 ) {

				// the point under the cursor on the plane through the target, perpendicular to the view
				const depth = _v3.copy( this._o.target ).sub( cam.position ).dot( _v2 );
				_v3.copy( cam.position ).addScaledVector( ray, depth / cos );
				g.target.sub( _v3 ).multiplyScalar( s ).add( _v3 );

			}

		}

		g.logDist = Math.log( dNew );

	}

	/** Moves the goal target in the view plane by a drag of ( dx, dy ) CSS pixels. */
	_panGoal( dx, dy ) {

		const g = this._g, cam = this.camera;
		if ( ! this.controls.enablePan ) return;
		const perPixel = ( 2 * Math.tan( MathUtils.degToRad( cam.fov ) / 2 ) * Math.exp( g.logDist ) ) / this._rect.height;
		const cy = Math.cos( g.yaw ), sy = Math.sin( g.yaw ), cp = Math.cos( g.pitch ), sp = Math.sin( g.pitch );
		// camera basis of the goal orbit: right = ( cos yaw, 0, −sin yaw ), up = ( −sin yaw sin pitch, cos pitch, −cos yaw sin pitch )
		g.target.x += ( - dx * cy + dy * ( - sy * sp ) ) * perPixel;
		g.target.y += dy * cp * perPixel;
		g.target.z += ( dx * sy + dy * ( - cy * sp ) ) * perPixel;

	}

	_rotateGoal( dx, dy, touch ) {

		if ( ! this.controls.enableRotate ) return;
		const k = ( ( 2 * Math.PI ) / this._rect.height ) * ( touch ? 0.9 : this.options.rotateSpeed );
		this._g.yaw -= dx * k;
		this._g.pitch = clamp( this._g.pitch + dy * k, - MAX_PITCH, MAX_PITCH );
		return k;

	}

	_pointerDown( e ) {

		this._updateRect();
		// a primary touch is the only finger down: forget pointers whose release never arrived
		if ( e.isPrimary && e.pointerType === 'touch' ) this._pointers.clear();
		this._pointers.set( e.pointerId, { x: e.clientX, y: e.clientY, type: e.pointerType } );
		this.notifyInteraction();
		if ( this._transition ) return;
		if ( this.immersive ) {

			if ( ! this._inputLocked ) this._beginImmersiveGesture( e );
			return;

		}

		if ( ! this.controls.enabled ) return;
		this._fling.yaw = this._fling.pitch = 0;
		try {

			this.canvas.setPointerCapture( e.pointerId );

		} catch ( _ ) { /* synthetic pointer */ }

		if ( this._pointers.size >= 2 ) {

			this._beginPinch();
			return;

		}

		if ( this._inputLocked ) return; // a press on a node: the pointer is tracked, nothing turns
		let mode = 'rotate';
		if ( e.pointerType === 'mouse' ) {

			if ( e.button === 2 || ( e.button === 0 && ( e.ctrlKey || e.metaKey || e.shiftKey ) ) ) mode = 'pan';
			else if ( e.button === 1 ) mode = 'dolly';
			else if ( e.button !== 0 ) return;
			if ( e.button === 1 ) e.preventDefault(); // no auto-scroll cursor

		}

		this._gesture = { mode, id: e.pointerId, x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, t: performance.now(), vx: 0, vy: 0, moved: false, touch: e.pointerType === 'touch' };

	}

	_beginPinch() {

		const [ a, b ] = [ ...this._pointers.values() ];
		this._gesture = { mode: 'pinch', d: Math.max( 1, Math.hypot( a.x - b.x, a.y - b.y ) ), cx: ( a.x + b.x ) / 2, cy: ( a.y + b.y ) / 2, moved: true };
		this._userGesture();

	}

	_beginImmersiveGesture( e ) {

		const L = this._look;
		this._dropLookTween();
		if ( this._pointers.size === 1 ) {

			try {

				this.canvas.setPointerCapture( e.pointerId );

			} catch ( _ ) { /* synthetic pointer */ }

			this._lookDrag = { id: e.pointerId, x: e.clientX, y: e.clientY, t: performance.now() };
			L.vyaw = L.vpitch = 0;
			this._pinch = null;

		} else if ( this._pointers.size === 2 ) {

			const [ a, b ] = [ ...this._pointers.values() ];
			this._pinch = { d: Math.max( 1, Math.hypot( a.x - b.x, a.y - b.y ) ), fov: L.fovTarget };
			this._lookDrag = null;

		}

	}

	_pointerMove( e ) {

		const p = this._pointers.get( e.pointerId );
		if ( ! p ) return;
		p.x = e.clientX; p.y = e.clientY;
		if ( this._transition ) return;
		if ( this.immersive ) {

			if ( ! this._inputLocked ) this._immersiveMove( e );
			return;

		}

		const g = this._gesture;
		if ( ! g || ! this.controls.enabled ) return;
		if ( g.mode === 'pinch' ) {

			if ( this._pointers.size < 2 ) return;
			const [ a, b ] = [ ...this._pointers.values() ];
			const d = Math.max( 1, Math.hypot( a.x - b.x, a.y - b.y ) );
			const cx = ( a.x + b.x ) / 2, cy = ( a.y + b.y ) / 2;
			this._panGoal( cx - g.cx, cy - g.cy );
			this._zoomGoal( g.d / d, { x: cx, y: cy } );
			g.d = d; g.cx = cx; g.cy = cy;
			this.notifyInteraction();
			return;

		}

		if ( g.id !== e.pointerId || this._inputLocked ) return;
		if ( ! g.moved ) {

			// a click is not a camera gesture: start past a small threshold, from where the pointer is now
			if ( Math.hypot( e.clientX - g.x0, e.clientY - g.y0 ) < ( g.touch ? 8 : 3 ) ) return;
			g.moved = true;
			g.x = e.clientX; g.y = e.clientY; g.t = performance.now();
			this._userGesture();
			return;

		}

		const now = performance.now();
		const dx = e.clientX - g.x, dy = e.clientY - g.y;
		const dts = Math.max( 1e-3, ( now - g.t ) / 1000 );
		g.x = e.clientX; g.y = e.clientY; g.t = now;
		if ( g.mode === 'rotate' ) {

			const k = this._rotateGoal( dx, dy, g.touch );
			if ( k ) {

				g.vx = lerp( g.vx, ( - dx * k ) / dts, 0.35 );
				g.vy = lerp( g.vy, ( dy * k ) / dts, 0.35 );

			}

		} else if ( g.mode === 'pan' ) this._panGoal( dx, dy );
		else if ( g.mode === 'dolly' ) this._zoomGoal( Math.exp( dy * 0.006 ), null );
		this.notifyInteraction();

	}

	_immersiveMove( e ) {

		const L = this._look;
		if ( this._pinch && this._pointers.size >= 2 ) {

			const [ a, b ] = [ ...this._pointers.values() ];
			const d = Math.max( 1, Math.hypot( a.x - b.x, a.y - b.y ) );
			L.fovTarget = MathUtils.clamp( this._pinch.fov * this._pinch.d / d, this.options.minFov, this.options.maxFov );
			this.notifyInteraction();
			return;

		}

		const ld = this._lookDrag;
		if ( ! ld || ld.id !== e.pointerId ) return;
		const now = performance.now();
		const k = MathUtils.degToRad( this.camera.fov ) / Math.max( 1, this.app.height ); // content follows the pointer
		const dyaw = ( e.clientX - ld.x ) * k, dpitch = ( e.clientY - ld.y ) * k;
		L.yaw += dyaw;
		L.pitch = MathUtils.clamp( L.pitch + dpitch, - 1.396, 1.396 );
		const dts = Math.max( 1e-3, ( now - ld.t ) / 1000 );
		const a = 0.35; // velocity smoothing
		L.vyaw = MathUtils.lerp( L.vyaw, dyaw / dts, a );
		L.vpitch = MathUtils.lerp( L.vpitch, dpitch / dts, a );
		ld.x = e.clientX; ld.y = e.clientY; ld.t = now;
		this.notifyInteraction();

	}

	_pointerUp( e ) {

		if ( ! this._pointers.delete( e.pointerId ) ) return;
		const L = this._look;
		const ld = this._lookDrag;
		if ( ld && ld.id === e.pointerId ) {

			// keep inertia only if the pointer was still moving when released
			if ( performance.now() - ld.t > 90 ) L.vyaw = L.vpitch = 0;
			const cap = 4;
			L.vyaw = MathUtils.clamp( L.vyaw, - cap, cap );
			L.vpitch = MathUtils.clamp( L.vpitch, - cap, cap );
			this._lookDrag = null;

		}

		if ( this._pinch && this._pointers.size < 2 ) {

			this._pinch = null;
			// continue looking with the remaining finger
			if ( this._pointers.size === 1 && this.immersive && ! this._inputLocked ) {

				const [ [ id, q ] ] = [ ...this._pointers.entries() ];
				this._lookDrag = { id, x: q.x, y: q.y, t: performance.now() };
				L.vyaw = L.vpitch = 0;

			}

		}

		const g = this._gesture;
		if ( ! g ) return;
		if ( g.mode === 'pinch' ) {

			// the finger that stays keeps rotating, from where it is
			this._gesture = null;
			if ( this._pointers.size === 1 && ! this._inputLocked && ! this.immersive ) {

				const [ [ id, q ] ] = [ ...this._pointers.entries() ];
				this._gesture = { mode: 'rotate', id, x: q.x, y: q.y, x0: q.x, y0: q.y, t: performance.now(), vx: 0, vy: 0, moved: true, touch: q.type === 'touch' };

			}

			return;

		}

		if ( g.id !== e.pointerId ) return;
		this._gesture = null;
		if ( g.mode === 'rotate' && g.moved && ! this.reducedMotion && e.type === 'pointerup' && performance.now() - g.t < 90 ) {

			// released in motion: the goal keeps turning and friction brings it to rest
			const cap = 5;
			this._fling.yaw = clamp( g.vx, - cap, cap );
			this._fling.pitch = clamp( g.vy, - cap, cap );

		}

	}

	_wheel( e ) {

		this.notifyInteraction();
		e.preventDefault();
		if ( this._transition ) return;
		if ( this.immersive ) {

			const L = this._look;
			this._dropLookTween();
			let dy = e.deltaY * ( e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1 );
			if ( e.ctrlKey ) dy *= 6; // trackpad pinch
			L.fovTarget = MathUtils.clamp( L.fovTarget * Math.exp( dy * 0.0012 ), this.options.minFov, this.options.maxFov );
			return;

		}

		if ( ! this.controls.enabled ) return;
		this._updateRect();
		this._userGesture();
		this._zoomGoal( wheelZoomFactor( e.deltaY, e.deltaMode, e.ctrlKey ), { x: e.clientX, y: e.clientY } );

	}

	// ── internals: keyboard ──────────────────────────────────────────────────────────────

	_keyDown( e ) {

		this.notifyInteraction();
		if ( ! this.options.keys || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || ! this.graph ) return;
		// only while nothing else wants the keys: a focused slider, list or field keeps its arrows
		const t = e.target;
		if ( t && t !== this.canvas && t.closest && t.closest( KEY_TARGETS ) ) return;
		const step = ( e.shiftKey ? 5 : 15 ) * ( Math.PI / 180 );
		let used = true;
		switch ( e.key ) {

			case 'ArrowLeft': this._keyTurn( - step, 0 ); break;
			case 'ArrowRight': this._keyTurn( step, 0 ); break;
			case 'ArrowUp': this._keyTurn( 0, step ); break;
			case 'ArrowDown': this._keyTurn( 0, - step ); break;
			case '+': case '=': this.zoomBy( 1 / 1.25 ); break;
			case '-': case '_': this.zoomBy( 1.25 ); break;
			case 'Home': this.fitToGraph(); break;
			default: used = false;

		}

		if ( used ) e.preventDefault();

	}

	_keyTurn( yaw, pitch ) {

		if ( this._transition ) return;
		if ( this.immersive ) {

			const L = this._look;
			this._dropLookTween();
			L.yaw -= yaw * 0.6;
			L.pitch = clamp( L.pitch + pitch * 0.6, - 1.396, 1.396 );
			return;

		}

		if ( ! this.controls.enabled || ! this.controls.enableRotate ) return;
		this._userGesture();
		this._fling.yaw = this._fling.pitch = 0;
		this._g.yaw += yaw;
		this._g.pitch = clamp( this._g.pitch + pitch, - MAX_PITCH, MAX_PITCH );

	}

}
