// Camera control (App module, order 31): tuned orbit controls, fit / fly-to / focus animations,
// idle auto-rotate and the immersive "inside the sphere" view of the paper's spherical layout.
//
// Design
//  • Orbit mode uses three's OrbitControls (App's `app.controls` is taken over, or one is created
//    and assigned to `app.controls`, so App keeps calling controls.update(dt) after the modules).
//    Damping, zoom-to-cursor, screen-space panning; rotate/pan speeds are switched per pointer
//    type on pointerdown (mouse vs. touch); touch: one finger rotates, two fingers pinch + pan.
//  • Animations (fitToGraph, flyTo, focusNode) are tweens of (target, distance, view direction):
//    target moves linearly, distance geometrically (log space) with a small "hop" when the target
//    travels far, view direction by spherical interpolation; ease-in-out cubic. While a tween
//    runs OrbitControls' inertia is cleared; any camera gesture cancels the tween.
//    focusNode(i) keeps tracking the node during the flight (it may move under the layout).
//  • Near/far planes follow the camera distance to the graph every frame (graph radius from
//    app.boundingSphere(), refreshed ~1×/s), so 34-node and 20k-node graphs both clip correctly.
//  • Auto-rotate: slow azimuthal drift, ramped in smoothly; stops on any interaction (pointer,
//    wheel, key, camera gesture) and resumes after `idleDelay` (8 s) of idle time. Off by default
//    when the user prefers reduced motion (`rig.reducedMotion`); an explicit setAutoRotate(true)
//    still wins. Reduced motion also shortens all camera animations.
//  • keepFramed(s): for s seconds (and until the user touches the camera) the view gently follows
//    the bounding sphere while the force layout expands/settles. Called automatically on setGraph.
//  • Immersive mode (paper's VR view adapted to screens): the camera sits at the sphere centre
//    (buffers.uniforms.graphCenter) and looks around: drag = yaw/pitch with inertia (pitch clamped
//    to ±80°), wheel/pinch = field of view (30–115°). Entering swings the camera around behind the
//    centre while dollying in and turning toward the layout's middle (direction from the centre
//    to the node centroid, +Z for prepare.js' projection); leaving pulls straight back along the
//    current view direction. OrbitControls is disabled meanwhile (its update() is bypassed).
//  • View insets: UI chrome (the settings panel, the phone bottom sheet) covers part of the
//    canvas. setInsets({left, right, top, bottom}) shifts the projection centre (camera view
//    offset, eased) so the graph is centred in the free area, and fitting uses the free area.
//  • Interaction (order 30) calls setInputLocked(true) while a press that started on a node is in
//    progress: rotation/pan/look are blocked but pointers stay tracked, so when a second finger
//    lands the lock is released and the pinch goes straight to the camera.
//
// API
//   const rig = app.use( new CameraRig( { orbitControls, autoRotate = true, autoRotateSpeed = 0.45,
//                                          idleDelay = 8, followSeconds = 8 } ) )
//   rig.controls                       OrbitControls
//   rig.fitToGraph( { animate = true, padding = 1.1, direction } )   direction = view direction
//   rig.flyTo( point, distance?, { duration, direction, track, animate } )
//   rig.focusNode( index, { animate = true, distance } )
//   rig.keepFramed( seconds )
//   rig.setAutoRotate( on ) / rig.autoRotate / rig.autoRotating (currently spinning)
//   rig.setImmersive( on, { animate = true } ) / rig.toggleImmersive() / rig.immersive
//   rig.notifyInteraction()            resets the idle timer (stops auto-rotate)
//   rig.setInputLocked( on )           used by Interaction during node presses/drags
//   rig.setInsets( { left, right, top, bottom } )   CSS px covered by UI (see Panel onInsets)
//   rig.busy                           true while a camera animation runs
//   rig.reducedMotion                  prefers-reduced-motion: reduce
// Events on app.events: 'immersive' {immersive}, 'autorotate' {active}.

import { Vector3, Quaternion, Matrix4, MathUtils, Sphere, TOUCH } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const UP = new Vector3( 0, 1, 0 );
const _v = new Vector3();
const _v2 = new Vector3();
const _axis = new Vector3();
const _q = new Quaternion();
const _m = new Matrix4();
const ZERO = new Vector3();

const easeInOutCubic = ( t ) => ( t < 0.5 ? 4 * t * t * t : 1 - Math.pow( - 2 * t + 2, 3 ) / 2 );
const smooth = ( t ) => t * t * ( 3 - 2 * t );

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

/** Camera quaternion looking along unit direction d (world up = +Y). */
function lookQuaternion( d, out ) {

	_m.lookAt( ZERO, d, Math.abs( d.y ) > 0.999 ? _v2.set( 0, 0, 1 ) : UP );
	return out.setFromRotationMatrix( _m );

}

export function prefersReducedMotion() {

	return typeof matchMedia === 'function' && matchMedia( '(prefers-reduced-motion: reduce)' ).matches;

}

export class CameraRig {

	constructor( options = {} ) {

		this.order = 31;
		this.options = {
			orbitControls: null,
			autoRotate: true,
			autoRotateSpeed: 0.45, // OrbitControls units: 2π/60 rad/s per unit → ~2.2 min per turn
			autoRotateRamp: 2.5, // seconds to reach full speed
			idleDelay: 8,
			startDelay: 2, // auto-rotate starts this long after a graph loads
			followSeconds: 8,
			minFov: 30,
			maxFov: 115,
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

		this._time = 0;
		this._idle = 0;
		this._autoAmount = 0;
		this._inputLocked = false;
		this._center = new Vector3();
		this._radius = 100;
		this._sphere = new Sphere();
		this._sphereTime = - Infinity;
		this._sphereVersion = - 1;
		this._orbitFov = 45;

		// orbit tween
		this._flight = null;
		this._fl = { t: 0, dur: 1, track: - 1, t0: new Vector3(), t1: new Vector3(), dir0: new Vector3(), dir1: new Vector3(), d0: 1, d1: 1, bump: 0 };
		// framing follow
		this._follow = null;
		this._followCenter = new Vector3();
		this._followDist = 0;
		this._followNext = 0;
		// immersive
		this._look = { yaw: 0, pitch: 0, vyaw: 0, vpitch: 0, fov: 75, fovTarget: 75 };
		this._lookTween = null;
		this._transition = null;
		this._tr = { t: 0, dur: 1.4, dir: 'in', p0: new Vector3(), u0: new Vector3(), u1: new Vector3(), d0: 1, d1: 1, q0: new Quaternion(), q1: new Quaternion(), f0: 45, f1: 75 };
		this._pointers = new Map(); // pointerId -> {x, y}
		this._lookDrag = null;
		this._pinch = null;
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

		let controls = this.options.orbitControls || app.controls;
		if ( ! controls ) {

			controls = new OrbitControls( this.camera, this.canvas );
			app.controls = controls;

		}

		this.controls = controls;
		controls.enableDamping = true;
		controls.dampingFactor = 0.075;
		controls.rotateSpeed = 0.6;
		controls.zoomSpeed = 1.1;
		controls.panSpeed = 0.9;
		controls.zoomToCursor = true;
		controls.screenSpacePanning = true;
		controls.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };
		controls.autoRotate = false;

		// bypass OrbitControls while immersive or while an immersive transition runs
		this._controlsUpdate = controls.update;
		const update = controls.update.bind( controls );
		controls.update = ( dt ) => ( this.immersive || this._transition ? false : update( dt ) );

		this._onControlsStart = () => {

			this.notifyInteraction();
			this._flight = null;

		};

		controls.addEventListener( 'start', this._onControlsStart );

		if ( typeof matchMedia === 'function' ) {

			this._mq = matchMedia( '(prefers-reduced-motion: reduce)' );
			this._onMq = () => {

				this.reducedMotion = this._mq.matches;

			};

			this._mq.addEventListener?.( 'change', this._onMq );

		}

		const c = this.canvas;
		this._onDownCapture = ( e ) => this._pointerDownCapture( e );
		this._onDown = ( e ) => this._pointerDown( e );
		this._onMove = ( e ) => this._pointerMove( e );
		this._onUp = ( e ) => this._pointerUp( e );
		this._onWheel = ( e ) => this._wheel( e );
		this._onHover = ( e ) => {

			if ( e.pointerType === 'mouse' ) this.notifyInteraction();

		};

		this._onKey = () => this.notifyInteraction();
		c.addEventListener( 'pointerdown', this._onDownCapture, { capture: true } );
		c.addEventListener( 'pointerdown', this._onDown );
		c.addEventListener( 'pointermove', this._onHover );
		window.addEventListener( 'pointermove', this._onMove );
		window.addEventListener( 'pointerup', this._onUp );
		window.addEventListener( 'pointercancel', this._onUp );
		c.addEventListener( 'wheel', this._onWheel, { passive: false } );
		window.addEventListener( 'keydown', this._onKey );

	}

	setGraph( buffers, graph, app ) {

		this.buffers = buffers;
		this.graph = graph;
		this._sphereVersion = - 1;
		this._refreshSphere( true );
		if ( this.immersive ) return;
		this._flight = null;
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
		else {

			if ( this._flight ) this._stepFlight( dt );
			else if ( this._follow ) this._stepFollow( dt );

		}

		this._stepAutoRotate( dt );
		this._stepInsets( dt );
		this._updateClipping();

	}

	resize() {

		this._insetsDirty = true;
		this._applyViewOffset();

	}

	dispose() {

		const c = this.canvas;
		if ( ! c ) return;
		c.removeEventListener( 'pointerdown', this._onDownCapture, { capture: true } );
		c.removeEventListener( 'pointerdown', this._onDown );
		c.removeEventListener( 'pointermove', this._onHover );
		window.removeEventListener( 'pointermove', this._onMove );
		window.removeEventListener( 'pointerup', this._onUp );
		window.removeEventListener( 'pointercancel', this._onUp );
		c.removeEventListener( 'wheel', this._onWheel );
		window.removeEventListener( 'keydown', this._onKey );
		this._mq?.removeEventListener?.( 'change', this._onMq );
		if ( this.controls ) {

			this.controls.removeEventListener( 'start', this._onControlsStart );
			this.controls.update = this._controlsUpdate;

		}

	}

	// ── public API ────────────────────────────────────────────────────────────────────────

	get busy() {

		return !! ( this._flight || this._transition || this._lookTween );

	}

	notifyInteraction() {

		this._idle = 0;
		this._autoAmount = 0;
		this._follow = null;

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

			this._lookDrag = null;
			this._look.vyaw = 0; this._look.vpitch = 0;
			this.notifyInteraction();

		}

	}

	/** Distance at which a sphere of radius r fills the free (non-inset) part of the view. */
	fitDistance( r ) {

		const cam = this.camera, ins = this._insets;
		const w = Math.max( 1, this.app?.width || 1 ), h = Math.max( 1, this.app?.height || 1 );
		const fw = Math.max( 0.25 * w, w - ins.left - ins.right ) / w;
		const fh = Math.max( 0.25 * h, h - ins.top - ins.bottom ) / h;
		const t = Math.tan( MathUtils.degToRad( this.immersive ? this._orbitFov : cam.fov ) / 2 );
		const half = Math.min( Math.atan( t * fh ), Math.atan( t * cam.aspect * fw ) );
		return r / Math.sin( half );

	}

	setInsets( { left = 0, right = 0, top = 0, bottom = 0 } = {}, { animate = true } = {} ) {

		const ins = this._insets;
		ins.left = left; ins.right = right; ins.top = top; ins.bottom = bottom;
		if ( ! animate ) Object.assign( this._insetCur, ins );
		this._insetsDirty = true;
		if ( ! animate && this.camera ) this._applyViewOffset();

	}

	fitToGraph( { animate = true, padding = 1.1, direction = null } = {} ) {

		this._refreshSphere( true );
		if ( this.immersive ) {

			this._layoutDirection( _v );
			this._startLookTween( _v, this._immersiveFov(), animate );
			return;

		}

		const s = this._sphere;
		const dist = this.fitDistance( s.radius * padding );
		const c = this.controls;
		c.minDistance = s.radius * 0.01;
		c.maxDistance = s.radius * 30;
		this.flyTo( s.center, dist, { animate, direction } );

	}

	flyTo( point, distance = null, { duration = null, direction = null, track = - 1, animate = true } = {} ) {

		if ( this.immersive ) {

			_v.copy( point ).sub( this._immersiveCenter( _v2 ) );
			if ( _v.lengthSq() > 1e-12 ) this._startLookTween( _v.normalize(), this._look.fovTarget, animate );
			return;

		}

		const c = this.controls, cam = this.camera, f = this._fl;
		f.t0.copy( c.target );
		f.dir0.copy( cam.position ).sub( c.target );
		f.d0 = Math.max( 1e-6, f.dir0.length() );
		f.dir0.divideScalar( f.d0 );
		f.t1.copy( point );
		f.d1 = distance && distance > 0 ? distance : f.d0;
		f.dir1.copy( direction ? _v.copy( direction ).negate().normalize() : f.dir0 );
		f.track = track;
		const travel = f.t0.distanceTo( f.t1 );
		const ref = Math.max( f.d0, f.d1 );
		f.bump = this.reducedMotion || travel < 0.25 * Math.min( f.d0, f.d1 ) ? 0 : Math.min( 0.45, ( travel / ref ) * 0.3 );
		const auto = MathUtils.clamp( 0.75 + 0.22 * Math.log2( 1 + travel / ref + Math.abs( Math.log2( f.d1 / f.d0 ) ) * 0.5 ), 0.75, 1.6 );
		f.dur = ( duration ?? auto ) * ( this.reducedMotion ? 0.35 : 1 );
		f.t = 0;
		this._clearInertia();
		this._follow = null;
		if ( ! animate || f.dur <= 0.01 ) {

			f.t = 1;
			this._applyFlight();
			this._flight = null;
			c.update();
			return;

		}

		this._flight = f;

	}

	focusNode( index, { animate = true, distance = null } = {} ) {

		const app = this.app, p = app.positions, g = app.graph;
		if ( ! p || ! g || index < 0 || index >= g.N ) return;
		const x = p[ 4 * index ], y = p[ 4 * index + 1 ], z = p[ 4 * index + 2 ];
		_v.set( x, y, z );
		if ( this.immersive ) {

			this.flyTo( _v, null, { animate } );
			return;

		}

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

		this.flyTo( _v, dist, { animate, track: index } );

	}

	keepFramed( seconds = this.options.followSeconds ) {

		if ( this.immersive ) return;
		this._follow = { until: this._time + seconds };
		this._followNext = 0;

	}

	toggleImmersive() {

		this.setImmersive( ! this.immersive );

	}

	setImmersive( on, { animate = true } = {} ) {

		on = !! on;
		if ( on === this.immersive || ! this.camera ) return;
		const cam = this.camera, c = this.controls, tr = this._tr, look = this._look;
		this._flight = null;
		this._follow = null;
		this._lookTween = null;
		this._clearInertia();
		const C = this._immersiveCenter( this._camCenter );
		tr.p0.copy( cam.position );
		tr.q0.copy( cam.quaternion );
		tr.f0 = cam.fov;
		tr.t = 0;
		if ( on ) {

			this._orbitFov = cam.fov;
			this._layoutDirection( _v ); // final look direction
			look.yaw = Math.atan2( _v.x, _v.z );
			look.pitch = Math.asin( MathUtils.clamp( _v.y, - 1, 1 ) );
			look.vyaw = look.vpitch = 0;
			look.fov = look.fovTarget = this._immersiveFov();
			tr.dir = 'in';
			tr.u0.copy( cam.position ).sub( C );
			tr.d0 = tr.u0.length();
			if ( tr.d0 < 1e-6 ) tr.u0.copy( _v ).negate();
			tr.u0.normalize();
			tr.u1.copy( _v ).negate();
			tr.q1.copy( lookQuaternion( _v, _q ) );
			tr.f1 = look.fov;
			tr.dur = ( this.reducedMotion ? 0.5 : 1.6 );
			c.enabled = false;

		} else {

			this._lookDirection( this._dir );
			const R = Math.max( this._radius, this.buffers ? this.buffers.uniforms.sphereRadius.value : 0 );
			tr.dir = 'out';
			tr.u0.set( 0, 0, 0 );
			tr.u1.copy( this._dir ).negate();
			tr.d0 = 0;
			tr.d1 = this.fitDistance( R * 1.15 );
			tr.q1.copy( lookQuaternion( this._dir, _q ) );
			tr.f1 = this._orbitFov;
			tr.dur = ( this.reducedMotion ? 0.45 : 1.25 );

		}

		this.immersive = on;
		this._transition = tr;
		if ( ! animate ) {

			tr.t = 1;
			this._stepTransition( 0 );

		}

		this.app.events.dispatchEvent( new CustomEvent( 'immersive', { detail: { immersive: on } } ) );

	}

	// ── internals: framing, flight, follow ───────────────────────────────────────────────

	_refreshSphere( force ) {

		const app = this.app;
		if ( ! app || ! app.graph ) return;
		if ( ! force && ( this._time - this._sphereTime < 1 || app.positionsVersion === this._sphereVersion ) ) return;
		this._sphereTime = this._time;
		this._sphereVersion = app.positionsVersion;
		app.boundingSphere( this._sphere );
		this._center.copy( this._sphere.center );
		this._radius = Math.max( 1e-3, this._sphere.radius );

	}

	_clearInertia() {

		const c = this.controls;
		if ( ! c ) return;
		c._sphericalDelta?.set( 0, 0, 0 );
		c._panOffset?.set( 0, 0, 0 );
		if ( c._scale !== undefined ) c._scale = 1;

	}

	_stepFlight( dt ) {

		const f = this._fl;
		f.t = Math.min( 1, f.t + dt / f.dur );
		if ( f.track >= 0 ) {

			const p = this.app.positions;
			if ( p && 4 * f.track + 2 < p.length ) f.t1.set( p[ 4 * f.track ], p[ 4 * f.track + 1 ], p[ 4 * f.track + 2 ] );

		}

		this._applyFlight();
		if ( f.t >= 1 ) this._flight = null;

	}

	_applyFlight() {

		const f = this._fl, c = this.controls, cam = this.camera;
		const e = easeInOutCubic( f.t );
		c.target.lerpVectors( f.t0, f.t1, e );
		slerpUnit( f.dir0, f.dir1, e, this._dir );
		const d = Math.exp( MathUtils.lerp( Math.log( f.d0 ), Math.log( f.d1 ), e ) ) * ( 1 + f.bump * Math.sin( Math.PI * f.t ) );
		cam.position.copy( c.target ).addScaledVector( this._dir, d );
		cam.lookAt( c.target );

	}

	_stepFollow( dt ) {

		const fo = this._follow, c = this.controls, cam = this.camera;
		if ( this._time > fo.until ) {

			this._follow = null;
			return;

		}

		if ( this._time >= this._followNext ) {

			this._followNext = this._time + 0.3;
			this._refreshSphere( true );
			this._followCenter.copy( this._sphere.center );
			this._followDist = this.fitDistance( this._sphere.radius * 1.1 );
			c.maxDistance = Math.max( c.maxDistance, this._sphere.radius * 30 );

		}

		const k = 1 - Math.exp( - dt * 2.2 );
		_v.copy( cam.position ).sub( c.target );
		const d = _v.length();
		if ( d < 1e-9 ) return;
		_v.divideScalar( d );
		c.target.lerp( this._followCenter, k );
		cam.position.copy( c.target ).addScaledVector( _v, d + ( this._followDist - d ) * k );

	}

	_stepAutoRotate( dt ) {

		const c = this.controls;
		if ( ! c ) return;
		const want = this.autoRotate && ! this.immersive && ! this._transition && ! this._flight && ! this._inputLocked &&
			this._idle >= this.idleDelay && c.state === - 1 && !! this.app.graph;
		this._autoAmount = want ? Math.min( 1, this._autoAmount + dt / this.options.autoRotateRamp ) : 0;
		const active = this._autoAmount > 0;
		c.autoRotate = active;
		c.autoRotateSpeed = this.autoRotateSpeed * smooth( this._autoAmount );
		if ( active !== this.autoRotating ) {

			this.autoRotating = active;
			this.app.events.dispatchEvent( new CustomEvent( 'autorotate', { detail: { active } } ) );

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

	_updateClipping() {

		const cam = this.camera;
		const R = Math.max( this._radius, this.buffers ? this.buffers.uniforms.sphereRadius.value * this.buffers.uniforms.layoutBlend.value : 0 );
		let near, far;
		if ( this.immersive || this._transition ) {

			const d = cam.position.distanceTo( this._immersiveCenter( _v2 ) );
			near = Math.max( R * 0.002, 0.01 );
			far = d + R * 6;

		} else {

			const d = cam.position.distanceTo( this._center );
			near = Math.max( R * 0.002, ( d - R * 1.6 ) * 0.5 );
			far = d + R * 4;

		}

		if ( Math.abs( near - cam.near ) > cam.near * 0.02 || Math.abs( far - cam.far ) > cam.far * 0.02 ) {

			cam.near = near;
			cam.far = far;
			cam.updateProjectionMatrix();

		}

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

	_startLookTween( dir, fov, animate = true ) {

		const L = this._look;
		const yaw1 = Math.atan2( dir.x, dir.z );
		const pitch1 = MathUtils.clamp( Math.asin( MathUtils.clamp( dir.y, - 1, 1 ) ), - 1.396, 1.396 );
		let dy = yaw1 - L.yaw;
		dy = Math.atan2( Math.sin( dy ), Math.cos( dy ) ); // shortest way round
		L.vyaw = L.vpitch = 0;
		if ( ! animate ) {

			L.yaw += dy; L.pitch = pitch1; L.fovTarget = fov;
			this._lookTween = null;
			return;

		}

		this._lookTween = { t: 0, dur: this.reducedMotion ? 0.3 : 0.9, y0: L.yaw, y1: L.yaw + dy, p0: L.pitch, p1: pitch1, f0: L.fovTarget, f1: fov };

	}

	_stepImmersive( dt ) {

		const L = this._look, cam = this.camera;
		const tw = this._lookTween;
		if ( tw ) {

			tw.t = Math.min( 1, tw.t + dt / tw.dur );
			const e = easeInOutCubic( tw.t );
			L.yaw = MathUtils.lerp( tw.y0, tw.y1, e );
			L.pitch = MathUtils.lerp( tw.p0, tw.p1, e );
			L.fovTarget = MathUtils.lerp( tw.f0, tw.f1, e );
			if ( tw.t >= 1 ) this._lookTween = null;

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
		this._immersiveCenter( cam.position );
		cam.quaternion.copy( lookQuaternion( this._lookDirection( this._dir ), _q ) );
		if ( Math.abs( cam.fov - L.fov ) > 1e-3 ) {

			cam.fov = L.fov;
			cam.updateProjectionMatrix();

		}

	}

	_stepTransition( dt ) {

		const tr = this._tr, cam = this.camera, c = this.controls;
		tr.t = Math.min( 1, tr.t + dt / tr.dur );
		const e = easeInOutCubic( tr.t );
		const C = this._immersiveCenter( this._camCenter );
		if ( tr.dir === 'in' ) {

			slerpUnit( tr.u0, tr.u1, e, _v );
			cam.position.copy( C ).addScaledVector( _v, tr.d0 * ( 1 - e ) );

		} else {

			cam.position.copy( C ).addScaledVector( tr.u1, tr.d1 * e );

		}

		cam.quaternion.slerpQuaternions( tr.q0, tr.q1, e );
		cam.fov = MathUtils.lerp( tr.f0, tr.f1, e );
		cam.updateProjectionMatrix();
		if ( tr.t < 1 ) return;

		this._transition = null;
		if ( tr.dir === 'out' ) {

			c.target.copy( C );
			c.enabled = true;
			this._clearInertia();
			this._controlsUpdate.call( c );
			this._idle = 0;

		}

	}

	// ── internals: pointer input (immersive look; pointer-type tuning) ───────────────────

	_pointerDownCapture( e ) {

		// runs before OrbitControls' own listener: tune speeds for the device in use
		const c = this.controls;
		if ( c ) {

			const touch = e.pointerType === 'touch';
			c.rotateSpeed = touch ? 0.9 : 0.6;
			c.panSpeed = touch ? 1 : 0.9;

		}

		this.notifyInteraction();

	}

	_pointerDown( e ) {

		this._pointers.set( e.pointerId, { x: e.clientX, y: e.clientY } );
		if ( ! this.immersive || this._transition || this._inputLocked ) return;
		this._beginImmersiveGesture( e );

	}

	_beginImmersiveGesture( e ) {

		const L = this._look;
		this._lookTween = null;
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
		if ( ! this.immersive || this._transition || this._inputLocked ) return;
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

	}

	_wheel( e ) {

		this.notifyInteraction();
		if ( ! this.immersive ) {

			this._flight = null;
			return;

		}

		e.preventDefault();
		if ( this._transition ) return;
		const L = this._look;
		this._lookTween = null;
		let dy = e.deltaY * ( e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1 );
		if ( e.ctrlKey ) dy *= 6; // trackpad pinch
		L.fovTarget = MathUtils.clamp( L.fovTarget * Math.exp( dy * 0.0012 ), this.options.minFov, this.options.maxFov );

	}

}
