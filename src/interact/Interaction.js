// Pointer, keyboard and file-drop interaction (App module, order 30): hover, selection with
// neighbour highlighting, node drag-and-drop, pinning, double-click focus, shortcuts.
//
// Design
//  • Pointer listeners sit in the capture phase on the canvas' parent (the "wrapper"), so they
//    run before OrbitControls / CameraRig (bubble listeners on the canvas). A press is picked at
//    pointerdown (CPU picking against app.positions, see Picking.js); when it lands on a node the
//    camera is locked (rig.setInputLocked(true) → OrbitControls rotate/pan off, immersive look
//    off) until release. OrbitControls still *tracks* the pointer, so when a second finger lands
//    during a node press/drag the drag is cancelled, the lock released and the second pointerdown
//    flows on to the camera as an ordinary pinch.
//  • Tap/click (movement ≤ 4 px mouse / 8 px touch) selects; on empty space it clears. A second
//    click within 320 ms: on a node → rig.focusNode(i); on empty space → rig.fitToGraph().
//  • Drag: beyond the threshold the node follows the pointer on the plane through the node,
//    perpendicular to the view direction (the grab offset is kept, so the node does not jump).
//    layout.startDrag(i) / drag(i, worldPos) / endDrag(i, {pin}). A plain drag keeps the node's
//    pin state, shift-drag toggles it, a touch long-press (500 ms, no movement) toggles it too
//    (and selects the node, with a short vibration where supported).
//  • Pin state: layout.isPinned(i) when the layout has it (ForceLayout keeps a CPU set), else a
//    local mirror resynchronised from nodePos.w of the readback.
//  • Highlight state: a CPU mirror of nodeStateTarget (4 floats/node: x selected, y neighbour,
//    z dimmed, w hovered) is flushed once per frame in update(): full upload after a selection
//    change (buffers.setStateTargets), 16-byte range uploads for hover changes. The GPU eases
//    nodeState toward it (ForceLayout). buffers.uniforms.focus is eased toward 1/0 here.
//  • Mouse hover is re-picked at most once per frame and only when the pointer, the camera or the
//    positions changed. Touch has no hover; the dragged node counts as hovered while dragging.
//  • Keys (ignored while typing in a form field): Esc clear, F fit, Space pause/resume layout,
//    R reheat, L cycle labels, ? help. Each maps to a command; onCommand(name) may handle it
//    (return true) else the default runs (labels/help have no default, use onCommand or the
//    'command' event).
//  • Files dragged over the window toggle `wgv-drop-active` on <html>; dropping a .json/.txt/.csv
//    calls onDropFile(file) (others: onDropRejected(file) + 'droprejected' event).
//  • DOM elements carrying data-node-index (e.g. labels) act as press targets for that node.
//
// API
//   const interaction = app.use( new Interaction( { layout, rig, picking, onDropFile, onDropRejected,
//                                  onCommand, dragThreshold: { mouse: 4, touch: 8 }, longPressMs: 500 } ) )
//   interaction.select( index, { focus = false } ) / clearSelection() / selectedIndex
//   interaction.hoverIndex / setHover( index )
//   interaction.isPinned( i ) / setPinned( i, on ) / togglePin( i ) / unpinAll()
//   interaction.command( name )        'escape' | 'fit' | 'toggle-run' | 'reheat' | 'cycle-labels' | 'help'
//   interaction.setLayout( layout ) / interaction.setRig( rig )
//   interaction.picking                Picking instance (setRadiusFunction, radiusScale)
//   interaction.neighbors( i )         Uint32Array view of i's neighbours
//   interaction.targets                CPU mirror of nodeStateTarget (Float32Array 4N)
//   interaction.dragging               index of the node being dragged or -1
//   interaction.enabled
// Events on app.events: 'hover' {index}, 'select' {index}, 'pin' {index, pinned},
//   'dragstart' {index}, 'dragend' {index, pinned}, 'command' {name}, 'droprejected' {file}.

import { Vector3, Vector2, Plane, Raycaster } from 'three/webgpu';
import { Picking } from './Picking.js';

const _ndc = new Vector2();
const _hit = new Vector3();
const _dir = new Vector3();

const TEXT_INPUT = 'textarea, select, [contenteditable=""], [contenteditable="true"], input:not([type]), input[type="text"], input[type="search"], input[type="number"], input[type="email"], input[type="url"]';
const INTERACTIVE = 'button, a, input, select, textarea, summary, [role="button"], [role="switch"], [role="radio"]';

export class Interaction {

	constructor( options = {} ) {

		this.order = 30;
		this.options = {
			dragThreshold: { mouse: 4, touch: 8 },
			longPressMs: 500,
			doubleMs: 320,
			accept: /\.(json|txt|csv)$/i,
			...options,
		};
		this.layout = options.layout || null;
		this.rig = options.rig || null;
		this.picking = options.picking || new Picking();
		this.onDropFile = options.onDropFile || null;
		this.onDropRejected = options.onDropRejected || null;
		this.onCommand = options.onCommand || null;
		this.enabled = true;

		this.selectedIndex = - 1;
		this.hoverIndex = - 1;
		this.dragging = - 1;
		this.targets = new Float32Array( 4 );
		this._pinned = new Uint8Array( 1 );
		this._pinVersion = - 1;
		this._press = null;
		this._lastClick = null;
		this._dirtyFull = false;
		this._dirtyNodes = [];
		this._focusTarget = 0;

		this._hoverX = 0;
		this._hoverY = 0;
		this._hoverIn = false;
		this._hoverDirty = false;
		this._hoverPosVersion = - 1;
		this._camSig = new Float64Array( 16 );
		this._rect = { left: 0, top: 0 };

		this._plane = new Plane();
		this._ray = new Raycaster();
		this._grabOffset = new Vector3();
		this._dragPos = new Vector3();
		this._dragDepth = 0;

	}

	// ── module hooks ──────────────────────────────────────────────────────────────────────

	init( app ) {

		this.app = app;
		this.canvas = app.renderer.domElement;
		this.wrapper = this.canvas.parentElement || app.container;
		if ( ! this.rig && app.cameraRig ) this.rig = app.cameraRig;

		const w = this.wrapper, opts = { capture: true };
		this._onDown = ( e ) => this._pointerDown( e );
		this._onMove = ( e ) => this._pointerMove( e );
		this._onUp = ( e ) => this._pointerUp( e );
		this._onLeave = ( e ) => {

			if ( e.pointerType !== 'touch' ) {

				this._hoverIn = false;
				if ( this.dragging < 0 ) this.setHover( - 1 );

			}

		};

		this._onKey = ( e ) => this._keyDown( e );
		this._onDragEnter = ( e ) => this._fileDrag( e, 'enter' );
		this._onDragOver = ( e ) => this._fileDrag( e, 'over' );
		this._onDragLeave = ( e ) => this._fileDrag( e, 'leave' );
		this._onDrop = ( e ) => this._fileDrop( e );
		w.addEventListener( 'pointerdown', this._onDown, opts );
		w.addEventListener( 'pointermove', this._onMove, opts );
		w.addEventListener( 'pointerup', this._onUp, opts );
		w.addEventListener( 'pointercancel', this._onUp, opts );
		this.canvas.addEventListener( 'pointerleave', this._onLeave );
		window.addEventListener( 'keydown', this._onKey );
		window.addEventListener( 'dragenter', this._onDragEnter );
		window.addEventListener( 'dragover', this._onDragOver );
		window.addEventListener( 'dragleave', this._onDragLeave );
		window.addEventListener( 'drop', this._onDrop );
		this._updateRect();

	}

	setGraph( buffers, graph ) {

		if ( this._press ) this._cancelPress();
		this._press = null;
		this.buffers = buffers;
		this.graph = graph;
		this.picking.setGraph( graph );
		const N = graph.N;
		this.targets = new Float32Array( 4 * Math.max( 1, N ) );
		this._pinned = new Uint8Array( Math.max( 1, N ) );
		const p = graph.initPos;
		for ( let i = 0; i < N; i ++ ) this._pinned[ i ] = p[ 4 * i + 3 ] > 0.5 ? 1 : 0;
		this._pinVersion = this.app ? this.app.positionsVersion : 0;
		this.selectedIndex = - 1;
		this.hoverIndex = - 1;
		this.dragging = - 1;
		this._lastClick = null;
		this._focusTarget = 0;
		this._dirtyNodes.length = 0;
		this._dirtyFull = true;
		const u = buffers.uniforms;
		u.selectedIndex.value = - 1;
		u.hoverIndex.value = - 1;
		u.focus.value = 0;
		this._flush();

	}

	update( dt, app ) {

		if ( ! this.buffers ) return;
		const u = this.buffers.uniforms;
		const f = u.focus.value;
		const k = 1 - Math.exp( - dt * 7 );
		u.focus.value = Math.abs( this._focusTarget - f ) < 1e-3 ? this._focusTarget : f + ( this._focusTarget - f ) * k;

		this._syncPinsFromReadback( app );
		this._updateHover( app );
		this._flush();

	}

	resize() {

		this._updateRect();

	}

	dispose() {

		const w = this.wrapper, opts = { capture: true };
		if ( ! w ) return;
		w.removeEventListener( 'pointerdown', this._onDown, opts );
		w.removeEventListener( 'pointermove', this._onMove, opts );
		w.removeEventListener( 'pointerup', this._onUp, opts );
		w.removeEventListener( 'pointercancel', this._onUp, opts );
		this.canvas.removeEventListener( 'pointerleave', this._onLeave );
		window.removeEventListener( 'keydown', this._onKey );
		window.removeEventListener( 'dragenter', this._onDragEnter );
		window.removeEventListener( 'dragover', this._onDragOver );
		window.removeEventListener( 'dragleave', this._onDragLeave );
		window.removeEventListener( 'drop', this._onDrop );
		clearTimeout( this._press?.timer );

	}

	// ── public API ────────────────────────────────────────────────────────────────────────

	setLayout( layout ) {

		this.layout = layout;

	}

	setRig( rig ) {

		this.rig = rig;

	}

	neighbors( i ) {

		const g = this.graph;
		if ( ! g || i < 0 || i >= g.N ) return new Uint32Array( 0 );
		return g.adjList.subarray( g.adjStart[ i ], g.adjStart[ i ] + g.degree[ i ] );

	}

	select( index, { focus = false, force = false } = {} ) {

		const g = this.graph;
		if ( ! g ) return;
		if ( ! ( index >= 0 && index < g.N ) ) index = - 1;
		if ( index === this.selectedIndex && ! force ) {

			if ( focus && index >= 0 ) this.rig?.focusNode( index );
			return;

		}

		const t = this.targets, N = g.N;
		const dim = index >= 0 ? 1 : 0;
		for ( let j = 0; j < N; j ++ ) {

			t[ 4 * j ] = 0; t[ 4 * j + 1 ] = 0; t[ 4 * j + 2 ] = dim;

		}

		if ( index >= 0 ) {

			t[ 4 * index ] = 1; t[ 4 * index + 2 ] = 0;
			const s = g.adjStart[ index ], e = s + g.degree[ index ];
			for ( let k = s; k < e; k ++ ) {

				const j = g.adjList[ k ];
				t[ 4 * j + 1 ] = 1; t[ 4 * j + 2 ] = 0;

			}

		}

		this.selectedIndex = index;
		this._focusTarget = index >= 0 ? 1 : 0;
		this.buffers.uniforms.selectedIndex.value = index;
		this._dirtyFull = true;
		this._emit( 'select', { index } );
		if ( focus && index >= 0 ) this.rig?.focusNode( index );

	}

	clearSelection() {

		this.select( - 1 );

	}

	setHover( index ) {

		const g = this.graph;
		if ( ! g ) return;
		if ( ! ( index >= 0 && index < g.N ) ) index = - 1;
		if ( index === this.hoverIndex ) return;
		const prev = this.hoverIndex;
		if ( prev >= 0 ) {

			this.targets[ 4 * prev + 3 ] = 0;
			this._dirtyNodes.push( prev );

		}

		if ( index >= 0 ) {

			this.targets[ 4 * index + 3 ] = 1;
			this._dirtyNodes.push( index );

		}

		this.hoverIndex = index;
		this.buffers.uniforms.hoverIndex.value = index;
		this._updateCursor();
		this._emit( 'hover', { index } );

	}

	isPinned( i ) {

		const L = this.layout;
		if ( L && typeof L.isPinned === 'function' ) return !! L.isPinned( i ); // authoritative (ForceLayout)
		return this._pinned[ i ] === 1;

	}

	setPinned( i, on ) {

		if ( ! this.graph || i < 0 || i >= this.graph.N ) return;
		on = !! on;
		this.layout?.setPinned?.( i, on );
		this._pinned[ i ] = on ? 1 : 0;
		this._pinVersion = this.app.positionsVersion;
		this._emit( 'pin', { index: i, pinned: on } );

	}

	togglePin( i ) {

		this.setPinned( i, ! this.isPinned( i ) );

	}

	unpinAll() {

		this.layout?.unpinAll?.();
		this._pinned.fill( 0 );
		this._pinVersion = this.app.positionsVersion;
		this._emit( 'pin', { index: - 1, pinned: false } );

	}

	command( name, event = null ) {

		this._emit( 'command', { name } );
		if ( this.onCommand && this.onCommand( name, this, event ) === true ) return;
		const L = this.layout;
		switch ( name ) {

			case 'escape': this.clearSelection(); break;
			case 'fit': this.rig?.fitToGraph(); break;
			case 'toggle-run': if ( L ) L.running ? L.pause() : L.resume(); break;
			case 'reheat': L?.reheat(); break;

		}

	}

	/** Uploads pending highlight-target changes (also run every frame by update()). */
	flush() {

		this._flush();

	}

	// ── internals: state upload ──────────────────────────────────────────────────────────

	_flush() {

		const b = this.buffers;
		if ( ! b ) return;
		const attr = b.nodeStateTarget.value;
		if ( this._dirtyFull ) {

			attr.clearUpdateRanges?.();
			b.setStateTargets( this.targets.subarray( 0, attr.array.length ) );
			this._dirtyFull = false;
			this._dirtyNodes.length = 0;

		} else if ( this._dirtyNodes.length ) {

			const arr = attr.array, t = this.targets;
			for ( const i of this._dirtyNodes ) {

				arr[ 4 * i ] = t[ 4 * i ]; arr[ 4 * i + 1 ] = t[ 4 * i + 1 ]; arr[ 4 * i + 2 ] = t[ 4 * i + 2 ]; arr[ 4 * i + 3 ] = t[ 4 * i + 3 ];
				attr.addUpdateRange( 4 * i, 4 );

			}

			attr.needsUpdate = true;
			this._dirtyNodes.length = 0;

		}

	}

	_emit( type, detail ) {

		this.app?.events.dispatchEvent( new CustomEvent( type, { detail } ) );

	}

	_syncPinsFromReadback( app ) {

		// pins changed elsewhere (layout reset, unpinAll) show up in nodePos.w of the readback;
		// wait a few readbacks after our own changes so a stale readback does not undo them
		const p = app.positions;
		if ( this.layout && typeof this.layout.isPinned === 'function' ) return; // layout knows better
		if ( ! p || app.positionsVersion < this._pinVersion + 3 || app.positionsVersion === this._pinSeen ) return;
		this._pinSeen = app.positionsVersion;
		const N = Math.min( this.graph.N, p.length >> 2 );
		for ( let i = 0; i < N; i ++ ) this._pinned[ i ] = p[ 4 * i + 3 ] > 0.5 ? 1 : 0;

	}

	_updateHover( app ) {

		if ( ! this._hoverIn || ! this.enabled ) return;
		if ( this.dragging >= 0 ) return;
		if ( this._press ) {

			if ( this._press.index < 0 && this._press.moved && this.hoverIndex >= 0 ) this.setHover( - 1 );
			return;

		}

		// re-pick only when the pointer, the camera or the positions changed
		const m = app.camera.matrixWorld.elements, sig = this._camSig;
		let camChanged = false;
		for ( let k = 0; k < 16; k ++ ) {

			if ( sig[ k ] !== m[ k ] ) {

				camChanged = true;
				sig[ k ] = m[ k ];

			}

		}

		if ( ! this._hoverDirty && ! camChanged && this._hoverPosVersion === app.positionsVersion ) return;
		this._hoverDirty = false;
		this._hoverPosVersion = app.positionsVersion;
		this.setHover( this._pick( this._hoverX, this._hoverY, 'mouse' ) );

	}

	_pick( x, y, pointerType ) {

		const app = this.app;
		if ( ! app.graph || ! app.positions ) return - 1;
		app.camera.updateMatrixWorld();
		return this.picking.pickApp( app, x, y, pointerType );

	}

	_updateRect() {

		if ( ! this.canvas ) return;
		const r = this.canvas.getBoundingClientRect();
		this._rect.left = r.left; this._rect.top = r.top;

	}

	_updateCursor() {

		const c = this.canvas;
		if ( ! c ) return;
		c.style.cursor = this.dragging >= 0 ? 'grabbing' : this.hoverIndex >= 0 ? 'pointer' : '';

	}

	_lockCamera( on ) {

		if ( this.rig ) this.rig.setInputLocked( on );
		else if ( this.app.controls ) {

			this.app.controls.enableRotate = ! on;
			this.app.controls.enablePan = ! on;

		}

	}

	// ── internals: pointer ───────────────────────────────────────────────────────────────

	_pointerDown( e ) {

		if ( ! this.enabled || ! this.graph ) return;
		const onCanvas = e.target === this.canvas;
		const label = ! onCanvas && e.target.closest ? e.target.closest( '[data-node-index]' ) : null;
		if ( ! onCanvas && ! label ) return;

		let p = this._press;
		if ( p && p.id === e.pointerId ) {

			// stale press (its pointerup never arrived): drop it
			this._cancelPress();
			this._press = p = null;

		}

		if ( p ) {

			// second pointer: hand the gesture (pinch) to the camera
			if ( ! p.cancelled ) this._cancelPress();
			return;

		}

		if ( e.pointerType === 'mouse' && e.button !== 0 ) return;
		this._updateRect();
		const x = e.clientX - this._rect.left, y = e.clientY - this._rect.top;
		const index = label ? ( + label.dataset.nodeIndex | 0 ) : this._pick( x, y, e.pointerType );
		p = this._press = {
			id: e.pointerId, type: e.pointerType, index, x0: x, y0: y, x, y, t0: performance.now(), stamp: e.timeStamp,
			shift: e.shiftKey, moved: false, dragging: false, cancelled: false, longFired: false, wasPinned: false, timer: 0,
		};
		if ( index < 0 ) return;

		this._lockCamera( true );
		if ( label ) e.preventDefault();
		try {

			this.canvas.setPointerCapture( e.pointerId );

		} catch ( _ ) { /* synthetic pointers */ }

		if ( e.pointerType === 'touch' ) p.timer = setTimeout( () => this._longPress( p ), this.options.longPressMs );

	}

	_pointerMove( e ) {

		if ( e.pointerType !== 'touch' && e.target === this.canvas ) {

			this._hoverX = e.clientX - this._rect.left;
			this._hoverY = e.clientY - this._rect.top;
			this._hoverIn = true;
			this._hoverDirty = true;

		}

		const p = this._press;
		if ( ! p || e.pointerId !== p.id || p.cancelled ) return;
		const x = e.clientX - this._rect.left, y = e.clientY - this._rect.top;
		p.x = x; p.y = y;
		p.shift = p.shift || e.shiftKey;
		const thr = p.type === 'touch' ? this.options.dragThreshold.touch : this.options.dragThreshold.mouse;
		if ( ! p.moved && Math.hypot( x - p.x0, y - p.y0 ) > thr ) {

			p.moved = true;
			clearTimeout( p.timer );
			this._undoStaleLongPress( p, e );
			if ( p.index >= 0 ) this._startDrag( p );

		}

		if ( p.dragging ) {

			this._dragTo( p, x, y );
			e.preventDefault();

		}

	}

	_pointerUp( e ) {

		const p = this._press;
		if ( ! p || e.pointerId !== p.id ) return;
		this._press = null;
		clearTimeout( p.timer ); // (pointer capture is released implicitly after pointerup)
		this._undoStaleLongPress( p, e );
		const ok = e.type === 'pointerup' && ! p.cancelled;
		if ( p.index >= 0 ) {

			if ( p.dragging ) this._endDrag( p, ! ok, e.shiftKey );
			else if ( ok && ! p.longFired ) this._click( p );
			if ( ! p.cancelled ) this._lockCamera( false );

		} else if ( ok && ! p.moved && performance.now() - p.t0 < 650 ) {

			this._click( p );

		}

	}

	_cancelPress() {

		const p = this._press;
		if ( ! p ) return;
		clearTimeout( p.timer );
		if ( p.dragging ) this._endDrag( p, true, false );
		if ( p.index >= 0 && ! p.cancelled ) this._lockCamera( false );
		p.cancelled = true;

	}

	_longPress( p ) {

		if ( this._press !== p || p.moved || p.cancelled || p.dragging ) return;
		p.longFired = true;
		this.togglePin( p.index );
		this.select( p.index );
		try {

			navigator.vibrate?.( 12 );

		} catch ( _ ) { /* ignore */ }

	}

	/**
	 * The long-press timer can fire while the release/move that ended the press is still queued
	 * (main thread busy). Event timestamps tell the truth: revert a pin toggle the user did not hold for.
	 */
	_undoStaleLongPress( p, e ) {

		if ( ! p.longFired || ! ( e.timeStamp - p.stamp < this.options.longPressMs * 0.9 ) ) return;
		p.longFired = false;
		this.togglePin( p.index );

	}

	_click( p ) {

		const now = performance.now();
		const last = this._lastClick;
		const tol = p.type === 'touch' ? 32 : 8;
		const isDouble = !! last && now - last.t < this.options.doubleMs && last.index === p.index && Math.hypot( p.x - last.x, p.y - last.y ) < tol;
		this._lastClick = isDouble ? null : { t: now, x: p.x, y: p.y, index: p.index };
		if ( p.index >= 0 ) {

			this.select( p.index );
			if ( isDouble ) this.rig?.focusNode( p.index );

		} else {

			this.clearSelection();
			if ( isDouble ) this.rig?.fitToGraph();

		}

	}

	_screenToPlane( x, y, out ) {

		const app = this.app;
		_ndc.set( ( x / app.width ) * 2 - 1, - ( y / app.height ) * 2 + 1 );
		this._ray.setFromCamera( _ndc, app.camera );
		return this._ray.ray.intersectPlane( this._plane, out );

	}

	_startDrag( p ) {

		const app = this.app, pos = app.positions, i = p.index;
		if ( ! pos ) return;
		app.camera.updateMatrixWorld();
		_hit.set( pos[ 4 * i ], pos[ 4 * i + 1 ], pos[ 4 * i + 2 ] );
		app.camera.getWorldDirection( _dir );
		this._plane.setFromNormalAndCoplanarPoint( _dir, _hit );
		this._grabOffset.set( 0, 0, 0 );
		const start = this._screenToPlane( p.x0, p.y0, this._dragPos );
		if ( start ) this._grabOffset.copy( _hit ).sub( start );
		p.dragging = true;
		p.wasPinned = this.isPinned( i );
		this.dragging = i;
		this.layout?.startDrag?.( i );
		this.setHover( i );
		this._updateCursor();
		this.rig?.notifyInteraction();
		this._emit( 'dragstart', { index: i } );
		this._dragTo( p, p.x, p.y );

	}

	_dragTo( p, x, y ) {

		const hit = this._screenToPlane( x, y, this._dragPos );
		if ( ! hit ) return;
		hit.add( this._grabOffset );
		this.layout?.drag?.( p.index, hit );

	}

	_endDrag( p, cancelled, shiftNow ) {

		const i = p.index;
		const pin = cancelled ? p.wasPinned : ( p.shift || shiftNow ) ? ! p.wasPinned : p.wasPinned;
		this.layout?.endDrag?.( i, { pin } );
		// endDrag only ever pins (ForceLayout semantics): unpinning needs an explicit call
		if ( ! pin && p.wasPinned ) this.layout?.setPinned?.( i, false );
		this._pinned[ i ] = pin ? 1 : 0;
		this._pinVersion = this.app.positionsVersion;
		p.dragging = false;
		this.dragging = - 1;
		if ( p.type === 'touch' || ! this._hoverIn ) this.setHover( - 1 );
		this._hoverDirty = true;
		this._updateCursor();
		this._emit( 'dragend', { index: i, pinned: pin } );
		if ( pin !== p.wasPinned ) this._emit( 'pin', { index: i, pinned: pin } );

	}

	// ── internals: keyboard, file drop ───────────────────────────────────────────────────

	_keyDown( e ) {

		if ( ! this.enabled || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey ) return;
		const t = e.target;
		if ( t && t.closest && t.closest( TEXT_INPUT ) ) return;
		let name = null;
		switch ( e.key ) {

			case 'Escape': name = 'escape'; break;
			case 'f': case 'F': name = 'fit'; break;
			case ' ': case 'Spacebar':
				if ( t && t.closest && t.closest( INTERACTIVE ) ) return; // let buttons activate
				name = 'toggle-run'; break;
			case 'r': case 'R': name = 'reheat'; break;
			case 'l': case 'L': name = 'cycle-labels'; break;
			case '?': name = 'help'; break;

		}

		if ( ! name ) return;
		if ( name === 'toggle-run' ) e.preventDefault();
		this.command( name, e );

	}

	_hasFiles( e ) {

		const types = e.dataTransfer && e.dataTransfer.types;
		if ( ! types ) return false;
		for ( let k = 0; k < types.length; k ++ ) if ( types[ k ] === 'Files' ) return true;
		return false;

	}

	_setDropActive( on ) {

		document.documentElement.classList.toggle( 'wgv-drop-active', on );

	}

	_fileDrag( e, phase ) {

		if ( ! this._hasFiles( e ) ) return;
		if ( phase === 'leave' ) {

			this._dragDepth = Math.max( 0, this._dragDepth - 1 );
			if ( this._dragDepth === 0 || e.relatedTarget === null ) {

				this._dragDepth = 0;
				this._setDropActive( false );

			}

			return;

		}

		e.preventDefault();
		if ( phase === 'enter' ) this._dragDepth ++;
		if ( e.dataTransfer ) e.dataTransfer.dropEffect = 'copy';
		this._setDropActive( true );

	}

	_fileDrop( e ) {

		if ( ! this._hasFiles( e ) ) return;
		e.preventDefault();
		this._dragDepth = 0;
		this._setDropActive( false );
		const files = Array.from( e.dataTransfer.files || [] );
		const file = files.find( ( f ) => this.options.accept.test( f.name ) );
		if ( file ) this.onDropFile?.( file );
		else if ( files.length ) {

			this.onDropRejected?.( files[ 0 ] );
			this._emit( 'droprejected', { file: files[ 0 ] } );

		}

	}

}
