// Photo mode: the picture alone, a lens to set, and a capture.
//
// Entering adds `wgv-ui-hidden` to <html> (every surface steps aside: the panel, the slots, the
// node card), centres the graph in the whole window, holds the automatic rotation and turns
// hover / selection off, so a click on the picture means "focus here". A minimal bar stays:
//   focus      click the picture: the node nearest to the click is kept in focus (its distance
//              along the view axis goes to gfx.setFocus every frame, so it stays sharp while the
//              camera orbits); without a click the automatic focus of the render graph applies
//   blur       strength of the depth of field (0 = off); local to photo mode and restored on exit
//   exposure   the `exposure` setting, in stops
//   size       1× / 2× / 4× of the window, limited by what a capture may cost on this device
//              (captureBudget: four times the pixel cap of the quality tier in use, the longer
//              side within 4096 px on touch devices); a size that would come out no larger than
//              the one below it is not offered
//   capture    gfx.capture() → PNG download named after the dataset
// Esc or the close button leave; the camera, the layout and the selection are as they were.
//
// API
//   const photo = new PhotoMode( { app, rig, interaction, panel, name: () => string,
//                                  getExposure: () => number, setExposure( v ), download = true,
//                                  insets: () => view insets to give back to the rig on exit
//                                  (default panel.insets()) } )
//   photo.enter() / exit() / toggle() / active
//   photo.setScale( 1 | 2 | 4 ) / scale (2; 1 on touch devices), photo.setBlur( 0..1 ) / blur
//   photo.focusAt( clientX, clientY ) -> node index or −1
//   photo.size( scale = photo.scale )   { width, height, ratio, limited } of a capture
//   photo.offers( scale )               whether that size is worth offering here
//   photo.capture() -> Promise<{ name, width, height, bytes, blob } | null>   also in photo.last
//   photo.dispose()
// Event on app.events: 'photo' { active }.
// Pure helpers (tested): captureBudget(), captureSize(), scaleOffered(), nearestOnScreen(),
// photoFileName().
import { h, icon } from './Panel.js';

export const PHOTO_SCALES = [ 1, 2, 4 ];
/** The largest pixel budget of one capture on any device: the chain holds several HDR targets of this size. */
export const PHOTO_MAX_PIXELS = 16e6;
/** Longer side of a capture on touch devices: the default limit of RenderGraph.capture(). */
export const PHOTO_MAX_SIDE_TOUCH = 4096;

/**
 * What one capture may cost on this device. The quality tier in use says what the device is
 * good for (its `maxPixels` is the cap of the live picture: 1.5 MP low … 6 MP ultra); a capture
 * renders a few frames only, so it may take four times that, and never more than
 * PHOTO_MAX_PIXELS. On touch devices the longer side also stays within PHOTO_MAX_SIDE_TOUCH.
 * @param {object|null} config  app.gfx.config
 * @returns {{ maxPixels: number, maxSide: number }}
 */
export function captureBudget( config, { coarse = false } = {} ) {

	const tier = config && config.maxPixels > 0 ? 4 * config.maxPixels : PHOTO_MAX_PIXELS;
	return { maxPixels: Math.min( PHOTO_MAX_PIXELS, tier ), maxSide: coarse ? PHOTO_MAX_SIDE_TOUCH : Infinity };

}

/**
 * Drawing buffer of a capture, the way RenderGraph.capture() sizes it: pixel ratio × scale,
 * limited so that the longer side stays within `maxSize` (and the device limit). Here maxSize is
 * derived from a pixel budget (and an optional limit of the longer side), so the 4× of a large
 * window degrades gracefully.
 * @returns {{ width: number, height: number, ratio: number, maxSize: number, limited: boolean }}
 */
export function captureSize( width, height, pixelRatio = 1, scale = 1, { maxPixels = PHOTO_MAX_PIXELS, deviceLimit = 8192, maxSide = Infinity } = {} ) {

	const w = Math.max( 1, width ), hgt = Math.max( 1, height );
	const long = Math.max( w, hgt ), short = Math.min( w, hgt );
	const maxSize = Math.max( 256, Math.min( deviceLimit, maxSide, Math.floor( Math.sqrt( maxPixels * long / short ) ) ) );
	const want = pixelRatio * scale;
	const ratio = Math.min( want, maxSize / long );
	return { width: Math.floor( w * ratio ), height: Math.floor( hgt * ratio ), ratio, maxSize, limited: ratio < want - 1e-9 };

}

/**
 * Whether a size is worth offering: the smallest always is; a larger one only when the budget
 * lets it come out clearly larger (15 % per side) than what the size below it asks for (a "4×"
 * that would give the picture of the 2× is not a choice).
 */
export function scaleOffered( scale, width, height, pixelRatio = 1, limits = {}, scales = PHOTO_SCALES ) {

	const i = scales.indexOf( scale );
	if ( i < 0 ) return false;
	if ( i === 0 ) return true;
	return captureSize( width, height, pixelRatio, scale, limits ).ratio > pixelRatio * scales[ i - 1 ] * 1.15;

}

/**
 * The node nearest to a point of the screen. positions: Float32Array( 4N ) world xyz, m: the
 * view-projection matrix (column-major, 16 numbers), point and size in CSS pixels.
 * @returns {{ index: number, distance: number (pixels), x: number, y: number }} index −1 when no node is in front of the camera
 */
export function nearestOnScreen( positions, m, px, py, width, height, count = positions.length >> 2 ) {

	let best = - 1, bestD = Infinity, bx = 0, by = 0;
	for ( let i = 0; i < count; i ++ ) {

		const x = positions[ 4 * i ], y = positions[ 4 * i + 1 ], z = positions[ 4 * i + 2 ];
		const w = m[ 3 ] * x + m[ 7 ] * y + m[ 11 ] * z + m[ 15 ];
		if ( ! ( w > 1e-6 ) ) continue;
		const sx = ( ( m[ 0 ] * x + m[ 4 ] * y + m[ 8 ] * z + m[ 12 ] ) / w * 0.5 + 0.5 ) * width;
		const sy = ( 0.5 - ( m[ 1 ] * x + m[ 5 ] * y + m[ 9 ] * z + m[ 13 ] ) / w * 0.5 ) * height;
		const d = ( sx - px ) * ( sx - px ) + ( sy - py ) * ( sy - py );
		if ( d < bestD ) {

			bestD = d; best = i; bx = sx; by = sy;

		}

	}

	return { index: best, distance: Math.sqrt( bestD ), x: bx, y: by };

}

/** "Les Misérables" + 2560 × 1600 → "les-miserables-2560x1600.png" */
export function photoFileName( name, width, height, ext = 'png' ) {

	const slug = String( name || 'graph' ).normalize( 'NFKD' ).replace( /[̀-ͯ]/g, '' ).toLowerCase()
		.replace( /\.[a-z0-9]{2,7}$/, '' ).replace( /[^a-z0-9]+/g, '-' ).replace( /^-+|-+$/g, '' ).slice( 0, 60 ) || 'graph';
	return `${slug}-${width}x${height}.${ext}`;

}

const fmtBytes = ( n ) => ( n >= 1e6 ? `${( n / 1e6 ).toFixed( 1 )} MB` : `${Math.max( 1, Math.round( n / 1e3 ) )} kB` );
const fmtEv = ( ev ) => `${ev > 0.005 ? '+' : ev < - 0.005 ? '−' : ''}${Math.abs( ev ).toFixed( 1 )} EV`;

export class PhotoMode {

	constructor( options = {} ) {

		this.options = { download: true, container: typeof document !== 'undefined' ? document.body : null, ...options };
		this.app = options.app;
		this.rig = options.rig || null;
		this.interaction = options.interaction || null;
		this.panel = options.panel || null;
		this.active = false;
		this.coarse = typeof matchMedia === 'function' && matchMedia( '(pointer: coarse)' ).matches;
		this.scale = this.coarse ? 1 : 2; // a phone's own resolution is a large picture already
		this.blur = 0.6;
		this.busy = false;
		this.last = null;
		this._focusNode = - 1;
		this._saved = null;
		this._build();
		this._onKey = ( e ) => this._key( e );
		this._onFrame = () => this._frame();
		this._onResize = () => this._sync(); // the sizes on offer follow the window
		this._onDown = ( e ) => this._pointer( e, true );
		this._onUp = ( e ) => this._pointer( e, false );

	}

	get gfx() {

		return this.app.gfx;

	}

	toggle() {

		return this.active ? this.exit() : this.enter();

	}

	enter() {

		if ( this.active ) return this;
		const { app, rig, interaction, gfx } = this;
		this.active = true;
		this._saved = {
			autoRotate: rig ? rig.autoRotate : false,
			enabled: interaction ? interaction.enabled : true,
			focus: document.activeElement,
			dof: gfx.get( 'dof' ), dofStrength: gfx.get( 'dofStrength' ), blurChanged: false,
		};
		interaction?.setHover?.( - 1 );
		if ( interaction ) interaction.enabled = false;
		rig?.setAutoRotate( false );
		rig?.setInsets( { left: 0, right: 0, top: 0, bottom: 0 } );
		// the bar starts from what the picture shows now
		const cfg = gfx.config || {};
		const set = gfx.get( 'dofStrength' );
		this.blur = cfg.dof ? ( typeof set === 'number' ? set : ( cfg.dofAmount ?? 0.6 ) ) : 0;
		this._focusNode = - 1;
		document.documentElement.classList.add( 'wgv-ui-hidden' );
		this.root.classList.add( 'is-visible' );
		this.root.removeAttribute( 'inert' );
		this._sync();
		window.addEventListener( 'keydown', this._onKey, true );
		window.addEventListener( 'resize', this._onResize );
		app.events.addEventListener( 'frame', this._onFrame );
		const el = app.renderer.domElement;
		el.addEventListener( 'pointerdown', this._onDown );
		el.addEventListener( 'pointerup', this._onUp );
		this._captureBtn.focus( { preventScroll: true } );
		app.events.dispatchEvent( new CustomEvent( 'photo', { detail: { active: true } } ) );
		return this;

	}

	exit() {

		if ( ! this.active ) return this;
		const { app, rig, interaction, gfx } = this;
		this.active = false;
		window.removeEventListener( 'keydown', this._onKey, true );
		window.removeEventListener( 'resize', this._onResize );
		app.events.removeEventListener( 'frame', this._onFrame );
		const el = app.renderer.domElement;
		el.removeEventListener( 'pointerdown', this._onDown );
		el.removeEventListener( 'pointerup', this._onUp );
		gfx.setFocus( null );
		this._focusNode = - 1;
		document.documentElement.classList.remove( 'wgv-ui-hidden' );
		this.root.classList.remove( 'is-visible' );
		this.root.setAttribute( 'inert', '' );
		const s = this._saved || {};
		if ( interaction ) interaction.enabled = s.enabled !== false;
		rig?.setAutoRotate( !! s.autoRotate );
		if ( rig && ( this.options.insets || this.panel ) ) rig.setInsets( this.options.insets ? this.options.insets() : this.panel.insets() );
		if ( s.blurChanged ) {

			// the depth of field goes back to what the settings had asked for
			gfx.set( 'dof', s.dof === undefined ? null : s.dof );
			gfx.set( 'dofStrength', s.dofStrength === undefined ? null : s.dofStrength );

		}

		if ( s.focus && s.focus.isConnected && s.focus.focus ) s.focus.focus( { preventScroll: true } );
		this._saved = null;
		app.events.dispatchEvent( new CustomEvent( 'photo', { detail: { active: false } } ) );
		return this;

	}

	setScale( scale ) {

		if ( ! PHOTO_SCALES.includes( scale ) || ! this.offers( scale ) ) return;
		this.scale = scale;
		this._sync();

	}

	setBlur( v ) {

		this.blur = Math.min( 1, Math.max( 0, + v || 0 ) );
		if ( this.active ) {

			this._saved.blurChanged = true;
			const on = this.blur > 0.005;
			if ( !! this.gfx.config.dof !== on ) this.gfx.set( 'dof', on ); // structural: only when it flips
			this.gfx.set( 'dofStrength', this.blur );

		}

		this._sync();

	}

	/** The limits of a capture here: the budget of the quality tier in use and what the device can hold. */
	_limits() {

		const app = this.app;
		const device = app.renderer.backend && app.renderer.backend.device;
		return { ...captureBudget( this.gfx.config, { coarse: this.coarse } ), deviceLimit: device ? device.limits.maxTextureDimension2D : 8192 };

	}

	size( scale = this.scale ) {

		const app = this.app;
		const pr = this._ratio ?? app.renderer.getPixelRatio();
		return captureSize( app.width, app.height, pr, scale, this._limits() );

	}

	offers( scale ) {

		const app = this.app;
		return scaleOffered( scale, app.width, app.height, this._ratio ?? app.renderer.getPixelRatio(), this._limits() );

	}

	/** Focuses the node nearest to a point of the window; returns its index. */
	focusAt( clientX, clientY ) {

		const app = this.app, p = app.positions, cam = app.camera;
		if ( ! p || ! app.graph ) return - 1;
		const rect = app.renderer.domElement.getBoundingClientRect();
		cam.updateMatrixWorld();
		const m = _m.multiplyMatrices( cam.projectionMatrix, cam.matrixWorldInverse ).elements;
		const hit = nearestOnScreen( p, m, clientX - rect.left, clientY - rect.top, rect.width, rect.height, app.graph.N );
		if ( hit.index < 0 ) return - 1;
		this._focusNode = hit.index;
		if ( this.blur < 0.05 ) this.setBlur( 0.6 ); // focusing with the blur off would show nothing
		this._frame();
		this._reticle.style.left = `${rect.left + hit.x}px`;
		this._reticle.style.top = `${rect.top + hit.y}px`;
		this._reticle.classList.remove( 'is-on' );
		void this._reticle.offsetWidth; // restart the animation
		this._reticle.classList.add( 'is-on' );
		return hit.index;

	}

	async capture() {

		if ( this.busy ) return null;
		const app = this.app, gfx = this.gfx;
		this.busy = true;
		this._sync();
		try {

			// the pixel ratio changes while the capture runs: the bar keeps showing the planned size
			this._ratio = app.renderer.getPixelRatio();
			const plan = this.size();
			const blob = await gfx.capture( { scale: this.scale, maxSize: plan.maxSize, type: 'image/png' } );
			const name = photoFileName( this.options.name ? this.options.name() : 'graph', plan.width, plan.height );
			this.last = { name, width: plan.width, height: plan.height, bytes: blob.size, type: blob.type, blob };
			if ( this.options.download ) this._download( blob, name );
			this.panel?.toast( `Saved ${name} · ${plan.width} × ${plan.height} · ${fmtBytes( blob.size )}`, { type: 'info', duration: 4200 } );
			return this.last;

		} catch ( e ) {

			console.error( e );
			this.panel?.toast( `The capture failed: ${e.message || e}. Try a smaller size.`, { type: 'error' } );
			return null;

		} finally {

			this._ratio = null;
			this.busy = false;
			this._sync();

		}

	}

	dispose() {

		this.exit();
		this.root.remove();
		this._reticle.remove();

	}

	// ── internals ─────────────────────────────────────────────────────────────────────────

	_build() {

		const range = ( label, min, max, step, onInput ) => {

			const out = h( 'output.wgv-val' );
			const input = h( 'input.wgv-range', { type: 'range', min, max, step, 'aria-label': label } );
			input.addEventListener( 'input', () => onInput( + input.value ) );
			return { el: h( 'label.wgv-photo-field', null, h( 'span', null, label ), input, out ), input, out, min, max };

		};

		this._blur = range( 'Blur', 0, 1, 0.01, ( v ) => this.setBlur( v ) );
		this._exposure = range( 'Exposure', - 2, 2, 0.1, ( ev ) => {

			this.options.setExposure?.( + Math.pow( 2, ev ).toFixed( 4 ) );
			this._sync();

		} );
		this._seg = h( 'div.wgv-seg', { role: 'radiogroup', 'aria-label': 'Size', style: { '--n': PHOTO_SCALES.length } } );
		this._segButtons = PHOTO_SCALES.map( ( s, i ) => {

			const b = h( 'button', { type: 'button', role: 'radio', 'aria-checked': 'false' }, `${s}×` );
			b.addEventListener( 'click', () => this.setScale( s ) );
			b.addEventListener( 'keydown', ( e ) => {

				const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? - 1 : 0;
				if ( ! d ) return;
				e.preventDefault();
				e.stopPropagation();
				let j = i;
				do j = ( j + d + PHOTO_SCALES.length ) % PHOTO_SCALES.length; while ( j !== i && ! this.offers( PHOTO_SCALES[ j ] ) );
				this.setScale( PHOTO_SCALES[ j ] );
				this._segButtons[ j ].focus();

			} );
			this._seg.append( b );
			return b;

		} );
		this._sizeText = h( 'span.wgv-photo-size' );
		this._captureLabel = h( 'span', null, 'Capture' );
		this._captureBtn = h( 'button.wgv-btn.wgv-primary', { type: 'button', onclick: () => this.capture() }, h( 'span', { html: icon( 'camera' ) } ), this._captureLabel );
		this.root = h( 'div.wgv-ui.wgv-photo.wgv-glass', { role: 'toolbar', 'aria-label': 'Photo mode', inert: true },
			h( 'span.wgv-photo-hint', null, h( 'span', { html: icon( 'focus' ) } ), `${this.coarse ? 'Tap' : 'Click'} the picture to focus` ),
			this._blur.el, this._exposure.el,
			this._seg, this._sizeText,
			h( 'span.wgv-photo-btns', null,
				this._captureBtn,
				h( 'button.wgv-icon', { type: 'button', 'aria-label': 'Leave photo mode', title: 'Leave photo mode (Esc)', html: icon( 'close' ), onclick: () => this.exit() } ) ) );
		this._reticle = h( 'div.wgv-reticle', { 'aria-hidden': 'true' } );
		this.options.container.append( this.root, this._reticle );

	}

	_sync() {

		const paint = ( r, v, text ) => {

			r.input.value = v;
			r.input.style.setProperty( '--p', `${( 100 * ( v - r.min ) / ( r.max - r.min ) ).toFixed( 1 )}%` );
			r.out.textContent = text;
			r.input.setAttribute( 'aria-valuetext', text );

		};

		paint( this._blur, this.blur, this.blur > 0.005 ? `${Math.round( 100 * this.blur )}%` : 'Off' );
		const exposure = this.options.getExposure ? this.options.getExposure() : 1;
		const ev = Math.min( 2, Math.max( - 2, Math.log2( Math.max( 1e-3, exposure ) ) ) );
		paint( this._exposure, ev, fmtEv( ev ) );
		// sizes the budget of this device and tier cannot deliver are not offered (the window or the
		// tier may have changed since the choice was made)
		const offered = this.app && this.app.renderer ? PHOTO_SCALES.map( ( s ) => this.offers( s ) ) : PHOTO_SCALES.map( () => true );
		while ( ! offered[ PHOTO_SCALES.indexOf( this.scale ) ] && this.scale > PHOTO_SCALES[ 0 ] ) this.scale = PHOTO_SCALES[ PHOTO_SCALES.indexOf( this.scale ) - 1 ];
		const idx = PHOTO_SCALES.indexOf( this.scale );
		this._segButtons.forEach( ( b, i ) => {

			b.setAttribute( 'aria-checked', String( i === idx ) );
			b.tabIndex = i === idx ? 0 : - 1;
			b.disabled = ! offered[ i ];
			b.title = offered[ i ] ? '' : 'More than one capture may cost at this quality';

		} );
		this._seg.style.setProperty( '--i', Math.max( 0, idx ) );
		this._captureBtn.disabled = this.busy;
		this._captureLabel.textContent = this.busy ? 'Rendering…' : 'Capture';
		this._syncSize();

	}

	_syncSize() {

		if ( ! this.app || ! this.app.renderer ) return;
		const s = this.size();
		const text = `${s.width} × ${s.height}`;
		if ( this._sizeText.textContent !== text ) this._sizeText.textContent = text;
		this._sizeText.title = s.limited ? 'Limited to what one capture may cost on this device at this quality' : '';
		this._sizeText.classList.toggle( 'is-limited', s.limited );

	}

	/** Keeps the focused node sharp while the camera moves. */
	_frame() {

		const i = this._focusNode, app = this.app, p = app.positions;
		if ( i < 0 || ! p || 4 * i + 2 >= p.length ) return;
		// the view axis is −Z of the camera's world matrix
		const e = app.camera.matrixWorld.elements;
		const l = Math.hypot( e[ 8 ], e[ 9 ], e[ 10 ] ) || 1;
		const distance = - ( ( p[ 4 * i ] - e[ 12 ] ) * e[ 8 ] + ( p[ 4 * i + 1 ] - e[ 13 ] ) * e[ 9 ] + ( p[ 4 * i + 2 ] - e[ 14 ] ) * e[ 10 ] ) / l;
		const R = app.graph ? app.graph.layoutRadius || 100 : 100;
		this.gfx.setFocus( { distance: Math.max( 1e-3, distance ), range: 0.75 * R } );

	}

	_pointer( e, down ) {

		if ( ! this.active || this.busy ) return;
		if ( down ) {

			this._press = e.isPrimary && ( e.pointerType !== 'mouse' || e.button === 0 ) ? { x: e.clientX, y: e.clientY, t: performance.now(), id: e.pointerId } : null;
			return;

		}

		const p = this._press;
		this._press = null;
		if ( ! p || p.id !== e.pointerId ) return;
		const moved = Math.hypot( e.clientX - p.x, e.clientY - p.y );
		if ( moved > ( e.pointerType === 'touch' ? 10 : 5 ) || performance.now() - p.t > 500 ) return; // an orbit, not a click
		this.focusAt( e.clientX, e.clientY );

	}

	_key( e ) {

		if ( e.key === 'Escape' ) {

			// the top-most thing closes first; nothing else may react to this Esc
			e.preventDefault();
			e.stopImmediatePropagation();
			if ( ! this.busy ) this.exit();

		}

	}

	_download( blob, name ) {

		const url = URL.createObjectURL( blob );
		const a = h( 'a', { href: url, download: name, hidden: true } );
		document.body.append( a );
		a.click();
		a.remove();
		setTimeout( () => URL.revokeObjectURL( url ), 30000 );

	}

}

// minimal column-major 4 × 4 product (no three import: the module stays testable)
const _m = {
	elements: new Float64Array( 16 ),
	multiplyMatrices( a, b ) {

		const ae = a.elements, be = b.elements, te = this.elements;
		for ( let c = 0; c < 4; c ++ ) for ( let r = 0; r < 4; r ++ ) {

			te[ 4 * c + r ] = ae[ r ] * be[ 4 * c ] + ae[ 4 + r ] * be[ 4 * c + 1 ] + ae[ 8 + r ] * be[ 4 * c + 2 ] + ae[ 12 + r ] * be[ 4 * c + 3 ];

		}

		return this;

	},
};
