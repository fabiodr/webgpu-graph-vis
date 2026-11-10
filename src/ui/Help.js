// Help overlay, first-run hint and the full-window drop target.
//
//  • Help: a dialog that lists the shortcuts and gestures in groups: the shell's own
//    (SHELL_SHORTCUTS; on touch screens TOUCH_SHORTCUTS replace its mouse gestures) and, read
//    when it opens, those of the explore layer ( [ { keys, title, group } ] ), which replace the
//    shell's line for the same keys. Focus moves into the dialog and returns on close; Esc, the
//    close button and a click on the backdrop close it.
//  • Hint: one quiet line for the first visit ("drag to orbit · click a node · / to search …").
//    It fades with the first interaction (pointer on the canvas, wheel, key, drop) and is
//    remembered through options.seen() / options.markSeen(), so it does not come back.
//  • Drop target: a full-window overlay shown while a file is dragged over the page (the class
//    `wgv-drop-active` on <html>, toggled by Interaction), naming the formats the data layer reads.
//
// API
//   const help = new Help( { shortcuts: () => [ { keys, title, group } ] (explore), items: () =>
//                            hint items, seen: () => bool, markSeen(), formats, touch: () => bool
//                            (default: a coarse pointer), container } )
//   help.show( on = true ) / toggle() / visible
//   help.showHint( { force = false } ) / hideHint( remember = true ) / hintVisible
//   help.setFormats( [ { name, extensions } ] )
//   help.dispose()
// Pure helpers (tested): groupShortcuts(), keyList(), withoutRestated(), isTypingTarget().
import { h, icon } from './Panel.js';

/** What the shell itself answers to. `keys`: alternatives, each shown as one key cap. `pointer`: a mouse gesture (left out on touch screens, which get TOUCH_SHORTCUTS). */
export const SHELL_SHORTCUTS = [
	{ group: 'Camera', keys: [ 'Drag' ], title: 'Orbit', pointer: true },
	{ group: 'Camera', keys: [ 'Right-drag', 'Shift-drag' ], title: 'Pan', pointer: true },
	{ group: 'Camera', keys: [ 'Wheel', 'Pinch' ], title: 'Zoom toward the pointer', pointer: true },
	{ group: 'Camera', keys: [ '←', '→', '↑', '↓' ], title: 'Orbit in steps' },
	{ group: 'Camera', keys: [ '+', '−' ], title: 'Zoom' },
	{ group: 'Camera', keys: [ 'F', 'Home' ], title: 'Fit the graph' },
	{ group: 'Graph', keys: [ 'Click' ], title: 'Select a node', pointer: true },
	{ group: 'Graph', keys: [ 'Double-click' ], title: 'Fly to a node, or fit', pointer: true },
	{ group: 'Graph', keys: [ 'Drag a node' ], title: 'Move it', pointer: true },
	{ group: 'Graph', keys: [ 'Shift-drag a node' ], title: 'Pin or unpin it', pointer: true },
	{ group: 'Graph', keys: [ 'Space' ], title: 'Pause or resume the layout' },
	{ group: 'Graph', keys: [ 'R' ], title: 'Reheat the layout' },
	{ group: 'Graph', keys: [ 'L' ], title: 'Labels: auto, all, off' },
	{ group: 'View', keys: [ 'P' ], title: 'Photo mode' },
	{ group: 'View', keys: [ 'H' ], title: 'Status chips' },
	{ group: 'View', keys: [ '?' ], title: 'This help' },
	{ group: 'View', keys: [ 'Esc' ], title: 'Close, or clear the selection' },
	{ group: 'View', keys: [ 'Drop a file' ], title: 'Open a graph', pointer: true },
];

/** The gestures of a touch screen: shown first, in place of the mouse gestures. */
export const TOUCH_SHORTCUTS = [
	{ group: 'Touch', keys: [ 'Drag' ], title: 'Orbit' },
	{ group: 'Touch', keys: [ 'Two fingers' ], title: 'Pan, and pinch to zoom' },
	{ group: 'Touch', keys: [ 'Tap' ], title: 'Select a node' },
	{ group: 'Touch', keys: [ 'Double-tap' ], title: 'Fly to a node, or fit' },
	{ group: 'Touch', keys: [ 'Drag a node' ], title: 'Move it' },
	{ group: 'Touch', keys: [ 'Long-press a node' ], title: 'Pin or unpin it' },
];

const TYPING = 'textarea, select, [contenteditable=""], [contenteditable="true"], input:not([type]), input[type="text"], input[type="search"], input[type="number"], input[type="email"], input[type="url"], input[type="password"]';

/** True when a key event belongs to a text field (shortcuts never fire while typing). */
export function isTypingTarget( target ) {

	return !! ( target && typeof target.closest === 'function' && target.closest( TYPING ) );

}

/** keys of a shortcut entry → list of key caps: arrays as given, 'Ctrl K' and '/' as one cap, 'A / B' and '1 2 3' (single signs) as several. */
export function keyList( keys ) {

	if ( Array.isArray( keys ) ) return keys.map( String ).filter( Boolean );
	if ( keys === undefined || keys === null || keys === '' ) return [];
	const text = String( keys ).trim();
	if ( text.length > 1 && / \/ | or /.test( text ) ) return text.split( / \/ | or / ).map( ( s ) => s.trim() ).filter( Boolean );
	// "1 2 3", "[ ]": alternatives of one sign each; "Ctrl K", "⌘ K", "Shift Click" are one combination
	const parts = text.split( /\s+/ );
	if ( parts.length > 1 && parts.every( ( t ) => /^[0-9[\]<>,.;'+\-=]$/.test( t ) ) ) return parts;
	return text ? [ text ] : [];

}

/** The entries of `own` that `extra` does not state again with the same keys (the more specific layer speaks for a key). */
export function withoutRestated( own, extra ) {

	const id = ( it ) => keyList( it.keys ).join( '|' ).toLowerCase();
	const taken = new Set( ( extra || [] ).filter( ( it ) => it && it.title ).map( id ) );
	return ( own || [] ).filter( ( it ) => ! taken.has( id( it ) ) );

}

/** Groups entries, keeping the order in which groups and entries first appear. */
export function groupShortcuts( ...lists ) {

	const groups = new Map();
	for ( const list of lists ) {

		for ( const it of list || [] ) {

			if ( ! it || ! it.title ) continue;
			const name = it.group || 'More';
			if ( ! groups.has( name ) ) groups.set( name, [] );
			groups.get( name ).push( { keys: keyList( it.keys ), title: String( it.title ) } );

		}

	}

	return [ ...groups ].map( ( [ title, items ] ) => ( { title, items } ) );

}

const listExtensions = ( formats ) => {

	const ext = [];
	for ( const f of formats || [] ) for ( const e of f.extensions || [] ) if ( ! ext.includes( e ) ) ext.push( e );
	return ext;

};

export class Help {

	constructor( options = {} ) {

		this.options = { container: typeof document !== 'undefined' ? document.body : null, ...options };
		this.visible = false;
		this.hintVisible = false;
		this._build();
		this.setFormats( options.formats || null );
		this._onKey = ( e ) => this._key( e );
		this._dismiss = ( e ) => {

			// keys that only modify, and anything typed into a field, do not count as "used it"
			if ( e.type === 'keydown' && ( e.key === 'Shift' || e.key === 'Control' || e.key === 'Alt' || e.key === 'Meta' || e.key === 'Tab' ) ) return;
			this.hideHint( true );

		};

	}

	// ── help overlay ──────────────────────────────────────────────────────────────────────

	show( on = true ) {

		on = !! on;
		if ( on === this.visible ) return;
		this.visible = on;
		if ( on ) {

			this._fill();
			this._returnTo = document.activeElement;
			this.overlay.classList.add( 'is-visible' );
			this.overlay.removeAttribute( 'inert' );
			window.addEventListener( 'keydown', this._onKey, true );
			this._close.focus( { preventScroll: true } );
			this.hideHint( true );

		} else {

			this.overlay.classList.remove( 'is-visible' );
			this.overlay.setAttribute( 'inert', '' );
			window.removeEventListener( 'keydown', this._onKey, true );
			const r = this._returnTo;
			if ( r && r.isConnected && r.focus ) r.focus( { preventScroll: true } );
			this._returnTo = null;

		}

		this.options.onToggle?.( on );

	}

	toggle() {

		this.show( ! this.visible );
		return this.visible;

	}

	// ── first-run hint ────────────────────────────────────────────────────────────────────

	showHint( { force = false } = {} ) {

		if ( this.hintVisible || ( ! force && this.options.seen?.() ) ) return false;
		const items = this.options.items ? this.options.items() : [];
		if ( ! items.length ) return false;
		this._hintList.textContent = '';
		for ( const it of items ) this._hintList.append( h( 'span.wgv-hint-item', null, it.key ? h( 'kbd.wgv-kbd', null, it.key ) : null, it.strong ? h( 'b', null, it.strong ) : null, it.text || null ) );
		this.hintVisible = true;
		this.hint.classList.add( 'is-visible' );
		this.hint.removeAttribute( 'inert' );
		// the hint has done its job as soon as the person does anything
		window.addEventListener( 'pointerdown', this._dismiss, true );
		window.addEventListener( 'wheel', this._dismiss, { capture: true, passive: true } );
		window.addEventListener( 'keydown', this._dismiss, true );
		window.addEventListener( 'drop', this._dismiss, true );
		return true;

	}

	hideHint( remember = true ) {

		if ( remember && this.hintVisible ) this.options.markSeen?.();
		if ( ! this.hintVisible ) return;
		this.hintVisible = false;
		this.hint.classList.remove( 'is-visible' );
		this.hint.setAttribute( 'inert', '' );
		window.removeEventListener( 'pointerdown', this._dismiss, true );
		window.removeEventListener( 'wheel', this._dismiss, true );
		window.removeEventListener( 'keydown', this._dismiss, true );
		window.removeEventListener( 'drop', this._dismiss, true );

	}

	// ── drop target ───────────────────────────────────────────────────────────────────────

	setFormats( formats ) {

		const ext = listExtensions( formats );
		const list = ext.length ? ext : [ '.json', '.txt', '.csv' ];
		this._formats.textContent = list.slice( 0, 12 ).join( '  ' ) + ( list.length > 12 ? '  …' : '' );

	}

	dispose() {

		this.show( false );
		this.hideHint( false );
		this.root.remove();

	}

	// ── internals ─────────────────────────────────────────────────────────────────────────

	_build() {

		this._close = h( 'button.wgv-icon', { type: 'button', 'aria-label': 'Close help', html: icon( 'close' ), onclick: () => this.show( false ) } );
		this._cols = h( 'div.wgv-help-cols' );
		this._foot = h( 'p.wgv-help-foot' );
		this.overlay = h( 'div.wgv-help', { inert: true, onclick: ( e ) => {

			if ( e.target === this.overlay ) this.show( false );

		} },
		h( 'div.wgv-help-card.wgv-glass', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Shortcuts and gestures' },
			h( 'div.wgv-help-head', null, h( 'h3', null, 'Shortcuts and gestures' ), this._close ),
			this._cols, this._foot ) );

		this._hintList = h( 'span.wgv-hint-items', { style: { display: 'contents' } } );
		this.hint = h( 'div.wgv-hint.wgv-glass', { role: 'note', 'aria-label': 'Getting started', inert: true },
			this._hintList,
			h( 'button.wgv-icon', { type: 'button', 'aria-label': 'Dismiss', html: icon( 'close', 14 ), onclick: () => this.hideHint( true ) } ) );

		this._formats = h( 'small.wgv-drop-formats' );
		this.dropzone = h( 'div.wgv-dropzone', { 'aria-hidden': 'true' },
			h( 'div.wgv-dropzone-card.wgv-glass', null,
				h( 'span', { html: icon( 'drop', 28 ) } ),
				h( 'b', null, 'Drop to open the graph' ),
				h( 'small', null, 'The file is read in this browser; nothing is uploaded.' ),
				this._formats ) );

		this.root = h( 'div.wgv-ui', null, this.overlay, this.hint, this.dropzone );
		this.options.container.append( this.root );

	}

	_fill() {

		let extra = [];
		try {

			extra = this.options.shortcuts ? this.options.shortcuts() || [] : [];

		} catch ( _ ) { /* a broken list must not keep the help closed */ }

		// touch screens: their gestures first, the mouse gestures left out; what the explore layer
		// states for a key replaces the shell's line for it
		const touch = this.options.touch ? !! this.options.touch() : typeof matchMedia === 'function' && matchMedia( '(pointer: coarse)' ).matches;
		const own = withoutRestated( touch ? SHELL_SHORTCUTS.filter( ( it ) => ! it.pointer ) : SHELL_SHORTCUTS, extra );
		const groups = groupShortcuts( touch ? TOUCH_SHORTCUTS : [], own, touch ? ( extra || [] ).filter( ( it ) => ! /click/i.test( String( it.keys ) ) ) : extra );
		this._cols.textContent = '';
		for ( const g of groups ) {

			const ul = h( 'ul' );
			for ( const it of g.items ) ul.append( h( 'li', null, h( 'span.wgv-keys', null, it.keys.map( ( k ) => h( 'kbd', null, k ) ) ), h( 'span', null, it.title ) ) );
			this._cols.append( h( 'div.wgv-help-col', null, h( 'h4', null, g.title ), ul ) );

		}

		this._foot.textContent = this.options.footer ? this.options.footer() : 'Settings are kept in this browser. The link button copies an address that reopens this view.';

	}

	_key( e ) {

		if ( e.key === 'Escape' || ( e.key === '?' && ! isTypingTarget( e.target ) ) ) {

			e.preventDefault();
			e.stopImmediatePropagation();
			this.show( false );

		} else if ( e.key === 'Tab' ) {

			// one focusable control: keep the focus in the dialog
			e.preventDefault();
			this._close.focus( { preventScroll: true } );

		}

	}

}
