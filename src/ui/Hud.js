// Status chips and tool buttons (App module, order 42).
//
// Status (slots.status, bottom left of the free area; under the top row on phones): a few quiet
// chips that answer "is it still working, and how hard": frame rate (+ GPU time where the adapter
// has timestamp queries), the resolved quality tier with the render scale, the state of the layout
// (running with its progress, settled, paused, computing) and the size of the graph. Texts are
// refreshed four times a second and only when they changed; nothing is read back from the DOM.
// Tool buttons (slots.topRight): small glass buttons, arranged with CSS `order` (the explore layer
// uses 0…9, the shell 10 and up) and inserted into the DOM in that order, so the Tab order is
// the order on screen.
//
// API
//   const hud = app.use( new Hud( { slots, layout, visible = true } ) )
//   hud.setVisible( on ) / hud.visible / hud.toggle()      the status chips
//   hud.addTool( { id, icon, label, shortcut?, order = 10, desktop = false, onClick } ) -> button
//                                       `desktop`: not shown on phones
//   hud.setTool( id, { icon?, label?, pressed? } )
//   hud.state                           what the chips show: { fps, gpuMs, tier, auto, scale,
//                                       layout: 'running'|'settled'|'paused'|'computing'|'moving',
//                                       progress (0..1 or null), nodes, edges }
//   hud.dispose()
// Pure helpers (tested): layoutProgress( alpha, alphaMin, plateau ), layoutState( layout ),
// formatCount( n ).
import { h, icon } from './Panel.js';

const cap = ( s ) => ( s ? s[ 0 ].toUpperCase() + s.slice( 1 ) : '' );

/** 12 345 → "12.3k", 1 234 567 → "1.23M": counts that fit a chip. */
export function formatCount( n ) {

	if ( ! Number.isFinite( n ) ) return '–';
	if ( n < 10000 ) return Math.round( n ).toLocaleString( 'en-US' );
	if ( n < 1e6 ) return ( n / 1e3 ).toFixed( n < 1e5 ? 1 : 0 ) + 'k';
	return ( n / 1e6 ).toFixed( n < 1e7 ? 2 : 1 ) + 'M';

}

/**
 * Progress of a cooling run from its alpha: the schedule holds alpha = 1 for the first `plateau`
 * of the ticks and then cools exponentially to alphaMin, so the logarithm of alpha is linear in
 * the remaining ticks. `held` (0..1) is how much of the plateau has passed.
 */
export function layoutProgress( alpha, alphaMin = 0.001, plateau = 0.3, held = 1 ) {

	if ( ! ( alpha > 0 ) ) return 1;
	if ( alpha >= 1 ) return plateau * Math.min( 1, Math.max( 0, held ) );
	const t = Math.log( alpha ) / Math.log( Math.max( 1e-9, Math.min( 0.999, alphaMin ) ) );
	return Math.min( 1, plateau + ( 1 - plateau ) * Math.max( 0, t ) );

}

/** One word for what the layout is doing. */
export function layoutState( layout ) {

	if ( ! layout ) return 'settled';
	if ( layout.pending ) return 'computing';
	if ( layout.paused ) return 'paused';
	if ( layout.settled ) return 'settled';
	const mode = layout.layoutMode;
	return mode === 'force' || mode === 'flat' || mode === undefined ? 'running' : 'moving';

}

// one word for a layout in motion: 'moving' (a blend to fixed targets) reads "running" too, without the progress bar
const LAYOUT_LABEL = { running: 'Layout running', settled: 'Layout settled', paused: 'Layout paused', computing: 'Computing layout', moving: 'Layout running' };

export class Hud {

	constructor( { slots, layout = null, visible = true } = {} ) {

		this.order = 42;
		this.slots = slots;
		this.layout = layout;
		this.visible = !! visible;
		this.state = { fps: 0, gpuMs: null, tier: '', auto: false, scale: 1, layout: 'settled', progress: null, nodes: 0, edges: 0 };
		this._tools = new Map();
		this._acc = 1; // 1 = refresh in the next frame
		this._at = 0;
		this._held = 0;
		this._text = {};
		this._build();

	}

	init( app ) {

		this.app = app;

	}

	setGraph( buffers, graph ) {

		this.state.nodes = graph ? graph.N : 0;
		this.state.edges = graph ? graph.E : 0;
		this._acc = 1;

	}

	update( dt ) {

		// the plateau of a run has no signal of its own: count it in ticks of the clock
		const L = this.layout;
		if ( L && L.alpha >= 1 && ! L.paused && ! L.settled ) this._held = Math.min( 1, this._held + dt / 1.6 );
		else if ( ! L || L.settled || L.alpha < 1 ) this._held = L && L.alpha < 1 ? 1 : 0;
		// four times a second by the clock: frame time is clamped, and a slow device is exactly where
		// the chips have to keep up
		const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
		if ( this._acc < 1 && now - this._at < 250 ) return;
		this._acc = 0;
		this._at = now;
		this._refresh();

	}

	setVisible( on ) {

		this.visible = !! on;
		this.root.hidden = ! this.visible;
		if ( this.visible ) this._refresh();

	}

	toggle() {

		this.setVisible( ! this.visible );
		return this.visible;

	}

	addTool( { id, icon: name, label, shortcut = null, order = 10, desktop = false, onClick } ) {

		const b = h( 'button.wgv-ui.wgv-tool.wgv-glass' + ( desktop ? '.is-desktop' : '' ), {
			type: 'button', 'aria-label': label, title: shortcut ? `${label} (${shortcut})` : label, 'data-tool': id, html: icon( name ),
			style: { order: String( order ) }, onclick: ( e ) => onClick?.( e ),
		} );
		this._tools.get( id )?.el.remove();
		this._tools.set( id, { el: b, shortcut } );
		// in the DOM in the order shown, so the Tab order follows the screen
		const after = Array.from( this.slots.topRight.children ).find( ( el ) => ( parseFloat( el.style.order ) || 0 ) > order );
		this.slots.topRight.insertBefore( b, after || null );
		return b;

	}

	setTool( id, { icon: name, label, pressed } = {} ) {

		const t = this._tools.get( id );
		if ( ! t ) return;
		if ( name ) t.el.innerHTML = icon( name );
		if ( label ) {

			t.el.setAttribute( 'aria-label', label );
			t.el.title = t.shortcut ? `${label} (${t.shortcut})` : label;

		}

		if ( pressed !== undefined ) t.el.setAttribute( 'aria-pressed', String( !! pressed ) );

	}

	dispose() {

		this.root.remove();
		for ( const t of this._tools.values() ) t.el.remove();
		this._tools.clear();

	}

	// ── internals ─────────────────────────────────────────────────────────────────────────

	_build() {

		const chip = ( name, ...children ) => h( 'div.wgv-chip.wgv-glass', { 'data-chip': name }, ...children );
		const E = this._el = {
			fps: h( 'b' ), fpsUnit: h( 'span', null, 'fps' ), gpu: h( 'small' ),
			tier: h( 'span' ), scale: h( 'small' ),
			dot: h( 'i.wgv-chip-dot' ), layout: h( 'span' ), bar: h( 'i' ),
			counts: h( 'span' ),
		};
		E.barWrap = h( 'span.wgv-chip-bar', { hidden: true }, E.bar );
		E.layoutChip = chip( 'layout', E.dot, E.layout, E.barWrap );
		E.countsChip = chip( 'counts', E.counts );
		this.root = h( 'div.wgv-ui.wgv-status', { role: 'status', 'aria-label': 'Status', 'aria-live': 'off' },
			chip( 'fps', E.fps, E.fpsUnit, E.gpu ),
			chip( 'quality', E.tier, E.scale ),
			E.layoutChip,
			E.countsChip );
		this.root.hidden = ! this.visible;
		this.slots.status.append( this.root );

	}

	_set( name, el, text ) {

		if ( this._text[ name ] === text ) return;
		this._text[ name ] = text;
		el.textContent = text;

	}

	_refresh() {

		const app = this.app;
		if ( ! app ) return;
		const S = this.state, G = app.gfx ? app.gfx.stats : null, L = this.layout;
		S.fps = app.fps || 0;
		S.gpuMs = G && typeof G.gpuMs === 'number' ? G.gpuMs : null;
		S.tier = G ? G.tier : '';
		S.auto = !! G && G.setting === 'auto';
		S.scale = G ? G.renderScale : 1;
		S.layout = layoutState( L );
		S.progress = S.layout === 'running' && L ? layoutProgress( L.alpha, L.alphaMin, 0.3, this._held ) : null;
		if ( ! this.visible ) return;

		const E = this._el;
		// without a graph (nothing could be loaded yet) there is no layout and nothing to count
		const empty = ! app.graph;
		if ( this._text.empty !== empty ) {

			this._text.empty = empty;
			E.layoutChip.hidden = E.countsChip.hidden = empty;

		}

		this._set( 'fps', E.fps, S.fps ? ( S.fps < 9.5 ? S.fps.toFixed( 1 ) : String( Math.round( S.fps ) ) ) : '–' );
		this._set( 'gpu', E.gpu, S.gpuMs !== null ? `${S.gpuMs.toFixed( S.gpuMs < 10 ? 1 : 0 )} ms GPU` : '' );
		this._set( 'tier', E.tier, S.tier ? ( S.auto ? `Auto · ${cap( S.tier )}` : cap( S.tier ) ) : '–' );
		this._set( 'scale', E.scale, `${Math.round( 100 * S.scale )}%` );
		this._set( 'layout', E.layout, LAYOUT_LABEL[ S.layout ] );
		if ( this._text.state !== S.layout ) {

			this._text.state = S.layout;
			E.layoutChip.classList.toggle( 'is-running', S.layout === 'running' || S.layout === 'moving' );
			E.layoutChip.classList.toggle( 'is-busy', S.layout === 'computing' );
			E.barWrap.hidden = S.layout !== 'running';

		}

		if ( S.progress !== null ) {

			const p = S.progress.toFixed( 2 );
			if ( this._text.p !== p ) {

				this._text.p = p;
				E.bar.style.transform = `scaleX(${p})`;

			}

		}

		this._set( 'counts', E.counts, `${formatCount( S.nodes )} nodes · ${formatCount( S.edges )} edges` );

	}

}
