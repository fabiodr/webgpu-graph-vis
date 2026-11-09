// Settings panel: a calm glass panel built from a declarative schema bound to the shared settings
// object. Desktop: top-left, collapsible. Phones (≤ 640 px wide): a bottom sheet with a grab handle,
// collapsed to a slim bar by default, draggable/tappable, safe-area aware, ≥ 40 px touch targets.
// Also owns the app-level overlays: loading indicator, toasts, keyboard help and the file-drop hint.
// Styles: src/ui/ui.css (import it once, e.g. `import './ui/ui.css'`).
//
// Schema (SETTINGS_SCHEMA): sections { id, title, collapsed?, items[] }; item types
//   slider    { key, label, min, max, step, format? }          double-click label resets to default
//   segmented { key, label?, options: [{ value, label }] }
//   toggle    { key, label, hint? }
//   palette   { key, label, options: [palette names] }        gradient swatches from util/color.js
//   actions   { items: [{ action, label, icon } | { key, labels: [off, on], icons: [off, on] }] }
//   dataset   (dataset picker + Open file… + Generate… sub-form + drop hint)
//   any item may carry visible( settings ) -> bool
// Generators (setGenerators / options.generators):
//   [{ id, name, description?, params: [{ key, label, type: 'int'|'float'|'select'|'bool',
//      min?, max?, step?, log?, value, options?: [{ value, label }] }] }]
//
// API
//   const panel = new Panel( { settings, onChange( key, value, settings ), onAction( name, payload ),
//                              onOpenFile( file ), manifest, generators, container, title, credit } )
//   panel.set( key, value )             update settings[key] + its control (no onChange)
//   panel.refresh()                     re-sync every control from settings
//   panel.setManifest( { groups, datasets } )
//   panel.setGenerators( list )
//   panel.setDatasetInfo( { id, name, nodes, edges, levels } )   for files/generated graphs
//   panel.update( stats )               { nodes, edges, clusters, levels, fps, alpha, running }
//                                       (throttled to ~2×/s; pass { force: true } to bypass)
//   panel.setLoading( message, fraction? )   fraction 0..1 or null (indeterminate); falsy hides
//   panel.toast( message, { type: 'error'|'info', duration } )
//   panel.showHelp( on ) / toggleHelp() / helpVisible
//   panel.setCollapsed( on ) / collapsed / isSheet
//   panel.setAccent( paletteName )      accent CSS variables (also on palette change)
//   panel.insets()                      { left, right, top, bottom } CSS px of canvas the panel
//                                       covers at rest; also sent to options.onInsets( insets )
//                                       on every change (wire to rig.setInsets)
//   panel.dispose()
// Actions sent to onAction: 'fit', 'reset', 'reheat', 'generate' ({ model, params }), 'open-file' (File).
// Toggling Pause/Resume goes through onChange( 'running', bool ). Sets --wgv-sheet-h on <html>
// (visible sheet height, 0 on desktop) so the node card can sit above the sheet.

import { paletteColor, linearToSrgb, PALETTES } from '../util/color.js';

export const DEFAULT_SETTINGS = {
	dataset: 'imsv/lesmis', layoutMode: 'force', running: true, autoRotate: true, immersive: false,
	bundling: 0.85, edgeOpacity: 1, edgeWidth: 1, edgeBlend: 'normal', nodeSize: 1, bloom: 0.8,
	palette: 'paper', labels: 'auto',
	charge: 1, linkDistance: 30, gravity: 1, theta: 0.8, damping: 0.4, quality: 'auto',
};

export const SETTINGS_SCHEMA = [
	{ id: 'dataset', title: 'Dataset', items: [ { type: 'dataset' } ] },
	{ id: 'layout', title: 'Layout', items: [
		{ type: 'segmented', key: 'layoutMode', big: true, options: [ { value: 'force', label: 'Force 3D' }, { value: 'sphere', label: 'Sphere (paper)' } ] },
		{ type: 'actions', items: [
			{ key: 'running', labels: [ 'Resume', 'Pause' ], icons: [ 'play', 'pause' ] },
			{ action: 'reheat', label: 'Reheat', icon: 'flame' },
			{ action: 'fit', label: 'Fit', icon: 'fit' },
			{ action: 'reset', label: 'Reset', icon: 'reset' },
		] },
		{ type: 'toggle', key: 'autoRotate', label: 'Auto-rotate' },
		{ type: 'toggle', key: 'immersive', label: 'Immersive view', hint: 'from the centre', visible: ( s ) => s.layoutMode === 'sphere' },
	] },
	{ id: 'look', title: 'Look', items: [
		{ type: 'slider', key: 'bundling', label: 'Bundling', min: 0, max: 1, step: 0.01 },
		{ type: 'slider', key: 'edgeOpacity', label: 'Edge opacity', min: 0, max: 1, step: 0.01 },
		{ type: 'slider', key: 'edgeWidth', label: 'Edge width', min: 0.25, max: 4, step: 0.05 },
		{ type: 'slider', key: 'nodeSize', label: 'Node size', min: 0.25, max: 4, step: 0.05 },
		{ type: 'slider', key: 'bloom', label: 'Bloom', min: 0, max: 2, step: 0.01 },
		{ type: 'palette', key: 'palette', label: 'Palette', options: [ 'paper', 'rainbow', 'aurora', 'ember' ] },
		{ type: 'segmented', key: 'labels', label: 'Labels', options: [ { value: 'auto', label: 'Auto' }, { value: 'all', label: 'All' }, { value: 'off', label: 'Off' } ] },
		{ type: 'segmented', key: 'edgeBlend', label: 'Edge blend', options: [ { value: 'normal', label: 'Normal' }, { value: 'additive', label: 'Additive' } ] },
		{ type: 'segmented', key: 'quality', label: 'Quality', options: [ { value: 'auto', label: 'Auto' }, { value: 'high', label: 'High' }, { value: 'low', label: 'Low' } ] },
	] },
	{ id: 'forces', title: 'Forces', collapsed: true, items: [
		{ type: 'slider', key: 'charge', label: 'Charge', min: 0, max: 4, step: 0.05 },
		{ type: 'slider', key: 'linkDistance', label: 'Link distance', min: 5, max: 150, step: 1 },
		{ type: 'slider', key: 'gravity', label: 'Gravity', min: 0, max: 4, step: 0.05 },
		{ type: 'slider', key: 'theta', label: 'Theta (BH)', min: 0.2, max: 1.6, step: 0.05 },
		{ type: 'slider', key: 'damping', label: 'Damping', min: 0.05, max: 0.95, step: 0.01 },
	] },
];

const PALETTE_ACCENT_T = { paper: 0.85, rainbow: 0.5, aurora: 0.25, ember: 0.75 };
const CREDIT = {
	text: 'Visual style after ImsvGraphVis',
	cite: 'Kwon et al., IEEE TVCG 2016',
	url: 'https://doi.org/10.1109/TVCG.2016.2520921',
};

// 16 px stroke icons
const ICONS = {
	chevron: '<path d="M4 6l4 4 4-4"/>',
	help: '<circle cx="8" cy="8" r="6.25"/><path d="M6.4 6.3a1.7 1.7 0 0 1 3.3.5c0 1.2-1.7 1.4-1.7 2.5"/><circle cx="8" cy="11.4" r=".35" fill="currentColor"/>',
	close: '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
	play: '<path d="M5.5 3.8v8.4L12 8z" fill="currentColor" stroke="none"/>',
	pause: '<path d="M5.5 4v8M10.5 4v8"/>',
	flame: '<path d="M8 14c2.5 0 4-1.6 4-3.8 0-2.6-2.2-3.6-2.6-6.2-1.6 1-2.2 2.6-2 4-.9-.4-1.4-1.2-1.5-2C4.7 7.2 4 8.6 4 10.2 4 12.4 5.5 14 8 14z"/>',
	fit: '<path d="M2.8 6V2.8H6M10 2.8h3.2V6M13.2 10v3.2H10M6 13.2H2.8V10"/><circle cx="8" cy="8" r="1.6"/>',
	reset: '<path d="M3.2 8a4.8 4.8 0 1 0 1.5-3.5"/><path d="M3 2.8v2.6h2.6"/>',
	file: '<path d="M4 2.5h5l3 3V13.5H4z"/><path d="M9 2.5v3h3"/>',
	spark: '<path d="M8 2.5l1.2 3.3 3.3 1.2-3.3 1.2L8 11.5 6.8 8.2 3.5 7l3.3-1.2zM12.5 11l.5 1.3 1.3.5-1.3.5-.5 1.3-.5-1.3-1.3-.5 1.3-.5z"/>',
	drop: '<path d="M8 2.5v8M4.8 7.5L8 10.7l3.2-3.2"/><path d="M3 12.5v1h10v-1"/>',
};

export function icon( name, size = 16 ) {

	return `<svg class="wgv-i" width="${size}" height="${size}" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[ name ] || ''}</svg>`;

}

/** Tiny DOM builder: h( 'div.cls1.cls2', { attr: v, onclick }, ...children ) */
export function h( tag, attrs = null, ...children ) {

	if ( attrs && ( attrs.nodeType || typeof attrs !== 'object' || Array.isArray( attrs ) ) ) {

		children.unshift( attrs ); // h( tag, child, … ) without an attributes object
		attrs = null;

	}

	const [ name, ...cls ] = tag.split( '.' );
	const el = document.createElement( name || 'div' );
	if ( cls.length ) el.className = cls.join( ' ' );
	if ( attrs ) {

		for ( const [ k, v ] of Object.entries( attrs ) ) {

			if ( v === undefined || v === null || v === false ) continue;
			if ( k.startsWith( 'on' ) && typeof v === 'function' ) el.addEventListener( k.slice( 2 ), v );
			else if ( k === 'html' ) el.innerHTML = v;
			else if ( k === 'style' && typeof v === 'object' ) {

				for ( const [ sk, sv ] of Object.entries( v ) ) {

					if ( sk.startsWith( '--' ) ) el.style.setProperty( sk, String( sv ) );
					else el.style[ sk ] = sv;

				}

			}
			else el.setAttribute( k, v === true ? '' : v );

		}

	}

	for ( const c of children.flat() ) if ( c !== null && c !== undefined && c !== false ) el.append( c.nodeType ? c : String( c ) );
	return el;

}

const toCss = ( lin, gain = 1 ) => {

	const c = lin.map( ( v ) => Math.round( 255 * linearToSrgb( Math.min( 1, Math.max( 0, v * gain ) ) ) ) );
	return `rgb(${c[ 0 ]} ${c[ 1 ]} ${c[ 2 ]})`;

};

/** CSS color of a palette at t, brightened like the glowing nodes (paper is stored dark). */
export function paletteCss( name, t ) {

	const c = paletteColor( name, t );
	const m = Math.max( c[ 0 ], c[ 1 ], c[ 2 ], 1e-6 );
	return toCss( c, Math.min( 4, 0.82 / m ) );

}

export function accentFor( name ) {

	const c = paletteColor( name, PALETTE_ACCENT_T[ name ] ?? 0.85 ).map( ( v ) => linearToSrgb( Math.min( 1, Math.max( 0, v ) ) ) );
	const m = Math.max( c[ 0 ], c[ 1 ], c[ 2 ], 1e-6 );
	// normalise brightness, then soften toward white so the accent reads calm on dark glass
	const rgb = c.map( ( v ) => Math.round( 255 * ( ( v / m ) * 0.86 * 0.72 + 0.28 * 0.92 ) ) );
	return rgb;

}

const fmtInt = ( n ) => ( n === undefined || n === null || Number.isNaN( n ) ? '–' : Math.round( n ).toLocaleString( 'en-US' ) );
const decimals = ( step ) => ( step >= 1 ? 0 : Math.min( 3, Math.ceil( - Math.log10( step ) - 1e-9 ) ) );

let uid = 0;

export class Panel {

	constructor( options = {} ) {

		this.options = { container: document.body, title: 'Graph · WebGPU', credit: CREDIT, generators: [], schema: SETTINGS_SCHEMA, ...options };
		this.settings = options.settings || { ...DEFAULT_SETTINGS };
		this.onChange = options.onChange || null;
		this.onAction = options.onAction || null;
		this.onOpenFile = options.onOpenFile || null;
		this.onInsets = options.onInsets || null;
		this.manifest = options.manifest || null;
		this.generators = options.generators || [];
		this.collapsed = false;
		this.isSheet = false;
		this.helpVisible = false;
		this._bindings = [];
		this._sections = {};
		this._lastStats = 0;
		this._customDataset = null;
		this._genState = {};
		this._genModel = null;
		this._build();
		this.setAccent( this.settings.palette );
		this.refresh();

	}

	// ── public API ────────────────────────────────────────────────────────────────────────

	set( key, value ) {

		this.settings[ key ] = value;
		this.refresh();

	}

	refresh() {

		for ( const b of this._bindings ) {

			b.sync?.( this.settings );
			if ( b.visible && b.row ) {

				const show = !! b.visible( this.settings );
				b.row.classList.toggle( 'is-hidden', ! show );
				b.row.toggleAttribute( 'inert', ! show );

			}

		}

	}

	setManifest( manifest ) {

		this.manifest = manifest;
		this._fillDatasets();
		this.refresh();

	}

	setGenerators( list ) {

		this.generators = list || [];
		this._buildGenerator();

	}

	setDatasetInfo( info ) {

		this._customDataset = info || null;
		this._fillDatasets();
		if ( info && info.id ) this.settings.dataset = info.id;
		this.refresh();

	}

	update( stats = {}, { force = false } = {} ) {

		const now = performance.now();
		if ( ! force && now - this._lastStats < 480 ) return;
		this._lastStats = now;
		const s = this._stats;
		s.nodes.textContent = fmtInt( stats.nodes );
		s.edges.textContent = fmtInt( stats.edges );
		s.clusters.textContent = stats.clusters !== undefined ? `${fmtInt( stats.clusters )}` : '–';
		s.levels.textContent = stats.levels !== undefined ? `${stats.levels}` : '–';
		s.fps.textContent = stats.fps ? Math.round( stats.fps ) : '–';
		const running = stats.running ?? this.settings.running;
		const alpha = Math.max( 0, Math.min( 1, stats.alpha ?? 0 ) );
		this._state.classList.toggle( 'is-running', !! running && alpha > 0.002 );
		s.status.textContent = running ? ( alpha > 0.002 ? 'Layout running' : 'Layout settled' ) : 'Layout paused';
		s.alpha.textContent = stats.alpha !== undefined ? `α ${alpha.toFixed( 3 )}` : '';
		s.alphaBar.style.transform = `scaleX(${Math.sqrt( alpha ).toFixed( 3 )})`;
		this._barCounts.textContent = stats.nodes !== undefined ? `${fmtInt( stats.nodes )} · ${fmtInt( stats.edges )}` : '';

	}

	setLoading( message, fraction = null ) {

		const L = this._loading;
		if ( ! message ) {

			L.root.classList.remove( 'is-visible' );
			this.root.classList.remove( 'is-loading' );
			return;

		}

		L.text.textContent = message;
		const det = typeof fraction === 'number' && isFinite( fraction );
		L.root.classList.toggle( 'is-indeterminate', ! det );
		L.bar.style.transform = det ? `scaleX(${Math.max( 0, Math.min( 1, fraction ) ).toFixed( 3 )})` : '';
		L.root.classList.add( 'is-visible' );
		this.root.classList.add( 'is-loading' );

	}

	toast( message, { type = 'error', duration = 5200 } = {} ) {

		const el = h( 'div.wgv-toast.wgv-glass', { role: type === 'error' ? 'alert' : 'status', 'data-type': type },
			h( 'i.wgv-toast-dot' ), h( 'span', null, String( message ) ),
			h( 'button.wgv-icon.wgv-toast-x', { 'aria-label': 'Dismiss', html: icon( 'close', 14 ) } ) );
		const remove = () => {

			el.classList.remove( 'is-in' );
			setTimeout( () => el.remove(), 260 );

		};

		el.querySelector( 'button' ).addEventListener( 'click', remove );
		this._toasts.append( el );
		while ( this._toasts.children.length > 4 ) this._toasts.firstChild.remove();
		requestAnimationFrame( () => el.classList.add( 'is-in' ) );
		if ( duration > 0 ) setTimeout( remove, duration );
		return el;

	}

	showHelp( on = true ) {

		this.helpVisible = !! on;
		this._help.classList.toggle( 'is-visible', this.helpVisible );
		this._help.toggleAttribute( 'inert', ! this.helpVisible );
		if ( this.helpVisible ) this._help.querySelector( 'button' )?.focus( { preventScroll: true } );

	}

	toggleHelp() {

		this.showHelp( ! this.helpVisible );

	}

	setCollapsed( on ) {

		this.collapsed = !! on;
		this.panel.classList.toggle( 'is-collapsed', this.collapsed );
		this._collapseBtn.setAttribute( 'aria-expanded', String( ! this.collapsed ) );
		this._collapseBtn.setAttribute( 'aria-label', this.collapsed ? 'Expand panel' : 'Collapse panel' );
		this._body.toggleAttribute( 'inert', this.collapsed );
		// phones: an open sheet owns the screen (the node card steps aside, see ui.css)
		document.documentElement.classList.toggle( 'wgv-sheet-open', this.isSheet && ! this.collapsed );
		this.panel.style.transform = '';
		this._updateSheetVar();
		this._emitInsets();

	}

	insets() {

		const r = { left: 0, right: 0, top: 0, bottom: 0 };
		if ( this.isSheet ) {

			// only the slim bar counts: the expanded sheet is transient
			r.bottom = this._head.offsetTop + this._head.offsetHeight + this._safeBottom();

		} else if ( ! this.collapsed && window.innerWidth >= 760 ) {

			r.left = this.panel.offsetLeft + this.panel.offsetWidth;

		}

		return r;

	}

	_emitInsets() {

		const r = this.insets();
		const key = `${r.left}|${r.right}|${r.top}|${r.bottom}`;
		if ( key === this._insetKey ) return;
		this._insetKey = key;
		this.onInsets?.( r );

	}

	setAccent( palette ) {

		const [ r, g, b ] = accentFor( palette || 'paper' );
		const root = document.documentElement.style;
		root.setProperty( '--wgv-accent', `rgb(${r} ${g} ${b})` );
		root.setProperty( '--wgv-accent-rgb', `${r} ${g} ${b}` );

	}

	dispose() {

		this._mq?.removeEventListener?.( 'change', this._onMq );
		window.removeEventListener( 'keydown', this._onKey, true );
		window.removeEventListener( 'resize', this._onResize );
		this.root.remove();
		document.documentElement.style.removeProperty( '--wgv-sheet-h' );
		document.documentElement.classList.remove( 'wgv-sheet-open', 'wgv-drop-active' );

	}

	// ── building ──────────────────────────────────────────────────────────────────────────

	_change( key, value ) {

		this.settings[ key ] = value;
		if ( key === 'palette' ) this.setAccent( value );
		this.refresh();
		this.onChange?.( key, value, this.settings );

	}

	_action( name, payload ) {

		this.onAction?.( name, payload );

	}

	_build() {

		const o = this.options;
		this.root = h( 'div.wgv-ui' );

		// header
		this._collapseBtn = h( 'button.wgv-icon.wgv-collapse', { 'aria-label': 'Collapse panel', 'aria-expanded': 'true', html: icon( 'chevron' ), onclick: ( e ) => {

			e.stopPropagation();
			this.setCollapsed( ! this.collapsed );

		} } );
		const [ t1, t2 ] = o.title.split( '·' ).map( ( s ) => s.trim() );
		this._dsName = h( 'span.wgv-sub-name' );
		this._barCounts = h( 'span.wgv-sub-counts' );
		this._head = h( 'header.wgv-head',
			h( 'div.wgv-grab', { 'aria-hidden': 'true' }, h( 'i' ) ),
			h( 'div.wgv-brand',
				h( 'i.wgv-logo', { 'aria-hidden': 'true' } ),
				h( 'div.wgv-brand-text',
					h( 'div.wgv-title', null, t1, t2 ? h( 'span.wgv-title-sep', null, '·' ) : null, t2 ? h( 'span.wgv-title-2', null, t2 ) : null ),
					h( 'div.wgv-sub', null, this._dsName, this._barCounts ) ) ),
			h( 'div.wgv-head-btns',
				h( 'button.wgv-icon', { 'aria-label': 'Keyboard and gestures', title: 'Help (?)', html: icon( 'help' ), onclick: ( e ) => {

					e.stopPropagation();
					this.toggleHelp();

				} } ),
				this._collapseBtn ) );

		// body
		this._body = h( 'div.wgv-body' );
		const scroller = h( 'div.wgv-scroll' );
		for ( const sec of o.schema ) scroller.append( this._buildSection( sec ) );
		scroller.append( this._buildFooter() );
		this._body.append( h( 'div.wgv-body-inner', null, scroller ) );

		this.panel = h( 'aside.wgv-panel.wgv-glass', { 'aria-label': 'Graph settings' }, this._head, this._body );
		this.root.append( this.panel );
		this._buildOverlays();
		o.container.append( this.root );
		this._setupResponsive();
		this._setupSheetDrag();

		this._onKey = ( e ) => {

			if ( e.key === 'Escape' && this.helpVisible ) {

				this.showHelp( false );
				e.preventDefault();

			}

		};

		window.addEventListener( 'keydown', this._onKey, true );

	}

	_buildSection( sec ) {

		const id = `wgv-sec-${sec.id}-${++ uid}`;
		const content = h( 'div.wgv-sec-c', { id } );
		const inner = h( 'div.wgv-sec-inner' );
		content.append( inner );
		const btn = h( 'button.wgv-sec-h', { 'aria-expanded': String( ! sec.collapsed ), 'aria-controls': id },
			h( 'span', null, sec.title ), h( 'span.wgv-sec-chev', { html: icon( 'chevron', 14 ) } ) );
		const el = h( 'section.wgv-sec', { 'data-section': sec.id }, btn, content );
		const setOpen = ( open ) => {

			el.classList.toggle( 'is-closed', ! open );
			btn.setAttribute( 'aria-expanded', String( open ) );
			content.toggleAttribute( 'inert', ! open );

		};

		btn.addEventListener( 'click', () => setOpen( el.classList.contains( 'is-closed' ) ) );
		setOpen( ! sec.collapsed );
		this._sections[ sec.id ] = { el, setOpen };
		for ( const item of sec.items ) {

			const row = this._buildItem( item );
			if ( row ) inner.append( row );

		}

		return el;

	}

	_buildItem( item ) {

		switch ( item.type ) {

			case 'slider': return this._slider( item );
			case 'segmented': return this._segmented( item );
			case 'toggle': return this._toggle( item );
			case 'palette': return this._palette( item );
			case 'actions': return this._actions( item );
			case 'dataset': return this._dataset();

		}

		return null;

	}

	_bind( binding ) {

		this._bindings.push( binding );
		return binding.row;

	}

	_slider( item, { get, set, defaultValue } = {} ) {

		const id = `wgv-s-${++ uid}`;
		const dec = item.decimals ?? decimals( item.step ?? 0.01 );
		const log = !! item.log;
		const toSlider = ( v ) => ( log ? ( Math.log( v / item.min ) / Math.log( item.max / item.min ) ) * 1000 : v );
		const fromSlider = ( s ) => {

			let v = log ? item.min * Math.pow( item.max / item.min, s / 1000 ) : s;
			const step = item.step ?? 0.01;
			v = Math.round( v / step ) * step;
			return + v.toFixed( dec );

		};

		const input = h( 'input.wgv-range', { id, type: 'range', min: log ? 0 : item.min, max: log ? 1000 : item.max, step: log ? 1 : ( item.step ?? 0.01 ) } );
		const out = h( 'output.wgv-val', { for: id } );
		const label = h( 'label.wgv-l', { for: id, title: 'Double-click to reset' }, item.label );
		const row = h( 'div.wgv-row.wgv-slider', null, label, input, out );
		const getV = get || ( () => this.settings[ item.key ] );
		const setV = set || ( ( v ) => this._change( item.key, v ) );
		const def = defaultValue ?? DEFAULT_SETTINGS[ item.key ];
		const fmt = item.format || ( ( v ) => ( dec === 0 ? fmtInt( v ) : Number( v ).toFixed( dec ) ) );
		const paint = ( v ) => {

			const s = toSlider( v );
			const lo = + input.min, hi = + input.max;
			input.style.setProperty( '--p', `${( ( 100 * ( s - lo ) ) / ( hi - lo || 1 ) ).toFixed( 2 )}%` );
			out.textContent = fmt( v );

		};

		input.addEventListener( 'input', () => {

			const v = fromSlider( + input.value );
			paint( v );
			setV( v );

		} );
		label.addEventListener( 'dblclick', () => {

			if ( def === undefined ) return;
			input.value = toSlider( def );
			paint( def );
			setV( def );

		} );
		return this._bind( {
			row, visible: item.visible,
			sync: () => {

				const v = getV();
				if ( v === undefined ) return;
				if ( document.activeElement !== input ) input.value = toSlider( v );
				paint( v );

			},
		} );

	}

	_segmented( item, { get, set } = {} ) {

		const n = item.options.length;
		const seg = h( 'div.wgv-seg', { role: 'radiogroup', 'aria-label': item.label || item.key, style: { '--n': n } } );
		const buttons = item.options.map( ( opt, i ) => {

			const b = h( 'button', { role: 'radio', 'aria-checked': 'false', 'data-value': String( opt.value ) }, opt.label );
			b.addEventListener( 'click', () => ( set ? set( opt.value ) : this._change( item.key, opt.value ) ) );
			b.addEventListener( 'keydown', ( e ) => {

				const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? - 1 : 0;
				if ( ! d ) return;
				e.preventDefault();
				const next = item.options[ ( i + d + n ) % n ];
				set ? set( next.value ) : this._change( item.key, next.value );
				buttons[ ( i + d + n ) % n ].focus();

			} );
			seg.append( b );
			return b;

		} );
		const row = item.label
			? h( 'div.wgv-row.wgv-segrow', null, h( 'span.wgv-l', null, item.label ), seg )
			: h( 'div.wgv-row.wgv-segrow.is-full' + ( item.big ? '.is-big' : '' ), null, seg );
		const getV = get || ( () => this.settings[ item.key ] );
		return this._bind( {
			row, visible: item.visible,
			sync: () => {

				const v = getV();
				let idx = 0;
				buttons.forEach( ( b, i ) => {

					const on = item.options[ i ].value === v;
					b.setAttribute( 'aria-checked', String( on ) );
					b.tabIndex = on ? 0 : - 1;
					if ( on ) idx = i;

				} );
				seg.style.setProperty( '--i', idx );

			},
		} );

	}

	_toggle( item, { get, set } = {} ) {

		const sw = h( 'i.wgv-switch', { 'aria-hidden': 'true' } );
		const row = h( 'button.wgv-row.wgv-toggle', { role: 'switch', 'aria-checked': 'false' },
			h( 'span.wgv-l', null, item.label, item.hint ? h( 'small.wgv-hint-inline', null, item.hint ) : null ), sw );
		const getV = get || ( () => this.settings[ item.key ] );
		row.addEventListener( 'click', () => {

			const v = ! getV();
			set ? set( v ) : this._change( item.key, v );

		} );
		return this._bind( {
			row, visible: item.visible,
			sync: () => row.setAttribute( 'aria-checked', String( !! getV() ) ),
		} );

	}

	_palette( item ) {

		const wrap = h( 'div.wgv-swatches', { role: 'radiogroup', 'aria-label': item.label } );
		const name = h( 'span.wgv-swatch-name' );
		const buttons = item.options.map( ( p ) => {

			const stops = [];
			for ( let k = 0; k <= 6; k ++ ) stops.push( paletteCss( p, k / 6 ) );
			const b = h( 'button.wgv-swatch', { role: 'radio', title: PALETTES[ p ]?.label || p, 'aria-label': PALETTES[ p ]?.label || p, style: { '--g': `linear-gradient(90deg, ${stops.join( ', ' )})` } } );
			b.addEventListener( 'click', () => this._change( item.key, p ) );
			wrap.append( b );
			return b;

		} );
		const row = h( 'div.wgv-row.wgv-palette', null, h( 'span.wgv-l', null, item.label, name ), wrap );
		return this._bind( {
			row, visible: item.visible,
			sync: ( s ) => {

				buttons.forEach( ( b, i ) => b.setAttribute( 'aria-checked', String( item.options[ i ] === s[ item.key ] ) ) );
				name.textContent = ( PALETTES[ s[ item.key ] ]?.label || s[ item.key ] || '' ).split( ' (' )[ 0 ];

			},
		} );

	}

	_actions( item ) {

		const row = h( 'div.wgv-row.wgv-actions', { style: { '--n': item.items.length } } );
		for ( const a of item.items ) {

			const ic = h( 'span.wgv-btn-i' );
			const lb = h( 'span.wgv-btn-l' );
			const b = h( 'button.wgv-btn', null, ic, lb );
			if ( a.key ) {

				b.addEventListener( 'click', () => this._change( a.key, ! this.settings[ a.key ] ) );
				this._bind( {
					row: null,
					sync: ( s ) => {

						const on = s[ a.key ] ? 1 : 0;
						if ( b._state === on ) return;
						b._state = on;
						ic.innerHTML = icon( a.icons[ on ] );
						lb.textContent = a.labels[ on ];
						b.setAttribute( 'aria-pressed', String( ! on ) );

					},
				} );

			} else {

				ic.innerHTML = icon( a.icon );
				lb.textContent = a.label;
				b.addEventListener( 'click', () => this._action( a.action ) );

			}

			row.append( b );

		}

		return this._bind( { row, visible: item.visible } );

	}

	_dataset() {

		const select = h( 'select.wgv-select-el', { 'aria-label': 'Dataset' } );
		select.addEventListener( 'change', () => this._change( 'dataset', select.value ) );
		this._dsSelect = select;
		this._dsMeta = h( 'div.wgv-meta' );
		this._dsDesc = h( 'div.wgv-desc' );
		const fileInput = h( 'input', { type: 'file', accept: '.json,.txt,.csv', hidden: true } );
		fileInput.addEventListener( 'change', () => {

			const f = fileInput.files && fileInput.files[ 0 ];
			if ( f ) {

				if ( this.onOpenFile ) this.onOpenFile( f );
				else this._action( 'open-file', f );

			}

			fileInput.value = '';

		} );
		this._genBox = h( 'div.wgv-gen-inner' );
		this._gen = h( 'div.wgv-gen.is-closed', null, this._genBox );
		this._genBtn = h( 'button.wgv-btn.wgv-btn-row', { 'aria-expanded': 'false', html: `${icon( 'spark' )}<span>Generate…</span>` } );
		this._genBtn.addEventListener( 'click', () => {

			const open = this._gen.classList.toggle( 'is-closed' ) === false;
			this._genBtn.setAttribute( 'aria-expanded', String( open ) );
			this._gen.toggleAttribute( 'inert', ! open );

		} );
		this._gen.toggleAttribute( 'inert', true );
		const row = h( 'div.wgv-dataset', null,
			h( 'div.wgv-select', null, select, h( 'span.wgv-select-chev', { html: icon( 'chevron', 14 ) } ) ),
			this._dsMeta, this._dsDesc,
			h( 'div.wgv-btns2', null,
				h( 'button.wgv-btn.wgv-btn-row', { html: `${icon( 'file' )}<span>Open file…</span>`, onclick: () => fileInput.click() } ),
				this._genBtn ),
			this._gen,
			h( 'div.wgv-drop-hint', null, h( 'span', { html: icon( 'drop', 14 ) } ), 'or drop a .json, .txt or .csv file anywhere' ),
			fileInput );
		this._fillDatasets();
		this._buildGenerator();
		return this._bind( { row, sync: ( s ) => this._syncDataset( s ) } );

	}

	_fillDatasets() {

		const select = this._dsSelect;
		if ( ! select ) return;
		select.textContent = '';
		const m = this.manifest;
		const groups = new Map();
		const add = ( group, label, value, text ) => {

			if ( ! groups.has( group ) ) {

				const og = h( 'optgroup', { label } );
				groups.set( group, og );
				select.append( og );

			}

			groups.get( group ).append( h( 'option', { value }, text ) );

		};

		if ( this._customDataset ) {

			const c = this._customDataset;
			add( '_custom', 'Loaded', c.id, c.name || c.id );

		}

		if ( m && m.datasets ) {

			for ( const d of m.datasets ) add( d.group, ( m.groups && m.groups[ d.group ] ) || d.group, d.id, d.name );

		}

		if ( ! select.options.length ) add( '_none', 'Datasets', '', 'Loading…' );

	}

	_datasetInfo( id ) {

		if ( this._customDataset && this._customDataset.id === id ) return this._customDataset;
		return this.manifest?.datasets?.find( ( d ) => d.id === id ) || null;

	}

	_syncDataset( s ) {

		const select = this._dsSelect;
		if ( [ ...select.options ].some( ( o ) => o.value === s.dataset ) ) select.value = s.dataset;
		const d = this._datasetInfo( s.dataset );
		this._dsName.textContent = d ? d.name : '';
		const parts = [];
		if ( d ) {

			if ( d.nodes !== undefined ) parts.push( `${fmtInt( d.nodes )} nodes` );
			if ( d.edges !== undefined ) parts.push( `${fmtInt( d.edges )} edges` );
			if ( d.levels ) parts.push( `${d.levels} level${d.levels > 1 ? 's' : ''}` );

		}

		this._dsMeta.textContent = parts.join( ' · ' );
		this._dsDesc.textContent = d && d.description ? d.description : '';
		this._dsDesc.title = d && d.source ? d.source : '';

	}

	_buildGenerator() {

		const box = this._genBox;
		if ( ! box ) return;
		box.textContent = '';
		const gens = this.generators;
		this._genBtn.hidden = ! gens.length;
		if ( ! gens.length ) return;
		if ( ! this._genModel || ! gens.some( ( g ) => g.id === this._genModel ) ) this._genModel = gens[ 0 ].id;
		for ( const g of gens ) {

			if ( ! this._genState[ g.id ] ) this._genState[ g.id ] = Object.fromEntries( ( g.params || [] ).map( ( p ) => [ p.key, p.value ] ) );

		}

		const model = gens.find( ( g ) => g.id === this._genModel );
		if ( gens.length > 1 ) {

			const sel = h( 'select.wgv-select-el', { 'aria-label': 'Generator model' } );
			for ( const g of gens ) sel.append( h( 'option', { value: g.id }, g.name ) );
			sel.value = model.id;
			sel.addEventListener( 'change', () => {

				this._genModel = sel.value;
				this._buildGenerator();

			} );
			box.append( h( 'div.wgv-select.wgv-select-sm', null, sel, h( 'span.wgv-select-chev', { html: icon( 'chevron', 14 ) } ) ) );

		}

		if ( model.description ) box.append( h( 'div.wgv-desc', null, model.description ) );
		const state = this._genState[ model.id ];
		const rows = [];
		for ( const p of model.params || [] ) {

			const get = () => state[ p.key ];
			const set = ( v ) => {

				state[ p.key ] = v;
				for ( const r of rows ) r.sync?.();

			};

			let b;
			if ( p.type === 'select' ) {

				this._segmented( { label: p.label, options: p.options }, { get, set } );
				b = this._bindings.pop();

			} else if ( p.type === 'bool' ) {

				this._toggle( { label: p.label }, { get, set } );
				b = this._bindings.pop();

			} else {

				const step = p.step ?? ( p.type === 'int' ? 1 : 0.01 );
				this._slider( { label: p.label, min: p.min, max: p.max, step, log: p.log, format: p.format }, { get, set, defaultValue: p.value } );
				b = this._bindings.pop();

			}

			rows.push( b );
			box.append( b.row );

		}

		for ( const r of rows ) r.sync?.();
		const go = h( 'button.wgv-btn.wgv-primary', { html: `${icon( 'spark' )}<span>Generate graph</span>` } );
		go.addEventListener( 'click', () => this._action( 'generate', { model: model.id, params: { ...state } } ) );
		box.append( go );

	}

	_buildFooter() {

		const cell = ( label ) => {

			const b = h( 'b' );
			return [ h( 'div.wgv-stat', null, b, h( 'span', null, label ) ), b ];

		};

		const [ cN, nodes ] = cell( 'nodes' );
		const [ cE, edges ] = cell( 'edges' );
		const [ cC, clusters ] = cell( 'clusters' );
		const [ cL, levels ] = cell( 'levels' );
		const [ cF, fps ] = cell( 'fps' );
		const status = h( 'span.wgv-status-t', null, 'Layout' );
		const alpha = h( 'span.wgv-alpha' );
		const alphaBar = h( 'i' );
		this._stats = { nodes, edges, clusters, levels, fps, status, alpha, alphaBar };
		this._state = h( 'div.wgv-state', null, h( 'i.wgv-pulse' ), status, alpha, h( 'div.wgv-alpha-bar', null, alphaBar ) );
		const c = this.options.credit;
		return h( 'footer.wgv-foot', null,
			h( 'div.wgv-stats', null, cN, cE, cC, cL, cF ),
			this._state,
			c ? h( 'div.wgv-credit', null, `${c.text} — `, h( 'a', { href: c.url, target: '_blank', rel: 'noopener' }, c.cite ) ) : null );

	}

	_buildOverlays() {

		// loading
		const text = h( 'span.wgv-loading-t' );
		const bar = h( 'i' );
		const lroot = h( 'div.wgv-loading.wgv-glass', { role: 'status', 'aria-live': 'polite' },
			h( 'div.wgv-loading-row', null, h( 'i.wgv-spinner' ), text ), h( 'div.wgv-progress', null, bar ) );
		this._loading = { root: lroot, text, bar };

		this._toasts = h( 'div.wgv-toasts', { 'aria-live': 'polite' } );

		// help
		const k = ( ...keys ) => h( 'span.wgv-keys', null, keys.map( ( x ) => h( 'kbd', null, x ) ) );
		const row = ( keys, text ) => h( 'li', null, keys, h( 'span', null, text ) );
		const col = ( title, items ) => h( 'div.wgv-help-col', null, h( 'h4', null, title ), h( 'ul', null, items ) );
		const card = h( 'div.wgv-help-card.wgv-glass', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Controls' },
			h( 'div.wgv-help-head', null, h( 'h3', null, 'Controls' ),
				h( 'button.wgv-icon', { 'aria-label': 'Close help', html: icon( 'close' ), onclick: () => this.showHelp( false ) } ) ),
			h( 'div.wgv-help-cols', null,
				col( 'Mouse', [
					row( k( 'drag' ), 'Rotate' ), row( k( 'right-drag' ), 'Pan' ), row( k( 'wheel' ), 'Zoom to cursor' ),
					row( k( 'click' ), 'Select node · empty clears' ), row( k( 'drag node' ), 'Move node' ),
					row( k( '⇧', 'drag node' ), 'Move and pin / unpin' ), row( k( 'double-click' ), 'Focus node · empty fits' ),
				] ),
				col( 'Touch', [
					row( k( '1 finger' ), 'Rotate · look around' ), row( k( 'pinch' ), 'Zoom and pan' ), row( k( 'tap' ), 'Select node' ),
					row( k( 'drag node' ), 'Move node' ), row( k( 'long-press' ), 'Pin / unpin node' ), row( k( 'double-tap' ), 'Focus · fit' ),
				] ),
				col( 'Keys', [
					row( k( 'Esc' ), 'Clear selection' ), row( k( 'F' ), 'Fit graph' ), row( k( 'Space' ), 'Pause / resume layout' ),
					row( k( 'R' ), 'Reheat layout' ), row( k( 'L' ), 'Cycle labels' ), row( k( '?' ), 'This help' ),
				] ) ),
			h( 'p.wgv-help-foot', null, 'Files: drop a graph (.json, .txt edge list, .csv) anywhere on the page.' ) );
		this._help = h( 'div.wgv-help', { inert: true }, card );
		this._help.addEventListener( 'pointerdown', ( e ) => {

			if ( e.target === this._help ) this.showHelp( false );

		} );

		const drop = h( 'div.wgv-dropzone', { 'aria-hidden': 'true' },
			h( 'div.wgv-dropzone-card.wgv-glass', null, h( 'span', { html: icon( 'drop', 22 ) } ), h( 'b', null, 'Drop to open graph' ), h( 'small', null, '.json · .txt · .csv' ) ) );
		this.root.append( lroot, this._toasts, this._help, drop );

	}

	// ── responsive: desktop panel vs. bottom sheet ───────────────────────────────────────

	_setupResponsive() {

		const apply = () => {

			const sheet = !! this._mq?.matches;
			const changed = sheet !== this.isSheet;
			this.isSheet = sheet;
			this.root.classList.toggle( 'is-sheet', sheet );
			this.panel.classList.toggle( 'is-sheet', sheet );
			if ( changed ) this.setCollapsed( sheet ); // sheet starts as a slim bar, desktop expanded
			this._updateSheetVar();
			this._emitInsets();

		};

		this._mq = typeof matchMedia === 'function' ? matchMedia( '(max-width: 640px)' ) : null;
		this._onMq = apply;
		this._mq?.addEventListener?.( 'change', apply );
		this._onResize = () => {

			this._updateSheetVar();
			this._emitInsets();

		};
		window.addEventListener( 'resize', this._onResize );
		this.isSheet = null;
		apply();
		this.panel.addEventListener( 'transitionend', ( e ) => {

			if ( e.target === this.panel ) this._updateSheetVar();

		} );

	}

	/** Publishes the visible sheet height (--wgv-sheet-h): the resting height of the current
	 * state, or the live height while the sheet is being dragged. */
	_updateSheetVar( live = false ) {

		let v = 0;
		if ( this.isSheet ) {

			const bar = this._head.offsetTop + this._head.offsetHeight;
			if ( bar > 0 ) this.panel.style.setProperty( '--wgv-bar-h', `${Math.round( bar )}px` );
			if ( live ) v = Math.max( 0, window.innerHeight - this.panel.getBoundingClientRect().top );
			else v = this.collapsed ? bar + this._safeBottom() : this.panel.offsetHeight;

		}

		document.documentElement.style.setProperty( '--wgv-sheet-h', `${Math.round( v )}px` );

	}

	_setupSheetDrag() {

		const head = this._head;
		let drag = null;
		head.addEventListener( 'click', ( e ) => {

			// tapping the bar of the collapsed sheet opens it
			if ( this.isSheet && ! e.target.closest( 'button' ) && ! this._dragJustEnded ) this.setCollapsed( ! this.collapsed );

		} );
		head.addEventListener( 'pointerdown', ( e ) => {

			if ( ! this.isSheet || e.target.closest( 'button' ) || ( e.pointerType === 'mouse' && e.button !== 0 ) ) return;
			const rect = this.panel.getBoundingClientRect();
			const barH = head.getBoundingClientRect().bottom - rect.top;
			const max = Math.max( 0, rect.height - barH - this._safeBottom() );
			drag = { id: e.pointerId, y0: e.clientY, base: this.collapsed ? max : 0, max, off: this.collapsed ? max : 0, t: performance.now(), v: 0, moved: false };
			head.setPointerCapture?.( e.pointerId );

		} );
		head.addEventListener( 'pointermove', ( e ) => {

			if ( ! drag || e.pointerId !== drag.id ) return;
			const dy = e.clientY - drag.y0;
			if ( ! drag.moved && Math.abs( dy ) < 6 ) return;
			if ( ! drag.moved ) {

				drag.moved = true;
				this.panel.classList.add( 'is-dragging' );

			}

			const now = performance.now();
			let off = drag.base + dy;
			if ( off < 0 ) off *= 0.3; // rubber band above fully open
			off = Math.min( drag.max, off );
			drag.v = ( off - drag.off ) / Math.max( 1, now - drag.t );
			drag.off = off; drag.t = now;
			this.panel.style.transform = `translateY(${off.toFixed( 1 )}px)`;
			this._updateSheetVar( true );

		} );
		const end = ( e ) => {

			if ( ! drag || e.pointerId !== drag.id ) return;
			const d = drag;
			drag = null;
			this.panel.classList.remove( 'is-dragging' );
			if ( ! d.moved ) return;
			this._dragJustEnded = true;
			setTimeout( () => {

				this._dragJustEnded = false;

			}, 50 );
			const collapse = d.v > 0.35 ? true : d.v < - 0.35 ? false : d.off > d.max * 0.5;
			this.panel.style.transform = '';
			this.setCollapsed( collapse );

		};

		head.addEventListener( 'pointerup', end );
		head.addEventListener( 'pointercancel', end );

	}

	_safeBottom() {

		const v = parseFloat( getComputedStyle( this.panel ).paddingBottom ) || 0;
		return Math.max( 0, v - 8 ); // .wgv-panel.is-sheet padding-bottom = 8px + safe area

	}

}
