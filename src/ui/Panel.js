// Settings panel: a calm glass panel built from a declarative layout over the settings schema
// (src/ui/settings.js) and bound to the shared settings object. Desktop: top-left, collapsible.
// Phones (≤ 640 px wide): a bottom sheet with a grab handle, collapsed to a slim bar by default,
// draggable / tappable, safe-area aware, ≥ 40 px touch targets. Also owns the loading indicator
// and the toasts. Styles: src/ui/ui.css (import it once, e.g. `import './ui/ui.css'`).
//
// Progressive disclosure: only Dataset and Layout are open at first; every other section is one
// line with a summary of its state ("Edges · hierarchy"), and the rarely needed parameters sit
// behind an "Advanced" row inside their section. Which sections and groups are open is remembered
// (options.ui). Controls whose feature is missing (options.features / setFeatures) are not shown.
//
// Layout (PANEL_LAYOUT): sections { id, open?, summary( settings, info ), items[] }; an item is
//   'key'                       a setting of the schema: number → slider, enum → segmented,
//                               bool → toggle (label, range, options, format, hint, visibility
//                               come from the schema)
//   { key, …overrides }         the same with overrides ( full: true = control on its own row )
//   { type: 'choice', key, icons, labels?, describe( value, settings, info ) }   icon tiles (with
//                               short labels where the schema's are long) + one line
//   { type: 'palette', key }    gradient swatches from util/color.js
//   { type: 'actions', items: [ { action, label, icon } | { key, labels: [off, on], icons } ] }
//   { type: 'preset' }          structure preset of the layout (LAYOUT_PRESETS)
//   { type: 'look' }            whole look (LOOKS of the settings model) with one line that says
//                               what the current one is: reported as the action 'look' with the
//                               id, because it changes several settings at once
//   { type: 'note', id, text( settings, info ) }   one or two quiet lines, hidden when empty
//   { type: 'status', id }      live state from info[ id ]: { busy, text, action: { name, label } }
//   { type: 'group', id, title, items }           "Advanced" disclosure
//   { type: 'dataset' }         dataset picker + Open file… + Generate… sub-form + drop hint
//   any object item may carry visible( settings, features, info ) -> bool
// Generators (setGenerators / options.generators):
//   [{ id, name, description?, params: [{ key, label, type: 'int'|'float'|'select'|'bool',
//      min?, max?, step?, log?, value, options?: [{ value, label }] }] }]
//
// API
//   const panel = new Panel( { settings, write( key, value ), onChange( key, value, settings ),
//                              onAction( name, payload ), onOpenFile( file ), onInsets( insets ),
//                              manifest, generators, features, ui: { get( name ), set( name, value ) },
//                              help, container, title, credit } )
//     write     who writes a setting when a control changes. Without it the panel writes into
//               `settings` itself. An app whose settings live in a store passes the store's setter:
//               the store stays the only writer (it validates, remembers the choice and tells its
//               listeners), and the panel shows what the store holds afterwards.
//     onChange  after the value is written and the controls are in step: apply it.
//   panel.set( key, value )             show a value that changed elsewhere (no onChange); without
//                                       `write` it is also stored in settings[key]
//   panel.refresh()                     re-sync every control from settings
//   panel.setFeatures( { name: bool, formats: [ { name, extensions } ] } )   feature flags of the
//                                       schema (`needs`) and the file formats of the drop hint
//   panel.setInfo( { id: value } )      live state for notes, summaries and status rows: layout
//                                       { mode, settled, pending, running }, quality { text },
//                                       render { text }, clustering { text, count }, bundle /
//                                       recluster { busy, text, action }, edgeStyle (resolved)
//   panel.openSection( id, open = true )
//   panel.setManifest( { groups, datasets } )
//   panel.setGenerators( list )
//   panel.setDatasetInfo( { id, name, nodes, edges, levels } )   for files / generated graphs
//   panel.update( stats )               { nodes, edges, communities, levels } for the footer
//                                       (throttled to ~2×/s; pass { force: true } to bypass)
//   panel.setLoading( message, fraction? )   fraction 0..1 or null (indeterminate); falsy hides
//   panel.toast( message, { type: 'error'|'info', duration, action: { label, run() } } )
//   panel.showHelp( on ) / toggleHelp() / helpVisible     forwarded to options.help (ui/Help.js)
//   panel.setCollapsed( on ) / collapsed / isSheet
//   panel.setAccent( paletteName )      accent CSS variables (also on palette change)
//   panel.insets()                      { left, right, top, bottom } CSS px of canvas the panel
//                                       covers at rest; also sent to options.onInsets( insets )
//                                       on every change (wire to rig.setInsets)
//   panel.dispose()
// Actions sent to onAction: 'fit', 'reset', 'reheat', 'generate' ( { model, params } ),
// 'open-file' ( File ), 'look' ( id ), 'reset-setting' ( key ) (a double click on the label of a
// slider, a segmented control or the palette), 'reset-all', and the names of status actions.
// Toggling Pause / Resume goes through onChange( 'running', bool ). Sets --wgv-sheet-h on <html>
// (visible sheet height, 0 on desktop) so other surfaces can sit above the sheet.
//
// Slider drags are cheap: the thumb and the number are painted at once, the change is applied
// once per animation frame, and only the controls that depend on the changed key are re-synced
// (no layout is read while dragging).

import { paletteColor, linearToSrgb, PALETTES } from '../util/color.js';
import { SECTIONS, LOOKS, settingSpec, settingDefaults, isVisible, lookOf } from './settings.js';

export const DEFAULT_SETTINGS = settingDefaults();

/** Layout presets of the force model (LAYOUT_PRESETS of ForceLayout.js; kept here so the panel has no GPU imports). */
export const STRUCTURE_PRESETS = [
	{ value: 'default', label: 'Balanced', clusterSeparation: 0.7, cohesion: 0.5 },
	{ value: 'graph-only', label: 'Graph only', clusterSeparation: 1, cohesion: 0 },
	{ value: 'communities', label: 'Communities', clusterSeparation: 0.4, cohesion: 1.5 },
];

const LAYOUT_TEXT = {
	force: 'Hierarchy-aware force layout in space',
	flat: 'The same force layout in a plane',
	sphere: 'Clusters on a sphere; the selection moves inward',
	radial: 'Shells by distance from the selected node',
	stress: 'Distances match graph distances (meshes, rings, trees)',
};

const cap = ( s ) => ( s ? s[ 0 ].toUpperCase() + s.slice( 1 ) : '' );
/** The schema's name of a palette, with what the colour module says about it in brackets. */
const paletteLabel = ( p ) => {

	const about = /\(.*\)/.exec( PALETTES[ p ]?.label || '' );
	return optionLabel( 'palette', p ) + ( about ? ' ' + about[ 0 ] : '' );

};

const optionLabel = ( key, value ) => settingSpec( key )?.options?.find( ( o ) => o.value === value )?.label || cap( String( value ) );

export const PANEL_LAYOUT = [
	{ id: 'dataset', open: true, items: [ { type: 'dataset' } ] },
	{ id: 'layout', open: true,
		summary: ( s ) => optionLabel( 'layoutMode', s.layoutMode ) + ( s.running ? '' : ' · paused' ),
		items: [
			{ type: 'choice', key: 'layoutMode', icons: { force: 'modeForce', flat: 'modeFlat', sphere: 'modeSphere', radial: 'modeRadial', stress: 'modeStress' },
				labels: { force: 'Force', flat: 'Flat' }, // one word per tile: five of them share the width of the panel
				describe: ( v, s, info ) => {

					if ( info.layout?.pending ) return { text: 'Computing the stress layout…', busy: true };
					if ( v === 'radial' ) return { text: LAYOUT_TEXT.radial + ( info.selection ? '' : '. Select a node to choose the centre' ) };
					return { text: LAYOUT_TEXT[ v ] || '' };

				} },
			{ type: 'actions', items: [
				{ key: 'running', labels: [ 'Resume', 'Pause' ], icons: [ 'play', 'pause' ] },
				{ action: 'reheat', label: 'Reheat', icon: 'flame' },
				{ action: 'fit', label: 'Fit', icon: 'fit' },
				{ action: 'reset', label: 'Reset', icon: 'reset' },
			] },
			'immersive',
			{ type: 'preset', visible: ( s ) => s.layoutMode === 'force' || s.layoutMode === 'flat' },
			{ type: 'group', id: 'layout-advanced', title: 'Advanced', items: [ 'clusterSeparation', 'cohesion', 'hubSpacing', 'layoutSize', 'repulsion', 'damping' ] },
		] },
	{ id: 'communities',
		summary: ( s, info ) => ( info.clustering?.count ? `${info.clustering.count.toLocaleString( 'en-US' )} found` : '' ),
		items: [
			{ type: 'note', id: 'clustering', text: ( s, info ) => info.clustering?.text || '' },
			'clusterMethod', 'clusterResolution',
			{ type: 'status', id: 'recluster' },
			'fieldStrength', 'clusterLabels',
		] },
	{ id: 'edges',
		summary: ( s, info ) => [ optionLabel( 'bundleMode', s.bundleMode ), info.features?.edgeStyle ? optionLabel( 'edgeStyle', info.edgeStyle || s.edgeStyle ) : null ].filter( Boolean ).join( ' · ' ),
		items: [
			{ key: 'bundleMode', full: true },
			{ type: 'status', id: 'bundle' },
			'bundling', 'edgeStyle', 'edgeFlow',
			{ type: 'group', id: 'edges-advanced', title: 'Advanced', items: [ 'edgeOpacity', 'edgeWidth', 'edgeBlend' ] },
		] },
	{ id: 'nodes',
		// "Aurora palette", not "Aurora": a look and a theme carry the same name two rows further down
		summary: ( s ) => ( s.palette ? optionLabel( 'palette', s.palette ) + ' palette' : '' ),
		items: [ 'nodeSize', 'labels', 'nodeGloss', 'nodeGlow', { type: 'palette', key: 'palette' } ] },
	{ id: 'scene',
		summary: ( s, info ) => ( info.features?.env ? optionLabel( 'theme', s.theme ) : '' ),
		items: [ { type: 'look' }, { key: 'theme', full: true, caption: 'Theme' }, 'keyLight', 'envLight', 'fog', 'dust' ] },
	{ id: 'camera',
		summary: ( s ) => ( s.dof === 'auto' ? '' : `Focus ${s.dof}` ),
		items: [
			'autoRotate', 'dof', 'dofStrength', 'bloom', 'exposure',
			{ type: 'group', id: 'camera-advanced', title: 'Advanced', items: [ 'aoStrength', 'grain', 'vignette' ] },
		] },
	{ id: 'quality',
		summary: ( s, info ) => ( info.quality?.tier ? ( s.quality === 'auto' ? `Auto · ${cap( info.quality.tier )}` : cap( info.quality.tier ) ) : optionLabel( 'quality', s.quality ) ),
		items: [
			{ key: 'quality', full: true },
			{ type: 'note', id: 'quality', text: ( s, info ) => info.quality?.text || '' },
			'dynamicResolution', 'showStatus',
			{ type: 'note', id: 'render', mono: true, text: ( s, info ) => info.render?.text || '' },
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
	camera: '<path d="M2.5 5.5h2.4l1-1.6h4.2l1 1.6h2.4v7h-11z"/><circle cx="8" cy="8.8" r="2.3"/>',
	link: '<path d="M6.8 9.2a2.6 2.6 0 0 0 3.7 0l2-2a2.6 2.6 0 0 0-3.7-3.7l-.7.7"/><path d="M9.2 6.8a2.6 2.6 0 0 0-3.7 0l-2 2a2.6 2.6 0 0 0 3.7 3.7l.7-.7"/>',
	expand: '<path d="M2.8 6V2.8H6M10 2.8h3.2V6M13.2 10v3.2H10M6 13.2H2.8V10"/>',
	shrink: '<path d="M6 2.8V6H2.8M13.2 6H10V2.8M10 13.2V10h3.2M2.8 10H6v3.2"/>',
	check: '<path d="M3.5 8.4l3 3 6-6.6"/>',
	focus: '<circle cx="8" cy="8" r="2.2"/><path d="M8 1.8v2.4M8 11.8v2.4M1.8 8h2.4M11.8 8h2.4"/>',
	download: '<path d="M8 2.5v8M4.8 7.5L8 10.7l3.2-3.2"/><path d="M3 13.2h10"/>',
	gauge: '<path d="M2.6 11.5a5.6 5.6 0 1 1 10.8 0"/><path d="M8 9.6l2.6-3.4"/><circle cx="8" cy="9.8" r=".9" fill="currentColor" stroke="none"/>',
	// layout modes
	modeForce: '<circle cx="8" cy="3.1" r="1.25"/><circle cx="3" cy="11.6" r="1.25"/><circle cx="13" cy="11.6" r="1.25"/><circle cx="8.3" cy="8.5" r=".95" fill="currentColor" stroke="none"/><path d="M7.4 4.2L3.7 10.5M8.6 4.2l3.7 6.3M4.3 11.6h7.4"/>',
	modeFlat: '<path d="M1.8 11.8l3.2-6.6h9.2L11 11.8z"/><circle cx="6.2" cy="9.6" r=".9" fill="currentColor" stroke="none"/><circle cx="9.8" cy="7.4" r=".9" fill="currentColor" stroke="none"/><path d="M6.9 9.2l2.2-1.4"/>',
	modeSphere: '<circle cx="8" cy="8" r="5.6"/><ellipse cx="8" cy="8" rx="5.6" ry="2.1"/><ellipse cx="8" cy="8" rx="2.1" ry="5.6"/>',
	modeRadial: '<circle cx="8" cy="8" r="1.1" fill="currentColor" stroke="none"/><circle cx="8" cy="8" r="3.3"/><circle cx="8" cy="8" r="5.8" stroke-dasharray="2.2 1.7"/>',
	modeStress: '<path d="M2.8 4.6c3.4-1 7-1 10.4 0M2.8 8c3.4-1 7-1 10.4 0M2.8 11.4c3.4-1 7-1 10.4 0M4.4 3.6c-.8 3-.8 6 0 9M8 3v9.6M11.6 3.6c.8 3 .8 6 0 9"/>',
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

/** The structure preset that the two parameters match, or 'custom'. */
export function structurePreset( s ) {

	const p = STRUCTURE_PRESETS.find( ( q ) => Math.abs( q.clusterSeparation - s.clusterSeparation ) < 1e-6 && Math.abs( q.cohesion - s.cohesion ) < 1e-6 );
	return p ? p.value : 'custom';

}

/** One line about a look ( an id of LOOKS, or 'custom' ): what it sets together. */
export function lookText( id ) {

	const look = LOOKS.find( ( l ) => l.value === id );
	return look ? look.hint : 'Your own mix of theme, palette and effects';

}

/** Keys of the panel layout, per section (the schema's `section` must agree; see the tests). */
export function layoutKeys( layout = PANEL_LAYOUT ) {

	const out = {};
	const walk = ( items, list ) => {

		for ( const it of items ) {

			if ( typeof it === 'string' ) list.push( it );
			else if ( it.type === 'group' ) walk( it.items, list );
			else if ( it.type === 'actions' ) {

				for ( const a of it.items ) if ( a.key ) list.push( a.key );

			} else if ( it.key ) list.push( it.key );
			else if ( it.type === 'dataset' ) list.push( 'dataset' );

		}

	};

	for ( const sec of layout ) walk( sec.items, ( out[ sec.id ] = [] ) );
	return out;

}

const fmtInt = ( n ) => ( n === undefined || n === null || Number.isNaN( n ) ? '–' : Math.round( n ).toLocaleString( 'en-US' ) );
const decimals = ( step ) => ( step >= 1 ? 0 : Math.min( 3, Math.ceil( - Math.log10( step ) - 1e-9 ) ) );
const listFormats = ( formats ) => {

	const ext = [];
	for ( const f of formats || [] ) for ( const e of f.extensions || [] ) if ( ! ext.includes( e ) ) ext.push( e );
	return ext;

};

let uid = 0;

export class Panel {

	constructor( options = {} ) {

		this.options = { container: typeof document !== 'undefined' ? document.body : null, title: 'Graph · WebGPU', credit: CREDIT, generators: [], layout: PANEL_LAYOUT, ...options };
		this.settings = options.settings || { ...DEFAULT_SETTINGS };
		this.onChange = options.onChange || null;
		this._write = options.write || null;
		this.onAction = options.onAction || null;
		this.onOpenFile = options.onOpenFile || null;
		this.onInsets = options.onInsets || null;
		this.manifest = options.manifest || null;
		this.generators = options.generators || [];
		this.features = { ...( options.features || {} ) };
		this.info = { features: this.features };
		this.help = options.help || null;
		this.ui = options.ui || null;
		this.collapsed = false;
		this.isSheet = false;
		this._bindings = [];
		this._byKey = new Map();
		this._sections = {};
		this._lastStats = 0;
		this._customDataset = null;
		this._genState = {};
		this._genModel = null;
		this._pending = new Map(); // slider changes waiting for the next frame
		this._raf = 0;
		this._build();
		this.setAccent( this.settings.palette );
		this.refresh();

	}

	// ── public API ────────────────────────────────────────────────────────────────────────

	get helpVisible() {

		return !! this.help?.visible;

	}

	set( key, value ) {

		if ( ! this._write ) this.settings[ key ] = value; // with a writer the settings are its to change
		this._pending.delete( key );
		if ( key === 'palette' ) this.setAccent( this.settings[ key ] );
		this._sync( key );

	}

	refresh() {

		for ( const b of this._bindings ) this._syncBinding( b );
		this._syncSummaries();

	}

	setFeatures( features = {} ) {

		Object.assign( this.features, features );
		if ( features.formats ) this._syncFormats();
		this.refresh();

	}

	setInfo( info = {} ) {

		let changed = false;
		for ( const [ k, v ] of Object.entries( info ) ) {

			if ( k === 'features' ) continue;
			const a = this.info[ k ];
			if ( a === v || ( a && v && typeof a === 'object' && JSON.stringify( a ) === JSON.stringify( v ) ) ) continue;
			this.info[ k ] = v;
			changed = true;

		}

		if ( ! changed ) return;
		// live rows show the info; for every other control only its visibility can depend on it
		for ( const b of this._bindings ) if ( b.live || b.visible ) this._syncBinding( b, ! b.live );
		this._syncSummaries();

	}

	openSection( id, open = true ) {

		this._sections[ id ]?.setOpen( open, true );

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

		this._customDataset = info && ! this.manifest?.datasets?.some( ( d ) => d.id === info.id ) ? info : null;
		this._fillDatasets();
		if ( info && info.id && ! this._write ) this.settings.dataset = info.id;
		this._liveDataset = info || null;
		this.refresh();

	}

	update( stats = {}, { force = false } = {} ) {

		const now = performance.now();
		if ( ! force && now - this._lastStats < 480 ) return;
		this._lastStats = now;
		const s = this._stats;
		const text = ( el, t ) => {

			if ( el.textContent !== t ) el.textContent = t;

		};

		text( s.nodes, fmtInt( stats.nodes ) );
		text( s.edges, fmtInt( stats.edges ) );
		text( s.communities, stats.communities !== undefined ? fmtInt( stats.communities ) : '–' );
		text( s.levels, stats.levels !== undefined ? `${stats.levels}` : '–' );
		text( this._barCounts, stats.nodes !== undefined ? `${fmtInt( stats.nodes )} · ${fmtInt( stats.edges )}` : '' );

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

	toast( message, { type = 'error', duration = 5200, action = null } = {} ) {

		const act = action && typeof action.run === 'function' ? h( 'button.wgv-link.wgv-toast-act', { type: 'button' }, action.label || 'Undo' ) : null;
		const close = h( 'button.wgv-icon.wgv-toast-x', { type: 'button', 'aria-label': 'Dismiss', html: icon( 'close', 14 ) } );
		const el = h( 'div.wgv-toast.wgv-glass', { role: type === 'error' ? 'alert' : 'status', 'data-type': type },
			h( 'i.wgv-toast-dot' ), h( 'span', null, String( message ) ), act, close );
		const remove = () => {

			el.classList.remove( 'is-in' );
			setTimeout( () => el.remove(), 260 );

		};

		close.addEventListener( 'click', remove );
		act?.addEventListener( 'click', () => {

			remove();
			action.run();

		} );
		this._toasts.append( el );
		while ( this._toasts.children.length > 4 ) this._toasts.firstChild.remove();
		requestAnimationFrame( () => el.classList.add( 'is-in' ) );
		if ( duration > 0 ) setTimeout( remove, duration );
		return el;

	}

	showHelp( on = true ) {

		this.help?.show( on );

	}

	toggleHelp() {

		this.help?.toggle();

	}

	setCollapsed( on ) {

		this.collapsed = !! on;
		this.panel.classList.toggle( 'is-collapsed', this.collapsed );
		this._collapseBtn.setAttribute( 'aria-expanded', String( ! this.collapsed ) );
		this._collapseBtn.setAttribute( 'aria-label', this.collapsed ? 'Expand settings' : 'Collapse settings' );
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
		window.removeEventListener( 'resize', this._onResize );
		cancelAnimationFrame( this._raf );
		this.root.remove();
		document.documentElement.style.removeProperty( '--wgv-sheet-h' );
		document.documentElement.classList.remove( 'wgv-sheet-open' );

	}

	// ── change plumbing ───────────────────────────────────────────────────────────────────

	_change( key, value ) {

		// the owner of the settings writes them (a store validates the value, remembers the choice
		// and tells its listeners); a panel over a plain object writes it itself
		if ( this._write ) this._write( key, value );
		else this.settings[ key ] = value;
		const v = this.settings[ key ];
		if ( key === 'palette' ) this.setAccent( v );
		this._sync( key );
		this.onChange?.( key, v, this.settings );

	}

	/** Slider drags: paint now, apply once per frame. */
	_queue( key, value ) {

		this._pending.set( key, value );
		if ( this._raf ) return;
		this._raf = requestAnimationFrame( () => {

			this._raf = 0;
			const pending = [ ...this._pending ];
			this._pending.clear();
			for ( const [ k, v ] of pending ) this._change( k, v );

		} );

	}

	_action( name, payload ) {

		this.onAction?.( name, payload );

	}

	/** Re-syncs what depends on one key: its own controls, visibility rules, summaries. */
	_sync( key ) {

		for ( const b of this._byKey.get( key ) || [] ) this._syncBinding( b );
		for ( const b of this._bindings ) if ( b.key !== key && ( b.visible || b.live ) ) this._syncBinding( b, ! b.live );
		this._syncSummaries();

	}

	_syncBinding( b, visibilityOnly = false ) {

		if ( b.visible && b.row ) {

			const show = !! b.visible( this.settings, this.features, this.info );
			if ( show !== b._shown ) {

				b._shown = show;
				b.row.classList.toggle( 'is-hidden', ! show );
				b.row.toggleAttribute( 'inert', ! show );

			}

		}

		if ( ! visibilityOnly ) b.sync?.( this.settings );

	}

	_syncSummaries() {

		for ( const id in this._sections ) {

			const sec = this._sections[ id ];
			if ( ! sec.summary ) continue;
			let text = '';
			try {

				text = sec.summary( this.settings, this.info ) || '';

			} catch ( _ ) { /* a summary must never break the panel */ }

			if ( text !== sec._text ) {

				sec._text = text;
				sec.summaryEl.textContent = text;

			}

			// a section whose controls are all hidden disappears
			const any = sec.rows.some( ( b ) => ! b.visible || b._shown !== false );
			if ( any !== sec._any ) {

				sec._any = any;
				sec.el.classList.toggle( 'is-empty', ! any );

			}

		}

	}

	// ── building ──────────────────────────────────────────────────────────────────────────

	_build() {

		const o = this.options;
		this.root = h( 'div.wgv-ui' );

		// header
		this._collapseBtn = h( 'button.wgv-icon.wgv-collapse', { 'aria-label': 'Collapse settings', 'aria-expanded': 'true', html: icon( 'chevron' ), onclick: ( e ) => {

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
			h( 'div.wgv-head-btns', null, this._collapseBtn ) );

		// body
		this._body = h( 'div.wgv-body' );
		const scroller = h( 'div.wgv-scroll' );
		const titles = Object.fromEntries( SECTIONS.map( ( s ) => [ s.id, s.title ] ) );
		for ( const sec of o.layout ) scroller.append( this._buildSection( { title: titles[ sec.id ] || cap( sec.id ), ...sec } ) );
		scroller.append( this._buildFooter() );
		this._body.append( h( 'div.wgv-body-inner', null, scroller ) );

		this.panel = h( 'aside.wgv-panel.wgv-glass', { 'aria-label': 'Graph settings' }, this._head, this._body );
		this.root.append( this.panel );
		this._buildOverlays();
		o.container.append( this.root );
		this._setupResponsive();
		this._setupSheetDrag();

	}

	_buildSection( sec ) {

		const id = `wgv-sec-${sec.id}-${++ uid}`;
		const content = h( 'div.wgv-sec-c', { id } );
		const inner = h( 'div.wgv-sec-inner' );
		content.append( inner );
		const summaryEl = h( 'span.wgv-sec-sum' );
		const btn = h( 'button.wgv-sec-h', { 'aria-expanded': 'true', 'aria-controls': id },
			h( 'span.wgv-sec-t', null, sec.title ), summaryEl, h( 'span.wgv-sec-chev', { html: icon( 'chevron', 14 ) } ) );
		const el = h( 'section.wgv-sec', { 'data-section': sec.id }, btn, content );
		const setOpen = ( open, remember = false ) => {

			el.classList.toggle( 'is-closed', ! open );
			btn.setAttribute( 'aria-expanded', String( open ) );
			content.toggleAttribute( 'inert', ! open );
			if ( remember ) this._remember( 'sec:' + sec.id, open );

		};

		btn.addEventListener( 'click', () => setOpen( el.classList.contains( 'is-closed' ), true ) );
		setOpen( this._recall( 'sec:' + sec.id, !! sec.open ) );
		const first = this._bindings.length;
		this._buildItems( sec.items, inner );
		this._sections[ sec.id ] = { el, setOpen, summary: sec.summary || null, summaryEl, rows: this._bindings.slice( first ).filter( ( b ) => b.row && b.top ) };
		return el;

	}

	_recall( name, fallback ) {

		const all = this.ui?.get( 'open' );
		return all && typeof all[ name ] === 'boolean' ? all[ name ] : fallback;

	}

	_remember( name, open ) {

		if ( ! this.ui ) return;
		this.ui.set( 'open', { ...( this.ui.get( 'open' ) || {} ), [ name ]: open } );

	}

	_buildItems( items, parent, top = true ) {

		for ( const raw of items ) {

			const item = typeof raw === 'string' ? { key: raw } : raw;
			const before = this._bindings.length;
			const row = this._buildItem( item );
			if ( ! row ) continue;
			for ( const b of this._bindings.slice( before ) ) if ( b.row === row ) b.top = top;
			parent.append( row );

		}

	}

	_buildItem( item ) {

		const spec = item.key ? settingSpec( item.key ) : null;
		let type = item.type;
		if ( ! type && spec ) type = spec.type === 'number' ? 'slider' : spec.type === 'enum' ? 'segmented' : spec.type === 'bool' ? 'toggle' : null;
		// the schema's visibility (feature flag + rule) and the item's own rule
		const visible = spec && ( spec.needs || spec.visible ) || item.visible
			? ( s, f, info ) => ( ! spec || isVisible( spec, s, f ) ) && ( ! item.visible || item.visible( s, f, info ) )
			: null;
		const merged = spec ? { label: spec.label, hint: spec.hint, min: spec.min, max: spec.max, step: spec.step, log: spec.log, format: spec.format, options: spec.options, ...item, visible } : { ...item, visible };
		switch ( type ) {

			case 'slider': return this._slider( merged );
			case 'segmented': return this._segmented( merged );
			case 'toggle': return this._toggle( merged );
			case 'palette': return this._palette( { ...merged, options: ( merged.options || [] ).map( ( o ) => o.value ?? o ) } );
			case 'choice': return this._choice( merged );
			case 'actions': return this._actions( merged );
			case 'preset': return this._preset( merged );
			case 'look': return this._look( merged );
			case 'note': return this._note( merged );
			case 'status': return this._status( merged );
			case 'group': return this._group( merged );
			case 'dataset': return this._dataset();

		}

		return null;

	}

	_bind( binding ) {

		this._bindings.push( binding );
		if ( binding.key ) {

			if ( ! this._byKey.has( binding.key ) ) this._byKey.set( binding.key, [] );
			this._byKey.get( binding.key ).push( binding );

		}

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
			return Math.min( item.max, Math.max( item.min, + v.toFixed( Math.max( dec, 3 ) ) ) );

		};

		const lo = log ? 0 : item.min, hi = log ? 1000 : item.max;
		const input = h( 'input.wgv-range', { id, type: 'range', min: lo, max: hi, step: log ? 1 : ( item.step ?? 0.01 ) } );
		const out = h( 'output.wgv-val', { for: id } );
		const label = h( 'label.wgv-l', { for: id, title: [ item.hint, 'Double-click to reset' ].filter( Boolean ).join( ' · ' ) }, item.label );
		const row = h( 'div.wgv-row.wgv-slider', null, label, input, out );
		const getV = get || ( () => this.settings[ item.key ] );
		const fmt = item.format || ( ( v ) => ( dec === 0 ? fmtInt( v ) : Number( v ).toFixed( dec ) ) );
		let shown;
		const paint = ( v ) => {

			if ( v === shown ) return;
			shown = v;
			input.style.setProperty( '--p', `${( ( 100 * ( toSlider( v ) - lo ) ) / ( hi - lo || 1 ) ).toFixed( 2 )}%` );
			const text = fmt( v );
			out.textContent = text;
			input.setAttribute( 'aria-valuetext', text );

		};

		input.addEventListener( 'input', () => {

			const v = fromSlider( + input.value );
			paint( v );
			if ( set ) set( v );
			else this._queue( item.key, v );

		} );
		if ( log ) {

			// a log slider runs over 0…1000 while its value is rounded to `step`: one native arrow
			// step would change nothing. The keys move by an even share of the range instead (about
			// 60 presses end to end), and by at least one value step
			const step = item.step ?? 0.01;
			const presses = Math.max( 8, Math.min( 60, Math.round( Math.log( item.max / item.min ) / Math.log( 1 + step / Math.sqrt( item.min * item.max ) ) ) ) );
			input.addEventListener( 'keydown', ( e ) => {

				const d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: - 1, ArrowDown: - 1, PageUp: 5, PageDown: - 5 }[ e.key ];
				if ( ! d || e.ctrlKey || e.metaKey || e.altKey ) return;
				e.preventDefault();
				const cur = fromSlider( + input.value );
				let next = fromSlider( Math.min( 1000, Math.max( 0, toSlider( cur ) + ( d * 1000 ) / presses ) ) );
				if ( Math.sign( next - cur ) !== Math.sign( d ) ) next = Math.min( item.max, Math.max( item.min, + ( cur + Math.sign( d ) * step ).toFixed( Math.max( dec, 3 ) ) ) );
				if ( next === cur ) return;
				input.value = toSlider( next );
				input.dispatchEvent( new Event( 'input', { bubbles: true } ) );

			} );

		}

		label.addEventListener( 'dblclick', () => {

			if ( set ) {

				if ( defaultValue === undefined ) return;
				input.value = toSlider( defaultValue );
				paint( defaultValue );
				set( defaultValue );

			} else this._action( 'reset-setting', item.key );

		} );
		return this._bind( {
			row, key: item.key, visible: item.visible,
			sync: () => {

				const v = this._pending.has( item.key ) ? this._pending.get( item.key ) : getV();
				if ( v === undefined || v === null ) return;
				if ( document.activeElement !== input || ! this._pending.size ) input.value = toSlider( v );
				paint( v );

			},
		} );

	}

	_segmented( item, { get, set } = {} ) {

		const n = item.options.length;
		const seg = h( 'div.wgv-seg', { role: 'radiogroup', 'aria-label': item.label || item.key, style: { '--n': n } } );
		const choose = ( value ) => ( set ? set( value ) : this._change( item.key, value ) );
		const buttons = item.options.map( ( opt, i ) => {

			const b = h( 'button', { role: 'radio', 'aria-checked': 'false', 'data-value': String( opt.value ), title: opt.hint || null }, opt.label );
			b.addEventListener( 'click', () => choose( opt.value ) );
			b.addEventListener( 'keydown', ( e ) => {

				const d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? - 1 : 0;
				if ( ! d ) return;
				e.preventDefault();
				let j = i;
				do j = ( j + d + n ) % n; while ( buttons[ j ].hidden && j !== i );
				choose( item.options[ j ].value );
				buttons[ j ].focus();

			} );
			seg.append( b );
			return b;

		} );
		const full = item.full || ! item.label;
		// a setting of the schema goes back to its default (and to its automatic choice) by a double click on its name
		const resettable = !! item.key && ! set;
		const name = full
			? ( item.caption ? h( 'span.wgv-caption', { title: resettable ? 'Double-click to reset' : null }, item.caption ) : null )
			: h( 'span.wgv-l', { title: [ item.hint, resettable ? 'Double-click to reset' : null ].filter( Boolean ).join( ' · ' ) || null }, item.label );
		if ( name && resettable ) name.addEventListener( 'dblclick', () => this._action( 'reset-setting', item.key ) );
		const row = full
			? h( 'div.wgv-row.wgv-segrow.is-full' + ( item.big ? '.is-big' : '' ) + ( item.caption ? '.has-caption' : '' ), null, name, seg )
			: h( 'div.wgv-row.wgv-segrow', null, name, seg );
		const getV = get || ( () => this.settings[ item.key ] );
		return this._bind( {
			row, key: item.key, visible: item.visible, live: item.live,
			sync: () => {

				const v = getV();
				// options whose feature is missing are not offered (the thumb counts the shown ones)
				let idx = - 1, first = - 1, shown = 0;
				buttons.forEach( ( b, i ) => {

					const opt = item.options[ i ];
					const hide = !! opt.needs && ! this.features[ opt.needs ];
					if ( b.hidden !== hide ) b.hidden = hide;
					const on = opt.value === v;
					b.setAttribute( 'aria-checked', String( on ) );
					if ( hide ) return;
					if ( first < 0 ) first = i;
					if ( on ) idx = shown;
					b._on = on;
					shown ++;

				} );
				// roving tabindex: the checked option, or the first when none matches ("custom")
				buttons.forEach( ( b, i ) => ( b.tabIndex = ! b.hidden && ( idx >= 0 ? b._on : i === first ) ? 0 : - 1 ) );
				seg.classList.toggle( 'is-none', idx < 0 );
				seg.classList.toggle( 'is-dense', shown >= 5 );
				seg.style.setProperty( '--n', Math.max( 1, shown ) );
				seg.style.setProperty( '--i', Math.max( 0, idx ) );

			},
		} );

	}

	_toggle( item, { get, set } = {} ) {

		const sw = h( 'i.wgv-switch', { 'aria-hidden': 'true' } );
		const row = h( 'button.wgv-row.wgv-toggle', { role: 'switch', 'aria-checked': 'false', title: item.hint || null },
			h( 'span.wgv-l', null, item.label ), sw );
		const getV = get || ( () => this.settings[ item.key ] );
		row.addEventListener( 'click', () => {

			const v = ! getV();
			set ? set( v ) : this._change( item.key, v );

		} );
		return this._bind( {
			row, key: item.key, visible: item.visible,
			sync: () => row.setAttribute( 'aria-checked', String( !! getV() ) ),
		} );

	}

	_palette( item ) {

		const wrap = h( 'div.wgv-swatches', { role: 'radiogroup', 'aria-label': item.label } );
		const buttons = item.options.map( ( p ) => {

			const stops = [];
			for ( let k = 0; k <= 6; k ++ ) stops.push( paletteCss( p, k / 6 ) );
			const b = h( 'button.wgv-swatch', { role: 'radio', title: paletteLabel( p ), 'aria-label': paletteLabel( p ), style: { '--g': `linear-gradient(90deg, ${stops.join( ', ' )})` } } );
			b.addEventListener( 'click', () => this._change( item.key, p ) );
			wrap.append( b );
			return b;

		} );
		const name = h( 'span.wgv-l', { title: 'Double-click to reset' }, item.label );
		name.addEventListener( 'dblclick', () => this._action( 'reset-setting', item.key ) );
		const row = h( 'div.wgv-row.wgv-palette', null, name, wrap );
		return this._bind( {
			row, key: item.key, visible: item.visible,
			sync: ( s ) => buttons.forEach( ( b, i ) => b.setAttribute( 'aria-checked', String( item.options[ i ] === s[ item.key ] ) ) ),
		} );

	}

	/** Icon tiles for a handful of exclusive modes, with one line that describes the current one. */
	_choice( item ) {

		const n = item.options.length;
		const tiles = h( 'div.wgv-tiles', { role: 'radiogroup', 'aria-label': item.label, style: { '--n': n } } );
		const desc = h( 'span.wgv-choice-t' );
		const spin = h( 'i.wgv-spinner.wgv-spinner-sm', { hidden: true } );
		const describe = ( value ) => {

			const d = item.describe ? item.describe( value, this.settings, this.info ) : null;
			const text = d ? d.text || '' : '';
			if ( desc.textContent !== text ) desc.textContent = text;
			spin.hidden = ! ( d && d.busy );

		};

		const buttons = item.options.map( ( opt, i ) => {

			const b = h( 'button.wgv-tile', { role: 'radio', 'aria-checked': 'false', 'data-value': String( opt.value ), title: LAYOUT_TEXT[ opt.value ] || null },
				h( 'span.wgv-tile-i', { html: icon( item.icons?.[ opt.value ] || 'spark', 18 ) } ), h( 'span.wgv-tile-l', null, item.labels?.[ opt.value ] || opt.label ) );
			b.addEventListener( 'click', () => this._change( item.key, opt.value ) );
			b.addEventListener( 'keydown', ( e ) => {

				const d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? - 1 : 0;
				if ( ! d ) return;
				e.preventDefault();
				const j = ( i + d + n ) % n;
				this._change( item.key, item.options[ j ].value );
				buttons[ j ].focus();

			} );
			tiles.append( b );
			return b;

		} );
		const row = h( 'div.wgv-row.wgv-choice', null, tiles, h( 'div.wgv-choice-d', { 'aria-live': 'polite' }, spin, desc ) );
		return this._bind( {
			row, key: item.key, visible: item.visible, live: true,
			sync: ( s ) => {

				buttons.forEach( ( b, i ) => {

					const on = item.options[ i ].value === s[ item.key ];
					b.setAttribute( 'aria-checked', String( on ) );
					b.tabIndex = on ? 0 : - 1;

				} );
				describe( s[ item.key ] );

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
					row: null, key: a.key,
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

	/** Structure preset: two layout parameters behind one clear choice. */
	_preset( item ) {

		const row = this._segmented( { label: 'Structure', caption: 'Structure', full: true, options: STRUCTURE_PRESETS, visible: item.visible, live: true }, {
			get: () => structurePreset( this.settings ),
			set: ( value ) => {

				const p = STRUCTURE_PRESETS.find( ( q ) => q.value === value );
				if ( ! p ) return;
				this._change( 'clusterSeparation', p.clusterSeparation );
				this._change( 'cohesion', p.cohesion );

			},
		} );
		return row;

	}

	/** Whole look: theme, palette, edge style and camera effects behind one choice. Nothing is lit when the settings match no look. */
	_look( item ) {

		const row = this._segmented( { label: 'Look', caption: 'Look', full: true, options: LOOKS, visible: item.visible, live: true }, {
			get: () => lookOf( this.settings ),
			set: ( value ) => this._action( 'look', value ),
		} );
		// what the look is, so that it is not mistaken for the theme below it (which it sets, among others)
		const text = h( 'span.wgv-choice-t' );
		row.append( h( 'div.wgv-choice-d', { 'aria-live': 'polite' }, text ) );
		const binding = this._bindings[ this._bindings.length - 1 ];
		const sync = binding.sync;
		binding.sync = ( s ) => {

			sync( s );
			const t = lookText( lookOf( this.settings ) );
			if ( text.textContent !== t ) text.textContent = t;

		};

		return row;

	}

	_note( item ) {

		const row = h( 'p.wgv-row.wgv-note' + ( item.mono ? '.is-mono' : '' ) );
		return this._bind( {
			row, live: true, visible: item.visible,
			sync: ( s ) => {

				let text = '';
				try {

					text = item.text( s, this.info ) || '';

				} catch ( _ ) { /* a note must never break the panel */ }

				if ( row.textContent !== text ) row.textContent = text;
				row.classList.toggle( 'is-blank', ! text );

			},
		} );

	}

	/** A live state line: spinner while busy, a sentence, optionally one button (cancel, go back). */
	_status( item ) {

		const text = h( 'span.wgv-status-text' );
		const spin = h( 'i.wgv-spinner.wgv-spinner-sm' );
		let action = null;
		const btn = h( 'button.wgv-btn.wgv-btn-sm', { onclick: () => action && this._action( action.name ) } );
		const row = h( 'div.wgv-row.wgv-statusrow', { role: 'status' }, spin, text, btn );
		return this._bind( {
			row, live: true, visible: () => !! this.info[ item.id ],
			sync: () => {

				const st = this.info[ item.id ];
				if ( ! st ) return;
				if ( text.textContent !== ( st.text || '' ) ) text.textContent = st.text || '';
				spin.hidden = ! st.busy;
				row.classList.toggle( 'is-warn', st.tone === 'warn' );
				action = st.action || null;
				btn.hidden = ! action;
				if ( action && btn.textContent !== action.label ) btn.textContent = action.label;

			},
		} );

	}

	_group( item ) {

		const id = `wgv-grp-${++ uid}`;
		const inner = h( 'div.wgv-group-inner' );
		const content = h( 'div.wgv-group-c', { id }, inner );
		const btn = h( 'button.wgv-group-h', { 'aria-expanded': 'false', 'aria-controls': id }, h( 'span.wgv-group-chev', { html: icon( 'chevron', 12 ) } ), h( 'span', null, item.title ) );
		const row = h( 'div.wgv-group.is-closed', null, btn, content );
		const setOpen = ( open, remember = false ) => {

			row.classList.toggle( 'is-closed', ! open );
			btn.setAttribute( 'aria-expanded', String( open ) );
			content.toggleAttribute( 'inert', ! open );
			if ( remember ) this._remember( 'grp:' + item.id, open );

		};

		btn.addEventListener( 'click', () => setOpen( row.classList.contains( 'is-closed' ), true ) );
		setOpen( this._recall( 'grp:' + item.id, false ) );
		const first = this._bindings.length;
		this._buildItems( item.items, inner, false );
		const kids = this._bindings.slice( first ).filter( ( b ) => b.row );
		// the group disappears when none of its controls is available
		return this._bind( { row, visible: () => ( ! item.visible || item.visible( this.settings, this.features, this.info ) ) && kids.some( ( b ) => ! b.visible || b.visible( this.settings, this.features, this.info ) ) } );

	}

	_dataset() {

		const select = h( 'select.wgv-select-el', { 'aria-label': 'Dataset' } );
		select.addEventListener( 'change', () => this._change( 'dataset', select.value ) );
		this._dsSelect = select;
		this._dsMeta = h( 'div.wgv-meta' );
		this._dsDesc = h( 'div.wgv-desc' );
		const fileInput = h( 'input', { type: 'file', accept: '.json,.txt,.csv', hidden: true, 'aria-hidden': 'true', tabindex: '-1' } );
		this._fileInput = fileInput;
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
		this._dropHint = h( 'span' );
		const row = h( 'div.wgv-dataset', null,
			h( 'div.wgv-select', null, select, h( 'span.wgv-select-chev', { html: icon( 'chevron', 14 ) } ) ),
			this._dsMeta, this._dsDesc,
			h( 'div.wgv-btns2', null,
				h( 'button.wgv-btn.wgv-btn-row', { html: `${icon( 'file' )}<span>Open file…</span>`, onclick: () => fileInput.click() } ),
				this._genBtn ),
			this._gen,
			h( 'div.wgv-drop-hint', null, h( 'span', { html: icon( 'drop', 14 ) } ), this._dropHint ),
			fileInput );
		this._fillDatasets();
		this._buildGenerator();
		this._syncFormats();
		return this._bind( { row, key: 'dataset', sync: ( s ) => this._syncDataset( s ) } );

	}

	/** Drop hint and the file dialog's filter follow the formats the data layer reads. */
	_syncFormats() {

		if ( ! this._dropHint ) return;
		const ext = listFormats( this.features.formats );
		const list = ext.length ? ext : [ '.json', '.txt', '.csv' ];
		this._fileInput.setAttribute( 'accept', list.join( ',' ) );
		const shown = list.slice( 0, 6 ).join( ' ' ) + ( list.length > 6 ? ' …' : '' );
		this._dropHint.textContent = `or drop a file anywhere: ${shown}`;
		this._dropHint.title = ( this.features.formats || [] ).map( ( f ) => `${f.name} (${( f.extensions || [] ).join( ', ' )})` ).join( '\n' );

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

		if ( ! select.options.length ) add( '_none', 'Datasets', '', m ? 'No built-in datasets' : 'Loading…' );

	}

	_datasetInfo( id ) {

		if ( this._customDataset && this._customDataset.id === id ) return this._customDataset;
		const entry = this.manifest?.datasets?.find( ( d ) => d.id === id ) || null;
		// counts of the graph on screen win over the manifest's (re-clustering changes the levels)
		return entry && this._liveDataset && this._liveDataset.id === id ? { ...entry, ...this._liveDataset, name: entry.name } : entry;

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
		// generator controls are private to the form: built with the same widgets, never bound to settings
		const local = ( build ) => {

			const n = this._bindings.length;
			build();
			return this._bindings.splice( n )[ 0 ];

		};

		for ( const p of model.params || [] ) {

			const get = () => state[ p.key ];
			const set = ( v ) => {

				state[ p.key ] = v;
				for ( const r of rows ) r.sync?.();

			};

			let b;
			if ( p.type === 'select' ) b = local( () => this._segmented( { label: p.label, options: p.options }, { get, set } ) );
			else if ( p.type === 'bool' ) b = local( () => this._toggle( { label: p.label }, { get, set } ) );
			else {

				const step = p.step ?? ( p.type === 'int' ? 1 : 0.01 );
				b = local( () => this._slider( { label: p.label, min: p.min, max: p.max, step, log: p.log, format: p.format }, { get, set, defaultValue: p.value } ) );

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

			const b = h( 'b', null, '–' );
			return [ h( 'div.wgv-stat', null, b, h( 'span', null, label ) ), b ];

		};

		const [ cN, nodes ] = cell( 'nodes' );
		const [ cE, edges ] = cell( 'edges' );
		const [ cC, communities ] = cell( 'communities' );
		const [ cL, levels ] = cell( 'levels' );
		this._stats = { nodes, edges, communities, levels };
		const c = this.options.credit;
		return h( 'footer.wgv-foot', null,
			h( 'div.wgv-stats', null, cN, cE, cC, cL ),
			h( 'div.wgv-foot-row', null,
				h( 'button.wgv-link', { onclick: () => this._action( 'reset-all' ), title: 'Back to the defaults; the graph stays' }, 'Reset all settings' ) ),
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
		this.root.append( lroot, this._toasts );

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

