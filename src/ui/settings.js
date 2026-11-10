// Settings model of the app shell: ONE schema that declares every setting, a store with the
// "touched by the user" rule and versioned persistence, and the compact URL codec of share links.
// Pure logic: no DOM, no three.js (its one import, explore/communities.js, is pure too); storage and URL parameters are passed in, so everything here
// runs under node:test (test/shell-settings.test.js).
//
// ─── Schema ─────────────────────────────────────────────────────────────────────────────────
//  SETTINGS: [ { key, type, default, section, label, code, save, share, … } ]
//    type      'number' { min, max, step, log? } | 'enum' { options: [ { value, label } ] } |
//              'bool' | 'string'
//    section   id of the panel section that shows it (SECTIONS), null = no control of its own
//    advanced  true = behind the section's "Advanced" disclosure
//    code      two-letter key of the share link
//    save      persisted in localStorage once the user has chosen a value (default true)
//    share     part of share links (default true)
//    auto      true = the app picks a value per graph while the user has not (autoChoices)
//    needs     name of a feature flag the control depends on (hidden while the flag is false)
//    visible   ( values, features ) -> bool, extra visibility rule
//    hint      one line for the control's tooltip
//  settingSpec( key ), settingDefaults(), isVisible( spec, values, features )
//
// ─── Store ──────────────────────────────────────────────────────────────────────────────────
//  const store = new SettingsStore( { storage, storageKey } )    storage: a Storage or null
//  store.values                   plain object with every key (the `settings` object of the app)
//  store.load()                   applies what was saved (migrated), returns the keys it set
//  store.set( key, value, { by } ) -> true when the value was accepted and changed
//      by 'user'  an explicit choice: marks the key touched and saves it
//         'auto'  a per-graph automatic choice: ignored when the key is touched, never saved
//         'link'  from a share link or URL parameter: touched (so automatic choices keep off)
//                 but not saved
//  store.beginLink( values )      starts the session from a share link: the shareable settings the
//                                 link names take its values, all the others go back to their
//                                 defaults (a link leaves out what was the default where it was
//                                 made, so what this browser has saved must not show through).
//                                 Nothing of it is saved; the saved choices stay in storage
//  store.auto( key, value )       = set( …, { by: 'auto' } )
//  store.isTouched( key ) / touched (Set)
//  store.reset( key? )            back to the default and untouched (all keys without argument);
//                                 returns the keys whose value changed
//  store.setDefault( key, value ) run-time default (a module that owns the default look reports
//                                 it); moves the value along while the key is untouched
//  store.ui( name, value? )       small free-form UI state saved next to the settings (which
//                                 sections are open, whether the first-run hint was seen)
//  store.shared()                 { key: value } for a share link: touched keys that differ from
//                                 the default, plus touched keys with an automatic rule
//  store.onChange( fn( key, value, by ) ) -> unsubscribe
//
// ─── Persistence ────────────────────────────────────────────────────────────────────────────
//  localStorage[ 'wgv:settings' ] = JSON { v: SETTINGS_VERSION, values: { key: value }, ui: {} }
//  Only touched keys with `save` are written; while a share link is open, the keys it decided keep
//  the value that was saved before. migrate( stored ) upgrades older shapes: version 1
//  was the flat settings object of the first viewer (charge, linkDistance, gravity, theta as d3
//  style multipliers); unknown keys and invalid values are dropped, never thrown on.
//
// ─── Share links ────────────────────────────────────────────────────────────────────────────
//  encodeShare( { dataset, settings, camera, explore } ) -> query string (no leading '?')
//      ds=<dataset id>                       built-in datasets only
//      s=<version>~<code>:<value>~…          settings, non-default values only; numbers with four
//                                            significant digits, booleans 1 / 0, enums literal
//      c=px,py,pz,tx,ty,tz[,fov]             camera pose (position, target, vertical fov in °)
//      x=<base64url( JSON )>                 explore.state() (opaque to the shell)
//  decodeShare( URLSearchParams | string ) -> { dataset, settings, camera, explore, version }
//      (null members where the link has none; unknown codes and invalid values are skipped)
//  parseLegacyParams( params, hash ) -> { dataset?, settings: {} }: the parameters that existed
//      before share links and keep working: ?ds=, #<dataset>, ?layout=<mode>, ?autorotate=0|1.
//      (?quality= and the graphics switches are parsed by the App itself.)
//
// ─── Per-graph automatic choices ────────────────────────────────────────────────────────────
//  autoChoices( graph, { tubes } ) -> { edgeBlend, edgeStyle, fieldStrength }: additive strands
//  for very large graphs, solid tubes for tiny ones (only where the tier draws tubes), no community
//  field where there are no communities. Edge blend and edge style have an explicit 'auto' value
//  that stands for this choice (so a person can always hand the setting back); the field strength
//  is a number and follows the `auto` rule of the store instead.
//  describeTier( config ) -> string[] of what a resolved quality configuration (app.gfx.config)
//  turns on, in plain words. describeClustering( graph ) -> one line about the hierarchy in use.
//  deferredStorage( storage, delay ) -> a storage that writes after a pause (with flush()), so
//  dragging a slider does not write on every frame.

import { communitySummary } from '../explore/communities.js';

export const SETTINGS_VERSION = 2;
export const STORAGE_KEY = 'wgv:settings';

/** Panel sections in display order. */
export const SECTIONS = [
	{ id: 'dataset', title: 'Dataset' },
	{ id: 'layout', title: 'Layout' },
	{ id: 'communities', title: 'Communities' },
	{ id: 'edges', title: 'Edges' },
	{ id: 'nodes', title: 'Nodes and labels' },
	{ id: 'scene', title: 'Scene' },
	{ id: 'camera', title: 'Camera and focus' },
	{ id: 'quality', title: 'Quality' },
];

const opts = ( ...pairs ) => pairs.map( ( p ) => ( Array.isArray( p ) ? { value: p[ 0 ], label: p[ 1 ] } : { value: p, label: p[ 0 ].toUpperCase() + p.slice( 1 ) } ) );
const times = ( v ) => `${Number( v ).toFixed( 2 )}×`;
const percent = ( v ) => `${Math.round( v * 100 )}%`;

export const SETTINGS = [
	// ── dataset ──
	{ key: 'dataset', type: 'string', default: 'imsv/lesmis', section: 'dataset', label: 'Dataset', code: 'ds', save: false },

	// ── layout ──
	{ key: 'layoutMode', type: 'enum', default: 'force', section: 'layout', label: 'Layout', code: 'ly', save: false,
		options: opts( [ 'force', 'Force 3D' ], [ 'flat', 'Force 2D' ], 'sphere', 'radial', 'stress' ) },
	{ key: 'running', type: 'bool', default: true, section: 'layout', label: 'Layout running', code: 'ru', save: false, share: false },
	{ key: 'immersive', type: 'bool', default: false, section: 'layout', label: 'Immersive view', code: 'im', save: false,
		hint: 'Look around from the centre of the sphere', visible: ( s ) => s.layoutMode === 'sphere' },
	{ key: 'clusterSeparation', type: 'number', default: 0.7, min: 0.1, max: 1, step: 0.01, section: 'layout', advanced: true, label: 'Separation', code: 'cs',
		hint: 'Lower values pull links between communities longer, so communities stand apart' },
	{ key: 'cohesion', type: 'number', default: 0.5, min: 0, max: 3, step: 0.05, section: 'layout', advanced: true, label: 'Cohesion', code: 'co',
		hint: 'How strongly the members of a community are held together' },
	{ key: 'hubSpacing', type: 'number', default: 1, min: 0, max: 2, step: 0.05, section: 'layout', advanced: true, label: 'Hub spacing', code: 'hs',
		hint: 'Room that high-degree nodes claim around themselves' },
	{ key: 'layoutSize', type: 'number', default: 1, min: 0.4, max: 3, step: 0.05, section: 'layout', advanced: true, label: 'Size', code: 'sz', format: times,
		hint: 'Overall size of the layout' },
	{ key: 'repulsion', type: 'number', default: 8, min: 1, max: 64, step: 0.5, log: true, section: 'layout', advanced: true, label: 'Repulsion', code: 'rp',
		hint: 'Short-range repulsion between nodes' },
	{ key: 'damping', type: 'number', default: 0.3, min: 0.05, max: 0.95, step: 0.01, section: 'layout', advanced: true, label: 'Damping', code: 'dm',
		hint: 'Friction of the simulation' },

	// ── communities ──
	{ key: 'clusterMethod', type: 'enum', default: 'default', section: 'communities', label: 'Method', code: 'cm', save: false, needs: 'recluster',
		options: opts( [ 'default', 'Original' ], 'louvain', 'leiden' ), hint: 'Original: the communities the graph came with' },
	{ key: 'clusterResolution', type: 'number', default: 1, min: 0.25, max: 4, step: 0.05, log: true, section: 'communities', label: 'Resolution', code: 'cr', save: false,
		needs: 'recluster', hint: 'Larger values give more, smaller communities', visible: ( s ) => s.clusterMethod !== 'default' },
	{ key: 'fieldStrength', type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, section: 'communities', label: 'Halo', code: 'fs', auto: true, needs: 'field',
		format: percent, hint: 'Soft glow volume around each community' },
	{ key: 'clusterLabels', type: 'enum', default: 'auto', section: 'communities', label: 'Names', code: 'cl', needs: 'clusterLabels',
		options: opts( 'auto', 'on', 'off' ), hint: 'Community names at overview distance' },

	// ── edges ──
	{ key: 'bundleMode', type: 'enum', default: 'hierarchy', section: 'edges', label: 'Bundling', code: 'bm', save: false,
		options: opts( 'hierarchy', [ 'paths', 'Edge paths' ], 'straight' ).map( ( o ) => ( o.value === 'paths' ? { ...o, needs: 'bundlePaths' } : o ) ),
		hint: 'Hierarchy: through the community centres. Edge paths: along shortest paths of the layout' },
	{ key: 'bundling', type: 'number', default: 0.85, min: 0, max: 1, step: 0.01, section: 'edges', label: 'Strength', code: 'bn',
		visible: ( s ) => s.bundleMode !== 'straight' },
	{ key: 'edgeStyle', type: 'enum', default: 'auto', section: 'edges', label: 'Style', code: 'es', needs: 'edgeStyle',
		options: opts( 'auto', [ 'luminous', 'Strands' ], [ 'solid', 'Tubes' ] ), hint: 'Luminous strands, or solid lit tubes. Auto: tubes on small graphs' },
	{ key: 'edgeFlow', type: 'enum', default: 'selection', section: 'edges', label: 'Flow', code: 'ef', needs: 'edgeFlow',
		options: opts( 'off', [ 'selection', 'Selected' ], 'all' ), hint: 'Pulses that travel from source to target: on the edges of the selection, or on all' },
	{ key: 'edgeOpacity', type: 'number', default: 1, min: 0, max: 1, step: 0.01, section: 'edges', advanced: true, label: 'Opacity', code: 'eo' },
	{ key: 'edgeWidth', type: 'number', default: 1, min: 0.25, max: 4, step: 0.05, section: 'edges', advanced: true, label: 'Width', code: 'ew', format: times },
	{ key: 'edgeBlend', type: 'enum', default: 'auto', section: 'edges', advanced: true, label: 'Blend', code: 'eb',
		options: opts( 'auto', 'normal', 'additive' ), hint: 'Additive strands glow where they are dense. Auto: additive on very large graphs only' },

	// ── nodes and labels ──
	{ key: 'nodeSize', type: 'number', default: 1, min: 0.25, max: 4, step: 0.05, section: 'nodes', label: 'Size', code: 'ns', format: times },
	{ key: 'labels', type: 'enum', default: 'auto', section: 'nodes', label: 'Labels', code: 'lb', options: opts( 'auto', 'all', 'off' ) },
	{ key: 'nodeGloss', type: 'number', default: 1, min: 0, max: 2, step: 0.02, section: 'nodes', label: 'Gloss', code: 'ng', format: times, needs: 'nodeLook',
		hint: 'Matte to mirror-like; 1 is the look of the theme' },
	{ key: 'nodeGlow', type: 'number', default: 1, min: 0, max: 2, step: 0.02, section: 'nodes', label: 'Glow', code: 'nw', format: times, needs: 'nodeLook',
		hint: 'Emissive core of the nodes; 1 is the look of the theme' },
	{ key: 'palette', type: 'enum', default: 'paper', section: 'nodes', label: 'Palette', code: 'pl',
		// 'paper' is the palette of the paper; it is called Spectrum, as a look and a theme carry names too
		options: opts( [ 'paper', 'Spectrum' ], 'rainbow', 'aurora', 'ember' ) },

	// ── scene ──
	{ key: 'theme', type: 'enum', default: 'aurora', section: 'scene', label: 'Theme', code: 'th', needs: 'env', options: opts( 'aurora', 'studio', 'void' ) },
	{ key: 'keyLight', type: 'number', default: 1, min: 0, max: 2, step: 0.02, section: 'scene', label: 'Key light', code: 'kl', format: times, needs: 'env' },
	{ key: 'envLight', type: 'number', default: 1, min: 0, max: 2, step: 0.02, section: 'scene', label: 'Environment', code: 'el', format: times, needs: 'env' },
	{ key: 'fog', type: 'number', default: 1, min: 0, max: 2, step: 0.02, section: 'scene', label: 'Fog', code: 'fg', format: times, needs: 'env' },
	{ key: 'dust', type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, section: 'scene', label: 'Depth motes', code: 'du', format: percent, needs: 'dust' },

	// ── camera and focus ──
	{ key: 'autoRotate', type: 'bool', default: true, section: 'camera', label: 'Auto-rotate', code: 'ar', hint: 'A slow turn while idle; after a minute it becomes a drift with a focus pull' },
	{ key: 'dof', type: 'enum', default: 'auto', section: 'camera', label: 'Depth of field', code: 'df', options: opts( 'auto', 'off', 'on' ),
		hint: 'Auto follows the quality tier' },
	{ key: 'dofStrength', type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, section: 'camera', label: 'Strength', code: 'dq', format: percent,
		visible: ( s ) => s.dof === 'on' },
	{ key: 'bloom', type: 'number', default: 0.8, min: 0, max: 2, step: 0.01, section: 'camera', label: 'Bloom', code: 'bl' },
	{ key: 'exposure', type: 'number', default: 1, min: 0.25, max: 4, step: 0.01, log: true, section: 'camera', label: 'Exposure', code: 'ex',
		format: ( v ) => {

			const ev = Math.log2( v );
			return `${ev > 0.05 ? '+' : ev < - 0.05 ? '−' : ''}${Math.abs( ev ).toFixed( 1 )} EV`;

		} },
	{ key: 'aoStrength', type: 'number', default: 1, min: 0, max: 1, step: 0.01, section: 'camera', advanced: true, label: 'Occlusion', code: 'ao', format: percent,
		hint: 'Ambient occlusion where nodes crowd (medium tier and up)' },
	{ key: 'grain', type: 'bool', default: true, section: 'camera', advanced: true, label: 'Film grain', code: 'gr', hint: 'Where the tier has it' },
	{ key: 'vignette', type: 'bool', default: true, section: 'camera', advanced: true, label: 'Vignette', code: 'vg' },

	// ── quality ──
	// quality describes the device, not the view: it is saved but never travels in a share link
	{ key: 'quality', type: 'enum', default: 'auto', section: 'quality', label: 'Quality', code: 'q', share: false, options: opts( 'auto', 'low', 'medium', 'high', 'ultra' ) },
	{ key: 'dynamicResolution', type: 'bool', default: true, section: 'quality', label: 'Dynamic resolution', code: 'dr', share: false,
		hint: 'Lowers the render resolution when frames get slow' },

	{ key: 'showStatus', type: 'bool', default: true, section: 'quality', label: 'Status chips', code: 'st', share: false,
		hint: 'Small chips with the frame rate, the quality tier, the layout state and the counts (H)' },
];

const BY_KEY = new Map( SETTINGS.map( ( s ) => [ s.key, s ] ) );
const BY_CODE = new Map( SETTINGS.map( ( s ) => [ s.code, s ] ) );

export function settingSpec( key ) {

	return BY_KEY.get( key ) || null;

}

export function settingDefaults() {

	return Object.fromEntries( SETTINGS.map( ( s ) => [ s.key, s.default ] ) );

}

/** Whether a setting's control is shown: its feature flag, then its own rule. */
export function isVisible( spec, values, features = {} ) {

	if ( typeof spec === 'string' ) spec = BY_KEY.get( spec );
	if ( ! spec ) return false;
	if ( spec.needs && ! features[ spec.needs ] ) return false;
	return spec.visible ? !! spec.visible( values, features ) : true;

}

/** A valid value of the setting's type and range, or undefined. */
export function coerce( spec, value ) {

	if ( typeof spec === 'string' ) spec = BY_KEY.get( spec );
	if ( ! spec || value === undefined || value === null ) return undefined;
	switch ( spec.type ) {

		case 'number': {

			const v = typeof value === 'string' && value.trim() === '' ? NaN : Number( value );
			if ( ! Number.isFinite( v ) ) return undefined;
			return Math.min( spec.max ?? Infinity, Math.max( spec.min ?? - Infinity, v ) );

		}

		case 'bool':
			if ( typeof value === 'boolean' ) return value;
			if ( value === 1 || value === '1' || value === 'true' ) return true;
			if ( value === 0 || value === '0' || value === 'false' ) return false;
			return undefined;
		case 'enum': {

			const v = String( value );
			return spec.options.some( ( o ) => o.value === v ) ? v : undefined;

		}

		case 'string': return typeof value === 'string' && value.length > 0 && value.length < 512 ? value : undefined;

	}

	return undefined;

}

const same = ( a, b ) => a === b || ( typeof a === 'number' && typeof b === 'number' && Math.abs( a - b ) < 1e-9 );

// ── persistence ────────────────────────────────────────────────────────────────────────────

/**
 * Brings a stored blob of any earlier version to { v: SETTINGS_VERSION, values, ui }. Never
 * throws: what cannot be understood is dropped.
 */
export function migrate( stored ) {

	const empty = { v: SETTINGS_VERSION, values: {}, ui: {} };
	if ( ! stored || typeof stored !== 'object' || Array.isArray( stored ) ) return empty;
	let version = Number.isInteger( stored.v ) ? stored.v : 1;
	let values, ui;
	if ( version <= 1 ) {

		// version 1: the flat settings object of the first viewer, without a version field
		const old = stored.values && typeof stored.values === 'object' ? stored.values : stored;
		values = { ...old };
		// d3-style multipliers became parameters of the hierarchy-aware layout
		if ( Number.isFinite( old.linkDistance ) ) values.layoutSize = old.linkDistance / 30;
		if ( Number.isFinite( old.charge ) ) values.repulsion = 8 * Math.max( 0.125, old.charge );
		for ( const key of [ 'charge', 'linkDistance', 'gravity', 'theta', 'v', 'ui' ] ) delete values[ key ];
		ui = stored.ui && typeof stored.ui === 'object' ? { ...stored.ui } : {};
		version = 2;

	} else {

		values = stored.values && typeof stored.values === 'object' ? { ...stored.values } : {};
		ui = stored.ui && typeof stored.ui === 'object' ? { ...stored.ui } : {};

	}

	if ( version > SETTINGS_VERSION ) return empty; // written by a newer build: start clean
	const out = {};
	for ( const [ key, value ] of Object.entries( values ) ) {

		const spec = BY_KEY.get( key );
		if ( ! spec || spec.save === false ) continue;
		const v = coerce( spec, value );
		if ( v !== undefined ) out[ key ] = v;

	}

	return { v: SETTINGS_VERSION, values: out, ui };

}

export class SettingsStore {

	constructor( { storage = null, storageKey = STORAGE_KEY } = {} ) {

		this.storage = storage;
		this.storageKey = storageKey;
		this.defaults = settingDefaults();
		this.values = { ...this.defaults };
		this.touched = new Set();
		this._ui = {};
		this._listeners = new Set();

	}

	spec( key ) {

		return BY_KEY.get( key ) || null;

	}

	get( key ) {

		return this.values[ key ];

	}

	isTouched( key ) {

		return this.touched.has( key );

	}

	onChange( fn ) {

		this._listeners.add( fn );
		return () => this._listeners.delete( fn );

	}

	/** Reads the saved state; saved values count as explicit choices. Returns the keys applied. */
	load() {

		let raw = null;
		try {

			raw = this.storage ? JSON.parse( this.storage.getItem( this.storageKey ) || 'null' ) : null;

		} catch ( _ ) {

			raw = null; // unreadable or blocked storage: defaults

		}

		const { values, ui } = migrate( raw );
		this._ui = ui;
		this._saved = { ...values };
		const applied = [];
		for ( const [ key, value ] of Object.entries( values ) ) {

			this.values[ key ] = value;
			this.touched.add( key );
			applied.push( key );

		}

		return applied;

	}

	save() {

		if ( ! this.storage ) return false;
		const values = {}, before = this._saved || {};
		for ( const spec of SETTINGS ) {

			const key = spec.key;
			if ( spec.save === false ) continue;
			if ( this._session?.has( key ) ) {

				// decided by a link (its value, or the default it stands for): not a preference, so
				// what was saved before stays
				if ( key in before ) values[ key ] = before[ key ];

			} else if ( this.touched.has( key ) ) values[ key ] = this.values[ key ];

		}

		this._saved = values;

		try {

			this.storage.setItem( this.storageKey, JSON.stringify( { v: SETTINGS_VERSION, values, ui: this._ui } ) );
			return true;

		} catch ( _ ) {

			return false; // private mode / quota: the session still works

		}

	}

	set( key, value, { by = 'user' } = {} ) {

		const spec = BY_KEY.get( key );
		if ( ! spec ) return false;
		const v = coerce( spec, value );
		if ( v === undefined ) return false;
		if ( by === 'auto' ) {

			if ( this.touched.has( key ) ) return false;

		} else {

			this.touched.add( key );
			// values that came with a link are explicit for this session but are not written to storage
			if ( by === 'link' ) ( this._session ??= new Set() ).add( key );
			else this._session?.delete( key );

		}

		const changed = ! same( this.values[ key ], v );
		this.values[ key ] = v;
		if ( by === 'user' ) this.save();
		if ( changed ) for ( const fn of this._listeners ) fn( key, v, by );
		return changed;

	}

	auto( key, value ) {

		return this.set( key, value, { by: 'auto' } );

	}

	/**
	 * Starts the session from a share link. A link carries only what differed from the defaults
	 * where it was made, so every shareable setting it leaves out stands for its default (and for
	 * the automatic per-graph choice), whatever this browser has saved: the named keys take the
	 * link's values, the others go back to the default and count as untouched. None of this is a
	 * preference: storage keeps what was saved, and a page opened without the link shows it again.
	 * Returns the keys whose value changed.
	 */
	beginLink( values = {} ) {

		const session = ( this._session ??= new Set() );
		const changed = [];
		for ( const spec of SETTINGS ) {

			const key = spec.key;
			if ( spec.share === false || key === 'dataset' ) continue;
			const before = this.values[ key ];
			const v = key in values ? coerce( spec, values[ key ] ) : undefined;
			if ( v !== undefined ) {

				this.values[ key ] = v;
				this.touched.add( key );

			} else {

				this.values[ key ] = this.defaults[ key ];
				this.touched.delete( key );

			}

			session.add( key );
			if ( ! same( before, this.values[ key ] ) ) changed.push( key );

		}

		for ( const key of changed ) for ( const fn of this._listeners ) fn( key, this.values[ key ], 'link' );
		return changed;

	}

	/** Run-time default (reported by the module that owns the look). Untouched values follow it. */
	setDefault( key, value ) {

		const spec = BY_KEY.get( key );
		const v = coerce( spec, value );
		if ( v === undefined ) return false;
		this.defaults[ key ] = v;
		if ( ! this.touched.has( key ) ) this.values[ key ] = v;
		return true;

	}

	/** Back to the defaults and untouched: one key, or everything. Returns the changed keys. */
	reset( key = null ) {

		const keys = key === null ? SETTINGS.map( ( s ) => s.key ) : [ key ];
		const changed = [];
		for ( const k of keys ) {

			if ( ! BY_KEY.has( k ) ) continue;
			// the dataset is not a preference: "Reset all" keeps the graph on screen
			if ( key === null && k === 'dataset' ) continue;
			this.touched.delete( k );
			this._session?.delete( k );
			if ( ! same( this.values[ k ], this.defaults[ k ] ) ) {

				this.values[ k ] = this.defaults[ k ];
				changed.push( k );

			}

		}

		this.save();
		for ( const k of changed ) for ( const fn of this._listeners ) fn( k, this.values[ k ], 'reset' );
		return changed;

	}

	ui( name, value ) {

		if ( value === undefined ) return this._ui[ name ];
		this._ui[ name ] = value;
		this.save();
		return value;

	}

	/** What a share link carries: explicit choices that differ from the default or that an automatic rule would change. */
	shared() {

		const out = {};
		for ( const spec of SETTINGS ) {

			if ( spec.share === false || spec.key === 'dataset' || ! this.touched.has( spec.key ) ) continue;
			const v = this.values[ spec.key ];
			if ( spec.auto || ! same( v, this.defaults[ spec.key ] ) ) out[ spec.key ] = v;

		}

		return out;

	}

}

// ── share links ────────────────────────────────────────────────────────────────────────────

const num = ( v, digits = 4 ) => String( + Number( v ).toPrecision( digits ) );

function toBase64Url( text ) {

	const bytes = new TextEncoder().encode( text );
	let bin = '';
	for ( let i = 0; i < bytes.length; i ++ ) bin += String.fromCharCode( bytes[ i ] );
	return btoa( bin ).replace( /\+/g, '-' ).replace( /\//g, '_' ).replace( /=+$/, '' );

}

function fromBase64Url( text ) {

	const bin = atob( text.replace( /-/g, '+' ).replace( /_/g, '/' ) );
	const bytes = new Uint8Array( bin.length );
	for ( let i = 0; i < bin.length; i ++ ) bytes[ i ] = bin.charCodeAt( i );
	return new TextDecoder().decode( bytes );

}

export function encodePose( pose ) {

	if ( ! pose || ! pose.position || ! pose.target ) return '';
	const parts = [ ...pose.position, ...pose.target ].map( ( v ) => num( v ) );
	if ( parts.length !== 6 || parts.some( ( p ) => p === 'NaN' ) ) return '';
	if ( Number.isFinite( pose.fov ) && Math.abs( pose.fov - 45 ) > 0.05 ) parts.push( num( pose.fov, 3 ) );
	return parts.join( ',' );

}

export function decodePose( text ) {

	if ( ! text ) return null;
	const v = String( text ).split( ',' ).map( Number );
	if ( ( v.length !== 6 && v.length !== 7 ) || ! v.every( Number.isFinite ) ) return null;
	const fov = v.length === 7 ? Math.min( 120, Math.max( 10, v[ 6 ] ) ) : 45;
	return { position: v.slice( 0, 3 ), target: v.slice( 3, 6 ), fov };

}

export function encodeSettings( settings ) {

	const parts = [ String( SETTINGS_VERSION ) ];
	for ( const spec of SETTINGS ) {

		if ( ! ( spec.key in settings ) || spec.share === false || spec.key === 'dataset' ) continue;
		const v = coerce( spec, settings[ spec.key ] );
		if ( v === undefined ) continue;
		parts.push( `${spec.code}:${spec.type === 'number' ? num( v ) : spec.type === 'bool' ? ( v ? 1 : 0 ) : v}` );

	}

	return parts.length > 1 ? parts.join( '~' ) : '';

}

export function decodeSettings( text ) {

	const out = { version: null, settings: {} };
	if ( ! text ) return out;
	const parts = String( text ).split( '~' );
	const version = Number( parts[ 0 ] );
	if ( ! Number.isInteger( version ) || version < 1 || version > SETTINGS_VERSION ) return out; // a newer or broken link: ignore the settings
	out.version = version;
	for ( const part of parts.slice( 1 ) ) {

		const at = part.indexOf( ':' );
		if ( at < 1 ) continue;
		const spec = BY_CODE.get( part.slice( 0, at ) );
		if ( ! spec || spec.share === false ) continue;
		const v = coerce( spec, part.slice( at + 1 ) );
		if ( v !== undefined ) out.settings[ spec.key ] = v;

	}

	return out;

}

/** Query string (without '?') of a share link. Empty members are left out. */
export function encodeShare( { dataset = null, settings = null, camera = null, explore = null } = {} ) {

	const parts = [];
	// dataset ids are path-like (imsv/lesmis): keep the slash readable
	if ( dataset ) parts.push( 'ds=' + encodeURIComponent( dataset ).replace( /%2F/gi, '/' ) );
	const s = settings ? encodeSettings( settings ) : '';
	if ( s ) parts.push( 's=' + s );
	const c = encodePose( camera );
	if ( c ) parts.push( 'c=' + c );
	if ( explore !== null && explore !== undefined ) {

		let json = '';
		try {

			json = JSON.stringify( explore );

		} catch ( _ ) { /* not serialisable: leave it out */ }

		if ( json && json !== '{}' && json !== 'null' ) parts.push( 'x=' + toBase64Url( json ) );

	}

	return parts.join( '&' );

}

export function decodeShare( query ) {

	const params = typeof query === 'string' ? new URLSearchParams( query.replace( /^[?#]/, '' ) ) : query;
	const out = { dataset: null, settings: {}, camera: null, explore: null, version: null };
	if ( ! params ) return out;
	const ds = params.get( 'ds' );
	if ( ds ) out.dataset = ds;
	const s = decodeSettings( params.get( 's' ) );
	out.settings = s.settings;
	out.version = s.version;
	out.camera = decodePose( params.get( 'c' ) );
	const x = params.get( 'x' );
	if ( x ) {

		try {

			out.explore = JSON.parse( fromBase64Url( x ) );

		} catch ( _ ) {

			out.explore = null;

		}

	}

	return out;

}

/** True when the query carries share-link members (settings, camera or explore state). */
export function hasShare( params ) {

	return !! params && ( params.has( 's' ) || params.has( 'c' ) || params.has( 'x' ) );

}

/** The URL parameters of the first viewer, which keep working. */
export function parseLegacyParams( params, hash = '' ) {

	const out = { settings: {} };
	const ds = params && params.get( 'ds' );
	if ( ds ) out.dataset = ds;
	else if ( hash && hash.length > 1 ) {

		try {

			out.dataset = decodeURIComponent( hash.replace( /^#/, '' ) );

		} catch ( _ ) { /* malformed escape: ignore the hash */ }

	}

	if ( ! params ) return out;
	const layout = coerce( 'layoutMode', params.get( 'layout' ) );
	if ( layout !== undefined ) out.settings.layoutMode = layout;
	const rotate = coerce( 'autoRotate', params.get( 'autorotate' ) );
	if ( rotate !== undefined ) out.settings.autoRotate = rotate;
	return out;

}

// ── per-graph automatic choices, tier description ──────────────────────────────────────────

export const AUTO = {
	// above this edge count strands are drawn additive (glow). Additive strands add up without a
	// bound and feed the bloom: at tens of thousands of edges they wash the picture out, so the
	// automatic choice keeps them for graphs whose single strands are nearly invisible
	additiveEdges: 150000,
	solidEdges: 120, // up to this edge count every edge may be a solid tube (style 'auto')
	fieldMinClusters: 3, // fewer communities than this: no community field
};

export function autoChoices( graph, { tubes = false, fieldDefault = 0.5 } = {} ) {

	const E = graph ? graph.E | 0 : 0;
	const clusters = communitySummary( graph ).count;
	return {
		edgeBlend: E > AUTO.additiveEdges ? 'additive' : 'normal',
		edgeStyle: tubes && E > 0 && E <= AUTO.solidEdges ? 'solid' : 'luminous',
		fieldStrength: clusters >= AUTO.fieldMinClusters ? fieldDefault : 0,
	};

}

// ── whole looks ────────────────────────────────────────────────────────────────────────────

/** The settings a look decides. Every look covers all of them, so choosing one after another leaves nothing behind. */
export const LOOK_KEYS = [ 'theme', 'palette', 'edgeStyle', 'edgeBlend', 'bloom', 'nodeGloss', 'nodeGlow', 'fog', 'dof', 'grain', 'vignette' ];

/**
 * Looks: one choice that sets the theme, the palette, the edge style and the camera effects together.
 * `values` names what differs from the defaults; a key a look does not name goes back to its
 * default, which for the edge blend and the edge style is 'auto', the per-graph choice.
 */
export const LOOKS = [
	{ value: 'aurora', label: 'Aurora', hint: 'The lit default: soft sky, glossy nodes', values: {} },
	{ value: 'paper', label: 'Classic', hint: 'As in the paper: black ground, additive glow',
		values: { theme: 'void', edgeStyle: 'luminous', edgeBlend: 'additive', bloom: 1.2, nodeGloss: 0.6, nodeGlow: 1.5, dof: 'off', vignette: false } },
	{ value: 'blueprint', label: 'Blueprint', hint: 'A technical drawing: cool, crisp, little glow',
		values: { theme: 'studio', palette: 'aurora', edgeBlend: 'normal', bloom: 0.25, nodeGloss: 1.3, nodeGlow: 0.5, fog: 0.4, dof: 'off', grain: false } },
];

/** { key: value | undefined } over LOOK_KEYS for a look id (undefined: back to the default / automatic), or null. */
export function lookValues( id ) {

	const look = LOOKS.find( ( l ) => l.value === id );
	if ( ! look ) return null;
	const out = {};
	for ( const key of LOOK_KEYS ) out[ key ] = key in look.values ? look.values[ key ] : undefined;
	return out;

}

/** The look the settings match, or 'custom'. A setting whose value the app picks per graph (schema `auto`) only counts where the look names it. */
export function lookOf( settings ) {

	for ( const look of LOOKS ) {

		let match = true;
		for ( const key of LOOK_KEYS ) {

			const spec = BY_KEY.get( key ), named = key in look.values;
			if ( ! named && spec.auto ) continue;
			const want = named ? look.values[ key ] : spec.default, have = settings[ key ];
			if ( spec.type === 'number' ? ! ( Math.abs( have - want ) < 1e-6 ) : have !== want ) {

				match = false;
				break;

			}

		}

		if ( match ) return look.value;

	}

	return 'custom';

}

/** What a resolved quality configuration (app.gfx.config) turns on, in plain words. */
export function describeTier( config ) {

	if ( ! config ) return [];
	const out = [];
	const aa = { traa: 'temporal anti-aliasing', msaa: '4× multisampling', smaa: 'SMAA anti-aliasing', fxaa: 'FXAA anti-aliasing', none: 'no anti-aliasing' }[ config.aa ];
	if ( aa ) out.push( aa );
	out.push( config.lit ? ( config.clearcoat ? 'lit nodes with clear coat' : 'lit nodes' ) : 'flat-shaded nodes' );
	if ( config.shadows ) out.push( `soft shadows (${config.shadowMapSize})` );
	if ( config.ao ) out.push( config.aoScale >= 1 ? 'full-resolution ambient occlusion' : 'ambient occlusion' );
	if ( config.dof ) out.push( 'depth of field' );
	out.push( config.streak ? 'bloom with streaks' : 'bloom' );
	if ( config.chromaticAberration ) out.push( 'lens fringing' );
	if ( config.sharpen ) out.push( 'sharpening' );
	if ( config.grain ) out.push( 'film grain' );
	if ( config.maxPixels ) out.push( `up to ${+ ( config.maxPixels / 1e6 ).toFixed( 1 )} MP` );
	return out;

}

/** One line about the hierarchy in use (graph.clustering of the data layer, else what the graph itself tells). */
export function describeClustering( graph ) {

	if ( ! graph ) return '';
	const c = graph.clustering || null;
	const h = graph.meta && graph.meta.hierarchy ? graph.meta.hierarchy : null;
	const names = { louvain: 'Louvain', leiden: 'Leiden', planted: 'Planted communities', file: 'From the file' };
	// files carry free text here ("Louvain (python-louvain 0.8, …)"): the first word names the method
	const method = String( ( c && c.method ) || ( h && h.method ) || '' ).split( /[\s(,;]/ )[ 0 ];
	const parts = [];
	if ( method ) parts.push( names[ method.toLowerCase() ] || method[ 0 ].toUpperCase() + method.slice( 1 ) );
	// the same number as the legend and the insights drawer (communitySummary), not the sum of all levels
	const { count, finest } = communitySummary( graph );
	if ( count ) parts.push( `${count.toLocaleString( 'en-US' )} ${count === 1 ? 'community' : 'communities'}${finest > count ? ` (${finest.toLocaleString( 'en-US' )} at the finest level)` : ''}` );
	const levels = ( c && c.levels ) || graph.D || 0;
	const n = Array.isArray( levels ) ? levels.length : levels | 0;
	if ( n > 1 ) parts.push( `${n} levels` );
	// the data layer moves the resolution on a grid: say the one that was used
	const res = c ? ( Number.isFinite( c.resolutionUsed ) ? c.resolutionUsed : c.resolution ) : NaN;
	if ( Number.isFinite( res ) && Math.abs( res - 1 ) > 1e-6 ) parts.push( `resolution ${+ res.toFixed( 2 )}` );
	const q = c && Number.isFinite( c.modularity ) ? c.modularity : null;
	if ( q !== null ) parts.push( `modularity ${q.toFixed( 2 )}` );
	return parts.join( ' · ' );

}

/**
 * A storage that writes late: setItem keeps the newest value and writes it after `delay` ms of
 * quiet (a slider drag is one write, not sixty). flush() writes at once (call it on pagehide).
 */
export function deferredStorage( storage, delay = 400 ) {

	if ( ! storage ) return null;
	const pending = new Map();
	let timer = null;
	const flush = () => {

		if ( timer !== null ) clearTimeout( timer );
		timer = null;
		for ( const [ k, v ] of pending ) {

			try {

				storage.setItem( k, v );

			} catch ( _ ) { /* quota / private mode: the session still works */ }

		}

		pending.clear();

	};

	return {
		getItem: ( k ) => ( pending.has( k ) ? pending.get( k ) : storage.getItem( k ) ),
		setItem( k, v ) {

			pending.set( k, String( v ) );
			if ( timer !== null ) clearTimeout( timer );
			timer = setTimeout( flush, delay );

		},
		removeItem( k ) {

			pending.delete( k );
			storage.removeItem( k );

		},
		flush,
	};

}
