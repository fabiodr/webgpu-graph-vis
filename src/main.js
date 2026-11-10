// Application entry: composes the modules and wires every setting to them.
//
//   ForceLayout (GPU compute, order 0) → EdgeRenderer (10) → NodeRenderer (11) → Environment (12)
//   → ClusterField (13) → TextLabels (20) → Lens, Tools (25) → Interaction (30) → CameraRig (31)
//   → NodeCard (41) → Hud (42)
//
// Rendering (scene pass, quality tiers, post-processing, label overlay, captures) belongs to
// app.gfx, the render graph the App creates. Data is loaded, clustered and prepared in a Web
// Worker (DataClient); graph metrics are computed in a second worker and mapped onto the picture
// by the Lens. The explore layer (src/explore) adds the surfaces for reading the graph; this file
// and src/ui are the shell around it: settings, camera, photo mode, status, help, share links.
//
// Settings: one schema (src/ui/settings.js). A SettingsStore holds the values (`settings` is its
// value object, shared with every module that reads settings), remembers which ones the user
// chose (those are saved, and automatic per-graph choices never override them) and encodes share
// links. The store is the only writer of that object: the panel, set( key, value ), the automatic
// rules, a reset and the values that follow a module (follow) all go through it, so its listeners
// hear every change. Every change ends in apply( key, value ), the one place that talks to the
// modules. Modules that other
// packages are still building are feature-detected (features()); their controls stay hidden
// until the module is there.
//
// URL: ?ds=<dataset id> or #<dataset id>, ?layout=<mode>, ?autorotate=0|1, ?hint=0|1 (never /
// always show the first-run hint), a share link's members (?s= settings, ?c= camera pose, ?x=
// explore state; see settings.js: a share link describes the whole view, so what it leaves out
// is the default for this visit, whatever is saved here), and the graphics switches read by the App (parseGfxParams in
// app/Quality.js): ?quality=auto|low|medium|high|ultra, ?governor=0|1, ?ao=0|1, ?dof=0|1,
// ?shadows=0|1, ?traa=0|1, ?scale=<number>, ?debug=ao|depth|velocity|coc|none. A quality named in
// the URL wins over the saved one.
//
// window.__ui: { app, settings, store, set, layout, rig, interaction, panel, card, nodes, edges,
// labels, gfx, data, env, field, lens, tools, explore, slots, hud, photo, help, share(),
// features, refreshFeatures(), registerCommands() } for the dev pages and the console.
import './ui/ui.css';
import { createApp } from './app/App.js';
import { fetchManifest } from './data/load.js';
import { DataClient } from './data/client.js';
import { ForceLayout, LAYOUT_MODES } from './gpu/ForceLayout.js';
import { NodeRenderer, NODE_LOOK } from './render/NodeRenderer.js';
import { EdgeRenderer } from './render/EdgeRenderer.js';
import { TextLabels } from './render/TextLabels.js';
import { Background, ENV_PRESETS } from './render/Background.js';
import { ClusterField } from './render/ClusterField.js';
import { CameraRig } from './interact/CameraRig.js';
import { Interaction } from './interact/Interaction.js';
import { Panel } from './ui/Panel.js';
import { NodeCard } from './ui/NodeCard.js';
import { uiSlots } from './ui/slots.js';
import { Hud } from './ui/Hud.js';
import { PhotoMode } from './ui/PhotoMode.js';
import { Help, isTypingTarget } from './ui/Help.js';
import {
	SettingsStore, settingSpec, coerce, decodeShare, encodeShare, hasShare, parseLegacyParams, autoChoices, describeTier,
	describeClustering, deferredStorage, LOOKS, LOOK_KEYS, lookValues,
} from './ui/settings.js';
import { Lens } from './analysis/Lens.js';
import { Tools } from './analysis/Tools.js';
import { installExplore } from './explore/index.js';
import { hierarchicalPaths } from './graph/bundling.js';

const params = new URLSearchParams( location.search );
const boot = document.getElementById( 'boot' );
const clamp = ( v, lo, hi ) => Math.min( hi, Math.max( lo, v ) );
const cap = ( s ) => ( s ? s[ 0 ].toUpperCase() + s.slice( 1 ) : '' );

/** What the data layer reads when it does not say so itself (data.formats()). */
const BASE_FORMATS = [
	{ id: 'json', name: 'Graph JSON (wgv-graph, node-link, ImsvGraphVis)', extensions: [ '.json' ] },
	{ id: 'edgelist', name: 'Edge list, Matrix Market', extensions: [ '.txt', '.csv', '.tsv', '.edges', '.mtx' ] },
];
const LAYOUT_PARAMS = [ 'clusterSeparation', 'cohesion', 'hubSpacing', 'layoutSize', 'repulsion', 'damping' ];
const SHARE_KEYS = [ 'ds', 's', 'c', 'x' ]; // members of the address that name a view

function fail( title, detail ) {

	boot.classList.remove( 'hidden' );
	boot.innerHTML = `<div class="msg"><h1>${title}</h1><p>${detail}</p></div>`;
	window.__error = title + ': ' + detail;

}

function localStore() {

	try {

		const s = window.localStorage;
		s.setItem( 'wgv:probe', '1' );
		s.removeItem( 'wgv:probe' );
		return s;

	} catch ( _ ) {

		return null; // blocked (private mode, embedded): the session works without it

	}

}

async function main() {

	let app;
	try {

		app = await createApp( document.getElementById( 'app' ) );

	} catch ( e ) {

		if ( e.code === 'NO_WEBGPU' ) {

			fail( 'WebGPU is not available', `${e.message}<br><br>Use a current Chrome, Edge or Safari (macOS/iOS 26+), or Firefox 141+ on Windows. On Linux and Android Chrome, WebGPU may need to be enabled in <code>chrome://flags</code>.` );
			return;

		}

		throw e;

	}

	window.__app = app;
	const gfx = app.gfx;

	// ── settings: saved choices, then what the address asks for ──────────────────────────
	const storage = deferredStorage( localStore() );
	const store = new SettingsStore( { storage } );
	store.load();
	const settings = store.values; // the one settings object every module reads
	const legacy = parseLegacyParams( params, location.hash );
	const shared = hasShare( params );
	const link = shared ? decodeShare( params ) : { dataset: null, settings: {}, camera: null, explore: null };
	// a share link describes the whole view: the settings it does not name are the defaults (they
	// were, where it was made), whatever this browser has saved. The saved choices stay in storage
	if ( shared ) store.beginLink( link.settings );
	for ( const [ key, value ] of Object.entries( legacy.settings ) ) if ( ! ( key in link.settings ) ) store.set( key, value, { by: 'link' } );
	const startDataset = link.dataset || legacy.dataset;
	if ( startDataset ) store.set( 'dataset', startDataset, { by: 'link' } );
	if ( params.has( 'quality' ) ) store.set( 'quality', gfx.setting, { by: 'link' } ); // parsed by the App
	else if ( settings.quality !== gfx.setting ) gfx.setQuality( settings.quality );
	const phone = typeof matchMedia === 'function' && matchMedia( '(max-width: 640px)' ).matches;
	if ( phone ) store.setDefault( 'showStatus', false );
	window.addEventListener( 'pagehide', () => storage?.flush() );

	// ── modules ──────────────────────────────────────────────────────────────────────────
	const rig = new CameraRig( { autoRotate: settings.autoRotate } );
	if ( rig.reducedMotion ) {

		store.setDefault( 'autoRotate', false );
		rig.autoRotate = settings.autoRotate;

	}

	const data = new DataClient();
	const layout = app.use( new ForceLayout( { data, mode: settings.layoutMode, ...Object.fromEntries( LAYOUT_PARAMS.map( ( k ) => [ k, settings[ k ] ] ) ) } ) );
	const edges = app.use( new EdgeRenderer( {
		beta: settings.bundleMode === 'straight' ? 0 : settings.bundling,
		opacity: settings.edgeOpacity, width: settings.edgeWidth,
		blendMode: settings.edgeBlend === 'auto' ? 'normal' : settings.edgeBlend, // 'auto' is decided per graph (applyEdgeBlend)
	} ) );
	const nodes = app.use( new NodeRenderer( { size: settings.nodeSize } ) );
	const env = app.use( new Background( { preset: settings.theme } ) ); // the Environment: sky, key light, shadows, fog
	const field = app.use( new ClusterField( { strength: settings.fieldStrength } ) ); // soft glow volume per community
	// GPU text in the overlay pass of the render graph: full resolution, after tone mapping, soft
	// occlusion from the scene depth
	const labels = app.use( new TextLabels( { mode: settings.labels } ) );
	gfx.attachLabels( labels );

	let panel = null, explore = {}, hud = null, photo = null, help = null;
	const slots = uiSlots(); // screen layout of the floating surfaces (search, lens bar, card, status)
	const cycle = ( list, v ) => list[ ( list.indexOf( v ) + 1 ) % list.length ];
	const formats = () => ( typeof data.formats === 'function' ? data.formats() : null ) || BASE_FORMATS;
	const extensions = () => [ ...new Set( formats().flatMap( ( f ) => f.extensions || [] ) ) ];
	const acceptPattern = () => new RegExp( '(' + extensions().map( ( e ) => e.replace( /[.*+?^${}()|[\]\\]/g, '\\$&' ) ).join( '|' ) + ')(\\.gz)?$', 'i' );
	const interaction = new Interaction( {
		layout, rig,
		accept: acceptPattern(),
		onDropFile: ( file ) => openFile( file ),
		onDropRejected: ( file ) => panel.toast( `“${file.name}” is not a graph file this viewer reads. It opens ${extensions().join( ', ' )}.` ),
		onCommand: ( name ) => {

			if ( name === 'toggle-run' ) return set( 'running', ! settings.running );
			if ( name === 'cycle-labels' ) return set( 'labels', cycle( [ 'auto', 'all', 'off' ], settings.labels ) );
			if ( name === 'help' ) return help.toggle(), true;
			return false;

		},
	} );
	interaction.picking.setRadiusFunction( ( i ) => nodes.worldRadius( i ) );
	app.use( interaction );
	app.use( rig );
	const card = app.use( new NodeCard( { interaction, rig } ) );
	// metric lens (size / colour / glow by a graph metric) and tools (ripple, k-hop, shortest path)
	const lens = app.use( new Lens() );
	const tools = app.use( new Tools() );
	app.events.addEventListener( 'lens', ( e ) => {

		if ( e.detail.reason !== 'progress' ) labels.setColors?.( lens.colors );

	} );

	// ── what is there: modules other packages are still building are feature-detected ─────
	const F = {};
	const setPaths = ( paths ) => ( typeof edges.setPaths === 'function' ? edges.setPaths( paths ) : edges.curves.setPaths( paths ) );
	function features() {

		return {
			recluster: typeof data.recluster === 'function',
			bundlePaths: typeof data.bundle === 'function' && ( typeof edges.setPaths === 'function' || typeof edges.curves?.setPaths === 'function' ),
			edgeStyle: 'style' in edges,
			edgeFlow: 'flow' in edges,
			field: typeof field.focus === 'function' || !! field.uniforms || !! field.mesh,
			clusterLabels: 'clusterLabels' in labels,
			nodeLook: typeof nodes.setLook === 'function',
			env: typeof env.setPreset === 'function',
			dust: 'dust' in env,
			formats: formats(),
		};

	}

	/** Reads the features again (a module arrived, or a dev page installed one) and shows their controls. */
	function refreshFeatures() {

		Object.assign( F, features() );
		interaction.options.accept = acceptPattern();
		help?.setFormats( F.formats );
		if ( panel ) {

			panel.setFeatures( F );
			if ( F.dust ) syncDust();
			applyEdgeStyle();

		}

		return F;

	}

	Object.assign( F, features() );

	/** A module may build what a feature needs with its first graph (the curve cache of the edges): look again. */
	function detectFeatures() {

		const now = features();
		let changed = false;
		for ( const k in now ) if ( k !== 'formats' && now[ k ] !== F[ k ] ) changed = true;
		if ( changed ) refreshFeatures();

	}

	// ── applying settings ─────────────────────────────────────────────────────────────────
	const preset = () => ENV_PRESETS[ settings.theme ] || ENV_PRESETS.aurora;

	/** Gloss and glow are multipliers on the node look of the theme. */
	function applyLook() {

		if ( ! F.nodeLook ) return;
		const base = { ...NODE_LOOK, ...( preset().nodes || {} ) };
		nodes.setLook( {
			roughness: clamp( base.roughness * Math.pow( 2, ( 1 - settings.nodeGloss ) * 1.6 ), 0.05, 1 ),
			emissive: base.emissive * settings.nodeGlow,
			rim: base.rim,
		} );

	}

	/** Key light, environment light and fog are multipliers on the theme's values. */
	function applyLight() {

		if ( ! F.env ) return;
		const p = preset();
		env.keyIntensity = p.key.light * settings.keyLight;
		env.envIntensity = p.envIntensity * settings.envLight;
		env.fog = clamp( p.fog * settings.fog, 0, 1 );

	}

	/** The motes have a value per theme; a value the user chose stays. */
	function syncDust() {

		if ( store.isTouched( 'dust' ) ) env.dust = settings.dust;
		else {

			store.setDefault( 'dust', env.dust );
			panel?.set( 'dust', settings.dust );

		}

	}

	function applyTheme() {

		if ( ! F.env ) return;
		env.setPreset( settings.theme, { nodes: false } ); // the node look is ours: preset × the user's multipliers
		applyLook();
		applyLight();
		if ( F.dust ) syncDust();

	}

	function applyDof() {

		const mode = settings.dof;
		gfx.set( 'dof', mode === 'auto' ? null : mode === 'on' );
		gfx.set( 'dofStrength', mode === 'on' ? settings.dofStrength : null );

	}

	/** 'auto' draws solid tubes on tiny graphs and luminous strands otherwise. */
	function applyEdgeStyle() {

		if ( ! F.edgeStyle ) return;
		const style = settings.edgeStyle === 'auto' ? autoChoices( app.graph, { tubes: true } ).edgeStyle : settings.edgeStyle;
		if ( edges.style !== style ) edges.style = style;
		panel?.setInfo( { edgeStyle: style } );

	}

	/** 'auto' draws additive strands on large graphs, where they glow in the dense places, and normal ones otherwise. */
	function applyEdgeBlend() {

		const blend = settings.edgeBlend === 'auto' ? autoChoices( app.graph ).edgeBlend : settings.edgeBlend;
		if ( edges.blendMode !== blend ) edges.blendMode = blend;

	}

	function apply( key, v ) {

		switch ( key ) {

			case 'dataset': loadDataset( v, { user: true } ); break;
			case 'layoutMode':
				layout.setLayoutMode( v ); // the camera refits by itself (the rig listens to 'layout')
				if ( v !== 'sphere' && settings.immersive ) set( 'immersive', false );
				break;
			case 'running': v ? layout.resume() : layout.pause(); break;
			case 'immersive': rig.setImmersive( v ); break;
			case 'clusterSeparation': case 'cohesion': case 'hubSpacing': case 'layoutSize': case 'repulsion': case 'damping':
				layout[ key ] = v; // the layout reheats by itself
				break;
			case 'clusterMethod': case 'clusterResolution': scheduleRecluster(); break;
			case 'fieldStrength': if ( F.field ) field.strength = v; break;
			case 'clusterLabels': if ( F.clusterLabels ) labels.clusterLabels = v; break;
			case 'bundleMode': applyBundle(); break;
			case 'bundling': if ( settings.bundleMode !== 'straight' ) edges.setBundling( v ); break;
			case 'edgeStyle': applyEdgeStyle(); break;
			case 'edgeFlow': if ( F.edgeFlow ) edges.flow = v; break;
			case 'edgeOpacity': edges.opacity = v; break;
			case 'edgeWidth': edges.width = v; break;
			case 'edgeBlend': applyEdgeBlend(); break;
			case 'nodeSize': nodes.size = v; break;
			case 'labels': labels.mode = v; break;
			case 'nodeGloss': case 'nodeGlow': applyLook(); break;
			case 'palette':
				lens.setPalette( v ); // the lens owns the node colours (cluster palette or a metric ramp)
				card.setPalette?.( v );
				break;
			case 'theme': applyTheme(); break;
			case 'keyLight': case 'envLight': case 'fog': applyLight(); break;
			case 'dust': if ( F.dust ) env.dust = v; break;
			case 'autoRotate': rig.setAutoRotate( v ); break;
			case 'dof': case 'dofStrength': applyDof(); break;
			case 'bloom': gfx.set( 'bloom', v ); break;
			case 'exposure': gfx.set( 'exposure', v ); break;
			case 'aoStrength': gfx.set( 'aoStrength', v ); break;
			case 'grain': gfx.set( 'grain', v ? null : false ); break;
			case 'vignette': gfx.set( 'vignette', v ? null : false ); break;
			case 'quality': gfx.setQuality( v ); break;
			case 'dynamicResolution': gfx.set( 'dynamicResolution', v ? null : false ); break;
			case 'showStatus': hud.setVisible( v ); break;

		}

	}

	/** Changes a setting as if the user had: remembered, shown in the panel, applied. Returns whether the value was accepted. */
	function set( key, v ) {

		const spec = settingSpec( key );
		if ( ! spec || coerce( spec, v ) === undefined ) return false; // no such setting, or not a value it can take
		store.set( key, v );
		dropLink();
		panel.set( key, settings[ key ] );
		apply( key, settings[ key ] );
		return true;

	}

	/**
	 * A setting that follows what happened elsewhere (the layout paused by a key, the immersive view
	 * left through the rig, the graph that was actually loaded): written through the store like
	 * every change and shown in the panel; there is nothing to apply.
	 */
	function follow( key, v ) {

		store.set( key, v );
		panel?.set( key, settings[ key ] );

	}

	/** An automatic choice: only where the user has not chosen. */
	function setAuto( key, v ) {

		if ( ! store.auto( key, v ) ) return;
		panel.set( key, settings[ key ] );
		apply( key, settings[ key ] );

	}

	function reset( key = null ) {

		const changed = store.reset( key );
		for ( const k of changed ) {

			panel.set( k, settings[ k ] );
			apply( k, settings[ k ] );

		}

		// the per-graph choices apply again where a setting was handed back to them
		if ( app.graph && ( key === null || settingSpec( key )?.auto ) ) autoForGraph( app.graph );
		return changed;

	}

	/** A whole look (LOOKS): every setting it names becomes the user's choice; what it leaves open goes back to the default, and to the automatic per-graph choice where there is one. */
	function setLook( id ) {

		const values = lookValues( id );
		if ( ! values ) return false;
		for ( const key of LOOK_KEYS ) {

			if ( values[ key ] === undefined ) reset( key );
			else set( key, values[ key ] );

		}

		if ( app.graph ) autoForGraph( app.graph );
		return true;

	}

	/** Startup: the modules got most values through their constructors. The rest is applied only where it differs from what a module starts with, so the App's URL switches (?dof=0 …) survive. */
	function applyInitial() {

		const d = store.defaults;
		if ( settings.theme !== d.theme || settings.nodeGloss !== d.nodeGloss || settings.nodeGlow !== d.nodeGlow ) applyLook();
		if ( settings.keyLight !== d.keyLight || settings.envLight !== d.envLight || settings.fog !== d.fog ) applyLight();
		for ( const key of [ 'exposure', 'aoStrength', 'grain', 'vignette', 'dynamicResolution', 'dof', 'edgeFlow', 'clusterLabels' ] ) {

			if ( settings[ key ] !== d[ key ] ) apply( key, settings[ key ] );

		}

		gfx.set( 'bloom', settings.bloom );
		if ( F.dust ) syncDust();
		if ( settings.palette !== d.palette ) {

			lens.setPalette( settings.palette );
			card.setPalette?.( settings.palette );

		}

	}

	// ── data ──────────────────────────────────────────────────────────────────────────────
	data.onProgress( ( { stage, fraction, message } ) => panel && panel.setLoading( stage === 'done' ? null : message, fraction ) );
	const prepareOptions = () => ( { palette: settings.palette } );
	let manifest = { datasets: [] };
	let current = { id: null, name: '', shareable: false, request: null }; // the graph on screen

	/** Per-graph automatic choices; none of them overrides what the user chose. */
	function autoForGraph( g ) {

		const a = autoChoices( g, { tubes: F.edgeStyle, fieldDefault: store.defaults.fieldStrength } );
		if ( F.field ) setAuto( 'fieldStrength', a.fieldStrength ); // no community field without communities
		applyEdgeBlend(); // 'auto': dense graphs read better with additive (glow) strands
		applyEdgeStyle(); // 'auto': tubes on tiny graphs

	}

	function describeGraph( g ) {

		panel.setInfo( { clustering: { count: g.stats ? g.stats.clusters : 0, text: describeClustering( g ) } } );
		panel.update( { nodes: g.N, edges: g.E, clusters: g.stats ? g.stats.clusters : undefined, levels: g.D }, { force: true } );

	}

	function showGraph( graph, info, { keepView = false } = {} ) {

		if ( keepView ) rig.keepPoseOnNextGraph();
		bundle.reset();
		app.setGraph( graph );
		detectFeatures();
		current = { ...current, ...info, name: graph.name || info.name || current.name };
		autoForGraph( graph );
		if ( settings.running ) layout.resume(); else layout.pause();
		if ( current.id ) follow( 'dataset', current.id ); // a file or a generated graph has an id of its own
		panel.setDatasetInfo( { id: current.id, name: current.name, nodes: graph.N, edges: graph.E, levels: graph.D } );
		describeGraph( graph );
		document.title = `${current.name} · Graph · WebGPU`;
		if ( settings.bundleMode === 'paths' ) applyBundle();

	}

	async function show( request, info ) {

		// a re-clustering that waits or runs belongs to the graph this load replaces
		const dropped = cancelRecluster();
		try {

			const r = await request();
			if ( r.cancelled ) return false;
			interaction.clearSelection?.();
			// the clustering choice belongs to the graph on screen: a new graph starts as loaded
			if ( settings.clusterMethod !== 'default' && ! info.keepClustering ) {

				store.reset( 'clusterMethod' );
				store.reset( 'clusterResolution' );
				panel.refresh();

			}

			panel.setInfo( { recluster: null } );
			showGraph( r.graph, { ...info, request } );
			return true;

		} catch ( e ) {

			console.warn( e ); // handled: the person sees a toast, the graph on screen stays
			// the data layer starts its message with the name of the graph, which the toast says already
			const stem = String( info.name || '' ).replace( /\.[a-z0-9]+(\.gz)?$/i, '' );
			let why = String( e && e.message || e );
			for ( const lead of [ info.name, stem ] ) if ( lead && why.startsWith( lead + ': ' ) ) why = why.slice( lead.length + 2 );
			panel.toast( `Could not open ${info.name ? `“${info.name}”` : 'the graph'}: ${why}`, { type: 'error' } );
			// the graph on screen stays, and so do its name in the picker, its address and the
			// communities that were being computed for it
			if ( current.id ) {

				follow( 'dataset', current.id );
				writeAddress( current.shareable ? current.id : null );
				if ( dropped ) scheduleRecluster();

			}

			return false;

		} finally {

			panel.setLoading( null );
			boot.classList.add( 'hidden' );

		}

	}

	/**
	 * The address follows the graph: built-in datasets live in the hash. `drop` removes ?ds= and a
	 * share link's members, which no longer describe the view once the person changed it.
	 */
	function writeAddress( id, { drop = false } = {} ) {

		const q = new URLSearchParams( location.search );
		const stale = drop && SHARE_KEYS.some( ( k ) => q.has( k ) );
		if ( stale ) {

			for ( const k of SHARE_KEYS ) q.delete( k );
			linkInAddress = false;

		}

		const query = stale ? ( q.toString() ? '?' + q.toString() : '' ) : location.search;
		const url = location.pathname + query + ( id && ! q.has( 'ds' ) ? '#' + id : '' );
		if ( url !== location.pathname + location.search + location.hash ) history.replaceState( null, '', url );

	}

	/** After the first change the link that opened the page is history. */
	let linkInAddress = hasShare( params ), started = false;
	function dropLink() {

		if ( ! linkInAddress || ! started ) return; // restoring the link is not a change
		linkInAddress = false;
		const q = new URLSearchParams( location.search );
		for ( const k of [ 's', 'c', 'x' ] ) q.delete( k );
		history.replaceState( null, '', location.pathname + ( q.toString() ? '?' + q.toString().replace( /%2F/gi, '/' ) : '' ) + location.hash );

	}

	function loadDataset( id, { user = false, keepClustering = false } = {} ) {

		const entry = manifest.datasets.find( ( d ) => d.id === id ) || manifest.datasets[ 0 ];
		if ( ! entry ) {

			// no dataset list (it could not be fetched): the page is up, files and the generator work
			panel.setLoading( null );
			boot.classList.add( 'hidden' );
			return Promise.resolve( false );

		}

		if ( entry.id !== id ) {

			if ( id && manifest.datasets.length ) panel.toast( `There is no dataset “${id}”. Showing ${entry.name}.`, { type: 'info' } );
			follow( 'dataset', entry.id );

		}

		writeAddress( entry.id, { drop: user } );
		return show( () => data.load( entry, prepareOptions() ), { id: entry.id, name: entry.name, shareable: true, keepClustering } );

	}

	function openFile( file ) {

		writeAddress( null, { drop: true } );
		return show( () => data.loadFile( file, prepareOptions() ), { id: 'file:' + file.name, name: file.name, shareable: false } );

	}

	function generate( payload ) {

		writeAddress( null, { drop: true } );
		return show( () => data.generate( { model: payload.model, params: payload.params, seed: payload.params?.seed }, prepareOptions() ), { id: `generated:${payload.model}`, name: payload.model, shareable: false } );

	}

	// ── communities: re-clustering the graph on screen ────────────────────────────────────
	let reclusterTimer = 0, reclusterJob = 0, reclusterPending = false;
	function scheduleRecluster() {

		if ( ! F.recluster || ! app.graph ) return;
		clearTimeout( reclusterTimer );
		reclusterPending = true;
		panel.setInfo( { recluster: { busy: true, text: 'Finding communities…' } } );
		reclusterTimer = setTimeout( recluster, 320 ); // a resolution drag is one request

	}

	/** Drops a re-clustering that waits or runs (its result would arrive for another graph). Returns whether there was one. */
	function cancelRecluster() {

		const was = reclusterPending;
		clearTimeout( reclusterTimer );
		reclusterJob ++;
		reclusterPending = false;
		if ( was ) panel.setInfo( { recluster: null } );
		return was;

	}

	async function recluster() {

		if ( ! F.recluster || ! app.graph ) return false;
		const job = ++ reclusterJob;
		reclusterPending = true;
		const method = settings.clusterMethod;
		const selected = interaction.selectedIndex;
		const positions = app.positions ? app.positions.slice() : undefined; // the layout morphs from where the nodes are
		try {

			let r;
			if ( method === 'default' ) {

				// back to the hierarchy the graph came with: its own request again
				if ( ! current.request ) return false;
				r = await current.request();

			} else {

				r = await data.recluster( { method, resolution: settings.clusterResolution }, { positions, ...prepareOptions() } );

			}

			if ( job !== reclusterJob || ! r || r.cancelled ) return false;
			showGraph( r.graph, {}, { keepView: true } );
			if ( selected >= 0 && selected < r.graph.N ) interaction.select( selected );
			panel.setInfo( { recluster: null } );
			app.events.dispatchEvent( new CustomEvent( 'recluster', { detail: { method, resolution: settings.clusterResolution, clustering: r.graph.clustering || null } } ) );
			return true;

		} catch ( e ) {

			if ( job !== reclusterJob ) return false;
			console.error( e );
			panel.setInfo( { recluster: { text: 'The communities could not be computed.', tone: 'warn', action: { name: 'recluster-retry', label: 'Retry' } } } );
			panel.toast( `Re-clustering failed: ${e.message || e}`, { type: 'error' } );
			return false;

		} finally {

			if ( job === reclusterJob ) {

				reclusterPending = false;
				panel.setLoading( null );

			}

		}

	}

	// ── edges: hierarchy bundling (default), edge-path bundling of the settled layout, straight ─
	const bundle = {
		job: 0, custom: false, waiting: false, running: false, timer: 0,
		/** Ends a request in flight (the data layer stops its worker job where it can). */
		cancel() {

			if ( this.running && typeof data.cancelBundle === 'function' ) data.cancelBundle();
			this.running = false;

		},
		reset() {

			this.job ++;
			this.cancel();
			this.custom = false; // a new graph starts with its hierarchical paths
			this.waiting = false;
			clearTimeout( this.timer );
			panel?.setInfo( { bundle: null } );

		},
	};

	async function applyBundle( { now = false } = {} ) {

		const mode = settings.bundleMode;
		const job = ++ bundle.job;
		bundle.cancel();
		bundle.waiting = false;
		clearTimeout( bundle.timer );
		edges.setBundling( mode === 'straight' ? 0 : settings.bundling );
		if ( ! F.bundlePaths || ! app.graph ) return;
		if ( mode !== 'paths' ) {

			panel.setInfo( { bundle: null } );
			if ( bundle.custom ) {

				setPaths( hierarchicalPaths( app.graph ) );
				bundle.custom = false;

			}

			return;

		}

		// edge paths are routed through the layout as it is: wait until it rests
		if ( ! now && ! layout.settled && ! layout.paused ) {

			bundle.waiting = true;
			panel.setInfo( { bundle: { busy: true, text: 'Waiting for the layout to settle…', action: { name: 'bundle-now', label: 'Now' } } } );
			return;

		}

		if ( ! app.positions ) {

			bundle.timer = setTimeout( () => job === bundle.job && applyBundle( { now } ), 200 );
			return;

		}

		const busy = ( fraction ) => panel.setInfo( { bundle: {
			busy: true, text: `Routing edges along the graph…${fraction > 0.005 && fraction < 1 ? ` ${Math.round( 100 * fraction )}%` : ''}`,
			action: { name: 'bundle-cancel', label: 'Cancel' },
		} } );
		busy( 0 );
		bundle.running = true;
		try {

			const r = await data.bundle( {
				positions: app.positions.slice(), mode: 'paths',
				onProgress: ( p ) => job === bundle.job && p && Number.isFinite( p.fraction ) && busy( p.fraction ),
			} );
			if ( job !== bundle.job ) return;
			bundle.running = false;
			if ( ! r || r.cancelled ) return;
			setPaths( r.paths );
			bundle.custom = true;
			const T = r.timings || {};
			const t = [ T.total, T.compute, T.roundTrip ].find( Number.isFinite );
			const ms = t === undefined ? '' : ` in ${t < 1000 ? Math.round( t ) + ' ms' : ( t / 1000 ).toFixed( 1 ) + ' s'}`;
			panel.setInfo( { bundle: { text: `Edges follow paths of the settled layout${ms}.`, action: { name: 'bundle-now', label: 'Update' } } } );

		} catch ( e ) {

			if ( job !== bundle.job ) return;
			bundle.running = false;
			console.error( e );
			panel.setInfo( { bundle: { text: 'Edge paths could not be computed.', tone: 'warn', action: { name: 'bundle-cancel', label: 'Back' } } } );
			panel.toast( `Edge-path bundling failed: ${e.message || e}`, { type: 'error' } );

		}

	}

	// ── share links ───────────────────────────────────────────────────────────────────────
	function shareUrl() {

		if ( ! current.shareable ) return null;
		let state = null;
		try {

			state = typeof explore.state === 'function' ? explore.state() : null;

		} catch ( e ) {

			console.warn( 'share: the explore state is not available', e );

		}

		const query = encodeShare( { dataset: current.id, settings: store.shared(), camera: rig.pose(), explore: state } );
		return `${location.origin}${location.pathname}?${query}`;

	}

	async function copyText( text ) {

		try {

			await navigator.clipboard.writeText( text );
			return true;

		} catch ( _ ) {

			// no clipboard permission (insecure context, embedded page): the classic way
			const ta = document.createElement( 'textarea' );
			ta.value = text;
			ta.setAttribute( 'readonly', '' );
			ta.style.cssText = 'position:fixed;left:-99px;top:0;opacity:0';
			document.body.append( ta );
			ta.select();
			let ok = false;
			try {

				ok = document.execCommand( 'copy' );

			} catch ( __ ) { /* not available */ }

			ta.remove();
			return ok;

		}

	}

	/** Copies a link that reopens this view: dataset, the chosen settings, camera pose, explore state. */
	async function share() {

		const url = shareUrl();
		if ( ! url ) {

			panel.toast( 'A link cannot carry a graph that came from a file or from the generator. Links work for the built-in datasets.', { type: 'info' } );
			return null;

		}

		if ( await copyText( url ) ) panel.toast( `Link copied · ${url.length} characters`, { type: 'info', duration: 3200 } );
		else {

			history.replaceState( null, '', url );
			panel.toast( 'Copying is not allowed here. The link is in the address bar.', { type: 'info' } );

		}

		return url;

	}

	// ── view insets: what the surfaces cover, so the camera centres the graph in what is left ─
	let exploreInsets = null, insetTimer = 0;

	/** The settings panel (or the bar of the phone sheet) and, when the explore layer reports them, its drawer, card and bars: each side counts from its window edge. */
	function viewInsets() {

		const p = panel ? panel.insets() : { left: 0, right: 0, top: 0, bottom: 0 }, x = exploreInsets;
		if ( ! x ) return p;
		const side = ( k ) => Math.max( p[ k ] || 0, Number.isFinite( x[ k ] ) ? x[ k ] : 0 );
		return { left: side( 'left' ), right: side( 'right' ), top: side( 'top' ), bottom: side( 'bottom' ) };

	}

	function applyInsets( options ) {

		if ( photo && photo.active ) return; // the picture owns the whole window
		rig.setInsets( viewInsets(), options );

	}

	// ── panel ─────────────────────────────────────────────────────────────────────────────
	help = new Help( {
		shortcuts: () => explore.shortcuts,
		formats: F.formats,
		seen: () => !! store.ui( 'hintSeen' ),
		markSeen: () => store.ui( 'hintSeen', true ),
		items: () => {

			const touch = panel.isSheet;
			const list = [ { strong: 'Drag', text: 'to orbit' }, { strong: touch ? 'Tap' : 'Click', text: 'a node' } ];
			if ( explore.search && ! touch ) list.push( { key: '/', text: 'to search' } );
			if ( touch ) list.push( { strong: 'Pinch', text: 'to zoom' } );
			else list.push( { strong: 'Drop', text: 'a file to open it' }, { key: '?', text: 'for help' } );
			return list;

		},
	} );
	panel = new Panel( {
		settings,
		generators: data.listModels(),
		features: F,
		help,
		ui: { get: ( name ) => store.ui( name ), set: ( name, value ) => store.ui( name, value ) },
		// the store is the only writer: an explicit choice is validated, remembered (automatic values
		// leave it alone from then on) and heard by the store's listeners; the panel shows the result
		write: ( key, v ) => store.set( key, v ),
		onChange: ( key ) => {

			dropLink();
			apply( key, settings[ key ] );

		},
		onAction: ( name, payload ) => {

			switch ( name ) {

				case 'fit': rig.fitToGraph(); break;
				case 'reheat': layout.reheat(); break;
				case 'reset':
					layout.resetPositions();
					rig.keepFramed( 4 );
					break;
				case 'generate': generate( payload ); break;
				case 'open-file': openFile( payload ); break;
				case 'look': setLook( payload ); break;
				case 'reset-setting': reset( payload ); break;
				case 'reset-all':
					reset();
					panel.toast( 'All settings are back to their defaults.', { type: 'info', duration: 3200 } );
					break;
				case 'bundle-now': applyBundle( { now: true } ); break;
				case 'bundle-cancel':
					bundle.cancel();
					set( 'bundleMode', 'hierarchy' );
					break;
				case 'recluster-retry': recluster(); break;

			}

		},
		onOpenFile: ( file ) => openFile( file ),
		onInsets: ( insets ) => {

			applyInsets();
			slots.setInsets( insets );

		},
	} );
	applyInsets( { animate: false } );
	slots.setInsets( panel.insets() );
	applyInitial();

	// ── status, tools, photo mode ─────────────────────────────────────────────────────────
	hud = app.use( new Hud( { slots, layout, visible: settings.showStatus } ) );
	photo = new PhotoMode( {
		app, rig, interaction, panel,
		insets: () => viewInsets(),
		name: () => current.name,
		getExposure: () => settings.exposure,
		setExposure: ( v ) => set( 'exposure', v ),
	} );
	const fullscreen = () => ( document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen() )?.catch?.( () => panel.toast( 'Full screen is not available here.', { type: 'info' } ) );
	hud.addTool( { id: 'photo', icon: 'camera', label: 'Photo mode', shortcut: 'P', order: 10, onClick: () => photo.toggle() } );
	hud.addTool( { id: 'share', icon: 'link', label: 'Copy a link to this view', order: 11, onClick: () => share() } );
	if ( document.fullscreenEnabled ) {

		hud.addTool( { id: 'fullscreen', icon: 'expand', label: 'Full screen', order: 12, desktop: true, onClick: fullscreen } );
		document.addEventListener( 'fullscreenchange', () => hud.setTool( 'fullscreen', document.fullscreenElement ? { icon: 'shrink', label: 'Leave full screen' } : { icon: 'expand', label: 'Full screen' } ) );

	}

	hud.addTool( { id: 'help', icon: 'help', label: 'Shortcuts and gestures', shortcut: '?', order: 13, onClick: () => help.toggle() } );

	window.addEventListener( 'keydown', ( e ) => {

		if ( e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.repeat || isTypingTarget( e.target ) || help.visible ) return;
		if ( e.key === 'p' || e.key === 'P' ) photo.toggle();
		else if ( ( e.key === 'h' || e.key === 'H' ) && ! photo.active ) set( 'showStatus', ! settings.showStatus );

	} );

	// ── the explore layer, and the shell's entries in its command palette ─────────────────
	explore = installExplore( {
		app, settings, set, data, layout, nodes, edges, labels, env, rig, interaction, card, lens, tools, panel, slots, gfx, field,
		store, hud, photo, help, share,
		cameraInsets: false, // the shell owns the rig: it combines the panel's insets with this layer's (below)
	} ) || {};

	// the explore layer says what its surfaces cover (the card and the drawer on the right, the bars
	// at the bottom). The view follows a moment later: a click that opens the card is often the
	// first half of a double click, and the node must still be under the pointer for the second
	app.events.addEventListener( 'exploreinsets', ( e ) => {

		exploreInsets = e.detail || null;
		clearTimeout( insetTimer );
		insetTimer = setTimeout( applyInsets, 360 );

	} );
	if ( typeof explore.insets === 'function' ) {

		try {

			exploreInsets = explore.insets();
			applyInsets( { animate: false } );

		} catch ( e ) {

			console.warn( 'the explore layer could not report its insets', e );

		}

	}

	function registerCommands() {

		const C = explore.commands;
		if ( ! C || typeof C.register !== 'function' ) return 0;
		let n = 0;
		const has = ( id ) => typeof C.has === 'function' && C.has( id );
		const add = ( id, title, run, { group = 'View', keywords, shortcut, suggested } = {} ) => {

			C.register( { id: 'shell.' + id, title, group, keywords, shortcut, suggested, run } );
			n ++;

		};

		for ( const m of LAYOUT_MODES ) add( 'layout.' + m.id, `Layout: ${m.label}`, () => set( 'layoutMode', m.id ), { group: 'Layout', keywords: m.description } );
		add( 'layout.pause', 'Pause or resume the layout', () => set( 'running', ! settings.running ), { group: 'Layout', shortcut: 'Space' } );
		add( 'layout.reheat', 'Reheat the layout', () => layout.reheat(), { group: 'Layout', shortcut: 'R' } );
		add( 'layout.reset', 'Start the layout again from the seed positions', () => layout.resetPositions(), { group: 'Layout' } );
		// the explore layer has its own "fit" entry: one is enough
		if ( ! has( 'explore.fit' ) ) add( 'fit', 'Fit the graph in the window', () => rig.fitToGraph(), { group: 'Camera', shortcut: 'F', keywords: 'zoom home centre' } );
		add( 'autorotate', 'Auto-rotate on or off', () => set( 'autoRotate', ! settings.autoRotate ), { group: 'Camera', keywords: 'spin turn' } );
		for ( const o of settingSpec( 'quality' ).options ) add( 'quality.' + o.value, `Quality: ${o.label}`, () => set( 'quality', o.value ), { group: 'Quality', keywords: 'graphics performance' } );
		if ( F.env ) for ( const o of settingSpec( 'theme' ).options ) add( 'theme.' + o.value, `Theme: ${o.label}`, () => set( 'theme', o.value ), { group: 'Scene', keywords: 'look background lighting' } );
		for ( const l of LOOKS ) add( 'look.' + l.value, `Look: ${l.label}`, () => setLook( l.value ), { group: 'Scene', keywords: 'preset style ' + l.hint } );
		for ( const o of settingSpec( 'palette' ).options ) add( 'palette.' + o.value, `Palette: ${o.label}`, () => set( 'palette', o.value ), { group: 'Scene', keywords: 'colours colors' } );
		for ( const o of settingSpec( 'labels' ).options ) add( 'labels.' + o.value, `Labels: ${o.label}`, () => set( 'labels', o.value ), { group: 'Scene', keywords: 'names text' } );
		add( 'photo', 'Photo mode', () => photo.toggle(), { shortcut: 'P', keywords: 'capture screenshot picture image png', suggested: true } );
		add( 'share', 'Copy a link to this view', () => share(), { keywords: 'share url address' } );
		add( 'status', 'Status chips on or off', () => set( 'showStatus', ! settings.showStatus ), { shortcut: 'H', keywords: 'fps frame rate hud' } );
		if ( document.fullscreenEnabled ) add( 'fullscreen', 'Full screen on or off', fullscreen );
		add( 'help', 'Shortcuts and gestures', () => help.show( true ), { shortcut: '?', keywords: 'keys keyboard' } );
		add( 'reset', 'Reset all settings', () => reset(), { keywords: 'defaults' } );
		return n;

	}

	registerCommands();

	// ── events: what the modules report goes to the panel and the status ──────────────────
	const E = app.events;
	const layoutInfo = () => ( { mode: layout.layoutMode, settled: !! layout.settled, pending: !! layout.pending, running: ! layout.paused } );
	const qualityInfo = () => {

		const list = describeTier( gfx.config );
		const tier = gfx.tier;
		return { tier, text: `${gfx.setting === 'auto' ? `Auto chose ${tier} for this device and graph` : cap( tier )}: ${list.join( ', ' )}.` };

	};

	E.addEventListener( 'immersive', ( e ) => {

		if ( settings.immersive !== e.detail.immersive ) follow( 'immersive', e.detail.immersive );

	} );
	E.addEventListener( 'select', ( e ) => panel.setInfo( { selection: e.detail.index >= 0 } ) );
	E.addEventListener( 'gfx', () => panel.setInfo( { quality: qualityInfo() } ) );
	E.addEventListener( 'layout', ( e ) => {

		panel.setInfo( { layout: layoutInfo() } );
		// edge paths belong to one settled layout: route them again when it rests in a new place
		if ( e.detail.settled && settings.bundleMode === 'paths' && F.bundlePaths && ( bundle.waiting || bundle.custom ) ) applyBundle( { now: true } );

	} );
	E.addEventListener( 'photo', ( e ) => {

		if ( e.detail.active ) help.show( false );

	} );
	let infoAt = 0;
	// cinematic idle (the rig drifts after a long quiet time): the lens pulls focus slowly through
	// the graph. A selected node keeps the automatic focus on itself.
	let pulling = false;
	E.addEventListener( 'frame', () => {

		if ( rig.drifting && ! photo.active && interaction.selectedIndex < 0 ) {

			gfx.setFocus( rig.driftFocus );
			pulling = true;

		} else if ( pulling ) {

			pulling = false;
			if ( ! photo.active ) gfx.setFocus( null );

		}

	} );
	E.addEventListener( 'frame', () => {

		const now = performance.now();
		if ( now - infoAt < 250 ) return;
		infoAt = now;
		// the layout can be paused and resumed from elsewhere (keys, the explore layer): the setting follows
		if ( settings.running !== ! layout.paused ) follow( 'running', ! layout.paused );

		const S = gfx.stats;
		const gpu = typeof S.gpuMs === 'number' ? ` · ${S.gpuMs.toFixed( 1 )} ms` : '';
		panel.setInfo( {
			layout: layoutInfo(),
			render: { text: `${S.width}×${S.height} · ${Math.round( 100 * S.renderScale )}% · ${Math.round( app.fps || 0 )} fps${gpu}` },
		} );

	} );
	panel.setInfo( { quality: qualityInfo(), layout: layoutInfo() } );
	app.renderer.backend?.device?.lost?.then( ( info ) => {

		if ( info && info.reason === 'destroyed' ) return;
		panel.toast( 'The graphics device was lost. Reload the page to continue.', { type: 'error', duration: 0 } );

	} );

	window.__ui = {
		app, settings, store, set, layout, rig, interaction, panel, card, nodes, edges, labels, gfx, data, env, field, lens, tools, explore, slots,
		hud, photo, help, share, shareUrl, features: F, refreshFeatures, registerCommands, reset, setLook,
	};

	// ── first graph ───────────────────────────────────────────────────────────────────────
	try {

		manifest = await fetchManifest( 'data/manifest.json' );
		panel.setManifest( manifest );

	} catch ( e ) {

		panel.setManifest( manifest ); // an empty list, not "Loading…"
		panel.toast( `Could not load the dataset list: ${e.message}. You can still open a file or generate a graph.`, { type: 'error', duration: 9000 } );

	}

	const linked = link.settings.clusterMethod && link.settings.clusterMethod !== 'default';
	const loaded = await loadDataset( settings.dataset, { keepClustering: linked } );
	if ( loaded ) {

		// a share link: the communities it was made with, then the camera and the explore state
		if ( linked && F.recluster ) {

			clearTimeout( reclusterTimer );
			await recluster();

		}

		if ( settings.immersive && settings.layoutMode === 'sphere' ) {

			// inside the sphere a link's camera is a look (pose() of the immersive view): its direction and field of view
			if ( link.camera ) rig.setPose( { ...link.camera, immersive: true }, { animate: false } );
			else rig.setImmersive( true, { animate: false } );

		} else {

			if ( settings.immersive ) follow( 'immersive', false ); // only the sphere layout has an inside
			if ( link.camera ) rig.setPose( link.camera, { animate: false } );

		}
		if ( link.explore && typeof explore.restore === 'function' ) {

			try {

				await explore.restore( link.explore );

			} catch ( e ) {

				console.warn( 'share: the explore state could not be restored', e );

			}

		}

		if ( params.get( 'hint' ) !== '0' ) help.showHint( { force: params.get( 'hint' ) === '1' } );

	}

	started = true;
	window.__ready = true;

}

main().catch( ( e ) => {

	console.error( e );
	fail( 'Something went wrong', String( e && e.message || e ) );

} );
