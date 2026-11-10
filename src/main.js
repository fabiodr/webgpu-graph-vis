// Application entry: wires the modules together.
//
//   ForceLayout (GPU compute, order 0) → EdgeRenderer (10) → NodeRenderer (11) → Environment (12)
//   → TextLabels (20) → Lens, Tools (25) → Interaction (30) → CameraRig (31) → NodeCard (41)
//
// Rendering (scene pass, quality tiers, post-processing, label overlay) belongs to app.gfx, the
// render graph the App creates; the labels are drawn in its overlay pass after tone mapping.
// Data is loaded, clustered (Louvain when needed) and prepared in a Web Worker (DataClient); graph
// metrics are computed in a second worker and mapped onto the picture by the Lens. The explore
// layer (src/explore) adds the surfaces for reading the graph.
// URL parameters: ?ds=<dataset id> (e.g. imsv/lesmis, generated/hsbm-5k), ?layout=sphere,
// ?autorotate=0, and the graphics switches read by the App (parseGfxParams in app/Quality.js):
// ?quality=auto|low|medium|high|ultra, ?governor=0|1, ?ao=0|1, ?dof=0|1, ?shadows=0|1, ?traa=0|1,
// ?scale=<number>, ?debug=ao|depth|velocity|coc|none.
import './ui/ui.css';
import { createApp } from './app/App.js';
import { fetchManifest } from './data/load.js';
import { DataClient } from './data/client.js';
import { ForceLayout } from './gpu/ForceLayout.js';
import { NodeRenderer } from './render/NodeRenderer.js';
import { EdgeRenderer } from './render/EdgeRenderer.js';
import { TextLabels } from './render/TextLabels.js';
import { Background } from './render/Background.js';
import { CameraRig } from './interact/CameraRig.js';
import { Interaction } from './interact/Interaction.js';
import { Panel, DEFAULT_SETTINGS } from './ui/Panel.js';
import { NodeCard } from './ui/NodeCard.js';
import { Lens } from './analysis/Lens.js';
import { Tools } from './analysis/Tools.js';
import { installExplore } from './explore/index.js';

const params = new URLSearchParams( location.search );
const boot = document.getElementById( 'boot' );

function fail( title, detail ) {

	boot.classList.remove( 'hidden' );
	boot.innerHTML = `<div class="msg"><h1>${title}</h1><p>${detail}</p></div>`;
	window.__error = title + ': ' + detail;

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
	const settings = { ...DEFAULT_SETTINGS };
	if ( params.get( 'ds' ) ) settings.dataset = params.get( 'ds' );
	else if ( location.hash.length > 1 ) settings.dataset = decodeURIComponent( location.hash.slice( 1 ) );
	if ( params.get( 'layout' ) === 'sphere' ) settings.layoutMode = 'sphere';
	settings.quality = app.gfx.setting; // 'auto' unless ?quality= names a tier (parsed by the App)
	const touched = new Set(); // settings the user changed explicitly (auto values do not override them)

	const rig = new CameraRig();
	if ( rig.reducedMotion || params.get( 'autorotate' ) === '0' ) settings.autoRotate = false;
	rig.autoRotate = settings.autoRotate;

	const data = new DataClient();
	const layout = app.use( new ForceLayout( { data } ) ); // stress layouts run in the data worker
	const edges = app.use( new EdgeRenderer( { beta: settings.bundling, opacity: settings.edgeOpacity, width: settings.edgeWidth, blendMode: settings.edgeBlend } ) );
	const nodes = app.use( new NodeRenderer( { size: settings.nodeSize } ) );
	const env = app.use( new Background() ); // the Environment: sky, key light, shadows, fog
	// GPU text in the overlay pass of the render graph: full resolution, after tone mapping, soft
	// occlusion from the scene depth
	const labels = app.use( new TextLabels( { mode: settings.labels } ) );
	const gfx = app.gfx;
	gfx.attachLabels( labels );
	gfx.set( 'bloom', settings.bloom );

	// UI values are multipliers on ForceLayout's d3-style units (charge −30, automatic gravity)
	function applyForces() {

		layout.charge = - 30 * settings.charge;
		layout.linkDistance = settings.linkDistance;
		layout.theta = settings.theta;
		layout.velocityDecay = settings.damping;
		layout.gravity = null;
		if ( settings.gravity !== 1 && app.graph ) layout.gravity = layout.autoGravity * settings.gravity;

	}

	let panel = null;
	const cycle = ( list, v ) => list[ ( list.indexOf( v ) + 1 ) % list.length ];
	const interaction = new Interaction( {
		layout, rig,
		onDropFile: ( file ) => openFile( file ),
		onDropRejected: ( file ) => panel.toast( `Unsupported file “${file.name}”. Use .json, .txt, .csv, .tsv, .edges or .mtx.` ),
		onCommand: ( name ) => {

			if ( name === 'toggle-run' ) return set( 'running', ! settings.running );
			if ( name === 'cycle-labels' ) return set( 'labels', cycle( [ 'auto', 'all', 'off' ], settings.labels ) );
			if ( name === 'help' ) return panel.toggleHelp(), true;
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

	function apply( key, v ) {

		switch ( key ) {

			case 'dataset': loadDataset( v ); break;
			case 'layoutMode':
				layout.setLayoutMode( v );
				if ( v !== 'sphere' && settings.immersive ) set( 'immersive', false );
				if ( ! settings.immersive ) setTimeout( () => rig.fitToGraph(), 1100 );
				break;
			case 'running': v ? layout.resume() : layout.pause(); break;
			case 'autoRotate': rig.setAutoRotate( v ); break;
			case 'immersive': rig.setImmersive( v ); break;
			case 'bundling': edges.setBundling( v ); break;
			case 'edgeOpacity': edges.opacity = v; break;
			case 'edgeWidth': edges.width = v; break;
			case 'edgeBlend': edges.blendMode = v; break;
			case 'nodeSize': nodes.size = v; break;
			case 'bloom': gfx.set( 'bloom', v ); break;
			case 'quality': gfx.setQuality( v ); break;
			case 'labels': labels.mode = v; break;
			case 'palette':
				lens.setPalette( v ); // the lens owns the node colours (cluster palette or a metric ramp)
				card.setPalette?.( v );
				break;
			case 'charge': case 'linkDistance': case 'gravity': case 'theta': case 'damping':
				applyForces();
				if ( settings.running ) layout.reheat( 0.3 );
				break;

		}

	}

	function set( key, v ) {

		settings[ key ] = v;
		panel.set( key, v );
		apply( key, v );
		return true;

	}

	// ── data ──────────────────────────────────────────────────────────────────────────────
	data.onProgress( ( { stage, fraction, message } ) => panel && panel.setLoading( stage === 'done' ? null : message, fraction ) );
	const prepareOptions = () => ( { palette: settings.palette, linkDistance: settings.linkDistance } );
	let manifest = { datasets: [] };

	/** Applies per-graph automatic choices that the user has not overridden. */
	function autoForGraph( g ) {

		// dense graphs read better with additive (glow) edges
		if ( ! touched.has( 'edgeBlend' ) ) {

			const blend = g.E > 20000 ? 'additive' : 'normal';
			if ( settings.edgeBlend !== blend ) set( 'edgeBlend', blend );

		}

	}

	async function show( request, info ) {

		try {

			const r = await request;
			if ( r.cancelled ) return;
			app.setGraph( r.graph );
			applyForces();
			autoForGraph( r.graph );
			if ( settings.layoutMode === 'sphere' ) layout.setLayoutMode( 'sphere' );
			if ( settings.running ) layout.resume(); else layout.pause();
			interaction.clearSelection?.();
			panel.setDatasetInfo?.( { id: info.id, name: r.graph.name || info.name, nodes: r.graph.N, edges: r.graph.E, levels: r.graph.D } );
			rig.fitToGraph( { animate: false } );
			rig.keepFramed?.( 6 );
			document.title = `${r.graph.name} · Graph · WebGPU`;

		} catch ( e ) {

			console.error( e );
			panel.toast( e.message || String( e ), { type: 'error' } );

		} finally {

			panel.setLoading( null );
			boot.classList.add( 'hidden' );

		}

	}

	function loadDataset( id ) {

		const entry = manifest.datasets.find( ( d ) => d.id === id ) || manifest.datasets[ 0 ];
		if ( ! entry ) return Promise.resolve();
		if ( entry.id !== id ) settings.dataset = entry.id;
		history.replaceState( null, '', '#' + entry.id );
		return show( data.load( entry, prepareOptions() ), { id: entry.id, name: entry.name } );

	}

	function openFile( file ) {

		return show( data.loadFile( file, prepareOptions() ), { id: 'file:' + file.name, name: file.name } );

	}

	panel = new Panel( {
		settings,
		generators: data.listModels(),
		onChange: ( key, v ) => {

			touched.add( key );
			apply( key, v );

		},
		onAction: ( name, payload ) => {

			if ( name === 'fit' ) rig.fitToGraph();
			else if ( name === 'reheat' ) layout.reheat();
			else if ( name === 'reset' ) {

				layout.resetPositions();
				rig.keepFramed( 4 );

			} else if ( name === 'generate' ) {

				show( data.generate( { model: payload.model, params: payload.params, seed: payload.params?.seed }, prepareOptions() ), { id: `generated:${payload.model}`, name: payload.model } );

			}

		},
		onOpenFile: ( file ) => openFile( file ),
		onInsets: ( insets ) => rig.setInsets( insets ),
	} );
	rig.setInsets( panel.insets(), { animate: false } );
	const explore = installExplore( { app, settings, set, data, layout, nodes, edges, labels, env, rig, interaction, card, lens, tools, panel } );
	app.events.addEventListener( 'immersive', ( e ) => {

		if ( settings.immersive !== e.detail.immersive ) panel.set( 'immersive', e.detail.immersive );

	} );
	app.events.addEventListener( 'frame', () => {

		const g = app.graph;
		if ( g ) panel.update( { nodes: g.N, edges: g.E, clusters: g.stats.clusters, levels: g.D, fps: app.fps, alpha: layout.alpha, running: layout.running } );

	} );

	try {

		manifest = await fetchManifest( 'data/manifest.json' );
		panel.setManifest( manifest );

	} catch ( e ) {

		panel.toast( `Could not load the dataset list: ${e.message}`, { type: 'error' } );

	}

	await loadDataset( settings.dataset );
	if ( settings.layoutMode === 'sphere' ) panel.set( 'layoutMode', 'sphere' );
	window.__ui = { app, settings, layout, rig, interaction, panel, card, nodes, edges, labels, gfx, data, env, lens, tools, explore, set };
	window.__ready = true;

}

main().catch( ( e ) => {

	console.error( e );
	fail( 'Something went wrong', String( e && e.message || e ) );

} );
