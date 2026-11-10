// Explore layer: everything a person uses to read the graph rather than to style it (search and
// command palette, lens bar with legend, insights drawer, node card actions, path strip, tour).
//
// main.js installs it once with the application context:
//
//   const explore = installExplore( { app, settings, set, data, layout, nodes, edges, labels, env,
//                                     rig, interaction, card, lens, tools, panel, slots, gfx, field,
//                                     cameraInsets = true } )
//
// The layer builds its own DOM surfaces into the slots of src/ui/slots.js and listens to
// app.events; it does not own the settings panel, the camera rig or main.js.
//
//   surface            slot              module
//   search / palette   slots.top         Search.js      "/" or Ctrl / Cmd + K
//   insights toggle    slots.topRight    Insights.js    "I"  (CSS order 0)
//   insights drawer    slots.right       Insights.js
//   node card          slots.right end   ../ui/NodeCard.js  ( card.attach )
//   lens bar           slots.bottom      LensBar.js     (first child = lowest)
//   path strip         slots.bottom      PathStrip.js
//   tour caption       slots.bottom      Tour.js
//
// A new path whose nodes are not all inside the free part of the view is framed by the camera
// ( rig.flyTo over the path's bounding sphere ); a path that is in view, and one that comes back
// with explore.restore(), leave the camera alone. While photo mode hides the surfaces
// ( <html class="wgv-ui-hidden"> ) the search and open menus close and the layer's keys rest.
//
// The highlight (selection, neighbourhood, path, community focus, search preview) is one model
// owned by the Interaction module ( src/interact/Interaction.js, highlight.js ); this layer only
// asks it for things.
//
// ─── API (the object installExplore returns) ────────────────────────────────────────────────
//  explore.commands        CommandRegistry ( commands.js ): register( { id, title, group?, keywords?,
//                          shortcut?, hint?, icon?, suggested?, enabled?(), run() } ) -> unregister fn,
//                          unregister( id ), run( id ), list(), search( query )
//  explore.shortcuts       [ { keys, title, group } ] of this layer, for a help overlay (a fresh
//                          array each time it is read)
//  explore.search          { open( text? ), close(), toggle(), isOpen, stats }
//  explore.insights        { toggle( on? ), isOpen, setTopMetric( id ), topMetric }
//  explore.tour            { start() -> Promise<completed>, stop(), next(), previous(), running, step, stops }
//  explore.lens            { presets, applyPreset( id ), setChannel( 'color' | 'size' | 'glow' | 'edges',
//                          metric | null, opts? ), setEdgeMetric( id | null, { opacity?, width? } ),
//                          edge, compare( on ) }
//  explore.focusCommunity( cluster | −1, { frame = true } )   community focus (highlight model,
//                          'cluster' event, field.focus when the field has it) and camera
//  explore.communities()   the communities listed in the drawer ( communities.js )
//  explore.state() -> plain JSON ( state.js ),  explore.restore( state ) -> Promise
//  explore.insets() -> { left, right, top, bottom }   CSS px, measured from the window edges, that
//                          the surfaces of this layer cover at rest: the search box at the top,
//                          the right column while the card or the drawer is open, the bars at the
//                          bottom; on phones the card and the drawer are sheets and count into
//                          `bottom`. Unless installExplore got `cameraInsets: false`, the layer
//                          keeps the camera rig centred on what is left: rig.setInsets( the larger
//                          of panel.insets() and these, per side ), again whenever either changes
//                          (the panel's are noticed through the CSS variables that
//                          slots.setInsets() writes on <html>).
//  explore.surfaces        { search, lensBar, pathStrip, insights, tour, card } (the modules)
//  explore.dispose()
//
// Events this layer adds to app.events: 'cluster' { cluster, level } (through Interaction),
// 'tour' { running, step, stop }, 'insights' { open }, 'exploreinsets' { left, right, top,
// bottom } whenever explore.insets() changes. The node card adds 'card' { index, folded } when its
// phone sheet folds to its head or unfolds (the insets are measured again).

import './explore.css';
import { CommandRegistry } from './commands.js';
import { Search } from './Search.js';
import { LensBar } from './LensBar.js';
import { PathStrip } from './PathStrip.js';
import { Insights } from './Insights.js';
import { Tour } from './Tour.js';
import { LENS_PRESETS } from './presets.js';
import { normalizeState, STATE_VERSION } from './state.js';
import { isPhone } from './ui.js';
import { closePopover } from './popover.js';
import { Vector3 } from 'three/webgpu';

const _p = new Vector3();
const _c = new Vector3();

export function installExplore( ctx ) {

	const { app, settings, layout, edges, rig, interaction, card, lens, tools, panel, slots, field } = ctx;
	const mac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test( navigator.platform || navigator.userAgent || '' );
	const disposers = [];
	const on = ( type, fn ) => {

		app.events.addEventListener( type, fn );
		disposers.push( () => app.events.removeEventListener( type, fn ) );

	};

	interaction.setTools?.( tools );

	const commands = new CommandRegistry();
	commands.onError = ( e, command ) => panel?.toast?.( `“${command.title}” failed: ${e.message || e}`, { type: 'error' } );
	const search = new Search( { app, interaction, rig, lens, slots, commands } );
	const lensBar = new LensBar( { app, lens, edges, interaction, slots } );
	const pathStrip = new PathStrip( { app, interaction, lens, slots, onSearch: () => search.open() } );
	const insights = new Insights( { app, lens, interaction, rig, slots } );
	const tour = new Tour( { app, lens, interaction, rig, slots, insights } );

	function focusCommunity( cluster, { frame = true, toggle = false } = {} ) {

		if ( ! ( cluster >= 0 ) ) return interaction.focusCluster( - 1 );
		insights.focusCommunity( cluster, { toggle, frame } );

	}

	card?.attach?.( { lens, tools, layout, slots, settings, onFocusCluster: ( cluster ) => focusCommunity( cluster, { toggle: true } ) } );

	// ── events ────────────────────────────────────────────────────────────────────────────
	on( 'graph', ( e ) => search.setGraph( e.detail.graph ) );
	if ( app.graph ) search.setGraph( app.graph );
	on( 'command', ( e ) => {

		if ( e.detail.name === 'search' ) search.open();
		else if ( e.detail.name === 'insights' ) insights.toggle();

	} );
	// the community glow volumes of the atmosphere package emphasise the focused community
	on( 'cluster', ( e ) => field?.focus?.( e.detail.cluster >= 0 ? e.detail.cluster : null ) );

	// compare: flip between the metric colouring and the community palette while a key is held
	let compared = null;
	function compare( onNow ) {

		if ( onNow && ! compared ) {

			const s = lens.state;
			if ( ! s.color ) return;
			compared = { color: s.color, ramp: s.ramp, reverse: s.reverse };
			lens.setColor( null );

		} else if ( ! onNow && compared ) {

			const c = compared;
			compared = null;
			lens.setColor( c.color, { ramp: c.ramp, reverse: c.reverse } );

		}

	}

	// photo mode ( <html class="wgv-ui-hidden"> ) hides every surface of this layer: the search and
	// an open menu close, and the layer's own keys rest until it is over
	const uiHidden = () => document.documentElement.classList.contains( 'wgv-ui-hidden' );
	on( 'photo', ( e ) => {

		if ( ! e.detail || ! e.detail.active ) return;
		closePopover( { focus: false } );
		if ( search.isOpen ) search.close();
		compare( false );

	} );

	// ── a path is shown where it can be seen ──────────────────────────────────────────────
	// The target of a path is often picked through the search, so it may lie off screen or under
	// the right column. When any node of a new path is outside the free part of the view, the
	// camera frames the whole path; a path that is in view already leaves the camera alone.
	let restoring = false, frameTimer = 0;
	function freeRect() {

		const own = insets(), p = panel && typeof panel.insets === 'function' ? panel.insets() || {} : {};
		const side = ( k ) => Math.max( + p[ k ] || 0, own[ k ] || 0 );
		return { left: side( 'left' ), right: window.innerWidth - side( 'right' ), top: side( 'top' ), bottom: window.innerHeight - side( 'bottom' ) };

	}

	function framePath( nodes ) {

		const g = app.graph, pos = app.positions, cam = app.camera;
		if ( ! g || ! pos || ! cam || ! nodes || ! nodes.length || rig.immersive || tour.running || uiHidden() ) return false;
		const r = freeRect(), W = window.innerWidth, H = window.innerHeight, margin = 28;
		let outside = false;
		_c.set( 0, 0, 0 );
		for ( let k = 0; k < nodes.length; k ++ ) {

			const i = 4 * nodes[ k ];
			_p.set( pos[ i ], pos[ i + 1 ], pos[ i + 2 ] );
			_c.add( _p );
			_p.project( cam ); // the projection carries the view offset of the rig's insets
			const x = ( _p.x * 0.5 + 0.5 ) * W, y = ( 0.5 - _p.y * 0.5 ) * H;
			if ( ! ( _p.z > - 1 && _p.z < 1 ) || x < r.left + margin || x > r.right - margin || y < r.top + margin || y > r.bottom - margin ) outside = true;

		}

		if ( ! outside ) return false;
		_c.multiplyScalar( 1 / nodes.length );
		let r2 = 0;
		for ( let k = 0; k < nodes.length; k ++ ) {

			const i = 4 * nodes[ k ];
			r2 = Math.max( r2, ( pos[ i ] - _c.x ) ** 2 + ( pos[ i + 1 ] - _c.y ) ** 2 + ( pos[ i + 2 ] - _c.z ) ** 2 );

		}

		const radius = Math.max( Math.sqrt( r2 ) * 1.3, ( g.layoutRadius || 1 ) * 0.1 );
		const half = cam.fov ? Math.atan( Math.tan( cam.fov * Math.PI / 360 ) * Math.min( 1, cam.aspect || 1 ) ) : 0.35;
		rig.flyTo( _c, rig.fitDistance ? rig.fitDistance( radius ) : radius / Math.sin( half ) );
		return true;

	}

	on( 'path', ( e ) => {

		clearTimeout( frameTimer );
		const nodes = e.detail && e.detail.nodes;
		// a moment later: the strip has appeared (and, on a phone, the card has folded), so the free
		// part of the view is the one the path will be seen in. A restored view keeps its camera.
		if ( nodes && nodes.length && ! restoring ) frameTimer = setTimeout( () => framePath( nodes ), 460 );

	} );
	on( 'graph', () => clearTimeout( frameTimer ) );
	disposers.push( () => clearTimeout( frameTimer ) );

	const typing = ( t ) => !! ( t && t.closest && t.closest( 'input, textarea, select, [contenteditable=""], [contenteditable="true"]' ) );
	const onKeyDown = ( e ) => {

		if ( e.repeat || e.ctrlKey || e.metaKey || e.altKey || typing( e.target ) || uiHidden() ) return;
		if ( e.key === 'c' || e.key === 'C' ) compare( true );

	};

	const onKeyUp = ( e ) => {

		if ( e.key === 'c' || e.key === 'C' ) compare( false );

	};

	const onBlur = () => compare( false );
	window.addEventListener( 'keydown', onKeyDown );
	window.addEventListener( 'keyup', onKeyUp );
	window.addEventListener( 'blur', onBlur );
	disposers.push( () => {

		window.removeEventListener( 'keydown', onKeyDown );
		window.removeEventListener( 'keyup', onKeyUp );
		window.removeEventListener( 'blur', onBlur );

	} );

	// ── commands ──────────────────────────────────────────────────────────────────────────
	const hasSelection = () => interaction.selectedIndex >= 0;
	commands.register( { id: 'explore.tour', title: 'Start the tour', group: 'Explore', icon: 'tour', keywords: 'guide intro walkthrough visit', suggested: true, run: () => tour.start() } );
	commands.register( { id: 'explore.insights', title: 'Insights: the graph in numbers', group: 'Explore', icon: 'insights', shortcut: 'I', keywords: 'statistics summary drawer communities top nodes', suggested: true, run: () => insights.toggle() } );
	for ( const p of LENS_PRESETS ) {

		commands.register( {
			id: 'lens.preset.' + p.id, title: `Lens: ${p.label}`, group: 'Lens', icon: 'color', hint: p.id === 'communities' ? 'community palette' : p.description.split( ':' )[ 0 ],
			keywords: [ 'preset', 'colour', 'color', 'size', p.color, p.size, p.glow ].filter( Boolean ), suggested: p.id === 'influence' || p.id === 'communities',
			run: () => lensBar.applyPreset( p.id ),
		} );

	}

	// one command per metric and channel; the attributes of a file come and go with the graph
	let metricCommands = [];
	function registerMetricCommands() {

		for ( const off of metricCommands ) off();
		metricCommands = [];
		for ( const m of lens.metrics ) metricCommands.push( commands.register( { id: 'lens.color.' + m.id, title: `Colour by ${m.label}`, group: 'Lens', icon: 'color', keywords: [ 'lens', 'color', 'metric' ], run: () => lensBar.setChannel( 'color', m.id ) } ) );
		for ( const m of lens.metrics ) if ( m.kind !== 'categorical' ) metricCommands.push( commands.register( { id: 'lens.size.' + m.id, title: `Size by ${m.label}`, group: 'Lens', icon: 'size', keywords: [ 'lens', 'metric' ], run: () => lensBar.setChannel( 'size', m.id ) } ) );

	}

	registerMetricCommands();
	on( 'graph', registerMetricCommands );
	commands.register( { id: 'explore.clear', title: 'Clear the highlight', group: 'Explore', icon: 'off', shortcut: 'esc', keywords: 'deselect reset selection path neighbourhood', enabled: () => interaction.highlight.active || interaction.pathPickFrom >= 0, run: () => interaction.clearAll() } );
	for ( const k of [ 1, 2, 3 ] ) commands.register( { id: 'explore.hops.' + k, title: `Neighbourhood within ${k} ${k === 1 ? 'hop' : 'hops'}`, group: 'Selection', icon: 'hops', shortcut: String( k ), keywords: 'khop neighbours ego', enabled: hasSelection, run: () => interaction.showNeighborhood( k ) } );
	commands.register( { id: 'explore.hops.0', title: 'Neighbourhood off', group: 'Selection', icon: 'hops', shortcut: '0', enabled: () => interaction.neighborhoodHops > 0, run: () => interaction.showNeighborhood( 0 ) } );
	commands.register( { id: 'explore.path', title: 'Shortest path from the selection…', group: 'Selection', icon: 'path', keywords: 'route distance geodesic', hint: 'then pick the target', enabled: hasSelection, run: () => {

		interaction.pickPathTarget();
		search.open();

	} } );
	commands.register( { id: 'explore.focus', title: 'Fly to the selection', group: 'Selection', icon: 'focus', keywords: 'zoom camera centre center', enabled: hasSelection, run: () => rig.focusNode( interaction.selectedIndex ) } );
	commands.register( { id: 'explore.pin', title: 'Pin or unpin the selection', group: 'Selection', icon: 'pin', keywords: 'fix freeze', enabled: hasSelection, run: () => interaction.togglePin( interaction.selectedIndex ) } );
	commands.register( { id: 'explore.ripple', title: 'Send a ripple from the selection', group: 'Selection', icon: 'ripple', keywords: 'wave spread distance', enabled: hasSelection, run: () => tools.ripple( interaction.selectedIndex ) } );
	commands.register( { id: 'explore.radial', title: 'Radial layout from the selection', group: 'Selection', icon: 'radial', keywords: 'rings shells distance tree', enabled: () => hasSelection() && !! layout?.setLayoutMode, run: () => {

		// the shell hears the 'layout' event and writes the mode into its settings (panel, share link)
		layout.setLayoutMode( 'radial', { root: interaction.selectedIndex } );
		rig.keepFramed?.( 4 );

	} } );
	commands.register( { id: 'explore.fit', title: 'Fit the graph to the view', group: 'View', icon: 'fit', shortcut: 'F', keywords: 'overview zoom all reset camera', run: () => rig.fitToGraph() } );

	// ── insets: what the surfaces cover, and a camera that centres the graph in what is left ──
	function insets() {

		const out = { left: 0, right: 0, top: 0, bottom: 0 };
		if ( typeof window === 'undefined' ) return out;
		const shown = ( slot ) => Array.from( slot.children ).filter( ( el ) => ! el.hidden && el.offsetParent !== null );
		const phone = isPhone();
		if ( ! phone && shown( slots.right ).length ) out.right = Math.max( 0, Math.round( window.innerWidth - slots.right.getBoundingClientRect().left ) );
		// the top row: the search box at rest (its open list is an overlay and does not count)
		for ( const el of shown( slots.top ) ) out.top = Math.max( out.top, Math.round( ( el.querySelector( '.wgx-search-box' ) || el ).getBoundingClientRect().bottom ) );
		// on a phone the card and the drawer are sheets above the bars: they count from the bottom
		let top = Infinity;
		for ( const el of [ ...shown( slots.bottom ), ...( phone ? shown( slots.right ) : [] ) ] ) top = Math.min( top, el.getBoundingClientRect().top );
		if ( Number.isFinite( top ) ) out.bottom = Math.max( 0, Math.round( window.innerHeight - top ) );
		return out;

	}

	// The rig centres and fits the graph in the part of the canvas that no surface covers: the
	// larger of the panel's inset and this layer's on every side (both are measured from the
	// window edge). Whoever owns the rig may take this over: installExplore( { cameraInsets: false } ).
	const manageCamera = ctx.cameraInsets !== false && !! rig && typeof rig.setInsets === 'function';
	function applyCameraInsets( own, animate = true ) {

		if ( ! manageCamera ) return;
		const p = panel && typeof panel.insets === 'function' ? panel.insets() || {} : {};
		const side = ( k ) => Math.max( + p[ k ] || 0, own[ k ] || 0 );
		rig.setInsets( { left: side( 'left' ), right: side( 'right' ), top: side( 'top' ), bottom: side( 'bottom' ) }, { animate } );

	}

	let insetKey = '', insetFrame = 0;
	const checkInsets = () => {

		if ( insetFrame || typeof requestAnimationFrame !== 'function' ) return;
		insetFrame = requestAnimationFrame( () => {

			insetFrame = 0;
			const r = insets(), key = `${r.left}|${r.right}|${r.top}|${r.bottom}`;
			// always: the panel may have handed the rig its own insets in the meantime
			applyCameraInsets( r );
			if ( key === insetKey ) return;
			insetKey = key;
			app.events.dispatchEvent( new CustomEvent( 'exploreinsets', { detail: r } ) );

		} );

	};

	for ( const type of [ 'select', 'card', 'insights', 'path', 'pathpick', 'nopath', 'tour', 'graph', 'photo' ] ) on( type, checkInsets );
	on( 'lens', ( e ) => {

		if ( e.detail.reason !== 'progress' ) checkInsets();

	} );
	window.addEventListener( 'resize', checkInsets );
	// the panel publishes its insets as CSS variables on <html> ( slots.setInsets ) and hides the
	// surfaces with a class there (photo mode): both change what is covered
	const rootObserver = typeof MutationObserver === 'function' ? new MutationObserver( checkInsets ) : null;
	rootObserver?.observe( document.documentElement, { attributes: true, attributeFilter: [ 'style', 'class' ] } );
	disposers.push( () => {

		window.removeEventListener( 'resize', checkInsets );
		rootObserver?.disconnect();
		if ( insetFrame ) cancelAnimationFrame( insetFrame );

	} );
	insetKey = ( ( r ) => `${r.left}|${r.right}|${r.top}|${r.bottom}` )( insets() );
	applyCameraInsets( insets(), false );

	// ── state ─────────────────────────────────────────────────────────────────────────────
	function state() {

		const path = tools.state.path;
		return {
			v: STATE_VERSION,
			lens: lensBar.state(),
			selection: interaction.selectedIndex >= 0 ? interaction.selectedIndex : null,
			hops: interaction.neighborhoodHops || 0,
			path: path ? { from: path.source, to: path.target, all: !! path.all } : null,
			cluster: interaction.clusterFocus >= 0 ? interaction.clusterFocus : null,
			insights: insights.isOpen,
			top: insights.topMetric,
		};

	}

	function restore( raw ) {

		const g = app.graph;
		const s = normalizeState( raw, {
			nodes: g ? g.N : 0, clusters: g ? g.C - 1 : 0,
			metrics: lens.metrics.map( ( m ) => m.id ), edgeMetrics: lens.edgeMetrics.map( ( m ) => m.id ), ramps: lens.ramps.map( ( r ) => r.id ),
		} );
		tour.stop();
		const done = lensBar.restore( s.lens );
		insights.setTopMetric( s.top );
		insights.toggle( s.insights );
		if ( g ) {

			restoring = true; // the view that is restored has its own camera: a path does not reframe it
			try {

				interaction.clearAll();
				if ( s.cluster !== null ) interaction.focusCluster( s.cluster );
				if ( s.selection !== null ) interaction.select( s.selection );
				if ( s.hops > 0 ) interaction.showNeighborhood( s.hops );
				if ( s.path ) interaction.showPath( s.path.from, s.path.to, { all: s.path.all } );

			} finally {

				restoring = false;

			}

		}

		return done.then( () => state() );

	}

	const shortcuts = () => [
		{ keys: '/', title: 'Search nodes and commands', group: 'Explore' },
		{ keys: mac ? '⌘ K' : 'Ctrl K', title: 'Search nodes and commands', group: 'Explore' },
		{ keys: 'I', title: 'Insights: the graph in numbers', group: 'Explore' },
		{ keys: 'Click', title: 'Select a node (a ripple shows its distances)', group: 'Explore' },
		{ keys: 'Shift Click', title: 'Shortest path from the selection to that node', group: 'Explore' },
		{ keys: '1 2 3', title: 'Neighbourhood of the selection within 1 to 3 hops', group: 'Explore' },
		{ keys: '0', title: 'Neighbourhood off', group: 'Explore' },
		{ keys: '[ ]', title: 'Step to the previous / next neighbour or search result', group: 'Explore' },
		{ keys: 'C', title: 'Hold: compare the metric colours with the communities', group: 'Explore' },
		{ keys: 'Esc', title: 'Leave one thing: path, neighbourhood, community focus, selection', group: 'Explore' },
	];

	const explore = {
		commands,
		get shortcuts() {

			return shortcuts();

		},
		search: {
			open: ( text ) => search.open( text ), close: () => search.close(), toggle: () => search.toggle(),
			get isOpen() {

				return search.isOpen;

			},
			get stats() {

				return search.stats;

			},
		},
		insights: {
			toggle: ( onNow ) => insights.toggle( onNow ), setTopMetric: ( id ) => insights.setTopMetric( id ),
			get isOpen() {

				return insights.isOpen;

			},
			get topMetric() {

				return insights.topMetric;

			},
		},
		tour: {
			start: () => tour.start(), stop: () => tour.stop(), next: () => tour.next(), previous: () => tour.previous(),
			get running() {

				return tour.running;

			},
			get step() {

				return tour.step;

			},
			get stops() {

				return tour.stops;

			},
		},
		lens: {
			presets: LENS_PRESETS,
			applyPreset: ( id ) => lensBar.applyPreset( id ),
			setChannel: ( channel, metric, opts ) => lensBar.setChannel( channel, metric, opts ),
			setEdgeMetric: ( id, opts ) => lensBar.setEdgeMetric( id, opts ),
			compare,
			get edge() {

				const e = lensBar.edge;
				return { supported: e.supported, metric: e.metric, opacity: e.opacity, width: e.width };

			},
		},
		focusCommunity,
		insets,
		communities: () => insights.communities(),
		state, restore,
		surfaces: { search, lensBar, pathStrip, insights, tour, card },
		dispose() {

			for ( const d of disposers.splice( 0 ) ) d();
			compare( false );
			tour.dispose();
			insights.dispose();
			pathStrip.dispose();
			lensBar.dispose();
			search.dispose();

		},
	};
	return explore;

}
