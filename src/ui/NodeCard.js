// Card for the selected node (App module, order 41), version 2: what the node is, how it ranks,
// where it sits, who its neighbours are, and what can be done from it.
//
//  · Head: community colour, label, degree, index, pinned badge.
//  · Two or three sentences about the node ( explain.js ): how many communities it links, what
//    its removal would cut off, how tightly its neighbours are linked to each other.
//  · Community path, coarsest level first: one chip per level, named after the best-connected
//    member of that community; a click focuses the community (when the explore layer is attached).
//  · Metrics with their rank ( lens.describe( i ): "PageRank 0.031 · #3 of 77" and a bar for the
//    rank position). The cheap metrics are requested when the card opens ( lens.load() ); slow
//    ones appear when something else has computed them and are marked "≈" while they are
//    estimates. Node attributes of the file follow ( lens.attributes( i ) ).
//  · Neighbours by degree as colour chips: a click walks to that node ( interaction.select and a
//    pan ), the chip under the pointer is previewed in the graph.
//  · Actions: Focus, Pin, Ripple; Neighbourhood within 1 / 2 / 3 hops; Path from here (then the
//    target is picked by a click, a tap or the search); Radial layout from here
//    ( layout.setLayoutMode( 'radial', { root } ) ) with the way back. On touch these cover what
//    Shift+click and the number keys do. On a phone the card is a sheet: the neighbours come
//    first, the actions fold into two rows with short labels. While a path target is being
//    picked and while a path is shown the sheet folds down to its head (name, degree, close), so
//    the graph the path runs through stays in view; a chevron in the head unfolds it, and it
//    comes back by itself when the path is closed.
//  · Keyboard: walking to a neighbour keeps the focus in the neighbour list (on its first chip),
//    so Enter walks on.
//  · The card can open under the pointer that selected the node. For the length of a double
//    click it lets presses through to the graph, and it only takes a click whose press began
//    on it (the release of a long press over the fresh card is not a click on a neighbour).
//
// Wiring: main.js only does  app.use( new NodeCard( { interaction, rig } ) )  — the card listens to
// 'select' / 'pin' / 'graph' itself and floats at the bottom right. The explore layer then calls
//   card.attach( { lens, tools, layout, slots, settings, onFocusCluster } )
// which moves it to the end of slots.right (class wgv-slot-end) and switches on the metrics and
// the actions that need those modules. Callbacks default to interaction.select / rig.focusNode /
// interaction.togglePin and can be overridden: { onSelect( i ), onFocus( i ), onTogglePin( i ),
// isPinned( i ), onClose() }.
//
// API
//   card.show( index ) / card.hide() / card.index / card.visible
//   card.setQuiet( on )         stay hidden for now (the tour is talking, 'tour' event) and come
//                               back for the selection afterwards
//   card.setPalette( name )     community tints follow the active palette
//   card.maxNeighbors           how many neighbours to list before "+n more" (default 12)
//   card.attach( modules )      see above;  card.refresh()  re-reads everything for the shown node
//   card.folded / card.setFolded( on )   phones: only the head is shown ('card' event { index, folded })
//   card.stats                  { ms } time the last show() took

import '../explore/explore.css';
import { h, icon, paletteCss } from './Panel.js';
import { xicon, iconEl } from '../explore/ui.js';
import { formatCount, formatMetric, formatRank } from '../explore/format.js';
import { labelOf } from '../explore/communities.js';
import { explainNode } from '../explore/explain.js';

const METRIC_ORDER = [ 'pagerank', 'degree', 'betweenness', 'closeness', 'core', 'clustering', 'eigenvector', 'harmonic', 'triangles', 'relevance' ];
const METRICS_SHOWN = 5;

export class NodeCard {

	constructor( options = {} ) {

		this.order = 41;
		this.options = { container: document.body, palette: 'paper', maxNeighbors: 12, ...options };
		this.interaction = options.interaction || null;
		this.rig = options.rig || null;
		this.lens = options.lens || null;
		this.tools = options.tools || null;
		this.layout = options.layout || null;
		this.settings = options.settings || null;
		this.onFocusCluster = options.onFocusCluster || null;
		this.palette = this.options.palette;
		this.maxNeighbors = this.options.maxNeighbors;
		this.index = - 1;
		this.visible = false;
		this.graph = null;
		this.stats = { ms: 0 };
		this.quiet = false;
		this.folded = false;
		this._pathBusy = false; // a path target is being picked, or a path is shown
		this._allMetrics = false;
		this._allNeighbors = false;
		this._hubs = new Map(); // cluster -> best-connected member
		this._build();

	}

	init( app ) {

		this.app = app;
		this._onSelect = ( e ) => ( e.detail.index >= 0 ? this.show( e.detail.index ) : this.hide() );
		this._onPin = ( e ) => {

			if ( this.index >= 0 && ( e.detail.index === this.index || e.detail.index < 0 ) ) this._syncPin();

		};

		this._onState = () => this._syncActions();
		this._onLens = ( e ) => {

			if ( this.index < 0 || e.detail.reason === 'progress' ) return;
			if ( e.detail.palette !== this.palette ) {

				this.palette = e.detail.palette;
				this.show( this.index );

			} else this._renderMetrics();

		};

		this._onTour = ( e ) => this.setQuiet( e.detail.running );
		// a path in the making or on screen: the sheet of a phone makes room for the graph
		this._picking = false;
		this._pathOn = false;
		this._onPath = ( e ) => {

			if ( e.type === 'path' ) this._pathOn = !! e.detail;
			else this._picking = !! ( e.detail && e.detail.active );
			const busy = this._picking || this._pathOn;
			if ( busy === this._pathBusy ) return;
			this._pathBusy = busy;
			this.setFolded( busy );

		};

		const ev = app.events;
		ev.addEventListener( 'path', this._onPath );
		ev.addEventListener( 'pathpick', this._onPath );
		ev.addEventListener( 'tour', this._onTour );
		ev.addEventListener( 'select', this._onSelect );
		ev.addEventListener( 'pin', this._onPin );
		for ( const type of [ 'khop', 'pathpick', 'layout', 'cluster' ] ) ev.addEventListener( type, this._onState );
		ev.addEventListener( 'lens', this._onLens );

	}

	setGraph( buffers, graph ) {

		this.graph = graph;
		this._hubs.clear();
		if ( graph && graph.palette && ! this.lens ) this.palette = graph.palette;
		this._picking = this._pathOn = this._pathBusy = false;
		this.setFolded( false );
		this.hide();

	}

	dispose() {

		const ev = this.app?.events;
		if ( ev ) {

			ev.removeEventListener( 'path', this._onPath );
			ev.removeEventListener( 'pathpick', this._onPath );
			ev.removeEventListener( 'tour', this._onTour );
			ev.removeEventListener( 'select', this._onSelect );
			ev.removeEventListener( 'pin', this._onPin );
			for ( const type of [ 'khop', 'pathpick', 'layout', 'cluster' ] ) ev.removeEventListener( type, this._onState );
			ev.removeEventListener( 'lens', this._onLens );

		}

		window.removeEventListener( 'pointerdown', this._onPress, true );
		clearTimeout( this._arriving );
		document.documentElement.classList.remove( 'wgx-card-open' );
		this.el.remove();

	}

	/** Explore wiring: the modules the metrics and actions need, and the slot the card lives in. */
	attach( { lens, tools, layout, slots, settings, onFocusCluster } = {} ) {

		if ( lens ) this.lens = lens;
		if ( tools ) this.tools = tools;
		if ( layout ) this.layout = layout;
		if ( settings ) this.settings = settings;
		if ( onFocusCluster ) this.onFocusCluster = onFocusCluster;
		if ( this.lens && this.lens.state ) this.palette = this.lens.state.palette || this.palette;
		if ( slots && slots.right ) {

			this.el.classList.remove( 'is-floating' );
			this.el.classList.add( 'wgv-slot-end' );
			slots.right.append( this.el );

		}

		this._syncCapabilities();
		if ( this.index >= 0 ) this.show( this.index );

	}

	setPalette( name ) {

		this.palette = name;
		if ( this.index >= 0 ) this.show( this.index );

	}

	label( i ) {

		return labelOf( this.graph, i );

	}

	refresh() {

		if ( this.index >= 0 ) this.show( this.index );

	}

	show( i ) {

		const g = this.graph;
		if ( ! g || i < 0 || i >= g.N ) return this.hide();
		const t0 = performance.now();
		const changed = i !== this.index || ! this.visible;
		this.index = i;
		if ( this.quiet ) return; // remembered, shown when the quiet time is over
		if ( changed ) this._allNeighbors = false;
		const css = paletteCss( this.palette, g.leafRank[ i ] );
		this._swatch.style.setProperty( '--wgx-dot', css );
		const label = this.label( i );
		this._title.textContent = label;
		this._title.title = label;
		this._degree.textContent = formatCount( g.degree[ i ] );
		this._id.textContent = `#${i}`;
		if ( changed ) this._renderExplain();
		this._renderPath();
		this._renderMetrics();
		this._renderNeighbors();
		this._syncPin();
		this._syncActions();
		if ( ! this.visible ) {

			this.visible = true;
			this.el.hidden = false;
			this.el.removeAttribute( 'inert' );
			document.documentElement.classList.add( 'wgx-card-open' );
			// the card can open under the pointer that selected the node: for the length of a double
			// click it lets presses through, so the second click still reaches the graph
			this.el.classList.add( 'is-arriving' );
			clearTimeout( this._arriving );
			this._arriving = setTimeout( () => this.el.classList.remove( 'is-arriving' ), 340 );

		}

		if ( changed ) {

			if ( ! this._list.contains( document.activeElement ) ) this._scroll.scrollTop = 0; // (a focused chip stays in view)
			// the cheap metrics, so the ranks are there the moment the card is read
			const lens = this.lens;
			if ( lens && lens.load ) lens.load().then( () => {

				if ( this.index === i && this.graph === g ) this._renderMetrics();

			}, () => {} );

		}

		this.stats.ms = performance.now() - t0;

	}

	/**
	 * Folded, the sheet of a phone shows only its head (the wide layout ignores it). The card folds
	 * by itself while a path is picked or shown; the chevron in the head calls this too.
	 */
	setFolded( on ) {

		on = !! on;
		if ( on === this.folded ) return;
		this.folded = on;
		this.el.classList.toggle( 'is-folded', on );
		this._foldBtn.setAttribute( 'aria-expanded', String( ! on ) );
		this._foldBtn.setAttribute( 'aria-label', on ? 'Show the details' : 'Hide the details' );
		this._foldBtn.innerHTML = xicon( on ? 'expand' : 'collapse', 14 );
		this.app?.events.dispatchEvent( new CustomEvent( 'card', { detail: { index: this.index, folded: on } } ) );

	}

	/** While quiet (the tour is talking) the card stays away; the node it would show is remembered. */
	setQuiet( on ) {

		on = !! on;
		if ( on === this.quiet ) return;
		this.quiet = on;
		const i = this.index;
		if ( on ) {

			this.hide();
			this.index = i;

		} else if ( i >= 0 && this.interaction && this.interaction.selectedIndex === i ) this.show( i );
		else this.index = - 1;

	}

	hide() {

		this.index = - 1;
		if ( ! this.visible ) return;
		this.visible = false;
		this.el.hidden = true;
		this.el.setAttribute( 'inert', '' );
		clearTimeout( this._arriving );
		this.el.classList.remove( 'is-arriving' );
		document.documentElement.classList.remove( 'wgx-card-open' );

	}

	// ── sections ──────────────────────────────────────────────────────────────────────────

	/** Best-connected member of each cluster on the node's path (one pass over the nodes). */
	_pathHubs( i ) {

		const g = this.graph, D = g.D, anc = g.anc, deg = g.degree;
		const mine = [];
		let missing = false;
		for ( let l = 0; l < D; l ++ ) {

			mine.push( anc[ i * D + l ] );
			if ( ! this._hubs.has( mine[ l ] ) ) missing = true;

		}

		if ( missing ) {

			const best = new Int32Array( D ).fill( - 1 );
			for ( let v = 0; v < g.N; v ++ ) {

				for ( let l = 0; l < D; l ++ ) {

					if ( anc[ v * D + l ] === mine[ l ] && ( best[ l ] < 0 || deg[ v ] > deg[ best[ l ] ] ) ) best[ l ] = v;

				}

			}

			for ( let l = 0; l < D; l ++ ) this._hubs.set( mine[ l ], best[ l ] );

		}

		return mine.map( ( c ) => this._hubs.get( c ) );

	}

	/** Two or three sentences about the node ( explain.js ): communities linked, what depends on it. */
	_renderExplain() {

		const { sentences } = explainNode( this.graph, this.index );
		this._explain.textContent = sentences.join( ' ' );
		this._explainRow.hidden = sentences.length === 0;

	}

	_renderPath() {

		const g = this.graph, i = this.index, D = g.D;
		this._path.textContent = '';
		this._pathRow.hidden = D === 0;
		if ( D === 0 ) return;
		const hubs = this._pathHubs( i );
		const focus = this.interaction ? this.interaction.clusterFocus : - 1;
		let lastMass = - 1;
		for ( let l = D - 1; l >= 0; l -- ) {

			const c = g.anc[ i * D + l ];
			if ( g.clusterMass[ c ] === lastMass ) continue; // a level that does not split its parent says nothing new
			if ( lastMass >= 0 ) this._path.append( h( 'span.wgx-crumb-sep', { 'aria-hidden': 'true', html: xicon( 'next', 10 ) } ) );
			lastMass = g.clusterMass[ c ];
			const mid = ( g.clusterRankRange[ 2 * c ] + g.clusterRankRange[ 2 * c + 1 ] ) / 2;
			const name = this.label( hubs[ l ] );
			const title = `Level ${l + 1} of ${D}: ${formatCount( g.clusterMass[ c ] )} nodes around ${this.label( hubs[ l ] )}`;
			const inner = [ h( 'i.wgx-dot', { style: { '--wgx-dot': paletteCss( this.palette, mid ) } } ), h( 'span.wgx-crumb-l', null, name ), h( 'b', null, formatCount( g.clusterMass[ c ] ) ) ];
			let chip;
			if ( this.onFocusCluster ) {

				chip = h( 'button.wgx-crumb', { type: 'button', title, 'aria-pressed': String( c === focus ) }, inner );
				chip.addEventListener( 'click', () => this.onFocusCluster( c, l ) );

			} else chip = h( 'span.wgx-crumb', { title }, inner );

			chip.dataset.cluster = c;
			this._path.append( chip );

		}

	}

	_renderMetrics() {

		const g = this.graph, i = this.index, lens = this.lens;
		const box = this._metrics;
		box.textContent = '';
		const rows = lens && lens.describe && i >= 0 ? lens.describe( i ).filter( ( m ) => METRIC_ORDER.includes( m.id ) && ( m.id !== 'relevance' || lens.state.sources.length > 0 ) ) : [];
		rows.sort( ( a, b ) => METRIC_ORDER.indexOf( a.id ) - METRIC_ORDER.indexOf( b.id ) );
		const attrs = lens && lens.attributes && i >= 0 ? lens.attributes( i ).filter( ( a ) => a.text !== '' ) : [];
		this._metricRow.hidden = ! rows.length && ! attrs.length;
		if ( this._metricRow.hidden ) return;
		const shown = this._allMetrics ? rows : rows.slice( 0, METRICS_SHOWN );
		for ( const m of shown ) {

			const estimate = ! m.exact;
			const share = m.of > 1 ? 1 - ( m.rank - 1 ) / ( m.of - 1 ) : 1;
			box.append( h( 'div.wgx-metric', { title: `${m.label}: ${estimate ? 'estimate, ' : ''}rank ${formatCount( m.rank )} of ${formatCount( m.of )}` },
				h( 'span.wgx-metric-k', null, m.label ),
				h( 'span.wgx-metric-v', null, ( estimate ? '≈ ' : '' ) + formatMetric( m.id, m.value, g.N ) ),
				h( 'span.wgx-metric-r', null, formatRank( m.rank, m.of ) ),
				h( 'span.wgx-rank-bar', { 'aria-hidden': 'true' }, h( 'i', { style: { width: ( 100 * share ).toFixed( 1 ) + '%' } } ) ) ) );

		}

		if ( rows.length > METRICS_SHOWN ) {

			const more = h( 'button.wgx-link', { type: 'button' }, this._allMetrics ? 'Fewer' : `All ${rows.length} metrics` );
			more.addEventListener( 'click', () => {

				this._allMetrics = ! this._allMetrics;
				this._renderMetrics();
				box.querySelector( 'button.wgx-link' )?.focus( { preventScroll: true } ); // the link was rebuilt

			} );
			box.append( more );

		}

		for ( const a of attrs ) box.append( h( 'div.wgx-metric.is-attr', null, h( 'span.wgx-metric-k', null, a.name ), h( 'span.wgx-metric-v', null, a.type === 'number' ? formatMetric( a.id, a.value, g.N ) : a.text ) ) );

	}

	/** focusAt: the chip that takes the keyboard focus afterwards (−1: the first, if the list had it). */
	_renderNeighbors( focusAt = - 1 ) {

		const g = this.graph, i = this.index;
		const s = g.adjStart[ i ], n = g.degree[ i ];
		this._nbCount.textContent = formatCount( n );
		// walking by keyboard: the chip that was activated is about to go, the focus stays in the list
		if ( focusAt < 0 && this._list.contains( document.activeElement ) ) focusAt = 0;
		this._list.textContent = '';
		this._nbRow.hidden = n === 0;
		if ( n === 0 ) return;
		const nb = Array.from( g.adjList.subarray( s, s + n ) );
		nb.sort( ( a, b ) => g.degree[ b ] - g.degree[ a ] || a - b );
		const limit = this._allNeighbors ? 240 : this.maxNeighbors;
		const shown = nb.slice( 0, limit );
		for ( const j of shown ) {

			const label = this.label( j );
			const b = h( 'button.wgx-nb', { type: 'button', title: `${label} · degree ${formatCount( g.degree[ j ] )}` },
				h( 'i.wgx-dot', { style: { '--wgx-dot': paletteCss( this.palette, g.leafRank[ j ] ) } } ),
				h( 'span.wgx-nb-l', null, label ), h( 'span.wgx-nb-d', null, formatCount( g.degree[ j ] ) ) );
			b.addEventListener( 'click', () => this._select( j ) );
			b.addEventListener( 'pointerenter', () => this.interaction?.setPreview?.( j ) );
			b.addEventListener( 'pointerleave', () => this.interaction?.setPreview?.( - 1 ) );
			b.addEventListener( 'focus', () => this.interaction?.setPreview?.( j ) );
			b.addEventListener( 'blur', () => this.interaction?.setPreview?.( - 1 ) );
			this._list.append( b );

		}

		if ( nb.length > shown.length ) {

			const rest = nb.length - shown.length;
			const more = h( 'button.wgx-nb.wgx-nb-more', { type: 'button' }, this._allNeighbors ? `${formatCount( rest )} more not listed` : `+${formatCount( rest )} more` );
			if ( this._allNeighbors ) more.disabled = true;
			more.addEventListener( 'click', () => {

				this._allNeighbors = true;
				this._renderNeighbors( shown.length ); // on to the first chip that is new

			} );
			this._list.append( more );

		}

		this._list.scrollTop = 0;
		this._list.scrollLeft = 0;
		const chip = focusAt >= 0 ? this._list.children[ Math.min( focusAt, this._list.children.length - 1 ) ] : null;
		if ( chip && ! chip.disabled ) chip.focus(); // scrolls it into view where the list scrolls

	}

	// ── actions ───────────────────────────────────────────────────────────────────────────

	_select( j ) {

		this.interaction?.setPreview?.( - 1 );
		if ( this.options.onSelect ) this.options.onSelect( j );
		else if ( this.interaction ) {

			if ( this.interaction.walk ) this.interaction.walk( j );
			else this.interaction.select( j );

		} else this.show( j );

	}

	_isPinned( i ) {

		if ( this.options.isPinned ) return this.options.isPinned( i );
		return this.interaction ? this.interaction.isPinned( i ) : false;

	}

	_syncPin() {

		const on = this.index >= 0 && this._isPinned( this.index );
		this._pinBtn.setAttribute( 'aria-pressed', String( on ) );
		this._pinBtn._setLabel( on ? 'Unpin' : 'Pin' );
		this._pinBadge.hidden = ! on;

	}

	/** Which actions exist depends on the modules at hand (tools, layout). */
	_syncCapabilities() {

		const I = this.interaction;
		const tools = !! this.tools || !! ( I && I.tools );
		this._rippleBtn.hidden = ! tools;
		this._hopRow.hidden = ! ( tools && I && I.showNeighborhood );
		this._pathBtn.hidden = ! ( tools && I && I.pickPathTarget );
		this._radialBtn.hidden = ! ( this.layout && this.layout.setLayoutMode );
		this._moreRow.hidden = this._pathBtn.hidden && this._radialBtn.hidden;

	}

	_syncActions() {

		const I = this.interaction, i = this.index;
		if ( i < 0 ) return;
		const hops = I ? I.neighborhoodHops || 0 : 0;
		this._hopBtns.forEach( ( b, k ) => b.setAttribute( 'aria-pressed', String( hops === k + 1 ) ) );
		const picking = !! I && I.pathPickFrom === i;
		this._pathBtn.setAttribute( 'aria-pressed', String( picking ) );
		this._pathBtn._setLabel( picking ? 'Pick the target…' : 'Path from here', picking ? 'Pick…' : 'Path' );
		const L = this.layout;
		const radial = !! L && L.layoutMode === 'radial';
		const here = radial && L.radialRoot === i;
		this._radialBtn.setAttribute( 'aria-pressed', String( here ) );
		this._radialBtn._setLabel( here ? 'Leave radial' : 'Radial from here', here ? 'Leave' : 'Radial' );
		const focus = I ? I.clusterFocus : - 1;
		for ( const chip of this._path.querySelectorAll( 'button.wgx-crumb' ) ) chip.setAttribute( 'aria-pressed', String( + chip.dataset.cluster === focus ) );

	}

	_radial() {

		const L = this.layout, i = this.index;
		if ( ! L || i < 0 ) return;
		if ( L.layoutMode === 'radial' && L.radialRoot === i ) {

			// the way back: the mode the settings name (never radial itself)
			const back = this._beforeRadial && this._beforeRadial !== 'radial' ? this._beforeRadial : 'force';
			L.setLayoutMode( back );

		} else {

			if ( L.layoutMode !== 'radial' ) this._beforeRadial = ( this.settings && this.settings.layoutMode ) || L.layoutMode || 'force';
			L.setLayoutMode( 'radial', { root: i } );

		}

		this.rig?.keepFramed?.( 4 );
		this._syncActions();

	}

	_build() {

		// a label has a short form for the narrow action row of a phone
		const btn = ( name, label, onClick, extra = {}, short = label ) => {

			const long = h( 'span.wgx-lbl-long', null, label ), brief = h( 'span.wgx-lbl-short', null, short );
			const b = h( 'button.wgx-btn', { type: 'button', 'data-act': name, ...extra }, iconEl( name ), h( 'span.wgx-lbl', null, long, brief ) );
			b.addEventListener( 'click', () => {

				if ( this.index >= 0 ) onClick( this.index );

			} );
			b._setLabel = ( text, shortText = text ) => {

				long.textContent = text;
				brief.textContent = shortText;

			};

			return b;

		};

		this._swatch = h( 'i.wgx-dot.wgx-card-dot' );
		this._title = h( 'h3.wgx-card-title' );
		this._degree = h( 'b' );
		this._id = h( 'span.wgx-card-id' );
		this._pinBadge = h( 'span.wgx-badge', { hidden: true }, 'pinned' );
		this._path = h( 'div.wgx-crumbs' );
		this._pathRow = h( 'div.wgx-card-row', null, h( 'span.wgx-k', null, 'Community' ), this._path );
		this._metrics = h( 'div.wgx-metrics' );
		this._metricRow = h( 'div.wgx-card-block', { hidden: true }, this._metrics );
		this._nbCount = h( 'span' );
		this._list = h( 'div.wgx-nbs' );
		this._nbRow = h( 'div.wgx-card-block.wgx-card-nb', null, h( 'div.wgx-k', null, 'Neighbours ', this._nbCount ), this._list );

		const focus = btn( 'focus', 'Focus', ( i ) => ( this.options.onFocus ? this.options.onFocus( i ) : this.rig?.focusNode( i ) ) );
		this._pinBtn = btn( 'pin', 'Pin', ( i ) => {

			if ( this.options.onTogglePin ) this.options.onTogglePin( i );
			else this.interaction?.togglePin( i );
			this._syncPin();

		}, { 'aria-pressed': 'false' } );
		this._rippleBtn = btn( 'ripple', 'Ripple', ( i ) => ( this.tools || this.interaction?.tools )?.ripple( i ) );
		this._hopBtns = [ 1, 2, 3 ].map( ( k ) => {

			const b = h( 'button', { type: 'button', 'aria-pressed': 'false', 'aria-label': `Neighbourhood within ${k} ${k === 1 ? 'hop' : 'hops'}`, title: `Within ${k} ${k === 1 ? 'hop' : 'hops'} (key ${k})` }, String( k ) );
			b.addEventListener( 'click', () => {

				const I = this.interaction;
				if ( I && this.index >= 0 ) I.showNeighborhood( I.neighborhoodHops === k ? 0 : k );
				this._syncActions();

			} );
			return b;

		} );
		this._hopRow = h( 'div.wgx-card-hops', null, h( 'span.wgx-card-hops-k', null, iconEl( 'hops', 14 ), 'Neighbourhood' ), h( 'div.wgx-seg', { role: 'group', 'aria-label': 'Neighbourhood, hops' }, this._hopBtns ) );
		this._pathBtn = btn( 'path', 'Path from here', ( i ) => {

			const I = this.interaction;
			if ( ! I ) return;
			if ( I.pathPickFrom === i ) I.cancelPathPick();
			else I.pickPathTarget( i );
			this._syncActions();

		}, { 'aria-pressed': 'false', title: 'Then click a node, or search for it (Shift+click does the same from the selection)' }, 'Path' );
		this._radialBtn = btn( 'radial', 'Radial from here', () => this._radial(), { 'aria-pressed': 'false', title: 'Arrange the graph in rings by distance from this node' }, 'Radial' );
		this._moreRow = h( 'div.wgx-card-actions', null, this._pathBtn, this._radialBtn );

		// phones: fold the sheet down to its head and back (hidden in the wide layout)
		this._foldBtn = h( 'button.wgx-icon-btn.wgx-card-fold', { type: 'button', 'aria-label': 'Hide the details', 'aria-expanded': 'true', html: xicon( 'collapse', 14 ) } );
		this._foldBtn.addEventListener( 'click', () => this.setFolded( ! this.folded ) );
		const close = h( 'button.wgx-icon-btn.wgx-card-x', { type: 'button', 'aria-label': 'Clear selection', html: icon( 'close' ) } );
		close.addEventListener( 'click', () => {

			if ( this.options.onClose ) this.options.onClose();
			else if ( this.interaction ) this.interaction.clearSelection();
			else this.hide();

		} );
		this._explain = h( 'p.wgx-card-explain' );
		this._explainRow = h( 'div.wgx-card-block', { hidden: true }, this._explain );
		this._scroll = h( 'div.wgx-card-scroll', null, this._explainRow, this._pathRow, this._metricRow, this._nbRow );
		this.el = h( 'div.wgx.wgx-card.wgv-glass.is-floating', { role: 'region', 'aria-label': 'Selected node', inert: true, hidden: true },
			h( 'div.wgx-card-head', null, this._swatch,
				h( 'div.wgx-card-titles', null, this._title,
					h( 'div.wgx-card-meta', null, h( 'span', null, 'degree ', this._degree ), this._id, this._pinBadge ) ),
				this._foldBtn, close ),
			this._scroll,
			h( 'div.wgx-card-foot', null,
				h( 'div.wgx-card-actions', null, focus, this._pinBtn, this._rippleBtn ),
				this._hopRow, this._moreRow ) );
		// A click belongs to the card only when its press began on it. The card can open under a
		// finger that is still down on the graph (a long press, the second tap of a double tap), and
		// the click a browser sends for that release is not meant for what has appeared beneath it.
		// (detail 0: the keyboard, or a click made by code)
		this._pressedHere = false;
		this._onPress = ( e ) => {

			this._pressedHere = this.el.contains( e.target );

		};

		window.addEventListener( 'pointerdown', this._onPress, true );
		this.el.addEventListener( 'click', ( e ) => {

			if ( this._pressedHere || e.detail === 0 ) return;
			e.stopPropagation();
			e.preventDefault();

		}, true );
		this.options.container.append( this.el );
		this._syncCapabilities();

	}

}
