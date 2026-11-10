// Lens bar (slots.bottom): the controls and the legend of the metric lens.
//
//  · Presets: one-click chips that set several channels ( presets.js ).
//  · Channels: Colour, Size, Glow and, when the edge renderer can show it ( edges.setEdgeMetric,
//    feature-detected ), Edges. Each opens a compact menu of metrics with a one-line description
//    and a cost badge (instant / fast / slow); node attributes of the file appear as a group of
//    their own ( lens.metrics ). The colour menu also holds the ramp and its reversal, the glow
//    menu the share of nodes that glow, the edge menu what the metric drives (opacity, width).
//  · Legend: what the picture means, read back from the lens ( lens.legend() ): the gradient with
//    the values at its ends and its middle in real units and the scale type; swatches with counts
//    for categories; the size range; the number of glowing nodes and the value they start at;
//    how many nodes have no value of a file attribute (drawn grey, or smallest). The community
//    palette counts the communities at the level the insights drawer and the tour use
//    ( communitySummary ), with the finest level as a second number when that is another.
//    While a slow metric refines, a quiet bar shows pivots done / total and the numbers carry an
//    "estimate" mark until they are exact.
//  · The edge channel asks the lens for a per-edge channel ( lens.edgeChannel ), hands every
//    refinement to edges.setEdgeMetric and releases the channel it stops showing
//    ( lens.releaseEdgeChannel ).
//
//   const bar = new LensBar( { app, lens, edges, interaction, slots } )
//   bar.applyPreset( id ), bar.setChannel( 'color' | 'size' | 'glow', metric | null )
//   bar.setEdgeMetric( id | null, { opacity?, width? } ), bar.edge -> { supported, metric, opacity, width }
//   bar.state() -> { color, size, glow, ramp, reverse, top, edge, edgeOpacity, edgeWidth }
//   bar.restore( state ) -> Promise
//   bar.refresh()   redraws (the edge channel appears once edges.setEdgeMetric exists)
//   bar.collapsed, bar.setCollapsed( on )   phones: the channels folded away (the default there);
//                                           one line then names the metrics that are on
//   bar.dispose()

import { h, xicon, iconEl, cssOfLinear, isPhone } from './ui.js';
import { formatMetric, formatCount, formatPercent, formatNumber } from './format.js';
import { LENS_PRESETS, applyPreset, matchPreset, applyChannels } from './presets.js';
import { openPopover, closePopover } from './popover.js';
import { communitySummary } from './communities.js';
import { paletteCss } from '../ui/Panel.js';

const CHANNELS = [
	{ id: 'color', label: 'Colour', icon: 'color', off: 'Communities', offNote: 'The community palette of the graph.' },
	{ id: 'size', label: 'Size', icon: 'size', off: 'Degree', offNote: 'The graph’s own sizes: larger means more links.' },
	{ id: 'glow', label: 'Glow', icon: 'glow', off: 'Off', offNote: 'No extra glow.' },
	{ id: 'edges', label: 'Edges', icon: 'edges', off: 'Off', offNote: 'Every edge drawn alike.' },
];
const GROUPS = { degree: 'Structure', core: 'Structure', clustering: 'Structure', triangles: 'Structure', component: 'Structure' };
const groupOf = ( m ) => ( m.attribute !== undefined ? 'Attributes of the file' : GROUPS[ m.id ] || 'Centrality' );
const GROUP_ORDER = [ 'Centrality', 'Structure', 'Attributes of the file' ];
const SCALE_NOTE = { linear: 'linear scale', log: 'logarithmic scale', rank: 'by rank: equal steps hold equal numbers of nodes', robust: 'linear between the 2nd and 98th percentile; the few values beyond take the end of the scale' };
/** The word shown for a scale ( lens legends call the percentile scale 'robust' ). */
const scaleWord = ( scale ) => ( scale === 'robust' ? 'outliers clipped' : scale );
const TOPS = [ 0.01, 0.05, 0.1 ];

export class LensBar {

	constructor( { app, lens, edges, interaction, slots } ) {

		this.app = app;
		this.lens = lens;
		this.edges = edges;
		this.interaction = interaction;
		this.collapsed = false;
		this.edge = { supported: !! edges && typeof edges.setEdgeMetric === 'function', metric: null, opacity: true, width: true, result: null, busy: false };
		this._ramp = null; // a ramp picked while no colour metric is on
		this._build();
		slots.bottom.append( this.el );

		this._onLens = ( e ) => {

			if ( e.detail.reason === 'progress' ) this._renderProgress();
			else this._render();

		};

		this._onGraph = () => {

			closePopover( { focus: false } );
			this.edge.result = null;
			if ( this.edge.metric ) this._requestEdges(); // the channel stays on across graphs
			this._render();

		};

		this._onSelect = () => {

			if ( this._usesRelevance() ) this._render();

		};

		app.events.addEventListener( 'lens', this._onLens );
		app.events.addEventListener( 'graph', this._onGraph );
		app.events.addEventListener( 'select', this._onSelect );
		this._render();

	}

	// ── public ────────────────────────────────────────────────────────────────────────────

	applyPreset( id ) {

		return applyPreset( this.lens, id );

	}

	setChannel( channel, metric, opts = {} ) {

		const lens = this.lens;
		metric = metric || null;
		// an id this graph does not have (an attribute of another file, a typo in a link) is ignored
		if ( metric && channel !== 'edges' && ! lens.metrics.some( ( m ) => m.id === metric ) ) return Promise.resolve( false );
		if ( channel === 'color' ) return lens.setColor( metric, metric ? { ramp: opts.ramp || this._ramp || undefined, reverse: opts.reverse ?? lens.state.reverse } : undefined );
		if ( channel === 'size' ) return lens.setSize( metric );
		if ( channel === 'glow' ) return lens.setGlow( metric, metric ? { top: opts.top || lens.state.top } : undefined );
		if ( channel === 'edges' ) return this.setEdgeMetric( metric, opts );
		return Promise.resolve( false );

	}

	setEdgeMetric( metric, { opacity, width } = {} ) {

		const e = this.edge;
		if ( opacity !== undefined ) e.opacity = !! opacity;
		if ( width !== undefined ) e.width = !! width;
		e.supported = !! this.edges && typeof this.edges.setEdgeMetric === 'function';
		if ( ! e.supported ) return Promise.resolve( false );
		metric = metric && this.lens.edgeMetrics.some( ( m ) => m.id === metric ) ? metric : null;
		if ( e.metric && e.metric !== metric ) this.lens.releaseEdgeChannel( e.metric ); // stop refining what is no longer shown
		const changed = e.metric !== metric;
		e.metric = metric;
		if ( changed ) e.result = null;
		let done = Promise.resolve( true );
		if ( ! metric ) {

			e.busy = false;
			this.edges.setEdgeMetric( null );

		} else if ( changed || ! e.result ) done = this._requestEdges();
		else this._pushEdges();

		this._render();
		return done;

	}

	state() {

		const s = this.lens.state, e = this.edge;
		return { color: s.color, size: s.size, glow: s.glow, ramp: s.ramp, reverse: s.reverse, top: s.top, edge: e.metric, edgeOpacity: e.opacity, edgeWidth: e.width };

	}

	restore( state = {} ) {

		const known = new Set( this.lens.metrics.map( ( m ) => m.id ) );
		const pick = ( id ) => ( id && known.has( id ) ? id : null );
		const ramps = this.lens.ramps.map( ( r ) => r.id );
		const jobs = [ applyChannels( this.lens, {
			color: pick( state.color ), size: pick( state.size ), glow: pick( state.glow ),
			ramp: ramps.includes( state.ramp ) ? state.ramp : undefined, reverse: !! state.reverse,
			top: state.top > 0 && state.top <= 1 ? state.top : undefined,
		} ) ];
		jobs.push( this.setEdgeMetric( state.edge || null, { opacity: state.edgeOpacity ?? true, width: state.edgeWidth ?? true } ) );
		return Promise.all( jobs );

	}

	dispose() {

		closePopover( { focus: false } );
		this.app.events.removeEventListener( 'lens', this._onLens );
		this.app.events.removeEventListener( 'graph', this._onGraph );
		this.app.events.removeEventListener( 'select', this._onSelect );
		if ( this.edge.metric ) {

			this.lens.releaseEdgeChannel( this.edge.metric );
			if ( this.edge.supported ) this.edges.setEdgeMetric( null );

		}

		this.el.remove();

	}

	// ── edge channel ──────────────────────────────────────────────────────────────────────

	_requestEdges() {

		const e = this.edge, metric = e.metric;
		if ( ! metric || ! this.app.graph ) return Promise.resolve( false );
		const take = ( r ) => {

			if ( e.metric !== metric || ! r ) return;
			e.result = r;
			e.busy = ! r.exact && ! r.final;
			this._pushEdges();
			this._render();

		};

		e.busy = true;
		return this.lens.edgeChannel( metric, { onUpdate: take } ).then( ( r ) => {

			if ( e.metric !== metric ) return false;
			e.busy = false;
			if ( r ) take( r );
			else this._render();
			return !! r;

		}, ( err ) => {

			console.error( err );
			e.busy = false;
			this._render();
			return false;

		} );

	}

	_pushEdges() {

		const e = this.edge;
		if ( ! e.supported || ! e.result ) return;
		// [ factor at 0, factor at 1 ]; [ 1, 1 ] leaves that property alone
		this.edges.setEdgeMetric( e.result.values, { opacity: e.opacity ? [ 0.25, 1.5 ] : [ 1, 1 ], width: e.width ? [ 0.7, 2 ] : [ 1, 1 ] } );

	}

	// ── DOM ───────────────────────────────────────────────────────────────────────────────

	_build() {

		this._presets = h( 'div.wgx-lens-presets', { role: 'group', 'aria-label': 'Lens presets' } );
		this._presetButtons = LENS_PRESETS.map( ( p ) => {

			const b = h( 'button.wgx-chip', { type: 'button', title: p.description, 'aria-pressed': 'false' }, p.label );
			b.addEventListener( 'click', () => this.applyPreset( p.id ) );
			this._presets.append( b );
			return b;

		} );
		this._fold = h( 'button.wgx-lens-fold', { type: 'button', 'aria-label': 'Hide the legend', 'aria-expanded': 'true', html: xicon( 'collapse', 14 ) } );
		this._fold.addEventListener( 'click', () => this.setCollapsed( ! this.collapsed ) );

		this._channels = h( 'div.wgx-lens-channels', { role: 'group', 'aria-label': 'Lens channels' } );
		this._channelButtons = {};
		for ( const c of CHANNELS ) {

			const value = h( 'span.wgx-chan-v' );
			const b = h( 'button.wgx-chan', { type: 'button', 'aria-haspopup': 'dialog', 'aria-expanded': 'false', 'data-channel': c.id },
				iconEl( c.icon, 14 ), h( 'span.wgx-chan-k', null, c.label ), value, h( 'span.wgx-chan-c', { html: xicon( 'expand', 12 ), 'aria-hidden': 'true' } ) );
			b._value = value;
			b.addEventListener( 'click', () => this._toggleMenu( c, b ) );
			this._channelButtons[ c.id ] = b;
			this._channels.append( b );

		}

		this._summary = h( 'div.wgx-lens-summary', { hidden: true } );
		this._legend = h( 'div.wgx-lens-legend', { 'aria-live': 'off' } );
		this._progress = h( 'div.wgx-lens-progress', { hidden: true } );
		this.el = h( 'div.wgx.wgx-lens.wgv-glass', { role: 'region', 'aria-label': 'Metric lens' },
			h( 'div.wgx-lens-top', null, this._presets, this._fold ), this._channels, this._summary, this._legend, this._progress );
		this.setCollapsed( isPhone() ); // a phone starts with the presets and the colour legend only

	}

	refresh() {

		this._render();

	}

	/** Phones: folded, the bar keeps the presets and the colour legend and puts the channels away. */
	setCollapsed( on ) {

		this.collapsed = !! on;
		closePopover( { focus: false } );
		this.el.classList.toggle( 'is-collapsed', this.collapsed );
		this._fold.setAttribute( 'aria-expanded', String( ! this.collapsed ) );
		this._fold.setAttribute( 'aria-label', this.collapsed ? 'Show the lens channels' : 'Hide the lens channels' );
		this._fold.innerHTML = xicon( this.collapsed ? 'expand' : 'collapse', 14 );

	}

	_usesRelevance() {

		const s = this.lens.state;
		return s.color === 'relevance' || s.size === 'relevance' || s.glow === 'relevance';

	}

	_metric( id ) {

		return this.lens.metrics.find( ( m ) => m.id === id ) || null;

	}

	_render() {

		const s = this.lens.state, g = this.app.graph;
		// the edge channel exists as long as the edge renderer can show a metric
		this.edge.supported = !! this.edges && typeof this.edges.setEdgeMetric === 'function';
		this._channelButtons.edges.hidden = ! this.edge.supported;
		const preset = matchPreset( s );
		LENS_PRESETS.forEach( ( p, k ) => this._presetButtons[ k ].setAttribute( 'aria-pressed', String( p.id === preset ) ) );
		// where the chips scroll (phones), the one that is on stays in sight
		const onChip = this._presetButtons[ LENS_PRESETS.findIndex( ( p ) => p.id === preset ) ], row = this._presets;
		if ( onChip && row.scrollWidth > row.clientWidth ) {

			if ( preset === 'communities' ) row.scrollLeft = 0; // at rest the row shows its start
			else if ( onChip.offsetLeft < row.scrollLeft ) row.scrollLeft = onChip.offsetLeft;
			else if ( onChip.offsetLeft + onChip.offsetWidth > row.scrollLeft + row.clientWidth ) row.scrollLeft = onChip.offsetLeft + onChip.offsetWidth - row.clientWidth;

		}

		const on = [];
		for ( const c of CHANNELS ) {

			const b = this._channelButtons[ c.id ];
			if ( ! b ) continue;
			const id = c.id === 'edges' ? this.edge.metric : s[ c.id ];
			const info = id ? ( c.id === 'edges' ? this.lens.edgeMetrics.find( ( m ) => m.id === id ) : this._metric( id ) ) : null;
			b._value.textContent = info ? info.label : c.off;
			b.classList.toggle( 'is-on', !! info );
			if ( info && ! b.hidden ) on.push( [ c.label, info.label ] );

		}

		// folded, the channel buttons are away: one line names what the picture shows
		this._summary.textContent = '';
		on.forEach( ( [ channel, metric ], k ) => this._summary.append( k ? ' · ' : '', `${channel} `, h( 'b', null, metric ) ) );
		this._summary.hidden = ! on.length;

		this._legend.textContent = '';
		if ( ! g ) return;
		this._legend.append( this._colorLegend( s, g ) );
		const notes = [ this._missingLegend(), this._sizeLegend( s, g ), this._glowLegend( s, g ), this._edgeLegend() ].filter( Boolean );
		for ( const note of notes ) if ( ! note.title ) note.title = note.textContent; // a narrow bar cuts the line
		if ( notes.length ) this._legend.append( h( 'div.wgx-lens-notes', null, notes ) );
		this._renderProgress();

	}

	_tick( metric, tick, estimate ) {

		const N = this.app.graph.N;
		return `${tick.bound}${estimate ? '≈ ' : ''}${formatMetric( metric, tick.value, N )}`;

	}

	_colorLegend( s, g ) {

		const lens = this.lens, L = lens.legend( 'color' );
		const box = h( 'div.wgx-legend' );
		if ( ! L ) {

			// the community palette: one strip across the leaf order, as the nodes are coloured
			const stops = [];
			for ( let k = 0; k <= 12; k ++ ) stops.push( paletteCss( s.palette, k / 12 ) );
			// the same count as the panel, the drawer and the tour ( communitySummary ); the finest
			// level of the hierarchy is a second number, never the same word for another one
			const { count: main, finest } = communitySummary( g );
			const text = main > 0 ? `${formatCount( main )} communities${finest > main ? ` (${formatCount( finest )} at the finest level)` : ''}; similar colours belong together`
				: finest > 0 ? 'A single community' : 'One colour: this graph has no communities';
			box.append( h( 'div.wgx-legend-bar', { style: { background: `linear-gradient(90deg, ${stops.join( ', ' )})` } } ),
				h( 'div.wgx-legend-ticks.is-text', null, h( 'span', null, text ) ) );
			return box;

		}

		const st = s.metrics[ L.metric ] || {}, estimate = L.ready && ! st.exact && st.total > 0;
		if ( ! L.ready ) {

			const waiting = L.metric === 'relevance' && this.interaction.selectedIndex < 0;
			box.append( h( 'div.wgx-legend-bar.is-wait', { style: { background: L.css } } ),
				h( 'div.wgx-legend-ticks.is-text', null, h( 'span', null, waiting ? 'Select a node: relevance is measured from the selection' : st.error ? `Could not compute ${L.label}` : `Computing ${L.label}…` ) ) );
			return box;

		}

		if ( L.categorical ) {

			// every category with a hue of its own is named; the rest share one grey and one entry
			const cats = L.categories || [], shown = cats.filter( ( c ) => c.distinct ), others = cats.filter( ( c ) => ! c.distinct );
			const row = h( 'div.wgx-legend-cats' );
			for ( const c of shown ) row.append( h( 'span.wgx-legend-cat', { title: `${c.label}: ${formatCount( c.count )} ${c.count === 1 ? 'node' : 'nodes'}` }, h( 'i', { style: { background: cssOfLinear( c.color[ 0 ], c.color[ 1 ], c.color[ 2 ] ) } } ), h( 'span', null, c.label ), h( 'b', null, formatCount( c.count ) ) ) );
			if ( others.length ) {

				const grey = others[ 0 ].color, count = others.reduce( ( n, c ) => n + c.count, 0 );
				const names = others.slice( 0, 12 ).map( ( c ) => `${c.label} ${formatCount( c.count )}` ).join( ', ' ) + ( others.length > 12 ? ', …' : '' );
				row.append( h( 'span.wgx-legend-cat', { title: `Drawn in one grey: ${names}` }, h( 'i', { style: { background: cssOfLinear( grey[ 0 ], grey[ 1 ], grey[ 2 ] ) } } ), h( 'span', null, `${formatCount( others.length )} other` ), h( 'b', null, formatCount( count ) ) ) );

			}

			box.append( row );
			return box;

		}

		const ticks = h( 'div.wgx-legend-ticks' );
		for ( const t of L.ticks ) ticks.append( h( 'span', { 'data-at': t.t === 0 ? 'start' : t.t === 1 ? 'end' : 'mid' }, this._tick( L.metric, t, estimate ) ) );
		box.append( h( 'div.wgx-legend-bar', { style: { background: L.css }, title: `${L.label}: ${SCALE_NOTE[ L.scale ] || L.scale}` } ), ticks );
		box.append( h( 'span.wgx-legend-scale', { title: SCALE_NOTE[ L.scale ] || '' }, scaleWord( L.scale ) ) );
		return box;

	}

	/** Nodes the colour metric has no value for (an attribute the file does not give them): grey. */
	_missingLegend() {

		const L = this.lens.legend( 'color' );
		if ( ! L || ! L.ready || ! ( L.missing > 0 ) ) return null;
		const c = L.missingColor;
		return h( 'span.wgx-lens-note', null, h( 'span.wgx-none-glyph', { 'aria-hidden': 'true', style: c ? { '--wgx-none': cssOfLinear( c[ 0 ], c[ 1 ], c[ 2 ] ) } : null } ),
			h( 'b', null, `${formatCount( L.missing )} ${L.missing === 1 ? 'node' : 'nodes'}` ), ` without a value of ${L.label}: grey` );

	}

	_sizeLegend( s, g ) {

		const L = this.lens.legend( 'size' );
		if ( ! L ) return null;
		const st = s.metrics[ L.metric ] || {}, estimate = L.ready && ! st.exact && st.total > 0;
		const note = h( 'span.wgx-lens-note', { title: SCALE_NOTE[ L.scale ] || '' }, h( 'span.wgx-size-glyph', { 'aria-hidden': 'true' }, h( 'i' ), h( 'i' ), h( 'i' ) ) );
		if ( ! L.ready ) note.append( L.metric === 'relevance' && this.interaction.selectedIndex < 0 ? 'Size: select a node' : `Size: computing ${L.label}…` );
		else {

			const a = L.ticks[ 0 ], b = L.ticks[ L.ticks.length - 1 ];
			note.append( 'Size ', h( 'b', null, `${this._tick( L.metric, a, estimate )} – ${this._tick( L.metric, b, false )}` ) );
			// no value is drawn at the small end: say so, or it reads as the smallest value. The line
			// is short, so the scale word gives way to it and the tooltip carries both in full
			if ( L.missing > 0 ) {

				note.append( ` · ${formatCount( L.missing )} with no value: smallest` );
				note.title = `Size: ${SCALE_NOTE[ L.scale ] || L.scale}; ${formatCount( L.missing )} ${L.missing === 1 ? 'node has' : 'nodes have'} no value of ${L.label} and ${L.missing === 1 ? 'is' : 'are'} drawn at the smallest size`;

			} else note.append( ` ${scaleWord( L.scale )}` );

		}

		return note;

	}

	_glowLegend( s, g ) {

		const L = this.lens.legend( 'glow' );
		if ( ! L ) return null;
		const note = h( 'span.wgx-lens-note', null, h( 'span.wgx-glow-glyph', { 'aria-hidden': 'true' } ) );
		if ( ! L.ready ) note.append( L.metric === 'relevance' && this.interaction.selectedIndex < 0 ? 'Glow: select a node' : `Glow: computing ${L.label}…` );
		else if ( ! ( L.count > 0 ) ) note.append( `No node glows: the top ${formatPercent( L.top )} by ${L.label} is a tie of many` );
		else note.append( h( 'b', null, `${formatCount( L.count )} ${L.count === 1 ? 'node glows' : 'nodes glow'}` ), ` · ${L.label} ≥ ${formatMetric( L.metric, L.threshold, g.N )}, top ${formatPercent( L.top )}` ); // the number first: a narrow bar cuts the end
		return note;

	}

	_edgeLegend() {

		const e = this.edge;
		if ( ! e.supported || ! e.metric ) return null;
		const info = this.lens.edgeMetrics.find( ( m ) => m.id === e.metric );
		const what = [ e.opacity ? 'opacity' : '', e.width ? 'width' : '' ].filter( Boolean ).join( ' and ' ) || 'nothing';
		const note = h( 'span.wgx-lens-note', null, h( 'span.wgx-edge-glyph', { 'aria-hidden': 'true' } ) );
		if ( ! e.result ) note.append( `Edges: computing ${info.label}…` );
		else {

			let min = Infinity, max = - Infinity;
			const raw = e.result.raw;
			for ( let k = 0; k < raw.length; k ++ ) {

				if ( raw[ k ] < min ) min = raw[ k ];
				if ( raw[ k ] > max ) max = raw[ k ];

			}

			note.append( 'Edges ', h( 'b', null, `${e.result.exact ? '' : '≈ '}${formatNumber( min )} – ${formatNumber( max )}` ), ` ${info.label.toLowerCase()} as ${what}, ${e.result.scale}` );

		}

		return note;

	}

	/** The quiet line under the legend: which shown metric is still an estimate, and how far along. */
	_renderProgress() {

		const s = this.lens.state, el = this._progress;
		const jobs = [];
		for ( const id of new Set( [ s.color, s.size, s.glow ].filter( Boolean ) ) ) {

			const m = s.metrics[ id ];
			if ( ! m || m.exact || ! ( m.total > 0 ) ) continue;
			jobs.push( { label: this._metric( id ).label, pivots: m.pivots, total: m.total, running: ! m.final } );

		}

		const e = this.edge;
		if ( e.metric && e.result && ! e.result.exact && e.result.total > 0 ) jobs.push( { label: this.lens.edgeMetrics.find( ( m ) => m.id === e.metric ).label, pivots: e.result.pivots, total: e.result.total, running: e.busy } );
		if ( ! jobs.length ) {

			el.hidden = true;
			return;

		}

		const job = jobs.reduce( ( a, b ) => ( b.pivots / b.total < a.pivots / a.total ? b : a ) );
		const text = `${job.label}: estimate from ${formatCount( job.pivots )} of ${formatCount( job.total )} pivots${job.running ? ', refining' : ''}${jobs.length > 1 ? ` · ${jobs.length - 1} more` : ''}`;
		if ( ! el.firstChild ) el.append( h( 'span.wgx-lens-progress-text' ), h( 'span.wgx-lens-progress-bar', { 'aria-hidden': 'true' }, h( 'i' ) ) );
		el.firstChild.textContent = text;
		el.lastChild.firstChild.style.width = ( 100 * Math.min( 1, job.pivots / job.total ) ).toFixed( 1 ) + '%';
		el.classList.toggle( 'is-running', job.running );
		el.hidden = false;

	}

	// ── menus ─────────────────────────────────────────────────────────────────────────────

	_toggleMenu( channel, anchor ) {

		const last = this._pop;
		this._pop = null;
		// the press that closed the menu was on its own button: stay closed
		if ( last && last.anchor === anchor && last.closedByAnchor ) return;

		const body = h( 'div.wgx-menu' );
		const fill = () => {

			body.textContent = '';
			body.append( ...this._menuContent( channel, () => pop.close(), fill ) );

		};

		const pop = openPopover( {
			host: this.el, anchor, content: body, label: `${channel.label} by`, className: 'wgx-lens-pop',
		} );
		this._pop = pop;
		fill();
		pop.reposition();
		const checked = body.querySelector( '[aria-checked="true"]' );
		if ( checked ) {

			checked.focus( { preventScroll: true } );
			checked.scrollIntoView( { block: 'nearest' } );

		}

	}

	_menuContent( channel, close, refill ) {

		const lens = this.lens, s = lens.state, isEdges = channel.id === 'edges';
		const current = isEdges ? this.edge.metric : s[ channel.id ];
		const out = [ h( 'div.wgx-menu-title', null, `${channel.label} by` ) ];
		const option = ( id, label, description, cost ) => {

			const b = h( 'button.wgx-menu-item', { type: 'button', role: 'menuitemradio', 'aria-checked': String( ( current || null ) === id ) },
				h( 'span.wgx-menu-check', { html: xicon( 'check', 14 ), 'aria-hidden': 'true' } ),
				h( 'span.wgx-menu-text', null, h( 'span.wgx-menu-label', null, label ), description ? h( 'span.wgx-menu-desc', null, description ) : null ),
				cost ? h( 'span.wgx-cost', { 'data-cost': cost, title: cost === 'slow' ? 'Slow: shown as an estimate first, then refined' : cost === 'fast' ? 'Fast: a moment on large graphs' : 'Instant' }, cost ) : null );
			b.addEventListener( 'click', () => {

				close();
				this.setChannel( channel.id, id );

			} );
			return b;

		};

		out.push( option( null, channel.off, channel.offNote, null ) );
		if ( isEdges ) {

			for ( const m of lens.edgeMetrics ) out.push( option( m.id, m.label, m.description, m.cost ) );

		} else {

			const list = lens.metrics.filter( ( m ) => channel.id === 'color' || m.kind !== 'categorical' );
			for ( const group of GROUP_ORDER ) {

				const items = list.filter( ( m ) => groupOf( m ) === group );
				if ( ! items.length ) continue;
				out.push( h( 'div.wgx-menu-head', null, group ) );
				for ( const m of items ) out.push( option( m.id, m.label, m.description, m.cost ) );

			}

		}

		// what else the channel can be told
		if ( channel.id === 'color' && s.color ) {

			// the ramp belongs to a metric: the community palette has none (it is a setting of the panel)
			const ramps = h( 'div.wgx-menu-ramps', { role: 'group', 'aria-label': 'Colour ramp' } );
			const active = this._ramp || s.ramp;
			for ( const r of lens.ramps ) {

				const b = h( 'button.wgx-ramp', { type: 'button', title: r.label, 'aria-label': `Ramp ${r.label}`, 'aria-pressed': String( r.id === active ), style: { background: r.css } } );
				b.addEventListener( 'click', () => {

					this._ramp = r.id;
					if ( lens.state.color ) lens.setColor( lens.state.color, { ramp: r.id, reverse: lens.state.reverse } );
					refill();

				} );
				ramps.append( b );

			}

			const rev = h( 'button.wgx-toggle', { type: 'button', 'aria-pressed': String( !! s.reverse ) }, iconEl( 'swap', 14 ), 'Reverse' );
			rev.addEventListener( 'click', () => {

				lens.setColor( lens.state.color, { reverse: ! lens.state.reverse } );
				refill();

			} );
			out.push( h( 'div.wgx-menu-foot', null, h( 'span.wgx-menu-k', null, 'Ramp' ), ramps, rev ) );

		} else if ( channel.id === 'glow' ) {

			const seg = h( 'div.wgx-seg', { role: 'group', 'aria-label': 'Share of nodes that glow' } );
			for ( const top of TOPS ) {

				const b = h( 'button', { type: 'button', 'aria-pressed': String( Math.abs( s.top - top ) < 1e-6 ) }, formatPercent( top ) );
				b.addEventListener( 'click', () => {

					if ( lens.state.glow ) lens.setGlow( lens.state.glow, { top } );
					else lens.options.top = top;
					refill();

				} );
				seg.append( b );

			}

			out.push( h( 'div.wgx-menu-foot', null, h( 'span.wgx-menu-k', null, 'Top' ), seg ) );

		} else if ( isEdges ) {

			const toggle = ( key, label ) => {

				const b = h( 'button.wgx-toggle', { type: 'button', 'aria-pressed': String( this.edge[ key ] ) }, label );
				b.addEventListener( 'click', () => {

					this.setEdgeMetric( this.edge.metric, { [ key ]: ! this.edge[ key ] } );
					refill();

				} );
				return b;

			};

			out.push( h( 'div.wgx-menu-foot', null, h( 'span.wgx-menu-k', null, 'Drives' ), toggle( 'opacity', 'Opacity' ), toggle( 'width', 'Width' ) ) );

		}

		return out;

	}

}
