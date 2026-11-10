// Insights drawer (slots.right, toggled by a tool button in slots.topRight and by "I"): the
// graph at a glance, from the staged summary of the analytics worker ( lens.summary() ).
//
//  · Observations: one or two plain sentences, only where the numbers support them
//    ( observations.js ).
//  · Figures: nodes, edges, density, average degree, components, degeneracy, clustering and
//    transitivity, assortativity, average path length and diameter. Sampled or bounded values say
//    so ( "≈", "sampled", "between 5 and 7" ); what a later stage brings shows a placeholder until
//    it arrives.
//  · Degree distribution and k-core profile as small bar charts ( charts.js ).
//  · Top nodes by a chosen metric with rank bars; a click selects the node and flies to it, the
//    row under the pointer is previewed in the graph.
//  · Communities of the coarsest useful level ( communities.js ): colour, a name taken from the
//    best-connected member, size and the share of its edges that stay inside. A click focuses the
//    community ( interaction.focusCluster → 'cluster' event, the highlight model ) and frames it
//    with the camera; a second click lets go.
//  The summary is computed when the drawer is first opened for a graph, never before.
//  Esc with the focus inside the drawer leaves a highlight first, then closes the drawer.
//
//   const insights = new Insights( { app, lens, interaction, rig, slots } )
//   insights.toggle( on? ) / isOpen / topMetric ( 'degree' | 'pagerank' | … ) / setTopMetric( id )
//   insights.focusCommunity( cluster )   focus + frame (what a click on a community row does)
//   insights.communities()               the list of the drawer ( communityList of the level shown )
//   insights.dispose()

import { h, xicon, nodeCss, clusterCss } from './ui.js';
import { formatNumber, formatRatio, formatCount, compactCount, formatPercent, formatMetric, formatRange } from './format.js';
import { observe } from './observations.js';
import { binCounts, barChart } from './charts.js';
import { usefulLevel, communityList, labelOf } from './communities.js';
import { clusterMembers } from './highlight.js';
import { Vector3 } from 'three/webgpu';

const TOP_METRICS = [
	{ id: 'degree', label: 'Degree' },
	{ id: 'pagerank', label: 'PageRank' },
	{ id: 'betweenness', label: 'Betweenness' },
	{ id: 'closeness', label: 'Closeness' },
	{ id: 'eigenvector', label: 'Eigenvector' },
];
const STAGES = [ 'basic', 'centrality', 'distance', 'done' ];
const STAGE_NOTE = { basic: 'ranking nodes…', centrality: 'measuring distances…', distance: 'finding brokers…' };
const MAX_COMMUNITIES = 8;
const _center = new Vector3();

export class Insights {

	constructor( { app, lens, interaction, rig, slots } ) {

		this.app = app;
		this.lens = lens;
		this.interaction = interaction;
		this.rig = rig;
		this.isOpen = false;
		this.topMetric = 'degree';
		this._summary = null;
		this._requested = null; // the graph the summary was asked for
		this._allCommunities = false;
		this._communities = null;
		this._build();
		slots.topRight.prepend( this.button ); // first on screen ( CSS order 0 ), first in the Tab order
		slots.right.prepend( this.el );

		this._onGraph = () => {

			this._summary = null;
			this._requested = null;
			this._failed = false;
			this._communities = null;
			this._allCommunities = false;
			if ( this.isOpen ) this._request();
			this._render();

		};

		this._onCluster = () => this._syncFocus();
		this._onLens = ( e ) => {

			// community colours follow the palette; the ranking follows better values of its metric
			if ( e.detail.reason === 'config' && this.isOpen && e.detail.palette !== this._palette ) this._render();
			else if ( e.detail.reason === 'data' ) this._refreshTop();

		};

		app.events.addEventListener( 'graph', this._onGraph );
		app.events.addEventListener( 'cluster', this._onCluster );
		app.events.addEventListener( 'lens', this._onLens );

	}

	toggle( on ) {

		const next = on === undefined ? ! this.isOpen : !! on;
		if ( next === this.isOpen ) return;
		this.isOpen = next;
		this.el.hidden = ! next;
		this.el.classList.toggle( 'is-open', next );
		this.button.setAttribute( 'aria-pressed', String( next ) );
		document.documentElement.classList.toggle( 'wgx-insights-open', next );
		if ( next ) {

			this._request();
			this._render();

		} else this.interaction.setPreview( - 1 );

		this.app.events.dispatchEvent( new CustomEvent( 'insights', { detail: { open: next } } ) );

	}

	setTopMetric( id ) {

		if ( ! TOP_METRICS.some( ( m ) => m.id === id ) ) return;
		this.topMetric = id;
		if ( this.isOpen ) this._renderTop();

	}

	communities() {

		const g = this.app.graph;
		if ( ! g ) return [];
		if ( ! this._communities || this._communities.graph !== g ) {

			const level = usefulLevel( g );
			this._communities = { graph: g, level, list: level >= 0 ? communityList( g, level ) : [] };

		}

		return this._communities.list;

	}

	/** Focuses a community (highlight model, 'cluster' event) and frames it; again = let go. */
	focusCommunity( cluster, { toggle = true, frame = true } = {} ) {

		const I = this.interaction, g = this.app.graph;
		if ( ! g ) return;
		if ( toggle && I.clusterFocus === cluster ) {

			I.focusCluster( - 1 );
			return;

		}

		I.focusCluster( cluster );
		if ( frame && I.clusterFocus === cluster ) this.frameCluster( cluster );

	}

	/** Moves the camera so the members of a cluster fill the view. */
	frameCluster( cluster, { duration } = {} ) {

		const g = this.app.graph, p = this.app.positions, rig = this.rig;
		if ( ! g || ! p || ! rig || rig.immersive ) return;
		const members = clusterMembers( g, cluster );
		if ( ! members.length ) return;
		let x = 0, y = 0, z = 0;
		for ( let k = 0; k < members.length; k ++ ) {

			const i = 4 * members[ k ];
			x += p[ i ]; y += p[ i + 1 ]; z += p[ i + 2 ];

		}

		x /= members.length; y /= members.length; z /= members.length;
		let r2 = 0;
		for ( let k = 0; k < members.length; k ++ ) {

			const i = 4 * members[ k ];
			r2 = Math.max( r2, ( p[ i ] - x ) ** 2 + ( p[ i + 1 ] - y ) ** 2 + ( p[ i + 2 ] - z ) ** 2 );

		}

		const radius = Math.max( Math.sqrt( r2 ) * 1.25, g.layoutRadius * 0.08 );
		// the rig knows the free part of the view; without that method, the plain camera frustum
		const cam = this.app.camera;
		const half = cam && cam.fov ? Math.atan( Math.tan( cam.fov * Math.PI / 360 ) * Math.min( 1, cam.aspect || 1 ) ) : 0.35;
		const distance = rig.fitDistance ? rig.fitDistance( radius ) : radius / Math.sin( half );
		rig.flyTo( _center.set( x, y, z ), distance, duration ? { duration } : {} );

	}

	dispose() {

		this.app.events.removeEventListener( 'graph', this._onGraph );
		this.app.events.removeEventListener( 'cluster', this._onCluster );
		this.app.events.removeEventListener( 'lens', this._onLens );
		document.documentElement.classList.remove( 'wgx-insights-open' );
		this.button.remove();
		this.el.remove();

	}

	// ── data ──────────────────────────────────────────────────────────────────────────────

	_request() {

		const g = this.app.graph;
		if ( ! g || this._requested === g ) return;
		this._requested = g;
		const take = ( s ) => {

			if ( this.app.graph !== g || ! s ) return;
			this._summary = s;
			if ( this.isOpen ) this._render();

		};

		this.lens.summary( { onPreview: take } ).then( take, ( e ) => {

			console.error( e );
			if ( this.app.graph === g ) {

				this._failed = true;
				this._render();

			}

		} );

	}

	// ── DOM ───────────────────────────────────────────────────────────────────────────────

	_build() {

		this.button = h( 'button.wgx.wgx-tool', { type: 'button', 'aria-pressed': 'false', 'aria-label': 'Insights', title: 'Insights (I)', html: xicon( 'insights' ) } );
		this.button.addEventListener( 'click', () => this.toggle() );
		this._title = h( 'span.wgx-drawer-name' );
		this._stage = h( 'span.wgx-drawer-stage', { role: 'status' } );
		const close = h( 'button.wgx-icon-btn', { type: 'button', 'aria-label': 'Close insights', html: xicon( 'close' ) } );
		close.addEventListener( 'click', () => this.toggle( false ) );
		this._body = h( 'div.wgx-drawer-body' );
		this.el = h( 'aside.wgx.wgx-drawer.wgv-glass', { 'aria-label': 'Graph insights', hidden: true },
			h( 'header.wgx-drawer-head', null, h( 'div.wgx-drawer-titles', null, h( 'h2', null, 'Insights' ), this._title ), this._stage, close ),
			this._body );
		// Esc from inside the drawer: a highlight is left first (the one rule of the canvas), then
		// the drawer itself closes and the focus goes back to its button
		this.el.addEventListener( 'keydown', ( e ) => {

			const I = this.interaction;
			if ( e.key !== 'Escape' || e.defaultPrevented || I.highlight.active || I.pathPickFrom >= 0 ) return;
			e.preventDefault();
			e.stopPropagation();
			this.toggle( false );
			this.button.focus( { preventScroll: true } );

		} );

	}

	_render() {

		if ( ! this.isOpen ) return;
		const g = this.app.graph, s = this._summary;
		this._palette = this.lens.state.palette;
		this._title.textContent = g ? ( g.name || 'Graph' ) : '';
		const body = this._body;
		const scroll = body.scrollTop;
		body.textContent = '';
		if ( ! g ) {

			body.append( h( 'p.wgx-empty', null, 'No graph loaded.' ) );
			return;

		}

		const stage = s ? s.stage : null;
		this._stage.textContent = this._failed ? 'failed' : stage === 'done' ? '' : stage ? STAGE_NOTE[ stage ] || '' : 'reading the graph…';
		this._stage.classList.toggle( 'is-busy', stage !== 'done' && ! this._failed );

		// observations
		const notes = s ? observe( s, { limit: 2 } ) : [];
		if ( notes.length ) body.append( h( 'section.wgx-notes', null, notes.map( ( n ) => h( 'p', { 'data-id': n.id }, n.text ) ) ) );

		body.append( this._figures( g, s ) );
		if ( s && s.degree ) {

			const bins = binCounts( s.degree.histogram, { maxBars: 12 } );
			body.append( h( 'section.wgx-sec', null, h( 'h3', null, 'Degree distribution', h( 'span', null, 'nodes by number of links' ) ),
				barChart( bins, { name: 'Degree distribution', unit: 'nodes', total: g.N, describe: ( b ) => `degree ${formatRange( b.from, b.to )}`, idle: `median ${formatNumber( s.degree.median )} · mean ${formatNumber( s.degree.mean, { digits: 3 } )} · max ${formatCount( s.degree.max )}` } ) ) );

		}

		if ( s && s.core && s.core.sizes && s.core.degeneracy > 0 ) {

			const sizes = Array.from( s.core.sizes );
			const bins = binCounts( sizes, { maxBars: 12, mode: sizes.length <= 12 ? 'unit' : 'linear' } );
			body.append( h( 'section.wgx-sec', null, h( 'h3', null, 'k-core profile', h( 'span', null, 'nodes by core number' ) ),
				barChart( bins, { name: 'k-core profile', unit: 'nodes', total: g.N, describe: ( b ) => `core ${formatRange( b.from, b.to )}`, idle: `innermost core k = ${formatCount( s.core.degeneracy )}: ${formatCount( sizes[ s.core.degeneracy ] )} nodes` } ) ) );

		}

		this._topSection = h( 'section.wgx-sec' );
		const catchUp = () => {

			if ( this._topStale ) setTimeout( () => this._refreshTop(), 0 );

		};

		this._topSection.addEventListener( 'pointerleave', catchUp );
		this._topSection.addEventListener( 'focusout', catchUp );
		body.append( this._topSection );
		this._renderTop();
		body.append( this._communitySection( g ) );
		body.scrollTop = scroll;
		this._syncFocus();

	}

	_figures( g, s ) {

		const tile = ( label, value, note, title ) => h( 'div.wgx-fig', title ? { title } : null, h( 'span.wgx-fig-k', null, label ), value === null ? h( 'span.wgx-fig-v.is-wait', { 'aria-label': 'computing' } ) : h( 'span.wgx-fig-v', null, value ), note ? h( 'span.wgx-fig-n', null, note ) : null );
		const N = g.N, E = g.E;
		const out = h( 'section.wgx-figs' );
		out.append( tile( 'Nodes', formatCount( N ) ), tile( 'Edges', formatCount( E ) ),
			tile( 'Average degree', formatNumber( N ? 2 * E / N : 0, { digits: 3 } ), s && s.degree ? `max ${formatCount( s.degree.max )}` : null ) );
		const density = N > 1 ? 2 * E / ( N * ( N - 1 ) ) : 0;
		out.append( tile( 'Density', density >= 0.001 ? formatPercent( density ) : formatNumber( density, { digits: 2 } ), null, 'Share of all possible links that exist' ) );
		const has = ( key ) => s && s[ key ] !== undefined && s[ key ] !== null;
		out.append( tile( 'Components', has( 'components' ) ? formatCount( s.components.count ) : null,
			has( 'components' ) && s.components.count > 1 ? `largest ${formatPercent( s.components.largestFraction )}` : has( 'components' ) ? 'connected' : null ) );
		out.append( tile( 'Degeneracy', has( 'core' ) ? formatCount( s.core.degeneracy ) : null, 'deepest k-core', 'The largest k for which a k-core exists' ) );
		out.append( tile( 'Clustering', has( 'clustering' ) ? formatRatio( s.clustering.average ) : null, 'average, local', 'Average over the nodes of the share of their neighbours that are linked' ) );
		out.append( tile( 'Transitivity', has( 'clustering' ) ? formatRatio( s.clustering.transitivity ) : null, has( 'clustering' ) ? `${compactCount( s.clustering.triangles )} triangles` : null, 'Share of connected triples that close into a triangle' ) );
		out.append( tile( 'Assortativity', s && Number.isFinite( s.assortativity ) ? formatRatio( s.assortativity ) : s && s.assortativity !== undefined ? '–' : null, 'by degree', 'Correlation of the degrees at the two ends of an edge' ) );
		const staged = ( stage ) => s && STAGES.indexOf( s.stage ) >= STAGES.indexOf( stage );
		if ( staged( 'distance' ) && s.distance ) {

			const d = s.distance;
			out.append( tile( 'Average distance', `${d.exact ? '' : '≈ '}${formatNumber( d.averagePathLength, { digits: 3 } )}`, d.exact ? 'exact' : `from ${compactCount( d.pivots )} sources`, 'Mean number of hops between two nodes that can reach each other' ) );

		} else out.append( tile( 'Average distance', staged( 'distance' ) ? '–' : null ) );

		if ( staged( 'distance' ) && s.diameter ) {

			const d = s.diameter;
			const exact = d.exact || d.lower === d.upper;
			const upper = Number.isFinite( d.upper ) && d.upper > 0;
			out.append( tile( 'Diameter', exact ? formatCount( d.lower ) : upper ? `${formatCount( d.lower )}–${formatCount( d.upper )}` : `≥ ${formatCount( d.lower )}`,
				exact ? 'exact' : upper ? 'bounds' : 'lower bound', 'Longest shortest path' + ( s.components && s.components.count > 1 ? ' within a component' : '' ) ) );

		} else out.append( tile( 'Diameter', staged( 'distance' ) ? '–' : null ) );

		if ( staged( 'distance' ) && s.distance && Number.isFinite( s.distance.effectiveDiameter ) ) {

			out.append( tile( '90 % within', `${formatNumber( s.distance.effectiveDiameter, { digits: 2 } )} hops`, s.distance.exact ? 'exact' : 'sampled', 'Effective diameter: the distance within which 90 % of the reachable pairs lie' ) );

		} else out.append( tile( '90 % within', staged( 'distance' ) ? '–' : null ) );

		return out;

	}

	/**
	 * The ranking to show: the summary's list, or the lens's own values of that metric once they
	 * are better (exact, or an estimate from more sources), e.g. while a lens refines betweenness.
	 */
	_topData( metric ) {

		const g = this.app.graph, s = this._summary;
		const listed = s && s.top ? s.top[ metric ] || null : null;
		const v = this.lens.values( metric );
		if ( v && v.raw && v.order && ( ! listed || ( ! listed.exact && ( v.exact || v.pivots > ( listed.pivots || 0 ) ) ) ) ) {

			const nodes = Array.from( v.order.subarray( 0, Math.min( listed ? listed.nodes.length : 10, g.N ) ) );
			return { nodes, values: nodes.map( ( i ) => v.raw[ i ] ), exact: !! v.exact, pivots: v.pivots };

		}

		return listed;

	}

	/** Redraws the ranking when its data changed, but not under the pointer or the keyboard focus. */
	_refreshTop() {

		const sec = this._topSection;
		if ( ! sec || ! this.isOpen ) return;
		if ( sec.matches( ':hover' ) || sec.contains( document.activeElement ) ) {

			this._topStale = true;
			return;

		}

		this._topStale = false;
		this._renderTop();

	}

	_renderTop() {

		const sec = this._topSection, g = this.app.graph, s = this._summary;
		if ( ! sec || ! g ) return;
		this._topStale = false;
		sec.textContent = '';
		const tabs = h( 'div.wgx-tabs', { role: 'tablist', 'aria-label': 'Rank nodes by' } );
		for ( const m of TOP_METRICS ) {

			const b = h( 'button', { type: 'button', role: 'tab', 'aria-selected': String( m.id === this.topMetric ) }, m.label );
			b.addEventListener( 'click', () => this.setTopMetric( m.id ) );
			tabs.append( b );

		}

		sec.append( h( 'h3', null, 'Top nodes' ), tabs );
		const top = this._topData( this.topMetric );
		if ( ! top ) {

			sec.append( h( 'p.wgx-empty.is-wait', null, s && s.stage === 'done' ? 'Not available for this graph.' : 'Computing…' ) );
			return;

		}

		const max = top.values.length ? Math.max( ...top.values ) : 0;
		const list = h( 'ol.wgx-rank', { 'aria-label': `Top nodes by ${this.topMetric}` } );
		top.nodes.forEach( ( node, k ) => {

			const label = labelOf( g, node );
			const value = formatMetric( this.topMetric, top.values[ k ], g.N );
			const row = h( 'button.wgx-rank-row', { type: 'button', title: label.length > 28 ? label : null },
				h( 'span.wgx-rank-n', null, String( k + 1 ) ),
				h( 'i.wgx-dot', { style: { '--wgx-dot': nodeCss( g, this.lens, node ) } } ),
				h( 'span.wgx-rank-l', null, label ),
				h( 'span.wgx-rank-v', null, ( top.exact ? '' : '≈ ' ) + value ),
				h( 'span.wgx-rank-bar', { 'aria-hidden': 'true' }, h( 'i', { style: { width: ( max > 0 ? 100 * top.values[ k ] / max : 0 ).toFixed( 1 ) + '%' } } ) ) );
			row.addEventListener( 'click', () => this.interaction.select( node, { focus: true, ripple: true } ) );
			row.addEventListener( 'pointerenter', () => this.interaction.setPreview( node ) );
			row.addEventListener( 'pointerleave', () => this.interaction.setPreview( - 1 ) );
			row.addEventListener( 'focus', () => this.interaction.setPreview( node ) );
			row.addEventListener( 'blur', () => this.interaction.setPreview( - 1 ) );
			list.append( h( 'li', null, row ) );

		} );
		sec.append( list );
		if ( ! top.exact ) sec.append( h( 'p.wgx-foot', null, `Estimate${top.pivots ? ` from ${formatCount( top.pivots )} sampled sources` : ' from sampled sources'}: the order of close values may differ.` ) );

	}

	_communitySection( g ) {

		const sec = h( 'section.wgx-sec.wgx-comm' );
		const list = this.communities();
		const level = this._communities.level;
		if ( ! list.length ) {

			sec.append( h( 'h3', null, 'Communities' ), h( 'p.wgx-empty', null, g.D ? 'The clustering found a single community.' : 'This graph has no community structure.' ) );
			return sec;

		}

		const c = g.clustering;
		const how = c && c.method ? `${String( c.method )}${Number.isFinite( c.modularity ) ? ` · modularity ${formatNumber( c.modularity, { digits: 2 } )}` : ''}` : g.D > 1 ? `coarsest of ${g.D} levels` : '';
		sec.append( h( 'h3', null, `${formatCount( list.length )} communities`, how ? h( 'span', null, how ) : null ) );
		const rows = h( 'ul.wgx-comm-list' );
		const shown = this._allCommunities ? list : list.slice( 0, MAX_COMMUNITIES );
		const maxSize = list[ 0 ].size;
		this._commRows = new Map();
		for ( const cm of shown ) {

			const share = Number.isFinite( cm.share ) ? formatPercent( cm.share ) : '–';
			const row = h( 'button.wgx-comm-row', { type: 'button', 'aria-pressed': 'false', title: `${formatCount( cm.size )} nodes around ${cm.name}; ${formatCount( cm.internal )} links inside, ${formatCount( cm.cut )} to other communities` },
				h( 'i.wgx-dot', { style: { '--wgx-dot': clusterCss( g, this.lens, cm.cluster ) } } ),
				h( 'span.wgx-comm-l', null, cm.name ),
				h( 'span.wgx-comm-s', null, formatCount( cm.size ) ),
				h( 'span.wgx-comm-i', { title: 'Share of its links that stay inside' }, share ),
				h( 'span.wgx-rank-bar', { 'aria-hidden': 'true' }, h( 'i', { style: { width: ( 100 * cm.size / maxSize ).toFixed( 1 ) + '%' } } ) ) );
			row.addEventListener( 'click', () => this.focusCommunity( cm.cluster ) );
			this._commRows.set( cm.cluster, row );
			rows.append( h( 'li', null, row ) );

		}

		sec.append( h( 'div.wgx-comm-cols', { 'aria-hidden': 'true' }, h( 'span', null, 'around' ), h( 'span', null, 'nodes' ), h( 'span', null, 'inside' ) ), rows );
		if ( list.length > MAX_COMMUNITIES ) {

			const more = h( 'button.wgx-link', { type: 'button' }, this._allCommunities ? 'Show fewer' : `Show all ${formatCount( list.length )}` );
			more.addEventListener( 'click', () => {

				this._allCommunities = ! this._allCommunities;
				this._render();

			} );
			sec.append( more );

		}

		if ( level >= 0 ) sec.append( h( 'p.wgx-foot', null, 'Named after the best-connected member. “Inside” is the share of a community’s links that stay within it.' ) );
		return sec;

	}

	_syncFocus() {

		if ( ! this._commRows ) return;
		const focus = this.interaction.clusterFocus;
		for ( const [ cluster, row ] of this._commRows ) row.setAttribute( 'aria-pressed', String( cluster === focus ) );

	}

}
