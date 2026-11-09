// Card for the selected node (App module, order 41): label, degree, cluster path (coarsest →
// finest level, each crumb tinted with its cluster's hue), neighbours sorted by degree (click to
// select), pin/unpin and focus buttons. Desktop: bottom-right; phones: above the bottom sheet
// (uses --wgv-sheet-h set by Panel). Styles in src/ui/ui.css.
//
// It listens to app.events 'select' / 'pin' / 'graph' itself, so wiring is just
//   app.use( new NodeCard( { interaction, rig } ) )
// Callbacks default to interaction.select / rig.focusNode / interaction.togglePin and can be
// overridden: { onSelect( i ), onFocus( i ), onTogglePin( i ), isPinned( i ), onClose() }.
//
// API
//   card.show( index ) / card.hide() / card.index
//   card.setPalette( name )     cluster/neighbour tints follow the active palette
//   card.maxNeighbors           how many neighbours to list (default 12; the rest is counted)

import { h, icon, paletteCss } from './Panel.js';

const PIN = '<path d="M9.8 2.6l3.6 3.6-2 .6-2.3 2.3.2 2.6-1.4 1.4-2.4-2.4L3 13.2M8 8.3l-2.4-2.4 1.4-1.4 2.6.2 2.3-2.3z"/>';
const FOCUS = '<circle cx="8" cy="8" r="2.2"/><path d="M8 1.8v2.4M8 11.8v2.4M1.8 8h2.4M11.8 8h2.4"/>';
const svg = ( p ) => `<svg class="wgv-i" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;

export class NodeCard {

	constructor( options = {} ) {

		this.order = 41;
		this.options = { container: document.body, palette: 'paper', maxNeighbors: 12, ...options };
		this.interaction = options.interaction || null;
		this.rig = options.rig || null;
		this.palette = this.options.palette;
		this.maxNeighbors = this.options.maxNeighbors;
		this.index = - 1;
		this.graph = null;
		this._build();

	}

	init( app ) {

		this.app = app;
		this._onSelect = ( e ) => ( e.detail.index >= 0 ? this.show( e.detail.index ) : this.hide() );
		this._onPin = ( e ) => {

			if ( this.index >= 0 && ( e.detail.index === this.index || e.detail.index < 0 ) ) this._syncPin();

		};

		app.events.addEventListener( 'select', this._onSelect );
		app.events.addEventListener( 'pin', this._onPin );

	}

	setGraph( buffers, graph ) {

		this.graph = graph;
		this.hide();

	}

	dispose() {

		this.app?.events.removeEventListener( 'select', this._onSelect );
		this.app?.events.removeEventListener( 'pin', this._onPin );
		this.el.remove();

	}

	setPalette( name ) {

		this.palette = name;
		if ( this.index >= 0 ) this.show( this.index );

	}

	label( i ) {

		const g = this.graph;
		return ( g && g.labels && g.labels[ i ] ) || `Node ${i}`;

	}

	show( i ) {

		const g = this.graph;
		if ( ! g || i < 0 || i >= g.N ) return this.hide();
		this.index = i;
		const css = paletteCss( this.palette, g.leafRank[ i ] );
		this._swatch.style.background = css;
		this._swatch.style.color = css;
		this._title.textContent = this.label( i );
		this._title.title = this.label( i );
		this._degree.textContent = String( g.degree[ i ] );
		this._id.textContent = `#${i}`;

		// cluster path, coarsest level first
		this._path.textContent = '';
		const D = g.D;
		for ( let l = D - 1; l >= 0; l -- ) {

			const c = g.anc[ i * D + l ];
			const local = c - g.levelOffset[ l ];
			const mid = ( g.clusterRankRange[ 2 * c ] + g.clusterRankRange[ 2 * c + 1 ] ) / 2;
			this._path.append( h( 'span.wgv-crumb', { title: `Level ${l} cluster ${local} · ${g.clusterMass[ c ]} nodes` },
				h( 'i', { style: { background: paletteCss( this.palette, mid ) } } ), `L${l}`, h( 'b', null, String( local ) ) ) );
			if ( l > 0 ) this._path.append( h( 'span.wgv-crumb-sep', { 'aria-hidden': 'true' }, '›' ) );

		}

		this._pathRow.hidden = D === 0;

		// neighbours by degree
		const s = g.adjStart[ i ], n = g.degree[ i ];
		const nb = Array.from( g.adjList.subarray( s, s + n ) );
		nb.sort( ( a, b ) => g.degree[ b ] - g.degree[ a ] || a - b );
		this._nbCount.textContent = String( n );
		this._list.textContent = '';
		const shown = nb.slice( 0, this.maxNeighbors );
		for ( const j of shown ) {

			const b = h( 'button.wgv-nb', { title: `${this.label( j )} · degree ${g.degree[ j ]}` },
				h( 'i', { style: { background: paletteCss( this.palette, g.leafRank[ j ] ) } } ),
				h( 'span.wgv-nb-l', null, this.label( j ) ), h( 'span.wgv-nb-d', null, String( g.degree[ j ] ) ) );
			b.addEventListener( 'click', () => this._select( j ) );
			this._list.append( b );

		}

		if ( nb.length > shown.length ) this._list.append( h( 'span.wgv-nb-more', null, `+${nb.length - shown.length} more` ) );
		this._nbRow.hidden = n === 0;
		this._syncPin();
		this.el.classList.add( 'is-visible' );
		this.el.removeAttribute( 'inert' );
		this._list.scrollTop = 0;
		this._list.scrollLeft = 0;

	}

	hide() {

		this.index = - 1;
		this.el.classList.remove( 'is-visible' );
		this.el.setAttribute( 'inert', '' );

	}

	_select( j ) {

		if ( this.options.onSelect ) this.options.onSelect( j );
		else if ( this.interaction ) this.interaction.select( j );
		else this.show( j );

	}

	_isPinned( i ) {

		if ( this.options.isPinned ) return this.options.isPinned( i );
		return this.interaction ? this.interaction.isPinned( i ) : false;

	}

	_syncPin() {

		const on = this.index >= 0 && this._isPinned( this.index );
		this._pinBtn.setAttribute( 'aria-pressed', String( on ) );
		this._pinLabel.textContent = on ? 'Unpin' : 'Pin';
		this._pinBadge.hidden = ! on;

	}

	_build() {

		this._swatch = h( 'i.wgv-card-dot' );
		this._title = h( 'h3.wgv-card-title' );
		this._degree = h( 'b' );
		this._id = h( 'span.wgv-card-id' );
		this._pinBadge = h( 'span.wgv-badge', { hidden: true }, 'pinned' );
		this._path = h( 'div.wgv-crumbs' );
		this._pathRow = h( 'div.wgv-card-row', null, h( 'span.wgv-card-k', null, 'Cluster' ), this._path );
		this._nbCount = h( 'span' );
		this._list = h( 'div.wgv-nbs' );
		this._nbRow = h( 'div.wgv-card-nbwrap', null, h( 'div.wgv-card-k', null, 'Neighbors ', this._nbCount ), this._list );
		this._pinLabel = h( 'span', null, 'Pin' );
		this._pinBtn = h( 'button.wgv-btn.wgv-btn-row', { 'aria-pressed': 'false' }, h( 'span', { html: svg( PIN ) } ), this._pinLabel );
		this._pinBtn.addEventListener( 'click', () => {

			if ( this.index < 0 ) return;
			if ( this.options.onTogglePin ) this.options.onTogglePin( this.index );
			else this.interaction?.togglePin( this.index );
			this._syncPin();

		} );
		const focus = h( 'button.wgv-btn.wgv-btn-row', { html: `${svg( FOCUS )}<span>Focus</span>` } );
		focus.addEventListener( 'click', () => {

			if ( this.index < 0 ) return;
			if ( this.options.onFocus ) this.options.onFocus( this.index );
			else this.rig?.focusNode( this.index );

		} );
		const close = h( 'button.wgv-icon.wgv-card-x', { 'aria-label': 'Clear selection', html: icon( 'close' ) } );
		close.addEventListener( 'click', () => {

			if ( this.options.onClose ) this.options.onClose();
			else if ( this.interaction ) this.interaction.clearSelection();
			else this.hide();

		} );
		this.el = h( 'div.wgv-card.wgv-glass', { role: 'region', 'aria-label': 'Selected node', inert: true },
			h( 'div.wgv-card-head', null, this._swatch,
				h( 'div.wgv-card-titles', null, this._title,
					h( 'div.wgv-card-meta', null, h( 'span', null, 'degree ', this._degree ), this._id, this._pinBadge ) ),
				close ),
			this._pathRow,
			this._nbRow,
			h( 'div.wgv-card-actions', null, this._pinBtn, focus ) );
		this.options.container.append( this.el );

	}

}
