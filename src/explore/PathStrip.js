// Path strip (slots.bottom, above the lens bar): the shortest path as something to read.
//
//  · While a target is being picked ( interaction.pickPathTarget, the card's "Path from here" ):
//    a prompt with the start node, a button that opens the search and one that cancels.
//  · With a path ( 'path' event of the analysis tools ): the hops as chips in order (community
//    colour, label), the length, an "all shortest paths" switch and a close button. A chip selects
//    its node (the path stays, because the node lies on it) and pans to it; the chip under the
//    pointer is previewed in the graph. With all shortest paths on, every hop position shows the
//    node most of them run through, the share of the paths through it and how many alternatives
//    there are (their names in the tooltip).
//  · Both ends of the path stay in sight. Where the hops do not fit beside the rest, the ones
//    before the target fold into one "+n" chip (their names in its tooltip, a click brings them
//    back as a list that scrolls); where even that is too wide, the hops take a row of their own
//    ( .is-stacked, always so on phones ). The strip measures itself again whenever the room for
//    it changes (window, panel, the right column opening or closing).
//  · When the target cannot be reached ( 'nopath' ): a short notice that goes away by itself;
//    Esc takes the notice away first and leaves the highlight as it is.
//
//   const strip = new PathStrip( { app, interaction, lens, slots, onSearch } )
//   strip.dispose()

import { h, xicon, iconEl, nodeCss, coarsePointer } from './ui.js';
import { labelOf } from './communities.js';
import { formatCount, formatPercent } from './format.js';

export class PathStrip {

	constructor( { app, interaction, lens, slots, onSearch } ) {

		this.app = app;
		this.interaction = interaction;
		this.lens = lens;
		this.onSearch = onSearch || null;
		this._path = null;
		this._notice = null;
		this.el = h( 'div.wgx.wgx-path.wgv-glass', { role: 'region', 'aria-label': 'Shortest path', hidden: true } );
		slots.bottom.append( this.el );

		this._onPath = ( e ) => {

			this._path = e.detail || null;
			this._expanded = false;
			if ( this._path ) this._clearNotice();
			this._render();

		};

		this._onPick = () => this._render();
		this._onNoPath = ( e ) => {

			this._clearNotice();
			this._notice = e.detail;
			this._timer = setTimeout( () => {

				this._notice = null;
				this._render();

			}, 5200 );
			this._render();

		};

		this._onGraph = () => {

			this._path = null;
			this._clearNotice();
			this._render();

		};

		this._onLens = ( e ) => {

			if ( e.detail.reason === 'config' && this._path ) this._render();

		};

		// walking along the path: the chip of the selected node is the current one
		this._onSelect = ( e ) => {

			for ( const chip of this.el.querySelectorAll( '.wgx-hop[data-node]' ) ) chip.setAttribute( 'aria-current', String( + chip.dataset.node === e.detail.index ) );
			this._fit(); // the card came or went: the room beside the right column changed

		};

		// the room for the strip changes with the window, the panel and the right column
		this._onRoom = () => this._fit();
		this._observer = typeof ResizeObserver === 'function' ? new ResizeObserver( this._onRoom ) : null;
		this._observer?.observe( slots.bottom );
		window.addEventListener( 'resize', this._onRoom );

		// Esc answers the notice before it peels a highlight (capture: ahead of the key handling
		// of the interaction); typing in a field keeps its own Esc
		this._onKey = ( e ) => {

			if ( e.key !== 'Escape' || ! this._notice || e.defaultPrevented ) return;
			if ( e.target && e.target.closest && e.target.closest( 'input, textarea, select, [contenteditable]' ) ) return;
			e.preventDefault();
			e.stopPropagation();
			this._clearNotice();
			this._render();

		};

		window.addEventListener( 'keydown', this._onKey, true );

		const ev = app.events;
		ev.addEventListener( 'select', this._onSelect );
		ev.addEventListener( 'path', this._onPath );
		ev.addEventListener( 'pathpick', this._onPick );
		ev.addEventListener( 'nopath', this._onNoPath );
		ev.addEventListener( 'graph', this._onGraph );
		ev.addEventListener( 'lens', this._onLens );
		ev.addEventListener( 'insights', this._onRoom );
		ev.addEventListener( 'exploreinsets', this._onRoom );

	}

	dispose() {

		this._clearNotice();
		window.removeEventListener( 'keydown', this._onKey, true );
		const ev = this.app.events;
		ev.removeEventListener( 'select', this._onSelect );
		ev.removeEventListener( 'path', this._onPath );
		ev.removeEventListener( 'pathpick', this._onPick );
		ev.removeEventListener( 'nopath', this._onNoPath );
		ev.removeEventListener( 'graph', this._onGraph );
		ev.removeEventListener( 'lens', this._onLens );
		ev.removeEventListener( 'insights', this._onRoom );
		ev.removeEventListener( 'exploreinsets', this._onRoom );
		window.removeEventListener( 'resize', this._onRoom );
		this._observer?.disconnect();
		this.el.remove();

	}

	_clearNotice() {

		clearTimeout( this._timer );
		this._notice = null;

	}

	_chip( node, extra = null ) {

		const g = this.app.graph, I = this.interaction;
		const label = labelOf( g, node );
		const b = h( 'button.wgx-hop', { type: 'button', title: label, 'aria-current': String( node === I.selectedIndex ) },
			h( 'i.wgx-dot', { style: { '--wgx-dot': nodeCss( g, this.lens, node ) } } ), h( 'span.wgx-hop-l', null, label ), extra );
		b.dataset.node = node;
		b.addEventListener( 'click', () => I.walk( node ) );
		b.addEventListener( 'pointerenter', () => I.setPreview( node ) );
		b.addEventListener( 'pointerleave', () => I.setPreview( - 1 ) );
		b.addEventListener( 'focus', () => I.setPreview( node ) );
		b.addEventListener( 'blur', () => I.setPreview( - 1 ) );
		return b;

	}

	_render() {

		const el = this.el, g = this.app.graph, I = this.interaction;
		el.textContent = '';
		el.classList.remove( 'is-stacked' );
		this._hops = null;
		const picking = I.pathPickFrom >= 0;
		const p = this._path;
		if ( ! g || ( ! picking && ! p && ! this._notice ) ) {

			el.hidden = true;
			return;

		}

		el.hidden = false;
		const closeBtn = ( label, fn ) => {

			const b = h( 'button.wgx-icon-btn', { type: 'button', 'aria-label': label, title: label, html: xicon( 'close', 14 ) } );
			b.addEventListener( 'click', fn );
			return b;

		};

		if ( picking ) {

			el.dataset.state = 'pick';
			const search = h( 'button.wgx-btn', { type: 'button' }, iconEl( 'search', 14 ), 'Search' );
			search.addEventListener( 'click', () => this.onSearch && this.onSearch() );
			el.append( iconEl( 'path' ), h( 'span.wgx-path-msg', null, 'Path from ', h( 'b', null, labelOf( g, I.pathPickFrom ) ), coarsePointer() ? ': tap the target node' : ': click the target node' ), search, closeBtn( 'Cancel', () => I.cancelPathPick() ) );
			return;

		}

		if ( ! p ) {

			el.dataset.state = 'none';
			const n = this._notice;
			el.append( iconEl( 'off' ), h( 'span.wgx-path-msg', null, 'No path: ', h( 'b', null, labelOf( g, n.source ) ), ' and ', h( 'b', null, labelOf( g, n.target ) ), ' are not connected' ),
				closeBtn( 'Dismiss', () => {

					this._clearNotice();
					this._render();

				} ) );
			return;

		}

		el.dataset.state = 'path';
		const hops = h( 'div.wgx-hops', { role: 'list', 'aria-label': 'Hops' } );
		const items = []; // one per hop position: { chip, sep, node, label }
		const add = ( chip, node ) => {

			const sep = items.length ? h( 'span.wgx-hop-sep', { 'aria-hidden': 'true', html: xicon( 'next', 10 ) } ) : null;
			items.push( { chip, sep, node, label: labelOf( g, node ) } );
			if ( sep ) hops.append( sep );
			hops.append( chip );

		};

		if ( ! p.all ) {

			for ( const node of p.nodes ) add( this._chip( node ), node );

		} else {

			// every geodesic: per hop position the node most paths use, and how many others there are
			const levels = [];
			for ( let k = 0; k < p.nodes.length; k ++ ) ( levels[ p.depth[ k ] ] ||= [] ).push( { node: p.nodes[ k ], share: p.nodeShare[ k ] } );
			levels.forEach( ( list, d ) => {

				list.sort( ( a, b ) => b.share - a.share || g.degree[ b.node ] - g.degree[ a.node ] );
				const lead = list[ 0 ];
				const extra = list.length > 1 ? h( 'span.wgx-hop-alt', null, `${formatPercent( lead.share )} · +${formatCount( list.length - 1 )}` ) : null;
				const chip = this._chip( lead.node, extra );
				if ( list.length > 1 ) chip.title = list.slice( 0, 12 ).map( ( x ) => `${labelOf( g, x.node )} ${formatPercent( x.share )}` ).join( '\n' ) + ( list.length > 12 ? `\n… ${formatCount( list.length - 12 )} more` : '' );
				add( chip, lead.node );

			} );

		}

		const all = h( 'button.wgx-toggle', { type: 'button', 'aria-pressed': String( !! p.all ), title: 'Show every shortest path between the two nodes' }, iconEl( 'layers', 14 ), ! p.all ? 'All shortest paths' : p.pathCount === 1 ? 'The only shortest path' : `All ${formatCount( p.pathCount )} shortest paths` );
		all.addEventListener( 'click', () => I.showPath( p.source, p.target, { all: ! p.all } ) );
		// the chip that stands for the hops folded away before the target ( _fit )
		if ( items.length > 2 ) {

			const last = items[ items.length - 1 ];
			const chip = h( 'button.wgx-hop.wgx-hop-more', { type: 'button', hidden: true } );
			chip.addEventListener( 'click', () => {

				this._expanded = true;
				this._fit();
				items[ 1 ].chip.focus( { preventScroll: true } );

			} );
			const sep = h( 'span.wgx-hop-sep', { 'aria-hidden': 'true', hidden: true, html: xicon( 'next', 10 ) } );
			hops.insertBefore( chip, last.sep );
			hops.insertBefore( sep, chip );
			this._more = { chip, sep };

		} else this._more = null;

		this._hops = hops;
		this._items = items;
		el.append( h( 'span.wgx-path-len', null, h( 'b', null, formatCount( p.length ) ), p.length === 1 ? ' hop' : ' hops' ), hops, all, closeBtn( 'Close the path', () => I.clearPath() ) );
		this._fit();

	}

	/**
	 * Makes the hops fit: all of them when there is room; else the ones before the target fold
	 * into the "+n" chip, as few as needed; else the hops take a row of their own and fold there.
	 */
	_fit() {

		const el = this.el, hops = this._hops, items = this._items, more = this._more;
		if ( ! hops || ! hops.isConnected || el.hidden ) return;
		const n = items.length;
		// hops from … n − 2 folded away (from = n − 1: none)
		const fold = ( from ) => {

			const names = [];
			let current = false;
			for ( let k = 1; k < n - 1; k ++ ) {

				const away = k >= from;
				items[ k ].chip.hidden = away;
				items[ k ].sep.hidden = away;
				if ( ! away ) continue;
				names.push( items[ k ].label );
				if ( items[ k ].node === this.interaction.selectedIndex ) current = true;

			}

			if ( ! more ) return;
			more.chip.hidden = more.sep.hidden = names.length === 0;
			more.chip.textContent = `+${formatCount( names.length )}`;
			more.chip.title = names.length ? `${names.slice( 0, 12 ).join( ' › ' )}${names.length > 12 ? ' › …' : ''}\nClick to show ${names.length === 1 ? 'it' : 'them'}` : '';
			more.chip.setAttribute( 'aria-label', `${formatCount( names.length )} more ${names.length === 1 ? 'hop' : 'hops'}: ${names.slice( 0, 12 ).join( ', ' )}. Show ${names.length === 1 ? 'it' : 'them'}` );
			more.chip.setAttribute( 'aria-current', String( current ) );

		};

		const overflows = () => hops.scrollWidth > hops.clientWidth + 1;
		const foldToFit = () => {

			for ( let from = n - 2; from >= 1; from -- ) {

				fold( from );
				if ( ! overflows() ) return true;

			}

			return false;

		};

		// (asked for in full, the list keeps every hop and scrolls where it has to)
		el.classList.remove( 'is-stacked' );
		fold( n - 1 );
		if ( ! overflows() ) return;
		if ( ! this._expanded && foldToFit() ) return;
		el.classList.add( 'is-stacked' );
		fold( n - 1 );
		if ( ! this._expanded && overflows() ) foldToFit();

	}

}
