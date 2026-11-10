// Search and command palette (slots.top): one box that finds nodes by label and runs commands.
//
//  · Closed it is a quiet search field (the visible affordance); "/" or Ctrl / Cmd + K opens it.
//  · Typing ranks the node labels (fuzzy.js: exact > prefix > word start > substring >
//    abbreviation, ties by degree) and the commands of the registry together ( mergeRanked ): the
//    better tier first, in a tier the nodes before the commands, so Enter goes to a node whenever
//    a node matches as well as the best command; where a tier holds both, its commands come after
//    its five best nodes and the rest of its nodes after them. A leading ">" lists commands
//    only. Nothing typed: the nodes selected last (or the most connected ones) and suggested
//    commands.
//  · A row shows the node's community colour, its label with the matched characters emphasised
//    and its degree; a command its title, group and shortcut.
//  · Arrow keys move, Enter selects the node and flies to it ( rig.focusNode ) or runs the
//    command, Esc closes. The row under the pointer or the cursor is previewed in the graph
//    ( interaction.setPreview: the node lights up and gets its label, the camera stays ).
//    The ranked nodes become the step list of [ and ].
//  · While a path target is being picked ( interaction.pathPickFrom ) Enter on a node completes
//    the path instead of selecting.
//  · Speed: the index is built once per graph (on idle after the graph arrived, at the latest
//    with the first query). A keystroke is one scan of the folded labels and a refill of the
//    visible rows only: the list is a fixed pool of row elements moved over a spacer, whatever
//    the number of matches. stats.ms is the time the last keystroke took (query + rows).
//
//   const search = new Search( { app, interaction, rig, lens, slots, commands } )
//   search.open( text = '' ) / close() / toggle() / isOpen
//                                   Closing by Esc, Enter or the close button gives the keyboard
//                                   focus back to what had it before the box opened.
//   search.setGraph( graph )        new graph: the index and the recent nodes are dropped
//   search.stats                    { ms, queryMs, matches, nodes } of the last keystroke
//   search.dispose()

import { FuzzyIndex, matchRanges, mergeRanked } from './fuzzy.js';
import { labelOf } from './communities.js';
import { formatCount } from './format.js';
import { h, xicon, kbd, nodeCss, coarsePointer, isPhone, onOutside } from './ui.js';

const MAX_NODE_ROWS = 200;
const MAX_VISIBLE = 9;
let uid = 0;

export class Search {

	constructor( { app, interaction, rig, lens, slots, commands } ) {

		this.app = app;
		this.interaction = interaction;
		this.rig = rig;
		this.lens = lens;
		this.commands = commands;
		this.graph = null;
		this.isOpen = false;
		this.stats = { ms: 0, queryMs: 0, matches: 0, nodes: 0 };
		this._index = null;
		this._recent = [];
		this._rows = [];
		this._active = - 1;
		this._pool = [];
		this._results = []; // node indices of the current ranking (for the step list)
		this._id = 'wgx-search-' + ( ++ uid );
		this._build();
		slots.top.append( this.el );

		this._onSelect = ( e ) => {

			const i = e.detail ? e.detail.index : - 1;
			if ( i >= 0 ) this._recent = [ i, ...this._recent.filter( ( r ) => r !== i ) ].slice( 0, 6 );

		};

		this._onPick = () => this._syncPlaceholder();
		this._onKey = ( e ) => {

			if ( ( e.ctrlKey || e.metaKey ) && ! e.altKey && ! e.shiftKey && ( e.key === 'k' || e.key === 'K' ) ) {

				e.preventDefault();
				// photo mode hides every surface: a search nobody can see must not take the keys
				if ( document.documentElement.classList.contains( 'wgv-ui-hidden' ) ) return;
				this.toggle();

			}

		};

		app.events.addEventListener( 'select', this._onSelect );
		app.events.addEventListener( 'pathpick', this._onPick );
		window.addEventListener( 'keydown', this._onKey, true );
		this._offCommands = commands.onChange( () => {

			if ( this.isOpen ) this._update();

		} );

	}

	setGraph( graph ) {

		this.graph = graph;
		this._index = null;
		this._recent = [];
		this._results = [];
		clearTimeout( this._warm );
		// build the index when the main thread has a moment, so the first keystroke is not the one that pays
		if ( graph ) this._warm = setTimeout( () => this._ensureIndex(), 400 );
		if ( this.isOpen ) this._update();

	}

	open( text = '', from = document.activeElement ) {

		if ( typeof text !== 'string' ) text = '';
		this.input.value = text;
		if ( ! this.isOpen ) {

			this.isOpen = true;
			// what had the keyboard focus gets it back when the box closes
			this._returnTo = from && from !== document.body && from.focus && ! this.el.contains( from ) ? from : null;
			this.el.classList.add( 'is-open' );
			this.input.setAttribute( 'aria-expanded', 'true' );
			this._offOutside = onOutside( this.el, () => this.close( { restore: false } ) ); // that press moves the focus itself
			this._rowH = coarsePointer() ? 44 : 34;
			this.el.style.setProperty( '--wgx-row-h', this._rowH + 'px' );
			this._syncPlaceholder();

		}

		this._update();
		this.input.focus( { preventScroll: true } );
		this.input.setSelectionRange( text.length, text.length );

	}

	/** blur: take the focus out of the field; restore: hand it back to what had it before open(). */
	close( { blur = true, restore = blur } = {} ) {

		if ( ! this.isOpen ) return;
		this.isOpen = false;
		const back = this._returnTo;
		this._returnTo = null;
		this.el.classList.remove( 'is-open' );
		this.input.setAttribute( 'aria-expanded', 'false' );
		this.input.removeAttribute( 'aria-activedescendant' );
		this.input.value = '';
		this._offOutside?.();
		this._offOutside = null;
		this.interaction.setPreview( - 1 );
		this._syncPlaceholder();
		const at = document.activeElement;
		if ( ! blur || ! at || ! this.el.contains( at ) ) return;
		// back to where the keyboard was, when that is still on screen
		if ( restore && back && back.isConnected && back.getClientRects().length ) back.focus( { preventScroll: true } );
		if ( document.activeElement === at ) at.blur();

	}

	toggle() {

		if ( this.isOpen ) this.close();
		else this.open();

	}

	dispose() {

		clearTimeout( this._warm );
		this.close();
		this.app.events.removeEventListener( 'select', this._onSelect );
		this.app.events.removeEventListener( 'pathpick', this._onPick );
		window.removeEventListener( 'keydown', this._onKey, true );
		this._offCommands?.();
		this.el.remove();

	}

	// ── internals ─────────────────────────────────────────────────────────────────────────

	_build() {

		this.input = h( 'input.wgx-search-input', {
			type: 'text', role: 'combobox', autocomplete: 'off', autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false', enterkeyhint: 'go',
			'aria-label': 'Search nodes and commands', 'aria-expanded': 'false', 'aria-autocomplete': 'list', 'aria-controls': this._id + '-list',
		} );
		this._hint = h( 'span.wgx-search-hint', { 'aria-hidden': 'true' }, kbd( '/' ) );
		this._clear = h( 'button.wgx-search-close', { type: 'button', 'aria-label': 'Close search', html: xicon( 'close', 14 ) } );
		this._clear.addEventListener( 'click', () => this.close() );
		const box = h( 'label.wgx-search-box.wgv-glass', null, h( 'span.wgx-search-ico', { html: xicon( 'search' ), 'aria-hidden': 'true' } ), this.input, this._hint, this._clear );

		this._spacer = h( 'div.wgx-search-spacer' );
		this._list = h( 'div.wgx-search-list', { role: 'listbox', id: this._id + '-list', 'aria-label': 'Results' }, this._spacer );
		this._foot = h( 'div.wgx-search-foot', { 'aria-hidden': 'true' },
			h( 'span', null, kbd( '↑' ), kbd( '↓' ), ' move' ), h( 'span', null, kbd( '↵' ), ' go' ), h( 'span', null, kbd( '>' ), ' commands' ), h( 'span', null, kbd( 'esc' ), ' close' ) );
		this._panel = h( 'div.wgx-search-panel.wgv-glass', null, this._list, this._foot );
		this.el = h( 'div.wgx.wgx-search', { role: 'search' }, box, this._panel );
		this._syncPlaceholder();

		this.input.addEventListener( 'focus', ( e ) => {

			if ( ! this.isOpen ) this.open( this.input.value, e.relatedTarget );

		} );
		this.input.addEventListener( 'input', () => this._update() );
		this.input.addEventListener( 'keydown', ( e ) => this._keyDown( e ) );
		this._list.addEventListener( 'scroll', () => this._renderRows(), { passive: true } );
		this._list.addEventListener( 'pointermove', ( e ) => {

			const k = this._rowAt( e.target );
			if ( k >= 0 && k !== this._active && this._selectable( k ) ) this._setActive( k, false );

		} );
		this._list.addEventListener( 'pointerleave', () => this.interaction.setPreview( - 1 ) );
		this._list.addEventListener( 'click', ( e ) => {

			const k = this._rowAt( e.target );
			if ( k >= 0 && this._selectable( k ) ) this._activate( k );

		} );
		// a press on a row must not take the focus away from the input
		this._list.addEventListener( 'pointerdown', ( e ) => e.preventDefault() );

	}

	_syncPlaceholder() {

		const picking = this.interaction.pathPickFrom >= 0;
		// the closed pill of a phone has room for one word
		const phone = isPhone();
		this.input.placeholder = picking ? ( phone ? 'Path target…' : 'Search the path target…' ) : phone && ! this.isOpen ? 'Search' : 'Search nodes, run commands';
		this.el.classList.toggle( 'is-picking', picking );

	}

	_ensureIndex() {

		const g = this.graph;
		if ( ! g ) return null;
		if ( ! this._index ) {

			const labelled = !! g.labels;
			this._index = new FuzzyIndex( g.N, ( i ) => labelOf( g, i ), { weights: g.degree, numeric: ! labelled } );

		}

		return this._index;

	}

	/** Recomputes the rows for the text in the box. */
	_update() {

		const t0 = performance.now();
		const g = this.graph;
		const raw = this.input.value;
		const commandsOnly = raw.trimStart().startsWith( '>' );
		const query = commandsOnly ? raw.trimStart().slice( 1 ).trim() : raw.trim();
		const rows = [];
		this._results = [];
		let matches = 0;
		this._query = commandsOnly ? '' : query;

		if ( commandsOnly ) {

			const list = query ? this.commands.search( query, { limit: 50 } ).map( ( r ) => r.command ) : this.commands.available();
			if ( list.length ) rows.push( { type: 'head', text: 'Commands', meta: formatCount( list.length ) } );
			for ( const command of list ) rows.push( { type: 'command', command } );
			if ( ! list.length ) rows.push( { type: 'note', text: `No command matches “${query}”.` } );

		} else if ( ! query ) {

			if ( g ) {

				const recent = this._recent.filter( ( i ) => i < g.N );
				if ( recent.length ) {

					rows.push( { type: 'head', text: 'Recent', meta: 'degree' } );
					for ( const node of recent.slice( 0, 5 ) ) rows.push( { type: 'node', node } );

				} else if ( g.N > 0 ) {

					const index = this._ensureIndex();
					rows.push( { type: 'head', text: 'Most connected', meta: 'degree' } );
					for ( const node of index.top( 5 ) ) rows.push( { type: 'node', node } );

				}

			}

			const suggested = this.commands.suggested( 6 );
			if ( suggested.length ) rows.push( { type: 'head', text: 'Commands', meta: 'type > for all' } );
			for ( const command of suggested ) rows.push( { type: 'command', command } );

		} else {

			const index = this._ensureIndex();
			const found = index ? index.search( query, { limit: MAX_NODE_ROWS } ) : { items: [], tiers: [], total: 0 };
			const cmds = this.commands.search( query, { limit: 5 } );
			matches = found.total;
			this._results = Array.from( found.items );
			// one ranking ( mergeRanked ): the better tier first, in a tier the nodes before the commands
			let headed = false;
			for ( const block of mergeRanked( found.tiers, cmds.map( ( r ) => r.tier ) ) ) {

				if ( block.kind === 'node' ) {

					rows.push( { type: 'head', text: headed ? 'More nodes' : found.total > found.items.length ? `Nodes · best ${formatCount( found.items.length )} of ${formatCount( found.total )}` : `Nodes · ${formatCount( found.total )}`, meta: 'degree' } );
					headed = true;
					for ( let k = block.from; k < block.to; k ++ ) rows.push( { type: 'node', node: found.items[ k ], position: k } );

				} else {

					rows.push( { type: 'head', text: 'Commands', meta: '' } );
					for ( let k = block.from; k < block.to; k ++ ) rows.push( { type: 'command', command: cmds[ k ].command } );

				}

			}

			if ( ! rows.length ) rows.push( { type: 'note', text: `Nothing matches “${query}”.` } );

		}

		this._rows = rows;
		const tq = performance.now();
		const first = rows.findIndex( ( r ) => r.type === 'node' || r.type === 'command' );
		this._list.scrollTop = 0;
		this._layout();
		this._setActive( first, false, ! query ); // nothing typed: no preview until the user moves
		this._renderRows( true );
		const t1 = performance.now();
		this.stats = { ms: t1 - t0, queryMs: tq - t0, matches, nodes: g ? g.N : 0 };

	}

	_layout() {

		const n = this._rows.length, rowH = this._rowH || 34;
		const visible = Math.min( n, MAX_VISIBLE );
		this._list.style.height = visible * rowH + 'px';
		this._spacer.style.height = n * rowH + 'px';
		const need = Math.min( n, MAX_VISIBLE + 2 );
		while ( this._pool.length < need ) {

			const el = h( 'div.wgx-row', { role: 'option' },
				h( 'span.wgx-row-lead', { 'aria-hidden': 'true' } ), h( 'span.wgx-row-label' ), h( 'span.wgx-row-meta' ) );
			el._slot = this._pool.length;
			this._pool.push( el );
			this._list.append( el );

		}

	}

	/** Fills the pooled row elements with the rows in view. */
	_renderRows( force = false ) {

		const rowH = this._rowH || 34, n = this._rows.length;
		const first = Math.max( 0, Math.min( Math.max( 0, n - this._pool.length ), Math.floor( this._list.scrollTop / rowH ) - 1 ) );
		for ( let s = 0; s < this._pool.length; s ++ ) {

			const el = this._pool[ s ], k = first + s;
			if ( k >= n ) {

				el.hidden = true;
				el._k = - 1;
				continue;

			}

			el.hidden = false;
			if ( force || el._k !== k || el._row !== this._rows[ k ] ) this._fill( el, k );
			el.classList.toggle( 'is-active', k === this._active );
			el.setAttribute( 'aria-selected', k === this._active ? 'true' : 'false' );

		}

	}

	_fill( el, k ) {

		const row = this._rows[ k ], g = this.graph;
		const [ lead, label, meta ] = el.children;
		el._k = k;
		el._row = row;
		el.id = `${this._id}-row-${k}`;
		el.style.transform = `translateY(${k * ( this._rowH || 34 )}px)`;
		el.className = 'wgx-row is-' + row.type;
		lead.textContent = '';
		lead.removeAttribute( 'style' );
		label.textContent = '';
		meta.textContent = '';
		el.removeAttribute( 'title' );
		if ( row.type === 'node' ) {

			const text = labelOf( g, row.node );
			lead.style.setProperty( '--wgx-dot', nodeCss( g, this.lens, row.node ) );
			emphasise( label, text, this._query );
			meta.textContent = formatCount( g.degree[ row.node ] );
			if ( text.length > 40 ) el.title = text;

		} else if ( row.type === 'command' ) {

			const c = row.command;
			lead.innerHTML = xicon( c.icon || 'command', 14 );
			emphasise( label, c.title, this._query );
			if ( c.hint || c.group ) label.append( h( 'span.wgx-row-sub', null, c.hint || c.group ) );
			if ( c.shortcut ) meta.append( ...String( c.shortcut ).split( ' ' ).map( ( s ) => kbd( s ) ) );

		} else {

			label.textContent = row.text;
			meta.textContent = row.meta || '';

		}

	}

	_selectable( k ) {

		const r = this._rows[ k ];
		return !! r && ( r.type === 'node' || r.type === 'command' );

	}

	_rowAt( target ) {

		const el = target && target.closest ? target.closest( '.wgx-row' ) : null;
		return el && ! el.hidden && typeof el._k === 'number' ? el._k : - 1;

	}

	_setActive( k, scroll = true, quiet = false ) {

		this._active = k;
		const row = this._rows[ k ];
		if ( scroll && k >= 0 ) {

			const rowH = this._rowH || 34, top = k * rowH, list = this._list;
			// keep a header above the first row of its section in view
			const lead = k > 0 && this._rows[ k - 1 ].type === 'head' ? rowH : 0;
			if ( top - lead < list.scrollTop ) list.scrollTop = top - lead;
			else if ( top + rowH > list.scrollTop + list.clientHeight ) list.scrollTop = top + rowH - list.clientHeight;

		}

		if ( k >= 0 ) this.input.setAttribute( 'aria-activedescendant', `${this._id}-row-${k}` );
		else this.input.removeAttribute( 'aria-activedescendant' );
		if ( ! quiet ) this.interaction.setPreview( row && row.type === 'node' ? row.node : - 1 );
		this._renderRows();

	}

	/** Moves the cursor by `delta` selectable rows (headers are skipped). */
	_move( delta ) {

		const n = this._rows.length, dir = Math.sign( delta );
		let k = this._active, target = k;
		for ( let steps = Math.abs( delta ); steps > 0; ) {

			k += dir;
			if ( k < 0 || k >= n ) break;
			if ( this._selectable( k ) ) {

				target = k;
				steps --;

			}

		}

		if ( target !== this._active && this._selectable( target ) ) this._setActive( target );

	}

	_keyDown( e ) {

		switch ( e.key ) {

			case 'ArrowDown': e.preventDefault(); this._move( 1 ); break;
			case 'ArrowUp': e.preventDefault(); this._move( - 1 ); break;
			case 'PageDown': e.preventDefault(); this._move( MAX_VISIBLE - 1 ); break;
			case 'PageUp': e.preventDefault(); this._move( - ( MAX_VISIBLE - 1 ) ); break;
			case 'Enter':
				e.preventDefault();
				if ( this._active >= 0 ) this._activate( this._active );
				break;
			case 'Escape':
				e.preventDefault();
				e.stopPropagation();
				this.close();
				break;
			case 'Tab': this.close( { blur: false } ); break;

		}

	}

	_activate( k ) {

		const row = this._rows[ k ];
		if ( ! row ) return;
		if ( row.type === 'node' ) {

			const node = row.node, results = this._results.slice(), position = row.position;
			const I = this.interaction;
			this.close();
			if ( I.pathPickFrom >= 0 ) {

				I.completePathPick( node );
				return;

			}

			I.select( node, { focus: true, ripple: true } );
			if ( results.length > 1 && position !== undefined ) I.setStepList( results, { position, label: 'search results' } );

		} else if ( row.type === 'command' ) {

			const id = row.command.id;
			this.close();
			this.commands.run( id );

		}

	}

}

/** Fills `el` with `text`, the characters the query matched in <b>. */
function emphasise( el, text, query ) {

	const ranges = query ? matchRanges( text, query ) : [];
	if ( ! ranges.length ) {

		el.textContent = text;
		return;

	}

	let at = 0;
	for ( const [ a, b ] of ranges ) {

		if ( a > at ) el.append( text.slice( at, a ) );
		el.append( h( 'b', null, text.slice( a, b ) ) );
		at = b;

	}

	if ( at < text.length ) el.append( text.slice( at ) );

}
