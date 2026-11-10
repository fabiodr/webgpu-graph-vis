// Tour: a short automatic visit that lets a first-time viewer understand the graph. It plays the
// stops of tourPlan.js: overview → the most connected node → the largest communities → the
// strongest broker → overview. Each stop moves the camera with the rig's own eased flights
// ( fitToGraph, focusNode, flyTo ), sets the lens to match, highlights its subject through the
// highlight model and shows a caption in slots.bottom (what it is and one number).
//
//  · Any pointer, wheel or key input outside the caption stops the tour where it is: the view
//    and the highlight of that moment stay, so the viewer can carry on from them (Esc and a
//    click on empty space only end the tour, they do not also clear the highlight; a click on a
//    node selects it).
//  · The lens is a guest setting: however the tour ends, the channels go back to what they were
//    before it, unless the viewer has changed them in the meantime. A tour that runs to its end
//    also clears the highlight and, when the rig can ( rig.pose / rig.setPose, feature-detected ),
//    returns the camera to where it was. A tour that is stopped leaves the camera to the viewer
//    and, with such a rig, offers the way back for a few seconds ( "Back to my view" ).
//  · Betweenness for the broker stop is requested when the tour starts and planned when the stop
//    is reached, with whatever estimate the worker has by then (said as "about" until exact).
//  · A new graph stops it.
//
//   const tour = new Tour( { app, lens, interaction, rig, slots, insights, touch } )
//   tour.start() -> Promise<boolean>   true when it ran to the end, false when it was stopped
//   tour.stop(), tour.next(), tour.previous(), tour.running, tour.step ( index, −1 = idle ), tour.stops
//   Event 'tour' on app.events: { running, step, stop }

import { h, xicon, iconEl } from './ui.js';
import { planTour } from './tourPlan.js';
import { applyChannels } from './presets.js';

export class Tour {

	constructor( { app, lens, interaction, rig, slots, insights } ) {

		this.app = app;
		this.lens = lens;
		this.interaction = interaction;
		this.rig = rig;
		this.insights = insights;
		this.running = false;
		this.step = - 1;
		this.stops = [];
		this._timer = 0;
		this._build();
		slots.bottom.append( this.el, this._back );

		this._onInput = ( e ) => {

			if ( ! this.running ) return;
			// the caption's own buttons (an event sent to the window itself has no node as its target)
			if ( e.target instanceof Node && this.el.contains( e.target ) ) return;
			if ( e.type === 'keydown' ) {

				if ( e.key === 'Shift' || e.key === 'Control' || e.key === 'Alt' || e.key === 'Meta' ) return;
				// Esc leaves the top-most thing, and that is the tour: the highlight under it stays
				if ( e.key === 'Escape' ) {

					e.preventDefault();
					e.stopPropagation();

				}

			} else if ( e.type === 'pointerdown' ) {

				// a press that ends the tour does not also wipe its picture (a click on empty space would)
				this.interaction.holdEmptyClick?.();

			}

			this.stop();

		};

		this._onGraph = () => {

			this.stop();
			this._offerBack( null ); // that view belonged to the graph that left

		};

		app.events.addEventListener( 'graph', this._onGraph );

	}

	start() {

		if ( this.running ) return this._done;
		const g = this.app.graph;
		if ( ! g ) return Promise.resolve( false );
		this.running = true;
		this._before = { ...this.lens.state };
		this._pose = typeof this.rig.pose === 'function' && typeof this.rig.setPose === 'function' ? this.rig.pose() : null;
		this._current = null;
		this._offerBack( null );
		this._graph = g;
		// the broker stop needs betweenness: start it now, it is ready (or far along) when we get there
		this.lens.request( 'betweenness' ).catch( () => {} );
		this._plan();
		// the sequence is fixed at the start; the broker stop is promised before its numbers are there
		this._sequence = this.stops.map( ( s ) => s.id );
		if ( ! this._sequence.includes( 'broker' ) && g.N > 2 && g.E > 0 && this._sequence.includes( 'outro' ) ) this._sequence.splice( this._sequence.indexOf( 'outro' ), 0, 'broker' );
		for ( const type of [ 'pointerdown', 'wheel', 'touchstart' ] ) window.addEventListener( type, this._onInput, { capture: true, passive: true } );
		window.addEventListener( 'keydown', this._onInput, { capture: true } );
		this.el.hidden = false;
		document.documentElement.classList.add( 'wgx-tour-on' );
		this._done = new Promise( ( resolve ) => ( this._resolve = resolve ) );
		this._show( 0 );
		return this._done;

	}

	stop() {

		this._end( false );

	}

	next() {

		if ( ! this.running ) return;
		if ( this.step + 1 >= this._sequence.length ) this._end( true );
		else this._show( this.step + 1, 1 );

	}

	previous() {

		if ( this.running && this.step > 0 ) this._show( this.step - 1, - 1 );

	}

	dispose() {

		this._end( false );
		this._offerBack( null );
		this.app.events.removeEventListener( 'graph', this._onGraph );
		this.el.remove();
		this._back.remove();

	}

	// ── internals ─────────────────────────────────────────────────────────────────────────

	_plan() {

		const v = this.lens.values( 'betweenness' );
		this.stops = planTour( this._graph, {
			betweenness: v ? v.raw : null, exact: v ? v.exact : true,
			communities: this.insights ? this.insights.communities() : null,
			touch: typeof matchMedia === 'function' && matchMedia( '(pointer: coarse)' ).matches,
		} );

	}

	/** Shows stop k of the sequence; a stop whose subject is not there (yet) is passed over. */
	_show( k, direction = 1 ) {

		clearTimeout( this._timer );
		this._plan(); // betweenness may have arrived since the last stop
		const seq = this._sequence;
		let stop = null;
		for ( ; k >= 0 && k < seq.length; k += direction ) {

			stop = this.stops.find( ( s ) => s.id === seq[ k ] ) || null;
			if ( stop ) break;

		}

		if ( ! stop ) return this._end( direction > 0 );
		this.step = k;
		this._current = stop;
		const I = this.interaction, rig = this.rig;

		applyChannels( this.lens, stop.lens );
		if ( stop.kind === 'overview' || stop.kind === 'outro' ) {

			I.clearAll();
			rig.fitToGraph();

		} else if ( stop.kind === 'community' ) {

			I.clearAll();
			I.focusCluster( stop.cluster );
			if ( this.insights ) this.insights.frameCluster( stop.cluster );

		} else {

			I.focusCluster( - 1 );
			I.select( stop.node, { ripple: true } );
			rig.focusNode( stop.node );

		}

		// caption
		this._stepEl.textContent = `${k + 1} / ${seq.length}`;
		this._titleEl.textContent = stop.title;
		this._textEl.textContent = stop.text;
		this._prev.disabled = k === 0;
		this._next.textContent = k + 1 >= seq.length ? 'Done' : 'Next';
		const bar = this._bar;
		bar.style.animation = 'none';
		void bar.offsetWidth; // restart the sweep
		bar.style.animation = `wgx-tour-sweep ${stop.seconds}s linear forwards`;
		this.app.events.dispatchEvent( new CustomEvent( 'tour', { detail: { running: true, step: k, stop } } ) );
		this._timer = setTimeout( () => this.next(), stop.seconds * 1000 );

	}

	_end( completed ) {

		if ( ! this.running ) return;
		this.running = false;
		clearTimeout( this._timer );
		for ( const type of [ 'pointerdown', 'wheel', 'keydown', 'touchstart' ] ) window.removeEventListener( type, this._onInput, { capture: true } );
		this.el.hidden = true;
		document.documentElement.classList.remove( 'wgx-tour-on' );
		this.step = - 1;
		if ( this.app.graph === this._graph ) {

			// the lens of a stop was the tour's, not the viewer's: back to what it was, unless the
			// viewer has set another one since
			const b = this._before, shown = this._current ? this._current.lens : null, s = this.lens.state;
			const same = ( x, y ) => ( x || null ) === ( y || null );
			if ( completed || ! shown || ( same( s.color, shown.color ) && same( s.size, shown.size ) && same( s.glow, shown.glow ) ) ) {

				applyChannels( this.lens, { color: b.color, size: b.size, glow: b.glow, ramp: b.ramp, reverse: b.reverse, top: b.top } );

			}

			const pose = this._pose;
			if ( completed ) {

				// a finished tour leaves no trace: nothing highlighted, the camera where it was
				this.interaction.clearAll();
				if ( pose ) this.rig.setPose( pose );

			} else if ( pose && moved( pose, this.rig.pose() ) ) this._offerBack( pose );

		}

		this._pose = null;

		this.app.events.dispatchEvent( new CustomEvent( 'tour', { detail: { running: false, step: - 1, stop: null, completed } } ) );
		const resolve = this._resolve;
		this._resolve = null;
		if ( resolve ) resolve( completed );

	}

	/** After a stopped tour: a short offer to return to the view from before it ( null: take it away ). */
	_offerBack( pose ) {

		clearTimeout( this._backTimer );
		this._backPose = pose;
		this._back.hidden = ! pose;
		if ( pose ) this._backTimer = setTimeout( () => this._offerBack( null ), 9000 );

	}

	_build() {

		const back = h( 'button.wgx-btn', { type: 'button' }, iconEl( 'back', 14 ), 'Back to my view' );
		back.addEventListener( 'click', () => {

			const pose = this._backPose;
			this._offerBack( null );
			if ( pose && ! this.running ) this.rig.setPose( pose );

		} );
		const dismiss = h( 'button.wgx-icon-btn', { type: 'button', 'aria-label': 'Dismiss', title: 'Dismiss', html: xicon( 'close', 14 ) } );
		dismiss.addEventListener( 'click', () => this._offerBack( null ) );
		this._back = h( 'div.wgx.wgx-tour-end.wgv-glass', { role: 'status', hidden: true }, iconEl( 'tour', 14 ), h( 'span.wgx-tour-end-msg', null, 'Tour ended' ), back, dismiss );

		this._stepEl = h( 'span.wgx-tour-step' );
		this._titleEl = h( 'h3.wgx-tour-title' );
		this._textEl = h( 'p.wgx-tour-text' );
		this._bar = h( 'i' );
		this._prev = h( 'button.wgx-icon-btn', { type: 'button', 'aria-label': 'Previous stop', title: 'Previous stop', html: xicon( 'prev', 14 ) } );
		this._prev.addEventListener( 'click', () => this.previous() );
		this._next = h( 'button.wgx-btn', { type: 'button' }, 'Next' );
		this._next.addEventListener( 'click', () => this.next() );
		const stop = h( 'button.wgx-btn', { type: 'button', title: 'Stop the tour (any click or key on the graph does too)' }, iconEl( 'stop', 14 ), 'Stop' );
		stop.addEventListener( 'click', () => this.stop() );
		this.el = h( 'div.wgx.wgx-tour.wgv-glass', { role: 'status', 'aria-live': 'polite', hidden: true },
			h( 'div.wgx-tour-main', null, h( 'div.wgx-tour-head', null, iconEl( 'tour', 14 ), this._stepEl, this._titleEl ), this._textEl ),
			h( 'div.wgx-tour-actions', null, this._prev, this._next, stop ),
			h( 'span.wgx-tour-bar', { 'aria-hidden': 'true' }, this._bar ) );

	}

}

/** Whether two camera poses ( rig.pose() ) differ by more than a nudge. */
function moved( a, b ) {

	if ( ! a || ! b || ! a.position || ! b.position || ! a.target || ! b.target ) return false;
	const d = ( p, q ) => Math.hypot( p[ 0 ] - q[ 0 ], p[ 1 ] - q[ 1 ], p[ 2 ] - q[ 2 ] );
	return d( a.position, b.position ) + d( a.target, b.target ) > 0.02 * Math.max( 1e-6, d( a.position, a.target ) );

}
