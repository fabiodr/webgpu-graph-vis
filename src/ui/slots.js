// Screen layout of the floating UI. Surfaces (search, lens bar, node card, drawers, status chips,
// tool buttons) do not position themselves: they are appended to one of a few fixed "slots" around
// the canvas, flex containers that keep them from overlapping and move them together when the
// settings panel changes size or, on phones, when the bottom sheet opens.
//
//   const slots = uiSlots();            // singleton; creates the containers on first use
//   slots.top        top centre of the free area (row)             search
//   slots.topRight   top right corner (row, right aligned)         tool buttons; arrange with CSS `order`
//   slots.right      right column under the top row (column)       drawers; class `wgv-slot-end` pushes a
//                                                                  child to the bottom (node card)
//   slots.bottom     bottom centre of the free area (column,       lens bar, path strip, captions
//                    first child lowest)
//   slots.status     bottom left of the free area (row)            status chips
//   slots.setInsets( { left, right, top, bottom } )   CSS px the settings panel covers at rest
//                                                      (Panel.insets()); sets --wgv-inset-* on <html>
//
// Children get pointer events, the containers do not, so the canvas stays reachable between them.
// CSS variables published on <html>: --wgv-inset-left / -right / -top / -bottom, --wgv-bottom-h
// (current height of the bottom slot, for surfaces that must stay clear of it on phones),
// --wgv-bottom-clear (that height plus a gap, 0 while the slot is empty: the status chips sit
// above it), --wgv-tools-w (current width of the top right slot: the top slot leaves that room free); the
// settings panel publishes --wgv-sheet-h (visible height of the phone bottom sheet).
// <html class="wgv-ui-hidden"> (photo mode) hides the slots root and everything in it.
import './slots.css';

let instance = null;

export function uiSlots( container = document.body ) {

	if ( instance ) return instance;
	const root = document.createElement( 'div' );
	root.className = 'wgv-slots';
	const make = ( name ) => {

		const el = document.createElement( 'div' );
		el.className = 'wgv-slot wgv-slot-' + name;
		root.appendChild( el );
		return el;

	};

	const slots = {
		root,
		top: make( 'top' ),
		topRight: make( 'top-right' ),
		right: make( 'right' ),
		bottom: make( 'bottom' ),
		status: make( 'status' ),
		setInsets( insets = {} ) {

			const style = document.documentElement.style;
			for ( const side of [ 'left', 'right', 'top', 'bottom' ] ) style.setProperty( '--wgv-inset-' + side, ( insets[ side ] || 0 ) + 'px' );

		},
		dispose() {

			observer?.disconnect();
			root.remove();
			instance = null;

		},
	};
	container.appendChild( root );
	const observer = typeof ResizeObserver === 'function' ? new ResizeObserver( () => {

		const style = document.documentElement.style;
		const h = slots.bottom.offsetHeight;
		style.setProperty( '--wgv-bottom-h', h + 'px' );
		style.setProperty( '--wgv-bottom-clear', ( h > 0 ? h + 8 : 0 ) + 'px' );
		style.setProperty( '--wgv-tools-w', slots.topRight.offsetWidth + 'px' );

	} ) : null;
	observer?.observe( slots.bottom );
	observer?.observe( slots.topRight );
	instance = slots;
	return slots;

}
