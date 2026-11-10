// Popover of the explore layer: a small glass menu that opens above (or below) the control that
// owns it, e.g. the metric menu of a lens channel. One is open at a time.
//
//   const pop = openPopover( { host, anchor, content, label, place = 'above', className, onClose } )
//        host      positioned element the popover is appended to (the bar the control sits in);
//                  it carries the class `has-pop` while a popover is open, so its stylesheet can
//                  lift it above neighbouring surfaces
//        anchor    the control: the popover is aligned with its left edge and kept inside the
//                  viewport (and inside the host when that is wide enough); it gets aria-expanded
//                  and the focus back when the popover closes
//        content   element or array of elements
//        place     'above' | 'below'
//   pop.el, pop.close( { focus = true } ), pop.reposition()
//   closePopover()   closes whatever is open
//
// Closing: Esc (handled in the capture phase and marked as handled, so the Esc of the canvas does
// not also peel a highlight layer), a press outside, the anchor pressed again, closePopover().
// Keys inside: ↑ / ↓ / Home / End move between the options ( [role="option"], [role="menuitemradio"],
// buttons ), Enter / Space activate (native button behaviour).

import { h } from './ui.js';

let current = null;

export function closePopover( opts ) {

	if ( current ) current.close( opts );

}

export function openPopover( { host, anchor, content, label = '', place = 'above', className = '', onClose = null } ) {

	closePopover( { focus: false } );
	const el = h( 'div.wgx-pop.wgv-glass' + ( className ? '.' + className : '' ), { role: 'dialog', 'aria-label': label, tabindex: '-1' }, content );
	el.dataset.place = place;
	host.append( el );
	host.classList.add( 'has-pop' ); // the host lifts itself above its neighbours while the menu is open
	anchor.setAttribute( 'aria-expanded', 'true' );

	const items = () => Array.from( el.querySelectorAll( '[role="option"], [role="menuitemradio"], button' ) ).filter( ( b ) => ! b.disabled && b.offsetParent !== null );
	const reposition = () => {

		const hr = host.getBoundingClientRect(), ar = anchor.getBoundingClientRect();
		const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
		const w = el.offsetWidth;
		// at the anchor's left edge, inside the viewport and, where the host is wide enough, inside the host
		const inHost = hr.width >= w ? hr.width - w : Infinity;
		const left = Math.max( 12 - hr.left, Math.min( ar.left - hr.left, vw - 12 - hr.left - w, inHost ) );
		el.style.left = Math.round( left ) + 'px';
		// stays clear of the top bar (search) when it opens upwards
		const room = place === 'above' ? hr.top - 68 : vh - hr.bottom - 20;
		el.style.maxHeight = Math.max( 160, Math.min( 560, Math.round( room ) ) ) + 'px';

	};

	const onKey = ( e ) => {

		if ( e.key === 'Escape' ) {

			e.preventDefault();
			e.stopPropagation();
			pop.close();
			return;

		}

		if ( ! el.contains( document.activeElement ) ) return;
		const move = e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? - 1 : e.key === 'Home' ? - Infinity : e.key === 'End' ? Infinity : 0;
		if ( ! move ) return;
		const list = items();
		if ( ! list.length ) return;
		e.preventDefault();
		const at = list.indexOf( document.activeElement );
		const next = move === - Infinity ? 0 : move === Infinity ? list.length - 1 : Math.max( 0, Math.min( list.length - 1, at + move ) );
		list[ next ].focus();

	};

	const onDown = ( e ) => {

		if ( el.contains( e.target ) ) return;
		// the anchor toggles by itself: closing here and reopening in its click handler would flicker
		pop.close( { focus: false, byAnchor: anchor.contains( e.target ) } );

	};

	const pop = {
		el, anchor, reposition,
		closedByAnchor: false,
		close( { focus = true, byAnchor = false } = {} ) {

			if ( current !== pop ) return;
			current = null;
			pop.closedByAnchor = byAnchor;
			window.removeEventListener( 'keydown', onKey, true );
			document.removeEventListener( 'pointerdown', onDown, true );
			window.removeEventListener( 'resize', reposition );
			anchor.setAttribute( 'aria-expanded', 'false' );
			el.remove();
			if ( ! host.querySelector( ':scope > .wgx-pop' ) ) host.classList.remove( 'has-pop' );
			if ( focus && document.contains( anchor ) ) anchor.focus( { preventScroll: true } );
			if ( onClose ) onClose( pop );

		},
	};

	current = pop;
	window.addEventListener( 'keydown', onKey, true );
	document.addEventListener( 'pointerdown', onDown, true );
	window.addEventListener( 'resize', reposition );
	reposition();
	// start at the chosen option, else at the first
	const list = items();
	const start = list.find( ( b ) => b.getAttribute( 'aria-checked' ) === 'true' || b.getAttribute( 'aria-selected' ) === 'true' ) || list[ 0 ];
	if ( start ) {

		start.focus( { preventScroll: true } );
		if ( start.scrollIntoView ) start.scrollIntoView( { block: 'nearest' } );

	} else el.focus( { preventScroll: true } );

	return pop;

}
