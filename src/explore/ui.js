// Shared DOM helpers of the explore layer: the element builder and base icons of the settings
// panel ( h, icon ), the layer's own stroke icons, colours for chips and a few environment checks.
//
//   h( 'div.cls', { attr }, ...children ), icon( name )     from src/ui/Panel.js
//   xicon( name, size = 16 ) -> SVG markup (string) of an explore icon, same 16 px stroke style
//   iconEl( name, size ) -> <span class="wgx-ico"> element holding it
//   kbd( text ) -> <kbd class="wgx-kbd">
//   cssOfLinear( r, g, b, lift = true ) -> CSS colour of a linear RGB triple; `lift` brightens it
//        like the glowing nodes (the stored palette is dark), so a chip matches the node on screen
//   nodeCss( graph, lens, i ) -> CSS colour of node i's community (the cluster palette, also while
//        a metric colours the nodes)
//   clusterCss( graph, lens, cluster ) -> CSS colour of a cluster of the prepared graph
//   coarsePointer(), reducedMotion(), isPhone() -> booleans (media queries, read when called)
//   onOutside( el, fn ) -> () => void: calls fn( event ) for a pointerdown outside el; returns the remover

import { h, icon, paletteCss } from '../ui/Panel.js';
import { linearToSrgb } from '../util/color.js';

export { h, icon };

const PATHS = {
	search: '<circle cx="7" cy="7" r="4.3"/><path d="M10.2 10.2l3.3 3.3"/>',
	command: '<path d="M4.5 4.5L8 8l-3.5 3.5M9 11.5h3"/>',
	insights: '<path d="M2.8 13.2h10.4M4.6 13V9M8 13V4.2M11.4 13V6.8"/>',
	color: '<path d="M8 2.4c2.2 2.6 3.7 4.5 3.7 6.6a3.7 3.7 0 0 1-7.4 0c0-2.1 1.5-4 3.7-6.6z"/>',
	size: '<circle cx="5" cy="10" r="2.2"/><circle cx="10.6" cy="6.6" r="3.6"/>',
	glow: '<circle cx="8" cy="8" r="2.6"/><path d="M8 1.8v1.7M8 12.5v1.7M1.8 8h1.7M12.5 8h1.7M3.6 3.6l1.2 1.2M11.2 11.2l1.2 1.2M12.4 3.6l-1.2 1.2M4.8 11.2l-1.2 1.2"/>',
	edges: '<circle cx="3.4" cy="11.6" r="1.5"/><circle cx="12.6" cy="4.4" r="1.5"/><path d="M4.7 10.8C8 9.6 8 6.4 11.3 5.2"/>',
	pin: '<path d="M9.8 2.6l3.6 3.6-2 .6-2.3 2.3.2 2.6-1.4 1.4-2.4-2.4L3 13.2M8 8.3l-2.4-2.4 1.4-1.4 2.6.2 2.3-2.3z"/>',
	focus: '<circle cx="8" cy="8" r="2.2"/><path d="M8 1.8v2.4M8 11.8v2.4M1.8 8h2.4M11.8 8h2.4"/>',
	ripple: '<circle cx="8" cy="8" r="1.3" fill="currentColor" stroke="none"/><circle cx="8" cy="8" r="3.6"/><path d="M3.1 12.9a6.9 6.9 0 0 1 0-9.8M12.9 3.1a6.9 6.9 0 0 1 0 9.8"/>',
	radial: '<circle cx="8" cy="8" r="1.4"/><circle cx="8" cy="8" r="5.8"/><path d="M8 2.2v1.6M13.8 8h-1.6M8 13.8v-1.6M2.2 8h1.6"/>',
	path: '<circle cx="3.4" cy="12.2" r="1.5"/><circle cx="12.6" cy="3.8" r="1.5"/><path d="M4.9 12.2h3.4a2.1 2.1 0 0 0 0-4.2H7.7a2.1 2.1 0 0 1 0-4.2h3.4"/>',
	hops: '<circle cx="8" cy="8" r="1.4" fill="currentColor" stroke="none"/><circle cx="8" cy="8" r="3.4"/><circle cx="8" cy="8" r="6"/>',
	tour: '<circle cx="8" cy="8" r="6"/><path d="M10.6 5.4L9.3 9.3 5.4 10.6 6.7 6.7z"/>',
	check: '<path d="M3.5 8.4l3 3 6-6.6"/>',
	swap: '<path d="M3 5.6h9.4M10 3.2l2.4 2.4L10 8M13 10.4H3.6M6 8l-2.4 2.4L6 12.8"/>',
	back: '<path d="M7 3.8L2.8 8 7 12.2M3 8h10.2"/>',
	next: '<path d="M6 3.8L10.2 8 6 12.2"/>',
	prev: '<path d="M10 3.8L5.8 8l4.2 4.2"/>',
	arrow: '<path d="M3 8h9.6M9 4.4L12.6 8 9 11.6"/>',
	stop: '<rect x="4.6" y="4.6" width="6.8" height="6.8" rx="1.2"/>',
	layers: '<path d="M8 2.6l5.6 3L8 8.6l-5.6-3zM2.4 8.4L8 11.4l5.6-3M2.4 11L8 14l5.6-3"/>',
	node: '<circle cx="8" cy="8" r="3.2"/>',
	more: '<circle cx="3.6" cy="8" r=".9" fill="currentColor" stroke="none"/><circle cx="8" cy="8" r=".9" fill="currentColor" stroke="none"/><circle cx="12.4" cy="8" r=".9" fill="currentColor" stroke="none"/>',
	off: '<circle cx="8" cy="8" r="5.6"/><path d="M4.1 11.9l7.8-7.8"/>',
	community: '<circle cx="5.2" cy="5.6" r="2"/><circle cx="11" cy="6.4" r="1.6"/><circle cx="7.6" cy="11" r="1.8"/>',
	spark: '<path d="M8 2.5l1.2 3.3 3.3 1.2-3.3 1.2L8 11.5 6.8 8.2 3.5 7l3.3-1.2zM12.5 11l.5 1.3 1.3.5-1.3.5-.5 1.3-.5-1.3-1.3-.5 1.3-.5z"/>',
	collapse: '<path d="M4 6l4 4 4-4"/>',
	expand: '<path d="M4 10l4-4 4 4"/>',
};

export function xicon( name, size = 16 ) {

	if ( ! PATHS[ name ] ) return icon( name, size );
	return `<svg class="wgv-i wgx-i" width="${size}" height="${size}" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[ name ]}</svg>`;

}

export function iconEl( name, size = 16 ) {

	return h( 'span.wgx-ico', { html: xicon( name, size ), 'aria-hidden': 'true' } );

}

export function kbd( text ) {

	return h( 'kbd.wgx-kbd', null, text );

}

export function cssOfLinear( r, g, b, lift = true ) {

	const m = Math.max( r, g, b, 1e-6 );
	const gain = lift ? Math.min( 4, 0.82 / m ) : 1;
	const c = [ r, g, b ].map( ( v ) => Math.round( 255 * linearToSrgb( Math.min( 1, Math.max( 0, v * gain ) ) ) ) );
	return `rgb(${c[ 0 ]} ${c[ 1 ]} ${c[ 2 ]})`;

}

const paletteOf = ( graph, lens ) => ( lens && lens.state && lens.state.palette ) || ( graph && graph.palette ) || 'paper';

export function nodeCss( graph, lens, i ) {

	return paletteCss( paletteOf( graph, lens ), graph.leafRank[ i ] );

}

export function clusterCss( graph, lens, cluster ) {

	const r = graph.clusterRankRange;
	return paletteCss( paletteOf( graph, lens ), ( r[ 2 * cluster ] + r[ 2 * cluster + 1 ] ) / 2 );

}

const media = ( q ) => typeof matchMedia === 'function' && matchMedia( q ).matches;
export const coarsePointer = () => media( '(pointer: coarse)' );
export const reducedMotion = () => media( '(prefers-reduced-motion: reduce)' );
export const isPhone = () => media( '(max-width: 640px)' );

export function onOutside( el, fn ) {

	const handler = ( e ) => {

		if ( ! el.contains( e.target ) ) fn( e );

	};

	document.addEventListener( 'pointerdown', handler, true );
	return () => document.removeEventListener( 'pointerdown', handler, true );

}
