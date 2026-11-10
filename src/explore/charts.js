// Small inline SVG charts of the insights drawer (degree histogram, k-core profile) and the
// binning behind them. No chart library: a bar chart is a few rects, two hairlines and a handful
// of labels.
//
// ─── Binning (pure) ─────────────────────────────────────────────────────────────────────────
//  binCounts( counts, { maxBars = 14, mode = 'auto' } ) -> [ { from, to, count } ]
//      counts[ v ] = how many items have the integer value v (a degree histogram, the sizes of
//      the k-shells). Leading and trailing zeros are dropped; the bins cover from the first to
//      the last value that occurs, without gaps, and their counts add up to the total.
//      'unit'    one bar per value (used when they fit into maxBars)
//      'log'     0, 1, 2, then doubling widths 3–4, 5–8, 9–16, … (heavy tails: a degree
//                distribution); widened to steps of four when even that does not fit
//      'linear'  equal widths
//      'auto'    unit when it fits, log when the largest value is at least eight times the
//                smallest (counting from 1), else linear
//  niceCeil( v ) -> the next "round" axis maximum (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 × 10^k, an
//      integer for counts): at most half as much again as the value
//
// ─── Chart (DOM) ────────────────────────────────────────────────────────────────────────────
//  barChart( bins, { width = 304, height = 104, name, unit = 'nodes', total, describe, idle } )
//      -> HTMLElement (a <figure>): the SVG, a readout line under it and a visually hidden table.
//      One series, one colour: columns at most 24 px wide with a 2 px gap, round at the data end
//      and square on the baseline, hairline grid, labels in text colours. Every bar is a hit
//      target as tall as the plot; hovering, focusing or tapping it writes its range, count and
//      share into the readout ( describe( bin ) -> "degree 3–4" names the range ), which shows
//      `idle` otherwise. The chart is ONE tab stop (roving tabindex): ← / → / Home / End move
//      between its bars. The largest bar carries its value.

import { formatRange, formatCount, compactCount, formatPercent } from './format.js';

export function binCounts( counts, { maxBars = 14, mode = 'auto' } = {} ) {

	let lo = 0, hi = counts ? counts.length - 1 : - 1;
	while ( lo <= hi && ! ( counts[ lo ] > 0 ) ) lo ++;
	while ( hi >= lo && ! ( counts[ hi ] > 0 ) ) hi --;
	if ( hi < lo ) return [];
	const bins = [];
	const push = ( from, to ) => {

		to = Math.min( to, hi );
		let count = 0;
		for ( let v = from; v <= to; v ++ ) count += counts[ v ] || 0;
		bins.push( { from, to, count } );

	};

	const span = hi - lo + 1;
	if ( mode === 'unit' || ( mode === 'auto' && span <= maxBars ) ) {

		for ( let v = lo; v <= hi; v ++ ) push( v, v );
		return bins;

	}

	if ( mode === 'auto' ) mode = hi >= 8 * Math.max( 1, lo ) ? 'log' : 'linear';
	if ( mode === 'log' ) {

		for ( let base = 2; ; base *= 2 ) {

			bins.length = 0;
			// single values up to 2, then upper edges at powers of `base`
			let from = lo;
			for ( ; from <= Math.min( 2, hi ); from ++ ) push( from, from );
			let edge = base;
			while ( edge < from ) edge *= base;
			for ( ; from <= hi; edge *= base ) {

				push( from, edge );
				from = edge + 1;

			}

			if ( bins.length <= maxBars || base > 1024 ) return bins;

		}

	}

	const width = Math.ceil( span / maxBars );
	for ( let from = lo; from <= hi; from += width ) push( from, from + width - 1 );
	return bins;

}

// steps close enough that the tallest bar fills at least two thirds of the plot
const STEPS = [ 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10 ];

export function niceCeil( v ) {

	if ( ! ( v > 0 ) ) return 1;
	const exp = Math.pow( 10, Math.floor( Math.log10( v ) ) );
	const f = v / exp;
	const step = STEPS.find( ( s ) => f <= s );
	return Math.max( 1, Math.ceil( step * exp - 1e-9 ) );

}

// ── DOM ──────────────────────────────────────────────────────────────────────────────────────

const NS = 'http://www.w3.org/2000/svg';

function svg( tag, attrs, text ) {

	const el = document.createElementNS( NS, tag );
	if ( attrs ) for ( const k in attrs ) el.setAttribute( k, attrs[ k ] );
	if ( text !== undefined ) el.textContent = text;
	return el;

}

/** A column with a round data end (top) and a square foot on the baseline. */
function columnPath( x, y, w, h, r ) {

	r = Math.max( 0, Math.min( r, w / 2, h ) );
	return `M${x} ${y + h}V${y + r}Q${x} ${y} ${x + r} ${y}H${x + w - r}Q${x + w} ${y} ${x + w} ${y + r}V${y + h}Z`;

}

export function barChart( bins, options = {} ) {

	const { width = 304, height = 104, name = 'Distribution', unit = 'nodes', describe = ( b ) => formatRange( b.from, b.to ), idle = '' } = options;
	const total = options.total !== undefined ? options.total : bins.reduce( ( s, b ) => s + b.count, 0 );
	const figure = document.createElement( 'figure' );
	figure.className = 'wgx-chart';
	const readout = document.createElement( 'figcaption' );
	readout.className = 'wgx-chart-readout';
	readout.textContent = idle;
	if ( ! bins.length ) {

		readout.textContent = 'Nothing to show.';
		figure.append( readout );
		return figure;

	}

	const max = bins.reduce( ( m, b ) => Math.max( m, b.count ), 0 );
	const top = niceCeil( max );
	const yLabels = [ 0, top ];
	const left = 8 + 6 * compactCount( top ).length, right = 4, padTop = 14, bottom = 17;
	const plotW = width - left - right, plotH = height - padTop - bottom;
	const slot = plotW / bins.length;
	const barW = Math.max( 1, Math.min( 24, slot - 2 ) );
	const y = ( v ) => padTop + plotH * ( 1 - v / top );

	const root = svg( 'svg', { class: 'wgx-chart-svg', viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': `${name}: ${bins.length} bars, largest ${describe( bins[ bins.findIndex( ( b ) => b.count === max ) ] )} with ${formatCount( max )} ${unit}` } );
	// hairline grid: the baseline and the top of the scale, half way between them a fainter one
	root.append( svg( 'line', { class: 'wgx-chart-grid is-faint', x1: left, x2: width - right, y1: y( top / 2 ), y2: y( top / 2 ) } ) );
	root.append( svg( 'line', { class: 'wgx-chart-grid is-faint', x1: left, x2: width - right, y1: y( top ), y2: y( top ) } ) );
	for ( const v of yLabels ) root.append( svg( 'text', { class: 'wgx-chart-tick', x: left - 5, y: y( v ) + 3.5, 'text-anchor': 'end' }, compactCount( v ) ) );

	// x labels: as many as fit side by side (first and last always, the rest thinned evenly)
	const texts = bins.map( describeShort );
	const widest = texts.reduce( ( m, t ) => Math.max( m, t.length ), 1 ) * 5.6 + 8;
	const every = Math.max( 1, Math.ceil( widest / slot ) );
	const labelled = new Set();
	for ( let k = 0; k < bins.length; k += every ) labelled.add( k );
	if ( ( bins.length - 1 ) % every !== 0 ) {

		// the last bar is always named; drop the neighbour it would run into
		labelled.add( bins.length - 1 );
		const prev = Math.floor( ( bins.length - 1 ) / every ) * every;
		if ( ( bins.length - 1 - prev ) * slot < widest && prev > 0 ) labelled.delete( prev );

	}

	const peak = bins.findIndex( ( b ) => b.count === max );
	const marks = [];
	const show = ( k ) => {

		for ( let j = 0; j < marks.length; j ++ ) marks[ j ].classList.toggle( 'is-on', j === k );
		if ( k < 0 ) {

			readout.textContent = idle;
			readout.classList.remove( 'is-live' );
			return;

		}

		const b = bins[ k ];
		readout.textContent = '';
		const strong = document.createElement( 'b' );
		strong.textContent = `${formatCount( b.count )} ${b.count === 1 ? unit.replace( /s$/, '' ) : unit}`;
		readout.append( strong, ` · ${describe( b )}${total > 0 ? ` · ${formatPercent( b.count / total )}` : ''}` );
		readout.classList.add( 'is-live' );

	};

	bins.forEach( ( b, k ) => {

		const x0 = left + k * slot, bx = x0 + ( slot - barW ) / 2;
		const g = svg( 'g', { class: 'wgx-chart-bar', tabindex: k === Math.max( 0, peak ) ? '0' : '-1', role: 'listitem', 'aria-label': `${describe( b )}: ${formatCount( b.count )} ${unit}` } );
		const h = b.count > 0 ? Math.max( 1.5, plotH * b.count / top ) : 0;
		if ( h > 0 ) g.append( svg( 'path', { class: 'wgx-chart-col', d: columnPath( bx, padTop + plotH - h, barW, h, 4 ) } ) );
		g.append( svg( 'rect', { class: 'wgx-chart-hit', x: x0, y: 0, width: slot, height: height - bottom + 2 } ) );
		if ( k === peak ) root.append( svg( 'text', { class: 'wgx-chart-value', x: Math.min( width - right, Math.max( left, x0 + slot / 2 ) ), y: Math.max( 9, padTop + plotH - h - 4 ), 'text-anchor': 'middle' }, formatCount( b.count ) ) );
		if ( labelled.has( k ) ) root.append( svg( 'text', { class: 'wgx-chart-tick', x: x0 + slot / 2, y: height - 4, 'text-anchor': k === bins.length - 1 && slot < widest ? 'end' : 'middle', ...( k === bins.length - 1 && slot < widest ? { x: width - right } : {} ) }, texts[ k ] ) );
		g.addEventListener( 'pointerenter', () => show( k ) );
		g.addEventListener( 'pointerleave', () => show( - 1 ) );
		g.addEventListener( 'focus', () => show( k ) );
		g.addEventListener( 'blur', () => show( - 1 ) );
		g.addEventListener( 'click', () => show( k ) );
		g.addEventListener( 'keydown', ( e ) => {

			const to = e.key === 'ArrowRight' || e.key === 'ArrowUp' ? k + 1 : e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? k - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? marks.length - 1 : - 1;
			if ( to < 0 || to >= marks.length || to === k || e.ctrlKey || e.metaKey || e.altKey ) return;
			e.preventDefault();
			e.stopPropagation(); // the arrows belong to the chart here, not to the camera
			g.setAttribute( 'tabindex', '-1' );
			marks[ to ].setAttribute( 'tabindex', '0' );
			marks[ to ].focus( { preventScroll: true } );

		} );
		marks.push( g );
		root.append( g );

	} );
	root.append( svg( 'line', { class: 'wgx-chart-grid', x1: left, x2: width - right, y1: y( 0 ), y2: y( 0 ) } ) );

	// the same numbers as a table, for readers who cannot see the picture
	const table = document.createElement( 'table' );
	const hidden = document.createElement( 'div' ); // a table ignores the clipping box itself
	hidden.className = 'wgx-sr';
	hidden.append( table );
	const caption = document.createElement( 'caption' );
	caption.textContent = name;
	table.append( caption );
	for ( const b of bins ) {

		const tr = document.createElement( 'tr' ), th = document.createElement( 'th' ), td = document.createElement( 'td' );
		th.textContent = describe( b );
		td.textContent = `${formatCount( b.count )} ${unit}`;
		tr.append( th, td );
		table.append( tr );

	}

	figure.append( root, readout, hidden );
	return figure;

}

function describeShort( b ) {

	return b.from === b.to ? compactCount( b.from ) : `${compactCount( b.from )}–${compactCount( b.to )}`;

}
