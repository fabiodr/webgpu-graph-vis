// Number formatting of the explore layer: every figure a person reads on a legend, in the
// insights drawer or on the node card goes through here, so the same value looks the same
// everywhere. Pure functions, no DOM.
//
//   formatNumber( v, { digits = 3 } )   a measured value in its own unit, about `digits`
//                                       significant digits: 1,234 · 56.8 · 2.63 · 0.0312 · 4.1e-5
//   formatRatio( v, { decimals = 2 } )  a coefficient between −1 and 1 with a fixed number of
//                                       decimals, so that neighbours line up: 0.57 · 0.50 · −0.17
//   formatCount( n )                    an exact count with thousands separators: 20,000
//   compactCount( n )                   a count for tight places: 950 · 1.2k · 20k · 3.4M
//   formatPercent( f, { digits } )      a fraction as a percentage: 98 % · 2.4 % · < 0.1 %
//   formatRank( rank, of )              "#3 of 77" (rank 1-based)
//   formatRange( from, to )             "3" for a single value, "3–6" for a range
//   formatSeconds( s )                  "0.4 s" · "12 s" · "3 min"
//   ordinal( n )                        "1st", "2nd", "3rd", "4th"
//   formatMetric( id, v, N )            a node metric in the unit a person expects: counts for
//                                       degree, k-core and triangles, betweenness as the share of
//                                       all pairs of other nodes ( "57 %", "0.24 %" ), the rest as
//                                       numbers
//
// Non-finite input gives an en dash, the sign of "no value".

const NONE = '–';
const MINUS = '−';
const GROUP = /\B(?=(\d{3})+(?!\d))/g;

const group = ( s ) => s.replace( GROUP, ',' );

/** Drops trailing zeros of a fixed-point string ("2.50" -> "2.5", "3.00" -> "3"). */
function trim( s ) {

	return s.indexOf( '.' ) < 0 ? s : s.replace( /\.?0+$/, '' );

}

export function formatNumber( v, { digits = 3 } = {} ) {

	if ( typeof v !== 'number' || ! Number.isFinite( v ) ) return NONE;
	if ( v === 0 ) return '0';
	const a = Math.abs( v ), sign = v < 0 ? MINUS : '';
	if ( a >= 1e15 ) return sign + a.toExponential( Math.max( 0, digits - 1 ) ).replace( 'e+', 'e' );
	if ( Number.isInteger( a ) ) return sign + group( String( a ) );
	if ( a >= 1000 ) return sign + group( String( Math.round( a ) ) );
	if ( a < 1e-4 ) {

		const [ m, e ] = a.toExponential( Math.max( 0, digits - 2 ) ).split( 'e' );
		return `${sign}${trim( m )}e${e.replace( '-', MINUS ).replace( '+', '' )}`;

	}

	// `digits` significant digits, never more than six decimals
	const magnitude = Math.floor( Math.log10( a ) );
	const decimals = Math.min( 6, Math.max( 0, digits - 1 - magnitude ) );
	return sign + trim( a.toFixed( decimals ) );

}

export function formatRatio( v, { decimals = 2 } = {} ) {

	if ( typeof v !== 'number' || ! Number.isFinite( v ) ) return NONE;
	const text = Math.abs( v ).toFixed( decimals );
	return ( v < 0 && + text !== 0 ? MINUS : '' ) + text;

}

export function formatCount( n ) {

	if ( typeof n !== 'number' || ! Number.isFinite( n ) ) return NONE;
	const r = Math.round( n );
	return ( r < 0 ? MINUS : '' ) + group( String( Math.abs( r ) ) );

}

export function compactCount( n ) {

	if ( typeof n !== 'number' || ! Number.isFinite( n ) ) return NONE;
	const a = Math.abs( Math.round( n ) ), sign = n < 0 ? MINUS : '';
	if ( a < 1000 ) return sign + String( a );
	const units = [ [ 1e9, 'B' ], [ 1e6, 'M' ], [ 1e3, 'k' ] ];
	for ( let u = 0; u < units.length; u ++ ) {

		const [ size, suffix ] = units[ u ];
		if ( a < size ) continue;
		const q = a / size;
		// 1.2k, 12k, 120k: one decimal only below ten
		const text = q < 9.95 ? trim( q.toFixed( 1 ) ) : String( Math.round( q ) );
		// 999,600 is "1M", not "1000k"
		if ( text === '1000' && u > 0 ) return sign + '1' + units[ u - 1 ][ 1 ];
		return sign + text + suffix;

	}

	return sign + String( a );

}

export function formatPercent( f, { digits = null } = {} ) {

	if ( typeof f !== 'number' || ! Number.isFinite( f ) ) return NONE;
	const p = 100 * f, a = Math.abs( p );
	if ( a === 0 ) return '0 %';
	if ( a < 0.1 && digits === null ) return `< 0.1 %`;
	const d = digits !== null ? digits : a >= 10 ? 0 : 1;
	let text = trim( a.toFixed( d ) );
	// "100 %" is a claim of its own: a share below one never rounds up to it
	if ( text === '100' && a < 100 ) text = d > 0 ? ( 100 - Math.pow( 10, - d ) ).toFixed( d ) : '99.9';
	return `${p < 0 ? MINUS : ''}${text} %`;

}

export function formatRank( rank, of ) {

	if ( ! Number.isFinite( rank ) ) return NONE;
	return Number.isFinite( of ) ? `#${formatCount( rank )} of ${formatCount( of )}` : `#${formatCount( rank )}`;

}

export function formatRange( from, to ) {

	return from === to || to === undefined ? formatCount( from ) : `${formatCount( from )}–${formatCount( to )}`;

}

export function formatSeconds( s ) {

	if ( typeof s !== 'number' || ! Number.isFinite( s ) ) return NONE;
	if ( s < 10 ) return `${trim( s.toFixed( 1 ) )} s`;
	if ( s < 90 ) return `${Math.round( s )} s`;
	return `${Math.round( s / 60 )} min`;

}

export function ordinal( n ) {

	const k = Math.abs( n ) % 100, d = k % 10;
	const suffix = k >= 11 && k <= 13 ? 'th' : d === 1 ? 'st' : d === 2 ? 'nd' : d === 3 ? 'rd' : 'th';
	return `${formatCount( n )}${suffix}`;

}

const COUNT_METRICS = new Set( [ 'degree', 'core', 'triangles' ] );

export function formatMetric( id, v, N ) {

	if ( typeof v !== 'number' || ! Number.isFinite( v ) ) return NONE;
	if ( COUNT_METRICS.has( id ) ) return formatCount( v );
	if ( id === 'betweenness' && N > 2 ) {

		const share = v / ( ( N - 1 ) * ( N - 2 ) / 2 );
		if ( share > 0 && share < 1e-5 ) return '< 0.001 %';
		// below one percent: two significant digits (at most three decimals), so that a ranking of
		// small shares does not read "0.2 %" ten times
		if ( share > 0 && share < 0.01 ) return formatPercent( share, { digits: Math.min( 3, 1 - Math.floor( Math.log10( 100 * share ) ) ) } );
		return formatPercent( share );

	}

	return formatNumber( v );

}
