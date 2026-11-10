// Comparing partitions: the scores the clustering comparison (dev/data.html, the tests) reports.
// Pure JS, no imports. A partition is an array-like of non-negative integer labels, one per node
// (not necessarily dense).
//
// API
//   nmi( a, b ) -> 0..1     normalised mutual information, 2 I(A;B) / ( H(A) + H(B) ); 1 when the
//                           partitions are identical up to relabelling (also when both have one part)
//   ari( a, b ) -> ≤ 1      adjusted Rand index (Hubert & Arabie): 1 identical, about 0 for
//                           independent partitions
//   partSizes( a ) -> number[]            sizes of the parts, in order of first appearance
//   bestMatch( levels, truth ) -> { level, nmi, ari, clusters }   the level of a hierarchy that
//                           matches a planted partition best (by NMI)
//   isNested( fine, coarse ) -> boolean   every part of `fine` lies in one part of `coarse`

function dense( a ) {

	const map = new Map();
	const out = new Int32Array( a.length );
	for ( let i = 0; i < a.length; i ++ ) {

		let id = map.get( a[ i ] );
		if ( id === undefined ) {

			id = map.size;
			map.set( a[ i ], id );

		}

		out[ i ] = id;

	}

	return { ids: out, count: map.size };

}

function contingency( a, b ) {

	const A = dense( a ), B = dense( b );
	const n = a.length;
	const cells = new Map();
	const rows = new Float64Array( A.count ), cols = new Float64Array( B.count );
	for ( let i = 0; i < n; i ++ ) {

		const key = A.ids[ i ] * B.count + B.ids[ i ];
		cells.set( key, ( cells.get( key ) || 0 ) + 1 );
		rows[ A.ids[ i ] ] ++;
		cols[ B.ids[ i ] ] ++;

	}

	return { n, cells, rows, cols, nb: B.count };

}

export function nmi( a, b ) {

	if ( a.length !== b.length ) throw new Error( 'nmi: partitions differ in length' );
	const { n, cells, rows, cols, nb } = contingency( a, b );
	if ( n === 0 ) return 1;
	let ha = 0, hb = 0, mi = 0;
	for ( const r of rows ) if ( r > 0 ) ha -= ( r / n ) * Math.log( r / n );
	for ( const c of cols ) if ( c > 0 ) hb -= ( c / n ) * Math.log( c / n );
	for ( const [ key, x ] of cells ) {

		const r = rows[ Math.floor( key / nb ) ], c = cols[ key % nb ];
		mi += ( x / n ) * Math.log( ( x * n ) / ( r * c ) );

	}

	if ( ha + hb <= 0 ) return 1;
	return Math.max( 0, Math.min( 1, ( 2 * mi ) / ( ha + hb ) ) );

}

export function ari( a, b ) {

	if ( a.length !== b.length ) throw new Error( 'ari: partitions differ in length' );
	const { n, cells, rows, cols } = contingency( a, b );
	const c2 = ( x ) => ( x * ( x - 1 ) ) / 2;
	let sum = 0, sa = 0, sb = 0;
	for ( const x of cells.values() ) sum += c2( x );
	for ( const r of rows ) sa += c2( r );
	for ( const c of cols ) sb += c2( c );
	const expected = ( sa * sb ) / Math.max( 1, c2( n ) );
	const max = ( sa + sb ) / 2;
	if ( max === expected ) return 1;
	return ( sum - expected ) / ( max - expected );

}

export function partSizes( a ) {

	const { ids, count } = dense( a );
	const sizes = new Array( count ).fill( 0 );
	for ( let i = 0; i < ids.length; i ++ ) sizes[ ids[ i ] ] ++;
	return sizes;

}

export function bestMatch( levels, truth ) {

	let best = { level: - 1, nmi: - 1, ari: 0, clusters: 0 };
	levels.forEach( ( level, l ) => {

		const score = nmi( level, truth );
		if ( score > best.nmi ) best = { level: l, nmi: score, ari: 0, clusters: 0 };

	} );
	if ( best.level >= 0 ) {

		best.ari = ari( levels[ best.level ], truth );
		best.clusters = partSizes( levels[ best.level ] ).length;

	}

	return best;

}

export function isNested( fine, coarse ) {

	const parent = new Map();
	for ( let i = 0; i < fine.length; i ++ ) {

		const p = parent.get( fine[ i ] );
		if ( p === undefined ) parent.set( fine[ i ], coarse[ i ] );
		else if ( p !== coarse[ i ] ) return false;

	}

	return true;

}
