// Layout quality metrics (pure JS, typed arrays): the definitions the layout research used, so
// numbers measured on the GPU layout (dev/layout.html) and in tests compare directly with it.
//
//   const ev = prepareEval( graph, { sources = 48, npSamples = 400, seed = 7 } )   once per graph
//   layoutMetrics( ev, positions, stride = 4 ) -> { stress, np, sil, edgeCV, meanEdge, hubNN,
//                                                   overlaps, overlapsN, ext, nan }
//
//  stress      scale-normalised stress against BFS distances: mean over sampled pairs (s, j) of
//              ( a·|x_s − x_j| / d_sj − 1 )², a = optimal global scale (weights d⁻²). Sources: all
//              nodes for N ≤ 400, else `sources` sampled ones (every reachable target). Lower is better.
//  np          neighbourhood preservation: mean Jaccard of the graph neighbours of i and its deg(i)
//              nearest nodes in space (sampled nodes with deg ≥ 1)
//  sil         centroid silhouette at the finest hierarchy level: mean of (b − a)/max(a, b),
//              a = distance to the own cluster centroid, b = to the nearest other centroid
//  edgeCV      coefficient of variation of the edge length; meanEdge in world units
//  hubNN       mean distance to the 5 nearest nodes in space, top 1% degree nodes / median node
//              (< 1: hubs sit in crowded places)
//  overlaps    node pairs closer than r_i + r_j with the NodeRenderer radii
//              r = 0.065·R/∛N·(0.7 + 0.9·√(deg/maxDeg)); overlapsN after rescaling the layout to
//              the reference extent (pass { extent } to count against another reference)
//  ext         rms radius about the centroid / (√(3/5)·layoutRadius): 1 = a uniform ball of
//              radius layoutRadius
//  nan         number of non-finite coordinates
//
// Also exported: bfsDistances, stress, neighbourhoodPreservation, communityMetrics, edgeStats,
// extent, countOverlaps, nodeRadii.

import { rng } from './prepare.js';

const UNREACHED = 65535;

/** BFS hop distances from `src` into `out` (Uint16Array N; 65535 = unreachable). */
export function bfsDistances( g, src, out = new Uint16Array( g.N ), queue = new Int32Array( g.N ) ) {

	const { adjStart, degree, adjList } = g;
	out.fill( UNREACHED );
	let head = 0, tail = 0;
	queue[ tail ++ ] = src;
	out[ src ] = 0;
	while ( head < tail ) {

		const u = queue[ head ++ ], du = out[ u ] + 1;
		for ( let k = adjStart[ u ], e = k + degree[ u ]; k < e; k ++ ) {

			const v = adjList[ k ];
			if ( out[ v ] === UNREACHED ) {

				out[ v ] = du;
				queue[ tail ++ ] = v;

			}

		}

	}

	return out;

}

/** World radii of the node spheres at size 1 (NodeRenderer contract). */
export function nodeRadii( g ) {

	const base = ( 0.065 * g.layoutRadius ) / Math.cbrt( g.N );
	const r = new Float64Array( g.N );
	for ( let i = 0; i < g.N; i ++ ) r[ i ] = base * ( 0.7 + 0.9 * Math.sqrt( g.degree[ i ] / Math.max( 1, g.maxDegree ) ) );
	return r;

}

/** Samples and BFS tables for the metrics of one graph. */
export function prepareEval( g, { sources = 48, npSamples = 400, seed = 7 } = {} ) {

	const N = g.N;
	const rand = rng( seed );
	const pick = ( count ) => {

		const perm = Int32Array.from( { length: N }, ( _, i ) => i );
		if ( count >= N ) return perm;
		for ( let i = 0; i < count; i ++ ) {

			const j = i + Math.floor( rand() * ( N - i ) );
			const t = perm[ i ]; perm[ i ] = perm[ j ]; perm[ j ] = t;

		}

		return perm.slice( 0, count );

	};

	const src = pick( N <= 400 ? N : sources );
	const dist = new Uint16Array( src.length * N );
	const queue = new Int32Array( N );
	for ( let s = 0; s < src.length; s ++ ) bfsDistances( g, src[ s ], dist.subarray( s * N, ( s + 1 ) * N ), queue );
	const npNodes = pick( npSamples ).filter( ( i ) => g.degree[ i ] > 0 );
	let clusters = null, K = 0;
	if ( g.D > 0 ) {

		clusters = new Int32Array( N );
		for ( let i = 0; i < N; i ++ ) {

			clusters[ i ] = g.anc[ i * g.D ] - g.levelOffset[ 0 ];
			if ( clusters[ i ] + 1 > K ) K = clusters[ i ] + 1;

		}

	}

	// hubs: top 1% by degree
	const order = Array.from( { length: N }, ( _, i ) => i ).sort( ( a, b ) => g.degree[ b ] - g.degree[ a ] || a - b );
	const hubs = order.slice( 0, Math.max( 1, Math.ceil( N / 100 ) ) );
	return { g, src, dist, npNodes, clusters, K, hubs, radii: nodeRadii( g ) };

}

export function stress( ev, pos, stride = 4 ) {

	const { g, src, dist } = ev;
	const N = g.N;
	let sEd = 0, sEe = 0, cnt = 0;
	// optimal scale a = Σ w e d / Σ w e² with w = d⁻²  ->  Σ (e/d) / Σ (e/d)²
	for ( let s = 0; s < src.length; s ++ ) {

		const i = src[ s ], o = s * N;
		const xi = pos[ stride * i ], yi = pos[ stride * i + 1 ], zi = pos[ stride * i + 2 ];
		for ( let j = 0; j < N; j ++ ) {

			const d = dist[ o + j ];
			if ( d === 0 || d === UNREACHED ) continue;
			const dx = pos[ stride * j ] - xi, dy = pos[ stride * j + 1 ] - yi, dz = pos[ stride * j + 2 ] - zi;
			const r = Math.sqrt( dx * dx + dy * dy + dz * dz ) / d;
			sEd += r; sEe += r * r; cnt ++;

		}

	}

	if ( cnt === 0 || sEe === 0 ) return { stress: NaN, scale: NaN };
	const a = sEd / sEe;
	// Σ (a r − 1)² / cnt = (a² Σr² − 2a Σr + cnt) / cnt
	return { stress: ( a * a * sEe - 2 * a * sEd + cnt ) / cnt, scale: a };

}

// quickselect on arr[0..n): returns the k-th smallest (arr is reordered)
function kthSmallest( arr, n, k ) {

	let lo = 0, hi = n - 1;
	while ( hi > lo ) {

		const pivot = arr[ ( lo + hi ) >> 1 ];
		let i = lo, j = hi;
		while ( i <= j ) {

			while ( arr[ i ] < pivot ) i ++;
			while ( arr[ j ] > pivot ) j --;
			if ( i <= j ) {

				const t = arr[ i ]; arr[ i ] = arr[ j ]; arr[ j ] = t;
				i ++; j --;

			}

		}

		if ( k <= j ) hi = j;
		else if ( k >= i ) lo = i;
		else break;

	}

	return arr[ k ];

}

export function neighbourhoodPreservation( ev, pos, stride = 4 ) {

	const { g, npNodes } = ev;
	const N = g.N;
	const d2 = new Float64Array( N ), scratch = new Float64Array( N );
	let sum = 0, n = 0;
	for ( let q = 0; q < npNodes.length; q ++ ) {

		const i = npNodes[ q ];
		const xi = pos[ stride * i ], yi = pos[ stride * i + 1 ], zi = pos[ stride * i + 2 ];
		for ( let j = 0; j < N; j ++ ) {

			const dx = pos[ stride * j ] - xi, dy = pos[ stride * j + 1 ] - yi, dz = pos[ stride * j + 2 ] - zi;
			d2[ j ] = dx * dx + dy * dy + dz * dz;

		}

		d2[ i ] = Infinity;
		const k = g.degree[ i ];
		let jac = 1;
		if ( k < N - 1 ) {

			scratch.set( d2 );
			const thr = kthSmallest( scratch, N, k - 1 );
			let inter = 0;
			for ( let t = g.adjStart[ i ], e = t + k; t < e; t ++ ) if ( d2[ g.adjList[ t ] ] <= thr ) inter ++;
			jac = inter / ( 2 * k - inter );

		}

		sum += jac;
		n ++;

	}

	return sum / Math.max( 1, n );

}

export function communityMetrics( ev, pos, stride = 4 ) {

	const { g, clusters, K } = ev;
	if ( ! clusters || K < 2 ) return { sil: NaN };
	const N = g.N;
	const c = new Float64Array( 3 * K ), m = new Float64Array( K );
	for ( let i = 0; i < N; i ++ ) {

		const k = clusters[ i ];
		c[ 3 * k ] += pos[ stride * i ]; c[ 3 * k + 1 ] += pos[ stride * i + 1 ]; c[ 3 * k + 2 ] += pos[ stride * i + 2 ]; m[ k ] ++;

	}

	for ( let k = 0; k < K; k ++ ) if ( m[ k ] ) {

		c[ 3 * k ] /= m[ k ]; c[ 3 * k + 1 ] /= m[ k ]; c[ 3 * k + 2 ] /= m[ k ];

	}

	let sil = 0;
	for ( let i = 0; i < N; i ++ ) {

		const own = clusters[ i ];
		let a = 0, b = Infinity;
		for ( let k = 0; k < K; k ++ ) {

			if ( ! m[ k ] ) continue;
			const dx = pos[ stride * i ] - c[ 3 * k ], dy = pos[ stride * i + 1 ] - c[ 3 * k + 1 ], dz = pos[ stride * i + 2 ] - c[ 3 * k + 2 ];
			const d2 = dx * dx + dy * dy + dz * dz;
			if ( k === own ) a = Math.sqrt( d2 );
			else if ( d2 < b ) b = d2;

		}

		b = Math.sqrt( b );
		sil += ( b - a ) / Math.max( a, b, 1e-12 );

	}

	return { sil: sil / N };

}

/**
 * Node pairs closer than (r_i + r_j)/scale, i.e. overlaps after scaling the layout by `scale`.
 * Uniform grid with cells of the largest diameter; every pair is counted once. (The cell key is
 * an exact integer below 2⁵³: the key of the research harness, 3 × 21 bits in a float64, lost
 * the low bits for negative x, so one cell was visited as three and pairs there counted 3×.)
 */
export function countOverlaps( pos, N, radii, scale = 1, stride = 4 ) {

	let rmax = 0;
	for ( let i = 0; i < N; i ++ ) if ( radii[ i ] > rmax ) rmax = radii[ i ];
	const cell = ( 2 * rmax ) / scale; // in layout units
	if ( ! ( cell > 0 ) || ! Number.isFinite( cell ) ) return 0;
	const inv = 1 / cell;
	// cell coordinates, offset so that the neighbours of every occupied cell are ≥ 0
	const cells = new Int32Array( 3 * N );
	const lo = [ Infinity, Infinity, Infinity ], hi = [ - Infinity, - Infinity, - Infinity ];
	for ( let i = 0; i < N; i ++ ) for ( let a = 0; a < 3; a ++ ) {

		const v = Math.floor( pos[ stride * i + a ] * inv );
		if ( ! Number.isFinite( v ) || Math.abs( v ) > 1e9 ) return NaN; // not a layout
		cells[ 3 * i + a ] = v;
		if ( v < lo[ a ] ) lo[ a ] = v;
		if ( v > hi[ a ] ) hi[ a ] = v;

	}

	const nb = hi[ 1 ] - lo[ 1 ] + 3, nc = hi[ 2 ] - lo[ 2 ] + 3;
	const exact = N > 0 && ( hi[ 0 ] - lo[ 0 ] + 3 ) * nb * nc < Number.MAX_SAFE_INTEGER;
	const key = exact
		? ( a, b, c ) => ( ( a - lo[ 0 ] + 1 ) * nb + ( b - lo[ 1 ] + 1 ) ) * nc + ( c - lo[ 2 ] + 1 )
		: ( a, b, c ) => `${a},${b},${c}`;
	const grid = new Map();
	for ( let i = 0; i < N; i ++ ) {

		const k = key( cells[ 3 * i ], cells[ 3 * i + 1 ], cells[ 3 * i + 2 ] );
		let list = grid.get( k );
		if ( ! list ) grid.set( k, list = [] );
		list.push( i );

	}

	let count = 0;
	for ( let i = 0; i < N; i ++ ) {

		const a = cells[ 3 * i ], b = cells[ 3 * i + 1 ], c = cells[ 3 * i + 2 ];
		for ( let da = - 1; da <= 1; da ++ ) for ( let db = - 1; db <= 1; db ++ ) for ( let dc = - 1; dc <= 1; dc ++ ) {

			const list = grid.get( key( a + da, b + db, c + dc ) );
			if ( ! list ) continue;
			for ( const j of list ) {

				if ( j <= i ) continue;
				const dx = pos[ stride * j ] - pos[ stride * i ], dy = pos[ stride * j + 1 ] - pos[ stride * i + 1 ], dz = pos[ stride * j + 2 ] - pos[ stride * i + 2 ];
				const rs = ( radii[ i ] + radii[ j ] ) / scale;
				if ( dx * dx + dy * dy + dz * dz < rs * rs ) count ++;

			}

		}

	}

	return count;

}

/** Centroid, rms radius about it and ext = rms / (√(3/5)·layoutRadius). */
export function extent( g, pos, stride = 4 ) {

	const N = g.N;
	let cx = 0, cy = 0, cz = 0;
	for ( let i = 0; i < N; i ++ ) {

		cx += pos[ stride * i ]; cy += pos[ stride * i + 1 ]; cz += pos[ stride * i + 2 ];

	}

	cx /= N; cy /= N; cz /= N;
	let s = 0;
	for ( let i = 0; i < N; i ++ ) {

		const dx = pos[ stride * i ] - cx, dy = pos[ stride * i + 1 ] - cy, dz = pos[ stride * i + 2 ] - cz;
		s += dx * dx + dy * dy + dz * dz;

	}

	const rms = Math.sqrt( s / N );
	return { rms, ext: rms / ( Math.sqrt( 0.6 ) * g.layoutRadius ), center: [ cx, cy, cz ] };

}

export function edgeStats( g, pos, stride = 4 ) {

	const { E, edges } = g;
	let s = 0, s2 = 0;
	for ( let e = 0; e < E; e ++ ) {

		const a = stride * edges[ 2 * e ], b = stride * edges[ 2 * e + 1 ];
		const l = Math.hypot( pos[ a ] - pos[ b ], pos[ a + 1 ] - pos[ b + 1 ], pos[ a + 2 ] - pos[ b + 2 ] );
		s += l; s2 += l * l;

	}

	const mean = s / Math.max( 1, E );
	const cv = Math.sqrt( Math.max( 0, s2 / Math.max( 1, E ) - mean * mean ) ) / Math.max( 1e-12, mean );
	return { meanEdge: mean, edgeCV: cv };

}

// mean distance to the k nearest nodes in space for the given nodes (brute force)
function knnMean( pos, N, nodes, stride, k = 5 ) {

	const out = [];
	const d2 = new Float64Array( N ), scratch = new Float64Array( N );
	for ( const i of nodes ) {

		for ( let j = 0; j < N; j ++ ) {

			const dx = pos[ stride * j ] - pos[ stride * i ], dy = pos[ stride * j + 1 ] - pos[ stride * i + 1 ], dz = pos[ stride * j + 2 ] - pos[ stride * i + 2 ];
			d2[ j ] = dx * dx + dy * dy + dz * dz;

		}

		d2[ i ] = Infinity;
		const kk = Math.min( k, N - 1 );
		if ( kk < 1 ) continue;
		scratch.set( d2 );
		const thr = kthSmallest( scratch, N, kk - 1 );
		let s = 0, c = 0;
		for ( let j = 0; j < N; j ++ ) if ( d2[ j ] <= thr && c < kk ) {

			s += Math.sqrt( d2[ j ] ); c ++;

		}

		out.push( s / Math.max( 1, c ) );

	}

	return out;

}

function median( values ) {

	const v = Array.from( values ).filter( ( x ) => Number.isFinite( x ) ).sort( ( a, b ) => a - b );
	return v.length ? v[ v.length >> 1 ] : NaN;

}

/**
 * All metrics of a layout. `extent` = reference extent for overlapsN (default 1 = the 3D reference).
 */
export function layoutMetrics( ev, pos, stride = 4, { extent: reference = 1 } = {} ) {

	const { g, hubs, radii } = ev;
	const N = g.N;
	let nan = 0;
	for ( let i = 0; i < N; i ++ ) for ( let a = 0; a < 3; a ++ ) if ( ! Number.isFinite( pos[ stride * i + a ] ) ) nan ++;
	const ex = extent( g, pos, stride );
	if ( nan ) return { nan, stress: NaN, np: NaN, sil: NaN, edgeCV: NaN, meanEdge: NaN, hubNN: NaN, overlaps: NaN, overlapsN: NaN, ext: ex.ext };
	const es = edgeStats( g, pos, stride );
	const nnAll = median( knnMean( pos, N, Array.from( ev.npNodes ).slice( 0, 300 ), stride ) );
	const hubList = hubs.slice( 0, 60 );
	const nn = knnMean( pos, N, hubList, stride );
	const nnHub = nn.reduce( ( a, b ) => a + b, 0 ) / Math.max( 1, nn.length );
	return {
		nan,
		stress: stress( ev, pos, stride ).stress,
		np: neighbourhoodPreservation( ev, pos, stride ),
		sil: communityMetrics( ev, pos, stride ).sil,
		edgeCV: es.edgeCV, meanEdge: es.meanEdge,
		hubNN: nnHub / nnAll,
		overlaps: countOverlaps( pos, N, radii, 1, stride ),
		overlapsN: countOverlaps( pos, N, radii, reference / Math.max( 1e-9, ex.ext ), stride ),
		ext: ex.ext,
	};

}
