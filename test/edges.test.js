// Tests for the edge renderer's curves (src/gpu/EdgeCurves.js, src/render/EdgeRenderer.js): the
// pure-JS curve mirror (evalEdgeCurveCPU), which the dev page (dev/edges.html, window.__verify)
// checks against the samples the GPU kernel wrote; explicit control paths against the implicit
// hierarchy formula of the first renderer; the path upload and the "run only when something
// changed" logic of the cache (with a stand-in renderer that counts compute calls).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';
import { hierarchicalPaths, straightPaths } from '../src/graph/bundling.js';
import { GraphBuffers } from '../src/gpu/GraphBuffers.js';
import { EdgeCurves, encodePathRefs, maxSegmentsForBudget } from '../src/gpu/EdgeCurves.js';
import { evalEdgeCurveCPU, autoSegments, autoOpacity, autoDefocusLimit } from '../src/render/EdgeRenderer.js';

const load = ( id, flat = false ) => {

	const raw = JSON.parse( readFileSync( `public/data/imsv/${id}.json`, 'utf8' ) );
	if ( flat ) delete raw.hierarchy;
	return prepareGraph( normalizeGraph( raw ) );

};

// cluster centroids exactly like GraphBuffers.computeClusterCentroidsCPU (mass-weighted)
function centroids( g, pos4 ) {

	const cp = new Float32Array( 4 * g.C );
	for ( const { height, start, count } of g.heightRanges ) {

		for ( let c = start; c < start + count; c ++ ) {

			let x = 0, y = 0, z = 0, m = 0;
			const s = g.clusterChildStart[ c ], n = g.clusterChildCount[ c ];
			for ( let k = 0; k < n; k ++ ) {

				const ch = g.childList[ s + k ];
				if ( height === 1 || g.D === 0 ) {

					x += pos4[ 4 * ch ]; y += pos4[ 4 * ch + 1 ]; z += pos4[ 4 * ch + 2 ]; m += 1;

				} else {

					const w = g.clusterMass[ ch ];
					x += cp[ 4 * ch ] * w; y += cp[ 4 * ch + 1 ] * w; z += cp[ 4 * ch + 2 ] * w; m += w;

				}

			}

			m = Math.max( m, 1e-9 );
			cp[ 4 * c ] = x / m; cp[ 4 * c + 1 ] = y / m; cp[ 4 * c + 2 ] = z / m;

		}

	}

	return cp;

}

const spherePos = ( g ) => {

	const p = new Float32Array( 4 * g.N );
	for ( let i = 0; i < 4 * g.N; i ++ ) p[ i ] = g.sphereDir[ i ] * g.layoutRadius;
	return p;

};

const dist = ( a, b ) => Math.hypot( a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] );
const nodeAt = ( pos, i ) => [ pos[ 4 * i ], pos[ 4 * i + 1 ], pos[ 4 * i + 2 ] ];
const finite = ( v ) => v.every( Number.isFinite );

for ( const id of [ 'karate', 'lesmis', 'football', 'celegansneural' ] ) {

	const g = load( id );
	const modes = [
		[ 'force', g.initPos, { layoutBlend: 0 } ],
		[ 'sphere', spherePos( g ), { layoutBlend: 1 } ],
		[ 'morph', g.initPos.map( ( v, i ) => 0.5 * v + 0.5 * spherePos( g )[ i ] ), { layoutBlend: 0.5 } ],
	];

	for ( const [ mode, pos, o ] of modes ) {

		const cp = centroids( g, pos );

		test( `${id} ${mode}: curve starts/ends at the node centers, finite everywhere`, () => {

			for ( let e = 0; e < g.E; e ++ ) {

				const s = g.edges[ 2 * e ], t = g.edges[ 2 * e + 1 ];
				for ( const beta of [ 0, 0.85, 1 ] ) {

					const a = evalEdgeCurveCPU( g, e, 0, pos, cp, { ...o, beta } );
					const b = evalEdgeCurveCPU( g, e, 1, pos, cp, { ...o, beta } );
					assert.ok( dist( a.position, nodeAt( pos, s ) ) < 1e-6 * g.layoutRadius, `start of edge ${e}` );
					assert.ok( dist( b.position, nodeAt( pos, t ) ) < 1e-6 * g.layoutRadius, `end of edge ${e}` );
					for ( const tt of [ 0, 0.13, 0.5, 0.77, 1 ] ) {

						const r = evalEdgeCurveCPU( g, e, tt, pos, cp, { ...o, beta } );
						assert.ok( finite( r.position ) && finite( r.tangent ), `finite e=${e} t=${tt}` );
						assert.ok( Math.hypot( ...r.tangent ) > 0, `non-degenerate tangent e=${e} t=${tt}` );

					}

				}

			}

		} );

		test( `${id} ${mode}: C0/C1 continuity across span boundaries, tangent = d position`, () => {

			const h = 1e-5;
			for ( let e = 0; e < g.E; e += Math.max( 1, Math.floor( g.E / 60 ) ) ) {

				const L = evalEdgeCurveCPU( g, e, 0, pos, cp, o ).L;
				for ( let k = 1; k <= L; k ++ ) {

					const tk = k / ( L + 1 );
					const lo = evalEdgeCurveCPU( g, e, tk - h, pos, cp, o ), hi = evalEdgeCurveCPU( g, e, tk + h, pos, cp, o );
					const step = dist( lo.position, hi.position );
					const speed = Math.hypot( ...lo.tangent ) * ( L + 1 ) * 2 * h; // expected |Δp| for 2h
					assert.ok( step <= speed * 1.05 + 1e-6 * g.layoutRadius, `C0 at knot ${k} of edge ${e}` );
					// tangent directions continuous
					const la = Math.hypot( ...lo.tangent ), ha = Math.hypot( ...hi.tangent );
					const cos = ( lo.tangent[ 0 ] * hi.tangent[ 0 ] + lo.tangent[ 1 ] * hi.tangent[ 1 ] + lo.tangent[ 2 ] * hi.tangent[ 2 ] ) / ( la * ha );
					assert.ok( cos > 0.999, `C1 at knot ${k} of edge ${e}: cos ${cos}` );

				}

				// analytic tangent vs central difference (d/du = d/dt / (L+1)) away from the ends
				for ( const tt of [ 0.21, 0.5, 0.66 ] ) {

					const a = evalEdgeCurveCPU( g, e, tt - h, pos, cp, o ).position, b = evalEdgeCurveCPU( g, e, tt + h, pos, cp, o ).position;
					const fd = [ 0, 1, 2 ].map( ( q ) => ( b[ q ] - a[ q ] ) / ( 2 * h * ( L + 1 ) ) );
					const an = evalEdgeCurveCPU( g, e, tt, pos, cp, o ).tangent;
					// the shader adds 1e-4 of the span chord (end fallback), so compare with a small tolerance
					const err = dist( fd, an ), mag = Math.hypot( ...an );
					// bias bound: 1e-4 * |chord| <= 1e-4 * 2.5 R (the chord spans at most the layout)
					assert.ok( err < 2e-3 * mag + 2.5e-4 * g.layoutRadius, `tangent e=${e} t=${tt}: ${err} vs ${mag}` );

				}

			}

		} );

	}

	test( `${id} force: beta = 0 gives the straight segment`, () => {

		const cp = centroids( g, g.initPos );
		for ( let e = 0; e < g.E; e ++ ) {

			const s = nodeAt( g.initPos, g.edges[ 2 * e ] ), t = nodeAt( g.initPos, g.edges[ 2 * e + 1 ] );
			const len = dist( s, t );
			for ( const tt of [ 0.1, 0.4, 0.9 ] ) {

				const p = evalEdgeCurveCPU( g, e, tt, g.initPos, cp, { beta: 0 } ).position;
				// distance from the line through s, t
				const d = [ t[ 0 ] - s[ 0 ], t[ 1 ] - s[ 1 ], t[ 2 ] - s[ 2 ] ].map( ( v ) => v / len );
				const w = [ p[ 0 ] - s[ 0 ], p[ 1 ] - s[ 1 ], p[ 2 ] - s[ 2 ] ];
				const along = w[ 0 ] * d[ 0 ] + w[ 1 ] * d[ 1 ] + w[ 2 ] * d[ 2 ];
				const off = Math.hypot( w[ 0 ] - along * d[ 0 ], w[ 1 ] - along * d[ 1 ], w[ 2 ] - along * d[ 2 ] );
				assert.ok( off < 1e-6 * g.layoutRadius && along >= - 1e-6 && along <= len + 1e-6 );

			}

		}

	} );

	test( `${id} force: beta = 1 follows the hierarchy (convex hull, passes near each centroid)`, () => {

		const cp = centroids( g, g.initPos );
		let worst = 0;
		for ( let e = 0; e < g.E; e ++ ) {

			const ref = evalEdgeCurveCPU( g, e, 0, g.initPos, cp, { beta: 1 } );
			const P = ref.controlPoints;
			const pts = [];
			for ( let k = 0; k <= 200; k ++ ) pts.push( evalEdgeCurveCPU( g, e, k / 200, g.initPos, cp, { beta: 1 } ).position );
			// convex hull property (checked with the bounding box of the control polygon)
			for ( const p of pts ) for ( let q = 0; q < 3; q ++ ) {

				const lo = Math.min( ...P.map( ( c ) => c[ q ] ) ), hi = Math.max( ...P.map( ( c ) => c[ q ] ) );
				assert.ok( p[ q ] >= lo - 1e-6 * g.layoutRadius && p[ q ] <= hi + 1e-6 * g.layoutRadius );

			}

			// every interior control point (a centroid) is approached within 1/3 of the polygon's
			// local "corner size" (B-spline distance bound: |P - curve| <= |P_k-1 - 2P_k + P_k+1| / 6
			// for an isolated corner; tripled ends make neighbors closer, so allow 1/3)
			for ( let k = 1; k < P.length - 1; k ++ ) {

				const corner = Math.hypot( ...[ 0, 1, 2 ].map( ( q ) => P[ k - 1 ][ q ] - 2 * P[ k ][ q ] + P[ k + 1 ][ q ] ) );
				const best = Math.min( ...pts.map( ( p ) => dist( p, P[ k ] ) ) );
				worst = Math.max( worst, best / Math.max( corner, 1e-9 ) );
				assert.ok( best <= corner / 3 + 1e-3, `edge ${e} cp ${k}: ${best} vs corner ${corner}` );

			}

		}

		assert.ok( worst < 0.34 );

	} );

	test( `${id} sphere: beta = 0 stays on the sphere; clusters push bundles outward`, () => {

		const pos = spherePos( g );
		const cp = centroids( g, pos );
		const R = g.layoutRadius;
		let maxOut = 0;
		for ( let e = 0; e < g.E; e ++ ) {

			for ( const tt of [ 0.25, 0.5, 0.75 ] ) {

				const p0 = evalEdgeCurveCPU( g, e, tt, pos, cp, { beta: 0, layoutBlend: 1 } ).position;
				assert.ok( Math.abs( Math.hypot( ...p0 ) - R ) < 1e-6 * R );
				const p1 = evalEdgeCurveCPU( g, e, tt, pos, cp, { beta: 1, layoutBlend: 1 } ).position;
				const r1 = Math.hypot( ...p1 );
				assert.ok( r1 >= R * ( 1 - 1e-6 ) );
				maxOut = Math.max( maxOut, r1 / R );
				// highlight pulls the cluster levels inward (toward highlightLevel)
				const ph = evalEdgeCurveCPU( g, e, tt, pos, cp, { beta: 1, layoutBlend: 1, highlight: 1, highlightLevel: 0.75 } ).position;
				assert.ok( Math.hypot( ...ph ) <= r1 + 1e-6 * R );

			}

		}

		assert.ok( maxOut > 1.05 );

	} );

}

test( 'D = 0 hierarchy: every edge is the straight segment (L = 2)', () => {

	const g = load( 'lesmis', true );
	assert.equal( g.D, 0 );
	const cp = centroids( g, g.initPos );
	for ( let e = 0; e < g.E; e ++ ) {

		const r = evalEdgeCurveCPU( g, e, 0.5, g.initPos, cp, { beta: 0.85 } );
		assert.equal( r.L, 2 );
		const mid = [ 0, 1, 2 ].map( ( q ) => 0.5 * ( g.initPos[ 4 * g.edges[ 2 * e ] + q ] + g.initPos[ 4 * g.edges[ 2 * e + 1 ] + q ] ) );
		assert.ok( dist( r.position, mid ) < 1e-6 * g.layoutRadius );

	}

} );

test( 'degenerate input: coincident endpoints and centroids stay finite', () => {

	const g = prepareGraph( normalizeGraph( { nodeCount: 4, edges: [ 0, 1, 1, 2, 2, 3, 0, 3, 0, 2 ], hierarchy: { method: 't', levels: [ [ 0, 0, 1, 1 ] ] } } ) );
	const pos = new Float32Array( 16 ); // all nodes at the origin = sphere center
	const cp = centroids( g, pos );
	for ( let e = 0; e < g.E; e ++ ) for ( const blend of [ 0, 0.5, 1 ] ) for ( const tt of [ 0, 0.3, 0.5, 1 ] ) {

		const r = evalEdgeCurveCPU( g, e, tt, pos, cp, { layoutBlend: blend } );
		assert.ok( finite( r.position ) && finite( r.tangent ) );

	}

} );

test( 'auto segments and opacity', () => {

	assert.equal( autoSegments( 254 ), 32 );
	assert.equal( autoSegments( 15000 ), 20 );
	assert.equal( autoSegments( 50000 ), 12 );
	assert.equal( autoSegments( 200000 ), 8 );
	assert.equal( autoSegments( 200000, 6, 2 ), 16 );
	assert.equal( autoSegments( 1000, 8 ), 41 ); // D = 3: 9 spans
	let prev = Infinity;
	for ( const E of [ 10, 100, 1000, 1e4, 1e5, 4e5 ] ) {

		const a = autoOpacity( E );
		assert.ok( a > 0 && a <= 0.8 && a <= prev );
		prev = a;

	}

	// strand defocus is bounded by density: full up to 5k edges, none from 200k
	assert.equal( autoDefocusLimit( 0 ), 10 );
	assert.equal( autoDefocusLimit( 254 ), 10 );
	assert.equal( autoDefocusLimit( 5000 ), 10 );
	assert.equal( autoDefocusLimit( 200000 ), 0 );
	assert.equal( autoDefocusLimit( 384444 ), 0 );
	prev = Infinity;
	for ( const E of [ 5000, 1e4, 4e4, 1e5, 2e5 ] ) {

		const d = autoDefocusLimit( E );
		assert.ok( d >= 0 && d <= prev );
		prev = d;

	}

	assert.ok( Math.abs( autoDefocusLimit( 40000 ) - 4.36 ) < 0.01 );

} );

// ── explicit control paths ───────────────────────────────────────────────────────────────────

/**
 * The curve of the first renderer, written out from the hierarchy (anc / edgeLca) without any
 * path arrays: the reference that hierarchicalPaths + the generalised mirror must reproduce.
 * Position only; frozen on purpose, do not "simplify" it into a call of evalEdgeCurveCPU.
 */
function legacyCurve( g, e, t, nodePos4, clusterPos4, o = {} ) {

	const beta = o.beta ?? 0.85, blend = o.layoutBlend ?? 0, R = o.sphereRadius ?? g.layoutRadius;
	const lvOff = 0.1, lvScale = 0.35, lvExp = 2, hl = o.highlight ?? 0, hlLevel = o.highlightLevel ?? 0.75;
	const D = g.D, s = g.edges[ 2 * e ], tg = g.edges[ 2 * e + 1 ];
	const lca = Math.min( g.edgeLca[ e ], D );
	const L = 2 + 2 * lca + ( lca < D ? 1 : 0 );
	const at = ( a, i ) => [ a[ 4 * i ], a[ 4 * i + 1 ], a[ 4 * i + 2 ] ];
	const pS = at( nodePos4, s ), pT = at( nodePos4, tg );
	const lerp3 = ( a, b, k ) => a.map( ( v, q ) => v + ( b[ q ] - v ) * k );
	const len = ( a ) => Math.hypot( a[ 0 ], a[ 1 ], a[ 2 ] );
	const logMap = ( r ) => {

		const l = Math.max( len( r ), 1e-6 ), dn = r.map( ( v ) => v / l );
		const perp = [ dn[ 0 ], dn[ 1 ], 0 ], sn = len( perp );
		const k = sn > 1e-7 ? Math.atan2( sn, dn[ 2 ] ) / sn : 1;
		return perp.map( ( v ) => v * k );

	};

	const levS = len( pS ) / R, levT = len( pT ) / R, vS = logMap( pS ), vT = logMap( pT );
	const point = ( k ) => {

		if ( k === 0 ) return { p: pS, lev: levS };
		if ( k === L - 1 ) return { p: pT, lev: levT };
		const srcSide = 2 * k <= L - 1;
		const lv = srcSide ? k - 1 : L - 2 - k;
		const cl = g.anc[ ( srcSide ? s : tg ) * D + lv ];
		const level = 1 + lvOff + lvScale * ( lv > 0 ? Math.pow( lv, lvExp ) : 0 );
		return { p: at( clusterPos4, cl ), lev: level + ( hlLevel - level ) * hl };

	};

	const x = Math.min( Math.max( t, 0 ), 1 ) * ( L + 1 );
	const j = Math.min( Math.floor( x ), L ), u = x - j, v = 1 - u, u2 = u * u, u3 = u2 * u;
	const b = [ ( v * v * v ) / 6, ( 3 * u3 - 6 * u2 + 4 ) / 6, ( - 3 * u3 + 3 * u2 + 3 * u + 1 ) / 6, u3 / 6 ];
	const cF = [ 0, 0, 0 ], V = [ 0, 0, 0 ];
	let cL = 0;
	for ( let i = 0; i < 4; i ++ ) {

		const k = Math.min( Math.max( j + i - 2, 0 ), L - 1 ), a = k / ( L - 1 );
		const { p, lev } = point( k );
		const f = lerp3( lerp3( pS, pT, a ), p, beta ), w = lerp3( lerp3( vS, vT, a ), logMap( p ), beta );
		for ( let q = 0; q < 3; q ++ ) {

			cF[ q ] += b[ i ] * ( f[ q ] - pS[ q ] ); V[ q ] += b[ i ] * w[ q ];

		}

		cL += b[ i ] * ( ( levS + ( levT - levS ) * a ) * ( 1 - beta ) + lev * beta );

	}

	const r = len( V ), sinc = r > 1e-4 ? Math.sin( r ) / r : 1 - ( r * r ) / 6;
	const dir = [ V[ 0 ] * sinc, V[ 1 ] * sinc, Math.cos( r ) + V[ 2 ] * sinc ];
	return lerp3( pS.map( ( p, q ) => p + cF[ q ] ), dir.map( ( d ) => d * R * cL ), blend );

}

for ( const id of [ 'karate', 'lesmis', 'football', 'celegansneural' ] ) {

	test( `${id}: explicit hierarchical paths reproduce the first renderer's curve`, () => {

		const g = load( id );
		const paths = hierarchicalPaths( g );
		const sphere = spherePos( g );
		const cases = [
			[ g.initPos, { layoutBlend: 0 } ],
			[ sphere, { layoutBlend: 1 } ],
			[ sphere, { layoutBlend: 1, highlight: 0.6 } ],
			[ g.initPos.map( ( v, i ) => 0.5 * v + 0.5 * sphere[ i ] ), { layoutBlend: 0.5 } ],
		];
		let worst = 0;
		for ( const [ pos, o ] of cases ) {

			const cp = centroids( g, pos );
			for ( let e = 0; e < g.E; e ++ ) for ( const beta of [ 0.3, 0.85, 1 ] ) for ( const tt of [ 0, 0.07, 0.25, 0.5, 0.81, 1 ] ) {

				const want = legacyCurve( g, e, tt, pos, cp, { ...o, beta } );
				const implicit = evalEdgeCurveCPU( g, e, tt, pos, cp, { ...o, beta } ); // default path set
				const explicit = evalEdgeCurveCPU( g, e, tt, pos, cp, { ...o, beta, paths } );
				worst = Math.max( worst, dist( want, implicit.position ), dist( want, explicit.position ) );

			}

		}

		assert.ok( worst < 1e-9 * g.layoutRadius, `max deviation ${worst}` );

	} );

}

test( 'explicit paths: no interior points = the straight segment; node refs are honoured', () => {

	const g = load( 'lesmis' );
	const pos = g.initPos, cp = centroids( g, pos );
	const straight = straightPaths( g.E );
	for ( let e = 0; e < g.E; e += 7 ) {

		// the spline of [ s, s, s, t, t, t ] stays on the segment (its speed is not uniform)
		const s = nodeAt( pos, g.edges[ 2 * e ] ), t = nodeAt( pos, g.edges[ 2 * e + 1 ] ), d = dist( s, t );
		let last = - 1;
		for ( const tt of [ 0, 0.2, 0.37, 0.5, 0.8, 1 ] ) {

			const r = evalEdgeCurveCPU( g, e, tt, pos, cp, { paths: straight, beta: 1 } );
			assert.equal( r.L, 2 );
			const k = dist( r.position, s ) / d;
			assert.ok( dist( r.position, s.map( ( v, q ) => v + ( t[ q ] - v ) * k ) ) < 1e-9 * g.layoutRadius );
			assert.ok( k > last && k <= 1 + 1e-12 );
			if ( tt === 0.5 ) assert.ok( Math.abs( k - 0.5 ) < 1e-12 );
			last = k;

		}

	}

	// edge 0 routed over two nodes: the control polygon is [ s, a, b, t ] with the nodes' positions
	const a = 5, b = 9;
	const pathStart = new Uint32Array( g.E + 1 ).fill( 2 );
	pathStart[ 0 ] = 0;
	const paths = { pathStart, pathRefs: Uint32Array.of( a, b ), maxInterior: 2 };
	const r = evalEdgeCurveCPU( g, 0, 0.5, pos, cp, { paths, beta: 1 } );
	assert.equal( r.L, 4 );
	assert.deepEqual( r.controlPoints, [ nodeAt( pos, g.edges[ 0 ] ), nodeAt( pos, a ), nodeAt( pos, b ), nodeAt( pos, g.edges[ 1 ] ) ] );
	assert.equal( evalEdgeCurveCPU( g, 1, 0.5, pos, cp, { paths } ).L, 2 ); // the other edges are straight
	// sphere layout: nodes on a path keep their own radial level, so the whole curve stays on the sphere
	const sp = spherePos( g ), scp = centroids( g, sp );
	for ( const tt of [ 0.1, 0.3, 0.5, 0.7, 0.9 ] ) {

		const p = evalEdgeCurveCPU( g, 0, tt, sp, scp, { paths, beta: 1, layoutBlend: 1 } ).position;
		assert.ok( Math.abs( Math.hypot( ...p ) - g.layoutRadius ) < 1e-6 * g.layoutRadius );

	}

} );

test( 'path refs: GPU encoding and memory budget', () => {

	const g = load( 'lesmis' );
	const paths = hierarchicalPaths( g );
	const out = encodePathRefs( g, paths.pathRefs, paths.pathRefs.length, new Uint32Array( paths.pathRefs.length ) );
	for ( let k = 0; k < out.length; k ++ ) {

		const c = paths.pathRefs[ k ] - g.N;
		assert.equal( out[ k ] >>> 31, 1 ); // hierarchical paths only hold clusters
		assert.equal( out[ k ] & 0x07ffffff, c );
		assert.equal( ( out[ k ] >>> 27 ) & 15, g.clusterHeight[ c ] );

	}

	assert.deepEqual( Array.from( encodePathRefs( g, Uint32Array.of( 3, g.N + 2 ), 2, new Uint32Array( 2 ) ) ), [ 3, ( 0x80000000 | ( g.clusterHeight[ 2 ] << 27 ) | 2 ) >>> 0 ] );
	// 384,444 edges (the dense LFR preset): 8 segments fit the 64 MiB budget, 16 do not
	assert.equal( maxSegmentsForBudget( 384444 ), 9 );
	assert.ok( 384444 * 9 * 16 < 64 * 1048576 );
	assert.equal( maxSegmentsForBudget( 0 ), Math.floor( 64 * 1048576 / 16 ) - 1 );
	assert.equal( maxSegmentsForBudget( 1000, 16 * 1000 ), 1 ); // never below one segment

} );

test( 'curve cache: path upload, validation, segments', () => {

	const g = load( 'lesmis' );
	const buffers = new GraphBuffers( g );
	const curves = new EdgeCurves();
	curves.setGraph( buffers, g, { segments: 12 } );
	const paths = hierarchicalPaths( g );
	const info = buffers.edgeInfo.value.array;
	assert.equal( info.length, 4 * ( g.E + 1 ) );
	for ( let e = 0; e <= g.E; e ++ ) assert.equal( info[ 4 * e + 3 ], paths.pathStart[ e ] );
	for ( let e = 0; e < g.E; e ++ ) assert.deepEqual( [ info[ 4 * e ], info[ 4 * e + 1 ], info[ 4 * e + 2 ] ], [ g.edges[ 2 * e ], g.edges[ 2 * e + 1 ], g.edgeLca[ e ] ] );
	assert.equal( curves.stats.pathRefs, paths.pathRefs.length );
	assert.equal( curves.stats.maxInterior, paths.maxInterior );
	assert.equal( curves.segments, 12 );
	assert.equal( curves.uniforms.stride.value, 13 );
	assert.equal( curves.samples.value.array.length, 4 * g.E * 13 );
	assert.equal( curves.stats.bytes, 16 * g.E * 13 );

	// more segments: the buffer grows once and the renderer is told; fewer: it is kept
	let reallocs = 0;
	curves.onRealloc = () => reallocs ++;
	assert.equal( curves.setSegments( 40 ), 40 );
	assert.equal( reallocs, 1 );
	assert.equal( curves.setSegments( 8 ), 8 );
	assert.equal( reallocs, 1 );
	assert.equal( curves.capacity, 41 );
	assert.equal( curves.uniforms.stride.value, 9 );
	curves.memoryBudget = 16 * g.E * 5; // room for 5 samples per edge
	assert.equal( curves.maxSegments(), 4 );
	assert.equal( curves.setSegments( 64 ), 4 );

	// swapping the set: straight lines, then a bad one
	curves.setPaths( straightPaths( g.E ) );
	assert.equal( curves.stats.pathRefs, 0 );
	for ( let e = 0; e <= g.E; e ++ ) assert.equal( info[ 4 * e + 3 ], 0 );
	assert.throws( () => curves.setPaths( { pathStart: new Uint32Array( g.E ), pathRefs: new Uint32Array( 0 ) } ), /expected E \+ 1/ );
	assert.throws( () => curves.setPaths( { pathStart: Array.from( paths.pathStart ), pathRefs: paths.pathRefs } ), TypeError );
	const short = { pathStart: paths.pathStart, pathRefs: paths.pathRefs.subarray( 0, 10 ) };
	assert.throws( () => curves.setPaths( short ), /exceeds pathRefs.length/ );
	const bad = { pathStart: paths.pathStart, pathRefs: paths.pathRefs.slice() };
	bad.pathRefs[ 3 ] = g.N + g.C;
	assert.throws( () => curves.setPaths( bad ), /neither a node/ );
	const down = { pathStart: paths.pathStart.slice(), pathRefs: paths.pathRefs };
	down.pathStart[ 5 ] = down.pathStart[ 4 ] - 1;
	assert.throws( () => curves.setPaths( down ), /decreases|exceeds/ );
	// a failed swap leaves the previous set in place
	assert.equal( curves.stats.pathRefs, 0 );
	curves.setPaths( null );
	assert.equal( curves.stats.pathRefs, paths.pathRefs.length );
	curves.dispose();

} );

test( 'curve cache: room reserved for a finer sampling (first sphere morph)', () => {

	const g = load( 'lesmis' );
	const buffers = new GraphBuffers( g );
	const curves = new EdgeCurves();
	let reallocs = 0;
	curves.onRealloc = () => reallocs ++;
	curves.setGraph( buffers, g, { segments: 32, reserve: 64 } );
	assert.equal( curves.segments, 32 );
	assert.equal( curves.uniforms.stride.value, 33 );
	assert.equal( curves.capacity, 65 );
	assert.equal( curves.stats.bytes, 16 * g.E * 65 );
	// the morph: only the stride changes
	const samples = curves.samples;
	assert.equal( curves.setSegments( 64 ), 64 );
	assert.equal( curves.samples, samples );
	assert.equal( curves.uniforms.stride.value, 65 );
	assert.equal( reallocs, 0 );
	assert.equal( curves.stats.reallocations, 0 );
	// not honoured past the reserve budget, nor past the memory budget
	curves.reserveBudget = 16 * g.E * 40;
	curves.setGraph( buffers, g, { segments: 32, reserve: 64 } );
	assert.equal( curves.capacity, 33 );
	curves.reserveBudget = Infinity;
	curves.memoryBudget = 16 * g.E * 41; // at most 40 segments
	curves.setGraph( buffers, g, { segments: 32, reserve: 64 } );
	assert.equal( curves.capacity, 41 );
	assert.equal( curves.segments, 32 );
	// a reserve below the wanted count changes nothing
	curves.setGraph( buffers, g, { segments: 32, reserve: 8 } );
	assert.equal( curves.capacity, 33 );
	assert.equal( reallocs, 0 );
	curves.dispose();

} );

test( 'curve cache: the kernel runs only when something changed', () => {

	const g = load( 'lesmis' );
	const buffers = new GraphBuffers( g );
	const curves = new EdgeCurves();
	curves.setGraph( buffers, g, { segments: 8 } );
	const app = { frame: 0, time: 0, renderer: { computes: 0, compute() { this.computes ++; } } };
	const step = ( n = 1 ) => {

		let ran = 0;
		for ( let i = 0; i < n; i ++ ) {

			app.frame ++; app.time += 1 / 60;
			if ( curves.update( app ) ) ran ++;

		}

		return ran;

	};

	// moveFrame = Infinity ("unknown"): every frame evaluates
	assert.equal( step( 5 ), 5 );
	// the layout stamps the frame in which it last moved the nodes: one more run, then nothing
	buffers.moveFrame = app.frame;
	assert.equal( step( 50 ), 1 );
	assert.equal( step( 500 ), 0 );
	assert.equal( curves.stats.updates, 6 );
	assert.equal( curves.stats.skipped, 549 );
	assert.equal( app.renderer.computes, 6 );

	// each of these costs exactly one evaluation
	curves.uniforms.beta.value = 0.5;
	assert.equal( step( 10 ), 1 );
	curves.setPaths( straightPaths( g.E ) );
	assert.equal( step( 10 ), 1 );
	curves.setSegments( 6 );
	assert.equal( step( 10 ), 1 );
	curves.invalidate();
	assert.equal( step( 10 ), 1 );
	buffers.uniforms.graphCenter.value.set( 1, 0, 0 );
	assert.equal( step( 10 ), 1 );
	curves.uniforms.highlightLevel.value = 0.6;
	assert.equal( step( 10 ), 1 );

	// nodes move for 20 frames (the layout runs before the curves in a frame)
	let ran = 0;
	for ( let i = 0; i < 20; i ++ ) {

		buffers.moveFrame = app.frame + 1;
		ran += step();

	}

	assert.equal( ran, 20 );
	assert.equal( step( 30 ), 1 ); // one settling run (covers a move stamped after our update)
	assert.equal( step( 30 ), 0 );

	// highlight easing bends curves only in sphere mode
	const target = buffers.nodeStateTarget.value, attr = buffers.edgeAttr.value;
	const write = ( attribute, index, value ) => {

		attribute.array[ index ] = value;
		attribute.needsUpdate = true;

	};

	write( target, 4 * 3, 1 ); // a CPU write: node 3 selected
	assert.equal( step( 100 ), 0 );
	buffers.uniforms.layoutBlend.value = 1;
	assert.equal( step( 100 ), 1 ); // the blend changed
	write( target, 4 * 3, 0 ); // deselected
	let eased = step( 200 );
	assert.ok( eased >= 60 && eased <= 80, `${eased} runs while the highlight eases (1.2 s at 60 fps)` );
	assert.equal( step( 100 ), 0 );
	// writes the kernel does not read cost nothing: hover (w), neighbour and dim flags (y, z), an
	// upload without a change, the per-edge width, metric and seed
	write( target, 4 * 5 + 3, 1 );
	assert.equal( step( 50 ), 0 );
	write( target, 4 * 5 + 1, 1 );
	write( target, 4 * 7 + 2, 1 );
	assert.equal( step( 50 ), 0 );
	target.needsUpdate = true;
	buffers.nodeState.value.needsUpdate = true;
	assert.equal( step( 50 ), 0 );
	write( attr, 4 * 2, 2 );
	write( attr, 4 * 2 + 2, 0.5 );
	assert.equal( step( 50 ), 0 );
	// the highlight weight of an edge and the eased state itself are read
	write( attr, 4 * 2 + 1, 1 );
	eased = step( 200 );
	assert.ok( eased >= 60 && eased <= 80, `${eased} runs after an edge highlight` );
	write( buffers.nodeState.value, 4 * 9, 1 );
	eased = step( 200 );
	assert.ok( eased >= 60 && eased <= 80, `${eased} runs after a write to the eased state` );
	assert.equal( step( 100 ), 0 );
	// morphing: every frame
	ran = 0;
	for ( let i = 0; i < 10; i ++ ) {

		buffers.uniforms.layoutBlend.value = 1 - ( i + 1 ) / 10;
		ran += step();

	}

	assert.equal( ran, 10 );
	assert.equal( step( 50 ), 0 );
	// invalidate( seconds ) keeps evaluating for that long
	curves.invalidate( 0.5 );
	const forced = step( 100 );
	assert.ok( forced >= 29 && forced <= 32, `${forced}` );
	curves.dispose();

} );
