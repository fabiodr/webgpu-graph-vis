// Tests for src/render/EdgeRenderer.js: the pure-JS curve mirror (evalEdgeCurveCPU), which the dev
// page (dev/edges.html, window.__verify) checks against the actual GPU shader code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';
import { evalEdgeCurveCPU, autoSegments, autoOpacity } from '../src/render/EdgeRenderer.js';

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

} );
