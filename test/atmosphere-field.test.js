import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PerspectiveCamera, Scene, Vector3 } from 'three/webgpu';
import { uniform, instancedArray } from 'three/tsl';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';
import {
	FIELD_TIERS, levelCounts, chooseFieldLevel, densityIntegral, opticalDepth, clusterStats, clusterColors,
	fieldColor, compactness, emphasisTargets, quadCoverage, fieldBudget, FIELD_MIN_SHRINK, ClusterField,
	typicalCoverage, overviewFactor, fieldGraphFactor, FIELD_RADIANCE, FIELD_FOCUS_KEEP,
} from '../src/render/ClusterField.js';
import { gfxState } from '../src/render/gfx/shared.js';

const close = ( a, b, eps, msg ) => assert.ok( Math.abs( a - b ) <= eps, `${msg || ''} ${a} vs ${b}` );
const STRONG = Math.fround( 1.6 ), WEAK = Math.fround( 0.4 ); // emphasis lives in a Float32Array
const lesmis = () => prepareGraph( normalizeGraph( JSON.parse( readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ) ) ) );

// ── level choice ─────────────────────────────────────────────────────────────────────────────

test( 'chooseFieldLevel: the level inside 4…40 nearest to 10, else the nearest to the range', () => {

	assert.equal( chooseFieldLevel( [ 12, 6 ] ), 1, 'lesmis' );
	assert.equal( chooseFieldLevel( [ 25, 9, 7 ] ), 2, 'hsbm-500' );
	assert.equal( chooseFieldLevel( [ 52, 42 ] ), 2, 'lfr-2k: nothing in range, 42 is nearest' );
	assert.equal( chooseFieldLevel( [ 125, 23 ] ), 2 );
	assert.equal( chooseFieldLevel( [ 178, 110 ] ), 2 );
	assert.equal( chooseFieldLevel( [ 400, 60, 3 ] ), 3, '3 is nearer to the range than 60 (log scale)' );
	assert.equal( chooseFieldLevel( [ 400, 45, 2 ] ), 2 );
	assert.equal( chooseFieldLevel( [ 10, 10 ] ), 1, 'ties keep the finer level' );

} );

test( 'chooseFieldLevel: levels without structure are never chosen', () => {

	assert.equal( chooseFieldLevel( [] ), 0 );
	assert.equal( chooseFieldLevel( [ 1 ] ), 0 );
	assert.equal( chooseFieldLevel( [ 1, 1 ] ), 0 );
	assert.equal( chooseFieldLevel( [ 300, 1 ] ), 1 );
	assert.equal( chooseFieldLevel( [ 2 ] ), 1 );
	assert.equal( chooseFieldLevel( [ NaN, 8 ] ), 2 );

} );

test( 'chooseFieldLevel: honours a custom range', () => {

	assert.equal( chooseFieldLevel( [ 25, 9, 7 ], { min: 20, max: 60, ideal: 30 } ), 1 );
	assert.equal( chooseFieldLevel( [ 25, 9, 7 ], { min: 2, max: 8, ideal: 6 } ), 3 );

} );

test( 'levelCounts: matches the prepared graph', () => {

	const g = lesmis();
	const counts = levelCounts( g );
	assert.equal( counts.length, g.D );
	assert.equal( counts.reduce( ( a, b ) => a + b, 0 ) + 1, g.C, 'all clusters but the root' );
	for ( let l = 0; l < g.D; l ++ ) {

		for ( let c = g.levelOffset[ l ]; c < g.levelOffset[ l + 1 ]; c ++ ) assert.equal( g.clusterHeight[ c ], l + 1 );

	}

} );

// ── density integral ─────────────────────────────────────────────────────────────────────────

/** ∫ ( 1 − r² / R² )² ds along the line, by the midpoint rule. */
function numeric( impact, R, s0, s1, steps = 20000 ) {

	const h = Math.sqrt( Math.max( 0, R * R - impact * impact ) );
	const lo = Math.max( - h, s0 ), hi = Math.min( h, s1 );
	if ( ! ( hi > lo ) ) return 0;
	const ds = ( hi - lo ) / steps;
	let sum = 0;
	for ( let i = 0; i < steps; i ++ ) {

		const s = lo + ( i + 0.5 ) * ds;
		const q = 1 - ( impact * impact + s * s ) / ( R * R );
		sum += q * q * ds;

	}

	return sum;

}

test( 'densityIntegral: equals numeric integration for whole chords and for parts of them', () => {

	for ( const R of [ 0.5, 1, 37 ] ) {

		for ( const b of [ 0, 0.2, 0.5, 0.8, 0.97 ] ) {

			const impact = b * R;
			close( densityIntegral( impact, R ), numeric( impact, R, - Infinity, Infinity ), 1e-6 * R, `whole R=${R} b=${b}` );
			close( densityIntegral( impact, R ), ( 16 / 15 ) * R * Math.pow( 1 - b * b, 2.5 ), 1e-9 * R, 'closed form of the whole chord' );
			for ( const [ s0, s1 ] of [ [ - 0.3 * R, Infinity ], [ 0.1 * R, Infinity ], [ - Infinity, 0.25 * R ], [ - 0.4 * R, 0.2 * R ], [ 0.6 * R, 0.7 * R ] ] ) {

				close( densityIntegral( impact, R, s0, s1 ), numeric( impact, R, s0, s1 ), 1e-6 * R, `part R=${R} b=${b} [${s0}, ${s1}]` );

			}

		}

	}

} );

test( 'densityIntegral: zero outside the sphere, for empty intervals and degenerate spheres', () => {

	assert.equal( densityIntegral( 1, 1 ), 0 );
	assert.equal( densityIntegral( 2, 1 ), 0 );
	assert.equal( densityIntegral( 0, 1, 2, 3 ), 0 );
	assert.equal( densityIntegral( 0, 1, 0.5, 0.2 ), 0 );
	assert.equal( densityIntegral( 0, 0 ), 0 );
	assert.equal( densityIntegral( NaN, 1 ), 0 );

} );

test( 'opticalDepth: 1 through the centre from outside, monotone to the rim, thinner from inside', () => {

	const R = 3;
	close( opticalDepth( [ 0, 0, - 1 ], [ 0, 0, - 10 ], R ), 1, 1e-12 );
	let last = 2;
	for ( let i = 0; i <= 20; i ++ ) {

		// rays that miss the centre by i / 20 of the radius
		const tau = opticalDepth( [ 0, 0, - 1 ], [ ( i / 20 ) * R, 0, - 10 ], R );
		assert.ok( tau >= 0 && tau <= 1 + 1e-12 && tau < last, `monotone at ${i}: ${tau}` );
		last = tau;

	}

	close( last, 0, 1e-9, 'the silhouette' );
	// a camera at the centre sees half of the densest ray, in every direction
	for ( const d of [ [ 0, 0, - 1 ], [ 1, 0, 0 ], [ 0.6, 0.8, 0 ] ] ) close( opticalDepth( d, [ 0, 0, 0 ], R ), 0.5, 1e-12 );
	// inside, looking through the centre: more than half, less than all
	const inside = opticalDepth( [ 0, 0, - 1 ], [ 0, 0, - 1.5 ], R );
	assert.ok( inside > 0.5 && inside < 1 );
	close( inside, numeric( 0, R, - 1.5, Infinity ) / ( 16 * R / 15 ), 1e-6 );
	// a sphere behind the camera contributes nothing
	assert.equal( opticalDepth( [ 0, 0, - 1 ], [ 0, 0, 10 ], R ), 0 );

} );

test( 'opticalDepth: continuous while the camera moves from outside into the volume', () => {

	const R = 2;
	let prev = opticalDepth( [ 0, 0, - 1 ], [ 0.3, 0, - 4 ], R );
	for ( let z = - 4; z <= 4; z += 0.01 ) {

		const tau = opticalDepth( [ 0, 0, - 1 ], [ 0.3, 0, z ], R );
		assert.ok( Math.abs( tau - prev ) < 0.012, `step at z=${z}` );
		prev = tau;

	}

	close( prev, 0, 1e-9 );

} );

// ── statistics, colour, emphasis ─────────────────────────────────────────────────────────────

test( 'clusterStats: sizes, centroids and rms radii of a level', () => {

	const g = lesmis();
	const pos = new Float32Array( 4 * g.N );
	for ( let i = 0; i < g.N; i ++ ) {

		pos[ 4 * i ] = Math.sin( i * 1.7 ) * 10; pos[ 4 * i + 1 ] = Math.cos( i * 0.9 ) * 6; pos[ 4 * i + 2 ] = Math.sin( i * 0.31 + 2 ) * 8;

	}

	for ( let height = 1; height <= g.D; height ++ ) {

		const s = clusterStats( g, pos, height );
		assert.equal( s.count, g.levelOffset[ height ] - g.levelOffset[ height - 1 ] );
		assert.equal( s.total, g.N );
		let total = 0;
		for ( let k = 0; k < s.count; k ++ ) {

			assert.equal( s.size[ k ], g.clusterMass[ s.offset + k ], 'size = mass (unit masses)' );
			total += s.size[ k ];
			// brute force
			let cx = 0, cy = 0, cz = 0, n = 0;
			for ( let i = 0; i < g.N; i ++ ) if ( g.anc[ i * g.D + height - 1 ] === s.offset + k ) {

				cx += pos[ 4 * i ]; cy += pos[ 4 * i + 1 ]; cz += pos[ 4 * i + 2 ]; n ++;

			}

			cx /= n; cy /= n; cz /= n;
			let r2 = 0;
			for ( let i = 0; i < g.N; i ++ ) if ( g.anc[ i * g.D + height - 1 ] === s.offset + k ) {

				r2 += ( pos[ 4 * i ] - cx ) ** 2 + ( pos[ 4 * i + 1 ] - cy ) ** 2 + ( pos[ 4 * i + 2 ] - cz ) ** 2;

			}

			close( s.center[ 3 * k ], cx, 1e-4 ); close( s.center[ 3 * k + 1 ], cy, 1e-4 ); close( s.center[ 3 * k + 2 ], cz, 1e-4 );
			close( s.rms[ k ], Math.sqrt( r2 / n ), 1e-3 );

		}

		assert.equal( total, g.N );
		assert.ok( s.graphRms > 0 );
		// reuse of the output object
		assert.equal( clusterStats( g, pos, height, s ), s );

	}

} );

test( 'clusterStats: ignores NaN positions', () => {

	const g = lesmis();
	const pos = new Float32Array( 4 * g.N ).fill( 1 );
	pos[ 0 ] = NaN;
	const s = clusterStats( g, pos, 1 );
	assert.equal( s.total, g.N - 1 );
	assert.ok( s.rms.every( ( v ) => v === 0 ) && s.center.every( Number.isFinite ) );

} );

test( 'clusterColors + fieldColor: mean member colour, luminance 1, inside the channel limit', () => {

	const g = lesmis();
	const colors = new Float32Array( 4 * g.N );
	// every node carries the index of its community in red, 1 in green
	for ( let i = 0; i < g.N; i ++ ) colors.set( [ g.anc[ i * g.D ] / 16, 1, ( i % 2 ) * 0.5, 1 ], 4 * i );
	const mean = clusterColors( g, colors, 1 );
	const K = g.levelOffset[ 1 ] - g.levelOffset[ 0 ];
	assert.equal( mean.length, 3 * K );
	for ( let k = 0; k < K; k ++ ) {

		close( mean[ 3 * k ], k / 16, 1e-6 );
		close( mean[ 3 * k + 1 ], 1, 1e-6 );
		assert.ok( mean[ 3 * k + 2 ] >= 0 && mean[ 3 * k + 2 ] <= 0.5 );

	}

	assert.equal( clusterColors( g, colors, 1, mean ), mean, 'reuses the output array' );

	const lum = ( c ) => 0.2126 * c[ 0 ] + 0.7152 * c[ 1 ] + 0.0722 * c[ 2 ];
	for ( const rgb of [ [ 0.8, 0.1, 0.1 ], [ 0.1, 0.6, 0.2 ], [ 0.9, 0.8, 0.1 ], [ 0.3, 0.3, 0.3 ] ] ) {

		close( lum( fieldColor( ...rgb ) ), 1, 1e-9, 'luminance 1' );
		close( lum( fieldColor( ...rgb, 0.5 ) ), 1, 1e-9, 'saturation keeps the luminance' );
		const grey = fieldColor( ...rgb, 0 );
		close( grey[ 0 ], grey[ 1 ], 1e-9 ); close( grey[ 1 ], grey[ 2 ], 1e-9 );

	}

	// deep blue cannot reach luminance 1 inside the limit: darker, never brighter than maxChannel
	const blue = fieldColor( 0.02, 0.03, 0.9 );
	assert.ok( Math.max( ...blue ) <= 5 + 1e-9 && lum( blue ) < 1 );
	assert.deepEqual( fieldColor( 0, 0, 0 ), [ 1, 1, 1 ], 'black members give a neutral glow' );

} );

test( 'compactness: 1 for clumps, 0 for communities spread through the graph, smooth between', () => {

	// a community that fills its share of the volume
	close( compactness( 10 * Math.cbrt( 0.1 ), 100, 10, 1000 ), 1, 1e-12 );
	assert.equal( compactness( 2, 100, 10, 1000 ), 1, 'a dense clump' );
	assert.equal( compactness( 10, 100, 10, 1000 ), 0, 'as wide as the whole graph' );
	let last = 1;
	for ( let ratio = 1; ratio <= 2.4; ratio += 0.05 ) {

		const w = compactness( ratio * 10 * Math.cbrt( 0.1 ), 100, 10, 1000 );
		assert.ok( w >= 0 && w <= 1 && w <= last + 1e-12 );
		last = w;

	}

	assert.equal( last, 0 );
	assert.equal( compactness( 1, 0, 10, 1000 ), 1, 'degenerate input does not hide a volume' );
	assert.equal( compactness( 1, 10, 0, 1000 ), 1 );

} );

test( 'emphasisTargets: selection, focus on the level, focus below and above the level', () => {

	const g = lesmis();
	const level1 = Uint32Array.from( { length: g.levelOffset[ 1 ] }, ( _, k ) => k );
	// nothing selected: neutral
	assert.ok( emphasisTargets( g, level1, - 1, - 1 ).every( ( v ) => v === 1 ) );
	// a selected node lights its community
	const node = 11, own = g.anc[ node * g.D ];
	let e = emphasisTargets( g, level1, - 1, node );
	for ( let k = 0; k < level1.length; k ++ ) assert.equal( e[ k ], level1[ k ] === own ? STRONG : WEAK );
	// focus wins over the selection
	const other = ( own + 1 ) % level1.length;
	e = emphasisTargets( g, level1, other, node );
	for ( let k = 0; k < level1.length; k ++ ) assert.equal( e[ k ], level1[ k ] === other ? STRONG : WEAK );
	// a coarser focused cluster lights every volume inside it
	if ( g.D >= 2 ) {

		const coarse = g.clusterParent[ own ];
		e = emphasisTargets( g, level1, coarse, - 1 );
		let lit = 0;
		for ( let k = 0; k < level1.length; k ++ ) {

			assert.equal( e[ k ], g.clusterParent[ level1[ k ] ] === coarse ? STRONG : WEAK );
			if ( e[ k ] > 1 ) lit ++;

		}

		assert.ok( lit >= 1 );
		// volumes of the coarse level, focus on a fine cluster: its parent is lit
		const level2 = Uint32Array.from( { length: g.levelOffset[ 2 ] - g.levelOffset[ 1 ] }, ( _, k ) => g.levelOffset[ 1 ] + k );
		e = emphasisTargets( g, level2, own, - 1 );
		for ( let k = 0; k < level2.length; k ++ ) assert.equal( e[ k ], level2[ k ] === coarse ? STRONG : WEAK );

	}

	// the root and invalid indices are no focus
	assert.ok( emphasisTargets( g, level1, g.root, - 1 ).every( ( v ) => v === 1 ) );
	assert.ok( emphasisTargets( g, level1, g.C + 5, - 1 ).every( ( v ) => v === 1 ) );
	assert.equal( emphasisTargets( g, new Uint32Array( 0 ), 0, 0 ).length, 0 );

} );

// ── overdraw budget ──────────────────────────────────────────────────────────────────────────

test( 'quadCoverage: share of the view of a volume quad, clipped to the view, 1 from inside', () => {

	const P11 = 1 / Math.tan( Math.PI / 8 ), P00 = P11 / 1.6; // 45° vertical, 16:10
	// far and centred: the tangent billboard, a square of half side h = R·d / √( d² − R² )
	const d = 100, R = 5, h = R * d / Math.sqrt( d * d - R * R );
	close( quadCoverage( 0, 0, - d, R, P00, P11 ), ( h * P00 / d ) * ( h * P11 / d ), 1e-9 );
	// twice as near: four times the area
	close( quadCoverage( 0, 0, - 50, R, P00, P11 ) / quadCoverage( 0, 0, - 100, R, P00, P11 ), 4, 0.05 );
	// off to the side: only the part inside the view counts; far outside: nothing
	const edge = d / P00; // x at the right border of the view at this depth
	close( quadCoverage( edge, 0, - d, R, P00, P11 ), 0.5 * quadCoverage( 0, 0, - d, R, P00, P11 ), 1e-3 );
	assert.equal( quadCoverage( 3 * edge, 0, - d, R, P00, P11 ), 0 );
	// larger than the view, inside, right in front: one view, never more
	assert.equal( quadCoverage( 0, 0, - 12, 10, P00, P11 ), 1 );
	assert.equal( quadCoverage( 1, 2, 3, 10, P00, P11 ), 1, 'the camera is inside' );
	assert.equal( quadCoverage( 0, 0, - 10.9, 10, P00, P11 ), 1, 'within 1.1 radii: the full-screen quad' );
	// behind the camera, degenerate
	assert.equal( quadCoverage( 0, 0, 50, 5, P00, P11 ), 0 );
	assert.equal( quadCoverage( 0, 0, - 50, 0, P00, P11 ), 0 );
	for ( let z = - 200; z < - 11; z += 7 ) {

		const c = quadCoverage( 3, - 2, z, 10, P00, P11 );
		assert.ok( c > 0 && c <= 1 );

	}

} );

test( 'fieldBudget: small volumes first, the largest hazes go, emphasis first, never above the budget', () => {

	// an overview: everything fits
	let r = fieldBudget( [ 0.2, 0.3, 0.1, 0.25 ], [ 1, 1, 1, 1 ], 3 );
	assert.deepEqual( Array.from( r.weights ), [ 1, 1, 1, 1 ] );
	close( r.total, 0.85, 1e-6 );
	assert.equal( r.culled, 0 );
	// inside the graph: six volumes fill the view, five small ones lie beyond
	const cover = [ 1, 1, 0.1, 1, 0.2, 1, 0.05, 1, 0.3, 0.15, 1 ];
	const ones = cover.map( () => 1 );
	r = fieldBudget( cover, ones, 3 );
	assert.ok( r.total <= 3 + 1e-9 );
	for ( let j = 0; j < cover.length; j ++ ) if ( cover[ j ] < 1 ) assert.equal( r.weights[ j ], 1, 'the small ones stay' );
	assert.equal( r.drawn, 5 + 2, 'two of the six hazes fit' );
	assert.equal( r.culled, 4 );
	// which hazes: the ones the camera is deepest inside (smallest distance / radius)
	const depth = [ 0.9, 0.2, 5, 0.5, 5, 0.1, 5, 0.7, 5, 5, 0.3 ];
	r = fieldBudget( cover, ones, 3, { depth } );
	assert.equal( r.weights[ 5 ], 1 );
	assert.equal( r.weights[ 1 ], 1 );
	assert.equal( r.weights[ 0 ], 0 );
	// an emphasised volume goes first whatever its size
	const prio = ones.slice();
	prio[ 0 ] = 1.6;
	r = fieldBudget( cover, prio, 3, { depth } );
	assert.equal( r.weights[ 0 ], 1 );
	assert.ok( r.total <= 3 + 1e-9 );
	// volumes without light are neither drawn nor counted
	r = fieldBudget( [ 1, 1, 1 ], [ 0, 1, 1 ], 1 );
	assert.deepEqual( Array.from( r.weights ), [ 0, 1, 0 ] );
	assert.equal( r.culled, 1 );
	// the first volume is drawn even when it alone exceeds the budget
	r = fieldBudget( [ 1, 1 ], [ 1, 1 ], 0.5 );
	assert.equal( r.drawn, 1 );
	// any budget, any view: the sum stays inside (or it is the single first volume)
	let seed = 7;
	const rnd = () => ( seed = ( seed * 16807 ) % 2147483647 ) / 2147483647;
	for ( let n = 0; n < 200; n ++ ) {

		const K = 1 + Math.floor( rnd() * 60 );
		const c = Array.from( { length: K }, () => ( rnd() < 0.3 ? 1 : rnd() * rnd() ) );
		const p = Array.from( { length: K }, () => ( rnd() < 0.1 ? 1.6 : rnd() < 0.1 ? 0 : 1 ) );
		const budget = 0.5 + rnd() * 6;
		const b = fieldBudget( c, p, budget );
		assert.ok( b.total <= budget + 1e-6 || b.drawn === 1, `${b.total} > ${budget}` );
		let sum = 0;
		for ( let j = 0; j < K; j ++ ) if ( b.weights[ j ] ) sum += c[ j ];
		close( sum, b.total, 1e-5 );

	}

} );

test( 'fieldBudget: a volume at the border of the budget does not toggle (hysteresis)', () => {

	const cover = [ 0.5, 0.5, 0.5, 0.5 ];
	const first = fieldBudget( cover, [ 1, 1, 1, 1 ], 1.6 );
	assert.equal( first.drawn, 3 );
	// the view grows a little: without a memory the third volume would go, with it it stays
	const grown = [ 0.55, 0.55, 0.55, 0.55 ];
	assert.equal( fieldBudget( grown, [ 1, 1, 1, 1 ], 1.6 ).drawn, 2 );
	assert.equal( fieldBudget( grown, [ 1, 1, 1, 1 ], 1.6, { previous: first.weights } ).drawn, 3 );
	// well above the budget it goes
	assert.equal( fieldBudget( [ 0.7, 0.7, 0.7, 0.7 ], [ 1, 1, 1, 1 ], 1.6, { previous: first.weights } ).drawn, 2 );
	assert.ok( FIELD_MIN_SHRINK > 0.5 && FIELD_MIN_SHRINK < 1 );
	assert.ok( FIELD_TIERS.medium.overdraw <= FIELD_TIERS.high.overdraw && FIELD_TIERS.high.overdraw <= FIELD_TIERS.ultra.overdraw );
	assert.ok( FIELD_TIERS.ultra.overdraw <= 8, 'a few views of blending at most' );

} );

// ── the module without a renderer ────────────────────────────────────────────────────────────

test( 'ClusterField: settings clamp, the contract of the entry point holds', () => {

	const field = new ClusterField();
	assert.equal( field.order, 13 );
	assert.equal( field.strength, 0.5 );
	assert.equal( field.level, 'auto' );
	assert.equal( field.visible, true );
	field.strength = 7; assert.equal( field.strength, 1 );
	field.strength = - 1; assert.equal( field.strength, 0 );
	field.strength = 'x'; assert.equal( field.strength, 0 );
	field.absorb = 3; assert.equal( field.absorb, 1 );
	field.visible = 0; assert.equal( field.visible, false );
	assert.equal( field.focus( 3 ), field );
	assert.equal( field.focus( null ), field );
	field.level = 2; assert.equal( field.level, 2 ); // no graph yet: stored only
	assert.deepEqual( field.stats, { height: 0, count: 0, quads: 0, coveredPixels: 0, overdraw: 0, intensity: 0, budget: 0, culled: 0, shrink: 1, typical: 0, overview: 1, graph: 1 } );
	assert.equal( field.adaptive, true );
	assert.equal( field.absorb, 1 ); // set above
	assert.equal( new ClusterField().absorb, 0.15, 'a light veil: its edge in depth must not show' );
	assert.equal( new ClusterField( { adaptive: false } ).adaptive, false );
	assert.equal( field.maxOverdraw, null );
	field.maxOverdraw = 0; assert.equal( field.maxOverdraw, 0.25 );
	field.maxOverdraw = null; assert.equal( field.maxOverdraw, null );
	assert.deepEqual( field.volumes(), [] );
	field.update( 0.016, null ); // no graph: nothing happens
	assert.ok( FIELD_TIERS.medium.max <= FIELD_TIERS.high.max && FIELD_TIERS.low.max <= FIELD_TIERS.medium.max );

} );

// ── an overview feature ──────────────────────────────────────────────────────────────────────

test( 'typicalCoverage: the size of the volume behind a typical field pixel, whatever the number of volumes', () => {

	// an overview: many small shapes
	close( typicalCoverage( [ 0.05, 0.05, 0.05, 0.05 ] ), 0.05, 1e-12 );
	close( typicalCoverage( new Array( 100 ).fill( 0.05 ) ), 0.05, 1e-12, 'does not grow with the count' );
	// inside the graph: a few hazes that fill the view dominate whatever lies beyond
	const inside = [ 1, 1, 1, ...new Array( 30 ).fill( 0.03 ) ];
	assert.ok( typicalCoverage( inside ) > 0.75, `${typicalCoverage( inside )}` );
	// one community fills the view, the others are far shapes
	assert.ok( typicalCoverage( [ 1, 0.05, 0.05, 0.05, 0.05 ] ) > 0.8 );
	// weights: volumes without light do not count, emphasised ones count more
	close( typicalCoverage( [ 1, 0.1 ], [ 0, 1 ] ), 0.1, 1e-12 );
	assert.ok( typicalCoverage( [ 0.6, 0.1 ], [ 1.6, 0.4 ] ) > typicalCoverage( [ 0.6, 0.1 ], [ 1, 1 ] ) );
	assert.equal( typicalCoverage( [] ), 0 );
	assert.equal( typicalCoverage( [ 0, 0 ] ), 0 );
	// between the size of the smallest and of the largest volume
	let seed = 11;
	const rnd = () => ( seed = ( seed * 16807 ) % 2147483647 ) / 2147483647;
	for ( let n = 0; n < 100; n ++ ) {

		const c = Array.from( { length: 1 + Math.floor( rnd() * 20 ) }, () => 0.01 + 0.99 * rnd() );
		const t = typicalCoverage( c );
		assert.ok( t >= Math.min( ...c ) - 1e-12 && t <= Math.max( ...c ) + 1e-12 );

	}

} );

test( 'overviewFactor: 1 while the volumes are shapes, 0 once they are hazes, smooth and monotone between', () => {

	assert.equal( overviewFactor( 0 ), 1 );
	assert.equal( overviewFactor( 0.09 ), 1, 'the overviews measured in the app: 0.01 … 0.09' );
	assert.equal( overviewFactor( 0.12 ), 1 );
	assert.equal( overviewFactor( 0.32 ), 0, 'halfway into a graph: 0.32 … 0.42' );
	assert.equal( overviewFactor( 0.9 ), 0, 'the camera is inside one or more volumes' );
	close( overviewFactor( 0.22 ), 0.5, 1e-9 );
	let last = 1;
	for ( let t = 0; t <= 1; t += 0.01 ) {

		const f = overviewFactor( t );
		assert.ok( f <= last + 1e-12 && f >= 0 && f <= 1 );
		assert.ok( last - f < 0.08, 'no step' );
		last = f;

	}

	assert.equal( overviewFactor( 0.3, { full: 0.4, none: 0.8 } ), 1 );

} );

test( 'fieldGraphFactor: nothing for tiny graphs, everything for ordinary ones, nothing where the nodes are the cloud', () => {

	assert.equal( fieldGraphFactor( 10 ), 0 );
	assert.equal( fieldGraphFactor( 16 ), 0 );
	close( fieldGraphFactor( 32 ), 0.5, 1e-9 );
	assert.equal( fieldGraphFactor( 77 ), 1 );
	assert.equal( fieldGraphFactor( 2000 ), 1 );
	assert.equal( fieldGraphFactor( 5000 ), 1 );
	close( fieldGraphFactor( 12500 ), 0.5, 1e-9 );
	assert.equal( fieldGraphFactor( 20000 ), 0 );
	assert.equal( fieldGraphFactor( 1e6 ), 0 );
	let last = 1;
	for ( let n = 5000; n <= 20000; n += 500 ) {

		const f = fieldGraphFactor( n );
		assert.ok( f <= last );
		last = f;

	}

	// without the upper end (field.adaptive = false) large graphs stay at 1
	assert.equal( fieldGraphFactor( 30000, { large: null } ), 1 );
	assert.equal( fieldGraphFactor( 10, { large: null } ), 0 );
	assert.equal( fieldGraphFactor( 300, { small: [ 100, 500 ] } ), 0.5 );

} );

/**
 * A field on a fake app: `sizes.length` communities on a ring of radius `ring` in the z = 0 plane,
 * each a disc of radius `spread`; one hierarchy level. The camera looks at the origin from z.
 */
function fakeField( options = {}, { sizes = [ 40, 40, 40, 40, 40, 40 ], ring = 300, spread = 40 } = {} ) {

	const N = sizes.reduce( ( a, b ) => a + b, 0 ), K = sizes.length;
	const anc = new Uint32Array( N ), positions = new Float32Array( 4 * N ), colors = new Float32Array( 4 * N );
	let i = 0;
	for ( let c = 0; c < K; c ++ ) {

		const cx = ring * Math.cos( 2 * Math.PI * c / K ), cy = ring * Math.sin( 2 * Math.PI * c / K );
		for ( let k = 0; k < sizes[ c ]; k ++, i ++ ) {

			anc[ i ] = c;
			const r = spread * Math.sqrt( ( k + 0.5 ) / sizes[ c ] ), a = k * 2.399963;
			positions[ 4 * i ] = cx + r * Math.cos( a );
			positions[ 4 * i + 1 ] = cy + r * Math.sin( a );
			colors.set( [ 0.2 + 0.1 * c, 0.5, 0.9 - 0.1 * c, 1 ], 4 * i );

		}

	}

	const clusterHeight = new Uint8Array( K + 1 ).fill( 1 );
	clusterHeight[ K ] = 2;
	const graph = {
		N, D: 1, C: K + 1, root: K, anc, levelOffset: Uint32Array.from( [ 0, K, K + 1 ] ), clusterHeight,
		clusterParent: new Int32Array( K + 1 ).fill( K ), clusterMass: Float32Array.from( [ ...sizes, N ] ), layoutRadius: ring + spread, initPos: positions,
	};
	const camera = new PerspectiveCamera( 45, 1.6, 0.5, 1e5 );
	const buffers = {
		clusterPos: instancedArray( K + 1, 'vec4' ),
		nodeColorTarget: { value: { array: colors, version: 1 } },
		uniforms: { selectedIndex: { value: - 1 }, layoutBlend: uniform( 0 ), graphCenter: uniform( new Vector3() ), sphereRadius: uniform( ring ) },
	};
	let computes = 0;
	const app = {
		scene: new Scene(), camera, width: 1280, height: 800, positions, positionsVersion: 1, options: {}, events: new EventTarget(),
		renderer: { compute: () => computes ++ },
	};
	const look = ( x, y, z, tx = 0, ty = 0 ) => {

		camera.position.set( x, y, z );
		camera.lookAt( new Vector3( tx, ty, 0 ) );
		camera.updateMatrixWorld();

	};

	look( 0, 0, 1100 );
	const field = new ClusterField( options );
	field.init( app );
	field.setGraph( buffers, graph, app );
	return { field, app, buffers, graph, look, computes: () => computes };

}

test( 'ClusterField: an overview shows the volumes; from inside the graph nothing is drawn or computed', () => {

	gfxState.tier = 'high';
	const { field, app, look, computes } = fakeField();
	field.update( 0.016, app );
	let s = field.stats;
	assert.equal( field.mesh.visible, true );
	assert.equal( s.quads, 6 );
	assert.ok( s.typical > 0 && s.typical < 0.12, `shapes: ${s.typical}` );
	assert.equal( s.overview, 1 );
	close( s.intensity, FIELD_RADIANCE * 0.5, 1e-9 );
	assert.equal( s.graph, 1 );
	assert.equal( field.uniforms.veil.value, 1, 'full absorption' );
	assert.equal( computes(), 1 );

	// the camera flies into the first community: its volume fills the view, the others lie beyond
	look( 300, 0, 60, 300, 0 );
	field.snap();
	field.update( 0.016, app );
	s = field.stats;
	assert.ok( s.typical > 0.5, `a haze: ${s.typical}` );
	assert.equal( s.overview, 0 );
	assert.equal( s.intensity, 0 );
	assert.equal( field.mesh.visible, false, 'not drawn' );
	assert.equal( s.quads, 0 );
	assert.equal( computes(), 1, 'the kernel does not run' );

	// back out: the light eases in (no snap), nothing pops
	look( 0, 0, 1100 );
	field.update( 0.1, app );
	const first = field.stats.overview;
	assert.ok( first > 0 && first < 0.5, `eases: ${first}` );
	assert.equal( field.uniforms.veil.value, first, 'the absorption comes back with the light' );
	for ( let k = 0; k < 60; k ++ ) field.update( 0.1, app );
	assert.equal( field.stats.overview, 1 );
	assert.equal( field.mesh.visible, true );
	assert.ok( computes() > 1 );

	// adaptive off: strength × tier only, also from inside
	look( 300, 0, 60, 300, 0 );
	field.adaptive = false;
	field.update( 0.016, app );
	assert.equal( field.mesh.visible, true );
	close( field.stats.intensity, FIELD_RADIANCE * 0.5, 1e-9 );
	assert.equal( field.uniforms.veil.value, 1 );
	assert.ok( field.stats.overdraw <= FIELD_TIERS.high.overdraw * 1.15 + 1e-6, 'the budget still holds' );
	field.dispose();

} );

test( 'ClusterField: graphs whose nodes are the cloud have no volumes; a selection keeps the volume of its community', () => {

	gfxState.tier = 'high';
	const sizes = new Array( 10 ).fill( 2000 );
	const { field, app, buffers, graph, computes } = fakeField( {}, { sizes, ring: 600, spread: 150 } );
	app.camera.position.set( 0, 0, 2600 );
	app.camera.updateMatrixWorld();
	assert.equal( graph.N, 20000 );
	field.update( 0.016, app );
	assert.equal( field.stats.graph, 0 );
	assert.equal( field.mesh.visible, false );
	assert.equal( field.stats.intensity, 0 );
	assert.equal( computes(), 0, 'nothing runs' );

	// a node of community 3 is selected: its volume, and only it, has light
	buffers.uniforms.selectedIndex.value = 3 * 2000 + 5;
	field.update( 0.016, app );
	assert.equal( field.mesh.visible, true );
	const light = Array.from( field._state.aux.value.array ).filter( ( _, o ) => o % 4 === 1 );
	for ( let j = 0; j < 10; j ++ ) {

		if ( field.clusters[ j ] === 3 ) close( light[ j ], 1.6 * FIELD_FOCUS_KEEP, 1e-6 );
		else assert.equal( light[ j ], 0 );

	}

	assert.equal( field.stats.quads, 1 );
	assert.equal( computes(), 1 );

	// the selection ends: the field stays for a moment while the light eases out, then it is off
	buffers.uniforms.selectedIndex.value = - 1;
	field.update( 0.3, app );
	assert.equal( field.mesh.visible, true, 'easing out' );
	assert.ok( Array.from( field._state.aux.value.array ).filter( ( _, o ) => o % 4 === 1 ).every( ( v ) => v === 0 ) );
	for ( let k = 0; k < 6; k ++ ) field.update( 0.3, app );
	assert.equal( field.mesh.visible, false );
	const ran = computes();
	field.update( 0.3, app );
	assert.equal( computes(), ran );

	// focus() does the same; adaptive off brings every volume back
	field.focus( 7 );
	field.update( 0.016, app );
	assert.equal( field.stats.quads, 1 );
	field.focus( null );
	field.adaptive = false;
	field.update( 0.016, app );
	assert.equal( field.stats.graph, 1 );
	assert.equal( field.stats.quads, 10 );
	field.dispose();

} );
