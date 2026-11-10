// Tests for src/graph/bundling.js: explicit control paths (hierarchical bundling made explicit,
// edge-path bundling in both variants), their invariants, the exact modes against literal
// re-implementations of the two papers' pseudocode, determinism, and a timing smoke test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph, rng } from '../src/graph/prepare.js';
import { evalEdgeCurveCPU } from '../src/render/EdgeRenderer.js';
import {
	EDGE_FREE, EDGE_BUNDLED, EDGE_CARRYING, BUNDLING_DEFAULTS,
	straightPaths, hierarchicalPaths, edgePathBundling, edgePathBundlingTask, mixedPaths,
	validatePaths, pathStats, pathPolygon, clusterCentroids,
} from '../src/graph/bundling.js';

const load = ( file, flat = false ) => {

	const raw = JSON.parse( readFileSync( `public/data/${file}.json`, 'utf8' ) );
	if ( flat ) delete raw.hierarchy;
	return prepareGraph( normalizeGraph( raw ) );

};

/** Minimal graph: the bundling functions only need N, E and edges. */
const makeGraph = ( N, pairs ) => ( { N, E: pairs.length, edges: Uint32Array.from( pairs.flat() ) } );
const positions = ( points ) => Float32Array.from( points.flatMap( ( p ) => [ p[ 0 ], p[ 1 ], p[ 2 ] ?? 0, 0 ] ) );
const dist = ( pos, a, b ) => Math.hypot( pos[ 4 * a ] - pos[ 4 * b ], pos[ 4 * a + 1 ] - pos[ 4 * b + 1 ], pos[ 4 * a + 2 ] - pos[ 4 * b + 2 ] );
const interior = ( paths, e ) => Array.from( paths.pathRefs.subarray( paths.pathStart[ e ], paths.pathStart[ e + 1 ] ) );
const METHODS = [ 'epb', 'spanner' ];

/** Random connected-ish geometric graph in 3D: k nearest neighbours plus a few long edges. */
function geometricGraph( n, seed, { nearest = 3, extra = 0.5 } = {} ) {

	const rand = rng( seed );
	const pts = [];
	for ( let i = 0; i < n; i ++ ) pts.push( [ rand() * 100, rand() * 100, rand() * 100 ] );
	const pos = positions( pts );
	const keys = new Set();
	const add = ( a, b ) => {

		if ( a !== b ) keys.add( Math.min( a, b ) * n + Math.max( a, b ) );

	};

	for ( let i = 0; i < n; i ++ ) {

		const order = Array.from( { length: n }, ( _, j ) => j ).filter( ( j ) => j !== i ).sort( ( a, b ) => dist( pos, i, a ) - dist( pos, i, b ) );
		for ( let k = 0; k < nearest; k ++ ) add( i, order[ k ] );

	}

	for ( let k = 0; k < extra * n; k ++ ) add( Math.floor( rand() * n ), Math.floor( rand() * n ) );
	const pairs = [ ...keys ].sort( ( a, b ) => a - b ).map( ( key ) => [ Math.floor( key / n ), key % n ] );
	return { g: makeGraph( n, pairs ), pos };

}

/** All the invariants of an edge-path bundling result. */
function checkBundling( g, pos, r, { maxDistortion = BUNDLING_DEFAULTS.maxDistortion, maxInterior = BUNDLING_DEFAULTS.maxInterior } = {} ) {

	const v = validatePaths( g, r, { realPaths: true, state: r.state, positions: pos, maxDistortion, maxInterior } );
	assert.deepEqual( v.errors, [] );
	assert.equal( r.state.length, g.E );
	assert.equal( r.load.length, g.E );
	let bundled = 0, carrying = 0;
	for ( let e = 0; e < g.E; e ++ ) {

		const n = r.pathStart[ e + 1 ] - r.pathStart[ e ];
		if ( n > 0 ) {

			bundled ++;
			assert.equal( r.state[ e ], EDGE_BUNDLED );
			assert.equal( r.load[ e ], 0, `bundled edge ${e} carries no other edge` );
			// from s to t, never through the end points, never the edge itself
			const poly = pathPolygon( g, r, e );
			assert.equal( poly[ 0 ], g.edges[ 2 * e ] );
			assert.equal( poly[ poly.length - 1 ], g.edges[ 2 * e + 1 ] );
			assert.equal( new Set( poly ).size, poly.length );

		} else if ( r.load[ e ] > 0 ) {

			carrying ++;
			assert.equal( r.state[ e ], EDGE_CARRYING );

		} else assert.equal( r.state[ e ], EDGE_FREE );

	}

	assert.equal( r.stats.bundled, bundled );
	assert.equal( r.stats.carrying, carrying );
	assert.equal( r.stats.free, g.E - bundled - carrying );
	assert.ok( r.stats.maxDistortion <= maxDistortion * ( 1 + 1e-9 ) );
	// a bundled edge adds one to the load of every edge of its path
	let loadSum = 0;
	for ( let e = 0; e < g.E; e ++ ) loadSum += r.load[ e ];
	assert.equal( loadSum, r.pathStart[ g.E ] + bundled );
	return r;

}

// ─── Literal re-implementations of the papers (no heap, no pruning) ──────────────────────────

/** Dijkstra by linear scans. weight( e ) per edge, `usable( e )` filters; returns edge ids s -> t or null. */
function slowPath( g, s, t, weight, usable ) {

	const { N, E, edges } = g;
	const d = new Float64Array( N ).fill( Infinity ), pred = new Int32Array( N ).fill( - 1 ), done = new Uint8Array( N );
	d[ s ] = 0;
	for ( ;; ) {

		let u = - 1;
		for ( let i = 0; i < N; i ++ ) if ( ! done[ i ] && d[ i ] < Infinity && ( u < 0 || d[ i ] < d[ u ] ) ) u = i;
		if ( u < 0 || u === t ) break;
		done[ u ] = 1;
		for ( let e = 0; e < E; e ++ ) {

			if ( ! usable( e ) ) continue;
			const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
			const v = a === u ? b : b === u ? a : - 1;
			if ( v < 0 || done[ v ] ) continue;
			if ( d[ u ] + weight( e ) < d[ v ] ) {

				d[ v ] = d[ u ] + weight( e );
				pred[ v ] = e;

			}

		}

	}

	if ( ! ( d[ t ] < Infinity ) ) return null;
	const path = [];
	for ( let v = t; v !== s; ) {

		const e = pred[ v ];
		path.unshift( e );
		v = edges[ 2 * e ] === v ? edges[ 2 * e + 1 ] : edges[ 2 * e ];

	}

	return path;

}

const pathInterior = ( g, s, path ) => {

	const nodes = [];
	let v = s;
	for ( const e of path.slice( 0, - 1 ) ) {

		v = g.edges[ 2 * e ] === v ? g.edges[ 2 * e + 1 ] : g.edges[ 2 * e ];
		nodes.push( v );

	}

	return nodes;

};

/** Algorithm 1 of Wallinger et al. 2022. Order: length descending, ties by id descending. */
function referenceEPB( g, pos, k = 2, d = 2 ) {

	const { E, edges } = g;
	const len = Array.from( { length: E }, ( _, e ) => dist( pos, edges[ 2 * e ], edges[ 2 * e + 1 ] ) );
	const order = Array.from( { length: E }, ( _, e ) => e ).sort( ( a, b ) => len[ b ] - len[ a ] || b - a );
	const lock = new Uint8Array( E ), skip = new Uint8Array( E );
	const paths = new Map();
	for ( const e of order ) {

		if ( lock[ e ] ) continue;
		skip[ e ] = 1;
		const s = edges[ 2 * e ], t = edges[ 2 * e + 1 ];
		const p = slowPath( g, s, t, ( m ) => len[ m ] ** d, ( m ) => ! skip[ m ] );
		if ( ! p || p.reduce( ( a, m ) => a + len[ m ], 0 ) > k * len[ e ] ) {

			skip[ e ] = 0;
			continue;

		}

		for ( const m of p ) lock[ m ] = 1;
		paths.set( e, pathInterior( g, s, p ) );

	}

	return paths;

}

/** Wallinger et al. 2023: greedy t-spanner (lengths ascending), then lightest paths in it. */
function referenceSpannerEPB( g, pos, t = 2, kappa = 2 ) {

	const { E, edges } = g;
	const len = Array.from( { length: E }, ( _, e ) => dist( pos, edges[ 2 * e ], edges[ 2 * e + 1 ] ) );
	const order = Array.from( { length: E }, ( _, e ) => e ).sort( ( a, b ) => len[ a ] - len[ b ] || a - b );
	const inH = new Uint8Array( E );
	for ( const e of order ) {

		const p = slowPath( g, edges[ 2 * e ], edges[ 2 * e + 1 ], ( m ) => len[ m ], ( m ) => inH[ m ] );
		if ( ! p || p.reduce( ( a, m ) => a + len[ m ], 0 ) > t * len[ e ] ) inH[ e ] = 1;

	}

	const paths = new Map();
	for ( let e = 0; e < E; e ++ ) {

		if ( inH[ e ] ) continue;
		const p = slowPath( g, edges[ 2 * e ], edges[ 2 * e + 1 ], ( m ) => len[ m ] ** kappa, ( m ) => inH[ m ] );
		if ( p.reduce( ( a, m ) => a + len[ m ], 0 ) > t * len[ e ] ) continue;
		paths.set( e, pathInterior( g, edges[ 2 * e ], p ) );

	}

	return { paths, spannerEdges: inH.reduce( ( a, b ) => a + b, 0 ) };

}

const pathsAsMap = ( g, r ) => {

	const m = new Map();
	for ( let e = 0; e < g.E; e ++ ) if ( r.pathStart[ e + 1 ] > r.pathStart[ e ] ) m.set( e, interior( r, e ) );
	return m;

};

// ─── Structure ───────────────────────────────────────────────────────────────────────────────

test( 'straightPaths: no interior points, valid for any graph', () => {

	const p = straightPaths( 5 );
	assert.deepEqual( Array.from( p.pathStart ), [ 0, 0, 0, 0, 0, 0 ] );
	assert.equal( p.pathRefs.length, 0 );
	assert.equal( p.maxInterior, 0 );
	const g = makeGraph( 4, [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 0 ], [ 0, 2 ] ] );
	assert.ok( validatePaths( g, p ).valid );
	assert.ok( validatePaths( g, p, { realPaths: true } ).valid );
	assert.deepEqual( Array.from( straightPaths( 0 ).pathStart ), [ 0 ] );

} );

test( 'validatePaths reports broken structures', () => {

	const g = makeGraph( 4, [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 0, 2 ] ] );
	const ok = { pathStart: Uint32Array.of( 0, 0, 0, 0, 1 ), pathRefs: Uint32Array.of( 1 ), maxInterior: 1 }; // 0-1-2 for edge (0,2)
	assert.deepEqual( validatePaths( g, ok, { realPaths: true } ), { valid: true, errors: [] } );
	const bad = ( paths, pattern, options ) => {

		const v = validatePaths( g, paths, options );
		assert.equal( v.valid, false );
		assert.match( v.errors.join( '\n' ), pattern );

	};

	bad( { pathStart: [ 0, 0, 0, 0, 1 ], pathRefs: Uint32Array.of( 1 ), maxInterior: 1 }, /Uint32Array/ );
	bad( { ...ok, pathStart: Uint32Array.of( 0, 0, 0, 1 ) }, /E \+ 1/ );
	bad( { ...ok, pathStart: Uint32Array.of( 1, 1, 1, 1, 1 ) }, /pathStart\[0\]/ );
	bad( { ...ok, pathStart: Uint32Array.of( 0, 1, 0, 0, 1 ) }, /decreases/ );
	bad( { ...ok, pathStart: Uint32Array.of( 0, 0, 0, 0, 2 ) }, /exceeds pathRefs/ );
	bad( { ...ok, maxInterior: 3 }, /maxInterior/ );
	bad( { ...ok, pathRefs: Uint32Array.of( 9 ) }, /out of range/ );
	bad( ok, /limit is 0/, { maxInterior: 0 } );
	// 3 is not adjacent to 0; and a path through an end point is not simple
	bad( { ...ok, pathRefs: Uint32Array.of( 3 ) }, /not adjacent/, { realPaths: true } );
	bad( { pathStart: Uint32Array.of( 0, 0, 0, 0, 2 ), pathRefs: Uint32Array.of( 1, 0 ), maxInterior: 2 }, /appears twice/, { realPaths: true } );
	// distortion: 0-1-2 with a far away node 1
	const pos = positions( [ [ 0, 0 ], [ 0, 50 ], [ 1, 0 ], [ 2, 0 ] ] );
	bad( ok, /exceeds 2 x/, { positions: pos, maxDistortion: 2 } );
	// state must agree with the paths
	bad( ok, /state/, { realPaths: true, state: Uint8Array.of( 0, 0, 0, 0 ) } );
	bad( ok, /expected 2/, { realPaths: true, state: Uint8Array.of( 0, 0, 0, 1 ) } );
	assert.ok( validatePaths( g, ok, { realPaths: true, state: Uint8Array.of( 2, 2, 0, 1 ) } ).valid );

} );

// ─── Hierarchical paths ──────────────────────────────────────────────────────────────────────

const HIERARCHICAL = [ [ 'imsv/karate', false ], [ 'imsv/lesmis', false ], [ 'imsv/football', false ], [ 'imsv/celegansneural', false ], [ 'imsv/lesmis', true ], [ 'generated/hsbm-500', false ], [ 'generated/lfr-2k', false ] ];

for ( const [ file, flat ] of HIERARCHICAL ) {

	test( `hierarchicalPaths ${file}${flat ? ' (no hierarchy)' : ''}: exactly the renderer's implicit polygon`, () => {

		const g = load( file, flat );
		const { N, E, D } = g;
		const paths = hierarchicalPaths( g );
		assert.deepEqual( validatePaths( g, paths ).errors, [] );
		assert.equal( paths.maxInterior, g.maxPath - 2 );
		// positions that encode the ref: node i at x = i, cluster c at x = N + c
		const nodePos = new Float32Array( 4 * N ), clusterPos = new Float32Array( 4 * g.C );
		for ( let i = 0; i < N; i ++ ) nodePos[ 4 * i ] = i;
		for ( let c = 0; c < g.C; c ++ ) clusterPos[ 4 * c ] = N + c;
		let total = 0;
		for ( let e = 0; e < E; e ++ ) {

			const s = g.edges[ 2 * e ], t = g.edges[ 2 * e + 1 ];
			// the L formula of prepare.js
			const lca = g.edgeLca[ e ];
			const L = 2 + 2 * lca + ( lca < D ? 1 : 0 );
			const refs = interior( paths, e );
			assert.equal( refs.length, L - 2, `interior count of edge ${e}` );
			total += L - 2;
			// the polygon written out from the definition
			const expected = [];
			for ( let l = 0; l < lca; l ++ ) expected.push( N + g.anc[ s * D + l ] );
			if ( lca < D ) expected.push( N + g.anc[ s * D + lca ] );
			for ( let l = lca - 1; l >= 0; l -- ) expected.push( N + g.anc[ t * D + l ] );
			assert.deepEqual( refs, expected, `refs of edge ${e}` );
			// the CPU mirror of the shader
			const curve = evalEdgeCurveCPU( g, e, 0.5, nodePos, clusterPos );
			assert.equal( curve.L, refs.length + 2 );
			assert.deepEqual( curve.controlPoints.map( ( p ) => p[ 0 ] ), pathPolygon( g, paths, e ), `control points of edge ${e}` );
			for ( const ref of refs ) assert.ok( ref >= N && ref < N + g.C - 1, 'a cluster, never the root' );

		}

		assert.equal( paths.pathRefs.length, total );
		assert.equal( paths.pathStart[ E ], total );

	} );

}

test( 'hierarchicalPaths: statistics with live centroids', () => {

	const g = load( 'imsv/football' );
	const paths = hierarchicalPaths( g );
	const s = pathStats( g, paths, g.initPos );
	assert.equal( s.edges, g.E );
	assert.equal( s.bundled, g.E ); // D >= 1: every edge passes at least one centroid
	assert.equal( s.maxInterior, paths.maxInterior );
	assert.ok( s.lengthRatio >= 1 && s.meanDistortion >= 1 && s.maxDistortion >= s.meanDistortion );
	// explicit centroids give the same numbers
	const s2 = pathStats( g, paths, g.initPos, { clusterPositions: clusterCentroids( g, g.initPos ) } );
	assert.deepEqual( s2, s );
	// straight paths: polygon = edge
	const flat = pathStats( g, straightPaths( g.E ), g.initPos );
	assert.equal( flat.bundled, 0 );
	assert.ok( Math.abs( flat.lengthRatio - 1 ) < 1e-12 && Math.abs( flat.inkRatio - 1 ) < 1e-12 );

} );

// ─── Edge-path bundling: small graphs with known answers ─────────────────────────────────────

for ( const method of METHODS ) {

	for ( const exact of [ false, true ] ) {

		const tag = `edgePathBundling ${method}${exact ? ' exact' : ''}`;
		const run = ( g, pos, o = {} ) => checkBundling( g, pos, edgePathBundling( g, pos, { method, exact, ...o } ), o );

		test( `${tag}: the long edge of a triangle runs over the two short ones`, () => {

			const g = makeGraph( 3, [ [ 0, 1 ], [ 0, 2 ], [ 2, 1 ] ] );
			const pos = positions( [ [ 0, 0, 0 ], [ 10, 0, 0 ], [ 5, 1, 2 ] ] );
			const r = run( g, pos );
			assert.deepEqual( interior( r, 0 ), [ 2 ] );
			assert.deepEqual( Array.from( r.state ), [ EDGE_BUNDLED, EDGE_CARRYING, EDGE_CARRYING ] );
			assert.deepEqual( Array.from( r.load ), [ 0, 1, 1 ] );
			assert.equal( r.maxInterior, 1 );
			assert.equal( r.stats.bundled, 1 );
			const short = dist( pos, 0, 2 ) + dist( pos, 2, 1 );
			assert.ok( Math.abs( r.stats.meanDistortion - short / 10 ) < 1e-12 );
			assert.ok( Math.abs( r.stats.inkRatio - short / ( short + 10 ) ) < 1e-12 );
			// the path is stored from the source to the target whatever the edge's direction
			const g2 = makeGraph( 4, [ [ 3, 0 ], [ 1, 3 ], [ 1, 2 ], [ 0, 2 ] ] ); // 3 … 0 over 1? no: over 2 and 1
			const pos2 = positions( [ [ 30, 0 ], [ 10, 2 ], [ 20, - 2 ], [ 0, 0 ] ] );
			const r2 = run( g2, pos2 );
			assert.deepEqual( interior( r2, 0 ), [ 1, 2 ] );
			// pathStats agrees with the task's own numbers
			const s = pathStats( g, r, pos );
			assert.equal( s.bundled, 1 );
			assert.ok( Math.abs( s.inkRatio - r.stats.inkRatio ) < 1e-6 );
			assert.ok( Math.abs( s.meanDistortion - r.stats.meanDistortion ) < 1e-6 );

		} );

		test( `${tag}: the distortion bound decides`, () => {

			// unit square: the way round is 3 edges for 1, too long for the default bound of 2
			const g = makeGraph( 4, [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 0 ] ] );
			const pos = positions( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1.01 ], [ 0, 1.02 ] ] );
			assert.equal( run( g, pos ).stats.bundled, 0 );
			const r = run( g, pos, { maxDistortion: 3.5 } );
			assert.equal( r.stats.bundled, 1 );
			// 'epb' bundles the longest edge first, the spanner is built without the longest edge:
			// the same one, (3,0), over 2 and 1
			assert.deepEqual( interior( r, 3 ), [ 2, 1 ] );
			// a hop limit of one interior point forbids that path again
			assert.equal( run( g, pos, { maxDistortion: 3.5, maxInterior: 1 } ).stats.bundled, 0 );

		} );

		test( `${tag}: a tree has no bundling`, () => {

			const rand = rng( 7 );
			const n = 60, pairs = [], pts = [];
			for ( let i = 0; i < n; i ++ ) {

				pts.push( [ rand() * 50, rand() * 50, rand() * 50 ] );
				if ( i > 0 ) pairs.push( [ Math.floor( rand() * i ), i ] );

			}

			const g = makeGraph( n, pairs ), pos = positions( pts );
			const r = run( g, pos );
			assert.equal( r.stats.bundled, 0 );
			assert.equal( r.pathRefs.length, 0 );
			assert.equal( r.maxInterior, 0 );
			assert.ok( r.state.every( ( s ) => s === EDGE_FREE ) );
			assert.equal( r.stats.inkRatio, 1 );
			assert.ok( r.stats.complete );

		} );

		test( `${tag}: invariants on random geometric graphs and real data`, () => {

			for ( const seed of [ 1, 2, 3 ] ) {

				const { g, pos } = geometricGraph( 40 + 20 * seed, seed );
				const r = run( g, pos );
				assert.ok( r.stats.bundled > 0, 'something is bundled' );
				for ( const o of [ { maxInterior: 2 }, { maxDistortion: 1.3 }, { exponent: 1 }, { exponent: 3, maxInterior: Infinity }, { exponent: 0.5 } ] ) run( g, pos, o );

			}

			for ( const file of [ 'imsv/karate', 'imsv/lesmis' ] ) {

				const g = load( file );
				const r = run( g, g.initPos );
				assert.ok( r.stats.bundled > 0.2 * g.E );

			}

		} );

	}

}

// ─── Edge-path bundling: the exact modes are the papers' algorithms ──────────────────────────

test( "edgePathBundling epb exact = Algorithm 1 of the paper, re-implemented literally", () => {

	const cases = [ 1, 2, 3, 4 ].map( ( seed ) => geometricGraph( 30 + 15 * seed, 10 + seed, { nearest: 2 + ( seed % 2 ), extra: 0.4 + 0.2 * seed } ) );
	const lesmis = load( 'imsv/lesmis' );
	cases.push( { g: lesmis, pos: lesmis.initPos } );
	for ( const { g, pos } of cases ) {

		for ( const [ k, d ] of [ [ 2, 2 ], [ 1.5, 2 ], [ 3, 1 ], [ 2, 3 ] ] ) {

			const r = edgePathBundling( g, pos, { method: 'epb', exact: true, maxDistortion: k, exponent: d, maxInterior: Infinity } );
			checkBundling( g, pos, r, { maxDistortion: k, maxInterior: Infinity } );
			assert.deepEqual( pathsAsMap( g, r ), referenceEPB( g, pos, k, d ), `N ${g.N} E ${g.E} k ${k} d ${d}` );

		}

	}

} );

test( "edgePathBundling spanner exact = the greedy spanner + lightest paths, re-implemented literally", () => {

	const cases = [ 1, 2, 3, 4 ].map( ( seed ) => geometricGraph( 30 + 15 * seed, 20 + seed, { nearest: 2 + ( seed % 2 ), extra: 0.4 + 0.2 * seed } ) );
	const lesmis = load( 'imsv/lesmis' );
	cases.push( { g: lesmis, pos: lesmis.initPos } );
	for ( const { g, pos } of cases ) {

		for ( const [ t, kappa ] of [ [ 2, 2 ], [ 1.5, 2 ], [ 3, 1 ], [ 2, 3 ] ] ) {

			const r = edgePathBundling( g, pos, { method: 'spanner', exact: true, maxDistortion: t, exponent: kappa, maxInterior: Infinity } );
			checkBundling( g, pos, r, { maxDistortion: t, maxInterior: Infinity } );
			const ref = referenceSpannerEPB( g, pos, t, kappa );
			assert.equal( r.stats.spannerEdges, ref.spannerEdges, 'spanner size' );
			assert.deepEqual( pathsAsMap( g, r ), ref.paths, `N ${g.N} E ${g.E} t ${t} kappa ${kappa}` );

		}

	}

} );

test( 'edgePathBundling: the default (pruned, bounded) searches stay close to the exact ones', () => {

	// not an identity: the pruned search bundles some edges the paper's rule leaves straight and
	// misses a few; with the hop limit lifted and effort unbounded the difference is small
	for ( const file of [ 'imsv/lesmis', 'imsv/football', 'generated/hsbm-500' ] ) {

		const g = load( file );
		for ( const method of METHODS ) {

			const exact = edgePathBundling( g, g.initPos, { method, exact: true, maxInterior: Infinity } );
			const fast = edgePathBundling( g, g.initPos, { method, maxInterior: Infinity, effort: Infinity } );
			const dflt = edgePathBundling( g, g.initPos, { method } );
			checkBundling( g, g.initPos, fast, { maxInterior: Infinity } );
			checkBundling( g, g.initPos, dflt );
			assert.ok( fast.stats.bundled >= 0.9 * exact.stats.bundled, `${file} ${method}: ${fast.stats.bundled} vs exact ${exact.stats.bundled}` );
			assert.ok( dflt.stats.bundled >= 0.85 * exact.stats.bundled, `${file} ${method}: ${dflt.stats.bundled} vs exact ${exact.stats.bundled}` );

		}

	}

} );

// ─── Determinism, tasks, budgets, odd input ──────────────────────────────────────────────────

test( 'edgePathBundling is deterministic and independent of how the task is chunked', () => {

	const g = load( 'generated/hsbm-500' );
	for ( const method of METHODS ) {

		const a = edgePathBundling( g, g.initPos, { method } );
		const b = edgePathBundling( g, g.initPos, { method } );
		const same = ( x, y ) => {

			for ( const key of [ 'pathStart', 'pathRefs', 'state', 'load' ] ) assert.deepEqual( x[ key ], y[ key ], key );
			assert.equal( x.maxInterior, y.maxInterior );
			const { timeMs: t1, ...s1 } = x.stats, { timeMs: t2, ...s2 } = y.stats;
			assert.deepEqual( s1, s2 );

		};

		same( a, b );
		for ( const chunk of [ 1, 7, 1000 ] ) {

			const task = edgePathBundlingTask( g, g.initPos, { method } );
			assert.equal( task.completed, 0 );
			let steps = 0, last = 0;
			while ( ! task.step( chunk ) ) {

				assert.ok( task.completed > last && task.completed <= task.total );
				last = task.completed;
				steps ++;

			}

			assert.ok( task.done && task.complete );
			assert.equal( task.completed, task.total );
			assert.ok( steps >= Math.floor( ( task.total - 1 ) / chunk ) );
			same( a, task.result() );
			assert.equal( task.step( 5 ), true ); // a finished task stays finished
			same( a, task.result() );

		}

		// the positions are read when the task is created
		const pos = Float32Array.from( g.initPos );
		const task = edgePathBundlingTask( g, pos, { method } );
		pos.fill( 0 );
		task.step( Infinity );
		same( a, task.result() );

	}

} );

test( 'edgePathBundlingTask: partial results are valid, phases and totals', () => {

	const g = load( 'generated/hsbm-500' );
	const epb = edgePathBundlingTask( g, g.initPos, { method: 'epb' } );
	assert.equal( epb.phase, 'bundle' );
	assert.equal( epb.total, g.E );
	let bundled = - 1;
	while ( ! epb.done ) {

		epb.step( 300 );
		const r = checkBundling( g, g.initPos, epb.result() );
		assert.ok( r.stats.bundled >= bundled, 'bundled edges only grow' );
		bundled = r.stats.bundled;
		assert.equal( r.stats.complete, epb.done );

	}

	const sp = edgePathBundlingTask( g, g.initPos, { method: 'spanner' } );
	assert.equal( sp.phase, 'spanner' );
	assert.equal( sp.total, 2 * g.E );
	const phases = [];
	while ( ! sp.done ) {

		sp.step( 500 );
		if ( phases[ phases.length - 1 ] !== sp.phase ) phases.push( sp.phase );
		checkBundling( g, g.initPos, sp.result() );

	}

	assert.deepEqual( phases, [ 'spanner', 'route' ] );
	// after the spanner phase alone every edge outside the spanner already has a path
	const half = edgePathBundlingTask( g, g.initPos, { method: 'spanner' } );
	half.step( g.E );
	assert.equal( half.phase, 'route' );
	const h = checkBundling( g, g.initPos, half.result() ), full = sp.result();
	assert.equal( h.stats.complete, false );
	assert.equal( h.stats.bundled, full.stats.bundled );
	assert.equal( h.stats.spannerEdges, full.stats.spannerEdges );
	assert.equal( h.stats.bundled + h.stats.spannerEdges, g.E );
	// in exact mode a path only counts once the route phase has confirmed it
	const ex = edgePathBundlingTask( g, g.initPos, { method: 'spanner', exact: true } );
	ex.step( g.E );
	assert.equal( ex.result().stats.bundled, 0 );
	ex.step( Infinity );
	assert.ok( ex.result().stats.bundled > 0 );

} );

test( 'edgePathBundling: time and work budgets cut the run, the result stays valid', () => {

	const g = load( 'generated/lfr-2k' );
	for ( const method of METHODS ) {

		const full = edgePathBundling( g, g.initPos, { method } );
		assert.ok( full.stats.complete );
		const none = checkBundling( g, g.initPos, edgePathBundling( g, g.initPos, { method, timeBudget: 0 } ) );
		assert.equal( none.stats.bundled, 0 );
		assert.equal( none.stats.complete, false );
		// a work budget is deterministic
		const work = Math.floor( full.stats.relaxations / 4 );
		const a = checkBundling( g, g.initPos, edgePathBundling( g, g.initPos, { method, maxWork: work } ) );
		const b = edgePathBundling( g, g.initPos, { method, maxWork: work } );
		assert.equal( a.stats.complete, false );
		assert.ok( a.stats.bundled > 0 && a.stats.bundled < full.stats.bundled );
		assert.ok( a.stats.relaxations >= work && a.stats.relaxations < 1.5 * work );
		assert.deepEqual( a.pathRefs, b.pathRefs );
		if ( method === 'epb' ) {

			// longest edges first: what was bundled is longer than what was not reached
			const len = ( e ) => dist( g.initPos, g.edges[ 2 * e ], g.edges[ 2 * e + 1 ] );
			let shortestBundled = Infinity;
			for ( let e = 0; e < g.E; e ++ ) if ( a.state[ e ] === EDGE_BUNDLED ) shortestBundled = Math.min( shortestBundled, len( e ) );
			const lengths = Array.from( { length: g.E }, ( _, e ) => len( e ) ).sort( ( x, y ) => x - y );
			assert.ok( shortestBundled > lengths[ Math.floor( g.E / 4 ) ], 'the cut left the short edges' );

		}

	}

} );

test( 'edgePathBundlingTask.hurry: quick spanner tests when the budget is short', () => {

	const g = load( 'generated/lfr-2k' );
	const full = edgePathBundling( g, g.initPos, { method: 'spanner' } );
	assert.equal( full.stats.hurried, false );
	// by hand
	const task = edgePathBundlingTask( g, g.initPos, { method: 'spanner' } );
	task.hurry();
	task.step( Infinity );
	const quick = checkBundling( g, g.initPos, task.result() );
	assert.ok( quick.stats.hurried && quick.stats.complete );
	assert.ok( quick.stats.spannerEdges > full.stats.spannerEdges, 'quick tests leave more edges in the spanner' );
	assert.ok( quick.stats.relaxations < full.stats.relaxations );
	assert.ok( quick.stats.bundled > 0.5 * full.stats.bundled );
	// by a work budget a little too small for the spanner phase: the phase switches to quick
	// tests in time and still finishes, deterministically
	const phase = edgePathBundlingTask( g, g.initPos, { method: 'spanner' } );
	phase.step( g.E );
	assert.equal( phase.phase, 'route' );
	const work = Math.floor( 0.8 * phase.result().stats.relaxations );
	const a = checkBundling( g, g.initPos, edgePathBundling( g, g.initPos, { method: 'spanner', maxWork: work } ) );
	const b = edgePathBundling( g, g.initPos, { method: 'spanner', maxWork: work } );
	assert.ok( a.stats.hurried );
	assert.equal( a.stats.complete, false );
	assert.equal( a.stats.phase, 'route', 'the spanner phase finished within the budget' );
	assert.ok( a.stats.bundled > 0.8 * full.stats.bundled );
	assert.deepEqual( a.pathRefs, b.pathRefs );
	assert.deepEqual( a.state, b.state );
	// far too small a budget: still a valid result
	checkBundling( g, g.initPos, edgePathBundling( g, g.initPos, { method: 'spanner', maxWork: Math.floor( work / 10 ) } ) );
	// the exact mode and 'epb' ignore it
	for ( const o of [ { method: 'spanner', exact: true }, { method: 'epb' } ] ) {

		const karate = load( 'imsv/karate' );
		const plain = edgePathBundling( karate, karate.initPos, o );
		const t = edgePathBundlingTask( karate, karate.initPos, o );
		t.hurry();
		t.step( Infinity );
		assert.deepEqual( t.result().pathRefs, plain.pathRefs );

	}

} );

test( 'edgePathBundling: empty graphs, odd positions, bad options', () => {

	for ( const method of METHODS ) {

		const empty = edgePathBundling( { N: 0, E: 0, edges: new Uint32Array( 0 ) }, new Float32Array( 0 ), { method } );
		assert.deepEqual( Array.from( empty.pathStart ), [ 0 ] );
		assert.equal( empty.stats.bundled, 0 );
		assert.ok( empty.stats.complete );
		const lonely = edgePathBundling( makeGraph( 3, [] ), positions( [ [ 0, 0 ], [ 1, 0 ], [ 2, 0 ] ] ), { method } );
		assert.equal( lonely.pathStart.length, 1 );

		// a node without a position: its edges stay straight and carry nothing, the rest bundles
		const g = makeGraph( 4, [ [ 0, 1 ], [ 0, 2 ], [ 2, 1 ], [ 0, 3 ], [ 3, 1 ] ] );
		const pos = positions( [ [ 0, 0 ], [ 10, 0 ], [ 5, 1 ], [ NaN, 0 ] ] );
		const r = edgePathBundling( g, pos, { method } );
		assert.deepEqual( interior( r, 0 ), [ 2 ] );
		assert.deepEqual( Array.from( r.state ), [ EDGE_BUNDLED, EDGE_CARRYING, EDGE_CARRYING, EDGE_FREE, EDGE_FREE ] );
		assert.ok( Number.isFinite( r.stats.inkRatio ) && Number.isFinite( r.stats.meanDistortion ) );

		// coincident nodes (zero-length edges) and a stride of 3
		const g0 = makeGraph( 4, [ [ 0, 1 ], [ 1, 2 ], [ 0, 2 ], [ 2, 3 ], [ 0, 3 ] ] );
		const p3 = Float64Array.of( 0, 0, 0, 0, 0, 0, 4, 0, 0, 8, 1, 0 );
		const r0 = edgePathBundling( g0, p3, { method, stride: 3 } );
		assert.deepEqual( validatePaths( g0, r0, { realPaths: true, state: r0.state, positions: p3, stride: 3, maxDistortion: 2 } ).errors, [] );
		// (0,3) runs over node 2; going through the coincident node 1 first costs nothing
		assert.equal( r0.state[ 4 ], EDGE_BUNDLED );
		assert.equal( interior( r0, 4 ).pop(), 2 );
		assert.equal( r0.state[ 0 ] === EDGE_BUNDLED, false, 'a zero-length edge is never bundled' );

	}

	const g = makeGraph( 3, [ [ 0, 1 ], [ 0, 2 ], [ 2, 1 ] ] ), pos = positions( [ [ 0, 0 ], [ 10, 0 ], [ 5, 1 ] ] );
	assert.throws( () => edgePathBundling( g, pos, { method: 'nope' } ), /unknown method/ );
	assert.throws( () => edgePathBundling( g, pos, { maxDistortion: 0.5 } ), /maxDistortion/ );
	assert.throws( () => edgePathBundling( g, pos, { exponent: 0 } ), /exponent/ );
	// undefined options fall back to the defaults
	assert.equal( edgePathBundling( g, pos, { method: undefined, maxInterior: undefined } ).stats.method, BUNDLING_DEFAULTS.method );

} );

// ─── Mixed mode ──────────────────────────────────────────────────────────────────────────────

test( 'mixedPaths: edge-path bundles, hierarchical paths for the edges they left free', () => {

	const g = load( 'imsv/football' );
	const epb = edgePathBundling( g, g.initPos, { method: 'epb' } );
	const heb = hierarchicalPaths( g );
	const mix = mixedPaths( g, epb, heb );
	assert.deepEqual( validatePaths( g, mix ).errors, [] );
	let fromEpb = 0, fromHeb = 0, straight = 0;
	for ( let e = 0; e < g.E; e ++ ) {

		if ( epb.state[ e ] === EDGE_BUNDLED ) {

			assert.equal( mix.source[ e ], 1 );
			assert.deepEqual( interior( mix, e ), interior( epb, e ) );
			fromEpb ++;

		} else if ( epb.state[ e ] === EDGE_CARRYING ) {

			assert.equal( mix.source[ e ], 0 );
			assert.deepEqual( interior( mix, e ), [] );
			straight ++;

		} else {

			assert.equal( mix.source[ e ], 2 );
			assert.deepEqual( interior( mix, e ), interior( heb, e ) );
			fromHeb ++;

		}

	}

	assert.ok( fromEpb > 0 && fromHeb > 0 && straight > 0 );
	assert.equal( mix.maxInterior, Math.max( epb.maxInterior, heb.maxInterior ) );
	// 'straight': the carrying edges take the fallback too
	const all = mixedPaths( g, epb, heb, { only: 'straight' } );
	assert.deepEqual( validatePaths( g, all ).errors, [] );
	for ( let e = 0; e < g.E; e ++ ) assert.equal( all.source[ e ], epb.state[ e ] === EDGE_BUNDLED ? 1 : 2 );
	// statistics work on mixed refs (nodes and clusters)
	const s = pathStats( g, mix, g.initPos );
	assert.equal( s.bundled, fromEpb + fromHeb );
	assert.ok( s.inkRatio > 0 && s.inkRatio < 2 );

} );

// ─── Timing smoke test ───────────────────────────────────────────────────────────────────────

test( 'edgePathBundling: timing smoke test (5,000 nodes / 39,817 edges, hierarchical seed positions)', ( t ) => {

	// The seed positions (graph.initPos) have the structure of a settled layout (compact
	// clusters, long edges between them) and give timings within about 20% of a relaxed one.
	const g = load( 'generated/hsbm-5k' );
	const lines = [];
	for ( const method of METHODS ) {

		let best = null;
		for ( let i = 0; i < 2; i ++ ) {

			const r = edgePathBundling( g, g.initPos, { method } );
			if ( ! best || r.stats.timeMs < best.stats.timeMs ) best = r;

		}

		const s = best.stats;
		lines.push( `${method.padEnd( 7 )} ${s.timeMs.toFixed( 0 ).padStart( 5 )} ms  bundled ${( 100 * s.bundledRatio ).toFixed( 1 )}%  ink ${s.inkRatio.toFixed( 3 )}  distortion ${s.meanDistortion.toFixed( 2 )}  interior ${s.meanInterior.toFixed( 2 )} (max ${s.maxInterior})  ${( s.relaxations / 1e6 ).toFixed( 1 )}M relaxations` );
		assert.ok( s.complete );
		assert.ok( s.bundledRatio > 0.4, 'a community graph bundles well' );
		assert.ok( s.timeMs < 20000, `far too slow: ${s.timeMs} ms` );
		const v = validatePaths( g, best, { realPaths: true, state: best.state, positions: g.initPos, maxDistortion: 2, maxInterior: 8 } );
		assert.deepEqual( v.errors, [] );

	}

	// a budget of 100 ms returns in about that time with a valid partial result
	const cut = edgePathBundling( g, g.initPos, { method: 'epb', timeBudget: 100 } );
	lines.push( `epb, timeBudget 100 ms: ${cut.stats.timeMs.toFixed( 0 )} ms, bundled ${( 100 * cut.stats.bundledRatio ).toFixed( 1 )}%, complete ${cut.stats.complete}` );
	assert.ok( cut.stats.timeMs < 1500 );
	const t0 = performance.now();
	const heb = hierarchicalPaths( g );
	lines.push( `hierarchicalPaths ${( performance.now() - t0 ).toFixed( 1 )} ms, ${heb.pathRefs.length} refs, max interior ${heb.maxInterior}` );
	for ( const line of lines ) t.diagnostic( line );
	console.log( lines.map( ( l ) => '    ' + l ).join( '\n' ) );

} );
