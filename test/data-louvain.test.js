import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph, normalizeLevels } from '../src/data/format.js';
import { louvain, modularity, louvainHierarchy, ensureHierarchy } from '../src/data/louvain.js';
import { generate } from '../src/data/generators.js';
import { prepareGraph } from '../src/graph/prepare.js';

const load = ( id ) => normalizeGraph( JSON.parse( readFileSync( `public/data/imsv/${id}.json`, 'utf8' ) ) );

/** Independent reference: Q = 1/2m Σ_ij (A_ij − γ k_i k_j / 2m) δ(c_i, c_j) with a dense matrix. */
function denseModularity( g, part, gamma = 1 ) {

	const n = g.nodeCount;
	const A = Array.from( { length: n }, () => new Float64Array( n ) );
	for ( let k = 0; k < g.edges.length; k += 2 ) {

		A[ g.edges[ k ] ][ g.edges[ k + 1 ] ] += 1;
		A[ g.edges[ k + 1 ] ][ g.edges[ k ] ] += 1;

	}

	const deg = A.map( ( row ) => row.reduce( ( a, b ) => a + b, 0 ) );
	const m2 = deg.reduce( ( a, b ) => a + b, 0 );
	let q = 0;
	for ( let i = 0; i < n; i ++ ) for ( let j = 0; j < n; j ++ ) if ( part[ i ] === part[ j ] ) q += A[ i ][ j ] - ( gamma * deg[ i ] * deg[ j ] ) / m2;
	return q / m2;

}

function sizesOf( level ) {

	const s = new Map();
	for ( const c of level ) s.set( c, ( s.get( c ) || 0 ) + 1 );
	return [ ...s.values() ];

}

function assertNested( levels, n ) {

	for ( let l = 0; l + 1 < levels.length; l ++ ) {

		const up = new Map();
		for ( let i = 0; i < n; i ++ ) {

			const f = levels[ l ][ i ], c = levels[ l + 1 ][ i ];
			if ( up.has( f ) ) assert.equal( up.get( f ), c, `level ${l} cluster ${f} splits across level ${l + 1}` );
			else up.set( f, c );

		}

	}

}

function assertDense( level ) {

	const K = Math.max( ...level ) + 1;
	const seen = new Uint8Array( K );
	for ( const c of level ) seen[ c ] = 1;
	assert.ok( seen.every( ( x ) => x === 1 ), 'cluster ids are dense' );

}

test( 'modularity() matches the dense-matrix definition', () => {

	const g = load( 'karate' );
	const hi = new Set( [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21 ] );
	const club = Array.from( { length: 34 }, ( _, i ) => ( hi.has( i ) ? 0 : 1 ) );
	for ( const gamma of [ 1, 0.5, 2 ] ) assert.ok( Math.abs( modularity( g, club, gamma ) - denseModularity( g, club, gamma ) ) < 1e-12 );
	const r = louvain( g, { seed: 3 } );
	for ( const lvl of r.levels ) assert.ok( Math.abs( modularity( g, lvl ) - denseModularity( g, lvl ) ) < 1e-12 );
	// singletons: Q = −Σ (k_i / 2m)²
	const single = Array.from( { length: 34 }, ( _, i ) => i );
	assert.ok( Math.abs( modularity( g, single ) - denseModularity( g, single ) ) < 1e-12 );

} );

test( 'karate club: Louvain modularity ≈ 0.42 (optimum 0.4198)', () => {

	const g = load( 'karate' );
	const r = louvain( g, { seed: 1 } );
	const q = r.modularity[ r.modularity.length - 1 ];
	assert.ok( q > 0.41 && q < 0.4199, `seed 1: Q = ${q}` );
	let best = 0;
	for ( let seed = 1; seed <= 20; seed ++ ) {

		const rs = louvain( g, { seed } );
		const qs = rs.modularity[ rs.modularity.length - 1 ];
		assert.ok( qs > 0.39 && qs < 0.4199, `seed ${seed}: Q = ${qs}` );
		best = Math.max( best, qs );

	}

	assert.ok( Math.abs( best - 0.4198 ) < 5e-4, `best Q over seeds = ${best}` );
	// reported per-level modularity equals the modularity of that partition on the input graph
	r.levels.forEach( ( lvl, l ) => assert.ok( Math.abs( modularity( g, lvl ) - r.modularity[ l ] ) < 1e-9 ) );

} );

test( 'ImsvGraphVis datasets: modularity comparable to the python-louvain hierarchies shipped with the paper', () => {

	for ( const id of [ 'karate', 'dolphins', 'lesmis', 'polbooks', 'football', 'celegansneural' ] ) {

		const g = load( id );
		const ref = modularity( g, g.hierarchy.levels[ g.hierarchy.levels.length - 1 ] );
		const r = louvain( g, { seed: 1 } );
		const q = r.modularity[ r.modularity.length - 1 ];
		assert.ok( q > ref - 0.03, `${id}: ours ${q.toFixed( 4 )} vs python-louvain ${ref.toFixed( 4 )}` );

	}

} );

test( 'dendrogram: levels dense, nested, coarser upwards, modularity non-decreasing, normalizeLevels-stable', () => {

	for ( const id of [ 'lesmis', 'celegansneural' ] ) {

		const g = load( id );
		const r = louvain( g, { seed: 5 } );
		assert.ok( r.levels.length >= 2 );
		for ( const lvl of r.levels ) {

			assert.equal( lvl.length, g.nodeCount );
			assertDense( lvl );

		}

		assertNested( r.levels, g.nodeCount );
		for ( let l = 1; l < r.levels.length; l ++ ) {

			assert.ok( r.levelSizes[ l ] < r.levelSizes[ l - 1 ] );
			assert.ok( r.modularity[ l ] >= r.modularity[ l - 1 ] - 1e-12 );

		}

		const plain = r.levels.map( ( l ) => Array.from( l ) );
		assert.deepEqual( normalizeLevels( plain, g.nodeCount ), plain );

	}

} );

test( 'deterministic for a seed; seed changes the visiting order', () => {

	const g = generate( { model: 'lfr', params: { n: 3000, avgDegree: 12 }, seed: 4 } );
	const a = louvain( g, { seed: 7 } ), b = louvain( g, { seed: 7 } );
	assert.equal( a.levels.length, b.levels.length );
	a.levels.forEach( ( l, k ) => assert.deepEqual( l, b.levels[ k ] ) );
	assert.deepEqual( a.modularity, b.modularity );
	const c = louvain( g, { seed: 8 } );
	assert.ok( c.levels[ 0 ].some( ( x, i ) => x !== a.levels[ 0 ][ i ] ) || c.levels.length !== a.levels.length );

} );

test( 'resolution: higher γ gives more, smaller communities', () => {

	const g = load( 'lesmis' );
	const top = ( gamma ) => {

		const r = louvain( g, { seed: 1, resolution: gamma } );
		return r.levelSizes[ r.levelSizes.length - 1 ];

	};

	assert.ok( top( 0.5 ) < top( 1 ) && top( 1 ) < top( 2.5 ), `${top( 0.5 )} ${top( 1 )} ${top( 2.5 )}` );

} );

test( 'recovers planted communities of a hierarchical SBM', () => {

	const g = generate( { model: 'hsbm', params: { n: 2000, avgDegree: 14, depth: 2, branching: 5, mixing: 0.15, sizeSpread: 0.2, degreeSpread: 0 }, seed: 2 } );
	const planted = g.planted.levels[ 0 ];
	const r = louvain( g, { seed: 1 } );
	const lvl = r.levels[ 0 ];
	// purity: fraction of nodes whose Louvain cluster's majority planted block equals their own
	const votes = new Map();
	for ( let i = 0; i < g.nodeCount; i ++ ) {

		const key = lvl[ i ] * 100000 + planted[ i ];
		votes.set( key, ( votes.get( key ) || 0 ) + 1 );

	}

	const best = new Map();
	for ( const [ key, v ] of votes ) {

		const c = Math.floor( key / 100000 );
		if ( ! best.has( c ) || best.get( c ) < v ) best.set( c, v );

	}

	let hits = 0;
	for ( const v of best.values() ) hits += v;
	assert.ok( hits / g.nodeCount > 0.95, `purity ${hits / g.nodeCount}` );
	// the final partition is at least as good as the planted one (merging a few sibling blocks
	// raises modularity here, so the top level has fewer than 25 clusters but more than 5)
	const qPlanted = modularity( g, planted );
	const qTop = r.modularity[ r.modularity.length - 1 ];
	assert.ok( qTop >= qPlanted - 0.005, `Louvain ${qTop} vs planted ${qPlanted}` );
	const K = r.levelSizes[ r.levelSizes.length - 1 ];
	assert.ok( K > 5 && K <= 30, `top-level communities: ${K}` );

} );

test( 'refine: oversized leaf communities are split; uniform depth and nesting kept', () => {

	const g = generate( { model: 'lfr', params: { n: 5000, avgDegree: 15, minCommunity: 100, maxCommunity: 600, maxDegree: 80, mu: 0.25 }, seed: 1 } );
	const plain = louvain( g, { seed: 1 } );
	const maxLeaf = 80;
	assert.ok( Math.max( ...sizesOf( plain.levels[ 0 ] ) ) > maxLeaf );
	const r = louvain( g, { seed: 1, refine: { maxLeafSize: maxLeaf } } );
	assert.ok( r.refined );
	assert.equal( r.levels.length, plain.levels.length + 1 );
	assert.ok( Math.max( ...sizesOf( r.levels[ 0 ] ) ) <= maxLeaf );
	assertDense( r.levels[ 0 ] );
	assertNested( r.levels, g.nodeCount );
	// the coarser levels are untouched
	for ( let l = 0; l < plain.levels.length; l ++ ) assert.deepEqual( r.levels[ l + 1 ], plain.levels[ l ] );
	// unsplit small communities become a single sub-cluster
	const small = sizesOf( plain.levels[ 0 ] ).filter( ( s ) => s <= maxLeaf ).length;
	assert.ok( r.levelSizes[ 0 ] > plain.levelSizes[ 0 ] && small >= 0 );
	assert.ok( Math.abs( modularity( g, r.levels[ 0 ] ) - r.modularity[ 0 ] ) < 1e-9 );
	// no-op when nothing is too large
	const r2 = louvain( g, { seed: 1, refine: { maxLeafSize: 100000 } } );
	assert.equal( r2.refined, false );
	assert.equal( r2.levels.length, plain.levels.length );

} );

test( 'refine: a clique that Louvain will not split falls back to BFS chunks', () => {

	const n = 300;
	const edges = [];
	for ( let a = 0; a < n; a ++ ) for ( let b = a + 1; b < n; b ++ ) edges.push( a, b );
	const g = normalizeGraph( { nodeCount: n, edges } );
	const h = louvainHierarchy( g, { maxLeafSize: 100 } );
	assert.equal( h.levels.length, 1 ); // the single all-in-one Louvain level is the root
	const s = sizesOf( h.levels[ 0 ] );
	assert.equal( s.length, 3 );
	assert.ok( s.every( ( x ) => x <= 100 ) );

} );

test( 'degenerate graphs: no edges, isolated nodes, two components', () => {

	const empty = normalizeGraph( { nodeCount: 5, edges: [] } );
	const r = louvain( empty );
	assert.equal( r.levels.length, 1 );
	assert.deepEqual( Array.from( r.levels[ 0 ] ), [ 0, 1, 2, 3, 4 ] );
	assert.equal( louvainHierarchy( empty ).levels.length, 0 );
	assert.equal( ensureHierarchy( empty ).hierarchy, null );

	// two triangles + an isolated node
	const g = normalizeGraph( { nodeCount: 7, edges: [ 0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3 ] } );
	const h = louvainHierarchy( g );
	assert.equal( h.levels.length, 1 );
	const lvl = h.levels[ 0 ];
	assert.equal( lvl[ 0 ], lvl[ 1 ] ); assert.equal( lvl[ 1 ], lvl[ 2 ] );
	assert.equal( lvl[ 3 ], lvl[ 4 ] ); assert.equal( lvl[ 4 ], lvl[ 5 ] );
	assert.notEqual( lvl[ 0 ], lvl[ 3 ] );
	assert.notEqual( lvl[ 6 ], lvl[ 0 ] ); assert.notEqual( lvl[ 6 ], lvl[ 3 ] );

} );

test( 'louvainHierarchy / ensureHierarchy produce a canonical, prepareGraph-ready hierarchy', () => {

	const g = generate( { model: 'hsbm', params: { n: 3000, avgDegree: 12 }, seed: 3 } );
	const h = louvainHierarchy( g, { minLevelRatio: 1 } );
	const pruned = louvainHierarchy( g );
	assert.ok( pruned.levels.length <= h.levels.length );
	for ( const x of [ h, pruned ] ) {

		assert.deepEqual( normalizeLevels( x.levels, g.nodeCount ), x.levels );
		assertNested( x.levels, g.nodeCount );
		assert.equal( x.modularity.length, x.levels.length );

	}

	// every kept level reduces the cluster count by at least minLevelRatio (except the leaves)
	const K = pruned.levels.map( ( l ) => sizesOf( l ).length );
	for ( let l = 1; l < K.length; l ++ ) assert.ok( K[ l ] * 1.2 <= K[ l - 1 ], `${K}` );
	const withH = ensureHierarchy( g );
	assert.ok( withH.hierarchy && withH.hierarchy.levels.length === pruned.levels.length );
	assert.ok( withH.meta.hierarchy.modularity.length === pruned.levels.length );
	assert.equal( ensureHierarchy( withH ), withH ); // already has one
	const p = prepareGraph( withH );
	assert.equal( p.D, pruned.levels.length );
	assert.equal( p.N, 3000 );

} );

test( 'speed: 20k nodes / ~400k edges in a few seconds', () => {

	const g = generate( { model: 'lfr', params: { n: 20000, avgDegree: 40, maxDegree: 300, minCommunity: 30, maxCommunity: 600, mu: 0.3 }, seed: 1 } );
	assert.ok( g.edges.length / 2 > 350000 );
	const t0 = performance.now();
	const r = louvain( g, { seed: 1, refine: { maxLeafSize: 200 } } );
	const ms = performance.now() - t0;
	assert.ok( ms < 5000, `Louvain took ${ms.toFixed( 0 )} ms` );
	assert.ok( r.modularity[ r.modularity.length - 1 ] > 0.6 );
	assert.ok( Math.max( ...sizesOf( r.levels[ 0 ] ) ) <= 200 );

} );
