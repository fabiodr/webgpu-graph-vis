// Leiden community detection (src/data/leiden.js): connectivity of every cluster, modularity
// against Louvain, planted partition recovery (NMI / ARI), hierarchy validity, determinism, edge
// cases and a performance smoke test. The thresholds are calibrated on measurements; each test
// states the measured values it is derived from.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph, normalizeLevels } from '../src/data/format.js';
import {
	leiden, leidenHierarchy, ensureHierarchy, modularity, randomGraphModularity,
	DEFAULT_MAX_LEAF_SIZE, DEFAULT_MIN_LEVEL_RATIO, DEFAULT_MAX_LEVELS, __internals,
} from '../src/data/leiden.js';
import * as louvainModule from '../src/data/louvain.js';
import { generate } from '../src/data/generators.js';
import { prepareGraph, rng } from '../src/graph/prepare.js';

const { louvain, louvainHierarchy } = louvainModule;

// ── datasets ─────────────────────────────────────────────────────────────────────────────────

const IMSV = [ 'karate', 'dolphins', 'lesmis', 'polbooks', 'adjnoun', 'football', 'serengeti-foodweb', 'celegansneural' ];
const load = ( id ) => normalizeGraph( JSON.parse( readFileSync( `public/data/imsv/${id}.json`, 'utf8' ) ) );

// The generator specs of the presets of scripts/generate.mjs, copied: the thresholds below are
// calibrated on exactly these graphs (hsbm / lfr / caveman carry planted ground truth).
const PRESETS = {
	'caveman-24': { model: 'caveman', params: { cliques: 24, cliqueSize: 8, rewire: 0.02 }, seed: 1 },
	'hsbm-500': { model: 'hsbm', params: { n: 500, avgDegree: 10, depth: 2, branching: 5, mixing: 0.2, sizeSpread: 0.3, degreeSpread: 0.4 }, seed: 1 },
	'lfr-2k': { model: 'lfr', params: { n: 2000, avgDegree: 12, maxDegree: 80, mu: 0.2, degreeExponent: 2.5, communityExponent: 1.5, minCommunity: 15, maxCommunity: 120 }, seed: 1 },
	'hsbm-5k': { model: 'hsbm', params: { n: 5000, avgDegree: 16, depth: 3, branching: 5, mixing: 0.3, sizeSpread: 0.4, degreeSpread: 0.4 }, seed: 1 },
	'ba-5k': { model: 'ba', params: { n: 5000, m: 4 }, seed: 1 },
	'lfr-10k': { model: 'lfr', params: { n: 10000, avgDegree: 16, maxDegree: 150, mu: 0.25, degreeExponent: 2.5, communityExponent: 1.5, minCommunity: 20, maxCommunity: 250 }, seed: 1 },
	'hsbm-20k': { model: 'hsbm', params: { n: 20000, avgDegree: 24, depth: 3, branching: 6, mixing: 0.3, sizeSpread: 0.4, degreeSpread: 0.4 }, seed: 1 },
	'lfr-20k-dense': { model: 'lfr', params: { n: 20000, avgDegree: 40, maxDegree: 300, mu: 0.3, degreeExponent: 2.5, communityExponent: 1.5, minCommunity: 30, maxCommunity: 600 }, seed: 1 },
	'lattice-12': { model: 'lattice3d', params: { nx: 12, ny: 12, nz: 12, periodic: false }, seed: 1 },
	'rgg-3k': { model: 'rgg3d', params: { n: 3000, avgDegree: 10, periodic: false }, seed: 1 },
	'ws-2k': { model: 'ws', params: { n: 2000, k: 8, beta: 0.02 }, seed: 1 },
};

const graphCache = new Map();
function dataset( id ) {

	if ( ! graphCache.has( id ) ) graphCache.set( id, PRESETS[ id ] ? generate( PRESETS[ id ] ) : load( id ) );
	return graphCache.get( id );

}

const ALL = [ ...IMSV, ...Object.keys( PRESETS ) ];

// ── helpers ──────────────────────────────────────────────────────────────────────────────────

/** Independent reference: Q = 1/2m Σ_ij (A_ij − γ k_i k_j / 2m) δ(c_i, c_j) with a dense matrix. */
function denseModularity( g, part, gamma = 1 ) {

	const n = g.nodeCount;
	const A = Array.from( { length: n }, () => new Float64Array( n ) );
	for ( let k = 0; k < g.edges.length; k += 2 ) {

		const w = g.weights ? g.weights[ k >> 1 ] : 1;
		const a = g.edges[ k ], b = g.edges[ k + 1 ];
		// a self loop counts twice in A_ii (it adds 2 to the degree), like any other edge end
		A[ a ][ b ] += w;
		A[ b ][ a ] += w;

	}

	const deg = A.map( ( row ) => row.reduce( ( x, y ) => x + y, 0 ) );
	const m2 = deg.reduce( ( x, y ) => x + y, 0 );
	let q = 0;
	for ( let i = 0; i < n; i ++ ) for ( let j = 0; j < n; j ++ ) if ( part[ i ] === part[ j ] ) q += A[ i ][ j ] - ( gamma * deg[ i ] * deg[ j ] ) / m2;
	return q / m2;

}

function countOf( level ) {

	let m = - 1;
	for ( let i = 0; i < level.length; i ++ ) if ( level[ i ] > m ) m = level[ i ];
	return m + 1;

}

function sizesOf( level ) {

	const s = new Int32Array( countOf( level ) );
	for ( let i = 0; i < level.length; i ++ ) s[ level[ i ] ] ++;
	return Array.from( s );

}

const maxSizeOf = ( level ) => sizesOf( level ).reduce( ( a, b ) => Math.max( a, b ), 0 );

function assertDense( level, n, what ) {

	assert.equal( level.length, n, `${what}: one entry per node` );
	const K = countOf( level );
	const seen = new Uint8Array( K );
	for ( let i = 0; i < n; i ++ ) {

		assert.ok( Number.isInteger( level[ i ] ) && level[ i ] >= 0, `${what}: id of node ${i}` );
		seen[ level[ i ] ] = 1;

	}

	assert.ok( seen.every( ( x ) => x === 1 ), `${what}: cluster ids are dense` );

}

/** Every cluster of levels[l] lies in exactly one cluster of levels[l + 1]. */
function assertNested( levels, n, what ) {

	for ( let l = 0; l + 1 < levels.length; l ++ ) {

		const up = new Int32Array( countOf( levels[ l ] ) ).fill( - 1 );
		for ( let i = 0; i < n; i ++ ) {

			const f = levels[ l ][ i ], c = levels[ l + 1 ][ i ];
			if ( up[ f ] < 0 ) up[ f ] = c;
			else if ( up[ f ] !== c ) assert.fail( `${what}: level ${l} cluster ${f} is split across level ${l + 1}` );

		}

	}

}

/** Number of clusters of `level` whose induced subgraph is not connected (union-find over the intra-cluster edges). */
function disconnectedClusters( g, level ) {

	const n = g.nodeCount;
	const parent = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) parent[ i ] = i;
	const find = ( x ) => {

		while ( parent[ x ] !== x ) {

			parent[ x ] = parent[ parent[ x ] ];
			x = parent[ x ];

		}

		return x;

	};

	for ( let k = 0; k < g.edges.length; k += 2 ) {

		const a = g.edges[ k ], b = g.edges[ k + 1 ];
		if ( level[ a ] !== level[ b ] ) continue;
		const ra = find( a ), rb = find( b );
		if ( ra !== rb ) parent[ ra ] = rb;

	}

	const components = new Int32Array( countOf( level ) );
	for ( let i = 0; i < n; i ++ ) if ( find( i ) === i ) components[ level[ i ] ] ++;
	let bad = 0;
	for ( const c of components ) if ( c > 1 ) bad ++;
	return bad;

}

function contingency( a, b ) {

	const n = a.length, Kb = countOf( b );
	const cells = new Map();
	const sa = new Float64Array( countOf( a ) ), sb = new Float64Array( Kb );
	for ( let i = 0; i < n; i ++ ) {

		const key = a[ i ] * Kb + b[ i ];
		cells.set( key, ( cells.get( key ) || 0 ) + 1 );
		sa[ a[ i ] ] ++;
		sb[ b[ i ] ] ++;

	}

	return { n, cells, sa, sb, Kb };

}

/** Normalized mutual information, arithmetic normalisation: 2 I(A; B) / ( H(A) + H(B) ). */
function nmi( a, b ) {

	const { n, cells, sa, sb, Kb } = contingency( a, b );
	let ha = 0, hb = 0, mi = 0;
	for ( const x of sa ) if ( x > 0 ) ha -= ( x / n ) * Math.log( x / n );
	for ( const x of sb ) if ( x > 0 ) hb -= ( x / n ) * Math.log( x / n );
	for ( const [ key, v ] of cells ) {

		const ia = Math.floor( key / Kb ), ib = key - ia * Kb;
		mi += ( v / n ) * Math.log( ( v * n ) / ( sa[ ia ] * sb[ ib ] ) );

	}

	return ha === 0 && hb === 0 ? 1 : mi / ( ( ha + hb ) / 2 );

}

/** Adjusted Rand index. */
function ari( a, b ) {

	const { n, cells, sa, sb } = contingency( a, b );
	const pairs = ( x ) => ( x * ( x - 1 ) ) / 2;
	let sum = 0, sA = 0, sB = 0;
	for ( const v of cells.values() ) sum += pairs( v );
	for ( const x of sa ) sA += pairs( x );
	for ( const x of sb ) sB += pairs( x );
	const expected = ( sA * sB ) / pairs( n ), max = ( sA + sB ) / 2;
	return max === expected ? 1 : ( sum - expected ) / ( max - expected );

}

/** For every planted level: the found level with the best NMI, and its ARI. */
function plantedScores( levels, planted ) {

	return planted.map( ( p ) => {

		let best = { nmi: - 1, ari: - 1, level: - 1 };
		levels.forEach( ( l, k ) => {

			const v = nmi( l, p );
			if ( v > best.nmi ) best = { nmi: v, ari: ari( l, p ), level: k };

		} );
		return best;

	} );

}

const fmt = ( scores ) => scores.map( ( s ) => `${s.nmi.toFixed( 3 )}/${s.ari.toFixed( 3 )}` ).join( ' ' );

/** Erdős–Rényi G(n, m) with average degree k: no community structure at all. */
function randomGraph( n, k, seed ) {

	const rand = rng( seed );
	const edges = [], seen = new Set();
	const E = Math.round( ( n * k ) / 2 );
	while ( edges.length < 2 * E ) {

		const a = Math.floor( rand() * n ), b = Math.floor( rand() * n );
		if ( a === b ) continue;
		const key = a < b ? a * n + b : b * n + a;
		if ( seen.has( key ) ) continue;
		seen.add( key );
		edges.push( a, b );

	}

	return normalizeGraph( { nodeCount: n, edges } );

}

function clique( n ) {

	const edges = [];
	for ( let a = 0; a < n; a ++ ) for ( let b = a + 1; b < n; b ++ ) edges.push( a, b );
	return normalizeGraph( { nodeCount: n, edges } );

}

// The runs that the connectivity and the validity tests share: every strategy, with and without
// the leaf size limit (the 20k graphs only with it, to keep the suite fast).
const STRATEGIES = [ 'recursive', 'aggregate', 'flat' ];
const runCache = new Map();
function runsOf( id ) {

	if ( runCache.has( id ) ) return runCache.get( id );
	const g = dataset( id );
	const runs = [];
	for ( const strategy of STRATEGIES ) {

		runs.push( { label: `${id} ${strategy} maxLeaf ${DEFAULT_MAX_LEAF_SIZE}`, r: leiden( g, { seed: 1, strategy, refine: { maxLeafSize: DEFAULT_MAX_LEAF_SIZE } } ) } );
		if ( g.nodeCount <= 10000 ) runs.push( { label: `${id} ${strategy}`, r: leiden( g, { seed: 1, strategy } ) } );

	}

	// a small size limit exercises the forced splits on the small graphs too
	if ( g.nodeCount <= 5000 ) runs.push( { label: `${id} recursive maxLeaf 12`, r: leiden( g, { seed: 2, refine: { maxLeafSize: 12 } } ) } );
	runCache.set( id, runs );
	return runs;

}

// ── modularity ───────────────────────────────────────────────────────────────────────────────

test( 'modularity() matches the dense-matrix definition and louvain.js (γ, weights, self loops)', () => {

	const g = load( 'karate' );
	const hi = new Set( [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21 ] );
	const club = Array.from( { length: 34 }, ( _, i ) => ( hi.has( i ) ? 0 : 1 ) );
	for ( const gamma of [ 1, 0.5, 2 ] ) {

		assert.ok( Math.abs( modularity( g, club, gamma ) - denseModularity( g, club, gamma ) ) < 1e-12 );
		assert.ok( Math.abs( modularity( g, club, gamma ) - louvainModule.modularity( g, club, gamma ) ) < 1e-12 );

	}

	const r = leiden( g, { seed: 3 } );
	r.levels.forEach( ( lvl, l ) => {

		assert.ok( Math.abs( r.modularity[ l ] - denseModularity( g, lvl ) ) < 1e-12 );

	} );
	assert.ok( Math.abs( r.partitionModularity - denseModularity( g, r.partition ) ) < 1e-12 );
	const single = Array.from( { length: 34 }, ( _, i ) => i );
	assert.ok( Math.abs( modularity( g, single ) - denseModularity( g, single ) ) < 1e-12 );

	// weights, a self loop and a duplicated edge
	const w = { nodeCount: 5, edges: [ 0, 1, 1, 2, 2, 0, 0, 0, 3, 4, 3, 4, 2, 3 ], weights: [ 1, 2.5, 0.5, 3, 1, 4, 0.25 ] };
	for ( const part of [ [ 0, 0, 0, 1, 1 ], [ 0, 1, 1, 2, 2 ], [ 0, 0, 0, 0, 0 ] ] ) {

		assert.ok( Math.abs( modularity( w, part ) - denseModularity( w, part ) ) < 1e-12, `${part}` );
		assert.ok( Math.abs( modularity( w, part, 1.7 ) - louvainModule.modularity( w, part, 1.7 ) ) < 1e-12, `${part}` );

	}

} );

test( 'karate club: the modularity optimum 0.4198 (4 communities) for every seed', () => {

	// measured: Q = 0.41979 for all of seeds 1..20 (Louvain: 0.39–0.4198 depending on the seed)
	const g = load( 'karate' );
	for ( let seed = 1; seed <= 20; seed ++ ) {

		const r = leiden( g, { seed } );
		assert.ok( Math.abs( r.partitionModularity - 0.4198 ) < 2e-4, `seed ${seed}: Q = ${r.partitionModularity}` );
		assert.equal( r.communities, 4 );
		assert.ok( r.stable, `seed ${seed}: stopped because no node moved` );

	}

} );

test( 'modularity at least Louvain\'s minus ε on every dataset, for 5 seeds (2 on the 20k graphs)', ( t ) => {

	// measured (seeds 1..10): mean Leiden − mean Louvain is ≥ 0 on every dataset (up to +0.031 on
	// ba-5k and lattice-12); the worst single seed is −0.0041 (adjnoun). ε = 0.01 per seed and
	// 0.002 on the mean leave a margin for that, not for a regression.
	let worst = Infinity, worstId = '', sum = 0, count = 0;
	for ( const id of ALL ) {

		const g = dataset( id );
		const seeds = g.nodeCount > 10000 ? 2 : 5;
		let meanLouvain = 0, meanLeiden = 0;
		for ( let seed = 1; seed <= seeds; seed ++ ) {

			const a = louvain( g, { seed } );
			const b = leiden( g, { seed, strategy: 'flat' } );
			const qa = a.modularity[ a.modularity.length - 1 ], qb = b.partitionModularity;
			assert.ok( qb >= qa - 0.01, `${id} seed ${seed}: Leiden ${qb.toFixed( 4 )} vs Louvain ${qa.toFixed( 4 )}` );
			meanLouvain += qa / seeds;
			meanLeiden += qb / seeds;
			if ( qb - qa < worst ) {

				worst = qb - qa;
				worstId = `${id} seed ${seed}`;

			}

		}

		assert.ok( meanLeiden >= meanLouvain - 0.002, `${id}: mean Leiden ${meanLeiden.toFixed( 4 )} vs mean Louvain ${meanLouvain.toFixed( 4 )}` );
		sum += meanLeiden - meanLouvain;
		count ++;

	}

	t.diagnostic( `Leiden − Louvain modularity: mean over datasets ${( sum / count ).toFixed( 4 )}, worst single run ${worst.toFixed( 4 )} (${worstId})` );

} );

test( 'the Leiden partition does not depend on the hierarchy strategy', () => {

	const g = dataset( 'hsbm-500' );
	const [ a, b, c ] = STRATEGIES.map( ( strategy ) => leiden( g, { seed: 4, strategy } ) );
	assert.deepEqual( a.partition, b.partition );
	assert.deepEqual( a.partition, c.partition );
	assert.equal( c.levels.length, 1 );
	assert.deepEqual( c.levels[ 0 ], c.partition );
	// 'aggregate': the last level is the partition
	assert.deepEqual( b.levels[ b.levels.length - 1 ], b.partition );

} );

// ── the two phases of one level (white box) ──────────────────────────────────────────────────

test( 'one level: local moving keeps exact bookkeeping; refinement merges well-connected nodes inside their community', () => {

	const { csrFromEdges, createScratch, moveNodesFast, refinePartition } = __internals;
	for ( const id of [ 'karate', 'celegansneural', 'hsbm-500', 'lfr-2k', 'ba-5k', 'lattice-12', 'ws-2k' ] ) {

		for ( const seed of [ 1, 2 ] ) {

			const what = `${id} seed ${seed}`;
			const graph = dataset( id );
			const g = csrFromEdges( graph.nodeCount, graph.edges, graph.weights );
			const { n, start, adj, w, k } = g;
			const gamma = 1 / g.m2;
			const S = createScratch( n, adj.length );
			const rand = rng( seed );
			const comm = Int32Array.from( { length: n }, ( _, i ) => i );
			assert.ok( moveNodesFast( g, k, comm, gamma, rand, S ) > 0, what );

			// bookkeeping against direct sums: community totals and sizes, and for every node the
			// weight to the rest of its community (kept incrementally while nodes move)
			const tot = new Float64Array( n ), size = new Int32Array( n ), own = new Float64Array( n );
			for ( let v = 0; v < n; v ++ ) {

				tot[ comm[ v ] ] += k[ v ];
				size[ comm[ v ] ] ++;
				for ( let e = start[ v ]; e < start[ v + 1 ]; e ++ ) if ( comm[ adj[ e ] ] === comm[ v ] ) own[ v ] += w[ e ];

			}

			for ( let v = 0; v < n; v ++ ) {

				assert.equal( S.csize[ v ], size[ v ], `${what}: csize of community ${v}` );
				assert.ok( Math.abs( S.tot[ v ] - tot[ v ] ) < 1e-9, `${what}: tot of community ${v}` );
				assert.ok( Math.abs( S.ext[ v ] - own[ v ] ) < 1e-9, `${what}: ext of node ${v}: ${S.ext[ v ]} vs ${own[ v ]}` );

			}

			// refinement: a sub-partition of the communities with dense ids and connected parts
			const refined = new Int32Array( n );
			const K = refinePartition( g, k, comm, gamma, 0.01, rand, S, refined );
			assertDense( refined, n, `${what} refined` );
			assert.equal( countOf( refined ), K, what );
			const communities = size.reduce( ( c, x ) => c + ( x > 0 ? 1 : 0 ), 0 );
			assert.ok( K < n && K >= communities, `${what}: ${communities} communities, ${K} refined, ${n} nodes` );
			const parent = new Int32Array( K ).fill( - 1 );
			const rsize = new Int32Array( K ), rtot = new Float64Array( K ), rext = new Float64Array( K ), inside = new Float64Array( n );
			for ( let v = 0; v < n; v ++ ) {

				const r = refined[ v ];
				if ( parent[ r ] < 0 ) parent[ r ] = comm[ v ];
				assert.equal( parent[ r ], comm[ v ], `${what}: refined community ${r} crosses a community border` );
				rsize[ r ] ++;
				rtot[ r ] += k[ v ];
				for ( let e = start[ v ]; e < start[ v + 1 ]; e ++ ) {

					if ( comm[ adj[ e ] ] !== comm[ v ] ) continue;
					if ( refined[ adj[ e ] ] === r ) inside[ v ] += w[ e ];
					else rext[ r ] += w[ e ];

				}

			}

			assert.equal( disconnectedClusters( graph, refined ), 0, `${what}: refined communities are connected` );
			// every node that merged is well connected to its community S: w(v, S − v) ≥ γ' k_v (tot_S − k_v)
			const canShrink = new Uint8Array( K );
			for ( let v = 0; v < n; v ++ ) {

				const r = refined[ v ];
				if ( rsize[ r ] < 2 ) continue;
				const totS = tot[ comm[ v ] ];
				assert.ok( own[ v ] >= gamma * k[ v ] * ( totS - k[ v ] ) - 1e-12, `${what}: node ${v} merged without being well connected` );
				// the community that the last node merged into, C − v, was well connected to S for some v
				const t = rtot[ r ] - k[ v ];
				const x = rext[ r ] - ( own[ v ] - inside[ v ] ) + inside[ v ];
				if ( x >= gamma * t * ( totS - t ) - 1e-12 ) canShrink[ r ] = 1;

			}

			for ( let r = 0; r < K; r ++ ) if ( rsize[ r ] >= 2 ) assert.equal( canShrink[ r ], 1, `${what}: refined community ${r} has no well-connected predecessor` );

		}

	}

} );

test( 'a stable run is node optimal: no single node gains by moving to another community or by leaving', ( t ) => {

	// The queue of the local moving does not revisit every node whose best community changed
	// (community totals shift under non-neighbours too), so one pass is not a fixed point. A run
	// that stops as stable made a full pass without a move: the paper's node optimality.
	let checked = 0;
	for ( const id of [ 'karate', 'lesmis', 'football', 'caveman-24', 'hsbm-500', 'lfr-2k', 'hsbm-5k', 'ws-2k' ] ) {

		const g = dataset( id ), n = g.nodeCount;
		for ( const seed of [ 1, 2 ] ) {

			const r = leiden( g, { seed, strategy: 'flat', iterations: 0 } );
			assert.ok( r.stable, `${id} seed ${seed}` );
			const part = r.partition;
			const deg = new Float64Array( n ), tot = new Float64Array( r.communities );
			const neighbours = Array.from( { length: n }, () => [] );
			for ( let e = 0; e < g.edges.length; e += 2 ) {

				const a = g.edges[ e ], b = g.edges[ e + 1 ];
				deg[ a ] ++;
				deg[ b ] ++;
				neighbours[ a ].push( b );
				neighbours[ b ].push( a );

			}

			const m2 = g.edges.length;
			for ( let v = 0; v < n; v ++ ) tot[ part[ v ] ] += deg[ v ];
			const to = new Map();
			for ( let v = 0; v < n; v ++ ) {

				to.clear();
				for ( const u of neighbours[ v ] ) to.set( part[ u ], ( to.get( part[ u ] ) || 0 ) + 1 );
				// gain of joining community c from outside it: w(v, c) − k_v tot_c / 2m
				const stay = ( to.get( part[ v ] ) || 0 ) - ( deg[ v ] * ( tot[ part[ v ] ] - deg[ v ] ) ) / m2;
				assert.ok( stay >= - 1e-9, `${id} seed ${seed}: node ${v} would rather be alone (${stay})` );
				for ( const [ c, weight ] of to ) if ( c !== part[ v ] ) assert.ok( weight - ( deg[ v ] * tot[ c ] ) / m2 <= stay + 1e-9, `${id} seed ${seed}: node ${v} gains by moving to community ${c}` );
				checked ++;

			}

		}

	}

	t.diagnostic( `node optimality checked for ${checked} nodes of stable runs` );

} );

test( 'fast local moving: one iteration is within 0.002 of the stable modularity where communities are clear', () => {

	// measured, seeds 1..5: at most 0.0001 below the until-stable result on these graphs (without
	// queueing the neighbours of a moved node again: 0.006 to 0.034 below)
	for ( const id of [ 'caveman-24', 'hsbm-500', 'lfr-2k', 'hsbm-5k', 'lfr-10k' ] ) {

		const g = dataset( id );
		for ( const seed of [ 1, 2, 3 ] ) {

			const one = leiden( g, { seed, strategy: 'flat', iterations: 1 } );
			const stable = leiden( g, { seed, strategy: 'flat', iterations: 0 } );
			assert.equal( one.iterations, 1 );
			assert.ok( stable.stable && stable.iterations <= 6, `${id} seed ${seed}: ${stable.iterations} iterations` );
			assert.ok( one.partitionModularity >= stable.partitionModularity - 0.002, `${id} seed ${seed}: ${one.partitionModularity} after one iteration, ${stable.partitionModularity} stable` );

		}

	}

} );

// ── connectivity and hierarchy validity ──────────────────────────────────────────────────────

test( 'every community of every level induces a connected subgraph (all datasets, all strategies)', ( t ) => {

	// the checker itself: two triangles in one cluster are one disconnected cluster
	const two = { nodeCount: 7, edges: [ 0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3 ] };
	assert.equal( disconnectedClusters( two, [ 0, 0, 0, 0, 0, 0, 1 ] ), 1 );
	assert.equal( disconnectedClusters( two, [ 0, 0, 0, 1, 1, 1, 2 ] ), 0 );

	let clusters = 0, levels = 0, runs = 0;
	for ( const id of ALL ) {

		const g = dataset( id );
		for ( const { label, r } of runsOf( id ) ) {

			runs ++;
			r.levels.forEach( ( lvl, l ) => {

				assert.equal( disconnectedClusters( g, lvl ), 0, `${label}: level ${l} (${r.levelSizes[ l ]} clusters) has disconnected clusters` );
				clusters += r.levelSizes[ l ];
				levels ++;

			} );
			assert.equal( disconnectedClusters( g, r.partition ), 0, `${label}: partition` );

		}

	}

	// for reference only (not asserted): Louvain's levels on the same graphs
	const louvainBad = ALL.filter( ( id ) => louvainHierarchy( dataset( id ) ).levels.some( ( l ) => disconnectedClusters( dataset( id ), l ) > 0 ) );
	t.diagnostic( `connected: ${clusters} clusters in ${levels} levels of ${runs} runs on ${ALL.length} graphs; Louvain has disconnected clusters on: ${louvainBad.join( ', ' ) || 'none'}` );

} );

test( 'hierarchy validity: dense ids, levels nest and get coarser, canonical numbering, modularity per level', () => {

	for ( const id of ALL ) {

		const g = dataset( id ), n = g.nodeCount;
		for ( const { label, r } of runsOf( id ) ) {

			assert.ok( r.levels.length >= 1, label );
			assert.equal( r.levelSizes.length, r.levels.length, label );
			assert.equal( r.modularity.length, r.levels.length, label );
			r.levels.forEach( ( lvl, l ) => {

				assert.ok( lvl instanceof Int32Array, label );
				assertDense( lvl, n, `${label} level ${l}` );
				assert.equal( countOf( lvl ), r.levelSizes[ l ], `${label} level ${l}: levelSizes` );
				if ( l > 0 ) assert.ok( r.levelSizes[ l ] < r.levelSizes[ l - 1 ], `${label}: level ${l} is coarser (${r.levelSizes})` );

			} );
			// every level-l cluster maps to exactly one level-(l+1) cluster
			assertNested( r.levels, n, label );
			assertDense( r.partition, n, `${label} partition` );
			assert.equal( countOf( r.partition ), r.communities, label );
			// ids are numbered by first appearance, so format.js normalizeLevels leaves them alone
			const plain = r.levels.map( ( l ) => Array.from( l ) );
			const kept = plain.filter( ( l ) => countOf( l ) > 1 && countOf( l ) < n );
			assert.deepEqual( normalizeLevels( plain, n ), kept, `${label}: normalizeLevels` );

		}

		// spot check of the reported modularity (the dense reference is O(n²): small graphs only)
		if ( n <= 2000 ) {

			const { r } = runsOf( id )[ 0 ];
			r.levels.forEach( ( lvl, l ) => assert.ok( Math.abs( r.modularity[ l ] - denseModularity( g, lvl ) ) < 1e-9, `${id} level ${l}` ) );

		}

	}

} );

test( 'leidenHierarchy on every dataset: canonical levels, leaves within maxLeafSize, pruning rule', () => {

	for ( const id of ALL ) {

		const g = dataset( id ), n = g.nodeCount;
		const h = leidenHierarchy( g );
		assert.ok( h.levels.length >= 1 && h.levels.length <= DEFAULT_MAX_LEVELS, `${id}: ${h.levels.length} levels` );
		assert.equal( h.modularity.length, h.levels.length );
		assert.ok( h.levels.every( ( l ) => Array.isArray( l ) && l.length === n ) );
		assert.deepEqual( normalizeLevels( h.levels, n ), h.levels, `${id}: canonical` );
		assertNested( h.levels, n, id );
		const K = h.levels.map( countOf );
		assert.ok( K[ 0 ] < n && K[ K.length - 1 ] > 1, `${id}: ${K}` );
		for ( let l = 1; l < K.length; l ++ ) assert.ok( K[ l ] * DEFAULT_MIN_LEVEL_RATIO <= K[ l - 1 ], `${id}: ${K}` );
		assert.ok( maxSizeOf( h.levels[ 0 ] ) <= DEFAULT_MAX_LEAF_SIZE, `${id}: largest leaf ${maxSizeOf( h.levels[ 0 ] )}` );
		h.levels.forEach( ( lvl, l ) => {

			assert.equal( disconnectedClusters( g, lvl ), 0, `${id} level ${l}` );
			assert.ok( Math.abs( h.modularity[ l ] - modularity( g, lvl ) ) < 1e-9, `${id} level ${l}: modularity` );

		} );
		assert.match( h.method, h.refined ? /^leiden \(refined, max leaf 200\)$/ : /^leiden$/ );

	}

} );

// ── planted structure ────────────────────────────────────────────────────────────────────────

test( 'planted partitions: ring of cliques and LFR benchmarks', ( t ) => {

	// measured (seeds 1..12): caveman-24 NMI 1.000 (24 clusters); lfr-2k NMI 1.000 (48 of 48);
	// lfr-10k NMI 0.989–0.992, ARI ≥ 0.939 (the 200-node leaf limit cuts planted communities of up to 250)
	const cave = dataset( 'caveman-24' );
	const hc = leidenHierarchy( cave );
	assert.equal( countOf( hc.levels[ 0 ] ), 24 );
	assert.ok( nmi( hc.levels[ 0 ], cave.planted.levels[ 0 ] ) > 0.999 );

	for ( const [ id, minNmi, minAri ] of [ [ 'lfr-2k', 0.99, 0.98 ], [ 'lfr-10k', 0.98, 0.9 ] ] ) {

		const g = dataset( id );
		for ( const seed of [ 1, 2 ] ) {

			const h = leidenHierarchy( g, { seed } );
			const [ s ] = plantedScores( h.levels, g.planted.levels );
			assert.ok( s.nmi >= minNmi && s.ari >= minAri, `${id} seed ${seed}: NMI ${s.nmi.toFixed( 4 )} ARI ${s.ari.toFixed( 4 )}` );
			if ( seed === 1 ) t.diagnostic( `${id}: found ${h.levels.map( countOf ).join( '/' )}, planted ${countOf( g.planted.levels[ 0 ] )}, NMI/ARI ${fmt( [ s ] )}; Louvain ${fmt( plantedScores( louvainHierarchy( g ).levels, g.planted.levels ) )}` );

		}

	}

	// without the size limit lfr-10k is closer to the planted communities (measured, seeds 1..12:
	// NMI 0.993–0.997 against 0.989–0.992 with the limit)
	const g = dataset( 'lfr-10k' );
	const [ s ] = plantedScores( leidenHierarchy( g, { maxLeafSize: 0 } ).levels, g.planted.levels );
	assert.ok( s.nmi >= 0.99, `lfr-10k without maxLeafSize: NMI ${s.nmi.toFixed( 4 )}` );

} );

test( 'hierarchical SBM: every planted level is recovered, including the ones above the modularity optimum', ( t ) => {

	// depth 2 (25 blocks in 5 groups), the graph of the Louvain test. measured, seeds 1..5: leaf
	// NMI ≥ 0.994, top NMI 0.939–0.941 (Louvain: 0.967–0.987 and 0.738–0.753)
	const g2 = generate( { model: 'hsbm', params: { n: 2000, avgDegree: 14, depth: 2, branching: 5, mixing: 0.15, sizeSpread: 0.2, degreeSpread: 0 }, seed: 2 } );
	for ( const seed of [ 1, 2, 3 ] ) {

		const r = leiden( g2, { seed } );
		const s = plantedScores( r.levels, g2.planted.levels );
		assert.ok( s[ 0 ].nmi >= 0.98 && s[ 1 ].nmi >= 0.9, `depth 2, seed ${seed}: levels ${r.levelSizes}, NMI ${fmt( s )}` );
		// the Leiden partition is at least as good as the planted one in modularity
		assert.ok( r.partitionModularity >= modularity( g2, g2.planted.levels[ 0 ] ) - 0.005 );

	}

	// depth 3 (hsbm-5k: 125 / 25 / 5). measured, seeds 1..6: 126–127 / 26 / 5 clusters with NMI
	// ≥ 0.994 / 0.993 / 1.000; Louvain's levels stop at 23 clusters, top NMI 0.676
	const g3 = dataset( 'hsbm-5k' );
	const planted = g3.planted.levels;
	assert.deepEqual( planted.map( countOf ), [ 125, 25, 5 ] );
	for ( const seed of [ 1, 2, 3 ] ) {

		const h = leidenHierarchy( g3, { seed } );
		const s = plantedScores( h.levels, planted );
		assert.equal( h.levels.length, 3, `seed ${seed}: ${h.levels.map( countOf )}` );
		assert.deepEqual( s.map( ( x ) => x.level ), [ 0, 1, 2 ], 'each planted level is matched by its own found level' );
		assert.ok( s[ 0 ].nmi >= 0.98 && s[ 1 ].nmi >= 0.98 && s[ 2 ].nmi >= 0.999, `depth 3, seed ${seed}: NMI ${fmt( s )}` );
		assert.ok( s[ 0 ].ari >= 0.95 && s[ 1 ].ari >= 0.95 && s[ 2 ].ari >= 0.999, `depth 3, seed ${seed}: ARI ${fmt( s )}` );
		assert.equal( countOf( h.levels[ 2 ] ), 5 );

	}

	const lv = louvainHierarchy( g3 );
	t.diagnostic( `hsbm-5k NMI/ARI per planted level (125/25/5): Leiden ${leidenHierarchy( g3 ).levels.map( countOf ).join( '/' )} ${fmt( plantedScores( leidenHierarchy( g3 ).levels, planted ) )}; Louvain ${lv.levels.map( countOf ).join( '/' )} ${fmt( plantedScores( lv.levels, planted ) )}` );

	// hsbm-20k (216 / 36 / 6): the 6 top blocks need the coarsening margin (coarsenResolution);
	// measured, seeds 1..3: 228–229 / 36 / 6 clusters, NMI 0.989 / 1.000 / 1.000
	const g20 = dataset( 'hsbm-20k' );
	const h20 = leidenHierarchy( g20 );
	const s20 = plantedScores( h20.levels, g20.planted.levels );
	assert.deepEqual( h20.levels.map( countOf ).slice( 1 ), [ 36, 6 ], `${h20.levels.map( countOf )}` );
	assert.ok( s20[ 0 ].nmi >= 0.97 && s20[ 1 ].nmi >= 0.999 && s20[ 2 ].nmi >= 0.999, `hsbm-20k: NMI ${fmt( s20 )}` );
	t.diagnostic( `hsbm-20k NMI/ARI per planted level (216/36/6): Leiden ${h20.levels.map( countOf ).join( '/' )} ${fmt( s20 )}` );

} );

test( 'a community with sub-structure among flat communities keeps its sub-clusters as a level of their own', () => {

	// 10 flat communities of 200 nodes and one community of 200 made of 8 strong blocks of 25. The
	// whole-graph optimum has 11 communities (the blocks are below its resolution limit); the
	// blocks are found inside their community and must stay nested in it: they are much smaller
	// than the other communities, so they are not clusters of the same kind (balanceTree).
	// measured on 3 graphs × 4 seeds: always 18 / 11 clusters, the upper level exactly as planted
	const flats = 10, size = 200, blocks = 8, n = ( flats + 1 ) * size;
	const top = new Int32Array( n ), block = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) {

		top[ i ] = Math.floor( i / size );
		block[ i ] = top[ i ] < flats ? top[ i ] : flats + Math.floor( ( i - flats * size ) / ( size / blocks ) );

	}

	const pFlat = 14 / ( size - 1 ), pBlock = 8 / ( size / blocks - 1 ), pGroup = 6 / ( size - size / blocks ), pOut = 0.5 / ( n - size );
	const rand = rng( 2 );
	const edges = [];
	for ( let a = 0; a < n; a ++ ) for ( let b = a + 1; b < n; b ++ ) {

		const p = top[ a ] !== top[ b ] ? pOut : top[ a ] < flats ? pFlat : block[ a ] === block[ b ] ? pBlock : pGroup;
		if ( rand() < p ) edges.push( a, b );

	}

	const g = normalizeGraph( { nodeCount: n, edges } );
	for ( const seed of [ 1, 2 ] ) {

		const r = leiden( g, { seed } );
		assert.deepEqual( r.levelSizes, [ flats + blocks, flats + 1 ], `seed ${seed}` );
		assert.ok( nmi( r.levels[ 1 ], top ) > 0.999, `seed ${seed}: communities` );
		assert.ok( nmi( r.levels[ 0 ], block ) > 0.95, `seed ${seed}: blocks, NMI ${nmi( r.levels[ 0 ], block )}` );
		assert.equal( r.communities, flats + 1 );

	}

} );

test( 'a graph without community structure gets no nested levels', () => {

	// measured on 72 runs (6 sizes × 4 graphs × 3 seeds): always a single level, and the modularity
	// of the partition is within 0.03 of the random-graph estimate (Louvain reports 3 levels here)
	for ( const [ n, k, graphSeed ] of [ [ 400, 8, 11 ], [ 2000, 10, 12 ] ] ) {

		const g = randomGraph( n, k, graphSeed );
		for ( const seed of [ 1, 2 ] ) {

			const r = leiden( g, { seed } );
			assert.equal( r.levels.length, 1, `G(${n}, k = ${k}) seed ${seed}: ${r.levelSizes}` );
			assert.ok( Math.abs( r.partitionModularity - randomGraphModularity( n, k ) ) < 0.05, `Q = ${r.partitionModularity}` );

		}

	}

} );

test( 'randomGraphModularity: (1 − 2/√n)(2/k)^(2/3), bounded, decreasing in the degree', () => {

	assert.ok( Math.abs( randomGraphModularity( 400, 8 ) - ( 1 - 2 / 20 ) * Math.pow( 0.25, 2 / 3 ) ) < 1e-12 );
	assert.equal( randomGraphModularity( 4, 2 ), 1 );
	assert.equal( randomGraphModularity( 100, 0 ), 1 );
	for ( const n of [ 12, 100, 5000 ] ) {

		let prev = 1;
		for ( const k of [ 2, 4, 8, 16, 64 ] ) {

			const q = randomGraphModularity( n, k );
			assert.ok( q > 0 && q <= prev, `n ${n} k ${k}: ${q}` );
			prev = q;

		}

	}

} );

// ── determinism and options ──────────────────────────────────────────────────────────────────

test( 'deterministic for a seed, independent of earlier calls; the seed changes the result', () => {

	const g = generate( { model: 'lfr', params: { n: 3000, avgDegree: 12 }, seed: 4 } );
	for ( const strategy of STRATEGIES ) {

		const options = { seed: 7, strategy, refine: { maxLeafSize: 60 } };
		const a = leiden( g, options );
		leiden( dataset( 'hsbm-500' ), { seed: 3 } ); // an unrelated run in between
		const b = leiden( g, options );
		assert.equal( a.levels.length, b.levels.length );
		a.levels.forEach( ( l, k ) => assert.deepEqual( l, b.levels[ k ], `${strategy} level ${k}` ) );
		assert.deepEqual( a.partition, b.partition );
		assert.deepEqual( a.modularity, b.modularity );
		assert.equal( a.iterations, b.iterations );

	}

	const h1 = leidenHierarchy( g, { seed: 7 } ), h2 = leidenHierarchy( g, { seed: 7 } );
	assert.deepEqual( h1.levels, h2.levels );
	assert.deepEqual( h1.modularity, h2.modularity );

	// a graph with a rugged landscape: different seeds end in different partitions
	const ba = dataset( 'ba-5k' );
	const p1 = leiden( ba, { seed: 1, strategy: 'flat' } ), p2 = leiden( ba, { seed: 2, strategy: 'flat' } );
	assert.ok( p1.partition.some( ( x, i ) => x !== p2.partition[ i ] ) );
	assert.deepEqual( leiden( ba, { seed: 1, strategy: 'flat' } ).partition, p1.partition );

} );

test( 'resolution: higher γ gives more, smaller communities', () => {

	const g = load( 'lesmis' );
	const K = ( gamma ) => leiden( g, { seed: 1, resolution: gamma } ).communities;
	assert.ok( K( 0.5 ) < K( 1 ) && K( 1 ) < K( 2.5 ), `${K( 0.5 )} ${K( 1 )} ${K( 2.5 )}` );
	const h = leidenHierarchy( g, { resolution: 2.5 } );
	assert.equal( h.resolution, 2.5 );
	assert.ok( Math.abs( h.modularity[ 0 ] - modularity( g, h.levels[ 0 ], 2.5 ) ) < 1e-9, 'level modularity uses the same γ' );

} );

test( 'iterations: more iterations never lower the modularity; ≤ 0 runs until stable', () => {

	const g = dataset( 'lattice-12' );
	let prev = - 1;
	for ( const iterations of [ 1, 2, 4, 8 ] ) {

		const r = leiden( g, { seed: 1, strategy: 'flat', iterations } );
		assert.ok( r.iterations <= iterations );
		assert.ok( r.partitionModularity >= prev - 1e-12, `iterations ${iterations}: ${r.partitionModularity} after ${prev}` );
		prev = r.partitionModularity;

	}

	const s = leiden( dataset( 'hsbm-500' ), { seed: 1, strategy: 'flat', iterations: 0 } );
	assert.ok( s.stable && s.iterations >= 2 );
	assert.equal( s.passes, s.iterations );

} );

test( 'refinement: candidates are drawn ∝ exp( gain / θ )', () => {

	// Two paths y – v – x – z with weights 10, 1, 10 (the weak edge listed first). At a low
	// resolution each path is one community; inside it the refinement merges every node with a
	// neighbour. The heavy neighbour has gain ≈ 10 and the weak one ≈ 1, so for θ = 0.01 and
	// θ = 1 the heavy one is taken with probability > 0.999 and the refined level is {y, v}
	// {x, z} per path ('aggregate' exposes it as the finest level). For θ → ∞ every candidate and
	// "stay alone" are equally likely. measured over 60 seeds: 60 / 60 / 9 runs with that level
	// (always taking the first candidate with a non-negligible weight instead: 60 / 4 / 9)
	const g = { nodeCount: 8, edges: [ 1, 2, 0, 1, 2, 3, 5, 6, 4, 5, 6, 7 ], weights: [ 1, 10, 10, 1, 10, 10 ] };
	const greedy = [ 0, 0, 1, 1, 2, 2, 3, 3 ];
	const hits = ( randomness ) => {

		let count = 0;
		for ( let seed = 1; seed <= 60; seed ++ ) {

			const r = leiden( g, { resolution: 0.01, strategy: 'aggregate', seed, randomness } );
			assert.deepEqual( Array.from( r.partition ), [ 0, 0, 0, 0, 1, 1, 1, 1 ], `θ = ${randomness}, seed ${seed}` );
			assert.equal( disconnectedClusters( g, r.levels[ 0 ] ), 0 );
			if ( r.levels[ 0 ].every( ( c, i ) => c === greedy[ i ] ) ) count ++;

		}

		return count;

	};

	assert.equal( hits( 0.01 ), 60 );
	const one = hits( 1 ), flat = hits( 1e9 );
	assert.ok( one >= 58, `θ = 1: ${one} of 60 runs` );
	assert.ok( flat <= 30, `θ → ∞: ${flat} of 60 runs` );

} );

test( 'randomness → ∞: a refinement that merges nothing keeps the communities, connected', () => {

	// With a huge θ every candidate is as likely as staying alone, so now and then the refinement
	// of a level merges nothing at all. The communities of that level are then kept as they are.
	const pairs = { nodeCount: 4, edges: [ 0, 1, 2, 3 ] };
	for ( let seed = 1; seed <= 40; seed ++ ) assert.deepEqual( Array.from( leiden( pairs, { seed, randomness: 1e9 } ).partition ), [ 0, 0, 1, 1 ], `seed ${seed}` );
	// measured: karate still reaches Q = 0.4198 for seeds 1..20, stable after 2 to 5 iterations
	const g = load( 'karate' );
	for ( let seed = 1; seed <= 10; seed ++ ) {

		const r = leiden( g, { seed, randomness: 1e9, strategy: 'flat', iterations: 0 } );
		assert.ok( r.stable, `seed ${seed}` );
		assert.ok( Math.abs( r.partitionModularity - 0.4198 ) < 2e-4, `seed ${seed}: Q = ${r.partitionModularity}` );
		assert.equal( disconnectedClusters( g, r.partition ), 0 );

	}

} );

test( 'CPM quality: resolution is a density threshold (ring of 24 cliques of 8)', () => {

	const g = generate( { model: 'caveman', params: { cliques: 24, cliqueSize: 8, rewire: 0 }, seed: 1 } );
	// a clique of 8 has density 1 and two neighbouring cliques are joined by one edge: every γ
	// between the two densities gives exactly the cliques
	for ( const gamma of [ 0.1, 0.5 ] ) {

		const r = leiden( g, { quality: 'cpm', resolution: gamma, strategy: 'flat' } );
		assert.equal( r.communities, 24, `γ = ${gamma}` );
		assert.ok( sizesOf( r.partition ).every( ( s ) => s === 8 ) );
		assert.equal( disconnectedClusters( g, r.partition ), 0 );

	}

	// a low γ merges neighbouring cliques, which modularity's resolution limit does as well
	assert.ok( leiden( g, { quality: 'cpm', resolution: 0.01, strategy: 'flat' } ).communities < 24 );
	assert.throws( () => leiden( g, { quality: 'surprise' } ), /unknown quality/ );
	assert.throws( () => leiden( g, { strategy: 'bottom-up' } ), /unknown strategy/ );

} );

test( 'CPM: a pendant node never stays in the clique it was pulled into (well-connected communities after one iteration)', () => {

	// 4 separate cliques of 6, each with one pendant node. With γ = 0.4 a clique is one community
	// (density 1) and its pendant is better alone: w(v, S − v) = 1 < γ |S − v| = 2.4. When the
	// pendant joins its neighbour before the clique gathers around it, it is not well connected
	// to that community any more: the refinement must leave it alone and the local moving of the
	// aggregate graph must move it out (to an empty community). The paper guarantees this after
	// every iteration, so it holds for iterations = 1 as well, for every visiting order.
	const cliques = 4, size = 6;
	const edges = [];
	for ( let c = 0; c < cliques; c ++ ) {

		const o = c * ( size + 1 );
		for ( let a = 0; a < size; a ++ ) for ( let b = a + 1; b < size; b ++ ) edges.push( o + a, o + b );
		edges.push( o, o + size );

	}

	const g = { nodeCount: cliques * ( size + 1 ), edges };
	for ( const iterations of [ 1, 8 ] ) {

		for ( let seed = 1; seed <= 40; seed ++ ) {

			const r = leiden( g, { quality: 'cpm', resolution: 0.4, strategy: 'flat', iterations, seed } );
			const what = `iterations ${iterations}, seed ${seed}: ${Array.from( r.partition )}`;
			assert.equal( r.communities, 2 * cliques, what );
			for ( let c = 0; c < cliques; c ++ ) {

				const o = c * ( size + 1 );
				for ( let a = 1; a < size; a ++ ) assert.equal( r.partition[ o + a ], r.partition[ o ], what );
				assert.notEqual( r.partition[ o + size ], r.partition[ o ], what );

			}

		}

	}

} );

test( 'weighted edges steer the partition', () => {

	// two triangles joined by one edge
	const tri = { nodeCount: 6, edges: [ 0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 2, 3 ] };
	assert.deepEqual( Array.from( leiden( tri ).partition ), [ 0, 0, 0, 1, 1, 1 ] );
	assert.deepEqual( Array.from( leiden( { ...tri, weights: [ 5, 5, 5, 5, 5, 5, 0.1 ] } ).partition ), [ 0, 0, 0, 1, 1, 1 ] );
	// a heavy bridge pulls its two ends together
	const heavy = leiden( { ...tri, weights: [ 1, 1, 1, 1, 1, 1, 50 ] } ).partition;
	assert.equal( heavy[ 2 ], heavy[ 3 ] );
	assert.notEqual( heavy[ 0 ], heavy[ 2 ] );
	assert.notEqual( heavy[ 4 ], heavy[ 3 ] );

	// a weighted graph behaves like the multigraph with repeated edges
	const g = dataset( 'hsbm-500' );
	const E = g.edges.length / 2;
	const weights = new Float64Array( E );
	const multi = [];
	for ( let e = 0; e < E; e ++ ) {

		weights[ e ] = 1 + ( e % 3 );
		for ( let c = 0; c < weights[ e ]; c ++ ) multi.push( g.edges[ 2 * e ], g.edges[ 2 * e + 1 ] );

	}

	const a = leiden( { nodeCount: g.nodeCount, edges: g.edges, weights }, { seed: 5 } );
	const b = leiden( { nodeCount: g.nodeCount, edges: multi }, { seed: 5 } );
	assert.deepEqual( a.partition, b.partition );
	assert.ok( Math.abs( a.partitionModularity - b.partitionModularity ) < 1e-12 );
	assert.ok( Math.abs( a.partitionModularity - louvainModule.modularity( { nodeCount: g.nodeCount, edges: g.edges, weights }, a.partition ) ) < 1e-12 );

} );

// ── size limit ───────────────────────────────────────────────────────────────────────────────

test( 'refine.maxLeafSize: oversized leaves are cut into connected pieces as a new finest level', () => {

	// measured: 21 communities of up to 518 nodes -> 87 leaves of at most 79, the 21 untouched
	const g = generate( { model: 'lfr', params: { n: 5000, avgDegree: 15, minCommunity: 100, maxCommunity: 600, maxDegree: 80, mu: 0.25 }, seed: 1 } );
	const plain = leiden( g, { seed: 1 } );
	const maxLeaf = 80;
	assert.equal( plain.refined, false );
	assert.ok( maxSizeOf( plain.levels[ 0 ] ) > maxLeaf );
	const r = leiden( g, { seed: 1, refine: { maxLeafSize: maxLeaf } } );
	assert.ok( r.refined );
	assert.equal( r.levels.length, plain.levels.length + 1 );
	assert.ok( maxSizeOf( r.levels[ 0 ] ) <= maxLeaf );
	assert.ok( r.levelSizes[ 0 ] > plain.levelSizes[ 0 ] );
	assertNested( r.levels, g.nodeCount, 'refined' );
	for ( const lvl of r.levels ) assert.equal( disconnectedClusters( g, lvl ), 0 );
	// the coarser levels are untouched
	for ( let l = 0; l < plain.levels.length; l ++ ) assert.deepEqual( r.levels[ l + 1 ], plain.levels[ l ] );
	// the pieces are not needlessly small: fewer than twice the minimum number of leaves
	let minimum = 0;
	for ( const s of sizesOf( plain.levels[ 0 ] ) ) minimum += Math.ceil( s / maxLeaf );
	assert.ok( r.levelSizes[ 0 ] < 2 * minimum, `${r.levelSizes[ 0 ]} leaves, at least ${minimum} needed` );
	// no-op when nothing is too large
	const r2 = leiden( g, { seed: 1, refine: { maxLeafSize: 100000 } } );
	assert.equal( r2.refined, false );
	assert.equal( r2.levels.length, plain.levels.length );
	const h = leidenHierarchy( g, { maxLeafSize: maxLeaf } );
	assert.equal( h.method, 'leiden (refined, max leaf 80)' );
	assert.equal( h.maxLeafSize, maxLeaf );
	assert.equal( leidenHierarchy( g, { maxLeafSize: 0 } ).method, 'leiden' );

} );

test( 'refine.maxLeafSize: a clique falls back to connected chunks; a star to one chunk and singletons', () => {

	const g = clique( 300 );
	assert.deepEqual( leiden( g ).levelSizes, [ 1 ] );
	assert.equal( leidenHierarchy( g, { maxLeafSize: 0 } ).levels.length, 0 ); // one community is the implicit root
	const h = leidenHierarchy( g, { maxLeafSize: 100 } );
	assert.equal( h.levels.length, 1 );
	assert.deepEqual( sizesOf( h.levels[ 0 ] ), [ 100, 100, 100 ] );
	assert.ok( h.refined );

	// a star: every connected piece must contain the centre, so the rest can only be singletons
	const n = 300;
	const star = normalizeGraph( { nodeCount: n, edges: Array.from( { length: n - 1 }, ( _, i ) => [ 0, i + 1 ] ).flat() } );
	const hs = leidenHierarchy( star, { maxLeafSize: 100 } );
	assert.equal( hs.levels.length, 1 );
	const sizes = sizesOf( hs.levels[ 0 ] );
	assert.equal( sizes[ hs.levels[ 0 ][ 0 ] ], 100 );
	assert.equal( sizes.filter( ( s ) => s === 1 ).length, n - 100 );
	assert.equal( disconnectedClusters( star, hs.levels[ 0 ] ), 0 );

} );

// ── degenerate graphs ────────────────────────────────────────────────────────────────────────

test( 'degenerate graphs: empty, no edges, isolated nodes, disconnected components, one edge', () => {

	const none = { nodeCount: 0, edges: [] };
	const r0 = leiden( none );
	assert.deepEqual( r0.levels, [] );
	assert.equal( r0.communities, 0 );
	assert.deepEqual( leidenHierarchy( none ).levels, [] );
	assert.equal( ensureHierarchy( none ).hierarchy, null );

	const empty = normalizeGraph( { nodeCount: 5, edges: [] } );
	const r = leiden( empty );
	assert.equal( r.levels.length, 1 );
	assert.deepEqual( Array.from( r.levels[ 0 ] ), [ 0, 1, 2, 3, 4 ] );
	assert.equal( r.partitionModularity, 0 );
	assert.equal( leidenHierarchy( empty ).levels.length, 0 );
	assert.equal( ensureHierarchy( empty ).hierarchy, null );
	for ( const strategy of STRATEGIES ) assert.deepEqual( Array.from( leiden( empty, { strategy, refine: { maxLeafSize: 2 } } ).partition ), [ 0, 1, 2, 3, 4 ] );

	// two triangles + an isolated node: components are never merged, the isolated node stays alone
	const g = normalizeGraph( { nodeCount: 7, edges: [ 0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3 ] } );
	for ( const strategy of STRATEGIES ) {

		const h = leidenHierarchy( g, { strategy } );
		assert.equal( h.levels.length, 1 );
		assert.deepEqual( h.levels[ 0 ], [ 0, 0, 0, 1, 1, 1, 2 ] );

	}

	// one edge among isolated nodes
	assert.deepEqual( leidenHierarchy( { nodeCount: 4, edges: [ 1, 2 ] } ).levels, [ [ 0, 1, 1, 2 ] ] );

	// many components of different sizes, with isolated nodes in between
	const parts = [ generate( { model: 'caveman', params: { cliques: 6, cliqueSize: 6, rewire: 0 }, seed: 1 } ), dataset( 'hsbm-500' ), load( 'karate' ) ];
	const edges = [];
	const component = [];
	let offset = 0;
	parts.forEach( ( p, c ) => {

		for ( const v of p.edges ) edges.push( v + offset );
		for ( let i = 0; i < p.nodeCount; i ++ ) component.push( c );
		offset += p.nodeCount + 3; // 3 isolated nodes after every component
		for ( let i = 0; i < 3; i ++ ) component.push( - 1 );

	} );
	const multi = normalizeGraph( { nodeCount: offset, edges } );
	for ( const options of [ {}, { strategy: 'aggregate' }, { maxLeafSize: 20 } ] ) {

		const h = leidenHierarchy( multi, options );
		assertNested( h.levels, multi.nodeCount, 'components' );
		h.levels.forEach( ( lvl, l ) => {

			assert.equal( disconnectedClusters( multi, lvl ), 0, `components level ${l}` );
			// connected clusters never span two components; isolated nodes are singletons
			const sizes = sizesOf( lvl );
			for ( let i = 0; i < multi.nodeCount; i ++ ) if ( component[ i ] < 0 ) assert.equal( sizes[ lvl[ i ] ], 1 );

		} );

	}

} );

// ── canonical hierarchy and the louvain.js interface ─────────────────────────────────────────

test( 'drop-in for louvain.js: same exports, option names and result shapes', () => {

	for ( const name of [ 'ensureHierarchy', 'modularity', 'DEFAULT_MAX_LEAF_SIZE', 'DEFAULT_MIN_LEVEL_RATIO' ] ) assert.ok( name in louvainModule, name );
	assert.equal( DEFAULT_MAX_LEAF_SIZE, louvainModule.DEFAULT_MAX_LEAF_SIZE );
	assert.equal( DEFAULT_MIN_LEVEL_RATIO, louvainModule.DEFAULT_MIN_LEVEL_RATIO );

	const g = generate( { model: 'hsbm', params: { n: 3000, avgDegree: 12 }, seed: 3 } );
	const a = louvainHierarchy( g ), b = leidenHierarchy( g );
	assert.deepEqual( Object.keys( b ), Object.keys( a ) );
	for ( const key of Object.keys( a ) ) assert.equal( typeof b[ key ], typeof a[ key ], key );
	const ra = louvain( g ), rb = leiden( g );
	for ( const key of Object.keys( ra ) ) assert.ok( key in rb, `leiden() result has ${key}` );

	// the call the data worker makes
	const options = { force: true, seed: 1, resolution: 1, maxLeafSize: undefined, onProgress: () => {} };
	const wa = louvainModule.ensureHierarchy( g, options ), wb = ensureHierarchy( g, options );
	assert.deepEqual( Object.keys( wb ), Object.keys( wa ) );
	assert.deepEqual( Object.keys( wb.hierarchy ), Object.keys( wa.hierarchy ) );
	assert.deepEqual( Object.keys( wb.meta.hierarchy ), Object.keys( wa.meta.hierarchy ) );
	assert.match( wb.hierarchy.method, /^leiden/ );
	assert.equal( wb.meta.hierarchy.method, wb.hierarchy.method );
	assert.equal( wb.meta.hierarchy.maxLeafSize, DEFAULT_MAX_LEAF_SIZE );
	assert.equal( wb.meta.hierarchy.modularity.length, wb.hierarchy.levels.length );
	assert.equal( wb.edges, g.edges ); // a shallow copy: the arrays are shared
	assert.equal( wb.nodeCount, g.nodeCount );

} );

test( 'ensureHierarchy: keeps an existing hierarchy unless forced; the result feeds prepareGraph', () => {

	const g = generate( { model: 'hsbm', params: { n: 3000, avgDegree: 12 }, seed: 3 } );
	const h = leidenHierarchy( g );
	const withH = ensureHierarchy( g );
	assert.notEqual( withH, g );
	assert.ok( ! g.hierarchy, 'the input graph is not modified' );
	assert.deepEqual( withH.hierarchy.levels, h.levels );
	assert.deepEqual( withH.meta.hierarchy, { method: h.method, modularity: h.modularity, resolution: 1, seed: 1, refined: h.refined, maxLeafSize: DEFAULT_MAX_LEAF_SIZE } );
	assert.equal( ensureHierarchy( withH ), withH ); // already has one
	const forced = ensureHierarchy( withH, { force: true, seed: 9, resolution: 1.5 } );
	assert.notEqual( forced, withH );
	assert.equal( forced.meta.hierarchy.seed, 9 );
	assert.equal( forced.meta.hierarchy.resolution, 1.5 );

	const p = prepareGraph( withH );
	assert.equal( p.N, 3000 );
	assert.equal( p.D, h.levels.length );
	assert.deepEqual( Array.from( p.stats.clustersPerLevel ), h.levels.map( countOf ) );

	// an imsv graph with the paper's hierarchy: kept, or replaced when forced
	const lesmis = load( 'lesmis' );
	assert.equal( ensureHierarchy( lesmis ), lesmis );
	const re = ensureHierarchy( lesmis, { force: true } );
	assert.equal( re.hierarchy.method, 'leiden' );
	assert.equal( prepareGraph( re ).D, re.hierarchy.levels.length );

} );

test( 'level pruning: minLevelRatio keeps the finer of two close levels, maxLevels caps the depth', () => {

	const g = dataset( 'ws-2k' );
	const all = leidenHierarchy( g, { minLevelRatio: 1, maxLevels: 0 } );
	const K = all.levels.map( countOf );
	assert.ok( K.length >= 3, `${K}` );
	assert.equal( all.dropped, 0 );
	for ( const ratio of [ 1.2, 2, 6 ] ) {

		const h = leidenHierarchy( g, { minLevelRatio: ratio, maxLevels: 0 } );
		const Kp = h.levels.map( countOf );
		assert.equal( Kp[ 0 ], K[ 0 ], 'the finest level always stays' );
		for ( let l = 1; l < Kp.length; l ++ ) assert.ok( Kp[ l ] * ratio <= Kp[ l - 1 ], `ratio ${ratio}: ${Kp}` );
		assert.ok( Kp.every( ( k ) => K.includes( k ) ) );
		assert.equal( h.dropped, K.length - Kp.length );
		assert.equal( h.modularity.length, Kp.length );
		assertNested( h.levels, g.nodeCount, `ratio ${ratio}` );

	}

	for ( const maxLevels of [ 1, 2 ] ) {

		const h = leidenHierarchy( g, { minLevelRatio: 1, maxLevels } );
		assert.equal( h.levels.length, maxLevels );
		assert.equal( countOf( h.levels[ 0 ] ), K[ 0 ] );
		h.levels.forEach( ( lvl, l ) => assert.ok( Math.abs( h.modularity[ l ] - modularity( g, lvl ) ) < 1e-9 ) );

	}

} );

test( 'onProgress: the events the data worker reads (phase, level, pass, communities, modularity)', () => {

	const g = dataset( 'hsbm-5k' );
	const events = [];
	ensureHierarchy( g, { force: true, onProgress: ( p ) => events.push( p ) } );
	const main = events.filter( ( e ) => e.phase === 'leiden' ), nested = events.filter( ( e ) => e.phase === 'refine' );
	assert.equal( main.length + nested.length, events.length );
	assert.ok( main.length >= 2 && nested.length >= 2 );
	// the whole-graph run comes first
	assert.ok( events.indexOf( nested[ 0 ] ) > events.indexOf( main[ main.length - 1 ] ) );
	for ( const e of main ) {

		assert.ok( Number.isInteger( e.level ) && e.level >= 0 );
		assert.ok( Number.isInteger( e.pass ) && e.pass >= 1 );
		assert.ok( e.communities >= 1 && e.communities <= g.nodeCount );
		assert.ok( Number.isFinite( e.modularity ) && e.modularity > - 0.5 && e.modularity <= 1 );
		assert.equal( typeof e.modularity.toFixed( 3 ), 'string' );

	}

	for ( let k = 1; k < main.length; k ++ ) {

		assert.ok( main[ k ].pass >= main[ k - 1 ].pass, 'iterations in order' );
		if ( main[ k ].pass === main[ k - 1 ].pass ) assert.equal( main[ k ].level, main[ k - 1 ].level + 1 );

	}

	// the last event of the run reports the partition
	const r = leiden( g );
	const last = main[ main.length - 1 ];
	assert.equal( last.communities, r.communities );
	assert.ok( Math.abs( last.modularity - r.partitionModularity ) < 1e-9 );
	// nested clustering: pass = nodes done of communities = nodeCount, non-decreasing, ends complete
	for ( let k = 0; k < nested.length; k ++ ) {

		assert.equal( nested[ k ].communities, g.nodeCount );
		assert.ok( nested[ k ].pass >= 1 && nested[ k ].pass <= g.nodeCount );
		if ( k > 0 ) assert.ok( nested[ k ].pass >= nested[ k - 1 ].pass );

	}

	assert.equal( nested[ nested.length - 1 ].pass, g.nodeCount );

} );

// ── performance ──────────────────────────────────────────────────────────────────────────────

test( 'performance smoke test: 20k nodes / 384k edges within 1.5 s (timings printed)', ( t ) => {

	// measured on the development machine (Node 22, 4 cores, median of 3): lfr-20k-dense Leiden
	// ≈ 270 ms / Louvain ≈ 130 ms; hsbm-20k ≈ 255 ms / ≈ 115 ms; unstructured 380k-edge graphs
	// (BA m = 19) ≈ 540 ms / ≈ 590 ms. The budget is the 1.5 s target, for the best of 3 runs.
	const best = ( fn, runs = 3 ) => {

		let ms = Infinity, out = null;
		for ( let k = 0; k < runs; k ++ ) {

			const t0 = performance.now();
			out = fn();
			ms = Math.min( ms, performance.now() - t0 );

		}

		return { ms, out };

	};

	const cases = [
		[ 'lfr-20k-dense', dataset( 'lfr-20k-dense' ) ],
		[ 'hsbm-20k', dataset( 'hsbm-20k' ) ],
		[ 'ba-20k (m = 19, no communities)', generate( { model: 'ba', params: { n: 20000, m: 19 }, seed: 1 } ) ],
	];
	assert.equal( cases[ 0 ][ 1 ].nodeCount, 20000 );
	assert.ok( cases[ 0 ][ 1 ].edges.length / 2 > 380000 );
	for ( const [ id, g ] of cases ) {

		const a = best( () => leidenHierarchy( g ) );
		const b = best( () => louvainHierarchy( g ), 2 );
		const flat = best( () => leiden( g, { strategy: 'flat' } ), 1 );
		const line = `${id}: ${g.nodeCount} nodes, ${g.edges.length / 2} edges: leidenHierarchy ${a.ms.toFixed( 0 )} ms (levels ${a.out.levels.map( countOf ).join( '/' )}), ` +
			`Leiden partition only ${flat.ms.toFixed( 0 )} ms (${flat.out.iterations} iterations), louvainHierarchy ${b.ms.toFixed( 0 )} ms (levels ${b.out.levels.map( countOf ).join( '/' )})`;
		t.diagnostic( line );
		assert.ok( a.ms < 1500, `${id}: leidenHierarchy took ${a.ms.toFixed( 0 )} ms` );
		assert.ok( maxSizeOf( a.out.levels[ 0 ] ) <= DEFAULT_MAX_LEAF_SIZE );
		const qa = Math.max( ...a.out.modularity ), qb = Math.max( ...b.out.modularity );
		assert.ok( qa >= qb - 0.01, `${id}: best level modularity ${qa.toFixed( 4 )} vs Louvain ${qb.toFixed( 4 )}` );

	}

} );
