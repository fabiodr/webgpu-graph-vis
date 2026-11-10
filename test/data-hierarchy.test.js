import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { clusterGraph, applyClustering, describeHierarchy, shapeLevels, levelModularity, levelSizes, mergeCommunities, resolutionSteps, RESOLUTION_STEPS, CLUSTER_DEFAULTS, LADDER_DEFAULTS } from '../src/data/hierarchy.js';
import { nmi, ari, partSizes, bestMatch, isNested } from '../src/data/partition.js';
import { leiden, DEFAULT_QUEUE_CAP } from '../src/data/leiden.js';
import { generate } from '../src/data/generators.js';
import { normalizeGraph } from '../src/data/format.js';
import { parseGraph } from '../src/data/formats.js';
import { prepareGraph } from '../src/graph/prepare.js';
import { runRequest } from '../src/data/worker.js';

const manifest = JSON.parse( readFileSync( 'public/data/manifest.json', 'utf8' ) );
const load = ( id ) => normalizeGraph( { ...JSON.parse( readFileSync( `public/data/${id}.json`, 'utf8' ) ), hierarchy: null } );
const graphOf = ( n, pairs, weights = null ) => ( { nodeCount: n, edges: Uint32Array.from( pairs.flat() ), weights: weights ? Float32Array.from( weights ) : null } );

function checkLevels( g, levels, what ) {

	const n = g.nodeCount;
	levels.forEach( ( level, l ) => {

		assert.equal( level.length, n, `${what}: level ${l} length` );
		const K = levelSizes( level );
		const seen = new Uint8Array( K );
		for ( const c of level ) {

			assert.ok( Number.isInteger( c ) && c >= 0, `${what}: level ${l} ids` );
			seen[ c ] = 1;

		}

		assert.ok( seen.every( ( x ) => x === 1 ), `${what}: level ${l} ids are dense` );
		if ( l > 0 ) {

			assert.ok( isNested( levels[ l - 1 ], level ), `${what}: level ${l - 1} is nested in level ${l}` );
			assert.ok( K < levelSizes( levels[ l - 1 ] ), `${what}: level ${l} is coarser` );

		}

		assert.ok( K >= 2, `${what}: level ${l} has more than one cluster` );

	} );

}

// ── partition scores ─────────────────────────────────────────────────────────────────────────

test( 'partition scores: NMI, ARI, sizes, nesting, best level', () => {

	const a = [ 0, 0, 0, 1, 1, 1, 2, 2, 2 ];
	assert.equal( nmi( a, a ), 1 );
	assert.equal( ari( a, a ), 1 );
	assert.equal( nmi( a, [ 5, 5, 5, 9, 9, 9, 7, 7, 7 ] ), 1 ); // labels do not matter
	assert.equal( ari( a, [ 5, 5, 5, 9, 9, 9, 7, 7, 7 ] ), 1 );
	const b = [ 0, 1, 2, 0, 1, 2, 0, 1, 2 ]; // independent of a
	assert.ok( nmi( a, b ) < 1e-9 );
	assert.ok( Math.abs( ari( a, b ) + 0.5 ) < 1e-9 || ari( a, b ) <= 0 );
	const c = [ 0, 0, 0, 1, 1, 1, 1, 2, 2 ];
	assert.ok( nmi( a, c ) > 0.5 && nmi( a, c ) < 1 );
	assert.ok( ari( a, c ) > 0.3 && ari( a, c ) < 1 );
	assert.equal( nmi( a, c ), nmi( c, a ) );
	assert.equal( nmi( [ 0, 0, 0 ], [ 4, 4, 4 ] ), 1 );
	assert.throws( () => nmi( [ 0 ], [ 0, 1 ] ), /differ in length/ );
	assert.deepEqual( partSizes( [ 3, 3, 9, 3, 9, 1 ] ), [ 3, 2, 1 ] );
	assert.ok( isNested( [ 0, 0, 1, 1, 2, 2 ], [ 0, 0, 0, 0, 1, 1 ] ) );
	assert.ok( ! isNested( [ 0, 0, 1, 1, 2, 2 ], [ 0, 1, 1, 1, 2, 2 ] ) );
	const best = bestMatch( [ [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ], a, [ 0, 0, 0, 0, 0, 0, 1, 1, 1 ] ], a );
	assert.equal( best.level, 1 );
	assert.equal( best.nmi, 1 );
	assert.equal( best.clusters, 3 );

} );

// ── options ──────────────────────────────────────────────────────────────────────────────────

test( 'defaults: Leiden, resolution 1, ladder on; info says what was used', () => {

	assert.deepEqual( CLUSTER_DEFAULTS, { method: 'leiden', resolution: 1, seed: 1, maxLeafSize: 200, ladder: true } );
	const g = load( 'generated/hsbm-500' );
	const { levels, info } = clusterGraph( g );
	checkLevels( g, levels, 'hsbm-500' );
	assert.equal( info.method, 'leiden' );
	assert.equal( info.resolution, 1 );
	assert.equal( info.seed, 1 );
	assert.equal( info.maxLeafSize, 200 );
	assert.equal( info.ladder, true );
	assert.equal( info.levels, levels.length );
	assert.deepEqual( info.clustersPerLevel, levels.map( levelSizes ) );
	assert.equal( info.kinds.length, levels.length );
	assert.ok( info.kinds.every( ( k ) => [ 'community', 'group', 'split', 'planted' ].includes( k ) ) );
	assert.ok( info.mainLevel >= 0 && info.mainLevel < levels.length );
	assert.equal( info.modularityPerLevel.length, levels.length );
	assert.ok( Math.abs( info.modularity - levelModularity( g.nodeCount, g.edges, null, levels[ info.mainLevel ] ) ) < 1e-12 );
	assert.equal( info.quality, info.modularity ); // resolution 1
	assert.ok( info.modularity > 0.6 );
	assert.equal( info.weighted, false );
	assert.equal( info.capped, 0 );
	assert.ok( info.ms >= 0 );
	// deterministic
	assert.deepEqual( clusterGraph( g ).levels, levels );
	// the options spelled out give the same result; null and undefined mean "default"
	assert.deepEqual( clusterGraph( g, { method: 'leiden', resolution: 1, seed: 1, maxLeafSize: 200, ladder: true } ).levels, levels );
	assert.deepEqual( clusterGraph( g, { method: undefined, resolution: null, seed: undefined } ).levels, levels );
	assert.throws( () => clusterGraph( g, { method: 'kmeans' } ), /unknown method "kmeans" \(leiden \| louvain\)/ );

} );

test( 'method: louvain and leiden both give valid, good hierarchies; leaves respect maxLeafSize', () => {

	for ( const id of [ 'imsv/lesmis', 'generated/lfr-2k', 'generated/ba-5k' ] ) {

		const g = load( id );
		for ( const method of [ 'leiden', 'louvain' ] ) {

			const { levels, info } = clusterGraph( g, { method } );
			checkLevels( g, levels, `${id} ${method}` );
			assert.equal( info.method, method );
			assert.match( info.detail, new RegExp( '^' + method ) );
			assert.ok( Math.max( ...partSizes( levels[ 0 ] ) ) <= 200, `${id} ${method}: leaves of at most 200 nodes` );
			assert.ok( levels.length >= 2 && levels.length <= LADDER_DEFAULTS.maxLevels, `${id} ${method}: ${levels.length} levels` );
			assert.ok( info.clustersPerLevel[ levels.length - 1 ] <= 24, `${id} ${method}: a narrow top (${info.clustersPerLevel.join( '/' )})` );

		}

		const small = clusterGraph( g, { maxLeafSize: 20 } );
		assert.ok( Math.max( ...partSizes( small.levels[ 0 ] ) ) <= 20, id );
		assert.equal( small.info.maxLeafSize, 20 );

	}

} );

test( 'planted truth: Leiden recovers LFR and hSBM communities (NMI), at the right level', () => {

	const lfr = generate( { model: 'lfr', params: { n: 2000, avgDegree: 12, maxDegree: 80, mu: 0.2, minCommunity: 15, maxCommunity: 120 }, seed: 1 } );
	const r = clusterGraph( lfr );
	const m = bestMatch( r.levels, lfr.planted.levels[ 0 ] );
	assert.ok( m.nmi > 0.98, `lfr-2k NMI ${m.nmi.toFixed( 3 )}` );
	assert.ok( ari( r.levels[ m.level ], lfr.planted.levels[ 0 ] ) > 0.95 );
	assert.equal( r.info.kinds[ m.level ], 'community' );

	const hsbm = generate( { model: 'hsbm', params: { n: 5000, avgDegree: 16, depth: 3, branching: 5, mixing: 0.3, sizeSpread: 0.4, degreeSpread: 0.4 }, seed: 1 } );
	const h = clusterGraph( hsbm );
	hsbm.planted.levels.forEach( ( truth, l ) => {

		const best = bestMatch( h.levels, truth );
		assert.ok( best.nmi > 0.97, `hsbm-5k planted level ${l}: NMI ${best.nmi.toFixed( 3 )}` );

	} );
	// Louvain's dendrogram misses the coarse levels here; that is why it is not the default
	const lv = clusterGraph( hsbm, { method: 'louvain', ladder: false } );
	const top = bestMatch( lv.levels, hsbm.planted.levels[ hsbm.planted.levels.length - 1 ] );
	assert.ok( top.nmi < bestMatch( h.levels, hsbm.planted.levels[ hsbm.planted.levels.length - 1 ] ).nmi );

} );

test( 'resolution: 1 is the default; the number of communities follows it monotonically', () => {

	const gammas = [ 0.25, 0.5, 1, 2, 4 ];
	const ids = manifest.datasets.filter( ( d ) => d.file ).map( ( d ) => d.id );
	assert.ok( ids.length >= 17 );
	for ( const method of [ 'leiden', 'louvain' ] ) {

		for ( const id of ids ) {

			const g = load( id );
			const infos = gammas.map( ( resolution ) => clusterGraph( g, { method, resolution } ) );
			const counts = infos.map( ( r ) => r.info.communities );
			for ( let k = 1; k < counts.length; k ++ ) assert.ok( counts[ k ] >= counts[ k - 1 ], `${id} ${method}: communities ${counts.join( ' ≤ ' )}` );
			assert.ok( counts[ 4 ] > counts[ 0 ], `${id} ${method}: resolution 4 gives more communities than 0.25 (${counts.join( ', ' )})` );
			assert.ok( counts[ 2 ] > counts[ 0 ] || counts[ 4 ] > counts[ 2 ] );
			infos.forEach( ( r, k ) => {

				const what = `${id} ${method} γ = ${gammas[ k ]}`;
				checkLevels( g, r.levels, what );
				assert.equal( r.info.resolution, gammas[ k ] );
				// above 1 the grid point of the request; below 1 the one the grouping was found at
				if ( gammas[ k ] >= 1 ) assert.equal( r.info.resolutionUsed, gammas[ k ], what );
				else assert.ok( r.info.resolutionUsed >= gammas[ k ] && r.info.resolutionUsed < 1, `${what}: used ${r.info.resolutionUsed}` );
				// two levels at every resolution: one level is one control point per bundled edge
				assert.ok( r.levels.length >= 2 && r.levels.length <= LADDER_DEFAULTS.maxLevels, `${what}: ${r.levels.length} levels` );
				// the count is the cluster count of a level, and that level is a community level
				assert.equal( r.info.clustersPerLevel[ r.info.mainLevel ], r.info.communities, `${what}: the main level shows the communities (${r.info.clustersPerLevel.join( '/' )}, main ${r.info.mainLevel}, ${r.info.communities})` );
				assert.equal( r.info.kinds[ r.info.mainLevel ], 'community', what );
				assert.deepEqual( r.info.dustClusters, r.levels.map( () => 0 ) );
				assert.ok( Math.max( ...partSizes( r.levels[ 0 ] ) ) <= 200, `${what}: leaves of at most 200 nodes` );

			} );
			// quality is the modularity of the main level at the requested resolution
			const hi = infos[ 4 ];
			assert.ok( Math.abs( hi.info.quality - levelModularity( g.nodeCount, g.edges, g.weights, hi.levels[ hi.info.mainLevel ], 4 ) ) < 1e-12 );
			assert.ok( hi.info.quality <= hi.info.modularity + 1e-12 );

		}

	}

	const g = load( 'imsv/lesmis' );
	assert.deepEqual( clusterGraph( g, { resolution: 0 } ).info.resolution, 1 ); // not a resolution: default
	assert.deepEqual( clusterGraph( g, { resolution: 'abc' } ).info.resolution, 1 );

} );

test( 'resolution grid: a slider in quarter-octave steps never loses communities on the built-in graphs', () => {

	// the 17 grid points 2^( k / 4 ) from 0.25 to 4, on every stored graph, with both methods
	const gammas = [];
	for ( let k = - 2 * RESOLUTION_STEPS; k <= 2 * RESOLUTION_STEPS; k ++ ) gammas.push( Math.pow( 2, k / RESOLUTION_STEPS ) );
	const ids = manifest.datasets.filter( ( d ) => d.file ).map( ( d ) => d.id );
	assert.ok( ids.length >= 17 );
	// different counts along the slider: 24 cliques have little to offer, 34 nodes not much more
	const visible = { 'generated/caveman-24': 3, 'imsv/karate': 5 };
	for ( const id of ids ) {

		const g = load( id );
		for ( const method of [ 'leiden', 'louvain' ] ) {

			const runs = gammas.map( ( resolution ) => clusterGraph( g, { method, resolution } ) );
			const counts = runs.map( ( r ) => r.info.communities );
			const what = `${id} ${method}`;
			for ( let k = 1; k < counts.length; k ++ ) assert.ok( counts[ k ] >= counts[ k - 1 ], `${what}: ${counts.join( ' ' )}` );
			assert.ok( counts[ 0 ] >= 2, `${what}: at least two communities at the lowest resolution` );
			assert.ok( new Set( counts ).size >= ( visible[ id ] || 8 ), `${what}: the slider is visible (${counts.join( ' ' )})` );
			runs.forEach( ( r, k ) => {

				// what the count says is what a level shows, at every stop
				assert.equal( r.info.clustersPerLevel[ r.info.mainLevel ] - r.info.dustClusters[ r.info.mainLevel ], counts[ k ], `${what} γ = ${gammas[ k ].toFixed( 2 )}: ${r.info.clustersPerLevel.join( '/' )}, main level ${r.info.mainLevel}` );
				assert.ok( r.levels.length >= 2, `${what} γ = ${gammas[ k ].toFixed( 2 )}: ${r.levels.length} level` );
				assert.ok( Math.max( ...partSizes( r.levels[ 0 ] ) ) <= 200, `${what} γ = ${gammas[ k ].toFixed( 2 )}: leaf size` );

			} );
			// one stop asked for on its own, on a fresh copy of the graph (nothing kept from the
			// stops before): the same hierarchy as on the way up the slider
			if ( g.nodeCount <= 3000 ) for ( const k of [ 3, 11, 16 ] ) assert.deepEqual( clusterGraph( load( id ), { method, resolution: gammas[ k ] } ).levels, runs[ k ].levels, `${what} γ = ${gammas[ k ].toFixed( 2 )}: independent of the requests before` );

		}

	}

	// the grid: a request is computed at the grid point next to it on the side of 1
	assert.deepEqual( [ 1, 1.05, 1.19, 1.4, 1.41, 2, 2.8, 4, 100000 ].map( resolutionSteps ), [ 0, 0, 1, 2, 2, 4, 6, 8, 16 ] );
	assert.deepEqual( [ 0.99, 0.9, 0.84, 0.7, 0.71, 0.5, 0.35, 0.25, 1e-9 ].map( resolutionSteps ), [ 0, 0, - 1, - 2, - 2, - 4, - 6, - 8, - 16 ] );
	const g = load( 'imsv/lesmis' );
	const near = clusterGraph( g, { resolution: 1.05 } ), one = clusterGraph( g );
	assert.deepEqual( near.levels, one.levels );
	assert.equal( near.info.resolution, 1.05 );
	assert.equal( near.info.resolutionUsed, 1 );
	assert.ok( Math.abs( clusterGraph( g, { resolution: 2.8 } ).info.resolutionUsed - Math.pow( 2, 1.5 ) ) < 1e-9 );

} );

test( 'resolution below 1: the communities of resolution 1 are merged, never cut', () => {

	for ( const [ id, method ] of [ [ 'generated/lfr-2k', 'leiden' ], [ 'generated/hsbm-5k', 'leiden' ], [ 'generated/hsbm-5k', 'louvain' ], [ 'imsv/football', 'leiden' ], [ 'generated/ba-5k', 'leiden' ] ] ) {

		const g = load( id );
		const one = clusterGraph( g, { method, ladder: false } );
		const main = one.levels[ one.info.mainLevel ];
		let previous = one.info.communities;
		for ( const resolution of [ 0.7, 0.5, 0.35, 0.25, 0.1 ] ) {

			const r = clusterGraph( g, { method, resolution, ladder: false } );
			const what = `${id} ${method} γ = ${resolution}`;
			checkLevels( g, r.levels, what );
			const merged = r.levels[ r.info.mainLevel ];
			assert.equal( levelSizes( merged ), r.info.communities, what );
			assert.ok( r.info.communities >= 2 && r.info.communities <= previous, `${what}: ${r.info.communities} ≤ ${previous}` );
			previous = r.info.communities;
			// Leiden's main level at resolution 1 is its partition unless balancing replaced it:
			// the merged communities are unions of the partition, so of the level when that is it
			if ( levelSizes( main ) === one.info.communities ) assert.ok( isNested( main, merged ), `${what}: unions of the communities of resolution 1` );
			assert.ok( r.info.resolutionUsed <= 1 && r.info.resolutionUsed >= resolution - 1e-9, what );
			if ( r.info.communities < one.info.communities ) assert.match( r.info.detail, /merged at resolution/ );
			// the leaves stay within maxLeafSize: the finer levels of resolution 1 are nested inside
			assert.ok( Math.max( ...partSizes( r.levels[ 0 ] ) ) <= 200, `${what}: leaf sizes` );
			// (without the ladder too) the merged communities are a level of their own, and the
			// levels below them are levels of resolution 1
			for ( let l = 0; l < r.info.mainLevel; l ++ ) assert.ok( one.levels.some( ( level ) => levelSizes( level ) === levelSizes( r.levels[ l ] ) && nmi( level, r.levels[ l ] ) === 1 ), `${what}: level ${l} is a level of resolution 1` );

		}

	}

	// the quotient graph with self loops: the modularity of a grouping there is the modularity of
	// the merged partition on the graph, at any resolution
	const g = load( 'generated/hsbm-500' );
	const one = clusterGraph( g, { ladder: false } );
	const part = one.levels[ one.info.mainLevel ], K = levelSizes( part );
	const m = mergeCommunities( g, part, K, 4, 1 );
	assert.ok( m && m.count >= 2 && m.count < K && m.group.length === K );
	assert.ok( Math.abs( m.gamma - 0.5 ) < 1e-9 || m.gamma > 0.5 );
	const mergedLevel = Array.from( part, ( c ) => m.group[ c ] );
	const direct = levelModularity( g.nodeCount, g.edges, g.weights, mergedLevel, m.gamma );
	// the same number from the quotient graph, built here independently
	const acc = new Map();
	for ( let e = 0; e < g.edges.length / 2; e ++ ) {

		const a = Math.min( part[ g.edges[ 2 * e ] ], part[ g.edges[ 2 * e + 1 ] ] ), b = Math.max( part[ g.edges[ 2 * e ] ], part[ g.edges[ 2 * e + 1 ] ] );
		acc.set( a * K + b, ( acc.get( a * K + b ) || 0 ) + 1 );

	}

	const qe = [], qw = [];
	for ( const [ key, w ] of acc ) {

		qe.push( Math.floor( key / K ), key % K );
		qw.push( w );

	}

	assert.ok( Math.abs( levelModularity( K, qe, qw, m.group, m.gamma ) - direct ) < 1e-12 );
	assert.equal( mergeCommunities( g, part, K, 0, 1 ), null );
	assert.equal( mergeCommunities( g, [ 0, 0, 1 ], 2, 4, 1 ), null ); // two communities: nothing to merge

	// a graph without communities at resolution 1 stays as it is below 1
	const clique = generate( { model: 'caveman', params: { cliques: 1, size: 30 }, seed: 1 } );
	assert.deepEqual( clusterGraph( clique, { resolution: 0.5 } ).levels, clusterGraph( clique ).levels );

} );

test( 'resolution above 1: the communities of resolution 1 are split, each inside itself', () => {

	for ( const [ id, method ] of [ [ 'generated/lfr-2k', 'leiden' ], [ 'generated/lfr-2k', 'louvain' ], [ 'generated/hsbm-5k', 'leiden' ], [ 'generated/hsbm-5k', 'louvain' ], [ 'imsv/football', 'leiden' ], [ 'generated/ba-5k', 'leiden' ], [ 'generated/lattice-12', 'louvain' ] ] ) {

		const g = load( id );
		const one = clusterGraph( g, { method, ladder: false } );
		const anchor = one.levels[ one.info.mainLevel ];
		assert.equal( levelSizes( anchor ), one.info.communities );
		let previous = one.info.communities;
		for ( const resolution of [ 1.19, 1.41, 2, 2.83, 4, 8 ] ) {

			const r = clusterGraph( g, { method, resolution, ladder: false } );
			const what = `${id} ${method} γ = ${resolution}`;
			checkLevels( g, r.levels, what );
			const main = r.levels[ r.info.mainLevel ];
			assert.equal( levelSizes( main ), r.info.communities, what );
			assert.ok( r.info.communities >= previous, `${what}: ${r.info.communities} ≥ ${previous}` );
			previous = r.info.communities;
			// parts of the communities of resolution 1: the two partitions can be compared
			assert.ok( isNested( main, anchor ), `${what}: every community lies inside one of resolution 1` );
			if ( r.info.communities > one.info.communities ) assert.match( r.info.detail, /split at resolution/, what );
			assert.equal( r.info.kinds[ r.info.mainLevel ], 'community' );
			assert.ok( Math.max( ...partSizes( r.levels[ 0 ] ) ) <= 200, `${what}: leaf sizes` );
			// no community of one or two nodes is cut out of a larger one
			const before = partSizes( anchor ).filter( ( size ) => size < 3 ).length;
			assert.ok( partSizes( main ).filter( ( size ) => size < 3 ).length <= before, `${what}: no parts of fewer than 3 nodes` );

		}

		assert.ok( previous > one.info.communities, `${id} ${method}: the slider does something above 1` );

	}

	// a planted hierarchy with three scales: the slider walks down to the finest one and stays there
	const hsbm = generate( { model: 'hsbm', params: { n: 5000, avgDegree: 16, depth: 3, branching: 5, mixing: 0.3, sizeSpread: 0.4, degreeSpread: 0.4 }, seed: 1 } );
	const fine = hsbm.planted.levels[ 0 ];
	const at2 = clusterGraph( hsbm, { resolution: 2 } ), at4 = clusterGraph( hsbm, { resolution: 4 } );
	for ( const r of [ at2, at4 ] ) {

		assert.ok( nmi( r.levels[ r.info.mainLevel ], fine ) > 0.97, `hsbm-5k γ = ${r.info.resolution}: the finest planted level is the main level (${r.info.clustersPerLevel.join( '/' )})` );
		// and nothing is invented below it: no level of crumbs
		assert.equal( r.info.mainLevel, 0 );
		assert.ok( r.info.clustersPerLevel[ 0 ] < 160, `${r.info.clustersPerLevel[ 0 ]} leaves` );

	}

	// the hierarchy of resolution 1 is computed once per graph object; other options do not reuse it
	const g = load( 'generated/hsbm-500' );
	const a = clusterGraph( g, { resolution: 2 } );
	assert.deepEqual( clusterGraph( g, { resolution: 2, seed: 7 } ).levels, clusterGraph( load( 'generated/hsbm-500' ), { resolution: 2, seed: 7 } ).levels );
	assert.deepEqual( clusterGraph( g, { resolution: 2, method: 'louvain' } ).levels, clusterGraph( load( 'generated/hsbm-500' ), { resolution: 2, method: 'louvain' } ).levels );
	assert.deepEqual( clusterGraph( g, { resolution: 2, maxLeafSize: 20 } ).levels, clusterGraph( load( 'generated/hsbm-500' ), { resolution: 2, maxLeafSize: 20 } ).levels );
	assert.deepEqual( clusterGraph( g, { resolution: 2 } ).levels, a.levels );
	assert.ok( Math.max( ...partSizes( clusterGraph( g, { resolution: 2, maxLeafSize: 20 } ).levels[ 0 ] ) ) <= 20 );

} );

test( 'seeds: different seeds give equally good, similar partitions', () => {

	const g = load( 'generated/lfr-2k' );
	const runs = [ 1, 2, 3 ].map( ( seed ) => clusterGraph( g, { seed } ) );
	runs.forEach( ( r, k ) => assert.equal( r.info.seed, k + 1 ) );
	const main = runs.map( ( r ) => r.levels[ r.info.mainLevel ] );
	assert.ok( nmi( main[ 0 ], main[ 1 ] ) > 0.95 && nmi( main[ 0 ], main[ 2 ] ) > 0.95 );
	const q = runs.map( ( r ) => r.info.modularity );
	assert.ok( Math.max( ...q ) - Math.min( ...q ) < 0.02, q.join( ', ' ) );

} );

// ── the ladder ───────────────────────────────────────────────────────────────────────────────

test( 'ladder: a flat partition gets groups above it; found levels are kept as they are', () => {

	const g = load( 'generated/lfr-2k' );
	const plain = clusterGraph( g, { ladder: false } );
	const shaped = clusterGraph( g );
	assert.equal( plain.info.ladder, false );
	assert.ok( plain.info.kinds.every( ( k ) => k === 'community' ) );
	assert.ok( shaped.levels.length > plain.levels.length, 'levels were added' );
	assert.match( shaped.info.detail, /\+ ladder$/ );
	// every level of the plain run is still there, unchanged
	for ( const level of plain.levels ) assert.ok( shaped.levels.some( ( l ) => nmi( l, level ) === 1 && levelSizes( l ) === levelSizes( level ) ) );
	const top = shaped.info.clustersPerLevel[ shaped.levels.length - 1 ];
	assert.ok( top >= 3 && top <= LADDER_DEFAULTS.maxTop, `top level ${top}` );
	assert.equal( shaped.info.kinds[ shaped.levels.length - 1 ], 'group' );
	assert.equal( shaped.info.kinds[ shaped.info.mainLevel ], 'community' );
	// branching between consecutive levels stays within reach of the Barnes-Hut tree and the bundles
	for ( let l = 0; l + 1 < shaped.levels.length; l ++ ) assert.ok( shaped.info.clustersPerLevel[ l ] / shaped.info.clustersPerLevel[ l + 1 ] <= 2 * LADDER_DEFAULTS.maxBranch );

	// shapeLevels on its own, with custom limits
	const one = [ Int32Array.from( plain.levels[ 0 ] ) ];
	const s = shapeLevels( g, one, { maxTop: 6, branch: 4, seed: 1 } );
	assert.ok( s.levels.length >= 2 );
	assert.ok( levelSizes( s.levels[ s.levels.length - 1 ] ) <= 6 );
	assert.equal( s.kinds[ 0 ], 'community' );
	for ( let l = 1; l < s.levels.length; l ++ ) assert.ok( isNested( s.levels[ l - 1 ], s.levels[ l ] ) );
	const never = clusterGraph( g, { ladder: { maxLevels: 2 } } );
	assert.ok( never.levels.length <= 2 );

} );

test( 'ladder: leaves are split when the method finds a single level of few communities', () => {

	const g = load( 'imsv/karate' );
	const r = clusterGraph( g );
	checkLevels( g, r.levels, 'karate' );
	assert.equal( r.levels.length, 2 );
	assert.deepEqual( r.info.kinds, [ 'split', 'community' ] );
	assert.equal( r.info.mainLevel, 1 );
	assert.equal( r.info.communities, 4 );
	assert.ok( r.info.modularity > 0.4 );

} );

test( 'dust: isolated nodes and pairs do not become thousands of clusters', () => {

	// 300 triangles joined in a ring, 500 isolated nodes, 200 lone pairs
	const pairs = [];
	let n = 0;
	for ( let k = 0; k < 300; k ++ ) {

		pairs.push( [ n, n + 1 ], [ n + 1, n + 2 ], [ n + 2, n ], [ n, ( n + 3 ) % 900 ] );
		n += 3;

	}

	for ( let k = 0; k < 200; k ++ ) {

		pairs.push( [ n, n + 1 ] );
		n += 2;

	}

	n += 500;
	const g = graphOf( n, pairs );
	for ( const method of [ 'leiden', 'louvain' ] ) {

		const { levels, info } = clusterGraph( g, { method } );
		checkLevels( g, levels, `dust ${method}` );
		assert.equal( info.dust, 900 );
		assert.ok( levels.length >= 2 );
		assert.ok( info.clustersPerLevel[ levels.length - 1 ] <= 24, `${method}: top ${info.clustersPerLevel.join( '/' )}` );
		// pairs stay together, and no dust shares a finest cluster with the ring
		for ( let k = 0; k < 200; k ++ ) assert.equal( levels[ 0 ][ 900 + 2 * k ], levels[ 0 ][ 900 + 2 * k + 1 ] );
		const ring = new Set( levels[ 0 ].slice( 0, 900 ) );
		for ( let i = 900; i < n; i ++ ) assert.ok( ! ring.has( levels[ 0 ][ i ] ) );
		assert.ok( Math.max( ...partSizes( levels[ 0 ] ) ) <= 200 );
		// the dust clusters are counted per level, and the communities are counted without them
		assert.equal( info.dustClusters.length, levels.length );
		levels.forEach( ( level, l ) => assert.equal( new Set( level.slice( 900 ) ).size, info.dustClusters[ l ], `${method}: dust clusters of level ${l}` ) );
		assert.equal( info.clustersPerLevel[ info.mainLevel ] - info.dustClusters[ info.mainLevel ], info.communities );
		assert.equal( new Set( levels[ info.mainLevel ].slice( 0, 900 ) ).size, info.communities, `${method}: the communities are those of the ring` );

	}

} );

test( 'dust: no dust leaf above maxLeafSize, however many isolated nodes a file brings', () => {

	// the layout's Barnes-Hut tree is this hierarchy: one leaf of 3,300 isolated nodes would be
	// eleven million pair interactions per step
	const base = load( 'generated/lfr-2k' );
	const withDust = ( g, isolated, pairs ) => {

		const n = g.nodeCount + isolated + 2 * pairs;
		const edges = new Uint32Array( g.edges.length + 2 * pairs );
		edges.set( g.edges );
		for ( let k = 0; k < pairs; k ++ ) edges.set( [ g.nodeCount + isolated + 2 * k, g.nodeCount + isolated + 2 * k + 1 ], g.edges.length + 2 * k );
		return { nodeCount: n, edges, weights: null };

	};

	for ( const [ name, g, isolated, pairs ] of [ [ 'lfr-2k + 20,000 isolated', withDust( base, 20000, 0 ), 20000, 0 ], [ 'hsbm-500 + 5,000 isolated + 2,000 pairs', withDust( load( 'generated/hsbm-500' ), 5000, 2000 ), 5000, 2000 ], [ 'lfr-2k + 50 isolated', withDust( base, 50, 0 ), 50, 0 ] ] ) {

		const dust = isolated + 2 * pairs, first = g.nodeCount - dust;
		for ( const method of [ 'leiden', 'louvain' ] ) {

			let previous = 0;
			for ( const resolution of [ 0.5, 1, 2 ] ) {

				const what = `${name} ${method} γ = ${resolution}`;
				const { levels, info } = clusterGraph( g, { method, resolution } );
				checkLevels( g, levels, what );
				assert.equal( info.dust, dust, what );
				const sizes = partSizes( levels[ 0 ] );
				assert.ok( Math.max( ...sizes ) <= 200, `${what}: largest leaf ${Math.max( ...sizes )}` );
				// pairs stay together, dust and graph share no leaf
				const real = new Set( levels[ 0 ].slice( 0, first ) );
				for ( let i = first; i < g.nodeCount; i ++ ) assert.ok( ! real.has( levels[ 0 ][ i ] ), what );
				for ( let e = g.edges.length / 2 - pairs; e < g.edges.length / 2; e ++ ) assert.equal( levels[ 0 ][ g.edges[ 2 * e ] ], levels[ 0 ][ g.edges[ 2 * e + 1 ] ], `${what}: a pair in one leaf` );
				// a tree, not a list: no parent with hundreds of dust children
				for ( let l = 0; l + 1 < levels.length; l ++ ) {

					assert.ok( info.dustClusters[ l ] >= info.dustClusters[ l + 1 ], what );
					assert.ok( info.dustClusters[ l ] <= 24 * info.dustClusters[ l + 1 ], `${what}: dust branching ${info.dustClusters.join( '/' )}` );

				}

				assert.ok( info.dustClusters[ levels.length - 1 ] <= 24, `${what}: dust clusters at the top ${info.dustClusters.join( '/' )}` );
				assert.ok( info.dustClusters[ 0 ] >= dust / 200, what );
				// the communities are counted without the dust and still follow the resolution
				assert.equal( info.clustersPerLevel[ info.mainLevel ] - info.dustClusters[ info.mainLevel ], info.communities, what );
				assert.equal( new Set( levels[ info.mainLevel ].slice( 0, first ) ).size, info.communities, what );
				assert.ok( info.communities >= previous, `${what}: ${info.communities} ≥ ${previous}` );
				previous = info.communities;

			}

		}

	}

	// a little dust keeps the shape it always had: one dust cluster at the top, 32 nodes per leaf
	const few = clusterGraph( withDust( base, 50, 0 ) );
	assert.equal( few.info.dustClusters[ few.levels.length - 1 ], 1 );
	assert.equal( few.info.dustClusters[ 0 ], 2 );
	// no limit asked for: none applied
	const free = clusterGraph( withDust( base, 20000, 0 ), { maxLeafSize: 0 } );
	assert.equal( free.info.dustClusters[ free.levels.length - 1 ], 1 );

} );

// ── hostile inputs ───────────────────────────────────────────────────────────────────────────

test( 'hostile inputs: stars, cliques, bipartite, paths, no edges, tiny and disconnected graphs', () => {

	const star = ( n ) => graphOf( n, Array.from( { length: n - 1 }, ( _, i ) => [ 0, i + 1 ] ) );
	const clique = ( n ) => {

		const p = [];
		for ( let i = 0; i < n; i ++ ) for ( let j = i + 1; j < n; j ++ ) p.push( [ i, j ] );
		return graphOf( n, p );

	};

	const cases = {
		'star 5000': star( 5000 ),
		'clique 120': clique( 120 ),
		'K60,60': graphOf( 120, Array.from( { length: 3600 }, ( _, k ) => [ k % 60, 60 + Math.floor( k / 60 ) ] ) ),
		'path 3000': graphOf( 3000, Array.from( { length: 2999 }, ( _, i ) => [ i, i + 1 ] ) ),
		'two cliques': ( () => {

			const a = clique( 30 ), p = [];
			for ( let k = 0; k < a.edges.length; k += 2 ) p.push( [ a.edges[ k ], a.edges[ k + 1 ] ], [ a.edges[ k ] + 30, a.edges[ k + 1 ] + 30 ] );
			return graphOf( 60, p );

		} )(),
	};
	cases[ 'hubs 8 x 400' ] = ( () => {

		const p = [];
		for ( let h = 0; h < 8; h ++ ) {

			for ( let i = 1; i <= 400; i ++ ) p.push( [ h * 401, h * 401 + i ] );
			if ( h ) p.push( [ ( h - 1 ) * 401, h * 401 ] );

		}

		return graphOf( 8 * 401, p );

	} )();
	for ( const [ name, g ] of Object.entries( cases ) ) {

		for ( const method of [ 'leiden', 'louvain' ] ) {

			for ( const resolution of [ 0.25, 1, 4 ] ) {

				const what = `${name} ${method} at ${resolution}`;
				const t0 = performance.now();
				const { levels, info } = clusterGraph( g, { method, resolution } );
				const ms = performance.now() - t0;
				checkLevels( g, levels, what );
				// (a clique of 120 nodes is one leaf: no levels at all)
				assert.equal( levels.length >= 1, g.nodeCount > 200 || name === 'two cliques' || name === 'K60,60', `${what}: there is a hierarchy` );
				if ( levels.length ) {

					assert.ok( Math.max( ...partSizes( levels[ 0 ] ) ) <= 200, `${what}: leaf size` );
					assert.ok( levelSizes( levels[ levels.length - 1 ] ) <= LADDER_DEFAULTS.maxTop, `${what}: top level` );
					assert.ok( info.mainLevel >= 0 && info.mainLevel < levels.length, what );
					assert.equal( info.clustersPerLevel[ info.mainLevel ] - info.dustClusters[ info.mainLevel ], info.communities, `${what}: the main level shows the count` );

				}

				assert.equal( info.capped, 0, `${what}: the queue cap was not needed` );
				assert.ok( ms < 5000, `${what}: ${ms.toFixed( 0 )} ms` );
				// the result feeds prepareGraph
				const p = prepareGraph( { nodeCount: g.nodeCount, edges: g.edges, hierarchy: levels.length ? { method: info.detail, levels } : null } );
				assert.equal( p.D, levels.length );

			}

		}

	}

	// a resolution that would leave the nodes alone falls back: a star and a clique have no
	// communities to find at any resolution, the picture stays the one of resolution 1
	for ( const name of [ 'star 5000', 'clique 120' ] ) {

		for ( const method of [ 'leiden', 'louvain' ] ) {

			const one = clusterGraph( cases[ name ], { method } ), four = clusterGraph( cases[ name ], { method, resolution: 4 } );
			assert.deepEqual( four.info.clustersPerLevel, one.info.clustersPerLevel, `${name} ${method}` );
			assert.equal( four.info.resolution, 4 );
			assert.equal( four.info.resolutionUsed, 1, `${name} ${method}: says what it used` );
			// no communities: the levels are cuts to size, and they say so
			assert.ok( one.info.kinds.every( ( kind ) => kind === 'split' || kind === 'group' ), `${name} ${method}: ${one.info.kinds.join( '/' )}` );
			assert.equal( one.info.mainLevel, one.levels.length - 1 );

		}

	}

	assert.equal( clusterGraph( cases[ 'two cliques' ] ).info.communities, 2 );
	// nothing to cluster: no levels, no error
	for ( const g of [ graphOf( 1, [] ), graphOf( 2, [ [ 0, 1 ] ] ), graphOf( 50, [] ) ] ) {

		for ( const method of [ 'leiden', 'louvain' ] ) {

			const r = clusterGraph( g, { method } );
			assert.deepEqual( r.levels, [] );
			assert.equal( r.info.levels, 0 );
			assert.equal( r.info.mainLevel, - 1 );
			assert.equal( applyClustering( { ...g, name: 'x', meta: {} }, { method } ).hierarchy, null );

		}

	}

} );

test( 'queue cap: Leiden stops a runaway local-moving loop and still returns a partition', () => {

	assert.equal( DEFAULT_QUEUE_CAP, 64 );
	const g = load( 'generated/hsbm-500' );
	const capped = leiden( g, { queueCap: 1 } ); // at most one pop per node: certainly cut short
	assert.ok( capped.capped > 0 );
	assert.equal( capped.levels[ 0 ].length, g.nodeCount );
	const normal = leiden( g );
	assert.equal( normal.capped, 0 );
	// through the front end the cap is reported
	const r = clusterGraph( g, { queueCap: 1 } );
	assert.ok( r.info.capped > 0 );
	checkLevels( g, r.levels, 'capped' );

} );

// ── weights ──────────────────────────────────────────────────────────────────────────────────

test( 'weights: the clustering uses them (weighted modularity)', () => {

	// a ring of 12 nodes: heavy edges pair the nodes one way, or the other way
	const ring = Array.from( { length: 12 }, ( _, i ) => [ i, ( i + 1 ) % 12 ] );
	const even = graphOf( 12, ring, ring.map( ( _, i ) => ( i % 2 === 0 ? 10 : 0.1 ) ) );
	const odd = graphOf( 12, ring, ring.map( ( _, i ) => ( i % 2 === 1 ? 10 : 0.1 ) ) );
	for ( const method of [ 'leiden', 'louvain' ] ) {

		const a = clusterGraph( even, { method, ladder: false } ), b = clusterGraph( odd, { method, ladder: false } );
		assert.equal( a.info.weighted, true );
		for ( let i = 0; i < 12; i += 2 ) assert.equal( a.levels[ 0 ][ i ], a.levels[ 0 ][ i + 1 ], `${method}: heavy pair ${i}-${i + 1}` );
		for ( let i = 1; i < 12; i += 2 ) assert.equal( b.levels[ 0 ][ i ], b.levels[ 0 ][ ( i + 1 ) % 12 ], `${method}: heavy pair ${i}-${( i + 1 ) % 12}` );
		assert.notEqual( a.levels[ 0 ][ 1 ], a.levels[ 0 ][ 2 ] );

	}

	// levelModularity with weights against a hand computation: two pairs joined by a light edge
	const g = graphOf( 4, [ [ 0, 1 ], [ 2, 3 ], [ 1, 2 ] ], [ 3, 3, 1 ] );
	const m = 7; // total weight
	const q = ( 3 / m - ( 7 / ( 2 * m ) ) ** 2 ) * 2;
	assert.ok( Math.abs( levelModularity( 4, g.edges, g.weights, [ 0, 0, 1, 1 ] ) - q ) < 1e-12 );
	assert.ok( Math.abs( levelModularity( 4, g.edges, null, [ 0, 0, 1, 1 ] ) - ( ( 1 / 3 - 0.25 ) * 2 ) ) < 1e-12 );

	// through the pipeline: a weighted file
	const lines = ring.map( ( [ a, b ], i ) => `${a} ${b} ${i % 2 === 0 ? 10 : 0.1}` ).join( '\n' );
	const parsed = parseGraph( lines, 'ring.txt' ).graph;
	assert.ok( parsed.weights instanceof Float32Array );
	const c = applyClustering( parsed, { ladder: false } );
	assert.equal( c.clustering.weighted, true );
	for ( let i = 0; i < 12; i += 2 ) assert.equal( c.hierarchy.levels[ 0 ][ i ], c.hierarchy.levels[ 0 ][ i + 1 ] );

} );

// ── planted partitions ───────────────────────────────────────────────────────────────────────

test( 'planted: generator truth is used as it is; groups of a file become the top level', async () => {

	const g = generate( { model: 'hsbm', params: { n: 600, avgDegree: 10, depth: 2, branching: 4 }, seed: 2 } );
	const full = clusterGraph( g, { planted: { levels: g.planted.levels, complete: true } } );
	assert.equal( full.info.method, 'planted' );
	assert.deepEqual( full.info.kinds, [ 'planted', 'planted' ] );
	assert.deepEqual( full.info.clustersPerLevel, [ 16, 4 ] );
	assert.equal( nmi( full.levels[ 0 ], g.planted.levels[ 0 ] ), 1 );

	// partial: a group attribute on top, computed communities inside every group
	const lfr = generate( { model: 'lfr', params: { n: 1500, avgDegree: 12, mu: 0.15, minCommunity: 20, maxCommunity: 80 }, seed: 3 } );
	const group = Int32Array.from( lfr.planted.levels[ 0 ], ( c ) => c % 5 ); // five arbitrary unions of communities
	const part = clusterGraph( lfr, { planted: { levels: [ group ], complete: false, source: 'attribute "team"' } } );
	checkLevels( lfr, part.levels, 'partial planted' );
	const top = part.levels[ part.levels.length - 1 ];
	assert.equal( nmi( top, group ), 1 );
	assert.equal( part.info.kinds[ part.levels.length - 1 ], 'planted' );
	assert.equal( part.info.method, 'planted' );
	assert.match( part.info.detail, /^attribute "team" \+ leiden/ );
	assert.ok( part.levels.length >= 2, 'computed levels below the groups' );
	assert.ok( bestMatch( part.levels, lfr.planted.levels[ 0 ] ).nmi > 0.95 );
	// the resolution moves the communities inside the groups; the groups stay the top
	const seen = [];
	for ( const resolution of [ 0.25, 0.5, 1, 2, 4 ] ) {

		const r = clusterGraph( lfr, { resolution, planted: { levels: [ group ], complete: false, source: 'attribute "team"' } } );
		const what = `partial planted γ = ${resolution} (${r.info.clustersPerLevel.join( '/' )})`;
		checkLevels( lfr, r.levels, what );
		assert.equal( nmi( r.levels[ r.levels.length - 1 ], group ), 1, what );
		assert.equal( r.info.kinds[ r.levels.length - 1 ], 'planted', what );
		assert.equal( r.info.clustersPerLevel[ r.info.mainLevel ], r.info.communities, what );
		assert.ok( r.info.communities >= 5, what );
		assert.ok( Math.max( ...partSizes( r.levels[ 0 ] ) ) <= 200, what );
		seen.push( r.info.communities );

	}

	for ( let k = 1; k < seen.length; k ++ ) assert.ok( seen[ k ] >= seen[ k - 1 ], `communities inside the groups ${seen.join( ', ' )}` );
	assert.ok( seen[ 4 ] > seen[ 0 ] );

	// through the worker: options.cluster.mode 'planted' and a file with a group attribute
	const nodes = Array.from( { length: lfr.nodeCount }, ( _, i ) => ( { id: i, community: `t${group[ i ]}` } ) );
	const links = [];
	for ( let k = 0; k < lfr.edges.length; k += 2 ) links.push( { source: lfr.edges[ k ], target: lfr.edges[ k + 1 ] } );
	const source = { kind: 'text', text: JSON.stringify( { nodes, links } ), name: 'teams.json' };
	const auto = await runRequest( { source, cluster: { mode: 'auto' } } );
	assert.equal( auto.graph.clustering.method, 'leiden' );
	assert.deepEqual( auto.meta.file.groups, { source: 'attribute "community"', count: 5, levels: 1 } );
	assert.deepEqual( auto.graph.attributes.community.categories.slice().sort(), [ 't0', 't1', 't2', 't3', 't4' ] );
	const planted = await runRequest( { source, cluster: { mode: 'planted' } } );
	assert.equal( planted.graph.clustering.method, 'planted' );
	assert.equal( planted.graph.stats.clustersPerLevel[ planted.graph.D - 1 ], 5 );
	assert.equal( planted.graph.clustering.kinds[ planted.graph.D - 1 ], 'planted' );
	assert.match( planted.graph.clustering.detail, /^attribute "community" \+ leiden/ );
	// the top level of the prepared graph is the attribute
	const D = planted.graph.D, anc = planted.graph.anc;
	const topOf = Array.from( { length: lfr.nodeCount }, ( _, i ) => anc[ i * D + D - 1 ] );
	assert.equal( nmi( topOf, group ), 1 );

} );

// ── what the worker and the preset script share ──────────────────────────────────────────────

test( 'applyClustering: canonical graph with hierarchy, clustering and meta; describeHierarchy', () => {

	const g = load( 'generated/hsbm-500' );
	const out = applyClustering( g );
	assert.notEqual( out, g );
	assert.equal( out.edges, g.edges ); // shallow copy
	assert.equal( g.hierarchy, null );
	assert.equal( out.hierarchy.method, out.clustering.detail );
	assert.deepEqual( out.meta.hierarchy, {
		method: out.clustering.detail, algorithm: 'leiden', modularity: out.clustering.modularityPerLevel, resolution: 1, seed: 1,
		maxLeafSize: 200, ladder: true, kinds: out.clustering.kinds, communities: out.clustering.communities, mainLevel: out.clustering.mainLevel,
	} );
	assert.equal( out.clustering.clustersPerLevel[ out.clustering.mainLevel ], out.clustering.communities );
	const p = prepareGraph( out );
	assert.equal( p.D, out.hierarchy.levels.length );
	assert.deepEqual( p.clustering, out.clustering );
	assert.deepEqual( p.stats.clustersPerLevel, out.clustering.clustersPerLevel );

	const d = describeHierarchy( g, out.hierarchy.levels );
	assert.equal( d.method, 'file' );
	assert.equal( d.stored, true );
	assert.deepEqual( d.clustersPerLevel, out.clustering.clustersPerLevel );
	assert.deepEqual( d.modularityPerLevel, out.clustering.modularityPerLevel );
	const told = describeHierarchy( g, out.hierarchy.levels, { method: 'leiden', kinds: out.clustering.kinds, mainCount: out.clustering.communities } );
	assert.equal( told.method, 'leiden' );
	assert.equal( told.mainLevel, out.clustering.mainLevel );
	// a file that names its main level is taken at its word
	const named = describeHierarchy( g, out.hierarchy.levels, { method: 'leiden', mainLevel: 0, mainCount: 3 } );
	assert.equal( named.mainLevel, 0 );
	assert.equal( named.communities, named.clustersPerLevel[ 0 ] );
	assert.equal( describeHierarchy( g, out.hierarchy.levels, { mainLevel: 9 } ).mainLevel, d.mainLevel, 'a level that does not exist is not a main level' );
	assert.equal( describeHierarchy( g, [] ).levels, 0 );

} );

test( 'stored presets are what the worker computes with the default options', async () => {

	for ( const d of manifest.datasets.filter( ( x ) => x.group === 'generated' && x.file ) ) {

		const stored = JSON.parse( readFileSync( 'public/' + d.file, 'utf8' ) );
		assert.equal( stored.meta.hierarchy.algorithm, 'leiden', d.id );
		assert.equal( stored.meta.hierarchy.resolution, 1 );
		assert.equal( stored.meta.hierarchy.ladder, true );
		// from the file: the stored hierarchy, described by its meta
		const file = await runRequest( { source: { kind: 'text', text: JSON.stringify( stored ), name: 'x.json' } } );
		assert.equal( file.graph.clustering.method, 'leiden' );
		assert.equal( file.graph.clustering.stored, true );
		assert.equal( file.timings.cluster, undefined );
		// computed afresh by the worker from the generator spec
		const spec = { model: stored.meta.generator, params: stored.meta.params, seed: stored.meta.seed };
		const fresh = await runRequest( { source: { kind: 'generate', spec }, cluster: { mode: 'always' } } );
		assert.deepEqual( fresh.graph.stats.clustersPerLevel, file.graph.stats.clustersPerLevel, d.id );
		assert.deepEqual( Array.from( fresh.graph.anc ), Array.from( file.graph.anc ), `${d.id}: identical hierarchy` );
		assert.deepEqual( fresh.graph.clustering.kinds, file.graph.clustering.kinds );
		assert.equal( fresh.graph.clustering.communities, file.graph.clustering.communities );
		assert.equal( fresh.graph.clustering.mainLevel, file.graph.clustering.mainLevel );
		assert.equal( d.levels, file.graph.D );

	}

} );
