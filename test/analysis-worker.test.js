// Analytics worker protocol (the handler is called directly, with a fake clock so that slices,
// previews and time limits are deterministic) and the AnalysisClient on top of it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as GA from '../src/graph/analytics.js';
import { createHandler, topK } from '../src/analysis/worker.js';
import { AnalysisClient } from '../src/analysis/client.js';
import { NODE_METRICS, EDGE_METRICS, metricInfo } from '../src/analysis/metrics.js';
import { parseGraphJson } from '../src/data/format.js';
import { prepareGraph, rng } from '../src/graph/prepare.js';

// ── helpers ──────────────────────────────────────────────────────────────────────────────────

/** Connected-ish random graph: a ring of `n` nodes plus `extra` random chords (deterministic). */
function randomGraph( n, extra, seed = 7 ) {

	const rand = rng( seed ), seen = new Set(), edges = [];
	const add = ( a, b ) => {

		if ( a === b ) return;
		const key = a < b ? a * n + b : b * n + a;
		if ( seen.has( key ) ) return;
		seen.add( key );
		edges.push( a, b );

	};

	for ( let i = 0; i < n; i ++ ) add( i, ( i + 1 ) % n );
	for ( let k = 0; k < extra; k ++ ) add( Math.floor( rand() * n ), Math.floor( rand() * n ) );
	return GA.graphFromEdges( n, edges );

}

const graphMessage = ( g, id ) => ( { type: 'graph', graph: id, N: g.N, E: g.E, edges: g.edges, adjStart: g.adjStart, degree: g.degree, adjList: g.adjList } );

/**
 * A handler with a fake clock: every look at the clock advances it by `tick` ms, so a slice of
 * `sliceMs` is a fixed number of pivots whatever the machine does. `between( n )` runs after the
 * n-th slice (while the handler yields), which is where a real worker would receive messages.
 */
function harness( { tick = 5, sliceMs = 20, progressMs = 0 } = {} ) {

	const h = { out: [], transfers: [], slices: 0, between: null, time: 0 };
	h.handle = createHandler( {
		post: ( msg, transfer = [] ) => {

			// as a real postMessage does: the arrays of the transfer list are detached inside the
			// handler at once. `transfers` keeps the names of the transferred fields.
			h.transfers.push( transfer.map( ( buffer ) => Object.keys( msg ).find( ( key ) => msg[ key ] && msg[ key ].buffer === buffer ) ) );
			h.out.push( structuredClone( msg, { transfer } ) );

		},
		now: () => ( h.time += tick ),
		wait: async () => {

			h.slices ++;
			if ( h.between ) h.between( h.slices );
			await null;

		},
		sliceMs, progressMs,
	} );
	h.of = ( id, type ) => h.out.filter( ( m ) => m.id === id && ( ! type || m.type === type ) );
	h.settled = ( id ) => h.out.some( ( m ) => m.id === id && ( m.type === 'cancelled' || m.type === 'error' || ( ( m.type === 'metric' || m.type === 'summary' ) && m.final ) ) );
	h.idle = async ( ...ids ) => {

		for ( let k = 0; k < 200000 && ! ids.every( h.settled ); k ++ ) await new Promise( ( r ) => setImmediate( r ) );
		assert.ok( ids.every( h.settled ), `requests ${ids} did not settle` );

	};

	return h;

}

/** A Worker stand-in on this thread whose handler uses the fake clock (deterministic previews). */
class FakeWorker {

	constructor( config = {} ) {

		let time = 0;
		this.posted = [];
		this.onmessage = null;
		this.handle = createHandler( {
			post: ( data, transfer = [] ) => {

				const copy = structuredClone( data, { transfer } ); // transferred, as by a real worker
				queueMicrotask( () => this.onmessage && this.onmessage( { data: copy } ) );

			},
			now: () => ( time += 5 ), wait: () => new Promise( ( r ) => setImmediate( r ) ), sliceMs: 20, progressMs: 0, ...config,
		} );

	}

	postMessage( msg ) {

		this.posted.push( msg );
		queueMicrotask( () => this.handle( msg ) );

	}

	terminate() {

		this.onmessage = null;

	}

}

const sameArray = ( a, b, message ) => assert.deepEqual( Array.from( a ), Array.from( b ), message );

// ── handler ──────────────────────────────────────────────────────────────────────────────────

test( 'catalogue: library metrics plus relevance, edge metrics, metricInfo', () => {

	const ids = NODE_METRICS.map( ( m ) => m.id );
	for ( const m of GA.NODE_METRICS ) assert.ok( ids.includes( m.id ) );
	assert.ok( ids.includes( 'relevance' ) );
	assert.equal( metricInfo( 'component' ).kind, 'categorical' );
	assert.equal( metricInfo( 'pagerank' ).kind, 'sequential' );
	assert.deepEqual( EDGE_METRICS.map( ( m ) => m.id ), [ 'triangles', 'embeddedness', 'betweenness' ] );
	assert.equal( metricInfo( 'betweenness', 'edge' ).scale, 'rank' );
	assert.equal( metricInfo( 'nope' ), null );
	assert.equal( metricInfo( 'pagerank', 'edge' ), null );
	for ( const m of [ ...NODE_METRICS, ...EDGE_METRICS ] ) {

		assert.ok( [ 'linear', 'log', 'rank', 'robust' ].includes( m.scale ) );
		assert.ok( [ 'instant', 'fast', 'slow' ].includes( m.cost ) );
		assert.ok( m.label && m.description );

	}

} );

test( 'worker: cheap metrics are answered at once, in order, transferred, and equal the library', () => {

	const g = randomGraph( 60, 90 );
	const h = harness();
	h.handle( graphMessage( g, 'g1' ) );
	const cheap = NODE_METRICS.filter( ( m ) => m.cost !== 'slow' && ! m.needs ).map( ( m ) => m.id );
	cheap.forEach( ( metric, k ) => h.handle( { type: 'metric', id: k + 1, graph: 'g1', slot: 's' + k, metric } ) );
	// synchronous: all replies are there, one per request, in request order
	assert.deepEqual( h.out.map( ( m ) => m.id ), cheap.map( ( _, k ) => k + 1 ) );
	h.out.forEach( ( m, k ) => {

		assert.equal( m.type, 'metric' );
		assert.equal( m.metric, cheap[ k ] );
		assert.equal( m.target, 'node' );
		assert.equal( m.final, true );
		assert.equal( m.exact, true );
		assert.equal( m.scale, metricInfo( cheap[ k ] ).scale );
		assert.ok( m.values instanceof Float32Array && m.values.length === g.N );
		sameArray( m.values, GA.nodeMetric( g, cheap[ k ] ), cheap[ k ] );
		assert.deepEqual( h.transfers[ k ], [ 'values' ], 'values are in the transfer list' );
		assert.equal( m.normalized, undefined );

	} );

	// asking again: equal values in a fresh array (the first one was transferred away)
	h.handle( { type: 'metric', id: 50, graph: 'g1', slot: 'again', metric: 'pagerank', normalize: true } );
	const again = h.of( 50 )[ 0 ], first = h.out[ cheap.indexOf( 'pagerank' ) ];
	assert.notEqual( again.values, first.values );
	assert.notEqual( again.values.buffer, first.values.buffer );
	sameArray( again.values, first.values );
	sameArray( again.normalized, GA.normalize( again.values, { mode: 'log' } ) );
	assert.equal( again.normalizedScale, 'log' );
	assert.deepEqual( h.transfers.at( - 1 ), [ 'values', 'normalized' ] );

} );

test( 'worker: relevance is personalised PageRank from the sources; zero without sources', () => {

	const g = randomGraph( 40, 30 );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 1, graph: 1, metric: 'relevance', options: { sources: [ 3, 17 ] } } );
	const weights = new Float32Array( g.N );
	weights[ 3 ] = weights[ 17 ] = 1;
	sameArray( h.of( 1 )[ 0 ].values, GA.pageRank( g, { personalization: weights } ) );
	assert.ok( h.of( 1 )[ 0 ].values[ 3 ] > 1 / g.N );
	h.handle( { type: 'metric', id: 2, graph: 1, metric: 'relevance', options: { sources: [] } } );
	assert.ok( h.of( 2 )[ 0 ].values.every( ( v ) => v === 0 ) );
	h.handle( { type: 'metric', id: 3, graph: 1, metric: 'relevance', options: { sources: [ - 1, 99, 2.5 ] } } );
	assert.ok( h.of( 3 )[ 0 ].values.every( ( v ) => v === 0 ), 'invalid sources are ignored' );

} );

test( 'worker: edge metrics (triangles, embeddedness, betweenness) with worker-side normalisation', async () => {

	const g = randomGraph( 50, 120 );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 1, graph: 1, target: 'edge', slot: 'a', metric: 'triangles', normalize: true } );
	const tri = GA.clusteringCoefficient( g, { edges: true } ).edgeTriangles;
	const m1 = h.of( 1 )[ 0 ];
	assert.equal( m1.target, 'edge' );
	assert.equal( m1.values.length, g.E );
	sameArray( m1.values, tri.subarray( 0, g.E ) );
	sameArray( m1.normalized, GA.normalize( m1.values, { mode: 'log' } ) );

	// embeddedness against a brute-force Jaccard of the open neighbourhoods
	h.handle( { type: 'metric', id: 2, graph: 1, target: 'edge', slot: 'a', metric: 'embeddedness' } );
	const nb = ( v ) => new Set( g.adjList.subarray( g.adjStart[ v ], g.adjStart[ v ] + g.degree[ v ] ) );
	const m2 = h.of( 2 )[ 0 ];
	for ( let e = 0; e < g.E; e ++ ) {

		const a = g.edges[ 2 * e ], b = g.edges[ 2 * e + 1 ];
		const A = nb( a ), B = nb( b );
		A.delete( b ); B.delete( a );
		let common = 0;
		for ( const x of A ) if ( B.has( x ) ) common ++;
		const union = A.size + B.size - common;
		assert.ok( Math.abs( m2.values[ e ] - ( union ? common / union : 0 ) ) < 1e-6, `edge ${e}` );

	}

	// edge betweenness: a slow metric, refined to exact; rank-normalised on request
	h.handle( { type: 'metric', id: 3, graph: 1, target: 'edge', slot: 'b', metric: 'betweenness', options: { samples: 'exact' }, preview: 30, normalize: 'rank' } );
	await h.idle( 3 );
	const final = h.of( 3, 'metric' ).at( - 1 );
	assert.equal( final.final, true );
	assert.equal( final.exact, true );
	sameArray( final.values, GA.edgeBetweenness( g, { samples: Infinity } ) );
	sameArray( final.normalized, GA.normalize( final.values, { mode: 'rank' } ) );
	assert.ok( h.of( 3, 'metric' ).length > 1, 'previews before the final result' );

	// node betweenness now rides on the finished edge task: answered at once, exact
	const before = h.slices;
	h.handle( { type: 'metric', id: 4, graph: 1, slot: 'c', metric: 'betweenness', options: { samples: 'exact' } } );
	assert.equal( h.slices, before );
	sameArray( h.of( 4, 'metric' )[ 0 ].values, GA.betweenness( g, { samples: Infinity } ) );

} );

test( 'worker: slow metrics start coarse and refine to the exact result; progress is monotone', async () => {

	const g = randomGraph( 240, 400 );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 7, graph: 1, metric: 'betweenness', options: { samples: 'exact' }, preview: 40 } );
	assert.equal( h.settled( 7 ), false, 'a slow metric is not answered inside the handler' );
	await h.idle( 7 );

	const results = h.of( 7, 'metric' );
	const previews = results.slice( 0, - 1 ), final = results.at( - 1 );
	assert.ok( previews.length >= 4, `${previews.length} previews` );
	assert.ok( previews.every( ( m ) => m.final === false && m.exact === false && m.total === g.N ) );
	for ( let k = 1; k < previews.length; k ++ ) assert.ok( previews[ k ].pivots > previews[ k - 1 ].pivots, 'each preview has more pivots' );
	assert.ok( previews[ 0 ].pivots <= 16, `the first preview is coarse (${previews[ 0 ].pivots} pivots)` );
	// previews get calmer: the gaps between them grow
	const gaps = previews.slice( 1 ).map( ( m, k ) => m.pivots - previews[ k ].pivots );
	assert.ok( gaps.at( - 1 ) > gaps[ 0 ], `gaps ${gaps}` );

	// every preview is the library's estimate from exactly that many pivots
	const reference = GA.betweennessTask( g, { samples: Infinity } );
	for ( const m of previews ) {

		reference.step( m.pivots - reference.completed );
		sameArray( m.values, reference.result().node, `preview at ${m.pivots} pivots` );
		assert.ok( m.values.every( Number.isFinite ) );

	}

	assert.equal( final.final, true );
	assert.equal( final.exact, true );
	assert.equal( final.pivots, g.N );
	sameArray( final.values, GA.betweenness( g, { samples: Infinity } ) );
	// each message owns its buffer (they are transferred one by one)
	assert.equal( new Set( results.map( ( m ) => m.values.buffer ) ).size, results.length );

	const progress = h.of( 7, 'progress' );
	assert.ok( progress.length > previews.length );
	for ( let k = 1; k < progress.length; k ++ ) assert.ok( progress[ k ].done >= progress[ k - 1 ].done );
	assert.ok( progress.every( ( p ) => p.total === g.N ) );
	assert.equal( progress.at( - 1 ).done, g.N );
	// nothing after the final result
	assert.equal( h.out.at( - 1 ), final );

} );

test( 'worker: jobs of different slots that share a task each get their own arrays, to the end', async () => {

	// post() transfers for real here (see harness): an array handed to one job is gone for the next
	const g = randomGraph( 200, 320 );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 1, graph: 1, slot: 'lens:betweenness', metric: 'betweenness', options: { samples: 'exact' }, preview: 40 } );
	h.handle( { type: 'metric', id: 2, graph: 1, slot: 'card', metric: 'betweenness', options: { samples: 'exact' }, preview: 40, normalize: true } );
	// the two names of one closeness task, and one of them twice
	h.handle( { type: 'metric', id: 3, graph: 1, slot: 'c', metric: 'closeness', options: { samples: 'exact' }, preview: 40 } );
	h.handle( { type: 'metric', id: 4, graph: 1, slot: 'h', metric: 'harmonic', options: { samples: 'exact' }, preview: 40 } );
	h.handle( { type: 'metric', id: 5, graph: 1, slot: 'c2', metric: 'closeness', options: { samples: 'exact' } } );
	await h.idle( 1, 2, 3, 4, 5 );
	assert.deepEqual( h.out.filter( ( m ) => m.type === 'error' ), [] );

	const exact = GA.betweenness( g, { samples: Infinity } ), cl = GA.closeness( g, { samples: Infinity } );
	const expected = { 1: exact, 2: exact, 3: cl.closeness, 4: cl.harmonic, 5: cl.closeness };
	for ( const id of [ 1, 2, 3, 4, 5 ] ) {

		const results = h.of( id, 'metric' ), final = results.at( - 1 );
		assert.equal( final.final, true, `job ${id}` );
		assert.equal( final.exact, true, `job ${id}` );
		assert.equal( final.pivots, g.N );
		sameArray( final.values, expected[ id ], `job ${id}` );
		if ( id !== 5 ) assert.ok( results.length > 2, `job ${id} had previews of its own (${results.length - 1})` );
		for ( const m of results ) assert.equal( m.values.length, g.N, `job ${id}: every message carries a whole array` );

	}

	sameArray( h.of( 2, 'metric' ).at( - 1 ).normalized, GA.normalize( exact, { mode: 'log' } ) );
	// every preview of either job is the library's estimate from that many pivots
	for ( const id of [ 1, 2 ] ) {

		const reference = GA.betweennessTask( g, { samples: Infinity } );
		for ( const m of h.of( id, 'metric' ) ) {

			reference.step( m.pivots - reference.completed );
			sameArray( m.values, reference.result().node, `job ${id} at ${m.pivots} pivots` );

		}

	}

	// a job that fails on its own (unknown normalize mode) does not take the shared task down
	h.handle( { type: 'metric', id: 20, graph: 1, slot: 'bad', metric: 'betweenness', options: { samples: 'exact', seed: 9 }, preview: 40, normalize: 'bogus' } );
	h.handle( { type: 'metric', id: 21, graph: 1, slot: 'good', metric: 'betweenness', options: { samples: 'exact', seed: 9 }, preview: 40 } );
	await h.idle( 20, 21 );
	assert.match( h.of( 20, 'error' )[ 0 ].message, /unknown mode/ );
	assert.equal( h.of( 20, 'metric' ).length, 0 );
	sameArray( h.of( 21, 'metric' ).at( - 1 ).values, exact );
	assert.equal( h.of( 21, 'error' ).length, 0 );

	// the same for an edge task that serves two normalisations and a node job
	h.handle( { type: 'metric', id: 6, graph: 1, target: 'edge', slot: 'e:rank', metric: 'betweenness', options: { samples: 'exact', seed: 5 }, preview: 40, normalize: 'rank' } );
	h.handle( { type: 'metric', id: 7, graph: 1, target: 'edge', slot: 'e:log', metric: 'betweenness', options: { samples: 'exact', seed: 5 }, preview: 40, normalize: 'log' } );
	h.handle( { type: 'metric', id: 8, graph: 1, slot: 'n', metric: 'betweenness', options: { samples: 'exact', seed: 5 }, preview: 40 } );
	await h.idle( 6, 7, 8 );
	assert.deepEqual( h.out.filter( ( m ) => m.type === 'error' && m.id !== 20 ), [] );
	const eb = GA.edgeBetweenness( g, { samples: Infinity } );
	sameArray( h.of( 6, 'metric' ).at( - 1 ).values, eb );
	sameArray( h.of( 7, 'metric' ).at( - 1 ).values, eb );
	sameArray( h.of( 6, 'metric' ).at( - 1 ).normalized, GA.normalize( eb, { mode: 'rank' } ) );
	sameArray( h.of( 7, 'metric' ).at( - 1 ).normalized, GA.normalize( eb, { mode: 'log' } ) );
	sameArray( h.of( 8, 'metric' ).at( - 1 ).values, exact );

} );

test( 'worker: samples auto / number are one-shot runs; preview 0 sends only the final result', async () => {

	const g = randomGraph( 300, 500 );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 1, graph: 1, slot: 'a', metric: 'closeness', options: { samples: 64, seed: 3 } } );
	await h.idle( 1 );
	const r = h.of( 1, 'metric' );
	assert.equal( r.length, 1 );
	assert.equal( r[ 0 ].final, true );
	assert.equal( r[ 0 ].exact, false );
	assert.equal( r[ 0 ].pivots, 64 );
	sameArray( r[ 0 ].values, GA.closeness( g, { samples: 64, seed: 3 } ).closeness );

	// default ('auto') = the library's default pivot count: exact on a graph this small
	h.handle( { type: 'metric', id: 2, graph: 1, slot: 'a', metric: 'harmonic' } );
	await h.idle( 2 );
	assert.equal( h.of( 2, 'metric' )[ 0 ].exact, true );
	sameArray( h.of( 2, 'metric' )[ 0 ].values, GA.closeness( g ).harmonic );

} );

test( 'worker: maxMs ends the refinement with what there is; asking again resumes the parked task', async () => {

	const g = randomGraph( 400, 700 );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 1, graph: 1, metric: 'betweenness', options: { samples: 'exact' }, preview: 40, maxMs: 300 } );
	await h.idle( 1 );
	const first = h.of( 1, 'metric' ).at( - 1 );
	assert.equal( first.final, true );
	assert.equal( first.exact, false );
	assert.ok( first.pivots > 0 && first.pivots < g.N, `${first.pivots} of ${g.N}` );
	assert.ok( first.ms >= 300 && first.ms < 420, `stopped after ${first.ms} fake ms` );
	const stoppedAt = h.slices;
	await new Promise( ( r ) => setTimeout( r, 5 ) );
	assert.equal( h.slices, stoppedAt, 'the task is parked, not running, once nobody waits for it' );

	// same metric again: the first thing that comes back is a preview with at least the old pivots
	h.handle( { type: 'metric', id: 2, graph: 1, metric: 'betweenness', options: { samples: 'exact' }, preview: 40 } );
	const resumed = h.of( 2, 'metric' )[ 0 ];
	assert.ok( resumed && resumed.final === false, 'an immediate preview from the parked task' );
	assert.equal( resumed.pivots, first.pivots );
	sameArray( resumed.values, first.values );
	await h.idle( 2 );
	const done = h.of( 2, 'metric' ).at( - 1 );
	assert.equal( done.exact, true );
	sameArray( done.values, GA.betweenness( g, { samples: Infinity } ) );
	// the work was not repeated: the second request needed fewer slices than a fresh exact run
	const h2 = harness();
	h2.handle( graphMessage( g, 1 ) );
	h2.handle( { type: 'metric', id: 1, graph: 1, metric: 'betweenness', options: { samples: 'exact' } } );
	await h2.idle( 1 );
	assert.ok( h.slices - stoppedAt < h2.slices, `${h.slices - stoppedAt} < ${h2.slices}` );

} );

test( 'worker: the newest request of a slot wins; other slots are untouched', async () => {

	const g = randomGraph( 300, 500 );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 1, graph: 1, slot: 'color', metric: 'betweenness', options: { samples: 'exact' }, preview: 40 } );
	h.handle( { type: 'metric', id: 2, graph: 1, slot: 'glow', metric: 'closeness', options: { samples: 'exact' }, preview: 40 } );
	h.between = ( n ) => {

		// while the worker yields after its 6th slice, a newer request arrives for slot 'color'
		if ( n === 6 ) h.handle( { type: 'metric', id: 3, graph: 1, slot: 'color', metric: 'harmonic', options: { samples: 'exact' }, preview: 40 } );

	};

	await h.idle( 1, 2, 3 );
	const cancelled = h.of( 1, 'cancelled' );
	assert.equal( cancelled.length, 1 );
	assert.equal( cancelled[ 0 ].reason, 'superseded' );
	const at = h.out.indexOf( cancelled[ 0 ] );
	assert.ok( h.out.slice( at + 1 ).every( ( m ) => m.id !== 1 ), 'no message of the superseded job after its cancellation' );
	assert.ok( h.of( 1, 'metric' ).every( ( m ) => ! m.final ) );
	assert.ok( h.out.findIndex( ( m ) => m.id === 3 ) > at, 'the new job starts after the old one was cancelled' );

	// both survivors finish exactly; closeness and harmonic came from one shared task
	const c = GA.closeness( g, { samples: Infinity } );
	sameArray( h.of( 2, 'metric' ).at( - 1 ).values, c.closeness );
	sameArray( h.of( 3, 'metric' ).at( - 1 ).values, c.harmonic );
	assert.ok( h.of( 2, 'metric' ).at( - 1 ).exact && h.of( 3, 'metric' ).at( - 1 ).exact );
	// harmonic joined the running closeness task: its first message already has pivots
	assert.ok( h.of( 3, 'metric' )[ 0 ].pivots > 0 && ! h.of( 3, 'metric' )[ 0 ].final );

	// a cheap request in a third slot is answered inside the handler even while slow jobs run
	const h3 = harness();
	h3.handle( graphMessage( g, 1 ) );
	h3.handle( { type: 'metric', id: 1, graph: 1, slot: 'color', metric: 'betweenness', options: { samples: 'exact' } } );
	h3.between = ( n ) => {

		if ( n !== 2 ) return;
		const count = h3.out.length;
		h3.handle( { type: 'metric', id: 2, graph: 1, slot: 'size', metric: 'degree' } );
		assert.equal( h3.out.length, count + 1 );
		assert.equal( h3.out.at( - 1 ).id, 2 );
		assert.equal( h3.out.at( - 1 ).final, true );

	};

	await h3.idle( 1, 2 );
	assert.equal( h3.of( 1, 'metric' ).at( - 1 ).exact, true, 'the slow job was not disturbed' );

} );

test( 'worker: cancel by id, by slot and all; a cancelled job posts nothing more', async () => {

	const g = randomGraph( 300, 500 );
	for ( const cancel of [ { type: 'cancel', id: 1 }, { type: 'cancel', slot: 'x' }, { type: 'cancel' } ] ) {

		const h = harness();
		h.handle( graphMessage( g, 1 ) );
		h.handle( { type: 'metric', id: 1, graph: 1, slot: 'x', metric: 'betweenness', options: { samples: 'exact' }, preview: 40 } );
		h.handle( { type: 'metric', id: 2, graph: 1, slot: 'y', metric: 'closeness', options: { samples: 'exact' }, preview: 40 } );
		h.between = ( n ) => {

			if ( n === 5 ) h.handle( cancel );

		};

		await h.idle( 1, 2 );
		const c = h.of( 1, 'cancelled' );
		assert.equal( c.length, 1 );
		assert.equal( c[ 0 ].reason, 'cancel' );
		assert.ok( h.out.slice( h.out.indexOf( c[ 0 ] ) + 1 ).every( ( m ) => m.id !== 1 ) );
		assert.ok( h.of( 1, 'metric' ).length >= 1 && h.of( 1, 'metric' ).every( ( m ) => ! m.final ), 'it had previews, no final' );
		if ( cancel.id === undefined && cancel.slot === undefined ) assert.equal( h.of( 2, 'cancelled' ).length, 1 );
		else assert.equal( h.of( 2, 'metric' ).at( - 1 ).exact, true, 'the other slot ran to the end' );
		const after = h.slices;
		await new Promise( ( r ) => setTimeout( r, 5 ) );
		assert.equal( h.slices, after, 'the pump stops when no job is left' );

	}

	// cancelling something unknown is harmless
	const h = harness();
	h.handle( { type: 'cancel', id: 99 } );
	h.handle( { type: 'cancel', slot: 'nope' } );
	h.handle( { type: 'cancel' } );
	assert.deepEqual( h.out, [] );

} );

test( 'worker: a new graph cancels running jobs, drops caches, and stale requests are refused', async () => {

	const a = randomGraph( 300, 500, 1 ), b = randomGraph( 80, 60, 2 );
	const h = harness();
	h.handle( { type: 'metric', id: 1, graph: 'A', metric: 'degree' } );
	assert.deepEqual( h.of( 1 ), [ { type: 'cancelled', id: 1, reason: 'graph' } ], 'no graph yet' );

	h.handle( graphMessage( a, 'A' ) );
	h.handle( { type: 'metric', id: 2, graph: 'A', slot: 'color', metric: 'betweenness', options: { samples: 'exact' }, preview: 40 } );
	h.handle( { type: 'summary', id: 3, graph: 'A', slot: 'summary' } );
	h.between = ( n ) => {

		if ( n !== 4 ) return;
		h.handle( graphMessage( b, 'B' ) );
		h.handle( { type: 'metric', id: 4, graph: 'A', slot: 'late', metric: 'degree' } ); // was meant for A
		h.handle( { type: 'metric', id: 5, graph: 'B', slot: 'size', metric: 'degree' } );
		h.handle( { type: 'metric', id: 6, graph: 'B', slot: 'color', metric: 'betweenness', options: { samples: 'exact' }, preview: 40 } );

	};

	await h.idle( 2, 3, 4, 5, 6 );
	assert.deepEqual( h.of( 2, 'cancelled' ).map( ( m ) => m.reason ), [ 'graph' ] );
	assert.deepEqual( h.of( 3, 'cancelled' ).map( ( m ) => m.reason ), [ 'graph' ] );
	assert.ok( h.of( 3, 'summary' ).every( ( m ) => ! m.final ) );
	assert.deepEqual( h.of( 4 ), [ { type: 'cancelled', id: 4, reason: 'graph' } ] );
	const at = h.out.findIndex( ( m ) => m.id === 3 && m.type === 'cancelled' );
	assert.equal( h.out[ at - 1 ].id, 2, 'both are cancelled together, when the graph message is handled' );
	assert.deepEqual( h.out.slice( at + 1 ).filter( ( m ) => m.id === 2 || m.id === 3 ), [], 'nothing from the old graph afterwards' );
	sameArray( h.of( 5, 'metric' )[ 0 ].values, b.degree );
	const final = h.of( 6, 'metric' ).at( - 1 );
	assert.equal( final.values.length, b.N );
	sameArray( final.values, GA.betweenness( b, { samples: Infinity } ) );

	// reset: no graph at all
	h.handle( { type: 'reset' } );
	h.handle( { type: 'metric', id: 7, graph: 'B', metric: 'degree' } );
	assert.deepEqual( h.of( 7 ), [ { type: 'cancelled', id: 7, reason: 'graph' } ] );

} );

test( 'worker: errors are reported per request; junk messages are ignored', async () => {

	const g = randomGraph( 20, 10 );
	const h = harness();
	h.handle( null );
	h.handle( 'hello' );
	h.handle( { type: 'unknown' } );
	assert.deepEqual( h.out, [] );
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'metric', id: 1, graph: 1, metric: 'nope' } );
	assert.equal( h.of( 1 )[ 0 ].type, 'error' );
	assert.match( h.of( 1 )[ 0 ].message, /Unknown node metric "nope"/ );
	h.handle( { type: 'metric', id: 2, graph: 1, target: 'edge', metric: 'pagerank' } );
	assert.match( h.of( 2 )[ 0 ].message, /Unknown edge metric/ );
	// the slot is free again after an error
	h.handle( { type: 'metric', id: 3, graph: 1, metric: 'degree' } );
	assert.equal( h.of( 3 )[ 0 ].final, true );
	assert.equal( h.out.filter( ( m ) => m.type === 'cancelled' ).length, 0 );

} );

test( 'worker: graphs with 0, 1 and 2 nodes', async () => {

	for ( const g of [ GA.graphFromEdges( 0, [] ), GA.graphFromEdges( 1, [] ), GA.graphFromEdges( 2, [ 0, 1 ] ), GA.graphFromEdges( 3, [] ) ] ) {

		const h = harness();
		h.handle( graphMessage( g, 1 ) );
		let id = 0;
		for ( const m of NODE_METRICS ) h.handle( { type: 'metric', id: ++ id, graph: 1, slot: m.id, metric: m.id, options: { samples: 'exact', sources: [ 0 ] }, preview: 10 } );
		for ( const m of EDGE_METRICS ) h.handle( { type: 'metric', id: ++ id, graph: 1, slot: 'e' + m.id, target: 'edge', metric: m.id, normalize: true } );
		h.handle( { type: 'summary', id: ++ id, graph: 1, slot: 'sum' } );
		await h.idle( ...Array.from( { length: id }, ( _, k ) => k + 1 ) );
		assert.equal( h.out.filter( ( m ) => m.type === 'error' || m.type === 'cancelled' ).length, 0, `N=${g.N}: ${JSON.stringify( h.out.filter( ( m ) => m.type === 'error' ) )}` );
		for ( const m of h.out.filter( ( x ) => x.type === 'metric' ) ) {

			assert.equal( m.values.length, m.target === 'edge' ? g.E : g.N );
			assert.ok( m.values.every( Number.isFinite ) );

		}

		const s = h.out.filter( ( m ) => m.type === 'summary' ).at( - 1 );
		assert.equal( s.final, true );
		assert.equal( s.summary.nodes, g.N );
		assert.equal( s.summary.edges, g.E );
		assert.equal( s.summary.components.count, g.N === 2 ? 1 : g.N );

	}

	// components of size one are counted: a triangle, a pair and three isolated nodes
	const parts = GA.graphFromEdges( 8, [ 0, 1, 1, 2, 2, 0, 3, 4 ] );
	const h = harness();
	h.handle( graphMessage( parts, 1 ) );
	h.handle( { type: 'summary', id: 1, graph: 1 } );
	await h.idle( 1 );
	const c = h.of( 1, 'summary' ).at( - 1 ).summary.components;
	assert.deepEqual( c, { count: 5, largest: 3, largestFraction: 3 / 8, singletons: 3, sizes: [ 3, 2, 1, 1, 1 ] } );

} );

test( 'worker: summary grows over four stages and matches the library', async () => {

	const text = readFileSync( 'public/data/imsv/lesmis.json', 'utf8' );
	const g = prepareGraph( parseGraphJson( JSON.parse( text ), 'lesmis' ) );
	const h = harness();
	h.handle( graphMessage( g, 1 ) );
	h.handle( { type: 'summary', id: 1, graph: 1, options: { top: 5 } } );
	await h.idle( 1 );
	const stages = h.of( 1, 'summary' );
	assert.deepEqual( stages.map( ( m ) => m.summary.stage ), [ 'basic', 'centrality', 'distance', 'done' ] );
	assert.deepEqual( stages.map( ( m ) => m.final ), [ false, false, false, true ] );
	assert.deepEqual( h.of( 1, 'progress' ).map( ( p ) => p.done ), [ 1, 2, 3, 4 ] );
	// earlier stages are snapshots: they do not gain fields later
	assert.equal( stages[ 0 ].summary.distance, null );
	assert.deepEqual( Object.keys( stages[ 0 ].summary.top ), [ 'degree' ] );
	assert.deepEqual( Object.keys( stages[ 1 ].summary.top ), [ 'degree', 'pagerank', 'eigenvector' ] );

	const s = stages.at( - 1 ).summary;
	const ref = GA.graphSummary( g );
	assert.equal( s.nodes, 77 );
	assert.equal( s.edges, 254 );
	assert.equal( s.density, ref.density );
	assert.deepEqual( { min: s.degree.min, max: s.degree.max, mean: s.degree.mean, median: s.degree.median, isolated: s.degree.isolated }, { min: ref.minDegree, max: ref.maxDegree, mean: ref.meanDegree, median: ref.medianDegree, isolated: ref.isolated } );
	sameArray( s.degree.histogram, ref.degreeHistogram );
	assert.equal( s.degree.bins.reduce( ( sum, b ) => sum + b.count, 0 ), g.N, 'the bins cover every node once' );
	for ( let k = 1; k < s.degree.bins.length; k ++ ) assert.equal( s.degree.bins[ k ].from, s.degree.bins[ k - 1 ].to + 1 );
	assert.equal( s.degree.bins.at( - 1 ).to, ref.maxDegree );
	assert.equal( s.assortativity, ref.assortativity );
	const comp = GA.connectedComponents( g );
	assert.deepEqual( s.components, { count: comp.count, largest: comp.sizes[ 0 ], largestFraction: comp.sizes[ 0 ] / g.N, singletons: 0, sizes: Array.from( comp.sizes ) } );
	const cc = GA.clusteringCoefficient( g );
	assert.deepEqual( s.clustering, { average: cc.average, transitivity: cc.global, triangles: cc.triangleCount } );
	const kc = GA.kCore( g );
	assert.equal( s.core.degeneracy, kc.degeneracy );
	assert.equal( s.core.sizes.reduce( ( a, b ) => a + b, 0 ), g.N );
	assert.equal( s.core.sizes[ kc.degeneracy ], kc.core.filter( ( c ) => c === kc.degeneracy ).length );

	// distances: exact at this size
	const apl = GA.averagePathLength( g, { samples: Infinity } );
	assert.equal( s.distance.exact, true );
	assert.ok( Math.abs( s.distance.averagePathLength - apl.averagePathLength ) < 1e-9 );
	assert.ok( Math.abs( s.distance.effectiveDiameter - apl.effectiveDiameter ) < 1e-9 );
	sameArray( s.distance.distribution, apl.distanceDistribution );
	assert.ok( s.diameter.lower <= apl.maxDistance && apl.maxDistance <= s.diameter.upper );
	assert.equal( s.diameter.lower, 5, 'the diameter of Les Misérables is 5' );
	assert.equal( s.diameter.exact, true );
	assert.equal( s.diameter.endpoints.length, 2 );

	// top lists: largest first, equal to a sort of the library's values
	const expectTop = ( list, values, k ) => {

		const order = Array.from( values.keys() ).sort( ( x, y ) => values[ y ] - values[ x ] || x - y ).slice( 0, k );
		assert.deepEqual( list.nodes, order );
		assert.deepEqual( list.values, order.map( ( i ) => values[ i ] ) );

	};

	expectTop( s.top.degree, g.degree, 5 );
	expectTop( s.top.pagerank, GA.pageRank( g ), 5 );
	expectTop( s.top.eigenvector, GA.eigenvectorCentrality( g ), 5 );
	expectTop( s.top.betweenness, GA.betweenness( g, { samples: Infinity } ), 5 );
	expectTop( s.top.closeness, GA.closeness( g, { samples: Infinity } ).closeness, 5 );
	assert.equal( s.top.betweenness.exact, true );
	assert.equal( g.labels[ s.top.betweenness.nodes[ 0 ] ], 'Valjean' );
	assert.ok( s.timings.total >= 0 );

} );

test( 'topK: largest first, ties by index, k larger than n, NaN skipped', () => {

	assert.deepEqual( topK( [ 3, 9, 1, 9, 5 ], 3 ), { nodes: [ 1, 3, 4 ], values: [ 9, 9, 5 ] } );
	assert.deepEqual( topK( [ 2, 1 ], 10 ), { nodes: [ 0, 1 ], values: [ 2, 1 ] } );
	assert.deepEqual( topK( [ NaN, 4, NaN, 7 ], 2 ), { nodes: [ 3, 1 ], values: [ 7, 4 ] } );
	assert.deepEqual( topK( [], 3 ), { nodes: [], values: [] } );
	assert.deepEqual( topK( [ 1, 2 ], 0 ), { nodes: [], values: [] } );

} );

// ── client ───────────────────────────────────────────────────────────────────────────────────

test( 'AnalysisClient: promises, progress and previews; the graph is sent once', async () => {

	const g = randomGraph( 240, 400 );
	g.labels = Array.from( { length: g.N }, ( _, i ) => 'n' + i );
	let worker = null;
	const client = new AnalysisClient( { createWorker: () => ( worker = new FakeWorker() ) } );
	assert.deepEqual( await client.metric( 'degree' ), { cancelled: true, reason: 'no-graph' } );
	client.setGraph( g );
	assert.equal( worker, null, 'no worker before the first request' );

	const r = await client.metric( 'pagerank', { normalize: true } );
	assert.equal( r.cancelled, false );
	assert.equal( r.metric, 'pagerank' );
	assert.equal( r.final, true );
	assert.equal( r.scale, 'log' );
	sameArray( r.values, GA.pageRank( g ) );
	sameArray( r.normalized, GA.normalize( r.values, { mode: 'log' } ) );

	const progress = [], previews = [];
	const b = await client.metric( 'betweenness', { slot: 'color', options: { samples: 'exact' }, preview: 40, onProgress: ( p ) => progress.push( p ), onPreview: ( p ) => previews.push( p ) } );
	assert.equal( b.exact, true );
	sameArray( b.values, GA.betweenness( g, { samples: Infinity } ) );
	assert.ok( previews.length >= 3 );
	assert.ok( previews.every( ( p ) => p.cancelled === false && p.final === false && p.values.length === g.N ) );
	assert.ok( progress.length > 3 && progress.at( - 1 ).fraction === 1 );
	for ( let k = 1; k < progress.length; k ++ ) assert.ok( progress[ k ].fraction >= progress[ k - 1 ].fraction );

	const e = await client.edgeMetric( 'triangles', { normalize: 'rank' } );
	assert.equal( e.target, 'edge' );
	assert.equal( e.values.length, g.E );
	assert.ok( e.normalized.every( ( v ) => v >= 0 && v <= 1 ) );

	assert.equal( worker.posted.filter( ( m ) => m.type === 'graph' ).length, 1, 'one graph transfer for all requests' );
	assert.equal( client.busy, false );
	assert.equal( client.pending, 0 );
	client.dispose();

} );

test( 'AnalysisClient: newest request of a slot wins; requests while busy; cancel', async () => {

	const g = randomGraph( 300, 500 );
	const client = new AnalysisClient( { createWorker: () => new FakeWorker() } );
	client.setGraph( g );
	const previews = [];
	const first = client.metric( 'betweenness', { slot: 'color', options: { samples: 'exact' }, preview: 40, onPreview: ( p ) => previews.push( p ) } );
	// several requests while the worker has not even seen the first one
	const second = client.metric( 'closeness', { slot: 'color', options: { samples: 'exact' } } );
	const third = client.metric( 'core', { slot: 'size' } );
	const fourth = client.metric( 'harmonic', { slot: 'glow', options: { samples: 'exact' } } );
	assert.equal( client.pending, 3 );
	assert.deepEqual( await first, { cancelled: true, reason: 'superseded' } );
	const [ c, k, hm ] = await Promise.all( [ second, third, fourth ] );
	const ref = GA.closeness( g, { samples: Infinity } );
	sameArray( c.values, ref.closeness );
	sameArray( hm.values, ref.harmonic );
	sameArray( k.values, GA.kCore( g ).core );
	assert.equal( previews.length, 0, 'no preview of a superseded request is delivered' );

	// cancel a slot mid-flight: the promise resolves as cancelled, late replies are dropped
	const got = [];
	const p = client.metric( 'betweenness', { slot: 'x', options: { samples: 'exact', seed: 5 }, preview: 20, onPreview: ( r ) => {

		got.push( r );
		if ( got.length === 2 ) client.cancel( 'x' );

	} } );
	assert.deepEqual( await p, { cancelled: true, reason: 'cancel' } );
	const count = got.length;
	await new Promise( ( r ) => setTimeout( r, 20 ) );
	assert.equal( got.length, count );

	// cancel everything
	const q = client.metric( 'betweenness', { slot: 'y', options: { samples: 'exact', seed: 6 } } );
	const s = client.summary();
	client.cancel();
	assert.deepEqual( [ await q, await s ], [ { cancelled: true, reason: 'cancel' }, { cancelled: true, reason: 'cancel' } ] );
	assert.equal( client.busy, false );
	// the client still works afterwards
	sameArray( ( await client.metric( 'degree' ) ).values, g.degree );
	client.dispose();
	assert.deepEqual( await client.metric( 'degree' ), { cancelled: true, reason: 'no-graph' } );

} );

test( 'AnalysisClient: a graph change mid-request cancels it and later requests use the new graph', async () => {

	const a = randomGraph( 300, 500, 1 ), b = randomGraph( 70, 50, 2 );
	let worker = null;
	const client = new AnalysisClient( { createWorker: () => ( worker = new FakeWorker() ) } );
	client.setGraph( a );
	let switched = false;
	const late = [];
	const p = client.metric( 'betweenness', { slot: 'color', options: { samples: 'exact' }, preview: 20, onPreview: ( r ) => {

		late.push( r.values.length );
		if ( ! switched ) {

			switched = true;
			client.setGraph( b );

		}

	} } );
	assert.deepEqual( await p, { cancelled: true, reason: 'graph' } );
	const r = await client.metric( 'betweenness', { slot: 'color', options: { samples: 'exact' } } );
	assert.equal( r.values.length, b.N );
	sameArray( r.values, GA.betweenness( b, { samples: Infinity } ) );
	assert.deepEqual( late, [ a.N ], 'no reply for the old graph reaches the callbacks after the switch' );
	assert.equal( worker.posted.filter( ( m ) => m.type === 'graph' ).length, 2 );
	assert.equal( worker.posted.filter( ( m ) => m.type === 'reset' ).length, 1 );

	// two graph changes in a row without a request in between: only the last graph is sent
	client.setGraph( a );
	client.setGraph( b );
	await client.metric( 'degree' );
	assert.equal( worker.posted.filter( ( m ) => m.type === 'graph' ).length, 3 );
	client.setGraph( null );
	assert.deepEqual( await client.metric( 'degree' ), { cancelled: true, reason: 'no-graph' } );
	client.dispose();

} );

test( 'AnalysisClient: summary carries labels; errors reject; a crashed worker is replaced', async () => {

	const text = readFileSync( 'public/data/imsv/karate.json', 'utf8' );
	const g = prepareGraph( parseGraphJson( JSON.parse( text ), 'karate' ) );
	const workers = [];
	const client = new AnalysisClient( { createWorker: () => {

		const w = new FakeWorker();
		workers.push( w );
		return w;

	} } );
	client.setGraph( g );
	const partial = [], progress = [];
	const s = await client.summary( { top: 3, onPreview: ( p ) => partial.push( p ), onProgress: ( p ) => progress.push( p.done ) } );
	assert.equal( s.cancelled, false );
	assert.equal( s.final, true );
	assert.equal( s.stage, 'done' );
	assert.equal( s.nodes, 34 );
	assert.equal( s.edges, 78 );
	assert.deepEqual( partial.map( ( p ) => p.stage ), [ 'basic', 'centrality', 'distance' ] );
	assert.deepEqual( progress, [ 1, 2, 3, 4 ] );
	for ( const key of [ 'degree', 'pagerank', 'eigenvector', 'closeness', 'betweenness' ] ) {

		assert.equal( s.top[ key ].nodes.length, 3, key );
		assert.deepEqual( s.top[ key ].labels, s.top[ key ].nodes.map( ( i ) => String( g.labels[ i ] ) ), key );

	}

	assert.equal( s.diameter.lower, 5 );
	assert.equal( s.diameter.endpointLabels.length, 2 );
	assert.equal( partial[ 0 ].top.degree.labels.length, 3 );

	await assert.rejects( client.metric( 'nope' ), /Unknown node metric/ );

	// the worker dies while a request is in flight
	const p = client.metric( 'betweenness', { options: { samples: 'exact' } } );
	workers[ 0 ].onerror( { message: 'boom' } );
	await assert.rejects( p, /boom/ );
	const r = await client.metric( 'degree' );
	sameArray( r.values, g.degree );
	assert.equal( workers.length, 2, 'a fresh worker was created' );
	assert.equal( workers[ 1 ].posted[ 0 ].type, 'graph', 'and it got the graph again' );
	client.dispose();

} );

test( 'AnalysisClient: inline mode (no Worker) runs the same handler on this thread', async () => {

	const g = randomGraph( 120, 200 );
	const client = new AnalysisClient( { inline: true } );
	client.setGraph( g );
	const [ d, b ] = await Promise.all( [ client.metric( 'degree', { slot: 'a' } ), client.metric( 'betweenness', { slot: 'b' } ) ] );
	sameArray( d.values, g.degree );
	sameArray( b.values, GA.betweenness( g ) );
	// Node has no Worker global: the default constructor falls back to inline as well
	const fallback = new AnalysisClient();
	fallback.setGraph( g );
	sameArray( ( await fallback.metric( 'core' ) ).values, GA.kCore( g ).core );
	client.dispose();
	fallback.dispose();

} );
