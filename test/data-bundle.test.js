import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runBundle, bundleBudget, BUNDLE_BUDGET_MS, BUNDLE_HURRY_AFTER } from '../src/data/bundle.js';
import { DataClient } from '../src/data/client.js';
import { createPort, runRequest } from '../src/data/worker.js';
import { edgePathBundling, mixedPaths, hierarchicalPaths, validatePaths, EDGE_BUNDLED } from '../src/graph/bundling.js';
import { parseGraphJson } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';

const prepared = ( id ) => prepareGraph( parseGraphJson( JSON.parse( readFileSync( `public/data/${id}.json`, 'utf8' ) ), id ) );
const tick = () => new Promise( ( r ) => setImmediate( r ) );
const until = async ( test, turns = 5000 ) => {

	for ( let k = 0; k < turns && ! test(); k ++ ) await tick();
	assert.ok( test(), 'condition reached' );

};

// (never deepEqual a bundle result: the diff of a failure prints megabytes of typed arrays)
const wasCancelled = ( r ) => {

	assert.equal( r.cancelled, true );
	assert.deepEqual( Object.keys( r ), [ 'cancelled' ] );

};

function fakeWorkers() {

	const workers = [];
	class FakeWorker {

		constructor() {

			this.terminated = false;
			this.sent = [];
			workers.push( this );

		}

		postMessage( msg ) {

			this.sent.push( msg );

		}

		terminate() {

			this.terminated = true;

		}

		reply( data ) {

			this.onmessage( { data } );

		}

	}

	return { workers, create: () => new FakeWorker() };

}

// ── the cooperative driver ───────────────────────────────────────────────────────────────────

test( 'runBundle paths: the same result as edgePathBundling, sliced; normalised load; stats', async () => {

	const g = prepared( 'generated/hsbm-500' );
	const direct = edgePathBundling( g, g.initPos );
	const progress = [];
	let yields = 0;
	const r = await runBundle( g, g.initPos, { mode: 'paths', slice: 0, onProgress: ( p ) => progress.push( p ), yieldTask: () => ( yields ++, tick() ) } );
	assert.equal( r.cancelled, false );
	assert.ok( yields > 20, `${yields} turns of the event loop` );
	assert.equal( r.timings.slices, yields + 1 );
	assert.deepEqual( Array.from( r.paths.pathStart ), Array.from( direct.pathStart ) );
	assert.deepEqual( Array.from( r.paths.pathRefs ), Array.from( direct.pathRefs ) );
	assert.equal( r.paths.maxInterior, direct.maxInterior );
	assert.ok( r.paths.pathStart instanceof Uint32Array && r.paths.pathStart.length === g.E + 1 );
	assert.ok( r.paths.pathRefs instanceof Uint32Array );
	assert.deepEqual( Object.keys( r.paths ), [ 'pathStart', 'pathRefs', 'maxInterior' ] );
	assert.deepEqual( validatePaths( g, r.paths, { realPaths: true, positions: g.initPos, maxDistortion: 2, maxInterior: 8 } ).errors, [] );

	// load: per edge, 0..1, 1 on the busiest edge, 0 exactly where nothing is routed
	assert.ok( r.load instanceof Float32Array && r.load.length === g.E );
	let max = 0;
	for ( let e = 0; e < g.E; e ++ ) if ( direct.load[ e ] > max ) max = direct.load[ e ];
	assert.equal( r.stats.maxLoad, max );
	for ( let e = 0; e < g.E; e ++ ) {

		assert.ok( Math.abs( r.load[ e ] - direct.load[ e ] / max ) < 1e-6 );
		if ( direct.state[ e ] === EDGE_BUNDLED ) assert.equal( r.load[ e ], 0 );

	}

	assert.equal( Math.max( ...r.load ), 1 );

	assert.equal( r.stats.mode, 'paths' );
	assert.equal( r.stats.method, 'spanner' );
	assert.equal( r.stats.complete, true );
	assert.equal( r.stats.outOfTime, false );
	assert.equal( r.stats.budget, BUNDLE_BUDGET_MS, 'the budget is there at every size; a run that is done sooner is not touched by it' );
	assert.equal( r.stats.hurried, false );
	assert.equal( r.stats.bundled, direct.stats.bundled );
	assert.ok( r.stats.bundledRatio > 0.4 );
	for ( const key of [ 'setup', 'compute', 'result', 'total' ] ) assert.ok( r.timings[ key ] >= 0, key );
	// progress: both phases, fractions rising to 1
	assert.ok( progress.length >= 2 );
	assert.equal( progress[ 0 ].fraction, 0 );
	assert.equal( progress.at( - 1 ).fraction, 1 );
	for ( let k = 1; k < progress.length; k ++ ) assert.ok( progress[ k ].fraction >= progress[ k - 1 ].fraction );
	assert.ok( progress.every( ( p ) => p.phase === 'spanner' || p.phase === 'route' ) );
	assert.ok( progress.every( ( p ) => p.completed <= p.total ) );

} );

test( 'runBundle mixed: edge paths, the hierarchy for the edges they left free (the recipe)', async () => {

	const g = prepared( 'imsv/lesmis' );
	const epb = edgePathBundling( g, g.initPos );
	const expected = mixedPaths( g, epb, hierarchicalPaths( g ), { only: 'free' } );
	const r = await runBundle( g, g.initPos, { mode: 'mixed' } );
	assert.deepEqual( Array.from( r.paths.pathStart ), Array.from( expected.pathStart ) );
	assert.deepEqual( Array.from( r.paths.pathRefs ), Array.from( expected.pathRefs ) );
	assert.equal( r.paths.maxInterior, expected.maxInterior );
	assert.deepEqual( validatePaths( g, r.paths ).errors, [] );
	const counts = [ 0, 0, 0 ];
	for ( const s of expected.source ) counts[ s ] ++;
	assert.deepEqual( r.stats.mixed, { fromPaths: counts[ 1 ], fromHierarchy: counts[ 2 ], straight: counts[ 0 ] } );
	assert.ok( counts[ 1 ] > 0 && counts[ 2 ] > 0 );
	assert.equal( r.stats.mode, 'mixed' );
	// cluster refs appear only in mixed mode
	assert.ok( Array.from( r.paths.pathRefs ).some( ( ref ) => ref >= g.N ) );
	const paths = await runBundle( g, g.initPos, { mode: 'paths' } );
	assert.ok( Array.from( paths.paths.pathRefs ).every( ( ref ) => ref < g.N ) );
	// only: 'straight' gives every edge without a path the hierarchy polygon
	const all = await runBundle( g, g.initPos, { mode: 'mixed', options: { only: 'straight' } } );
	assert.equal( all.stats.mixed.straight, mixedPaths( g, epb, hierarchicalPaths( g ), { only: 'straight' } ).source.filter( ( s ) => s === 0 ).length );
	// a graph without a hierarchy: mixed falls back to the edge paths
	const flat = { N: g.N, E: g.E, edges: g.edges };
	const f = await runBundle( flat, g.initPos, { mode: 'mixed' } );
	assert.deepEqual( Array.from( f.paths.pathRefs ), Array.from( epb.pathRefs ) );
	// options reach the algorithm
	const loose = await runBundle( g, g.initPos, { mode: 'paths', options: { maxDistortion: 3, method: 'epb' } } );
	assert.equal( loose.stats.method, 'epb' );
	assert.notDeepEqual( Array.from( loose.paths.pathRefs ), Array.from( paths.paths.pathRefs ) );

} );

test( 'runBundle: cancel between slices; time budget; bad input', async () => {

	const g = prepared( 'generated/lfr-2k' );
	let polls = 0;
	const cancelled = await runBundle( g, g.initPos, { slice: 0, isCancelled: () => ++ polls >= 3 } );
	wasCancelled( cancelled );
	assert.equal( polls, 3 );

	// out of time: a valid, incomplete result
	const cut = await runBundle( g, g.initPos, { options: { timeBudget: 5 } } );
	assert.equal( cut.cancelled, false );
	assert.equal( cut.stats.outOfTime, true );
	assert.equal( cut.stats.budget, 5 );
	assert.equal( cut.stats.complete, false );
	assert.deepEqual( validatePaths( g, cut.paths, { realPaths: true, positions: g.initPos, maxDistortion: 2, maxInterior: 8 } ).errors, [] );
	const whole = await runBundle( g, g.initPos, { options: { timeBudget: 0 } } ); // 0 = none
	assert.equal( whole.stats.complete, true );
	assert.ok( whole.stats.bundled >= cut.stats.bundled );
	assert.equal( whole.stats.budget, null );
	assert.equal( BUNDLE_BUDGET_MS, 3000 );

	await assert.rejects( runBundle( g, g.initPos, { mode: 'wavy' } ), /unknown mode "wavy" \(paths \| mixed\)/ );
	await assert.rejects( runBundle( g, new Float32Array( 8 ) ), /positions needs 4 floats per node \(8000\), got 8/ );
	await assert.rejects( runBundle( { N: 3 }, new Float32Array( 12 ) ), /graph needs N, E and edges/ );
	await assert.rejects( runBundle( g, null ), /positions/ );

} );

test( 'runBundle: the 3 s budget holds at every size (a slow machine waits no longer for 60,000 edges)', async () => {

	assert.equal( BUNDLE_BUDGET_MS, 3000 );
	assert.ok( BUNDLE_HURRY_AFTER > 0 && BUNDLE_HURRY_AFTER <= 0.25 );
	for ( const edgeCount of [ 0, 3, 5000, 60000, 99999, 100000, 1e6 ] ) assert.equal( bundleBudget( edgeCount ), 3000, `${edgeCount} edges` );
	assert.equal( bundleBudget( 100000, 'auto' ), 3000 );
	assert.equal( bundleBudget( 100000, null ), 3000 );
	assert.equal( bundleBudget( 100000, 0 ), Infinity, '0 switches the budget off' );
	assert.equal( bundleBudget( 100000, false ), Infinity );
	assert.equal( bundleBudget( 500, 250 ), 250 );
	assert.equal( bundleBudget( 1e6, 250 ), 250 );

	// a random graph with 100,000 edges between 12,000 nodes in a ball: far more work than the
	// 120 ms this test allows it
	const N = 12000, E = 100000;
	let seed = 12345;
	const rnd = () => ( ( seed = ( Math.imul( seed, 1664525 ) + 1013904223 ) >>> 0 ) / 4294967296 );
	const positions = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) positions.set( [ rnd() * 1000, rnd() * 1000, rnd() * 1000, 0 ], 4 * i );
	const seen = new Set();
	const edges = new Uint32Array( 2 * E );
	for ( let e = 0; e < E; ) {

		const a = Math.floor( rnd() * N ), b = Math.floor( rnd() * N );
		const key = a < b ? a * N + b : b * N + a;
		if ( a === b || seen.has( key ) ) continue;
		seen.add( key );
		edges[ 2 * e ] = a;
		edges[ 2 * e + 1 ] = b;
		e ++;

	}

	const g = { N, E, edges };
	// stopped at the first poll: no result, whatever the budget
	let polls = 0;
	const stopped = await runBundle( g, positions, { isCancelled: () => ++ polls > 0 } );
	assert.equal( stopped.cancelled, true );
	assert.equal( polls, 1 );
	// cut by the clock: a valid result for every edge, flagged as out of time or as hurried
	const t0 = performance.now();
	const cut = await runBundle( g, positions, { options: { timeBudget: 120 } } );
	const ms = performance.now() - t0;
	assert.equal( cut.cancelled, false );
	assert.equal( cut.stats.budget, 120 );
	assert.ok( cut.stats.outOfTime || cut.stats.hurried, 'the run was cut short' );
	assert.equal( cut.stats.complete, ! cut.stats.outOfTime );
	assert.equal( cut.load.length, E );
	assert.equal( cut.paths.pathStart.length, E + 1 );
	assert.equal( cut.paths.pathStart[ E ], cut.paths.pathRefs.length );
	assert.ok( ms < 5000, `a 120 ms budget ends the run soon (${ms.toFixed( 0 )} ms)` );
	// below 100,000 edges the clock cuts as well: 60,000 of these edges in 120 ms
	const part = { N, E: 60000, edges: edges.subarray( 0, 120000 ) };
	const t1 = performance.now();
	const less = await runBundle( part, positions, { options: { timeBudget: 120 } } );
	assert.ok( less.stats.outOfTime || less.stats.hurried, '60,000 edges: the run was cut short' );
	assert.equal( less.paths.pathStart.length, 60001 );
	assert.ok( performance.now() - t1 < 5000 );
	// a small graph has the budget too and never notices
	const small = await runBundle( { N: 4, E: 3, edges: Uint32Array.of( 0, 1, 1, 2, 2, 3 ) }, new Float32Array( 16 ) );
	assert.equal( small.stats.budget, 3000 );
	assert.equal( small.stats.outOfTime, false );
	assert.equal( small.stats.hurried, false );
	assert.equal( small.stats.complete, true );

} );

test( 'bundle protocol: graph arrays once per worker, newest wins, cancel, supersede', async () => {

	const { workers, create } = fakeWorkers();
	const client = new DataClient( { createWorker: create } );
	await assert.rejects( client.bundle( { positions: new Float32Array( 4 ) } ), /no graph/ );
	const load = client.loadText( '1 2\n2 3\n', 'a.txt' );
	const w = workers[ 0 ];
	const graph = { N: 3, E: 2, D: 1, edges: Uint32Array.of( 0, 1, 1, 2 ), edgeLca: Uint8Array.of( 0, 0 ), anc: Uint32Array.of( 0, 0, 0 ), colors: new Float32Array( 12 ) };
	w.reply( { id: w.sent[ 0 ].id, type: 'result', graph, meta: {}, timings: {} } );
	await load;

	await assert.rejects( client.bundle( { positions: new Float32Array( 4 ), mode: 'paths' } ), /positions must be a Float32Array\( 4N \) \(12 floats, got 4\)/ );
	// positions of a larger graph (a readback from before a load) are not this graph's either
	await assert.rejects( client.bundle( { positions: new Float32Array( 16 ), mode: 'paths' } ), /\(12 floats, got 16\)/ );
	await assert.rejects( client.bundle( { mode: 'paths' } ), /\(12 floats\)/ );
	await assert.rejects( client.bundle( { positions: new Float32Array( 12 ), mode: 'loops' } ), /unknown mode "loops"/ );

	const positions = new Float32Array( 12 );
	const seen = [];
	const first = client.bundle( { positions, mode: 'mixed', options: { maxDistortion: 3 }, onProgress: ( p ) => seen.push( p ) } );
	const m1 = w.sent[ 1 ];
	assert.equal( m1.type, 'bundle' );
	assert.equal( m1.mode, 'mixed' );
	assert.deepEqual( m1.options, { maxDistortion: 3 } );
	assert.equal( m1.positions, positions );
	// what bundling needs of the prepared graph, and nothing else
	assert.deepEqual( Object.keys( m1.graph ), [ 'N', 'E', 'D', 'edges', 'edgeLca', 'anc' ] );
	assert.equal( m1.graph.edges, graph.edges );
	w.reply( { id: m1.id, type: 'bundle-progress', phase: 'spanner', fraction: 0.25, completed: 1, total: 4 } );
	assert.deepEqual( seen, [ { phase: 'spanner', fraction: 0.25, completed: 1, total: 4 } ] );

	// newest wins: the first promise resolves cancelled at once; the worker is told nothing extra
	// (it ends the older job when the new message arrives)
	const second = client.bundle( { positions, mode: 'paths' } );
	wasCancelled( await first );
	const m2 = w.sent[ 2 ];
	assert.equal( m2.type, 'bundle' );
	assert.equal( m2.graphId, m1.graphId );
	assert.equal( m2.graph, undefined, 'the worker already holds this graph' );
	assert.deepEqual( m2.options, {} );
	// late replies of the first request are ignored
	w.reply( { id: m1.id, type: 'bundle-progress', phase: 'route', fraction: 0.9, completed: 3, total: 4 } );
	w.reply( { id: m1.id, type: 'bundle-result', cancelled: true } );
	assert.equal( seen.length, 1 );
	const paths = { pathStart: Uint32Array.of( 0, 0, 0 ), pathRefs: new Uint32Array( 0 ), maxInterior: 0 };
	w.reply( { id: m2.id, type: 'bundle-result', cancelled: false, paths, load: new Float32Array( 2 ), stats: { mode: 'paths' }, timings: { total: 1 } } );
	const r2 = await second;
	assert.equal( r2.cancelled, false );
	assert.equal( r2.paths, paths );
	assert.equal( r2.load.length, 2 );
	assert.deepEqual( r2.stats, { mode: 'paths' } );
	assert.ok( r2.timings.roundTrip >= 0 && r2.timings.total === 1 );

	// cancelBundle: resolves cancelled and tells the worker
	const third = client.bundle( { positions } );
	const m3 = w.sent[ 3 ];
	client.cancelBundle();
	wasCancelled( await third );
	assert.deepEqual( w.sent[ 4 ], { type: 'bundle-cancel', id: m3.id } );
	client.cancelBundle(); // nothing in flight: no message
	assert.equal( w.sent.length, 5 );

	// the worker reports a cancel (a layout request ended the job): resolves cancelled
	const fourth = client.bundle( { positions } );
	w.reply( { id: w.sent[ 5 ].id, type: 'bundle-result', cancelled: true } );
	wasCancelled( await fourth );
	// errors reject
	const fifth = client.bundle( { positions } );
	w.reply( { id: w.sent[ 6 ].id, type: 'error', message: 'bundle: boom' } );
	await assert.rejects( fifth, /bundle: boom/ );

	// another graph object (passed explicitly): its arrays are sent, under another id
	const other = { ...graph, edges: Uint32Array.of( 0, 2, 1, 2 ) };
	const sixth = client.bundle( { positions, graph: other } );
	const m7 = w.sent[ 7 ];
	assert.notEqual( m7.graphId, m1.graphId );
	assert.equal( m7.graph.edges, other.edges );
	// a load supersedes the bundle: its graph is about to be replaced
	const next = client.loadText( '7 8\n', 'b.txt' );
	wasCancelled( await sixth );
	assert.ok( ! w.terminated, 'no load was in flight: the worker is kept' );
	w.reply( { id: w.sent[ 8 ].id, type: 'result', graph, meta: {}, timings: {} } );
	await next;

	// a restarted worker gets the arrays again
	const slow = client.loadText( '9 10\n', 'c.txt' );
	const seventh = client.bundle( { positions } ); // while a load is in flight: allowed, for the last graph
	assert.equal( w.sent.at( - 1 ).type, 'bundle' );
	// the worker keeps one graph, and that was the other one: the arrays travel again
	assert.equal( w.sent.at( - 1 ).graph.edges, graph.edges );
	assert.equal( w.sent.at( - 1 ).graphId, m1.graphId );
	const newer = client.loadText( '11 12\n', 'd.txt' ); // supersedes the load: terminates the worker
	wasCancelled( await slow );
	wasCancelled( await seventh );
	assert.ok( w.terminated );
	const w2 = workers[ 1 ];
	w2.reply( { id: w2.sent[ 0 ].id, type: 'result', graph, meta: {}, timings: {} } );
	await newer;
	// a bundle asked for while a load is in flight is for the graph before it: when the load
	// delivers its graph, the bundle resolves cancelled and the worker is told to stop
	const coming = client.loadText( '13 14\n', 'e.txt' );
	const stale = client.bundle( { positions } );
	const loadMsg = w2.sent.at( - 2 ), staleMsg = w2.sent.at( - 1 );
	assert.equal( loadMsg.type, 'load' );
	assert.equal( staleMsg.type, 'bundle' );
	const fresh = { ...graph };
	w2.reply( { id: loadMsg.id, type: 'result', graph: fresh, meta: {}, timings: {} } );
	assert.equal( ( await coming ).graph, fresh );
	wasCancelled( await stale );
	assert.deepEqual( w2.sent.at( - 1 ), { type: 'bundle-cancel', id: staleMsg.id } );
	// its late result does not reach anyone
	w2.reply( { id: staleMsg.id, type: 'bundle-result', cancelled: false, paths: {}, load: new Float32Array( 2 ), stats: {}, timings: {} } );
	const eighth = client.bundle( { positions } );
	assert.deepEqual( Object.keys( w2.sent.at( - 1 ).graph ), [ 'N', 'E', 'D', 'edges', 'edgeLca', 'anc' ] );
	client.dispose();
	wasCancelled( await eighth );

} );

// ── the worker's side of it (the real port, in this thread) ──────────────────────────────────

test( 'worker port: a newer bundle, a cancel, a load or a layout ends the running bundle', async () => {

	const g = prepared( 'generated/lfr-2k' );
	const graph = { N: g.N, E: g.E, D: g.D, edges: g.edges, edgeLca: g.edgeLca, anc: g.anc };
	const replies = [];
	const port = createPort( ( m, transfer ) => replies.push( { ...m, transfer } ) );
	const results = () => replies.filter( ( m ) => m.type === 'bundle-result' );

	// two requests in one go: the older is cancelled before it starts, the newer runs
	port.receive( { id: 1, type: 'bundle', graphId: 7, graph, positions: g.initPos, mode: 'paths' } );
	port.receive( { id: 2, type: 'bundle', graphId: 7, positions: g.initPos, mode: 'mixed' } );
	await until( () => results().length === 2 );
	assert.deepEqual( results()[ 0 ], { id: 1, type: 'bundle-result', cancelled: true, transfer: undefined } );
	const done = results()[ 1 ];
	assert.equal( done.id, 2 );
	assert.equal( done.cancelled, false );
	assert.equal( done.stats.mode, 'mixed' );
	assert.equal( done.paths.pathStart.length, g.E + 1 );
	assert.equal( done.load.length, g.E );
	// the result arrays are transferred
	assert.deepEqual( done.transfer, [ done.paths.pathStart.buffer, done.paths.pathRefs.buffer, done.load.buffer ] );
	assert.ok( replies.some( ( m ) => m.type === 'bundle-progress' && m.id === 2 ) );
	assert.ok( ! replies.some( ( m ) => m.type === 'bundle-progress' && m.id === 1 ) );

	// each of these arrives while a bundle is running and ends it
	const interrupts = [
		{ type: 'bundle-cancel', id: 10 },
		{ id: 99, type: 'bundle', graphId: 7, positions: g.initPos, mode: 'paths', options: { timeBudget: 1 } },
		{ id: 98, type: 'layout', kind: 'radial', graph: { N: g.N, E: g.E, edges: g.edges, adjStart: g.adjStart, degree: g.degree, adjList: g.adjList }, options: { root: 0 } },
		{ id: 97, type: 'load', source: { kind: 'text', text: '1 2\n2 3\n', name: 't.txt' } },
	];
	for ( const interrupt of interrupts ) {

		replies.length = 0;
		port.receive( { id: 10, type: 'bundle', graphId: 7, positions: g.initPos, mode: 'paths' } );
		await until( () => replies.some( ( m ) => m.type === 'bundle-progress' ) ); // it is running
		port.receive( interrupt );
		await until( () => results().some( ( m ) => m.id === 10 ) );
		assert.equal( results().find( ( m ) => m.id === 10 ).cancelled, true, interrupt.type );
		if ( interrupt.type === 'bundle' ) await until( () => results().some( ( m ) => m.id === 99 && m.cancelled === false ) );
		if ( interrupt.type === 'layout' ) await until( () => replies.some( ( m ) => m.type === 'layout-result' && m.id === 98 ) );
		if ( interrupt.type === 'load' ) await until( () => replies.some( ( m ) => m.type === 'result' && m.id === 97 ) );

	}

	// a cancel for another id does nothing
	replies.length = 0;
	port.receive( { id: 20, type: 'bundle', graphId: 7, positions: g.initPos, mode: 'paths' } );
	port.receive( { type: 'bundle-cancel', id: 12345 } );
	await until( () => results().length === 1 );
	assert.equal( results()[ 0 ].cancelled, false );

	// errors: unknown graph, bad positions
	replies.length = 0;
	port.receive( { id: 30, type: 'bundle', graphId: 8, positions: g.initPos, mode: 'paths' } );
	port.receive( { id: 31, type: 'bundle', graphId: 7, positions: new Float32Array( 4 ), mode: 'paths' } );
	await until( () => replies.filter( ( m ) => m.type === 'error' || m.type === 'bundle-result' ).length === 2 );
	assert.equal( results().find( ( m ) => m.id === 30 ).cancelled, true ); // superseded by 31 before it ran
	assert.match( replies.find( ( m ) => m.id === 31 ).message, /positions needs 4 floats per node/ );
	replies.length = 0;
	port.receive( { id: 32, type: 'bundle', graphId: 8, positions: g.initPos, mode: 'paths' } );
	await until( () => replies.length === 1 );
	assert.match( replies[ 0 ].message, /the graph was not sent/ );

} );

test( 'DataClient.bundle end to end (inline): paths for the graph loaded last', async () => {

	const client = new DataClient( { inline: true } );
	const loaded = await client.loadText( readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ), 'lesmis.json' );
	const g = loaded.graph;
	const r = await client.bundle( { positions: g.initPos, mode: 'mixed' } );
	assert.equal( r.cancelled, false );
	assert.deepEqual( validatePaths( g, r.paths ).errors, [] );
	assert.equal( r.paths.pathStart.length, g.E + 1 );
	assert.equal( r.load.length, g.E );
	assert.ok( r.stats.bundled > 0 && r.stats.mixed.fromHierarchy > 0 );
	assert.ok( r.timings.total > 0 && r.timings.roundTrip >= r.timings.total );
	// the graph on the main thread is untouched (its arrays were copied, not transferred)
	assert.equal( g.edges.length, 2 * g.E );
	assert.equal( g.initPos.length, 4 * g.N );
	// a recluster after a bundle request: the bundle is for the old hierarchy
	const stale = client.bundle( { positions: g.initPos, mode: 'mixed' } );
	const again = client.recluster( { resolution: 2 } );
	wasCancelled( await stale );
	const g2 = ( await again ).graph;
	const fresh = await client.bundle( { positions: g2.initPos, mode: 'mixed' } );
	assert.deepEqual( validatePaths( g2, fresh.paths ).errors, [] );
	client.dispose();

	// the same through runRequest + runBundle on a file with weights (weights do not matter here)
	const w = await runRequest( { source: { kind: 'text', text: 'a b 2\nb c 1\nc a 3\nc d 1\nd e 2\ne c 1\n', name: 'w.txt' } } );
	const b = await runBundle( w.graph, w.graph.initPos, { mode: 'paths' } );
	assert.equal( b.paths.pathStart.length, w.graph.E + 1 );

} );
