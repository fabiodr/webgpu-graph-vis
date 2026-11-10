import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DataClient } from '../src/data/client.js';
import { runRequest, handleMessage, createPort, NeedSource } from '../src/data/worker.js';
import { prepareGraph } from '../src/graph/prepare.js';
import { generate } from '../src/data/generators.js';
import { applyClustering } from '../src/data/hierarchy.js';

const hsbm = readFileSync( 'public/data/generated/hsbm-500.json', 'utf8' );
const tick = () => new Promise( ( r ) => setImmediate( r ) );

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

test( 'recluster: same nodes and edges in the same order, new hierarchy, positions kept, no parse', async () => {

	const client = new DataClient( { inline: true } );
	await assert.rejects( client.recluster( { method: 'louvain' } ), /nothing loaded yet/ );
	const stages = [];
	client.onProgress( ( p ) => stages.push( p.stage ) );
	const a = await client.loadText( hsbm, 'hsbm-500.json', { palette: 'ember' } );
	assert.equal( a.graph.clustering.method, 'leiden' ); // the stored hierarchy
	assert.ok( a.timings.parse >= 0 );
	const labels = a.graph.labels.slice(), edges = Array.from( a.graph.edges ), N = a.graph.N;

	// a readback: any positions
	const positions = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) {

		positions[ 4 * i ] = Math.sin( i ) * 100;
		positions[ 4 * i + 1 ] = Math.cos( 3 * i ) * 80;
		positions[ 4 * i + 2 ] = i - N / 2;
		positions[ 4 * i + 3 ] = 7; // w is not a coordinate

	}

	stages.length = 0;
	const b = await client.recluster( { method: 'louvain', resolution: 2, seed: 5 }, { positions } );
	assert.equal( b.cancelled, false );
	assert.equal( b.graph.N, N );
	assert.deepEqual( b.graph.labels, labels );
	assert.deepEqual( Array.from( b.graph.edges ), edges );
	assert.equal( b.graph.clustering.method, 'louvain' );
	assert.equal( b.graph.clustering.resolution, 2 );
	assert.equal( b.graph.clustering.seed, 5 );
	assert.equal( b.graph.clustering.levels, b.graph.D );
	assert.equal( b.graph.clustering.stored, undefined );
	assert.notDeepEqual( b.graph.stats.clustersPerLevel, a.graph.stats.clustersPerLevel );
	assert.equal( b.graph.palette, 'ember', 'prepare options of the load are kept' );
	// no fetch, no parse: clustering and prepare only
	assert.equal( b.timings.parse, undefined );
	assert.equal( b.timings.fetch, undefined );
	assert.ok( b.timings.cluster > 0 );
	assert.deepEqual( [ ...new Set( stages ) ], [ 'cluster', 'prepare', 'transfer', 'done' ] );
	// initPos is the readback
	assert.equal( b.meta.repositioned, true );
	for ( let i = 0; i < N; i ++ ) {

		for ( let k = 0; k < 3; k ++ ) assert.equal( b.graph.initPos[ 4 * i + k ], positions[ 4 * i + k ] );
		assert.equal( b.graph.initPos[ 4 * i + 3 ], 0 );

	}

	assert.equal( positions[ 3 ], 7, 'the caller keeps its array' );

	// without positions the layout starts from the seed of the new hierarchy
	const c = await client.recluster( { method: 'leiden', resolution: 0.5 }, { palette: 'paper' } );
	assert.equal( c.meta.repositioned, false );
	assert.equal( c.graph.palette, 'paper' );
	assert.equal( c.graph.clustering.method, 'leiden' );
	assert.equal( c.graph.clustering.resolution, 0.5 );
	assert.ok( c.graph.clustering.communities < b.graph.clustering.communities );
	assert.deepEqual( Array.from( c.graph.edges ), edges );
	// the same options again: served from the worker's cache
	const again = await client.recluster( { method: 'leiden', resolution: 0.5 } );
	assert.equal( again.meta.cached, true );
	assert.equal( again.timings.cluster, undefined );
	// reprepare keeps the re-clustered hierarchy
	const re = await client.reprepare( { palette: 'aurora' } );
	assert.deepEqual( re.graph.stats.clustersPerLevel, c.graph.stats.clustersPerLevel );
	assert.equal( re.graph.palette, 'aurora' );
	// mode can be given: the stored hierarchy comes back
	const stored = await client.recluster( { mode: 'auto' } );
	assert.deepEqual( stored.graph.stats.clustersPerLevel, a.graph.stats.clustersPerLevel );

	// positions of another graph (too long, too short) are not used: the first 4N floats of a
	// larger readback would scatter the nodes at random
	for ( const length of [ 4 * N + 4, 4 * N - 4, 8 * N ] ) {

		const wrong = await client.recluster( { method: 'leiden', resolution: 2 }, { positions: new Float32Array( length ).fill( 3 ) } );
		assert.equal( wrong.meta.repositioned, false, `${length} floats` );
		assert.ok( wrong.meta.notes.some( ( note ) => note === `The start positions given were for ${( length >> 2 ).toLocaleString( 'en-US' )} nodes, this graph has 500: the layout starts from its own seed.` ), wrong.meta.notes.join( ' | ' ) );
		assert.notEqual( wrong.graph.initPos[ 0 ], 3 );

	}

	const right = await client.recluster( { method: 'leiden', resolution: 2 }, { positions } );
	assert.equal( right.meta.repositioned, true );
	assert.ok( ! right.meta.notes.some( ( note ) => /start positions/.test( note ) ) );
	client.dispose();

} );

test( 'recluster: a slider sends many requests, the worker runs the newest', async () => {

	const client = new DataClient( { inline: true } );
	await client.loadText( hsbm, 'hsbm-500.json' );
	const requests = [ 0.25, 0.5, 1, 2, 4 ].map( ( resolution ) => client.recluster( { resolution } ) );
	const results = await Promise.all( requests );
	assert.deepEqual( results.slice( 0, 4 ), [ { cancelled: true }, { cancelled: true }, { cancelled: true }, { cancelled: true } ] );
	assert.equal( results[ 4 ].graph.clustering.resolution, 4 );
	assert.ok( ! client.busy );
	// cancel() resolves the request; the next one still works (the worker reloads the source)
	const p = client.recluster( { resolution: 3 } );
	client.cancel();
	assert.deepEqual( await p, { cancelled: true } );
	const q = await client.recluster( { resolution: 3 } );
	assert.equal( q.graph.clustering.resolution, 3 );
	assert.equal( q.graph.N, 500 );
	// (a real Worker is terminated by cancel() and parses again; inline, the caches survive)
	client.dispose();

} );

test( 'recluster: resolution changes the community count on the built-in graphs', async () => {

	const client = new DataClient( { inline: true } );
	await client.loadText( readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ), 'lesmis.json' );
	const counts = [];
	for ( const resolution of [ 0.25, 1, 4 ] ) counts.push( ( await client.recluster( { resolution } ) ).graph.clustering.communities );
	assert.ok( counts[ 0 ] < counts[ 1 ] && counts[ 1 ] < counts[ 2 ], counts.join( ' < ' ) );
	client.dispose();

} );

test( 'recluster protocol: no source when the worker has the graph, need-source, worker kept', async () => {

	const { workers, create } = fakeWorkers();
	const client = new DataClient( { createWorker: create } );
	const load = client.loadText( '1 2\n2 3\n', 'a.txt', { palette: 'ember' } );
	const w = workers[ 0 ];
	const first = w.sent[ 0 ];
	assert.equal( first.type, 'load' );
	assert.equal( first.source.kind, 'text' );
	assert.equal( first.coalesce, false );
	assert.ok( first.key.startsWith( first.baseKey + '|' ) );
	assert.deepEqual( first.cluster, { mode: 'auto' } );
	w.reply( { id: first.id, type: 'result', graph: { N: 3 }, meta: {}, timings: {} } );
	await load;

	const positions = new Float32Array( 12 );
	const r1 = client.recluster( { method: 'louvain', resolution: 2 }, { positions } );
	const m1 = w.sent[ 1 ];
	assert.equal( m1.type, 'load' );
	assert.equal( m1.source, null, 'the worker has the parsed graph: the source is not sent again' );
	assert.equal( m1.coalesce, true );
	assert.equal( m1.baseKey, first.baseKey );
	assert.notEqual( m1.key, first.key );
	assert.deepEqual( m1.cluster, { mode: 'always', method: 'louvain', resolution: 2 } );
	assert.equal( m1.positions, positions );
	assert.deepEqual( m1.prepareOptions, { palette: 'ember' } );

	// a second re-clustering: the first resolves cancelled, the worker stays
	const r2 = client.recluster( { resolution: 3 } );
	assert.deepEqual( await r1, { cancelled: true } );
	assert.equal( workers.length, 1 );
	assert.ok( ! w.terminated );
	const m2 = w.sent[ 2 ];
	assert.deepEqual( m2.cluster, { mode: 'always', method: 'louvain', resolution: 3 } ); // options accumulate
	assert.equal( m2.positions, null );
	// replies to the stale request are ignored
	w.reply( { id: m1.id, type: 'skipped' } );
	w.reply( { id: m1.id, type: 'result', graph: { N: - 1 }, meta: {}, timings: {} } );
	// the worker lost its cache: it asks, the client sends the same request with the source
	w.reply( { id: m2.id, type: 'need-source' } );
	const m3 = w.sent[ 3 ];
	assert.equal( m3.id, m2.id );
	assert.equal( m3.source.kind, 'text' );
	assert.equal( m3.key, m2.key );
	w.reply( { id: m2.id, type: 'progress', stage: 'cluster', fraction: 0.5, message: 'x' } );
	w.reply( { id: m2.id, type: 'result', graph: { N: 3, tag: 'second' }, meta: {}, timings: {} } );
	assert.equal( ( await r2 ).graph.tag, 'second' );

	// a load after a re-clustering in flight terminates the worker, like any load over a load
	const r3 = client.recluster( { resolution: 4 } );
	const other = client.loadText( '5 6\n', 'b.txt' );
	assert.deepEqual( await r3, { cancelled: true } );
	assert.ok( w.terminated );
	assert.equal( workers.length, 2 );
	assert.equal( workers[ 1 ].sent[ 0 ].source.text, '5 6\n' );
	// a re-clustering over a load in flight starts over with the source (the new worker has nothing)
	const r4 = client.recluster( { resolution: 2 } );
	assert.deepEqual( await other, { cancelled: true } );
	assert.equal( workers.length, 3 );
	assert.equal( workers[ 2 ].sent[ 0 ].source.text, '5 6\n' );
	assert.equal( workers[ 2 ].sent[ 0 ].cluster.resolution, 2 );
	// (a readback is of the graph on screen, a.txt: it is not sent along with another source)
	assert.equal( workers[ 2 ].sent[ 0 ].positions, null );
	const r4b = client.recluster( { resolution: 2 }, { positions } );
	assert.deepEqual( await r4, { cancelled: true } );
	assert.equal( workers.at( - 1 ).sent.at( - 1 ).baseKey, workers[ 2 ].sent[ 0 ].baseKey );
	assert.equal( workers.at( - 1 ).sent.at( - 1 ).positions, null );
	// errors carry the worker's code
	const w4 = workers.at( - 1 );
	w4.reply( { id: w4.sent.at( - 1 ).id, type: 'error', message: 'too big', code: 'too-large' } );
	await assert.rejects( r4b, ( e ) => e.message === 'too big' && e.code === 'too-large' );
	client.dispose();

} );

test( 'recluster, reprepare and export act on the graph on screen after a load that failed or was cancelled', async () => {

	const { workers, create } = fakeWorkers();
	const client = new DataClient( { createWorker: create } );
	const answer = ( graph ) => {

		const w = workers.at( - 1 );
		w.reply( { id: w.sent.at( - 1 ).id, type: 'result', graph, meta: {}, timings: {} } );

	};

	const fail = () => {

		const w = workers.at( - 1 );
		w.reply( { id: w.sent.at( - 1 ).id, type: 'error', message: 'not a graph', code: 'format' } );

	};

	const last = () => workers.at( - 1 ).sent.at( - 1 );
	// the source of a request, whether it travelled with it or the worker has to ask for it
	const sourceOf = ( message ) => {

		if ( message.source ) return message.source;
		const w = workers.at( - 1 );
		w.reply( { id: message.id, type: 'need-source' } );
		return w.sent.at( - 1 ).source;

	};

	const onScreen = { N: 3, tag: 'a' };
	const a = client.loadText( '1 2\n2 3\n', 'a.txt', { palette: 'ember' }, { cluster: { method: 'louvain' } } );
	const aKey = last().baseKey;
	answer( onScreen );
	assert.equal( ( await a ).graph, onScreen );

	// 1. a load that fails
	const bad = client.loadText( 'garbage', 'bad.txt', { palette: 'paper' } );
	assert.notEqual( last().baseKey, aKey );
	fail();
	await assert.rejects( bad, /not a graph/ );
	const r1 = client.recluster( { resolution: 2 }, { positions: new Float32Array( 12 ) } );
	let m = last();
	assert.equal( m.baseKey, aKey, 'the re-clustering is of a.txt, not of the file that failed' );
	assert.deepEqual( m.cluster, { mode: 'always', method: 'louvain', resolution: 2 } );
	assert.deepEqual( m.prepareOptions, { palette: 'ember' } );
	assert.equal( m.positions.length, 12 );
	assert.equal( sourceOf( m ).text, '1 2\n2 3\n' );
	answer( { N: 3, tag: 'a2' } );
	assert.equal( ( await r1 ).graph.tag, 'a2' );

	// 2. a load that is cancelled
	const gone = client.loadText( '7 8\n', 'gone.txt' );
	client.cancel();
	assert.deepEqual( await gone, { cancelled: true } );
	const r2 = client.reprepare( { palette: 'aurora' } );
	m = last();
	assert.equal( m.baseKey, aKey );
	assert.equal( sourceOf( m ).text, '1 2\n2 3\n' );
	assert.deepEqual( m.cluster, { mode: 'always', method: 'louvain', resolution: 2 }, 'the hierarchy on screen is kept' );
	assert.deepEqual( last().prepareOptions, { palette: 'aurora' } );
	answer( { N: 3, tag: 'a3' } );
	assert.equal( ( await r2 ).graph.tag, 'a3' );

	// 3. a load superseded by another load that fails: still a.txt
	const one = client.loadText( '9 10\n', 'one.txt' );
	const two = client.loadText( 'garbage', 'two.txt' );
	assert.deepEqual( await one, { cancelled: true } );
	fail();
	await assert.rejects( two, /not a graph/ );
	const exported = client.export( 'csv' );
	m = last();
	assert.equal( m.type, 'export' );
	assert.equal( m.baseKey, aKey, 'the export is of the graph on screen' );
	assert.deepEqual( m.cluster, { mode: 'always', method: 'louvain', resolution: 2 } );
	assert.equal( sourceOf( m ).text, '1 2\n2 3\n' );
	client.dispose();
	await exported.catch( () => {} );

} );

test( 'recluster: the mode of the load is kept when it is "planted"; a load in flight gets the new options', async () => {

	const { workers, create } = fakeWorkers();
	const client = new DataClient( { createWorker: create } );
	const w = () => workers.at( - 1 );
	const answer = () => w().reply( { id: w().sent.at( - 1 ).id, type: 'result', graph: { N: 3 }, meta: {}, timings: {} } );
	const load = client.loadText( '1 2\n2 3\n', 'groups.txt', {}, { cluster: { mode: 'planted', method: 'louvain' } } );
	assert.deepEqual( w().sent[ 0 ].cluster, { mode: 'planted', method: 'louvain' } );
	answer();
	await load;
	// the groups of the file stay the top of the hierarchy when the slider moves
	const r1 = client.recluster( { resolution: 2 } );
	assert.deepEqual( w().sent.at( - 1 ).cluster, { mode: 'planted', method: 'louvain', resolution: 2 } );
	answer();
	await r1;
	// asked for by name, any mode replaces it
	const r2 = client.recluster( { mode: 'always' } );
	assert.deepEqual( w().sent.at( - 1 ).cluster, { mode: 'always', method: 'louvain', resolution: 2 } );
	answer();
	await r2;
	// 'auto' and 'never' of a load become 'always' (they would give the stored hierarchy or none)
	for ( const mode of [ 'auto', 'never' ] ) {

		const l = client.loadText( '1 2\n', `${mode}.txt`, {}, { cluster: { mode } } );
		answer();
		await l;
		const r = client.recluster( { seed: 2 } );
		assert.deepEqual( w().sent.at( - 1 ).cluster, { mode: 'always', seed: 2 } );
		answer();
		await r;

	}

	// a reprepare in flight is the same source: positions (a readback of the screen) stay valid
	const positions = new Float32Array( 12 );
	const re = client.reprepare( { palette: 'paper' } );
	const r3 = client.recluster( { resolution: 4 }, { positions } );
	assert.deepEqual( await re, { cancelled: true } );
	assert.equal( w().sent.at( - 1 ).positions, positions );
	assert.deepEqual( w().sent.at( - 1 ).prepareOptions, { palette: 'paper' } );
	assert.deepEqual( w().sent.at( - 1 ).cluster, { mode: 'always', seed: 2, resolution: 4 } );
	answer();
	await r3;
	client.dispose();

} );

test( 'worker: two-level cache, need-source, coalescing in the port', async () => {

	const source = { kind: 'text', text: hsbm, name: 'hsbm-500.json' };
	const base = 'test:recluster:' + Math.random();
	const a = await runRequest( { key: base + '|auto', baseKey: base, source, cluster: { mode: 'auto' } } );
	assert.ok( a.timings.parse >= 0 );
	// same source, other clustering: the parsed graph is reused, no source needed
	const b = await runRequest( { key: base + '|louvain', baseKey: base, source: null, cluster: { mode: 'always', method: 'louvain' } } );
	assert.equal( b.timings.parse, undefined );
	assert.ok( b.timings.cluster > 0 );
	assert.equal( b.meta.cached, false );
	assert.equal( b.graph.clustering.method, 'louvain' );
	// the first clustering is still cached
	const c = await runRequest( { key: base + '|auto', baseKey: base, source: null, cluster: { mode: 'auto' } } );
	assert.equal( c.meta.cached, true );
	assert.deepEqual( Array.from( c.graph.anc ), Array.from( a.graph.anc ) );
	// the transferred arrays are copies: the cached graph still has its edges, weights and attributes
	structuredClone( c.graph, { transfer: [ c.graph.edges.buffer ] } );
	const d = await runRequest( { key: base + '|auto', baseKey: base, source: null, cluster: { mode: 'auto' } } );
	assert.equal( d.graph.edges.length, 2 * d.graph.E );

	await assert.rejects( runRequest( { key: 'unknown|x', baseKey: 'unknown', source: null, cluster: {} } ), NeedSource );
	const posts = [];
	await handleMessage( { id: 3, type: 'load', key: 'unknown|x', baseKey: 'unknown', source: null }, ( m ) => posts.push( m ) );
	assert.deepEqual( posts, [ { id: 3, type: 'need-source' } ] );

	// the port: three coalescing loads in one go, the first two are skipped
	const replies = [];
	const port = createPort( ( m ) => replies.push( m ) );
	for ( const [ id, resolution ] of [ [ 11, 0.5 ], [ 12, 2 ], [ 13, 4 ] ] ) port.receive( { id, type: 'load', key: `${base}|r${resolution}`, baseKey: base, source: null, cluster: { mode: 'always', resolution }, coalesce: true } );
	for ( let k = 0; k < 200 && ! replies.some( ( m ) => m.type === 'result' ); k ++ ) await tick();
	assert.deepEqual( replies.filter( ( m ) => m.type === 'skipped' ).map( ( m ) => m.id ), [ 11, 12 ] );
	const result = replies.find( ( m ) => m.type === 'result' );
	assert.equal( result.id, 13 );
	assert.equal( result.graph.clustering.resolution, 4 );
	// loads that do not coalesce all run, in order
	replies.length = 0;
	for ( const id of [ 21, 22 ] ) port.receive( { id, type: 'load', key: `${base}|auto`, baseKey: base, source: null, cluster: { mode: 'auto' } } );
	for ( let k = 0; k < 200 && replies.filter( ( m ) => m.type === 'result' ).length < 2; k ++ ) await tick();
	assert.deepEqual( replies.filter( ( m ) => m.type === 'result' ).map( ( m ) => m.id ), [ 21, 22 ] );

} );

test( 'prepareGraph: initPos replaces the seed; broken coordinates fall back to it', () => {

	const g = applyClustering( generate( { model: 'caveman', params: { cliques: 6, cliqueSize: 6 }, seed: 1 } ) );
	const seed = prepareGraph( g );
	const N = seed.N;
	const given = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) given.set( [ i, 2 * i, 0 - i, 9 ], 4 * i );
	const p = prepareGraph( g, { initPos: given } );
	assert.notEqual( p.initPos, given, 'a copy' );
	for ( let i = 0; i < N; i ++ ) assert.deepEqual( Array.from( p.initPos.subarray( 4 * i, 4 * i + 4 ) ), [ i, 2 * i, 0 - i, 0 ] );
	// everything else is unchanged
	assert.deepEqual( Array.from( p.anc ), Array.from( seed.anc ) );
	assert.deepEqual( Array.from( p.colors ), Array.from( seed.colors ) );
	assert.equal( p.layoutRadius, seed.layoutRadius );
	// a NaN or infinite coordinate: that node starts at its seed position
	given[ 4 * 3 + 1 ] = NaN;
	given[ 4 * 7 ] = Infinity;
	const q = prepareGraph( g, { initPos: given } );
	for ( const i of [ 3, 7 ] ) for ( let k = 0; k < 3; k ++ ) assert.ok( Number.isFinite( q.initPos[ 4 * i + k ] ) );
	assert.deepEqual( Array.from( q.initPos.subarray( 0, 4 ) ), [ 0, 0, 0, 0 ] );
	assert.deepEqual( Array.from( q.initPos.subarray( 16, 20 ) ), [ 4, 8, - 4, 0 ] );
	// too short: ignored
	assert.deepEqual( Array.from( prepareGraph( g, { initPos: new Float32Array( 8 ) } ).initPos ), Array.from( seed.initPos ) );
	// too long: the positions of another graph, ignored as well (not its first 4N floats)
	assert.deepEqual( Array.from( prepareGraph( g, { initPos: new Float32Array( 4 * N + 4 ).fill( 5 ) } ).initPos ), Array.from( seed.initPos ) );

} );

test( 'prepareGraph: attributes, edge weights and the clustering record pass through', () => {

	const raw = generate( { model: 'caveman', params: { cliques: 4, cliqueSize: 5 }, seed: 1 } );
	const E = raw.edges.length / 2, N = raw.nodeCount;
	const g = applyClustering( {
		...raw,
		weights: Float32Array.from( { length: E }, ( _, e ) => 1 + ( e % 3 ) ),
		attributes: {
			age: { type: 'number', values: Float32Array.from( { length: N }, ( _, i ) => ( i === 2 ? NaN : i ) ) },
			kind: { type: 'category', values: Int32Array.from( { length: N }, ( _, i ) => ( i % 3 ) - 1 ), categories: [ 'a', 'b' ] },
		},
	} );
	const p = prepareGraph( g );
	assert.equal( p.edgeWeight.length, p.E );
	assert.ok( p.edgeWeight instanceof Float32Array );
	for ( let e = 0; e < p.E; e ++ ) assert.equal( p.edgeWeight[ e ], 1 + ( e % 3 ) );
	assert.deepEqual( Object.keys( p.attributes ), [ 'age', 'kind' ] );
	assert.ok( Number.isNaN( p.attributes.age.values[ 2 ] ) );
	assert.equal( p.attributes.kind.values[ 0 ], - 1 );
	assert.deepEqual( p.attributes.kind.categories, [ 'a', 'b' ] );
	assert.equal( p.clustering.method, 'leiden' );
	assert.equal( p.clustering.weighted, true );
	// a graph without any of it
	const plain = prepareGraph( { nodeCount: 3, edges: [ 0, 1, 1, 2 ] } );
	assert.deepEqual( plain.attributes, {} );
	assert.equal( plain.edgeWeight, null );
	assert.equal( plain.clustering, null );

} );
