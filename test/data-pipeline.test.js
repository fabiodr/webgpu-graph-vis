// Worker pipeline (run inline in Node), DataClient request semantics, and the generated presets.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { runRequest, handleMessage, parseGraphText, transferables } from '../src/data/worker.js';
import { DataClient } from '../src/data/client.js';
import { serializeGraph, normalizeGraph } from '../src/data/format.js';
import { generate } from '../src/data/generators.js';

const manifest = JSON.parse( readFileSync( 'public/data/manifest.json', 'utf8' ) );

test( 'parseGraphText: wgv-graph, node-link, ImsvGraphVis igv, edge list', () => {

	const lesmis = readFileSync( 'public/data/imsv/lesmis.json', 'utf8' );
	const a = parseGraphText( lesmis, 'lesmis.json' );
	assert.equal( a.format, 'wgv-graph' );
	assert.equal( a.graph.nodeCount, 77 );
	assert.ok( a.graph.hierarchy.levels.length === 2 );

	const nl = JSON.stringify( { nodes: [ { id: 'x' }, { id: 'y' }, { id: 'z' } ], links: [ { source: 'x', target: 'y' }, { source: 'y', target: 'z' } ] } );
	const b = parseGraphText( nl, 'graph.json' );
	assert.equal( b.format, 'node-link' );
	assert.deepEqual( b.graph.labels, [ 'x', 'y', 'z' ] );

	const igv = JSON.stringify( {
		nodes: [ { idx: 0, label: 'p', ancIdxs: [ 3 ] }, { idx: 1, label: 'q', ancIdxs: [ 3 ] }, { idx: 2, label: 'r', ancIdxs: [ 4 ] } ],
		links: [ { sourceIdx: 0, targetIdx: 1 }, { sourceIdx: 1, targetIdx: 2 } ],
		clusters: [],
	} );
	const c = parseGraphText( igv, 'tiny.igv.json' );
	assert.equal( c.format, 'igv' );
	assert.equal( c.graph.name, 'tiny' );
	assert.deepEqual( c.graph.hierarchy.levels, [ [ 0, 0, 1 ] ] );

	// sniffed by content when the extension says nothing
	assert.equal( parseGraphText( nl, 'download' ).format, 'node-link' );
	assert.equal( parseGraphText( 'a b\nb c\n', 'download' ).format, 'edgelist' );
	assert.equal( parseGraphText( '1,2\n', 'edges.csv' ).format, 'edgelist' );
	assert.throws( () => parseGraphText( '{ nope', 'x.json' ), /invalid JSON/ );

} );

test( 'runRequest: edge list -> Louvain -> PreparedGraph, progress, cache', async () => {

	const text = readFileSync( 'public/data/imsv/celegansneural.json', 'utf8' );
	const g = JSON.parse( text );
	const lines = [];
	for ( let k = 0; k < g.edges.length; k += 2 ) lines.push( `${g.labels[ g.edges[ k ] ]}\t${g.labels[ g.edges[ k + 1 ] ]}` );
	const events = [];
	const msg = { id: 1, type: 'load', key: 'test:celegans', source: { kind: 'text', text: lines.join( '\n' ), name: 'celegans.tsv' }, prepareOptions: { palette: 'rainbow' }, cluster: { mode: 'auto' } };
	const r = await runRequest( msg, ( p ) => events.push( p ) );
	assert.equal( r.graph.N, 297 );
	assert.equal( r.graph.E, 2148 );
	assert.ok( r.graph.D >= 1 );
	assert.equal( r.graph.palette, 'rainbow' );
	assert.equal( r.meta.format, 'edgelist' );
	assert.match( r.meta.hierarchy.method, /^louvain/ );
	assert.equal( r.meta.hierarchy.modularity.length, r.graph.D );
	assert.deepEqual( [ ...r.graph.labels ].sort(), [ ...g.labels ].sort() );
	const stages = [ ...new Set( events.map( ( e ) => e.stage ) ) ];
	assert.deepEqual( stages, [ 'parse', 'cluster', 'prepare', 'transfer' ] );
	for ( let k = 1; k < events.length; k ++ ) assert.ok( events[ k ].fraction >= events[ k - 1 ].fraction );
	assert.ok( events.every( ( e ) => e.fraction >= 0 && e.fraction <= 1 && typeof e.message === 'string' ) );

	// same key: cached canonical graph, only prepare runs; different palette applies
	const r2 = await runRequest( { ...msg, prepareOptions: { palette: 'ember' } }, () => {} );
	assert.equal( r2.meta.cached, true );
	assert.equal( r2.timings.cluster, undefined );
	assert.equal( r2.graph.palette, 'ember' );
	assert.deepEqual( r2.graph.anc, r.graph.anc );

} );

test( 'runRequest: cluster modes (planted / never / keep existing)', async () => {

	const spec = { model: 'hsbm', params: { n: 600, avgDegree: 10, depth: 2, branching: 4 }, seed: 2 };
	const planted = await runRequest( { source: { kind: 'generate', spec }, cluster: { mode: 'planted' } } );
	assert.equal( planted.meta.hierarchy.method, 'planted' );
	assert.equal( planted.graph.D, 2 );
	assert.deepEqual( planted.graph.stats.clustersPerLevel, [ 16, 4 ] );
	const none = await runRequest( { source: { kind: 'generate', spec }, cluster: { mode: 'never' } } );
	assert.equal( none.graph.D, 0 );
	const lesmis = await runRequest( { source: { kind: 'text', text: readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ), name: 'lesmis.json' } } );
	assert.equal( lesmis.meta.hierarchy.method, 'louvain (python-louvain 0.8, ImsvGraphVis preprocessing)' );
	assert.equal( lesmis.timings.cluster, undefined );
	const again = await runRequest( { source: { kind: 'text', text: readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ), name: 'lesmis.json' } }, null );
	assert.equal( again.graph.D, 2 );
	const forced = await runRequest( { source: { kind: 'text', text: readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ), name: 'lesmis.json' }, cluster: { mode: 'always', seed: 3 } } );
	assert.match( forced.meta.hierarchy.method, /^louvain$/ );

} );

test( 'handleMessage: result transfers every typed array; errors are reported', async () => {

	const posts = [];
	await handleMessage( { id: 7, type: 'load', source: { kind: 'generate', spec: { model: 'ws', params: { n: 300, k: 6 } } } }, ( data, transfer ) => posts.push( { data, transfer } ) );
	const last = posts[ posts.length - 1 ];
	assert.equal( last.data.type, 'result' );
	assert.equal( last.data.id, 7 );
	assert.ok( posts.slice( 0, - 1 ).every( ( p ) => p.data.type === 'progress' && p.data.id === 7 ) );
	const buffers = last.transfer;
	assert.equal( new Set( buffers ).size, buffers.length );
	assert.deepEqual( buffers, transferables( last.data.graph ) );
	assert.ok( buffers.length >= 20 );
	// structured clone with transfer works and detaches the sources
	const before = last.data.graph.N;
	const copy = structuredClone( last.data, { transfer: buffers } );
	assert.equal( copy.graph.N, before );
	assert.equal( copy.graph.adjList.length, 2 * copy.graph.E );
	assert.equal( last.data.graph.adjList.length, 0 );

	const errs = [];
	await handleMessage( { id: 8, type: 'load', source: { kind: 'generate', spec: { model: 'bogus' } } }, ( d ) => errs.push( d ) );
	assert.equal( errs.at( - 1 ).type, 'error' );
	assert.match( errs.at( - 1 ).message, /Unknown generator/ );

} );

test( 'DataClient: newer requests supersede older ones (resolve as cancelled, no race)', async () => {

	const client = new DataClient( { inline: true } );
	const progress = [];
	client.onProgress( ( p ) => progress.push( p ) );
	const first = client.generate( { model: 'lfr', params: { n: 6000, avgDegree: 20 }, seed: 1 } );
	const second = client.loadText( '1 2\n2 3\n3 1\n3 4\n4 5\n5 6\n6 4\n', 'two-triangles.txt' );
	assert.ok( client.busy );
	assert.deepEqual( await first, { cancelled: true } );
	const r = await second;
	assert.equal( r.cancelled, false );
	assert.equal( r.graph.N, 6 );
	assert.equal( r.graph.D, 1 );
	assert.deepEqual( r.graph.labels, [ '1', '2', '3', '4', '5', '6' ] );
	assert.equal( r.meta.name, 'two-triangles' );
	assert.ok( r.timings.roundTrip >= 0 );
	assert.equal( progress.at( - 1 ).stage, 'done' );
	assert.ok( ! client.busy );

	// cancel()
	const third = client.generate( { model: 'ba', params: { n: 500, m: 2 } } );
	client.cancel();
	assert.deepEqual( await third, { cancelled: true } );

	// errors reject
	await assert.rejects( client.generate( { model: 'nope' } ), /Unknown generator/ );
	await assert.rejects( client.load( {} ), /file.*generate/ );

	// reprepare: same source, new prepare options
	await client.generate( { model: 'caveman', params: { cliques: 6, cliqueSize: 5 } }, { palette: 'paper' } );
	const re = await client.reprepare( { palette: 'aurora' } );
	assert.equal( re.graph.palette, 'aurora' );
	assert.equal( re.meta.cached, true );
	client.dispose();

} );

test( 'DataClient: supersede terminates the worker and ignores its stale messages', async () => {

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

	}

	const client = new DataClient( { createWorker: () => new FakeWorker() } );
	const seen = [];
	client.onProgress( ( p ) => seen.push( p.message ) );
	const a = client.loadUrl( 'https://example.org/a.json' );
	const w0 = workers[ 0 ];
	assert.equal( w0.sent[ 0 ].source.url, 'https://example.org/a.json' );
	const idA = w0.sent[ 0 ].id;
	w0.onmessage( { data: { id: idA, type: 'progress', stage: 'fetch', fraction: 0.1, message: 'A' } } );
	const b = client.generate( { model: 'ba', params: { n: 100 } } );
	assert.ok( w0.terminated, 'old worker terminated' );
	assert.equal( workers.length, 2 );
	assert.deepEqual( await a, { cancelled: true } );
	const w1 = workers[ 1 ];
	const idB = w1.sent[ 0 ].id;
	assert.notEqual( idA, idB );
	// a late message from the first request must not resolve the second
	w1.onmessage( { data: { id: idA, type: 'result', graph: { N: - 1 }, meta: {}, timings: {} } } );
	w1.onmessage( { data: { id: idB, type: 'progress', stage: 'generate', fraction: 0.2, message: 'B' } } );
	w1.onmessage( { data: { id: idB, type: 'result', graph: { N: 100 }, meta: { name: 'b' }, timings: { total: 1 } } } );
	const rb = await b;
	assert.equal( rb.graph.N, 100 );
	assert.deepEqual( seen, [ 'A', 'B', 'Ready' ] );
	// an idle client keeps its worker (and the worker's cache) for the next request
	const d = client.generate( { model: 'ba', params: { n: 100 } } );
	assert.equal( workers.length, 2 );
	assert.ok( ! w1.terminated );
	w1.onmessage( { data: { id: w1.sent[ 1 ].id, type: 'error', message: 'bad input' } } );
	await assert.rejects( d, /bad input/ );
	// a worker failure (e.g. script load error) rejects the pending request and drops the worker
	const c = client.generate( { model: 'ws' } );
	assert.equal( workers.length, 2 );
	w1.onerror( { message: 'boom' } );
	await assert.rejects( c, /boom/ );
	assert.ok( w1.terminated );
	client.dispose();

} );

test( 'listModels via the client matches the generator registry', () => {

	const models = new DataClient( { inline: true } ).listModels();
	assert.deepEqual( models.map( ( m ) => m.id ), [ 'hsbm', 'lfr', 'ba', 'ws', 'rgg3d', 'lattice3d', 'caveman' ] );

} );

test( 'manifest: generated entries carry either a file or a generate spec, with correct counts', async () => {

	const gen = manifest.datasets.filter( ( d ) => d.group === 'generated' );
	assert.ok( gen.length >= 9 );
	for ( const d of manifest.datasets ) {

		assert.ok( d.id && d.group && d.name && Number.isFinite( d.nodes ) && Number.isFinite( d.edges ) && Number.isFinite( d.levels ), d.id );
		assert.ok( !! d.file !== !! d.generate, `${d.id}: file xor generate` );

	}

	assert.ok( gen.some( ( d ) => d.generate ) && gen.some( ( d ) => d.file ) );
	for ( const d of gen.filter( ( x ) => x.file ) ) {

		const path = 'public/' + d.file;
		assert.ok( existsSync( path ), path );
		assert.ok( d.bytes <= 1.5 * 1024 * 1024 );
		const g = normalizeGraph( JSON.parse( readFileSync( path, 'utf8' ) ) );
		assert.equal( g.nodeCount, d.nodes );
		assert.equal( g.edges.length / 2, d.edges );
		assert.equal( g.hierarchy ? g.hierarchy.levels.length : 0, d.levels );

	}

	// file presets are reproducible from their stored generator parameters
	const small = JSON.parse( readFileSync( 'public/data/generated/hsbm-500.json', 'utf8' ) );
	const again = generate( { model: small.meta.generator, params: small.meta.params, seed: small.meta.seed } );
	assert.deepEqual( Array.from( again.edges ), small.edges );

	// generate-spec presets: the worker pipeline reproduces the advertised counts exactly
	for ( const d of gen.filter( ( x ) => x.generate ) ) {

		const r = await runRequest( { source: { kind: 'generate', spec: d.generate }, cluster: { mode: 'auto' } } );
		assert.equal( r.graph.N, d.nodes, d.id );
		assert.equal( r.graph.E, d.edges, d.id );
		assert.equal( r.graph.D, d.levels, d.id );

	}

} );

test( 'serialized generated graph loads through the pipeline unchanged', async () => {

	const g = generate( { model: 'rgg3d', params: { n: 400, avgDegree: 8 }, seed: 5 } );
	const r = await runRequest( { source: { kind: 'text', text: serializeGraph( g ), name: 'rgg.json' } } );
	assert.equal( r.graph.E, g.edges.length / 2 );
	assert.equal( r.meta.format, 'wgv-graph' );
	assert.equal( r.meta.generator.model, 'rgg3d' );

} );
