// Data worker (module Web Worker): fetch / read, parse, generate, cluster (Louvain) and
// prepareGraph off the main thread, posting progress and transferring the PreparedGraph's typed
// arrays back. Created by src/data/client.js (DataClient); the same pipeline also runs inline
// (main thread or Node tests) through the exported functions.
//
// ─── Protocol ───────────────────────────────────────────────────────────────────────────────
//  main → worker  { id, type: 'load', key, source, prepareOptions, cluster }
//      source   { kind: 'url', url }            absolute URL (resolve against the page first)
//               { kind: 'file', file }          File / Blob (name used for format detection)
//               { kind: 'text', text, name }
//               { kind: 'generate', spec }      { model, params, seed, labels, name, description }
//      cluster  { mode: 'auto' | 'always' | 'never' | 'planted', seed, resolution, maxLeafSize }
//               auto: Louvain only when the graph has no hierarchy; planted: generator ground
//               truth when available (else Louvain).
//      key      cache key; the last few canonical graphs (after clustering) are kept so a
//               repeated request (e.g. new palette) skips fetch, parse and Louvain.
//  main → worker  { type: 'clear-cache' }
//  worker → main  { id, type: 'progress', stage, fraction, message }   fraction: 0..1 overall
//                 { id, type: 'result', graph, meta, timings }          graph: PreparedGraph
//                 { id, type: 'error', message }
//  Stages: fetch | read | parse | generate | cluster | prepare | transfer.
//
// ─── Formats ────────────────────────────────────────────────────────────────────────────────
//  JSON (by extension .json, or content starting with { / [): wgv-graph, ImsvGraphVis .igv.json,
//  node-link ({nodes, links|edges}). Anything else: text edge list (src/data/edgelist.js).
//  A trailing .gz is decompressed with DecompressionStream when the server did not already.
//
// ─── Exports (for inline use / tests) ───────────────────────────────────────────────────────
//  runRequest( msg, onProgress ) -> Promise<{ graph, meta, timings }>
//  handleMessage( msg, post )     post( data, transfer ) receives progress/result/error messages
//  parseGraphText( text, name ) -> { graph, format }
//  transferables( preparedGraph ) -> ArrayBuffer[]

import { parseGraphJson, normalizeLevels } from './format.js';
import { parseEdgeList, looksLikeEdgeList } from './edgelist.js';
import { generate } from './generators.js';
import { ensureHierarchy } from './louvain.js';
import { prepareGraph } from '../graph/prepare.js';

const CACHE_SIZE = 3;
const cache = new Map(); // key -> canonical graph with hierarchy

const EDGELIST_EXT = /\.(txt|csv|tsv|ssv|edges?|edgelist|el|mtx|dat|tab|ncol)$/i;

const PLANS = {
	url: [ [ 'fetch', 30 ], [ 'parse', 10 ], [ 'cluster', 30 ], [ 'prepare', 25 ], [ 'transfer', 5 ] ],
	file: [ [ 'read', 15 ], [ 'parse', 15 ], [ 'cluster', 35 ], [ 'prepare', 30 ], [ 'transfer', 5 ] ],
	text: [ [ 'parse', 25 ], [ 'cluster', 40 ], [ 'prepare', 30 ], [ 'transfer', 5 ] ],
	generate: [ [ 'generate', 20 ], [ 'cluster', 40 ], [ 'prepare', 35 ], [ 'transfer', 5 ] ],
	cached: [ [ 'prepare', 90 ], [ 'transfer', 10 ] ],
};

/** Maps per-stage progress to an overall fraction and throttles messages (~20 per second). */
class Progress {

	constructor( plan, emit ) {

		this.plan = plan;
		this.emit = emit || ( () => {} );
		this.total = plan.reduce( ( s, [ , w ] ) => s + w, 0 );
		this.stageName = null;
		this.base = 0;
		this.weight = 0;
		this.last = 0;
		this.lastFraction = 0;

	}

	stage( name, message = '' ) {

		let base = 0, weight = 0;
		for ( const [ s, w ] of this.plan ) {

			if ( s === name ) {

				weight = w;
				break;

			}

			base += w;

		}

		this.stageName = name;
		this.base = base;
		this.weight = weight;
		this.update( 0, message, true );

	}

	update( f, message = '', force = false ) {

		const t = now();
		if ( ! force && t - this.last < 50 ) return;
		this.last = t;
		const fraction = Math.max( this.lastFraction, Math.min( 1, ( this.base + this.weight * Math.min( 1, Math.max( 0, f ) ) ) / this.total ) );
		this.lastFraction = fraction;
		this.emit( { stage: this.stageName, fraction, message } );

	}

}

function baseName( name ) {

	const file = String( name || 'graph' ).split( /[\\/]/ ).pop().split( '?' )[ 0 ];
	return file.replace( /\.gz$/i, '' ).replace( /\.igv\.json$/i, '' ).replace( /\.[a-z0-9]+$/i, '' ) || 'graph';

}

/** Parses text by name/content: JSON graph formats or an edge list. */
export function parseGraphText( text, name = 'graph' ) {

	const lower = String( name ).toLowerCase().replace( /\.gz$/, '' ).split( '?' )[ 0 ];
	const json = /\.json$/.test( lower ) || ( ! EDGELIST_EXT.test( lower ) && ! looksLikeEdgeList( text ) );
	if ( json ) {

		let parsed;
		try {

			parsed = JSON.parse( text );

		} catch ( e ) {

			throw new Error( `${baseName( name )}: invalid JSON (${e.message})` );

		}

		const graph = parseGraphJson( parsed, baseName( name ) );
		const format = parsed.format === 'wgv-graph' ? 'wgv-graph' : parsed.clusters ? 'igv' : 'node-link';
		return { graph, format };

	}

	return { graph: parseEdgeList( text, { name: baseName( name ) } ), format: 'edgelist' };

}

async function readStream( stream, total, gz, progress, label ) {

	const reader = stream.getReader();
	const chunks = [];
	let bytes = 0;
	for ( ;; ) {

		const { done, value } = await reader.read();
		if ( done ) break;
		chunks.push( value );
		bytes += value.byteLength;
		if ( total ) progress.update( bytes / total, `${label} ${formatBytes( bytes )} / ${formatBytes( total )}` );
		else progress.update( 0.5, `${label} ${formatBytes( bytes )}` );

	}

	let blob = new Blob( chunks );
	if ( gz ) {

		if ( typeof DecompressionStream === 'undefined' ) throw new Error( 'gzip input is not supported in this browser' );
		blob = await new Response( blob.stream().pipeThrough( new DecompressionStream( 'gzip' ) ) ).blob();

	}

	return { text: await blob.text(), bytes };

}

function isGzipName( name ) {

	return /\.gz$/i.test( String( name ).split( '?' )[ 0 ] );

}

async function loadSource( source, progress, timings ) {

	let t = now();
	if ( source.kind === 'generate' ) {

		progress.stage( 'generate', `Generating ${source.spec.model}…` );
		const graph = generate( source.spec );
		timings.generate = now() - t;
		return { graph, format: 'generated' };

	}

	let text, name;
	if ( source.kind === 'url' ) {

		name = source.url;
		progress.stage( 'fetch', `Downloading ${baseName( name )}…` );
		const res = await fetch( source.url );
		if ( ! res.ok ) throw new Error( `${baseName( name )}: HTTP ${res.status}` );
		const encoded = !! res.headers.get( 'content-encoding' );
		const total = encoded ? 0 : Number( res.headers.get( 'content-length' ) ) || 0;
		const gz = isGzipName( name ) && ! /gzip/i.test( res.headers.get( 'content-encoding' ) || '' );
		( { text } = await readStream( res.body, total, gz, progress, 'Downloading' ) );
		timings.fetch = now() - t;

	} else if ( source.kind === 'file' ) {

		const file = source.file;
		name = file.name || source.name || 'file';
		progress.stage( 'read', `Reading ${name}…` );
		( { text } = await readStream( file.stream(), file.size, isGzipName( name ), progress, 'Reading' ) );
		timings.read = now() - t;

	} else if ( source.kind === 'text' ) {

		name = source.name || 'graph.txt';
		text = source.text;

	} else throw new Error( `Unknown source kind "${source.kind}"` );

	t = now();
	progress.stage( 'parse', `Parsing ${baseName( name )}…` );
	const parsed = parseGraphText( text, name );
	timings.parse = now() - t;
	return parsed;

}

function cluster( graph, opts, progress, timings ) {

	const mode = opts.mode || 'auto';
	const t = now();
	if ( mode === 'never' ) return { ...graph, hierarchy: null };
	if ( mode === 'planted' && graph.planted ) {

		const levels = normalizeLevels( graph.planted.levels.map( ( l ) => Array.from( l ) ), graph.nodeCount );
		return { ...graph, hierarchy: levels.length ? { method: 'planted', levels } : null };

	}

	const has = graph.hierarchy && graph.hierarchy.levels && graph.hierarchy.levels.length;
	if ( has && mode !== 'always' ) return graph;
	progress.stage( 'cluster', `Clustering ${graph.nodeCount.toLocaleString( 'en-US' )} nodes (Louvain)…` );
	let level = 0;
	const out = ensureHierarchy( graph, {
		force: true,
		seed: opts.seed ?? 1,
		resolution: opts.resolution ?? 1,
		maxLeafSize: opts.maxLeafSize,
		onProgress: ( p ) => {

			if ( p.phase === 'refine' ) {

				progress.update( 0.85 + 0.15 * ( p.pass / Math.max( 1, p.communities ) ), `Refining large clusters ${p.pass}/${p.communities}…` );
				return;

			}

			level = Math.max( level, p.level );
			// levels shrink geometrically: level l covers roughly the next half of the remaining work
			const f = 1 - Math.pow( 0.5, level + Math.min( 1, p.pass / 12 ) );
			progress.update( 0.85 * f, `Louvain level ${p.level + 1}, pass ${p.pass} (Q = ${p.modularity.toFixed( 3 )})` );

		},
	} );
	timings.cluster = now() - t;
	return out;

}

/** Runs one load request; resolves to { graph (PreparedGraph), meta, timings }. */
export async function runRequest( msg, onProgress ) {

	const t0 = now();
	const timings = {};
	const key = msg.key || null;
	const source = msg.source;
	const clusterOpts = msg.cluster || {};
	let canonical = key ? cache.get( key ) : null;
	const hit = !! canonical;
	let format = canonical ? canonical.__format : null;
	const progress = new Progress( PLANS[ canonical ? 'cached' : source.kind ] || PLANS.text, onProgress );
	if ( canonical ) {

		cache.delete( key ); // refresh LRU position
		cache.set( key, canonical );

	} else {

		const loaded = await loadSource( source, progress, timings );
		format = loaded.format;
		canonical = cluster( loaded.graph, clusterOpts, progress, timings );
		if ( key ) {

			Object.defineProperty( canonical, '__format', { value: format, enumerable: false, configurable: true } );
			cache.set( key, canonical );
			while ( cache.size > CACHE_SIZE ) cache.delete( cache.keys().next().value );

		}

	}

	const t = now();
	progress.stage( 'prepare', 'Preparing GPU buffers…' );
	// prepareGraph keeps a Uint32Array edges array as is; copy so the cached graph survives transfer
	const graph = prepareGraph( { ...canonical, edges: Uint32Array.from( canonical.edges ) }, msg.prepareOptions || {} );
	timings.prepare = now() - t;

	const h = canonical.hierarchy;
	const meta = {
		key,
		kind: source.kind,
		format,
		name: canonical.name,
		description: canonical.description,
		source: canonical.source,
		nodes: graph.N,
		edges: graph.E,
		levels: graph.D,
		clustersPerLevel: graph.stats.clustersPerLevel,
		hierarchy: h ? { method: h.method, ...( canonical.meta?.hierarchy || {} ) } : null,
		generator: canonical.meta?.generator ? { model: canonical.meta.generator, params: canonical.meta.params, seed: canonical.meta.seed } : null,
		stats: canonical.meta?.stats || null,
		file: canonical.meta?.format === 'edgelist' ? canonical.meta : null,
		cached: hit,
	};
	graph.meta = meta;
	progress.stage( 'transfer', 'Transferring…' );
	timings.total = now() - t0;
	return { graph, meta, timings };

}

/** Every distinct ArrayBuffer referenced by typed arrays at the top level of the prepared graph. */
export function transferables( graph ) {

	const set = new Set();
	for ( const v of Object.values( graph ) ) {

		if ( ArrayBuffer.isView( v ) && v.buffer instanceof ArrayBuffer ) set.add( v.buffer );

	}

	return [ ...set ];

}

/** Handles one protocol message; post( data, transfer? ) sends the replies. */
export async function handleMessage( msg, post ) {

	if ( ! msg || typeof msg !== 'object' ) return;
	if ( msg.type === 'clear-cache' ) {

		cache.clear();
		return;

	}

	if ( msg.type !== 'load' ) return;
	const id = msg.id;
	try {

		const result = await runRequest( msg, ( p ) => post( { id, type: 'progress', ...p } ) );
		post( { id, type: 'result', graph: result.graph, meta: result.meta, timings: result.timings }, transferables( result.graph ) );

	} catch ( e ) {

		post( { id, type: 'error', message: ( e && e.message ) || String( e ) } );

	}

}

function formatBytes( b ) {

	if ( b < 1024 ) return `${b} B`;
	if ( b < 1024 * 1024 ) return `${( b / 1024 ).toFixed( 0 )} KB`;
	return `${( b / 1024 / 1024 ).toFixed( 1 )} MB`;

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}

// ── worker entry ─────────────────────────────────────────────────────────────────────────────

const isWorker = typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope; // eslint-disable-line no-undef
if ( isWorker ) {

	// one request at a time, in arrival order (the client terminates the worker to cancel)
	let queue = Promise.resolve();
	self.onmessage = ( e ) => {

		queue = queue.then( () => handleMessage( e.data, ( data, transfer ) => self.postMessage( data, transfer || [] ) ) );

	};

}
