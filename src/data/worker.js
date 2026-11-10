// Data worker (module Web Worker): fetch / read, parse, generate, cluster and prepareGraph off
// the main thread, posting progress and transferring the PreparedGraph's typed arrays back;
// also static layouts and edge bundling for the graph on screen. Created by src/data/client.js
// (DataClient); the same pipeline also runs inline (main thread or Node tests) through the
// exported functions.
//
// ─── Protocol ───────────────────────────────────────────────────────────────────────────────
//  main → worker  { id, type: 'load', key, baseKey, source, prepareOptions, cluster, positions?,
//                   limits?, coalesce? }
//      source   { kind: 'url', url }            absolute URL (resolve against the page first)
//               { kind: 'file', file }          File / Blob (name used for format detection)
//               { kind: 'text', text, name }
//               { kind: 'generate', spec }      { model, params, seed, labels, name, description }
//               null                            "you have it cached" (re-clustering); when the
//                                               cache does not, the reply is 'need-source' and
//                                               the client sends the request again with it
//      cluster  { mode: 'auto' | 'always' | 'never' | 'planted', method: 'leiden' | 'louvain',
//                 resolution, seed, maxLeafSize, ladder }
//               auto: cluster only when the graph has no hierarchy of its own; always: cluster
//               even then; planted: the generator's ground truth, or the groups of the file as
//               the top level with computed communities below (else like auto).
//               method / resolution / … : src/data/hierarchy.js (default Leiden, resolution 1).
//      baseKey  identifies the source: the parsed graph is cached under it (the last 2), so a
//               request that differs only in `cluster` skips fetch and parse
//      key      baseKey plus the cluster options: the clustered graph is cached under it (the
//               last 4), so a repeated request (new palette) skips the clustering too
//      positions  Float32Array( 4N ): start positions for the prepared graph (prepareGraph
//               options.initPos), e.g. a readback, so a new hierarchy morphs in place. An
//               array of another length is ignored (meta.repositioned false, a note in meta)
//      limits   { maxNodes, maxEdges } (default DEFAULT_LIMITS of builder.js), maxBytes (of a
//               file or download, packed and unpacked; default 256 MB)
//      coalesce true: when a newer load has arrived by the time this one would start, it is
//               skipped (reply 'skipped'); for requests that follow a slider
//  main → worker  { type: 'clear-cache' }
//  main → worker  { id, type: 'layout', kind: 'stress' | 'radial', graph, options }
//      graph    { N, E, edges, adjStart, degree, adjList, linkDistance, layoutRadius } of a
//               PreparedGraph (structured-cloned: the main thread keeps its arrays)
//      options  stress: { pivots, iters, eps, seed, dim, radius }   (sgd2Layout, graph/layouts.js)
//               radial: { root, dim, radius, shellGap, minSpacing } (radialLayout)
//  main → worker  { id, type: 'bundle', graphId, graph?, positions, mode, options }
//      graph    { N, E, D, edges, edgeLca, anc } of the PreparedGraph, sent with the first
//               request for a graphId and kept here for the following ones
//      mode / options: src/data/bundle.js. Runs in slices: a newer 'bundle', a 'bundle-cancel',
//      a 'load' or a 'layout' ends it (reply 'bundle-result' with cancelled: true).
//  main → worker  { type: 'bundle-cancel', id }
//  main → worker  { id, type: 'export', key, baseKey, cluster, source?, format: 'json' | 'graphml' |
//                   'csv', positions? }
//      serialises the cached graph (src/data/export.js); 'need-source' when it is not cached
//      and the message has no source
//  worker → main  { id, type: 'progress', stage, fraction, message }   fraction: 0..1 overall
//                 { id, type: 'result', graph, meta, timings }          graph: PreparedGraph
//                 { id, type: 'need-source' } | { id, type: 'skipped' }
//                 { id, type: 'layout-result', kind, positions, depth, info, timings }
//                                positions: Float32Array( 4N ) (transferred), depth: radial only
//                 { id, type: 'bundle-progress', phase, fraction, completed, total }
//                 { id, type: 'bundle-result', cancelled, paths, load, stats, timings }
//                 { id, type: 'export-result', text, mime, extension, name }
//                 { id, type: 'error', message, code }   code: 'parse' | 'format' | 'empty' |
//                                'too-large' (GraphError, builder.js) or undefined
//  Stages: fetch | read | parse | generate | cluster | prepare | transfer.
//
// ─── Formats ────────────────────────────────────────────────────────────────────────────────
//  src/data/formats.js: GraphML, GEXF, GML, DOT, Pajek, Matrix Market, edge lists and the JSON
//  shapes, detected by content first and extension second. Gzip is recognised by its first two
//  bytes (with or without .gz in the name) and unpacked with DecompressionStream, under the
//  same size limit as the file itself (256 MB).
//
// ─── meta ───────────────────────────────────────────────────────────────────────────────────
//  { key, kind, format, name, description, source, nodes, edges, levels, clustersPerLevel,
//    hierarchy: { method, … } | null, clustering (= graph.clustering), generator, stats,
//    file: the parser's report ({ format, directed, mirrored, weighted, dropped: { selfLoops,
//          duplicates, invalid }, labels, attributes, droppedAttributes, groups, warnings, … })
//          plus format details (delimiter, header, …), or null for generated graphs,
//    weighted, attributes: [ { name, type } ],
//    simplified: true when the graph on screen is not literally the file (directions merged,
//          duplicates merged, self loops or edges dropped, group nodes folded),
//    notes: string[]  sentences for the user about the above and about dropped attributes,
//    limits: { maxNodes, maxEdges }, timings: { fetch | read, parse | generate, cluster,
//    prepare, total } in ms, cached }
//
// ─── Exports (for inline use / tests) ───────────────────────────────────────────────────────
//  runRequest( msg, onProgress ) -> Promise<{ graph, meta, timings }>
//  runLayout( msg ) -> { kind, positions, depth, info, timings }      static layout request
//  createPort( post ) -> { receive( msg ) }   one client's connection: the message queue, the
//                 coalescing of re-clusterings and the bundle job live here
//  handleMessage( msg, post )     one message, no queue (post( data, transfer ) gets the replies)
//  parseGraphText( text, name, context ) -> { graph, format }
//  transferables( preparedGraph ) -> ArrayBuffer[]
//  NeedSource

import { parseGraph, baseName } from './formats.js';
import { DEFAULT_LIMITS, tooLarge, GraphError } from './builder.js';
import { generate } from './generators.js';
import { applyClustering, describeHierarchy, CLUSTER_DEFAULTS } from './hierarchy.js';
import { runBundle, defaultYield } from './bundle.js';
import { exportGraph } from './export.js';
import { prepareGraph } from '../graph/prepare.js';
import { sgd2Layout, radialLayout } from '../graph/layouts.js';

const PARSED_CACHE = 2, CLUSTERED_CACHE = 4;
const parsedCache = new Map(); // baseKey -> { graph, format }
const clusteredCache = new Map(); // key -> canonical graph with hierarchy and clustering

/** Largest text the worker reads (a string of this many characters still fits a JS engine). */
const MAX_BYTES = 256 * 1024 * 1024;

const PLANS = {
	url: [ [ 'fetch', 30 ], [ 'parse', 15 ], [ 'cluster', 30 ], [ 'prepare', 20 ], [ 'transfer', 5 ] ],
	file: [ [ 'read', 15 ], [ 'parse', 25 ], [ 'cluster', 30 ], [ 'prepare', 25 ], [ 'transfer', 5 ] ],
	text: [ [ 'parse', 30 ], [ 'cluster', 35 ], [ 'prepare', 30 ], [ 'transfer', 5 ] ],
	generate: [ [ 'generate', 20 ], [ 'cluster', 40 ], [ 'prepare', 35 ], [ 'transfer', 5 ] ],
	recluster: [ [ 'cluster', 60 ], [ 'prepare', 35 ], [ 'transfer', 5 ] ],
	cached: [ [ 'prepare', 90 ], [ 'transfer', 10 ] ],
};

/** Thrown by runRequest when the request has no source and the graph is not cached. */
export class NeedSource extends Error {

	constructor() {

		super( 'The graph is no longer cached; the source is needed again' );
		this.name = 'NeedSource';

	}

}

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

function remember( cache, key, value, size ) {

	cache.delete( key );
	cache.set( key, value );
	while ( cache.size > size ) cache.delete( cache.keys().next().value );

}

const count = ( x ) => Number( x ).toLocaleString( 'en-US' );

/** Parses text by content and name (src/data/formats.js). */
export function parseGraphText( text, name = 'graph', context = {} ) {

	const { graph, format } = parseGraph( text, name, context );
	return { graph, format };

}

/** Reads a stream to the end in chunks, with progress and the size limit. */
async function readChunks( stream, total, progress, label, what, maxBytes ) {

	const reader = stream.getReader();
	const chunks = [];
	let bytes = 0;
	for ( ;; ) {

		const { done, value } = await reader.read();
		if ( done ) break;
		chunks.push( value );
		bytes += value.byteLength;
		if ( bytes > maxBytes ) {

			reader.cancel().catch( () => {} );
			throw new GraphError( `This file ${what} more than ${formatBytes( maxBytes )}, the most the viewer reads`, 'too-large' );

		}

		if ( total ) progress.update( Math.min( 1, bytes / total ), `${label} ${formatBytes( bytes )} / ${formatBytes( total )}` );
		else progress.update( 0.5, `${label} ${formatBytes( bytes )}` );

	}

	return { chunks, bytes };

}

/**
 * The text of a stream. Gzip is recognised by its first two bytes (1f 8b), whatever the name
 * says, and unpacked chunk by chunk under the same size limit (an archive that unpacks to
 * gigabytes is stopped at the limit, not after it filled the memory).
 */
async function readStream( stream, total, name, progress, label, maxBytes ) {

	const { chunks, bytes } = await readChunks( stream, total, progress, label, 'is', maxBytes );
	let blob = new Blob( chunks );
	const head = new Uint8Array( await blob.slice( 0, 2 ).arrayBuffer() );
	if ( head.length === 2 && head[ 0 ] === 0x1f && head[ 1 ] === 0x8b ) {

		if ( typeof DecompressionStream === 'undefined' ) throw new GraphError( `${fileName( name )} is gzipped, which this browser cannot unpack. Unpack it first.`, 'format' );
		let unpacked;
		try {

			unpacked = await readChunks( blob.stream().pipeThrough( new DecompressionStream( 'gzip' ) ), 0, progress, 'Unpacking', 'unpacks to', maxBytes );

		} catch ( e ) {

			if ( e instanceof GraphError ) throw e;
			throw new GraphError( `${fileName( name )} cannot be unpacked: the gzip data is damaged or cut off`, 'parse' );

		}

		blob = new Blob( unpacked.chunks );

	}

	return { text: await blob.text(), bytes };

}

/** The last path segment of a file name or URL, with its extensions. */
function fileName( name ) {

	return String( name ).split( /[?#]/ )[ 0 ].split( /[\\/]/ ).pop() || 'The file';

}

async function loadSource( source, progress, timings, limits ) {

	let t = now();
	if ( source.kind === 'generate' ) {

		progress.stage( 'generate', `Generating ${source.spec.model}…` );
		const wanted = source.spec.params && Number( source.spec.params.n );
		if ( wanted > limits.maxNodes ) throw tooLarge( 'nodes', wanted, limits );
		const graph = generate( source.spec );
		if ( graph.nodeCount > limits.maxNodes ) throw tooLarge( 'nodes', graph.nodeCount, limits );
		if ( graph.edges.length / 2 > limits.maxEdges ) throw tooLarge( 'edges', graph.edges.length / 2, limits );
		timings.generate = now() - t;
		return { graph, format: 'generated' };

	}

	let text, name;
	const maxBytes = limits.maxBytes > 0 ? limits.maxBytes : MAX_BYTES;
	if ( source.kind === 'url' ) {

		name = source.url;
		progress.stage( 'fetch', `Downloading ${baseName( name )}…` );
		let res;
		try {

			res = await fetch( source.url );

		} catch ( e ) {

			// a network failure or a server that does not allow the page to read it (CORS)
			throw new Error( `Could not download ${fileName( name )}: the server did not answer, or it does not allow this page to read the file` );

		}

		if ( ! res.ok ) throw new Error( `Could not download ${fileName( name )}: the server answered ${res.status}${res.statusText ? ' ' + res.statusText : ''}` );
		const encoded = !! res.headers.get( 'content-encoding' );
		const total = encoded ? 0 : Number( res.headers.get( 'content-length' ) ) || 0;
		( { text } = await readStream( res.body, total, name, progress, 'Downloading', maxBytes ) );
		timings.fetch = now() - t;

	} else if ( source.kind === 'file' ) {

		const file = source.file;
		name = file.name || source.name || 'file';
		if ( file.size > maxBytes ) throw new GraphError( `${name} is ${formatBytes( file.size )}; the viewer reads files up to ${formatBytes( maxBytes )}`, 'too-large' );
		progress.stage( 'read', `Reading ${name}…` );
		( { text } = await readStream( file.stream(), file.size, name, progress, 'Reading', maxBytes ) );
		timings.read = now() - t;

	} else if ( source.kind === 'text' ) {

		name = source.name || 'graph.txt';
		text = source.text;

	} else throw new Error( `Unknown source kind "${source.kind}"` );

	t = now();
	progress.stage( 'parse', `Parsing ${baseName( name )}…` );
	const parsed = parseGraphText( text, name, {
		limits,
		onProgress: ( f ) => progress.update( f, `Parsing ${baseName( name )}… ${Math.round( f * 100 )}%` ),
	} );
	timings.parse = now() - t;
	return parsed;

}

function cluster( graph, opts, progress, timings ) {

	const mode = opts.mode || 'auto';
	const t = now();
	if ( mode === 'never' ) return { ...graph, hierarchy: null, clustering: null };
	const fileLevels = graph.hierarchy && graph.hierarchy.levels && graph.hierarchy.levels.length ? graph.hierarchy.levels : null;
	const planted = graph.planted && graph.planted.levels && graph.planted.levels.length ? graph.planted : null;
	if ( fileLevels && ( mode === 'auto' || ( mode === 'planted' && ! planted ) ) ) {

		// the hierarchy that came with the file; meta.hierarchy says how it was made, when it does
		const stored = ( graph.meta && graph.meta.hierarchy ) || {};
		const info = describeHierarchy( graph, fileLevels, {
			method: stored.algorithm || 'file', detail: graph.hierarchy.method || 'file',
			resolution: stored.resolution ?? 1, seed: stored.seed ?? null, maxLeafSize: stored.maxLeafSize ?? null,
			ladder: !! stored.ladder, kinds: stored.kinds, mainLevel: stored.mainLevel, mainCount: stored.communities || 0,
		} );
		return { ...graph, clustering: info };

	}

	const method = opts.method || CLUSTER_DEFAULTS.method;
	const title = method === 'louvain' ? 'Louvain' : 'Leiden';
	progress.stage( 'cluster', `Clustering ${count( graph.nodeCount )} nodes (${title})…` );
	let level = 0;
	const out = applyClustering( graph, {
		method,
		seed: opts.seed ?? 1,
		resolution: opts.resolution ?? 1,
		maxLeafSize: opts.maxLeafSize,
		ladder: opts.ladder,
		// generator ground truth is a complete hierarchy; the groups of a file are its top level
		planted: mode === 'planted' && planted ? { levels: planted.levels, complete: planted.complete !== false, source: planted.source } : null,
		onProgress: ( p ) => {

			if ( p.phase === 'ladder' ) {

				progress.update( 0.97, 'Shaping the hierarchy…', true );
				return;

			}

			if ( p.phase === 'refine' ) {

				progress.update( 0.8 + 0.15 * ( p.pass / Math.max( 1, p.communities ) ), `Refining clusters ${count( p.pass )}/${count( p.communities )}…` );
				return;

			}

			level = Math.max( level, p.level );
			// levels shrink geometrically: level l covers roughly the next half of the remaining work
			const f = 1 - Math.pow( 0.5, level + Math.min( 1, p.pass / 12 ) );
			progress.update( 0.8 * f, `${title} level ${p.level + 1}, pass ${p.pass}${Number.isFinite( p.modularity ) ? ` (Q = ${p.modularity.toFixed( 3 )})` : ''}` );

		},
	} );
	timings.cluster = now() - t;
	if ( graph.planted ) Object.defineProperty( out, 'planted', { value: graph.planted, enumerable: false, writable: true } );
	return out;

}

function copyAttributes( attributes ) {

	const out = {};
	for ( const name in attributes ) {

		const a = attributes[ name ];
		out[ name ] = a.type === 'category' ? { type: 'category', values: Int32Array.from( a.values ), categories: a.categories.slice() } : { type: 'number', values: Float32Array.from( a.values ) };

	}

	return out;

}

/** Sentences for the user about what differs between the file and the graph on screen. */
function describeChanges( report, clustering ) {

	const notes = [];
	let simplified = false;
	if ( report ) {

		const d = report.dropped || {};
		if ( report.mirrored ) notes.push( 'Every edge is listed in both directions; each pair is one undirected edge.' );
		else if ( report.directed ) {

			notes.push( d.duplicates > 0 ? `Directed graph, shown undirected: ${count( d.duplicates )} edges that run both ways or repeat were merged and their weights added.` : 'Directed graph, shown undirected.' );
			simplified = true;

		} else if ( d.duplicates > 0 ) {

			notes.push( `${count( d.duplicates )} duplicate edge${d.duplicates === 1 ? '' : 's'} merged (weights added).` );
			simplified = true;

		}

		if ( d.selfLoops > 0 ) {

			notes.push( `${count( d.selfLoops )} self loop${d.selfLoops === 1 ? '' : 's'} dropped.` );
			simplified = true;

		}

		if ( d.invalid > 0 ) {

			notes.push( `${count( d.invalid )} edge${d.invalid === 1 ? '' : 's'} with an unknown end point dropped.` );
			simplified = true;

		}

		if ( report.containers > 0 ) {

			notes.push( `${count( report.containers )} group node${report.containers === 1 ? '' : 's'} without edges became groups.` );
			simplified = true;

		}

		const lost = report.droppedAttributes || [];
		lost.slice( 0, 4 ).forEach( ( a ) => notes.push( `Attribute "${a.name}" not kept: ${a.reason}.` ) );
		if ( lost.length > 4 ) notes.push( `${lost.length - 4} more attributes not kept.` );
		( report.warnings || [] ).slice( 0, 4 ).forEach( ( w ) => notes.push( w.charAt( 0 ).toUpperCase() + w.slice( 1 ) + ( /[.!?]$/.test( w ) ? '' : '.' ) ) );
		if ( report.warningCount > 4 ) notes.push( `${count( report.warningCount - 4 )} more warnings.` );

	}

	if ( clustering && clustering.dust > 0 ) notes.push( `${count( clustering.dust )} isolated nodes and pairs are collected in "dust" clusters.` );
	return { notes, simplified };

}

/**
 * The canonical graph of a request, clustered as asked: from the caches, or loaded from the
 * source (and cached). Throws NeedSource when it is not cached and the request has no source.
 */
async function obtainCanonical( msg, onProgress, timings ) {

	const key = msg.key || null;
	const baseKey = msg.baseKey || key;
	const source = msg.source || null;
	const limits = { ...DEFAULT_LIMITS, ...( msg.limits || {} ) };
	let canonical = key ? clusteredCache.get( key ) : null;
	let parsed = ! canonical && baseKey ? parsedCache.get( baseKey ) : null;
	const hit = !! canonical;
	if ( ! canonical && ! parsed && ! source ) throw new NeedSource();
	const progress = new Progress( PLANS[ canonical ? 'cached' : parsed ? 'recluster' : source.kind ] || PLANS.text, onProgress );
	if ( canonical ) remember( clusteredCache, key, canonical, CLUSTERED_CACHE );
	else {

		if ( parsed ) remember( parsedCache, baseKey, parsed, PARSED_CACHE );
		else {

			parsed = await loadSource( source, progress, timings, limits );
			if ( baseKey ) remember( parsedCache, baseKey, parsed, PARSED_CACHE );

		}

		canonical = cluster( parsed.graph, msg.cluster || {}, progress, timings );
		Object.defineProperty( canonical, '__format', { value: parsed.format, enumerable: false, configurable: true } );
		if ( key ) remember( clusteredCache, key, canonical, CLUSTERED_CACHE );

	}

	return { canonical, progress, hit, limits };

}

/** Runs one load request; resolves to { graph (PreparedGraph), meta, timings }. */
export async function runRequest( msg, onProgress ) {

	const t0 = now();
	const timings = {};
	const key = msg.key || null;
	const source = msg.source || null;
	const { canonical, progress, hit, limits } = await obtainCanonical( msg, onProgress, timings );

	const format = canonical.__format;
	const t = now();
	progress.stage( 'prepare', 'Preparing GPU buffers…' );
	// the prepared graph's arrays are transferred: copy what the cached graph must keep
	// start positions only when they are for exactly these nodes (anything else is a readback
	// of another graph: taking its first 4N floats would scatter this one at random)
	let positions = msg.positions && msg.positions.length === 4 * canonical.nodeCount ? msg.positions : null;
	const mismatch = msg.positions && ! positions ? msg.positions.length >> 2 : - 1;
	if ( ! positions && canonical.positions ) {

		// a layout saved with the file
		positions = new Float32Array( 4 * canonical.nodeCount );
		for ( let i = 0; i < canonical.nodeCount; i ++ ) for ( let k = 0; k < 3; k ++ ) positions[ 4 * i + k ] = canonical.positions[ 3 * i + k ];

	}

	const graph = prepareGraph( {
		...canonical,
		edges: Uint32Array.from( canonical.edges ),
		weights: canonical.weights ? Float32Array.from( canonical.weights ) : null,
		attributes: copyAttributes( canonical.attributes || {} ),
	}, { ...( msg.prepareOptions || {} ), ...( positions ? { initPos: positions } : {} ) } );
	timings.prepare = now() - t;

	const h = canonical.hierarchy;
	const { report, hierarchy: storedHierarchy, ...fileMeta } = canonical.meta || {}; // eslint-disable-line no-unused-vars
	const changes = describeChanges( report || null, canonical.clustering );
	if ( mismatch >= 0 ) changes.notes.push( `The start positions given were for ${count( mismatch )} nodes, this graph has ${count( canonical.nodeCount )}: the layout starts from its own seed.` );
	timings.total = now() - t0;
	const meta = {
		key,
		kind: source ? source.kind : null,
		format,
		name: canonical.name,
		description: canonical.description,
		source: canonical.source,
		nodes: graph.N,
		edges: graph.E,
		levels: graph.D,
		clustersPerLevel: graph.stats.clustersPerLevel,
		hierarchy: h ? { method: h.method, ...( canonical.meta?.hierarchy || {} ) } : null,
		clustering: canonical.clustering || null,
		generator: canonical.meta?.generator ? { model: canonical.meta.generator, params: canonical.meta.params, seed: canonical.meta.seed } : null,
		stats: canonical.meta?.stats || null,
		file: report ? { ...fileMeta, ...report } : null,
		weighted: !! canonical.weights,
		attributes: Object.entries( canonical.attributes || {} ).map( ( [ name, a ] ) => ( a.type === 'category' ? { name, type: 'category', categories: a.categories.length } : { name, type: 'number' } ) ),
		simplified: changes.simplified,
		notes: changes.notes,
		limits,
		timings: Object.fromEntries( Object.entries( timings ).map( ( [ k, v ] ) => [ k, Math.round( v * 10 ) / 10 ] ) ),
		cached: hit,
		repositioned: !! positions,
	};
	graph.meta = meta;
	progress.stage( 'transfer', 'Transferring…' );
	timings.total = now() - t0;
	return { graph, meta, timings };

}

/** Every distinct ArrayBuffer of the prepared graph: its typed arrays and those of its attributes. */
export function transferables( graph ) {

	const set = new Set();
	const add = ( v ) => {

		if ( ArrayBuffer.isView( v ) && v.buffer instanceof ArrayBuffer ) set.add( v.buffer );

	};

	for ( const v of Object.values( graph ) ) add( v );
	if ( graph.attributes ) for ( const a of Object.values( graph.attributes ) ) if ( a ) add( a.values );
	return [ ...set ];

}

/**
 * Static layout request ({ kind, graph, options }, see the protocol): the sparse stress layout
 * ('stress', seconds on large graphs, the reason it runs here) or the radial focus layout
 * ('radial'; cheap, offered for symmetry).
 */
export function runLayout( msg ) {

	const g = msg.graph, options = msg.options || {};
	if ( ! g || ! Number.isInteger( g.N ) || ! g.adjStart || ! g.degree || ! g.adjList ) throw new Error( 'layout: graph needs N, adjStart, degree, adjList' );
	const t0 = now();
	let r;
	if ( msg.kind === 'stress' ) {

		if ( ! g.edges || ! Number.isInteger( g.E ) ) throw new Error( 'layout: the stress layout needs E and edges' );
		r = sgd2Layout( g, options );

	} else if ( msg.kind === 'radial' ) {

		const root = Number.isInteger( options.root ) && options.root >= 0 && options.root < g.N ? options.root : 0;
		r = radialLayout( g, root, options );

	} else throw new Error( `layout: unknown kind "${msg.kind}"` );

	return { kind: msg.kind, positions: r.positions, depth: r.depth || null, info: r.info, timings: { compute: now() - t0 } };

}

function errorReply( id, e ) {

	return { id, type: 'error', message: ( e && e.message ) || String( e ), code: e && e.code ? e.code : undefined };

}

async function handleLoad( msg, post ) {

	const id = msg.id;
	try {

		const result = await runRequest( msg, ( p ) => post( { id, type: 'progress', ...p } ) );
		post( { id, type: 'result', graph: result.graph, meta: result.meta, timings: result.timings }, transferables( result.graph ) );

	} catch ( e ) {

		post( e instanceof NeedSource ? { id, type: 'need-source' } : errorReply( id, e ) );

	}

}

function handleLayout( msg, post ) {

	try {

		const r = runLayout( msg );
		post( { id: msg.id, type: 'layout-result', ...r }, [ r.positions.buffer, ...( r.depth ? [ r.depth.buffer ] : [] ) ] );

	} catch ( e ) {

		post( errorReply( msg.id, e ) );

	}

}

async function handleExport( msg, post ) {

	try {

		// from the caches; with a source in the message the graph is loaded again when it is gone
		const { canonical } = await obtainCanonical( msg, null, {} );
		const out = exportGraph( canonical, msg.format, { positions: msg.positions || null } );
		post( { id: msg.id, type: 'export-result', ...out, name: canonical.name } );

	} catch ( e ) {

		post( e instanceof NeedSource ? { id: msg.id, type: 'need-source' } : errorReply( msg.id, e ) );

	}

}

/**
 * One client's connection to the pipeline. Requests run one at a time in arrival order; what
 * needs the newest state of the connection is decided here:
 *  · a load marked `coalesce` is skipped when a newer load arrived before it started (the
 *    queue takes a turn of the event loop before each load, so waiting messages are seen);
 *  · a bundle runs in slices and ends early when a newer bundle, a cancel, a load or a layout
 *    request arrives: its result would be for an old graph or old positions.
 */
export function createPort( post ) {

	let queue = Promise.resolve();
	let latestLoad = 0;
	let bundle = null; // the job in flight or waiting: { id, cancelled }
	let held = null; // { graphId, graph } for bundling
	const cancelBundle = () => {

		if ( bundle ) bundle.cancelled = true;

	};

	const runBundleJob = async ( msg, job ) => {

		const id = msg.id;
		try {

			if ( msg.graph ) held = { graphId: msg.graphId, graph: msg.graph };
			if ( job.cancelled ) {

				post( { id, type: 'bundle-result', cancelled: true } );
				return;

			}

			if ( ! held || held.graphId !== msg.graphId ) throw new Error( 'bundle: the graph was not sent with the request' );
			const r = await runBundle( held.graph, msg.positions, {
				mode: msg.mode || 'paths',
				options: msg.options || {},
				isCancelled: () => job.cancelled,
				onProgress: ( p ) => post( { id, type: 'bundle-progress', ...p } ),
			} );
			if ( r.cancelled || job.cancelled ) post( { id, type: 'bundle-result', cancelled: true } );
			else post( { id, type: 'bundle-result', ...r }, [ r.paths.pathStart.buffer, r.paths.pathRefs.buffer, r.load.buffer ] );

		} catch ( e ) {

			post( errorReply( id, e ) );

		} finally {

			if ( bundle === job ) bundle = null;

		}

	};

	return {
		receive( msg ) {

			if ( ! msg || typeof msg !== 'object' ) return;
			switch ( msg.type ) {

				case 'clear-cache':
					parsedCache.clear();
					clusteredCache.clear();
					break;
				case 'bundle-cancel':
					if ( bundle && ( msg.id === undefined || bundle.id === msg.id ) ) bundle.cancelled = true;
					break;
				case 'load':
					latestLoad = msg.id;
					cancelBundle();
					queue = queue.then( defaultYield ).then( () => {

						if ( msg.coalesce && msg.id !== latestLoad ) {

							post( { id: msg.id, type: 'skipped' } );
							return undefined;

						}

						return handleLoad( msg, post );

					} );
					break;
				case 'layout':
					cancelBundle();
					queue = queue.then( () => handleLayout( msg, post ) );
					break;
				case 'bundle': {

					cancelBundle();
					const job = bundle = { id: msg.id, cancelled: false };
					queue = queue.then( () => runBundleJob( msg, job ) );
					break;

				}

				case 'export':
					queue = queue.then( () => handleExport( msg, post ) );
					break;
				default:

			}

		},
	};

}

/** Handles one protocol message on its own (no queue); post( data, transfer? ) sends the replies. */
export async function handleMessage( msg, post ) {

	if ( ! msg || typeof msg !== 'object' ) return;
	if ( msg.type === 'clear-cache' ) {

		parsedCache.clear();
		clusteredCache.clear();

	} else if ( msg.type === 'layout' ) handleLayout( msg, post );
	else if ( msg.type === 'load' ) await handleLoad( msg, post );
	else if ( msg.type === 'export' ) await handleExport( msg, post );

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

	const port = createPort( ( data, transfer ) => self.postMessage( data, transfer || [] ) );
	self.onmessage = ( e ) => port.receive( e.data );

}
