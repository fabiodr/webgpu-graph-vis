// Main-thread API of the data pipeline. Loading, parsing, generation, clustering, prepareGraph,
// static layouts and edge bundling run in a module Web Worker (src/data/worker.js); results
// arrive as PreparedGraph objects (typed arrays transferred, not copied) ready for app.setGraph().
//
//   const data = new DataClient();
//   const off = data.onProgress( ( { stage, fraction, message } ) => panel.setLoading( message, fraction ) );
//   const r = await data.load( manifestEntry, { palette: settings.palette } );
//   if ( r.cancelled ) return;            // a newer request superseded this one
//   app.setGraph( r.graph );              // r.meta: { name, nodes, edges, levels, clustering, file, notes, … }
//
// Requests: every loading call returns a Promise of
//   { cancelled: false, graph: PreparedGraph (graph.meta = meta), meta, timings }  or  { cancelled: true }
// and rejects only on real errors (network, parse, unknown model, too large). An error that
// comes from the file is a GraphError in the worker: error.code is 'parse' | 'format' | 'empty' |
// 'too-large' and error.message is written for the user. A new request supersedes the one in
// flight: the older promise resolves { cancelled: true } immediately and the worker is
// terminated (stopping a long clustering run at once) and recreated lazily, so results never
// race. The exception is recluster() after recluster(): the worker is kept (it holds the parsed
// graph) and skips the stale request.
//
// API
//   new DataClient( { baseUrl?, createWorker?, inline?, limits? } )
//                                                  baseUrl: what relative URLs (manifest "file"
//                                                  paths) resolve against; default document.baseURI.
//                                                  inline: run on the main thread (also the
//                                                  automatic fallback where Worker is missing).
//                                                  limits: { maxNodes, maxEdges, maxBytes } for
//                                                  every request (default 250,000 nodes, 1,000,000
//                                                  edges, builder.js; 256 MB per file, packed and
//                                                  unpacked, worker.js)
//   loadUrl( url, prepareOptions?, options? )      resolved on the main thread (workers resolve
//                                                  relative URLs against their own script URL)
//   loadFile( file, prepareOptions?, options? )    File/Blob from drag & drop or <input type=file>
//   loadText( text, name, prepareOptions?, options? )
//   generate( { model, params, seed, labels?, name?, description? }, prepareOptions?, options? )
//   load( manifestEntry, prepareOptions?, options? )   entry.file -> loadUrl, entry.generate -> generate
//   reprepare( prepareOptions )                    the graph loaded last again with new prepare
//                                                  options (worker cache: skips fetch, parse and
//                                                  clustering); while a load is in flight, that
//                                                  load again
//   recluster( { method?, resolution?, seed?, mode?, maxLeafSize?, ladder? }, { positions?, ...prepareOptions } )
//                                                  the graph loaded last (the last load that
//                                                  succeeded: a failed or cancelled one changes
//                                                  nothing; while a request is in flight, that
//                                                  request with the new hierarchy options, and
//                                                  positions only if it is for the same source)
//                                                  with another hierarchy: same nodes
//                                                  and edges in the same order, no fetch and no
//                                                  parse (worker cache). Options left out keep
//                                                  their value; mode 'planted' of the load is
//                                                  kept (the file's groups stay on top), any
//                                                  other mode becomes 'always'. method:
//                                                  'leiden' | 'louvain'; resolution: 1 = default,
//                                                  larger = more and smaller communities, in
//                                                  quarter-octave steps (hierarchy.js); the result's
//                                                  graph.clustering says what was used:
//                                                  .communities never falls when the resolution
//                                                  rises and is the cluster count of level
//                                                  .mainLevel (less .dustClusters[ mainLevel ]),
//                                                  the level to show and to colour by.
//                                                  positions: Float32Array( 4N ) (a readback) become
//                                                  graph.initPos, so the layout morphs from where
//                                                  the nodes are (an array of another length is
//                                                  ignored: meta.repositioned is false and
//                                                  meta.notes says why). Resolves like a load;
//                                                  reports progress; a newer request supersedes it.
//   bundle( { positions, mode, options?, graph?, onProgress? } )
//                                                  edge-path bundling of the graph loaded last (or
//                                                  `graph`) at the given positions, in the worker
//                                                  (src/data/bundle.js). mode 'paths' | 'mixed'.
//                                                  positions: exactly 4N floats (anything else
//                                                  rejects: it is another graph's readback).
//                                                  options.timeBudget: ms, default 3,000 at every
//                                                  size, 0 = none; stats.outOfTime / stats.hurried
//                                                  say when the clock cut the run.
//                                                  Resolves { cancelled: true } or { cancelled: false,
//                                                  paths: { pathStart: Uint32Array( E + 1 ), pathRefs:
//                                                  Uint32Array, maxInterior }, load: Float32Array( E )
//                                                  (0..1), stats, timings }. The newest call wins:
//                                                  the previous one resolves cancelled, and so does
//                                                  a bundle when a load, recluster or layout request
//                                                  follows it, or when a load that was in flight
//                                                  delivers its graph (its input is stale then).
//                                                  onProgress( { phase, fraction, completed, total } )
//   cancelBundle()
//   export( 'json' | 'graphml' | 'csv', { positions? } ) -> Promise<Blob>
//                                                  the graph loaded last as a file (export.js);
//                                                  blob.suggestedName is a file name for it
//   formats() -> [ { id, name, extensions, description } ]   what loadFile / loadUrl / loadText read
//   acceptExtensions() -> string[]                 every extension of formats() and '.gz': the
//                                                  `accept` list of a file input
//   accepts( fileName ) -> boolean                 the test of a drop filter (formats.js
//                                                  acceptsName): a registered extension, also
//                                                  gzipped, or none
//   onProgress( cb ) -> unsubscribe                cb({ stage, fraction, message }) for the current
//                                                  request; stage 'done' (fraction 1) at the end
//   layout( kind, options? )                       static layout of a prepared graph, computed in
//                                                  the worker (graph/layouts.js):
//                                                    'stress'  sparse SGD² stress layout; options
//                                                              { graph, pivots = 'auto' (200 up to
//                                                              2000 nodes, 50 above), iters = 30,
//                                                              seed = 1, dim = 3, radius }
//                                                    'radial'  BFS shells around options.root
//                                                              { graph, root, dim = 3, radius }
//                                                  graph defaults to the last loaded graph. Resolves
//                                                  { cancelled: false, kind, positions: Float32Array
//                                                  ( 4N ) (xyz world units, w = 0), depth (radial),
//                                                  info, timings } or { cancelled: true } when a newer
//                                                  layout() call superseded it or the worker was
//                                                  terminated (a load superseding a load, cancel(),
//                                                  dispose()). Loads and layouts otherwise queue
//                                                  behind each other; neither cancels the other.
//   listModels()                                   generator descriptions for the UI form
//   cancel()                                       resolve the current request as cancelled
//   busy                                           true while a load / recluster is in flight
//   dispose()
// options.cluster: { mode: 'auto'|'always'|'never'|'planted', method: 'leiden'|'louvain', resolution,
//   seed, maxLeafSize, ladder } (see worker.js and hierarchy.js; default auto = Leiden with the
//   ladder when the graph has no hierarchy of its own, leaves ≤ 200 nodes).
// options.limits: { maxNodes, maxEdges, maxBytes } for this request.

import { listModels } from './generators.js';
import { listFormats, acceptExtensions, acceptsName } from './formats.js';

function defaultWorker() {

	return new Worker( new URL( './worker.js', import.meta.url ), { type: 'module' } );

}

/** Same message interface as a Worker, but runs the pipeline on this thread (fallback / tests). */
class InlineWorker {

	constructor() {

		this.onmessage = null;
		this.onerror = null;
		this._dead = false;
		this._port = import( './worker.js' ).then( ( m ) => m.createPort( ( data ) => {

			if ( this._dead ) return;
			Promise.resolve().then( () => this._dead || this.onmessage?.( { data } ) );

		} ) );

	}

	postMessage( msg ) {

		this._port.then( ( port ) => this._dead || port.receive( msg ) ).catch( ( e ) => this._dead || this.onerror?.( e ) );

	}

	terminate() {

		this._dead = true;

	}

}

function stableStringify( v ) {

	if ( v === null || typeof v !== 'object' ) return JSON.stringify( v );
	if ( Array.isArray( v ) ) return '[' + v.map( stableStringify ).join( ',' ) + ']';
	return '{' + Object.keys( v ).filter( ( k ) => v[ k ] !== undefined ).sort().map( ( k ) => JSON.stringify( k ) + ':' + stableStringify( v[ k ] ) ).join( ',' ) + '}';

}

function resolveUrl( url, baseUrl ) {

	let base = ( typeof document !== 'undefined' && document.baseURI ) || ( typeof location !== 'undefined' && location.href ) || undefined;
	try {

		if ( baseUrl ) base = new URL( baseUrl, base ).href;
		return new URL( url, base ).href;

	} catch ( e ) {

		return url;

	}

}

function toError( msg, fallback ) {

	const error = new Error( msg.message || fallback );
	if ( msg.code ) error.code = msg.code;
	return error;

}

export class DataClient {

	constructor( options = {} ) {

		this.options = { createWorker: null, inline: false, baseUrl: null, limits: null, ...options };
		this._worker = null;
		this._current = null;
		this._nextId = 1;
		this._listeners = new Set();
		this._last = null;
		this._layoutCurrent = null;
		this._bundleCurrent = null;
		this._bundleSent = null; // { worker, graph }: the worker holds this graph's arrays
		this._graphIds = new WeakMap();
		this._exports = new Map();
		this._graph = null; // the last graph a load returned (default subject of layout() and bundle())

	}

	get busy() {

		return !! this._current;

	}

	onProgress( callback ) {

		this._listeners.add( callback );
		return () => this._listeners.delete( callback );

	}

	listModels() {

		return listModels();

	}

	formats() {

		return listFormats();

	}

	acceptExtensions() {

		return acceptExtensions();

	}

	accepts( fileName ) {

		return acceptsName( fileName );

	}

	loadUrl( url, prepareOptions, options ) {

		const abs = resolveUrl( url, this.options.baseUrl );
		return this._request( { kind: 'url', url: abs }, `url:${abs}`, prepareOptions, options );

	}

	loadFile( file, prepareOptions, options ) {

		const key = `file:${file.name || 'blob'}:${file.size}:${file.lastModified || 0}`;
		return this._request( { kind: 'file', file }, key, prepareOptions, options );

	}

	loadText( text, name = 'graph.txt', prepareOptions, options ) {

		// cheap content key: length + sampled characters
		let h = 2166136261;
		const step = Math.max( 1, Math.floor( text.length / 4096 ) );
		for ( let i = 0; i < text.length; i += step ) h = Math.imul( h ^ text.charCodeAt( i ), 16777619 );
		return this._request( { kind: 'text', text, name }, `text:${name}:${text.length}:${h >>> 0}`, prepareOptions, options );

	}

	generate( spec, prepareOptions, options ) {

		if ( ! spec || ! spec.model ) return Promise.reject( new Error( 'generate: spec.model is required' ) );
		const s = {
			model: spec.model,
			params: { ...( spec.params || {} ) },
			seed: spec.seed ?? spec.params?.seed ?? 1,
			labels: spec.labels ?? 'auto',
			name: spec.name,
			description: spec.description,
		};
		delete s.params.seed;
		return this._request( { kind: 'generate', spec: s }, `gen:${stableStringify( s )}`, prepareOptions, options );

	}

	/** Loads a manifest entry: { file } (path relative to the app) or { generate: spec }. */
	load( entry, prepareOptions, options ) {

		if ( entry && entry.generate ) return this.generate( { name: entry.name, description: entry.description, ...entry.generate }, prepareOptions, options );
		if ( entry && ( entry.file || entry.url ) ) return this.loadUrl( entry.file || entry.url, prepareOptions, options );
		return Promise.reject( new Error( 'load: entry needs "file" or "generate"' ) );

	}

	reprepare( prepareOptions ) {

		// the request in flight (it starts again with the new options), else the graph on screen
		const last = this._current ? this._current.record : this._last;
		if ( ! last ) return Promise.reject( new Error( 'reprepare: nothing loaded yet' ) );
		const { source, key, options } = last;
		return this._request( source, key, prepareOptions, options );

	}

	/** The graph loaded last with another hierarchy (see the header). */
	recluster( cluster = {}, options = {} ) {

		// what it starts from: the request in flight (two quick changes, the method and then the
		// resolution, add up; a load that is still running gets the new hierarchy options), else
		// the graph on screen. Never a request that failed or was cancelled.
		const flying = this._current ? this._current.record : null;
		const last = flying || this._last;
		if ( ! last ) return Promise.reject( new Error( 'recluster: nothing loaded yet' ) );
		let { positions = null, ...prepareOptions } = options || {};
		// a readback is of the graph on screen: it says nothing about another source in flight
		if ( flying && ! ( this._last && this._last.key === flying.key ) ) positions = null;
		const before = ( last.options && last.options.cluster ) || {};
		// 'auto' and 'never' would give back the stored hierarchy or none; 'planted' stays, so the
		// groups of the file (or the generator's truth) remain the top of the hierarchy
		const mode = before.mode === 'planted' ? 'planted' : 'always';
		const merged = { ...( last.options || {} ), cluster: { ...before, mode, ...( cluster || {} ) } };
		return this._request( last.source, last.key, { ...( last.prepareOptions || {} ), ...prepareOptions }, merged, { positions, soft: true } );

	}

	/** Static layout of a prepared graph in the worker (see the header). */
	layout( kind = 'stress', options = {} ) {

		const { graph = this._graph, ...rest } = options || {};
		if ( ! graph ) return Promise.reject( new Error( 'layout: no graph (load one first or pass options.graph)' ) );
		const prev = this._layoutCurrent;
		if ( prev ) {

			// the worker still finishes the stale request (its reply is ignored)
			this._layoutCurrent = null;
			prev.resolve( { cancelled: true } );

		}

		const id = this._nextId ++;
		const g = {
			N: graph.N, E: graph.E, edges: graph.edges, adjStart: graph.adjStart, degree: graph.degree, adjList: graph.adjList,
			linkDistance: graph.linkDistance, layoutRadius: graph.layoutRadius,
		};
		return new Promise( ( resolve, reject ) => {

			this._layoutCurrent = { id, resolve, reject, t0: now() };
			try {

				this._ensureWorker().postMessage( { id, type: 'layout', kind, graph: g, options: rest } );

			} catch ( e ) {

				if ( this._layoutCurrent && this._layoutCurrent.id === id ) this._layoutCurrent = null;
				reject( e );

			}

		} );

	}

	/** Edge-path bundling in the worker (see the header). */
	bundle( spec = {} ) {

		const { positions, mode = 'paths', options = {}, graph = this._graph, onProgress = null } = spec || {};
		if ( ! graph ) return Promise.reject( new Error( 'bundle: no graph (load one first or pass graph)' ) );
		if ( ! positions || positions.length !== 4 * graph.N ) return Promise.reject( new Error( `bundle: positions must be a Float32Array( 4N ) (${4 * graph.N} floats${positions ? `, got ${positions.length}` : ''})` ) );
		if ( mode !== 'paths' && mode !== 'mixed' ) return Promise.reject( new Error( `bundle: unknown mode "${mode}" (paths | mixed)` ) );
		// newest wins: the worker ends the older job when this message arrives
		this._resolveBundle( { cancelled: true } );
		const id = this._nextId ++;
		return new Promise( ( resolve, reject ) => {

			this._bundleCurrent = { id, resolve, reject, t0: now(), onProgress };
			try {

				const worker = this._ensureWorker();
				let graphId = this._graphIds.get( graph );
				if ( ! graphId ) {

					graphId = this._nextId ++;
					this._graphIds.set( graph, graphId );

				}

				const known = this._bundleSent && this._bundleSent.worker === worker && this._bundleSent.graph === graph;
				const g = known ? undefined : { N: graph.N, E: graph.E, D: graph.D, edges: graph.edges, edgeLca: graph.edgeLca, anc: graph.anc };
				worker.postMessage( { id, type: 'bundle', graphId, graph: g, positions, mode, options: { ...options } } );
				this._bundleSent = { worker, graph };

			} catch ( e ) {

				if ( this._bundleCurrent && this._bundleCurrent.id === id ) this._bundleCurrent = null;
				reject( e );

			}

		} );

	}

	cancelBundle() {

		const cur = this._bundleCurrent;
		if ( ! cur ) return;
		if ( this._worker ) this._worker.postMessage( { type: 'bundle-cancel', id: cur.id } );
		this._resolveBundle( { cancelled: true } );

	}

	/** The graph loaded last as a file (see the header). */
	export( format = 'json', options = {} ) {

		if ( ! this._last ) return Promise.reject( new Error( 'export: nothing loaded yet' ) );
		if ( format !== 'json' && format !== 'graphml' && format !== 'csv' ) return Promise.reject( new Error( `export: unknown format "${format}" (json | graphml | csv)` ) );
		const last = this._last;
		const id = this._nextId ++;
		const message = { id, type: 'export', key: last.fullKey, baseKey: last.key, cluster: last.cluster, limits: last.limits, format, positions: ( options && options.positions ) || null };
		return new Promise( ( resolve, reject ) => {

			this._exports.set( id, { resolve, reject, message, source: last.source } );
			try {

				this._ensureWorker().postMessage( message );

			} catch ( e ) {

				this._exports.delete( id );
				reject( e );

			}

		} );

	}

	cancel() {

		this._supersede();

	}

	dispose() {

		this._supersede();
		this._terminate();
		this._listeners.clear();

	}

	// ── internals ──────────────────────────────────────────────────────────────────────────

	_request( source, key, prepareOptions = {}, options = {}, extra = {} ) {

		const soft = !! extra.soft;
		// a re-clustering that follows a re-clustering keeps the worker and its parsed graph
		const keepWorker = soft && !! this._current && this._current.soft;
		const hadWorker = !! this._worker;
		this._supersede( keepWorker );
		const id = this._nextId ++;
		const cluster = { mode: 'auto', ...( ( options && options.cluster ) || {} ) };
		const limits = ( options && options.limits ) || this.options.limits || undefined;
		const fullKey = `${key}|${stableStringify( cluster )}`;
		// becomes this._last when the result arrives: reprepare(), recluster() and export() act on
		// the graph that was loaded, never on a request that failed or was cancelled
		const record = { source, key, options, prepareOptions: { ...prepareOptions }, cluster, limits, fullKey };
		return new Promise( ( resolve, reject ) => {

			// the source travels only when the worker cannot have the graph (it asks otherwise)
			const withSource = ! soft || ! hadWorker || ! this._worker;
			const message = { id, type: 'load', key: fullKey, baseKey: key, source: withSource ? source : null, prepareOptions: { ...prepareOptions }, cluster, limits, positions: extra.positions || null, coalesce: soft };
			this._current = { id, resolve, reject, t0: now(), soft, message, source, record };
			try {

				// throws synchronously e.g. when prepareOptions are not structured-cloneable
				this._ensureWorker().postMessage( message );

			} catch ( e ) {

				if ( this._current && this._current.id === id ) this._current = null;
				reject( e );

			}

		} );

	}

	_supersede( keepWorker = false ) {

		// whatever follows changes the graph: a bundle in flight is for the old one
		this._resolveBundle( { cancelled: true } );
		const cur = this._current;
		if ( ! cur ) return;
		this._current = null;
		// stop the stale computation now; a fresh worker is created by the next request
		if ( ! keepWorker ) this._terminate();
		cur.resolve( { cancelled: true } );

	}

	_resolveBundle( result ) {

		const cur = this._bundleCurrent;
		if ( ! cur ) return;
		this._bundleCurrent = null;
		cur.resolve( result );

	}

	_ensureWorker() {

		if ( this._worker ) return this._worker;
		const inline = this.options.inline || ( ! this.options.createWorker && typeof Worker === 'undefined' );
		const w = inline ? new InlineWorker() : ( this.options.createWorker || defaultWorker )();
		w.onmessage = ( e ) => this._onMessage( e.data );
		w.onerror = ( e ) => {

			if ( e && e.preventDefault ) e.preventDefault();
			this._fail( new Error( ( e && e.message ) || 'Data worker failed to start' ) );

		};

		w.onmessageerror = () => this._fail( new Error( 'Data worker: message could not be deserialized' ) );
		this._worker = w;
		return w;

	}

	_terminate() {

		// layout, bundle and export requests die with their worker
		const lay = this._layoutCurrent;
		this._layoutCurrent = null;
		if ( lay ) lay.resolve( { cancelled: true } );
		this._resolveBundle( { cancelled: true } );
		this._bundleSent = null;
		const pending = [ ...this._exports.values() ];
		this._exports.clear();
		if ( ! this._worker ) {

			for ( const p of pending ) p.reject( new Error( 'export: cancelled' ) );
			return;

		}

		this._worker.onmessage = null;
		this._worker.onerror = null;
		this._worker.terminate();
		this._worker = null;
		for ( const p of pending ) p.reject( new Error( 'export: the data worker was restarted' ) );

	}

	_fail( error ) {

		const cur = this._current, lay = this._layoutCurrent, bun = this._bundleCurrent;
		this._layoutCurrent = null;
		this._bundleCurrent = null;
		this._current = null;
		this._terminate();
		if ( lay ) lay.reject( error );
		if ( bun ) bun.reject( error );
		if ( cur ) cur.reject( error );

	}

	_emit( p ) {

		for ( const cb of this._listeners ) {

			try {

				cb( p );

			} catch ( e ) {

				console.error( e );

			}

		}

	}

	_onMessage( msg ) {

		if ( ! msg ) return;
		const lay = this._layoutCurrent;
		if ( lay && msg.id === lay.id ) {

			this._layoutCurrent = null;
			if ( msg.type === 'layout-result' ) lay.resolve( { cancelled: false, kind: msg.kind, positions: msg.positions, depth: msg.depth, info: msg.info, timings: { ...msg.timings, roundTrip: now() - lay.t0 } } );
			else lay.reject( toError( msg, 'layout failed' ) );
			return;

		}

		const bun = this._bundleCurrent;
		if ( bun && msg.id === bun.id ) {

			if ( msg.type === 'bundle-progress' ) {

				if ( bun.onProgress ) {

					try {

						bun.onProgress( { phase: msg.phase, fraction: msg.fraction, completed: msg.completed, total: msg.total } );

					} catch ( e ) {

						console.error( e );

					}

				}

				return;

			}

			this._bundleCurrent = null;
			if ( msg.type === 'bundle-result' ) bun.resolve( msg.cancelled ? { cancelled: true } : { cancelled: false, paths: msg.paths, load: msg.load, stats: msg.stats, timings: { ...msg.timings, roundTrip: now() - bun.t0 } } );
			else bun.reject( toError( msg, 'bundle failed' ) );
			return;

		}

		const exp = this._exports.get( msg.id );
		if ( exp ) {

			if ( msg.type === 'need-source' && ! exp.message.source ) {

				// the worker was restarted since the load: it loads the graph again, silently
				exp.message.source = exp.source;
				this._worker.postMessage( exp.message );
				return;

			}

			this._exports.delete( msg.id );
			if ( msg.type === 'export-result' ) {

				const blob = new Blob( [ msg.text ], { type: msg.mime } );
				blob.suggestedName = `${String( msg.name || 'graph' ).replace( /[^\w.-]+/g, '_' ).replace( /^_+|_+$/g, '' ) || 'graph'}${msg.extension}`;
				exp.resolve( blob );

			} else exp.reject( toError( msg, 'export failed' ) );

			return;

		}

		const cur = this._current;
		if ( ! cur || msg.id !== cur.id ) return; // stale message of a superseded request
		if ( msg.type === 'progress' ) {

			this._emit( { stage: msg.stage, fraction: msg.fraction, message: msg.message } );

		} else if ( msg.type === 'need-source' ) {

			// the worker does not have the parsed graph (any more): the same request with the source
			if ( cur.message.source ) {

				this._current = null;
				cur.reject( new Error( 'Data worker: the source was sent but not used' ) );
				return;

			}

			cur.message.source = cur.source;
			this._worker.postMessage( cur.message );

		} else if ( msg.type === 'result' ) {

			this._current = null;
			const timings = { ...msg.timings, roundTrip: now() - cur.t0 };
			this._emit( { stage: 'done', fraction: 1, message: 'Ready' } );
			// a bundle asked for while this load was running would be for the graph before it
			if ( this._bundleCurrent ) {

				if ( this._worker ) this._worker.postMessage( { type: 'bundle-cancel', id: this._bundleCurrent.id } );
				this._resolveBundle( { cancelled: true } );

			}

			this._graph = msg.graph;
			this._last = cur.record;
			cur.resolve( { cancelled: false, graph: msg.graph, meta: msg.meta, timings } );

		} else if ( msg.type === 'error' ) {

			this._current = null;
			cur.reject( toError( msg, 'load failed' ) );

		}

	}

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}
