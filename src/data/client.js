// Main-thread API of the data pipeline. Loading, parsing, generation, Louvain clustering and
// prepareGraph run in a module Web Worker (src/data/worker.js); results arrive as PreparedGraph
// objects (typed arrays transferred, not copied) ready for app.setGraph().
//
//   const data = new DataClient();
//   const off = data.onProgress( ( { stage, fraction, message } ) => panel.setLoading( message, fraction ) );
//   const r = await data.load( manifestEntry, { palette: settings.palette } );
//   if ( r.cancelled ) return;            // a newer request superseded this one
//   app.setGraph( r.graph );              // r.meta: { name, nodes, edges, levels, hierarchy, generator, … }
//
// Requests: every call returns a Promise of
//   { cancelled: false, graph: PreparedGraph (graph.meta = meta), meta, timings }  or  { cancelled: true }
// and rejects only on real errors (network, parse, unknown model). A new request supersedes the
// one in flight: the older promise resolves { cancelled: true } immediately and the worker is
// terminated (stopping a long Louvain run at once) and recreated lazily, so results never race.
//
// API
//   new DataClient( { baseUrl?, createWorker?, inline? } )
//                                                  baseUrl: what relative URLs (manifest "file"
//                                                  paths) resolve against; default document.baseURI.
//                                                  inline: run on the main thread (also the
//                                                  automatic fallback where Worker is missing)
//   loadUrl( url, prepareOptions?, options? )      resolved on the main thread (workers resolve
//                                                  relative URLs against their own script URL)
//   loadFile( file, prepareOptions?, options? )    File/Blob from drag & drop or <input type=file>
//   loadText( text, name, prepareOptions?, options? )
//   generate( { model, params, seed, labels?, name?, description? }, prepareOptions?, options? )
//   load( manifestEntry, prepareOptions?, options? )   entry.file -> loadUrl, entry.generate -> generate
//   reprepare( prepareOptions )                    same source again with new prepare options
//                                                  (worker cache: skips fetch, parse and Louvain)
//   onProgress( cb ) -> unsubscribe                cb({ stage, fraction, message }) for the current
//                                                  request; stage 'done' (fraction 1) at the end
//   listModels()                                   generator descriptions for the UI form
//   cancel()                                       resolve the current request as cancelled
//   busy                                           true while a request is in flight
//   dispose()
// options.cluster: { mode: 'auto'|'always'|'never'|'planted', seed, resolution, maxLeafSize }
//   (see worker.js; default auto = Louvain when the graph has no hierarchy, leaves ≤ 200 nodes).

import { listModels } from './generators.js';

function defaultWorker() {

	return new Worker( new URL( './worker.js', import.meta.url ), { type: 'module' } );

}

/** Same message interface as a Worker, but runs the pipeline on this thread (fallback / tests). */
class InlineWorker {

	constructor() {

		this.onmessage = null;
		this.onerror = null;
		this._dead = false;
		this._module = import( './worker.js' );

	}

	postMessage( msg ) {

		this._module.then( ( m ) => m.handleMessage( msg, ( data ) => {

			if ( this._dead ) return;
			Promise.resolve().then( () => this._dead || this.onmessage?.( { data } ) );

		} ) ).catch( ( e ) => this._dead || this.onerror?.( e ) );

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

export class DataClient {

	constructor( options = {} ) {

		this.options = { createWorker: null, inline: false, baseUrl: null, ...options };
		this._worker = null;
		this._current = null;
		this._nextId = 1;
		this._listeners = new Set();
		this._last = null;

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

		if ( ! this._last ) return Promise.reject( new Error( 'reprepare: nothing loaded yet' ) );
		const { source, key, options } = this._last;
		return this._request( source, key, prepareOptions, options );

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

	_request( source, key, prepareOptions = {}, options = {} ) {

		this._supersede();
		const id = this._nextId ++;
		const cluster = { mode: 'auto', ...( options.cluster || {} ) };
		this._last = { source, key, options };
		return new Promise( ( resolve, reject ) => {

			this._current = { id, resolve, reject, t0: now() };
			try {

				// throws synchronously e.g. when prepareOptions are not structured-cloneable
				this._ensureWorker().postMessage( { id, type: 'load', key: `${key}|${stableStringify( cluster )}`, source, prepareOptions: { ...prepareOptions }, cluster } );

			} catch ( e ) {

				if ( this._current && this._current.id === id ) this._current = null;
				reject( e );

			}

		} );

	}

	_supersede() {

		const cur = this._current;
		if ( ! cur ) return;
		this._current = null;
		// stop the stale computation now; a fresh worker is created by the next request
		this._terminate();
		cur.resolve( { cancelled: true } );

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

		if ( ! this._worker ) return;
		this._worker.onmessage = null;
		this._worker.onerror = null;
		this._worker.terminate();
		this._worker = null;

	}

	_fail( error ) {

		const cur = this._current;
		this._terminate();
		if ( ! cur ) return;
		this._current = null;
		cur.reject( error );

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

		const cur = this._current;
		if ( ! msg || ! cur || msg.id !== cur.id ) return; // stale message of a superseded request
		if ( msg.type === 'progress' ) {

			this._emit( { stage: msg.stage, fraction: msg.fraction, message: msg.message } );

		} else if ( msg.type === 'result' ) {

			this._current = null;
			const timings = { ...msg.timings, roundTrip: now() - cur.t0 };
			this._emit( { stage: 'done', fraction: 1, message: 'Ready' } );
			cur.resolve( { cancelled: false, graph: msg.graph, meta: msg.meta, timings } );

		} else if ( msg.type === 'error' ) {

			this._current = null;
			cur.reject( new Error( msg.message ) );

		}

	}

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}
