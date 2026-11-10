// Main-thread API of the analytics worker (src/analysis/worker.js): node and edge metrics and
// the graph summary as promises, with progress and progressive previews.
//
//   const analysis = new AnalysisClient();
//   analysis.setGraph( app.graph );                               // once per graph (lazy: nothing
//                                                                 // is sent until the first request)
//   const r = await analysis.metric( 'pagerank' );                // { cancelled: false, values, … }
//   const b = await analysis.metric( 'betweenness', {
//       slot: 'color', options: { samples: 'exact' }, maxMs: 20000,
//       onProgress: ( { done, total, fraction } ) => bar.set( fraction ),
//       onPreview: ( r ) => show( r.values ),                     // coarse first, then refined
//   } );
//   const s = await analysis.summary( { onPreview: ( partial ) => panel.set( partial ) } );
//
// Requests
//   Every call returns a Promise of { cancelled: true, reason } or a result; it rejects only on
//   real errors (unknown metric, worker failure). Each request belongs to a `slot` (default
//   'default'): a NEWER request of the same slot supersedes the one in flight, whose promise
//   resolves { cancelled: true, reason: 'superseded' } at once. Requests of different slots run
//   side by side (the worker shares the work between them where it can). Requests may be made
//   while the worker is busy: they are queued by the worker and answered in order. setGraph()
//   resolves everything in flight as { cancelled: true, reason: 'graph' }; replies that arrive
//   for an older graph are dropped.
//
// API
//   new AnalysisClient( { createWorker?, inline?, preview = 200 } )
//        inline: run the handler on this thread (also the fallback where Worker is missing)
//        preview: default ms between progressive results of slow metrics
//   setGraph( graph | null )       PreparedGraph (N, E, edges, adjStart, degree, adjList, labels)
//   metric( id, { slot, target = 'node', options, preview, maxMs = 0, normalize = false,
//                 onProgress, onPreview } ) -> Promise<MetricResult>
//        MetricResult { cancelled: false, metric, target, values: Float32Array (N or E, raw),
//          normalized: Float32Array | null (when `normalize` was a mode or true), scale (the
//          catalogue's normalize() mode), final, exact, pivots, total, ms }
//        options.samples for slow metrics: 'auto' (default), a pivot count, or 'exact' (refine
//          until exact or until maxMs has passed; the promise then resolves with exact: false)
//   edgeMetric( id, options )      = metric( id, { ...options, target: 'edge' } )
//   summary( { top = 10, maxBfs = 32, slot = 'summary', onProgress, onPreview } ) -> Promise<Summary>
//        the summary object of worker.js with `cancelled: false`; every top list also carries
//        `labels`, and diameter carries `endpointLabels`
//   cancel( slot? )                resolve the request of a slot (or all) as cancelled
//   busy, pending                  requests in flight
//   dispose()

function defaultWorker() {

	return new Worker( new URL( './worker.js', import.meta.url ), { type: 'module' } );

}

/** Same message interface as a Worker, but runs the handler on this thread (fallback / tests). */
class InlineWorker {

	constructor() {

		this.onmessage = null;
		this.onerror = null;
		this._dead = false;
		this._handler = import( './worker.js' ).then( ( m ) => m.createHandler( {
			post: ( data ) => {

				if ( this._dead ) return;
				Promise.resolve().then( () => this._dead || this.onmessage?.( { data } ) );

			},
		} ) );

	}

	postMessage( msg ) {

		this._handler.then( ( handle ) => this._dead || handle( msg ) ).catch( ( e ) => this._dead || this.onerror?.( e ) );

	}

	terminate() {

		this._dead = true;

	}

}

function safely( fn, arg ) {

	if ( ! fn ) return;
	try {

		fn( arg );

	} catch ( e ) {

		console.error( e );

	}

}

export class AnalysisClient {

	constructor( options = {} ) {

		this.options = { createWorker: null, inline: false, preview: 200, ...options };
		this._worker = null;
		this._graph = null;
		this._graphId = 0;
		this._graphSent = false;
		this._pending = new Map(); // id -> { id, slot, resolve, reject, onProgress, onPreview }
		this._slots = new Map(); // slot -> id
		this._nextId = 1;

	}

	get pending() {

		return this._pending.size;

	}

	get busy() {

		return this._pending.size > 0;

	}

	/** The graph later requests refer to. Everything in flight resolves as cancelled. */
	setGraph( graph ) {

		this._settleAll( { cancelled: true, reason: 'graph' } );
		this._graph = graph && graph.edges && graph.adjList ? graph : null;
		this._graphId ++;
		this._graphSent = false;
		if ( this._worker ) this._post( { type: 'reset' } ); // stop the old graph's tasks now

	}

	metric( metric, opts = {} ) {

		const { slot = 'default', target = 'node', options = {}, preview = this.options.preview, maxMs = 0, normalize = false, onProgress, onPreview } = opts;
		return this._request( { type: 'metric', slot: String( slot ), target, metric, options, preview, maxMs, normalize }, onProgress, onPreview );

	}

	edgeMetric( metric, opts = {} ) {

		return this.metric( metric, { slot: 'edge', ...opts, target: 'edge' } );

	}

	summary( opts = {} ) {

		const { slot = 'summary', top = 10, maxBfs = 32, onProgress, onPreview } = opts;
		return this._request( { type: 'summary', slot: String( slot ), options: { top, maxBfs } }, onProgress, onPreview );

	}

	cancel( slot ) {

		if ( slot === undefined ) {

			this._settleAll( { cancelled: true, reason: 'cancel' } );
			if ( this._worker ) this._post( { type: 'cancel' } );
			return;

		}

		const id = this._slots.get( String( slot ) );
		if ( id === undefined ) return;
		this._settle( id, { cancelled: true, reason: 'cancel' } );
		if ( this._worker ) this._post( { type: 'cancel', id } );

	}

	dispose() {

		this._settleAll( { cancelled: true, reason: 'dispose' } );
		this._terminate();
		this._graph = null;

	}

	// ── internals ──────────────────────────────────────────────────────────────────────────

	_request( msg, onProgress, onPreview ) {

		if ( ! this._graph ) return Promise.resolve( { cancelled: true, reason: 'no-graph' } );
		// newest wins: the older request of this slot is settled here, the worker stops its job
		// when the new message arrives
		const old = this._slots.get( msg.slot );
		if ( old !== undefined ) this._settle( old, { cancelled: true, reason: 'superseded' } );
		const id = this._nextId ++;
		return new Promise( ( resolve, reject ) => {

			this._pending.set( id, { id, slot: msg.slot, resolve, reject, onProgress, onPreview } );
			this._slots.set( msg.slot, id );
			try {

				this._ensureWorker();
				if ( ! this._graphSent ) {

					const g = this._graph;
					// structured clone: the main thread keeps its arrays (GPU upload, picking, tools)
					this._post( { type: 'graph', graph: this._graphId, N: g.N, E: g.E, edges: g.edges, adjStart: g.adjStart, degree: g.degree, adjList: g.adjList } );
					this._graphSent = true;

				}

				this._post( { ...msg, id, graph: this._graphId } );

			} catch ( e ) {

				this._pending.delete( id );
				if ( this._slots.get( msg.slot ) === id ) this._slots.delete( msg.slot );
				reject( e );

			}

		} );

	}

	_post( msg ) {

		this._worker.postMessage( msg );

	}

	_settle( id, value, error ) {

		const r = this._pending.get( id );
		if ( ! r ) return;
		this._pending.delete( id );
		if ( this._slots.get( r.slot ) === id ) this._slots.delete( r.slot );
		if ( error ) r.reject( error ); else r.resolve( value );

	}

	_settleAll( value, error ) {

		for ( const id of [ ...this._pending.keys() ] ) this._settle( id, value, error );

	}

	_ensureWorker() {

		if ( this._worker ) return this._worker;
		const inline = this.options.inline || ( ! this.options.createWorker && typeof Worker === 'undefined' );
		const w = inline ? new InlineWorker() : ( this.options.createWorker || defaultWorker )();
		w.onmessage = ( e ) => this._onMessage( e.data );
		w.onerror = ( e ) => {

			if ( e && e.preventDefault ) e.preventDefault();
			this._fail( new Error( ( e && e.message ) || 'Analytics worker failed' ) );

		};

		w.onmessageerror = () => this._fail( new Error( 'Analytics worker: message could not be deserialized' ) );
		this._worker = w;
		this._graphSent = false;
		return w;

	}

	_terminate() {

		if ( ! this._worker ) return;
		this._worker.onmessage = null;
		this._worker.onerror = null;
		this._worker.terminate();
		this._worker = null;
		this._graphSent = false;

	}

	/** Worker crashed: reject what is in flight; the next request starts a fresh worker. */
	_fail( error ) {

		this._terminate();
		this._settleAll( null, error );

	}

	_labels( nodes ) {

		const labels = this._graph && this._graph.labels;
		return Array.from( nodes, ( i ) => ( labels && labels[ i ] !== undefined && labels[ i ] !== null ? String( labels[ i ] ) : String( i ) ) );

	}

	_onMessage( msg ) {

		const r = msg && this._pending.get( msg.id );
		if ( ! r ) return; // reply of a request that was superseded, cancelled, or meant for an older graph
		switch ( msg.type ) {

			case 'progress':
				safely( r.onProgress, { done: msg.done, total: msg.total, fraction: msg.total > 0 ? msg.done / msg.total : 1 } );
				break;
			case 'metric': {

				const result = {
					cancelled: false, metric: msg.metric, target: msg.target, values: msg.values, normalized: msg.normalized || null,
					scale: msg.scale, final: msg.final, exact: msg.exact, pivots: msg.pivots, total: msg.total, ms: msg.ms,
				};
				if ( msg.final ) this._settle( msg.id, result ); else safely( r.onPreview, result );
				break;

			}

			case 'summary': {

				const summary = { cancelled: false, ...msg.summary, final: msg.final };
				summary.top = {};
				for ( const [ key, list ] of Object.entries( msg.summary.top || {} ) ) summary.top[ key ] = { ...list, labels: this._labels( list.nodes ) };
				if ( summary.diameter ) summary.diameter = { ...summary.diameter, endpointLabels: this._labels( summary.diameter.endpoints.filter( ( i ) => i >= 0 ) ) };
				if ( msg.final ) this._settle( msg.id, summary ); else safely( r.onPreview, summary );
				break;

			}

			case 'cancelled':
				this._settle( msg.id, { cancelled: true, reason: msg.reason } );
				break;
			case 'error':
				this._settle( msg.id, null, new Error( msg.message ) );
				break;

		}

	}

}
