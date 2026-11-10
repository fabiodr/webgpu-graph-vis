// Analytics worker (module Web Worker, separate from the data worker so that a long centrality
// run never delays a graph load): node and edge metrics of src/graph/analytics.js and the graph
// summary of the insights panel, computed off the main thread. Created by src/analysis/client.js
// (AnalysisClient); the same logic runs without a Worker through createHandler() (Node tests,
// inline fallback).
//
// ─── Protocol ───────────────────────────────────────────────────────────────────────────────
//  main → worker
//    { type: 'graph', graph, N, E, edges, adjStart, degree, adjList }
//        graph    any id; later requests must carry the same id. Sent once per graph (the four
//                 CSR arrays of the PreparedGraph, structured-cloned). Cancels every running job
//                 and drops all caches.
//    { type: 'metric', id, graph, slot, target, metric, options, preview, maxMs, normalize }
//        id       request id, echoed in every reply
//        slot     string, default 'default'. The NEWEST request of a slot wins: an older one that
//                 is still running is cancelled ('cancelled', reason 'superseded'). Different
//                 slots run side by side.
//        target   'node' (default) | 'edge'; metric: an id of metrics.js
//        options  { samples, seed, sources, damping, … }
//                 samples (slow metrics only): 'auto' (default) = the library's pivot count
//                 (exact when N ( N + 2E ) ≤ 1e8), a number, or 'exact' / Infinity = every node,
//                 in which case the estimate refines until it is exact
//                 sources ('relevance'): node indices of the restart set
//        preview  ms between progressive results of a slow metric (0 = none). The interval
//                 grows by 1.6× per preview up to 8× (at least 1.5 s): coarse first, then calmer.
//        maxMs    stop refining after this time and post what there is as final (0 = no limit)
//        normalize  a normalize() mode, or true for the catalogue's: also return values in [0,1]
//    { type: 'summary', id, graph, slot, options: { top = 10, maxBfs = 32 } }
//    { type: 'cancel', id } | { type: 'cancel', slot } | { type: 'cancel' }  (everything)
//    { type: 'reset' }       forget the graph (cancels everything, frees memory)
//  worker → main
//    { type: 'progress', id, done, total }        pivots (metrics) or stages (summary); ≥ 80 ms apart
//    { type: 'metric', id, metric, target, values, normalized?, scale, final, pivots, total, exact, ms }
//        values   Float32Array( N | E ), transferred. final: false = a progressive estimate from
//                 `pivots` of `total` pivots (always finite and correctly scaled), more follow.
//                 exact: true when the values are not sampled.
//    { type: 'summary', id, summary, final }      see "Summary" below; grows over four stages
//    { type: 'cancelled', id, reason }            'superseded' | 'cancel' | 'graph'
//    { type: 'error', id, message }
//
// ─── Scheduling ─────────────────────────────────────────────────────────────────────────────
//  Instant and fast metrics are computed inside the message handler and answered at once (raw
//  values are cached per graph, so asking again costs a copy). The slow ones (betweenness,
//  closeness, harmonic, edge betweenness) are resumable tasks of the library. A task is shared by
//  every job that needs it (closeness and harmonic are one task; node betweenness rides on an
//  edge-betweenness task that is further along; every job gets its own copy of the values,
//  because a posted array is transferred) and is stepped in slices of `sliceMs` (24 ms),
//  round-robin over the tasks that have a waiting job, with a yield to the event loop after each
//  slice: that is what lets a newer request, a cancel or a new graph get through. A task whose
//  jobs are all gone is parked, not dropped (the four most recent are kept), so returning to a
//  metric resumes the refinement where it stopped.
//
// ─── Summary ────────────────────────────────────────────────────────────────────────────────
//  { stage: 'basic' | 'centrality' | 'distance' | 'done',
//    nodes, edges, density,
//    degree: { min, max, mean, median, isolated, histogram: Uint32Array, bins: [ { from, to, count } ] },
//    assortativity,                     NaN for regular graphs / no edges
//    components: { count, largest, largestFraction, singletons, sizes: number[] (first 12) },
//    clustering: { average, transitivity, triangles },
//    core: { degeneracy, sizes: Uint32Array }           nodes per core number
//    distance: { averagePathLength, effectiveDiameter, distribution: Float64Array, maxDistance,
//                pivots, exact } | null                 sampled unless exact
//    diameter: { lower, upper, exact, endpoints: [ a, b ], bfsCount } | null
//    top: { degree, pagerank, eigenvector, closeness, betweenness }  each
//         { nodes: number[], values: number[], exact: boolean } (largest first), filled in as
//         the stages complete
//    timings: { basic, centrality, distance, betweenness, total }  ms }
//
// ─── Exports (inline use / tests) ───────────────────────────────────────────────────────────
//  createHandler( { post, now?, wait?, sliceMs = 24, progressMs = 80 } ) -> handle( msg )
//      post( data, transfer? ) receives the replies. State lives in the handler: one per worker.
//  topK( values, k ) -> { nodes, values }

import * as GA from '../graph/analytics.js';
import { metricInfo } from './metrics.js';

const PARKED_TASKS = 4;

const defaultNow = () => ( typeof performance !== 'undefined' ? performance.now() : Date.now() );
const defaultWait = () => new Promise( ( resolve ) => setTimeout( resolve, 0 ) );

/** The k largest values with their indices, largest first (ties: lower index first). O( n k ). */
export function topK( values, k ) {

	const n = values.length;
	k = Math.max( 0, Math.min( k | 0, n ) );
	const nodes = [], best = [];
	for ( let i = 0; i < n; i ++ ) {

		const v = values[ i ];
		if ( ! ( v === v ) ) continue; // NaN
		if ( best.length === k && ! ( v > best[ k - 1 ] ) ) continue;
		let p = best.length;
		while ( p > 0 && v > best[ p - 1 ] ) p --;
		best.splice( p, 0, v );
		nodes.splice( p, 0, i );
		if ( best.length > k ) {

			best.pop();
			nodes.pop();

		}

	}

	return { nodes, values: best };

}

/** Logarithmic bins of a degree histogram, for a compact chart: 0, 1, 2, 3–4, 5–8, 9–16, … */
function degreeBins( histogram ) {

	const bins = [];
	const push = ( from, to ) => {

		let count = 0;
		for ( let d = from; d <= to && d < histogram.length; d ++ ) count += histogram[ d ];
		bins.push( { from, to: Math.min( to, histogram.length - 1 ), count } );

	};

	const max = histogram.length - 1;
	for ( let d = 0; d <= Math.min( 2, max ); d ++ ) push( d, d );
	for ( let from = 3, width = 2; from <= max; from += width, width *= 2 ) push( from, from + width - 1 );
	return bins;

}

function sampleKey( samples ) {

	if ( samples === 'exact' || samples === Infinity ) return 'exact';
	if ( samples === undefined || samples === null || samples === 'auto' || Number.isNaN( + samples ) ) return 'auto';
	return String( Math.max( 1, Math.floor( + samples ) ) );

}

/**
 * Creates the message handler of one analytics worker.
 *
 * @param {object} config
 * @param {function(object, ArrayBuffer[]=): void} config.post receives the replies
 * @param {function(): number} [config.now] clock in ms
 * @param {function(): Promise} [config.wait] how to yield between slices (default setTimeout 0)
 * @param {number} [config.sliceMs=24] length of one slice of a slow task
 * @param {number} [config.progressMs=80] least time between two progress messages of a job
 * @returns {function(object): void}
 */
export function createHandler( config = {} ) {

	const post = config.post || ( () => {} );
	const now = config.now || defaultNow;
	const wait = config.wait || defaultWait;
	const sliceMs = config.sliceMs === undefined ? 24 : config.sliceMs;
	const progressMs = config.progressMs === undefined ? 80 : config.progressMs;

	let graph = null; // { id, N, E, edges, adjStart, degree, adjList, adjEdge? }
	let memo = null; // per graph: shared intermediates and raw results
	let entries = new Map(); // key -> { key, kind, edges, task, consumers: Set, used }
	const jobs = new Map(); // id -> job
	const slots = new Map(); // slot -> job
	let pumping = false, cursor = 0, useCounter = 0;

	// ── shared intermediates ─────────────────────────────────────────────────────────────────

	const components = () => memo.components || ( memo.components = GA.connectedComponents( graph ) );
	const core = () => memo.core || ( memo.core = GA.kCore( graph ) );
	const adjEdge = () => graph.adjEdge || ( graph.adjEdge = GA.buildEdgeIndex( graph ) );

	function clustering( withEdges ) {

		if ( ! memo.clustering || ( withEdges && ! memo.clustering.edgeTriangles ) ) {

			memo.clustering = GA.clusteringCoefficient( graph, { edges: withEdges } );

		}

		return memo.clustering;

	}

	/** Raw values of an instant / fast metric (a fresh array the caller may transfer). */
	function cheapValues( target, metric, options ) {

		const { N, E } = graph;
		if ( target === 'node' && metric === 'relevance' ) {

			const weights = new Float32Array( N );
			let any = false;
			for ( const s of options.sources || [] ) {

				if ( Number.isInteger( s ) && s >= 0 && s < N ) {

					weights[ s ] = 1;
					any = true;

				}

			}

			if ( ! any ) return weights; // no restart set: nothing is relevant
			return GA.pageRank( graph, { damping: options.damping, tolerance: options.tolerance, personalization: weights, components: components() } );

		}

		const { damping, tolerance, maxIterations } = options;
		const key = `${target}:${metric}:${damping ?? ''}:${tolerance ?? ''}:${maxIterations ?? ''}`;
		let values = memo.values.get( key );
		if ( ! values ) {

			if ( target === 'edge' ) {

				const tri = clustering( true ).edgeTriangles;
				values = new Float32Array( E );
				if ( metric === 'triangles' ) values.set( tri.subarray( 0, E ) );
				else if ( metric === 'embeddedness' ) {

					const { edges, degree } = graph;
					for ( let e = 0; e < E; e ++ ) {

						// |N(a) ∩ N(b)| / |N(a) ∪ N(b) − { a, b }|
						const union = degree[ edges[ 2 * e ] ] + degree[ edges[ 2 * e + 1 ] ] - 2 - tri[ e ];
						values[ e ] = union > 0 ? tri[ e ] / union : 0;

					}

				} else throw new Error( `Edge metric "${metric}" is not a cheap metric` );

			} else {

				switch ( metric ) {

					case 'degree': values = Float32Array.from( graph.degree ); break;
					case 'core': values = Float32Array.from( core().core ); break;
					case 'pagerank': values = GA.pageRank( graph, { damping, tolerance, maxIterations, components: components() } ); break;
					case 'eigenvector': values = GA.eigenvectorCentrality( graph, { tolerance, maxIterations, components: components() } ); break;
					case 'clustering': values = Float32Array.from( clustering( false ).local ); break;
					case 'triangles': values = Float32Array.from( clustering( false ).triangles ); break;
					case 'component': values = Float32Array.from( components().component ); break;
					default: throw new Error( `Node metric "${metric}" is not a cheap metric` );

				}

			}

			memo.values.set( key, values );

		}

		return values.slice();

	}

	// ── slow tasks ───────────────────────────────────────────────────────────────────────────

	function entryFor( kind, samples, seed, edges ) {

		const key = `${kind}|${sampleKey( samples )}|${seed}|${edges ? 1 : 0}`;
		let entry = entries.get( key );
		if ( ! entry ) {

			const s = sampleKey( samples );
			const options = { samples: s === 'exact' ? Infinity : s === 'auto' ? undefined : + s, seed, components: components() };
			const task = kind === 'betweenness'
				? GA.betweennessTask( graph, { ...options, edges, adjEdge: edges ? adjEdge() : undefined } )
				: GA.closenessTask( graph, options );
			entry = { key, kind, edges, task, consumers: new Set(), used: 0 };
			entries.set( key, entry );

		}

		entry.used = ++ useCounter;
		// keep only the most recent parked tasks (each holds O( N + E ) of accumulators)
		const parked = [ ...entries.values() ].filter( ( e ) => e.consumers.size === 0 && e !== entry ).sort( ( a, b ) => b.used - a.used );
		for ( const e of parked.slice( PARKED_TASKS ) ) entries.delete( e.key );
		return entry;

	}

	/** Task that serves a slow metric; node betweenness reuses an edge task that is further along. */
	function entryForMetric( target, metric, options ) {

		const seed = options.seed === undefined ? 1 : options.seed;
		if ( metric !== 'betweenness' ) return entryFor( 'closeness', options.samples, seed, false );
		if ( target === 'edge' ) return entryFor( 'betweenness', options.samples, seed, true );
		const withEdges = entries.get( `betweenness|${sampleKey( options.samples )}|${seed}|1` );
		const plain = entries.get( `betweenness|${sampleKey( options.samples )}|${seed}|0` );
		if ( withEdges && ( ! plain || withEdges.task.completed >= plain.task.completed ) ) {

			withEdges.used = ++ useCounter;
			return withEdges;

		}

		return entryFor( 'betweenness', options.samples, seed, false );

	}

	/** Lets every consumer of a task look at its state: progress, previews, completion. */
	function service( entry ) {

		const t = now();
		let snapshot = null;
		// one result() per service, shared by its consumers. Every taker gets a copy of its own:
		// a posted array is transferred, i.e. detached at once, so the array one consumer was
		// given can no longer be read (or copied) for the next consumer of the same pass.
		const pick = ( name ) => {

			if ( ! snapshot ) snapshot = entry.task.result();
			return snapshot[ name ].slice();

		};

		for ( const consumer of [ ...entry.consumers ] ) {

			let finished = true;
			try {

				finished = consumer.job.cancelled || consumer.update( entry, t, pick );

			} catch ( e ) {

				fail( consumer.job, e ); // its own failure (e.g. a bad normalize mode): the task and its other jobs go on

			}

			if ( finished ) entry.consumers.delete( consumer );

		}

	}

	function nextEntry() {

		const live = [];
		for ( const entry of entries.values() ) {

			for ( const consumer of entry.consumers ) if ( consumer.job.cancelled ) entry.consumers.delete( consumer );
			if ( entry.consumers.size > 0 ) live.push( entry );

		}

		if ( live.length === 0 ) return null;
		cursor = ( cursor + 1 ) % live.length;
		return live[ cursor ];

	}

	async function pump() {

		if ( pumping ) return;
		pumping = true;
		try {

			for ( ;; ) {

				const entry = nextEntry();
				if ( ! entry ) break;
				try {

					let goal = 0;
					for ( const consumer of entry.consumers ) if ( consumer.goal > goal ) goal = consumer.goal;
					const t0 = now();
					while ( entry.task.completed < goal && ! entry.task.done ) {

						entry.task.step( 1 );
						if ( now() - t0 >= sliceMs ) break;

					}

					service( entry );
					if ( entry.task.done ) entry.consumers.clear(); // nothing is left to wait for

				} catch ( e ) {

					for ( const consumer of [ ...entry.consumers ] ) fail( consumer.job, e );
					entry.consumers.clear();
					entries.delete( entry.key );

				}

				await wait();

			}

		} finally {

			pumping = false;

		}

	}

	/** Resolves true once `entry` has done `goal` pivots, false when the job is cancelled first. */
	function until( job, entry, goal ) {

		return new Promise( ( resolve ) => {

			const consumer = {
				job, goal: Math.min( goal, entry.task.total ),
				update( e ) {

					if ( e.task.completed < consumer.goal && ! e.task.done ) return false;
					job.waiting = null;
					resolve( true );
					return true;

				},
			};
			if ( consumer.update( entry ) ) return;
			job.waiting = () => resolve( false );
			entry.consumers.add( consumer );
			pump();

		} );

	}

	// ── jobs ─────────────────────────────────────────────────────────────────────────────────

	function release( job ) {

		jobs.delete( job.id );
		if ( slots.get( job.slot ) === job ) slots.delete( job.slot );

	}

	function cancelJob( job, reason ) {

		if ( job.cancelled || job.finished ) return;
		job.cancelled = true;
		release( job );
		if ( job.waiting ) job.waiting();
		post( { type: 'cancelled', id: job.id, reason } );

	}

	function fail( job, error ) {

		if ( job.cancelled || job.finished ) return;
		job.cancelled = true;
		release( job );
		if ( job.waiting ) job.waiting();
		post( { type: 'error', id: job.id, message: ( error && error.message ) || String( error ) } );

	}

	function postValues( job, values, extra ) {

		const msg = {
			type: 'metric', id: job.id, metric: job.metric, target: job.target, values, scale: job.info.scale,
			final: true, pivots: 0, total: 0, exact: true, ms: now() - job.t0, ...extra,
		};
		const transfer = [ values.buffer ];
		if ( job.normalize ) {

			msg.normalized = GA.normalize( values, { mode: job.normalize } );
			msg.normalizedScale = job.normalize;
			transfer.push( msg.normalized.buffer );

		}

		if ( msg.final ) {

			job.finished = true;
			release( job );

		}

		post( msg, transfer );

	}

	function addJob( msg, type ) {

		const slot = msg.slot === undefined ? 'default' : String( msg.slot );
		const old = slots.get( slot );
		if ( old ) cancelJob( old, 'superseded' );
		const job = { id: msg.id, slot, type, cancelled: false, finished: false, waiting: null, t0: now(), options: msg.options || {} };
		if ( ! graph || msg.graph !== graph.id ) {

			// a request that was meant for another graph (it raced with a graph change)
			post( { type: 'cancelled', id: job.id, reason: 'graph' } );
			return null;

		}

		jobs.set( job.id, job );
		slots.set( slot, job );
		return job;

	}

	function startMetric( msg ) {

		const job = addJob( msg, 'metric' );
		if ( ! job ) return;
		try {

			job.target = msg.target === 'edge' ? 'edge' : 'node';
			job.metric = msg.metric;
			job.info = metricInfo( msg.metric, job.target );
			if ( ! job.info ) throw new Error( `Unknown ${job.target} metric "${msg.metric}"` );
			job.normalize = msg.normalize === true ? job.info.scale : msg.normalize || null;
			if ( job.info.cost !== 'slow' ) {

				postValues( job, cheapValues( job.target, job.metric, job.options ), {} );
				return;

			}

			const entry = entryForMetric( job.target, job.metric, job.options );
			const name = job.target === 'edge' ? 'edge' : entry.kind === 'betweenness' ? 'node' : job.metric;
			const base = msg.preview > 0 ? + msg.preview : 0;
			const deadline = msg.maxMs > 0 ? job.t0 + msg.maxMs : Infinity;
			let interval = base, nextPreview = job.t0 + base, nextProgress = 0, previewed = - 1;
			const state = ( task, final ) => ( { final, pivots: task.completed, total: task.total, exact: task.exact && task.done } );
			const consumer = {
				job, goal: entry.task.total,
				update( e, t, pick ) {

					const task = e.task;
					if ( task.done || ( t >= deadline && task.completed > 0 ) ) {

						post( { type: 'progress', id: job.id, done: task.completed, total: task.total } );
						postValues( job, pick( name ), state( task, true ) );
						return true;

					}

					if ( t >= nextProgress ) {

						post( { type: 'progress', id: job.id, done: task.completed, total: task.total } );
						nextProgress = t + progressMs;

					}

					if ( base > 0 && t >= nextPreview && task.completed > previewed && task.completed > 0 ) {

						postValues( job, pick( name ), state( task, false ) );
						previewed = task.completed;
						interval = Math.min( interval * 1.6, Math.max( 8 * base, 1500 ) );
						nextPreview = t + interval;

					}

					return false;

				},
			};
			// a task that already has pivots (parked, or shared with another job): show them now
			if ( entry.task.completed > 0 ) nextPreview = job.t0;
			if ( consumer.update( entry, now(), ( n ) => entry.task.result()[ n ] ) ) return;
			entry.consumers.add( consumer );
			pump();

		} catch ( e ) {

			fail( job, e );

		}

	}

	async function runSummary( job ) {

		const g = graph;
		const alive = () => ! job.cancelled && graph === g;
		const k = Math.max( 0, Math.min( 100, job.options.top === undefined ? 10 : job.options.top | 0 ) );
		const timings = {};
		const out = { stage: 'basic', nodes: g.N, edges: g.E, distance: null, diameter: null, top: {}, timings };
		const emit = ( stage, done, final ) => {

			out.stage = stage;
			post( { type: 'progress', id: job.id, done, total: 4 } );
			post( { type: 'summary', id: job.id, summary: { ...out, top: { ...out.top }, timings: { ...timings } }, final } );

		};

		const top = ( values, exact ) => ( { ...topK( values, k ), exact } );

		// 1. one-pass statistics
		let t = now();
		const s = GA.graphSummary( g ), comp = components(), kc = core(), cc = clustering( false );
		out.density = s.density;
		out.degree = { min: s.minDegree, max: s.maxDegree, mean: s.meanDegree, median: s.medianDegree, isolated: s.isolated, histogram: s.degreeHistogram, bins: degreeBins( s.degreeHistogram ) };
		out.assortativity = s.assortativity;
		let singletons = 0;
		for ( let c = comp.count - 1; c >= 0 && comp.sizes[ c ] === 1; c -- ) singletons ++;
		out.components = {
			count: comp.count, largest: comp.count ? comp.sizes[ 0 ] : 0, largestFraction: g.N ? comp.sizes[ 0 ] / g.N : 0,
			singletons, sizes: Array.from( comp.sizes.subarray( 0, 12 ) ),
		};
		out.clustering = { average: cc.average, transitivity: cc.global, triangles: cc.triangleCount };
		const coreSizes = new Uint32Array( kc.degeneracy + 1 );
		for ( let v = 0; v < g.N; v ++ ) coreSizes[ kc.core[ v ] ] ++;
		out.core = { degeneracy: kc.degeneracy, sizes: coreSizes };
		out.top.degree = top( g.degree, true );
		timings.basic = now() - t;
		emit( 'basic', 1, false );
		await wait();
		if ( ! alive() ) return;

		// 2. iterative centralities
		t = now();
		out.top.pagerank = top( cheapValues( 'node', 'pagerank', {} ), true );
		out.top.eigenvector = top( cheapValues( 'node', 'eigenvector', {} ), true );
		timings.centrality = now() - t;
		emit( 'centrality', 2, false );
		await wait();
		if ( ! alive() ) return;

		// 3. distances: a sampled closeness task (shared with the closeness / harmonic metrics;
		// an exact one that is already further along is used instead) and the diameter bounds
		t = now();
		const want = GA.defaultSamples( g );
		const pickEntry = ( kind ) => {

			const refined = entries.get( `${kind}|exact|1|0` ) || ( kind === 'betweenness' ? entries.get( 'betweenness|exact|1|1' ) : null );
			return refined && refined.task.completed >= Math.min( want, refined.task.total ) ? refined : entryFor( kind, 'auto', 1, false );

		};

		let entry = pickEntry( 'closeness' );
		if ( ! await until( job, entry, entry.key.includes( '|exact|' ) ? want : entry.task.total ) || ! alive() ) return;
		let r = entry.task.result();
		out.distance = {
			averagePathLength: r.averagePathLength, effectiveDiameter: r.effectiveDiameter, distribution: r.distanceDistribution,
			maxDistance: r.maxDistance, pivots: r.pivots, exact: r.exact,
		};
		out.top.closeness = top( r.closeness, r.exact );
		const d = GA.estimateDiameter( g, { maxBfs: job.options.maxBfs === undefined ? 32 : job.options.maxBfs, components: comp } );
		out.diameter = { lower: d.diameter, upper: d.diameterUpper, exact: d.exact, endpoints: Array.from( d.endpoints ), bfsCount: d.bfsCount };
		timings.distance = now() - t;
		emit( 'distance', 3, false );
		await wait();
		if ( ! alive() ) return;

		// 4. betweenness leaders
		t = now();
		entry = pickEntry( 'betweenness' );
		if ( ! await until( job, entry, entry.key.includes( '|exact|' ) ? want : entry.task.total ) || ! alive() ) return;
		r = entry.task.result();
		out.top.betweenness = top( r.node, r.exact );
		out.top.betweenness.pivots = r.pivots;
		timings.betweenness = now() - t;
		timings.total = now() - job.t0;
		job.finished = true;
		release( job );
		emit( 'done', 4, true );

	}

	function startSummary( msg ) {

		const job = addJob( msg, 'summary' );
		if ( job ) runSummary( job ).catch( ( e ) => fail( job, e ) );

	}

	function cancelAll( reason ) {

		for ( const job of [ ...jobs.values() ] ) cancelJob( job, reason );
		for ( const entry of entries.values() ) entry.consumers.clear();

	}

	return function handle( msg ) {

		if ( ! msg || typeof msg !== 'object' ) return;
		try {

			switch ( msg.type ) {

				case 'graph':
					cancelAll( 'graph' );
					graph = { id: msg.graph, N: msg.N >>> 0, E: msg.E >>> 0, edges: msg.edges, adjStart: msg.adjStart, degree: msg.degree, adjList: msg.adjList };
					memo = { values: new Map() };
					entries = new Map();
					break;
				case 'reset':
					cancelAll( 'graph' );
					graph = null;
					memo = null;
					entries = new Map();
					break;
				case 'metric': startMetric( msg ); break;
				case 'summary': startSummary( msg ); break;
				case 'cancel':
					if ( msg.id !== undefined ) {

						const job = jobs.get( msg.id );
						if ( job ) cancelJob( job, 'cancel' );

					} else if ( msg.slot !== undefined ) {

						const job = slots.get( String( msg.slot ) );
						if ( job ) cancelJob( job, 'cancel' );

					} else cancelAll( 'cancel' );

					break;

			}

		} catch ( e ) {

			post( { type: 'error', id: msg.id, message: ( e && e.message ) || String( e ) } );

		}

	};

}

// ── worker entry ─────────────────────────────────────────────────────────────────────────────

const isWorker = typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope; // eslint-disable-line no-undef
if ( isWorker ) {

	// Yield between slices through a message port instead of setTimeout( 0 ): nested timers are
	// clamped to 4 ms, which would leave the worker idle for a sixth of every 24 ms slice. Port
	// messages and the messages of the main thread are tasks of the same queue, in order of
	// arrival, so a request, a cancel or a new graph still gets its turn after the slice it
	// arrived in.
	const channel = new MessageChannel(), resume = [];
	channel.port1.onmessage = () => resume.shift()();
	const wait = () => new Promise( ( resolve ) => {

		resume.push( resolve );
		channel.port2.postMessage( 0 );

	} );
	const handle = createHandler( { post: ( data, transfer ) => self.postMessage( data, transfer || [] ), wait } );
	self.onmessage = ( e ) => handle( e.data );

}
