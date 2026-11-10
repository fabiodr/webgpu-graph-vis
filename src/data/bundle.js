// Cooperative driver for edge-path bundling (src/graph/bundling.js) in the data worker: runs
// the resumable task in short slices with a turn of the event loop between them, so the worker
// keeps reading its messages (a newer request, a cancel) while a large graph is being bundled.
// Pure JS; the same code runs inline on the main thread and in Node (tests).
//
// runBundle( graph, positions, settings ) -> Promise of
//     { cancelled: true }
//   | { cancelled: false,
//       paths: { pathStart: Uint32Array( E + 1 ), pathRefs: Uint32Array, maxInterior },
//       load: Float32Array( E ),   number of bundled edges routed over each edge divided by the
//                                  largest such number (0..1; 0 for edges that carry nothing)
//       stats, timings }
//   graph      { N, E, edges } and, for mode 'mixed', { D, edgeLca, anc } of a PreparedGraph
//   positions  Float32Array( 4N ) (xyz of node i at 4i; options.stride for another layout)
//   settings   mode: 'paths'  edge-path bundles only (the other edges are straight)
//                    'mixed'  edge-path bundles, and the hierarchical polygon for the edges
//                             that neither got a path nor carry one (mixedPaths, only: 'free')
//              options: BUNDLING_DEFAULTS overrides (method, exponent, maxDistortion,
//                       maxInterior, effort, greed) and
//                       timeBudget  ms; default 'auto' = BUNDLE_BUDGET_MS (3 s) at every
//                                   size; 0 = no limit. The budget is about the wait, not
//                                   about the edge count: a run that is done sooner is not
//                                   touched by it (most graphs below 100,000 edges on a
//                                   desktop), a slow machine gets a result in the same time.
//                                   A result cut by the clock is valid but weaker (the task
//                                   switches to quick tests when its first phase would not
//                                   finish in time at the pace measured after the first
//                                   BUNDLE_HURRY_AFTER of the budget; the second phase is a
//                                   refinement and is simply cut), and depends on the machine.
//                       only        'free' | 'straight', for mode 'mixed'
//              isCancelled()  polled between slices
//              onProgress( { phase, fraction, completed, total } )   about 10 per second
//              slice          ms of work between two turns of the event loop (default 24)
//              yieldTask      the turn itself (default: setImmediate in Node, a MessageChannel
//                             ping in browsers, which unlike setTimeout is not clamped to 4 ms)
//   stats      the task's stats (method, bundled, carrying, free, bundledRatio, inkRatio, complete,
//              hurried, phase, …) plus mode, maxLoad, budget (ms or null), outOfTime, and for
//              'mixed' the edge counts { fromPaths, fromHierarchy, straight }
//   timings    { setup, compute (ms inside the task), result, total (wall), slices }
//
// BUNDLE_BUDGET_MS, BUNDLE_HURRY_AFTER, bundleBudget( edgeCount, timeBudget ) -> ms | Infinity,
// defaultYield

import { edgePathBundlingTask, mixedPaths, hierarchicalPaths } from '../graph/bundling.js';

export const BUNDLE_BUDGET_MS = 3000;
/** Fraction of the budget that passes before the pace is trusted enough to hurry. */
export const BUNDLE_HURRY_AFTER = 0.05;

let channel = null;
const waiting = [];

/** One turn of the event loop (a macrotask), as cheaply as the platform allows. */
export function defaultYield() {

	if ( typeof setImmediate === 'function' ) return new Promise( ( resolve ) => setImmediate( resolve ) );
	if ( typeof MessageChannel !== 'undefined' ) {

		if ( ! channel ) {

			channel = new MessageChannel();
			channel.port1.onmessage = () => {

				const resolve = waiting.shift();
				if ( resolve ) resolve();

			};

		}

		return new Promise( ( resolve ) => {

			waiting.push( resolve );
			channel.port2.postMessage( 0 );

		} );

	}

	return new Promise( ( resolve ) => setTimeout( resolve, 0 ) );

}

/**
 * The time budget of a run in ms (Infinity = none): 'auto' gives BUNDLE_BUDGET_MS at every
 * size (edgeCount is not used any more: a person waits as long for 80,000 edges on a phone as
 * for 400,000 on a desktop); a positive number is taken as it is; 0, false or a negative number
 * switch the budget off.
 */
export function bundleBudget( edgeCount, timeBudget = 'auto' ) {

	if ( timeBudget === 'auto' || timeBudget === undefined || timeBudget === null ) return BUNDLE_BUDGET_MS;
	return + timeBudget > 0 ? + timeBudget : Infinity;

}

export async function runBundle( graph, positions, settings = {} ) {

	const { mode = 'paths', options = {}, isCancelled = null, onProgress = null, slice = 24, yieldTask = defaultYield } = settings;
	if ( mode !== 'paths' && mode !== 'mixed' ) throw new Error( `bundle: unknown mode "${mode}" (paths | mixed)` );
	if ( ! graph || ! Number.isInteger( graph.N ) || ! Number.isInteger( graph.E ) || ! graph.edges ) throw new Error( 'bundle: graph needs N, E and edges' );
	const stride = options.stride || 4;
	if ( ! positions || positions.length < graph.N * stride ) throw new Error( `bundle: positions needs ${stride} floats per node (${graph.N * stride}), got ${positions ? positions.length : 0}` );

	const t0 = now();
	const { timeBudget = 'auto', only = 'free', ...bundling } = options;
	const budget = bundleBudget( graph.E, timeBudget );

	const task = edgePathBundlingTask( graph, positions, bundling );
	const setup = now() - t0;
	// pace = ms per unit of work, smoothed (the rule of edgePathBundling's own budget loop)
	let pace = 0, last = now(), units = 0, compute = 0, slices = 0, lastReport = 0, outOfTime = false;
	const report = ( force ) => {

		if ( ! onProgress ) return;
		const t = now();
		if ( ! force && t - lastReport < 100 ) return;
		lastReport = t;
		onProgress( { phase: task.phase, fraction: task.total > 0 ? Math.min( 1, task.completed / task.total ) : 1, completed: task.completed, total: task.total } );

	};

	report( true );
	while ( ! task.done ) {

		const sliceStart = now();
		last = sliceStart;
		while ( ! task.done ) {

			const used = last - t0;
			if ( used >= budget ) {

				outOfTime = true;
				break;

			}

			// the first phase will not finish at this pace: quick tests from here on (not before
			// the pace has settled: the first, shortest edges are not typical)
			if ( budget < Infinity && task.phase === 'spanner' && used >= BUNDLE_HURRY_AFTER * budget && ( task.total / 2 - task.completed ) * pace > budget - used ) task.hurry();
			task.step( 32 );
			const t = now();
			const p = ( t - last ) / Math.max( 1, task.completed - units );
			pace += ( p - pace ) * ( pace > 0 ? 0.05 : 1 );
			last = t;
			units = task.completed;
			if ( t - sliceStart >= slice ) break;

		}

		compute += now() - sliceStart;
		slices ++;
		if ( outOfTime ) break;
		report( false );
		if ( task.done ) break;
		await yieldTask();
		if ( isCancelled && isCancelled() ) return { cancelled: true };

	}

	const tResult = now();
	const r = task.result();
	let paths = { pathStart: r.pathStart, pathRefs: r.pathRefs, maxInterior: r.maxInterior };
	const stats = { ...r.stats, mode, budget: budget < Infinity ? budget : null, outOfTime };
	if ( mode === 'mixed' && graph.anc && graph.edgeLca && graph.D > 0 ) {

		const mixed = mixedPaths( graph, r, hierarchicalPaths( graph ), { only } );
		paths = { pathStart: mixed.pathStart, pathRefs: mixed.pathRefs, maxInterior: mixed.maxInterior };
		let fromPaths = 0, fromHierarchy = 0;
		for ( let e = 0; e < graph.E; e ++ ) {

			if ( mixed.source[ e ] === 1 ) fromPaths ++;
			else if ( mixed.source[ e ] === 2 ) fromHierarchy ++;

		}

		stats.mixed = { fromPaths, fromHierarchy, straight: graph.E - fromPaths - fromHierarchy };

	}

	let maxLoad = 0;
	for ( let e = 0; e < graph.E; e ++ ) if ( r.load[ e ] > maxLoad ) maxLoad = r.load[ e ];
	const load = new Float32Array( graph.E );
	if ( maxLoad > 0 ) for ( let e = 0; e < graph.E; e ++ ) load[ e ] = r.load[ e ] / maxLoad;
	stats.maxLoad = maxLoad;
	stats.timeMs = now() - t0;
	report( true );
	return { cancelled: false, paths, load, stats, timings: { setup, compute, result: now() - tResult, total: now() - t0, slices } };

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}
