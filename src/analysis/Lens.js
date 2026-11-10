// Metric lens (App module, order 25): maps node metrics of the analytics worker to the three
// visual channels of a node (size, colour, glow) and keeps the metric values at hand for the UI.
//
// ─── What it writes ─────────────────────────────────────────────────────────────────────────
//  size     nodeColorTarget.w = lo + ( hi − lo ) · t ^ gamma   t = the metric normalised to
//           [0,1] with the scale the catalogue suggests (log for heavy tails, robust for
//           closeness, …). [ lo, hi ] = sizeRange, by default the whole range of radius factors
//           of the node size contract, 0.7 … 1.6 (SIZE_RANGE, what the degree radii of
//           prepare.js span); a custom range is clamped to it, because labels, picking, halos
//           and the node renderer are built for that range. Contrast therefore comes from the
//           mapping, not from a wider range: gamma (sizeGamma, default 1.5) keeps the bulk
//           small and sets the leaders apart.
//  colour   nodeColorTarget.rgb = rampColor( t, ramp ) · emissiveGain( …, intensity )   a
//           sequential ramp of util/color.js; categorical metrics (component) get distinct hues
//           for the first eight labels. A node without a value (a number the file does not
//           give for it) is drawn in the neutral grey of the unnamed categories instead of the
//           colour of the smallest value; on the size channel it gets the smallest radius, and
//           legend().missing says how many there are. The node and edge shaders are emissive and feed the
//           bloom, and the upper half of a ramp is far brighter than any cluster palette
//           (luminance up to 0.95 against 0.05 … 0.59), so as defined the high values burn out
//           into one glowing mass. `intensity` (default 0.4) holds the bright end back with a
//           soft shoulder, colour / ( 1 + ( 1 / intensity − 1 ) · luminance ): luminance
//           0.09 … 0.39 instead of 0.11 … 0.95, the dark end almost untouched, the same hues,
//           lightness still strictly rising. 1 = the ramp as defined, which is also what
//           legends and pickers show (legend().css).
//  glow     nodeFx.z = 1 − rank / K for the K = ⌈top · N⌉ nodes with the largest values, 0 for
//           the rest (rank = number of nodes with a larger value, so equal values glow alike).
//           A group of equal values that does not fit into the top K is not a top: it fades
//           and is dark once it reaches 3 K (the 9-core of a graph, a regular graph's degree).
//  value    nodeFx.w = t of the colour metric (else of the size metric, else of the glow
//           metric): the normalised value itself, without the gamma of the channel
//  Colours and radii go through buffers.setColors(), so the GPU eases to them (~0.25 s); glow
//  and value go through buffers.setFx() after rewriting only z / w of the CPU array (x / y
//  belong to the analysis tools). A channel without a metric shows the graph's own values: the
//  cluster palette ( colorize( leafRank, palette ) ) and the degree radii.
//  While a metric is still being computed the channel keeps what it shows. Slow metrics
//  (betweenness, closeness, harmonic) arrive as a coarse estimate first and are refined on
//  screen; a metric that a channel shows is refined until it is exact (`untilExact`), one that
//  only a request() / refine() waits for stops after `refineMs` with the estimate it has. A
//  metric nobody uses any more is stopped at once (the worker keeps its state for later).
//
//  With `mirror` (default) the current targets are also written into graph.radii and
//  graph.colors, the arrays that labels, picking and halos already read, so every consumer of
//  the node size contract follows the lens without knowing it. The graph's own values are in
//  lens.base, and are put back into the graph when the lens leaves it (new graph, dispose).
//  Code that caches per-node colours on the CPU (the pills of the DOM labels) re-reads
//  lens.colors on a 'lens' event.
//
// ─── API ────────────────────────────────────────────────────────────────────────────────────
//  const lens = app.use( new Lens( { ramp: 'viridis', intensity: 0.4, sizeRange: [ 0.7, 1.6 ],
//                                    sizeGamma: 1.5, top: 0.05, refineMs: 20000, untilExact: true } ) );
//  lens.setSize( id | null, { scale?, gamma = sizeGamma, range? } ) -> Promise<boolean>
//  lens.setColor( id | null, { ramp?, reverse?, intensity?, scale?, gamma = 1 } ) -> Promise<boolean>
//  lens.setGlow( id | null, { top = 0.05 } )                       -> Promise<boolean>
//      The promise resolves true once the FINAL values are on the buffers, false when the call
//      was superseded (another metric on that channel, a new graph) first. id: see lens.metrics.
//      gamma: t ^ gamma after the normalisation; above 1 it sets the leaders apart from the bulk.
//      intensity: 0.05 … 1. ramp and intensity stay as they are when a later setColor() does not
//      name them.
//  lens.clear()                                         all three channels off
//  lens.setPalette( name )                              cluster palette of the "no colour lens" state
//  lens.setSources( nodes )                             restart set of the 'relevance' metric
//                                                       (follows the 'select' event by itself)
//  lens.refine( id, ms = refineMs )                     continue a slow metric that stopped early
//                                                       (ms 0 = until exact)
//  lens.values( id )    -> { id, label, scale, raw, normalized, rank, order, domain, final, exact,
//                            pivots, total } | null     null until the metric has values.
//                            rank[ i ] = number of nodes with a larger value (0 = top; ties
//                            share a rank), order = node indices, largest value first
//  lens.request( id )   -> Promise<values>              compute a metric without showing it
//  lens.load( ids? )    -> Promise<void>                all instant and fast metrics (node card)
//  lens.describe( i )   -> [ { id, label, value, normalized, rank, of, exact, final } ]
//                                                       every metric that has values, rank 1-based
//  lens.legend( channel ) -> { metric, label, scale, min, max, ready, ticks, missing, ramp?, css?, intensity?,
//                              categorical?, categories?, gamma?, top?, count?, threshold?, range?,
//                              missingColor? } | null
//                                                       ticks: [ { t, value, bound } ] what the two
//                                                       ends and the middle of the channel show, read
//                                                       back from the data: value = a value some node
//                                                       has at that position, bound = '≤' / '≥' when
//                                                       the end also stands for everything beyond it
//                                                       (log and robust scales clamp), else ''.
//                                                       categories: [ { value, label, count, color } ]
//                                                       of a categorical metric (the first eight have
//                                                       a hue; color = linear RGB as on the nodes).
//                                                       count: the number of nodes that glow,
//                                                       threshold: the smallest value among them.
//                                                       missing: nodes without a (finite) value of a
//                                                       sequential metric, i.e. outside the ticks;
//                                                       missingColor: linear RGB they are drawn in
//                                                       on the colour channel
//  lens.attributes( i ) -> [ { id, name, type, value, text } ]   the node attributes of the file
//                                                       for node i (graph.attributes), text = the
//                                                       category name or the number
//  lens.summary( { onPreview, top, maxBfs } ) -> Promise<Summary | null>   see worker.js; computed
//                                                       once per graph, lazily; labels attached
//  lens.edgeChannel( id, { scale?, onUpdate?, maxMs? } ) -> Promise<{ metric, scale, values, raw,
//      final, exact, pivots, total } | null>            per-edge channel for the edge renderer:
//      values = Float32Array( E ) in [0,1] (normalised in the worker), id: see lens.edgeMetrics.
//      Every ( id, scale ) is a request of its own, so several channels can be asked for at
//      once; callers of the same channel share one request. Slow channels (edge betweenness)
//      call onUpdate with each estimate and resolve when exact (or after maxMs; asking again
//      continues). null: no graph, a new graph came first, or the channel was released.
//  lens.releaseEdgeChannel( id?, { scale? } )           stop refining a channel (all without id)
//                                                       that is no longer shown
//  lens.state           { size, color, glow, ramp, reverse, intensity, top, palette, sizeRange,
//                         sources, busy, metrics: { [id]: { ready, final, exact, pivots, total,
//                         progress, error } } }
//  lens.metrics, lens.edgeMetrics, lens.ramps           catalogues for a UI. lens.metrics is the
//                                                       catalogue of metrics.js plus one entry per
//                                                       node attribute of the file ( id 'attr:<name>',
//                                                       graph.attributes: numbers on the robust scale,
//                                                       categories as distinct hues, most frequent
//                                                       first ); they need no worker
//  lens.radii, lens.colors                              current targets (Float32Array N / 4N)
//  lens.base            { radii, colors }               the graph's own values
//  lens.client          the AnalysisClient
//  SIZE_RANGE           [ 0.7, 1.6 ]                    the radius factors a size lens may use
//
//  Events on app.events: 'lens', detail = lens.state plus `reason`:
//    'config' (a channel, the ramp or the palette changed), 'data' (values arrived and were
//    applied), 'progress' (at most every 200 ms), 'graph'.

import { colorize } from '../graph/prepare.js';
import { normalize } from '../graph/analytics.js';
import { rampColors, rampCss, rampList, categoryColor, emissiveGain, RAMPS } from '../util/color.js';
import { AnalysisClient } from './client.js';
import { NODE_METRICS, EDGE_METRICS, metricInfo } from './metrics.js';

const now = () => ( typeof performance !== 'undefined' ? performance.now() : Date.now() );

/**
 * Radius factors of the node size contract ( nodeColor.w, see GraphBuffers ): what the degree
 * radii of prepare.js span, and what a size lens may use.
 */
export const SIZE_RANGE = Object.freeze( [ 0.7, 1.6 ] );

/** A [ lo, hi ] pair of radius factors, each end clamped to SIZE_RANGE (lo > hi inverts the lens). */
function sizeRange( range ) {

	const [ min, max ] = SIZE_RANGE;
	const clamp = ( v, fallback ) => ( Number.isFinite( + v ) ? Math.min( max, Math.max( min, + v ) ) : fallback );
	return [ clamp( range ? range[ 0 ] : NaN, min ), clamp( range ? range[ 1 ] : NaN, max ) ];

}

/** An intensity: 0.05 … 1 (below that the colours are too dark to tell apart). */
const unit = ( v, fallback ) => ( Number.isFinite( + v ) ? Math.min( 1, Math.max( 0.05, + v ) ) : fallback );

function safely( fn, arg ) {

	if ( ! fn ) return;
	try {

		fn( arg );

	} catch ( e ) {

		console.error( e );

	}

}

const isAttribute = ( id ) => typeof id === 'string' && id.startsWith( 'attr:' );

/** Linear RGB of "no value" on the colour channel: the grey of the categories without a hue. */
function neutralColor( intensity ) {

	const c = categoryColor( - 1 );
	const gain = emissiveGain( c[ 0 ], c[ 1 ], c[ 2 ], intensity );
	return [ gain * c[ 0 ], gain * c[ 1 ], gain * c[ 2 ] ];

}

/**
 * Catalogue entries for the node attributes of a graph ( graph.attributes, see the data package ):
 * numbers as sequential metrics on the robust scale, categories as categorical ones.
 */
function attributeMetrics( graph ) {

	const out = [];
	const attrs = graph && graph.attributes;
	if ( ! attrs || typeof attrs !== 'object' ) return out;
	for ( const name of Object.keys( attrs ) ) {

		const a = attrs[ name ];
		if ( ! a || ! a.values || a.values.length < graph.N ) continue;
		if ( a.type === 'category' ) {

			// categories by frequency: the eight most frequent get the distinct hues
			const counts = new Map();
			for ( let i = 0; i < graph.N; i ++ ) if ( a.values[ i ] >= 0 ) counts.set( a.values[ i ], ( counts.get( a.values[ i ] ) || 0 ) + 1 );
			const order = [ ...counts.entries() ].sort( ( x, y ) => y[ 1 ] - x[ 1 ] || x[ 0 ] - y[ 0 ] ).map( ( e ) => e[ 0 ] );
			out.push( {
				id: 'attr:' + name, label: name, scale: 'rank', cost: 'instant', kind: 'categorical', attribute: name,
				description: `Attribute of the file: ${order.length} ${order.length === 1 ? 'category' : 'categories'}.`,
				order, categories: order.map( ( v ) => ( a.categories && a.categories[ v ] !== undefined ? String( a.categories[ v ] ) : String( v ) ) ),
			} );

		} else if ( a.type === 'number' ) {

			out.push( { id: 'attr:' + name, label: name, scale: 'robust', cost: 'instant', kind: 'sequential', attribute: name, description: 'Numeric attribute of the file.' } );

		}

	}

	return out;

}

/** Raw lens values of an attribute: the numbers, or the frequency rank of the category (−1 = none). */
function attributeValues( graph, info ) {

	const a = graph.attributes[ info.attribute ], N = graph.N;
	const raw = new Float32Array( N );
	if ( info.kind === 'categorical' ) {

		const rank = new Map( info.order.map( ( v, k ) => [ v, k ] ) );
		for ( let i = 0; i < N; i ++ ) raw[ i ] = rank.has( a.values[ i ] ) ? rank.get( a.values[ i ] ) : - 1;

	} else for ( let i = 0; i < N; i ++ ) raw[ i ] = a.values[ i ];

	return raw;

}

export class Lens {

	constructor( options = {} ) {

		this.order = 25;
		this.isLens = true;
		this.options = { ramp: 'viridis', intensity: 0.4, top: 0.05, sizeRange: SIZE_RANGE, sizeGamma: 1.5, preview: 200, refineMs: 20000, untilExact: true, mirror: true, onChange: null, ...options };
		this.options.sizeRange = sizeRange( this.options.sizeRange );
		this.options.intensity = unit( this.options.intensity, 0.4 );
		this.client = options.client || new AnalysisClient( { preview: this.options.preview } );
		this._ownsClient = ! options.client;
		this.app = null;
		this.buffers = null;
		this.graph = null;
		this.base = null;
		this.radii = null;
		this.colors = null;
		this._size = null;
		this._color = null;
		this._glow = null;
		this._records = new Map(); // metric id -> record
		this._edges = new Map(); // 'id:scale' -> { result, promise, listeners }
		this._glowCount = 0;
		this._glowMin = NaN;
		this._palette = 'paper';
		this._sources = [];
		this._generation = 0;
		this._summary = null;
		this._lastProgress = - Infinity;
		this._attributes = []; // catalogue entries of the graph's node attributes

	}

	get metrics() {

		return this._attributes.length ? [ ...NODE_METRICS, ...this._attributes ] : NODE_METRICS;

	}

	get edgeMetrics() {

		return EDGE_METRICS;

	}

	get ramps() {

		return rampList();

	}

	get state() {

		const metrics = {};
		let busy = false;
		for ( const rec of this._records.values() ) {

			metrics[ rec.id ] = {
				ready: !! rec.raw, final: rec.final, exact: rec.exact, pivots: rec.pivots, total: rec.total,
				progress: rec.progress, error: rec.error ? rec.error.message : null,
			};
			if ( rec.request ) busy = true;

		}

		return {
			size: this._size ? this._size.metric : null,
			color: this._color ? this._color.metric : null,
			glow: this._glow ? this._glow.metric : null,
			ramp: this._color ? this._color.ramp : this.options.ramp,
			reverse: this._color ? this._color.reverse : false,
			intensity: this._color ? this._color.intensity : this.options.intensity,
			top: this._glow ? this._glow.top : this.options.top,
			palette: this._palette,
			sizeRange: ( this._size ? this._size.range : this.options.sizeRange ).slice(),
			sources: this._sources.slice(),
			busy, metrics,
		};

	}

	// ── module hooks ──────────────────────────────────────────────────────────────────────

	init( app ) {

		this.app = app;
		this._onSelect = ( e ) => {

			const i = e.detail ? e.detail.index : - 1;
			this.setSources( i >= 0 ? [ i ] : [] );

		};

		app.events.addEventListener( 'select', this._onSelect );

	}

	setGraph( buffers, graph ) {

		this._generation ++;
		const old = this.graph, fresh = buffers !== this.buffers;
		// mirror: the graph the lens leaves (or is given again) gets its own values back
		if ( old && this.base && this.options.mirror ) {

			old.radii.set( this.base.radii );
			old.colors.set( this.base.colors );

		}

		this.buffers = buffers;
		this.graph = graph;
		this._records.clear();
		this._edges.clear();
		this._summary = null;
		this._sources = [];
		this._glowCount = 0;
		this._palette = graph.palette || this._palette;
		this._attributes = attributeMetrics( graph );
		// a channel on an attribute the new graph does not have falls back to the graph's own look
		for ( const key of [ '_size', '_color', '_glow' ] ) if ( this[ key ] && ! this._info( this[ key ].metric ) ) this[ key ] = null;
		this.base = { radii: Float32Array.from( graph.radii ), colors: Float32Array.from( graph.colors ) };
		if ( this.options.mirror ) {

			this.radii = graph.radii;
			this.colors = graph.colors;
			// the same graph with new buffers: they were filled from arrays that held lens values
			if ( graph === old ) buffers.setColors( graph.colors, graph.radii, { immediate: fresh } );

		} else {

			this.radii = Float32Array.from( graph.radii );
			this.colors = Float32Array.from( graph.colors );

		}

		this.client.setGraph( graph );
		// the lens stays on across graphs: ask for the active metrics of the new graph
		for ( const ch of [ this._size, this._color, this._glow ] ) if ( ch ) this._ensure( ch.metric );
		this._notify( 'graph' );

	}

	dispose() {

		if ( this.app && this._onSelect ) this.app.events.removeEventListener( 'select', this._onSelect );
		this._generation ++;
		// leave the picture as the graph defines it: cluster colours, degree radii, no glow, no
		// value, on the buffers as well as in the mirrored arrays of the graph
		this._size = this._color = this._glow = null;
		this._apply();
		if ( this._ownsClient ) this.client.dispose();
		else {

			for ( const rec of this._records.values() ) if ( rec.request ) this.client.cancel( 'lens:' + rec.id );
			this.releaseEdgeChannel();

		}

		this._records.clear();
		this._edges.clear();

	}

	// ── channels ──────────────────────────────────────────────────────────────────────────

	setSize( metric, opts = {} ) {

		if ( metric === null || metric === undefined || metric === '' ) return this._off( '_size' );
		const gamma = opts.gamma > 0 ? + opts.gamma : this.options.sizeGamma;
		return this._on( '_size', { metric, scale: opts.scale || null, gamma, range: sizeRange( opts.range || this.options.sizeRange ) } );

	}

	setColor( metric, opts = {} ) {

		if ( metric === null || metric === undefined || metric === '' ) return this._off( '_color' );
		const ramp = opts.ramp || ( this._color ? this._color.ramp : this.options.ramp );
		if ( ! RAMPS[ ramp ] ) throw new RangeError( `Lens: unknown ramp "${ramp}" (${Object.keys( RAMPS ).join( ', ' )})` );
		const intensity = unit( opts.intensity, this._color ? this._color.intensity : this.options.intensity );
		return this._on( '_color', { metric, scale: opts.scale || null, gamma: opts.gamma > 0 ? + opts.gamma : 1, ramp, reverse: !! opts.reverse, intensity } );

	}

	setGlow( metric, opts = {} ) {

		if ( metric === null || metric === undefined || metric === '' ) return this._off( '_glow' );
		const top = opts.top === undefined ? this.options.top : + opts.top;
		return this._on( '_glow', { metric, top: Math.min( 1, Math.max( 1e-6, top ) ) } );

	}

	clear() {

		this._size = this._color = this._glow = null;
		this._apply();
		this._release();
		this._notify( 'config' );

	}

	/** Cluster palette shown while no colour lens is active (and by the node card, the legend, …). */
	setPalette( name ) {

		this._palette = name;
		if ( ! this.graph ) return;
		this.graph.palette = name;
		this.base.colors = colorize( this.graph.leafRank, name );
		this._apply();
		this._notify( 'config' );

	}

	/** Restart set of the 'relevance' metric (personalised PageRank): node indices. */
	setSources( nodes ) {

		const N = this.graph ? this.graph.N : 0;
		const next = Array.from( nodes || [] ).filter( ( i ) => Number.isInteger( i ) && i >= 0 && i < N );
		if ( next.length === this._sources.length && next.every( ( v, k ) => v === this._sources[ k ] ) ) return;
		this._sources = next;
		const rec = this._records.get( 'relevance' );
		if ( ! rec ) return;
		rec.final = false;
		rec.request = null; // a request in flight is superseded through its slot, or cancelled below
		if ( next.length === 0 ) {

			this.client.cancel( 'lens:relevance' );
			this._store( rec, null, { final: true, exact: true, pivots: 0, total: 0 } );

		} else if ( rec.pins > 0 || this._uses( 'relevance' ) ) this._ensure( 'relevance' );

	}

	/**
	 * Continues a slow metric that stopped at the time limit, for `ms` more (0 = until exact).
	 * Resolves true when that run has ended (see values( id ).exact), false when it was cut short.
	 */
	refine( metric, ms = this.options.refineMs ) {

		const rec = this._records.get( metric );
		if ( ! rec || ( ! rec.raw && ! rec.request ) ) return Promise.resolve( false );
		if ( rec.exact ) return Promise.resolve( true );
		rec.pins ++; // keep it running even when no channel shows it
		if ( ! rec.request ) {

			rec.final = false;
			this._ensure( metric, ms );

		}

		return this._whenFinal( rec, () => true ).then( ( ok ) => {

			rec.pins --;
			return ok;

		} );

	}

	// ── values for the UI ─────────────────────────────────────────────────────────────────

	values( metric ) {

		const rec = this._records.get( metric );
		if ( ! rec || ! rec.raw ) return null;
		const lens = this;
		return {
			id: rec.id, label: rec.info.label, scale: rec.info.scale, raw: rec.raw,
			get normalized() {

				return lens._normalized( rec, rec.info.scale );

			},
			get rank() {

				return lens._ranks( rec ).rank;

			},
			get order() {

				return lens._ranks( rec ).order;

			},
			get domain() {

				return lens._domain( rec );

			},
			final: rec.final, exact: rec.exact, pivots: rec.pivots, total: rec.total,
		};

	}

	request( metric ) {

		const rec = this._ensure( metric );
		if ( rec.final || ! rec.request ) return Promise.resolve( this.values( metric ) );
		rec.pins ++;
		return rec.request.then( () => {

			rec.pins --;
			this._release();
			return this._records.get( metric ) === rec ? this.values( metric ) : null;

		} );

	}

	load( ids ) {

		const list = ids || NODE_METRICS.filter( ( m ) => m.cost !== 'slow' && ! m.needs ).map( ( m ) => m.id );
		return Promise.all( list.map( ( id ) => this.request( id ) ) ).then( () => undefined );

	}

	describe( i ) {

		const out = [];
		if ( ! this.graph || ! ( i >= 0 && i < this.graph.N ) ) return out;
		for ( const info of NODE_METRICS ) {

			const rec = this._records.get( info.id );
			if ( ! rec || ! rec.raw ) continue;
			out.push( {
				id: info.id, label: info.label, value: rec.raw[ i ], normalized: this._normalized( rec, info.scale )[ i ],
				rank: this._ranks( rec ).rank[ i ] + 1, of: this.graph.N, exact: rec.exact, final: rec.final,
			} );

		}

		return out;

	}

	legend( channel ) {

		const ch = channel === 'size' ? this._size : channel === 'glow' ? this._glow : this._color;
		if ( ! ch ) return null;
		const rec = this._records.get( ch.metric );
		const domain = rec && rec.raw ? this._domain( rec ) : [ NaN, NaN ];
		const out = { metric: ch.metric, label: rec ? rec.info.label : ch.metric, scale: ch.scale || ( rec ? rec.info.scale : null ), min: domain[ 0 ], max: domain[ 1 ], ready: !! ( rec && rec.raw ) };
		out.ticks = rec && rec.raw && ch !== this._glow ? this._ticks( rec, out.scale, ch.gamma || 1 ) : [];
		// a category that is not given is the entry "none" of the categories, not a missing number
		out.missing = rec && rec.raw && rec.info.kind !== 'categorical' ? this._missing( rec ) : 0;
		if ( ch === this._color ) {

			out.categorical = !! rec && rec.info.kind === 'categorical';
			if ( out.categorical && rec.raw ) out.categories = this._categories( rec, ch.intensity );
			if ( out.missing > 0 ) out.missingColor = neutralColor( ch.intensity );
			out.ramp = ch.ramp;
			out.css = rampCss( ch.ramp, { reverse: ch.reverse } );
			out.intensity = ch.intensity;
			out.gamma = ch.gamma;

		}

		if ( ch === this._glow ) {

			out.top = ch.top;
			out.count = this._glowCount;
			out.threshold = this._glowCount > 0 ? this._glowMin : NaN;

		}

		if ( ch === this._size ) {

			out.range = ch.range.slice();
			out.gamma = ch.gamma;

		}

		return out;

	}

	attributes( i ) {

		const out = [];
		const g = this.graph;
		if ( ! g || ! g.attributes || ! ( i >= 0 && i < g.N ) ) return out;
		for ( const info of this._attributes ) {

			const a = g.attributes[ info.attribute ];
			const v = a.values[ i ];
			if ( a.type === 'category' ) out.push( { id: info.id, name: info.label, type: 'category', value: v, text: v >= 0 && a.categories && a.categories[ v ] !== undefined ? String( a.categories[ v ] ) : '' } );
			else out.push( { id: info.id, name: info.label, type: 'number', value: v, text: Number.isFinite( v ) ? String( v ) : '' } );

		}

		return out;

	}

	summary( opts = {} ) {

		if ( ! this.graph ) return Promise.resolve( null );
		if ( ! this._summary ) {

			const s = this._summary = { partial: null, listeners: new Set(), promise: null };
			s.promise = this.client.summary( {
				top: opts.top, maxBfs: opts.maxBfs,
				onPreview: ( partial ) => {

					s.partial = partial;
					for ( const cb of s.listeners ) safely( cb, partial );

				},
			} ).then( ( r ) => {

				if ( r.cancelled ) {

					if ( this._summary === s ) this._summary = null;
					return null;

				}

				s.partial = r;
				return r;

			}, ( e ) => {

				if ( this._summary === s ) this._summary = null;
				throw e;

			} );

		}

		const s = this._summary;
		if ( opts.onPreview ) {

			s.listeners.add( opts.onPreview );
			if ( s.partial ) safely( opts.onPreview, s.partial );

		}

		return s.promise;

	}

	edgeChannel( metric, opts = {} ) {

		const info = metricInfo( metric, 'edge' );
		if ( ! info ) throw new RangeError( `Lens: unknown edge metric "${metric}" (${EDGE_METRICS.map( ( m ) => m.id ).join( ', ' )})` );
		if ( ! this.graph ) return Promise.resolve( null );
		const scale = opts.scale || info.scale, key = `${metric}:${scale}`;
		let ch = this._edges.get( key );
		if ( ! ch ) this._edges.set( key, ch = { result: null, promise: null, listeners: new Set() } );
		if ( ch.result && ch.result.exact ) return Promise.resolve( ch.result );
		if ( opts.onUpdate ) ch.listeners.add( opts.onUpdate );
		if ( ch.promise ) {

			// already on its way: one request serves every caller of the channel
			if ( ch.result ) safely( opts.onUpdate, ch.result );
			return ch.promise;

		}

		const generation = this._generation, slow = info.cost === 'slow';
		const pack = ( r ) => ( { metric, scale, values: r.normalized, raw: r.values, final: r.final, exact: r.exact, pivots: r.pivots, total: r.total } );
		const settle = () => {

			if ( ch.promise !== promise ) return;
			ch.promise = null;
			ch.listeners.clear();

		};

		// a channel is asked for to be shown: refine it until exact, like a node metric on a channel
		const limit = opts.maxMs !== undefined ? Math.max( 0, + opts.maxMs || 0 ) : this.options.untilExact ? 0 : this.options.refineMs;
		// a slot per channel: channels asked for together must not supersede each other
		const promise = ch.promise = this.client.metric( metric, {
			target: 'edge', slot: 'lens:edge:' + key, normalize: scale,
			options: slow ? { samples: 'exact' } : {},
			preview: slow ? Math.max( 500, this.options.preview ) : 0,
			maxMs: slow ? limit : 0,
			onPreview: ( r ) => {

				if ( generation !== this._generation ) return;
				ch.result = pack( r );
				for ( const cb of [ ...ch.listeners ] ) safely( cb, ch.result );

			},
		} ).then( ( r ) => {

			settle();
			if ( r.cancelled || generation !== this._generation ) return null;
			return ( ch.result = pack( r ) );

		}, ( e ) => {

			settle();
			throw e;

		} );
		return promise;

	}

	/**
	 * Stops the refinement of an edge channel that is no longer shown (of all of them without a
	 * metric); its edgeChannel() promise resolves null. What was computed stays in the worker and
	 * is continued by the next edgeChannel() call.
	 */
	releaseEdgeChannel( metric, opts = {} ) {

		const info = metric === undefined ? null : metricInfo( metric, 'edge' );
		const only = info ? `${metric}:${opts.scale || info.scale}` : null;
		if ( metric !== undefined && ! info ) return;
		for ( const [ key, ch ] of this._edges ) if ( ch.promise && ( only === null || key === only ) ) this.client.cancel( 'lens:edge:' + key );

	}

	// ── internals ─────────────────────────────────────────────────────────────────────────

	_on( key, channel ) {

		// (without a graph an attribute cannot be checked yet: setGraph() drops what the graph lacks)
		if ( ! this._info( channel.metric ) && ( this.graph || ! isAttribute( channel.metric ) ) ) throw new RangeError( `Lens: unknown metric "${channel.metric}" (${this.metrics.map( ( m ) => m.id ).join( ', ' )})` );
		this[ key ] = channel;
		if ( ! this.graph ) return Promise.resolve( false ); // applied when a graph arrives
		const rec = this._ensure( channel.metric );
		this._apply(); // values that are already there show at once
		this._release();
		this._notify( 'config' );
		return this._whenFinal( rec, () => this[ key ] === channel );

	}

	_off( key ) {

		this[ key ] = null;
		this._apply();
		this._release();
		this._notify( 'config' );
		return Promise.resolve( true );

	}

	_uses( metric ) {

		return ( !! this._size && this._size.metric === metric ) || ( !! this._color && this._color.metric === metric ) || ( !! this._glow && this._glow.metric === metric );

	}

	_whenFinal( rec, current ) {

		if ( rec.final ) return Promise.resolve( current() );
		if ( ! rec.request ) return Promise.resolve( false );
		return rec.request.then( ( ok ) => ok && current() );

	}

	/**
	 * The record of a metric; starts its request unless the values are final or on their way.
	 * `limit`: ms a slow metric may refine (0 = until exact). Left out, a metric that a channel
	 * shows refines until exact (`untilExact`) and any other for `refineMs`.
	 */
	_ensure( metric, limit ) {

		let rec = this._records.get( metric );
		if ( ! rec ) {

			const info = this._info( metric );
			if ( ! info ) throw new RangeError( `Lens: unknown metric "${metric}"` );
			rec = { id: metric, info, raw: null, final: false, exact: false, pivots: 0, total: 0, progress: 0, cache: new Map(), ranks: null, domain: null, ticks: null, missing: - 1, request: null, open: false, pins: 0, error: null };
			this._records.set( metric, rec );

		}

		if ( rec.info.attribute !== undefined ) {

			// a node attribute of the file: the values are at hand, no worker involved
			if ( ! rec.final && this.graph ) {

				rec.raw = attributeValues( this.graph, rec.info );
				rec.missing = - 1;
				rec.final = rec.exact = true;
				rec.progress = 1;

			}

			return rec;

		}

		const slow = rec.info.cost === 'slow', generation = this._generation;
		const shown = this.options.untilExact && this._uses( metric );
		// an estimate that stopped at its time limit is taken up again when a channel shows it
		if ( rec.final && slow && rec.raw && ! rec.exact && shown && limit === undefined ) rec.final = false;
		if ( rec.final || rec.request || ! this.graph ) return rec;
		if ( rec.info.needs === 'sources' && this._sources.length === 0 ) {

			rec.final = true; // nothing to compute: the channel shows the graph's own values
			return rec;

		}

		const maxMs = ! slow ? 0 : limit !== undefined ? Math.max( 0, + limit || 0 ) : shown ? 0 : this.options.refineMs;
		// open: no time limit only because a channel shows the metric (see _release)
		rec.open = slow && limit === undefined && shown && this.options.refineMs > 0;
		const current = () => generation === this._generation && this._records.get( metric ) === rec;
		let tracked = null;
		const done = ( ok ) => {

			if ( rec.request === tracked ) rec.request = null;
			return ok;

		};

		rec.error = null;
		tracked = rec.request = this.client.metric( metric, {
			slot: 'lens:' + metric,
			options: slow ? { samples: 'exact' } : rec.info.needs === 'sources' ? { sources: this._sources.slice() } : {},
			preview: slow ? this.options.preview : 0,
			maxMs,
			onProgress: ( p ) => {

				if ( ! current() ) return;
				rec.progress = p.fraction;
				this._notify( 'progress' );

			},
			onPreview: ( r ) => {

				if ( current() && rec.request === tracked ) this._store( rec, r.values, r );

			},
		} ).then( ( r ) => {

			if ( r.cancelled || ! current() || rec.request !== tracked ) {

				// replaced by a newer request for the same record (other sources, another time
				// limit): whoever waits for this one waits for that one
				if ( current() && rec.request && rec.request !== tracked ) return rec.request;
				return done( false );

			}

			done( true );
			// the time limit ran out, but a channel shows the metric by now: keep refining
			if ( slow && r.final && ! r.exact && this.options.untilExact && this._uses( metric ) ) {

				rec.final = false;
				this._ensure( metric );
				this._store( rec, r.values, { ...r, final: false } );
				if ( rec.request ) return rec.request;

			} else this._store( rec, r.values, r );

			return true;

		}, ( e ) => {

			console.error( e );
			if ( current() ) rec.error = e;
			return done( false );

		} );
		return rec;

	}

	_store( rec, raw, r ) {

		rec.raw = raw;
		rec.final = r.final;
		rec.exact = r.exact;
		rec.pivots = r.pivots;
		rec.total = r.total;
		rec.progress = r.total > 0 ? r.pivots / r.total : r.final ? 1 : rec.progress;
		rec.cache.clear();
		rec.ranks = null;
		rec.domain = null;
		rec.ticks = null;
		rec.missing = - 1;
		if ( this._uses( rec.id ) ) this._apply();
		this._notify( 'data' );

	}

	/**
	 * Stops requests nobody is waiting for (their partial values stay; the worker parks the task),
	 * and puts a refinement without time limit back under `refineMs` when no channel shows the
	 * metric any more and only a request() / refine() waits for it.
	 */
	_release() {

		for ( const rec of this._records.values() ) {

			if ( ! rec.request || this._uses( rec.id ) ) continue;
			if ( rec.pins === 0 ) {

				rec.request = null;
				this.client.cancel( 'lens:' + rec.id );

			} else if ( rec.open ) {

				rec.request = null;
				this._ensure( rec.id ); // supersedes the open request; the worker carries on from its state

			}

		}

	}

	_normalized( rec, scale, gamma = 1 ) {

		const key = gamma === 1 ? scale : `${scale}^${gamma}`;
		let t = rec.cache.get( key );
		if ( ! t ) {

			t = normalize( rec.raw, { mode: scale, gamma } );
			rec.cache.set( key, t );

		}

		return t;

	}

	_ranks( rec ) {

		if ( ! rec.ranks ) {

			const raw = rec.raw, n = raw.length;
			const order = new Uint32Array( n );
			for ( let i = 0; i < n; i ++ ) order[ i ] = i;
			// largest first; values that are not numbers (a missing attribute) go last
			order.sort( ( a, b ) => ( raw[ b ] - raw[ a ] ) || ( ( raw[ a ] !== raw[ a ] ) - ( raw[ b ] !== raw[ b ] ) ) || a - b );
			const rank = new Uint32Array( n );
			for ( let k = 1; k < n; k ++ ) rank[ order[ k ] ] = raw[ order[ k ] ] === raw[ order[ k - 1 ] ] ? rank[ order[ k - 1 ] ] : k;
			rec.ranks = { order, rank };

		}

		return rec.ranks;

	}

	_domain( rec ) {

		if ( ! rec.domain ) {

			let min = Infinity, max = - Infinity;
			for ( let i = 0; i < rec.raw.length; i ++ ) {

				const v = rec.raw[ i ];
				if ( v < min ) min = v;
				if ( v > max ) max = v;

			}

			rec.domain = [ min, max ];

		}

		return rec.domain;

	}

	/** The number of nodes without a finite value (an attribute the file does not give for them). */
	_missing( rec ) {

		if ( rec.missing < 0 ) {

			let n = 0;
			for ( let i = 0; i < rec.raw.length; i ++ ) if ( ! Number.isFinite( rec.raw[ i ] ) ) n ++;
			rec.missing = n;

		}

		return rec.missing;

	}

	/** Catalogue entry of a node metric: the library's, or an attribute of the current graph. */
	_info( metric ) {

		return metricInfo( metric ) || this._attributes.find( ( m ) => m.id === metric ) || null;

	}

	/**
	 * What the ends and the middle of a channel show, read back from the data instead of derived
	 * from the scale's formula (so it is true for every scale and gamma): for t = 0, 0.5 and 1 the
	 * value of a node that sits there. An end is a bound ( '≤', '≥' ) when values beyond it exist
	 * and are clamped onto it. The middle is left out when no node is near it.
	 */
	_ticks( rec, scale, gamma = 1 ) {

		const key = `${scale}^${gamma}`;
		if ( ! rec.ticks ) rec.ticks = new Map();
		let ticks = rec.ticks.get( key );
		if ( ticks ) return ticks;
		const raw = rec.raw, t = this._normalized( rec, scale, gamma ), n = raw.length;
		const [ min, max ] = this._domain( rec );
		let lo = - Infinity, hi = Infinity, mid = NaN, midGap = Infinity;
		for ( let i = 0; i < n; i ++ ) {

			const v = raw[ i ];
			if ( ! Number.isFinite( v ) ) continue;
			if ( t[ i ] <= 0 && v > lo ) lo = v;
			if ( t[ i ] >= 1 && v < hi ) hi = v;
			const gap = Math.abs( t[ i ] - 0.5 );
			if ( gap < midGap ) {

				midGap = gap;
				mid = v;

			}

		}

		ticks = [];
		if ( Number.isFinite( lo ) ) ticks.push( { t: 0, value: lo, bound: lo > min ? '≤' : '' } );
		else if ( Number.isFinite( min ) ) ticks.push( { t: 0, value: min, bound: '' } );
		if ( midGap <= 0.12 && max > min ) ticks.push( { t: 0.5, value: mid, bound: '' } );
		if ( Number.isFinite( hi ) ) ticks.push( { t: 1, value: hi, bound: hi < max ? '≥' : '' } );
		else if ( Number.isFinite( max ) ) ticks.push( { t: 1, value: max, bound: '' } );
		rec.ticks.set( key, ticks );
		return ticks;

	}

	/** The labels of a categorical metric with their counts and node colours, most frequent first. */
	_categories( rec, intensity = 1 ) {

		const raw = rec.raw, counts = new Map();
		for ( let i = 0; i < raw.length; i ++ ) counts.set( raw[ i ], ( counts.get( raw[ i ] ) || 0 ) + 1 );
		const names = rec.info.categories || null;
		const c = [ 0, 0, 0 ];
		return [ ...counts.entries() ].sort( ( a, b ) => ( a[ 0 ] < 0 ) - ( b[ 0 ] < 0 ) || a[ 0 ] - b[ 0 ] ).map( ( [ value, count ] ) => {

			categoryColor( value, c );
			const gain = emissiveGain( c[ 0 ], c[ 1 ], c[ 2 ], intensity );
			const label = value < 0 ? 'none' : names ? String( names[ value ] ) : rec.id === 'component' ? `Component ${value + 1}` : String( value );
			return { value, label, count, color: [ gain * c[ 0 ], gain * c[ 1 ], gain * c[ 2 ] ], distinct: value >= 0 && value < 8 };

		} );

	}

	/**
	 * The record whose values a channel shows: false when the channel is off (or its metric has
	 * nothing to show), null while the metric has no values yet (the channel keeps its picture).
	 */
	_source( ch ) {

		if ( ! ch ) return false;
		const rec = this._records.get( ch.metric );
		if ( ! rec ) return null;
		if ( ! rec.raw ) return rec.final ? false : null;
		return rec;

	}

	/** Values of a channel in [0,1] as it maps them (its scale and gamma); false / null as _source(). */
	_channel( ch ) {

		const rec = this._source( ch );
		return rec ? this._normalized( rec, ch.scale || rec.info.scale, ch.gamma || 1 ) : rec;

	}

	/**
	 * Glow weights into fx[ 4 i + 2 ]: 1 − rank / K for the K = ⌈top · N⌉ largest values, by
	 * groups of equal values. A group that ends beyond K fades, and is off at 3 K: a plateau
	 * that large is not "the top". Returns the number of nodes that glow.
	 */
	_writeGlow( rec, top, fx ) {

		const raw = rec.raw, N = raw.length, order = this._ranks( rec ).order;
		for ( let i = 0; i < N; i ++ ) fx[ 4 * i + 2 ] = 0;
		const K = Math.max( 1, Math.ceil( top * N - 1e-9 ) );
		let count = 0;
		this._glowMin = NaN;
		for ( let k = 0; k < N && k < K; ) {

			let end = k + 1; // the group of equal values at positions k … end − 1 has rank k
			while ( end < N && raw[ order[ end ] ] === raw[ order[ k ] ] ) end ++;
			const w = ( 1 - k / K ) * Math.min( 1, Math.max( 0, ( 3 * K - end ) / ( 2 * K ) ) );
			if ( w > 0 ) {

				for ( let j = k; j < end; j ++ ) fx[ 4 * order[ j ] + 2 ] = w;
				count += end - k;
				this._glowMin = raw[ order[ k ] ];

			}

			k = end;

		}

		return count;

	}

	/** Recomputes the targets of all channels and uploads them. */
	_apply() {

		const { graph, buffers } = this;
		if ( ! graph || ! buffers ) return;
		const N = graph.N, radii = this.radii, colors = this.colors;

		const sizeT = this._channel( this._size );
		if ( sizeT === false ) radii.set( this.base.radii );
		else if ( sizeT ) {

			const [ lo, hi ] = this._size.range;
			for ( let i = 0; i < N; i ++ ) radii[ i ] = lo + ( hi - lo ) * sizeT[ i ];

		}

		const colorT = this._channel( this._color );
		if ( colorT === false ) colors.set( this.base.colors );
		else if ( colorT ) {

			const rec = this._records.get( this._color.metric ), intensity = this._color.intensity;
			if ( rec.info.kind === 'categorical' ) {

				const c = [ 0, 0, 0 ];
				for ( let i = 0; i < N; i ++ ) {

					categoryColor( rec.raw[ i ], c );
					const gain = emissiveGain( c[ 0 ], c[ 1 ], c[ 2 ], intensity );
					colors[ 4 * i ] = gain * c[ 0 ]; colors[ 4 * i + 1 ] = gain * c[ 1 ]; colors[ 4 * i + 2 ] = gain * c[ 2 ]; colors[ 4 * i + 3 ] = 1;

				}

			} else {

				rampColors( colorT, this._color.ramp, colors, this._color.reverse, intensity );
				if ( this._missing( rec ) > 0 ) {

					// no value is not the smallest value: the neutral grey, not the low end of the ramp
					const c = neutralColor( intensity ), raw = rec.raw;
					for ( let i = 0; i < N; i ++ ) {

						if ( Number.isFinite( raw[ i ] ) ) continue;
						colors[ 4 * i ] = c[ 0 ]; colors[ 4 * i + 1 ] = c[ 1 ]; colors[ 4 * i + 2 ] = c[ 2 ];

					}

				}

			}

		}

		buffers.setColors( colors, radii );

		// glow and value: only z / w of the effect channels are ours
		const fx = buffers.nodeFx.value.array;
		const glow = this._source( this._glow );
		if ( glow === false ) {

			for ( let i = 0; i < N; i ++ ) fx[ 4 * i + 2 ] = 0;
			this._glowCount = 0;

		} else if ( glow ) this._glowCount = this._writeGlow( glow, this._glow.top, fx );

		// the value channel is the plain normalised metric: the scale of its channel, no gamma
		const primary = this._color || this._size || this._glow;
		const valueOf = this._source( primary );
		const valueT = valueOf ? this._normalized( valueOf, primary.scale || valueOf.info.scale ) : valueOf;
		if ( valueT === false ) for ( let i = 0; i < N; i ++ ) fx[ 4 * i + 3 ] = 0;
		else if ( valueT ) for ( let i = 0; i < N; i ++ ) fx[ 4 * i + 3 ] = valueT[ i ];
		buffers.setFx( fx );

	}

	_notify( reason ) {

		if ( reason === 'progress' ) {

			const t = now();
			if ( t - this._lastProgress < 200 ) return;
			this._lastProgress = t;

		}

		const events = this.app && this.app.events;
		if ( ! events && ! this.options.onChange ) return;
		const detail = { ...this.state, reason };
		safely( this.options.onChange, detail );
		if ( events ) events.dispatchEvent( new CustomEvent( 'lens', { detail } ) );

	}

}
