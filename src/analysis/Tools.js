// Interactive analysis tools (App module, order 25): ripple, k-hop neighbourhood and shortest
// paths. The data side only: every tool runs a traversal of src/graph/analytics.js on the main
// thread (a BFS is about 2 ms at 384k edges, a path query a few microseconds), writes its result
// into the per-node effect channels and the ripple uniforms of GraphBuffers, and dispatches an
// event; renderers and the interaction module decide how it looks.
//
// ─── What it writes ─────────────────────────────────────────────────────────────────────────
//  nodeFx.x   hop distance from the ripple source(s) while a ripple runs, otherwise from the
//             k-hop source(s); −1 = none / not reached
//  nodeFx.y   position along the highlighted path in [0,1] (hops from the start / path length;
//             0 for a path of one node); −1 = not on it
//  uniforms.rippleFront   the wavefront, in hops: 0 → maxDist + rippleWidth, eased (fast start,
//                         slow finish); −1e6 while idle
//  uniforms.rippleGain    0 → gain within the first 8 % of the run, fading to 60 % of it toward
//                         the end and to 0 over the last 10 %; 0 while idle
//  uniforms.rippleWidth   half width of the pulse in hops (default 0.75)
//  z / w of nodeFx belong to the lens and are never touched: the CPU array is rewritten in
//  place and handed back through buffers.setFx().
//
// ─── API ────────────────────────────────────────────────────────────────────────────────────
//  const tools = app.use( new Tools( { rippleSpeed: 2.5 } ) );
//  tools.ripple( source | sources, { duration?, gain = 1, width = 0.75 } )
//        -> { sources, dist: Int32Array, maxDist, reached, duration (s), finished: Promise<boolean> } | null
//        Restartable: a new ripple replaces the running one (whose `finished` resolves false).
//        duration defaults to ( maxDist + 1 ) / rippleSpeed, within 1.2 … 4 s.
//  tools.khop( source | sources, k = 2 ) -> { sources, k, nodes, hops, levelOffsets, maxHop } | null
//        nodes nearest first with their hop counts in parallel
//  tools.path( a, b, { all = false } ) -> { nodes, edges, length, source, target, all, pathCount,
//        depth, nodeShare, edgeShare } | null
//        One shortest path, or with `all` every node and edge on some shortest path (the
//        geodesic DAG) with nodeShare / edgeShare = fraction of the shortest paths through each
//        (1 everywhere for a single path). null when b cannot be reached (the old path is cleared).
//  tools.clear( what? )          'ripple' | 'khop' | 'path'; nothing = all three (also done by
//                                dispose())
//  tools.highlightSet()  -> Float32Array( N ) | null
//        What a selection highlight should show for the active tools, as one weight per node:
//        1 = primary (path ends, k-hop sources), 0.35 … 0.85 = secondary (path interior by its
//        share of the geodesics, neighbourhood by hop), 0 = context. null when no tool is active.
//  tools.stateTargets( out? ) -> Float32Array( 4N ) | null
//        The same in the layout of nodeStateTarget (x selected, y neighbour, z dimmed, w 0) for
//        Interaction to merge; this module never writes nodeStateTarget itself.
//  tools.state           { ripple: { sources, maxDist, front, gain, progress } | null,
//                          khop: { sources, k, count } | null, path: { source, target, length, all, pathCount } | null }
//
//  Events on app.events: 'ripple' { phase: 'start' | 'end', sources, maxDist, duration, completed },
//  'khop' ( the khop result | null ), 'path' ( the path result | null; nodes and edges are
//  Uint32Array as the contract asks ), 'highlight' ( whenever highlightSet() changed ). Every
//  start has its end: clear(), dispose() and a new graph send 'ripple' end ( completed: false ),
//  'khop' null, 'path' null and 'highlight' for what was active.

import { bfs, khopNeighborhood, shortestPath, shortestPathDag } from '../graph/analytics.js';

const IDLE_FRONT = - 1e6;

export class Tools {

	constructor( options = {} ) {

		this.order = 25;
		this.isTools = true;
		this.options = { rippleSpeed: 2.5, rippleMin: 1.2, rippleMax: 4, rippleWidth: 0.75, ...options };
		this.app = null;
		this.buffers = null;
		this.graph = null;
		this._ripple = null;
		this._khop = null;
		this._path = null;

	}

	get state() {

		const r = this._ripple, k = this._khop, p = this._path;
		const u = this.buffers ? this.buffers.uniforms : null;
		return {
			ripple: r ? { sources: r.sources, maxDist: r.maxDist, front: u.rippleFront.value, gain: u.rippleGain.value, progress: Math.min( 1, r.time / r.duration ) } : null,
			khop: k ? { sources: k.sources, k: k.k, count: k.nodes.length } : null,
			path: p ? { source: p.source, target: p.target, length: p.length, all: p.all, pathCount: p.pathCount } : null,
		};

	}

	// ── module hooks ──────────────────────────────────────────────────────────────────────

	init( app ) {

		this.app = app;

	}

	setGraph( buffers, graph ) {

		// the old buffers are gone: end what was running without writing to them, and tell the
		// listeners (the new buffers start idle: front −1e6, gain 0, x / y −1)
		const r = this._ripple, k = this._khop, p = this._path;
		this._ripple = this._khop = this._path = null;
		this.buffers = buffers;
		this.graph = graph;
		if ( r ) {

			r.resolve( false );
			this._emit( 'ripple', { phase: 'end', sources: r.sources, maxDist: r.maxDist, duration: r.duration, completed: false } );

		}

		if ( k ) this._emit( 'khop', null );
		if ( p ) this._emit( 'path', null );
		if ( k || p ) this._emit( 'highlight', null );

	}

	update( dt ) {

		const r = this._ripple;
		if ( ! r ) return;
		r.time += dt;
		const t = Math.min( 1, r.time / r.duration );
		const u = this.buffers.uniforms;
		// the front decelerates a little, like a wave that spreads; the pulse loses energy on the way
		u.rippleFront.value = ( r.maxDist + r.width ) * ( 1 - Math.pow( 1 - t, 1.6 ) );
		const fadeIn = Math.min( 1, t / 0.08 ), fadeOut = Math.min( 1, ( 1 - t ) / 0.1 );
		u.rippleGain.value = r.gain * fadeIn * fadeOut * ( 1 - 0.4 * t );
		if ( t >= 1 ) this._endRipple( true );

	}

	dispose() {

		// idle uniforms, x / y back to −1, and the closing events of whatever was active
		this.clear();

	}

	// ── tools ─────────────────────────────────────────────────────────────────────────────

	ripple( sources, opts = {} ) {

		const list = this._nodes( sources );
		if ( ! list.length ) return null;
		const old = this._ripple;
		this._ripple = null;
		if ( old ) {

			old.resolve( false );
			this._emit( 'ripple', { phase: 'end', sources: old.sources, maxDist: old.maxDist, duration: old.duration, completed: false } );

		}

		const b = bfs( this.graph, list );
		const fx = this.buffers.nodeFx.value.array, N = this.graph.N;
		for ( let i = 0; i < N; i ++ ) fx[ 4 * i ] = b.dist[ i ];
		this.buffers.setFx( fx );

		const o = this.options;
		const duration = opts.duration > 0 ? + opts.duration : Math.min( o.rippleMax, Math.max( o.rippleMin, ( b.maxDist + 1 ) / o.rippleSpeed ) );
		const width = opts.width > 0 ? + opts.width : o.rippleWidth;
		let resolve;
		const finished = new Promise( ( r ) => ( resolve = r ) );
		this._ripple = { sources: list, maxDist: b.maxDist, duration, width, gain: opts.gain === undefined ? 1 : + opts.gain, time: 0, resolve };
		const u = this.buffers.uniforms;
		u.rippleWidth.value = width;
		u.rippleFront.value = 0;
		u.rippleGain.value = 0;
		this._emit( 'ripple', { phase: 'start', sources: list, maxDist: b.maxDist, duration, completed: false } );
		return { sources: list, dist: b.dist, maxDist: b.maxDist, reached: b.reached, duration, finished };

	}

	khop( sources, k = 2 ) {

		const list = this._nodes( sources );
		if ( ! list.length ) return null;
		k = Math.max( 0, Math.floor( + k || 0 ) );
		const r = khopNeighborhood( this.graph, list, k );
		this._khop = { sources: list, k, nodes: r.nodes, hops: r.hops, levelOffsets: r.levelOffsets, maxHop: r.maxHop, dist: r.dist };
		if ( ! this._ripple ) this._writeHops(); // a running ripple owns nodeFx.x until it ends
		const out = { sources: list, k, nodes: r.nodes, hops: r.hops, levelOffsets: r.levelOffsets, maxHop: r.maxHop };
		this._emit( 'khop', out );
		this._emit( 'highlight', null );
		return out;

	}

	path( a, b, opts = {} ) {

		const g = this.graph;
		if ( ! g || ! this._valid( a ) || ! this._valid( b ) ) return null;
		const all = !! opts.all;
		let out = null;
		if ( all ) {

			const d = shortestPathDag( g, a, b );
			if ( d ) {

				const nodeShare = new Float32Array( d.nodes.length ), edgeShare = new Float32Array( d.edges.length );
				for ( let i = 0; i < nodeShare.length; i ++ ) nodeShare[ i ] = d.nodePaths[ i ] / d.pathCount;
				for ( let i = 0; i < edgeShare.length; i ++ ) edgeShare[ i ] = d.edgePaths[ i ] / d.pathCount;
				out = { nodes: d.nodes, edges: d.edges, length: d.length, source: a, target: b, all, pathCount: d.pathCount, depth: d.depth, nodeShare, edgeShare };

			}

		} else {

			const p = shortestPath( g, a, b );
			if ( p ) {

				const depth = new Uint32Array( p.nodes.length );
				for ( let i = 0; i < depth.length; i ++ ) depth[ i ] = i;
				out = { nodes: p.nodes, edges: p.edges, length: p.length, source: a, target: b, all, pathCount: 1, depth, nodeShare: new Float32Array( p.nodes.length ).fill( 1 ), edgeShare: new Float32Array( p.edges.length ).fill( 1 ) };

			}

		}

		this._path = out;
		this._writePath();
		this._emit( 'path', out );
		this._emit( 'highlight', null );
		return out;

	}

	clear( what ) {

		if ( ! this.buffers ) return;
		if ( ( what === undefined || what === 'khop' ) && this._khop ) {

			this._khop = null;
			if ( ! this._ripple ) this._writeHops();
			this._emit( 'khop', null );
			this._emit( 'highlight', null );

		}

		if ( ( what === undefined || what === 'path' ) && this._path ) {

			this._path = null;
			this._writePath();
			this._emit( 'path', null );
			this._emit( 'highlight', null );

		}

		if ( ( what === undefined || what === 'ripple' ) && this._ripple ) this._endRipple( false );

	}

	highlightSet() {

		const g = this.graph, k = this._khop, p = this._path;
		if ( ! g || ( ! k && ! p ) ) return null;
		const w = new Float32Array( g.N );
		if ( k ) {

			// 1 at the sources, then 0.85 down to 0.35 at the last ring
			for ( let i = 0; i < k.nodes.length; i ++ ) w[ k.nodes[ i ] ] = k.hops[ i ] === 0 ? 1 : 0.85 - 0.5 * ( k.hops[ i ] - 1 ) / Math.max( 1, k.k - 1 );

		}

		if ( p ) {

			for ( let i = 0; i < p.nodes.length; i ++ ) {

				const v = p.nodes[ i ];
				w[ v ] = Math.max( w[ v ], v === p.source || v === p.target ? 1 : 0.35 + 0.5 * p.nodeShare[ i ] );

			}

		}

		return w;

	}

	stateTargets( out ) {

		const w = this.highlightSet();
		if ( ! w ) return null;
		const N = w.length;
		const t = out && out.length >= 4 * N ? out : new Float32Array( 4 * N );
		for ( let i = 0; i < N; i ++ ) {

			t[ 4 * i ] = w[ i ] >= 1 ? 1 : 0;
			t[ 4 * i + 1 ] = w[ i ] < 1 ? w[ i ] : 0;
			t[ 4 * i + 2 ] = w[ i ] > 0 ? 0 : 1;
			t[ 4 * i + 3 ] = 0;

		}

		return t;

	}

	// ── internals ─────────────────────────────────────────────────────────────────────────

	_valid( i ) {

		return !! this.graph && Number.isInteger( i ) && i >= 0 && i < this.graph.N;

	}

	_nodes( sources ) {

		if ( ! this.graph || ! this.buffers || sources === null || sources === undefined ) return [];
		const list = typeof sources === 'number' ? [ sources ] : Array.from( sources );
		return [ ...new Set( list.filter( ( i ) => this._valid( i ) ) ) ];

	}

	/** nodeFx.x ← hop distances of the k-hop neighbourhood, or −1 everywhere. */
	_writeHops() {

		const fx = this.buffers.nodeFx.value.array, N = this.graph.N, k = this._khop;
		for ( let i = 0; i < N; i ++ ) fx[ 4 * i ] = k ? k.dist[ i ] : - 1;
		this.buffers.setFx( fx );

	}

	/** nodeFx.y ← position along the path, or −1 everywhere. */
	_writePath() {

		const fx = this.buffers.nodeFx.value.array, N = this.graph.N, p = this._path;
		for ( let i = 0; i < N; i ++ ) fx[ 4 * i + 1 ] = - 1;
		if ( p ) for ( let i = 0; i < p.nodes.length; i ++ ) fx[ 4 * p.nodes[ i ] + 1 ] = p.length > 0 ? p.depth[ i ] / p.length : 0;
		this.buffers.setFx( fx );

	}

	_endRipple( completed ) {

		const r = this._ripple;
		if ( ! r ) return;
		this._ripple = null;
		const u = this.buffers.uniforms;
		u.rippleFront.value = IDLE_FRONT;
		u.rippleGain.value = 0;
		this._writeHops();
		r.resolve( completed );
		this._emit( 'ripple', { phase: 'end', sources: r.sources, maxDist: r.maxDist, duration: r.duration, completed } );

	}

	_emit( type, detail ) {

		const events = this.app && this.app.events;
		if ( events ) events.dispatchEvent( new CustomEvent( type, { detail } ) );

	}

}
