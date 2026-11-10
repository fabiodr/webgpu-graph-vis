// Analysis tools on small graphs: what ripple, k-hop and path write into nodeFx and the ripple
// uniforms, what they return and dispatch, and what they leave alone.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as GA from '../src/graph/analytics.js';
import { Tools } from '../src/analysis/Tools.js';

// ── helpers ──────────────────────────────────────────────────────────────────────────────────

/** The parts of GraphBuffers the tools use. z / w carry made-up lens data. */
function fakeBuffers( graph ) {

	const N = graph.N;
	const fx = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) {

		fx[ 4 * i ] = fx[ 4 * i + 1 ] = - 1;
		fx[ 4 * i + 2 ] = ( i % 3 ) / 2;
		fx[ 4 * i + 3 ] = ( i % 4 ) / 3;

	}

	const b = {
		N, graph, fxCalls: 0,
		nodeFx: { value: { array: fx, needsUpdate: false } },
		uniforms: { rippleFront: { value: - 1e6 }, rippleWidth: { value: 0.75 }, rippleGain: { value: 0 } },
		setFx( fx4 ) {

			b.nodeFx.value.array.set( fx4 );
			b.nodeFx.value.needsUpdate = true;
			b.fxCalls ++;

		},
	};
	return b;

}

function setup( graph, options ) {

	const app = { events: new EventTarget() };
	const events = [];
	for ( const type of [ 'ripple', 'khop', 'path', 'highlight' ] ) app.events.addEventListener( type, ( e ) => events.push( { type, detail: e.detail } ) );
	const tools = new Tools( options );
	tools.init( app );
	const buffers = fakeBuffers( graph );
	tools.setGraph( buffers, graph );
	const col = ( k ) => Array.from( { length: graph.N }, ( _, i ) => buffers.nodeFx.value.array[ 4 * i + k ] );
	const lensData = () => [ col( 2 ), col( 3 ) ];
	return { tools, buffers, app, events, col, lensData };

}

const path5 = () => GA.graphFromEdges( 7, [ 0, 1, 1, 2, 2, 3, 3, 4, 5, 6 ] ); // 0-1-2-3-4 and 5-6
const cycle4 = () => GA.graphFromEdges( 4, [ 0, 1, 1, 2, 2, 3, 3, 0 ] );
function grid( w, h ) {

	const edges = [];
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

		if ( x + 1 < w ) edges.push( y * w + x, y * w + x + 1 );
		if ( y + 1 < h ) edges.push( y * w + x, ( y + 1 ) * w + x );

	}

	return GA.graphFromEdges( w * h, edges );

}

const near = ( a, b, eps = 1e-6 ) => assert.ok( Math.abs( a - b ) <= eps, `${a} vs ${b}` );

// ── ripple ───────────────────────────────────────────────────────────────────────────────────

test( 'Tools.ripple: hop distances in nodeFx.x, eased front and gain, cleared at the end', async () => {

	const g = path5();
	const { tools, buffers, events, col, lensData } = setup( g );
	const lens = lensData();
	assert.equal( tools.order, 25 );
	const r = tools.ripple( 0 );
	assert.deepEqual( r.sources, [ 0 ] );
	assert.deepEqual( Array.from( r.dist ), [ 0, 1, 2, 3, 4, - 1, - 1 ] );
	assert.equal( r.maxDist, 4 );
	assert.equal( r.reached, 5 );
	near( r.duration, 5 / 2.5 ); // ( maxDist + 1 ) / rippleSpeed
	assert.deepEqual( col( 0 ), [ 0, 1, 2, 3, 4, - 1, - 1 ], 'unreached nodes stay at -1' );
	assert.deepEqual( col( 1 ), new Array( 7 ).fill( - 1 ), 'nodeFx.y untouched' );
	assert.deepEqual( lensData(), lens, 'nodeFx.z / .w untouched' );
	const u = buffers.uniforms;
	assert.equal( u.rippleFront.value, 0 );
	assert.equal( u.rippleGain.value, 0 );
	assert.equal( u.rippleWidth.value, 0.75 );
	assert.deepEqual( events.map( ( e ) => [ e.type, e.detail.phase ] ), [ [ 'ripple', 'start' ] ] );
	assert.equal( tools.state.ripple.maxDist, 4 );

	// animate in 20 steps of duration / 20
	const fronts = [], gains = [];
	let finished = null;
	r.finished.then( ( ok ) => ( finished = ok ) );
	for ( let k = 0; k < 19; k ++ ) {

		tools.update( r.duration / 20 );
		fronts.push( u.rippleFront.value );
		gains.push( u.rippleGain.value );

	}

	for ( let k = 1; k < fronts.length; k ++ ) assert.ok( fronts[ k ] > fronts[ k - 1 ], 'the front only moves outward' );
	assert.ok( fronts[ 0 ] > 0 && fronts.at( - 1 ) < 4.75 && fronts.at( - 1 ) > 4.6, `front ${fronts[ 0 ]} .. ${fronts.at( - 1 )}` );
	// eased: fast at the start, slow at the end
	assert.ok( fronts[ 1 ] - fronts[ 0 ] > 1.5 * ( fronts[ 18 ] - fronts[ 17 ] ) );
	// it passes every level: the front reaches hop 4 before the gain is gone
	const reach = fronts.findIndex( ( f ) => f >= 4 );
	assert.ok( reach > 0 && gains[ reach ] > 0.3, `gain ${gains[ reach ]} when the front arrives at the last level` );
	assert.ok( Math.max( ...gains ) > 0.9 && Math.max( ...gains ) <= 1 );
	assert.ok( gains[ 0 ] < Math.max( ...gains ), 'the gain fades in' );
	assert.ok( gains.at( - 1 ) < 0.4, 'and out' );
	assert.ok( gains.every( ( v ) => v >= 0 ) );
	assert.equal( finished, null );
	near( tools.state.ripple.progress, 0.95, 1e-6 );

	tools.update( r.duration / 20 + 1e-6 );
	assert.equal( u.rippleFront.value, - 1e6 );
	assert.equal( u.rippleGain.value, 0 );
	assert.deepEqual( col( 0 ), new Array( 7 ).fill( - 1 ), 'hop distances are cleared' );
	assert.deepEqual( lensData(), lens );
	assert.equal( await r.finished, true );
	assert.equal( tools.state.ripple, null );
	assert.deepEqual( events.at( - 1 ).detail, { phase: 'end', sources: [ 0 ], maxDist: 4, duration: r.duration, completed: true } );
	// idle updates do nothing
	const calls = buffers.fxCalls;
	tools.update( 1 );
	assert.equal( buffers.fxCalls, calls );

} );

test( 'Tools.ripple: several sources, restart, options, invalid input', async () => {

	const g = path5();
	const { tools, buffers, events, col } = setup( g, { rippleSpeed: 1, rippleMin: 0.5, rippleMax: 3 } );
	const a = tools.ripple( [ 0, 4, 4, 99 ] );
	assert.deepEqual( a.sources, [ 0, 4 ], 'duplicates and invalid nodes are dropped' );
	assert.deepEqual( col( 0 ), [ 0, 1, 2, 1, 0, - 1, - 1 ] );
	assert.equal( a.duration, 3, '( 2 + 1 ) / 1 s' );
	tools.update( 1 );
	assert.ok( buffers.uniforms.rippleFront.value > 0 );

	// restart from the other component while the first one runs
	const b = tools.ripple( 5, { duration: 0.5, gain: 0.5, width: 1.5 } );
	assert.equal( await a.finished, false, 'the replaced ripple reports that it did not finish' );
	assert.deepEqual( col( 0 ), [ - 1, - 1, - 1, - 1, - 1, 0, 1 ] );
	assert.equal( buffers.uniforms.rippleFront.value, 0, 'the front starts again' );
	assert.equal( buffers.uniforms.rippleWidth.value, 1.5 );
	assert.equal( b.duration, 0.5 );
	tools.update( 0.25 );
	assert.ok( buffers.uniforms.rippleGain.value <= 0.5 && buffers.uniforms.rippleGain.value > 0.3 );
	near( buffers.uniforms.rippleFront.value, ( 1 + 1.5 ) * ( 1 - Math.pow( 0.5, 1.6 ) ) );
	tools.update( 0.3 );
	assert.equal( await b.finished, true );
	assert.deepEqual( events.filter( ( e ) => e.type === 'ripple' ).map( ( e ) => [ e.detail.phase, e.detail.completed ] ), [ [ 'start', false ], [ 'end', false ], [ 'start', false ], [ 'end', true ] ] );

	// duration clamps
	assert.equal( tools.ripple( 5 ).duration, 2 ); // ( 1 + 1 ) / 1
	assert.equal( tools.ripple( 6, { duration: 9 } ).duration, 9 );
	tools.clear( 'ripple' );
	assert.equal( buffers.uniforms.rippleGain.value, 0 );
	assert.equal( buffers.uniforms.rippleFront.value, - 1e6 );

	assert.equal( tools.ripple( 99 ), null );
	assert.equal( tools.ripple( [] ), null );
	assert.equal( tools.ripple( null ), null );
	assert.equal( tools.ripple( 1.5 ), null );

	// a long chain is capped at rippleMax, a single isolated node gets rippleMin
	const lone = setup( GA.graphFromEdges( 3, [ 0, 1 ] ), { rippleSpeed: 10 } );
	assert.equal( lone.tools.ripple( 2 ).duration, 1.2 );
	assert.equal( lone.tools.ripple( 2 ).maxDist, 0 );

} );

test( 'Tools: a new graph ends what was active, says so, and leaves the old buffers alone', async () => {

	const g = path5();
	const { tools, buffers, events, app } = setup( g );
	const r = tools.ripple( 2 );
	tools.khop( 0, 1 );
	tools.path( 0, 4 );
	const calls = buffers.fxCalls, before = Float32Array.from( buffers.nodeFx.value.array );
	const front = buffers.uniforms.rippleFront.value;
	events.length = 0;
	// what a listener sees while the closing events arrive: the tools are already empty
	const seen = [];
	for ( const type of [ 'ripple', 'khop', 'path', 'highlight' ] ) app.events.addEventListener( type, () => seen.push( tools.highlightSet() === null && tools.state.ripple === null ) );
	const next = fakeBuffers( cycle4() );
	tools.setGraph( next, next.graph );
	assert.equal( await r.finished, false );
	assert.equal( buffers.fxCalls, calls );
	assert.deepEqual( buffers.nodeFx.value.array, before );
	assert.equal( buffers.uniforms.rippleFront.value, front );
	// every start gets its end: a renderer that pairs them does not keep the old state
	assert.deepEqual( events, [
		{ type: 'ripple', detail: { phase: 'end', sources: [ 2 ], maxDist: 2, duration: r.duration, completed: false } },
		{ type: 'khop', detail: null },
		{ type: 'path', detail: null },
		{ type: 'highlight', detail: null },
	] );
	assert.deepEqual( seen, [ true, true, true, true ] );
	assert.deepEqual( tools.state, { ripple: null, khop: null, path: null } );
	assert.equal( tools.highlightSet(), null );
	tools.update( 0.5 );
	assert.equal( next.fxCalls, 0 );
	assert.equal( next.uniforms.rippleFront.value, - 1e6 );

	// nothing active: a graph change is silent
	events.length = 0;
	const third = fakeBuffers( cycle4() );
	tools.setGraph( third, third.graph );
	assert.deepEqual( events, [] );
	// and the tools work on the new graph
	assert.deepEqual( Array.from( tools.khop( 0, 1 ).nodes ).sort(), [ 0, 1, 3 ] );
	tools.dispose();

} );

test( 'Tools.dispose: idle uniforms, x / y cleared, closing events; z / w stay', async () => {

	const g = path5();
	const { tools, buffers, events, col, lensData } = setup( g );
	const lens = lensData();
	tools.khop( 0, 2 );
	tools.path( 0, 3 );
	const r = tools.ripple( 1 );
	tools.update( r.duration / 2 );
	const u = buffers.uniforms;
	assert.ok( u.rippleFront.value > 0 && u.rippleGain.value > 0, 'mid-pulse' );
	events.length = 0;
	tools.dispose();
	assert.equal( u.rippleFront.value, - 1e6 );
	assert.equal( u.rippleGain.value, 0 );
	assert.deepEqual( col( 0 ), new Array( 7 ).fill( - 1 ) );
	assert.deepEqual( col( 1 ), new Array( 7 ).fill( - 1 ) );
	assert.deepEqual( lensData(), lens );
	assert.equal( await r.finished, false );
	assert.deepEqual( events.map( ( e ) => [ e.type, e.detail && e.detail.phase ? e.detail.phase : e.detail ] ), [
		[ 'khop', null ], [ 'highlight', null ], [ 'path', null ], [ 'highlight', null ], [ 'ripple', 'end' ],
	] );
	assert.equal( events.at( - 1 ).detail.completed, false );
	assert.deepEqual( tools.state, { ripple: null, khop: null, path: null } );
	// before any graph: nothing to do
	new Tools().dispose();

} );

// ── k-hop ────────────────────────────────────────────────────────────────────────────────────

test( 'Tools.khop: neighbourhood with hop counts; shares nodeFx.x with the ripple', async () => {

	const g = grid( 5, 5 );
	const { tools, events, col, lensData } = setup( g );
	const lens = lensData();
	const k = tools.khop( 12, 2 ); // centre of the grid
	assert.deepEqual( k.sources, [ 12 ] );
	assert.equal( k.k, 2 );
	assert.equal( k.maxHop, 2 );
	assert.equal( k.nodes.length, 13, '1 + 4 + 8 nodes within two steps of the centre' );
	assert.equal( k.nodes[ 0 ], 12 );
	assert.deepEqual( Array.from( k.hops ), [ 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 ] );
	assert.deepEqual( Array.from( k.levelOffsets ), [ 0, 1, 5, 13 ] );
	const manhattan = ( i ) => Math.abs( ( i % 5 ) - 2 ) + Math.abs( Math.floor( i / 5 ) - 2 );
	assert.deepEqual( col( 0 ), Array.from( { length: 25 }, ( _, i ) => ( manhattan( i ) <= 2 ? manhattan( i ) : - 1 ) ) );
	assert.deepEqual( lensData(), lens );
	assert.deepEqual( tools.state.khop, { sources: [ 12 ], k: 2, count: 13 } );
	assert.equal( events.find( ( e ) => e.type === 'khop' ).detail, k );

	// a ripple borrows the channel and gives it back
	const r = tools.ripple( 0, { duration: 1 } );
	assert.equal( col( 0 )[ 24 ], 8 );
	tools.khop( 12, 1 ); // changed while the ripple runs: written when the ripple ends
	assert.equal( col( 0 )[ 24 ], 8 );
	tools.update( 1.01 );
	assert.equal( await r.finished, true );
	assert.deepEqual( col( 0 ), Array.from( { length: 25 }, ( _, i ) => ( manhattan( i ) <= 1 ? manhattan( i ) : - 1 ) ) );

	// k = 0, several sources, clear
	assert.deepEqual( Array.from( tools.khop( 3, 0 ).nodes ), [ 3 ] );
	const two = tools.khop( [ 0, 24 ], 1 );
	assert.deepEqual( Array.from( two.nodes ).sort( ( a, b ) => a - b ), [ 0, 1, 5, 19, 23, 24 ] );
	tools.clear( 'khop' );
	assert.deepEqual( col( 0 ), new Array( 25 ).fill( - 1 ) );
	assert.equal( events.filter( ( e ) => e.type === 'khop' ).at( - 1 ).detail, null );
	assert.equal( tools.khop( - 1, 2 ), null );

} );

// ── paths ────────────────────────────────────────────────────────────────────────────────────

test( 'Tools.path: one shortest path, nodeFx.y along it, the path event, unreachable targets', () => {

	const g = path5();
	const { tools, events, col, lensData } = setup( g );
	const lens = lensData();
	const p = tools.path( 4, 1 );
	assert.ok( p.nodes instanceof Uint32Array && p.edges instanceof Uint32Array );
	assert.deepEqual( Array.from( p.nodes ), [ 4, 3, 2, 1 ] );
	assert.deepEqual( Array.from( p.edges ), [ 3, 2, 1 ] );
	assert.equal( p.length, 3 );
	assert.deepEqual( [ p.source, p.target, p.all, p.pathCount ], [ 4, 1, false, 1 ] );
	// every edge id really joins consecutive nodes
	p.edges.forEach( ( e, k ) => assert.deepEqual( [ g.edges[ 2 * e ], g.edges[ 2 * e + 1 ] ].sort(), [ p.nodes[ k ], p.nodes[ k + 1 ] ].sort() ) );
	const y = col( 1 );
	assert.equal( y[ 0 ], - 1 );
	near( y[ 4 ], 0 ); near( y[ 3 ], 1 / 3 ); near( y[ 2 ], 2 / 3 ); near( y[ 1 ], 1 );
	assert.deepEqual( [ y[ 5 ], y[ 6 ] ], [ - 1, - 1 ] );
	assert.deepEqual( col( 0 ), new Array( 7 ).fill( - 1 ), 'nodeFx.x untouched' );
	assert.deepEqual( lensData(), lens );
	const event = events.filter( ( e ) => e.type === 'path' ).at( - 1 );
	assert.equal( event.detail, p );
	assert.deepEqual( tools.state.path, { source: 4, target: 1, length: 3, all: false, pathCount: 1 } );

	// a new path replaces the old one
	tools.path( 5, 6 );
	assert.deepEqual( col( 1 ), [ - 1, - 1, - 1, - 1, - 1, 0, 1 ] );

	// same node: one node, position 0
	const self = tools.path( 2, 2 );
	assert.deepEqual( [ Array.from( self.nodes ), self.edges.length, self.length ], [ [ 2 ], 0, 0 ] );
	assert.deepEqual( col( 1 ), [ - 1, - 1, 0, - 1, - 1, - 1, - 1 ] );

	// unreachable: null, the old highlight is removed, the event says so
	assert.equal( tools.path( 0, 6 ), null );
	assert.deepEqual( col( 1 ), new Array( 7 ).fill( - 1 ) );
	assert.equal( events.filter( ( e ) => e.type === 'path' ).at( - 1 ).detail, null );
	assert.equal( tools.state.path, null );

	// invalid nodes: null, nothing changes
	tools.path( 0, 2 );
	const count = events.length;
	assert.equal( tools.path( 0, 70 ), null );
	assert.equal( tools.path( 0.5, 1 ), null );
	assert.equal( events.length, count );
	assert.deepEqual( tools.state.path.target, 2 );

	tools.clear();
	assert.deepEqual( col( 1 ), new Array( 7 ).fill( - 1 ) );
	assert.equal( events.at( - 2 ).type, 'path' );
	assert.equal( events.at( - 2 ).detail, null );

} );

test( 'Tools.path all: the geodesic DAG with shares; matches the library on a grid', () => {

	const { tools, col } = setup( cycle4() );
	const p = tools.path( 0, 2, { all: true } );
	assert.equal( p.length, 2 );
	assert.equal( p.pathCount, 2 );
	assert.deepEqual( Array.from( p.nodes ).sort(), [ 0, 1, 2, 3 ] );
	assert.equal( p.nodes[ 0 ], 0 );
	assert.equal( p.nodes.at( - 1 ), 2 );
	assert.deepEqual( Array.from( p.edges ).sort(), [ 0, 1, 2, 3 ] );
	const share = Object.fromEntries( Array.from( p.nodes, ( v, i ) => [ v, p.nodeShare[ i ] ] ) );
	assert.deepEqual( share, { 0: 1, 1: 0.5, 2: 1, 3: 0.5 } );
	assert.ok( p.edgeShare.every( ( s ) => s === 0.5 ) );
	assert.deepEqual( col( 1 ), [ 0, 0.5, 1, 0.5 ] );

	// corner to corner of a 4 x 4 grid: C( 6, 3 ) = 20 geodesics through all 16 nodes and 24 edges
	const g = grid( 4, 4 );
	const big = setup( g );
	const d = big.tools.path( 0, 15, { all: true } );
	const ref = GA.shortestPathDag( g, 0, 15 );
	assert.equal( d.pathCount, 20 );
	assert.equal( d.length, 6 );
	assert.deepEqual( Array.from( d.nodes ), Array.from( ref.nodes ) );
	assert.deepEqual( Array.from( d.edges ), Array.from( ref.edges ) );
	d.nodes.forEach( ( v, i ) => {

		near( d.nodeShare[ i ], ref.nodePaths[ i ] / 20 );
		near( big.col( 1 )[ v ], ( ( v % 4 ) + Math.floor( v / 4 ) ) / 6 ); // depth = Manhattan distance from the corner

	} );
	assert.ok( d.nodeShare.every( ( s ) => s > 0 && s <= 1 ) );

	// a single path is one of them
	const one = big.tools.path( 0, 15 );
	assert.equal( one.length, 6 );
	assert.ok( Array.from( one.nodes ).every( ( v ) => ref.nodes.includes( v ) ) );
	assert.ok( one.nodeShare.every( ( s ) => s === 1 ) );

} );

// ── highlight ────────────────────────────────────────────────────────────────────────────────

test( 'Tools.highlightSet / stateTargets: weights for the selection highlight', () => {

	const g = grid( 5, 5 );
	const { tools, events } = setup( g );
	assert.equal( tools.highlightSet(), null );
	assert.equal( tools.stateTargets(), null );

	tools.khop( 12, 2 );
	let w = tools.highlightSet();
	assert.ok( w instanceof Float32Array && w.length === 25 );
	assert.equal( w[ 12 ], 1 );
	near( w[ 7 ], 0.85 ); near( w[ 11 ], 0.85 );
	near( w[ 2 ], 0.35 ); near( w[ 6 ], 0.35 );
	assert.equal( w[ 0 ], 0 );
	assert.equal( w.filter( ( v ) => v > 0 ).length, 13 );

	let t = tools.stateTargets();
	assert.equal( t.length, 100 );
	assert.deepEqual( Array.from( t.subarray( 48, 52 ) ), [ 1, 0, 0, 0 ], 'the source is "selected"' );
	near( t[ 4 * 7 + 1 ], 0.85 );
	assert.deepEqual( [ t[ 4 * 7 ], t[ 4 * 7 + 2 ] ], [ 0, 0 ] );
	assert.deepEqual( Array.from( t.subarray( 0, 4 ) ), [ 0, 0, 1, 0 ], 'everything else is dimmed' );
	const out = new Float32Array( 100 ).fill( 9 );
	assert.equal( tools.stateTargets( out ), out );
	assert.ok( out.every( ( v ) => v !== 9 ) );

	// a path on top: its ends are primary, the interior weighted by its share of the geodesics
	tools.path( 0, 24, { all: true } );
	w = tools.highlightSet();
	assert.equal( w[ 0 ], 1 );
	assert.equal( w[ 24 ], 1 );
	assert.equal( w[ 12 ], 1, 'the k-hop source stays primary' );
	assert.ok( w[ 4 ] > 0.35 && w[ 4 ] < 0.4, 'a corner node carries one of 70 geodesics' );
	assert.ok( w.every( ( v ) => v > 0 && v <= 1 ), 'all 25 nodes lie on some corner-to-corner geodesic' );
	tools.clear( 'khop' );
	w = tools.highlightSet();
	assert.ok( w[ 12 ] < 1 && w[ 12 ] > 0.35 + 0.5 * 0.5, 'the centre carries more than half of them' );
	tools.clear();
	assert.equal( tools.highlightSet(), null );
	assert.ok( events.filter( ( e ) => e.type === 'highlight' ).length >= 4 );

	// k = 1: one ring at 0.85
	tools.khop( 0, 1 );
	w = tools.highlightSet();
	near( w[ 1 ], 0.85 ); near( w[ 5 ], 0.85 );

} );

test( 'Tools.highlightSet( what ): the neighbourhood and the path as separate sets', () => {

	const g = grid( 5, 5 );
	const { tools } = setup( g );
	assert.equal( tools.highlightSet( 'khop' ), null );
	assert.equal( tools.highlightSet( 'path' ), null );
	tools.khop( 12, 1 );
	const ring = tools.highlightSet( 'khop' );
	assert.deepEqual( Array.from( ring ).map( ( v, i ) => ( v > 0 ? i : - 1 ) ).filter( ( i ) => i >= 0 ), [ 7, 11, 12, 13, 17 ] );
	assert.equal( tools.highlightSet( 'path' ), null, 'no path yet' );
	tools.path( 0, 4 );
	const path = tools.highlightSet( 'path' );
	assert.deepEqual( Array.from( path ).map( ( v, i ) => ( v > 0 ? i : - 1 ) ).filter( ( i ) => i >= 0 ), [ 0, 1, 2, 3, 4 ] );
	assert.deepEqual( [ path[ 0 ], path[ 4 ] ], [ 1, 1 ], 'the ends are primary' );
	assert.equal( path[ 12 ], 0, 'the neighbourhood is not part of the path set' );
	assert.deepEqual( Array.from( tools.highlightSet( 'khop' ) ), Array.from( ring ), 'and the path not part of the neighbourhood' );
	// without an argument: both, as before
	const both = tools.highlightSet();
	assert.equal( both.filter( ( v ) => v > 0 ).length, 10 );
	tools.clear( 'khop' );
	assert.equal( tools.highlightSet( 'khop' ), null );
	assert.ok( tools.highlightSet( 'path' ) );
	tools.clear();
	assert.equal( tools.highlightSet( 'path' ), null );

} );
