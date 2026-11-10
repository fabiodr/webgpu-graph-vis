// State of the explore layer as plain JSON (share links): the round trip, what happens to a
// state of another graph or a hand-edited one, and the lens presets.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { STATE_VERSION, defaultState, normalizeState } from '../src/explore/state.js';
import { LENS_PRESETS, applyPreset, applyChannels, matchPreset } from '../src/explore/presets.js';

const limits = { nodes: 77, clusters: 18, metrics: [ 'degree', 'pagerank', 'betweenness', 'core', 'attr:field' ], edgeMetrics: [ 'betweenness', 'jaccard' ], ramps: [ 'viridis', 'magma', 'ice' ] };

const full = {
	v: STATE_VERSION,
	lens: { color: 'betweenness', size: 'pagerank', glow: 'degree', ramp: 'magma', reverse: true, top: 0.1, edge: 'jaccard', edgeOpacity: false, edgeWidth: true },
	selection: 11, hops: 2, path: { from: 11, to: 48, all: true }, cluster: 13, insights: true, top: 'betweenness',
};

test( 'round trip: a state survives JSON and normalizing unchanged', () => {

	const text = JSON.stringify( full );
	assert.deepEqual( normalizeState( JSON.parse( text ), limits ), full );
	assert.deepEqual( JSON.parse( JSON.stringify( normalizeState( full, limits ) ) ), full, 'plain JSON: nothing is lost in a link' );
	// normalizing is idempotent
	const once = normalizeState( { ...full, hops: 9, junk: 1 }, limits );
	assert.deepEqual( normalizeState( once, limits ), once );
	// the default state round-trips too
	assert.deepEqual( normalizeState( defaultState(), limits ), defaultState() );
	assert.equal( defaultState().v, STATE_VERSION );

} );

test( 'anything that is not a state gives the default state', () => {

	for ( const raw of [ null, undefined, 42, 'x', [], {} ] ) assert.deepEqual( normalizeState( raw, limits ), defaultState(), JSON.stringify( raw ) );
	assert.deepEqual( normalizeState( { lens: 'viridis', path: 'a-b', selection: '11', hops: '2', cluster: 1.5, insights: 'yes', top: 'degree; drop' }, limits ), defaultState() );
	const a = normalizeState( null ), b = normalizeState( null );
	a.lens.color = 'x';
	assert.equal( b.lens.color, null, 'a fresh object every time' );

} );

test( 'a state of another graph: what does not exist here is dropped', () => {

	const s = normalizeState( { ...full, selection: 500, path: { from: 3, to: 900 }, cluster: 18, lens: { ...full.lens, color: 'attr:missing', edge: 'flow', ramp: 'rainbow' } }, limits );
	assert.equal( s.selection, null );
	assert.equal( s.hops, 0, 'a neighbourhood needs its selection' );
	assert.equal( s.path, null );
	assert.equal( s.cluster, null );
	assert.equal( s.lens.color, null );
	assert.equal( s.lens.size, 'pagerank', 'what exists stays' );
	assert.equal( s.lens.edge, null );
	assert.equal( s.lens.ramp, 'viridis' );
	assert.equal( s.top, 'betweenness' );
	// no graph at all
	const none = normalizeState( full, { ...limits, nodes: 0, clusters: 0 } );
	assert.deepEqual( [ none.selection, none.hops, none.path, none.cluster ], [ null, 0, null, null ] );
	assert.equal( none.lens.color, 'betweenness', 'the lens does not depend on the nodes' );

} );

test( 'values are clamped and checked', () => {

	const n = ( patch ) => normalizeState( { ...full, ...patch }, limits );
	assert.equal( n( { hops: 9 } ).hops, 3 );
	assert.equal( n( { hops: - 2 } ).hops, 0 );
	assert.equal( n( { hops: 1.5 } ).hops, 0 );
	assert.equal( n( { selection: - 1 } ).selection, null );
	assert.equal( n( { selection: 76 } ).selection, 76 );
	assert.equal( n( { selection: 77 } ).selection, null );
	assert.equal( n( { path: { from: 4, to: 4 } } ).path, null, 'a path needs two different ends' );
	assert.deepEqual( n( { path: { from: 4, to: 5 } } ).path, { from: 4, to: 5, all: false } );
	assert.equal( n( { top: 'triangles' } ).top, 'degree', 'only the rankings the drawer has' );
	assert.equal( n( { lens: { ...full.lens, top: 0 } } ).lens.top, 0.05 );
	assert.equal( n( { lens: { ...full.lens, top: 2 } } ).lens.top, 0.05 );
	assert.equal( n( { lens: { ...full.lens, top: 1 } } ).lens.top, 1 );
	assert.equal( n( { lens: { color: 'degree' } } ).lens.edgeOpacity, true, 'missing switches take their default' );
	assert.equal( n( { lens: { reverse: 1 } } ).lens.reverse, false, 'only a real true' );
	// without catalogues every id is accepted as a string (the caller checks later)
	assert.equal( normalizeState( { lens: { color: 'anything' } } ).lens.color, 'anything' );
	assert.equal( normalizeState( { lens: { color: 7 } } ).lens.color, null );
	assert.equal( normalizeState( { selection: 123456 } ).selection, 123456 );

} );

// ── presets ──────────────────────────────────────────────────────────────────────────────────

/** Records what a preset asks of the lens. */
function fakeLens() {

	const calls = [];
	return {
		calls,
		setColor: ( metric, opts ) => ( calls.push( [ 'color', metric, opts ] ), Promise.resolve( true ) ),
		setSize: ( metric, opts ) => ( calls.push( [ 'size', metric, opts ] ), Promise.resolve( true ) ),
		setGlow: ( metric, opts ) => ( calls.push( [ 'glow', metric, opts ] ), Promise.resolve( true ) ),
	};

}

test( 'presets: the six of the lens bar, each sets all three channels', async () => {

	assert.deepEqual( LENS_PRESETS.map( ( p ) => p.id ), [ 'influence', 'brokers', 'cohesion', 'reach', 'relevance', 'communities' ] );
	for ( const p of LENS_PRESETS ) assert.ok( p.label && p.description, p.id );

	let lens = fakeLens();
	assert.deepEqual( await applyPreset( lens, 'influence' ), [ true, true, true ] );
	assert.deepEqual( lens.calls, [ [ 'color', 'betweenness', { ramp: 'viridis', reverse: false } ], [ 'size', 'pagerank', undefined ], [ 'glow', null, undefined ] ], 'size PageRank, colour betweenness' );

	lens = fakeLens();
	await applyPreset( lens, 'brokers' );
	assert.deepEqual( lens.calls, [ [ 'color', null, undefined ], [ 'size', 'betweenness', undefined ], [ 'glow', 'betweenness', { top: 0.03 } ] ] );

	lens = fakeLens();
	await applyPreset( lens, LENS_PRESETS.find( ( p ) => p.id === 'communities' ) );
	assert.deepEqual( lens.calls.map( ( c ) => c[ 1 ] ), [ null, null, null ], 'back to the graph\'s own look' );

	lens = fakeLens();
	assert.deepEqual( await applyPreset( lens, 'nope' ), [] );
	assert.deepEqual( lens.calls, [] );

	lens = fakeLens();
	await applyChannels( lens, { color: 'core', ramp: 'ice', reverse: true, glow: 'degree' } );
	assert.deepEqual( lens.calls, [ [ 'color', 'core', { ramp: 'ice', reverse: true } ], [ 'size', null, undefined ], [ 'glow', 'degree', undefined ] ] );

} );

test( 'matchPreset: which chip is on', () => {

	for ( const p of LENS_PRESETS ) assert.equal( matchPreset( { color: p.color, size: p.size, glow: p.glow } ), p.id );
	assert.equal( matchPreset( { color: null, size: null, glow: null } ), 'communities' );
	assert.equal( matchPreset( { color: undefined, size: undefined, glow: undefined } ), 'communities' );
	assert.equal( matchPreset( { color: 'degree', size: null, glow: null } ), null, 'a hand-made combination is no preset' );
	assert.equal( matchPreset( { color: 'betweenness', size: 'pagerank', glow: 'degree' } ), null );
	assert.equal( matchPreset( { color: 'betweenness', size: 'pagerank', glow: null, ramp: 'magma', reverse: true } ), 'influence', 'the ramp is a matter of taste' );
	assert.equal( matchPreset( null ), null );

} );
