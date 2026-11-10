// The tour planner (which stops, in which order, what each caption says) and the community
// figures it and the insights drawer read.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as GA from '../src/graph/analytics.js';
import { planTour, argmax, betweennessShare } from '../src/explore/tourPlan.js';
import { communityList, usefulLevel, labelOf, communityOf } from '../src/explore/communities.js';
import { parseGraphJson } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';

const load = ( name ) => prepareGraph( parseGraphJson( JSON.parse( readFileSync( `public/data/${name}.json`, 'utf8' ) ), name ) );

/** The parts of a PreparedGraph the planner reads, without a hierarchy. */
function plain( N, edges, labels = null, name = '' ) {

	const degree = new Uint32Array( N );
	for ( const [ a, b ] of edges ) {

		degree[ a ] ++;
		degree[ b ] ++;

	}

	return { N, E: edges.length, D: 0, degree, labels, name };

}

test( 'argmax and betweennessShare', () => {

	assert.equal( argmax( [ 1, 5, 5, 2 ] ), 1, 'the first of equals' );
	assert.equal( argmax( [] ), - 1 );
	assert.equal( argmax( new Float32Array( [ 0, 0 ] ) ), 0 );
	assert.equal( argmax( [ 3, 9, 4 ], 1 ), 0, 'only the first N' );
	// a star of five: the centre lies on all 6 pairs of leaves
	assert.equal( betweennessShare( 6, 5 ), 1 );
	assert.equal( betweennessShare( 3, 5 ), 0.5 );
	assert.equal( betweennessShare( 99, 5 ), 1, 'clamped' );
	assert.equal( betweennessShare( 1, 2 ), 0, 'no pair of other nodes' );

} );

test( 'a graph with communities: overview, hub, two communities, broker, outro', () => {

	const g = load( 'imsv/lesmis' );
	const communities = communityList( g, usefulLevel( g ) );
	const betweenness = GA.betweenness( g ); // raw: the number of pairs whose shortest paths pass
	const stops = planTour( g, { betweenness, communities } );
	assert.deepEqual( stops.map( ( s ) => s.id ), [ 'overview', 'hub', 'community-0', 'community-1', 'broker', 'outro' ] );
	assert.deepEqual( stops.map( ( s ) => s.kind ), [ 'overview', 'hub', 'community', 'community', 'broker', 'outro' ] );
	const [ overview, hub, c0, c1, broker, outro ] = stops;
	const valjean = g.labels.indexOf( 'Valjean' );

	assert.equal( overview.title, 'Les Misérables' );
	assert.equal( overview.text, '77 nodes and 254 links in 6 communities; similar colours belong together.' );
	assert.deepEqual( overview.lens, { color: null, size: null, glow: null } );

	assert.equal( hub.node, valjean );
	assert.equal( hub.title, 'Valjean' );
	assert.equal( hub.text, 'The most connected node: 36 links, 5.5× the average.' );
	assert.equal( hub.value, 36 );
	assert.deepEqual( hub.lens, { color: null, size: 'degree', glow: 'degree' } );

	assert.equal( c0.cluster, communities[ 0 ].cluster );
	assert.equal( c0.title, 'Around Valjean' );
	assert.equal( c0.text, 'The largest community: 20 nodes; 57 % of its links stay inside.' );
	assert.equal( c1.title, 'Around Gavroche' );
	assert.equal( c1.text, 'The next largest community: 17 nodes; 75 % of its links stay inside.' );
	assert.equal( c1.level, usefulLevel( g ) );

	assert.equal( broker.node, valjean );
	assert.equal( broker.text, 'Also the strongest broker: 57 % of all shortest paths between other nodes run through it.' );
	assert.ok( Math.abs( broker.value - 0.57 ) < 0.005 );
	assert.deepEqual( broker.lens, { color: 'betweenness', size: 'betweenness', glow: null } );

	assert.equal( outro.title, 'Your turn' );
	assert.ok( stops.every( ( s ) => s.seconds >= 3 && s.seconds <= 8 && s.title && s.text ) );
	assert.match( planTour( g, { touch: true } ).at( - 1 ).text, /^Tap a node/ );

} );

test( 'stops are only planned when their subject exists', () => {

	const g = load( 'imsv/lesmis' );
	const communities = communityList( g, usefulLevel( g ) );
	// no betweenness yet: no broker stop
	assert.deepEqual( planTour( g, { communities } ).map( ( s ) => s.id ), [ 'overview', 'hub', 'community-0', 'community-1', 'outro' ] );
	// no communities: no community stops, and the overview does not mention them
	const flat = planTour( g, {} );
	assert.deepEqual( flat.map( ( s ) => s.id ), [ 'overview', 'hub', 'outro' ] );
	assert.equal( flat[ 0 ].text, '77 nodes and 254 links.' );
	// a single community is no community structure
	assert.deepEqual( planTour( g, { communities: communities.slice( 0, 1 ) } ).map( ( s ) => s.id ), [ 'overview', 'hub', 'outro' ] );
	assert.deepEqual( planTour( g, { communities, maxCommunities: 1 } ).map( ( s ) => s.id ), [ 'overview', 'hub', 'community-0', 'outro' ] );
	assert.deepEqual( planTour( g, { communities, maxCommunities: 0 } ).map( ( s ) => s.id ), [ 'overview', 'hub', 'outro' ] );
	// betweenness of zero everywhere (a clique): nobody is a broker
	assert.equal( planTour( g, { betweenness: new Float32Array( g.N ) } ).some( ( s ) => s.kind === 'broker' ), false );
	// no graph, an empty graph
	assert.deepEqual( planTour( null ), [] );
	assert.deepEqual( planTour( plain( 0, [] ) ), [] );
	// nodes without a single edge: only the overview, and then no "your turn" either
	const dust = planTour( plain( 3, [], [ 'a', 'b', 'c' ] ) );
	assert.deepEqual( dust.map( ( s ) => s.id ), [ 'overview' ] );
	assert.equal( dust[ 0 ].title, 'The whole graph' );
	assert.equal( dust[ 0 ].text, '3 nodes and 0 links.' );

} );

test( 'captions: singular, estimates, unlabeled nodes, a broker that is not the hub', () => {

	// 0–1–2 plus a hub 3 with leaves 4…8, joined through 2–3: the hub is 3, the broker too;
	// with made-up betweenness the broker is node 2
	const edges = [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ], [ 3, 5 ], [ 3, 6 ], [ 3, 7 ], [ 3, 8 ] ];
	const g = plain( 9, edges );
	const b = new Float32Array( 9 );
	b[ 2 ] = 14;
	b[ 3 ] = 7;
	const stops = planTour( g, { betweenness: b, exact: false } );
	assert.deepEqual( stops.map( ( s ) => s.id ), [ 'overview', 'hub', 'broker', 'outro' ] );
	assert.equal( stops[ 1 ].title, 'Node 3', 'a graph without labels names nodes by index' );
	assert.equal( stops[ 1 ].text, 'The most connected node: 6 links, 3.4× the average.' );
	assert.equal( stops[ 2 ].node, 2 );
	assert.equal( stops[ 2 ].text, 'The strongest broker: about 50 % of all shortest paths between other nodes run through it.', 'an estimate says "about"' );
	// one edge: singular forms, and a hub that is not far above the average
	const pair = planTour( plain( 2, [ [ 0, 1 ] ], [ 'A', 'B' ], 'Pair' ) );
	assert.equal( pair[ 0 ].title, 'Pair' );
	assert.equal( pair[ 0 ].text, '2 nodes and 1 link.' );
	assert.equal( pair[ 1 ].text, 'The most connected node, with 1 link.' );

} );

test( 'communities: the coarsest useful level, sizes, names and the share of internal edges', () => {

	const g = load( 'imsv/lesmis' );
	const level = usefulLevel( g );
	assert.equal( level, g.D - 1, 'the coarsest level splits the graph' );
	const list = communityList( g, level );
	assert.equal( list.length, 6 );
	assert.deepEqual( list.map( ( c ) => c.size ), [ 20, 17, 11, 11, 10, 8 ], 'largest first' );
	assert.equal( list.reduce( ( s, c ) => s + c.size, 0 ), g.N, 'every node is in exactly one' );
	assert.deepEqual( list.map( ( c ) => c.name ), [ 'Valjean', 'Gavroche', 'Marius', 'Javert', 'Fantine', 'Myriel' ] );
	// every edge is internal to one community or cut between two
	const internal = list.reduce( ( s, c ) => s + c.internal, 0 ), cut = list.reduce( ( s, c ) => s + c.cut, 0 );
	assert.equal( internal + cut / 2, g.E );
	for ( const c of list ) {

		assert.equal( c.level, level );
		assert.equal( communityOf( g, c.hub, level ), c.cluster, 'the hub is a member' );
		assert.equal( c.hubDegree, g.degree[ c.hub ] );
		assert.ok( Math.abs( c.share - c.internal / ( c.internal + c.cut ) ) < 1e-9 );
		assert.ok( c.rank >= 0 && c.rank <= 1 );
		for ( let i = 0; i < g.N; i ++ ) if ( communityOf( g, i, level ) === c.cluster ) assert.ok( g.degree[ i ] <= c.hubDegree );

	}

	// the finer level has more, smaller communities
	const fine = communityList( g, 0 );
	assert.equal( fine.length, 12 );
	assert.equal( fine.reduce( ( s, c ) => s + c.size, 0 ), g.N );
	assert.deepEqual( communityList( g, - 1 ), [] );
	assert.deepEqual( communityList( g, 7 ), [] );
	assert.equal( labelOf( g, g.labels.indexOf( 'Javert' ) ), 'Javert' );
	assert.equal( labelOf( { labels: null }, 12 ), 'Node 12' );
	assert.equal( labelOf( { labels: [ '', 'x' ] }, 0 ), 'Node 0' );

} );

test( 'usefulLevel: prefers a level a person can read', () => {

	const big = load( 'generated/lfr-2k' );
	const level = usefulLevel( big );
	assert.ok( level >= 0 && level < big.D );
	const n = communityList( big, level ).length;
	assert.ok( n >= 2, `${n} communities` );
	// every coarser level would have fewer than two communities
	for ( let l = level + 1; l < big.D; l ++ ) assert.ok( communityList( big, l ).length < 2 );
	assert.equal( usefulLevel( { D: 0 } ), - 1 );
	assert.equal( usefulLevel( null ), - 1 );

} );
