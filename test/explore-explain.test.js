// "Explain this node": the facts ( communities linked, what a removal cuts off, links among the
// neighbours ) and the sentences made from them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { nodeFacts, explainNode } from '../src/explore/explain.js';
import { usefulLevel, communityOf } from '../src/explore/communities.js';
import { parseGraphJson } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';

const load = ( name ) => prepareGraph( parseGraphJson( JSON.parse( readFileSync( `public/data/${name}.json`, 'utf8' ) ), name ) );

/** The parts of a prepared graph the module reads; `community` gives one level of communities. */
function graphOf( N, edges, community = null ) {

	const adj = Array.from( { length: N }, () => [] );
	for ( const [ a, b ] of edges ) {

		adj[ a ].push( b );
		adj[ b ].push( a );

	}

	const degree = Uint32Array.from( adj, ( l ) => l.length );
	const adjStart = new Uint32Array( N + 1 );
	for ( let i = 0; i < N; i ++ ) adjStart[ i + 1 ] = adjStart[ i ] + degree[ i ];
	const g = { N, E: edges.length, degree, adjStart, adjList: Uint32Array.from( adj.flat() ), D: 0 };
	if ( community ) Object.assign( g, { D: 1, anc: Uint32Array.from( community ) } );
	return g;

}

// a path 0–1–2–3–4–5 with a triangle 2–6–7 hanging off it, and a lone node 8
const EDGES = [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ], [ 4, 5 ], [ 2, 6 ], [ 6, 7 ], [ 7, 2 ] ];
const small = () => graphOf( 9, EDGES, [ 0, 0, 0, 1, 1, 1, 2, 2, 3 ] );

test( 'nodeFacts: what a removal cuts off', () => {

	const g = small();
	// without 2: { 0, 1 }, { 3, 4, 5 }, { 6, 7 } — the largest is the rest, two pieces with four nodes fall off
	assert.deepEqual( ( ( f ) => [ f.cutOff, f.pieces ] )( nodeFacts( g, 2, { level: 0 } ) ), [ 4, 2 ] );
	// without 1: { 0 } falls off
	assert.deepEqual( ( ( f ) => [ f.cutOff, f.pieces ] )( nodeFacts( g, 1, { level: 0 } ) ), [ 1, 1 ] );
	// 6 lies on a triangle, 0 is a leaf: nothing depends on them
	assert.deepEqual( ( ( f ) => [ f.cutOff, f.pieces ] )( nodeFacts( g, 6, { level: 0 } ) ), [ 0, 0 ] );
	assert.deepEqual( ( ( f ) => [ f.cutOff, f.pieces ] )( nodeFacts( g, 0, { level: 0 } ) ), [ 0, 0 ] );
	// over the budget the search is left out and says so
	assert.equal( nodeFacts( g, 2, { level: 0, budget: 4 } ).cutOff, - 1 );
	assert.ok( Number.isNaN( nodeFacts( g, 2, { level: 0, budget: 4 } ).clustering ) );

} );

test( 'nodeFacts: communities and links among the neighbours', () => {

	const g = small();
	const f = nodeFacts( g, 2, { level: 0 } );
	assert.equal( f.degree, 4 );
	assert.equal( f.communities, 3 ); // its own (1), that of 3 and that of 6 and 7
	assert.equal( f.own, 1 );
	assert.equal( f.clustering, 1 / 6 ); // one link ( 6–7 ) of six possible among 1, 3, 6, 7
	assert.equal( nodeFacts( g, 6, { level: 0 } ).clustering, 1 ); // 2 and 7 are linked
	assert.equal( nodeFacts( g, 4, { level: 0 } ).clustering, 0 );
	assert.ok( Number.isNaN( nodeFacts( g, 0, { level: 0 } ).clustering ) ); // one neighbour
	// no hierarchy: no statement about communities
	assert.equal( nodeFacts( graphOf( 9, EDGES ), 2 ).communities, 0 );
	// out of range
	assert.equal( nodeFacts( g, 99, { level: 0 } ).degree, 0 );

} );

test( 'explainNode: sentences', () => {

	const g = small();
	assert.deepEqual( explainNode( g, 2, { level: 0 } ).sentences, [
		'Connects 3 communities: 1 of its 4 neighbours are in its own, 3 in 2 others.',
		'Removing it would cut 4 nodes off from the rest, in 2 pieces.',
	] );
	assert.deepEqual( explainNode( g, 1, { level: 0 } ).sentences, [
		'All 2 neighbours are in its own community.',
		'Removing it would cut 1 node off from the rest.',
		'None of its neighbours are linked to each other.',
	] );
	assert.deepEqual( explainNode( g, 6, { level: 0 } ).sentences, [
		'Connects 2 communities: 1 of its 2 neighbours are in its own, 1 in another.',
		'Removing it would cut nothing off: its neighbours stay connected without it.',
		'Its neighbours are all linked to each other.',
	] );
	assert.deepEqual( explainNode( g, 3, { level: 0 } ).sentences[ 0 ], 'Connects 2 communities: 1 of its 2 neighbours are in its own, 1 in another.' );
	assert.deepEqual( explainNode( g, 0, { level: 0 } ).sentences, [ 'Its one neighbour is in its own community.' ] );
	assert.deepEqual( explainNode( g, 8, { level: 0 } ).sentences, [ 'No links: this node stands alone.' ] );
	assert.deepEqual( explainNode( g, - 1, { level: 0 } ).sentences, [] );
	// without communities the first sentence is left out
	assert.deepEqual( explainNode( graphOf( 9, EDGES ), 2 ).sentences, [ 'Removing it would cut 4 nodes off from the rest, in 2 pieces.' ] );

} );

test( 'explainNode: Les Misérables', () => {

	const g = load( 'imsv/lesmis' );
	const byLabel = ( name ) => g.labels.indexOf( name );
	const level = usefulLevel( g );
	// Myriel holds seven characters that appear with nobody else
	const myriel = explainNode( g, byLabel( 'Myriel' ) );
	assert.equal( myriel.cutOff, 7 );
	assert.equal( myriel.pieces, 7 );
	assert.ok( myriel.sentences.includes( 'Removing it would cut 7 nodes off from the rest, in 7 pieces.' ) );
	// Valjean: the counts agree with a direct count over his neighbours
	const v = byLabel( 'Valjean' ), f = explainNode( g, v );
	const nb = Array.from( g.adjList.subarray( g.adjStart[ v ], g.adjStart[ v ] + g.degree[ v ] ) );
	const mine = communityOf( g, v, level );
	assert.equal( f.degree, 36 );
	assert.equal( f.own, nb.filter( ( j ) => communityOf( g, j, level ) === mine ).length );
	assert.equal( f.communities, new Set( [ mine, ...nb.map( ( j ) => communityOf( g, j, level ) ) ] ).size );
	assert.ok( f.communities >= 3 );
	assert.ok( f.cutOff > 0 );
	assert.ok( f.sentences.length >= 2 && f.sentences.length <= 3 );
	assert.match( f.sentences[ 0 ], /^Connects \d+ communities: \d+ of its 36 neighbours are in its own, \d+ in \d+ others\.$/ );

} );
