import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseEdgeList, looksLikeEdgeList } from '../src/data/edgelist.js';
import { rng } from '../src/graph/prepare.js';

const pairs = ( g ) => {

	const out = [];
	for ( let k = 0; k < g.edges.length; k += 2 ) out.push( [ g.labels[ g.edges[ k ] ], g.labels[ g.edges[ k + 1 ] ] ].sort().join( '-' ) );
	return out.sort();

};

test( 'whitespace-separated integers with comments; numeric node order', () => {

	const g = parseEdgeList( '# a comment\n% another\n// and another\n10 2\n2   3\n\n3\t10\n7 7\n2 10\n', { name: 'x' } );
	assert.equal( g.name, 'x' );
	assert.equal( g.nodeCount, 4 );
	assert.deepEqual( g.labels, [ '2', '3', '7', '10' ] ); // numeric order, not first appearance
	assert.deepEqual( pairs( g ), [ '10-2', '10-3', '2-3' ] ); // self loop and duplicate dropped
	assert.equal( g.hierarchy, null );
	assert.equal( g.meta.format, 'edgelist' );

} );

test( 'string ids keep first-appearance order; "a b" first line is data, not a header', () => {

	const g = parseEdgeList( 'a b\nb c\nc a\nd\n' );
	assert.deepEqual( g.labels, [ 'a', 'b', 'c', 'd' ] );
	assert.equal( g.edges.length / 2, 3 );
	assert.equal( g.nodeCount, 4 ); // "d" alone declares an isolated node

} );

test( 'CSV with header, names containing spaces, quoted fields, extra columns', () => {

	const csv = 'Source,Target,Weight\r\nJean Valjean,Fantine,3\r\n"Valjean, Jean",Cosette,1\r\nFantine , "Marius ""M"" Pontmercy",2\r\n';
	const g = parseEdgeList( csv );
	assert.deepEqual( g.meta.header, [ 'Source', 'Target', 'Weight' ] );
	assert.equal( g.meta.delimiter, ',' );
	assert.deepEqual( g.labels, [ 'Jean Valjean', 'Fantine', 'Valjean, Jean', 'Cosette', 'Marius "M" Pontmercy' ] );
	assert.equal( g.edges.length / 2, 3 );

} );

test( 'tab and semicolon delimiters; header detected from non-numeric first line', () => {

	const t = parseEdgeList( 'from\tto\n1\t2\n2\t3\n' );
	assert.equal( t.meta.delimiter, 'tab' );
	assert.equal( t.nodeCount, 3 );
	const s = parseEdgeList( 'node a;node b\n5;6\n6;7\n7;5\n' );
	assert.equal( s.meta.delimiter, ';' );
	assert.deepEqual( s.labels, [ '5', '6', '7' ] );
	assert.equal( s.edges.length / 2, 3 );

} );

test( 'Matrix Market coordinate file', () => {

	const mm = '%%MatrixMarket matrix coordinate pattern symmetric\n% comment\n4 4 3\n2 1\n3 1\n4 3\n';
	const g = parseEdgeList( mm );
	assert.ok( g.meta.matrixMarket );
	assert.deepEqual( g.labels, [ '1', '2', '3', '4' ] );
	assert.equal( g.edges.length / 2, 3 );

} );

test( 'BOM, trailing comments after ids, CRLF', () => {

	const g = parseEdgeList( '﻿x y # note\r\ny z\r\n' );
	assert.deepEqual( g.labels, [ 'x', 'y', 'z' ] );
	assert.equal( g.edges.length / 2, 2 );

} );

test( 'empty input throws; looksLikeEdgeList sniffing', () => {

	assert.throws( () => parseEdgeList( '# nothing\n\n' ), /no edges/ );
	assert.ok( looksLikeEdgeList( '1 2\n' ) );
	assert.ok( ! looksLikeEdgeList( '  {"nodes": []}' ) );
	assert.ok( ! looksLikeEdgeList( '\n[1,2]' ) );

} );

test( 'large edge list parses quickly', () => {

	const lines = [];
	const rand = rng( 5 );
	for ( let i = 0; i < 200000; i ++ ) lines.push( `n${i % 20000} n${Math.floor( rand() * 20000 )}` );
	const t0 = performance.now();
	const g = parseEdgeList( lines.join( '\n' ) );
	const ms = performance.now() - t0;
	assert.equal( g.nodeCount, 20000 );
	assert.ok( g.edges.length / 2 > 190000, `${g.edges.length / 2} edges` );
	assert.ok( ms < 3000, `${ms.toFixed( 0 )} ms` );

} );
