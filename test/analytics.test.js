// Graph analytics (src/graph/analytics.js): brute-force references on small random graphs, closed
// forms on known graphs (path, star, cycle, complete, two cliques and a bridge, disconnected),
// Zachary's karate club through the data pipeline, invariants, determinism, tasks, and a
// performance smoke test at 20k nodes / 384k edges.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runRequest } from '../src/data/worker.js';
import { generate } from '../src/data/generators.js';
import { rng } from '../src/graph/prepare.js';
import {
	graphFromEdges, buildEdgeIndex, bfs, khopNeighborhood, shortestPath, shortestPathWeighted, shortestPathDag,
	connectedComponents, kCore, clusteringCoefficient, pageRank, eigenvectorCentrality,
	betweenness, edgeBetweenness, betweennessTask, closeness, closenessTask, averagePathLength, estimateDiameter,
	density, degreeHistogram, degreeAssortativity, graphSummary, normalize, quantiles,
	NODE_METRICS, nodeMetric, runTask, runTaskAsync, defaultSamples,
} from '../src/graph/analytics.js';

// ─── graphs ──────────────────────────────────────────────────────────────────────────────────

const fromPairs = ( n, pairs ) => graphFromEdges( n, pairs.flat() );
const path = ( n ) => fromPairs( n, Array.from( { length: n - 1 }, ( _, i ) => [ i, i + 1 ] ) );
const star = ( n ) => fromPairs( n, Array.from( { length: n - 1 }, ( _, i ) => [ 0, i + 1 ] ) );
const cycle = ( n ) => fromPairs( n, Array.from( { length: n }, ( _, i ) => [ i, ( i + 1 ) % n ] ) );

function cliquePairs( first, size ) {

	const pairs = [];
	for ( let i = 0; i < size; i ++ ) for ( let j = i + 1; j < size; j ++ ) pairs.push( [ first + i, first + j ] );
	return pairs;

}

const complete = ( n ) => fromPairs( n, cliquePairs( 0, n ) );
// cliques on 0..a-1 and a..a+b-1, joined by the bridge ( a - 1, a )
const barbell = ( a, b ) => fromPairs( a + b, [ ...cliquePairs( 0, a ), ...cliquePairs( a, b ), [ a - 1, a ] ] );

/** G( n, p ), optionally with node labels shuffled so that components interleave. */
function randomGraph( n, p, seed ) {

	const rand = rng( seed );
	const pairs = [];
	for ( let i = 0; i < n; i ++ ) for ( let j = i + 1; j < n; j ++ ) if ( rand() < p ) pairs.push( [ i, j ] );
	return fromPairs( n, pairs );

}

/** Several random components of different kinds plus isolated nodes, labels shuffled. */
function messyGraph( seed ) {

	const rand = rng( seed );
	const pairs = [];
	let n = 0;
	const add = ( size, p ) => {

		// a random spanning tree keeps the part connected; extra edges with probability p
		for ( let i = 1; i < size; i ++ ) pairs.push( [ n + Math.floor( rand() * i ), n + i ] );
		for ( let i = 0; i < size; i ++ ) for ( let j = i + 1; j < size; j ++ ) if ( rand() < p ) pairs.push( [ n + i, n + j ] );
		n += size;

	};

	add( 14 + Math.floor( rand() * 8 ), 0.12 );
	add( 6 + Math.floor( rand() * 5 ), 0.3 );
	add( 3, 0 );
	add( 2, 0 );
	n += 2; // isolated nodes
	const label = Array.from( { length: n }, ( _, i ) => i );
	for ( let i = n - 1; i > 0; i -- ) {

		const j = Math.floor( rand() * ( i + 1 ) );
		[ label[ i ], label[ j ] ] = [ label[ j ], label[ i ] ];

	}

	const seen = new Set(), clean = [];
	for ( const [ a, b ] of pairs ) {

		const x = Math.min( label[ a ], label[ b ] ), y = Math.max( label[ a ], label[ b ] );
		if ( seen.has( x * n + y ) ) continue;
		seen.add( x * n + y );
		clean.push( [ x, y ] );

	}

	return fromPairs( n, clean );

}

const SMALL = [
	[ 'G(12, .3)', randomGraph( 12, 0.3, 1 ) ],
	[ 'G(25, .15)', randomGraph( 25, 0.15, 2 ) ],
	[ 'G(40, .08)', randomGraph( 40, 0.08, 3 ) ], // sparse: several components and isolated nodes
	[ 'G(30, .5)', randomGraph( 30, 0.5, 4 ) ],
	[ 'G(9, 0)', randomGraph( 9, 0, 5 ) ], // no edges
	[ 'messy 6', messyGraph( 6 ) ],
	[ 'messy 7', messyGraph( 7 ) ],
	[ 'path 9', path( 9 ) ],
	[ 'star 8', star( 8 ) ],
	[ 'cycle 10', cycle( 10 ) ],
	[ 'K7', complete( 7 ) ],
	[ 'barbell 5+4', barbell( 5, 4 ) ],
	[ 'grid 5x4', fromPairs( 20, [ ...Array.from( { length: 20 }, ( _, i ) => ( i % 5 < 4 ? [ [ i, i + 1 ] ] : [] ) ).flat(), ...Array.from( { length: 15 }, ( _, i ) => [ i, i + 5 ] ) ] ) ],
];

// ─── brute-force references (independent of the module) ──────────────────────────────────────

const neighbours = ( g, v ) => Array.from( g.adjList.subarray( g.adjStart[ v ], g.adjStart[ v ] + g.degree[ v ] ) );

/** dist[ s ][ v ] (Infinity when unreachable) and sigma[ s ][ v ] (number of shortest paths). */
function allPairs( g ) {

	const dist = [], sigma = [];
	for ( let s = 0; s < g.N; s ++ ) {

		const d = new Array( g.N ).fill( Infinity ), c = new Array( g.N ).fill( 0 );
		d[ s ] = 0; c[ s ] = 1;
		const queue = [ s ];
		while ( queue.length ) {

			const v = queue.shift();
			for ( const w of neighbours( g, v ) ) {

				if ( d[ w ] === Infinity ) {

					d[ w ] = d[ v ] + 1;
					queue.push( w );

				}

				if ( d[ w ] === d[ v ] + 1 ) c[ w ] += c[ v ];

			}

		}

		dist.push( d );
		sigma.push( c );

	}

	return { dist, sigma };

}

/** Betweenness from its definition: O( n^3 ) over pairs for nodes, O( n^2 E ) for edges. */
function bruteBetweenness( g, { dist, sigma } ) {

	const node = new Array( g.N ).fill( 0 ), edge = new Array( g.E ).fill( 0 );
	for ( let s = 0; s < g.N; s ++ ) for ( let t = s + 1; t < g.N; t ++ ) {

		if ( dist[ s ][ t ] === Infinity ) continue;
		for ( let v = 0; v < g.N; v ++ ) {

			if ( v !== s && v !== t && dist[ s ][ v ] + dist[ v ][ t ] === dist[ s ][ t ] ) node[ v ] += sigma[ s ][ v ] * sigma[ v ][ t ] / sigma[ s ][ t ];

		}

		for ( let e = 0; e < g.E; e ++ ) {

			const a = g.edges[ 2 * e ], b = g.edges[ 2 * e + 1 ];
			if ( dist[ s ][ a ] + 1 + dist[ b ][ t ] === dist[ s ][ t ] ) edge[ e ] += sigma[ s ][ a ] * sigma[ b ][ t ] / sigma[ s ][ t ];
			if ( dist[ s ][ b ] + 1 + dist[ a ][ t ] === dist[ s ][ t ] ) edge[ e ] += sigma[ s ][ b ] * sigma[ a ][ t ] / sigma[ s ][ t ];

		}

	}

	return { node, edge };

}

function bruteComponents( { dist }, n ) {

	const root = Array.from( { length: n }, ( _, v ) => dist[ v ].findIndex( ( d ) => d < Infinity ) ); // smallest reachable index
	const members = new Map();
	root.forEach( ( r, v ) => members.set( r, ( members.get( r ) || 0 ) + 1 ) );
	const order = [ ...members.keys() ].sort( ( a, b ) => members.get( b ) - members.get( a ) || a - b );
	return { component: root.map( ( r ) => order.indexOf( r ) ), sizes: order.map( ( r ) => members.get( r ) ) };

}

function bruteCore( g ) {

	const core = new Array( g.N ).fill( 0 );
	for ( let k = 1; ; k ++ ) {

		const alive = core.map( ( c ) => c === k - 1 );
		for ( let changed = true; changed; ) {

			changed = false;
			for ( let v = 0; v < g.N; v ++ ) {

				if ( alive[ v ] && neighbours( g, v ).filter( ( w ) => alive[ w ] ).length < k ) {

					alive[ v ] = false;
					changed = true;

				}

			}

		}

		if ( ! alive.some( Boolean ) ) return core;
		alive.forEach( ( a, v ) => {

			if ( a ) core[ v ] = k;

		} );

	}

}

function adjacencyMatrix( g ) {

	const m = Array.from( { length: g.N }, () => new Array( g.N ).fill( 0 ) );
	for ( let e = 0; e < g.E; e ++ ) m[ g.edges[ 2 * e ] ][ g.edges[ 2 * e + 1 ] ] = m[ g.edges[ 2 * e + 1 ] ][ g.edges[ 2 * e ] ] = 1;
	return m;

}

function bruteTriangles( g ) {

	const m = adjacencyMatrix( g );
	const node = new Array( g.N ).fill( 0 ), edge = new Array( g.E ).fill( 0 );
	let total = 0;
	for ( let a = 0; a < g.N; a ++ ) for ( let b = a + 1; b < g.N; b ++ ) for ( let c = b + 1; c < g.N; c ++ ) {

		if ( m[ a ][ b ] && m[ b ][ c ] && m[ a ][ c ] ) {

			node[ a ] ++; node[ b ] ++; node[ c ] ++; total ++;

		}

	}

	for ( let e = 0; e < g.E; e ++ ) for ( let c = 0; c < g.N; c ++ ) if ( m[ g.edges[ 2 * e ] ][ c ] && m[ g.edges[ 2 * e + 1 ] ][ c ] ) edge[ e ] ++;
	return { node, edge, total };

}

/** Solves A x = b by Gaussian elimination with partial pivoting. */
function solve( A, b ) {

	const n = b.length, M = A.map( ( row, i ) => [ ...row, b[ i ] ] );
	for ( let c = 0; c < n; c ++ ) {

		let p = c;
		for ( let r = c + 1; r < n; r ++ ) if ( Math.abs( M[ r ][ c ] ) > Math.abs( M[ p ][ c ] ) ) p = r;
		[ M[ c ], M[ p ] ] = [ M[ p ], M[ c ] ];
		for ( let r = c + 1; r < n; r ++ ) {

			const f = M[ r ][ c ] / M[ c ][ c ];
			for ( let k = c; k <= n; k ++ ) M[ r ][ k ] -= f * M[ c ][ k ];

		}

	}

	const x = new Array( n ).fill( 0 );
	for ( let r = n - 1; r >= 0; r -- ) {

		let s = M[ r ][ n ];
		for ( let k = r + 1; k < n; k ++ ) s -= M[ r ][ k ] * x[ k ];
		x[ r ] = s / M[ r ][ r ];

	}

	return x;

}

/** PageRank as the solution of its linear system: p = ( 1 - d ) t + d ( W p + dangling mass * t ). */
function brutePageRank( g, damping, teleport ) {

	const n = g.N;
	const t = teleport || new Array( n ).fill( 1 / n );
	const A = Array.from( { length: n }, ( _, v ) => Array.from( { length: n }, ( _, u ) => ( u === v ? 1 : 0 ) - damping * ( g.degree[ u ] === 0 ? t[ v ] : 0 ) ) );
	for ( let u = 0; u < n; u ++ ) for ( const v of neighbours( g, u ) ) A[ v ][ u ] -= damping / g.degree[ u ];
	return solve( A, t.map( ( x ) => ( 1 - damping ) * x ) );

}

/** Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations: { values, vectors (columns) }. */
function jacobi( S ) {

	const n = S.length, a = S.map( ( r ) => r.slice() );
	const v = Array.from( { length: n }, ( _, i ) => Array.from( { length: n }, ( _, j ) => ( i === j ? 1 : 0 ) ) );
	for ( let sweep = 0; sweep < 100; sweep ++ ) {

		let off = 0;
		for ( let p = 0; p < n; p ++ ) for ( let q = p + 1; q < n; q ++ ) off += a[ p ][ q ] * a[ p ][ q ];
		if ( off < 1e-22 ) break;
		for ( let p = 0; p < n; p ++ ) for ( let q = p + 1; q < n; q ++ ) {

			if ( Math.abs( a[ p ][ q ] ) < 1e-300 ) continue;
			const theta = ( a[ q ][ q ] - a[ p ][ p ] ) / ( 2 * a[ p ][ q ] );
			const t = Math.sign( theta || 1 ) / ( Math.abs( theta ) + Math.sqrt( theta * theta + 1 ) );
			const c = 1 / Math.sqrt( t * t + 1 ), s = t * c;
			for ( let k = 0; k < n; k ++ ) {

				const akp = a[ k ][ p ], akq = a[ k ][ q ];
				a[ k ][ p ] = c * akp - s * akq; a[ k ][ q ] = s * akp + c * akq;

			}

			for ( let k = 0; k < n; k ++ ) {

				const apk = a[ p ][ k ], aqk = a[ q ][ k ];
				a[ p ][ k ] = c * apk - s * aqk; a[ q ][ k ] = s * apk + c * aqk;

			}

			for ( let k = 0; k < n; k ++ ) {

				const vkp = v[ k ][ p ], vkq = v[ k ][ q ];
				v[ k ][ p ] = c * vkp - s * vkq; v[ k ][ q ] = s * vkp + c * vkq;

			}

		}

	}

	return { values: a.map( ( r, i ) => r[ i ] ), vectors: v };

}

/** Principal eigenvector of every component (unit root-mean-square), all divided by sqrt( active nodes ). */
function bruteEigenvector( g, comp ) {

	const out = new Array( g.N ).fill( 0 );
	const active = Array.from( g.degree ).filter( ( d ) => d > 0 ).length;
	const m = adjacencyMatrix( g );
	comp.sizes.forEach( ( size, c ) => {

		if ( size < 2 ) return;
		const nodes = Array.from( comp.component, ( x, v ) => ( x === c ? v : - 1 ) ).filter( ( v ) => v >= 0 );
		const { values, vectors } = jacobi( nodes.map( ( a ) => nodes.map( ( b ) => m[ a ][ b ] ) ) );
		const top = values.indexOf( Math.max( ...values ) );
		const vec = vectors.map( ( r ) => r[ top ] );
		const sign = vec.reduce( ( s, x ) => s + x, 0 ) < 0 ? - 1 : 1;
		nodes.forEach( ( v, i ) => {

			out[ v ] = sign * vec[ i ] * Math.sqrt( size ) / Math.sqrt( active );

		} );

	} );
	return out;

}

function pearson( x, y ) {

	const n = x.length, mx = x.reduce( ( s, v ) => s + v, 0 ) / n, my = y.reduce( ( s, v ) => s + v, 0 ) / n;
	let sxy = 0, sxx = 0, syy = 0;
	for ( let i = 0; i < n; i ++ ) {

		sxy += ( x[ i ] - mx ) * ( y[ i ] - my ); sxx += ( x[ i ] - mx ) ** 2; syy += ( y[ i ] - my ) ** 2;

	}

	return sxy / Math.sqrt( sxx * syy );

}

function assertClose( actual, expected, tolerance, message ) {

	assert.equal( actual.length, expected.length, `${message}: length` );
	for ( let i = 0; i < expected.length; i ++ ) {

		const limit = tolerance * Math.max( 1, Math.abs( expected[ i ] ) );
		assert.ok( Math.abs( actual[ i ] - expected[ i ] ) <= limit, `${message}: [${i}] ${actual[ i ]} vs ${expected[ i ]}` );

	}

}

const near = ( actual, expected, tolerance, message ) => assert.ok( Math.abs( actual - expected ) <= tolerance * Math.max( 1, Math.abs( expected ) ), `${message || 'value'}: ${actual} vs ${expected}` );

function assertPath( g, r, s, t, hops ) {

	assert.equal( r.nodes[ 0 ], s );
	assert.equal( r.nodes[ r.nodes.length - 1 ], t );
	assert.equal( r.edges.length, r.nodes.length - 1 );
	if ( hops !== undefined ) assert.equal( r.edges.length, hops );
	for ( let i = 0; i < r.edges.length; i ++ ) {

		const a = g.edges[ 2 * r.edges[ i ] ], b = g.edges[ 2 * r.edges[ i ] + 1 ];
		assert.deepEqual( [ a, b ].sort(), [ r.nodes[ i ], r.nodes[ i + 1 ] ].sort(), `edge ${i} joins consecutive path nodes` );

	}

	assert.equal( new Set( r.nodes ).size, r.nodes.length, 'simple path' );

}

// ─── construction ────────────────────────────────────────────────────────────────────────────

test( 'graphFromEdges matches the PreparedGraph CSR; buildEdgeIndex handles any adjacency order', async () => {

	const r = await runRequest( { source: { kind: 'text', text: readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ), name: 'lesmis.json' } } );
	const p = r.graph;
	const g = graphFromEdges( p.N, p.edges );
	assert.equal( g.E, p.E );
	assert.deepEqual( g.degree, p.degree );
	assert.deepEqual( g.adjStart, p.adjStart );
	assert.deepEqual( g.adjList, p.adjList );
	assert.deepEqual( buildEdgeIndex( p ), g.adjEdge );
	for ( let v = 0; v < g.N; v ++ ) for ( let k = g.adjStart[ v ]; k < g.adjStart[ v ] + g.degree[ v ]; k ++ ) {

		const e = g.adjEdge[ k ];
		assert.deepEqual( [ g.edges[ 2 * e ], g.edges[ 2 * e + 1 ] ].sort( ( a, b ) => a - b ), [ v, g.adjList[ k ] ].sort( ( a, b ) => a - b ) );

	}

	// sorted neighbour lists (not the edge order): the generic path must give a consistent index
	const sorted = { ...g, adjList: g.adjList.slice(), adjEdge: undefined };
	for ( let v = 0; v < g.N; v ++ ) sorted.adjList.subarray( g.adjStart[ v ], g.adjStart[ v ] + g.degree[ v ] ).sort().reverse();
	const index = buildEdgeIndex( sorted );
	for ( let v = 0; v < g.N; v ++ ) for ( let k = g.adjStart[ v ]; k < g.adjStart[ v ] + g.degree[ v ]; k ++ ) {

		const e = index[ k ];
		assert.deepEqual( [ g.edges[ 2 * e ], g.edges[ 2 * e + 1 ] ].sort( ( a, b ) => a - b ), [ v, sorted.adjList[ k ] ].sort( ( a, b ) => a - b ) );

	}

	assert.deepEqual( shortestPath( sorted, 0, 50 ).length, shortestPath( g, 0, 50 ).length );
	// adjacency that is not the edge list's
	const wrong = { ...g, adjList: g.adjList.slice(), adjEdge: undefined };
	wrong.adjList[ 0 ] = wrong.adjList[ 0 ] === 5 ? 6 : 5;
	wrong.adjList[ 1 ] = 76;
	assert.throws( () => buildEdgeIndex( wrong ), /not in the edge list/ );
	// empty graphs
	const empty = graphFromEdges( 3, [] );
	assert.equal( empty.E, 0 );
	assert.equal( empty.adjList.length, 1 );
	assert.deepEqual( Array.from( empty.degree ), [ 0, 0, 0 ] );

} );

// ─── traversal ───────────────────────────────────────────────────────────────────────────────

test( 'bfs: distances, tree, order, levels, maxDepth, multi-source, scratch reuse', () => {

	for ( const [ name, g ] of SMALL ) {

		const { dist } = allPairs( g );
		let out;
		for ( let s = 0; s < g.N; s ++ ) {

			const r = bfs( g, s, { out } );
			out = r;
			assert.deepEqual( Array.from( r.dist ), dist[ s ].map( ( d ) => ( d === Infinity ? - 1 : d ) ), `${name} from ${s}` );
			assert.equal( r.reached, dist[ s ].filter( ( d ) => d < Infinity ).length );
			assert.equal( r.maxDist, Math.max( ...dist[ s ].filter( ( d ) => d < Infinity ) ) );
			assert.equal( r.order.length, r.reached );
			assert.equal( new Set( r.order ).size, r.reached );
			assert.equal( r.levelOffsets.length, r.maxDist + 2 );
			assert.equal( r.levelOffsets[ r.maxDist + 1 ], r.reached );
			for ( let d = 0; d <= r.maxDist; d ++ ) {

				assert.ok( r.levelOffsets[ d + 1 ] > r.levelOffsets[ d ], 'no empty level' );
				for ( let i = r.levelOffsets[ d ]; i < r.levelOffsets[ d + 1 ]; i ++ ) assert.equal( r.dist[ r.order[ i ] ], d );

			}

			for ( let v = 0; v < g.N; v ++ ) {

				if ( r.dist[ v ] > 0 ) {

					assert.equal( r.dist[ r.parent[ v ] ], r.dist[ v ] - 1 );
					assert.ok( neighbours( g, v ).includes( r.parent[ v ] ) );

				} else assert.equal( r.parent[ v ], - 1 );

			}

		}

		if ( g.N < 4 ) continue;
		const limited = bfs( g, 0, { maxDepth: 2 } );
		assert.deepEqual( Array.from( limited.dist ), dist[ 0 ].map( ( d ) => ( d <= 2 ? d : - 1 ) ), `${name} maxDepth` );
		assert.ok( limited.maxDist <= 2 );
		assert.deepEqual( Array.from( bfs( g, 0, { maxDepth: 0 } ).order ), [ 0 ] );
		const multi = bfs( g, [ 0, 3, 3, g.N - 1 ] );
		assert.deepEqual( Array.from( multi.dist ), dist[ 0 ].map( ( d, v ) => {

			const m = Math.min( d, dist[ 3 ][ v ], dist[ g.N - 1 ][ v ] );
			return m === Infinity ? - 1 : m;

		} ), `${name} multi-source` );
		assert.deepEqual( Array.from( multi.order.subarray( 0, 3 ) ), [ 0, 3, g.N - 1 ] );
		assert.deepEqual( Array.from( bfs( g, new Uint32Array( [ 0 ] ) ).dist ), Array.from( bfs( g, 0 ).dist ) );

	}

	// reuse really reuses, and a result for another size is ignored
	const g = SMALL[ 1 ][ 1 ];
	const first = bfs( g, 0 );
	const again = bfs( g, 5, { out: first } );
	assert.equal( again.dist, first.dist );
	assert.equal( again.order.buffer, first.order.buffer );
	assert.notEqual( bfs( SMALL[ 0 ][ 1 ], 0, { out: first } ).dist, first.dist );
	assert.throws( () => bfs( g, g.N ), RangeError );
	assert.throws( () => bfs( g, [ 0, - 1 ] ), RangeError );

} );

test( 'khopNeighborhood: nodes within k hops with hop counts', () => {

	for ( const [ name, g ] of SMALL ) {

		const { dist } = allPairs( g );
		for ( const k of [ 0, 1, 2, 3 ] ) {

			for ( let s = 0; s < g.N; s += 3 ) {

				const r = khopNeighborhood( g, s, k );
				const expected = dist[ s ].map( ( d, v ) => [ d, v ] ).filter( ( [ d ] ) => d <= k );
				assert.deepEqual( Array.from( r.nodes ).sort( ( a, b ) => a - b ), expected.map( ( [ , v ] ) => v ), `${name} s=${s} k=${k}` );
				assert.equal( r.nodes[ 0 ], s );
				r.nodes.forEach( ( v, i ) => {

					assert.equal( r.hops[ i ], dist[ s ][ v ] );
					assert.equal( r.dist[ v ], dist[ s ][ v ] );

				} );
				assert.ok( r.hops.every( ( h, i ) => i === 0 || h >= r.hops[ i - 1 ] ), 'nearest first' );
				assert.equal( r.maxHop, Math.max( ...r.hops ) );
				assert.equal( r.levelOffsets[ r.maxHop + 1 ], r.nodes.length );

			}

		}

	}

	const s = star( 6 );
	assert.deepEqual( Array.from( khopNeighborhood( s, [ 1, 2 ], 1 ).nodes ), [ 1, 2, 0 ] );

} );

// ─── shortest paths ──────────────────────────────────────────────────────────────────────────

test( 'shortestPath (bidirectional BFS): optimal length, valid nodes and edge ids, every pair', () => {

	for ( const [ name, g ] of SMALL ) {

		const { dist } = allPairs( g );
		for ( let s = 0; s < g.N; s ++ ) for ( let t = 0; t < g.N; t ++ ) {

			const r = shortestPath( g, s, t );
			if ( dist[ s ][ t ] === Infinity ) {

				assert.equal( r, null, `${name} ${s}->${t}` );
				continue;

			}

			assert.equal( r.length, dist[ s ][ t ], `${name} ${s}->${t}` );
			assertPath( g, r, s, t, dist[ s ][ t ] );

		}

	}

	const g = path( 5 );
	assert.deepEqual( shortestPath( g, 2, 2 ), { nodes: Uint32Array.of( 2 ), edges: new Uint32Array( 0 ), length: 0 } );
	assert.deepEqual( Array.from( shortestPath( g, 4, 0 ).nodes ), [ 4, 3, 2, 1, 0 ] );
	assert.deepEqual( Array.from( shortestPath( g, 4, 0 ).edges ), [ 3, 2, 1, 0 ] );
	assert.throws( () => shortestPath( g, 0, 5 ), RangeError );
	assert.throws( () => shortestPath( g, 0.5, 1 ), RangeError );
	// deterministic, and independent of earlier queries (stamped scratch)
	const big = randomGraph( 60, 0.07, 11 );
	const a = shortestPath( big, 3, 41 );
	for ( let i = 0; i < 50; i ++ ) shortestPath( big, i, 59 - i );
	assert.deepEqual( shortestPath( big, 3, 41 ), a );

} );

test( 'shortestPathWeighted: A* and Dijkstra agree with Floyd-Warshall on Euclidean lengths', () => {

	for ( const [ name, g ] of SMALL.slice( 0, 7 ) ) {

		const rand = rng( 77 );
		const stride = 4;
		const pos = new Float32Array( stride * g.N );
		for ( let i = 0; i < pos.length; i ++ ) pos[ i ] = rand() * 100 - 50;
		const length = ( a, b ) => Math.hypot( pos[ 4 * a ] - pos[ 4 * b ], pos[ 4 * a + 1 ] - pos[ 4 * b + 1 ], pos[ 4 * a + 2 ] - pos[ 4 * b + 2 ] );
		const d = Array.from( { length: g.N }, ( _, i ) => Array.from( { length: g.N }, ( _, j ) => ( i === j ? 0 : Infinity ) ) );
		for ( let e = 0; e < g.E; e ++ ) {

			const a = g.edges[ 2 * e ], b = g.edges[ 2 * e + 1 ];
			d[ a ][ b ] = d[ b ][ a ] = length( a, b );

		}

		for ( let k = 0; k < g.N; k ++ ) for ( let i = 0; i < g.N; i ++ ) for ( let j = 0; j < g.N; j ++ ) if ( d[ i ][ k ] + d[ k ][ j ] < d[ i ][ j ] ) d[ i ][ j ] = d[ i ][ k ] + d[ k ][ j ];
		for ( let s = 0; s < g.N; s ++ ) for ( let t = 0; t < g.N; t ++ ) {

			const astar = shortestPathWeighted( g, s, t, pos );
			const dijkstra = shortestPathWeighted( g, s, t, pos, { heuristic: false } );
			if ( d[ s ][ t ] === Infinity ) {

				assert.equal( astar, null, `${name} ${s}->${t}` );
				assert.equal( dijkstra, null );
				continue;

			}

			near( astar.length, d[ s ][ t ], 1e-9, `${name} A* ${s}->${t}` );
			near( dijkstra.length, d[ s ][ t ], 1e-9, `${name} Dijkstra ${s}->${t}` );
			assertPath( g, astar, s, t );
			assertPath( g, dijkstra, s, t );
			let sum = 0;
			for ( let i = 0; i + 1 < astar.nodes.length; i ++ ) sum += length( astar.nodes[ i ], astar.nodes[ i + 1 ] );
			near( sum, astar.length, 1e-9, 'length is the sum of the edge lengths' );

		}

	}

	// the weighted path may take more hops than the fewest-hops path; stride 3 positions work too
	const g = fromPairs( 4, [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 0, 3 ] ] );
	const pos3 = Float32Array.from( [ 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 100, 0 ] );
	assert.deepEqual( Array.from( shortestPathWeighted( g, 0, 2, pos3, { stride: 3 } ).nodes ), [ 0, 1, 2 ] );
	const far = Float32Array.from( [ 0, 0, 0, 50, 80, 0, 1, 1, 0, 1, 0, 0 ] );
	assert.deepEqual( Array.from( shortestPathWeighted( g, 0, 2, far, { stride: 3 } ).nodes ), [ 0, 3, 2 ] );
	assert.equal( shortestPath( g, 0, 2 ).length, 2 );
	// NaN positions: the edges of that node are unusable, everything else still works
	const broken = Float32Array.from( [ 0, 0, 0, NaN, 0, 0, 2, 0, 0, 3, 100, 0 ] );
	assert.deepEqual( Array.from( shortestPathWeighted( g, 0, 2, broken, { stride: 3 } ).nodes ), [ 0, 3, 2 ] );
	assert.equal( shortestPathWeighted( g, 0, 1, broken, { stride: 3 } ), null );
	assert.equal( shortestPathWeighted( g, 2, 2, pos3, { stride: 3 } ).length, 0 );

} );

test( 'shortestPathDag: every geodesic node and edge with its path count', () => {

	for ( const [ name, g ] of SMALL ) {

		const { dist, sigma } = allPairs( g );
		for ( let s = 0; s < g.N; s ++ ) for ( let t = 0; t < g.N; t ++ ) {

			const r = shortestPathDag( g, s, t );
			if ( dist[ s ][ t ] === Infinity ) {

				assert.equal( r, null );
				continue;

			}

			const D = dist[ s ][ t ];
			assert.equal( r.length, D, `${name} ${s}->${t}` );
			assert.equal( r.pathCount, sigma[ s ][ t ] );
			const nodes = [];
			for ( let v = 0; v < g.N; v ++ ) if ( dist[ s ][ v ] + dist[ v ][ t ] === D ) nodes.push( v );
			assert.deepEqual( Array.from( r.nodes ).sort( ( a, b ) => a - b ), nodes );
			assert.equal( r.nodes[ 0 ], s );
			assert.equal( r.nodes[ r.nodes.length - 1 ], t );
			r.nodes.forEach( ( v, i ) => {

				assert.equal( r.depth[ i ], dist[ s ][ v ] );
				assert.equal( r.nodePaths[ i ], sigma[ s ][ v ] * sigma[ v ][ t ] );
				assert.ok( i === 0 || r.depth[ i ] >= r.depth[ i - 1 ] );

			} );
			const edges = new Map();
			for ( let e = 0; e < g.E; e ++ ) {

				const a = g.edges[ 2 * e ], b = g.edges[ 2 * e + 1 ];
				if ( dist[ s ][ a ] + 1 + dist[ b ][ t ] === D ) edges.set( e, sigma[ s ][ a ] * sigma[ b ][ t ] );
				if ( dist[ s ][ b ] + 1 + dist[ a ][ t ] === D ) edges.set( e, sigma[ s ][ b ] * sigma[ a ][ t ] );

			}

			assert.deepEqual( new Map( Array.from( r.edges, ( e, i ) => [ e, r.edgePaths[ i ] ] ) ), edges );
			assert.equal( r.edges.length, edges.size );
			// every level is a cut of the DAG: its node counts sum to the number of geodesics
			for ( let d = 0; d <= D; d ++ ) assert.equal( r.nodePaths.reduce( ( sum, c, i ) => sum + ( r.depth[ i ] === d ? c : 0 ), 0 ), r.pathCount );

		}

	}

	// 4 x 4 grid, corner to corner: C( 6, 3 ) = 20 geodesics through all 16 nodes and 24 edges
	const pairs = [];
	for ( let y = 0; y < 4; y ++ ) for ( let x = 0; x < 4; x ++ ) {

		if ( x < 3 ) pairs.push( [ y * 4 + x, y * 4 + x + 1 ] );
		if ( y < 3 ) pairs.push( [ y * 4 + x, y * 4 + x + 4 ] );

	}

	const r = shortestPathDag( fromPairs( 16, pairs ), 0, 15 );
	assert.equal( r.pathCount, 20 );
	assert.equal( r.nodes.length, 16 );
	assert.equal( r.edges.length, 24 );

} );

// ─── structure ───────────────────────────────────────────────────────────────────────────────

test( 'connectedComponents: labels sorted by size, sizes, count', () => {

	for ( const [ name, g ] of SMALL ) {

		const ref = bruteComponents( allPairs( g ), g.N );
		const r = connectedComponents( g );
		assert.deepEqual( Array.from( r.component ), ref.component, name );
		assert.deepEqual( Array.from( r.sizes ), ref.sizes, name );
		assert.equal( r.count, ref.sizes.length );
		assert.ok( r.sizes.every( ( s, i ) => i === 0 || s <= r.sizes[ i - 1 ] ) );

	}

	const two = fromPairs( 7, [ [ 0, 1 ], [ 2, 3 ], [ 3, 4 ], [ 4, 2 ] ] ); // sizes 3, 2, 1, 1
	assert.deepEqual( Array.from( connectedComponents( two ).component ), [ 1, 1, 0, 0, 0, 2, 3 ] );
	assert.deepEqual( Array.from( connectedComponents( two ).sizes ), [ 3, 2, 1, 1 ] );
	assert.equal( connectedComponents( graphFromEdges( 0, [] ) ).count, 0 );

} );

test( 'kCore: Batagelj-Zaversnik equals brute-force peeling; degeneracy order', () => {

	for ( const [ name, g ] of SMALL ) {

		const r = kCore( g );
		assert.deepEqual( Array.from( r.core ), bruteCore( g ), name );
		assert.equal( r.degeneracy, Math.max( 0, ...r.core ) );
		// peeling order: a permutation in which every node has at most `degeneracy` later neighbours
		assert.deepEqual( Array.from( r.order ).sort( ( a, b ) => a - b ), Array.from( { length: g.N }, ( _, i ) => i ) );
		const position = new Array( g.N );
		r.order.forEach( ( v, i ) => position[ v ] = i );
		for ( let v = 0; v < g.N; v ++ ) {

			const later = neighbours( g, v ).filter( ( w ) => position[ w ] > position[ v ] ).length;
			assert.ok( later <= r.core[ v ], `${name}: node ${v} has ${later} later neighbours, core ${r.core[ v ]}` );

		}

		assert.ok( Array.from( r.order ).every( ( v, i ) => i === 0 || r.core[ v ] >= r.core[ r.order[ i - 1 ] ] ), 'cores never decrease along the order' );

	}

	assert.deepEqual( Array.from( kCore( path( 6 ) ).core ), [ 1, 1, 1, 1, 1, 1 ] );
	assert.deepEqual( Array.from( kCore( star( 6 ) ).core ), [ 1, 1, 1, 1, 1, 1 ] );
	assert.deepEqual( Array.from( kCore( cycle( 6 ) ).core ), [ 2, 2, 2, 2, 2, 2 ] );
	assert.equal( kCore( complete( 6 ) ).degeneracy, 5 );
	assert.deepEqual( Array.from( kCore( barbell( 5, 3 ) ).core ), [ 4, 4, 4, 4, 4, 2, 2, 2 ] );
	// a clique with a pendant path: the path stays in the 1-core
	assert.deepEqual( Array.from( kCore( fromPairs( 6, [ ...cliquePairs( 0, 4 ), [ 3, 4 ], [ 4, 5 ] ] ) ).core ), [ 3, 3, 3, 3, 1, 1 ] );

} );

test( 'clusteringCoefficient: triangles per node and per edge, local, global, average', () => {

	for ( const [ name, g ] of SMALL ) {

		const ref = bruteTriangles( g );
		const r = clusteringCoefficient( g, { edges: true } );
		assert.deepEqual( Array.from( r.triangles ), ref.node, name );
		assert.deepEqual( Array.from( r.edgeTriangles ), ref.edge, name );
		assert.equal( r.edgeTriangles.length, g.E );
		assert.equal( r.triangleCount, ref.total );
		const local = Array.from( g.degree, ( d, v ) => ( d < 2 ? 0 : ref.node[ v ] / ( d * ( d - 1 ) / 2 ) ) );
		assertClose( r.local, local, 1e-6, name );
		const triples = Array.from( g.degree ).reduce( ( s, d ) => s + d * ( d - 1 ) / 2, 0 );
		near( r.global, triples ? 3 * ref.total / triples : 0, 1e-12, `${name} global` );
		near( r.average, g.N ? local.reduce( ( s, c ) => s + c, 0 ) / g.N : 0, 1e-12, `${name} average` );
		const plain = clusteringCoefficient( g );
		assert.equal( plain.edgeTriangles, null );
		assert.deepEqual( plain.triangles, r.triangles );
		assert.deepEqual( plain.local, r.local );

	}

	const k = clusteringCoefficient( complete( 7 ), { edges: true } );
	assert.equal( k.triangleCount, 35 );
	assert.equal( k.global, 1 );
	assert.equal( k.average, 1 );
	assert.ok( k.triangles.every( ( t ) => t === 15 ) && k.edgeTriangles.every( ( t ) => t === 5 ) && k.local.every( ( c ) => c === 1 ) );
	for ( const g of [ path( 8 ), star( 8 ), cycle( 8 ) ] ) assert.equal( clusteringCoefficient( g ).triangleCount, 0 );
	const b = clusteringCoefficient( barbell( 4, 4 ), { edges: true } );
	assert.equal( b.triangleCount, 8 );
	assert.equal( b.edgeTriangles[ b.edgeTriangles.length - 1 ], 0 ); // the bridge closes no triangle
	near( b.local[ 3 ], 3 / 6, 1e-6 ); // a bridge end: 3 clique triangles among C( 4, 2 ) pairs

} );

// ─── spectral centralities ───────────────────────────────────────────────────────────────────

test( 'pageRank: equals the solution of the linear system (dangling nodes, damping, personalization)', () => {

	for ( const [ name, g ] of SMALL ) {

		for ( const damping of [ 0.85, 0.5 ] ) {

			const stats = {};
			const r = pageRank( g, { damping, tolerance: 1e-13, maxIterations: 2000, stats } );
			assert.ok( stats.converged, `${name}: converged` );
			assertClose( r, brutePageRank( g, damping ), 2e-7, `${name} d=${damping}` );
			near( r.reduce( ( s, x ) => s + x, 0 ), 1, 1e-6, 'sums to 1' );
			assert.ok( r.every( ( x ) => x > 0 ) );

		}

		if ( g.N < 3 ) continue;
		// personalised: weights in any scale; one-hot restart
		const weights = Array.from( { length: g.N }, ( _, v ) => ( v % 3 === 0 ? 2 : v % 3 === 1 ? 0.5 : 0 ) );
		const total = weights.reduce( ( s, w ) => s + w, 0 );
		const r = pageRank( g, { personalization: Float32Array.from( weights ), tolerance: 1e-13, maxIterations: 2000 } );
		assertClose( r, brutePageRank( g, 0.85, weights.map( ( w ) => w / total ) ), 2e-7, `${name} personalised` );
		const hot = new Array( g.N ).fill( 0 );
		hot[ 1 ] = 1;
		const walk = pageRank( g, { personalization: hot, tolerance: 1e-13, maxIterations: 2000 } );
		assertClose( walk, brutePageRank( g, 0.85, hot ), 2e-7, `${name} one-hot` );
		// restart mass never leaves the component of the restart node
		const comp = connectedComponents( g );
		walk.forEach( ( x, v ) => {

			if ( comp.component[ v ] !== comp.component[ 1 ] ) assert.equal( x, 0 );

		} );
		// all-zero or invalid personalization falls back to uniform
		assert.deepEqual( pageRank( g, { personalization: new Float32Array( g.N ) } ), pageRank( g ) );
		assert.deepEqual( pageRank( g, { personalization: new Array( g.N ).fill( NaN ) } ), pageRank( g ) );

	}

	// closed forms: regular graphs are uniform; the star solves p0 = ( 1 - d ) / n + d ( 1 - p0 )
	assertClose( pageRank( cycle( 9 ) ), new Array( 9 ).fill( 1 / 9 ), 1e-7, 'cycle' );
	assertClose( pageRank( complete( 6 ) ), new Array( 6 ).fill( 1 / 6 ), 1e-7, 'complete' );
	const n = 8, d = 0.85, centre = ( ( 1 - d ) / n + d ) / ( 1 + d );
	assertClose( pageRank( star( n ), { tolerance: 1e-12, maxIterations: 1000 } ), [ centre, ...new Array( n - 1 ).fill( ( 1 - centre ) / ( n - 1 ) ) ], 1e-7, 'star' );
	// damping 0 is the teleport vector; no edges is uniform; default options converge
	assertClose( pageRank( star( 5 ), { damping: 0 } ), new Array( 5 ).fill( 0.2 ), 1e-7, 'damping 0' );
	assertClose( pageRank( graphFromEdges( 4, [] ) ), new Array( 4 ).fill( 0.25 ), 1e-7, 'no edges' );
	assert.equal( pageRank( graphFromEdges( 0, [] ) ).length, 0 );
	const stats = {};
	pageRank( randomGraph( 40, 0.1, 9 ), { stats } );
	assert.ok( stats.converged && stats.iterations <= 85 && stats.error < 1e-6 );
	pageRank( randomGraph( 40, 0.1, 9 ), { maxIterations: 2, tolerance: 0, stats } );
	assert.deepEqual( [ stats.iterations, stats.converged ], [ 2, false ] );

} );

test( 'eigenvectorCentrality: principal eigenvector per component (Jacobi reference), bipartite, disconnected', () => {

	for ( const [ name, g ] of SMALL ) {

		const comp = connectedComponents( g );
		const stats = {};
		const r = eigenvectorCentrality( g, { tolerance: 1e-13, maxIterations: 20000, stats } );
		assert.ok( stats.converged, `${name}: converged in ${stats.iterations}` );
		assertClose( r, bruteEigenvector( g, comp ), 2e-6, name );
		assert.ok( r.every( ( x, v ) => ( g.degree[ v ] === 0 ? x === 0 : x > 0 ) ), 'positive on every node with an edge' );
		if ( g.E > 0 ) near( r.reduce( ( s, x ) => s + x * x, 0 ), 1, 1e-6, 'unit length' );
		// every component has the same mean square
		const active = Array.from( g.degree ).filter( ( d ) => d > 0 ).length;
		comp.sizes.forEach( ( size, c ) => {

			if ( size > 1 ) near( r.reduce( ( s, x, v ) => s + ( comp.component[ v ] === c ? x * x : 0 ), 0 ) / size, 1 / active, 1e-5, `${name}: component ${c}` );

		} );
		assert.deepEqual( eigenvectorCentrality( g, { tolerance: 1e-13, maxIterations: 20000, components: comp } ), r );

	}

	// bipartite graphs (plain power iteration oscillates on these)
	const n = 9;
	assertClose( eigenvectorCentrality( star( n ), { tolerance: 1e-12, maxIterations: 5000 } ), [ Math.SQRT1_2, ...new Array( n - 1 ).fill( 1 / Math.sqrt( 2 * ( n - 1 ) ) ) ], 1e-6, 'star' );
	assertClose( eigenvectorCentrality( cycle( 8 ) ), new Array( 8 ).fill( 1 / Math.sqrt( 8 ) ), 1e-6, 'even cycle' );
	const p = eigenvectorCentrality( path( 7 ), { tolerance: 1e-12, maxIterations: 5000 } );
	assertClose( p, Array.from( { length: 7 }, ( _, i ) => Math.sin( ( i + 1 ) * Math.PI / 8 ) / 2 ), 1e-6, 'path: sin( k pi / ( n + 1 ) )' );
	assertClose( eigenvectorCentrality( complete( 5 ) ), new Array( 5 ).fill( 1 / Math.sqrt( 5 ) ), 1e-6, 'complete' );
	// two components with different spectral radius: K5 (4) and a triangle (2), plus an isolated node
	const g = fromPairs( 9, [ ...cliquePairs( 0, 5 ), ...cliquePairs( 5, 3 ) ] );
	assertClose( eigenvectorCentrality( g ), [ ...new Array( 8 ).fill( 1 / Math.sqrt( 8 ) ), 0 ], 1e-6, 'per component: equal for regular components' );
	const global = eigenvectorCentrality( g, { perComponent: false, tolerance: 1e-12, maxIterations: 5000 } );
	assertClose( global, [ ...new Array( 5 ).fill( 1 / Math.sqrt( 5 ) ), 0, 0, 0, 0 ], 1e-6, 'global: only the dominant component' );
	assert.deepEqual( Array.from( eigenvectorCentrality( graphFromEdges( 3, [] ) ) ), [ 0, 0, 0 ] );
	const stats = {};
	eigenvectorCentrality( path( 30 ), { maxIterations: 3, stats } );
	assert.deepEqual( [ stats.iterations, stats.converged ], [ 3, false ] );

} );

// ─── betweenness ─────────────────────────────────────────────────────────────────────────────

test( 'betweenness / edgeBetweenness (exact Brandes) equal the pair-counting definition', () => {

	for ( const [ name, g ] of SMALL ) {

		const ref = bruteBetweenness( g, allPairs( g ) );
		const task = betweennessTask( g, { samples: Infinity, edges: true } );
		assert.equal( task.exact, true );
		runTask( task );
		const r = task.result();
		assert.equal( r.exact, true );
		assertClose( r.node, ref.node, 1e-6, `${name} nodes` );
		assertClose( r.edge, ref.edge, 1e-6, `${name} edges` );
		assertClose( betweenness( g, { samples: g.N } ), ref.node, 1e-6, `${name} betweenness()` );
		assertClose( edgeBetweenness( g, { samples: g.N } ), ref.edge, 1e-6, `${name} edgeBetweenness()` );
		// small graphs are exact by default, and the seed is then irrelevant up to rounding
		assert.equal( defaultSamples( g ), g.N );
		assertClose( betweenness( g, { seed: 99 } ), ref.node, 1e-6, `${name} default` );
		// normalised: fractions of the node pairs
		const pairs = ( g.N - 1 ) * ( g.N - 2 ) / 2;
		if ( pairs > 0 ) assertClose( betweenness( g, { normalized: true } ), ref.node.map( ( x ) => x / pairs ), 1e-6, `${name} normalised` );
		if ( g.N > 1 ) assertClose( edgeBetweenness( g, { normalized: true } ), ref.edge.map( ( x ) => x / ( g.N * ( g.N - 1 ) / 2 ) ), 1e-6, `${name} normalised edges` );
		// exact zeros stay exact zeros (leaves, clique members)
		ref.node.forEach( ( x, v ) => {

			if ( x === 0 ) assert.equal( r.node[ v ], 0, `${name}: node ${v}` );

		} );
		// every edge carries at least its own endpoints' pair; the edge values sum to the sum of all distances
		assert.ok( r.edge.every( ( x ) => x >= 1 - 1e-6 ) );
		const { dist } = allPairs( g );
		let total = 0;
		for ( let s = 0; s < g.N; s ++ ) for ( let t = s + 1; t < g.N; t ++ ) if ( dist[ s ][ t ] < Infinity ) total += dist[ s ][ t ];
		near( r.edge.reduce( ( s, x ) => s + x, 0 ), total, 1e-5, `${name}: sum of edge betweenness = sum of distances` );

	}

} );

test( 'betweenness: closed forms on path, star, cycle, complete graph, two cliques and a bridge', () => {

	const n = 9;
	assertClose( betweenness( path( n ) ), Array.from( { length: n }, ( _, i ) => i * ( n - 1 - i ) ), 1e-6, 'path' );
	assertClose( edgeBetweenness( path( n ) ), Array.from( { length: n - 1 }, ( _, i ) => ( i + 1 ) * ( n - 1 - i ) ), 1e-6, 'path edges' );
	assertClose( betweenness( star( n ) ), [ ( n - 1 ) * ( n - 2 ) / 2, ...new Array( n - 1 ).fill( 0 ) ], 1e-6, 'star' );
	assertClose( betweenness( star( n ), { normalized: true } ), [ 1, ...new Array( n - 1 ).fill( 0 ) ], 1e-6, 'star normalised' );
	assertClose( edgeBetweenness( star( n ) ), new Array( n - 1 ).fill( n - 1 ), 1e-6, 'star edges' );
	assertClose( betweenness( cycle( 9 ) ), new Array( 9 ).fill( 8 * 6 / 8 ), 1e-6, 'odd cycle: ( n - 1 )( n - 3 ) / 8' );
	assertClose( betweenness( cycle( 10 ) ), new Array( 10 ).fill( 8 * 8 / 8 ), 1e-6, 'even cycle: ( n - 2 )^2 / 8' );
	assertClose( betweenness( complete( 7 ) ), new Array( 7 ).fill( 0 ), 1e-6, 'complete' );
	assertClose( edgeBetweenness( complete( 7 ) ), new Array( 21 ).fill( 1 ), 1e-6, 'complete edges' );
	const a = 6, b = 4, g = barbell( a, b );
	const node = new Array( a + b ).fill( 0 );
	node[ a - 1 ] = ( a - 1 ) * b; // pairs ( rest of the first clique, second clique )
	node[ a ] = ( b - 1 ) * a;
	assertClose( betweenness( g ), node, 1e-6, 'barbell' );
	const edge = edgeBetweenness( g );
	assert.equal( edge[ g.E - 1 ], a * b ); // the bridge carries every pair across
	for ( let e = 0; e < g.E - 1; e ++ ) {

		const x = g.edges[ 2 * e ], y = g.edges[ 2 * e + 1 ];
		const expected = x === a - 1 || y === a - 1 ? 1 + b : x === a || y === a ? 1 + a : 1;
		assert.equal( edge[ e ], expected, `barbell edge ${x}-${y}` );

	}

	// disconnected: each component on its own
	const two = fromPairs( 9, [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ], [ 5, 6 ], [ 6, 7 ] ] );
	assertClose( betweenness( two ), [ 0, 3, 4, 3, 0, 0, 1, 0, 0 ], 1e-6, 'two paths and an isolated node' );
	assert.equal( betweenness( graphFromEdges( 5, [] ) ).every( ( x ) => x === 0 ), true );
	assert.equal( edgeBetweenness( graphFromEdges( 5, [] ) ).length, 0 );

} );

test( 'betweenness sampling: deterministic, stratified by component, unbiased, accurate at 256 pivots', () => {

	// determinism and seed sensitivity
	const g = generate( { model: 'hsbm', params: { n: 1500, avgDegree: 10, depth: 2, branching: 4 }, seed: 3 } );
	const big = graphFromEdges( g.nodeCount, g.edges );
	const a = betweenness( big, { samples: 64, seed: 5 } ), b = betweenness( big, { samples: 64, seed: 5 } ), c = betweenness( big, { samples: 64, seed: 6 } );
	assert.deepEqual( a, b );
	assert.notDeepEqual( a, c );
	assert.ok( a.every( ( x ) => Number.isFinite( x ) && x >= 0 ) );

	// accuracy of 256 pivots against the exact values (1500 BFS). On this graph (communities, no
	// dominant hubs: a hard case) seeds 1..8 give r = 0.93..0.94 for nodes and 0.75..0.78 for edges.
	const exact = betweenness( big, { samples: Infinity } );
	const estimate = betweenness( big, { samples: 256, seed: 1 } );
	const r = pearson( Array.from( exact ), Array.from( estimate ) );
	assert.ok( r > 0.9, `correlation with exact values: ${r.toFixed( 4 )}` );
	const sumExact = exact.reduce( ( s, x ) => s + x, 0 ), sumEstimate = estimate.reduce( ( s, x ) => s + x, 0 );
	assert.ok( Math.abs( sumEstimate / sumExact - 1 ) < 0.02, `total: ${sumEstimate} vs ${sumExact}` );
	const top = ( values, k ) => new Set( Array.from( values, ( x, v ) => [ x, v ] ).sort( ( p, q ) => q[ 0 ] - p[ 0 ] ).slice( 0, k ).map( ( p ) => p[ 1 ] ) );
	const topExact = top( exact, 20 ), topEstimate = top( estimate, 60 );
	assert.ok( [ ...topExact ].filter( ( v ) => topEstimate.has( v ) ).length >= 16, 'most of the 20 most central nodes are among the estimated top 60' );
	const edgeExact = edgeBetweenness( big, { samples: Infinity } ), edgeEstimate = edgeBetweenness( big, { samples: 256, seed: 1 } );
	const edgeR = pearson( Array.from( edgeExact ), Array.from( edgeEstimate ) );
	assert.ok( edgeR > 0.7, `edge correlation with exact values: ${edgeR.toFixed( 4 )}` );
	near( edgeEstimate.reduce( ( s, x ) => s + x, 0 ) / edgeExact.reduce( ( s, x ) => s + x, 0 ), 1, 0.02, 'edge total' );
	// more pivots, better estimate
	const coarse = pearson( Array.from( exact ), Array.from( betweenness( big, { samples: 32, seed: 1 } ) ) );
	const fine = pearson( Array.from( exact ), Array.from( betweenness( big, { samples: 750, seed: 1 } ) ) );
	assert.ok( coarse < r && r < fine && fine > 0.98, `32 / 256 / 750 pivots: r = ${coarse.toFixed( 3 )} / ${r.toFixed( 3 )} / ${fine.toFixed( 3 )}` );

	// stratification: every component of 2+ nodes gets its share of the pivots, at least
	// min( 32, samples ); a component no larger than its quota is exact whatever the sample
	const messy = messyGraph( 12 );
	const comp = connectedComponents( messy );
	const ref = bruteBetweenness( messy, allPairs( messy ) );
	const task = betweennessTask( messy, { samples: 6, seed: 4, edges: true } );
	assert.equal( task.exact, false );
	assert.deepEqual( Array.from( comp.sizes ), [ 16, 6, 3, 2, 1, 1 ] );
	const quota = [ 6, 6, 3, 2, 0, 0 ]; // only the largest component is sampled
	assert.equal( task.total, 17 );
	runTask( task );
	const sampled = task.result();
	assert.equal( sampled.exact, false );
	assert.equal( sampled.pivots, task.total );
	for ( let v = 0; v < messy.N; v ++ ) if ( quota[ comp.component[ v ] ] === comp.sizes[ comp.component[ v ] ] ) near( sampled.node[ v ], ref.node[ v ], 1e-6, `node ${v} of a fully sampled component` );
	for ( let e = 0; e < messy.E; e ++ ) if ( quota[ comp.component[ messy.edges[ 2 * e ] ] ] === comp.sizes[ comp.component[ messy.edges[ 2 * e ] ] ] ) near( sampled.edge[ e ], ref.edge[ e ], 1e-6 );
	// a sample as large as the graph is the exact algorithm, component by component
	const full = betweennessTask( messy, { samples: messy.N, edges: true } );
	assert.equal( full.exact, true );
	assert.equal( full.total, messy.N - 2 ); // every node but the two isolated ones
	// default settings: a large sample leaves no component with fewer than 32 pivots, so the
	// small ones are exact; an explicit small sample is respected (but never below 2 per component)
	const islands = fromPairs( 400, [ ...Array.from( { length: 299 }, ( _, i ) => [ i, i + 1 ] ), ...Array.from( { length: 39 }, ( _, i ) => [ 300 + i, 301 + i ] ), ...cliquePairs( 340, 20 ), ...Array.from( { length: 39 }, ( _, i ) => [ 360 + i, 361 + i ] ) ] );
	const plan = betweennessTask( islands, { samples: 100 } );
	assert.equal( plan.total, Math.round( 100 * 300 / 400 ) + 32 + 20 + 32 ); // sizes 300, 40, 20, 40
	runTask( plan );
	const islandsExact = betweenness( islands, { samples: Infinity } );
	for ( let v = 340; v < 360; v ++ ) assert.equal( plan.result().node[ v ], islandsExact[ v ] );
	assert.equal( betweennessTask( islands, { samples: 1 } ).total, 8 );
	assert.equal( betweennessTask( islands, { samples: NaN } ).total, 400 ); // falls back to the default, here exact
	for ( const samples of [ undefined, null, 'auto' ] ) assert.equal( betweennessTask( islands, { samples } ).total, 400 );

	// unbiased: the mean over many seeds converges to the exact value, for nodes and edges
	const runs = 600;
	const meanNode = new Float64Array( messy.N ), sqNode = new Float64Array( messy.N );
	const meanEdge = new Float64Array( messy.E ), sqEdge = new Float64Array( messy.E );
	for ( let seed = 1; seed <= runs; seed ++ ) {

		const t = betweennessTask( messy, { samples: 6, seed, edges: true } );
		runTask( t );
		const x = t.result();
		x.node.forEach( ( value, v ) => {

			meanNode[ v ] += value / runs; sqNode[ v ] += value * value / runs;

		} );
		x.edge.forEach( ( value, e ) => {

			meanEdge[ e ] += value / runs; sqEdge[ e ] += value * value / runs;

		} );

	}

	const check = ( mean, sq, expected, what ) => {

		for ( let i = 0; i < expected.length; i ++ ) {

			const standardError = Math.sqrt( Math.max( 0, sq[ i ] - mean[ i ] * mean[ i ] ) / runs );
			assert.ok( Math.abs( mean[ i ] - expected[ i ] ) <= 4.5 * standardError + 1e-6, `${what} ${i}: mean ${mean[ i ].toFixed( 3 )} vs exact ${expected[ i ].toFixed( 3 )} (se ${standardError.toFixed( 3 )})` );

		}

	};

	check( meanNode, sqNode, ref.node, 'node' );
	check( meanEdge, sqEdge, ref.edge, 'edge' );
	// the estimates are not trivially exact: this graph's big component really is sampled
	assert.ok( Array.from( sqNode ).some( ( s, v ) => s - meanNode[ v ] ** 2 > 1 ) );

} );

// ─── tasks ───────────────────────────────────────────────────────────────────────────────────

test( 'tasks: chunking does not change the result; partial results are valid; progress and cancellation', async () => {

	const g = randomGraph( 60, 0.08, 21 );
	for ( const make of [ ( o ) => betweennessTask( g, { edges: true, ...o } ), ( o ) => closenessTask( g, o ) ] ) {

		const whole = make( { samples: 24, seed: 3 } );
		assert.equal( whole.done, false );
		assert.equal( whole.completed, 0 );
		assert.equal( whole.step( whole.total ), true );
		const single = make( { samples: 24, seed: 3 } );
		let steps = 0;
		while ( ! single.step() ) steps ++;
		assert.equal( steps + 1, single.total );
		const uneven = make( { samples: 24, seed: 3 } );
		for ( const n of [ 1, 5, 2, 1000 ] ) uneven.step( n );
		assert.ok( uneven.done && uneven.completed === uneven.total );
		assert.deepEqual( single.result(), whole.result() );
		assert.deepEqual( uneven.result(), whole.result() );
		assert.equal( whole.step( 3 ), true ); // stepping a finished task is harmless
		assert.deepEqual( whole.result(), single.result() );

		// partial result = the result of a run that stopped there; values are finite
		const partial = make( { samples: 24, seed: 3 } );
		partial.step( 7 );
		const early = partial.result();
		assert.equal( early.pivots, 7 );
		assert.equal( early.exact, false );
		for ( const key of Object.keys( early ) ) if ( early[ key ] && early[ key ].length !== undefined ) assert.ok( Array.from( early[ key ] ).every( Number.isFinite ), key );
		partial.step( 1000 );
		assert.deepEqual( partial.result(), whole.result() );

		// runTask: progress is monotonic and ends at total; returning false stops
		const seen = [];
		const t = make( { samples: 24, seed: 3 } );
		assert.equal( runTask( t, { chunk: 5, onProgress: ( done, total ) => {

			seen.push( done );
			assert.equal( total, t.total );

		} } ), true );
		assert.deepEqual( seen, Array.from( { length: Math.ceil( t.total / 5 ) }, ( _, i ) => Math.min( t.total, 5 * ( i + 1 ) ) ) );
		const stopped = make( { samples: 24, seed: 3 } );
		assert.equal( runTask( stopped, { chunk: 4, onProgress: ( done ) => done < 8 } ), false );
		assert.equal( stopped.completed, 8 );

		// runTaskAsync: same result, yields between slices, abortable
		const slow = make( { samples: 24, seed: 3 } );
		let yields = 0;
		const finished = await runTaskAsync( slow, { budgetMs: 0, wait: () => {

			yields ++;
			return Promise.resolve();

		} } );
		assert.equal( finished, true );
		assert.equal( yields, slow.total - 1 ); // budget 0: one pivot per slice
		assert.deepEqual( slow.result(), whole.result() );
		const controller = new AbortController();
		const aborted = make( { samples: 24, seed: 3 } );
		const progress = [];
		const outcome = await runTaskAsync( aborted, { budgetMs: 0, signal: controller.signal, onProgress: ( done ) => {

			progress.push( done );
			if ( done === 3 ) controller.abort();

		} } );
		assert.equal( outcome, false );
		assert.deepEqual( progress, [ 1, 2, 3 ] );
		assert.equal( aborted.completed, 3 );
		assert.equal( await runTaskAsync( make( { samples: 24 } ), { signal: controller.signal } ), false ); // already aborted
		assert.equal( await runTaskAsync( make( { samples: 24 } ), { budgetMs: 0, onProgress: () => false } ), false );
		assert.equal( await runTaskAsync( make( { samples: 24 } ) ), true ); // default budget and setTimeout yield

	}

	// betweenness() with onProgress: cancelling returns the estimate of the pivots done so far
	let calls = 0;
	const cut = betweenness( g, { samples: 24, seed: 3, chunk: 6, onProgress: () => ++ calls < 2 } );
	const reference = betweennessTask( g, { samples: 24, seed: 3 } );
	reference.step( 12 );
	assert.deepEqual( cut, reference.result().node );
	// a graph without edges has nothing to do
	const empty = betweennessTask( graphFromEdges( 4, [] ) );
	assert.deepEqual( [ empty.total, empty.done, empty.exact ], [ 0, true, true ] );
	assert.equal( runTask( empty ), true );
	assert.equal( await runTaskAsync( closenessTask( graphFromEdges( 4, [] ) ) ), true );

} );

// ─── closeness, distances ────────────────────────────────────────────────────────────────────

test( 'closeness / harmonic / distance distribution (exact) equal the all-pairs reference', () => {

	for ( const [ name, g ] of SMALL ) {

		const { dist } = allPairs( g );
		const r = closeness( g, { samples: Infinity } );
		assert.equal( r.exact, true );
		const expectedCloseness = [], expectedHarmonic = [];
		const histogram = [];
		let pairs = 0, total = 0, maxDistance = 0;
		for ( let v = 0; v < g.N; v ++ ) {

			let reach = 0, far = 0, inv = 0;
			for ( let u = 0; u < g.N; u ++ ) {

				const d = dist[ v ][ u ];
				if ( u === v || d === Infinity ) continue;
				reach ++; far += d; inv += 1 / d;
				histogram[ d ] = ( histogram[ d ] || 0 ) + 1;
				pairs ++; total += d;
				maxDistance = Math.max( maxDistance, d );

			}

			expectedCloseness.push( reach ? reach / far * reach / ( g.N - 1 ) : 0 );
			expectedHarmonic.push( g.N > 1 ? inv / ( g.N - 1 ) : 0 );

		}

		assertClose( r.closeness, expectedCloseness, 1e-6, `${name} closeness` );
		assertClose( r.harmonic, expectedHarmonic, 1e-6, `${name} harmonic` );
		near( r.averagePathLength, pairs ? total / pairs : 0, 1e-12, `${name} average path length` );
		assert.equal( r.maxDistance, maxDistance );
		assert.equal( r.distanceDistribution.length, maxDistance + 1 );
		for ( let d = 1; d <= maxDistance; d ++ ) near( r.distanceDistribution[ d ], ( histogram[ d ] || 0 ) / pairs, 1e-12, `${name} P( d = ${d} )` );
		if ( pairs ) {

			near( r.distanceDistribution.reduce( ( s, x ) => s + x, 0 ), 1, 1e-12 );
			assert.ok( r.effectiveDiameter > 0 && r.effectiveDiameter <= maxDistance );
			// 90% of the pairs are within ceil( effective diameter )
			let within = 0;
			for ( let d = 1; d <= Math.ceil( r.effectiveDiameter - 1e-9 ); d ++ ) within += r.distanceDistribution[ d ];
			assert.ok( within >= 0.9 - 1e-9 );

		}

		assert.ok( r.closeness.every( ( x ) => x >= 0 && x <= 1 + 1e-6 ) && r.harmonic.every( ( x ) => x >= 0 && x <= 1 + 1e-6 ) );
		const apl = averagePathLength( g, { samples: Infinity } );
		assert.equal( apl.averagePathLength, r.averagePathLength );
		assert.deepEqual( apl.distanceDistribution, r.distanceDistribution );
		assert.equal( apl.exact, true );

	}

	// closed forms
	assertClose( closeness( complete( 6 ) ).closeness, new Array( 6 ).fill( 1 ), 1e-6, 'complete' );
	assertClose( closeness( complete( 6 ) ).harmonic, new Array( 6 ).fill( 1 ), 1e-6, 'complete harmonic' );
	const n = 7, s = closeness( star( n ) );
	assertClose( s.closeness, [ 1, ...new Array( n - 1 ).fill( ( n - 1 ) / ( 1 + 2 * ( n - 2 ) ) ) ], 1e-6, 'star' );
	assertClose( s.harmonic, [ 1, ...new Array( n - 1 ).fill( ( 1 + ( n - 2 ) / 2 ) / ( n - 1 ) ) ], 1e-6, 'star harmonic' );
	near( s.averagePathLength, 2 * ( n - 1 ) * ( 1 + ( n - 2 ) ) / ( n * ( n - 1 ) ), 1e-12, 'star average path length' );
	near( closeness( path( 5 ) ).averagePathLength, 2, 1e-12, 'path: ( n + 1 ) / 3' );
	near( closeness( cycle( 8 ) ).averagePathLength, 16 / 7, 1e-12, 'cycle' );
	// disconnected: a triangle and an edge. Triangle nodes reach 2 of 4 others at distance 1.
	const g = fromPairs( 5, [ [ 0, 1 ], [ 1, 2 ], [ 0, 2 ], [ 3, 4 ] ] );
	assertClose( closeness( g ).closeness, [ 0.5, 0.5, 0.5, 0.25, 0.25 ], 1e-6, 'Wasserman-Faust scaling' );
	assertClose( closeness( g ).harmonic, [ 0.5, 0.5, 0.5, 0.25, 0.25 ], 1e-6, 'harmonic' );
	const none = closeness( graphFromEdges( 3, [] ) );
	assert.deepEqual( [ none.averagePathLength, none.effectiveDiameter, none.maxDistance, none.exact ], [ 0, 0, 0, true ] );
	assert.deepEqual( Array.from( none.closeness ), [ 0, 0, 0 ] );

} );

test( 'closeness sampling: deterministic, pivots exact, estimates unbiased and accurate', () => {

	const raw = generate( { model: 'hsbm', params: { n: 1500, avgDegree: 10, depth: 2, branching: 4 }, seed: 3 } );
	const big = graphFromEdges( raw.nodeCount, raw.edges );
	const exact = closeness( big, { samples: Infinity } );
	const a = closeness( big, { samples: 128, seed: 2 } );
	assert.deepEqual( closeness( big, { samples: 128, seed: 2 } ), a );
	assert.notDeepEqual( closeness( big, { samples: 128, seed: 3 } ).closeness, a.closeness );
	assert.equal( a.exact, false );
	// measured over seeds 1..8 with 128 pivots: r = 0.94..0.97 (closeness), 0.90..0.94 (harmonic)
	assert.ok( pearson( Array.from( exact.closeness ), Array.from( a.closeness ) ) > 0.9 );
	assert.ok( pearson( Array.from( exact.harmonic ), Array.from( a.harmonic ) ) > 0.85 );
	let worst = 0;
	for ( let v = 0; v < big.N; v ++ ) if ( exact.closeness[ v ] > 0 ) worst = Math.max( worst, Math.abs( a.closeness[ v ] / exact.closeness[ v ] - 1 ) );
	assert.ok( worst < 0.12, `worst relative closeness error with 128 pivots: ${worst.toFixed( 3 )}` );
	near( a.averagePathLength, exact.averagePathLength, 0.02, 'average path length' );
	near( a.effectiveDiameter, exact.effectiveDiameter, 0.05, 'effective diameter' );
	assert.ok( a.maxDistance <= exact.maxDistance );
	near( averagePathLength( big ).averagePathLength, exact.averagePathLength, 0.03, '64 pivots by default' );
	assert.equal( averagePathLength( big ).pivots, 64 );

	// distance statistics when components are sampled at different rates: every pivot's pairs
	// count for size / quota pivots. K40 and C20 are vertex-transitive, so whichever pivots are
	// drawn, the weighted estimate must be the exact value: ( 40 * 39 * 1 + 20 * 100 ) / ( 40 * 39 + 20 * 19 )
	const mixed = fromPairs( 60, [ ...cliquePairs( 0, 40 ), ...Array.from( { length: 20 }, ( _, i ) => [ 40 + i, 40 + ( i + 1 ) % 20 ] ) ] );
	const sampledMixed = closeness( mixed, { samples: 12, seed: 5 } );
	assert.deepEqual( [ sampledMixed.pivots, sampledMixed.exact ], [ 24, false ] ); // 12 of 40 and 12 of 20
	near( sampledMixed.averagePathLength, 3560 / 1940, 1e-12, 'weighted average path length' );
	assertClose( sampledMixed.distanceDistribution, closeness( mixed, { samples: Infinity } ).distanceDistribution, 1e-12, 'weighted distance distribution' );
	assert.equal( sampledMixed.maxDistance, 10 );

	// pivots carry their exact values; small components are exact; means over seeds converge
	const messy = messyGraph( 12 );
	const comp = connectedComponents( messy );
	const ref = closeness( messy, { samples: Infinity } );
	const task = closenessTask( messy, { samples: 6, seed: 1 } );
	task.step( 2 );
	const partial = task.result();
	runTask( task );
	const sampled = task.result();
	let exactNodes = 0;
	for ( let v = 0; v < messy.N; v ++ ) {

		if ( comp.sizes[ comp.component[ v ] ] <= 6 ) near( sampled.closeness[ v ], ref.closeness[ v ], 1e-6, `node ${v} of a fully sampled component` );
		if ( Math.abs( sampled.closeness[ v ] - ref.closeness[ v ] ) < 1e-7 && Math.abs( sampled.harmonic[ v ] - ref.harmonic[ v ] ) < 1e-7 ) exactNodes ++;
		assert.ok( Number.isFinite( partial.closeness[ v ] ) && partial.closeness[ v ] >= 0 );

	}

	assert.ok( exactNodes >= task.total, 'at least the pivots are exact' );
	assert.ok( exactNodes < messy.N, 'the big component is estimated' );
	const runs = 600;
	const meanFar = new Float64Array( messy.N ), sqFar = new Float64Array( messy.N ), meanHarmonic = new Float64Array( messy.N ), sqHarmonic = new Float64Array( messy.N );
	const refFar = Array.from( ref.closeness, ( c ) => ( c > 0 ? 1 / c : 0 ) );
	for ( let seed = 1; seed <= runs; seed ++ ) {

		const x = closeness( messy, { samples: 6, seed } );
		for ( let v = 0; v < messy.N; v ++ ) {

			const far = x.closeness[ v ] > 0 ? 1 / x.closeness[ v ] : 0; // proportional to the estimated distance sum
			meanFar[ v ] += far / runs; sqFar[ v ] += far * far / runs;
			meanHarmonic[ v ] += x.harmonic[ v ] / runs; sqHarmonic[ v ] += x.harmonic[ v ] ** 2 / runs;

		}

	}

	for ( let v = 0; v < messy.N; v ++ ) {

		const seFar = Math.sqrt( Math.max( 0, sqFar[ v ] - meanFar[ v ] ** 2 ) / runs ), seHarmonic = Math.sqrt( Math.max( 0, sqHarmonic[ v ] - meanHarmonic[ v ] ** 2 ) / runs );
		assert.ok( Math.abs( meanFar[ v ] - refFar[ v ] ) <= 4.5 * seFar + 1e-5, `distance sum of node ${v}: ${meanFar[ v ]} vs ${refFar[ v ]} (se ${seFar})` );
		assert.ok( Math.abs( meanHarmonic[ v ] - ref.harmonic[ v ] ) <= 4.5 * seHarmonic + 1e-6, `harmonic of node ${v}: ${meanHarmonic[ v ]} vs ${ref.harmonic[ v ]} (se ${seHarmonic})` );

	}

} );

test( 'estimateDiameter: bounds always hold, exact with enough BFS, endpoints realise the diameter', () => {

	for ( const [ name, g ] of SMALL ) {

		const { dist } = allPairs( g );
		const ecc = dist.map( ( row ) => Math.max( ...row.filter( ( d ) => d < Infinity ) ) );
		const diameter = Math.max( 0, ...ecc );
		for ( const maxBfs of [ 0, 1, 2, 3, 5, 1000 ] ) {

			const r = estimateDiameter( g, { maxBfs } );
			assert.ok( r.bfsCount <= maxBfs );
			assert.ok( r.diameter <= diameter && diameter <= r.diameterUpper, `${name} maxBfs=${maxBfs}: ${r.diameter} <= ${diameter} <= ${r.diameterUpper}` );
			for ( let v = 0; v < g.N; v ++ ) assert.ok( r.eccLower[ v ] <= ecc[ v ] && ecc[ v ] <= r.eccUpper[ v ], `${name} maxBfs=${maxBfs}: ecc bounds of node ${v}: ${r.eccLower[ v ]} <= ${ecc[ v ]} <= ${r.eccUpper[ v ]}` );
			if ( r.exact ) assert.equal( r.diameter, r.diameterUpper );
			if ( g.E > 0 ) assert.equal( dist[ r.endpoints[ 0 ] ][ r.endpoints[ 1 ] ], r.diameter, 'the endpoints are that far apart' );
			else assert.deepEqual( Array.from( r.endpoints ), [ - 1, - 1 ] );
			if ( maxBfs === 1000 ) {

				assert.equal( r.exact, true, `${name}: exact` );
				assert.equal( r.diameter, diameter );

			}

			if ( maxBfs === 0 ) assert.equal( r.bfsCount, 0 );

		}

	}

	assert.equal( estimateDiameter( path( 40 ) ).diameter, 39 );
	assert.ok( estimateDiameter( path( 40 ) ).exact && estimateDiameter( path( 40 ) ).bfsCount <= 3, 'a path needs the double sweep only' );
	assert.equal( estimateDiameter( cycle( 21 ) ).diameter, 10 );
	assert.deepEqual( [ estimateDiameter( complete( 9 ) ).diameter, estimateDiameter( star( 9 ) ).diameter ], [ 1, 2 ] );
	// disconnected: the largest finite distance, wherever it is (here in the smaller component)
	const g = fromPairs( 12, [ ...cliquePairs( 0, 7 ), [ 7, 8 ], [ 8, 9 ], [ 9, 10 ], [ 10, 11 ] ] );
	const r = estimateDiameter( g );
	assert.deepEqual( [ r.diameter, r.exact ], [ 4, true ] );
	assert.deepEqual( Array.from( r.endpoints ).sort( ( a, b ) => a - b ), [ 7, 11 ] );
	const components = connectedComponents( g );
	assert.deepEqual( estimateDiameter( g, { components } ), r );

} );

// ─── statistics ──────────────────────────────────────────────────────────────────────────────

test( 'density, degree histogram, assortativity, graphSummary', () => {

	for ( const [ name, g ] of SMALL ) {

		near( density( g ), g.N > 1 ? g.E / ( g.N * ( g.N - 1 ) / 2 ) : 0, 1e-12, name );
		const h = degreeHistogram( g );
		const degrees = Array.from( g.degree );
		assert.equal( h.length, Math.max( 0, ...degrees ) + 1 );
		h.forEach( ( count, d ) => assert.equal( count, degrees.filter( ( x ) => x === d ).length ) );
		// Pearson correlation over both directions of every edge
		const x = [], y = [];
		for ( let e = 0; e < g.E; e ++ ) {

			const a = g.degree[ g.edges[ 2 * e ] ], b = g.degree[ g.edges[ 2 * e + 1 ] ];
			x.push( a, b ); y.push( b, a );

		}

		const expected = g.E ? pearson( x, y ) : NaN;
		const r = degreeAssortativity( g );
		if ( Number.isNaN( expected ) || new Set( x ).size === 1 ) assert.ok( Number.isNaN( r ), `${name}: undefined for regular graphs` );
		else near( r, expected, 1e-9, `${name} assortativity` );
		const s = graphSummary( g );
		const sorted = degrees.slice().sort( ( a, b ) => a - b );
		assert.deepEqual(
			[ s.nodes, s.edges, s.minDegree, s.maxDegree, s.isolated, s.medianDegree ],
			[ g.N, g.E, sorted[ 0 ], sorted[ g.N - 1 ], degrees.filter( ( d ) => d === 0 ).length, ( sorted[ ( g.N - 1 ) >> 1 ] + sorted[ g.N >> 1 ] ) / 2 ], name );
		near( s.meanDegree, 2 * g.E / g.N, 1e-12 );
		assert.equal( s.density, density( g ) );
		assert.deepEqual( s.degreeHistogram, h );
		assert.ok( Object.is( s.assortativity, r ) );

	}

	assert.equal( degreeAssortativity( star( 9 ) ), - 1 );
	assert.equal( density( complete( 9 ) ), 1 );
	assert.ok( Number.isNaN( degreeAssortativity( cycle( 9 ) ) ) && Number.isNaN( degreeAssortativity( complete( 5 ) ) ) && Number.isNaN( degreeAssortativity( graphFromEdges( 3, [] ) ) ) );
	near( degreeAssortativity( path( 4 ) ), - 0.5, 1e-12, 'path of 4' );
	const empty = graphSummary( graphFromEdges( 0, [] ) );
	assert.deepEqual( [ empty.nodes, empty.density, empty.meanDegree, empty.medianDegree, empty.isolated ], [ 0, 0, 0, 0, 0 ] );

} );

// ─── normalisation ───────────────────────────────────────────────────────────────────────────

test( 'normalize: linear, log, rank, robust; ties, constants, non-finite values, gamma, in place', () => {

	const v = Float32Array.from( [ 3, 1, 4, 1, 5, 9, 2, 6 ] );
	assertClose( normalize( v ), Array.from( v, ( x ) => ( x - 1 ) / 8 ), 1e-7, 'linear' );
	assertClose( normalize( v, { mode: 'log' } ), Array.from( v, ( x ) => Math.log( x ) / Math.log( 9 ) ), 1e-6, 'log of positive values' );
	// rank: share of the values above the minimum that are <= v. Sorted: 1 1 2 3 4 5 6 9
	assertClose( normalize( v, { mode: 'rank' } ), [ 2, 0, 3, 0, 4, 6, 1, 5 ].map( ( r ) => r / 6 ), 1e-7, 'rank' );
	assertClose( normalize( [ 10, 30, 20, 40 ], { mode: 'rank' } ), [ 0, 2 / 3, 1 / 3, 1 ], 1e-7, 'rank without ties = rank / ( n - 1 )' );
	// many ties at both ends (the zeros of betweenness, the ones of clustering): still 0 and 1
	assertClose( normalize( [ 0, 0, 0, 0, 0, 0, 2, 5, 9, 9 ], { mode: 'rank' } ), [ 0, 0, 0, 0, 0, 0, 0.25, 0.5, 1, 1 ], 1e-7, 'rank with ties at the ends' );
	// with zeros: log1p in units of the smallest positive value
	const z = Float32Array.from( [ 0, 0, 1, 10, 100 ] );
	assertClose( normalize( z, { mode: 'log' } ), Array.from( z, ( x ) => Math.log1p( x ) / Math.log1p( 100 ) ), 1e-6, 'log with zeros' );
	// logRange caps the dynamic range: with 2 decades, values 1e-6 and 1e-3 of the maximum collapse
	assertClose( normalize( [ 1e-6, 1e-3, 1e-2, 0.1, 1 ], { mode: 'log', logRange: 100 } ), [ 0, 0, 0, 0.5, 1 ], 1e-6, 'logRange' );
	assertClose( normalize( [ 0, 1e-9, 1, 100 ], { mode: 'log', logRange: 100 } ), [ 0, 0, Math.log1p( 1 ) / Math.log1p( 100 ), 1 ], 1e-6, 'logRange with zeros' );
	// robust: one outlier does not flatten the rest
	const o = Float32Array.from( [ ...Array.from( { length: 99 }, ( _, i ) => i ), 1e6 ] );
	const lin = normalize( o ), rob = normalize( o, { mode: 'robust' } );
	assert.ok( lin[ 98 ] < 1e-3 && rob[ 50 ] > 0.45 && rob[ 50 ] < 0.55 && rob[ 99 ] === 1 && rob[ 0 ] === 0 );
	const [ q2, q98 ] = quantiles( o, [ 0.02, 0.98 ] );
	assertClose( rob, Array.from( o, ( x ) => Math.min( 1, Math.max( 0, ( x - q2 ) / ( q98 - q2 ) ) ) ), 1e-6, 'robust = clamped percentiles' );
	assertClose( normalize( o, { mode: 'robust', percentiles: [ 0, 1 ] } ), lin, 1e-7, 'percentiles 0..1 = linear' );
	// more than 98% identical values: falls back to min / max instead of dividing by zero
	const sparse = new Float32Array( 200 );
	sparse[ 7 ] = 5;
	assert.deepEqual( normalize( sparse, { mode: 'robust' } ), normalize( sparse ) );

	for ( const mode of [ 'linear', 'log', 'rank', 'robust' ] ) {

		// constants map to `flat`, non-finite values to `invalid`, the output stays in [0,1]
		assert.deepEqual( Array.from( normalize( [ 7, 7, 7 ], { mode } ) ), [ 0.5, 0.5, 0.5 ] );
		assert.deepEqual( Array.from( normalize( [ 7, NaN, 7 ], { mode, flat: 1, invalid: 0.25 } ) ), [ 1, 0.25, 1 ] );
		assert.equal( normalize( [], { mode } ).length, 0 );
		const messy = Float32Array.from( [ 2, NaN, 8, Infinity, 4, - Infinity, 2 ] );
		const m = normalize( messy, { mode, invalid: - 1 } );
		assert.deepEqual( [ m[ 1 ], m[ 3 ], m[ 5 ] ], [ - 1, - 1, - 1 ], mode );
		assert.ok( m[ 0 ] === m[ 6 ] && m[ 0 ] < m[ 4 ] && m[ 4 ] < m[ 2 ], `${mode}: order kept` );
		// monotonic and within [0,1] on random data with ties, negative values and zeros
		const rand = rng( 5 );
		const data = Float32Array.from( { length: 500 }, () => Math.round( ( rand() - 0.3 ) * 40 ) / 4 );
		const t = normalize( data, { mode } );
		const order = Array.from( data.keys() ).sort( ( a, b ) => data[ a ] - data[ b ] );
		for ( let i = 1; i < order.length; i ++ ) {

			const a = order[ i - 1 ], b = order[ i ];
			assert.ok( data[ a ] === data[ b ] ? t[ a ] === t[ b ] : t[ a ] <= t[ b ], `${mode}: monotonic` );

		}

		assert.ok( t.every( ( x ) => x >= 0 && x <= 1 ) );
		assert.deepEqual( [ Math.min( ...t ), Math.max( ...t ) ], [ 0, 1 ], `${mode}: uses the whole range` );
		// gamma, and writing into a given array, the input included
		const g2 = normalize( data, { mode, gamma: 0.5 } );
		assertClose( g2, Array.from( t, Math.sqrt ), 1e-6, `${mode} gamma` );
		const out = new Float32Array( data.length );
		assert.equal( normalize( data, { mode, out } ), out );
		assert.deepEqual( out, t );
		const inPlace = Float32Array.from( messy );
		normalize( inPlace, { mode, gamma: 2, invalid: - 1, out: inPlace } );
		assertClose( inPlace, Array.from( m, ( x ) => ( x < 0 ? - 1 : x * x ) ), 1e-6, `${mode} in place` );

	}

	// rank: the Float32Array fast path equals the generic path (plain arrays, Float64Array, integer arrays)
	const rand = rng( 8 );
	const f32 = Float32Array.from( { length: 3000 }, ( _, i ) => ( i % 7 === 0 ? 0 : i % 11 === 0 ? - 0 : i % 13 === 0 ? NaN : Math.round( ( rand() - 0.5 ) * 200 ) / 8 ) );
	const fast = normalize( f32, { mode: 'rank' } );
	assert.deepEqual( normalize( Array.from( f32 ), { mode: 'rank' } ), fast );
	assert.deepEqual( normalize( Float64Array.from( f32 ), { mode: 'rank' } ), fast );
	const finite = Array.from( f32 ).filter( Number.isFinite );
	const atMin = finite.filter( ( y ) => y === Math.min( ...finite ) ).length;
	f32.forEach( ( x, i ) => {

		if ( ! Number.isFinite( x ) ) return assert.equal( fast[ i ], 0 );
		near( fast[ i ], ( finite.filter( ( y ) => y <= x ).length - atMin ) / ( finite.length - atMin ), 1e-6, 'rank' );

	} );
	assertClose( normalize( Uint32Array.from( [ 5, 1, 3, 3 ] ), { mode: 'rank' } ), [ 1, 0, 2 / 3, 2 / 3 ], 1e-7, 'integer input' );
	assert.throws( () => normalize( [ 1, 2 ], { mode: 'sigmoid' } ), /unknown mode/ );

	assert.deepEqual( quantiles( [ 4, 1, 3, 2 ], [ 0, 0.5, 1, 0.25 ] ), [ 1, 2.5, 4, 1.75 ] );
	assert.deepEqual( quantiles( [ NaN, 2, Infinity ], [ 0.5 ] ), [ 2 ] );
	assert.ok( Number.isNaN( quantiles( [], [ 0.5 ] )[ 0 ] ) );

} );

// ─── degenerate graphs ───────────────────────────────────────────────────────────────────────

test( 'graphs with 0, 1 and 2 nodes: nothing throws, shapes and values are the trivial ones', () => {

	const cases = [ graphFromEdges( 0, [] ), graphFromEdges( 1, [] ), graphFromEdges( 2, [] ), graphFromEdges( 2, [ 0, 1 ] ) ];
	for ( const g of cases ) {

		const { N, E } = g, name = `N = ${N}, E = ${E}`;
		assert.equal( buildEdgeIndex( { ...g, adjEdge: undefined } ).length, Math.max( 1, 2 * E ), name );
		assert.equal( connectedComponents( g ).count, E ? 1 : N, name );
		assert.equal( kCore( g ).degeneracy, E, name );
		const triangles = clusteringCoefficient( g, { edges: true } );
		assert.deepEqual( [ triangles.global, triangles.average, triangles.triangleCount, triangles.edgeTriangles.length ], [ 0, 0, 0, E ], name );
		assertClose( pageRank( g ), new Array( N ).fill( 1 / N ), 1e-6, name );
		assertClose( eigenvectorCentrality( g ), new Array( N ).fill( E ? Math.SQRT1_2 : 0 ), 1e-6, name );
		assert.deepEqual( Array.from( betweenness( g ) ), new Array( N ).fill( 0 ), name );
		assert.deepEqual( Array.from( edgeBetweenness( g ) ), new Array( E ).fill( 1 ), name );
		const c = closeness( g );
		assert.deepEqual( [ Array.from( c.closeness ), Array.from( c.harmonic ), c.averagePathLength, c.maxDistance, c.exact ], [ new Array( N ).fill( E ), new Array( N ).fill( E ), E, E, true ], name );
		assert.equal( averagePathLength( g ).averagePathLength, E, name );
		const d = estimateDiameter( g );
		assert.deepEqual( [ d.diameter, d.diameterUpper, d.exact, Array.from( d.endpoints ) ], [ E, E, true, E ? [ 0, 1 ] : [ - 1, - 1 ] ], name );
		const summary = graphSummary( g );
		assert.deepEqual( [ summary.nodes, summary.edges, summary.density, summary.maxDegree, summary.meanDegree, summary.isolated ], [ N, E, E, E, E, E ? 0 : N ], name );
		assert.ok( Number.isNaN( summary.assortativity ), name );
		assert.equal( defaultSamples( g ), N, name );
		for ( const m of NODE_METRICS ) {

			const values = nodeMetric( g, m.id );
			assert.ok( values.length === N && values.every( Number.isFinite ), `${name}: ${m.id}` );
			assert.ok( normalize( values, { mode: m.scale } ).every( ( x ) => x >= 0 && x <= 1 ), `${name}: ${m.id}` );

		}

		if ( N === 0 ) {

			assert.equal( bfs( g, [] ).reached, 0 );
			assert.equal( khopNeighborhood( g, [], 2 ).nodes.length, 0 );
			assert.throws( () => bfs( g, 0 ), RangeError );
			continue;

		}

		assert.equal( bfs( g, 0 ).reached, 1 + E, name );
		assert.equal( khopNeighborhood( g, 0, 2 ).nodes.length, 1 + E, name );
		const positions = Float32Array.from( { length: 4 * N }, ( _, i ) => i );
		for ( const find of [ shortestPath, ( h, s, t ) => shortestPathWeighted( h, s, t, positions ), shortestPathDag ] ) {

			const self = find( g, 0, 0 );
			assert.deepEqual( [ Array.from( self.nodes ), Array.from( self.edges ), self.length ], [ [ 0 ], [], 0 ], name );
			if ( N === 2 ) {

				const across = find( g, 0, 1 );
				if ( E === 0 ) assert.equal( across, null, name );
				else assert.deepEqual( [ Array.from( across.nodes ), Array.from( across.edges ) ], [ [ 0, 1 ], [ 0 ] ], name );

			}

		}

	}

} );

// ─── catalogue ───────────────────────────────────────────────────────────────────────────────

test( 'NODE_METRICS / nodeMetric: every listed metric computes a finite Float32Array of length N', () => {

	const g = messyGraph( 3 );
	const modes = new Set( [ 'linear', 'log', 'rank', 'robust' ] );
	assert.equal( new Set( NODE_METRICS.map( ( m ) => m.id ) ).size, NODE_METRICS.length );
	for ( const m of NODE_METRICS ) {

		assert.ok( m.label && m.description && modes.has( m.scale ) && [ 'instant', 'fast', 'slow' ].includes( m.cost ), m.id );
		const values = nodeMetric( g, m.id );
		assert.ok( values instanceof Float32Array && values.length === g.N, m.id );
		assert.ok( values.every( ( x ) => Number.isFinite( x ) && x >= 0 ), m.id );
		const t = normalize( values, { mode: m.scale } );
		assert.ok( t.every( ( x ) => x >= 0 && x <= 1 ), m.id );

	}

	assert.deepEqual( nodeMetric( g, 'degree' ), Float32Array.from( g.degree ) );
	assert.deepEqual( nodeMetric( g, 'core' ), Float32Array.from( kCore( g ).core ) );
	assert.deepEqual( nodeMetric( g, 'betweenness', { normalized: true } ), betweenness( g, { normalized: true } ) );
	assert.deepEqual( nodeMetric( g, 'harmonic' ), closeness( g ).harmonic );
	assert.throws( () => nodeMetric( g, 'nope' ), /unknown metric/ );

} );

// ─── Zachary's karate club ───────────────────────────────────────────────────────────────────

test( 'karate club through the data pipeline: published values and brute force', async () => {

	const loaded = await runRequest( { source: { kind: 'text', text: readFileSync( 'public/data/imsv/karate.json', 'utf8' ), name: 'karate.json' } } );
	const g = loaded.graph; // a PreparedGraph: no adjEdge, the module builds and caches the index
	assert.deepEqual( [ g.N, g.E, g.adjEdge ], [ 34, 78, undefined ] );
	const pairs = allPairs( g );

	// structure
	const summary = graphSummary( g );
	near( summary.density, 0.13903743315508021, 1e-12 );
	near( summary.assortativity, - 0.47561309768461, 1e-10 );
	assert.deepEqual( [ summary.minDegree, summary.maxDegree, summary.isolated ], [ 1, 17, 0 ] );
	assert.deepEqual( [ g.degree[ 0 ], g.degree[ 33 ] ], [ 16, 17 ] ); // the instructor and the administrator
	assert.deepEqual( connectedComponents( g ).sizes, Uint32Array.of( 34 ) );
	const cores = kCore( g );
	assert.equal( cores.degeneracy, 4 );
	assert.deepEqual( Array.from( cores.core ).map( ( c, v ) => ( c === 4 ? v : - 1 ) ).filter( ( v ) => v >= 0 ), [ 0, 1, 2, 3, 7, 8, 13, 30, 32, 33 ] );
	assert.deepEqual( Array.from( cores.core ), bruteCore( g ) );
	const clustering = clusteringCoefficient( g, { edges: true } );
	assert.equal( clustering.triangleCount, 45 );
	near( clustering.global, 0.2556818181818182, 1e-12, 'transitivity' );
	near( clustering.average, 0.5706384782076823, 1e-7, 'average clustering' );
	assert.deepEqual( Array.from( clustering.triangles ), bruteTriangles( g ).node );
	assert.deepEqual( Array.from( clustering.edgeTriangles ), bruteTriangles( g ).edge );

	// distances
	const diameter = estimateDiameter( g );
	assert.deepEqual( [ diameter.diameter, diameter.exact ], [ 5, true ] );
	assert.ok( diameter.bfsCount <= 8, `diameter proven with ${diameter.bfsCount} BFS` );
	const distances = closeness( g );
	assert.equal( distances.exact, true );
	near( distances.averagePathLength, 2.408199643493761, 1e-12 );
	near( distances.closeness[ 0 ], 0.5689655172413793, 1e-6 );
	near( distances.closeness[ 33 ], 0.55, 1e-6 );
	assert.equal( distances.maxDistance, 5 );

	// betweenness: networkx values (normalised), and the pair-counting definition
	const ref = bruteBetweenness( g, pairs );
	const bc = betweenness( g, { normalized: true } );
	near( bc[ 0 ], 0.43763528138528146, 1e-6 );
	near( bc[ 33 ], 0.30407497594997596, 1e-6 );
	near( bc[ 32 ], 0.145247113997114, 1e-6 );
	near( bc[ 2 ], 0.14365680615680618, 1e-6 );
	near( bc[ 31 ], 0.13827561327561325, 1e-6 );
	assertClose( betweenness( g ), ref.node, 1e-6, 'betweenness' );
	const eb = edgeBetweenness( g );
	assertClose( eb, ref.edge, 1e-6, 'edge betweenness' );
	const busiest = eb.indexOf( Math.max( ...eb ) );
	assert.deepEqual( [ g.edges[ 2 * busiest ], g.edges[ 2 * busiest + 1 ] ], [ 0, 31 ] ); // the Girvan-Newman first cut
	near( eb[ busiest ], 71.39285714285712, 1e-6 );

	// spectral: networkx values (computed there to 1e-6, hence the tolerance) and the references
	const pr = pageRank( g, { tolerance: 1e-12, maxIterations: 1000 } );
	near( pr[ 33 ], 0.1009179167487121, 2e-5 );
	near( pr[ 0 ], 0.09700181758983709, 2e-5 );
	assertClose( pr, brutePageRank( g, 0.85 ), 1e-7, 'PageRank' );
	assert.equal( pr.indexOf( Math.max( ...pr ) ), 33 );
	const ev = eigenvectorCentrality( g, { tolerance: 1e-12, maxIterations: 5000 } );
	near( ev[ 33 ], 0.373371213013235, 2e-5 );
	near( ev[ 0 ], 0.3554834941851943, 2e-5 );
	assertClose( ev, bruteEigenvector( g, connectedComponents( g ) ), 1e-6, 'eigenvector centrality' );

	// paths, with edge ids from the cached index
	const p = shortestPath( g, 16, 26 );
	assert.equal( p.length, 5 );
	assertPath( g, p, 16, 26, 5 );
	const dag = shortestPathDag( g, 16, 26 );
	assert.deepEqual( [ dag.length, dag.pathCount ], [ 5, pairs.sigma[ 16 ][ 26 ] ] );
	assert.ok( p.nodes.every( ( v ) => dag.nodes.includes( v ) ) && p.edges.every( ( e ) => dag.edges.includes( e ) ), 'the single path lies in the geodesic DAG' );
	const weighted = shortestPathWeighted( g, 16, 26, loaded.graph.initPos );
	assertPath( g, weighted, 16, 26 );
	assert.ok( weighted.edges.length >= 5 && weighted.length > 0 );
	assert.deepEqual( Array.from( khopNeighborhood( g, 0, 1 ).nodes ).sort( ( a, b ) => a - b ), [ 0, ...neighbours( g, 0 ) ].sort( ( a, b ) => a - b ) );
	const ripple = bfs( g, 0 );
	assert.equal( ripple.maxDist, 3 );
	assert.deepEqual( Array.from( ripple.levelOffsets ), [ 0, 1, 17, 26, 34 ] );

} );

// ─── performance ─────────────────────────────────────────────────────────────────────────────

test( 'performance at 20k nodes / 384k edges', () => {

	// the "LFR · 20k dense" preset (public/data/generated/presets.json)
	const raw = generate( { model: 'lfr', params: { n: 20000, avgDegree: 40, maxDegree: 300, mu: 0.3, degreeExponent: 2.5, communityExponent: 1.5, minCommunity: 30, maxCommunity: 600 }, seed: 1 } );
	const built = graphFromEdges( raw.nodeCount, raw.edges );
	const g = { N: built.N, E: built.E, edges: built.edges, degree: built.degree, adjStart: built.adjStart, adjList: built.adjList }; // as a PreparedGraph: no adjEdge
	assert.equal( g.N, 20000 );
	assert.ok( g.E > 380000, `${g.E} edges` );
	const positions = new Float32Array( 4 * g.N );
	const rand = rng( 1 );
	for ( let i = 0; i < positions.length; i ++ ) positions[ i ] = rand() * 1000;

	// first = first call in this process (includes JIT warm-up), best = fastest of the following
	// runs; `limit` (generous: several times the target, for loaded CI machines) applies to best
	const rows = [];
	const time = ( name, target, limit, fn, repeats = 3 ) => {

		let t0 = performance.now();
		let result = fn();
		const first = performance.now() - t0;
		let best = first;
		for ( let i = 0; i < repeats; i ++ ) {

			t0 = performance.now();
			result = fn();
			best = Math.min( best, performance.now() - t0 );

		}

		rows.push( `  ${name.padEnd( 40 )} first ${first.toFixed( 1 ).padStart( 8 )} ms   best ${best.toFixed( 1 ).padStart( 8 )} ms   target ${String( target ).padStart( 4 )} ms ${best <= target ? 'ok' : 'MISSED'}` );
		assert.ok( best < limit, `${name}: ${best.toFixed( 1 )} ms (limit ${limit} ms)` );
		return result;

	};

	// O( N + E ) algorithms: target 50 ms each
	time( 'graphFromEdges', 50, 400, () => graphFromEdges( g.N, g.edges ) );
	time( 'buildEdgeIndex', 50, 400, () => buildEdgeIndex( g ) );
	const ripple = time( 'bfs', 50, 400, () => bfs( g, 0 ) );
	assert.equal( ripple.reached, g.N );
	time( 'khopNeighborhood ( k = 2 )', 50, 400, () => khopNeighborhood( g, 0, 2 ) );
	const comp = time( 'connectedComponents', 50, 400, () => connectedComponents( g ) );
	assert.equal( comp.count, 1 );
	const cores = time( 'kCore', 50, 400, () => kCore( g ) );
	assert.ok( cores.degeneracy >= 10 );
	time( 'graphSummary', 50, 400, () => graphSummary( g ) );
	time( 'shortestPath x 100', 50, 400, () => {

		for ( let i = 0; i < 100; i ++ ) assert.ok( shortestPath( g, ( i * 7919 ) % g.N, ( i * 104729 + 13 ) % g.N ).length > 0 );

	} );
	time( 'shortestPathWeighted ( A* ) x 10', 50, 400, () => {

		for ( let i = 0; i < 10; i ++ ) assert.ok( shortestPathWeighted( g, ( i * 7919 ) % g.N, ( i * 104729 + 13 ) % g.N, positions ).length > 0 );

	} );
	time( 'shortestPathDag x 10', 50, 400, () => {

		for ( let i = 0; i < 10; i ++ ) assert.ok( shortestPathDag( g, ( i * 7919 ) % g.N, ( i * 104729 + 13 ) % g.N ).pathCount >= 1 );

	} );
	// O( E sqrt E ) worst case, near-linear here: same target
	const clustering = time( 'clusteringCoefficient', 50, 400, () => clusteringCoefficient( g ) );
	assert.ok( clustering.triangleCount > 100000 );
	time( 'clusteringCoefficient ( + edges )', 100, 800, () => clusteringCoefficient( g, { edges: true } ) );
	// iterative
	const stats = {};
	const pr = time( 'pageRank', 150, 1000, () => pageRank( g, { stats } ) );
	assert.ok( stats.converged );
	near( pr.reduce( ( s, x ) => s + x, 0 ), 1, 1e-4 );
	time( 'eigenvectorCentrality', 150, 1500, () => eigenvectorCentrality( g, { stats } ) );
	assert.ok( stats.converged );
	// a few BFS
	const diameter = time( 'estimateDiameter ( 32 BFS )', 150, 1500, () => estimateDiameter( g ) );
	assert.ok( diameter.diameter >= 3 && diameter.diameter <= diameter.diameterUpper );
	time( 'averagePathLength ( 64 pivots )', 150, 1500, () => averagePathLength( g ) );
	// pivot sampling: target 2 s for 64 to 256 pivots
	const b64 = time( 'betweenness ( 64 pivots )', 2000, 6000, () => betweenness( g, { samples: 64 } ), 1 );
	const b256 = time( 'betweenness ( 256 pivots )', 2000, 8000, () => betweenness( g, { samples: 256 } ), 1 );
	assert.equal( defaultSamples( g ), 256 );
	const eb = time( 'edgeBetweenness ( 256 pivots )', 2000, 8000, () => edgeBetweenness( g, { samples: 256 } ), 1 );
	assert.ok( eb.every( ( x ) => x > 0 ) );
	const cl = time( 'closeness ( 256 pivots )', 2000, 8000, () => closeness( g, { samples: 256 } ), 1 );
	assert.ok( cl.closeness.every( ( x ) => x > 0 && x <= 1 ) );
	// The scaling is right at this size too. In a connected graph every pair { s, t } contributes
	// d( s, t ) - 1 to the node values and d( s, t ) to the edge values, so both totals follow
	// from the mean distance, which the closeness run estimates independently.
	const pairCount = g.N * ( g.N - 1 ) / 2;
	for ( const b of [ b64, b256 ] ) near( b.reduce( ( s, x ) => s + x, 0 ) / ( pairCount * ( cl.averagePathLength - 1 ) ), 1, 0.03, 'sum of node betweenness' );
	near( eb.reduce( ( s, x ) => s + x, 0 ) / ( pairCount * cl.averagePathLength ), 1, 0.03, 'sum of edge betweenness' );
	// normalisation of node and edge metrics
	time( 'normalize log ( 20k values )', 50, 400, () => normalize( b256, { mode: 'log' } ) );
	time( 'normalize rank ( 20k values )', 50, 400, () => normalize( b256, { mode: 'rank' } ) );
	time( 'normalize rank ( 384k values )', 100, 800, () => normalize( eb, { mode: 'rank' } ) );
	time( 'normalize robust ( 384k values )', 100, 800, () => normalize( eb, { mode: 'robust' } ) );
	console.log( `  analytics timings, ${g.N} nodes / ${g.E} edges:\n${rows.join( '\n' )}` );

} );
