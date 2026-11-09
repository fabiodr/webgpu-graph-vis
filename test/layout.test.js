// Tests for the CPU side of the force layout (src/gpu/forceKernels.js): the extended Barnes–Hut
// structure (virtual leaf splitting), the resort, and the CPU mirror of the GPU repulsion walk.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';
import { synthGraph } from '../src/debug/synth.js';
import { buildStructure, resortStructure, aggregateCPU, repulsionCPU } from '../src/gpu/forceKernels.js';

const load = ( id ) => prepareGraph( normalizeGraph( JSON.parse( readFileSync( `public/data/imsv/${id}.json`, 'utf8' ) ) ) );

function bruteForce( pos, N, charge = - 30, dmin = 1 ) {

	const f = new Float64Array( 3 * N );
	for ( let i = 0; i < N; i ++ ) for ( let j = 0; j < N; j ++ ) {

		if ( i === j ) continue;
		const dx = pos[ 4 * j ] - pos[ 4 * i ], dy = pos[ 4 * j + 1 ] - pos[ 4 * i + 1 ], dz = pos[ 4 * j + 2 ] - pos[ 4 * i + 2 ];
		let l2 = dx * dx + dy * dy + dz * dz;
		if ( l2 < 1e-12 ) continue;
		if ( l2 < dmin * dmin ) l2 = Math.sqrt( dmin * dmin * l2 );
		f[ 3 * i ] += ( dx * charge ) / l2; f[ 3 * i + 1 ] += ( dy * charge ) / l2; f[ 3 * i + 2 ] += ( dz * charge ) / l2;

	}

	return f;

}

function checkStructure( g, s ) {

	const { N, C, V, TT, tree, nodeOrder, vInfo, vChildList } = s;
	// nodeOrder is a permutation
	const seen = new Uint8Array( N );
	for ( let t = 0; t < N; t ++ ) seen[ nodeOrder[ t ] ] ++;
	assert.ok( seen.every( ( x ) => x === 1 ), 'nodeOrder is a permutation' );
	// tree: preorder, skips in range, slots appear in order 0..N-1, masses = number of slots in subtree
	let nextSlot = 0;
	for ( let k = 0; k < TT; k ++ ) {

		const ref = tree[ 4 * k ], skip = tree[ 4 * k + 1 ], mass = tree[ 4 * k + 2 ];
		assert.ok( skip > k && skip <= TT );
		if ( ref < N ) {

			assert.equal( ref, nextSlot ++ );
			assert.equal( skip, k + 1 );

		} else {

			assert.ok( ref - N < C + V );
			let slots = 0;
			for ( let j = k + 1; j < skip; j ++ ) if ( tree[ 4 * j ] < N ) slots ++;
			assert.equal( slots, mass, 'cluster mass = slots in subtree' );
			if ( ref - N < C ) assert.equal( mass, g.clusterMass[ ref - N ] );

		}

	}

	assert.equal( nextSlot, N );
	// virtual clusters sorted by height, children of lower height
	for ( let v = 1; v < V; v ++ ) assert.ok( vInfo[ 4 * v + 2 ] >= vInfo[ 4 * ( v - 1 ) + 2 ] );
	for ( let v = 0; v < V; v ++ ) {

		const h = vInfo[ 4 * v + 2 ];
		if ( h === 1 ) assert.ok( vInfo[ 4 * v + 1 ] <= s.leafSize );
		else for ( let k = 0; k < vInfo[ 4 * v + 1 ]; k ++ ) {

			const ch = vChildList[ vInfo[ 4 * v ] + k ] - C;
			assert.ok( vInfo[ 4 * ch + 2 ] < h );

		}

	}

}

for ( const id of [ 'karate', 'lesmis', 'football', 'celegansneural' ] ) {

	test( `structure ${id}`, () => {

		const g = load( id );
		for ( const leafSize of [ 4, 32 ] ) {

			const s = buildStructure( g, { leafSize } );
			checkStructure( g, s );
			resortStructure( s, g.initPos );
			checkStructure( g, s );

		}

	} );

}

test( 'theta = 0 equals brute force; theta = 0.8 is close (synthetic 1500, D = 2)', () => {

	const g = prepareGraph( synthGraph( { n: 1500, avgDegree: 12, branching: [ 3, 3 ] } ) ); // ~167-node leaves -> split
	const s = buildStructure( g, { leafSize: 32 } );
	assert.ok( s.V > 0 );
	resortStructure( s, g.initPos );
	checkStructure( g, s );
	const bh = aggregateCPU( g, s, g.initPos );
	// conservative radii: every node lies inside all its tree ancestors' bounding spheres
	{

		const stack = [];
		for ( let k = 0; k < s.TT; k ++ ) {

			while ( stack.length && stack[ stack.length - 1 ].skip <= k ) stack.pop();
			const ref = s.tree[ 4 * k ];
			if ( ref < s.N ) {

				const node = s.nodeOrder[ ref ];
				for ( const a of stack ) {

					const o = 4 * a.id;
					const d = Math.hypot( g.initPos[ 4 * node ] - bh[ o ], g.initPos[ 4 * node + 1 ] - bh[ o + 1 ], g.initPos[ 4 * node + 2 ] - bh[ o + 2 ] );
					assert.ok( d <= bh[ o + 3 ] * ( 1 + 1e-5 ) + 1e-4, 'radius is conservative' );

				}

			} else stack.push( { id: ref - s.N, skip: s.tree[ 4 * k + 1 ] } );

		}

	}

	const ref = bruteForce( g.initPos, g.N );
	const exact = repulsionCPU( s, g.initPos, bh, { theta: 0, maxIterations: 1 << 20 } ).force;
	let maxRel = 0;
	for ( let i = 0; i < g.N; i ++ ) {

		const e = Math.hypot( exact[ 4 * i ] - ref[ 3 * i ], exact[ 4 * i + 1 ] - ref[ 3 * i + 1 ], exact[ 4 * i + 2 ] - ref[ 3 * i + 2 ] );
		maxRel = Math.max( maxRel, e / Math.hypot( ref[ 3 * i ], ref[ 3 * i + 1 ], ref[ 3 * i + 2 ] ) );

	}

	assert.ok( maxRel < 1e-4, `theta 0 exact (max rel ${maxRel})` );
	const approx = repulsionCPU( s, g.initPos, bh, { theta: 0.8 } );
	const rel = [];
	for ( let i = 0; i < g.N; i ++ ) {

		const e = Math.hypot( approx.force[ 4 * i ] - ref[ 3 * i ], approx.force[ 4 * i + 1 ] - ref[ 3 * i + 1 ], approx.force[ 4 * i + 2 ] - ref[ 3 * i + 2 ] );
		rel.push( e / Math.hypot( ref[ 3 * i ], ref[ 3 * i + 1 ], ref[ 3 * i + 2 ] ) );

	}

	rel.sort( ( a, b ) => a - b );
	const visits = approx.visits.reduce( ( a, b ) => a + b, 0 ) / g.N;
	assert.ok( rel[ Math.floor( 0.5 * g.N ) ] < 0.05, `median rel err ${rel[ Math.floor( 0.5 * g.N ) ]}` );
	assert.ok( visits < g.N * 0.6, `visits/node ${visits}` );

} );

test( 'D = 0 graph gets a spatial virtual tree; tiny graphs work', () => {

	const raw = synthGraph( { n: 600, avgDegree: 6, branching: [ 1 ] } );
	const g = prepareGraph( { ...raw, hierarchy: null } );
	assert.equal( g.D, 0 );
	const s = buildStructure( g, { leafSize: 16 } );
	resortStructure( s, g.initPos );
	checkStructure( g, s );
	assert.ok( s.V >= 600 / 16 );
	const bh = aggregateCPU( g, s, g.initPos );
	const exact = repulsionCPU( s, g.initPos, bh, { theta: 0, maxIterations: 1 << 20 } ).force;
	const ref = bruteForce( g.initPos, g.N );
	for ( let i = 0; i < g.N; i += 37 ) assert.ok( Math.abs( exact[ 4 * i ] - ref[ 3 * i ] ) <= 1e-3 * ( 1 + Math.abs( ref[ 3 * i ] ) ) );

	const one = prepareGraph( normalizeGraph( { nodeCount: 1, edges: [] } ) );
	const s1 = buildStructure( one );
	checkStructure( one, s1 );
	const f1 = repulsionCPU( s1, one.initPos, aggregateCPU( one, s1, one.initPos ) ).force;
	assert.equal( f1[ 0 ], 0 );

} );
