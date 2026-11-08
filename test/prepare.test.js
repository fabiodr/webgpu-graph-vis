import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';

const files = readdirSync( 'public/data/imsv' ).map( ( f ) => 'public/data/imsv/' + f );

for ( const f of files ) {

	test( `prepare ${f}`, () => {

		const g = normalizeGraph( JSON.parse( readFileSync( f, 'utf8' ) ) );
		const p = prepareGraph( g );
		const { N, C, D, T, root, bhTree } = p;
		assert.equal( p.clusterHeight[ root ], D + 1 );
		assert.equal( p.clusterMass[ root ], N );
		// BH tree: preorder, skip pointers within range, each node appears once
		const seen = new Uint8Array( N );
		for ( let i = 0; i < T; i ++ ) {

			const ref = bhTree[ 2 * i ], skip = bhTree[ 2 * i + 1 ];
			assert.ok( skip > i && skip <= T );
			if ( ref < N ) {

				assert.equal( skip, i + 1 ); seen[ ref ] ++;

			} else {

				// subtree size equals 1 + descendants (clusters + nodes)
				const c = ref - N;
				let nodes = 0;
				for ( let k = i + 1; k < skip; k ++ ) if ( bhTree[ 2 * k ] < N ) nodes ++;
				assert.equal( nodes, p.clusterMass[ c ] );

			}

		}

		assert.ok( seen.every( ( v ) => v === 1 ) );
		// children CSR: masses add up
		for ( let c = 0; c < C; c ++ ) {

			let m = 0;
			for ( let k = 0; k < p.clusterChildCount[ c ]; k ++ ) {

				const ch = p.childList[ p.clusterChildStart[ c ] + k ];
				m += p.clusterHeight[ c ] === 1 || ( D === 0 && c === root ) ? 1 : p.clusterMass[ ch ];

			}

			assert.equal( m, p.clusterMass[ c ] );

		}

		// sphere directions are unit vectors
		for ( let i = 0; i < N; i ++ ) {

			const l = Math.hypot( p.sphereDir[ 4 * i ], p.sphereDir[ 4 * i + 1 ], p.sphereDir[ 4 * i + 2 ] );
			assert.ok( Math.abs( l - 1 ) < 1e-4, `unit ${l}` );

		}

		assert.ok( p.initPos.every( Number.isFinite ) );
		assert.ok( p.colors.every( ( v ) => v >= 0 && v <= 1 ) );

	} );

}

test( 'graph without hierarchy', () => {

	const g = normalizeGraph( { nodeCount: 5, edges: [ 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 1, 0 ] } );
	assert.equal( g.edges.length, 10 );
	const p = prepareGraph( g );
	assert.equal( p.D, 0 );
	assert.equal( p.C, 1 );
	assert.equal( p.T, 5 );
	assert.equal( p.maxPath, 2 );

} );
