// Fast synthetic clustered graph with a planted hierarchy, for dev pages and scale tests.
// (The real generators with Louvain clustering live in src/data/generators.js.)
//
//   synthGraph({ n: 20000, avgDegree: 20, branching: [8, 8], seed: 1 })
//
// Nodes are split into prod(branching) leaf blocks; each node draws avgDegree/2 partners: within
// its leaf block with probability pIn, within its level-1 block with probability pMid, otherwise
// uniformly. The planted hierarchy is returned in canonical form (levels[0] finest).
import { normalizeGraph } from '../data/format.js';
import { rng } from '../graph/prepare.js';

export function synthGraph( { n = 2000, avgDegree = 10, branching = [ 6, 6 ], pIn = 0.75, pMid = 0.18, seed = 1, name } = {} ) {

	const rand = rng( seed );
	const leaves = branching.reduce( ( a, b ) => a * b, 1 );
	const leafOf = ( i ) => Math.floor( ( i * leaves ) / n );
	const blockStart = ( b, size ) => Math.ceil( ( b * n ) / size );
	const edges = [];
	const half = Math.max( 1, Math.round( avgDegree / 2 ) );
	const midSize = leaves / branching[ branching.length - 1 ]; // leaves per level-1 block (2 levels)
	for ( let i = 0; i < n; i ++ ) {

		const leaf = leafOf( i );
		for ( let k = 0; k < half; k ++ ) {

			const r = rand();
			let j;
			if ( r < pIn ) {

				const a = blockStart( leaf, leaves ), b = blockStart( leaf + 1, leaves );
				j = a + Math.floor( rand() * ( b - a ) );

			} else if ( r < pIn + pMid && branching.length > 1 ) {

				const mid = Math.floor( leaf / branching[ 0 ] );
				const a = blockStart( mid * branching[ 0 ], leaves ), b = blockStart( Math.min( leaves, ( mid + 1 ) * branching[ 0 ] ), leaves );
				j = a + Math.floor( rand() * ( b - a ) );

			} else {

				j = Math.floor( rand() * n );

			}

			edges.push( i, j );

		}

	}

	const levels = [];
	let size = 1;
	for ( let l = 0; l < branching.length; l ++ ) {

		const lvl = new Array( n );
		for ( let i = 0; i < n; i ++ ) lvl[ i ] = Math.floor( leafOf( i ) / size );
		levels.push( lvl );
		size *= branching[ l ];

	}

	void midSize;
	return normalizeGraph( {
		name: name || `synthetic ${n} nodes`,
		nodeCount: n,
		edges,
		hierarchy: { method: 'planted', levels },
	} );

}
