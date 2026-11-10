// Plain-language observations about a graph, derived from the staged summary of the analytics
// worker ( lens.summary() ). Each observation states a number the summary contains and is only
// made when that number supports it; a summary that is still arriving simply yields fewer of
// them. Pure logic, no DOM.
//
//   observe( summary, { limit = 2 } ) -> [ { id, text, strength } ]   strongest first
//
// Rules (thresholds are deliberately conservative: saying nothing is better than a weak claim):
//   giant        more than one component and the largest clearly dominates: it holds at least
//                80 % of the nodes, or more than half while no other component reaches a tenth
//                of its size ( components.sizes, when the summary has them )
//   split        more than one component, the largest holds at least half of the nodes but another
//                one is of comparable size: the count and the share, no "giant", no "small"
//   fragmented   more than one component and the largest holds less than half
//   small-world  at least 30 nodes in the largest component, the average distance (exact or
//                sampled) is at most 1.5 × ln N / ln k̄ + 1 (what a random graph of that density
//                has) AND the average clustering is at least 0.1 and five times the k̄ / N of a
//                random graph: short paths together with local clustering (Watts and Strogatz)
//   hubs         the largest degree is at least six times the mean (and at least 12)
//   assortative  degree assortativity ≥ 0.2: well-connected nodes link to each other
//   disassortative   ≤ −0.2: hubs link mostly to low-degree nodes
//   dense        density ≥ 0.25
//   tree         one component with exactly N − 1 edges
//   core         the innermost k-core has k ≥ 4 and at most a fifth of the nodes
//   uniform      every node has the same degree
// "strength" orders them: structure first (components), then distances, then degrees.

import { formatNumber, formatCount, formatPercent } from './format.js';

export function observe( summary, { limit = 2 } = {} ) {

	const out = [];
	if ( ! summary || ! ( summary.nodes > 0 ) ) return out;
	const N = summary.nodes, E = summary.edges;
	const add = ( id, strength, text ) => out.push( { id, strength, text } );
	const comp = summary.components, deg = summary.degree, cl = summary.clustering, dist = summary.distance;

	if ( comp && comp.count > 1 ) {

		const others = comp.count - 1;
		// the second largest component, when the summary lists the sizes (largest first)
		const second = comp.sizes && comp.sizes.length > 1 && comp.largest > 0 ? comp.sizes[ 1 ] / comp.largest : NaN;
		const dominant = comp.largestFraction >= 0.8 || ( comp.largestFraction > 0.5 && second <= 0.1 );
		if ( dominant ) {

			add( 'giant', 90 + 10 * ( 1 - comp.largestFraction ),
				`One giant component holds ${formatPercent( comp.largestFraction )} of the nodes; ${formatCount( others )} small ${others === 1 ? 'piece is' : 'pieces are'} cut off from it.` );

		} else if ( comp.largestFraction >= 0.5 ) {

			add( 'split', 92,
				`${formatCount( comp.count )} separate components; the largest holds ${formatPercent( comp.largestFraction )} of the nodes.` );

		} else {

			add( 'fragmented', 95,
				`Fragmented: ${formatCount( comp.count )} separate components, the largest with ${formatPercent( comp.largestFraction )} of the nodes.` );

		}

	} else if ( comp && comp.count === 1 && E === N - 1 && N > 2 ) {

		add( 'tree', 85, `A tree: ${formatCount( N )} nodes joined by exactly ${formatCount( E )} links, with no cycle.` );

	}

	if ( dist && deg && cl && comp && Number.isFinite( dist.averagePathLength ) && comp.largest >= 30 && deg.mean > 1.2 ) {

		const n = comp.largest;
		const randomDistance = Math.log( n ) / Math.log( deg.mean );
		const randomClustering = deg.mean / n;
		if ( dist.averagePathLength <= 1.5 * randomDistance + 1 && cl.average >= 0.1 && cl.average >= 5 * randomClustering ) {

			add( 'small-world', 80,
				`Small world: average distance ${dist.exact ? '' : 'about '}${formatNumber( dist.averagePathLength, { digits: 2 } )} across ${formatCount( n )} nodes, with clustering ${formatNumber( cl.average, { digits: 2 } )}.` );

		}

	}

	if ( deg && deg.mean > 0 ) {

		if ( deg.min === deg.max && N > 2 ) {

			add( 'uniform', 60, `Regular: every node has exactly ${formatCount( deg.max )} links.` );

		} else if ( deg.max >= 12 && deg.max >= 6 * deg.mean ) {

			add( 'hubs', 70 + Math.min( 9, deg.max / deg.mean / 10 ),
				`Hub-dominated: the best-connected node has ${formatCount( deg.max )} links, ${formatNumber( deg.max / deg.mean, { digits: 2 } )}× the average of ${formatNumber( deg.mean, { digits: 2 } )}.` );

		}

	}

	if ( Number.isFinite( summary.assortativity ) && E >= 30 ) {

		const r = summary.assortativity;
		if ( r >= 0.2 ) add( 'assortative', 50 + 10 * Math.min( 1, r ), `Well-connected nodes link to each other (degree assortativity ${formatNumber( r, { digits: 2 } )}).` );
		else if ( r <= - 0.2 ) add( 'disassortative', 50 + 10 * Math.min( 1, - r ), `Hubs link mostly to low-degree nodes (degree assortativity ${formatNumber( r, { digits: 2 } )}).` );

	}

	if ( Number.isFinite( summary.density ) && summary.density >= 0.25 && N >= 8 ) {

		add( 'dense', 55, `Dense: ${formatPercent( summary.density )} of all possible links exist.` );

	}

	if ( summary.core && summary.core.sizes && summary.core.degeneracy >= 4 ) {

		const k = summary.core.degeneracy, inner = summary.core.sizes[ k ];
		if ( inner > 0 && inner <= N / 5 ) {

			add( 'core', 40, `A dense core: ${formatCount( inner )} nodes, each linked to at least ${formatCount( k )} others of the core.` );

		}

	}

	out.sort( ( a, b ) => b.strength - a.strength );
	return out.slice( 0, Math.max( 0, limit ) );

}
