// Tour planner: turns a graph and a few metric values into the stops of the automatic first
// visit (overview → the most connected node → the largest communities → the strongest broker →
// overview). Pure logic: no camera, no DOM; Tour.js plays what this returns.
//
//   planTour( graph, { betweenness?, exact?, communities?, maxCommunities = 2, touch = false } ) -> [ stop ]
//        graph         PreparedGraph (N, E, degree, labels, name)
//        betweenness   Float32Array( N ) of raw betweenness (pair counts, as the analytics give
//                      it) or null: without it there is no broker stop
//        exact         whether those values are exact (an estimate is called "about")
//        communities   communityList() of the level to visit (largest first) or null
//        touch         word the last caption for a touch screen
//   stop = { id, kind, title, text, seconds, node?, cluster?, level?, lens, value? }
//        kind     'overview' | 'hub' | 'community' | 'broker' | 'outro'
//        title    short heading of the caption; text = one sentence with one number
//        seconds  how long the stop is held
//        lens     { color, size, glow } metric ids (null = the graph's own look) the stop shows
//        value    the number the caption quotes
//   Stops are only planned when their subject exists: a graph without edges has no hub, a graph
//   without (at least two) communities has no community stops, and the broker stop needs a node
//   with betweenness above zero. The broker caption says so when it is the hub again.
//
//   argmax( values, N ) -> index of the largest value (first on ties), −1 for an empty array
//   betweennessShare( value, N ) -> the value as a share of all pairs of other nodes, 0 … 1

import { formatCount, formatNumber, formatPercent } from './format.js';
import { labelOf } from './communities.js';

const PALETTE = { color: null, size: null, glow: null };

export function argmax( values, N = values ? values.length : 0 ) {

	let best = - 1, bestValue = - Infinity;
	for ( let i = 0; i < N; i ++ ) {

		const v = values[ i ];
		if ( v > bestValue ) {

			bestValue = v;
			best = i;

		}

	}

	return best;

}

export function betweennessShare( value, N ) {

	const pairs = ( N - 1 ) * ( N - 2 ) / 2;
	return pairs > 0 ? Math.min( 1, Math.max( 0, value / pairs ) ) : 0;

}

export function planTour( graph, { betweenness = null, exact = true, communities = null, maxCommunities = 2, touch = false } = {} ) {

	const stops = [];
	if ( ! graph || ! ( graph.N > 0 ) ) return stops;
	const N = graph.N, E = graph.E;
	const groups = communities && communities.length >= 2 ? communities : [];
	const plural = ( n, one, many ) => `${formatCount( n )} ${n === 1 ? one : many}`;

	stops.push( {
		id: 'overview', kind: 'overview', seconds: 5, lens: PALETTE,
		title: graph.name || 'The whole graph',
		text: groups.length
			? `${plural( N, 'node', 'nodes' )} and ${plural( E, 'link', 'links' )} in ${formatCount( groups.length )} communities; similar colours belong together.`
			: `${plural( N, 'node', 'nodes' )} and ${plural( E, 'link', 'links' )}.`,
		value: N,
	} );

	const hub = E > 0 ? argmax( graph.degree, N ) : - 1;
	if ( hub >= 0 && graph.degree[ hub ] > 0 ) {

		const d = graph.degree[ hub ], mean = 2 * E / N;
		const ratio = d / mean;
		stops.push( {
			id: 'hub', kind: 'hub', node: hub, seconds: 6, lens: { color: null, size: 'degree', glow: 'degree' },
			title: labelOf( graph, hub ),
			text: ratio >= 1.5
				? `The most connected node: ${plural( d, 'link', 'links' )}, ${formatNumber( ratio, { digits: 2 } )}× the average.`
				: `The most connected node, with ${plural( d, 'link', 'links' )}.`,
			value: d,
		} );

	}

	const visit = groups.slice( 0, Math.max( 0, maxCommunities ) ).filter( ( c ) => c.size >= 2 );
	visit.forEach( ( c, k ) => {

		const inside = Number.isFinite( c.share ) ? `; ${formatPercent( c.share )} of its links stay inside` : '';
		stops.push( {
			id: `community-${k}`, kind: 'community', cluster: c.cluster, level: c.level, node: c.hub, seconds: 6, lens: PALETTE,
			title: `Around ${c.name}`,
			text: `${k === 0 ? 'The largest community' : 'The next largest community'}: ${plural( c.size, 'node', 'nodes' )}${inside}.`,
			value: c.size,
		} );

	} );

	if ( betweenness && N > 2 ) {

		const broker = argmax( betweenness, N );
		const share = broker >= 0 ? betweennessShare( betweenness[ broker ], N ) : 0;
		if ( broker >= 0 && share > 0 ) {

			const about = exact ? '' : 'about ';
			const pct = formatPercent( share );
			stops.push( {
				id: 'broker', kind: 'broker', node: broker, seconds: 6, lens: { color: 'betweenness', size: 'betweenness', glow: null },
				title: labelOf( graph, broker ),
				text: broker === hub
					? `Also the strongest broker: ${about}${pct} of all shortest paths between other nodes run through it.`
					: `The strongest broker: ${about}${pct} of all shortest paths between other nodes run through it.`,
				value: share,
			} );

		}

	}

	if ( stops.length > 1 ) {

		stops.push( {
			id: 'outro', kind: 'outro', seconds: 4, lens: PALETTE,
			title: 'Your turn',
			text: touch ? 'Tap a node to follow its links; the search and the numbers are at the top.' : 'Click a node to follow its links, press / to search, I for the numbers.',
		} );

	}

	return stops;

}
