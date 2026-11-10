// "Explain this node": two or three plain sentences made from numbers of the node, for the node
// card. Pure logic over the typed arrays of a prepared graph ( degree, adjStart, adjList, anc ).
//
//   nodeFacts( graph, i, { level = usefulLevel( graph ), budget = 4e6 } ) -> {
//        degree,
//        communities   how many communities of that level the node and its neighbours lie in
//                      ( 0 when the graph has no hierarchy ), own = neighbours in the node's own
//        cutOff        nodes that lose their connection to the rest when the node is removed: after
//                      the removal its neighbours fall into pieces, the largest piece is "the rest",
//                      cutOff counts the others, pieces how many of them there are. 0 = the node is
//                      no articulation point; −1 = not computed
//        clustering    share of the possible links among the neighbours that exist ( NaN below two
//                      neighbours or when not computed )
//      }
//        The cut is one breadth-first search over the node's component, O( edges ); it and the
//        clustering are skipped when they would visit more than `budget` adjacency entries.
//   explainNode( graph, i, options ) -> { sentences: [ string ], ...facts }
//        At most three sentences: the communities it links, what its removal would cut off, and
//        how tightly its neighbours are linked to each other when that is remarkably much or
//        little. A node without links gets one sentence.

import { usefulLevel, communityOf } from './communities.js';
import { formatCount, formatPercent } from './format.js';

export function nodeFacts( graph, i, { level = usefulLevel( graph ), budget = 4e6 } = {} ) {

	const facts = { degree: 0, communities: 0, own: 0, cutOff: - 1, pieces: 0, clustering: NaN };
	if ( ! graph || ! ( i >= 0 && i < graph.N ) ) return facts;
	const N = graph.N, degree = graph.degree, start = graph.adjStart, list = graph.adjList;
	const deg = facts.degree = degree[ i ];
	if ( deg === 0 ) return facts;
	const a = start[ i ], b = a + deg;

	if ( level >= 0 && graph.D > 0 && graph.anc ) {

		const mine = communityOf( graph, i, level );
		const seen = new Set( [ mine ] );
		for ( let k = a; k < b; k ++ ) {

			const c = communityOf( graph, list[ k ], level );
			if ( c === mine ) facts.own ++;
			else seen.add( c );

		}

		facts.communities = seen.size;

	}

	// without the node: the pieces its neighbours fall into
	if ( list.length <= budget ) {

		const mark = new Uint8Array( N ), queue = new Uint32Array( N );
		mark[ i ] = 1;
		let reached = 0, largest = 0, pieces = 0;
		for ( let k = a; k < b; k ++ ) {

			const s = list[ k ];
			if ( mark[ s ] ) continue;
			let head = 0, tail = 0;
			queue[ tail ++ ] = s;
			mark[ s ] = 1;
			while ( head < tail ) {

				const u = queue[ head ++ ];
				for ( let e = start[ u ], e1 = e + degree[ u ]; e < e1; e ++ ) {

					const v = list[ e ];
					if ( mark[ v ] ) continue;
					mark[ v ] = 1;
					queue[ tail ++ ] = v;

				}

			}

			pieces ++;
			reached += tail;
			if ( tail > largest ) largest = tail;

		}

		facts.cutOff = reached - largest;
		facts.pieces = pieces - 1;

	}

	// links among the neighbours
	if ( deg >= 2 ) {

		let work = 0;
		for ( let k = a; k < b && work <= budget; k ++ ) work += degree[ list[ k ] ];
		if ( work <= budget ) {

			const nb = new Set();
			for ( let k = a; k < b; k ++ ) if ( list[ k ] !== i ) nb.add( list[ k ] );
			let twice = 0;
			for ( const u of nb ) for ( let e = start[ u ], e1 = e + degree[ u ]; e < e1; e ++ ) if ( list[ e ] !== u && nb.has( list[ e ] ) ) twice ++;
			const n = nb.size;
			facts.clustering = n >= 2 ? Math.min( 1, twice / ( n * ( n - 1 ) ) ) : NaN;

		}

	}

	return facts;

}

export function explainNode( graph, i, options ) {

	const f = nodeFacts( graph, i, options );
	const sentences = [];
	const n = formatCount, deg = f.degree;
	if ( ! graph || ! ( i >= 0 && i < graph.N ) ) return { sentences, ...f };
	if ( deg === 0 ) return { sentences: [ 'No links: this node stands alone.' ], ...f };

	if ( f.communities === 1 ) sentences.push( deg === 1 ? 'Its one neighbour is in its own community.' : `All ${n( deg )} neighbours are in its own community.` );
	else if ( f.communities > 1 ) {

		if ( deg === 1 ) sentences.push( 'Its one neighbour belongs to another community.' );
		else {

			const others = f.communities - 1 === 1 ? 'another' : `${n( f.communities - 1 )} others`;
			const own = f.own === 0 ? 'none' : n( f.own );
			sentences.push( `Connects ${n( f.communities )} communities: ${own} of its ${n( deg )} neighbours are in its own, ${n( deg - f.own )} in ${others}.` );

		}

	}

	if ( f.cutOff > 0 ) sentences.push( `Removing it would cut ${n( f.cutOff )} ${f.cutOff === 1 ? 'node' : 'nodes'} off from the rest${f.pieces > 1 ? `, in ${n( f.pieces )} pieces` : ''}.` );
	else if ( f.cutOff === 0 && deg >= 2 ) sentences.push( 'Removing it would cut nothing off: its neighbours stay connected without it.' );

	const c = f.clustering;
	if ( Number.isFinite( c ) ) {

		if ( c === 0 ) sentences.push( 'None of its neighbours are linked to each other.' );
		else if ( c <= 0.15 ) sentences.push( `Few of its neighbours are linked to each other: ${formatPercent( c )} of the possible links between them exist.` );
		else if ( c >= 0.6 ) sentences.push( c === 1 ? 'Its neighbours are all linked to each other.' : `Its neighbours form a tight group: ${formatPercent( c )} of the possible links between them exist.` );

	}

	return { sentences, ...f };

}
