// Communities of a prepared graph for the insights drawer and the tour: which level of the
// clustering hierarchy to show, and what to say about each community. Pure logic over the typed
// arrays of src/graph/prepare.js (anc, levelOffset, clusterMass, clusterRankRange, edges, degree).
//
//   usefulLevel( graph, { min = 2, max = 40 } ) -> level | −1
//        The level to show (D − 1 = coarsest, 0 = finest): graph.clustering.mainLevel where the data
//        layer names one that splits the graph, else the coarsest level that has at least `min`
//        communities (when every level has more than `max`, the coarsest one). −1 when the graph
//        has no hierarchy or no level splits it.
//   communitySummary( graph ) -> { level, count, finest }
//        THE number of communities, for every surface that says one (panel, legend, drawer,
//        status): graph.clustering.communities of the data layer where it is given (the clusters
//        of the main level without the dust clusters), else the clusters of usefulLevel, else
//        graph.stats.clusters. `finest` is the cluster count of level 0.
//   communityList( graph, level ) -> [ { cluster, level, size, hub, hubDegree, name, internal, cut,
//        share, rank } ], largest first
//        cluster   index in the prepared graph's cluster space (what 'cluster' events carry)
//        hub       the member with the highest degree; name = its label ( "Node 12" without labels )
//        internal  edges with both ends inside, cut = edges with one end inside
//        share     internal / ( internal + cut ): the share of the community's edges that stay
//                  inside (NaN for a community without edges)
//        rank      position 0 … 1 in the leaf order, the palette coordinate of its colour
//   communityCount( graph, level ) -> the number of clusters of that level (0 = finest)
//   labelOf( graph, i ) -> the node's label or "Node i"
//   communityOf( graph, i, level ) -> cluster index of node i at that level

export function labelOf( graph, i ) {

	const l = graph && graph.labels ? graph.labels[ i ] : null;
	return l !== undefined && l !== null && l !== '' ? String( l ) : `Node ${i}`;

}

export function communityOf( graph, i, level ) {

	return graph.anc[ i * graph.D + level ];

}

function countAt( graph, level ) {

	const lo = graph.levelOffset[ level ];
	const hi = level + 1 < graph.D ? graph.levelOffset[ level + 1 ] : graph.root;
	return hi - lo;

}

export function communityCount( graph, level ) {

	return graph && level >= 0 && level < graph.D ? countAt( graph, level ) : 0;

}

export function usefulLevel( graph, { min = 2, max = 40 } = {} ) {

	if ( ! graph || ! graph.D ) return - 1;
	const main = graph.clustering ? graph.clustering.mainLevel : undefined;
	if ( Number.isInteger( main ) && main >= 0 && main < graph.D && graph.levelOffset && countAt( graph, main ) >= min ) return main;
	let fallback = - 1;
	for ( let level = graph.D - 1; level >= 0; level -- ) {

		const k = countAt( graph, level );
		if ( k < min ) continue;
		if ( k <= max ) return level;
		if ( fallback < 0 ) fallback = level;

	}

	return fallback;

}

export function communitySummary( graph ) {

	if ( ! graph ) return { level: - 1, count: 0, finest: 0 };
	const c = graph.clustering || null;
	const level = graph.levelOffset ? usefulLevel( graph ) : - 1;
	const given = c && Number.isFinite( c.communities ) && c.communities >= 0 && ( level < 0 || level === c.mainLevel );
	const count = given ? c.communities : level >= 0 ? countAt( graph, level ) : graph.levelOffset ? 0 : ( graph.stats && graph.stats.clusters ) | 0;
	const finest = graph.levelOffset && graph.D ? countAt( graph, 0 ) : c && Array.isArray( c.clustersPerLevel ) && c.clustersPerLevel.length ? c.clustersPerLevel[ 0 ] | 0 : count;
	return { level, count, finest };

}

export function communityList( graph, level ) {

	if ( ! graph || level < 0 || level >= graph.D ) return [];
	const { N, E, D, anc, edges, degree } = graph;
	const lo = graph.levelOffset[ level ], K = countAt( graph, level );
	const size = new Uint32Array( K ), hub = new Int32Array( K ).fill( - 1 ), internal = new Uint32Array( K ), cut = new Uint32Array( K );
	for ( let i = 0; i < N; i ++ ) {

		const c = anc[ i * D + level ] - lo;
		size[ c ] ++;
		if ( hub[ c ] < 0 || degree[ i ] > degree[ hub[ c ] ] ) hub[ c ] = i;

	}

	for ( let e = 0; e < E; e ++ ) {

		const a = anc[ edges[ 2 * e ] * D + level ] - lo, b = anc[ edges[ 2 * e + 1 ] * D + level ] - lo;
		if ( a === b ) internal[ a ] ++;
		else {

			cut[ a ] ++;
			cut[ b ] ++;

		}

	}

	const out = [];
	for ( let c = 0; c < K; c ++ ) {

		if ( size[ c ] === 0 ) continue;
		const cluster = lo + c, touching = internal[ c ] + cut[ c ];
		const range = graph.clusterRankRange;
		out.push( {
			cluster, level, size: size[ c ], hub: hub[ c ], hubDegree: degree[ hub[ c ] ], name: labelOf( graph, hub[ c ] ),
			internal: internal[ c ], cut: cut[ c ], share: touching > 0 ? internal[ c ] / touching : NaN,
			rank: range ? ( range[ 2 * cluster ] + range[ 2 * cluster + 1 ] ) / 2 : ( c + 0.5 ) / K,
		} );

	}

	out.sort( ( a, b ) => b.size - a.size || b.hubDegree - a.hubDegree || a.cluster - b.cluster );
	return out;

}
