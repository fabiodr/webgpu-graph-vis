// Metric catalogue of the analysis package: what the analytics worker can compute per node and
// per edge, with the normalisation that suits each metric and a cost class. Shared by the worker
// (dispatch), the client and the lens (labels, scales, UI lists). Pure data, no three.js.
//
//   NODE_METRICS   [ { id, label, scale, cost, description, kind, needs? } ]
//                  the library's catalogue (degree, core, pagerank, eigenvector, betweenness,
//                  closeness, harmonic, clustering, triangles, component) plus
//                  'relevance': personalised PageRank (random walk with restart) from a set of
//                  source nodes, options.sources = [ node, … ]; all 0 without sources.
//   EDGE_METRICS   [ { id, label, scale, cost, description } ]
//                  'triangles'     common neighbours of the two endpoints (0 = bridge-like tie)
//                  'embeddedness'  Jaccard overlap of the two neighbourhoods, 0..1
//                  'betweenness'   share of shortest paths over the edge (the backbone)
//   metricInfo( id, target = 'node' ) -> catalogue entry or null
//
// Fields: scale = suggested normalize() mode ('linear' | 'log' | 'rank' | 'robust');
// cost = 'instant' (one near-linear pass) | 'fast' (iterative, tens of ms at 20k nodes / 384k
// edges) | 'slow' (one BFS per pivot: computed progressively, see worker.js);
// kind = 'sequential' (a quantity: ordered ramp) | 'categorical' (a label: distinct hues).

import { NODE_METRICS as LIBRARY_METRICS } from '../graph/analytics.js';

export const NODE_METRICS = [
	...LIBRARY_METRICS.map( ( m ) => ( { ...m, kind: m.id === 'component' ? 'categorical' : 'sequential' } ) ),
	{
		id: 'relevance', label: 'Relevance', scale: 'log', cost: 'fast', kind: 'sequential', needs: 'sources',
		description: 'Random walk with restart from the selection: how strongly a node is tied to it.',
	},
];

export const EDGE_METRICS = [
	{ id: 'triangles', label: 'Shared neighbours', scale: 'log', cost: 'instant', description: 'Triangles through the edge: 0 for bridges and weak ties, high inside dense groups.' },
	{ id: 'embeddedness', label: 'Embeddedness', scale: 'linear', cost: 'instant', description: 'Overlap of the two neighbourhoods (Jaccard), 0 to 1.' },
	{ id: 'betweenness', label: 'Edge betweenness', scale: 'rank', cost: 'slow', description: 'Share of shortest paths over the edge: the backbone between communities.' },
];

/** Catalogue entry of a metric, or null. */
export function metricInfo( id, target = 'node' ) {

	const list = target === 'edge' ? EDGE_METRICS : NODE_METRICS;
	return list.find( ( m ) => m.id === id ) || null;

}
