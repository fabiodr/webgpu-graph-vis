// State of the explore layer as plain JSON, for share links: what explore.state() returns and
// explore.restore() accepts. Pure logic: normalizeState() turns anything (an old link, a hand-
// edited one, a state of another graph) into a state that is safe to apply.
//
//   STATE_VERSION
//   normalizeState( raw, { nodes = Infinity, clusters = Infinity, metrics?, edgeMetrics?, ramps? } ) -> state
//   state = {
//     v: 1,
//     lens: { color, size, glow,          metric ids or null (null = the graph's own look)
//             ramp, reverse, top,         colour ramp id, reversed?, share of nodes that glow (0 … 1]
//             edge, edgeOpacity, edgeWidth },   edge metric id or null and what it drives
//     selection: node index | null,
//     hops: 0 … 3,                        k-hop neighbourhood of the selection (0 = off)
//     path: { from, to, all } | null,     shortest path end points (node indices), every geodesic?
//     cluster: cluster index | null,      focused community (prepared-graph cluster space)
//     insights: boolean,                  drawer open?
//     top: metric id,                     ranking shown under "Top nodes"
//   }
//   Unknown metric / ramp ids become null / the default, indices outside the graph are dropped, a
//   path whose ends coincide or are missing is dropped, hops without a selection are 0.

export const STATE_VERSION = 1;
const TOP_METRICS = [ 'degree', 'pagerank', 'betweenness', 'closeness', 'eigenvector' ];

export function defaultState() {

	return {
		v: STATE_VERSION,
		lens: { color: null, size: null, glow: null, ramp: 'viridis', reverse: false, top: 0.05, edge: null, edgeOpacity: true, edgeWidth: true },
		selection: null, hops: 0, path: null, cluster: null, insights: false, top: 'degree',
	};

}

export function normalizeState( raw, { nodes = Infinity, clusters = Infinity, metrics = null, edgeMetrics = null, ramps = null } = {} ) {

	const out = defaultState();
	if ( ! raw || typeof raw !== 'object' ) return out;
	const index = ( v, limit ) => ( Number.isInteger( v ) && v >= 0 && v < limit ? v : null );
	const id = ( v, known ) => ( typeof v === 'string' && v && ( ! known || known.includes( v ) ) ? v : null );
	const L = raw.lens && typeof raw.lens === 'object' ? raw.lens : {};
	out.lens.color = id( L.color, metrics );
	out.lens.size = id( L.size, metrics );
	out.lens.glow = id( L.glow, metrics );
	out.lens.ramp = id( L.ramp, ramps ) || out.lens.ramp;
	out.lens.reverse = L.reverse === true;
	if ( typeof L.top === 'number' && L.top > 0 && L.top <= 1 ) out.lens.top = L.top;
	out.lens.edge = id( L.edge, edgeMetrics );
	out.lens.edgeOpacity = L.edgeOpacity !== false;
	out.lens.edgeWidth = L.edgeWidth !== false;
	out.selection = index( raw.selection, nodes );
	out.hops = out.selection !== null && Number.isInteger( raw.hops ) ? Math.max( 0, Math.min( 3, raw.hops ) ) : 0;
	const P = raw.path && typeof raw.path === 'object' ? raw.path : null;
	if ( P ) {

		const from = index( P.from, nodes ), to = index( P.to, nodes );
		if ( from !== null && to !== null && from !== to ) out.path = { from, to, all: P.all === true };

	}

	out.cluster = index( raw.cluster, clusters );
	out.insights = raw.insights === true;
	out.top = TOP_METRICS.includes( raw.top ) ? raw.top : out.top;
	return out;

}
