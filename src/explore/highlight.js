// The highlight model: one answer to "which nodes stand out right now". Selection, community
// focus, k-hop neighbourhood and shortest path are layers of one stack; hover and the search
// preview ride on top. The Interaction module owns an instance and uploads what compose()
// returns to nodeStateTarget ( x selected, y neighbour, z dimmed, w hovered ), which the GPU
// eases toward as before. Pure logic, no three.js, no DOM.
//
// ─── Layers, bottom to top ──────────────────────────────────────────────────────────────────
//   selection   the selected node and its neighbours
//   cluster     the members of a focused community
//   khop        the k-hop neighbourhood of the tools ( tools.highlightSet( 'khop' ) )
//   path        the shortest path of the tools ( tools.highlightSet( 'path' ) )
//  A layer is a weight per node in the convention of Tools.highlightSet(): 1 = primary,
//  0.35 … 0.85 = secondary, 0 = not part of it.
//
// ─── Precedence ─────────────────────────────────────────────────────────────────────────────
//  · As soon as one layer is active everything outside the picture is dimmed ( z = 1 ).
//  · khop and path are exclusive: the top-most of them alone decides the picture, the layers
//    under it are kept and come back when it is peeled. A neighbourhood is a superset of the
//    selection's own neighbours; a path is read best alone.
//  · selection and cluster add up: the members of the focused community stay lit ( a calm
//    y = 0.5 ) while the selected node and its neighbours stand out inside or outside it.
//  · The selected node is never dimmed, whatever is on top.
//  · In a path the two ends are marked with x = 0.4, not 1: the renderers light every edge of a
//    node with x = 1 ( the star of a selection ), which would bury the path under two stars.
//    The selected node on a path is drawn the same way for the same reason.
//  · In a neighbourhood the rings fade: y from 1 (first ring) to 0.45 (last ring), z from 0 to
//    0.35, so the hop distance stays readable without leaving the focus.
//  · hover and preview only write w: the shaders bring a hovered node out of the dimming
//    themselves.
//  · Esc peels one layer, top-most first: path, khop, cluster, selection ( peel() ).
//
// ─── API ────────────────────────────────────────────────────────────────────────────────────
//  const model = new HighlightModel();
//  model.reset( N )                       new graph: no layers, no hover
//  model.set( name, weights | null, meta? )   weights: Float32Array( N ); meta is kept for the UI
//  model.get( name ) -> { weights, meta } | null,  model.has( name )
//  model.selected = index | −1            the selected node (kept in front of every layer)
//  model.hover / model.preview            node indices or −1
//  model.active                           true when any layer is set
//  model.top() -> name | null             the layer Esc removes next
//  model.visible() -> name[]              the layers that decide the picture, bottom to top
//  model.compose( out? ) -> Float32Array( 4 N )
//  selectionWeights( graph, index, out? ), clusterWeights( graph, cluster, out? )
//  clusterMembers( graph, cluster ) -> Uint32Array of node indices
//  LAYERS, PEEL_ORDER, PATH_END

export const LAYERS = [ 'selection', 'cluster', 'khop', 'path' ];
export const PEEL_ORDER = [ 'path', 'khop', 'cluster', 'selection' ];
const EXCLUSIVE = [ 'path', 'khop' ]; // top-most first

/** x of the ends of a path (and of the selected node on it): marked, without lighting their stars. */
export const PATH_END = 0.4;
const CLUSTER_GLOW = 0.5;
const SECONDARY = 0.85; // weight of the first ring / a direct neighbour, see Tools.highlightSet()
const RING_SPAN = 0.5; // … down to 0.35 at the last ring

export class HighlightModel {

	constructor() {

		this.N = 0;
		this.selected = - 1;
		this.hover = - 1;
		this.preview = - 1;
		this._layers = { selection: null, cluster: null, khop: null, path: null };

	}

	reset( N ) {

		this.N = Math.max( 0, N | 0 );
		this.selected = this.hover = this.preview = - 1;
		for ( const name of LAYERS ) this._layers[ name ] = null;

	}

	set( name, weights, meta = null ) {

		if ( ! LAYERS.includes( name ) ) throw new RangeError( `HighlightModel: unknown layer "${name}" (${LAYERS.join( ', ' )})` );
		if ( weights && weights.length < this.N ) throw new RangeError( `HighlightModel: layer "${name}" needs ${this.N} weights, got ${weights.length}` );
		this._layers[ name ] = weights ? { weights, meta } : null;

	}

	get( name ) {

		return this._layers[ name ] || null;

	}

	has( name ) {

		return !! this._layers[ name ];

	}

	get active() {

		return LAYERS.some( ( name ) => this._layers[ name ] );

	}

	top() {

		return PEEL_ORDER.find( ( name ) => this._layers[ name ] ) || null;

	}

	visible() {

		const exclusive = EXCLUSIVE.find( ( name ) => this._layers[ name ] );
		if ( exclusive ) return [ exclusive ];
		return [ 'selection', 'cluster' ].filter( ( name ) => this._layers[ name ] );

	}

	compose( out ) {

		const N = this.N;
		const t = out && out.length >= 4 * N ? out : new Float32Array( 4 * Math.max( 1, N ) );
		const dim = this.active ? 1 : 0;
		for ( let i = 0; i < N; i ++ ) {

			t[ 4 * i ] = 0; t[ 4 * i + 1 ] = 0; t[ 4 * i + 2 ] = dim; t[ 4 * i + 3 ] = 0;

		}

		const L = this._layers;
		const exclusive = EXCLUSIVE.find( ( name ) => L[ name ] );
		if ( exclusive === 'path' ) {

			const w = L.path.weights;
			for ( let i = 0; i < N; i ++ ) {

				const v = w[ i ];
				if ( ! ( v > 0 ) ) continue;
				if ( v >= 1 ) {

					t[ 4 * i ] = PATH_END; t[ 4 * i + 1 ] = 1;

				} else t[ 4 * i + 1 ] = Math.min( 1, v + 0.15 );

				t[ 4 * i + 2 ] = 0;

			}

		} else if ( exclusive === 'khop' ) {

			const w = L.khop.weights;
			for ( let i = 0; i < N; i ++ ) {

				const v = w[ i ];
				if ( ! ( v > 0 ) ) continue;
				if ( v >= 1 ) {

					t[ 4 * i ] = 1; t[ 4 * i + 2 ] = 0;

				} else {

					const far = Math.min( 1, Math.max( 0, ( SECONDARY - v ) / RING_SPAN ) ); // 0 first ring … 1 last ring
					t[ 4 * i + 1 ] = 1 - 0.55 * far;
					t[ 4 * i + 2 ] = 0.35 * far;

				}

			}

		} else {

			if ( L.cluster ) {

				const w = L.cluster.weights;
				for ( let i = 0; i < N; i ++ ) {

					if ( ! ( w[ i ] > 0 ) ) continue;
					t[ 4 * i + 1 ] = CLUSTER_GLOW; t[ 4 * i + 2 ] = 0;

				}

			}

			if ( L.selection ) {

				const w = L.selection.weights;
				for ( let i = 0; i < N; i ++ ) {

					const v = w[ i ];
					if ( ! ( v > 0 ) ) continue;
					if ( v >= 1 ) t[ 4 * i ] = 1;
					else t[ 4 * i + 1 ] = 1;
					t[ 4 * i + 2 ] = 0;

				}

			}

		}

		// the selected node stays in front of whatever is on top
		const s = this.selected;
		if ( s >= 0 && s < N ) {

			t[ 4 * s ] = exclusive === 'path' ? Math.max( t[ 4 * s ], PATH_END ) : 1;
			t[ 4 * s + 2 ] = 0;

		}

		if ( this.hover >= 0 && this.hover < N ) t[ 4 * this.hover + 3 ] = 1;
		if ( this.preview >= 0 && this.preview < N ) t[ 4 * this.preview + 3 ] = 1;
		return t;

	}

}

/** Weights of a selection: 1 at the node, 0.85 at its neighbours. */
export function selectionWeights( graph, index, out ) {

	const N = graph.N;
	const w = out && out.length >= N ? out.fill( 0 ) : new Float32Array( Math.max( 1, N ) );
	if ( ! ( index >= 0 && index < N ) ) return w;
	const s = graph.adjStart[ index ], e = s + graph.degree[ index ];
	for ( let k = s; k < e; k ++ ) w[ graph.adjList[ k ] ] = SECONDARY;
	w[ index ] = 1;
	return w;

}

/** Level (0 = finest) of a cluster index of the prepared graph, −1 for the root or an invalid index. */
export function clusterLevel( graph, cluster ) {

	if ( ! ( cluster >= 0 && cluster < graph.C ) || cluster === graph.root ) return - 1;
	return graph.clusterHeight[ cluster ] - 1;

}

/** The nodes of a cluster (index in the prepared graph's cluster space). */
export function clusterMembers( graph, cluster ) {

	const level = clusterLevel( graph, cluster );
	if ( level < 0 || level >= graph.D ) return new Uint32Array( 0 );
	const N = graph.N, D = graph.D, anc = graph.anc;
	const out = new Uint32Array( graph.clusterMass ? graph.clusterMass[ cluster ] : N );
	let n = 0;
	for ( let i = 0; i < N && n < out.length; i ++ ) if ( anc[ i * D + level ] === cluster ) out[ n ++ ] = i;
	return out.subarray( 0, n );

}

/** Weights of a community focus: 0.6 for every member. */
export function clusterWeights( graph, cluster, out ) {

	const N = graph.N;
	const w = out && out.length >= N ? out.fill( 0 ) : new Float32Array( Math.max( 1, N ) );
	const members = clusterMembers( graph, cluster );
	for ( let k = 0; k < members.length; k ++ ) w[ members[ k ] ] = 0.6;
	return w;

}
