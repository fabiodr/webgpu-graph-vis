// Kernels and CPU-side structures of the GPU force layout (see ForceLayout.js for scheduling).
//
// ─── Barnes–Hut structure ("BH tree") ───────────────────────────────────────────────────────
//  Repulsion walks the clustering hierarchy (prepare.js bhTree order), extended with *virtual
//  clusters*: every cluster whose children are nodes (height-1 clusters, or the root when D = 0)
//  and that has more than `leafSize` of them is split, recursively and up to `fanout` ways, into
//  spatially compact groups of at most `leafSize` nodes. Without this the own-leaf-cluster cost is
//  Σ size² (a single 5000-node Louvain community costs 25M interactions per tick; D = 0 is O(N²));
//  with it a 20k-node graph needs ~100–250 walk steps per node whatever the hierarchy.
//
//  The *shape* of the virtual subtrees depends only on cluster sizes, so all buffers and dispatch
//  ranges are fixed per graph. Which node sits in which group depends on positions: the groups
//  are re-partitioned from position readbacks (`resortStructure`, recursive median bisection) by
//  rewriting one N-sized permutation (`nodeOrder`); nothing else is re-uploaded.
//
//  Index spaces (private to the layout):
//    slots     0..N-1   preorder leaf order of the tree; nodeOrder[slot] = node index
//    bh ids    0..C-1 real clusters (same ids as prepare.js), C..C+V-1 virtual clusters,
//              sorted by virtual height ascending
//    tree      0..TT-1  preorder, uvec4 (ref, skip, mass, 0); ref < N: slot, else N + bh id
//
//  Private storage buffers (createPrivateBuffers; "ro" = only the CPU writes them):
//    tree       uvec4 TT    ro  BH tree (static per graph)
//    slotTree   uint  N     ro  slot -> tree index (static)
//    nodeOrder  uint  N     ro  slot -> node (rewritten by resorts)
//    vInfo      uvec4 V     ro  virtual clusters: (start, count, height, mass). height 1: children
//                               are slots start..start+count-1; otherwise vChildList[start..]
//    vChildList uint        ro  children (bh ids) of virtual clusters of height > 1
//    rInfo      uvec4 C     ro  real height-1 clusters: (start, count, virtual?, mass) into
//    rChildList uint        ro  node ids, or bh ids of the top virtual clusters when split
//    bhCluster  vec4  C+V       centroid xyz, bounding radius w, real and virtual clusters
//    treePos    vec4  TT        tree-ordered copy: node xyz (w = 0) / cluster centroid + radius
//
// ─── Kernels (TSL compute nodes; one thread per element, 64-wide workgroups) ────────────────
//  aggregateVirtualKernel   virtual clusters of one height -> bhCluster
//  aggregateRealKernel      clusters of one height: mass-weighted centroid + conservative
//                           (slightly padded) radius -> clusterPos (shared) and bhCluster
//  flattenKernel            per tree entry -> treePos (so the walk reads memory sequentially)
//  repulsionKernel          per slot: stackless BH walk over treePos -> nodeForce (charge-weighted,
//                           alpha-free)
//  velocityKernel           per node: v = mix(d3 update, sphere spring, layoutBlend)
//  positionKernel           per node: x += clamp(v) − centering; pinned / dragged handling
//  commandKernel            per node: pin commands, unpin-all, explicit moves (drop/paused drag)
//  stateKernel              per node: framerate-independent easing of nodeState
//
// Storage buffers per kernel stay ≤ 7 (WebGPU default/mobile limit is 8 per stage). Buffers a
// kernel only reads are bound through read-only views (readOnly()).
//
// CPU mirrors (aggregateCPU, repulsionCPU) replicate the GPU math (the GPU walk matches the
// mirror to ~1e-7); they back the unit tests, the per-graph work estimate (estimateVisits) and
// the dev page's accuracy check.

import {
	Fn, If, Loop, Break, instanceIndex, uint, int, float, vec3, vec4, dot, sqrt, max, min, mix,
	length, select, step, abs, instancedArray, storage,
} from 'three/tsl';

export const DEFAULT_LEAF_SIZE = 16;
export const DEFAULT_FANOUT = 8;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// CPU: structure
// ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Builds the extended Barnes–Hut structure for a PreparedGraph.
 * @returns {{ N, C, V, TT, tree: Uint32Array, slotTree: Uint32Array, nodeOrder: Uint32Array,
 *   vInfo: Uint32Array, vChildList: Uint32Array, vHeightRanges: Array<{height,start,count}>,
 *   rInfo: Uint32Array, rChildList: Uint32Array, groups: Array<{cluster,lo,hi,parts}>, leafSize, fanout }}
 */
export function buildStructure( g, { leafSize = DEFAULT_LEAF_SIZE, fanout = DEFAULT_FANOUT } = {} ) {

	const { N, C, root, clusterChildStart, clusterChildCount, childRefs, clusterHeight, clusterMass } = g;
	leafSize = Math.max( 2, leafSize | 0 );
	fanout = Math.max( 2, fanout | 0 );

	const tree = [];
	const nodeOrder = new Uint32Array( N );
	let slot = 0;
	// virtual clusters before height sorting: { lo, count, height, children: [vtmp ids] | null }
	const vtmp = [];
	// oversized leaf-owning clusters: { lo, hi, parts } where parts is the shape for resorting
	const groups = [];
	const vTreeIndex = []; // vtmp id -> tree index (to patch the ref after sorting by height)

	const emitSlots = ( nodes ) => {

		for ( const node of nodes ) {

			nodeOrder[ slot ] = node;
			const idx = tree.length / 4;
			tree.push( slot, idx + 1, 1, 0 );
			slot ++;

		}

	};

	// splits `nodes` (array) into virtual clusters; returns the shape: array of parts { lo, hi, parts | null }
	const emitVirtual = ( nodes ) => {

		const n = nodes.length;
		const parts = Math.min( fanout, Math.ceil( n / leafSize ) );
		const shape = [];
		const ids = [];
		let off = 0;
		for ( let p = 0; p < parts; p ++ ) {

			const size = Math.floor( ( n * ( p + 1 ) ) / parts ) - Math.floor( ( n * p ) / parts );
			const sub = nodes.slice( off, off + size );
			off += size;
			const v = vtmp.length;
			const rec = { lo: slot, count: size, height: 1, children: null };
			vtmp.push( rec );
			ids.push( v );
			const idx = tree.length / 4;
			vTreeIndex[ v ] = idx;
			tree.push( 0, 0, size, 0 ); // ref patched later
			const part = { lo: slot, hi: slot + size, parts: null };
			if ( size > leafSize ) {

				const r = emitVirtual( sub );
				rec.children = r.ids;
				rec.height = 1 + Math.max( ...r.ids.map( ( c ) => vtmp[ c ].height ) );
				part.parts = r.shape;

			} else {

				emitSlots( sub );

			}

			tree[ 4 * idx + 1 ] = tree.length / 4;
			shape.push( part );

		}

		return { shape, ids };

	};

	const emitRef = ( ref ) => {

		if ( ref < N ) {

			emitSlots( [ ref ] );
			return;

		}

		const c = ref - N;
		const idx = tree.length / 4;
		tree.push( ref, 0, clusterMass[ c ], 0 );
		const s = clusterChildStart[ c ], n = clusterChildCount[ c ];
		const leafOwner = clusterHeight[ c ] === 1;
		if ( leafOwner && n > leafSize ) {

			const lo = slot;
			const nodes = Array.from( childRefs.subarray( s, s + n ) );
			const r = emitVirtual( nodes );
			groups.push( { cluster: c, lo, hi: slot, parts: r.shape, top: r.ids } );

		} else {

			for ( let k = 0; k < n; k ++ ) emitRef( childRefs[ s + k ] );

		}

		tree[ 4 * idx + 1 ] = tree.length / 4;

	};

	// the root is not part of the tree; a D = 0 root owns the nodes directly
	{

		const s = clusterChildStart[ root ], n = clusterChildCount[ root ];
		if ( clusterHeight[ root ] === 1 && n > leafSize ) {

			const nodes = Array.from( childRefs.subarray( s, s + n ) );
			const r = emitVirtual( nodes );
			groups.push( { cluster: root, lo: 0, hi: slot, parts: r.shape, top: r.ids } );

		} else {

			for ( let k = 0; k < n; k ++ ) emitRef( childRefs[ s + k ] );

		}

	}

	if ( slot !== N ) throw new Error( `buildStructure: slot count mismatch (${slot} vs ${N})` );

	// sort virtual clusters by height -> final ids
	const V = vtmp.length;
	const perm = Array.from( { length: V }, ( _, i ) => i ).sort( ( a, b ) => vtmp[ a ].height - vtmp[ b ].height || a - b );
	const finalId = new Uint32Array( V );
	perm.forEach( ( v, k ) => ( finalId[ v ] = k ) );
	const vInfo = new Uint32Array( 4 * Math.max( 1, V ) );
	const vChild = [];
	const vHeightRanges = [];
	for ( let k = 0; k < V; k ++ ) {

		const rec = vtmp[ perm[ k ] ];
		if ( rec.children ) {

			vInfo[ 4 * k ] = vChild.length;
			vInfo[ 4 * k + 1 ] = rec.children.length;
			for ( const ch of rec.children ) vChild.push( C + finalId[ ch ] );

		} else {

			vInfo[ 4 * k ] = rec.lo;
			vInfo[ 4 * k + 1 ] = rec.count;

		}

		vInfo[ 4 * k + 2 ] = rec.height;
		vInfo[ 4 * k + 3 ] = rec.count;
		const last = vHeightRanges[ vHeightRanges.length - 1 ];
		if ( last && last.height === rec.height ) last.count ++;
		else vHeightRanges.push( { height: rec.height, start: k, count: 1 } );

	}

	for ( let v = 0; v < V; v ++ ) tree[ 4 * vTreeIndex[ v ] ] = N + C + finalId[ v ];

	// children of real height-1 clusters for aggregation: split clusters aggregate their few
	// top-level virtual clusters (bh ids, flag 1) instead of all their nodes (flag 0)
	const rInfo = new Uint32Array( 4 * Math.max( 1, C ) );
	const rChild = [];
	const split = new Map( groups.map( ( gr ) => [ gr.cluster, gr ] ) );
	for ( let c = 0; c < C; c ++ ) {

		if ( clusterHeight[ c ] !== 1 ) continue;
		rInfo[ 4 * c ] = rChild.length;
		const gr = split.get( c );
		if ( gr ) {

			for ( const v of gr.top ) rChild.push( C + finalId[ v ] );
			rInfo[ 4 * c + 2 ] = 1;

		} else {

			const s = clusterChildStart[ c ];
			for ( let k = 0; k < clusterChildCount[ c ]; k ++ ) rChild.push( childRefs[ s + k ] );

		}

		rInfo[ 4 * c + 1 ] = rChild.length - rInfo[ 4 * c ];
		rInfo[ 4 * c + 3 ] = clusterMass[ c ];

	}

	for ( const gr of groups ) delete gr.top;

	const TT = tree.length / 4;
	const slotTree = new Uint32Array( Math.max( 1, N ) );
	for ( let k = 0; k < TT; k ++ ) if ( tree[ 4 * k ] < N ) slotTree[ tree[ 4 * k ] ] = k;
	return {
		N, C, V, TT, leafSize, fanout,
		tree: Uint32Array.from( tree.length ? tree : [ 0, 1, 0, 0 ] ),
		slotTree,
		nodeOrder,
		vInfo,
		vChildList: Uint32Array.from( vChild.length ? vChild : [ 0 ] ),
		vHeightRanges,
		rInfo,
		rChildList: Uint32Array.from( rChild.length ? rChild : [ 0 ] ),
		groups,
	};

}

/**
 * Re-partitions the nodes of oversized leaf clusters (groups[from..to)) from positions `pos4`
 * (4 floats per node) by recursive median bisection along the longest axis, consistent with the
 * fixed shape. Rewrites `struct.nodeOrder` in place.
 * @returns {{ changed: boolean, lo: number, hi: number }} slot range that changed (lo >= hi: none)
 */
export function resortStructure( struct, pos4, from = 0, to = struct.groups.length ) {

	const queue = resortTasks( struct, from, to );
	let lo = Infinity, hi = - Infinity;
	while ( queue.length ) {

		const r = runResortTask( struct, pos4, queue.shift(), queue );
		if ( r.changed ) {

			lo = Math.min( lo, r.lo );
			hi = Math.max( hi, r.hi );

		}

	}

	return { changed: hi > lo, lo, hi };

}

/** Top-level resort tasks ({ lo, hi, parts }) of groups[from..to). */
export function resortTasks( struct, from = 0, to = struct.groups.length ) {

	return struct.groups.slice( from, to ).map( ( g ) => ( { lo: g.lo, hi: g.hi, parts: g.parts } ) );

}

/**
 * One unit of resort work: distributes the slots [task.lo, task.hi) over task.parts by median
 * bisection, then appends the sub-tasks (parts that are split further) to `queue`. Every
 * intermediate state is a valid permutation, so the work can be spread over frames.
 * @returns {{ changed, lo, hi }}
 */
export function runResortTask( struct, pos4, task, queue ) {

	const order = struct.nodeOrder;
	const key = struct._resortKey || ( struct._resortKey = new Float32Array( order.length ) ); // scratch
	const before = order.slice( task.lo, task.hi );

	const bisect = ( lo, hi, parts ) => {

		if ( parts.length <= 1 || hi - lo <= 1 ) return;
		const k1 = Math.ceil( parts.length / 2 );
		const mid = parts[ k1 ].lo;
		// longest axis of the bounding box
		let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = - Infinity, y1 = - Infinity, z1 = - Infinity;
		for ( let s = lo; s < hi; s ++ ) {

			const o = 4 * order[ s ];
			const x = pos4[ o ], y = pos4[ o + 1 ], z = pos4[ o + 2 ];
			if ( x < x0 ) x0 = x; if ( x > x1 ) x1 = x;
			if ( y < y0 ) y0 = y; if ( y > y1 ) y1 = y;
			if ( z < z0 ) z0 = z; if ( z > z1 ) z1 = z;

		}

		const ex = x1 - x0, ey = y1 - y0, ez = z1 - z0;
		const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
		for ( let s = lo; s < hi; s ++ ) {

			const v = pos4[ 4 * order[ s ] + axis ];
			key[ s ] = Number.isFinite( v ) ? v : 0;

		}

		quickselect( order, key, lo, hi - 1, mid );
		bisect( lo, mid, parts.slice( 0, k1 ) );
		bisect( mid, hi, parts.slice( k1 ) );

	};

	bisect( task.lo, task.hi, task.parts );
	for ( const p of task.parts ) if ( p.parts ) queue.push( { lo: p.lo, hi: p.hi, parts: p.parts } );
	let changed = false;
	for ( let s = task.lo; s < task.hi && ! changed; s ++ ) changed = order[ s ] !== before[ s - task.lo ];
	return { changed, lo: task.lo, hi: task.hi };

}

// in-place nth_element on order[lo..hi] keyed by key[], so that order[k] is the k-th smallest
function quickselect( order, key, lo, hi, k ) {

	while ( hi > lo ) {

		const m = ( lo + hi ) >> 1;
		// median of three as pivot
		const a = key[ lo ], b = key[ m ], c = key[ hi ];
		const pivot = a < b ? ( b < c ? b : a < c ? c : a ) : ( a < c ? a : b < c ? c : b );
		let i = lo, j = hi;
		while ( i <= j ) {

			while ( key[ i ] < pivot ) i ++;
			while ( key[ j ] > pivot ) j --;
			if ( i <= j ) {

				const t = order[ i ]; order[ i ] = order[ j ]; order[ j ] = t;
				const tk = key[ i ]; key[ i ] = key[ j ]; key[ j ] = tk;
				i ++; j --;

			}

		}

		if ( k <= j ) hi = j;
		else if ( k >= i ) lo = i;
		else return;

	}

}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// CPU mirrors
// ─────────────────────────────────────────────────────────────────────────────────────────────

const padCPU = ( r ) => r * ( 1 + 1e-5 ) + 1e-4; // mirrors pad() of the GPU kernels

/** CPU mirror of the aggregation passes: returns bhCluster (Float32Array 4·(C+V)). */
export function aggregateCPU( g, struct, pos4 ) {

	const { N, C, V, vInfo, vChildList, nodeOrder } = struct;
	const out = new Float32Array( 4 * ( C + V ) );
	for ( let v = 0; v < V; v ++ ) {

		const start = vInfo[ 4 * v ], count = vInfo[ 4 * v + 1 ], h = vInfo[ 4 * v + 2 ];
		let x = 0, y = 0, z = 0, m = 0;
		const child = ( k ) => ( h === 1 ? 4 * nodeOrder[ start + k ] : 4 * vChildList[ start + k ] );
		const src = h === 1 ? pos4 : out;
		for ( let k = 0; k < count; k ++ ) {

			const o = child( k );
			const w = h === 1 ? 1 : vInfo[ 4 * ( vChildList[ start + k ] - C ) + 3 ];
			x += src[ o ] * w; y += src[ o + 1 ] * w; z += src[ o + 2 ] * w; m += w;

		}

		m = Math.max( m, 1e-9 ); x /= m; y /= m; z /= m;
		let r = 0;
		for ( let k = 0; k < count; k ++ ) {

			const o = child( k );
			const d = Math.hypot( src[ o ] - x, src[ o + 1 ] - y, src[ o + 2 ] - z ) + ( h === 1 ? 0 : src[ o + 3 ] );
			if ( d > r ) r = d;

		}

		const o = 4 * ( C + v );
		out[ o ] = x; out[ o + 1 ] = y; out[ o + 2 ] = z; out[ o + 3 ] = padCPU( r );

	}

	const { rInfo, rChildList } = struct;
	for ( const { height, start, count } of g.heightRanges ) {

		for ( let c = start; c < start + count; c ++ ) {

			// height 1: rInfo children (nodes, or bh ids of virtual clusters when split)
			const h1 = height === 1;
			const s = h1 ? rInfo[ 4 * c ] : g.clusterChildStart[ c ], n = h1 ? rInfo[ 4 * c + 1 ] : g.clusterChildCount[ c ];
			const list = h1 ? rChildList : g.childList;
			const nodes = h1 && rInfo[ 4 * c + 2 ] === 0;
			const weight = ( ch ) => ( nodes ? 1 : h1 ? vInfo[ 4 * ( ch - C ) + 3 ] : g.clusterMass[ ch ] );
			let x = 0, y = 0, z = 0, m = 0;
			for ( let k = 0; k < n; k ++ ) {

				const ch = list[ s + k ];
				const src = nodes ? pos4 : out, w = weight( ch );
				x += src[ 4 * ch ] * w; y += src[ 4 * ch + 1 ] * w; z += src[ 4 * ch + 2 ] * w; m += w;

			}

			m = Math.max( m, 1e-9 ); x /= m; y /= m; z /= m;
			let r = 0;
			for ( let k = 0; k < n; k ++ ) {

				const ch = list[ s + k ];
				const src = nodes ? pos4 : out;
				const d = Math.hypot( src[ 4 * ch ] - x, src[ 4 * ch + 1 ] - y, src[ 4 * ch + 2 ] - z ) + ( nodes ? 0 : src[ 4 * ch + 3 ] );
				if ( d > r ) r = d;

			}

			out[ 4 * c ] = x; out[ 4 * c + 1 ] = y; out[ 4 * c + 2 ] = z; out[ 4 * c + 3 ] = padCPU( r );

		}

	}

	void N;
	return out;

}

/** Deterministic antisymmetric direction for coincident pairs (mirrors the GPU hash). */
function jitterCPU( a, b ) {

	const lo = Math.min( a, b ), hi = Math.max( a, b );
	const h = ( Math.imul( lo, 73856093 ) ^ Math.imul( hi, 19349663 ) ) >>> 0;
	const s = a < b ? 1 : - 1;
	return [ ( ( h & 1023 ) / 1023 - 0.5 ) * s, ( ( ( h >>> 10 ) & 1023 ) / 1023 - 0.5 ) * s, ( ( ( h >>> 20 ) & 1023 ) / 1023 - 0.5 ) * s ];

}

/**
 * CPU mirror of repulsionKernel. Returns { force: Float32Array 4N (node-indexed, like nodeForce),
 * visits: Uint32Array N (loop iterations per node) }. `slots` restricts the work to a subset.
 */
export function repulsionCPU( struct, pos4, bhCluster, { theta = 0.8, charge = - 30, distanceMin = 1, maxIterations = 8192, slots = null } = {} ) {

	const { N, TT, tree, nodeOrder } = struct;
	const theta2 = theta * theta, dmin2 = distanceMin * distanceMin;
	const soft = maxIterations >>> 1;
	const force = new Float32Array( 4 * N );
	const visits = new Uint32Array( N );
	const list = slots || Array.from( { length: N }, ( _, i ) => i );
	for ( const t of list ) {

		const node = nodeOrder[ t ];
		const px = pos4[ 4 * node ], py = pos4[ 4 * node + 1 ], pz = pos4[ 4 * node + 2 ];
		let fx = 0, fy = 0, fz = 0;
		let k = 0, it = 0;
		for ( ; it < maxIterations; it ++ ) {

			if ( k >= TT ) break;
			const ref = tree[ 4 * k ];
			if ( ref < N ) {

				if ( ref !== t ) {

					const other = nodeOrder[ ref ];
					let dx = pos4[ 4 * other ] - px, dy = pos4[ 4 * other + 1 ] - py, dz = pos4[ 4 * other + 2 ] - pz;
					let l2 = dx * dx + dy * dy + dz * dz;
					if ( l2 < 1e-12 ) {

						const j = jitterCPU( node, other );
						dx = j[ 0 ] * 1e-3; dy = j[ 1 ] * 1e-3; dz = j[ 2 ] * 1e-3;
						l2 = dx * dx + dy * dy + dz * dz;

					}

					const l2c = l2 < dmin2 ? Math.sqrt( dmin2 * l2 ) : l2;
					const w = charge / l2c;
					fx += dx * w; fy += dy * w; fz += dz * w;

				}

				k ++;

			} else {

				const o = 4 * ( ref - N );
				const dx = bhCluster[ o ] - px, dy = bhCluster[ o + 1 ] - py, dz = bhCluster[ o + 2 ] - pz;
				const l2 = dx * dx + dy * dy + dz * dz;
				const r2 = bhCluster[ o + 3 ] * bhCluster[ o + 3 ];
				const outside = r2 < l2;
				if ( outside && ( r2 <= theta2 * l2 || it >= soft ) ) {

					const l2c = l2 < dmin2 ? Math.sqrt( dmin2 * l2 ) : l2;
					const w = ( charge * tree[ 4 * k + 2 ] ) / l2c;
					fx += dx * w; fy += dy * w; fz += dz * w;
					k = tree[ 4 * k + 1 ];

				} else k ++;

			}

		}

		force[ 4 * node ] = fx; force[ 4 * node + 1 ] = fy; force[ 4 * node + 2 ] = fz;
		visits[ node ] = it;

	}

	return { force, visits };

}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// GPU
// ─────────────────────────────────────────────────────────────────────────────────────────────

/** Allocates the private storage buffers for a structure (see header). */
export function createPrivateBuffers( struct ) {

	// CPU-written only: bound read-only everywhere
	const tree = instancedArray( struct.tree, 'uvec4' ).setName( 'bhTreeExt' ).toReadOnly();
	const slotTree = instancedArray( struct.slotTree, 'uint' ).setName( 'bhSlotTree' ).toReadOnly();
	const nodeOrder = instancedArray( new Uint32Array( struct.nodeOrder.length ? struct.nodeOrder : [ 0 ] ), 'uint' ).setName( 'bhNodeOrder' ).toReadOnly();
	const vInfo = instancedArray( struct.vInfo, 'uvec4' ).setName( 'bhVirtualInfo' ).toReadOnly();
	const vChildList = instancedArray( struct.vChildList, 'uint' ).setName( 'bhVirtualChildren' ).toReadOnly();
	const rInfo = instancedArray( struct.rInfo, 'uvec4' ).setName( 'bhRealInfo' ).toReadOnly();
	const rChildList = instancedArray( struct.rChildList, 'uint' ).setName( 'bhRealChildren' ).toReadOnly();
	// GPU-written
	const treePos = instancedArray( Math.max( 1, struct.TT ), 'vec4' ).setName( 'bhTreePos' );
	const bhCluster = instancedArray( Math.max( 1, struct.C + struct.V ), 'vec4' ).setName( 'bhCluster' );
	return { tree, treePos, slotTree, nodeOrder, bhCluster, vInfo, vChildList, rInfo, rChildList };

}

const WG = () => [ 64 ];

const roViews = new WeakMap();
/**
 * Read-only binding of a storage node's attribute (`read` access lets drivers use their read-only
 * cache paths). A separate node, so the original (shared) node keeps its read_write access; never
 * use both in the same kernel.
 */
export function readOnly( node ) {

	let v = roViews.get( node );
	if ( ! v ) {

		v = storage( node.value, node.bufferType, node.bufferCount ).setName( ( node.name || 'buffer' ) + 'RO' ).toReadOnly();
		roViews.set( node, v );

	}

	return v;

}

// antisymmetric pseudo-random direction for coincident pairs (a, b: uint nodes)
const jitter = ( a, b ) => {

	const lo = min( a, b ), hi = max( a, b );
	const h = lo.mul( uint( 73856093 ) ).bitXor( hi.mul( uint( 19349663 ) ) ).toVar();
	const s = select( a.lessThan( b ), float( 1 ), float( - 1 ) );
	const f = ( shift ) => float( h.shiftRight( uint( shift ) ).bitAnd( uint( 1023 ) ) ).div( 1023 ).sub( 0.5 );
	return vec3( f( 0 ), f( 10 ), f( 20 ) ).mul( s );

};

// d3 manyBody contribution: d = other - self, l2 = |d|², w = charge·mass (negative repels)
const chargeTerm = ( d, l2, w, U ) => {

	const l2c = select( l2.lessThan( U.distanceMin2 ), sqrt( U.distanceMin2.mul( l2 ) ), l2 );
	return d.mul( w.div( l2c ) );

};

// bounding radii are padded slightly so float rounding can never put a member node outside
// (the walk relies on "outside the bounding sphere" to never accept a node's own cluster)
const RADIUS_PAD_REL = 1e-5, RADIUS_PAD_ABS = 1e-4;
const pad = ( r ) => r.mul( 1 + RADIUS_PAD_REL ).add( RADIUS_PAD_ABS );

/**
 * Aggregation of real clusters [start, start+count) of one height (baked).
 * Height 1 reads the private rInfo/rChildList: nodes, or (for clusters split into virtual
 * clusters) their few top-level virtual clusters, so no thread loops over thousands of nodes.
 * Higher clusters read the shared clusterInfo/childList with child mass = clusterInfo.w.
 */
export function aggregateRealKernel( { buffers, priv, C, start, count, nodeChildren } ) {

	const { clusterPos } = buffers;
	const { bhCluster } = priv;
	const nodePos = readOnly( buffers.nodePos );
	const clusterInfo = readOnly( buffers.clusterInfo ), childList = readOnly( buffers.childList );
	const { rInfo, rChildList, vInfo } = priv;

	// one pass over a child list: weighted centroid, then conservative radius
	const reduce = ( cs, cn, pos, mass, radius ) => {

		const sum = vec3( 0 ).toVar();
		const m = float( 0 ).toVar();
		Loop( { start: uint( 0 ), end: cn, type: 'uint', condition: '<' }, ( { i } ) => {

			const w = mass( cs.add( i ) ).toVar();
			sum.addAssign( pos( cs.add( i ) ).mul( w ) );
			m.addAssign( w );

		} );
		const ctr = sum.div( max( m, 1e-9 ) ).toVar();
		const r = float( 0 ).toVar();
		Loop( { start: uint( 0 ), end: cn, type: 'uint', condition: '<' }, ( { i } ) => {

			r.assign( max( r, length( pos( cs.add( i ) ).sub( ctr ) ).add( radius( cs.add( i ) ) ) ) );

		} );
		return vec4( ctr, pad( r ) );

	};

	return Fn( () => {

		const c = instanceIndex.add( uint( start ) ).toVar();
		const out = vec4( 0 ).toVar();
		if ( nodeChildren ) {

			const info = rInfo.element( c ).toVar();
			If( info.z.equal( uint( 0 ) ), () => {

				out.assign( reduce( info.x, info.y,
					( k ) => nodePos.element( rChildList.element( k ) ).xyz,
					() => float( 1 ),
					() => float( 0 ) ) );

			} ).Else( () => {

				out.assign( reduce( info.x, info.y,
					( k ) => bhCluster.element( rChildList.element( k ) ).xyz,
					( k ) => float( vInfo.element( rChildList.element( k ).sub( uint( C ) ) ).w ),
					( k ) => bhCluster.element( rChildList.element( k ) ).w ) );

			} );

		} else {

			const info = clusterInfo.element( c ).toVar();
			out.assign( reduce( info.x, info.y,
				( k ) => clusterPos.element( childList.element( k ) ).xyz,
				( k ) => float( clusterInfo.element( childList.element( k ) ).w ),
				( k ) => clusterPos.element( childList.element( k ) ).w ) );

		}

		clusterPos.element( c ).assign( out );
		bhCluster.element( c ).assign( out );

	} )().compute( count, WG() ).setName( `layout.aggregate.h${nodeChildren ? 1 : 'n'}.${start}` );

}

/** Aggregation of virtual clusters [start, start+count) of one virtual height (baked). */
export function aggregateVirtualKernel( { buffers, priv, C, start, count, height } ) {

	const nodePos = readOnly( buffers.nodePos );
	const { bhCluster, vInfo, vChildList, nodeOrder } = priv;
	const leaf = height === 1;
	return Fn( () => {

		const v = instanceIndex.add( uint( start ) ).toVar();
		const info = vInfo.element( v ).toVar();
		const cs = info.x, cn = info.y;
		const sum = vec3( 0 ).toVar();
		const m = float( 0 ).toVar();
		Loop( { start: uint( 0 ), end: cn, type: 'uint', condition: '<' }, ( { i } ) => {

			if ( leaf ) {

				sum.addAssign( nodePos.element( nodeOrder.element( cs.add( i ) ) ).xyz );
				m.addAssign( 1 );

			} else {

				const ch = vChildList.element( cs.add( i ) );
				const w = float( vInfo.element( ch.sub( uint( C ) ) ).w );
				sum.addAssign( bhCluster.element( ch ).xyz.mul( w ) );
				m.addAssign( w );

			}

		} );

		const ctr = sum.div( max( m, 1e-9 ) ).toVar();
		const r = float( 0 ).toVar();
		Loop( { start: uint( 0 ), end: cn, type: 'uint', condition: '<' }, ( { i } ) => {

			if ( leaf ) {

				r.assign( max( r, length( nodePos.element( nodeOrder.element( cs.add( i ) ) ).xyz.sub( ctr ) ) ) );

			} else {

				const cp = bhCluster.element( vChildList.element( cs.add( i ) ) );
				r.assign( max( r, length( cp.xyz.sub( ctr ) ).add( cp.w ) ) );

			}

		} );

		bhCluster.element( v.add( uint( C ) ) ).assign( vec4( ctr, pad( r ) ) );

	} )().compute( count, WG() ).setName( `layout.aggregate.v${height}` );

}

/**
 * Copies positions (nodes, radius 0) and centroids/radii (clusters) into tree order, so the
 * Barnes–Hut walk reads one vec4 + one uvec4 per step, sequentially in the tree index.
 */
export function flattenKernel( { buffers, priv, N, TT } ) {

	const nodePos = readOnly( buffers.nodePos );
	const { tree, treePos, nodeOrder } = priv;
	const bhCluster = readOnly( priv.bhCluster );
	return Fn( () => {

		const k = instanceIndex;
		const ref = tree.element( k ).x.toVar();
		If( ref.lessThan( uint( N ) ), () => {

			treePos.element( k ).assign( vec4( nodePos.element( nodeOrder.element( ref ) ).xyz, 0 ) );

		} ).Else( () => {

			treePos.element( k ).assign( bhCluster.element( ref.sub( uint( N ) ) ) );

		} );

	} )().compute( TT, WG() ).setName( 'layout.flatten' );

}

/**
 * Barnes–Hut repulsion. One thread per slot (tree leaf order, so a workgroup walks nearly the
 * same path). Nodes and clusters are handled alike (a node is a cluster of radius 0, mass 1):
 * an entry is accepted as a point mass when the node lies outside its bounding sphere and
 * radius < theta·distance; otherwise the walk descends (k + 1). The node's own entry has
 * distance 0 and is never accepted, which skips self-interaction. After maxIterations/2 loop
 * iterations any entry the node is outside of is accepted (graceful degradation), and
 * maxIterations is a hard cap. Exactly coincident distinct nodes get a deterministic jitter.
 */
export function repulsionKernel( { buffers, priv, U, N, TT } ) {

	const { nodeForce } = buffers;
	const { tree, slotTree, nodeOrder } = priv;
	const treePos = readOnly( priv.treePos );
	return Fn( () => {

		const t = instanceIndex;
		const node = nodeOrder.element( t ).toVar();
		const self = slotTree.element( t ).toVar();
		const p = treePos.element( self ).xyz.toVar();
		const f = vec3( 0 ).toVar();
		const k = uint( 0 ).toVar();
		Loop( { start: uint( 0 ), end: U.maxIterations, type: 'uint', condition: '<', name: 'it' }, ( { it } ) => {

			If( k.greaterThanEqual( uint( TT ) ), () => {

				Break();

			} );

			const e = treePos.element( k ).toVar();
			const m = tree.element( k ).toVar(); // ref, skip, mass
			const d = e.xyz.sub( p ).toVar();
			const l2 = dot( d, d ).toVar();
			const r2 = e.w.mul( e.w ).toVar();
			const accept = r2.lessThan( l2 ).and( r2.lessThanEqual( U.theta2.mul( l2 ) ).or( it.greaterThanEqual( U.softIterations ) ) );
			If( accept, () => {

				f.addAssign( chargeTerm( d, l2, U.charge.mul( float( m.z ) ), U ) );
				k.assign( m.y );

			} ).ElseIf( l2.lessThan( 1e-12 ).and( m.y.equal( k.add( uint( 1 ) ) ) ).and( k.notEqual( self ) ), () => {

				// a distinct node at exactly our position (rare): push apart deterministically
				const j = jitter( node, nodeOrder.element( m.x ) ).mul( 1e-3 ).toVar();
				f.addAssign( chargeTerm( j, dot( j, j ), U.charge, U ) );
				k.addAssign( uint( 1 ) );

			} ).Else( () => {

				k.addAssign( uint( 1 ) );

			} );

		} );

		nodeForce.element( node ).assign( vec4( f, 0 ) );

	} )().compute( N, WG() ).setName( 'layout.repulsion' );

}

/**
 * Velocity update (gather; each thread writes only its own node):
 *   force mode   v_f = (v + α·(repulsion + links + gravity)) · (1 − velocityDecay)
 *                links: d3 forceLink, strength = linkStrength / min(deg_i, deg_j),
 *                bias = deg_j / (deg_i + deg_j), rest length linkDistance
 *   sphere mode  v_s = λ²·v − (1−λ)²·(x − target): a discrete critically damped spring
 *                (double eigenvalue λ = exp(−ω·h)), so nodes converge exactly onto targets
 *   v' = mix(v_f, v_s, layoutBlend)
 */
export function velocityKernel( { buffers, U, N } ) {

	const { nodeVel } = buffers;
	const [ nodePos, nodeForce, nodeInfo, adjList, nodeTarget, nodeState ] = [ buffers.nodePos, buffers.nodeForce, buffers.nodeInfo, buffers.adjList, buffers.nodeTarget, buffers.nodeState ].map( readOnly );
	const { layoutBlend, sphereRadius } = buffers.uniforms;
	return Fn( () => {

		const i = instanceIndex;
		const p = nodePos.element( i ).xyz.toVar();
		const v = nodeVel.element( i ).xyz.mul( U.velRescale ).toVar();
		const b = layoutBlend;
		const vf = vec3( 0 ).toVar();
		const vs = vec3( 0 ).toVar();

		If( b.lessThan( 1 ), () => {

			const F = nodeForce.element( i ).xyz.toVar();
			const info = nodeInfo.element( i ).toVar();
			const s = info.x, deg = info.y;
			const degF = float( deg ).toVar();
			const L = vec3( 0 ).toVar();
			Loop( { start: uint( 0 ), end: deg, type: 'uint', condition: '<', name: 'k' }, ( { k } ) => {

				const j = adjList.element( s.add( k ) ).toVar();
				const d = nodePos.element( j ).xyz.sub( p ).toVar();
				const l = length( d ).toVar();
				If( l.lessThan( 1e-6 ), () => {

					d.assign( jitter( i, j ).mul( 1e-3 ) );
					l.assign( length( d ) );

				} );
				const dj = float( nodeInfo.element( j ).y ).toVar();
				const strength = U.linkStrength.div( min( degF, dj ) );
				const bias = dj.div( degF.add( dj ) );
				L.addAssign( d.mul( l.sub( U.linkDistance ).div( l ).mul( strength ).mul( bias ) ) );

			} );

			F.addAssign( L );
			F.subAssign( p.mul( U.gravity ) );
			vf.assign( v.add( F.mul( U.alpha ) ).mul( float( 1 ).sub( U.velocityDecay ) ) );

		} );

		If( b.greaterThan( 0 ), () => {

			const st = nodeState.element( i );
			const scale = mix( mix( float( 1 ), float( 0.75 ), st.y ), float( 0.5 ), st.x );
			const target = nodeTarget.element( i ).xyz.mul( sphereRadius.mul( scale ) );
			vs.assign( v.mul( U.lam2 ).sub( p.sub( target ).mul( U.springK ) ) );

		} );

		nodeVel.element( i ).assign( vec4( mix( vf, vs, b ), 0 ) );

	} )().compute( N, WG() ).setName( 'layout.velocity' );

}

/**
 * Position update: x += clamp(v, maxDisp) − centering·(1 − layoutBlend)·rootCentroid.
 * Pinned nodes (w = 1) keep their position, the dragged node is held, then placed on the drag
 * target; both get zero velocity.
 */
export function positionKernel( { buffers, U, N, root } ) {

	const { nodePos, nodeVel } = buffers;
	const clusterPos = readOnly( buffers.clusterPos );
	const { layoutBlend } = buffers.uniforms;
	return Fn( () => {

		const i = instanceIndex;
		const pos = nodePos.element( i ).toVar();
		const p = pos.xyz.toVar();
		const v = nodeVel.element( i ).xyz.toVar();
		If( int( i ).equal( U.dragIndex ), () => {

			// grabbed: held in place until the first drag target arrives, then placed on it
			If( U.dragHasTarget.greaterThan( 0.5 ), () => {

				p.assign( U.dragTarget );

			} );
			v.assign( vec3( 0 ) );

		} ).ElseIf( pos.w.greaterThan( 0.5 ), () => {

			v.assign( vec3( 0 ) );

		} ).Else( () => {

			const len = length( v );
			If( len.greaterThan( U.maxDisp ), () => {

				v.mulAssign( U.maxDisp.div( len ) );

			} );
			const center = clusterPos.element( uint( root ) ).xyz;
			p.addAssign( v.sub( center.mul( U.centering.mul( float( 1 ).sub( layoutBlend ) ) ) ) );

		} );

		nodePos.element( i ).assign( vec4( p, pos.w ) );
		nodeVel.element( i ).assign( vec4( v, 0 ) );

	} )().compute( N, WG() ).setName( 'layout.position' );

}

/**
 * Applies queued commands: unpin-all flag, then up to U.cmds.length (index, pinned) entries in
 * order, then an explicit move of U.moveIndex to U.dragTarget (drop / drag while not ticking).
 */
export function commandKernel( { buffers, U, N } ) {

	const { nodePos, nodeVel } = buffers;
	return Fn( () => {

		const i = int( instanceIndex ).toVar();
		const pos = nodePos.element( instanceIndex ).toVar();
		const w = pos.w.toVar();
		const p = pos.xyz.toVar();
		const touched = float( 0 ).toVar();
		If( U.unpinAll.greaterThan( 0.5 ), () => {

			w.assign( 0 );

		} );
		Loop( { start: uint( 0 ), end: U.cmdCount, type: 'uint', condition: '<', name: 'k' }, ( { k } ) => {

			const c = U.cmds.element( k );
			If( int( c.x ).equal( i ), () => {

				w.assign( c.y );
				touched.assign( 1 );

			} );

		} );
		If( i.equal( U.moveIndex ), () => {

			p.assign( U.dragTarget );
			touched.assign( 1 );

		} );
		nodePos.element( instanceIndex ).assign( vec4( p, w ) );
		If( touched.greaterThan( 0.5 ), () => {

			nodeVel.element( instanceIndex ).assign( vec4( 0 ) );

		} );

	} )().compute( N, WG() ).setName( 'layout.commands' );

}

/** nodeState easing: xyz -> nodeStateTarget.xyz, w -> (i == hoverIndex); snaps when close. */
export function stateKernel( { buffers, U, N } ) {

	const { nodeState, nodeColor } = buffers;
	const nodeStateTarget = readOnly( buffers.nodeStateTarget );
	const nodeColorTarget = readOnly( buffers.nodeColorTarget );
	const { hoverIndex } = buffers.uniforms;
	return Fn( () => {

		const i = instanceIndex;
		const s = nodeState.element( i ).toVar();
		const tg = nodeStateTarget.element( i ).toVar();
		const t = vec4( tg.xyz, select( int( i ).equal( hoverIndex ), float( 1 ), float( 0 ) ) ).toVar();
		const n = mix( s, t, U.stateEase ).toVar();
		const snap = vec4( 1 ).sub( step( vec4( 1e-3 ), abs( n.sub( t ) ) ) );
		nodeState.element( i ).assign( mix( n, t, snap ) );

		// color + radius factor follow their targets (palette / metric lens changes animate)
		const c = nodeColor.element( i ).toVar();
		const ct = nodeColorTarget.element( i ).toVar();
		const cn = mix( c, ct, U.colorEase ).toVar();
		const csnap = vec4( 1 ).sub( step( vec4( 1e-3 ), abs( cn.sub( ct ) ) ) );
		nodeColor.element( i ).assign( mix( cn, ct, csnap ) );

	} )().compute( N, WG() ).setName( 'layout.state' );

}
