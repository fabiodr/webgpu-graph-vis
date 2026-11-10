// Kernels, CPU-side structures and CPU mirrors of the GPU force layout (see ForceLayout.js for
// the model, its parameters and the scheduling).
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
//  rewriting one N-sized permutation (`nodeOrder`).
//
//  Masses. Every node has the mass m_i = (1 + deg_i / avgDeg)^p (p = hub spacing, 0..2), stored as
//  the fixed-point integer round(256·m_i) (MASS_ONE = 256), so cluster masses are exact integer
//  sums whatever the order of summation. For p = 1 the total is 2·256·N (uint32 holds 8 million
//  nodes); for p > 1 a hub can dominate it (star with 100k leaves, p = 2: 6·10¹¹), so setMasses
//  saturates the largest masses at the level that keeps the total below 2³² (no wrap-around; the
//  hub then repels less than the exponent asks for). The GPU converts the integers to float32
//  (exact up to 2²⁴, relative error ≤ 6·10⁻⁸ above).
//  Tree entries carry the mass of their subtree. Masses of real clusters are static; the
//  members of virtual clusters (and the node behind every slot) change at resorts, so the resort
//  recomputes the masses of the touched tree range (`updateTreeMasses`) and the layout uploads
//  that range of `tree` and `vInfo` together with `nodeOrder`.
//
//  Index spaces (private to the layout):
//    slots     0..N-1   preorder leaf order of the tree; nodeOrder[slot] = node index
//    bh ids    0..C-1 real clusters (same ids as prepare.js), C..C+V-1 virtual clusters,
//              sorted by virtual height ascending
//    tree      0..TT-1  preorder, uvec4 (ref, skip, mass, count); ref < N: slot, else N + bh id;
//              mass = fixed-point mass of the subtree, count = nodes in the subtree
//
//  Private storage buffers (createPrivateBuffers; "ro" = only the CPU writes them):
//    tree       uvec4 TT    ro  BH tree (ref, skip, count static; mass rewritten at resorts)
//    slotTree   uint  N     ro  slot -> tree index (static)
//    nodeOrder  uint  N     ro  slot -> node (rewritten by resorts)
//    vInfo      uvec4 V     ro  virtual clusters: (start, count, height, mass). height 1: children
//                               are slots start..start+count-1; otherwise vChildList[start..]
//    vChildList uint        ro  children (bh ids) of virtual clusters of height > 1
//    rInfo      uvec4 C     ro  real clusters: (start, count, kind, mass) into rChildList;
//                               kind 0: node ids, 1: bh ids of the top virtual clusters of a
//                               split height-1 cluster, 2: cluster ids (height > 1)
//    rChildList uint        ro  children of real clusters
//    adjLinks   uint  2E    ro  CSR neighbour lists (same order as graph.adjList), packed as
//                               neighbour | min(lca level, 15) << 28
//    bhCluster  vec4  C+V       centre of mass xyz, bounding radius w, real and virtual clusters
//    treePos    vec4  TT        tree-ordered copy: node xyz (w = 0) / cluster centre + radius
//    zMem       float N         z of every node when the flat mode was entered (see positionKernel)
//
// ─── Kernels (TSL compute nodes; one thread per element, 64-wide workgroups) ────────────────
//  kernel                    per          storage bindings
//  aggregateVirtualKernel    v. cluster   nodePos, nodeInfo, nodeOrder | vChildList, vInfo, bhCluster      ≤ 5
//  aggregateRealKernel       cluster      nodePos, nodeInfo, vInfo, rInfo, rChildList, clusterPos, bhCluster  ≤ 7
//  flattenKernel             tree entry   nodePos, nodeOrder, tree, bhCluster, treePos                    5
//  repulsionKernel           slot         tree, slotTree, nodeOrder, treePos, nodeForce                   5
//  linksKernel               node         nodeForce, nodePos, nodeVel, nodeInfo, adjLinks, clusterPos     6
//  velocityKernel            node         nodeForce, nodePos, nodeTarget, nodeState, nodeVel              5
//  positionKernel            node         nodeVel, nodePos, clusterPos, zMem                              4
//  commandKernel             node         nodePos, nodeVel                                                2
//  zSnapshotKernel           node         nodePos, zMem                                                   2
//  stateKernel               node         nodeState, nodeStateTarget, nodeColor, nodeColorTarget          4
//  Every kernel stays at 7 storage buffers or fewer (WebGPU default / mobile limit: 8 per stage).
//  Buffers a kernel only reads are bound through read-only views (readOnly()). All passes are
//  gathers: a thread reads other elements of the previous state and writes only its own.
//
// ─── CPU mirrors ────────────────────────────────────────────────────────────────────────────
//  aggregateCPU, repulsionCPU, linksCPU and LayoutMirror (the whole tick: velocity, position,
//  flat lock, resort) replicate the GPU math on Float32Arrays; AlphaSchedule, ScaleController and
//  flatTuning are the CPU-side controllers shared by ForceLayout and the mirror. They back the unit tests,
//  the per-graph work estimate (estimateVisits) and the dev page's GPU-vs-CPU checks.

import {
	Fn, If, Loop, Break, instanceIndex, uint, int, float, vec3, vec4, dot, max, min, mix,
	length, select, step, abs, instancedArray, storage,
} from 'three/tsl';

export const DEFAULT_LEAF_SIZE = 16;
export const DEFAULT_FANOUT = 8;

/** Fixed-point scale of node and cluster masses. */
export const MASS_ONE = 256;
/** adjLinks packing: neighbour index in the low bits, LCA level of the edge above LEVEL_SHIFT. */
export const LEVEL_SHIFT = 28;
export const LEVEL_MAX = 15;
export const NODE_MASK = ( 1 << LEVEL_SHIFT ) - 1;

/** Fixed constants of the model (see ForceLayout.js; not exposed as parameters). */
export const MODEL = {
	rho: 0.5, // share of the repulsion stiffness in the Jacobi preconditioner
	alphaMin: 0.001,
	holdShare: 0.3, // plateau = 30% of the tick budget (60 of 200 ticks)
	heatDecay: 0.92, // per tick
	heatSpread: 0.6, // per hop
	heatCut: 1e-4,
	idleDecay: 0.985, // alphaDrag per tick while the pointer is down but does not move
	heatTail: 120, // ticks the layout keeps running after a release
	scaleGain: 0.06, // auto-scale: f = ext^(−gain) per tick …
	scaleMin: 0.985, scaleMax: 1.015, // … clamped to ±1.5%
	// … while alpha is above this. The research froze the scale below alpha 0.05; scaling is an
	// exact symmetry of the model, so it can run almost to the end, which keeps the extent on
	// target after partial reheats too (parameter changes at alpha 0.3: 0.91–1.27 with 0.05,
	// 0.995–1.012 with 0.002; metrics of a full run unchanged)
	scaleMinAlpha: 0.002,
	// pinned nodes are not scaled, so the scale is measured on the free nodes; it is frozen when
	// more than this share of the nodes is pinned (the free rest could only follow by distortion)
	scaleMaxPinned: 0.25,
	// scaling is a symmetry of the model, so it needs ticks but no heat: while the extent is still
	// off by more than scaleWaitExt (|ln ext|) the schedule waits at alpha scaleWaitAlpha instead
	// of cooling to rest, for at most scaleWaitTicks ticks per reheat (a parameter change that
	// moves the natural size several times, e.g. hub spacing 2 -> 0 on a star, needs more ticks
	// at ±1.5% than a partial reheat has)
	scaleWaitExt: 0.04, scaleWaitAlpha: 0.02, scaleWaitTicks: 150,
	jitter: 0.02, // coincident nodes are pushed apart as if they were jitter·u·hash apart
	// flat mode (see flatTuning): factors at full strength, and the node counts between which
	// the strength rises from 0 to 1 (logarithmically)
	flatBeta: 8, flatLambdaPower: 2, flatKappa: 2, flatFrom: 300, flatFull: 2000,
};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// CPU: structure
// ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Builds the extended Barnes–Hut structure for a PreparedGraph. Masses are filled for hub
 * spacing p = `massPower` (see setMasses).
 * @returns {{ N, C, V, TT, tree: Uint32Array, slotTree: Uint32Array, nodeOrder: Uint32Array,
 *   nodeMass: Uint32Array, vInfo: Uint32Array, vChildList: Uint32Array,
 *   vHeightRanges: Array<{height,start,count}>, rInfo: Uint32Array, rChildList: Uint32Array,
 *   groups: Array<{cluster,lo,hi,parts,t0,t1}>, leafSize, fanout }}
 */
export function buildStructure( g, { leafSize = DEFAULT_LEAF_SIZE, fanout = DEFAULT_FANOUT, massPower = 1 } = {} ) {

	const { N, C, root, clusterChildStart, clusterChildCount, childRefs, clusterHeight, clusterMass } = g;
	leafSize = Math.max( 2, leafSize | 0 );
	fanout = Math.max( 2, fanout | 0 );

	const tree = [];
	const nodeOrder = new Uint32Array( N );
	let slot = 0;
	// virtual clusters before height sorting: { lo, count, height, children: [vtmp ids] | null }
	const vtmp = [];
	// oversized leaf-owning clusters: { lo, hi, parts, t0, t1 }; parts is the shape for resorting,
	// [t0, t1) the tree range of the parts
	const groups = [];
	const vTreeIndex = []; // vtmp id -> tree index (to patch the ref after sorting by height)

	const emitSlots = ( nodes ) => {

		for ( const node of nodes ) {

			nodeOrder[ slot ] = node;
			const idx = tree.length / 4;
			tree.push( slot, idx + 1, 0, 1 );
			slot ++;

		}

	};

	// splits `nodes` (array) into virtual clusters; returns the shape: array of parts
	// { lo, hi, parts | null, t0, t1 } (slot range, sub-parts, tree range of the part's subtree)
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
			tree.push( 0, 0, 0, size ); // ref patched later
			const part = { lo: slot, hi: slot + size, parts: null, t0: idx, t1: 0 };
			if ( size > leafSize ) {

				const r = emitVirtual( sub );
				rec.children = r.ids;
				rec.height = 1 + Math.max( ...r.ids.map( ( c ) => vtmp[ c ].height ) );
				part.parts = r.shape;

			} else {

				emitSlots( sub );

			}

			tree[ 4 * idx + 1 ] = part.t1 = tree.length / 4;
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
		tree.push( ref, 0, 0, clusterMass[ c ] );
		const s = clusterChildStart[ c ], n = clusterChildCount[ c ];
		const leafOwner = clusterHeight[ c ] === 1;
		if ( leafOwner && n > leafSize ) {

			const lo = slot, t0 = tree.length / 4;
			const nodes = Array.from( childRefs.subarray( s, s + n ) );
			const r = emitVirtual( nodes );
			groups.push( { cluster: c, lo, hi: slot, parts: r.shape, top: r.ids, t0, t1: tree.length / 4 } );

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
			groups.push( { cluster: root, lo: 0, hi: slot, parts: r.shape, top: r.ids, t0: 0, t1: tree.length / 4 } );

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
		const last = vHeightRanges[ vHeightRanges.length - 1 ];
		if ( last && last.height === rec.height ) last.count ++;
		else vHeightRanges.push( { height: rec.height, start: k, count: 1 } );

	}

	for ( let v = 0; v < V; v ++ ) tree[ 4 * vTreeIndex[ v ] ] = N + C + finalId[ v ];

	// children of real clusters for aggregation. Height 1: nodes (kind 0), or for split clusters
	// their few top-level virtual clusters (kind 1), so no thread loops over thousands of nodes.
	// Higher clusters: their child clusters (kind 2).
	const rInfo = new Uint32Array( 4 * Math.max( 1, C ) );
	const rChild = [];
	const split = new Map( groups.map( ( gr ) => [ gr.cluster, gr ] ) );
	for ( let c = 0; c < C; c ++ ) {

		rInfo[ 4 * c ] = rChild.length;
		const s = clusterChildStart[ c ], n = clusterChildCount[ c ];
		const gr = split.get( c );
		if ( gr ) {

			for ( const v of gr.top ) rChild.push( C + finalId[ v ] );
			rInfo[ 4 * c + 2 ] = 1;

		} else if ( clusterHeight[ c ] === 1 ) {

			for ( let k = 0; k < n; k ++ ) rChild.push( childRefs[ s + k ] );

		} else {

			for ( let k = 0; k < n; k ++ ) rChild.push( childRefs[ s + k ] - N );
			rInfo[ 4 * c + 2 ] = 2;

		}

		rInfo[ 4 * c + 1 ] = rChild.length - rInfo[ 4 * c ];

	}

	for ( const gr of groups ) delete gr.top;

	const TT = tree.length / 4;
	const slotTree = new Uint32Array( Math.max( 1, N ) );
	for ( let k = 0; k < TT; k ++ ) if ( tree[ 4 * k ] < N ) slotTree[ tree[ 4 * k ] ] = k;
	const struct = {
		N, C, V, TT, leafSize, fanout, root,
		tree: Uint32Array.from( tree.length ? tree : [ 0, 1, 0, 0 ] ),
		slotTree,
		nodeOrder,
		nodeMass: new Uint32Array( Math.max( 1, N ) ),
		vInfo,
		vChildList: Uint32Array.from( vChild.length ? vChild : [ 0 ] ),
		vHeightRanges,
		rInfo,
		rChildList: Uint32Array.from( rChild.length ? rChild : [ 0 ] ),
		groups,
	};
	setMasses( struct, g, massPower );
	return struct;

}

/** Largest total fixed-point mass of a graph (cluster masses are uint32 sums). */
export const MASS_TOTAL_MAX = 0xffffffff;

/**
 * Fixed-point node mass round(256·(1 + deg/avgDeg)^p), at least 1 (a float64 integer: not yet
 * saturated, see setMasses).
 */
export function nodeMassFixed( degree, avgDegree, p = 1 ) {

	const m = p === 0 ? 1 : Math.pow( 1 + degree / Math.max( 1e-9, avgDegree ), p );
	return Math.max( 1, Math.round( MASS_ONE * m ) );

}

/**
 * Sets the node masses m_i = (1 + deg_i/avgDeg)^p of a structure (hub spacing p) and recomputes
 * every cluster mass. When the total would not fit in uint32, the largest masses are cut to the
 * highest common level that makes it fit (struct.massCap; Infinity when nothing was cut). The
 * caller re-uploads tree, vInfo, rInfo and nodeInfo.w.
 */
export function setMasses( struct, g, p = 1 ) {

	const N = struct.N;
	const avg = g.N ? ( 2 * g.E ) / g.N : 0;
	const mass = new Float64Array( N );
	let sum = 0;
	for ( let i = 0; i < N; i ++ ) sum += mass[ i ] = nodeMassFixed( g.degree[ i ], avg, p );
	let cap = Infinity;
	if ( sum > MASS_TOTAL_MAX ) {

		// water filling: the largest level L with Σ min(m_i, L) ≤ MASS_TOTAL_MAX
		const sorted = Float64Array.from( mass ).sort();
		let rest = MASS_TOTAL_MAX;
		for ( let k = 0; k < N; k ++ ) {

			if ( sorted[ k ] * ( N - k ) > rest ) {

				cap = Math.max( 1, Math.floor( rest / ( N - k ) ) );
				break;

			}

			rest -= sorted[ k ];

		}

	}

	for ( let i = 0; i < N; i ++ ) struct.nodeMass[ i ] = Math.min( mass[ i ], cap );
	struct.massCap = cap;
	struct.massPower = p;
	updateTreeMasses( struct );
	let total = 0;
	for ( let i = 0; i < struct.N; i ++ ) total += struct.nodeMass[ i ];
	struct.rInfo[ 4 * struct.root + 3 ] = total; // the root is not in the tree
	struct.totalMass = total;

}

/**
 * Recomputes the fixed-point masses of the tree entries [t0, t1) (a union of complete subtrees)
 * bottom-up from nodeMass and nodeOrder, and the mass fields of vInfo / rInfo of the clusters in
 * that range. O(entries).
 * @returns {{ vLo, vHi }} range of virtual cluster ids whose vInfo changed (vLo >= vHi: none)
 */
export function updateTreeMasses( struct, t0 = 0, t1 = struct.TT ) {

	const { N, C, tree, nodeOrder, nodeMass, vInfo, rInfo } = struct;
	let vLo = Infinity, vHi = - Infinity;
	for ( let k = t1 - 1; k >= t0; k -- ) {

		const ref = tree[ 4 * k ];
		if ( ref < N ) {

			tree[ 4 * k + 2 ] = nodeMass[ nodeOrder[ ref ] ];
			continue;

		}

		let m = 0;
		for ( let j = k + 1, end = tree[ 4 * k + 1 ]; j < end; j = tree[ 4 * j + 1 ] ) m += tree[ 4 * j + 2 ];
		tree[ 4 * k + 2 ] = m;
		const id = ref - N;
		if ( id >= C ) {

			vInfo[ 4 * ( id - C ) + 3 ] = m;
			if ( id - C < vLo ) vLo = id - C;
			if ( id - C + 1 > vHi ) vHi = id - C + 1;

		} else rInfo[ 4 * id + 3 ] = m;

	}

	return { vLo, vHi };

}

/**
 * Packed neighbour lists for the links pass: graph.adjList with the LCA level of every edge in
 * the top bits (neighbour | min(level, LEVEL_MAX) << LEVEL_SHIFT). Level = first hierarchy level
 * at which both ends share a cluster, D when they only share the root.
 */
export function packLinks( g ) {

	const { N, D, anc, adjStart, degree, adjList } = g;
	if ( N > NODE_MASK ) throw new Error( `packLinks: more than ${NODE_MASK} nodes are not supported` );
	const out = new Uint32Array( Math.max( 1, adjList.length ) );
	for ( let i = 0; i < N; i ++ ) {

		for ( let k = adjStart[ i ], e = k + degree[ i ]; k < e; k ++ ) {

			const j = adjList[ k ];
			let l = 0;
			while ( l < D && anc[ i * D + l ] !== anc[ j * D + l ] ) l ++;
			out[ k ] = ( j | ( Math.min( l, LEVEL_MAX ) << LEVEL_SHIFT ) ) >>> 0;

		}

	}

	return out;

}

/**
 * Re-partitions the nodes of oversized leaf clusters (groups[from..to)) from positions `pos4`
 * (4 floats per node) by recursive median bisection along the longest axis, consistent with the
 * fixed shape. Rewrites `struct.nodeOrder` and the masses of the touched tree range in place.
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

/** Top-level resort tasks ({ lo, hi, parts, t0, t1 }) of groups[from..to). */
export function resortTasks( struct, from = 0, to = struct.groups.length ) {

	return struct.groups.slice( from, to ).map( ( g ) => ( { lo: g.lo, hi: g.hi, parts: g.parts, t0: g.t0, t1: g.t1 } ) );

}

/**
 * One unit of resort work: distributes the slots [task.lo, task.hi) over task.parts by median
 * bisection, then appends the sub-tasks (parts that are split further) to `queue`. Every
 * intermediate state is a valid permutation with matching masses, so the work can be spread over
 * frames.
 * @returns {{ changed, lo, hi, t0, t1, vLo, vHi }} changed slot range, tree range and range of
 *   virtual cluster ids whose masses were rewritten
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
	// the node sets of the direct parts are final now; their sub-parts are re-partitioned by later tasks
	for ( const p of task.parts ) if ( p.parts ) queue.push( { lo: p.lo, hi: p.hi, parts: p.parts, t0: p.t0 + 1, t1: p.t1 } );
	let changed = false;
	for ( let s = task.lo; s < task.hi && ! changed; s ++ ) changed = order[ s ] !== before[ s - task.lo ];
	let vLo = Infinity, vHi = - Infinity;
	if ( changed ) ( { vLo, vHi } = updateTreeMasses( struct, task.t0, task.t1 ) );
	return { changed, lo: task.lo, hi: task.hi, t0: task.t0, t1: task.t1, vLo, vHi };

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
// CPU: controllers shared by ForceLayout and the mirror
// ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Plateau-then-cool alpha schedule: alpha stays where reheat() put it for `hold` ticks, then
 * approaches `target` exponentially, reaching alphaMin after `coolTicks` ticks (from alpha 1).
 *   budget = hold + cool ticks of a full run (default 200 = 60 + 140)
 */
export class AlphaSchedule {

	constructor( { budget = 200, alphaMin = MODEL.alphaMin, alphaDecay = null } = {} ) {

		this.alphaMin = alphaMin;
		this._decayOverride = alphaDecay;
		this.alpha = 1;
		this.target = 0;
		this.waited = 0; // ticks spent waiting for the auto-scale since the last reheat
		this.budget = budget;
		this.reheat( 1 );

	}

	get budget() {

		return this._budget;

	}

	set budget( ticks ) {

		this._budget = Math.max( 10, Math.round( ticks ) );
		this.holdTicks = Math.round( MODEL.holdShare * this._budget );
		this.coolTicks = this._budget - this.holdTicks;
		this.decay = this._decayOverride ?? 1 - Math.pow( this.alphaMin, 1 / this.coolTicks );

	}

	/** Start (or restart) from `alpha`; the plateau is proportional to it (60 ticks at alpha 1). */
	reheat( alpha = 1 ) {

		this.alpha = Math.max( 0, Math.min( 1, alpha ) );
		this.hold = Math.round( this.holdTicks * this.alpha );
		this.waited = 0;

	}

	/**
	 * Called after step() in ticks in which the auto-scale is active but the extent is still off
	 * (ScaleController.off): keeps alpha from cooling below MODEL.scaleWaitAlpha, for at most
	 * MODEL.scaleWaitTicks ticks per reheat. Returns the alpha of this tick.
	 */
	wait() {

		if ( this.alpha >= MODEL.scaleWaitAlpha || this.target > 0 || this.alphaMin >= MODEL.scaleWaitAlpha ) return this.alpha;
		if ( this.waited >= MODEL.scaleWaitTicks ) return this.alpha;
		this.waited ++;
		this.alpha = MODEL.scaleWaitAlpha;
		return this.alpha;

	}

	/** Advance one tick; returns the alpha of this tick. */
	step() {

		if ( this.hold > 0 && this.target <= 0 ) this.hold --;
		else this.alpha += ( this.target - this.alpha ) * this.decay;
		return this.alpha;

	}

	get cooled() {

		return this.alpha < this.alphaMin && this.target < this.alphaMin;

	}

}

/**
 * Auto-scale controller. The model is homogeneous of degree 1 in (positions, u): scaling all
 * positions, velocities and the kernel unit u by f keeps the state, so the size of the layout is
 * controlled by a slow per-tick factor instead of a gravity constant:
 *   ext = rms radius / target rms;  f = clamp( ext^(−gain), 0.985, 1.015 ) per tick
 * Measurements come from position readbacks that are several ticks old: measure() corrects the
 * measured extent by the scale applied since the readback was requested (logScale then / now).
 *   unit       kernel unit u (world units); starts at linkDistance
 *   f          factor to apply in the next tick (1 = inactive)
 *   logScale   Σ ln f over all ticks so far
 */
export class ScaleController {

	constructor( unit = 30 ) {

		this.reset( unit );

	}

	reset( unit ) {

		this.unit = unit;
		this.f = 1;
		this.logScale = 0;
		this.ext = 1; // last estimate of rms / target
		this.measured = false;

	}

	/**
	 * New measurement: rms radius of positions that are as old as `logScaleThen`, target rms.
	 * Returns the corrected extent estimate.
	 */
	measure( rms, targetRms, logScaleThen = this.logScale ) {

		if ( ! ( rms > 0 ) || ! ( targetRms > 0 ) || ! Number.isFinite( rms ) ) return this.ext;
		this.ext = ( rms / targetRms ) * Math.exp( this.logScale - logScaleThen );
		this.measured = true;
		return this.ext;

	}

	/** True while the (estimated) extent is off target by more than MODEL.scaleWaitExt. */
	get off() {

		return this.measured && Math.abs( Math.log( this.ext ) ) > MODEL.scaleWaitExt;

	}

	/** Factor for the next tick (and bookkeeping). `active` false freezes the scale. */
	step( active ) {

		let f = 1;
		if ( active && this.measured ) {

			f = Math.min( MODEL.scaleMax, Math.max( MODEL.scaleMin, Math.pow( this.ext, - MODEL.scaleGain ) ) );
			this.unit *= f;
			this.logScale += Math.log( f );
			this.ext *= f; // what the measurement would be after this tick

		}

		this.f = f;
		return f;

	}

}

/**
 * rms radius of positions (stride 4) about their centroid. With `freeOnly`, pinned nodes
 * (w > 0.5, the pin flag of nodePos) are left out: the auto-scale steers the nodes it can move.
 */
export function rmsRadius( pos4, N, freeOnly = false ) {

	let cx = 0, cy = 0, cz = 0, n = 0;
	for ( let i = 0; i < N; i ++ ) {

		if ( freeOnly && pos4[ 4 * i + 3 ] > 0.5 ) continue;
		cx += pos4[ 4 * i ]; cy += pos4[ 4 * i + 1 ]; cz += pos4[ 4 * i + 2 ];
		n ++;

	}

	if ( n === 0 ) return 0;
	cx /= n; cy /= n; cz /= n;
	let q = 0;
	for ( let i = 0; i < N; i ++ ) {

		if ( freeOnly && pos4[ 4 * i + 3 ] > 0.5 ) continue;
		const dx = pos4[ 4 * i ] - cx, dy = pos4[ 4 * i + 1 ] - cy, dz = pos4[ 4 * i + 2 ] - cz;
		q += dx * dx + dy * dy + dz * dz;

	}

	return Math.sqrt( q / n );

}

/**
 * Radius of the disc the flat (2D) mode fills. A plane has less room than a ball: at the 3D
 * radius R the nearest neighbour of a node is 1.5–2.8× closer than in 3D, and equal crowding
 * would need R·0.7·N^(1/6), which makes nodes tiny on screen (their size is fixed in world units).
 * The compromise R·max(1.3, 0.5·N^(1/6)) keeps the median nearest-neighbour gap at 3–6 node
 * diameters (CPU mirror, lesmis … ws-2k; see ForceLayout.js for the numbers). The extent alone
 * does not keep nodes from overlapping (a disc of 2.5·R still has overlapping pairs on lfr-2k):
 * that is the job of flatTuning.
 */
export function flatRadius( R, N ) {

	return R * Math.max( 1.3, 0.5 * Math.pow( Math.max( 1, N ), 1 / 6 ) );

}

/**
 * Flat-mode adaptation of the model parameters. With the user's 3D parameters the auto-scale
 * ends at a kernel unit of 1.4–2 link distances in the plane (0.7–1 in space), so the nodes of a
 * dense community sit inside the linear core of the bounded repulsion, where a close pair feels
 * nothing that separates it: node pairs then overlap (CPU mirror, three runs each: 15–22 pairs on
 * hsbm-5k, 13–23 on ba-5k, 150–167 on lfr-10k, 2–5 on lfr-2k; none below ~1000 nodes). A
 * stronger repulsion shortens the unit and evens the spacing out, but alone it blurs the
 * communities (β × 4: silhouette 0.32 → 0.25 on hsbm-5k), so the cluster terms are raised with it:
 *   β × 8^t,   λ → λ^(1 + t),   κ × 2^t,     t = flat · clamp( ln(N/300) / ln(2000/300), 0, 1 )
 * (flat = the eased 0..1 blend of the mode). Measured at t = 1: 0 overlapping pairs in every run
 * on lfr-2k, hsbm-5k, ba-5k, rgg-3k, ws-2k and lattice-12, 3–9 on lfr-10k; silhouette and
 * neighbourhood preservation equal or higher than without, stress +3–5%. Graphs below 300 nodes
 * have room and keep the plain parameters (their stress would rise 14–39% for nothing). λ = 1
 * stays 1 and κ = 0 stays 0, so the 'graph-only' preset keeps its meaning.
 * @returns {{ t, beta, lambdaPower, kappa }} factor on β, exponent on λ, factor on κ
 */
export function flatTuning( N, flat = 1 ) {

	const ramp = Math.log( Math.max( 1, N ) / MODEL.flatFrom ) / Math.log( MODEL.flatFull / MODEL.flatFrom );
	const t = Math.max( 0, Math.min( 1, flat ) ) * Math.max( 0, Math.min( 1, ramp ) );
	return { t, beta: Math.pow( MODEL.flatBeta, t ), lambdaPower: 1 + ( MODEL.flatLambdaPower - 1 ) * t, kappa: Math.pow( MODEL.flatKappa, t ) };

}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// CPU mirrors
// ─────────────────────────────────────────────────────────────────────────────────────────────

const padCPU = ( r ) => r * ( 1 + 1e-5 ) + 1e-4; // mirrors pad() of the GPU kernels

/**
 * CPU mirror of the aggregation passes: returns bhCluster (Float32Array 4·(C+V): centre of mass
 * and conservative bounding radius; entries 0..C-1 equal clusterPos).
 */
export function aggregateCPU( g, struct, pos4, out = null ) {

	const { C, V, vInfo, vChildList, nodeOrder, nodeMass, rInfo, rChildList } = struct;
	out = out || new Float32Array( 4 * ( C + V ) );

	// weighted centre + radius of `count` children; child( k ) -> [ array, offset, weight, radius ]
	const reduce = ( count, child, o ) => {

		let x = 0, y = 0, z = 0, m = 0;
		for ( let k = 0; k < count; k ++ ) {

			const [ src, s, w ] = child( k );
			x += src[ s ] * w; y += src[ s + 1 ] * w; z += src[ s + 2 ] * w; m += w;

		}

		m = Math.max( m, 1e-9 ); x /= m; y /= m; z /= m;
		let r = 0;
		for ( let k = 0; k < count; k ++ ) {

			const [ src, s, , rad ] = child( k );
			const d = Math.hypot( src[ s ] - x, src[ s + 1 ] - y, src[ s + 2 ] - z ) + rad;
			if ( d > r ) r = d;

		}

		out[ o ] = x; out[ o + 1 ] = y; out[ o + 2 ] = z; out[ o + 3 ] = padCPU( r );

	};

	const nodeChild = ( n ) => [ pos4, 4 * n, nodeMass[ n ], 0 ];
	const virtualChild = ( id ) => [ out, 4 * id, vInfo[ 4 * ( id - C ) + 3 ], out[ 4 * id + 3 ] ];
	const realChild = ( id ) => [ out, 4 * id, rInfo[ 4 * id + 3 ], out[ 4 * id + 3 ] ];
	for ( let v = 0; v < V; v ++ ) {

		const start = vInfo[ 4 * v ], count = vInfo[ 4 * v + 1 ], h = vInfo[ 4 * v + 2 ];
		reduce( count, h === 1 ? ( k ) => nodeChild( nodeOrder[ start + k ] ) : ( k ) => virtualChild( vChildList[ start + k ] ), 4 * ( C + v ) );

	}

	for ( const { start, count } of g.heightRanges ) {

		for ( let c = start; c < start + count; c ++ ) {

			const s = rInfo[ 4 * c ], n = rInfo[ 4 * c + 1 ], kind = rInfo[ 4 * c + 2 ];
			reduce( n, kind === 0 ? ( k ) => nodeChild( rChildList[ s + k ] ) : kind === 1 ? ( k ) => virtualChild( rChildList[ s + k ] ) : ( k ) => realChild( rChildList[ s + k ] ), 4 * c );

		}

	}

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
 * CPU mirror of repulsionKernel: F_i = − Σ β·m_i·m_e·d / (1 + |d|²/u²)², S_i = Σ β·m_i·m_e / (…)².
 * Returns { force: Float32Array 4N (node-indexed like nodeForce: xyz force, w = S),
 * visits: Uint32Array N (loop iterations per node) }. `slots` restricts the work to a subset.
 */
export function repulsionCPU( struct, pos4, bhCluster, { theta = 0.8, beta = 8, unit = 30, maxIterations = 8192, slots = null, force = null } = {} ) {

	const { N, TT, tree, nodeOrder, slotTree } = struct;
	const theta2 = theta * theta, iu2 = 1 / ( unit * unit );
	const soft = maxIterations >>> 1;
	const jit = MODEL.jitter * unit;
	force = force || new Float32Array( 4 * N );
	const visits = new Uint32Array( N );
	const count = slots ? slots.length : N;
	for ( let q = 0; q < count; q ++ ) {

		const t = slots ? slots[ q ] : q;
		const node = nodeOrder[ t ], self = slotTree[ t ];
		const cm = ( beta * tree[ 4 * self + 2 ] ) / ( MASS_ONE * MASS_ONE );
		const px = pos4[ 4 * node ], py = pos4[ 4 * node + 1 ], pz = pos4[ 4 * node + 2 ];
		let fx = 0, fy = 0, fz = 0, S = 0;
		let k = 0, it = 0;
		for ( ; it < maxIterations; it ++ ) {

			if ( k >= TT ) break;
			const ref = tree[ 4 * k ];
			let dx, dy, dz, r2 = 0;
			if ( ref < N ) {

				const o = 4 * nodeOrder[ ref ];
				dx = pos4[ o ] - px; dy = pos4[ o + 1 ] - py; dz = pos4[ o + 2 ] - pz;

			} else {

				const o = 4 * ( ref - N );
				dx = bhCluster[ o ] - px; dy = bhCluster[ o + 1 ] - py; dz = bhCluster[ o + 2 ] - pz;
				r2 = bhCluster[ o + 3 ] * bhCluster[ o + 3 ];

			}

			let l2 = dx * dx + dy * dy + dz * dz;
			if ( r2 < l2 && ( r2 <= theta2 * l2 || it >= soft ) ) {

				const b = 1 + l2 * iu2;
				const w = ( cm * tree[ 4 * k + 2 ] ) / ( b * b );
				fx -= dx * w; fy -= dy * w; fz -= dz * w; S += w;
				k = tree[ 4 * k + 1 ];

			} else {

				if ( l2 < 1e-12 && ref < N && k !== self ) {

					// a distinct node at exactly our position (rare): push apart deterministically
					const j = jitterCPU( node, nodeOrder[ ref ] );
					dx = j[ 0 ] * jit; dy = j[ 1 ] * jit; dz = j[ 2 ] * jit;
					l2 = dx * dx + dy * dy + dz * dz;
					const b = 1 + l2 * iu2;
					const w = ( cm * tree[ 4 * k + 2 ] ) / ( b * b );
					fx -= dx * w; fy -= dy * w; fz -= dz * w; S += w;

				}

				k ++;

			}

		}

		force[ 4 * node ] = fx; force[ 4 * node + 1 ] = fy; force[ 4 * node + 2 ] = fz; force[ 4 * node + 3 ] = S;
		visits[ node ] = it;

	}

	return { force, visits };

}

/**
 * CPU mirror of linksKernel. Reads force4 = (F_rep, S) and writes (F/M, heat) in place:
 *   F = F_rep + Σ_j w_ij·(x_j − x_i) + κ·(c_A(i) − x_i) − g·x_i,   w_ij = lcaWeights[level]
 *   M = 1 + κ + g + Σ_j w_ij + ρ·S
 *   heat = max(heatDecay·heat_i, heatSpread·max_j heat_j), 1 for the heat source, 0 below 1e-4
 * `links` = packLinks( g ); `clusterPos` = aggregateCPU output; `vel4.w` = heat of the last tick.
 */
export function linksCPU( g, links, pos4, force4, clusterPos, vel4, { lcaWeights, kappa = 0.5, gravity = 0.03, heatIndex = - 1, heatActive = true } = {} ) {

	const { N, D, anc, adjStart, degree } = g;
	const kap = D > 0 ? kappa : 0;
	const heat = new Float32Array( N );
	for ( let i = 0; i < N; i ++ ) {

		const o = 4 * i;
		const px = pos4[ o ], py = pos4[ o + 1 ], pz = pos4[ o + 2 ];
		let fx = force4[ o ], fy = force4[ o + 1 ], fz = force4[ o + 2 ], W = 0, hn = 0;
		for ( let k = adjStart[ i ], e = k + degree[ i ]; k < e; k ++ ) {

			const a = links[ k ], j = 4 * ( a & NODE_MASK );
			const w = lcaWeights[ a >>> LEVEL_SHIFT ];
			fx += w * ( pos4[ j ] - px ); fy += w * ( pos4[ j + 1 ] - py ); fz += w * ( pos4[ j + 2 ] - pz );
			W += w;
			if ( vel4[ j + 3 ] > hn ) hn = vel4[ j + 3 ];

		}

		if ( kap > 0 ) {

			const c = 4 * anc[ i * D ];
			fx += kap * ( clusterPos[ c ] - px ); fy += kap * ( clusterPos[ c + 1 ] - py ); fz += kap * ( clusterPos[ c + 2 ] - pz );

		}

		fx -= gravity * px; fy -= gravity * py; fz -= gravity * pz;
		const M = 1 + kap + gravity + W + MODEL.rho * force4[ o + 3 ];
		let h = 0;
		if ( heatActive ) {

			h = Math.max( MODEL.heatDecay * vel4[ o + 3 ], MODEL.heatSpread * hn );
			if ( i === heatIndex ) h = 1;
			if ( h < MODEL.heatCut ) h = 0;

		}

		force4[ o ] = fx / M; force4[ o + 1 ] = fy / M; force4[ o + 2 ] = fz / M;
		heat[ i ] = h;

	}

	for ( let i = 0; i < N; i ++ ) force4[ 4 * i + 3 ] = heat[ i ];
	return force4;

}

/**
 * CPU mirror of the whole layout: the same passes in the same order on Float32Arrays, with the
 * same controllers (AlphaSchedule, ScaleController) as ForceLayout. Synchronous resorts, and an
 * auto-scale measurement that is taken every tick unless `scaleLag` / `scalePeriod` emulate the
 * readback delay. Used by the tests and for offline tuning; not used at run time.
 *
 *   const sim = new LayoutMirror( graph, { clusterSeparation, cohesion, hubSpacing, layoutSize,
 *     damping, repulsion, gravity, theta, budget, flat, flatTune = true, scaleLag, scalePeriod,
 *     resortEvery } );
 *   sim.run();                       until cooled (budget ticks)
 *   sim.tick();                      one tick
 *   sim.pos (Float32Array 4N, w = pinned), sim.vel (xyz, w = heat), sim.alpha, sim.scale.unit
 *   sim.drag = { index, target: [x, y, z] } | null; sim.alphaDrag; sim.pin( i, bool )
 *   sim.setTargets( positions4, blend )   critically damped spring toward absolute targets
 */
export class LayoutMirror {

	constructor( g, options = {} ) {

		const o = this.options = {
			clusterSeparation: 0.7, cohesion: 0.5, hubSpacing: 1, layoutSize: 1, damping: 0.3, repulsion: 8,
			gravity: 0.03, centering: 0.3, theta: 0.8, budget: 200, maxIterations: 8192,
			leafSize: DEFAULT_LEAF_SIZE, fanout: DEFAULT_FANOUT, resortEvery: 5, flat: false,
			scaleLag: 0, scalePeriod: 1, autoScale: true, springLambda: Math.exp( - 8 / 60 ), flatTune: true,
			...options,
		};
		this.g = g;
		const N = this.N = g.N;
		this.struct = buildStructure( g, { leafSize: o.leafSize, fanout: o.fanout, massPower: o.hubSpacing } );
		this.links = packLinks( g );
		this.pos = new Float32Array( o.positions || g.initPos );
		this.vel = new Float32Array( 4 * N );
		this.force = new Float32Array( 4 * N );
		this.zMem = new Float32Array( N );
		this.flat = o.flat ? 1 : 0;
		if ( this.flat ) for ( let i = 0; i < N; i ++ ) {

			this.zMem[ i ] = this.pos[ 4 * i + 2 ];
			this.pos[ 4 * i + 2 ] = 0;

		}

		resortStructure( this.struct, this.pos );
		this.bh = aggregateCPU( g, this.struct, this.pos );
		this.schedule = new AlphaSchedule( { budget: o.budget } );
		this.L = g.linkDistance ?? 30;
		this.scale = new ScaleController( this.L );
		this.maxDisp = Math.max( 2 * this.L, 0.05 * g.layoutRadius );
		this.ticks = 0;
		this.drag = null;
		this.alphaDrag = 0;
		this.heatActive = false;
		this.pinned = 0;
		this.targets = null;
		this.targetBlend = 0;
		this._hist = [];
		this.lcaWeights = new Float32Array( LEVEL_MAX + 1 );
		this.setParams( {} );

	}

	setParams( p ) {

		Object.assign( this.options, p );
		// effective parameters: the user's, adapted in the flat mode (flatTuning)
		const ft = flatTuning( this.N, this.options.flatTune ? this.flat : 0 );
		this.beta = this.options.repulsion * ft.beta;
		this.kappa = this.options.cohesion * ft.kappa;
		for ( let l = 0; l <= LEVEL_MAX; l ++ ) this.lcaWeights[ l ] = Math.pow( this.options.clusterSeparation, l * ft.lambdaPower );

	}

	get alpha() {

		return this.schedule.alpha;

	}

	/** rms radius the auto-scale steers to (3D ball of radius R, or the flat disc). */
	get targetRms() {

		const R = this.g.layoutRadius * this.options.layoutSize;
		return this.flat > 0.5 ? Math.SQRT1_2 * ( this.options.flatRadius ?? flatRadius( R, this.N ) ) : Math.sqrt( 0.6 ) * R;

	}

	pin( i, on = true ) {

		const was = this.pos[ 4 * i + 3 ] > 0.5;
		this.pos[ 4 * i + 3 ] = on ? 1 : 0;
		if ( was !== on ) this.pinned += on ? 1 : - 1;

	}

	setTargets( positions4, blend = 1 ) {

		this.targets = positions4;
		this.targetBlend = positions4 ? blend : 0;

	}

	tick() {

		const { g, struct, pos, vel, force, options: o, N } = this;
		let alpha = this.schedule.step();
		const dragIndex = this.drag ? this.drag.index : - 1;

		// auto-scale: measurement (optionally lagged) and the factor of this tick
		const active = o.autoScale && alpha > MODEL.scaleMinAlpha && dragIndex < 0 && this.pinned <= MODEL.scaleMaxPinned * N && this.targetBlend === 0;
		this._hist.push( [ rmsRadius( pos, N, true ), this.scale.logScale ] );
		if ( this._hist.length > o.scaleLag + 1 ) this._hist.shift();
		if ( this.ticks % Math.max( 1, o.scalePeriod ) === 0 ) {

			const h = this._hist[ 0 ];
			this.scale.measure( h[ 0 ], this.targetRms, h[ 1 ] );

		}

		if ( active && this.scale.off ) alpha = this.schedule.wait(); // the scale needs more ticks

		const f = this.scale.step( active );
		const unit = this.scale.unit / f; // the passes of this tick still see the old positions

		if ( struct.groups.length && this.ticks % Math.max( 1, o.resortEvery ) === 0 ) {

			if ( resortStructure( struct, pos ).changed ) aggregateCPU( g, struct, pos, this.bh );

		}

		const b = this.targets ? this.targetBlend : 0;
		if ( b < 1 ) {

			repulsionCPU( struct, pos, this.bh, { theta: o.theta, beta: this.beta, unit, maxIterations: o.maxIterations, force } );
			linksCPU( g, this.links, pos, force, this.bh, vel, { lcaWeights: this.lcaWeights, kappa: this.kappa, gravity: o.gravity, heatIndex: dragIndex, heatActive: this.heatActive || dragIndex >= 0 } );

		} else force.fill( 0 );

		// velocity
		const keep = 1 - o.damping;
		const lam = o.springLambda, lam2 = lam * lam, springK = ( 1 - lam ) * ( 1 - lam );
		for ( let i = 0; i < N; i ++ ) {

			const q = 4 * i;
			const heat = force[ q + 3 ];
			const a = Math.max( alpha, this.alphaDrag * heat );
			for ( let c = 0; c < 3; c ++ ) {

				const vf = ( vel[ q + c ] + a * force[ q + c ] ) * keep;
				const vs = b > 0 ? vel[ q + c ] * lam2 - ( pos[ q + c ] - this.targets[ q + c ] ) * springK : 0;
				vel[ q + c ] = vf + ( vs - vf ) * b;

			}

			vel[ q + 3 ] = heat;

		}

		// position (centre of mass of the root from the previous aggregation)
		const root = 4 * struct.root;
		const cx = this.bh[ root ], cy = this.bh[ root + 1 ], cz = this.bh[ root + 2 ];
		const cen = o.centering * ( 1 - b );
		const flatActive = this.flat > 0, lock = 1 - b;
		for ( let i = 0; i < N; i ++ ) {

			const q = 4 * i;
			if ( i === dragIndex ) {

				const t = this.drag.target;
				if ( t ) {

					pos[ q ] = t[ 0 ]; pos[ q + 1 ] = t[ 1 ]; pos[ q + 2 ] = t[ 2 ];

				}

				vel[ q ] = vel[ q + 1 ] = vel[ q + 2 ] = 0;

			} else if ( pos[ q + 3 ] > 0.5 ) {

				vel[ q ] = vel[ q + 1 ] = vel[ q + 2 ] = 0;

			} else {

				let vx = vel[ q ], vy = vel[ q + 1 ], vz = vel[ q + 2 ];
				const len = Math.sqrt( vx * vx + vy * vy + vz * vz );
				if ( len > this.maxDisp ) {

					const s = this.maxDisp / len;
					vx *= s; vy *= s; vz *= s;

				}

				pos[ q ] = cx + ( pos[ q ] + vx - cen * cx - cx ) * f;
				pos[ q + 1 ] = cy + ( pos[ q + 1 ] + vy - cen * cy - cy ) * f;
				pos[ q + 2 ] = cz + ( pos[ q + 2 ] + vz - cen * cz - cz ) * f;
				vel[ q ] = vx * f; vel[ q + 1 ] = vy * f; vel[ q + 2 ] = vz * f;

			}

			if ( flatActive ) {

				const zm = this.zMem[ i ] * f;
				pos[ q + 2 ] += ( zm * ( 1 - this.flat ) - pos[ q + 2 ] ) * lock;
				vel[ q + 2 ] *= 1 - lock;
				this.zMem[ i ] = zm;

			}

		}

		aggregateCPU( g, struct, pos, this.bh );
		this.ticks ++;

	}

	/** Runs until the schedule has cooled (at most `maxTicks`). Returns the number of ticks run. */
	run( maxTicks = 100000 ) {

		let n = 0;
		while ( ! this.schedule.cooled && n < maxTicks ) {

			this.tick();
			n ++;

		}

		return n;

	}

}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// GPU
// ─────────────────────────────────────────────────────────────────────────────────────────────

/** Allocates the private storage buffers for a structure and graph (see header). */
export function createPrivateBuffers( struct, g ) {

	// CPU-written only: bound read-only everywhere
	const ro = ( array, type, name ) => instancedArray( array, type ).setName( name ).toReadOnly();
	const tree = ro( struct.tree, 'uvec4', 'bhTreeExt' );
	const slotTree = ro( struct.slotTree, 'uint', 'bhSlotTree' );
	const nodeOrder = ro( new Uint32Array( struct.nodeOrder.length ? struct.nodeOrder : [ 0 ] ), 'uint', 'bhNodeOrder' );
	const vInfo = ro( struct.vInfo, 'uvec4', 'bhVirtualInfo' );
	const vChildList = ro( struct.vChildList, 'uint', 'bhVirtualChildren' );
	const rInfo = ro( struct.rInfo, 'uvec4', 'bhRealInfo' );
	const rChildList = ro( struct.rChildList, 'uint', 'bhRealChildren' );
	const adjLinks = ro( packLinks( g ), 'uint', 'layoutLinks' );
	// GPU-written
	const treePos = instancedArray( Math.max( 1, struct.TT ), 'vec4' ).setName( 'bhTreePos' );
	const bhCluster = instancedArray( Math.max( 1, struct.C + struct.V ), 'vec4' ).setName( 'bhCluster' );
	const zMem = instancedArray( Math.max( 1, struct.N ), 'float' ).setName( 'layoutZMem' );
	return { tree, treePos, slotTree, nodeOrder, bhCluster, vInfo, vChildList, rInfo, rChildList, adjLinks, zMem };

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

// bounding radii are padded slightly so float rounding can never put a member node outside
// (the walk relies on "outside the bounding sphere" to never accept a node's own cluster)
const RADIUS_PAD_REL = 1e-5, RADIUS_PAD_ABS = 1e-4;
const pad = ( r ) => r.mul( 1 + RADIUS_PAD_REL ).add( RADIUS_PAD_ABS );

// one pass over a child list: centre of mass, then conservative radius -> vec4( centre, radius )
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

/**
 * Aggregation of real clusters [start, start+count) of one height (baked): centre of mass and
 * conservative radius -> clusterPos (shared) and bhCluster. Children come from the private
 * rInfo / rChildList: nodes (weight nodeInfo.w), the top-level virtual clusters of a split
 * cluster (weight vInfo.w), or child clusters (weight rInfo.w).
 */
export function aggregateRealKernel( { buffers, priv, C, start, count, nodeChildren } ) {

	const { clusterPos } = buffers;
	const { bhCluster, rInfo, rChildList, vInfo } = priv;
	const nodePos = readOnly( buffers.nodePos ), nodeInfo = readOnly( buffers.nodeInfo );
	return Fn( () => {

		const c = instanceIndex.add( uint( start ) ).toVar();
		const info = rInfo.element( c ).toVar();
		const out = vec4( 0 ).toVar();
		if ( nodeChildren ) {

			If( info.z.equal( uint( 0 ) ), () => {

				out.assign( reduce( info.x, info.y,
					( k ) => nodePos.element( rChildList.element( k ) ).xyz,
					( k ) => float( nodeInfo.element( rChildList.element( k ) ).w ),
					() => float( 0 ) ) );

			} ).Else( () => {

				out.assign( reduce( info.x, info.y,
					( k ) => bhCluster.element( rChildList.element( k ) ).xyz,
					( k ) => float( vInfo.element( rChildList.element( k ).sub( uint( C ) ) ).w ),
					( k ) => bhCluster.element( rChildList.element( k ) ).w ) );

			} );

		} else {

			out.assign( reduce( info.x, info.y,
				( k ) => clusterPos.element( rChildList.element( k ) ).xyz,
				( k ) => float( rInfo.element( rChildList.element( k ) ).w ),
				( k ) => clusterPos.element( rChildList.element( k ) ).w ) );

		}

		clusterPos.element( c ).assign( out );
		bhCluster.element( c ).assign( out );

	} )().compute( count, WG() ).setName( `layout.aggregate.h${nodeChildren ? 1 : 'n'}.${start}` );

}

/** Aggregation of virtual clusters [start, start+count) of one virtual height (baked). */
export function aggregateVirtualKernel( { buffers, priv, C, start, count, height } ) {

	const nodePos = readOnly( buffers.nodePos ), nodeInfo = readOnly( buffers.nodeInfo );
	const { bhCluster, vInfo, vChildList, nodeOrder } = priv;
	const leaf = height === 1;
	return Fn( () => {

		const v = instanceIndex.add( uint( start ) ).toVar();
		const info = vInfo.element( v ).toVar();
		const out = leaf
			? reduce( info.x, info.y,
				( k ) => nodePos.element( nodeOrder.element( k ) ).xyz,
				( k ) => float( nodeInfo.element( nodeOrder.element( k ) ).w ),
				() => float( 0 ) )
			: reduce( info.x, info.y,
				( k ) => bhCluster.element( vChildList.element( k ) ).xyz,
				( k ) => float( vInfo.element( vChildList.element( k ).sub( uint( C ) ) ).w ),
				( k ) => bhCluster.element( vChildList.element( k ) ).w );
		bhCluster.element( v.add( uint( C ) ) ).assign( out );

	} )().compute( count, WG() ).setName( `layout.aggregate.v${height}` );

}

/**
 * Copies positions (nodes, radius 0) and centres/radii (clusters) into tree order, so the
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
 * Barnes–Hut repulsion with the bounded short-range kernel. One thread per slot (tree leaf
 * order, so a workgroup walks nearly the same path). Nodes and clusters are handled alike (a
 * node is a cluster of radius 0): an entry is accepted as a point mass when the node lies
 * outside its bounding sphere and radius ≤ theta·distance; otherwise the walk descends (k + 1).
 * The node's own entry has distance 0 and is never accepted, which skips self-interaction.
 * Accepted entry at offset d with mass m_e (tree.z / 256):
 *   w = β·m_i·m_e / (1 + |d|²/u²)²;   F −= d·w;   S += w        (no sqrt, no distance clamp)
 * After maxIterations/2 loop iterations any entry the node is outside of is accepted (graceful
 * degradation), and maxIterations is a hard cap. Exactly coincident distinct nodes feel no
 * force from this kernel, so they are pushed apart along a deterministic hash direction.
 * Output: nodeForce = (F_rep, S); S is the repulsion stiffness the preconditioner uses.
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
		const cm = float( tree.element( self ).z ).mul( U.betaScaled ).toVar(); // β·m_i / 256 (betaScaled = β / 256²)
		const f = vec3( 0 ).toVar();
		const S = float( 0 ).toVar();
		const k = uint( 0 ).toVar();
		// contribution of an entry at offset d with fixed-point mass mz
		const add = ( d, l2, mz ) => {

			const q = l2.mul( U.invUnit2 ).add( 1 ).toVar();
			const w = cm.mul( float( mz ) ).div( q.mul( q ) ).toVar();
			f.subAssign( d.mul( w ) );
			S.addAssign( w );

		};

		Loop( { start: uint( 0 ), end: U.maxIterations, type: 'uint', condition: '<', name: 'it' }, ( { it } ) => {

			If( k.greaterThanEqual( uint( TT ) ), () => {

				Break();

			} );

			const e = treePos.element( k ).toVar();
			const m = tree.element( k ).toVar(); // ref, skip, mass, count
			const d = e.xyz.sub( p ).toVar();
			const l2 = dot( d, d ).toVar();
			const r2 = e.w.mul( e.w ).toVar();
			const accept = r2.lessThan( l2 ).and( r2.lessThanEqual( U.theta2.mul( l2 ) ).or( it.greaterThanEqual( U.softIterations ) ) );
			If( accept, () => {

				add( d, l2, m.z );
				k.assign( m.y );

			} ).ElseIf( l2.lessThan( 1e-12 ).and( m.y.equal( k.add( uint( 1 ) ) ) ).and( k.notEqual( self ) ), () => {

				// a distinct node at exactly our position (rare): push apart deterministically
				const j = jitter( node, nodeOrder.element( m.x ) ).mul( U.jitterLength ).toVar();
				add( j, dot( j, j ), m.z );
				k.addAssign( uint( 1 ) );

			} ).Else( () => {

				k.addAssign( uint( 1 ) );

			} );

		} );

		nodeForce.element( node ).assign( vec4( f, S ) );

	} )().compute( N, WG() ).setName( 'layout.repulsion' );

}

/**
 * Links pass (gather; each thread writes only its own node). Reads nodeForce = (F_rep, S):
 *   F = F_rep + Σ_j w_ij·(x_j − x_i) + κ·(c_A(i) − x_i) − g·x_i        w_ij = lcaWeight[level]
 *   M = 1 + κ + g + Σ_j w_ij + ρ·S                                      Jacobi preconditioner
 *   heat_i = max(heatDecay·heat_i, heatSpread·max_j heat_j), 1 for the dragged node, 0 below 1e-4
 * and writes nodeForce = (F/M, heat). The neighbours' heat is read from nodeVel.w, which this
 * pass does not write (the velocity pass copies the new heat there), so there is no race. The
 * attraction has rest length 0 and needs no length() or division per neighbour; the neighbour
 * and the LCA level of the edge come from one packed uint (adjLinks).
 */
export function linksKernel( { buffers, priv, U, N, D } ) {

	const { nodeForce } = buffers;
	const [ nodePos, nodeVel, nodeInfo, clusterPos ] = [ buffers.nodePos, buffers.nodeVel, buffers.nodeInfo, buffers.clusterPos ].map( readOnly );
	const { adjLinks } = priv;
	return Fn( () => {

		const i = instanceIndex;
		const p = nodePos.element( i ).xyz.toVar();
		const fs = nodeForce.element( i ).toVar();
		const info = nodeInfo.element( i ).toVar(); // adjStart, degree, finest cluster, mass
		const F = fs.xyz.toVar();
		const W = float( 0 ).toVar();
		const hn = float( 0 ).toVar();
		const heatOn = U.heatActive.greaterThan( 0.5 ).toVar();
		Loop( { start: uint( 0 ), end: info.y, type: 'uint', condition: '<', name: 'k' }, ( { k } ) => {

			const a = adjLinks.element( info.x.add( k ) ).toVar();
			const j = a.bitAnd( uint( NODE_MASK ) ).toVar();
			const w = U.lcaWeight.element( a.shiftRight( uint( LEVEL_SHIFT ) ) ).toVar();
			F.addAssign( nodePos.element( j ).xyz.sub( p ).mul( w ) );
			W.addAssign( w );
			If( heatOn, () => {

				hn.assign( max( hn, nodeVel.element( j ).w ) );

			} );

		} );

		if ( D > 0 ) F.addAssign( clusterPos.element( info.z ).xyz.sub( p ).mul( U.kappa ) );
		F.subAssign( p.mul( U.gravity ) );
		const M = W.add( fs.w.mul( U.rho ) ).add( U.kappa ).add( U.gravity ).add( 1 );
		const h = float( 0 ).toVar();
		If( heatOn, () => {

			h.assign( max( nodeVel.element( i ).w.mul( U.heatDecay ), hn.mul( U.heatSpread ) ) );
			If( int( i ).equal( U.heatIndex ), () => {

				h.assign( 1 );

			} );
			If( h.lessThan( MODEL.heatCut ), () => {

				h.assign( 0 );

			} );

		} );
		nodeForce.element( i ).assign( vec4( F.div( M ), h ) );

	} )().compute( N, WG() ).setName( 'layout.links' );

}

/**
 * Velocity update. With nodeForce = (F/M, heat) from the links pass:
 *   force part    v_f = (v + α_i·F/M)·(1 − decay),  α_i = max(α, α_drag·heat)
 *   target part   v_s = λ²·v − (1−λ)²·(x − target): a discrete critically damped spring (double
 *                 eigenvalue λ = exp(−ω·h)), so nodes converge exactly onto their targets
 *   v' = mix(v_f, v_s, targetBlend·nodeTarget.w)
 * target = nodeTarget.xyz, in the paper's sphere layout (layoutBlend) scaled toward the centre
 * for the selected node (× 0.5) and its neighbours (× 0.75). The heat is copied to nodeVel.w
 * (0 in ticks without a links pass, U.hasForces = 0).
 */
export function velocityKernel( { buffers, U, N } ) {

	const { nodeVel } = buffers;
	const [ nodePos, nodeForce, nodeTarget, nodeState ] = [ buffers.nodePos, buffers.nodeForce, buffers.nodeTarget, buffers.nodeState ].map( readOnly );
	const { layoutBlend, targetBlend } = buffers.uniforms;
	return Fn( () => {

		const i = instanceIndex;
		const p = nodePos.element( i ).xyz.toVar();
		const v = nodeVel.element( i ).xyz.mul( U.velRescale ).toVar();
		const fh = nodeForce.element( i ).toVar();
		const heat = fh.w.mul( U.hasForces ).toVar();
		const tg = nodeTarget.element( i ).toVar();
		const b = targetBlend.mul( tg.w ).toVar();
		const vf = vec3( 0 ).toVar();
		const vs = vec3( 0 ).toVar();

		If( b.lessThan( 1 ), () => {

			const a = max( U.alpha, U.alphaDrag.mul( heat ) );
			vf.assign( v.add( fh.xyz.mul( a ) ).mul( float( 1 ).sub( U.velocityDecay ) ) );

		} );

		If( b.greaterThan( 0 ), () => {

			const st = nodeState.element( i );
			const inward = mix( mix( float( 1 ), float( 0.75 ), st.y ), float( 0.5 ), st.x );
			const target = tg.xyz.mul( mix( float( 1 ), inward, layoutBlend ) );
			vs.assign( v.mul( U.lam2 ).sub( p.sub( target ).mul( U.springK ) ) );

		} );

		nodeVel.element( i ).assign( vec4( mix( vf, vs, b ), heat ) );

	} )().compute( N, WG() ).setName( 'layout.velocity' );

}

/**
 * Position update:
 *   free nodes   v ← clamp(v, maxDisp);  x ← c + (x + v − centering·(1 − targetBlend)·c − c)·f;
 *                v ← v·f     c = centre of mass of the root, f = auto-scale factor of this tick
 *   pinned nodes (w = 1) keep their position, the dragged node is held, then placed on the drag
 *   target; both get zero velocity and are not scaled.
 * Flat mode (U.flatActive): z is not simulated but locked to zMem·(1 − flat), where zMem is the
 * z of the node when the flat mode was entered (scaled along with the layout) and flat the eased
 * 0..1 blend: the layout is squashed into the z = 0 plane and unfolds into the remembered shape
 * when the mode is left. The lock fades with targetBlend so target layouts can leave the plane.
 * A perfectly planar state is an (unstable) equilibrium of the 3D model, since no force has a z
 * component there: when a 3D force layout restarts (after a planar target layout, say), free
 * nodes get a tiny hash-based z offset for one tick (U.zKick, 2% of the kernel unit).
 * nodeVel.w (heat) is preserved.
 */
export function positionKernel( { buffers, priv, U, N, root } ) {

	const { nodePos, nodeVel } = buffers;
	const clusterPos = readOnly( buffers.clusterPos );
	const { zMem } = priv;
	const { targetBlend } = buffers.uniforms;
	return Fn( () => {

		const i = instanceIndex;
		const pos = nodePos.element( i ).toVar();
		const p = pos.xyz.toVar();
		const vel = nodeVel.element( i ).toVar();
		const v = vel.xyz.toVar();
		const free = float( 1 ).sub( targetBlend ).toVar();
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
			const c = clusterPos.element( uint( root ) ).xyz.toVar();
			p.assign( c.add( p.add( v ).sub( c.mul( U.centering.mul( free ) ) ).sub( c ).mul( U.scaleF ) ) );
			v.mulAssign( U.scaleF );
			// one-shot symmetry breaking when a 3D force layout restarts (U.zKick > 0 for one tick)
			const hz = float( instanceIndex.mul( uint( 2654435761 ) ).shiftRight( uint( 16 ) ) ).div( 65535 ).sub( 0.5 );
			p.assign( vec3( p.xy, p.z.add( hz.mul( U.zKick ) ) ) );

		} );

		If( U.flatActive.greaterThan( 0.5 ), () => {

			const zm = zMem.element( i ).mul( U.scaleF ).toVar();
			p.assign( vec3( p.xy, mix( p.z, zm.mul( float( 1 ).sub( U.flat ) ), free ) ) );
			v.assign( vec3( v.xy, v.z.mul( targetBlend ) ) );
			zMem.element( i ).assign( zm );

		} );

		nodePos.element( i ).assign( vec4( p, pos.w ) );
		nodeVel.element( i ).assign( vec4( v, vel.w ) );

	} )().compute( N, WG() ).setName( 'layout.position' );

}

/** Remembers the current z of every node (run once when the flat mode is entered). */
export function zSnapshotKernel( { buffers, priv, N } ) {

	const nodePos = readOnly( buffers.nodePos );
	const { zMem } = priv;
	return Fn( () => {

		zMem.element( instanceIndex ).assign( nodePos.element( instanceIndex ).z );

	} )().compute( N, WG() ).setName( 'layout.zSnapshot' );

}

/**
 * Applies queued commands: unpin-all flag, then up to U.cmds.length (index, pinned, heat) entries
 * in order, then an explicit move of U.moveIndex to U.dragTarget (drop / drag while not ticking).
 * Touched nodes get zero velocity and at least the command's heat (a released pin relaxes its
 * neighbourhood through local heat instead of a global reheat).
 */
export function commandKernel( { buffers, U, N } ) {

	const { nodePos, nodeVel } = buffers;
	return Fn( () => {

		const i = int( instanceIndex ).toVar();
		const pos = nodePos.element( instanceIndex ).toVar();
		const w = pos.w.toVar();
		const p = pos.xyz.toVar();
		const touched = float( 0 ).toVar();
		const heat = float( 0 ).toVar();
		If( U.unpinAll.greaterThan( 0.5 ), () => {

			w.assign( 0 );

		} );
		Loop( { start: uint( 0 ), end: U.cmdCount, type: 'uint', condition: '<', name: 'k' }, ( { k } ) => {

			const c = U.cmds.element( k );
			If( int( c.x ).equal( i ), () => {

				w.assign( c.y );
				heat.assign( max( heat, c.z ) );
				touched.assign( 1 );

			} );

		} );
		If( i.equal( U.moveIndex ), () => {

			p.assign( U.dragTarget );
			touched.assign( 1 );

		} );
		nodePos.element( instanceIndex ).assign( vec4( p, w ) );
		If( touched.greaterThan( 0.5 ), () => {

			nodeVel.element( instanceIndex ).assign( vec4( 0, 0, 0, max( heat, nodeVel.element( instanceIndex ).w ) ) );

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
