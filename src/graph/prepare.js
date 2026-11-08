// Turns a canonical graph (see src/data/format.js) into the flat typed arrays the GPU modules
// consume. Pure JS, no three.js: runs in the main thread, a worker, or Node (tests).
//
// ─── Index spaces ───────────────────────────────────────────────────────────────────────────
//  nodes     0..N-1
//  clusters  0..C-1, sorted by height ascending. Height 1 = finest hierarchy level (level 0),
//            height D = coarsest level (level D-1), height D+1 = the root, always index C-1.
//            Cluster ids of level l occupy [levelOffset[l], levelOffset[l] + K_l).
//  edges     0..E-1 (undirected, deduplicated, no self loops)
//
// ─── Edge bundling path (Holten-style hierarchical edge bundling, as in ImsvGraphVis) ─────────
//  For an edge (s, t) with lowest-common-ancestor level `lca` (first level where both endpoints
//  share a cluster, or D when only the root is shared), the B-spline control polygon is
//    s, anc[s][0], …, anc[s][lca-1], [anc[s][lca] if lca < D], anc[t][lca-1], …, anc[t][0], t
//  i.e. L = 2 + 2*lca + (lca < D ? 1 : 0) control points. The root is never a control point.
//
// ─── Barnes–Hut tree ────────────────────────────────────────────────────────────────────────
//  The clustering hierarchy doubles as the Barnes–Hut tree for repulsion. `bhTree` stores the
//  tree (root excluded) in depth-first preorder as pairs (ref, skip): ref < N is a node, ref >= N
//  is cluster ref-N; skip is the preorder index just past the element's subtree. A stackless
//  traversal is: i = 0; while (i < T) { if (leaf or accepted) { use; i = skip } else i++ }.

import { paletteColor } from '../util/color.js';

export const DEFAULT_PREPARE_OPTIONS = {
	palette: 'paper',
	seed: 1,
	linkDistance: 30, // world units; the force layout uses the same value by default
	sphere: { fov: 240, aspect: 1.6, nesting: 0.1, projection: 'stereographic-radial' },
};

/** Deterministic PRNG (mulberry32). */
export function rng( seed ) {

	let a = ( seed >>> 0 ) || 0x9e3779b9;
	return () => {

		a = ( a + 0x6d2b79f5 ) >>> 0;
		let t = a;
		t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
		t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

/** Radius (world units) a force layout of n nodes tends to occupy. */
export function layoutRadius( n, linkDistance = 30 ) {

	return linkDistance * ( 1.5 + 1.25 * Math.cbrt( n ) );

}

/**
 * @param {object} g canonical graph (normalizeGraph output)
 * @param {object} [options]
 * @returns {PreparedGraph}
 */
export function prepareGraph( g, options = {} ) {

	const opts = { ...DEFAULT_PREPARE_OPTIONS, ...options, sphere: { ...DEFAULT_PREPARE_OPTIONS.sphere, ...( options.sphere || {} ) } };
	const N = g.nodeCount;
	const edges = g.edges instanceof Uint32Array ? g.edges : Uint32Array.from( g.edges );
	const E = edges.length / 2;
	const levels = g.hierarchy ? g.hierarchy.levels : [];
	const D = levels.length;

	// ── clusters ──────────────────────────────────────────────────────────────────────────
	const K = levels.map( ( lvl ) => {

		let m = - 1;
		for ( let i = 0; i < N; i ++ ) if ( lvl[ i ] > m ) m = lvl[ i ];
		return m + 1;

	} );
	const levelOffset = new Uint32Array( D + 1 );
	for ( let l = 0; l < D; l ++ ) levelOffset[ l + 1 ] = levelOffset[ l ] + K[ l ];
	const C = levelOffset[ D ] + 1;
	const root = C - 1;

	const anc = new Uint32Array( Math.max( 1, N * D ) );
	for ( let i = 0; i < N; i ++ ) for ( let l = 0; l < D; l ++ ) anc[ i * D + l ] = levelOffset[ l ] + levels[ l ][ i ];

	const clusterHeight = new Uint32Array( C );
	const clusterParent = new Int32Array( C ).fill( - 1 );
	const clusterMass = new Uint32Array( C );
	for ( let l = 0; l < D; l ++ ) for ( let k = 0; k < K[ l ]; k ++ ) clusterHeight[ levelOffset[ l ] + k ] = l + 1;
	clusterHeight[ root ] = D + 1;
	clusterMass[ root ] = N;
	for ( let i = 0; i < N; i ++ ) {

		for ( let l = 0; l < D; l ++ ) {

			const c = anc[ i * D + l ];
			clusterMass[ c ] ++;
			clusterParent[ c ] = l + 1 < D ? anc[ i * D + l + 1 ] : root;

		}

	}

	// children lists (CSR); height-1 clusters (and the root when D = 0) have node children
	const childCount = new Uint32Array( C );
	const nodeParent = new Uint32Array( N );
	for ( let i = 0; i < N; i ++ ) {

		nodeParent[ i ] = D > 0 ? anc[ i * D ] : root;
		childCount[ nodeParent[ i ] ] ++;

	}

	for ( let c = 0; c < root; c ++ ) childCount[ clusterParent[ c ] ] ++;

	// degree and adjacency (needed for child ordering)
	const degree = new Uint32Array( N );
	for ( let k = 0; k < 2 * E; k ++ ) degree[ edges[ k ] ] ++;
	const adjStart = new Uint32Array( N );
	for ( let i = 1; i < N; i ++ ) adjStart[ i ] = adjStart[ i - 1 ] + degree[ i - 1 ];
	const adjList = new Uint32Array( Math.max( 1, 2 * E ) );
	{

		const fill = adjStart.slice();
		for ( let e = 0; e < E; e ++ ) {

			const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
			adjList[ fill[ a ] ++ ] = b;
			adjList[ fill[ b ] ++ ] = a;

		}

	}

	let maxDegree = 0;
	for ( let i = 0; i < N; i ++ ) if ( degree[ i ] > maxDegree ) maxDegree = degree[ i ];

	// children as "refs" (ref < N: node, else N + cluster), sorted: clusters by mass desc,
	// nodes by degree desc. This order drives DFS (colors) and the BH tree.
	const childStart = new Uint32Array( C );
	for ( let c = 1; c < C; c ++ ) childStart[ c ] = childStart[ c - 1 ] + childCount[ c - 1 ];
	const totalChildren = childStart[ C - 1 ] + childCount[ C - 1 ];
	const childRefs = new Uint32Array( totalChildren );
	{

		const fill = childStart.slice();
		for ( let i = 0; i < N; i ++ ) childRefs[ fill[ nodeParent[ i ] ] ++ ] = i;
		for ( let c = 0; c < root; c ++ ) childRefs[ fill[ clusterParent[ c ] ] ++ ] = N + c;
		const weight = ( ref ) => ( ref < N ? degree[ ref ] : clusterMass[ ref - N ] * 1e6 );
		for ( let c = 0; c < C; c ++ ) {

			const sub = Array.from( childRefs.subarray( childStart[ c ], childStart[ c ] + childCount[ c ] ) );
			sub.sort( ( a, b ) => weight( b ) - weight( a ) || a - b );
			childRefs.set( sub, childStart[ c ] );

		}

	}

	// GPU child list: plain indices (node index for height-1 clusters, cluster index otherwise)
	const childList = new Uint32Array( Math.max( 1, totalChildren ) );
	for ( let k = 0; k < totalChildren; k ++ ) childList[ k ] = childRefs[ k ] < N ? childRefs[ k ] : childRefs[ k ] - N;

	const heightRanges = [];
	for ( let l = 0; l < D; l ++ ) heightRanges.push( { height: l + 1, start: levelOffset[ l ], count: K[ l ] } );
	heightRanges.push( { height: D + 1, start: root, count: 1 } );

	// ── DFS preorder: BH tree, leaf ranks ──────────────────────────────────────────────────
	const T = N + C - 1;
	const bhTree = new Uint32Array( Math.max( 2, 2 * T ) );
	const leafRank = new Float32Array( N );
	const clusterRankRange = new Float32Array( 2 * C ); // [min, max] leaf rank under cluster
	{

		let pos = 0, leaf = 0;
		// explicit stack of [ref, phase]; phase 0 = enter, 1 = exit
		const stack = [];
		for ( let k = childStart[ root ] + childCount[ root ] - 1; k >= childStart[ root ]; k -- ) stack.push( childRefs[ k ], 0 );
		const preIndex = new Int32Array( C ).fill( - 1 );
		const firstLeaf = new Int32Array( C );
		while ( stack.length ) {

			const phase = stack.pop(), ref = stack.pop();
			if ( ref < N ) {

				bhTree[ 2 * pos ] = ref;
				bhTree[ 2 * pos + 1 ] = pos + 1;
				pos ++;
				leafRank[ ref ] = leaf ++;

			} else if ( phase === 0 ) {

				const c = ref - N;
				preIndex[ c ] = pos;
				firstLeaf[ c ] = leaf;
				bhTree[ 2 * pos ] = ref;
				pos ++;
				stack.push( ref, 1 );
				for ( let k = childStart[ c ] + childCount[ c ] - 1; k >= childStart[ c ]; k -- ) stack.push( childRefs[ k ], 0 );

			} else {

				const c = ref - N;
				bhTree[ 2 * preIndex[ c ] + 1 ] = pos;
				clusterRankRange[ 2 * c ] = firstLeaf[ c ];
				clusterRankRange[ 2 * c + 1 ] = leaf - 1;

			}

		}

		if ( pos !== T ) throw new Error( `prepareGraph: BH tree size mismatch (${pos} vs ${T})` );
		const denom = Math.max( 1, N - 1 );
		for ( let i = 0; i < N; i ++ ) leafRank[ i ] /= denom;
		for ( let k = 0; k < 2 * C; k ++ ) clusterRankRange[ k ] /= denom;
		clusterRankRange[ 2 * root ] = 0;
		clusterRankRange[ 2 * root + 1 ] = 1;

	}

	// ── edges: lowest common ancestor level ────────────────────────────────────────────────
	const edgeLca = new Uint32Array( Math.max( 1, E ) );
	let maxPath = 2;
	for ( let e = 0; e < E; e ++ ) {

		const s = edges[ 2 * e ], t = edges[ 2 * e + 1 ];
		let l = 0;
		while ( l < D && anc[ s * D + l ] !== anc[ t * D + l ] ) l ++;
		edgeLca[ e ] = l;
		const L = 2 + 2 * l + ( l < D ? 1 : 0 );
		if ( L > maxPath ) maxPath = L;

	}

	// ── colors and sizes ───────────────────────────────────────────────────────────────────
	const colors = colorize( leafRank, opts.palette );
	const radii = new Float32Array( N );
	for ( let i = 0; i < N; i ++ ) radii[ i ] = 0.7 + 0.9 * Math.sqrt( degree[ i ] / Math.max( 1, maxDegree ) );

	// ── initial positions (hierarchical seeding) and the paper's spherical layout ───────────
	const R = layoutRadius( N, opts.linkDistance );
	const initPos = seedPositions( { N, C, root, childStart, childCount, childRefs, clusterMass, R, seed: opts.seed } );
	const sphereDir = sphereLayout( { N, C, root, childStart, childCount, childRefs, seed: opts.seed, ...opts.sphere } );

	const clustersPerLevel = K.slice();

	return {
		name: g.name,
		description: g.description,
		source: g.source,
		labels: g.labels,
		N, E, D, C, T, root,
		edges, edgeLca, maxPath,
		degree, maxDegree, adjStart, adjList,
		anc, levelOffset,
		clusterHeight, clusterParent, clusterMass,
		clusterChildStart: childStart, clusterChildCount: childCount, childList, childRefs,
		heightRanges,
		bhTree,
		leafRank, clusterRankRange,
		colors, radii,
		initPos, sphereDir,
		layoutRadius: R,
		linkDistance: opts.linkDistance,
		palette: opts.palette,
		stats: {
			nodes: N, edges: E, levels: D, clusters: C - 1, clustersPerLevel,
			maxDegree, avgDegree: N ? ( 2 * E ) / N : 0, maxPath,
		},
	};

}

/** Per-node linear-RGB colors (vec4, alpha 1) from a palette over DFS leaf rank. */
export function colorize( leafRank, palette = 'paper' ) {

	const N = leafRank.length;
	const colors = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) {

		const [ r, g, b ] = paletteColor( palette, leafRank[ i ] );
		colors[ 4 * i ] = r; colors[ 4 * i + 1 ] = g; colors[ 4 * i + 2 ] = b; colors[ 4 * i + 3 ] = 1;

	}

	return colors;

}

function fibonacciSphere( k, m, out ) {

	// k-th of m points evenly spread on the unit sphere
	const y = m === 1 ? 0 : 1 - ( 2 * ( k + 0.5 ) ) / m;
	const r = Math.sqrt( Math.max( 0, 1 - y * y ) );
	const phi = k * Math.PI * ( 3 - Math.sqrt( 5 ) );
	out[ 0 ] = Math.cos( phi ) * r; out[ 1 ] = y; out[ 2 ] = Math.sin( phi ) * r;
	return out;

}

/**
 * Hierarchical seeding: children of a cluster are spread on a sphere around its center with radii
 * proportional to cube roots of their masses, so clusters start spatially compact. This makes the
 * hierarchy-based Barnes–Hut repulsion efficient from the first iteration and speeds convergence.
 */
function seedPositions( { N, root, childStart, childCount, childRefs, clusterMass, R, seed } ) {

	const rand = rng( seed * 7919 + 13 );
	const pos = new Float32Array( 4 * N );
	const dir = [ 0, 0, 0 ];
	// rotate each child set randomly so the Fibonacci pattern does not align across levels
	const randomRotate = ( v ) => {

		const a = rand() * Math.PI * 2, b = Math.acos( 2 * rand() - 1 );
		const ca = Math.cos( a ), sa = Math.sin( a ), cb = Math.cos( b ), sb = Math.sin( b );
		const x = v[ 0 ] * ca - v[ 2 ] * sa, z = v[ 0 ] * sa + v[ 2 ] * ca;
		const y = v[ 1 ] * cb - z * sb, z2 = v[ 1 ] * sb + z * cb;
		v[ 0 ] = x; v[ 1 ] = y; v[ 2 ] = z2;

	};

	const stack = [ [ N + root, 0, 0, 0, R ] ];
	while ( stack.length ) {

		const [ ref, cx, cy, cz, r ] = stack.pop();
		if ( ref < N ) {

			pos[ 4 * ref ] = cx; pos[ 4 * ref + 1 ] = cy; pos[ 4 * ref + 2 ] = cz; pos[ 4 * ref + 3 ] = 0;
			continue;

		}

		const c = ref - N;
		const m = childCount[ c ];
		const mass = clusterMass[ c ] || 1;
		for ( let k = 0; k < m; k ++ ) {

			const child = childRefs[ childStart[ c ] + k ];
			const childMass = child < N ? 1 : clusterMass[ child - N ];
			if ( child < N ) {

				// nodes: uniform in the ball
				const u = Math.cbrt( rand() ) * r;
				fibonacciSphere( Math.floor( rand() * 4096 ), 4096, dir );
				randomRotate( dir );
				stack.push( [ child, cx + dir[ 0 ] * u, cy + dir[ 1 ] * u, cz + dir[ 2 ] * u, 0 ] );

			} else {

				const cr = r * Math.cbrt( childMass / mass ) * 0.9;
				const spread = m === 1 ? 0 : r - cr * 0.6;
				fibonacciSphere( k, m, dir );
				stack.push( [ child, cx + dir[ 0 ] * spread, cy + dir[ 1 ] * spread, cz + dir[ 2 ] * spread, cr ] );

			}

		}

	}

	return pos;

}

// ── The paper's layout: squarified treemap of the hierarchy, projected onto a sphere ──────────
// Port of FIGVTreemapLayout (HCIL treemap) + AIGVGraphActor::NormalizeNodePosition +
// UIGVProjection::ToSphere_Stereographic_RadialWarping. Output: unit direction per node (vec4).
// Coordinates: the layout center faces +Z (towards a default camera on the +Z axis), Y is up.

function sphereLayout( { N, root, childStart, childCount, childRefs, seed, fov, aspect, nesting, projection } ) {

	const rand = rng( seed * 104729 + 7 );
	const X = ( fov * Math.PI ) / 360; // half extent in radians
	const Y = X / aspect;

	// treemap nodes are refs; weights: leaves random in [1,2) (as in the paper), clusters = sum
	const weightOf = new Map();
	const postorder = [];
	{

		const st = [ N + root ];
		while ( st.length ) {

			const ref = st.pop();
			postorder.push( ref );
			if ( ref >= N ) {

				const c = ref - N;
				for ( let k = 0; k < childCount[ c ]; k ++ ) st.push( childRefs[ childStart[ c ] + k ] );

			}

		}

	}

	for ( let k = postorder.length - 1; k >= 0; k -- ) {

		const ref = postorder[ k ];
		if ( ref < N ) weightOf.set( ref, 1 + rand() );
		else {

			const c = ref - N;
			let w = 0;
			for ( let j = 0; j < childCount[ c ]; j ++ ) w += weightOf.get( childRefs[ childStart[ c ] + j ] );
			weightOf.set( ref, w );

		}

	}

	const rect = new Map(); // ref -> [minX, minY, maxX, maxY]
	rect.set( N + root, [ - X, - Y, X, Y ] );
	const nest = Math.min( 0.4999, Math.max( 0, nesting ) );

	// ancestor-first traversal
	const queue = [ N + root ];
	while ( queue.length ) {

		const ref = queue.pop();
		if ( ref < N ) continue;
		const c = ref - N;
		const kids = [];
		for ( let k = 0; k < childCount[ c ]; k ++ ) kids.push( childRefs[ childStart[ c ] + k ] );
		if ( ! kids.length ) continue;
		kids.sort( ( a, b ) => weightOf.get( b ) - weightOf.get( a ) );
		const [ x0, y0, x1, y1 ] = rect.get( ref );
		const shrink = Math.min( x1 - x0, y1 - y0 ) * nest * 0.5;
		squarified( kids, 0, kids.length - 1, [ x0 + shrink, y0 + shrink, x1 - shrink, y1 - shrink ], weightOf, rect );
		for ( const k of kids ) queue.push( k );

	}

	// node 2D positions: rect center + random offset within half extent (paper: FIGVTreemapLayout::Compute)
	const p2 = new Float64Array( 2 * N );
	let bx0 = Infinity, by0 = Infinity, bx1 = - Infinity, by1 = - Infinity;
	for ( let i = 0; i < N; i ++ ) {

		const [ x0, y0, x1, y1 ] = rect.get( i );
		const cx = ( x0 + x1 ) / 2, cy = - ( y0 + y1 ) / 2;
		const ex = ( x1 - x0 ) / 2, ey = ( y1 - y0 ) / 2;
		const px = cx + 0.5 * ( rand() * 2 - 1 ) * ex;
		const py = cy + 0.5 * ( rand() * 2 - 1 ) * ey;
		p2[ 2 * i ] = px; p2[ 2 * i + 1 ] = py;
		bx0 = Math.min( bx0, px ); by0 = Math.min( by0, py ); bx1 = Math.max( bx1, px ); by1 = Math.max( by1, py );

	}

	const cx = ( bx0 + bx1 ) / 2, cy = ( by0 + by1 ) / 2;
	const ex = Math.max( 1e-9, ( bx1 - bx0 ) / 2 ), ey = Math.max( 1e-9, ( by1 - by0 ) / 2 );
	const out = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) {

		const px = ( ( p2[ 2 * i ] - cx ) / ex ) * X;
		const py = ( ( p2[ 2 * i + 1 ] - cy ) / ey ) * Y;
		const v = projectToSphere( px, py, projection );
		out[ 4 * i ] = v[ 0 ]; out[ 4 * i + 1 ] = v[ 1 ]; out[ 4 * i + 2 ] = v[ 2 ]; out[ 4 * i + 3 ] = 0;

	}

	return out;

}

/**
 * Planar point (radians) -> unit vector, three.js axes, layout center at +Z.
 * Unreal axes (X forward, Y right, Z up) map to three.js (x, y, z) = (-Y, Z, X) here, a proper
 * rotation so the layout center faces a camera placed on the +Z axis.
 */
export function projectToSphere( px, py, projection = 'stereographic-radial' ) {

	let ux, uy, uz; // Unreal-space unit vector
	const s = Math.hypot( px, py );
	if ( projection === 'gnomonic-radial' ) {

		const n = s > 0 ? [ px / s, py / s ] : [ 0, 0 ];
		const t = Math.tan( Math.min( s, Math.PI * 0.4999 ) );
		const l = Math.hypot( 1, n[ 0 ] * t, n[ 1 ] * t );
		ux = 1 / l; uy = ( n[ 0 ] * t ) / l; uz = ( n[ 1 ] * t ) / l;

	} else if ( projection === 'spherical' ) {

		const theta = Math.min( Math.PI, Math.max( 0, - py + Math.PI / 2 ) );
		const phi = px;
		ux = Math.sin( theta ) * Math.cos( phi ); uy = Math.sin( theta ) * Math.sin( phi ); uz = Math.cos( theta );

	} else {

		// stereographic with radial warping (paper default)
		const R = Math.tan( Math.min( s, Math.PI * 0.999 ) * 0.5 );
		const phi = Math.atan2( py, px );
		const theta = R > 0 ? 2 * Math.atan( 1 / R ) : Math.PI;
		const vx = Math.sin( theta ) * Math.cos( phi ), vy = Math.sin( theta ) * Math.sin( phi ), vz = Math.cos( theta );
		ux = - vz; uy = vx; uz = vy;

	}

	return [ - uy, uz, ux ];

}

function squarified( kids, first, last, bounds, weightOf, rect ) {

	if ( first > last ) return;
	if ( last - first < 2 ) return sliceAndDice( kids, first, last, bounds, weightOf, rect );
	const [ x, y, x1, y1 ] = bounds;
	const w = x1 - x, h = y1 - y;
	let acc = 0;
	for ( let k = first; k <= last; k ++ ) acc += weightOf.get( kids[ k ] );
	const firstRel = weightOf.get( kids[ first ] ) / acc;
	let mid = first;
	let off = firstRel;
	const big = w < h ? h : w, small = w < h ? w : h;
	// grow the row while the aspect ratio of its first (largest) item keeps improving
	while ( mid + 1 <= last ) {

		const rel = weightOf.get( kids[ mid + 1 ] ) / acc;
		if ( normalizedAspect( big, small, firstRel, off + rel ) > normalizedAspect( big, small, firstRel, off ) ) break;
		mid ++;
		off += rel;

	}

	if ( w < h ) {

		sliceAndDice( kids, first, mid, [ x, y, x + w, y + h * off ], weightOf, rect );
		squarified( kids, mid + 1, last, [ x, y + h * off, x + w, y1 ], weightOf, rect );

	} else {

		sliceAndDice( kids, first, mid, [ x, y, x + w * off, y + h ], weightOf, rect );
		squarified( kids, mid + 1, last, [ x + w * off, y, x1, y + h ], weightOf, rect );

	}

}

function sliceAndDice( kids, first, last, bounds, weightOf, rect ) {

	const [ x0, y0, x1, y1 ] = bounds;
	if ( first === last ) {

		rect.set( kids[ first ], [ x0, y0, x1, y1 ] );
		return;

	}

	let acc = 0;
	for ( let k = first; k <= last; k ++ ) acc += weightOf.get( kids[ k ] );
	const horizontal = x1 - x0 > y1 - y0;
	let off = 0;
	for ( let k = first; k <= last; k ++ ) {

		const rel = weightOf.get( kids[ k ] ) / acc;
		if ( horizontal ) rect.set( kids[ k ], [ x0 + ( x1 - x0 ) * off, y0, x0 + ( x1 - x0 ) * ( off + rel ), y1 ] );
		else rect.set( kids[ k ], [ x0, y0 + ( y1 - y0 ) * off, x1, y0 + ( y1 - y0 ) * ( off + rel ) ] );
		off += rel;

	}

}

function normalizedAspect( big, small, relWeight, weightOffset ) {

	const ratio = ( big / ( small * relWeight ) ) * weightOffset * weightOffset;
	return ratio < 1 ? 1 / ratio : ratio;

}
