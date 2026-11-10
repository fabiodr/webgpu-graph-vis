// GPU-side graph state shared by every module (force layout, renderers, interaction).
// Each field is a TSL storage node (`instancedArray`); its backing attribute is `node.value`
// (a StorageInstancedBufferAttribute whose `.array` can be rewritten from the CPU followed by
// `.needsUpdate = true`).
//
// ┌───────────────────┬───────┬───────┬────────────────────────────────────────────────────┐
// │ field             │ type  │ count │ meaning                                            │
// ├───────────────────┼───────┼───────┼────────────────────────────────────────────────────┤
// │ nodePos           │ vec4  │ N     │ xyz world position, w = pinned (1) / free (0)       │
// │ nodeVel           │ vec4  │ N     │ xyz velocity per tick (force layout), w = heat 0..1 │
// │                   │       │       │   (local re-activation around a dragged node)       │
// │ nodeForce         │ vec4  │ N     │ scratch of the layout: (repulsion, stiffness), then │
// │                   │       │       │   (preconditioned force, heat)                      │
// │ nodeTarget        │ vec4  │ N     │ ABSOLUTE target position xyz (world units) of the   │
// │                   │       │       │   current target layout (sphere, radial, stress, …), │
// │                   │       │       │   w = weight 0..1 of the pull. Written by the layout │
// │                   │       │       │   module; initially the paper's sphere layout        │
// │                   │       │       │   (sphereDir · layoutRadius, w = 1)                  │
// │ nodeInfo          │ uvec4 │ N     │ (adjStart, adjCount, finest cluster id, layout mass │
// │                   │       │       │   in 1/256 units: round(256·(1 + deg/avgDeg)^p))     │
// │ nodeColor         │ vec4  │ N     │ linear RGB, w = radius factor (≈0.7..1.6). Animated: │
// │                   │       │       │   eased toward nodeColorTarget every frame          │
// │ nodeColorTarget   │ vec4  │ N     │ CPU-written targets for nodeColor (setColors)       │
// │ nodePrev          │ vec4  │ N     │ nodePos at the previous rendered frame (motion       │
// │                   │       │       │   vectors); kept by snapshotPrev() while             │
// │                   │       │       │   gfxState.velocity is on, see render/gfx/shared.js  │
// │ nodeFx            │ vec4  │ N     │ CPU-written effect channels (analysis tools):        │
// │                   │       │       │   x hop distance from the ripple / k-hop source      │
// │                   │       │       │     (−1 = none), y position along the highlighted    │
// │                   │       │       │     path in [0,1] (−1 = not on it), z glow weight    │
// │                   │       │       │     0..1, w normalized metric value 0..1             │
// │ nodeState         │ vec4  │ N     │ animated: x selected, y neighbor-of-selected,       │
// │                   │       │       │   z dimmed (context when a selection exists),      │
// │                   │       │       │   w hovered. Each in [0,1], eased toward target.    │
// │ nodeStateTarget   │ vec4  │ N     │ CPU-written targets for nodeState                  │
// │ adjList           │ uint  │ 2E    │ CSR neighbor lists                                 │
// │ anc               │ uint  │ N·D   │ anc[i*D+l] = cluster of node i at level l           │
// │ clusterPos        │ vec4  │ C     │ xyz centroid (mass-weighted), w = bounding radius   │
// │ clusterInfo       │ uvec4 │ C     │ (childStart, childCount, height, mass)              │
// │ childList         │ uint  │ Σ     │ children (node idx for height 1, else cluster idx)  │
// │ bhTree            │ uvec2 │ T     │ preorder (ref, skip); ref<N node, else N+cluster    │
// │ edgeInfo          │ uvec4 │ E     │ (src, tgt, lcaLevel, 0)                             │
// └───────────────────┴───────┴───────┴────────────────────────────────────────────────────┘
//
// Shared uniforms live in `buffers.uniforms` (see below). Buffers are allocated with at least one
// element so empty graphs/hierarchies never create zero-sized bindings.

import { Fn, instanceIndex, instancedArray, uniform } from 'three/tsl';
import { Vector3 } from 'three/webgpu';

const at = ( n ) => Math.max( 1, n );

export class GraphBuffers {

	/** @param {import('../graph/prepare.js').PreparedGraph} g */
	constructor( g ) {

		this.graph = g;
		const { N, E, D, C, T } = g;
		this.N = N; this.E = E; this.D = D; this.C = C; this.T = T;

		this.nodePos = instancedArray( at( N ), 'vec4' ).setName( 'nodePos' );
		this.nodeVel = instancedArray( at( N ), 'vec4' ).setName( 'nodeVel' );
		this.nodeForce = instancedArray( at( N ), 'vec4' ).setName( 'nodeForce' );
		this.nodeTarget = instancedArray( at( N ), 'vec4' ).setName( 'nodeTarget' );
		this.nodeInfo = instancedArray( at( N ), 'uvec4' ).setName( 'nodeInfo' );
		this.nodeColor = instancedArray( at( N ), 'vec4' ).setName( 'nodeColor' );
		this.nodeColorTarget = instancedArray( at( N ), 'vec4' ).setName( 'nodeColorTarget' );
		this.nodePrev = instancedArray( at( N ), 'vec4' ).setName( 'nodePrev' );
		this.nodeFx = instancedArray( at( N ), 'vec4' ).setName( 'nodeFx' );
		this.nodeState = instancedArray( at( N ), 'vec4' ).setName( 'nodeState' );
		this.nodeStateTarget = instancedArray( at( N ), 'vec4' ).setName( 'nodeStateTarget' );
		this.adjList = instancedArray( at( 2 * E ), 'uint' ).setName( 'adjList' );
		this.anc = instancedArray( at( N * D ), 'uint' ).setName( 'anc' );
		this.clusterPos = instancedArray( at( C ), 'vec4' ).setName( 'clusterPos' );
		this.clusterInfo = instancedArray( at( C ), 'uvec4' ).setName( 'clusterInfo' );
		this.childList = instancedArray( at( g.childList.length ), 'uint' ).setName( 'childList' );
		this.bhTree = instancedArray( at( T ), 'uvec2' ).setName( 'bhTree' );
		this.edgeInfo = instancedArray( at( E ), 'uvec4' ).setName( 'edgeInfo' );

		// fill static data
		this.nodePos.value.array.set( g.initPos );
		this.nodePrev.value.array.set( g.initPos );
		const fx = this.nodeFx.value.array;
		for ( let i = 0; i < N; i ++ ) fx[ 4 * i ] = fx[ 4 * i + 1 ] = - 1;
		// absolute targets, initially the paper's sphere layout; the layout module rewrites them
		const target = this.nodeTarget.value.array;
		for ( let i = 0; i < N; i ++ ) {

			target[ 4 * i ] = g.sphereDir[ 4 * i ] * g.layoutRadius;
			target[ 4 * i + 1 ] = g.sphereDir[ 4 * i + 1 ] * g.layoutRadius;
			target[ 4 * i + 2 ] = g.sphereDir[ 4 * i + 2 ] * g.layoutRadius;
			target[ 4 * i + 3 ] = 1;

		}

		// z: finest cluster of the node (the root when there is no hierarchy); w: layout mass for
		// hub spacing 1 (fixed point, 256 = 1; the layout module rewrites it for other exponents)
		const info = this.nodeInfo.value.array;
		const avgDegree = N ? Math.max( 1e-9, ( 2 * E ) / N ) : 1;
		for ( let i = 0; i < N; i ++ ) {

			info[ 4 * i ] = g.adjStart[ i ];
			info[ 4 * i + 1 ] = g.degree[ i ];
			info[ 4 * i + 2 ] = D > 0 ? g.anc[ i * D ] : g.root;
			info[ 4 * i + 3 ] = Math.max( 1, Math.round( 256 * ( 1 + g.degree[ i ] / avgDegree ) ) );

		}

		this.setColors( g.colors, g.radii, { upload: false, immediate: true } );
		this.adjList.value.array.set( g.adjList.subarray( 0, Math.min( g.adjList.length, this.adjList.value.array.length ) ) );
		this.anc.value.array.set( g.anc.subarray( 0, Math.min( g.anc.length, this.anc.value.array.length ) ) );
		const ci = this.clusterInfo.value.array;
		for ( let c = 0; c < C; c ++ ) {

			ci[ 4 * c ] = g.clusterChildStart[ c ];
			ci[ 4 * c + 1 ] = g.clusterChildCount[ c ];
			ci[ 4 * c + 2 ] = g.clusterHeight[ c ];
			ci[ 4 * c + 3 ] = g.clusterMass[ c ];

		}

		this.childList.value.array.set( g.childList );
		this.bhTree.value.array.set( g.bhTree.subarray( 0, 2 * at( T ) ) );
		const ei = this.edgeInfo.value.array;
		for ( let e = 0; e < E; e ++ ) {

			ei[ 4 * e ] = g.edges[ 2 * e ];
			ei[ 4 * e + 1 ] = g.edges[ 2 * e + 1 ];
			ei[ 4 * e + 2 ] = g.edgeLca[ e ];

		}

		// initial cluster centroids (CPU) so renderers have sane data before the first compute
		this.computeClusterCentroidsCPU( g.initPos );

		/**
		 * Shared uniforms.
		 *  layoutBlend   0..1 "sphere-ness": 1 only in the paper's sphere layout (animated by the
		 *                layout module; edge curves and the sphere guide follow it)
		 *  sphereRadius  radius of the sphere layout in world units
		 *  focus         0..1 eased; 1 while a selection exists (drives context dimming)
		 *  hoverIndex    hovered node or -1
		 *  selectedIndex primary selected node or -1
		 *  time          seconds since start
		 *  levelOffset/levelScale/levelExponent  paper's cluster "level" radii for sphere mode:
		 *                level(h) = 1 + offset + scale * (h-1)^exponent  (h = cluster height)
		 */
		this.uniforms = {
			layoutBlend: uniform( 0 ),
			sphereRadius: uniform( g.layoutRadius ),
			focus: uniform( 0 ),
			hoverIndex: uniform( - 1, 'int' ),
			selectedIndex: uniform( - 1, 'int' ),
			time: uniform( 0 ),
			levelOffset: uniform( 0.1 ),
			levelScale: uniform( 0.35 ),
			levelExponent: uniform( 2 ),
			graphCenter: uniform( new Vector3() ),
			// 0..1, eased by the layout: how strongly nodes are pulled to nodeTarget (any target
			// layout: sphere, radial, stress, …). layoutBlend stays "sphere-ness" (edge curves, guide).
			targetBlend: uniform( 0 ),
			// ripple (analysis tools): nodes pulse while |nodeFx.x − rippleFront| < rippleWidth (hops)
			rippleFront: uniform( - 1e6 ),
			rippleWidth: uniform( 0.75 ),
			rippleGain: uniform( 0 ),
		};

		/**
		 * app.frame of the last frame in which node positions changed on the GPU (layout ticks,
		 * drags, setPositions). Consumers (shadow map, edge curve cache, …) compare it with the
		 * frame they last processed. Infinity = "assume moved" until a layout module maintains it.
		 */
		this.moveFrame = Infinity;
		this._snapshotKernel = null;

	}

	/**
	 * Replace node colors (linear RGB vec4 array) and optionally radii. Writes the easing targets;
	 * the GPU animates nodeColor toward them (layout state kernel, ~0.25 s). `immediate` also writes
	 * nodeColor itself (no animation). Legacy call setColors( colors, radii, uploadBoolean ) works.
	 */
	setColors( colors, radii = this.graph.radii, options = {} ) {

		const { upload = true, immediate = false } = typeof options === 'boolean' ? { upload: options } : options;
		const arr = this.nodeColorTarget.value.array;
		for ( let i = 0; i < this.N; i ++ ) {

			arr[ 4 * i ] = colors[ 4 * i ];
			arr[ 4 * i + 1 ] = colors[ 4 * i + 1 ];
			arr[ 4 * i + 2 ] = colors[ 4 * i + 2 ];
			arr[ 4 * i + 3 ] = radii[ i ];

		}

		if ( immediate ) this.nodeColor.value.array.set( arr );
		if ( upload ) {

			this.nodeColorTarget.value.needsUpdate = true;
			if ( immediate ) this.nodeColor.value.needsUpdate = true;

		}

	}

	/** Write the CPU-side effect channels (Float32Array 4N, see nodeFx in the table above). */
	setFx( fx4 ) {

		this.nodeFx.value.array.set( fx4 );
		this.nodeFx.value.needsUpdate = true;

	}

	/**
	 * Copies nodePos → nodePrev on the GPU. The App calls it once per frame before the modules
	 * update (i.e. before the layout moves anything) while motion vectors are in use, so during
	 * rendering nodePrev holds the positions of the previous rendered frame.
	 */
	snapshotPrev( renderer ) {

		if ( ! this._snapshotKernel ) {

			const { nodePos, nodePrev } = this;
			this._snapshotKernel = Fn( () => {

				nodePrev.element( instanceIndex ).assign( nodePos.element( instanceIndex ) );

			} )().compute( at( this.N ) ).setName( 'graph.snapshotPrev' );

		}

		renderer.compute( this._snapshotKernel );

	}

	/** Overwrite node positions from the CPU (vec4 array). Also resets velocities. */
	setPositions( pos4 ) {

		this.nodePos.value.array.set( pos4 );
		this.nodePos.value.needsUpdate = true;
		this.nodePrev.value.array.set( pos4 );
		this.nodePrev.value.needsUpdate = true;
		this.moveFrame = Infinity;
		this.nodeVel.value.array.fill( 0 );
		this.nodeVel.value.needsUpdate = true;
		this.computeClusterCentroidsCPU( pos4 );

	}

	/** CPU fallback for cluster centroids/radii (used before the GPU layout has run). */
	computeClusterCentroidsCPU( pos4 ) {

		const g = this.graph;
		const cp = this.clusterPos.value.array;
		cp.fill( 0 );
		for ( const { height, start, count } of g.heightRanges ) {

			for ( let c = start; c < start + count; c ++ ) {

				let x = 0, y = 0, z = 0, m = 0;
				const s = g.clusterChildStart[ c ], n = g.clusterChildCount[ c ];
				const nodeChildren = height === 1 || g.D === 0;
				for ( let k = 0; k < n; k ++ ) {

					const ch = g.childList[ s + k ];
					if ( nodeChildren ) {

						x += pos4[ 4 * ch ]; y += pos4[ 4 * ch + 1 ]; z += pos4[ 4 * ch + 2 ]; m += 1;

					} else {

						const w = g.clusterMass[ ch ];
						x += cp[ 4 * ch ] * w; y += cp[ 4 * ch + 1 ] * w; z += cp[ 4 * ch + 2 ] * w; m += w;

					}

				}

				m = Math.max( m, 1e-9 );
				x /= m; y /= m; z /= m;
				let r = 0;
				for ( let k = 0; k < n; k ++ ) {

					const ch = g.childList[ s + k ];
					if ( nodeChildren ) r = Math.max( r, Math.hypot( pos4[ 4 * ch ] - x, pos4[ 4 * ch + 1 ] - y, pos4[ 4 * ch + 2 ] - z ) );
					else r = Math.max( r, Math.hypot( cp[ 4 * ch ] - x, cp[ 4 * ch + 1 ] - y, cp[ 4 * ch + 2 ] - z ) + cp[ 4 * ch + 3 ] );

				}

				cp[ 4 * c ] = x; cp[ 4 * c + 1 ] = y; cp[ 4 * c + 2 ] = z; cp[ 4 * c + 3 ] = r;

			}

		}

		this.clusterPos.value.needsUpdate = true;

	}

	/** Write CPU-side highlight targets (Float32Array 4N) for nodeState easing. */
	setStateTargets( targets4 ) {

		this.nodeStateTarget.value.array.set( targets4 );
		this.nodeStateTarget.value.needsUpdate = true;

	}

	dispose() {

		this._snapshotKernel?.dispose?.();
		this._snapshotKernel = null;
		for ( const key of Object.keys( this ) ) {

			const v = this[ key ];
			if ( v && v.isStorageBufferNode && v.value && v.value.dispose ) v.value.dispose?.();

		}

	}

}
