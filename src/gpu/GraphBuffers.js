// GPU-side graph state shared by every module (force layout, renderers, interaction).
// Each field is a TSL storage node (`instancedArray`); its backing attribute is `node.value`
// (a StorageInstancedBufferAttribute whose `.array` can be rewritten from the CPU followed by
// `.needsUpdate = true`).
//
// ┌───────────────────┬───────┬───────┬────────────────────────────────────────────────────┐
// │ field             │ type  │ count │ meaning                                            │
// ├───────────────────┼───────┼───────┼────────────────────────────────────────────────────┤
// │ nodePos           │ vec4  │ N     │ xyz world position, w = pinned (1) / free (0)       │
// │ nodeVel           │ vec4  │ N     │ xyz velocity (force layout), w unused               │
// │ nodeForce         │ vec4  │ N     │ scratch: repulsion accumulated by the layout        │
// │ nodeTarget        │ vec4  │ N     │ unit direction of the paper's sphere layout         │
// │ nodeInfo          │ uvec4 │ N     │ (adjStart, adjCount, 0, 0)                          │
// │ nodeColor         │ vec4  │ N     │ linear RGB, w = radius factor (≈0.7..1.6)          │
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

import { instancedArray, uniform } from 'three/tsl';
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
		this.nodeTarget.value.array.set( g.sphereDir );
		const info = this.nodeInfo.value.array;
		for ( let i = 0; i < N; i ++ ) {

			info[ 4 * i ] = g.adjStart[ i ];
			info[ 4 * i + 1 ] = g.degree[ i ];

		}

		this.setColors( g.colors, g.radii, false );
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
		 *  layoutBlend   0 = force layout, 1 = paper sphere layout (animated by the layout module)
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
		};

	}

	/** Replace node colors (linear RGB vec4 array) and optionally radii. */
	setColors( colors, radii = this.graph.radii, upload = true ) {

		const arr = this.nodeColor.value.array;
		for ( let i = 0; i < this.N; i ++ ) {

			arr[ 4 * i ] = colors[ 4 * i ];
			arr[ 4 * i + 1 ] = colors[ 4 * i + 1 ];
			arr[ 4 * i + 2 ] = colors[ 4 * i + 2 ];
			arr[ 4 * i + 3 ] = radii[ i ];

		}

		if ( upload ) this.nodeColor.value.needsUpdate = true;

	}

	/** Overwrite node positions from the CPU (vec4 array). Also resets velocities. */
	setPositions( pos4 ) {

		this.nodePos.value.array.set( pos4 );
		this.nodePos.value.needsUpdate = true;
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

		for ( const key of Object.keys( this ) ) {

			const v = this[ key ];
			if ( v && v.isStorageBufferNode && v.value && v.value.dispose ) v.value.dispose?.();

		}

	}

}
