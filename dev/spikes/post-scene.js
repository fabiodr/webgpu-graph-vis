// Spike scene for the post-processing chain (dev/spikes/post.js). NOT production code.
//
// Provides the two kinds of opaque, depth-writing geometry the screen-space effects need, each with
// a view-space normal written to the G-buffer through `material.mrtNode`:
//
//   SpikeNodes   subclass of src/render/NodeRenderer.js. The impostor sphere material is taken
//                unchanged from the parent and only extended with
//                    material.mrtNode = mrt( { normal: vec4( packNormalToRGB( n ), 1 ) } )
//                where n is recomputed from the parent's varyings (looked up by name). An optional
//                HDR `emissive` multiplier makes the spheres act as light sources for SSGI / bloom.
//   TubeEdges    one instanced quad per edge, straight from node to node, expanded perpendicular to
//                the view ray and shaded / depth-offset as a cylinder impostor (per-pixel normal and
//                depth). Reads edgeInfo / nodePos / nodeColor (3 storage buffers, vertex stage).
//
//   cpuLayout( graph, spread )   small O(N²) spring layout so that the scene is a dense 3D cluster.
//
// MRT contract (see post-chain.js): the scene pass declares  mrt( { output, normal } ); materials
// override single attachments with material.mrtNode (NodeMaterial merges it into the pass MRT:
// `mrt.merge( material.mrtNode )` in NodeMaterial.setup()). Normals are stored packed (n·0.5+0.5)
// in an RGBA8 attachment.
import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, vec3, vec4, float, uniform, varyingProperty, positionGeometry, cameraViewMatrix,
	cameraProjectionMatrix, cameraNear, cameraFar, normalize, dot, cross, max, sqrt, mix, clamp, pow,
	mrt, output, packNormalToRGB, viewZToPerspectiveDepth,
} from 'three/tsl';
import { NodeRenderer } from '../../src/render/NodeRenderer.js';

const KEY_LIGHT = new THREE.Vector3( - 0.45, 0.65, 0.62 ).normalize(); // same as NodeRenderer (view space)

/** NodeRenderer + G-buffer normal + optional emissive boost. `samples` = MSAA samples of the pass it is drawn in. */
export class SpikeNodes extends NodeRenderer {

	constructor( options = {} ) {

		super( options );
		this.passSamples = options.passSamples ?? 4;
		this.emissive = uniform( options.emissive ?? 1 );
		this.haloMRT = options.haloMRT ?? true;

	}

	update( dt, app ) {

		super.update( dt, app );
		// the parent derives the coverage cut from renderer.samples; our scene pass may differ
		this.uniforms.cut.value = this.passSamples > 1 ? 0.002 : 0.5;

	}

	_sphereMaterial( buffers ) {

		const material = super._sphereMaterial( buffers );

		// the parent's varyings, by name
		const vCenter = varyingProperty( 'vec3', 'vNodeCenter' );
		const vViewPos = varyingProperty( 'vec3', 'vNodeViewPos' );
		const vRadius = varyingProperty( 'float', 'vNodeRadius' );

		const normal = Fn( () => {

			const rd = normalize( vViewPos );
			const b = dot( rd, vCenter );
			const m = vCenter.sub( rd.mul( b ) );
			const disc = max( vRadius.mul( vRadius ).sub( dot( m, m ) ), 0 );
			const hit = rd.mul( b.sub( sqrt( disc ) ) );
			return normalize( hit.sub( vCenter ) );

		} )();

		// `output` must be listed too: in a pass WITHOUT an MRT the material's mrtNode is used as it
		// is (NodeMaterial.setup), and an MRT without `output` then has no colour output at all
		material.mrtNode = mrt( { output, normal: vec4( packNormalToRGB( normal ), 1 ) } );

		// HDR boost of the whole hand-lit colour (spheres as emitters for SSGI / bloom tests)
		const base = material.colorNode;
		material.colorNode = Fn( () => {

			const c = vec4( base ).toVar();
			return vec4( c.rgb.mul( this.emissive ), c.a );

		} )();

		return material;

	}

	_haloMaterial( buffers ) {

		const material = super._haloMaterial( buffers );
		// transparent, additive: must not touch the normal attachment. The pass MRT uses
		// MaterialBlending for 'normal' (see post-chain.js), additive × alpha 0 = no change.
		if ( this.haloMRT ) material.mrtNode = mrt( { output, normal: vec4( 0 ) } );
		return material;

	}

}

/** Opaque cylinder-impostor ribbons, one straight tube per edge. */
export class TubeEdges {

	constructor( { radius = 0.3, grey = 0, maxEdges = Infinity } = {} ) {

		this.order = 10;
		this.mesh = null;
		this.maxEdges = maxEdges;
		this.uniforms = {
			radius: uniform( radius ), // × nodes.baseSize, set in setGraph
			grey: uniform( grey ), // 0 = node colour gradient, 1 = neutral grey (to see colour bleeding)
		};
		this.radiusFactor = radius;
		const g = new THREE.BufferGeometry();
		// position = ( t along the edge, side ±1, 0 )
		g.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ 0, - 1, 0, 1, - 1, 0, 1, 1, 0, 0, 1, 0 ] ), 3 ) );
		g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
		this._geometry = g;

	}

	init( app ) {

		this.app = app;

	}

	setGraph( buffers, graph, app ) {

		if ( this.mesh ) {

			this.mesh.removeFromParent();
			this.mesh.material.dispose();

		}

		const U = this.uniforms;
		const baseSize = 0.065 * graph.layoutRadius / Math.cbrt( Math.max( 1, graph.N ) );
		U.radius.value = this.radiusFactor * baseSize;

		const vPos = varyingProperty( 'vec3', 'vTubePos' );
		const vSide = varyingProperty( 'vec3', 'vTubeSide' );
		const vFront = varyingProperty( 'vec3', 'vTubeFront' );
		const vS = varyingProperty( 'float', 'vTubeS' );
		const vColor = varyingProperty( 'vec3', 'vTubeColor' );

		const material = new THREE.NodeMaterial();
		material.name = 'TubeEdges';
		material.lights = false;
		material.fog = false;
		material.transparent = false;
		material.depthWrite = true;
		material.side = THREE.DoubleSide;

		material.vertexNode = Fn( () => {

			const info = buffers.edgeInfo.element( instanceIndex );
			const a = cameraViewMatrix.mul( vec4( buffers.nodePos.element( info.x ).xyz, 1 ) ).xyz;
			const b = cameraViewMatrix.mul( vec4( buffers.nodePos.element( info.y ).xyz, 1 ) ).xyz;
			const t = positionGeometry.x, s = positionGeometry.y;
			const p = mix( a, b, t );
			const axis = normalize( b.sub( a ) );
			const side = normalize( cross( axis, p.negate() ) ); // ⟂ axis and view ray
			const front = normalize( cross( side, axis ) ); // ⟂ axis, towards the camera
			const pos = p.add( side.mul( s.mul( U.radius ) ) );
			vPos.assign( pos );
			vSide.assign( side );
			vFront.assign( front );
			vS.assign( s );
			const ca = buffers.nodeColor.element( info.x ).xyz, cb = buffers.nodeColor.element( info.y ).xyz;
			vColor.assign( mix( mix( ca, cb, t ), vec3( 0.55 ), U.grey ) );
			return cameraProjectionMatrix.mul( vec4( pos, 1 ) );

		} )();

		const tubeNormal = () => {

			const h = sqrt( max( float( 1 ).sub( vS.mul( vS ) ), 0 ) );
			return { h, n: normalize( vSide.mul( vS ).add( vFront.mul( h ) ) ) };

		};

		material.colorNode = Fn( () => {

			const { n } = tubeNormal();
			const L = vec3( KEY_LIGHT.x, KEY_LIGHT.y, KEY_LIGHT.z );
			const V = normalize( vPos.negate() );
			const wrap = clamp( dot( n, L ).mul( 0.5 ).add( 0.5 ), 0, 1 );
			const rim = pow( float( 1 ).sub( clamp( dot( n, V ), 0, 1 ) ), 3 );
			return vec4( vColor.mul( wrap.mul( wrap ).mul( 0.9 ).add( 0.22 ) ).add( vColor.mul( rim ).mul( 0.25 ) ), 1 );

		} )();

		material.depthNode = Fn( () => {

			const { h } = tubeNormal();
			const z = vPos.z.add( vFront.z.mul( h ).mul( U.radius ) );
			return clamp( viewZToPerspectiveDepth( z, cameraNear, cameraFar ), 0, 1 );

		} )();

		material.mrtNode = mrt( { output, normal: Fn( () => vec4( packNormalToRGB( tubeNormal().n ), 1 ) )() } );

		this.mesh = new THREE.Mesh( this._geometry, material );
		this.mesh.name = 'TubeEdges';
		this.mesh.count = Math.min( graph.E, this.maxEdges );
		this.mesh.frustumCulled = false;
		( app || this.app ).scene.add( this.mesh );

	}

	dispose() {

		this.mesh?.removeFromParent();
		this.mesh?.material.dispose();
		this._geometry.dispose();

	}

}

/** Quick O(N²) spring layout (after dev/visual.html), rescaled to `spread` × graph.layoutRadius. */
export function cpuLayout( g, spread = 0.8 ) {

	const N = g.N, E = g.E, L = g.linkDistance;
	const p = new Float64Array( 3 * N ), f = new Float64Array( 3 * N );
	for ( let i = 0; i < N; i ++ ) for ( let k = 0; k < 3; k ++ ) p[ 3 * i + k ] = g.initPos[ 4 * i + k ] * 0.6;
	const iters = 220;
	for ( let it = 0; it < iters; it ++ ) {

		f.fill( 0 );
		for ( let i = 0; i < N; i ++ ) for ( let j = i + 1; j < N; j ++ ) {

			const dx = p[ 3 * i ] - p[ 3 * j ], dy = p[ 3 * i + 1 ] - p[ 3 * j + 1 ], dz = p[ 3 * i + 2 ] - p[ 3 * j + 2 ];
			const s = ( L * L * 1.2 ) / ( dx * dx + dy * dy + dz * dz + 1 );
			f[ 3 * i ] += dx * s; f[ 3 * i + 1 ] += dy * s; f[ 3 * i + 2 ] += dz * s;
			f[ 3 * j ] -= dx * s; f[ 3 * j + 1 ] -= dy * s; f[ 3 * j + 2 ] -= dz * s;

		}

		for ( let e = 0; e < E; e ++ ) {

			const a = g.edges[ 2 * e ], b = g.edges[ 2 * e + 1 ];
			const dx = p[ 3 * a ] - p[ 3 * b ], dy = p[ 3 * a + 1 ] - p[ 3 * b + 1 ], dz = p[ 3 * a + 2 ] - p[ 3 * b + 2 ];
			const d = Math.sqrt( dx * dx + dy * dy + dz * dz ) + 1e-6;
			const s = ( ( d - L ) / d ) * 0.5 / Math.min( g.degree[ a ], g.degree[ b ] );
			f[ 3 * a ] -= dx * s; f[ 3 * a + 1 ] -= dy * s; f[ 3 * a + 2 ] -= dz * s;
			f[ 3 * b ] += dx * s; f[ 3 * b + 1 ] += dy * s; f[ 3 * b + 2 ] += dz * s;

		}

		const temp = L * 0.6 * ( 1 - it / iters ) + 0.5;
		for ( let i = 0; i < N; i ++ ) {

			for ( let k = 0; k < 3; k ++ ) f[ 3 * i + k ] -= p[ 3 * i + k ] * 0.012;
			const fl = Math.hypot( f[ 3 * i ], f[ 3 * i + 1 ], f[ 3 * i + 2 ] ) + 1e-9;
			const s = Math.min( fl, temp ) / fl;
			for ( let k = 0; k < 3; k ++ ) p[ 3 * i + k ] += f[ 3 * i + k ] * s;

		}

	}

	const c = [ 0, 0, 0 ];
	for ( let i = 0; i < N; i ++ ) for ( let k = 0; k < 3; k ++ ) c[ k ] += p[ 3 * i + k ] / N;
	const d = Array.from( { length: N }, ( _, i ) => Math.hypot( p[ 3 * i ] - c[ 0 ], p[ 3 * i + 1 ] - c[ 1 ], p[ 3 * i + 2 ] - c[ 2 ] ) ).sort( ( a, b ) => a - b );
	const scale = ( spread * g.layoutRadius ) / Math.max( 1e-6, d[ Math.floor( 0.9 * ( N - 1 ) ) ] );
	const out = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) for ( let k = 0; k < 3; k ++ ) out[ 4 * i + k ] = ( p[ 3 * i + k ] - c[ k ] ) * scale;
	return out;

}
