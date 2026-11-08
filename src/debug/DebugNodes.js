// Minimal node renderer for dev pages: one camera-facing sprite per node, colored, read straight
// from the GPU position buffer. The real renderer lives in src/render/NodeRenderer.js.
import * as THREE from 'three/webgpu';
import { instanceIndex, vec3, float, uv, smoothstep, length, vec2 } from 'three/tsl';

export class DebugNodes {

	constructor( { size = 4 } = {} ) {

		this.order = 10;
		this.size = size;

	}

	init( app ) {

		this.app = app;

	}

	setGraph( buffers, graph, app ) {

		if ( this.mesh ) {

			app.scene.remove( this.mesh );
			this.mesh.material.dispose();

		}

		const material = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false } );
		const c = buffers.nodeColor.element( instanceIndex );
		material.positionNode = buffers.nodePos.element( instanceIndex ).xyz;
		material.scaleNode = c.w.mul( this.size );
		const d = length( uv().sub( vec2( 0.5 ) ) ).mul( 2 );
		material.colorNode = vec3( c.xyz ).mul( 1.5 );
		material.opacityNode = smoothstep( float( 1 ), float( 0.8 ), d );
		const mesh = new THREE.Sprite( material );
		mesh.count = graph.N;
		mesh.frustumCulled = false;
		app.scene.add( mesh );
		this.mesh = mesh;

	}

}
