// Background: the "dark observatory" behind the graph, plus the paper's faint latitude/longitude
// guide sphere that fades in with the sphere layout.
//
//  • scene.backgroundNode: near-black, slightly blue radial gradient (brightest a little above
//    the screen center) times a very faint large-scale pattern defined on the view direction, so
//    it turns with the camera like a distant nebula instead of sticking to the screen. Values are
//    tiny (≈ 1–3 % linear); PostFX dithers after the sRGB encode so the gradient does not band.
//  • guide sphere: thin anti-aliased lat/long lines (fwidth-based, constant pixel width) on a
//    sphere of radius uniforms.sphereRadius around uniforms.graphCenter, additive, no depth write,
//    opacity × uniforms.layoutBlend (0 in force mode → the mesh is skipped entirely on the CPU).
//    Longitude lines fade towards the poles where they converge; back-facing lines are dimmer.
//
// Public API:
//   const bg = app.use( new Background( { noise: true, guide: true } ) );   // order 12
//   bg.guide = false;              // hide the sphere guide
//   bg.guideOpacity = 0.08;        // peak line intensity (live)
//   bg.setColors( center, edge );  // linear RGB arrays (live)
//   bg.setGraph( buffers, graph, app ), bg.dispose()
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, vec2, vec3, vec4, float, screenUV, screenSize, length, exp, mix, smoothstep,
	normalWorldGeometry, uv, fract, abs, fwidth, clamp, max, sin, frontFacing,
	positionGeometry, select,
} from 'three/tsl';

export class Background {

	constructor( { noise = true, guide = true } = {} ) {

		this.order = 12;
		this.app = null;
		this.noise = noise;
		this._guide = guide;
		this.guideMesh = null;
		this.buffers = null;

		this.uniforms = {
			// pre-compensated for the Neutral tone mapper's quadratic toe (which subtracts the
			// min channel): display ≈ sRGB (6, 9, 18) at the center, (1, 2, 4) at the corners
			center: uniform( new THREE.Color( 0.0170, 0.0179, 0.0212 ) ),
			edge: uniform( new THREE.Color( 0.0069, 0.0072, 0.0078 ) ),
			noiseAmount: uniform( 0.22 ),
			guideOpacity: uniform( 0.08 ),
		};

	}

	get guide() {

		return this._guide;

	}

	set guide( v ) {

		this._guide = !! v;

	}

	get guideOpacity() {

		return this.uniforms.guideOpacity.value;

	}

	set guideOpacity( v ) {

		this.uniforms.guideOpacity.value = v;

	}

	setColors( center, edge ) {

		this.uniforms.center.value.setRGB( ...center );
		this.uniforms.edge.value.setRGB( ...edge );

	}

	init( app ) {

		this.app = app;
		app.scene.backgroundNode = this._backgroundNode();

	}

	_backgroundNode() {

		const U = this.uniforms;
		return Fn( () => {

			const aspect = screenSize.x.div( max( screenSize.y, 1 ) );
			const p = screenUV.sub( vec2( 0.5, 0.44 ) ).mul( vec2( aspect, 1 ) );
			const r2 = p.dot( p );
			const radial = exp( r2.mul( - 2.4 ) );
			const c = mix( vec3( U.edge ), vec3( U.center ), radial ).toVar();
			if ( this.noise ) {

				// very low-frequency "nebula" on the view direction: a few domain-warped sines
				// (far cheaper per pixel than gradient noise, and only ±20 % of a ~1 % signal)
				const d = normalWorldGeometry;
				const w = sin( d.y.mul( 1.7 ).add( d.z.mul( 0.9 ) ) ).mul( 1.3 );
				const n = sin( d.x.mul( 2.1 ).add( w ) ).mul( sin( d.y.mul( 2.6 ).add( d.z.mul( 1.9 ) ).add( 0.7 ) ) ).mul( 0.6 )
					.add( sin( d.z.mul( 3.3 ).add( d.x.mul( 1.1 ) ).add( w.mul( 0.5 ) ) ).mul( 0.4 ) );
				c.mulAssign( float( 1 ).add( n.mul( U.noiseAmount ) ) );

			}

			return vec4( c, 1 );

		} )();

	}

	setGraph( buffers, graph, app ) {

		this._disposeGuide();
		this.buffers = buffers;
		const U = this.uniforms;
		const S = buffers.uniforms;

		const geometry = new THREE.SphereGeometry( 1, 144, 72 );
		const material = new THREE.NodeMaterial();
		material.name = 'Background.guide';
		material.transparent = true;
		material.depthWrite = false;
		material.depthTest = true;
		material.blending = THREE.AdditiveBlending;
		material.side = THREE.DoubleSide;
		material.lights = false;
		material.fog = false;
		material.positionNode = positionGeometry.mul( S.sphereRadius ).add( S.graphCenter );

		const line = ( f ) => {

			// distance to the nearest integer of f, in pixels → 1 px wide AA line
			const d = abs( fract( f.sub( 0.5 ) ).sub( 0.5 ) ).div( max( fwidth( f ), 1e-5 ) );
			return clamp( float( 1.1 ).sub( d ), 0, 1 );

		};

		material.colorNode = Fn( () => {

			const t = uv();
			const lat = sin( t.y.mul( Math.PI ) ); // 0 at the poles
			const lon = line( t.x.mul( 24 ) ).mul( smoothstep( 0.05, 0.35, lat ) );
			const par = line( t.y.mul( 12 ) ).mul( smoothstep( 0.0, 0.12, lat ) );
			const equator = line( t.y.mul( 2 ) ).mul( 0.6 );
			const l = max( max( lon, par ), equator );
			const facing = select( frontFacing, float( 1 ), float( 0.4 ) );
			const a = l.mul( facing ).mul( U.guideOpacity ).mul( S.layoutBlend );
			return vec4( vec3( 0.42, 0.58, 0.95 ).mul( a ), 1 );

		} )();

		const mesh = new THREE.Mesh( geometry, material );
		mesh.name = 'Background.guide';
		mesh.frustumCulled = false;
		mesh.renderOrder = - 10;
		mesh.visible = false;
		( app || this.app ).scene.add( mesh );
		this.guideMesh = mesh;

	}

	update() {

		if ( ! this.guideMesh || ! this.buffers ) return;
		this.guideMesh.visible = this._guide && this.buffers.uniforms.layoutBlend.value > 0.002;

	}

	_disposeGuide() {

		if ( ! this.guideMesh ) return;
		this.guideMesh.removeFromParent();
		this.guideMesh.geometry.dispose();
		this.guideMesh.material.dispose();
		this.guideMesh = null;

	}

	dispose() {

		this._disposeGuide();
		if ( this.app && this.app.scene ) this.app.scene.backgroundNode = null;

	}

}
