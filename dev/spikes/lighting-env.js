// Spike "lighting" — procedural HDR environment -> PMREM, without loading any file.
//
// A tiny scene (one inside-out sphere whose colour is a TSL function of the view direction: dark
// nebula gradient + a warm key soft box + a cool fill + a thin magenta rim strip) is rendered by
// PMREMGenerator.fromScene() into a CubeRenderTarget (256² RGBA16F, 6 mip levels, GGX-prefiltered).
// The result is used as scene.environment; regenerating renders into the SAME target, so no
// material has to be rebuilt.
//
//   const env = createEnvironment( renderer );
//   scene.environment = env.texture;                 // PMREM cube texture (isPMREMTexture)
//   scene.environmentIntensity = 1;                  // runtime uniform
//   scene.environmentRotation.y = 0.5;               // runtime uniform (Euler)
//   env.uniforms.keyIntensity.value = 20; env.update();   // re-render + re-filter in place
import * as THREE from 'three/webgpu';
import { Fn, float, vec3, uniform, normalize, positionLocal, dot, smoothstep, mix, max, pow, abs, mx_fractal_noise_float } from 'three/tsl';

/** Soft-edged disc light around direction `dir` (cosine of the half angle `cosSize`). */
const softBox = ( d, dir, cosSize, soft ) => smoothstep( cosSize - soft, cosSize + soft, dot( d, dir ) );

export function createEnvironment( renderer, { size = 256 } = {} ) {

	const uniforms = {
		keyDir: uniform( new THREE.Vector3( - 0.5, 0.6, 0.62 ).normalize() ),
		keyColor: uniform( new THREE.Color( 1.0, 0.92, 0.8 ) ),
		keyIntensity: uniform( 14 ),
		fillColor: uniform( new THREE.Color( 0.25, 0.45, 1.0 ) ),
		fillIntensity: uniform( 1.2 ),
		rimColor: uniform( new THREE.Color( 1.0, 0.25, 0.7 ) ),
		rimIntensity: uniform( 3 ),
		nebula: uniform( 0.35 ),
	};

	const sky = Fn( () => {

		const d = normalize( positionLocal ).toVar();
		// base: almost black void, slightly lighter and bluer towards the zenith
		const base = mix( vec3( 0.004, 0.005, 0.009 ), vec3( 0.02, 0.035, 0.07 ), smoothstep( - 1, 1, d.y ) );
		// nebula: low-frequency fractal noise tinted between violet and teal
		const n = mx_fractal_noise_float( d.mul( 1.7 ), 4, 2, 0.5 ).mul( 0.5 ).add( 0.5 );
		const tint = mix( vec3( 0.45, 0.12, 0.6 ), vec3( 0.08, 0.5, 0.55 ), smoothstep( 0.3, 0.7, mx_fractal_noise_float( d.mul( 0.9 ).add( 7.3 ), 2, 2, 0.5 ).mul( 0.5 ).add( 0.5 ) ) );
		const neb = tint.mul( pow( max( n, 0 ), 3 ) ).mul( uniforms.nebula );
		// lights (HDR): key soft box, big cool fill on the opposite side, thin rim strip near the horizon behind
		const key = uniforms.keyColor.mul( uniforms.keyIntensity ).mul( softBox( d, uniforms.keyDir, 0.93, 0.03 ) );
		const fill = uniforms.fillColor.mul( uniforms.fillIntensity ).mul( softBox( d, uniforms.keyDir.negate(), 0.55, 0.35 ) );
		const strip = smoothstep( 0.12, 0.02, abs( d.y.add( 0.15 ) ) ).mul( smoothstep( 0.2, 0.8, d.z.negate() ) );
		const rim = uniforms.rimColor.mul( uniforms.rimIntensity ).mul( strip );
		return base.add( neb ).add( key ).add( fill ).add( rim );

	} );

	const material = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide, depthWrite: false, fog: false } );
	material.colorNode = sky();
	const envScene = new THREE.Scene();
	envScene.add( new THREE.Mesh( new THREE.SphereGeometry( 10, 48, 24 ), material ) );

	const generator = new THREE.PMREMGenerator( renderer );
	let target = null;
	const timings = [];

	const update = () => {

		const t0 = performance.now();
		// passing the previous target re-renders into it: same GPU texture, no material rebuild
		target = generator.fromScene( envScene, 0, 0.1, 100, { size, renderTarget: target } );
		timings.push( performance.now() - t0 ); // CPU-side submit time only (GPU work is asynchronous)
		return target;

	};

	update();

	return {
		uniforms, envScene, timings, update,
		get texture() {

			return target.texture;

		},
		dispose() {

			generator.dispose();
			target.dispose();
			material.dispose();

		},
	};

}
