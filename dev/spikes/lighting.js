// Spike "lighting" — physically based lighting, shadows and a G-buffer for impostor spheres and
// strand ribbons (see lighting-impostors.js for the materials, lighting-env.js for the environment).
//
// Scenes (?scene=):
//   field   (default) N impostor spheres with random roughness / metalness / colour / emissive core,
//           C cubic Bézier tubes between random spheres; environment + one directional key light.
//   chart   A/B check: rows of REAL SphereGeometry meshes above rows of impostor spheres with the same
//           roughness (x axis) and metalness (two blocks), plus real TubeGeometry vs strand tubes,
//           over a shadow-receiving ground plane.
//   shadow  a large impostor sphere between the light and a wall of small spheres and tubes; a real
//           plane behind the wall shows the shapes of the cast shadows.
//   strands fans of strands whose width goes from 8 px down to 0.3 px (tube ↔ thin-strand shading).
//
// Query parameters:
//   n=600 c=300          sphere / curve count (field)
//   strand=auto|tube|thin  strand shading mode        physical=1  MeshPhysicalNodeMaterial + clearcoat
//   aniso=<0..1>         with physical=1: anisotropy on all spheres (real ones too), tangent around the y axis
//   blend=1 opacity=0.5  alpha-blended strands (no depth write) instead of opaque + alpha-to-coverage
//   sdither=1            strands thinner than a shadow-map texel cast proportionally lighter shadows
//   formats=1|0          cheaper G-buffer formats (rgba8 / rg8 / rg16f) instead of RGBA16F everywhere
//   vnode=1              negative test: spheres through material.vertexNode instead of positionNode
//   point=1 override=0   a close point light; override=0 (negative test) drops the positionView overrides
//   bytes=32|64          requested maxColorAttachmentBytesPerSample (blend=1 or formats=0 need 64)
//   shadow=pcf|basic|vsm|off|pcfsoft   map=1024  radius=2  bias=-0.0006  nbias=0.04  shadowrate=1
//   msaa=4|0             samples of the scene pass    mrt=1|0     G-buffer attachments
//   view=beauty|normal|emissive|metalrough|velocity|depth|grid
//   ao=<0..1>            constant ambient occlusion through scenePass.contextNode ( getAO )
//   env=1|0  envint=1  envrot=0 (radians, y)  light=3 (intensity)  regen=0 (re-generate the PMREM n times)
//   ldir=x,y,z (direction towards the light)  cam=main|light (look through the shadow camera)
//   zoom=1 yaw=0 pitch=0  width=1 (strand width multiplier)  minpx=1  exposure=1  spin=0
//
// window.__stats: timings and counters for the headless harness (scripts/shot.mjs --report).
import * as THREE from 'three/webgpu';
import {
	pass, mrt, output, normalView, packNormalToRGB, emissive, metalness, roughness, velocity, vec2, vec3, vec4, float,
	screenUV, renderOutput, instancedArray, uniform, select, floor, fract, perspectiveDepthToViewZ, context,
} from 'three/tsl';
import { createEnvironment } from './lighting-env.js';
import {
	createSphereMaterial, createStrandMaterial, quadGeometry, ribbonGeometry, velocityUniforms, updateVelocityUniforms, commitVelocityUniforms,
} from './lighting-impostors.js';

const params = new URLSearchParams( location.search );
const P = ( k, d ) => ( params.has( k ) ? params.get( k ) : d );
const num = ( k, d ) => + P( k, d );

const sceneName = P( 'scene', 'field' );
const view = P( 'view', 'beauty' );
const msaa = num( 'msaa', 4 );
const useMRT = num( 'mrt', 1 ) === 1;
const shadowMode = P( 'shadow', 'pcf' );
const strandMode = P( 'strand', 'auto' );
const physical = num( 'physical', 0 ) === 1;
const anisotropy = num( 'aniso', 0 );

const GBUFFER = [ 'normal', 'emissive', 'metalrough', 'velocity' ]; // MRT attachments besides 'output'

const stats = window.__stats = { frames: 0, shadowRenders: 0 };

/** Deterministic RNG. */
function mulberry32( a ) {

	return () => {

		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul( a ^ a >>> 15, 1 | a );
		t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};

}

const rng = mulberry32( 7 );
const _c = new THREE.Color();
const hue = ( h, s = 0.75, l = 0.55 ) => _c.setHSL( h, s, l ).toArray(); // linear working space

// ── scene descriptions: spheres [ x, y, z, r, cr, cg, cb, rough, metal, emissive, clearcoat ], curves [ p0, p1, p2, p3, color, radius ] ──

function sceneField() {

	const N = num( 'n', 600 ), C = num( 'c', 300 );
	const spheres = [], curves = [];
	for ( let i = 0; i < N; i ++ ) {

		let x, y, z;
		do {

			x = rng() * 2 - 1; y = rng() * 2 - 1; z = rng() * 2 - 1;

		} while ( x * x + y * y + z * z > 1 );

		const big = rng() < 0.04;
		const r = big ? 3 + rng() * 2.5 : 0.6 + 1.3 * Math.pow( rng(), 2 );
		const metal = rng() < 0.35 ? 1 : 0;
		const glow = rng() < 0.15 ? 1.5 + rng() * 2 : 0;
		spheres.push( [ x * 40, y * 40, z * 40, r, ...hue( rng(), 0.8, metal ? 0.6 : 0.5 ), 0.08 + 0.8 * rng(), metal, glow, rng() < 0.3 ? 1 : 0 ] );

	}

	for ( let i = 0; i < C; i ++ ) {

		const a = spheres[ Math.floor( rng() * N ) ], b = spheres[ Math.floor( rng() * N ) ];
		const mid = [ 0, 1, 2 ].map( ( k ) => ( a[ k ] + b[ k ] ) * 0.25 ); // pulled towards the centre: a crude bundle
		const j = () => ( rng() - 0.5 ) * 14;
		curves.push( [ a.slice( 0, 3 ), [ mid[ 0 ] + j(), mid[ 1 ] + j(), mid[ 2 ] + j() ], [ mid[ 0 ] + j(), mid[ 1 ] + j(), mid[ 2 ] + j() ], b.slice( 0, 3 ),
			hue( rng(), 0.7, 0.55 ), 0.12 + 0.25 * rng() ] );

	}

	return { spheres, curves, camera: [ 0, 18, 125 ], target: [ 0, 0, 0 ], lightDir: [ - 0.5, 0.6, 0.62 ] };

}

function sceneChart() {

	const spheres = [], curves = [], meshes = [];
	const cols = 7, R = 3, dx = 7.4;
	const blocks = [ { metal: 1, y: 13, color: [ 1.0, 0.77, 0.34 ] }, { metal: 0, y: - 4, color: [ 0.8, 0.05, 0.05 ] } ];
	for ( const blk of blocks ) {

		for ( let i = 0; i < cols; i ++ ) {

			const rough = 0.05 + 0.95 * i / ( cols - 1 );
			const x = ( i - ( cols - 1 ) / 2 ) * dx;
			meshes.push( { type: 'sphere', pos: [ x, blk.y + 3.6, 0 ], r: R, color: blk.color, rough, metal: blk.metal } );
			spheres.push( [ x, blk.y - 3.6, 0, R, ...blk.color, rough, blk.metal, 0, 0 ] );

		}

	}

	// one more impostor pair with an emissive core and clearcoat (physical=1)
	spheres.push( [ - 31, 4.5, 0, R, 0.1, 0.5, 1.0, 0.4, 0, 2.5, 0 ], [ 31, 4.5, 0, R, 0.05, 0.05, 0.06, 0.6, 0, 0, 1 ] );
	// real tube above, strand tube below (same curve shape, radius, material)
	const tube = ( y ) => [ [ - 26, y, 2 ], [ - 9, y + 5, 9 ], [ 9, y - 5, - 5 ], [ 26, y, 2 ] ];
	meshes.push( { type: 'tube', ctrl: tube( - 13 ), r: 0.8, color: [ 0.1, 0.7, 0.9 ], rough: 0.35, metal: 0 } );
	curves.push( [ ...tube( - 16.5 ), [ 0.1, 0.7, 0.9 ], 0.8 ] );
	return { spheres, curves, meshes, ground: - 20.5, boundsScale: 1.6, camera: [ 0, 2, 92 ], target: [ 0, - 1, 0 ], lightDir: [ - 0.45, 0.75, 0.6 ] };

}

function sceneShadow() {

	const spheres = [], curves = [];
	const big = [ 0, 0, 14, 9, 0.9, 0.55, 0.15, 0.3, 1, 0, 0 ]; // the large caster
	if ( num( 'bigfirst', 1 ) ) spheres.push( big );
	for ( let iy = 0; iy < 21 * num( 'small', 1 ); iy ++ ) {

		for ( let ix = 0; ix < 25; ix ++ ) {

			const x = ( ix - 12 ) * 2.6 + ( rng() - 0.5 ) * 0.8, y = ( iy - 10 ) * 2.6 + ( rng() - 0.5 ) * 0.8;
			spheres.push( [ x, y, - 4 + ( rng() - 0.5 ) * 1.5, 0.75 + rng() * 0.3, ...hue( 0.5 + 0.2 * rng(), 0.6, 0.6 ), 0.5, 0, 0, 0 ] );

		}

	}

	if ( ! num( 'bigfirst', 1 ) ) spheres.push( big );
	for ( let i = 0; i < 40 * num( 'tubes', 1 ); i ++ ) {

		const y = - 26 + i * 1.35, z = 1 + ( rng() - 0.5 ) * 3;
		curves.push( [ [ - 34, y, z ], [ - 12, y + ( rng() - 0.5 ) * 10, z + 2 ], [ 12, y + ( rng() - 0.5 ) * 10, z - 2 ], [ 34, y + ( rng() - 0.5 ) * 6, z ],
			hue( 0.05 + 0.1 * rng(), 0.7, 0.6 ), 0.22 + 0.2 * rng() ] );

	}

	return { spheres, curves, wall: - 9, camera: [ 44, 14, 78 ], target: [ 2, - 3, 0 ], lightDir: [ - 0.45, 0.5, 1 ] };

}

function sceneStrands() {

	const spheres = [ [ 0, 0, 0, 2.5, 0.9, 0.9, 0.9, 0.3, 0, 0, 0 ] ], curves = [];
	// 5 fans, world radius halves from fan to fan: on-screen widths from ~8 px to ~0.3 px
	for ( let f = 0; f < 5; f ++ ) {

		const r = 0.42 / Math.pow( 2.3, f );
		for ( let i = 0; i < 26; i ++ ) {

			const x0 = - 44 + f * 22, a = ( i / 25 - 0.5 ) * 1.2;
			const top = [ x0 + Math.sin( a ) * 16, 22 * Math.cos( a ), ( rng() - 0.5 ) * 6 ];
			curves.push( [ [ x0, - 22, 0 ], [ x0 + ( rng() - 0.5 ) * 6, - 6, 8 ], [ top[ 0 ] + ( rng() - 0.5 ) * 6, 8, - 8 ], top, hue( 0.55 + 0.1 * f, 0.65, 0.6 ), r ] );

		}

	}

	return { spheres, curves, camera: [ 0, 0, 105 ], target: [ 0, 0, 0 ], lightDir: [ - 0.5, 0.6, 0.62 ] };

}

const desc = ( { field: sceneField, chart: sceneChart, shadow: sceneShadow, strands: sceneStrands }[ sceneName ] || sceneField )();

// ── renderer ───────────────────────────────────────────────────────────────────────────────────

async function main() {

	// MSAA is requested on the scene pass (pass option `samples`), not on the canvas
	// WebGPU allows 32 bytes per sample over all colour attachments by default (rgba8unorm and rgba16float
	// both count 8, rg16float 4, rg8unorm 2). bytes=64 asks the adapter for more (needed for 5 RGBA targets).
	const requiredLimits = {};
	const adapter = await navigator.gpu.requestAdapter();
	stats.maxColorAttachmentBytesPerSample = adapter.limits.maxColorAttachmentBytesPerSample;
	if ( num( 'bytes', 32 ) > 32 ) requiredLimits.maxColorAttachmentBytesPerSample = Math.min( num( 'bytes', 32 ), adapter.limits.maxColorAttachmentBytesPerSample );
	const renderer = new THREE.WebGPURenderer( { antialias: false, requiredLimits } );
	await renderer.init();
	if ( ! renderer.backend.isWebGPUBackend ) throw new Error( 'no WebGPU backend' );
	renderer.setPixelRatio( 1 );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.toneMapping = THREE.NeutralToneMapping;
	renderer.toneMappingExposure = num( 'exposure', 1 );
	document.body.appendChild( renderer.domElement );

	const shadowTypes = { basic: THREE.BasicShadowMap, pcf: THREE.PCFShadowMap, pcfsoft: THREE.PCFSoftShadowMap, vsm: THREE.VSMShadowMap };
	renderer.shadowMap.enabled = shadowMode !== 'off';
	renderer.shadowMap.type = shadowTypes[ shadowMode ] ?? THREE.PCFShadowMap;

	const scene = new THREE.Scene();
	scene.background = new THREE.Color( 0x020308 );
	const camera = new THREE.PerspectiveCamera( 40, window.innerWidth / window.innerHeight, 0.5, 2000 );
	const target = new THREE.Vector3( ...desc.target );
	const offset = new THREE.Vector3( ...desc.camera ).sub( target ).multiplyScalar( 1 / num( 'zoom', 1 ) );
	offset.applyAxisAngle( new THREE.Vector3( 1, 0, 0 ), - num( 'pitch', 0 ) ).applyAxisAngle( new THREE.Vector3( 0, 1, 0 ), num( 'yaw', 0 ) );
	camera.position.copy( target ).add( offset );
	camera.lookAt( target );

	// ── environment (Q5) ──
	if ( num( 'env', 1 ) === 1 ) {

		const t0 = performance.now();
		const env = createEnvironment( renderer );
		scene.environment = env.texture;
		scene.environmentIntensity = num( 'envint', 1 );
		scene.environmentRotation.y = num( 'envrot', 0 );
		for ( let i = 0; i < num( 'regen', 0 ); i ++ ) env.update();
		stats.pmremSubmitMs = env.timings.map( ( t ) => + t.toFixed( 1 ) );
		const face = Array.isArray( env.texture.image ) ? env.texture.image[ 0 ] : env.texture.image;
		stats.pmremSize = [ face.width, env.texture.mipmaps.length ]; // cube face size, mip levels (256² … 8²)
		// time until the GPU has really finished the PMREM work
		await renderer.backend.device.queue.onSubmittedWorkDone();
		stats.pmremTotalMs = + ( performance.now() - t0 ).toFixed( 1 );
		window.__env = env;

	}

	// ── storage buffers ──
	const S = desc.spheres, C = desc.curves;
	const buffers = {
		nodePos: instancedArray( S.length, 'vec4' ).setName( 'nodePos' ),
		nodeColor: instancedArray( S.length, 'vec4' ).setName( 'nodeColor' ),
		nodeMat: instancedArray( S.length, 'vec4' ).setName( 'nodeMat' ),
		nodeDelta: instancedArray( S.length, 'vec4' ).setName( 'nodeDelta' ),
		curveCtrl: instancedArray( Math.max( 1, C.length ) * 4, 'vec4' ).setName( 'curveCtrl' ),
		curveColor: instancedArray( Math.max( 1, C.length ), 'vec4' ).setName( 'curveColor' ),
	};
	S.forEach( ( s, i ) => {

		buffers.nodePos.value.array.set( s.slice( 0, 4 ), 4 * i );
		buffers.nodeColor.value.array.set( [ s[ 4 ], s[ 5 ], s[ 6 ], 1 ], 4 * i );
		buffers.nodeMat.value.array.set( s.slice( 7, 11 ), 4 * i );
		// every 5th sphere "moved" 1.5 units along +x since the last frame (motion-vector test)
		buffers.nodeDelta.value.array.set( i % 5 === 0 ? [ 1.5, 0, 0, 0 ] : [ 0, 0, 0, 0 ], 4 * i );

	} );
	C.forEach( ( c, i ) => {

		for ( let k = 0; k < 4; k ++ ) buffers.curveCtrl.value.array.set( [ ...c[ k ], 1 ], 16 * i + 4 * k );
		buffers.curveColor.value.array.set( [ ...c[ 4 ], c[ 5 ] ], 4 * i );

	} );

	// ── impostor meshes ──
	const sphereMaterial = createSphereMaterial( buffers, { physical, anisotropy, msaa: msaa > 1, velocity: useMRT, useVertexNode: num( 'vnode', 0 ) === 1, overrides: num( 'override', 1 ) === 1 } );
	const spheres = new THREE.Mesh( quadGeometry(), sphereMaterial );
	spheres.count = S.length;
	spheres.frustumCulled = false; // also disables culling against the shadow camera
	spheres.castShadow = spheres.receiveShadow = true;
	scene.add( spheres );

	const strandUniforms = { width: uniform( num( 'width', 1 ) ), minPx: uniform( num( 'minpx', 1 ) ), maxPx: uniform( 64 ), opacity: uniform( num( 'opacity', 0.5 ) ) };
	if ( C.length > 0 ) {

		const blend = num( 'blend', 0 ) === 1;
		const strandMaterial = createStrandMaterial( buffers, { mode: strandMode, msaa: msaa > 1, velocity: useMRT, blend, shadowDither: num( 'sdither', 0 ) === 1, gbuffer: useMRT ? GBUFFER : [], uniforms: strandUniforms } );
		const geometry = ribbonGeometry( 48 );
		geometry.instanceCount = C.length;
		const strands = new THREE.Mesh( geometry, strandMaterial );
		strands.frustumCulled = false;
		strands.castShadow = ! blend;
		strands.receiveShadow = true;
		scene.add( strands );

	}

	// ── real geometry for A/B comparison ──
	for ( const m of desc.meshes || [] ) {

		const material = physical && anisotropy > 0
			? new THREE.MeshPhysicalNodeMaterial( { color: new THREE.Color( ...m.color ), roughness: m.rough, metalness: m.metal, anisotropy } )
			: new THREE.MeshStandardNodeMaterial( { color: new THREE.Color( ...m.color ), roughness: m.rough, metalness: m.metal } );
		let geometry;
		if ( m.type === 'sphere' ) geometry = new THREE.SphereGeometry( m.r, 96, 48 );
		else geometry = new THREE.TubeGeometry( new THREE.CubicBezierCurve3( ...m.ctrl.map( ( p ) => new THREE.Vector3( ...p ) ) ), 96, m.r, 24 );
		const mesh = new THREE.Mesh( geometry, material );
		if ( m.pos ) mesh.position.set( ...m.pos );
		mesh.castShadow = mesh.receiveShadow = true;
		scene.add( mesh );

	}

	if ( desc.ground !== undefined || desc.wall !== undefined ) {

		const plane = new THREE.Mesh( new THREE.PlaneGeometry( 400, 400 ), new THREE.MeshStandardNodeMaterial( { color: 0x8a8f9a, roughness: 0.85, metalness: 0 } ) );
		if ( desc.ground !== undefined ) {

			plane.rotation.x = - Math.PI / 2;
			plane.position.y = desc.ground;

		} else plane.position.z = desc.wall;

		plane.receiveShadow = true;
		scene.add( plane );

	}

	// ── key light + shadow (Q3) ──
	const bounds = new THREE.Sphere( new THREE.Vector3(), 0 );
	for ( const s of S ) bounds.center.add( new THREE.Vector3( s[ 0 ], s[ 1 ], s[ 2 ] ) );
	bounds.center.multiplyScalar( 1 / S.length );
	for ( const s of S ) bounds.radius = Math.max( bounds.radius, bounds.center.distanceTo( new THREE.Vector3( s[ 0 ], s[ 1 ], s[ 2 ] ) ) + s[ 3 ] );
	for ( const c of C ) for ( let k = 0; k < 4; k ++ ) bounds.radius = Math.max( bounds.radius, bounds.center.distanceTo( new THREE.Vector3( ...c[ k ] ) ) );

	const light = new THREE.DirectionalLight( 0xfff0dd, num( 'light', 3 ) );
	light.castShadow = shadowMode !== 'off';
	const mapSize = num( 'map', 1024 );
	light.shadow.mapSize.set( mapSize, mapSize );
	light.shadow.bias = num( 'bias', - 0.0006 );
	light.shadow.normalBias = num( 'nbias', 0.04 );
	light.shadow.radius = num( 'radius', 2 ); // PCF: Vogel disc radius in texels; VSM: blur radius in texels
	light.shadow.blurSamples = num( 'blur', 8 ); // VSM only
	scene.add( light, light.target );
	if ( desc.boundsScale ) bounds.radius *= desc.boundsScale; // room for shadows on the ground plane
	const lightDir = params.has( 'ldir' ) ? P( 'ldir' ).split( ',' ).map( Number ) : desc.lightDir;
	fitDirectionalShadow( light, bounds, new THREE.Vector3( ...lightDir ) );

	// point=1: a close point light; its light vector needs the true surface position (positionView override)
	if ( num( 'point', 0 ) === 1 ) {

		const point = new THREE.PointLight( 0x66ffcc, 900, 0, 2 );
		point.position.set( 0, 4.5, 7 );
		scene.add( point );

	}

	// re-fit at runtime (the graph's bounding sphere changes while the layout moves): uniforms only, no rebuild
	window.__fitShadow = ( scale = 1 ) => fitDirectionalShadow( light, new THREE.Sphere( bounds.center, bounds.radius * scale ), new THREE.Vector3( ...lightDir ) );

	const shadowRate = num( 'shadowrate', 1 ); // render the shadow map every n-th frame
	if ( shadowRate !== 1 ) light.shadow.autoUpdate = false;
	const updateMatrices = light.shadow.updateMatrices.bind( light.shadow );
	light.shadow.updateMatrices = ( l ) => {

		stats.shadowRenders ++; // called once per shadow map render
		updateMatrices( l );

	};

	// ── scene pass + G-buffer (Q4) ──
	// cam=light renders the main pass through the shadow camera: shows what the light "sees" (ortho path)
	const viewCamera = P( 'cam', 'main' ) === 'light' ? light.shadow.camera : camera;
	const scenePass = pass( scene, viewCamera, { samples: msaa } );
	if ( useMRT ) {

		const gbuffer = mrt( {
			output,
			normal: vec4( packNormalToRGB( normalView ), 1 ), // view-space normal, n * 0.5 + 0.5
			emissive: vec4( emissive, 1 ),
			metalrough: vec4( metalness, roughness, 0, 1 ),
			velocity: vec4( velocity, 0, 1 ), // three's own node for regular meshes; the impostors override it through material.mrtNode
		} );
		// Attachments other than 'output' are written without blending by default, also by alpha-blended
		// materials. With an alpha blend mode a blended material can leave them untouched by writing
		// alpha 0 (opaque ones write alpha 1). That needs an alpha channel in the shader output, which
		// three derives from the texture format: RG attachments + alpha blend mode fail pipeline validation.
		const blendedStrands = num( 'blend', 0 ) === 1;
		if ( blendedStrands ) for ( const name of GBUFFER ) gbuffer.setBlendMode( name, new THREE.BlendMode( THREE.NormalBlending ) );
		scenePass.setMRT( gbuffer );
		// attachments are clones of the colour texture (RGBA16F, 8 bytes/px) unless changed here
		if ( num( 'formats', 1 ) === 1 ) {

			scenePass.getTexture( 'normal' ).type = THREE.UnsignedByteType; // rgba8unorm, 4 bytes
			const mr = scenePass.getTexture( 'metalrough' );
			mr.type = THREE.UnsignedByteType; // rgba8unorm, 4 bytes
			if ( ! blendedStrands ) {

				mr.format = THREE.RGFormat; // rg8unorm, 2 bytes
				scenePass.getTexture( 'velocity' ).format = THREE.RGFormat; // rg16float, 4 bytes

			}

		}

		window.__gbuffer = () => scenePass.renderTarget.textures.map( ( t ) => `${t.name}: format ${t.format} type ${t.type}` );

	}

	// ao=<0..1>: constant ambient occlusion through the pass context, the hook a screen-space AO pass
	// uses to feed three's lighting (it attenuates the indirect / environment part only)
	if ( params.has( 'ao' ) ) scenePass.contextNode = context( { getAO: () => float( num( 'ao', 1 ) ) } );

	const sample = ( name, uv ) => ( uv ? scenePass.getTextureNode( name ).sample( uv ) : scenePass.getTextureNode( name ) );
	const dist = camera.position.distanceTo( target );
	const views = {
		beauty: ( uv ) => renderOutput( sample( 'output', uv ) ),
		normal: ( uv ) => vec4( sample( 'normal', uv ).rgb, 1 ),
		emissive: ( uv ) => vec4( sample( 'emissive', uv ).rgb, 1 ),
		metalrough: ( uv ) => vec4( sample( 'metalrough', uv ).rg, 0, 1 ),
		velocity: ( uv ) => vec4( sample( 'velocity', uv ).rg.mul( 25 ).add( 0.5 ), 0.5, 1 ), // 0.02 NDC → full scale
		depth: ( uv ) => {

			const z = perspectiveDepthToViewZ( sample( 'depth', uv ).r, float( camera.near ), float( camera.far ) );
			return vec4( vec3( z.negate().sub( dist * 0.5 ).div( dist ).oneMinus().clamp() ), 1 );

		},
	};

	let outputNode;
	if ( view === 'grid' && useMRT ) {

		// 3 × 2 montage: beauty | normal | depth / emissive | metalrough | velocity
		const g = screenUV.mul( vec2( 3, 2 ) );
		const cell = floor( g ), uv = fract( g );
		const order = [ 'beauty', 'normal', 'depth', 'emissive', 'metalrough', 'velocity' ];
		outputNode = order.map( ( name ) => views[ name ]( uv ) ).reduceRight( ( rest, node, i ) => select( cell.x.add( cell.y.mul( 3 ) ).lessThan( i + 0.5 ), node, rest ) );

	} else outputNode = ( useMRT ? views[ view ] || views.beauty : views.beauty )();

	const pipeline = new THREE.RenderPipeline( renderer );
	pipeline.outputColorTransform = false; // tone mapping + sRGB are applied by renderOutput() above
	pipeline.outputNode = outputNode;

	// ── frame loop ──
	const spin = num( 'spin', 0 );
	const t0 = performance.now();
	const frame = () => {

		if ( spin ) {

			offset.applyAxisAngle( new THREE.Vector3( 0, 1, 0 ), spin * 0.01 );
			camera.position.copy( target ).add( offset );
			camera.lookAt( target );

		}

		if ( shadowRate > 1 && stats.frames % shadowRate === 0 ) light.shadow.needsUpdate = true;
		updateVelocityUniforms( camera );
		pipeline.render();
		commitVelocityUniforms();
		stats.frames ++;
		if ( stats.frames === 1 ) stats.firstFrameMs = + ( performance.now() - t0 ).toFixed( 0 );
		if ( stats.frames === 3 ) window.__ready = true;

	};

	renderer.setAnimationLoop( frame );
	// for the motion-vector test with a moving camera: stop the loop, move the camera, call __frame() once
	window.__frame = frame;
	window.__viewProjection = () => velocityUniforms.viewProjection.value.toArray();

	window.addEventListener( 'resize', () => {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		renderer.setSize( window.innerWidth, window.innerHeight );

	} );

	Object.assign( window, { __renderer: renderer, __scene: scene, __camera: camera, __light: light, __scenePass: scenePass, __desc: desc } );
	stats.config = { scene: sceneName, spheres: S.length, curves: C.length, msaa, mrt: useMRT, shadow: shadowMode, strand: strandMode, physical };
	stats.bounds = [ ...bounds.center.toArray().map( ( v ) => + v.toFixed( 2 ) ), + bounds.radius.toFixed( 2 ) ];

	/**
	 * Motion-vector check: reads the velocity attachment at the centres of unoccluded spheres and compares
	 * with NDC( now ) − NDC( previous ) computed on the CPU. `previousViewProjection` (16 numbers) is the
	 * view-projection matrix of the frame before the one in the attachment; omit it for a static camera.
	 */
	window.__checkVelocity = async ( previousViewProjection ) => {

		if ( ! useMRT ) return null;
		const rt = scenePass.renderTarget;
		const index = rt.textures.findIndex( ( t ) => t.name === 'velocity' );
		const out = [];
		const center = new THREE.Vector3(), prev = new THREE.Vector3();
		const ray = new THREE.Ray(), sph = new THREE.Sphere();
		for ( let i = 0; i < S.length && out.length < 12; i ++ ) {

			center.set( S[ i ][ 0 ], S[ i ][ 1 ], S[ i ][ 2 ] );
			ray.origin.copy( camera.position );
			ray.direction.copy( center ).sub( camera.position ).normalize();
			const d = center.distanceTo( camera.position );
			let occluded = false;
			for ( let j = 0; j < S.length && ! occluded; j ++ ) {

				if ( j === i ) continue;
				sph.center.set( S[ j ][ 0 ], S[ j ][ 1 ], S[ j ][ 2 ] ); sph.radius = S[ j ][ 3 ] + 0.5;
				occluded = ray.intersectsSphere( sph ) && sph.center.distanceTo( camera.position ) < d;

			}

			if ( occluded ) continue;
			// expected: front point of the sphere now, and the same point one frame ago
			const front = center.clone().addScaledVector( ray.direction, - S[ i ][ 3 ] );
			const moved = i % 5 === 0 ? 1.5 : 0;
			prev.copy( front ); prev.x -= moved;
			const a = front.clone().project( camera );
			const b = previousViewProjection ? prev.clone().applyMatrix4( new THREE.Matrix4().fromArray( previousViewProjection ) ) : prev.clone().project( camera );
			const px = Math.floor( ( a.x * 0.5 + 0.5 ) * rt.width ), py = Math.floor( ( 0.5 - a.y * 0.5 ) * rt.height );
			if ( px < 2 || py < 2 || px > rt.width - 3 || py > rt.height - 3 ) continue;
			const read = async ( y ) => {

				const data = await renderer.readRenderTargetPixelsAsync( rt, px, y, 1, 1, index );
				const dec = ( v ) => ( data instanceof Uint16Array ? THREE.DataUtils.fromHalfFloat( v ) : v );
				return [ + dec( data[ 0 ] ).toFixed( 5 ), + dec( data[ 1 ] ).toFixed( 5 ) ];

			};

			out.push( { node: i, moved, expected: [ + ( a.x - b.x ).toFixed( 5 ), + ( a.y - b.y ).toFixed( 5 ) ], read: await read( py ) } ); // readback origin is the top-left corner

		}

		return out;

	};

}

/**
 * Fits the orthographic shadow camera of a directional light tightly around a bounding sphere.
 * The light direction stays fixed; the box centre is snapped to shadow-map texels so that a moving
 * sphere (live layout) does not make the shadow edges crawl.
 */
export function fitDirectionalShadow( light, sphere, direction ) {

	const dir = direction.clone().normalize(); // from the scene towards the light
	const r = sphere.radius * 1.02;
	const texel = 2 * r / light.shadow.mapSize.x;
	// light-space basis
	const up = Math.abs( dir.y ) > 0.99 ? new THREE.Vector3( 1, 0, 0 ) : new THREE.Vector3( 0, 1, 0 );
	const x = new THREE.Vector3().crossVectors( up, dir ).normalize();
	const y = new THREE.Vector3().crossVectors( dir, x );
	const c = sphere.center;
	const snapped = new THREE.Vector3()
		.addScaledVector( x, Math.round( c.dot( x ) / texel ) * texel )
		.addScaledVector( y, Math.round( c.dot( y ) / texel ) * texel )
		.addScaledVector( dir, c.dot( dir ) );
	light.target.position.copy( snapped );
	light.position.copy( snapped ).addScaledVector( dir, 2 * r );
	light.target.updateMatrixWorld();
	const cam = light.shadow.camera;
	cam.left = - r; cam.right = r; cam.top = r; cam.bottom = - r;
	cam.near = r; cam.far = 3 * r; // depth range = the sphere's diameter: best depth precision
	cam.updateProjectionMatrix();

}

main().catch( ( e ) => {

	console.error( e );
	window.__error = e.message || String( e );

} );
