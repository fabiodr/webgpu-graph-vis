// Spike "post": screen-space lighting effects and the cinematic post-processing chain.
// Page: /dev/spikes/post.html — reference code for recipes/post.md, not production.
//
// Scene: SpikeNodes (NodeRenderer + G-buffer normal) and TubeEdges (opaque cylinder impostors) on a
// small dataset with a static CPU layout (post-scene.js). Pipeline: post-chain.js.
//
// Query parameters (all optional):
//   ds=imsv/lesmis   dataset          spread=0.55  layout radius factor (smaller = denser cluster)
//   size=2.2         node size        tube=0.3     tube radius (× node base size)   grey=0|1 grey tubes
//   emissive=1       HDR multiplier of the node colour      sel=<index>  select a node (halo pass)
//   zoom=1 yaw=0 pitch=0 fov=45       camera (fitted to the bounding sphere), autorotate=0|1
//   tier=low|medium|high|ultra        preset of structural options (post-chain.js TIERS)
//   -- structural chain options (post-chain.js DEFAULT_OPTIONS), e.g.
//   msaa=4|0  ao=gtao|ssgi|none  aoRes=0.5  aoSamples=16  aoDenoise=rtt|inline|bilateral|box|none  aoNormals=mrt|depth
//   aoApply=composite|context  sss=0|1  sssRes=1  dof=0|1  bloom=1|0  bloomRes=0.5  dirt=0|1  flare=0|1
//   streak=0|1  ca=0|1  sharpen=0|1  grain=custom|film|none  tone=neutral|agx|aces|…  fog=1|0
//   autoExposure=0|1|gpu   dof=0|1|graph  dofTaps=48  dofJitter=1|0  motionBlur=0|1  spin=<deg/frame>
//   view=final|ao|normal|depth|gi|sss|lit|bloom|dof
//   -- live uniforms: u.<name>=<number> for any key of createUniforms(), e.g. u.aoStrength=1.5
//      aoRadiusK=3 (AO radius in node radii), focusK=1 (focus distance × camera distance),
//      focalK=0.25 (focal range × scene radius), fogK=0 (fog density × 1/scene radius)
//
// Globals for the harness: __app, __chain, __U, __rebuild( options ), __mem(), __ready.
import * as THREE from 'three/webgpu';
import { createApp } from '../../src/app/App.js';
import { loadPrepared } from '../../src/data/load.js';
import { SpikeNodes, TubeEdges, cpuLayout } from './post-scene.js';
import { buildChain, createUniforms, DEFAULT_OPTIONS, TIERS } from './post-chain.js';

const params = new URLSearchParams( location.search );
const P = ( k, d ) => ( params.has( k ) ? params.get( k ) : d );
const num = ( k, d ) => + P( k, d );

try {

	const app = await createApp( document.getElementById( 'app' ), { maxPixelRatio: num( 'maxdpr', 2 ), readback: false } );
	window.__app = app;
	const renderer = app.renderer;
	app.scene.background = new THREE.Color( P( 'bg', '#020409' ) );

	// structural options from the query string (typed like the defaults)
	const options = {};
	for ( const [ k, d ] of Object.entries( DEFAULT_OPTIONS ) ) {

		if ( ! params.has( k ) ) continue;
		const v = params.get( k );
		options[ k ] = typeof d === 'number' ? + v : typeof d === 'boolean' ? ( /^(0|1|true|false)$/.test( v ) ? v === '1' || v === 'true' : v ) : v;

	}

	// tier=low|medium|high|ultra: start from a tier preset (explicit parameters win)
	if ( params.has( 'tier' ) ) for ( const [ k, v ] of Object.entries( TIERS[ params.get( 'tier' ) ] ) ) if ( ! ( k in options ) ) options[ k ] = v;
	const msaa = options.msaa ?? DEFAULT_OPTIONS.msaa;

	// camera-attached key light (same view-space direction as the hand-lit shading); used by SSS only
	const keyLight = new THREE.DirectionalLight( 0xffffff, 1 );
	const KEY = new THREE.Vector3( - 0.45, 0.65, 0.62 ).normalize();
	app.scene.add( keyLight, keyLight.target );
	app.keyLight = keyLight;

	const tubes = app.use( new TubeEdges( { radius: num( 'tube', 0.3 ), grey: num( 'grey', 0 ) } ) );
	const nodes = app.use( new SpikeNodes( { size: num( 'size', 2.2 ), passSamples: msaa, emissive: num( 'emissive', 1 ), halo: true } ) );

	const graph = await loadPrepared( `/data/${P( 'ds', 'imsv/lesmis' )}.json` );
	app.setGraph( graph );
	const buffers = app.buffers;
	const pos = cpuLayout( graph, num( 'spread', 0.55 ) );
	buffers.setPositions( pos );
	app.positions = new Float32Array( pos );
	app.positionsVersion ++;

	// optional selection (drives the additive halo pass → transparent material in the MRT pass)
	const sel = num( 'sel', - 1 );
	if ( sel >= 0 ) {

		const t = new Float32Array( 4 * graph.N );
		t[ 4 * sel ] = 1;
		buffers.setStateTargets( t );
		buffers.nodeState.value.array.set( t );
		buffers.nodeState.value.needsUpdate = true;

	}

	// camera
	const sphere = app.boundingSphere();
	app.camera.fov = num( 'fov', 45 );
	app.camera.updateProjectionMatrix();
	const vfov = THREE.MathUtils.degToRad( app.camera.fov );
	const fov = Math.min( vfov, 2 * Math.atan( Math.tan( vfov / 2 ) * app.camera.aspect ) );
	const dist = ( sphere.radius / Math.sin( fov / 2 ) ) * 1.02 / num( 'zoom', 1 );
	const yaw = THREE.MathUtils.degToRad( num( 'yaw', 20 ) ), pitch = THREE.MathUtils.degToRad( num( 'pitch', 12 ) );
	app.camera.position.set( sphere.center.x + dist * Math.sin( yaw ) * Math.cos( pitch ), sphere.center.y + dist * Math.sin( pitch ), sphere.center.z + dist * Math.cos( yaw ) * Math.cos( pitch ) );
	app.controls.target.copy( sphere.center );
	app.controls.autoRotate = num( 'autorotate', 0 ) === 1;
	app.camera.lookAt( sphere.center );
	if ( params.has( 'near' ) ) app.camera.near = num( 'near', 0.5 );
	if ( params.has( 'far' ) ) app.camera.far = num( 'far', 20000 );
	app.camera.updateProjectionMatrix();

	// uniforms: scene-scale dependent defaults, then u.<name> overrides
	const U = createUniforms();
	const nodeR = nodes.baseSize * nodes.size;
	U.aoRadius.value = num( 'aoRadiusK', 3 ) * nodeR;
	U.aoThickness.value = num( 'aoThickK', 3 ) * nodeR;
	U.focus.value = num( 'focusK', 1 ) * dist;
	U.focalRange.value = num( 'focalK', 0.25 ) * sphere.radius;
	U.fogDensity.value = num( 'fogK', 0 ) / sphere.radius;
	U.fogStart.value = Math.max( 0, dist - sphere.radius );
	U.fogColor.value.copy( app.scene.background );
	for ( const [ k, v ] of params ) if ( k.startsWith( 'u.' ) && U[ k.slice( 2 ) ] ) U[ k.slice( 2 ) ].value = + v;
	window.__U = U;

	// chain
	let chain = null;
	app.pipeline.outputColorTransform = false;
	const rebuild = ( opts = {} ) => {

		const old = chain;
		Object.assign( options, opts );
		chain = buildChain( app, options, U );
		renderer.toneMappingExposure = 1;
		app.pipeline.outputNode = chain.output;
		app.pipeline.needsUpdate = true;
		old?.dispose();
		window.__chain = chain;
		// SSGI / SSS tuning that scales with the scene
		if ( chain.nodes.sssPass ) {

			chain.nodes.sssPass.maxDistance.value = num( 'sssDistK', 4 ) * nodeR;
			chain.nodes.sssPass.thickness.value = num( 'sssThickK', 1.5 ) * nodeR;
			chain.nodes.sssPass.quality.value = num( 'sssQuality', 0.5 );

		}

		if ( chain.options.ao === 'ssgi' ) {

			const g = chain.nodes.aoPass;
			g.radius.value = num( 'giRadius', 6 );
			g.giIntensity.value = num( 'giIntensity', 10 );
			g.aoIntensity.value = num( 'giAO', 1 );
			g.useScreenSpaceSampling.value = num( 'giScreen', 1 ) === 1;
			g.backfaceLighting.value = num( 'giBackface', 0 );

		}

		return chain;

	};

	rebuild();
	// App.init() created its own scene pass (MSAA HDR + depth); nothing references it any more
	window.__mem = () => ( { textures: renderer.info.memory.textures, texturesMB: + ( renderer.info.memory.texturesSize / 1048576 ).toFixed( 2 ), totalMB: + ( renderer.info.memory.total / 1048576 ).toFixed( 2 ) } );
	const before = window.__mem();
	app.scenePass.dispose();
	window.__appPassFreed = [ before, window.__mem() ];
	window.__rebuild = rebuild;
	window.__TIERS = TIERS;

	const hud = document.getElementById( 'hud' );
	const spin = num( 'spin', 0 );
	const q = new THREE.Quaternion();
	app.use( {
		order: 5,
		update( dt ) {

			// key light follows the camera (fixed direction in view space)
			app.camera.getWorldQuaternion( q );
			keyLight.target.position.copy( app.camera.position );
			keyLight.position.copy( KEY ).applyQuaternion( q ).multiplyScalar( 100 ).add( app.camera.position );
			// spin=<degrees per frame>: constant camera motion for the motion-blur test
			if ( spin ) {

				app.camera.position.sub( app.controls.target ).applyAxisAngle( new THREE.Vector3( 0, 1, 0 ), THREE.MathUtils.degToRad( spin ) ).add( app.controls.target );
				app.camera.lookAt( app.controls.target );

			}

			chain.update( dt );
			if ( app.frame % 15 === 0 && num( 'hud', 1 ) ) hud.textContent = `${graph.name} N=${graph.N} E=${graph.E}  ${app.fps.toFixed( 1 )} fps  ${JSON.stringify( window.__mem() )}\n${Object.entries( chain.options ).map( ( [ k, v ] ) => k + '=' + v ).join( ' ' )}`;

		},
	} );

	// ready after a few rendered frames
	await new Promise( ( resolve ) => {

		const f0 = app.frame;
		const check = () => ( app.frame - f0 >= num( 'frames', 3 ) ? resolve() : requestAnimationFrame( check ) );
		check();

	} );
	window.__ready = true;

} catch ( e ) {

	console.error( e );
	window.__error = e && e.stack ? e.stack : String( e );

}
