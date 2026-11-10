// Spike "aa": anti-aliasing and thin-strand transparency strategies, compared on one scene.
// NOT production code. Scene: aa-scene.js; strategies and runtime switching: aa-pipeline.js.
//
// Page: /dev/spikes/aa.html?s=<strategy>&...
//   s=none|s1|s2|s2d|s2m|s3|s3b|s3o|s4|s5|s5b|s5s|s5f|s6|s6s|s6o     (see STRATEGIES)
//   edges= samples= scale= post= dither=1 view=final|depth|velocity|raw   overrides of the strategy
//   (also: sharpness, velocityGain, depthThreshold, currentFrameWeight, maxVelocityLength,
//   subpixel=0|1 for TRAA.useSubpixelCorrection, oitWeight=flat, oitDirect, freshMRT)
//   w=640 h=400        size of the WebGPU canvas in CSS px (independent of the browser viewport)
//   ss=1               renderer pixel ratio (supersampling; grabs are box-filtered back to w x h)
//   yaw=20 pitch=12 dist=300   camera (degrees, world units)
//   orbit=0            degrees of yaw per frame        warm=24   frames rendered before __ready
//   move=0             node swing amplitude (world units), phase=0.25 rad per frame
//   vel=true|camera    motion vectors from the previous-position buffers, or camera only
//   mv=custom|builtin|zero   velocity MRT slot: the materials' own vectors, three's `velocity`
//                      node (wrong for storage-buffer geometry), or zero
//   tnoise=1|0         stochastic modes: new noise every frame (default: on for TRAA / TAAU)
//   opacity=0.4 minpx=1 width=0.1 pad=0.5   ribbon parameters; minpx and pad are OUTPUT pixels,
//   pxscale=<ss>       pass pixels per output pixel (set it to `scale` for a reduced-resolution pass)
//   nodes=150 nedges=3000 segments=16   scene size
//   crop=x,y,w,h&zoom=4    show a nearest-neighbour zoom of that canvas region instead
//   grid=s1,s2,...     render the strategies one after the other (runtime switch) and show their
//                      crops side by side; gridmode=image|shimmer; cols=3; gain=8; show=error.
//                      frames=6 (shimmer), warm (history frames of the temporal strategies).
//                      A cell can carry overrides: s3:mv=builtin:edges=blend:vel=camera:pxscale=0.5
//                      A cell "ref@<cell>" (or ref=<cell> for the first one) renders that cell
//                      supersampled (refss=3) at the final pose, averaged over refacc=1 frames of
//                      fresh noise; the cells after it report their mean abs error against it
//
// Harness (all deterministic, no animation loop):
//   await __step( n )              render n frames, advancing orbit / node motion
//   __grab()                       { width, height, data } of the canvas (sRGB8, box filtered)
//   await __measure( { frames } )  temporal metrics of the current strategy (see measure())
//   await __use( name, over )      switch strategy at runtime
//   await __switchTest( cycles, { names, over, disposeMaterials } )
//                                  cycle through the strategies, report GPU resource counts
//   __gpu                          live GPU object counters (createTexture / destroy hooks)
import * as THREE from 'three/webgpu';
import { createTestScene } from './aa-scene.js';
import { STRATEGIES, StrategySwitcher } from './aa-pipeline.js';

const params = new URLSearchParams( location.search );
const P = ( k, d ) => ( params.has( k ) ? params.get( k ) : d );
const num = ( k, d ) => ( params.has( k ) ? + params.get( k ) : d );

// ── GPU object accounting (installed before the renderer creates its device) ──────────────────
const BYTES = { rgba16float: 8, rgba8unorm: 4, 'rgba8unorm-srgb': 4, bgra8unorm: 4, r8unorm: 1, rg16float: 4, depth24plus: 4, depth32float: 4, 'depth24plus-stencil8': 4, rgba32float: 16, r16float: 2, rg8unorm: 2 };
const gpu = { texturesCreated: 0, texturesDestroyed: 0, liveTextures: 0, liveTextureBytes: 0, renderPipelines: 0, buffersCreated: 0, buffersDestroyed: 0 };
window.__gpu = gpu;
if ( typeof GPUDevice !== 'undefined' ) {

	const createTexture = GPUDevice.prototype.createTexture;
	GPUDevice.prototype.createTexture = function ( d ) {

		const t = createTexture.call( this, d );
		const size = Array.isArray( d.size ) ? d.size : [ d.size.width, d.size.height ?? 1, d.size.depthOrArrayLayers ?? 1 ];
		t.__bytes = size[ 0 ] * ( size[ 1 ] ?? 1 ) * ( size[ 2 ] ?? 1 ) * ( d.sampleCount ?? 1 ) * ( BYTES[ d.format ] ?? 4 );
		gpu.texturesCreated ++; gpu.liveTextures ++; gpu.liveTextureBytes += t.__bytes;
		return t;

	};

	const destroyTexture = GPUTexture.prototype.destroy;
	GPUTexture.prototype.destroy = function () {

		if ( this.__bytes !== undefined && ! this.__gone ) {

			this.__gone = true;
			gpu.texturesDestroyed ++; gpu.liveTextures --; gpu.liveTextureBytes -= this.__bytes;

		}

		return destroyTexture.call( this );

	};

	for ( const fn of [ 'createRenderPipeline', 'createRenderPipelineAsync' ] ) {

		const original = GPUDevice.prototype[ fn ];
		GPUDevice.prototype[ fn ] = function ( d ) {

			gpu.renderPipelines ++;
			return original.call( this, d );

		};

	}

	const createBuffer = GPUDevice.prototype.createBuffer;
	GPUDevice.prototype.createBuffer = function ( d ) {

		gpu.buffersCreated ++;
		return createBuffer.call( this, d );

	};

	const destroyBuffer = GPUBuffer.prototype.destroy;
	GPUBuffer.prototype.destroy = function () {

		if ( ! this.__gone ) {

			this.__gone = true; gpu.buffersDestroyed ++;

		}

		return destroyBuffer.call( this );

	};

}

async function main() {

	const W = num( 'w', 640 ), H = num( 'h', 400 ), SS = num( 'ss', 1 );
	const hud = document.getElementById( 'hud' );

	// antialias: false -> renderer.samples = 0; MSAA is requested per pass (pass option `samples`)
	const renderer = new THREE.WebGPURenderer( { antialias: false } );
	await renderer.init();
	if ( ! renderer.backend.isWebGPUBackend ) throw new Error( 'WebGPU backend not available' );
	renderer.setPixelRatio( SS );
	renderer.setSize( W, H );
	document.getElementById( 'gpu' ).appendChild( renderer.domElement );
	const device = renderer.backend.device;

	const test = createTestScene( { nodes: num( 'nodes', 150 ), edges: num( 'nedges', 3000 ), segments: num( 'segments', 16 ) } );
	const U = test.uniforms;
	U.opacity.value = num( 'opacity', 0.4 );
	U.minPx.value = num( 'minpx', 1 );
	U.halfWidth.value = num( 'width', 0.1 );
	U.edgePad.value = num( 'pad', 0.5 ) * num( 'pxscale', SS );
	U.pxScale.value = num( 'pxscale', SS );
	U.trueMotion.value = P( 'vel', 'true' ) === 'camera' ? 0 : 1;
	test.motion.amplitude = num( 'move', 0 );

	const camera = new THREE.PerspectiveCamera( 45, W / H, 20, 1200 );
	const state = { frame: 0, yaw: num( 'yaw', 20 ), pitch: num( 'pitch', 12 ), dist: num( 'dist', 300 ), orbit: num( 'orbit', 0 ), phaseStep: num( 'phase', 0.25 ) };
	const placeCamera = () => {

		const y = THREE.MathUtils.degToRad( state.yaw ), p = THREE.MathUtils.degToRad( state.pitch );
		camera.position.set( Math.sin( y ) * Math.cos( p ), Math.sin( p ), Math.cos( y ) * Math.cos( p ) ).multiplyScalar( state.dist );
		camera.lookAt( 0, 0, 0 );
		camera.updateMatrixWorld();

	};

	const switcher = new StrategySwitcher( renderer, { scene: test.scene, camera, test } );
	const overrides = () => {

		const o = {};
		for ( const k of [ 'edges', 'post', 'view', 'oitWeight', 'mv' ] ) if ( params.has( k ) ) o[ k ] = params.get( k );
		for ( const k of [ 'samples', 'scale', 'sharpness', 'velocityGain', 'depthThreshold', 'currentFrameWeight', 'maxVelocityLength' ] ) if ( params.has( k ) ) o[ k ] = + params.get( k );
		if ( params.has( 'dither' ) ) o.dither = params.get( 'dither' ) === '1';
		if ( params.has( 'subpixel' ) ) o.subpixel = params.get( 'subpixel' ) === '1';
		return o;

	};

	let temporalNoise = true;
	const use = ( name, over = overrides() ) => {

		const s = switcher.use( name, over );
		temporalNoise = params.has( 'tnoise' ) ? params.get( 'tnoise' ) === '1' : !! s.def.temporal;
		return s;

	};

	// ── frame stepping ─────────────────────────────────────────────────────────────────────────
	let wasMoving = false;
	function renderFrame( advance = true ) {

		if ( advance ) {

			state.frame ++;
			state.yaw += state.orbit;
			if ( test.motion.amplitude > 0 ) test.motion.phase += state.phaseStep;

		}

		const moving = test.motion.amplitude > 0;
		if ( moving || wasMoving ) test.writePositions(); // one extra write after a stop: previous = current
		wasMoving = moving;
		placeCamera();
		U.frame.value = temporalNoise ? state.frame % 64 : 0;
		test.updateMotionUniforms( camera ); // unjittered: TRAA / TAAU jitter only inside render()
		switcher.render();
		test.commitMotionUniforms();

	}

	async function step( n = 1 ) {

		for ( let i = 0; i < n; i ++ ) {

			renderFrame();
			await device.queue.onSubmittedWorkDone();

		}

	}

	// ── pixel access: the WebGPU canvas is copied to a 2D canvas right after render() ────────────
	const grabCanvas = document.createElement( 'canvas' );
	const grabCtx = grabCanvas.getContext( '2d', { willReadFrequently: true } );
	const toLinear = new Float32Array( 256 ).map( ( _, i ) => ( i <= 10 ? i / 3294.6 : Math.pow( ( i / 255 + 0.055 ) / 1.055, 2.4 ) ) );
	const toSRGB = ( v ) => 255 * ( v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow( v, 1 / 2.4 ) - 0.055 );

	/** Returns { width, height, lin: Float32Array rgb (linear) } at CSS resolution (box filter over ss²). */
	function grabLinear() {

		const c = renderer.domElement;
		grabCanvas.width = c.width; grabCanvas.height = c.height;
		grabCtx.drawImage( c, 0, 0 );
		const src = grabCtx.getImageData( 0, 0, c.width, c.height ).data;
		const k = Math.round( c.width / W );
		const lin = new Float32Array( W * H * 3 );
		for ( let y = 0; y < H; y ++ ) for ( let x = 0; x < W; x ++ ) {

			let r = 0, g = 0, b = 0;
			for ( let j = 0; j < k; j ++ ) for ( let i = 0; i < k; i ++ ) {

				const o = 4 * ( ( y * k + j ) * c.width + x * k + i );
				r += toLinear[ src[ o ] ]; g += toLinear[ src[ o + 1 ] ]; b += toLinear[ src[ o + 2 ] ];

			}

			const q = 3 * ( y * W + x ), n = 1 / ( k * k );
			lin[ q ] = r * n; lin[ q + 1 ] = g * n; lin[ q + 2 ] = b * n;

		}

		return { width: W, height: H, lin };

	}

	const toImage = ( g ) => {

		const data = new Uint8ClampedArray( W * H * 4 );
		for ( let i = 0, q = 0; i < W * H; i ++, q += 3 ) {

			data[ 4 * i ] = toSRGB( g.lin[ q ] ); data[ 4 * i + 1 ] = toSRGB( g.lin[ q + 1 ] ); data[ 4 * i + 2 ] = toSRGB( g.lin[ q + 2 ] ); data[ 4 * i + 3 ] = 255;

		}

		return new ImageData( data, W, H );

	};

	/** sRGB-encoded luma (0..255) per pixel. */
	const luma = ( g ) => {

		const y = new Float32Array( W * H );
		for ( let i = 0, q = 0; i < W * H; i ++, q += 3 ) y[ i ] = 0.2126 * toSRGB( g.lin[ q ] ) + 0.7152 * toSRGB( g.lin[ q + 1 ] ) + 0.0722 * toSRGB( g.lin[ q + 2 ] );
		return y;

	};

	/** Renders n frames without advancing the camera and averages them (linear light). */
	async function grabAverage( n ) {

		let acc = null;
		for ( let i = 0; i < n; i ++ ) {

			state.frame ++;
			renderFrame( false );
			const g = grabLinear();
			await device.queue.onSubmittedWorkDone();
			if ( ! acc ) acc = g;
			else for ( let q = 0; q < acc.lin.length; q ++ ) acc.lin[ q ] += g.lin[ q ];

		}

		for ( let q = 0; q < acc.lin.length; q ++ ) acc.lin[ q ] /= n;
		return acc;

	}

	/**
	 * Temporal metrics over `frames` consecutive frames (after `warm` frames with the same motion).
	 * All in sRGB luma units (0..255), averaged over ALL pixels of the w x h image:
	 *   mean     mean luma (to judge brightness / energy differences between strategies)
	 *   d1       mean |Y[t] - Y[t-1]|                 frame-to-frame change (motion + shimmer + noise)
	 *   d2       mean |Y[t] - (Y[t-1] + Y[t+1]) / 2|   non-linear-in-time change: shimmer, crawl, noise
	 *            (a smoothly moving, perfectly filtered image gives ~0)
	 *   d2max    99.9th percentile of the per-pixel d2 average
	 */
	async function measure( { frames = 8, warm = 0 } = {} ) {

		await step( warm );
		const Y = [];
		for ( let i = 0; i < frames; i ++ ) {

			renderFrame();
			Y.push( luma( grabLinear() ) );
			await device.queue.onSubmittedWorkDone();

		}

		const n = W * H;
		let mean = 0, d1 = 0, d2 = 0;
		const per = new Float32Array( n );
		for ( let i = 0; i < n; i ++ ) {

			for ( let t = 0; t < frames; t ++ ) {

				mean += Y[ t ][ i ];
				if ( t > 0 ) d1 += Math.abs( Y[ t ][ i ] - Y[ t - 1 ][ i ] );
				if ( t > 0 && t < frames - 1 ) per[ i ] += Math.abs( Y[ t ][ i ] - 0.5 * ( Y[ t - 1 ][ i ] + Y[ t + 1 ][ i ] ) );

			}

			per[ i ] /= Math.max( 1, frames - 2 );
			d2 += per[ i ];

		}

		const sorted = Float32Array.from( per ).sort();
		const r = ( v ) => Math.round( v * 1000 ) / 1000;
		return { strategy: switcher.current.def.name, frames, mean: r( mean / ( n * frames ) ), d1: r( d1 / ( n * Math.max( 1, frames - 1 ) ) ), d2: r( d2 / n ), d2max: r( sorted[ Math.floor( n * 0.999 ) ] ), shimmerMap: per };

	}

	// ── overlay: zoomed crops and the comparison grid ──────────────────────────────────────────
	const overlay = document.getElementById( 'overlay' );
	const octx = overlay.getContext( '2d' );
	const tmp = document.createElement( 'canvas' );
	function drawCrop( image, crop, zoom, x, y, label ) {

		tmp.width = image.width; tmp.height = image.height;
		tmp.getContext( '2d' ).putImageData( image, 0, 0 );
		octx.imageSmoothingEnabled = false;
		octx.drawImage( tmp, crop[ 0 ], crop[ 1 ], crop[ 2 ], crop[ 3 ], x, y, crop[ 2 ] * zoom, crop[ 3 ] * zoom );
		if ( label ) {

			octx.font = '12px system-ui';
			octx.fillStyle = 'rgba(0,0,0,0.6)';
			octx.fillRect( x, y, octx.measureText( label ).width + 10, 17 );
			octx.fillStyle = '#e8eefc';
			octx.fillText( label, x + 5, y + 13 );

		}

	}

	const mapImage = ( values, gain ) => {

		const data = new Uint8ClampedArray( W * H * 4 );
		for ( let i = 0; i < W * H; i ++ ) {

			const v = Math.min( 255, values[ i ] * gain );
			data[ 4 * i ] = v; data[ 4 * i + 1 ] = v * 0.8; data[ 4 * i + 2 ] = v * 0.4; data[ 4 * i + 3 ] = 255;

		}

		return new ImageData( data, W, H );

	};

	/** Ribbon widths and the AA pad are given in OUTPUT pixels; k = pass pixels per output pixel. */
	const setPxScale = ( k ) => {

		U.pxScale.value = k; U.edgePad.value = num( 'pad', 0.5 ) * k;

	};

	/** "name[:key=value]..." -> { name, over, pairs }; `vel` and `pxscale` are applied to uniforms. */
	function parseCell( spec ) {

		const [ name, ...pairs ] = spec.split( ':' );
		const over = overrides();
		U.trueMotion.value = P( 'vel', 'true' ) === 'camera' ? 0 : 1;
		setPxScale( num( 'pxscale', SS ) );
		for ( const pair of pairs ) {

			const [ k, v ] = pair.split( '=' );
			if ( k === 'vel' ) U.trueMotion.value = v === 'camera' ? 0 : 1;
			else if ( k === 'pxscale' ) setPxScale( + v );
			else over[ k ] = v === 'true' ? true : v === 'false' ? false : isNaN( + v ) ? v : + v;

		}

		return { name, over, pairs };

	}

	async function runGrid() {

		const names = P( 'grid', '' ).split( ',' ).filter( Boolean );
		if ( params.has( 'ref' ) ) names.unshift( 'ref@' + params.get( 'ref' ) );
		const crop = P( 'crop', `0,0,${W},${H}` ).split( ',' ).map( Number );
		const zoom = num( 'zoom', 1 ), cols = num( 'cols', 3 ), mode = P( 'gridmode', 'image' ), gain = num( 'gain', 8 );
		const warm = num( 'warm', 24 ), frames = num( 'frames', 6 );
		const cw = crop[ 2 ] * zoom + 6, ch = crop[ 3 ] * zoom + 6;
		overlay.width = cols * cw; overlay.height = Math.ceil( names.length / cols ) * ch;
		overlay.style.display = 'block';
		octx.fillStyle = '#101218'; octx.fillRect( 0, 0, overlay.width, overlay.height );
		const report = [];
		let cell = 0;
		const place = ( image, label ) => {

			drawCrop( image, crop, zoom, ( cell % cols ) * cw, Math.floor( cell / cols ) * ch, label );
			cell ++;

		};

		const yaw0 = state.yaw, frame0 = state.frame;
		const total = warm + ( mode === 'image' ? 0 : frames ); // every cell ends at the same pose
		let reference = null;

		for ( const spec of names ) {

			const t0 = performance.now();

			// "ref@<cell>": a supersampled (refss) render of that cell at the final pose, optionally
			// averaged over refacc frames of fresh noise; the following cells are compared with it
			if ( spec.startsWith( 'ref@' ) ) {

				const { name, over, pairs } = parseCell( spec.slice( 4 ) );
				const refSS = num( 'refss', 3 ), refAcc = num( 'refacc', 1 );
				renderer.setPixelRatio( refSS );
				renderer.setSize( W, H );
				setPxScale( refSS );
				const orbit = state.orbit, phaseStep = state.phaseStep;
				state.orbit = 0; state.yaw = yaw0 + orbit * total;
				state.phaseStep = 0; test.motion.phase = phaseStep * total;
				const s = use( name, over );
				if ( refAcc > 1 ) temporalNoise = true;
				await step( 2 );
				reference = await grabAverage( refAcc );
				place( toImage( reference ), `reference: ${s.def.label} ${pairs.join( ' ' )} @ ${refSS}x SSAA${refAcc > 1 ? ' x ' + refAcc + ' frames' : ''}` );
				state.orbit = orbit; state.phaseStep = phaseStep;
				renderer.setPixelRatio( SS );
				renderer.setSize( W, H );
				continue;

			}

			const { name, over, pairs } = parseCell( spec );
			const s = use( name, over );
			if ( pairs.length ) s.def.label += ' [' + pairs.join( ' ' ) + ']';
			// only temporal strategies need history
			const w = s.def.temporal ? warm : Math.min( warm, 2 );
			state.yaw = yaw0 + state.orbit * ( warm - w ); state.frame = frame0; test.motion.phase = state.phaseStep * ( warm - w );
			await step( mode === 'image' ? w - 1 : w );
			const entry = { strategy: spec, label: s.def.label };
			let image;
			if ( mode === 'image' ) {

				renderFrame(); // the grabbed frame is still a moving frame
				const g = grabLinear();
				await device.queue.onSubmittedWorkDone();
				image = toImage( g );
				if ( reference ) {

					const a = luma( g ), b = luma( reference );
					let e = 0, ma = 0, mb = 0;
					for ( let i = 0; i < a.length; i ++ ) {

						e += Math.abs( a[ i ] - b[ i ] ); ma += a[ i ]; mb += b[ i ];

					}

					entry.maeVsRef = Math.round( 1000 * e / a.length ) / 1000;
					entry.meanLuma = Math.round( 1000 * ma / a.length ) / 1000;
					entry.refMeanLuma = Math.round( 1000 * mb / a.length ) / 1000;
					if ( P( 'show', '' ) === 'error' ) image = mapImage( a.map( ( v, i ) => Math.abs( v - b[ i ] ) ), gain );

				}

			} else {

				const m = await measure( { frames } );
				image = mapImage( m.shimmerMap, gain );
				delete m.shimmerMap;
				Object.assign( entry, m );

			}

			entry.ms = Math.round( performance.now() - t0 );
			place( image, `${s.def.label}${entry.d2 !== undefined ? '   d2=' + entry.d2 : ''}${entry.maeVsRef !== undefined ? '   mae=' + entry.maeVsRef : ''}` );
			report.push( entry );

		}

		return report;

	}

	// ── runtime switching / leak test ──────────────────────────────────────────────────────────
	async function switchTest( cycles = 3, { names = Object.keys( STRATEGIES ), over = {}, disposeMaterials = false } = {} ) {

		const snapshot = () => ( { liveTextures: gpu.liveTextures, liveTextureMB: Math.round( gpu.liveTextureBytes / 1048576 * 10 ) / 10, renderPipelines: gpu.renderPipelines, liveBuffers: gpu.buffersCreated - gpu.buffersDestroyed, info: { ...renderer.info.memory } } );
		const log = [ { cycle: 'start', ...snapshot() } ];
		for ( let c = 0; c < cycles; c ++ ) {

			for ( const name of [ ...names, names[ 0 ] ] ) {

				use( name, over );
				if ( disposeMaterials ) test.disposeUnusedMaterials();
				await step( 2 );

			}

			log.push( { cycle: c + 1, ...snapshot() } );

		}

		return log;

	}

	Object.assign( window, { __step: step, __measure: async ( o ) => {

		const m = await measure( o );
		delete m.shimmerMap;
		return m;

	}, __use: async ( name, over ) => {

		use( name, over );
		await step( 2 );
		return switcher.current.def;

	}, __switchTest: switchTest, __grab: () => {

		const image = toImage( grabLinear() );
		return { width: W, height: H, data: image.data };

	}, __state: state, __test: test, __renderer: renderer, __switcher: switcher } );

	// ── start ──────────────────────────────────────────────────────────────────────────────────
	placeCamera();
	if ( params.has( 'grid' ) ) {

		window.__report = await runGrid();

	} else {

		const s = use( P( 's', 's1' ) );
		await step( num( 'warm', 24 ) );
		hud.textContent = `${s.def.label}   edges=${s.def.edges}  samples=${s.def.samples}  ${W}x${H} ss=${SS}  frame ${state.frame}`;
		if ( params.has( 'crop' ) ) {

			const crop = params.get( 'crop' ).split( ',' ).map( Number ), zoom = num( 'zoom', 4 );
			renderFrame( false );
			const image = toImage( grabLinear() );
			overlay.width = crop[ 2 ] * zoom; overlay.height = crop[ 3 ] * zoom;
			overlay.style.display = 'block';
			drawCrop( image, crop, zoom, 0, 0, s.def.label );

		}

	}

	window.__ready = true;

}

main().catch( ( e ) => {

	console.error( e );
	window.__error = String( e && e.stack || e );

} );
