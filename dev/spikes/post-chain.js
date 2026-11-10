// Spike: the composed cinematic post-processing chain (reference code, see recipes/post.md).
//
//   const chain = buildChain( app, options, uniforms );   // options = structural (need a rebuild)
//   app.pipeline.outputColorTransform = false;            // the chain tone maps + encodes itself
//   app.pipeline.outputNode = chain.output; app.pipeline.needsUpdate = true;
//   …
//   chain.dispose();                                       // frees every render target it created
//
// Order (each stage is optional; `uniforms` hold the live strengths):
//
//   scene pass (HDR RGBA16F + packed view normal RGBA8 + depth; MSAA n× or 1×)
//     ├─ GTAO (R8, resolution scale)  ─► denoise ─┐          or SSGI (AO R8 + GI RG11B10)
//     ├─ SSS contact shadows (R8)  ───────────────┤
//     └────────────────────────────────────────────► "lit" composite: beauty·AO·shadow + GI, depth fog
//   lit (RGBA16F rtt) ─► DoF (CoC, ½-res bokeh blurs, composite) ─► [motion blur] ─┐
//        ┌──────────────────────────────────────────────────────────────────────────┘
//        ├─► bloom (bright pass + 5 mips) ─► lens dirt / ghost flare / anamorphic streak
//        └─► final composite: chromatic aberration (3 taps of the DoF texture) + bloom + lens
//            extras, vignette, exposure (manual × auto), tone mapping, sRGB encode
//   ─► [RCAS sharpen] ─► grain + dither ─► canvas
//
// Everything that owns a render target is pushed to `disposables`, including the rtt() nodes this
// file creates on purpose instead of letting the effect nodes call convertToTexture() internally
// (those hidden RTTNodes are never disposed by the effects' own dispose()).
import * as THREE from 'three/webgpu';
import {
	pass, mrt, output, rtt, sample, unpackRGBToNormal, uniform, vec2, vec3, vec4, float, Fn, mix, max, min,
	smoothstep, luminance, length, exp, screenUV, screenSize, screenCoordinate, renderOutput, time, uv,
	interleavedGradientNoise, rand, fract, texture, builtinAOContext, clamp, passTexture, int, abs, textureSize,
	perspectiveDepthToViewZ,
} from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { ssgi } from 'three/addons/tsl/display/SSGINode.js';
import { sss } from 'three/addons/tsl/display/SSSNode.js';
import { denoise } from 'three/addons/tsl/display/DenoiseNode.js';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { lensDirt } from 'three/addons/tsl/display/lensDirt.js';
import { lensflare } from 'three/addons/tsl/display/LensflareNode.js';
import { gaussianBlur } from 'three/addons/tsl/display/GaussianBlurNode.js';
import { chromaticAberration } from 'three/addons/tsl/display/ChromaticAberrationNode.js';
import { sharpen } from 'three/addons/tsl/display/SharpenNode.js';
import { film } from 'three/addons/tsl/display/FilmNode.js';
import { boxBlur } from 'three/addons/tsl/display/boxBlur.js';
import { motionBlur } from 'three/addons/tsl/display/MotionBlur.js';
import { graphDof } from './post-dof.js';

export const TONE_MAPPINGS = {
	agx: THREE.AgXToneMapping,
	neutral: THREE.NeutralToneMapping,
	aces: THREE.ACESFilmicToneMapping,
	reinhard: THREE.ReinhardToneMapping,
	cineon: THREE.CineonToneMapping,
	linear: THREE.LinearToneMapping,
	none: THREE.NoToneMapping,
};

/** Structural options: changing any of them needs chain.dispose() + buildChain(). */
export const DEFAULT_OPTIONS = {
	msaa: 4, // samples of the scene pass (0/1 = off)
	discardMSAA: true, // do not store the multisampled colour (and depth, if nothing reads it)
	ao: 'gtao', // 'none' | 'gtao' | 'ssgi'
	aoRes: 0.5, // resolution scale of the AO / SSGI pass
	aoSamples: 16, // GTAO samples (baked into the shader)
	aoDenoise: 'rtt', // 'none' | 'box' (3×3 at AO res) | 'bilateral' (3×3 depth-aware upsample, own R8 target) | 'inline' (DenoiseNode in the composite) | 'rtt' (DenoiseNode, own R8 target)
	aoNormals: 'mrt', // 'mrt' (G-buffer normal) | 'depth' (reconstructed from depth, no normal attachment needed)
	aoApply: 'composite', // 'composite' | 'context' (builtinAOContext on the scene pass; needs lit materials + a pre-pass)
	giSlices: 2, giSteps: 8, // SSGI quality
	sss: false, sssRes: 1, sssBlur: true,
	fog: true,
	dof: false, // false | true (three DepthOfFieldNode) | 'graph' (post-dof.js)
	dofTaps: 48, dofRes: 0.5, dofJitter: true, // 'graph' only
	motionBlur: false, mbSamples: 12, // camera-motion blur from depth reprojection (no velocity buffer)
	bloom: true, bloomRes: 0.5,
	dirt: false, flare: false, streak: false,
	ca: false,
	sharpen: false,
	grain: 'custom', // 'none' | 'custom' | 'film' (three FilmNode)
	tone: 'neutral',
	autoExposure: false,
	litRT: false, // force the lit composite into its own render target
	view: 'final', // debug: 'final' | 'ao' | 'normal' | 'depth' | 'gi' | 'sss' | 'lit' | 'bloom' | 'dof'
};

/**
 * Recommended structural options per quality tier (see recipes/post.md for costs). Fog is off here
 * because production fog belongs in the materials (transparent edges have no depth).
 */
export const TIERS = {
	// phones: MSAA scene pass with transient attachments, ¼-res bloom, tone map, dither
	low: { msaa: 4, ao: 'none', fog: false, dof: false, bloom: true, bloomRes: 0.25, sss: false, dirt: false, flare: false, streak: false, ca: false, sharpen: false, motionBlur: false, autoExposure: false, grain: 'none' },
	// integrated GPUs / good phones: + half-res GTAO from depth only (no normal attachment), 3×3 box
	medium: { msaa: 4, ao: 'gtao', aoRes: 0.5, aoSamples: 16, aoDenoise: 'box', aoNormals: 'depth', fog: false, dof: false, bloom: true, bloomRes: 0.5, sss: false, dirt: false, flare: false, streak: false, ca: false, sharpen: false, motionBlur: false, autoExposure: false, grain: 'custom' },
	// desktop: + normal attachment, depth-aware denoise; DoF is a user toggle ( dof: 'graph' )
	high: { msaa: 4, ao: 'gtao', aoRes: 0.5, aoSamples: 16, aoDenoise: 'rtt', aoNormals: 'mrt', fog: false, dof: false, bloom: true, bloomRes: 0.5, sss: false, dirt: false, flare: false, streak: false, ca: false, sharpen: false, motionBlur: false, autoExposure: false, grain: 'custom' },
	// discrete GPU: full-res 32-sample GTAO, bokeh DoF, streak, subtle CA, RCAS
	ultra: { msaa: 4, ao: 'gtao', aoRes: 1, aoSamples: 32, aoDenoise: 'rtt', aoNormals: 'mrt', fog: false, dof: 'graph', dofTaps: 48, dofRes: 0.5, bloom: true, bloomRes: 0.5, sss: false, dirt: false, flare: false, streak: true, ca: true, sharpen: true, motionBlur: false, autoExposure: false, grain: 'custom' },
};

/** Live uniforms shared across rebuilds. */
export function createUniforms() {

	return {
		aoStrength: uniform( 1 ),
		aoRadius: uniform( 1 ), // world units → GTAONode.radius (set from the scene scale)
		aoThickness: uniform( 1 ), // world units
		aoScale: uniform( 1 ), // GTAO exponent
		giStrength: uniform( 1 ),
		sssStrength: uniform( 0.8 ),
		fogDensity: uniform( 0 ), // 1 / world units; fog = 1 - exp( -(density·dist)² ) beyond fogStart
		fogStart: uniform( 0 ),
		fogColor: uniform( new THREE.Color( 0x000000 ) ),
		focus: uniform( 100 ), // world units along the view direction
		focalRange: uniform( 50 ), // world units until fully out of focus
		bokeh: uniform( 4 ), // DoF bokeh scale ≈ max blur radius in full-res pixels (per blur pass)
		bloomStrength: uniform( 0.72 ),
		bloomRadius: uniform( 0.4 ),
		bloomThreshold: uniform( 0.24 ),
		dirtStrength: uniform( 1.5 ),
		flareStrength: uniform( 0.6 ),
		streakStrength: uniform( 0.5 ),
		caStrength: uniform( 1 ),
		vignette: uniform( 0.22 ),
		exposure: uniform( 1 ), // manual exposure (multiplies the auto exposure)
		autoKey: uniform( 0.35 ), // target mean luminance of the lit (non-black) pixels for auto exposure
		autoExposure: uniform( 1 ), // written by the chain's CPU easing (see updateAutoExposure)
		sharpness: uniform( 0.4 ), // RCAS: 0 = max, 2 = off
		mbStrength: uniform( 1 ), // motion blur: fraction of the frame-to-frame motion (shutter)
		grain: uniform( 0.035 ),
		dither: uniform( 1 / 255 ),
	};

}

/** Procedural lens-dirt texture: soft out-of-focus dust discs (no asset needed). */
function makeDirtTexture( size = 256 ) {

	const data = new Uint8Array( size * size * 4 );
	let seed = 7;
	const rnd = () => ( seed = ( seed * 16807 ) % 2147483647 ) / 2147483647;
	const acc = new Float32Array( size * size );
	for ( let k = 0; k < 90; k ++ ) {

		const cx = rnd() * size, cy = rnd() * size, r = 4 + rnd() * rnd() * 26, a = 0.15 + rnd() * 0.35;
		const x0 = Math.max( 0, Math.floor( cx - r ) ), x1 = Math.min( size - 1, Math.ceil( cx + r ) );
		const y0 = Math.max( 0, Math.floor( cy - r ) ), y1 = Math.min( size - 1, Math.ceil( cy + r ) );
		for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) {

			const d = Math.hypot( x - cx, y - cy ) / r;
			if ( d < 1 ) acc[ y * size + x ] += a * ( 0.6 + 0.4 * d ) * Math.min( 1, ( 1 - d ) * 6 );

		}

	}

	for ( let i = 0; i < size * size; i ++ ) {

		const v = Math.min( 255, acc[ i ] * 255 );
		data[ 4 * i ] = v; data[ 4 * i + 1 ] = v; data[ 4 * i + 2 ] = v; data[ 4 * i + 3 ] = 255;

	}

	const t = new THREE.DataTexture( data, size, size );
	t.magFilter = t.minFilter = THREE.LinearFilter;
	t.needsUpdate = true;
	return t;

}

export function buildChain( app, options = {}, U = createUniforms() ) {

	const o = { ...DEFAULT_OPTIONS, ...options };
	const { scene, camera, renderer } = app;
	const disposables = [];
	const track = ( n ) => ( disposables.push( n ), n );
	// PITFALL: rtt() creates its RenderTarget with the default depthBuffer: true, i.e. every rtt()
	// silently allocates a depth texture of the same size (4 bytes / pixel) unless told otherwise
	const RT = ( node, opts = {}, w = null, h = null ) => track( rtt( node, w, h, { depthBuffer: false, ...opts } ) );
	const info = { passes: [], notes: [] };

	// ── scene pass + G-buffer ────────────────────────────────────────────────────────────────
	const samples = o.msaa > 1 ? o.msaa : 0;
	const needNormal = ( o.ao !== 'none' && o.aoNormals === 'mrt' ) || o.view === 'normal';
	// discardMSAA: the multisampled colour samples are only needed for the resolve, and the
	// multisampled depth only if something samples the depth texture afterwards. Not storing them
	// lets the backend use transient attachments (tile-based GPUs keep them out of main memory).
	const needDepth = o.ao !== 'none' || o.sss || o.dof || o.fog || o.motionBlur || o.view === 'depth';
	const passOptions = { samples };
	if ( o.discardMSAA && samples > 1 ) {

		passOptions.storeMultisampledColorBuffer = false;
		if ( ! needDepth ) passOptions.storeMultisampledDepthBuffer = false;

	}

	const scenePass = track( pass( scene, camera, passOptions ) );
	scenePass.name = 'scene';
	if ( needNormal ) {

		const sceneMRT = mrt( { output, normal: vec4( 0.5, 0.5, 1, 1 ) } );
		// transparent materials blend into 'normal' with their own blending (and output alpha 0 there)
		sceneMRT.setBlendMode( 'normal', new THREE.BlendMode( THREE.MaterialBlending ) );
		// background = packed ( 0, 0, 1 ). Without it the attachment is cleared to the scene background
		// colour, which unpacks to a normal pointing away from the camera: every MSAA-resolved or
		// half-res silhouette pixel then gets a black AO outline
		sceneMRT.setClearColor( 'normal', new THREE.Color( 0.5, 0.5, 1 ), 1 );
		scenePass.setMRT( sceneMRT );
		scenePass.getTexture( 'normal' ).type = THREE.UnsignedByteType; // RGBA8 instead of RGBA16F

	}

	const beauty = scenePass.getTextureNode( 'output' );
	const depth = scenePass.getTextureNode( 'depth' );
	const viewZ = scenePass.getViewZNode();
	const normalTex = needNormal ? scenePass.getTextureNode( 'normal' ) : null;
	const normal = needNormal && o.aoNormals === 'mrt' ? sample( ( uvNode ) => unpackRGBToNormal( normalTex.sample( uvNode ).rgb ) ) : null;

	// ── ambient occlusion / GI ───────────────────────────────────────────────────────────────
	let aoNode = null, giNode = null, aoPass = null;
	if ( o.ao === 'gtao' ) {

		aoPass = track( ao( depth, normal, camera ) );
		aoPass.resolutionScale = o.aoRes;
		aoPass.samples.value = o.aoSamples;
		aoPass.radius = U.aoRadius;
		aoPass.thickness = U.aoThickness;
		aoPass.scale = U.aoScale;
		aoNode = aoPass.getTextureNode();

	} else if ( o.ao === 'ssgi' ) {

		aoPass = track( ssgi( beauty, depth, normal, camera ) );
		aoPass.resolutionScale = o.aoRes;
		aoPass.sliceCount.value = o.giSlices;
		aoPass.stepCount.value = o.giSteps;
		aoPass.thickness = U.aoThickness;
		aoPass.useTemporalFiltering = false; // no TRAA in this spike
		aoNode = aoPass.getAONode();
		giNode = aoPass.getGINode();

	}

	if ( aoNode && o.aoDenoise === 'box' ) {

		// cheap alternative: 3×3 box at the AO resolution (9 taps), not depth aware
		aoNode = RT( boxBlur( aoNode, { size: 1, separation: 1 } ), { resolutionScale: o.aoRes, format: THREE.RedFormat, type: THREE.UnsignedByteType } );

	} else if ( aoNode && o.aoDenoise === 'bilateral' ) {

		// joint bilateral upsample (not a three.js node): 3×3 taps of the low-res AO around the pixel,
		// each weighted by a Gaussian and by the view-space depth difference between the tap and the
		// full-res centre. 9 × ( AO + depth ) + 1 = 19 taps per full-res pixel, no normals needed.
		const aoTex = aoNode;
		// cameraNear / cameraFar would be those of the quad camera inside an rtt(): use own uniforms
		const camNear = uniform( camera.near ).onRenderUpdate( () => camera.near );
		const camFar = uniform( camera.far ).onRenderUpdate( () => camera.far );
		const zAt = ( q ) => perspectiveDepthToViewZ( depth.sample( q ).r, camNear, camFar );
		aoNode = RT( Fn( () => {

			const p = uv();
			const texel = vec2( 1 ).div( vec2( textureSize( aoTex ) ) );
			const zc = zAt( p ).toVar();
			const phi = U.aoRadius.mul( 0.1 ); // world units
			const sum = float( 0 ).toVar(), wsum = float( 0 ).toVar();
			for ( let j = - 1; j <= 1; j ++ ) for ( let i = - 1; i <= 1; i ++ ) {

				const q = p.add( vec2( i, j ).mul( texel ) );
				const g = Math.exp( - ( i * i + j * j ) / 2 );
				const w = exp( abs( zAt( q ).sub( zc ) ).div( phi ).negate() ).mul( g );
				sum.addAssign( aoTex.sample( q ).r.mul( w ) );
				wsum.addAssign( w );

			}

			return vec4( vec3( sum.div( max( wsum, 1e-4 ) ) ), 1 );

		} )(), { format: THREE.RedFormat, type: THREE.UnsignedByteType } );

	} else if ( aoNode && o.aoDenoise !== 'none' ) {

		// DenoiseNode is an inline node (16 × ( value + depth + normal ) taps where it is evaluated)
		const mk = ( tex ) => {

			const d = track( denoise( tex, depth, normal, camera ) );
			d.depthPhi = U.aoThickness; // world units
			d.radius.value = 5;
			return d;

		};

		const dAO = mk( aoNode );
		aoNode = o.aoDenoise === 'rtt' ? RT( dAO, { format: THREE.RedFormat, type: THREE.UnsignedByteType } ) : dAO;
		if ( giNode ) {

			const dGI = mk( giNode );
			giNode = o.aoDenoise === 'rtt' ? RT( dGI ) : dGI;

		}

	}

	if ( aoNode && o.aoApply === 'context' ) {

		// Only affects materials with lights = true (NodeMaterial.setupLighting → AONode); and the AO
		// then comes from THIS frame's scene pass depth, which is a feedback loop: a real setup
		// needs a separate depth/normal pre-pass. Kept to demonstrate that hand-lit materials ignore it.
		scenePass.contextNode = builtinAOContext( aoNode.sample( screenUV ).r );

	}

	// ── screen-space contact shadows ─────────────────────────────────────────────────────────
	let sssNode = null, sssPass = null;
	if ( o.sss && app.keyLight ) {

		sssPass = track( sss( depth, camera, app.keyLight ) );
		sssPass.resolutionScale = o.sssRes;
		sssNode = sssPass.getTextureNode();
		if ( o.sssBlur ) sssNode = RT( boxBlur( sssNode, { size: 2, separation: 1 } ), { format: THREE.RedFormat, type: THREE.UnsignedByteType } );

	}

	// ── lit composite: AO, shadows, GI, fog ──────────────────────────────────────────────────
	const litFn = Fn( () => {

		const c = beauty.rgb.toVar();
		// bright (emissive / HDR) pixels are protected: AO and contact shadows only belong on
		// reflected light, and without a separate emissive buffer luminance is the best proxy
		const protect = smoothstep( 0.9, 2.5, luminance( c ) ).oneMinus();
		if ( aoNode && o.aoApply === 'composite' ) c.mulAssign( mix( float( 1 ), aoNode.r, U.aoStrength.mul( protect ) ) );
		if ( sssNode ) c.mulAssign( mix( float( 1 ), sssNode.r, U.sssStrength.mul( protect ) ) );
		if ( giNode ) c.addAssign( giNode.rgb.mul( U.giStrength ) );
		if ( o.fog ) {

			const dist = max( viewZ.negate().sub( U.fogStart ), 0 );
			const f = float( 1 ).sub( exp( dist.mul( U.fogDensity ).pow2().negate() ) );
			c.assign( mix( c, U.fogColor, f ) );

		}

		return vec4( c, 1 );

	} );

	// DoF, chromatic aberration, motion blur and the exposure meter sample the lit image at other
	// UVs, so they need it as a texture (one full-res RGBA16F pass). Otherwise the composite stays
	// an inline node that the bloom high-pass and the final pass each evaluate (2–4 taps).
	const litChanges = aoNode || sssNode || giNode || o.fog;
	const needLitRT = litChanges && ( o.dof || o.ca || o.motionBlur || o.autoExposure || o.litRT );
	let hdrTex = needLitRT ? RT( litFn() ) : litChanges ? litFn() : beauty;
	const litTex = hdrTex;

	// ── depth of field ───────────────────────────────────────────────────────────────────────
	let dofPass = null;
	if ( o.dof === 'graph' ) {

		// custom CoC-weighted gather (post-dof.js): no halo around in-focus nodes on the far background
		dofPass = track( graphDof( hdrTex, viewZ, { focus: U.focus, range: U.focalRange, maxBlur: U.bokeh, taps: o.dofTaps, resolutionScale: o.dofRes, jitter: o.dofJitter } ) );
		hdrTex = RT( dofPass.output ); // bloom and the final composite want a texture

	} else if ( o.dof ) {

		dofPass = track( dof( hdrTex, viewZ, U.focus, U.focalRange, U.bokeh ) );
		// PITFALL: DepthOfFieldNode.getTextureNode() is a plain texture() node without a link back
		// to the DoF node, so using only that leaves the DoF passes un-built and the result black.
		// passTexture( node, texture ) creates a sampleable texture node that keeps the dependency.
		hdrTex = passTexture( dofPass, dofPass.getTextureNode().value );

	}

	// ── motion blur (camera motion only): velocity from depth + previous view-projection ──────
	// uvPrev = project( prevViewProj · unproject( uv, depth ) ); MotionBlur.js samples along it.
	let mb = null;
	if ( o.motionBlur ) {

		const invVP = uniform( new THREE.Matrix4() ), prevVP = uniform( new THREE.Matrix4() );
		const velocity = Fn( () => {

			const p = uv();
			const ndc = vec4( p.x.mul( 2 ).sub( 1 ), p.y.oneMinus().mul( 2 ).sub( 1 ), depth.sample( p ).r, 1 );
			const w = invVP.mul( ndc );
			const prev = prevVP.mul( vec4( w.xyz.div( w.w ), 1 ) );
			const prevUV = vec2( prev.x.div( prev.w ).mul( 0.5 ).add( 0.5 ), prev.y.div( prev.w ).mul( 0.5 ).add( 0.5 ).oneMinus() );
			// clamp to 5 % of the screen so a camera cut does not smear the whole frame
			const v = p.sub( prevUV ).mul( U.mbStrength );
			return v.mul( min( float( 1 ), float( 0.05 ).div( max( length( v ), 1e-6 ) ) ) );

		} )();
		const blurred = RT( motionBlur( hdrTex, velocity, int( o.mbSamples ) ) );
		hdrTex = blurred;
		const cur = new THREE.Matrix4();
		mb = {
			/** call once per frame BEFORE rendering: stores last frame's matrix, then the current one */
			update() {

				prevVP.value.copy( cur );
				camera.updateMatrixWorld();
				cur.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
				invVP.value.copy( cur ).invert();

			},
		};

	}

	// ── bloom + lens extras ──────────────────────────────────────────────────────────────────
	let bloomPass = null, flareNode = null, streakNode = null, dirtTex = null;
	if ( o.bloom ) {

		bloomPass = track( bloom( hdrTex, U.bloomStrength, U.bloomRadius, U.bloomThreshold ) );
		bloomPass.smoothWidth.value = 0.5;
		bloomPass.setResolutionScale( o.bloomRes );
		if ( o.flare ) {

			const ghosts = track( lensflare( bloomPass.getTextureNode(), { threshold: float( 0.05 ), ghostSamples: float( 3 ), ghostSpacing: float( 0.3 ), ghostAttenuationFactor: float( 12 ), downSampleRatio: 4 } ) );
			flareNode = track( gaussianBlur( ghosts.getTextureNode(), null, 3 ) ).getTextureNode();

		}

		if ( o.streak ) {

			// anamorphic streak: the bloom texture blurred horizontally only, at ¼ resolution
			const bright = RT( bloomPass.getTextureNode(), { resolutionScale: 0.25 } );
			streakNode = track( gaussianBlur( bright, vec2( 4, 0 ), 8 ) ).getTextureNode();

		}

		if ( o.dirt ) dirtTex = track( makeDirtTexture() );

	}

	// ── auto exposure: mean luminance of the non-black pixels through a mip chain ────────────
	//   lumRT   64×64 RG16F: ( lum·w, w ), w = 1 where the pixel is not background-dark (a plain
	//           log-average is dominated by the black void and would always over-expose)
	//   meterRT 1×1 RGBA32F: top mip of lumRT, read back asynchronously and eased on the CPU
	let lumRT = null, meterRT = null;
	if ( o.autoExposure ) {

		lumRT = RT( Fn( () => {

			// 4×4 taps per texel so that small bright nodes are not missed entirely
			const acc = vec2( 0 ).toVar();
			for ( let j = 0; j < 4; j ++ ) for ( let i = 0; i < 4; i ++ ) {

				const l = luminance( hdrTex.sample( uv().add( vec2( ( i - 1.5 ) / 256, ( j - 1.5 ) / 256 ) ) ).rgb );
				const w = smoothstep( 0.02, 0.06, l );
				acc.addAssign( vec2( min( l, 8 ).mul( w ), w ) );

			}

			return vec4( acc.div( 16 ), 0, 1 );

		} )(), { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter }, 64, 64 );
		meterRT = RT( Fn( () => vec4( lumRT.level( 6 ).rg, 0, 1 ) )(), { type: THREE.FloatType }, 1, 1 );

	}

	// ── final composite (HDR) ────────────────────────────────────────────────────────────────
	const toneMapping = TONE_MAPPINGS[ o.tone ] ?? THREE.NeutralToneMapping;
	const hdrFn = Fn( () => {

		// chromatic aberration: 3 taps of the sharp layer, bloom is blurry enough to skip
		const c = ( o.ca ? chromaticAberration( hdrTex, U.caStrength, vec2( 0.5, 0.5 ), float( 1.1 ) ).rgb : hdrTex.rgb ).toVar();
		if ( bloomPass ) {

			const b = bloomPass.getTextureNode();
			c.addAssign( b.rgb );
			if ( dirtTex ) c.addAssign( lensDirt( b, texture( dirtTex ), U.dirtStrength ).rgb );
			if ( flareNode ) c.addAssign( flareNode.rgb.mul( U.flareStrength ) );
			if ( streakNode ) c.addAssign( streakNode.rgb.mul( vec3( 0.55, 0.75, 1.0 ) ).mul( U.streakStrength ) );

		}

		// vignette (aspect-correct, linear light)
		const aspect = screenSize.x.div( max( screenSize.y, 1 ) );
		const p = screenUV.sub( 0.5 ).mul( vec2( aspect, 1 ) );
		const maxR = length( vec2( aspect.mul( 0.5 ), 0.5 ) );
		c.mulAssign( float( 1 ).sub( U.vignette.mul( smoothstep( 0.3, 1.0, length( p ).div( maxR ) ) ) ) );

		// exposure: manual × auto (renderer.toneMappingExposure is applied by the tone mapping node too)
		let e = U.exposure;
		if ( o.autoExposure === 'gpu' ) {

			const m = lumRT.level( 6 ).rg; // instant, no easing
			e = e.mul( clamp( U.autoKey.div( max( m.x.div( max( m.y, 1e-4 ) ), 1e-3 ) ), 0.25, 4 ) );

		} else if ( o.autoExposure ) {

			// meterRT must be part of the node graph to be rendered: sample it with weight 0
			e = e.mul( U.autoExposure ).add( meterRT.r.mul( 0 ) );

		}
		return vec4( c.mul( e ), 1 );

	} );

	const debugView = {
		ao: () => vec4( vec3( aoNode ? aoNode.r : 1 ), 1 ),
		normal: () => vec4( normalTex ? normalTex.rgb : vec3( 0 ), 1 ),
		depth: () => vec4( vec3( viewZ.negate().div( U.focus.mul( 2 ) ) ), 1 ),
		gi: () => vec4( giNode ? giNode.rgb.mul( U.giStrength ) : vec3( 0 ), 1 ),
		sss: () => vec4( vec3( sssNode ? sssNode.r : 1 ), 1 ),
		lit: () => vec4( litTex.rgb, 1 ),
		bloom: () => vec4( bloomPass ? bloomPass.getTextureNode().rgb : vec3( 0 ), 1 ),
		dof: () => vec4( hdrTex.rgb, 1 ),
	}[ o.view ];

	// tone mapping + sRGB encode (the pipeline's own transform is off)
	let ldr = debugView
		? renderOutput( debugView(), THREE.NoToneMapping, o.view === 'lit' || o.view === 'bloom' || o.view === 'dof' || o.view === 'gi' ? THREE.SRGBColorSpace : THREE.NoColorSpace )
		: renderOutput( hdrFn(), toneMapping, THREE.SRGBColorSpace );

	// ── RCAS sharpening: on the tone mapped, encoded [0,1] image, before grain / dither ──────
	let sharpenPass = null;
	if ( o.sharpen && ! debugView ) {

		const ldrTex = RT( ldr );
		sharpenPass = track( sharpen( ldrTex, U.sharpness, false ) );
		ldr = sharpenPass.getTextureNode();

	}

	// ── grain + dither (display referred) ────────────────────────────────────────────────────
	const outFn = Fn( () => {

		const c = vec4( ldr ).rgb.toVar();
		if ( ! debugView ) {

			if ( o.grain === 'film' ) c.assign( film( vec4( c, 1 ), U.grain.mul( 4 ) ).rgb );
			if ( o.grain === 'custom' ) {

				// zero-mean, strongest in the mid tones, animated
				const n = rand( fract( screenUV.add( fract( time ) ) ) ).sub( 0.5 );
				const l = luminance( c );
				c.addAssign( n.mul( U.grain ).mul( l.mul( float( 1 ).sub( l ) ).mul( 4 ).add( 0.15 ) ) );

			}

			// triangular-ish dither (two IGN taps), ±1 LSB, after the sRGB encode
			const n1 = interleavedGradientNoise( screenCoordinate.xy );
			const n2 = interleavedGradientNoise( screenCoordinate.xy.add( vec2( 17.3, 41.7 ) ) );
			c.addAssign( vec3( n1.add( n2 ).sub( 1 ).mul( U.dither ) ) );

		}

		return vec4( c, 1 );

	} );

	const chain = {
		options: o,
		uniforms: U,
		output: outFn(),
		nodes: { scenePass, aoPass, sssPass, dofPass, bloomPass, sharpenPass, lumRT, meterRT },
		/** per-frame CPU work of the chain (before pipeline.render()) */
		update( dt ) {

			mb?.update();
			this.updateAutoExposure( dt );

		},
		info,
		_lumBusy: false,
		_lumBuf: null,

		/**
		 * CPU-eased auto exposure: reads the 1×1 meter target back (asynchronously, one request
		 * in flight) and eases U.autoExposure towards key / average.
		 */
		updateAutoExposure( dt ) {

			// the first calls come before the meter target has a GPU texture: skip a few frames
			if ( ! meterRT || o.autoExposure === 'gpu' || this._lumBusy || ( this._frames = ( this._frames || 0 ) + 1 ) < 4 ) return;
			this._lumBusy = true;
			renderer.readRenderTargetPixelsAsync( meterRT.renderTarget, 0, 0, 1, 1 ).then( ( px ) => {

				const avg = px[ 1 ] > 1e-4 ? px[ 0 ] / px[ 1 ] : 0; // mean luminance of the lit pixels
				const target = avg > 0 ? Math.min( 4, Math.max( 0.25, U.autoKey.value / avg ) ) : 1;
				const k = 1 - Math.exp( - Math.max( dt, 1 / 60 ) * 2.5 );
				U.autoExposure.value += ( target - U.autoExposure.value ) * k;
				this.meter = { avg, coverage: px[ 1 ], target };

			} ).catch( ( e ) => console.warn( 'auto exposure readback failed', e ) ).finally( () => ( this._lumBusy = false ) );

		},

		dispose() {

			for ( const d of disposables ) d.dispose?.();
			disposables.length = 0;

		},
	};

	return chain;

}
