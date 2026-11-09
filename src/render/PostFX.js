// Post-processing: the final look of the frame. Builds the render pipeline output through
// app.setOutput( scenePass => ... ):
//
//   scene (HDR, MSAA) ──► bloom (three/addons BloomNode, luminance threshold) ──┐
//        └──────────────────────────────────────────────────────────────────── + ─► vignette
//        ─► tone mapping + sRGB (renderOutput, done here instead of by the pipeline) ─► dither
//
// Dithering must happen after the sRGB encode to break 8-bit banding of the dark background
// gradient, so this module turns off the pipeline's own output transform
// (app.pipeline.outputColorTransform = false) and applies tone mapping + color space itself.
//
// Look (chosen by comparing screenshots, see dev/visual.html?tone=…&bloom=…):
//   tone mapping  Neutral (Khronos PBR Neutral) — keeps the neon cluster hues saturated and
//                 rolls only the HDR node cores / dense bundles towards white. AgX washed the
//                 palette out (olive yellows, grey-green), ACES darkened reds and shifted hues.
//   bloom         threshold 0.24, smoothWidth 0.5, radius 0.4, strength 0.9 × settings.bloom
//                 (default 0.8 → 0.72): node cores (HDR emissive) and dense or highlighted
//                 bundles get a soft glow while single edges and text-sized details stay crisp.
//   vignette      −22 % luminance in the far corners; IGN dither ±1 LSB after the sRGB encode.
//
// Quality: 'high' bloom at ½ resolution; 'low' (phones) at ¼ resolution (≈4× cheaper blur);
// 'auto' picks 'low' on coarse-pointer / small screens.
//
// Public API:
//   const post = app.use( new PostFX( { bloom: 0.8, quality: 'auto' } ) );
//   post.bloomStrength = 1.2;          // settings.bloom (0 … 2), live (uniform)
//   post.enabled = false;              // plain tone mapped output, no bloom/vignette (rebuilds)
//   post.quality = 'high' | 'low' | 'auto';
//   post.toneMapping = 'agx' | 'neutral' | 'aces' | 'none';   // rebuilds
//   post.exposure = 1;                 // renderer.toneMappingExposure
//   post.threshold / post.radius / post.vignette               // live uniforms
//   post.dispose();
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, vec2, vec3, vec4, float, screenUV, screenSize, screenCoordinate, length, smoothstep,
	renderOutput, interleavedGradientNoise, max,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';

/** settings.bloom → BloomNode strength. */
const BLOOM_SCALE = 0.9;

export const TONE_MAPPINGS = {
	agx: THREE.AgXToneMapping,
	neutral: THREE.NeutralToneMapping,
	aces: THREE.ACESFilmicToneMapping,
	none: THREE.NoToneMapping,
};

/** 'low' on phones / coarse pointers / small screens, 'high' otherwise. */
export function autoQuality() {

	if ( typeof window === 'undefined' ) return 'high';
	const coarse = typeof matchMedia !== 'undefined' && matchMedia( '(pointer:coarse)' ).matches;
	const small = Math.min( window.innerWidth, window.innerHeight ) < 600;
	return coarse || small ? 'low' : 'high';

}

export class PostFX {

	constructor( { bloom: bloomStrength = 0.8, quality = 'auto', enabled = true, toneMapping = 'neutral', exposure = 1 } = {} ) {

		this.order = 90;
		this.app = null;
		this._enabled = enabled;
		this._quality = quality;
		this._toneMapping = toneMapping;
		this._exposure = exposure;
		this.bloomNode = null;

		this.uniforms = {
			strength: uniform( BLOOM_SCALE * bloomStrength ),
			threshold: uniform( 0.24 ),
			radius: uniform( 0.4 ), // effective; see `radius` (halved at low quality)
			vignette: uniform( 0.22 ),
			dither: uniform( 1 / 255 ),
		};
		this._bloomStrength = bloomStrength;
		this._radius = 0.4;

	}

	init( app ) {

		this.app = app;
		this._applyQuality();
		this.build();

	}

	get bloomStrength() {

		return this._bloomStrength;

	}

	set bloomStrength( v ) {

		this._bloomStrength = Math.max( 0, + v || 0 );
		this.uniforms.strength.value = BLOOM_SCALE * this._bloomStrength;

	}

	get threshold() {

		return this.uniforms.threshold.value;

	}

	set threshold( v ) {

		this.uniforms.threshold.value = v;

	}

	/** Bloom radius in [0, 1]. Low quality blurs at ¼ resolution (2× wider mips), so it uses half. */
	get radius() {

		return this._radius;

	}

	set radius( v ) {

		this._radius = v;
		this._applyQuality();

	}

	get vignette() {

		return this.uniforms.vignette.value;

	}

	set vignette( v ) {

		this.uniforms.vignette.value = v;

	}

	get enabled() {

		return this._enabled;

	}

	set enabled( v ) {

		v = !! v;
		if ( v === this._enabled ) return;
		this._enabled = v;
		this.build();

	}

	get quality() {

		return this._quality;

	}

	set quality( q ) {

		this._quality = q;
		this._applyQuality();

	}

	/** The effective quality ('high' | 'low') after resolving 'auto'. */
	get resolvedQuality() {

		return this._quality === 'auto' ? autoQuality() : this._quality === 'low' ? 'low' : 'high';

	}

	get toneMapping() {

		return this._toneMapping;

	}

	set toneMapping( t ) {

		if ( t === this._toneMapping ) return;
		this._toneMapping = t;
		this.build();

	}

	get exposure() {

		return this._exposure;

	}

	set exposure( v ) {

		this._exposure = v;
		if ( this.app ) this.app.renderer.toneMappingExposure = v;

	}

	/** App module hook: 'auto' quality depends on the viewport (e.g. a rotated phone). */
	resize() {

		this._applyQuality();

	}

	_applyQuality() {

		const low = this.resolvedQuality === 'low';
		this.uniforms.radius.value = this._radius * ( low ? 0.5 : 1 );
		if ( this.bloomNode ) this.bloomNode.setResolutionScale( low ? 0.25 : 0.5 );

	}

	/** (Re)builds the pipeline output node. */
	build() {

		const app = this.app;
		if ( ! app ) return;
		const U = this.uniforms;
		const toneMapping = TONE_MAPPINGS[ this._toneMapping ] ?? THREE.NeutralToneMapping;
		app.renderer.toneMapping = toneMapping;
		app.renderer.toneMappingExposure = this._exposure;
		app.pipeline.outputColorTransform = false;

		const oldBloom = this.bloomNode;
		this.bloomNode = null;

		app.setOutput( ( scenePass ) => {

			const color = scenePass.getTextureNode( 'output' );
			let hdr = color.rgb;

			if ( this._enabled ) {

				const b = bloom( color, U.strength, U.radius, U.threshold );
				b.smoothWidth.value = 0.5;
				this.bloomNode = b;
				this._applyQuality();
				hdr = hdr.add( b.rgb );

				// vignette (aspect-correct, linear light, before tone mapping)
				const aspect = screenSize.x.div( max( screenSize.y, 1 ) );
				const p = screenUV.sub( 0.5 ).mul( vec2( aspect, 1 ) );
				const maxR = length( vec2( aspect.mul( 0.5 ), 0.5 ) );
				const v = float( 1 ).sub( U.vignette.mul( smoothstep( 0.3, 1.0, length( p ).div( maxR ) ) ) );
				hdr = hdr.mul( v );

			}

			const mapped = renderOutput( vec4( hdr, 1 ), toneMapping, THREE.SRGBColorSpace );

			// triangular-ish dither (two IGN taps), ±1 LSB, after the sRGB encode
			const out = Fn( () => {

				const n1 = interleavedGradientNoise( screenCoordinate.xy );
				const n2 = interleavedGradientNoise( screenCoordinate.xy.add( vec2( 17.3, 41.7 ) ) );
				const d = n1.add( n2 ).sub( 1 ).mul( U.dither );
				return vec4( mapped.rgb.add( vec3( d ) ), 1 );

			} );

			return out();

		} );

		if ( oldBloom && oldBloom !== this.bloomNode ) oldBloom.dispose();

	}

	dispose() {

		this.bloomNode?.dispose();
		this.bloomNode = null;
		if ( this.app ) {

			this.app.pipeline.outputColorTransform = true;
			this.app.setOutput( ( scenePass ) => scenePass );

		}

	}

}
