// Depth of field for a graph in a dark void: "scatter-as-gather" bokeh (the tap test of
// D. Gustafsson, "Bokeh depth of field in a single pass", 2018, with energy-conserving weights
// instead of his running average) driven by the shared DoF uniforms of src/render/gfx/shared.js.
// The stock DepthOfFieldNode blurs the far field without looking at the taps' own circle of
// confusion, so every in-focus node gets a halo of itself on the background; here each tap
// contributes according to ITS circle of confusion, which removes the halo and still lets
// defocused nodes spread over the background and over in-focus geometry behind them.
//
// Circle of confusion (radius, scene-pass pixels): cocPixels( view distance ) of shared.js, i.e.
//   clamp( |distance − dofFocus| / dofRange, 0, 1 ) · dofMaxBlur · dofAmount · pxScale
// with one rule on top: pixels without geometry (depth at the far plane) are IN FOCUS. Blended
// strands write no depth; with this rule a strand over empty space stays sharp instead of being
// smeared at the maximum radius (strands get an analytic defocus in their own shader). A
// background sample still counts as fully blurred when a defocused surface in front of it
// gathers it, so the discs of defocused nodes keep their soft inner edge.
//
//   const dof = graphDof( colorTex, depthTex, { cameraNear, cameraFar, taps: 48, sceneScale: 1 } );
//   const hdr = rtt( dof.output, null, null, { depthBuffer: false, resolutionScale: sceneScale } );
//   dof.setSceneScale( s );     // dynamic resolution: follow scenePass.setResolutionScale( s )
//   dof.dispose();
//
//   colorTex     texture node of the HDR image (scene-pass resolution, RGBA16F)
//   depthTex     depth texture node of the scene pass (single-sample or MSAA: sample 0 is read)
//   cameraNear / cameraFar   uniform nodes with the scene camera's planes (cameraNear / cameraFar
//                of TSL would be the quad camera's inside rtt())
//   taps         gather taps (Vogel disc); clean up to a radius of ≈ 12 px with 48
//   sceneScale   resolution scale of the scene pass relative to the drawing buffer
//   innerScale   resolution of the two blur targets relative to the scene pass (default ½)
//
// Passes: prefilter (colour, bilinear 2×2 box, + normalized view distance in alpha) and gather
// (taps + 1 samples), both RGBA16F at sceneScale · innerScale; the blend with the sharp image is
// `output`, an inline node of 2 colour taps + 1 depth tap for the consumer's own target.
// Known limit of the half-resolution gather: the silhouette of an in-focus surface in front of a
// defocused one is softened by about one gather texel.
// sceneCoc() is exported for debug views and for consumers that need the same rule.
import {
	Fn, Loop, rtt, uv, vec2, vec3, vec4, float, int, smoothstep, mix, sqrt, cos, sin, min, max,
	interleavedGradientNoise, screenCoordinate, textureSize, perspectiveDepthToViewZ,
} from 'three/tsl';
import { gfxUniforms, cocPixels } from './shared.js';

const GOLDEN_ANGLE = 2.39996323;
/** Normalized view distance ( distance / far ) above which a sample is background. */
const BACKGROUND = 0.999;
/** Smallest circle of confusion (scene-pass pixels) the gather resolves: one texel of its target. */
const MIN_COC = 2;

/** View distance / far of a depth-texture sample (1 = far plane = no geometry). */
export const normalizedDistance = ( depth, cameraNear, cameraFar ) => perspectiveDepthToViewZ( depth, cameraNear, cameraFar ).negate().div( cameraFar );

/** Circle-of-confusion radius (scene-pass pixels) of a normalized distance; background → 0. */
export const sceneCoc = ( nd, cameraFar ) => nd.greaterThan( BACKGROUND ).select( float( 0 ), cocPixels( nd.mul( cameraFar ) ) );

export function graphDof( colorTex, depthTex, { cameraNear, cameraFar, taps = 48, sceneScale = 1, innerScale = 0.5, jitter = true } ) {

	const U = gfxUniforms;
	const maxBlur = U.dofMaxBlur.mul( U.dofAmount ).mul( U.pxScale ); // scene-pass pixels
	const options = { depthBuffer: false, resolutionScale: sceneScale * innerScale }; // rtt() allocates a depth texture otherwise

	// prefilter: colour (bilinear = 2×2 box at half resolution) + normalized view distance
	const pre = rtt( Fn( () => vec4( colorTex.sample( uv() ).rgb, normalizedDistance( depthTex.sample( uv() ).r, cameraNear, cameraFar ) ) )(), null, null, options );
	pre.name = 'GraphDoF.prefilter';

	// Gather: every tap spreads its light over a disc of ITS circle of confusion, so it reaches the
	// centre when that disc is at least as large as the tap's distance, and then with the fraction
	// ( maximum disc / own disc ) of its colour. Summed over the uniformly distributed taps this is
	// an energy-conserving scatter; the sum of the weights is the coverage: how much of the centre
	// is replaced by defocused light (1 inside a uniformly defocused surface, 0 where nothing in
	// reach is defocused). Output: premultiplied scattered colour, coverage in alpha.
	const gather = rtt( Fn( () => {

		const uvN = uv();
		const texel = vec2( 1 ).div( vec2( textureSize( pre ) ) );
		const center = pre.sample( uvN );
		const centerSize = sceneCoc( center.a, cameraFar ).toVar();
		const radius = max( maxBlur, 1e-3 ).toVar();
		const acc = vec3( 0 ).toVar();
		const coverage = float( 0 ).toVar();
		// per-pixel kernel rotation: trades the fixed tap pattern for fine noise
		const phi = jitter ? interleavedGradientNoise( screenCoordinate ).mul( 6.2831853 ) : float( 0 );

		Loop( { start: int( 0 ), end: int( taps ), type: 'int', condition: '<' }, ( { i } ) => {

			const r = sqrt( float( i ).add( 0.5 ).div( taps ) ).mul( radius ); // scene-pass pixels
			const a = float( i ).mul( GOLDEN_ANGLE ).add( phi );
			const s = pre.sample( uvN.add( vec2( cos( a ), sin( a ) ).mul( r ).mul( innerScale ).mul( texel ) ) );
			// as a tap the background is "far and blurred" (it fades the edge of defocused surfaces) …
			const background = s.a.greaterThan( BACKGROUND );
			const own = background.select( radius, cocPixels( s.a.mul( cameraFar ) ) );
			// … but a tap at or behind the centre never blurs over it by more than twice the centre's own
			// blur: in-focus surfaces stay sharp, and background next to background stays untouched
			const sSize = background.or( s.a.greaterThanEqual( center.a ) ).select( min( own, centerSize.mul( 2 ) ), own ).toVar();
			// below MIN_COC the disc is not resolved by the taps: the weight stops growing there,
			// which also fades the effect in between "in focus" and MIN_COC of blur
			const w = smoothstep( r.sub( 0.5 ), r.add( 0.5 ), sSize ).mul( radius.div( max( sSize, MIN_COC ) ).pow2() );
			acc.addAssign( s.rgb.mul( w ) );
			coverage.addAssign( w );

		} );

		return vec4( acc, coverage ).div( taps );

	} )(), null, null, options );
	gather.name = 'GraphDoF.gather';

	// Blend at scene resolution. A pixel in focus keeps its sharp colour under the light scattered
	// over it ( sharp · ( 1 − coverage ) + scattered ); a defocused pixel is the scattered light
	// alone, renormalized, so sharp taps that were rejected around it leave no sharp residue.
	const output = Fn( () => {

		const sharp = colorTex.sample( uv() ).rgb;
		const g = gather.sample( uv() );
		const own = sceneCoc( normalizedDistance( depthTex.sample( uv() ).r, cameraNear, cameraFar ), cameraFar );
		const over = sharp.mul( g.a.oneMinus().max( 0 ) ).add( g.rgb.div( max( g.a, 1 ) ) );
		const blurred = g.rgb.div( max( g.a, 1e-3 ) );
		const k = smoothstep( 0.5, MIN_COC, own ).mul( smoothstep( 0.05, 0.3, g.a ) );
		return vec4( mix( over, blurred, k ), 1 );

	} )();

	return {
		output,
		blurTex: gather,
		passes: [ pre, gather ],
		innerScale,
		setSceneScale( s ) {

			pre.setResolutionScale( s * innerScale );
			gather.setResolutionScale( s * innerScale );

		},
		dispose() {

			pre.dispose();
			gather.dispose();

		},
	};

}
