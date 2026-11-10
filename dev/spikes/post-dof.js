// Spike: single-pass "scatter-as-gather" bokeh depth of field for scenes in a dark void.
// Reference code for recipes/post.md (Q4) — the stock DepthOfFieldNode blurs the far field without
// looking at the taps' own circle of confusion, so every in-focus node gets a halo of itself on
// the (far = fully blurred) background. This variant weights each tap by ITS circle of confusion
// (D. Gustafsson, "Bokeh depth of field in a single pass", 2018), which removes the halo and still
// lets blurred nodes spread over the background and over in-focus geometry in front of them.
//
//   const d = graphDof( colorTex, viewZ, { focus, range, maxBlur, taps: 48, resolutionScale: 0.5 } );
//   hdr = d.output;            // vec4 node (sample it at uv() / screenUV), d.blurTex is sampleable
//   d.dispose();
//
//   colorTex   texture node of the lit HDR image (RGBA16F)
//   viewZ      scenePass.getViewZNode()  (negative view-space z; background = -far)
//   focus      uniform, world units along the view direction
//   range      uniform, world units from the focus plane to the maximum blur
//   maxBlur    uniform, maximum blur RADIUS in pixels of the full-resolution image
//
// Passes: 1 prefilter (colour + view distance in alpha, RGBA16F at `resolutionScale`), 1 gather
// (`taps` + 1 bilinear taps, same size); the full-resolution blend with the sharp image is inline
// in the consumer (2 taps). View distance must stay below 65504 (half float).
import { Fn, Loop, rtt, uv, vec2, vec4, float, int, clamp, smoothstep, mix, sqrt, cos, sin, min, abs, uniform, interleavedGradientNoise, screenCoordinate, textureSize, max } from 'three/tsl';

const GOLDEN_ANGLE = 2.39996323;

export function graphDof( colorTex, viewZ, { focus, range, maxBlur, taps = 48, resolutionScale = 0.5, jitter = true } ) {

	// circle of confusion radius in full-res pixels from a view distance
	const coc = ( dist ) => clamp( abs( dist.sub( focus ) ).div( range ), 0, 1 ).mul( maxBlur );

	// prefilter: colour (bilinear = 2×2 box at half res) + linear view distance
	const pre = rtt( Fn( () => vec4( colorTex.sample( uv() ).rgb, min( viewZ.negate(), 65000 ) ) )(), null, null, { resolutionScale, depthBuffer: false } );
	pre.name = 'GraphDoF.prefilter';

	const scale = uniform( resolutionScale ); // full-res pixels → texels of `pre`

	const gather = rtt( Fn( () => {

		const uvN = uv();
		const texel = vec2( 1 ).div( vec2( textureSize( pre ) ) );
		const center = pre.sample( uvN );
		const centerSize = coc( center.a ).toVar();
		const color = center.rgb.toVar();
		const size = centerSize.toVar(); // running average of the contributing blur sizes
		const tot = float( 1 ).toVar();
		// per-pixel kernel rotation: trades the fixed tap pattern for noise (see recipe for the comparison)
		const phi = jitter ? interleavedGradientNoise( screenCoordinate ).mul( 6.2831853 ) : float( 0 );

		Loop( { start: int( 0 ), end: int( taps ), type: 'int', condition: '<' }, ( { i } ) => {

			const r = sqrt( float( i ).add( 0.5 ).div( taps ) ).mul( maxBlur ); // full-res pixels
			const a = float( i ).mul( GOLDEN_ANGLE ).add( phi );
			const s = pre.sample( uvN.add( vec2( cos( a ), sin( a ) ).mul( r ).mul( scale ).mul( texel ) ) );
			const sSize = coc( s.a ).toVar();
			// a tap BEHIND the centre may not blur over it by more than the centre's own blur
			sSize.assign( s.a.greaterThan( center.a ).select( clamp( sSize, 0, centerSize.mul( 2 ) ), sSize ) );
			const m = smoothstep( r.sub( 0.5 ), r.add( 0.5 ), sSize );
			color.addAssign( mix( color.div( tot ), s.rgb, m ) );
			size.addAssign( mix( size.div( tot ), sSize, m ) );
			tot.addAssign( 1 );

		} );

		return vec4( color.div( tot ), size.div( tot ) );

	} )(), null, null, { resolutionScale, depthBuffer: false } ); // rtt() would allocate a depth texture otherwise
	gather.name = 'GraphDoF.gather';

	// full-res blend: keep the sharp image where both the pixel itself and what blurs over it are in focus
	const output = Fn( () => {

		const sharp = colorTex.sample( uv() );
		const blur = gather.sample( uv() );
		const own = coc( min( viewZ.negate(), 65000 ) );
		const k = smoothstep( 0.5, 2.0, max( own, blur.a ) );
		return vec4( mix( sharp.rgb, blur.rgb, k ), 1 );

	} )();

	return {
		output,
		blurTex: gather,
		setResolutionScale( s ) {

			scale.value = s;
			pre.setResolutionScale( s );
			gather.setResolutionScale( s );

		},
		dispose() {

			pre.dispose();
			gather.dispose();

		},
	};

}
