// Color utilities.
//
// `hclToRgb` is a port of the HCL color space used by ImsvGraphVis (KWColorSpace.cpp /
// ColorSpace.usf, after Sarifuddin & Missaoui 2005). Nodes are colored by walking the
// clustering hierarchy depth-first and spreading hue over [hueMin, hueMax] degrees, with
// fixed chroma and luminance, exactly as AIGVGraphActor::UpdateColors does.
//
// The HCL output is treated as sRGB-encoded (Unreal stored it as FLinearColor but rendered with
// its own tone mapping); we convert to linear for rendering so the on-screen hue matches.

const HCL_GAMMA = 3;
const HCL_Y0 = 100;
const HCL_MAX_L = 0.530454533953517; // == exp(HCL_GAMMA / HCL_Y0) - 0.5

const frac = ( x ) => x - Math.floor( x );

/**
 * @param {number} h hue in [0,1]
 * @param {number} c chroma (paper default 0.5)
 * @param {number} l luminance (paper default 0.5)
 * @returns {[number, number, number]} sRGB in [0,1] (clamped)
 */
export function hclToRgb( h, c, l ) {

	if ( l === 0 ) return [ 0, 0, 0 ];
	let r = 0, g = 0, b = 0;
	l = l * HCL_MAX_L;
	const q = Math.exp( ( 1 - c / ( 2 * l ) ) * ( HCL_GAMMA / HCL_Y0 ) );
	const u = ( 2 * l - c ) / ( 2 * q - 1 );
	const v = c / q;
	const t = Math.tan( ( h + Math.min( frac( 2 * h ) / 4, frac( - 2 * h ) / 8 ) ) * Math.PI * 2 );
	h *= 6;
	if ( h <= 1 ) {

		r = 1; g = t / ( 1 + t );

	} else if ( h <= 2 ) {

		r = ( 1 + t ) / t; g = 1;

	} else if ( h <= 3 ) {

		g = 1; b = 1 + t;

	} else if ( h <= 4 ) {

		g = 1 / ( 1 + t ); b = 1;

	} else if ( h <= 5 ) {

		r = - 1 / t; b = 1;

	} else {

		r = 1; b = - t;

	}

	const clamp = ( x ) => Math.min( 1, Math.max( 0, x ) );
	return [ clamp( r * v + u ), clamp( g * v + u ), clamp( b * v + u ) ];

}

export const srgbToLinear = ( c ) => ( c <= 0.04045 ? c / 12.92 : Math.pow( ( c + 0.055 ) / 1.055, 2.4 ) );
export const linearToSrgb = ( c ) => ( c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow( c, 1 / 2.4 ) - 0.055 );

/** OKLCH (L in [0,1], C ~[0,0.37], h in degrees) -> linear sRGB (unclamped). */
export function oklchToLinear( L, C, hDeg ) {

	const h = ( hDeg * Math.PI ) / 180;
	const a = C * Math.cos( h ), b = C * Math.sin( h );
	const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
	const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
	const s_ = L - 0.0894841775 * a - 1.291485548 * b;
	const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
	return [
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		- 1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		- 0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
	];

}

/**
 * Palettes map a position t in [0,1] (DFS rank of the node in the cluster hierarchy) to a
 * linear-sRGB color. 'paper' reproduces ImsvGraphVis defaults (hue 0..210 deg, C=.5, L=.5).
 */
export const PALETTES = {
	paper: { label: 'Paper (HCL 0–210°)', fn: ( t ) => hclToRgb( ( 0 + 210 * t ) / 360, 0.5, 0.5 ).map( srgbToLinear ) },
	rainbow: { label: 'OKLCH rainbow', fn: ( t ) => oklchToLinear( 0.75, 0.15, 20 + 320 * t ).map( ( x ) => Math.max( 0, x ) ) },
	aurora: { label: 'Aurora (teal→violet)', fn: ( t ) => oklchToLinear( 0.78, 0.14, 150 + 170 * t ).map( ( x ) => Math.max( 0, x ) ) },
	ember: { label: 'Ember (red→gold)', fn: ( t ) => oklchToLinear( 0.72 + 0.12 * t, 0.17, 15 + 75 * t ).map( ( x ) => Math.max( 0, x ) ) },
};

export function paletteColor( name, t ) {

	return ( PALETTES[ name ] || PALETTES.paper ).fn( Math.min( 1, Math.max( 0, t ) ) );

}
