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

// ─── Sequential ramps (metric lens) ──────────────────────────────────────────────────────────
// Ordered colour ramps for mapping a scalar in [0,1] to a node / edge colour on the dark
// backdrop. Each ramp is a list of evenly spaced OKLCH stops. OKLab lightness rises strictly from
// the first stop to the last (so "low → high" also reads in greyscale and for colour-vision
// deficiencies), and the dark end is lifted well above the usual viridis / magma start so that
// low values stay visible on black (≥ 3 : 1 contrast, the WCAG guidance for graphical objects).
// Lightness and hue are interpolated linearly between stops, chroma too, and a colour that falls
// outside the sRGB gamut is brought back by reducing its chroma only, so the lightness that was
// asked for is the lightness that is shown. Sampled into a 256-entry table on first use.
//
//   rampColor( t, name = 'viridis', out = [] ) -> [ r, g, b ] linear sRGB, t clamped to [0,1]
//   rampColors( values, name, out?, reverse = false, intensity = 1 ) -> Float32Array( 4n ) linear
//                                               RGBA (alpha 1); intensity < 1: see emissiveGain()
//   emissiveGain( r, g, b, intensity )          -> factor that holds a bright colour back for an
//                                               emissive, blooming renderer
//   rampList()                                  -> [ { id, label, css } ] for a picker / legend
//   rampCss( name, { steps = 12, reverse, angle = 90 } ) -> CSS linear-gradient( … ) string
//   RAMPS                                       { id: { label, stops: [ [ L, C, hueDeg ], … ] } }
//   linearToOklab( r, g, b )                    -> [ L, a, b ]

export const RAMPS = {
	viridis: { label: 'Viridis (violet → green → yellow)', stops: [ [ 0.50, 0.17, 302 ], [ 0.57, 0.13, 258 ], [ 0.66, 0.11, 205 ], [ 0.76, 0.15, 160 ], [ 0.86, 0.18, 128 ], [ 0.95, 0.17, 104 ] ] },
	magma: { label: 'Magma (purple → rose → peach)', stops: [ [ 0.50, 0.19, 305 ], [ 0.57, 0.21, 338 ], [ 0.66, 0.20, 372 ], [ 0.76, 0.16, 405 ], [ 0.87, 0.13, 435 ], [ 0.96, 0.08, 462 ] ] },
	ice: { label: 'Ice (deep blue → cyan → white)', stops: [ [ 0.50, 0.19, 268 ], [ 0.58, 0.17, 252 ], [ 0.69, 0.14, 234 ], [ 0.80, 0.12, 214 ], [ 0.90, 0.08, 200 ], [ 0.98, 0.02, 195 ] ] },
	ember: { label: 'Ember (red → orange → gold)', stops: [ [ 0.50, 0.19, 22 ], [ 0.58, 0.19, 36 ], [ 0.68, 0.17, 52 ], [ 0.78, 0.16, 70 ], [ 0.88, 0.16, 90 ], [ 0.96, 0.11, 104 ] ] },
	mint: { label: 'Mint (teal → green → pale lime)', stops: [ [ 0.50, 0.09, 222 ], [ 0.58, 0.11, 198 ], [ 0.68, 0.13, 176 ], [ 0.78, 0.15, 158 ], [ 0.88, 0.16, 140 ], [ 0.96, 0.12, 122 ] ] },
};

const RAMP_SIZE = 256;
const rampTables = new Map();
const tonedTables = new Map(); // 'name|intensity' -> table held back for emissive display

/**
 * Gain for showing a linear-sRGB colour through emissive, blooming shaders: 1 / ( 1 + c Y ) with
 * Y the colour's luminance and c = 1 / intensity − 1. A soft shoulder: dark colours pass almost
 * unchanged (they must stay visible on the backdrop), bright ones are held back, white keeping
 * `intensity` of its luminance. The gain multiplies all three channels, so hue and saturation
 * stay, and Y / ( 1 + c Y ) rises with Y, so an ordered ramp stays ordered. intensity 1 = off.
 */
export function emissiveGain( r, g, b, intensity = 1 ) {

	if ( ! ( intensity < 1 ) ) return 1;
	const c = 1 / Math.max( intensity, 1e-3 ) - 1;
	return 1 / ( 1 + c * Math.max( 0, 0.2126 * r + 0.7152 * g + 0.0722 * b ) );

}

/** Linear sRGB -> OKLab [ L, a, b ] (Ottosson 2020). */
export function linearToOklab( r, g, b ) {

	const l = Math.cbrt( 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b );
	const m = Math.cbrt( 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b );
	const s = Math.cbrt( 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b );
	return [
		0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
		1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
		0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
	];

}

function inGamut( rgb ) {

	const e = 1e-4;
	return rgb[ 0 ] >= - e && rgb[ 1 ] >= - e && rgb[ 2 ] >= - e && rgb[ 0 ] <= 1 + e && rgb[ 1 ] <= 1 + e && rgb[ 2 ] <= 1 + e;

}

function rampTable( name, intensity = 1 ) {

	const key = RAMPS[ name ] ? name : 'viridis';
	if ( intensity < 1 ) {

		const tonedKey = key + '|' + intensity;
		let toned = tonedTables.get( tonedKey );
		if ( ! toned ) {

			const full = rampTable( key );
			toned = new Float32Array( full.length );
			for ( let o = 0; o < full.length; o += 3 ) {

				const gain = emissiveGain( full[ o ], full[ o + 1 ], full[ o + 2 ], intensity );
				toned[ o ] = gain * full[ o ]; toned[ o + 1 ] = gain * full[ o + 1 ]; toned[ o + 2 ] = gain * full[ o + 2 ];

			}

			if ( tonedTables.size >= 16 ) tonedTables.clear(); // a slider leaves many behind
			tonedTables.set( tonedKey, toned );

		}

		return toned;

	}

	let table = rampTables.get( key );
	if ( table ) return table;
	const stops = RAMPS[ key ].stops;
	table = new Float32Array( 3 * RAMP_SIZE );
	for ( let k = 0; k < RAMP_SIZE; k ++ ) {

		const x = ( k / ( RAMP_SIZE - 1 ) ) * ( stops.length - 1 );
		const i = Math.min( stops.length - 2, Math.floor( x ) ), f = x - i;
		const a = stops[ i ], b = stops[ i + 1 ];
		const L = a[ 0 ] + ( b[ 0 ] - a[ 0 ] ) * f, C = a[ 1 ] + ( b[ 1 ] - a[ 1 ] ) * f, h = a[ 2 ] + ( b[ 2 ] - a[ 2 ] ) * f;
		let rgb = oklchToLinear( L, C, h );
		if ( ! inGamut( rgb ) ) {

			// largest chroma that fits sRGB at this lightness and hue
			let lo = 0, hi = C;
			for ( let it = 0; it < 24; it ++ ) {

				const mid = ( lo + hi ) / 2;
				if ( inGamut( oklchToLinear( L, mid, h ) ) ) lo = mid; else hi = mid;

			}

			rgb = oklchToLinear( L, lo, h );

		}

		for ( let c = 0; c < 3; c ++ ) table[ 3 * k + c ] = Math.min( 1, Math.max( 0, rgb[ c ] ) );

	}

	rampTables.set( key, table );
	return table;

}

/**
 * Colour of a sequential ramp at t in [0,1] (clamped; NaN reads as 0), as linear sRGB.
 * Unknown names fall back to 'viridis'.
 */
export function rampColor( t, name = 'viridis', out = [ 0, 0, 0 ] ) {

	const table = rampTable( name );
	const x = ( t > 0 ? ( t < 1 ? t : 1 ) : 0 ) * ( RAMP_SIZE - 1 );
	const i = Math.min( RAMP_SIZE - 2, Math.floor( x ) ), f = x - i, o = 3 * i;
	out[ 0 ] = table[ o ] + ( table[ o + 3 ] - table[ o ] ) * f;
	out[ 1 ] = table[ o + 1 ] + ( table[ o + 4 ] - table[ o + 1 ] ) * f;
	out[ 2 ] = table[ o + 2 ] + ( table[ o + 5 ] - table[ o + 2 ] ) * f;
	return out;

}

/**
 * Ramp colours of an array of values in [0,1]: linear RGBA (alpha 1), 4 floats per value.
 * `intensity` below 1 holds the bright end of the ramp back with emissiveGain() (the same hues,
 * the same order): what an emissive, blooming renderer needs so that high values do not burn
 * out (see the metric lens).
 */
export function rampColors( values, name = 'viridis', out = new Float32Array( 4 * values.length ), reverse = false, intensity = 1 ) {

	const table = rampTable( name, intensity );
	for ( let k = 0, n = values.length; k < n; k ++ ) {

		let t = values[ k ];
		t = t > 0 ? ( t < 1 ? t : 1 ) : 0;
		const x = ( reverse ? 1 - t : t ) * ( RAMP_SIZE - 1 );
		const i = x < RAMP_SIZE - 2 ? Math.floor( x ) : RAMP_SIZE - 2, f = x - i, o = 3 * i;
		out[ 4 * k ] = table[ o ] + ( table[ o + 3 ] - table[ o ] ) * f;
		out[ 4 * k + 1 ] = table[ o + 1 ] + ( table[ o + 4 ] - table[ o + 1 ] ) * f;
		out[ 4 * k + 2 ] = table[ o + 2 ] + ( table[ o + 5 ] - table[ o + 2 ] ) * f;
		out[ 4 * k + 3 ] = 1;

	}

	return out;

}

const hex2 = ( c ) => Math.round( 255 * Math.min( 1, Math.max( 0, linearToSrgb( c ) ) ) ).toString( 16 ).padStart( 2, '0' );

/** CSS gradient of a ramp (sRGB hex stops), e.g. for a legend bar or a picker swatch. */
export function rampCss( name = 'viridis', { steps = 12, reverse = false, angle = 90 } = {} ) {

	const parts = [], c = [ 0, 0, 0 ];
	for ( let k = 0; k < steps; k ++ ) {

		const t = k / ( steps - 1 );
		rampColor( reverse ? 1 - t : t, name, c );
		parts.push( `#${hex2( c[ 0 ] )}${hex2( c[ 1 ] )}${hex2( c[ 2 ] )} ${( 100 * t ).toFixed( 1 )}%` );

	}

	return `linear-gradient(${angle}deg, ${parts.join( ', ' )})`;

}

/** The ramps a UI can offer: [ { id, label, css } ]. */
export function rampList() {

	return Object.keys( RAMPS ).map( ( id ) => ( { id, label: RAMPS[ id ].label, css: rampCss( id ) } ) );

}

// Distinct hues for the first few labels of a categorical metric (connected components, sorted
// by size); everything beyond them is a neutral grey. Linear sRGB.
const CATEGORY_HUES = [ 250, 70, 150, 340, 25, 200, 300, 110 ];

/** Colour of category k (0 = first / largest): eight distinct hues, then grey. Linear sRGB. */
export function categoryColor( k, out = [ 0, 0, 0 ] ) {

	const rgb = k >= 0 && k < CATEGORY_HUES.length ? oklchToLinear( 0.78, 0.13, CATEGORY_HUES[ k ] ) : oklchToLinear( 0.6, 0, 0 );
	for ( let c = 0; c < 3; c ++ ) out[ c ] = Math.min( 1, Math.max( 0, rgb[ c ] ) );
	return out;

}
