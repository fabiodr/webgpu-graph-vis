// Shading helpers of the edge renderer (src/render/EdgeRenderer.js). The camera-generic impostor
// helpers and applySurface() that the lit tubes use live in gfx/impostor.js.
//
// ─── Strand shading ─────────────────────────────────────────────────────────────────────────
//    lmsRoot( rgb ), lmsRootToLinear( q )   linear sRGB ↔ cube-root LMS. OKLab is a linear map of
//                                the latter, so a linear blend of lmsRoot values IS the OKLab blend
//    gradientColor( a, b, t, restore )   OKLab blend of two linear colours whose chroma is pushed
//                                back toward the mix of the two end chromas (no grey middle
//                                between hues that lie far apart)
//    strandLight( T, V, o )      analytic light on a strand much thinner than a pixel, no texture
//                                taps: Lambert averaged over the visible half of a cylinder, a
//                                Kajiya-Kay highlight around the tangent, hemisphere ambient; all
//                                from gfxUniforms.keyDir / keyColor / ambientSky / ambientGround.
//                                Returns { diffuse, specular, front } (vec3 radiance factors).
//    tubeFrame( viewPos, tangent, signedPx, halfPx, radius )   per-fragment cylinder frame of a
//                                screen-space ribbon: { V, T, front, normal, hit, edge }
//    flowPulse( arc, seed, o )   travelling energy pulses along the arc length
import { float, vec2, vec3, select, clamp, max, abs, sqrt, pow, dot, cross, normalize, mix, exp, fract, length } from 'three/tsl';
import { gfxUniforms } from './shared.js';
import { viewRay } from './impostor.js';

// ── colour ─────────────────────────────────────────────────────────────────────────────────

/** Linear sRGB → cube-root LMS (the non-linearity of OKLab). */
export const lmsRoot = ( c ) => {

	const r = max( c.x, 0 ), g = max( c.y, 0 ), b = max( c.z, 0 );
	const l = r.mul( 0.4122214708 ).add( g.mul( 0.5363325363 ) ).add( b.mul( 0.0514459929 ) );
	const m = r.mul( 0.2119034982 ).add( g.mul( 0.6806995451 ) ).add( b.mul( 0.1073969566 ) );
	const s = r.mul( 0.0883024619 ).add( g.mul( 0.2817188376 ) ).add( b.mul( 0.6299787005 ) );
	return pow( max( vec3( l, m, s ), vec3( 1e-9 ) ), vec3( 1 / 3 ) );

};

/** Cube-root LMS → linear sRGB. */
export const lmsRootToLinear = ( q ) => {

	const c = q.mul( q ).mul( q );
	return max( vec3(
		dot( c, vec3( 4.0767416621, - 3.3077115913, 0.2309699292 ) ),
		dot( c, vec3( - 1.2684380046, 2.6097574011, - 0.3413193965 ) ),
		dot( c, vec3( - 0.0041960863, - 0.7034186147, 1.7076147010 ) ),
	), vec3( 0 ) );

};

/** The a and b axes of OKLab from cube-root LMS. */
const oklabAB = ( q ) => vec2(
	dot( q, vec3( 1.9779984951, - 2.4285922050, 0.4505937099 ) ),
	dot( q, vec3( 0.0259040371, 0.7827717662, - 0.8086757660 ) ),
);

/** Largest factor by which gradientColor() raises the chroma of a blend. */
export const CHROMA_GAIN_MAX = 2.5;

/**
 * Gradient colour between two linear sRGB colours at t: the OKLab blend, with its chroma moved by
 * `restore` (0..1) toward the mix of the two end chromas. A straight line through OKLab between
 * two saturated hues passes close to the grey axis (orange → cyan turns grey-white in the middle);
 * scaling ( a, b ) keeps the hue and the lightness of the blend and gives the saturation back. The
 * factor is limited to CHROMA_GAIN_MAX (exactly opposite hues have no hue in the middle to amplify)
 * and never lowers the chroma.
 */
export const gradientColor = ( colorA, colorB, t, restore = float( 1 ) ) => {

	const qA = lmsRoot( colorA ).toVar(), qB = lmsRoot( colorB ).toVar();
	const q = mix( qA, qB, t ).toVar();
	const ab = oklabAB( q ).toVar();
	const wanted = mix( length( oklabAB( qA ) ), length( oklabAB( qB ) ), t );
	const gain = clamp( wanted.div( max( length( ab ), 1e-4 ) ), 1, CHROMA_GAIN_MAX );
	const d = ab.mul( mix( float( 1 ), gain, restore ).sub( 1 ) ); // what is added to ( a, b ); L stays
	// OKLab → cube-root LMS is linear: only the change has to be mapped back
	return lmsRootToLinear( q.add( vec3(
		d.x.mul( 0.3963377774 ).add( d.y.mul( 0.2158037573 ) ),
		d.x.mul( - 0.1055613458 ).add( d.y.mul( - 0.0638541728 ) ),
		d.x.mul( - 0.0894841775 ).add( d.y.mul( - 1.2914855480 ) ),
	) ) );

};

// ── thin-strand light model ──────────────────────────────────────────────────────────────────

/**
 * Analytic light on a thin strand: what a lit cylinder looks like once it is narrower than a
 * pixel and only its average over the visible half matters. No texture taps, no shadow taps.
 *
 *  diffuse   Lambert integrated over the visible normals N( θ ) = front cos θ + side sin θ,
 *            weighted by projected width: E = sin( T, L ) · ( sin φ + ( π − φ ) cos φ ) / 4 with φ
 *            the azimuth of L around T measured from the viewer side (π/4 for light from the
 *            viewer, 0 from behind). The bracket is evaluated as the fit 0.4292 (1 + c) +
 *            0.5708 (1 + c)², c = cos φ (within 5 %, exact at c = −1, 0, 1), which saves the acos.
 *            Returned as radiance per unit albedo: keyColor · E / π + hemisphere ambient / π.
 *  specular  Kajiya-Kay: a lobe sin( T, H )^n around the cone of normals perpendicular to the
 *            tangent, scaled by sin( T, L ) and by how much of the lit half the viewer sees.
 *            `shift` slides the lobe along the strand (per-strand variation, as hair shaders do).
 *
 * @param {Node<vec3>} T unit tangent (world space)
 * @param {Node<vec3>} V unit vector to the viewer (world space)
 * @param {object} [o] { gloss: lobe exponent n (float node), shift: float node added to T·H }
 * @returns {{ diffuse: Node<vec3>, specular: Node<vec3>, front: Node<vec3> }}
 */
export function strandLight( T, V, { gloss = float( 40 ), shift = float( 0 ) } = {} ) {

	const U = gfxUniforms;
	const L = U.keyDir;
	const f0 = V.sub( T.mul( dot( T, V ) ) ).toVar();
	const front = f0.div( max( sqrt( dot( f0, f0 ) ), 1e-4 ) ).toVar(); // ⟂ T, toward the viewer
	const TdotL = dot( T, L ).toVar();
	const sinTL = sqrt( max( TdotL.mul( TdotL ).oneMinus(), 0 ) ).toVar();
	const Lp = L.sub( T.mul( TdotL ) ).div( max( sinTL, 1e-4 ) );
	const c1 = clamp( dot( Lp, front ), - 1, 1 ).add( 1 ).toVar(); // 1 + cos φ
	const E = sinTL.mul( c1.mul( 0.4292 ).add( c1.mul( c1 ).mul( 0.5708 ) ) ).mul( 0.25 );
	// hemisphere ambient at the average visible normal ( ≈ front · π/4 )
	const ambient = mix( U.ambientGround, U.ambientSky, front.y.mul( 0.39 ).add( 0.5 ) );
	const diffuse = U.keyColor.mul( E ).add( ambient ).mul( 1 / Math.PI );

	const H = normalize( L.add( V ) );
	const TdotH = clamp( dot( T, H ).add( shift ), - 1, 1 );
	const sinTH = sqrt( max( TdotH.mul( TdotH ).oneMinus(), 0 ) );
	const lobe = pow( sinTH, gloss ).mul( gloss.add( 2 ) ).mul( 1 / ( 8 * Math.PI ) );
	const specular = U.keyColor.mul( lobe.mul( sinTL ).mul( c1.mul( 0.5 ) ) );
	return { diffuse, specular, front };

}

// ── tube frame ──────────────────────────────────────────────────────────────────────────────

/**
 * Per-fragment cylinder frame of a ribbon that was expanded in screen space around a curve
 * (lighting recipe, section 3.2, tube mode). All vectors in view space.
 * @param {Node<vec3>} viewPos ribbon point (varying)
 * @param {Node<vec3>} tangent curve tangent (varying, need not be unit)
 * @param {Node<float>} signedPx signed distance from the centre line in pixels
 * @param {Node<float>} halfPx half width in pixels
 * @param {Node<float>} radius world radius as drawn
 * @returns {{ V, T, front, normal, hit, edge }} V unit vector to the camera, T unit tangent,
 *   front ⟂ T toward the viewer, normal and hit of the cylinder surface, edge = 1 px coverage ramp
 */
export function tubeFrame( viewPos, tangent, signedPx, halfPx, radius ) {

	const V = viewRay( viewPos ).negate().toVar( 'strV' );
	const T = normalize( tangent ).toVar( 'strT' );
	const f0 = V.sub( T.mul( dot( T, V ) ) );
	const front = select( dot( f0, f0 ).greaterThan( 1e-8 ), normalize( f0 ), V ).toVar( 'strFront' );
	const sideways = cross( front, T );
	const across = clamp( signedPx.div( max( halfPx, 1e-4 ) ), - 1, 1 );
	const bulge = sqrt( max( across.mul( across ).oneMinus(), 0 ) );
	const normal = normalize( front.mul( bulge ).add( sideways.mul( across ) ) ).toVar( 'strNormal' );
	// the tube surface is radius · √( 1 − v² ) in front of the ribbon plane
	const hit = viewPos.add( front.mul( radius.mul( bulge ) ) ).toVar( 'strHit' );
	const edge = clamp( halfPx.add( 0.5 ).sub( abs( signedPx ) ), 0, 1 );
	return { V, T, front, normal, hit, edge };

}

// ── flow ────────────────────────────────────────────────────────────────────────────────────

/**
 * Energy pulses that travel along a curve at an even speed: a comet with a steep front and an
 * exponential tail, repeated every `spacing` world units. Returns a float in [0, ~1].
 * @param {Node<float>} arc arc length from the source (world units)
 * @param {Node<float>} phase per-edge offset in [0,1) (0 = all edges in step)
 * @param {object} o { time (s), speed (world units / s), spacing (world units), tail (decay rate
 *                     per period, larger = shorter tail) }
 */
export function flowPulse( arc, phase, { time, speed, spacing, tail = float( 5 ) } ) {

	// f = 0 at the pulse head, grows toward the source behind it, wraps to 1 just ahead of the next head
	const f = fract( time.mul( speed ).sub( arc ).div( max( spacing, 1e-6 ) ).add( phase ) );
	return exp( f.mul( tail ).negate() ).add( exp( f.oneMinus().mul( - 40 ) ) ).min( 1 );

}
