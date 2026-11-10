// Shared render state: the one place where the render pipeline (RenderGraph), the environment
// (lights, sky) and the materials (nodes, edges, labels, shells) meet. Everything here is a
// module-level singleton: one renderer / one scene pass per page.
//
//   gfxUniforms   TSL uniforms read by materials and post effects (see the table below)
//   gfxState      plain JS mirror of the active pipeline configuration, for decisions that select
//                 a material variant (alpha-to-coverage, motion vectors, …). Whoever changes it
//                 dispatches app.events 'gfx' (detail: gfxState) so renderers can swap materials.
//   TSL helpers   motionVector(), velocityOutput(), fogTransmittance(), applyFog(), cocPixels()
//
// Units and conventions
//   · "CSS px" quantities in scene-pass shaders are multiplied by gfxUniforms.pxScale
//     (= renderer pixel ratio × scene-pass resolution scale), so on-screen sizes do not change
//     with DPR or dynamic resolution.
//   · Motion vectors follow three's VelocityNode / TRAANode convention: NDC now − NDC previous,
//     computed with UNJITTERED view-projection matrices (TRAA jitters the camera only inside
//     RenderPipeline.render(); the App updates viewProj before and commits it after the render).
//   · The scene pass velocity attachment (when gfxState.velocity) is RGBA with normal blending per
//     attachment: write vec4( v.xy, 0, 1 ) to set a pixel's velocity, vec4( 0 ) to leave it alone
//     (see velocityOutput / skipVelocity). Materials list `output` in their mrtNode as well, so the
//     same material also works in passes without MRT.
import { uniform, vec4, float, exp, max, abs, clamp, mrt, output } from 'three/tsl';
import { Matrix4, Vector3, Color } from 'three/webgpu';

export const gfxUniforms = {
	// ── resolution ────────────────────────────────────────────────────────────────────────
	pxScale: uniform( 1 ), // scene-pass pixels per CSS pixel
	// ── motion vectors ────────────────────────────────────────────────────────────────────
	viewProj: uniform( new Matrix4() ), // unjittered projection · view of this frame
	prevViewProj: uniform( new Matrix4() ), // … of the previous rendered frame
	// ── lighting (set by the Environment module; cheap analytic consumers, e.g. thin strands) ─
	keyDir: uniform( new Vector3( - 0.45, 0.65, 0.62 ).normalize() ), // world space, toward the light
	keyColor: uniform( new Color( 3.0, 2.85, 2.6 ) ), // linear RGB × intensity
	ambientSky: uniform( new Color( 0.10, 0.13, 0.20 ) ), // hemisphere irradiance, up
	ambientGround: uniform( new Color( 0.03, 0.03, 0.045 ) ), // … down
	// ── distance fog, evaluated in the materials (transparent strands have no depth) ───────
	fogColor: uniform( new Color( 0.012, 0.014, 0.022 ) ),
	fogDensity: uniform( 0 ), // 1 / world units; 0 = off
	fogStart: uniform( 0 ), // world units from the camera
	// ── depth of field (post effect + analytic defocus of blended strands) ─────────────────
	dofAmount: uniform( 0 ), // 0 = off … 1
	dofFocus: uniform( 400 ), // view-space distance of the focus plane, world units
	dofRange: uniform( 200 ), // distance from the focus plane to full blur, world units
	dofMaxBlur: uniform( 10 ), // circle-of-confusion radius at full blur, CSS px
};

export const gfxState = {
	tier: 'high', // resolved quality tier: 'low' | 'medium' | 'high' | 'ultra'
	samples: 4, // MSAA samples of the scene pass (1 = none). > 1 ⇒ alpha-to-coverage silhouettes
	velocity: false, // scene pass has a velocity attachment (TRAA): write motion vectors
	shadows: true, // key light casts shadows
	shadowMapSize: 2048,
	lit: true, // false: cheapest shading (no IBL / shadows), phones
	renderScale: 1, // scene-pass resolution scale (dynamic resolution)
};

const _vp = /*@__PURE__*/ new Matrix4();
let _hasPrev = false;

/** Call before RenderPipeline.render(): refreshes gfxUniforms.viewProj from the (unjittered) camera. */
export function updateMotionUniforms( camera ) {

	camera.updateMatrixWorld();
	_vp.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
	gfxUniforms.viewProj.value.copy( _vp );
	if ( ! _hasPrev ) commitMotionUniforms();

}

/** Call after RenderPipeline.render(): this frame's matrix becomes the previous one. */
export function commitMotionUniforms() {

	gfxUniforms.prevViewProj.value.copy( gfxUniforms.viewProj.value );
	_hasPrev = true;

}

/** Forget the history (camera cut, new graph): the next frame reports zero camera motion. */
export function resetMotionUniforms() {

	_hasPrev = false;

}

// ─── TSL helpers ─────────────────────────────────────────────────────────────────────────────

/** NDC motion (vec2) of a surface point: world position now and at the previous frame. */
export const motionVector = ( worldNow, worldPrev ) => {

	const a = gfxUniforms.viewProj.mul( vec4( worldNow, 1 ) );
	const b = gfxUniforms.prevViewProj.mul( vec4( worldPrev, 1 ) );
	return a.xy.div( a.w ).sub( b.xy.div( b.w ) );

};

/** material.mrtNode for a material that writes its own motion vector (vec2 NDC motion). */
export const velocityOutput = ( motion ) => mrt( { output, velocity: vec4( motion, 0, 1 ) } );

/** material.mrtNode for blended / additive materials that must not touch the velocity attachment. */
export const skipVelocity = () => mrt( { output, velocity: vec4( 0 ) } );

/** Fraction of light that survives the fog over `viewDistance` (float node, world units). */
export const fogTransmittance = ( viewDistance ) => {

	const d = max( float( viewDistance ).sub( gfxUniforms.fogStart ), 0 ).mul( gfxUniforms.fogDensity );
	return exp( d.mul( d ).negate() );

};

/** Fogged color for opaque / normal-blended surfaces. Additive materials use rgb · fogTransmittance(). */
export const applyFog = ( rgb, viewDistance ) => {

	const t = fogTransmittance( viewDistance );
	return rgb.mul( t ).add( gfxUniforms.fogColor.mul( t.oneMinus() ) );

};

/** Circle-of-confusion radius in scene-pass pixels at `viewDistance` (0 when DoF is off). */
export const cocPixels = ( viewDistance ) => {

	const U = gfxUniforms;
	const k = clamp( abs( float( viewDistance ).sub( U.dofFocus ) ).div( max( U.dofRange, 1e-3 ) ), 0, 1 );
	return k.mul( U.dofMaxBlur ).mul( U.dofAmount ).mul( U.pxScale );

};
