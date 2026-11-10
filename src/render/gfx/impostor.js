// Impostor surfaces for three's own lit node materials (MeshStandardNodeMaterial,
// MeshPhysicalNodeMaterial, …): the geometry is a camera-facing quad or ribbon built in the vertex
// stage, the real surface (sphere, tube) is reconstructed per fragment. three's lighting only has to
// be told four things per fragment: where the surface is, what its normal is, what its depth is and
// where to look up the shadow. applySurface() routes them into the supported r187 hooks:
//
//   vertex    material.positionNode   WORLD-space corner of the quad / ribbon, built from storage
//                                     buffers and the CAMERA NODES (cameraViewMatrix, …). The shadow
//                                     pass copies positionNode (or castShadowPositionNode) into its
//                                     override material and renders with the light's camera, so the
//                                     same node graph faces the light there. Never use
//                                     material.vertexNode: the shadow pass ignores it (no shadows).
//   fragment  material.normalNode     view-space normal of the surface
//             material.depthNode      depth of the surface point for whatever camera renders
//             material.contextNode    overrideNodes( positionView, positionViewDirection,
//                                     positionWorld ) in the FRAGMENT stage only, so lights, the
//                                     specular view vector and the environment lookup see the
//                                     surface point instead of the flat quad
//             material.receivedShadowPositionNode   world-space surface point
//
// Everything here is camera generic: valid in the main pass (perspective) and in the shadow pass of
// a DirectionalLight (orthographic). Pixel quantities are in pixels of the render target that is
// being drawn (scene pass: multiply CSS px by gfxUniforms.pxScale; shadow pass: shadow-map texels).
//
// ─── Exports ─────────────────────────────────────────────────────────────────────────────────
//  isOrthographic()            bool node, true while an orthographic camera renders
//  pixelsPerUnit( viewZ )      float node, target pixels per world unit at view-space depth viewZ
//  viewToDepth( p )            float node, fragment depth of a view-space point (clamped to [0,1])
//  viewToWorld( p )            vec3 node, view space → world space
//  viewRay( p )                vec3 node, unit direction from the camera through view-space point p
//  inFragment( orig, value )   override callback that applies in the fragment stage only
//  applySurface( material, { hit, normal, viewDir, tangent? } ) → world-space hit point (vec3 node)
//  sphereBillboard( center, R )          vertex stage: frame of the quad that bounds a sphere
//  raySphere( quadPoint, center, R )     fragment stage: { hit, normal, viewDir }
//  discCoverage( q )                     fragment stage: analytic coverage of the unit disc |q| ≤ 1
//  SurfaceStandardMaterial, SurfacePhysicalMaterial   three's lit materials with a `finish` hook:
//      material.finish = ( rgb ) => rgb'   applied to the final lit colour (emissive included)
//      before it is written, e.g. fog and state emphasis. Unlike material.outputNode it also
//      reaches the `output` member of material.mrtNode in every kind of pass.
//  quadGeometry()              4 corners in [-1, 1]², counter-clockwise
import * as THREE from 'three/webgpu';
import {
	float, vec3, vec4, select, clamp, max, abs, sqrt, dot, cross, normalize, length, fwidth,
	cameraProjectionMatrix, cameraWorldMatrix, viewport, positionView, positionViewDirection,
	positionWorld, tangentView, bitangentView, overrideNodes,
} from 'three/tsl';

// ── camera-generic helpers ───────────────────────────────────────────────────────────────────

/** true while an orthographic camera renders (the shadow pass of a DirectionalLight). */
export const isOrthographic = () => cameraProjectionMatrix.element( 3 ).element( 3 ).greaterThan( 0.5 );

/** Pixels of the current render target per world unit at view-space depth `viewZ` (negative in front). */
export const pixelsPerUnit = ( viewZ ) => cameraProjectionMatrix.element( 1 ).element( 1 ).mul( viewport.w ).mul( 0.5 )
	.div( select( isOrthographic(), float( 1 ), max( viewZ.negate(), 1e-4 ) ) );

/** View-space point → fragment depth, for whatever camera is rendering (perspective / orthographic). */
export const viewToDepth = ( p ) => {

	const clip = cameraProjectionMatrix.mul( vec4( p, 1 ) );
	return clip.z.div( clip.w ).clamp( 0, 1 );

};

/** View space → world space (the camera's world matrix; meshes that use this have an identity transform). */
export const viewToWorld = ( p ) => cameraWorldMatrix.mul( vec4( p, 1 ) ).xyz;

/** Unit direction from the camera through view-space point p (parallel rays for orthographic cameras). */
export const viewRay = ( p ) => select( isOrthographic(), vec3( 0, 0, - 1 ), normalize( p ) );

/** Override callback for overrideNodes(): `value` in the fragment stage, the original node elsewhere. */
export const inFragment = ( original, value ) => ( builder ) => ( builder.shaderStage === 'fragment' ? value : original );

// ── routing a per-fragment surface into three's lighting ─────────────────────────────────────

/**
 * @param {NodeMaterial} material a lit node material (lights = true)
 * @param {object} s { hit: view-space surface point, normal: view-space unit normal,
 *                     viewDir: unit vector from the surface to the camera (view space),
 *                     tangent: optional view-space unit tangent (only needed for anisotropy) }
 * @return {Node<vec3>} the world-space surface point (also material.receivedShadowPositionNode)
 */
export function applySurface( material, s ) {

	const hitWorld = viewToWorld( s.hit ).toVar( 'impHitWorld' );
	material.normalNode = s.normal;
	material.depthNode = viewToDepth( s.hit );
	material.receivedShadowPositionNode = hitWorld;
	// fragment stage only: the vertex stage must keep the real positionView for the clip position
	const list = [
		[ positionView, inFragment( positionView, s.hit ) ],
		[ positionViewDirection, inFragment( positionViewDirection, s.viewDir ) ],
		[ positionWorld, inFragment( positionWorld, hitWorld ) ],
	];
	// anisotropy reads a tangent frame; three's default derives it from uv derivatives (there is no uv)
	if ( s.tangent ) list.push(
		[ tangentView, inFragment( tangentView, s.tangent ) ],
		[ bitangentView, inFragment( bitangentView, cross( s.normal, s.tangent ) ) ],
	);
	material.contextNode = overrideNodes( list );
	return hitWorld;

}

// ── spheres ──────────────────────────────────────────────────────────────────────────────────

/**
 * Vertex stage. Frame of the camera-facing quad that bounds a sphere exactly.
 * Perspective: the quad lies in the plane through the centre perpendicular to the eye direction;
 * the silhouette cone cuts that plane in a circle of radius h = R·d/√(d²−R²). Orthographic: the
 * plane is perpendicular to the view axis and h = R.
 * A corner is center + ( right·q.x + up·q.y )·h with q in units of the silhouette radius.
 *
 * @param {Node<vec3>} center view-space centre
 * @param {Node<float>} R world radius
 * @return {{ h, right, up, d2, inside }} h: silhouette radius in the quad plane (world units),
 *   inside: the (perspective) camera is inside the sphere, the quad should be collapsed
 */
export function sphereBillboard( center, R ) {

	const ortho = isOrthographic();
	const d2 = dot( center, center );
	const d = sqrt( d2 );
	const r2 = R.mul( R );
	const h = select( ortho, R, R.mul( d ).div( sqrt( max( d2.sub( r2 ), 1e-6 ) ) ) );
	const e = select( ortho, vec3( 0, 0, 1 ), center.negate().div( max( d, 1e-6 ) ) ).toVar();
	const upRef = select( abs( e.y ).greaterThan( 0.999 ), vec3( 1, 0, 0 ), vec3( 0, 1, 0 ) );
	const right = normalize( cross( upRef, e ) ).toVar();
	const up = cross( e, right );
	const inside = ortho.not().and( d2.lessThan( r2.mul( 1.0001 ) ) );
	return { h, right, up, d2, inside };

}

/**
 * Fragment stage. Ray–sphere intersection in view space (closest-approach form: stable far from
 * the camera). Inside the silhouette the ray always hits; outside the result is clamped to the
 * closest point of the silhouette circle, so the surface stays finite in the coverage ramp.
 *
 * @param {Node<vec3>} quadPoint view-space point of the billboard under this fragment
 * @param {Node<vec3>} center view-space centre
 * @param {Node<float>} R world radius
 * @return {{ hit, normal, viewDir }} view-space surface point, unit normal, unit vector to the camera
 */
export function raySphere( quadPoint, center, R ) {

	const ro = select( isOrthographic(), vec3( quadPoint.xy, 0 ), vec3( 0 ) );
	const rd = viewRay( quadPoint ).toVar();
	const oc = center.sub( ro );
	const b = dot( rd, oc );
	const m = oc.sub( rd.mul( b ) );
	const disc = max( R.mul( R ).sub( dot( m, m ) ), 0 );
	const hit = ro.add( rd.mul( b.sub( sqrt( disc ) ) ) );
	const normal = normalize( hit.sub( center ) );
	return { hit, normal, viewDir: rd.negate() };

}

/** Fragment stage. Analytic coverage (1 px ramp) of the unit disc for quad coordinates q. */
export const discCoverage = ( q ) => {

	const ql = length( q );
	return clamp( float( 1 ).sub( ql ).div( max( fwidth( ql ), 1e-4 ) ).add( 0.5 ), 0, 1 );

};

// ── lit materials with a hook on the final colour ────────────────────────────────────────────

const withFinish = ( Base ) => class extends Base {

	constructor( parameters ) {

		super( parameters );
		/** ( rgb: Node<vec3> ) => Node<vec3>, applied to the lit result (see the header). */
		this.finish = null;
		/** Distinguishes programs of materials that differ only in `finish`. */
		this.finishKey = '';

	}

	setupOutput( builder, outputNode ) {

		const out = super.setupOutput( builder, outputNode );
		if ( this.finish === null ) return out;
		const o = vec4( out ).toVar( 'impLit' );
		return vec4( this.finish( o.rgb ), o.a );

	}

	customProgramCacheKey() {

		return super.customProgramCacheKey() + this.finishKey;

	}

};

export class SurfaceStandardMaterial extends withFinish( THREE.MeshStandardNodeMaterial ) {}
export class SurfacePhysicalMaterial extends withFinish( THREE.MeshPhysicalNodeMaterial ) {}

// ── geometry ─────────────────────────────────────────────────────────────────────────────────

/** 4 corners in [-1, 1]², counter-clockwise (front face toward +z). */
export function quadGeometry() {

	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ] ), 3 ) );
	g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	return g;

}
