// Spike "lighting" — impostor spheres and strand ribbons lit by three's OWN lighting pipeline
// (MeshStandardNodeMaterial / MeshPhysicalNodeMaterial: direct lights, PMREM image-based lighting,
// shadow receiving and casting, MRT G-buffer), three.js r187 WebGPURenderer + TSL.
//
// ─── How the impostor surface is fed into three's lighting (all supported r187 API) ─────────────
//  vertex    material.positionNode   world-space corner of the camera-facing quad / ribbon, built from
//                                    storage buffers and the CAMERA NODES (cameraViewMatrix, …). The
//                                    shadow pass copies positionNode into its override material and
//                                    renders with the light's camera, so the same node faces the light
//                                    there. (material.vertexNode is NOT copied: do not use it.)
//  fragment  material.normalNode     view-space normal of the ray-traced surface
//            material.depthNode      depth of the surface point (generic: projection matrix of
//                                    whatever camera renders, so it is also right in the shadow pass)
//            material.contextNode    overrideNodes( [ positionView, positionViewDirection, positionWorld ] )
//                                    → lights, specular and the environment reflection see the surface
//                                    point instead of the flat quad (+ tangentView / bitangentView
//                                    when the material uses anisotropy)
//            material.receivedShadowPositionNode   world-space surface point for the shadow lookup
//            material.maskNode / maskShadowNode    analytic silhouette (discard) in main / shadow pass
//            material.mrtNode        per-material value for an MRT slot (custom motion vectors)
//
// ─── Exports ─────────────────────────────────────────────────────────────────────────────────
//  createSphereMaterial( buffers, options )  → MeshStandardNodeMaterial | MeshPhysicalNodeMaterial
//  createStrandMaterial( buffers, options )  → StrandNodeMaterial ('tube' | 'thin' | 'auto')
//  quadGeometry(), ribbonGeometry( segments )
//  velocityUniforms, updateVelocityUniforms( camera ), commitVelocityUniforms()
import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, positionGeometry, float, vec2, vec3, vec4, uniform, select, clamp, max, abs, sqrt, pow, dot, cross,
	normalize, length, fwidth, smoothstep, acos, mix, varyingProperty, cameraViewMatrix, cameraProjectionMatrix, cameraWorldMatrix,
	viewport, positionView, positionViewDirection, positionWorld, tangentView, bitangentView, overrideNodes, mrt, diffuseContribution,
	specularColorBlended, specularF90, roughness, F_Schlick, BRDF_Lambert, interleavedGradientNoise, screenCoordinate,
} from 'three/tsl';

// ── camera-generic helpers (valid in the main pass and in the shadow pass) ─────────────────────

/** true while an orthographic camera renders (the shadow pass of a DirectionalLight). */
const isOrthographic = () => cameraProjectionMatrix.element( 3 ).element( 3 ).greaterThan( 0.5 );

/** Pixels per world unit at view-space depth `viewZ` (negative in front of the camera). */
const pixelsPerUnit = ( viewZ ) => cameraProjectionMatrix.element( 1 ).element( 1 ).mul( viewport.w ).mul( 0.5 )
	.div( select( isOrthographic(), float( 1 ), max( viewZ.negate(), 1e-4 ) ) );

/** View-space point → fragment depth, for whatever camera is rendering (perspective / ortho). */
const viewToDepth = ( p ) => {

	const clip = cameraProjectionMatrix.mul( vec4( p, 1 ) );
	return clip.z.div( clip.w ).clamp( 0, 1 );

};

const viewToWorld = ( p ) => cameraWorldMatrix.mul( vec4( p, 1 ) ).xyz;

/** Unit direction from the camera through view-space point p (parallel rays for ortho cameras). */
const viewRay = ( p ) => select( isOrthographic(), vec3( 0, 0, - 1 ), normalize( p ) );

// ── motion vectors ─────────────────────────────────────────────────────────────────────────────
// three's `velocity` node only knows object matrices; positions that live in storage buffers need
// their own value. Same convention as VelocityNode: NDC( now ) − NDC( previous ), unjittered.

export const velocityUniforms = {
	viewProjection: uniform( new THREE.Matrix4() ), // current, unjittered
	previousViewProjection: uniform( new THREE.Matrix4() ),
};

let _hasPrevious = false;

/** Call once per frame before rendering (camera.projectionMatrix must be unjittered here). */
export function updateVelocityUniforms( camera ) {

	camera.updateMatrixWorld();
	velocityUniforms.viewProjection.value.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
	if ( ! _hasPrevious ) commitVelocityUniforms();

}

/** Call once per frame after rendering. */
export function commitVelocityUniforms() {

	velocityUniforms.previousViewProjection.value.copy( velocityUniforms.viewProjection.value );
	_hasPrevious = true;

}

/** Motion vector of a world-space surface point that moved by `worldDelta` since the last frame. */
const surfaceVelocity = ( hitWorld, worldDelta ) => {

	const now = velocityUniforms.viewProjection.mul( vec4( hitWorld, 1 ) );
	const before = velocityUniforms.previousViewProjection.mul( vec4( hitWorld.sub( worldDelta ), 1 ) );
	return vec4( now.xy.div( now.w ).sub( before.xy.div( before.w ) ), 0, 1 );

};

// ── shared: route a per-fragment surface into three's lighting ──────────────────────────────────

/**
 * @param {NodeMaterial} material a lit node material (lights = true)
 * @param {object} s { hit: view-space surface point, normal: view-space unit normal, viewDir: unit vector to the camera,
 *                     tangent: optional view-space unit tangent (only needed for anisotropy) }
 * @return {Node<vec3>} the world-space surface point
 */
function applySurface( material, s, overrides = true ) {

	const hitWorld = viewToWorld( s.hit ).toVar( 'impHitWorld' );
	// the accessors are overridden in the fragment stage only; the vertex stage keeps the originals
	const inFragment = ( original, value ) => ( builder ) => ( builder.shaderStage === 'fragment' ? value : original );
	material.normalNode = s.normal;
	material.depthNode = viewToDepth( s.hit );
	material.receivedShadowPositionNode = hitWorld;
	// overrides = false is a NEGATIVE TEST: point / spot lights then light the flat quad (see ?point=1&override=0)
	if ( overrides ) {

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

	}

	return hitWorld;

}

// ── geometry ───────────────────────────────────────────────────────────────────────────────────

/** 4 corners in [-1, 1]², counter-clockwise. */
export function quadGeometry() {

	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ] ), 3 ) );
	g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	return g;

}

/** Ribbon strip: position = ( t in [0,1], side ±1, 0 ), as in src/render/EdgeRenderer.js. */
export function ribbonGeometry( S ) {

	const g = new THREE.InstancedBufferGeometry();
	const pos = new Float32Array( ( S + 1 ) * 6 );
	const index = [];
	for ( let i = 0; i <= S; i ++ ) pos.set( [ i / S, - 1, 0, i / S, 1, 0 ], i * 6 );
	for ( let i = 0; i < S; i ++ ) index.push( 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2 );
	g.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
	g.setIndex( index );
	return g;

}

// ── impostor spheres ───────────────────────────────────────────────────────────────────────────

/**
 * @param {object} buffers { nodePos: vec4 (xyz, radius), nodeColor: vec4 (linear rgb, -),
 *                           nodeMat: vec4 (roughness, metalness, emissive strength, clearcoat),
 *                           nodeDelta: vec4 (xyz world motion since the previous frame, -) }
 * @param {object} [o] { physical: MeshPhysicalNodeMaterial with clearcoat, msaa: pass sample count > 1,
 *                       anisotropy: 0..1 (physical only), tangent = around the world y axis,
 *                       velocity: write a custom 'velocity' MRT slot (only when the pass has that slot),
 *                       useVertexNode, overrides: negative tests, see below and applySurface() }
 */
export function createSphereMaterial( buffers, { physical = false, anisotropy = 0, msaa = true, velocity = false, useVertexNode = false, overrides = true } = {} ) {

	const vCenter = varyingProperty( 'vec3', 'vSphCenter' ); // view space
	const vViewPos = varyingProperty( 'vec3', 'vSphViewPos' ); // quad point, view space
	const vQuad = varyingProperty( 'vec2', 'vSphQuad' ); // quad coords, silhouette at |q| = 1
	const vRadius = varyingProperty( 'float', 'vSphRadius' );
	const vColor = varyingProperty( 'vec3', 'vSphColor' );
	const vMat = varyingProperty( 'vec4', 'vSphMat' );
	const vDelta = varyingProperty( 'vec3', 'vSphDelta' );

	const material = physical ? new THREE.MeshPhysicalNodeMaterial() : new THREE.MeshStandardNodeMaterial();
	material.name = 'spike.sphere';
	material.fog = false;
	material.transparent = false;
	material.side = THREE.FrontSide;
	// the shadow pass swaps FrontSide → BackSide unless shadowSide is set: the quad would be culled
	material.shadowSide = THREE.FrontSide;
	material.alphaToCoverage = msaa;

	const billboard = Fn( () => {

		const p = buffers.nodePos.element( instanceIndex );
		const center = cameraViewMatrix.mul( vec4( p.xyz, 1 ) ).xyz.toVar();
		const R = p.w.toVar();
		const ortho = isOrthographic();
		const d2 = dot( center, center );
		const d = sqrt( d2 );
		// perspective: the silhouette cone cuts the plane through the centre (⟂ eye direction) in a circle
		// of radius h = R·d/√(d²−R²); orthographic: the plane ⟂ view axis and h = R
		const h = select( ortho, R, R.mul( d ).div( sqrt( max( d2.sub( R.mul( R ) ), 1e-6 ) ) ) ).toVar();
		const e = select( ortho, vec3( 0, 0, 1 ), center.negate().div( max( d, 1e-6 ) ) ).toVar();
		const upRef = select( abs( e.y ).greaterThan( 0.999 ), vec3( 1, 0, 0 ), vec3( 0, 1, 0 ) );
		const right = normalize( cross( upRef, e ) ).toVar();
		const up = cross( e, right );
		// 1.5 px of padding for the coverage ramp
		const pad = float( 1 ).add( float( 1.5 ).div( max( h.mul( pixelsPerUnit( center.z ) ), 1e-3 ) ) );
		const q = positionGeometry.xy.mul( pad ).toVar();
		const viewPos = center.add( right.mul( q.x ).add( up.mul( q.y ) ).mul( h ) ).toVar();

		vCenter.assign( center );
		vViewPos.assign( viewPos );
		vQuad.assign( q );
		vRadius.assign( R );
		vColor.assign( buffers.nodeColor.element( instanceIndex ).xyz );
		vMat.assign( buffers.nodeMat.element( instanceIndex ) );
		if ( velocity ) vDelta.assign( buffers.nodeDelta.element( instanceIndex ).xyz );

		// positionNode: local == world (the mesh transform is the identity); three applies view and projection
		return useVertexNode ? cameraProjectionMatrix.mul( vec4( viewPos, 1 ) ) : viewToWorld( viewPos );

	} )();

	// NEGATIVE TEST (useVertexNode): lighting still works, but the shadow pass ignores material.vertexNode
	// and draws the raw unit quad of the geometry, so the spheres cast no (or wrong) shadows.
	if ( useVertexNode ) material.vertexNode = billboard;
	else material.positionNode = billboard;

	// ray–sphere in view space (closest-approach form: stable far from the camera)
	const ortho = isOrthographic();
	const ro = select( ortho, vec3( vViewPos.xy, 0 ), vec3( 0 ) );
	const rd = viewRay( vViewPos ).toVar( 'sphRay' );
	const oc = vCenter.sub( ro );
	const b = dot( rd, oc );
	const m = oc.sub( rd.mul( b ) );
	const disc = max( vRadius.mul( vRadius ).sub( dot( m, m ) ), 0 );
	const hit = ro.add( rd.mul( b.sub( sqrt( disc ) ) ) ).toVar( 'sphHit' );
	const normal = normalize( hit.sub( vCenter ) ).toVar( 'sphNormal' );
	const viewDir = rd.negate();
	// analytic silhouette coverage (1 px ramp)
	const ql = length( vQuad );
	const coverage = clamp( float( 1 ).sub( ql ).div( max( fwidth( ql ), 1e-4 ) ).add( 0.5 ), 0, 1 ).toVar( 'sphCoverage' );

	// tangent for anisotropy: the direction of the circles of latitude around the world y axis
	const useAnisotropy = physical && anisotropy > 0;
	const tangent = useAnisotropy ? normalize( cross( cameraViewMatrix.mul( vec4( 0, 1, 0, 0 ) ).xyz, normal ) ).toVar( 'sphTangent' ) : null;

	const hitWorld = applySurface( material, { hit, normal, viewDir, tangent }, overrides );

	// MSAA: alpha-to-coverage turns the ramp into sub-sample coverage; otherwise a hard 50 % cut
	material.maskNode = coverage.greaterThan( msaa ? 0.002 : 0.5 );
	material.maskShadowNode = coverage.greaterThan( 0.5 );
	material.colorNode = vec4( vColor, 1 );
	material.opacityNode = coverage;
	material.roughnessNode = vMat.x;
	material.metalnessNode = vMat.y;
	// emissive core: strongest where the surface faces the viewer (HDR, feeds the bloom)
	material.emissiveNode = vColor.mul( vMat.z ).mul( pow( max( dot( normal, viewDir ), 0 ), 3 ) );

	if ( physical ) {

		material.clearcoatNode = vMat.w;
		material.clearcoatRoughnessNode = float( 0.06 );
		material.clearcoatNormalNode = normal; // default reads the geometry's normal attribute (there is none)
		if ( useAnisotropy ) material.anisotropyNode = vec2( anisotropy, 0 ); // strength along the tangent

	}

	if ( velocity ) material.mrtNode = mrt( { velocity: surfaceVelocity( hitWorld, vDelta ) } );

	return material;

}

// ── strand ribbons ─────────────────────────────────────────────────────────────────────────────

/**
 * Lighting model of the strand material. Wide ribbons are true tubes shaded by three's GGX model
 * (super.direct). Thin strands (sub-pixel tubes) use the average of a lit cylinder over its visible
 * half instead: an analytic diffuse term and a Kajiya-Kay style specular cone around the tangent.
 * `tubeWeight` blends between them; the indirect (environment) part always comes from three.
 */
class StrandLightingModel extends THREE.PhysicalLightingModel {

	constructor( strand ) {

		super();
		this.strand = strand; // { mode, tubeWeight, tangent, front }

	}

	direct( input, builder ) {

		const { lightDirection, lightColor, reflectedLight } = input;
		const { mode, tubeWeight, tangent: T, front } = this.strand;

		if ( mode !== 'thin' ) super.direct( { ...input, lightColor: mode === 'tube' ? lightColor : lightColor.mul( tubeWeight ) }, builder );
		if ( mode === 'tube' ) return;

		const light = mode === 'thin' ? lightColor : lightColor.mul( tubeWeight.oneMinus() );
		const V = positionViewDirection;

		// diffuse: Lambert averaged over the visible half cylinder,
		// E = sin(T,L) · ( sin φ + (π − φ) cos φ ) / 4, φ = azimuth of L around T measured from the viewer side
		const TdotL = dot( T, lightDirection );
		const sinTL = sqrt( max( TdotL.mul( TdotL ).oneMinus(), 0 ) );
		const Lp = lightDirection.sub( T.mul( TdotL ) ).div( max( sinTL, 1e-4 ) );
		const cosPhi = clamp( dot( Lp, front ), - 1, 1 );
		const sinPhi = sqrt( max( cosPhi.mul( cosPhi ).oneMinus(), 0 ) );
		const E = sinTL.mul( sinPhi.add( float( Math.PI ).sub( acos( cosPhi ) ).mul( cosPhi ) ) ).mul( 0.25 );
		reflectedLight.directDiffuse.addAssign( light.mul( E ).mul( BRDF_Lambert( { diffuseColor: diffuseContribution } ) ) );

		// specular: normalized Blinn-Phong lobe around the cone of normals ⟂ T (Kajiya-Kay), exponent from
		// roughness, Schlick fresnel, attenuated by how much of the lit half cylinder the viewer sees
		const H = normalize( lightDirection.add( V ) );
		const TdotH = dot( T, H );
		const sinTH = sqrt( max( TdotH.mul( TdotH ).oneMinus(), 0 ) );
		const alpha = roughness.mul( roughness );
		const n = float( 2 ).div( max( alpha.mul( alpha ), 1e-4 ) ).sub( 2 ).clamp( 1, 2000 );
		const D = pow( sinTH, n ).mul( n.add( 2 ) ).mul( 1 / ( 8 * Math.PI ) );
		const F = F_Schlick( { f0: specularColorBlended, f90: specularF90, dotVH: dot( V, H ).clamp() } );
		const visible = cosPhi.mul( 0.5 ).add( 0.5 );
		reflectedLight.directSpecular.addAssign( light.mul( F ).mul( D.mul( sinTL ).mul( visible ) ) );

	}

}

class StrandNodeMaterial extends THREE.MeshStandardNodeMaterial {

	setupLightingModel() {

		return new StrandLightingModel( this.strand );

	}

	// the lighting model is not a node property of the material: keep the programs apart
	customProgramCacheKey() {

		return super.customProgramCacheKey() + this.strand.mode;

	}

}

/**
 * @param {object} buffers { curveCtrl: vec4 × 4 per curve (cubic Bézier control points, world xyz),
 *                           curveColor: vec4 per curve (linear rgb, world radius) }
 * @param {object} [o] { mode: 'tube' | 'thin' | 'auto', msaa, velocity, roughness, metalness,
 *                       shadowDither: thin strands cast a shadow in proportion to the shadow-map texel fraction
 *                       they cover (dithered discard in the shadow pass) instead of a full texel,
 *                       blend: alpha-blended without depth writes instead of opaque + alpha-to-coverage,
 *                       gbuffer: names of the pass's MRT attachments besides 'output' (blend mode only),
 *                       uniforms: { width, minPx, maxPx (full on-screen width clamp in physical px), opacity } }
 */
export function createStrandMaterial( buffers, { mode = 'auto', msaa = true, velocity = false, blend = false, shadowDither = false, gbuffer = [], roughness: rough = 0.35, metalness: metal = 0, uniforms: U } = {} ) {

	const vViewPos = varyingProperty( 'vec3', 'vStrViewPos' ); // ribbon point, view space
	const vTangent = varyingProperty( 'vec3', 'vStrTangent' ); // view space
	const vProfile = varyingProperty( 'vec4', 'vStrProfile' ); // signed px across, half width px, world radius as drawn, coverage
	const vColor = varyingProperty( 'vec3', 'vStrColor' );

	const material = new StrandNodeMaterial();
	material.name = 'spike.strand.' + mode;
	material.fog = false;
	material.transparent = false;
	material.side = THREE.DoubleSide; // the winding depends on the screen direction of the curve
	material.shadowSide = THREE.DoubleSide;
	material.alphaToCoverage = msaa;

	material.positionNode = Fn( () => {

		const t = positionGeometry.x;
		const side = positionGeometry.y;
		const i0 = instanceIndex.mul( 4 );
		const p0 = buffers.curveCtrl.element( i0 ).xyz.toVar();
		const p1 = buffers.curveCtrl.element( i0.add( 1 ) ).xyz.toVar();
		const p2 = buffers.curveCtrl.element( i0.add( 2 ) ).xyz.toVar();
		const p3 = buffers.curveCtrl.element( i0.add( 3 ) ).xyz.toVar();
		const col = buffers.curveColor.element( instanceIndex );

		// cubic Bézier and its derivative
		const u = t.oneMinus();
		const pos = p0.mul( u.mul( u ).mul( u ) ).add( p1.mul( u.mul( u ).mul( t ).mul( 3 ) ) ).add( p2.mul( u.mul( t ).mul( t ).mul( 3 ) ) ).add( p3.mul( t.mul( t ).mul( t ) ) );
		const tan = p1.sub( p0 ).mul( u.mul( u ).mul( 3 ) ).add( p2.sub( p1 ).mul( u.mul( t ).mul( 6 ) ) ).add( p3.sub( p2 ).mul( t.mul( t ).mul( 3 ) ) );

		const C = cameraViewMatrix.mul( vec4( pos, 1 ) ).xyz.toVar();
		const T = normalize( cameraViewMatrix.mul( vec4( tan, 0 ) ).xyz ).toVar();

		// screen-space direction of the curve (exact derivative of the projection), as in EdgeRenderer
		const clip = cameraProjectionMatrix.mul( vec4( C, 1 ) ).toVar();
		const dclip = cameraProjectionMatrix.mul( vec4( T, 0 ) ).toVar();
		const sT = dclip.xy.mul( clip.w ).sub( clip.xy.mul( dclip.w ) ).mul( viewport.zw ).toVar();
		const sLen = length( sT );
		const dir = select( sLen.greaterThan( 1e-12 ), sT.div( max( sLen, 1e-12 ) ), vec2( 1, 0 ) );
		const across = vec2( dir.y.negate(), dir.x );

		// half width in px: projected world radius, clamped; below 1 px the ribbon stays 1 px wide and
		// fades by coverage; +0.5 px on each side for the analytic edge profile
		const pxPerUnit = pixelsPerUnit( C.z ).toVar();
		const half0 = clamp( col.w.mul( U.width ).mul( pxPerUnit ), U.minPx.mul( 0.5 ), U.maxPx.mul( 0.5 ) ).toVar();
		const halfPx = max( half0, 0.5 );
		const geomHalf = halfPx.add( 0.5 );
		// a pixel offset at constant depth is a view-space xy offset of px / pxPerUnit (any camera)
		const viewPos = C.add( vec3( across, 0 ).mul( side.mul( geomHalf ).div( pxPerUnit ) ) ).toVar();

		vViewPos.assign( viewPos );
		vTangent.assign( T );
		vProfile.assign( vec4( side.mul( geomHalf ), halfPx, halfPx.div( pxPerUnit ), clamp( half0.mul( 2 ), 0, 1 ) ) );
		vColor.assign( col.xyz );
		return viewToWorld( viewPos );

	} )();

	// tube frame per fragment: T (axis), front (⟂ T, towards the viewer), sideways = front × T
	const V = viewRay( vViewPos ).negate().toVar( 'strV' );
	const T = normalize( vTangent ).toVar( 'strT' );
	const f0 = V.sub( T.mul( dot( T, V ) ) );
	const front = select( dot( f0, f0 ).greaterThan( 1e-8 ), normalize( f0 ), V ).toVar( 'strFront' );
	const sideways = cross( front, T );

	const signedPx = vProfile.x;
	const halfPx = vProfile.y;
	// 1 = shade as a tube, 0 = shade as a thin strand; cross-fade between 1 and 2.5 px of full width
	const tubeWeight = ( mode === 'auto' ? smoothstep( 1.0, 2.5, halfPx.mul( 2 ) ) : float( mode === 'tube' ? 1 : 0 ) ).toVar( 'strTube' );
	// position across the tube in [-1, 1]; flattened to the front line as the strand gets thin
	const across = clamp( signedPx.div( halfPx ), - 1, 1 ).mul( tubeWeight );
	const bulge = sqrt( max( across.mul( across ).oneMinus(), 0 ) );
	const normal = normalize( front.mul( bulge ).add( sideways.mul( across ) ) ).toVar( 'strNormal' );
	// depth offset: the tube surface is radius·√(1−v²) in front of the ribbon plane
	const hit = vViewPos.add( front.mul( vProfile.z.mul( bulge ) ) ).toVar( 'strHit' );
	const edge = clamp( halfPx.add( 0.5 ).sub( abs( signedPx ) ), 0, 1 );
	const coverage = edge.mul( vProfile.w ).toVar( 'strCoverage' );

	const hitWorld = applySurface( material, { hit, normal, viewDir: V } );
	material.strand = { mode, tubeWeight, tangent: T, front };

	material.maskNode = coverage.greaterThan( msaa ? 0.002 : 0.5 );
	// Shadow pass: a strand is at least one shadow-map texel wide there, so a strand thinner than a texel
	// casts a shadow that is too wide. shadowDither discards in proportion to the covered texel fraction
	// (vProfile.w, computed with the shadow camera in that pass); PCF / VSM filtering averages the noise.
	material.maskShadowNode = shadowDither
		? edge.mul( vProfile.w ).greaterThan( interleavedGradientNoise( screenCoordinate.xy ).mul( 0.98 ).add( 0.01 ) )
		: edge.greaterThan( 0.5 );
	material.colorNode = vec4( vColor, 1 );
	material.opacityNode = coverage;
	// a pixel-wide strand averages the whole fan of tube normals: widen the lobe the environment sees
	material.roughnessNode = mix( max( float( rough ), 0.6 ), float( rough ), tubeWeight );
	material.metalnessNode = float( metal );

	if ( blend ) {

		// alpha-blended variant for dense, thin, semi-transparent bundles: still lit and shadow-receiving,
		// depth-TESTED per fragment against the tube surface (depthNode) but not depth-written
		material.transparent = true;
		material.depthWrite = false;
		material.alphaToCoverage = false;
		material.forceSinglePass = true; // transparent + DoubleSide is otherwise drawn twice (back faces, then front faces)
		material.maskNode = null;
		material.opacityNode = coverage.mul( U.opacity );
		// it must not overwrite the G-buffer of the opaque surface behind it: alpha 0 in every attachment
		// (the pass's MRT gives the attachments an alpha blend mode, see lighting.js)
		if ( gbuffer.length > 0 ) material.mrtNode = mrt( Object.fromEntries( gbuffer.map( ( name ) => [ name, vec4( 0 ) ] ) ) );

	} else if ( velocity ) {

		material.mrtNode = mrt( { velocity: surfaceVelocity( hitWorld, vec3( 0 ) ) } ); // static curves: camera motion only

	}

	return material;

}
