// Node renderer: one instanced, camera-facing impostor quad per node, ray-traced into a perfect
// sphere in the fragment shader (exact silhouette and per-pixel depth at any zoom, 4 vertices per
// node, so 20k+ nodes are trivial). Reads positions, colors and the eased highlight state straight
// from the shared GPU buffers (see src/gpu/GraphBuffers.js); nothing is copied per frame.
//
// ─── Geometry ───────────────────────────────────────────────────────────────────────────────
//  The quad lies in the plane through the sphere center perpendicular to the eye direction. The
//  silhouette cone (half angle asin(R/d)) cuts that plane in a circle of radius h = R·d/√(d²−R²),
//  so a quad of half size h (+1.5 px for anti-aliasing) bounds the sphere exactly under any
//  perspective. In the fragment shader `vQuad` (quad coords in units of h) gives an analytic,
//  pixel-accurate coverage for the silhouette and the ray–sphere hit gives normal and depth
//  (written through material.depthNode, so edges entering a node are correctly clipped by it).
//
// ─── Anti-aliasing ──────────────────────────────────────────────────────────────────────────
//  The sphere pass is opaque with depth writes and alpha-to-coverage (the scene pass is MSAA), so
//  the analytic coverage becomes a smooth MSAA edge without any sorting. Without MSAA the shader
//  falls back to a hard 50% coverage cut.
//
// ─── Size ───────────────────────────────────────────────────────────────────────────────────
//  radius = baseSize · nodeColor.w · size · (1 + 0.5·selected + 0.25·hovered)
//  baseSize = 0.065 · graph.layoutRadius / ∛N (≈ 0.08 · linkDistance for large graphs), so small
//  and big graphs look alike at the fitted zoom. Below `minPixelRadius` (CSS px) a node keeps
//  that on-screen size and fades out instead of shrinking into sub-pixel flicker.
//
// ─── Shading (view-space, camera-attached key light) ───────────────────────────────────────
//  wrapped soft diffuse + emissive core (HDR, feeds the bloom) + fresnel rim in a lighter tint +
//  small specular. nodeState (eased on the GPU) drives emphasis: selected/hovered/neighbors get
//  brighter, context nodes (state.z, gated by uniforms.focus) are darkened and desaturated.
//  A second additive pass draws the paper's "halo": a crisp ring plus a faint corona around the
//  selected node (strong), hovered node and the neighbors of the selection (mild). Instances
//  without any highlight are collapsed in the vertex shader, so the pass costs ~nothing.
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//  const nodes = app.use( new NodeRenderer( { size: 1 } ) );   // App module, order 11
//  nodes.size = 1.5;            // user node-size multiplier (settings.nodeSize)
//  nodes.visible = false;
//  nodes.minPixelRadius = 1.5;  // CSS px
//  nodes.halo = true;           // halo pass on/off
//  nodes.baseSize               // world radius of a radius-factor-1 node (read only, per graph)
//  nodes.worldRadius( i )       // world radius of node i at rest (no emphasis), for labels/picking
//  nodes.haloRadius( rPx, sel ) // outer halo ring radius (CSS px) for a sphere of rPx CSS px
//  nodes.setGraph( buffers, graph ), nodes.dispose()
import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, vec2, vec3, vec4, float, uniform, varyingProperty, positionGeometry,
	cameraViewMatrix, cameraProjectionMatrix, cameraNear, cameraFar, normalize, length, dot, cross,
	max, min, clamp, sqrt, pow, mix, smoothstep, fwidth, select, abs, luminance, exp, sin,
	viewZToPerspectiveDepth, Discard,
} from 'three/tsl';

const KEY_LIGHT = new THREE.Vector3( - 0.45, 0.65, 0.62 ).normalize();

/** Builds the quad geometry shared by both passes: 4 corners in [-1, 1]². */
function quadGeometry() {

	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ] ), 3 ) );
	g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	return g;

}

export class NodeRenderer {

	constructor( { size = 1, minPixelRadius = 1.5, halo = true } = {} ) {

		this.order = 11;
		this.isNodeRenderer = true;
		this.app = null;
		this.graph = null;
		this.buffers = null;
		this.mesh = null;
		this.haloMesh = null;
		this.baseSize = 1;
		this._visible = true;
		this._halo = halo;

		this.uniforms = {
			size: uniform( size ),
			baseSize: uniform( 1 ),
			minPx: uniform( minPixelRadius ),
			dpr: uniform( 1 ),
			viewportH: uniform( 800 ),
			cut: uniform( 0.002 ), // coverage discard threshold (0.5 when the scene pass has no MSAA)
		};

		this._geometry = quadGeometry();
		this._v2 = new THREE.Vector2();

	}

	get size() {

		return this.uniforms.size.value;

	}

	set size( v ) {

		this.uniforms.size.value = Math.max( 0.05, + v || 1 );

	}

	get minPixelRadius() {

		return this.uniforms.minPx.value;

	}

	set minPixelRadius( v ) {

		this.uniforms.minPx.value = Math.max( 0, v );

	}

	get visible() {

		return this._visible;

	}

	set visible( v ) {

		this._visible = !! v;
		if ( this.mesh ) this.mesh.visible = this._visible;
		if ( this.haloMesh ) this.haloMesh.visible = this._visible && this._halo;

	}

	get halo() {

		return this._halo;

	}

	set halo( v ) {

		this._halo = !! v;
		if ( this.haloMesh ) this.haloMesh.visible = this._visible && this._halo;

	}

	/** World radius of node i without emphasis (what the sphere pass draws at rest). */
	worldRadius( i ) {

		const r = this.graph && this.graph.radii ? this.graph.radii[ i ] : 1;
		return this.baseSize * r * this.size;

	}

	/**
	 * Outer edge (CSS px) of the halo ring drawn around a node whose sphere has an on-screen
	 * radius of rPx CSS px; `selected` in [0,1]. Mirrors ringR/ringW in the halo shader.
	 */
	haloRadius( rPx, selected = 0 ) {

		return rPx * 1.3 + 2.5 + 1.5 * selected + 0.5 * ( 1 + 0.8 * selected );

	}

	init( app ) {

		this.app = app;

	}

	setGraph( buffers, graph, app ) {

		this._disposeMeshes( app || this.app );
		this.buffers = buffers;
		this.graph = graph;
		const N = Math.max( 1, graph.N );
		this.baseSize = 0.065 * graph.layoutRadius / Math.cbrt( N );
		this.uniforms.baseSize.value = this.baseSize;

		this.mesh = new THREE.Mesh( this._geometry, this._sphereMaterial( buffers ) );
		this.mesh.name = 'NodeRenderer.spheres';
		this.mesh.count = graph.N;
		this.mesh.frustumCulled = false;
		this.mesh.visible = this._visible;

		this.haloMesh = new THREE.Mesh( this._geometry, this._haloMaterial( buffers ) );
		this.haloMesh.name = 'NodeRenderer.halo';
		this.haloMesh.count = graph.N;
		this.haloMesh.frustumCulled = false;
		this.haloMesh.renderOrder = 5; // after edges (transparent), it is additive anyway
		this.haloMesh.visible = this._visible && this._halo;

		( app || this.app ).scene.add( this.mesh, this.haloMesh );

	}

	update( /* dt, app */ ) {

		const r = this.app && this.app.renderer;
		if ( ! r ) return;
		r.getDrawingBufferSize( this._v2 );
		this.uniforms.viewportH.value = Math.max( 1, this._v2.y );
		this.uniforms.dpr.value = r.getPixelRatio();
		this.uniforms.cut.value = ( r.samples || 0 ) > 1 ? 0.002 : 0.5;

	}

	// ── shared vertex logic ─────────────────────────────────────────────────────────────────

	/**
	 * Per-instance billboard. Returns nodes for the clip position and the quantities the
	 * fragment stages need. `extent( rPx, state )` gives the quad half size in units of the
	 * silhouette radius (≥ 1); rPx is the silhouette radius in physical pixels.
	 */
	_billboard( buffers, extent ) {

		const U = this.uniforms;
		const pos = buffers.nodePos.element( instanceIndex );
		const col = buffers.nodeColor.element( instanceIndex );
		const st = buffers.nodeState.element( instanceIndex );

		const center = cameraViewMatrix.mul( vec4( pos.xyz, 1 ) ).xyz.toVar( 'nCenter' );
		const R0 = U.baseSize.mul( col.w ).mul( U.size ).mul( st.x.mul( 0.5 ).add( st.w.mul( 0.25 ) ).add( 1 ) ).toVar( 'nR0' );

		// pixels per world unit at the node's depth (physical pixels)
		const P11 = cameraProjectionMatrix.element( 1 ).element( 1 );
		const pxScale = P11.mul( U.viewportH ).mul( 0.5 ).div( max( center.z.negate(), cameraNear ) ).toVar( 'nPxScale' );
		const minPx = U.minPx.mul( U.dpr );
		const R = max( R0, minPx.div( pxScale ) ).toVar( 'nR' );
		const fade = clamp( R0.mul( pxScale ).div( max( minPx, 1e-3 ) ), 0, 1 );

		const d2 = dot( center, center );
		const r2 = R.mul( R );
		const inside = d2.lessThan( r2.mul( 1.0001 ) );
		const d = sqrt( d2 );
		const h = R.mul( d ).div( sqrt( max( d2.sub( r2 ), 1e-6 ) ) ).toVar( 'nH' );
		const rPx = h.mul( pxScale ).toVar( 'nRPx' );

		const e = center.negate().div( max( d, 1e-6 ) );
		const upRef = select( abs( e.y ).greaterThan( 0.999 ), vec3( 1, 0, 0 ), vec3( 0, 1, 0 ) );
		const right = normalize( cross( upRef, e ) );
		const up = cross( e, right );
		const k = extent( rPx, st ).toVar( 'nExtent' );
		const q = positionGeometry.xy.mul( k );
		const viewPos = center.add( right.mul( q.x ).add( up.mul( q.y ) ).mul( h ) ).toVar( 'nViewPos' );
		const clip = cameraProjectionMatrix.mul( vec4( viewPos, 1 ) );

		return { center, R, h, rPx, q, extent: k, viewPos, clip, inside, fade, col, st };

	}

	// ── sphere pass ─────────────────────────────────────────────────────────────────────────

	_sphereMaterial( buffers ) {

		const U = this.uniforms;
		const focus = buffers.uniforms.focus;

		const vCenter = varyingProperty( 'vec3', 'vNodeCenter' );
		const vViewPos = varyingProperty( 'vec3', 'vNodeViewPos' );
		const vQuad = varyingProperty( 'vec2', 'vNodeQuad' );
		const vRadius = varyingProperty( 'float', 'vNodeRadius' );
		const vColor = varyingProperty( 'vec4', 'vNodeColor' ); // rgb, fade
		const vState = varyingProperty( 'vec4', 'vNodeState' );

		const material = new THREE.NodeMaterial();
		material.name = 'NodeRenderer.sphere';
		material.lights = false;
		material.fog = false;
		material.transparent = false;
		material.depthWrite = true;
		material.depthTest = true;
		material.alphaToCoverage = true;

		material.vertexNode = Fn( () => {

			// 1.5 px of padding around the silhouette for the coverage ramp
			const b = this._billboard( buffers, ( rPx ) => float( 1 ).add( U.dpr.mul( 1.5 ).div( max( rPx, 1e-3 ) ) ) );
			vCenter.assign( b.center );
			vViewPos.assign( b.viewPos );
			vQuad.assign( b.q );
			vRadius.assign( b.R );
			vColor.assign( vec4( b.col.xyz, b.fade ) );
			vState.assign( b.st );
			return select( b.inside, vec4( 0, 0, 2, 1 ), b.clip );

		} )();

		const shade = Fn( () => {

			// analytic silhouette coverage (1 px ramp)
			const ql = length( vQuad );
			const aa = max( fwidth( ql ), 1e-4 );
			const cov = clamp( float( 1 ).sub( ql ).div( aa ).add( 0.5 ), 0, 1 ).toVar();
			Discard( cov.lessThanEqual( U.cut ) );

			// ray–sphere (closest-approach form: stable far from the camera)
			const rd = normalize( vViewPos );
			const b = dot( rd, vCenter );
			const m = vCenter.sub( rd.mul( b ) );
			const disc = max( vRadius.mul( vRadius ).sub( dot( m, m ) ), 0 );
			const t = b.sub( sqrt( disc ) );
			const hit = rd.mul( t ).toVar( 'nHit' );
			const n = normalize( hit.sub( vCenter ) ).toVar( 'nNormal' );
			const V = rd.negate();
			const L = vec3( KEY_LIGHT.x, KEY_LIGHT.y, KEY_LIGHT.z );

			// partial luminance normalization: bright palettes (aurora, rainbow) and dark hues (blue)
			// end up in a similar brightness band, so the bloom does not favour some clusters
			const base = vColor.xyz.mul( mix( float( 1 ), float( 0.28 ).div( max( luminance( vColor.xyz ), 0.04 ) ), 0.45 ) ).toVar( 'nBase' );
			const sel = vState.x, nb = vState.y, hov = vState.w;
			const dim = min( vState.z, focus ).mul( float( 1 ).sub( hov ) ); // hovering lifts the dimming

			const wrap = clamp( dot( n, L ).mul( 0.55 ).add( 0.45 ), 0, 1 );
			const diff = wrap.mul( wrap );
			const ndv = clamp( dot( n, V ), 0, 1 );
			const rim = pow( float( 1 ).sub( ndv ), 2.6 );
			const H = normalize( L.add( V ) );
			const spec = pow( max( dot( n, H ), 0 ), 56 );
			const core = pow( ndv, 3 );

			// emissive core, luminance equalized a bit further (HDR, feeds the bloom)
			const lum = max( luminance( base ), 0.04 );
			const emissive = base.mul( mix( float( 1 ), float( 0.3 ).div( lum ), 0.35 ) );
			const tint = mix( base, vec3( 1 ), 0.22 ).mul( 1.25 );
			const c = base.mul( diff.mul( 1.0 ).add( 0.18 ) )
				.add( emissive.mul( core ).mul( 1.4 ) )
				.add( tint.mul( rim ).mul( 1.1 ) )
				.add( vec3( spec.mul( 0.6 ) ) ).toVar( 'nShade' );

			// emphasis: brighter selection/hover/neighbors, a touch whiter when selected
			const boost = float( 1 ).add( sel.mul( 0.55 ) ).add( hov.mul( 0.4 ) ).add( nb.mul( 0.3 ) );
			c.assign( mix( c, c.add( vec3( luminance( c ).mul( 0.6 ) ) ), sel.mul( 0.2 ) ).mul( boost ) );

			// context dimming: desaturate + darken
			const gray = vec3( luminance( c ).mul( 0.5 ) );
			c.assign( mix( c, gray, dim.mul( 0.75 ) ).mul( float( 1 ).sub( dim.mul( 0.6 ) ) ) );

			// alpha-to-coverage turns this into MSAA coverage (see `cut` for the non-MSAA case)
			return vec4( c, cov.mul( vColor.w ) );

		} );

		material.colorNode = shade();
		material.depthNode = Fn( () => {

			const rd = normalize( vViewPos );
			const b = dot( rd, vCenter );
			const m = vCenter.sub( rd.mul( b ) );
			const disc = max( vRadius.mul( vRadius ).sub( dot( m, m ) ), 0 );
			const z = rd.z.mul( b.sub( sqrt( disc ) ) );
			return clamp( viewZToPerspectiveDepth( z, cameraNear, cameraFar ), 0, 1 );

		} )();

		return material;

	}

	// ── halo pass ───────────────────────────────────────────────────────────────────────────

	_haloMaterial( buffers ) {

		const U = this.uniforms;
		const time = buffers.uniforms.time;

		const vQuadPx = varyingProperty( 'vec2', 'vHaloQuadPx' ); // quad coords in physical px
		const vRPx = varyingProperty( 'float', 'vHaloRPx' );
		const vEdgePx = varyingProperty( 'float', 'vHaloEdgePx' ); // inscribed radius of the quad
		const vColor = varyingProperty( 'vec4', 'vHaloColor' ); // rgb, strength
		const vSel = varyingProperty( 'float', 'vHaloSel' );

		const material = new THREE.NodeMaterial();
		material.name = 'NodeRenderer.halo';
		material.lights = false;
		material.fog = false;
		material.transparent = true;
		material.depthWrite = false;
		material.depthTest = true;
		material.blending = THREE.AdditiveBlending;

		// ring radius / width in px for a silhouette of rPx px
		const ringR = ( rPx, sel ) => rPx.mul( 1.3 ).add( U.dpr.mul( float( 2.5 ).add( sel.mul( 1.5 ) ) ) );
		const ringW = ( sel ) => U.dpr.mul( float( 1.0 ).add( sel.mul( 0.8 ) ) );

		material.vertexNode = Fn( () => {

			const b = this._billboard( buffers, ( rPx, st ) => ringR( rPx, st.x ).add( ringW( st.x ).mul( 3 ) ).add( U.dpr.mul( 8 ) ).div( max( rPx, 1e-3 ) ) );
			// selected: full; hovered: strong; neighbors of the selection: mild
			const strength = max( b.st.x, max( b.st.y.mul( 0.5 ), b.st.w.mul( 0.8 ) ) );
			vQuadPx.assign( b.q.mul( b.rPx ) );
			vRPx.assign( b.rPx );
			vEdgePx.assign( b.rPx.mul( b.extent ) );
			vColor.assign( vec4( b.col.xyz, strength.mul( b.fade.mul( 0.6 ).add( 0.4 ) ) ) );
			vSel.assign( max( b.st.x, b.st.w.mul( 0.5 ) ) );
			const off = b.inside.or( strength.lessThan( 0.004 ) );
			return select( off, vec4( 0, 0, 2, 1 ), b.clip );

		} )();

		material.colorNode = Fn( () => {

			const r = length( vQuadPx );
			const sel = vSel;
			const rr = ringR( vRPx, sel );
			const hw = ringW( sel ).mul( 0.5 );
			// crisp ring: 1 px anti-aliased edges around a constant pixel width
			const ring = clamp( hw.add( 0.5 ).sub( abs( r.sub( rr ) ) ), 0, 1 );
			// faint corona from the silhouette out to the ring, soft glow around the ring;
			// both windowed to reach exactly 0 before the quad border
			const outside = smoothstep( vRPx.sub( 0.5 ), vRPx.add( 1.0 ), r );
			const win = smoothstep( vEdgePx, vEdgePx.mul( 0.7 ), r );
			const corona = exp( r.sub( vRPx ).max( 0 ).negate().div( vRPx.mul( 0.3 ).add( U.dpr.mul( 2.5 ) ) ) ).mul( 0.16 );
			const glow = exp( abs( r.sub( rr ) ).negate().div( U.dpr.mul( 2.2 ) ) ).mul( 0.22 );
			const pulse = sin( time.mul( 2.4 ) ).mul( 0.1 ).mul( vSel ).add( 1 );
			// neighbors keep their hue, the selection goes towards white-hot
			const tint = mix( vColor.xyz, vec3( 1 ), sel.mul( 0.45 ).add( 0.12 ) );
			const a = ring.mul( 0.95 ).add( corona.add( glow ).mul( win ) ).mul( outside ).mul( vColor.w ).mul( pulse );
			return vec4( tint.mul( a ).mul( float( 1.5 ).add( sel.mul( 0.9 ) ) ), 1 );

		} )();

		return material;

	}

	_disposeMeshes( app ) {

		for ( const m of [ this.mesh, this.haloMesh ] ) {

			if ( ! m ) continue;
			m.removeFromParent();
			if ( app && app.scene ) app.scene.remove( m );
			m.material.dispose();

		}

		this.mesh = null;
		this.haloMesh = null;

	}

	dispose() {

		this._disposeMeshes( this.app );
		this._geometry.dispose();

	}

}
