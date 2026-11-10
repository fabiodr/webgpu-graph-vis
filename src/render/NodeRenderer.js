// Node renderer: one instanced, camera-facing impostor quad per node, ray-traced into a perfect
// sphere in the fragment shader (exact silhouette and per-pixel depth at any zoom, 4 vertices per
// node) and lit by three's own physically based lighting: the key DirectionalLight with PCF
// shadows (cast and received), the procedural PMREM environment (see gfx/Environment.js) and an
// emissive core in the node colour. Positions, colours, the eased highlight state and the effect
// channels are read straight from the shared GPU buffers (src/gpu/GraphBuffers.js).
//
// ─── Geometry ───────────────────────────────────────────────────────────────────────────────
//  material.positionNode builds the billboard from the camera nodes (gfx/impostor.js), so the
//  shadow pass, which renders the same node graph with the light's orthographic camera, gets a
//  quad that faces the light: spheres cast round shadows. The fragment stage intersects the view
//  ray with the sphere and hands hit point, normal and depth to three (applySurface()).
//  castShadowPositionNode is the same billboard without hover / ripple scaling and without the
//  minimum pixel radius, so hovering never invalidates the shadow map.
//
// ─── Material variants (chosen from gfxState, swapped on the 'gfx' event) ───────────────────
//  shading   'std'    MeshStandardNodeMaterial                         (gfxState.lit)
//            'phys'   MeshPhysicalNodeMaterial + clearcoat              (lit and tier 'ultra')
//            'unlit'  plain NodeMaterial: hemisphere ambient + key light + Blinn highlight + rim,
//                     no texture taps, no shadows                       (gfxState.lit === false)
//  coverage  'a2c'    alpha-to-coverage of the analytic silhouette      (gfxState.samples > 1)
//            'cut'    hard cut at 50 % coverage (TRAA / FXAA do the rest)
//  velocity  'mv'     binds nodePrev, writes velocityOutput( motionVector( hit now, hit before ) );
//                     the halo then carries skipVelocity()
//            'still'  nodePrev is not bound, no material has an mrtNode
//  One material per used variant and graph is kept and re-used when a variant comes back.
//  ('mv' materials can only be drawn into a target that has an `output` texture, i.e. through a
//  PassNode: three uses material.mrtNode as the whole fragment output otherwise.)
//  Storage buffers per stage: vertex 4 (nodePos, nodeColor, nodeState, nodeFx), 5 with 'mv'
//  (+ nodePrev); shadow pass vertex 3; halo vertex 3; fragment 0 in every variant.
//
// ─── Size ───────────────────────────────────────────────────────────────────────────────────
//  radius = baseSize · nodeColor.w · size · (1 + 0.5·selected + 0.25·hovered)
//  baseSize = 0.065 · graph.layoutRadius / ∛N. Below `minPixelRadius` (CSS px) a node keeps that
//  on-screen size and fades out instead of shrinking into sub-pixel flicker: through coverage with
//  MSAA ('a2c'); in the single-sample variants ('cut') the dot stays solid and its colour fades
//  toward the backdrop (gfxUniforms.fogColor), which is what the coverage fade averages to. That is
//  deterministic: no stipple, nothing that crawls while the camera moves, with or without a
//  temporal resolve. Dots that would keep less than 4 % are dropped in the vertex stage.
//  A ripple pulse (below) adds a transient scale-up of rippleScale · gain.
//
// ─── Look ("glass enamel") ──────────────────────────────────────────────────────────────────
//  Dielectric, glossy base in the node colour (partly luminance-equalised so dark hues stay
//  readable), an HDR emissive core (feeds the bloom) that sits on the side facing away from the key
//  light, like light leaving a glass bead: highlight on one side, saturated glow on the other. Its
//  strength falls to 55 % for spheres larger than ~90 CSS px, so distant nodes are luminous dots
//  and close ones show their surface. A fresnel rim in a lighter tint, specular highlight and
//  reflections from the key light and the environment, clearcoat in the ultra tier. All look
//  parameters are uniforms (setLook).
//  Shadows: the key light's shadow term only removes the key light, and a bead that glows by
//  itself in a bright environment hardly changes when it loses that. So where a shadow is CAST on
//  the lit side (weight smoothstep( 0, 0.35, N·L ): never the bead's own far side, where the term is
//  self-shadowing) the whole colour is dimmed by `shade` × the shadow, hue and saturation
//  preserved: light that would have entered the bead does not arrive. The term is taken from
//  three's shadow node through material.receivedShadowNode (no extra taps).
//  Specular anti-aliasing: on a sphere of r pixels the normal turns by 1 / r radians per pixel, so
//  a glossy lobe (GGX α = roughness²) narrower than that is a highlight smaller than a pixel that
//  appears and vanishes with sub-pixel motion (and the bloom amplifies it). The roughness is
//  therefore floored at √( min( 0.6 / r, 0.6 ) ), r in target pixels: spheres above ~9 px keep the
//  look roughness, smaller ones get a wider, dimmer sheen. Same floor for the unlit highlight and
//  the clearcoat.
//  nodeState (eased on the GPU) drives emphasis: selected / hovered / neighbour nodes glow
//  stronger, context nodes (state.z, gated by uniforms.focus) recede into dark, cool glass beads
//  (their own glow goes out, hue almost gone, a little gloss and a cool rim keep the form).
//  nodeFx: x = hop distance of the ripple (gain = rippleGain·exp(−((x−front)/width)²), a flash of
//  the node's OWN core colour plus the scale-up, so clusters stay distinguishable on the front),
//  z = glow weight (extra emissive). Fog (gfx/shared.js) is applied to the final colour, emissive
//  included.
//  A second additive pass draws the "halo": a crisp ring plus a faint corona around the selected
//  node (strong), the hovered node and the neighbours of the selection (mild). It writes no depth
//  and no motion vector; instances without any highlight are collapsed in the vertex shader.
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
//  nodes.setLook( { roughness, metalness, clearcoat, clearcoatRoughness, emissive, coreShift, rim,
//                   shade, flash, glow, rippleScale, variation } )   // any subset, live (uniforms)
//  nodes.look                   // current values (plain object copy)
//  nodes.variant                // key of the active material variant, e.g. 'std|a2c|still'
//  nodes.setGraph( buffers, graph ), nodes.dispose()
import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, vec3, vec4, float, uniform, varyingProperty, property, positionGeometry, cameraViewMatrix,
	normalize, length, dot, max, min, clamp, pow, mix, smoothstep, select, abs, luminance, exp, sin, sqrt,
} from 'three/tsl';
import { gfxUniforms, gfxState, motionVector, velocityOutput, skipVelocity, applyFog, fogTransmittance } from './gfx/shared.js';
import {
	pixelsPerUnit, viewToDepth, viewToWorld, applySurface, sphereBillboard, raySphere, discCoverage,
	SurfaceStandardMaterial, SurfacePhysicalMaterial, quadGeometry,
} from './gfx/impostor.js';

/** Default look; see setLook(). */
export const NODE_LOOK = {
	roughness: 0.26, // base roughness of the enamel
	metalness: 0,
	clearcoat: 1, // ultra tier only
	clearcoatRoughness: 0.2,
	emissive: 2, // strength of the emissive core
	coreShift: 0.45, // 0..1: how far the core sits away from the key light (0 = centred on the view axis)
	rim: 0.55, // strength of the fresnel rim
	shade: 0.5, // 0..1: how much a cast shadow also dims the bead's own light (hue preserved)
	flash: 1.2, // emissive strength of a ripple pulse at gain 1 (in the node's own core colour)
	glow: 1.1, // extra emissive gain of nodeFx.z = 1
	rippleScale: 0.18, // transient radius increase of a ripple pulse at gain 1
	variation: 0.5, // 0..1: larger (higher-degree) nodes are glossier, small ones more matte
};

/** Material variant for a pipeline configuration (see the header). */
export function nodeVariant( state = gfxState ) {

	const shading = state.lit === false ? 'unlit' : state.tier === 'ultra' ? 'phys' : 'std';
	return `${shading}|${state.samples > 1 ? 'a2c' : 'cut'}|${state.velocity ? 'mv' : 'still'}`;

}

export class NodeRenderer {

	constructor( { size = 1, minPixelRadius = 1.5, halo = true, look = null } = {} ) {

		this.order = 11;
		this.isNodeRenderer = true;
		this.app = null;
		this.graph = null;
		this.buffers = null;
		this.mesh = null;
		this.haloMesh = null;
		this.baseSize = 1;
		this.variant = '';
		this._visible = true;
		this._halo = halo;
		this._materials = new Map(); // variant key → material (spheres and halo, for the current graph)
		this._castShadow = null;

		this.uniforms = {
			size: uniform( size ),
			baseSize: uniform( 1 ),
			minPx: uniform( minPixelRadius ),
		};
		for ( const key in NODE_LOOK ) this.uniforms[ key ] = uniform( NODE_LOOK[ key ] );
		if ( look ) this.setLook( look );

		this._geometry = quadGeometry();
		this._onGfx = () => this._applyVariant();

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

	/** Current look parameters (a copy; change them with setLook). */
	get look() {

		const out = {};
		for ( const key in NODE_LOOK ) out[ key ] = this.uniforms[ key ].value;
		return out;

	}

	/** Sets any subset of the look parameters (see NODE_LOOK). Live: uniforms only. */
	setLook( look = {} ) {

		for ( const key in look ) {

			if ( key in NODE_LOOK && Number.isFinite( + look[ key ] ) ) this.uniforms[ key ].value = + look[ key ];

		}

		return this;

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
		app.events.addEventListener( 'gfx', this._onGfx );

	}

	setGraph( buffers, graph, app ) {

		this._disposeMeshes( app || this.app );
		this.buffers = buffers;
		this.graph = graph;
		const N = Math.max( 1, graph.N );
		this.baseSize = 0.065 * graph.layoutRadius / Math.cbrt( N );
		this.uniforms.baseSize.value = this.baseSize;

		this.mesh = new THREE.Mesh( this._geometry, null );
		this.mesh.name = 'NodeRenderer.spheres';
		this.mesh.count = graph.N;
		this.mesh.frustumCulled = false; // also against the shadow camera
		this.mesh.visible = this._visible;

		this.haloMesh = new THREE.Mesh( this._geometry, null );
		this.haloMesh.name = 'NodeRenderer.halo';
		this.haloMesh.count = graph.N;
		this.haloMesh.frustumCulled = false;
		this.haloMesh.renderOrder = 5; // after edges (transparent), it is additive anyway
		this.haloMesh.visible = this._visible && this._halo;
		this.haloMesh.castShadow = this.haloMesh.receiveShadow = false;
		this._applyVariant();

		( app || this.app ).scene.add( this.mesh, this.haloMesh );

	}

	update() {

		// robust against a missed 'gfx' event: the key is a short string compare
		if ( this.mesh && this.variant !== nodeVariant() ) this._applyVariant();
		if ( this.mesh && this._castShadow !== ( gfxState.shadows && gfxState.lit !== false ) ) this._applyVariant();

	}

	/** Picks (and lazily builds) the material for the current gfxState. */
	_applyVariant() {

		if ( ! this.mesh || ! this.buffers ) return;
		const key = nodeVariant();
		let material = this._materials.get( key );
		if ( ! material ) {

			const [ shading, coverage, motion ] = key.split( '|' );
			material = this._sphereMaterial( this.buffers, { shading, a2c: coverage === 'a2c', velocity: motion === 'mv' } );
			material.name = 'NodeRenderer.sphere.' + key;
			this._materials.set( key, material );

		}

		this.variant = key;
		this.mesh.material = material;
		this._castShadow = gfxState.shadows && gfxState.lit !== false;
		this.mesh.castShadow = this._castShadow;
		this.mesh.receiveShadow = this._castShadow;

		if ( this.haloMesh ) {

			const haloKey = gfxState.velocity ? 'halo|mv' : 'halo|still';
			let halo = this._materials.get( haloKey );
			if ( ! halo ) {

				halo = this._haloMaterial( this.buffers, { velocity: !! gfxState.velocity } );
				this._materials.set( haloKey, halo );

			}

			this.haloMesh.material = halo;

		}

	}

	// ── shared vertex logic ─────────────────────────────────────────────────────────────────

	/**
	 * Per-instance quantities (vertex stage). `shadow`: the variant for the shadow pass (rest size:
	 * no hover / ripple scaling, no minimum pixel radius). `fx`: read the nodeFx channels.
	 */
	_instance( buffers, { shadow = false, fx = true } = {} ) {

		const U = this.uniforms, S = buffers.uniforms;
		const pos = buffers.nodePos.element( instanceIndex );
		const col = buffers.nodeColor.element( instanceIndex );
		const st = buffers.nodeState.element( instanceIndex );

		const center = cameraViewMatrix.mul( vec4( pos.xyz, 1 ) ).xyz.toVar( 'nCenter' );
		let scale = st.x.mul( 0.5 ).add( 1 );
		if ( ! shadow ) scale = scale.add( st.w.mul( 0.25 ) );
		let ripple = float( 0 ), glow = float( 0 );
		if ( fx && ! shadow ) {

			const f = buffers.nodeFx.element( instanceIndex );
			const k = f.x.sub( S.rippleFront ).div( max( S.rippleWidth, 1e-3 ) );
			ripple = select( f.x.greaterThanEqual( 0 ), S.rippleGain.mul( exp( k.mul( k ).negate() ) ), float( 0 ) ).toVar( 'nRipple' );
			glow = f.z;
			scale = scale.mul( ripple.mul( U.rippleScale ).add( 1 ) );

		}

		const R0 = U.baseSize.mul( col.w ).mul( U.size ).mul( scale ).toVar( 'nR0' );
		const ppu = pixelsPerUnit( center.z ).toVar( 'nPpu' ); // target pixels per world unit at the node
		let R = R0, fade = float( 1 );
		if ( ! shadow ) {

			const minPx = U.minPx.mul( gfxUniforms.pxScale );
			R = max( R0, minPx.div( ppu ) ).toVar( 'nR' );
			fade = clamp( R0.mul( ppu ).div( max( minPx, 1e-3 ) ), 0, 1 );

		}

		const bb = sphereBillboard( center, R );
		const h = bb.h.toVar( 'nH' );
		const rPx = h.mul( ppu ).toVar( 'nRPx' ); // silhouette radius in target pixels
		return { pos, col, st, center, R, fade, h, rPx, ripple, glow, right: bb.right, up: bb.up, inside: bb.inside };

	}

	/** Quad corner for an instance: `k` = half size in units of the silhouette radius (≥ 1). */
	_corner( I, k ) {

		const q = positionGeometry.xy.mul( k ).toVar( 'nQuad' );
		const viewPos = I.center.add( I.right.mul( q.x ).add( I.up.mul( q.y ) ).mul( I.h ) ).toVar( 'nViewPos' );
		return { q, viewPos };

	}

	// ── sphere pass ─────────────────────────────────────────────────────────────────────────

	_sphereMaterial( buffers, { shading = 'std', a2c = true, velocity = false } = {} ) {

		const U = this.uniforms, G = gfxUniforms;
		const focus = buffers.uniforms.focus;
		const lit = shading !== 'unlit';

		const vSphere = varyingProperty( 'vec4', 'vNodeSphere' ); // view-space centre, radius
		const vPoint = varyingProperty( 'vec4', 'vNodePoint' ); // view-space quad point, fade
		const vQuad = varyingProperty( 'vec4', 'vNodeQuad' ); // quad coords (silhouette at |q| = 1), ripple gain, glow weight
		const vColor = varyingProperty( 'vec4', 'vNodeColor' ); // linear rgb, radius factor
		const vState = varyingProperty( 'vec4', 'vNodeState' ); // selected, neighbour, dimmed, hovered
		const vDelta = varyingProperty( 'vec3', 'vNodeDelta' ); // world motion since the previous frame

		const material = lit ? ( shading === 'phys' ? new SurfacePhysicalMaterial() : new SurfaceStandardMaterial() ) : new THREE.NodeMaterial();
		material.fog = false;
		material.lights = lit;
		material.transparent = false;
		material.depthWrite = true;
		material.depthTest = true;
		material.side = THREE.FrontSide;
		// the shadow pass swaps FrontSide → BackSide unless shadowSide is set: the quad would be culled
		material.shadowSide = THREE.FrontSide;
		material.alphaToCoverage = a2c;

		const position = ( shadow ) => Fn( () => {

			const I = this._instance( buffers, { shadow } );
			// 1.5 px of padding around the silhouette for the coverage ramp
			const { q, viewPos } = this._corner( I, float( 1 ).add( float( 1.5 ).div( max( I.rPx, 1e-3 ) ) ) );
			vSphere.assign( vec4( I.center, I.R ) );
			vPoint.assign( vec4( viewPos, I.fade ) );
			vQuad.assign( vec4( q, I.ripple, I.glow ) );
			vColor.assign( I.col );
			vState.assign( I.st );
			if ( velocity && ! shadow ) vDelta.assign( I.pos.xyz.sub( buffers.nodePrev.element( instanceIndex ).xyz ) );
			// positionNode is local space; the mesh transform is the identity. Camera inside the sphere: collapse.
			// Single-sample variants: a dot that has faded to (almost) the backdrop colour would only punch a
			// hole into what is behind it, so it is dropped (alpha-to-coverage gives it no sample either)
			const off = a2c || shadow ? I.inside : I.inside.or( I.fade.lessThan( 0.04 ) );
			return viewToWorld( select( off, I.center, viewPos ) );

		} )();

		material.positionNode = position( false );
		material.castShadowPositionNode = position( true );

		// ── fragment: the surface ──
		const s = raySphere( vPoint.xyz, vSphere.xyz, vSphere.w );
		const hit = s.hit.toVar( 'nodeHit' );
		const normal = s.normal.toVar( 'nodeNormal' );
		const viewDir = s.viewDir;
		const coverage = discCoverage( vQuad.xy ).toVar( 'nodeCoverage' );
		const fade = vPoint.w;

		if ( a2c ) {

			// alpha-to-coverage turns the ramp (and the minimum-size fade) into sub-sample coverage
			material.maskNode = coverage.mul( fade ).greaterThan( 0.002 );
			material.opacityNode = coverage.mul( fade );

		} else {

			// solid dot; the minimum-size fade is applied to the colour in finish() below
			material.maskNode = coverage.greaterThan( 0.5 );

		}

		material.maskShadowNode = coverage.greaterThan( 0.5 );

		// ── fragment: the look ──
		const sel = vState.x, nb = vState.y, hov = vState.w;
		const dim = min( vState.z, focus ).mul( hov.oneMinus() ).toVar( 'nodeDim' ); // hovering lifts the dimming
		// partial luminance normalization: bright palettes (aurora, rainbow) and dark hues (blue) end
		// up in a similar brightness band, so neither the lighting nor the bloom favours some clusters.
		// Dark hues are lifted more than bright ones are lowered: a darkened yellow reads as olive
		const lum = luminance( vColor.xyz ).toVar( 'nodeLum' );
		const base = vColor.xyz.mul( mix( float( 1 ), float( 0.28 ).div( max( lum, 0.04 ) ), select( lum.greaterThan( 0.28 ), float( 0.22 ), float( 0.45 ) ) ) ).toVar( 'nodeBase' );
		const albedo = min( base, vec3( 1.2 ) );
		// on-screen radius in target pixels, and the roughness below which the highlight is smaller than a
		// pixel (see "Specular anti-aliasing" in the header)
		const rTarget = vSphere.w.mul( pixelsPerUnit( vSphere.z ) ).toVar( 'nodeRPx' );
		const roughFloor = sqrt( clamp( float( 0.6 ).div( max( rTarget, 1e-3 ) ), 0, 0.6 ) ).toVar( 'nodeRoughFloor' );
		// larger (higher-degree) nodes are a little glossier, small ones more matte; context keeps a little gloss
		const rough = clamp( U.roughness.add( U.variation.mul( float( 1.1 ).sub( vColor.w ) ).mul( 0.3 ) ), 0.08, 1 );
		const roughness = max( mix( rough, float( 0.5 ), dim.mul( 0.8 ) ), roughFloor );

		const ndv = clamp( dot( normal, viewDir ), 0, 1 ).toVar( 'nodeNdV' );
		// the core sits opposite the key light, where light that entered a glass bead would leave it again:
		// highlight on one side, saturated glow on the other, which reads as a lit volume instead of a disc
		const keyV = normalize( cameraViewMatrix.mul( vec4( G.keyDir, 0 ) ).xyz ).toVar( 'nodeKeyDir' ); // toward the light
		const core = pow( clamp( dot( normal, normalize( viewDir.sub( keyV.mul( U.coreShift ) ) ) ), 0, 1 ), 3 ).toVar( 'nodeCore' );
		const rim = pow( ndv.oneMinus(), 2.6 );
		// emissive core, luminance equalized a bit further (HDR, feeds the bloom)
		const baseLum = luminance( base );
		const coreColor = base.mul( mix( float( 1 ), float( 0.3 ).div( max( baseLum, 0.04 ) ), select( baseLum.greaterThan( 0.3 ), float( 0.15 ), float( 0.35 ) ) ) );
		// seen from afar a node is a luminous dot, close up the glow steps back and the surface shows
		const sizePx = rTarget.div( G.pxScale ); // on-screen radius, CSS px
		const closeUp = mix( float( 1 ), float( 0.55 ), smoothstep( 10, 90, sizePx ) );
		// context nodes stop glowing: a dimmed core in a darkened hue is what reads as mud
		const lit01 = dim.mul( 0.85 ).oneMinus();
		const gain = float( 1 ).add( sel.mul( 0.5 ) ).add( hov.mul( 0.4 ) ).add( nb.mul( 0.3 ) ).add( vQuad.w.mul( U.glow ) ).mul( closeUp ).mul( lit01 );
		const tint = mix( base, vec3( 1 ), 0.22 ).mul( 1.25 );
		// ripple: more of the node's own light (a hint of white only), so the hues on the front stay apart
		const flash = mix( coreColor, vec3( 1 ), 0.08 ).mul( vQuad.z ).mul( U.flash ).mul( core.mul( 0.7 ).add( 0.3 ) );
		const emissive = coreColor.mul( core ).mul( U.emissive ).mul( gain )
			.add( tint.mul( rim ).mul( U.rim ).mul( sel.mul( 0.6 ).add( 1 ) ) )
			.add( flash );

		// the key light's shadow term as three computed it for this fragment (1 = lit … 1 − shadow intensity);
		// stays 1 while the light casts no shadows (the hook below is then never called)
		const shadowTerm = property( 'float', 'nodeShadow' );

		// final colour: cast shadow, emphasis, context dimming (cool, dark, nearly hue-free), fog, minimum-size fade
		const finish = ( rgb ) => {

			if ( lit ) rgb = rgb.mul( shadowTerm.oneMinus().mul( smoothstep( 0, 0.35, dot( normal, keyV ) ) ).mul( U.shade ).oneMinus() );
			const boost = float( 1 ).add( sel.mul( 0.25 ) ).add( hov.mul( 0.2 ) ).add( nb.mul( 0.1 ) );
			const c = mix( rgb, rgb.add( vec3( luminance( rgb ).mul( 0.6 ) ) ), sel.mul( 0.15 ) ).mul( boost );
			// context: a cool dark grey with only a trace of the hue (a darkened yellow or orange reads as
			// olive / brown), so it recedes behind the focus without looking dirty
			const gray = vec3( 0.40, 0.47, 0.62 ).mul( luminance( c ) );
			const d = mix( c, gray, dim.mul( 0.92 ) ).mul( float( 1 ).sub( dim.mul( 0.6 ) ) );
			const fogged = applyFog( d, length( hit ) );
			// single-sample variants: what the coverage fade of a sub-pixel node averages to over the backdrop
			return a2c ? fogged : mix( vec3( G.fogColor ), fogged, fade );

		};

		let hitWorld;
		if ( lit ) {

			hitWorld = applySurface( material, { hit, normal, viewDir } );
			// the colour is the first thing three evaluates in the fragment: reset the shadow term there
			material.colorNode = Fn( () => {

				shadowTerm.assign( 1 );
				return vec4( albedo, 1 );

			} )();
			material.receivedShadowNode = ( shadow ) => {

				shadowTerm.assign( float( shadow ) );
				return shadow;

			};
			material.roughnessNode = roughness;
			material.metalnessNode = U.metalness;
			material.emissiveNode = emissive;
			material.finish = finish;
			material.finishKey = 'node';
			if ( shading === 'phys' ) {

				material.clearcoatNode = U.clearcoat.mul( dim.mul( 0.8 ).oneMinus() );
				material.clearcoatRoughnessNode = max( U.clearcoatRoughness, roughFloor );
				material.clearcoatNormalNode = normal; // the default reads the geometry normal attribute (there is none)

			}

		} else {

			// analytic stand-in for the lit path, no texture taps: hemisphere irradiance, the key light
			// (Lambert + normalized Blinn-Phong with the GGX-equivalent exponent) and a fresnel
			// reflection of the hemisphere. Units as three's lights: irradiance · albedo / π.
			hitWorld = viewToWorld( hit );
			const L = keyV;
			const upV = cameraViewMatrix.mul( vec4( 0, 1, 0, 0 ) ).xyz.toVar( 'nodeUp' );
			const hemi = ( d ) => mix( vec3( G.ambientGround ), vec3( G.ambientSky ), dot( d, upV ).mul( 0.5 ).add( 0.5 ) );
			const ndl = clamp( dot( normal, L ), 0, 1 );
			const diffuse = albedo.mul( 1 / Math.PI ).mul( vec3( G.keyColor ).mul( ndl ).add( hemi( normal ) ) );
			const a2 = pow( roughness, 4 );
			const n = float( 2 ).div( max( a2, 1e-4 ) ).sub( 2 ).clamp( 1, 2000 );
			const H = normalize( L.add( viewDir ) );
			const F = float( 0.04 ).add( float( 0.96 ).mul( pow( ndv.oneMinus(), 5 ) ) );
			const spec = vec3( G.keyColor ).mul( pow( max( dot( normal, H ), 0 ), n ).mul( n.add( 8 ) ).mul( 1 / ( 8 * Math.PI ) ) ).mul( ndl ).mul( 0.04 );
			const refl = viewDir.negate().add( normal.mul( ndv.mul( 2 ) ) ); // reflect( −V, N )
			const envSpec = hemi( refl ).mul( 1 / Math.PI ).mul( F ).mul( roughness.oneMinus() );
			material.colorNode = vec4( finish( diffuse.add( spec ).add( envSpec ).add( emissive ) ), 1 );
			material.depthNode = viewToDepth( hit );

		}

		if ( velocity ) material.mrtNode = velocityOutput( motionVector( hitWorld, hitWorld.sub( vDelta ) ) );

		return material;

	}

	// ── halo pass ───────────────────────────────────────────────────────────────────────────

	_haloMaterial( buffers, { velocity = false } = {} ) {

		const px = gfxUniforms.pxScale;
		const time = buffers.uniforms.time;

		const vHalo = varyingProperty( 'vec4', 'vHaloQuad' ); // quad coords in target px, silhouette radius px, inscribed quad radius px
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
		// additive: leave the motion vector of what is behind it. Only while the scene pass has a velocity
		// attachment: a material with an mrtNode cannot be drawn into a target without an `output` texture
		if ( velocity ) material.mrtNode = skipVelocity();

		// ring radius / width in target px for a silhouette of rPx px (mirrored by haloRadius())
		const ringR = ( rPx, sel ) => rPx.mul( 1.3 ).add( px.mul( float( 2.5 ).add( sel.mul( 1.5 ) ) ) );
		const ringW = ( sel ) => px.mul( float( 1.0 ).add( sel.mul( 0.8 ) ) );

		material.positionNode = Fn( () => {

			const I = this._instance( buffers, { fx: false } );
			const k = ringR( I.rPx, I.st.x ).add( ringW( I.st.x ).mul( 3 ) ).add( px.mul( 8 ) ).div( max( I.rPx, 1e-3 ) ).toVar( 'nExtent' );
			const { q, viewPos } = this._corner( I, k );
			// selected: full; hovered: strong; neighbors of the selection: mild
			const strength = max( I.st.x, max( I.st.y.mul( 0.5 ), I.st.w.mul( 0.8 ) ) ).toVar( 'nStrength' );
			vHalo.assign( vec4( q.mul( I.rPx ), I.rPx, I.rPx.mul( k ) ) );
			vColor.assign( vec4( I.col.xyz, strength.mul( I.fade.mul( 0.6 ).add( 0.4 ) ).mul( fogTransmittance( length( I.center ) ) ) ) );
			vSel.assign( max( I.st.x, I.st.w.mul( 0.5 ) ) );
			const off = I.inside.or( strength.lessThan( 0.004 ) );
			return viewToWorld( select( off, I.center, viewPos ) );

		} )();

		material.colorNode = Fn( () => {

			const r = length( vHalo.xy );
			const rPx = vHalo.z, edgePx = vHalo.w;
			const sel = vSel;
			const rr = ringR( rPx, sel );
			const hw = ringW( sel ).mul( 0.5 );
			// crisp ring: 1 px anti-aliased edges around a constant pixel width
			const ring = clamp( hw.add( 0.5 ).sub( abs( r.sub( rr ) ) ), 0, 1 );
			// faint corona from the silhouette out to the ring, soft glow around the ring;
			// both windowed to reach exactly 0 before the quad border
			const outside = smoothstep( rPx.sub( 0.5 ), rPx.add( 1.0 ), r );
			const win = smoothstep( edgePx, edgePx.mul( 0.7 ), r );
			const corona = exp( r.sub( rPx ).max( 0 ).negate().div( rPx.mul( 0.3 ).add( px.mul( 2.5 ) ) ) ).mul( 0.16 );
			const glow = exp( abs( r.sub( rr ) ).negate().div( px.mul( 2.2 ) ) ).mul( 0.22 );
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

		}

		for ( const material of this._materials.values() ) material.dispose();
		this._materials.clear();
		this.mesh = null;
		this.haloMesh = null;
		this.variant = '';

	}

	dispose() {

		this.app?.events.removeEventListener( 'gfx', this._onGfx );
		this._disposeMeshes( this.app );
		this._geometry.dispose();

	}

}
