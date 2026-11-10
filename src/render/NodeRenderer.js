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
//  (+ nodePrev); shadow pass vertex 3; halo vertex 4, 5 with 'mv'; fragment 0 in every variant.
//
// ─── Size ───────────────────────────────────────────────────────────────────────────────────
//  radius = baseSize · nodeColor.w · size · (1 + 0.5·selected + 0.25·hovered) · (1 − 0.4·context)
//  (context = the eased dimming of nodes outside a selection's neighbourhood, lifted by hovering;
//  the shadow pass and worldRadius() keep the rest size)
//  baseSize = 0.065 · graph.layoutRadius / ∛N. A node whose radius falls below `minPixelRadius`
//  (CSS px) is not shrunk into sub-pixel flicker; it turns into a POINT with an energy-conserving
//  response: with f = true radius / minimum radius, the true disc covers f² of the minimum dot, so
//  the node sends f² of the light a dot of the minimum size would (look.point = 1; 0 gives the
//  linear fade f, which lets a sub-pixel node send 1 / f times the light it has).
//   'a2c' (MSAA)   the sphere stays, drawn at the minimum size, and alpha-to-coverage gives it
//                  that share of the samples: its colour over a share of the pixel, what lies
//                  behind it over the rest.
//   'cut' (single sample: TRAA, SMAA, FXAA)   an opaque dot cannot be partly there. Fading its
//                  colour toward the backdrop (the stage-1 approach) is right over empty space,
//                  but over the luminous edge mass of a large graph it leaves a dark, depth
//                  writing speck per node: an overview of 20k nodes was pepper on colour. So the
//                  sphere is dropped below the minimum size and the node is drawn by the halo
//                  pass as a blended POINT SPRITE: a Gaussian (σ = 0.6 minimum radii) whose value
//                  is the node's COVERAGE of the pixel and whose integral is the area of the true
//                  disc, f² · π · minimum radius². The pixel becomes coverage · C + ( 1 − coverage ) · what is
//                  behind it, C being the mean radiance of the bead (emissive core + key and
//                  ambient light on the enamel, emphasis and context dimming as on the sphere,
//                  fogged): exactly what alpha-to-coverage averages to. Sprites that pile up
//                  converge to the bead colour instead of adding up without bound (20k of them
//                  blew the bloom out when they were additive), strands behind them are covered
//                  by that share. A sprite has no silhouette to alias (nothing crawls or pops
//                  under the TRAA jitter: a point that moves by a sub-pixel changes by a
//                  fraction), widens with the shared circle of confusion (σ² + coc² / 4, coverage
//                  scaled down: defocused points turn into faint discs like the strands do), and
//                  writes its own motion vector in the velocity tiers: over the disc of the
//                  minimum radius (what the opaque dot it replaces owned), for every sprite that
//                  covers at least SPRITE_OWN (4 %) at its centre, i.e. down to a fifth of the
//                  minimum radius. The TRAA resolve has no other source for these pixels (sprites
//                  write no depth, strands no vector): without it the history of an overview is
//                  taken from the same pixel and the whole cloud softens under camera motion.
//                  Sprites that cover less than 0.5 % are dropped. The hand-over at f = 1 keeps
//                  size and mean colour (calibrated against true-size spheres: total light within 5 %).
//  In an overview of a large graph the leaves therefore are a fine dust of light and the hubs,
//  still near their true size, stand out: brightness follows size, as it does for the beads close
//  up. Important nodes additionally get a soft point glow (the halo pass, below) while they are
//  small, so the hierarchy also reads where no disc is resolved.
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
//  stronger, context nodes (state.z, gated by uniforms.focus) recede into small, dark, matte beads
//  (60 % of their radius, their own glow goes out, hue almost gone, no gloss and no clearcoat: a
//  full-size glossy black marble in front of a highlighted tube or a label reads as an object of
//  its own; a cool rim keeps the form). They stay opaque and depth writing.
//  nodeFx: x = hop distance of the ripple (gain = rippleGain·exp(−((x−front)/width)²), a flash of
//  the node's OWN core colour plus the scale-up, so clusters stay distinguishable on the front),
//  z = glow weight (extra emissive). Fog (gfx/shared.js) is applied to the final colour, emissive
//  included.
//  A second, blended pass draws the "halo": a crisp ring plus a faint corona around the selected
//  node (strong), the hovered node and the neighbours of the selection (mild). The halo of the
//  SELECTED node is drawn in front of everything (its quad is pulled to the near plane along the
//  view rays, which keeps its place and size on screen): the ring marks the selection and must not
//  be cut by the tubes that run into the node or by a node in front of it. Also the point sprites
//  of the single-sample variants (see "Size"); and the point glow: a small Gaussian in the node's
//  hue around nodes that are IMPORTANT (radius factor above the average: degree, or the size
//  lens; or the glow lens, nodeFx.z) and SMALL on screen (it is gone once the bead is 7 CSS px in
//  radius and shows its own core). Strength look.pointGlow, × 0.7 in the medium tier, off in the
//  low tier; context nodes have none. The pass writes no depth; instances without highlight, glow
//  and sprite are collapsed in the vertex shader. Cost in a 20k-node overview: a sprite quad of
//  about 7 × 7 CSS px per node (1 MP of additive fragments at 1080p, a Gaussian each) and a glow
//  quad of ~20 × 20 CSS px for about a tenth of the nodes.
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
//                   shade, flash, glow, rippleScale, variation, point, pointGlow } )   // any subset, live
//  nodes.look                   // current values (plain object copy)
//  nodes.variant                // key of the active material variant, e.g. 'std|a2c|still'
//  nodes.setGraph( buffers, graph ), nodes.dispose()
import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, vec2, vec3, vec4, float, uniform, varyingProperty, property, positionGeometry, cameraViewMatrix, cameraNear,
	normalize, length, dot, max, min, clamp, pow, mix, smoothstep, select, abs, luminance, exp, sin, sqrt, step, mrt, output,
} from 'three/tsl';
import { gfxUniforms, gfxState, motionVector, velocityOutput, applyFog, fogTransmittance, cocPixels } from './gfx/shared.js';
import {
	pixelsPerUnit, viewToDepth, viewToWorld, applySurface, sphereBillboard, raySphere, discCoverage, isOrthographic,
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
	point: 1, // 0..1: fade of nodes below the minimum size, 0 = linear in the radius, 1 = energy conserving (area)
	pointGlow: 0.6, // strength of the soft glow around important nodes while they are dots
};

/** Share of the emissive core's peak that a bead averages to over its disc (point sprites). */
export const POINT_CORE = 0.35;

/** Share of its radius a context node loses at full dimming. */
export const CONTEXT_SHRINK = 0.4;

/** Emissive gain and overall boost of the selected node (hovered and neighbour nodes get less). */
const SEL_GAIN = 0.25, SEL_BOOST = 0.12;

/**
 * σ of a point sprite in minimum radii. 1 / √2 would put the coverage f² at the centre; 0.6 gives a
 * tighter point with 1.39 · f² there (limited to 1: nodes near the minimum size get a flat top,
 * i.e. nearly the disc they are about to become).
 */
export const POINT_SIGMA = 0.6;

/** Centre coverage from which a point sprite writes its motion vector (over the disc of the minimum radius). */
export const SPRITE_OWN = 0.04;

/** Factor on look.pointGlow per quality tier. */
export const POINT_GLOW_TIER = { low: 0, medium: 0.7, high: 1, ultra: 1 };

/**
 * Point sprite of a node below the minimum size: the CPU twin of the halo pass's vertex stage (tests,
 * tools). A Gaussian of variance σ² = ( POINT_SIGMA · minPx )² + cocPx² / 4 whose centre coverage is
 * chosen so that the coverage integrated over the sprite, 2πσ² · cover, is the area that the fade
 * leaves of the minimum disc: π · minPx² · fade with fade = f · mix( 1, f, point ), f = truePx / minPx.
 * With point = 1 that is π · truePx², the true disc of the node: its light does not depend on the
 * zoom or on the focus. The fragment stage limits the coverage to 1.
 * @param {number} truePx true radius in pixels
 * @param {number} minPx minimum radius in pixels (minPixelRadius)
 * @param {number} [cocPx] diameter of the circle of confusion in pixels
 * @param {number} [point] look.point
 * @returns {{ sigma: number, cover: number, fade: number }} σ in pixels, coverage at the centre
 */
export function pointSprite( truePx, minPx, cocPx = 0, point = 1 ) {

	const f = Math.min( 1, Math.max( 0, truePx / minPx ) );
	const fade = f * ( 1 + ( f - 1 ) * point );
	const var2 = ( POINT_SIGMA * minPx ) ** 2 + cocPx * cocPx / 4;
	return { sigma: Math.sqrt( var2 ), cover: fade * minPx * minPx / ( 2 * var2 ), fade };

}

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
			pointGlowTier: uniform( 1 ), // POINT_GLOW_TIER of the active tier
			sprites: uniform( 1 ), // 1: nodes below the minimum size are blended point sprites (single-sample variants)
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
		this.uniforms.pointGlowTier.value = POINT_GLOW_TIER[ gfxState.tier ] ?? 1;
		// with MSAA the sphere fades by coverage; without the halo pass nobody would draw the sprites
		this.uniforms.sprites.value = gfxState.samples > 1 || ! this._halo ? 0 : 1;

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
		// hover grows a node; context nodes (outside a selection's neighbourhood) shrink out of the way
		if ( ! shadow ) scale = scale.add( st.w.mul( 0.25 ) ).mul( min( st.z, S.focus ).mul( st.w.oneMinus() ).mul( CONTEXT_SHRINK ).oneMinus() );
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
		let R = R0, fade = float( 1 ), sub = float( 1 );
		if ( ! shadow ) {

			const minPx = U.minPx.mul( gfxUniforms.pxScale );
			R = max( R0, minPx.div( ppu ) ).toVar( 'nR' );
			// share of the minimum dot the true disc covers: f in the radius, f² in the area (see "Size")
			sub = clamp( R0.mul( ppu ).div( max( minPx, 1e-3 ) ), 0, 1 ).toVar( 'nSub' );
			fade = sub.mul( mix( float( 1 ), sub, U.point ) );

		}

		const bb = sphereBillboard( center, R );
		const h = bb.h.toVar( 'nH' );
		const rPx = h.mul( ppu ).toVar( 'nRPx' ); // silhouette radius in target pixels
		return { pos, col, st, center, R, fade, sub, h, rPx, ripple, glow, right: bb.right, up: bb.up, inside: bb.inside };

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
			// Single-sample variants: below the minimum size the node is a point sprite of the halo pass
			// ( uniforms.sprites; with the halo pass switched off it stays a disc of the minimum size )
			const off = a2c || shadow ? I.inside : I.inside.or( I.sub.lessThan( 0.999 ).and( U.sprites.greaterThan( 0.5 ) ) );
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
		const fade = vPoint.w; // 'a2c' only

		if ( a2c ) {

			// alpha-to-coverage turns the ramp (and the minimum-size fade) into sub-sample coverage
			material.maskNode = coverage.mul( fade ).greaterThan( 0.002 );
			material.opacityNode = coverage.mul( fade );

		} else {

			// solid sphere; below the minimum size the halo pass draws the node (the vertex stage drops it here)
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
		// context nodes lose their gloss with the dimming (a dark glossy bead still catches the eye)
		const roughness = max( mix( rough, float( 1 ), dim ), roughFloor );

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
		// emphasis adds a little light only: under the Neutral tone mapper a saturated colour that is pushed
		// far above 1 turns pastel (a selected red bead was pink), and the ring, the corona and the growth
		// already say "selected". The hue is the identity of the node and stays
		const gain = float( 1 ).add( sel.mul( SEL_GAIN ) ).add( hov.mul( 0.2 ) ).add( nb.mul( 0.2 ) ).add( vQuad.w.mul( U.glow ) ).mul( closeUp ).mul( lit01 );
		const tint = mix( base, vec3( 1 ), 0.22 ).mul( 1.25 );
		// ripple: more of the node's own light (a hint of white only), so the hues on the front stay apart
		const flash = mix( coreColor, vec3( 1 ), 0.08 ).mul( vQuad.z ).mul( U.flash ).mul( core.mul( 0.7 ).add( 0.3 ) );
		const emissive = coreColor.mul( core ).mul( U.emissive ).mul( gain )
			.add( tint.mul( rim ).mul( U.rim ).mul( sel.mul( 0.6 ).add( 1 ) ) )
			.add( flash );

		// the key light's shadow term as three computed it for this fragment (1 = lit … 1 − shadow intensity);
		// stays 1 while the light casts no shadows (the hook below is then never called)
		const shadowTerm = property( 'float', 'nodeShadow' );

		// final colour: cast shadow, emphasis, context dimming (cool, dark, nearly hue-free), fog
		const finish = ( rgb ) => {

			if ( lit ) rgb = rgb.mul( shadowTerm.oneMinus().mul( smoothstep( 0, 0.35, dot( normal, keyV ) ) ).mul( U.shade ).oneMinus() );
			const c = rgb.mul( float( 1 ).add( sel.mul( SEL_BOOST ) ).add( hov.mul( 0.1 ) ).add( nb.mul( 0.06 ) ) );
			// context: a cool dark grey with only a trace of the hue (a darkened yellow or orange reads as
			// olive / brown), so it recedes behind the focus without looking dirty
			const gray = vec3( 0.40, 0.47, 0.62 ).mul( luminance( c ) );
			const d = mix( c, gray, dim.mul( 0.92 ) ).mul( float( 1 ).sub( dim.mul( 0.6 ) ) );
			return applyFog( d, length( hit ) );

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

				material.clearcoatNode = U.clearcoat.mul( dim.oneMinus() );
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

		const px = gfxUniforms.pxScale, G = gfxUniforms;
		const time = buffers.uniforms.time;

		const focus = buffers.uniforms.focus;
		const U = this.uniforms;
		const vHalo = varyingProperty( 'vec4', 'vHaloQuad' ); // quad coords in target px, silhouette radius px, inscribed quad radius px
		const vColor = varyingProperty( 'vec4', 'vHaloColor' ); // rgb, strength
		const vSel = varyingProperty( 'vec2', 'vHaloSel' ); // selection weight, point glow strength
		const vSprite = varyingProperty( 'vec4', 'vHaloSprite' ); // point sprite: mean radiance of the bead, 1 / ( 2 σ² ) in target px
		const vCover = varyingProperty( 'float', 'vHaloCover' ); // point sprite: coverage at its centre
		const vMotion = varyingProperty( 'vec2', 'vHaloMotion' ); // NDC motion of the node since the previous frame

		const material = new THREE.NodeMaterial();
		material.name = 'NodeRenderer.halo';
		material.lights = false;
		material.fog = false;
		material.transparent = true;
		material.depthWrite = false;
		material.depthTest = true;
		// rgb + dst · ( 1 − a ): rings and glows are light ( a = 0 ), a point sprite is the node itself
		// and covers its share of what is behind it ( a = coverage ). Custom blending, because three
		// multiplies the colour of a `premultipliedAlpha` material by its alpha
		material.blending = THREE.CustomBlending;
		material.blendEquation = THREE.AddEquation;
		material.blendSrc = material.blendSrcAlpha = THREE.OneFactor;
		material.blendDst = material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;

		// ring radius / width in target px for a silhouette of rPx px (mirrored by haloRadius())
		const ringR = ( rPx, sel ) => rPx.mul( 1.3 ).add( px.mul( float( 2.5 ).add( sel.mul( 1.5 ) ) ) );
		const ringW = ( sel ) => px.mul( float( 1.0 ).add( sel.mul( 0.8 ) ) );
		// width of the point glow in target px for a dot of rPx px
		const pointSigma = ( rPx ) => rPx.mul( 1.2 ).add( px.mul( 1.5 ) );

		material.positionNode = Fn( () => {

			const I = this._instance( buffers );
			// selected: full; hovered: strong; neighbors of the selection: mild
			const strength = max( I.st.x, max( I.st.y.mul( 0.5 ), I.st.w.mul( 0.8 ) ) ).toVar( 'nStrength' );
			const dist = length( I.center ).toVar( 'nDist' );
			const fog = fogTransmittance( dist ).toVar( 'nFog' );
			const dim = min( I.st.z, focus ).mul( I.st.w.oneMinus() ).toVar( 'nDim' );

			// point sprite: the node itself while it is below the minimum size (single-sample variants).
			// Peak = mean radiance of the bead × the share of the minimum dot its true disc covers
			const below = step( I.sub, 0.999 ).mul( U.sprites );
			const minR = U.minPx.mul( px );
			const sigma0 = minR.mul( POINT_SIGMA );
			const coc = cocPixels( I.center.z.negate() );
			const var2 = sigma0.mul( sigma0 ).add( coc.mul( coc ).mul( 0.25 ) ).toVar( 'nVar' ); // σ² in target px²
			// coverage at the centre: the Gaussian holds what the true disc covers, f² · π · minimum radius²
			const cover = below.mul( I.fade ).mul( minR.mul( minR ).div( var2.mul( 2 ) ) ).toVar( 'nCover' );
			const radiance = applyFog( this._pointRadiance( I, dim ), dist ).toVar( 'nSprite' );
			const sprite = cover.greaterThan( 0.005 );
			const spriteExtent = sqrt( var2 ).mul( 3 );

			// point glow: important nodes (radius factor above the average, or the glow lens) while they are dots
			const importance = clamp( smoothstep( 1, 1.6, I.col.w ).add( I.glow.mul( 0.8 ) ), 0, 1.5 );
			const dots = smoothstep( 7, 2.5, I.rPx.div( px ) );
			const point = importance.mul( dots ).mul( U.pointGlow ).mul( U.pointGlowTier ).mul( dim.oneMinus() ).mul( fog ).toVar( 'nPoint' );
			const glowExtent = select( point.greaterThanEqual( 0.01 ), pointSigma( I.rPx ).mul( 3 ), float( 0 ) );

			// the quad reaches whatever it has to show: ring and its glow, point glow ( 3 σ ), sprite ( 3 σ )
			const ringed = ringR( I.rPx, I.st.x ).add( ringW( I.st.x ).mul( 3 ) ).add( px.mul( 8 ) );
			const extent = max( select( strength.lessThan( 0.004 ), float( 0 ), ringed ), max( glowExtent, select( sprite, spriteExtent, float( 0 ) ) ) );
			const k = extent.div( max( I.rPx, 1e-3 ) ).toVar( 'nExtent' );
			const { q, viewPos } = this._corner( I, k );
			vHalo.assign( vec4( q.mul( I.rPx ), I.rPx, I.rPx.mul( k ) ) );
			vColor.assign( vec4( I.col.xyz, strength.mul( I.fade.mul( 0.6 ).add( 0.4 ) ).mul( fog ) ) );
			vSel.assign( vec2( max( I.st.x, I.st.w.mul( 0.5 ) ), point ) );
			vSprite.assign( vec4( radiance, float( 0.5 ).div( var2 ) ) );
			vCover.assign( select( sprite, cover, float( 0 ) ) );
			if ( velocity ) vMotion.assign( motionVector( I.pos.xyz, buffers.nodePrev.element( instanceIndex ).xyz ) );
			const off = I.inside.or( extent.lessThan( 0.5 ) );
			// the selected node's halo goes in front of everything: scaled along the view rays to just
			// behind the near plane (same place and size on screen; perspective cameras only)
			const front = clamp( cameraNear.mul( 1.02 ).div( max( viewPos.z.negate(), 1e-6 ) ), 0, 1 );
			const pull = select( I.st.x.greaterThan( 0.02 ).and( isOrthographic().not() ), front, float( 1 ) );
			return viewToWorld( select( off, I.center, viewPos.mul( pull ) ) );

		} )();

		const r2 = dot( vHalo.xy, vHalo.xy ).toVar( 'haloR2' );
		// the sprite: coverage of the node at this pixel, shifted down so that it is exactly 0 at the border of its quad ( 3 σ )
		const spriteCover = exp( r2.mul( vSprite.w ).negate() ).sub( 0.0111 ).max( 0 ).mul( vCover ).min( 1 ).toVar( 'haloSprite' );

		material.colorNode = Fn( () => {

			const r = sqrt( r2 );
			const rPx = vHalo.z, edgePx = vHalo.w;
			const sel = vSel.x;
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
			const pulse = sin( time.mul( 2.4 ) ).mul( 0.1 ).mul( sel ).add( 1 );
			// neighbors keep their hue, the selection goes towards white-hot
			const tint = mix( vColor.xyz, vec3( 1 ), sel.mul( 0.45 ).add( 0.12 ) );
			const a = ring.mul( 0.95 ).add( corona.add( glow ).mul( win ) ).mul( outside ).mul( vColor.w ).mul( pulse );
			// point glow: a Gaussian in the node's hue, equalised in luminance like the core, 0 at the quad border
			const sigma = pointSigma( rPx );
			const lum = max( luminance( vColor.xyz ), 0.04 );
			const hue = vColor.xyz.mul( mix( float( 1 ), float( 0.3 ).div( lum ), 0.4 ) );
			const gauss = exp( r2.div( sigma.mul( sigma ).mul( - 2 ) ) ).sub( 0.0111 ).max( 0 ).mul( win ); // exp( −4.5 ) at 3 σ
			return vec4( tint.mul( a ).mul( float( 1.5 ).add( sel.mul( 0.9 ) ) ).add( hue.mul( gauss ).mul( vSel.y ) ).add( vSprite.xyz.mul( spriteCover ) ), spriteCover );

		} )();

		// additive: ring and glows leave the motion vector of what is behind them; a point sprite owns
		// the disc of the minimum radius (the velocity attachment blends per pixel: alpha 1 sets the
		// vector), so the history follows the node, however faint it is, and the strands that end in it.
		// Only while the scene pass has a velocity attachment: a material with an mrtNode cannot be
		// drawn into a target without an `output` texture
		if ( velocity ) material.mrtNode = mrt( { output, velocity: vec4( vMotion, 0, step( SPRITE_OWN, vCover ).mul( step( r2, vHalo.z.mul( vHalo.z ) ) ) ) } );

		return material;

	}

	/**
	 * Mean radiance of a bead that is too small to show its surface (vertex stage): what the sphere
	 * shader averages to over the disc. Emissive core (the core lobe averages to about a third over
	 * the disc) with the emphasis gains, the enamel under the key light and the hemisphere, context
	 * dimming as in the sphere's finish().
	 */
	_pointRadiance( I, dim ) {

		const U = this.uniforms, G = gfxUniforms;
		const col = I.col.xyz, st = I.st;
		const lum = luminance( col );
		const base = col.mul( mix( float( 1 ), float( 0.28 ).div( max( lum, 0.04 ) ), select( lum.greaterThan( 0.28 ), float( 0.22 ), float( 0.45 ) ) ) ).toVar( 'nBase' );
		const baseLum = luminance( base );
		const coreColor = base.mul( mix( float( 1 ), float( 0.3 ).div( max( baseLum, 0.04 ) ), select( baseLum.greaterThan( 0.3 ), float( 0.15 ), float( 0.35 ) ) ) );
		const gain = float( 1 ).add( st.x.mul( SEL_GAIN ) ).add( st.w.mul( 0.2 ) ).add( st.y.mul( 0.2 ) ).add( I.glow.mul( U.glow ) ).mul( dim.mul( 0.85 ).oneMinus() );
		const light = vec3( G.keyColor ).mul( 0.3 ).add( vec3( G.ambientSky ).add( vec3( G.ambientGround ) ).mul( 0.5 ) ).mul( 1 / Math.PI );
		const boost = float( 1 ).add( st.x.mul( SEL_BOOST ) ).add( st.w.mul( 0.1 ) ).add( st.y.mul( 0.06 ) );
		const c = coreColor.mul( U.emissive ).mul( POINT_CORE ).mul( gain ).add( min( base, vec3( 1.2 ) ).mul( light ) ).mul( boost ).toVar( 'nMean' );
		const gray = vec3( 0.40, 0.47, 0.62 ).mul( luminance( c ) );
		return mix( c, gray, dim.mul( 0.92 ) ).mul( float( 1 ).sub( dim.mul( 0.6 ) ) );

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
