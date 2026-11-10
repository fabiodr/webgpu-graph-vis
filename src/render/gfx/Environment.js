// Environment: everything around the graph that lights it or sits behind it (App module, order 12).
//
//  • Sky → scene.environment. A tiny scene (one inside-out sphere whose colour is a TSL function of
//    the direction) is rendered by PMREMGenerator.fromScene() into a 256² RGBA16F cube with a
//    GGX-prefiltered mip chain. The sky is never shown; it exists for image-based lighting: a dim
//    zenith-to-nadir gradient, a faint two-coloured nebula, an HDR key soft box around the key
//    light direction (the soft highlight on glossy spheres), a large cool fill opposite to it and a
//    broad, soft coloured rim panel around the horizon on the far side. It is regenerated only
//    when a preset or a sky parameter changed, always into the same render target (no material
//    is rebuilt). While gfxState.lit is false nothing samples it: it is not generated at all, and
//    the cube and the generator are freed when lit turns off (regenerated when it turns on).
//  • Backdrop → scene.backgroundNode: near-black radial gradient (brightest a little above the
//    screen centre) times a very faint, low-frequency pattern on the view direction plus a hint of
//    the nebula colours, so it turns with the camera like something far away. No fine detail (it
//    needs no motion vectors), drawn at the far plane by three whatever camera.far is. Values are
//    tiny (1–3 % linear); the post chain dithers after the sRGB encode.
//  • Key light: ONE world-fixed DirectionalLight (the sky's soft box is centred on its direction),
//    PCF shadows. The orthographic shadow camera is fitted to the bounding sphere of the node
//    positions (radius quantised to powers of 1.25 with hysteresis, centre snapped to shadow
//    texels, so nothing crawls while the layout moves). shadow.autoUpdate is off: the map is
//    re-rendered only in frames in which the positions changed (buffers.moveFrame), the fit, the
//    light, the map size, the node size or the highlight state changed (selection, the 'path' /
//    'lens' events, any write to nodeStateTarget other than a hover change; held for the easing),
//    or after updateShadow().
//    renderer.shadowMap.enabled follows gfxState.shadows (and gfxState.lit), the map size follows
//    gfxState.shadowMapSize, and light.castShadow additionally needs the map to RESOLVE the
//    nodes: a sphere of a few texels casts no shape, only PCF noise and a lost highlight, at the
//    cost of a shadow pass and 5 compare taps per lit fragment. stats.shadowTexels is the radius of
//    a size-1 node in shadow texels; the shadow strength fades in between 2.5 and 4.5 texels and
//    below 2.5 the light stops casting (shadowResolve(); with 2048² that is above roughly 3k
//    nodes, with 4096² above roughly 25k). The map is freed whenever the light stops casting.
//  • Shared uniforms (gfx/shared.js), kept in sync for cheap analytic consumers (unlit nodes, thin
//    strands): keyDir, keyColor (= light colour × intensity, three's light units: multiply by
//    N·L and the BRDF, Lambert = albedo / π), ambientSky / ambientGround (irradiance of the sky on
//    a surface facing up / down in the same units, environment intensity included), fogColor
//    (the backdrop colour distant things fade to), fogStart / fogDensity (see `fog`).
//  • The paper's faint latitude / longitude guide sphere that fades in with the sphere layout
//    (thin anti-aliased lines, additive, no depth write, opacity × uniforms.layoutBlend).
//
// ─── Public API ─────────────────────────────────────────────────────────────────────────────
//   const env = app.use( new Environment( { preset: 'aurora', guide: true } ) );   // order 12
//   env.setPreset( 'aurora' | 'studio' | 'void', { nodes = true } )   look preset; with `nodes`
//                                  the preset's node look hints go to NodeRenderer.setLook()
//   env.preset                     name of the active preset (read only)
//   env.keyIntensity               intensity of the key light (three's units; the sky's soft box
//                                  scales along). Changing it regenerates the sky.
//   env.envIntensity               scene.environmentIntensity (live uniform)
//   env.fog                        0..1 depth cue: 0 off, 1 = the far side of the graph keeps
//                                  about 40 % of its light. Starts a little in front of the
//                                  graph centre, density is relative to the graph radius.
//   env.setKeyDirection( v )       world-space direction TOWARD the light (Vector3 or [x,y,z])
//   env.keyDirection               Vector3 (read only; use setKeyDirection)
//   env.shadowSoftness             penumbra as a fraction of the base node radius (default 0.3)
//   env.light                      the THREE.DirectionalLight
//   env.updateShadow( seconds = 0 )   force a shadow map render in the next frame; with `seconds`
//                                  keep re-rendering for that long (eased changes of a caster, e.g.
//                                  highlight sizes that ease for ~0.2 s: use 0.9)
//   env.updateSky()                force a regeneration of the environment in the next frame
//   env.stats                      { shadowRenders, skyRenders, shadowRadius, shadowTexel (world
//                                  units), shadowTexels (node radius in texels), shadowStrength
//                                  (0..1 from shadowResolve), shadowCasting, boundsRadius, … }
//   env.guide = false; env.guideOpacity = 0.08; env.setColors( center, edge )   (as Background)
//   env.setGraph( buffers, graph, app ), env.update( dt, app ), env.dispose()
//
// Pure helpers (tested in test/environment.test.js): skyRadiance, skyIrradiance, quantizeRadius,
// fitShadowBox, shadowResolve, fogParameters, boundsOfPositions.
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, vec2, vec3, vec4, float, screenUV, screenSize, exp, mix, smoothstep, normalWorldGeometry,
	uv, fract, abs, fwidth, clamp, max, sin, frontFacing, positionGeometry, positionLocal, select, normalize,
	dot, length,
} from 'three/tsl';
import { gfxUniforms, gfxState, skipVelocity } from './shared.js';

// ── presets ──────────────────────────────────────────────────────────────────────────────────
// Colours are linear RGB. `key.light` is the DirectionalLight intensity, `key.box` the radiance of
// the sky's soft box at that intensity, `key.size` = [ cosine of the half angle, softness ].
// Backdrop values are pre-compensated for the Neutral tone mapper's toe. `shadow` is the shadow
// intensity of the key light (the nodes also dim their own light in a cast shadow, NODE_LOOK.shade).

export const ENV_PRESETS = {
	// the default: luminous nodes in a deep blue-violet space, warm key, cool fill, magenta rim
	aurora: {
		backdrop: { center: [ 0.0170, 0.0182, 0.0225 ], edge: [ 0.0066, 0.0070, 0.0080 ], noise: 0.22, tint: 0.0045 },
		sky: {
			zenith: [ 0.030, 0.050, 0.095 ], nadir: [ 0.006, 0.007, 0.014 ],
			fill: [ 0.25, 0.45, 1.0 ], fillIntensity: 0.55, fillSize: [ 0.35, 0.45 ],
			rim: [ 1.0, 0.3, 0.75 ], rimIntensity: 0.9,
			nebulaA: [ 0.45, 0.12, 0.60 ], nebulaB: [ 0.08, 0.50, 0.55 ], nebula: 0.10,
		},
		key: { color: [ 1.0, 0.94, 0.84 ], light: 3.6, box: 5, size: [ 0.972, 0.018 ] },
		envIntensity: 1.2,
		shadow: 0.8,
		fog: 0.45,
		nodes: { roughness: 0.26, emissive: 2, rim: 0.55 },
	},
	// neutral product-shot lighting: larger white soft box, grey sweep, stronger key-to-fill ratio
	studio: {
		backdrop: { center: [ 0.034, 0.035, 0.038 ], edge: [ 0.0105, 0.0108, 0.0115 ], noise: 0.06, tint: 0 },
		sky: {
			zenith: [ 0.11, 0.115, 0.125 ], nadir: [ 0.022, 0.022, 0.024 ],
			fill: [ 0.80, 0.86, 1.0 ], fillIntensity: 0.22, fillSize: [ 0.35, 0.45 ],
			rim: [ 1.0, 1.0, 1.0 ], rimIntensity: 0.7,
			nebulaA: [ 0, 0, 0 ], nebulaB: [ 0, 0, 0 ], nebula: 0,
		},
		key: { color: [ 1.0, 0.985, 0.96 ], light: 4.4, box: 7, size: [ 0.955, 0.03 ] },
		envIntensity: 0.6,
		shadow: 1,
		fog: 0.3,
		nodes: { roughness: 0.34, emissive: 0.7, rim: 0.3 },
	},
	// pure black behind the graph, maximum contrast; lighting as aurora, a little less fill
	void: {
		backdrop: { center: [ 0, 0, 0 ], edge: [ 0, 0, 0 ], noise: 0, tint: 0 },
		sky: {
			zenith: [ 0.022, 0.036, 0.070 ], nadir: [ 0.003, 0.004, 0.008 ],
			fill: [ 0.25, 0.45, 1.0 ], fillIntensity: 0.4, fillSize: [ 0.35, 0.45 ],
			rim: [ 1.0, 0.3, 0.75 ], rimIntensity: 0.8,
			nebulaA: [ 0.45, 0.12, 0.60 ], nebulaB: [ 0.08, 0.50, 0.55 ], nebula: 0.06,
		},
		key: { color: [ 1.0, 0.94, 0.84 ], light: 3.6, box: 5, size: [ 0.972, 0.018 ] },
		envIntensity: 1.2,
		shadow: 0.8,
		fog: 0.4,
		nodes: { roughness: 0.26, emissive: 2.2, rim: 0.6 },
	},
};

export const DEFAULT_KEY_DIRECTION = /*@__PURE__*/ new THREE.Vector3( - 0.45, 0.65, 0.62 ).normalize();

// ── pure helpers (CPU mirror of the sky, shadow fitting, fog) ────────────────────────────────

const smooth = ( a, b, x ) => {

	const t = Math.min( 1, Math.max( 0, ( x - a ) / ( b - a ) ) );
	return t * t * ( 3 - 2 * t );

};

/** The low-frequency pattern shared by the backdrop and the sky: [ n in −1..1, hue mix 0..1 ]. */
export function nebulaPattern( x, y, z ) {

	const w = Math.sin( y * 1.7 + z * 0.9 ) * 1.3;
	const n = Math.sin( x * 2.1 + w ) * Math.sin( y * 2.6 + z * 1.9 + 0.7 ) * 0.6 + Math.sin( z * 3.3 + x * 1.1 + w * 0.5 ) * 0.4;
	const hue = Math.sin( x * 1.3 - y * 0.8 + z * 1.1 + 2.0 ) * 0.5 + 0.5;
	return [ n, hue ];

}

/**
 * Radiance of the procedural sky in direction d (unit, world space): the CPU mirror of the TSL sky.
 * @param {number[]} d direction [x,y,z]
 * @param {object} sky preset.sky
 * @param {object} key { dir: [x,y,z] toward the light, radiance: [r,g,b] of the soft box, size: [cos, soft] }
 * @return {number[]} linear RGB
 */
export function skyRadiance( d, sky, key ) {

	const [ x, y, z ] = d;
	const out = [ 0, 0, 0 ];
	const g = smooth( - 1, 1, y );
	const [ n, hue ] = nebulaPattern( x, y, z );
	const neb = Math.pow( Math.max( n * 0.5 + 0.5, 0 ), 2 ) * sky.nebula;
	const kd = key.dir;
	const cosKey = x * kd[ 0 ] + y * kd[ 1 ] + z * kd[ 2 ];
	const box = smooth( key.size[ 0 ] - key.size[ 1 ], key.size[ 0 ] + key.size[ 1 ], cosKey );
	const fill = smooth( sky.fillSize[ 0 ] - sky.fillSize[ 1 ], sky.fillSize[ 0 ] + sky.fillSize[ 1 ], - cosKey ) * sky.fillIntensity;
	// rim panel: broad and soft around the horizon on the side facing away from the key light (a thin
	// strip would mirror as a crease on glossy spheres)
	const kl = Math.hypot( kd[ 0 ], kd[ 2 ] ) || 1;
	const away = - ( x * kd[ 0 ] + z * kd[ 2 ] ) / kl;
	const strip = smooth( 0.55, 0.05, Math.abs( y + 0.1 ) ) * smooth( 0.2, 0.8, away ) * sky.rimIntensity;
	for ( let c = 0; c < 3; c ++ ) {

		out[ c ] = sky.nadir[ c ] + ( sky.zenith[ c ] - sky.nadir[ c ] ) * g
			+ ( sky.nebulaA[ c ] + ( sky.nebulaB[ c ] - sky.nebulaA[ c ] ) * hue ) * neb
			+ key.radiance[ c ] * box + sky.fill[ c ] * fill + sky.rim[ c ] * strip;

	}

	return out;

}

/**
 * Irradiance of the sky on a surface with unit normal n: ∫ L(ω)·max(n·ω, 0) dω, Fibonacci sphere.
 * Same units as a DirectionalLight's colour × intensity × N·L.
 */
export function skyIrradiance( n, sky, key, samples = 1536 ) {

	const out = [ 0, 0, 0 ];
	const golden = Math.PI * ( 3 - Math.sqrt( 5 ) );
	for ( let i = 0; i < samples; i ++ ) {

		const y = 1 - 2 * ( i + 0.5 ) / samples;
		const r = Math.sqrt( Math.max( 0, 1 - y * y ) );
		const d = [ Math.cos( golden * i ) * r, y, Math.sin( golden * i ) * r ];
		const c = d[ 0 ] * n[ 0 ] + d[ 1 ] * n[ 1 ] + d[ 2 ] * n[ 2 ];
		if ( c <= 0 ) continue;
		const L = skyRadiance( d, sky, key );
		out[ 0 ] += L[ 0 ] * c; out[ 1 ] += L[ 1 ] * c; out[ 2 ] += L[ 2 ] * c;

	}

	const w = 4 * Math.PI / samples;
	return [ out[ 0 ] * w, out[ 1 ] * w, out[ 2 ] * w ];

}

/**
 * Shadow-box radius for a bounding radius r: the next power of 1.25 above r, kept while r stays
 * inside [ previous / 1.5, previous ] so a breathing layout does not flip between two sizes.
 */
export function quantizeRadius( r, previous = 0 ) {

	r = Math.max( r, 1e-6 );
	if ( previous > 0 && r <= previous && r >= previous / 1.5 ) return previous;
	return Math.pow( 1.25, Math.ceil( Math.log( r ) / Math.log( 1.25 ) - 1e-9 ) );

}

/**
 * Orthographic shadow box of a directional light around a bounding sphere.
 * @param {number[]} dir unit direction toward the light
 * @param {number[]} center sphere centre
 * @param {number} radius box half size (use quantizeRadius)
 * @param {number} mapSize shadow map size in texels
 * @return {{ target: number[], position: number[], radius, texel, near, far }} the centre is snapped
 *   to shadow texels across the light direction; the light sits 2·radius away, depth range 2·radius
 */
export function fitShadowBox( dir, center, radius, mapSize ) {

	const texel = 2 * radius / mapSize;
	// light-space basis (the same one for every call: the snapping grid must not rotate)
	const up = Math.abs( dir[ 1 ] ) > 0.99 ? [ 1, 0, 0 ] : [ 0, 1, 0 ];
	let x = [ up[ 1 ] * dir[ 2 ] - up[ 2 ] * dir[ 1 ], up[ 2 ] * dir[ 0 ] - up[ 0 ] * dir[ 2 ], up[ 0 ] * dir[ 1 ] - up[ 1 ] * dir[ 0 ] ];
	const xl = Math.hypot( x[ 0 ], x[ 1 ], x[ 2 ] );
	x = [ x[ 0 ] / xl, x[ 1 ] / xl, x[ 2 ] / xl ];
	const y = [ dir[ 1 ] * x[ 2 ] - dir[ 2 ] * x[ 1 ], dir[ 2 ] * x[ 0 ] - dir[ 0 ] * x[ 2 ], dir[ 0 ] * x[ 1 ] - dir[ 1 ] * x[ 0 ] ];
	const d3 = ( a, b ) => a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];
	const sx = Math.round( d3( center, x ) / texel ) * texel;
	const sy = Math.round( d3( center, y ) / texel ) * texel;
	const sz = d3( center, dir );
	const target = [ 0, 1, 2 ].map( ( k ) => x[ k ] * sx + y[ k ] * sy + dir[ k ] * sz );
	const position = target.map( ( v, k ) => v + dir[ k ] * 2 * radius );
	return { target, position, radius, texel, near: radius, far: 3 * radius };

}

/**
 * Whether the shadow map can resolve the nodes. `texels`: world radius of a size-1 node divided by
 * the shadow texel. Below 2.5 texels a sphere casts no recognisable shape (PCF grain over whole
 * spheres, lost highlights), so the light should not cast at all; up to 4.5 texels the shadow
 * strength fades in. `wasCasting` adds hysteresis (back on at 2.9) so a breathing layout does not
 * toggle the light, which rebuilds the lit materials.
 * @return {{ cast: boolean, strength: number }} strength 0..1 scales the preset's shadow intensity
 */
export function shadowResolve( texels, wasCasting = true ) {

	if ( ! ( texels >= ( wasCasting ? 2.5 : 2.9 ) ) ) return { cast: false, strength: 0 };
	return { cast: true, strength: smooth( 2.5, 4.5, texels ) };

}

/**
 * Fog uniforms for a camera at `distance` from the centre of a graph of radius `radius`.
 * strength 0..1: the far side of the graph ( distance + radius ) keeps 1 − 0.62·strength of its light.
 * @return {{ start: number, density: number }} for exp( −( ( d − start )·density )² )
 */
export function fogParameters( strength, distance, radius ) {

	const s = Math.min( 1, Math.max( 0, strength ) );
	if ( s <= 0 || ! ( radius > 0 ) ) return { start: 0, density: 0 };
	const start = Math.max( 0, distance - 0.35 * radius );
	const far = Math.sqrt( - Math.log( 1 - 0.62 * s ) ); // ( far − start )·density at the far side
	return { start, density: far / ( 1.35 * radius ) };

}

/**
 * Bounding sphere (box centre + largest distance) of the first `count` positions of a vec4 array.
 * @return {{ center: number[], radius: number }}
 */
export function boundsOfPositions( positions, count, stride = 1 ) {

	let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = - Infinity, y1 = - Infinity, z1 = - Infinity, n = 0;
	for ( let i = 0; i < count; i += stride ) {

		const x = positions[ 4 * i ], y = positions[ 4 * i + 1 ], z = positions[ 4 * i + 2 ];
		if ( ! ( Number.isFinite( x ) && Number.isFinite( y ) && Number.isFinite( z ) ) ) continue;
		if ( x < x0 ) x0 = x; if ( x > x1 ) x1 = x;
		if ( y < y0 ) y0 = y; if ( y > y1 ) y1 = y;
		if ( z < z0 ) z0 = z; if ( z > z1 ) z1 = z;
		n ++;

	}

	if ( n === 0 ) return { center: [ 0, 0, 0 ], radius: 0 };
	const c = [ ( x0 + x1 ) / 2, ( y0 + y1 ) / 2, ( z0 + z1 ) / 2 ];
	let r2 = 0;
	for ( let i = 0; i < count; i += stride ) {

		const dx = positions[ 4 * i ] - c[ 0 ], dy = positions[ 4 * i + 1 ] - c[ 1 ], dz = positions[ 4 * i + 2 ] - c[ 2 ];
		const d2 = dx * dx + dy * dy + dz * dz;
		if ( d2 > r2 ) r2 = d2;

	}

	return { center: c, radius: Math.sqrt( r2 ) };

}

// ── TSL: the shared pattern and the sky ──────────────────────────────────────────────────────

/** TSL twin of nebulaPattern(): { n: −1..1, hue: 0..1 } for a unit direction node. */
const nebulaNodes = ( d ) => {

	const w = sin( d.y.mul( 1.7 ).add( d.z.mul( 0.9 ) ) ).mul( 1.3 );
	const n = sin( d.x.mul( 2.1 ).add( w ) ).mul( sin( d.y.mul( 2.6 ).add( d.z.mul( 1.9 ) ).add( 0.7 ) ) ).mul( 0.6 )
		.add( sin( d.z.mul( 3.3 ).add( d.x.mul( 1.1 ) ).add( w.mul( 0.5 ) ) ).mul( 0.4 ) );
	const hue = sin( d.x.mul( 1.3 ).sub( d.y.mul( 0.8 ) ).add( d.z.mul( 1.1 ) ).add( 2.0 ) ).mul( 0.5 ).add( 0.5 );
	return { n, hue };

};

const _v = /*@__PURE__*/ new THREE.Vector3();

export class Environment {

	constructor( { preset = 'aurora', noise = true, guide = true, keyDirection = DEFAULT_KEY_DIRECTION, shadowSoftness = 0.3, pmremSize = 256 } = {} ) {

		this.order = 12;
		this.isEnvironment = true;
		this.app = null;
		this.buffers = null;
		this.graph = null;
		this.noise = noise;
		this._guide = guide;
		this.guideMesh = null;
		this.shadowSoftness = shadowSoftness;
		this.pmremSize = pmremSize;

		this._presetName = ENV_PRESETS[ preset ] ? preset : 'aurora';
		this._preset = ENV_PRESETS[ this._presetName ];
		this._keyIntensity = this._preset.key.light;
		this._envIntensity = this._preset.envIntensity;
		this._fog = this._preset.fog;
		this._keyDir = new THREE.Vector3().copy( keyDirection ).normalize();

		this.uniforms = {
			// backdrop
			center: uniform( new THREE.Color() ),
			edge: uniform( new THREE.Color() ),
			noiseAmount: uniform( 0.22 ),
			tintAmount: uniform( 0 ),
			guideOpacity: uniform( 0.08 ),
			// sky (radiance, linear)
			zenith: uniform( new THREE.Color() ),
			nadir: uniform( new THREE.Color() ),
			keyBox: uniform( new THREE.Color() ),
			keySize: uniform( new THREE.Vector2( 0.97, 0.02 ) ),
			fill: uniform( new THREE.Color() ),
			fillSize: uniform( new THREE.Vector2( 0.35, 0.45 ) ),
			rim: uniform( new THREE.Color() ),
			nebulaA: uniform( new THREE.Color() ),
			nebulaB: uniform( new THREE.Color() ),
		};

		this.light = new THREE.DirectionalLight( 0xffffff, this._keyIntensity );
		this.light.name = 'Environment.key';
		this.light.shadow.autoUpdate = false;

		this.stats = { shadowRenders: 0, skyRenders: 0, shadowRadius: 0, shadowTexel: 0, shadowTexels: 0, shadowStrength: 1, shadowCasting: false, boundsRadius: 0, lastShadowFrame: - 1 };

		// counts actual shadow map renders (updateMatrices is called once per render of the map)
		const updateMatrices = this.light.shadow.updateMatrices.bind( this.light.shadow );
		this.light.shadow.updateMatrices = ( light ) => {

			this.stats.shadowRenders ++;
			this.stats.lastShadowFrame = this.app ? this.app.frame : - 1;
			updateMatrices( light );

		};

		this._generator = null;
		this._pmrem = null;
		this._skyScene = null;
		this._skyDirty = true;
		this._shadowDirty = true;
		this._moveMark = 0; // positions moved in a frame ≥ this → the shadow map is stale
		this._holdUntil = 0; // keep updating until this app.time (eased selection / size changes)
		this._seen = { selected: - 2, hover: - 2, state: - 1, size: NaN, visible: null, mapSize: 0, shadows: null, lit: null, positions: - 1, colors: - 1 };
		this._resolved = true; // the shadow map resolves the nodes (shadowResolve), see _updateShadow
		this._bounds = { center: [ 0, 0, 0 ], radius: 0, valid: false };
		this._fit = { radius: 0, target: [ NaN, NaN, NaN ] };
		this._fogRadius = 0;
		this._skipVelocity = skipVelocity(); // for the additive guide while the pass has a velocity attachment
		this._onGfx = () => this._applyGfx();
		this._onSelect = () => this._hold( 0.9 ); // also 'path' and 'lens': highlight sizes ease on the GPU

	}

	// ── properties ──────────────────────────────────────────────────────────────────────────

	get preset() {

		return this._presetName;

	}

	get keyIntensity() {

		return this._keyIntensity;

	}

	set keyIntensity( v ) {

		v = Math.max( 0, + v || 0 );
		if ( v === this._keyIntensity ) return;
		this._keyIntensity = v;
		this._applyLight();

	}

	get envIntensity() {

		return this._envIntensity;

	}

	set envIntensity( v ) {

		this._envIntensity = Math.max( 0, + v || 0 );
		if ( this.app ) this.app.scene.environmentIntensity = this._envIntensity;
		this._applyAmbient();

	}

	get fog() {

		return this._fog;

	}

	set fog( v ) {

		this._fog = Math.min( 1, Math.max( 0, + v || 0 ) );

	}

	get keyDirection() {

		return this._keyDir;

	}

	/** World-space direction toward the key light (Vector3 or [x, y, z]). Regenerates sky and shadows. */
	setKeyDirection( v ) {

		if ( Array.isArray( v ) ) _v.fromArray( v ); else _v.copy( v );
		if ( _v.lengthSq() < 1e-12 ) return this;
		this._keyDir.copy( _v ).normalize();
		this._fit.radius = 0; // the snapping grid turns with the light
		this._applyLight();
		return this;

	}

	get guide() {

		return this._guide;

	}

	set guide( v ) {

		this._guide = !! v;

	}

	get guideOpacity() {

		return this.uniforms.guideOpacity.value;

	}

	set guideOpacity( v ) {

		this.uniforms.guideOpacity.value = v;

	}

	/** Backdrop colours, linear RGB arrays (live). Overrides the preset until the next setPreset(). */
	setColors( center, edge ) {

		this.uniforms.center.value.setRGB( ...center );
		this.uniforms.edge.value.setRGB( ...edge );
		this._applyFogColor();

	}

	/**
	 * Switches the look preset. `nodes`: also apply the preset's node look hints to the
	 * NodeRenderer module (roughness, emissive, rim), so one call changes the whole look.
	 */
	setPreset( name, { nodes = true } = {} ) {

		const preset = ENV_PRESETS[ name ];
		if ( ! preset ) return this;
		this._presetName = name;
		this._preset = preset;
		this._keyIntensity = preset.key.light;
		this._envIntensity = preset.envIntensity;
		this._fog = preset.fog;
		this._applyPreset();
		if ( nodes && preset.nodes && this.app ) this._nodeRenderer()?.setLook?.( preset.nodes );
		return this;

	}

	/**
	 * Forces a shadow map render in the next frame (e.g. after changing a shadow caster). `seconds`:
	 * keep re-rendering for that long, for casters that ease to their new state (highlight sizes
	 * written to nodeStateTarget ease for about 0.2 s; 0.9 covers the tail).
	 */
	updateShadow( seconds = 0 ) {

		this._shadowDirty = true;
		if ( seconds > 0 ) this._hold( seconds );

	}

	/** Forces a regeneration of the environment map in the next frame. */
	updateSky() {

		this._skyDirty = true;

	}

	// ── App module ──────────────────────────────────────────────────────────────────────────

	init( app ) {

		this.app = app;
		const scene = app.scene;
		scene.backgroundNode = this._backgroundNode();
		scene.add( this.light, this.light.target );

		// the sky scene: one inside-out sphere, colour = radiance( direction )
		const material = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide, depthWrite: false, fog: false } );
		material.name = 'Environment.sky';
		material.colorNode = this._skyNode();
		this._skyScene = new THREE.Scene();
		this._skyScene.add( new THREE.Mesh( new THREE.SphereGeometry( 10, 48, 24 ), material ) );

		app.renderer.shadowMap.type = THREE.PCFShadowMap;
		this._applyPreset();
		this._applyGfx(); // renders the sky unless gfxState.lit is false
		app.events.addEventListener( 'gfx', this._onGfx );
		for ( const type of [ 'select', 'path', 'lens' ] ) app.events.addEventListener( type, this._onSelect );

	}

	setGraph( buffers, graph, app ) {

		this._disposeGuide();
		this.buffers = buffers;
		this.graph = graph;
		this._bounds.valid = false;
		this._fit.radius = 0;
		this._fogRadius = 0;
		this._seen.positions = - 1;
		this._seen.selected = - 2;
		this._seen.state = - 1;
		this._resolved = true;
		this._shadowDirty = true;
		this._buildGuide( buffers, app || this.app );

	}

	update( dt, app ) {

		const buffers = this.buffers;
		if ( this.guideMesh && buffers ) this.guideMesh.visible = this._guide && buffers.uniforms.layoutBlend.value > 0.002;
		this._applyGfx();
		if ( this._skyDirty ) this._renderSky();
		if ( ! buffers || ! this.graph ) return;

		this._updateBounds( app );
		const b = this._bounds;

		// fog: a depth cue relative to the graph, eased so a growing layout does not pump it
		const k = this._fogRadius > 0 ? 1 - Math.exp( - dt * 3 ) : 1;
		this._fogRadius += ( b.radius - this._fogRadius ) * k;
		const distance = app.camera.position.distanceTo( _v.fromArray( b.center ) );
		const fog = fogParameters( this._fog, distance, this._fogRadius );
		gfxUniforms.fogStart.value = fog.start;
		gfxUniforms.fogDensity.value = fog.density;

		this._updateShadow( app );

	}

	// ── sky and backdrop ────────────────────────────────────────────────────────────────────

	_backgroundNode() {

		const U = this.uniforms;
		return Fn( () => {

			const aspect = screenSize.x.div( max( screenSize.y, 1 ) );
			const p = screenUV.sub( vec2( 0.5, 0.44 ) ).mul( vec2( aspect, 1 ) );
			const radial = exp( p.dot( p ).mul( - 2.4 ) );
			const c = mix( vec3( U.edge ), vec3( U.center ), radial ).toVar();
			if ( this.noise ) {

				// a few domain-warped sines on the view direction: far cheaper per pixel than gradient
				// noise, and only ±20 % of a ~1 % signal; the tint adds a hint of the nebula colours
				const { n, hue } = nebulaNodes( normalWorldGeometry );
				c.mulAssign( float( 1 ).add( n.mul( U.noiseAmount ) ) );
				const cloud = n.mul( 0.5 ).add( 0.5 );
				c.addAssign( mix( vec3( U.nebulaA ), vec3( U.nebulaB ), hue ).mul( cloud.mul( cloud ) ).mul( U.tintAmount ).mul( radial.mul( 0.6 ).add( 0.4 ) ) );

			}

			return vec4( c, 1 );

		} )();

	}

	_skyNode() {

		const U = this.uniforms;
		return Fn( () => {

			const d = normalize( positionLocal ).toVar();
			const kd = vec3( gfxUniforms.keyDir );
			const base = mix( vec3( U.nadir ), vec3( U.zenith ), smoothstep( - 1, 1, d.y ) );
			const { n, hue } = nebulaNodes( d );
			const cloud = max( n.mul( 0.5 ).add( 0.5 ), 0 );
			// nebulaA / nebulaB already carry the nebula amount
			const nebula = mix( vec3( U.nebulaA ), vec3( U.nebulaB ), hue ).mul( cloud.mul( cloud ) );
			const cosKey = dot( d, kd );
			const box = smoothstep( U.keySize.x.sub( U.keySize.y ), U.keySize.x.add( U.keySize.y ), cosKey );
			const fill = smoothstep( U.fillSize.x.sub( U.fillSize.y ), U.fillSize.x.add( U.fillSize.y ), cosKey.negate() );
			const away = dot( d.xz, kd.xz ).negate().div( max( length( kd.xz ), 1e-4 ) );
			const strip = smoothstep( 0.55, 0.05, abs( d.y.add( 0.1 ) ) ).mul( smoothstep( 0.2, 0.8, away ) );
			return base.add( nebula ).add( vec3( U.keyBox ).mul( box ) ).add( vec3( U.fill ).mul( fill ) ).add( vec3( U.rim ).mul( strip ) );

		} )();

	}

	/** Sky parameters of the active preset with the current key intensity (input of skyRadiance). */
	_skyParameters() {

		const p = this._preset;
		const scale = p.key.light > 0 ? this._keyIntensity / p.key.light : 1;
		return {
			sky: p.sky,
			key: { dir: this._keyDir.toArray(), radiance: p.key.color.map( ( c ) => c * p.key.box * scale ), size: p.key.size },
		};

	}

	_applyPreset() {

		const p = this._preset, U = this.uniforms;
		U.center.value.setRGB( ...p.backdrop.center );
		U.edge.value.setRGB( ...p.backdrop.edge );
		U.noiseAmount.value = p.backdrop.noise;
		U.tintAmount.value = p.backdrop.tint;
		U.zenith.value.setRGB( ...p.sky.zenith );
		U.nadir.value.setRGB( ...p.sky.nadir );
		U.fill.value.setRGB( ...p.sky.fill ).multiplyScalar( p.sky.fillIntensity );
		U.fillSize.value.fromArray( p.sky.fillSize );
		U.rim.value.setRGB( ...p.sky.rim ).multiplyScalar( p.sky.rimIntensity );
		U.nebulaA.value.setRGB( ...p.sky.nebulaA ).multiplyScalar( p.sky.nebula );
		U.nebulaB.value.setRGB( ...p.sky.nebulaB ).multiplyScalar( p.sky.nebula );
		U.keySize.value.fromArray( p.key.size );
		this.light.shadow.intensity = p.shadow * this.stats.shadowStrength;
		if ( this.app ) this.app.scene.environmentIntensity = this._envIntensity;
		this._applyFogColor();
		this._applyLight();

	}

	/** Key light colour / intensity / direction → light, sky soft box, shared uniforms. */
	_applyLight() {

		const p = this._preset;
		this.light.color.setRGB( ...p.key.color );
		this.light.intensity = this._keyIntensity;
		const scale = p.key.light > 0 ? this._keyIntensity / p.key.light : 1;
		this.uniforms.keyBox.value.setRGB( ...p.key.color ).multiplyScalar( p.key.box * scale );
		gfxUniforms.keyDir.value.copy( this._keyDir );
		gfxUniforms.keyColor.value.copy( this.light.color ).multiplyScalar( this._keyIntensity );
		this._skyDirty = true;
		this._shadowDirty = true;
		this._applyAmbient();

	}

	/** Hemisphere irradiance of the sky (CPU mirror) → gfxUniforms.ambientSky / ambientGround. */
	_applyAmbient() {

		const { sky, key } = this._skyParameters();
		const up = skyIrradiance( [ 0, 1, 0 ], sky, key );
		const down = skyIrradiance( [ 0, - 1, 0 ], sky, key );
		gfxUniforms.ambientSky.value.setRGB( ...up ).multiplyScalar( this._envIntensity );
		gfxUniforms.ambientGround.value.setRGB( ...down ).multiplyScalar( this._envIntensity );

	}

	_applyFogColor() {

		// what a distant thing fades to: the backdrop between its centre and its corners
		const U = this.uniforms;
		gfxUniforms.fogColor.value.copy( U.edge.value ).lerp( U.center.value, 0.55 );

	}

	_renderSky() {

		// nothing samples the environment while the shading is unlit (no texture taps): do not spend
		// 6 faces + the GGX filter chain and ~4 MB on it; it stays dirty and is rendered when lit turns on
		if ( ! this._skyScene || gfxState.lit === false ) return;
		this._skyDirty = false;
		if ( ! this._generator ) this._generator = new THREE.PMREMGenerator( this.app.renderer );
		// passing the previous target re-renders into it: same GPU texture, no material rebuild
		this._pmrem = this._generator.fromScene( this._skyScene, 0, 0.1, 100, { size: this.pmremSize, renderTarget: this._pmrem } );
		this.app.scene.environment = this._pmrem.texture;
		this.app.scene.environmentIntensity = this._envIntensity;
		this.stats.skyRenders ++;

	}

	/** Frees the environment cube and the generator's work targets (unlit shading). */
	_disposeSky() {

		if ( this.app && this.app.scene && this._pmrem && this.app.scene.environment === this._pmrem.texture ) this.app.scene.environment = null;
		this._pmrem?.dispose();
		this._generator?.dispose();
		this._pmrem = this._generator = null;
		this._skyDirty = true;

	}

	// ── shadows ─────────────────────────────────────────────────────────────────────────────

	_nodeRenderer() {

		const app = this.app;
		return ( app && app.modules && app.modules.find( ( m ) => m && m.isNodeRenderer ) ) || null;

	}

	_hold( seconds ) {

		if ( this.app ) this._holdUntil = Math.max( this._holdUntil, this.app.time + seconds );

	}

	/** renderer / light shadow switches from gfxState. */
	_applyGfx() {

		const app = this.app;
		if ( ! app ) return;
		const guide = this.guideMesh && this.guideMesh.material;
		const skip = gfxState.velocity ? this._skipVelocity : null;
		if ( guide && guide.mrtNode !== skip ) {

			guide.mrtNode = skip;
			guide.needsUpdate = true;

		}

		const lit = gfxState.lit !== false;
		if ( this._seen.lit !== lit ) {

			this._seen.lit = lit;
			if ( lit ) this._renderSky(); else this._disposeSky();

		}

		const on = !! gfxState.shadows && lit;
		const size = Math.max( 256, gfxState.shadowMapSize | 0 );
		if ( this._seen.shadows === on && this._seen.mapSize === size ) return;
		this._seen.shadows = on;
		this._seen.mapSize = size;
		app.renderer.shadowMap.enabled = on;
		this.light.shadow.mapSize.set( size, size );
		this._fit.radius = 0; // texel size changed: refit
		this._setCasting( on && this._resolved );
		this._shadowDirty = true;

	}

	/**
	 * light.castShadow = shadows wanted (gfxState) and resolvable (shadowResolve). It is part of
	 * three's lights cache key: the lit materials rebuild by themselves, without the shadow taps.
	 */
	_setCasting( cast ) {

		const light = this.light;
		this.stats.shadowCasting = cast;
		if ( light.castShadow === cast ) return;
		light.castShadow = cast;
		this._shadowDirty = true;
		// three frees the map only when a lit material is rebuilt without it, which never happens when
		// the shading went unlit at the same time. Its targets are re-created on the next shadow render
		if ( ! cast ) light.shadow.dispose();

	}

	_updateBounds( app ) {

		const graph = this.graph, b = this._bounds;
		if ( b.valid && this._seen.positions === app.positionsVersion ) return;
		this._seen.positions = app.positionsVersion;
		const N = graph.N;
		const pos = app.positions && app.positions.length >= 4 * N ? app.positions : graph.initPos;
		const stride = N > 60000 ? Math.ceil( N / 60000 ) : 1;
		const s = boundsOfPositions( pos, N, stride );
		b.center = s.center;
		b.radius = Math.max( s.radius * ( stride > 1 ? 1.05 : 1 ), graph.layoutRadius * 0.02, 1e-3 );
		b.valid = true;
		this.stats.boundsRadius = b.radius;

	}

	_updateShadow( app ) {

		const buffers = this.buffers, seen = this._seen, light = this.light;
		if ( ! seen.shadows ) return; // not wanted (gfxState.shadows, gfxState.lit)

		// what changes the casters besides their positions: node size, visibility, radii, selection
		const nodes = this._nodeRenderer();
		const size = nodes ? nodes.size : 1;
		const visible = nodes ? nodes.visible : true;
		if ( size !== seen.size || visible !== seen.visible ) {

			seen.size = size; seen.visible = visible;
			this._fit.radius = 0;
			this._shadowDirty = true;

		}

		// highlight sizes (1 + 0.5·nodeState.x) are eased on the GPU: follow the transition. Any write to
		// the state targets counts (path / k-hop highlights leave selectedIndex alone), except a hover
		// change, which the shadow casters ignore
		const selected = buffers.uniforms.selectedIndex.value, hover = buffers.uniforms.hoverIndex.value;
		const state = buffers.nodeStateTarget.value.version;
		if ( selected !== seen.selected || ( state !== seen.state && hover === seen.hover ) ) this._hold( 0.9 );
		seen.selected = selected; seen.hover = hover; seen.state = state;

		const colors = buffers.nodeColorTarget.value.version;
		if ( colors !== seen.colors ) {

			seen.colors = colors;
			this._hold( 0.6 ); // radius factors are eased as well

		}

		if ( buffers.moveFrame === Infinity || buffers.moveFrame >= this._moveMark ) this._shadowDirty = true;
		this._moveMark = app.frame + 1;
		if ( app.time < this._holdUntil ) this._shadowDirty = true;

		// fit: bounding sphere + the largest drawn node (selected nodes are 1.5× larger)
		const baseSize = nodes ? nodes.baseSize : 0.065 * this.graph.layoutRadius / Math.cbrt( Math.max( 1, this.graph.N ) );
		const nodeRadius = baseSize * size;
		const b = this._bounds, fit = this._fit;
		const mapSize = light.shadow.mapSize.x;
		const radius = quantizeRadius( ( b.radius + nodeRadius * 2.6 ) * 1.02, fit.radius );
		const box = fitShadowBox( this._keyDir.toArray(), b.center, radius, mapSize );
		if ( radius !== fit.radius || box.target[ 0 ] !== fit.target[ 0 ] || box.target[ 1 ] !== fit.target[ 1 ] || box.target[ 2 ] !== fit.target[ 2 ] ) {

			fit.radius = radius;
			fit.target = box.target;
			light.target.position.fromArray( box.target );
			light.position.fromArray( box.position );
			light.target.updateMatrixWorld();
			light.updateMatrixWorld();
			const cam = light.shadow.camera;
			cam.left = - radius; cam.right = radius; cam.top = radius; cam.bottom = - radius;
			cam.near = box.near; cam.far = box.far; // depth range = the sphere's diameter
			cam.updateProjectionMatrix();
			// depth bias in texels: the depth written by the impostors is exact, so little is needed
			// (the normal bias follows the filter radius, below)
			light.shadow.bias = - 0.75 / mapSize;
			this.stats.shadowRadius = radius;
			this.stats.shadowTexel = box.texel;
			this._shadowDirty = true;

		}

		// can the map resolve the nodes at all? (changes only with the fit, the map size or the node size)
		const texels = nodeRadius / box.texel;
		const resolve = shadowResolve( texels, this._resolved );
		this._resolved = resolve.cast;
		this.stats.shadowTexels = texels;
		this.stats.shadowStrength = resolve.strength;
		light.shadow.intensity = this._preset.shadow * resolve.strength;
		this._setCasting( resolve.cast );
		if ( ! resolve.cast ) return;

		// penumbra: a fraction of the node radius, in texels; 5 PCF taps get grainy when it is wide,
		// which only a temporal resolve hides
		const maxTexels = gfxState.velocity ? ( gfxState.tier === 'ultra' ? 5 : 3.5 ) : 2.5;
		light.shadow.radius = Math.min( maxTexels, Math.max( 1.25, this.shadowSoftness * nodeRadius / box.texel ) );
		// normal bias = reach of the filter (radius + the bilinear comparison footprint): a tap ρ texels
		// "uphill" on a surface tilted by θ sees a depth ρ·tan θ nearer to the light, and moving the lookup
		// b along the normal gains b / cos θ, so b ≥ ρ keeps a sphere from shadowing itself anywhere on its
		// lit side. Less than that shows as a dithered hatch over the whole lit hemisphere
		light.shadow.normalBias = ( light.shadow.radius + 1 ) * box.texel;

		if ( this._shadowDirty ) {

			light.shadow.needsUpdate = true;
			this._shadowDirty = false;

		}

	}

	// ── guide sphere ────────────────────────────────────────────────────────────────────────

	_buildGuide( buffers, app ) {

		const U = this.uniforms;
		const S = buffers.uniforms;

		const geometry = new THREE.SphereGeometry( 1, 144, 72 );
		const material = new THREE.NodeMaterial();
		material.name = 'Environment.guide';
		material.transparent = true;
		material.depthWrite = false;
		material.depthTest = true;
		material.blending = THREE.AdditiveBlending;
		material.side = THREE.DoubleSide;
		material.lights = false;
		material.fog = false;
		material.mrtNode = gfxState.velocity ? this._skipVelocity : null;
		material.positionNode = positionGeometry.mul( S.sphereRadius ).add( S.graphCenter );

		const line = ( f ) => {

			// distance to the nearest integer of f, in pixels → 1 px wide AA line
			const d = abs( fract( f.sub( 0.5 ) ).sub( 0.5 ) ).div( max( fwidth( f ), 1e-5 ) );
			return clamp( float( 1.1 ).sub( d ), 0, 1 );

		};

		material.colorNode = Fn( () => {

			const t = uv();
			const lat = sin( t.y.mul( Math.PI ) ); // 0 at the poles
			const lon = line( t.x.mul( 24 ) ).mul( smoothstep( 0.05, 0.35, lat ) );
			const par = line( t.y.mul( 12 ) ).mul( smoothstep( 0.0, 0.12, lat ) );
			const equator = line( t.y.mul( 2 ) ).mul( 0.6 );
			const l = max( max( lon, par ), equator );
			const facing = select( frontFacing, float( 1 ), float( 0.4 ) );
			const a = l.mul( facing ).mul( U.guideOpacity ).mul( S.layoutBlend );
			return vec4( vec3( 0.42, 0.58, 0.95 ).mul( a ), 1 );

		} )();

		const mesh = new THREE.Mesh( geometry, material );
		mesh.name = 'Environment.guide';
		mesh.frustumCulled = false;
		mesh.renderOrder = - 10;
		mesh.visible = false;
		mesh.castShadow = false;
		mesh.receiveShadow = false;
		app.scene.add( mesh );
		this.guideMesh = mesh;

	}

	_disposeGuide() {

		if ( ! this.guideMesh ) return;
		this.guideMesh.removeFromParent();
		this.guideMesh.geometry.dispose();
		this.guideMesh.material.dispose();
		this.guideMesh = null;

	}

	dispose() {

		this._disposeGuide();
		const app = this.app;
		if ( app ) {

			app.events.removeEventListener( 'gfx', this._onGfx );
			for ( const type of [ 'select', 'path', 'lens' ] ) app.events.removeEventListener( type, this._onSelect );
			if ( app.scene ) {

				app.scene.backgroundNode = null;
				app.scene.environment = null;
				app.scene.remove( this.light, this.light.target );

			}

		}

		this.light.dispose();
		if ( this._skyScene ) {

			const sky = this._skyScene.children[ 0 ];
			sky.geometry.dispose();
			sky.material.dispose();

		}

		this._pmrem?.dispose();
		this._generator?.dispose();
		this._pmrem = this._generator = this._skyScene = null;

	}

}
