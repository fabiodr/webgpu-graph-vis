// Quality tiers as data, the resolution of a quality setting into a pipeline configuration, and
// the frame-time governor. Pure logic: no three.js, no DOM, no clock of its own (the caller
// passes times), so everything here runs under node:test (test/quality.test.js).
//
// ─── Tiers ──────────────────────────────────────────────────────────────────────────────────
//  TIERS[ 'low' | 'medium' | 'high' | 'ultra' ] is the table of the overhaul brief (3.5):
//
//               low (phones)      medium (integrated)      high (desktop)     ultra (discrete)
//  scene pass   1 sample          MSAA 4× (1 > 20k edges)  1 + velocity MRT   1 + velocity MRT
//  AA           FXAA (DPR < 2.5)  MSAA + a2c | SMAA        TRAA               TRAA
//  lighting     unlit             lit                      lit                lit + clearcoat
//  shadows      off               PCF 1024                 PCF 2048           PCF 2048, wide radius
//  AO           off               GTAO ½, 16, bilateral    GTAO ½, 16, temp.  GTAO 1, 32, temporal
//  DoF          off               off                      off unless asked   on (mild)
//  bloom        ¼                 ½                        ½                  ½ + streak
//  lens         vignette, dither  + grain                  + grain            + CA 0.15, RCAS
//  edges        blended, unlit    + strand light           + selection tubes  + strand defocus
//  pixel cap    1.5 MP            2.5 MP                   4 MP               6 MP
//
// ─── resolveQuality( setting, env, overrides ) → config ─────────────────────────────────────
//  setting    'auto' | 'low' | 'medium' | 'high' | 'ultra' (anything else counts as 'auto')
//  env        { coarsePointer, smallScreen, softwareAdapter, gpuClass, compat, edges, dpr, deviceDpr,
//               cssWidth, cssHeight }
//             gpuClass = gpuClass( GPUAdapterInfo ): 'software' | 'integrated' | 'mobile' | 'discrete' |
//             'apple' | 'unknown' (default; the browser masks the adapter): what 'auto' starts with,
//             see autoTier(). compat = the renderer runs in WebGPU compatibility mode (no depth
//             texture gather, no multisampled float targets, no per-attachment blending): every
//             setting resolves to 'low' there (config.compat, config.requested = the tier asked for).
//             dpr = renderer pixel ratio (drawing-buffer pixels per CSS pixel, the App caps it),
//             deviceDpr = window.devicePixelRatio (default: dpr), decides the FXAA rule of low
//  overrides  forced switches (settings / URL); undefined or null = tier default:
//             { tier, ao, dof, shadows, traa, sharpen, streak, grain, vignette,
//               chromaticAberration, scale, dynamicResolution, debug, post }
//             `post: false` strips every effect (AO, DoF, bloom, lens): scene, AA, tone mapping,
//             `tier` replaces the tier that 'auto' would pick (used by the governor),
//             `scale` fixes the scene-pass resolution scale (no pixel cap, no governor),
//             `debug` is 'ao' | 'depth' | 'velocity' | 'coc' | null.
//  config     a plain object with every structural decision of the render pipeline:
//             setting, tier, explicit, compat, requested, samples, aa ('none'|'fxaa'|'smaa'|'msaa'|'traa'), velocity,
//             lit, clearcoat, shadows, shadowMapSize, shadowRadius, ao, aoScale, aoSamples,
//             aoDenoise ('none'|'bilateral'), aoTemporal, dof, dofTaps, dofAmount, dofMaxBlur,
//             bloom, bloomScale, streak, vignette, grain, chromaticAberration, sharpen, edges { lit, tubes,
//             defocus }, maxPixels, capScale, renderScale, fixedScale, dynamicResolution, debug.
//             structureKey( config ) is equal for two configs that need the same node graph.
//  parseGfxParams( urlSearchParams ) → { quality?, governor?, overrides }: the URL switches
//             (?quality= ?governor= ?ao= ?dof= ?shadows= ?traa= ?scale= ?debug= …).
//
// ─── Governor ───────────────────────────────────────────────────────────────────────────────
//  Fed with frame times, it answers with at most one action per decision window:
//    { type: 'scale', scale }   set the scene-pass resolution scale (dynamic resolution)
//    { type: 'tier', tier }     switch the tier ('auto' only)
//  What "slow" means is measured, not assumed: the display rate is the fastest pace the loop has
//  shown lately (a low quantile of the frame intervals of each window, the best of the last
//  windows, limited to minTargetFps … targetFps), and a window is slow below lowRatio of it. A
//  loop that the display, a battery saver or the browser caps at 30 fps is therefore fine.
//  Rules:
//  · Sustained slow windows lower the scale in steps; after a long good stretch it creeps back up
//    (a climb that hurts is taken back and doubles the waiting time, so it cannot oscillate).
//  · The second step of a descent reaches half the pixels of its start (probeScale). A frame rate
//    that has not gained gainRatio by then is not limited by the resolution: the scale goes back
//    and resolution is left alone for retryMs, doubled after every further failure. Nothing is
//    switched off for good: a load that changes later is found by the next try.
//  · A steady pace below targetFps can also be a GPU-bound loop locked to a divisor of a faster
//    display. One step to probeScale asks the question: at once when GPU timings show a GPU that
//    is busy for more than half of the frame, never when they show an idle one, and after
//    discoverWindows without timings. The answer "the display" takes the step back and doubles
//    the pause before the next question (a loop that turns slow meanwhile may still descend).
//    Below the scale where such a step still fits above the floor the nominal targetFps counts.
//    A climb after a descent whose cause went away (the pace is pinned at the display rate and no
//    climb has failed) takes fastUpFactor steps every fastUpWindows, never past a scale that failed.
//  · 'auto' only: at the lowest scale and still slow, the tier steps down. Where the resolution
//    is not the reason (see above, or dynamic resolution is off) the tier steps down ON TRIAL:
//    no gain within trialWindows and it goes back up and stays (tierUseless).
//  · A TRANSIENT load is not charged to the tier: while the caller flags frames as transient (the
//    layout is running, a graph was just loaded) only the scale moves, for at most transientMs in
//    a row (a layout that never rests on a device that is too slow still gets its cheaper tier).
//  · Nothing is for good on the way up either: a tier below the ceiling (configure( { ceiling } ),
//    what 'auto' may reach on this device: never 'ultra', which changes the look) is tried again
//    after upgradeWindows good windows at full scale; with GPU timings only while the render
//    passes leave headroom, without them on the frame rate alone. A step up that turns slow
//    within upTrialWindows is taken back, and so is the next try twice as late (upRetryMs,
//    doubled after every step down, up to maxRetryMs). So a device that was demoted by a thermal
//    dip, or one whose adapter the browser masks and that therefore starts at 'medium', finds its
//    tier, and a device that cannot hold it pays one short trial per pause.
//  · An explicit tier is never changed (only its scale, and only with dynamic resolution on).
//  · Frames longer than dropMs (hidden tab) are ignored; after every change the governor waits
//    settleMs / settleFrames (pipeline compiles) before it measures again.
//
//    const gov = new Governor( options );
//    gov.configure( { tier, explicit, ceiling, scale, maxScale, dynamicResolution, enabled } );  // after builds / resizes
//    const action = gov.sample( frameMs, nowMs, gpuMs?, { transient, renderMs }? );       // every frame
//                              // gpuMs: all GPU passes of the frame; renderMs: its render passes only
//    gov.settle();             // the loop was paused (hidden tab, idle): wait before measuring again
//    gov.reset();              // a new quality setting: forget everything but the display rate
//    gov.reset( false );       // a new graph: forget the measurements, keep the tier history
//    gov.enabled = false;                                                                 // ?governor=0
//    gov.displayFps / .targetFps / .fps / .state / .log                                   // read only

export const TIER_ORDER = [ 'low', 'medium', 'high', 'ultra' ];

/** Above this edge count the medium tier renders single-sample (blend traffic of the overdraw). */
export const MSAA_MAX_EDGES = 20000;

export const TIERS = {
	low: {
		samples: 1, aa: 'fxaa', fxaaMaxDpr: 2.5, velocity: false,
		lit: false, clearcoat: false,
		shadows: false, shadowMapSize: 0, shadowRadius: 0,
		ao: false, aoScale: 0.5, aoSamples: 16, aoDenoise: 'none', aoTemporal: false,
		dof: false, dofTaps: 32, dofAmount: 0.6, dofMaxBlur: 8,
		bloomScale: 0.25, streak: false,
		vignette: true, grain: 0, chromaticAberration: 0, sharpen: false,
		edges: { lit: false, tubes: false, defocus: false },
		maxPixels: 1.5e6,
	},
	medium: {
		samples: 4, aa: 'msaa', fxaaMaxDpr: 2.5, velocity: false,
		lit: true, clearcoat: false,
		shadows: true, shadowMapSize: 1024, shadowRadius: 1,
		ao: true, aoScale: 0.5, aoSamples: 16, aoDenoise: 'bilateral', aoTemporal: false,
		dof: false, dofTaps: 32, dofAmount: 0.6, dofMaxBlur: 8,
		bloomScale: 0.5, streak: false,
		vignette: true, grain: 0.035, chromaticAberration: 0, sharpen: false,
		edges: { lit: true, tubes: false, defocus: false },
		maxPixels: 2.5e6,
	},
	high: {
		samples: 1, aa: 'traa', fxaaMaxDpr: 2.5, velocity: true,
		lit: true, clearcoat: false,
		shadows: true, shadowMapSize: 2048, shadowRadius: 1,
		ao: true, aoScale: 0.5, aoSamples: 16, aoDenoise: 'none', aoTemporal: true,
		dof: false, dofTaps: 48, dofAmount: 0.6, dofMaxBlur: 8,
		bloomScale: 0.5, streak: false,
		vignette: true, grain: 0.035, chromaticAberration: 0, sharpen: false,
		edges: { lit: true, tubes: true, defocus: false },
		maxPixels: 4e6,
	},
	ultra: {
		samples: 1, aa: 'traa', fxaaMaxDpr: 2.5, velocity: true,
		lit: true, clearcoat: true,
		shadows: true, shadowMapSize: 2048, shadowRadius: 2.5,
		ao: true, aoScale: 1, aoSamples: 32, aoDenoise: 'none', aoTemporal: true,
		dof: true, dofTaps: 48, dofAmount: 0.6, dofMaxBlur: 8,
		bloomScale: 0.5, streak: true,
		vignette: true, grain: 0.035, chromaticAberration: 0.15, sharpen: true,
		edges: { lit: true, tubes: true, defocus: true },
		maxPixels: 6e6,
	},
};

export const DEBUG_VIEWS = [ 'ao', 'depth', 'velocity', 'coc' ];

const isSet = ( v ) => v !== undefined && v !== null;

/** True for CPU rasterizers (SwiftShader, llvmpipe, WARP …): GPUAdapterInfo-like object. */
export function isSoftwareAdapter( info ) {

	if ( ! info ) return false;
	if ( info.isFallbackAdapter === true ) return true;
	const text = [ info.vendor, info.architecture, info.device, info.description ].filter( Boolean ).join( ' ' ).toLowerCase();
	return /swiftshader|llvmpipe|lavapipe|softpipe|software|basic render|warp/.test( text );

}

/**
 * What kind of GPU an adapter is, from its GPUAdapterInfo (vendor, architecture, device,
 * description; browsers fill in what they are willing to tell):
 *   'software'    CPU rasterizers
 *   'apple'       Apple silicon
 *   'discrete'    NVIDIA, AMD RDNA, Intel Arc
 *   'integrated'  Intel gen-* / xe-lp*, AMD APUs and the GCN generation (cannot be told apart)
 *   'mobile'      ARM Mali, Qualcomm Adreno, Imagination, Samsung, Broadcom
 *   'unknown'     no information (masked), or a vendor that is not known here
 */
export function gpuClass( info ) {

	if ( ! info ) return 'unknown';
	if ( isSoftwareAdapter( info ) ) return 'software';
	const vendor = String( info.vendor || '' ).toLowerCase();
	const text = [ info.vendor, info.architecture, info.device, info.description ].filter( Boolean ).join( ' ' ).toLowerCase();
	if ( text === '' ) return 'unknown';
	if ( /apple/.test( vendor ) ) return 'apple';
	if ( /nvidia/.test( vendor ) || /geforce|quadro|\brtx\b|\bgtx\b/.test( text ) ) return /tegra/.test( text ) ? 'mobile' : 'discrete';
	if ( /intel/.test( vendor ) ) return /\barc\b|dg2|alchemist|battlemage|xe2?-hpg/.test( text ) ? 'discrete' : 'integrated';
	if ( /amd|ati\b|advanced micro/.test( vendor ) || /radeon/.test( text ) ) {

		if ( /vega \d+ graphics|radeon\(tm\) graphics|radeon graphics|\b\d{3}m\b|gcn/.test( text ) ) return 'integrated';
		return 'discrete';

	}

	if ( /\barm\b|qualcomm|adreno|mali|\bimg\b|imagination|powervr|samsung|xclipse|broadcom|videocore/.test( text ) ) return 'mobile';
	return 'unknown';

}

/**
 * The tier 'auto' starts with on a device, decided before the first frame from what the adapter
 * says (the governor corrects it afterwards, in both directions, see Governor): 'high' only for
 * GPUs known to be discrete or Apple silicon; integrated, mobile and unknown GPUs start at
 * 'medium' (a start that is too low costs a later step up, one that is too high costs the first
 * impression: seconds below 30 fps while the layout unfolds).
 */
export function autoTier( env = {} ) {

	const gpu = env.softwareAdapter ? 'software' : ( env.gpuClass || 'unknown' );
	if ( env.compat || gpu === 'software' ) return 'low';
	if ( env.coarsePointer && env.smallScreen ) return 'low'; // phones
	if ( env.coarsePointer || env.smallScreen ) return 'medium'; // tablets, small windows
	return gpu === 'discrete' || gpu === 'apple' ? 'high' : 'medium';

}

/**
 * The highest tier the governor may lift 'auto' to on a device: 'high' on desktops (never 'ultra':
 * depth of field, streak and grain are a look the user chooses), the starting tier on phones,
 * tablets, small windows and software rasterizers.
 */
export function autoCeiling( env = {} ) {

	const start = autoTier( env );
	if ( env.compat || env.softwareAdapter || env.gpuClass === 'software' || env.coarsePointer || env.smallScreen ) return start;
	return 'high';

}

/** Pixel-cap scales above this are rounded to 1 (native resolution). */
export const CAP_SNAP = 0.95;

/** Largest scene-pass scale (≤ 1) that keeps the rendered pixel count under `maxPixels`. */
export function pixelCapScale( maxPixels, cssWidth, cssHeight, dpr = 1 ) {

	const pixels = Math.max( 1, cssWidth * dpr ) * Math.max( 1, cssHeight * dpr );
	if ( ! ( maxPixels > 0 ) || pixels <= maxPixels ) return 1;
	const scale = Math.sqrt( maxPixels / pixels );
	// a few percent over the budget are not worth a resampled image
	return scale > CAP_SNAP ? 1 : scale;

}

export function normalizeSetting( setting ) {

	return TIER_ORDER.includes( setting ) ? setting : 'auto';

}

/**
 * RenderGraph options from URL parameters (URLSearchParams, or anything with get( name )):
 *   quality=auto|low|medium|high|ultra      governor=0|1      gputime=0|1 (GPU timestamp queries)
 *   ao= dof= shadows= traa= sharpen= streak= vignette= grain= ca= dynres= post=   (0|1)
 *   scale=<number>  fixed scene-pass resolution scale      debug=ao|depth|velocity|coc|none
 * Returns { quality?, governor?, timestamps?, overrides }; parameters that are absent stay absent,
 * so the tier defaults apply.
 */
export function parseGfxParams( params ) {

	const get = ( k ) => {

		const v = params && typeof params.get === 'function' ? params.get( k ) : null;
		return v === null || v === undefined || v === '' ? null : String( v );

	};

	const flag = ( k ) => {

		const v = get( k );
		return v === null ? null : ! [ '0', 'false', 'off', 'no' ].includes( v.toLowerCase() );

	};

	const out = { overrides: {} };
	const quality = get( 'quality' );
	if ( quality !== null ) out.quality = normalizeSetting( quality.toLowerCase() );
	const governor = flag( 'governor' );
	if ( governor !== null ) out.governor = governor;
	const gputime = flag( 'gputime' );
	if ( gputime !== null ) out.timestamps = gputime;
	const switches = { post: 'post', ao: 'ao', dof: 'dof', shadows: 'shadows', traa: 'traa', sharpen: 'sharpen', streak: 'streak', vignette: 'vignette', grain: 'grain', ca: 'chromaticAberration', dynres: 'dynamicResolution' };
	for ( const param in switches ) {

		const value = flag( param );
		if ( value !== null ) out.overrides[ switches[ param ] ] = value;

	}

	const scale = parseFloat( get( 'scale' ) );
	if ( scale > 0 ) out.overrides.scale = scale;
	const debug = get( 'debug' );
	if ( debug !== null && DEBUG_VIEWS.includes( debug ) ) out.overrides.debug = debug;
	return out;

}

export function resolveQuality( setting = 'auto', env = {}, overrides = {} ) {

	const e = { coarsePointer: false, smallScreen: false, softwareAdapter: false, compat: false, edges: 0, dpr: 1, cssWidth: 1280, cssHeight: 800, ...env };
	let o = overrides || {};
	setting = normalizeSetting( setting );
	const explicit = setting !== 'auto';
	const requested = explicit ? setting : ( TIER_ORDER.includes( o.tier ) ? o.tier : autoTier( e ) );
	// compatibility mode: the chains above 'low' are invalid there (see the header), and so are the
	// switches that would bring their passes back
	const compat = !! e.compat;
	const tier = compat ? 'low' : requested;
	if ( compat ) o = { ...o, ao: null, traa: null, dof: null, shadows: null, debug: null };
	const t = TIERS[ tier ];
	const c = { setting, tier, explicit, compat, requested, ...t, edges: { ...t.edges } };

	// per-graph and per-display rules of the table
	if ( c.aa === 'msaa' && e.edges > MSAA_MAX_EDGES ) {

		c.samples = 1;
		c.aa = 'smaa';

	}

	// pixels this small need no FXAA; the device decides (the drawing buffer may be capped below it,
	// then the image is upscaled anyway and a full-resolution pass buys nothing)
	if ( c.aa === 'fxaa' && ( e.deviceDpr > 0 ? e.deviceDpr : e.dpr ) >= c.fxaaMaxDpr ) c.aa = 'none';

	// forced switches
	if ( isSet( o.traa ) ) {

		if ( o.traa && c.aa !== 'traa' ) {

			c.aa = 'traa'; c.samples = 1; c.velocity = true;
			if ( c.ao ) {

				c.aoTemporal = true; c.aoDenoise = 'none';

			}

		} else if ( ! o.traa && c.aa === 'traa' ) {

			// the spatial fallback of a single-sample pass
			c.aa = 'smaa'; c.velocity = false;
			if ( c.ao ) {

				c.aoTemporal = false; c.aoDenoise = 'bilateral';

			}

		}

	}

	if ( isSet( o.ao ) ) {

		c.ao = !! o.ao;
		if ( c.ao && ! t.ao ) {

			// a tier without AO borrows the cheapest variant that fits its anti-aliasing
			c.aoTemporal = c.aa === 'traa';
			c.aoDenoise = c.aoTemporal ? 'none' : 'bilateral';

		}

	}

	if ( isSet( o.dof ) ) c.dof = !! o.dof;
	if ( isSet( o.shadows ) ) {

		c.shadows = !! o.shadows;
		if ( c.shadows && ! c.shadowMapSize ) {

			c.shadowMapSize = 1024; c.shadowRadius = 1;

		}

	}

	c.bloom = true;
	if ( isSet( o.sharpen ) ) c.sharpen = !! o.sharpen;
	if ( isSet( o.streak ) ) c.streak = !! o.streak;
	if ( isSet( o.vignette ) ) c.vignette = o.vignette === true || + o.vignette > 0;
	if ( isSet( o.grain ) ) c.grain = o.grain === true ? ( t.grain || 0.035 ) : Math.max( 0, + o.grain || 0 );
	if ( isSet( o.chromaticAberration ) ) c.chromaticAberration = o.chromaticAberration === true ? ( t.chromaticAberration || 0.15 ) : Math.max( 0, + o.chromaticAberration || 0 );
	if ( o.post === false ) {

		// plain output: scene → anti-aliasing → tone mapping → dither (→ overlay)
		c.ao = false; c.dof = false; c.bloom = false; c.streak = false; c.vignette = false; c.sharpen = false;
		c.grain = 0; c.chromaticAberration = 0;

	}

	if ( ! c.shadows ) {

		c.shadowMapSize = 0; c.shadowRadius = 0;

	}

	c.edges.defocus = c.dof; // strands defocus with the nodes wherever the depth of field is on, also when it is forced
	c.debug = DEBUG_VIEWS.includes( o.debug ) ? o.debug : null;

	// resolution: pixel cap of the tier, or a fixed scale
	c.capScale = pixelCapScale( c.maxPixels, e.cssWidth, e.cssHeight, e.dpr );
	c.fixedScale = isSet( o.scale ) && + o.scale > 0;
	c.renderScale = c.fixedScale ? Math.min( 2, Math.max( 0.1, + o.scale ) ) : c.capScale;
	c.dynamicResolution = ! c.fixedScale && ( isSet( o.dynamicResolution ) ? !! o.dynamicResolution : true );
	return c;

}

/** Two configs with the same key share one node graph (everything else is a uniform or a scale). */
export function structureKey( c ) {

	return [
		c.samples, c.aa, c.velocity ? 1 : 0,
		c.ao ? `${c.aoSamples}:${c.aoDenoise}:${c.aoTemporal ? 1 : 0}` : 0,
		c.dof ? c.dofTaps : 0,
		c.bloom ? 1 : 0, c.streak ? 1 : 0, c.vignette ? 1 : 0, c.grain > 0 ? 1 : 0, c.chromaticAberration > 0 ? 1 : 0, c.sharpen ? 1 : 0,
		c.debug || '-',
	].join( '|' );

}


// ─── Governor ───────────────────────────────────────────────────────────────────────────────

export const GOVERNOR_DEFAULTS = {
	targetFps: 60, // the rate aimed for where the display is at least that fast (a faster display is not chased)
	minTargetFps: 30, // the slowest display rate that is believed: below lowRatio × this a loop is slow however steady it runs
	ratePercentile: 0.15, // display rate of a window = 1000 / this quantile of its frame intervals …
	rateWindows: 20, // … and the display rate is the fastest of the last windows
	lowRatio: 0.75, // below this fraction of the display rate: too slow (45 of 60 fps)
	highRatio: 0.92, // above this fraction: fine (55 of 60 fps)
	windowMs: 1000, // decision window (sum of frame times) …
	minFrames: 10, // … with at least this many frames …
	maxWindowMs: 5000, // … but never longer than this (very low frame rates)
	settleMs: 1500, // ignored after configure(): pipeline compiles, history warm-up
	settleFrames: 8,
	dropMs: 2000, // a frame longer than this is a stall (hidden tab), not a rendering cost
	downWindows: 2, // consecutive slow windows before the scale drops
	upWindows: 8, // consecutive good windows before the scale climbs one step
	maxUpWindows: 128, // cap of the doubled waiting time after failed climbs
	downFactor: 0.85, // scale step down (pixels × 0.72)
	upFactor: 1.08,
	fastUpWindows: 2, // … while the pace is pinned at the display rate and no climb has failed: this soon …
	fastUpFactor: 1.2, // … and in steps of this size (never past a scale where a climb failed)
	pinnedRatio: 0.97, // at or above this fraction of the display rate the loop is pinned to it
	minScaleRatio: 0.5, // floor relative to maxScale (a quarter of the pixels)
	probeScale: 0.7, // half the pixels: moves even a rate that is locked to a divisor of the display rate, if the load is the pixels
	gainRatio: 1.12, // the frame rate must be this much better there, or the resolution is not the bottleneck
	retryMs: 60000, // resolution is then left alone for this long, doubled after every further failure …
	maxRetryMs: 1800000, // … up to half an hour
	discoverWindows: 8, // windows at a steady pace below targetFps before one step asks "display or GPU?" (without GPU timings)
	busyRatio: 0.5, // GPU time above this fraction of the frame time: the pace can be the GPU's (with GPU timings)
	tierDownWindows: 3, // slow windows at the lowest usable scale before the tier steps down
	trialWindows: 2, // windows a tier that was stepped down on trial has to show a gain
	transientMs: 45000, // longest stretch of transient frames (layout running) that is kept from the tier
	upgradeWindows: 8, // good windows at full scale before the tier above is tried (below the ceiling only)
	headroom: 0.35, // … with GPU timings: only while the render passes take less than this fraction of the frame budget
	upTrialWindows: 3, // windows the tier above must not be slow in, or the step is taken back
	upRetryMs: 60000, // pause before the tier above is tried again after a step down, doubled every time (≤ maxRetryMs)
};

export class Governor {

	constructor( options = {} ) {

		this.options = { ...GOVERNOR_DEFAULTS, ...options };
		this.enabled = true;
		this.fps = 0; // frame rate of the last decision window
		this.gpuMs = null;
		this.displayFps = 0; // measured display rate (0 = nothing measured yet)
		this.log = []; // the latest actions: { t, type, scale | tier, reason }
		this.upgraded = false; // the governor has stepped the tier up at least once
		this.downgraded = false; // its last tier step was a step down (it is tried again, see the header)
		this.scaleUseless = false; // a descent without gain: the resolution is left alone until the pause is over
		this.tierUseless = false; // a lower tier did not help either: the tier stays
		this._state = { tier: 'high', explicit: false, ceiling: 'high', scale: 1, maxScale: 1, dynamicResolution: true };
		this._intervals = new Float32Array( 256 ); // frame times of the running window
		this._rates = []; // display-rate estimates of the last windows
		this._settleFrames = this.options.settleFrames;
		this._settleUntil = null; // set from the first sample's clock
		this._pendingTier = null; // the tier of the governor's own last tier action
		this._transientSince = null; // clock time at which the running stretch of transient frames began
		this._renderMs = null; // GPU time of the render passes of the last sample
		this._resetWindow();
		this.reset();

	}

	/**
	 * Forgets what was measured about the load (descents, pauses, trials); the display rate stays.
	 * `tierHistory = false` keeps what is known about the tiers: the upgraded / downgraded flags and
	 * the pause before the tier above is tried again (a new graph on the same device).
	 */
	reset( tierHistory = true ) {

		this._descent = null; // { fps, ratio, gain } where the scale started to drop
		this._probe = null; // { fps, ratio, windows } the one step that asks "display or GPU?"
		this._trial = null; // { from, to, fps, windows } a tier stepped down on trial
		this._upTrial = null; // { from, to, windows } a tier stepped up, on probation
		this._lastClimb = null; // { from, windows } scale before the last step up, on probation
		this._climbFails = 0;
		this._climbCap = Infinity; // scale / maxScale at which a climb failed: fast climbs stay below it
		this._fails = 0; // descents without gain in a row
		this._retryAt = 0; // clock time at which the resolution may be tried again
		this._asks = 0; // "display or GPU?" steps answered "display" in a row
		this._askAt = 0; // clock time at which the question may be asked again
		this.scaleUseless = false;
		this.tierUseless = false;
		if ( tierHistory ) {

			this.upgraded = this.downgraded = false;
			this._upFails = 0; // steps down in a row that a step up was (or would have to be) taken back for
			this._upAt = 0; // clock time at which the tier above may be tried again

		}

		this._low = 0; this._high = 0; this._floor = 0; this._top = 0; this._paced = 0;
		return this;

	}

	/** The current pipeline state. Call after every build, tier change, resize or manual scale change. */
	configure( { tier, explicit, ceiling, scale, maxScale, dynamicResolution, enabled } = {} ) {

		const s = this._state;
		const tierChanged = tier !== undefined && tier !== s.tier;
		if ( tier !== undefined ) s.tier = tier;
		if ( explicit !== undefined ) s.explicit = !! explicit;
		if ( TIER_ORDER.includes( ceiling ) ) s.ceiling = ceiling;
		if ( maxScale !== undefined ) s.maxScale = maxScale;
		if ( scale !== undefined ) s.scale = scale;
		if ( dynamicResolution !== undefined ) s.dynamicResolution = !! dynamicResolution;
		if ( enabled !== undefined ) this.enabled = !! enabled;
		if ( tierChanged ) {

			// the scale starts again with a tier, and its measurements say nothing about another one
			this._descent = null; this._probe = null; this._lastClimb = null; this._climbFails = 0; this._climbCap = Infinity;
			if ( tier !== this._pendingTier ) {

				// not a step of the governor: nothing it knows applies any more
				this._trial = null;
				this._upTrial = null;
				this._forgetPauses();

			}

			this._pendingTier = null;

		}

		if ( ! s.dynamicResolution ) {

			this._descent = null; this._probe = null; this._lastClimb = null;

		}

		// a canvas resize (maxScale) keeps everything: scales are remembered relative to the cap
		this._low = this._high = this._floor = this._top = this._paced = 0;
		return this.settle();

	}

	/** Waits settleMs / settleFrames before measuring again (after a build, a resize, a paused loop). */
	settle() {

		this._resetWindow();
		this._settleFrames = this.options.settleFrames;
		this._settleUntil = null; // set from the next sample's clock
		return this;

	}

	get state() {

		return {
			...this._state, fps: this.fps, displayFps: this.displayFps, targetFps: this.targetFps,
			upgraded: this.upgraded, downgraded: this.downgraded, scaleUseless: this.scaleUseless, tierUseless: this.tierUseless, retryAt: this._retryAt, askAt: this._askAt, upAt: this._upAt,
		};

	}

	get minScale() {

		return this._state.maxScale * this.options.minScaleRatio;

	}

	/** The frame rate that counts as full speed: the measured display rate within minTargetFps … targetFps. */
	get targetFps() {

		const o = this.options;
		return this.displayFps > 0 ? Math.min( o.targetFps, Math.max( o.minTargetFps, this.displayFps ) ) : o.targetFps;

	}

	_resetWindow() {

		this._acc = 0; this._frames = 0; this._gpuAcc = 0; this._gpuFrames = 0;

	}

	/**
	 * One rendered frame: its duration and the caller's clock (both ms), optionally the GPU time
	 * of the frame (ms, timestamp queries) and { transient: the frame carries a load that will go
	 * away (layout running), renderMs: GPU time of the render passes alone }. Returns an action or null.
	 */
	sample( frameMs, now, gpuMs = null, info = null ) {

		const o = this.options;
		if ( ! this.enabled || ! ( frameMs > 0 ) ) return null;
		if ( info && info.transient ) {

			if ( this._transientSince === null ) this._transientSince = now;

		} else this._transientSince = null;

		this._renderMs = info && info.renderMs !== null && info.renderMs >= 0 ? info.renderMs : gpuMs;
		if ( frameMs > o.dropMs ) {

			this._resetWindow();
			return null;

		}

		if ( this._settleUntil === null ) this._settleUntil = now + o.settleMs;
		if ( this._settleFrames > 0 || now < this._settleUntil ) {

			this._settleFrames --;
			return null;

		}

		if ( this._frames < this._intervals.length ) this._intervals[ this._frames ] = frameMs;
		this._acc += frameMs; this._frames ++;
		if ( gpuMs !== null && gpuMs >= 0 ) {

			this._gpuAcc += gpuMs; this._gpuFrames ++;

		}

		if ( ! ( ( this._acc >= o.windowMs && this._frames >= o.minFrames ) || this._acc >= o.maxWindowMs ) ) return null;
		const fps = 1000 * this._frames / this._acc;
		const gpu = this._gpuFrames > 0 ? this._gpuAcc / this._gpuFrames : null;
		this._measureDisplay();
		this._resetWindow();
		this.fps = fps;
		this.gpuMs = gpu;
		// a stretch of transient frames is kept from the tier, but not for ever
		const transient = this._transientSince !== null && now - this._transientSince < o.transientMs;
		const action = this._decide( fps, gpu, now, transient );
		if ( action ) {

			this.log.push( { t: now, ...action } );
			if ( this.log.length > 256 ) this.log.shift();

		}

		return action;

	}

	/**
	 * Display rate from the window that just ended. Frames are presented on the display's refresh
	 * grid, so a loop that is slower than the display because of its load still shows intervals of
	 * one refresh now and then (unless it sits exactly on a divisor of the rate), while a capped
	 * loop never does. A low quantile instead of the minimum ignores the short catch-up frame after
	 * a hitch.
	 */
	_measureDisplay() {

		const o = this.options;
		const n = Math.min( this._frames, this._intervals.length );
		if ( n < 2 ) return;
		const sorted = this._intervals.subarray( 0, n ).sort(); // in place: the window is over
		const q = sorted[ Math.floor( o.ratePercentile * ( n - 1 ) ) ];
		if ( ! ( q > 0 ) ) return;
		this._rates.push( 1000 / q );
		if ( this._rates.length > o.rateWindows ) this._rates.shift();
		let best = 0;
		for ( const r of this._rates ) if ( r > best ) best = r;
		this.displayFps = best;

	}

	_decide( fps, gpu, now, transient = false ) {

		const o = this.options, s = this._state;
		const dyn = s.dynamicResolution;
		const floor = this.minScale;
		// The measured display rate decides what is slow as long as a step to half the pixels could
		// still put it to the test. Further down, the scale only got there because fewer pixels were
		// faster: the GPU set the pace on the way, and the nominal rate applies again.
		const room = ! dyn || s.scale * o.probeScale >= floor * 0.999;
		const target = room ? this.targetFps : o.targetFps;
		const slow = fps < target * o.lowRatio;
		const good = fps >= target * o.highRatio;
		// paced below the nominal rate: by the display, or by a GPU that is locked to a divisor of it
		const capped = target < o.targetFps * o.highRatio;
		if ( this.scaleUseless && now >= this._retryAt ) this.scaleUseless = false; // the pause is over
		const canDescend = dyn && ! this.scaleUseless;
		const atFloor = ! canDescend || s.scale <= floor * 1.001;
		const atTop = ! dyn || s.scale >= s.maxScale * 0.999;
		const ratio = s.scale / s.maxScale;

		this._low = slow ? this._low + 1 : 0;
		this._high = good ? this._high + 1 : 0;
		this._floor = slow && atFloor && ! transient ? this._floor + 1 : 0;
		this._top = good && atTop ? this._top + 1 : 0;
		this._paced = capped && ! slow ? this._paced + 1 : 0;

		// a tier that was stepped down on trial stays only if it pays
		const trial = this._trial;
		if ( trial ) {

			if ( s.tier !== trial.to ) this._trial = null;
			else if ( fps >= trial.fps * o.gainRatio ) {

				// it does; and in another tier the resolution may matter again
				this._trial = null;
				this._forgetPauses();
				return null;

			} else if ( ++ trial.windows >= o.trialWindows ) {

				this._trial = null;
				this.tierUseless = true;
				return this._setTier( trial.from, 'the lower tier is no faster' );

			} else return null;

		}

		// a tier that was stepped up stays unless it is slow
		const up = this._upTrial;
		if ( up ) {

			if ( s.tier !== up.to ) this._upTrial = null;
			else if ( slow ) {

				this._upTrial = null;
				this._failUp( now );
				this._low = 0; this._floor = 0;
				return this._setTier( up.from, 'the tier above is too slow' );

			} else if ( ++ up.windows >= o.upTrialWindows ) {

				this._upTrial = null; // it holds
				this.downgraded = false;

			} else return null;

		}

		// the step that asked "display or GPU?"
		const probe = this._probe;
		if ( probe ) {

			if ( fps >= probe.fps * o.gainRatio ) {

				// the GPU: fewer pixels are faster. The display rate measured from here on says how far to go.
				this._probe = null;
				this._asks = 0;
				return null;

			}

			if ( ++ probe.windows < o.downWindows ) return null;
			// the display (or something else that is not the pixels): ask again later, twice as late every
			// time. A loop that turns slow in the meantime is still free to try the resolution.
			this._probe = null;
			this._askAt = now + Math.min( o.maxRetryMs, o.retryMs * 2 ** this._asks );
			this._asks ++;
			this._paced = 0;
			return this._setScale( probe.ratio * s.maxScale, 'the display sets the pace' );

		}

		// a climb that made the frame rate drop: take it back at once and wait twice as long next time
		if ( this._lastClimb ) {

			if ( slow && dyn ) {

				const back = this._lastClimb.from;
				this._lastClimb = null;
				this._climbFails ++;
				this._climbCap = ratio;
				this._low = 0; this._high = 0;
				return this._setScale( back, 'climb failed' );

			}

			if ( good ) {

				// it held, and well: the load has changed, the next step need not wait as long
				this._lastClimb = null;
				this._climbFails = 0;
				if ( ratio >= this._climbCap * 0.999 ) this._climbCap = Infinity; // … and what failed before does not any more

			} else if ( ++ this._lastClimb.windows >= 2 ) this._lastClimb = null; // it held: accept it

		}

		// fewer pixels made it faster: the descent is worth continuing, and the next failure is a first one again
		const d = this._descent;
		if ( d && ! d.gain && fps >= d.fps * o.gainRatio ) {

			d.gain = true;
			this._fails = 0;

		}

		if ( slow ) {

			if ( canDescend && this._low >= o.downWindows ) {

				// half the pixels (or the floor) and no better: the resolution is not the bottleneck
				if ( d && ! d.gain && ( ratio <= d.ratio * o.probeScale * 1.001 || s.scale <= floor * 1.001 ) ) return this._giveUp( d.ratio * s.maxScale, now, 'scaling has no effect' );
				if ( s.scale > floor * 1.001 ) {

					const start = d || ( this._descent = { fps, ratio, gain: false } );
					let next = s.scale * o.downFactor;
					// until a gain was seen the descent ends at half the pixels, and its last step lands there
					const mark = start.ratio * s.maxScale * o.probeScale;
					if ( ! start.gain && s.scale > mark * 1.001 && next * o.downFactor < mark ) next = mark;
					this._low = 0;
					return this._setScale( Math.max( floor, next ), 'slow' );

				}

			}

			if ( atFloor && ! s.explicit && this._floor >= o.tierDownWindows ) {

				const i = TIER_ORDER.indexOf( s.tier );
				if ( i > 0 && canDescend ) {

					// the pixels are the load and a quarter of them is still too many: a cheaper tier
					this._failUp( now );
					this._floor = 0; this._low = 0;
					this._forgetPauses();
					return this._setTier( TIER_ORDER[ i - 1 ], 'slow at the lowest resolution' );

				}

				if ( i > 0 && ! this.tierUseless ) {

					// the resolution is not the reason (or must not change): fewer passes may be. On trial.
					this._failUp( now );
					this._floor = 0; this._low = 0;
					this._trial = { from: s.tier, to: TIER_ORDER[ i - 1 ], fps, windows: 0 };
					return this._setTier( TIER_ORDER[ i - 1 ], 'slow, and not because of the resolution' );

				}

			}

			return null;

		}

		if ( good ) {

			this._descent = null;
			if ( dyn && ! atTop ) {

				// The cause of the descent went away (a layout that came to rest, a thermal dip): the pace is
				// pinned at the display rate and no climb has failed. Larger steps, sooner, up to where one failed.
				const cap = this._climbCap * s.maxScale;
				const fast = this._climbFails === 0 && fps >= target * o.pinnedRatio && s.scale * o.fastUpFactor < cap * 0.999;
				const wait = fast ? o.fastUpWindows : Math.min( o.maxUpWindows, o.upWindows * 2 ** this._climbFails );
				if ( this._high >= wait ) {

					this._high = 0;
					this._lastClimb = { from: s.scale, windows: 0 };
					return this._setScale( Math.min( s.maxScale, s.scale * ( fast ? o.fastUpFactor : o.upFactor ) ), 'headroom' );

				}

				return null;

			}

		}

		// A steady pace below the nominal rate. The display's: nothing to do. A GPU-bound loop locked to
		// a divisor of a faster display: resolution would help. The GPU time tells them apart (busy for
		// more than half of the frame); without it one step to half the pixels does, with a doubling pause.
		if ( capped && canDescend && room && now >= this._askAt ) {

			const wait = gpu === null ? o.discoverWindows : ( gpu >= o.busyRatio * 1000 / fps ? o.downWindows : Infinity );
			if ( this._paced >= wait ) {

				this._paced = 0;
				this._lastClimb = null;
				this._probe = { fps, ratio, windows: 0 };
				return this._setScale( Math.max( floor, s.scale * o.probeScale ), 'display or GPU?' );

			}

		}

		// The tier above, below the ceiling only: after a good stretch at full scale, once the pause
		// since the last step down is over, and with GPU timings only while the render passes leave
		// headroom (the layout's compute passes are not what another tier adds to). On probation.
		if ( good && ! s.explicit && s.scale >= s.maxScale * 0.999 && this._top >= o.upgradeWindows && now >= this._upAt ) {

			const i = TIER_ORDER.indexOf( s.tier ), top = TIER_ORDER.indexOf( s.ceiling );
			const render = this._renderMs;
			if ( i >= 0 && i < top && ( gpu === null || render === null || render < o.headroom * 1000 / target ) ) {

				this.upgraded = true;
				this._top = 0;
				this._forgetPauses();
				this._upTrial = { from: s.tier, to: TIER_ORDER[ i + 1 ], windows: 0 };
				return this._setTier( TIER_ORDER[ i + 1 ], gpu === null ? 'trying the tier above' : 'GPU headroom' );

			}

		}

		return null;

	}

	/** A descent without gain: back to `scale`, and no further try for a while (doubled every time). */
	_giveUp( scale, now, reason ) {

		const o = this.options;
		this._descent = null; this._probe = null; this._lastClimb = null;
		this._retryAt = now + Math.min( o.maxRetryMs, o.retryMs * 2 ** this._fails );
		this._fails ++;
		this.scaleUseless = true;
		this._low = 0; this._paced = 0;
		return this._setScale( scale, reason );

	}

	/** The tier steps down: the tier above is not tried again before a pause that doubles every time. */
	_failUp( now ) {

		const o = this.options;
		this.downgraded = true;
		this._upAt = now + Math.min( o.maxRetryMs, o.upRetryMs * 2 ** this._upFails );
		this._upFails ++;

	}

	/** Another tier is another load: resolution may matter there, and the pace may be the GPU's. */
	_forgetPauses() {

		this._fails = 0; this._retryAt = 0; this.scaleUseless = false;
		this._asks = 0; this._askAt = 0;

	}

	_setScale( scale, reason ) {

		const s = this._state;
		scale = scale >= s.maxScale * 0.999 ? s.maxScale : Math.round( scale * 1000 ) / 1000;
		if ( Math.abs( scale - s.scale ) < 1e-3 ) return null;
		return { type: 'scale', scale, reason };

	}

	_setTier( tier, reason ) {

		this._pendingTier = tier;
		return { type: 'tier', tier, reason };

	}

}
