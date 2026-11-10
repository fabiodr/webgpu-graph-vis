// Quality tiers (src/app/Quality.js): the tier table, resolveQuality() and the Governor.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	TIERS, TIER_ORDER, MSAA_MAX_EDGES, resolveQuality, structureKey, autoTier, pixelCapScale, isSoftwareAdapter,
	normalizeSetting, parseGfxParams, Governor, gpuClass, autoCeiling,
} from '../src/app/Quality.js';

const DESKTOP = { coarsePointer: false, smallScreen: false, softwareAdapter: false, gpuClass: 'discrete', edges: 254, dpr: 1, cssWidth: 1280, cssHeight: 800 };
const PHONE = { coarsePointer: true, smallScreen: true, softwareAdapter: false, edges: 254, dpr: 2, cssWidth: 412, cssHeight: 915 };

test( 'tier table: the brief', () => {

	assert.deepEqual( TIER_ORDER, [ 'low', 'medium', 'high', 'ultra' ] );
	const { low, medium, high, ultra } = TIERS;
	// scene pass and anti-aliasing
	assert.deepEqual( [ low.samples, medium.samples, high.samples, ultra.samples ], [ 1, 4, 1, 1 ] );
	assert.deepEqual( [ low.aa, medium.aa, high.aa, ultra.aa ], [ 'fxaa', 'msaa', 'traa', 'traa' ] );
	assert.deepEqual( [ low.velocity, medium.velocity, high.velocity, ultra.velocity ], [ false, false, true, true ] );
	// lighting and shadows
	assert.deepEqual( [ low.lit, medium.lit, high.lit, ultra.lit ], [ false, true, true, true ] );
	assert.deepEqual( [ low.clearcoat, ultra.clearcoat ], [ false, true ] );
	assert.deepEqual( [ low.shadows, medium.shadowMapSize, high.shadowMapSize, ultra.shadowMapSize ], [ false, 1024, 2048, 2048 ] );
	assert.ok( ultra.shadowRadius > high.shadowRadius );
	// ambient occlusion
	assert.equal( low.ao, false );
	assert.deepEqual( [ medium.aoScale, medium.aoSamples, medium.aoTemporal ], [ 0.5, 16, false ] );
	assert.notEqual( medium.aoDenoise, 'none' );
	assert.deepEqual( [ high.aoScale, high.aoSamples, high.aoTemporal ], [ 0.5, 16, true ] );
	assert.deepEqual( [ ultra.aoScale, ultra.aoSamples, ultra.aoTemporal ], [ 1, 32, true ] );
	// depth of field, bloom, lens
	assert.deepEqual( [ low.dof, medium.dof, high.dof, ultra.dof ], [ false, false, false, true ] );
	assert.deepEqual( [ low.bloomScale, medium.bloomScale, high.bloomScale, ultra.bloomScale ], [ 0.25, 0.5, 0.5, 0.5 ] );
	assert.deepEqual( [ low.streak, high.streak, ultra.streak ], [ false, false, true ] );
	assert.deepEqual( [ low.grain > 0, medium.grain > 0, high.grain > 0 ], [ false, true, true ] );
	assert.ok( ultra.chromaticAberration > 0 && ultra.chromaticAberration <= 0.2 );
	assert.deepEqual( [ high.sharpen, ultra.sharpen ], [ false, true ] );
	// edges and pixel caps
	assert.deepEqual( low.edges, { lit: false, tubes: false, defocus: false } );
	assert.deepEqual( ultra.edges, { lit: true, tubes: true, defocus: true } );
	assert.deepEqual( [ low.maxPixels, medium.maxPixels, high.maxPixels, ultra.maxPixels ], [ 1.5e6, 2.5e6, 4e6, 6e6 ] );
	// MRT is never combined with MSAA
	for ( const name of TIER_ORDER ) assert.ok( ! ( TIERS[ name ].velocity && TIERS[ name ].samples > 1 ), name );

} );

test( 'autoTier and software adapters', () => {

	assert.equal( autoTier( DESKTOP ), 'high' );
	assert.equal( autoTier( PHONE ), 'low' );
	assert.equal( autoTier( { ...DESKTOP, coarsePointer: true } ), 'medium' ); // tablet
	assert.equal( autoTier( { ...DESKTOP, smallScreen: true } ), 'medium' ); // small window
	assert.equal( autoTier( { ...DESKTOP, softwareAdapter: true } ), 'low' );
	// the adapter decides before the first frame: only known discrete GPUs and Apple silicon start at high
	assert.equal( autoTier( { ...DESKTOP, gpuClass: 'apple' } ), 'high' );
	for ( const gpu of [ 'integrated', 'mobile', 'unknown', undefined ] ) assert.equal( autoTier( { ...DESKTOP, gpuClass: gpu } ), 'medium', String( gpu ) );
	assert.equal( autoTier( { ...DESKTOP, gpuClass: 'software' } ), 'low' );
	assert.equal( autoTier( { ...DESKTOP, compat: true } ), 'low' );
	// … and the governor may lift a desktop to high, never to ultra, and nothing else above its start
	assert.equal( autoCeiling( { ...DESKTOP, gpuClass: 'integrated' } ), 'high' );
	assert.equal( autoCeiling( { ...DESKTOP, gpuClass: 'unknown' } ), 'high' );
	assert.equal( autoCeiling( DESKTOP ), 'high' );
	assert.equal( autoCeiling( PHONE ), 'low' );
	assert.equal( autoCeiling( { ...DESKTOP, coarsePointer: true } ), 'medium' );
	assert.equal( autoCeiling( { ...DESKTOP, softwareAdapter: true } ), 'low' );
	assert.equal( autoCeiling( { ...DESKTOP, compat: true } ), 'low' );
	// GPUAdapterInfo → class
	assert.equal( gpuClass( { vendor: 'nvidia', architecture: 'lovelace' } ), 'discrete' );
	assert.equal( gpuClass( { vendor: 'apple', architecture: 'metal-3' } ), 'apple' );
	assert.equal( gpuClass( { vendor: 'intel', architecture: 'gen-12lp' } ), 'integrated' );
	assert.equal( gpuClass( { vendor: 'intel', architecture: 'xe-lpg' } ), 'integrated' );
	assert.equal( gpuClass( { vendor: 'intel', architecture: 'xe-hpg' } ), 'discrete' );
	assert.equal( gpuClass( { vendor: 'intel', description: 'Intel(R) Arc(TM) A770 Graphics' } ), 'discrete' );
	assert.equal( gpuClass( { vendor: 'amd', architecture: 'rdna-3' } ), 'discrete' );
	assert.equal( gpuClass( { vendor: 'amd', architecture: 'gcn-5' } ), 'integrated' );
	assert.equal( gpuClass( { vendor: 'amd', architecture: 'rdna-2', description: 'AMD Radeon(TM) Graphics' } ), 'integrated' );
	assert.equal( gpuClass( { vendor: 'amd', description: 'AMD Radeon 680M' } ), 'integrated' );
	assert.equal( gpuClass( { vendor: 'qualcomm', architecture: 'adreno-7xx' } ), 'mobile' );
	assert.equal( gpuClass( { vendor: 'arm', architecture: 'valhall' } ), 'mobile' );
	assert.equal( gpuClass( { vendor: 'img-tec' } ), 'mobile' );
	assert.equal( gpuClass( { vendor: 'acme' } ), 'unknown' );
	assert.equal( gpuClass( { vendor: 'imagination technologies' } ), 'mobile' );
	assert.equal( gpuClass( { vendor: 'google', architecture: 'swiftshader' } ), 'software' );
	assert.equal( gpuClass( { vendor: '', architecture: '' } ), 'unknown' );
	assert.equal( gpuClass( {} ), 'unknown' );
	assert.equal( gpuClass( null ), 'unknown' );
	assert.equal( isSoftwareAdapter( { vendor: 'google', architecture: 'swiftshader' } ), true );
	assert.equal( isSoftwareAdapter( { vendor: 'mesa', description: 'llvmpipe (LLVM 15)' } ), true );
	assert.equal( isSoftwareAdapter( { vendor: 'nvidia', architecture: 'lovelace', isFallbackAdapter: false } ), false );
	assert.equal( isSoftwareAdapter( { vendor: 'apple', isFallbackAdapter: true } ), true );
	assert.equal( isSoftwareAdapter( null ), false );
	assert.equal( normalizeSetting( 'ultra' ), 'ultra' );
	assert.equal( normalizeSetting( 'bogus' ), 'auto' );
	assert.equal( normalizeSetting( undefined ), 'auto' );

} );

test( 'resolveQuality: auto, explicit tiers and the per-graph rules', () => {

	const a = resolveQuality( 'auto', DESKTOP );
	assert.deepEqual( [ a.setting, a.tier, a.explicit ], [ 'auto', 'high', false ] );
	assert.equal( resolveQuality( 'auto', PHONE ).tier, 'low' );
	assert.equal( resolveQuality( 'auto', { ...DESKTOP, softwareAdapter: true } ).tier, 'low' );
	assert.equal( resolveQuality( 'nonsense', DESKTOP ).setting, 'auto' );
	// an explicit tier is taken as it is, whatever the device
	const u = resolveQuality( 'ultra', { ...PHONE, softwareAdapter: true } );
	assert.deepEqual( [ u.tier, u.explicit, u.aa, u.velocity, u.dof, u.sharpen ], [ 'ultra', true, 'traa', true, true, true ] );
	// an integrated or masked GPU starts 'auto' at medium
	assert.equal( resolveQuality( 'auto', { ...DESKTOP, gpuClass: 'integrated' } ).tier, 'medium' );
	assert.equal( resolveQuality( 'auto', { ...DESKTOP, gpuClass: undefined } ).tier, 'medium' );
	// compatibility mode: 'low' is the only valid chain, whatever is asked for or forced
	for ( const setting of [ 'auto', 'medium', 'high', 'ultra' ] ) {

		const c = resolveQuality( setting, { ...DESKTOP, compat: true }, { ao: true, traa: true, dof: true, shadows: true, debug: 'ao', tier: 'high' } );
		assert.deepEqual( [ c.tier, c.compat, c.samples, c.velocity, c.ao, c.dof, c.shadows, c.debug ], [ 'low', true, 1, false, false, false, false, null ], setting );
		assert.ok( c.aa === 'fxaa' || c.aa === 'none' );
		assert.equal( c.requested, setting === 'auto' ? 'high' : setting );

	}

	assert.deepEqual( [ a.compat, a.requested ], [ false, 'high' ] );
	// the governor's tier only applies to 'auto'
	assert.equal( resolveQuality( 'auto', DESKTOP, { tier: 'medium' } ).tier, 'medium' );
	assert.equal( resolveQuality( 'ultra', DESKTOP, { tier: 'medium' } ).tier, 'ultra' );
	// medium: MSAA up to 20k edges, single sample + SMAA above
	const m1 = resolveQuality( 'medium', { ...DESKTOP, edges: MSAA_MAX_EDGES } );
	const m2 = resolveQuality( 'medium', { ...DESKTOP, edges: MSAA_MAX_EDGES + 1 } );
	assert.deepEqual( [ m1.samples, m1.aa ], [ 4, 'msaa' ] );
	assert.deepEqual( [ m2.samples, m2.aa ], [ 1, 'smaa' ] );
	// low: FXAA below DPR 2.5
	assert.equal( resolveQuality( 'low', { ...PHONE, dpr: 2 } ).aa, 'fxaa' );
	assert.equal( resolveQuality( 'low', { ...PHONE, dpr: 2.625 } ).aa, 'none' );
	// … of the device: the App caps the renderer pixel ratio at 2, a DPR 2.625 or 3 phone still needs none
	assert.equal( resolveQuality( 'low', { ...PHONE, dpr: 2, deviceDpr: 2.625 } ).aa, 'none' );
	assert.equal( resolveQuality( 'low', { ...PHONE, dpr: 2, deviceDpr: 3 } ).aa, 'none' );
	assert.equal( resolveQuality( 'low', { ...PHONE, dpr: 2, deviceDpr: 2 } ).aa, 'fxaa' );
	assert.equal( resolveQuality( 'low', { ...DESKTOP, dpr: 1, deviceDpr: 1.25 } ).aa, 'fxaa' );
	// the pixel cap keeps using the renderer pixel ratio
	assert.equal( resolveQuality( 'low', { ...PHONE, dpr: 2, deviceDpr: 3 } ).capScale, resolveQuality( 'low', { ...PHONE, dpr: 2 } ).capScale );
	// the returned object is a copy
	a.edges.lit = false; a.samples = 99;
	assert.equal( TIERS.high.edges.lit, true );
	assert.equal( TIERS.high.samples, 1 );

} );

test( 'resolveQuality: forced switches', () => {

	const h = resolveQuality( 'high', DESKTOP );
	assert.deepEqual( [ h.ao, h.aoTemporal, h.dof, h.shadows, h.shadowMapSize ], [ true, true, false, true, 2048 ] );
	assert.equal( resolveQuality( 'high', DESKTOP, { ao: false } ).ao, false );
	assert.equal( resolveQuality( 'high', DESKTOP, { dof: true } ).dof, true );
	const ns = resolveQuality( 'high', DESKTOP, { shadows: false } );
	assert.deepEqual( [ ns.shadows, ns.shadowMapSize ], [ false, 0 ] );
	const ls = resolveQuality( 'low', DESKTOP, { shadows: true } );
	assert.deepEqual( [ ls.shadows, ls.shadowMapSize > 0 ], [ true, true ] );
	// TRAA off in a TRAA tier: spatial AA, no velocity, AO loses its temporal filter
	const nt = resolveQuality( 'high', DESKTOP, { traa: false } );
	assert.deepEqual( [ nt.aa, nt.velocity, nt.samples, nt.aoTemporal, nt.aoDenoise ], [ 'smaa', false, 1, false, 'bilateral' ] );
	// TRAA on in the MSAA tier: MSAA goes (TRAA needs a single-sample pass)
	const mt = resolveQuality( 'medium', DESKTOP, { traa: true } );
	assert.deepEqual( [ mt.aa, mt.velocity, mt.samples, mt.aoTemporal ], [ 'traa', true, 1, true ] );
	// AO forced on in the tier without AO
	const la = resolveQuality( 'low', DESKTOP, { ao: true } );
	assert.deepEqual( [ la.ao, la.aoTemporal, la.aoDenoise ], [ true, false, 'bilateral' ] );
	// strengths that are structural at zero
	assert.equal( resolveQuality( 'ultra', DESKTOP, { chromaticAberration: 0 } ).chromaticAberration, 0 );
	assert.equal( resolveQuality( 'low', DESKTOP, { grain: true } ).grain > 0, true );
	assert.equal( resolveQuality( 'high', DESKTOP, { grain: 0 } ).grain, 0 );
	assert.equal( resolveQuality( 'high', DESKTOP, { vignette: 0 } ).vignette, false );
	// strand defocus follows the depth of field, also a forced one
	assert.equal( resolveQuality( 'ultra', DESKTOP ).edges.defocus, true );
	assert.equal( resolveQuality( 'ultra', DESKTOP, { dof: false } ).edges.defocus, false );
	assert.equal( resolveQuality( 'high', DESKTOP ).edges.defocus, false );
	assert.equal( resolveQuality( 'high', DESKTOP, { dof: true } ).edges.defocus, true );
	// null / undefined mean "tier default"
	assert.deepEqual( resolveQuality( 'ultra', DESKTOP, { ao: null, dof: undefined, traa: null } ), resolveQuality( 'ultra', DESKTOP ) );
	// debug views
	assert.equal( resolveQuality( 'high', DESKTOP, { debug: 'ao' } ).debug, 'ao' );
	assert.equal( resolveQuality( 'high', DESKTOP, { debug: 'none' } ).debug, null );

} );

test( 'resolveQuality: pixel cap and fixed scale', () => {

	assert.equal( pixelCapScale( 4e6, 1280, 800, 1 ), 1 );
	const k = pixelCapScale( 4e6, 1920, 1080, 2 ); // 8.29 MP
	assert.ok( Math.abs( k * k * 3840 * 2160 - 4e6 ) < 1 );
	const c = resolveQuality( 'high', { ...DESKTOP, cssWidth: 1920, cssHeight: 1080, dpr: 2 } );
	assert.ok( Math.abs( c.capScale - k ) < 1e-12 );
	assert.equal( c.renderScale, c.capScale );
	assert.equal( c.dynamicResolution, true );
	// ultra has the larger budget
	assert.ok( resolveQuality( 'ultra', { ...DESKTOP, cssWidth: 1920, cssHeight: 1080, dpr: 2 } ).renderScale > c.renderScale );
	// phones: 412×915 at DPR 2 is 1.508 MP, a hair over the cap: native resolution, not 99.7 % of it
	assert.equal( resolveQuality( 'low', PHONE ).renderScale, 1 );
	assert.equal( pixelCapScale( 1.5e6, 412, 915, 2 ), 1 );
	assert.ok( pixelCapScale( 1.5e6, 440, 960, 2 ) < 0.95 );
	// a fixed scale wins over the cap and turns dynamic resolution off
	const f = resolveQuality( 'high', { ...DESKTOP, cssWidth: 1920, cssHeight: 1080, dpr: 2 }, { scale: 0.5 } );
	assert.deepEqual( [ f.renderScale, f.fixedScale, f.dynamicResolution ], [ 0.5, true, false ] );
	assert.equal( resolveQuality( 'high', DESKTOP, { dynamicResolution: false } ).dynamicResolution, false );

} );

test( 'structureKey: what needs a rebuild', () => {

	const h = resolveQuality( 'high', DESKTOP );
	const same = [
		resolveQuality( 'high', { ...DESKTOP, cssWidth: 3000, cssHeight: 2000 } ), // other scale
		resolveQuality( 'high', DESKTOP, { shadows: false } ), // shadows live in the materials
		resolveQuality( 'high', DESKTOP, { scale: 0.7 } ),
		resolveQuality( 'high', DESKTOP, { grain: 0.1 } ),
	];
	for ( const c of same ) assert.equal( structureKey( c ), structureKey( h ) );
	const different = [
		resolveQuality( 'medium', DESKTOP ), resolveQuality( 'low', DESKTOP ), resolveQuality( 'ultra', DESKTOP ),
		resolveQuality( 'high', DESKTOP, { ao: false } ), resolveQuality( 'high', DESKTOP, { dof: true } ),
		resolveQuality( 'high', DESKTOP, { traa: false } ), resolveQuality( 'high', DESKTOP, { grain: 0 } ),
		resolveQuality( 'high', DESKTOP, { debug: 'depth' } ), resolveQuality( 'high', DESKTOP, { sharpen: true } ),
	];
	const keys = new Set( different.map( structureKey ) );
	assert.equal( keys.size, different.length );
	assert.ok( ! keys.has( structureKey( h ) ) );

} );

// ─── Governor ───────────────────────────────────────────────────────────────────────────────

/**
 * Drives a governor with a synthetic device: fpsOf( state ) → frames per second the device could
 * render for the current { tier, scale }. Applies the actions the way the render graph does and
 * returns them.
 *   hz > 0   a fixed-rate display: frames start on its refresh grid, the work is pipelined (the
 *            average rate is the device's, at most hz; a slower device shows a mix of intervals)
 *   hz = 0   no grid: every frame takes exactly 1000 / fps (a loop locked to a divisor of the
 *            display rate, or a variable-rate display)
 *   gpuMsOf( state, fps ) → GPU time per frame (timestamp queries), or leave it out
 *   infoOf( state, now ) → the fourth argument of sample(): { transient, renderMs }
 */
function simulate( gov, state, fpsOf, seconds, { gpuMsOf = null, infoOf = null, maxScaleOf = () => state.maxScale, hz = 60 } = {} ) {

	const actions = [];
	let now = state.now || 0;
	const start = now, end = now + seconds * 1000;
	let tick = 0, ideal = 0; // in refresh intervals
	gov.configure( state );
	while ( now < end ) {

		const fps = fpsOf( state, now );
		let ms = 1000 / fps;
		if ( hz > 0 ) {

			const v = 1000 / hz;
			ideal = Math.max( ideal, tick - 1 ) + ms / v;
			const next = Math.max( tick + 1, Math.ceil( ideal - 1e-9 ) );
			ms = ( next - tick ) * v;
			tick = next;
			now = start + tick * v;

		} else now += ms;

		const a = gov.sample( ms, now, gpuMsOf ? gpuMsOf( state, fps ) : null, infoOf ? infoOf( state, now ) : null );
		if ( ! a ) continue;
		actions.push( { t: now, ...a } );
		if ( a.type === 'scale' ) state.scale = a.scale;
		if ( a.type === 'tier' ) {

			state.tier = a.tier;
			state.maxScale = maxScaleOf( state );
			state.scale = state.maxScale;

		}

		gov.configure( state );

	}

	state.now = now;
	return actions;

}

const fresh = ( over = {} ) => ( { tier: 'high', explicit: false, scale: 1, maxScale: 1, dynamicResolution: true, enabled: true, ...over } );
// a fill-bound device: frame time proportional to the pixel count
const fillBound = ( fpsAtFull, tierCost = { low: 0.4, medium: 0.7, high: 1, ultra: 1.6 } ) => ( s ) => Math.min( 60, fpsAtFull / ( tierCost[ s.tier ] * s.scale * s.scale ) );
const scales = ( actions ) => actions.filter( ( a ) => a.type === 'scale' );
const tiers = ( actions ) => actions.filter( ( a ) => a.type === 'tier' ).map( ( a ) => a.tier );
/** Seconds of [ from, to ] spent below the full scale, from an action log that starts at full scale. */
function secondsBelowFull( actions, to ) {

	let below = 0, since = null;
	for ( const a of scales( actions ) ) {

		if ( a.scale < 0.999 && since === null ) since = a.t;
		if ( a.scale >= 0.999 && since !== null ) {

			below += a.t - since;
			since = null;

		}

	}

	if ( since !== null ) below += to - since;
	return below / 1000;

}

test( 'governor: a fast device is left alone', () => {

	const gov = new Governor();
	const actions = simulate( gov, fresh(), () => 60, 120 );
	assert.deepEqual( actions, [] );
	assert.ok( Math.abs( gov.fps - 60 ) < 0.5 );
	assert.ok( Math.abs( gov.displayFps - 60 ) < 0.5 );
	assert.equal( gov.targetFps, 60 );
	// a faster display is not chased: 60 fps on it is fine, and so is its full rate
	for ( const fps of [ 60, 500 ] ) {

		const g2 = new Governor();
		assert.deepEqual( simulate( g2, fresh(), () => fps, 120, { hz: 120 } ), [] );
		assert.ok( Math.abs( g2.displayFps - Math.min( fps, 120 ) ) < 1 );
		assert.equal( g2.targetFps, 60 );

	}

} );

test( 'governor: disabled, it never acts', () => {

	const gov = new Governor();
	gov.enabled = false;
	assert.deepEqual( simulate( gov, fresh( { enabled: false } ), () => 12, 120 ), [] );
	// configure( { enabled } ) is the switch
	const g2 = new Governor();
	g2.configure( { enabled: false } );
	assert.equal( g2.enabled, false );

} );

test( 'governor: sustained low frame rate lowers the scale in steps, then holds', () => {

	const gov = new Governor();
	const state = fresh();
	const actions = simulate( gov, state, fillBound( 36 ), 60 );
	assert.ok( actions.length >= 1 && actions.length <= 4, `steps: ${actions.length}` );
	assert.ok( actions.every( ( a ) => a.type === 'scale' ) );
	for ( let i = 1; i < actions.length; i ++ ) assert.ok( actions[ i ].scale < actions[ i - 1 ].scale );
	// the mixed frame intervals of a slow loop on a 60 Hz display give the display rate away
	assert.ok( Math.abs( gov.displayFps - 60 ) < 0.5 );
	// it stops as soon as the rate is acceptable (≥ 45 fps), well above the floor
	assert.ok( fillBound( 36 )( state ) >= 45 );
	assert.ok( state.scale > 0.5 && state.scale < 1 );
	// the first step needs the settle time plus two slow windows
	assert.ok( actions[ 0 ].t > 1500 + 2000 * 0.9 );
	assert.equal( state.tier, 'high' );

} );

test( 'governor: no thrash between two scales', () => {

	// 50 fps at full scale would be fine, a slightly lower scale reaches 60 and tempts a climb back
	const gov = new Governor();
	const state = fresh();
	const device = ( s ) => Math.min( 60, 40 / ( s.scale * s.scale ) );
	const actions = simulate( gov, state, device, 3600 ); // one hour
	// the climb is tried again with doubled waiting times: a handful of changes, not hundreds
	assert.ok( actions.length <= 24, `changes in an hour: ${actions.length}` );
	const lastTen = actions.filter( ( a ) => a.t > 1800e3 );
	assert.ok( lastTen.length <= 8, `changes in the second half hour: ${lastTen.length}` );
	assert.ok( device( state ) >= 45 );
	assert.ok( actions.every( ( a ) => a.type === 'scale' ) );

} );

test( 'governor: at the floor the tier steps down (auto only); the way back is tried with doubling pauses', () => {

	const gov = new Governor();
	const state = fresh();
	// even a quarter of the pixels is too slow in 'high'; 'medium' at full scale is fine
	const device = fillBound( 9, { low: 0.1, medium: 0.12, high: 1, ultra: 2 } );
	const actions = simulate( gov, state, device, 60 );
	assert.deepEqual( tiers( actions ), [ 'medium' ] );
	assert.equal( state.tier, 'medium' );
	assert.equal( gov.downgraded, true );
	assert.equal( gov.tierUseless, false );
	// before the step the scale went down to the floor
	const before = actions.slice( 0, actions.findIndex( ( a ) => a.type === 'tier' ) );
	assert.ok( before.length >= 3 && Math.abs( before[ before.length - 1 ].scale - 0.5 ) < 1e-3 );
	for ( let i = 1; i < before.length; i ++ ) assert.ok( before[ i ].scale < before[ i - 1 ].scale );
	// the device stays what it is: 'high' is tried again and taken back within a few seconds, each
	// time after twice the pause. A handful of short trials in an hour, and it ends in 'medium'
	const later = simulate( gov, state, device, 3600 );
	const ups = later.filter( ( a ) => a.type === 'tier' && a.tier === 'high' );
	const downs = later.filter( ( a ) => a.type === 'tier' && a.tier === 'medium' );
	assert.ok( ups.length >= 2 && ups.length <= 6, `trials in an hour: ${ups.length}` );
	assert.equal( downs.length, ups.length );
	assert.equal( state.tier, 'medium' );
	assert.ok( later.every( ( a ) => a.type === 'tier' ), 'a trial is taken back before the scale moves' );
	let inHigh = 0;
	for ( let i = 0; i < ups.length; i ++ ) inHigh += downs[ i ].t - ups[ i ].t;
	assert.ok( inHigh / ups.length < 6000, `ms per failed trial: ${inHigh / ups.length}` );
	for ( let i = 2; i < ups.length; i ++ ) assert.ok( ups[ i ].t - ups[ i - 1 ].t > 1.7 * ( ups[ i - 1 ].t - ups[ i - 2 ].t ) - 20000, 'the pauses double' );
	// the first try waits for the pause after the step down
	assert.ok( ups[ 0 ].t - actions[ actions.length - 1 ].t >= 60000 );

} );

test( 'governor: a demotion by a load that went away is undone', () => {

	// the reviewer's case: no GPU timings, a thermal dip (or a load nobody flagged) long enough to
	// reach the floor and the tier below, then a device that is fast again
	for ( const timings of [ false, true ] ) {

		const gov = new Governor();
		const state = fresh();
		const device = ( s, now ) => ( now < 40000 ? fillBound( 9 )( s ) : 60 );
		const gpuMsOf = timings ? ( s, fps ) => ( fps < 59 ? 950 / fps : 3 ) : null;
		const actions = simulate( gov, state, device, 240, { gpuMsOf } );
		assert.ok( tiers( actions ).includes( 'medium' ), 'demoted during the dip' );
		assert.equal( state.tier, 'high', `back in high (timings ${timings})` );
		assert.equal( state.scale, 1 );
		assert.equal( gov.downgraded, false );
		const back = actions.filter( ( a ) => a.type === 'tier' && a.tier === 'high' );
		assert.equal( back.length, 1 );
		assert.ok( back[ 0 ].t < 150000, `back after ${back[ 0 ].t / 1000} s` );
		// and it stays: nothing more happens
		assert.deepEqual( simulate( gov, state, () => 60, 600, { gpuMsOf } ), [] );

	}

	// with GPU timings a GPU without headroom in the lower tier is not sent up to fail
	const gov = new Governor();
	const state = fresh( { tier: 'medium' } );
	assert.deepEqual( simulate( gov, state, () => 60, 600, { gpuMsOf: () => 9 } ), [] );
	// … where the headroom test looks at the render passes, not at the layout's compute passes
	const g2 = new Governor();
	const s2 = fresh( { tier: 'medium' } );
	const a2 = simulate( g2, s2, () => 60, 60, { gpuMsOf: () => 9, infoOf: () => ( { renderMs: 3 } ) } );
	assert.deepEqual( tiers( a2 ), [ 'high' ] );

} );

test( 'governor: a masked adapter starts at medium and is lifted to the ceiling, never above', () => {

	// no GPU timings (the browsers that mask the adapter have none either)
	const gov = new Governor();
	const state = fresh( { tier: 'medium', ceiling: 'high' } );
	const actions = simulate( gov, state, () => 60, 600 );
	assert.deepEqual( actions.map( ( a ) => [ a.type, a.tier ] ), [ [ 'tier', 'high' ] ] );
	assert.ok( actions[ 0 ].t < 15000, `lifted after ${actions[ 0 ].t / 1000} s` );
	assert.equal( gov.upgraded, true );
	// 'high' is the default ceiling: 'ultra' is a look the user chooses
	assert.deepEqual( simulate( new Governor(), fresh(), () => 60, 600, { gpuMsOf: () => 1 } ), [] );
	// a tablet or phone keeps its starting tier
	assert.deepEqual( simulate( new Governor(), fresh( { tier: 'medium', ceiling: 'medium' } ), () => 60, 600, { gpuMsOf: () => 1 } ), [] );
	assert.deepEqual( simulate( new Governor(), fresh( { tier: 'low', ceiling: 'low' } ), () => 60, 600 ), [] );
	// a device that cannot hold the tier above: one short trial, then the pause
	const g2 = new Governor();
	const s2 = fresh( { tier: 'medium' } );
	const device = ( s ) => ( s.tier === 'medium' ? 60 : 25 );
	const a2 = simulate( g2, s2, device, 60 );
	assert.deepEqual( tiers( a2 ), [ 'high', 'medium' ] );
	assert.equal( a2[ 1 ].reason, 'the tier above is too slow' );
	assert.ok( a2[ 1 ].t - a2[ 0 ].t < 5000 );

} );

test( 'governor: a transient load moves the scale, not the tier', () => {

	// the layout of a large graph: 20 ms on top of every frame for 40 s, flagged by the caller
	const cost = { low: 4, medium: 8, high: 14, ultra: 22 };
	const layoutUntil = 40000;
	const device = ( s, now ) => 1000 / Math.max( 16.7, ( now < layoutUntil ? 20 : 0 ) + cost[ s.tier ] * s.scale * s.scale );
	const gov = new Governor();
	const state = fresh();
	const actions = simulate( gov, state, device, 100, { infoOf: ( s, now ) => ( { transient: now < layoutUntil } ) } );
	assert.deepEqual( tiers( actions ), [], 'the tier is left alone' );
	assert.equal( state.tier, 'high' );
	assert.ok( actions.some( ( a ) => a.type === 'scale' && a.scale < 1 ), 'the scale did react' );
	// … and is back at full resolution soon after the layout came to rest
	const full = actions.filter( ( a ) => a.t > layoutUntil && a.scale === 1 );
	assert.ok( full.length >= 1 && full[ 0 ].t < layoutUntil + 30000, `full resolution ${full.length ? ( full[ 0 ].t - layoutUntil ) / 1000 : '∞'} s after the layout` );
	assert.equal( state.scale, 1 );
	// without the flag the same load costs the tier (and the retry rule brings it back)
	const g2 = new Governor();
	const s2 = fresh();
	const a2 = simulate( g2, s2, device, 60 );
	assert.ok( tiers( a2 ).includes( 'medium' ) );
	// a "transient" that never ends is a slow device: after transientMs the tier may step down
	const g3 = new Governor();
	const s3 = fresh();
	const a3 = simulate( g3, s3, fillBound( 9, { low: 0.1, medium: 0.12, high: 1, ultra: 2 } ), 90, { infoOf: () => ( { transient: true } ) } );
	assert.deepEqual( tiers( a3 ), [ 'medium' ] );
	assert.ok( a3.find( ( a ) => a.type === 'tier' ).t > 45000 );

} );

test( 'governor: an explicit tier is never changed', () => {

	const gov = new Governor();
	const state = fresh( { tier: 'ultra', explicit: true } );
	const actions = simulate( gov, state, fillBound( 5 ), 300, { gpuMsOf: ( s, fps ) => 950 / fps } );
	assert.ok( actions.length > 0 );
	assert.ok( actions.every( ( a ) => a.type === 'scale' ) );
	assert.equal( state.tier, 'ultra' );
	assert.ok( Math.abs( state.scale - 0.5 ) < 1e-3 ); // scale floor, then nothing more
	// without dynamic resolution not even the scale
	const g2 = new Governor();
	assert.deepEqual( simulate( g2, fresh( { tier: 'ultra', explicit: true, dynamicResolution: false } ), fillBound( 5 ), 300 ), [] );
	// and a fast explicit tier is not upgraded
	const g3 = new Governor();
	assert.deepEqual( simulate( g3, fresh( { tier: 'medium', explicit: true } ), () => 60, 600, { gpuMsOf: () => 1 } ), [] );

} );

test( 'governor: a loop capped at 30 fps is not slow', () => {

	// a 30 Hz display, a battery saver or a browser that halves the rate; the device itself is fast.
	// With GPU timings the idle GPU settles it: nothing happens at all, whatever the tier setting
	for ( const explicit of [ true, false ] ) {

		const gov = new Governor();
		const state = fresh( { explicit } );
		const actions = simulate( gov, state, () => 200, 900, { hz: 30, gpuMsOf: () => 5 } );
		assert.deepEqual( scales( actions ), [], `explicit ${explicit}` );
		assert.ok( Math.abs( gov.displayFps - 30 ) < 0.5 );
		assert.ok( Math.abs( gov.targetFps - 30 ) < 0.5 );
		assert.equal( state.scale, 1 );
		assert.deepEqual( tiers( actions ), [] );

	}

	// Without GPU timings the pace could also be a GPU that is locked to half the display rate: a
	// single short step to half the pixels asks, and the pause doubles after every "no"
	const gov = new Governor();
	const state = fresh( { explicit: true } );
	const actions = simulate( gov, state, () => 200, 900, { hz: 30 } );
	assert.ok( actions.length >= 2 && actions.length <= 10, `actions in 15 minutes: ${actions.length}` );
	for ( let i = 0; i < actions.length; i += 2 ) {

		assert.deepEqual( [ actions[ i ].type, actions[ i ].scale ], [ 'scale', 0.7 ] ); // never the long descent to the floor
		if ( actions[ i + 1 ] ) assert.deepEqual( [ actions[ i + 1 ].type, actions[ i + 1 ].scale ], [ 'scale', 1 ] );

	}

	const asked = actions.filter( ( a ) => a.scale < 1 ).map( ( a ) => a.t );
	for ( let i = 2; i < asked.length; i ++ ) assert.ok( asked[ i ] - asked[ i - 1 ] > 1.8 * ( asked[ i - 1 ] - asked[ i - 2 ] ) - 15000, 'the pauses double' );
	assert.ok( secondsBelowFull( actions, state.now ) < 25, `seconds below full resolution: ${secondsBelowFull( actions, state.now )}` );
	assert.equal( state.tier, 'high' );
	// the same in 'auto': the tier stays where it is
	const g2 = new Governor();
	const s2 = fresh();
	assert.deepEqual( tiers( simulate( g2, s2, () => 200, 900, { hz: 30 } ) ), [] );
	// a resize in the pause (another pixel cap) does not start the questions again
	const g3 = new Governor();
	const s3 = fresh( { explicit: true } );
	const first = simulate( g3, s3, () => 200, 20, { hz: 30 } );
	assert.equal( first.length, 2 );
	s3.maxScale = 0.8; s3.scale = 0.8;
	assert.deepEqual( simulate( g3, s3, () => 200, 40, { hz: 30 } ), [] );
	// a loop that turns slow in that pause is not kept from lowering the resolution, and 'auto' tries
	// that before it touches the tier
	const g4 = new Governor();
	const s4 = fresh();
	const heavy = ( s, now ) => ( now < 20000 ? 200 : 14 / ( s.scale * s.scale ) );
	const a4 = simulate( g4, s4, heavy, 80, { hz: 30 } );
	const then = a4.filter( ( a ) => a.t > 20000 );
	assert.deepEqual( [ then[ 0 ].type, then[ 0 ].reason ], [ 'scale', 'slow' ] );
	assert.ok( then[ 0 ].t < 26000 );
	assert.deepEqual( tiers( a4 ), [] );
	assert.ok( heavy( s4, s4.now ) >= 22.5 );

} );

test( 'governor: a GPU locked to half the display rate is found', () => {

	// 30 fps at full scale, 60 at 0.7 and below, in perfectly even frames either way
	const device = ( s ) => ( s.scale > 0.7001 ? 30 : 60 );
	// with GPU timings: the GPU is busy for the whole frame, so the pace is not the display's
	const gov = new Governor();
	const state = fresh( { explicit: true } );
	const actions = simulate( gov, state, device, 60, { hz: 0, gpuMsOf: ( s, fps ) => ( s.scale > 0.7001 ? 30 : 15 ) } );
	assert.equal( device( state ), 60 );
	assert.ok( actions[ 0 ].t < 8000, 'found within seconds' );
	assert.ok( Math.abs( gov.displayFps - 60 ) < 0.5 );
	// without: the question is asked after a few steady windows
	const g2 = new Governor();
	const s2 = fresh( { explicit: true } );
	simulate( g2, s2, device, 60, { hz: 0 } );
	assert.equal( device( s2 ), 60 );
	// and it stays there: a climb back is tried with doubling pauses (two minutes in the end) and taken back at once
	const later = simulate( g2, s2, device, 3600, { hz: 0 } );
	assert.equal( device( s2 ), 60 );
	assert.ok( later.length <= 64, `changes in an hour: ${later.length}` );
	assert.ok( later.every( ( a, i ) => ( i % 2 === 0 ? a.reason === 'headroom' : a.reason === 'climb failed' ) ) );
	let at30 = 0;
	for ( let i = 0; i + 1 < later.length; i += 2 ) at30 += later[ i + 1 ].t - later[ i ].t;
	assert.ok( at30 / 3600e3 < 0.03, `share of the hour spent in failed climbs: ${at30 / 3600e3}` );

} );

test( 'governor: a start that is slow for another reason does not switch scaling off for good', () => {

	// 25 fps whatever the scale for the first 30 s (the layout), then a GPU locked to 30 fps that
	// would reach 60 at scale 0.7
	for ( const timings of [ false, true ] ) {

		const gov = new Governor();
		const state = fresh( { explicit: true } );
		const device = ( s, now ) => ( now < 30000 ? 25 : ( s.scale > 0.7001 ? 30 : 60 ) );
		const gpuMsOf = timings ? ( s, fps ) => 900 / fps : null;
		const actions = simulate( gov, state, device, 240, { hz: 0, gpuMsOf } );
		assert.equal( device( state, state.now ), 60, `timings ${timings}` );
		assert.ok( state.scale <= 0.7001 && state.scale >= 0.5 );
		// what was tried during the slow start was taken back
		const early = actions.filter( ( a ) => a.t < 30000 );
		if ( early.length ) assert.equal( early[ early.length - 1 ].scale, 1 );
		assert.deepEqual( tiers( actions ), [] );

	}

	// the same with a load that is first bound by something else and later by the pixels
	const gov = new Governor();
	const state = fresh( { explicit: true } );
	const device = ( s, now ) => ( now < 150000 ? 20 : Math.min( 60, 20 / ( s.scale * s.scale ) ) );
	const actions = simulate( gov, state, device, 600 );
	const early = actions.filter( ( a ) => a.t < 150000 );
	assert.ok( early.length >= 3 );
	assert.ok( Math.min( ...early.map( ( a ) => a.scale ) ) >= 0.7 - 1e-9, 'a descent without gain ends at half the pixels' );
	assert.equal( early[ early.length - 1 ].scale, 1 );
	assert.equal( gov.scaleUseless, false ); // the pause is over
	assert.ok( device( state, state.now ) >= 45, `fps at the end: ${device( state, state.now )}` );

} );

test( 'governor: slow for another reason than the pixels, the tier steps down on trial', () => {

	// 20 fps in high / ultra at any resolution (too many passes, a CPU-bound driver), 55 fps below
	const device = ( s ) => ( s.tier === 'high' || s.tier === 'ultra' ? 20 : 55 );
	const gov = new Governor();
	const state = fresh();
	const actions = simulate( gov, state, device, 3600 );
	assert.deepEqual( tiers( actions ), [ 'medium' ] );
	assert.equal( state.tier, 'medium' );
	assert.equal( state.scale, 1 );
	assert.equal( gov.tierUseless, false );
	// resolution was tried first, briefly: two steps to half the pixels, then back
	const firstTier = actions.findIndex( ( a ) => a.type === 'tier' );
	assert.deepEqual( actions.slice( 0, firstTier ).map( ( a ) => a.scale ), [ 0.85, 0.7, 1 ] );
	assert.ok( actions[ firstTier ].t < 30000 );
	assert.ok( actions.length <= 12, `changes in an hour: ${actions.length}` );

	// a lower tier that is no faster is taken back, once and for all
	const g2 = new Governor();
	const s2 = fresh();
	const a2 = simulate( g2, s2, () => 20, 3600 );
	assert.deepEqual( tiers( a2 ), [ 'medium', 'high' ] );
	assert.equal( s2.tier, 'high' );
	assert.equal( s2.scale, 1 );
	assert.equal( g2.tierUseless, true );
	// resolution is tried again now and then, with doubling pauses: not a thrash
	assert.ok( scales( a2 ).length <= 3 * 8, `scale changes in an hour: ${scales( a2 ).length}` );
	assert.ok( secondsBelowFull( a2, s2.now ) < 90, `seconds below full resolution: ${secondsBelowFull( a2, s2.now )}` );

	// with dynamic resolution off the trial is the only tool
	const g3 = new Governor();
	const s3 = fresh( { dynamicResolution: false } );
	const a3 = simulate( g3, s3, device, 600 );
	assert.deepEqual( a3.map( ( a ) => [ a.type, a.tier ] ), [ [ 'tier', 'medium' ] ] );

	// an explicit tier: resolution is tried and taken back, the tier stays
	const g4 = new Governor();
	const s4 = fresh( { explicit: true } );
	const a4 = simulate( g4, s4, () => 20, 3600 );
	assert.deepEqual( tiers( a4 ), [] );
	assert.equal( s4.scale, 1 );
	assert.equal( a4[ a4.length - 1 ].scale, 1 );

	// reset(): a new quality setting or a new graph starts the measurements again
	g2.reset( false );
	assert.deepEqual( [ g2.tierUseless, g2.scaleUseless, g2.downgraded ], [ false, false, true ] );
	g2.reset();
	assert.equal( g2.downgraded, false );
	assert.ok( g2.displayFps > 0 ); // the display is still the same

} );

test( 'governor: stalls and the settle time are ignored', () => {

	const gov = new Governor();
	gov.configure( fresh() );
	let now = 0, acted = null;
	// a hidden tab: one 30 s frame between normal ones
	for ( let i = 0; i < 2000 && ! acted; i ++ ) {

		const ms = i % 100 === 50 ? 30000 : 16.7;
		now += ms;
		acted = gov.sample( ms, now );

	}

	assert.equal( acted, null );
	// slow frames inside the settle time do not count
	const g2 = new Governor( { settleMs: 5000 } );
	g2.configure( fresh() );
	now = 0; acted = null;
	for ( let i = 0; i < 80 && ! acted; i ++ ) {

		now += 50;
		acted = g2.sample( 50, now ); // 20 fps for 4 s
	}

	assert.equal( acted, null );
	// the short catch-up frames after hitches are not the display rate
	const g3 = new Governor();
	g3.configure( fresh( { explicit: true } ) );
	now = 0; acted = null;
	for ( let i = 0; i < 3000 && ! acted; i ++ ) {

		const ms = i % 30 === 10 ? 60 : ( i % 30 === 11 ? 6.7 : 33.33 ); // a 30 fps cap with a hitch every second
		now += ms;
		acted = g3.sample( ms, now, 4 );

	}

	assert.equal( acted, null );
	assert.ok( Math.abs( g3.displayFps - 30 ) < 1 );
	// bad input
	assert.equal( gov.sample( 0, now ), null );
	assert.equal( gov.sample( NaN, now ), null );

} );

test( 'governor: the scale climbs back when the load goes away', () => {

	const gov = new Governor();
	const state = fresh();
	let heavy = true;
	const device = ( s ) => Math.min( 60, ( heavy ? 33 : 200 ) / ( s.scale * s.scale ) );
	simulate( gov, state, device, 60 );
	assert.ok( state.scale < 0.9 );
	heavy = false;
	const t0 = state.now;
	const actions = simulate( gov, state, device, 300 );
	assert.equal( state.scale, 1 );
	for ( let i = 1; i < actions.length; i ++ ) assert.ok( actions[ i ].scale > actions[ i - 1 ].scale );
	// quickly: the pace is pinned at the display rate, so the steps are large and soon
	assert.ok( actions[ actions.length - 1 ].t - t0 < 25000, `back at full scale after ${( actions[ actions.length - 1 ].t - t0 ) / 1000} s` );
	// from the floor as well (the reviewer's case: 0.5 after a layout, 60-70 s with the slow climb)
	const g2 = new Governor();
	const s2 = fresh( { scale: 0.5 } );
	const a2 = simulate( g2, s2, () => 60, 60 );
	assert.equal( s2.scale, 1 );
	assert.ok( a2[ a2.length - 1 ].t < 25000, `0.5 → 1 in ${a2[ a2.length - 1 ].t / 1000} s` );

} );

test( 'governor: very low frame rates still reach a decision', () => {

	// 2 fps (a software rasterizer): windows close on time instead of on the frame count
	const gov = new Governor();
	const state = fresh( { tier: 'low' } );
	const actions = simulate( gov, state, ( s ) => 2 / ( s.scale * s.scale ), 120 );
	assert.ok( actions.length >= 1 );
	assert.ok( state.scale < 1 );
	assert.equal( state.tier, 'low' ); // nothing below low

} );

test( 'parseGfxParams: URL switches', () => {

	const parse = ( q ) => parseGfxParams( new URLSearchParams( q ) );
	assert.deepEqual( parse( '' ), { overrides: {} } );
	assert.deepEqual( parse( 'ds=lesmis&layout=sphere' ), { overrides: {} } );
	assert.equal( parse( 'quality=ultra' ).quality, 'ultra' );
	assert.equal( parse( 'quality=HIGH' ).quality, 'high' );
	assert.equal( parse( 'quality=bogus' ).quality, 'auto' );
	assert.equal( parse( 'governor=0' ).governor, false );
	assert.equal( parse( 'governor=1' ).governor, true );
	assert.equal( 'governor' in parse( 'quality=low' ), false );
	assert.deepEqual( parse( 'ao=0&dof=1&shadows=0&traa=0' ).overrides, { ao: false, dof: true, shadows: false, traa: false } );
	assert.deepEqual( parse( 'ca=1&dynres=0' ).overrides, { chromaticAberration: true, dynamicResolution: false } );
	assert.deepEqual( parse( 'scale=0.5' ).overrides, { scale: 0.5 } );
	assert.deepEqual( parse( 'scale=abc' ).overrides, {} );
	assert.deepEqual( parse( 'debug=coc' ).overrides, { debug: 'coc' } );
	assert.deepEqual( parse( 'debug=none' ).overrides, {} );
	// the result feeds resolveQuality directly
	const p = parse( 'quality=high&traa=0&ao=0&scale=0.75' );
	const c = resolveQuality( p.quality, DESKTOP, p.overrides );
	assert.equal( c.aa, 'smaa' );
	assert.equal( c.ao, false );
	assert.equal( c.renderScale, 0.75 );
	assert.equal( c.dynamicResolution, false );

} );

test( 'resolveQuality: post=0 leaves a plain, anti-aliased image', () => {

	for ( const tier of TIER_ORDER ) {

		const full = resolveQuality( tier, DESKTOP );
		const c = resolveQuality( tier, DESKTOP, { post: false } );
		assert.equal( full.bloom, true );
		assert.deepEqual( [ c.ao, c.dof, c.bloom, c.streak, c.vignette, c.sharpen, c.grain, c.chromaticAberration ], [ false, false, false, false, false, false, 0, 0 ] );
		assert.equal( c.aa, full.aa );
		assert.equal( c.samples, full.samples );
		assert.equal( c.shadows, full.shadows ); // lighting is not a post effect
		assert.notEqual( structureKey( c ), structureKey( full ) );

	}

	assert.deepEqual( parseGfxParams( new URLSearchParams( 'post=0' ) ).overrides, { post: false } );

} );
