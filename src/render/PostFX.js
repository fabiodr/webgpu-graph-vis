// Compatibility adapter. Post-processing is built by the render graph (app.gfx, see
// src/render/gfx/RenderGraph.js): this module only keeps the previous PostFX interface alive for
// pages that still create one, and forwards every property to app.gfx. New code talks to app.gfx
// directly:
//
//   post.bloomStrength = v   →  app.gfx.set( 'bloom', v )
//   post.quality = q         →  app.gfx.setQuality( q )      ('auto' | 'low' | 'medium' | 'high' | 'ultra')
//   post.enabled = false     →  app.gfx.set( 'post', false ) (plain tone mapped output)
//   post.toneMapping = t     →  app.gfx.set( 'toneMapping', t )
//   post.exposure = e        →  app.gfx.set( 'exposure', e )
//   post.threshold / radius / vignette  →  bloomThreshold / bloomRadius / vignette
//
// Public API (unchanged):
//   const post = app.use( new PostFX( { bloom: 0.8, quality, enabled, toneMapping, exposure } ) );
//   A `quality` that is not given leaves the tier alone (URL parameter or the 'auto' choice).
import { TONE_MAPPINGS } from './gfx/RenderGraph.js';
import { autoTier } from '../app/Quality.js';

export { TONE_MAPPINGS };

/** The tier 'auto' would pick from pointer and screen size alone (kept for old callers). */
export function autoQuality() {

	if ( typeof window === 'undefined' ) return 'high';
	const coarsePointer = typeof matchMedia !== 'undefined' && matchMedia( '(pointer:coarse)' ).matches;
	const smallScreen = Math.min( window.innerWidth, window.innerHeight ) < 600;
	return autoTier( { coarsePointer, smallScreen, softwareAdapter: false } );

}

export class PostFX {

	constructor( { bloom = 0.8, quality, enabled = true, toneMapping, exposure } = {} ) {

		this.order = 90;
		this.app = null;
		this._initial = { bloom, quality, enabled, toneMapping, exposure };

	}

	init( app ) {

		this.app = app;
		const { bloom, quality, enabled, toneMapping, exposure } = this._initial;
		const gfx = app.gfx;
		gfx.set( 'bloom', bloom );
		if ( quality !== undefined && quality !== gfx.setting ) gfx.setQuality( quality );
		if ( enabled === false ) gfx.set( 'post', false );
		if ( toneMapping !== undefined && toneMapping !== gfx.get( 'toneMapping' ) ) gfx.set( 'toneMapping', toneMapping );
		if ( exposure !== undefined ) gfx.set( 'exposure', exposure );

	}

	get gfx() {

		return this.app ? this.app.gfx : null;

	}

	get bloomStrength() {

		return this.gfx ? this.gfx.get( 'bloom' ) : this._initial.bloom;

	}

	set bloomStrength( v ) {

		this._initial.bloom = v;
		this.gfx?.set( 'bloom', v );

	}

	get threshold() {

		return this.gfx ? this.gfx.uniforms.bloomThreshold.value : 0.24;

	}

	set threshold( v ) {

		this.gfx?.set( 'bloomThreshold', v );

	}

	get radius() {

		return this.gfx ? this.gfx.get( 'bloomRadius' ) : 0.4;

	}

	set radius( v ) {

		this.gfx?.set( 'bloomRadius', v );

	}

	get vignette() {

		return this.gfx ? this.gfx.uniforms.vignette.value : 0.22;

	}

	set vignette( v ) {

		this.gfx?.set( 'vignette', + v > 0 ? + v : false );

	}

	get enabled() {

		return this.gfx ? this.gfx.config.bloom : this._initial.enabled;

	}

	set enabled( v ) {

		this._initial.enabled = !! v;
		this.gfx?.set( 'post', v ? null : false );

	}

	get quality() {

		return this.gfx ? this.gfx.setting : this._initial.quality;

	}

	set quality( q ) {

		this._initial.quality = q;
		if ( this.gfx && q !== this.gfx.setting ) this.gfx.setQuality( q );

	}

	get toneMapping() {

		return this.gfx ? this.gfx.get( 'toneMapping' ) : this._initial.toneMapping;

	}

	set toneMapping( t ) {

		this._initial.toneMapping = t;
		if ( this.gfx && t !== this.gfx.get( 'toneMapping' ) ) this.gfx.set( 'toneMapping', t );

	}

	get exposure() {

		return this.gfx ? this.gfx.uniforms.exposure.value : 1;

	}

	set exposure( v ) {

		this.gfx?.set( 'exposure', v );

	}

	dispose() {}

}
