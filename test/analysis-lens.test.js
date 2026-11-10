// Metric lens against a fake GraphBuffers (what it writes, what it leaves alone), with the
// analytics handler on this thread behind a fake clock, and the sequential colour ramps.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as GA from '../src/graph/analytics.js';
import { createHandler } from '../src/analysis/worker.js';
import { AnalysisClient } from '../src/analysis/client.js';
import { Lens, SIZE_RANGE } from '../src/analysis/Lens.js';
import { parseGraphJson } from '../src/data/format.js';
import { prepareGraph, colorize } from '../src/graph/prepare.js';
import {
	RAMPS, rampColor, rampColors, rampCss, rampList, categoryColor, emissiveGain, linearToOklab, oklchToLinear, linearToSrgb, srgbToLinear,
	PALETTES, paletteColor, hclToRgb,
} from '../src/util/color.js';

// ── helpers ──────────────────────────────────────────────────────────────────────────────────

const load = ( name, options ) => prepareGraph( parseGraphJson( JSON.parse( readFileSync( `public/data/${name}.json`, 'utf8' ) ), name ), options );

/** Worker stand-in on this thread; the fake clock makes slices and previews deterministic. */
class FakeWorker {

	constructor() {

		let time = 0;
		this.posted = [];
		this.onmessage = null;
		this.handle = createHandler( {
			post: ( data, transfer = [] ) => {

				const copy = structuredClone( data, { transfer } ); // transferred, as by a real worker
				queueMicrotask( () => this.onmessage && this.onmessage( { data: copy } ) );

			},
			now: () => ( time += 5 ), wait: () => new Promise( ( r ) => setImmediate( r ) ), sliceMs: 20, progressMs: 0,
		} );

	}

	postMessage( msg ) {

		this.posted.push( msg );
		queueMicrotask( () => this.handle( msg ) );

	}

	terminate() {

		this.onmessage = null;

	}

}

/** The parts of GraphBuffers the lens uses, recording every upload. */
function fakeBuffers( graph ) {

	const N = graph.N;
	const fx = new Float32Array( 4 * N );
	// x / y carry made-up tool data that the lens must not touch
	for ( let i = 0; i < N; i ++ ) {

		fx[ 4 * i ] = i % 7;
		fx[ 4 * i + 1 ] = ( i % 5 ) / 4;

	}

	const b = {
		N, graph,
		nodeFx: { value: { array: fx, needsUpdate: false } },
		colorCalls: [], colorOptions: [], fxCalls: 0,
		target: new Float32Array( 4 * N ),
		setColors( colors, radii, options = {} ) {

			b.colorOptions.push( options );
			assert.equal( colors.length, 4 * N );
			assert.equal( radii.length, N );
			for ( let i = 0; i < N; i ++ ) {

				b.target[ 4 * i ] = colors[ 4 * i ]; b.target[ 4 * i + 1 ] = colors[ 4 * i + 1 ]; b.target[ 4 * i + 2 ] = colors[ 4 * i + 2 ]; b.target[ 4 * i + 3 ] = radii[ i ];

			}

			b.colorCalls.push( Float32Array.from( b.target ) );

		},
		setFx( fx4 ) {

			b.nodeFx.value.array.set( fx4 );
			b.nodeFx.value.needsUpdate = true;
			b.fxCalls ++;

		},
	};
	return b;

}

function setup( graph, options = {} ) {

	const workers = [];
	const client = new AnalysisClient( { createWorker: () => {

		const w = new FakeWorker();
		workers.push( w );
		return w;

	} } );
	const app = { events: new EventTarget() };
	const events = [];
	app.events.addEventListener( 'lens', ( e ) => events.push( e.detail ) );
	const lens = new Lens( { client, preview: 40, ...options } );
	lens.init( app );
	const buffers = fakeBuffers( graph );
	lens.setGraph( buffers, graph );
	return { lens, client, buffers, app, events, workers };

}

/** Ramp colours as the lens writes them: at its default intensity of 0.4 unless told otherwise. */
const lit = ( t, ramp, reverse = false, intensity = 0.4 ) => rampColors( t, ramp, undefined, reverse, intensity );
/** Radius factors as the lens writes them by default: the contract's range, gamma 1.5. */
const sized = ( t, lo = 0.7, hi = 1.6, gamma = 1.5 ) => Float32Array.from( t, ( v ) => lo + ( hi - lo ) * Math.pow( v, gamma ) );
const luminance = ( c ) => 0.2126 * c[ 0 ] + 0.7152 * c[ 1 ] + 0.0722 * c[ 2 ];
const idle = async ( lens ) => {

	for ( let k = 0; k < 200000 && lens.state.busy; k ++ ) await new Promise( ( r ) => setImmediate( r ) );
	assert.equal( lens.state.busy, false );

};

const column = ( array, stride, offset ) => Float32Array.from( { length: array.length / stride }, ( _, i ) => array[ stride * i + offset ] );
const rgb = ( array ) => Float32Array.from( { length: ( array.length / 4 ) * 3 }, ( _, k ) => array[ 4 * Math.floor( k / 3 ) + ( k % 3 ) ] );
const near = ( a, b, eps = 1e-6 ) => assert.ok( Math.abs( a - b ) <= eps, `${a} vs ${b}` );
const close = ( a, b, eps = 1e-6, message = '' ) => {

	assert.equal( a.length, b.length, message );
	for ( let i = 0; i < a.length; i ++ ) assert.ok( Math.abs( a[ i ] - b[ i ] ) <= eps, `${message} [${i}] ${a[ i ]} vs ${b[ i ]}` );

};

// ── ramps ────────────────────────────────────────────────────────────────────────────────────

test( 'ramps: at least four, OKLab lightness strictly monotone, bright enough on black', () => {

	const names = Object.keys( RAMPS );
	assert.ok( names.length >= 4 );
	for ( const name of [ 'viridis', 'magma', 'ice', 'ember' ] ) assert.ok( names.includes( name ), name );
	for ( const name of names ) {

		let previous = - 1;
		for ( let k = 0; k <= 400; k ++ ) {

			const c = rampColor( k / 400, name );
			assert.ok( c.every( ( v ) => v >= 0 && v <= 1 ), `${name}: in gamut at ${k / 400}` );
			const L = linearToOklab( c[ 0 ], c[ 1 ], c[ 2 ] )[ 0 ];
			assert.ok( L > previous, `${name}: lightness rises at t = ${k / 400} (${previous} -> ${L})` );
			previous = L;

		}

		// dark end: at least 3:1 against black (WCAG relative luminance), light end: nearly white-hot
		const lo = rampColor( 0, name ), hi = rampColor( 1, name );
		const Y = ( c ) => 0.2126 * c[ 0 ] + 0.7152 * c[ 1 ] + 0.0722 * c[ 2 ];
		assert.ok( ( Y( lo ) + 0.05 ) / 0.05 >= 3, `${name}: contrast of the dark end ${( ( Y( lo ) + 0.05 ) / 0.05 ).toFixed( 2 )}` );
		assert.ok( Y( hi ) > 0.75, `${name}: bright end ${Y( hi )}` );
		assert.ok( linearToOklab( ...hi )[ 0 ] - linearToOklab( ...lo )[ 0 ] > 0.4, `${name}: lightness range` );

	}

} );

test( 'ramps: endpoints are the first and last stop, t is clamped, names fall back', () => {

	for ( const [ name, ramp ] of Object.entries( RAMPS ) ) {

		for ( const [ t, stop ] of [ [ 0, ramp.stops[ 0 ] ], [ 1, ramp.stops.at( - 1 ) ] ] ) {

			const [ L, a, b ] = linearToOklab( ...rampColor( t, name ) );
			assert.ok( Math.abs( L - stop[ 0 ] ) < 2e-3, `${name}: lightness of the stop at t = ${t}` );
			// hue as asked; chroma as asked or less (reduced only to fit sRGB)
			const C = Math.hypot( a, b );
			assert.ok( C <= stop[ 1 ] + 2e-3 && C > 0.4 * stop[ 1 ], `${name}: chroma ${C} of ${stop[ 1 ]}` );
			if ( C > 0.03 ) {

				const hue = ( ( Math.atan2( b, a ) * 180 ) / Math.PI + 360 ) % 360;
				const d = Math.abs( ( ( hue - stop[ 2 ] ) % 360 + 540 ) % 360 - 180 );
				assert.ok( d < 2, `${name}: hue ${hue} vs ${stop[ 2 ]}` );

			}

		}

		assert.deepEqual( rampColor( - 3, name ), rampColor( 0, name ) );
		assert.deepEqual( rampColor( 7, name ), rampColor( 1, name ) );
		assert.deepEqual( rampColor( NaN, name ), rampColor( 0, name ) );

	}

	assert.deepEqual( rampColor( 0.3, 'no-such-ramp' ), rampColor( 0.3, 'viridis' ) );
	assert.deepEqual( rampColor( 0.3 ), rampColor( 0.3, 'viridis' ) );
	const out = [ 9, 9, 9 ];
	assert.equal( rampColor( 0.5, 'ice', out ), out, 'writes into the array it is given' );

} );

test( 'ramps: rampColors, reverse, CSS gradient, list; existing colour exports unchanged', () => {

	const t = Float32Array.from( [ 0, 0.25, 0.5, 1, - 1, 2, NaN ] );
	const colors = rampColors( t, 'magma' );
	assert.equal( colors.length, 4 * t.length );
	t.forEach( ( v, i ) => {

		const c = rampColor( v, 'magma' );
		close( colors.subarray( 4 * i, 4 * i + 3 ), c, 1e-7 );
		assert.equal( colors[ 4 * i + 3 ], 1 );

	} );
	const reversed = rampColors( t, 'magma', undefined, true );
	close( reversed.subarray( 0, 3 ), rampColor( 1, 'magma' ), 1e-7 );
	close( reversed.subarray( 12, 15 ), rampColor( 0, 'magma' ), 1e-7 );
	const into = new Float32Array( 4 * t.length );
	assert.equal( rampColors( t, 'ice', into ), into );

	const css = rampCss( 'viridis', { steps: 5 } );
	assert.match( css, /^linear-gradient\(90deg, #[0-9a-f]{6} 0\.0%, .* #[0-9a-f]{6} 100\.0%\)$/ );
	assert.equal( css.split( '#' ).length - 1, 5 );
	const hex = ( c ) => '#' + c.map( ( v ) => Math.round( 255 * linearToSrgb( v ) ).toString( 16 ).padStart( 2, '0' ) ).join( '' );
	assert.ok( css.includes( hex( rampColor( 0, 'viridis' ) ) + ' 0.0%' ) );
	assert.ok( rampCss( 'viridis', { steps: 5, reverse: true } ).includes( hex( rampColor( 1, 'viridis' ) ) + ' 0.0%' ) );

	const list = rampList();
	assert.deepEqual( list.map( ( r ) => r.id ), Object.keys( RAMPS ) );
	assert.ok( list.every( ( r ) => r.label && r.css.startsWith( 'linear-gradient(' ) ) );

	// categorical colours: eight distinct hues, then grey
	const seen = new Set();
	for ( let k = 0; k < 8; k ++ ) seen.add( categoryColor( k ).map( ( v ) => v.toFixed( 3 ) ).join() );
	assert.equal( seen.size, 8 );
	assert.deepEqual( categoryColor( 8 ), categoryColor( 500 ) );
	const grey = categoryColor( 99 );
	assert.ok( Math.abs( grey[ 0 ] - grey[ 1 ] ) < 1e-3 && Math.abs( grey[ 1 ] - grey[ 2 ] ) < 1e-3 );

	// what was there before still is
	assert.deepEqual( Object.keys( PALETTES ), [ 'paper', 'rainbow', 'aurora', 'ember' ] );
	assert.deepEqual( paletteColor( 'paper', 0.5 ), hclToRgb( 105 / 360, 0.5, 0.5 ).map( srgbToLinear ) );
	close( oklchToLinear( 1, 0, 0 ), [ 1, 1, 1 ], 1e-3 );
	close( linearToOklab( 1, 1, 1 ), [ 1, 0, 0 ], 1e-3 );
	const round = linearToOklab( ...oklchToLinear( 0.7, 0.1, 200 ) );
	close( [ round[ 0 ], Math.hypot( round[ 1 ], round[ 2 ] ) ], [ 0.7, 0.1 ], 1e-4 );

} );

test( 'ramps: intensity holds the bright end back, keeps hue, order and a visible dark end', () => {

	const ts = Float32Array.from( { length: 401 }, ( _, k ) => k / 400 );
	for ( const name of Object.keys( RAMPS ) ) {

		const full = rampColors( ts, name );
		close( rampColors( ts, name, undefined, false, 1 ), full, 0, 'intensity 1 is the ramp as defined' );
		for ( const intensity of [ 0.6, 0.4, 0.2 ] ) {

			const held = rampColors( ts, name, undefined, false, intensity );
			let previous = - 1;
			for ( let k = 0; k < ts.length; k ++ ) {

				const c = held.subarray( 4 * k, 4 * k + 3 ), f = full.subarray( 4 * k, 4 * k + 3 );
				const L = linearToOklab( c[ 0 ], c[ 1 ], c[ 2 ] )[ 0 ];
				assert.ok( L > previous, `${name} at ${intensity}: lightness still rises at t = ${ts[ k ]}` );
				previous = L;
				// one gain for the three channels (the hue stays), the one emissiveGain() names
				const gain = luminance( c ) / luminance( f );
				close( c, f.map( ( v ) => v * gain ), 1e-3 );
				near( gain, emissiveGain( f[ 0 ], f[ 1 ], f[ 2 ], intensity ), 5e-3 );
				assert.equal( held[ 4 * k + 3 ], 1 );

			}

		}

		// the lens default, 0.4: no brighter than the cluster palettes (aurora 0.51, ember 0.59),
		// and the dark end keeps at least four fifths of its light
		const held = rampColors( ts, name, undefined, false, 0.4 );
		const top = luminance( held.subarray( 1600, 1603 ) ), bottom = luminance( held.subarray( 0, 3 ) );
		assert.ok( top > 0.3 && top < 0.4, `${name}: top luminance ${top}` );
		assert.ok( bottom > 0.8 * luminance( full.subarray( 0, 3 ) ) && ( bottom + 0.05 ) / 0.05 > 2.8, `${name}: bottom luminance ${bottom}` );
		close( rampColors( ts, name, undefined, true, 0.4 ).subarray( 0, 3 ), held.subarray( 1600, 1603 ), 1e-7, 'reverse' );

	}

	near( emissiveGain( 1, 1, 1, 0.4 ), 0.4, 1e-6 );
	assert.equal( emissiveGain( 1, 1, 1, 1 ), 1 );
	assert.equal( emissiveGain( 0, 0, 0, 0.3 ), 1 );
	assert.equal( emissiveGain( 1, 1, 1 ), 1 );

} );

// ── lens ─────────────────────────────────────────────────────────────────────────────────────

test( 'Lens: size by PageRank writes radii (log scale), keeps colours, x / y of nodeFx untouched', async () => {

	const g = load( 'imsv/lesmis' );
	const baseColors = Float32Array.from( g.colors ), baseRadii = Float32Array.from( g.radii );
	const { lens, buffers, events } = setup( g );
	const fxBefore = Float32Array.from( buffers.nodeFx.value.array );
	assert.equal( lens.order, 25 );
	assert.deepEqual( lens.state.size, null );

	assert.equal( await lens.setSize( 'pagerank' ), true );
	const t = GA.normalize( GA.pageRank( g ), { mode: 'log' } );
	const got = buffers.colorCalls.at( - 1 );
	close( column( got, 4, 3 ), sized( t ), 1e-6, 'radii: the range of the node size contract, 0.7 … 1.6, gamma 1.5' );
	near( Math.min( ...column( got, 4, 3 ) ), 0.7 );
	near( Math.max( ...column( got, 4, 3 ) ), 1.6 );
	assert.deepEqual( SIZE_RANGE, [ 0.7, 1.6 ] );
	assert.deepEqual( lens.state.sizeRange, [ 0.7, 1.6 ] );
	assert.deepEqual( lens.legend( 'size' ).range, [ 0.7, 1.6 ] );
	assert.equal( lens.legend( 'size' ).gamma, 1.5 );
	close( rgb( got ), rgb( baseColors ), 0, 'colours are still the cluster palette' );

	const fx = buffers.nodeFx.value.array;
	close( column( fx, 4, 0 ), column( fxBefore, 4, 0 ), 0, 'nodeFx.x' );
	close( column( fx, 4, 1 ), column( fxBefore, 4, 1 ), 0, 'nodeFx.y' );
	close( column( fx, 4, 2 ), new Float32Array( g.N ), 0, 'no glow' );
	close( column( fx, 4, 3 ), t, 1e-7, 'nodeFx.w = normalised value of the size metric' );
	assert.equal( buffers.nodeFx.value.needsUpdate, true );

	// mirror: the arrays labels / picking read follow, the originals are kept
	close( g.radii, column( got, 4, 3 ), 0 );
	close( lens.base.radii, baseRadii, 0 );
	assert.equal( lens.radii, g.radii );

	assert.equal( lens.state.size, 'pagerank' );
	assert.equal( lens.state.busy, false );
	assert.deepEqual( lens.state.metrics.pagerank, { ready: true, final: true, exact: true, pivots: 0, total: 0, progress: 1, error: null } );
	assert.deepEqual( events.map( ( e ) => e.reason ), [ 'graph', 'config', 'data' ] );
	assert.equal( events.at( - 1 ).size, 'pagerank' );
	assert.equal( events.at( - 1 ).color, null );

	// custom range and scale
	await lens.setSize( 'degree', { range: [ 0.9, 1.2 ], scale: 'linear', gamma: 1 } );
	close( column( buffers.colorCalls.at( - 1 ), 4, 3 ), GA.normalize( g.degree, { mode: 'linear' } ).map( ( v ) => 0.9 + 0.3 * v ), 1e-6 );
	// a range beyond the contract is clamped to it, end by end: other modules rely on 0.7 … 1.6
	await lens.setSize( 'degree', { range: [ 0.2, 3 ], scale: 'linear' } );
	close( column( buffers.colorCalls.at( - 1 ), 4, 3 ), sized( GA.normalize( g.degree, { mode: 'linear' } ) ), 1e-6 );
	assert.deepEqual( lens.state.sizeRange, [ 0.7, 1.6 ] );
	await lens.setSize( 'degree', { range: [ 1, 3 ] } );
	assert.deepEqual( lens.state.sizeRange, [ 1, 1.6 ] );
	await lens.setSize( 'degree', { range: [ 9, 0 ] } );
	assert.deepEqual( lens.state.sizeRange, [ 1.6, 0.7 ], 'an inverted range stays inverted' );
	assert.deepEqual( new Lens( { client: lens.client, sizeRange: [ 0.6, 2 ] } ).options.sizeRange, [ 0.7, 1.6 ], 'the option too' );
	// more contrast comes from the normalisation: gamma sets the leaders apart, inside the range
	await lens.setSize( 'degree', { gamma: 2 } );
	const squared = column( buffers.colorCalls.at( - 1 ), 4, 3 );
	close( squared, GA.normalize( g.degree, { mode: 'log' } ).map( ( v ) => 0.7 + 0.9 * v * v ), 1e-6 );
	near( Math.max( ...squared ), 1.6 );
	close( column( buffers.nodeFx.value.array, 4, 3 ), GA.normalize( g.degree, { mode: 'log' } ), 1e-7, 'w stays the plain normalised value' );
	assert.equal( new Lens( { client: lens.client } ).options.sizeGamma, 1.5 );

	// off: the degree radii of prepareGraph are back, w is cleared
	assert.equal( await lens.setSize( null ), true );
	close( column( buffers.colorCalls.at( - 1 ), 4, 3 ), baseRadii, 0 );
	close( g.radii, baseRadii, 0 );
	close( column( buffers.nodeFx.value.array, 4, 3 ), new Float32Array( g.N ), 0 );
	lens.dispose();

} );

test( 'Lens: colour by k-core through a ramp, restore to the cluster palette, palette change', async () => {

	const g = load( 'imsv/lesmis', { palette: 'rainbow' } );
	const { lens, buffers } = setup( g );
	assert.equal( lens.state.palette, 'rainbow' );
	await lens.setSize( 'pagerank' );
	const radii = column( buffers.colorCalls.at( - 1 ), 4, 3 );

	assert.equal( await lens.setColor( 'core', { ramp: 'magma' } ), true );
	const t = GA.normalize( GA.kCore( g ).core, { mode: 'linear' } );
	let got = buffers.colorCalls.at( - 1 );
	close( rgb( got ), rgb( lit( t, 'magma' ) ), 1e-7, 'ramp colours at the default intensity' );
	// which is the ramp held back by its luminance: colour / ( 1 + 1.5 Y )
	const full = rampColors( t, 'magma' );
	for ( let i = 0; i < g.N; i ++ ) {

		const c = full.subarray( 4 * i, 4 * i + 3 ), gain = 1 / ( 1 + 1.5 * luminance( c ) );
		close( got.subarray( 4 * i, 4 * i + 3 ), c.map( ( v ) => v * gain ), 2e-3 );

	}

	assert.equal( lens.state.intensity, 0.4 );
	assert.equal( lens.legend( 'color' ).intensity, 0.4 );
	close( column( got, 4, 3 ), radii, 0, 'the size lens is not disturbed' );
	close( column( buffers.nodeFx.value.array, 4, 3 ), t, 1e-7, 'nodeFx.w follows the colour metric when there is one' );
	assert.deepEqual( { ramp: lens.state.ramp, color: lens.state.color }, { ramp: 'magma', color: 'core' } );
	const legend = lens.legend( 'color' );
	assert.deepEqual( { metric: legend.metric, scale: legend.scale, min: legend.min, max: legend.max, ramp: legend.ramp }, { metric: 'core', scale: 'linear', min: 1, max: GA.kCore( g ).degeneracy, ramp: 'magma' } );
	assert.ok( legend.css.startsWith( 'linear-gradient(' ) );

	// same metric, other ramp / reversed: no new request, new colours
	const requests = lens.client._nextId;
	await lens.setColor( 'core', { ramp: 'ice', reverse: true } );
	assert.equal( lens.client._nextId, requests );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lit( t, 'ice', true ) ), 1e-7 );

	// intensity: a gain on the ramp (1 = the ramp as defined); like the ramp it stays until changed
	await lens.setColor( 'core', { intensity: 1 } );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( rampColors( t, 'ice' ) ), 1e-7 );
	await lens.setColor( 'core', { reverse: true } );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( rampColors( t, 'ice', undefined, true ) ), 1e-7 );
	assert.equal( lens.state.intensity, 1 );
	await lens.setColor( 'core', { intensity: 0 } );
	assert.equal( lens.state.intensity, 0.05, 'clamped to 0.05 … 1' );
	await lens.setColor( 'core', { intensity: 7, gamma: 2 } );
	assert.equal( lens.state.intensity, 1 );
	assert.equal( lens.legend( 'color' ).gamma, 2 );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( rampColors( t.map( ( v ) => v * v ), 'ice' ) ), 1e-6, 'gamma shapes the colour channel too' );
	await lens.setColor( 'core', { intensity: 0.4, reverse: true } );
	assert.equal( lens.client._nextId, requests );

	// a palette change while the colour lens is on does not show, but is what comes back later
	lens.setPalette( 'aurora' );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lit( t, 'ice', true ) ), 1e-7 );
	assert.equal( await lens.setColor( null ), true );
	got = buffers.colorCalls.at( - 1 );
	close( rgb( got ), rgb( colorize( g.leafRank, 'aurora' ) ), 0, 'cluster palette restored' );
	close( column( got, 4, 3 ), radii, 0 );
	assert.equal( g.palette, 'aurora' );
	// without a colour lens a palette change shows at once
	lens.setPalette( 'paper' );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( colorize( g.leafRank, 'paper' ) ), 0 );

	assert.throws( () => lens.setColor( 'core', { ramp: 'nope' } ), /unknown ramp/ );
	assert.throws( () => lens.setColor( 'nope' ), /unknown metric/ );
	assert.throws( () => lens.setSize( 'nope' ), /unknown metric/ );
	assert.equal( lens.state.color, null, 'a rejected call changes nothing' );
	lens.dispose();
	close( g.radii, lens.base.radii, 0, 'dispose puts the graph arrays back' );

} );

test( 'Lens: glow lights the top fraction by rank, ties alike, no plateaus; clear() resets every channel', async () => {

	const g = load( 'imsv/lesmis' );
	const { lens, buffers } = setup( g );
	const before = Float32Array.from( buffers.nodeFx.value.array );
	const z = () => column( buffers.nodeFx.value.array, 4, 2 );
	// the definition, from scratch: K = ceil( top N ) places; a node's weight falls with the number
	// of larger values; a group of equal values that overflows K fades and is off at 3 K
	const expected = ( values, top ) => {

		const N = values.length, K = Math.max( 1, Math.ceil( top * N - 1e-9 ) );
		return Float32Array.from( values, ( v ) => {

			const larger = values.filter( ( x ) => x > v ).length, equal = values.filter( ( x ) => x === v ).length;
			return larger < K ? ( 1 - larger / K ) * Math.min( 1, Math.max( 0, ( 3 * K - larger - equal ) / ( 2 * K ) ) ) : 0;

		} );

	};

	const lit = ( values ) => values.filter( ( v ) => v > 0 ).length;

	// betweenness: 43 of 77 values tie at zero, the leaders are distinct
	const bc = Array.from( GA.betweenness( g, { samples: Infinity } ) );
	assert.equal( await lens.setGlow( 'betweenness', { top: 0.1 } ), true );
	close( z(), expected( bc, 0.1 ), 1e-6 );
	assert.equal( lit( z() ), 8, 'top 10 % of 77 nodes = 8 nodes' );
	assert.equal( lens.legend( 'glow' ).count, 8 );
	assert.equal( z()[ bc.indexOf( Math.max( ...bc ) ) ], 1, 'the top node glows fully' );
	near( Math.min( ...z().filter( ( v ) => v > 0 ) ), 1 / 8 );
	assert.equal( lens.state.top, 0.1 );
	assert.equal( lens.legend( 'glow' ).top, 0.1 );
	close( column( buffers.nodeFx.value.array, 4, 3 ), GA.normalize( Float32Array.from( bc ), { mode: 'log' } ), 1e-7, 'w: the glow metric when it is the only one' );

	// default top 5 %: 4 nodes
	await lens.setGlow( 'betweenness' );
	close( z(), expected( bc, 0.05 ), 1e-6 );
	assert.equal( lit( z() ), 4 );

	// degree has ties among the leaders: equal values glow alike
	const degree = Array.from( g.degree );
	await lens.setGlow( 'degree', { top: 0.1 } );
	close( z(), expected( degree, 0.1 ), 1e-6 );
	for ( let i = 0; i < g.N; i ++ ) for ( let j = 0; j < i; j ++ ) if ( degree[ i ] === degree[ j ] ) assert.equal( z()[ i ], z()[ j ] );
	assert.ok( lit( z() ) >= 1 && lit( z() ) <= 3 * 8, `${lit( z() )} nodes glow` );
	assert.equal( z()[ degree.indexOf( Math.max( ...degree ) ) ], 1 );
	assert.ok( z().every( ( v ) => v >= 0 && v <= 1 ) );

	// k-core: the top "5 %" would be the whole 9-core, 12 nodes for 4 places. A plateau is not a top.
	const core = Array.from( GA.kCore( g ).core );
	assert.equal( core.filter( ( c ) => c === Math.max( ...core ) ).length, 12 );
	await lens.setGlow( 'core' );
	close( z(), new Float32Array( g.N ), 0 );
	assert.equal( lens.legend( 'glow' ).count, 0 );
	// with room for it (top 20 % = 16 places) the same core glows, all of it alike; the 8-core
	// behind it (8 nodes from place 13 on, 4 more than fit) is faint
	await lens.setGlow( 'core', { top: 0.2 } );
	close( z(), expected( core, 0.2 ), 1e-6 );
	assert.equal( lit( z() ), 20 );
	for ( let i = 0; i < g.N; i ++ ) near( z()[ i ], core[ i ] === 9 ? 1 : core[ i ] === 8 ? ( 1 - 12 / 16 ) * ( 48 - 20 ) / 32 : 0 );

	await lens.setSize( 'core' );
	await lens.setColor( 'clustering' );
	lens.clear();
	assert.deepEqual( [ lens.state.size, lens.state.color, lens.state.glow ], [ null, null, null ] );
	const got = buffers.colorCalls.at( - 1 );
	close( column( got, 4, 3 ), lens.base.radii, 0 );
	close( rgb( got ), rgb( lens.base.colors ), 0 );
	const fx = buffers.nodeFx.value.array;
	close( column( fx, 4, 2 ), new Float32Array( g.N ), 0 );
	close( column( fx, 4, 3 ), new Float32Array( g.N ), 0 );
	close( column( fx, 4, 0 ), column( before, 4, 0 ), 0 );
	close( column( fx, 4, 1 ), column( before, 4, 1 ), 0 );
	lens.dispose();

} );

test( 'Lens: a slow metric refines on screen (previews, then exact); superseding resolves false', async () => {

	const g = load( 'generated/hsbm-500' );
	const { lens, buffers, events, client } = setup( g );
	const first = buffers.colorCalls.length;
	const promise = lens.setColor( 'betweenness', { ramp: 'viridis' } );
	assert.equal( lens.state.busy, true );
	assert.equal( buffers.colorCalls.length, first + 1 );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lens.base.colors ), 0, 'until the first estimate arrives the picture stays' );
	assert.equal( await promise, true );

	const writes = buffers.colorCalls.slice( first + 1 );
	assert.ok( writes.length >= 4, `${writes.length} uploads while refining` );
	const exact = GA.normalize( GA.betweenness( g, { samples: Infinity } ), { mode: 'log' } );
	close( rgb( writes.at( - 1 ) ), rgb( lit( exact, 'viridis' ) ), 1e-7, 'the last upload is the exact metric' );
	// the estimates approach the exact picture
	const error = ( w ) => rgb( w ).reduce( ( s, v, k ) => s + Math.abs( v - rgb( writes.at( - 1 ) )[ k ] ), 0 );
	assert.ok( error( writes[ 0 ] ) > error( writes[ writes.length - 2 ] ), 'later previews are closer to the truth than the first' );
	assert.ok( error( writes[ 0 ] ) > 0 );
	const v = lens.values( 'betweenness' );
	assert.deepEqual( { final: v.final, exact: v.exact, pivots: v.pivots, total: v.total }, { final: true, exact: true, pivots: g.N, total: g.N } );
	assert.equal( lens.state.busy, false );
	const data = events.filter( ( e ) => e.reason === 'data' );
	assert.equal( data.length, writes.length );
	assert.equal( data[ 0 ].metrics.betweenness.final, false );
	assert.ok( data[ 0 ].metrics.betweenness.pivots < g.N );
	assert.ok( events.some( ( e ) => e.reason === 'progress' ) );

	// asked again: final values are reused, no request, resolves at once
	const id = client._nextId;
	assert.equal( await lens.setGlow( 'betweenness' ), true );
	assert.equal( client._nextId, id );

	// superseded on the same channel before anything arrived: false, and core is what shows
	const slow = lens.setColor( 'closeness' );
	const fast = lens.setColor( 'core', { ramp: 'ember' } );
	assert.deepEqual( await Promise.all( [ slow, fast ] ), [ false, true ] );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lit( GA.normalize( GA.kCore( g ).core, { mode: 'linear' } ), 'ember' ) ), 1e-7 );
	assert.equal( lens.state.busy, false, 'the closeness request was cancelled when nothing used it any more' );
	assert.equal( client.busy, false );
	lens.dispose();

} );

test( 'Lens: a metric on a channel refines until exact, whatever refineMs says', async () => {

	const g = load( 'generated/hsbm-500' );
	const { lens, buffers, client, events } = setup( g, { refineMs: 200 } );
	assert.equal( await lens.setColor( 'betweenness' ), true );
	let v = lens.values( 'betweenness' );
	assert.deepEqual( { final: v.final, exact: v.exact, pivots: v.pivots }, { final: true, exact: true, pivots: g.N } );
	const exact = GA.normalize( GA.betweenness( g, { samples: Infinity } ), { mode: 'log' } );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lit( exact, 'viridis' ) ), 1e-7 );
	assert.equal( client._nextId, 2, 'one request, without a time limit' );
	lens.clear();

	// a metric nobody shows stops at the time limit with the estimate it has ...
	v = await lens.request( 'closeness' );
	assert.deepEqual( { final: v.final, exact: v.exact }, { final: true, exact: false } );
	assert.ok( v.pivots > 0 && v.pivots < g.N );
	const partial = v.pivots;
	assert.equal( lens.state.busy, false );
	// ... and is taken up again, from where it stopped and to the end, once a channel shows it
	events.length = 0;
	assert.equal( await lens.setSize( 'closeness' ), true );
	v = lens.values( 'closeness' );
	assert.deepEqual( { final: v.final, exact: v.exact, pivots: v.pivots }, { final: true, exact: true, pivots: g.N } );
	const data = events.filter( ( e ) => e.reason === 'data' ).map( ( e ) => e.metrics.closeness.pivots );
	assert.ok( data[ 0 ] >= partial, `resumed at ${data[ 0 ]} of ${partial} pivots, not from zero` );
	close( column( buffers.colorCalls.at( - 1 ), 4, 3 ), sized( GA.normalize( GA.closeness( g, { samples: Infinity } ).closeness, { mode: 'robust' } ) ), 1e-6 );
	lens.dispose();

} );

test( 'Lens: the time limit comes and goes with the channel while a request() waits', async () => {

	const g = load( 'generated/hsbm-500' );
	const { lens, client, events } = setup( g, { refineMs: 200 } );

	// asked for without a channel (time limit), then a channel shows it: when the limit runs
	// out the refinement goes on, and both callers get the exact metric
	const card = lens.request( 'betweenness' );
	const shown = lens.setColor( 'betweenness' );
	assert.equal( await shown, true );
	const v = await card;
	assert.deepEqual( { final: v.final, exact: v.exact, pivots: v.pivots }, { final: true, exact: true, pivots: g.N } );
	assert.equal( client._nextId, 3, 'the request with the limit, then one without' );
	// the limit that ran out in between never showed as the end of the refinement
	const seen = events.filter( ( e ) => e.reason === 'data' ).map( ( e ) => e.metrics.betweenness );
	assert.ok( seen.length > 3 );
	assert.deepEqual( seen.filter( ( m ) => m.final ).map( ( m ) => m.exact ), [ true ] );
	for ( let k = 1; k < seen.length; k ++ ) assert.ok( seen[ k ].pivots >= seen[ k - 1 ].pivots, 'it carries on, it does not start again' );
	lens.clear();

	// shown without a limit, then the channel goes while a request() still waits: back under
	// the limit, and the request gets the estimate reached by then
	const colour = lens.setColor( 'harmonic' );
	const waiting = lens.request( 'harmonic' );
	await new Promise( ( r ) => setImmediate( r ) );
	const id = client._nextId;
	lens.setColor( null );
	assert.equal( client._nextId, id + 1, 'asked again, with the limit' );
	assert.equal( await colour, false, 'the channel moved on' );
	const h = await waiting;
	assert.deepEqual( { final: h.final, exact: h.exact }, { final: true, exact: false } );
	assert.ok( h.pivots > 0 && h.pivots < g.N, `${h.pivots} pivots` );
	assert.equal( lens.state.busy, false );
	assert.equal( client.busy, false );
	// refine( id, 0 ) is a wish for the exact metric: no channel, no limit, and it stays that way
	const exact = lens.refine( 'harmonic', 0 );
	lens.setGlow( 'degree' );
	lens.setGlow( null );
	assert.equal( await exact, true );
	assert.equal( lens.values( 'harmonic' ).exact, true );
	lens.dispose();

} );

test( 'Lens: untilExact off: refineMs stops every slow metric early; refine() continues it', async () => {

	const g = load( 'generated/hsbm-500' );
	const { lens, buffers } = setup( g, { refineMs: 200, untilExact: false } );
	assert.equal( await lens.setColor( 'betweenness' ), true );
	let v = lens.values( 'betweenness' );
	assert.equal( v.final, true );
	assert.equal( v.exact, false );
	assert.ok( v.pivots > 0 && v.pivots < g.N );
	const partial = v.pivots;
	const reference = GA.betweennessTask( g, { samples: Infinity } );
	reference.step( partial );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lit( GA.normalize( reference.result().node, { mode: 'log' } ), 'viridis' ) ), 1e-7 );

	// another 200 ms: further, still not exact
	assert.equal( await lens.refine( 'betweenness' ), true );
	v = lens.values( 'betweenness' );
	assert.ok( v.pivots > partial && v.exact === false, `${partial} -> ${v.pivots} pivots` );
	// two callers of one refinement share it
	const both = await Promise.all( [ lens.refine( 'betweenness', 0 ), lens.refine( 'betweenness', 0 ) ] );
	assert.deepEqual( both, [ true, true ] );
	v = lens.values( 'betweenness' );
	assert.equal( v.exact, true );
	assert.equal( await lens.refine( 'betweenness' ), true, 'already exact' );
	assert.equal( await lens.refine( 'never-asked' ), false );
	lens.dispose();

} );

test( 'Lens: a graph change re-applies the active lens to the new graph', async () => {

	const a = load( 'generated/hsbm-500' ), b = load( 'imsv/lesmis' );
	const { lens, buffers } = setup( a );
	await lens.setSize( 'pagerank' );
	const pending = lens.setColor( 'betweenness' ); // still refining when the graph changes
	const next = fakeBuffers( b );
	lens.setGraph( next, b );
	assert.equal( await pending, false, 'the promise of the old graph resolves false' );
	const calls = buffers.colorCalls.length;
	assert.deepEqual( [ lens.state.size, lens.state.color ], [ 'pagerank', 'betweenness' ] );
	assert.equal( lens.values( 'pagerank' ), null, 'values of the old graph are gone' );

	// wait until both metrics of the new graph are final
	for ( let k = 0; k < 20000 && lens.state.busy; k ++ ) await new Promise( ( r ) => setImmediate( r ) );
	assert.equal( lens.state.busy, false );
	const got = next.colorCalls.at( - 1 );
	close( column( got, 4, 3 ), sized( GA.normalize( GA.pageRank( b ), { mode: 'log' } ) ), 1e-6 );
	close( rgb( got ), rgb( lit( GA.normalize( GA.betweenness( b, { samples: Infinity } ), { mode: 'log' } ), 'viridis' ) ), 1e-7 );
	assert.equal( buffers.colorCalls.length, calls, 'nothing is written to the old buffers after the switch' );
	assert.equal( lens.values( 'betweenness' ).raw.length, b.N );
	lens.dispose();

} );

test( 'Lens: values(), request(), load() and describe() for the UI', async () => {

	const g = load( 'imsv/karate' );
	const { lens, buffers } = setup( g );
	assert.equal( lens.values( 'pagerank' ), null );
	assert.deepEqual( lens.describe( 0 ), [] );
	const uploads = buffers.colorCalls.length;
	const v = await lens.request( 'pagerank' );
	assert.equal( buffers.colorCalls.length, uploads, 'request() does not touch the picture' );
	const pr = GA.pageRank( g );
	close( v.raw, pr, 0 );
	close( v.normalized, GA.normalize( pr, { mode: 'log' } ), 0 );
	assert.equal( v.scale, 'log' );
	assert.equal( v.label, 'PageRank' );
	const order = Array.from( pr.keys() ).sort( ( x, y ) => pr[ y ] - pr[ x ] || x - y );
	assert.deepEqual( Array.from( v.order ), order );
	// rank = number of nodes with a larger value (structurally equivalent nodes tie)
	for ( let i = 0; i < g.N; i ++ ) assert.equal( v.rank[ i ], pr.filter( ( x ) => x > pr[ i ] ).length );
	assert.equal( v.rank[ order[ 0 ] ], 0 );
	assert.deepEqual( v.domain, [ Math.min( ...pr ), Math.max( ...pr ) ] );
	assert.equal( lens.values( 'pagerank' ).rank, v.rank, 'ranks are computed once' );

	// ties share a rank (k-core has few distinct values)
	const core = await lens.request( 'core' );
	const kc = GA.kCore( g ).core;
	for ( let i = 0; i < g.N; i ++ ) assert.equal( core.rank[ i ], kc.filter( ( c ) => c > kc[ i ] ).length );

	await lens.load();
	const cheap = lens.metrics.filter( ( m ) => m.cost !== 'slow' && ! m.needs ).map( ( m ) => m.id );
	const d = lens.describe( 33 );
	assert.deepEqual( d.map( ( e ) => e.id ), cheap );
	const entry = d.find( ( e ) => e.id === 'degree' );
	assert.deepEqual( entry, { id: 'degree', label: 'Degree', value: 17, normalized: 1, rank: 1, of: 34, exact: true, final: true } );
	assert.equal( d.find( ( e ) => e.id === 'pagerank' ).rank, 1 );
	assert.deepEqual( lens.describe( - 1 ), [] );
	assert.deepEqual( lens.describe( 34 ), [] );

	assert.ok( lens.metrics.some( ( m ) => m.id === 'relevance' ) );
	assert.ok( lens.edgeMetrics.length >= 2 );
	assert.ok( lens.ramps.length >= 4 && lens.ramps[ 0 ].css );
	lens.dispose();

} );

test( 'Lens: relevance (personalised PageRank) follows the selection', async () => {

	const g = load( 'imsv/lesmis' );
	const { lens, buffers, app } = setup( g );
	// no selection yet: the channel shows the graph's own colours
	assert.equal( await lens.setColor( 'relevance', { ramp: 'ice' } ), true );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lens.base.colors ), 0 );
	assert.equal( lens.legend( 'color' ).ready, false );

	const wait = async () => {

		for ( let k = 0; k < 20000 && lens.state.busy; k ++ ) await new Promise( ( r ) => setImmediate( r ) );

	};

	app.events.dispatchEvent( new CustomEvent( 'select', { detail: { index: 11 } } ) );
	assert.deepEqual( lens.state.sources, [ 11 ] );
	await wait();
	const weights = new Float32Array( g.N );
	weights[ 11 ] = 1;
	const ppr = GA.pageRank( g, { personalization: weights } );
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lit( GA.normalize( ppr, { mode: 'log' } ), 'ice' ) ), 1e-7 );
	assert.equal( lens.values( 'relevance' ).order[ 0 ], 11, 'the selected node is the most relevant one' );

	// two quick selections: only the last one is shown
	lens.setSources( [ 3 ] );
	lens.setSources( [ 48 ] );
	await wait();
	weights.fill( 0 );
	weights[ 48 ] = 1;
	close( lens.values( 'relevance' ).raw, GA.pageRank( g, { personalization: weights } ), 0 );

	// selection cleared: back to the cluster palette
	app.events.dispatchEvent( new CustomEvent( 'select', { detail: { index: - 1 } } ) );
	await wait();
	close( rgb( buffers.colorCalls.at( - 1 ) ), rgb( lens.base.colors ), 0 );
	assert.equal( lens.values( 'relevance' ), null );
	lens.dispose();

} );

test( 'Lens: component is coloured categorically', async () => {

	// two triangles and an isolated node
	const g = prepareGraph( { name: 'parts', nodeCount: 7, edges: [ 0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3 ], labels: null, hierarchy: null } );
	const { lens, buffers } = setup( g );
	await lens.setColor( 'component' );
	const comp = GA.connectedComponents( g ).component;
	const got = buffers.colorCalls.at( - 1 );
	for ( let i = 0; i < g.N; i ++ ) {

		const c = categoryColor( comp[ i ] ), gain = emissiveGain( ...c, 0.4 );
		close( got.subarray( 4 * i, 4 * i + 3 ), c.map( ( v ) => v * gain ), 1e-7 );

	}

	assert.ok( emissiveGain( ...categoryColor( 0 ), 0.4 ) < 0.7 );
	assert.equal( lens.legend( 'color' ).categorical, true );
	lens.dispose();

} );

test( 'Lens: summary (cached per graph, labels) and the per-edge channel', async () => {

	const g = load( 'imsv/karate' );
	const { lens, client, buffers } = setup( g );
	const partial = [];
	const p1 = lens.summary( { onPreview: ( s ) => partial.push( s.stage ) } );
	const p2 = lens.summary();
	assert.equal( p1, p2, 'one computation per graph' );
	const s = await p1;
	assert.equal( s.nodes, 34 );
	assert.equal( s.stage, 'done' );
	assert.equal( s.top.betweenness.labels.length, 10 );
	assert.deepEqual( partial, [ 'basic', 'centrality', 'distance' ] );
	const id = client._nextId;
	const late = [];
	assert.equal( await lens.summary( { onPreview: ( x ) => late.push( x.stage ) } ), s );
	assert.deepEqual( late, [ 'done' ], 'a late listener gets what there is' );
	assert.equal( client._nextId, id );

	const tri = await lens.edgeChannel( 'triangles' );
	assert.equal( tri.values.length, g.E );
	assert.equal( tri.scale, 'log' );
	assert.equal( tri.exact, true );
	const raw = GA.clusteringCoefficient( g, { edges: true } ).edgeTriangles.subarray( 0, g.E );
	close( tri.raw, Float32Array.from( raw ), 0 );
	close( tri.values, GA.normalize( Float32Array.from( raw ), { mode: 'log' } ), 0 );
	assert.equal( await lens.edgeChannel( 'triangles' ), tri, 'final channels are cached' );

	const updates = [];
	const eb = await lens.edgeChannel( 'betweenness', { onUpdate: ( u ) => updates.push( u.pivots ) } );
	assert.equal( eb.exact, true );
	close( eb.values, GA.normalize( GA.edgeBetweenness( g, { samples: Infinity } ), { mode: 'rank' } ), 0 );
	assert.ok( eb.values.every( ( v ) => v >= 0 && v <= 1 ) );
	assert.throws( () => lens.edgeChannel( 'nope' ), /unknown edge metric/ );

	// a new graph forgets both
	const b = load( 'imsv/lesmis' );
	lens.setGraph( fakeBuffers( b ), b );
	assert.equal( ( await lens.summary() ).nodes, 77 );
	assert.equal( ( await lens.edgeChannel( 'triangles' ) ).values.length, b.E );
	assert.ok( buffers.colorCalls.length >= 0 );
	lens.dispose();

} );

test( 'Lens: edge channels asked for together each get their values; callers of one channel share it', async () => {

	const g = load( 'generated/hsbm-500' );
	const { lens, client } = setup( g );
	const id = client._nextId;
	const a = [], b = [];
	const first = lens.edgeChannel( 'betweenness', { onUpdate: ( u ) => a.push( u.pivots ) } );
	const all = await Promise.all( [
		lens.edgeChannel( 'triangles' ), lens.edgeChannel( 'embeddedness' ), first,
		lens.edgeChannel( 'betweenness', { scale: 'log' } ),
		lens.edgeChannel( 'betweenness', { onUpdate: ( u ) => b.push( u.pivots ) } ),
	] );
	assert.equal( client._nextId, id + 4, 'four channels, four requests' );
	const [ tri, emb, rank, log, again ] = all;
	assert.ok( all.every( ( r ) => r && r.values.length === g.E && r.exact ), 'none was superseded by another' );
	assert.deepEqual( [ tri.metric, emb.metric, rank.scale, log.scale ], [ 'triangles', 'embeddedness', 'rank', 'log' ] );
	assert.equal( again, rank, 'one result for both callers' );
	assert.ok( a.length >= 2, `${a.length} estimates on the way` );
	assert.deepEqual( b, a, 'and the same updates' );
	const eb = GA.edgeBetweenness( g, { samples: Infinity } );
	close( rank.raw, eb, 0 );
	close( rank.values, GA.normalize( eb, { mode: 'rank' } ), 0 );
	close( log.values, GA.normalize( eb, { mode: 'log' } ), 0 );
	close( emb.values, GA.normalize( emb.raw, { mode: 'linear' } ), 0 );
	assert.equal( await lens.edgeChannel( 'betweenness' ), rank, 'exact channels are kept' );
	assert.equal( client._nextId, id + 4 );

	// a channel that is no longer shown is released: null, and the worker stops
	const next = load( 'generated/hsbm-500' );
	lens.setGraph( fakeBuffers( next ), next );
	const dropped = lens.edgeChannel( 'betweenness' );
	const kept = lens.edgeChannel( 'betweenness', { scale: 'log' } );
	await new Promise( ( r ) => setImmediate( r ) );
	lens.releaseEdgeChannel( 'betweenness' );
	assert.equal( await dropped, null );
	assert.equal( ( await kept ).exact, true, 'the other scale of the metric was not touched' );
	assert.equal( client.busy, false );
	lens.releaseEdgeChannel( 'nope' );
	lens.releaseEdgeChannel();

	// a time limit: an estimate, and asking again carries on to the end
	const third = load( 'generated/hsbm-500' );
	lens.setGraph( fakeBuffers( third ), third );
	const capped = await lens.edgeChannel( 'betweenness', { maxMs: 150 } );
	assert.deepEqual( { final: capped.final, exact: capped.exact }, { final: true, exact: false } );
	assert.ok( capped.pivots > 0 && capped.pivots < third.N );
	const updates = [];
	const full = await lens.edgeChannel( 'betweenness', { onUpdate: ( u ) => updates.push( u.pivots ) } );
	assert.equal( full.exact, true );
	assert.ok( updates.length > 0 && updates[ 0 ] >= capped.pivots, `continued at ${updates[ 0 ]} of ${capped.pivots}` );
	lens.dispose();

} );

test( 'Lens.dispose: buffers and graph get the cluster colours and degree radii back; no listener stays', async () => {

	const g = load( 'imsv/lesmis' );
	const baseColors = Float32Array.from( g.colors ), baseRadii = Float32Array.from( g.radii );
	const { lens, buffers, app, client } = setup( g );
	await lens.setSize( 'pagerank' );
	await lens.setColor( 'core' );
	await lens.setGlow( 'degree', { top: 0.2 } );
	const fx = buffers.nodeFx.value.array, before = Float32Array.from( fx );
	assert.ok( column( fx, 4, 2 ).some( ( v ) => v > 0 ) && column( fx, 4, 3 ).some( ( v ) => v > 0 ) );
	assert.ok( rgb( buffers.target ).some( ( v, k ) => v !== rgb( baseColors )[ k ] ) );

	lens.dispose();
	close( rgb( buffers.target ), rgb( baseColors ), 0, 'the targets on the buffers' );
	close( column( buffers.target, 4, 3 ), baseRadii, 0 );
	close( g.colors, baseColors, 0, 'the mirrored arrays of the graph' );
	close( g.radii, baseRadii, 0 );
	close( column( fx, 4, 2 ), new Float32Array( g.N ), 0, 'no glow' );
	close( column( fx, 4, 3 ), new Float32Array( g.N ), 0, 'no value' );
	close( column( fx, 4, 0 ), column( before, 4, 0 ), 0, 'x / y are not ours' );
	close( column( fx, 4, 1 ), column( before, 4, 1 ), 0 );
	assert.equal( client.busy, false );
	// the 'select' listener is gone
	app.events.dispatchEvent( new CustomEvent( 'select', { detail: { index: 3 } } ) );
	assert.deepEqual( lens.state.sources, [] );

	// mirror off: the buffers are put back all the same
	const h = load( 'imsv/karate' );
	const other = setup( h, { mirror: false } );
	await other.lens.setColor( 'core' );
	other.lens.dispose();
	close( rgb( other.buffers.target ), rgb( h.colors ), 0 );
	// before any graph: nothing to do
	new Lens( { client } ).dispose();

} );

test( 'Lens: the same graph again (new buffers) keeps the cluster palette as its base', async () => {

	const g = load( 'imsv/lesmis' );
	const baseColors = Float32Array.from( g.colors ), baseRadii = Float32Array.from( g.radii );
	const { lens, buffers } = setup( g );
	await lens.setSize( 'pagerank' );
	await lens.setColor( 'core' );
	assert.ok( g.colors.some( ( v, k ) => v !== baseColors[ k ] ), 'mirror: the graph arrays hold lens values' );

	// buffers rebuilt for the graph on screen start from those arrays, as GraphBuffers does
	const next = fakeBuffers( g );
	next.setColors( g.colors, g.radii );
	lens.setGraph( next, g );
	close( lens.base.colors, baseColors, 0, 'the base is still the cluster palette' );
	close( lens.base.radii, baseRadii, 0 );
	close( rgb( next.target ), rgb( baseColors ), 0, 'the new buffers were put right at once' );
	assert.deepEqual( next.colorOptions.at( - 1 ), { immediate: true } );
	await idle( lens );
	close( rgb( next.target ), rgb( lit( GA.normalize( GA.kCore( g ).core, { mode: 'linear' } ), 'viridis' ) ), 1e-7, 'and the lens is back on' );
	lens.clear();
	close( rgb( next.target ), rgb( colorize( g.leafRank, g.palette ) ), 0, 'clear() restores the cluster palette' );
	close( column( next.target, 4, 3 ), baseRadii, 0 );
	// the same buffers again: eased, not snapped
	await lens.setColor( 'core' );
	lens.setGraph( next, g );
	assert.deepEqual( next.colorOptions.at( - 1 ), { immediate: false } );
	close( lens.base.colors, baseColors, 0 );
	await idle( lens );

	// a palette chosen through the lens is the base from then on
	lens.setPalette( 'aurora' );
	lens.setGraph( fakeBuffers( g ), g );
	close( lens.base.colors, colorize( g.leafRank, 'aurora' ), 0 );
	await idle( lens );

	// leaving a graph puts its arrays back, so it can be shown again later
	const other = load( 'imsv/karate' );
	lens.setGraph( fakeBuffers( other ), other );
	close( g.colors, colorize( g.leafRank, 'aurora' ), 0 );
	close( g.radii, baseRadii, 0 );
	assert.equal( buffers.colorCalls.length > 0, true );
	await idle( lens );
	lens.dispose();

} );

test( 'Lens: mirror off keeps the graph arrays; a lens set before the first graph is applied to it', async () => {

	const g = load( 'imsv/karate' );
	const radii = Float32Array.from( g.radii ), colors = Float32Array.from( g.colors );
	const client = new AnalysisClient( { createWorker: () => new FakeWorker() } );
	const changes = [];
	const lens = new Lens( { client, mirror: false, ramp: 'ember', sizeRange: [ 0.7, 1.6 ], onChange: ( s ) => changes.push( s.reason ) } );
	assert.equal( await lens.setSize( 'degree' ), false, 'no graph yet' );
	assert.equal( await lens.setColor( 'clustering' ), false );
	assert.deepEqual( [ lens.state.size, lens.state.color, lens.state.ramp ], [ 'degree', 'clustering', 'ember' ] );
	const buffers = fakeBuffers( g );
	lens.setGraph( buffers, g );
	for ( let k = 0; k < 20000 && lens.state.busy; k ++ ) await new Promise( ( r ) => setImmediate( r ) );
	const got = buffers.colorCalls.at( - 1 );
	close( column( got, 4, 3 ), sized( GA.normalize( g.degree, { mode: 'log' } ) ), 1e-6 );
	close( rgb( got ), rgb( lit( GA.clusteringCoefficient( g ).local, 'ember' ) ), 1e-6 );
	close( g.radii, radii, 0, 'graph.radii is left alone' );
	close( g.colors, colors, 0 );
	assert.notEqual( lens.radii, g.radii );
	close( lens.radii, column( got, 4, 3 ), 0 );
	assert.ok( changes.includes( 'data' ) && changes.includes( 'graph' ) );
	lens.dispose();

} );
