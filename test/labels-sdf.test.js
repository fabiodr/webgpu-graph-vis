// Tests for the pure parts of the GPU labels: src/render/sdfAtlas.js (distance transform, SDF
// encoding, shelf packing, dynamic atlas, segmentation, layout) and the CPU side of
// src/render/TextLabels.js (visibility decisions, label state easing, glyph buffer) with an
// injected glyph source instead of a canvas. The shaders are checked in dev/labels.html.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PerspectiveCamera, Vector3 } from 'three/webgpu';
import {
	edt1d, edt2d, sdfFromAlpha, ShelfPacker, GlyphAtlas, graphemes, segmentText, layoutUnits, makeKerning, SDF_DEFAULTS,
} from '../src/render/sdfAtlas.js';
import { TextLabels, CTX, NB, HOV, SEL } from '../src/render/TextLabels.js';

const INF = 1e20;

function lcg( seed ) {

	let s = seed >>> 0;
	return () => ( ( s = ( Math.imul( s, 1664525 ) + 1013904223 ) >>> 0 ) / 4294967296 );

}

// ── distance transform ──────────────────────────────────────────────────────────────────────

test( 'edt1d: lower envelope equals brute force (finite and INF samples)', () => {

	const rnd = lcg( 7 );
	for ( let round = 0; round < 200; round ++ ) {

		const n = 1 + Math.floor( rnd() * 40 );
		const f = new Float64Array( n );
		for ( let i = 0; i < n; i ++ ) f[ i ] = rnd() < 0.5 ? INF : rnd() < 0.5 ? 0 : rnd() * 9;
		if ( round % 10 === 0 ) f.fill( INF ); // no seed at all
		const expect = new Float64Array( n );
		for ( let q = 0; q < n; q ++ ) {

			let best = Infinity;
			for ( let p = 0; p < n; p ++ ) best = Math.min( best, ( q - p ) * ( q - p ) + f[ p ] );
			expect[ q ] = best;

		}

		const grid = Float64Array.from( f );
		edt1d( grid, 0, 1, n, new Float64Array( n ), new Int32Array( n ), new Float64Array( n + 1 ) );
		for ( let q = 0; q < n; q ++ ) {

			if ( expect[ q ] > 1e19 ) assert.ok( grid[ q ] > 1e19, 'unreachable stays far' );
			else assert.ok( Math.abs( grid[ q ] - expect[ q ] ) < 1e-9, `round ${round} q ${q}: ${grid[ q ]} vs ${expect[ q ]}` );

		}

	}

} );

test( 'edt1d: strided access leaves other samples alone', () => {

	const grid = new Float64Array( 12 ).fill( 5 );
	for ( let q = 0; q < 4; q ++ ) grid[ 1 + 3 * q ] = INF;
	grid[ 1 + 3 * 2 ] = 0;
	edt1d( grid, 1, 3, 4, new Float64Array( 4 ), new Int32Array( 4 ), new Float64Array( 5 ) );
	assert.deepEqual( [ grid[ 1 ], grid[ 4 ], grid[ 7 ], grid[ 10 ] ], [ 4, 1, 0, 1 ] );
	for ( let i = 0; i < 12; i ++ ) if ( i % 3 !== 1 ) assert.equal( grid[ i ], 5 );

} );

test( 'edt2d: exact squared Euclidean distances to the nearest seed', () => {

	const W = 23, H = 17, rnd = lcg( 3 );
	const seeds = [];
	const grid = new Float64Array( W * H ).fill( INF );
	for ( let k = 0; k < 9; k ++ ) {

		const x = Math.floor( rnd() * W ), y = Math.floor( rnd() * H );
		seeds.push( [ x, y ] );
		grid[ y * W + x ] = 0;

	}

	const m = Math.max( W, H );
	edt2d( grid, 0, 0, W, H, W, new Float64Array( m ), new Int32Array( m ), new Float64Array( m + 1 ) );
	for ( let y = 0; y < H; y ++ ) {

		for ( let x = 0; x < W; x ++ ) {

			let best = Infinity;
			for ( const [ sx, sy ] of seeds ) best = Math.min( best, ( x - sx ) ** 2 + ( y - sy ) ** 2 );
			assert.equal( grid[ y * W + x ], best, `(${x}, ${y})` );

		}

	}

	// sub-rectangle: cells outside it are not touched
	const g2 = new Float64Array( W * H ).fill( INF );
	g2[ 5 * W + 6 ] = 0;
	edt2d( g2, 4, 3, 8, 6, W, new Float64Array( m ), new Int32Array( m ), new Float64Array( m + 1 ) );
	assert.equal( g2[ 5 * W + 9 ], 9 );
	assert.equal( g2[ 8 * W + 4 ], 13 );
	assert.equal( g2[ 5 * W + 12 ], INF, 'outside the sub-rectangle' );
	assert.equal( g2[ 2 * W + 6 ], INF, 'outside the sub-rectangle' );

} );

// ── SDF of simple shapes ────────────────────────────────────────────────────────────────────

/** value → signed distance in texels (positive outside), the inverse of the 8-bit encoding. */
const decode = ( v, { radius = SDF_DEFAULTS.radius, cutoff = SDF_DEFAULTS.cutoff } = {} ) => ( 1 - cutoff - v / 255 ) * radius;

test( 'sdfFromAlpha: anti-aliased vertical edge (half plane)', () => {

	// ink for x < 10, the column x = 10 is half covered: the outline runs through its center
	const w = 24, h = 20, alpha = new Uint8Array( w * h );
	for ( let y = 0; y < h; y ++ ) {

		for ( let x = 0; x < 10; x ++ ) alpha[ y * w + x ] = 255;
		alpha[ y * w + 10 ] = 128;

	}

	const B = SDF_DEFAULTS.buffer, W = w + 2 * B, H = h + 2 * B;
	const sdf = sdfFromAlpha( alpha, w, h );
	assert.equal( sdf.length, W * H );
	const row = ( B + 10 ) * W;
	const step = 16 / 255; // texels per 8-bit level
	for ( let x = 7; x <= 20; x ++ ) {

		const d = decode( sdf[ row + B + x ] );
		const expect = Math.max( - 4, Math.min( 12, x - 10 ) ); // clamped to the encoded range
		assert.ok( Math.abs( d - expect ) <= 0.02 + step, `x ${x}: ${d} vs ${expect}` );

	}

	// the outline value is 1 − cutoff, deep inside saturates at 255, monotone along the row
	assert.ok( Math.abs( sdf[ row + B + 10 ] - 191 ) <= 1 );
	assert.equal( sdf[ row + B + 2 ], 255 );
	for ( let x = 1; x < w; x ++ ) assert.ok( sdf[ row + B + x ] <= sdf[ row + B + x - 1 ], 'monotone' );

} );

test( 'sdfFromAlpha: box with hard edges and the padded border', () => {

	const w = 12, h = 9, alpha = new Uint8Array( w * h ).fill( 255 );
	const B = SDF_DEFAULTS.buffer, W = w + 2 * B, H = h + 2 * B;
	const sdf = sdfFromAlpha( alpha, w, h );
	// hard edges lie between pixel centers: the first pixel outside is at +1 (centers are 1 apart),
	// the outline (value 191) is crossed between the two
	const mid = ( B + 4 ) * W;
	assert.ok( sdf[ mid + B - 1 ] < 191 && sdf[ mid + B ] > 191 );
	assert.ok( Math.abs( decode( sdf[ mid + B - 3 ] ) - 3 ) < 0.07 );
	// corners: Euclidean, not city-block
	const dCorner = decode( sdf[ ( B - 3 ) * W + B - 4 ] );
	assert.ok( Math.abs( dCorner - 5 ) < 0.07, `corner distance ${dCorner}` );
	// the field has run out at the cell border (no halo clipping, no bleeding between cells)
	for ( let x = 0; x < W; x ++ ) assert.equal( sdf[ x ] + sdf[ ( H - 1 ) * W + x ], 0 );
	for ( let y = 0; y < H; y ++ ) assert.equal( sdf[ y * W ] + sdf[ y * W + W - 1 ], 0 );
	// empty bitmap: everything far outside
	assert.ok( sdfFromAlpha( new Uint8Array( 16 ), 4, 4 ).every( ( v ) => v === 0 ) );

} );

test( 'sdfFromAlpha: anti-aliased disc matches the analytic distance', () => {

	const R = 9.3, n = 28, c = n / 2 + 0.2;
	const alpha = new Uint8Array( n * n );
	for ( let y = 0; y < n; y ++ ) {

		for ( let x = 0; x < n; x ++ ) {

			// 8×8 supersampled coverage
			let cov = 0;
			for ( let j = 0; j < 8; j ++ ) for ( let i = 0; i < 8; i ++ ) if ( Math.hypot( x + ( i + 0.5 ) / 8 - c, y + ( j + 0.5 ) / 8 - c ) <= R ) cov ++;
			alpha[ y * n + x ] = Math.round( 255 * cov / 64 );

		}

	}

	const B = SDF_DEFAULTS.buffer, W = n + 2 * B;
	const sdf = sdfFromAlpha( alpha, n, n );
	let maxErr = 0, maxErrCut = 0, count = 0;
	for ( let y = 0; y < W; y ++ ) {

		for ( let x = 0; x < W; x ++ ) {

			const truth = Math.hypot( x - B + 0.5 - c, y - B + 0.5 - c ) - R;
			if ( truth < - 3.5 || truth > 11.5 ) continue; // inside the encoded range
			const err = Math.abs( decode( sdf[ y * W + x ] ) - truth );
			maxErr = Math.max( maxErr, err );
			const ax = x - B, ay = y - B;
			const a = ax >= 0 && ay >= 0 && ax < n && ay < n ? alpha[ ay * n + ax ] : 0;
			if ( a > 0 && a < 255 ) maxErrCut = Math.max( maxErrCut, err );
			count ++;

		}

	}

	assert.ok( count > 800 );
	// Pixels cut by the outline carry their sub-pixel coverage: these fix where the outline is
	// (measured 0.15 texel on this disc; coverage is not exactly linear in distance on diagonals).
	// Everywhere else the transform measures to pixel centers and adds the sub-pixel offsets in
	// quadrature (the tiny-sdf approximation): up to ~0.6 texel off, i.e. 0.14 px at 11 px text.
	assert.ok( maxErrCut < 0.25, `error on the outline ${maxErrCut}` );
	assert.ok( maxErr < 0.7, `max error ${maxErr}` );

} );

// ── packing ─────────────────────────────────────────────────────────────────────────────────

function overlaps( a, b ) {

	return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

}

test( 'ShelfPacker: no overlaps, inside the bounds, grows without moving anything', () => {

	const rnd = lcg( 11 );
	const packer = new ShelfPacker( 256, 128 );
	const placed = [];
	let grew = 0;
	for ( let k = 0; k < 300; k ++ ) {

		const w = 30 + Math.floor( rnd() * 40 ), h = 44 + Math.floor( rnd() * 30 ); // glyph cells at 48 px
		let at = packer.add( w, h );
		while ( at === null ) {

			if ( packer.width <= packer.height ) packer.resize( packer.width * 2, packer.height );
			else packer.resize( packer.width, packer.height * 2 );
			grew ++;
			assert.ok( grew < 12, 'growth terminates' );
			at = packer.add( w, h );

		}

		placed.push( { x: at.x, y: at.y, w, h } );

	}

	assert.ok( grew >= 3 );
	for ( let i = 0; i < placed.length; i ++ ) {

		const a = placed[ i ];
		assert.ok( a.x >= 0 && a.y >= 0 && a.x + a.w <= packer.width && a.y + a.h <= packer.height, 'inside' );
		for ( let j = 0; j < i; j ++ ) assert.ok( ! overlaps( a, placed[ j ] ), `${i} overlaps ${j}` );

	}

	const area = placed.reduce( ( s, r ) => s + r.w * r.h, 0 );
	assert.equal( packer.used, area );
	assert.ok( area / ( packer.width * packer.bottom ) > 0.6, `packing efficiency ${( area / ( packer.width * packer.bottom ) ).toFixed( 2 )}` );
	// too large for the atlas
	assert.equal( new ShelfPacker( 64, 64 ).add( 65, 10 ), null );
	assert.equal( new ShelfPacker( 64, 64 ).add( 10, 65 ), null );

} );

// ── atlas with an injected raster ───────────────────────────────────────────────────────────

/** Fake glyph source: every character is a solid box; width depends on the character. */
function fakeSource( { size = 48, calls = { raster: 0, measure: 0 } } = {} ) {

	const adv = ( ch ) => ( ch === ' ' ? 12 : ch === 'i' || ch === 'l' || ch === '.' ? 12 : ch === 'W' || ch === 'M' ? 40 : 26 );
	const raster = ( unit ) => {

		calls.raster ++;
		if ( unit === '\u{1F600}' ) return { skip: true, advance: 0 };
		const a = [ ...unit ].reduce( ( s, ch ) => s + adv( ch ), 0 );
		if ( unit === ' ' ) return { width: 0, height: 0, alpha: null, left: 0, top: 0, advance: a };
		const w = a - 4, h = /[a-z]/.test( unit ) ? 26 : 34;
		return { width: w, height: h, alpha: new Uint8Array( w * h ).fill( 255 ), left: 2, top: h, advance: a };

	};

	// kerning: "AV" pairs are tightened by 4 px
	const measure = ( s ) => {

		calls.measure ++;
		let w = 0;
		const chars = [ ...s ];
		for ( let k = 0; k < chars.length; k ++ ) {

			w += adv( chars[ k ] );
			if ( chars[ k ] === 'A' && chars[ k + 1 ] === 'V' ) w -= 4;

		}

		return w;

	};

	return { raster, measure, metrics: { ascent: 0.9, descent: 0.2, capHeight: 0.7, xHeight: 0.5 }, calls, size };

}

test( 'GlyphAtlas: on-demand glyphs, metrics, caching', () => {

	const src = fakeSource();
	const atlas = new GlyphAtlas( { raster: src.raster, width: 256, height: 128 } );
	const B = atlas.buffer;
	assert.equal( atlas.version, 0 );
	const a = atlas.glyph( 'a' );
	assert.equal( atlas.version, 1 );
	assert.equal( atlas.glyph( 'a' ), a, 'cached' );
	assert.equal( src.calls.raster, 1 );
	assert.deepEqual( [ a.w, a.h, a.advance ], [ 22 + 2 * B, 26 + 2 * B, 26 ] );
	assert.equal( a.left, 2 - B, 'cell left edge relative to the pen' );
	assert.equal( a.top, 26 + B, 'cell top edge above the baseline' );
	// texels: ink in the middle, nothing on the cell border, outline value in between
	const at = ( g, x, y ) => atlas.data[ ( g.y + y ) * atlas.width + g.x + x ];
	assert.equal( at( a, a.w >> 1, a.h >> 1 ), 255 );
	assert.equal( at( a, 0, a.h >> 1 ), 0 );
	assert.equal( at( a, a.w - 1, a.h - 1 ), 0 );
	assert.ok( at( a, B, a.h >> 1 ) > 191 && at( a, B - 1, a.h >> 1 ) < 191 );

	const space = atlas.glyph( ' ' );
	assert.deepEqual( [ space.w, space.h, space.advance, space.skip ], [ 0, 0, 12, false ] );
	const emoji = atlas.glyph( '\u{1F600}' );
	assert.ok( emoji.skip );
	assert.equal( emoji.advance, 0 );
	assert.equal( atlas.skipped, 1 );
	assert.equal( atlas.version, 1, 'blank glyphs do not touch the texture' );
	assert.throws( () => new GlyphAtlas( {} ), /raster/ );

} );

test( 'GlyphAtlas: grows by doubling, cells keep their position and content', () => {

	const src = fakeSource();
	const atlas = new GlyphAtlas( { raster: src.raster, width: 64, height: 64, maxSize: 256 } );
	const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
	const seen = [];
	const snapshot = ( g ) => {

		const out = [];
		for ( let y = 0; y < g.h; y ++ ) out.push( ...atlas.data.subarray( ( g.y + y ) * atlas.width + g.x, ( g.y + y ) * atlas.width + g.x + g.w ) );
		return out;

	};

	let sizeVersion = atlas.sizeVersion;
	for ( const ch of chars ) {

		const g = atlas.glyph( ch );
		if ( g.missing ) break;
		seen.push( { g, x: g.x, y: g.y, texels: snapshot( g ) } );
		if ( atlas.sizeVersion !== sizeVersion ) {

			sizeVersion = atlas.sizeVersion;
			assert.equal( atlas.data.length, atlas.width * atlas.height );
			for ( const s of seen ) {

				assert.deepEqual( [ s.g.x, s.g.y ], [ s.x, s.y ], 'cells never move' );
				assert.deepEqual( snapshot( s.g ), s.texels, 'cell content survives the reallocation' );

			}

		}

	}

	assert.ok( atlas.sizeVersion >= 3, 'grew several times' );
	assert.ok( atlas.width <= 256 && atlas.height <= 256 );
	assert.ok( atlas.missing > 0, 'full at maxSize: later glyphs are reported missing' );
	const last = atlas.glyph( '9' );
	assert.ok( last.missing && last.w === 0 && last.advance === 26, 'missing glyphs keep their advance' );
	for ( let i = 0; i < seen.length; i ++ ) for ( let j = 0; j < i; j ++ ) assert.ok( ! overlaps( seen[ i ].g, seen[ j ].g ) );
	// a cell that can never fit is missing, not an endless loop
	const wide = new GlyphAtlas( { raster: () => ( { width: 400, height: 10, alpha: new Uint8Array( 4000 ), left: 0, top: 10, advance: 400 } ), width: 64, height: 64, maxSize: 256 } );
	assert.ok( wide.glyph( 'x' ).missing );

} );

// ── segmentation and layout ─────────────────────────────────────────────────────────────────

test( 'segmentText: graphemes, accents, emoji, whitespace, shaped runs, truncation', () => {

	assert.deepEqual( segmentText( 'Mlle.Baptistine' ), [ ...'Mlle.Baptistine' ] );
	assert.deepEqual( segmentText( 'Éponine' ), [ 'É', 'p', 'o', 'n', 'i', 'n', 'e' ] );
	// decomposed accent: one unit (e + combining acute), so the canvas draws the pair
	assert.deepEqual( segmentText( 'Café' ), [ 'C', 'a', 'f', 'é' ] );
	assert.deepEqual( graphemes( 'éa' ), [ 'é', 'a' ] );
	// emoji (single, ZWJ family, flag, keycap, VS16 heart) are dropped; © and ♥ stay text
	assert.deepEqual( segmentText( 'a\u{1F600}b' ), [ 'a', 'b' ] );
	assert.deepEqual( segmentText( '\u{1F468}‍\u{1F469}‍\u{1F467}x\u{1F1EB}\u{1F1F7}y1️⃣z❤️' ), [ 'x', 'y', 'z' ] );
	assert.deepEqual( segmentText( '©♥' ), [ '©', '♥' ] );
	assert.deepEqual( segmentText( 'a \u{1F600} b' ), [ 'a', ' ', 'b' ], 'spaces around a dropped emoji collapse' );
	assert.deepEqual( segmentText( '\u{1F600} a  b' ), [ 'a', ' ', ' ', 'b' ], 'ordinary double spaces stay' );
	// whitespace and control characters
	assert.deepEqual( segmentText( 'a\tb c​d\u0007' ), [ 'a', ' ', 'b', ' ', 'c', 'd' ] );
	// a run of Arabic (with its inner space) is one unit; the surrounding Latin stays per grapheme
	assert.deepEqual( segmentText( 'ab مرحبا بالعالم cd' ), [ 'a', 'b', ' ', 'مرحبا بالعالم', ' ', 'c', 'd' ] );
	assert.deepEqual( segmentText( 'नमस्ते' ), [ 'नमस्ते' ] );
	// CJK is plain per-character text
	assert.deepEqual( segmentText( '図書館' ), [ '図', '書', '館' ] );
	// numbers and non-strings
	assert.deepEqual( segmentText( 1234 ), [ '1', '2', '3', '4' ] );
	assert.deepEqual( segmentText( '' ), [] );
	// truncation
	const cut = segmentText( 'abcdefghijklmnop', { maxUnits: 8 } );
	assert.equal( cut.length, 8 );
	assert.equal( cut.join( '' ), 'abcdefg…' );
	assert.deepEqual( segmentText( 'abc    defgh', { maxUnits: 5 } ), [ 'a', 'b', 'c', '…' ], 'no space before the ellipsis' );

} );

test( 'layoutUnits + makeKerning: pen positions, spaces, skipped units, cached pair kerning', () => {

	const src = fakeSource();
	const atlas = new GlyphAtlas( { raster: src.raster } );
	const kerning = makeKerning( src.measure, atlas );

	const plain = layoutUnits( segmentText( 'ab c' ), atlas );
	assert.deepEqual( plain.x, [ 0, 26, 64 ], 'space advances without a quad' );
	assert.deepEqual( plain.glyphs.map( ( g ) => g.unit ), [ 'a', 'b', 'c' ] );
	assert.equal( plain.width, 90 );

	const kerned = layoutUnits( segmentText( 'AVA' ), atlas, { kerning } );
	assert.deepEqual( kerned.x, [ 0, 22, 48 ], 'A–V tightened by 4, V–A not' );
	assert.equal( kerned.width, 74 );
	const calls = src.calls.measure;
	layoutUnits( segmentText( 'AVAVAV' ), atlas, { kerning } );
	assert.equal( src.calls.measure, calls, 'pairs are measured once' );

	const spaced = layoutUnits( segmentText( 'il.' ), atlas, { letterSpacing: 3 } );
	assert.deepEqual( spaced.x, [ 0, 15, 30 ] );
	assert.equal( spaced.width, 42, 'no spacing after the last unit' );

	const withEmoji = layoutUnits( [ 'a', '\u{1F600}', 'b' ], atlas, { kerning } );
	assert.deepEqual( withEmoji.x, [ 0, 26 ], 'skipped units take no room' );
	assert.deepEqual( layoutUnits( [], atlas ), { glyphs: [], x: [], width: 0 } );
	// shaped runs are never kerned against neighbours
	assert.equal( kerning( 'مرحبا', 'a' ), 0 );

} );

// ── TextLabels: CPU side ────────────────────────────────────────────────────────────────────

/** Star (node 0 with 60 leaves) + a few extra nodes, on a ring in the z = 0 plane. */
function fakeScene( options = {}, { N = 70 } = {} ) {

	const edges = [];
	for ( let i = 1; i <= 60; i ++ ) edges.push( 0, i );
	edges.push( 1, 2, 1, 3, 2, 3 ); // leaves 1..3 get a higher degree
	const E = edges.length / 2;
	const degree = new Uint32Array( N );
	for ( const v of edges ) degree[ v ] ++;
	const adjStart = new Uint32Array( N );
	for ( let i = 1; i < N; i ++ ) adjStart[ i ] = adjStart[ i - 1 ] + degree[ i - 1 ];
	const fill = adjStart.slice(), adjList = new Uint32Array( 2 * E );
	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		adjList[ fill[ a ] ++ ] = b; adjList[ fill[ b ] ++ ] = a;

	}

	const positions = new Float32Array( 4 * N );
	for ( let i = 1; i < N; i ++ ) {

		const a = ( i / N ) * Math.PI * 2;
		positions[ 4 * i ] = Math.cos( a ) * 100;
		positions[ 4 * i + 1 ] = Math.sin( a ) * 100;

	}

	const labels = Array.from( { length: N }, ( _, i ) => 'node' + i );
	const graph = { N, E, degree, adjStart, adjList, labels, radii: new Float32Array( N ).fill( 1 ), layoutRadius: 150 };
	const camera = new PerspectiveCamera( 45, 1600 / 1000, 0.5, 20000 );
	camera.position.set( 0, 0, 400 );
	camera.lookAt( new Vector3() );
	camera.updateMatrixWorld();
	const u = ( v ) => ( { value: v } );
	const buffers = { uniforms: { selectedIndex: u( - 1 ), hoverIndex: u( - 1 ), focus: u( 0 ) }, nodePos: { value: { array: positions } } };
	const app = { width: 1600, height: 1000, camera, positions, positionsVersion: 1, modules: [] };
	const L = new TextLabels( { source: fakeSource(), ...options } );
	L.init( app );
	L.setGraph( buffers, graph, app );
	return { L, app, buffers, graph };

}

const shown = ( L ) => [ ...L._active ].filter( ( i ) => L._L[ 4 * i + 1 ] > 0 );
const kindOf = ( L, i ) => L._L[ 4 * i + 3 ];

test( 'TextLabels: selected, hovered and neighbours by degree', () => {

	const { L, app } = fakeScene( { mode: 'off', maxNeighbors: 10 } );
	L._decide( app, 0, 0, 5 );
	// mode 'off' shows nothing, not even the selection
	assert.equal( shown( L ).length, 0 );
	L.mode = 'auto';
	L.autoMaxNodes = 10; // large graph: only selection-driven labels
	L._decide( app, 0, 0, 5 );
	const s = shown( L );
	assert.equal( kindOf( L, 0 ), SEL );
	assert.equal( kindOf( L, 5 ), HOV );
	const nb = s.filter( ( i ) => kindOf( L, i ) === NB );
	assert.ok( nb.length > 0 && nb.length <= 10, 'neighbour cap' );
	for ( const i of [ 1, 2, 3 ] ) assert.ok( nb.includes( i ), `high-degree neighbour ${i} labelled` );
	assert.equal( s.filter( ( i ) => kindOf( L, i ) === CTX ).length, 0 );
	assert.equal( L.stats.labels, s.length );
	assert.equal( L.mesh.parent, null, 'the module does not put the mesh into a scene' );

} );

test( 'TextLabels: context labels by degree, culled, capped; nodes behind the camera', () => {

	const { L, app, graph } = fakeScene( { mode: 'auto' } );
	L._decide( app, 0, - 1, - 1 );
	const s = shown( L );
	assert.ok( s.length > 10 && s.length <= graph.N, `some context labels (${s.length})` );
	assert.ok( s.includes( 0 ), 'highest degree first' );
	for ( const i of s ) assert.equal( kindOf( L, i ), CTX );

	// no two shown labels overlap on screen (rects rebuilt from the same quantities)
	const m = L._viewProj( app.camera ), P11 = app.camera.projectionMatrix.elements[ 5 ];
	const rects = s.map( ( i ) => {

		const x = app.positions[ 4 * i ], y = app.positions[ 4 * i + 1 ], z = app.positions[ 4 * i + 2 ];
		const cw = m[ 3 ] * x + m[ 7 ] * y + m[ 11 ] * z + m[ 15 ];
		const sx = ( ( m[ 0 ] * x + m[ 4 ] * y + m[ 8 ] * z + m[ 12 ] ) / cw * 0.5 + 0.5 ) * app.width;
		const sy = ( 0.5 - ( m[ 1 ] * x + m[ 5 ] * y + m[ 9 ] * z + m[ 13 ] ) / cw * 0.5 ) * app.height;
		const x0 = sx + L._offset( i, CTX, cw, P11, app.height );
		return { x: x0, y: sy - 7, w: L._widthEm[ i ] * L.fontSize, h: 14 };

	} );
	for ( let i = 0; i < rects.length; i ++ ) for ( let j = 0; j < i; j ++ ) assert.ok( ! overlaps( rects[ i ], rects[ j ] ), 'labels do not overlap' );

	L.mode = 'all';
	L.maxLabels = 7;
	L._decide( app, 1, - 1, - 1 );
	assert.equal( shown( L ).length, 7, 'mode all respects maxLabels' );

	// only labels that showed have glyphs on the GPU; every label was measured at most once
	const withGlyphs = [ ...L._glyphStart ].filter( ( v ) => v >= 0 ).length;
	assert.equal( withGlyphs, s.length );
	assert.equal( L._G, s.reduce( ( n, i ) => n + L._glyphCount[ i ], 0 ) );

	app.camera.position.set( 0, 0, 0.1 );
	app.camera.lookAt( new Vector3( 0, 0, 1 ) ); // looking away from the z = 0 plane
	app.camera.updateMatrixWorld();
	L.maxLabels = 300;
	L._decide( app, 2, 0, - 1 );
	assert.equal( shown( L ).length, 0, 'nothing behind the camera, not even the selection' );

} );

test( 'TextLabels: glyph records and lazy layout', () => {

	const { L, app } = fakeScene( { mode: 'off', letterSpacing: 0 } );
	L.mode = 'auto';
	L.autoMaxNodes = 0;
	L._decide( app, 0, 65, - 1 ); // only node 65 (no neighbours) selected
	assert.deepEqual( shown( L ), [ 65 ] );
	assert.equal( L._G, 6 );
	const A = L._A, S = L.atlas.size, B = L.atlas.buffer;
	const start = L._glyphStart[ 65 ];
	let pen = 0;
	[ ...'node65' ].forEach( ( ch, k ) => {

		const g = L.atlas.glyph( ch ), o = 8 * ( start + k );
		assert.equal( A[ o ], 65, 'owner node' );
		assert.ok( Math.abs( A[ o + 1 ] - ( pen + 2 - B ) / S ) < 1e-6, 'cell left edge in em' );
		assert.ok( Math.abs( A[ o + 2 ] - g.top / S ) < 1e-6, 'cell top above the baseline in em' );
		assert.deepEqual( [ A[ o + 4 ], A[ o + 5 ], A[ o + 6 ], A[ o + 7 ] ], [ g.x, g.y, g.w, g.h ], 'atlas rect' );
		pen += g.advance;

	} );
	assert.ok( Math.abs( L._widthEm[ 65 ] - pen / S ) < 1e-6 );
	assert.equal( L.atlas.glyphs.size, 6, 'only the characters in use were rasterized' );

	// update() uploads and sets the draw count: one halo and one fill instance per glyph
	L.update( 1 / 60, app );
	assert.equal( L.mesh.count, 2 * L._G );
	assert.equal( L.uniforms.glyphCount.value, L._G );
	assert.equal( L.mesh.visible, true );
	assert.equal( L.text( 3 ), 'node3' );

} );

test( 'TextLabels: alpha end points ease like the shader and retarget from the current value', () => {

	const { L, app } = fakeScene( { mode: 'auto', fadeTime: 0.2 } );
	L._decide( app, 10, - 1, - 1 );
	const i = shown( L )[ 0 ];
	assert.deepEqual( [ ...L._L.subarray( 4 * i, 4 * i + 3 ) ], [ 0, 1, 10 ] );
	assert.equal( L.alphaOf( i, 10 ), 0 );
	assert.ok( Math.abs( L.alphaOf( i, 10.1 ) - 0.5 ) < 1e-9, 'smoothstep midpoint' );
	assert.ok( Math.abs( L.alphaOf( i, 10.05 ) - 0.15625 ) < 1e-6 );
	assert.equal( L.alphaOf( i, 11 ), 1 );

	// unchanged decisions write nothing
	L._stateRange = [ Infinity, - 1 ];
	L._decide( app, 10.05, - 1, - 1 );
	assert.equal( L._stateRange[ 1 ], - 1, 'no label state touched' );

	// hide in the middle of the fade-in: starts from the current alpha, no jump
	L.mode = 'off';
	L._decide( app, 10.1, - 1, - 1 );
	assert.deepEqual( [ ...L._L.subarray( 4 * i, 4 * i + 3 ) ].map( ( v ) => Math.round( v * 1e4 ) / 1e4 ), [ 0.5, 0, 10.1 ] );
	assert.ok( Math.abs( L.alphaOf( i, 10.1 ) - 0.5 ) < 1e-6 );
	assert.ok( L._inActive[ i ], 'still fading: stays active' );
	L._decide( app, 10.5, - 1, - 1 );
	assert.equal( L._inActive[ i ], 0, 'fully faded: dropped from the active list' );
	assert.equal( L._active.length, 0 );

	// selected labels appear fast (start at 60 %)
	L.mode = 'auto';
	L.autoMaxNodes = 0;
	L._decide( app, 20, 4, - 1 );
	assert.ok( Math.abs( L._L[ 4 * 4 ] - 0.6 ) < 1e-6 );
	// a class change alone keeps the alpha end points
	L._decide( app, 21, - 1, 4 );
	assert.equal( kindOf( L, 4 ), HOV );
	assert.equal( L._L[ 4 * 4 + 2 ], 20 );

} );

test( 'TextLabels: hysteresis keeps a shown label until it is clearly overlapped', () => {

	const { L, app } = fakeScene( { mode: 'all', hysteresis: 3 }, { N: 3 } );
	const place = ( x1, y1 ) => {

		app.positions.set( [ 0, 0, 0, 0, x1, y1, 0, 0, 500, 500, 0, 0 ] );

	};

	// world units → px at this camera: 1000 px / ( 2 · 400 · tan 22.5° ) ≈ 3.02 px per unit
	place( 0, 20 );
	L._decide( app, 0, - 1, - 1 );
	assert.deepEqual( shown( L ).sort(), [ 0, 1 ] ); // node 2 is off screen
	place( 0, 4.2 ); // 12.7 px apart vertically: boxes of 14 px overlap by ~1.3 px < hysteresis
	L._decide( app, 1, - 1, - 1 );
	assert.deepEqual( shown( L ).sort(), [ 0, 1 ], 'slight overlap tolerated for a shown label' );
	place( 0, 2 ); // 6 px apart: clearly overlapping
	L._decide( app, 2, - 1, - 1 );
	assert.deepEqual( shown( L ), [ 0 ], 'lower-priority label hidden' );
	place( 0, 4.9 ); // 14.8 px apart: free, but not by the hysteresis margin
	L._decide( app, 3, - 1, - 1 );
	assert.deepEqual( shown( L ), [ 0 ], 'a hidden label needs clear room to come back' );
	place( 0, 7 );
	L._decide( app, 4, - 1, - 1 );
	assert.deepEqual( shown( L ).sort(), [ 0, 1 ] );

} );

test( 'TextLabels: glyph buffer grows, then compacts to the active labels at maxGlyphs', () => {

	const { L, app, graph } = fakeScene( { mode: 'all', glyphCapacity: 16, maxGlyphs: 64, maxLabels: 4 } );
	L._decide( app, 0, - 1, - 1 );
	assert.equal( shown( L ).length, 4 );
	assert.ok( L._capacity >= 32 && L._capacity <= 64, `grew to ${L._capacity}` );
	assert.equal( L._A.length, 8 * L._capacity );
	const first = shown( L );

	// show other labels until the buffer has to be compacted
	const order = [ ...L._byDegree ];
	let t = 1, next = 0;
	while ( L.stats.compactions === 0 && t < 40 ) {

		// move four different nodes in front, everything else off screen
		const pos = app.positions;
		for ( let i = 0; i < graph.N; i ++ ) pos[ 4 * i ] = 1e5;
		for ( let k = 0; k < 4; k ++ ) {

			const i = order[ ( next + k ) % graph.N ];
			pos[ 4 * i ] = - 150 + 20 * k; pos[ 4 * i + 1 ] = - 60 + 40 * k;

		}

		next += 4;
		L._decide( app, t, - 1, - 1 );
		t += 1;

	}

	assert.ok( L.stats.compactions >= 1, 'compacted' );
	assert.ok( L._G <= L._capacity );
	const s = shown( L );
	assert.equal( s.length, 4 );
	// shown labels own disjoint glyph ranges with the right owner; dropped labels were reset
	const ranges = s.map( ( i ) => [ L._glyphStart[ i ], L._glyphStart[ i ] + L._glyphCount[ i ] ] ).sort( ( a, b ) => a[ 0 ] - b[ 0 ] );
	for ( let k = 1; k < ranges.length; k ++ ) assert.ok( ranges[ k ][ 0 ] >= ranges[ k - 1 ][ 1 ] );
	for ( const i of s ) for ( let g = L._glyphStart[ i ]; g < L._glyphStart[ i ] + L._glyphCount[ i ]; g ++ ) assert.equal( L._A[ 8 * g ], i );
	for ( const i of first ) if ( ! s.includes( i ) && ! L._inActive[ i ] ) assert.equal( L._glyphStart[ i ], - 1 );

} );

test( 'TextLabels: glyphs of gone labels are trimmed before they dominate the buffer', () => {

	const { L, app, graph } = fakeScene( { mode: 'all', glyphCapacity: 256, trimGlyphs: 24, maxLabels: 4 } );
	const order = [ ...L._byDegree ];
	let maxG = 0, next = 0, everShown = 0;
	for ( let t = 0; t < 30; t ++ ) {

		// four other nodes in front at every decision, everything else off screen
		const pos = app.positions;
		for ( let i = 0; i < graph.N; i ++ ) pos[ 4 * i ] = 1e5;
		for ( let k = 0; k < 4; k ++ ) {

			const i = order[ ( next + k ) % graph.N ];
			pos[ 4 * i ] = - 150 + 20 * k; pos[ 4 * i + 1 ] = - 60 + 40 * k;

		}

		next += 4;
		L._decide( app, t, - 1, - 1 );
		everShown += L.stats.labels;
		maxG = Math.max( maxG, L._G );
		// invariant after every decision: at most 4× the glyphs of the active labels (or the threshold)
		let live = 0;
		for ( const i of L._active ) if ( L._glyphStart[ i ] >= 0 ) live += L._glyphCount[ i ];
		assert.ok( L._G <= Math.max( 24, 4 * live ), `G ${L._G} within 4 × live ${live} at t = ${t}` );

	}

	assert.equal( everShown, 120 );
	assert.ok( L.stats.compactions >= 2, `trimmed ${L.stats.compactions} times` );
	assert.equal( L._capacity, 256, 'the buffer never had to grow' );
	assert.ok( maxG < 256 );
	// the labels on screen (and the ones fading out) kept their glyphs, with the right owner
	const s = shown( L );
	assert.equal( s.length, 4 );
	for ( const i of L._active ) {

		assert.ok( L._glyphStart[ i ] >= 0, `active label ${i} still has glyphs` );
		for ( let g = L._glyphStart[ i ]; g < L._glyphStart[ i ] + L._glyphCount[ i ]; g ++ ) assert.equal( L._A[ 8 * g ], i );

	}

	// a trimmed label is laid out again when it comes back
	const gone = order.find( ( i ) => ! L._inActive[ i ] && L._widthEm[ i ] > 0 && L._glyphStart[ i ] < 0 );
	assert.ok( gone !== undefined, 'some label lost its glyphs' );
	for ( let i = 0; i < graph.N; i ++ ) app.positions[ 4 * i ] = 1e5;
	app.positions[ 4 * gone ] = 50; app.positions[ 4 * gone + 1 ] = 20;
	L._decide( app, 40, - 1, - 1 );
	assert.ok( shown( L ).includes( gone ) );
	assert.ok( L._glyphStart[ gone ] >= 0 );
	assert.equal( L._A[ 8 * L._glyphStart[ gone ] ], gone );

} );

test( 'TextLabels: a decision that ran out of measure budget continues although nothing moves', () => {

	const { L, app } = fakeScene( { mode: 'all', interval: 0, measureBudget: 5 } );
	const measured = () => L._widthEm.reduce( ( n, w ) => n + ( w >= 0 ? 1 : 0 ), 0 );
	L.update( 1 / 60, app );
	assert.equal( measured(), 5, 'the first decision measures measureBudget labels' );
	assert.equal( L._starved, true );
	let guard = 0;
	while ( L._starved && guard ++ < 100 ) L.update( 1 / 60, app ); // static scene, no selection change
	assert.equal( measured(), 70, 'every candidate on screen got its turn' );
	assert.equal( L.stats.decisions, 14, '70 labels / 5 per decision; the last one turned no candidate away' );
	const shownAfter = shown( L ).length;
	assert.ok( shownAfter > 5 );
	// then it rests
	L.update( 1 / 60, app );
	L.update( 1 / 60, app );
	assert.equal( L.stats.decisions, 14 );
	// forced labels (selected, hovered) ignore the budget
	const B = fakeScene( { mode: 'auto', autoMaxNodes: 10, measureBudget: 0 } );
	B.L._decide( B.app, 0, 7, 9 );
	assert.deepEqual( shown( B.L ).sort( ( a, b ) => a - b ), [ 7, 9 ] );

} );

test( 'TextLabels: update() decides only when something changed; settings', () => {

	const { L, app, buffers } = fakeScene( { mode: 'auto', interval: 0 } );
	L.update( 1 / 60, app );
	const d0 = L.stats.decisions;
	assert.equal( d0, 1 );
	L.update( 1 / 60, app );
	L.update( 1 / 60, app );
	assert.equal( L.stats.decisions, d0, 'static scene: no decisions' );
	app.positionsVersion ++; // a readback with identical content
	L.update( 1 / 60, app );
	assert.equal( L.stats.decisions, d0, 'same positions: no decision' );
	app.positions[ 4 ] += 1;
	app.positionsVersion ++;
	L.update( 1 / 60, app );
	assert.equal( L.stats.decisions, d0 + 1, 'moved node' );
	app.camera.position.x += 5;
	app.camera.updateMatrixWorld();
	L.update( 1 / 60, app );
	assert.equal( L.stats.decisions, d0 + 2, 'moved camera' );
	buffers.uniforms.selectedIndex.value = 3;
	L.update( 1 / 60, app );
	assert.equal( L.stats.decisions, d0 + 3, 'selection' );
	assert.equal( kindOf( L, 3 ), SEL );
	L.maxLabels = 5;
	L.update( 1 / 60, app );
	assert.equal( L.stats.decisions, d0 + 4, 'setting' );

	// uniforms follow the app: CSS height, font size, node sizing fallback
	assert.equal( L.uniforms.cssHeight.value, 1000 );
	assert.equal( L.uniforms.fontSize.value, 11 );
	L.fontSize = 13;
	L.update( 1 / 60, app );
	assert.equal( L.uniforms.fontSize.value, 13 );
	assert.ok( Math.abs( L.uniforms.nodeBase.value - 0.065 * 150 / Math.cbrt( 70 ) ) < 1e-6 );
	app.modules.push( { isNodeRenderer: true, baseSize: 2, size: 1.5, minPixelRadius: 2 } );
	L.update( 1 / 60, app );
	assert.equal( L.uniforms.nodeBase.value, 3 );
	assert.equal( L.uniforms.minPx.value, 2 );
	// baseline centers the font box on the node: ( ascent − descent ) / 2
	assert.ok( Math.abs( L.uniforms.baseline.value - 0.35 ) < 1e-6 );

	// output switch: linear-light blending is compensated, display-space blending is not
	assert.equal( L.output, 'linear' );
	assert.ok( L.uniforms.blendGamma.value > 1 );
	L.output = 'srgb';
	assert.equal( L.uniforms.blendGamma.value, 1 );
	L.dispose();
	assert.equal( L.graph, null );

} );
