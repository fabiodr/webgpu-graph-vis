// Runtime SDF glyph atlas for GPU text (used by src/render/TextLabels.js). No dependencies; the
// distance transform, the packer, the atlas and the text layout are pure (typed arrays in, typed
// arrays out) and run in Node with an injected raster function (see test/labels-sdf.test.js). Only
// createCanvasRaster() touches the DOM (Canvas 2D, OffscreenCanvas where available).
//
// ─── Pipeline ───────────────────────────────────────────────────────────────────────────────
//  text ── segmentText() ──► units (grapheme clusters; emoji dropped; runs of scripts that need
//       shaping / bidi merged into one unit that the canvas shapes as a whole)
//  unit ── raster( unit ) ──► coverage bitmap + metrics (bearing, advance), measured on a canvas
//       ── sdfFromAlpha() ──► 8-bit signed distance field (Felzenszwalb–Huttenlocher EDT of the
//          anti-aliased coverage, as in mapbox/tiny-sdf: two transforms, outside and inside, with
//          sub-pixel edge offsets taken from the coverage of the edge pixels)
//       ── ShelfPacker ──► a cell in the atlas (R8, grows by doubling; cells never move, so
//          texel rectangles handed out earlier stay valid)
//  units ── layoutUnits() ──► pen positions (advance + pair kerning measured on the canvas)
//
// ─── SDF parameters (defaults, see SDF_DEFAULTS) ───────────────────────────────────────────
//  size    48   raster px per em. On-screen text is 10–40 device px, i.e. the field is minified
//               up to 4.8× (harmless: a point sample of an exact distance field is still the
//               exact distance) and magnified at most 0.83× px per texel (corners stay sharp;
//               bilinear interpolation rounds corners by ~0.2–0.3 texel, here < 0.25 px).
//               Strokes of a regular-weight UI font are ~4.5 texels wide, well above the ~2
//               texels a bilinear SDF needs, so counters ('e', 'a', accents) do not fill in.
//  radius  16   texels of distance mapped onto the 8-bit range → 1/16 texel per level
//               (≤ 0.08 device px at 40 px text): no visible banding in the edge ramp.
//  cutoff  0.25 value = 1 − (d / radius + cutoff), d > 0 outside. The outline sits at 0.75;
//               4 texels of range inside (enough for the 0.5 px anti-aliasing ramp plus weight
//               changes down to 10 px text) and 12 texels outside.
//  buffer  12   texels of padding around the glyph box = the whole outside range, so the field
//               reaches exactly 0 at the cell border: halos fade out instead of being clipped
//               by the quad, and neighbouring cells cannot bleed (no gutter needed).
//               12 texels = 0.25 em → a halo / soft shadow of up to 2.75 px at 11 px text
//               (5.5 device px at DPR 2), which is what the dark outline + glow needs.
//  Memory: a Latin glyph cell is about 54 × 62 texels; ASCII + accents (~130 glyphs) fit in a
//  1024 × 512 R8 texture (0.5 MB). The atlas starts at 512 × 256 and doubles up to 4096².
//
// ─── Units ──────────────────────────────────────────────────────────────────────────────────
//  Glyph records are in raster px (= texels): { x, y, w, h } is the cell in the atlas (with the
//  buffer), `left` the x of the cell's left edge relative to the pen position, `top` the y of the
//  cell's top edge above the baseline, `advance` the pen advance. Divide by atlas.size for em.

export const SDF_DEFAULTS = { size: 48, buffer: 12, radius: 16, cutoff: 0.25 };

const INF = 1e20;

// ── Euclidean distance transform ────────────────────────────────────────────────────────────

/**
 * 1D squared distance transform (Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled
 * Functions", 2012): replaces f(q) = grid[offset + q·stride] by min_p ( (q − p)² + f(p) ), the
 * lower envelope of the parabolas rooted at every sample, in O(length).
 * @param {Float64Array} grid in/out
 * @param {number} offset index of the first sample
 * @param {number} stride distance between samples in `grid`
 * @param {number} length number of samples
 * @param {Float64Array} f scratch, ≥ length
 * @param {Uint16Array|Int32Array} v scratch, ≥ length (parabola positions)
 * @param {Float64Array} z scratch, ≥ length + 1 (envelope boundaries)
 */
export function edt1d( grid, offset, stride, length, f, v, z ) {

	for ( let q = 0; q < length; q ++ ) f[ q ] = grid[ offset + q * stride ];

	let k = 0; // index of the rightmost parabola in the envelope
	v[ 0 ] = 0;
	z[ 0 ] = - INF;
	z[ 1 ] = INF;
	for ( let q = 1; q < length; q ++ ) {

		// intersection of the parabola from q with the rightmost one; pop the ones it hides
		// (z[ 0 ] = −INF is never reached, so k stays ≥ 0)
		let p = v[ k ];
		let s = ( ( f[ q ] - f[ p ] ) + ( q * q - p * p ) ) / ( 2 * ( q - p ) );
		while ( s <= z[ k ] ) {

			k --;
			p = v[ k ];
			s = ( ( f[ q ] - f[ p ] ) + ( q * q - p * p ) ) / ( 2 * ( q - p ) );

		}

		k ++;
		v[ k ] = q;
		z[ k ] = s;
		z[ k + 1 ] = INF;

	}

	k = 0;
	for ( let q = 0; q < length; q ++ ) {

		while ( z[ k + 1 ] < q ) k ++;
		const p = v[ k ];
		grid[ offset + q * stride ] = f[ p ] + ( q - p ) * ( q - p );

	}

}

/**
 * 2D squared Euclidean distance transform, in place, of the sub-rectangle (x0, y0, w, h) of a
 * row-major grid with `stride` columns: columns first, then rows (the transform is separable).
 */
export function edt2d( grid, x0, y0, w, h, stride, f, v, z ) {

	for ( let x = x0; x < x0 + w; x ++ ) edt1d( grid, y0 * stride + x, stride, h, f, v, z );
	for ( let y = y0; y < y0 + h; y ++ ) edt1d( grid, y * stride + x0, 1, w, f, v, z );

}

/** Reusable scratch memory for sdfFromAlpha (grows on demand). */
export function createSdfScratch() {

	return { outer: new Float64Array( 0 ), inner: new Float64Array( 0 ), f: new Float64Array( 0 ), z: new Float64Array( 0 ), v: new Int32Array( 0 ) };

}

/**
 * Signed distance field of an anti-aliased coverage bitmap.
 * @param {Uint8Array|Uint8ClampedArray} alpha coverage 0..255, row-major, width × height
 * @param {number} width
 * @param {number} height
 * @param {object} [options] { buffer, radius, cutoff } (see the header; texels)
 * @param {Uint8Array} [out] ( width + 2·buffer ) × ( height + 2·buffer ), allocated when omitted
 * @param {object} [scratch] from createSdfScratch()
 * @returns {Uint8Array} out: value/255 = 1 − ( d / radius + cutoff ), d = distance to the outline
 *   in texels, positive outside. The outline is where value/255 = 1 − cutoff.
 */
export function sdfFromAlpha( alpha, width, height, options = {}, out = null, scratch = null ) {

	const { buffer = SDF_DEFAULTS.buffer, radius = SDF_DEFAULTS.radius, cutoff = SDF_DEFAULTS.cutoff } = options;
	const W = width + 2 * buffer, H = height + 2 * buffer, len = W * H;
	const s = scratch || createSdfScratch();
	if ( s.outer.length < len ) {

		s.outer = new Float64Array( len );
		s.inner = new Float64Array( len );

	}

	const m = Math.max( W, H );
	if ( s.f.length < m ) {

		s.f = new Float64Array( m );
		s.z = new Float64Array( m + 1 );
		s.v = new Int32Array( m );

	}

	const outer = s.outer, inner = s.inner;
	// outer: squared distance to the nearest covered point; inner: to the nearest empty point.
	// Partially covered pixels seed both with their sub-pixel distance to the 50 % iso-line.
	outer.fill( INF, 0, len );
	inner.fill( 0, 0, len );
	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const a = alpha[ y * width + x ];
			if ( a === 0 ) continue;
			const j = ( y + buffer ) * W + x + buffer;
			if ( a === 255 ) {

				outer[ j ] = 0;
				inner[ j ] = INF;

			} else {

				const d = 0.5 - a / 255;
				outer[ j ] = d > 0 ? d * d : 0;
				inner[ j ] = d < 0 ? d * d : 0;

			}

		}

	}

	edt2d( outer, 0, 0, W, H, W, s.f, s.v, s.z );
	edt2d( inner, buffer, buffer, width, height, W, s.f, s.v, s.z );

	const data = out || new Uint8Array( len );
	for ( let i = 0; i < len; i ++ ) {

		const d = Math.sqrt( outer[ i ] ) - Math.sqrt( inner[ i ] );
		const val = 255 - 255 * ( d / radius + cutoff );
		data[ i ] = val <= 0 ? 0 : val >= 255 ? 255 : Math.round( val );

	}

	return data;

}

// ── packing ─────────────────────────────────────────────────────────────────────────────────

/**
 * Online shelf packer: rectangles go left to right on shelves (rows); a rectangle joins the
 * existing shelf that wastes the least height (if that is ≤ 25 % of the shelf) or opens a new one
 * whose height is rounded up to a multiple of 4, so glyphs of similar height share shelves.
 * Positions never change afterwards; resize() only adds room on the right and at the bottom.
 */
export class ShelfPacker {

	constructor( width, height, gap = 0 ) {

		this.width = width;
		this.height = height;
		this.gap = gap;
		this.shelves = []; // { y, h, x }
		this.bottom = 0;
		this.used = 0; // texels handed out

	}

	resize( width, height ) {

		this.width = Math.max( this.width, width );
		this.height = Math.max( this.height, height );

	}

	/** @returns {{x: number, y: number}|null} null when the rectangle does not fit (grow and retry). */
	add( w, h ) {

		const gap = this.gap;
		if ( w > this.width || h > this.height ) return null;
		let best = null, bestWaste = Infinity;
		for ( let k = 0; k < this.shelves.length; k ++ ) {

			const s = this.shelves[ k ];
			if ( h > s.h || s.x + w > this.width ) continue;
			const waste = s.h - h;
			if ( waste < bestWaste ) {

				best = s;
				bestWaste = waste;

			}

		}

		if ( best === null || bestWaste > 0.25 * best.h ) {

			const sh = Math.min( Math.ceil( h / 4 ) * 4, this.height - this.bottom );
			if ( sh >= h ) {

				best = { y: this.bottom, h: sh, x: 0 };
				this.shelves.push( best );
				this.bottom += sh + gap;

			}

		}

		if ( best === null ) return null;
		const at = { x: best.x, y: best.y };
		best.x += w + gap;
		this.used += w * h;
		return at;

	}

}

// ── atlas ───────────────────────────────────────────────────────────────────────────────────

/**
 * Dynamic single-channel SDF atlas. Glyphs are rasterized on demand through the injected
 * `raster( unit )`, which returns
 *   { width, height, alpha, left, top, advance }   coverage bitmap (Uint8, 0..255, row-major),
 *       `left`: x of the bitmap's left edge relative to the pen, `top`: y of its top edge above
 *       the baseline, `advance`: pen advance — all in raster px at `size` px per em;
 *       width = 0 (or alpha = null) for blank glyphs (spaces);
 *   or { skip: true, advance } for something that cannot be drawn as a monochrome glyph (color
 *       emoji): the unit then takes no room at all.
 */
export class GlyphAtlas {

	constructor( { raster, size = SDF_DEFAULTS.size, buffer = SDF_DEFAULTS.buffer, radius = SDF_DEFAULTS.radius, cutoff = SDF_DEFAULTS.cutoff, width = 512, height = 256, maxSize = 4096 } = {} ) {

		if ( typeof raster !== 'function' ) throw new Error( 'GlyphAtlas: a raster( unit ) function is required' );
		this.raster = raster;
		this.size = size;
		this.buffer = buffer;
		this.radius = radius;
		this.cutoff = cutoff;
		this.width = width;
		this.height = height;
		this.maxSize = maxSize;
		this.data = new Uint8Array( width * height );
		this.packer = new ShelfPacker( width, height );
		this.glyphs = new Map();
		/** Bumped whenever texels change (re-upload the texture). */
		this.version = 0;
		/** Bumped whenever `data` is reallocated with a new width / height (new texture). */
		this.sizeVersion = 0;
		/** Units dropped because the atlas reached maxSize, or that the raster asked to skip. */
		this.missing = 0;
		this.skipped = 0;
		this._scratch = createSdfScratch();
		this._cell = new Uint8Array( 0 );

	}

	/** Field value (0..1) at the outline. */
	get edge() {

		return 1 - this.cutoff;

	}

	/** Returns (and creates on first use) the glyph record of a unit; never null. */
	glyph( unit ) {

		let g = this.glyphs.get( unit );
		if ( g === undefined ) {

			g = this._create( unit );
			this.glyphs.set( unit, g );

		}

		return g;

	}

	has( unit ) {

		return this.glyphs.has( unit );

	}

	_create( unit ) {

		const r = this.raster( unit ) || { skip: true, advance: 0 };
		const g = { unit, x: 0, y: 0, w: 0, h: 0, left: 0, top: 0, advance: r.advance || 0, skip: false, missing: false };
		if ( r.skip ) {

			g.skip = true;
			g.advance = 0;
			this.skipped ++;
			return g;

		}

		if ( ! r.alpha || ! ( r.width > 0 ) || ! ( r.height > 0 ) ) return g; // blank (space)

		const B = this.buffer;
		const w = r.width + 2 * B, h = r.height + 2 * B;
		let at = this.packer.add( w, h );
		while ( at === null ) {

			if ( ! this._grow( w, h ) ) {

				g.missing = true;
				this.missing ++;
				return g;

			}

			at = this.packer.add( w, h );

		}

		if ( this._cell.length < w * h ) this._cell = new Uint8Array( w * h );
		const cell = sdfFromAlpha( r.alpha, r.width, r.height, this, this._cell, this._scratch );
		const W = this.width, data = this.data;
		for ( let y = 0; y < h; y ++ ) data.set( cell.subarray( y * w, y * w + w ), ( at.y + y ) * W + at.x );
		g.x = at.x; g.y = at.y; g.w = w; g.h = h;
		g.left = r.left - B;
		g.top = r.top + B;
		this.version ++;
		return g;

	}

	/** Doubles the smaller dimension (cells keep their positions). False at maxSize. */
	_grow( needW, needH ) {

		let W = this.width, H = this.height;
		if ( needW > this.maxSize || needH > this.maxSize ) return false;
		if ( ( W <= H || needW > W ) && W < this.maxSize ) W *= 2;
		else if ( H < this.maxSize ) H *= 2;
		else if ( W < this.maxSize ) W *= 2;
		else return false;
		const data = new Uint8Array( W * H );
		for ( let y = 0; y < this.height; y ++ ) data.set( this.data.subarray( y * this.width, ( y + 1 ) * this.width ), y * W );
		this.data = data;
		this.width = W;
		this.height = H;
		this.packer.resize( W, H );
		this.sizeVersion ++;
		this.version ++;
		return true;

	}

}

// ── text segmentation ───────────────────────────────────────────────────────────────────────

// default-emoji-presentation characters, flags, keycaps, and anything forced to emoji by VS16 /
// built with ZWJ: these are color glyphs, which a single-channel SDF cannot represent
const EMOJI = /[\p{Emoji_Presentation}\p{Regional_Indicator}️⃣‍]/u;
// scripts with contextual shaping, reordering or right-to-left runs: per-grapheme quads would
// break joining forms and visual order, so a whole run becomes one unit (the canvas shapes it)
const COMPLEX = /[\p{Script=Arabic}\p{Script=Hebrew}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Nko}\p{Script=Devanagari}\p{Script=Bengali}\p{Script=Gurmukhi}\p{Script=Gujarati}\p{Script=Oriya}\p{Script=Tamil}\p{Script=Telugu}\p{Script=Kannada}\p{Script=Malayalam}\p{Script=Sinhala}\p{Script=Khmer}\p{Script=Myanmar}\p{Script=Tibetan}\p{Script=Mongolian}]/u;
const INVISIBLE = /^[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]+$/u;
const SPACE = /^\s+$/u;
const MARK = /^\p{M}+$/u;

let _segmenter;

/** Grapheme clusters of a string (Intl.Segmenter when available, else code points + marks). */
export function graphemes( text ) {

	if ( _segmenter === undefined ) {

		_segmenter = typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function' ? new Intl.Segmenter( undefined, { granularity: 'grapheme' } ) : null;

	}

	const out = [];
	if ( _segmenter ) {

		for ( const s of _segmenter.segment( text ) ) out.push( s.segment );
		return out;

	}

	for ( const ch of text ) {

		if ( out.length && MARK.test( ch ) ) out[ out.length - 1 ] += ch;
		else out.push( ch );

	}

	return out;

}

/**
 * Splits a label into the units the atlas rasterizes: one grapheme cluster each, except that
 *  - emoji and invisible (control / format) clusters are dropped,
 *  - all whitespace becomes a plain space,
 *  - a run of clusters from a script that needs shaping (Arabic, Hebrew, Indic, …; inner spaces
 *    included) is one unit, so the canvas applies joining forms and bidi inside it,
 *  - more than `maxUnits` units are cut and end with an ellipsis.
 * @returns {string[]}
 */
export function segmentText( text, { maxUnits = 64, ellipsis = '…' } = {} ) {

	const gs = graphemes( String( text ) );
	const units = [];
	let run = '', pendingSpace = '', dropped = false;
	const flush = () => {

		if ( run ) units.push( run );
		run = '';
		if ( pendingSpace ) units.push( ' ' );
		pendingSpace = '';

	};

	for ( let k = 0; k < gs.length; k ++ ) {

		const g = gs[ k ];
		if ( COMPLEX.test( g ) ) {

			run += pendingSpace + g;
			pendingSpace = '';
			continue;

		}

		const space = SPACE.test( g );
		if ( ! space && ( EMOJI.test( g ) || INVISIBLE.test( g ) ) ) {

			dropped = true;
			continue;

		}

		// "a 😀 b": the spaces around something dropped collapse into one
		if ( space && dropped && ! run && ! pendingSpace && ( units.length === 0 || units[ units.length - 1 ] === ' ' ) ) continue;
		dropped = false;

		if ( space && run ) {

			pendingSpace = ' '; // joins the run only if more of the script follows
			continue;

		}

		flush();
		units.push( space ? ' ' : g );

	}

	flush();
	if ( units.length > maxUnits ) {

		units.length = Math.max( 0, maxUnits - 1 );
		while ( units.length && units[ units.length - 1 ] === ' ' ) units.pop();
		units.push( ellipsis );

	}

	return units;

}

// ── layout ──────────────────────────────────────────────────────────────────────────────────

/**
 * Pair kerning from string measurements: kern( a, b ) = width( a + b ) − advance( a ) − advance( b ),
 * cached per pair. Only for simple units (one code point, or with a combining mark); shaped runs
 * are never kerned against their neighbours.
 * @param {(s: string) => number} measure advance width of a string in raster px
 * @param {GlyphAtlas} atlas
 * @param {number} [limit] cache size bound
 */
export function makeKerning( measure, atlas, limit = 20000 ) {

	const cache = new Map();
	const eps = atlas.size / 512;
	return ( a, b ) => {

		if ( a.length > 2 || b.length > 2 ) return 0;
		const key = a + '\u0001' + b;
		let k = cache.get( key );
		if ( k === undefined ) {

			k = measure( a + b ) - atlas.glyph( a ).advance - atlas.glyph( b ).advance;
			if ( ! ( Math.abs( k ) > eps ) || Math.abs( k ) > atlas.size ) k = 0;
			if ( cache.size >= limit ) cache.clear();
			cache.set( key, k );

		}

		return k;

	};

}

/**
 * Lays units out on one line, left to right from pen = 0.
 * @param {string[]} units from segmentText()
 * @param {GlyphAtlas} atlas
 * @param {object} [options] { kerning: ( a, b ) => px, letterSpacing: px } (raster px)
 * @returns {{ glyphs: object[], x: number[], width: number }} `glyphs[k]` is the atlas record of
 *   the k-th visible glyph and `x[k]` its pen position; `width` is the advance width of the line.
 */
export function layoutUnits( units, atlas, { kerning = null, letterSpacing = 0 } = {} ) {

	const glyphs = [], x = [];
	let pen = 0, prev = null;
	for ( let k = 0; k < units.length; k ++ ) {

		const u = units[ k ];
		const g = atlas.glyph( u );
		if ( g.skip ) continue;
		if ( prev !== null ) pen += letterSpacing + ( kerning ? kerning( prev, u ) : 0 );
		if ( g.w > 0 ) {

			glyphs.push( g );
			x.push( pen );

		}

		pen += g.advance;
		prev = u;

	}

	return { glyphs, x, width: pen };

}

// ── canvas raster (browser) ─────────────────────────────────────────────────────────────────

export const SYSTEM_UI_FONT = 'system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';

/**
 * Canvas 2D glyph source for GlyphAtlas.
 * @returns {{ raster: Function, measure: Function, metrics: object, font: string }}
 *   raster( unit )  see GlyphAtlas; measure( string ) advance width in raster px;
 *   metrics { ascent, descent, capHeight, xHeight } in em (font box and the heights of H and x).
 */
export function createCanvasRaster( { fontFamily = SYSTEM_UI_FONT, fontWeight = 500, fontStyle = 'normal', size = SDF_DEFAULTS.size, maxWidth = 1024 } = {} ) {

	const font = `${fontStyle} ${fontWeight} ${size}px ${fontFamily}`;
	let cw = 2 * size, ch = 2 * size;
	const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( cw, ch ) : Object.assign( document.createElement( 'canvas' ), { width: cw, height: ch } );
	const ctx = canvas.getContext( '2d', { willReadFrequently: true } );
	const setup = () => {

		ctx.font = font;
		ctx.textBaseline = 'alphabetic';
		ctx.textAlign = 'left';
		ctx.fillStyle = '#000';
		// outlines as designed: no grid fitting (it would bake one pixel size into the field)
		if ( 'textRendering' in ctx ) ctx.textRendering = 'geometricPrecision';
		if ( 'fontKerning' in ctx ) ctx.fontKerning = 'normal';

	};

	setup();

	const em = ( v, fallback ) => ( Number.isFinite( v ) && v > 0 ? v : fallback ) / size;
	const mH = ctx.measureText( 'H' ), mx = ctx.measureText( 'x' );
	const metrics = {
		ascent: em( mH.fontBoundingBoxAscent, 0.93 * size ),
		descent: em( mH.fontBoundingBoxDescent, 0.24 * size ),
		capHeight: em( mH.actualBoundingBoxAscent, 0.72 * size ),
		xHeight: em( mx.actualBoundingBoxAscent, 0.52 * size ),
	};

	const measure = ( s ) => ctx.measureText( s ).width;

	const raster = ( unit ) => {

		const m = ctx.measureText( unit );
		const advance = m.width;
		// integer bitmap box around the ink, one pixel of slack for anti-aliased edges
		const left = Math.floor( - m.actualBoundingBoxLeft ) - 1;
		const right = Math.ceil( m.actualBoundingBoxRight ) + 1;
		const top = Math.ceil( m.actualBoundingBoxAscent ) + 1;
		const bottom = Math.ceil( m.actualBoundingBoxDescent ) + 1;
		const w = right - left, h = top + bottom;
		if ( ! Number.isFinite( w ) || ! Number.isFinite( h ) || w <= 2 || h <= 2 ) return { width: 0, height: 0, alpha: null, left: 0, top: 0, advance };
		if ( w > maxWidth || h > 4 * size ) return { skip: true, advance: 0 };
		if ( w > cw || h > ch ) {

			cw = Math.max( cw, w );
			ch = Math.max( ch, h );
			canvas.width = cw;
			canvas.height = ch;
			setup(); // resizing a canvas resets its state

		}

		ctx.clearRect( 0, 0, w, h );
		ctx.fillText( unit, - left, top );
		const rgba = ctx.getImageData( 0, 0, w, h ).data;
		const alpha = new Uint8Array( w * h );
		let ink = 0, colored = 0;
		for ( let i = 0, j = 0; i < alpha.length; i ++, j += 4 ) {

			const a = rgba[ j + 3 ];
			alpha[ i ] = a;
			if ( a > 96 ) {

				ink ++;
				// drawn in black: any color means a color font (emoji) painted its own pixels
				if ( rgba[ j ] + rgba[ j + 1 ] + rgba[ j + 2 ] > 96 ) colored ++;

			}

		}

		if ( ink === 0 ) return { width: 0, height: 0, alpha: null, left: 0, top: 0, advance };
		if ( colored > 0.1 * ink ) return { skip: true, advance: 0 };
		return { width: w, height: h, alpha, left, top, advance };

	};

	return { raster, measure, metrics, font };

}
