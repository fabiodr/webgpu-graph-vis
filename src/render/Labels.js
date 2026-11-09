// Node labels: an HTML overlay of pooled, absolutely positioned <div>s moved with CSS transforms
// (no layout reads, no reflow; only transform/opacity/class writes, and only when they change).
// Positions come from the CPU readback `app.positions` (may lag the GPU by a frame) projected with
// `app.camera` in afterRender(), i.e. with the matrices of the frame just drawn.
//
// What is shown (in priority order):
//   1. the selected node (buffers.uniforms.selectedIndex), emphasized pill
//   2. the hovered node (buffers.uniforms.hoverIndex)
//   3. neighbors of the selected node, by degree, at most `maxNeighbors` (graph.adjStart/degree/adjList)
//   4. context labels: mode 'auto' → all nodes when N <= `autoMaxNodes`; mode 'all' → always;
//      in degree order, until `maxLabels` labels are placed. Context labels fade back while a
//      selection exists (uniforms.focus).
//  Everything except 1–2 goes through greedy screen-space overlap culling on a coarse grid
//  (LabelGrid), so more important labels win; the selected/hovered node disc (with its halo) is
//  an obstacle too, so no other label covers it. Labels fade with depth (relative to the graph's
//  bounding sphere) and fade in/out smoothly instead of popping (eased per label in JS).
//
// Labels sit to the right of their node, offset by the node's on-screen radius (from the
// NodeRenderer module when present). Text is crisp: transforms are snapped to device pixels.
//
// Public API:
//   const labels = app.use( new Labels( { mode: 'auto', maxLabels: 160 } ) );   // order 20
//   labels.mode = 'auto' | 'all' | 'off';
//   labels.maxLabels = 200;  labels.maxNeighbors = 40;  labels.autoMaxNodes = 150;
//   labels.setColors( colors4 )   // after a palette change (pill border colors)
//   labels.setGraph( buffers, graph, app ), labels.dispose()
//   LabelGrid: the culling grid (exported for tests).

import { linearToSrgb } from '../util/color.js';

const STYLE_ID = 'wgv-labels-style';
const CSS = `
.wgv-labels{position:absolute;inset:0;overflow:hidden;pointer-events:none;contain:strict;z-index:1;
	font:500 11px system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
	-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;text-rendering:optimizeLegibility}
.wgv-label{position:absolute;left:0;top:0;box-sizing:border-box;height:14px;line-height:14px;white-space:nowrap;
	will-change:transform,opacity;opacity:0;color:rgba(236,242,255,.92);letter-spacing:.01em;padding:0 2px;
	text-shadow:0 0 2px rgba(0,0,0,.95),0 1px 3px rgba(0,0,0,.85),0 0 8px rgba(0,0,0,.6)}
.wgv-label.ctx{color:rgba(214,224,244,.84);font-weight:400}
.wgv-label.nb{color:#fff}
.wgv-label.hov,.wgv-label.sel{color:#fff;font-weight:600;height:20px;line-height:18px;padding:0 8px;border-radius:999px;
	background:rgba(6,10,20,.62);border:1px solid var(--c,rgba(255,255,255,.5));
	box-shadow:0 0 12px -3px var(--c,transparent),inset 0 0 6px -3px var(--c,transparent);text-shadow:none}
.wgv-label.sel{font-size:12.5px;letter-spacing:.015em}
.wgv-label.hov{background:rgba(6,10,20,.45)}
@media (pointer:coarse){.wgv-labels{font-size:12px}.wgv-label{height:16px;line-height:16px}
	.wgv-label.hov,.wgv-label.sel{height:22px;line-height:20px}.wgv-label.sel{font-size:13.5px}}
`;

/** Greedy overlap culling on a uniform grid of screen cells (CSS px). */
export class LabelGrid {

	constructor( cellW = 96, cellH = 24 ) {

		this.cellW = cellW;
		this.cellH = cellH;
		this.cols = 0;
		this.rows = 0;
		this.cells = [];
		this.rects = new Float32Array( 4 * 256 );
		this.count = 0;

	}

	reset( width, height ) {

		const cols = Math.max( 1, Math.ceil( width / this.cellW ) );
		const rows = Math.max( 1, Math.ceil( height / this.cellH ) );
		if ( cols * rows !== this.cells.length ) {

			this.cells = new Array( cols * rows );
			for ( let k = 0; k < this.cells.length; k ++ ) this.cells[ k ] = [];

		} else {

			for ( let k = 0; k < this.cells.length; k ++ ) this.cells[ k ].length = 0;

		}

		this.cols = cols;
		this.rows = rows;
		this.count = 0;

	}

	_range( x0, y0, x1, y1 ) {

		const c0 = Math.max( 0, Math.floor( x0 / this.cellW ) ), c1 = Math.min( this.cols - 1, Math.floor( x1 / this.cellW ) );
		const r0 = Math.max( 0, Math.floor( y0 / this.cellH ) ), r1 = Math.min( this.rows - 1, Math.floor( y1 / this.cellH ) );
		return [ c0, c1, r0, r1 ];

	}

	/** True when the rect overlaps nothing inserted so far. */
	free( x0, y0, x1, y1 ) {

		const [ c0, c1, r0, r1 ] = this._range( x0, y0, x1, y1 );
		const R = this.rects;
		for ( let r = r0; r <= r1; r ++ ) {

			for ( let c = c0; c <= c1; c ++ ) {

				const cell = this.cells[ r * this.cols + c ];
				for ( let k = 0; k < cell.length; k ++ ) {

					const j = 4 * cell[ k ];
					if ( x0 < R[ j + 2 ] && x1 > R[ j ] && y0 < R[ j + 3 ] && y1 > R[ j + 1 ] ) return false;

				}

			}

		}

		return true;

	}

	insert( x0, y0, x1, y1 ) {

		if ( 4 * ( this.count + 1 ) > this.rects.length ) {

			const grown = new Float32Array( this.rects.length * 2 );
			grown.set( this.rects );
			this.rects = grown;

		}

		const id = this.count ++;
		this.rects.set( [ x0, y0, x1, y1 ], 4 * id );
		const [ c0, c1, r0, r1 ] = this._range( x0, y0, x1, y1 );
		for ( let r = r0; r <= r1; r ++ ) for ( let c = c0; c <= c1; c ++ ) this.cells[ r * this.cols + c ].push( id );
		return id;

	}

	/** Insert if free (or forced). Returns whether the rect was placed. */
	tryInsert( x0, y0, x1, y1, force = false ) {

		if ( ! force && ! this.free( x0, y0, x1, y1 ) ) return false;
		this.insert( x0, y0, x1, y1 );
		return true;

	}

}

const KIND_CLASS = [ 'wgv-label ctx', 'wgv-label nb', 'wgv-label hov', 'wgv-label sel' ];
const CTX = 0, NB = 1, HOV = 2, SEL = 3;

export class Labels {

	constructor( { mode = 'auto', maxLabels = 160, maxNeighbors = 40, autoMaxNodes = 150, container = null } = {} ) {

		this.order = 20;
		this.mode = mode;
		this.maxLabels = maxLabels;
		this.maxNeighbors = maxNeighbors;
		this.autoMaxNodes = autoMaxNodes;
		this.container = container;
		this.app = null;
		this.graph = null;
		this.buffers = null;
		this.root = null;

		this._active = new Map(); // node -> pooled entry {el, alpha, target, kind, x, y, tx, op, cls}
		this._pool = [];
		this._grid = new LabelGrid();
		this._byDegree = null;
		this._widths = null;
		this._texts = null;
		this._cssColors = null;
		this._neighbors = [];
		this._neighborsOf = - 2;
		this._wanted = new Map(); // node -> kind, rebuilt every frame
		this._bounds = { x: 0, y: 0, z: 0, r: 1, version: - 1, t: 0 };
		this._measure = null;
		this._lastTime = 0;

	}

	init( app ) {

		this.app = app;
		if ( typeof document === 'undefined' ) return;
		if ( ! document.getElementById( STYLE_ID ) ) {

			const style = document.createElement( 'style' );
			style.id = STYLE_ID;
			style.textContent = CSS;
			document.head.appendChild( style );

		}

		const parent = this.container || app.container || document.body;
		this.root = document.createElement( 'div' );
		this.root.className = 'wgv-labels';
		parent.appendChild( this.root );
		const canvas = document.createElement( 'canvas' );
		this._measure = canvas.getContext( '2d' );

	}

	setGraph( buffers, graph ) {

		this.buffers = buffers;
		this.graph = graph;
		for ( const [ , e ] of this._active ) this._release( e );
		this._active.clear();
		const N = graph.N;
		const order = new Uint32Array( N );
		for ( let i = 0; i < N; i ++ ) order[ i ] = i;
		order.sort( ( a, b ) => graph.degree[ b ] - graph.degree[ a ] || a - b );
		this._byDegree = order;
		this._widths = new Float32Array( N ).fill( - 1 );
		this._texts = graph.labels || null;
		this._cssColors = new Array( N );
		this._colors = null;
		this._neighborsOf = - 2;
		this._bounds.version = - 1;

	}

	/** Call after a palette change (linear RGBA, 4 per node) so pill borders follow the nodes. */
	setColors( colors ) {

		this._colors = colors;
		if ( this._cssColors ) this._cssColors.fill( undefined );
		for ( const [ i, e ] of this._active ) if ( e.kind >= HOV ) e.el.style.setProperty( '--c', this._cssColor( i ) );

	}

	text( i ) {

		return this._texts ? this._texts[ i ] : String( i );

	}

	// ── per frame ───────────────────────────────────────────────────────────────────────────

	afterRender( app ) {

		if ( ! this.root || ! this.graph || ! app.positions ) return;
		const now = performance.now();
		const dt = this._lastTime ? Math.min( 0.1, ( now - this._lastTime ) / 1000 ) : 1 / 60;
		this._lastTime = now;

		const U = this.buffers.uniforms;
		const sel = U.selectedIndex.value | 0;
		const hov = U.hoverIndex.value | 0;
		const focus = U.focus.value;
		const wanted = this._wanted;
		if ( this.mode !== 'off' ) this._collect( sel, hov, focus, app );
		else wanted.clear();

		// ease and write
		const ease = 1 - Math.exp( - dt * 10 );
		for ( const [ i, e ] of this._active ) {

			const w = wanted.get( i );
			e.target = w ? w.alpha : 0;
			if ( w ) {

				e.x = w.x; e.y = w.y;
				if ( w.kind !== e.kind ) this._setKind( e, i, w.kind );

			} else {

				// keep fading labels attached to their node
				this._projectEntry( i, e, app );

			}

			e.alpha += ( e.target - e.alpha ) * ease;
			if ( e.target === 0 && e.alpha < 0.01 ) {

				this._release( e );
				this._active.delete( i );
				continue;

			}

			this._write( e );

		}

		for ( const [ i, w ] of wanted ) {

			if ( this._active.has( i ) ) continue;
			const e = this._acquire( i, w.kind );
			e.x = w.x; e.y = w.y; e.target = w.alpha;
			e.alpha = w.kind >= HOV ? Math.min( 1, w.alpha ) * 0.6 : 0; // hovered/selected appear fast
			this._active.set( i, e );
			this._write( e );

		}

	}

	_collect( sel, hov, focus, app ) {

		const g = this.graph;
		const N = g.N;
		const pos = app.positions;
		const cam = app.camera;
		const W = app.width, H = app.height;
		const m = this._viewProj( cam );
		const P11 = cam.projectionMatrix.elements[ 5 ];
		const nodes = this._nodeRenderer( app );
		const grid = this._grid;
		grid.reset( W, H );
		this._updateBounds( app );
		const b = this._bounds;
		// depth range of the graph as seen from the camera
		const cx = cam.position.x - b.x, cy = cam.position.y - b.y, cz = cam.position.z - b.z;
		const dc = Math.hypot( cx, cy, cz );
		const near = Math.max( 1e-3, dc - b.r ), far = dc + b.r;
		const fontPx = this._fontPx();
		const lineH = this._lineH( CTX );
		const wanted = this._wanted;
		wanted.clear();
		let placed = 0;

		const place = ( i, kind, force, alphaScale ) => {

			if ( wanted.has( i ) ) return false;
			const x = pos[ 4 * i ], y = pos[ 4 * i + 1 ], z = pos[ 4 * i + 2 ];
			const cw = m[ 3 ] * x + m[ 7 ] * y + m[ 11 ] * z + m[ 15 ];
			if ( cw <= cam.near ) return false;
			const sx = ( ( m[ 0 ] * x + m[ 4 ] * y + m[ 8 ] * z + m[ 12 ] ) / cw * 0.5 + 0.5 ) * W;
			const sy = ( 0.5 - ( m[ 1 ] * x + m[ 5 ] * y + m[ 9 ] * z + m[ 13 ] ) / cw * 0.5 ) * H;
			if ( sx < - 40 || sy < - 20 || sx > W + 4 || sy > H + 20 ) return false;
			const off = this._offset( i, kind, cw, P11, H, nodes );
			const lx = sx + off;
			const wPx = this._width( i, fontPx ) * ( kind === SEL ? 1.15 : 1 ) + ( kind >= HOV ? 16 : 4 );
			const hh = kind >= HOV ? this._lineH( kind ) : lineH;
			const x0 = lx, x1 = lx + wPx, y0 = sy - hh * 0.5, y1 = sy + hh * 0.5;
			if ( ! grid.tryInsert( x0 - 1, y0, x1 + 1, y1, force ) ) return false;
			// keep other labels off the selected / hovered node and its halo
			if ( kind >= HOV ) {

				const ring = off - 5;
				grid.insert( sx - ring, sy - ring, sx + ring, sy + ring );

			}

			// depth fade: nearest part of the graph 1, farthest 0.35
			const t = Math.min( 1, Math.max( 0, ( cw - near ) / Math.max( 1e-3, far - near ) ) );
			const alpha = ( kind >= HOV ? 1 : 1 - 0.65 * t ) * alphaScale;
			wanted.set( i, { kind, x: lx, y: sy, alpha } );
			placed ++;
			return true;

		};

		if ( sel >= 0 && sel < N ) place( sel, SEL, true, 1 );
		if ( hov >= 0 && hov < N && hov !== sel ) place( hov, HOV, true, 1 );

		if ( sel >= 0 && sel < N ) {

			const nb = this._neighborsSorted( sel );
			for ( let k = 0; k < nb.length; k ++ ) place( nb[ k ], NB, false, 1 );

		}

		const showAll = this.mode === 'all' || ( this.mode === 'auto' && N <= this.autoMaxNodes );
		if ( showAll ) {

			const ctxAlpha = 1 - 0.8 * Math.min( 1, Math.max( 0, focus ) );
			if ( ctxAlpha > 0.02 ) {

				const order = this._byDegree;
				const cap = this.maxLabels;
				for ( let k = 0; k < N && placed < cap; k ++ ) place( order[ k ], CTX, false, ctxAlpha );

			}

		}

	}

	_neighborsSorted( sel ) {

		if ( this._neighborsOf === sel ) return this._neighbors;
		const g = this.graph;
		const s = g.adjStart[ sel ], n = g.degree[ sel ];
		const list = Array.from( g.adjList.subarray( s, s + n ) );
		list.sort( ( a, b ) => g.degree[ b ] - g.degree[ a ] || a - b );
		this._neighbors = list.slice( 0, this.maxNeighbors );
		this._neighborsOf = sel;
		return this._neighbors;

	}

	_viewProj( cam ) {

		// column-major P · V
		const p = cam.projectionMatrix.elements, v = cam.matrixWorldInverse.elements;
		const out = this._m || ( this._m = new Float64Array( 16 ) );
		for ( let c = 0; c < 4; c ++ ) {

			for ( let r = 0; r < 4; r ++ ) {

				out[ c * 4 + r ] = p[ r ] * v[ c * 4 ] + p[ 4 + r ] * v[ c * 4 + 1 ] + p[ 8 + r ] * v[ c * 4 + 2 ] + p[ 12 + r ] * v[ c * 4 + 3 ];

			}

		}

		return out;

	}

	_projectEntry( i, e, app ) {

		const pos = app.positions, m = this._m;
		if ( ! m ) return;
		const x = pos[ 4 * i ], y = pos[ 4 * i + 1 ], z = pos[ 4 * i + 2 ];
		const cw = m[ 3 ] * x + m[ 7 ] * y + m[ 11 ] * z + m[ 15 ];
		if ( cw <= app.camera.near ) {

			e.target = 0; e.alpha = 0;
			return;

		}

		const nodes = this._nodeRenderer( app );
		const sx = ( ( m[ 0 ] * x + m[ 4 ] * y + m[ 8 ] * z + m[ 12 ] ) / cw * 0.5 + 0.5 ) * app.width;
		const sy = ( 0.5 - ( m[ 1 ] * x + m[ 5 ] * y + m[ 9 ] * z + m[ 13 ] ) / cw * 0.5 ) * app.height;
		e.x = sx + this._offset( i, e.kind, cw, app.camera.projectionMatrix.elements[ 5 ], app.height, nodes );
		e.y = sy;

	}

	/**
	 * Horizontal distance (CSS px) from the node center to the label: the node's on-screen radius
	 * (with the renderer's selection/hover growth) or, for kinds that get a halo ring (selected,
	 * hovered, neighbors of the selection), the ring's outer edge; plus a small gap.
	 */
	_offset( i, kind, cw, P11, H, nodes ) {

		const emph = kind === SEL ? 1.5 : kind === HOV ? 1.25 : 1;
		const rWorld = ( nodes ? nodes.worldRadius( i ) : 3 ) * emph;
		const rPx = Math.max( nodes ? nodes.minPixelRadius : 1.5, ( rWorld * P11 * H * 0.5 ) / cw );
		if ( kind === CTX ) return rPx + 3;
		const ring = nodes && nodes.haloRadius ? nodes.haloRadius( rPx, kind === SEL ? 1 : kind === HOV ? 0.5 : 0 ) : rPx * 1.3 + 4;
		return ring + ( kind >= HOV ? 5 : 3 );

	}

	_updateBounds( app ) {

		const b = this._bounds;
		const now = performance.now();
		if ( b.version === app.positionsVersion || ( b.version >= 0 && now - b.t < 250 ) ) return;
		const p = app.positions, n = this.graph.N;
		let x = 0, y = 0, z = 0;
		for ( let i = 0; i < n; i ++ ) {

			x += p[ 4 * i ]; y += p[ 4 * i + 1 ]; z += p[ 4 * i + 2 ];

		}

		x /= n; y /= n; z /= n;
		let s = 0;
		for ( let i = 0; i < n; i ++ ) s += ( p[ 4 * i ] - x ) ** 2 + ( p[ 4 * i + 1 ] - y ) ** 2 + ( p[ 4 * i + 2 ] - z ) ** 2;
		b.x = x; b.y = y; b.z = z;
		b.r = Math.max( 1, 1.6 * Math.sqrt( s / n ) );
		b.version = app.positionsVersion;
		b.t = now;

	}

	_nodeRenderer( app ) {

		if ( this._nodes && app.modules.includes( this._nodes ) ) return this._nodes;
		this._nodes = app.modules.find( ( m ) => m && m.isNodeRenderer ) || null;
		return this._nodes;

	}

	_fontPx() {

		if ( this._font === undefined ) {

			this._font = typeof matchMedia !== 'undefined' && matchMedia( '(pointer:coarse)' ).matches ? 12 : 11;

		}

		return this._font;

	}

	/** Box height (CSS px) of a label of the given kind; must match the CSS above. */
	_lineH( kind ) {

		const coarse = this._fontPx() === 12;
		return kind >= HOV ? ( coarse ? 22 : 20 ) : ( coarse ? 16 : 14 );

	}

	_width( i, fontPx ) {

		let w = this._widths[ i ];
		if ( w >= 0 ) return w;
		const t = this.text( i );
		if ( this._measure ) {

			this._measure.font = `500 ${fontPx}px system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif`;
			w = this._measure.measureText( t ).width + 4;

		} else {

			w = t.length * fontPx * 0.56 + 4;

		}

		this._widths[ i ] = w;
		return w;

	}

	_cssColor( i ) {

		let c = this._cssColors[ i ];
		if ( c ) return c;
		const col = this._colors || this.graph.colors;
		const to = ( v ) => Math.round( 255 * Math.min( 1, Math.max( 0, linearToSrgb( v ) ) ) );
		c = `rgba(${to( col[ 4 * i ] )},${to( col[ 4 * i + 1 ] )},${to( col[ 4 * i + 2 ] )},0.75)`;
		this._cssColors[ i ] = c;
		return c;

	}

	// ── DOM pool ────────────────────────────────────────────────────────────────────────────

	_acquire( i, kind ) {

		let e = this._pool.pop();
		if ( ! e ) {

			const el = document.createElement( 'div' );
			e = { el, alpha: 0, target: 0, kind: - 1, x: 0, y: 0, tx: '', op: - 1, cls: '' };

		}

		e.el.textContent = this.text( i );
		e.kind = - 1;
		e.tx = ''; e.op = - 1;
		this._setKind( e, i, kind );
		if ( ! e.el.parentNode ) this.root.appendChild( e.el );
		e.el.style.display = '';
		return e;

	}

	_setKind( e, i, kind ) {

		e.kind = kind;
		const cls = KIND_CLASS[ kind ];
		if ( e.cls !== cls ) {

			e.el.className = cls;
			e.cls = cls;

		}

		if ( kind >= HOV ) e.el.style.setProperty( '--c', this._cssColor( i ) );
		else e.el.style.removeProperty( '--c' );

	}

	_release( e ) {

		e.el.style.display = 'none';
		e.el.style.opacity = '0';
		e.op = 0;
		this._pool.push( e );

	}

	_write( e ) {

		const dpr = ( typeof window !== 'undefined' && window.devicePixelRatio ) || 1;
		// snap to device pixels so text stays crisp (box heights are integers, see _lineH)
		const x = Math.round( e.x * dpr ) / dpr, y = Math.round( ( e.y - 0.5 * this._lineH( e.kind ) ) * dpr ) / dpr;
		const tx = `translate(${x}px,${y}px)`;
		if ( tx !== e.tx ) {

			e.el.style.transform = tx;
			e.tx = tx;

		}

		const op = Math.round( Math.min( 1, Math.max( 0, e.alpha ) ) * 100 ) / 100;
		if ( op !== e.op ) {

			e.el.style.opacity = String( op );
			e.op = op;

		}

	}

	dispose() {

		for ( const [ , e ] of this._active ) e.el.remove();
		for ( const e of this._pool ) e.el.remove();
		this._active.clear();
		this._pool.length = 0;
		this.root?.remove();
		this.root = null;

	}

}
