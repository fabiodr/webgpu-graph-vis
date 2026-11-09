// Bundled edge renderer: the signature look of ImsvGraphVis (Kwon et al., TVCG 2016) — every edge is
// a hierarchically bundled cubic B-spline through the cluster centroids of its endpoints' ancestors,
// drawn as a glowing, camera-facing ribbon colored from the source to the target node hue.
//
// ─── Draw call ───────────────────────────────────────────────────────────────────────────────
//  ONE instanced draw for all edges: an InstancedBufferGeometry ribbon strip of S segments
//  ((S+1)*2 vertices, attribute position = (t in [0,1], side ±1, 0)), instanceCount = E. S is chosen
//  from E (<=2k: 32, <=20k: 20, <=100k: 12, else 8; x(maxPath+1)/7 for deep hierarchies, x2 while
//  layoutBlend > 0) unless overridden. Nothing per frame is O(E) on the CPU; the curve is re-evaluated by the vertex shader every frame from the live GPU buffers,
//  so the live force layout, node drag and the sphere morph are followed for free.
//
// ─── Curve (vertex stage) ────────────────────────────────────────────────────────────────────
//  Control polygon (see src/graph/prepare.js): P_0 = src, then the source's ancestors' centroids up
//  to the LCA level, back down the target's, P_{L-1} = tgt; L = 2 + 2*lca + (lca < D ? 1 : 0),
//  the root is never used. Holten straightening: P'_k = mix( lerp( src, tgt, k/(L-1) ), P_k, beta ).
//  Uniform cubic B-spline over P' with the end points tripled (L + 1 spans), so the curve starts
//  and ends exactly at the node centers. A vertex at parameter t evaluates only its span:
//  x = t*(L+1), span j = min(floor(x), L), u = x - j, control points Q[j..j+3] with
//  Q[i] = P'[clamp(i-2, 0, L-1)]; each is a node position or clusterPos[anc[node*D + level]].
//  The tangent is the analytic derivative (plus 1e-4 of the span chord, which only matters at the
//  two curve ends where the tripled knots make the derivative vanish).
//  Storage reads per vertex: edgeInfo 1, nodePos 2, nodeColor 2, nodeState 2, anc <=4,
//  clusterPos <=4 (<= 15); six storage bindings in the vertex stage, none in the fragment stage.
//
//  Sphere mode (uniforms.layoutBlend > 0, the paper's layout): every control point is split into a
//  direction from the sphere center (uniforms.graphCenter) and a radial level. Nodes keep their own
//  (|p - c| / sphereRadius), cluster points use level(h) = 1 + levelOffset + levelScale*(h-1)^exp
//  with h = l + 1 for a path entry at hierarchy level l, so bundles arc outward behind the nodes.
//  Directions are B-spline-blended in the sphere's log map around `sphereAxis` (+Z, the direction
//  the layout faces; this azimuthal-equidistant plane is exactly the planar treemap space that
//  prepare.js projects from), levels separately, then recombined as c + exp(V) * R * level. Unlike
//  normalize(Σ w d), the log map keeps the angular speed nearly uniform on 150° hops, so S samples
//  stay smooth. Straightening by beta works the same way (beta = 0: straight lines in the treemap
//  plane, at node level). Highlighted edges pull their cluster levels toward `highlightLevel`
//  (inside the sphere, in front of the context). Final point/tangent = mix(force, sphere, blend).
//
// ─── Ribbon ──────────────────────────────────────────────────────────────────────────────────
//  Expanded in screen space perpendicular to the projected tangent (exact derivative of the
//  perspective projection). Half width in pixels = clamp(projected world half width, minPixelWidth,
//  maxPixelWidth); the world width scales with graph.layoutRadius. Widths below 1 physical pixel
//  are clamped to 1 px and their alpha is scaled by the coverage ratio, and the strip is padded by
//  0.5 px on each side for an analytic anti-aliased edge profile (1 px box filter) in the fragment
//  stage. Degenerate screen tangents (curve pointing at the camera) fall back to a fixed direction.
//
// ─── Shading (fragment stage) ────────────────────────────────────────────────────────────────
//  Soft edge profile + tube shading mix(0.55, 1, sqrt(1 - v²)) (the paper's lit tubes) ; color
//  gradient src → tgt interpolated in OKLab (done as a linear blend of cube-root LMS, which is the
//  same thing since OKLab is linear in it). Opacity = auto(E) * opacity. Highlight (from the eased
//  nodeState): edges incident to the selected node (max(state[s].x, state[t].x)) become opaque,
//  ~1.5x wider and HDR-bright (bloom halo) while the rest fades to `contextOpacity` with
//  uniforms.focus; edges of the hovered node get a mild boost. HDR (> 1) output is intentional.
//
// ─── Public API ──────────────────────────────────────────────────────────────────────────────
//  new EdgeRenderer( { beta, opacity, width, minPixelWidth, maxPixelWidth, blendMode, segments,
//                      colorMode, visible, contextOpacity, highlightGlow, highlightLevel,
//                      hoverBoost, brightness, sphereSegmentScale, sphereAxis, renderOrder } )
//  App module (order 10): init( app ), setGraph( buffers, graph, app ), update( dt, app ), dispose()
//  setBundling( beta )            Holten bundling strength in [0,1] (default 0.85; paper 0.9)
//  .opacity                       multiplier on the automatic opacity (default 1)
//  .width                         multiplier on the world-space width (default 1)
//  .minPixelWidth / .maxPixelWidth   full ribbon width clamp, CSS pixels (defaults 0.75 / 24)
//  .blendMode                     'normal' | 'additive' (glow; good for dense graphs)
//  .segments                      null = auto from E, or an integer >= 1
//  .sphereSegmentScale            auto segment multiplier while layoutBlend > 0 (default 2)
//  .colorMode                     'gradient' | 'source'
//  .visible
//  .contextOpacity / .highlightGlow / .highlightLevel / .hoverBoost / .brightness
//  .sphereAxis                    THREE.Vector3, direction the sphere layout faces (default +Z)
//  .stats                         { edges, segments, vertices, autoOpacity }
//  .mesh                          the THREE.Mesh (renderOrder -1: drawn before nodes; world space,
//                                 its own transform is ignored; raycast disabled)
//
//  Also exported: autoSegments( E ), autoOpacity( E ), edgeCurveNode( … ) (the TSL curve builder,
//  reusable in compute kernels, e.g. GPU picking or tests) and evalEdgeCurveCPU( … ), a pure-JS
//  mirror of the shader used by tests and handy for CPU-side edge picking.

import * as THREE from 'three/webgpu';
import {
	Fn, If, instanceIndex, positionGeometry, uniform, float, int, uint, vec2, vec3, vec4, select, mix, clamp,
	max, min, abs, sqrt, pow, floor, length, dot, sin, cos, atan, smoothstep, varyingProperty,
	cameraProjectionMatrix, cameraViewMatrix, viewport, screenDPR,
} from 'three/tsl';

export const EDGE_DEFAULTS = {
	beta: 0.85,
	opacity: 1,
	width: 1,
	minPixelWidth: 0.75, // CSS px (full width); thinner edges fade by coverage below 1 device px
	maxPixelWidth: 24, // CSS px (full width)
	blendMode: 'normal',
	segments: null,
	colorMode: 'gradient',
	visible: true,
	contextOpacity: 0.12, // opacity factor of non-highlighted edges while a selection exists
	highlightGlow: 1.6, // extra HDR brightness of highlighted edges (bloom halo)
	highlightOpacity: 0.9,
	highlightLevel: 0.75, // sphere mode: level the cluster points of highlighted edges move to
	hoverBoost: 0.6,
	brightness: 1.25,
	sphereSegmentScale: 2,
	renderOrder: - 1,
};

/**
 * Ribbon segments per edge. Base by E (<=2k: 32, <=20k: 20, <=100k: 12, else 8), scaled up for
 * hierarchies deeper than 2 levels (more B-spline spans per edge) and by `sphereScale` while the
 * sphere layout is shown (its arcs are longer and cross the sphere silhouette).
 */
export function autoSegments( E, maxPath = 6, sphereScale = 1 ) {

	const base = E <= 2000 ? 32 : E <= 20000 ? 20 : E <= 100000 ? 12 : 8;
	return Math.min( 96, Math.round( base * Math.max( 1, ( maxPath + 1 ) / 7 ) * sphereScale ) );

}

/** Base edge opacity for E edges: dense graphs need lower alpha so bundles do not saturate. */
export function autoOpacity( E ) {

	return Math.min( 0.8, Math.max( 0.035, 0.85 * Math.pow( 400 / Math.max( 1, E ), 0.42 ) ) );

}

/** World half width factor for E edges (fraction of graph.layoutRadius). */
function autoHalfWidth( E ) {

	return 0.0022 * Math.min( 1, Math.max( 0.45, Math.pow( 1500 / Math.max( 1, E ), 0.2 ) ) );

}

// ── shared curve math ───────────────────────────────────────────────────────────────────────

/** Uniform cubic B-spline basis (b) and derivative (d) at u, as JS numbers. */
function basisCPU( u ) {

	const v = 1 - u, u2 = u * u, u3 = u2 * u;
	return {
		b: [ ( v * v * v ) / 6, ( 3 * u3 - 6 * u2 + 4 ) / 6, ( - 3 * u3 + 3 * u2 + 3 * u + 1 ) / 6, u3 / 6 ],
		d: [ - 0.5 * v * v, 1.5 * u2 - 2 * u, - 1.5 * u2 + u + 0.5, 0.5 * u2 ],
	};

}

/**
 * Pure-JS mirror of the vertex shader's curve (for tests and CPU picking).
 * @param {object} g PreparedGraph
 * @param {number} e edge index
 * @param {number} t curve parameter in [0,1]
 * @param {Float32Array} nodePos4 node positions (vec4 per node)
 * @param {Float32Array} clusterPos4 cluster centroids (vec4 per cluster)
 * @param {object} [o] { beta, layoutBlend, sphereRadius, center:[x,y,z], sphereAxis:[x,y,z],
 *                       levelOffset, levelScale, levelExponent, highlight, highlightLevel }
 * @returns {{ position: number[], tangent: number[], L: number, controlPoints: number[][] }}
 */
export function evalEdgeCurveCPU( g, e, t, nodePos4, clusterPos4, o = {} ) {

	const beta = o.beta ?? EDGE_DEFAULTS.beta;
	const blend = o.layoutBlend ?? 0;
	const R = o.sphereRadius ?? g.layoutRadius;
	const c = o.center ?? [ 0, 0, 0 ];
	const lvOff = o.levelOffset ?? 0.1, lvScale = o.levelScale ?? 0.35, lvExp = o.levelExponent ?? 2;
	const hl = o.highlight ?? 0, hlLevel = o.highlightLevel ?? EDGE_DEFAULTS.highlightLevel;
	const D = g.D;
	const s = g.edges[ 2 * e ], tg = g.edges[ 2 * e + 1 ];
	const lca = Math.min( g.edgeLca[ e ], D );
	const L = 2 + 2 * lca + ( lca < D ? 1 : 0 );
	const pS = [ nodePos4[ 4 * s ], nodePos4[ 4 * s + 1 ], nodePos4[ 4 * s + 2 ] ];
	const pT = [ nodePos4[ 4 * tg ], nodePos4[ 4 * tg + 1 ], nodePos4[ 4 * tg + 2 ] ];
	const sub = ( a, b ) => [ a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] ];
	const lerp3 = ( a, b, k ) => [ a[ 0 ] + ( b[ 0 ] - a[ 0 ] ) * k, a[ 1 ] + ( b[ 1 ] - a[ 1 ] ) * k, a[ 2 ] + ( b[ 2 ] - a[ 2 ] ) * k ];
	const len = ( a ) => Math.hypot( a[ 0 ], a[ 1 ], a[ 2 ] );
	const nrm = ( a ) => {

		const l = Math.max( len( a ), 1e-6 );
		return [ a[ 0 ] / l, a[ 1 ] / l, a[ 2 ] / l ];

	};

	const ax = o.sphereAxis ?? [ 0, 0, 1 ];
	const dot3 = ( a, b ) => a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];
	const logMap = ( r ) => {

		const dn = nrm( r );
		const cs = dot3( dn, ax );
		const perp = [ dn[ 0 ] - ax[ 0 ] * cs, dn[ 1 ] - ax[ 1 ] * cs, dn[ 2 ] - ax[ 2 ] * cs ];
		const sn = len( perp );
		const k = sn > 1e-7 ? Math.atan2( sn, cs ) / sn : 1;
		return [ perp[ 0 ] * k, perp[ 1 ] * k, perp[ 2 ] * k ];

	};

	const rS = sub( pS, c ), rT = sub( pT, c );
	const levS = len( rS ) / R, levT = len( rT ) / R, vS = logMap( rS ), vT = logMap( rT );
	const point = ( k ) => {

		if ( k === 0 ) return { p: pS, lev: levS };
		if ( k === L - 1 ) return { p: pT, lev: levT };
		const srcSide = 2 * k <= L - 1;
		const lv = srcSide ? k - 1 : L - 2 - k;
		const node = srcSide ? s : tg;
		const cl = g.anc[ node * D + lv ];
		const h = lv + 1;
		const level = 1 + lvOff + lvScale * ( h > 1 ? Math.pow( h - 1, lvExp ) : 0 );
		return { p: [ clusterPos4[ 4 * cl ], clusterPos4[ 4 * cl + 1 ], clusterPos4[ 4 * cl + 2 ] ], lev: level + ( hlLevel - level ) * hl };

	};

	const controlPoints = [];
	for ( let k = 0; k < L; k ++ ) controlPoints.push( point( k ).p );

	const x = Math.min( Math.max( t, 0 ), 1 ) * ( L + 1 );
	const j = Math.min( Math.floor( x ), L );
	const u = x - j;
	const { b, d } = basisCPU( u );
	const cF = [ 0, 0, 0 ], tF = [ 0, 0, 0 ], V = [ 0, 0, 0 ], dV = [ 0, 0, 0 ];
	let cL = 0, dL = 0;
	const F = [], Sv = [], Sl = [];
	for ( let i = 0; i < 4; i ++ ) {

		const k = Math.min( Math.max( j + i - 2, 0 ), L - 1 );
		const a = k / ( L - 1 );
		const { p, lev } = point( k );
		const f = sub( lerp3( lerp3( pS, pT, a ), p, beta ), pS );
		const v = lerp3( lerp3( vS, vT, a ), logMap( sub( p, c ) ), beta );
		const lv = ( levS + ( levT - levS ) * a ) * ( 1 - beta ) + lev * beta;
		F.push( f ); Sv.push( v ); Sl.push( lv );
		for ( let q = 0; q < 3; q ++ ) {

			cF[ q ] += b[ i ] * f[ q ]; tF[ q ] += d[ i ] * f[ q ];
			V[ q ] += b[ i ] * v[ q ]; dV[ q ] += d[ i ] * v[ q ];

		}

		cL += b[ i ] * lv; dL += d[ i ] * lv;

	}

	for ( let q = 0; q < 3; q ++ ) {

		tF[ q ] += 1e-4 * ( F[ 3 ][ q ] - F[ 0 ][ q ] );
		dV[ q ] += 1e-4 * ( Sv[ 3 ][ q ] - Sv[ 0 ][ q ] );

	}

	dL += 1e-4 * ( Sl[ 3 ] - Sl[ 0 ] );
	const posF = [ pS[ 0 ] + cF[ 0 ], pS[ 1 ] + cF[ 1 ], pS[ 2 ] + cF[ 2 ] ];
	const r = len( V );
	const sinc = r > 1e-4 ? Math.sin( r ) / r : 1 - ( r * r ) / 6;
	const cr = Math.cos( r );
	const E = [ 0, 1, 2 ].map( ( q ) => ax[ q ] * cr + V[ q ] * sinc );
	const n = V.map( ( v ) => v / Math.max( r, 1e-12 ) );
	const nd = dot3( n, dV );
	const dE = [ 0, 1, 2 ].map( ( q ) => - ax[ q ] * r * sinc * nd + n[ q ] * cr * nd + ( dV[ q ] - n[ q ] * nd ) * sinc );
	const posS = [ 0, 1, 2 ].map( ( q ) => c[ q ] + E[ q ] * R * cL );
	const tanS = [ 0, 1, 2 ].map( ( q ) => R * ( E[ q ] * dL + dE[ q ] * cL ) );
	const position = lerp3( posF, posS, blend );
	const tangent = lerp3( tF, tanS, blend );
	return { position, tangent, L, controlPoints };

}

/**
 * TSL builder for the bundled curve of one edge. Must be called inside a Fn() body (it declares
 * variables and branches). Returns { position, tangent } (vec3 nodes, world space).
 * @param {import('../gpu/GraphBuffers.js').GraphBuffers} buffers
 * @param {object} graph PreparedGraph (only D is read; it is baked into the shader)
 * @param {object} p { beta, highlightLevel, sphereAxis } uniform nodes (float, float, vec3 unit)
 * @param {Node} edge uvec4 node: edgeInfo element (src, tgt, lca, 0)
 * @param {Node} t float node in [0,1]
 * @param {Node} [highlight] float node in [0,1]: pulls cluster levels in sphere mode
 */
export function edgeCurveNode( buffers, graph, p, edge, t, highlight = float( 0 ) ) {

	const U = buffers.uniforms;
	const D = graph.D;
	const s = edge.x, tg = edge.y;
	const pS = buffers.nodePos.element( s ).xyz.toVar();
	const pT = buffers.nodePos.element( tg ).xyz.toVar();
	const lca = min( int( edge.z ), int( D ) );
	const L = ( D > 0 ? int( 2 ).add( lca.mul( 2 ) ).add( select( lca.lessThan( int( D ) ), int( 1 ), int( 0 ) ) ) : int( 2 ) ).toVar();
	const Lm1 = L.sub( 1 ).toVar();
	const x = clamp( t, 0, 1 ).mul( float( L.add( 1 ) ) );
	const j = min( int( floor( x ) ), L ).toVar();
	const u = x.sub( float( j ) ).toVar();

	// uniform cubic B-spline basis and derivative
	const v = float( 1 ).sub( u );
	const u2 = u.mul( u ), u3 = u2.mul( u );
	const b = [
		v.mul( v ).mul( v ).div( 6 ),
		u3.mul( 3 ).sub( u2.mul( 6 ) ).add( 4 ).div( 6 ),
		u3.mul( - 3 ).add( u2.mul( 3 ) ).add( u.mul( 3 ) ).add( 1 ).div( 6 ),
		u3.div( 6 ),
	].map( ( n ) => n.toVar() );
	const d = [
		v.mul( v ).mul( - 0.5 ),
		u2.mul( 1.5 ).sub( u.mul( 2 ) ),
		u2.mul( - 1.5 ).add( u ).add( 0.5 ),
		u2.mul( 0.5 ),
	].map( ( n ) => n.toVar() );

	const center = U.graphCenter;
	const R = max( U.sphereRadius, 1e-6 );
	const beta = clamp( p.beta, 0, 1 );
	const blend = clamp( U.layoutBlend, 0, 1 );
	const rS = pS.sub( center ), rT = pT.sub( center );
	const levS = length( rS ).div( R ).toVar(), levT = length( rT ).div( R ).toVar();

	// gather the span's 4 control points: position (force mode) and level (sphere mode)
	const P = [], LV = [], A = [];
	for ( let i = 0; i < 4; i ++ ) {

		const k = clamp( j.add( int( i - 2 ) ), int( 0 ), Lm1 ).toVar();
		const pk = vec3( 0 ).toVar();
		const lev = float( 1 ).toVar();
		if ( D > 0 ) {

			If( k.greaterThan( int( 0 ) ).and( k.lessThan( Lm1 ) ), () => {

				const srcSide = k.mul( 2 ).lessThanEqual( Lm1 );
				const lv = select( srcSide, k.sub( 1 ), Lm1.sub( 1 ).sub( k ) );
				const node = select( srcSide, s, tg );
				const cl = buffers.anc.element( node.mul( uint( D ) ).add( uint( lv ) ) );
				pk.assign( buffers.clusterPos.element( cl ).xyz );
				const hm1 = float( lv );
				const level = float( 1 ).add( U.levelOffset ).add( U.levelScale.mul( select( lv.greaterThan( int( 0 ) ), pow( max( hm1, 1e-6 ), U.levelExponent ), float( 0 ) ) ) );
				lev.assign( mix( level, p.highlightLevel, highlight ) );

			} ).ElseIf( k.equal( int( 0 ) ), () => {

				pk.assign( pS ); lev.assign( levS );

			} ).Else( () => {

				pk.assign( pT ); lev.assign( levT );

			} );

		} else {

			If( k.equal( int( 0 ) ), () => {

				pk.assign( pS ); lev.assign( levS );

			} ).Else( () => {

				pk.assign( pT ); lev.assign( levT );

			} );

		}

		P.push( pk ); LV.push( lev ); A.push( float( k ).div( float( Lm1 ) ).toVar() );

	}

	// force mode: Euclidean B-spline over straightened points, relative to the source for precision
	const F = P.map( ( pk, i ) => mix( mix( pS, pT, A[ i ] ), pk, beta ).sub( pS ).toVar() );
	const cF = F[ 0 ].mul( b[ 0 ] ).add( F[ 1 ].mul( b[ 1 ] ) ).add( F[ 2 ].mul( b[ 2 ] ) ).add( F[ 3 ].mul( b[ 3 ] ) );
	const tF = F[ 0 ].mul( d[ 0 ] ).add( F[ 1 ].mul( d[ 1 ] ) ).add( F[ 2 ].mul( d[ 2 ] ) ).add( F[ 3 ].mul( d[ 3 ] ) );
	const position = pS.add( cF ).toVar();
	const tangent = tF.add( F[ 3 ].sub( F[ 0 ] ).mul( 1e-4 ) ).toVar();

	// sphere mode (skipped while layoutBlend == 0): directions are blended in the log map of the
	// sphere around `axis` (azimuthal-equidistant tangent plane: the paper's planar treemap space,
	// since prepare.js projects with exactly this map), levels separately; this keeps the angular
	// speed nearly uniform even for 150° hops, where normalize(Σ w d) would bunch samples at the
	// ends of a span and leave visible polygon corners in its middle.
	If( blend.greaterThan( 0 ), () => {

		const axis = p.sphereAxis;
		const logMap = ( r ) => {

			const dn = r.div( max( length( r ), 1e-12 ) );
			const c = dot( dn, axis );
			const perp = dn.sub( axis.mul( c ) );
			const s = length( perp );
			const theta = atan( s, c );
			return perp.mul( select( s.greaterThan( 1e-7 ), theta.div( max( s, 1e-7 ) ), float( 1 ) ) );

		};

		const vS = logMap( rS ).toVar(), vT = logMap( rT ).toVar();
		const Sv = P.map( ( pk, i ) => mix( mix( vS, vT, A[ i ] ), logMap( pk.sub( center ) ), beta ).toVar() );
		const Sl = LV.map( ( lev, i ) => mix( mix( levS, levT, A[ i ] ), lev, beta ).toVar() );
		const V = Sv[ 0 ].mul( b[ 0 ] ).add( Sv[ 1 ].mul( b[ 1 ] ) ).add( Sv[ 2 ].mul( b[ 2 ] ) ).add( Sv[ 3 ].mul( b[ 3 ] ) ).toVar();
		const cL = Sl[ 0 ].mul( b[ 0 ] ).add( Sl[ 1 ].mul( b[ 1 ] ) ).add( Sl[ 2 ].mul( b[ 2 ] ) ).add( Sl[ 3 ].mul( b[ 3 ] ) ).toVar();
		// derivative (plus 1e-4 of the span chord, as in force mode; the map is linear in it)
		const dV = Sv[ 0 ].mul( d[ 0 ] ).add( Sv[ 1 ].mul( d[ 1 ] ) ).add( Sv[ 2 ].mul( d[ 2 ] ) ).add( Sv[ 3 ].mul( d[ 3 ] ) ).add( Sv[ 3 ].sub( Sv[ 0 ] ).mul( 1e-4 ) ).toVar();
		const dL = Sl[ 0 ].mul( d[ 0 ] ).add( Sl[ 1 ].mul( d[ 1 ] ) ).add( Sl[ 2 ].mul( d[ 2 ] ) ).add( Sl[ 3 ].mul( d[ 3 ] ) ).add( Sl[ 3 ].sub( Sl[ 0 ] ).mul( 1e-4 ) );
		// exponential map: E = axis cos r + V sin(r)/r
		const r = length( V ).toVar();
		const sinc = select( r.greaterThan( 1e-4 ), sin( r ).div( max( r, 1e-4 ) ), r.mul( r ).div( - 6 ).add( 1 ) ).toVar();
		const cr = cos( r ).toVar();
		const E = axis.mul( cr ).add( V.mul( sinc ) ).toVar();
		const n = V.div( max( r, 1e-12 ) );
		const nd = dot( n, dV );
		const dE = axis.mul( r.mul( sinc ).negate().mul( nd ) ).add( n.mul( cr.mul( nd ) ) ).add( dV.sub( n.mul( nd ) ).mul( sinc ) );
		const posS = center.add( E.mul( R.mul( cL ) ) );
		const tanS = E.mul( dL ).add( dE.mul( cL ) ).mul( R );
		position.assign( mix( position, posS, blend ) );
		tangent.assign( mix( tangent, tanS, blend ) );

	} );

	// the curve ends are the node centers by construction; snap them so the sphere-mode trig
	// (WGSL sin/cos/atan are only ~1e-4 accurate) cannot leave a visible offset
	If( t.lessThanEqual( 0 ), () => {

		position.assign( pS );

	} ).ElseIf( t.greaterThanEqual( 1 ), () => {

		position.assign( pT );

	} );

	return { position, tangent };

}

// linear sRGB <-> cube-root LMS (OKLab is a linear map of the latter, so blends are identical)
const lmsRoot = ( c ) => {

	const r = max( c.x, 0 ), g = max( c.y, 0 ), b = max( c.z, 0 );
	const l = r.mul( 0.4122214708 ).add( g.mul( 0.5363325363 ) ).add( b.mul( 0.0514459929 ) );
	const m = r.mul( 0.2119034982 ).add( g.mul( 0.6806995451 ) ).add( b.mul( 0.1073969566 ) );
	const s = r.mul( 0.0883024619 ).add( g.mul( 0.2817188376 ) ).add( b.mul( 0.6299787005 ) );
	return pow( max( vec3( l, m, s ), vec3( 1e-9 ) ), vec3( 1 / 3 ) );

};

const lmsRootToLinear = ( q ) => {

	const c = q.mul( q ).mul( q );
	return max( vec3(
		dot( c, vec3( 4.0767416621, - 3.3077115913, 0.2309699292 ) ),
		dot( c, vec3( - 1.2684380046, 2.6097574011, - 0.3413193965 ) ),
		dot( c, vec3( - 0.0041960863, - 0.7034186147, 1.7076147010 ) ),
	), vec3( 0 ) );

};

function ribbonGeometry( S ) {

	const geometry = new THREE.InstancedBufferGeometry();
	const pos = new Float32Array( ( S + 1 ) * 2 * 3 );
	for ( let i = 0; i <= S; i ++ ) {

		const t = i / S;
		pos.set( [ t, - 1, 0, t, 1, 0 ], i * 6 );

	}

	const index = [];
	for ( let i = 0; i < S; i ++ ) {

		const a = 2 * i;
		index.push( a, a + 1, a + 2, a + 1, a + 3, a + 2 );

	}

	geometry.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
	geometry.setIndex( index );
	geometry.boundingSphere = new THREE.Sphere( new THREE.Vector3(), Infinity );
	return geometry;

}

export class EdgeRenderer {

	constructor( options = {} ) {

		const o = { ...EDGE_DEFAULTS, ...options };
		this.order = 10;
		this.app = null;
		this.buffers = null;
		this.graph = null;
		this.mesh = null;
		this.material = null;
		this.geometry = null;
		this._S = 0;
		this._E = 0;
		this._auto = 1;
		this._opacity = o.opacity;
		this._width = o.width;
		this._segments = o.segments;
		this._blendMode = o.blendMode;
		this._colorMode = o.colorMode;
		this._visible = o.visible;
		this._sphereSegments = false;
		this.renderOrder = o.renderOrder;
		this.sphereSegmentScale = o.sphereSegmentScale; // auto segments multiplier in sphere mode

		// uniforms owned by the renderer (they survive graph changes)
		this.u = {
			beta: uniform( o.beta ),
			opacity: uniform( 1 ), // auto(E) * opacity multiplier
			halfWidth: uniform( 1 ), // world-space half width
			minPx: uniform( o.minPixelWidth ),
			maxPx: uniform( o.maxPixelWidth ),
			sourceColor: uniform( o.colorMode === 'source' ? 1 : 0 ),
			contextOpacity: uniform( o.contextOpacity ),
			highlightGlow: uniform( o.highlightGlow ),
			highlightOpacity: uniform( o.highlightOpacity ),
			highlightLevel: uniform( o.highlightLevel ),
			sphereAxis: uniform( new THREE.Vector3( ...( o.sphereAxis || [ 0, 0, 1 ] ) ).normalize() ),
			hoverBoost: uniform( o.hoverBoost ),
			brightness: uniform( o.brightness ),
		};

	}

	init( app ) {

		this.app = app;

	}

	// ── API ────────────────────────────────────────────────────────────────────────────────

	setBundling( beta ) {

		this.u.beta.value = Math.min( 1, Math.max( 0, beta ) );

	}

	get bundling() {

		return this.u.beta.value;

	}

	get opacity() {

		return this._opacity;

	}

	set opacity( v ) {

		this._opacity = Math.max( 0, v );
		this._updateUniforms();

	}

	get width() {

		return this._width;

	}

	set width( v ) {

		this._width = Math.max( 0, v );
		this._updateUniforms();

	}

	get minPixelWidth() {

		return this.u.minPx.value;

	}

	set minPixelWidth( v ) {

		this.u.minPx.value = Math.max( 0, v );

	}

	get maxPixelWidth() {

		return this.u.maxPx.value;

	}

	set maxPixelWidth( v ) {

		this.u.maxPx.value = Math.max( 0.1, v );

	}

	get blendMode() {

		return this._blendMode;

	}

	set blendMode( v ) {

		this._blendMode = v === 'additive' ? 'additive' : 'normal';
		if ( this.material ) {

			this.material.blending = this._blendMode === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending;
			this.material.needsUpdate = true;

		}

	}

	get segments() {

		return this._segments;

	}

	set segments( v ) {

		this._segments = v == null ? null : Math.max( 1, Math.min( 256, Math.round( v ) ) );
		this._updateGeometry();

	}

	get colorMode() {

		return this._colorMode;

	}

	set colorMode( v ) {

		this._colorMode = v === 'source' ? 'source' : 'gradient';
		this.u.sourceColor.value = this._colorMode === 'source' ? 1 : 0;

	}

	get visible() {

		return this._visible;

	}

	set visible( v ) {

		this._visible = !! v;
		if ( this.mesh ) this.mesh.visible = this._visible && this._E > 0;

	}

	get contextOpacity() {

		return this.u.contextOpacity.value;

	}

	set contextOpacity( v ) {

		this.u.contextOpacity.value = v;

	}

	get highlightGlow() {

		return this.u.highlightGlow.value;

	}

	set highlightGlow( v ) {

		this.u.highlightGlow.value = v;

	}

	get highlightLevel() {

		return this.u.highlightLevel.value;

	}

	set highlightLevel( v ) {

		this.u.highlightLevel.value = v;

	}

	get hoverBoost() {

		return this.u.hoverBoost.value;

	}

	set hoverBoost( v ) {

		this.u.hoverBoost.value = v;

	}

	get sphereAxis() {

		return this.u.sphereAxis.value;

	}

	set sphereAxis( v ) {

		this.u.sphereAxis.value.copy( v ).normalize();

	}

	get brightness() {

		return this.u.brightness.value;

	}

	set brightness( v ) {

		this.u.brightness.value = v;

	}

	get stats() {

		return { edges: this._E, segments: this._S, vertices: this._E * ( this._S + 1 ) * 2, triangles: this._E * this._S * 2, autoOpacity: this._auto };

	}

	// ── module hooks ───────────────────────────────────────────────────────────────────────

	setGraph( buffers, graph, app ) {

		this.app = app || this.app;
		this._disposeGPU();
		this.buffers = buffers;
		this.graph = graph;
		this._E = graph.E;
		this._auto = autoOpacity( graph.E );
		this._updateUniforms();

		this.material = this._createMaterial( buffers, graph );
		this._updateGeometry();
		const mesh = new THREE.Mesh( this.geometry, this.material );
		mesh.name = 'edges';
		mesh.frustumCulled = false;
		mesh.raycast = () => {}; // geometry holds (t, side), not positions: never raycast it
		mesh.renderOrder = this.renderOrder;
		mesh.visible = this._visible && this._E > 0;
		this.mesh = mesh;
		this.app.scene.add( mesh );

	}

	update() {

		// the curve is evaluated on the GPU from the live buffers; the only CPU work is O(1): pick
		// the denser auto segment count while the sphere layout is (partly) shown
		if ( this.buffers && this._segments == null && ( this.buffers.uniforms.layoutBlend.value > 0 ) !== this._sphereSegments ) this._updateGeometry();

	}

	dispose() {

		this._disposeGPU();
		this.buffers = null;
		this.graph = null;

	}

	// ── internals ──────────────────────────────────────────────────────────────────────────

	_updateUniforms() {

		this.u.opacity.value = this._auto * this._opacity;
		if ( this.graph ) this.u.halfWidth.value = this.graph.layoutRadius * autoHalfWidth( this.graph.E ) * this._width;

	}

	_updateGeometry() {

		if ( ! this.graph ) return;
		const sphere = this.buffers && this.buffers.uniforms.layoutBlend.value > 0;
		this._sphereSegments = sphere;
		const S = this._segments ?? autoSegments( this.graph.E, this.graph.maxPath, sphere ? this.sphereSegmentScale : 1 );
		if ( S !== this._S || ! this.geometry ) {

			const old = this.geometry;
			this.geometry = ribbonGeometry( S );
			this._S = S;
			if ( this.mesh ) this.mesh.geometry = this.geometry;
			if ( old ) old.dispose();

		}

		this.geometry.instanceCount = this.graph.E;

	}

	_disposeGPU() {

		if ( this.mesh ) {

			this.mesh.removeFromParent();
			this.mesh = null;

		}

		if ( this.material ) {

			this.material.dispose();
			this.material = null;

		}

		if ( this.geometry ) {

			this.geometry.dispose();
			this.geometry = null;
			this._S = 0;

		}

	}

	_createMaterial( buffers, graph ) {

		const u = this.u;
		const U = buffers.uniforms;
		const vColor = varyingProperty( 'vec4', 'vEdgeColor' );
		const vProfile = varyingProperty( 'vec3', 'vEdgeProfile' ); // (signed px, half width px, highlight)

		const material = new THREE.MeshBasicNodeMaterial( {
			transparent: true,
			depthWrite: false,
			depthTest: true,
			side: THREE.DoubleSide,
			blending: this._blendMode === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending,
		} );
		material.name = 'EdgeRenderer';
		material.fog = false;

		material.vertexNode = Fn( () => {

			const edge = buffers.edgeInfo.element( instanceIndex ).toVar();
			const t = positionGeometry.x;
			const side = positionGeometry.y;

			// highlight state (eased per node)
			const stS = buffers.nodeState.element( edge.x ).toVar();
			const stT = buffers.nodeState.element( edge.y ).toVar();
			const hl = max( stS.x, stT.x ).toVar();
			const hov = max( stS.w, stT.w ).toVar();

			const { position, tangent } = edgeCurveNode( buffers, graph, u, edge, t, hl );

			// color: OKLab gradient (cube-root LMS blend) or source color
			const cS = buffers.nodeColor.element( edge.x ).xyz.toVar();
			const cT = buffers.nodeColor.element( edge.y ).xyz.toVar();
			const grad = lmsRootToLinear( mix( lmsRoot( cS ), lmsRoot( cT ), clamp( t, 0, 1 ) ) );
			const base = mix( grad, cS, u.sourceColor );

			// opacity and brightness (focus fades the context, highlight/hover boost)
			const focus = clamp( U.focus, 0, 1 );
			const ctx = mix( float( 1 ), u.contextOpacity, focus.mul( hl.oneMinus() ) );
			const a0 = u.opacity.mul( ctx );
			const a1 = mix( a0, max( a0, u.highlightOpacity ), hl );
			const alpha = min( a1.mul( hov.mul( u.hoverBoost ).add( 1 ) ), 1 ).toVar();
			const bright = u.brightness.mul( hl.mul( u.highlightGlow ).add( hov.mul( u.hoverBoost ).mul( 0.5 ) ).add( 1 ) );

			// projection and screen-space expansion (graph buffers are world space; the mesh
			// transform is ignored, which also saves a per-vertex mat4 x mat4)
			const clip = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( position, 1 ) ) ).toVar();
			const dclip = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( tangent, 0 ) ) ).toVar();
			const vp = viewport.zw;
			const sT = dclip.xy.mul( clip.w ).sub( clip.xy.mul( dclip.w ) ).mul( vp ).toVar();
			const sLen = length( sT );
			const dir = select( sLen.greaterThan( 1e-30 ), sT.div( max( sLen, 1e-30 ) ), vec2( 1, 0 ) );
			const normal = vec2( dir.y.negate(), dir.x );

			const projScale = abs( cameraProjectionMatrix.element( 1 ).element( 1 ) ).mul( vp.y ).mul( 0.5 );
			const projected = u.halfWidth.mul( projScale ).div( max( clip.w, 1e-5 ) );
			const dpr = screenDPR;
			const widthBoost = hl.mul( 0.5 ).add( hov.mul( 0.25 ) ).add( 1 );
			const half0 = clamp( projected, u.minPx.mul( dpr ).mul( 0.5 ), u.maxPx.mul( dpr ).mul( 0.5 ) ).mul( widthBoost ).toVar();
			const coverage = clamp( half0.mul( 2 ), 0, 1 );
			const halfPx = max( half0, 0.5 );
			const geomHalf = halfPx.add( 0.5 ); // AA pad: the profile below reaches 0 at halfPx + 0.5
			const offset = normal.mul( side.mul( geomHalf ) ).mul( 2 ).div( vp );

			vColor.assign( vec4( base.mul( bright ), alpha.mul( coverage ) ) );
			vProfile.assign( vec3( side.mul( geomHalf ), halfPx, hl ) );

			// max(w, 0): a ribbon crossing the camera plane tapers instead of twisting
			return vec4( clip.xy.add( offset.mul( max( clip.w, 0 ) ) ), clip.z, clip.w );

		} )();

		const d = abs( vProfile.x );
		const halfPx = vProfile.y;
		const r = min( d.div( halfPx ), 1 );
		const aa = clamp( halfPx.add( 0.5 ).sub( d ), 0, 1 );
		const tube = mix( float( 0.55 ), float( 1 ), sqrt( max( r.mul( r ).oneMinus(), 0 ) ) );
		const shade = mix( float( 1 ), tube, smoothstep( 0.6, 1.8, halfPx ) );
		const q = r.oneMinus();
		const core = q.mul( q ).mul( q ).mul( vProfile.z ).mul( u.highlightGlow ).mul( 0.6 );
		material.colorNode = vColor.xyz.mul( shade.add( core ) );
		material.opacityNode = vColor.w.mul( aa );
		return material;

	}

}
