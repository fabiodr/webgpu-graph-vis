// CPU node picking against the asynchronous position readback (app.positions, 4 floats per node).
//
// Design
//   Every node is projected with the camera's view-projection matrix (the matrices of the last
//   rendered frame, i.e. what the user sees). A node is a candidate when the pointer lies within
//   max(screen radius, slop) of its projected centre; slop is 8 px for a mouse and 18 px for touch
//   so small nodes stay easy to hit with a finger. Candidates whose disc really contains the
//   pointer beat near misses; among those the node nearest to the camera wins (it is drawn on
//   top); among near misses the smallest gap to the disc edge wins, depth breaking ties.
//   One linear pass, no allocations (the result object is reused): ~0.2-0.5 ms for 20k nodes.
//
//   The world radius of node i comes from an injectable function, because only the node renderer
//   knows the real size (settings.nodeSize, degree scaling, …). Default:
//     radius(i) = graph.radii[i] * baseRadius,  baseRadius = graph.layoutRadius / (6 * cbrt(N))
//   multiplied by `radiusScale`.
//
// API
//   const picking = new Picking( { radius, radiusScale = 1, slopMouse = 8, slopTouch = 18 } )
//   picking.setGraph( graph )               rebuilds the default radius for a new PreparedGraph
//   picking.setRadiusFunction( fn | null )  fn( i ) -> world radius; null restores the default
//   picking.radiusScale                     extra multiplier applied to every radius
//   picking.slopFor( pointerType )          8 (mouse/pen) or 18 (touch)
//   picking.pick( positions, count, camera, width, height, x, y, slop ) -> node index | -1
//       x, y in CSS pixels relative to the canvas' top-left corner, width/height in CSS pixels.
//   picking.pickApp( app, x, y, pointerType ) -> index | -1     (uses app.positions/camera/size)
//   picking.result                          { index, distance, depth, x, y, radius } of the last pick
//   picking.project( positions, camera, width, height, i, out ) -> out {x, y, depth, radius} | null
//       screen position of node i (null when behind the camera); used by tests and overlays.
//   resolveClick( last | null, { x, y, index }, now, { doubleMs = 320, tolerance = 8 } ) -> { index, double }
//       which node a click means, given the click before it ( { t, x, y, index } ). A second click
//       close to the first in time and place is a double click on what the FIRST one hit: by then
//       the view may have moved under the pointer (the node card opens, the camera makes room for
//       it), so what lies under the pointer now is not what the person aimed at.

export function resolveClick( last, click, now, { doubleMs = 320, tolerance = 8 } = {} ) {

	const double = !! last && now - last.t < doubleMs && Math.hypot( click.x - last.x, click.y - last.y ) < tolerance;
	return { index: double ? last.index : click.index, double };

}

export class Picking {

	constructor( { radius = null, radiusScale = 1, slopMouse = 8, slopTouch = 18 } = {} ) {

		this.radiusFn = radius;
		this.radiusScale = radiusScale;
		this.slopMouse = slopMouse;
		this.slopTouch = slopTouch;
		this.graph = null;
		this.baseRadius = 1;
		this.result = { index: - 1, distance: Infinity, depth: Infinity, x: 0, y: 0, radius: 0 };
		this._m = new Float64Array( 16 );

	}

	setGraph( graph ) {

		this.graph = graph;
		const n = Math.max( 1, graph ? graph.N : 1 );
		this.baseRadius = graph ? graph.layoutRadius / ( 6 * Math.cbrt( n ) ) : 1;

	}

	setRadiusFunction( fn ) {

		this.radiusFn = fn || null;

	}

	slopFor( pointerType ) {

		return pointerType === 'touch' ? this.slopTouch : this.slopMouse;

	}

	/** World radius of node i (before radiusScale). */
	radiusOf( i ) {

		if ( this.radiusFn ) return this.radiusFn( i );
		const r = this.graph && this.graph.radii ? this.graph.radii[ i ] : 1;
		return r * this.baseRadius;

	}

	_viewProjection( camera ) {

		// m = P * V (column-major, like three.js Matrix4.elements)
		const p = camera.projectionMatrix.elements, v = camera.matrixWorldInverse.elements, m = this._m;
		for ( let c = 0; c < 4; c ++ ) {

			const v0 = v[ 4 * c ], v1 = v[ 4 * c + 1 ], v2 = v[ 4 * c + 2 ], v3 = v[ 4 * c + 3 ];
			for ( let r = 0; r < 4; r ++ ) m[ 4 * c + r ] = p[ r ] * v0 + p[ 4 + r ] * v1 + p[ 8 + r ] * v2 + p[ 12 + r ] * v3;

		}

		return m;

	}

	pickApp( app, x, y, pointerType = 'mouse' ) {

		if ( ! app.graph || ! app.positions ) return - 1;
		return this.pick( app.positions, app.graph.N, app.camera, app.width, app.height, x, y, this.slopFor( pointerType ) );

	}

	pick( positions, count, camera, width, height, x, y, slop = this.slopMouse ) {

		const res = this.result;
		res.index = - 1; res.distance = Infinity; res.depth = Infinity; res.radius = 0;
		if ( ! positions || ! camera || ! ( width > 0 ) || ! ( height > 0 ) ) return - 1;
		const n = Math.min( count, positions.length >> 2 );
		const m = this._viewProjection( camera );
		const m0 = m[ 0 ], m1 = m[ 1 ], m3 = m[ 3 ], m4 = m[ 4 ], m5 = m[ 5 ], m7 = m[ 7 ];
		const m8 = m[ 8 ], m9 = m[ 9 ], m11 = m[ 11 ], m12 = m[ 12 ], m13 = m[ 13 ], m15 = m[ 15 ];
		const hw = width * 0.5, hh = height * 0.5;
		const far2 = Math.max( width, height ) ** 2;
		// projected radius in px = worldRadius * focal / w, focal = P[5] * height / 2
		const focal = camera.projectionMatrix.elements[ 5 ] * hh * this.radiusScale;
		const minW = Math.max( 1e-6, ( camera.near || 0 ) * 0.5 );
		const fn = this.radiusFn;
		const radii = this.graph && this.graph.radii && this.graph.radii.length >= n ? this.graph.radii : null;
		const base = this.baseRadius;

		let best = - 1, bestTier = 2, bestKey = Infinity, bestD = 0, bestW = 0, bestR = 0, bestX = 0, bestY = 0;
		for ( let i = 0; i < n; i ++ ) {

			const k = 4 * i;
			const px = positions[ k ], py = positions[ k + 1 ], pz = positions[ k + 2 ];
			const w = m3 * px + m7 * py + m11 * pz + m15;
			if ( w <= minW ) continue; // behind the camera
			const iw = 1 / w;
			const sx = ( ( m0 * px + m4 * py + m8 * pz + m12 ) * iw + 1 ) * hw;
			const sy = ( 1 - ( m1 * px + m5 * py + m9 * pz + m13 ) * iw ) * hh;
			const dx = sx - x, dy = sy - y;
			const d2 = dx * dx + dy * dy;
			// cheap reject before evaluating the radius (a node disc is never larger than the screen)
			if ( d2 > far2 && d2 > slop * slop ) continue;
			const wr = fn ? fn( i ) : ( radii ? radii[ i ] : 1 ) * base;
			const r = wr * focal * iw;
			const reach = r > slop ? r : slop;
			if ( d2 > reach * reach ) continue;
			const d = Math.sqrt( d2 );
			let tier, key;
			if ( d <= r ) {

				tier = 0; key = w; // pointer inside the disc: nearest to the camera wins

			} else {

				tier = 1; key = ( d - r ) + w * 1e-9; // near miss: smallest gap wins, depth breaks ties

			}

			if ( tier < bestTier || ( tier === bestTier && key < bestKey ) ) {

				best = i; bestTier = tier; bestKey = key; bestD = d; bestW = w; bestR = r; bestX = sx; bestY = sy;

			}

		}

		if ( best >= 0 ) {

			res.index = best; res.distance = bestD; res.depth = bestW; res.radius = bestR; res.x = bestX; res.y = bestY;

		}

		return best;

	}

	project( positions, camera, width, height, i, out = { x: 0, y: 0, depth: 0, radius: 0 } ) {

		const m = this._viewProjection( camera );
		const k = 4 * i;
		const px = positions[ k ], py = positions[ k + 1 ], pz = positions[ k + 2 ];
		const w = m[ 3 ] * px + m[ 7 ] * py + m[ 11 ] * pz + m[ 15 ];
		if ( w <= 1e-6 ) return null;
		out.x = ( ( m[ 0 ] * px + m[ 4 ] * py + m[ 8 ] * pz + m[ 12 ] ) / w + 1 ) * width * 0.5;
		out.y = ( 1 - ( m[ 1 ] * px + m[ 5 ] * py + m[ 9 ] * pz + m[ 13 ] ) / w ) * height * 0.5;
		out.depth = w;
		out.radius = this.radiusOf( i ) * this.radiusScale * camera.projectionMatrix.elements[ 5 ] * height * 0.5 / w;
		return out;

	}

}
