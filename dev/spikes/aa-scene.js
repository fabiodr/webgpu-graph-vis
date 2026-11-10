// Spike "aa": the test scene shared by every anti-aliasing strategy. NOT production code.
//
//   createTestScene( { nodes, edges, clusters, seed, segments } )
//
// ~150 opaque impostor spheres + ~3,000 thin curved ribbons, both driven by storage buffers the way
// the app does it (src/render/NodeRenderer.js, src/render/EdgeRenderer.js), reduced to what matters
// for anti-aliasing:
//
//   spheres   one instanced quad per node, ray-traced in the fragment stage, per-pixel depth
//             (material.depthNode). Silhouette: 'a2c' (analytic coverage -> alphaToCoverage, needs
//             MSAA) or 'cut' (hard 50 % discard, for single-sample passes).
//   ribbons   one instanced strip per edge; a cubic Bezier (src, toward src cluster, toward tgt
//             cluster, tgt) evaluated in the vertex stage and expanded in screen space. Width is
//             clamped to >= minPx pixels; narrower ribbons keep 1 px and scale alpha by coverage.
//             Edge modes (how transparency / coverage is resolved):
//               blend    alpha blended, no depth write, analytic 1 px edge profile   (app today)
//               a2c      opaque + depth write, alpha -> alphaToCoverage              (needs MSAA)
//               mask     opaque + depth write, alpha -> stochastic @builtin(sample_mask),
//                        decorrelated per edge (stochastic transparency)            (needs MSAA)
//               hash     opaque + depth write, alpha -> stochastic Discard (IGN, per edge and
//                        optionally per frame), meant to be accumulated by TRAA / TAAU
//               alphahash  opaque + depth write, material.alphaHash = true (three's built-in)
//               opaque   opaque + depth write, hard edges, no transparency
//
// Motion vectors: three's `velocity` node only knows object matrices (modelViewMatrix *
// positionLocal), which is meaningless for materials whose vertexNode reads positions from storage
// buffers. With `velocity: true` every material writes its own value into the pass's 'velocity'
// MRT slot through material.mrtNode:  ndc( viewProj * P_now ) - ndc( prevViewProj * P_prev ),
// where P_prev comes from the previous-position buffers (nodePrev / clusterPrev) and both matrices
// are UNJITTERED (see updateMotionUniforms / commitMotionUniforms).
import * as THREE from 'three/webgpu';
import {
	Fn, instanceIndex, instancedArray, uniform, varyingProperty, positionGeometry, float, uint, vec2, vec3, vec4, mix,
	clamp, max, min, abs, sqrt, dot, cross, length, normalize, select, fract, floor, fwidth, Discard, screenCoordinate,
	viewport, cameraProjectionMatrix, cameraViewMatrix, cameraNear, cameraFar, viewZToPerspectiveDepth,
	interleavedGradientNoise, mrt, output, nodeImmutable,
} from 'three/tsl';

function mulberry32( a ) {

	return () => {

		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul( a ^ a >>> 15, 1 | a );
		t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};

}

/**
 * Fragment output @builtin(sample_mask). TSL r187 has no node for it; WGSLNodeBuilder.getBuiltin()
 * can register any output builtin, exactly like builder.getFragDepth() does for frag_depth.
 * Usage inside a fragment Fn:  sampleMask.assign( uintNode )
 */
class SampleMaskNode extends THREE.Node {

	static get type() {

		return 'SampleMaskNode';

	}

	constructor() {

		super( 'uint' );

	}

	generate( builder ) {

		return 'output.' + builder.getBuiltin( 'sample_mask', 'sampleMask', 'u32', 'output' );

	}

}

const sampleMask = nodeImmutable( SampleMaskNode );

function quadGeometry() {

	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ] ), 3 ) );
	g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	return g;

}

/** Strip of S segments; attribute position = ( t in [0,1], side ±1, 0 ). */
function ribbonGeometry( S ) {

	const g = new THREE.InstancedBufferGeometry();
	const pos = new Float32Array( ( S + 1 ) * 6 );
	const index = [];
	for ( let i = 0; i <= S; i ++ ) pos.set( [ i / S, - 1, 0, i / S, 1, 0 ], i * 6 );
	for ( let i = 0; i < S; i ++ ) index.push( 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2 );
	g.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
	g.setIndex( index );
	return g;

}

export function createTestScene( { nodes: N = 150, edges: E = 3000, clusters: K = 8, seed = 7, segments: S = 16 } = {} ) {

	const rnd = mulberry32( seed );
	const gauss = () => Math.sqrt( - 2 * Math.log( 1 - rnd() ) ) * Math.cos( 6.2831853 * rnd() );

	// ── data: K clusters on a sphere, nodes around them, edges mostly inside clusters ───────────
	const centre = [];
	for ( let k = 0; k < K; k ++ ) centre.push( new THREE.Vector3( gauss(), gauss(), gauss() ).normalize().multiplyScalar( 85 ) );
	const cluster = new Uint32Array( N );
	const rest = new Float32Array( 4 * N ); // positions at rest (node motion is an offset from these)
	const colors = new Float32Array( 4 * N );
	const color = new THREE.Color();
	for ( let i = 0; i < N; i ++ ) {

		const k = cluster[ i ] = i % K;
		rest.set( [ centre[ k ].x + gauss() * 17, centre[ k ].y + gauss() * 17, centre[ k ].z + gauss() * 17, 0 ], 4 * i );
		color.setHSL( ( k / K + 0.03 * gauss() + 1 ) % 1, 0.75, 0.55 );
		colors.set( [ color.r, color.g, color.b, 0.7 + 0.9 * rnd() ], 4 * i ); // w = radius factor

	}

	const edgeData = new Uint32Array( 4 * E );
	for ( let e = 0; e < E; e ++ ) {

		const s = Math.floor( rnd() * N );
		let t = s;
		while ( t === s ) {

			t = Math.floor( rnd() * N );
			if ( rnd() < 0.7 && cluster[ t ] !== cluster[ s ] ) t = s; // 70 % of the edges stay inside a cluster

		}

		edgeData.set( [ s, t, cluster[ s ], cluster[ t ] ], 4 * e );

	}

	// ── GPU buffers: current and previous positions (the app would add the *Prev buffers) ──────
	const B = {
		nodePos: instancedArray( N, 'vec4' ).setName( 'nodePos' ),
		nodePrev: instancedArray( N, 'vec4' ).setName( 'nodePrev' ),
		nodeColor: instancedArray( N, 'vec4' ).setName( 'nodeColor' ),
		clusterPos: instancedArray( K, 'vec4' ).setName( 'clusterPos' ),
		clusterPrev: instancedArray( K, 'vec4' ).setName( 'clusterPrev' ),
		edgeInfo: instancedArray( E, 'uvec4' ).setName( 'edgeInfo' ),
	};
	B.nodeColor.value.array.set( colors );
	B.edgeInfo.value.array.set( edgeData );

	const U = {
		viewProj: uniform( new THREE.Matrix4() ), // unjittered projection * view of this frame
		prevViewProj: uniform( new THREE.Matrix4() ), // ... of the previous frame
		trueMotion: uniform( 1 ), // 1: previous positions from the *Prev buffers; 0: camera-only reprojection
		frame: uniform( 0 ), // temporal seed of the stochastic modes (0 = frozen pattern)
		opacity: uniform( 0.4 ), // per-edge opacity
		halfWidth: uniform( 0.1 ), // world-space half width
		minPx: uniform( 1 ), // minimum full width in output pixels
		maxPx: uniform( 16 ),
		pxScale: uniform( 1 ), // pass pixels per output pixel (supersampled reference: 3; half-resolution pass: 0.5 or 1)
		beta: uniform( 0.8 ), // bundling strength
		nodeRadius: uniform( 3 ),
		edgePad: uniform( 0.5 ), // ribbon AA pad in pass pixels (blend / a2c / mask / hash profile)
	};

	/** Writes node positions at motion phase `phase` (0 = rest) and keeps the previous ones. */
	const motion = { phase: 0, amplitude: 0 };
	function writePositions( first = false ) {

		const pos = B.nodePos.value.array, prev = B.nodePrev.value.array;
		const cpos = B.clusterPos.value.array, cprev = B.clusterPrev.value.array;
		prev.set( pos ); cprev.set( cpos );
		cpos.fill( 0 );
		const count = new Float32Array( K );
		for ( let i = 0; i < N; i ++ ) {

			// every second node swings sideways; the others stay put (so trails are easy to see)
			const a = i % 2 === 0 ? motion.amplitude * Math.sin( motion.phase + i * 1.7 ) : 0;
			const b = i % 2 === 0 ? motion.amplitude * Math.cos( motion.phase * 0.8 + i * 0.9 ) : 0;
			pos[ 4 * i ] = rest[ 4 * i ] + a;
			pos[ 4 * i + 1 ] = rest[ 4 * i + 1 ] + b;
			pos[ 4 * i + 2 ] = rest[ 4 * i + 2 ];
			const k = cluster[ i ];
			cpos[ 4 * k ] += pos[ 4 * i ]; cpos[ 4 * k + 1 ] += pos[ 4 * i + 1 ]; cpos[ 4 * k + 2 ] += pos[ 4 * i + 2 ];
			count[ k ] ++;

		}

		for ( let k = 0; k < K; k ++ ) for ( let c = 0; c < 3; c ++ ) cpos[ 4 * k + c ] /= Math.max( 1, count[ k ] );
		if ( first ) {

			prev.set( pos ); cprev.set( cpos );

		}

		for ( const b of [ B.nodePos, B.nodePrev, B.clusterPos, B.clusterPrev ] ) b.value.needsUpdate = true;

	}

	writePositions( true );

	// ── motion vectors ─────────────────────────────────────────────────────────────────────────
	let hasPrevious = false;
	/** Call BEFORE pipeline.render(): TRAA / TAAU jitter the projection only inside render(). */
	function updateMotionUniforms( camera ) {

		camera.updateMatrixWorld();
		U.viewProj.value.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
		if ( ! hasPrevious ) commitMotionUniforms();

	}

	/** Call AFTER pipeline.render(). */
	function commitMotionUniforms() {

		U.prevViewProj.value.copy( U.viewProj.value );
		hasPrevious = true;

	}

	/** NDC motion of a world point (vertex stage). */
	const ndcMotion = ( now, before ) => {

		const a = U.viewProj.mul( vec4( now, 1 ) );
		const b = U.prevViewProj.mul( vec4( mix( now, before, U.trueMotion ), 1 ) );
		return a.xy.div( a.w ).sub( b.xy.div( b.w ) );

	};

	// ── spheres ────────────────────────────────────────────────────────────────────────────────
	function sphereMaterial( { msaa, velocity } ) {

		const vCenter = varyingProperty( 'vec3', 'vCenter' );
		const vViewPos = varyingProperty( 'vec3', 'vViewPos' );
		const vQuad = varyingProperty( 'vec2', 'vQuad' );
		const vColor = varyingProperty( 'vec4', 'vColor' ); // rgb, radius
		const vVelocity = varyingProperty( 'vec2', 'vVelocity' );

		const material = new THREE.NodeMaterial();
		material.name = 'aa.sphere.' + ( msaa ? 'a2c' : 'cut' );
		material.lights = false;
		material.alphaToCoverage = msaa;

		material.vertexNode = Fn( () => {

			const world = B.nodePos.element( instanceIndex ).xyz;
			const col = B.nodeColor.element( instanceIndex );
			const center = cameraViewMatrix.mul( vec4( world, 1 ) ).xyz.toVar();
			const R = U.nodeRadius.mul( col.w ).toVar();
			const d2 = dot( center, center );
			const d = sqrt( d2 );
			const h = R.mul( d ).div( sqrt( max( d2.sub( R.mul( R ) ), 1e-6 ) ) ); // silhouette radius in the quad plane
			const pxScale = cameraProjectionMatrix.element( 1 ).element( 1 ).mul( viewport.w ).mul( 0.5 ).div( max( center.z.negate(), cameraNear ) );
			const k = float( 1 ).add( float( 1.5 ).div( max( h.mul( pxScale ), 1e-3 ) ) ); // + 1.5 px pad
			const e = center.negate().div( max( d, 1e-6 ) );
			const right = normalize( cross( vec3( 0, 1, 0 ), e ) );
			const up = cross( e, right );
			const q = positionGeometry.xy.mul( k );
			const viewPos = center.add( right.mul( q.x ).add( up.mul( q.y ) ).mul( h ) ).toVar();
			vCenter.assign( center );
			vViewPos.assign( viewPos );
			vQuad.assign( q );
			vColor.assign( vec4( col.xyz, R ) );
			if ( velocity === 'custom' ) vVelocity.assign( ndcMotion( world, B.nodePrev.element( instanceIndex ).xyz ) ); // rigid: one vector per sphere
			return cameraProjectionMatrix.mul( vec4( viewPos, 1 ) );

		} )();

		const hit = () => {

			const rd = normalize( vViewPos );
			const b = dot( rd, vCenter );
			const m = vCenter.sub( rd.mul( b ) );
			const disc = max( vColor.w.mul( vColor.w ).sub( dot( m, m ) ), 0 );
			return rd.mul( b.sub( sqrt( disc ) ) );

		};

		material.colorNode = Fn( () => {

			const ql = length( vQuad );
			const cov = clamp( float( 1 ).sub( ql ).div( max( fwidth( ql ), 1e-4 ) ).add( 0.5 ), 0, 1 ).toVar();
			Discard( cov.lessThanEqual( msaa ? 0.002 : 0.5 ) );
			const n = normalize( hit().sub( vCenter ) );
			const light = clamp( dot( n, normalize( vec3( - 0.45, 0.65, 0.62 ) ) ).mul( 0.55 ).add( 0.45 ), 0, 1 );
			const rim = float( 1 ).sub( clamp( n.z, 0, 1 ) ).pow( 3 );
			return vec4( vColor.xyz.mul( light.mul( light ).add( 0.15 ) ).add( rim.mul( 0.5 ) ), msaa ? cov : 1 );

		} )();

		material.depthNode = Fn( () => clamp( viewZToPerspectiveDepth( hit().z, cameraNear, cameraFar ), 0, 1 ) )();

		// `output` is listed so that the same material also works in a pass WITHOUT an MRT (there
		// NodeMaterial uses material.mrtNode as it is; slots the target does not have are skipped)
		if ( velocity === 'zero' ) material.mrtNode = mrt( { output, velocity: vec4( 0, 0, 0, 1 ) } );
		else if ( velocity && velocity !== 'builtin' ) material.mrtNode = mrt( { output, velocity: vec4( vVelocity, 0, 1 ) } );
		return material;

	}

	// ── ribbons ────────────────────────────────────────────────────────────────────────────────
	const bezier = ( nodeBuf, clusterBuf, edge, t ) => {

		const p0 = nodeBuf.element( edge.x ).xyz, p3 = nodeBuf.element( edge.y ).xyz;
		const p1 = mix( p0, clusterBuf.element( edge.z ).xyz, U.beta ), p2 = mix( p3, clusterBuf.element( edge.w ).xyz, U.beta );
		const s = t.oneMinus();
		const position = p0.mul( s.mul( s ).mul( s ) ).add( p1.mul( s.mul( s ).mul( t ).mul( 3 ) ) ).add( p2.mul( s.mul( t ).mul( t ).mul( 3 ) ) ).add( p3.mul( t.mul( t ).mul( t ) ) );
		const tangent = p1.sub( p0 ).mul( s.mul( s ).mul( 3 ) ).add( p2.sub( p1 ).mul( s.mul( t ).mul( 6 ) ) ).add( p3.sub( p2 ).mul( t.mul( t ).mul( 3 ) ) );
		return { position, tangent };

	};

	/**
	 * @param {object} o
	 * @param {'blend'|'a2c'|'mask'|'hash'|'alphahash'|'opaque'} o.mode
	 * @param {number} o.samples  MSAA samples of the pass the material is drawn in (mask mode)
	 * @param {false|'custom'|'builtin'|'zero'} o.velocity  what goes into the 'velocity' MRT slot
	 * @param {boolean} o.dither  a2c: add ±half a coverage step of per-edge noise before quantization
	 */
	function ribbonMaterial( { mode, samples = 4, velocity = false, dither = false } ) {

		const vColor = varyingProperty( 'vec4', 'vEdgeColor' ); // rgb, opacity * sub-pixel coverage
		const vProfile = varyingProperty( 'vec2', 'vEdgeProfile' ); // signed distance from the centre line (px), half width (px)
		const vSeed = varyingProperty( 'float', 'vEdgeSeed' ); // per-edge noise offset
		const vVelocity = varyingProperty( 'vec2', 'vEdgeVelocity' );
		const hard = mode === 'opaque';

		const material = new THREE.NodeMaterial();
		material.name = 'aa.ribbon.' + mode;
		material.lights = false;
		material.side = THREE.DoubleSide;
		material.transparent = mode === 'blend';
		material.depthWrite = mode !== 'blend';
		material.alphaToCoverage = mode === 'a2c';
		// three's built-in hashed alpha (Wyman 2017) hashes positionLocal, which for this geometry is
		// the strip parameter ( t, side, 0 ): a pattern anchored to the ribbon, constant in time
		material.alphaHash = mode === 'alphahash';

		material.vertexNode = Fn( () => {

			const edge = B.edgeInfo.element( instanceIndex ).toVar();
			const t = positionGeometry.x, side = positionGeometry.y;
			const now = bezier( B.nodePos, B.clusterPos, edge, t );
			const position = now.position.toVar();

			// screen-space expansion perpendicular to the projected tangent (as in EdgeRenderer.js)
			const clip = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( position, 1 ) ) ).toVar();
			const dclip = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( now.tangent, 0 ) ) ).toVar();
			const vp = viewport.zw;
			const sT = dclip.xy.mul( clip.w ).sub( clip.xy.mul( dclip.w ) ).mul( vp ).toVar();
			const sLen = length( sT );
			const dir = select( sLen.greaterThan( 1e-30 ), sT.div( max( sLen, 1e-30 ) ), vec2( 1, 0 ) );
			const normal = vec2( dir.y.negate(), dir.x );
			const projScale = abs( cameraProjectionMatrix.element( 1 ).element( 1 ) ).mul( vp.y ).mul( 0.5 );
			const half0 = clamp( U.halfWidth.mul( projScale ).div( max( clip.w, 1e-5 ) ), U.minPx.mul( U.pxScale ).mul( 0.5 ), U.maxPx.mul( U.pxScale ).mul( 0.5 ) ).toVar();
			const coverage = clamp( half0.mul( 2 ).div( U.pxScale ), 0, 1 ); // sub-pixel ribbons: 1 px wide, fainter
			const halfPx = max( half0, U.pxScale.mul( 0.5 ) );
			const geomHalf = hard ? halfPx : halfPx.add( U.edgePad );
			const offset = normal.mul( side.mul( geomHalf ) ).mul( 2 ).div( vp );

			const cS = B.nodeColor.element( edge.x ).xyz, cT = B.nodeColor.element( edge.y ).xyz;
			vColor.assign( vec4( mix( cS, cT, t ).mul( 1.25 ), hard ? 1 : U.opacity.mul( coverage ) ) );
			vProfile.assign( vec2( side.mul( geomHalf ), halfPx ) );
			vSeed.assign( fract( float( instanceIndex ).mul( 0.61803398875 ) ) );
			// motion of the centre line: the screen-space offset is the same in both frames and cancels
			if ( velocity === 'custom' ) vVelocity.assign( ndcMotion( position, bezier( B.nodePrev, B.clusterPrev, edge, t ).position ) );
			return vec4( clip.xy.add( offset.mul( max( clip.w, 0 ) ) ), clip.z, clip.w );

		} )();

		material.colorNode = Fn( () => {

			const d = abs( vProfile.x ), halfPx = vProfile.y;
			const r = min( d.div( halfPx ), 1 );
			const profile = clamp( halfPx.add( U.edgePad ).sub( d ).div( max( U.edgePad.mul( 2 ), 1e-3 ) ), 0, 1 ); // 1 px box filter of the edge
			const tube = mix( float( 0.55 ), float( 1 ), sqrt( max( r.mul( r ).oneMinus(), 0 ) ) );
			const rgb = vColor.xyz.mul( tube );
			const alpha = vColor.w.mul( profile ).toVar();

			// per-pixel, per-edge (and per-frame while U.frame advances) uniform noise in [0,1)
			const noise = fract( interleavedGradientNoise( screenCoordinate.xy.add( U.frame.mul( 5.588238 ) ) ).add( vSeed ) );

			if ( mode === 'hash' ) {

				Discard( alpha.lessThanEqual( noise ) ); // survives with probability alpha
				return vec4( rgb, 1 );

			}

			if ( mode === 'mask' ) {

				// stochastic transparency (Enderton et al. 2010): cover round( alpha * samples ) of
				// the samples (stochastic rounding), starting at a per-edge / per-pixel random sample
				const n = uint( floor( alpha.mul( samples ).add( noise ) ) ).toVar();
				Discard( n.equal( uint( 0 ) ) );
				const rot = uint( floor( fract( noise.mul( 7.31 ).add( vSeed.mul( 3.7 ) ) ).mul( samples ) ) );
				const full = uint( ( 1 << samples ) - 1 );
				const bits = uint( 1 ).shiftLeft( min( n, uint( samples ) ) ).sub( uint( 1 ) );
				sampleMask.assign( bits.shiftLeft( rot ).bitOr( bits.shiftRight( uint( samples ).sub( rot ) ) ).bitAnd( full ) );
				return vec4( rgb, 1 );

			}

			if ( mode === 'a2c' && dither ) alpha.addAssign( noise.sub( 0.5 ).div( samples ) );
			return vec4( rgb, hard ? 1 : alpha );

		} )();

		// 'builtin' leaves the slot to the pass MRT (three's `velocity` node: WRONG for these
		// materials, kept to show what happens); 'zero' writes no motion at all
		if ( velocity === 'zero' ) material.mrtNode = mrt( { output, velocity: vec4( 0, 0, 0, 1 ) } );
		else if ( velocity && velocity !== 'builtin' ) material.mrtNode = mrt( { output, velocity: vec4( vVelocity, 0, 1 ) } );
		return material;

	}

	// ── scene graph ────────────────────────────────────────────────────────────────────────────
	const scene = new THREE.Scene();
	scene.background = new THREE.Color( 0x020308 );
	const sphereMesh = new THREE.Mesh( quadGeometry(), null );
	sphereMesh.count = N;
	sphereMesh.frustumCulled = false;
	const geometry = ribbonGeometry( S );
	geometry.instanceCount = E;
	const ribbonMesh = new THREE.Mesh( geometry, null );
	ribbonMesh.frustumCulled = false;
	ribbonMesh.renderOrder = - 1; // blended ribbons are still depth tested against the spheres
	scene.add( sphereMesh, ribbonMesh );

	const cache = new Map();
	/** Picks (and caches) the materials for a strategy: { edges, samples, velocity: false|'custom'|'builtin'|'zero', dither }. */
	function setMode( { edges = 'blend', samples = 4, velocity = false, dither = false } ) {

		const msaa = samples > 1;
		const get = ( key, make ) => cache.get( key ) || cache.set( key, make() ).get( key );
		sphereMesh.material = get( `sphere:${msaa}:${velocity}`, () => sphereMaterial( { msaa, velocity } ) );
		ribbonMesh.material = get( `ribbon:${edges}:${samples}:${velocity}:${dither}`, () => ribbonMaterial( { mode: edges, samples, velocity, dither } ) );

	}

	/** Disposes cached materials that are not on a mesh (releases their render objects / pipelines). */
	function disposeUnusedMaterials() {

		for ( const [ key, m ] of cache ) if ( m !== sphereMesh.material && m !== ribbonMesh.material ) {

			m.dispose();
			cache.delete( key );

		}

	}

	function dispose() {

		for ( const m of cache.values() ) m.dispose();
		sphereMesh.geometry.dispose(); geometry.dispose();
		for ( const b of Object.values( B ) ) b.value.dispose?.();

	}

	return { scene, buffers: B, uniforms: U, motion, writePositions, updateMotionUniforms, commitMotionUniforms, setMode, disposeUnusedMaterials, sphereMesh, ribbonMesh, materials: cache, dispose, N, E, S };

}
