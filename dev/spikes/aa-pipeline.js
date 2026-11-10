// Spike "aa": the anti-aliasing / transparency strategies as RenderPipeline graphs, and the
// runtime switch between them. NOT production code.
//
//   STRATEGIES                          table: name -> { label, samples, edges, temporal, oit, ... }
//   buildStrategy( name, ctx, over )    -> { def, scenePass, outputNode, temporalNode, dispose() }
//   StrategySwitcher                    owns the RenderPipeline; .use( name, over ) disposes the
//                                       previous graph and installs the new one
//
// Every graph ends in  renderOutput( hdr, toneMapping, sRGB )  with
// pipeline.outputColorTransform = false, so that FXAA (which needs sRGB input) can sit after the
// transform and everything else before it.
//
// Ownership rule that makes switching leak-free: every node that owns GPU resources (PassNode,
// TRAANode, TAAUNode, RTTNode, SMAANode, SharpenNode, FSR1Node) is created here and pushed to
// `owned`; dispose() calls .dispose() on each. Nodes created implicitly by convertToTexture()
// inside the addon factories (fxaa( x ), sharpen( x ), ... when x is not a texture node) would be
// unreachable, so texture nodes are always passed in explicitly.
import * as THREE from 'three/webgpu';
import {
	pass, mrt, output, velocity, renderOutput, rtt, vec3, vec4, uniform, clamp, perspectiveDepthToViewZ, cameraNear, cameraFar,
	nodeImmutable,
} from 'three/tsl';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { taau } from 'three/addons/tsl/display/TAAUNode.js';
import { oitPass } from 'three/addons/tsl/display/OITPassNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { smaa } from 'three/addons/tsl/display/SMAANode.js';
import { sharpen } from 'three/addons/tsl/display/SharpenNode.js';
import { fsr1 } from 'three/addons/tsl/display/FSR1Node.js';

export const STRATEGIES = {
	none: { label: 'no AA + blended', samples: 0, edges: 'blend' },
	s1: { label: 'S1 MSAA4 + blended', samples: 4, edges: 'blend' },
	s2: { label: 'S2 MSAA4 + alphaToCoverage', samples: 4, edges: 'a2c' },
	s2d: { label: 'S2 MSAA4 + a2c + dither', samples: 4, edges: 'a2c', dither: true },
	s2m: { label: 'S2b MSAA4 + stochastic mask', samples: 4, edges: 'mask' },
	s3: { label: 'S3 TRAA + hashed', samples: 0, edges: 'hash', temporal: 'traa' },
	s3b: { label: 'S3 TRAA + blended', samples: 0, edges: 'blend', temporal: 'traa' },
	s3o: { label: 'S3 TRAA + opaque', samples: 0, edges: 'opaque', temporal: 'traa' },
	s4: { label: 'S4 WBOIT (MSAA4)', samples: 4, edges: 'blend', oit: true },
	s5: { label: 'S5 TAAU 0.5x + hashed', samples: 0, edges: 'hash', temporal: 'taau', scale: 0.5 },
	s5b: { label: 'S5 TAAU 0.5x + blended', samples: 0, edges: 'blend', temporal: 'taau', scale: 0.5 },
	s5s: { label: 'S5 TAAU 0.5x + blended + RCAS', samples: 0, edges: 'blend', temporal: 'taau', scale: 0.5, post: 'sharpen' },
	s5f: { label: 'S5 FSR1 from MSAA4 0.5x', samples: 4, edges: 'blend', scale: 0.5, post: 'fsr1' },
	s6: { label: 'S6 FXAA + blended', samples: 0, edges: 'blend', post: 'fxaa' },
	s6s: { label: 'S6 SMAA + blended', samples: 0, edges: 'blend', post: 'smaa' },
	s6o: { label: 'S6 FXAA + opaque', samples: 0, edges: 'opaque', post: 'fxaa' },
};

/** Fragment input depth (@builtin(position).z); TSL's screenCoordinate only exposes .xy. */
class FragCoordZNode extends THREE.Node {

	static get type() {

		return 'FragCoordZNode';

	}

	constructor() {

		super( 'float' );

	}

	generate( builder ) {

		return builder.getBuiltin( 'position', 'fragCoord', 'vec4<f32>' ) + '.z';

	}

}

const fragCoordZ = nodeImmutable( FragCoordZNode );

// Shared across rebuilds on purpose: RenderContexts are keyed by the id of the MRT node, and
// render objects (pipelines, bind groups, uniform buffers) by render context. A fresh mrt() per
// rebuild would add a render context plus one render object per material every time, and those
// are only released when the MATERIAL is disposed (measured: + 2 uniform buffers per switch).
const velocityMRT = mrt( { output, velocity } ); // `velocity`: three's node for ordinary meshes; the spike materials replace the slot through material.mrtNode

// OITPassNode builds its accumulation MRT privately (one per instance), so the instances are kept
// and only their render targets are released while another strategy is active.
const oitPasses = new Map();

/**
 * @param {string} name key of STRATEGIES
 * @param {object} ctx { scene, camera, test (createTestScene result), toneMapping }
 * @param {object} [over] overrides of the strategy definition, plus { view: 'final'|'depth'|'velocity'|'raw' }
 */
export function buildStrategy( name, ctx, over = {} ) {

	const def = { ...STRATEGIES[ name ], ...over, name };
	const { scene, camera, test } = ctx;
	const toneMapping = ctx.toneMapping ?? THREE.NeutralToneMapping;
	const owned = [];
	const own = ( node ) => ( owned.push( node ), node );
	const options = { samples: def.samples };

	// materials first: they depend on the pass they are drawn in
	test.setMode( { edges: def.edges, samples: def.samples, velocity: def.temporal ? ( def.mv ?? 'custom' ) : false, dither: !! def.dither } );

	let scenePass, hdr, temporalNode = null;
	if ( def.oit ) {

		const key = `${def.samples}:${def.oitWeight ?? 'depth'}`;
		scenePass = oitPasses.get( key );
		if ( ! scenePass ) {

			scenePass = oitPass( scene, camera, options );
			// The default weight uses positionView.z, which is wrong for materials with a custom
			// vertexNode. Equation (10) of the paper on the fragment depth instead, fitted to the
			// scene. Must be assigned before the first render.
			const zNear = uniform( def.oitNear ?? 180 ), zFar = uniform( def.oitFar ?? 460 );
			const d = clamp( perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar ).negate().sub( zNear ).div( zFar.sub( zNear ) ), 0, 1 );
			scenePass.weightNode = def.oitWeight === 'flat' ? output.a : output.a.mul( clamp( d.oneMinus().pow( 3 ).mul( 1e3 ), 1e-2, 3e3 ) );
			oitPasses.set( key, scenePass );

		}

		own( scenePass ); // dispose() only frees its render targets; the node is reused
		hdr = scenePass; // the composite is the node itself; scenePass.getTextureNode() is the pass WITHOUT the OIT layer

	} else {

		scenePass = own( pass( scene, camera, options ) );
		hdr = scenePass.getTextureNode( 'output' );

	}

	if ( def.scale ) scenePass.setResolutionScale( def.scale );

	if ( def.temporal ) {

		scenePass.setMRT( def.freshMRT ? mrt( { output, velocity } ) : velocityMRT );
		const args = [ scenePass.getTextureNode( 'output' ), scenePass.getTextureNode( 'depth' ), scenePass.getTextureNode( 'velocity' ), camera ];
		temporalNode = own( def.temporal === 'traa' ? traa( ...args ) : taau( ...args ) );
		if ( def.depthThreshold !== undefined ) temporalNode.depthThreshold = def.depthThreshold;
		if ( def.currentFrameWeight !== undefined ) temporalNode.currentFrameWeight = def.currentFrameWeight;
		if ( def.subpixel !== undefined ) temporalNode.useSubpixelCorrection = def.subpixel;
		if ( def.maxVelocityLength !== undefined ) temporalNode.maxVelocityLength = def.maxVelocityLength; // px of motion at which history is dropped
		hdr = temporalNode.getTextureNode();

	}

	// the addon factories call convertToTexture(): a PassNode would be replaced by its plain
	// getTextureNode() (losing the OIT composite), any other non-texture node by a hidden RTTNode
	// (def.oitDirect hands the OIT pass over as it is, to demonstrate that pitfall)
	const asTexture = ( node ) => ( node.isTextureNode || def.oitDirect ? node : own( rtt( node ) ) );

	let outputNode;
	if ( def.view === 'depth' ) {

		const z = scenePass.getViewZNode().negate();
		outputNode = vec4( vec3( clamp( z.sub( 200 ).div( 240 ), 0, 1 ).oneMinus() ), 1 );

	} else if ( def.view === 'velocity' ) {

		outputNode = vec4( scenePass.getTextureNode( 'velocity' ).rg.mul( def.velocityGain ?? 50 ).add( 0.5 ), 0.5, 1 );

	} else if ( def.view === 'raw' ) {

		outputNode = renderOutput( scenePass.getTextureNode( 'output' ), toneMapping, THREE.SRGBColorSpace );

	} else if ( def.post === 'fxaa' ) {

		// FXAA wants tone mapped sRGB input
		outputNode = fxaa( asTexture( renderOutput( hdr, toneMapping, THREE.SRGBColorSpace ) ) );

	} else {

		if ( def.post === 'smaa' ) hdr = own( smaa( asTexture( hdr ) ) );
		if ( def.post === 'sharpen' ) hdr = own( sharpen( asTexture( hdr ), def.sharpness ?? 0.2 ) );
		if ( def.post === 'fsr1' ) hdr = own( fsr1( asTexture( hdr ), def.sharpness ?? 0.2 ) );
		outputNode = renderOutput( hdr, toneMapping, THREE.SRGBColorSpace );

	}

	return {
		def, scenePass, temporalNode, outputNode,
		dispose() {

			for ( const node of owned ) node.dispose();

		},
	};

}

/** Owns the RenderPipeline and swaps strategies at runtime. */
export class StrategySwitcher {

	constructor( renderer, ctx ) {

		this.renderer = renderer;
		this.ctx = ctx;
		this.pipeline = new THREE.RenderPipeline( renderer );
		this.pipeline.outputColorTransform = false;
		this.current = null;

	}

	use( name, over = {} ) {

		const previous = this.current;
		this.current = buildStrategy( name, this.ctx, over );
		this.pipeline.outputNode = this.current.outputNode;
		this.pipeline.needsUpdate = true; // rebuilds the output quad's material on the next render()
		if ( previous ) previous.dispose();
		return this.current;

	}

	render() {

		this.pipeline.render();

	}

	dispose() {

		this.current?.dispose();
		this.current = null;
		this.pipeline.dispose();

	}

}
