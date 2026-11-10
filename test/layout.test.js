// Tests for the CPU side of the GPU layout (src/gpu/forceKernels.js): the extended Barnes–Hut
// structure (virtual leaf splitting, exact masses), the resort, the CPU mirrors of the GPU passes,
// the controllers (alpha schedule, auto-scale) and the whole-tick mirror, whose results are
// compared with the reference numbers of the layout research.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';
import { synthGraph } from '../src/debug/synth.js';
import { runRequest } from '../src/data/worker.js';
import { prepareEval, layoutMetrics, bfsDistances } from '../src/graph/layoutMetrics.js';
import {
	buildStructure, resortStructure, resortTasks, runResortTask, updateTreeMasses, setMasses, nodeMassFixed, packLinks,
	aggregateCPU, repulsionCPU, linksCPU, LayoutMirror, AlphaSchedule, ScaleController, rmsRadius, flatRadius, flatTuning,
	MASS_ONE, MASS_TOTAL_MAX, LEVEL_SHIFT, NODE_MASK, MODEL,
} from '../src/gpu/forceKernels.js';

const load = ( id ) => prepareGraph( normalizeGraph( JSON.parse( readFileSync( `public/data/imsv/${id}.json`, 'utf8' ) ) ) );
// through the data pipeline (Louvain when the file has no hierarchy), as the app loads it
const loadPipeline = async ( file ) => ( await runRequest( { source: { kind: 'text', text: readFileSync( `public/data/${file}`, 'utf8' ), name: file.split( '/' ).pop() }, cluster: { mode: 'auto' } } ) ).graph;

// a hub with n leaves (no hierarchy: D = 0)
const star = ( n ) => prepareGraph( normalizeGraph( { nodeCount: n + 1, edges: Array.from( { length: 2 * n }, ( _, k ) => ( k % 2 ? ( k >> 1 ) + 1 : 0 ) ) } ) );

// brute-force repulsion of the model: F_i = −Σ β m_i m_j d / (1 + |d|²/u²)², S_i = Σ β m_i m_j / (…)²
function bruteForce( pos, N, mass, { beta = 8, unit = 30 } = {} ) {

	const f = new Float64Array( 4 * N );
	const iu2 = 1 / ( unit * unit );
	for ( let i = 0; i < N; i ++ ) for ( let j = 0; j < N; j ++ ) {

		if ( i === j ) continue;
		const dx = pos[ 4 * j ] - pos[ 4 * i ], dy = pos[ 4 * j + 1 ] - pos[ 4 * i + 1 ], dz = pos[ 4 * j + 2 ] - pos[ 4 * i + 2 ];
		const q = 1 + ( dx * dx + dy * dy + dz * dz ) * iu2;
		const w = ( beta * mass[ i ] * mass[ j ] ) / ( MASS_ONE * MASS_ONE * q * q );
		f[ 4 * i ] -= dx * w; f[ 4 * i + 1 ] -= dy * w; f[ 4 * i + 2 ] -= dz * w; f[ 4 * i + 3 ] += w;

	}

	return f;

}

function checkStructure( g, s ) {

	const { N, C, V, TT, tree, nodeOrder, nodeMass, vInfo, vChildList, rInfo, rChildList } = s;
	// nodeOrder is a permutation
	const seen = new Uint8Array( N );
	for ( let t = 0; t < N; t ++ ) seen[ nodeOrder[ t ] ] ++;
	assert.ok( seen.every( ( x ) => x === 1 ), 'nodeOrder is a permutation' );
	// tree: preorder, skips in range, slots appear in order 0..N-1; count = slots in the subtree,
	// mass = fixed-point mass of the nodes behind those slots
	let nextSlot = 0, total = 0;
	for ( let k = 0; k < TT; k ++ ) {

		const ref = tree[ 4 * k ], skip = tree[ 4 * k + 1 ], mass = tree[ 4 * k + 2 ], count = tree[ 4 * k + 3 ];
		assert.ok( skip > k && skip <= TT );
		if ( ref < N ) {

			assert.equal( ref, nextSlot ++ );
			assert.equal( skip, k + 1 );
			assert.equal( count, 1 );
			assert.equal( mass, nodeMass[ nodeOrder[ ref ] ], 'leaf mass = mass of the node in the slot' );
			total += mass;

		} else {

			assert.ok( ref - N < C + V );
			let slots = 0, m = 0;
			for ( let j = k + 1; j < skip; j ++ ) if ( tree[ 4 * j ] < N ) {

				slots ++;
				m += nodeMass[ nodeOrder[ tree[ 4 * j ] ] ];

			}

			assert.equal( slots, count, 'cluster count = slots in subtree' );
			assert.equal( m, mass, 'cluster mass = sum of its node masses' );
			const id = ref - N;
			if ( id < C ) {

				assert.equal( count, g.clusterMass[ id ] );
				assert.equal( rInfo[ 4 * id + 3 ], mass, 'rInfo mass' );

			} else assert.equal( vInfo[ 4 * ( id - C ) + 3 ], mass, 'vInfo mass' );

		}

	}

	assert.equal( nextSlot, N );
	assert.equal( rInfo[ 4 * s.root + 3 ], total, 'root mass = total mass' );
	assert.equal( s.totalMass, total );
	// virtual clusters sorted by height, children of lower height
	for ( let v = 1; v < V; v ++ ) assert.ok( vInfo[ 4 * v + 2 ] >= vInfo[ 4 * ( v - 1 ) + 2 ] );
	for ( let v = 0; v < V; v ++ ) {

		const h = vInfo[ 4 * v + 2 ];
		if ( h === 1 ) assert.ok( vInfo[ 4 * v + 1 ] <= s.leafSize );
		else for ( let k = 0; k < vInfo[ 4 * v + 1 ]; k ++ ) {

			const ch = vChildList[ vInfo[ 4 * v ] + k ] - C;
			assert.ok( vInfo[ 4 * ch + 2 ] < h );

		}

	}

	// children of real clusters: every node is reached exactly once from the root
	const reached = new Uint8Array( N );
	const visit = ( c ) => {

		const start = rInfo[ 4 * c ], n = rInfo[ 4 * c + 1 ], kind = rInfo[ 4 * c + 2 ];
		for ( let k = 0; k < n; k ++ ) {

			const ch = rChildList[ start + k ];
			if ( kind === 0 ) reached[ ch ] ++;
			else if ( kind === 2 ) visit( ch );
			else {

				const stack = [ ch - C ];
				while ( stack.length ) {

					const v = stack.pop();
					const vs = vInfo[ 4 * v ], vn = vInfo[ 4 * v + 1 ];
					if ( vInfo[ 4 * v + 2 ] === 1 ) for ( let q = 0; q < vn; q ++ ) reached[ nodeOrder[ vs + q ] ] ++;
					else for ( let q = 0; q < vn; q ++ ) stack.push( vChildList[ vs + q ] - C );

				}

			}

		}

	};

	visit( s.root );
	assert.ok( reached.every( ( x ) => x === 1 ), 'rInfo / rChildList cover every node once' );

}

for ( const id of [ 'karate', 'lesmis', 'football', 'celegansneural' ] ) {

	test( `structure ${id}`, () => {

		const g = load( id );
		for ( const leafSize of [ 4, 32 ] ) {

			const s = buildStructure( g, { leafSize } );
			checkStructure( g, s );
			resortStructure( s, g.initPos );
			checkStructure( g, s );

		}

	} );

}

test( 'masses: hub spacing exponent, incremental update after resort tasks', () => {

	const g = prepareGraph( synthGraph( { n: 1500, avgDegree: 12, branching: [ 3, 3 ] } ) );
	const avg = ( 2 * g.E ) / g.N;
	assert.equal( nodeMassFixed( 0, avg, 1 ), MASS_ONE );
	assert.equal( nodeMassFixed( avg, avg, 1 ), 2 * MASS_ONE );
	assert.equal( nodeMassFixed( avg, avg, 2 ), 4 * MASS_ONE );
	assert.equal( nodeMassFixed( 7, avg, 0 ), MASS_ONE );
	const s = buildStructure( g, { leafSize: 8, massPower: 1 } );
	assert.ok( s.V > 0 );
	// tasks one by one (as the layout does over frames): every intermediate state is consistent
	const queue = resortTasks( s );
	let changedAny = false, steps = 0;
	while ( queue.length ) {

		const r = runResortTask( s, g.initPos, queue.shift(), queue );
		changedAny = changedAny || r.changed;
		if ( r.changed ) assert.ok( r.t1 > r.t0 && r.vHi > r.vLo, 'changed tasks report the tree and vInfo ranges' );
		if ( steps ++ % 7 === 0 ) checkStructure( g, s );

	}

	assert.ok( changedAny );
	checkStructure( g, s );
	// the incremental masses equal a full recomputation
	const tree = s.tree.slice(), vInfo = s.vInfo.slice();
	updateTreeMasses( s );
	assert.deepEqual( s.tree, tree );
	assert.deepEqual( s.vInfo, vInfo );
	setMasses( s, g, 0 );
	checkStructure( g, s );
	assert.equal( s.totalMass, g.N * MASS_ONE );
	setMasses( s, g, 2 );
	checkStructure( g, s );

} );

test( 'masses: a hub that would overflow uint32 is saturated, sums stay consistent', () => {

	// 1000 leaves, p = 2: the hub is (1 + 1000/avgDeg)²·256 ≈ 6.4e7, which fits
	const g = star( 1000 );
	const avg = ( 2 * g.E ) / g.N;
	const s = buildStructure( g, { massPower: 2 } );
	assert.equal( s.massCap, Infinity );
	assert.equal( s.nodeMass[ 0 ], nodeMassFixed( 1000, avg, 2 ) );
	assert.equal( s.nodeMass[ 1 ], nodeMassFixed( 1, avg, 2 ) );
	checkStructure( g, s );
	// 10000 leaves, p = 2: the hub alone would be 6.4e9 > 2³²: it is cut so that the total fits
	const big = star( 10000 );
	const avgBig = ( 2 * big.E ) / big.N;
	assert.ok( nodeMassFixed( 10000, avgBig, 2 ) > MASS_TOTAL_MAX );
	const sb = buildStructure( big, { massPower: 2 } );
	assert.ok( Number.isFinite( sb.massCap ) );
	assert.equal( sb.nodeMass[ 0 ], sb.massCap, 'the hub sits at the cap' );
	assert.equal( sb.nodeMass[ 1 ], nodeMassFixed( 1, avgBig, 2 ), 'leaves keep their mass' );
	assert.ok( sb.totalMass <= MASS_TOTAL_MAX && sb.totalMass > MASS_TOTAL_MAX - big.N, `total ${sb.totalMass}` );
	checkStructure( big, sb ); // every cluster mass equals the sum of its (saturated) node masses
	resortStructure( sb, big.initPos );
	checkStructure( big, sb );
	// several large masses share one cap: 3 hubs, each with 6000 leaves
	{

		const edges = [];
		for ( let h = 0; h < 3; h ++ ) for ( let k = 0; k < 6000; k ++ ) edges.push( h, 3 + h * 6000 + k );
		const multi = prepareGraph( normalizeGraph( { nodeCount: 3 + 18000, edges } ) );
		const sm = buildStructure( multi, { massPower: 2 } );
		assert.ok( Number.isFinite( sm.massCap ) && sm.totalMass <= MASS_TOTAL_MAX );
		assert.ok( sm.nodeMass[ 0 ] === sm.massCap && sm.nodeMass[ 1 ] === sm.massCap && sm.nodeMass[ 2 ] === sm.massCap );
		assert.ok( MASS_TOTAL_MAX - sm.totalMass < 3, 'the cap is the highest common level that fits' );
		checkStructure( multi, sm );

	}

	// back to p = 1: nothing is cut
	setMasses( sb, big, 1 );
	assert.equal( sb.massCap, Infinity );
	assert.equal( sb.nodeMass[ 0 ], nodeMassFixed( 10000, avgBig, 1 ) );
	checkStructure( big, sb );

} );

test( 'mirror: hub spacing 2 on a star stays finite and on its extent after one reheat', () => {

	const g = star( 600 );
	assert.equal( g.D, 0 );
	const sim = new LayoutMirror( g, { scaleLag: 8, scalePeriod: 4 } );
	sim.run();
	const ext = () => rmsRadius( sim.pos, g.N ) / sim.targetRms;
	assert.ok( Math.abs( ext() - 1 ) < 0.02, `extent at p = 1: ${ext()}` );
	// what ForceLayout does when hubSpacing is set: new masses, reheat to 0.3. The kernel unit
	// has to move by a factor of ~4 (p 1 -> 2) and ~5 (p 2 -> 0): more than the ticks of that
	// reheat allow at ±1.5% per tick, so the schedule waits for the auto-scale
	for ( const p of [ 2, 0 ] ) {

		setMasses( sim.struct, g, p );
		aggregateCPU( g, sim.struct, sim.pos, sim.bh );
		sim.schedule.reheat( 0.3 );
		const ticks = sim.run();
		assert.ok( sim.pos.every( Number.isFinite ) );
		assert.ok( Math.abs( ext() - 1 ) < 0.02, `extent at p = ${p}: ${ext()}` );
		assert.ok( sim.schedule.waited > 0 && sim.schedule.waited <= MODEL.scaleWaitTicks && ticks < 110 + MODEL.scaleWaitTicks + 60, `waited ${sim.schedule.waited} of ${ticks} ticks` );

	}

} );

test( 'theta = 0 equals brute force; theta = 0.8 is close (synthetic 1500, D = 2)', () => {

	const g = prepareGraph( synthGraph( { n: 1500, avgDegree: 12, branching: [ 3, 3 ] } ) ); // ~167-node leaves -> split
	const s = buildStructure( g, { leafSize: 32 } );
	assert.ok( s.V > 0 );
	resortStructure( s, g.initPos );
	checkStructure( g, s );
	const bh = aggregateCPU( g, s, g.initPos );
	// conservative radii: every node lies inside all its tree ancestors' bounding spheres
	{

		const stack = [];
		for ( let k = 0; k < s.TT; k ++ ) {

			while ( stack.length && stack[ stack.length - 1 ].skip <= k ) stack.pop();
			const ref = s.tree[ 4 * k ];
			if ( ref < s.N ) {

				const node = s.nodeOrder[ ref ];
				for ( const a of stack ) {

					const o = 4 * a.id;
					const d = Math.hypot( g.initPos[ 4 * node ] - bh[ o ], g.initPos[ 4 * node + 1 ] - bh[ o + 1 ], g.initPos[ 4 * node + 2 ] - bh[ o + 2 ] );
					assert.ok( d <= bh[ o + 3 ] * ( 1 + 1e-5 ) + 1e-4, 'radius is conservative' );

				}

			} else stack.push( { id: ref - s.N, skip: s.tree[ 4 * k + 1 ] } );

		}

	}

	// centres are mass-weighted: the root centre is the centre of mass of all nodes
	{

		let x = 0, m = 0;
		for ( let i = 0; i < g.N; i ++ ) {

			x += g.initPos[ 4 * i ] * s.nodeMass[ i ]; m += s.nodeMass[ i ];

		}

		assert.ok( Math.abs( bh[ 4 * g.root ] - x / m ) < 1e-2 );

	}

	const unit = g.linkDistance;
	const ref = bruteForce( g.initPos, g.N, s.nodeMass, { unit } );
	const exact = repulsionCPU( s, g.initPos, bh, { theta: 0, unit, maxIterations: 1 << 20 } ).force;
	let maxRel = 0, maxS = 0;
	for ( let i = 0; i < g.N; i ++ ) {

		const e = Math.hypot( exact[ 4 * i ] - ref[ 4 * i ], exact[ 4 * i + 1 ] - ref[ 4 * i + 1 ], exact[ 4 * i + 2 ] - ref[ 4 * i + 2 ] );
		maxRel = Math.max( maxRel, e / Math.hypot( ref[ 4 * i ], ref[ 4 * i + 1 ], ref[ 4 * i + 2 ] ) );
		maxS = Math.max( maxS, Math.abs( exact[ 4 * i + 3 ] - ref[ 4 * i + 3 ] ) / ref[ 4 * i + 3 ] );

	}

	assert.ok( maxRel < 1e-3, `theta 0 exact (max rel ${maxRel})` );
	assert.ok( maxS < 1e-4, `theta 0 stiffness exact (max rel ${maxS})` );
	const approx = repulsionCPU( s, g.initPos, bh, { theta: 0.8, unit } );
	const rel = [];
	for ( let i = 0; i < g.N; i ++ ) {

		const e = Math.hypot( approx.force[ 4 * i ] - ref[ 4 * i ], approx.force[ 4 * i + 1 ] - ref[ 4 * i + 1 ], approx.force[ 4 * i + 2 ] - ref[ 4 * i + 2 ] );
		rel.push( e / Math.hypot( ref[ 4 * i ], ref[ 4 * i + 1 ], ref[ 4 * i + 2 ] ) );

	}

	rel.sort( ( a, b ) => a - b );
	const visits = approx.visits.reduce( ( a, b ) => a + b, 0 ) / g.N;
	// the short-range kernel has a weak far field (∝ d⁻³), so the relative error of the summed
	// repulsion is a little larger than for a 1/d force at the same theta
	assert.ok( rel[ Math.floor( 0.5 * g.N ) ] < 0.08, `median rel err ${rel[ Math.floor( 0.5 * g.N ) ]}` );
	assert.ok( visits < g.N * 0.6, `visits/node ${visits}` );

} );

test( 'D = 0 graph gets a spatial virtual tree; tiny graphs work', () => {

	const raw = synthGraph( { n: 600, avgDegree: 6, branching: [ 1 ] } );
	const g = prepareGraph( { ...raw, hierarchy: null } );
	assert.equal( g.D, 0 );
	const s = buildStructure( g, { leafSize: 16 } );
	resortStructure( s, g.initPos );
	checkStructure( g, s );
	assert.ok( s.V >= 600 / 16 );
	const bh = aggregateCPU( g, s, g.initPos );
	const exact = repulsionCPU( s, g.initPos, bh, { theta: 0, maxIterations: 1 << 20 } ).force;
	const ref = bruteForce( g.initPos, g.N, s.nodeMass );
	for ( let i = 0; i < g.N; i += 37 ) assert.ok( Math.abs( exact[ 4 * i ] - ref[ 4 * i ] ) <= 1e-3 * ( 1 + Math.abs( ref[ 4 * i ] ) ) );
	// the whole tick works without a hierarchy (no cohesion term)
	const sim = new LayoutMirror( g, { budget: 60 } );
	sim.run();
	assert.ok( sim.pos.every( Number.isFinite ) );
	assert.ok( Math.abs( rmsRadius( sim.pos, g.N ) / sim.targetRms - 1 ) < 0.05 );

	const one = prepareGraph( normalizeGraph( { nodeCount: 1, edges: [] } ) );
	const s1 = buildStructure( one );
	checkStructure( one, s1 );
	const f1 = repulsionCPU( s1, one.initPos, aggregateCPU( one, s1, one.initPos ) ).force;
	assert.equal( f1[ 0 ], 0 );
	const sim1 = new LayoutMirror( one, { budget: 20 } );
	sim1.run();
	assert.ok( sim1.pos.every( Number.isFinite ) );

} );

test( 'coincident nodes are pushed apart deterministically', () => {

	const g = load( 'karate' );
	const pos = new Float32Array( g.initPos );
	for ( let c = 0; c < 3; c ++ ) pos[ 4 * 5 + c ] = pos[ 4 * 9 + c ]; // node 5 onto node 9
	const s = buildStructure( g );
	const f = repulsionCPU( s, pos, aggregateCPU( g, s, pos ), { theta: 0 } ).force;
	const ref = bruteForce( pos, g.N, s.nodeMass );
	const diff = ( i ) => [ f[ 4 * i ] - ref[ 4 * i ], f[ 4 * i + 1 ] - ref[ 4 * i + 1 ], f[ 4 * i + 2 ] - ref[ 4 * i + 2 ] ];
	const a = diff( 5 ), b = diff( 9 );
	assert.ok( Math.hypot( ...a ) > 0, 'a force separates the pair' );
	for ( let c = 0; c < 3; c ++ ) assert.ok( Math.abs( a[ c ] + b[ c ] ) < 1e-3 * ( 1 + Math.abs( a[ c ] ) ), 'equal and opposite' ); // float32 sums
	assert.ok( f.every( Number.isFinite ) );

} );

test( 'packLinks and the links pass (attraction by LCA level, cohesion, gravity, preconditioner, heat)', () => {

	const g = load( 'lesmis' );
	const links = packLinks( g );
	const { N, D } = g;
	// levels match edgeLca for every edge, both directions
	const lca = new Map();
	for ( let e = 0; e < g.E; e ++ ) lca.set( g.edges[ 2 * e ] * N + g.edges[ 2 * e + 1 ], g.edgeLca[ e ] );
	for ( let i = 0; i < N; i ++ ) for ( let k = g.adjStart[ i ], end = k + g.degree[ i ]; k < end; k ++ ) {

		const j = links[ k ] & NODE_MASK, l = links[ k ] >>> LEVEL_SHIFT;
		assert.equal( j, g.adjList[ k ] );
		const expected = lca.get( i * N + j ) ?? lca.get( j * N + i );
		assert.equal( l, expected );

	}

	const s = buildStructure( g );
	const pos = g.initPos;
	const bh = aggregateCPU( g, s, pos );
	const lambda = 0.7, kappa = 0.5, gravity = 0.03;
	const lcaWeights = Float32Array.from( { length: 16 }, ( _, l ) => Math.pow( lambda, l ) );
	const rep = repulsionCPU( s, pos, bh, { theta: 0 } ).force;
	const vel = new Float32Array( 4 * N );
	const src = 11; // heat source
	vel[ 4 * src + 3 ] = 1;
	const out = linksCPU( g, links, pos, new Float32Array( rep ), bh, vel, { lcaWeights, kappa, gravity, heatIndex: src } );
	for ( let i = 0; i < N; i ++ ) {

		let fx = rep[ 4 * i ], W = 0, hot = 0;
		for ( let k = g.adjStart[ i ], end = k + g.degree[ i ]; k < end; k ++ ) {

			const j = g.adjList[ k ];
			let l = 0;
			while ( l < D && g.anc[ i * D + l ] !== g.anc[ j * D + l ] ) l ++;
			const w = Math.pow( lambda, l );
			fx += w * ( pos[ 4 * j ] - pos[ 4 * i ] );
			W += w;
			if ( j === src ) hot = 1;

		}

		fx += kappa * ( bh[ 4 * g.anc[ i * D ] ] - pos[ 4 * i ] ) - gravity * pos[ 4 * i ];
		const M = 1 + kappa + gravity + W + MODEL.rho * rep[ 4 * i + 3 ];
		assert.ok( Math.abs( out[ 4 * i ] - fx / M ) < 1e-3 * ( 1 + Math.abs( fx / M ) ), `F/M of node ${i}` );
		const heat = i === src ? 1 : hot ? MODEL.heatSpread : 0;
		assert.ok( Math.abs( out[ 4 * i + 3 ] - heat ) < 1e-6, `heat of node ${i}` );

	}

} );

test( 'alpha schedule: plateau, cooling, reheat', () => {

	const s = new AlphaSchedule();
	assert.equal( s.holdTicks, 60 );
	assert.equal( s.coolTicks, 140 );
	let n = 0;
	for ( ; n < 60; n ++ ) assert.equal( s.step(), 1 );
	while ( ! s.cooled && n < 1000 ) {

		s.step();
		n ++;

	}

	assert.ok( n >= 200 && n <= 202, `cooled after ${n} ticks` );
	// wait(): holds alpha at scaleWaitAlpha once it has cooled below, at most scaleWaitTicks ticks
	{

		const w = new AlphaSchedule();
		assert.equal( w.wait(), 1, 'no effect while hot' );
		while ( w.step() >= MODEL.scaleWaitAlpha );
		let held = 0;
		while ( w.wait() === MODEL.scaleWaitAlpha && held < 1000 ) {

			w.step();
			held ++;

		}

		assert.equal( held, MODEL.scaleWaitTicks );
		assert.ok( w.alpha < MODEL.scaleWaitAlpha );
		w.reheat( 0.3 );
		assert.equal( w.waited, 0 );

	}

	s.reheat( 0.3 );
	assert.equal( s.hold, 18 );
	assert.equal( s.alpha, 0.3 );
	assert.ok( ! s.cooled );
	s.budget = 100;
	assert.equal( s.holdTicks, 30 );
	s.reheat( 1 );
	n = 0;
	while ( ! s.cooled && n < 1000 ) {

		s.step();
		n ++;

	}

	assert.ok( n >= 100 && n <= 102 );
	// an alpha target keeps the simulation warm (d3 semantics)
	s.target = 0.3;
	s.reheat( 1 );
	for ( let k = 0; k < 500; k ++ ) s.step();
	assert.ok( Math.abs( s.alpha - 0.3 ) < 1e-3 && ! s.cooled );

} );

test( 'auto-scale controller: clamps, converges, corrects lagged measurements', () => {

	const c = new ScaleController( 30 );
	assert.equal( c.step( true ), 1, 'no factor before the first measurement' );
	c.measure( 200, 100 ); // twice too large
	assert.equal( c.step( true ), MODEL.scaleMin );
	assert.equal( c.step( false ), 1, 'inactive: frozen' );
	// simulate: the true rms follows the factor; measurements arrive 8 ticks late, every 4 ticks
	let rms = 200;
	const hist = [];
	for ( let t = 0; t < 400; t ++ ) {

		hist.push( [ rms, c.logScale ] );
		if ( hist.length > 9 ) hist.shift();
		if ( t % 4 === 0 ) c.measure( hist[ 0 ][ 0 ], 100, hist[ 0 ][ 1 ] );
		rms *= c.step( true );

	}

	assert.ok( Math.abs( rms / 100 - 1 ) < 0.01, `converged to the target (${rms})` );
	assert.ok( Math.abs( c.unit / 30 - rms / 200 * ( 1 / MODEL.scaleMin ) ) < 0.5 ); // unit scaled along with the layout
	assert.ok( flatRadius( 100, 77 ) >= 130 && flatRadius( 100, 20000 ) > flatRadius( 100, 2000 ) );

} );

test( 'rmsRadius: free nodes only on request', () => {

	// three free nodes on a line and one pinned node (w = 1) far away
	const pos = Float32Array.from( [ - 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 100, 0, 0, 1 ] );
	assert.ok( Math.abs( rmsRadius( pos, 4, true ) - Math.sqrt( 2 / 3 ) ) < 1e-6 );
	assert.ok( rmsRadius( pos, 4 ) > 40 );
	assert.equal( rmsRadius( Float32Array.from( [ 5, 5, 5, 1 ] ), 1, true ), 0, 'no free node: no measurement' );

} );

test( 'flat tuning: none below 300 nodes, full from 2000 nodes, follows the blend', () => {

	assert.deepEqual( flatTuning( 77, 1 ), { t: 0, beta: 1, lambdaPower: 1, kappa: 1 } );
	assert.deepEqual( flatTuning( 300, 1 ), { t: 0, beta: 1, lambdaPower: 1, kappa: 1 } );
	assert.deepEqual( flatTuning( 5000, 0 ), { t: 0, beta: 1, lambdaPower: 1, kappa: 1 } );
	const full = flatTuning( 2000, 1 );
	assert.ok( Math.abs( full.beta - MODEL.flatBeta ) < 1e-9 && Math.abs( full.lambdaPower - MODEL.flatLambdaPower ) < 1e-9 && Math.abs( full.kappa - MODEL.flatKappa ) < 1e-9 );
	assert.deepEqual( flatTuning( 20000, 1 ), flatTuning( 2000, 1 ) );
	const mid = flatTuning( 800, 1 ), half = flatTuning( 5000, 0.5 );
	assert.ok( mid.t > 0 && mid.t < 1 && mid.beta > 1 && mid.beta < MODEL.flatBeta );
	assert.ok( Math.abs( half.t - 0.5 ) < 1e-9 && Math.abs( half.beta - Math.sqrt( MODEL.flatBeta ) ) < 1e-9 );
	assert.deepEqual( flatTuning( 0, 1 ), { t: 0, beta: 1, lambdaPower: 1, kappa: 1 } );

} );

// reference numbers of the layout research (recipe "H-gpu": stress, neighbourhood preservation,
// silhouette); the mirror runs the same model with the production seed
const REFERENCE = {
	'imsv/karate.json': { stress: 0.0593, np: 0.51, sil: 0.563 },
	'imsv/lesmis.json': { stress: 0.0707, np: 0.602, sil: 0.617 },
	'generated/hsbm-500.json': { stress: 0.0971, np: 0.399, sil: 0.56 },
};

for ( const [ file, ref ] of Object.entries( REFERENCE ) ) {

	test( `whole-tick mirror reaches the reference quality: ${file}`, async () => {

		const g = await loadPipeline( file );
		// lagged scale measurements as on the GPU (readbacks arrive a few ticks late)
		const sim = new LayoutMirror( g, { scaleLag: 8, scalePeriod: 4 } );
		const ticks = sim.run();
		assert.ok( ticks >= 200 && ticks <= 202 );
		const m = layoutMetrics( prepareEval( g ), sim.pos, 4 );
		assert.equal( m.nan, 0 );
		assert.ok( Math.abs( m.ext - 1 ) < 0.05, `extent ${m.ext}` );
		assert.ok( m.stress < ref.stress * 1.05, `stress ${m.stress} vs ${ref.stress}` );
		assert.ok( m.np > ref.np - 0.03, `np ${m.np} vs ${ref.np}` );
		assert.ok( m.sil > ref.sil - 0.03, `silhouette ${m.sil} vs ${ref.sil}` );
		assert.equal( m.overlaps, 0 );

	} );

}

test( 'mirror: flat mode stays in the plane and fills its disc', async () => {

	const g = await loadPipeline( 'imsv/lesmis.json' );
	const sim = new LayoutMirror( g, { flat: true } );
	sim.run();
	let maxZ = 0;
	for ( let i = 0; i < g.N; i ++ ) maxZ = Math.max( maxZ, Math.abs( sim.pos[ 4 * i + 2 ] ) );
	assert.equal( maxZ, 0 );
	const target = Math.SQRT1_2 * flatRadius( g.layoutRadius, g.N );
	assert.ok( Math.abs( rmsRadius( sim.pos, g.N ) / target - 1 ) < 0.05 );
	const m = layoutMetrics( prepareEval( g ), sim.pos, 4 );
	assert.ok( m.stress < 0.14, `flat stress ${m.stress}` ); // research: 0.128
	assert.equal( m.overlaps, 0 );

} );

test( 'mirror: flat mode on a 2k community graph has no overlapping nodes', async () => {

	const g = await loadPipeline( 'generated/lfr-2k.json' );
	const ev = prepareEval( g );
	const run = ( flatTune ) => {

		const sim = new LayoutMirror( g, { flat: true, flatTune, scaleLag: 8, scalePeriod: 4 } );
		sim.run();
		const target = Math.SQRT1_2 * flatRadius( g.layoutRadius, g.N );
		assert.ok( Math.abs( rmsRadius( sim.pos, g.N ) / target - 1 ) < 0.05 );
		return layoutMetrics( ev, sim.pos, 4 );

	};

	const m = run( true ), plain = run( false );
	assert.equal( m.nan, 0 );
	assert.equal( m.overlaps, 0 );
	assert.ok( plain.overlaps > 0, 'the plain 3D parameters leave overlapping pairs in the plane' );
	// the price: stress within a few percent; communities at least as clear
	assert.ok( m.stress < plain.stress * 1.06, `flat stress ${m.stress} vs ${plain.stress}` );
	assert.ok( m.sil > plain.sil && m.np > plain.np - 0.01, `sil ${m.sil} vs ${plain.sil}, np ${m.np} vs ${plain.np}` );

} );

test( 'mirror: the auto-scale keeps working with pinned nodes', async () => {

	const g = await loadPipeline( 'generated/hsbm-500.json' );
	const N = g.N, R = g.layoutRadius;
	const sim = new LayoutMirror( g, { scaleLag: 8, scalePeriod: 4 } );
	sim.run();
	const freeExt = () => rmsRadius( sim.pos, N, true ) / ( Math.sqrt( 0.6 ) * R );
	const at = ( i ) => Array.from( sim.pos.subarray( 4 * i, 4 * i + 3 ) );
	const pins = [ 3, 77, 191, 304, 450 ];
	for ( const i of pins ) sim.pin( i, true );
	const held = pins.map( at );
	const stress0 = layoutMetrics( prepareEval( g ), sim.pos, 4 ).stress;
	// the size parameters still work …
	sim.setParams( { layoutSize: 1.5 } );
	sim.schedule.reheat( 0.5 );
	sim.run();
	assert.ok( Math.abs( freeExt() / 1.5 - 1 ) < 0.03, `layoutSize 1.5 with pins: ${freeExt()}` );
	// … and the repulsion strength does not change the size
	sim.setParams( { repulsion: 32 } );
	sim.schedule.reheat( 0.3 );
	sim.run();
	assert.ok( Math.abs( freeExt() / 1.5 - 1 ) < 0.03, `repulsion 32 with pins: ${freeExt()}` );
	sim.setParams( { repulsion: 8, layoutSize: 1 } );
	sim.schedule.reheat( 0.5 );
	sim.run();
	assert.ok( Math.abs( freeExt() - 1 ) < 0.03, `back to size 1: ${freeExt()}` );
	assert.deepEqual( pins.map( at ), held, 'pins hold exactly' );
	const m = layoutMetrics( prepareEval( g ), sim.pos, 4 );
	assert.equal( m.nan, 0 );
	assert.ok( m.stress < stress0 * 1.05, `stress ${m.stress} vs ${stress0}` );
	// a node pinned far outside does not shrink the rest (the scale is measured on free nodes)
	sim.pos[ 4 * 3 ] = 3 * R; sim.pos[ 4 * 3 + 1 ] = 0; sim.pos[ 4 * 3 + 2 ] = 0;
	sim.schedule.reheat( 1 );
	sim.run();
	assert.ok( Math.abs( freeExt() - 1 ) < 0.03, `free extent with an outlying pin: ${freeExt()}` );
	assert.deepEqual( at( 3 ), [ Math.fround( 3 * R ), 0, 0 ] );
	assert.ok( sim.pos.every( Number.isFinite ) );
	// more than a quarter of the nodes pinned: the scale is frozen (unit and size stay)
	for ( let i = 0; i < N; i += 2 ) sim.pin( i, true );
	assert.ok( sim.pinned > MODEL.scaleMaxPinned * N );
	const unit = sim.scale.unit, before = freeExt();
	sim.setParams( { layoutSize: 2 } );
	sim.schedule.reheat( 0.5 );
	sim.run();
	assert.equal( sim.scale.unit, unit );
	assert.ok( Math.abs( freeExt() / before - 1 ) < 0.1, `frozen: ${freeExt()} vs ${before}` );
	assert.ok( sim.pos.every( Number.isFinite ) );

} );

test( 'mirror: a drag moves the neighbourhood, not the graph; pins hold; targets are reached', async () => {

	const g = await loadPipeline( 'generated/hsbm-500.json' );
	const sim = new LayoutMirror( g );
	sim.run();
	const N = g.N, R = g.layoutRadius, D = 0.5 * R;
	const cand = Array.from( { length: N }, ( _, i ) => i ).filter( ( i ) => g.degree[ i ] >= 2 ).sort( ( a, b ) => g.degree[ a ] - g.degree[ b ] || a - b );
	const node = cand[ cand.length >> 1 ];
	const dist = bfsDistances( g, node );
	const pos0 = new Float32Array( sim.pos );
	const start = [ pos0[ 4 * node ], pos0[ 4 * node + 1 ], pos0[ 4 * node + 2 ] ];
	const u = [ 0.6, 0.64, 0.48 ];
	const alpha0 = sim.alpha;
	sim.alphaDrag = 1;
	for ( let t = 1; t <= 30; t ++ ) {

		sim.drag = { index: node, target: start.map( ( v, c ) => v + u[ c ] * D * t / 30 ) };
		sim.tick();

	}

	for ( let t = 0; t < 300; t ++ ) {

		sim.alphaDrag *= MODEL.idleDecay; // pointer rests
		sim.tick();

	}

	assert.ok( sim.alpha <= alpha0, 'the global alpha is not reheated by a drag' );
	const moved = ( i ) => Math.hypot( sim.pos[ 4 * i ] - pos0[ 4 * i ], sim.pos[ 4 * i + 1 ] - pos0[ 4 * i + 1 ], sim.pos[ 4 * i + 2 ] - pos0[ 4 * i + 2 ] );
	assert.ok( Math.abs( moved( node ) - D ) < 1e-2 * D, 'the dragged node is on its target' );
	let near = 0, nn = 0, far = 0, fn = 0;
	for ( let i = 0; i < N; i ++ ) {

		if ( dist[ i ] === 1 ) {

			near += moved( i ); nn ++;

		} else if ( dist[ i ] >= 3 && dist[ i ] !== 65535 ) {

			far += moved( i ); fn ++;

		}

	}

	assert.ok( near / nn > 0.03 * D, `neighbours follow (${near / nn / D})` );
	assert.ok( far / fn < 0.15 * D, `far nodes barely move (${far / fn / D})` ); // research: ≤ 0.13 for N ≥ 297
	// release as a pin, reheat fully: the pin holds
	sim.drag = null;
	sim.pin( node, true );
	const pinned = [ sim.pos[ 4 * node ], sim.pos[ 4 * node + 1 ], sim.pos[ 4 * node + 2 ] ];
	sim.schedule.reheat( 1 );
	sim.run();
	assert.deepEqual( [ sim.pos[ 4 * node ], sim.pos[ 4 * node + 1 ], sim.pos[ 4 * node + 2 ] ], pinned );
	assert.ok( sim.pos.every( Number.isFinite ) );
	// target layout: the critically damped spring ends on the targets without overshoot
	sim.pin( node, false );
	const targets = new Float32Array( 4 * N );
	for ( let i = 0; i < N; i ++ ) for ( let c = 0; c < 3; c ++ ) targets[ 4 * i + c ] = g.sphereDir[ 4 * i + c ] * R;
	sim.setTargets( targets, 1 );
	let prevErr = Infinity;
	for ( let t = 0; t < 240; t ++ ) {

		sim.tick();
		let err = 0;
		for ( let i = 0; i < N; i ++ ) err += Math.hypot( sim.pos[ 4 * i ] - targets[ 4 * i ], sim.pos[ 4 * i + 1 ] - targets[ 4 * i + 1 ], sim.pos[ 4 * i + 2 ] - targets[ 4 * i + 2 ] );
		if ( t > 5 ) assert.ok( err <= prevErr * ( 1 + 1e-6 ) + 1e-3, 'monotone approach' );
		prevErr = err;

	}

	assert.ok( prevErr / N < 1e-3 * R, `on target (${prevErr / N})` );

} );
