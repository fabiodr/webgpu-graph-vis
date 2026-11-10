import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeGraph, serializeGraph, parseGraphJson } from '../src/data/format.js';
import { generate, listModels, resolveParams, MODELS, pseudoWords, finishEdges } from '../src/data/generators.js';

function assertCanonicalEdges( g ) {

	const { nodeCount: n, edges } = g;
	assert.ok( edges instanceof Uint32Array );
	for ( let k = 0; k < edges.length; k += 2 ) {

		const a = edges[ k ], b = edges[ k + 1 ];
		assert.ok( a < b && b < n, `edge ${a}-${b}` );
		if ( k ) {

			const pa = edges[ k - 2 ], pb = edges[ k - 1 ];
			assert.ok( pa < a || ( pa === a && pb < b ), 'sorted and unique' );

		}

	}

	// already normalized: normalizeGraph keeps every edge
	assert.equal( normalizeGraph( g ).edges.length, edges.length );

}

function degrees( g ) {

	const d = new Uint32Array( g.nodeCount );
	for ( const x of g.edges ) d[ x ] ++;
	return d;

}

const SMALL = {
	hsbm: { n: 1200, avgDegree: 12 },
	lfr: { n: 1500, avgDegree: 12 },
	ba: { n: 1000, m: 3 },
	ws: { n: 800, k: 8, beta: 0.1 },
	rgg3d: { n: 1500, avgDegree: 10 },
	lattice3d: { nx: 6, ny: 5, nz: 4 },
	caveman: { cliques: 10, cliqueSize: 6, rewire: 0.1 },
	galaxy: { n: 1200, groups: 4, systems: 5 },
	geosf: { n: 1500, avgDegree: 8 },
	mesh: { nu: 12, nv: 9 },
};

for ( const m of MODELS ) {

	test( `${m.id}: canonical, deterministic, seed-sensitive`, () => {

		const spec = { model: m.id, params: SMALL[ m.id ], seed: 3 };
		const a = generate( spec ), b = generate( spec );
		assertCanonicalEdges( a );
		assert.deepEqual( a.edges, b.edges );
		assert.deepEqual( a.labels, b.labels );
		assert.equal( a.format, 'wgv-graph' );
		assert.equal( a.hierarchy, null );
		assert.equal( a.meta.generator, m.id );
		assert.equal( a.meta.seed, 3 );
		assert.equal( a.meta.stats.edges, a.edges.length / 2 );
		assert.equal( a.meta.stats.nodes, a.nodeCount );
		assert.ok( a.name && a.description );
		if ( m.id !== 'lattice3d' && m.id !== 'mesh' ) {

			const c = generate( { ...spec, seed: 4 } );
			assert.ok( c.edges.length !== a.edges.length || c.edges.some( ( x, i ) => x !== a.edges[ i ] ), 'seed changes the graph' );

		}

		// round trip through the on-disk format
		const back = parseGraphJson( JSON.parse( serializeGraph( a ) ) );
		assert.deepEqual( Array.from( back.edges ), Array.from( a.edges ) );
		assert.equal( back.nodeCount, a.nodeCount );

	} );

}

test( 'hsbm: degree, mixing and planted nested levels', () => {

	const g = generate( { model: 'hsbm', params: { n: 4000, avgDegree: 16, depth: 3, branching: 4, mixing: 0.25 }, seed: 1 } );
	const s = g.meta.stats;
	assert.ok( Math.abs( s.avgDegree - 16 ) / 16 < 0.03, `avg degree ${s.avgDegree}` );
	assert.ok( Math.abs( s.mixing - 0.25 ) < 0.04, `leaf mixing ${s.mixing}` );
	assert.equal( s.isolated, 0 );
	const levels = g.planted.levels;
	assert.equal( levels.length, 3 );
	assert.deepEqual( s.blocksPerLevel, [ 64, 16, 4 ] );
	for ( let l = 0; l + 1 < levels.length; l ++ ) {

		const up = new Map();
		for ( let i = 0; i < g.nodeCount; i ++ ) {

			if ( up.has( levels[ l ][ i ] ) ) assert.equal( up.get( levels[ l ][ i ] ), levels[ l + 1 ][ i ] );
			else up.set( levels[ l ][ i ], levels[ l + 1 ][ i ] );

		}

	}

	// per-level mixing array: no edges leave the top-level blocks when the last value is 0
	const h = generate( { model: 'hsbm', params: { n: 2000, avgDegree: 10, depth: 2, branching: 4, mixing: [ 0.3, 0 ] }, seed: 1 } );
	const top = h.planted.levels[ 1 ];
	for ( let k = 0; k < h.edges.length; k += 2 ) assert.equal( top[ h.edges[ k ] ], top[ h.edges[ k + 1 ] ] );
	// planted ground truth is not serialized
	assert.ok( ! ( 'planted' in JSON.parse( serializeGraph( g ) ) ) );

} );

test( 'lfr: degrees, power-law tail, mixing μ', () => {

	const g = generate( { model: 'lfr', params: { n: 5000, avgDegree: 20, maxDegree: 120, mu: 0.3, minCommunity: 20, maxCommunity: 200 }, seed: 2 } );
	const s = g.meta.stats;
	assert.ok( Math.abs( s.avgDegree - 20 ) / 20 < 0.1, `avg degree ${s.avgDegree}` );
	assert.ok( Math.abs( s.mixing - 0.3 ) < 0.06, `mixing ${s.mixing}` );
	assert.ok( s.maxDegree <= 120 && s.maxDegree > 60 );
	assert.ok( s.smallestCommunity >= 20 && s.largestCommunity <= 200 );
	const d = Array.from( degrees( g ) ).sort( ( a, b ) => a - b );
	assert.ok( d[ Math.floor( d.length * 0.5 ) ] < 20, 'median below mean (heavy tail)' );

} );

test( 'ba: m edges per node, hubs', () => {

	const n = 3000, m = 4;
	const g = generate( { model: 'ba', params: { n, m }, seed: 1 } );
	assert.equal( g.edges.length / 2, ( ( m + 1 ) * m ) / 2 + ( n - m - 1 ) * m );
	const d = degrees( g );
	assert.ok( d.every( ( x ) => x >= m ) );
	assert.ok( g.meta.stats.maxDegree > 15 * m, `max degree ${g.meta.stats.maxDegree}` );
	assert.equal( g.meta.stats.components, 1 );

} );

test( 'ws: ring lattice degree k, rewiring keeps the edge count', () => {

	const g0 = generate( { model: 'ws', params: { n: 1000, k: 10, beta: 0 }, seed: 1 } );
	assert.equal( g0.edges.length / 2, 5000 );
	assert.ok( degrees( g0 ).every( ( x ) => x === 10 ) );
	const g = generate( { model: 'ws', params: { n: 1000, k: 10, beta: 0.2 }, seed: 1 } );
	assert.equal( g.edges.length / 2, 5000 );
	let far = 0;
	for ( let k = 0; k < g.edges.length; k += 2 ) {

		const dd = Math.abs( g.edges[ k ] - g.edges[ k + 1 ] );
		if ( Math.min( dd, 1000 - dd ) > 5 ) far ++;

	}

	assert.ok( Math.abs( far / 5000 - 0.2 ) < 0.04, `rewired fraction ${far / 5000}` );

} );

test( 'rgg3d: average degree close to target (cube and torus)', () => {

	for ( const periodic of [ false, true ] ) {

		const g = generate( { model: 'rgg3d', params: { n: 4000, avgDegree: 12, periodic }, seed: 1 } );
		assert.ok( Math.abs( g.meta.stats.avgDegree - 12 ) / 12 < 0.08, `periodic=${periodic}: ${g.meta.stats.avgDegree}` );

	}

} );

test( 'lattice3d: exact edge counts, coordinate labels', () => {

	const g = generate( { model: 'lattice3d', params: { nx: 5, ny: 4, nz: 3 } } );
	assert.equal( g.nodeCount, 60 );
	assert.equal( g.edges.length / 2, 4 * 4 * 3 + 5 * 3 * 3 + 5 * 4 * 2 );
	assert.equal( g.labels[ 0 ], '0,0,0' );
	assert.equal( g.labels[ 59 ], '4,3,2' );
	const t = generate( { model: 'lattice3d', params: { nx: 5, ny: 4, nz: 3, periodic: true } } );
	assert.equal( t.edges.length / 2, 3 * 60 );
	assert.ok( degrees( t ).every( ( x ) => x === 6 ) );

} );

test( 'caveman: ring of cliques', () => {

	const L = 12, K = 7;
	const g = generate( { model: 'caveman', params: { cliques: L, cliqueSize: K } } );
	assert.equal( g.nodeCount, L * K );
	assert.equal( g.edges.length / 2, L * ( ( K * ( K - 1 ) ) / 2 ) );
	assert.equal( g.meta.stats.components, 1 );
	assert.equal( g.planted.levels[ 0 ][ K ], 1 );

} );

test( 'galaxy: hubs with satellites and moons, two planted levels, connected', () => {

	const p = { n: 3000, groups: 6, systems: 8, moons: 0.3, mesh: 0.15, bridges: 2 };
	const g = generate( { model: 'galaxy', params: p, seed: 2 } );
	assert.equal( g.nodeCount, 3000 );
	assert.equal( g.meta.stats.components, 1 );
	assert.equal( g.meta.stats.isolated, 0 );
	const [ system, group ] = g.planted.levels;
	assert.equal( new Set( system ).size, 48 );
	assert.equal( new Set( group ).size, 6 );
	// systems are nested in groups
	const parent = new Map();
	for ( let i = 0; i < g.nodeCount; i ++ ) {

		if ( ! parent.has( system[ i ] ) ) parent.set( system[ i ], group[ i ] );
		assert.equal( parent.get( system[ i ] ), group[ i ] );

	}

	// the hub of every system (its first node) has the highest degree of the system, and most
	// edges stay inside a system
	const d = degrees( g );
	const top = new Map();
	for ( let i = 0; i < g.nodeCount; i ++ ) if ( ! top.has( system[ i ] ) || d[ i ] > d[ top.get( system[ i ] ) ] ) top.set( system[ i ], i );
	let hubsFirst = 0;
	for ( const [ sys, i ] of top ) if ( i === system.indexOf( sys ) ) hubsFirst ++;
	assert.ok( hubsFirst >= 46, `${hubsFirst} of 48 hubs lead their system` );
	let inside = 0;
	for ( let k = 0; k < g.edges.length; k += 2 ) if ( system[ g.edges[ k ] ] === system[ g.edges[ k + 1 ] ] ) inside ++;
	assert.ok( inside / ( g.edges.length / 2 ) > 0.9 );
	assert.ok( g.meta.stats.moons > 500 && g.meta.stats.moons < 1200, `${g.meta.stats.moons} moons` );
	// no moons, no mesh: every system is a star, the graph is a tree plus the links between hubs
	const stars = generate( { model: 'galaxy', params: { ...p, moons: 0, mesh: 0 }, seed: 2 } );
	assert.equal( stars.meta.stats.moons, 0 );
	const ds = degrees( stars );
	let leaves = 0;
	for ( const x of ds ) if ( x === 1 ) leaves ++;
	assert.equal( leaves, 3000 - 48 );
	// tiny requests still give every system a hub and a satellite
	const tiny = generate( { model: 'galaxy', params: { n: 100, groups: 16, systems: 40 } } );
	assert.equal( tiny.nodeCount, 2 * 16 * 40 );
	assert.equal( tiny.meta.stats.components, 1 );

} );

test( 'geosf: exactly the pairs within reach, power-law degrees, target average degree', async () => {

	const { rng } = await import( '../src/graph/prepare.js' );
	for ( const params of [ { n: 900, avgDegree: 8, exponent: 2.6, dimension: 2, periodic: false }, { n: 700, avgDegree: 10, exponent: 2.3, dimension: 3, periodic: true }, { n: 600, avgDegree: 30, exponent: 3.5, dimension: 2, periodic: true } ] ) {

		const seed = 5;
		const g = generate( { model: 'geosf', params, seed } );
		assertCanonicalEdges( g );
		// the same stream as the generator: positions, then weights
		const { n, dimension: d } = params;
		const rand = rng( seed );
		const pos = new Float64Array( d * n );
		for ( let k = 0; k < d * n; k ++ ) pos[ k ] = rand();
		const cap = Math.pow( n, 1 / ( params.exponent - 1 ) );
		const w = new Float64Array( n );
		for ( let i = 0; i < n; i ++ ) w[ i ] = Math.min( cap, Math.pow( 1 - rand(), - 1 / ( params.exponent - 1 ) ) );
		const c = g.meta.stats.threshold;
		const expected = [];
		for ( let i = 0; i < n; i ++ ) for ( let j = i + 1; j < n; j ++ ) {

			let dist2 = 0;
			for ( let k = 0; k < d; k ++ ) {

				let dx = Math.abs( pos[ d * i + k ] - pos[ d * j + k ] );
				if ( params.periodic && dx > 0.5 ) dx = 1 - dx;
				dist2 += dx * dx;

			}

			const r = Math.sqrt( dist2 );
			if ( ( d === 3 ? r * r * r : dist2 ) <= c * w[ i ] * w[ j ] ) expected.push( i, j );

		}

		assert.deepEqual( Array.from( g.edges ), expected, `all pairs within reach, d = ${d}${params.periodic ? ', periodic' : ''}` );
		assert.ok( Math.abs( g.meta.stats.avgDegree / params.avgDegree - 1 ) < 0.1, `average degree ${g.meta.stats.avgDegree} for ${params.avgDegree}` );

	}

	// hubs: the heaviest node has far more neighbours than the average; degree follows weight
	const big = generate( { model: 'geosf', params: { n: 8000, avgDegree: 8, exponent: 2.5 }, seed: 1 } );
	assert.ok( Math.abs( big.meta.stats.avgDegree / 8 - 1 ) < 0.08, `average degree ${big.meta.stats.avgDegree}` );
	assert.ok( big.meta.stats.maxDegree > 25 * 8, `max degree ${big.meta.stats.maxDegree}` );
	const sorted = Array.from( degrees( big ) ).sort( ( a, b ) => b - a );
	assert.ok( sorted[ 80 ] > 3 * 8, 'the top percent has well above the average degree' );
	assert.ok( sorted[ 4000 ] <= 8, 'the median is at most the average' );

} );

test( 'mesh: sheet, tube and torus, triangles or quads', () => {

	const count = ( params ) => {

		const g = generate( { model: 'mesh', params } );
		assert.equal( g.meta.stats.components, 1 );
		return { g, edges: g.edges.length / 2, d: degrees( g ) };

	};

	const nu = 14, nv = 9;
	const torus = count( { nu, nv, wrapU: true, wrapV: true, triangles: true } );
	assert.equal( torus.edges, 3 * nu * nv );
	assert.ok( torus.d.every( ( x ) => x === 6 ), 'every node of a triangulated torus has 6 neighbours' );
	const quadTorus = count( { nu, nv, wrapU: true, wrapV: true, triangles: false } );
	assert.equal( quadTorus.edges, 2 * nu * nv );
	assert.ok( quadTorus.d.every( ( x ) => x === 4 ) );
	const tube = count( { nu, nv, wrapU: true, wrapV: false, triangles: true } );
	assert.equal( tube.edges, nu * nv + 2 * nu * ( nv - 1 ) );
	const sheet = count( { nu, nv, wrapU: false, wrapV: false, triangles: true } );
	assert.equal( sheet.edges, ( nu - 1 ) * nv + nu * ( nv - 1 ) + ( nu - 1 ) * ( nv - 1 ) );
	const quadSheet = count( { nu, nv, wrapU: false, wrapV: false, triangles: false } );
	assert.equal( quadSheet.edges, ( nu - 1 ) * nv + nu * ( nv - 1 ) );
	assert.equal( sheet.g.labels[ nu + 2 ], '2,1' );
	assert.match( torus.g.name, /torus/ );
	assert.match( tube.g.name, /tube/ );
	assert.match( sheet.g.name, /sheet/ );

} );

test( 'speed: 20k nodes / 300k-400k edges well under 2 s', () => {

	for ( const spec of [
		{ model: 'hsbm', params: { n: 20000, avgDegree: 30, depth: 3, branching: 6 } },
		{ model: 'lfr', params: { n: 20000, avgDegree: 40, maxDegree: 300, minCommunity: 30, maxCommunity: 600 } },
		{ model: 'ba', params: { n: 20000, m: 15 } },
		{ model: 'rgg3d', params: { n: 20000, avgDegree: 30 } },
	] ) {

		const t0 = performance.now();
		const g = generate( spec );
		const ms = performance.now() - t0;
		assert.ok( g.edges.length / 2 > 250000, `${spec.model}: ${g.edges.length / 2} edges` );
		assert.ok( ms < 2000, `${spec.model}: ${ms.toFixed( 0 )} ms` );

	}

} );

test( 'listModels: UI form description', () => {

	const models = listModels();
	assert.equal( models.length, MODELS.length );
	for ( const m of models ) {

		assert.ok( m.id && m.name && m.label && m.description );
		assert.ok( m.params.some( ( p ) => p.key === 'seed' ) );
		for ( const p of m.params ) {

			assert.ok( [ 'int', 'float', 'bool' ].includes( p.type ), p.key );
			assert.equal( p.name, p.key );
			assert.equal( p.default, p.value );
			if ( p.type !== 'bool' ) assert.ok( p.min <= p.value && p.value <= p.max && p.step > 0, `${m.id}.${p.key}` );

		}

		// defaults generate a valid graph
		const d = Object.fromEntries( m.params.map( ( p ) => [ p.key, p.default ] ) );
		const g = generate( { model: m.id, params: d } );
		assert.ok( g.nodeCount > 0 && g.edges.length > 0 );

	}

	// copies: editing the result does not change MODELS
	models[ 0 ].params[ 0 ].value = - 1;
	assert.notEqual( MODELS[ 0 ].params[ 0 ].value, - 1 );

} );

test( 'resolveParams clamps and fills defaults; unknown models throw', () => {

	const p = resolveParams( 'hsbm', { n: 10, avgDegree: '7.5', depth: 9.6, mixing: [ 0.2, 5 ] } );
	assert.equal( p.n, 50 );
	assert.equal( p.avgDegree, 7.5 );
	assert.equal( p.depth, 4 );
	assert.deepEqual( p.mixing, [ 0.2, 0.9 ] );
	assert.equal( p.branching, 5 );
	assert.equal( resolveParams( 'rgg3d', { periodic: 1 } ).periodic, true );
	assert.throws( () => generate( { model: 'nope' } ), /Unknown generator/ );

} );

test( 'labels: pseudo-words for small graphs (auto), none for large; seed in params works', () => {

	const small = generate( { model: 'ba', params: { n: 300, m: 2 } } );
	assert.equal( small.labels.length, 300 );
	assert.equal( new Set( small.labels ).size, 300 );
	assert.equal( generate( { model: 'ba', params: { n: 3000, m: 2 } } ).labels, null );
	assert.equal( generate( { model: 'ba', params: { n: 300, m: 2 }, labels: 'none' } ).labels, null );
	assert.equal( generate( { model: 'ba', params: { n: 3000, m: 2 }, labels: 'words' } ).labels.length, 3000 );
	const w = pseudoWords( 5000, 2 );
	assert.equal( new Set( w ).size, 5000 );
	assert.ok( w.every( ( s ) => /^[A-Z][a-z]+( \d+)?$/.test( s ) ) );
	const a = generate( { model: 'ba', params: { n: 300, m: 2, seed: 9 } } );
	assert.equal( a.meta.seed, 9 );
	assert.deepEqual( a.edges, generate( { model: 'ba', params: { n: 300, m: 2 }, seed: 9 } ).edges );

} );

test( 'finishEdges: drops loops and duplicates, sorts', () => {

	assert.deepEqual( Array.from( finishEdges( 5, [ 3, 1, 1, 3, 2, 2, 0, 4, 1, 0 ] ) ), [ 0, 1, 0, 4, 1, 3 ] );

} );
