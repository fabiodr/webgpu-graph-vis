// Seeded, deterministic graph generators. Each returns a canonical graph (see format.js) without
// a hierarchy (the data pipeline computes one with Louvain afterwards):
//   { format, version, name, description, source, nodeCount, labels, edges: Uint32Array (sorted,
//     deduplicated, a < b, no self loops), hierarchy: null,
//     meta: { generator, params, seed, stats: { nodes, edges, avgDegree, maxDegree, isolated,
//             components, ... } },
//     planted?: { levels } }   ground-truth communities (finest first), not serialized
//
// Models (ids as used in specs { model, params, seed }):
//   hsbm       hierarchical stochastic block model: nested blocks (depth × branching), a mixing
//              parameter per level (probability that an edge leaves its level-l block given it
//              left its level-(l-1) block; scalar or array, finest first), lognormal block-size
//              and degree spreads. The ideal input for bundled-edge views.
//   lfr        LFR-like benchmark (Lancichinetti, Fortunato & Radicchi 2008): power-law degrees and
//              community sizes, mixing mu; configuration model inside/between communities.
//   ba         Barabási–Albert preferential attachment (hubs, weak communities).
//   ws         Watts–Strogatz small world (ring lattice + rewiring).
//   rgg3d      random geometric graph in the unit cube (or 3-torus), grid hashing. Layout sanity.
//   lattice3d  3D grid / torus. Layout sanity check (should unfold into a box).
//   caveman    ring of cliques (connected caveman), optional relaxed rewiring.
//   galaxy     hubs and satellites in two planted levels: systems (a hub, its satellites, moons
//              around some satellites) in groups; hubs of a group linked in a ring with chords,
//              groups joined at their largest hubs. Tree-like, strong hubs, clear hierarchy.
//   geosf      geometric inhomogeneous random graph (Bringmann, Keusch & Lengler 2019, threshold
//              case): points in the unit square or cube with Pareto weights, an edge when
//              distance^d <= c · w_i · w_j. Power-law degrees with geometric locality. Sampled
//              per pair of weight layers with a grid of the layer pair's reach.
//   mesh       triangle / quad mesh of a sheet, a tube or a torus (wrap flags).
//
// API
//   MODELS                     model descriptions (id, name, label, description, params[])
//   listModels()               deep copy of MODELS, params in the shape the UI Panel expects:
//                              { key, name, label, type: 'int'|'float'|'bool', min, max, step, log?,
//                                value, default }
//   resolveParams( model, params )  defaults filled in, values clamped to [min, max]
//   generate( { model, params, seed = params.seed ?? 1, labels = 'auto', name, description } )
//                              labels: 'auto' (pseudo-word names up to 2,000 nodes; coordinates
//                              for lattices), 'words', 'none'
//   hsbm(p, seed), lfr(p, seed), barabasiAlbert(p, seed), wattsStrogatz(p, seed),
//   randomGeometric3D(p, seed), lattice3D(p), caveman(p, seed), galaxy(p, seed),
//   geometricScaleFree(p, seed), surfaceMesh(p)                     raw generators (resolved params)
//   pseudoWords( n, seed )     unique pronounceable labels

import { rng } from '../graph/prepare.js';
import { FORMAT, VERSION } from './format.js';

const SEED_PARAM = { key: 'seed', label: 'Seed', type: 'int', min: 1, max: 9999, value: 1 };

export const MODELS = [
	{
		id: 'hsbm', name: 'Hierarchical SBM', label: 'Hierarchical communities',
		description: 'Nested communities (depth × branching) with a mixing ratio per level. Best for bundled-edge views.',
		params: [
			{ key: 'n', label: 'Nodes', type: 'int', min: 50, max: 30000, step: 10, log: true, value: 2000 },
			{ key: 'avgDegree', label: 'Avg degree', type: 'float', min: 2, max: 60, step: 0.5, value: 12 },
			{ key: 'depth', label: 'Levels', type: 'int', min: 1, max: 4, value: 3 },
			{ key: 'branching', label: 'Branching', type: 'int', min: 2, max: 12, value: 5 },
			{ key: 'mixing', label: 'Mixing', type: 'float', min: 0, max: 0.9, step: 0.01, value: 0.3 },
			{ key: 'sizeSpread', label: 'Size spread', type: 'float', min: 0, max: 1.5, step: 0.05, value: 0.4 },
			{ key: 'degreeSpread', label: 'Degree spread', type: 'float', min: 0, max: 1.5, step: 0.05, value: 0.3 },
		],
	},
	{
		id: 'lfr', name: 'LFR benchmark', label: 'LFR (power-law)',
		description: 'Power-law degrees and community sizes; mu is the fraction of each node’s edges leaving its community.',
		params: [
			{ key: 'n', label: 'Nodes', type: 'int', min: 100, max: 30000, step: 10, log: true, value: 2000 },
			{ key: 'avgDegree', label: 'Avg degree', type: 'float', min: 3, max: 60, step: 0.5, value: 15 },
			{ key: 'maxDegree', label: 'Max degree', type: 'int', min: 10, max: 1000, log: true, value: 60 },
			{ key: 'mu', label: 'Mixing μ', type: 'float', min: 0, max: 0.8, step: 0.01, value: 0.25 },
			{ key: 'degreeExponent', label: 'Degree exponent', type: 'float', min: 2, max: 3.5, step: 0.05, value: 2.5 },
			{ key: 'communityExponent', label: 'Size exponent', type: 'float', min: 1, max: 2.5, step: 0.05, value: 1.5 },
			{ key: 'minCommunity', label: 'Min community', type: 'int', min: 5, max: 500, log: true, value: 20 },
			{ key: 'maxCommunity', label: 'Max community', type: 'int', min: 10, max: 2000, log: true, value: 100 },
		],
	},
	{
		id: 'ba', name: 'Barabási–Albert', label: 'Scale-free (hubs)',
		description: 'Preferential attachment: a few hubs, weak communities. Stresses bundling and hub rendering.',
		params: [
			{ key: 'n', label: 'Nodes', type: 'int', min: 50, max: 30000, step: 10, log: true, value: 2000 },
			{ key: 'm', label: 'Edges per node', type: 'int', min: 1, max: 30, value: 4 },
		],
	},
	{
		id: 'ws', name: 'Watts–Strogatz', label: 'Small world',
		description: 'Ring lattice with random rewiring: high clustering, short paths.',
		params: [
			{ key: 'n', label: 'Nodes', type: 'int', min: 50, max: 30000, step: 10, log: true, value: 2000 },
			{ key: 'k', label: 'Neighbors', type: 'int', min: 2, max: 60, step: 2, value: 10 },
			{ key: 'beta', label: 'Rewiring', type: 'float', min: 0, max: 1, step: 0.01, value: 0.05 },
		],
	},
	{
		id: 'rgg3d', name: 'Random geometric 3D', label: 'Geometric 3D',
		description: 'Random points in a cube, linked when close. A layout sanity check: should look like a solid blob.',
		params: [
			{ key: 'n', label: 'Nodes', type: 'int', min: 50, max: 30000, step: 10, log: true, value: 3000 },
			{ key: 'avgDegree', label: 'Avg degree', type: 'float', min: 2, max: 40, step: 0.5, value: 10 },
			{ key: 'periodic', label: 'Periodic (torus)', type: 'bool', value: false },
		],
	},
	{
		id: 'lattice3d', name: '3D lattice', label: 'Lattice / torus',
		description: 'Regular grid. A layout sanity check: should unfold into a box (or a torus when periodic).',
		params: [
			{ key: 'nx', label: 'Size X', type: 'int', min: 2, max: 40, value: 12 },
			{ key: 'ny', label: 'Size Y', type: 'int', min: 2, max: 40, value: 12 },
			{ key: 'nz', label: 'Size Z', type: 'int', min: 1, max: 40, value: 12 },
			{ key: 'periodic', label: 'Periodic (torus)', type: 'bool', value: false },
		],
	},
	{
		id: 'caveman', name: 'Ring of cliques', label: 'Caveman',
		description: 'Cliques joined in a ring (connected caveman); rewiring relaxes it.',
		params: [
			{ key: 'cliques', label: 'Cliques', type: 'int', min: 3, max: 2000, log: true, value: 24 },
			{ key: 'cliqueSize', label: 'Clique size', type: 'int', min: 3, max: 40, value: 8 },
			{ key: 'rewire', label: 'Rewiring', type: 'float', min: 0, max: 0.5, step: 0.01, value: 0 },
		],
	},
	{
		id: 'galaxy', name: 'Hubs and satellites', label: 'Galaxy (hubs)',
		description: 'Groups of systems: a hub with satellites, some satellites with moons of their own; hubs of a group are linked, groups meet at their largest hubs. A tree-like hierarchy with strong hubs.',
		params: [
			{ key: 'n', label: 'Nodes', type: 'int', min: 100, max: 30000, step: 10, log: true, value: 3000 },
			{ key: 'groups', label: 'Groups', type: 'int', min: 2, max: 16, value: 6 },
			{ key: 'systems', label: 'Systems per group', type: 'int', min: 2, max: 40, value: 8 },
			{ key: 'sizeSpread', label: 'Size spread', type: 'float', min: 0, max: 1.5, step: 0.05, value: 0.7 },
			{ key: 'moons', label: 'Moons', type: 'float', min: 0, max: 0.8, step: 0.01, value: 0.3 },
			{ key: 'mesh', label: 'Links between satellites', type: 'float', min: 0, max: 2, step: 0.05, value: 0.15 },
			{ key: 'bridges', label: 'Links per hub', type: 'int', min: 1, max: 6, value: 2 },
		],
	},
	{
		id: 'geosf', name: 'Geometric scale-free', label: 'Geometric scale-free',
		description: 'Points in space with power-law weights, linked when close relative to their weights (a geometric inhomogeneous random graph): hubs and locality at once.',
		params: [
			{ key: 'n', label: 'Nodes', type: 'int', min: 100, max: 30000, step: 10, log: true, value: 3000 },
			{ key: 'avgDegree', label: 'Avg degree', type: 'float', min: 2, max: 40, step: 0.5, value: 8 },
			{ key: 'exponent', label: 'Degree exponent', type: 'float', min: 2.1, max: 4, step: 0.05, value: 2.6 },
			{ key: 'dimension', label: 'Dimension', type: 'int', min: 2, max: 3, value: 2 },
			{ key: 'periodic', label: 'Periodic (torus)', type: 'bool', value: false },
		],
	},
	{
		id: 'mesh', name: 'Surface mesh', label: 'Mesh / torus',
		description: 'A triangle or quad mesh of a sheet, a tube (wrapped one way) or a torus (wrapped both ways). A layout sanity check: should unfold into that surface.',
		params: [
			{ key: 'nu', label: 'Size U', type: 'int', min: 3, max: 300, value: 60 },
			{ key: 'nv', label: 'Size V', type: 'int', min: 3, max: 300, value: 30 },
			{ key: 'wrapU', label: 'Wrap U', type: 'bool', value: true },
			{ key: 'wrapV', label: 'Wrap V', type: 'bool', value: true },
			{ key: 'triangles', label: 'Triangles', type: 'bool', value: true },
		],
	},
];

for ( const m of MODELS ) m.params.push( { ...SEED_PARAM } );

const GENERATORS = {
	hsbm: ( p, seed ) => hsbm( p, seed ),
	lfr: ( p, seed ) => lfr( p, seed ),
	ba: ( p, seed ) => barabasiAlbert( p, seed ),
	ws: ( p, seed ) => wattsStrogatz( p, seed ),
	rgg3d: ( p, seed ) => randomGeometric3D( p, seed ),
	lattice3d: ( p ) => lattice3D( p ),
	caveman: ( p, seed ) => caveman( p, seed ),
	galaxy: ( p, seed ) => galaxy( p, seed ),
	geosf: ( p, seed ) => geometricScaleFree( p, seed ),
	mesh: ( p ) => surfaceMesh( p ),
};

export function listModels() {

	return MODELS.map( ( m ) => ( {
		id: m.id, name: m.name, label: m.label, description: m.description,
		params: m.params.map( ( p ) => ( { ...p, name: p.key, default: p.value, step: p.step ?? ( p.type === 'int' ? 1 : p.type === 'float' ? 0.01 : undefined ) } ) ),
	} ) );

}

export function getModel( id ) {

	const m = MODELS.find( ( x ) => x.id === id );
	if ( ! m ) throw new Error( `Unknown generator model "${id}" (known: ${MODELS.map( ( x ) => x.id ).join( ', ' )})` );
	return m;

}

/** Defaults filled in; numbers clamped to [min, max], ints rounded, bools coerced. Extra keys kept. */
export function resolveParams( model, params = {} ) {

	const m = typeof model === 'string' ? getModel( model ) : model;
	const out = { ...params };
	for ( const p of m.params ) {

		if ( p.key === 'seed' ) continue;
		let v = params[ p.key ];
		if ( p.type === 'bool' ) {

			out[ p.key ] = v === undefined ? p.value : Boolean( v );
			continue;

		}

		if ( Array.isArray( v ) ) {

			out[ p.key ] = v.map( ( x ) => clampParam( p, x ) );
			continue;

		}

		out[ p.key ] = clampParam( p, v === undefined || v === null || v === '' ? p.value : v );

	}

	delete out.seed;
	return out;

}

function clampParam( p, v ) {

	v = Number( v );
	if ( ! Number.isFinite( v ) ) v = p.value;
	if ( p.type === 'int' ) v = Math.round( v );
	return Math.min( p.max, Math.max( p.min, v ) );

}

/**
 * Generates a graph from a spec { model, params, seed, labels, name, description }.
 * @returns canonical graph (see header)
 */
export function generate( spec ) {

	const model = getModel( spec.model );
	const params = resolveParams( model, spec.params || {} );
	const seed = ( Number( spec.seed ?? spec.params?.seed ?? 1 ) >>> 0 ) || 1;
	const t0 = now();
	const g = GENERATORS[ model.id ]( params, seed );
	const ms = now() - t0;
	const mode = spec.labels ?? 'auto';
	let labels = null;
	if ( g.labels && mode !== 'none' && ( mode !== 'auto' || g.nodeCount <= 2000 ) ) labels = g.labels;
	else if ( mode === 'words' || ( mode === 'auto' && g.nodeCount <= 2000 ) ) labels = pseudoWords( g.nodeCount, labelSeed( model.id, seed ) );
	const stats = graphStats( g.nodeCount, g.edges );
	const out = {
		format: FORMAT,
		version: VERSION,
		name: spec.name || g.name,
		description: spec.description || g.description,
		source: `Generated: ${model.name} (webgpu-graph-vis src/data/generators.js), seed ${seed}`,
		nodeCount: g.nodeCount,
		labels,
		edges: g.edges,
		hierarchy: null,
		meta: { generator: model.id, params, seed, stats: { ...stats, ...( g.stats || {} ), ms: Math.round( ms ) } },
	};
	if ( g.planted ) Object.defineProperty( out, 'planted', { value: g.planted, enumerable: false, writable: true } );
	return out;

}

// ── shared helpers ───────────────────────────────────────────────────────────────────────────

/** Growable flat pair buffer. */
class PairBuffer {

	constructor( capacity = 1024 ) {

		this.data = new Uint32Array( Math.max( 2, 2 * capacity ) );
		this.length = 0; // in pairs

	}

	push( a, b ) {

		if ( 2 * this.length + 2 > this.data.length ) {

			const next = new Uint32Array( this.data.length * 2 );
			next.set( this.data );
			this.data = next;

		}

		this.data[ 2 * this.length ] = a;
		this.data[ 2 * this.length + 1 ] = b;
		this.length ++;

	}

	view() {

		return this.data.subarray( 0, 2 * this.length );

	}

}

/** Sorted (a < b), deduplicated, loop-free flat edge array from raw pairs. */
export function finishEdges( n, pairs ) {

	const P = pairs.length >> 1;
	const keys = new Float64Array( P );
	let k = 0;
	for ( let e = 0; e < P; e ++ ) {

		let a = pairs[ 2 * e ], b = pairs[ 2 * e + 1 ];
		if ( a === b ) continue;
		if ( a > b ) {

			const t = a; a = b; b = t;

		}

		keys[ k ++ ] = a * n + b;

	}

	const sorted = keys.subarray( 0, k ).sort();
	let u = 0;
	for ( let i = 0; i < k; i ++ ) if ( i === 0 || sorted[ i ] !== sorted[ i - 1 ] ) sorted[ u ++ ] = sorted[ i ];
	const out = new Uint32Array( 2 * u );
	for ( let i = 0; i < u; i ++ ) {

		const key = sorted[ i ];
		const a = Math.floor( key / n );
		out[ 2 * i ] = a;
		out[ 2 * i + 1 ] = key - a * n;

	}

	return out;

}

/** Degree / component statistics of a flat edge array. */
export function graphStats( n, edges ) {

	const E = edges.length >> 1;
	const deg = new Uint32Array( n );
	const parent = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) parent[ i ] = i;
	const find = ( x ) => {

		while ( parent[ x ] !== x ) {

			parent[ x ] = parent[ parent[ x ] ];
			x = parent[ x ];

		}

		return x;

	};

	for ( let e = 0; e < E; e ++ ) {

		const a = edges[ 2 * e ], b = edges[ 2 * e + 1 ];
		deg[ a ] ++; deg[ b ] ++;
		const ra = find( a ), rb = find( b );
		if ( ra !== rb ) parent[ ra ] = rb;

	}

	let maxDegree = 0, isolated = 0, components = 0;
	for ( let i = 0; i < n; i ++ ) {

		if ( deg[ i ] > maxDegree ) maxDegree = deg[ i ];
		if ( deg[ i ] === 0 ) isolated ++;
		if ( find( i ) === i ) components ++;

	}

	return { nodes: n, edges: E, avgDegree: n ? round3( ( 2 * E ) / n ) : 0, maxDegree, isolated, components };

}

function round3( x ) {

	return Math.round( x * 1000 ) / 1000;

}

function gaussian( rand ) {

	// Box–Muller (one value per call keeps the stream simple and deterministic)
	const u = 1 - rand(), v = rand();
	return Math.sqrt( - 2 * Math.log( u ) ) * Math.cos( 2 * Math.PI * v );

}

/** Lognormal weights with mean 1 (sigma = 0: all ones). */
function lognormalWeights( n, sigma, rand ) {

	const w = new Float64Array( n );
	if ( sigma <= 0 ) return w.fill( 1 );
	let s = 0;
	for ( let i = 0; i < n; i ++ ) {

		w[ i ] = Math.exp( sigma * gaussian( rand ) );
		s += w[ i ];

	}

	const mean = s / n;
	for ( let i = 0; i < n; i ++ ) w[ i ] /= mean;
	return w;

}

/** Smallest index i with prefix[i + 1] > x (prefix: n + 1 cumulative weights). */
function searchPrefix( prefix, x, lo, hi ) {

	// invariant: prefix[lo] <= x < prefix[hi]
	while ( hi - lo > 1 ) {

		const mid = ( lo + hi ) >> 1;
		if ( prefix[ mid ] <= x ) lo = mid;
		else hi = mid;

	}

	return lo;

}

function fmt( n ) {

	return n.toLocaleString( 'en-US' );

}

// ── hierarchical stochastic block model ──────────────────────────────────────────────────────

export function hsbm( p, seed = 1 ) {

	const n = p.n, depth = Math.max( 1, p.depth | 0 ), branching = Math.max( 2, p.branching | 0 );
	const rand = rng( seed );
	const mix = [];
	for ( let l = 0; l < depth; l ++ ) mix.push( Array.isArray( p.mixing ) ? ( p.mixing[ Math.min( l, p.mixing.length - 1 ) ] ?? 0.3 ) : p.mixing );
	const minLeaf = 4;

	// block ranges, coarse to fine; nodes are contiguous within every block
	// rangesByLevel[l] (l = 0 finest .. depth-1 coarsest) = Int32Array pairs [start, end)
	const coarseToFine = [];
	let ranges = [ 0, n ];
	for ( let d = 0; d < depth; d ++ ) {

		const remaining = depth - d; // splits left including this one
		const next = [];
		for ( let r = 0; r < ranges.length; r += 2 ) {

			const a = ranges[ r ], b = ranges[ r + 1 ], size = b - a;
			const parts = Math.max( 1, Math.min( branching, Math.floor( size / ( minLeaf * Math.pow( branching, remaining - 1 ) ) ) ) );
			const w = lognormalWeights( parts, p.sizeSpread, rand );
			let acc = 0, prev = a;
			for ( let k = 0; k < parts; k ++ ) {

				acc += w[ k ];
				let cut = k === parts - 1 ? b : a + Math.round( ( size * acc ) / parts );
				cut = Math.max( cut, prev + 1 );
				cut = Math.min( cut, b - ( parts - 1 - k ) );
				next.push( prev, cut );
				prev = cut;

			}

		}

		coarseToFine.push( Int32Array.from( next ) );
		ranges = next;

	}

	const rangesByLevel = coarseToFine.slice().reverse();
	// block id per node per level
	const blockOf = rangesByLevel.map( ( rg ) => {

		const ids = new Int32Array( n );
		for ( let b = 0; b < rg.length / 2; b ++ ) ids.fill( b, rg[ 2 * b ], rg[ 2 * b + 1 ] );
		return ids;

	} );

	const w = lognormalWeights( n, p.degreeSpread, rand );
	const prefix = new Float64Array( n + 1 );
	for ( let i = 0; i < n; i ++ ) prefix[ i + 1 ] = prefix[ i ] + w[ i ];
	const total = prefix[ n ];

	const sampleIn = ( a, b ) => searchPrefix( prefix, prefix[ a ] + rand() * ( prefix[ b ] - prefix[ a ] ), a, b );
	const partner = ( i ) => {

		let l = 0;
		while ( l < depth && rand() < mix[ l ] ) l ++;
		if ( l === 0 ) {

			const rg = rangesByLevel[ 0 ], b = blockOf[ 0 ][ i ];
			return sampleIn( rg[ 2 * b ], rg[ 2 * b + 1 ] );

		}

		// inside the level-l block (the whole graph when l === depth), outside the level-(l-1) block
		let a = 0, bEnd = n;
		if ( l < depth ) {

			const rg = rangesByLevel[ l ], b = blockOf[ l ][ i ];
			a = rg[ 2 * b ]; bEnd = rg[ 2 * b + 1 ];

		}

		const crg = rangesByLevel[ l - 1 ], cb = blockOf[ l - 1 ][ i ];
		const ca = crg[ 2 * cb ], ce = crg[ 2 * cb + 1 ];
		const left = prefix[ ca ] - prefix[ a ], right = prefix[ bEnd ] - prefix[ ce ];
		if ( left + right <= 0 ) return sampleIn( a, bEnd );
		const u = rand() * ( left + right );
		return u < left ? searchPrefix( prefix, prefix[ a ] + u, a, ca ) : searchPrefix( prefix, prefix[ ce ] + ( u - left ), ce, bEnd );

	};

	const target = Math.round( ( n * p.avgDegree ) / 2 );
	const buf = new PairBuffer( Math.ceil( target * 1.1 ) + 16 );
	const half = p.avgDegree / 2;
	for ( let i = 0; i < n; i ++ ) {

		const s = Math.floor( half * w[ i ] + rand() );
		for ( let k = 0; k < s; k ++ ) buf.push( i, partner( i ) );

	}

	let edges = finishEdges( n, buf.view() );
	// top up duplicates/self loops lost in small dense blocks (initiators drawn ∝ weight)
	for ( let round = 0; round < 4 && edges.length / 2 < target * 0.995; round ++ ) {

		const missing = target - edges.length / 2;
		const extra = new PairBuffer( missing + 16 );
		for ( let k = 0; k < missing * 1.05; k ++ ) {

			const i = searchPrefix( prefix, rand() * total, 0, n );
			extra.push( i, partner( i ) );

		}

		const merged = new Uint32Array( edges.length + 2 * extra.length );
		merged.set( edges );
		merged.set( extra.view(), edges.length );
		edges = finishEdges( n, merged );

	}

	const planted = { levels: blockOf };
	return {
		name: `Hierarchical SBM · ${fmt( n )} nodes`,
		description: `Hierarchical stochastic block model: ${depth} level${depth > 1 ? 's' : ''}, branching ${branching}, mixing ${mix.map( ( x ) => + x.toFixed( 3 ) ).join( '/' )}, average degree ${p.avgDegree}.`,
		nodeCount: n,
		edges,
		planted,
		stats: { blocksPerLevel: rangesByLevel.map( ( r ) => r.length / 2 ), mixing: measuredMixing( edges, blockOf[ 0 ] ) },
	};

}

function measuredMixing( edges, block ) {

	const E = edges.length >> 1;
	let out = 0;
	for ( let e = 0; e < E; e ++ ) if ( block[ edges[ 2 * e ] ] !== block[ edges[ 2 * e + 1 ] ] ) out ++;
	return E ? round3( out / E ) : 0;

}

// ── LFR-like benchmark ───────────────────────────────────────────────────────────────────────

function powerLawMean( a, b, t ) {

	if ( b <= a ) return a;
	if ( Math.abs( t - 1 ) < 1e-9 ) return ( b - a ) / Math.log( b / a );
	if ( Math.abs( t - 2 ) < 1e-9 ) return Math.log( b / a ) / ( 1 / a - 1 / b );
	return ( ( ( Math.pow( b, 2 - t ) - Math.pow( a, 2 - t ) ) / ( 2 - t ) ) / ( ( Math.pow( b, 1 - t ) - Math.pow( a, 1 - t ) ) / ( 1 - t ) ) );

}

function samplePowerLaw( a, b, t, rand ) {

	if ( b <= a ) return a;
	const u = rand();
	if ( Math.abs( t - 1 ) < 1e-9 ) return a * Math.pow( b / a, u );
	const e = 1 - t;
	return Math.pow( Math.pow( a, e ) + u * ( Math.pow( b, e ) - Math.pow( a, e ) ), 1 / e );

}

function shuffleInPlace( arr, rand, from = 0, to = arr.length ) {

	for ( let i = to - 1; i > from; i -- ) {

		const j = from + Math.floor( rand() * ( i - from + 1 ) );
		const t = arr[ i ]; arr[ i ] = arr[ j ]; arr[ j ] = t;

	}

}

export function lfr( p, seed = 1 ) {

	const n = p.n;
	const rand = rng( seed );
	const t1 = p.degreeExponent, t2 = p.communityExponent, mu = p.mu;
	const kmax = Math.max( 2, Math.min( p.maxDegree, n - 1 ) );
	const kavg = Math.min( p.avgDegree, kmax );
	// minimum degree so that the truncated power law has the requested mean
	let lo = 1, hi = kmax;
	if ( powerLawMean( 1, kmax, t1 ) >= kavg ) hi = 1;
	else for ( let it = 0; it < 60; it ++ ) {

		const mid = ( lo + hi ) / 2;
		if ( powerLawMean( mid, kmax, t1 ) < kavg ) lo = mid;
		else hi = mid;

	}

	const kmin = hi;
	const deg = new Int32Array( n );
	const din = new Int32Array( n );
	let maxIn = 0;
	for ( let i = 0; i < n; i ++ ) {

		deg[ i ] = Math.max( 1, Math.min( kmax, Math.floor( samplePowerLaw( kmin, kmax, t1, rand ) + rand() ) ) );
		din[ i ] = Math.min( deg[ i ], Math.floor( ( 1 - mu ) * deg[ i ] + rand() ) );
		if ( din[ i ] > maxIn ) maxIn = din[ i ];

	}

	// community sizes: power law in [minc, maxc]; the largest internal degree must fit
	let maxc = Math.min( n, Math.max( p.maxCommunity, maxIn + 1 ) );
	let minc = Math.min( maxc, Math.max( 2, p.minCommunity ) );
	const sizes = [];
	let sum = 0;
	while ( sum < n ) {

		const s = Math.round( samplePowerLaw( minc, maxc, t2, rand ) );
		sizes.push( s );
		sum += s;

	}

	let over = sum - n;
	sizes[ sizes.length - 1 ] -= over;
	if ( sizes[ sizes.length - 1 ] < minc && sizes.length > 1 ) {

		let rest = sizes.pop();
		for ( let k = 0; rest > 0; k = ( k + 1 ) % sizes.length, rest -- ) sizes[ k ] ++;

	}

	const C = sizes.length;

	// assignment: nodes by internal degree (desc) to random communities with room and size > din
	const order = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) order[ i ] = i;
	shuffleInPlace( order, rand );
	order.sort( ( a, b ) => din[ b ] - din[ a ] );
	const free = Int32Array.from( sizes );
	const comm = new Int32Array( n );
	for ( let t = 0; t < n; t ++ ) {

		const i = order[ t ];
		let c = - 1;
		for ( let tries = 0; tries < 24 && c < 0; tries ++ ) {

			const x = Math.floor( rand() * C );
			if ( free[ x ] > 0 && sizes[ x ] > din[ i ] ) c = x;

		}

		if ( c < 0 ) {

			const off = Math.floor( rand() * C );
			for ( let k = 0; k < C && c < 0; k ++ ) {

				const x = ( off + k ) % C;
				if ( free[ x ] > 0 && sizes[ x ] > din[ i ] ) c = x;

			}

		}

		if ( c < 0 ) {

			let best = 0;
			for ( let x = 1; x < C; x ++ ) if ( free[ x ] > free[ best ] ) best = x;
			c = best;
			din[ i ] = Math.min( din[ i ], sizes[ c ] - 1 );

		}

		comm[ i ] = c;
		free[ c ] --;

	}

	const seen = new Set();
	const buf = new PairBuffer( Math.ceil( ( n * kavg ) / 2 ) + 16 );
	const tryAdd = ( a, b ) => {

		if ( a === b ) return false;
		const key = a < b ? a * n + b : b * n + a;
		if ( seen.has( key ) ) return false;
		seen.add( key );
		buf.push( a, b );
		return true;

	};

	// stub matching with a few retry rounds for rejected pairs (self loops / multi-edges)
	const match = ( stubs, sameCommunityOnly ) => {

		let list = stubs;
		for ( let round = 0; round < 6 && list.length > 1; round ++ ) {

			shuffleInPlace( list, rand );
			const failed = [];
			for ( let k = 0; k + 1 < list.length; k += 2 ) {

				let a = list[ k ], b = list[ k + 1 ];
				if ( ! sameCommunityOnly && comm[ a ] === comm[ b ] ) {

					// external stub pairs must cross communities: try a swap with a later stub
					let swapped = false;
					for ( let s = 0; s < 4 && ! swapped; s ++ ) {

						const j = k + 2 + Math.floor( rand() * Math.max( 0, list.length - k - 2 ) );
						if ( j < list.length && comm[ list[ j ] ] !== comm[ a ] ) {

							const t = list[ j ]; list[ j ] = b; list[ k + 1 ] = t; b = t;
							swapped = true;

						}

					}

					if ( ! swapped ) {

						failed.push( a, b );
						continue;

					}

				}

				if ( ! tryAdd( a, b ) ) failed.push( a, b );

			}

			list = failed;

		}

	};

	// internal edges, community by community
	const members = Array.from( { length: C }, () => [] );
	for ( let i = 0; i < n; i ++ ) members[ comm[ i ] ].push( i );
	for ( let c = 0; c < C; c ++ ) {

		const stubs = [];
		for ( const i of members[ c ] ) for ( let k = 0; k < din[ i ]; k ++ ) stubs.push( i );
		match( stubs, true );

	}

	const ext = [];
	for ( let i = 0; i < n; i ++ ) for ( let k = din[ i ]; k < deg[ i ]; k ++ ) ext.push( i );
	match( ext, false );

	const edges = finishEdges( n, buf.view() );
	const sizesSorted = sizes.slice().sort( ( a, b ) => b - a );
	return {
		name: `LFR benchmark · ${fmt( n )} nodes`,
		description: `LFR-like benchmark: ${C} communities (${minc}–${maxc} nodes, exponent ${t2}), degree exponent ${t1}, average degree ${kavg}, max ${kmax}, mixing μ = ${mu}.`,
		nodeCount: n,
		edges,
		planted: { levels: [ comm ] },
		stats: { communities: C, largestCommunity: sizesSorted[ 0 ], smallestCommunity: sizesSorted[ C - 1 ], mixing: measuredMixing( edges, comm ) },
	};

}

// ── Barabási–Albert ─────────────────────────────────────────────────────────────────────────

export function barabasiAlbert( p, seed = 1 ) {

	const n = p.n, m = Math.max( 1, Math.min( p.m, n - 1 ) );
	const rand = rng( seed );
	const m0 = m + 1;
	const buf = new PairBuffer( ( m0 * ( m0 - 1 ) ) / 2 + ( n - m0 ) * m + 16 );
	const rep = new Uint32Array( 2 * ( ( m0 * ( m0 - 1 ) ) / 2 + Math.max( 0, n - m0 ) * m ) + 2 );
	let r = 0;
	for ( let a = 0; a < Math.min( m0, n ); a ++ ) for ( let b = a + 1; b < Math.min( m0, n ); b ++ ) {

		buf.push( a, b );
		rep[ r ++ ] = a; rep[ r ++ ] = b;

	}

	const picked = new Int32Array( m );
	for ( let v = m0; v < n; v ++ ) {

		let cnt = 0;
		while ( cnt < m ) {

			const t = rep[ Math.floor( rand() * r ) ];
			let dup = false;
			for ( let q = 0; q < cnt; q ++ ) if ( picked[ q ] === t ) dup = true;
			if ( ! dup ) picked[ cnt ++ ] = t;

		}

		for ( let q = 0; q < m; q ++ ) {

			buf.push( v, picked[ q ] );
			rep[ r ++ ] = v; rep[ r ++ ] = picked[ q ];

		}

	}

	return {
		name: `Barabási–Albert · ${fmt( n )} nodes`,
		description: `Preferential attachment, ${m} edge${m > 1 ? 's' : ''} per new node.`,
		nodeCount: n,
		edges: finishEdges( n, buf.view() ),
	};

}

// ── Watts–Strogatz ──────────────────────────────────────────────────────────────────────────

export function wattsStrogatz( p, seed = 1 ) {

	const n = p.n;
	const half = Math.max( 1, Math.min( Math.floor( p.k / 2 ), Math.floor( ( n - 1 ) / 2 ) ) );
	const rand = rng( seed );
	const E = n * half;
	const src = new Uint32Array( E ), dst = new Uint32Array( E );
	const seen = new Set();
	let e = 0;
	for ( let j = 1; j <= half; j ++ ) for ( let i = 0; i < n; i ++ ) {

		const t = ( i + j ) % n;
		src[ e ] = i; dst[ e ] = t;
		seen.add( i < t ? i * n + t : t * n + i );
		e ++;

	}

	// rewire the far endpoint of each lattice edge with probability beta (networkx order)
	for ( let k = 0; k < E; k ++ ) {

		if ( rand() >= p.beta ) continue;
		const u = src[ k ];
		const old = dst[ k ];
		for ( let tries = 0; tries < 16; tries ++ ) {

			const w = Math.floor( rand() * n );
			if ( w === u ) continue;
			const key = u < w ? u * n + w : w * n + u;
			if ( seen.has( key ) ) continue;
			seen.delete( u < old ? u * n + old : old * n + u );
			seen.add( key );
			dst[ k ] = w;
			break;

		}

	}

	const pairs = new Uint32Array( 2 * E );
	for ( let k = 0; k < E; k ++ ) {

		pairs[ 2 * k ] = src[ k ]; pairs[ 2 * k + 1 ] = dst[ k ];

	}

	return {
		name: `Watts–Strogatz · ${fmt( n )} nodes`,
		description: `Small world: ring lattice with ${2 * half} neighbors per node, rewiring probability ${p.beta}.`,
		nodeCount: n,
		edges: finishEdges( n, pairs ),
	};

}

// ── random geometric graph in 3D ─────────────────────────────────────────────────────────────

/** Mean volume of a radius-r ball intersected with the unit cube (uniform centers, r ≤ 1). */
function cubeBallVolume( r ) {

	return ( 4 * Math.PI * r ** 3 ) / 3 - ( 3 * Math.PI * r ** 4 ) / 2 + ( 8 * r ** 5 ) / 5 - r ** 6 / 6;

}

export function randomGeometric3D( p, seed = 1 ) {

	const n = p.n, periodic = !! p.periodic;
	const rand = rng( seed );
	const want = Math.min( p.avgDegree, n - 1 ) / Math.max( 1, n - 1 );
	let r;
	if ( periodic ) r = Math.cbrt( ( 3 * want ) / ( 4 * Math.PI ) );
	else {

		let lo = 0, hi = 1;
		for ( let it = 0; it < 60; it ++ ) {

			const mid = ( lo + hi ) / 2;
			if ( cubeBallVolume( mid ) < want ) lo = mid;
			else hi = mid;

		}

		r = hi;

	}

	r = Math.min( r, 0.5 );
	const pos = new Float64Array( 3 * n );
	for ( let k = 0; k < 3 * n; k ++ ) pos[ k ] = rand();
	const G = Math.max( 1, Math.floor( 1 / r ) );
	const cellOf = new Int32Array( n );
	const cellStart = new Int32Array( G * G * G + 1 );
	const cellIdx = ( x ) => Math.min( G - 1, Math.floor( x * G ) );
	for ( let i = 0; i < n; i ++ ) {

		const c = cellIdx( pos[ 3 * i ] ) + G * ( cellIdx( pos[ 3 * i + 1 ] ) + G * cellIdx( pos[ 3 * i + 2 ] ) );
		cellOf[ i ] = c;
		cellStart[ c + 1 ] ++;

	}

	for ( let c = 0; c < G * G * G; c ++ ) cellStart[ c + 1 ] += cellStart[ c ];
	const items = new Int32Array( n );
	{

		const fill = cellStart.slice( 0, G * G * G );
		for ( let i = 0; i < n; i ++ ) items[ fill[ cellOf[ i ] ] ++ ] = i;

	}

	const r2 = r * r;
	const buf = new PairBuffer( Math.ceil( ( n * p.avgDegree ) / 2 ) + 16 );
	const visited = new Int32Array( G * G * G ).fill( - 1 );
	for ( let i = 0; i < n; i ++ ) {

		const x = pos[ 3 * i ], y = pos[ 3 * i + 1 ], z = pos[ 3 * i + 2 ];
		const cx = cellIdx( x ), cy = cellIdx( y ), cz = cellIdx( z );
		for ( let dz = - 1; dz <= 1; dz ++ ) for ( let dy = - 1; dy <= 1; dy ++ ) for ( let dx = - 1; dx <= 1; dx ++ ) {

			let gx = cx + dx, gy = cy + dy, gz = cz + dz;
			if ( periodic ) {

				gx = ( gx + G ) % G; gy = ( gy + G ) % G; gz = ( gz + G ) % G;

			} else if ( gx < 0 || gy < 0 || gz < 0 || gx >= G || gy >= G || gz >= G ) continue;

			const c = gx + G * ( gy + G * gz );
			if ( visited[ c ] === i ) continue; // small G with wrapping can revisit a cell
			visited[ c ] = i;
			for ( let q = cellStart[ c ]; q < cellStart[ c + 1 ]; q ++ ) {

				const j = items[ q ];
				if ( j <= i ) continue;
				let ddx = Math.abs( pos[ 3 * j ] - x ), ddy = Math.abs( pos[ 3 * j + 1 ] - y ), ddz = Math.abs( pos[ 3 * j + 2 ] - z );
				if ( periodic ) {

					ddx = Math.min( ddx, 1 - ddx ); ddy = Math.min( ddy, 1 - ddy ); ddz = Math.min( ddz, 1 - ddz );

				}

				if ( ddx * ddx + ddy * ddy + ddz * ddz <= r2 ) buf.push( i, j );

			}

		}

	}

	return {
		name: `Random geometric 3D · ${fmt( n )} nodes`,
		description: `Uniform points in the unit ${periodic ? '3-torus' : 'cube'} joined within radius ${r.toFixed( 4 )} (target average degree ${p.avgDegree}).`,
		nodeCount: n,
		edges: finishEdges( n, buf.view() ),
		stats: { radius: round3( r ) },
	};

}

// ── 3D lattice / torus ───────────────────────────────────────────────────────────────────────

export function lattice3D( p ) {

	const nx = p.nx, ny = p.ny, nz = p.nz, periodic = !! p.periodic;
	const n = nx * ny * nz;
	const id = ( x, y, z ) => x + nx * ( y + ny * z );
	const buf = new PairBuffer( 3 * n );
	const labels = n <= 20000 ? new Array( n ) : null;
	for ( let z = 0; z < nz; z ++ ) for ( let y = 0; y < ny; y ++ ) for ( let x = 0; x < nx; x ++ ) {

		const i = id( x, y, z );
		if ( labels ) labels[ i ] = `${x},${y},${z}`;
		if ( x + 1 < nx ) buf.push( i, id( x + 1, y, z ) );
		else if ( periodic && nx > 2 ) buf.push( i, id( 0, y, z ) );
		if ( y + 1 < ny ) buf.push( i, id( x, y + 1, z ) );
		else if ( periodic && ny > 2 ) buf.push( i, id( x, 0, z ) );
		if ( z + 1 < nz ) buf.push( i, id( x, y, z + 1 ) );
		else if ( periodic && nz > 2 ) buf.push( i, id( x, y, 0 ) );

	}

	return {
		name: `${periodic ? '3D torus' : '3D lattice'} ${nx}×${ny}×${nz}`,
		description: `Regular ${periodic ? 'periodic ' : ''}grid of ${fmt( n )} nodes with 6-neighborhoods.`,
		nodeCount: n,
		labels,
		edges: finishEdges( n, buf.view() ),
	};

}

// ── ring of cliques (connected caveman) ──────────────────────────────────────────────────────

export function caveman( p, seed = 1 ) {

	const L = p.cliques, K = p.cliqueSize, n = L * K;
	const rand = rng( seed );
	const pairs = [];
	for ( let c = 0; c < L; c ++ ) {

		const s = c * K;
		for ( let a = 0; a < K; a ++ ) for ( let b = a + 1; b < K; b ++ ) {

			if ( a === 0 && b === 1 ) continue; // the edge replaced by the ring link
			pairs.push( s + a, s + b );

		}

		pairs.push( s, ( s - 1 + n ) % n ); // link to the previous clique

	}

	if ( p.rewire > 0 ) {

		const seen = new Set();
		const key = ( a, b ) => ( a < b ? a * n + b : b * n + a );
		for ( let k = 0; k < pairs.length; k += 2 ) seen.add( key( pairs[ k ], pairs[ k + 1 ] ) );
		for ( let k = 0; k < pairs.length; k += 2 ) {

			if ( rand() >= p.rewire ) continue;
			const u = pairs[ k ], v = pairs[ k + 1 ];
			const w = Math.floor( rand() * n );
			if ( w === u || seen.has( key( u, w ) ) ) continue;
			seen.delete( key( u, v ) );
			seen.add( key( u, w ) );
			pairs[ k + 1 ] = w;

		}

	}

	const comm = new Int32Array( n );
	for ( let i = 0; i < n; i ++ ) comm[ i ] = Math.floor( i / K );
	return {
		name: `Ring of ${L} cliques`,
		description: `Connected caveman graph: ${L} cliques of ${K} nodes in a ring${p.rewire > 0 ? `, rewiring ${p.rewire}` : ''}.`,
		nodeCount: n,
		edges: finishEdges( n, Uint32Array.from( pairs ) ),
		planted: { levels: [ comm ] },
	};

}

// ── hubs and satellites ──────────────────────────────────────────────────────────────────────

export function galaxy( p, seed = 1 ) {

	const rand = rng( seed );
	const G = p.groups, perGroup = p.systems, S = G * perGroup;
	const n = Math.max( p.n, 2 * S ); // a hub and one satellite per system at least
	// system sizes: lognormal shares of the nodes, each at least a hub and a satellite
	const share = lognormalWeights( S, p.sizeSpread, rand );
	const size = new Int32Array( S );
	let used = 0;
	for ( let k = 0; k < S; k ++ ) {

		size[ k ] = 2 + Math.floor( ( ( n - 2 * S ) * share[ k ] ) / S );
		used += size[ k ];

	}

	for ( let k = 0; used < n; k = ( k + 1 ) % S, used ++ ) size[ k ] ++;
	const start = new Int32Array( S + 1 );
	for ( let k = 0; k < S; k ++ ) start[ k + 1 ] = start[ k ] + size[ k ];

	const buf = new PairBuffer( 2 * n );
	const system = new Int32Array( n ), group = new Int32Array( n );
	let moons = 0;
	for ( let k = 0; k < S; k ++ ) {

		const hub = start[ k ], m = size[ k ] - 1;
		system.fill( k, hub, hub + size[ k ] );
		group.fill( Math.floor( k / perGroup ), hub, hub + size[ k ] );
		// a satellite orbits the hub; a moon orbits an earlier satellite of the system
		for ( let q = 1; q <= m; q ++ ) {

			if ( q > 1 && rand() < p.moons ) {

				buf.push( hub + q, hub + 1 + Math.floor( rand() * ( q - 1 ) ) );
				moons ++;

			} else buf.push( hub + q, hub );

		}

		for ( let e = Math.round( p.mesh * m ); e > 0 && m > 1; e -- ) {

			const a = hub + 1 + Math.floor( rand() * m ), b = hub + 1 + Math.floor( rand() * m );
			if ( a !== b ) buf.push( a, b );

		}

	}

	// hubs of a group: a ring, and chords up to `bridges` links per hub
	const core = new Int32Array( G );
	for ( let g = 0; g < G; g ++ ) {

		const first = g * perGroup;
		let largest = first;
		for ( let j = 0; j < perGroup; j ++ ) {

			const k = first + j;
			if ( size[ k ] > size[ largest ] ) largest = k;
			if ( perGroup > 2 || j === 0 ) buf.push( start[ k ], start[ first + ( ( j + 1 ) % perGroup ) ] );
			for ( let b = 1; b < p.bridges && perGroup > 3; b ++ ) {

				const other = first + Math.floor( rand() * perGroup );
				if ( other !== k ) buf.push( start[ k ], start[ other ] );

			}

		}

		core[ g ] = start[ largest ];

	}

	// the groups meet at their largest hubs: a ring, and a few chords across it
	for ( let g = 0; g < G; g ++ ) if ( G > 2 || g === 0 ) buf.push( core[ g ], core[ ( g + 1 ) % G ] );
	for ( let e = Math.floor( G / 3 ); e > 0; e -- ) {

		const a = Math.floor( rand() * G ), b = Math.floor( rand() * G );
		if ( a !== b ) buf.push( core[ a ], core[ b ] );

	}

	return {
		name: `Hubs and satellites · ${fmt( n )} nodes`,
		description: `${G} groups of ${perGroup} systems (a hub with satellites, ${Math.round( ( 100 * moons ) / Math.max( 1, n - S ) )}% of them moons of another satellite); hubs of a group are linked, the groups meet at their largest hubs.`,
		nodeCount: n,
		edges: finishEdges( n, buf.view() ),
		planted: { levels: [ system, group ] },
		stats: { systems: S, moons },
	};

}

// ── geometric scale-free (geometric inhomogeneous random graph) ──────────────────────────────

/**
 * Threshold GIRG: node i has a position x_i in [0,1)^d and a weight w_i >= 1 with density
 * ∝ w^-exponent; i ~ j when |x_i − x_j|^d <= c · w_i · w_j. The expected degree of a node is
 * proportional to its weight, so the degrees follow the same power law, and neighbours are close
 * unless one of them is a hub. Sampling: the nodes are put in layers of weight [ 2^a, 2^(a+1) );
 * for a pair of layers the reach is bounded by ( c · 2^(a+1) · 2^(b+1) )^(1/d), and a grid of
 * that cell size over the heavier layer finds the candidates. Linear in nodes + edges, up to
 * the logarithmic number of layer pairs. c is set from the expected degree and corrected once
 * from the measured one (the border of the square or cube costs edges).
 */
export function geometricScaleFree( p, seed = 1 ) {

	const n = p.n, d = p.dimension === 3 ? 3 : 2, periodic = !! p.periodic;
	const rand = rng( seed );
	const pos = new Float64Array( d * n );
	for ( let k = 0; k < d * n; k ++ ) pos[ k ] = rand();
	// Pareto weights, capped so that no node expects more than n − 1 neighbours by a wide margin
	const w = new Float64Array( n );
	const cap = Math.pow( n, 1 / ( p.exponent - 1 ) );
	let mean = 0;
	for ( let i = 0; i < n; i ++ ) {

		w[ i ] = Math.min( cap, Math.pow( 1 - rand(), - 1 / ( p.exponent - 1 ) ) );
		mean += w[ i ] / n;

	}

	// layers by weight
	const layerOf = new Int32Array( n );
	let layers = 0;
	for ( let i = 0; i < n; i ++ ) {

		layerOf[ i ] = Math.floor( Math.log2( w[ i ] ) );
		if ( layerOf[ i ] + 1 > layers ) layers = layerOf[ i ] + 1;

	}

	const members = Array.from( { length: layers }, () => [] );
	for ( let i = 0; i < n; i ++ ) members[ layerOf[ i ] ].push( i );
	const ballVolume = d === 3 ? ( 4 * Math.PI ) / 3 : Math.PI;
	const want = Math.min( p.avgDegree, n - 1 );

	const sample = ( c ) => {

		const buf = new PairBuffer( Math.ceil( ( n * want ) / 2 ) + 16 );
		for ( let a = 0; a < layers; a ++ ) {

			if ( ! members[ a ].length ) continue;
			for ( let b = a; b < layers; b ++ ) {

				const heavy = members[ b ];
				if ( ! heavy.length ) continue;
				const reach = Math.min( 0.5, Math.pow( c * Math.pow( 2, a + b + 2 ), 1 / d ) );
				const cells = Math.max( 1, Math.min( 256, Math.floor( 1 / reach ) ) );
				const cellCount = Math.pow( cells, d );
				const cellOf = ( i ) => {

					let cell = 0;
					for ( let k = d - 1; k >= 0; k -- ) cell = cell * cells + Math.min( cells - 1, Math.floor( pos[ d * i + k ] * cells ) );
					return cell;

				};

				// the heavier layer in a grid
				const head = new Int32Array( cellCount + 1 );
				for ( const j of heavy ) head[ cellOf( j ) + 1 ] ++;
				for ( let q = 0; q < cellCount; q ++ ) head[ q + 1 ] += head[ q ];
				const items = new Int32Array( heavy.length ), fill = head.slice( 0, cellCount );
				for ( const j of heavy ) items[ fill[ cellOf( j ) ] ++ ] = j;
				const span = cells >= 3 ? 1 : 0; // fewer than 3 cells per axis: every cell is a neighbour
				for ( const i of members[ a ] ) {

					const ci = [ 0, 0, 0 ];
					for ( let k = 0; k < d; k ++ ) ci[ k ] = Math.min( cells - 1, Math.floor( pos[ d * i + k ] * cells ) );
					const visit = ( cell ) => {

						for ( let q = head[ cell ]; q < head[ cell + 1 ]; q ++ ) {

							const j = items[ q ];
							if ( a === b && j <= i ) continue;
							let dist2 = 0;
							for ( let k = 0; k < d; k ++ ) {

								let dx = Math.abs( pos[ d * i + k ] - pos[ d * j + k ] );
								if ( periodic && dx > 0.5 ) dx = 1 - dx;
								dist2 += dx * dx;

							}

							const r = Math.sqrt( dist2 );
							if ( ( d === 3 ? r * r * r : dist2 ) <= c * w[ i ] * w[ j ] ) buf.push( i, j );

						}

					};

					if ( span === 0 ) {

						for ( let cell = 0; cell < cellCount; cell ++ ) visit( cell );
						continue;

					}

					for ( let dz = d === 3 ? - 1 : 0; dz <= ( d === 3 ? 1 : 0 ); dz ++ ) for ( let dy = - 1; dy <= 1; dy ++ ) for ( let dx = - 1; dx <= 1; dx ++ ) {

						let gx = ci[ 0 ] + dx, gy = ci[ 1 ] + dy, gz = ci[ 2 ] + dz;
						if ( periodic ) {

							gx = ( gx + cells ) % cells; gy = ( gy + cells ) % cells; gz = ( gz + cells ) % cells;

						} else if ( gx < 0 || gy < 0 || gz < 0 || gx >= cells || gy >= cells || gz >= cells ) continue;

						visit( gx + cells * ( gy + ( d === 3 ? cells * gz : 0 ) ) );

					}

				}

			}

		}

		return finishEdges( n, buf.view() );

	};

	let c = want / ( n * ballVolume * mean * mean );
	let edges = sample( c );
	for ( let pass = 0; pass < 2; pass ++ ) {

		const got = edges.length / n; // 2E / n
		if ( got <= 0 || Math.abs( got / want - 1 ) < 0.03 ) break;
		c *= want / got;
		edges = sample( c );

	}

	return {
		name: `Geometric scale-free · ${fmt( n )} nodes`,
		description: `Random points in the unit ${d === 3 ? ( periodic ? '3-torus' : 'cube' ) : ( periodic ? 'torus' : 'square' )} with power-law weights (exponent ${p.exponent}), linked when close relative to their weights: hubs with long links over a geometric background (target average degree ${p.avgDegree}).`,
		nodeCount: n,
		edges,
		stats: { maxWeight: round3( w.reduce( ( m, x ) => Math.max( m, x ), 0 ) ), threshold: c },
	};

}

// ── surface mesh: sheet, tube, torus ─────────────────────────────────────────────────────────

export function surfaceMesh( p ) {

	const nu = p.nu, nv = p.nv, wrapU = !! p.wrapU, wrapV = !! p.wrapV, triangles = !! p.triangles;
	const n = nu * nv;
	const id = ( u, v ) => u + nu * v;
	const buf = new PairBuffer( 3 * n );
	const labels = n <= 20000 ? new Array( n ) : null;
	for ( let v = 0; v < nv; v ++ ) for ( let u = 0; u < nu; u ++ ) {

		const i = id( u, v );
		if ( labels ) labels[ i ] = `${u},${v}`;
		const u1 = u + 1 < nu ? u + 1 : wrapU ? 0 : - 1;
		const v1 = v + 1 < nv ? v + 1 : wrapV ? 0 : - 1;
		if ( u1 >= 0 ) buf.push( i, id( u1, v ) );
		if ( v1 >= 0 ) buf.push( i, id( u, v1 ) );
		if ( triangles && u1 >= 0 && v1 >= 0 ) buf.push( i, id( u1, v1 ) );

	}

	const shape = wrapU && wrapV ? 'torus' : wrapU || wrapV ? 'tube' : 'sheet';
	return {
		name: `${triangles ? 'Triangle' : 'Quad'} mesh · ${shape} ${nu}×${nv}`,
		description: `A ${triangles ? 'triangle' : 'quad'} mesh of ${fmt( n )} nodes on a ${shape}${shape === 'tube' ? ` (wrapped along ${wrapU ? 'U' : 'V'})` : ''}. Layout sanity check: should unfold into that surface.`,
		nodeCount: n,
		labels,
		edges: finishEdges( n, buf.view() ),
	};

}

// ── labels ───────────────────────────────────────────────────────────────────────────────────

const ONSETS = [ 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'z', 'br', 'ch', 'dr', 'fl', 'gr', 'kr', 'sh', 'st', 'th', 'tr' ];
const VOWELS = [ 'a', 'e', 'i', 'o', 'u', 'a', 'e', 'i', 'o', 'ai', 'ea', 'ou' ];
const CODAS = [ '', '', '', '', 'n', 'r', 's', 'l', 'm', 'x', 'th', 'nd' ];

function labelSeed( id, seed ) {

	let h = seed >>> 0;
	for ( let i = 0; i < id.length; i ++ ) h = Math.imul( h ^ id.charCodeAt( i ), 16777619 ) >>> 0;
	return h || 1;

}

/** n unique pronounceable pseudo-words (deterministic). */
export function pseudoWords( n, seed = 1 ) {

	const rand = rng( ( seed * 2654435761 ) >>> 0 );
	const pick = ( arr ) => arr[ Math.floor( rand() * arr.length ) ];
	const used = new Set();
	const out = new Array( n );
	for ( let i = 0; i < n; i ++ ) {

		let word;
		for ( let tries = 0; ; tries ++ ) {

			const syl = 2 + ( rand() < 0.35 ? 1 : 0 ) + ( tries > 8 ? 1 : 0 );
			word = '';
			for ( let s = 0; s < syl; s ++ ) word += pick( ONSETS ) + pick( VOWELS );
			word += pick( CODAS );
			word = word[ 0 ].toUpperCase() + word.slice( 1 );
			if ( ! used.has( word ) ) break;
			if ( tries > 40 ) {

				word += ` ${i}`;
				break;

			}

		}

		used.add( word );
		out[ i ] = word;

	}

	return out;

}

function now() {

	return typeof performance !== 'undefined' ? performance.now() : Date.now();

}
