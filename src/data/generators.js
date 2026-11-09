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
//   randomGeometric3D(p, seed), lattice3D(p), caveman(p, seed)     raw generators (resolved params)
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
