// Builds the ladder of generated datasets (npm run data:generate).
//
// Every preset is a deterministic generator spec { model, params, seed } (src/data/generators.js)
// clustered with the default Louvain settings (src/data/louvain.js: seed 1, resolution 1, leaf
// clusters refined to ≤ 200 nodes), exactly what the data worker does on demand. Presets whose
// wgv-graph JSON is at most MAX_FILE_BYTES are written to public/data/generated/<id>.json; larger
// ones are listed in public/data/generated/presets.json ("wgv-presets") with their spec and
// counts, so the app generates them in the worker instead of downloading megabytes. Then
// scripts/manifest.mjs rebuilds public/data/manifest.json (entries carry "file" or "generate").
//
// Usage: node scripts/generate.mjs [--only id,id] [--max-bytes N] [--dry]
import { writeFileSync, mkdirSync, readdirSync, unlinkSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { generate } from '../src/data/generators.js';
import { ensureHierarchy } from '../src/data/louvain.js';
import { serializeGraph } from '../src/data/format.js';
import { buildManifest } from './manifest.mjs';

const OUT = 'public/data/generated';
const args = process.argv.slice( 2 );
const opt = ( name, def ) => {

	const i = args.indexOf( '--' + name );
	return i >= 0 ? args[ i + 1 ] : def;

};

const MAX_FILE_BYTES = Number( opt( 'max-bytes', 1.5 * 1024 * 1024 ) );
const ONLY = opt( 'only', null )?.split( ',' );
const DRY = args.includes( '--dry' );

// tier: readable (labels and single edges legible), overview (cluster structure), stress (target
// scale, GPU cost), sanity (layout checks with known geometry)
export const PRESETS = [
	{
		id: 'caveman-24', tier: 'readable', name: 'Ring of 24 cliques',
		description: '24 cliques of 8 joined in a ring. Every edge and label is legible; clusters are unambiguous.',
		spec: { model: 'caveman', params: { cliques: 24, cliqueSize: 8, rewire: 0.02 }, seed: 1 },
	},
	{
		id: 'hsbm-500', tier: 'readable', name: 'Communities · 500',
		description: 'Hierarchical SBM: 5 groups of 5 communities, average degree 10. Readable labels, visible individual edges.',
		spec: { model: 'hsbm', params: { n: 500, avgDegree: 10, depth: 2, branching: 5, mixing: 0.2, sizeSpread: 0.3, degreeSpread: 0.4 }, seed: 1 },
	},
	{
		id: 'lfr-2k', tier: 'overview', name: 'LFR · 2k',
		description: 'LFR benchmark, 2,000 nodes, power-law degrees and community sizes, μ = 0.2. Labels on hover/zoom.',
		spec: { model: 'lfr', params: { n: 2000, avgDegree: 12, maxDegree: 80, mu: 0.2, degreeExponent: 2.5, communityExponent: 1.5, minCommunity: 15, maxCommunity: 120 }, seed: 1 },
	},
	{
		id: 'hsbm-5k', tier: 'overview', name: 'Communities · 5k',
		description: 'Hierarchical SBM, 5,000 nodes in 3 nested levels, average degree 16. Clear bundles at every level.',
		spec: { model: 'hsbm', params: { n: 5000, avgDegree: 16, depth: 3, branching: 5, mixing: 0.3, sizeSpread: 0.4, degreeSpread: 0.4 }, seed: 1 },
	},
	{
		id: 'ba-5k', tier: 'overview', name: 'Scale-free · 5k',
		description: 'Barabási–Albert, 5,000 nodes, 4 edges per node: hubs and weak communities. Hard case for bundling.',
		spec: { model: 'ba', params: { n: 5000, m: 4 }, seed: 1 },
	},
	{
		id: 'lfr-10k', tier: 'overview', name: 'LFR · 10k',
		description: 'LFR benchmark, 10,000 nodes, average degree 16, μ = 0.25. Structure overview at medium scale.',
		spec: { model: 'lfr', params: { n: 10000, avgDegree: 16, maxDegree: 150, mu: 0.25, degreeExponent: 2.5, communityExponent: 1.5, minCommunity: 20, maxCommunity: 250 }, seed: 1 },
	},
	{
		id: 'hsbm-20k', tier: 'stress', name: 'Communities · 20k',
		description: 'Hierarchical SBM, 20,000 nodes, average degree 24 (240k edges), 3 nested levels. Target scale.',
		spec: { model: 'hsbm', params: { n: 20000, avgDegree: 24, depth: 3, branching: 6, mixing: 0.3, sizeSpread: 0.4, degreeSpread: 0.4 }, seed: 1 },
	},
	{
		id: 'lfr-20k-dense', tier: 'stress', name: 'LFR · 20k dense',
		description: 'LFR benchmark, 20,000 nodes, average degree 40 (≈400k edges), μ = 0.3. Maximum stress test.',
		spec: { model: 'lfr', params: { n: 20000, avgDegree: 40, maxDegree: 300, mu: 0.3, degreeExponent: 2.5, communityExponent: 1.5, minCommunity: 30, maxCommunity: 600 }, seed: 1 },
	},
	{
		id: 'lattice-12', tier: 'sanity', name: '3D lattice 12³',
		description: '1,728-node cubic grid. Layout sanity check: should unfold into a box.',
		spec: { model: 'lattice3d', params: { nx: 12, ny: 12, nz: 12, periodic: false }, seed: 1 },
	},
	{
		id: 'rgg-3k', tier: 'sanity', name: 'Geometric 3D · 3k',
		description: 'Random geometric graph, 3,000 points in a cube, average degree 10. Layout sanity check: a solid blob.',
		spec: { model: 'rgg3d', params: { n: 3000, avgDegree: 10, periodic: false }, seed: 1 },
	},
	{
		id: 'ws-2k', tier: 'sanity', name: 'Small world · 2k',
		description: 'Watts–Strogatz ring, 2,000 nodes, 8 neighbors, 2% rewiring. Should lay out as a closed loop.',
		spec: { model: 'ws', params: { n: 2000, k: 8, beta: 0.02 }, seed: 1 },
	},
];

function quantiles( values ) {

	const v = Array.from( values ).sort( ( a, b ) => a - b );
	const q = ( f ) => v[ Math.min( v.length - 1, Math.floor( f * v.length ) ) ];
	return { count: v.length, min: v[ 0 ], median: q( 0.5 ), p90: q( 0.9 ), max: v[ v.length - 1 ] };

}

function clusterSizes( level ) {

	const c = new Map();
	for ( const x of level ) c.set( x, ( c.get( x ) || 0 ) + 1 );
	return [ ...c.values() ];

}

const kb = ( b ) => ( b >= 1024 * 1024 ? `${( b / 1024 / 1024 ).toFixed( 2 )} MB` : `${( b / 1024 ).toFixed( 0 )} KB` );

function build() {

	if ( ! DRY ) mkdirSync( OUT, { recursive: true } );
	const presets = [];
	const written = new Set();
	const rows = [];
	PRESETS.forEach( ( p, order ) => {

		if ( ONLY && ! ONLY.includes( p.id ) ) return;
		let t = performance.now();
		const raw = generate( { ...p.spec, name: p.name, description: p.description } );
		const tGen = performance.now() - t;
		t = performance.now();
		const g = ensureHierarchy( raw );
		const tCluster = performance.now() - t;
		g.meta = { ...g.meta, order, tier: p.tier, preset: p.id };
		const json = serializeGraph( g ) + '\n';
		const bytes = Buffer.byteLength( json );
		const levels = g.hierarchy ? g.hierarchy.levels : [];
		const leaf = levels.length ? quantiles( clusterSizes( levels[ 0 ] ) ) : null;
		const asFile = bytes <= MAX_FILE_BYTES;
		const entry = {
			id: `generated/${p.id}`,
			tier: p.tier,
			order,
			name: p.name,
			description: p.description,
			source: g.source,
			nodes: g.nodeCount,
			edges: g.edges.length / 2,
			levels: levels.length,
			clustersPerLevel: levels.map( ( l ) => clusterSizes( l ).length ),
			leafSizes: leaf,
			modularity: ( g.meta.hierarchy?.modularity || [] ).map( ( q ) => + q.toFixed( 4 ) ),
		};
		if ( asFile ) {

			if ( ! DRY ) writeFileSync( join( OUT, `${p.id}.json` ), json );
			written.add( `${p.id}.json` );

		} else {

			presets.push( { ...entry, generate: { model: p.spec.model, params: g.meta.params, seed: g.meta.seed }, estimatedBytes: bytes } );

		}

		rows.push( {
			id: p.id, tier: p.tier, model: p.spec.model, nodes: entry.nodes, edges: entry.edges, avgDeg: g.meta.stats.avgDegree,
			levels: `${entry.levels} (${entry.clustersPerLevel.join( '/' )})`,
			leaf: leaf ? `${leaf.min}/${leaf.median}/${leaf.max}` : '-',
			Q: entry.modularity.length ? entry.modularity[ entry.modularity.length - 1 ] : '-',
			stored: asFile ? `file ${kb( bytes )}` : `spec (${kb( bytes )} as file)`,
			ms: `${tGen.toFixed( 0 )}+${tCluster.toFixed( 0 )}`,
		} );

	} );

	if ( ! DRY ) {

		// remove stale outputs of this script (only when building the full ladder)
		if ( ! ONLY && existsSync( OUT ) ) {

			for ( const f of readdirSync( OUT ) ) if ( f.endsWith( '.json' ) && f !== 'presets.json' && ! written.has( f ) ) unlinkSync( join( OUT, f ) );

		}

		if ( ! ONLY ) writeFileSync( join( OUT, 'presets.json' ), JSON.stringify( { format: 'wgv-presets', version: 1, presets }, null, '\t' ) + '\n' );
		const ds = buildManifest();
		console.log( `manifest: ${ds.length} datasets` );

	}

	console.table( rows );

}

if ( import.meta.url === `file://${process.argv[ 1 ]}` ) build();
