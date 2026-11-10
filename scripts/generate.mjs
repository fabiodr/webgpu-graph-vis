// Builds the ladder of generated datasets (npm run data:generate).
//
// Every preset is a deterministic generator spec { model, params, seed } (src/data/generators.js)
// clustered with the default settings of the clustering front end (src/data/hierarchy.js:
// Leiden, seed 1, resolution 1, leaves of at most 200 nodes, the hierarchy ladder), exactly what
// the data worker computes on demand (test/data-pipeline.test.js compares the two). Presets whose
// wgv-graph JSON is at most MAX_FILE_BYTES are written to public/data/generated/<id>.json; larger
// ones are listed in public/data/generated/presets.json ("wgv-presets") with their spec and
// counts, so the app generates them in the worker instead of downloading megabytes. Then
// scripts/manifest.mjs rebuilds public/data/manifest.json (entries carry "file" or "generate").
//
//
// --samples writes public/data/samples/: one small graph (Les Misérables: 77 characters, 254
// co-appearances) in every file format the pipeline reads, for the format tests, dev/data.html
// and for trying drag and drop. Every file carries the names, a category attribute "community"
// (the top level of the stored hierarchy), a numeric attribute "degree" and edge weights
// (1 + the number of common neighbours, a quantity derived from the edges, not the original
// co-appearance counts), as far as the format can express them.
//
// Usage: node scripts/generate.mjs [--only id,id] [--max-bytes N] [--dry] | --samples
import { writeFileSync, mkdirSync, readdirSync, unlinkSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { generate } from '../src/data/generators.js';
import { applyClustering } from '../src/data/hierarchy.js';
import { serializeGraph, normalizeGraph } from '../src/data/format.js';
import { exportGraph } from '../src/data/export.js';
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
		id: 'galaxy-3k', tier: 'overview', name: 'Hubs and satellites · 3k',
		description: '6 groups of 8 systems: a hub with satellites, some with moons. Tree-like, strong hubs, a clear two-level hierarchy.',
		spec: { model: 'galaxy', params: { n: 3000, groups: 6, systems: 8, sizeSpread: 0.7, moons: 0.3, mesh: 0.15, bridges: 2 }, seed: 1 },
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
		id: 'mesh-torus', tier: 'sanity', name: 'Torus mesh 60×30',
		description: '1,800-node triangle mesh wrapped both ways. Layout sanity check: should unfold into a torus.',
		spec: { model: 'mesh', params: { nu: 60, nv: 30, wrapU: true, wrapV: true, triangles: true }, seed: 1 },
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
		const g = applyClustering( raw );
		delete g.clustering; // the worker derives it from meta.hierarchy when the file is loaded
		const tCluster = performance.now() - t;
		// (the generation time stays out of the file: it would change it on every run)
		const { ms: generatedIn, ...stats } = g.meta.stats;
		g.meta = { ...g.meta, stats, order, tier: p.tier, preset: p.id };
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

// ── sample files ─────────────────────────────────────────────────────────────────────────────

function samples() {

	const dir = 'public/data/samples';
	const src = normalizeGraph( JSON.parse( readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ) ) );
	const n = src.nodeCount, E = src.edges.length / 2;
	const names = src.labels;
	const top = src.hierarchy.levels[ src.hierarchy.levels.length - 1 ];
	const adj = Array.from( { length: n }, () => new Set() );
	for ( let e = 0; e < E; e ++ ) {

		adj[ src.edges[ 2 * e ] ].add( src.edges[ 2 * e + 1 ] );
		adj[ src.edges[ 2 * e + 1 ] ].add( src.edges[ 2 * e ] );

	}

	const degree = adj.map( ( a ) => a.size );
	const community = Array.from( top, ( c ) => `c${c}` );
	const weight = [];
	const pairs = [];
	for ( let e = 0; e < E; e ++ ) {

		const s = src.edges[ 2 * e ], t = src.edges[ 2 * e + 1 ];
		let common = 0;
		for ( const v of adj[ s ] ) if ( adj[ t ].has( v ) ) common ++;
		weight.push( 1 + common );
		pairs.push( [ s, t ] );

	}

	const categories = [ ...new Set( community ) ].sort();
	const g = normalizeGraph( {
		format: 'wgv-graph', version: 1, name: 'Les Misérables (sample)', nodeCount: n, labels: names,
		edges: Array.from( src.edges ), weights: weight,
		attributes: {
			community: { type: 'category', values: community.map( ( c ) => categories.indexOf( c ) ), categories },
			degree: { type: 'number', values: degree },
		},
	} );
	const q = ( v ) => JSON.stringify( String( v ) ); // a double-quoted string with escapes
	const xml = ( v ) => String( v ).replace( /[<>&"]/g, ( c ) => ( { '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[ c ] ) );
	const files = {};

	files[ 'lesmis.graphml' ] = exportGraph( g, 'graphml' ).text;
	files[ 'lesmis.csv' ] = exportGraph( g, 'csv' ).text;
	files[ 'lesmis.tsv' ] = '# Les Misérables: source, target, weight (tab separated, no header)\n' + pairs.map( ( [ s, t ], e ) => `${names[ s ]}\t${names[ t ]}\t${weight[ e ]}` ).join( '\n' ) + '\n';

	files[ 'lesmis.gexf' ] = [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<gexf xmlns="http://gexf.net/1.3" version="1.3">',
		'  <meta><description>Les Misérables (sample)</description></meta>',
		'  <graph defaultedgetype="undirected" mode="static">',
		'    <attributes class="node">',
		'      <attribute id="0" title="community" type="string"/>',
		'      <attribute id="1" title="degree" type="integer"/>',
		'    </attributes>',
		'    <nodes>',
		...names.map( ( name, i ) => `      <node id="${i}" label="${xml( name )}"><attvalues><attvalue for="0" value="${community[ i ]}"/><attvalue for="1" value="${degree[ i ]}"/></attvalues></node>` ),
		'    </nodes>',
		'    <edges>',
		...pairs.map( ( [ s, t ], e ) => `      <edge id="${e}" source="${s}" target="${t}" weight="${weight[ e ]}"/>` ),
		'    </edges>',
		'  </graph>',
		'</gexf>',
	].join( '\n' ) + '\n';

	files[ 'lesmis.gml' ] = [
		'Creator "scripts/generate.mjs --samples"',
		'graph [',
		'  directed 0',
		...names.map( ( name, i ) => `  node [ id ${i} label ${q( name )} community ${q( community[ i ] )} degree ${degree[ i ]} ]` ),
		...pairs.map( ( [ s, t ], e ) => `  edge [ source ${s} target ${t} value ${weight[ e ]} ]` ),
		']',
	].join( '\n' ) + '\n';

	files[ 'lesmis.dot' ] = [
		'// Les Misérables (sample): one cluster subgraph per community',
		'graph lesmis {',
		'  node [shape=ellipse];',
		...categories.map( ( c ) => `  subgraph cluster_${c} {\n    label=${q( c )};\n${names.map( ( name, i ) => ( community[ i ] === c ? `    ${q( name )} [degree=${degree[ i ]}];\n` : '' ) ).join( '' )}  }` ),
		...pairs.map( ( [ s, t ], e ) => `  ${q( names[ s ] )} -- ${q( names[ t ] )} [weight=${weight[ e ]}];` ),
		'}',
	].join( '\n' ) + '\n';

	files[ 'lesmis.net' ] = [
		`*Vertices ${n}`,
		...names.map( ( name, i ) => `${i + 1} ${q( name )}` ),
		'*Edges',
		...pairs.map( ( [ s, t ], e ) => `${s + 1} ${t + 1} ${weight[ e ]}` ),
		'*Partition community',
		`*Vertices ${n}`,
		...community.map( ( c ) => String( categories.indexOf( c ) + 1 ) ),
	].join( '\n' ) + '\n';

	files[ 'lesmis.mtx' ] = [
		'%%MatrixMarket matrix coordinate integer symmetric',
		'% Les Misérables (sample): weighted adjacency matrix, lower triangle',
		`${n} ${n} ${E}`,
		...pairs.map( ( [ s, t ], e ) => `${Math.max( s, t ) + 1} ${Math.min( s, t ) + 1} ${weight[ e ]}` ),
	].join( '\n' ) + '\n';

	files[ 'lesmis.node-link.json' ] = JSON.stringify( {
		directed: false, multigraph: false, graph: { name: 'Les Misérables (sample)' },
		nodes: names.map( ( name, i ) => ( { id: name, community: community[ i ], degree: degree[ i ] } ) ),
		links: pairs.map( ( [ s, t ], e ) => ( { source: names[ s ], target: names[ t ], value: weight[ e ] } ) ),
	} ) + '\n';

	files[ 'lesmis.cytoscape.json' ] = JSON.stringify( { elements: {
		nodes: names.map( ( name, i ) => ( { data: { id: `n${i}`, label: name, community: community[ i ], degree: degree[ i ] } } ) ),
		edges: pairs.map( ( [ s, t ], e ) => ( { data: { id: `e${e}`, source: `n${s}`, target: `n${t}`, weight: weight[ e ] } } ) ),
	} } ) + '\n';

	files[ 'lesmis.graphology.json' ] = JSON.stringify( {
		attributes: { name: 'Les Misérables (sample)' }, options: { type: 'undirected', multi: false, allowSelfLoops: false },
		nodes: names.map( ( name, i ) => ( { key: `n${i}`, attributes: { label: name, community: community[ i ], degree: degree[ i ] } } ) ),
		edges: pairs.map( ( [ s, t ], e ) => ( { key: `e${e}`, source: `n${s}`, target: `n${t}`, attributes: { weight: weight[ e ] } } ) ),
	} ) + '\n';

	files[ 'lesmis.adjacency.json' ] = JSON.stringify( Object.fromEntries( names.map( ( name, i ) => [ name, [ ...adj[ i ] ].sort( ( a, b ) => a - b ).map( ( j ) => names[ j ] ) ] ) ) ) + '\n';

	mkdirSync( dir, { recursive: true } );
	const rows = [];
	for ( const [ file, text ] of Object.entries( files ) ) {

		writeFileSync( join( dir, file ), text );
		rows.push( { file, bytes: Buffer.byteLength( text ) } );

	}

	writeFileSync( join( dir, 'index.json' ), JSON.stringify( { format: 'wgv-samples', version: 1, nodes: n, edges: E, weightSum: weight.reduce( ( a, b ) => a + b, 0 ), files: Object.keys( files ) }, null, '\t' ) + '\n' );
	console.table( rows );

}

if ( import.meta.url === `file://${process.argv[ 1 ]}` ) {

	if ( args.includes( '--samples' ) ) samples();
	else build();

}
