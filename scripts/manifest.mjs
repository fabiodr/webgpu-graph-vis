// Scans public/data/**/*.json and writes public/data/manifest.json, which the app uses to populate
// the dataset picker. Run automatically by the data scripts.
//
// Entry fields: id, group, name, description, source, nodes, edges, levels, order, and either
//   file      "data/<group>/<id>.json" (wgv-graph files; also `bytes`), or
//   generate  { model, params, seed } for presets listed in a "wgv-presets" file (written by
//             scripts/generate.mjs for graphs too large to commit); the app generates them in the
//             data worker (DataClient.load(entry)), deterministically.
// Optional: tier ('readable' | 'overview' | 'stress' | 'sanity') for generated presets.
import { readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = 'public/data';
const GROUP_ORDER = [ 'imsv', 'generated' ];

function walk( dir ) {

	const out = [];
	for ( const f of readdirSync( dir ) ) {

		const p = join( dir, f );
		if ( statSync( p ).isDirectory() ) out.push( ...walk( p ) );
		else if ( f.endsWith( '.json' ) && f !== 'manifest.json' ) out.push( p );

	}

	return out;

}

export function buildManifest() {

	const datasets = [];
	for ( const file of walk( ROOT ) ) {

		const g = JSON.parse( readFileSync( file, 'utf8' ) );
		const rel = relative( ROOT, file ).split( '\\' ).join( '/' );
		const group = rel.split( '/' )[ 0 ];
		if ( g.format === 'wgv-presets' ) {

			for ( const p of g.presets || [] ) {

				datasets.push( {
					id: p.id,
					generate: p.generate,
					group: p.group || group,
					name: p.name,
					description: p.description || '',
					source: p.source || '',
					nodes: p.nodes,
					edges: p.edges,
					levels: p.levels,
					order: Number.isFinite( p.order ) ? p.order : 1e9,
					tier: p.tier,
				} );

			}

			continue;

		}

		if ( g.format !== 'wgv-graph' ) continue;
		datasets.push( {
			id: rel.replace( /\.json$/, '' ),
			file: 'data/' + rel,
			group,
			name: g.name,
			description: g.description || '',
			source: g.source || '',
			nodes: g.nodeCount,
			edges: g.edges.length / 2,
			levels: g.hierarchy ? g.hierarchy.levels.length : 0,
			bytes: statSync( file ).size,
			order: g.meta && Number.isFinite( g.meta.order ) ? g.meta.order : 1e9,
			tier: g.meta && g.meta.tier ? g.meta.tier : undefined,
		} );

	}

	datasets.sort( ( a, b ) => ( GROUP_ORDER.indexOf( a.group ) - GROUP_ORDER.indexOf( b.group ) ) || ( a.order - b.order ) || ( a.nodes - b.nodes ) );
	const groups = {
		imsv: 'ImsvGraphVis datasets',
		generated: 'Generated',
	};
	writeFileSync( join( ROOT, 'manifest.json' ), JSON.stringify( { groups, datasets }, null, '\t' ) + '\n' );
	return datasets;

}

if ( import.meta.url === `file://${process.argv[ 1 ]}` ) {

	const ds = buildManifest();
	console.log( `manifest: ${ds.length} datasets` );

}
