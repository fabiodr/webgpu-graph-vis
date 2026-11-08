// Scans public/data/**/*.json (wgv-graph files) and writes public/data/manifest.json, which the
// app uses to populate the dataset picker. Run automatically by the data scripts.
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
		if ( g.format !== 'wgv-graph' ) continue;
		const rel = relative( ROOT, file ).split( '\\' ).join( '/' );
		const group = rel.split( '/' )[ 0 ];
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
