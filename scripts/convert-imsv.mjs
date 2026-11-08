// Converts the preprocessed ImsvGraphVis datasets (Saved/Data/Graph/*.igv.json, which already
// contain the python-louvain clustering hierarchy used in the paper) into wgv-graph files under
// public/data/imsv/.
//
// Usage: node scripts/convert-imsv.mjs [path/to/ImsvGraphVis]
// Without a path, the repository is shallow-cloned into a temporary directory.
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { fromIgv, serializeGraph } from '../src/data/format.js';
import { buildManifest } from './manifest.mjs';

const DATASETS = [
	[ 'karate', "Zachary's karate club", 'Social network of a university karate club that split in two.', 'W. W. Zachary, J. Anthropol. Res. 33, 452 (1977)' ],
	[ 'dolphins', 'Dolphins', 'Frequent associations among 62 bottlenose dolphins in Doubtful Sound, New Zealand.', 'D. Lusseau et al., Behav. Ecol. Sociobiol. 54, 396 (2003)' ],
	[ 'lesmis', 'Les Misérables', 'Co-appearance of characters in Victor Hugo’s novel.', 'D. E. Knuth, The Stanford GraphBase (1993)' ],
	[ 'polbooks', 'Political books', 'Books about US politics co-purchased on Amazon around the 2004 election.', 'V. Krebs, unpublished; via M. Newman' ],
	[ 'adjnoun', 'Adjectives & nouns', 'Adjacency of common adjectives and nouns in Dickens’ David Copperfield.', 'M. E. J. Newman, Phys. Rev. E 74, 036104 (2006)' ],
	[ 'football', 'College football', 'Games between Division IA colleges, fall 2000 season.', 'M. Girvan and M. E. J. Newman, PNAS 99, 7821 (2002)' ],
	[ 'serengeti-foodweb', 'Serengeti food web', 'Feeding links among plants and herbivores of the Serengeti ecosystem.', 'E. B. Baskerville et al., PLoS Comput. Biol. 7, e1002321 (2011)' ],
	[ 'celegansneural', 'C. elegans neurons', 'Neural network of the nematode C. elegans (treated as undirected).', 'D. J. Watts and S. H. Strogatz, Nature 393, 440 (1998)' ],
];

let repo = process.argv[ 2 ];
if ( ! repo ) {

	repo = join( mkdtempSync( join( tmpdir(), 'imsv-' ) ), 'ImsvGraphVis' );
	execFileSync( 'git', [ 'clone', '--depth', '1', 'https://github.com/kwonoh/ImsvGraphVis.git', repo ], { stdio: 'inherit' } );

}

const dir = join( repo, 'Saved/Data/Graph' );
if ( ! existsSync( dir ) ) throw new Error( `Not found: ${dir}` );
mkdirSync( 'public/data/imsv', { recursive: true } );

DATASETS.forEach( ( [ id, name, description, source ], order ) => {

	const json = JSON.parse( readFileSync( join( dir, `${id}.igv.json` ), 'utf8' ) );
	const g = fromIgv( json, name );
	g.description = description;
	g.source = `${source}. Clustering hierarchy from ImsvGraphVis (O.-H. Kwon et al., IEEE TVCG 22(7), 2016).`;
	g.meta = { order, origin: `https://github.com/kwonoh/ImsvGraphVis/blob/master/Saved/Data/Graph/${id}.igv.json` };
	writeFileSync( `public/data/imsv/${id}.json`, serializeGraph( g ) + '\n' );
	console.log( `${id}: ${g.nodeCount} nodes, ${g.edges.length / 2} edges, ${g.hierarchy.levels.length} levels` );

} );

const ds = buildManifest();
console.log( `manifest: ${ds.length} datasets` );
