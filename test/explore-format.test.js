// Numbers a person reads in the explore layer: formatting, the binning of the histograms and the
// plain-language observations (which must stay silent unless the summary supports them).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatNumber, formatRatio, formatCount, compactCount, formatPercent, formatRank, formatRange, formatSeconds, ordinal, formatMetric } from '../src/explore/format.js';
import { binCounts, niceCeil } from '../src/explore/charts.js';
import { observe } from '../src/explore/observations.js';

test( 'formatRatio: fixed decimals, so coefficients side by side line up', () => {

	assert.equal( formatRatio( 0.4989 ), '0.50' );
	assert.equal( formatRatio( 0.5731 ), '0.57' );
	assert.equal( formatRatio( - 0.1652 ), '−0.17' );
	assert.equal( formatRatio( - 0.001 ), '0.00', 'no minus sign on a rounded zero' );
	assert.equal( formatRatio( 1 ), '1.00' );
	assert.equal( formatRatio( 0.1234, { decimals: 3 } ), '0.123' );
	assert.equal( formatRatio( NaN ), '–' );

} );

test( 'formatNumber: about three significant digits, grouped thousands, a real minus sign', () => {

	assert.equal( formatNumber( 0 ), '0' );
	assert.equal( formatNumber( 2.6349 ), '2.63' );
	assert.equal( formatNumber( 12.3456 ), '12.3' );
	assert.equal( formatNumber( 0.5 ), '0.5' );
	assert.equal( formatNumber( 0.0754 ), '0.0754' );
	assert.equal( formatNumber( 0.000123 ), '0.000123' );
	assert.equal( formatNumber( 1234567 ), '1,234,567' );
	assert.equal( formatNumber( - 2.5 ), '−2.5' );
	assert.equal( formatNumber( 5.457, { digits: 2 } ), '5.5' );
	assert.equal( formatNumber( NaN ), '–' );
	assert.equal( formatNumber( Infinity ), '–' );
	assert.equal( formatNumber( undefined ), '–' );

} );

test( 'counts, compact counts, percentages', () => {

	assert.equal( formatCount( 12 ), '12' );
	assert.equal( formatCount( 20000 ), '20,000' );
	assert.equal( formatCount( 1234567 ), '1,234,567' );
	assert.equal( compactCount( 999 ), '999' );
	assert.equal( compactCount( 1500 ), '1.5k' );
	assert.equal( compactCount( 20000 ), '20k' );
	assert.equal( compactCount( 999999 ), '1M', 'rolls over to the next unit, never "1000k"' );
	assert.equal( compactCount( 1234567 ), '1.2M' );
	assert.equal( formatPercent( 0.123 ), '12 %' );
	assert.equal( formatPercent( 0.0871 ), '8.7 %' );
	assert.equal( formatPercent( 1 ), '100 %' );
	assert.equal( formatPercent( 0 ), '0 %' );
	assert.equal( formatPercent( 0.00012 ), '< 0.1 %', 'small but not nothing' );
	assert.equal( formatPercent( NaN ), '–' );

} );

test( 'ranks, ranges, durations, ordinals', () => {

	assert.equal( formatRank( 3, 77 ), '#3 of 77' );
	assert.equal( formatRank( 1200, 20000 ), '#1,200 of 20,000' );
	assert.equal( formatRange( 1, 1 ), '1' );
	assert.equal( formatRange( 3, 4 ), '3–4' );
	assert.equal( formatRange( 1000, 2500 ), '1,000–2,500' );
	assert.equal( formatSeconds( 0.4 ), '0.4 s' );
	assert.equal( formatSeconds( 12 ), '12 s' );
	assert.equal( formatSeconds( 3700 ), '62 min' );
	assert.deepEqual( [ 1, 2, 3, 4, 11, 12, 13, 21, 22, 103, 111 ].map( ordinal ), [ '1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '103rd', '111th' ] );

} );

test( 'formatMetric: each metric in the unit a person expects', () => {

	assert.equal( formatMetric( 'degree', 36, 77 ), '36' );
	assert.equal( formatMetric( 'core', 8, 77 ), '8' );
	assert.equal( formatMetric( 'triangles', 1234, 5000 ), '1,234' );
	assert.equal( formatMetric( 'pagerank', 0.0754, 77 ), '0.0754' );
	assert.equal( formatMetric( 'clustering', 0.1213, 77 ), '0.121' );
	// betweenness is stored as a number of pairs: shown as the share of all pairs of other nodes
	assert.equal( formatMetric( 'betweenness', 75 * 76 / 2, 77 ), '100 %' );
	assert.equal( formatMetric( 'betweenness', 1000, 77 ), '35 %' );
	assert.equal( formatMetric( 'betweenness', 0, 77 ), '0 %' );
	assert.equal( formatMetric( 'betweenness', 0.5, 20000 ), '< 0.001 %', 'tiny shares do not read as zero' );
	const pairs = 19999 * 19998 / 2;
	assert.equal( formatMetric( 'betweenness', 0.0024 * pairs, 20000 ), '0.24 %', 'two significant digits below one percent' );
	assert.equal( formatMetric( 'betweenness', 0.0021 * pairs, 20000 ), '0.21 %' );
	assert.equal( formatMetric( 'betweenness', 0.003 * pairs, 20000 ), '0.3 %' );
	assert.equal( formatMetric( 'betweenness', 0.00021 * pairs, 20000 ), '0.021 %' );
	assert.equal( formatMetric( 'betweenness', 0.000034 * pairs, 20000 ), '0.003 %' );
	assert.equal( formatMetric( 'betweenness', 0.034 * pairs, 20000 ), '3.4 %' );
	assert.equal( formatMetric( 'degree', NaN, 77 ), '–' );

} );

// ── histogram binning ────────────────────────────────────────────────────────────────────────

const total = ( bins ) => bins.reduce( ( s, b ) => s + b.count, 0 );
const spans = ( bins ) => bins.map( ( b ) => ( b.from === b.to ? String( b.from ) : `${b.from}-${b.to}` ) );
const contiguous = ( bins ) => bins.every( ( b, k ) => b.from <= b.to && ( k === 0 || b.from === bins[ k - 1 ].to + 1 ) );

test( 'binCounts: one bar per value while they fit, without the empty ends', () => {

	assert.deepEqual( binCounts( [ 0, 5, 3, 0, 2 ] ), [ { from: 1, to: 1, count: 5 }, { from: 2, to: 2, count: 3 }, { from: 3, to: 3, count: 0 }, { from: 4, to: 4, count: 2 } ] );
	assert.deepEqual( spans( binCounts( [ 0, 0, 4, 0, 0, 1, 0, 0 ] ) ), [ '2', '3', '4', '5' ], 'gaps inside stay, as empty bars' );
	assert.deepEqual( binCounts( [ 7 ] ), [ { from: 0, to: 0, count: 7 } ], 'isolated nodes are a bar too' );
	assert.deepEqual( binCounts( [] ), [] );
	assert.deepEqual( binCounts( [ 0, 0, 0 ] ), [] );
	assert.deepEqual( binCounts( null ), [] );

} );

test( 'binCounts: a heavy tail gets doubling bins', () => {

	const counts = new Array( 101 ).fill( 0 );
	for ( let d = 1; d <= 100; d ++ ) counts[ d ] = Math.round( 60 / d );
	counts[ 100 ] = 1;
	const bins = binCounts( counts );
	assert.deepEqual( spans( bins ), [ '1', '2', '3-4', '5-8', '9-16', '17-32', '33-64', '65-100' ] );
	assert.equal( total( bins ), total( counts.map( ( count ) => ( { count } ) ) ), 'nothing lost' );
	assert.ok( contiguous( bins ) );
	assert.equal( bins[ 2 ].count, counts[ 3 ] + counts[ 4 ] );
	// the last bin ends at the largest value that occurs, not at the power of two
	assert.equal( bins.at( - 1 ).to, 100 );

	// a very wide range still fits into the bars it may use
	const wide = new Array( 5001 ).fill( 0 );
	wide[ 1 ] = 10;
	wide[ 40 ] = 3;
	wide[ 5000 ] = 1;
	const w = binCounts( wide );
	assert.ok( w.length <= 14, `${w.length} bars` );
	assert.equal( total( w ), 14 );
	assert.ok( contiguous( w ) );
	assert.deepEqual( [ w[ 0 ].from, w.at( - 1 ).to ], [ 1, 5000 ] );
	const narrow = binCounts( wide, { maxBars: 6 } );
	assert.ok( narrow.length <= 6, `${narrow.length} bars` );
	assert.equal( total( narrow ), 14 );
	assert.ok( contiguous( narrow ) );

} );

test( 'binCounts: a narrow range far from zero gets equal bins', () => {

	const counts = new Array( 40 ).fill( 0 );
	for ( let v = 20; v < 40; v ++ ) counts[ v ] = v;
	const bins = binCounts( counts, { maxBars: 8 } );
	assert.ok( bins.length <= 8 );
	assert.deepEqual( spans( bins ), [ '20-22', '23-25', '26-28', '29-31', '32-34', '35-37', '38-39' ] );
	assert.equal( total( bins ), total( counts.map( ( count ) => ( { count } ) ) ) );
	assert.deepEqual( spans( binCounts( counts, { mode: 'linear', maxBars: 4 } ) ), [ '20-24', '25-29', '30-34', '35-39' ] );
	// a forced mode
	const heavy = new Array( 101 ).fill( 1 );
	heavy[ 0 ] = 0;
	assert.equal( binCounts( heavy, { mode: 'linear', maxBars: 10 } ).length, 10 );
	assert.equal( binCounts( heavy, { mode: 'unit', maxBars: 10 } ).length, 100, 'unit keeps one bar per value' );

} );

test( 'niceCeil: a round maximum for the count axis', () => {

	assert.deepEqual( [ 0, 1, 3, 7, 12, 26, 51, 99, 100, 101, 259, 1234 ].map( niceCeil ), [ 1, 1, 3, 8, 15, 30, 60, 100, 100, 150, 300, 1500 ] );
	// the tallest bar fills at least two thirds of the plot
	for ( const v of [ 2, 9, 17, 259, 333, 4999, 20000 ] ) assert.ok( niceCeil( v ) >= v && niceCeil( v ) <= 1.5 * v, String( v ) );

} );

// ── observations ─────────────────────────────────────────────────────────────────────────────

const lesmis = {
	nodes: 77, edges: 254, density: 0.0868, assortativity: - 0.165,
	degree: { min: 1, max: 36, mean: 6.6, median: 6 },
	components: { count: 1, largest: 77, largestFraction: 1 },
	clustering: { average: 0.573, transitivity: 0.499 },
	core: { degeneracy: 9, sizes: [ 0, 18, 11, 7, 3, 0, 7, 11, 8, 12 ] },
	distance: { averagePathLength: 2.64, exact: true },
};

test( 'observe: says what the numbers support, strongest first', () => {

	const notes = observe( lesmis );
	assert.deepEqual( notes.map( ( n ) => n.id ), [ 'small-world', 'core' ] );
	assert.equal( notes[ 0 ].text, 'Small world: average distance 2.6 across 77 nodes, with clustering 0.57.' );
	assert.equal( notes[ 1 ].text, 'A dense core: 12 nodes, each linked to at least 9 others of the core.' );
	assert.equal( observe( lesmis, { limit: 1 } ).length, 1 );
	assert.equal( observe( lesmis, { limit: 0 } ).length, 0 );
	// a sampled distance is said as an estimate
	assert.match( observe( { ...lesmis, distance: { averagePathLength: 2.7, exact: false } } )[ 0 ].text, /average distance about 2\.7/ );

} );

test( 'observe: stays silent without support', () => {

	assert.deepEqual( observe( null ), [] );
	assert.deepEqual( observe( { nodes: 0, edges: 0 } ), [] );
	assert.deepEqual( observe( { nodes: 77, edges: 254 } ), [], 'a summary that has only just started' );
	// the first stage alone: degrees and components, nothing about distances yet
	const early = observe( { nodes: 77, edges: 254, density: 0.0868, degree: lesmis.degree, components: lesmis.components } );
	assert.deepEqual( early, [], 'max degree 36 is 5.5× the mean: under the threshold for "hub-dominated"' );
	// long paths: not a small world although it is clustered
	assert.equal( observe( { ...lesmis, core: null, distance: { averagePathLength: 9, exact: true } } ).some( ( n ) => n.id === 'small-world' ), false );
	// short paths without clustering (a random graph): not a small world either
	assert.equal( observe( { ...lesmis, core: null, clustering: { average: 0.05 } } ).some( ( n ) => n.id === 'small-world' ), false );
	// a core that is most of the graph is no core
	assert.equal( observe( { ...lesmis, distance: null, core: { degeneracy: 9, sizes: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 60 ] } } ).some( ( n ) => n.id === 'core' ), false );

} );

test( 'observe: components, trees, hubs, mixing, density', () => {

	const ids = ( s, limit = 9 ) => observe( s, { limit } ).map( ( n ) => n.id );
	const giant = observe( { nodes: 1000, edges: 3000, components: { count: 13, largest: 960, largestFraction: 0.96 } } );
	assert.equal( giant[ 0 ].id, 'giant' );
	assert.equal( giant[ 0 ].text, 'One giant component holds 96 % of the nodes; 12 small pieces are cut off from it.' );
	assert.match( observe( { nodes: 10, edges: 8, components: { count: 2, largest: 9, largestFraction: 0.9 } } )[ 0 ].text, /1 small piece is cut off/ );
	// two halves are not "a giant and a small piece": the count and the share, nothing more
	const halves = observe( { nodes: 100, edges: 98, components: { count: 2, largest: 50, largestFraction: 0.5 } } );
	assert.equal( halves[ 0 ].id, 'split' );
	assert.equal( halves[ 0 ].text, '2 separate components; the largest holds 50 % of the nodes.' );
	assert.equal( observe( { nodes: 100, edges: 98, components: { count: 2, largest: 55, largestFraction: 0.55, sizes: [ 55, 45 ] } } )[ 0 ].id, 'split' );
	assert.equal( observe( { nodes: 100, edges: 98, components: { count: 3, largest: 79, largestFraction: 0.79 } } )[ 0 ].id, 'split', 'below 80 % without the sizes' );
	// a majority next to nothing but crumbs is a giant, when the sizes say so
	const crumbs = observe( { nodes: 1000, edges: 900, components: { count: 401, largest: 600, largestFraction: 0.6, sizes: [ 600, 1, 1, 1 ] } } );
	assert.equal( crumbs[ 0 ].id, 'giant' );
	assert.match( crumbs[ 0 ].text, /^One giant component holds 60 % of the nodes; 400 small pieces/ );
	const pieces = observe( { nodes: 1000, edges: 600, components: { count: 400, largest: 30, largestFraction: 0.03 } } );
	assert.equal( pieces[ 0 ].id, 'fragmented' );
	assert.equal( pieces[ 0 ].text, 'Fragmented: 400 separate components, the largest with 3 % of the nodes.' );
	assert.deepEqual( ids( { nodes: 50, edges: 49, components: { count: 1, largest: 50, largestFraction: 1 } } ), [ 'tree' ] );
	assert.deepEqual( ids( { nodes: 50, edges: 50, components: { count: 1, largest: 50, largestFraction: 1 } } ), [], 'one edge more: a cycle' );
	const hubs = observe( { nodes: 5000, edges: 10000, degree: { min: 1, max: 480, mean: 4 } } );
	assert.equal( hubs[ 0 ].id, 'hubs' );
	assert.equal( hubs[ 0 ].text, 'Hub-dominated: the best-connected node has 480 links, 120× the average of 4.' );
	assert.deepEqual( ids( { nodes: 5000, edges: 10000, degree: { min: 1, max: 11, mean: 1.5 } } ), [], 'eleven links are not a hub' );
	assert.deepEqual( ids( { nodes: 1728, edges: 5184, degree: { min: 6, max: 6, mean: 6 } } ), [ 'uniform' ] );
	assert.deepEqual( ids( { nodes: 100, edges: 400, assortativity: 0.31 } ), [ 'assortative' ] );
	assert.deepEqual( ids( { nodes: 100, edges: 400, assortativity: - 0.4 } ), [ 'disassortative' ] );
	assert.deepEqual( ids( { nodes: 100, edges: 400, assortativity: 0.1 } ), [] );
	assert.deepEqual( ids( { nodes: 100, edges: 20, assortativity: 0.9 } ), [], 'too few links to say' );
	assert.deepEqual( ids( { nodes: 24, edges: 120, density: 0.43 } ), [ 'dense' ] );
	// several at once come ordered: structure, then degrees, then mixing
	assert.deepEqual( ids( { nodes: 1000, edges: 3000, assortativity: - 0.3, degree: { min: 0, max: 300, mean: 6 }, components: { count: 5, largest: 990, largestFraction: 0.99 } } ), [ 'giant', 'hubs', 'disassortative' ] );

} );
