// Fuzzy search of the explore layer: the ranking (exact > prefix > word start > substring >
// subsequence, ties by weight), folding, the incremental narrowing and the command registry.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { foldText, matchTier, matchRanges, FuzzyIndex, rankTexts, mergeRanked } from '../src/explore/fuzzy.js';
import { CommandRegistry } from '../src/explore/commands.js';

const indexOf = ( texts, weights, options ) => new FuzzyIndex( texts.length, ( i ) => texts[ i ], { weights, ...options } );
const ranked = ( index, texts, query, limit ) => Array.from( index.search( query, { limit } ).items, ( i ) => texts[ i ] );

test( 'foldText: case, accents and white space do not matter', () => {

	assert.equal( foldText( '  Les   Misérables ' ), 'les miserables' );
	assert.equal( foldText( 'Ångström' ), 'angstrom' );
	assert.equal( foldText( 'plain text' ), 'plain text' );
	assert.equal( foldText( null ), '' );
	assert.equal( foldText( 42 ), '42' );

} );

test( 'matchTier: one tier per kind of match, the best occurrence counts', () => {

	assert.equal( matchTier( 'valjean', 'valjean' ), 0 );
	assert.equal( matchTier( 'valjean', 'val' ), 1 );
	assert.equal( matchTier( 'mme valjean', 'val' ), 2 );
	assert.equal( matchTier( 'mme.magloire', 'mag' ), 2, 'a dot starts a word' );
	assert.equal( matchTier( 'interval', 'val' ), 3 );
	assert.equal( matchTier( 'interval valve', 'val' ), 2, 'a later word start beats an earlier inner match' );
	assert.equal( matchTier( 'valjean', 'vjn' ), 4 );
	assert.equal( matchTier( 'jean valjean', 'jv' ), 4, 'initials' );
	assert.equal( matchTier( 'valjean', 'njv' ), - 1, 'order matters' );
	assert.equal( matchTier( 'valjean', 'ljn' ), - 1, 'an abbreviation starts at the start of a word' );
	assert.equal( matchTier( 'lens: relevance to selection', 'cos' ), - 1, 'letters scattered over a text are no match' );
	assert.equal( matchTier( 'colour by betweenness', 'cos' ), - 1, 'a far letter has to start a word' );
	assert.equal( matchTier( 'colour by size', 'cos' ), 4, '… as here' );
	assert.equal( matchTier( 'science donaudampfschifffahrtsgesellschaft computational phonetics group', 'comp ling' ), - 1 );
	assert.equal( matchTier( 'aaaaaaaaaaaaaaaaaaaaxxxxxxxxb', 'aaaaaaaaab' ), - 1, 'no blow-up on repeated letters' );
	assert.equal( matchTier( 'valjean', 'x' ), - 1 );
	assert.equal( matchTier( 'valjean', '' ), - 1 );

} );

test( 'several words match word by word, in any order', () => {

	assert.equal( matchTier( 'department of computational linguistics', 'comp ling' ), 2, 'each word of the query starts a word' );
	assert.equal( matchTier( 'linguistics, computational', 'comp ling' ), 2, 'in any order' );
	assert.equal( matchTier( 'computational linguistics', 'computational linguistics' ), 0, 'as written: the usual tiers' );
	assert.equal( matchTier( 'computational linguistics lab', 'computational ling' ), 1 );
	assert.equal( matchTier( 'psycholinguistics, computational', 'comp ling' ), 3, 'a word found only inside another word' );
	assert.equal( matchTier( 'computational biology', 'comp ling' ), - 1, 'every word has to appear' );
	assert.equal( matchTier( 'c o m p l i n g', 'comp ling' ), 4, 'the letters in order are still the weakest kind of match' );
	assert.deepEqual( matchRanges( 'Linguistics, Computational', 'comp ling' ), [ [ 0, 4 ], [ 13, 17 ] ] );
	assert.deepEqual( matchRanges( 'Computational Linguistics', 'comp ling' ), [ [ 0, 4 ], [ 14, 18 ] ] );
	// typing on narrows correctly across the space
	const texts = [ 'Linguistics, Computational', 'Computational Biology', 'Department of Computational Linguistics', 'Compling' ];
	const index = new FuzzyIndex( texts.length, ( i ) => texts[ i ] );
	const seen = [];
	for ( const q of [ 'c', 'comp', 'comp ', 'comp l', 'comp ling' ] ) seen.push( Array.from( index.search( q ).items, ( i ) => texts[ i ] ) );
	assert.deepEqual( seen.at( - 1 ), [ 'Linguistics, Computational', 'Department of Computational Linguistics', 'Compling' ], 'word starts first, then words found inside a word' );
	assert.deepEqual( seen.at( - 1 ), Array.from( new FuzzyIndex( texts.length, ( i ) => texts[ i ] ).search( 'comp ling' ).items, ( i ) => texts[ i ] ) );
	assert.equal( seen[ 2 ].length, 4, '"comp " is "comp"' );

} );

test( 'search: exact > prefix > word start > substring > subsequence', () => {

	const texts = [ 'Marianne', 'Anna Maria', 'Mari', 'Rosemarie', 'Mxaxrxi', 'Other' ];
	const index = indexOf( texts, null );
	const r = index.search( 'mari' );
	assert.deepEqual( Array.from( r.items, ( i ) => texts[ i ] ), [ 'Mari', 'Marianne', 'Anna Maria', 'Rosemarie', 'Mxaxrxi' ] );
	assert.deepEqual( Array.from( r.tiers ), [ 0, 1, 2, 3, 4 ] );
	assert.equal( r.total, 5 );

} );

test( 'search: ties inside a tier go to the larger weight, then the shorter text, then the index', () => {

	const texts = [ 'Alpha one', 'Alpha two', 'Alpha', 'Alphabet', 'Alpine', 'Alps' ];
	const degree = [ 3, 9, 1, 9, 2, 2 ];
	const index = indexOf( texts, degree );
	// "alp": every text is a prefix match; degree 9 twice (shorter text first), then 3, 2, 2, 1
	assert.deepEqual( ranked( index, texts, 'alp' ), [ 'Alphabet', 'Alpha two', 'Alpha one', 'Alps', 'Alpine', 'Alpha' ] );
	// the tier still comes first: the exact match wins against any weight
	assert.deepEqual( ranked( index, texts, 'alpha' ).slice( 0, 2 ), [ 'Alpha', 'Alphabet' ] );
	assert.deepEqual( index.top( 2 ).map( ( i ) => texts[ i ] ), [ 'Alphabet', 'Alpha two' ] );

} );

test( 'search: the limit cuts the list, not the count, and keeps the best', () => {

	const texts = Array.from( { length: 500 }, ( _, i ) => ( i % 50 === 0 ? `zeta ${i}` : `item zeta ${i}` ) );
	const index = indexOf( texts, null );
	const r = index.search( 'zeta', { limit: 12 } );
	assert.equal( r.total, 500 );
	assert.equal( r.items.length, 12 );
	assert.ok( Array.from( r.tiers ).slice( 0, 10 ).every( ( t ) => t === 1 ), 'the ten prefix matches come first' );
	assert.deepEqual( Array.from( r.tiers ).slice( 10 ), [ 2, 2 ] );
	assert.equal( index.search( '', { limit: 12 } ).total, 0, 'an empty query matches nothing' );
	assert.equal( index.search( '   ' ).items.length, 0 );

} );

test( 'search: typing on gives the same result as a fresh search (narrowing)', () => {

	let seed = 7;
	const rand = () => ( seed = ( seed * 1103515245 + 12345 ) >>> 0 ) / 4294967296;
	const syllables = [ 'ka', 'ro', 'mi', 'ne', 'ta', 'lu', 'so', 'ver', 'an', 'el', 'ix', 'or', 'ma', 'ri' ];
	const texts = Array.from( { length: 3000 }, () => Array.from( { length: 2 + Math.floor( rand() * 3 ) }, () => syllables[ Math.floor( rand() * syllables.length ) ] ).join( rand() < 0.3 ? ' ' : '' ) );
	const weights = Array.from( { length: texts.length }, () => Math.floor( rand() * 40 ) );
	const typed = indexOf( texts, weights );
	for ( const word of [ 'mari', 'karo', 'an el', 'sovr', 'xx' ] ) {

		for ( let k = 1; k <= word.length; k ++ ) {

			const q = word.slice( 0, k );
			const a = typed.search( q, { limit: 40 } );
			const b = indexOf( texts, weights ).search( q, { limit: 40 } );
			assert.deepEqual( Array.from( a.items ), Array.from( b.items ), `"${q}"` );
			assert.deepEqual( Array.from( a.tiers ), Array.from( b.tiers ), `tiers of "${q}"` );
			assert.equal( a.total, b.total );

		}

		typed.search( '' );

	}

	// deleting characters widens again
	typed.search( 'mari' );
	assert.equal( typed.search( 'ma' ).total, indexOf( texts, weights ).search( 'ma' ).total );

} );

test( 'search: a node number finds that node, also after typing its first digit', () => {

	const texts = Array.from( { length: 40 }, ( _, i ) => ( i === 12 ? 'Valjean' : `Node ${i}` ) );
	const index = indexOf( texts, null, { numeric: true } );
	index.search( '1' );
	const r = index.search( '12' );
	assert.equal( r.items[ 0 ], 12, 'node 12 by its index although its label has no 12' );
	assert.equal( r.tiers[ 0 ], 0 );
	assert.equal( indexOf( texts, null ).search( '12' ).total, 0, 'only when numbers are node indices' );

} );

test( 'search: 20,000 labels', () => {

	const N = 20000;
	const texts = Array.from( { length: N }, ( _, i ) => `node-${( i * 2654435761 >>> 0 ).toString( 36 )}` );
	const weights = Float32Array.from( { length: N }, ( _, i ) => ( i * 7919 ) % 97 );
	const index = indexOf( texts, weights );
	const r = index.search( 'node' );
	assert.equal( r.total, N );
	assert.equal( r.items.length, 50 );
	for ( let k = 1; k < r.items.length; k ++ ) assert.ok( weights[ r.items[ k - 1 ] ] >= weights[ r.items[ k ] ], 'by weight' );
	assert.equal( index.search( 'node-zzzzzzzzzz' ).total, 0 );

} );

test( 'matchRanges: what to emphasise in a row', () => {

	assert.deepEqual( matchRanges( 'Valjean', 'val' ), [ [ 0, 3 ] ] );
	assert.deepEqual( matchRanges( 'Mme.Magloire', 'mag' ), [ [ 4, 7 ] ] );
	assert.deepEqual( matchRanges( 'interval valve', 'val' ), [ [ 9, 12 ] ], 'the occurrence at a word start' );
	assert.deepEqual( matchRanges( 'Valjean', 'vjn' ), [ [ 0, 1 ], [ 3, 4 ], [ 6, 7 ] ] );
	assert.deepEqual( matchRanges( 'Jean Valjean', 'jv' ), [ [ 0, 1 ], [ 5, 6 ] ], 'the characters the ranking counted' );
	assert.deepEqual( matchRanges( 'Start the tour', 'stt' ), [ [ 0, 2 ], [ 4, 5 ] ] );
	assert.deepEqual( matchRanges( 'Lens: Relevance to selection', 'cos' ), [] );
	assert.deepEqual( matchRanges( 'Valjean', 'vea' ), [ [ 0, 1 ], [ 4, 6 ] ], 'adjacent characters join' );
	assert.deepEqual( matchRanges( 'Valjean', 'xyz' ), [] );
	assert.deepEqual( matchRanges( 'Valjean', '' ), [] );

} );

test( 'rankTexts: short lists, stable inside a tier', () => {

	assert.deepEqual( rankTexts( [ 'Fit the graph', 'Graph tour', 'Paragraph', 'Tour' ], 'graph' ), [ { index: 1, tier: 1 }, { index: 0, tier: 2 }, { index: 2, tier: 3 } ] );
	assert.deepEqual( rankTexts( [ 'a', 'b' ], '' ), [] );

} );

// ── command registry ─────────────────────────────────────────────────────────────────────────

test( 'commands: register validates, replaces and unregisters', () => {

	const c = new CommandRegistry();
	const changes = [];
	c.onChange( () => changes.push( c.list().length ) );
	assert.throws( () => c.register( null ), TypeError );
	assert.throws( () => c.register( { id: '', title: 'x', run() {} } ), TypeError );
	assert.throws( () => c.register( { id: 'a', run() {} } ), TypeError );
	assert.throws( () => c.register( { id: 'a', title: 'A' } ), TypeError );
	const offA = c.register( { id: 'a', title: 'Alpha', run() {} } );
	c.register( { id: 'b', title: 'Beta', run() {} } );
	const offA2 = c.register( { id: 'a', title: 'Alpha again', run() {} } );
	assert.deepEqual( c.list().map( ( x ) => x.title ), [ 'Beta', 'Alpha again' ], 'a second registration replaces the first' );
	offA(); // the handle of the replaced entry must not remove its successor
	assert.equal( c.has( 'a' ), true );
	offA2();
	assert.equal( c.has( 'a' ), false );
	assert.equal( c.unregister( 'nope' ), false );
	assert.equal( c.get( 'b' ).title, 'Beta' );
	assert.equal( c.get( 'a' ), null );
	assert.deepEqual( changes, [ 1, 2, 2, 1 ] );

} );

test( 'commands: search ranks titles before keywords and hides disabled commands', () => {

	const c = new CommandRegistry();
	let selection = false;
	c.register( { id: 'tour', title: 'Start the tour', group: 'Explore', keywords: 'guide intro', run() {} } );
	c.register( { id: 'lens', title: 'Lens: Influence', group: 'Lens', keywords: [ 'colour', 'pagerank' ], run() {} } );
	c.register( { id: 'guide', title: 'Guide lines', run() {} } );
	c.register( { id: 'pin', title: 'Pin the selection', keywords: 'fix', enabled: () => selection, run() {} } );
	c.register( { id: 'broken', title: 'Pin everything', enabled: () => {

		throw new Error( 'no' );

	}, run() {} } );
	assert.deepEqual( c.search( 'guide' ).map( ( r ) => [ r.command.id, r.tier, r.field ] ), [ [ 'guide', 1, 'title' ], [ 'tour', 1, 'keyword' ] ], 'an exact keyword counts as a prefix, and after a title of the same tier' );
	assert.deepEqual( c.search( 'lens' ).map( ( r ) => r.command.id ), [ 'lens' ] );
	assert.deepEqual( c.search( 'pagerank' ).map( ( r ) => [ r.command.id, r.tier ] ), [ [ 'lens', 1 ] ], 'an exact keyword is never an exact match' );
	assert.deepEqual( c.search( 'pin' ), [], 'disabled, and one whose test throws' );
	selection = true;
	assert.deepEqual( c.search( 'pin' ).map( ( r ) => r.command.id ), [ 'pin' ] );
	assert.deepEqual( c.search( '' ), [] );
	assert.equal( c.search( 'i', { limit: 2 } ).length, 2 );
	assert.deepEqual( c.available().map( ( x ) => x.id ), [ 'tour', 'lens', 'guide', 'pin' ] );

} );

test( 'commands: run reports the outcome and feeds the suggestions', async () => {

	const c = new CommandRegistry();
	const ran = [];
	const errors = [];
	c.onError = ( e, command ) => errors.push( `${command.id}: ${e.message}` );
	c.register( { id: 'a', title: 'A', suggested: true, run: ( ctx ) => ran.push( [ 'a', ctx ] ) } );
	c.register( { id: 'b', title: 'B', run: async () => ran.push( [ 'b' ] ) } );
	c.register( { id: 'c', title: 'C', suggested: true, run() {} } );
	c.register( { id: 'off', title: 'Off', suggested: true, enabled: false, run: () => ran.push( [ 'off' ] ) } );
	c.register( { id: 'bad', title: 'Bad', run: () => {

		throw new Error( 'boom' );

	} } );
	assert.deepEqual( c.suggested().map( ( x ) => x.id ), [ 'a', 'c' ] );
	assert.equal( await c.run( 'b' ), true );
	assert.equal( await c.run( 'a', { from: 'test' } ), true );
	assert.equal( await c.run( 'off' ), false );
	assert.equal( await c.run( 'missing' ), false );
	const error = console.error;
	console.error = () => {};
	try {

		assert.equal( await c.run( 'bad' ), false );

	} finally {

		console.error = error;

	}

	assert.deepEqual( ran, [ [ 'b' ], [ 'a', { from: 'test' } ] ] );
	assert.deepEqual( errors, [ 'bad: boom' ] );
	assert.deepEqual( c.suggested().map( ( x ) => x.id ), [ 'bad', 'a', 'b', 'c' ], 'recently run first, each once' );
	assert.deepEqual( c.suggested( 2 ).map( ( x ) => x.id ), [ 'bad', 'a' ] );
	c.unregister( 'a' );
	assert.deepEqual( c.suggested().map( ( x ) => x.id ), [ 'bad', 'b', 'c' ] );

} );

test( 'mergeRanked: a better tier first, in a tier the nodes before the commands', () => {

	const order = ( nodes, commands, opts ) => mergeRanked( nodes, commands, opts ).map( ( b ) => `${b.kind[ 0 ]}${b.from}-${b.to}` ).join( ' ' );
	assert.equal( order( [], [] ), '' );
	assert.equal( order( [ 1, 1, 3 ], [] ), 'n0-3' );
	assert.equal( order( [], [ 1, 2 ] ), 'c0-2' );
	// the same tier: the nodes lead ( "co" finds Courfeyrac and Cosette before "Colour by …" )
	assert.equal( order( [ 1, 1 ], [ 1, 1, 1, 1, 1 ] ), 'n0-2 c0-5' );
	assert.equal( order( [ 1, 1, 3, 3 ], [ 1, 1 ] ), 'n0-2 c0-2 n2-4' );
	// a command that matches better than every node leads ( "ins" → Insights before a substring )
	assert.equal( order( [ 3, 3 ], [ 1, 4 ] ), 'c0-1 n0-2 c1-2' );
	// a worse command waits for every better node
	assert.equal( order( [ 1, 1, 1, 1, 1, 1, 1, 1 ], [ 2 ] ), 'n0-8 c0-1' );
	// hundreds of equally good nodes do not bury the commands of their tier: five, then the commands
	const many = new Array( 200 ).fill( 1 );
	assert.equal( order( many, [ 1, 1, 1 ] ), 'n0-5 c0-3 n5-200' );
	assert.equal( order( many, [ 1 ], { lead: 2 } ), 'n0-2 c0-1 n2-200' );
	assert.equal( order( [ 0, 1, 1, 1, 1, 1, 1, 2 ], [ 1, 2 ] ), 'n0-6 c0-1 n6-8 c1-2', 'the exact node and five of the prefix tier, the command, the rest' );

} );

test( 'mergeRanked: every entry once, each list in its own order, a node first whenever one is as good as the best command', () => {

	let seed = 7;
	const random = () => ( seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff ) / 0x7fffffff;
	const tiers = ( n ) => Array.from( { length: n }, () => Math.floor( random() * 5 ) ).sort( ( a, b ) => a - b );
	for ( let round = 0; round < 200; round ++ ) {

		const nodes = tiers( Math.floor( random() * 30 ) ), commands = tiers( Math.floor( random() * 6 ) );
		const blocks = mergeRanked( nodes, commands, { lead: 1 + Math.floor( random() * 6 ) } );
		const seen = { node: [], command: [] };
		for ( const b of blocks ) for ( let k = b.from; k < b.to; k ++ ) seen[ b.kind ].push( k );
		assert.deepEqual( seen.node, nodes.map( ( _, k ) => k ) );
		assert.deepEqual( seen.command, commands.map( ( _, k ) => k ) );
		for ( let k = 1; k < blocks.length; k ++ ) assert.notEqual( blocks[ k ].kind, blocks[ k - 1 ].kind, 'neighbouring blocks are merged' );
		if ( nodes.length && commands.length ) assert.equal( blocks[ 0 ].kind, nodes[ 0 ] <= commands[ 0 ] ? 'node' : 'command' );
		// a command never stands before a node of a better tier
		let worst = - 1;
		for ( const b of blocks ) {

			if ( b.kind === 'command' ) worst = Math.max( worst, commands[ b.to - 1 ] );
			else assert.ok( nodes[ b.from ] >= worst, 'a node after a command is not better than it' );

		}

	}

} );

test( 'palette order on a real index: "co" goes to a node, not to "Colour by …"', () => {

	const labels = [ 'Courfeyrac', 'Cosette', 'Valjean', 'Mme.Hucheloup', 'Bahorel' ];
	const index = new FuzzyIndex( labels.length, ( i ) => labels[ i ], { weights: [ 13, 11, 36, 7, 12 ] } );
	const commands = new CommandRegistry();
	for ( const metric of [ 'Degree', 'k-core', 'PageRank' ] ) commands.register( { id: 'color.' + metric, title: `Colour by ${metric}`, group: 'Lens', run() {} } );
	commands.register( { id: 'tour', title: 'Start the tour', group: 'Explore', run() {} } );
	const rows = ( query ) => {

		const found = index.search( query ), cmds = commands.search( query, { limit: 5 } );
		const out = [];
		for ( const b of mergeRanked( found.tiers, cmds.map( ( r ) => r.tier ) ) ) for ( let k = b.from; k < b.to; k ++ ) out.push( b.kind === 'node' ? labels[ found.items[ k ] ] : cmds[ k ].command.title );
		return out;

	};

	assert.deepEqual( rows( 'co' ), [ 'Courfeyrac', 'Cosette', 'Colour by Degree', 'Colour by k-core', 'Colour by PageRank' ] );
	assert.deepEqual( rows( 'c' ), [ 'Courfeyrac', 'Cosette', 'Colour by Degree', 'Colour by k-core', 'Colour by PageRank', 'Mme.Hucheloup' ] );
	assert.equal( rows( 'tour' )[ 0 ], 'Start the tour', 'no node matches: the command' );
	assert.equal( rows( 'col' )[ 0 ], 'Colour by Degree' );
	assert.deepEqual( rows( 'start' ), [ 'Start the tour' ] );

} );
