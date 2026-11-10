// Fuzzy text search of the explore layer: node labels and command titles, ranked by how a query
// matches. Pure functions, no DOM; built for a keystroke at 20k and more nodes.
//
// ─── Ranking ────────────────────────────────────────────────────────────────────────────────
//  A match has a tier, lower is better:
//    0 exact        the whole text equals the query
//    1 prefix       the text starts with the query
//    2 word start   a later word of the text starts with the query ( "jean" in "Val Jean" )
//    3 substring    the query appears somewhere inside a word
//                   A query of several words that does not appear as written matches word by
//                   word, in any order: tier 2 when every word starts a word of the text
//                   ( "comp ling" in "Linguistics, Computational" ), tier 3 when every word
//                   appears somewhere.
//    4 subsequence  the characters of the query appear in order, with gaps, the way a person
//                   abbreviates: the first one starts a word, every further one follows within
//                   three characters of the one before or starts a later word ( "vjn" in
//                   "Valjean", "jv" in "Jean Valjean" ). Letters scattered over a long text
//                   ( "cos" in "Relevance to selection" ) are not a match.
//  Ties are broken by weight (the node degree), larger first, then by the shorter text, then by
//  index. Text and query are compared folded: lower case, accents removed, runs of white space
//  collapsed. On a graph without labels the texts are "Node 12"; a query that is the plain number
//  of a node ( "12" ) is an exact match of that node.
//
// ─── Speed ──────────────────────────────────────────────────────────────────────────────────
//  The index folds every text once and keeps the node order by weight, so a query is one linear
//  scan in that order: the buckets come out sorted and nothing is sorted per keystroke. A query
//  that extends the previous one ( "va" → "val" ) only rescans the previous matches, because
//  whatever matches the longer query also matches the shorter one. A scan of 20k labels takes well under a millisecond
//  per tier test; 200k stay interactive.
//
// ─── API ────────────────────────────────────────────────────────────────────────────────────
//  foldText( s )                         -> folded string
//  matchTier( foldedText, foldedQuery )  -> 0 … 4, or −1 for no match
//  matchRanges( text, query )            -> [ [ start, end ), … ] character ranges of `text` to
//                                           emphasise (empty when the folding moved characters)
//  const index = new FuzzyIndex( count, ( i ) => text, { weights?: ArrayLike, numeric = false } )
//  index.search( query, { limit = 50 } ) -> { items: Uint32Array (ranked indices, at most limit),
//                                             tiers: Uint8Array (parallel), total } — total counts
//                                             every match, also those beyond the limit
//  index.text( i ), index.count, index.top( k ) -> the k heaviest indices
//  rankTexts( texts, query )             -> [ { index, tier } ] for short lists (commands)
//  mergeRanked( nodeTiers, commandTiers, { lead = 5 } ) -> [ { kind: 'node' | 'command', from, to } ]
//                                           the order in which two ranked lists (the tiers of the
//                                           node matches and of the command matches, each best
//                                           first) are shown as one: blocks of consecutive entries
//                                           [ from, to ) of either list. The box is a node search
//                                           first: a better tier always comes first, and inside a
//                                           tier the nodes do, so the first row is a node whenever
//                                           a node matches as well as the best command. A tier that
//                                           holds both shows at most `lead` of its nodes before its
//                                           commands, so a handful of commands is not buried under
//                                           hundreds of equally good nodes; the rest of the tier's
//                                           nodes follow the commands.

const MARKS = /[̀-ͯ]/g;
const SPACES = /\s+/g;
const WORD_BREAK = /[\s\-_./:,;()[\]{}'"|&+#@]/;

export function foldText( s ) {

	if ( s === null || s === undefined ) return '';
	let t = String( s );
	// fast path: plain lower-case ASCII needs no work
	if ( /^[a-z0-9 ]*$/.test( t ) ) return t.indexOf( '  ' ) < 0 ? t.trim() : t.replace( SPACES, ' ' ).trim();
	t = t.normalize( 'NFD' ).replace( MARKS, '' ).toLowerCase();
	return t.replace( SPACES, ' ' ).trim();

}

const isBreak = ( ch ) => WORD_BREAK.test( ch );

/** True when the characters of q appear in t in order. */
function subsequence( t, q ) {

	let k = 0;
	const n = q.length;
	for ( let i = 0; i < t.length && k < n; i ++ ) if ( t.charCodeAt( i ) === q.charCodeAt( k ) ) k ++;
	return k === n;

}

/** Characters that may lie between two characters of an abbreviation inside a word. */
const GAP = 3;
const startsWord = ( t, i ) => i === 0 || isBreak( t[ i - 1 ] );

/**
 * Reads q as an abbreviation of t: the first character starts a word, each further one comes at
 * most GAP characters after the one before or starts a later word. Returns the positions of the
 * characters in t (the tightest reading from the back), true / false when `positions` is not set.
 * Any prefix of an abbreviation is one too, which keeps the incremental narrowing of the index valid.
 */
function abbreviation( t, q, positions = false ) {

	const n = t.length, m = q.length;
	// ends: where the abbreviation read so far can end, ascending
	let ends = [];
	const c0 = q.charCodeAt( 0 );
	for ( let i = 0; i < n; i ++ ) if ( t.charCodeAt( i ) === c0 && startsWord( t, i ) ) ends.push( i );
	const levels = positions ? [ ends ] : null;
	for ( let k = 1; k < m && ends.length; k ++ ) {

		const c = q.charCodeAt( k ), next = [];
		let e = 0; // ends[ e ]: the last end before i
		for ( let i = ends[ 0 ] + 1; i < n; i ++ ) {

			if ( t.charCodeAt( i ) !== c ) continue;
			while ( e + 1 < ends.length && ends[ e + 1 ] < i ) e ++;
			if ( i - ends[ e ] <= GAP + 1 || startsWord( t, i ) ) next.push( i );

		}

		ends = next;
		if ( levels ) levels.push( ends );

	}

	if ( ! positions ) return ends.length > 0;
	if ( ! ends.length ) return null;
	// back from the first possible end: the nearest earlier end of each level before it
	const out = new Array( m );
	let at = ends[ 0 ];
	out[ m - 1 ] = at;
	for ( let k = m - 2; k >= 0; k -- ) {

		const list = levels[ k ];
		let j = list.length - 1;
		while ( j > 0 && list[ j ] >= at ) j --;
		at = list[ j ];
		out[ k ] = at;

	}

	return out;

}

export function matchTier( t, q ) {

	if ( q.length === 0 ) return - 1;
	let p = t.indexOf( q );
	if ( p === 0 ) return t.length === q.length ? 0 : 1;
	if ( p > 0 ) {

		// any occurrence at the start of a word beats one inside a word
		while ( p >= 0 ) {

			if ( isBreak( t[ p - 1 ] ) ) return 2;
			p = t.indexOf( q, p + 1 );

		}

		return 3;

	}

	// several words: every one of them has to appear, in any order ( "comp ling" finds
	// "Linguistics, Computational" )
	if ( q.indexOf( ' ' ) > 0 ) {

		const tier = tokenTier( t, q );
		if ( tier >= 0 ) return tier;

	}

	// one character alone would be a substring, and was not found; the cheap test sorts out most texts
	return q.length > 1 && subsequence( t, q ) && abbreviation( t, q ) ? 4 : - 1;

}

/** Where a word of the query is found in t: at the start of a word if there is such a place, else anywhere; −1 if nowhere. */
function findWord( t, word ) {

	let p = t.indexOf( word );
	const first = p;
	while ( p >= 0 ) {

		if ( p === 0 || isBreak( t[ p - 1 ] ) ) return p;
		p = t.indexOf( word, p + 1 );

	}

	return first;

}

/** Tier of a query of several words: 2 when each starts a word of t, 3 when each appears somewhere, −1 otherwise. */
function tokenTier( t, q ) {

	let tier = 2;
	for ( const word of q.split( ' ' ) ) {

		const p = findWord( t, word );
		if ( p < 0 ) return - 1;
		if ( p > 0 && ! isBreak( t[ p - 1 ] ) ) tier = 3;

	}

	return tier;

}

export function matchRanges( text, query ) {

	const t = foldText( text ), q = foldText( query );
	const raw = String( text );
	if ( ! q || t.length !== raw.length ) return [];
	let p = t.indexOf( q );
	if ( p >= 0 ) {

		// prefer the occurrence the ranking counted: text start, else a word start
		let best = p;
		while ( p > 0 ) {

			if ( isBreak( t[ p - 1 ] ) ) {

				best = p;
				break;

			}

			p = t.indexOf( q, p + 1 );

		}

		return [ [ best, best + q.length ] ];

	}

	if ( q.indexOf( ' ' ) > 0 && tokenTier( t, q ) >= 0 ) {

		// one range per word of the query, merged where they touch
		const found = q.split( ' ' ).map( ( word ) => [ findWord( t, word ), word.length ] ).sort( ( a, b ) => a[ 0 ] - b[ 0 ] );
		const out = [];
		for ( const [ at, n ] of found ) {

			const last = out[ out.length - 1 ];
			if ( last && at <= last[ 1 ] ) last[ 1 ] = Math.max( last[ 1 ], at + n );
			else out.push( [ at, at + n ] );

		}

		return out;

	}

	const at = q.length > 1 ? abbreviation( t, q, true ) : null;
	if ( ! at ) return [];
	const ranges = [];
	for ( const i of at ) {

		const last = ranges[ ranges.length - 1 ];
		if ( last && last[ 1 ] === i ) last[ 1 ] = i + 1;
		else ranges.push( [ i, i + 1 ] );

	}

	return ranges;

}

export class FuzzyIndex {

	constructor( count, textOf, { weights = null, numeric = false } = {} ) {

		this.count = count;
		this.numeric = numeric;
		this._textOf = textOf;
		this._folded = new Array( count );
		for ( let i = 0; i < count; i ++ ) this._folded[ i ] = foldText( textOf( i ) );
		// scan order: heaviest first, then the shorter text, then the index
		const order = new Uint32Array( count );
		for ( let i = 0; i < count; i ++ ) order[ i ] = i;
		const f = this._folded;
		if ( weights ) order.sort( ( a, b ) => weights[ b ] - weights[ a ] || f[ a ].length - f[ b ].length || a - b );
		else order.sort( ( a, b ) => f[ a ].length - f[ b ].length || a - b );
		this._order = order;
		this._lastQuery = '';
		this._lastMatches = null; // every match of the last query, in scan order

	}

	text( i ) {

		return this._textOf( i );

	}

	/** The k heaviest items (the scan order), e.g. the most connected nodes. */
	top( k ) {

		return Array.from( this._order.subarray( 0, Math.max( 0, k ) ) );

	}

	search( query, { limit = 50 } = {} ) {

		const q = foldText( query );
		const empty = { items: new Uint32Array( 0 ), tiers: new Uint8Array( 0 ), total: 0 };
		if ( ! q ) {

			this._lastQuery = '';
			this._lastMatches = null;
			return empty;

		}

		const f = this._folded;
		const exactNode = this.numeric && /^\d+$/.test( q ) ? + q : - 1;
		// a longer query can only match what the shorter one matched (a node number matches by
		// index, whatever the text says, so it scans everything)
		const narrow = exactNode < 0 && this._lastMatches && this._lastQuery && q.length > this._lastQuery.length && q.startsWith( this._lastQuery );
		const source = narrow ? this._lastMatches : this._order;
		const buckets = [ [], [], [], [], [] ];
		const matches = new Uint32Array( source.length );
		let total = 0;
		for ( let s = 0; s < source.length; s ++ ) {

			const i = source[ s ];
			const tier = i === exactNode ? 0 : matchTier( f[ i ], q );
			if ( tier < 0 ) continue;
			matches[ total ++ ] = i;
			const b = buckets[ tier ];
			if ( b.length < limit ) b.push( i );

		}

		this._lastQuery = q;
		this._lastMatches = matches.subarray( 0, total );

		const n = Math.min( limit, total );
		const items = new Uint32Array( n ), tiers = new Uint8Array( n );
		let k = 0;
		for ( let tier = 0; tier < 5 && k < n; tier ++ ) {

			const b = buckets[ tier ];
			for ( let j = 0; j < b.length && k < n; j ++ ) {

				items[ k ] = b[ j ];
				tiers[ k ++ ] = tier;

			}

		}

		return { items, tiers, total };

	}

}

/** Ranks a short list of texts (stable within a tier): [ { index, tier } ], best first. */
export function rankTexts( texts, query ) {

	const q = foldText( query );
	const out = [];
	if ( ! q ) return out;
	for ( let i = 0; i < texts.length; i ++ ) {

		const tier = matchTier( foldText( texts[ i ] ), q );
		if ( tier >= 0 ) out.push( { index: i, tier } );

	}

	out.sort( ( a, b ) => a.tier - b.tier || a.index - b.index );
	return out;

}

export function mergeRanked( nodeTiers, commandTiers, { lead = 5 } = {} ) {

	const n = nodeTiers.length, m = commandTiers.length;
	const blocks = [];
	const push = ( kind, from, to ) => {

		if ( to <= from ) return;
		const last = blocks[ blocks.length - 1 ];
		if ( last && last.kind === kind ) last.to = to;
		else blocks.push( { kind, from, to } );

	};

	lead = Math.max( 1, lead );
	let i = 0, c = 0;
	while ( i < n || c < m ) {

		const tier = Math.min( i < n ? nodeTiers[ i ] : Infinity, c < m ? commandTiers[ c ] : Infinity );
		let iEnd = i, cEnd = c;
		while ( iEnd < n && nodeTiers[ iEnd ] === tier ) iEnd ++;
		while ( cEnd < m && commandTiers[ cEnd ] === tier ) cEnd ++;
		// the nodes of the tier first; with commands in the same tier, a handful of them
		const head = cEnd > c ? Math.min( iEnd, i + lead ) : iEnd;
		push( 'node', i, head );
		push( 'command', c, cEnd );
		push( 'node', head, iEnd );
		i = iEnd;
		c = cEnd;

	}

	return blocks;

}
