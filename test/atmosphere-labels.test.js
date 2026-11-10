// Community names and semantic zoom: the pure helpers of src/render/Labels.js and the CPU side of
// the community labels in src/render/TextLabels.js (level, names, cross-fade with the node labels,
// collisions), with an injected glyph source instead of a canvas.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PerspectiveCamera, Vector3 } from 'three/webgpu';
import { normalizeGraph } from '../src/data/format.js';
import { prepareGraph } from '../src/graph/prepare.js';
import {
	communityNames, communityFade, communityWeights, communityScale, communitySpacing, shortenName, COMMUNITY_SEPARATOR,
	nameFits, looseNameArea, nameScale, zoomLabelBudget, nameBudget,
} from '../src/render/Labels.js';
import { TextLabels, CTX } from '../src/render/TextLabels.js';

const close = ( a, b, eps, msg ) => assert.ok( Math.abs( a - b ) <= eps, `${msg || ''} ${a} vs ${b}` );
const lesmis = () => prepareGraph( normalizeGraph( JSON.parse( readFileSync( 'public/data/imsv/lesmis.json', 'utf8' ) ) ) );

// ── naming ───────────────────────────────────────────────────────────────────────────────────

test( 'communityNames: the label of the highest-degree member and the member count', () => {

	const g = lesmis();
	for ( let height = 1; height <= g.D; height ++ ) {

		const names = communityNames( g, height );
		assert.equal( names.length, g.levelOffset[ height ] - g.levelOffset[ height - 1 ] );
		let total = 0;
		for ( const r of names ) {

			total += r.count;
			assert.equal( g.clusterHeight[ r.cluster ], height );
			// the representative is a member and no member has a higher degree
			assert.equal( g.anc[ r.representative * g.D + height - 1 ], r.cluster );
			for ( let i = 0; i < g.N; i ++ ) if ( g.anc[ i * g.D + height - 1 ] === r.cluster ) {

				assert.ok( g.degree[ i ] < g.degree[ r.representative ] || ( g.degree[ i ] === g.degree[ r.representative ] && i >= r.representative ) );

			}

			assert.equal( r.name, String( g.labels[ r.representative ] ) );
			assert.equal( r.tail, COMMUNITY_SEPARATOR + r.count );
			assert.equal( r.text, r.name + r.tail );

		}

		assert.equal( total, g.N );

	}

	// the community of the protagonist carries his name
	const valjean = g.labels.indexOf( 'Valjean' );
	assert.ok( valjean >= 0 );
	const own = communityNames( g, 1 ).find( ( r ) => r.cluster === g.anc[ valjean * g.D ] );
	assert.match( own.text, /^Valjean · \d+$/ );

} );

test( 'communityNames: names of the data win, unlabelled data falls back to the count', () => {

	const g = lesmis();
	const c = g.anc[ 0 ];
	const named = communityNames( g, 1, { names: { [ c ]: '  The Bishop\'s household  ' } } );
	assert.equal( named.find( ( r ) => r.cluster === c ).name, 'The Bishop\'s household' );
	assert.ok( named.filter( ( r ) => r.cluster !== c ).every( ( r ) => r.name === String( g.labels[ r.representative ] ) ) );
	const viaMap = communityNames( g, 1, { names: new Map( [ [ c, 'Digne' ] ] ) } );
	assert.equal( viaMap.find( ( r ) => r.cluster === c ).text, 'Digne' + COMMUNITY_SEPARATOR + viaMap.find( ( r ) => r.cluster === c ).count );
	const fromGraph = communityNames( { ...g, clusterNames: { [ c ]: 'Digne' } }, 1 );
	assert.equal( fromGraph.find( ( r ) => r.cluster === c ).name, 'Digne' );

	const bare = communityNames( { ...g, labels: null }, 1 );
	for ( const r of bare ) {

		assert.equal( r.name, '' );
		assert.equal( r.text, r.count === 1 ? '1 node' : r.count + ' nodes' );

	}

	assert.deepEqual( communityNames( g, 0 ), [] );
	assert.deepEqual( communityNames( g, g.D + 1 ), [] );

} );

test( 'shortenName: long names are cut at a word with an ellipsis', () => {

	assert.equal( shortenName( 'Valjean' ), 'Valjean' );
	assert.equal( shortenName( '  padded  ' ), 'padded' );
	const cut = shortenName( 'The quick brown fox jumps over the lazy dog', 22 );
	assert.equal( cut, 'The quick brown fox…' );
	assert.ok( Array.from( cut ).length <= 22 );
	const solid = shortenName( 'Pneumonoultramicroscopicsilicovolcanoconiosis', 12 );
	assert.equal( Array.from( solid ).length, 12 );
	assert.ok( solid.endsWith( '…' ) );
	// counts characters, not UTF-16 units
	assert.equal( Array.from( shortenName( '\u{1F600}'.repeat( 30 ), 10 ) ).length, 10 );

} );

// ── cross-fade ───────────────────────────────────────────────────────────────────────────────

test( 'communityFade: 0 below lo, 1 above hi, smooth and monotone between, driven by spacing', () => {

	const members = 50;
	const radiusFor = ( spacing ) => spacing / Math.sqrt( Math.PI / members );
	assert.equal( communityFade( radiusFor( 5 ), members ), 0 );
	assert.equal( communityFade( radiusFor( 28 ), members ), 0 );
	assert.equal( communityFade( radiusFor( 44 ), members ), 1 );
	assert.equal( communityFade( radiusFor( 500 ), members ), 1 );
	close( communityFade( radiusFor( 36 ), members ), 0.5, 1e-9, 'midpoint' );
	let last = 0;
	for ( let s = 20; s <= 50; s += 0.25 ) {

		const t = communityFade( radiusFor( s ), members );
		assert.ok( t >= last - 1e-12 && t <= 1 );
		assert.ok( t - last < 0.04, 'no step' );
		last = t;

	}

	// the same radius resolves a small community earlier than a large one
	assert.ok( communityFade( 100, 10 ) > communityFade( 100, 30 ) );
	assert.equal( communityFade( 100, 10 ), 1 );
	assert.equal( communityFade( 100, 400 ), 0 );
	// custom range, degenerate input
	close( communityFade( radiusFor( 15 ), members, { lo: 10, hi: 20 } ), 0.5, 1e-9 );
	assert.equal( communityFade( 0, 10 ), 0 );
	assert.equal( communityFade( NaN, 10 ), 0 );
	assert.equal( communityFade( 50, 0 ), communityFade( 50, 1 ) );

} );

test( 'communityWeights: a cross-fade with a dip, names lead exactly in the first half', () => {

	assert.deepEqual( communityWeights( 0 ), { name: 1, nodes: 0 } );
	assert.deepEqual( communityWeights( 1 ), { name: 0, nodes: 1 } );
	const half = communityWeights( 0.5 );
	close( half.name, half.nodes, 1e-12, 'they cross at one half' );
	assert.ok( half.name > 0.3 && half.name < 0.45, 'with a dip' );
	let last = communityWeights( 0 );
	for ( let t = 0; t <= 1.0001; t += 1 / 64 ) {

		const w = communityWeights( t );
		assert.ok( w.name <= last.name + 1e-12 && w.nodes >= last.nodes - 1e-12, 'monotone' );
		assert.ok( w.name + w.nodes <= 1 + 1e-12, 'never a double exposure' );
		if ( Math.abs( t - 0.5 ) > 1e-9 ) assert.equal( w.name > w.nodes, t < 0.5, `the name leads up to one half (${t})` );
		last = w;

	}

	// node labels do not come in as a faint noise at the start of the zoom, names do not linger at its end
	assert.equal( communityWeights( 0.25 ).nodes, 0 );
	assert.equal( communityWeights( 0.75 ).name, 0 );

} );

test( 'communityScale: 1 … 1 + range with the square root of the size', () => {

	assert.equal( communityScale( 100, 100 ), 1.4 );
	close( communityScale( 25, 100 ), 1.2, 1e-12 );
	assert.ok( communityScale( 1, 100 ) > 1 && communityScale( 1, 100 ) < 1.05 );
	assert.equal( communityScale( 500, 100 ), 1.4, 'clamped' );
	assert.equal( communityScale( 0, 100 ), 1 );
	assert.equal( communityScale( 10, 0 ), 1 );
	assert.equal( communityScale( 100, 100, 1 ), 2 );

} );

test( 'communitySpacing: the smaller of the community\'s own and its finest communities\' estimate', () => {

	// no finer level: rms / √members
	assert.equal( communitySpacing( 30, 100 ), 3 );
	assert.equal( communitySpacing( 30, 2 ), 0, 'two nodes have no spacing of their own' );
	// a community of four dense clumps of 25 (rms 4 each) spread over rms 30: the clumps count
	const clumps = communitySpacing( 30, 100, 4 * Math.sqrt( 25 ) * 4, 100 );
	assert.equal( clumps, 4 / Math.sqrt( 25 ) );
	assert.ok( clumps < communitySpacing( 30, 100 ) );
	// a community of 1000 whose finest level is triples joined by long edges (rms 20 each): the
	// triples would say 11.5, the community as a whole says 0.95 and is right
	const triples = communitySpacing( 30, 1000, 333 * Math.sqrt( 3 ) * 20, 999 );
	assert.equal( triples, 30 / Math.sqrt( 1000 ) );
	assert.ok( 20 / Math.sqrt( 3 ) > 10 * triples );
	// a pair-sized community inside a level with fine statistics takes those
	assert.equal( communitySpacing( 5, 2, 12, 4 ), 3 );

} );

test( 'nameFits / looseNameArea: a name sits on its community, small projections get a few names, dots none', () => {

	// a name as wide as the community's projected diameter fits, a wider one does not
	assert.equal( nameFits( 100, 50 ), true );
	assert.equal( nameFits( 101, 50 ), false );
	assert.equal( nameFits( 120, 50, 1.25 ), true, 'the fit factor' );
	assert.equal( nameFits( 100, 0 ), false, 'behind the camera / no size' );
	// the far views measured in the app: a graph of 130 … 140 px radius has room for three or four names
	// of 100 … 160 × 26 px that do not fit their community
	const room = looseNameArea( 140 );
	assert.ok( room > 3 * 150 * 26 && room < 4 * 150 * 26, `${room}` );
	close( looseNameArea( 100, 0.25 ), 0.25 * Math.PI * 1e4, 1e-6 );
	// a graph of the size of a name: nothing
	assert.ok( looseNameArea( 40 ) < 150 * 26 );
	assert.equal( looseNameArea( 0 ), 0 );
	assert.equal( looseNameArea( - 5 ), 0 );
	assert.equal( looseNameArea( Infinity ), Infinity, 'the camera is inside the graph' );

} );

test( 'nameScale: no name is wider than a share of the view, down to a floor', () => {

	// desktop: untouched
	assert.equal( nameScale( 1.4, 120, 1280 ), 1.4 );
	// a phone: 170 px at scale 1.4 would be 41 % of 412 px; limited to 30 %
	const s = nameScale( 1.4, 121.4, 412 );
	close( s * 121.4, 0.3 * 412, 1e-9 );
	assert.ok( s < 1.4 && s > 0.85 );
	// very long names stop at the floor, short ones keep their own (smaller) scale
	assert.equal( nameScale( 1.4, 400, 412 ), 0.85 );
	assert.equal( nameScale( 0.8, 400, 412 ), 0.8 );
	assert.equal( nameScale( 1.2, 400, 412, { share: 0.5, floor: 1 } ), 1 );
	assert.equal( nameScale( 1.3, 0, 412 ), 1.3 );
	// monotone in the width of the view
	let last = 0;
	for ( let w = 200; w <= 2000; w += 50 ) {

		const v = nameScale( 1.4, 130, w );
		assert.ok( v >= last && v >= 0.85 && v <= 1.4 );
		last = v;

	}

} );

test( 'zoomLabelBudget / nameBudget: text by the size of the view', () => {

	assert.equal( zoomLabelBudget( 1280, 800 ), 51 );
	assert.equal( zoomLabelBudget( 412, 915 ), 19 );
	assert.equal( zoomLabelBudget( 320, 480 ), 16, 'the floor' );
	assert.equal( zoomLabelBudget( 2560, 1440 ), 80, 'the ceiling' );
	assert.equal( zoomLabelBudget( 1000, 1000, { area: 10000, max: 500 } ), 100 );
	// labels of 60 × 14 px: the budget covers a few percent of the view at most
	for ( const [ w, h ] of [ [ 412, 915 ], [ 1280, 800 ], [ 1920, 1080 ] ] ) assert.ok( zoomLabelBudget( w, h ) * 60 * 14 < 0.05 * w * h );
	assert.equal( nameBudget( 1280, 800 ), 13 );
	assert.equal( nameBudget( 412, 915 ), 6 );
	assert.equal( nameBudget( 3840, 2160 ), 24 );
	assert.ok( nameBudget( 1920, 1080 ) < zoomLabelBudget( 1920, 1080 ), 'names are larger and fewer than node labels' );

} );

// ── TextLabels: community names on the CPU side ──────────────────────────────────────────────

function fakeSource( size = 48 ) {

	const adv = ( ch ) => ( ch === ' ' ? 12 : 26 );
	const raster = ( unit ) => {

		const a = [ ...unit ].reduce( ( s, ch ) => s + adv( ch ), 0 );
		if ( unit === ' ' ) return { width: 0, height: 0, alpha: null, left: 0, top: 0, advance: a };
		const w = a - 4, h = 30;
		return { width: w, height: h, alpha: new Uint8Array( w * h ).fill( 255 ), left: 2, top: h, advance: a };

	};

	return { raster, measure: ( s ) => [ ...s ].reduce( ( w, ch ) => w + adv( ch ), 0 ), metrics: { ascent: 0.9, descent: 0.25, xHeight: 0.5 }, size };

}

/**
 * `sizes.length` communities on a line along x, 200 apart, each a blob of radius ~28 in the z = 0
 * plane; in every community the first node is the hub (connected to all other members).
 */
function fakeScene( options = {}, { sizes = [ 20, 20, 20 ], gap = 200, spread = 28, cameraZ = 400 } = {} ) {

	const N = sizes.reduce( ( a, b ) => a + b, 0 ), K = sizes.length;
	const anc = new Uint32Array( N ), degree = new Uint32Array( N ), labels = [];
	const positions = new Float32Array( 4 * N );
	const edges = [];
	let i = 0;
	for ( let c = 0; c < K; c ++ ) {

		const hub = i;
		for ( let k = 0; k < sizes[ c ]; k ++, i ++ ) {

			anc[ i ] = c;
			labels.push( k === 0 ? 'Hub' + c : 'n' + i );
			// a sunflower disc: uniform density, rms radius = spread / √2
			const r = spread * Math.sqrt( ( k + 0.5 ) / sizes[ c ] ), a = k * 2.399963;
			positions[ 4 * i ] = ( c - ( K - 1 ) / 2 ) * gap + r * Math.cos( a );
			positions[ 4 * i + 1 ] = r * Math.sin( a );
			if ( k > 0 ) edges.push( hub, i );

		}

	}

	for ( const v of edges ) degree[ v ] ++;
	const adjStart = new Uint32Array( N );
	for ( let v = 1; v < N; v ++ ) adjStart[ v ] = adjStart[ v - 1 ] + degree[ v - 1 ];
	const fill = adjStart.slice(), adjList = new Uint32Array( edges.length );
	for ( let e = 0; e < edges.length; e += 2 ) {

		adjList[ fill[ edges[ e ] ] ++ ] = edges[ e + 1 ]; adjList[ fill[ edges[ e + 1 ] ] ++ ] = edges[ e ];

	}

	const clusterHeight = new Uint8Array( K + 1 ).fill( 1 );
	clusterHeight[ K ] = 2;
	const graph = {
		N, E: edges.length / 2, D: 1, C: K + 1, root: K, anc, levelOffset: Uint32Array.from( [ 0, K, K + 1 ] ), clusterHeight,
		clusterParent: new Int32Array( K + 1 ).fill( K ), clusterMass: Float32Array.from( [ ...sizes, N ] ),
		degree, adjStart, adjList, labels, radii: new Float32Array( N ).fill( 1 ), layoutRadius: 300,
	};
	const camera = new PerspectiveCamera( 45, 1600 / 1000, 0.5, 20000 );
	const u = ( v ) => ( { value: v } );
	const buffers = { uniforms: { selectedIndex: u( - 1 ), hoverIndex: u( - 1 ), focus: u( 0 ) }, nodePos: { value: { array: positions } } };
	const app = { width: 1600, height: 1000, camera, positions, positionsVersion: 1, modules: [] };
	const moveCamera = ( z, x = 0 ) => {

		camera.position.set( x, 0, z );
		camera.lookAt( new Vector3( x, 0, 0 ) );
		camera.updateMatrixWorld();

	};

	moveCamera( cameraZ );
	const L = new TextLabels( { source: fakeSource(), ...options } );
	L.init( app );
	L.setGraph( buffers, graph, app );
	return { L, app, buffers, graph, moveCamera };

}

const nodeAlpha = ( L, i ) => L._L[ 4 * i + 1 ];
const shownNodes = ( L ) => [ ...L._active ].filter( ( i ) => nodeAlpha( L, i ) > 0 );
const nameAlpha = ( L ) => ( L._cl && L._cl.built ? L._cl.slots.map( ( _, j ) => L._cl.S[ 8 * j + 1 ] ) : [] );
// px per world unit at distance z for the fake camera (45°, 1000 px high)
const pxPerUnit = ( z ) => ( 1 / Math.tan( Math.PI / 8 ) ) * 500 / z;

test( 'community names: level, texts, sizes and glyph records', () => {

	const { L, app } = fakeScene( {}, { sizes: [ 30, 12, 20, 2 ] } );
	assert.equal( L.clusterLabels, 'auto' );
	assert.equal( L.clusterText( 0 ), 'Hub0' + COMMUNITY_SEPARATOR + '30' );
	assert.equal( L.clusterText( 3 ), null, 'two nodes are no community worth a name' );
	L._decide( app, 0, - 1, - 1 );
	const cl = L._cl;
	assert.equal( cl.height, 1 );
	assert.deepEqual( cl.slots.map( ( s ) => s.cluster ), [ 0, 2, 1 ], 'largest first' );
	assert.equal( cl.slots[ 0 ].scale, 1.4 );
	assert.ok( cl.slots[ 2 ].scale > 1 && cl.slots[ 2 ].scale < cl.slots[ 1 ].scale );

	// glyphs: "Hub0" in style 0, "· 30" in style 1 (the space has no glyph), centred on the origin
	const A = cl.glyphNode.value.array;
	const first = [];
	for ( let r = 0; r < cl.G; r ++ ) if ( A[ 8 * r ] === 0 ) first.push( { x: A[ 8 * r + 1 ], style: A[ 8 * r + 3 ], w: A[ 8 * r + 6 ] } );
	assert.deepEqual( first.map( ( f ) => f.style ), [ 0, 0, 0, 0, 1, 1, 1 ] );
	for ( let k = 1; k < first.length; k ++ ) assert.ok( first[ k ].x > first[ k - 1 ].x, 'left to right' );
	const S = L.atlas.size, slot = cl.slots[ 0 ];
	const left = first[ 0 ].x, right = first[ first.length - 1 ].x + first[ first.length - 1 ].w / S;
	close( ( left + right ) / 2, 0, 0.3, 'centred' );
	// the cells carry the SDF margin on both sides of the ink
	assert.ok( Math.abs( slot.widthEm - ( right - left ) ) < 2 * L.atlas.buffer / S + 0.2, `${slot.widthEm} vs ${right - left}` );
	assert.equal( L.clusterMesh.count, 2 * cl.G );
	assert.equal( L.clusterMesh.parent, L.mesh );
	assert.equal( L.stats.clusterGlyphs, cl.G );

	// records: font scale and cluster index per name
	for ( let j = 0; j < cl.K; j ++ ) {

		assert.equal( cl.S[ 8 * j + 3 ], Math.fround( cl.slots[ j ].scale ) );
		assert.equal( cl.S[ 8 * j + 4 ], cl.slots[ j ].cluster );

	}

} );

test( 'semantic zoom: far = names, near = node labels, between a cross-fade per community', () => {

	const { L, app, moveCamera, graph } = fakeScene();
	// rms of a uniform disc of radius 28 is 19.8; projected radius 1.29 · rms; spacing = r · √( π / 20 )
	const spacing = ( z ) => 1.29 * 19.8 * pxPerUnit( z ) * Math.sqrt( Math.PI / 20 );
	const distanceFor = ( px ) => 1.29 * 19.8 * ( 1 / Math.tan( Math.PI / 8 ) ) * 500 * Math.sqrt( Math.PI / 20 ) / px;
	const q = ( v ) => Math.round( v * 16 ) / 16;

	moveCamera( 1400 );
	assert.ok( spacing( 1400 ) < 28 );
	L._decide( app, 0, - 1, - 1 );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1 ] );
	assert.equal( shownNodes( L ).length, 0, 'no node label while every community is a name' );
	assert.equal( L.stats.clusters, 3 );
	assert.equal( L.stats.labels, 0 );

	moveCamera( distanceFor( 50 ) );
	L._decide( app, 1, - 1, - 1 );
	assert.deepEqual( nameAlpha( L ), [ 0, 0, 0 ] );
	const near = shownNodes( L );
	assert.ok( near.length >= 10, `node labels (${near.length})` );
	for ( const i of near ) assert.equal( nodeAlpha( L, i ), 1 );
	assert.equal( L.stats.clusters, 0 );

	// first half: the names lead and keep their place, node labels come in around them
	moveCamera( distanceFor( 34.5 ) );
	L._decide( app, 2, - 1, - 1 );
	let t = L._cl.fade[ 0 ];
	assert.ok( t > 0.25 && t < 0.5, `t = ${t}` );
	assert.equal( t * 16, Math.round( t * 16 ), 'quantised: no upload for every pixel of zoom' );
	let w = communityWeights( t );
	let mid = shownNodes( L );
	assert.ok( mid.length > 0 );
	for ( const i of mid ) close( nodeAlpha( L, i ), q( w.nodes ), 1e-6 );
	for ( const a of nameAlpha( L ) ) close( a, q( w.name ), 1e-6, 'dominant names keep their place' );
	for ( const i of mid ) assert.equal( L._L[ 4 * i + 3 ], CTX );

	// second half: the node labels lead, names only where room is left
	moveCamera( distanceFor( 38 ) );
	L._decide( app, 3, - 1, - 1 );
	t = L._cl.fade[ 0 ];
	assert.ok( t > 0.5 && t < 0.75, `t = ${t}` );
	w = communityWeights( t );
	mid = shownNodes( L );
	for ( const i of mid ) close( nodeAlpha( L, i ), q( w.nodes ), 1e-6 );
	for ( const a of nameAlpha( L ) ) assert.ok( a === 0 || Math.abs( a - q( w.name ) ) < 1e-6 );
	assert.ok( mid.includes( 0 ) && mid.includes( 20 ) && mid.includes( 40 ), 'the hubs carry their own label now' );
	assert.equal( graph.N, 60 );

} );

test( 'semantic zoom is per community: the near one shows its nodes, the far ones their names', () => {

	// the camera sits close to the first community and looks along the row
	const { L, app } = fakeScene( {}, { sizes: [ 20, 20, 20 ], gap: 260 } );
	app.camera.position.set( - 260 - 190, 0, 70 );
	app.camera.lookAt( new Vector3( 0, 0, 0 ) );
	app.camera.updateMatrixWorld();
	L._decide( app, 0, - 1, - 1 );
	const fade = Array.from( L._cl.fade );
	assert.ok( fade[ 0 ] > fade[ 1 ] && fade[ 1 ] >= fade[ 2 ], `nearer communities resolve first: ${fade}` );
	assert.equal( fade[ 0 ], 1 );
	assert.ok( fade[ 2 ] < 0.5 );
	const nodes = shownNodes( L );
	assert.ok( nodes.some( ( i ) => i < 20 ), 'labels in the near community' );
	for ( const i of nodes ) close( nodeAlpha( L, i ), L._cl.nodes[ Math.floor( i / 20 ) ], 1e-6 );

} );

test( 'clusterLabels modes: off keeps the old behaviour, on keeps every name, mode all keeps every node label', () => {

	const far = 1400;
	{

		const { L, app, moveCamera } = fakeScene( { clusterLabels: 'off' } );
		moveCamera( far );
		L._decide( app, 0, - 1, - 1 );
		assert.ok( shownNodes( L ).length > 0, 'context labels as before' );
		for ( const i of shownNodes( L ) ) assert.equal( nodeAlpha( L, i ), 1 );
		assert.equal( L._cl, null, 'nothing is built for names that are off' );
		assert.equal( L.stats.clusters, 0 );
		assert.equal( L.clusterMesh.visible, false );

	}

	{

		const { L, app, moveCamera } = fakeScene( { clusterLabels: 'on' } );
		moveCamera( 220 ); // only the middle community is on screen
		L._decide( app, 0, - 1, - 1 );
		assert.equal( L._cl.fade[ 1 ], 1 );
		assert.deepEqual( nameAlpha( L ), [ 0, 1, 0 ], 'the name stays at close range' );
		assert.ok( shownNodes( L ).length > 0 );
		for ( const i of shownNodes( L ) ) assert.equal( nodeAlpha( L, i ), 1 );
		// names without node labels
		L.mode = 'off';
		L._decide( app, 1, - 1, - 1 );
		assert.equal( shownNodes( L ).length, 0 );
		assert.deepEqual( nameAlpha( L ), [ 0, 1, 0 ] );
		// 'auto' follows the node labels' switch
		L.clusterLabels = 'auto';
		L._decide( app, 2, - 1, - 1 );
		assert.ok( nameAlpha( L ).every( ( a ) => a === 0 ) );

	}

	{

		const { L, app, moveCamera } = fakeScene( { mode: 'all' } );
		moveCamera( far );
		L._decide( app, 0, - 1, - 1 );
		assert.deepEqual( nameAlpha( L ), [ 1, 1, 1 ] );
		assert.ok( shownNodes( L ).length > 0, 'mode all: node labels at every distance' );
		for ( const i of shownNodes( L ) ) assert.equal( nodeAlpha( L, i ), 1 );

	}

	const L = new TextLabels( { source: fakeSource() } );
	L.clusterLabels = true; assert.equal( L.clusterLabels, 'on' );
	L.clusterLabels = false; assert.equal( L.clusterLabels, 'off' );
	L.clusterLabels = 'nonsense'; assert.equal( L.clusterLabels, 'auto' );

} );

test( 'big graphs: no node labels at overview, the labels of a community the camera has resolved', () => {

	const { L, app, moveCamera } = fakeScene( { autoMaxNodes: 30 }, { sizes: [ 40, 40, 40 ], gap: 300, spread: 40 } );
	moveCamera( 2000 );
	L._decide( app, 0, - 1, - 1 );
	assert.equal( shownNodes( L ).length, 0 );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1 ] );

	// fly to the middle community
	moveCamera( 240 );
	L._decide( app, 1, - 1, - 1 );
	assert.equal( L._cl.fade[ 1 ], 1 );
	const nodes = shownNodes( L );
	assert.ok( nodes.length >= 8, `labels of the resolved community (${nodes.length})` );
	assert.ok( nodes.includes( 40 ), 'its hub' );
	assert.ok( nodes.length <= L.maxLabels );

	// with names off a big graph has no context labels at all, as before
	L.clusterLabels = 'off';
	L._decide( app, 2, - 1, - 1 );
	assert.equal( shownNodes( L ).length, 0 );
	assert.ok( nameAlpha( L ).every( ( a ) => a === 0 ) );

} );

test( 'collisions: names do not overlap each other, the larger community wins; a selection goes first', () => {

	// two communities 30 apart: their names would lie on top of each other (near enough for both to
	// fit their community: from afar this pair is a dot that gets no name at all, see the fit test)
	const { L, app, moveCamera, buffers } = fakeScene( {}, { sizes: [ 30, 12 ], gap: 30 } );
	moveCamera( 600 );
	L._decide( app, 0, - 1, - 1 );
	assert.deepEqual( Array.from( L._cl.fade ), [ 0, 0 ], 'both are names at this distance' );
	assert.deepEqual( nameAlpha( L ), [ 1, 0 ], 'only the larger one is named' );
	assert.equal( L.stats.clusters, 1 );
	assert.equal( L._cl.loose, 0 );

	// apart: both
	const b = fakeScene( {}, { sizes: [ 30, 12 ], gap: 700 } );
	b.moveCamera( 1600 );
	b.L._decide( b.app, 0, - 1, - 1 );
	assert.deepEqual( nameAlpha( b.L ), [ 1, 1 ] );

	// the selected node's label is placed first and keeps its place: a name that would cover it yields
	const c = fakeScene( {}, { sizes: [ 30, 12 ], gap: 700 } );
	c.moveCamera( 1600 );
	const hub = 0;
	c.buffers.uniforms.selectedIndex.value = hub;
	c.L._decide( c.app, 0, hub, - 1 );
	assert.equal( nodeAlpha( c.L, hub ), 1 );
	assert.equal( nameAlpha( c.L )[ 0 ], 0, 'the name under the selection label gives way' );
	assert.equal( nameAlpha( c.L )[ 1 ], 1 );
	assert.equal( buffers.uniforms.selectedIndex.value, - 1 );

} );

test( 'a community that is spread through the graph gets no name', () => {

	const { L, app, moveCamera, graph } = fakeScene( {}, { sizes: [ 20, 20, 20 ], gap: 200 } );
	// scatter the members of community 1 over the whole row
	for ( let k = 0; k < 20; k ++ ) app.positions[ 4 * ( 20 + k ) ] = - 300 + 30 * k + ( k % 2 ) * 7;
	moveCamera( 1400 );
	L._decide( app, 0, - 1, - 1 );
	const alphaOf = ( cluster ) => nameAlpha( L )[ L._cl.slots.findIndex( ( s ) => s.cluster === cluster ) ];
	assert.equal( alphaOf( 1 ), 0 );
	assert.equal( alphaOf( 0 ), 1 );
	assert.equal( alphaOf( 2 ), 1 );
	assert.equal( graph.D, 1 );

} );

test( 'names ease like node labels, survive a rebase and are rebuilt with the level and the font', () => {

	const { L, app, moveCamera } = fakeScene();
	moveCamera( 1400 );
	L._decide( app, 10, - 1, - 1 );
	close( L.clusterAlphaOf( 0, 10 ), 0, 1e-9 );
	close( L.clusterAlphaOf( 0, 10 + L.fadeTime / 2 ), 0.5, 1e-6 );
	close( L.clusterAlphaOf( 0, 10 + L.fadeTime ), 1, 1e-9 );
	// retarget half way: the ease restarts from the current value
	moveCamera( 220 );
	L._decide( app, 10 + L.fadeTime / 2, - 1, - 1 );
	close( L.clusterAlphaOf( 0, 10 + L.fadeTime / 2 ), 0.5, 1e-6 );
	close( L.clusterAlphaOf( 0, 10 + 2 * L.fadeTime ), 0, 1e-9 );

	L._flush();
	assert.equal( L._cl.live, 3, 'still fading out' );
	L.uniforms.time.value = 100;
	L._flush();
	assert.equal( L._cl.live, 0 );
	assert.equal( L.clusterMesh.visible, false );

	const shifted = L._rebase( 3000 );
	assert.ok( shifted < 3000 && L._cl.S[ 2 ] >= - 1e3 );

	const before = L._cl;
	L.clusterLevel = 1;
	assert.equal( L._cl, null, 'a new level drops the names' );
	moveCamera( 1400 );
	L._decide( app, 20, - 1, - 1 );
	assert.notEqual( L._cl, before );
	assert.equal( L._cl.height, 1 );
	L.rebuildFont();
	assert.equal( L._cl, null );
	L._decide( app, 21, - 1, - 1 );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1 ] );
	L.maxClusterLabels = 1;
	L._decide( app, 22, - 1, - 1 );
	assert.deepEqual( nameAlpha( L ), [ 1 ] );
	L.maxClusterLabels = 0;
	L._decide( app, 23, - 1, - 1 );
	assert.deepEqual( nameAlpha( L ), [] );
	L.dispose();

} );

test( 'graphs without a hierarchy have no community names and behave as before', () => {

	const { L, app, graph } = fakeScene();
	const flat = { ...graph, D: 0 };
	L.setGraph( { uniforms: { selectedIndex: { value: - 1 }, hoverIndex: { value: - 1 }, focus: { value: 0 } }, nodePos: { value: { array: app.positions } } }, flat, app );
	L._decide( app, 0, - 1, - 1 );
	assert.ok( shownNodes( L ).length > 0 );
	assert.equal( L.stats.clusters, 0 );
	assert.equal( L.clusterText( 0 ), null );

} );

// ── names fit what they name; the zoom's budget ──────────────────────────────────────────────

/** Width in CSS px of the name of slot j at its largest size (a community that is large on screen). */
const nameWidth = ( L, j ) => L._cl.slots[ j ].widthEm * L.fontSize * L.uniforms.clusterScale.value * L._cl.slots[ j ].shown;
/** … and as it is shown above a community whose ball has a radius of ballPx: 0.22 radii, at least 11 px. */
const shownWidth = ( L, j, ballPx ) => Math.min( nameWidth( L, j ), L._cl.slots[ j ].widthEm * Math.max( 11, 0.22 * ballPx ) );
/** The rules of the first version: every name at its largest size, whatever the size of its community. */
const fullSize = ( sc ) => {

	sc.L.uniforms.clusterNameMin.value = 1e3;
	return sc;

};

test( 'a name sits on its community: from afar only the largest communities are named, a dot gets none', () => {

	// ball radius of a community: 1.29 × rms = 1.29 × 80 / √2 = 73 world units; 330 apart in a row
	const sizes = [ 30, 26, 22, 18, 14, 10 ];
	const { L, app, moveCamera } = fakeScene( {}, { sizes, gap: 330, spread: 80 } );
	const ball = 1.29 * 80 / Math.SQRT2;

	// near enough: every name is narrower than its community
	moveCamera( 1700 );
	L._decide( app, 0, - 1, - 1 );
	let cl = L._cl;
	for ( let j = 0; j < cl.K; j ++ ) assert.ok( nameFits( nameWidth( L, j ), ball * pxPerUnit( 1700 ) ), `name ${j} fits` );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1, 1, 1, 1 ] );
	assert.equal( cl.loose, 0 );
	for ( let j = 0; j < cl.K; j ++ ) close( cl.radius[ cl.slots[ j ].k ], ball * pxPerUnit( 1700 ), 2.5, 'projected radius' );

	// a name is sized by its community (0.22 projected radii, at least 11 px) and sits above its ball,
	// not on the centroid, where the hub and its label are
	for ( let j = 0; j < cl.K; j ++ ) {

		const slot = cl.slots[ j ], box = L.clusterBox( slot.cluster ), r = ball * pxPerUnit( 1700 );
		close( box.font, Math.max( 11, 0.22 * r ), 0.3, 'font size' );
		assert.ok( box.font < L.fontSize * L.uniforms.clusterScale.value, 'smaller than a name on a large community' );
		close( ( box.x0 + box.x1 ) / 2, cl.sx[ slot.k ], 1e-3, 'centred over the centroid' );
		assert.ok( box.y1 <= cl.sy[ slot.k ] - r + 0.5 * box.font, `the text is above the ball: ${box.y1} vs ${cl.sy[ slot.k ] - r}` );
		assert.ok( box.y0 > cl.sy[ slot.k ] - r - 3 * box.font, 'and close to it' );

	}

	assert.equal( L.clusterBox( 99 ), null );
	// from here on a name is 11 px: it still fits a ball that the full-size name is far too wide for
	moveCamera( 2600 );
	L._decide( app, 1, - 1, - 1 );
	for ( let j = 0; j < cl.K; j ++ ) assert.equal( nameFits( nameWidth( L, j ), ball * pxPerUnit( 2600 ), 1.1 ), false );
	for ( let j = 0; j < cl.K; j ++ ) assert.equal( nameFits( shownWidth( L, j, ball * pxPerUnit( 2600 ) ), ball * pxPerUnit( 2600 ) ), true );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1, 1, 1, 1 ] );
	assert.equal( L._cl.loose, 0 );

	// farther: no name fits its community any more (not even with the hysteresis of a shown name), the
	// names would still have room next to each other: the largest few keep theirs
	moveCamera( 4000 );
	L._decide( app, 1, - 1, - 1 );
	cl = L._cl;
	for ( let j = 0; j < cl.K; j ++ ) assert.equal( nameFits( shownWidth( L, j, ball * pxPerUnit( 4000 ) ), ball * pxPerUnit( 4000 ), 1.1 ), false );
	close( cl.looseRoom, looseNameArea( cl.graphPx ), 1e-6 );
	assert.equal( L.options.maxLooseNames, 4 );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1, 1, 0, 0 ], 'the largest communities, in order of size' );
	assert.equal( cl.loose, 4 );
	assert.equal( L.stats.clusters, 4 );

	// a community that shrinks to a dot loses its name gradually (clusterMinRadius = [ 10, 18 ] px) …
	moveCamera( 6000 );
	L._decide( app, 2, - 1, - 1 );
	assert.ok( ball * pxPerUnit( 6000 ) > 10 && ball * pxPerUnit( 6000 ) < 18 );
	assert.ok( nameAlpha( L ).slice( 0, 4 ).every( ( a ) => a > 0 && a < 1 ), `fading: ${nameAlpha( L )}` );
	moveCamera( 12000 );
	L._decide( app, 2, - 1, - 1 );
	assert.ok( nameAlpha( L ).every( ( a ) => a === 0 ), 'dots have no names' );
	// … and from so far that the graph is the size of a name there are none either
	moveCamera( 40000 );
	L._decide( app, 2, - 1, - 1 );
	assert.ok( L._cl.graphPx < 60 );
	assert.ok( nameAlpha( L ).every( ( a ) => a === 0 ) );
	assert.equal( L.stats.clusters, 0 );

	// 'on' means every name that finds room, also for dots; clusterFit 0 switches the fit rule off
	L.clusterLabels = 'on';
	moveCamera( 4000 );
	L._decide( app, 3, - 1, - 1 );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1, 1, 1, 1 ] );
	const free = fakeScene( { clusterFit: 0 }, { sizes, gap: 330, spread: 80 } );
	free.moveCamera( 4000 );
	free.L._decide( free.app, 0, - 1, - 1 );
	assert.deepEqual( nameAlpha( free.L ), [ 1, 1, 1, 1, 1, 1 ] );

	// a small projection: the names that do not fit share a part of the graph's disc. Three
	// communities seen from afar have room for one name; the third community, whose name would not
	// touch the first one's, gets none because of that (with the rule off it does)
	// (shown with names at their full size and without the fade of small communities: with the
	// defaults a view like this has no names at all)
	const small = fullSize( fakeScene( { clusterMinRadius: [ 0, 0 ] }, { sizes: [ 30, 26, 22 ], gap: 330, spread: 80 } ) );
	small.moveCamera( 6000 );
	small.L._decide( small.app, 0, - 1, - 1 );
	const F = small.L.fontSize * small.L.uniforms.clusterScale.value * 1.4, box = nameWidth( small.L, 0 ) * 2 * 0.66 * F;
	assert.ok( small.L._cl.looseRoom >= box && small.L._cl.looseRoom < 2 * box * 0.95, `room for one: ${small.L._cl.looseRoom} / ${box}` );
	assert.deepEqual( nameAlpha( small.L ), [ 1, 0, 0 ] );
	const off = fullSize( fakeScene( { clusterFit: 0, clusterMinRadius: [ 0, 0 ] }, { sizes: [ 30, 26, 22 ], gap: 330, spread: 80 } ) );
	off.moveCamera( 6000 );
	off.L._decide( off.app, 0, - 1, - 1 );
	assert.deepEqual( nameAlpha( off.L ), [ 1, 0, 1 ], 'the middle one collides, the third does not' );

} );

test( 'names: a view carries a limited number of them, and they keep air between each other', () => {

	// six communities in a row, all large enough for their name
	const sizes = [ 34, 30, 26, 22, 18, 14 ];
	const { L, app, moveCamera } = fakeScene( { maxNamesInView: 3 }, { sizes, gap: 330, spread: 80 } );
	moveCamera( 1700 );
	L._decide( app, 0, - 1, - 1 );
	assert.deepEqual( nameAlpha( L ), [ 1, 1, 1, 0, 0, 0 ], 'the three largest' );
	assert.equal( L.stats.clusters, 3 );
	assert.equal( L._cl.loose, 0 );
	// the default follows the size of the view
	const auto = fakeScene( {}, { sizes, gap: 330, spread: 80 } );
	auto.moveCamera( 1700 );
	auto.L._decide( auto.app, 0, - 1, - 1 );
	assert.equal( nameBudget( 1600, 1000 ), 20 );
	assert.deepEqual( nameAlpha( auto.L ), [ 1, 1, 1, 1, 1, 1 ] );

	// air: two names whose boxes are 4 px apart are not both placed, without the margin they are
	const pair = ( options ) => {

		const sc = fullSize( fakeScene( { clusterFit: 0, clusterMinRadius: [ 0, 0 ], ...options }, { sizes: [ 30, 28 ], gap: 100, spread: 20 } ) );
		const slots = sc.L._ensureClusters().slots, fontPx = sc.L.fontSize * sc.L.uniforms.clusterScale.value;
		const half = ( slots[ 0 ].widthEm * slots[ 0 ].scale + slots[ 1 ].widthEm * slots[ 1 ].scale ) * fontPx / 2 + 6; // two half boxes
		sc.moveCamera( ( 1 / Math.tan( Math.PI / 8 ) ) * 500 * 100 / ( half + 4 ) );
		sc.L._decide( sc.app, 0, - 1, - 1 );
		return nameAlpha( sc.L );

	};

	assert.deepEqual( pair( {} ), [ 1, 0 ], 'half a font size of air is needed' );
	assert.deepEqual( pair( { clusterSpacing: 0, hysteresis: 0 } ), [ 1, 1 ], 'without the margin both fit' );

} );

test( 'names on a narrow view: the font shrinks so that no name is wider than a share of it', () => {

	// the names of the test font are short: let a name take 15 % of the view at most
	const { L, app, moveCamera } = fakeScene( { clusterMaxWidth: 0.15 }, { sizes: [ 30, 26, 22 ], gap: 330, spread: 80 } );
	moveCamera( 1700 );
	L._decide( app, 0, - 1, - 1 );
	const cl = L._cl;
	for ( let j = 0; j < cl.K; j ++ ) assert.equal( cl.slots[ j ].shown, cl.slots[ j ].scale, 'a desktop view: untouched' );
	// a phone in portrait
	app.width = 412; app.height = 915;
	L._decide( app, 1, - 1, - 1 );
	for ( let j = 0; j < cl.K; j ++ ) {

		const slot = cl.slots[ j ];
		assert.ok( slot.shown < slot.scale && slot.shown >= 0.85, `${slot.shown} of ${slot.scale}` );
		assert.ok( nameWidth( L, j ) <= 0.15 * 412 + 1e-6 || slot.shown === 0.85 );
		assert.equal( cl.S[ 8 * j + 3 ], Math.fround( slot.shown ), 'the record the shader reads' );

	}

	assert.equal( cl.stateDirty, true );
	// back on the desktop
	app.width = 1600; app.height = 1000;
	L._decide( app, 2, - 1, - 1 );
	for ( let j = 0; j < cl.K; j ++ ) assert.equal( cl.S[ 8 * j + 3 ], Math.fround( cl.slots[ j ].scale ) );

} );

test( 'big graphs: the zoom names the larger nodes only (a budget of its own), a selection is not counted', () => {

	const { L, app, moveCamera, buffers } = fakeScene( { autoMaxNodes: 30, maxZoomLabels: 6 }, { sizes: [ 60, 60, 60 ], gap: 400, spread: 60 } );
	moveCamera( 300 ); // the middle community fills the view
	L._decide( app, 0, - 1, - 1 );
	assert.equal( L._cl.fade[ 1 ], 1 );
	let nodes = shownNodes( L );
	assert.equal( nodes.length, 6, 'the budget, not the 60 members' );
	assert.ok( nodes.includes( 60 ), 'in degree order: the hub first' );
	assert.equal( L.stats.zoomBudget, 6 );
	assert.equal( L.stats.labels, 6 );

	// the selection and its neighbours come on top of the budget
	buffers.uniforms.selectedIndex.value = 60;
	L.maxNeighbors = 8;
	L._decide( app, 1, 60, - 1 );
	nodes = shownNodes( L );
	const context = nodes.filter( ( i ) => L._L[ 4 * i + 3 ] === CTX );
	assert.ok( nodes.includes( 60 ) );
	assert.ok( context.length <= 6 );
	assert.ok( nodes.length > context.length + 1, 'selected + neighbours + context' );
	buffers.uniforms.selectedIndex.value = - 1;

	// default: by the size of the view (never more than maxLabels)
	L.maxZoomLabels = null;
	L._decide( app, 2, - 1, - 1 );
	assert.equal( L.stats.zoomBudget, zoomLabelBudget( 1600, 1000 ) );
	L.maxLabels = 4;
	L._decide( app, 3, - 1, - 1 );
	assert.equal( L.stats.zoomBudget, 4 );
	assert.ok( shownNodes( L ).length <= 4 );

	// graphs within autoMaxNodes are not limited by it (mode 'auto' shows what fits, as before)
	const small = fakeScene( { maxZoomLabels: 3 }, { sizes: [ 20, 20, 20 ] } );
	small.moveCamera( 150 );
	small.L._decide( small.app, 0, - 1, - 1 );
	assert.equal( small.L.stats.zoomBudget, 0 );
	assert.ok( shownNodes( small.L ).length > 3 );

} );

test( 'big graphs without node names: the zoom brings no labels (an index says nothing), the selection keeps its own', () => {

	const { L, app, moveCamera, buffers, graph } = fakeScene( { autoMaxNodes: 30 }, { sizes: [ 60, 60, 60 ], gap: 400, spread: 60 } );
	const bare = { ...graph, labels: null };
	L.setGraph( buffers, bare, app );
	assert.equal( L.text( 7 ), '7' );
	assert.equal( L.clusterText( 1 ), '60 nodes' );
	moveCamera( 300 );
	L._decide( app, 0, - 1, - 1 );
	assert.equal( L._cl.fade[ 1 ], 1, 'the community is resolved' );
	assert.equal( shownNodes( L ).length, 0, 'no wall of indices' );
	assert.equal( L.stats.labels, 0 );
	// far: the community names (counts) still show
	moveCamera( 3000 );
	L._decide( app, 1, - 1, - 1 );
	assert.ok( nameAlpha( L ).some( ( a ) => a === 1 ) );
	// a selection shows its label and those of its neighbours, as it always did
	moveCamera( 300 );
	buffers.uniforms.selectedIndex.value = 60;
	L._decide( app, 2, 60, - 1 );
	const nodes = shownNodes( L );
	assert.ok( nodes.includes( 60 ) && nodes.length > 1 );
	assert.ok( nodes.every( ( i ) => L._L[ 4 * i + 3 ] !== CTX ) );
	// mode 'all' is the user's explicit wish: indices and all
	buffers.uniforms.selectedIndex.value = - 1;
	L.mode = 'all';
	L._decide( app, 3, - 1, - 1 );
	assert.ok( shownNodes( L ).length > 5 );
	// empty label arrays count as no names
	const empty = fakeScene( { autoMaxNodes: 30 }, { sizes: [ 60, 60, 60 ], gap: 400, spread: 60 } );
	empty.L.setGraph( empty.buffers, { ...empty.graph, labels: new Array( 180 ).fill( '' ) }, empty.app );
	empty.moveCamera( 300 );
	empty.L._decide( empty.app, 0, - 1, - 1 );
	assert.equal( shownNodes( empty.L ).length, 0 );

} );

test( 'clusterLevel auto: the names sit on the level of the community field and follow it', () => {

	const graph = lesmis();
	assert.ok( graph.D >= 2, 'lesmis has at least two hierarchy levels' );
	const camera = new PerspectiveCamera( 45, 1.6, 0.5, 1e5 );
	camera.position.set( 0, 0, graph.layoutRadius * 6 );
	camera.updateMatrixWorld();
	const u = ( v ) => ( { value: v } );
	const buffers = { uniforms: { selectedIndex: u( - 1 ), hoverIndex: u( - 1 ), focus: u( 0 ) }, nodePos: { value: { array: graph.initPos } } };
	const field = { isClusterField: true, graph, height: 2 };
	const app = { width: 1600, height: 1000, camera, positions: graph.initPos, positionsVersion: 1, modules: [ field ] };
	const L = new TextLabels( { source: fakeSource() } );
	L.init( app );
	L.setGraph( buffers, graph, app );
	L.update( 0.016, app );
	assert.equal( L._cl.height, 2, 'the level of the field' );
	const offset2 = graph.levelOffset[ 1 ];
	assert.ok( L._cl.slots.every( ( s ) => s.cluster >= offset2 ) );

	// the field goes to the finest level: the names are rebuilt on it in the next update
	field.height = 1;
	L.update( 0.016, app );
	assert.equal( L._cl.height, 1 );
	assert.ok( L._cl.slots.every( ( s ) => s.cluster < offset2 ) );
	assert.notEqual( L.clusterText( L._cl.slots[ 0 ].cluster ), null );

	// an explicit level is kept whatever the field does
	L.clusterLevel = 2;
	L.update( 0.016, app );
	assert.equal( L._cl.height, 2 );
	field.height = 1;
	L.update( 0.016, app );
	assert.equal( L._cl.height, 2 );

	// a field of another graph, or without a level, is not followed: the names choose themselves
	L.clusterLevel = 'auto';
	field.height = 0;
	L.update( 0.016, app );
	const own = L._cl.height;
	assert.ok( own >= 1 );
	field.height = own === 1 ? 2 : 1;
	field.graph = {};
	L.update( 0.016, app );
	assert.equal( L._cl.height, own );
	L.dispose();

} );
