// Shell UI: the panel layout against the settings schema, and the pure helpers of the status
// chips, photo mode and the help overlay.
import test from 'node:test';
import assert from 'node:assert/strict';
import { PANEL_LAYOUT, STRUCTURE_PRESETS, layoutKeys, structurePreset, DEFAULT_SETTINGS, accentFor, paletteCss, lookText, Panel } from '../src/ui/Panel.js';
import { SETTINGS, SECTIONS, settingSpec, settingDefaults, describeClustering, deferredStorage, SettingsStore, encodeSettings, LOOKS, LOOK_KEYS, lookValues, lookOf, coerce } from '../src/ui/settings.js';
import { formatCount, layoutProgress, layoutState } from '../src/ui/Hud.js';
import { captureSize, captureBudget, scaleOffered, nearestOnScreen, photoFileName, PHOTO_SCALES, PHOTO_MAX_PIXELS, PHOTO_MAX_SIDE_TOUCH } from '../src/ui/PhotoMode.js';
import { TIERS } from '../src/app/Quality.js';
import { SHELL_SHORTCUTS, TOUCH_SHORTCUTS, groupShortcuts, keyList, withoutRestated, isTypingTarget } from '../src/ui/Help.js';
import { LAYOUT_PRESETS } from '../src/gpu/ForceLayout.js';

test( 'panel layout: every control is a setting of its section, and every setting has a control', () => {

	const keys = layoutKeys( PANEL_LAYOUT );
	const sections = new Set( SECTIONS.map( ( s ) => s.id ) );
	const seen = new Map();
	for ( const [ section, list ] of Object.entries( keys ) ) {

		assert.ok( sections.has( section ), `unknown section ${section}` );
		for ( const key of list ) {

			const spec = settingSpec( key );
			assert.ok( spec, `panel shows "${key}", which the schema does not have` );
			assert.equal( spec.section, section, `"${key}" sits in ${section} but the schema says ${spec.section}` );
			assert.ok( ! seen.has( key ), `"${key}" appears twice` );
			seen.set( key, section );

		}

	}

	for ( const spec of SETTINGS ) {

		if ( spec.section === null ) continue; // shell state without a panel control (none today)
		assert.ok( seen.has( spec.key ), `setting "${spec.key}" has no control in the panel` );

	}

	// the order of the sections is the order of the schema
	assert.deepEqual( PANEL_LAYOUT.map( ( s ) => s.id ), SECTIONS.map( ( s ) => s.id ) );

} );

test( 'panel layout: only the first two sections start open, advanced settings sit in groups', () => {

	assert.deepEqual( PANEL_LAYOUT.filter( ( s ) => s.open ).map( ( s ) => s.id ), [ 'dataset', 'layout' ] );
	const grouped = new Set();
	for ( const sec of PANEL_LAYOUT ) for ( const it of sec.items ) if ( it && it.type === 'group' ) for ( const k of it.items ) grouped.add( typeof k === 'string' ? k : k.key );
	for ( const spec of SETTINGS ) {

		if ( spec.section === null ) continue;
		assert.equal( grouped.has( spec.key ), !! spec.advanced, `"${spec.key}": advanced in the schema ⇔ inside an Advanced group` );

	}

} );

test( 'panel: summaries and notes never throw, with or without live info', () => {

	const s = settingDefaults();
	for ( const sec of PANEL_LAYOUT ) {

		if ( sec.summary ) {

			assert.equal( typeof sec.summary( s, { features: {} } ), 'string' );
			assert.equal( typeof sec.summary( s, { features: { env: true, edgeStyle: true }, clustering: { count: 12 }, quality: { tier: 'high' }, edgeStyle: 'solid' } ), 'string' );

		}

		for ( const it of sec.items ) {

			if ( it && it.type === 'note' ) assert.equal( typeof it.text( s, {} ), 'string' );
			if ( it && it.type === 'choice' ) {

				assert.ok( it.describe( 'radial', s, {} ).text.includes( 'Select a node' ) );
				assert.ok( ! it.describe( 'radial', s, { selection: true } ).text.includes( 'Select a node' ) );
				assert.equal( it.describe( 'stress', s, { layout: { pending: true } } ).busy, true );

			}

		}

	}

	assert.equal( PANEL_LAYOUT.find( ( x ) => x.id === 'quality' ).summary( { ...s, quality: 'auto' }, { quality: { tier: 'high' } } ), 'Auto · High' );
	assert.equal( PANEL_LAYOUT.find( ( x ) => x.id === 'layout' ).summary( { ...s, running: false }, {} ), 'Force 3D · paused' );
	assert.match( PANEL_LAYOUT.find( ( x ) => x.id === 'nodes' ).summary( s, {} ), / palette$/, 'the palette is named as one: looks and themes share its names' );

} );

test( 'panel: a change made in the panel is written by the store and reaches its listeners', () => {

	// the plumbing of Panel._change on a stand-in panel (the real one needs a document)
	const store = new SettingsStore();
	const heard = [], applied = [];
	store.onChange( ( key, value, by ) => heard.push( [ key, value, by ] ) );
	const panel = {
		settings: store.values,
		_write: ( key, value ) => store.set( key, value ),
		onChange: ( key, value ) => applied.push( [ key, value ] ),
		setAccent() {}, _sync() {}, _pending: new Map(),
	};
	Panel.prototype._change.call( panel, 'nodeSize', 2 );
	assert.deepEqual( heard, [ [ 'nodeSize', 2, 'user' ] ], 'the store saw the change before the value was there' );
	assert.equal( store.isTouched( 'nodeSize' ), true );
	// the store has the last word on the value: out of range is clamped, and the panel passes on what the store holds
	Panel.prototype._change.call( panel, 'nodeSize', 99 );
	assert.deepEqual( applied, [ [ 'nodeSize', 2 ], [ 'nodeSize', 4 ] ] );
	assert.equal( store.values.nodeSize, 4 );
	// showing a value that changed elsewhere never writes it
	Panel.prototype.set.call( panel, 'nodeSize', 1 );
	assert.equal( store.values.nodeSize, 4 );
	assert.equal( heard.length, 2 );

	// a panel over a plain object writes it itself
	const plain = { settings: { nodeSize: 1 }, _write: null, onChange: null, setAccent() {}, _sync() {}, _pending: new Map() };
	Panel.prototype._change.call( plain, 'nodeSize', 3 );
	assert.equal( plain.settings.nodeSize, 3 );
	Panel.prototype.set.call( plain, 'nodeSize', 1.5 );
	assert.equal( plain.settings.nodeSize, 1.5 );

} );

test( 'panel: the layout tiles carry one word each, the look says what it is', () => {

	const choice = PANEL_LAYOUT.find( ( s ) => s.id === 'layout' ).items.find( ( it ) => it.type === 'choice' );
	for ( const o of settingSpec( 'layoutMode' ).options ) {

		const label = choice.labels?.[ o.value ] || o.label;
		assert.ok( ! /\s/.test( label ) && label.length <= 7, `tile "${label}"` );

	}

	for ( const look of LOOKS ) assert.equal( lookText( look.value ), look.hint );
	assert.match( lookText( 'custom' ), /own mix/ );
	// no two controls of the panel share a name inside the same or a neighbouring concept
	const labels = SETTINGS.filter( ( x ) => x.section === 'communities' || x.section === 'nodes' ).map( ( x ) => x.label );
	assert.equal( new Set( labels ).size, labels.length, `distinct labels: ${labels.join( ', ' )}` );

} );

test( 'structure presets mirror the layout presets', () => {

	for ( const p of STRUCTURE_PRESETS ) {

		assert.deepEqual( { clusterSeparation: p.clusterSeparation, cohesion: p.cohesion }, LAYOUT_PRESETS[ p.value ] );
		assert.equal( structurePreset( p ), p.value );

	}

	assert.equal( structurePreset( DEFAULT_SETTINGS ), 'default' );
	assert.equal( structurePreset( { clusterSeparation: 0.55, cohesion: 0.5 } ), 'custom' );

} );

test( 'accent and palette colours are valid CSS', () => {

	for ( const name of settingSpec( 'palette' ).options.map( ( o ) => o.value ) ) {

		const rgb = accentFor( name );
		assert.equal( rgb.length, 3 );
		for ( const c of rgb ) assert.ok( Number.isInteger( c ) && c >= 0 && c <= 255 );
		assert.match( paletteCss( name, 0.5 ), /^rgb\(\d+ \d+ \d+\)$/ );

	}

} );

test( 'quality is saved but does not travel in a share link', () => {

	const store = new SettingsStore();
	store.set( 'quality', 'ultra' );
	store.set( 'dynamicResolution', false );
	store.set( 'bloom', 1.5 );
	assert.deepEqual( store.shared(), { bloom: 1.5 } );
	assert.equal( encodeSettings( store.shared() ), '2~bl:1.5' );
	assert.notEqual( settingSpec( 'quality' ).save, false );

} );

test( 'describeClustering: method, communities, levels, resolution, modularity', () => {

	assert.equal( describeClustering( null ), '' );
	assert.equal( describeClustering( { stats: { clusters: 18 }, D: 2, meta: { hierarchy: { method: 'Louvain (python-louvain 0.8, preprocessing)' } } } ), 'Louvain · 18 communities · 2 levels' );
	assert.equal( describeClustering( { stats: { clusters: 1 }, D: 1, clustering: { method: 'leiden', resolution: 2, levels: 1, modularity: 0.4321 } } ), 'Leiden · 1 community · resolution 2 · modularity 0.43' );
	assert.equal( describeClustering( { stats: { clusters: 1200 }, D: 3, meta: { hierarchy: { method: 'planted' } } } ), 'Planted communities · 1,200 communities · 3 levels' );
	assert.equal( describeClustering( { stats: {}, D: 0 } ), '' );

} );

test( 'deferredStorage writes once after a pause, reads through, flushes on demand', async () => {

	const writes = [];
	const backing = new Map( [ [ 'a', 'old' ] ] );
	const raw = { getItem: ( k ) => ( backing.has( k ) ? backing.get( k ) : null ), setItem: ( k, v ) => ( writes.push( [ k, v ] ), backing.set( k, v ) ), removeItem: ( k ) => backing.delete( k ) };
	const s = deferredStorage( raw, 15 );
	assert.equal( deferredStorage( null ), null );
	assert.equal( s.getItem( 'a' ), 'old' );
	for ( let i = 0; i < 20; i ++ ) s.setItem( 'a', 'v' + i );
	assert.equal( writes.length, 0, 'nothing is written while values keep coming' );
	assert.equal( s.getItem( 'a' ), 'v19', 'reads see the pending value' );
	await new Promise( ( r ) => setTimeout( r, 60 ) );
	assert.deepEqual( writes, [ [ 'a', 'v19' ] ] );
	s.setItem( 'b', '1' );
	s.flush();
	assert.deepEqual( writes[ 1 ], [ 'b', '1' ] );
	// a store on top of it persists like on a plain storage
	const store = new SettingsStore( { storage: s } );
	store.set( 'bloom', 1.25 );
	s.flush();
	const again = new SettingsStore( { storage: raw } );
	again.load();
	assert.equal( again.values.bloom, 1.25 );
	// a storage that throws must not break the session
	const broken = deferredStorage( { getItem: () => null, setItem: () => {

		throw new Error( 'quota' );

	}, removeItem: () => {} }, 1 );
	broken.setItem( 'x', '1' );
	assert.doesNotThrow( () => broken.flush() );

} );

// ── status chips ────────────────────────────────────────────────────────────────────────────

test( 'formatCount keeps counts short', () => {

	assert.equal( formatCount( 77 ), '77' );
	assert.equal( formatCount( 2492 ), '2,492' );
	assert.equal( formatCount( 12345 ), '12.3k' );
	assert.equal( formatCount( 250000 ), '250k' );
	assert.equal( formatCount( 1234567 ), '1.23M' );
	assert.equal( formatCount( NaN ), '–' );

} );

test( 'layoutProgress rises monotonically from the plateau to the end of the cooling', () => {

	assert.equal( layoutProgress( 1, 0.001, 0.3, 0 ), 0 );
	assert.ok( Math.abs( layoutProgress( 1, 0.001, 0.3, 1 ) - 0.3 ) < 1e-12 );
	let last = 0.3;
	for ( let a = 0.999; a > 0.001; a *= 0.9 ) {

		const p = layoutProgress( a );
		assert.ok( p >= last - 1e-12 && p <= 1, `alpha ${a}` );
		last = p;

	}

	assert.ok( Math.abs( layoutProgress( 0.001 ) - 1 ) < 1e-9 );
	assert.equal( layoutProgress( 0 ), 1 );
	assert.equal( layoutProgress( 1e-9 ), 1 );
	// half of the cooling in the logarithm of alpha is half of the remaining bar
	assert.ok( Math.abs( layoutProgress( Math.sqrt( 0.001 ) ) - 0.65 ) < 1e-9 );

} );

test( 'layoutState names what the layout is doing', () => {

	assert.equal( layoutState( null ), 'settled' );
	assert.equal( layoutState( { pending: true, paused: true } ), 'computing' );
	assert.equal( layoutState( { paused: true, settled: false } ), 'paused' );
	assert.equal( layoutState( { settled: true, layoutMode: 'force' } ), 'settled' );
	assert.equal( layoutState( { settled: false, layoutMode: 'flat' } ), 'running' );
	assert.equal( layoutState( { settled: false, layoutMode: 'sphere' } ), 'moving' );

} );

// ── photo mode ──────────────────────────────────────────────────────────────────────────────

test( 'captureSize: scale × pixel ratio, limited by the pixel budget and the device', () => {

	assert.deepEqual( PHOTO_SCALES, [ 1, 2, 4 ] );
	const a = captureSize( 1280, 800, 1, 2 );
	assert.deepEqual( [ a.width, a.height, a.ratio, a.limited ], [ 2560, 1600, 2, false ] );
	const b = captureSize( 1280, 800, 1, 4 );
	assert.ok( b.limited && b.width * b.height <= PHOTO_MAX_PIXELS && b.width * b.height > 0.98 * PHOTO_MAX_PIXELS, 'the 4× of a large window uses the budget, not more' );
	assert.ok( Math.abs( b.width / b.height - 1.6 ) < 0.01, 'aspect kept' );
	assert.equal( Math.max( b.width, b.height ), Math.floor( 1280 * b.ratio ) );
	// a phone at DPR 2.625 in portrait
	const c = captureSize( 412, 915, 2.625, 4 );
	assert.ok( c.height <= 8192 && c.width * c.height <= PHOTO_MAX_PIXELS );
	assert.ok( c.height > c.width );
	// the device limit wins over the budget
	const d = captureSize( 4000, 400, 1, 4, { deviceLimit: 4096 } );
	assert.equal( d.width, 4096 );
	// the same formula RenderGraph.capture uses: ratio = min( pr × scale, maxSize / longer side )
	for ( const [ w, h, pr, s ] of [ [ 1280, 800, 1, 1 ], [ 1280, 800, 2, 2 ], [ 1920, 1080, 1.5, 4 ], [ 390, 844, 3, 2 ] ] ) {

		const r = captureSize( w, h, pr, s );
		assert.ok( Math.abs( r.ratio - Math.min( pr * s, r.maxSize / Math.max( w, h ) ) ) < 1e-12 );
		assert.deepEqual( [ r.width, r.height ], [ Math.floor( w * r.ratio ), Math.floor( h * r.ratio ) ] );

	}

} );

test( 'captureBudget: four times the pixel cap of the tier, 16 MP at most, 4096 px on touch devices', () => {

	assert.deepEqual( captureBudget( TIERS.low ), { maxPixels: 6e6, maxSide: Infinity } );
	assert.equal( captureBudget( TIERS.medium ).maxPixels, 10e6 );
	assert.equal( captureBudget( TIERS.high ).maxPixels, PHOTO_MAX_PIXELS );
	assert.equal( captureBudget( TIERS.ultra ).maxPixels, PHOTO_MAX_PIXELS );
	assert.equal( captureBudget( null ).maxPixels, PHOTO_MAX_PIXELS );
	assert.deepEqual( captureBudget( TIERS.low, { coarse: true } ), { maxPixels: 6e6, maxSide: PHOTO_MAX_SIDE_TOUCH } );
	assert.equal( PHOTO_MAX_SIDE_TOUCH, 4096 );

	// a phone ( 412 × 915 at DPR 2.625 ) in the low tier: nothing above 6 MP or 4096 px
	const phone = captureBudget( TIERS.low, { coarse: true } );
	for ( const scale of PHOTO_SCALES ) {

		const s = captureSize( 412, 915, 2.625, scale, phone );
		assert.ok( s.width * s.height <= 6e6 && Math.max( s.width, s.height ) <= 4096, `${scale}×: ${s.width} × ${s.height}` );

	}

	// a strong phone in the high tier: the side limit holds where the pixel budget would not
	const strong = captureSize( 412, 915, 2.625, 4, captureBudget( TIERS.high, { coarse: true } ) );
	assert.equal( strong.height, 4096 );
	// a desktop in the low tier: 4× of a 1280 × 800 window comes out as 6 MP
	const low = captureSize( 1280, 800, 1, 4, captureBudget( TIERS.low ) );
	assert.ok( low.limited && low.width * low.height <= 6e6 && low.width * low.height > 0.98 * 6e6 );

} );

test( 'scaleOffered: a size that cannot come out larger than the one below it is not a choice', () => {

	const offered = ( w, h, pr, limits ) => PHOTO_SCALES.map( ( s ) => scaleOffered( s, w, h, pr, limits ) );
	// desktop, high tier: everything (the 4× is limited to the budget but well above the 2×)
	assert.deepEqual( offered( 1280, 800, 1, captureBudget( TIERS.high ) ), [ true, true, true ] );
	assert.deepEqual( offered( 1280, 800, 1, captureBudget( TIERS.low ) ), [ true, true, true ] );
	// the phone in the low tier: 2× is limited ( 3.99 instead of 5.25 ) and 4× would be the same picture
	assert.deepEqual( offered( 412, 915, 2.625, captureBudget( TIERS.low, { coarse: true } ) ), [ true, true, false ] );
	// a large window at DPR 2 in the high tier: its own resolution is half the budget already
	assert.deepEqual( offered( 1920, 1080, 2, captureBudget( TIERS.high ) ), [ true, true, false ] );
	// phone emulation as the app sees it ( 412 × 839, pixel ratio capped at 2 ): the 4× would be 4 % larger than the 2×
	assert.deepEqual( offered( 412, 839, 2, captureBudget( TIERS.low, { coarse: true } ) ), [ true, true, false ] );
	// a 4K window at DPR 2: only its own size
	assert.deepEqual( offered( 3840, 2160, 2, captureBudget( TIERS.ultra ) ), [ true, false, false ] );
	assert.equal( scaleOffered( 3, 1280, 800 ), false, 'not one of the sizes' );

} );

test( 'nearestOnScreen picks the node nearest to the click, in front of the camera only', () => {

	// identity view, a projection with w = −z (a camera at the origin looking down −z)
	const m = [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, - 1, - 1, 0, 0, 0, 0 ];
	const positions = new Float32Array( [
		0, 0, - 10, 0, // centre of the screen
		5, 0, - 10, 0, // right of it: x/w = 0.5 → 3/4 of the width
		0, 5, - 10, 0, // above
		0, 0, 10, 0, // behind the camera: never picked
	] );
	const at = ( x, y ) => nearestOnScreen( positions, m, x, y, 800, 600 );
	assert.equal( at( 400, 300 ).index, 0 );
	const right = at( 590, 310 );
	assert.equal( right.index, 1 );
	assert.ok( Math.abs( right.x - 600 ) < 1e-6 && Math.abs( right.y - 300 ) < 1e-6 );
	assert.ok( Math.abs( right.distance - Math.hypot( 10, 10 ) ) < 1e-6 );
	assert.equal( at( 400, 100 ).index, 2 );
	assert.equal( nearestOnScreen( positions.subarray( 12 ), m, 400, 300, 800, 600 ).index, - 1 );
	assert.equal( nearestOnScreen( positions, m, 400, 300, 800, 600, 0 ).index, - 1 );

} );

test( 'photoFileName: a safe file name from the dataset name and the size', () => {

	assert.equal( photoFileName( 'Les Misérables', 2560, 1600 ), 'les-miserables-2560x1600.png' );
	assert.equal( photoFileName( 'my graph.graphml', 10, 20 ), 'my-graph-10x20.png' );
	assert.equal( photoFileName( 'Communities · 500', 1, 1 ), 'communities-500-1x1.png' );
	assert.equal( photoFileName( '', 1, 1 ), 'graph-1x1.png' );
	assert.equal( photoFileName( '../../etc/passwd', 1, 1 ), 'etc-passwd-1x1.png' );
	assert.ok( photoFileName( 'x'.repeat( 300 ), 1, 1 ).length < 80 );

} );

// ── help ────────────────────────────────────────────────────────────────────────────────────

test( 'help: shell shortcuts are complete entries, grouped in order with the explore layer\'s', () => {

	for ( const s of SHELL_SHORTCUTS ) assert.ok( s.group && s.title && Array.isArray( s.keys ) && s.keys.length );
	for ( const key of [ 'P', 'H', '?', 'Esc', 'F', 'Space', 'R', 'L' ] ) assert.ok( SHELL_SHORTCUTS.some( ( s ) => s.keys.includes( key ) ), `${key} is listed` );
	const groups = groupShortcuts( SHELL_SHORTCUTS, [
		{ keys: '/', title: 'Search', group: 'Explore' },
		{ keys: [ 'I' ], title: 'Insights', group: 'Explore' },
		{ keys: 'Ctrl+K', title: 'Commands' }, // no group
		{ keys: 'G', title: 'Extra camera thing', group: 'Camera' },
		null, { keys: 'x' }, // broken entries are skipped
	] );
	assert.deepEqual( groups.map( ( g ) => g.title ), [ 'Camera', 'Graph', 'View', 'Explore', 'More' ] );
	assert.equal( groups[ 0 ].items.at( - 1 ).title, 'Extra camera thing' );
	assert.deepEqual( groups[ 3 ].items, [ { keys: [ '/' ], title: 'Search' }, { keys: [ 'I' ], title: 'Insights' } ] );
	assert.deepEqual( groupShortcuts( undefined, null ), [] );

} );

test( 'keyList: arrays as given, one cap for a combination, alternatives split', () => {

	assert.deepEqual( keyList( [ 'A', 'B' ] ), [ 'A', 'B' ] );
	assert.deepEqual( keyList( '/' ), [ '/' ] );
	assert.deepEqual( keyList( 'Ctrl+K' ), [ 'Ctrl+K' ] );
	assert.deepEqual( keyList( '[ / ]' ), [ '[', ']' ] );
	assert.deepEqual( keyList( 'F or Home' ), [ 'F', 'Home' ] );
	assert.deepEqual( keyList( null ), [] );
	// lists of single signs, as the explore layer writes them, are alternatives
	assert.deepEqual( keyList( '1 2 3' ), [ '1', '2', '3' ] );
	assert.deepEqual( keyList( '[ ]' ), [ '[', ']' ] );
	// … but a combination stays one cap
	assert.deepEqual( keyList( 'Ctrl K' ), [ 'Ctrl K' ] );
	assert.deepEqual( keyList( '⌘ K' ), [ '⌘ K' ] );
	assert.deepEqual( keyList( 'Shift Click' ), [ 'Shift Click' ] );
	assert.deepEqual( keyList( ' Esc ' ), [ 'Esc' ] );

} );

test( 'help: a shortcut the explore layer states is not listed twice, touch screens get their gestures', () => {

	const explore = [
		{ keys: 'Esc', title: 'Step back', group: 'Explore' },
		{ keys: 'space', title: 'Pause the layout', group: 'Explore' },
		{ keys: [ 'F', 'Home' ], title: 'Fit', group: 'Explore' },
		{ keys: 'F', title: 'Something else on F alone', group: 'Explore' },
		{ keys: 'Q' }, // no title: says nothing, takes nothing
	];
	const own = withoutRestated( SHELL_SHORTCUTS, explore );
	const titles = ( list, key ) => list.filter( ( s ) => keyList( s.keys ).includes( key ) ).map( ( s ) => s.title );
	assert.deepEqual( titles( own, 'Esc' ), [], 'Esc is the explore layer\'s line now' );
	assert.deepEqual( titles( own, 'Space' ), [], 'keys compare without case' );
	assert.deepEqual( titles( own, 'Home' ), [], 'the same alternatives are the same shortcut' );
	assert.deepEqual( titles( own, 'P' ), [ 'Photo mode' ] );
	assert.equal( own.length, SHELL_SHORTCUTS.length - 3 );
	assert.deepEqual( withoutRestated( SHELL_SHORTCUTS, null ), SHELL_SHORTCUTS );
	assert.deepEqual( withoutRestated( null, explore ), [] );

	// touch: complete entries of one group; the mouse gestures of the shell are marked so they can be left out
	for ( const s of TOUCH_SHORTCUTS ) assert.ok( s.group === 'Touch' && s.title && Array.isArray( s.keys ) && s.keys.length );
	const keyboard = SHELL_SHORTCUTS.filter( ( s ) => ! s.pointer );
	assert.ok( keyboard.length >= 8 );
	for ( const s of keyboard ) assert.ok( ! /drag|click|wheel|pinch|drop/i.test( s.keys.join( ' ' ) ), `${s.keys} is a keyboard shortcut` );
	for ( const s of SHELL_SHORTCUTS.filter( ( t ) => t.pointer ) ) assert.ok( /drag|click|wheel|pinch|drop/i.test( s.keys.join( ' ' ) ), `${s.keys} is a pointer gesture` );
	assert.deepEqual( groupShortcuts( TOUCH_SHORTCUTS, keyboard ).map( ( g ) => g.title ), [ 'Touch', 'Camera', 'Graph', 'View' ] );

} );

test( 'isTypingTarget: shortcuts stay quiet in text fields only', () => {

	const el = ( matches ) => ( { closest: ( sel ) => ( matches( sel ) ? {} : null ) } );
	assert.equal( isTypingTarget( el( () => true ) ), true );
	assert.equal( isTypingTarget( el( () => false ) ), false );
	assert.equal( isTypingTarget( null ), false );
	assert.equal( isTypingTarget( {} ), false );
	// the selector covers text inputs but not buttons, sliders or checkboxes
	let selector = '';
	isTypingTarget( el( ( s ) => ( selector = s, false ) ) );
	assert.ok( /textarea/.test( selector ) && /input\[type="search"\]/.test( selector ) && /contenteditable/.test( selector ) );
	assert.ok( ! /range|button|checkbox/.test( selector ) );

} );

test( 'looks: every look names known settings with valid values, and is recognised again', () => {

	assert.deepEqual( LOOKS.map( ( l ) => l.label ), [ 'Aurora', 'Paper', 'Blueprint' ] );
	const d = settingDefaults();
	for ( const key of LOOK_KEYS ) assert.ok( settingSpec( key ), key );
	for ( const look of LOOKS ) {

		const values = lookValues( look.value );
		assert.deepEqual( Object.keys( values ), LOOK_KEYS );
		const s = { ...d };
		for ( const [ key, v ] of Object.entries( look.values ) ) {

			assert.ok( LOOK_KEYS.includes( key ), `${look.value}: ${key} is a look key` );
			assert.equal( coerce( settingSpec( key ), v ), v, `${look.value}.${key} is a valid value` );
			if ( ! settingSpec( key ).auto ) assert.notEqual( v, d[ key ], `${look.value}.${key} differs from the default` );
			s[ key ] = v;

		}

		assert.equal( lookOf( s ), look.value );

	}

	assert.equal( lookValues( 'nope' ), null );
	assert.equal( lookOf( d ), 'aurora' ); // the defaults are a look
	assert.equal( lookOf( { ...d, edgeBlend: 'additive' } ), 'custom' ); // Aurora leaves the blend to the graph ('auto')
	assert.equal( lookOf( { ...d, bloom: 1.9 } ), 'custom' );
	assert.equal( lookOf( { ...d, ...LOOKS[ 1 ].values, edgeBlend: 'normal' } ), 'custom' ); // Paper names its blend

} );

test( 'looks: the panel offers them in the Scene section', () => {

	const scene = PANEL_LAYOUT.find( ( s ) => s.id === 'scene' );
	assert.equal( scene.items[ 0 ].type, 'look' );
	const labels = new Set( LOOKS.map( ( l ) => l.value ) );
	assert.equal( labels.size, LOOKS.length );

} );
