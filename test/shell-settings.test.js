// Settings model of the shell: schema, store (touched rule, persistence, migration, reset) and
// the share-link codec.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	SETTINGS, SECTIONS, SETTINGS_VERSION, STORAGE_KEY, SettingsStore, settingSpec, settingDefaults, isVisible, coerce, migrate,
	encodeShare, decodeShare, encodeSettings, decodeSettings, encodePose, decodePose, hasShare, parseLegacyParams, autoChoices,
	describeTier, AUTO,
} from '../src/ui/settings.js';
import { TIERS, resolveQuality } from '../src/app/Quality.js';
import { LAYOUT_MODES } from '../src/gpu/ForceLayout.js';
import { ENV_PRESETS } from '../src/render/gfx/Environment.js';
import { PALETTES } from '../src/util/color.js';

class MemoryStorage {

	constructor( initial = {} ) {

		this.map = new Map( Object.entries( initial ) );
		this.writes = 0;

	}

	getItem( k ) {

		return this.map.has( k ) ? this.map.get( k ) : null;

	}

	setItem( k, v ) {

		this.writes ++;
		this.map.set( k, String( v ) );

	}

	removeItem( k ) {

		this.map.delete( k );

	}

}

const stored = ( storage ) => JSON.parse( storage.getItem( STORAGE_KEY ) );

test( 'schema: unique keys and codes, valid defaults, known sections', () => {

	const keys = new Set(), codes = new Set(), sections = new Set( SECTIONS.map( ( s ) => s.id ) );
	for ( const s of SETTINGS ) {

		assert.ok( ! keys.has( s.key ), `duplicate key ${s.key}` );
		assert.ok( ! codes.has( s.code ), `duplicate code ${s.code}` );
		assert.match( s.code, /^[a-z]{1,2}$/, `code of ${s.key}` );
		keys.add( s.key ); codes.add( s.code );
		assert.ok( s.section === null || sections.has( s.section ), `section of ${s.key}` );
		assert.ok( typeof s.label === 'string' && s.label.length > 0, `label of ${s.key}` );
		assert.deepEqual( coerce( s, s.default ), s.default, `default of ${s.key} is valid` );
		if ( s.type === 'number' ) assert.ok( s.min < s.max && s.step > 0 && ( ! s.log || s.min > 0 ), `range of ${s.key}` );
		if ( s.type === 'enum' ) assert.ok( s.options.length >= 2 && s.options.every( ( o ) => o.value && o.label ), `options of ${s.key}` );

	}

	assert.equal( settingSpec( 'bloom' ).default, 0.8 );
	assert.equal( settingSpec( 'nope' ), null );
	assert.equal( settingDefaults().layoutMode, 'force' );

} );

test( 'schema: enums follow the modules they drive', () => {

	const values = ( key ) => settingSpec( key ).options.map( ( o ) => o.value );
	assert.deepEqual( values( 'layoutMode' ), LAYOUT_MODES.map( ( m ) => m.id ) );
	assert.deepEqual( values( 'theme' ), Object.keys( ENV_PRESETS ) );
	assert.deepEqual( values( 'palette' ), Object.keys( PALETTES ) );
	assert.deepEqual( values( 'quality' ), [ 'auto', ...Object.keys( TIERS ) ] );

} );

test( 'coerce: clamps numbers, parses booleans, rejects unknown enum values', () => {

	assert.equal( coerce( 'bloom', 5 ), 2 );
	assert.equal( coerce( 'bloom', '0.25' ), 0.25 );
	assert.equal( coerce( 'bloom', 'abc' ), undefined );
	assert.equal( coerce( 'bloom', '' ), undefined );
	assert.equal( coerce( 'autoRotate', '0' ), false );
	assert.equal( coerce( 'autoRotate', 'true' ), true );
	assert.equal( coerce( 'autoRotate', 'yes' ), undefined );
	assert.equal( coerce( 'layoutMode', 'sphere' ), 'sphere' );
	assert.equal( coerce( 'layoutMode', 'donut' ), undefined );
	assert.equal( coerce( 'unknown', 1 ), undefined );

} );

test( 'visibility: feature flags and rules', () => {

	const s = settingDefaults();
	assert.equal( isVisible( 'clusterMethod', s, {} ), false );
	assert.equal( isVisible( 'clusterMethod', s, { recluster: true } ), true );
	assert.equal( isVisible( 'clusterResolution', s, { recluster: true } ), false );
	assert.equal( isVisible( 'clusterResolution', { ...s, clusterMethod: 'leiden' }, { recluster: true } ), true );
	assert.equal( isVisible( 'immersive', s ), false );
	assert.equal( isVisible( 'immersive', { ...s, layoutMode: 'sphere' } ), true );
	assert.equal( isVisible( 'bundling', { ...s, bundleMode: 'straight' } ), false );
	assert.equal( isVisible( 'dofStrength', { ...s, dof: 'on' } ), true );
	assert.equal( isVisible( 'bloom', s ), true );

} );

test( 'touched: automatic choices never override an explicit one', () => {

	const store = new SettingsStore();
	assert.equal( settingSpec( 'fieldStrength' ).auto, true );
	// a graph without communities: no field
	assert.equal( store.auto( 'fieldStrength', 0 ), true );
	assert.equal( store.values.fieldStrength, 0 );
	assert.equal( store.isTouched( 'fieldStrength' ), false );
	// another graph, another automatic value
	assert.equal( store.auto( 'fieldStrength', 0.5 ), true );
	// the user decides
	store.set( 'fieldStrength', 0.8 );
	assert.equal( store.isTouched( 'fieldStrength' ), true );
	assert.equal( store.auto( 'fieldStrength', 0 ), false );
	assert.equal( store.values.fieldStrength, 0.8 );
	// choosing the default value explicitly is still a choice
	store.set( 'fieldStrength', 0.5 );
	assert.equal( store.auto( 'fieldStrength', 0 ), false );
	assert.equal( store.values.fieldStrength, 0.5 );
	// reset gives the key back to the automatic rule
	store.reset( 'fieldStrength' );
	assert.equal( store.auto( 'fieldStrength', 0 ), true );

} );

test( 'automatic settings can always be handed back: enums carry an explicit "auto"', () => {

	// a choice the app makes per graph must not become permanent by being overridden once:
	// enums offer 'auto' as a value (and it is the default), numbers go back by a reset
	for ( const key of [ 'edgeBlend', 'edgeStyle' ] ) {

		const spec = settingSpec( key );
		assert.equal( spec.default, 'auto', key );
		assert.ok( spec.options.some( ( o ) => o.value === 'auto' ), key );
		assert.ok( ! spec.auto, `${key}: the value itself says "automatic"` );

	}

	const storage = new MemoryStorage();
	const store = new SettingsStore( { storage } );
	store.set( 'edgeBlend', 'additive' ); // on a dense graph
	assert.deepEqual( stored( storage ).values, { edgeBlend: 'additive' } );
	store.set( 'edgeBlend', 'auto' ); // handed back from its own control
	assert.deepEqual( store.shared(), {}, 'the default again: nothing to share' );
	const next = new SettingsStore( { storage } );
	next.load();
	assert.equal( next.values.edgeBlend, 'auto' );
	// values saved before 'auto' existed stay valid choices
	assert.equal( migrate( { v: 2, values: { edgeBlend: 'normal' } } ).values.edgeBlend, 'normal' );
	assert.deepEqual( decodeSettings( '2~eb:additive' ).settings, { edgeBlend: 'additive' } );

} );

test( 'store: invalid values are refused, listeners hear real changes only', () => {

	const store = new SettingsStore();
	const heard = [];
	store.onChange( ( key, value, by ) => heard.push( [ key, value, by ] ) );
	assert.equal( store.set( 'bloom', 'x' ), false );
	assert.equal( store.set( 'nope', 1 ), false );
	assert.equal( store.set( 'bloom', 0.8 ), false, 'same value: no change' );
	assert.equal( store.isTouched( 'bloom' ), true, 'but touched' );
	assert.equal( store.set( 'bloom', 1.2 ), true );
	assert.equal( store.set( 'bloom', 9 ), true );
	assert.deepEqual( heard, [ [ 'bloom', 1.2, 'user' ], [ 'bloom', 2, 'user' ] ] );

} );

test( 'persistence: only explicit, saveable choices are written and come back as touched', () => {

	const storage = new MemoryStorage();
	const a = new SettingsStore( { storage } );
	a.load();
	a.auto( 'fieldStrength', 0 ); // automatic: not saved
	a.set( 'bloom', 1.4 );
	a.set( 'theme', 'studio' );
	a.set( 'layoutMode', 'sphere' ); // save: false
	a.set( 'running', false ); // save: false
	a.ui( 'sections', { edges: true } );
	const blob = stored( storage );
	assert.equal( blob.v, SETTINGS_VERSION );
	assert.deepEqual( blob.values, { bloom: 1.4, theme: 'studio' } );
	assert.deepEqual( blob.ui, { sections: { edges: true } } );

	const b = new SettingsStore( { storage } );
	assert.deepEqual( b.load().sort(), [ 'bloom', 'theme' ] );
	assert.equal( b.values.bloom, 1.4 );
	assert.equal( b.values.theme, 'studio' );
	assert.equal( b.values.layoutMode, 'force' );
	assert.equal( b.values.fieldStrength, 0.5 );
	assert.equal( b.isTouched( 'theme' ), true );
	assert.deepEqual( b.ui( 'sections' ), { edges: true } );

} );

test( 'persistence: link values are explicit for the session but do not replace saved ones', () => {

	const storage = new MemoryStorage();
	const a = new SettingsStore( { storage } );
	a.set( 'bloom', 1.4 );
	const b = new SettingsStore( { storage } );
	b.load();
	b.set( 'bloom', 0.2, { by: 'link' } );
	b.set( 'theme', 'void', { by: 'link' } );
	assert.equal( b.values.bloom, 0.2 );
	assert.equal( b.isTouched( 'theme' ), true );
	b.set( 'nodeSize', 2 ); // a real choice triggers a save
	assert.deepEqual( stored( storage ).values, { bloom: 1.4, nodeSize: 2 } );
	b.set( 'theme', 'studio' ); // the user takes the key over
	assert.deepEqual( stored( storage ).values, { bloom: 1.4, nodeSize: 2, theme: 'studio' } );

} );

test( 'share link: what it leaves out is the default, whatever this browser has saved', () => {

	// the sender: a clean profile, two changes, one automatic value
	const sender = new SettingsStore();
	sender.auto( 'fieldStrength', 0 );
	sender.set( 'bloom', 1.3 );
	sender.set( 'layoutMode', 'sphere' );
	const link = decodeShare( encodeShare( { dataset: 'imsv/lesmis', settings: sender.shared() } ) );
	assert.deepEqual( link.settings, { layoutMode: 'sphere', bloom: 1.3 } );

	// the receiver once chose the Paper look, a quality and a node size
	const saved = { theme: 'void', edgeBlend: 'additive', bloom: 1.2, nodeGlow: 1.5, nodeSize: 2, fieldStrength: 0.9, quality: 'low', showStatus: false };
	const storage = new MemoryStorage( { [ STORAGE_KEY ]: JSON.stringify( { v: SETTINGS_VERSION, values: saved, ui: { hintSeen: true } } ) } );
	const store = new SettingsStore( { storage } );
	store.load();
	assert.equal( store.values.theme, 'void' );
	const heard = [];
	store.onChange( ( key, value, by ) => heard.push( `${key}:${by}` ) );
	const changed = store.beginLink( link.settings );
	assert.deepEqual( changed.sort(), [ 'bloom', 'edgeBlend', 'fieldStrength', 'layoutMode', 'nodeGlow', 'nodeSize', 'theme' ] );
	assert.deepEqual( heard.sort(), changed.map( ( k ) => `${k}:link` ) );

	// every shareable setting now equals the sender's; what describes the device stays the receiver's
	for ( const spec of SETTINGS ) {

		if ( spec.share === false || spec.key === 'dataset' || spec.key === 'fieldStrength' ) continue;
		assert.deepEqual( store.values[ spec.key ], sender.values[ spec.key ], spec.key );

	}

	assert.equal( store.values.quality, 'low' );
	assert.equal( store.values.showStatus, false );
	// the automatic choice applies again where the link did not decide
	assert.equal( store.isTouched( 'fieldStrength' ), false );
	assert.equal( store.auto( 'fieldStrength', 0 ), true );
	assert.equal( store.isTouched( 'bloom' ), true, 'the link\'s own values are explicit for this visit' );
	// sharing again gives the same link
	assert.deepEqual( store.shared(), link.settings );

	// nothing of it is a preference: storage keeps what was saved, through every later save
	store.ui( 'open', { edges: true } );
	assert.deepEqual( stored( storage ).values, saved );
	store.set( 'exposure', 2 ); // a real choice during the visit is saved next to them
	assert.deepEqual( stored( storage ).values, { ...saved, exposure: 2 } );
	store.set( 'theme', 'studio' ); // and one that replaces a saved value replaces it
	assert.deepEqual( stored( storage ).values, { ...saved, exposure: 2, theme: 'studio' } );
	store.reset( 'nodeSize' ); // an explicit reset of one setting forgets its saved value
	const after = { ...saved, exposure: 2, theme: 'studio' };
	delete after.nodeSize;
	assert.deepEqual( stored( storage ).values, after );

	// the next visit without the link: the receiver's own settings are back
	const again = new SettingsStore( { storage } );
	again.load();
	assert.equal( again.values.edgeBlend, 'additive' );
	assert.equal( again.values.nodeGlow, 1.5 );
	assert.equal( again.values.bloom, 1.2 );
	assert.equal( again.values.theme, 'studio' );

} );

test( 'share link: an empty or unreadable settings member still means "the defaults"', () => {

	const storage = new MemoryStorage( { [ STORAGE_KEY ]: JSON.stringify( { v: SETTINGS_VERSION, values: { theme: 'void', palette: 'ember' } } ) } );
	const store = new SettingsStore( { storage } );
	store.load();
	// a link from a clean profile carries the dataset and the camera only
	store.beginLink( decodeShare( 'ds=imsv/lesmis&c=1,2,300,0,0,0' ).settings );
	assert.deepEqual( { theme: store.values.theme, palette: store.values.palette }, { theme: 'aurora', palette: 'paper' } );
	assert.equal( store.touched.size, 0 );
	// invalid values in a link are skipped, not applied
	const other = new SettingsStore();
	other.beginLink( { bloom: 'x', theme: 'neon', nodeSize: 3 } );
	assert.deepEqual( [ other.values.bloom, other.values.theme, other.values.nodeSize ], [ 0.8, 'aurora', 3 ] );
	assert.deepEqual( [ ...other.touched ], [ 'nodeSize' ] );
	// run-time defaults still move what the link left open
	other.setDefault( 'dust', 0.2 );
	assert.equal( other.values.dust, 0.2 );

} );

test( 'persistence: broken or blocked storage never throws', () => {

	const broken = new SettingsStore( { storage: new MemoryStorage( { [ STORAGE_KEY ]: '{not json' } ) } );
	assert.deepEqual( broken.load(), [] );
	const blocked = new SettingsStore( { storage: { getItem() { throw new Error( 'denied' ); }, setItem() { throw new Error( 'denied' ); } } } );
	assert.deepEqual( blocked.load(), [] );
	assert.equal( blocked.set( 'bloom', 1 ), true );
	assert.equal( new SettingsStore().save(), false );

} );

test( 'migration: version 1 (flat object, d3-style forces) becomes version 2', () => {

	const v1 = { dataset: 'imsv/lesmis', layoutMode: 'sphere', bloom: 1.1, palette: 'ember', charge: 2, linkDistance: 60, gravity: 1.5, theta: 0.9, damping: 0.4, quality: 'high', bogus: 1 };
	const m = migrate( v1 );
	assert.equal( m.v, SETTINGS_VERSION );
	assert.deepEqual( m.values, { bloom: 1.1, palette: 'ember', damping: 0.4, quality: 'high', layoutSize: 2, repulsion: 16 } );
	// through the store
	const store = new SettingsStore( { storage: new MemoryStorage( { [ STORAGE_KEY ]: JSON.stringify( v1 ) } ) } );
	store.load();
	assert.equal( store.values.layoutSize, 2 );
	assert.equal( store.values.repulsion, 16 );
	assert.equal( store.values.layoutMode, 'force', 'session-only keys are not restored' );

} );

test( 'migration: invalid values are dropped, newer versions start clean', () => {

	assert.deepEqual( migrate( { v: 2, values: { bloom: 'abc', theme: 'neon', nodeSize: 99, labels: 'all' }, ui: { hintSeen: true } } ),
		{ v: 2, values: { nodeSize: 4, labels: 'all' }, ui: { hintSeen: true } } );
	assert.deepEqual( migrate( { v: 99, values: { bloom: 1 } } ), { v: 2, values: {}, ui: {} } );
	for ( const junk of [ null, undefined, 3, 'x', [] ] ) assert.deepEqual( migrate( junk ), { v: 2, values: {}, ui: {} } );

} );

test( 'reset all: defaults, untouched, storage cleared of values, dataset kept', () => {

	const storage = new MemoryStorage();
	const store = new SettingsStore( { storage } );
	store.set( 'dataset', 'generated/hsbm-500' );
	store.set( 'bloom', 1.4 );
	store.set( 'theme', 'void' );
	store.auto( 'fieldStrength', 0 );
	store.ui( 'hintSeen', true );
	const heard = [];
	store.onChange( ( key, value, by ) => heard.push( `${key}:${by}` ) );
	const changed = store.reset();
	assert.deepEqual( changed.sort(), [ 'bloom', 'fieldStrength', 'theme' ] );
	assert.deepEqual( heard.sort(), [ 'bloom:reset', 'fieldStrength:reset', 'theme:reset' ] );
	assert.equal( store.values.dataset, 'generated/hsbm-500' );
	assert.equal( store.values.bloom, 0.8 );
	assert.equal( store.touched.size, 1, 'only the dataset stays touched' );
	assert.deepEqual( stored( storage ).values, {} );
	assert.equal( stored( storage ).ui.hintSeen, true, 'UI state survives' );

} );

test( 'run-time defaults move untouched values and define what "non-default" means', () => {

	const store = new SettingsStore();
	store.setDefault( 'dust', 0.2 );
	assert.equal( store.values.dust, 0.2 );
	store.set( 'dust', 0.2 );
	assert.deepEqual( store.shared(), {}, 'equal to the default: not shared' );
	store.set( 'dust', 0.7 );
	store.setDefault( 'dust', 0.3 );
	assert.equal( store.values.dust, 0.7, 'a touched value stays' );
	assert.deepEqual( store.shared(), { dust: 0.7 } );
	assert.equal( store.reset( 'dust' ).length, 1 );
	assert.equal( store.values.dust, 0.3 );

} );

test( 'shared(): explicit non-default values, plus explicit values of automatic keys', () => {

	const store = new SettingsStore();
	store.auto( 'fieldStrength', 0 ); // automatic: the receiver computes it again
	assert.deepEqual( store.shared(), {} );
	store.set( 'bloom', 1.25 );
	store.set( 'theme', 'aurora' ); // explicit but default
	store.set( 'fieldStrength', 0.5 ); // explicit default of an automatic key: must travel
	store.set( 'showStatus', false ); // share: false
	store.set( 'layoutMode', 'radial' );
	assert.deepEqual( store.shared(), { layoutMode: 'radial', fieldStrength: 0.5, bloom: 1.25 } );

} );

test( 'codec: settings round trip, compact, tolerant', () => {

	const settings = { layoutMode: 'stress', bloom: 1.23456, autoRotate: false, theme: 'void', exposure: 0.5, edgeBlend: 'additive', repulsion: 22.5 };
	const text = encodeSettings( settings );
	assert.equal( text, '2~ly:stress~rp:22.5~eb:additive~th:void~ar:0~bl:1.235~ex:0.5' );
	assert.deepEqual( decodeSettings( text ), { version: 2, settings: { ...settings, bloom: 1.235 } } );
	assert.equal( encodeSettings( {} ), '' );
	// unknown codes, broken pairs and out-of-range values
	assert.deepEqual( decodeSettings( '2~zz:1~bl~bl:7~th:neon~ly:flat' ).settings, { bloom: 2, layoutMode: 'flat' } );
	assert.deepEqual( decodeSettings( '99~bl:1' ), { version: null, settings: {} } );
	assert.deepEqual( decodeSettings( '' ), { version: null, settings: {} } );
	assert.deepEqual( decodeSettings( 'garbage' ), { version: null, settings: {} } );

} );

test( 'codec: every shareable setting survives a round trip at a non-default value', () => {

	const settings = {};
	for ( const s of SETTINGS ) {

		if ( s.share === false || s.key === 'dataset' ) continue;
		if ( s.type === 'number' ) settings[ s.key ] = + ( s.min + ( s.max - s.min ) * 0.37 ).toPrecision( 3 );
		else if ( s.type === 'bool' ) settings[ s.key ] = ! s.default;
		else settings[ s.key ] = s.options.find( ( o ) => o.value !== s.default ).value;

	}

	const query = encodeShare( { dataset: 'generated/hsbm-500', settings } );
	const back = decodeShare( query );
	assert.deepEqual( back.settings, settings );
	assert.equal( back.dataset, 'generated/hsbm-500' );
	assert.ok( query.length < 420, `all settings at once stay compact (${query.length})` );
	assert.match( query, /^[A-Za-z0-9\-._~:/?&=,+%]*$/, 'URL-safe without escaping' );

} );

test( 'codec: camera pose', () => {

	const pose = { position: [ 123.4567, - 0.000123456, 9876.54 ], target: [ 0, 1.5, - 20 ], fov: 45 };
	const text = encodePose( pose );
	assert.equal( text, '123.5,-0.0001235,9877,0,1.5,-20' );
	const back = decodePose( text );
	assert.deepEqual( back.target, [ 0, 1.5, - 20 ] );
	assert.equal( back.fov, 45 );
	for ( let k = 0; k < 3; k ++ ) assert.ok( Math.abs( back.position[ k ] - pose.position[ k ] ) <= Math.abs( pose.position[ k ] ) * 1e-3 );
	assert.equal( decodePose( encodePose( { ...pose, fov: 75 } ) ).fov, 75 );
	assert.equal( decodePose( '1,2,3' ), null );
	assert.equal( decodePose( '1,2,3,4,5,x' ), null );
	assert.equal( decodePose( null ), null );
	assert.equal( encodePose( { position: [ 1, 2, NaN ], target: [ 0, 0, 0 ] } ), '' );
	assert.equal( encodePose( null ), '' );

} );

test( 'codec: full share link round trip with explore state', () => {

	const share = {
		dataset: 'imsv/lesmis',
		settings: { layoutMode: 'sphere', theme: 'studio', bloom: 1.2 },
		camera: { position: [ 10, 20, 300 ], target: [ 1, 2, 3 ], fov: 45 },
		explore: { lens: { color: 'pagerank', ramp: 'magma' }, selection: 11, path: [ 3, 48 ], note: 'Zoë → “Valjean”' },
	};
	const query = encodeShare( share );
	assert.ok( query.startsWith( 'ds=imsv/lesmis&s=2~ly:sphere~th:studio~bl:1.2&c=10,20,300,1,2,3&x=' ) );
	const back = decodeShare( new URLSearchParams( query ) );
	assert.deepEqual( back, { ...share, version: 2 } );
	assert.deepEqual( decodeShare( '?' + query ), { ...share, version: 2 } );
	assert.equal( hasShare( new URLSearchParams( query ) ), true );
	assert.equal( hasShare( new URLSearchParams( 'ds=imsv/lesmis&quality=high' ) ), false );
	// empty members are left out; a broken explore blob is ignored
	assert.equal( encodeShare( { dataset: 'a/b', settings: {}, camera: null, explore: {} } ), 'ds=a/b' );
	assert.equal( decodeShare( 'ds=a/b&x=@@@' ).explore, null );
	assert.deepEqual( decodeShare( '' ), { dataset: null, settings: {}, camera: null, explore: null, version: null } );

} );

test( 'legacy URL parameters keep working', () => {

	const p = ( q ) => new URLSearchParams( q );
	assert.deepEqual( parseLegacyParams( p( 'ds=generated/hsbm-5k&layout=sphere&autorotate=0' ) ), { dataset: 'generated/hsbm-5k', settings: { layoutMode: 'sphere', autoRotate: false } } );
	assert.deepEqual( parseLegacyParams( p( '' ), '#imsv/karate' ), { dataset: 'imsv/karate', settings: {} } );
	assert.deepEqual( parseLegacyParams( p( 'ds=a/b' ), '#c/d' ).dataset, 'a/b', '?ds= wins over the hash' );
	assert.deepEqual( parseLegacyParams( p( 'layout=radial&autorotate=1' ) ).settings, { layoutMode: 'radial', autoRotate: true } );
	assert.deepEqual( parseLegacyParams( p( 'layout=bogus&quality=high' ) ), { settings: {} } );
	assert.deepEqual( parseLegacyParams( p( '' ), '#%E0%A4%A' ), { settings: {} }, 'malformed hash' );

} );

test( 'automatic per-graph choices', () => {

	const g = ( E, clusters ) => ( { E, stats: { clusters } } );
	assert.deepEqual( autoChoices( g( 254, 6 ), { tubes: true } ), { edgeBlend: 'normal', edgeStyle: 'luminous', fieldStrength: 0.5 } );
	assert.equal( autoChoices( g( AUTO.additiveEdges + 1, 40 ) ).edgeBlend, 'additive' );
	assert.equal( autoChoices( g( 39817, 40 ) ).edgeBlend, 'normal', 'tens of thousands of additive strands wash the picture out' );
	assert.equal( autoChoices( { E: 500, stats: { clusters: 18 }, clustering: { communities: 2, mainLevel: 1 } } ).fieldStrength, 0, 'the communities of the main level count, not all levels' );
	assert.equal( autoChoices( g( 78, 4 ), { tubes: true } ).edgeStyle, 'solid' );
	assert.equal( autoChoices( g( 78, 4 ), { tubes: false } ).edgeStyle, 'luminous', 'no tubes in this tier' );
	assert.equal( autoChoices( g( 500, 1 ) ).fieldStrength, 0 );
	assert.equal( autoChoices( g( 500, 9 ), { fieldDefault: 0.3 } ).fieldStrength, 0.3 );
	assert.deepEqual( autoChoices( null ), { edgeBlend: 'normal', edgeStyle: 'luminous', fieldStrength: 0 } );

} );

test( 'tier description names what the configuration turns on', () => {

	const env = { coarsePointer: false, smallScreen: false, softwareAdapter: false, edges: 500, dpr: 1, cssWidth: 1280, cssHeight: 800 };
	const low = describeTier( resolveQuality( 'low', env ) ).join( ', ' );
	const ultra = describeTier( resolveQuality( 'ultra', env ) ).join( ', ' );
	assert.match( low, /flat-shaded nodes/ );
	assert.doesNotMatch( low, /shadows|occlusion|depth of field/ );
	assert.match( ultra, /temporal anti-aliasing/ );
	assert.match( ultra, /clear coat/ );
	assert.match( ultra, /soft shadows \(2048\)/ );
	assert.match( ultra, /depth of field/ );
	assert.match( ultra, /6 MP/ );
	assert.deepEqual( describeTier( null ), [] );

} );
