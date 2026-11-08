// Fetch + parse + prepare helpers shared by the app and dev pages.
import { parseGraphJson } from './format.js';
import { prepareGraph } from '../graph/prepare.js';

export async function fetchManifest( base = 'data/manifest.json' ) {

	const res = await fetch( base );
	if ( ! res.ok ) throw new Error( `manifest: HTTP ${res.status}` );
	return res.json();

}

export async function fetchGraph( url ) {

	const res = await fetch( url );
	if ( ! res.ok ) throw new Error( `${url}: HTTP ${res.status}` );
	const json = await res.json();
	return parseGraphJson( json, url.split( '/' ).pop().replace( /\.json$/, '' ) );

}

/** Fetches and prepares a dataset (canonical graph must carry a hierarchy or be small). */
export async function loadPrepared( url, prepareOptions ) {

	const g = await fetchGraph( url );
	return prepareGraph( g, prepareOptions );

}
