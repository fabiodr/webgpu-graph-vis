import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { readdirSync, existsSync } from 'node:fs';

// Dev pages under dev/ are served by `npm run dev` (http://localhost:5173/dev/<name>.html)
// but are not part of the production build.
const devPages = existsSync( 'dev' )
	? Object.fromEntries( readdirSync( 'dev' ).filter( ( f ) => f.endsWith( '.html' ) ).map( ( f ) => [ 'dev_' + f.replace( '.html', '' ), resolve( 'dev', f ) ] ) )
	: {};

export default defineConfig( ( { command } ) => ( {
	base: './',
	server: { host: true },
	build: {
		target: 'es2022',
		chunkSizeWarningLimit: 2000,
		rollupOptions: {
			input: command === 'build' ? { main: resolve( 'index.html' ) } : { main: resolve( 'index.html' ), ...devPages },
		},
	},
	worker: { format: 'es' },
} ) );
