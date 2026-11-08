// Headless WebGPU screenshot / smoke-test harness (Chromium + SwiftShader Vulkan).
//
//   node scripts/shot.mjs <path> [options]
//     path            page path served by Vite, e.g. "/" or "/dev/core.html?ds=imsv/lesmis"
//     --out FILE      screenshot path (default shots/<slug>.png)
//     --wait MS       extra time after window.__ready (default 3000)
//     --until EXPR    poll (every 250 ms, up to --timeout) until this JS expression is truthy
//     --eval EXPR     JS evaluated (awaited) after waiting, before the screenshot
//     --report EXPR   JS expression whose JSON value is printed in the summary
//     --size WxH      viewport (default 1280x800)
//     --mobile        emulate a phone (Pixel 7: 412x915, touch, DPR 2.625)
//     --dpr N         deviceScaleFactor (default 1)
//     --timeout MS    overall timeout (default 120000)
//     --base URL      use an already running server instead of starting Vite
//     --verbose       include console.log/info messages in the summary
//
// Prints a JSON summary with console errors/warnings and page errors. Exit code 1 on page
// errors or if window.__ready never becomes true. SwiftShader is a CPU rasterizer, so frame
// rates here are NOT representative of real GPUs.
import { createRequire } from 'node:module';
import { createServer } from 'vite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

// Playwright is not a project dependency: use a globally installed copy (or `npm i -D playwright`).
const { chromium, devices } = await import( 'playwright' ).catch( () => createRequire( import.meta.url )( 'playwright' ) );

const args = process.argv.slice( 2 );
const path = args.find( ( a ) => ! a.startsWith( '--' ) && ! args[ args.indexOf( a ) - 1 ]?.startsWith( '--' ) ) || '/';
const opt = ( name, def ) => {

	const i = args.indexOf( '--' + name );
	return i >= 0 ? args[ i + 1 ] : def;

};

const flag = ( name ) => args.includes( '--' + name );
const slug = path.replace( /[^a-z0-9]+/gi, '_' ).replace( /^_|_$/g, '' ) || 'index';
const out = opt( 'out', `shots/${slug}.png` );
const wait = + opt( 'wait', 3000 );
const timeout = + opt( 'timeout', 120000 );
const [ w, h ] = opt( 'size', '1280x800' ).split( 'x' ).map( Number );

let server = null;
let base = opt( 'base', null );
if ( ! base ) {

	server = await createServer( { server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' } );
	await server.listen();
	const addr = server.httpServer.address();
	base = `http://127.0.0.1:${addr.port}`;

}

const browser = await chromium.launch( {
	headless: true,
	// These flags matter: without ANGLE/SwiftShader for GL the WebGPU canvas swap chain is lost.
	args: [ '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-sandbox', '--ignore-gpu-blocklist' ],
} );

const contextOptions = flag( 'mobile' ) ? { ...devices[ 'Pixel 7' ] } : { viewport: { width: w, height: h }, deviceScaleFactor: + opt( 'dpr', 1 ) };
const context = await browser.newContext( contextOptions );
const page = await context.newPage();
const consoleMsgs = [];
const pageErrors = [];
page.on( 'console', ( m ) => {

	if ( m.type() !== 'error' && m.type() !== 'warning' && ! flag( 'verbose' ) ) return;
	const text = `[${m.type()}] ${m.text()}`.slice( 0, 400 );
	const prev = consoleMsgs.find( ( c ) => c.text === text );
	if ( prev ) prev.count ++;
	else consoleMsgs.push( { text, count: 1 } );

} );
page.on( 'pageerror', ( e ) => pageErrors.push( String( e && e.stack || e ) ) );

const summary = { url: base + path, out, ok: false };
const t0 = Date.now();
try {

	await page.goto( base + path, { timeout } );
	await page.waitForFunction( () => window.__ready === true || window.__error, null, { timeout, polling: 250 } );
	const err = await page.evaluate( () => window.__error && String( window.__error ) );
	if ( err ) throw new Error( 'page reported: ' + err );
	const until = opt( 'until', null );
	if ( until ) await page.waitForFunction( until, null, { timeout: Math.max( 1000, timeout - ( Date.now() - t0 ) ), polling: 250 } );
	await page.waitForTimeout( wait );
	const ev = opt( 'eval', null );
	if ( ev ) summary.eval = await page.evaluate( `(async () => { ${ev} })()` );
	const rep = opt( 'report', null );
	if ( rep ) summary.report = await page.evaluate( `(async () => (${rep}))()` );
	summary.fps = await page.evaluate( () => window.__app && window.__app.fps );
	mkdirSync( dirname( out ), { recursive: true } );
	await page.screenshot( { path: out } );
	summary.ok = pageErrors.length === 0;

} catch ( e ) {

	summary.error = String( e.message || e );
	try {

		mkdirSync( dirname( out ), { recursive: true } );
		await page.screenshot( { path: out } );

	} catch ( _ ) { /* ignore */ }

}

summary.seconds = ( Date.now() - t0 ) / 1000;
summary.console = consoleMsgs.slice( 0, 30 ).map( ( c ) => ( c.count > 1 ? `${c.text} (x${c.count})` : c.text ) );
summary.pageErrors = pageErrors.slice( 0, 10 );
console.log( JSON.stringify( summary, null, 2 ) );
await browser.close();
if ( server ) await server.close();
process.exit( summary.ok ? 0 : 1 );
