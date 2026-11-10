// Explore layer: everything a person uses to read the graph rather than to style it (search, metric
// lens controls and legend, graph insights, node card actions, path / neighbourhood tools, tours).
//
// main.js installs it once with the application context:
//
//   installExplore( { app, settings, set, data, layout, nodes, edges, labels, env, rig, interaction,
//                     card, lens, tools, panel, slots, gfx } ) -> explore API (window.__ui.explore)
//
// The layer builds its own DOM surfaces and listens to app.events; it does not own the settings
// panel, the camera rig or main.js.
export function installExplore( /* ctx */ ) {

	return {};

}
