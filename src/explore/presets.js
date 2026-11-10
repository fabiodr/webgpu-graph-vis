// Lens presets: one click that sets several channels of the metric lens at once. Pure data and
// two small functions; the lens bar shows them as chips, the command palette as commands, the
// tour uses the same channel triples for its stops.
//
//   LENS_PRESETS  [ { id, label, description, color, size, glow, ramp?, top?, needsSelection? } ]
//        color / size / glow: metric ids of the lens, null = the graph's own look (community
//        palette, degree radii, no glow)
//   applyPreset( lens, preset | id ) -> Promise<boolean[]>   sets the three channels
//   matchPreset( state ) -> id | null   the preset that the lens state ( lens.state ) shows now
//   applyChannels( lens, { color, size, glow, ramp?, reverse?, top? } ) -> Promise<boolean[]>

export const LENS_PRESETS = [
	{ id: 'influence', label: 'Influence', description: 'Size by PageRank, colour by betweenness: who matters and who connects.', color: 'betweenness', size: 'pagerank', glow: null, ramp: 'viridis' },
	{ id: 'brokers', label: 'Brokers', description: 'Size and glow by betweenness on the community colours: who links the groups.', color: null, size: 'betweenness', glow: 'betweenness', top: 0.03 },
	{ id: 'cohesion', label: 'Cohesion', description: 'Colour by k-core: how deep a node sits in the dense core.', color: 'core', size: null, glow: null, ramp: 'mint' },
	{ id: 'reach', label: 'Reach', description: 'Colour by closeness: how near a node is to everything else.', color: 'closeness', size: null, glow: null, ramp: 'ice' },
	{ id: 'relevance', label: 'Relevance to selection', description: 'Colour and size by a random walk that restarts at the selected node.', color: 'relevance', size: 'relevance', glow: null, ramp: 'magma', needsSelection: true },
	{ id: 'communities', label: 'Communities', description: 'Back to the community palette and degree sizes.', color: null, size: null, glow: null },
];

export function applyChannels( lens, { color = null, size = null, glow = null, ramp, reverse, top } = {} ) {

	const colorOpts = {};
	if ( ramp ) colorOpts.ramp = ramp;
	if ( reverse !== undefined ) colorOpts.reverse = !! reverse;
	return Promise.all( [
		lens.setColor( color, color ? colorOpts : undefined ),
		lens.setSize( size ),
		lens.setGlow( glow, glow && top ? { top } : undefined ),
	] );

}

export function applyPreset( lens, preset ) {

	const p = typeof preset === 'string' ? LENS_PRESETS.find( ( x ) => x.id === preset ) : preset;
	if ( ! p ) return Promise.resolve( [] );
	return applyChannels( lens, { ...p, reverse: false } );

}

export function matchPreset( state ) {

	if ( ! state ) return null;
	const same = ( a, b ) => ( a || null ) === ( b || null );
	const p = LENS_PRESETS.find( ( x ) => same( x.color, state.color ) && same( x.size, state.size ) && same( x.glow, state.glow ) );
	return p ? p.id : null;

}
