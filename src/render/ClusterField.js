// Community field (App module, order 13): a soft glow volume per community that makes the cluster
// structure readable at a glance and gives the scene depth.
//
// API
//   const field = app.use( new ClusterField( { strength = 0.5, level = 'auto' } ) )
//   field.strength     0..1, 0 = off
//   field.level        'auto' | hierarchy height (1 = finest communities … D = coarsest)
//   field.visible
export class ClusterField {

	constructor( options = {} ) {

		this.order = 13;
		this.isClusterField = true;
		this.strength = options.strength ?? 0.5;
		this.level = options.level ?? 'auto';
		this.visible = true;

	}

	init() {}

	setGraph() {}

	update() {}

	dispose() {}

}
