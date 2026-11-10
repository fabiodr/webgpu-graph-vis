// Command registry of the explore layer: the entries of the command palette. The explore layer
// registers its own commands, the application shell adds its own (layout modes, quality tiers,
// themes, …) through explore.commands.register(). Pure logic, no DOM.
//
//   const commands = new CommandRegistry();
//   commands.register( { id, title, group?, keywords?, shortcut?, hint?, suggested?, enabled?(), run() } )
//         -> () => void (unregisters). id: unique string, a second registration replaces the
//         first. title: what the palette shows. group: a short heading ("Lens", "View", …).
//         keywords: string | string[] of extra search terms. shortcut: display text ("I", "⌘K").
//         hint: a short note shown beside the title. suggested: listed in the empty palette.
//         enabled(): return false to hide the command for now (no selection, no graph, …).
//   commands.unregister( id ) -> boolean
//   commands.get( id ), commands.has( id ), commands.list() -> every registered command, in
//         registration order (disabled ones included)
//   commands.available() -> the commands whose enabled() does not return false
//   commands.search( query, { limit = 20 } ) -> [ { command, tier, field } ], best first. Tiers are
//         those of fuzzy.js on the title; a match of the group or of a keyword counts one tier
//         lower than the same match on the title, and never as exact; in a tie the title match
//         comes first, then the order of registration.
//   commands.suggested( limit = 6 ) -> recently run commands first, then the ones marked suggested
//   commands.run( id, context? ) -> Promise<boolean>: false when the command is unknown, disabled or
//         threw (the error goes to console.error and to onError); remembers it as recently run
//   commands.onChange( fn ) -> () => void: called after register / unregister
//   commands.onError = ( error, command ) => void

import { foldText, matchTier } from './fuzzy.js';

export class CommandRegistry {

	constructor() {

		this._commands = new Map();
		this._listeners = new Set();
		this._recent = [];
		this.onError = null;

	}

	register( command ) {

		if ( ! command || typeof command !== 'object' ) throw new TypeError( 'commands.register: a command object is required' );
		const id = command.id;
		if ( typeof id !== 'string' || ! id ) throw new TypeError( 'commands.register: `id` must be a non-empty string' );
		if ( typeof command.title !== 'string' || ! command.title ) throw new TypeError( `commands.register( ${id} ): \`title\` must be a non-empty string` );
		if ( typeof command.run !== 'function' ) throw new TypeError( `commands.register( ${id} ): \`run\` must be a function` );
		const keywords = command.keywords === undefined || command.keywords === null ? [] : Array.isArray( command.keywords ) ? command.keywords.map( String ) : String( command.keywords ).split( /[,\s]+/ ).filter( Boolean );
		const entry = {
			...command, id, keywords,
			group: command.group ? String( command.group ) : '',
			_title: foldText( command.title ),
			_extra: [ command.group || '', ...keywords ].map( foldText ).filter( Boolean ),
		};
		this._commands.delete( id ); // a replacement moves to the end, like a fresh registration
		this._commands.set( id, entry );
		this._changed();
		return () => {

			if ( this._commands.get( id ) === entry ) this.unregister( id );

		};

	}

	unregister( id ) {

		const had = this._commands.delete( id );
		if ( had ) {

			this._recent = this._recent.filter( ( r ) => r !== id );
			this._changed();

		}

		return had;

	}

	get( id ) {

		return this._commands.get( id ) || null;

	}

	has( id ) {

		return this._commands.has( id );

	}

	list() {

		return [ ...this._commands.values() ];

	}

	available() {

		return this.list().filter( ( c ) => this._enabled( c ) );

	}

	search( query, { limit = 20 } = {} ) {

		const q = foldText( query );
		if ( ! q ) return [];
		const out = [];
		let order = 0;
		for ( const command of this._commands.values() ) {

			order ++;
			if ( ! this._enabled( command ) ) continue;
			let tier = matchTier( command._title, q ), field = 'title';
			for ( const extra of command._extra ) {

				const t = matchTier( extra, q );
				if ( t < 0 ) continue;
				const lowered = Math.min( 4, t + 1 );
				if ( tier < 0 || lowered < tier ) {

					tier = lowered;
					field = 'keyword';

				}

			}

			if ( tier >= 0 ) out.push( { command, tier, field, order } );

		}

		out.sort( ( a, b ) => a.tier - b.tier || ( a.field === 'title' ? 0 : 1 ) - ( b.field === 'title' ? 0 : 1 ) || a.order - b.order );
		return out.slice( 0, limit ).map( ( { command, tier, field } ) => ( { command, tier, field } ) );

	}

	suggested( limit = 6 ) {

		const out = [];
		const add = ( c ) => {

			if ( c && this._enabled( c ) && ! out.includes( c ) && out.length < limit ) out.push( c );

		};

		for ( const id of this._recent ) add( this._commands.get( id ) );
		for ( const c of this._commands.values() ) if ( c.suggested ) add( c );
		return out;

	}

	async run( id, context ) {

		const command = this._commands.get( id );
		if ( ! command || ! this._enabled( command ) ) return false;
		this._recent = [ id, ...this._recent.filter( ( r ) => r !== id ) ].slice( 0, 8 );
		try {

			await command.run( context );
			return true;

		} catch ( e ) {

			console.error( e );
			if ( this.onError ) this.onError( e, command );
			return false;

		}

	}

	onChange( fn ) {

		this._listeners.add( fn );
		return () => this._listeners.delete( fn );

	}

	_enabled( command ) {

		if ( typeof command.enabled !== 'function' ) return command.enabled !== false;
		try {

			return command.enabled() !== false;

		} catch ( e ) {

			return false;

		}

	}

	_changed() {

		for ( const fn of [ ...this._listeners ] ) {

			try {

				fn( this );

			} catch ( e ) {

				console.error( e );

			}

		}

	}

}
