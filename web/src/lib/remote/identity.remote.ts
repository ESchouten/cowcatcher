import { command, query } from '$app/server';
import * as v from 'valibot';
import type { DatabaseSync } from 'node:sqlite';
import { getIdentityDatabases, openIdentityDatabase } from '$lib/server/identity-databases';
import {
	mergeIdentities,
	requestIdentityFinalization,
	setIdentityAnimalNumber,
	splitTracklet
} from '$lib/server/identity-store';

export interface CowIdentity {
	identity: string;
	animalNumber: string | null;
	tracklets: number;
	preview: string | null;
}

export interface CowTracklet {
	id: string;
	source: string;
	observations: number;
	preview: string;
}

export interface CowIdentityDatabase {
	database: string;
	label: string;
	tracklets: number;
	pendingTracklets: number;
	pending: CowTracklet[];
	finalizeRequested: boolean;
	finalizeError: string | null;
	identities: CowIdentity[];
}

function control(database: DatabaseSync, key: string): string | null {
	const row = database.prepare('SELECT value FROM control WHERE key = ?').get(key) as
		| { value: string }
		| undefined;
	return row?.value ?? null;
}

export const getCowIdentities = query(async (): Promise<CowIdentityDatabase[]> => {
	const configured = await getIdentityDatabases();
	return Promise.all(
		configured.map(async ({ database, label }) => {
			let connection: DatabaseSync;
			try {
				connection = await openIdentityDatabase(database);
			} catch {
				return {
					database,
					label,
					tracklets: 0,
					pendingTracklets: 0,
					pending: [],
					finalizeRequested: false,
					finalizeError: null,
					identities: []
				};
			}

			try {
				const tracklets = Number(
					(connection.prepare('SELECT COUNT(*) AS count FROM tracklets').get() as { count: number })
						.count
				);
				const pendingTracklets = Number(
					(
						connection
							.prepare(
								`SELECT COUNT(*) AS count
								 FROM tracklets t
								 LEFT JOIN identity_tracklets it ON it.tracklet_id = t.id
								 WHERE it.tracklet_id IS NULL`
							)
							.get() as { count: number }
					).count
				);
				const rows = connection
					.prepare(
						`SELECT i.identity, i.animal_number, COUNT(it.tracklet_id) AS tracklets,
							(SELECT t.preview FROM tracklets t
							 JOIN identity_tracklets preview_it ON preview_it.tracklet_id = t.id
							 WHERE preview_it.identity = i.identity
							 ORDER BY t.observations DESC LIMIT 1) AS preview
						 FROM identities i
						 LEFT JOIN identity_tracklets it ON it.identity = i.identity
						 GROUP BY i.identity, i.animal_number
						 ORDER BY i.identity`
					)
					.all() as Array<{
					identity: string;
					animal_number: string | null;
					tracklets: number;
					preview: Uint8Array | null;
				}>;
				const pending = connection
					.prepare(
						`SELECT t.id, t.source, t.observations, t.preview
						 FROM tracklets t
						 LEFT JOIN identity_tracklets it ON it.tracklet_id = t.id
						 WHERE it.tracklet_id IS NULL
						 ORDER BY t.observations DESC, t.id
						 LIMIT 12`
					)
					.all() as Array<{
					id: string;
					source: string;
					observations: number;
					preview: Uint8Array;
				}>;
				return {
					database,
					label,
					tracklets,
					pendingTracklets,
					pending: pending.map((tracklet) => ({
						id: tracklet.id,
						source: tracklet.source,
						observations: Number(tracklet.observations),
						preview: `data:image/jpeg;base64,${Buffer.from(tracklet.preview).toString('base64')}`
					})),
					finalizeRequested: control(connection, 'finalize_requested') === '1',
					finalizeError: control(connection, 'finalize_error') || null,
					identities: rows.map((row) => ({
						identity: row.identity,
						animalNumber: row.animal_number,
						tracklets: Number(row.tracklets),
						preview: row.preview
							? `data:image/jpeg;base64,${Buffer.from(row.preview).toString('base64')}`
							: null
					}))
				};
			} finally {
				connection.close();
			}
		})
	);
});

export const getCowIdentityTracklets = query(
	v.object({ database: v.string(), identity: v.string() }),
	async ({ database, identity }): Promise<CowTracklet[]> => {
		const connection = await openIdentityDatabase(database);
		try {
			const rows = connection
				.prepare(
					`SELECT t.id, t.source, t.observations, t.preview
					 FROM tracklets t
					 JOIN identity_tracklets it ON it.tracklet_id = t.id
					 WHERE it.identity = ?
					 ORDER BY t.source, t.first_frame, t.id`
				)
				.all(identity) as Array<{
				id: string;
				source: string;
				observations: number;
				preview: Uint8Array;
			}>;
			return rows.map((row) => ({
				id: row.id,
				source: row.source,
				observations: Number(row.observations),
				preview: `data:image/jpeg;base64,${Buffer.from(row.preview).toString('base64')}`
			}));
		} finally {
			connection.close();
		}
	}
);

const databaseInput = v.object({ database: v.string() });

export const finalizeCowIdentities = command(databaseInput, async ({ database }) => {
	const connection = await openIdentityDatabase(database);
	try {
		requestIdentityFinalization(connection);
	} finally {
		connection.close();
	}
});

export const setCowAnimalNumber = command(
	v.object({
		database: v.string(),
		identity: v.string(),
		animalNumber: v.string()
	}),
	async ({ database, identity, animalNumber }) => {
		const connection = await openIdentityDatabase(database);
		try {
			setIdentityAnimalNumber(connection, identity, animalNumber);
		} finally {
			connection.close();
		}
	}
);

export const mergeCowIdentities = command(
	v.object({ database: v.string(), source: v.string(), target: v.string() }),
	async ({ database, source, target }) => {
		if (source === target) throw new Error('Choose a different target identity');
		const connection = await openIdentityDatabase(database);
		try {
			mergeIdentities(connection, source, target);
		} finally {
			connection.close();
		}
	}
);

export const splitCowTracklet = command(
	v.object({ database: v.string(), identity: v.string(), tracklet: v.string() }),
	async ({ database, identity, tracklet }) => {
		const connection = await openIdentityDatabase(database);
		try {
			splitTracklet(connection, identity, tracklet);
		} finally {
			connection.close();
		}
	}
);
