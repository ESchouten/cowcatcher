import type { DatabaseSync } from 'node:sqlite';

function transaction<T>(database: DatabaseSync, action: () => T): T {
	database.exec('BEGIN IMMEDIATE');
	try {
		const result = action();
		database.exec('COMMIT');
		return result;
	} catch (error) {
		database.exec('ROLLBACK');
		throw error;
	}
}

function bumpRevision(database: DatabaseSync): void {
	database
		.prepare(
			`INSERT INTO control (key, value) VALUES ('revision', '1')
			 ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1`
		)
		.run();
}

function nextIdentity(database: DatabaseSync, sourceIdentity: string): string {
	const prefix = /^(.*)-(\d+)$/.exec(sourceIdentity)?.[1] ?? 'cow';
	const pattern = new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-(\\d+)$`);
	const identities = database.prepare('SELECT identity FROM identities').all() as Array<{
		identity: string;
	}>;
	const used = new Set(identities.map(({ identity }) => identity));
	let number =
		identities.reduce((maximum, { identity }) => {
			const candidate = pattern.exec(identity);
			return candidate ? Math.max(maximum, Number(candidate[1])) : maximum;
		}, 0) + 1;
	while (used.has(`${prefix}-${String(number).padStart(4, '0')}`)) number += 1;
	return `${prefix}-${String(number).padStart(4, '0')}`;
}

export function requestIdentityFinalization(database: DatabaseSync): void {
	transaction(database, () => {
		database
			.prepare(
				`INSERT INTO control (key, value) VALUES ('finalize_error', '')
				 ON CONFLICT(key) DO UPDATE SET value = excluded.value`
			)
			.run();
		database
			.prepare(
				`INSERT INTO control (key, value) VALUES ('finalize_requested', '1')
				 ON CONFLICT(key) DO UPDATE SET value = excluded.value`
			)
			.run();
	});
}

export function setIdentityAnimalNumber(
	database: DatabaseSync,
	identity: string,
	animalNumber: string
): void {
	transaction(database, () => {
		const result = database
			.prepare('UPDATE identities SET animal_number = ? WHERE identity = ?')
			.run(animalNumber.trim() || null, identity);
		if (Number(result.changes) !== 1) throw new Error('Identity does not exist');
		bumpRevision(database);
	});
}

export function mergeIdentities(database: DatabaseSync, source: string, target: string): void {
	if (source === target) throw new Error('Choose a different target identity');
	transaction(database, () => {
		const rows = database
			.prepare('SELECT identity, animal_number FROM identities WHERE identity IN (?, ?)')
			.all(source, target) as Array<{ identity: string; animal_number: string | null }>;
		if (rows.length !== 2) throw new Error('Identity does not exist');
		const sourceNumber = rows.find((row) => row.identity === source)?.animal_number ?? null;
		const targetNumber = rows.find((row) => row.identity === target)?.animal_number ?? null;
		if (sourceNumber && targetNumber && sourceNumber !== targetNumber) {
			throw new Error('Remove one of the animal numbers before merging these identities');
		}

		database
			.prepare('UPDATE identity_tracklets SET identity = ? WHERE identity = ?')
			.run(target, source);
		if (sourceNumber && !targetNumber) {
			database.prepare('UPDATE identities SET animal_number = NULL WHERE identity = ?').run(source);
			database
				.prepare('UPDATE identities SET animal_number = ? WHERE identity = ?')
				.run(sourceNumber, target);
		}
		database.prepare('DELETE FROM identities WHERE identity = ?').run(source);
		bumpRevision(database);
	});
}

export function splitTracklet(database: DatabaseSync, identity: string, tracklet: string): string {
	return transaction(database, () => {
		const count = Number(
			(
				database
					.prepare('SELECT COUNT(*) AS count FROM identity_tracklets WHERE identity = ?')
					.get(identity) as { count: number }
			).count
		);
		if (count <= 1) throw new Error('An identity must keep at least one tracklet');
		const assignment = database
			.prepare('SELECT identity FROM identity_tracklets WHERE tracklet_id = ?')
			.get(tracklet) as { identity: string } | undefined;
		if (assignment?.identity !== identity) throw new Error('Tracklet does not belong to identity');

		const newIdentity = nextIdentity(database, identity);
		database.prepare('INSERT INTO identities (identity) VALUES (?)').run(newIdentity);
		database
			.prepare('UPDATE identity_tracklets SET identity = ? WHERE tracklet_id = ?')
			.run(newIdentity, tracklet);
		bumpRevision(database);
		return newIdentity;
	});
}
