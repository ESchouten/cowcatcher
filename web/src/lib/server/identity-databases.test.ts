import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { describe, expect, it } from 'vitest';

import {
	IDENTITY_BUSY_TIMEOUT_MS,
	IDENTITY_SCHEMA_VERSION,
	openConfiguredIdentityDatabase,
	validateIdentityDatabase
} from './identity-databases';

function initializedDatabase(): DatabaseSync {
	const database = new DatabaseSync(':memory:');
	const schema = readFileSync(
		resolve(process.cwd(), '../detector/src/aidetector/reid/schema.sql'),
		'utf8'
	);
	database.exec(schema);
	return database;
}

describe('shared identity database contract', () => {
	it('accepts the exact Python-owned schema and required control row', () => {
		const database = initializedDatabase();

		expect(() => validateIdentityDatabase(database)).not.toThrow();
		expect(
			(database.prepare('PRAGMA user_version').get() as { user_version: number }).user_version
		).toBe(IDENTITY_SCHEMA_VERSION);
		database.close();
	});

	it('rejects an old or incomplete identity database', () => {
		const database = new DatabaseSync(':memory:');
		database.exec('CREATE TABLE identities (identity TEXT PRIMARY KEY)');

		expect(() => validateIdentityDatabase(database)).toThrow(
			'Unsupported identity database schema'
		);
		database.close();
	});

	it('rejects the pre-zone schema without migrating it', () => {
		const database = initializedDatabase();
		database.exec('PRAGMA user_version = 1');

		expect(() => validateIdentityDatabase(database)).toThrow(
			'Unsupported identity database schema 1'
		);
		expect(
			(database.prepare('PRAGMA user_version').get() as { user_version: number }).user_version
		).toBe(1);
		database.close();
	});

	it('uses the same busy timeout expected by the Python catalog', () => {
		const database = initializedDatabase();
		database.exec(`PRAGMA busy_timeout = ${IDENTITY_BUSY_TIMEOUT_MS}`);

		expect((database.prepare('PRAGMA busy_timeout').get() as { timeout: number }).timeout).toBe(
			5_000
		);
		database.close();
	});

	it('does not initialize a missing catalog from the web process', async () => {
		const directory = mkdtempSync(resolve(tmpdir(), 'identity-preinit-'));
		const databasePath = resolve(directory, 'identities.sqlite');
		try {
			await expect(openConfiguredIdentityDatabase({ path: databasePath })).rejects.toThrow(
				'Start the detector to initialize identities.'
			);
			expect(() => readFileSync(databasePath)).toThrow();
		} finally {
			rmSync(directory, { recursive: true });
		}
	});
});
