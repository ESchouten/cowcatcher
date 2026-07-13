import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { DatabaseSync } from 'node:sqlite';
import {
	mergeIdentities,
	requestIdentityFinalization,
	splitTracklet
} from '../src/lib/server/identity-store.ts';

function databaseFixture() {
	const directory = mkdtempSync(path.join(tmpdir(), 'identity-store-'));
	const database = new DatabaseSync(path.join(directory, 'identities.sqlite'));
	database.exec(`
		PRAGMA foreign_keys = ON;
		CREATE TABLE identities (identity TEXT PRIMARY KEY, animal_number TEXT UNIQUE);
		CREATE TABLE tracklets (id TEXT PRIMARY KEY);
		CREATE TABLE identity_tracklets (
			tracklet_id TEXT PRIMARY KEY REFERENCES tracklets(id) ON DELETE CASCADE,
			identity TEXT NOT NULL REFERENCES identities(identity) ON DELETE CASCADE
		);
		CREATE TABLE control (key TEXT PRIMARY KEY, value TEXT NOT NULL);
		INSERT INTO control VALUES ('revision', '0');
		INSERT INTO identities VALUES ('identity-0001', NULL), ('identity-0002', NULL);
		INSERT INTO tracklets VALUES ('track-1'), ('track-2'), ('track-3');
		INSERT INTO identity_tracklets VALUES
			('track-1', 'identity-0001'), ('track-2', 'identity-0001'), ('track-3', 'identity-0002');
	`);
	return {
		database,
		close() {
			database.close();
			rmSync(directory, { recursive: true });
		}
	};
}

test('splits a tracklet and merges it into another identity', () => {
	const fixture = databaseFixture();
	try {
		const created = splitTracklet(fixture.database, 'identity-0001', 'track-2');
		assert.equal(created, 'identity-0003');
		assert.equal(
			fixture.database
				.prepare('SELECT identity FROM identity_tracklets WHERE tracklet_id = ?')
				.get('track-2').identity,
			'identity-0003'
		);

		mergeIdentities(fixture.database, 'identity-0003', 'identity-0002');
		assert.equal(
			fixture.database
				.prepare('SELECT identity FROM identity_tracklets WHERE tracklet_id = ?')
				.get('track-2').identity,
			'identity-0002'
		);
		assert.equal(
			fixture.database.prepare("SELECT value FROM control WHERE key = 'revision'").get().value,
			'2'
		);
	} finally {
		fixture.close();
	}
});

test('keeps both identities when conflicting animal numbers prevent a merge', () => {
	const fixture = databaseFixture();
	try {
		fixture.database.exec(
			"UPDATE identities SET animal_number = 'NL-1' WHERE identity = 'identity-0001';" +
				"UPDATE identities SET animal_number = 'NL-2' WHERE identity = 'identity-0002';"
		);
		assert.throws(
			() => mergeIdentities(fixture.database, 'identity-0001', 'identity-0002'),
			/Remove one of the animal numbers/
		);
		assert.equal(
			fixture.database.prepare('SELECT COUNT(*) AS count FROM identities').get().count,
			2
		);
		assert.equal(
			fixture.database.prepare("SELECT value FROM control WHERE key = 'revision'").get().value,
			'0'
		);
	} finally {
		fixture.close();
	}
});

test('requests identity finalization and clears an earlier error', () => {
	const fixture = databaseFixture();
	try {
		fixture.database.exec(`
			INSERT INTO control VALUES ('finalize_requested', '0');
			INSERT INTO control VALUES ('finalize_error', 'Not enough evidence');
		`);

		requestIdentityFinalization(fixture.database);

		assert.equal(
			fixture.database.prepare("SELECT value FROM control WHERE key = 'finalize_requested'").get()
				.value,
			'1'
		);
		assert.equal(
			fixture.database.prepare("SELECT value FROM control WHERE key = 'finalize_error'").get()
				.value,
			''
		);
	} finally {
		fixture.close();
	}
});
