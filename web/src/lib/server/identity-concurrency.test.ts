import { once } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn, spawnSync, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { DatabaseSync } from 'node:sqlite';
import { afterEach, describe, expect, it } from 'vitest';

import { createOfficialIdentity, readCatalogControl } from './identity-catalog';
import { openConfiguredIdentityDatabase } from './identity-databases';

const SCHEMA_PATH = resolve(process.cwd(), '../detector/src/aidetector/reid/schema.sql');
const temporaryDirectories: string[] = [];
const openDatabases: DatabaseSync[] = [];
const runningChildren: ChildProcessWithoutNullStreams[] = [];

function pythonInitializedCatalog(): string {
	const directory = mkdtempSync(join(tmpdir(), 'identity-python-node-'));
	temporaryDirectories.push(directory);
	const databasePath = join(directory, 'identity.sqlite');
	const result = spawnSync(
		'python3',
		[
			'-c',
			[
				'import pathlib, sqlite3, sys',
				'connection = sqlite3.connect(sys.argv[1], isolation_level=None)',
				"connection.execute('PRAGMA journal_mode = WAL')",
				"connection.execute('PRAGMA foreign_keys = ON')",
				'connection.executescript(pathlib.Path(sys.argv[2]).read_text())',
				'connection.close()'
			].join('\n'),
			databasePath,
			SCHEMA_PATH
		],
		{ encoding: 'utf8' }
	);
	if (result.status !== 0) {
		throw new Error(`Python catalog initialization failed: ${result.stderr}`);
	}
	return databasePath;
}

function waitForOutput(child: ChildProcessWithoutNullStreams, marker: string): Promise<void> {
	return new Promise((resolveReady, reject) => {
		let output = '';
		const handleData = (chunk: Buffer) => {
			output += chunk.toString('utf8');
			if (!output.includes(marker)) return;
			child.stdout.off('data', handleData);
			child.off('error', handleError);
			resolveReady();
		};
		const handleError = (error: Error) => {
			child.stdout.off('data', handleData);
			reject(error);
		};
		child.stdout.on('data', handleData);
		child.once('error', handleError);
	});
}

afterEach(() => {
	while (openDatabases.length) openDatabases.pop()?.close();
	while (runningChildren.length) {
		const child = runningChildren.pop();
		if (child && child.exitCode === null) child.kill();
	}
	while (temporaryDirectories.length) {
		rmSync(temporaryDirectories.pop()!, { recursive: true, force: true });
	}
});

describe('simultaneous Python and Node SQLite ownership', () => {
	it('keeps a Python WAL connection open while Node commits an audited operator change', async () => {
		const databasePath = pythonInitializedCatalog();
		const python = spawn(
			'python3',
			[
				'-u',
				'-c',
				[
					'import sqlite3, sys',
					'connection = sqlite3.connect(sys.argv[1], timeout=5, isolation_level=None)',
					"connection.execute('PRAGMA journal_mode = WAL')",
					"connection.execute('PRAGMA foreign_keys = ON')",
					"connection.execute('PRAGMA busy_timeout = 5000')",
					"print('PYTHON_READY', flush=True)",
					'sys.stdin.readline()',
					'connection.close()'
				].join('\n'),
				databasePath
			],
			{ stdio: ['pipe', 'pipe', 'pipe'] }
		);
		runningChildren.push(python);
		const exited = once(python, 'exit');
		await waitForOutput(python, 'PYTHON_READY');

		const database = await openConfiguredIdentityDatabase({ path: databasePath });
		openDatabases.push(database);
		createOfficialIdentity(
			database,
			{ officialId: 'NL-CONCURRENT', displayName: 'Concurrent write' },
			{ expectedRevision: 0 }
		);
		expect(readCatalogControl(database).operatorRevision).toBe(1);
		expect(
			(
				database
					.prepare(`SELECT event_type FROM audit_events WHERE operator_revision = 1`)
					.get() as { event_type: string }
			).event_type
		).toBe('official_identity_created');

		python.stdin.end('\n');
		const [exitCode] = await exited;
		expect(exitCode).toBe(0);
	});

	it('rolls back an uncommitted Python process crash before Node continues', async () => {
		const databasePath = pythonInitializedCatalog();
		const database = await openConfiguredIdentityDatabase({ path: databasePath });
		openDatabases.push(database);
		const python = spawn(
			'python3',
			[
				'-u',
				'-c',
				[
					'import os, sqlite3, sys',
					'connection = sqlite3.connect(sys.argv[1], timeout=5, isolation_level=None)',
					"connection.execute('PRAGMA journal_mode = WAL')",
					"connection.execute('PRAGMA busy_timeout = 5000')",
					"connection.execute('BEGIN IMMEDIATE')",
					"connection.execute('UPDATE control SET runtime_revision = runtime_revision + 1')",
					"print('UNCOMMITTED_WRITE', flush=True)",
					'os._exit(19)'
				].join('\n'),
				databasePath
			],
			{ stdio: ['pipe', 'pipe', 'pipe'] }
		);
		runningChildren.push(python);
		const exited = once(python, 'exit');
		await waitForOutput(python, 'UNCOMMITTED_WRITE');
		const [exitCode] = await exited;
		expect(exitCode).toBe(19);

		expect(readCatalogControl(database).runtimeRevision).toBe(0);
		createOfficialIdentity(database, { officialId: 'NL-AFTER-CRASH' }, { expectedRevision: 0 });
		expect(readCatalogControl(database).operatorRevision).toBe(1);
	});
});
