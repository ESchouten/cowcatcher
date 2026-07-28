import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import type { AppConfig, Config } from '$lib/schema';
import { APP_CONFIG_PATH, CONFIG_PATH, resolveWithinDirectory } from './shared-paths';

export const IDENTITY_SCHEMA_VERSION = 3;
export const IDENTITY_BUSY_TIMEOUT_MS = 5_000;

export interface IdentityDatabaseConfig {
	id: string;
	database: string;
	label: string;
	path: string;
	targetLabel: string;
}

export async function getIdentityDatabases(): Promise<IdentityDatabaseConfig[]> {
	const [config, app] = await Promise.all([
		readFile(CONFIG_PATH, 'utf8')
			.then((contents) => JSON.parse(contents) as Config)
			.catch(() => ({ detectors: [] }) as Config),
		readFile(APP_CONFIG_PATH, 'utf8')
			.then((contents) => JSON.parse(contents) as AppConfig)
			.catch(() => ({ detectors: [] }) as Partial<AppConfig>)
	]);
	const databases = new Map<string, IdentityDatabaseConfig>();

	for (const [index, detector] of config.detectors.entries()) {
		const identity = detector.identity;
		const database = identity?.database;
		if (!database || path.isAbsolute(database)) continue;
		const resolved = resolveWithinDirectory(path.dirname(CONFIG_PATH), database);
		if (!resolved) continue;
		databases.set(database, {
			id: `catalog_${index + 1}`,
			database,
			label: app.detectors?.[index]?.label ?? `Detector ${index + 1}`,
			path: resolved,
			targetLabel: identity.target_label
		});
	}

	return [...databases.values()];
}

export async function openIdentityDatabase(catalogId: string): Promise<DatabaseSync> {
	const configured = (await getIdentityDatabases()).find((item) => item.id === catalogId);
	if (!configured) {
		throw new Error('Identity catalog is not configured');
	}
	return openConfiguredIdentityDatabase(configured);
}

export async function openConfiguredIdentityDatabase(
	configured: Pick<IdentityDatabaseConfig, 'path'>
): Promise<DatabaseSync> {
	try {
		await access(configured.path);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
			throw new Error('Start the detector to initialize identities.', { cause: error });
		}
		throw error;
	}
	const connection = new DatabaseSync(configured.path);
	try {
		connection.exec(`
			PRAGMA foreign_keys = ON;
			PRAGMA busy_timeout = ${IDENTITY_BUSY_TIMEOUT_MS};
			PRAGMA journal_mode = WAL;
		`);
		validateIdentityDatabase(connection);
		return connection;
	} catch (error) {
		connection.close();
		throw error;
	}
}

export function validateIdentityDatabase(connection: DatabaseSync): void {
	const version = connection.prepare('PRAGMA user_version').get() as
		| { user_version: number }
		| undefined;
	if (version?.user_version !== IDENTITY_SCHEMA_VERSION) {
		throw new Error(
			`Unsupported identity database schema ${version?.user_version ?? 'unknown'}; ` +
				`expected ${IDENTITY_SCHEMA_VERSION}`
		);
	}
}
