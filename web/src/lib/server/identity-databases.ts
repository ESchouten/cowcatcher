import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import type { AppConfig, Config } from '$lib/schema';
import { APP_CONFIG_PATH, CONFIG_PATH, resolveWithinDirectory } from './shared-paths';

export const IDENTITY_SCHEMA_VERSION = 2;
export const IDENTITY_BUSY_TIMEOUT_MS = 5_000;
export const IDENTITY_TABLES = [
	'audit_events',
	'control',
	'evidence_frames',
	'gallery_items',
	'gallery_versions',
	'mappings',
	'official_identities',
	'tracklets',
	'visual_identities',
	'visual_identity_tracklets'
] as const;

export interface IdentityDatabaseConfig {
	id: string;
	database: string;
	label: string;
	path: string;
	targetLabel: string;
	galleryFrames: number;
	display: {
		singular: string;
		plural: string;
		officialIdLabel: string;
	};
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
			targetLabel: identity.target_label,
			galleryFrames: identity.gallery_frames,
			display: {
				singular: identity.display.singular,
				plural: identity.display.plural,
				officialIdLabel: identity.display.official_id_label
			}
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
	const rows = connection
		.prepare(
			`SELECT name
			 FROM sqlite_schema
			 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'`
		)
		.all() as Array<{ name: string }>;
	const tables = new Set(rows.map(({ name }) => name));
	const missing = IDENTITY_TABLES.filter((table) => !tables.has(table));
	if (missing.length) {
		throw new Error(`Identity database is missing required tables: ${missing.join(', ')}`);
	}
	const control = connection
		.prepare('SELECT schema_version FROM control WHERE singleton = 1')
		.get() as { schema_version: number } | undefined;
	if (control?.schema_version !== IDENTITY_SCHEMA_VERSION) {
		throw new Error('Identity database control record does not match its schema');
	}
	const foreignKeyViolation = connection.prepare('PRAGMA foreign_key_check').get();
	if (foreignKeyViolation) {
		throw new Error('Identity database contains foreign-key violations');
	}
}
