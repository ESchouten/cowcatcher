import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import type { AppConfig, Config } from '$lib/schema';
import { APP_CONFIG_PATH, CONFIG_PATH, DATA_DIR, resolveWithinDirectory } from './shared-paths';

export interface IdentityDatabaseConfig {
	database: string;
	label: string;
	path: string;
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
		const database = detector.dazzlecow?.enrollment?.database;
		if (!database || path.isAbsolute(database)) continue;
		const resolved = resolveWithinDirectory(DATA_DIR, database);
		if (!resolved) continue;
		databases.set(database, {
			database,
			label: app.detectors?.[index]?.label ?? `Detector ${index + 1}`,
			path: resolved
		});
	}

	return [...databases.values()];
}

export async function openIdentityDatabase(database: string): Promise<DatabaseSync> {
	const configured = (await getIdentityDatabases()).find((item) => item.database === database);
	if (!configured) {
		throw new Error('Identity database is not configured');
	}
	await access(configured.path);
	return new DatabaseSync(configured.path);
}
