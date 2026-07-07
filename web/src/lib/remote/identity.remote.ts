import { command, query } from '$app/server';
import { error } from '@sveltejs/kit';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import * as v from 'valibot';
import type { DetectorIdentityConfig, IdentityMeta, IdentityProviderConfig } from '$lib/schema';
import { identityProviderConfig } from '$lib/identity-provider';
import { CONFIG_PATH } from '$lib/server/shared-paths';
import { getConfig, getConfigSchema, saveConfig } from './config.remote';

type SQLiteRow = Record<string, unknown>;

export type IdentityRow = {
	identity: string;
	sampleCount: number;
	sampleRows: number;
	lastSampleAt: string | null;
	createdAt: string;
	updatedAt: string;
};

export type IdentityStore = {
	label: string | null;
	providerId: string;
	database: string;
	resolvedDatabase: string;
	model: string | null;
	status: 'ready' | 'missing' | 'error';
	error: string | null;
	metadata: Record<string, string>;
	identityCount: number;
	sampleCount: number;
	unknownCandidateCount: number;
	identities: IdentityRow[];
};

const configDirectory = path.dirname(CONFIG_PATH);
const localIdentityPresetDirectory = path.resolve(process.cwd(), '..', 'config', 'identity');

function normalizeIdentityProvider(identity: unknown): IdentityProviderConfig | null {
	if (!identity || typeof identity !== 'object') {
		return null;
	}

	const value = identity as Partial<IdentityProviderConfig>;
	return typeof value.id === 'string' &&
		value.id.trim().length > 0 &&
		typeof value.database === 'string' &&
		value.database.trim().length > 0
		? (value as IdentityProviderConfig)
		: null;
}

function normalizeIdentityMeta(identity: unknown): IdentityMeta | null {
	const config = normalizeIdentityProvider(identity);
	if (!config || !identity || typeof identity !== 'object') {
		return null;
	}

	const label = (identity as Partial<IdentityMeta>).label;
	return typeof label === 'string' && label.trim().length > 0
		? ({ label, ...config } as IdentityMeta)
		: null;
}

function normalizeDetectorIdentity(identity: unknown): DetectorIdentityConfig | null {
	if (!identity || typeof identity !== 'object') {
		return null;
	}

	const value = identity as Partial<DetectorIdentityConfig>;
	return typeof value.provider === 'string' && value.provider.trim().length > 0
		? (value as DetectorIdentityConfig)
		: null;
}

function identityKey(identity: IdentityProviderConfig) {
	return `${identity.id}\0${identity.database}`;
}

function resolveDatabasePath(database: string) {
	return path.isAbsolute(database) ? database : path.resolve(configDirectory, database);
}

function detectorUsesIdentity(detectorIdentity: unknown, identity: IdentityProviderConfig) {
	const normalized = normalizeDetectorIdentity(detectorIdentity);
	return normalized?.provider === identity.id;
}

function configuredIdentities(
	appIdentities: IdentityMeta[],
	configProviders: unknown[]
): IdentityMeta[] {
	const identities = new Map<string, IdentityMeta>();

	for (const identity of appIdentities) {
		const normalized = normalizeIdentityMeta(identity);
		if (normalized) {
			identities.set(identityKey(normalized), {
				label: normalized.label,
				...identityProviderConfig(normalized)
			});
		}
	}

	for (const provider of configProviders) {
		const normalized = normalizeIdentityProvider(provider);
		if (normalized) {
			const key = identityKey(normalized);
			if (!identities.has(key)) {
				identities.set(key, {
					label: normalized.id,
					...identityProviderConfig(normalized)
				});
			}
		}
	}

	return [...identities.values()];
}

function emptyStore(
	provider: IdentityMeta,
	status: IdentityStore['status'],
	message: string | null = null
): IdentityStore {
	return {
		label: provider.label,
		providerId: provider.id,
		database: provider.database,
		resolvedDatabase: resolveDatabasePath(provider.database),
		model: typeof provider.model === 'string' ? provider.model : null,
		status,
		error: message,
		metadata: {},
		identityCount: 0,
		sampleCount: 0,
		unknownCandidateCount: 0,
		identities: []
	};
}

function readNumber(row: SQLiteRow | undefined, key: string): number {
	const value = row?.[key];
	return typeof value === 'number' ? value : Number(value ?? 0);
}

function readString(row: SQLiteRow, key: string): string {
	const value = row[key];
	return typeof value === 'string' ? value : '';
}

function readNullableString(row: SQLiteRow, key: string): string | null {
	const value = row[key];
	return typeof value === 'string' && value.length > 0 ? value : null;
}

function hasTable(database: DatabaseSync, table: string) {
	return Boolean(
		database.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?").get(table)
	);
}

function ensureIdentitySchema(database: DatabaseSync) {
	for (const table of ['metadata', 'identities', 'samples', 'unknown_candidates']) {
		if (!hasTable(database, table)) {
			throw new Error(`Missing ${table} table`);
		}
	}
}

function readStore(provider: IdentityMeta): IdentityStore {
	const databasePath = resolveDatabasePath(provider.database);

	if (!existsSync(databasePath)) {
		return emptyStore(provider, 'missing', 'Database file not found');
	}

	try {
		if (!statSync(databasePath).isFile()) {
			return emptyStore(provider, 'error', 'Configured database path is not a file');
		}

		const database = new DatabaseSync(databasePath, { readOnly: true, timeout: 5000 });
		try {
			ensureIdentitySchema(database);

			const metadata = Object.fromEntries(
				database
					.prepare('SELECT key, value FROM metadata ORDER BY key')
					.all()
					.map((row) => [readString(row, 'key'), readString(row, 'value')])
			);
			const sampleCount = readNumber(
				database.prepare('SELECT COUNT(*) AS count FROM samples').get(),
				'count'
			);
			const unknownCandidateCount = readNumber(
				database.prepare('SELECT COUNT(*) AS count FROM unknown_candidates').get(),
				'count'
			);
			const identities = database
				.prepare(
					`
					SELECT
						identities.identity AS identity,
						identities.sample_count AS sampleCount,
						identities.created_at AS createdAt,
						identities.updated_at AS updatedAt,
						COUNT(samples.sample_id) AS sampleRows,
						MAX(samples.created_at) AS lastSampleAt
					FROM identities
					LEFT JOIN samples ON samples.identity = identities.identity
					GROUP BY identities.identity
					ORDER BY identities.identity
					`
				)
				.all()
				.map((row) => ({
					identity: readString(row, 'identity'),
					sampleCount: readNumber(row, 'sampleCount'),
					sampleRows: readNumber(row, 'sampleRows'),
					lastSampleAt: readNullableString(row, 'lastSampleAt'),
					createdAt: readString(row, 'createdAt'),
					updatedAt: readString(row, 'updatedAt')
				}));

			return {
				label: provider.label,
				providerId: provider.id,
				database: provider.database,
				resolvedDatabase: databasePath,
				model: typeof provider.model === 'string' ? provider.model : null,
				status: 'ready',
				error: null,
				metadata,
				identityCount: identities.length,
				sampleCount,
				unknownCandidateCount,
				identities
			};
		} finally {
			database.close();
		}
	} catch (readError) {
		return emptyStore(
			provider,
			'error',
			readError instanceof Error ? readError.message : 'Database could not be read'
		);
	}
}

function localIsoSeconds() {
	const date = new Date();
	const offset = date.getTimezoneOffset() * 60_000;
	return new Date(date.getTime() - offset).toISOString().slice(0, 19);
}

async function readLocalPreset(file: string): Promise<IdentityProviderConfig | null> {
	const resolvedPath = path.resolve(localIdentityPresetDirectory, file);
	const relativePath = path.relative(localIdentityPresetDirectory, resolvedPath);
	if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
		error(400, 'Invalid identity preset path');
	}

	return readFile(resolvedPath, 'utf8')
		.then((contents) => JSON.parse(contents) as IdentityProviderConfig)
		.catch(() => null);
}

async function findIdentityProvider(providerId: string, databasePath: string) {
	const { config, app } = await getConfig();
	const provider = configuredIdentities(app.identities, config.identity?.providers ?? []).find(
		(provider) => provider.id === providerId && provider.database === databasePath
	);

	if (!provider) {
		error(404, 'Identity provider not found in config');
	}

	return provider;
}

export const getIdentities = query(async (): Promise<IdentityMeta[]> => {
	const { config, app } = await getConfig();
	return configuredIdentities(app.identities, config.identity?.providers ?? []);
});

export const getIdentity = query(
	v.object({
		label: v.string()
	}),
	async ({ label }) => {
		const identities = await getIdentities();
		return identities.find((identity) => identity.label === label);
	}
);

export const getIdentityPresets = query(async () => {
	if (existsSync(localIdentityPresetDirectory)) {
		return readdirSync(localIdentityPresetDirectory)
			.filter((file) => file.endsWith('.json'))
			.sort();
	}

	const response = await fetch(
		'https://api.github.com/repos/ESchouten/ai-detector/contents/config/identity',
		{
			headers: {
				Accept: 'application/vnd.github+json',
				'User-Agent': 'ai-detector-web'
			}
		}
	);
	if (!response.ok) {
		return [];
	}

	const items: { name: string }[] = await response.json();
	return items.map((item) => item.name).filter((file) => file.endsWith('.json'));
});

export const getIdentityPreset = query(
	v.object({
		file: v.string()
	}),
	async ({ file }): Promise<IdentityProviderConfig> => {
		const localPreset = await readLocalPreset(file);
		if (localPreset) {
			return localPreset;
		}

		const response = await fetch(
			`https://raw.githubusercontent.com/ESchouten/ai-detector/main/config/identity/${file}`
		);
		if (!response.ok) {
			error(404, 'Identity preset not found');
		}
		return response.json();
	}
);

export const getIdentitySchema = query(async () => {
	const configSchema = await getConfigSchema();

	return {
		$defs: configSchema.$defs,
		...(configSchema.$defs.IdentityProviderConfig as Record<string, unknown>)
	};
});

export const saveIdentityConfig = command(
	v.object({
		original: v.optional(v.string()),
		identity: v.any(),
		meta: v.object({
			label: v.pipe(v.string(), v.trim(), v.minLength(1))
		})
	}),
	async ({ original, identity, meta }) => {
		const normalized = normalizeIdentityProvider(identity);
		if (!normalized) {
			error(400, 'Identity must have an id and database');
		}

		const { config, app } = await getConfig();
		const labelConflict = app.identities.find(
			(identity) => identity.label === meta.label && identity.label !== original
		);
		if (labelConflict) {
			error(409, 'Identity label already exists');
		}

		const idConflict = app.identities.find(
			(identity) => identity.id === normalized.id && identity.label !== original
		);
		if (idConflict) {
			error(409, 'Identity id already exists');
		}

		const provider = identityProviderConfig(normalized);
		const nextIdentity: IdentityMeta = { label: meta.label, ...provider };
		const index = original
			? app.identities.findIndex((identity) => identity.label === original)
			: -1;
		const previousIdentity = index >= 0 ? app.identities[index] : null;

		if (index >= 0) {
			app.identities[index] = nextIdentity;
		} else {
			app.identities.push(nextIdentity);
		}

		config.identity ??= { providers: [] };
		config.identity.providers ??= [];
		const providerIndex = config.identity.providers.findIndex(
			(provider) => provider.id === (previousIdentity?.id ?? normalized.id)
		);
		if (providerIndex >= 0) {
			config.identity.providers[providerIndex] = provider;
		} else {
			config.identity.providers.push(provider);
		}

		if (previousIdentity) {
			for (const detector of config.detectors) {
				if (detectorUsesIdentity(detector.identity, previousIdentity)) {
					const detectorIdentity = normalizeDetectorIdentity(detector.identity);
					detector.identity = {
						...(detectorIdentity ?? { provider: normalized.id }),
						provider: normalized.id
					};
				}
			}
		}

		await saveConfig({ config, app });
	}
);

export const deleteIdentityConfig = command(
	v.object({
		label: v.string()
	}),
	async ({ label }) => {
		const { config, app } = await getConfig();
		const identity = app.identities.find((identity) => identity.label === label);
		app.identities = app.identities.filter((identity) => identity.label !== label);

		if (identity) {
			if (config.identity) {
				config.identity.providers = config.identity.providers.filter(
					(provider) => provider.id !== identity.id || provider.database !== identity.database
				);
				if (config.identity.providers.length === 0) {
					delete config.identity;
				}
			}

			for (const detector of config.detectors) {
				if (detectorUsesIdentity(detector.identity, identity)) {
					delete detector.identity;
				}
			}
		}

		await saveConfig({ config, app });
	}
);

export const getIdentityStores = query(async (): Promise<IdentityStore[]> => {
	const { config, app } = await getConfig();
	return configuredIdentities(app.identities, config.identity?.providers ?? []).map(readStore);
});

export const renameIdentity = command(
	v.object({
		providerId: v.string(),
		database: v.string(),
		identity: v.string(),
		nextIdentity: v.pipe(v.string(), v.trim(), v.minLength(1))
	}),
	async ({ providerId, database, identity, nextIdentity }) => {
		if (nextIdentity === identity) {
			return;
		}

		const provider = await findIdentityProvider(providerId, database);
		const databasePath = resolveDatabasePath(provider.database);

		if (!existsSync(databasePath)) {
			error(404, 'Identity database not found');
		}

		const sqlite = new DatabaseSync(databasePath, { timeout: 5000 });
		try {
			ensureIdentitySchema(sqlite);

			const current = sqlite
				.prepare('SELECT identity FROM identities WHERE identity = ?')
				.get(identity);
			if (!current) {
				error(404, 'Identity not found');
			}

			const conflict = sqlite
				.prepare('SELECT identity FROM identities WHERE identity = ?')
				.get(nextIdentity);
			if (conflict) {
				error(409, 'Identity already exists');
			}

			sqlite.exec('BEGIN IMMEDIATE');
			try {
				sqlite
					.prepare(
						`
						UPDATE identities
						SET identity = ?, updated_at = ?
						WHERE identity = ?
						`
					)
					.run(nextIdentity, localIsoSeconds(), identity);
				sqlite
					.prepare('UPDATE samples SET identity = ? WHERE identity = ?')
					.run(nextIdentity, identity);
				sqlite.exec('COMMIT');
			} catch (transactionError) {
				sqlite.exec('ROLLBACK');
				throw transactionError;
			}
		} finally {
			sqlite.close();
		}
	}
);
