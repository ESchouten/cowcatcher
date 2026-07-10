import { command, query } from '$app/server';
import type { AppConfig, Config } from '$lib/schema';
import { DEFAULT_SCHEMA_URL } from '$lib/schema';
import { readConfigState, replaceConfigState } from '$lib/server/config-store';

async function fetchSchema(schemaUrl: string) {
	const response = await fetch(schemaUrl);
	if (!response.ok) {
		throw new Error(`Failed to load config schema from ${schemaUrl}`);
	}

	return response.json();
}

export const getConfig = query(async (): Promise<{ config: Config; app: AppConfig }> => {
	return readConfigState();
});

export const getConfigSchema = query(async () => {
	const { config } = await readConfigState();
	const schemaUrl = config.$schema ?? DEFAULT_SCHEMA_URL;

	try {
		return await fetchSchema(schemaUrl);
	} catch (error) {
		if (schemaUrl === DEFAULT_SCHEMA_URL) {
			throw error;
		}

		return fetchSchema(DEFAULT_SCHEMA_URL);
	}
});

export const saveConfig = command('unchecked', async ({ config, app }) => {
	await replaceConfigState({ config, app });
});
