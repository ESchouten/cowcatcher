import { command, query } from '$app/server';
import {
	APP_CONFIG_PATH,
	CONFIG_PATH,
	saveConfig as saveConfigShared
} from '$lib/server/shared-paths';
import { readFile } from 'node:fs/promises';
import type { AppConfig, Config } from '$lib/schema';
import { DEFAULT_SCHEMA_URL } from '$lib/schema';

async function readConfigDocument(): Promise<Config | null> {
	const config = (await readFile(CONFIG_PATH, 'utf8')
		.then(JSON.parse)
		.catch(() => null)) as Config | null;
	if (!config) {
		return null;
	}
	config.detectors = config.detectors.map((detector) => {
		const source = detector.detection?.source ?? [];
		detector.detection.source = Array.isArray(source) ? source : [source];
		detector.exporters = Object.entries(detector.exporters ?? {}).reduce(
			(acc, [key, value]) => {
				acc[key] = value ? (Array.isArray(value) ? value : [value]) : [];
				return acc;
			},
			{} as Record<string, unknown[]>
		);
		return detector;
	});
	return config;
}

async function fetchSchema(schemaUrl: string) {
	const response = await fetch(schemaUrl);
	if (!response.ok) {
		throw new Error(`Failed to load config schema from ${schemaUrl}`);
	}

	return response.json();
}

export const getConfig = query(async (): Promise<{ config: Config; app: AppConfig }> => {
	const config = await readConfigDocument().then(
		(res) =>
			res ?? {
				$schema: DEFAULT_SCHEMA_URL,
				detectors: []
			}
	);
	const appConfig: AppConfig = await readFile(APP_CONFIG_PATH, 'utf8')
		.then((res) => JSON.parse(res))
		.catch(() => ({
			streams: [],
			telegrams: [],
			detectors: []
		}));

	const detectorLengthDiff = config.detectors.length - appConfig.detectors.length;
	for (let i = 0; i < detectorLengthDiff; i++) {
		appConfig.detectors.push({ label: 'Detector ' + (appConfig.detectors.length + 1) });
	}

	const unknownStreams = config.detectors
		.flatMap((detector) => detector.detection?.source ?? [])
		.filter((source) => !appConfig.streams.some((s) => s.source === source));
	unknownStreams.forEach((stream) => {
		appConfig.streams.push({ label: stream, source: stream });
	});

	const unknownTelegrams = config.detectors
		.flatMap((detector) => detector.exporters?.telegram ?? [])
		.filter(
			(telegram) =>
				!appConfig.telegrams.some((t) => t.token === telegram.token && t.chat === telegram.chat)
		);
	unknownTelegrams.forEach((telegram) => {
		appConfig.telegrams.push({ label: telegram.chat, token: telegram.token, chat: telegram.chat });
	});

	await saveConfigShared({ config, app: appConfig });
	return { config, app: appConfig };
});

export const getConfigSchema = query(async () => {
	const config = await readConfigDocument();
	const schemaUrl = config?.$schema ?? DEFAULT_SCHEMA_URL;

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
	await saveConfigShared({ config, app });
});
