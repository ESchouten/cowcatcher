import type { AppConfig, Config, DetectorConfig } from '$lib/schema';
import { DEFAULT_SCHEMA_URL } from '$lib/schema';
import type {
	Config as ConfigDocument,
	DetectorConfig as DetectorDocument
} from '$lib/generated/config';
import { APP_CONFIG_PATH, CONFIG_PATH } from '$lib/server/shared-paths';
import { randomUUID } from 'node:crypto';
import { readFile, rename, rm, writeFile } from 'node:fs/promises';

export interface ConfigState {
	config: Config;
	app: AppConfig;
}

let updateTail: Promise<void> = Promise.resolve();

async function readJsonObject<T>(filePath: string): Promise<T | null> {
	try {
		const value: unknown = JSON.parse(await readFile(filePath, 'utf8'));
		if (value === null || typeof value !== 'object' || Array.isArray(value)) {
			throw new TypeError(`${filePath} must contain a JSON object`);
		}
		return value as T;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
			return null;
		}
		throw error;
	}
}

function normalizeDetector(detector: DetectorDocument): DetectorConfig {
	const source = detector.detection?.source ?? [];
	const exporters = Object.fromEntries(
		Object.entries(detector.exporters ?? {}).map(([key, value]) => [
			key,
			value ? (Array.isArray(value) ? value : [value]) : []
		])
	) as DetectorConfig['exporters'];

	return {
		...detector,
		detection: {
			...detector.detection,
			source: Array.isArray(source) ? source : [source]
		},
		exporters
	};
}

function reconcileAppConfig(config: Config, app: AppConfig): AppConfig {
	const reconciled: AppConfig = {
		streams: [...app.streams],
		telegrams: [...app.telegrams],
		detectors: app.detectors.slice(0, config.detectors.length)
	};

	while (reconciled.detectors.length < config.detectors.length) {
		reconciled.detectors.push({ label: `Detector ${reconciled.detectors.length + 1}` });
	}

	for (const source of config.detectors.flatMap((detector) => detector.detection.source)) {
		if (!reconciled.streams.some((stream) => stream.source === source)) {
			reconciled.streams.push({ label: source, source });
		}
	}

	for (const telegram of config.detectors.flatMap(
		(detector) => detector.exporters?.telegram ?? []
	)) {
		if (
			!reconciled.telegrams.some(
				(item) => item.token === telegram.token && item.chat === telegram.chat
			)
		) {
			reconciled.telegrams.push({
				label: telegram.chat,
				token: telegram.token,
				chat: telegram.chat
			});
		}
	}

	return reconciled;
}

async function readState(): Promise<ConfigState> {
	const document = await readJsonObject<ConfigDocument & { $schema?: string }>(CONFIG_PATH);
	const config: Config = document
		? { ...document, detectors: document.detectors.map(normalizeDetector) }
		: { $schema: DEFAULT_SCHEMA_URL, detectors: [] };
	const app =
		(await readJsonObject<AppConfig>(APP_CONFIG_PATH)) ??
		({ streams: [], telegrams: [], detectors: [] } satisfies AppConfig);

	return { config, app: reconcileAppConfig(config, app) };
}

async function writeJsonAtomic(filePath: string, value: unknown): Promise<void> {
	const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
	try {
		await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`);
		await rename(temporaryPath, filePath);
	} finally {
		await rm(temporaryPath, { force: true });
	}
}

async function writeState({ config, app }: ConfigState): Promise<void> {
	await Promise.all([writeJsonAtomic(CONFIG_PATH, config), writeJsonAtomic(APP_CONFIG_PATH, app)]);
}

function enqueue<T>(operation: () => Promise<T>): Promise<T> {
	const result = updateTail.then(operation);
	updateTail = result.then(
		() => undefined,
		() => undefined
	);
	return result;
}

export function readConfigState(): Promise<ConfigState> {
	return enqueue(readState);
}

export function updateConfigState<T>(mutate: (state: ConfigState) => T | Promise<T>): Promise<T> {
	return enqueue(async () => {
		const state = await readState();
		const result = await mutate(state);
		await writeState(state);
		return result;
	});
}

export function replaceConfigState(state: ConfigState): Promise<void> {
	return enqueue(() => writeState(state));
}
