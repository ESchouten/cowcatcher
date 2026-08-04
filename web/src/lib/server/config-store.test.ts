import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const paths = vi.hoisted(() => {
	const directory = `/tmp/ai-detector-config-store-${process.pid}`;
	return {
		directory,
		CONFIG_PATH: `${directory}/config.json`,
		APP_CONFIG_PATH: `${directory}/app.json`
	};
});

vi.mock('$lib/server/shared-paths', () => ({
	CONFIG_PATH: paths.CONFIG_PATH,
	APP_CONFIG_PATH: paths.APP_CONFIG_PATH
}));

import { readConfigState, updateConfigState } from './config-store';

async function writeJson(filePath: string, value: unknown) {
	await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

beforeEach(async () => {
	await rm(paths.directory, { recursive: true, force: true });
	await mkdir(paths.directory, { recursive: true });
});

describe('config store', () => {
	it('normalizes reads without writing either document', async () => {
		const original = {
			$schema: 'schema.json',
			detectors: [{ detection: { source: 'rtsp://camera' } }]
		};
		await writeJson(paths.CONFIG_PATH, original);

		const state = await readConfigState();

		expect(state.config.detectors[0]?.detection.source).toEqual(['rtsp://camera']);
		expect(state.app.streams).toEqual([{ label: 'rtsp://camera', source: 'rtsp://camera' }]);
		expect(JSON.parse(await readFile(paths.CONFIG_PATH, 'utf8'))).toEqual(original);
		await expect(readFile(paths.APP_CONFIG_PATH, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' });
	});

	it('serializes concurrent read-modify-write updates', async () => {
		await writeJson(paths.CONFIG_PATH, { detectors: [] });
		await writeJson(paths.APP_CONFIG_PATH, { streams: [], telegrams: [], detectors: [] });

		const first = updateConfigState(async ({ app }) => {
			await new Promise((resolve) => setTimeout(resolve, 20));
			app.streams.push({ label: 'First', source: 'first' });
		});
		const second = updateConfigState(({ app }) => {
			app.streams.push({ label: 'Second', source: 'second' });
		});
		await Promise.all([first, second]);

		const saved = JSON.parse(await readFile(paths.APP_CONFIG_PATH, 'utf8'));
		expect(saved.streams).toEqual([
			{ label: 'First', source: 'first' },
			{ label: 'Second', source: 'second' }
		]);
	});

	it('does not replace malformed configuration', async () => {
		await writeFile(paths.CONFIG_PATH, '{broken');

		await expect(readConfigState()).rejects.toBeInstanceOf(SyntaxError);
		expect(await readFile(paths.CONFIG_PATH, 'utf8')).toBe('{broken');
	});
});
