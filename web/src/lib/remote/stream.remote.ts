import { command, form, query } from '$app/server';
import { getConfig, saveConfig } from './config.remote';
import type { StreamMeta } from '$lib/schema';
import * as v from 'valibot';
import { redirect } from '@sveltejs/kit';

function getRedirectTarget(next: string | undefined, fallback: string) {
	return next?.startsWith('/') && !next.startsWith('//') ? next : fallback;
}

export const getStreams = query(async () => {
	const { config, app } = await getConfig();
	const detectorSources = config.detectors.flatMap((detector) => detector.detection.source);
	const detectorStreams = detectorSources.filter((source) => source.trim().match(/rtsps?:\/\//i));
	const allStreams = [
		...new Set([...app.streams, ...detectorStreams.map((source) => ({ source }) as StreamMeta)])
	];
	const uniqueStreams = allStreams.filter(
		(stream, index) => allStreams.findIndex((s) => s.source === stream.source) === index
	);

	return uniqueStreams.map((stream, index) => ({
		source: stream.source,
		label: stream.label ?? 'Stream ' + (index + 1)
	}));
});

export const getStreamSettings = query(async () => {
	const { config } = await getConfig();
	const sseExporters = config.detectors.flatMap((detector) => detector.exporters?.sse ?? []);
	const sse = sseExporters[0];
	const endpoint = sse?.endpoint?.trim() || '/events';

	return {
		tracksEndpoint: endpoint,
		tracksPort: sse?.port ?? 8765
	};
});

export const saveStream = form(
	v.object({
		original: v.optional(v.string()),
		label: v.string(),
		source: v.string(),
		next: v.optional(v.string())
	}),
	async ({ source, label, original, next }) => {
		const { config, app } = await getConfig();
		let found = false;
		app.streams.forEach((stream) => {
			if (stream.source === original) {
				stream.label = label;
				stream.source = source;
				found = true;
			}
		});
		if (!found) {
			app.streams.push({ source, label });
		}
		config.detectors.forEach((detector) => {
			detector.detection.source = detector.detection.source.map((s) =>
				s === original ? source : s
			);
		});
		await saveConfig({ config, app });
		redirect(302, getRedirectTarget(next, '/streams'));
	}
);

export const deleteStream = command(
	v.object({
		source: v.string()
	}),
	async ({ source }) => {
		const { config, app } = await getConfig();
		app.streams = app.streams.filter((stream) => stream.source !== source);
		config.detectors.forEach((detector) => {
			detector.detection.source = detector.detection.source.filter((s) => s !== source);
		});
		await saveConfig({ config, app });
	}
);

export const reorderStream = command(
	v.object({
		index0: v.number(),
		index1: v.number()
	}),
	async ({ index0, index1 }) => {
		const { config, app } = await getConfig();

		const [stream] = app.streams.splice(index0, 1);
		if (stream) {
			app.streams.splice(index1, 0, stream);
		}

		await saveConfig({ config, app });
	}
);
