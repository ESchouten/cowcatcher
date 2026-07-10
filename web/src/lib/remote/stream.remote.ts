import { command, form, query } from '$app/server';
import { getConfig } from './config.remote';
import type { StreamMeta } from '$lib/schema';
import { updateConfigState } from '$lib/server/config-store';
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
	const tracksBySource: Record<string, { tracksUrl: string; tracksSource: string }> = {};
	let fallback = {
		tracksUrl: '/api/tracks/0',
		tracksSource: '0:0'
	};

	config.detectors.forEach((detector, detectorIndex) => {
		const sse = detector.exporters?.sse?.[0];
		if (!sse) {
			return;
		}

		const tracksUrl = `/api/tracks/${detectorIndex}`;
		if (!Object.keys(tracksBySource).length) {
			fallback = { tracksUrl, tracksSource: `${detectorIndex}:0` };
		}

		detector.detection.source.forEach((source, sourceIndex) => {
			tracksBySource[source] = {
				tracksUrl,
				tracksSource: `${detectorIndex}:${sourceIndex}`
			};
		});
	});

	return {
		...fallback,
		tracksBySource
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
		await updateConfigState(({ config, app }) => {
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
				detector.detection.source = detector.detection.source.map((item) =>
					item === original ? source : item
				);
			});
		});
		redirect(302, getRedirectTarget(next, '/streams'));
	}
);

export const deleteStream = command(
	v.object({
		source: v.string()
	}),
	async ({ source }) => {
		await updateConfigState(({ config, app }) => {
			app.streams = app.streams.filter((stream) => stream.source !== source);
			config.detectors.forEach((detector) => {
				detector.detection.source = detector.detection.source.filter((item) => item !== source);
			});
		});
	}
);

export const reorderStream = command(
	v.object({
		index0: v.number(),
		index1: v.number()
	}),
	async ({ index0, index1 }) => {
		await updateConfigState(({ app }) => {
			const [stream] = app.streams.splice(index0, 1);
			if (stream) {
				app.streams.splice(index1, 0, stream);
			}
		});
	}
);
