import { command, form, query } from '$app/server';
import { getConfig, saveConfig } from './config.remote';
import type { StreamMeta } from '$lib/schema';
import * as v from 'valibot';
import { redirect } from '@sveltejs/kit';

function getRedirectTarget(next: string | undefined, fallback: string) {
	return next?.startsWith('/') && !next.startsWith('//') ? next : fallback;
}

function defaultSseEndpoint(detectorIndex: number) {
	return `/events/${detectorIndex}`;
}

function getSseEndpoint(endpoint: string | null | undefined, detectorIndex: number) {
	const configured = endpoint?.trim();
	return configured || defaultSseEndpoint(detectorIndex);
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
	const tracksBySource: Record<string, { tracksEndpoint: string; tracksPort: number }> = {};
	let fallback = { tracksEndpoint: defaultSseEndpoint(0), tracksPort: 8765 };

	config.detectors.forEach((detector, detectorIndex) => {
		const sse = detector.exporters?.sse?.[0];
		if (!sse) {
			return;
		}

		const tracksEndpoint = getSseEndpoint(sse.endpoint, detectorIndex);
		const tracksPort = sse.port ?? 8765;
		if (!Object.keys(tracksBySource).length) {
			fallback = { tracksEndpoint, tracksPort };
		}

		detector.detection.source.forEach((source) => {
			tracksBySource[source] = { tracksEndpoint, tracksPort };
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
