import { readConfigState } from '$lib/server/config-store';
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

function detectorOrigin(port: number): URL {
	const host = process.env.DETECTOR_HOST?.trim() || '127.0.0.1';
	const origin = new URL(`http://${host}`);
	origin.port = String(port);
	return origin;
}

export const GET: RequestHandler = async ({ params, request, fetch }) => {
	const detectorIndex = Number(params.detector);
	if (!Number.isInteger(detectorIndex) || detectorIndex < 0) {
		error(404, 'Detector not found');
	}

	const { config } = await readConfigState();
	const sse = config.detectors[detectorIndex]?.exporters?.sse?.[0];
	if (!sse) {
		error(404, 'Live detections are not configured');
	}

	const endpoint = (sse.endpoint?.trim() || `/events/${detectorIndex}`).replace(/^\/*/, '/');
	const upstreamUrl = detectorOrigin(sse.port ?? 8765);
	upstreamUrl.pathname = endpoint.split('?', 1)[0];

	let upstream: Response;
	try {
		upstream = await fetch(upstreamUrl, {
			headers: { Accept: 'text/event-stream' },
			signal: request.signal
		});
	} catch {
		error(502, 'Detector live stream is unavailable');
	}

	if (!upstream.ok || !upstream.body) {
		error(502, 'Detector live stream is unavailable');
	}

	return new Response(upstream.body, {
		headers: {
			'Cache-Control': 'no-cache, no-transform',
			'Content-Type': 'text/event-stream',
			'X-Accel-Buffering': 'no'
		}
	});
};
