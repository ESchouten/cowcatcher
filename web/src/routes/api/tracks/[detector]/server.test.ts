import { afterEach, describe, expect, it, vi } from 'vitest';

const readConfigState = vi.hoisted(() => vi.fn());

vi.mock('$lib/server/config-store', () => ({ readConfigState }));

import { GET } from './+server';

afterEach(() => {
	delete process.env.DETECTOR_HOST;
	vi.clearAllMocks();
});

describe('tracks SSE proxy', () => {
	it('streams the configured detector endpoint through the web origin', async () => {
		process.env.DETECTOR_HOST = 'detector.internal';
		readConfigState.mockResolvedValue({
			config: {
				detectors: [
					{
						detection: { source: ['camera'] },
						exporters: { sse: [{ port: 9000, endpoint: '/custom-events' }] }
					}
				]
			},
			app: { streams: [], telegrams: [], detectors: [] }
		});
		const upstreamFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
			void input;
			void init;
			return new Response('event: tracks\ndata: {}\n\n', {
				headers: { 'Content-Type': 'text/event-stream' }
			});
		});

		const response = await GET({
			params: { detector: '0' },
			request: new Request('http://web.test/api/tracks/0'),
			fetch: upstreamFetch
		} as never);

		expect(upstreamFetch).toHaveBeenCalledOnce();
		expect(String(upstreamFetch.mock.calls[0]?.[0])).toBe(
			'http://detector.internal:9000/custom-events'
		);
		expect(response.headers.get('Content-Type')).toBe('text/event-stream');
		expect(await response.text()).toContain('event: tracks');
	});
});
