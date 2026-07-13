import { describe, expect, it } from 'vitest';
import { tracksBySource } from './stream-tracks';

describe('stream track settings', () => {
	it('maps overlays only to their configured source', async () => {
		const settings = tracksBySource({
			detectors: [
				{
					detection: { source: ['rtsp://camera-1', 'rtsp://camera-2'] },
					exporters: { sse: [{ port: 8765 }] }
				}
			]
		});

		expect(settings).toEqual({
			'rtsp://camera-1': {
				tracksUrl: '/api/tracks/0',
				tracksSource: '0:0'
			},
			'rtsp://camera-2': {
				tracksUrl: '/api/tracks/0',
				tracksSource: '0:1'
			}
		});
		expect(settings['rtsp://unconfigured']).toBeUndefined();
	});
});
