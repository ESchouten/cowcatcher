import { describe, expect, test } from 'vitest';
import { addCredentialsToStream, describeDiscoveredCamera } from './onvif';

describe('ONVIF camera normalization', () => {
	test('creates an opaque discovery result without exposing the service path', () => {
		const result = describeDiscoveredCamera(
			{
				hostname: '192.168.1.42',
				port: '8080',
				path: '/onvif/device_service',
				urn: 'urn:uuid:camera'
			},
			1_000
		);

		expect(result).toMatchObject({
			name: 'Camera at 192.168.1.42',
			host: '192.168.1.42',
			port: 8080,
			secure: false
		});
		expect(result.id).toMatch(/^[0-9a-f-]{36}$/);
		expect(result).not.toHaveProperty('path');
		expect(result).not.toHaveProperty('urn');
	});

	test('embeds escaped credentials and repairs loopback stream hosts', () => {
		expect(
			addCredentialsToStream(
				'rtsp://127.0.0.1:554/live/main',
				'192.168.1.42',
				'farm user',
				'p@ss/word'
			)
		).toBe('rtsp://farm%20user:p%40ss%2Fword@192.168.1.42:554/live/main');
	});

	test('rejects non-RTSP camera responses', () => {
		expect(() =>
			addCredentialsToStream('http://192.168.1.42/video', '192.168.1.42', '', '')
		).toThrow('Unsupported camera stream protocol');
	});
});
