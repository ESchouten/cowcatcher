import { describe, expect, it } from 'vitest';

import { redactCameraSource } from './camera-source';

describe('camera source display', () => {
	it('removes credentials, query parameters, and fragments', () => {
		expect(
			redactCameraSource(
				'rtsp://farmer:secret@192.168.1.42:554/barn/main?token=another-secret#camera'
			)
		).toBe('rtsp://192.168.1.42:554/barn/main');
	});

	it('does not echo malformed or unsupported addresses', () => {
		expect(redactCameraSource('not a URL with a password')).toBe('Camera stream');
		expect(redactCameraSource('https://user:secret@example.com/stream')).toBe('Camera stream');
	});
});
