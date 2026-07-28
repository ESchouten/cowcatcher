import { describe, expect, it } from 'vitest';

import type { DetectorConfig } from '$lib/schema';
import { applyDetectorPreset } from './preset-fragments';

describe('detector preset fragments', () => {
	it('copies detector defaults without changing sources, identity, or exporters', () => {
		const detector = {
			detection: { source: ['rtsp://barn'], interval: 7 },
			yolo: { model: 'old.pt', tracking: false, confidence: 0.8 },
			identity: {
				target_label: 'cow',
				database: 'identities/cows.sqlite',
				margin: 0.2
			},
			exporters: { sse: [{ port: 8765 }] }
		} satisfies DetectorConfig;
		const identity = detector.identity;
		const exporters = detector.exporters;

		const resolved = applyDetectorPreset(detector, {
			detection: { interval: 1 },
			yolo: {
				model: 'yolo26m-seg.pt',
				task: 'segment',
				tracking: true,
				confidence: { cow: 0.1 },
				imgsz: 640,
				iou: 0.5
			}
		});

		expect(resolved.detection).toEqual({ source: ['rtsp://barn'], interval: 1 });
		expect(resolved.yolo?.model).toBe('yolo26m-seg.pt');
		expect(resolved.identity).toBe(identity);
		expect(resolved.exporters).toBe(exporters);
	});
});
