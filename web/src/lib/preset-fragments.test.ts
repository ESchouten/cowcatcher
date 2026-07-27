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
				display: { singular: 'cow', plural: 'cows', official_id_label: 'Cow ID' },
				database: 'identities/cows.sqlite',
				candidate_filter: {
					min_area_ratio: 0.005,
					max_area_ratio: 0.3,
					frame_edge_margin: 0.2
				},
				controlled_zone: {
					zone_id: 'identity_observation',
					x1: 0.2,
					y1: 0.2,
					x2: 0.8,
					y2: 0.8,
					minimum_box_inside_ratio: 0.9,
					minimum_stable_frames: 2,
					clear_frames: 2
				},
				encoder: 'miewid-dual-crop-v1',
				similarity_threshold: 0.75,
				similarity_margin: 0.05,
				query_frames: 2,
				gallery_frames: 4,
				track_max_age: 10
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
