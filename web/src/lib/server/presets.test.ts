import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';

import { decodePresetBlob, validatePreset, validateResolvedDetector } from './presets';

const identity = {
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
};

describe('remote preset validation', () => {
	it('accepts typed detector fragments that omit runtime sources', () => {
		expect(() =>
			validatePreset('detector', {
				detection: { interval: 1 },
				yolo: {
					model: 'yolo26m-seg.pt',
					task: 'segment',
					tracking: true,
					confidence: { cow: 0.1 },
					imgsz: 640,
					iou: 0.5
				}
			})
		).not.toThrow();
	});

	it('rejects unknown, incomplete, and semantically changed identity policy', () => {
		expect(() => validatePreset('identity', { ...identity, hidden_default: 1 })).toThrow(
			'unknown property'
		);
		const missing = Object.fromEntries(
			Object.entries(identity).filter(([key]) => key !== 'similarity_margin')
		);
		expect(() => validatePreset('identity', missing)).toThrow('required property');
		expect(() => validatePreset('identity', { ...identity, query_frames: 3 })).toThrow(
			'requires two query'
		);
		expect(() =>
			validatePreset('identity', {
				...identity,
				controlled_zone: { ...identity.controlled_zone, x2: 0.2 }
			})
		).toThrow('positive extent');
	});

	it('verifies and returns the GitHub blob SHA with the resolved value', () => {
		const bytes = Buffer.from(JSON.stringify(identity));
		const sha = createHash('sha1').update(`blob ${bytes.byteLength}\0`).update(bytes).digest('hex');

		expect(decodePresetBlob('identity', 'cow.json', sha, bytes.toString('base64'))).toEqual({
			filename: 'cow.json',
			blobSha: sha,
			value: identity
		});
		expect(() =>
			decodePresetBlob('identity', 'cow.json', '0'.repeat(40), bytes.toString('base64'))
		).toThrow('blob SHA mismatch');
	});

	it('rejects incompatible resolved detector and identity combinations', () => {
		expect(() =>
			validateResolvedDetector({
				detection: { source: ['camera'] },
				yolo: {
					model: 'model.pt',
					tracking: true,
					confidence: { horse: 0.1 }
				},
				identity
			})
		).toThrow("target 'cow'");
		expect(() =>
			validateResolvedDetector({
				detection: { source: ['camera'] },
				yolo: {
					model: 'model.pt',
					tracking: true,
					confidence: { cow: 0.1 }
				},
				identity: {
					...identity,
					controlled_zone: { ...identity.controlled_zone, y2: 0.2 }
				}
			})
		).toThrow('positive extent');
	});
});
