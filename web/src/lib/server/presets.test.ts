import { describe, expect, it } from 'vitest';

import { decodePresetBlob, validatePreset, validateResolvedDetector } from './presets';

const identity = {
	target_label: 'cow',
	database: 'identities/cows.sqlite',
	margin: 0.2
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

	it('rejects unknown, incomplete, and invalid identity presets', () => {
		expect(() => validatePreset('identity', { ...identity, hidden_default: 1 })).toThrow(
			'additional properties'
		);
		const missing = Object.fromEntries(
			Object.entries(identity).filter(([key]) => key !== 'database')
		);
		expect(() => validatePreset('identity', missing)).toThrow('required');
		expect(() =>
			validatePreset('identity', {
				...identity,
				margin: 0.5
			})
		).toThrow('must be < 0.5');
	});

	it('decodes a GitHub preset', () => {
		const bytes = Buffer.from(JSON.stringify(identity));

		expect(decodePresetBlob('identity', 'cow.json', 'revision', bytes.toString('base64'))).toEqual({
			filename: 'cow.json',
			blobSha: 'revision',
			value: identity
		});
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
					margin: -0.1
				}
			})
		).toThrow('must be >= 0');
	});
});
