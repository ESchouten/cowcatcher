import type { DetectorConfig } from '$lib/schema';

export type DetectorPresetFragment = Omit<
	Partial<DetectorConfig>,
	'detection' | 'yolo' | 'identity' | 'exporters'
> & {
	detection?: Partial<DetectorConfig['detection']>;
	yolo?: Partial<NonNullable<DetectorConfig['yolo']>>;
};

export function createDetectorFromPreset(fragment: DetectorPresetFragment): DetectorConfig {
	return applyDetectorPreset(
		{
			detection: { source: [] },
			yolo: { model: '', confidence: 0.8 },
			exporters: {
				disk: [{}],
				sse: [{}]
			}
		},
		fragment
	);
}

export function applyDetectorPreset(
	detector: DetectorConfig,
	fragment: DetectorPresetFragment
): DetectorConfig {
	const yolo =
		typeof fragment.yolo?.model === 'string'
			? {
					...fragment.yolo,
					model: fragment.yolo.model
				}
			: undefined;
	return {
		...detector,
		...fragment,
		detection: {
			...fragment.detection,
			source: detector.detection.source
		},
		yolo,
		identity: detector.identity,
		exporters: detector.exporters
	};
}
