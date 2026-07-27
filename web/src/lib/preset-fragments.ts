import type { DetectorConfig } from '$lib/schema';

export type DetectorPresetFragment = Omit<
	Partial<DetectorConfig>,
	'detection' | 'yolo' | 'identity' | 'exporters'
> & {
	detection?: Partial<DetectorConfig['detection']>;
	yolo?: Partial<NonNullable<DetectorConfig['yolo']>>;
};

export function applyDetectorPreset(
	detector: DetectorConfig,
	fragment: DetectorPresetFragment
): DetectorConfig {
	const mergedYolo = {
		...detector.yolo,
		...fragment.yolo
	};
	const yolo =
		typeof mergedYolo.model === 'string'
			? {
					...mergedYolo,
					model: mergedYolo.model
				}
			: undefined;
	return {
		...detector,
		...fragment,
		detection: {
			...detector.detection,
			...fragment.detection,
			source: detector.detection.source
		},
		yolo,
		identity: detector.identity,
		exporters: detector.exporters
	};
}
