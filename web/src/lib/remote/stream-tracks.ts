type TrackSourceConfig = {
	detectors: Array<{
		detection: { source: string[] };
		exporters?: { sse?: unknown[] };
	}>;
};

export type StreamTrackSettings = {
	tracksUrl: string;
	tracksSource: string;
};

export function tracksBySource(config: TrackSourceConfig): Record<string, StreamTrackSettings> {
	const settings: Record<string, StreamTrackSettings> = {};
	config.detectors.forEach((detector, detectorIndex) => {
		if (!detector.exporters?.sse?.[0]) {
			return;
		}

		detector.detection.source.forEach((source, sourceIndex) => {
			settings[source] = {
				tracksUrl: `/api/tracks/${detectorIndex}`,
				tracksSource: `${detectorIndex}:${sourceIndex}`
			};
		});
	});
	return settings;
}
