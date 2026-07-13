import { command, query } from '$app/server';
import { getConfig, getConfigSchema } from './config.remote';
import * as v from 'valibot';
import { updateConfigState } from '$lib/server/config-store';

export const getDetectorSchema = query(async () => {
	const configSchema = await getConfigSchema();

	return {
		$defs: configSchema.$defs,
		...(configSchema.$defs.DetectorConfig as Record<string, unknown>)
	};
});

export const getDetectors = query(async () => {
	const { config, app } = await getConfig();

	const detectorZip = config.detectors.map((detector, index) => {
		return { detector, meta: app.detectors[index] };
	});
	return detectorZip;
});

export const getDetector = query(
	v.object({
		label: v.string()
	}),
	async ({ label }) => {
		const detectors = await getDetectors();
		return detectors.find((detector) => detector.meta.label === label);
	}
);

export const saveDetector = command(
	v.object({
		original: v.optional(v.string()),
		detector: v.any(),
		meta: v.object({
			label: v.string()
		})
	}),
	async ({ original, detector, meta }) => {
		await updateConfigState(({ config, app }) => {
			if (original) {
				const index = app.detectors.findIndex((item) => item.label === original);
				if (index < 0) {
					throw new Error(`Detector '${original}' no longer exists`);
				}
				config.detectors[index] = detector;
				app.detectors[index] = meta;
			} else {
				config.detectors.push(detector);
				app.detectors.push(meta);
			}
		});
	}
);

export const deleteDetector = command(
	v.object({
		label: v.string()
	}),
	async ({ label }) => {
		await updateConfigState(({ config, app }) => {
			const index = app.detectors.findIndex((detector) => detector.label === label);
			if (index < 0) {
				throw new Error(`Detector '${label}' no longer exists`);
			}
			config.detectors.splice(index, 1);
			app.detectors.splice(index, 1);
		});
	}
);
