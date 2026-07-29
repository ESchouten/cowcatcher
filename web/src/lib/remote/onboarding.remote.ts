import { command } from '$app/server';
import type { AppliedPreset, DetectorConfig, DetectorMeta, TelegramMeta } from '$lib/schema';
import { updateConfigState } from '$lib/server/config-store';
import {
	discoverOnvifCameras,
	resolveOnvifStream,
	type ResolvedOnvifStream
} from '$lib/server/onvif';
import { validateResolvedDetector } from '$lib/server/presets';
import * as v from 'valibot';

const camera = v.object({
	label: v.pipe(v.string(), v.trim(), v.minLength(1), v.maxLength(120)),
	source: v.pipe(v.string(), v.url(), v.regex(/^rtsps?:\/\//i))
});
const preset = v.object({
	filename: v.string(),
	blob_sha: v.string()
});
const telegram = v.object({
	label: v.pipe(v.string(), v.trim(), v.minLength(1)),
	token: v.pipe(v.string(), v.trim(), v.minLength(1)),
	chat: v.pipe(v.string(), v.trim(), v.minLength(1))
});

export const discoverCameras = command(
	v.object({
		timeoutMs: v.optional(v.pipe(v.number(), v.integer(), v.minValue(1_000), v.maxValue(10_000)))
	}),
	async ({ timeoutMs }) => discoverOnvifCameras(timeoutMs)
);

export const connectOnvifCamera = command(
	v.object({
		id: v.pipe(v.string(), v.uuid()),
		username: v.pipe(v.string(), v.maxLength(256)),
		password: v.pipe(v.string(), v.maxLength(512))
	}),
	async ({ id, username, password }): Promise<ResolvedOnvifStream> =>
		resolveOnvifStream(id, username, password)
);

export const completeOnboarding = command(
	v.object({
		cameras: v.pipe(v.array(camera), v.minLength(1), v.maxLength(64)),
		detector: v.any(),
		label: v.pipe(v.string(), v.trim(), v.minLength(1), v.maxLength(120)),
		detectorPreset: v.optional(preset),
		identityPreset: v.optional(preset),
		telegram: v.optional(telegram)
	}),
	async ({ cameras, detector, label, detectorPreset, identityPreset, telegram }) => {
		const sources = [...new Set(cameras.map((item) => item.source))];
		const resolvedDetector = structuredClone(detector) as DetectorConfig;
		resolvedDetector.detection.source = sources;
		validateResolvedDetector(resolvedDetector);

		const meta: DetectorMeta = {
			label,
			presets:
				detectorPreset || identityPreset
					? {
							detector: detectorPreset as AppliedPreset | undefined,
							identity: identityPreset as AppliedPreset | undefined
						}
					: undefined
		};
		const telegramMeta = telegram as TelegramMeta | undefined;

		await updateConfigState(({ config, app }) => {
			for (const camera of cameras) {
				const existing = app.streams.find((item) => item.source === camera.source);
				if (existing) existing.label = camera.label;
				else app.streams.push({ label: camera.label, source: camera.source });
			}

			if (
				telegramMeta &&
				!app.telegrams.some(
					(item) => item.token === telegramMeta.token && item.chat === telegramMeta.chat
				)
			) {
				app.telegrams.push(telegramMeta);
			}

			const existingIndex = app.detectors.findIndex((item) => item.label === label);
			if (existingIndex >= 0) {
				config.detectors[existingIndex] = resolvedDetector;
				app.detectors[existingIndex] = meta;
			} else {
				config.detectors.push(resolvedDetector);
				app.detectors.push(meta);
			}
		});

		return { detectorLabel: label, cameraCount: sources.length };
	}
);
