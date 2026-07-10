import fs from 'node:fs/promises';
import path from 'node:path';
import { query } from '$app/server';
import * as v from 'valibot';
import { STAGES, type DetectionMetadata, type Stage } from '$lib/schema';
import { DETECTIONS_DIR } from '$lib/server/shared-paths';

interface DetectionPage {
	items: DetectionMetadata[];
	hasMore: boolean;
	nextOffset: number;
}

interface DetectionLocator {
	type: string;
	stage: Stage;
	timestamp: string;
	epoch: number;
}

async function listFolders(directoryPath: string): Promise<string[]> {
	try {
		const entries = await fs.readdir(directoryPath, { withFileTypes: true });
		return entries
			.filter((entry) => entry.isDirectory())
			.map((entry) => entry.name)
			.sort((a, b) => a.localeCompare(b));
	} catch {
		return [];
	}
}

async function readDetection(
	type: string,
	stage: Stage,
	timestamp: string
): Promise<DetectionMetadata> {
	const metadataPath = path.join(DETECTIONS_DIR, type, stage, timestamp, 'metadata.json');
	const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf8'));
	return { ...metadata, type } as DetectionMetadata;
}

function toEpoch(value: unknown): number {
	if (value instanceof Date) {
		return value.getTime();
	}
	if (typeof value === 'string') {
		const normalized = value.replace(
			/^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})(\.\d+)?$/,
			'$1T$2:$3:$4$5'
		);
		const parsed = Date.parse(normalized);
		return Number.isFinite(parsed) ? parsed : 0;
	}
	return 0;
}

export const getTypes = query(async () => {
	return listFolders(DETECTIONS_DIR);
});

async function listDetectionLocators(
	types: string[],
	stages: readonly Stage[]
): Promise<DetectionLocator[]> {
	const locators: DetectionLocator[] = [];

	for (const type of types) {
		for (const stage of stages) {
			const stagePath = path.join(DETECTIONS_DIR, type, stage);
			const timestamps = await listFolders(stagePath);
			for (const timestamp of timestamps) {
				locators.push({
					type,
					stage,
					timestamp,
					epoch: toEpoch(timestamp)
				});
			}
		}
	}

	return locators.sort(
		(a, b) =>
			b.epoch - a.epoch ||
			b.timestamp.localeCompare(a.timestamp) ||
			a.type.localeCompare(b.type) ||
			a.stage.localeCompare(b.stage)
	);
}

export const getDetectionPage = query(
	v.object({
		type: v.optional(v.string()),
		stage: v.optional(v.picklist(STAGES)),
		offset: v.pipe(v.number(), v.integer(), v.minValue(0)),
		limit: v.pipe(v.number(), v.integer(), v.minValue(1), v.maxValue(100))
	}),
	async ({ type, stage, offset, limit }): Promise<DetectionPage> => {
		const types = type ? [type] : await listFolders(DETECTIONS_DIR);
		const stages = stage ? [stage] : STAGES;
		const locators = await listDetectionLocators(types, stages);
		const page = locators.slice(offset, offset + limit);
		const items = await Promise.all(
			page.map(({ type, stage, timestamp }) => readDetection(type, stage, timestamp))
		);

		return {
			items,
			hasMore: offset + items.length < locators.length,
			nextOffset: offset + items.length
		};
	}
);
