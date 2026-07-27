import { query } from '$app/server';
import * as v from 'valibot';
import { decodePresetBlob, type PresetCategory, type PresetDocument } from '$lib/server/presets';

const category = v.picklist(['detector', 'identity', 'vlm']);
const file = v.pipe(v.string(), v.regex(/^[^/\\]+\.json$/));
const repository = 'https://api.github.com/repos/ESchouten/ai-detector/contents/config';
const githubHeaders = {
	Accept: 'application/vnd.github+json',
	'User-Agent': 'ai-detector-web',
	'X-GitHub-Api-Version': '2022-11-28'
};
const listingSchema = v.array(
	v.object({
		name: v.string(),
		type: v.string()
	})
);
const contentSchema = v.object({
	type: v.literal('file'),
	sha: v.pipe(v.string(), v.regex(/^[0-9a-f]{40,64}$/)),
	encoding: v.literal('base64'),
	content: v.string()
});

export const getPresets = query(v.object({ category }), async ({ category }) => {
	const response = await fetch(`${repository}/${category}?ref=main`, {
		headers: githubHeaders
	});
	if (!response.ok) throw new Error(`Failed to load ${category} presets: ${response.status}`);

	const items = v.parse(listingSchema, await response.json());
	return items
		.filter((item) => item.type === 'file' && item.name.endsWith('.json'))
		.map((item) => item.name);
});

export const getPreset = query(
	v.object({ category, file }),
	async ({ category, file }): Promise<PresetDocument> => {
		const response = await fetch(`${repository}/${category}/${file}?ref=main`, {
			headers: githubHeaders
		});
		if (!response.ok) {
			throw new Error(`Failed to load ${category} preset '${file}': ${response.status}`);
		}
		const document = v.parse(contentSchema, await response.json());
		return decodePresetBlob(category as PresetCategory, file, document.sha, document.content);
	}
);
