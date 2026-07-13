import { query } from '$app/server';
import * as v from 'valibot';

const category = v.picklist(['detector', 'identity', 'vlm']);
const file = v.pipe(v.string(), v.regex(/^[^/\\]+\.json$/));
const repository = 'https://api.github.com/repos/ESchouten/ai-detector/contents/config';

export const getPresets = query(v.object({ category }), async ({ category }) => {
	const response = await fetch(`${repository}/${category}`, {
		headers: {
			Accept: 'application/vnd.github+json',
			'User-Agent': 'ai-detector-web'
		}
	});
	if (!response.ok) throw new Error(`Failed to load ${category} presets: ${response.status}`);

	const items = (await response.json()) as Array<{ name: string; type: string }>;
	return items
		.filter((item) => item.type === 'file' && item.name.endsWith('.json'))
		.map((item) => item.name);
});

export const getPreset = query(
	v.object({ category, file }),
	async ({ category, file }): Promise<unknown> => {
		const response = await fetch(
			`https://raw.githubusercontent.com/ESchouten/ai-detector/main/config/${category}/${file}`
		);
		if (!response.ok) {
			throw new Error(`Failed to load ${category} preset '${file}': ${response.status}`);
		}
		return response.json();
	}
);
