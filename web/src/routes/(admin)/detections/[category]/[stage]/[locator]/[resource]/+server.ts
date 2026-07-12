import { DETECTIONS_DIR } from '$lib/server/shared-paths';
import fs from 'node:fs/promises';
import path from 'node:path';

const contentTypes: Record<string, string> = {
	'.jpg': 'image/jpeg',
	'.jpeg': 'image/jpeg',
	'.png': 'image/png',
	'.gif': 'image/gif',
	'.webp': 'image/webp',
	'.bmp': 'image/bmp',
	'.svg': 'image/svg+xml',
	'.mp4': 'video/mp4',
	'.webm': 'video/webm',
	'.mov': 'video/quicktime',
	'.json': 'application/json'
};

export async function GET({ params }) {
	const resolvedPath = path.join(
		DETECTIONS_DIR,
		params.category,
		params.stage,
		params.locator,
		params.resource
	);

	const file = await fs.readFile(resolvedPath);
	const extension = path.extname(resolvedPath).toLowerCase();

	return new Response(file, {
		headers: {
			'Content-Type': contentTypes[extension] ?? 'application/octet-stream',
			'Cache-Control': 'public, max-age=30'
		}
	});
}
