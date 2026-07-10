import { dev } from '$app/environment';
import path from 'node:path';

declare const __AI_DETECTOR_WEB_TARGET__: string;

const buildTarget = __AI_DETECTOR_WEB_TARGET__;

function getRuntimeDataDirectory(): string {
	if (dev || buildTarget === 'node') {
		return process.cwd();
	}

	return path.dirname(process.execPath);
}

const runtimeDataDirectory = getRuntimeDataDirectory();

export const CONFIG_PATH = path.resolve(runtimeDataDirectory, 'config.json');
export const APP_CONFIG_PATH = path.resolve(runtimeDataDirectory, 'app.json');
export const DETECTIONS_DIR = path.resolve(runtimeDataDirectory, 'detections');

export function resolveWithinDirectory(
	directoryPath: string,
	requestedPath: string
): string | null {
	const normalizedPath = requestedPath.replaceAll('\\', '/').replace(/^\/+/, '');
	const resolvedPath = path.resolve(directoryPath, normalizedPath);
	const relativePath = path.relative(directoryPath, resolvedPath);

	if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
		return null;
	}

	return resolvedPath;
}
