import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { chmod, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import ffmpegStatic from 'ffmpeg-static';

let ffmpegPath: Promise<string | null> | undefined;

export const getExecutableName = () => (process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg');

export const isRtspSource = (source: string) => /^rtsps?:\/\//i.test(source.trim());

export function sanitizeSourceForLogs(source: string): string {
	const sanitized = source.replace(
		/((?:^|[?&;/])(?:user(?:name)?|pass(?:word)?|pwd|token|key)=)([^&;/?\s]+)/gi,
		'$1***'
	);

	try {
		const url = new URL(sanitized);
		url.username = url.username ? '***' : '';
		url.password = url.password ? '***' : '';
		return url.toString();
	} catch {
		return sanitized;
	}
}

export function getRtspInputArgs(source: string): string[] {
	return [
		'-rtsp_transport',
		'tcp',
		'-timeout',
		'10000000',
		'-thread_queue_size',
		'512',
		'-analyzeduration',
		'10000000',
		'-probesize',
		'10000000',
		'-i',
		source
	];
}

function canRun(command: string): boolean {
	const probe = spawnSync(command, ['-version'], {
		stdio: ['ignore', 'ignore', 'ignore'],
		windowsHide: true
	});
	return !probe.error && probe.status === 0;
}

async function extractFfmpeg(requestUrl: URL): Promise<string | null> {
	const name = getExecutableName();
	const file = path.resolve(tmpdir(), 'ai-detector-web', 'bin', name);
	if (existsSync(file)) {
		return file;
	}

	let response: Response;
	try {
		response = await fetch(new URL(`/_internal/${name}`, requestUrl));
	} catch {
		return null;
	}

	if (!response.ok) {
		return null;
	}

	const bytes = new Uint8Array(await response.arrayBuffer());
	if (bytes.byteLength === 0) {
		return null;
	}

	await mkdir(path.dirname(file), { recursive: true });
	await writeFile(file, bytes);
	if (process.platform !== 'win32') {
		await chmod(file, 0o755);
	}
	return file;
}

async function resolveFfmpegPath(requestUrl: URL): Promise<string | null> {
	const configured = process.env.FFMPEG_PATH?.trim();
	if (configured && (existsSync(configured) || canRun(configured))) {
		return configured;
	}

	if (typeof ffmpegStatic === 'string' && existsSync(ffmpegStatic)) {
		return ffmpegStatic;
	}

	const name = getExecutableName();
	for (const candidate of [
		path.resolve(path.dirname(process.execPath), name),
		path.resolve(path.dirname(process.execPath), 'bin', name),
		path.resolve(process.cwd(), name),
		path.resolve(process.cwd(), 'bin', name)
	]) {
		if (existsSync(candidate)) {
			return candidate;
		}
	}

	return (await extractFfmpeg(requestUrl)) ?? (canRun('ffmpeg') ? 'ffmpeg' : null);
}

export function getFfmpegPathWithFallback(requestUrl: URL): Promise<string | null> {
	return (ffmpegPath ??= resolveFfmpegPath(requestUrl));
}
