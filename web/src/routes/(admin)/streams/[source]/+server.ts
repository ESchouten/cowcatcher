import { spawn } from 'node:child_process';
import path from 'node:path';
import { error, type RequestHandler } from '@sveltejs/kit';
import {
	getExecutableName,
	getFfmpegPathWithFallback,
	getRtspInputArgs,
	isRtspSource,
	sanitizeSourceForLogs,
	sanitizeTextForLogs
} from '$lib/server/ffmpeg';

const MJPEG_BOUNDARY = 'frame';
const FIRST_FRAME_TIMEOUT_MS = 20_000;
const NO_FRAME_TIMEOUT_MS = 8_000;
const FORCE_KILL_DELAY_MS = 2_000;
const MAX_STDERR_TAIL_LENGTH = 4_000;

type Timeout = ReturnType<typeof setTimeout>;

const clearTimer = (timer: Timeout | null) => {
	if (timer) {
		clearTimeout(timer);
	}
	return null;
};

function appendStderrTail(stderr: string, chunk: Buffer<ArrayBufferLike>) {
	const next = stderr + chunk.toString('utf8');
	return next.length > MAX_STDERR_TAIL_LENGTH ? next.slice(-MAX_STDERR_TAIL_LENGTH) : next;
}

function createStream(source: string, ffmpegPath: string, signal: AbortSignal) {
	let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
	let ffmpeg: ReturnType<typeof spawn> | null = null;
	let firstFrameTimer: Timeout | null = null;
	let noFrameTimer: Timeout | null = null;
	let killTimer: Timeout | null = null;
	let stderr = '';
	let hadFrame = false;
	let closed = false;
	let stopReason: string | null = null;

	const finish = (reason?: Error) => {
		if (closed) {
			return;
		}

		closed = true;
		firstFrameTimer = clearTimer(firstFrameTimer);
		noFrameTimer = clearTimer(noFrameTimer);
		killTimer = clearTimer(killTimer);
		signal.removeEventListener('abort', onAbort);

		if (!controller) {
			return;
		}

		try {
			reason ? controller.error(reason) : controller.close();
		} catch (error) {
			if (
				!(
					error instanceof TypeError &&
					(error as NodeJS.ErrnoException).code === 'ERR_INVALID_STATE'
				)
			) {
				throw error;
			}
		}
	};

	const stop = (reason: string, hard = false) => {
		if (!stopReason) {
			stopReason = reason;
		}

		if (!ffmpeg || ffmpeg.exitCode !== null) {
			finish();
			return;
		}

		ffmpeg.kill(hard ? 'SIGKILL' : 'SIGTERM');
		if (!hard) {
			killTimer = clearTimer(killTimer);
			killTimer = setTimeout(() => {
				if (ffmpeg && ffmpeg.exitCode === null) {
					ffmpeg.kill('SIGKILL');
				}
			}, FORCE_KILL_DELAY_MS);
		}
	};

	const onAbort = () => {
		stop('Client disconnected.');
		finish();
	};

	return new ReadableStream<Uint8Array>({
		start(nextController) {
			controller = nextController;
			if (signal.aborted) {
				onAbort();
				return;
			}

			signal.addEventListener('abort', onAbort, { once: true });
			ffmpeg = spawn(
				ffmpegPath,
				[
					'-hide_banner',
					'-loglevel',
					'error',
					'-nostdin',
					...getRtspInputArgs(source),
					'-map',
					'0:v:0',
					'-an',
					'-sn',
					'-dn',
					'-c:v',
					'mjpeg',
					'-vf',
					'fps=8,scale=960:-1:flags=lanczos',
					'-q:v',
					'7',
					'-f',
					'mpjpeg',
					'-boundary_tag',
					MJPEG_BOUNDARY,
					'pipe:1'
				],
				{
					stdio: ['ignore', 'pipe', 'pipe'],
					windowsHide: true
				}
			);

			firstFrameTimer = setTimeout(() => {
				console.warn('FFmpeg preview stalled before first frame', {
					source: sanitizeSourceForLogs(source)
				});
				stop('No first frame received before timeout.');
			}, FIRST_FRAME_TIMEOUT_MS);

			ffmpeg.stdout?.on('data', (chunk: Buffer<ArrayBufferLike>) => {
				if (closed) {
					return;
				}

				if (!hadFrame) {
					hadFrame = true;
					firstFrameTimer = clearTimer(firstFrameTimer);
				}

				noFrameTimer = clearTimer(noFrameTimer);
				noFrameTimer = setTimeout(() => {
					console.warn('FFmpeg preview stalled after frames stopped', {
						source: sanitizeSourceForLogs(source)
					});
					stop('No frames received before timeout.');
				}, NO_FRAME_TIMEOUT_MS);

				try {
					nextController.enqueue(chunk);
				} catch {
					stop('Client could not accept more stream data.');
					finish();
				}
			});

			ffmpeg.stderr?.on('data', (chunk: Buffer<ArrayBufferLike>) => {
				stderr = appendStderrTail(stderr, chunk);
			});

			ffmpeg.once('error', (error) => {
				console.error('FFmpeg preview process error', {
					source: sanitizeSourceForLogs(source),
					error
				});
				finish(new Error('Failed to start live stream preview.'));
			});

			ffmpeg.once('close', (exitCode, signal) => {
				if (closed) {
					return;
				}

				console.warn(
					hadFrame ? 'FFmpeg preview ended' : 'FFmpeg preview exited before first frame',
					{
						source: sanitizeSourceForLogs(source),
						exitCode: exitCode ?? 'unknown',
						signal: signal ?? undefined,
						hadFrame,
						reason: stopReason ?? undefined,
						stderr: stderr.trim() ? sanitizeTextForLogs(stderr.trim()) : undefined
					}
				);
				finish(new Error(hadFrame ? 'Live stream ended.' : 'Live stream unavailable.'));
			});
		},
		cancel() {
			stop('Stream response cancelled.');
			finish();
		}
	});
}

export const GET: RequestHandler = async ({ params, request }) => {
	const source = params.source?.trim();
	if (!source || !isRtspSource(source)) {
		throw error(400, 'Only RTSP and RTSPS sources are supported for live preview.');
	}

	const ffmpegPath = await getFfmpegPathWithFallback(new URL(request.url));
	if (!ffmpegPath) {
		throw error(
			500,
			`FFmpeg binary not available. Set FFMPEG_PATH, install ffmpeg in PATH, or place ${getExecutableName()} next to ${path.basename(process.execPath)}.`
		);
	}

	return new Response(createStream(source, ffmpegPath, request.signal), {
		headers: {
			'Content-Type': `multipart/x-mixed-replace; boundary=${MJPEG_BOUNDARY}`,
			'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0',
			Pragma: 'no-cache',
			Connection: 'keep-alive',
			'X-Accel-Buffering': 'no'
		}
	});
};
