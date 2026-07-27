import { spawn } from 'node:child_process';
import path from 'node:path';
import { error, type RequestHandler } from '@sveltejs/kit';
import {
	getExecutableName,
	getFfmpegPathWithFallback,
	getRtspInputArgs,
	isRtspSource,
	sanitizeSourceForLogs
} from '$lib/server/ffmpeg';

const MJPEG_BOUNDARY = 'frame';
const FIRST_FRAME_TIMEOUT_MS = 20_000;

function createStream(source: string, ffmpegPath: string, signal: AbortSignal) {
	let cancelProcess = () => {};

	return new ReadableStream<Uint8Array>({
		start(controller) {
			if (signal.aborted) {
				controller.close();
				return;
			}

			const ffmpeg = spawn(
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
			let ended = false;
			let receivedFrame = false;
			const firstFrameTimer = setTimeout(() => {
				console.warn('FFmpeg preview stalled before first frame', {
					source: sanitizeSourceForLogs(source)
				});
				ffmpeg.kill();
				finish(new Error('Live stream unavailable.'));
			}, FIRST_FRAME_TIMEOUT_MS);

			const cleanup = () => {
				clearTimeout(firstFrameTimer);
				signal.removeEventListener('abort', cancelProcess);
			};
			const finish = (failure?: Error) => {
				if (ended) return;
				ended = true;
				cleanup();
				if (failure) controller.error(failure);
				else controller.close();
			};
			cancelProcess = () => {
				if (ended) return;
				ended = true;
				cleanup();
				ffmpeg.kill();
			};
			signal.addEventListener('abort', cancelProcess, { once: true });

			ffmpeg.stdout?.on('data', (chunk: Buffer<ArrayBufferLike>) => {
				if (ended) return;
				if (!receivedFrame) {
					receivedFrame = true;
					clearTimeout(firstFrameTimer);
				}
				controller.enqueue(chunk);
			});

			ffmpeg.stderr?.resume();

			ffmpeg.once('error', (error) => {
				console.error('FFmpeg preview process error', {
					source: sanitizeSourceForLogs(source),
					error
				});
				finish(new Error('Failed to start live stream preview.'));
			});

			ffmpeg.once('close', (exitCode) => {
				if (exitCode && !ended) {
					console.warn('FFmpeg preview ended', {
						source: sanitizeSourceForLogs(source),
						exitCode
					});
				}
				finish();
			});
		},
		cancel() {
			cancelProcess();
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
			'Cache-Control': 'no-store',
			'X-Accel-Buffering': 'no'
		}
	});
};
