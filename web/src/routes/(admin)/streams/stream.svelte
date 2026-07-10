<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import CardOverlay from '$lib/components/card-overlay.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Spinner } from '$lib/components/ui/spinner';
	import { onDestroy } from 'svelte';

	type Props = {
		label: string;
		source: string;
		showLoading?: boolean;
		hideOverlay?: boolean;
		disableLink?: boolean;
		showTracks?: boolean;
		tracksUrl?: string;
		tracksSource?: string;
	};

	let {
		label,
		source,
		showLoading = false,
		hideOverlay = false,
		disableLink = false,
		showTracks = false,
		tracksUrl = '/api/tracks/0',
		tracksSource = '0:0'
	}: Props = $props();

	type TrackCrop = {
		x1: number;
		y1: number;
		x2: number;
		y2: number;
		label?: string | null;
		confidence?: number | null;
	};

	type TrackObject = {
		id?: number;
		track_id?: number | null;
		label: string | null;
		confidence: number | null;
		crop: TrackCrop;
	};

	type TracksPayload = {
		type: 'tracks';
		source: string;
		timestamp: string;
		width: number;
		height: number;
		objects: TrackObject[];
	};

	const TRACK_STALE_MS = 2500;

	let imageReady = $state(false);
	let unavailable = $state(false);
	let tracks = $state<TracksPayload | null>(null);
	let image: HTMLImageElement | null = null;
	let trackStaleTimer: ReturnType<typeof setTimeout> | null = null;
	let frameWidth = $state(0);
	let frameHeight = $state(0);

	const streamUrl = $derived(resolve(`/streams/${encodeURIComponent(source)}`));
	const loading = $derived(!imageReady && !unavailable);

	function handleLoad() {
		imageReady = true;
		unavailable = false;
	}

	function handleError() {
		imageReady = false;
		unavailable = true;
	}

	function stopStream() {
		if (!image) {
			return;
		}

		image.removeAttribute('src');
		image.src = 'data:,';
	}

	function clearTrackStaleTimer() {
		if (trackStaleTimer) {
			clearTimeout(trackStaleTimer);
			trackStaleTimer = null;
		}
	}

	function clearTracksSoon() {
		clearTrackStaleTimer();
		trackStaleTimer = setTimeout(() => {
			tracks = null;
			trackStaleTimer = null;
		}, TRACK_STALE_MS);
	}

	function parseTracks(event: MessageEvent<string>) {
		try {
			const payload = JSON.parse(event.data) as TracksPayload;
			if (
				payload?.type !== 'tracks' ||
				payload.source !== tracksSource ||
				!Number.isFinite(payload.width) ||
				!Number.isFinite(payload.height)
			) {
				return;
			}
			tracks = payload;
			clearTracksSoon();
		} catch {
			return;
		}
	}

	function formatPercent(value: number | null | undefined) {
		return typeof value === 'number' ? `${Math.round(value * 100)}%` : '';
	}

	function objectLabel(object: TrackObject) {
		const objectName = object.label ?? object.crop.label ?? 'object';
		const confidence = formatPercent(object.confidence ?? object.crop.confidence);
		return confidence ? `${objectName} ${confidence}` : objectName;
	}

	function clamp(value: number, min: number, max: number) {
		return Math.min(Math.max(value, min), max);
	}

	function cropProjection(crop: TrackCrop, payload: TracksPayload) {
		if (frameWidth <= 0 || frameHeight <= 0) {
			return {
				x: (crop.x1 / payload.width) * 100,
				y: (crop.y1 / payload.height) * 100,
				width: ((crop.x2 - crop.x1) / payload.width) * 100,
				height: ((crop.y2 - crop.y1) / payload.height) * 100,
				unit: '%'
			};
		}

		const frameRatio = frameWidth / frameHeight;
		const streamRatio = payload.width / payload.height;
		let renderedWidth = frameWidth;
		let renderedHeight = frameHeight;
		let offsetX = 0;
		let offsetY = 0;

		if (frameRatio > streamRatio) {
			renderedWidth = frameHeight * streamRatio;
			offsetX = (frameWidth - renderedWidth) / 2;
		} else {
			renderedHeight = frameWidth / streamRatio;
			offsetY = (frameHeight - renderedHeight) / 2;
		}

		return {
			x: offsetX + (crop.x1 / payload.width) * renderedWidth,
			y: offsetY + (crop.y1 / payload.height) * renderedHeight,
			width: ((crop.x2 - crop.x1) / payload.width) * renderedWidth,
			height: ((crop.y2 - crop.y1) / payload.height) * renderedHeight,
			unit: 'px'
		};
	}

	function cropBoxStyle(object: TrackObject, payload: TracksPayload) {
		const box = cropProjection(object.crop, payload);
		return `left:${box.x}${box.unit};top:${box.y}${box.unit};width:${box.width}${box.unit};height:${box.height}${box.unit};`;
	}

	function cropBadgeStyle(object: TrackObject, payload: TracksPayload) {
		const box = cropProjection(object.crop, payload);
		const x =
			box.unit === 'px' ? clamp(box.x + box.width / 2, 20, frameWidth - 20) : box.x + box.width / 2;
		const y = box.unit === 'px' ? clamp(box.y, 18, frameHeight - 18) : box.y;
		return `left:${x}${box.unit};top:${y}${box.unit};`;
	}

	function trackObjectKey(object: TrackObject, index: number) {
		if (object.track_id !== null && object.track_id !== undefined) {
			return `track-${object.track_id}`;
		}
		return `object-${object.id ?? index}`;
	}

	$effect(() => {
		if (streamUrl) {
			imageReady = false;
			unavailable = false;
		}

		return stopStream;
	});

	$effect(() => {
		if (!showTracks || typeof window === 'undefined') {
			tracks = null;
			clearTrackStaleTimer();
			return;
		}

		const events = new EventSource(tracksUrl);
		events.addEventListener('tracks', parseTracks);
		events.addEventListener('error', () => {
			tracks = null;
			clearTrackStaleTimer();
		});

		return () => {
			events.removeEventListener('tracks', parseTracks);
			events.close();
			tracks = null;
			clearTrackStaleTimer();
		};
	});

	onDestroy(stopStream);
</script>

{#if showLoading && loading}
	<Spinner class="size-8" />
{/if}
<CardOverlay overlay={hideOverlay ? undefined : overlay}>
	<button
		bind:clientWidth={frameWidth}
		bind:clientHeight={frameHeight}
		class="relative block aspect-video w-full cursor-pointer bg-black"
		onclick={disableLink
			? undefined
			: () =>
					goto(
						resolve(
							`/streams/add?source=${encodeURIComponent(source)}&label=${encodeURIComponent(label)}`
						)
					)}
	>
		<img
			bind:this={image}
			src={streamUrl}
			alt={label}
			class="block h-full w-full object-contain"
			onload={handleLoad}
			onerror={handleError}
		/>

		{#if showTracks && tracks && tracks.width > 0 && tracks.height > 0}
			<div class="pointer-events-none absolute inset-0" aria-hidden="true">
				{#each tracks.objects as object, index (trackObjectKey(object, index))}
					<div
						class="absolute border-2 border-blue-500/90 shadow-[0_0_0_1px_rgba(255,255,255,0.75)] transition-[left,top,width,height] duration-300 ease-linear"
						style={cropBoxStyle(object, tracks)}
					></div>
					<div
						class="absolute z-10 -translate-x-1/2 -translate-y-full drop-shadow-[0_1px_4px_rgba(0,0,0,0.55)] transition-[left,top] duration-300 ease-linear"
						style={cropBadgeStyle(object, tracks)}
					>
						<Badge
							variant="secondary"
							class="border border-white/80 bg-blue-600 px-2 py-1 text-[11px] leading-none font-bold text-white shadow-none"
						>
							{objectLabel(object)}
						</Badge>
					</div>
				{/each}
			</div>
		{/if}

		{#if unavailable}
			<div
				class="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-white/70"
			>
				Live stream unavailable.
			</div>
		{:else if loading && !showLoading}
			<div
				class="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-white/60"
			>
				Loading...
			</div>
		{/if}
	</button>
</CardOverlay>

{#snippet overlay()}
	<div class="flex flex-wrap items-center gap-2 text-xs">
		<Badge variant="secondary" class="bg-black/50 text-white">{label}</Badge>
		<Badge variant="secondary" class="bg-black/50 text-white">{source}</Badge>
	</div>
{/snippet}
