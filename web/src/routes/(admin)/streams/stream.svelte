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
		tracksEndpoint?: string;
		tracksPort?: number;
	};

	let {
		label,
		source,
		showLoading = false,
		hideOverlay = false,
		disableLink = false,
		showTracks = false,
		tracksEndpoint = '/events',
		tracksPort = 8765
	}: Props = $props();

	type TrackIdentity = {
		identity_id: string | null;
		name: string | null;
		status: 'matched' | 'created' | 'unknown';
		similarity?: number | null;
	};

	type TrackObject = {
		track_id: number | null;
		label: string | null;
		confidence: number | null;
		crop: {
			x1: number;
			y1: number;
			x2: number;
			y2: number;
		};
		identity: TrackIdentity | null;
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

	const streamUrl = $derived(resolve(`/streams/${encodeURIComponent(source)}`));
	const loading = $derived(!imageReady && !unavailable);
	let frameWidth = $state(0);
	let frameHeight = $state(0);

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

	function getTracksUrl() {
		const configuredEndpoint = tracksEndpoint.trim() || '/events';
		const endpoint = configuredEndpoint.startsWith('/')
			? configuredEndpoint
			: `/${configuredEndpoint}`;
		return `${window.location.protocol}//${window.location.hostname}:${tracksPort}${endpoint}`;
	}

	function parseTracks(event: MessageEvent<string>) {
		try {
			const payload = JSON.parse(event.data) as TracksPayload;
			if (
				payload?.type !== 'tracks' ||
				payload.source !== source ||
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

	function identityLabel(identity: TrackIdentity | null) {
		if (!identity) {
			return null;
		}
		if (identity.status === 'unknown') {
			return 'unknown';
		}
		const name = identity.name ?? identity.identity_id;
		if (!name) {
			return null;
		}
		return name;
	}

	function identityDetail(identity: TrackIdentity) {
		if (identity.status === 'created') {
			return 'new';
		}
		return formatPercent(identity.similarity);
	}

	function identityTone(identity: TrackIdentity) {
		if (identity.status === 'created') {
			return 'bg-emerald-600 text-white';
		}
		if (identity.status === 'unknown') {
			return 'bg-amber-500 text-white';
		}
		return 'bg-blue-600 text-white';
	}

	function clamp(value: number, min: number, max: number) {
		return Math.min(Math.max(value, min), max);
	}

	function trackBadgeStyle(object: TrackObject, payload: TracksPayload) {
		if (frameWidth <= 0 || frameHeight <= 0) {
			const x = ((object.crop.x1 + object.crop.x2) / 2 / payload.width) * 100;
			const y = (object.crop.y1 / payload.height) * 100;
			return `left:${x}%;top:${y}%;`;
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

		const x = offsetX + (((object.crop.x1 + object.crop.x2) / 2) / payload.width) * renderedWidth;
		const y = offsetY + (object.crop.y1 / payload.height) * renderedHeight;

		return `left:${clamp(x, 16, frameWidth - 16)}px;top:${clamp(y + 8, 18, frameHeight - 18)}px;`;
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

		const events = new EventSource(getTracksUrl());
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
				{#each tracks.objects as object, index (`${object.track_id ?? index}-${object.crop.x1}-${object.crop.y1}-${object.crop.x2}-${object.crop.y2}`)}
					{@const identity = object.identity}
					{@const label = identityLabel(identity)}
					{#if identity && label}
						{@const detail = identityDetail(identity)}
						<div
							class="absolute z-10 flex -translate-x-1/2 -translate-y-1/2 items-center drop-shadow-[0_1px_4px_rgba(0,0,0,0.55)]"
							style={trackBadgeStyle(object, tracks)}
						>
							<Badge
								variant="secondary"
								class="rounded-r-none border border-white/80 bg-white px-2 py-1 text-[11px] font-bold leading-none text-slate-950 shadow-none"
							>
								{label}
							</Badge>
							{#if detail}
								<Badge
									variant="secondary"
									class={`rounded-l-none border border-l-0 border-white/80 px-2 py-1 text-[11px] font-bold leading-none shadow-none ${identityTone(identity)}`}
								>
									{detail}
								</Badge>
							{/if}
						</div>
					{/if}
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
