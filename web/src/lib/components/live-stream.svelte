<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { onDestroy } from 'svelte';

	type IdentityResult = {
		status: 'matched' | 'unknown' | 'ambiguous' | 'insufficient_evidence' | 'switch_risk';
		visual_identity_id: string | null;
		official_id: string | null;
		similarity: number | null;
		margin: number | null;
	};

	type TrackCrop = {
		x1: number;
		y1: number;
		x2: number;
		y2: number;
		label?: string | null;
		confidence?: number | null;
		identity?: IdentityResult | null;
	};

	type TrackObject = {
		id?: number;
		track_id?: number | null;
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

	let {
		label,
		source,
		showTracks = false,
		tracksUrl = '/api/tracks/0',
		tracksSource = '0:0'
	}: {
		label: string;
		source: string;
		showTracks?: boolean;
		tracksUrl?: string;
		tracksSource?: string;
	} = $props();

	const TRACK_STALE_MS = 2_500;
	let imageReady = $state(false);
	let unavailable = $state(false);
	let tracks = $state<TracksPayload | null>(null);
	let image: HTMLImageElement | null = null;
	let trackStaleTimer: ReturnType<typeof setTimeout> | null = null;
	let frameWidth = $state(0);
	let frameHeight = $state(0);
	const streamUrl = $derived(`/streams/${encodeURIComponent(source)}`);

	function stopStream() {
		if (!image) return;
		image.removeAttribute('src');
		image.src = 'data:,';
	}

	function clearTrackStaleTimer() {
		if (!trackStaleTimer) return;
		clearTimeout(trackStaleTimer);
		trackStaleTimer = null;
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
			clearTrackStaleTimer();
			trackStaleTimer = setTimeout(() => {
				tracks = null;
				trackStaleTimer = null;
			}, TRACK_STALE_MS);
		} catch {
			return;
		}
	}

	function percent(value: number | null | undefined): string {
		return typeof value === 'number' ? `${Math.round(value * 100)}%` : '';
	}

	function identityLabel(identity: IdentityResult): string {
		switch (identity.status) {
			case 'matched':
				return `${identity.official_id ?? 'Matched'} ${percent(identity.similarity)}`.trim();
			case 'ambiguous':
				return 'Identity uncertain';
			case 'insufficient_evidence':
				return 'Collecting identity';
			case 'switch_risk':
				return 'Track changed';
			default:
				return 'Unknown';
		}
	}

	function objectLabel(object: TrackObject): string {
		if (object.crop.identity) return identityLabel(object.crop.identity);
		const objectName = object.crop.label ?? 'object';
		const confidence = percent(object.crop.confidence);
		return confidence ? `${objectName} ${confidence}` : objectName;
	}

	function clamp(value: number, minimum: number, maximum: number): number {
		return Math.min(Math.max(value, minimum), maximum);
	}

	function projection(crop: TrackCrop, payload: TracksPayload) {
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

	function boxStyle(object: TrackObject, payload: TracksPayload): string {
		const box = projection(object.crop, payload);
		return `left:${box.x}${box.unit};top:${box.y}${box.unit};width:${box.width}${box.unit};height:${box.height}${box.unit};`;
	}

	function badgeStyle(object: TrackObject, payload: TracksPayload): string {
		const box = projection(object.crop, payload);
		const x =
			box.unit === 'px' ? clamp(box.x + box.width / 2, 20, frameWidth - 20) : box.x + box.width / 2;
		const y = box.unit === 'px' ? clamp(box.y, 18, frameHeight - 18) : box.y;
		return `left:${x}${box.unit};top:${y}${box.unit};`;
	}

	function objectKey(object: TrackObject, index: number): string {
		return object.track_id === null || object.track_id === undefined
			? `object-${object.id ?? index}`
			: `track-${object.track_id}`;
	}

	$effect(() => {
		imageReady = false;
		unavailable = false;
		return stopStream;
	});

	$effect(() => {
		if (!showTracks) {
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

<article class="relative overflow-hidden rounded-md border bg-card">
	<div
		bind:clientWidth={frameWidth}
		bind:clientHeight={frameHeight}
		class="relative aspect-video w-full bg-black"
	>
		<img
			bind:this={image}
			src={streamUrl}
			alt={label}
			class="block h-full w-full object-contain"
			onload={() => {
				imageReady = true;
				unavailable = false;
			}}
			onerror={() => {
				imageReady = false;
				unavailable = true;
			}}
		/>

		{#if showTracks && tracks && tracks.width > 0 && tracks.height > 0}
			<div class="pointer-events-none absolute inset-0" aria-hidden="true">
				{#each tracks.objects as object, index (objectKey(object, index))}
					<div
						class:border-emerald-400={object.crop.identity?.status === 'matched'}
						class:border-amber-400={object.crop.identity &&
							object.crop.identity.status !== 'matched'}
						class:border-white={!object.crop.identity}
						class="absolute border-2 shadow-[0_0_0_1px_rgba(0,0,0,0.25)] transition-[left,top,width,height] duration-300 ease-linear"
						style={boxStyle(object, tracks)}
					></div>
					<div
						class="absolute z-10 -translate-x-1/2 -translate-y-full transition-[left,top] duration-300 ease-linear"
						style={badgeStyle(object, tracks)}
					>
						<Badge
							class={`border border-white/80 px-2 py-1 text-[11px] leading-none font-bold text-white shadow-none ${
								object.crop.identity?.status === 'matched'
									? 'bg-emerald-700'
									: object.crop.identity
										? 'bg-amber-600'
										: 'bg-blue-600'
							}`}
						>
							{objectLabel(object)}
						</Badge>
					</div>
				{/each}
			</div>
		{/if}

		{#if unavailable}
			<div class="absolute inset-0 flex items-center justify-center text-sm text-white/65">
				Live stream unavailable
			</div>
		{:else if !imageReady}
			<div class="absolute inset-0 flex items-center justify-center text-sm text-white/55">
				Connecting…
			</div>
		{/if}
	</div>
	<div
		class="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center gap-2 bg-linear-to-b from-black/80 via-black/45 to-transparent p-3 text-white"
	>
		<Badge variant="secondary" class="max-w-48 truncate bg-black/50 text-white shadow-none">
			{label}
		</Badge>
		<Badge variant="secondary" class="max-w-64 min-w-0 truncate bg-black/50 text-white shadow-none">
			{source}
		</Badge>
		<span
			class:bg-emerald-500={imageReady}
			class:bg-amber-500={!imageReady}
			class="ms-auto size-2 shrink-0 rounded-full"
			title={imageReady ? 'Connected' : 'Not connected'}
		></span>
	</div>
</article>
