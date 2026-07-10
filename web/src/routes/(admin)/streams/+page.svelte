<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { getStreams, getStreamSettings, reorderStream } from '$lib/remote/stream.remote';
	import Stream from './stream.svelte';
	import { Plus } from '@lucide/svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { flip } from 'svelte/animate';
	import { dndzone, type DndEvent } from 'svelte-dnd-action';

	const ACTIVE_STREAM_LIMIT = 4;
	const FLIP_DURATION_MS = 150;
	const [streams, streamSettings] = await Promise.all([getStreams(), getStreamSettings()]);

	type StreamItem = (typeof streams)[number] & { id: string };

	const initialStreams = streams.map((stream) => ({ ...stream, id: stream.source }));

	let orderedStreams = $state<StreamItem[]>([...initialStreams]);
	let visibleSources = new SvelteSet<string>();
	let saveError = $state(false);
	let committedStreams = [...initialStreams];

	const activeSources = $derived(
		orderedStreams
			.filter((stream) => visibleSources.has(stream.source))
			.slice(0, ACTIVE_STREAM_LIMIT)
			.map((stream) => stream.source)
	);

	function setStreamVisibility(source: string, visible: boolean) {
		if (visibleSources.has(source) === visible) {
			return;
		}

		if (visible) {
			visibleSources.add(source);
		} else {
			visibleSources.delete(source);
		}
	}

	function handleDndConsider(event: CustomEvent<DndEvent<StreamItem>>) {
		orderedStreams = event.detail.items;
	}

	async function handleDndFinalize(event: CustomEvent<DndEvent<StreamItem>>) {
		const nextStreams = event.detail.items;
		const previousStreams = committedStreams;
		const index0 = previousStreams.findIndex((stream) => stream.id === event.detail.info.id);
		const index1 = nextStreams.findIndex((stream) => stream.id === event.detail.info.id);

		orderedStreams = nextStreams;
		if (index0 === -1 || index1 === -1 || index0 === index1) {
			committedStreams = nextStreams;
			return;
		}

		try {
			await reorderStream({ index0, index1 });
			committedStreams = nextStreams;
			saveError = false;
		} catch {
			orderedStreams = previousStreams;
			committedStreams = previousStreams;
			saveError = true;
		}
	}

	function trackStreamVisibility(node: HTMLElement, source: string) {
		const observer = new IntersectionObserver(
			([entry]) => {
				setStreamVisibility(source, entry.isIntersecting);
			},
			{ threshold: 0.25 }
		);
		observer.observe(node);

		return {
			destroy() {
				observer.disconnect();
			}
		};
	}
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-semibold tracking-tight">Live streams</h1>
			<Button href="/streams/add" variant="outline"><Plus /> Add stream</Button>
		</div>
		<p class="text-sm text-muted-foreground">Live stream from your RTSP sources.</p>
		{#if saveError}
			<p class="text-sm text-destructive">Stream order could not be saved.</p>
		{/if}
	</header>

	<div
		class="grid cursor-grab gap-2 active:cursor-grabbing lg:grid-cols-2"
		use:dndzone={{
			items: orderedStreams,
			flipDurationMs: FLIP_DURATION_MS,
			delayTouchStart: true,
			type: 'streams'
		}}
		onconsider={handleDndConsider}
		onfinalize={handleDndFinalize}
	>
		{#each orderedStreams as stream (stream.id)}
			<div
				use:trackStreamVisibility={stream.source}
				class="relative transition-opacity"
				aria-label={stream.label}
				animate:flip={{ duration: FLIP_DURATION_MS }}
			>
				{#if activeSources.includes(stream.source)}
					<Stream
						label={stream.label}
						source={stream.source}
						showTracks
						tracksUrl={streamSettings.tracksBySource[stream.source]?.tracksUrl ??
							streamSettings.tracksUrl}
						tracksSource={streamSettings.tracksBySource[stream.source]?.tracksSource ??
							streamSettings.tracksSource}
					/>
				{:else}
					<div class="relative aspect-video w-full bg-black">
						<div
							class="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-white/60"
						>
							Preview paused
						</div>
					</div>
				{/if}
			</div>
		{/each}
	</div>
</section>
