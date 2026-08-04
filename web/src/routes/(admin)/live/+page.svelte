<script lang="ts">
	import { resolve } from '$app/paths';
	import LiveStream from '$lib/components/live-stream.svelte';
	import { Button } from '$lib/components/ui/button';
	import { getStreams, getStreamSettings, reorderStream } from '$lib/remote/stream.remote';
	import SettingsIcon from '@lucide/svelte/icons/settings-2';
	import { flip } from 'svelte/animate';
	import { SvelteSet } from 'svelte/reactivity';
	import { dndzone, type DndEvent } from 'svelte-dnd-action';

	const ACTIVE_STREAM_LIMIT = 4;
	const FLIP_DURATION_MS = 150;
	const [streams, streamSettings] = await Promise.all([getStreams(), getStreamSettings()]);
	type StreamItem = (typeof streams)[number] & { id: string };
	const initialStreams = streams.map((stream) => ({ ...stream, id: stream.source }));
	let orderedStreams = $state<StreamItem[]>([...initialStreams]);
	let committedStreams = [...initialStreams];
	let visibleSources = new SvelteSet<string>();
	let saveError = $state(false);
	const activeSources = $derived(
		orderedStreams
			.filter((stream) => visibleSources.has(stream.source))
			.slice(0, ACTIVE_STREAM_LIMIT)
			.map((stream) => stream.source)
	);

	function visibility(node: HTMLElement, source: string) {
		const observer = new IntersectionObserver(
			([entry]) => {
				if (entry.isIntersecting) visibleSources.add(source);
				else visibleSources.delete(source);
			},
			{ threshold: 0.25 }
		);
		observer.observe(node);
		return { destroy: () => observer.disconnect() };
	}

	function consider(event: CustomEvent<DndEvent<StreamItem>>) {
		orderedStreams = event.detail.items;
	}

	async function finalize(event: CustomEvent<DndEvent<StreamItem>>) {
		const next = event.detail.items;
		const previous = committedStreams;
		const from = previous.findIndex((stream) => stream.id === event.detail.info.id);
		const to = next.findIndex((stream) => stream.id === event.detail.info.id);
		orderedStreams = next;
		if (from === -1 || to === -1 || from === to) {
			committedStreams = next;
			return;
		}
		try {
			await reorderStream({ index0: from, index1: to });
			committedStreams = next;
			saveError = false;
		} catch {
			orderedStreams = previous;
			committedStreams = previous;
			saveError = true;
		}
	}
</script>

<section class="space-y-6">
	<header class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
		<div class="space-y-1">
			<h1 class="text-2xl font-semibold tracking-tight">Live streams</h1>
			<p class="text-sm text-muted-foreground">
				Current camera feeds with detector and identity results.
			</p>
			{#if saveError}
				<p class="text-sm font-medium text-destructive">Camera order could not be saved.</p>
			{/if}
		</div>
		<Button href={resolve('/setup/cameras')} variant="outline">
			<SettingsIcon /> Configure cameras
		</Button>
	</header>

	{#if orderedStreams.length === 0}
		<div class="rounded-md border border-dashed p-8 text-center">
			<h3 class="font-semibold">No cameras configured</h3>
			<p class="mt-1 text-sm text-muted-foreground">Add a camera from Setup to begin.</p>
			<Button href={resolve('/setup/cameras/add')} class="mt-4">Add camera</Button>
		</div>
	{:else}
		<div
			class="grid cursor-grab gap-2 active:cursor-grabbing lg:grid-cols-2"
			use:dndzone={{
				items: orderedStreams,
				flipDurationMs: FLIP_DURATION_MS,
				delayTouchStart: true,
				type: 'live-streams'
			}}
			onconsider={consider}
			onfinalize={finalize}
		>
			{#each orderedStreams as stream (stream.id)}
				{@const tracks = streamSettings.tracksBySource[stream.source]}
				<div
					use:visibility={stream.source}
					aria-label={stream.label}
					animate:flip={{ duration: FLIP_DURATION_MS }}
				>
					{#if activeSources.includes(stream.source)}
						<LiveStream
							label={stream.label}
							source={stream.source}
							showTracks={tracks !== undefined}
							tracksUrl={tracks?.tracksUrl}
							tracksSource={tracks?.tracksSource}
						/>
					{:else}
						<div
							class="flex aspect-video items-center justify-center rounded-md border bg-black text-xs text-white/60"
						>
							Preview paused
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</section>
