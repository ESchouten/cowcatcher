<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { getDetectionPage, getTypes } from '$lib/remote/detections.remote';
	import { STAGES, type DetectionMetadata, type Stage } from '$lib/schema';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import type { Action } from 'svelte/action';
	import DetectionCard from './detection-card.svelte';
	import { SvelteMap, SvelteURLSearchParams } from 'svelte/reactivity';

	const PAGE_SIZE = 24;
	const dayFormatter = new Intl.DateTimeFormat(undefined, { dateStyle: 'full' });

	const type = $derived(page.url.searchParams.get('type') || undefined);
	const stage = $derived((page.url.searchParams.get('stage') as Stage | null) || undefined);
	const types = $derived(await getTypes());

	let entries = $state<DetectionMetadata[]>([]);
	let isLoading = $state(false);
	let hasMore = $state(true);
	let nextOffset = $state(0);
	let errorMessage = $state<string | null>(null);
	let requestVersion = 0;

	const detectionsByDay = $derived.by(() => {
		const dayDetections = new SvelteMap<string, Array<DetectionMetadata>>();
		for (const detection of entries) {
			const day = String(detection.timestamp).split('T')[0];
			if (!dayDetections.has(day)) {
				dayDetections.set(day, []);
			}
			dayDetections.set(day, [...(dayDetections.get(day) ?? []), detection]);
		}
		return Array.from(dayDetections.entries());
	});

	function capitalize(value: string) {
		return value.charAt(0).toUpperCase() + value.slice(1);
	}

	async function loadNextPage(reset = false) {
		if (!reset && (isLoading || !hasMore)) {
			return;
		}

		if (reset) {
			requestVersion += 1;
			entries = [];
			nextOffset = 0;
			hasMore = true;
			errorMessage = null;
		}
		const version = requestVersion;

		isLoading = true;

		try {
			const result = await getDetectionPage({
				type,
				stage,
				offset: reset ? 0 : nextOffset,
				limit: PAGE_SIZE
			});

			if (version !== requestVersion) {
				return;
			}

			entries = reset ? result.items : [...entries, ...result.items];
			nextOffset = result.nextOffset;
			hasMore = result.hasMore;
		} catch (error) {
			if (version === requestVersion) {
				errorMessage = error instanceof Error ? error.message : 'Failed to load detections.';
			}
		} finally {
			if (version === requestVersion) {
				isLoading = false;
			}
		}
	}

	async function updateSearchParams(type?: string, stage?: string) {
		const searchParams = new SvelteURLSearchParams(page.url.searchParams);

		if (type) {
			searchParams.set('type', type);
		} else {
			searchParams.delete('type');
		}

		if (stage) {
			searchParams.set('stage', stage);
		} else {
			searchParams.delete('stage');
		}

		const search = searchParams.toString();
		const nextUrl = search ? `${page.url.pathname}?${search}` : page.url.pathname;
		const currentUrl = `${page.url.pathname}${page.url.search}`;

		if (nextUrl !== currentUrl) {
			await goto(nextUrl, {
				replaceState: true,
				noScroll: true,
				keepFocus: true,
				invalidateAll: false
			});
		}
	}

	const infiniteTrigger: Action<HTMLDivElement> = (node) => {
		const shouldLoadMore = () =>
			window.scrollY > 0 || document.documentElement.scrollHeight <= window.innerHeight + 32;

		const observer = new IntersectionObserver(
			([entry]) => {
				if (entry?.isIntersecting && shouldLoadMore()) {
					void loadNextPage();
				}
			},
			{ rootMargin: '200px 0px' }
		);

		observer.observe(node);

		return {
			destroy() {
				observer.disconnect();
			}
		};
	};

	$effect(() => {
		type;
		stage;
		void loadNextPage(true);
	});
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<h1 class="text-2xl font-semibold tracking-tight">Detections</h1>
		<p class="text-sm text-muted-foreground">
			Review detections grouped by day and quickly play each clip.
		</p>
	</header>

	<div class="flex flex-col gap-2">
		{#if types.length > 0}
			<div class="flex flex-wrap gap-2">
				{#each [undefined, ...types] as t (t)}
					<Button
						type="button"
						size="sm"
						variant={t === type ? 'default' : 'outline'}
						aria-pressed={t === type}
						onclick={() => updateSearchParams(t, stage || undefined)}
					>
						{t ? capitalize(t) : 'All categories'}
					</Button>
				{/each}
			</div>
		{/if}
		{#if STAGES.length > 0}
			<div class="flex flex-wrap gap-2">
				{#each [undefined, ...STAGES] as s (s)}
					<Button
						type="button"
						size="sm"
						variant={s === stage ? 'default' : 'outline'}
						aria-pressed={s === stage}
						onclick={() => updateSearchParams(type || undefined, s)}
					>
						{s ? capitalize(s) : 'All stages'}
					</Button>
				{/each}
			</div>
		{/if}
	</div>

	{#if entries.length === 0 && isLoading}
		<h2 class="text-sm font-semibold text-muted-foreground">Loading detections...</h2>
	{:else if detectionsByDay.length === 0}
		<p class="text-sm font-semibold text-muted-foreground">No detections found.</p>
	{:else}
		<div class="space-y-8">
			{#each detectionsByDay as dayGroup (dayGroup[0])}
				<section class="space-y-3">
					<div class="flex items-center gap-2">
						<h2 class="text-sm font-semibold text-muted-foreground">
							{dayFormatter.format(new Date(dayGroup[0]))}
						</h2>
						<Badge variant="outline">{dayGroup[1].length}</Badge>
					</div>
					<div class="grid gap-2 lg:grid-cols-2 2xl:grid-cols-3">
						{#each dayGroup[1] as entry (`${entry.type}:${entry.stage}:${entry.locator}`)}
							<DetectionCard {entry} />
						{/each}
					</div>
				</section>
			{/each}
		</div>
	{/if}

	{#if errorMessage}
		<div class="flex items-center gap-3">
			<p class="text-sm font-semibold text-destructive">{errorMessage}</p>
			<Button type="button" size="sm" variant="outline" onclick={() => void loadNextPage()}>
				Retry
			</Button>
		</div>
	{/if}

	{#if entries.length > 0}
		<div use:infiniteTrigger class="flex min-h-16 items-center justify-center">
			{#if isLoading}
				<p class="text-sm font-semibold text-muted-foreground">Loading more detections...</p>
			{:else if !hasMore}
				<p class="text-sm font-semibold text-muted-foreground">You reached the end.</p>
			{/if}
		</div>
	{/if}
</section>
