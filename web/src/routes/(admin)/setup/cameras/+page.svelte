<script lang="ts">
	import { resolve } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
	import { getStreams } from '$lib/remote/stream.remote';
	import ArrowLeftIcon from '@lucide/svelte/icons/arrow-left';
	import CameraIcon from '@lucide/svelte/icons/camera';
	import PlusIcon from '@lucide/svelte/icons/plus';

	const streams = await getStreams();
</script>

<section class="max-w-4xl space-y-6">
	<header class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
		<div class="space-y-1">
			<a
				href={resolve('/setup')}
				class="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground"
			>
				<ArrowLeftIcon class="size-3.5" /> Setup
			</a>
			<h2 class="text-2xl font-semibold tracking-tight">Cameras</h2>
			<p class="text-sm text-muted-foreground">Camera labels and detector source addresses.</p>
		</div>
		<Button href={resolve('/setup/cameras/add')}><PlusIcon /> Add camera</Button>
	</header>

	<div class="grid gap-3">
		{#each streams as stream (stream.source)}
			<a
				href={resolve(
					`/setup/cameras/add?source=${encodeURIComponent(stream.source)}&label=${encodeURIComponent(stream.label)}`
				)}
				class="flex items-center gap-3 rounded-md border p-3 transition-colors hover:bg-muted/50"
			>
				<span
					class="flex size-9 shrink-0 items-center justify-center rounded-md border bg-background text-muted-foreground"
				>
					<CameraIcon class="size-4" />
				</span>
				<span class="min-w-0 flex-1">
					<span class="block truncate text-sm font-medium">{stream.label}</span>
					<span class="block truncate text-xs text-muted-foreground">{stream.source}</span>
				</span>
			</a>
		{:else}
			<div class="rounded-md border border-dashed p-8 text-center">
				<p class="font-semibold">No cameras configured</p>
				<p class="mt-1 text-sm text-muted-foreground">Add the first source to begin.</p>
			</div>
		{/each}
	</div>
</section>
