<script lang="ts">
	import { page } from '$app/state';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { tick } from 'svelte';
	import LiveStream from '$lib/components/live-stream.svelte';
	import { deleteStream, saveStream } from '$lib/remote/stream.remote';
	import { resolve } from '$app/paths';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';

	let originalSource = $state(page.url.searchParams.get('source') ?? '');
	let label = $state(page.url.searchParams.get('label') ?? '');
	let source = $state(originalSource);
	const setupMode = $derived(page.url.searchParams.get('setup') === '1');

	let test = $state(originalSource);

	async function setTest(url: string) {
		test = '';
		await tick();
		test = url;
	}
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<h1 class="text-2xl font-semibold tracking-tight">
			{setupMode ? 'Setup: Add stream' : 'Add stream'}
		</h1>
		<p class="text-sm text-muted-foreground">
			{setupMode
				? 'Add a camera source, then return to setup when you are done.'
				: 'Add a new live stream to your system.'}
		</p>
	</header>

	<div class="flex justify-between gap-6">
		<form
			{...saveStream.enhance(async ({ submit }) => {
				try {
					await submit();
					toast.info('Saved!');
				} catch {
					toast.error('Something went wrong');
				}
			})}
			class="flex w-lg flex-col gap-2"
		>
			<Input type="hidden" name="original" value={originalSource} />
			<Label for="label">Label</Label>
			<Input id="label" name="label" bind:value={label} placeholder="e.g. Front door" />
			<Label class="mt-2" for="source">Source</Label>
			<div class="flex gap-6">
				<Input
					id="source"
					name="source"
					bind:value={source}
					placeholder="e.g. rtsp://[USER]:[PASSWORD]@[IP_ADDRESS]/h264Preview_01_main"
				/>
				<Button variant="outline" onclick={() => setTest(source)}>Test</Button>
			</div>
			{#if setupMode && !originalSource}
				<div class="mt-2 flex gap-2">
					<Button
						type="submit"
						name="next"
						value="/setup/cameras/add"
						variant="outline"
						class="flex-1">Save and add another</Button
					>
					<Button type="submit" name="next" value="/setup" class="flex-1"
						>Save and return to setup</Button
					>
				</div>
			{:else}
				<div class="mt-2 flex gap-6">
					{#if originalSource}
						<Button
							onclick={() =>
								deleteStream({ source: originalSource }).then(() =>
									goto(resolve('/setup/cameras'))
								)}
							variant="destructive"
							class="flex-1">Delete</Button
						>
					{/if}
					<Button type="submit" class="flex-1">Save</Button>
				</div>
			{/if}
		</form>
		{#if test}
			<div class="flex max-w-lg">
				<LiveStream {label} source={test} />
			</div>
		{/if}
	</div>
</section>
