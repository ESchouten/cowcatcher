<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { deleteTelegram, saveTelegram, testTelegram } from '$lib/remote/exporter.remote';
	import { toast } from 'svelte-sonner';

	let originalLabel = $state(page.url.searchParams.get('label') ?? '');
	let label = $state(originalLabel);
	let token = $state(page.url.searchParams.get('token') ?? '');
	let chat = $state(page.url.searchParams.get('chat') ?? '');
	const setupMode = $derived(page.url.searchParams.get('setup') === '1');
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<h1 class="text-2xl font-semibold tracking-tight">
			{setupMode ? 'Setup: Add Telegram' : 'Add Telegram'}
		</h1>
		<p class="text-sm text-muted-foreground">
			{setupMode
				? 'Add a bot and channel for detector alerts.'
				: 'Add a new Telegram notification channel.'}
		</p>
	</header>

	<div class="flex justify-between gap-6">
		<form
			{...saveTelegram.enhance(async ({ submit }) => {
				try {
					await submit();
					toast.info('Saved!');
				} catch {
					toast.error('Something went wrong');
				}
			})}
			class="flex w-lg flex-col gap-2"
		>
			<Input type="hidden" name="original" value={originalLabel} />
			<Label for="label">Label</Label>
			<Input id="label" name="label" bind:value={label} placeholder="e.g. Groupchat X" />
			<Label class="mt-2" for="token">Token</Label>
			<div class="flex gap-2">
				<Input
					id="token"
					name="token"
					bind:value={token}
					placeholder="e.g. 1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ"
				/>
			</div>
			<Label class="mt-2" for="chat">Chat ID</Label>
			<div class="flex gap-6">
				<Input id="chat" name="chat" bind:value={chat} placeholder="e.g. 1234567890" />
				<Button variant="outline" onclick={() => testTelegram({ token, chat })}
					>Test notification</Button
				>
			</div>
			{#if setupMode && !originalLabel}
				<div class="mt-2 flex gap-2">
					<Button
						type="submit"
						name="next"
						value="/setup/notifications/add"
						variant="outline"
						class="flex-1">Save and add another</Button
					>
					<Button type="submit" name="next" value="/setup" class="flex-1"
						>Save and return to setup</Button
					>
				</div>
			{:else}
				<div class="mt-2 flex gap-6">
					{#if originalLabel}
						<Button
							onclick={() =>
								deleteTelegram({ label: originalLabel }).then(() =>
									goto(resolve('/setup/notifications'))
								)}
							variant="destructive"
							class="flex-1">Delete</Button
						>
					{/if}
					<Button type="submit" class="flex-1">Save</Button>
				</div>
			{/if}
		</form>
	</div>
</section>
