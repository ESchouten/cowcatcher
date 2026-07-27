<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
	import * as Table from '$lib/components/ui/table';
	import { getTelegrams } from '$lib/remote/exporter.remote';
	import type { TelegramMeta } from '$lib/schema';
	import { Plus } from '@lucide/svelte';

	let telegrams = $state<TelegramMeta[]>(await getTelegrams());
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-semibold tracking-tight">Notifications</h1>
			<Button href="/setup/notifications/add" variant="outline"><Plus /> Add Telegram</Button>
		</div>
		<p class="text-sm text-muted-foreground">Configure notification channels for detections.</p>
	</header>

	<Table.Root>
		<Table.Header>
			<Table.Row>
				<Table.Head>Name</Table.Head>
				<Table.Head>Token</Table.Head>
				<Table.Head>Chat</Table.Head>
			</Table.Row>
		</Table.Header>
		<Table.Body>
			{#each telegrams as telegram (telegram.label)}
				<Table.Row
					onclick={() =>
						goto(
							resolve(
								`/setup/notifications/add?label=${encodeURIComponent(telegram.label)}&token=${encodeURIComponent(telegram.token)}&chat=${encodeURIComponent(telegram.chat)}`
							)
						)}
				>
					<Table.Cell>{telegram.label}</Table.Cell>
					<Table.Cell>{telegram.token}</Table.Cell>
					<Table.Cell>{telegram.chat}</Table.Cell>
				</Table.Row>
			{/each}
		</Table.Body>
	</Table.Root>
</section>
