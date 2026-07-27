<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
	import * as Table from '$lib/components/ui/table';
	import { getDetectors } from '$lib/remote/detector.remote';
	import { Plus } from '@lucide/svelte';

	const detectors = $state(await getDetectors());
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-semibold tracking-tight">Detectors</h1>
			<Button href="/setup/detectors/add" variant="outline"><Plus /> Add Detector</Button>
		</div>
		<p class="text-sm text-muted-foreground">Configure detectors.</p>
	</header>

	<Table.Root>
		<Table.Header>
			<Table.Row>
				<Table.Head>Name</Table.Head>
				<Table.Head>Model</Table.Head>
				<Table.Head>Streams</Table.Head>
			</Table.Row>
		</Table.Header>
		<Table.Body>
			{#each detectors as { detector, meta } (meta.label)}
				<Table.Row
					onclick={() =>
						goto(resolve(`/setup/detectors/add?label=${encodeURIComponent(meta.label)}`))}
				>
					<Table.Cell>{meta.label}</Table.Cell>
					<Table.Cell>{detector.yolo?.model?.split('/').pop() || ''}</Table.Cell>
					<Table.Cell>{detector.detection.source.length} stream(s)</Table.Cell>
				</Table.Row>
			{/each}
		</Table.Body>
	</Table.Root>
</section>
