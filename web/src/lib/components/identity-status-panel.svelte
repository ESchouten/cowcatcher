<script lang="ts">
	import { getDetectorConnection, getIdentityStatus } from '$lib/remote/identity.remote';
	import CircleAlertIcon from '@lucide/svelte/icons/circle-alert';
	import CircleCheckIcon from '@lucide/svelte/icons/circle-check';
	import DatabaseIcon from '@lucide/svelte/icons/database';
	import LayersIcon from '@lucide/svelte/icons/layers-3';
	import RadioIcon from '@lucide/svelte/icons/radio';

	const statusQuery = getIdentityStatus();
	const detectorQuery = getDetectorConnection();
	const statusData = $derived(await Promise.all([statusQuery, detectorQuery]));
	const catalogs = $derived(statusData[0]);
	const catalog = $derived(catalogs[0] ?? null);
	const detectorConnection = $derived(statusData[1]);

	$effect(() => {
		const timer = window.setInterval(() => {
			void statusQuery.refresh();
			void detectorQuery.refresh();
		}, 5_000);
		return () => window.clearInterval(timer);
	});

	const detectorLabel = $derived(
		detectorConnection === 'connected'
			? 'Connected'
			: detectorConnection === 'unconfigured'
				? 'Not configured'
				: 'Disconnected'
	);
	const databaseLabel = $derived(
		!catalog
			? 'Not configured'
			: catalog.state === 'ready'
				? 'Ready'
				: catalog.state === 'not_initialized'
					? 'Not initialized'
					: 'Error'
	);
</script>

<section
	class="overflow-hidden rounded-md border border-sidebar-border bg-sidebar"
	aria-label="System status"
	data-testid="identity-status-panel"
>
	<div class="flex min-w-0 items-center gap-2 border-b border-sidebar-border px-2 py-2">
		<RadioIcon class="size-3.5 shrink-0 text-muted-foreground" />
		<span
			class:bg-emerald-500={detectorConnection === 'connected'}
			class:bg-amber-500={detectorConnection !== 'connected'}
			class="size-1.5 shrink-0 rounded-full"
		></span>
		<span class="min-w-0 flex-1 truncate text-xs text-muted-foreground">Detector</span>
		<span class="truncate text-xs font-medium" data-testid="detector-connection">
			{detectorLabel}
		</span>
	</div>

	<div class="flex min-w-0 items-center gap-2 border-b border-sidebar-border px-2 py-2">
		<DatabaseIcon class="size-3.5 shrink-0 text-muted-foreground" />
		<span
			class:bg-emerald-500={catalog?.state === 'ready'}
			class:bg-amber-500={catalog?.state !== 'ready'}
			class="size-1.5 shrink-0 rounded-full"
		></span>
		<span class="min-w-0 flex-1 truncate text-xs text-muted-foreground">Database</span>
		<span
			class="truncate text-xs font-medium"
			title={catalog?.message ?? undefined}
			data-testid="database-state"
		>
			{databaseLabel}
		</span>
	</div>

	<div class="flex min-w-0 items-center gap-2 border-b border-sidebar-border px-2 py-2">
		<LayersIcon class="size-3.5 shrink-0 text-muted-foreground" />
		<span class="min-w-0 flex-1 truncate text-xs text-muted-foreground">Gallery</span>
		<span class="truncate text-xs font-medium" data-testid="gallery-version">
			{catalog?.activeGalleryVersion ? `v${catalog.activeGalleryVersion}` : 'Not active'}
		</span>
	</div>

	<div class="flex min-w-0 items-center gap-2 px-2 py-2">
		{#if catalog?.lastIdentityError}
			<CircleAlertIcon class="size-3.5 shrink-0 text-destructive" />
		{:else}
			<CircleCheckIcon class="size-3.5 shrink-0 text-emerald-600" />
		{/if}
		<span class="min-w-0 flex-1 truncate text-xs text-muted-foreground"> Last identity error </span>
		<span
			class:text-destructive={Boolean(catalog?.lastIdentityError)}
			class="max-w-24 truncate text-xs font-medium"
			title={catalog?.lastIdentityError ?? undefined}
			data-testid="last-identity-error"
		>
			{catalog?.lastIdentityError ?? 'None'}
		</span>
	</div>
</section>
