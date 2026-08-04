<script lang="ts">
	import IdentityStatusPanel from '$lib/components/identity-status-panel.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import CCTVIcon from '@lucide/svelte/icons/cctv';
	import NavMain from './nav-main.svelte';
	import type { NavMenu } from './types';

	let {
		title,
		subtitle,
		homeUrl,
		menu,
		secondaryMenu = []
	}: {
		title: string;
		subtitle: string;
		homeUrl: string;
		menu: NavMenu[];
		secondaryMenu?: NavMenu[];
	} = $props();
</script>

<Sidebar.Root variant="inset" collapsible="icon">
	<Sidebar.Header>
		<Sidebar.Menu>
			<Sidebar.MenuItem>
				<Sidebar.MenuButton size="lg" tooltipContent={title}>
					{#snippet child({ props })}
						<!-- `homeUrl` is resolved by the route-aware layout before it reaches this component. -->
						<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
						<a href={homeUrl} {...props}>
							<span
								class="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground"
							>
								<CCTVIcon class="size-4" />
							</span>
							<span class="grid min-w-0 flex-1 text-start text-sm leading-tight">
								<span class="truncate font-medium">{title}</span>
								<span class="truncate text-xs">{subtitle}</span>
							</span>
						</a>
					{/snippet}
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>
	</Sidebar.Header>

	<Sidebar.Content>
		{#each menu as group (group.title)}
			<NavMain title={group.title} items={group.items} />
		{/each}

		<Sidebar.Group class="mt-auto group-data-[collapsible=icon]:hidden">
			<Sidebar.GroupLabel>Status</Sidebar.GroupLabel>
			<IdentityStatusPanel />
		</Sidebar.Group>
	</Sidebar.Content>

	{#if secondaryMenu.length > 0}
		<Sidebar.Footer>
			{#each secondaryMenu as group (group.title)}
				<NavMain title={group.title} items={group.items} size="sm" class="p-0" />
			{/each}
		</Sidebar.Footer>
	{/if}
	<Sidebar.Rail />
</Sidebar.Root>
