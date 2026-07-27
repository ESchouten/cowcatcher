<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import type { WithoutChildren } from '$lib/utils';
	import type { ComponentProps } from 'svelte';
	import type { NavItem } from './types';

	let {
		title,
		items,
		size = 'default',
		...restProps
	}: {
		title: string;
		items: NavItem[];
		size?: 'lg' | 'default' | 'sm';
	} & WithoutChildren<ComponentProps<typeof Sidebar.Group>> = $props();

	const sidebar = Sidebar.useSidebar();

	function closeMobileSidebar() {
		if (sidebar.isMobile) sidebar.setOpenMobile(false);
	}
</script>

<!-- Menu URLs are either route-resolved by the layout or intentionally external support links. -->
<!-- eslint-disable svelte/no-navigation-without-resolve -->
<Sidebar.Group {...restProps}>
	<Sidebar.GroupLabel>{title}</Sidebar.GroupLabel>
	<Sidebar.Menu>
		{#each items as item (item.title)}
			<Sidebar.MenuItem>
				<Sidebar.MenuButton {size} isActive={item.active} tooltipContent={item.title}>
					{#snippet child({ props })}
						<a
							{...props}
							href={item.url}
							aria-current={item.active ? 'page' : undefined}
							onclick={closeMobileSidebar}
						>
							<item.icon />
							<span>{item.title}</span>
						</a>
					{/snippet}
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		{/each}
	</Sidebar.Menu>
</Sidebar.Group>
<!-- eslint-enable svelte/no-navigation-without-resolve -->
