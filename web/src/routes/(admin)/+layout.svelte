<script lang="ts">
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import AppSidebar from '$lib/components/app-sidebar.svelte';
	import type { NavMenu } from '$lib/components/types';
	import * as Breadcrumb from '$lib/components/ui/breadcrumb/index.js';
	import { Separator } from '$lib/components/ui/separator/index.js';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { version } from '$lib/version';
	import CameraIcon from '@lucide/svelte/icons/camera';
	import GithubIcon from '@lucide/svelte/icons/github';
	import RadioIcon from '@lucide/svelte/icons/radio';
	import ScanFaceIcon from '@lucide/svelte/icons/scan-face';
	import SettingsIcon from '@lucide/svelte/icons/settings-2';
	import type { Component } from 'svelte';

	let { children } = $props();

	type NavigationPath = '/live' | '/detections' | '/identities' | '/setup';

	type NavigationItem = {
		title: string;
		url: NavigationPath;
		icon: Component;
	};

	const navigation: NavigationItem[] = [
		{ title: 'Live', url: '/live', icon: RadioIcon },
		{ title: 'Detections', url: '/detections', icon: CameraIcon },
		{ title: 'Identities', url: '/identities', icon: ScanFaceIcon },
		{ title: 'Setup', url: '/setup', icon: SettingsIcon }
	];

	function active(url: string): boolean {
		return page.url.pathname === url || page.url.pathname.startsWith(`${url}/`);
	}

	const menu = $derived<NavMenu[]>([
		{
			title: 'Overview',
			items: navigation
				.filter((item) => item.url !== '/setup')
				.map((item) => ({
					...item,
					url: resolve(item.url),
					active: active(item.url)
				}))
		},
		{
			title: 'Settings',
			items: navigation
				.filter((item) => item.url === '/setup')
				.map((item) => ({
					...item,
					url: resolve(item.url),
					active: active(item.url)
				}))
		}
	]);
	const secondaryMenu: NavMenu[] = [
		{
			title: 'Support',
			items: [
				{
					title: 'GitHub',
					url: 'https://github.com/ESchouten/ai-detector',
					icon: GithubIcon
				}
			]
		}
	];
	const breadcrumbSegments = $derived(page.url.pathname.split('/').filter(Boolean));

	function breadcrumbLabel(segment: string): string {
		return segment
			.split('-')
			.map((part) => part.charAt(0).toLocaleUpperCase() + part.slice(1))
			.join(' ');
	}
</script>

<Sidebar.Provider>
	<AppSidebar
		title="AI Detector"
		subtitle={version}
		homeUrl={resolve('/live')}
		{menu}
		{secondaryMenu}
	/>
	<Sidebar.Inset data-testid="app-shell">
		<header class="flex h-16 shrink-0 items-center gap-2">
			<div class="flex min-w-0 items-center gap-2 px-4">
				<Sidebar.Trigger class="-ms-1" />
				<Separator orientation="vertical" class="me-2 data-[orientation=vertical]:h-4" />
				<Breadcrumb.Root>
					<Breadcrumb.List>
						<Breadcrumb.Item class="hidden md:block">
							<Breadcrumb.Link href={resolve('/live')}>AI Detector</Breadcrumb.Link>
						</Breadcrumb.Item>
						{#each breadcrumbSegments as segment, index (`${index}:${segment}`)}
							<Breadcrumb.Separator class="hidden md:block" />
							<Breadcrumb.Item>
								<Breadcrumb.Page>{breadcrumbLabel(segment)}</Breadcrumb.Page>
							</Breadcrumb.Item>
						{/each}
					</Breadcrumb.List>
				</Breadcrumb.Root>
			</div>
		</header>
		<div class="flex flex-1 flex-col gap-4 p-4 pt-0">
			{@render children()}
		</div>
	</Sidebar.Inset>
</Sidebar.Provider>
