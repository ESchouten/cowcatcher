<script lang="ts">
	import { page } from '$app/state';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { getDetectors } from '$lib/remote/detector.remote';
	import { getTelegrams } from '$lib/remote/exporter.remote';
	import { getStreams } from '$lib/remote/stream.remote';
	import ArrowRightIcon from '@lucide/svelte/icons/arrow-right';
	import BellIcon from '@lucide/svelte/icons/bell';
	import CheckIcon from '@lucide/svelte/icons/check';
	import CircleCheckIcon from '@lucide/svelte/icons/circle-check';
	import CircleIcon from '@lucide/svelte/icons/circle';
	import PlusIcon from '@lucide/svelte/icons/plus';
	import SparklesIcon from '@lucide/svelte/icons/sparkles';
	import TvIcon from '@lucide/svelte/icons/tv';
	import WrenchIcon from '@lucide/svelte/icons/wrench';
	import type { Component } from 'svelte';

	type Step = {
		title: string;
		description: string;
		icon: Component;
		status: 'complete' | 'recommended' | 'available';
		badge: string;
		count: number;
		countLabel: string;
		href: string;
		viewHref: string;
		action: string;
	};

	const streams = await getStreams();
	const telegrams = await getTelegrams();
	const detectors = await getDetectors();
	const complete = $derived(page.url.searchParams.has('complete'));
	const hasStreams = $derived(streams.length > 0);
	const hasTelegrams = $derived(telegrams.length > 0);
	const hasDetectors = $derived(detectors.length > 0);
	const setupDone = $derived(hasStreams && hasDetectors);
	const steps = $derived<Step[]>([
		{
			title: 'Streams',
			description: 'Camera labels and RTSP sources.',
			icon: TvIcon,
			status: hasStreams ? 'complete' : 'recommended',
			badge: hasStreams ? 'Done' : 'Recommended',
			count: streams.length,
			countLabel: streams.length === 1 ? 'stream' : 'streams',
			href: '/setup/cameras/add',
			viewHref: '/setup/cameras',
			action: hasStreams ? 'Add another' : 'Add camera'
		},
		{
			title: 'Telegram',
			description: 'Optional bot token and chat channel for alerts.',
			icon: BellIcon,
			status: hasTelegrams ? 'complete' : 'available',
			badge: hasTelegrams ? 'Done' : 'Optional',
			count: telegrams.length,
			countLabel: telegrams.length === 1 ? 'channel' : 'channels',
			href: '/setup/notifications/add',
			viewHref: '/setup/notifications',
			action: hasTelegrams ? 'Add another' : 'Add Telegram'
		},
		{
			title: 'Detector',
			description: 'Streams, alerts, model, and thresholds.',
			icon: WrenchIcon,
			status: hasDetectors ? 'complete' : hasStreams ? 'recommended' : 'available',
			badge: hasDetectors ? 'Done' : hasStreams ? 'Recommended' : 'Available',
			count: detectors.length,
			countLabel: detectors.length === 1 ? 'detector' : 'detectors',
			href: '/setup/detectors/add?setup=1',
			viewHref: '/setup/detectors',
			action: hasDetectors ? 'Add another' : 'Add detector'
		}
	]);
	const primaryHref = $derived(setupDone ? '/setup/detectors' : '/setup/wizard');
	const primaryLabel = $derived(setupDone ? 'View detectors' : 'Start guided setup');
	const primaryIcon = $derived(setupDone ? CheckIcon : SparklesIcon);
	const CurrentIcon = $derived(setupDone ? CircleCheckIcon : SparklesIcon);
	const PrimaryIcon = $derived(primaryIcon);
</script>

<section class="max-w-5xl space-y-6">
	<header class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
		<div class="space-y-1">
			<h1 class="text-2xl font-semibold tracking-tight">Setup</h1>
			<p class="text-sm text-muted-foreground">
				Use guided setup for a new installation, or manage individual parts below.
			</p>
		</div>
		<div class="flex flex-wrap gap-2">
			<Badge variant={hasStreams ? 'default' : 'secondary'}>{streams.length} streams</Badge>
			<Badge variant={hasTelegrams ? 'default' : 'secondary'}>{telegrams.length} telegrams</Badge>
			<Badge variant={hasDetectors ? 'default' : 'secondary'}>{detectors.length} detectors</Badge>
		</div>
	</header>

	{#if complete}
		<div class="flex items-start gap-3 rounded-md border p-4">
			<CircleCheckIcon class="mt-0.5 size-5 text-primary" />
			<div class="space-y-1">
				<h2 class="font-medium">Setup saved</h2>
				<p class="text-sm text-muted-foreground">
					The detector will reload the new configuration automatically.
				</p>
			</div>
		</div>
	{/if}

	<section class="rounded-md border">
		<div class="grid gap-4 p-4 lg:grid-cols-[1fr_auto] lg:items-center">
			<div class="flex min-w-0 gap-3">
				<div
					class="flex size-10 shrink-0 items-center justify-center rounded-md border bg-background"
				>
					<CurrentIcon class="size-5" />
				</div>
				<div class="min-w-0 space-y-1">
					<div class="flex flex-wrap items-center gap-2">
						<h2 class="text-lg font-medium">
							{setupDone ? 'Configuration is ready' : 'Guided setup'}
						</h2>
						<Badge variant={setupDone ? 'default' : 'outline'}>
							{setupDone ? 'Ready' : 'Recommended'}
						</Badge>
					</div>
					<p class="text-sm text-muted-foreground">
						{setupDone
							? 'Review the detector or add more cameras, Telegram channels, and detectors as needed.'
							: 'Find ONVIF cameras, choose detection settings, and start the application in a few steps.'}
					</p>
				</div>
			</div>
			<div class="flex flex-wrap gap-2 lg:justify-end">
				<Button href={primaryHref}>
					<PrimaryIcon />
					{primaryLabel}
					<ArrowRightIcon />
				</Button>
			</div>
		</div>
	</section>

	<section class="space-y-2">
		{#each steps as step, index (step.title)}
			<div class="grid gap-3 rounded-md border p-4 md:grid-cols-[2rem_1fr_auto] md:items-center">
				<div
					class="flex size-8 items-center justify-center rounded-md border bg-background"
					class:border-primary={step.status === 'complete'}
				>
					{#if step.status === 'complete'}
						<CheckIcon class="size-4" />
					{:else if step.status === 'recommended'}
						<step.icon class="size-4" />
					{:else}
						<CircleIcon class="size-4" />
					{/if}
				</div>
				<div class="min-w-0 space-y-1">
					<div class="flex flex-wrap items-center gap-2">
						<h2 class="font-medium">{index + 1}. {step.title}</h2>
						<Badge
							variant={step.status === 'complete'
								? 'default'
								: step.status === 'recommended'
									? 'outline'
									: 'secondary'}
						>
							{step.badge}
						</Badge>
						<span class="text-xs text-muted-foreground">
							{step.count}
							{step.countLabel}
						</span>
					</div>
					<p class="text-sm text-muted-foreground">{step.description}</p>
				</div>
				<div class="flex flex-wrap gap-2 md:justify-end">
					<Button href={step.viewHref} variant="outline">View</Button>
					<Button href={step.href}>
						<PlusIcon />
						{step.action}
					</Button>
				</div>
			</div>
		{/each}
	</section>
</section>
