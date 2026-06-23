<script lang="ts">
	import { page } from '$app/state';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { getDetectors } from '$lib/remote/detector.remote';
	import { getTelegrams } from '$lib/remote/exporter.remote';
	import { getIdentities } from '$lib/remote/identity.remote';
	import { getStreams } from '$lib/remote/stream.remote';
	import ArrowRightIcon from '@lucide/svelte/icons/arrow-right';
	import BellIcon from '@lucide/svelte/icons/bell';
	import CheckIcon from '@lucide/svelte/icons/check';
	import CircleCheckIcon from '@lucide/svelte/icons/circle-check';
	import CircleIcon from '@lucide/svelte/icons/circle';
	import FingerprintIcon from '@lucide/svelte/icons/fingerprint';
	import PlusIcon from '@lucide/svelte/icons/plus';
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
	const identities = await getIdentities();
	const detectors = await getDetectors();
	const complete = $derived(page.url.searchParams.has('complete'));
	const hasStreams = $derived(streams.length > 0);
	const hasTelegrams = $derived(telegrams.length > 0);
	const hasIdentities = $derived(identities.length > 0);
	const hasDetectors = $derived(detectors.length > 0);
	const setupDone = $derived(hasStreams && hasDetectors);
	const nextStep = $derived(!hasStreams ? 'streams' : !hasDetectors ? 'detector' : 'done');
	const steps = $derived<Step[]>([
		{
			title: 'Streams',
			description: 'Camera labels and RTSP sources.',
			icon: TvIcon,
			status: hasStreams ? 'complete' : 'recommended',
			badge: hasStreams ? 'Done' : 'Recommended',
			count: streams.length,
			countLabel: streams.length === 1 ? 'stream' : 'streams',
			href: '/streams/add?setup=1',
			viewHref: '/streams',
			action: hasStreams ? 'Add another' : 'Add stream'
		},
		{
			title: 'Telegram',
			description: 'Optional bot token and chat channel for alerts.',
			icon: BellIcon,
			status: hasTelegrams ? 'complete' : 'available',
			badge: hasTelegrams ? 'Done' : 'Optional',
			count: telegrams.length,
			countLabel: telegrams.length === 1 ? 'channel' : 'channels',
			href: '/notifications/add?setup=1',
			viewHref: '/notifications',
			action: hasTelegrams ? 'Add another' : 'Add Telegram'
		},
		{
			title: 'Identities',
			description: 'Optional identity database and matching model.',
			icon: FingerprintIcon,
			status: hasIdentities ? 'complete' : 'available',
			badge: hasIdentities ? 'Done' : 'Optional',
			count: identities.length,
			countLabel: identities.length === 1 ? 'identity' : 'identities',
			href: '/identities/add?setup=1',
			viewHref: '/identities',
			action: hasIdentities ? 'Add another' : 'Add identity'
		},
		{
			title: 'Detector',
			description: 'Streams, alerts, identity, model, and thresholds.',
			icon: WrenchIcon,
			status: hasDetectors ? 'complete' : hasStreams ? 'recommended' : 'available',
			badge: hasDetectors ? 'Done' : hasStreams ? 'Recommended' : 'Available',
			count: detectors.length,
			countLabel: detectors.length === 1 ? 'detector' : 'detectors',
			href: '/detectors/add?setup=1',
			viewHref: '/detectors',
			action: hasDetectors ? 'Add another' : 'Add detector'
		}
	]);
	const current = $derived(
		steps.find((step) => step.status === 'recommended') ?? steps[steps.length - 1]
	);
	const primaryHref = $derived(
		nextStep === 'done'
			? '/detectors'
			: nextStep === 'streams'
				? '/streams/add?setup=1'
				: '/detectors/add?setup=1'
	);
	const primaryLabel = $derived(
		nextStep === 'done'
			? 'View detectors'
			: nextStep === 'streams'
				? 'Add first stream'
				: 'Add detector'
	);
	const primaryIcon = $derived(
		nextStep === 'done' ? CheckIcon : nextStep === 'streams' ? TvIcon : WrenchIcon
	);
	const CurrentIcon = $derived(setupDone ? CircleCheckIcon : current.icon);
	const PrimaryIcon = $derived(primaryIcon);
</script>

<section class="max-w-5xl space-y-6">
	<header class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
		<div class="space-y-1">
			<h1 class="text-2xl font-semibold tracking-tight">Setup</h1>
			<p class="text-sm text-muted-foreground">
				Open the existing setup screens from one place and complete them in the order you need.
			</p>
		</div>
		<div class="flex flex-wrap gap-2">
			<Badge variant={hasStreams ? 'default' : 'secondary'}>{streams.length} streams</Badge>
			<Badge variant={hasTelegrams ? 'default' : 'secondary'}>{telegrams.length} telegrams</Badge>
			<Badge variant={hasIdentities ? 'default' : 'secondary'}>{identities.length} identities</Badge
			>
			<Badge variant={hasDetectors ? 'default' : 'secondary'}>{detectors.length} detectors</Badge>
		</div>
	</header>

	{#if complete}
		<div class="flex items-start gap-3 rounded-md border p-4">
			<CircleCheckIcon class="mt-0.5 size-5 text-primary" />
			<div class="space-y-1">
				<h2 class="font-medium">Setup saved</h2>
				<p class="text-sm text-muted-foreground">
					Restart the detector service for the new configuration to take effect.
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
							{setupDone ? 'Configuration is ready' : `Recommended: ${current.title}`}
						</h2>
						<Badge variant={setupDone ? 'default' : 'outline'}>
							{setupDone ? 'Ready' : 'Recommended'}
						</Badge>
					</div>
					<p class="text-sm text-muted-foreground">
						{setupDone
							? 'Review the detector or add more cameras, Telegram channels, identities, and detectors as needed.'
							: current.description}
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
