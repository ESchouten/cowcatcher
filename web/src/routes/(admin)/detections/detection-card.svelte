<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import CardOverlay from '$lib/components/card-overlay.svelte';
	import type { DetectionMetadata } from '$lib/schema';

	const stageLabels = {
		approved: 'Approved',
		rejected: 'Rejected',
		unvalidated: 'Unvalidated'
	} as const;

	const stageBadgeVariants = {
		approved: 'default',
		rejected: 'destructive',
		unvalidated: 'secondary'
	} as const;

	const stageBadgeClasses = {
		approved: 'bg-emerald-600 text-white',
		rejected: '',
		unvalidated: 'bg-amber-500 text-black'
	} as const;

	const overlayBadgeClass = 'bg-black/50 text-white';

	type Props = {
		entry: DetectionMetadata;
	};

	let { entry }: Props = $props();
	let isPlaying = $state(false);

	function getResource(resource: string) {
		return `/detections/${[entry.type, entry.stage, entry.locator, resource]
			.map((segment) => encodeURIComponent(segment))
			.join('/')}`;
	}

	function capitalize(value: string) {
		return value.charAt(0).toUpperCase() + value.slice(1);
	}

	function formatTime(value: string) {
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) {
			return value;
		}
		return new Intl.DateTimeFormat(undefined, { timeStyle: 'medium' }).format(date);
	}
</script>

<CardOverlay hide={isPlaying}>
	<video
		class="block h-auto w-full bg-black"
		controls
		preload="none"
		poster={getResource('best.jpg')}
		onplay={() => (isPlaying = true)}
		onpause={() => (isPlaying = false)}
		onended={() => (isPlaying = false)}
	>
		<source src={getResource('video.mp4')} type="video/mp4" />
		Your browser cannot play this video.
	</video>

	{#snippet overlay()}
		<div class="flex flex-wrap items-center gap-2 text-xs">
			<Badge variant={stageBadgeVariants[entry.stage]} class={stageBadgeClasses[entry.stage]}>
				{stageLabels[entry.stage]}
			</Badge>
			<Badge variant="secondary" class={overlayBadgeClass}>
				{capitalize(entry.type)}
			</Badge>
			<Badge variant="secondary" class={overlayBadgeClass}>Time: {formatTime(entry.start)}</Badge>
			<Badge variant="secondary" class={overlayBadgeClass}
				>Duration: {entry.duration.toFixed(2)}s</Badge
			>
			<Badge variant="secondary" class={overlayBadgeClass}>
				Confidence: {(entry.confidence * 100).toFixed(1)}%
			</Badge>
		</div>
	{/snippet}
</CardOverlay>
