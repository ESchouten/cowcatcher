<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { Button } from '$lib/components/ui/button';
	import JsonEditor from '$lib/components/json-editor.svelte';
	import type { DetectorConfig, SSEConfig, TelegramConfig } from '$lib/schema';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import {
		deleteDetector,
		getDetector,
		getDetectorPreset,
		getDetectorPresets,
		getDetectorSchema,
		saveDetector
	} from '$lib/remote/detector.remote';
	import { toast } from 'svelte-sonner';
	import * as Select from '$lib/components/ui/select';
	import { getStreams } from '$lib/remote/stream.remote';
	import { getTelegrams, testTelegram } from '$lib/remote/exporter.remote';
	import Stream from '../../streams/stream.svelte';
	import { Plus } from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { SvelteSet } from 'svelte/reactivity';

	const EMPTY_DETECTOR = {
		detection: {
			source: []
		},
		yolo: {
			model: '',
			confidence: 0.8
		},
		exporters: { sse: [{}] } as { telegram?: TelegramConfig[]; sse?: SSEConfig[] }
	};
	const EMPTY_DAZZLECOW = {
		model: 'hf-hub:BVRA/MegaDescriptor-S-224',
		enrollment: {
			database: 'identities/cows.sqlite',
			identity_count: undefined as number | undefined
		},
		owl_interval: 1,
		confidence: 0.3,
		match_threshold: 0.75,
		match_margin: 0,
		margin: 0.2,
		track_samples: 5,
		frames_min: 1
	};
	const INLINE_STREAM_PREVIEW_LIMIT = 5;

	function mergeWithEmptyDetector(detector?: Partial<DetectorConfig>) {
		const merged = {
			...EMPTY_DETECTOR,
			...detector,
			detection: {
				...EMPTY_DETECTOR.detection,
				...detector?.detection
			},
			yolo: detector?.dazzlecow
				? undefined
				: {
						...EMPTY_DETECTOR.yolo,
						...detector?.yolo
					},
			dazzlecow: detector?.dazzlecow
				? {
						...EMPTY_DAZZLECOW,
						...detector.dazzlecow,
						enrollment: detector.dazzlecow.enrollment
							? { ...EMPTY_DAZZLECOW.enrollment, ...detector.dazzlecow.enrollment }
							: undefined
					}
				: undefined,
			exporters: {
				...EMPTY_DETECTOR.exporters,
				...detector?.exporters
			}
		} as DetectorConfig;
		return merged;
	}

	const originalLabel = page.url.searchParams.get('label') ?? '';
	const isEditing = !!originalLabel;
	const setupMode = page.url.searchParams.get('setup') === '1';
	const detectorPresets = $state(await getDetectorPresets());
	const detectorSchema = $state(await getDetectorSchema());
	const streams = $derived(await getStreams());
	const telegrams = $derived(await getTelegrams());
	const initialDetector = mergeWithEmptyDetector(
		isEditing ? (await getDetector({ label: originalLabel }))?.detector : undefined
	);

	let label = $state(originalLabel);
	let detector = $state(initialDetector);
	let modelType = $state<'yolo' | 'dazzlecow'>(initialDetector.dazzlecow ? 'dazzlecow' : 'yolo');
	let visiblePreviewSources = new SvelteSet<string>();
	let editorHasErrors = $state(false);
	let preset = $state<string>('Custom');
	let advanced = $state(false);
	const activePreviewSources = $derived(
		new Set(
			streams
				.filter((stream) => isRtspStream(stream.source) && visiblePreviewSources.has(stream.source))
				.slice(0, INLINE_STREAM_PREVIEW_LIMIT)
				.map((stream) => stream.source)
		)
	);

	function isRtspStream(source: string) {
		return source.startsWith('rtsp://') || source.startsWith('rtsps://');
	}

	function setPreviewVisibility(source: string, visible: boolean) {
		if (visiblePreviewSources.has(source) === visible) {
			return;
		}

		if (visible) {
			visiblePreviewSources.add(source);
		} else {
			visiblePreviewSources.delete(source);
		}
	}

	function trackPreviewVisibility(node: HTMLElement, source: string) {
		const observer = new IntersectionObserver(
			([entry]) => {
				setPreviewVisibility(source, entry.isIntersecting);
			},
			{ threshold: 0.25 }
		);
		observer.observe(node);

		return {
			destroy() {
				observer.disconnect();
				setPreviewVisibility(source, false);
			}
		};
	}

	async function handlePresetChange(file: string) {
		const preset = await getDetectorPreset({ file });
		detector = mergeWithEmptyDetector(preset);
		modelType = detector.dazzlecow ? 'dazzlecow' : 'yolo';
	}

	function setModelType(value: string) {
		modelType = value === 'dazzlecow' ? 'dazzlecow' : 'yolo';
		if (modelType === 'dazzlecow') {
			delete detector.yolo;
			detector.dazzlecow ??= { ...EMPTY_DAZZLECOW };
		} else {
			delete detector.dazzlecow;
			detector.yolo ??= { ...EMPTY_DETECTOR.yolo };
		}
	}

	function setDazzleCowMode(value: string) {
		if (!detector.dazzlecow) return;
		if (value === 'enrollment') {
			delete detector.dazzlecow.gallery;
			detector.dazzlecow.enrollment ??= { ...EMPTY_DAZZLECOW.enrollment };
		} else {
			delete detector.dazzlecow.enrollment;
			detector.dazzlecow.gallery ??= 'identities/cows.npz';
		}
	}

	async function handleSave(event: SubmitEvent) {
		event.preventDefault();
		if (editorHasErrors) {
			return;
		}
		detector.exporters.telegram?.forEach((telegram) => {
			if (telegram.alert_every && telegram.alert_every <= 1) {
				delete telegram.alert_every;
			}
		});
		if (detector.dazzlecow?.enrollment) {
			delete detector.dazzlecow.gallery;
			if (!detector.dazzlecow.enrollment.identity_count) {
				delete detector.dazzlecow.enrollment.identity_count;
			}
		}

		await saveDetector({
			original: originalLabel || undefined,
			detector,
			meta: { label }
		});
		toast.warning(
			`Detector configuration '${label}' saved. Restart the detector to apply the changes.`,
			{ duration: Number.POSITIVE_INFINITY, closeButton: true }
		);
		await goto(resolve(setupMode ? '/setup?complete=1' : '/detectors'));
	}

	function getPresetLabel(presetFile: string) {
		return presetFile
			.replace(/\.json$/i, '')
			.split('-')
			.filter(Boolean)
			.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
			.join(' ');
	}
</script>

<svelte:document
	onvisibilitychange={() =>
		document.visibilityState === 'visible' && (getStreams().refresh(), getTelegrams().refresh())}
/>

<section class="space-y-6">
	<header class="space-y-1">
		<h1 class="text-2xl font-semibold tracking-tight">
			{setupMode ? 'Setup: Add Detector' : isEditing ? 'Edit Detector' : 'Add Detector'}
		</h1>
		<p class="text-sm text-muted-foreground">
			{setupMode ? 'Select the streams and alerts for the detector.' : 'Configure a detector.'}
		</p>
	</header>

	<form class="flex max-w-2xl flex-col gap-2" onsubmit={handleSave}>
		<div class="flex gap-6">
			<div class="flex flex-1 flex-col gap-2">
				<Label for="label">Label</Label>
				<Input id="label" name="label" bind:value={label} placeholder="e.g. Detector X" />
			</div>

			<div class="flex flex-1 flex-col gap-2">
				<Label for="presets">Presets</Label>
				<Select.Root
					type="single"
					bind:value={preset}
					onValueChange={handlePresetChange}
					items={['Custom', ...detectorPresets].map((preset) => ({
						value: preset,
						label: getPresetLabel(preset)
					}))}
				>
					<Select.Trigger id="presets" class="w-full">
						{getPresetLabel(preset)}
					</Select.Trigger>
					<Select.Content>
						{#each detectorPresets as presetFile (presetFile)}
							<Select.Item value={presetFile} label={getPresetLabel(presetFile)}></Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>
			</div>
		</div>

		<Label for="streams" class="mt-2">Streams</Label>
		<div class="flex gap-6">
			<Select.Root
				type="multiple"
				bind:value={detector.detection.source}
				items={streams.map((stream) => ({
					value: stream.source,
					label: stream.label
				}))}
			>
				<Select.Trigger id="streams" class="w-full">
					{detector.detection.source.length
						? `${detector.detection.source.length} stream${detector.detection.source.length === 1 ? '' : 's'} selected`
						: 'Select streams'}
				</Select.Trigger>
				<Select.Content>
					{#each streams as source (source.source)}
						<Select.Item value={source.source} label={source.label ?? source.source} class="gap-6">
							<div class="w-xs">
								{#if isRtspStream(source.source)}
									<div use:trackPreviewVisibility={source.source}>
										{#if activePreviewSources.has(source.source)}
											<Stream label={source.label} source={source.source} disableLink hideOverlay />
										{:else}
											<div class="relative aspect-video w-full bg-black">
												<div
													class="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-white/60"
												>
													Preview paused
												</div>
											</div>
										{/if}
									</div>
								{/if}
							</div>
							<div class="flex flex-col">
								<span>{source.label}</span>
								<span class="text-xs text-muted-foreground">{source.source}</span>
							</div>
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
			<Button
				target="_blank"
				href={setupMode ? '/streams/add?setup=1' : '/streams/add'}
				variant="outline"><Plus /></Button
			>
		</div>

		<Label for="telegrams" class="mt-2">Telegram</Label>
		<div class="flex gap-6">
			<Select.Root
				type="multiple"
				bind:value={
					() =>
						(detector.exporters.telegram ?? []).map(
							(telegram) =>
								telegrams.find((t) => t.token === telegram.token && t.chat === telegram.chat)?.label
						),
					(selectedTelegrams) => {
						detector.exporters.telegram = (selectedTelegrams ?? [])
							.map((telegram) => {
								const t = telegrams.find((t) => t.label === telegram);
								const curr = detector.exporters.telegram?.find(
									(t) => t.token === t.token && t.chat === t.chat
								);
								return { token: t!.token, chat: t!.chat, alert_every: curr?.alert_every ?? 1 };
							})
							.filter(Boolean);
					}
				}
				items={telegrams.map((telegram) => ({ value: telegram.label, label: telegram.label }))}
			>
				<Select.Trigger id="telegrams" class="w-full">
					{detector.exporters.telegram && detector.exporters.telegram.length
						? `${detector.exporters.telegram.length} telegram${detector.exporters.telegram.length === 1 ? '' : 's'} selected`
						: 'Select telegrams'}
				</Select.Trigger>
				<Select.Content>
					{#each telegrams as telegram (telegram.label)}
						{@const exporter = detector.exporters.telegram?.find(
							(exporter) => exporter.token === telegram.token && exporter.chat === telegram.chat
						)}
						<Select.Item value={telegram.label} label={telegram.label} class="gap-6">
							<div class="flex flex-1 flex-col">
								<Button
									variant="outline"
									onpointerdown={(e) => e.stopPropagation()}
									onpointerup={(e) => e.stopPropagation()}
									onkeydown={(e) => e.stopPropagation()}
									onclick={(e) => {
										e.stopPropagation();
										testTelegram({ token: telegram.token, chat: telegram.chat });
									}}>Test notification</Button
								>
							</div>
							<div class="flex flex-1 flex-col">
								<span>{telegram.label}</span>
								<span class="text-xs text-muted-foreground">{telegram.chat}</span>
							</div>
							<div
								role="presentation"
								class="flex flex-1 flex-col"
								onpointerdown={(e) => e.stopPropagation()}
								onpointerup={(e) => e.stopPropagation()}
								onclick={(e) => e.stopPropagation()}
								onkeydown={(e) => e.stopPropagation()}
							>
								{#if exporter}
									<Label for={`alert_every_${telegram.label}`} class="text-xs">Alert every</Label>
									<Input
										type="number"
										min="1"
										step="1"
										id={`alert_every_${telegram.label}`}
										bind:value={exporter.alert_every}
									/>
								{/if}
							</div>
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
			<Button
				target="_blank"
				href={setupMode ? '/notifications/add?setup=1' : '/notifications/add'}
				variant="outline"><Plus /></Button
			>
		</div>

		<Label for="model-type" class="mt-2">Model type</Label>
		<Select.Root
			type="single"
			value={modelType}
			onValueChange={setModelType}
			items={[
				{ value: 'yolo', label: 'YOLO' },
				{ value: 'dazzlecow', label: 'DazzleCow' }
			]}
		>
			<Select.Trigger id="model-type" class="w-full"
				>{modelType === 'yolo' ? 'YOLO' : 'DazzleCow'}</Select.Trigger
			>
			<Select.Content>
				<Select.Item value="yolo" label="YOLO" />
				<Select.Item value="dazzlecow" label="DazzleCow" />
			</Select.Content>
		</Select.Root>

		{#if detector.dazzlecow}
			<div class="mt-2 grid grid-cols-2 gap-6">
				<div class="flex flex-col gap-2">
					<Label for="dazzlecow-model">Identity model</Label>
					<Input id="dazzlecow-model" bind:value={detector.dazzlecow.model} />
				</div>
				<div class="flex flex-col gap-2">
					<Label for="dazzlecow-mode">Identity setup</Label>
					<Select.Root
						type="single"
						value={detector.dazzlecow.enrollment ? 'enrollment' : 'gallery'}
						onValueChange={setDazzleCowMode}
					>
						<Select.Trigger id="dazzlecow-mode" class="w-full">
							{detector.dazzlecow.enrollment ? 'Scan cattle' : 'Existing gallery'}
						</Select.Trigger>
						<Select.Content>
							<Select.Item value="enrollment" label="Scan cattle" />
							<Select.Item value="gallery" label="Existing gallery" />
						</Select.Content>
					</Select.Root>
				</div>
				{#if detector.dazzlecow.enrollment}
					<div class="flex flex-col gap-2">
						<Label for="dazzlecow-database">Identity database</Label>
						<Input
							id="dazzlecow-database"
							required
							bind:value={detector.dazzlecow.enrollment.database}
						/>
					</div>
					<div class="flex flex-col gap-2">
						<Label for="dazzlecow-identity-count">Number of cattle</Label>
						<Input
							id="dazzlecow-identity-count"
							type="number"
							min="1"
							step="1"
							placeholder="Optional"
							bind:value={detector.dazzlecow.enrollment.identity_count}
						/>
					</div>
				{:else}
					<div class="flex flex-col gap-2">
						<Label for="dazzlecow-gallery">Identity gallery</Label>
						<Input id="dazzlecow-gallery" required bind:value={detector.dazzlecow.gallery} />
					</div>
				{/if}
				<div class="flex flex-col gap-2">
					<Label for="dazzlecow-confidence">Cow confidence</Label>
					<Input
						id="dazzlecow-confidence"
						type="number"
						min="0"
						max="1"
						step="0.01"
						bind:value={detector.dazzlecow.confidence}
					/>
				</div>
				<div class="flex flex-col gap-2">
					<Label for="owl-interval">OWL interval</Label>
					<Input
						id="owl-interval"
						type="number"
						min="0"
						step="0.1"
						bind:value={detector.dazzlecow.owl_interval}
					/>
				</div>
				<div class="flex flex-col gap-2">
					<Label for="dazzlecow-margin">Frame margin</Label>
					<Input
						id="dazzlecow-margin"
						type="number"
						min="0"
						max="0.49"
						step="0.01"
						bind:value={detector.dazzlecow.margin}
					/>
				</div>
				<div class="flex flex-col gap-2">
					<Label for="match-threshold">Identity threshold</Label>
					<Input
						id="match-threshold"
						type="number"
						min="0"
						max="1"
						step="0.01"
						bind:value={detector.dazzlecow.match_threshold}
					/>
				</div>
				<div class="flex flex-col gap-2">
					<Label for="match-margin">Identity margin</Label>
					<Input
						id="match-margin"
						type="number"
						min="0"
						max="1"
						step="0.01"
						bind:value={detector.dazzlecow.match_margin}
					/>
				</div>
				<div class="flex flex-col gap-2">
					<Label for="track-samples">Track samples</Label>
					<Input
						id="track-samples"
						type="number"
						min="1"
						step="1"
						bind:value={detector.dazzlecow.track_samples}
					/>
				</div>
			</div>
		{:else if detector.yolo}
			<Label for="model" class="mt-2">Model</Label>
			<Input id="model" name="model" bind:value={detector.yolo.model} />

			{#if typeof detector.yolo.confidence === 'number'}
				<div class="mt-2 flex gap-6">
					<div class="flex flex-1 flex-col gap-2">
						<Label for="confidence">Confidence</Label>
						<Input
							type="number"
							min="0"
							max="1"
							step="0.01"
							id="confidence"
							name="confidence"
							bind:value={detector.yolo.confidence}
						/>
					</div>
					<div class="flex flex-1 flex-col gap-2">
						<Label for="frames_min">Required detected frames</Label>
						<Input
							type="number"
							min="1"
							step="1"
							id="frames_min"
							bind:value={detector.yolo.frames_min}
						/>
					</div>
				</div>
			{:else}
				<Label for="confidence" class="mt-2">Confidence</Label>
				<div class="grid grid-cols-3 gap-x-6 gap-y-2">
					{#each Object.keys(detector.yolo.confidence) as key (key)}
						<div class="flex flex-col gap-2">
							<Label for={key}>{key}</Label>
							<Input
								type="number"
								min="0"
								max="1"
								step="0.01"
								id={key}
								name={key}
								bind:value={detector.yolo.confidence[key]}
							/>
						</div>
					{/each}
				</div>
				<Label for="frames_min" class="mt-2">Required detected frames</Label>
				<Input
					type="number"
					min="1"
					step="1"
					id="frames_min"
					bind:value={detector.yolo.frames_min}
				/>
			{/if}
		{/if}

		<div class="mt-2 flex items-center justify-end space-x-2">
			<Switch id="advanced" bind:checked={advanced} />
			<Label for="advanced">Advanced</Label>
		</div>
		{#if advanced}
			<Label class="mt-2">config.json</Label>
			<JsonEditor
				bind:value={
					() => JSON.stringify(detector, null, 2),
					(value) => {
						try {
							detector = JSON.parse(value);
							modelType = detector.dazzlecow ? 'dazzlecow' : 'yolo';
						} catch {
							// Do nothing
						}
					}
				}
				bind:hasErrors={editorHasErrors}
				schema={detectorSchema}
				height={420}
			/>
		{/if}

		<div class="mt-2 flex gap-2">
			{#if isEditing}
				<Button
					type="button"
					onclick={async () => {
						await deleteDetector({ label: originalLabel });
						await goto(resolve('/detectors'));
					}}
					variant="destructive"
					class="flex-1">Delete</Button
				>
			{/if}
			<Button type="submit" class="flex-1" disabled={editorHasErrors}
				>{setupMode ? 'Save setup' : 'Save'}</Button
			>
		</div>
	</form>
</section>
