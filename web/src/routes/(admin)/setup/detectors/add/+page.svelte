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
		getDetectorSchema,
		saveDetector
	} from '$lib/remote/detector.remote';
	import { getPreset, getPresets } from '$lib/remote/preset.remote';
	import { toast } from 'svelte-sonner';
	import * as Select from '$lib/components/ui/select';
	import { getStreams } from '$lib/remote/stream.remote';
	import { getTelegrams, testTelegram } from '$lib/remote/exporter.remote';
	import LiveStream from '$lib/components/live-stream.svelte';
	import { Plus } from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { SvelteSet } from 'svelte/reactivity';
	import { untrack } from 'svelte';
	import type { AppliedPreset } from '$lib/schema';
	import { applyDetectorPreset, type DetectorPresetFragment } from '$lib/preset-fragments';

	type EditableDetector = DetectorConfig & {
		yolo: NonNullable<DetectorConfig['yolo']> & {
			confidence: NonNullable<NonNullable<DetectorConfig['yolo']>['confidence']>;
		};
		exporters: NonNullable<DetectorConfig['exporters']>;
	};
	type IdentityConfig = NonNullable<DetectorConfig['identity']>;

	const DETECTOR_FORM_VALUES = {
		detection: {
			source: []
		},
		yolo: {
			model: '',
			confidence: 0.8
		},
		exporters: { sse: [{}] } as { telegram?: TelegramConfig[]; sse?: SSEConfig[] }
	};
	const INLINE_STREAM_PREVIEW_LIMIT = 5;

	function mergeWithEmptyDetector(detector?: Partial<DetectorConfig>) {
		return {
			...DETECTOR_FORM_VALUES,
			...detector,
			detection: {
				...DETECTOR_FORM_VALUES.detection,
				...detector?.detection
			},
			yolo: {
				...DETECTOR_FORM_VALUES.yolo,
				...detector?.yolo
			},
			identity: detector?.identity,
			exporters: {
				...DETECTOR_FORM_VALUES.exporters,
				...detector?.exporters
			}
		} as EditableDetector;
	}

	const originalLabel = page.url.searchParams.get('label') ?? '';
	const isEditing = !!originalLabel;
	const setupMode = page.url.searchParams.get('setup') === '1';
	const [detectorPresets, identityPresets, detectorSchema, existingDetector] = await Promise.all([
		getPresets({ category: 'detector' }),
		getPresets({ category: 'identity' }),
		getDetectorSchema(),
		isEditing ? getDetector({ label: originalLabel }) : Promise.resolve(undefined)
	]);
	const initialDetector = mergeWithEmptyDetector(existingDetector?.detector);

	let label = $state(originalLabel);
	let detector = $state(untrack(() => initialDetector));
	let visiblePreviewSources = new SvelteSet<string>();
	let editorHasErrors = $state(false);
	let preset = $state<string>(existingDetector?.meta.presets?.detector?.filename ?? 'Custom');
	let identityPreset = $state<string>(
		existingDetector?.meta.presets?.identity?.filename ?? 'Custom'
	);
	let appliedDetectorPreset = $state<AppliedPreset | undefined>(
		existingDetector?.meta.presets?.detector
	);
	let appliedIdentityPreset = $state<AppliedPreset | undefined>(
		existingDetector?.meta.presets?.identity
	);
	let advanced = $state(false);
	const streams = $derived(await getStreams());
	const telegrams = $derived(await getTelegrams());
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
		if (file === 'Custom') {
			appliedDetectorPreset = undefined;
			return;
		}
		const selected = await getPreset({ category: 'detector', file });
		const fragment = selected.value as DetectorPresetFragment;
		detector = mergeWithEmptyDetector(applyDetectorPreset(detector, fragment));
		appliedDetectorPreset = {
			filename: selected.filename,
			blob_sha: selected.blobSha
		};
	}

	async function handleIdentityPresetChange(file: string) {
		if (file === 'Custom') {
			appliedIdentityPreset = undefined;
			return;
		}
		const selected = await getPreset({ category: 'identity', file });
		detector.identity = selected.value as IdentityConfig;
		appliedIdentityPreset = {
			filename: selected.filename,
			blob_sha: selected.blobSha
		};
	}

	function setIdentityEnabled(enabled: boolean) {
		if (enabled) {
			const firstPreset = identityPresets[0];
			if (!firstPreset) {
				toast.error('No identity preset is available.');
				return;
			}
			identityPreset = firstPreset;
			void handleIdentityPresetChange(firstPreset);
		} else {
			delete detector.identity;
			appliedIdentityPreset = undefined;
			identityPreset = 'Custom';
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
		await saveDetector({
			original: originalLabel || undefined,
			detector,
			meta: {
				label,
				presets:
					appliedDetectorPreset || appliedIdentityPreset
						? {
								detector: appliedDetectorPreset,
								identity: appliedIdentityPreset
							}
						: undefined
			}
		});
		toast.warning(
			`Detector configuration '${label}' saved. Restart the detector to apply the changes.`,
			{ duration: Number.POSITIVE_INFINITY, closeButton: true }
		);
		await goto(resolve(setupMode ? '/setup?complete=1' : '/setup/detectors'));
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
											<LiveStream label={source.label} source={source.source} />
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
			<Button target="_blank" href="/setup/cameras/add" variant="outline"><Plus /></Button>
		</div>

		<Label for="telegrams" class="mt-2">Telegram</Label>
		<div class="flex gap-6">
			<Select.Root
				type="multiple"
				bind:value={
					() =>
						(detector.exporters.telegram ?? []).flatMap((exporter) => {
							const telegram = telegrams.find(
								(item) => item.token === exporter.token && item.chat === exporter.chat
							);
							return telegram ? [telegram.label] : [];
						}),
					(selectedTelegrams) => {
						const currentExporters = detector.exporters.telegram ?? [];
						detector.exporters.telegram = (selectedTelegrams ?? []).flatMap((label) => {
							const telegram = telegrams.find((item) => item.label === label);
							if (!telegram) return [];
							const current = currentExporters.find(
								(item) => item.token === telegram.token && item.chat === telegram.chat
							);
							return [
								{
									token: telegram.token,
									chat: telegram.chat,
									alert_every: current?.alert_every ?? 1
								}
							];
						});
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
			<Button target="_blank" href="/setup/notifications/add" variant="outline"><Plus /></Button>
		</div>

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
			<Input type="number" min="1" step="1" id="frames_min" bind:value={detector.yolo.frames_min} />
		{/if}

		<div class="mt-4 flex items-center justify-between">
			<Label for="identity">Identity</Label>
			<Switch id="identity" checked={!!detector.identity} onCheckedChange={setIdentityEnabled} />
		</div>

		{#if detector.identity}
			<div class="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-2">
				<div class="flex flex-col gap-2">
					<Label for="identity-preset">Identity preset</Label>
					<Select.Root
						type="single"
						bind:value={identityPreset}
						onValueChange={handleIdentityPresetChange}
						items={['Custom', ...identityPresets].map((preset) => ({
							value: preset,
							label: getPresetLabel(preset)
						}))}
					>
						<Select.Trigger id="identity-preset" class="w-full">
							{getPresetLabel(identityPreset)}
						</Select.Trigger>
						<Select.Content>
							<Select.Item value="Custom" label="Custom" />
							{#each identityPresets as presetFile (presetFile)}
								<Select.Item value={presetFile} label={getPresetLabel(presetFile)} />
							{/each}
						</Select.Content>
					</Select.Root>
				</div>
				<div class="flex flex-col gap-2">
					<Label for="identity-database">Identity database</Label>
					<Input id="identity-database" required bind:value={detector.identity.database} />
				</div>
				<div class="flex flex-col gap-2">
					<Label for="identity-label">Target label</Label>
					<Input id="identity-label" required bind:value={detector.identity.target_label} />
				</div>
				<div class="flex flex-col gap-2">
					<Label for="identity-confidence">Identity similarity</Label>
					<Input
						id="identity-confidence"
						type="number"
						min="0"
						max="1"
						step="0.01"
						bind:value={detector.identity.similarity_threshold}
					/>
				</div>
				<div class="flex flex-col gap-2">
					<Label for="identity-margin">Runner-up margin</Label>
					<Input
						id="identity-margin"
						type="number"
						min="0"
						max="1"
						step="0.01"
						bind:value={detector.identity.similarity_margin}
					/>
				</div>
				<div class="space-y-4 rounded-md border p-4 sm:col-span-2">
					<div class="space-y-1">
						<p class="text-sm font-medium">Controlled observation zone</p>
						<p class="text-xs text-muted-foreground">
							Only one tracked target inside this normalized rectangle can produce identity
							evidence. Calibrate it for the selected camera before use.
						</p>
					</div>
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
						<div class="flex flex-col gap-2">
							<Label for="identity-zone-x1">Left</Label>
							<Input
								id="identity-zone-x1"
								type="number"
								min="0"
								max="1"
								step="0.01"
								bind:value={detector.identity.controlled_zone.x1}
							/>
						</div>
						<div class="flex flex-col gap-2">
							<Label for="identity-zone-y1">Top</Label>
							<Input
								id="identity-zone-y1"
								type="number"
								min="0"
								max="1"
								step="0.01"
								bind:value={detector.identity.controlled_zone.y1}
							/>
						</div>
						<div class="flex flex-col gap-2">
							<Label for="identity-zone-x2">Right</Label>
							<Input
								id="identity-zone-x2"
								type="number"
								min="0"
								max="1"
								step="0.01"
								bind:value={detector.identity.controlled_zone.x2}
							/>
						</div>
						<div class="flex flex-col gap-2">
							<Label for="identity-zone-y2">Bottom</Label>
							<Input
								id="identity-zone-y2"
								type="number"
								min="0"
								max="1"
								step="0.01"
								bind:value={detector.identity.controlled_zone.y2}
							/>
						</div>
					</div>
					<div class="grid grid-cols-1 gap-3 sm:grid-cols-3">
						<div class="flex flex-col gap-2">
							<Label for="identity-zone-containment">Box inside zone</Label>
							<Input
								id="identity-zone-containment"
								type="number"
								min="0.01"
								max="1"
								step="0.01"
								bind:value={detector.identity.controlled_zone.minimum_box_inside_ratio}
							/>
						</div>
						<div class="flex flex-col gap-2">
							<Label for="identity-zone-stable">Stable frames</Label>
							<Input
								id="identity-zone-stable"
								type="number"
								min="1"
								step="1"
								bind:value={detector.identity.controlled_zone.minimum_stable_frames}
							/>
						</div>
						<div class="flex flex-col gap-2">
							<Label for="identity-zone-clear">Clear frames</Label>
							<Input
								id="identity-zone-clear"
								type="number"
								min="1"
								step="1"
								bind:value={detector.identity.controlled_zone.clear_frames}
							/>
						</div>
					</div>
					<p class="text-xs text-muted-foreground">
						Use separate detector entries when cameras need different zone geometry.
					</p>
				</div>
			</div>
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
						await goto(resolve('/setup/detectors'));
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
