<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { redactCameraSource } from '$lib/camera-source';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Card from '$lib/components/ui/card';
	import * as Empty from '$lib/components/ui/empty';
	import * as Field from '$lib/components/ui/field';
	import { Input } from '$lib/components/ui/input';
	import { Progress } from '$lib/components/ui/progress';
	import * as Select from '$lib/components/ui/select';
	import { Spinner } from '$lib/components/ui/spinner';
	import { Switch } from '$lib/components/ui/switch';
	import {
		completeOnboarding,
		connectOnvifCamera,
		discoverCameras
	} from '$lib/remote/onboarding.remote';
	import { getPreset, getPresets } from '$lib/remote/preset.remote';
	import { createDetectorFromPreset, type DetectorPresetFragment } from '$lib/preset-fragments';
	import type { AppliedPreset, DetectorConfig } from '$lib/schema';
	import ArrowLeftIcon from '@lucide/svelte/icons/arrow-left';
	import ArrowRightIcon from '@lucide/svelte/icons/arrow-right';
	import CameraIcon from '@lucide/svelte/icons/camera';
	import CheckIcon from '@lucide/svelte/icons/check';
	import CircleCheckIcon from '@lucide/svelte/icons/circle-check';
	import PlusIcon from '@lucide/svelte/icons/plus';
	import RadarIcon from '@lucide/svelte/icons/radar';
	import RefreshCwIcon from '@lucide/svelte/icons/refresh-cw';
	import ShieldCheckIcon from '@lucide/svelte/icons/shield-check';
	import SparklesIcon from '@lucide/svelte/icons/sparkles';
	import TrashIcon from '@lucide/svelte/icons/trash-2';
	import { toast } from 'svelte-sonner';

	type Camera = { label: string; source: string };
	type DiscoveryResult = Awaited<ReturnType<typeof discoverCameras>>[number];
	type PresetDocument = Awaited<ReturnType<typeof getPreset>>;
	type SetupStep = 'welcome' | 'cameras' | 'detector' | 'notifications' | 'review';

	const steps: Array<{ value: SetupStep; label: string }> = [
		{ value: 'welcome', label: 'Welcome' },
		{ value: 'cameras', label: 'Cameras' },
		{ value: 'detector', label: 'Detection' },
		{ value: 'notifications', label: 'Alerts' },
		{ value: 'review', label: 'Finish' }
	];
	const [detectorPresets, identityPresets] = await Promise.all([
		getPresets({ category: 'detector' }),
		getPresets({ category: 'identity' })
	]);
	const initialDetectorPreset =
		detectorPresets.find((filename) => filename === 'cow.json') ?? detectorPresets[0];
	const initialIdentityPreset =
		identityPresets.find((filename) => filename === 'cow.json') ?? identityPresets[0];
	if (!initialDetectorPreset) throw new Error('No detector preset is available');
	const [initialDetectorDocument, initialIdentityDocument] = await Promise.all([
		getPreset({ category: 'detector', file: initialDetectorPreset }),
		initialIdentityPreset
			? getPreset({ category: 'identity', file: initialIdentityPreset })
			: Promise.resolve(undefined)
	]);

	function presetLabel(filename: string): string {
		return filename
			.replace(/\.json$/i, '')
			.split('-')
			.filter(Boolean)
			.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
			.join(' ');
	}

	let stepIndex = $state(0);
	let discovered = $state<DiscoveryResult[]>([]);
	let discoveryState = $state<'idle' | 'scanning' | 'done' | 'error'>('idle');
	let discoveryError = $state('');
	let selectedDiscoveryIds = $state<string[]>([]);
	let username = $state('');
	let password = $state('');
	let connecting = $state(false);
	let cameras = $state<Camera[]>([]);
	let manualLabel = $state('');
	let manualSource = $state('');
	let detectorPreset = $state(initialDetectorPreset);
	let detectorDocument = $state<PresetDocument>(initialDetectorDocument);
	let identityPreset = $state(initialIdentityPreset ?? '');
	let identityDocument = $state<PresetDocument | undefined>(initialIdentityDocument);
	let detector = $state(
		createDetectorFromPreset(initialDetectorDocument.value as DetectorPresetFragment)
	);
	let detectorLabel = $state(`${presetLabel(initialDetectorPreset)} detector`);
	let identityEnabled = $state(!!initialIdentityDocument);
	let alertsEnabled = $state(false);
	let telegramLabel = $state('Farm alerts');
	let telegramToken = $state('');
	let telegramChat = $state('');
	let saving = $state(false);
	let completed = $state<{ detectorLabel: string; cameraCount: number } | null>(null);

	const currentStep = $derived(steps[stepIndex]);
	const progress = $derived(((stepIndex + 1) / steps.length) * 100);
	const canContinue = $derived(
		currentStep.value === 'cameras'
			? cameras.length > 0
			: currentStep.value === 'detector'
				? !!detector.yolo?.model && !!detectorLabel.trim()
				: currentStep.value === 'notifications' && alertsEnabled
					? !!telegramToken.trim() && !!telegramChat.trim()
					: true
	);

	function toggleDiscovery(id: string, checked: boolean): void {
		selectedDiscoveryIds = checked
			? [...new Set([...selectedDiscoveryIds, id])]
			: selectedDiscoveryIds.filter((item) => item !== id);
	}

	async function scanNetwork(): Promise<void> {
		discoveryState = 'scanning';
		discoveryError = '';
		selectedDiscoveryIds = [];
		try {
			discovered = await discoverCameras({ timeoutMs: 4_000 });
			discoveryState = 'done';
		} catch (error) {
			discovered = [];
			discoveryState = 'error';
			discoveryError = error instanceof Error ? error.message : 'Camera discovery failed';
		}
	}

	async function connectSelectedCameras(): Promise<void> {
		if (!selectedDiscoveryIds.length) return;
		connecting = true;
		const selected = discovered.filter((camera) => selectedDiscoveryIds.includes(camera.id));
		const outcomes = await Promise.allSettled(
			selected.map(async (camera) => {
				const stream = await connectOnvifCamera({
					id: camera.id,
					username,
					password
				});
				return { ...stream, label: camera.name };
			})
		);
		let failureCount = 0;
		for (const outcome of outcomes) {
			if (outcome.status === 'rejected') {
				failureCount += 1;
				continue;
			}
			const camera = outcome.value;
			if (!cameras.some((item) => item.source === camera.source)) {
				cameras = [...cameras, camera];
			}
		}
		selectedDiscoveryIds = [];
		connecting = false;
		if (failureCount) {
			toast.error(
				`${failureCount} camera${failureCount === 1 ? '' : 's'} could not be connected. Check the ONVIF credentials.`
			);
		} else {
			toast.success(`${outcomes.length} camera${outcomes.length === 1 ? '' : 's'} added.`);
		}
	}

	function addManualCamera(): void {
		const label = manualLabel.trim();
		const source = manualSource.trim();
		try {
			const parsed = new URL(source);
			if (!['rtsp:', 'rtsps:'].includes(parsed.protocol)) throw new Error();
		} catch {
			toast.error('Enter a valid RTSP or RTSPS address.');
			return;
		}
		if (!label) {
			toast.error('Give the camera a name.');
			return;
		}
		if (!cameras.some((camera) => camera.source === source)) {
			cameras = [...cameras, { label, source }];
		}
		manualLabel = '';
		manualSource = '';
	}

	async function changeDetectorPreset(filename: string): Promise<void> {
		const document = await getPreset({ category: 'detector', file: filename });
		detectorPreset = filename;
		detectorDocument = document;
		detector = createDetectorFromPreset(document.value as DetectorPresetFragment);
		if (!detectorLabel.trim() || detectorLabel.endsWith(' detector')) {
			detectorLabel = `${presetLabel(filename)} detector`;
		}
	}

	async function changeIdentityPreset(filename: string): Promise<void> {
		if (!filename) return;
		identityPreset = filename;
		identityDocument = await getPreset({ category: 'identity', file: filename });
	}

	function setIdentityEnabled(enabled: boolean): void {
		identityEnabled = enabled;
		if (enabled && !identityDocument && initialIdentityDocument) {
			identityDocument = initialIdentityDocument;
			identityPreset = initialIdentityPreset ?? '';
		}
	}

	async function next(): Promise<void> {
		if (!canContinue) return;
		if (currentStep.value === 'welcome') {
			stepIndex = 1;
			if (discoveryState === 'idle') await scanNetwork();
			return;
		}
		if (stepIndex < steps.length - 1) stepIndex += 1;
	}

	function back(): void {
		if (stepIndex > 0) stepIndex -= 1;
	}

	async function saveSetup(): Promise<void> {
		saving = true;
		try {
			const resolvedDetector = $state.snapshot(detector);
			resolvedDetector.detection.source = cameras.map((camera) => camera.source);
			resolvedDetector.identity =
				identityEnabled && identityDocument
					? ($state.snapshot(identityDocument.value) as NonNullable<DetectorConfig['identity']>)
					: undefined;
			resolvedDetector.exporters ??= {};
			resolvedDetector.exporters.disk ??= [{}];
			resolvedDetector.exporters.sse ??= [{}];
			resolvedDetector.exporters.telegram = alertsEnabled
				? [
						{
							token: telegramToken.trim(),
							chat: telegramChat.trim(),
							include_video: true
						}
					]
				: undefined;

			completed = await completeOnboarding({
				cameras,
				detector: resolvedDetector,
				label: detectorLabel.trim(),
				detectorPreset: {
					filename: detectorDocument.filename,
					blob_sha: detectorDocument.blobSha
				} satisfies AppliedPreset,
				identityPreset:
					identityEnabled && identityDocument
						? ({
								filename: identityDocument.filename,
								blob_sha: identityDocument.blobSha
							} satisfies AppliedPreset)
						: undefined,
				telegram: alertsEnabled
					? {
							label: telegramLabel.trim() || 'Alerts',
							token: telegramToken.trim(),
							chat: telegramChat.trim()
						}
					: undefined
			});
			toast.success('Setup saved. The detector is starting with the new configuration.');
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Setup could not be saved.');
		} finally {
			saving = false;
		}
	}
</script>

<section class="mx-auto flex w-full max-w-4xl flex-col gap-6" data-testid="setup-wizard">
	<header class="flex flex-col gap-4">
		<div class="flex flex-col gap-1">
			<a
				href={resolve('/setup')}
				class="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground"
			>
				<ArrowLeftIcon class="size-3.5" /> Setup
			</a>
			<h1 class="text-2xl font-semibold tracking-tight">Guided setup</h1>
			<p class="text-sm text-muted-foreground">
				Connect cameras and create a working detector in a few steps.
			</p>
		</div>
		<div class="flex flex-col gap-2">
			<div class="flex items-center justify-between gap-4 text-xs text-muted-foreground">
				<span>Step {stepIndex + 1} of {steps.length}</span>
				<span>{currentStep.label}</span>
			</div>
			<Progress value={progress} aria-label={`Setup progress: ${Math.round(progress)}%`} />
		</div>
	</header>

	{#if completed}
		<Card.Root>
			<Card.Header>
				<div class="flex items-center gap-3">
					<span class="flex size-10 items-center justify-center rounded-md border text-primary">
						<CircleCheckIcon class="size-5" />
					</span>
					<div class="flex flex-col gap-1">
						<Card.Title>Setup complete</Card.Title>
						<Card.Description>
							{completed.detectorLabel} is configured with {completed.cameraCount}
							{completed.cameraCount === 1 ? 'camera' : 'cameras'}.
						</Card.Description>
					</div>
				</div>
			</Card.Header>
			<Card.Content class="flex flex-col gap-3">
				<p class="text-sm text-muted-foreground">
					The detector watches config.json and starts or reloads automatically. The first model
					download can take a few minutes.
				</p>
			</Card.Content>
			<Card.Footer class="flex flex-wrap gap-2">
				<Button onclick={() => goto(resolve('/live'))}>
					Open live view
					<ArrowRightIcon data-icon="inline-end" />
				</Button>
				<Button variant="outline" onclick={() => goto(resolve('/setup'))}>Review setup</Button>
			</Card.Footer>
		</Card.Root>
	{:else}
		<Card.Root>
			{#if currentStep.value === 'welcome'}
				<Card.Header>
					<div class="flex items-center gap-3">
						<span class="flex size-10 items-center justify-center rounded-md border">
							<SparklesIcon class="size-5" />
						</span>
						<div class="flex flex-col gap-1">
							<Card.Title>Let’s get AI Detector ready</Card.Title>
							<Card.Description>
								We will find cameras, select what to detect, and configure optional alerts.
							</Card.Description>
						</div>
					</div>
				</Card.Header>
				<Card.Content class="grid gap-3 md:grid-cols-3">
					<div class="flex flex-col gap-2 rounded-md border p-4">
						<RadarIcon class="size-5 text-muted-foreground" />
						<h2 class="font-medium">Find cameras</h2>
						<p class="text-sm text-muted-foreground">
							ONVIF cameras on this network are discovered automatically.
						</p>
					</div>
					<div class="flex flex-col gap-2 rounded-md border p-4">
						<CameraIcon class="size-5 text-muted-foreground" />
						<h2 class="font-medium">Choose a preset</h2>
						<p class="text-sm text-muted-foreground">
							Start from reviewed detection and identity settings.
						</p>
					</div>
					<div class="flex flex-col gap-2 rounded-md border p-4">
						<ShieldCheckIcon class="size-5 text-muted-foreground" />
						<h2 class="font-medium">Review and start</h2>
						<p class="text-sm text-muted-foreground">
							Nothing is saved until the final confirmation.
						</p>
					</div>
				</Card.Content>
			{:else if currentStep.value === 'cameras'}
				<Card.Header>
					<Card.Title>Connect cameras</Card.Title>
					<Card.Description>
						The scan uses ONVIF WS-Discovery on your local network. You can always add an RTSP
						address manually.
					</Card.Description>
					<Card.Action>
						<Button
							variant="outline"
							onclick={scanNetwork}
							disabled={discoveryState === 'scanning'}
						>
							{#if discoveryState === 'scanning'}
								<Spinner data-icon="inline-start" />
								Scanning…
							{:else}
								<RefreshCwIcon data-icon="inline-start" />
								Scan again
							{/if}
						</Button>
					</Card.Action>
				</Card.Header>
				<Card.Content class="flex flex-col gap-6">
					{#if discoveryState === 'scanning'}
						<div class="flex items-center gap-3 rounded-md border p-4">
							<Spinner />
							<div class="flex flex-col gap-1">
								<p class="font-medium">Looking for ONVIF cameras</p>
								<p class="text-sm text-muted-foreground">This takes about four seconds.</p>
							</div>
						</div>
					{:else if discoveryState === 'error'}
						<div class="rounded-md border p-4">
							<p class="font-medium">Automatic discovery was unavailable</p>
							<p class="text-sm text-muted-foreground">{discoveryError}</p>
						</div>
					{:else if discoveryState === 'done' && !discovered.length}
						<Empty.Root class="border">
							<Empty.Header>
								<Empty.Media variant="icon"><RadarIcon /></Empty.Media>
								<Empty.Title>No ONVIF cameras found</Empty.Title>
								<Empty.Description>
									Check that cameras are on the same LAN, or enter an RTSP address below.
								</Empty.Description>
							</Empty.Header>
						</Empty.Root>
					{:else if discovered.length}
						<Field.Set>
							<Field.Legend>Discovered cameras</Field.Legend>
							<Field.Description>
								Select cameras that share these ONVIF credentials, then connect them.
							</Field.Description>
							<Field.Group>
								{#each discovered as camera (camera.id)}
									<Field.Field orientation="horizontal">
										<Checkbox
											id={`camera-${camera.id}`}
											checked={selectedDiscoveryIds.includes(camera.id)}
											onCheckedChange={(checked) => toggleDiscovery(camera.id, checked === true)}
										/>
										<Field.Content>
											<Field.Label for={`camera-${camera.id}`}>{camera.name}</Field.Label>
											<Field.Description>
												{camera.host}:{camera.port} · {camera.secure ? 'HTTPS ONVIF' : 'ONVIF'}
											</Field.Description>
										</Field.Content>
									</Field.Field>
								{/each}
							</Field.Group>
						</Field.Set>

						<Field.Set>
							<Field.Legend variant="label">Camera credentials</Field.Legend>
							<Field.Description>
								Used once to request the RTSP stream address. They are stored only inside the
								selected RTSP URL.
							</Field.Description>
							<Field.Group>
								<div class="grid gap-4 sm:grid-cols-2">
									<Field.Field>
										<Field.Label for="onvif-username">Username</Field.Label>
										<Input id="onvif-username" autocomplete="username" bind:value={username} />
									</Field.Field>
									<Field.Field>
										<Field.Label for="onvif-password">Password</Field.Label>
										<Input
											id="onvif-password"
											type="password"
											autocomplete="current-password"
											bind:value={password}
										/>
									</Field.Field>
								</div>
								<Button
									onclick={connectSelectedCameras}
									disabled={!selectedDiscoveryIds.length || connecting}
								>
									{#if connecting}
										<Spinner data-icon="inline-start" />
										Connecting…
									{:else}
										<CameraIcon data-icon="inline-start" />
										Connect selected
									{/if}
								</Button>
							</Field.Group>
						</Field.Set>
					{/if}

					<Field.Set>
						<Field.Legend variant="label">Manual RTSP camera</Field.Legend>
						<Field.Group>
							<div class="grid gap-4 sm:grid-cols-2">
								<Field.Field>
									<Field.Label for="manual-label">Camera name</Field.Label>
									<Input id="manual-label" placeholder="Barn entrance" bind:value={manualLabel} />
								</Field.Field>
								<Field.Field>
									<Field.Label for="manual-source">RTSP address</Field.Label>
									<Input
										id="manual-source"
										placeholder="rtsp://user:password@192.168.1.20/stream"
										bind:value={manualSource}
									/>
								</Field.Field>
							</div>
							<Button variant="outline" onclick={addManualCamera}>
								<PlusIcon data-icon="inline-start" />
								Add RTSP camera
							</Button>
						</Field.Group>
					</Field.Set>

					{#if cameras.length}
						<div class="flex flex-col gap-2">
							<h2 class="text-sm font-medium">Connected cameras</h2>
							{#each cameras as camera (camera.source)}
								<div class="flex items-center gap-3 rounded-md border p-3">
									<CameraIcon class="size-4 shrink-0 text-muted-foreground" />
									<div class="min-w-0 flex-1">
										<p class="truncate text-sm font-medium">{camera.label}</p>
										<p class="truncate text-xs text-muted-foreground">
											{redactCameraSource(camera.source)}
										</p>
									</div>
									<Button
										size="icon-sm"
										variant="ghost"
										aria-label={`Remove ${camera.label}`}
										onclick={() =>
											(cameras = cameras.filter((item) => item.source !== camera.source))}
									>
										<TrashIcon />
									</Button>
								</div>
							{/each}
						</div>
					{/if}
				</Card.Content>
			{:else if currentStep.value === 'detector'}
				<Card.Header>
					<Card.Title>Choose what to detect</Card.Title>
					<Card.Description>
						Reviewed presets fill in the model, tracking, thresholds, and runtime settings.
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<Field.Group>
						<Field.Field>
							<Field.Label for="detector-name">Detector name</Field.Label>
							<Input id="detector-name" bind:value={detectorLabel} />
						</Field.Field>
						<Field.Field>
							<Field.Label for="detector-preset">Detection preset</Field.Label>
							<Select.Root
								type="single"
								value={detectorPreset}
								onValueChange={(value) => value && changeDetectorPreset(value)}
								items={detectorPresets.map((filename) => ({
									value: filename,
									label: presetLabel(filename)
								}))}
							>
								<Select.Trigger id="detector-preset" class="w-full">
									{presetLabel(detectorPreset)}
								</Select.Trigger>
								<Select.Content>
									<Select.Group>
										{#each detectorPresets as filename (filename)}
											<Select.Item value={filename} label={presetLabel(filename)} />
										{/each}
									</Select.Group>
								</Select.Content>
							</Select.Root>
						</Field.Field>
						<Field.Field orientation="horizontal">
							<Field.Content>
								<Field.Label for="identity-enabled">Recognize individual identities</Field.Label>
								<Field.Description>
									{identityPresets.length
										? 'Keep this enabled when the selected detector tracks a supported identity label.'
										: 'No reviewed identity preset is currently available. Detection can still be configured.'}
								</Field.Description>
							</Field.Content>
							<Switch
								id="identity-enabled"
								checked={identityEnabled}
								onCheckedChange={setIdentityEnabled}
								disabled={!identityPresets.length}
							/>
						</Field.Field>
						{#if identityEnabled && identityPresets.length}
							<Field.Field>
								<Field.Label for="identity-preset">Identity preset</Field.Label>
								<Select.Root
									type="single"
									value={identityPreset}
									onValueChange={(value) => value && changeIdentityPreset(value)}
									items={identityPresets.map((filename) => ({
										value: filename,
										label: presetLabel(filename)
									}))}
								>
									<Select.Trigger id="identity-preset" class="w-full">
										{presetLabel(identityPreset)}
									</Select.Trigger>
									<Select.Content>
										<Select.Group>
											{#each identityPresets as filename (filename)}
												<Select.Item value={filename} label={presetLabel(filename)} />
											{/each}
										</Select.Group>
									</Select.Content>
								</Select.Root>
							</Field.Field>
						{/if}
					</Field.Group>
				</Card.Content>
			{:else if currentStep.value === 'notifications'}
				<Card.Header>
					<Card.Title>Configure alerts</Card.Title>
					<Card.Description>
						Telegram is optional. Detections are always saved locally and visible in the web app.
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<Field.Group>
						<Field.Field orientation="horizontal">
							<Field.Content>
								<Field.Label for="alerts-enabled">Send Telegram alerts</Field.Label>
								<Field.Description>
									You can skip this and add notifications later from Setup.
								</Field.Description>
							</Field.Content>
							<Switch id="alerts-enabled" bind:checked={alertsEnabled} />
						</Field.Field>
						{#if alertsEnabled}
							<Field.Field>
								<Field.Label for="telegram-label">Channel name</Field.Label>
								<Input id="telegram-label" bind:value={telegramLabel} />
							</Field.Field>
							<Field.Field>
								<Field.Label for="telegram-token">Bot token</Field.Label>
								<Input
									id="telegram-token"
									type="password"
									autocomplete="off"
									bind:value={telegramToken}
								/>
							</Field.Field>
							<Field.Field>
								<Field.Label for="telegram-chat">Chat ID</Field.Label>
								<Input id="telegram-chat" bind:value={telegramChat} />
							</Field.Field>
						{/if}
					</Field.Group>
				</Card.Content>
			{:else}
				<Card.Header>
					<Card.Title>Review setup</Card.Title>
					<Card.Description>
						These settings are written atomically. The detector reloads them automatically.
					</Card.Description>
				</Card.Header>
				<Card.Content class="flex flex-col gap-3">
					<div class="grid gap-3 sm:grid-cols-2">
						<div class="flex flex-col gap-2 rounded-md border p-4">
							<span class="text-sm text-muted-foreground">Cameras</span>
							<span class="text-lg font-medium">{cameras.length}</span>
							<span class="text-sm">{cameras.map((camera) => camera.label).join(', ')}</span>
						</div>
						<div class="flex flex-col gap-2 rounded-md border p-4">
							<span class="text-sm text-muted-foreground">Detector</span>
							<span class="text-lg font-medium">{detectorLabel}</span>
							<span class="text-sm">{presetLabel(detectorPreset)}</span>
						</div>
						<div class="flex flex-col gap-2 rounded-md border p-4">
							<span class="text-sm text-muted-foreground">Identity</span>
							<span class="text-lg font-medium">{identityEnabled ? 'Enabled' : 'Disabled'}</span>
							<span class="text-sm">
								{identityEnabled ? presetLabel(identityPreset) : 'Can be enabled later'}
							</span>
						</div>
						<div class="flex flex-col gap-2 rounded-md border p-4">
							<span class="text-sm text-muted-foreground">Alerts</span>
							<span class="text-lg font-medium">{alertsEnabled ? 'Telegram' : 'Local only'}</span>
							<span class="text-sm">
								{alertsEnabled ? telegramLabel : 'Detections remain available in the web app'}
							</span>
						</div>
					</div>
				</Card.Content>
			{/if}

			<Card.Footer class="flex flex-row justify-between gap-2">
				<Button variant="outline" onclick={back} disabled={stepIndex === 0 || saving}>
					<ArrowLeftIcon data-icon="inline-start" />
					Back
				</Button>
				{#if currentStep.value === 'review'}
					<Button onclick={saveSetup} disabled={!canContinue || saving}>
						{#if saving}
							<Spinner data-icon="inline-start" />
							Saving…
						{:else}
							<CheckIcon data-icon="inline-start" />
							Save and start
						{/if}
					</Button>
				{:else}
					<Button onclick={next} disabled={!canContinue || saving}>
						Continue
						<ArrowRightIcon data-icon="inline-end" />
					</Button>
				{/if}
			</Card.Footer>
		</Card.Root>
	{/if}

	<div class="flex flex-wrap gap-2" aria-label="Setup steps">
		{#each steps as item, index (item.value)}
			<Badge
				variant={index === stepIndex ? 'default' : index < stepIndex ? 'secondary' : 'outline'}
			>
				{#if index < stepIndex}<CheckIcon />{/if}
				{item.label}
			</Badge>
		{/each}
	</div>
</section>
