<script lang="ts">
	import { resolve } from '$app/paths';
	import * as Alert from '$lib/components/ui/alert';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Empty from '$lib/components/ui/empty';
	import { Input } from '$lib/components/ui/input';
	import * as NativeSelect from '$lib/components/ui/native-select';
	import * as Sheet from '$lib/components/ui/sheet';
	import { Textarea } from '$lib/components/ui/textarea';
	import {
		addOfficialIdentity,
		confirmIdentity,
		correctIdentity,
		deactivateIdentity,
		editOfficialIdentity,
		getIdentityCatalogs,
		mergeIdentityEvidence,
		provisionIdentity,
		splitIdentityEvidence
	} from '$lib/remote/identity.remote';
	import type { IdentityCatalogView } from '$lib/remote/identity.remote';
	import ArchiveIcon from '@lucide/svelte/icons/archive';
	import ArrowRightIcon from '@lucide/svelte/icons/arrow-right';
	import CheckCircleIcon from '@lucide/svelte/icons/circle-check';
	import CircleAlertIcon from '@lucide/svelte/icons/circle-alert';
	import ClockIcon from '@lucide/svelte/icons/clock-3';
	import DatabaseIcon from '@lucide/svelte/icons/database';
	import EyeIcon from '@lucide/svelte/icons/eye';
	import MergeIcon from '@lucide/svelte/icons/merge';
	import PlusIcon from '@lucide/svelte/icons/plus';
	import RefreshIcon from '@lucide/svelte/icons/refresh-cw';
	import ScissorsIcon from '@lucide/svelte/icons/scissors';
	import SearchIcon from '@lucide/svelte/icons/search';

	const catalogsQuery = getIdentityCatalogs();
	const catalogs = $derived(await catalogsQuery);
	let selectedCatalogId = $state<string | null>(null);
	const catalog = $derived<IdentityCatalogView | null>(
		catalogs.find((item) => item.catalogId === selectedCatalogId) ?? catalogs[0] ?? null
	);
	const snapshot = $derived(catalog?.snapshot ?? null);
	const revision = $derived(snapshot?.control.operatorRevision ?? 0);
	const singular = $derived(catalog?.targetLabel ?? 'identity');
	const plural = $derived(singular === 'identity' ? 'identities' : `${singular}s`);
	const officialIdLabel = $derived(`${titleCase(singular)} ID`);

	let search = $state('');
	let addOpen = $state(false);
	let addOfficialId = $state('');
	let addDisplayName = $state('');
	let addNotes = $state('');
	let pending = $state<string | null>(null);
	let errorMessage = $state<string | null>(null);
	let detailsOpen = $state(false);
	let selectedOfficialId = $state<string | null>(null);
	let selectedVisualIdentityId = $state<string | null>(null);
	let editKey = $state<string | null>(null);
	let editDisplayName = $state('');
	let editNotes = $state('');
	let editStatus = $state<'active' | 'archived'>('active');
	let assignmentOfficials = $state<Record<string, string>>({});
	let assignmentTracklets = $state<Record<string, string>>({});
	let correctionOfficialId = $state('');
	let correctionTrackletId = $state('');
	let mergeTargetId = $state('');
	let splitTracklets = $state<Record<string, boolean>>({});

	const activeOfficials = $derived(
		snapshot?.officialIdentities.filter((identity) => identity.status === 'active') ?? []
	);
	const visibleOfficials = $derived(
		(snapshot?.officialIdentities ?? []).filter((identity) => {
			const query = search.trim().toLocaleLowerCase();
			return (
				!query ||
				identity.officialId.toLocaleLowerCase().includes(query) ||
				identity.displayName?.toLocaleLowerCase().includes(query) ||
				identity.notes.toLocaleLowerCase().includes(query)
			);
		})
	);
	const reviewVisuals = $derived(
		(snapshot?.visualIdentities ?? []).filter((identity) => identity.mappingState === null)
	);
	const selectedOfficial = $derived(
		snapshot?.officialIdentities.find((identity) => identity.officialId === selectedOfficialId) ??
			null
	);
	const selectedVisual = $derived(
		snapshot?.visualIdentities.find(
			(identity) => identity.visualIdentityId === selectedVisualIdentityId
		) ?? null
	);
	const confirmationTracklets = $derived(
		selectedVisual?.tracklets.filter(
			(tracklet) =>
				tracklet.trackletId !== selectedVisual.provisionalTrackletId &&
				tracklet.evidenceStatus === 'eligible'
		) ?? []
	);
	const mergeTargets = $derived(
		(snapshot?.visualIdentities ?? []).filter(
			(identity) => identity.visualIdentityId !== selectedVisualIdentityId
		)
	);

	$effect(() => {
		if (!selectedOfficial || editKey === selectedOfficial.officialId) return;
		editKey = selectedOfficial.officialId;
		editDisplayName = selectedOfficial.displayName ?? '';
		editNotes = selectedOfficial.notes;
		editStatus = selectedOfficial.status;
		correctionOfficialId = '';
		correctionTrackletId = selectedVisual?.provisionalTrackletId ?? '';
	});

	function titleCase(value: string): string {
		return value ? value.charAt(0).toLocaleUpperCase() + value.slice(1) : value;
	}

	function formatDate(value: string | undefined): string {
		if (!value) return 'No evidence yet';
		return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(
			new Date(value)
		);
	}

	function lastSeen(visualIdentityId: string | null): string {
		const visual = snapshot?.visualIdentities.find(
			(identity) => identity.visualIdentityId === visualIdentityId
		);
		return formatDate(visual?.tracklets[0]?.lastCapturedAt);
	}

	function mappingLabel(state: 'provisional' | 'confirmed' | null): string {
		if (state === 'confirmed') return 'Confirmed';
		if (state === 'provisional') return 'Needs confirmation';
		return 'Not mapped';
	}

	async function mutate(key: string, action: () => Promise<unknown>, fallback: string) {
		pending = key;
		errorMessage = null;
		try {
			await action();
		} catch (error) {
			errorMessage = error instanceof Error ? error.message : fallback;
		} finally {
			pending = null;
		}
	}

	function openOfficial(officialId: string, visualIdentityId: string | null) {
		selectedOfficialId = officialId;
		selectedVisualIdentityId = visualIdentityId;
		editKey = null;
		splitTracklets = {};
		mergeTargetId = '';
		detailsOpen = true;
	}

	function openVisual(visualIdentityId: string) {
		const visual = snapshot?.visualIdentities.find(
			(identity) => identity.visualIdentityId === visualIdentityId
		);
		selectedOfficialId = visual?.officialId ?? null;
		selectedVisualIdentityId = visualIdentityId;
		editKey = null;
		splitTracklets = {};
		mergeTargetId = '';
		detailsOpen = true;
	}

	async function createOfficial() {
		if (!catalog) return;
		await mutate(
			'add-official',
			async () => {
				await addOfficialIdentity({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					officialId: addOfficialId,
					displayName: addDisplayName,
					notes: addNotes
				}).updates(catalogsQuery);
				addOpen = false;
				addOfficialId = '';
				addDisplayName = '';
				addNotes = '';
			},
			`Could not add ${singular}.`
		);
	}

	async function saveOfficial() {
		if (!catalog || !selectedOfficial) return;
		await mutate(
			'edit-official',
			() =>
				editOfficialIdentity({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					officialId: selectedOfficial.officialId,
					displayName: editDisplayName,
					status: editStatus,
					notes: editNotes
				}).updates(catalogsQuery),
			`Could not update ${singular}.`
		);
	}

	async function assignVisual(visualIdentityId: string) {
		if (!catalog) return;
		const visual = snapshot?.visualIdentities.find(
			(identity) => identity.visualIdentityId === visualIdentityId
		);
		const officialId = assignmentOfficials[visualIdentityId] ?? activeOfficials[0]?.officialId;
		const trackletId =
			assignmentTracklets[visualIdentityId] ??
			visual?.tracklets.find((tracklet) => tracklet.evidenceStatus === 'eligible')?.trackletId;
		if (!officialId || !trackletId) {
			errorMessage = `Choose an official ${singular} and eligible evidence first.`;
			return;
		}
		await mutate(
			`assign:${visualIdentityId}`,
			() =>
				provisionIdentity({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					visualIdentityId,
					officialId,
					trackletId
				}).updates(catalogsQuery),
			'Could not create the provisional mapping.'
		);
	}

	async function confirm(trackletId: string) {
		if (!catalog || selectedVisual?.mappingState !== 'provisional') return;
		await mutate(
			`confirm:${trackletId}`,
			() =>
				confirmIdentity({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					visualIdentityId: selectedVisual.visualIdentityId,
					confirmationTrackletId: trackletId
				}).updates(catalogsQuery),
			'Could not confirm the mapping.'
		);
	}

	async function correct() {
		if (
			!catalog ||
			!selectedVisual?.mappingState ||
			!correctionOfficialId ||
			!correctionTrackletId
		) {
			return;
		}
		await mutate(
			'correct-mapping',
			() =>
				correctIdentity({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					visualIdentityId: selectedVisual.visualIdentityId,
					officialId: correctionOfficialId,
					provisionalTrackletId: correctionTrackletId
				}).updates(catalogsQuery),
			'Could not correct the mapping.'
		);
	}

	async function deactivate() {
		if (!catalog || !selectedVisual?.mappingState) return;
		if (
			!window.confirm('Deactivate this identity mapping? Identity output will stop until remapped.')
		) {
			return;
		}
		await mutate(
			'deactivate-mapping',
			() =>
				deactivateIdentity({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					visualIdentityId: selectedVisual.visualIdentityId
				}).updates(catalogsQuery),
			'Could not deactivate the mapping.'
		);
	}

	async function merge() {
		if (!catalog || !selectedVisual || !mergeTargetId) return;
		if (!window.confirm('Merge this visual evidence into the selected identity?')) return;
		await mutate(
			'merge-visual',
			() =>
				mergeIdentityEvidence({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					sourceVisualIdentityId: selectedVisual.visualIdentityId,
					targetVisualIdentityId: mergeTargetId
				}).updates(catalogsQuery),
			'Could not merge the visual identities.'
		);
	}

	async function split() {
		if (!catalog || !selectedVisual) return;
		const trackletIds = Object.entries(splitTracklets)
			.filter(([, selected]) => selected)
			.map(([trackletId]) => trackletId);
		if (!trackletIds.length) return;
		if (!window.confirm('Split the selected evidence into a new visual identity?')) return;
		await mutate(
			'split-visual',
			() =>
				splitIdentityEvidence({
					catalogId: catalog.catalogId,
					expectedRevision: revision,
					sourceVisualIdentityId: selectedVisual.visualIdentityId,
					trackletIds
				}).updates(catalogsQuery),
			'Could not split the visual identity.'
		);
	}
</script>

<section class="space-y-6" data-testid="identities-page">
	<header class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
		<div class="space-y-1">
			<div class="flex flex-wrap items-center gap-2">
				<h1 class="text-2xl font-semibold tracking-tight">{titleCase(plural)}</h1>
				{#if catalogs.length > 1}
					<NativeSelect.Root
						class="h-8 w-auto text-xs"
						value={catalog?.catalogId ?? ''}
						onchange={(event) => (selectedCatalogId = event.currentTarget.value)}
						aria-label="Identity catalog"
					>
						{#each catalogs as item (item.catalogId)}
							<NativeSelect.Option value={item.catalogId}>{item.label}</NativeSelect.Option>
						{/each}
					</NativeSelect.Root>
				{/if}
			</div>
			<p class="max-w-2xl text-sm text-muted-foreground">
				Review new visual evidence and manage official {singular} records.
			</p>
		</div>
		<Button onclick={() => (addOpen = true)} disabled={catalog?.state !== 'ready'}>
			<PlusIcon /> Add {singular}
		</Button>
	</header>

	{#if errorMessage}
		<Alert.Root variant="destructive">
			<CircleAlertIcon />
			<Alert.Title>Identity change was not saved</Alert.Title>
			<Alert.Description>{errorMessage}</Alert.Description>
		</Alert.Root>
	{/if}

	{#if !catalog}
		<Empty.Root class="border">
			<Empty.Header>
				<Empty.Media variant="icon"><DatabaseIcon /></Empty.Media>
				<Empty.Title>Identity is not configured</Empty.Title>
				<Empty.Description>
					Apply an identity preset from Setup before managing official records.
				</Empty.Description>
			</Empty.Header>
			<Empty.Content>
				<Button href={resolve('/setup')}>Open Setup</Button>
			</Empty.Content>
		</Empty.Root>
	{:else if catalog.state !== 'ready' || !snapshot}
		<Alert.Root>
			<CircleAlertIcon />
			<Alert.Title>
				{catalog.state === 'not_initialized'
					? 'Start the detector to initialize identities'
					: 'Identity database unavailable'}
			</Alert.Title>
			<Alert.Description>
				<p>{catalog.message}</p>
				<Button variant="outline" size="sm" onclick={() => catalogsQuery.refresh()}>
					<RefreshIcon /> Check again
				</Button>
			</Alert.Description>
		</Alert.Root>
	{:else}
		{#if reviewVisuals.length > 0}
			<Card.Root class="gap-0 overflow-hidden rounded-lg py-0 shadow-none">
				<Card.Header class="border-b px-4 py-4">
					<div class="flex items-start justify-between gap-3">
						<div class="space-y-1">
							<Card.Title id="review-title">Needs review</Card.Title>
							<Card.Description>
								Assign new visual evidence to an official {singular}. The first assignment remains
								provisional.
							</Card.Description>
						</div>
						<Badge variant="secondary">{reviewVisuals.length} waiting</Badge>
					</div>
				</Card.Header>
				<Card.Content class="p-0" aria-labelledby="review-title">
					{#each reviewVisuals as visual (visual.visualIdentityId)}
						{@const eligibleTracklets = visual.tracklets.filter(
							(tracklet) => tracklet.evidenceStatus === 'eligible'
						)}
						<article
							class="grid gap-4 border-t p-4 first:border-t-0 sm:grid-cols-[4.5rem_minmax(0,1fr)]"
						>
							<button
								class="size-18 overflow-hidden rounded-md bg-muted"
								onclick={() => openVisual(visual.visualIdentityId)}
								aria-label="View visual evidence"
							>
								{#if visual.tracklets[0]?.preview}
									<img
										src={visual.tracklets[0].preview}
										alt=""
										class="h-full w-full object-cover"
									/>
								{/if}
							</button>
							<div class="min-w-0 space-y-3">
								<div class="flex flex-wrap items-start justify-between gap-2">
									<div>
										<p class="text-sm font-medium">Unassigned {singular}</p>
										<p class="text-xs text-muted-foreground">
											{visual.tracklets.length} tracklets ·
											{visual.tracklets.reduce(
												(total, tracklet) => total + tracklet.evidenceCount,
												0
											)}
											frames
										</p>
									</div>
									<Button
										variant="ghost"
										size="sm"
										onclick={() => openVisual(visual.visualIdentityId)}
									>
										<EyeIcon /> View evidence
									</Button>
								</div>
								{#if activeOfficials.length > 0 && eligibleTracklets.length > 0}
									<div class="grid gap-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
										<NativeSelect.Root
											class="w-full"
											value={assignmentOfficials[visual.visualIdentityId] ??
												activeOfficials[0]?.officialId}
											onchange={(event) =>
												(assignmentOfficials[visual.visualIdentityId] = event.currentTarget.value)}
											aria-label={`Official ${singular}`}
										>
											{#each activeOfficials as official (official.officialId)}
												<NativeSelect.Option value={official.officialId}>
													{official.officialId}{official.displayName
														? ` · ${official.displayName}`
														: ''}
												</NativeSelect.Option>
											{/each}
										</NativeSelect.Root>
										<NativeSelect.Root
											class="w-full"
											value={assignmentTracklets[visual.visualIdentityId] ??
												eligibleTracklets[0]?.trackletId}
											onchange={(event) =>
												(assignmentTracklets[visual.visualIdentityId] = event.currentTarget.value)}
											aria-label="First evidence tracklet"
										>
											{#each eligibleTracklets as tracklet, index (tracklet.trackletId)}
												<NativeSelect.Option value={tracklet.trackletId}>
													Evidence {index + 1} · {tracklet.evidenceCount} frames
												</NativeSelect.Option>
											{/each}
										</NativeSelect.Root>
										<Button
											disabled={pending === `assign:${visual.visualIdentityId}`}
											onclick={() => assignVisual(visual.visualIdentityId)}
										>
											Assign provisionally
										</Button>
									</div>
								{:else if activeOfficials.length === 0}
									<Button variant="outline" size="sm" onclick={() => (addOpen = true)}>
										<PlusIcon /> Add an official record first
									</Button>
								{:else}
									<p class="text-sm text-muted-foreground">
										No eligible evidence is available for assignment.
									</p>
								{/if}
							</div>
						</article>
					{/each}
				</Card.Content>
			</Card.Root>
		{/if}

		<Card.Root class="gap-0 overflow-hidden rounded-lg py-0 shadow-none">
			<Card.Header class="border-b px-4 py-4">
				<div class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
					<div class="space-y-1">
						<Card.Title id="catalog-title">Official records</Card.Title>
						<Card.Description>
							{snapshot.officialIdentities.length} official
							{snapshot.officialIdentities.length === 1 ? singular : plural}
						</Card.Description>
					</div>
					<div class="relative w-full sm:w-72">
						<SearchIcon
							class="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
						/>
						<Input bind:value={search} class="pl-9" placeholder={`Search ${plural}`} />
					</div>
				</div>
			</Card.Header>
			<Card.Content class="divide-y p-0" aria-labelledby="catalog-title">
				{#each visibleOfficials as official (official.officialId)}
					<button
						class="grid w-full grid-cols-[3.5rem_minmax(0,1fr)_auto] items-center gap-3 p-3 text-left transition-colors hover:bg-muted/50 sm:grid-cols-[3.5rem_minmax(0,1fr)_minmax(10rem,auto)_auto]"
						onclick={() => openOfficial(official.officialId, official.visualIdentityId)}
					>
						{#if official.preview}
							<img src={official.preview} alt="" class="size-14 rounded-md object-cover" />
						{:else}
							<span class="size-14 rounded-md bg-muted"></span>
						{/if}
						<span class="min-w-0">
							<span class="block truncate text-sm font-medium">{official.officialId}</span>
							<span class="block truncate text-xs text-muted-foreground">
								{official.displayName ?? `Unnamed ${singular}`}
							</span>
							<Badge
								class="mt-1"
								variant={official.mappingState === 'confirmed'
									? 'default'
									: official.mappingState === 'provisional'
										? 'secondary'
										: 'outline'}
							>
								{mappingLabel(official.mappingState)}
							</Badge>
						</span>
						<span class="hidden text-xs text-muted-foreground sm:block">
							{lastSeen(official.visualIdentityId)}
						</span>
						<ArrowRightIcon class="size-4 text-muted-foreground" />
					</button>
				{:else}
					<p class="py-8 text-center text-sm text-muted-foreground">
						No official {plural} found.
					</p>
				{/each}
			</Card.Content>
		</Card.Root>
	{/if}
</section>

<Dialog.Root bind:open={addOpen}>
	<Dialog.Content class="sm:max-w-md">
		<Dialog.Header>
			<Dialog.Title>Add {singular}</Dialog.Title>
			<Dialog.Description>
				Create the official record first. Visual evidence can then be assigned provisionally.
			</Dialog.Description>
		</Dialog.Header>
		<div class="space-y-4 py-2">
			<label class="grid gap-1.5 text-sm font-medium">
				{officialIdLabel}
				<Input bind:value={addOfficialId} autocomplete="off" placeholder="Required" />
			</label>
			<label class="grid gap-1.5 text-sm font-medium">
				Display name
				<Input bind:value={addDisplayName} autocomplete="off" placeholder="Optional" />
			</label>
			<label class="grid gap-1.5 text-sm font-medium">
				Notes
				<Textarea bind:value={addNotes} placeholder="Optional" />
			</label>
		</div>
		<Dialog.Footer>
			<Dialog.Close>
				{#snippet child({ props })}
					<Button variant="outline" {...props}>Cancel</Button>
				{/snippet}
			</Dialog.Close>
			<Button
				disabled={!addOfficialId.trim() || pending === 'add-official'}
				onclick={createOfficial}
			>
				Add {singular}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<Sheet.Root bind:open={detailsOpen}>
	<Sheet.Content
		class="w-full gap-0 overflow-y-auto max-sm:inset-x-0 max-sm:inset-y-auto max-sm:bottom-0 max-sm:h-[88svh] max-sm:border-t sm:max-w-xl"
	>
		<Sheet.Header class="border-b px-4 py-4">
			<Sheet.Title>
				{selectedOfficial?.displayName ?? selectedOfficial?.officialId ?? `Visual ${singular}`}
			</Sheet.Title>
			<Sheet.Description>
				{selectedOfficial
					? `${selectedOfficial.officialId} · ${mappingLabel(selectedOfficial.mappingState)}`
					: `Unassigned ${singular} evidence`}
			</Sheet.Description>
		</Sheet.Header>

		<div class="space-y-6 p-4">
			{#if selectedVisual}
				<section class="space-y-3">
					<div class="flex items-center justify-between">
						<h3 class="font-semibold">Evidence</h3>
						<Badge variant="outline">{selectedVisual.tracklets.length} tracklets</Badge>
					</div>
					<div class="grid grid-cols-2 gap-2 sm:grid-cols-3">
						{#each selectedVisual.tracklets as tracklet (tracklet.trackletId)}
							<figure class="overflow-hidden rounded-md border">
								<img src={tracklet.preview} alt="" class="aspect-square w-full object-cover" />
								<figcaption class="space-y-0.5 p-2">
									<p class="truncate text-xs font-medium">{tracklet.source}</p>
									<p class="text-[11px] text-muted-foreground">
										{tracklet.evidenceCount} frames · {tracklet.evidenceStatus}
									</p>
								</figcaption>
							</figure>
						{/each}
					</div>
				</section>

				{#if selectedVisual.mappingState === 'provisional'}
					<section class="space-y-3">
						<Alert.Root>
							<ClockIcon />
							<Alert.Title>Independent confirmation needed</Alert.Title>
							<Alert.Description>
								<p>
									Confirm with a different eligible tracklet. The first mapping is not used as final
									gallery truth.
								</p>
							</Alert.Description>
						</Alert.Root>
						{#if confirmationTracklets.length > 0}
							<div class="divide-y overflow-hidden rounded-md border">
								{#each confirmationTracklets as tracklet (tracklet.trackletId)}
									<div class="flex items-center gap-3 p-2">
										<img src={tracklet.preview} alt="" class="size-12 rounded-md object-cover" />
										<div class="min-w-0 flex-1">
											<p class="truncate text-sm font-medium">{tracklet.source}</p>
											<p class="text-xs text-muted-foreground">
												{tracklet.evidenceCount} evidence frames
											</p>
										</div>
										<Button
											size="sm"
											disabled={pending === `confirm:${tracklet.trackletId}`}
											onclick={() => confirm(tracklet.trackletId)}
										>
											<CheckCircleIcon /> Confirm
										</Button>
									</div>
								{/each}
							</div>
						{:else}
							<p class="text-sm text-muted-foreground">Waiting for another eligible tracklet.</p>
						{/if}
					</section>
				{:else if selectedVisual.mappingState === 'confirmed'}
					<Alert.Root>
						<CheckCircleIcon class="text-emerald-600" />
						<Alert.Title>Identity confirmed</Alert.Title>
						<Alert.Description>
							<p>Two distinct tracklets provide the gallery evidence for this identity.</p>
						</Alert.Description>
					</Alert.Root>
				{/if}
			{/if}

			{#if selectedOfficial}
				<Card.Root class="gap-0 rounded-lg py-0 shadow-none">
					<Card.Header class="border-b px-4 py-3">
						<Card.Title class="text-base">Official record</Card.Title>
						<Card.Description>Farmer-facing identity information.</Card.Description>
					</Card.Header>
					<Card.Content class="grid gap-3 p-4">
						<label class="grid gap-1.5 text-sm font-medium">
							{officialIdLabel}
							<Input value={selectedOfficial.officialId} disabled />
						</label>
						<label class="grid gap-1.5 text-sm font-medium">
							Display name
							<Input bind:value={editDisplayName} />
						</label>
						<label class="grid gap-1.5 text-sm font-medium">
							Status
							<NativeSelect.Root bind:value={editStatus} class="w-full">
								<NativeSelect.Option value="active">Active</NativeSelect.Option>
								<NativeSelect.Option value="archived">Archived</NativeSelect.Option>
							</NativeSelect.Root>
						</label>
						<label class="grid gap-1.5 text-sm font-medium">
							Notes
							<Textarea bind:value={editNotes} />
						</label>
						<Button variant="outline" disabled={pending === 'edit-official'} onclick={saveOfficial}>
							Save record
						</Button>
					</Card.Content>
				</Card.Root>
			{/if}

			{#if selectedVisual}
				<details class="rounded-lg border">
					<summary class="cursor-pointer px-4 py-3 text-sm font-semibold">
						Advanced identity details
					</summary>
					<div class="space-y-5 border-t p-4">
						<div class="space-y-1">
							<p class="text-xs font-medium tracking-wide text-muted-foreground uppercase">
								Visual identity ID
							</p>
							<code class="block overflow-hidden text-xs text-ellipsis">
								{selectedVisual.visualIdentityId}
							</code>
						</div>

						{#if selectedVisual.mappingState}
							<section class="space-y-2">
								<h4 class="text-sm font-semibold">Correct mapping</h4>
								<div class="grid gap-2">
									<NativeSelect.Root bind:value={correctionOfficialId} class="w-full">
										<NativeSelect.Option value="">Choose another official ID</NativeSelect.Option>
										{#each activeOfficials.filter((official) => official.officialId !== selectedVisual?.officialId && !official.mappingState) as official (official.officialId)}
											<NativeSelect.Option value={official.officialId}>
												{official.officialId}
											</NativeSelect.Option>
										{/each}
									</NativeSelect.Root>
									<NativeSelect.Root bind:value={correctionTrackletId} class="w-full">
										{#each selectedVisual.tracklets.filter((tracklet) => tracklet.evidenceStatus === 'eligible') as tracklet, index (tracklet.trackletId)}
											<NativeSelect.Option value={tracklet.trackletId}>
												Evidence {index + 1} · {tracklet.evidenceCount} frames
											</NativeSelect.Option>
										{/each}
									</NativeSelect.Root>
									<Button
										variant="outline"
										disabled={!correctionOfficialId || !correctionTrackletId}
										onclick={correct}
									>
										Correct mapping
									</Button>
								</div>
							</section>
							<Button variant="outline" class="w-full" onclick={deactivate}>
								<ArchiveIcon /> Deactivate
							</Button>
						{:else}
							<section class="space-y-2">
								<h4 class="text-sm font-semibold">Merge visual evidence</h4>
								<NativeSelect.Root bind:value={mergeTargetId} class="w-full">
									<NativeSelect.Option value="">Choose target identity</NativeSelect.Option>
									{#each mergeTargets as target, index (target.visualIdentityId)}
										<NativeSelect.Option value={target.visualIdentityId}>
											{target.officialId ?? `Visual identity ${index + 1}`}
										</NativeSelect.Option>
									{/each}
								</NativeSelect.Root>
								<Button variant="outline" class="w-full" disabled={!mergeTargetId} onclick={merge}>
									<MergeIcon /> Merge evidence
								</Button>
							</section>
						{/if}

						{#if selectedVisual.tracklets.length > 1}
							<section class="space-y-2">
								<h4 class="text-sm font-semibold">Split mixed evidence</h4>
								{#each selectedVisual.tracklets as tracklet (tracklet.trackletId)}
									<label class="flex items-center gap-3 rounded-md border p-2 text-sm">
										<input
											type="checkbox"
											checked={splitTracklets[tracklet.trackletId] ?? false}
											onchange={(event) =>
												(splitTracklets[tracklet.trackletId] = event.currentTarget.checked)}
										/>
										<img src={tracklet.preview} alt="" class="size-9 rounded-md object-cover" />
										<span class="min-w-0 flex-1 truncate">{tracklet.source}</span>
									</label>
								{/each}
								<Button
									variant="outline"
									class="w-full"
									disabled={!Object.values(splitTracklets).some(Boolean)}
									onclick={split}
								>
									<ScissorsIcon /> Split selected evidence
								</Button>
							</section>
						{/if}
					</div>
				</details>
			{/if}
		</div>
	</Sheet.Content>
</Sheet.Root>
