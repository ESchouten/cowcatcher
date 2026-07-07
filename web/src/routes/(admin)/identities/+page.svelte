<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Table from '$lib/components/ui/table';
	import { Badge } from '$lib/components/ui/badge';
	import * as Empty from '$lib/components/ui/empty';
	import {
		getIdentities,
		getIdentityStores,
		renameIdentity,
		type IdentityRow,
		type IdentityStore
	} from '$lib/remote/identity.remote';
	import CircleAlertIcon from '@lucide/svelte/icons/circle-alert';
	import CircleCheckIcon from '@lucide/svelte/icons/circle-check';
	import DatabaseIcon from '@lucide/svelte/icons/database';
	import FingerprintIcon from '@lucide/svelte/icons/fingerprint';
	import PencilIcon from '@lucide/svelte/icons/pencil';
	import PlusIcon from '@lucide/svelte/icons/plus';
	import SaveIcon from '@lucide/svelte/icons/save';
	import { SvelteSet } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';

	const identitiesQuery = getIdentities();
	const storesQuery = getIdentityStores();
	const identities = $derived(await identitiesQuery);
	const stores = $derived(await storesQuery);

	let draftIdentities = $state<Record<string, string>>({});
	let saveErrors = $state<Record<string, string>>({});
	let savingKeys = new SvelteSet<string>();

	function rowKey(store: IdentityStore, identity: IdentityRow) {
		return [store.providerId, store.database, identity.identity]
			.map(encodeURIComponent)
			.join('__');
	}

	function draftIdentity(store: IdentityStore, identity: IdentityRow) {
		return draftIdentities[rowKey(store, identity)] ?? identity.identity;
	}

	function setDraftIdentity(store: IdentityStore, identity: IdentityRow, value: string) {
		draftIdentities = {
			...draftIdentities,
			[rowKey(store, identity)]: value
		};
	}

	function clearRowState(key: string) {
		const nextDrafts = { ...draftIdentities };
		const nextErrors = { ...saveErrors };
		delete nextDrafts[key];
		delete nextErrors[key];
		draftIdentities = nextDrafts;
		saveErrors = nextErrors;
	}

	function errorMessage(error: unknown) {
		if (error && typeof error === 'object' && 'body' in error) {
			const body = (error as { body?: { message?: string } }).body;
			if (typeof body?.message === 'string') {
				return body.message;
			}
		}

		return error instanceof Error ? error.message : 'Identity could not be saved';
	}

	async function saveIdentity(store: IdentityStore, identity: IdentityRow) {
		const key = rowKey(store, identity);
		const nextIdentity = draftIdentity(store, identity).trim();

		savingKeys.add(key);
		saveErrors = { ...saveErrors, [key]: '' };

		try {
			await renameIdentity({
				providerId: store.providerId,
				database: store.database,
				identity: identity.identity,
				nextIdentity
			}).updates(storesQuery);
			clearRowState(key);
			toast.warning(
				`Identity '${nextIdentity}' saved. Restart the detector service for the new identity to take effect.`,
				{ duration: Number.POSITIVE_INFINITY, closeButton: true }
			);
		} catch (saveError) {
			const message = errorMessage(saveError);
			saveErrors = { ...saveErrors, [key]: message };
			toast.error(message);
		} finally {
			savingKeys.delete(key);
		}
	}

	function formatDate(value: string | null) {
		if (!value) {
			return 'Never';
		}

		const date = new Date(value);
		if (Number.isNaN(date.getTime())) {
			return value;
		}

		return new Intl.DateTimeFormat(undefined, {
			dateStyle: 'medium',
			timeStyle: 'short'
		}).format(date);
	}

	function statusVariant(status: IdentityStore['status']) {
		return status === 'ready' ? 'default' : status === 'missing' ? 'secondary' : 'destructive';
	}

	function editIdentityHref(label: string) {
		return `/identities/add?label=${encodeURIComponent(label)}`;
	}
</script>

<section class="space-y-6">
	<header class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
		<div class="space-y-1">
			<h1 class="text-2xl font-semibold tracking-tight">Identities</h1>
			<p class="text-sm text-muted-foreground">
				Configure identity providers and review detections.
			</p>
		</div>
		<Button href="/identities/add">
			<PlusIcon />
			Add
		</Button>
	</header>

	{#if identities.length === 0}
		<Empty.Root>
			<Empty.Header>
				<Empty.Media variant="icon">
					<FingerprintIcon />
				</Empty.Media>
				<Empty.Title>No identities configured</Empty.Title>
			</Empty.Header>
			<Empty.Content>
				<Button href="/identities/add">
					<PlusIcon />
					Add identity
				</Button>
			</Empty.Content>
		</Empty.Root>
	{:else}
		<div class="space-y-4">
			<section class="space-y-3">
				<h2 class="text-lg font-medium">Configurations</h2>
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head>Label</Table.Head>
							<Table.Head>ID</Table.Head>
							<Table.Head>Database</Table.Head>
							<Table.Head>Model</Table.Head>
							<Table.Head class="text-right">Action</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each identities as identity (identity.label)}
							<Table.Row>
								<Table.Cell class="font-medium">{identity.label}</Table.Cell>
								<Table.Cell class="font-mono text-xs">{identity.id}</Table.Cell>
								<Table.Cell class="font-mono text-xs break-all">{identity.database}</Table.Cell>
								<Table.Cell class="text-xs">{identity.model ?? '-'}</Table.Cell>
								<Table.Cell class="text-right">
									<Button href={editIdentityHref(identity.label)} variant="outline">
										<PencilIcon />
										Edit
									</Button>
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			</section>

			{#each stores as store (store.providerId + store.database)}
				<section class="space-y-4 rounded-md border p-4">
					<div class="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
						<div class="min-w-0 space-y-1">
							<div class="flex flex-wrap items-center gap-2">
								<h2 class="font-medium">{store.label ?? store.providerId}</h2>
								{#if store.label && store.label !== store.providerId}
									<Badge variant="secondary">{store.providerId}</Badge>
								{/if}
								<Badge variant={statusVariant(store.status)}>{store.status}</Badge>
								{#if store.metadata.model || store.model}
									<Badge variant="secondary">{store.metadata.model ?? store.model}</Badge>
								{/if}
							</div>
							<p class="font-mono text-xs break-all text-muted-foreground">{store.database}</p>
						</div>
						<div class="flex flex-wrap gap-2 text-xs text-muted-foreground">
							<span class="rounded-md border px-2 py-1">{store.identityCount} identities</span>
							<span class="rounded-md border px-2 py-1">{store.sampleCount} samples</span>
							<span class="rounded-md border px-2 py-1">
								{store.unknownCandidateCount} candidates
							</span>
						</div>
					</div>

					{#if store.status === 'error'}
						<div class="flex items-start gap-3 rounded-md border border-destructive/30 p-4">
							<CircleAlertIcon class="mt-0.5 size-5 text-destructive" />
							<div class="space-y-1">
								<h3 class="font-medium">Database could not be opened</h3>
								<p class="text-sm text-muted-foreground">{store.error}</p>
							</div>
						</div>
					{:else if store.status === 'missing'}
						<div class="flex items-start gap-3 rounded-md border p-4">
							<DatabaseIcon class="mt-0.5 size-5 text-muted-foreground" />
							<div class="space-y-1">
								<h3 class="font-medium">Waiting for database</h3>
								<p class="text-sm text-muted-foreground">
									Run the detector with identity enabled to create {store.database}.
								</p>
							</div>
						</div>
					{:else if store.identities.length === 0}
						<div class="flex items-start gap-3 rounded-md border p-4">
							<CircleCheckIcon class="mt-0.5 size-5 text-muted-foreground" />
							<div class="space-y-1">
								<h3 class="font-medium">No identities yet</h3>
								<p class="text-sm text-muted-foreground">
									New identities appear here after repeated sightings pass the provider threshold.
								</p>
							</div>
						</div>
					{:else}
						<Table.Root>
							<Table.Header>
								<Table.Row>
									<Table.Head class="min-w-72">Identity</Table.Head>
									<Table.Head>Samples</Table.Head>
									<Table.Head>Last sample</Table.Head>
									<Table.Head>Updated</Table.Head>
									<Table.Head class="text-right">Action</Table.Head>
								</Table.Row>
							</Table.Header>
							<Table.Body>
								{#each store.identities as identity (identity.identity)}
									{@const key = rowKey(store, identity)}
									{@const draft = draftIdentity(store, identity)}
									<Table.Row>
										<Table.Cell class="min-w-72 align-top">
											<form
												id={`rename-${key}`}
												class="space-y-2"
												onsubmit={(event) => {
													event.preventDefault();
													saveIdentity(store, identity);
												}}
											>
												<Label for={`identity-${key}`} class="sr-only">Identity</Label>
												<Input
													id={`identity-${key}`}
													class="font-mono"
													value={draft}
													aria-invalid={saveErrors[key] ? 'true' : undefined}
													oninput={(event) =>
														setDraftIdentity(store, identity, event.currentTarget.value)}
												/>
												{#if saveErrors[key]}
													<p class="text-xs text-destructive">{saveErrors[key]}</p>
												{/if}
											</form>
										</Table.Cell>
										<Table.Cell class="align-top">
											<div class="space-y-0.5">
												<div>{identity.sampleCount}</div>
												{#if identity.sampleRows !== identity.sampleCount}
													<div class="text-xs text-muted-foreground">
														{identity.sampleRows} rows
													</div>
												{/if}
											</div>
										</Table.Cell>
										<Table.Cell class="align-top">{formatDate(identity.lastSampleAt)}</Table.Cell>
										<Table.Cell class="align-top">{formatDate(identity.updatedAt)}</Table.Cell>
										<Table.Cell class="text-right align-top">
											<Button
												type="submit"
												form={`rename-${key}`}
												disabled={savingKeys.has(key) || draft.trim() === identity.identity}
											>
												<SaveIcon />
												Save
											</Button>
										</Table.Cell>
									</Table.Row>
								{/each}
							</Table.Body>
						</Table.Root>
					{/if}
				</section>
			{/each}
		</div>
	{/if}
</section>
