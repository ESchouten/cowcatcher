<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import * as Select from '$lib/components/ui/select';
	import * as Table from '$lib/components/ui/table';
	import {
		finalizeCowIdentities,
		getCowIdentities,
		getCowIdentityTracklets,
		mergeCowIdentities,
		splitCowTracklet,
		setCowAnimalNumber
	} from '$lib/remote/identity.remote';
	import type { CowTracklet } from '$lib/remote/identity.remote';
	import CheckCheckIcon from '@lucide/svelte/icons/check-check';
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	import ChevronRightIcon from '@lucide/svelte/icons/chevron-right';
	import MergeIcon from '@lucide/svelte/icons/merge';
	import ScissorsIcon from '@lucide/svelte/icons/scissors';

	const identitiesQuery = getCowIdentities();
	const databases = $derived(await identitiesQuery);
	let pending = $state<string | null>(null);
	let errorMessage = $state<string | null>(null);
	let expanded = $state<string | null>(null);
	let loadingTracklets = $state<string | null>(null);
	let tracklets = $state<Record<string, CowTracklet[]>>({});
	let mergeTargets = $state<Record<string, string>>({});

	function identityKey(database: string, identity: string) {
		return `${database}:${identity}`;
	}

	async function loadTracklets(database: string, identity: string, refresh = false) {
		const key = identityKey(database, identity);
		loadingTracklets = key;
		try {
			const query = getCowIdentityTracklets({ database, identity });
			if (refresh) await query.refresh();
			tracklets[key] = await query;
		} finally {
			loadingTracklets = null;
		}
	}

	async function toggleIdentity(database: string, identity: string) {
		const key = identityKey(database, identity);
		if (expanded === key) {
			expanded = null;
			return;
		}
		expanded = key;
		if (!tracklets[key]) await loadTracklets(database, identity);
	}

	async function finalize(database: string) {
		pending = `finalize:${database}`;
		errorMessage = null;
		try {
			await finalizeCowIdentities({ database }).updates(identitiesQuery);
		} catch (error) {
			errorMessage = error instanceof Error ? error.message : 'Could not finalize enrollment.';
		} finally {
			pending = null;
		}
	}

	async function saveAnimalNumber(database: string, identity: string, animalNumber: string) {
		const key = `${database}:${identity}`;
		pending = key;
		errorMessage = null;
		try {
			await setCowAnimalNumber({ database, identity, animalNumber }).updates(identitiesQuery);
		} catch (error) {
			errorMessage = error instanceof Error ? error.message : 'Could not save animal number.';
		} finally {
			pending = null;
		}
	}

	async function mergeIdentity(database: string, source: string) {
		const identity = identityKey(database, source);
		const target = mergeTargets[identity];
		if (!target) return;
		if (!window.confirm(`Merge ${source} into ${target}?`)) return;
		pending = `merge:${database}:${source}`;
		errorMessage = null;
		try {
			await mergeCowIdentities({ database, source, target }).updates(identitiesQuery);
			expanded = null;
			delete tracklets[identityKey(database, source)];
			delete mergeTargets[identity];
		} catch (error) {
			errorMessage = error instanceof Error ? error.message : 'Could not merge identities.';
		} finally {
			pending = null;
		}
	}

	async function splitTracklet(database: string, identity: string, tracklet: string) {
		if (!window.confirm(`Split this tracklet from ${identity}?`)) return;
		const key = `split:${tracklet}`;
		pending = key;
		errorMessage = null;
		try {
			await splitCowTracklet({ database, identity, tracklet }).updates(identitiesQuery);
			await loadTracklets(database, identity, true);
		} catch (error) {
			errorMessage = error instanceof Error ? error.message : 'Could not split tracklet.';
		} finally {
			pending = null;
		}
	}

	$effect(() => {
		if (!databases.some((database) => database.finalizeRequested)) return;
		const timer = window.setInterval(() => void identitiesQuery.refresh(), 1500);
		return () => window.clearInterval(timer);
	});
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<h1 class="text-2xl font-semibold tracking-tight">Cattle identities</h1>
		<p class="text-sm text-muted-foreground">Review scanned cattle and assign animal numbers.</p>
	</header>

	{#if errorMessage}
		<p class="text-sm font-medium text-destructive">{errorMessage}</p>
	{/if}

	{#if databases.length === 0}
		<p class="text-sm text-muted-foreground">No DazzleCow enrollment database is configured.</p>
	{:else}
		{#each databases as database (database.database)}
			<section class="space-y-3">
				<header class="flex flex-wrap items-center justify-between gap-3">
					<div class="flex items-center gap-2">
						<h2 class="text-base font-semibold">{database.label}</h2>
						{#if database.identities.length > 0}
							<Badge variant="secondary">{database.identities.length} identities</Badge>
						{:else if database.finalizeRequested}
							<Badge variant="outline">Finalizing</Badge>
						{:else}
							<Badge variant="outline">{database.tracklets} tracklets</Badge>
						{/if}
					</div>

					{#if database.identities.length === 0 && database.tracklets > 0}
						<Button
							type="button"
							size="sm"
							disabled={database.finalizeRequested || pending === `finalize:${database.database}`}
							onclick={() => finalize(database.database)}
						>
							<CheckCheckIcon />
							{database.finalizeRequested ? 'Finalizing' : 'Finish scan'}
						</Button>
					{/if}
				</header>

				{#if database.finalizeError}
					<p class="text-sm font-medium text-destructive">{database.finalizeError}</p>
				{/if}

				{#if database.identities.length > 0}
					<div class="overflow-hidden border">
						<Table.Root>
							<Table.Header>
								<Table.Row>
									<Table.Head class="w-12"></Table.Head>
									<Table.Head class="w-20">Preview</Table.Head>
									<Table.Head>Identity</Table.Head>
									<Table.Head class="w-28">Tracklets</Table.Head>
									<Table.Head class="w-72">Animal number</Table.Head>
								</Table.Row>
							</Table.Header>
							<Table.Body>
								{#each database.identities as identity (identity.identity)}
									<Table.Row>
										<Table.Cell>
											<Button
												type="button"
												variant="ghost"
												size="icon"
												title="Review tracklets"
												onclick={() => toggleIdentity(database.database, identity.identity)}
											>
												{#if expanded === identityKey(database.database, identity.identity)}
													<ChevronDownIcon />
												{:else}
													<ChevronRightIcon />
												{/if}
											</Button>
										</Table.Cell>
										<Table.Cell>
											{#if identity.preview}
												<img src={identity.preview} alt="" class="size-12 object-cover" />
											{:else}
												<div class="size-12 bg-muted"></div>
											{/if}
										</Table.Cell>
										<Table.Cell class="font-mono text-xs">{identity.identity}</Table.Cell>
										<Table.Cell>{identity.tracklets}</Table.Cell>
										<Table.Cell>
											<Input
												value={identity.animalNumber ?? ''}
												placeholder="Unassigned"
												disabled={pending === `${database.database}:${identity.identity}`}
												onchange={(event) =>
													saveAnimalNumber(
														database.database,
														identity.identity,
														event.currentTarget.value
													)}
											/>
										</Table.Cell>
									</Table.Row>
									{#if expanded === identityKey(database.database, identity.identity)}
										{@const mergeKey = identityKey(database.database, identity.identity)}
										<Table.Row>
											<Table.Cell colspan={5} class="bg-muted/30 p-4">
												<div class="mb-4 flex flex-wrap items-center justify-between gap-3">
													<p class="text-sm font-medium">Identity evidence</p>
													<div class="flex min-w-72 items-center gap-2">
														<Select.Root
															type="single"
															value={mergeTargets[mergeKey] ?? ''}
															onValueChange={(value) => (mergeTargets[mergeKey] = value)}
														>
															<Select.Trigger class="h-8 flex-1">
																{mergeTargets[mergeKey] ?? 'Merge into...'}
															</Select.Trigger>
															<Select.Content>
																{#each database.identities.filter((item) => item.identity !== identity.identity) as target}
																	<Select.Item value={target.identity} label={target.identity} />
																{/each}
															</Select.Content>
														</Select.Root>
														<Button
															type="button"
															size="sm"
															variant="outline"
															disabled={!mergeTargets[mergeKey] ||
																pending === `merge:${database.database}:${identity.identity}`}
															onclick={() => mergeIdentity(database.database, identity.identity)}
														>
															<MergeIcon />
															Merge
														</Button>
													</div>
												</div>

												{#if loadingTracklets === identityKey(database.database, identity.identity)}
													<p class="text-sm text-muted-foreground">Loading tracklets...</p>
												{:else}
													<div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
														{#each tracklets[identityKey(database.database, identity.identity)] ?? [] as tracklet (tracklet.id)}
															<div class="flex min-w-0 items-center gap-3 border bg-background p-2">
																<img
																	src={tracklet.preview}
																	alt=""
																	class="size-16 shrink-0 object-cover"
																/>
																<div class="min-w-0 flex-1">
																	<p class="truncate text-xs font-medium" title={tracklet.source}>
																		{tracklet.source}
																	</p>
																	<p class="text-xs text-muted-foreground">
																		{tracklet.observations} observations
																	</p>
																</div>
																<Button
																	type="button"
																	variant="ghost"
																	size="icon"
																	title="Split into a new identity"
																	disabled={identity.tracklets <= 1 ||
																		pending === `split:${tracklet.id}`}
																	onclick={() =>
																		splitTracklet(
																			database.database,
																			identity.identity,
																			tracklet.id
																		)}
																>
																	<ScissorsIcon />
																</Button>
															</div>
														{/each}
													</div>
												{/if}
											</Table.Cell>
										</Table.Row>
									{/if}
								{/each}
							</Table.Body>
						</Table.Root>
					</div>
				{:else if database.tracklets === 0}
					<p class="text-sm text-muted-foreground">Waiting for cattle tracklets.</p>
				{/if}
			</section>
		{/each}
	{/if}
</section>
