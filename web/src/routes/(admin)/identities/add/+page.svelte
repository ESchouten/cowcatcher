<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import JsonEditor from '$lib/components/json-editor.svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import { Switch } from '$lib/components/ui/switch';
	import {
		deleteIdentityConfig,
		getIdentity,
		getIdentityPreset,
		getIdentityPresets,
		getIdentitySchema,
		saveIdentityConfig
	} from '$lib/remote/identity.remote';
	import type { IdentityMeta, IdentityProviderConfig } from '$lib/schema';
	import { toast } from 'svelte-sonner';

	const EMPTY_IDENTITY: IdentityProviderConfig = {
		id: 'cow-main',
		type: 'wildlife_tools',
		database: 'identities.sqlite',
		model: 'hf-hub:BVRA/MegaDescriptor-T-224',
		segment_labels: ['cow']
	};

	function stripMeta(identity?: IdentityMeta): Partial<IdentityProviderConfig> | undefined {
		if (!identity) {
			return undefined;
		}

		const config = structuredClone(identity) as Partial<IdentityProviderConfig> & {
			label?: string;
		};
		delete config.label;
		return config;
	}

	function mergeWithEmptyIdentity(identity?: Partial<IdentityProviderConfig>) {
		return {
			...EMPTY_IDENTITY,
			...identity
		};
	}

	function getPresetLabel(presetFile: string) {
		return presetFile
			.replace(/\.json$/i, '')
			.split('-')
			.filter(Boolean)
			.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
			.join(' ');
	}

	async function handlePresetChange(file: string) {
		const preset = await getIdentityPreset({ file });
		identity = mergeWithEmptyIdentity(preset);
		if (!isEditing && !label.trim()) {
			label = getPresetLabel(file);
		}
	}

	async function handleSave(event: SubmitEvent) {
		event.preventDefault();
		if (editorHasErrors) {
			return;
		}

		await saveIdentityConfig({
			original: originalLabel || undefined,
			identity,
			meta: { label }
		});
		toast.warning(
			`Identity configuration '${label}' saved. Restart the detector to apply the changes.`,
			{ duration: Number.POSITIVE_INFINITY, closeButton: true }
		);
		await goto(resolve(setupMode ? '/setup' : '/identities'));
	}

	const originalLabel = page.url.searchParams.get('label') ?? '';
	const isEditing = !!originalLabel;
	const setupMode = page.url.searchParams.get('setup') === '1';
	const identityPresets = $state(await getIdentityPresets());
	const identitySchema = $state(await getIdentitySchema());
	const loadedIdentity = isEditing ? await getIdentity({ label: originalLabel }) : undefined;

	let label = $state(loadedIdentity?.label ?? originalLabel);
	let identity = $state(mergeWithEmptyIdentity(stripMeta(loadedIdentity)));
	let preset = $state<string>('Custom');
	let advanced = $state(false);
	let editorHasErrors = $state(false);
</script>

<section class="space-y-6">
	<header class="space-y-1">
		<h1 class="text-2xl font-semibold tracking-tight">
			{setupMode ? 'Setup: Add Identity' : isEditing ? 'Edit Identity' : 'Add Identity'}
		</h1>
		<p class="text-sm text-muted-foreground">Configure an identity provider.</p>
	</header>

	<form class="flex max-w-2xl flex-col gap-2" onsubmit={handleSave}>
		<div class="flex gap-6">
			<div class="flex flex-1 flex-col gap-2">
				<Label for="label">Label</Label>
				<Input id="label" name="label" bind:value={label} placeholder="e.g. Cow identity" />
			</div>

			<div class="flex flex-1 flex-col gap-2">
				<Label for="presets">Presets</Label>
				<Select.Root
					type="single"
					bind:value={preset}
					onValueChange={handlePresetChange}
					items={['Custom', ...identityPresets].map((preset) => ({
						value: preset,
						label: getPresetLabel(preset)
					}))}
				>
					<Select.Trigger id="presets" class="w-full">
						{getPresetLabel(preset)}
					</Select.Trigger>
					<Select.Content>
						{#each identityPresets as presetFile (presetFile)}
							<Select.Item value={presetFile} label={getPresetLabel(presetFile)}></Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>
			</div>
		</div>

		<div class="mt-2 grid gap-6 md:grid-cols-2">
			<div class="flex flex-col gap-2">
				<Label for="id">ID</Label>
				<Input id="id" name="id" bind:value={identity.id} placeholder="e.g. cow-main" />
			</div>
			<div class="flex flex-col gap-2">
				<Label for="database">Database</Label>
				<Input
					id="database"
					name="database"
					bind:value={identity.database}
					placeholder="e.g. identities.sqlite"
				/>
			</div>
		</div>

		<Label for="model" class="mt-2">Model</Label>
		<Input id="model" name="model" bind:value={identity.model} />

		<div class="mt-2 flex items-center justify-end space-x-2">
			<Switch id="advanced" bind:checked={advanced} />
			<Label for="advanced">Advanced</Label>
		</div>
		{#if advanced}
			<Label class="mt-2">identity</Label>
			<JsonEditor
				bind:value={
					() => JSON.stringify(identity, null, 2),
					(value) => {
						try {
							identity = JSON.parse(value);
						} catch {
							// Do nothing
						}
					}
				}
				bind:hasErrors={editorHasErrors}
				schema={identitySchema}
				height={420}
			/>
		{/if}

		<div class="mt-2 flex gap-2">
			{#if isEditing}
				<Button
					type="button"
					onclick={async () => {
						await deleteIdentityConfig({ label: originalLabel });
						await goto(resolve('/identities'));
					}}
					variant="destructive"
					class="flex-1">Delete</Button
				>
			{/if}
			<Button type="submit" class="flex-1" disabled={editorHasErrors}>Save</Button>
		</div>
	</form>
</section>
