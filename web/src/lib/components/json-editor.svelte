<script lang="ts">
	import { onMount } from 'svelte';
	import type * as Monaco from 'monaco-editor';
	let {
		value = $bindable(''),
		schema,
		height = 420,
		hasErrors = $bindable(false)
	}: {
		value?: string;
		schema?: Record<string, unknown>;
		height?: number | string;
		hasErrors?: boolean;
	} = $props();

	let container = $state<HTMLDivElement | null>(null);
	let issues = $state<string[]>([]);
	let monaco: typeof import('monaco-editor') | undefined;
	let editor: Monaco.editor.IStandaloneCodeEditor | undefined;
	let model: Monaco.editor.ITextModel | undefined;
	const modelUri = 'inmemory://model/editor.json';
	const schemaUri = 'inmemory://schema/editor.schema.json';
	type JsonLanguageContribution = {
		jsonDefaults: {
			setDiagnosticsOptions(options: {
				validate: boolean;
				allowComments: boolean;
				enableSchemaRequest: boolean;
				schemas: Array<{ uri: string; fileMatch: string[]; schema: unknown }>;
			}): void;
		};
	};

	const syncMarkers = () => {
		if (!monaco || !model) return;
		const markers = monaco.editor.getModelMarkers({ resource: model.uri });
		hasErrors = markers.some((marker) => marker.severity === monaco?.MarkerSeverity.Error);
		issues = markers
			.filter(
				(marker) => marker.severity >= (monaco?.MarkerSeverity.Warning ?? Number.POSITIVE_INFINITY)
			)
			.map((marker) => `Line ${marker.startLineNumber}: ${marker.message}`);
	};

	$effect(() => {
		const nextValue = value ?? '';
		if (!editor) {
			return;
		}

		if (editor.hasWidgetFocus()) {
			return;
		}

		if (nextValue === editor.getValue()) {
			return;
		}

		editor.setValue(nextValue);
	});

	onMount(() => {
		let disposed = false;
		let cleanup = () => {};

		void (async () => {
			const [
				{ default: editorWorker },
				{ default: jsonWorker },
				monacoModule,
				{ default: jsonContribution }
			] = await Promise.all([
				import('monaco-editor/esm/vs/editor/editor.worker?worker'),
				import('monaco-editor/esm/vs/language/json/json.worker?worker'),
				import('monaco-editor'),
				import('monaco-editor/esm/vs/language/json/monaco.contribution.js')
			]);
			if (disposed) {
				return;
			}

			monaco = monacoModule;
			(
				self as typeof self & {
					MonacoEnvironment: {
						getWorker(moduleId: string, label: string): Worker;
					};
				}
			).MonacoEnvironment = {
				getWorker(_: string, label: string) {
					if (label === 'json') {
						return new jsonWorker();
					}

					return new editorWorker();
				}
			};

			(jsonContribution as unknown as JsonLanguageContribution).jsonDefaults.setDiagnosticsOptions({
				validate: true,
				allowComments: false,
				enableSchemaRequest: false,
				schemas: schema
					? [
							{
								uri: schemaUri,
								fileMatch: [modelUri],
								schema: JSON.parse(JSON.stringify(schema))
							}
						]
					: []
			});

			if (!container) {
				return;
			}

			model = monaco.editor.createModel(value ?? '', 'json', monaco.Uri.parse(modelUri));
			editor = monaco.editor.create(container, {
				model,
				automaticLayout: true,
				formatOnPaste: true,
				formatOnType: true,
				minimap: { enabled: false },
				scrollBeyondLastLine: false,
				tabSize: 2,
				insertSpaces: true,
				wordWrap: 'on'
			});

			const contentSubscription = editor.onDidChangeModelContent((event) => {
				if (!model) {
					return;
				}

				if (event.isFlush) {
					syncMarkers();
					return;
				}

				value = model.getValue();
				syncMarkers();
			});
			const modelUriString = model.uri.toString();

			const markerSubscription = monaco.editor.onDidChangeMarkers((resources) => {
				if (resources.some((resource) => resource.toString() === modelUriString)) {
					syncMarkers();
				}
			});

			syncMarkers();

			cleanup = () => {
				contentSubscription.dispose();
				markerSubscription.dispose();
				editor?.dispose();
				model?.dispose();
			};
			if (disposed) {
				cleanup();
			}
		})();

		return () => {
			disposed = true;
			cleanup();
		};
	});
</script>

<div class="space-y-2">
	<div
		bind:this={container}
		class="overflow-hidden rounded-md border border-input"
		style:height={typeof height === 'number' ? `${height}px` : height}
	></div>

	{#if issues.length > 0}
		<div
			class="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
		>
			{#each issues as issue, index (index)}
				<p>{issue}</p>
			{/each}
		</div>
	{/if}
</div>
