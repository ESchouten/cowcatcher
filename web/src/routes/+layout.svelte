<script lang="ts">
	import './layout.css';
	import { Toaster } from '$lib/components/ui/sonner';

	let { children } = $props();
</script>

<svelte:head>
	<title>AI Detector</title>
</svelte:head>

{@render children()}
<Toaster />
