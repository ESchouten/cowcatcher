import tailwindcss from '@tailwindcss/vite';
import devtoolsJson from 'vite-plugin-devtools-json';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const buildTarget = process.env.AI_DETECTOR_WEB_TARGET?.trim().toLowerCase() || 'node';

export default defineConfig({
	define: {
		__AI_DETECTOR_WEB_TARGET__: JSON.stringify(buildTarget)
	},
	server: {
		allowedHosts: ['.local']
	},
	plugins: [tailwindcss(), sveltekit(), devtoolsJson()]
});
