import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const root = process.cwd();

function source(relativePath) {
	return readFileSync(path.join(root, relativePath), 'utf8');
}

test('active product navigation contains only the four generic areas', () => {
	const layout = source('src/routes/(admin)/+layout.svelte');
	const navigationBlock = layout.slice(
		layout.indexOf('const navigation:'),
		layout.indexOf('function active')
	);

	assert.deepEqual(
		[...navigationBlock.matchAll(/title: '([^']+)'/g)].map((match) => match[1]),
		['Live', 'Detections', 'Identities', 'Setup']
	);
	assert.doesNotMatch(layout, /Holstein|Monitoring|New cows|Mapping review|Live enrollment/);
});

test('the original neutral shadcn shell replaces the farm-themed mobile chrome', () => {
	const layout = source('src/routes/(admin)/+layout.svelte');
	const sidebar = source('src/lib/components/app-sidebar.svelte');
	const cameras = source('src/routes/(admin)/setup/cameras/+page.svelte');
	const theme = source('src/routes/layout.css');

	for (const contract of [
		/<Sidebar\.Provider>/,
		/<Sidebar\.Inset/,
		/<Sidebar\.Trigger/,
		/<Breadcrumb\.Root>/
	]) {
		assert.match(layout, contract);
	}
	assert.match(sidebar, /variant="inset"/);
	assert.match(sidebar, /collapsible="icon"/);
	assert.match(layout, /url: resolve\(item\.url\)/);
	assert.match(layout, /homeUrl=\{resolve\('\/live'\)\}/);
	assert.match(cameras, /href=\{resolve\('\/setup\/cameras\/add'\)\}/);
	assert.doesNotMatch(layout, /fixed inset-x-0 bottom-0|Private farm network|#123d2d/);
	assert.doesNotMatch(theme, /#123d2d|#668575/);
});

test('browser commands use opaque catalog IDs and expose no database path input', () => {
	const remote = source('src/lib/remote/identity.remote.ts');

	assert.match(remote, /catalogId:/);
	assert.doesNotMatch(remote, /database:\s*v\./);
	assert.doesNotMatch(remote, /HOLSTEIN_API|holsteinApi/);
});

test('the compact status panel reports the required four states', () => {
	const status = source('src/lib/components/identity-status-panel.svelte');

	for (const label of ['Detector', 'Database', 'Gallery', 'Last identity error']) {
		assert.match(status, new RegExp(label));
	}
});

test('the simplified identity workflow keeps every audited catalog action wired', () => {
	const page = source('src/routes/(admin)/identities/+page.svelte');
	const normalWorkflow = page.slice(page.indexOf('</script>'), page.indexOf('<Dialog.Root'));
	const advancedWorkflow = page.slice(page.indexOf('Advanced identity details'));

	for (const label of ['Needs review', 'Official records', 'Assign provisionally']) {
		assert.match(normalWorkflow, new RegExp(label));
	}
	for (const action of [
		'addOfficialIdentity',
		'editOfficialIdentity',
		'provisionIdentity',
		'confirmIdentity',
		'correctIdentity',
		'deactivateIdentity',
		'rollbackIdentity',
		'mergeIdentityEvidence',
		'splitIdentityEvidence'
	]) {
		assert.match(page, new RegExp(action));
	}

	assert.doesNotMatch(normalWorkflow, /Visual identity ID/);
	assert.match(advancedWorkflow, /Visual identity ID/);
	assert.doesNotMatch(page, /#123d2d|#668575|rounded-2xl|bg-white/);
});

test('legacy Holstein product and management proxy files are absent', () => {
	for (const relativePath of [
		'src/lib/remote/holstein.remote.ts',
		'src/lib/server/holstein-api.ts',
		'src/lib/server/identity-store.ts',
		'src/routes/(admin)/holstein-identities/+page.svelte',
		'src/routes/(admin)/holstein-monitoring/+page.svelte',
		'src/routes/api/holstein/evidence/[evidence]/+server.ts'
	]) {
		assert.equal(existsSync(path.join(root, relativePath)), false, relativePath);
	}
});
