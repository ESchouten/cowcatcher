import { redirect } from '@sveltejs/kit';
import { getDetectors } from '$lib/remote/detector.remote';

export async function load() {
	const detectors = await getDetectors();

	throw redirect(302, detectors.length > 0 ? '/live' : '/setup/wizard');
}
