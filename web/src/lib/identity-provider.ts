import type { IdentityProviderConfig } from '$lib/schema';

const IDENTITY_PROVIDER_KEYS = [
	'id',
	'type',
	'database',
	'model',
	'segment_model',
	'segment_labels',
	'segment_confidence',
	'debug_directory',
	'match_threshold',
	'candidate_threshold',
	'create_after',
	'crop_padding'
] as const;

export function identityProviderConfig(identity: IdentityProviderConfig): IdentityProviderConfig {
	const source = identity as Record<string, unknown>;
	const provider: Record<string, unknown> = {};

	for (const key of IDENTITY_PROVIDER_KEYS) {
		if (source[key] !== undefined) {
			provider[key] = source[key];
		}
	}

	return provider as IdentityProviderConfig;
}
