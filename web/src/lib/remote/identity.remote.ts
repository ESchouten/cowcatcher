import { command, query } from '$app/server';
import * as v from 'valibot';
import {
	confirmMapping,
	correctMapping,
	createOfficialIdentity,
	createProvisionalMapping,
	deactivateMapping,
	mergeVisualIdentities,
	readIdentityCatalog,
	readCatalogControl,
	rollbackMapping,
	splitVisualIdentity,
	updateOfficialIdentity,
	type IdentityCatalogSnapshot
} from '$lib/server/identity-catalog';
import { getIdentityDatabases, openIdentityDatabase } from '$lib/server/identity-databases';
import { readConfigState } from '$lib/server/config-store';

export interface IdentityCatalogView {
	catalogId: string;
	label: string;
	targetLabel: string;
	state: 'ready' | 'not_initialized' | 'error';
	message: string | null;
	snapshot: IdentityCatalogSnapshot | null;
}

export interface IdentityStatusView {
	catalogId: string;
	label: string;
	state: IdentityCatalogView['state'];
	message: string | null;
	activeGalleryVersion: number | null;
}

function unavailableState(error: unknown): Pick<IdentityCatalogView, 'state' | 'message'> {
	const message = error instanceof Error ? error.message : 'Identity catalog is unavailable';
	return {
		state: message.includes('Start the detector')
			? ('not_initialized' as const)
			: ('error' as const),
		message
	};
}

export const getIdentityCatalogs = query(async (): Promise<IdentityCatalogView[]> => {
	const configured = await getIdentityDatabases();
	return Promise.all(
		configured.map(async (catalog) => {
			try {
				const database = await openIdentityDatabase(catalog.id);
				try {
					return {
						catalogId: catalog.id,
						label: catalog.label,
						targetLabel: catalog.targetLabel,
						state: 'ready' as const,
						message: null,
						snapshot: readIdentityCatalog(database)
					};
				} finally {
					database.close();
				}
			} catch (error) {
				return {
					catalogId: catalog.id,
					label: catalog.label,
					targetLabel: catalog.targetLabel,
					...unavailableState(error),
					snapshot: null
				};
			}
		})
	);
});

export const getIdentityStatus = query(async (): Promise<IdentityStatusView[]> => {
	const configured = await getIdentityDatabases();
	return Promise.all(
		configured.map(async (catalog) => {
			try {
				const database = await openIdentityDatabase(catalog.id);
				try {
					return {
						catalogId: catalog.id,
						label: catalog.label,
						state: 'ready' as const,
						message: null,
						activeGalleryVersion: readCatalogControl(database).activeGalleryVersion
					};
				} finally {
					database.close();
				}
			} catch (error) {
				return {
					catalogId: catalog.id,
					label: catalog.label,
					...unavailableState(error),
					activeGalleryVersion: null
				};
			}
		})
	);
});

export const getDetectorConnection = query(
	async (): Promise<'unconfigured' | 'connected' | 'disconnected'> => {
		const { config } = await readConfigState();
		const detector = config.detectors.find((item) => item.exporters?.sse?.[0]);
		const sse = detector?.exporters?.sse?.[0];
		if (!sse) return 'unconfigured';
		const host = process.env.DETECTOR_HOST?.trim() || '127.0.0.1';
		const endpoint = (sse.endpoint?.trim() || '/events/0').replace(/^\/*/, '/');
		const url = new URL(`http://${host}`);
		url.port = String(sse.port ?? 8765);
		url.pathname = endpoint.split('?', 1)[0];
		try {
			const response = await fetch(url, {
				headers: { Accept: 'text/event-stream' },
				signal: AbortSignal.timeout(750)
			});
			await response.body?.cancel();
			return response.ok ? 'connected' : 'disconnected';
		} catch {
			return 'disconnected';
		}
	}
);

const catalogInput = {
	catalogId: v.pipe(v.string(), v.trim(), v.minLength(1)),
	expectedRevision: v.pipe(v.number(), v.integer(), v.minValue(0))
};

async function withCatalog<T>(
	catalogId: string,
	action: (database: Awaited<ReturnType<typeof openIdentityDatabase>>) => T
): Promise<T> {
	const database = await openIdentityDatabase(catalogId);
	try {
		return action(database);
	} finally {
		database.close();
	}
}

export const addOfficialIdentity = command(
	v.object({
		...catalogInput,
		officialId: v.pipe(v.string(), v.trim(), v.minLength(1)),
		displayName: v.optional(v.string()),
		notes: v.optional(v.string())
	}),
	async ({ catalogId, expectedRevision, officialId, displayName, notes }) =>
		withCatalog(catalogId, (database) =>
			createOfficialIdentity(database, { officialId, displayName, notes }, { expectedRevision })
		)
);

export const editOfficialIdentity = command(
	v.object({
		...catalogInput,
		officialId: v.pipe(v.string(), v.trim(), v.minLength(1)),
		displayName: v.optional(v.string()),
		status: v.picklist(['active', 'archived']),
		notes: v.optional(v.string())
	}),
	async ({ catalogId, expectedRevision, officialId, displayName, status, notes }) =>
		withCatalog(catalogId, (database) =>
			updateOfficialIdentity(
				database,
				{ officialId, displayName, status, notes },
				{ expectedRevision }
			)
		)
);

export const provisionIdentity = command(
	v.object({
		...catalogInput,
		visualIdentityId: v.pipe(v.string(), v.regex(/^vid_/)),
		officialId: v.pipe(v.string(), v.trim(), v.minLength(1)),
		trackletId: v.pipe(v.string(), v.regex(/^trk_/))
	}),
	async ({ catalogId, expectedRevision, visualIdentityId, officialId, trackletId }) =>
		withCatalog(catalogId, (database) =>
			createProvisionalMapping(
				database,
				{ visualIdentityId, officialId, trackletId },
				{ expectedRevision }
			)
		)
);

export const confirmIdentity = command(
	v.object({
		...catalogInput,
		mappingId: v.pipe(v.string(), v.regex(/^map_/)),
		confirmationTrackletId: v.pipe(v.string(), v.regex(/^trk_/))
	}),
	async ({ catalogId, expectedRevision, mappingId, confirmationTrackletId }) =>
		withCatalog(catalogId, (database) =>
			confirmMapping(database, { mappingId, confirmationTrackletId }, { expectedRevision })
		)
);

export const correctIdentity = command(
	v.object({
		...catalogInput,
		mappingId: v.pipe(v.string(), v.regex(/^map_/)),
		officialId: v.pipe(v.string(), v.trim(), v.minLength(1)),
		provisionalTrackletId: v.pipe(v.string(), v.regex(/^trk_/))
	}),
	async ({ catalogId, expectedRevision, mappingId, officialId, provisionalTrackletId }) =>
		withCatalog(catalogId, (database) =>
			correctMapping(
				database,
				{ mappingId, officialId, provisionalTrackletId },
				{ expectedRevision }
			)
		)
);

export const deactivateIdentity = command(
	v.object({
		...catalogInput,
		mappingId: v.pipe(v.string(), v.regex(/^map_/))
	}),
	async ({ catalogId, expectedRevision, mappingId }) =>
		withCatalog(catalogId, (database) =>
			deactivateMapping(database, mappingId, { expectedRevision })
		)
);

export const rollbackIdentity = command(
	v.object({
		...catalogInput,
		mappingId: v.pipe(v.string(), v.regex(/^map_/))
	}),
	async ({ catalogId, expectedRevision, mappingId }) =>
		withCatalog(catalogId, (database) => rollbackMapping(database, mappingId, { expectedRevision }))
);

export const mergeIdentityEvidence = command(
	v.object({
		...catalogInput,
		sourceVisualIdentityId: v.pipe(v.string(), v.regex(/^vid_/)),
		targetVisualIdentityId: v.pipe(v.string(), v.regex(/^vid_/))
	}),
	async ({ catalogId, expectedRevision, sourceVisualIdentityId, targetVisualIdentityId }) =>
		withCatalog(catalogId, (database) =>
			mergeVisualIdentities(
				database,
				{ sourceVisualIdentityId, targetVisualIdentityId },
				{ expectedRevision }
			)
		)
);

export const splitIdentityEvidence = command(
	v.object({
		...catalogInput,
		sourceVisualIdentityId: v.pipe(v.string(), v.regex(/^vid_/)),
		trackletIds: v.pipe(v.array(v.pipe(v.string(), v.regex(/^trk_/))), v.minLength(1))
	}),
	async ({ catalogId, expectedRevision, sourceVisualIdentityId, trackletIds }) =>
		withCatalog(catalogId, (database) =>
			splitVisualIdentity(database, { sourceVisualIdentityId, trackletIds }, { expectedRevision })
		)
);
