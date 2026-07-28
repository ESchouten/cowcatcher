import { randomUUID } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';

export type OfficialIdentityStatus = 'active' | 'archived';
export type MappingState = 'provisional' | 'confirmed';
export type EvidenceStatus = 'eligible' | 'insufficient' | 'switch_risk';

export interface CatalogControl {
	operatorRevision: number;
}

export interface OfficialIdentityRecord {
	officialId: string;
	displayName: string | null;
	status: OfficialIdentityStatus;
	notes: string;
	visualIdentityId: string | null;
	mappingState: MappingState | null;
	preview: string | null;
}

export interface IdentityTrackletRecord {
	trackletId: string;
	source: string;
	lastCapturedAt: string;
	evidenceStatus: EvidenceStatus;
	evidenceCount: number;
	preview: string;
}

export interface VisualIdentityRecord {
	visualIdentityId: string;
	officialId: string | null;
	mappingState: MappingState | null;
	provisionalTrackletId: string | null;
	confirmationTrackletId: string | null;
	tracklets: IdentityTrackletRecord[];
}

export interface IdentityCatalogSnapshot {
	control: CatalogControl;
	officialIdentities: OfficialIdentityRecord[];
	visualIdentities: VisualIdentityRecord[];
}

export interface MutationOptions {
	expectedRevision: number;
}

interface MappingRow {
	visual_identity_id: string;
	official_id: string;
	state: MappingState;
	provisional_tracklet_id: string;
	confirmation_tracklet_id: string | null;
}

const EVIDENCE_FRAMES = 2;

function id(prefix: string): string {
	return `${prefix}_${randomUUID().replaceAll('-', '')}`;
}

function jpegDataUrl(value: Uint8Array | null): string | null {
	return value ? `data:image/jpeg;base64,${Buffer.from(value).toString('base64')}` : null;
}

function transaction<T>(database: DatabaseSync, options: MutationOptions, action: () => T): T {
	database.exec('BEGIN IMMEDIATE');
	try {
		const control = database
			.prepare('SELECT operator_revision FROM control WHERE singleton = 1')
			.get() as { operator_revision: number } | undefined;
		if (!control) throw new Error('Identity catalog control record is missing');
		if (Number(control.operator_revision) !== options.expectedRevision) {
			throw new Error('The identity catalog changed. Refresh and try again.');
		}

		const result = action();
		database
			.prepare(
				`UPDATE control
				 SET operator_revision = ?
				 WHERE singleton = 1`
			)
			.run(Number(control.operator_revision) + 1);
		database.exec('COMMIT');
		return result;
	} catch (error) {
		database.exec('ROLLBACK');
		throw error;
	}
}

export function readCatalogControl(database: DatabaseSync): CatalogControl {
	const row = database
		.prepare('SELECT operator_revision FROM control WHERE singleton = 1')
		.get() as { operator_revision: number } | undefined;
	if (!row) throw new Error('Identity catalog control record is missing');
	return { operatorRevision: Number(row.operator_revision) };
}

export function readIdentityCatalog(database: DatabaseSync): IdentityCatalogSnapshot {
	const officials = database
		.prepare(
			`SELECT oi.official_id, oi.display_name, oi.status, oi.notes,
			        m.visual_identity_id, m.state AS mapping_state,
			        (
			          SELECT t.preview_jpeg
			          FROM visual_identity_tracklets vit
			          JOIN tracklets t ON t.tracklet_id = vit.tracklet_id
			          WHERE vit.visual_identity_id = m.visual_identity_id
			          ORDER BY t.last_captured_at DESC, t.tracklet_id
			          LIMIT 1
			        ) AS preview_jpeg
			 FROM official_identities oi
			 LEFT JOIN mappings m ON m.official_id = oi.official_id
			 ORDER BY oi.status, oi.official_id`
		)
		.all() as Array<{
		official_id: string;
		display_name: string | null;
		status: OfficialIdentityStatus;
		notes: string;
		visual_identity_id: string | null;
		mapping_state: MappingState | null;
		preview_jpeg: Uint8Array | null;
	}>;

	const visuals = database
		.prepare(
			`SELECT vi.visual_identity_id, m.official_id,
			        m.state AS mapping_state,
			        m.provisional_tracklet_id, m.confirmation_tracklet_id
			 FROM visual_identities vi
			 LEFT JOIN mappings m ON m.visual_identity_id = vi.visual_identity_id
			 WHERE vi.merged_into_visual_identity_id IS NULL
			 ORDER BY
			   CASE WHEN m.visual_identity_id IS NULL THEN 0 ELSE 1 END,
			   vi.visual_identity_id`
		)
		.all() as Array<{
		visual_identity_id: string;
		official_id: string | null;
		mapping_state: MappingState | null;
		provisional_tracklet_id: string | null;
		confirmation_tracklet_id: string | null;
	}>;

	const tracklets = database.prepare(
		`SELECT t.tracklet_id, t.source, t.last_captured_at,
		        t.evidence_status, t.preview_jpeg,
		        COUNT(ef.frame_index) AS evidence_count
		 FROM visual_identity_tracklets vit
		 JOIN tracklets t ON t.tracklet_id = vit.tracklet_id
		 LEFT JOIN evidence_frames ef ON ef.tracklet_id = t.tracklet_id
		 WHERE vit.visual_identity_id = ?
		 GROUP BY t.tracklet_id
		 ORDER BY t.last_captured_at DESC, t.tracklet_id`
	);

	return {
		control: readCatalogControl(database),
		officialIdentities: officials.map((row) => ({
			officialId: row.official_id,
			displayName: row.display_name,
			status: row.status,
			notes: row.notes,
			visualIdentityId: row.visual_identity_id,
			mappingState: row.mapping_state,
			preview: jpegDataUrl(row.preview_jpeg)
		})),
		visualIdentities: visuals.map((visual) => ({
			visualIdentityId: visual.visual_identity_id,
			officialId: visual.official_id,
			mappingState: visual.mapping_state,
			provisionalTrackletId: visual.provisional_tracklet_id,
			confirmationTrackletId: visual.confirmation_tracklet_id,
			tracklets: (
				tracklets.all(visual.visual_identity_id) as Array<{
					tracklet_id: string;
					source: string;
					last_captured_at: string;
					evidence_status: EvidenceStatus;
					preview_jpeg: Uint8Array;
					evidence_count: number;
				}>
			).map((tracklet) => ({
				trackletId: tracklet.tracklet_id,
				source: tracklet.source,
				lastCapturedAt: tracklet.last_captured_at,
				evidenceStatus: tracklet.evidence_status,
				evidenceCount: Number(tracklet.evidence_count),
				preview: jpegDataUrl(tracklet.preview_jpeg) ?? ''
			}))
		}))
	};
}

function officialIdentity(
	database: DatabaseSync,
	officialId: string
): { status: OfficialIdentityStatus } {
	const row = database
		.prepare('SELECT status FROM official_identities WHERE official_id = ?')
		.get(officialId) as { status: OfficialIdentityStatus } | undefined;
	if (!row) throw new Error('Official identity does not exist');
	return row;
}

function mappingForVisual(database: DatabaseSync, visualIdentityId: string): MappingRow | null {
	return (
		(database
			.prepare(
				`SELECT visual_identity_id, official_id, state,
				        provisional_tracklet_id, confirmation_tracklet_id
				 FROM mappings
				 WHERE visual_identity_id = ?`
			)
			.get(visualIdentityId) as MappingRow | undefined) ?? null
	);
}

function mappingForOfficial(database: DatabaseSync, officialId: string): MappingRow | null {
	return (
		(database
			.prepare(
				`SELECT visual_identity_id, official_id, state,
				        provisional_tracklet_id, confirmation_tracklet_id
				 FROM mappings
				 WHERE official_id = ?`
			)
			.get(officialId) as MappingRow | undefined) ?? null
	);
}

function eligibleTracklet(
	database: DatabaseSync,
	visualIdentityId: string,
	trackletId: string
): void {
	const row = database
		.prepare(
			`SELECT t.evidence_status, vit.visual_identity_id,
			        COUNT(ef.frame_index) AS evidence_count
			 FROM tracklets t
			 JOIN visual_identity_tracklets vit ON vit.tracklet_id = t.tracklet_id
			 LEFT JOIN evidence_frames ef ON ef.tracklet_id = t.tracklet_id
			 WHERE t.tracklet_id = ?
			 GROUP BY t.tracklet_id`
		)
		.get(trackletId) as
		| {
				evidence_status: EvidenceStatus;
				visual_identity_id: string;
				evidence_count: number;
		  }
		| undefined;
	if (!row || row.visual_identity_id !== visualIdentityId) {
		throw new Error('The selected evidence does not belong to this visual identity');
	}
	if (row.evidence_status !== 'eligible') {
		throw new Error('Only eligible, switch-free evidence can map an identity');
	}
	if (Number(row.evidence_count) !== EVIDENCE_FRAMES) {
		throw new Error(`This tracklet needs ${EVIDENCE_FRAMES} eligible evidence frames`);
	}
}

export function createOfficialIdentity(
	database: DatabaseSync,
	record: {
		officialId: string;
		displayName?: string;
		notes?: string;
	},
	options: MutationOptions
): string {
	const officialId = record.officialId.trim();
	if (!officialId) throw new Error('Official ID is required');
	return transaction(database, options, () => {
		database
			.prepare(
				`INSERT INTO official_identities (
					official_id, display_name, notes
				) VALUES (?, ?, ?)`
			)
			.run(officialId, record.displayName?.trim() || null, record.notes?.trim() || '');
		return officialId;
	});
}

export function updateOfficialIdentity(
	database: DatabaseSync,
	record: {
		officialId: string;
		displayName?: string;
		status: OfficialIdentityStatus;
		notes?: string;
	},
	options: MutationOptions
): void {
	transaction(database, options, () => {
		officialIdentity(database, record.officialId);
		if (record.status === 'archived' && mappingForOfficial(database, record.officialId)) {
			throw new Error('Deactivate the active mapping before archiving this identity');
		}
		database
			.prepare(
				`UPDATE official_identities
				 SET display_name = ?, status = ?, notes = ?
				 WHERE official_id = ?`
			)
			.run(
				record.displayName?.trim() || null,
				record.status,
				record.notes?.trim() || '',
				record.officialId
			);
	});
}

export function createProvisionalMapping(
	database: DatabaseSync,
	input: {
		visualIdentityId: string;
		officialId: string;
		trackletId: string;
	},
	options: MutationOptions
): void {
	transaction(database, options, () => {
		const official = officialIdentity(database, input.officialId);
		if (official.status !== 'active') throw new Error('Archived identities cannot be mapped');
		if (mappingForVisual(database, input.visualIdentityId)) {
			throw new Error('This visual identity already has a mapping');
		}
		if (mappingForOfficial(database, input.officialId)) {
			throw new Error('This official identity already has a mapping');
		}
		eligibleTracklet(database, input.visualIdentityId, input.trackletId);
		database
			.prepare(
				`INSERT INTO mappings (
					visual_identity_id, official_id, state,
					provisional_tracklet_id, confirmation_tracklet_id
				) VALUES (?, ?, 'provisional', ?, NULL)`
			)
			.run(input.visualIdentityId, input.officialId, input.trackletId);
	});
}

export function confirmMapping(
	database: DatabaseSync,
	input: { visualIdentityId: string; confirmationTrackletId: string },
	options: MutationOptions
): void {
	transaction(database, options, () => {
		const mapping = mappingForVisual(database, input.visualIdentityId);
		if (!mapping || mapping.state !== 'provisional') {
			throw new Error('Only a provisional mapping can be confirmed');
		}
		if (mapping.provisional_tracklet_id === input.confirmationTrackletId) {
			throw new Error('Confirmation requires a distinct tracklet');
		}
		eligibleTracklet(database, input.visualIdentityId, mapping.provisional_tracklet_id);
		eligibleTracklet(database, input.visualIdentityId, input.confirmationTrackletId);
		database
			.prepare(
				`UPDATE mappings
				 SET state = 'confirmed', confirmation_tracklet_id = ?
				 WHERE visual_identity_id = ?`
			)
			.run(input.confirmationTrackletId, input.visualIdentityId);
	});
}

export function correctMapping(
	database: DatabaseSync,
	input: {
		visualIdentityId: string;
		officialId: string;
		provisionalTrackletId: string;
	},
	options: MutationOptions
): void {
	transaction(database, options, () => {
		const mapping = mappingForVisual(database, input.visualIdentityId);
		if (!mapping) throw new Error('Identity mapping does not exist');
		if (mapping.official_id === input.officialId) {
			throw new Error('Choose a different official identity for a correction');
		}
		const official = officialIdentity(database, input.officialId);
		if (official.status !== 'active') throw new Error('Archived identities cannot be mapped');
		if (mappingForOfficial(database, input.officialId)) {
			throw new Error('This official identity already has a mapping');
		}
		eligibleTracklet(database, input.visualIdentityId, input.provisionalTrackletId);
		database
			.prepare(
				`UPDATE mappings
				 SET official_id = ?, state = 'provisional',
				     provisional_tracklet_id = ?, confirmation_tracklet_id = NULL
				 WHERE visual_identity_id = ?`
			)
			.run(input.officialId, input.provisionalTrackletId, input.visualIdentityId);
	});
}

export function deactivateMapping(
	database: DatabaseSync,
	visualIdentityId: string,
	options: MutationOptions
): void {
	transaction(database, options, () => {
		const result = database
			.prepare('DELETE FROM mappings WHERE visual_identity_id = ?')
			.run(visualIdentityId);
		if (!result.changes) throw new Error('Identity mapping does not exist');
	});
}

export function mergeVisualIdentities(
	database: DatabaseSync,
	input: { sourceVisualIdentityId: string; targetVisualIdentityId: string },
	options: MutationOptions
): void {
	if (input.sourceVisualIdentityId === input.targetVisualIdentityId) {
		throw new Error('Choose two different visual identities');
	}
	transaction(database, options, () => {
		const rows = database
			.prepare(
				`SELECT visual_identity_id, merged_into_visual_identity_id
				 FROM visual_identities
				 WHERE visual_identity_id IN (?, ?)`
			)
			.all(input.sourceVisualIdentityId, input.targetVisualIdentityId) as Array<{
			visual_identity_id: string;
			merged_into_visual_identity_id: string | null;
		}>;
		if (rows.length !== 2 || rows.some((row) => row.merged_into_visual_identity_id !== null)) {
			throw new Error('Both visual identities must be active');
		}
		if (mappingForVisual(database, input.sourceVisualIdentityId)) {
			throw new Error('Deactivate or correct the source mapping before merging evidence');
		}
		const evidence = database
			.prepare(
				`SELECT 1 FROM visual_identity_tracklets
				 WHERE visual_identity_id = ? LIMIT 1`
			)
			.get(input.sourceVisualIdentityId);
		if (!evidence) throw new Error('The source identity has no evidence to merge');
		database
			.prepare(
				`UPDATE visual_identity_tracklets
				 SET visual_identity_id = ?
				 WHERE visual_identity_id = ?`
			)
			.run(input.targetVisualIdentityId, input.sourceVisualIdentityId);
		database
			.prepare(
				`UPDATE visual_identities
				 SET merged_into_visual_identity_id = ?
				 WHERE visual_identity_id = ?`
			)
			.run(input.targetVisualIdentityId, input.sourceVisualIdentityId);
	});
}

export function splitVisualIdentity(
	database: DatabaseSync,
	input: { sourceVisualIdentityId: string; trackletIds: string[] },
	options: MutationOptions
): string {
	const selected = [...new Set(input.trackletIds)];
	if (!selected.length) throw new Error('Select at least one tracklet to split');
	return transaction(database, options, () => {
		const all = database
			.prepare(
				`SELECT tracklet_id FROM visual_identity_tracklets
				 WHERE visual_identity_id = ? ORDER BY tracklet_id`
			)
			.all(input.sourceVisualIdentityId) as Array<{ tracklet_id: string }>;
		const allIds = new Set(all.map((row) => row.tracklet_id));
		if (selected.some((trackletId) => !allIds.has(trackletId))) {
			throw new Error('A selected tracklet does not belong to this visual identity');
		}
		if (selected.length >= all.length) {
			throw new Error('A split must leave evidence with the source identity');
		}

		const newVisualIdentityId = id('vid');
		database
			.prepare('INSERT INTO visual_identities (visual_identity_id) VALUES (?)')
			.run(newVisualIdentityId);
		const placeholders = selected.map(() => '?').join(',');
		database
			.prepare(
				`UPDATE visual_identity_tracklets
				 SET visual_identity_id = ?
				 WHERE visual_identity_id = ?
				   AND tracklet_id IN (${placeholders})`
			)
			.run(newVisualIdentityId, input.sourceVisualIdentityId, ...selected);

		const mapping = mappingForVisual(database, input.sourceVisualIdentityId);
		if (
			mapping &&
			(selected.includes(mapping.provisional_tracklet_id) ||
				(mapping.confirmation_tracklet_id !== null &&
					selected.includes(mapping.confirmation_tracklet_id)))
		) {
			database
				.prepare('DELETE FROM mappings WHERE visual_identity_id = ?')
				.run(input.sourceVisualIdentityId);
		}
		return newVisualIdentityId;
	});
}
