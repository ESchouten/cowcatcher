import { randomUUID } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';

export type OfficialIdentityStatus = 'active' | 'archived';
export type MappingState = 'provisional' | 'confirmed' | 'inactive' | 'rejected';
export type EvidenceStatus = 'eligible' | 'insufficient' | 'switch_risk';

export interface CatalogControl {
	operatorRevision: number;
	activeGalleryVersion: number | null;
}

export interface OfficialIdentityRecord {
	officialId: string;
	displayName: string | null;
	status: OfficialIdentityStatus;
	notes: string;
	visualIdentityId: string | null;
	mappingId: string | null;
	mappingState: 'provisional' | 'confirmed' | null;
	trackletCount: number;
	preview: string | null;
}

export interface IdentityTrackletRecord {
	trackletId: string;
	source: string;
	trackId: number;
	firstCapturedAt: string;
	lastCapturedAt: string;
	observationCount: number;
	evidenceStatus: EvidenceStatus;
	evidenceCount: number;
	preview: string;
}

export interface VisualIdentityRecord {
	visualIdentityId: string;
	status: 'pending' | 'active';
	mappingId: string | null;
	officialId: string | null;
	mappingState: 'provisional' | 'confirmed' | null;
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
	mapping_id: string;
	visual_identity_id: string;
	official_id: string;
	state: MappingState;
	provisional_tracklet_id: string;
	confirmation_tracklet_id: string | null;
	version: number;
}

const EVIDENCE_FRAMES = 2;

function now(): string {
	return new Date().toISOString();
}

function id(prefix: string): string {
	return `${prefix}_${randomUUID().replaceAll('-', '')}`;
}

function jpegDataUrl(value: Uint8Array | null): string | null {
	return value ? `data:image/jpeg;base64,${Buffer.from(value).toString('base64')}` : null;
}

function transaction<T>(
	database: DatabaseSync,
	options: MutationOptions,
	action: (occurredAt: string) => T
): T {
	database.exec('BEGIN IMMEDIATE');
	try {
		const control = database
			.prepare('SELECT operator_revision FROM control WHERE singleton = 1')
			.get() as { operator_revision: number } | undefined;
		if (!control) throw new Error('Identity catalog control record is missing');
		if (Number(control.operator_revision) !== options.expectedRevision) {
			throw new Error('The identity catalog changed. Refresh and try again.');
		}

		const revision = Number(control.operator_revision) + 1;
		const occurredAt = now();
		const result = action(occurredAt);
		database
			.prepare(
				`UPDATE control
				 SET operator_revision = ?, updated_at = ?
				 WHERE singleton = 1`
			)
			.run(revision, occurredAt);
		database.exec('COMMIT');
		return result;
	} catch (error) {
		database.exec('ROLLBACK');
		throw error;
	}
}

export function readCatalogControl(database: DatabaseSync): CatalogControl {
	const row = database
		.prepare(
			`SELECT operator_revision, active_gallery_version
			 FROM control WHERE singleton = 1`
		)
		.get() as
		| {
				operator_revision: number;
				active_gallery_version: number | null;
		  }
		| undefined;
	if (!row) throw new Error('Identity catalog control record is missing');
	return {
		operatorRevision: Number(row.operator_revision),
		activeGalleryVersion:
			row.active_gallery_version === null ? null : Number(row.active_gallery_version)
	};
}

export function readIdentityCatalog(database: DatabaseSync): IdentityCatalogSnapshot {
	const officials = database
		.prepare(
			`SELECT oi.official_id, oi.display_name, oi.status, oi.notes,
			        m.mapping_id, m.visual_identity_id, m.state AS mapping_state,
			        COUNT(DISTINCT vit.tracklet_id) AS tracklet_count,
			        (
			          SELECT t.preview_jpeg
			          FROM visual_identity_tracklets pv
			          JOIN tracklets t ON t.tracklet_id = pv.tracklet_id
			          WHERE pv.visual_identity_id = m.visual_identity_id
			          ORDER BY t.observation_count DESC, t.tracklet_id
			          LIMIT 1
			        ) AS preview_jpeg
			 FROM official_identities oi
			 LEFT JOIN mappings m
			   ON m.official_id = oi.official_id
			  AND m.state IN ('provisional', 'confirmed')
			 LEFT JOIN visual_identity_tracklets vit
			   ON vit.visual_identity_id = m.visual_identity_id
			 GROUP BY oi.official_id, oi.display_name, oi.status, oi.notes,
			          m.mapping_id, m.visual_identity_id, m.state
			 ORDER BY oi.status, oi.official_id`
		)
		.all() as Array<{
		official_id: string;
		display_name: string | null;
		status: OfficialIdentityStatus;
		notes: string;
		mapping_id: string | null;
		visual_identity_id: string | null;
		mapping_state: 'provisional' | 'confirmed' | null;
		tracklet_count: number;
		preview_jpeg: Uint8Array | null;
	}>;

	const visuals = database
		.prepare(
			`SELECT vi.visual_identity_id, vi.status, m.mapping_id,
			        m.official_id, m.state AS mapping_state,
			        m.provisional_tracklet_id, m.confirmation_tracklet_id
			 FROM visual_identities vi
			 LEFT JOIN mappings m
			   ON m.visual_identity_id = vi.visual_identity_id
			  AND m.state IN ('provisional', 'confirmed')
			 WHERE vi.status <> 'merged'
			 ORDER BY
			   CASE WHEN m.mapping_id IS NULL THEN 0 ELSE 1 END,
			   vi.updated_at DESC,
			   vi.visual_identity_id`
		)
		.all() as Array<{
		visual_identity_id: string;
		status: 'pending' | 'active';
		mapping_id: string | null;
		official_id: string | null;
		mapping_state: 'provisional' | 'confirmed' | null;
		provisional_tracklet_id: string | null;
		confirmation_tracklet_id: string | null;
	}>;

	const trackletStatement = database.prepare(
		`SELECT t.tracklet_id, t.source, t.track_id,
		        t.first_captured_at, t.last_captured_at,
		        t.observation_count, t.evidence_status,
		        t.preview_jpeg, COUNT(ef.evidence_id) AS evidence_count
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
			mappingId: row.mapping_id,
			mappingState: row.mapping_state,
			trackletCount: Number(row.tracklet_count),
			preview: jpegDataUrl(row.preview_jpeg)
		})),
		visualIdentities: visuals.map((visual) => ({
			visualIdentityId: visual.visual_identity_id,
			status: visual.status,
			mappingId: visual.mapping_id,
			officialId: visual.official_id,
			mappingState: visual.mapping_state,
			provisionalTrackletId: visual.provisional_tracklet_id,
			confirmationTrackletId: visual.confirmation_tracklet_id,
			tracklets: (
				trackletStatement.all(visual.visual_identity_id) as Array<{
					tracklet_id: string;
					source: string;
					track_id: number;
					first_captured_at: string;
					last_captured_at: string;
					observation_count: number;
					evidence_status: EvidenceStatus;
					preview_jpeg: Uint8Array;
					evidence_count: number;
				}>
			).map((tracklet) => ({
				trackletId: tracklet.tracklet_id,
				source: tracklet.source,
				trackId: Number(tracklet.track_id),
				firstCapturedAt: tracklet.first_captured_at,
				lastCapturedAt: tracklet.last_captured_at,
				observationCount: Number(tracklet.observation_count),
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
): {
	official_id: string;
	display_name: string | null;
	status: OfficialIdentityStatus;
	notes: string;
} {
	const row = database
		.prepare(
			`SELECT official_id, display_name, status, notes
			 FROM official_identities WHERE official_id = ?`
		)
		.get(officialId) as
		| {
				official_id: string;
				display_name: string | null;
				status: OfficialIdentityStatus;
				notes: string;
		  }
		| undefined;
	if (!row) throw new Error('Official identity does not exist');
	return row;
}

function activeMappingForVisual(
	database: DatabaseSync,
	visualIdentityId: string
): MappingRow | null {
	return (
		(database
			.prepare(
				`SELECT mapping_id, visual_identity_id, official_id, state,
				        provisional_tracklet_id, confirmation_tracklet_id, version
				 FROM mappings
				 WHERE visual_identity_id = ?
				   AND state IN ('provisional', 'confirmed')`
			)
			.get(visualIdentityId) as MappingRow | undefined) ?? null
	);
}

function activeMappingForOfficial(database: DatabaseSync, officialId: string): MappingRow | null {
	return (
		(database
			.prepare(
				`SELECT mapping_id, visual_identity_id, official_id, state,
				        provisional_tracklet_id, confirmation_tracklet_id, version
				 FROM mappings
				 WHERE official_id = ?
				   AND state IN ('provisional', 'confirmed')`
			)
			.get(officialId) as MappingRow | undefined) ?? null
	);
}

function mappingById(database: DatabaseSync, mappingId: string): MappingRow {
	const mapping = database
		.prepare(
			`SELECT mapping_id, visual_identity_id, official_id, state,
			        provisional_tracklet_id, confirmation_tracklet_id, version
			 FROM mappings WHERE mapping_id = ?`
		)
		.get(mappingId) as MappingRow | undefined;
	if (!mapping) throw new Error('Identity mapping does not exist');
	return mapping;
}

function eligibleTracklet(
	database: DatabaseSync,
	visualIdentityId: string,
	trackletId: string
): void {
	const row = database
		.prepare(
			`SELECT t.tracklet_id, t.evidence_status, vit.visual_identity_id
			 FROM tracklets t
			 JOIN visual_identity_tracklets vit ON vit.tracklet_id = t.tracklet_id
			 WHERE t.tracklet_id = ?`
		)
		.get(trackletId) as
		| {
				tracklet_id: string;
				evidence_status: EvidenceStatus;
				visual_identity_id: string;
		  }
		| undefined;
	if (!row || row.visual_identity_id !== visualIdentityId) {
		throw new Error('The selected evidence does not belong to this visual identity');
	}
	if (row.evidence_status !== 'eligible') {
		throw new Error('Only eligible, switch-free evidence can map an identity');
	}
	const evidence = database
		.prepare(
			`SELECT COUNT(*) AS count
			 FROM evidence_frames
			 WHERE tracklet_id = ?`
		)
		.get(trackletId) as { count: number };
	if (Number(evidence.count) < EVIDENCE_FRAMES) {
		throw new Error(`This tracklet needs ${EVIDENCE_FRAMES} eligible evidence frames`);
	}
}

function nextMappingVersion(database: DatabaseSync, visualIdentityId: string): number {
	const row = database
		.prepare(
			`SELECT COALESCE(MAX(version), 0) AS version
			 FROM mappings WHERE visual_identity_id = ?`
		)
		.get(visualIdentityId) as { version: number };
	return Number(row.version) + 1;
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
	const displayName = record.displayName?.trim() || null;
	const notes = record.notes?.trim() || '';
	if (!officialId) throw new Error('Official ID is required');
	return transaction(database, options, (occurredAt) => {
		database
			.prepare(
				`INSERT INTO official_identities (
					official_id, display_name, status, notes, created_at, updated_at
				) VALUES (?, ?, 'active', ?, ?, ?)`
			)
			.run(officialId, displayName, notes, occurredAt, occurredAt);
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
	transaction(database, options, (occurredAt) => {
		officialIdentity(database, record.officialId);
		if (record.status === 'archived' && activeMappingForOfficial(database, record.officialId)) {
			throw new Error('Deactivate the active mapping before archiving this identity');
		}
		const displayName = record.displayName?.trim() || null;
		const notes = record.notes?.trim() || '';
		database
			.prepare(
				`UPDATE official_identities
				 SET display_name = ?, status = ?, notes = ?, updated_at = ?
				 WHERE official_id = ?`
			)
			.run(displayName, record.status, notes, occurredAt, record.officialId);
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
): string {
	return transaction(database, options, (occurredAt) => {
		const official = officialIdentity(database, input.officialId);
		if (official.status !== 'active') throw new Error('Archived identities cannot be mapped');
		if (activeMappingForVisual(database, input.visualIdentityId)) {
			throw new Error('This visual identity already has an active mapping');
		}
		if (activeMappingForOfficial(database, input.officialId)) {
			throw new Error('This official identity already has an active mapping');
		}
		eligibleTracklet(database, input.visualIdentityId, input.trackletId);
		const mappingId = id('map');
		const version = nextMappingVersion(database, input.visualIdentityId);
		database
			.prepare(
				`INSERT INTO mappings (
					mapping_id, visual_identity_id, official_id, state,
					provisional_tracklet_id, confirmation_tracklet_id,
					version, created_at, updated_at
				) VALUES (?, ?, ?, 'provisional', ?, NULL, ?, ?, ?)`
			)
			.run(
				mappingId,
				input.visualIdentityId,
				input.officialId,
				input.trackletId,
				version,
				occurredAt,
				occurredAt
			);
		database
			.prepare(
				`UPDATE visual_identities
				 SET status = 'active', updated_at = ?
				 WHERE visual_identity_id = ? AND status IN ('pending', 'active')`
			)
			.run(occurredAt, input.visualIdentityId);
		return mappingId;
	});
}

export function confirmMapping(
	database: DatabaseSync,
	input: { mappingId: string; confirmationTrackletId: string },
	options: MutationOptions
): void {
	transaction(database, options, (occurredAt) => {
		const before = mappingById(database, input.mappingId);
		if (before.state !== 'provisional') {
			throw new Error('Only a provisional mapping can be confirmed');
		}
		if (before.provisional_tracklet_id === input.confirmationTrackletId) {
			throw new Error('Confirmation requires a distinct tracklet');
		}
		eligibleTracklet(database, before.visual_identity_id, before.provisional_tracklet_id);
		eligibleTracklet(database, before.visual_identity_id, input.confirmationTrackletId);
		database
			.prepare(
				`UPDATE mappings
				 SET state = 'confirmed', confirmation_tracklet_id = ?, updated_at = ?
				 WHERE mapping_id = ?`
			)
			.run(input.confirmationTrackletId, occurredAt, input.mappingId);
	});
}

export function correctMapping(
	database: DatabaseSync,
	input: {
		mappingId: string;
		officialId: string;
		provisionalTrackletId: string;
	},
	options: MutationOptions
): string {
	return transaction(database, options, (occurredAt) => {
		const before = mappingById(database, input.mappingId);
		if (!['provisional', 'confirmed'].includes(before.state)) {
			throw new Error('Only an active mapping can be corrected');
		}
		if (before.official_id === input.officialId) {
			throw new Error('Choose a different official identity for a correction');
		}
		const official = officialIdentity(database, input.officialId);
		if (official.status !== 'active') throw new Error('Archived identities cannot be mapped');
		const occupied = activeMappingForOfficial(database, input.officialId);
		if (occupied && occupied.mapping_id !== input.mappingId) {
			throw new Error('This official identity already has an active mapping');
		}
		eligibleTracklet(database, before.visual_identity_id, input.provisionalTrackletId);
		database
			.prepare(`UPDATE mappings SET state = 'inactive', updated_at = ? WHERE mapping_id = ?`)
			.run(occurredAt, before.mapping_id);
		const mappingId = id('map');
		const version = nextMappingVersion(database, before.visual_identity_id);
		database
			.prepare(
				`INSERT INTO mappings (
					mapping_id, visual_identity_id, official_id, state,
					provisional_tracklet_id, confirmation_tracklet_id,
					version, created_at, updated_at
				) VALUES (?, ?, ?, 'provisional', ?, NULL, ?, ?, ?)`
			)
			.run(
				mappingId,
				before.visual_identity_id,
				input.officialId,
				input.provisionalTrackletId,
				version,
				occurredAt,
				occurredAt
			);
		return mappingId;
	});
}

export function deactivateMapping(
	database: DatabaseSync,
	mappingId: string,
	options: MutationOptions
): void {
	transaction(database, options, (occurredAt) => {
		const before = mappingById(database, mappingId);
		if (!['provisional', 'confirmed'].includes(before.state)) {
			throw new Error('Identity mapping is already inactive');
		}
		database
			.prepare(`UPDATE mappings SET state = 'inactive', updated_at = ? WHERE mapping_id = ?`)
			.run(occurredAt, mappingId);
	});
}

export function rollbackMapping(
	database: DatabaseSync,
	mappingId: string,
	options: MutationOptions
): string | null {
	return transaction(database, options, (occurredAt) => {
		const before = mappingById(database, mappingId);
		if (!['provisional', 'confirmed'].includes(before.state)) {
			throw new Error('Only the active mapping can be rolled back');
		}
		database
			.prepare(`UPDATE mappings SET state = 'rejected', updated_at = ? WHERE mapping_id = ?`)
			.run(occurredAt, mappingId);

		const previous = database
			.prepare(
				`SELECT mapping_id, visual_identity_id, official_id, state,
				        provisional_tracklet_id, confirmation_tracklet_id, version
				 FROM mappings
				 WHERE visual_identity_id = ? AND version < ? AND state = 'inactive'
				 ORDER BY version DESC LIMIT 1`
			)
			.get(before.visual_identity_id, before.version) as MappingRow | undefined;
		let restoredState: 'provisional' | 'confirmed' | null = null;
		if (previous && !activeMappingForOfficial(database, previous.official_id)) {
			restoredState = previous.confirmation_tracklet_id ? 'confirmed' : 'provisional';
			database
				.prepare(`UPDATE mappings SET state = ?, updated_at = ? WHERE mapping_id = ?`)
				.run(restoredState, occurredAt, previous.mapping_id);
		}
		return previous && restoredState ? previous.mapping_id : null;
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
	transaction(database, options, (occurredAt) => {
		const rows = database
			.prepare(
				`SELECT visual_identity_id, status
				 FROM visual_identities
				 WHERE visual_identity_id IN (?, ?)`
			)
			.all(input.sourceVisualIdentityId, input.targetVisualIdentityId) as Array<{
			visual_identity_id: string;
			status: string;
		}>;
		if (rows.length !== 2 || rows.some((row) => row.status === 'merged')) {
			throw new Error('Both visual identities must be active');
		}
		if (activeMappingForVisual(database, input.sourceVisualIdentityId)) {
			throw new Error('Deactivate or correct the source mapping before merging evidence');
		}
		const hasEvidence = database
			.prepare(
				`SELECT 1 FROM visual_identity_tracklets
				 WHERE visual_identity_id = ? LIMIT 1`
			)
			.get(input.sourceVisualIdentityId);
		if (!hasEvidence) throw new Error('The source identity has no evidence to merge');
		database
			.prepare(
				`UPDATE visual_identity_tracklets
				 SET visual_identity_id = ?, assignment_kind = 'human_merge',
				     assigned_at = ?
				 WHERE visual_identity_id = ?`
			)
			.run(input.targetVisualIdentityId, occurredAt, input.sourceVisualIdentityId);
		database
			.prepare(
				`UPDATE visual_identities
				 SET status = 'merged', merged_into_visual_identity_id = ?, updated_at = ?
				 WHERE visual_identity_id = ?`
			)
			.run(input.targetVisualIdentityId, occurredAt, input.sourceVisualIdentityId);
		database
			.prepare(
				`UPDATE visual_identities
				 SET status = CASE WHEN status = 'pending' THEN 'active' ELSE status END,
				     updated_at = ?
				 WHERE visual_identity_id = ?`
			)
			.run(occurredAt, input.targetVisualIdentityId);
	});
}

export function splitVisualIdentity(
	database: DatabaseSync,
	input: { sourceVisualIdentityId: string; trackletIds: string[] },
	options: MutationOptions
): string {
	const selected = [...new Set(input.trackletIds)];
	if (!selected.length) throw new Error('Select at least one tracklet to split');
	return transaction(database, options, (occurredAt) => {
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
			.prepare(
				`INSERT INTO visual_identities (
					visual_identity_id, status, created_at, updated_at
				) VALUES (?, 'pending', ?, ?)`
			)
			.run(newVisualIdentityId, occurredAt, occurredAt);
		const placeholders = selected.map(() => '?').join(',');
		database
			.prepare(
				`UPDATE visual_identity_tracklets
				 SET visual_identity_id = ?, assignment_kind = 'human_split',
				     assigned_at = ?
				 WHERE visual_identity_id = ?
				   AND tracklet_id IN (${placeholders})`
			)
			.run(newVisualIdentityId, occurredAt, input.sourceVisualIdentityId, ...selected);
		const mapping = activeMappingForVisual(database, input.sourceVisualIdentityId);
		const mappingInvalidated =
			mapping !== null &&
			(selected.includes(mapping.provisional_tracklet_id) ||
				(mapping.confirmation_tracklet_id !== null &&
					selected.includes(mapping.confirmation_tracklet_id)));
		if (mappingInvalidated) {
			database
				.prepare(`UPDATE mappings SET state = 'inactive', updated_at = ? WHERE mapping_id = ?`)
				.run(occurredAt, mapping.mapping_id);
		}
		database
			.prepare(`UPDATE visual_identities SET updated_at = ? WHERE visual_identity_id = ?`)
			.run(occurredAt, input.sourceVisualIdentityId);
		return newVisualIdentityId;
	});
}
