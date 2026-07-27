import { createHash, randomUUID } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';

export type OfficialIdentityStatus = 'active' | 'archived';
export type MappingState = 'provisional' | 'confirmed' | 'inactive' | 'rejected';
export type EvidenceStatus = 'eligible' | 'insufficient' | 'impure' | 'switch_risk';

export interface CatalogControl {
	operatorRevision: number;
	runtimeRevision: number;
	activeGalleryVersion: number | null;
	lastIdentityError: string | null;
}

export interface CatalogPolicy {
	galleryFrames: number;
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
	status: 'pending' | 'active' | 'split';
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
	reason?: string;
	actor?: string;
}

interface AuditEvent {
	eventType: string;
	entityType: string;
	entityId: string;
	before: unknown;
	after: unknown;
	reason: string;
}

interface MutationResult<T> {
	value: T;
	audit: AuditEvent;
	afterAudit?: (eventId: string) => void;
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

interface TrackletEligibility {
	trackletId: string;
	visualIdentityId: string;
	evidenceIds: string[];
}

const DEFAULT_ACTOR = 'private-network-web';

function now(): string {
	return new Date().toISOString();
}

function id(prefix: string): string {
	return `${prefix}_${randomUUID().replaceAll('-', '')}`;
}

function jpegDataUrl(value: Uint8Array | null): string | null {
	return value ? `data:image/jpeg;base64,${Buffer.from(value).toString('base64')}` : null;
}

function canonicalJson(value: unknown): string {
	if (value === null || typeof value !== 'object') return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
	const object = value as Record<string, unknown>;
	return `{${Object.keys(object)
		.sort()
		.map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`)
		.join(',')}}`;
}

function transaction<T>(
	database: DatabaseSync,
	options: MutationOptions,
	action: (revision: number, occurredAt: string) => MutationResult<T>
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
		const result = action(revision, occurredAt);
		database
			.prepare(
				`UPDATE control
				 SET operator_revision = ?, updated_at = ?
				 WHERE singleton = 1`
			)
			.run(revision, occurredAt);
		const eventId = insertAuditEvent(
			database,
			{
				...result.audit,
				reason: result.audit.reason || options.reason?.trim() || 'operator action'
			},
			options.actor?.trim() || DEFAULT_ACTOR,
			revision,
			occurredAt
		);
		result.afterAudit?.(eventId);
		database.exec('COMMIT');
		return result.value;
	} catch (error) {
		database.exec('ROLLBACK');
		throw error;
	}
}

function insertAuditEvent(
	database: DatabaseSync,
	event: AuditEvent,
	actor: string,
	operatorRevision: number,
	occurredAt: string
): string {
	const previous = database
		.prepare('SELECT event_id FROM audit_events ORDER BY sequence DESC LIMIT 1')
		.get() as { event_id: string } | undefined;
	const eventId = id('evt');
	const beforeJson = event.before === null ? null : canonicalJson(event.before);
	const afterJson = event.after === null ? null : canonicalJson(event.after);
	const content = {
		event_id: eventId,
		event_type: event.eventType,
		actor,
		entity_type: event.entityType,
		entity_id: event.entityId,
		operator_revision: operatorRevision,
		previous_event_id: previous?.event_id ?? null,
		before_json: beforeJson,
		after_json: afterJson,
		reason: event.reason,
		occurred_at: occurredAt
	};
	const contentSha256 = createHash('sha256').update(canonicalJson(content)).digest('hex');
	database
		.prepare(
			`INSERT INTO audit_events (
				event_id, event_type, actor, entity_type, entity_id,
				operator_revision, previous_event_id, before_json, after_json,
				reason, occurred_at, content_sha256
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
		)
		.run(
			eventId,
			event.eventType,
			actor,
			event.entityType,
			event.entityId,
			operatorRevision,
			previous?.event_id ?? null,
			beforeJson,
			afterJson,
			event.reason,
			occurredAt,
			contentSha256
		);
	return eventId;
}

export function readCatalogControl(database: DatabaseSync): CatalogControl {
	const row = database
		.prepare(
			`SELECT operator_revision, runtime_revision,
			        active_gallery_version, last_identity_error
			 FROM control WHERE singleton = 1`
		)
		.get() as
		| {
				operator_revision: number;
				runtime_revision: number;
				active_gallery_version: number | null;
				last_identity_error: string | null;
		  }
		| undefined;
	if (!row) throw new Error('Identity catalog control record is missing');
	return {
		operatorRevision: Number(row.operator_revision),
		runtimeRevision: Number(row.runtime_revision),
		activeGalleryVersion:
			row.active_gallery_version === null ? null : Number(row.active_gallery_version),
		lastIdentityError: row.last_identity_error
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
			 WHERE vi.status NOT IN ('merged', 'archived')
			 ORDER BY
			   CASE WHEN m.mapping_id IS NULL THEN 0 ELSE 1 END,
			   vi.updated_at DESC,
			   vi.visual_identity_id`
		)
		.all() as Array<{
		visual_identity_id: string;
		status: 'pending' | 'active' | 'split';
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

function requiredFrames(policy: CatalogPolicy, confirmation: boolean): number {
	if (!Number.isInteger(policy.galleryFrames) || policy.galleryFrames < 2) {
		throw new Error('Identity gallery policy is invalid');
	}
	const half = Math.floor(policy.galleryFrames / 2);
	return confirmation ? half : half + (policy.galleryFrames % 2);
}

function eligibleTracklet(
	database: DatabaseSync,
	visualIdentityId: string,
	trackletId: string,
	frameCount: number
): TrackletEligibility {
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
			`SELECT evidence_id
			 FROM evidence_frames
			 WHERE tracklet_id = ?
			 ORDER BY frame_index
			 LIMIT ?`
		)
		.all(trackletId, frameCount) as Array<{ evidence_id: string }>;
	if (evidence.length !== frameCount) {
		throw new Error(`This tracklet needs ${frameCount} eligible evidence frames`);
	}
	return {
		trackletId,
		visualIdentityId,
		evidenceIds: evidence.map((item) => item.evidence_id)
	};
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
	return transaction(database, options, (_revision, occurredAt) => {
		database
			.prepare(
				`INSERT INTO official_identities (
					official_id, display_name, status, notes, created_at, updated_at
				) VALUES (?, ?, 'active', ?, ?, ?)`
			)
			.run(officialId, displayName, notes, occurredAt, occurredAt);
		const after = { official_id: officialId, display_name: displayName, status: 'active', notes };
		return {
			value: officialId,
			audit: {
				eventType: 'official_identity_created',
				entityType: 'official_identity',
				entityId: officialId,
				before: null,
				after,
				reason: options.reason ?? 'official identity created by operator'
			}
		};
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
	transaction(database, options, (_revision, occurredAt) => {
		const before = officialIdentity(database, record.officialId);
		if (record.status === 'archived' && activeMappingForOfficial(database, record.officialId)) {
			throw new Error('Deactivate the active mapping before archiving this identity');
		}
		const after = {
			official_id: before.official_id,
			display_name: record.displayName?.trim() || null,
			status: record.status,
			notes: record.notes?.trim() || ''
		};
		database
			.prepare(
				`UPDATE official_identities
				 SET display_name = ?, status = ?, notes = ?, updated_at = ?
				 WHERE official_id = ?`
			)
			.run(after.display_name, after.status, after.notes, occurredAt, record.officialId);
		return {
			value: undefined,
			audit: {
				eventType: 'official_identity_updated',
				entityType: 'official_identity',
				entityId: record.officialId,
				before,
				after,
				reason: options.reason ?? 'official identity updated by operator'
			}
		};
	});
}

export function createProvisionalMapping(
	database: DatabaseSync,
	input: {
		visualIdentityId: string;
		officialId: string;
		trackletId: string;
	},
	policy: CatalogPolicy,
	options: MutationOptions
): string {
	return transaction(database, options, (_revision, occurredAt) => {
		const official = officialIdentity(database, input.officialId);
		if (official.status !== 'active') throw new Error('Archived identities cannot be mapped');
		if (activeMappingForVisual(database, input.visualIdentityId)) {
			throw new Error('This visual identity already has an active mapping');
		}
		if (activeMappingForOfficial(database, input.officialId)) {
			throw new Error('This official identity already has an active mapping');
		}
		const evidence = eligibleTracklet(
			database,
			input.visualIdentityId,
			input.trackletId,
			requiredFrames(policy, false)
		);
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
				 WHERE visual_identity_id = ? AND status IN ('pending', 'active', 'split')`
			)
			.run(occurredAt, input.visualIdentityId);
		return {
			value: mappingId,
			audit: {
				eventType: 'mapping_provisioned',
				entityType: 'mapping',
				entityId: mappingId,
				before: null,
				after: {
					mapping_id: mappingId,
					visual_identity_id: input.visualIdentityId,
					official_id: input.officialId,
					state: 'provisional',
					provisional_tracklet_id: input.trackletId,
					evidence_ids: evidence.evidenceIds,
					version
				},
				reason: options.reason ?? 'first human mapping evidence'
			}
		};
	});
}

export function confirmMapping(
	database: DatabaseSync,
	input: { mappingId: string; confirmationTrackletId: string },
	policy: CatalogPolicy,
	options: MutationOptions
): void {
	transaction(database, options, (_revision, occurredAt) => {
		const before = mappingById(database, input.mappingId);
		if (before.state !== 'provisional') {
			throw new Error('Only a provisional mapping can be confirmed');
		}
		if (before.provisional_tracklet_id === input.confirmationTrackletId) {
			throw new Error('Confirmation requires a distinct tracklet');
		}
		const first = eligibleTracklet(
			database,
			before.visual_identity_id,
			before.provisional_tracklet_id,
			requiredFrames(policy, false)
		);
		const confirmation = eligibleTracklet(
			database,
			before.visual_identity_id,
			input.confirmationTrackletId,
			requiredFrames(policy, true)
		);
		database
			.prepare(
				`UPDATE mappings
				 SET state = 'confirmed', confirmation_tracklet_id = ?, updated_at = ?
				 WHERE mapping_id = ?`
			)
			.run(input.confirmationTrackletId, occurredAt, input.mappingId);
		return {
			value: undefined,
			audit: {
				eventType: 'mapping_confirmed',
				entityType: 'mapping',
				entityId: input.mappingId,
				before,
				after: {
					...before,
					state: 'confirmed',
					confirmation_tracklet_id: input.confirmationTrackletId,
					evidence_ids: [...first.evidenceIds, ...confirmation.evidenceIds]
				},
				reason: options.reason ?? 'independent human confirmation'
			}
		};
	});
}

export function correctMapping(
	database: DatabaseSync,
	input: {
		mappingId: string;
		officialId: string;
		provisionalTrackletId: string;
	},
	policy: CatalogPolicy,
	options: MutationOptions
): string {
	return transaction(database, options, (_revision, occurredAt) => {
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
		const evidence = eligibleTracklet(
			database,
			before.visual_identity_id,
			input.provisionalTrackletId,
			requiredFrames(policy, false)
		);
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
		return {
			value: mappingId,
			audit: {
				eventType: 'mapping_corrected',
				entityType: 'mapping',
				entityId: mappingId,
				before,
				after: {
					mapping_id: mappingId,
					visual_identity_id: before.visual_identity_id,
					official_id: input.officialId,
					state: 'provisional',
					provisional_tracklet_id: input.provisionalTrackletId,
					evidence_ids: evidence.evidenceIds,
					version
				},
				reason: options.reason ?? 'operator corrected official mapping'
			}
		};
	});
}

export function deactivateMapping(
	database: DatabaseSync,
	mappingId: string,
	options: MutationOptions
): void {
	transaction(database, options, (_revision, occurredAt) => {
		const before = mappingById(database, mappingId);
		if (!['provisional', 'confirmed'].includes(before.state)) {
			throw new Error('Identity mapping is already inactive');
		}
		database
			.prepare(`UPDATE mappings SET state = 'inactive', updated_at = ? WHERE mapping_id = ?`)
			.run(occurredAt, mappingId);
		return {
			value: undefined,
			audit: {
				eventType: 'mapping_deactivated',
				entityType: 'mapping',
				entityId: mappingId,
				before,
				after: { ...before, state: 'inactive' },
				reason: options.reason ?? 'operator deactivated mapping'
			}
		};
	});
}

export function rollbackMapping(
	database: DatabaseSync,
	mappingId: string,
	options: MutationOptions
): string | null {
	return transaction(database, options, (_revision, occurredAt) => {
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
		return {
			value: previous && restoredState ? previous.mapping_id : null,
			audit: {
				eventType: 'mapping_rolled_back',
				entityType: 'mapping',
				entityId: mappingId,
				before,
				after: {
					rolled_back_mapping_id: mappingId,
					restored_mapping_id: previous && restoredState ? previous.mapping_id : null,
					restored_state: restoredState
				},
				reason: options.reason ?? 'operator rollback'
			}
		};
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
	transaction(database, options, (_revision, occurredAt) => {
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
		if (rows.length !== 2 || rows.some((row) => ['merged', 'archived'].includes(row.status))) {
			throw new Error('Both visual identities must be active');
		}
		if (activeMappingForVisual(database, input.sourceVisualIdentityId)) {
			throw new Error('Deactivate or correct the source mapping before merging evidence');
		}
		const moved = database
			.prepare(
				`SELECT tracklet_id FROM visual_identity_tracklets
				 WHERE visual_identity_id = ? ORDER BY tracklet_id`
			)
			.all(input.sourceVisualIdentityId) as Array<{ tracklet_id: string }>;
		if (!moved.length) throw new Error('The source identity has no evidence to merge');
		database
			.prepare(
				`UPDATE visual_identity_tracklets
				 SET visual_identity_id = ?, assignment_kind = 'human_merge',
				     audit_event_id = NULL, assigned_at = ?
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
		return {
			value: undefined,
			audit: {
				eventType: 'visual_identities_merged',
				entityType: 'visual_identity',
				entityId: input.sourceVisualIdentityId,
				before: {
					source_visual_identity_id: input.sourceVisualIdentityId,
					target_visual_identity_id: input.targetVisualIdentityId,
					tracklet_ids: moved.map((row) => row.tracklet_id)
				},
				after: {
					source_status: 'merged',
					merged_into_visual_identity_id: input.targetVisualIdentityId
				},
				reason: options.reason ?? 'operator merged visual evidence'
			},
			afterAudit: (eventId) => {
				database
					.prepare(
						`UPDATE visual_identity_tracklets
						 SET audit_event_id = ?
						 WHERE visual_identity_id = ?
						   AND tracklet_id IN (${moved.map(() => '?').join(',')})`
					)
					.run(eventId, input.targetVisualIdentityId, ...moved.map((row) => row.tracklet_id));
			}
		};
	});
}

export function splitVisualIdentity(
	database: DatabaseSync,
	input: { sourceVisualIdentityId: string; trackletIds: string[] },
	options: MutationOptions
): string {
	const selected = [...new Set(input.trackletIds)];
	if (!selected.length) throw new Error('Select at least one tracklet to split');
	return transaction(database, options, (_revision, occurredAt) => {
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
				     audit_event_id = NULL, assigned_at = ?
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
		return {
			value: newVisualIdentityId,
			audit: {
				eventType: 'visual_identity_split',
				entityType: 'visual_identity',
				entityId: newVisualIdentityId,
				before: {
					source_visual_identity_id: input.sourceVisualIdentityId,
					active_mapping_id: mapping?.mapping_id ?? null
				},
				after: {
					new_visual_identity_id: newVisualIdentityId,
					tracklet_ids: selected,
					invalidated_mapping_id: mappingInvalidated ? mapping?.mapping_id : null
				},
				reason: options.reason ?? 'operator split visual evidence'
			},
			afterAudit: (eventId) => {
				database
					.prepare(
						`UPDATE visual_identity_tracklets
						 SET audit_event_id = ?
						 WHERE visual_identity_id = ?`
					)
					.run(eventId, newVisualIdentityId);
			}
		};
	});
}
