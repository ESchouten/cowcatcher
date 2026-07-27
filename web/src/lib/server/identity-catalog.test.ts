import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { afterEach, describe, expect, it } from 'vitest';

import {
	confirmMapping,
	correctMapping,
	createOfficialIdentity,
	createProvisionalMapping,
	mergeVisualIdentities,
	readIdentityCatalog,
	rollbackMapping,
	splitVisualIdentity
} from './identity-catalog';

const openDatabases: DatabaseSync[] = [];
const POLICY = { galleryFrames: 4 };
const TIMESTAMP = '2026-07-24T10:00:00.000Z';

function databaseFixture(): DatabaseSync {
	const database = new DatabaseSync(':memory:');
	database.exec(
		readFileSync(resolve(process.cwd(), '../detector/src/aidetector/reid/schema.sql'), 'utf8')
	);
	openDatabases.push(database);
	return database;
}

function visualIdentity(
	database: DatabaseSync,
	visualIdentityId: string,
	tracklets: Array<{ id: string; evidence?: number }>
): void {
	database
		.prepare(
			`INSERT INTO visual_identities (
				visual_identity_id, status, created_at, updated_at
			) VALUES (?, 'pending', ?, ?)`
		)
		.run(visualIdentityId, TIMESTAMP, TIMESTAMP);
	for (const [index, tracklet] of tracklets.entries()) {
		database
			.prepare(
				`INSERT INTO tracklets (
					tracklet_id, run_id, source, track_id,
					first_captured_at, last_captured_at, observation_count,
					evidence_status, preview_jpeg, created_at, updated_at
				) VALUES (?, ?, 'camera', ?, ?, ?, 4, 'eligible', ?, ?, ?)`
			)
			.run(
				tracklet.id,
				visualIdentityId,
				index + 1,
				TIMESTAMP,
				TIMESTAMP,
				Buffer.from([0xff, 0xd8, index, 0xff, 0xd9]),
				TIMESTAMP,
				TIMESTAMP
			);
		database
			.prepare(
				`INSERT INTO visual_identity_tracklets (
					tracklet_id, visual_identity_id, assignment_kind, assigned_at
				) VALUES (?, ?, 'initial', ?)`
			)
			.run(tracklet.id, visualIdentityId, TIMESTAMP);
		for (let frame = 0; frame < (tracklet.evidence ?? 2); frame += 1) {
			database
				.prepare(
					`INSERT INTO evidence_frames (
						evidence_id, tracklet_id, frame_index, captured_at,
						image_sha256, preview_jpeg, embedding,
						embedding_dimension, quality, created_at
					) VALUES (?, ?, ?, ?, ?, ?, ?, 2, 1.0, ?)`
				)
				.run(
					`evd_${tracklet.id.slice(4)}_${frame}`,
					tracklet.id,
					frame,
					TIMESTAMP,
					'0'.repeat(64),
					Buffer.from([0xff, 0xd8, frame, 0xff, 0xd9]),
					Buffer.alloc(8),
					TIMESTAMP
				);
		}
	}
}

afterEach(() => {
	while (openDatabases.length) openDatabases.pop()?.close();
});

describe('generic identity catalog operator transactions', () => {
	it('keeps the first mapping provisional and audits it with explicit evidence', () => {
		const database = databaseFixture();
		visualIdentity(database, 'vid_alpha', [{ id: 'trk_alpha_1' }]);
		createOfficialIdentity(
			database,
			{ officialId: 'NL-001', displayName: 'Ada' },
			{ expectedRevision: 0 }
		);

		const mappingId = createProvisionalMapping(
			database,
			{
				visualIdentityId: 'vid_alpha',
				officialId: 'NL-001',
				trackletId: 'trk_alpha_1'
			},
			POLICY,
			{ expectedRevision: 1 }
		);

		const catalog = readIdentityCatalog(database);
		expect(catalog.control.operatorRevision).toBe(2);
		expect(catalog.control.activeGalleryVersion).toBeNull();
		expect(catalog.officialIdentities[0]).toMatchObject({
			officialId: 'NL-001',
			displayName: 'Ada',
			mappingId,
			mappingState: 'provisional'
		});
		const audit = database
			.prepare(`SELECT event_type, after_json FROM audit_events WHERE entity_id = ?`)
			.get(mappingId) as { event_type: string; after_json: string };
		expect(audit.event_type).toBe('mapping_provisioned');
		expect(JSON.parse(audit.after_json).evidence_ids).toHaveLength(2);
	});

	it('requires a distinct eligible confirmation tracklet and four total frames', () => {
		const database = databaseFixture();
		visualIdentity(database, 'vid_beta', [{ id: 'trk_beta_1' }, { id: 'trk_beta_2', evidence: 1 }]);
		createOfficialIdentity(database, { officialId: 'NL-002' }, { expectedRevision: 0 });
		const mappingId = createProvisionalMapping(
			database,
			{ visualIdentityId: 'vid_beta', officialId: 'NL-002', trackletId: 'trk_beta_1' },
			POLICY,
			{ expectedRevision: 1 }
		);

		expect(() =>
			confirmMapping(database, { mappingId, confirmationTrackletId: 'trk_beta_1' }, POLICY, {
				expectedRevision: 2
			})
		).toThrow('distinct tracklet');
		expect(() =>
			confirmMapping(database, { mappingId, confirmationTrackletId: 'trk_beta_2' }, POLICY, {
				expectedRevision: 2
			})
		).toThrow('needs 2 eligible evidence frames');
		expect(readIdentityCatalog(database).control.operatorRevision).toBe(2);

		database
			.prepare(
				`INSERT INTO evidence_frames (
					evidence_id, tracklet_id, frame_index, captured_at,
					image_sha256, preview_jpeg, embedding,
					embedding_dimension, quality, created_at
				) VALUES ('evd_beta_2_1', 'trk_beta_2', 1, ?, ?, ?, ?, 2, 1.0, ?)`
			)
			.run(TIMESTAMP, '1'.repeat(64), Buffer.from([0xff]), Buffer.alloc(8), TIMESTAMP);
		confirmMapping(database, { mappingId, confirmationTrackletId: 'trk_beta_2' }, POLICY, {
			expectedRevision: 2
		});

		expect(readIdentityCatalog(database).officialIdentities[0].mappingState).toBe('confirmed');
		const audit = database
			.prepare(`SELECT after_json FROM audit_events WHERE event_type = 'mapping_confirmed'`)
			.get() as { after_json: string };
		expect(JSON.parse(audit.after_json).evidence_ids).toHaveLength(4);
	});

	it('corrects to a provisional mapping and rolls back to the prior mapping', () => {
		const database = databaseFixture();
		visualIdentity(database, 'vid_gamma', [{ id: 'trk_gamma_1' }]);
		createOfficialIdentity(database, { officialId: 'NL-003' }, { expectedRevision: 0 });
		createOfficialIdentity(database, { officialId: 'NL-004' }, { expectedRevision: 1 });
		const firstMapping = createProvisionalMapping(
			database,
			{ visualIdentityId: 'vid_gamma', officialId: 'NL-003', trackletId: 'trk_gamma_1' },
			POLICY,
			{ expectedRevision: 2 }
		);
		const correction = correctMapping(
			database,
			{
				mappingId: firstMapping,
				officialId: 'NL-004',
				provisionalTrackletId: 'trk_gamma_1'
			},
			POLICY,
			{ expectedRevision: 3 }
		);

		expect(readIdentityCatalog(database).officialIdentities).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ officialId: 'NL-004', mappingState: 'provisional' })
			])
		);
		expect(
			rollbackMapping(database, correction, {
				expectedRevision: 4,
				reason: 'test correction rollback'
			})
		).toBe(firstMapping);
		const restored = database
			.prepare(`SELECT state FROM mappings WHERE mapping_id = ?`)
			.get(firstMapping) as { state: string };
		expect(restored.state).toBe('provisional');
		expect(
			(
				database
					.prepare(`SELECT COUNT(*) AS count FROM audit_events WHERE operator_revision IS NOT NULL`)
					.get() as { count: number }
			).count
		).toBe(5);
	});

	it('fails a stale concurrent operator revision without partial writes', () => {
		const database = databaseFixture();
		createOfficialIdentity(database, { officialId: 'NL-005' }, { expectedRevision: 0 });

		expect(() =>
			createOfficialIdentity(database, { officialId: 'NL-006' }, { expectedRevision: 0 })
		).toThrow('changed');
		expect(
			(
				database.prepare(`SELECT COUNT(*) AS count FROM official_identities`).get() as {
					count: number;
				}
			).count
		).toBe(1);
		expect(readIdentityCatalog(database).control.operatorRevision).toBe(1);
	});

	it('merges and splits visual evidence with direct immutable audit links', () => {
		const database = databaseFixture();
		visualIdentity(database, 'vid_merge_source', [{ id: 'trk_merge_1' }, { id: 'trk_merge_2' }]);
		visualIdentity(database, 'vid_merge_target', [{ id: 'trk_merge_3' }]);

		mergeVisualIdentities(
			database,
			{
				sourceVisualIdentityId: 'vid_merge_source',
				targetVisualIdentityId: 'vid_merge_target'
			},
			{ expectedRevision: 0 }
		);
		expect(
			(
				database
					.prepare(
						`SELECT status, merged_into_visual_identity_id
						 FROM visual_identities WHERE visual_identity_id = 'vid_merge_source'`
					)
					.get() as { status: string; merged_into_visual_identity_id: string }
			).status
		).toBe('merged');
		expect(
			(
				database
					.prepare(
						`SELECT COUNT(*) AS count FROM visual_identity_tracklets
						 WHERE visual_identity_id = 'vid_merge_target'
						   AND audit_event_id IS NOT NULL`
					)
					.get() as { count: number }
			).count
		).toBe(2);

		const split = splitVisualIdentity(
			database,
			{ sourceVisualIdentityId: 'vid_merge_target', trackletIds: ['trk_merge_2'] },
			{ expectedRevision: 1 }
		);
		expect(split).toMatch(/^vid_/);
		expect(
			database
				.prepare(
					`SELECT assignment_kind, audit_event_id
						 FROM visual_identity_tracklets WHERE tracklet_id = 'trk_merge_2'`
				)
				.get() as { assignment_kind: string; audit_event_id: string | null }
		).toMatchObject({ assignment_kind: 'human_split' });
		expect(readIdentityCatalog(database).control.operatorRevision).toBe(2);
		expect(() =>
			database.exec(`UPDATE audit_events SET reason = 'tampered' WHERE sequence = 1`)
		).toThrow('audit events are immutable');
	});
});
