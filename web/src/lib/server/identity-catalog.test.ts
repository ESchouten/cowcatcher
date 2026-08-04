import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { afterEach, describe, expect, it } from 'vitest';

import {
	confirmMapping,
	correctMapping,
	createOfficialIdentity,
	createProvisionalMapping,
	deactivateMapping,
	mergeVisualIdentities,
	readIdentityCatalog,
	splitVisualIdentity
} from './identity-catalog';

const openDatabases: DatabaseSync[] = [];
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
		.prepare('INSERT INTO visual_identities (visual_identity_id) VALUES (?)')
		.run(visualIdentityId);
	for (const [index, tracklet] of tracklets.entries()) {
		database
			.prepare(
				`INSERT INTO tracklets (
					tracklet_id, source, last_captured_at,
					evidence_status, preview_jpeg
				) VALUES (?, 'camera', ?, 'eligible', ?)`
			)
			.run(tracklet.id, TIMESTAMP, Buffer.from([0xff, 0xd8, index, 0xff, 0xd9]));
		database
			.prepare(
				`INSERT INTO visual_identity_tracklets (
					tracklet_id, visual_identity_id
				) VALUES (?, ?)`
			)
			.run(tracklet.id, visualIdentityId);
		for (let frame = 0; frame < (tracklet.evidence ?? 2); frame += 1) {
			database
				.prepare(
					`INSERT INTO evidence_frames (
						tracklet_id, frame_index, embedding
					) VALUES (?, ?, ?)`
				)
				.run(tracklet.id, frame, Buffer.alloc(8));
		}
	}
}

afterEach(() => {
	while (openDatabases.length) openDatabases.pop()?.close();
});

describe('identity catalog operator transactions', () => {
	it('keeps the first mapping provisional', () => {
		const database = databaseFixture();
		visualIdentity(database, 'vid_alpha', [{ id: 'trk_alpha_1' }]);
		createOfficialIdentity(
			database,
			{ officialId: 'NL-001', displayName: 'Ada' },
			{ expectedRevision: 0 }
		);

		createProvisionalMapping(
			database,
			{
				visualIdentityId: 'vid_alpha',
				officialId: 'NL-001',
				trackletId: 'trk_alpha_1'
			},
			{ expectedRevision: 1 }
		);

		const catalog = readIdentityCatalog(database);
		expect(catalog.control.operatorRevision).toBe(2);
		expect(catalog.officialIdentities[0]).toMatchObject({
			officialId: 'NL-001',
			displayName: 'Ada',
			mappingState: 'provisional'
		});
	});

	it('requires a distinct eligible confirmation tracklet and four total frames', () => {
		const database = databaseFixture();
		visualIdentity(database, 'vid_beta', [{ id: 'trk_beta_1' }, { id: 'trk_beta_2', evidence: 1 }]);
		createOfficialIdentity(database, { officialId: 'NL-002' }, { expectedRevision: 0 });
		createProvisionalMapping(
			database,
			{ visualIdentityId: 'vid_beta', officialId: 'NL-002', trackletId: 'trk_beta_1' },
			{ expectedRevision: 1 }
		);

		expect(() =>
			confirmMapping(
				database,
				{ visualIdentityId: 'vid_beta', confirmationTrackletId: 'trk_beta_1' },
				{ expectedRevision: 2 }
			)
		).toThrow('distinct tracklet');
		expect(() =>
			confirmMapping(
				database,
				{ visualIdentityId: 'vid_beta', confirmationTrackletId: 'trk_beta_2' },
				{ expectedRevision: 2 }
			)
		).toThrow('needs 2 eligible evidence frames');
		expect(readIdentityCatalog(database).control.operatorRevision).toBe(2);

		database
			.prepare(
				`INSERT INTO evidence_frames (
					tracklet_id, frame_index, embedding
				) VALUES ('trk_beta_2', 1, ?)`
			)
			.run(Buffer.alloc(8));
		confirmMapping(
			database,
			{ visualIdentityId: 'vid_beta', confirmationTrackletId: 'trk_beta_2' },
			{ expectedRevision: 2 }
		);

		expect(readIdentityCatalog(database).officialIdentities[0].mappingState).toBe('confirmed');
	});

	it('corrects the current mapping and can deactivate it', () => {
		const database = databaseFixture();
		visualIdentity(database, 'vid_gamma', [{ id: 'trk_gamma_1' }]);
		createOfficialIdentity(database, { officialId: 'NL-003' }, { expectedRevision: 0 });
		createOfficialIdentity(database, { officialId: 'NL-004' }, { expectedRevision: 1 });
		createProvisionalMapping(
			database,
			{ visualIdentityId: 'vid_gamma', officialId: 'NL-003', trackletId: 'trk_gamma_1' },
			{ expectedRevision: 2 }
		);
		correctMapping(
			database,
			{
				visualIdentityId: 'vid_gamma',
				officialId: 'NL-004',
				provisionalTrackletId: 'trk_gamma_1'
			},
			{ expectedRevision: 3 }
		);

		expect(readIdentityCatalog(database).officialIdentities).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ officialId: 'NL-004', mappingState: 'provisional' })
			])
		);
		expect(
			(database.prepare('SELECT COUNT(*) AS count FROM mappings').get() as { count: number }).count
		).toBe(1);

		deactivateMapping(database, 'vid_gamma', { expectedRevision: 4 });
		expect(readIdentityCatalog(database).visualIdentities[0].mappingState).toBeNull();
	});

	it('fails a stale concurrent operator revision without partial writes', () => {
		const database = databaseFixture();
		createOfficialIdentity(database, { officialId: 'NL-005' }, { expectedRevision: 0 });

		expect(() =>
			createOfficialIdentity(database, { officialId: 'NL-006' }, { expectedRevision: 0 })
		).toThrow('changed');
		expect(
			(
				database.prepare('SELECT COUNT(*) AS count FROM official_identities').get() as {
					count: number;
				}
			).count
		).toBe(1);
		expect(readIdentityCatalog(database).control.operatorRevision).toBe(1);
	});

	it('merges and splits visual evidence', () => {
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
						`SELECT merged_into_visual_identity_id
						 FROM visual_identities WHERE visual_identity_id = 'vid_merge_source'`
					)
					.get() as { merged_into_visual_identity_id: string }
			).merged_into_visual_identity_id
		).toBe('vid_merge_target');

		const split = splitVisualIdentity(
			database,
			{ sourceVisualIdentityId: 'vid_merge_target', trackletIds: ['trk_merge_2'] },
			{ expectedRevision: 1 }
		);
		expect(split).toMatch(/^vid_/);
		expect(
			(
				database
					.prepare(
						`SELECT visual_identity_id
						 FROM visual_identity_tracklets WHERE tracklet_id = 'trk_merge_2'`
					)
					.get() as { visual_identity_id: string }
			).visual_identity_id
		).toBe(split);
		expect(readIdentityCatalog(database).control.operatorRevision).toBe(2);
	});
});
