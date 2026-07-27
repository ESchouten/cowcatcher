PRAGMA foreign_keys = ON;

CREATE TABLE official_identities (
    official_id TEXT PRIMARY KEY
        CHECK (length(trim(official_id)) > 0),
    display_name TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived')),
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE visual_identities (
    visual_identity_id TEXT PRIMARY KEY
        CHECK (visual_identity_id GLOB 'vid_*'),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'active', 'merged')),
    merged_into_visual_identity_id TEXT
        REFERENCES visual_identities(visual_identity_id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (status = 'merged' AND merged_into_visual_identity_id IS NOT NULL)
        OR (status <> 'merged' AND merged_into_visual_identity_id IS NULL)
    )
);

CREATE TABLE tracklets (
    tracklet_id TEXT PRIMARY KEY
        CHECK (tracklet_id GLOB 'trk_*'),
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    first_captured_at TEXT NOT NULL,
    last_captured_at TEXT NOT NULL,
    observation_count INTEGER NOT NULL CHECK (observation_count > 0),
    evidence_status TEXT NOT NULL
        CHECK (
            evidence_status IN (
                'eligible',
                'insufficient',
                'switch_risk'
            )
        ),
    preview_jpeg BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, source, track_id)
);

CREATE TABLE evidence_frames (
    evidence_id TEXT PRIMARY KEY
        CHECK (evidence_id GLOB 'evd_*'),
    tracklet_id TEXT NOT NULL
        REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL CHECK (frame_index >= 0),
    captured_at TEXT NOT NULL,
    image_sha256 TEXT NOT NULL
        CHECK (length(image_sha256) = 64),
    preview_jpeg BLOB NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    quality REAL NOT NULL CHECK (quality >= 0.0 AND quality <= 1.0),
    created_at TEXT NOT NULL,
    UNIQUE (tracklet_id, frame_index),
    CHECK (length(embedding) = embedding_dimension * 4)
);

CREATE TABLE visual_identity_tracklets (
    tracklet_id TEXT PRIMARY KEY
        REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    visual_identity_id TEXT NOT NULL
        REFERENCES visual_identities(visual_identity_id),
    assignment_kind TEXT NOT NULL
        CHECK (assignment_kind IN ('initial', 'human_merge', 'human_split')),
    assigned_at TEXT NOT NULL
);

CREATE TABLE mappings (
    mapping_id TEXT PRIMARY KEY CHECK (mapping_id GLOB 'map_*'),
    visual_identity_id TEXT NOT NULL
        REFERENCES visual_identities(visual_identity_id),
    official_id TEXT NOT NULL
        REFERENCES official_identities(official_id),
    state TEXT NOT NULL
        CHECK (state IN ('provisional', 'confirmed', 'inactive', 'rejected')),
    provisional_tracklet_id TEXT NOT NULL
        REFERENCES tracklets(tracklet_id),
    confirmation_tracklet_id TEXT
        REFERENCES tracklets(tracklet_id),
    version INTEGER NOT NULL CHECK (version > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (visual_identity_id, version),
    CHECK (
        (
            state = 'confirmed'
            AND confirmation_tracklet_id IS NOT NULL
            AND confirmation_tracklet_id <> provisional_tracklet_id
        )
        OR state <> 'confirmed'
    )
);

CREATE UNIQUE INDEX mappings_one_active_visual
ON mappings(visual_identity_id)
WHERE state IN ('provisional', 'confirmed');

CREATE UNIQUE INDEX mappings_one_active_official
ON mappings(official_id)
WHERE state IN ('provisional', 'confirmed');

CREATE TABLE gallery_versions (
    gallery_version INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_revision INTEGER NOT NULL,
    encoder_key TEXT NOT NULL,
    configuration_sha256 TEXT NOT NULL CHECK (length(configuration_sha256) = 64),
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    state TEXT NOT NULL CHECK (state IN ('active', 'superseded')),
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX gallery_one_active_version
ON gallery_versions(state)
WHERE state = 'active';

CREATE TABLE gallery_items (
    gallery_version INTEGER NOT NULL
        REFERENCES gallery_versions(gallery_version) ON DELETE CASCADE,
    visual_identity_id TEXT NOT NULL
        REFERENCES visual_identities(visual_identity_id),
    official_id TEXT NOT NULL
        REFERENCES official_identities(official_id),
    prototype BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    evidence_ids_json TEXT NOT NULL
        CHECK (
            json_valid(evidence_ids_json)
            AND json_type(evidence_ids_json) = 'array'
        ),
    PRIMARY KEY (gallery_version, visual_identity_id),
    UNIQUE (gallery_version, official_id),
    CHECK (length(prototype) = embedding_dimension * 4)
);

CREATE TABLE control (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    operator_revision INTEGER NOT NULL DEFAULT 0 CHECK (operator_revision >= 0),
    active_gallery_version INTEGER
        REFERENCES gallery_versions(gallery_version),
    configuration_sha256 TEXT,
    encoder_key TEXT,
    embedding_dimension INTEGER,
    updated_at TEXT NOT NULL
);

INSERT INTO control (
    singleton,
    operator_revision,
    updated_at
) VALUES (1, 0, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

PRAGMA user_version = 2;
