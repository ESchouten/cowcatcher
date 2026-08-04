PRAGMA foreign_keys = ON;

CREATE TABLE official_identities (
    official_id TEXT PRIMARY KEY CHECK (length(trim(official_id)) > 0),
    display_name TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived')),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE visual_identities (
    visual_identity_id TEXT PRIMARY KEY
        CHECK (visual_identity_id GLOB 'vid_*'),
    merged_into_visual_identity_id TEXT
        REFERENCES visual_identities(visual_identity_id),
    CHECK (merged_into_visual_identity_id <> visual_identity_id)
);

CREATE TABLE tracklets (
    tracklet_id TEXT PRIMARY KEY CHECK (tracklet_id GLOB 'trk_*'),
    source TEXT NOT NULL,
    last_captured_at TEXT NOT NULL,
    evidence_status TEXT NOT NULL
        CHECK (evidence_status IN ('eligible', 'insufficient', 'switch_risk')),
    preview_jpeg BLOB NOT NULL
);

CREATE TABLE evidence_frames (
    tracklet_id TEXT NOT NULL
        REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL CHECK (frame_index IN (0, 1)),
    embedding BLOB NOT NULL,
    PRIMARY KEY (tracklet_id, frame_index)
);

CREATE TABLE visual_identity_tracklets (
    tracklet_id TEXT PRIMARY KEY
        REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    visual_identity_id TEXT NOT NULL
        REFERENCES visual_identities(visual_identity_id)
);

CREATE TABLE mappings (
    visual_identity_id TEXT PRIMARY KEY
        REFERENCES visual_identities(visual_identity_id),
    official_id TEXT NOT NULL UNIQUE
        REFERENCES official_identities(official_id),
    state TEXT NOT NULL CHECK (state IN ('provisional', 'confirmed')),
    provisional_tracklet_id TEXT NOT NULL
        REFERENCES tracklets(tracklet_id),
    confirmation_tracklet_id TEXT
        REFERENCES tracklets(tracklet_id),
    CHECK (
        (
            state = 'confirmed'
            AND confirmation_tracklet_id IS NOT NULL
            AND confirmation_tracklet_id <> provisional_tracklet_id
        )
        OR (
            state = 'provisional'
            AND confirmation_tracklet_id IS NULL
        )
    )
);

CREATE TABLE control (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    operator_revision INTEGER NOT NULL DEFAULT 0 CHECK (operator_revision >= 0),
    encoder_signature TEXT,
    embedding_dimension INTEGER
);

INSERT INTO control (singleton) VALUES (1);

PRAGMA user_version = 3;
