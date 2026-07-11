import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from aidetector.dazzlecow.tracks import TrackletSnapshot
from numpy import ndarray


@dataclass(frozen=True)
class StoredTracklet:
    id: str
    session: str
    source: str
    track_id: int
    first_frame: int
    last_frame: int
    observations: int
    embedding: ndarray


class TrackletStore:
    def __init__(self, path: Path, *, session: str | None = None):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tracklets (
                id TEXT PRIMARY KEY,
                session TEXT NOT NULL,
                source TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                observations INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                preview BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS identities (
                identity TEXT PRIMARY KEY,
                animal_number TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS identity_tracklets (
                tracklet_id TEXT PRIMARY KEY REFERENCES tracklets(id) ON DELETE CASCADE,
                identity TEXT NOT NULL REFERENCES identities(identity) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS control (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        active_session = self._control("active_session")
        self.session = session or active_session or uuid.uuid4().hex
        if active_session is None:
            self._set_control("active_session", self.session)
        if self._control("revision") is None:
            self._set_control("revision", "0")
        if self._control("finalize_requested") is None:
            self._set_control("finalize_requested", "0")
        if self._control("finalize_error") is None:
            self._set_control("finalize_error", "")
        self.connection.commit()

    def upsert(self, snapshot: TrackletSnapshot) -> str:
        tracklet_id = f"{self.session}:{snapshot.source}:{snapshot.track_id}"
        success, preview = cv2.imencode(".jpg", snapshot.preview)
        if not success:
            raise ValueError("Could not encode tracklet preview")
        embedding = np.asarray(snapshot.embedding, dtype=np.float32)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO tracklets (
                    id, session, source, track_id, first_frame, last_frame,
                    observations, embedding, preview
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_frame = excluded.last_frame,
                    observations = excluded.observations,
                    embedding = excluded.embedding,
                    preview = excluded.preview
                """,
                (
                    tracklet_id,
                    self.session,
                    snapshot.source,
                    snapshot.track_id,
                    snapshot.first_frame,
                    snapshot.last_frame,
                    snapshot.observations,
                    embedding.tobytes(),
                    preview.tobytes(),
                ),
            )
        return tracklet_id

    def tracklets(self) -> list[StoredTracklet]:
        rows = self.connection.execute(
            """
            SELECT id, session, source, track_id, first_frame, last_frame,
                   observations, embedding
            FROM tracklets
            ORDER BY id
            """
        )
        return [
            StoredTracklet(
                row["id"],
                row["session"],
                row["source"],
                row["track_id"],
                row["first_frame"],
                row["last_frame"],
                row["observations"],
                np.frombuffer(row["embedding"], dtype=np.float32).copy(),
            )
            for row in rows
        ]

    def preview(self, tracklet_id: str) -> bytes | None:
        row = self.connection.execute(
            "SELECT preview FROM tracklets WHERE id = ?", (tracklet_id,)
        ).fetchone()
        return bytes(row["preview"]) if row is not None else None

    def replace_assignments(self, assignments: dict[str, str]) -> None:
        tracklet_ids = {tracklet.id for tracklet in self.tracklets()}
        if set(assignments) != tracklet_ids:
            raise ValueError("Every stored tracklet must have exactly one identity")
        named = self.connection.execute(
            "SELECT COUNT(*) FROM identities WHERE animal_number IS NOT NULL"
        ).fetchone()[0]
        if named:
            raise ValueError("Cannot replace enrollment after naming identities")
        with self.connection:
            self.connection.execute("DELETE FROM identity_tracklets")
            self.connection.execute("DELETE FROM identities")
            self.connection.executemany(
                "INSERT INTO identities (identity) VALUES (?)",
                [(identity,) for identity in sorted(set(assignments.values()))],
            )
            self.connection.executemany(
                "INSERT INTO identity_tracklets (tracklet_id, identity) VALUES (?, ?)",
                sorted(assignments.items()),
            )
            self._set_control("finalize_requested", "0")
            self._set_control("finalize_error", "")
            self._bump_revision()

    def identities(self) -> list[dict[str, str | None | int]]:
        rows = self.connection.execute(
            """
            SELECT i.identity, i.animal_number, COUNT(it.tracklet_id) AS tracklets
            FROM identities i
            LEFT JOIN identity_tracklets it ON it.identity = i.identity
            GROUP BY i.identity, i.animal_number
            ORDER BY i.identity
            """
        )
        return [dict(row) for row in rows]

    def set_animal_number(self, identity: str, animal_number: str | None) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE identities SET animal_number = ? WHERE identity = ?",
                (animal_number or None, identity),
            )
            if cursor.rowcount == 1:
                self._bump_revision()
        if cursor.rowcount != 1:
            raise ValueError(f"Unknown identity: {identity}")

    def request_finalize(self) -> None:
        with self.connection:
            self._set_control("finalize_error", "")
            self._set_control("finalize_requested", "1")

    def finalize_requested(self) -> bool:
        return self._control("finalize_requested") == "1"

    def finalize_error(self) -> str | None:
        return self._control("finalize_error") or None

    def fail_finalize(self, error: str) -> None:
        with self.connection:
            self._set_control("finalize_requested", "0")
            self._set_control("finalize_error", error)

    def revision(self) -> int:
        return int(self._control("revision") or 0)

    def is_finalized(self) -> bool:
        return bool(
            self.connection.execute("SELECT 1 FROM identities LIMIT 1").fetchone()
        )

    def gallery_data(self) -> tuple[ndarray, list[str]]:
        rows = list(
            self.connection.execute(
                """
                SELECT t.embedding, COALESCE(i.animal_number, i.identity) AS label
                FROM tracklets t
                JOIN identity_tracklets it ON it.tracklet_id = t.id
                JOIN identities i ON i.identity = it.identity
                ORDER BY t.id
                """
            )
        )
        if not rows:
            raise ValueError("Enrollment has no assigned tracklets")
        return (
            np.asarray(
                [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
            ),
            [row["label"] for row in rows],
        )

    def close(self) -> None:
        self.connection.close()

    def _control(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM control WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row is not None else None

    def _set_control(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO control (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _bump_revision(self) -> None:
        self._set_control("revision", str(self.revision() + 1))

    def __enter__(self) -> "TrackletStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
