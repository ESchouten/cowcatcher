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
    run: str
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
                run TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                observations INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                preview BLOB NOT NULL,
                learned INTEGER NOT NULL DEFAULT 0
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
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(tracklets)")
        }
        if "run" not in columns:
            self.connection.execute(
                "ALTER TABLE tracklets ADD COLUMN run TEXT NOT NULL DEFAULT ''"
            )
        if "learned" not in columns:
            self.connection.execute(
                "ALTER TABLE tracklets ADD COLUMN learned INTEGER NOT NULL DEFAULT 0"
            )
        active_session = self._control("active_session")
        self.session = session or active_session or uuid.uuid4().hex
        self.run = uuid.uuid4().hex[:8]
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
        tracklet_id = self._tracklet_id(snapshot)
        preview = _encode_preview(snapshot.preview)
        with self.connection:
            self._upsert_tracklet(tracklet_id, snapshot, preview, learned=False)
        return tracklet_id

    def update_identity(
        self,
        snapshot: TrackletSnapshot,
        *,
        max_samples: int = 20,
        duplicate_similarity: float = 0.995,
    ) -> bool:
        if snapshot.identity_key is None:
            raise ValueError("Cannot update an identity without an identity key")
        if max_samples < 1:
            raise ValueError("Maximum identity samples must be positive")

        tracklet_id = self._tracklet_id(snapshot)
        preview = _encode_preview(snapshot.preview)
        with self.connection:
            identity = self.connection.execute(
                "SELECT 1 FROM identities WHERE identity = ?",
                (snapshot.identity_key,),
            ).fetchone()
            if identity is None:
                raise ValueError(f"Unknown identity: {snapshot.identity_key}")
            exists = self.connection.execute(
                "SELECT 1 FROM tracklets WHERE id = ?", (tracklet_id,)
            ).fetchone()
            if exists is None and not self._is_novel(
                snapshot.identity_key,
                snapshot.embedding,
                duplicate_similarity,
            ):
                return False
            self._upsert_tracklet(tracklet_id, snapshot, preview, learned=True)
            self.connection.execute(
                """
                INSERT INTO identity_tracklets (tracklet_id, identity) VALUES (?, ?)
                ON CONFLICT(tracklet_id) DO UPDATE SET identity = excluded.identity
                """,
                (tracklet_id, snapshot.identity_key),
            )
            self._trim_learned_samples(snapshot.identity_key, max_samples)
            self._bump_revision()
        return True

    def update_pending(
        self,
        snapshot: TrackletSnapshot,
        *,
        max_samples: int = 500,
        duplicate_similarity: float = 0.995,
    ) -> bool:
        if max_samples < 1:
            raise ValueError("Maximum pending samples must be positive")
        tracklet_id = self._tracklet_id(snapshot)
        preview = _encode_preview(snapshot.preview)
        with self.connection:
            assigned = self.connection.execute(
                "SELECT 1 FROM identity_tracklets WHERE tracklet_id = ?",
                (tracklet_id,),
            ).fetchone()
            if assigned is not None:
                return False
            exists = self.connection.execute(
                "SELECT 1 FROM tracklets WHERE id = ?", (tracklet_id,)
            ).fetchone()
            if exists is None and not self._is_pending_novel(
                snapshot.embedding, duplicate_similarity
            ):
                return False
            self._upsert_tracklet(tracklet_id, snapshot, preview, learned=False)
            self.connection.execute(
                """
                DELETE FROM tracklets
                WHERE id IN (
                    SELECT t.id
                    FROM tracklets t
                    LEFT JOIN identity_tracklets it ON it.tracklet_id = t.id
                    WHERE it.tracklet_id IS NULL
                    ORDER BY t.rowid DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (max_samples,),
            )
        return True

    def ensure_embedding_model(self, model: str, dimension: int) -> None:
        if not model or dimension < 1:
            raise ValueError("Identity model and embedding dimension are required")
        expected_bytes = dimension * np.dtype(np.float32).itemsize
        stored_lengths = {
            int(row["bytes"])
            for row in self.connection.execute(
                "SELECT DISTINCT length(embedding) AS bytes FROM tracklets"
            )
        }
        stored_model = self._control("embedding_model")
        stored_dimension = self._control("embedding_dimension")
        if stored_lengths and stored_lengths != {expected_bytes}:
            raise ValueError(
                "Identity database uses incompatible embeddings; remove it and "
                "run enrollment again"
            )
        if stored_model is not None and stored_model != model:
            raise ValueError(
                f"Identity database uses {stored_model}, expected {model}; remove "
                "it and run enrollment again"
            )
        if stored_dimension is not None and int(stored_dimension) != dimension:
            raise ValueError(
                "Identity database uses an incompatible embedding dimension; "
                "remove it and run enrollment again"
            )
        with self.connection:
            self._set_control("embedding_model", model)
            self._set_control("embedding_dimension", str(dimension))

    def tracklets(self) -> list[StoredTracklet]:
        rows = self.connection.execute(
            """
            SELECT id, session, run, source, track_id, first_frame, last_frame,
                   observations, embedding
            FROM tracklets
            ORDER BY id
            """
        )
        return [
            StoredTracklet(
                row["id"],
                row["session"],
                row["run"],
                row["source"],
                row["track_id"],
                row["first_frame"],
                row["last_frame"],
                row["observations"],
                np.frombuffer(row["embedding"], dtype=np.float32).copy(),
            )
            for row in rows
        ]

    def pending_tracklets(self) -> list[StoredTracklet]:
        pending = {
            row["id"]
            for row in self.connection.execute(
                """
                SELECT t.id
                FROM tracklets t
                LEFT JOIN identity_tracklets it ON it.tracklet_id = t.id
                WHERE it.tracklet_id IS NULL
                """
            )
        }
        return [tracklet for tracklet in self.tracklets() if tracklet.id in pending]

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

    def assign_pending(
        self,
        assignments: dict[str, str],
        *,
        max_learned_samples: int = 20,
    ) -> None:
        pending = {tracklet.id for tracklet in self.pending_tracklets()}
        if not assignments or not set(assignments) <= pending:
            raise ValueError("Assignments must reference pending tracklets")
        if max_learned_samples < 1:
            raise ValueError("Maximum learned samples must be positive")
        identities = set(assignments.values())
        existing = set(self.identity_keys())
        with self.connection:
            self.connection.executemany(
                "INSERT INTO identities (identity) VALUES (?)",
                [(identity,) for identity in sorted(identities - existing)],
            )
            self.connection.executemany(
                "INSERT INTO identity_tracklets (tracklet_id, identity) VALUES (?, ?)",
                sorted(assignments.items()),
            )
            learned = [
                (tracklet,)
                for tracklet, identity in assignments.items()
                if identity in existing
            ]
            self.connection.executemany(
                "UPDATE tracklets SET learned = 1 WHERE id = ?",
                learned,
            )
            for identity in identities & existing:
                self._trim_learned_samples(identity, max_learned_samples)
            self._set_control("finalize_requested", "0")
            self._set_control("finalize_error", "")
            self._bump_revision()

    def identity_keys(self) -> list[str]:
        return [
            row["identity"]
            for row in self.connection.execute(
                "SELECT identity FROM identities ORDER BY identity"
            )
        ]

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

    def gallery_data(self) -> tuple[ndarray, list[str], list[str]]:
        rows = list(
            self.connection.execute(
                """
                SELECT t.embedding, i.identity,
                       COALESCE(i.animal_number, i.identity) AS label
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
            [row["identity"] for row in rows],
            [row["label"] for row in rows],
        )

    def _tracklet_id(self, snapshot: TrackletSnapshot) -> str:
        return f"{self.session}:{self.run}:{snapshot.source}:{snapshot.track_id}"

    def _is_novel(
        self,
        identity: str,
        embedding: ndarray,
        duplicate_similarity: float,
    ) -> bool:
        rows = self.connection.execute(
            """
            SELECT t.embedding
            FROM tracklets t
            JOIN identity_tracklets it ON it.tracklet_id = t.id
            WHERE it.identity = ?
            """,
            (identity,),
        )
        existing = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        if not existing:
            return True
        candidate = _normalize(np.asarray(embedding, dtype=np.float32))
        similarities = np.asarray([_normalize(value) for value in existing]) @ candidate
        return float(np.max(similarities)) < duplicate_similarity

    def _is_pending_novel(
        self,
        embedding: ndarray,
        duplicate_similarity: float,
    ) -> bool:
        rows = self.connection.execute(
            """
            SELECT t.embedding
            FROM tracklets t
            LEFT JOIN identity_tracklets it ON it.tracklet_id = t.id
            WHERE it.tracklet_id IS NULL
            """
        )
        existing = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        if not existing:
            return True
        candidate = _normalize(np.asarray(embedding, dtype=np.float32))
        similarities = np.asarray([_normalize(value) for value in existing]) @ candidate
        return float(np.max(similarities)) < duplicate_similarity

    def _upsert_tracklet(
        self,
        tracklet_id: str,
        snapshot: TrackletSnapshot,
        preview: bytes,
        *,
        learned: bool,
    ) -> None:
        embedding = np.asarray(snapshot.embedding, dtype=np.float32)
        self.connection.execute(
            """
            INSERT INTO tracklets (
                id, session, run, source, track_id, first_frame, last_frame,
                observations, embedding, preview, learned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_frame = excluded.last_frame,
                observations = excluded.observations,
                embedding = excluded.embedding,
                preview = excluded.preview,
                learned = excluded.learned
            """,
            (
                tracklet_id,
                self.session,
                self.run,
                snapshot.source,
                snapshot.track_id,
                snapshot.first_frame,
                snapshot.last_frame,
                snapshot.observations,
                embedding.tobytes(),
                preview,
                learned,
            ),
        )

    def _trim_learned_samples(self, identity: str, maximum: int) -> None:
        self.connection.execute(
            """
            DELETE FROM tracklets
            WHERE id IN (
                SELECT t.id
                FROM tracklets t
                JOIN identity_tracklets it ON it.tracklet_id = t.id
                WHERE it.identity = ? AND t.learned = 1
                ORDER BY t.rowid DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (identity, maximum),
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


def _encode_preview(preview: ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", preview)
    if not success:
        raise ValueError("Could not encode tracklet preview")
    return encoded.tobytes()


def _normalize(embedding: ndarray) -> ndarray:
    norm = np.linalg.norm(embedding)
    return embedding / max(float(norm), np.finfo(np.float32).eps)
