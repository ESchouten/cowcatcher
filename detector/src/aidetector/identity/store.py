import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock

import numpy as np
from aidetector.utils.config import IdentityResult


@dataclass
class IdentityRecord:
    identity: str
    centroid: np.ndarray
    sample_count: int


@dataclass
class CandidateRecord:
    candidate_id: int
    centroid: np.ndarray
    sample_count: int


class SQLiteIdentityStore:
    identities: dict[str, IdentityRecord]
    candidates: dict[int, CandidateRecord]

    def __init__(
        self,
        database: Path,
        provider_id: str,
        model: str | None = None,
    ):
        self.database = database
        self.provider_id = provider_id
        self.model = model
        self.lock = RLock()
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        self.identities = {}
        self.candidates = {}
        self.reload()
        self._ensure_store_metadata()

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def reload(self) -> None:
        with self.lock:
            self.identities = {
                row[0]: IdentityRecord(
                    identity=row[0],
                    centroid=_embedding_from_blob(row[1]),
                    sample_count=row[2],
                )
                for row in self.connection.execute(
                    "SELECT identity, centroid, sample_count FROM identities"
                )
            }
            self.candidates = {
                row[0]: CandidateRecord(
                    candidate_id=row[0],
                    centroid=_embedding_from_blob(row[1]),
                    sample_count=row[2],
                )
                for row in self.connection.execute(
                    "SELECT candidate_id, centroid, sample_count FROM unknown_candidates"
                )
            }

    def create_identity(self, embedding: np.ndarray, sample_count: int = 1) -> str:
        with self.lock:
            try:
                with self.connection:
                    embedding = self._prepare_embedding(embedding)
                    return self._create_identity(embedding, sample_count)
            except Exception:
                self.reload()
                raise

    def identify(
        self,
        embedding: np.ndarray,
        match_threshold: float,
        candidate_threshold: float,
        create_after: int,
    ) -> IdentityResult | None:
        with self.lock:
            try:
                with self.connection:
                    return self._identify(
                        self._prepare_embedding(embedding),
                        match_threshold,
                        candidate_threshold,
                        create_after,
                    )
            except Exception:
                self.reload()
                raise

    def match(
        self,
        embedding: np.ndarray,
        match_threshold: float,
    ) -> IdentityResult | None:
        with self.lock:
            return self._match_identity(
                self._prepare_embedding(embedding),
                match_threshold,
            )

    def update_identity(
        self,
        identity: str,
        embedding: np.ndarray,
        match_threshold: float,
    ) -> IdentityResult | None:
        with self.lock:
            try:
                with self.connection:
                    return self._update_known_identity(
                        identity,
                        self._prepare_embedding(embedding),
                        match_threshold,
                    )
            except Exception:
                self.reload()
                raise

    def _identify(
        self,
        embedding: np.ndarray,
        match_threshold: float,
        candidate_threshold: float,
        create_after: int,
    ) -> IdentityResult | None:
        record, identity_similarity = self._best_identity(embedding)
        if record and identity_similarity >= match_threshold:
            self._update_identity(record, embedding)
            self._insert_sample(
                embedding,
                status="matched",
                identity=record.identity,
                similarity=identity_similarity,
            )
            return IdentityResult(
                identity=record.identity,
                status="matched",
                similarity=identity_similarity,
            )

        candidate, candidate_similarity = self._best_candidate(embedding)
        if candidate and candidate_similarity >= candidate_threshold:
            updated = self._update_candidate(candidate, embedding)
            if updated.sample_count >= create_after:
                return self._promote_candidate(updated, candidate_similarity)

            self._insert_sample(
                embedding,
                status="candidate",
                candidate_id=updated.candidate_id,
                similarity=candidate_similarity,
            )
            return None

        candidate = self._create_candidate(embedding)
        if candidate.sample_count >= create_after:
            return self._promote_candidate(candidate, None)

        self._insert_sample(
            embedding,
            status="candidate",
            candidate_id=candidate.candidate_id,
            similarity=None,
        )
        return None

    def _match_identity(
        self,
        embedding: np.ndarray,
        match_threshold: float,
    ) -> IdentityResult | None:
        record, similarity = self._best_identity(embedding)
        if record is None or similarity < match_threshold:
            return None
        return IdentityResult(
            identity=record.identity,
            status="matched",
            similarity=similarity,
        )

    def _update_known_identity(
        self,
        identity: str,
        embedding: np.ndarray,
        match_threshold: float,
    ) -> IdentityResult | None:
        record = self.identities.get(identity)
        if record is None:
            return None

        similarity = _cosine_similarity(embedding, record.centroid)
        if similarity < match_threshold:
            return None

        updated = self._update_identity(record, embedding)
        self._insert_sample(
            embedding,
            status="matched",
            identity=updated.identity,
            similarity=similarity,
        )
        return IdentityResult(
            identity=updated.identity,
            status="matched",
            similarity=similarity,
        )

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS identities (
                identity TEXT PRIMARY KEY,
                centroid BLOB NOT NULL,
                sample_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unknown_candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                centroid BLOB NOT NULL,
                sample_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity TEXT,
                candidate_id INTEGER,
                embedding BLOB NOT NULL,
                similarity REAL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def _ensure_store_metadata(self) -> None:
        with self.lock:
            with self.connection:
                stored_provider = self._get_metadata("provider_id")
                if stored_provider and stored_provider != self.provider_id:
                    raise ValueError(
                        f"Identity database belongs to provider {stored_provider}, "
                        f"not {self.provider_id}"
                    )
                if not stored_provider:
                    self._set_metadata("provider_id", self.provider_id)

                if self.model is not None:
                    stored_model = self._get_metadata("model")
                    if stored_model and stored_model != self.model:
                        raise ValueError(
                            f"Identity database uses model {stored_model}, "
                            f"not {self.model}"
                        )
                    if not stored_model:
                        self._set_metadata("model", self.model)

                embedding_dim = self._get_metadata("embedding_dim")
                loaded_dims = self._loaded_embedding_dims()
                if len(loaded_dims) > 1:
                    raise ValueError(
                        "Identity database contains mixed embedding dimensions: "
                        f"{sorted(loaded_dims)}"
                    )

                loaded_dim = next(iter(loaded_dims), None)
                if embedding_dim and loaded_dim is not None:
                    expected = int(embedding_dim)
                    if loaded_dim != expected:
                        raise ValueError(
                            f"Identity database contains {loaded_dim}D embeddings, "
                            f"but metadata expects {expected}D"
                        )
                elif loaded_dim is not None:
                    self._set_metadata("embedding_dim", str(loaded_dim))

    def _prepare_embedding(self, embedding: np.ndarray) -> np.ndarray:
        embedding = _normalize_embedding(embedding)
        embedding_dim = self._get_metadata("embedding_dim")
        if embedding_dim is None:
            self._set_metadata("embedding_dim", str(len(embedding)))
        elif int(embedding_dim) != len(embedding):
            raise ValueError(
                f"Identity embedding has {len(embedding)} dimensions, "
                f"but database expects {embedding_dim}"
            )
        return embedding

    def _best_identity(
        self, embedding: np.ndarray
    ) -> tuple[IdentityRecord | None, float]:
        if not self.identities:
            return None, 0
        return max(
            (
                (identity, _cosine_similarity(embedding, identity.centroid))
                for identity in self.identities.values()
            ),
            key=lambda item: item[1],
        )

    def _best_candidate(
        self, embedding: np.ndarray
    ) -> tuple[CandidateRecord | None, float]:
        if not self.candidates:
            return None, 0
        return max(
            (
                (candidate, _cosine_similarity(embedding, candidate.centroid))
                for candidate in self.candidates.values()
            ),
            key=lambda item: item[1],
        )

    def _update_identity(
        self, identity: IdentityRecord, embedding: np.ndarray
    ) -> IdentityRecord:
        sample_count = identity.sample_count + 1
        centroid = _normalize_embedding(
            identity.centroid * identity.sample_count + embedding
        )
        self.connection.execute(
            """
            UPDATE identities
            SET centroid = ?, sample_count = ?, updated_at = ?
            WHERE identity = ?
            """,
            (
                _embedding_to_blob(centroid),
                sample_count,
                _now(),
                identity.identity,
            ),
        )
        updated = IdentityRecord(
            identity=identity.identity,
            centroid=centroid,
            sample_count=sample_count,
        )
        self.identities[identity.identity] = updated
        return updated

    def _create_candidate(self, embedding: np.ndarray) -> CandidateRecord:
        now = _now()
        cursor = self.connection.execute(
            """
            INSERT INTO unknown_candidates (centroid, sample_count, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (_embedding_to_blob(embedding), 1, now, now),
        )
        candidate_id = cursor.lastrowid
        if candidate_id is None:
            raise ValueError("Failed to create identity candidate")
        candidate = CandidateRecord(
            candidate_id=candidate_id,
            centroid=embedding,
            sample_count=1,
        )
        self.candidates[candidate.candidate_id] = candidate
        return candidate

    def _update_candidate(
        self, candidate: CandidateRecord, embedding: np.ndarray
    ) -> CandidateRecord:
        sample_count = candidate.sample_count + 1
        centroid = _normalize_embedding(
            candidate.centroid * candidate.sample_count + embedding
        )
        self.connection.execute(
            """
            UPDATE unknown_candidates
            SET centroid = ?, sample_count = ?, updated_at = ?
            WHERE candidate_id = ?
            """,
            (
                _embedding_to_blob(centroid),
                sample_count,
                _now(),
                candidate.candidate_id,
            ),
        )
        updated = CandidateRecord(
            candidate_id=candidate.candidate_id,
            centroid=centroid,
            sample_count=sample_count,
        )
        self.candidates[candidate.candidate_id] = updated
        return updated

    def _promote_candidate(
        self,
        candidate: CandidateRecord,
        similarity: float | None,
    ) -> IdentityResult:
        identity = self._create_identity(
            candidate.centroid,
            sample_count=candidate.sample_count,
        )
        self.connection.execute(
            "UPDATE samples SET identity = ? WHERE candidate_id = ?",
            (identity, candidate.candidate_id),
        )
        self.connection.execute(
            "DELETE FROM unknown_candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        )
        del self.candidates[candidate.candidate_id]
        self._insert_sample(
            candidate.centroid,
            status="created",
            identity=identity,
            candidate_id=candidate.candidate_id,
            similarity=1.0 if similarity is None else similarity,
        )
        return IdentityResult(
            identity=identity,
            status="created",
            similarity=1.0 if similarity is None else similarity,
        )

    def _insert_sample(
        self,
        embedding: np.ndarray,
        status: str,
        identity: str | None = None,
        candidate_id: int | None = None,
        similarity: float | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO samples (
                identity, candidate_id, embedding, similarity, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                identity,
                candidate_id,
                _embedding_to_blob(embedding),
                similarity,
                status,
                _now(),
            ),
        )

    def _next_identity(self) -> str:
        prefix = f"{self.provider_id}-"
        max_number = 0
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        for identity in self.identities:
            match = pattern.match(identity)
            if match:
                max_number = max(max_number, int(match.group(1)))
        return f"{prefix}{max_number + 1:04d}"

    def _create_identity(
        self,
        embedding: np.ndarray,
        sample_count: int = 1,
    ) -> str:
        identity = self._next_identity()
        centroid = _normalize_embedding(embedding)
        now = _now()
        self.connection.execute(
            """
            INSERT INTO identities (
                identity, centroid, sample_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                identity,
                _embedding_to_blob(centroid),
                sample_count,
                now,
                now,
            ),
        )
        self.identities[identity] = IdentityRecord(
            identity=identity,
            centroid=centroid,
            sample_count=sample_count,
        )
        return identity

    def _get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _loaded_embedding_dims(self) -> set[int]:
        return {
            len(record.centroid)
            for record in [*self.identities.values(), *self.candidates.values()]
        }


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(normalized)
    if norm == 0:
        return normalized
    return normalized / norm


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _embedding_from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
