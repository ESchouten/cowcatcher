from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aidetector.utils.config import IdentityResult
from numpy import ndarray


@dataclass(frozen=True)
class GalleryScore:
    identity: str
    similarity: float
    margin: float


class DazzleCowGallery:
    def __init__(
        self,
        path: Path,
        *,
        neighbors: int = 5,
        match_threshold: float = 0.75,
        match_margin: float = 0,
    ):
        data = np.load(path, allow_pickle=False)
        self._initialize(
            data["embeddings"],
            data["identities"],
            neighbors,
            match_threshold,
            match_margin,
        )

    @classmethod
    def from_data(
        cls,
        embeddings: ndarray,
        identities: ndarray,
        *,
        neighbors: int = 5,
        match_threshold: float = 0.75,
        match_margin: float = 0,
    ) -> "DazzleCowGallery":
        gallery = cls.__new__(cls)
        gallery._initialize(
            embeddings,
            identities,
            neighbors,
            match_threshold,
            match_margin,
        )
        return gallery

    def _initialize(
        self,
        embeddings: ndarray,
        identities: ndarray,
        neighbors: int,
        match_threshold: float,
        match_margin: float,
    ) -> None:
        self.embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))
        self.identities = np.asarray(identities, dtype=str)
        if self.embeddings.ndim != 2:
            raise ValueError("DazzleCow gallery embeddings must be a 2D array")
        if self.identities.ndim != 1:
            raise ValueError("DazzleCow gallery identities must be a 1D array")
        if len(self.embeddings) != len(self.identities):
            raise ValueError("DazzleCow gallery embeddings and identities differ in length")
        if len(self.identities) == 0:
            raise ValueError("DazzleCow gallery is empty")
        self.neighbors = max(1, neighbors)
        self.match_threshold = match_threshold
        self.match_margin = match_margin

    def match(self, embedding: ndarray) -> IdentityResult | None:
        score = self.score(embedding)
        if (
            score.similarity < self.match_threshold
            or score.margin < self.match_margin
        ):
            return None
        return IdentityResult(score.identity, score.similarity)

    def score(self, embedding: ndarray) -> GalleryScore:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1 or embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "DazzleCow embedding dimension does not match the gallery "
                f"({embedding.shape} != ({self.embeddings.shape[1]},))"
            )
        similarities = self.embeddings @ _normalize(embedding.reshape(1, -1))[0]
        count = min(self.neighbors, len(similarities))
        indices = np.argpartition(similarities, -count)[-count:]

        votes: dict[str, float] = {}
        for index in indices:
            identity = str(self.identities[index])
            votes[identity] = votes.get(identity, 0) + max(0, float(similarities[index]))

        ranked = sorted(votes, key=lambda item: votes[item], reverse=True)
        identity = ranked[0]
        similarity = max(
            float(similarities[index])
            for index in indices
            if self.identities[index] == identity
        )
        total = sum(votes.values())
        second_vote = votes[ranked[1]] if len(ranked) > 1 else 0
        margin = (votes[identity] - second_vote) / total if total else 0
        return GalleryScore(identity, similarity, margin)


def save_gallery(path: Path, embeddings: ndarray, identities: list[str]) -> None:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("DazzleCow gallery embeddings must be a 2D array")
    if len(embeddings) != len(identities):
        raise ValueError("DazzleCow gallery embeddings and identities differ in length")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=_normalize(embeddings),
        identities=np.asarray(identities, dtype=str),
    )


def _normalize(values: ndarray) -> ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)
