from dataclasses import dataclass
from typing import Protocol

import numpy as np
from aidetector.utils.config import IdentityResult
from numpy import ndarray


@dataclass(frozen=True)
class IdentityMatch:
    key: str
    label: str
    similarity: float
    margin: float

    @property
    def result(self) -> IdentityResult:
        return IdentityResult(self.label, self.similarity)


class IdentityMatcher(Protocol):
    def match(self, embedding: ndarray, /) -> IdentityMatch | None: ...


class CowIdentityGallery:
    def __init__(
        self,
        embeddings: ndarray,
        keys: list[str] | ndarray,
        labels: list[str] | ndarray,
        *,
        match_threshold: float = 0.68,
        match_margin: float = 0.05,
    ):
        self._initialize(
            embeddings,
            np.asarray(keys),
            np.asarray(labels),
            match_threshold,
            match_margin,
        )

    def _initialize(
        self,
        embeddings: ndarray,
        keys: ndarray,
        labels: ndarray,
        match_threshold: float,
        match_margin: float,
    ) -> None:
        self.embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))
        self.keys = np.asarray(keys, dtype=str)
        self.labels = np.asarray(labels, dtype=str)
        if self.embeddings.ndim != 2:
            raise ValueError("Cow identity embeddings must be a 2D array")
        if self.keys.ndim != 1 or self.labels.ndim != 1:
            raise ValueError("Cow identity keys and labels must be 1D arrays")
        if len(self.embeddings) != len(self.keys) or len(self.keys) != len(self.labels):
            raise ValueError(
                "Cow identity embeddings, keys, and labels differ in length"
            )
        if len(self.keys) == 0:
            raise ValueError("Cow identity database has no assigned tracklets")
        self.identity_keys = np.unique(self.keys)
        self.identity_labels = {}
        for key in self.identity_keys:
            key_labels = np.unique(self.labels[self.keys == key])
            if len(key_labels) != 1:
                raise ValueError(f"Cow identity {key} has conflicting labels")
            self.identity_labels[str(key)] = str(key_labels[0])
        self.match_threshold = match_threshold
        self.match_margin = match_margin

    def match(self, embedding: ndarray) -> IdentityMatch | None:
        score = self.score(embedding)
        if score.similarity < self.match_threshold or score.margin < self.match_margin:
            return None
        return score

    def score(self, embedding: ndarray) -> IdentityMatch:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1 or embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Cow identity embedding dimension does not match the database "
                f"({embedding.shape} != ({self.embeddings.shape[1]},))"
            )
        similarities = self.embeddings @ _normalize(embedding.reshape(1, -1))[0]
        scores = np.asarray(
            [np.max(similarities[self.keys == key]) for key in self.identity_keys]
        )
        order = np.argsort(scores)
        key = str(self.identity_keys[order[-1]])
        similarity = float(scores[order[-1]])
        margin = (
            similarity - float(scores[order[-2]]) if len(order) > 1 else float("inf")
        )
        return IdentityMatch(
            key,
            self.identity_labels[key],
            similarity,
            margin,
        )


def _normalize(values: ndarray) -> ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)
