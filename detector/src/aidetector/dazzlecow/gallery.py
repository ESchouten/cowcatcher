import numpy as np
from aidetector.domain.identity import IdentityMatch
from aidetector.domain.vectors import normalize_rows, normalize_vector
from numpy import ndarray


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
        self.embeddings = normalize_rows(embeddings)
        self.keys = np.asarray(keys, dtype=str)
        self.labels = np.asarray(labels, dtype=str)
        if self.keys.ndim != 1 or self.labels.ndim != 1:
            raise ValueError("Cow identity keys and labels must be 1D arrays")
        if len(self.embeddings) != len(self.keys) or len(self.keys) != len(self.labels):
            raise ValueError(
                "Cow identity embeddings, keys, and labels differ in length"
            )
        if len(self.keys) == 0:
            raise ValueError("Cow identity database has no assigned tracklets")
        self.identity_keys = np.unique(self.keys)
        self.identity_labels = {
            str(key): str(self.labels[np.flatnonzero(self.keys == key)[0]])
            for key in self.identity_keys
        }
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
        similarities = self.embeddings @ normalize_vector(embedding)
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
