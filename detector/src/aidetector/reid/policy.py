from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReidPolicy:
    enrollment_similarity: float = 0.76
    enrollment_margin: float = 0
    pending_create_after: int = 3
    learning_similarity: float = 0.75
    learning_margin: float = 0.1
    max_identity_samples: int = 20
    max_pending_samples: int = 500
    duplicate_similarity: float = 0.995


DEFAULT_REID_POLICY = ReidPolicy()
