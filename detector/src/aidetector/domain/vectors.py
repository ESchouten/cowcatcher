import numpy as np
from numpy import ndarray


def normalize_vector(values: ndarray) -> ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError("Expected a one-dimensional embedding")
    norm = np.linalg.norm(vector)
    return vector / max(float(norm), np.finfo(np.float32).eps)


def normalize_rows(values: ndarray) -> ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)
