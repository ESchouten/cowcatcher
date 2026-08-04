import numpy as np
from numpy import ndarray


def normalize_rows(values: ndarray) -> ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def normalized_mean(values: ndarray) -> ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("Expected a non-empty embedding matrix")
    mean = np.mean(matrix.astype(np.float64), axis=0, keepdims=True)
    return normalize_rows(mean.astype(np.float32))[0]
