import numpy as np
from numpy import ndarray


def clustering_metrics(
    embeddings: ndarray,
    labels: list[str] | list[int],
    *,
    neighbors: int = 5,
) -> dict[str, float]:
    try:
        from scipy.optimize import linear_sum_assignment
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
        )
        from sklearn.neighbors import NearestNeighbors
    except ImportError as error:
        raise RuntimeError(
            "Paper metrics require the 'dazzlecow' optional dependencies"
        ) from error

    embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))
    labels = np.asarray(labels)
    unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
    if len(embeddings) < 2 or len(unique_labels) < 2:
        raise ValueError("Clustering metrics require at least two identities")

    clusters = KMeans(
        n_clusters=len(unique_labels),
        n_init=50,
        max_iter=500,
        random_state=42,
    ).fit_predict(embeddings)
    confusion = np.zeros((len(unique_labels), len(unique_labels)), dtype=np.int64)
    for label, cluster in zip(encoded_labels, clusters, strict=True):
        confusion[label, cluster] += 1
    rows, columns = linear_sum_assignment(-confusion)

    count = min(neighbors + 1, len(embeddings))
    nearest = NearestNeighbors(n_neighbors=count).fit(embeddings)
    indices = nearest.kneighbors(embeddings, return_distance=False)
    correct = 0
    for index, neighbors_for_sample in enumerate(indices):
        candidates = [item for item in neighbors_for_sample if item != index][
            :neighbors
        ]
        votes = np.bincount(encoded_labels[candidates])
        correct += int(np.argmax(votes)) == encoded_labels[index]

    return {
        "knn_accuracy": correct / len(embeddings),
        "adjusted_rand_index": adjusted_rand_score(encoded_labels, clusters),
        "adjusted_mutual_info": adjusted_mutual_info_score(
            encoded_labels,
            clusters,
        ),
        "normalized_mutual_info": normalized_mutual_info_score(
            encoded_labels,
            clusters,
        ),
        "hungarian_accuracy": float(confusion[rows, columns].sum())
        / len(embeddings),
    }


def _normalize(values: ndarray) -> ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)
