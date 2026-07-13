import re
from dataclasses import dataclass, field

import numpy as np
from aidetector.dazzlecow.gallery import CowIdentityGallery
from aidetector.dazzlecow.tracklet_store import StoredTracklet, TrackletStore
from aidetector.domain.vectors import normalize_rows, normalize_vector
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

DEFAULT_ENROLLMENT_SIMILARITY = 0.76
DEFAULT_ENROLLMENT_MARGIN = 0.0


@dataclass(frozen=True)
class EnrollmentTrack:
    key: str
    embedding: ndarray
    cannot_link: frozenset[str] = field(default_factory=frozenset)


def cluster_tracks(
    tracks: list[EnrollmentTrack],
    *,
    similarity_threshold: float,
    neighbors: int = 5,
    identity_prefix: str = "cow",
) -> dict[str, str]:
    if not tracks:
        return {}
    if len({track.key for track in tracks}) != len(tracks):
        raise ValueError("Enrollment track keys must be unique")

    embeddings = normalize_rows(
        np.asarray([track.embedding for track in tracks], dtype=np.float32)
    )
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)
    nearest_count = min(max(1, neighbors), len(tracks) - 1)
    nearest = (
        [set()]
        if len(tracks) == 1
        else [
            set(np.argpartition(row, -nearest_count)[-nearest_count:])
            for row in similarities
        ]
    )
    edges = sorted(
        (
            (float(similarities[left, right]), left, right)
            for left in range(len(tracks))
            for right in range(left + 1, len(tracks))
            if right in nearest[left]
            and left in nearest[right]
            and similarities[left, right] >= similarity_threshold
        ),
        reverse=True,
    )

    parent = list(range(len(tracks)))
    members = {index: {index} for index in range(len(tracks))}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for _, left, right in edges:
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            continue
        left_members = members[left_root]
        right_members = members[right_root]
        if _cannot_merge(tracks, left_members, right_members):
            continue
        if (
            np.min(similarities[np.ix_(list(left_members), list(right_members))])
            < similarity_threshold
        ):
            continue
        parent[right_root] = left_root
        members[left_root] |= members.pop(right_root)

    components: dict[int, list[int]] = {}
    for index in range(len(tracks)):
        components.setdefault(root(index), []).append(index)
    ordered = sorted(
        components.values(),
        key=lambda component: min(tracks[index].key for index in component),
    )
    return {
        tracks[index].key: f"{identity_prefix}-{number:04d}"
        for number, component in enumerate(ordered, 1)
        for index in component
    }


def cluster_tracklets(
    tracks: list[EnrollmentTrack],
    *,
    similarity_threshold: float,
    margin_threshold: float = 0.0,
    identity_prefix: str = "cow",
) -> dict[str, str]:
    """Cluster fragmented tracks while preserving known cannot-link constraints."""
    if not tracks:
        return {}
    if len({track.key for track in tracks}) != len(tracks):
        raise ValueError("Enrollment track keys must be unique")

    embeddings = normalize_rows(
        np.asarray([track.embedding for track in tracks], dtype=np.float32)
    )
    clusters = {index: {index} for index in range(len(tracks))}
    centroids = {index: embeddings[index] for index in range(len(tracks))}

    while True:
        similarities = _cluster_similarities(tracks, clusters, centroids)
        best, second = _nearest_clusters(similarities)
        candidates = _merge_candidates(
            best,
            second,
            similarity_threshold,
            margin_threshold,
        )
        if not candidates:
            break

        _, left, right = max(candidates)
        clusters[left] |= clusters.pop(right)
        centroid = embeddings[list(clusters[left])].mean(axis=0, keepdims=True)
        centroids[left] = normalize_rows(centroid)[0]
        del centroids[right]

    ordered = sorted(
        clusters.values(),
        key=lambda component: min(tracks[index].key for index in component),
    )
    return {
        tracks[index].key: f"{identity_prefix}-{number:04d}"
        for number, component in enumerate(ordered, 1)
        for index in component
    }


def _cluster_similarities(
    tracks: list[EnrollmentTrack],
    clusters: dict[int, set[int]],
    centroids: dict[int, ndarray],
) -> dict[tuple[int, int], float]:
    similarities = {}
    cluster_ids = sorted(clusters)
    for offset, left in enumerate(cluster_ids):
        for right in cluster_ids[offset + 1 :]:
            if not _cannot_merge(tracks, clusters[left], clusters[right]):
                similarities[left, right] = float(centroids[left] @ centroids[right])
    return similarities


def _nearest_clusters(
    similarities: dict[tuple[int, int], float],
) -> tuple[dict[int, tuple[float, int]], dict[int, float]]:
    best: dict[int, tuple[float, int]] = {}
    second: dict[int, float] = {}
    for (left, right), similarity in similarities.items():
        for current, other in ((left, right), (right, left)):
            if current not in best or similarity > best[current][0]:
                if current in best:
                    second[current] = best[current][0]
                best[current] = similarity, other
            elif similarity > second.get(current, -np.inf):
                second[current] = similarity
    return best, second


def _merge_candidates(
    best: dict[int, tuple[float, int]],
    second: dict[int, float],
    similarity_threshold: float,
    margin_threshold: float,
) -> list[tuple[float, int, int]]:
    candidates = []
    for left, (similarity, right) in best.items():
        mutual = best.get(right, (-np.inf, -1))[1] == left
        alternative = max(second.get(left, -np.inf), second.get(right, -np.inf))
        if (
            left < right
            and mutual
            and similarity >= similarity_threshold
            and similarity - alternative >= margin_threshold
        ):
            candidates.append((similarity, left, right))
    return candidates


def cluster_known_count(
    tracks: list[EnrollmentTrack],
    identity_count: int,
    *,
    attempts: int = 5000,
    max_iterations: int = 100,
    seed: int = 84000,
    identity_prefix: str = "cow",
) -> dict[str, str]:
    if not 1 <= identity_count <= len(tracks):
        raise ValueError("Identity count must be between one and the track count")
    if attempts < 1 or max_iterations < 1:
        raise ValueError("Clustering attempts and iterations must be positive")

    embeddings = normalize_rows(
        np.asarray([track.embedding for track in tracks], dtype=np.float32)
    )
    forbidden = np.asarray(
        [
            [
                right.key in left.cannot_link or left.key in right.cannot_link
                for right in tracks
            ]
            for left in tracks
        ]
    )
    if not forbidden.any():
        labels = _kmeans(embeddings, identity_count, max_iterations, seed)
        return {
            track.key: f"{identity_prefix}-{int(label) + 1:04d}"
            for track, label in zip(tracks, labels, strict=True)
        }

    best_labels = _best_constrained_labels(
        embeddings,
        forbidden,
        identity_count,
        attempts,
        max_iterations,
        seed,
    )
    if best_labels is None:
        raise ValueError("Cannot satisfy enrollment track constraints")
    return {
        track.key: f"{identity_prefix}-{int(label) + 1:04d}"
        for track, label in zip(tracks, best_labels, strict=True)
    }


def _kmeans(
    embeddings: ndarray,
    identity_count: int,
    max_iterations: int,
    seed: int,
) -> ndarray:
    from sklearn.cluster import KMeans

    return KMeans(
        n_clusters=identity_count,
        n_init=50,
        max_iter=max_iterations,
        random_state=seed,
    ).fit_predict(embeddings)


def _best_constrained_labels(
    embeddings: ndarray,
    forbidden: ndarray,
    identity_count: int,
    attempts: int,
    max_iterations: int,
    seed: int,
) -> ndarray | None:
    generator = np.random.default_rng(seed)
    best_labels = None
    best_score = np.inf
    for _ in range(attempts):
        result = _constrained_kmeans(
            embeddings,
            forbidden,
            identity_count,
            max_iterations,
            generator,
        )
        if result is None:
            continue
        labels, centers = result
        score = float(np.sum(1 - np.sum(embeddings * centers[labels], axis=1)))
        if score < best_score:
            best_score = score
            best_labels = labels.copy()
    return best_labels


def _constrained_kmeans(
    embeddings: ndarray,
    forbidden: ndarray,
    identity_count: int,
    max_iterations: int,
    generator: np.random.Generator,
) -> tuple[ndarray, ndarray] | None:
    centers = embeddings[
        generator.choice(len(embeddings), identity_count, replace=False)
    ]
    labels = None
    for _ in range(max_iterations):
        updated = _constrained_assign(embeddings, centers, forbidden, generator)
        if updated is None:
            return None
        converged = labels is not None and np.array_equal(labels, updated)
        labels = updated
        if converged:
            break
        centers = normalize_rows(
            np.asarray(
                [
                    embeddings[labels == label].mean(axis=0)
                    for label in range(identity_count)
                ]
            )
        )
    return (labels, centers) if labels is not None else None


def match_camera_tracks(
    tracks: list[EnrollmentTrack],
    cameras: dict[str, str],
    *,
    similarity_threshold: float | None = None,
    margin_threshold: float = 0.0,
    identity_prefix: str = "cow",
) -> dict[str, str]:
    if set(cameras) != {track.key for track in tracks}:
        raise ValueError("Every enrollment track must have exactly one camera")
    if not tracks:
        return {}

    embeddings = {track.key: normalize_vector(track.embedding) for track in tracks}
    by_camera: dict[str, list[str]] = {}
    for key, camera in cameras.items():
        by_camera.setdefault(camera, []).append(key)
    ordered_cameras = sorted(
        by_camera, key=lambda camera: (-len(by_camera[camera]), camera)
    )
    clusters = [[key] for key in sorted(by_camera[ordered_cameras[0]])]

    for camera in ordered_cameras[1:]:
        keys = sorted(by_camera[camera])
        centroids = [
            normalize_rows(
                np.asarray(
                    [np.mean([embeddings[member] for member in cluster], axis=0)],
                    dtype=np.float32,
                )
            )[0]
            for cluster in clusters
        ]
        similarities = np.asarray(
            [
                [float(centroid @ embeddings[key]) for key in keys]
                for centroid in centroids
            ],
            dtype=np.float32,
        )
        rows, columns = linear_sum_assignment(similarities, maximize=True)
        matched = set()
        for row, column in zip(rows, columns, strict=True):
            key = keys[column]
            similarity = float(similarities[row, column])
            alternatives = np.concatenate(
                (
                    similarities[row, :column],
                    similarities[row, column + 1 :],
                    similarities[:row, column],
                    similarities[row + 1 :, column],
                )
            )
            margin = (
                similarity - float(np.max(alternatives))
                if alternatives.size
                else float("inf")
            )
            if similarity_threshold is not None and (
                similarity < similarity_threshold or margin < margin_threshold
            ):
                continue
            clusters[row].append(key)
            matched.add(key)
        clusters.extend([key] for key in keys if key not in matched)

    ordered = sorted(clusters, key=lambda cluster: min(cluster))
    return {
        key: f"{identity_prefix}-{number:04d}"
        for number, cluster in enumerate(ordered, 1)
        for key in cluster
    }


def finalize_enrollment(
    store: TrackletStore,
    *,
    similarity_threshold: float = DEFAULT_ENROLLMENT_SIMILARITY,
    margin_threshold: float = DEFAULT_ENROLLMENT_MARGIN,
    identity_count: int | None = None,
) -> dict[str, str]:
    stored = store.tracklets()
    if not stored:
        raise ValueError("Enrollment database has no tracklets")
    sessions = {tracklet.session for tracklet in stored}
    if len(sessions) != 1:
        raise ValueError("Finalize one enrollment session per database")
    tracks = _enrollment_tracks(stored)
    if identity_count is None:
        assignments = cluster_tracklets(
            tracks,
            similarity_threshold=similarity_threshold,
            margin_threshold=margin_threshold,
        )
    else:
        sources = {tracklet.id: tracklet.source for tracklet in stored}
        source_counts = {
            source: sum(value == source for value in sources.values())
            for source in set(sources.values())
        }
        assignments = (
            match_camera_tracks(
                tracks,
                sources,
            )
            if source_counts
            and all(count == identity_count for count in source_counts.values())
            else cluster_known_count(tracks, identity_count)
        )
    store.replace_assignments(assignments)
    return assignments


def finalize_pending_enrollment(
    store: TrackletStore,
    *,
    similarity_threshold: float = DEFAULT_ENROLLMENT_SIMILARITY,
    margin_threshold: float = DEFAULT_ENROLLMENT_MARGIN,
    create_after: int = 3,
    gallery: CowIdentityGallery | None = None,
    existing_match_threshold: float = 0.9,
    existing_match_margin: float = 0.2,
) -> dict[str, str]:
    if create_after < 1:
        raise ValueError("Pending identity maturity must be positive")
    if gallery is None:
        embeddings, keys, labels = store.gallery_data()
        gallery = CowIdentityGallery(embeddings, keys, labels)
    stored = store.pending_tracklets()
    if not stored:
        raise ValueError("Identity database has no pending tracklets")
    assignments = cluster_tracklets(
        _enrollment_tracks(stored),
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        identity_prefix="pending",
    )
    group_counts = {
        group: sum(value == group for value in assignments.values())
        for group in set(assignments.values())
    }
    groups = sorted(
        group for group, count in group_counts.items() if count >= create_after
    )
    if not groups:
        raise ValueError(
            f"Pending identities need at least {create_after} tracklets each"
        )
    grouped_tracklets = {
        group: [tracklet for tracklet in stored if assignments[tracklet.id] == group]
        for group in groups
    }
    group_identities = {}
    new_groups = []
    for group in groups:
        identity = _match_existing_identity(
            grouped_tracklets[group],
            gallery,
            existing_match_threshold,
            existing_match_margin,
        )
        if identity is None:
            new_groups.append(group)
        else:
            group_identities[group] = identity
    new_identities = iter(_next_identity_ids(store.identity_keys(), len(new_groups)))
    for group in new_groups:
        group_identities[group] = next(new_identities)
    assignments = {
        tracklet: group_identities[group]
        for tracklet, group in assignments.items()
        if group in group_identities
    }
    store.assign_pending(assignments)
    return assignments


def _match_existing_identity(
    tracklets: list[StoredTracklet],
    gallery: CowIdentityGallery,
    similarity_threshold: float,
    margin_threshold: float,
) -> str | None:
    embedding = normalize_vector(
        np.mean([tracklet.embedding for tracklet in tracklets], axis=0)
    )
    match = gallery.score(embedding)
    if match.similarity < similarity_threshold or match.margin < margin_threshold:
        return None
    return match.key


def _enrollment_tracks(stored) -> list[EnrollmentTrack]:
    return [
        EnrollmentTrack(
            tracklet.id,
            tracklet.embedding,
            frozenset(
                other.id
                for other in stored
                if other.run == tracklet.run
                and other.source == tracklet.source
                and other.id != tracklet.id
                and max(other.first_frame, tracklet.first_frame)
                <= min(other.last_frame, tracklet.last_frame)
            ),
        )
        for tracklet in stored
    ]


def _next_identity_ids(existing: list[str], count: int) -> list[str]:
    used = set(existing)
    number = max(
        (
            int(match.group(1))
            for identity in existing
            if (match := re.fullmatch(r"cow-(\d+)", identity))
        ),
        default=0,
    )
    identities = []
    while len(identities) < count:
        number += 1
        identity = f"cow-{number:04d}"
        if identity not in used:
            identities.append(identity)
    return identities


def _cannot_merge(
    tracks: list[EnrollmentTrack],
    left: set[int],
    right: set[int],
) -> bool:
    return any(
        tracks[right_index].key in tracks[left_index].cannot_link
        or tracks[left_index].key in tracks[right_index].cannot_link
        for left_index in left
        for right_index in right
    )


def _constrained_assign(
    embeddings: ndarray,
    centers: ndarray,
    forbidden: ndarray,
    generator: np.random.Generator,
) -> ndarray | None:
    labels = np.full(len(embeddings), -1, dtype=int)
    tie_breakers = generator.random(len(embeddings))
    order = sorted(
        range(len(embeddings)),
        key=lambda index: (-int(forbidden[index].sum()), tie_breakers[index]),
    )
    for index in order:
        for label in np.argsort(-(centers @ embeddings[index])):
            members = labels == label
            if not np.any(forbidden[index] & members):
                labels[index] = label
                break
        if labels[index] < 0:
            return None
    if len(set(labels)) != len(centers):
        return None
    return labels
