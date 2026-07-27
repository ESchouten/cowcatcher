from aidetector.domain.detections import DetectedObject
from aidetector.reid.zone import ZoneGate

BOUNDS = (0.2, 0.2, 0.8, 0.8)


def cow(
    track_id: int | None = 7,
    box: tuple[int, int, int, int] = (20, 20, 60, 60),
) -> DetectedObject:
    return DetectedObject(*box, label="cow", confidence=0.9, track_id=track_id)


def evaluate(gate: ZoneGate, *targets: tuple[int, DetectedObject]):
    return gate.evaluate(
        "camera",
        targets,
        frame_width=100,
        frame_height=100,
    )


def test_zone_requires_stability_and_clearance_between_visits():
    gate = ZoneGate(BOUNDS)

    assert evaluate(gate, (0, cow())).target_states == {0: "locking"}
    first = evaluate(gate, (0, cow()))
    assert first.target_states == {0: "eligible"}

    assert evaluate(gate).closed_visit is None
    closed = evaluate(gate).closed_visit
    assert closed and closed.track_id == 7

    assert evaluate(gate, (0, cow())).target_states == {0: "locking"}
    assert evaluate(gate, (0, cow())).target_states == {0: "eligible"}


def test_zone_allows_tracker_warmup_and_ignores_targets_outside_it():
    gate = ZoneGate(BOUNDS)
    queued = cow(8, (0, 0, 10, 10))

    assert evaluate(gate, (0, cow(None)), (1, queued)).target_states == {
        0: "locking",
        1: "outside_zone",
    }
    assert evaluate(gate, (0, cow(7)), (1, queued)).target_states[0] == "locking"
    assert evaluate(gate, (0, cow(7)), (1, queued)).target_states[0] == "eligible"


def test_zone_marks_multiple_or_replaced_tracks_as_switch_risk():
    gate = ZoneGate(BOUNDS)
    evaluate(gate, (0, cow(7)))
    evaluate(gate, (0, cow(7)))

    multiple = evaluate(
        gate,
        (0, cow(7)),
        (1, cow(8, (40, 20, 80, 60))),
    )
    assert multiple.target_states == {0: "switch_risk", 1: "switch_risk"}
    assert multiple.tainted_track_ids == (7, 8)

    replacement = evaluate(gate, (0, cow(9)))
    assert replacement.target_states == {0: "switch_risk"}
    assert replacement.active_track_id == 7
    assert replacement.tainted_track_ids == (7, 9)


def test_duplicate_track_id_is_ambiguous_even_outside_the_zone():
    gate = ZoneGate(BOUNDS)

    result = evaluate(
        gate,
        (0, cow(7)),
        (1, cow(7, (0, 0, 10, 10))),
    )

    assert result.target_states == {0: "switch_risk", 1: "outside_zone"}
    assert result.tainted_track_ids == (7,)
