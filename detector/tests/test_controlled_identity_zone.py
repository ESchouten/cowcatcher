from aidetector.domain.detections import DetectedObject
from aidetector.reid.controlled_zone import (
    ControlledZoneGate,
    ControlledZonePolicy,
)


def policy() -> ControlledZonePolicy:
    return ControlledZonePolicy(
        zone_id="identity_observation",
        x1=0.2,
        y1=0.2,
        x2=0.8,
        y2=0.8,
        minimum_box_inside_ratio=0.9,
        minimum_stable_frames=2,
        clear_frames=2,
    )


def cow(
    track_id: int | None = 7,
    *,
    box: tuple[int, int, int, int] = (20, 20, 60, 60),
) -> DetectedObject:
    return DetectedObject(
        *box,
        label="cow",
        confidence=0.9,
        track_id=track_id,
    )


def evaluate(
    gate: ControlledZoneGate,
    *targets: tuple[int, DetectedObject],
):
    return gate.evaluate(
        "camera",
        targets,
        frame_width=100,
        frame_height=100,
    )


def test_zone_requires_stability_and_full_clearance_between_visits() -> None:
    gate = ControlledZoneGate(policy())

    locking = evaluate(gate, (0, cow()))
    eligible = evaluate(gate, (0, cow()))
    brief_gap = evaluate(gate)
    locking_again = evaluate(gate, (0, cow()))
    eligible_again = evaluate(gate, (0, cow()))
    first_clear = evaluate(gate)
    closed = evaluate(gate)
    revisit_locking = evaluate(gate, (0, cow()))
    revisit_eligible = evaluate(gate, (0, cow()))

    assert locking.target_states == {0: "locking"}
    assert eligible.target_states == {0: "eligible"}
    assert eligible.visit_number == 1
    assert brief_gap.closed_visit is None
    assert locking_again.target_states == {0: "locking"}
    assert eligible_again.visit_number == 1
    assert first_clear.closed_visit is None
    assert closed.closed_visit is not None
    assert closed.closed_visit.track_id == 7
    assert closed.closed_visit.visit_number == 1
    assert not closed.closed_visit.switch_risk
    assert revisit_locking.target_states == {0: "locking"}
    assert revisit_eligible.visit_number == 2


def test_zone_blocks_partial_boxes_without_tainting_the_visit() -> None:
    gate = ControlledZoneGate(policy())

    partial = evaluate(gate, (0, cow(box=(10, 20, 50, 60))))
    stable = evaluate(gate, (0, cow()))
    eligible = evaluate(gate, (0, cow()))

    assert partial.target_states == {0: "locking"}
    assert stable.target_states == {0: "locking"}
    assert eligible.target_states == {0: "eligible"}
    assert eligible.tainted_track_ids == ()


def test_zone_ignores_a_queued_cow_that_does_not_intersect() -> None:
    gate = ControlledZoneGate(policy())
    queued = cow(8, box=(0, 0, 10, 10))

    locking = evaluate(gate, (0, cow(7)), (1, queued))
    eligible = evaluate(gate, (0, cow(7)), (1, queued))

    assert locking.target_states == {0: "locking", 1: "outside_zone"}
    assert eligible.target_states == {0: "eligible", 1: "outside_zone"}
    assert eligible.tainted_track_ids == ()


def test_zone_taints_multiple_cows_and_stays_blocked_until_clear() -> None:
    gate = ControlledZoneGate(policy())
    evaluate(gate, (0, cow(7)))
    evaluate(gate, (0, cow(7)))

    multiple = evaluate(
        gate,
        (0, cow(7)),
        (1, cow(8, box=(40, 20, 80, 60))),
    )
    still_blocked = evaluate(gate, (0, cow(7)))
    evaluate(gate)
    closed = evaluate(gate)

    assert multiple.target_states == {0: "switch_risk", 1: "switch_risk"}
    assert multiple.tainted_track_ids == (7, 8)
    assert still_blocked.target_states == {0: "switch_risk"}
    assert closed.closed_visit is not None
    assert closed.closed_visit.switch_risk


def test_zone_taints_track_replacement_before_clearance() -> None:
    gate = ControlledZoneGate(policy())
    evaluate(gate, (0, cow(7)))
    evaluate(gate, (0, cow(7)))

    replacement = evaluate(gate, (0, cow(8)))

    assert replacement.target_states == {0: "switch_risk"}
    assert replacement.active_track_id == 7
    assert replacement.tainted_track_ids == (7, 8)


def test_zone_treats_duplicate_track_id_as_ambiguous_even_outside_roi() -> None:
    gate = ControlledZoneGate(policy())

    duplicate = evaluate(
        gate,
        (0, cow(7)),
        (1, cow(7, box=(0, 0, 10, 10))),
    )

    assert duplicate.target_states == {
        0: "switch_risk",
        1: "outside_zone",
    }
    assert duplicate.tainted_track_ids == (7,)


def test_zone_allows_tracker_warmup_before_locking_one_target() -> None:
    gate = ControlledZoneGate(policy())

    untracked = evaluate(gate, (0, cow(None)))
    locking = evaluate(gate, (0, cow(7)))
    eligible = evaluate(gate, (0, cow(7)))

    assert untracked.target_states == {0: "locking"}
    assert locking.target_states == {0: "locking"}
    assert eligible.target_states == {0: "eligible"}
    assert eligible.visit_number == 1


def test_sources_have_independent_visit_counters() -> None:
    gate = ControlledZoneGate(policy())

    first_a = gate.evaluate(
        "camera-a",
        ((0, cow(7)),),
        frame_width=100,
        frame_height=100,
    )
    first_b = gate.evaluate(
        "camera-b",
        ((0, cow(7)),),
        frame_width=100,
        frame_height=100,
    )
    second_a = gate.evaluate(
        "camera-a",
        ((0, cow(7)),),
        frame_width=100,
        frame_height=100,
    )
    second_b = gate.evaluate(
        "camera-b",
        ((0, cow(7)),),
        frame_width=100,
        frame_height=100,
    )

    assert first_a.target_states == {0: "locking"}
    assert first_b.target_states == {0: "locking"}
    assert second_a.visit_number == 1
    assert second_b.visit_number == 1
