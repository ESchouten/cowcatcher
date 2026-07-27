from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from aidetector.domain.detections import DetectedObject

ZoneTargetState = Literal[
    "outside_zone",
    "locking",
    "eligible",
    "switch_risk",
]


@dataclass(frozen=True, slots=True)
class ControlledZonePolicy:
    zone_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    minimum_box_inside_ratio: float
    minimum_stable_frames: int
    clear_frames: int

    def __post_init__(self) -> None:
        if not self.zone_id.strip():
            raise ValueError("Controlled identity zone ID is required")
        if not (0 <= self.x1 < self.x2 <= 1 and 0 <= self.y1 < self.y2 <= 1):
            raise ValueError("Controlled identity zone extent is invalid")
        if not 0 < self.minimum_box_inside_ratio <= 1:
            raise ValueError("Controlled identity box-containment ratio is invalid")
        if self.minimum_stable_frames < 1:
            raise ValueError("Controlled identity stable-frame count must be positive")
        if self.clear_frames < 1:
            raise ValueError("Controlled identity clear-frame count must be positive")


@dataclass(frozen=True, slots=True)
class ZoneVisitClosure:
    visit_number: int
    track_id: int
    switch_risk: bool


@dataclass(frozen=True, slots=True)
class ZoneEvaluation:
    target_states: dict[int, ZoneTargetState]
    eligible_index: int | None
    visit_number: int | None
    active_track_id: int | None
    tainted_track_ids: tuple[int, ...]
    closed_visit: ZoneVisitClosure | None


@dataclass(slots=True)
class _SourceZoneState:
    clear_frames_seen: int
    locked_track_id: int | None = None
    stable_frames_seen: int = 0
    visit_number: int = 0
    switch_risk: bool = False
    pending_untracked_occupancy: bool = False


class ControlledZoneGate:
    """Create conservative, source-scoped identity visits from tracked boxes."""

    def __init__(self, policy: ControlledZonePolicy) -> None:
        self.policy = policy
        self._sources: dict[str, _SourceZoneState] = {}

    def evaluate(
        self,
        source: str,
        targets: Sequence[tuple[int, DetectedObject]],
        *,
        frame_width: int,
        frame_height: int,
    ) -> ZoneEvaluation:
        if not source or frame_width < 1 or frame_height < 1:
            raise ValueError(
                "Controlled identity zone requires a source and frame size"
            )
        state = self._sources.setdefault(
            source,
            _SourceZoneState(clear_frames_seen=self.policy.clear_frames),
        )
        target_states: dict[int, ZoneTargetState] = {
            index: "outside_zone" for index, _target in targets
        }
        inside_ratios = {
            index: _box_inside_ratio(
                target,
                self.policy,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            for index, target in targets
        }
        intersecting = [
            (index, target, inside_ratios[index])
            for index, target in targets
            if inside_ratios[index] > 0
        ]
        if not intersecting:
            closed = self._advance_clear_state(state)
            return ZoneEvaluation(
                target_states=target_states,
                eligible_index=None,
                visit_number=None,
                active_track_id=state.locked_track_id,
                tainted_track_ids=(),
                closed_visit=closed,
            )

        was_clear = state.clear_frames_seen >= self.policy.clear_frames
        state.clear_frames_seen = 0
        track_counts = Counter(
            target.track_id for _index, target in targets if target.track_id is not None
        )
        duplicate_ids = {
            track_id
            for track_id, count in track_counts.items()
            if count > 1
            and (
                track_id == state.locked_track_id
                or any(target.track_id == track_id for _, target, _ in intersecting)
            )
        }
        if len(intersecting) != 1 or duplicate_ids:
            state.switch_risk = True
            state.stable_frames_seen = 0
            state.pending_untracked_occupancy = True
            for index, _target, _ratio in intersecting:
                target_states[index] = "switch_risk"
            tainted = {
                *(track_id for track_id in duplicate_ids),
                *(
                    target.track_id
                    for _index, target, _ratio in intersecting
                    if target.track_id is not None
                ),
            }
            if state.locked_track_id is not None:
                tainted.add(state.locked_track_id)
            return ZoneEvaluation(
                target_states=target_states,
                eligible_index=None,
                visit_number=None,
                active_track_id=state.locked_track_id,
                tainted_track_ids=tuple(sorted(tainted)),
                closed_visit=None,
            )

        index, target, inside_ratio = intersecting[0]
        track_id = target.track_id
        if track_id is None:
            state.stable_frames_seen = 0
            state.pending_untracked_occupancy = True
            target_states[index] = "switch_risk" if state.switch_risk else "locking"
            return ZoneEvaluation(
                target_states=target_states,
                eligible_index=None,
                visit_number=None,
                active_track_id=state.locked_track_id,
                tainted_track_ids=(),
                closed_visit=None,
            )

        if state.locked_track_id is None:
            if state.switch_risk:
                target_states[index] = "switch_risk"
                return ZoneEvaluation(
                    target_states=target_states,
                    eligible_index=None,
                    visit_number=None,
                    active_track_id=None,
                    tainted_track_ids=(track_id,),
                    closed_visit=None,
                )
            if not was_clear and not state.pending_untracked_occupancy:
                state.switch_risk = True
                target_states[index] = "switch_risk"
                return ZoneEvaluation(
                    target_states=target_states,
                    eligible_index=None,
                    visit_number=None,
                    active_track_id=None,
                    tainted_track_ids=(track_id,),
                    closed_visit=None,
                )
            state.locked_track_id = track_id
            state.visit_number += 1
            state.pending_untracked_occupancy = False
        elif state.locked_track_id != track_id:
            previous_track_id = state.locked_track_id
            state.switch_risk = True
            state.stable_frames_seen = 0
            target_states[index] = "switch_risk"
            return ZoneEvaluation(
                target_states=target_states,
                eligible_index=None,
                visit_number=None,
                active_track_id=previous_track_id,
                tainted_track_ids=tuple(sorted((previous_track_id, track_id))),
                closed_visit=None,
            )

        if state.switch_risk:
            target_states[index] = "switch_risk"
            return ZoneEvaluation(
                target_states=target_states,
                eligible_index=None,
                visit_number=None,
                active_track_id=state.locked_track_id,
                tainted_track_ids=(track_id,),
                closed_visit=None,
            )
        if inside_ratio < self.policy.minimum_box_inside_ratio:
            state.stable_frames_seen = 0
            target_states[index] = "locking"
            return ZoneEvaluation(
                target_states=target_states,
                eligible_index=None,
                visit_number=None,
                active_track_id=state.locked_track_id,
                tainted_track_ids=(),
                closed_visit=None,
            )

        state.stable_frames_seen += 1
        if state.stable_frames_seen < self.policy.minimum_stable_frames:
            target_states[index] = "locking"
            return ZoneEvaluation(
                target_states=target_states,
                eligible_index=None,
                visit_number=None,
                active_track_id=state.locked_track_id,
                tainted_track_ids=(),
                closed_visit=None,
            )
        target_states[index] = "eligible"
        return ZoneEvaluation(
            target_states=target_states,
            eligible_index=index,
            visit_number=state.visit_number,
            active_track_id=state.locked_track_id,
            tainted_track_ids=(),
            closed_visit=None,
        )

    def _advance_clear_state(
        self,
        state: _SourceZoneState,
    ) -> ZoneVisitClosure | None:
        state.stable_frames_seen = 0
        state.clear_frames_seen = min(
            state.clear_frames_seen + 1,
            self.policy.clear_frames,
        )
        if state.clear_frames_seen < self.policy.clear_frames:
            return None
        closed = (
            ZoneVisitClosure(
                visit_number=state.visit_number,
                track_id=state.locked_track_id,
                switch_risk=state.switch_risk,
            )
            if state.locked_track_id is not None
            else None
        )
        state.locked_track_id = None
        state.stable_frames_seen = 0
        state.switch_risk = False
        state.pending_untracked_occupancy = False
        return closed


def _box_inside_ratio(
    target: DetectedObject,
    policy: ControlledZonePolicy,
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    box_width = target.x2 - target.x1
    box_height = target.y2 - target.y1
    if box_width <= 0 or box_height <= 0:
        return 0.0
    zone_x1 = policy.x1 * frame_width
    zone_y1 = policy.y1 * frame_height
    zone_x2 = policy.x2 * frame_width
    zone_y2 = policy.y2 * frame_height
    intersection_width = max(
        0.0,
        min(float(target.x2), zone_x2) - max(float(target.x1), zone_x1),
    )
    intersection_height = max(
        0.0,
        min(float(target.y2), zone_y2) - max(float(target.y1), zone_y1),
    )
    return intersection_width * intersection_height / (box_width * box_height)
