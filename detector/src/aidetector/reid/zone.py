from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from aidetector.domain.detections import DetectedObject

MINIMUM_BOX_INSIDE_RATIO = 0.9
STABLE_FRAMES = 2
CLEAR_FRAMES = 2
ZoneState = Literal["outside_zone", "locking", "eligible", "switch_risk"]


@dataclass(frozen=True, slots=True)
class ZoneVisit:
    track_id: int
    switch_risk: bool


@dataclass(frozen=True, slots=True)
class ZoneResult:
    target_states: dict[int, ZoneState]
    active_track_id: int | None
    tainted_track_ids: tuple[int, ...] = ()
    closed_visit: ZoneVisit | None = None


@dataclass(slots=True)
class _State:
    clear_frames: int = CLEAR_FRAMES
    track_id: int | None = None
    stable_frames: int = 0
    switch_risk: bool = False
    saw_untracked_target: bool = False


class ZoneGate:
    """Allow identity evidence from one tracked target in a calibrated zone."""

    def __init__(self, margin: float) -> None:
        self.margin = margin
        self._sources: dict[str, _State] = {}

    def evaluate(
        self,
        source: str,
        targets: Sequence[tuple[int, DetectedObject]],
        *,
        frame_width: int,
        frame_height: int,
    ) -> ZoneResult:
        state = self._sources.setdefault(source, _State())
        target_states: dict[int, ZoneState] = {
            index: "outside_zone" for index, _ in targets
        }
        intersecting = [
            (index, target, ratio)
            for index, target in targets
            if (
                ratio := _inside_ratio(
                    target,
                    self.margin,
                    frame_width,
                    frame_height,
                )
            )
            > 0
        ]
        if not intersecting:
            return self._result(state, target_states, closed=self._clear(state))

        was_clear = state.clear_frames >= CLEAR_FRAMES
        state.clear_frames = 0
        counts = Counter(
            target.track_id for _, target in targets if target.track_id is not None
        )
        intersecting_ids = {
            target.track_id
            for _, target, _ in intersecting
            if target.track_id is not None
        }
        duplicate_ids = {
            track_id
            for track_id, count in counts.items()
            if count > 1
            and (track_id == state.track_id or track_id in intersecting_ids)
        }
        if len(intersecting) != 1 or duplicate_ids:
            state.switch_risk = True
            state.stable_frames = 0
            state.saw_untracked_target = True
            for index, _, _ in intersecting:
                target_states[index] = "switch_risk"
            tainted = duplicate_ids | intersecting_ids
            if state.track_id is not None:
                tainted.add(state.track_id)
            return self._result(state, target_states, tainted)

        index, target, inside_ratio = intersecting[0]
        track_id = target.track_id
        if track_id is None:
            state.stable_frames = 0
            state.saw_untracked_target = True
            target_states[index] = "switch_risk" if state.switch_risk else "locking"
            return self._result(state, target_states)

        if state.track_id is None:
            if state.switch_risk or (not was_clear and not state.saw_untracked_target):
                state.switch_risk = True
                target_states[index] = "switch_risk"
                return self._result(state, target_states, {track_id})
            state.track_id = track_id
            state.saw_untracked_target = False
        elif state.track_id != track_id:
            previous = state.track_id
            state.switch_risk = True
            state.stable_frames = 0
            target_states[index] = "switch_risk"
            return self._result(state, target_states, {previous, track_id})

        if state.switch_risk:
            target_states[index] = "switch_risk"
            return self._result(state, target_states, {track_id})

        if inside_ratio < MINIMUM_BOX_INSIDE_RATIO:
            state.stable_frames = 0
            target_states[index] = "locking"
        else:
            state.stable_frames += 1
            target_states[index] = (
                "eligible" if state.stable_frames >= STABLE_FRAMES else "locking"
            )
        return self._result(state, target_states)

    @staticmethod
    def _result(
        state: _State,
        target_states: dict[int, ZoneState],
        tainted: set[int] | None = None,
        *,
        closed: ZoneVisit | None = None,
    ) -> ZoneResult:
        return ZoneResult(
            target_states=target_states,
            active_track_id=state.track_id,
            tainted_track_ids=tuple(sorted(tainted or ())),
            closed_visit=closed,
        )

    @staticmethod
    def _clear(state: _State) -> ZoneVisit | None:
        state.stable_frames = 0
        state.clear_frames = min(state.clear_frames + 1, CLEAR_FRAMES)
        if state.clear_frames < CLEAR_FRAMES:
            return None
        closed = (
            ZoneVisit(state.track_id, state.switch_risk)
            if state.track_id is not None
            else None
        )
        state.track_id = None
        state.switch_risk = False
        state.saw_untracked_target = False
        return closed


def _inside_ratio(
    target: DetectedObject,
    margin: float,
    frame_width: int,
    frame_height: int,
) -> float:
    width = target.x2 - target.x1
    height = target.y2 - target.y1
    if width <= 0 or height <= 0:
        return 0
    intersection_width = max(
        0,
        min(target.x2, (1 - margin) * frame_width)
        - max(target.x1, margin * frame_width),
    )
    intersection_height = max(
        0,
        min(target.y2, (1 - margin) * frame_height)
        - max(target.y1, margin * frame_height),
    )
    return intersection_width * intersection_height / (width * height)
