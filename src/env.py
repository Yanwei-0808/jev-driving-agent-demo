"""Toy 2D driving environment.

A deliberately simple environment: one ego vehicle, an optional front car, and
two lanes. It is NOT a physics simulator and NOT an autonomous-driving system.
Its only job is to give the Jev decision model a small, observable state to
reason about, and to update that state after each action.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from . import config as C


@dataclass
class VehicleState:
    """The complete, structured state fed to Jev each step.

    Units are explicit so the model (and the reader) knows what the numbers mean.
    """

    ego_speed: float          # km/h
    ego_lane: int             # 0 .. NUM_LANES-1
    front_distance: float     # metres to the front car (large if no front car)
    front_speed: float        # km/h of the front car (0 if no front car)
    has_front_car: bool       # whether a front car actually exists
    left_lane_available: bool # left lane exists AND appears free
    right_lane_available: bool
    speed_limit: float        # km/h
    lane_offset: float = 0.0  # metres from lane centre (cosmetic, for rendering)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jev_state(self) -> dict:
        """The object sent to Jev as ``state``.

        Keys carry units in their names so the model has unambiguous context.
        """
        d = self.to_dict()
        d["ego_speed_kmh"] = d.pop("ego_speed")
        d["front_speed_kmh"] = d.pop("front_speed")
        d["front_distance_m"] = d.pop("front_distance")
        d["speed_limit_kmh"] = d.pop("speed_limit")
        d["lane_offset_m"] = d.pop("lane_offset")
        d["num_lanes"] = C.NUM_LANES
        return d


@dataclass
class Action:
    """A combined longitudinal + lateral action."""

    longitudinal: str  # accelerate | maintain | brake
    lateral: str       # keep_lane | change_left | change_right

    def __str__(self) -> str:
        lon = self.longitudinal.upper()
        lat = self.lateral.upper()
        if self.lateral == "keep_lane":
            return lon
        return f"{lon} + {lat}"


# --------------------------------------------------------------------------- #
# Validation / safety helpers
# --------------------------------------------------------------------------- #
def validate(state: VehicleState) -> list[str]:
    """Return a list of basic constraint violations (empty == OK).

    These are sanity checks for *generated* states, not a claim of physical
    realism.
    """
    problems: list[str] = []
    if state.ego_speed < 0:
        problems.append("ego_speed < 0")
    if state.front_distance <= 0:
        problems.append("front_distance <= 0")
    if state.speed_limit <= 0:
        problems.append("speed_limit <= 0")
    if state.ego_lane < 0 or state.ego_lane >= C.NUM_LANES:
        problems.append(f"ego_lane {state.ego_lane} out of range")
    if state.ego_lane == 0 and state.left_lane_available:
        problems.append("left_lane_available but ego already in leftmost lane")
    if state.ego_lane == C.NUM_LANES - 1 and state.right_lane_available:
        problems.append("right_lane_available but ego already in rightmost lane")
    return problems


def is_lane_change_legal(state: VehicleState, lateral: str) -> bool:
    """Whether a lateral action is physically possible right now."""
    if lateral == "change_left":
        return state.ego_lane > 0 and state.left_lane_available
    if lateral == "change_right":
        return state.ego_lane < C.NUM_LANES - 1 and state.right_lane_available
    return lateral == "keep_lane"


# --------------------------------------------------------------------------- #
# Environment step
# --------------------------------------------------------------------------- #
def step(state: VehicleState, action: Action) -> VehicleState:
    """Apply ``action`` to ``state`` and return the next state.

    Rules are intentionally trivial (see README): fixed speed deltas, lane +/- 1,
    and a relative-speed update for front_distance. Results are clamped to keep
    the state sane.
    """
    speed = state.ego_speed
    lane = state.ego_lane

    # Longitudinal.
    if action.longitudinal == "accelerate":
        speed += C.ACCEL_DELTA_KMH
    elif action.longitudinal == "brake":
        speed -= C.BRAKE_DELTA_KMH
    # maintain -> no change

    # Lateral (only if legal; illegal changes are silently ignored so the loop
    # never drives the car off the road).
    if action.lateral == "change_left" and is_lane_change_legal(state, "change_left"):
        lane -= 1
    elif action.lateral == "change_right" and is_lane_change_legal(state, "change_right"):
        lane += 1

    # Clamp speed.
    speed = max(C.MIN_SPEED_KMH, min(speed, state.speed_limit))

    # Update front distance from relative speed (km/h -> m/s).
    next_state = VehicleState(
        ego_speed=speed,
        ego_lane=lane,
        front_distance=state.front_distance,
        front_speed=state.front_speed,
        has_front_car=state.has_front_car,
        left_lane_available=state.left_lane_available,
        right_lane_available=state.right_lane_available,
        speed_limit=state.speed_limit,
        lane_offset=state.lane_offset,
    )
    _update_front_distance(next_state)
    return next_state


def _update_front_distance(state: VehicleState) -> None:
    """Move front_distance by the ego/front relative speed over one DT_S step."""
    if not state.has_front_car:
        # No front car: keep a large, harmless distance.
        state.front_distance = max(state.front_distance, 200.0)
        return
    rel_speed_kmh = state.front_speed - state.ego_speed
    rel_speed_ms = rel_speed_kmh / 3.6
    state.front_distance += rel_speed_ms * C.DT_S
    # Distance can collapse toward 0 but never go negative.
    state.front_distance = max(state.front_distance, 0.0)


# --------------------------------------------------------------------------- #
# State generation
# --------------------------------------------------------------------------- #
def random_state(rng: random.Random | None = None) -> VehicleState:
    """Generate a plausible random state with the constraints from the spec."""
    rng = rng or random.Random()
    speed_limit = rng.randint(40, 120)
    ego_lane = rng.randint(0, C.NUM_LANES - 1)
    ego_speed = rng.randint(20, speed_limit)

    has_front_car = rng.random() < 0.7
    if has_front_car:
        front_distance = round(rng.uniform(6, 80), 1)
        front_speed = rng.randint(0, speed_limit)
    else:
        front_distance = 999.0
        front_speed = 0

    left_available = ego_lane > 0 and rng.random() < 0.6
    right_available = ego_lane < C.NUM_LANES - 1 and rng.random() < 0.6

    state = VehicleState(
        ego_speed=ego_speed,
        ego_lane=ego_lane,
        front_distance=front_distance,
        front_speed=front_speed,
        has_front_car=has_front_car,
        left_lane_available=left_available,
        right_lane_available=right_available,
        speed_limit=speed_limit,
        lane_offset=round(rng.uniform(-0.3, 0.3), 2),
    )
    # Guarantee constraints hold.
    problems = validate(state)
    if problems:
        raise RuntimeError(f"generated invalid state: {problems}")
    return state


# --------------------------------------------------------------------------- #
# Preset scenarios (Mode A manual testing)
# --------------------------------------------------------------------------- #
def preset_scenarios() -> list[tuple[str, str, VehicleState]]:
    """Five representative scenarios. (name, intent, state).

    The ``intent`` is documentation only; we record what Jev actually outputs,
    we do NOT adjust the scenario to make the output look correct.
    """
    return [
        (
            "Front far, similar speed",
            "Should keep or mildly accelerate",
            VehicleState(72, 1, 60, 70, True, True, False, 80),
        ),
        (
            "Front clearly slower",
            "Lean toward decelerate or change lane",
            VehicleState(70, 1, 30, 45, True, True, False, 80),
        ),
        (
            "Front very close",
            "High risk, lean toward brake",
            VehicleState(72, 1, 10, 55, True, False, False, 80),
        ),
        (
            "Front slow, left lane free",
            "Can test change_left",
            VehicleState(68, 1, 25, 40, True, True, False, 80),
        ),
        (
            "Front slow, no lane change possible",
            "Should brake / keep_lane",
            VehicleState(70, 0, 22, 40, True, False, True, 80),
        ),
        (
            "Slow & comfortable cruise (慢速舒适)",
            "Low speed, plenty of space; expect keep or mild accelerate",
            VehicleState(45, 1, 120, 55, True, True, False, 80),
        ),
        (
            "Rushing, late (快速赶时间)",
            "Fast ego closing on a slow car; expect change_right or brake",
            VehicleState(78, 0, 25, 50, True, False, True, 80),
        ),
    ]


def state_from_text(text: str) -> VehicleState:
    """Parse a manual state block like ``ego_speed = 72`` (one per line)."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()

    def _get(key: str, cast, default):
        return cast(values[key]) if key in values else default

    return VehicleState(
        ego_speed=float(_get("ego_speed", float, 0)),
        ego_lane=int(_get("ego_lane", int, 0)),
        front_distance=float(_get("front_distance", float, 999.0)),
        front_speed=float(_get("front_speed", float, 0)),
        has_front_car=bool(_get("has_front_car", lambda v: v.lower() == "true", True)),
        left_lane_available=bool(
            _get("left_lane_available", lambda v: v.lower() == "true", False)
        ),
        right_lane_available=bool(
            _get("right_lane_available", lambda v: v.lower() == "true", False)
        ),
        speed_limit=float(_get("speed_limit", float, 80)),
        lane_offset=float(_get("lane_offset", float, 0.0)),
    )


__all__ = [
    "VehicleState",
    "Action",
    "validate",
    "is_lane_change_legal",
    "step",
    "random_state",
    "preset_scenarios",
    "state_from_text",
]
