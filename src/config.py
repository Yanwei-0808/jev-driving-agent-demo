"""Central configuration for the Jev Driving Agent.

All tunable parameters live here so you can experiment without touching logic.
Every value can also be overridden with an environment variable of the same name.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Jev / TypeSafe API
# --------------------------------------------------------------------------- #
# The model alias to send. ``jev-latest`` is TypeSafe's flagship model.
JEV_MODEL: str = os.getenv("JEV_MODEL", "jev-latest")
# Per-request timeout in seconds. Jev normally answers in 70-500ms; this leaves
# headroom for slow networks.
JEV_TIMEOUT_S: float = _env_float("JEV_TIMEOUT_S", 15.0)
# Environment variable that the official SDK reads for the API key.
API_KEY_ENV_VAR: str = "TYPESAFE_API_KEY"

# --------------------------------------------------------------------------- #
# Decision thresholds (confidence gating + safety)
# --------------------------------------------------------------------------- #
# Noul risk probability above this -> force longitudinal brake regardless of the
# Choice answer. Noul has no separate confidence field; the value IS the belief.
RISK_THRESHOLD: float = _env_float("RISK_THRESHOLD", 0.8)

# If the chosen lateral action's confidence is below this, fall back to
# keep_lane. Lower = more willing to act on a weak lane-change signal.
LANE_CHANGE_CONFIDENCE_THRESHOLD: float = _env_float(
    "LANE_CHANGE_CONFIDENCE_THRESHOLD", 0.35
)

# If the chosen longitudinal action's confidence is below this, fall back to
# maintain (a conservative default when the model is unsure).
LONGITUDINAL_CONFIDENCE_THRESHOLD: float = _env_float(
    "LONGITUDINAL_CONFIDENCE_THRESHOLD", 0.5
)

# --------------------------------------------------------------------------- #
# Driving mode (driver intent) -> target speed + urgency label
# --------------------------------------------------------------------------- #
# A selectable driver intent. Sets a target speed (km/h, the speed the driver
# aims for, at or below the limit) and an urgency label sent to Jev so it can
# weigh progress vs caution. Keys are stable ids used by the web UI / CLI.
DRIVING_MODES: dict[str, dict] = {
    "comfort": {"label": "舒适", "target_speed": 45, "urgency": "relaxed"},
    "normal":  {"label": "正常", "target_speed": 60, "urgency": "normal"},
    "rush":    {"label": "赶时间", "target_speed": 80, "urgency": "rushing"},
}
DEFAULT_DRIVING_MODE: str = "normal"

# --------------------------------------------------------------------------- #
# Toy environment physics
# --------------------------------------------------------------------------- #
# Length of one simulation step (seconds). Used to update front_distance from
# the relative speed: delta = (front_speed - ego_speed) / 3.6 * DT_S.
DT_S: float = _env_float("DT_S", 1.0)

# Speed deltas per step (km/h).
ACCEL_DELTA_KMH: float = _env_float("ACCEL_DELTA_KMH", 5.0)
BRAKE_DELTA_KMH: float = _env_float("BRAKE_DELTA_KMH", 10.0)

# Hard limits.
MIN_SPEED_KMH: float = 0.0
MIN_FRONT_DISTANCE_M: float = _env_float("MIN_FRONT_DISTANCE_M", 5.0)
# When ego is within this distance of a real front car, force brake as a safety
# net regardless of what Jev says.
CRITICAL_FRONT_DISTANCE_M: float = _env_float("CRITICAL_FRONT_DISTANCE_M", 8.0)

# Road geometry.
NUM_LANES: int = _env_int("NUM_LANES", 3)  # lane ids are 0 .. NUM_LANES-1

# --------------------------------------------------------------------------- #
# Simulation loop
# --------------------------------------------------------------------------- #
DEFAULT_CONTINUOUS_STEPS: int = _env_int("DEFAULT_CONTINUOUS_STEPS", 30)
DEFAULT_SINGLE_STEPS: int = _env_int("DEFAULT_SINGLE_STEPS", 1)
LOOP_SLEEP_S: float = _env_float("LOOP_SLEEP_S", 1.0)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_DIR: str = os.getenv("LOG_DIR", "logs")


__all__ = [
    "JEV_MODEL",
    "JEV_TIMEOUT_S",
    "API_KEY_ENV_VAR",
    "RISK_THRESHOLD",
    "LANE_CHANGE_CONFIDENCE_THRESHOLD",
    "LONGITUDINAL_CONFIDENCE_THRESHOLD",
    "DRIVING_MODES",
    "DEFAULT_DRIVING_MODE",
    "DT_S",
    "ACCEL_DELTA_KMH",
    "BRAKE_DELTA_KMH",
    "MIN_SPEED_KMH",
    "MIN_FRONT_DISTANCE_M",
    "CRITICAL_FRONT_DISTANCE_M",
    "NUM_LANES",
    "DEFAULT_CONTINUOUS_STEPS",
    "DEFAULT_SINGLE_STEPS",
    "LOOP_SLEEP_S",
    "LOG_DIR",
]
