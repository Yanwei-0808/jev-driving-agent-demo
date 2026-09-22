"""Minimal terminal dashboard + ASCII road rendering.

Intentionally simple and dependency-free. The dashboard shows the vehicle
state, the structured Jev decisions (with probabilities + confidence), the
executed action, and the latency. The road view is a tiny ASCII sketch.
"""

from __future__ import annotations

import os
import sys

from . import config as C
from .env import Action, VehicleState
from .jev_agent import RISK_LEVELS, JevDecision


BAR_WIDTH = 26


def _bar(label: str, value: float, lo: float, hi: float) -> str:
    """A tiny text bar for a probability/value in [lo, hi]."""
    span = max(hi - lo, 1e-9)
    frac = max(0.0, min(1.0, (value - lo) / span))
    filled = int(round(frac * BAR_WIDTH))
    return f"{label:<14}{'#' * filled}{'.' * (BAR_WIDTH - filled)} {value:.2f}"


def _yes_no(v: bool) -> str:
    return "Yes" if v else "No"


def render_road(state: VehicleState) -> str:
    """A very small two-lane ASCII view with ego + front car."""
    width = 28
    top = [" "] * width
    bot = [" "] * width

    # Place ego car in its lane.
    ego_col = 4
    front_col = ego_col + max(0, min(width - ego_col - 2, int(state.front_distance / 3)))

    if state.ego_lane == 0:
        bot[ego_col] = "E"
    else:
        top[ego_col] = "E"

    if state.has_front_car and 0 <= front_col < width:
        front_lane = state.ego_lane
        if front_lane == 0:
            bot[front_col] = "F"
        else:
            top[front_col] = "F"

    sep = "-" * width
    border = "=" * width
    left_open = " " if state.left_lane_available else "X"
    right_open = " " if state.right_lane_available else "X"

    lines = [
        "+" + border + "+",
        "|" + "".join(top) + "|",
        "|" + sep + "|",
        "|" + "".join(bot) + "|",
        "+" + border + "+",
    ]
    # Legend line.
    lines.append(
        f" E=ego  F=front  L={left_open}  R={right_open}  "
        f"(lanes={C.NUM_LANES}, rows show 2 of them)"
    )
    return "\n".join(lines)


def render_dashboard(
    step_no: int,
    state: VehicleState,
    decision: JevDecision,
    trace,
    executed: Action,
    next_state: VehicleState,
) -> str:
    """Render the full dashboard frame as a string."""
    L = []
    L.append("=" * 60)
    L.append("    Jev Driving Agent  (Toy Environment)")
    L.append("=" * 60)
    L.append(f"Step: {step_no}")
    L.append("")
    L.append("Vehicle State")
    L.append("-" * 60)
    L.append(f"  Speed:            {state.ego_speed:.0f} km/h  (limit {state.speed_limit:.0f})")
    L.append(f"  Lane:             {state.ego_lane}")
    L.append(f"  Front car:        {_yes_no(state.has_front_car)}")
    if state.has_front_car:
        L.append(f"  Front Distance:   {state.front_distance:.1f} m")
        L.append(f"  Front Speed:      {state.front_speed:.0f} km/h")
    else:
        L.append("  Front Distance:   (no front car)")
    L.append(f"  Left Lane Free:   {_yes_no(state.left_lane_available)}")
    L.append(f"  Right Lane Free:  {_yes_no(state.right_lane_available)}")
    L.append("")

    L.append("Jev Decisions")
    L.append("-" * 60)
    if decision.fallback:
        L.append(f"  *** JEV REQUEST FAILED ***  {decision.error}")
        L.append("  fallback executed (brake + keep_lane)")
    else:
        L.append(f"  Model: {decision.model}")
        L.append("  Risk (Noul, 0=no .. 1=yes):")
        L.append("    " + _bar("collision_risk", decision.risk_noul, 0.0, 1.0))
        if decision.risk_level_score is not None:
            level_idx = max(0, min(len(RISK_LEVELS) - 1, int(round(decision.risk_level_score))))
            label = RISK_LEVELS[level_idx]
            L.append(
                f"  Risk level (Score): {decision.risk_level_score:.2f} -> {label}  "
                f"confidence {decision.risk_level_confidence:.2f}"
            )
        L.append("")
        L.append("  Longitudinal (Choice):")
        L.append(f"    -> {decision.longitudinal_choice.upper()}")
        L.append(f"    confidence: {decision.longitudinal_confidence:.2f}")
        L.append("    " + _prob_line(decision.longitudinal_probabilities))
        L.append("")
        L.append("  Lateral (Choice):")
        L.append(f"    -> {decision.lateral_choice.upper()}")
        L.append(f"    confidence: {decision.lateral_confidence:.2f}")
        L.append("    " + _prob_line(decision.lateral_probabilities))
    L.append("")

    L.append("Executed Action")
    L.append("-" * 60)
    L.append(f"  {executed}")
    if trace.reasons:
        L.append("  reasons:")
        for r in trace.reasons:
            L.append(f"    - {r}")
    L.append("")

    L.append("Next State")
    L.append("-" * 60)
    L.append(f"  Speed:          {next_state.ego_speed:.0f} km/h")
    L.append(f"  Lane:           {next_state.ego_lane}")
    if next_state.has_front_car:
        L.append(f"  Front Distance: {next_state.front_distance:.1f} m")
    L.append("")

    L.append(f"Latency: {decision.latency_ms:.1f} ms")
    L.append("")
    L.append(render_road(next_state))
    L.append("=" * 60)
    return "\n".join(L)


def _prob_line(probs: dict) -> str:
    """One-line summary of a probability distribution."""
    if not probs:
        return "(no probabilities)"
    parts = [f"{k}={v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])]
    return "  ".join(parts)


def clear_screen() -> None:
    """Best-effort terminal clear (works on Windows and POSIX).

    No-op when stdout is not a TTY (tests, CI, piped output) so we never clobber
    captured output.
    """
    if not sys.stdout.isatty():
        return
    os.system("cls" if os.name == "nt" else "clear")


__all__ = ["render_dashboard", "render_road", "clear_screen"]
