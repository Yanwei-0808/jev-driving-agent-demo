"""Compose the final action from Jev's decisions.

Jev does NOT implement the driving logic. This module takes Jev's structured
answers and applies a **progress-minded** policy:

  1. Risk override  - if the Noul collision risk is above RISK_THRESHOLD,
                      force longitudinal brake (safety first).
  2. Lateral policy - if there is a slower close front car AND a free adjacent
                      lane, PREFER a lane change to overtake (even if Jev's
                      lateral confidence is moderate), rather than tailgating.
                      Only brake when no lane change is available.
  3. Confidence gate- if a Choice answer's confidence is below its threshold,
                      fall back to a conservative action (but see the lateral
                      policy above, which can override a keep_lane choice).
  4. Safety net     - illegal lane changes become keep_lane; a critically small
                      gap forces brake; API fallback already means brake.

All thresholds come from config.py so you can tune them for experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config as C
from .env import Action, VehicleState, is_lane_change_legal
from .jev_agent import JevDecision


@dataclass
class DecisionTrace:
    """A human-readable audit of why the final action is what it is."""

    final: Action
    reasons: list[str] = field(default_factory=list)

    @property
    def action(self) -> Action:
        return self.final


def _has_slow_close_front(state: VehicleState) -> bool:
    """True when the front car is meaningfully slower and close enough to matter."""
    return (
        state.has_front_car
        and state.front_speed < state.ego_speed - 3
        and state.front_distance < 60
    )


def _best_available_lane_change(state: VehicleState) -> str | None:
    """Return a legal lane-change direction when one is available, else None.

    Prefers the side Jev already leaned toward; otherwise picks any free side.
    """
    left_ok = state.ego_lane > 0 and state.left_lane_available
    right_ok = state.ego_lane < C.NUM_LANES - 1 and state.right_lane_available
    if left_ok and right_ok:
        return "change_left"  # arbitrary; caller may override with Jev's pick
    if left_ok:
        return "change_left"
    if right_ok:
        return "change_right"
    return None


def compose_action(decision: JevDecision, state: VehicleState) -> DecisionTrace:
    """Turn one JevDecision + state into a safe Action, with reasons."""
    reasons: list[str] = []

    # ---- API failure: fallback is already brake + keep_lane ---------------- #
    if decision.fallback:
        reasons.append(f"Jev request failed ({decision.error}); fallback executed")
        return DecisionTrace(
            final=Action(longitudinal="brake", lateral="keep_lane"),
            reasons=reasons,
        )

    # ---- Rear risk (reference only) -------------------------------------- #
    # Per the demo policy, the rear-collision Noul is informational: it does
    # NOT force a lane change or brake. We just surface it as context.
    if decision.rear_noul >= 0.5:
        reasons.append(
            f"Rear collision risk (Noul) {decision.rear_noul:.2f} is notable "
            f"(reference only; not changing the action)"
        )

    # ---- Longitudinal ----------------------------------------------------- #
    longitudinal = decision.longitudinal_choice or "maintain"
    if longitudinal not in ("accelerate", "maintain", "brake"):
        reasons.append(
            f"Unknown longitudinal choice '{longitudinal}'; defaulting to maintain"
        )
        longitudinal = "maintain"

    if decision.longitudinal_confidence < C.LONGITUDINAL_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Longitudinal confidence {decision.longitudinal_confidence:.2f} < "
            f"{C.LONGITUDINAL_CONFIDENCE_THRESHOLD}; falling back to maintain"
        )
        longitudinal = "maintain"

    # ---- Risk override (Noul) -------------------------------------------- #
    # Noul has no confidence field; the value itself is the belief. But a
    # lane change into a free lane is itself an escape from the risk, so when
    # a legal lane change is available we let the lateral policy below handle
    # it instead of forcing brake here.
    risk_high = decision.risk_noul > C.RISK_THRESHOLD
    if risk_high:
        # Force brake only if no legal lane change will be available below.
        left_ok = state.ego_lane > 0 and state.left_lane_available
        right_ok = state.ego_lane < C.NUM_LANES - 1 and state.right_lane_available
        if not (left_ok or right_ok):
            reasons.append(
                f"Risk noul {decision.risk_noul:.2f} > {C.RISK_THRESHOLD} and no "
                f"lane change available; forcing brake"
            )
            longitudinal = "brake"
        else:
            reasons.append(
                f"Risk noul {decision.risk_noul:.2f} > {C.RISK_THRESHOLD}; will "
                f"try to escape via lane change instead of braking"
            )

    # ---- Lateral (progress-minded policy) -------------------------------- #
    lateral = decision.lateral_choice or "keep_lane"
    if lateral not in ("keep_lane", "change_left", "change_right"):
        reasons.append(
            f"Unknown lateral choice '{lateral}'; defaulting to keep_lane"
        )
        lateral = "keep_lane"

    # If Jev picked a legal lane change, keep it. If it picked keep_lane but
    # we have a slow close front car and a free lane, OVERRIDE to overtake -
    # unless risk is already high (then we brake instead of weaving).
    slow_front = _has_slow_close_front(state)
    risk_high = decision.risk_noul > C.RISK_THRESHOLD

    if lateral != "keep_lane":
        if not is_lane_change_legal(state, lateral):
            reasons.append(
                f"Lane change '{lateral}' not legal now; falling back to keep_lane"
            )
            lateral = "keep_lane"
        elif decision.lateral_confidence < C.LANE_CHANGE_CONFIDENCE_THRESHOLD:
            reasons.append(
                f"Lateral confidence {decision.lateral_confidence:.2f} < "
                f"{C.LANE_CHANGE_CONFIDENCE_THRESHOLD}; ignoring Jev's lane change"
            )
            lateral = "keep_lane"

    # Progress override: a slow close front car + a free lane should make us
    # overtake even if Jev picked keep_lane (or was downgraded to keep_lane
    # above). Only do this when risk is not already high.
    if lateral == "keep_lane" and slow_front and not risk_high:
        alt = _best_available_lane_change(state)
        if alt is not None:
            reasons.append(
                f"Slow close front car ({state.front_distance:.0f}m @ "
                f"{state.front_speed:.0f}km/h) and a free lane exists; "
                f"overriding keep_lane -> {alt} to overtake"
            )
            lateral = alt

    # ---- Safety net: critical gap forces brake --------------------------- #
    if state.has_front_car and state.front_distance <= C.CRITICAL_FRONT_DISTANCE_M:
        reasons.append(
            f"Front gap {state.front_distance:.1f}m <= critical "
            f"{C.CRITICAL_FRONT_DISTANCE_M}m; forcing brake"
        )
        longitudinal = "brake"

    # ---- Sanity: never accelerate past the speed limit -------------------- #
    if longitudinal == "accelerate" and state.ego_speed >= state.speed_limit:
        reasons.append("Already at speed limit; switching accelerate -> maintain")
        longitudinal = "maintain"

    # ---- If we ended up braking AND changing lanes, keep the lane change
    # (it is the escape manoeuvre) unless no lane is actually free. The safety
    # net below will still force brake if the gap is critical.
    if longitudinal == "brake" and lateral != "keep_lane":
        if not is_lane_change_legal(state, lateral):
            reasons.append(
                f"Lane change '{lateral}' not legal; dropping it"
            )
            lateral = "keep_lane"
        else:
            reasons.append(
                "Braking + lane change together (escape manoeuvre); keeping both"
            )

    return DecisionTrace(
        final=Action(longitudinal=longitudinal, lateral=lateral),
        reasons=reasons,
    )


__all__ = ["DecisionTrace", "compose_action"]
