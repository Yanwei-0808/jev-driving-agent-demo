"""The Jev decision layer.

This is the ONLY module that talks to the official TypeSafe (Jev) API. Everything
here mirrors the current docs (https://docs.typesafe.ai):

  * Endpoint  : POST https://api.typesafe.ai/v1/systemone
  * Python SDK: ``typesafe-sdk`` -> ``from typesafe_sdk import ...``
  * Auth      : ``TYPESAFE_API_KEY`` env var (read automatically by the SDK)
  * Primitives: Noul (yes/no prob, no confidence), Choice (choice + probs +
                confidence), Score (score + probs + legend + confidence)
  * Multiple questions are sent in ONE request and evaluated in parallel.

Design of the Atomic Questions for this driving task (all in one request):

  collision_risk  -> Noul   : "is there a high collision risk?" -> 0..1
  risk_level      -> Score  : ordered rubric safe/marginal/high/critical
  longitudinal    -> Choice : accelerate / maintain / brake
  lateral         -> Choice : keep_lane / change_left / change_right
                             (options filtered by lane availability)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from . import config as C
from .env import VehicleState

# Official SDK import. If this fails, the package is not installed.
from typesafe_sdk import (  # noqa: E402
    Choice,
    Noul,
    Score,
    TypeSafeClient,
)
from typesafe_sdk import (  # noqa: E402
    TypeSafeAPIError,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafeRateLimitError,
)


# Longitudinal / lateral option universes (lateral is filtered per-state).
LONGITUDINAL_OPTIONS = ["accelerate", "maintain", "brake"]
LATERAL_OPTIONS_ALL = ["keep_lane", "change_left", "change_right"]
RISK_LEVELS = ["safe", "marginal", "high", "critical"]


@dataclass
class JevDecision:
    """Structured result parsed from one Jev request.

    Noul has no ``confidence`` field by design; ``risk_noul`` is the belief
    itself. Choice/Score answers carry ``confidence`` derived from the
    probability distribution.
    """

    risk_noul: float                                   # 0..1
    rear_noul: float = 0.0                              # 0..1 (rear collision risk)
    risk_level_score: float | None = None              # 0..len(levels)-1
    risk_level_confidence: float | None = None
    risk_level_probabilities: dict[int, float] = field(default_factory=dict)
    risk_level_legend: dict[int, str] = field(default_factory=dict)

    longitudinal_choice: str = ""
    longitudinal_confidence: float = 0.0
    longitudinal_probabilities: dict[str, float] = field(default_factory=dict)

    lateral_choice: str = ""
    lateral_confidence: float = 0.0
    lateral_probabilities: dict[str, float] = field(default_factory=dict)

    latency_ms: float = 0.0
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None

    # True when the API call failed and we used a fallback decision.
    fallback: bool = False
    error: str | None = None
    raw_answers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "risk_noul": self.risk_noul,
            "rear_noul": self.rear_noul,
            "risk_level_score": self.risk_level_score,
            "risk_level_confidence": self.risk_level_confidence,
            "risk_level_probabilities": dict(self.risk_level_probabilities),
            "risk_level_legend": dict(self.risk_level_legend),
            "longitudinal": {
                "choice": self.longitudinal_choice,
                "confidence": self.longitudinal_confidence,
                "probabilities": dict(self.longitudinal_probabilities),
            },
            "lateral": {
                "choice": self.lateral_choice,
                "confidence": self.lateral_confidence,
                "probabilities": dict(self.lateral_probabilities),
            },
            "latency_ms": self.latency_ms,
            "model": self.model,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "fallback": self.fallback,
            "error": self.error,
            "raw_answers": self.raw_answers,
        }


def _fallback_decision(error: str, latency_ms: float) -> JevDecision:
    """Conservative decision used when the API call fails.

    Per spec: on API failure the agent must NOT execute a dangerous action, so
    we default to brake + keep_lane and flag it as a fallback.
    """
    return JevDecision(
        risk_noul=1.0,  # assume worst case
        rear_noul=1.0,  # assume worst case
        risk_level_score=None,
        risk_level_confidence=None,
        longitudinal_choice="brake",
        longitudinal_confidence=0.0,
        lateral_choice="keep_lane",
        lateral_confidence=0.0,
        latency_ms=latency_ms,
        model="",
        fallback=True,
        error=error,
    )


def _build_longitudinal_question(state: VehicleState) -> Choice:
    target = state.target_speed_kmh or state.speed_limit
    urgency_hint = {
        "rushing": "The driver is in a hurry (urgency=rushing): more willing to accelerate and overtake slow traffic to reach the target speed.",
        "relaxed": "The driver prefers a calm pace (urgency=relaxed): avoid needless acceleration; keep speed steady when safe.",
    }.get(state.urgency, "The driver wants steady progress (urgency=normal).")
    return Choice(
        instructions=(
            f"The ego driver targets {target:.0f} km/h (hard cap = speed limit "
            f"{state.speed_limit:.0f} km/h). {urgency_hint} Given the gap to the "
            "front car, relative speeds and collision risk, what longitudinal "
            "action should it take next toward the target speed? Prefer "
            "accelerate whenever it is safe and below the target/limit. "
            "Choose exactly one."
        ),
        criteria={
            "accelerate": "Safe to speed up toward the target; room ahead and no closing threat.",
            "maintain": "Hold current speed; gap is adequate but not enough to safely accelerate.",
            "brake": "Reduce speed now; gap is small, front car is slower, or risk is high.",
        },
    )


def _build_lateral_question(state: VehicleState) -> Choice:
    """Lateral Choice with options dynamically filtered by lane availability.

    The question is framed so the model prefers changing into a free adjacent
    lane to overtake a slower front car, instead of tailgating. If a lane
    change is not legal we OMIT that option entirely, so the model can never
    pick an impossible action.
    """
    has_slow_front = (
        state.has_front_car
        and state.front_speed < state.ego_speed - 3
        and state.front_distance < 60
    )
    criteria: dict[str, str | None] = {
        "keep_lane": "Stay in the current lane (used when no lane change is available or safe).",
    }
    instructions = (
        "The ego vehicle wants to keep making progress. Given the ego state "
        "and which lanes are available, what lateral action should it take "
        "next? Only the legally available options are listed. "
    )
    if state.ego_lane > 0 and state.left_lane_available:
        criteria["change_left"] = (
            "Move one lane to the left to overtake a slower front car or to "
            "regain speed; the left lane is free ahead."
        )
    if state.ego_lane < C.NUM_LANES - 1 and state.right_lane_available:
        criteria["change_right"] = (
            "Move one lane to the right to overtake a slower front car or to "
            "regain speed; the right lane is free ahead."
        )
    if has_slow_front:
        instructions += (
            "There is a slower car close ahead in the current lane; prefer "
            "changing into a free adjacent lane to overtake it rather than "
            "tailgating."
        )
    else:
        instructions += (
            "The current lane is clear; keep_lane is preferred unless "
            "overtaking clearly helps."
        )
    instructions += " Choose exactly one."
    return Choice(instructions=instructions, criteria=criteria)


def _build_risk_noul(state: VehicleState) -> Noul:
    return Noul(
        instructions=(
            "Given the ego vehicle state, is there a high risk of a collision "
            "in the near term? Consider the gap to the front car and the "
            "relative closing speed."
        ),
        criteria={
            "true": "Gap is small or closing fast; a collision is plausible.",
            "false": "Gap is ample and stable; collision is unlikely.",
        },
    )


def _build_rear_risk_noul(state: VehicleState) -> Noul:
    """Rear collision risk: is a faster car approaching from behind?"""
    instructions = (
        "Is there a high risk of a rear-end collision in the near term? "
        "Consider whether a car behind in the ego lane is closing the gap "
        "fast."
    )
    if state.has_rear_car:
        instructions += (
            f" A car is behind at {state.rear_distance:.0f} m doing "
            f"{state.rear_speed:.0f} km/h."
        )
    else:
        instructions += " No car is currently behind in the ego lane."
    return Noul(
        instructions=instructions,
        criteria={
            "true": "A faster car is close behind and closing the gap quickly.",
            "false": "No close fast-approaching car behind; rear risk is low.",
        },
    )


def _build_risk_level_score(state: VehicleState) -> Score:
    return Score(
        instructions=(
            "Rate the current collision-risk level of the ego vehicle along "
            "this ordered rubric, from safe to critical."
        ),
        criteria=RISK_LEVELS,
    )


class JevAgent:
    """Wraps the official TypeSafeClient and turns one state into one decision."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or os.getenv(C.API_KEY_ENV_VAR)
        if not key:
            raise RuntimeError(
                f"No API key found. Set the {C.API_KEY_ENV_VAR} environment "
                f"variable (see .env.example)."
            )
        self._client = TypeSafeClient(
            api_key=key,
            model=model or C.JEV_MODEL,
            timeout=C.JEV_TIMEOUT_S,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "JevAgent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def decide(self, state: VehicleState) -> JevDecision:
        """Build the questions, call Jev once, and parse the structured answer."""
        questions = {
            "collision_risk": _build_risk_noul(state),
            "rear_collision_risk": _build_rear_risk_noul(state),
            "risk_level": _build_risk_level_score(state),
            "longitudinal": _build_longitudinal_question(state),
            "lateral": _build_lateral_question(state),
        }

        t0 = time.perf_counter()
        try:
            response = self._client.system_one(
                state=state.to_jev_state(),
                questions=questions,
            )
        except (
            TypeSafeAPIError,
            TypeSafeAPIConnectionError,
            TypeSafeAPITimeoutError,
            TypeSafeAuthenticationError,
            TypeSafeRateLimitError,
            TypeSafeError,
        ) as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            err = f"{type(exc).__name__}: {exc}"
            return _fallback_decision(err, latency_ms)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse(response, latency_ms)

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _parse(self, response, latency_ms: float) -> JevDecision:
        # The SDK groups answers by type for convenient, typed access.
        nouls = getattr(response, "nouls", {}) or {}
        choices = getattr(response, "choices", {}) or {}
        scores = getattr(response, "scores", {}) or {}

        risk = nouls.get("collision_risk")
        rear = nouls.get("rear_collision_risk")
        level = scores.get("risk_level")
        lon = choices.get("longitudinal")
        lat = choices.get("lateral")

        raw: dict = {}
        try:
            raw = response.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:
            raw = {}

        risk_noul = float(getattr(risk, "noul", 0.0)) if risk else 0.0
        rear_noul = float(getattr(rear, "noul", 0.0)) if rear else 0.0

        return JevDecision(
            risk_noul=risk_noul,
            rear_noul=rear_noul,
            risk_level_score=float(getattr(level, "score", 0.0)) if level else None,
            risk_level_confidence=(
                float(getattr(level, "confidence", 0.0)) if level else None
            ),
            risk_level_probabilities=dict(getattr(level, "probabilities", {})) if level else {},
            risk_level_legend=_legend_to_str(getattr(level, "legend", {})) if level else {},
            longitudinal_choice=getattr(lon, "choice", "") if lon else "",
            longitudinal_confidence=float(getattr(lon, "confidence", 0.0)) if lon else 0.0,
            longitudinal_probabilities=dict(getattr(lon, "probabilities", {})) if lon else {},
            lateral_choice=getattr(lat, "choice", "") if lat else "",
            lateral_confidence=float(getattr(lat, "confidence", 0.0)) if lat else 0.0,
            lateral_probabilities=dict(getattr(lat, "probabilities", {})) if lat else {},
            latency_ms=latency_ms,
            model=getattr(response, "model", "") or "",
            input_tokens=getattr(getattr(response, "usage", None), "input_tokens", None),
            output_tokens=getattr(getattr(response, "usage", None), "output_tokens", None),
            raw_answers=raw,
        )


def _legend_to_str(legend) -> dict:
    """Normalise legend keys (may be int) to strings for JSON logging."""
    return {str(k): v for k, v in dict(legend).items()}


__all__ = [
    "JevAgent",
    "JevDecision",
    "LONGITUDINAL_OPTIONS",
    "LATERAL_OPTIONS_ALL",
    "RISK_LEVELS",
]
