"""Tests for the toy environment (no API key needed)."""

from __future__ import annotations

import random

from src import env


def test_validate_clean_state():
    s = env.VehicleState(72, 1, 18, 55, True, True, False, 80)
    assert env.validate(s) == []


def test_validate_detects_problems():
    s = env.VehicleState(-5, 3, -1, 55, True, True, True, 0)
    problems = env.validate(s)
    assert any("ego_speed" in p for p in problems)
    assert any("speed_limit" in p for p in problems)
    assert any("lane" in p for p in problems)


def test_step_accelerate_increases_speed():
    s = env.VehicleState(50, 0, 60, 50, True, False, True, 80)
    nxt = env.step(s, env.Action("accelerate", "keep_lane"))
    assert nxt.ego_speed == 55
    assert nxt.ego_lane == 0


def test_step_brake_decreases_speed_but_not_negative():
    s = env.VehicleState(5, 0, 60, 50, True, False, True, 80)
    nxt = env.step(s, env.Action("brake", "keep_lane"))
    assert nxt.ego_speed == 0  # clamped, not -5


def test_step_speed_limit_clamp():
    s = env.VehicleState(78, 0, 60, 50, True, False, True, 80)
    nxt = env.step(s, env.Action("accelerate", "keep_lane"))
    assert nxt.ego_speed == 80  # 78+5 clamped to 80


def test_step_change_left_legal():
    s = env.VehicleState(50, 1, 60, 50, True, True, False, 80)
    nxt = env.step(s, env.Action("maintain", "change_left"))
    assert nxt.ego_lane == 0


def test_step_change_left_illegal_ignored():
    # ego in lane 0, left not available -> change_left must be ignored.
    s = env.VehicleState(50, 0, 60, 50, True, False, True, 80)
    nxt = env.step(s, env.Action("maintain", "change_left"))
    assert nxt.ego_lane == 0


def test_step_change_right_illegal_ignored():
    # ego in rightmost lane (NUM_LANES-1), right not available.
    from src import config as C

    s = env.VehicleState(50, C.NUM_LANES - 1, 60, 50, True, False, False, 80)
    nxt = env.step(s, env.Action("maintain", "change_right"))
    assert nxt.ego_lane == C.NUM_LANES - 1


def test_front_distance_updates_with_relative_speed():
    # ego faster than front car -> distance shrinks.
    s = env.VehicleState(72, 0, 30, 36, True, False, True, 80)
    nxt = env.step(s, env.Action("maintain", "keep_lane"))
    # rel = (36 - 72) / 3.6 * 1.0 = -10 m
    assert abs(nxt.front_distance - 20.0) < 1e-6


def test_front_distance_no_front_car_stays_large():
    s = env.VehicleState(50, 0, 999, 0, False, False, True, 80)
    nxt = env.step(s, env.Action("maintain", "keep_lane"))
    assert nxt.front_distance >= 200


def test_random_state_valid():
    rng = random.Random(42)
    for _ in range(100):
        s = env.random_state(rng)
        assert env.validate(s) == []
        assert 0 <= s.ego_lane < 100


def test_random_state_lane_constraints():
    rng = random.Random(7)
    for _ in range(200):
        s = env.random_state(rng)
        if s.ego_lane == 0:
            assert not s.left_lane_available
        from src import config as C

        if s.ego_lane == C.NUM_LANES - 1:
            assert not s.right_lane_available


def test_preset_scenarios_count_and_valid():
    presets = env.preset_scenarios()
    assert len(presets) == 7
    for name, intent, state in presets:
        assert env.validate(state) == [], f"preset '{name}' invalid"


def test_state_from_text_roundtrip():
    text = """
    ego_speed = 72
    ego_lane = 1
    front_distance = 18
    front_speed = 55
    has_front_car = True
    left_lane_available = True
    right_lane_available = False
    speed_limit = 80
    """
    s = env.state_from_text(text)
    assert s.ego_speed == 72
    assert s.ego_lane == 1
    assert s.front_distance == 18
    assert s.has_front_car is True
    assert s.right_lane_available is False


def test_to_jev_state_has_unit_keys():
    s = env.VehicleState(72, 1, 18, 55, True, True, False, 80)
    d = s.to_jev_state()
    assert "ego_speed_kmh" in d
    assert "front_distance_m" in d
    assert "speed_limit_kmh" in d
    assert "num_lanes" in d


def test_decision_compose_risk_override():
    """Phase 4 logic: high noul risk with NO lane change available forces brake
    even if Choice said accelerate."""
    from src.decision import compose_action
    from src.jev_agent import JevDecision

    # No lane available on either side -> brake is forced.
    s = env.VehicleState(72, 1, 18, 55, True, False, False, 80)
    dec = JevDecision(
        risk_noul=0.9,
        risk_level_score=None,
        risk_level_confidence=None,
        longitudinal_choice="accelerate",
        longitudinal_confidence=0.99,
        lateral_choice="keep_lane",
        lateral_confidence=0.99,
    )
    trace = compose_action(dec, s)
    assert trace.final.longitudinal == "brake"


def test_decision_compose_low_confidence_lateral():
    """Low lateral confidence on a change_left -> ignored, but the
    progress-minded policy may still override to change_left because there is
    a slow close front car and the left lane is free."""
    from src.decision import compose_action
    from src.jev_agent import JevDecision

    s = env.VehicleState(68, 1, 25, 40, True, True, False, 80)
    dec = JevDecision(
        risk_noul=0.2,
        risk_level_score=None,
        risk_level_confidence=None,
        longitudinal_choice="maintain",
        longitudinal_confidence=0.9,
        lateral_choice="change_left",
        lateral_confidence=0.3,
    )
    trace = compose_action(dec, s)
    # Slow close front car + free left lane -> override to change_left
    assert trace.final.lateral == "change_left"


def test_decision_compose_progress_override_when_jev_says_keep_lane():
    """Even when Jev picks keep_lane, a slow close front car + free lane
    triggers an override to overtake."""
    from src.decision import compose_action
    from src.jev_agent import JevDecision

    s = env.VehicleState(70, 1, 25, 40, True, True, False, 80)
    dec = JevDecision(
        risk_noul=0.2,
        risk_level_score=None,
        risk_level_confidence=None,
        longitudinal_choice="maintain",
        longitudinal_confidence=0.9,
        lateral_choice="keep_lane",
        lateral_confidence=0.9,
    )
    trace = compose_action(dec, s)
    assert trace.final.lateral == "change_left"
    assert any("overriding keep_lane" in r for r in trace.reasons)


def test_decision_compose_no_override_when_no_lane_available():
    """No free lane -> no override; brake instead (the safety net may force it)."""
    from src.decision import compose_action
    from src.jev_agent import JevDecision

    s = env.VehicleState(70, 0, 22, 40, True, False, False, 80)
    dec = JevDecision(
        risk_noul=0.2,
        risk_level_score=None,
        risk_level_confidence=None,
        longitudinal_choice="maintain",
        longitudinal_confidence=0.9,
        lateral_choice="keep_lane",
        lateral_confidence=0.9,
    )
    trace = compose_action(dec, s)
    assert trace.final.lateral == "keep_lane"


def test_decision_compose_critical_gap_forces_brake():
    from src import config as C
    from src.decision import compose_action
    from src.jev_agent import JevDecision

    s = env.VehicleState(72, 1, C.CRITICAL_FRONT_DISTANCE_M - 1, 55, True, False, False, 80)
    dec = JevDecision(
        risk_noul=0.1,
        risk_level_score=None,
        risk_level_confidence=None,
        longitudinal_choice="accelerate",
        longitudinal_confidence=0.99,
        lateral_choice="keep_lane",
        lateral_confidence=0.99,
    )
    trace = compose_action(dec, s)
    assert trace.final.longitudinal == "brake"


def test_decision_compose_fallback():
    from src.decision import compose_action
    from src.jev_agent import JevDecision

    s = env.VehicleState(72, 1, 18, 55, True, True, False, 80)
    dec = JevDecision(
        risk_noul=1.0,
        risk_level_score=None,
        risk_level_confidence=None,
        fallback=True,
        error="simulated",
    )
    trace = compose_action(dec, s)
    assert trace.final.longitudinal == "brake"
    assert trace.final.lateral == "keep_lane"
