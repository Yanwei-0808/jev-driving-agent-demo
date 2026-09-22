"""Offline end-to-end smoke test for the agent loop.

Uses a fake JevAgent so the loop, UI rendering, logging and latency stats can be
exercised WITHOUT a real API key or network.
"""

from __future__ import annotations

import io
import random
from contextlib import redirect_stdout

from src import env
from src.jev_agent import JevDecision
from src.logger import ExperimentLogger, format_stats, latency_stats
from src.simulator import run_steps


class FakeAgent:
    """Mimics JevAgent: context manager + decide(state)->JevDecision."""

    def __init__(self):
        self._n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def decide(self, state: env.VehicleState) -> JevDecision:
        self._n += 1
        # Vary the canned decision a little per step so rendering differs.
        risk = 0.2 if state.front_distance > 30 else 0.9
        return JevDecision(
            risk_noul=risk,
            risk_level_score=1.0,
            risk_level_confidence=0.8,
            risk_level_probabilities={0: 0.1, 1: 0.7, 2: 0.2},
            risk_level_legend={0: "safe", 1: "marginal", 2: "high"},
            longitudinal_choice="brake" if risk > 0.5 else "maintain",
            longitudinal_confidence=0.9,
            longitudinal_probabilities={"accelerate": 0.05, "maintain": 0.9, "brake": 0.05},
            lateral_choice="keep_lane",
            lateral_confidence=0.95,
            lateral_probabilities={"keep_lane": 1.0},
            latency_ms=120.0 + self._n,
            model="jev-fake",
            input_tokens=100,
            output_tokens=10,
        )


def _factory():
    return FakeAgent()


def test_loop_runs_and_logs(tmp_path):
    logger = ExperimentLogger(run_name="offline", log_dir=str(tmp_path))
    log_file = logger.path
    buf = io.StringIO()
    state = env.VehicleState(72, 1, 18, 55, True, True, False, 80)
    with redirect_stdout(buf):
        records = run_steps(
            state,
            num_steps=3,
            agent_factory=_factory,
            logger=logger,
            sleep_s=0.0,
            render=True,
        )
    logger.close()

    assert len(records) == 3
    for i, rec in enumerate(records, 1):
        assert rec["step"] == i
        assert "latency_ms" in rec
        assert "decisions" in rec
        assert "executed_action" in rec
        assert "next_state" in rec
        assert "state" in rec

    # The log file should exist with 3 lines.
    assert log_file.exists()
    text = log_file.read_text(encoding="utf-8")
    assert text.count("\n") == 3

    # Latency stats should compute without error.
    stats = latency_stats(records)
    rendered = format_stats(stats)
    assert "avg" in rendered
    assert stats["all_steps"]["count"] == 3
    assert stats["non_fallback_steps"]["count"] == 3


def test_loop_dashboard_rendered(tmp_path):
    logger = ExperimentLogger(run_name="offline2", log_dir=str(tmp_path))
    buf = io.StringIO()
    state = env.VehicleState(70, 1, 25, 40, True, True, False, 80)
    with redirect_stdout(buf):
        run_steps(
            state,
            num_steps=1,
            agent_factory=_factory,
            logger=logger,
            render=True,
        )
    logger.close()
    out = buf.getvalue()
    assert "Jev Driving Agent" in out
    assert "Latency:" in out
    assert "Longitudinal" in out
    assert "Lateral" in out
    assert "E=ego" in out  # road legend


def test_loop_fallback_path(tmp_path):
    """A fallback decision still produces a safe action and a record."""

    class FailAgent(FakeAgent):
        def decide(self, state):
            from src.jev_agent import _fallback_decision

            return _fallback_decision("TypeSafeAPIError: boom", 50.0)

    logger = ExperimentLogger(run_name="fail", log_dir=str(tmp_path))
    state = env.VehicleState(72, 1, 18, 55, True, True, False, 80)
    records = run_steps(
        state,
        num_steps=1,
        agent_factory=lambda: FailAgent(),
        logger=logger,
        render=False,
    )
    logger.close()
    rec = records[0]
    assert rec["decisions"]["fallback"] is True
    assert rec["executed_action"]["longitudinal"] == "brake"
    assert rec["executed_action"]["lateral"] == "keep_lane"
    assert any("fallback" in r for r in rec["reasons"])
