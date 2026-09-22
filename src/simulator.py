"""The real-time agent loop and its three run modes.

Loop (one step):
    1. observe current state
    2. build Jev request (one state, multiple typed questions)
    3. call Jev
    4. parse structured decisions
    5. record latency
    6. apply confidence / safety rules (decision.py)
    7. execute action
    8. update environment
    9. render state
    10. continue

Modes:
  - manual      : pick a preset (or type a state), run a few steps
  - random      : random initial state, run a few steps
  - continuous  : loop for N steps with a sleep between them
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict
from typing import Callable

from . import config as C
from .decision import compose_action
from .env import VehicleState, preset_scenarios, random_state, step
from .jev_agent import JevAgent, JevDecision
from .logger import ExperimentLogger, format_stats, latency_stats
from .ui import clear_screen, render_dashboard


# A factory that returns a fresh JevAgent, so the simulator does not import the
# SDK at module load time and tests can swap in a fake.
AgentFactory = Callable[[], "JevAgent"]


def _default_agent_factory() -> JevAgent:
    return JevAgent()


def _record(
    logger: ExperimentLogger,
    step_no: int,
    state: VehicleState,
    decision: JevDecision,
    trace,
    executed,
    next_state: VehicleState,
) -> dict:
    record = {
        "step": step_no,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": state.to_dict(),
        "questions": ["collision_risk", "risk_level", "longitudinal", "lateral"],
        "decisions": decision.to_dict(),
        "latency_ms": decision.latency_ms,
        "executed_action": {
            "longitudinal": executed.longitudinal,
            "lateral": executed.lateral,
        },
        "reasons": list(trace.reasons),
        "next_state": next_state.to_dict(),
    }
    logger.log(record)
    return record


def run_steps(
    initial: VehicleState,
    num_steps: int,
    *,
    agent_factory: AgentFactory = _default_agent_factory,
    logger: ExperimentLogger,
    sleep_s: float = 0.0,
    render: bool = True,
    rng: random.Random | None = None,
) -> list[dict]:
    """Run the agent loop for ``num_steps`` from ``initial``.

    Each step calls Jev once (one state, multiple questions), composes the
    action, steps the environment, renders, and logs. Returns the log records.
    """
    rng = rng or random.Random()
    state = initial
    records: list[dict] = []

    with agent_factory() as agent:
        for step_no in range(1, num_steps + 1):
            # 2-5. build request, call Jev, parse, record latency.
            decision = agent.decide(state)

            # 6. confidence / safety rules.
            trace = compose_action(decision, state)
            executed = trace.final

            # 7-8. execute + update environment.
            next_state = step(state, executed)

            # 9. render.
            if render:
                clear_screen()
                print(
                    render_dashboard(step_no, state, decision, trace, executed, next_state)
                )

            # logging.
            rec = _record(logger, step_no, state, decision, trace, executed, next_state)
            records.append(rec)

            state = next_state
            if sleep_s > 0:
                time.sleep(sleep_s)

    # Final stats.
    if render:
        stats = latency_stats(records)
        print()
        print(format_stats(stats))
        print(f"\nLog written to: {logger.path}")
    return records


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_manual(
    state: VehicleState,
    steps: int = C.DEFAULT_SINGLE_STEPS,
    *,
    agent_factory: AgentFactory = _default_agent_factory,
    logger: ExperimentLogger | None = None,
    render: bool = True,
) -> list[dict]:
    own_logger = logger is None
    if own_logger:
        logger = ExperimentLogger(run_name="manual")
    try:
        return run_steps(
            state,
            steps,
            agent_factory=agent_factory,
            logger=logger,
            sleep_s=0.0,
            render=render,
        )
    finally:
        if own_logger:
            logger.close()


def run_random(
    steps: int = C.DEFAULT_SINGLE_STEPS,
    *,
    agent_factory: AgentFactory = _default_agent_factory,
    logger: ExperimentLogger | None = None,
    seed: int | None = None,
    render: bool = True,
) -> list[dict]:
    rng = random.Random(seed)
    state = random_state(rng)
    own_logger = logger is None
    if own_logger:
        logger = ExperimentLogger(run_name="random")
    try:
        return run_steps(
            state,
            steps,
            agent_factory=agent_factory,
            logger=logger,
            sleep_s=0.0,
            render=render,
        )
    finally:
        if own_logger:
            logger.close()


def run_continuous(
    steps: int = C.DEFAULT_CONTINUOUS_STEPS,
    *,
    agent_factory: AgentFactory = _default_agent_factory,
    logger: ExperimentLogger | None = None,
    seed: int | None = None,
    sleep_s: float | None = None,
    render: bool = True,
) -> list[dict]:
    rng = random.Random(seed)
    state = random_state(rng)
    own_logger = logger is None
    if own_logger:
        logger = ExperimentLogger(run_name="continuous")
    try:
        return run_steps(
            state,
            steps,
            agent_factory=agent_factory,
            logger=logger,
            sleep_s=C.LOOP_SLEEP_S if sleep_s is None else sleep_s,
            render=render,
            rng=rng,
        )
    finally:
        if own_logger:
            logger.close()


__all__ = [
    "run_steps",
    "run_manual",
    "run_random",
    "run_continuous",
    "preset_scenarios",
]
