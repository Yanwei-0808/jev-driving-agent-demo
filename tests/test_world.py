"""Tests for the multi-lane world model (no API key needed)."""

from __future__ import annotations

import random

from src import config as C
from src.env import VehicleState, step as env_step
from src.world import DENSITIES, Npc, World, make_world


def _world_with(npcs, lane=1, speed=60) -> World:
    ego = VehicleState(speed, lane, 200, 0, False, False, False, 80)
    next_id = max((n.id for n in npcs), default=0) + 1
    return World(ego=ego, npcs=list(npcs), density="normal", _next_id=next_id)


def test_observe_nearest_front_car():
    w = _world_with([Npc(1, 1, 80, 50), Npc(2, 1, 30, 45)])
    obs = w.observe()
    assert obs.has_front_car
    assert obs.front_distance == 30
    assert obs.front_speed == 45


def test_observe_no_front_car():
    w = _world_with([Npc(1, 0, 50, 50)])
    obs = w.observe()
    assert not obs.has_front_car
    assert obs.front_distance == 200


def test_lane_free_window():
    # Same-lane car inside the 45m window -> lane not available.
    w = _world_with([Npc(1, 2, 30, 50)])
    obs = w.observe()
    assert obs.right_lane_available is False
    assert obs.left_lane_available is True  # empty
    # Car just outside the window -> available.
    w2 = _world_with([Npc(1, 2, 60, 50)])
    assert w2.observe().right_lane_available is True
    # Car slightly behind -> blocks the window too.
    w3 = _world_with([Npc(1, 2, -5, 50)])
    assert w3.observe().right_lane_available is False


def test_lane_free_boundaries():
    w = _world_with([], lane=0)
    assert w.observe().left_lane_available is False  # no lane 0-1
    w2 = _world_with([], lane=C.NUM_LANES - 1)
    assert w2.observe().right_lane_available is False  # no lane beyond


def test_advance_moves_npcs_relative_to_ego():
    w = _world_with([Npc(1, 1, 50, 40)])
    ego_after = env_step(w.observe(), type("A", (), {"longitudinal": "maintain", "lateral": "keep_lane"})())
    w.advance(ego_after)
    # ego 60, npc 40 -> relative -20 km/h -> -5.56 m per 1s step
    assert abs(w.npcs[0].pos - (50 - 20 / 3.6)) < 1e-6


def test_crash_detection():
    w = _world_with([Npc(1, 1, 3.0, 60)])
    ego_after = env_step(w.observe(), type("A", (), {"longitudinal": "maintain", "lateral": "keep_lane"})())
    w.advance(ego_after)
    assert w.crashed


def test_no_crash_when_far():
    w = _world_with([Npc(1, 1, 40, 60)])
    ego_after = env_step(w.observe(), type("A", (), {"longitudinal": "brake", "lateral": "keep_lane"})())
    w.advance(ego_after)
    assert not w.crashed


def test_npc_ids_stable_and_despawn():
    w = _world_with([Npc(7, 1, -100, 50)])  # far behind -> despawned
    ego_after = env_step(w.observe(), type("A", (), {"longitudinal": "maintain", "lateral": "keep_lane"})())
    w.advance(ego_after)
    assert all(n.id != 7 for n in w.npcs)


def test_make_world_all_densities():
    for density in DENSITIES:
        w = make_world(density, rng=random.Random(1))
        assert w.density == density
        assert not w.crashed
        assert C.NUM_LANES // 2 == w.ego.ego_lane
        for n in w.npcs:
            assert 0 <= n.lane < C.NUM_LANES
            # Ahead cars have positive pos; rear cars (overtaking) have negative pos.
            assert n.pos != 0
        # ego-centric observation must be valid
        from src.env import validate

        assert validate(w.observe()) == []


def test_make_world_creates_rear_cars():
    """make_world seeds at least one rear car so rear risk shows up early."""
    w = make_world("normal", rng=random.Random(1))
    behind = [n for n in w.npcs if n.pos < 0]
    assert len(behind) >= 1
    # Rear cars are faster than ego so they will catch up.
    assert all(n.speed > w.ego.ego_speed for n in behind)


def test_make_world_driving_mode_passthrough():
    """driving_mode is stored on the world and flows into observe() urgency/target."""
    for mode_key, mcfg in C.DRIVING_MODES.items():
        w = make_world("normal", rng=random.Random(1), mode=mode_key)
        assert w.driving_mode == mode_key
        obs = w.observe()
        assert obs.urgency == mcfg["urgency"]
        assert obs.target_speed_kmh == float(mcfg["target_speed"])


def test_observe_rear_car_nearest_behind():
    """observe() reports the closest behind car (least negative pos)."""
    w = _world_with([Npc(1, 1, -30, 70), Npc(2, 1, -120, 65)])
    obs = w.observe()
    assert obs.has_rear_car is True
    assert obs.rear_distance == 30      # -(-30)
    assert obs.rear_speed == 70


def test_observe_no_rear_car():
    """No car behind -> has_rear_car False, defaults."""
    w = _world_with([Npc(1, 0, 50, 50)], lane=1)
    obs = w.observe()
    assert obs.has_rear_car is False
    assert obs.rear_distance == 200
    assert obs.rear_speed == 0


def test_rear_car_tailgates_not_through_ego():
    """A faster rear car in the ego lane is clamped behind ego (virtual lead)."""
    # 5 m behind, 12 km/h faster -> after 1s moves +3.33m to -1.67, still behind;
    # car-following then clamps it to 10m behind ego (pos=-10).
    w = _world_with([Npc(1, 1, -5, 72)])
    ego_after = env_step(w.observe(), type("A", (), {"longitudinal": "maintain", "lateral": "keep_lane"})())
    w.advance(ego_after, rng=random.Random(0))
    # The original rear car (id=1, lane=1) must still be behind ego.
    rear = [n for n in w.npcs if n.id == 1 and n.lane == 1]
    assert len(rear) == 1
    assert rear[0].pos < 0
    assert rear[0].speed <= w.ego.ego_speed  # slowed down to match ego


def test_world_roundtrip():
    w = make_world("normal", rng=random.Random(3))
    d = w.to_dict()
    w2 = World.from_dict(d)
    assert w2.density == w.density
    assert len(w2.npcs) == len(w.npcs)
    assert w2.ego.ego_speed == w.ego.ego_speed
    assert [n.id for n in w2.npcs] == [n.id for n in w.npcs]


def test_following_prevents_overlap():
    w = _world_with([Npc(1, 1, 40, 60), Npc(2, 1, 30, 90)])
    ego_after = env_step(w.observe(), type("A", (), {"longitudinal": "maintain", "lateral": "keep_lane"})())
    w.advance(ego_after, rng=random.Random(0))
    # Only check same-lane NPCs near the ego (spawned cars are 190+ m away).
    cars = sorted([n for n in w.npcs if n.lane == 1 and n.pos < 100], key=lambda n: -n.pos)
    # front car must still be ahead of the back car by >= 10m
    assert cars[0].pos - cars[1].pos >= 10.0 - 1e-6
