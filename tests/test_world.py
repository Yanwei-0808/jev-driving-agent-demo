"""Tests for the multi-lane world model (no API key needed)."""

from __future__ import annotations

import random

from src import config as C
from src.env import VehicleState, step as env_step
from src.world import DENSITIES, Npc, World, make_world


def _world_with(npcs, lane=1, speed=60) -> World:
    ego = VehicleState(speed, lane, 200, 0, False, False, False, 80)
    return World(ego=ego, npcs=list(npcs), density="normal")


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
            assert n.pos > 0
        # ego-centric observation must be valid
        from src.env import validate

        assert validate(w.observe()) == []


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
    w.advance(ego_after)
    cars = sorted(w.npcs, key=lambda n: -n.pos)
    # front car must still be ahead of the back car by >= 10m
    assert cars[0].pos - cars[1].pos >= 10.0 - 1e-6
