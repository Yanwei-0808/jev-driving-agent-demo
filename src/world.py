"""Multi-lane world for the web demo (God-view, Subway-Surfers-style).

Ego remains the only decision maker. Jev still only ever sees an EGO-CENTRIC
structured state produced by :meth:`World.observe` (nearest front car in the
same lane + lane-availability windows), so the official API call, confidence
gating and env stepping in ``jev_agent`` / ``decision`` / ``env`` are reused
unchanged. NPC cars exist to shape that state and for the visual demo.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from . import config as C
from .env import VehicleState

# Traffic density presets. spawn_p: per-step spawn probability per free lane;
# init_p: initial car probability per 30-60m slot per lane; speeds in km/h.
DENSITIES: dict[str, dict] = {
    "sparse": {"label": "空旷", "spawn_p": 0.10, "init_p": 0.15, "speed_lo": 45, "speed_frac": 1.00},
    "normal": {"label": "普通", "spawn_p": 0.25, "init_p": 0.45, "speed_lo": 35, "speed_frac": 0.95},
    "jam":    {"label": "拥堵", "spawn_p": 0.45, "init_p": 0.80, "speed_lo": 20, "speed_frac": 0.60},
}

# A target lane counts as "available" when no NPC sits inside this window.
LANE_WINDOW_AHEAD_M = 45.0
LANE_WINDOW_BEHIND_M = 8.0

SPAWN_MIN_M, SPAWN_MAX_M = 190.0, 250.0     # where new NPCs appear (far ahead)
DESPAWN_BEHIND_M, DESPAWN_AHEAD_M = 30.0, 280.0
CRASH_GAP_M = 4.5                            # ego-to-front gap that means crash
CRASH_PASSED_M = -6.0                        # just clipped the car ahead


@dataclass
class Npc:
    id: int
    lane: int
    pos: float    # metres ahead of ego (negative = behind ego)
    speed: float  # km/h


@dataclass
class World:
    ego: VehicleState
    npcs: list[Npc] = field(default_factory=list)
    density: str = "normal"
    crashed: bool = False
    travelled: float = 0.0
    _next_id: int = 1

    # ---------------- serialisation ---------------- #
    def to_dict(self) -> dict:
        return {
            "ego": self.ego.to_dict(),
            "npcs": [asdict(n) for n in self.npcs],
            "density": self.density,
            "crashed": self.crashed,
            "travelled": round(self.travelled, 1),
        }

    @staticmethod
    def from_dict(d: dict) -> "World":
        return World(
            ego=VehicleState(**d["ego"]),
            npcs=[Npc(**n) for n in d.get("npcs", [])],
            density=d.get("density", "normal"),
            crashed=bool(d.get("crashed", False)),
            travelled=float(d.get("travelled", 0.0)),
            _next_id=int(d.get("_next_id", 0))
            or (max((int(n["id"]) for n in d.get("npcs", [])), default=0) + 1),
        )

    # ---------------- ego-centric observation ---------------- #
    def observe(self) -> VehicleState:
        """Compress the world into the structured state Jev is asked about."""
        lane = self.ego.ego_lane
        ahead = [n for n in self.npcs if n.lane == lane and n.pos > 0]
        if ahead:
            f = min(ahead, key=lambda n: n.pos)
            fd, fs, has = f.pos, f.speed, True
        else:
            fd, fs, has = 200.0, 0.0, False
        return VehicleState(
            ego_speed=self.ego.ego_speed,
            ego_lane=lane,
            front_distance=fd,
            front_speed=fs,
            has_front_car=has,
            left_lane_available=self._lane_free(lane - 1),
            right_lane_available=self._lane_free(lane + 1),
            speed_limit=self.ego.speed_limit,
            lane_offset=0.0,
        )

    def _lane_free(self, lane: int) -> bool:
        if lane < 0 or lane >= C.NUM_LANES:
            return False
        return not any(
            n.lane == lane and -LANE_WINDOW_BEHIND_M < n.pos < LANE_WINDOW_AHEAD_M
            for n in self.npcs
        )

    # ---------------- world update ---------------- #
    def advance(self, ego_after: VehicleState, rng: random.Random | None = None) -> None:
        """Move the world one DT_S step using the post-action ego state.

        Only ``ego_after.ego_speed`` / ``ego_lane`` / ``speed_limit`` are used;
        front-car fields of ``ego_after`` are ignored (NPCs are the truth).
        """
        rng = rng or random.Random()
        dt = C.DT_S

        # 1. Move NPCs relative to ego.
        for n in self.npcs:
            n.pos += (n.speed - ego_after.ego_speed) / 3.6 * dt

        # 2. Simple car-following so NPCs never overlap (front-most first).
        by_lane: dict[int, list[Npc]] = {}
        for n in self.npcs:
            by_lane.setdefault(n.lane, []).append(n)
        for cars in by_lane.values():
            cars.sort(key=lambda n: -n.pos)
            for i in range(1, len(cars)):
                front, back = cars[i - 1], cars[i]
                if front.pos - back.pos < 10.0:
                    back.speed = min(back.speed, front.speed)
                    back.pos = min(back.pos, front.pos - 10.0)

        # 3. Despawn cars that left the relevant window.
        self.npcs = [
            n for n in self.npcs if -DESPAWN_BEHIND_M < n.pos < DESPAWN_AHEAD_M
        ]

        # 4. Adopt the new ego state and accumulate distance.
        self.ego = VehicleState(
            ego_speed=ego_after.ego_speed,
            ego_lane=ego_after.ego_lane,
            front_distance=ego_after.front_distance,
            front_speed=ego_after.front_speed,
            has_front_car=ego_after.has_front_car,
            left_lane_available=ego_after.left_lane_available,
            right_lane_available=ego_after.right_lane_available,
            speed_limit=ego_after.speed_limit,
            lane_offset=ego_after.lane_offset,
        )
        self.travelled += ego_after.ego_speed / 3.6 * dt

        # 5. Spawn new traffic far ahead.
        self._spawn(rng)

        # 6. Crash detection: ego too close to (or clipping) a same-lane car.
        for n in self.npcs:
            if n.lane == self.ego.ego_lane and CRASH_PASSED_M < n.pos < CRASH_GAP_M:
                self.crashed = True
                break

    def _spawn(self, rng: random.Random) -> None:
        cfg = DENSITIES.get(self.density, DENSITIES["normal"])
        for lane in range(C.NUM_LANES):
            if any(n.lane == lane and SPAWN_MIN_M - 30 < n.pos < SPAWN_MAX_M + 20 for n in self.npcs):
                continue
            if rng.random() < cfg["spawn_p"]:
                hi = max(int(cfg["speed_lo"]) + 1, int(self.ego.speed_limit * cfg["speed_frac"]))
                speed = min(self.ego.speed_limit, rng.randint(int(cfg["speed_lo"]), hi))
                self.npcs.append(
                    Npc(id=self._next_id, lane=lane, pos=rng.uniform(SPAWN_MIN_M, SPAWN_MAX_M), speed=speed)
                )
                self._next_id += 1


def make_world(density: str, rng: random.Random | None = None) -> World:
    """Create an initial world with the requested traffic density."""
    rng = rng or random.Random()
    cfg = DENSITIES.get(density, DENSITIES["normal"])
    ego = VehicleState(
        ego_speed=50,
        ego_lane=C.NUM_LANES // 2,
        front_distance=200.0,
        front_speed=0,
        has_front_car=False,
        left_lane_available=False,   # re-derived by observe(); placeholder
        right_lane_available=False,
        speed_limit=80,
    )
    w = World(ego=ego, density=density)
    for lane in range(C.NUM_LANES):
        pos = rng.uniform(25, SPAWN_MAX_M - 20)
        while pos < SPAWN_MAX_M:
            if rng.random() < cfg["init_p"]:
                hi = max(int(cfg["speed_lo"]) + 1, int(ego.speed_limit * cfg["speed_frac"]))
                speed = min(ego.speed_limit, rng.randint(int(cfg["speed_lo"]), hi))
                w.npcs.append(Npc(id=w._next_id, lane=lane, pos=pos, speed=speed))
                w._next_id += 1
            pos += rng.uniform(30, 60)
    return w


__all__ = ["World", "Npc", "make_world", "DENSITIES"]
