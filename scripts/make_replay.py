"""Generate a pre-recorded demo log by running real Jev steps.

The output is a self-contained JSON file under web/replay.json that the web UI
can replay without an API key. Records one run per traffic density.

Requires TYPESAFE_API_KEY. Run once (by the repo owner) before publishing:

    python scripts/make_replay.py

The generated web/replay.json is committed to the repo so anyone can replay.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env for the repo owner's key (only used locally to generate the log).
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass

from src.webapp import run_world_step  # noqa: E402
from src.world import DENSITIES, World, make_world  # noqa: E402


STEPS_PER_DENSITY = 25


def main() -> int:
    if not os.getenv("TYPESAFE_API_KEY"):
        print("ERROR: TYPESAFE_API_KEY not set. This script is run by the repo")
        print("owner locally to generate the replay log. It is NOT needed by")
        print("people who just want to replay the demo.")
        return 1

    out_path = ROOT / "web" / "replay.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[dict]] = {}
    for density in DENSITIES:
        print(f"Recording '{density}' ({STEPS_PER_DENSITY} steps) ...", flush=True)
        world = make_world(density)
        records: list[dict] = []
        for _ in range(STEPS_PER_DENSITY):
            out = run_world_step(world)
            if out["record"] is None:  # crashed
                records.append({"crashed": True, "world": out["world"]})
                break
            records.append({"record": out["record"], "world": out["world"]})
        runs[density] = records
        print(f"  -> {len(records)} steps recorded")

    payload = {
        "generated_with": "scripts/make_replay.py",
        "steps_per_density": STEPS_PER_DENSITY,
        "note": "Pre-recorded real Jev decisions. Replayable without an API key.",
        "runs": runs,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    print("Commit this file so others can replay the demo without a key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
