"""Phase 1: minimal Jev API verification.

Runs the smallest possible request against the official TypeSafe (Jev) API to
confirm the SDK is installed, the API key works, and we can read a structured
answer. Requires TYPESAFE_API_KEY to be set.

    python scripts/verify_jev.py
"""

from __future__ import annotations

import os
import sys
import time

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load a local .env so the script works by just creating .env.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from typesafe_sdk import Choice, Noul, TypeSafeClient  # noqa: E402


def main() -> int:
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        print("ERROR: TYPESAFE_API_KEY is not set.")
        print("Create one at https://console.typesafe.ai/keys and put it in .env")
        return 1

    print("Sending a minimal request to Jev (one Noul + one Choice) ...")
    with TypeSafeClient(api_key=api_key, timeout=15.0) as client:
        t0 = time.perf_counter()
        response = client.system_one(
            state="A car is 12 metres ahead travelling 40 km/h; ego is at 72 km/h.",
            questions={
                "is_risky": Noul(
                    instructions="Is there a high collision risk?",
                    criteria={"true": "Gap is small or closing fast.",
                              "false": "Gap is ample and stable."},
                ),
                "action": Choice(
                    instructions="What longitudinal action should the ego take?",
                    criteria={
                        "accelerate": "Safe to speed up.",
                        "maintain": "Hold current speed.",
                        "brake": "Reduce speed.",
                    },
                ),
            },
        )
        latency = (time.perf_counter() - t0) * 1000.0

    print(f"\nmodel: {response.model}")
    print(f"latency: {latency:.1f} ms")
    print(f"usage: {response.usage}")
    print(f"is_risky.noul = {response.nouls['is_risky'].noul}")
    a = response.choices["action"]
    print(f"action.choice = {a.choice}")
    print(f"action.confidence = {a.confidence}")
    print(f"action.probabilities = {a.probabilities}")
    print("\nOK: Jev API is reachable and returns structured answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
