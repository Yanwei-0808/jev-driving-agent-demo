"""Command-line entry point.

Usage:
    python -m src.main                  # interactive menu
    python -m src.main manual           # pick a preset (1-5) or type a state
    python -m src.main random           # one random state, a few steps
    python -m src.main continuous       # continuous simulation loop

Common options:
    --steps N       number of steps to run (default: 1 for manual/random, 30 for continuous)
    --preset 1-5    use preset scenario N in manual mode (skip the menu)
    --no-render     do not print the dashboard (still logs)
    --sleep S       seconds between steps in continuous mode
"""

from __future__ import annotations

import argparse
import os
import sys

# Load variables from a local .env file (if present) so TYPESAFE_API_KEY and
# any tuning overrides are picked up automatically. This is optional; the SDK
# itself only reads real environment variables.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv is a declared dependency
    pass

from . import config as C
from .env import preset_scenarios, state_from_text
from .simulator import run_continuous, run_manual, run_random


def _print_presets() -> None:
    print("\nPreset scenarios:\n")
    for i, (name, intent, state) in enumerate(preset_scenarios(), 1):
        print(f"  [{i}] {name}")
        print(f"      intent: {intent}")
        print(
            f"      state:  ego_speed={state.ego_speed} ego_lane={state.ego_lane} "
            f"front_distance={state.front_distance} front_speed={state.front_speed} "
            f"has_front_car={state.has_front_car}"
        )
        print(
            f"              left={state.left_lane_available} "
            f"right={state.right_lane_available} speed_limit={state.speed_limit}"
        )
        print()


def _choose_manual_state(args: argparse.Namespace):
    presets = preset_scenarios()
    if args.preset is not None:
        if not (1 <= args.preset <= len(presets)):
            print(f"--preset must be 1..{len(presets)}")
            sys.exit(2)
        name, _intent, state = presets[args.preset - 1]
        print(f"Using preset [{args.preset}] {name}\n")
        return state

    _print_presets()
    print("Choose: 1-5 for a preset, 'c' to type a custom state, Enter to abort.")
    choice = input("> ").strip().lower()
    if choice == "":
        print("Aborted.")
        sys.exit(0)
    if choice == "c":
        print(
            "Paste a state block (one field per line, blank line to finish)."
        )
        print("Fields: ego_speed, ego_lane, front_distance, front_speed,")
        print("        has_front_car, left_lane_available, right_lane_available,")
        print("        speed_limit, lane_offset")
        lines: list[str] = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        return state_from_text("\n".join(lines))
    if choice.isdigit() and 1 <= int(choice) <= len(presets):
        name, _intent, state = presets[int(choice) - 1]
        print(f"\nUsing preset [{choice}] {name}\n")
        return state
    print(f"Invalid choice: {choice!r}")
    sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jev-driving-agent",
        description="Jev real-time vehicle decision agent (toy environment).",
    )
    sub = p.add_subparsers(dest="mode")

    pm = sub.add_parser("manual", help="manual scenario (preset or custom state)")
    pm.add_argument("--steps", type=int, default=C.DEFAULT_SINGLE_STEPS)
    pm.add_argument("--preset", type=int, default=None)
    pm.add_argument("--no-render", action="store_true")

    pr = sub.add_parser("random", help="random scenario")
    pr.add_argument("--steps", type=int, default=C.DEFAULT_SINGLE_STEPS)
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--no-render", action="store_true")

    pc = sub.add_parser("continuous", help="continuous simulation loop")
    pc.add_argument("--steps", type=int, default=C.DEFAULT_CONTINUOUS_STEPS)
    pc.add_argument("--seed", type=int, default=None)
    pc.add_argument("--sleep", type=float, default=None)
    pc.add_argument("--no-render", action="store_true")

    pw = sub.add_parser("web", help="launch the web UI (animated road demo)")
    pw.add_argument("--port", type=int, default=8000)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    render = not getattr(args, "no_render", False)

    if args.mode is None:
        # Interactive top-level menu.
        print("=" * 60)
        print("    Jev Driving Agent  (Toy Environment)")
        print("=" * 60)
        print("\nModes:")
        print("  1) Manual scenario   (preset or custom state)")
        print("  2) Random scenario   (random state, a few steps)")
        print("  3) Continuous simulation")
        print("  4) Web UI            (animated road demo in your browser)")
        print("\nNote: requires the TYPESAFE_API_KEY env var to be set.")
        choice = input("\nChoose 1/2/3/4 > ").strip()
        if choice == "1":
            args = build_parser().parse_args(["manual"])
            args.no_render = not render
            state = _choose_manual_state(args)
            run_manual(state, steps=C.DEFAULT_SINGLE_STEPS, render=render)
        elif choice == "2":
            run_random(steps=C.DEFAULT_SINGLE_STEPS, render=render)
        elif choice == "3":
            run_continuous(steps=C.DEFAULT_CONTINUOUS_STEPS, render=render)
        elif choice == "4":
            from .webapp import main as web_main

            web_main()
        else:
            print(f"Invalid choice: {choice!r}")
            return 2
        return 0

    if args.mode == "manual":
        state = _choose_manual_state(args)
        run_manual(state, steps=args.steps, render=render)
    elif args.mode == "random":
        run_random(steps=args.steps, seed=args.seed, render=render)
    elif args.mode == "continuous":
        run_continuous(steps=args.steps, seed=args.seed, sleep_s=args.sleep, render=render)
    elif args.mode == "web":
        from .webapp import main as web_main

        web_main(port=args.port)
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
