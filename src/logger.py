"""JSONL experiment logging + latency statistics.

Every decision step is appended to a JSONL file under ``logs/``. After a run we
compute latency statistics. These numbers are LOCAL experiment results, not a
Jev performance benchmark (see README).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from . import config as C


class ExperimentLogger:
    """Append one JSON object per step to a timestamped JSONL file."""

    def __init__(self, run_name: str | None = None, log_dir: str | None = None):
        self._log_dir = Path(log_dir or C.LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = run_name or "run"
        # Sanitise the name so it is safe as a filename.
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        self.path = self._log_dir / f"{stamp}_{safe}.jsonl"
        self._records: list[dict] = []
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def records(self) -> list[dict]:
        return list(self._records)


def latency_stats(records: list[dict]) -> dict[str, float]:
    """Compute latency statistics over the recorded steps.

    Only real (non-fallback) latencies are informative, but we report both the
    full set and the non-fallback subset so you can see the difference.
    """
    all_lat = [float(r.get("latency_ms", 0.0)) for r in records]
    real_lat = [
        float(r.get("latency_ms", 0.0))
        for r in records
        if not r.get("fallback", False)
    ]

    def _summary(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"count": 0, "avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(xs),
            "avg": statistics.fmean(xs),
            "median": statistics.median(xs),
            "min": min(xs),
            "max": max(xs),
        }

    return {
        "all_steps": _summary(all_lat),
        "non_fallback_steps": _summary(real_lat),
    }


def format_stats(stats: dict[str, float]) -> str:
    """Render latency stats as a readable block for the UI / report."""
    lines = ["Latency statistics (LOCAL experiment, not a Jev benchmark)"]

    def _block(title: str, s: dict) -> list[str]:
        return [
            f"  {title}:",
            f"    count  : {s.get('count', 0)}",
            f"    avg    : {s.get('avg', 0.0):.1f} ms",
            f"    median : {s.get('median', 0.0):.1f} ms",
            f"    min    : {s.get('min', 0.0):.1f} ms",
            f"    max    : {s.get('max', 0.0):.1f} ms",
        ]

    lines += _block("all steps", stats.get("all_steps", {}))
    lines += _block("non-fallback (real API) steps", stats.get("non_fallback_steps", {}))
    lines.append(
        "  Note: latency depends on network, API service, model version and "
        "environment."
    )
    return "\n".join(lines)


__all__ = ["ExperimentLogger", "latency_stats", "format_stats"]
