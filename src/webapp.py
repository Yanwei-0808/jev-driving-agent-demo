"""Web UI backend for the Jev driving agent (God-view multi-car world).

Stdlib only (http.server) - no extra dependencies. Serves ``web/index.html``
and two JSON endpoints:

    POST /api/world/init  body {"density": "sparse|normal|jam"}
        -> {"world": {...}}   fresh world with the requested traffic density

    POST /api/world/step  body {"world": {...}}
        -> {"record": {...}, "world": {...}}
           one full agent step: observe -> Jev (one request, 4 questions)
           -> confidence/safety gating -> env step -> world advance.

The frontend owns the world JSON and echoes it back each step (stateless).

Run:
    python -m src.main web          # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Load a local .env so TYPESAFE_API_KEY is picked up when launched directly.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from .decision import compose_action
from .env import step as env_step
from .jev_agent import JevAgent
from .world import DENSITIES, World, make_world

WEB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"
)

_agent: JevAgent | None = None
_agent_lock = threading.Lock()


def get_agent() -> JevAgent:
    """Lazily create one shared JevAgent (connection pooling across steps)."""
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = JevAgent()
        return _agent


def run_world_step(world: World) -> dict:
    """One full agent step over the world. Returns the API payload."""
    if world.crashed:
        # Do not keep deciding after a crash; let the frontend show it.
        return {"record": None, "world": world.to_dict()}

    # 1. Ego-centric structured state (what Jev actually sees).
    obs = world.observe()

    # 2. Jev: one request, multiple typed questions.
    decision = get_agent().decide(obs)

    # 3. Confidence gating + safety rules -> final action.
    trace = compose_action(decision, obs)
    executed = trace.final

    # 4. Ego env step (speed / lane rules reused from the toy environment).
    ego_after = env_step(obs, executed)

    # 5. Advance the whole world (NPCs, spawning, crash detection).
    world.advance(ego_after)

    record = {
        "state": obs.to_dict(),
        "decisions": decision.to_dict(),
        "latency_ms": decision.latency_ms,
        "executed_action": {
            "longitudinal": executed.longitudinal,
            "lateral": executed.lateral,
        },
        "reasons": list(trace.reasons),
        "fallback": decision.fallback,
    }
    return {"record": record, "world": world.to_dict()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "application/json; charset=utf-8"):
        data = (
            body
            if isinstance(body, bytes)
            else json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/replay.json":
                # Pre-recorded Jev decisions; lets users without an API key
                # watch a real Jev run in replay mode.
                with open(os.path.join(WEB_DIR, "replay.json"), "rb") as f:
                    self._send(200, f.read(), "application/json; charset=utf-8")
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._send(400, {"error": f"bad json: {exc}"})
            return

        try:
            if path == "/api/world/init":
                density = payload.get("density", "normal")
                if density not in DENSITIES:
                    self._send(400, {"error": f"density must be one of {list(DENSITIES)}"})
                    return
                self._send(200, {"world": make_world(density).to_dict()})
            elif path == "/api/world/step":
                world = World.from_dict(payload["world"])
                self._send(200, run_world_step(world))
            else:
                self._send(404, {"error": "not found"})
        except RuntimeError as exc:  # e.g. missing API key
            self._send(500, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):  # keep the console quiet
        pass


def main(port: int | None = None) -> None:
    port = port or int(os.getenv("WEB_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Jev Driving Agent web UI:  http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
