# Jev Driving Agent

A **toy** real-time vehicle decision agent built on the [Jev](https://docs.typesafe.ai)
API (TypeSafe AI's *System One Model*). The goal is **not** autonomous driving —
it is a small, observable, experimentable environment for understanding how Jev
works as a **real-time decision model**: you send it a structured *state* and
typed *questions*, and it returns structured, probabilistic *answers* your code
can branch on.

```
Vehicle State  ->  Jev (one request, multiple typed questions)
              ->  Structured Decisions (Noul / Choice / Score)
              ->  Python logic (confidence gating + safety)
              ->  Action
              ->  Environment update
              ->  next State  ->  Jev  ...  (loop)
```

> This is a **Toy Environment**. It is **not** an autonomous-driving system, it
> is **not** a benchmark of Jev's official performance, and API latency reported
> here is a **local experiment result**. The demo uses **structured state**, not
> camera/vision input.

---

## 1. How this maps to your Jev research

| Jev concept | Where it shows up in this project |
|---|---|
| **System One** (fast, local decisions) | Each loop step asks Jev for an instant decision instead of a long reasoning trace. |
| **Atomic Questions** | The driving task is split into independent questions: risk, rear-collision-risk, longitudinal, lateral, risk-level. |
| **Choice** | Longitudinal (`accelerate`/`maintain`/`brake`) and lateral (`keep_lane`/`change_left`/`change_right`) action selection. |
| **Noul** | Collision-risk judgement (`0..1`). Noul has **no separate `confidence`** — the number itself is the belief. A second Noul question covers rear-collision (tailgating) risk. |
| **Score** | Risk-severity rating on an ordered rubric `safe/marginal/high/critical`, to also showcase the Score primitive. |
| **Probabilities** | Every Choice/Score answer exposes the full probability distribution over options. |
| **Confidence** | Choice/Score answers carry `confidence`; it gates whether an action is auto-executed. |
| **One request, many questions** | All questions go in a **single** `system_one` call (parallel evaluation). |
| **Agent Loop** | `State -> Jev -> Action -> New State -> Jev ...` runs continuously. |
| **Latency** | Each step records `latency_ms`; stats are printed at the end. |

### What is official Jev capability vs. project design

- **Official Jev (TypeSafe) capability** — everything in
  [`src/jev_agent.py`](file:///d:/phyagentos/cvpr/jev-demo/src/jev_agent.py): the
  SDK call (`TypeSafeClient.system_one`), the three primitives
  (`Noul`/`Choice`/`Score`), sending multiple questions in one request, and the
  returned `noul` / `choice` / `probabilities` / `confidence` / `score` / `legend`
  fields. These match the current docs at <https://docs.typesafe.ai/api>.
- **This project's own design** — the driving environment
  ([`src/env.py`](file:///d:/phyagentos/cvpr/jev-demo/src/env.py)), the question
  *wording* and option *sets*, the confidence-gating + safety rules
  ([`src/decision.py`](file:///d:/phyagentos/cvpr/jev-demo/src/decision.py)), the
  thresholds in [`src/config.py`](file:///d:/phyagentos/cvpr/jev-demo/src/config.py),
  the loop ([`src/simulator.py`](file:///d:/phyagentos/cvpr/jev-demo/src/simulator.py)),
  logging/UI, and the preset scenarios. None of this is part of Jev.

> **A note on confidence:** `confidence` is the model's signal of how decided it
> is, derived from the shape of the probability distribution. It is used here to
> gate actions. It is **not** a guarantee of correctness, and the right
> threshold must be tested per task. Tune it in `config.py`.

---

## 2. Project structure

```
jev-demo/
├── README.md
├── pyproject.toml
├── .env.example          # copy to .env and add your key
├── scripts/
│   ├── verify_jev.py     # Phase 1: minimal Jev API check
│   └── make_replay.py     # pre-record real Jev decisions -> web/replay.json
├── src/
│   ├── config.py         # ALL tunable thresholds/params (one place)
│   ├── env.py            # VehicleState + toy environment step + scenarios
│   ├── jev_agent.py      # <-- the ONLY file that calls the official Jev API
│   ├── decision.py       # combine Jev answers + confidence gating + safety
│   ├── simulator.py      # the real-time loop + 3 run modes
│   ├── world.py          # multi-lane world (NPC traffic) for the web demo
│   ├── logger.py         # JSONL logging + latency statistics
│   ├── ui.py             # terminal dashboard + ASCII road
│   ├── webapp.py         # web UI backend (stdlib HTTP, no extra deps)
│   └── main.py           # CLI entry point
├── web/
│   ├── index.html        # animated road web demo (single file, no build)
│   └── replay.json       # pre-recorded real Jev run (for replay mode)
├── logs/                 # generated JSONL run logs (auto-created)
└── tests/                # pytest, no API key needed
```

---

## 3. Quick Start（两种体验路径）

### 路径 A：没有 API key（回放模式，推荐先试）

> 无需申请 key、无需联网调 Jev，直接看预录的真实 Jev 决策。

```bash
# 1. 克隆仓库
git clone https://github.com/Yanwei-0808/jev-driving-agent-demo.git
cd jev-driving-agent-demo

# 2. 安装依赖（仅 2 个包）
pip install typesafe-sdk python-dotenv

# 3. 启动 Web Demo
python -m src.main web
```

浏览器打开 <http://127.0.0.1:8000>，然后：

1. 点击 **「模式：实时」** 按钮 → 切换为 **「模式：回放」**
2. 选择交通密度（空旷 / 普通 / 拥堵）
3. 点击 **「▶ 单步」** 逐步观看 Jev 决策，或 **「自动循环」** 连续播放
4. 观察右侧面板：碰撞风险（Noul）、后车追尾风险（Noul）、风险等级（Score）、纵向/横向动作（Choice）

### 路径 B：有 API key（实时模式）

> 让 Jev 对当前场景做实时决策，延迟和结果都是真实的。

```bash
# 1. 克隆 + 安装（同路径 A）
git clone https://github.com/Yanwei-0808/jev-driving-agent-demo.git
cd jev-driving-agent-demo
pip install typesafe-sdk python-dotenv

# 2. 配置 API key
cp .env.example .env          # Windows: copy .env.example .env
# 编辑 .env，填入 TYPESAFE_API_KEY=your_key_here

# 3.（可选）验证 key 是否可用
python scripts/verify_jev.py

# 4. 启动 Web Demo
python -m src.main web
```

浏览器打开 <http://127.0.0.1:8000>，保持 **「模式：实时」**（默认），选驾驶模式（舒适 / 正常 / 赶时间），点 **「▶ 单步」** 或 **「自动循环」** 即可。

> API key 申请：<https://console.typesafe.ai/keys>

### Web Demo 界面说明

| 界面元素 | 作用 |
|---|---|
| **交通** 按钮（空旷/普通/拥堵） | 切换车流密度，点击即重置世界 |
| **驾驶** 按钮（舒适/正常/赶时间） | 切换驾驶模式 → 目标车速 + 是否着急发给 Jev，点击即重置世界 |
| **模式：实时/回放** | 实时 = 调真实 Jev API（需 key）；回放 = 播放预录（无需 key） |
| **▶ 单步** | 执行一步 Jev 决策并更新世界 |
| **自动循环** | 连续执行 Jev 决策循环 |
| 右侧决策面板 | 碰撞风险 Noul、后车追尾 Noul、风险等级 Score、纵向/横向 Choice（含概率分布 + 置信度） |
| 右侧 EGO 状态面板 | 喂给 Jev 的结构化状态：车速、目标车速、是否着急、车道、前车/后车距离与速度、左右道可用性 |
| 道路画布 | 上帝视角 3 车道，ego 固定底部（蓝色），前车灰白，后车暖色（会超越 ego） |

---

## 4. Install (detailed)

Requires Python >= 3.10.

```bash
pip install -e .
# or just the runtime deps directly:
pip install typesafe-sdk python-dotenv
# dev (tests):
pip install pytest
```

## 5. Configure the API key (detailed)

> **No API key?** See [Quick Start 路径 A](#3-quick-start两种体验路径) above —
> the web demo has a **回放 (replay)** mode that plays back a pre-recorded real
> Jev run (`web/replay.json`), no key needed.

1. Create a key at <https://console.typesafe.ai/keys>.
2. Copy `.env.example` to `.env` and fill in:

   ```env
   TYPESAFE_API_KEY=your_key_here
   ```

`main.py` loads `.env` automatically. The SDK reads `TYPESAFE_API_KEY` from the
environment. You can also export it directly:

```bash
# Windows PowerShell
$env:TYPESAFE_API_KEY="your_key"
# Linux/macOS
export TYPESAFE_API_KEY=your_key
```

### Phase 1 sanity check

```bash
python scripts/verify_jev.py
```

This sends one `Noul` + one `Choice` to Jev and prints the structured answer +
latency. If this works, the rest of the project will too.

---

## 6. Run

```bash
# interactive menu
python -m src.main

# Web UI (animated road demo in your browser) - RECOMMENDED
python -m src.main web            # or: python -m src.webapp
# then open http://127.0.0.1:8000

# Manual scenario (pick preset 1-7, or type a custom state)
python -m src.main manual
python -m src.main manual --preset 3 --steps 5

# Random scenario
python -m src.main random --steps 5 --seed 42

# Continuous simulation (loop)
python -m src.main continuous --steps 30 --sleep 1
```

| Mode | What it does |
|---|---|
| `web` | **God-view animated demo** (recommended): 3-lane street with selectable traffic density (空旷 / 普通 / 拥堵) and **driving mode** (舒适 / 正常 / 赶时间 → target speed + urgency sent to Jev). Multi-car world rendered top-down with ego fixed at the bottom. **Rear traffic** spawns faster cars behind the ego that catch up and overtake; Jev evaluates rear-collision risk via a dedicated Noul question (reference only — does not change the ego action). Two modes via the **模式：实时/回放** button: **实时 (live)** runs real Jev decisions against the API (needs a key); **回放 (replay)** plays back a pre-recorded real Jev run (`web/replay.json`) so visitors without a key can still watch what Jev decides. Starts paused so you can single-step through decisions. |
| `manual` | Choose one of 7 preset scenarios (or type a custom state block), run N steps. For quick testing. |
| `random` | Generate one plausible random state, run N steps. |
| `continuous` | Continuous loop for N steps with a sleep between them. Shows the real-time agent loop. |

`--no-render` suppresses the dashboard (the run still logs to `logs/`).

### The 7 preset scenarios (in `src/env.py`)

1. Front far, similar speed → expect keep/mild accelerate.
2. Front clearly slower → expect decelerate or change lane.
3. Front very close → high risk, expect brake.
4. Front slow + left lane free → can test `change_left`.
5. Front slow + no lane change possible → expect brake / keep_lane.
6. Slow & comfortable cruise (慢速舒适) → low speed, ample space.
7. Rushing, late (快速赶时间) → fast ego closing on a slow car.

> These are for testing the **system**, not for forcing a "correct" answer. We
> record what Jev actually outputs; we do not retcon scenarios to look right.

### Manual custom state format

```
ego_speed = 72
ego_lane = 1
front_distance = 18
front_speed = 55
has_front_car = True
left_lane_available = True
right_lane_available = False
speed_limit = 80
```

Units: speed in km/h, distance in m, lane in `0..NUM_LANES-1`.

---

## 7. An actual run example

Dashboard frame from `python -m src.main manual --preset 3` (step 17 of a run;
values are illustrative — actual numbers come from Jev when you run it):

```
============================================================
    Jev Driving Agent  (Toy Environment)
============================================================
Step: 17

Vehicle State
------------------------------------------------------------
  Speed:            72 km/h  (limit 80)
  Lane:             1
  Front car:        Yes
  Front Distance:   18.0 m
  Front Speed:      55 km/h
  Left Lane Free:   Yes
  Right Lane Free:  No

Jev Decisions
------------------------------------------------------------
  Model: jev-1.13.0
  Risk (Noul, 0=no .. 1=yes):
    collision_risk#####################..... 0.82
  Risk level (Score): 2.10 -> high  confidence 0.88

  Longitudinal (Choice):
    -> BRAKE
    confidence: 0.94
    brake=0.94  maintain=0.04  accelerate=0.02

  Lateral (Choice):
    -> CHANGE_LEFT
    confidence: 0.61
    change_left=0.61  keep_lane=0.35  change_right=0.04

Executed Action
------------------------------------------------------------
  BRAKE + CHANGE_LEFT
  reasons:
    - Risk noul 0.82 > 0.8; forcing brake

Next State
------------------------------------------------------------
  Speed:          62 km/h
  Lane:           0
  Front Distance: 16.1 m

Latency: 142.0 ms

+============================+
|                            |
|----------------------------|
|    E    F                  |
+============================+
 E=ego  F=front  L=   R=X  (lane0=lower, lane1=upper)
============================================================
```

---

## 8. View latency and confidence

- **Per step:** the dashboard prints `Latency: <ms>` and a `confidence:` line
  under each Choice/Score answer.
- **Per run:** after the loop finishes, latency statistics are printed:

  ```
  Latency statistics (LOCAL experiment, not a Jev benchmark)
    all steps:
      count  : 30
      avg    : 143.2 ms
      median : 138.0 ms
      min    : 98.0 ms
      max    : 312.0 ms
    non-fallback (real API) steps:
      ...
    Note: latency depends on network, API service, model version and environment.
  ```

- **Full detail:** every step is appended to `logs/<timestamp>_<mode>.jsonl`,
  including `state`, `decisions` (with `probabilities` and `confidence`),
  `latency_ms`, `executed_action`, `reasons`, and `next_state`. Use this for
  your research report.

Example JSONL record:

```json
{
  "step": 10,
  "timestamp": "2026-09-22T14:03:11",
  "state": {"ego_speed": 72, "ego_lane": 1, "front_distance": 18.0, "...": "..."},
  "questions": ["collision_risk", "rear_collision_risk", "risk_level", "longitudinal", "lateral"],
  "decisions": {
    "risk_noul": 0.82,
    "rear_noul": 0.15,
    "longitudinal": {"choice": "brake", "confidence": 0.94, "probabilities": {"brake": 0.94, "...": "..."}},
    "lateral": {"choice": "change_left", "confidence": 0.61, "probabilities": {"...": "..."}},
    "latency_ms": 142.0,
    "fallback": false
  },
  "latency_ms": 142.0,
  "executed_action": {"longitudinal": "brake", "lateral": "change_left"},
  "reasons": ["Risk noul 0.82 > 0.8; forcing brake"],
  "next_state": {"ego_speed": 62, "ego_lane": 0, "...": "..."}
}
```

---

## 9. Where the official API call lives

All official Jev / TypeSafe API usage is isolated in
[`src/jev_agent.py`](file:///d:/phyagentos/cvpr/jev-demo/src/jev_agent.py):

- `JevAgent.__init__` constructs the official `TypeSafeClient(api_key=..., model=..., timeout=...)`.
- `JevAgent.decide` builds the `Noul`/`Score`/`Choice` questions and calls
  `client.system_one(state=..., questions=...)` — **one request, five questions**.
- `_parse` reads `response.nouls` / `response.scores` / `response.choices`.
- API failures are caught (`TypeSafeAPIError`, `TypeSafeAPITimeoutError`, etc.)
  and turned into a conservative **fallback decision (brake + keep_lane)**.

The minimal standalone call is in
[`scripts/verify_jev.py`](file:///d:/phyagentos/cvpr/jev-demo/scripts/verify_jev.py).

---

## 10. Configuration & safety

All thresholds live in [`src/config.py`](file:///d:/phyagentos/cvpr/jev-demo/src/config.py)
and can be overridden by environment variables:

| Parameter | Default | Meaning |
|---|---|---|
| `RISK_THRESHOLD` | 0.8 | Noul collision risk above this forces brake. |
| `LANE_CHANGE_CONFIDENCE_THRESHOLD` | 0.35 | Lateral confidence below this → keep_lane. Lower = more willing to act on a weak lane-change signal. |
| `LONGITUDINAL_CONFIDENCE_THRESHOLD` | 0.5 | Longitudinal confidence below this → maintain. |
| `CRITICAL_FRONT_DISTANCE_M` | 8.0 | Gap at/under this forces brake (safety net). |
| `DT_S` | 1.0 | Time step for the distance update. |
| `NUM_LANES` | 3 | Road width (lane ids `0..N-1`). |
| `DRIVING_MODES` | comfort/normal/rush | Maps mode label → `target_speed` + `urgency` sent to Jev. `comfort`=45 km/h relaxed, `normal`=60 km/h, `rush`=80 km/h rushing. |
| `DEFAULT_DRIVING_MODE` | `normal` | Initial driving mode when the web demo starts. |

Safety rules (in `decision.py`): speed never goes below 0 or above the limit;
illegal lane changes become `keep_lane`; critical gaps force brake; and **API
failure always falls back to `brake`** — the agent never executes a dangerous
action because Jev was unreachable.

---

## 11. Tests

```bash
pytest -q
```

The tests cover the environment, validation, safety rules, confidence gating,
the fallback path, and an **offline end-to-end loop** (using a fake agent, no
API key needed). Real-API behaviour is verified by `scripts/verify_jev.py`.

---

## 12. Disclaimers

- This is a **Toy Environment**, not an autonomous-driving system.
- It is **not** a benchmark of Jev's official performance.
- Reported latency is a **local experiment** result; it depends on network, API
  service, model version and environment.
- The demo uses **structured state**, not vision/camera input.
- Where the official API differed from any assumption, the official current docs
  at <https://docs.typesafe.ai> were followed.
