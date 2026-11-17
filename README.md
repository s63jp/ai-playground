# AI Experimental Playground

[![tests](https://github.com/s63jp/ai-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/s63jp/ai-playground/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![mujoco](https://img.shields.io/badge/physics-MuJoCo_4.0.1-8A2BE2)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

A laboratory where autonomous AI agents live on a simulated 3D island in a
**real physics engine**, walk as an **articulated humanoid**, handle objects,
run, jump, fall, get back up, carry things, throw, catch, fight with equipment,
and keep written memory across sessions — then prove from recorded physics what
they actually learned.

It is designed to answer one question with evidence: **does an agent's written
experience actually make it better at things later?**

The simulator is independent of any AI provider. OpenAI, Claude, Qwen via LM
Studio or Ollama Cloud, fine-tunes, hand-written reference policies, or a human
with `curl` — all drive the same body through the same 65-tool structured
interface, and every action lands in the same SHA-256-chained action log.

> **November 2026 update — the agents got a body.**
> - `cd68da2` **Physics extension**: articulated humanoid (contact-driven
>   walking, PD joints, measured recovery), manipulation, projectiles +
>   equipment, multi-agent arena, human-player controller, RL environment,
>   16-skill catalogue. See the [phase tracker](docs/PHYSICS_EXTENSION.md).
> - `7731dae` **Phase 12 — autonomous agent integration**: the whole physical
>   surface is now agent-facing (10 new tools, skill composition, lifecycle
>   statuses). See [the contract](docs/AGENT_PHYSICAL_INTEGRATION.md).

---

## 1. The island, honestly

Numbers measured on this machine, not marketing. Sources: `scripts/engine_probe.py`
and the test suite (`run_tests.sh` — 547 passed, 3 skipped, Nov 2026).

| check | measured |
|---|---|
| bitwise determinism, two identical runs | equal SHA-256 of full state |
| 5-block tower + 50 kg body on top, 10 s | worst drift 0.24 mm |
| contact normal force under a 50 kg body | 490.50 N (weight 490.50 N) |
| jump apex vs analytic ballistics | 0.8115 m vs 0.8155 m |
| physics stepping speed | ~11,800 steps/s (benchmark) |
| dashboard / API serving | session-per-session replay from frames |

### Why MuJoCo

Phase 1 weighed Unreal Engine 5.8, Godot 4.7.2 and Blender 5.2.1 (installed) against
the requirements: deterministic physics, headless operation for many agents, exact
replay and session recovery. MuJoCo 4.0.1 won on measurements (one `pip install`, no
GPU). The dashboard renders the island in the browser with three.js from recorded
frames; Unreal/Godot remain options for a visual client later.

---

## 2. The physical body

Two rigs share the same world, tool interface and action log
(`config/default.yaml: agent_body.type: box | humanoid`):

**The box** — a non-rotating root on three slide joints with a force-limited walking
controller. Everything obeys Newton's third law: carried mass is added to the body,
a throw recoils the body, pushing off a loose object pushes that object.

**The humanoid** (November 2026) — an articulated rig built by the world's own MJCF
generator: waist yaw, hips, knees, ankles, feet, spine, head, arms — PD-controlled
through a single torque application point. No teleports, ever: no qpos/qvel writes
on any agent path.

| humanoid capability | measured |
|---|---|
| contact-driven walking (root traction disabled — legs do the work) | 0.80–0.96 m per 8 s, z held at 0.79 m, heading 92° |
| speed scaling | displacement-based, monotone in requested speed |
| heading control | walk a heading (±30° verified) |
| falling | the fold-sit at ~0.42 m is the rig's honest fall mode; `agent_fallen()` fires after a 0.3 s sustained window |
| recovery | `stand_up()` drives a measured ROLL → CATCH → EXIT joint schedule; `recovery_complete` is only emitted after upright **sustains** 0.5 s |
| jumping | a force rocket on the payload (rig fact: the column has no stride-jump; documented cap) |
| carrying | a hold pin in front of the chest; the load is added to body mass and turns with the figure |

### Rig facts agents discover (and we publish, not hide)

* Turning happens **through the stride** — the waist twists against planted-foot
  friction while striding. `turn_to` therefore walks a short real arc; there is no
  in-place turn servo on this rig.
* Long straight walks **drift off-course** (waist twist vs foot friction); the world
  reports `off_course` with the measured bearing error instead of pretending.
* Deep leg-fold crumples **stay wedged** (an armless column cannot push off) —
  recovery is honestly `recovery_failed` beyond the recoverable sit pose.
* A 12 kg rock **rolls** when walked into; loose 0.5 m cubes get shoved. The walk
  verdicts are measured: `blocked` (stalled), deflected-off-course (reported), or
  completed.
* The arm reach is 1.5 m horizontal / 2.1 m up; `reach_for` aims the arm with
  measured joint tracking and reports the **measured shortfall** — never "reached"
  because a return value happened.

---

## 3. What agents can actually do — 65 tools, honest outcomes

Every tool call: validated → executed in MuJoCo → logged with its physics events
(hash-chained) → answered with the **measured** result and a lifecycle status:

| status | meaning |
|---|---|
| `completed` | the world measured the requested outcome |
| `partial` | made measured progress, then was stopped (blocked walk carrying `distance_travelled`; a chain that finished some steps before failing) |
| `failed` | no progress evidence (invalid argument, unknown object, recovery_failed, …) |

Never reported success because a function returned.

Highlights by category (see `docs/TOOLS.md` and `playground/interface/schema.py`
for the full catalogue):

* **observation** — look_around, inspect_object, inspect collision records, world
  state incl. `fallen`, held object, carried mass, facing
* **movement** — `move`, `walk_to`, `run_to`, `turn_to`, `stop_moving`, `stand`,
  `stand_up`, `climb`, `jump`, `wait`, `set_joint_target` (PD control)
* **objects** — `pick_up`, `put_down`, `place`, `throw` (release-velocity physics),
  `catch`, `push/pull/slide/nudge/kick/roll`, `rotate_object`, `stack_on`,
  `build_row`, `reach_for`, `carry_to`
* **skills** — `run_skill` / `run_chain`: the 16-skill physical catalogue
  (standing, walking, running, turning, jumping, landing, balancing, falling,
  recovery, reaching, grasping, lifting, carrying, throwing, catching, climbing)
  dispatched onto the *existing* world implementations, composable in order with a
  per-step ledger:

  ```
  run_chain(steps=[
      {"skill": "grasping", "args": {"object_id": "block_01"}},
      {"skill": "carrying", "args": {"direction": "west", "distance": 0.4}},
  ])
  # → ledger: [succeeded, succeeded] · carried block displaced, measured
  ```

* **experiment** — measure_height/distance/velocity/flight-time, physics sandbox
  (`run_physics_test`), inspect_world_history
* **world** — saved programs (`save_program`/`execute_program`, compose actions),
  inspect_recent_events
* **memory** — diary, notes, evidence-gated discoveries, goals, achievements;
  statuses granted by engine-checked evidence, never by the author
* **learning** — `nn_list/nn_make_dataset/nn_train/nn_predict/nn_evaluate`:
  agent-operable neural controllers built from its own recorded behaviour
  (`docs/ML_LAB.md`)

---

## 4. Modes beyond free exploration

* **Multi-agent arena** (`playground/arena/`): matches, scoring, configured modes,
  result reports — agents meet under rules and the referee reads physics, not
  intentions.
* **Combat** (`playground/combat/`): projectile payloads, measured impact/splat FX,
  an equipment registry (grip, ammo, spread) with a frozen launcher signature.
* **Human players** (`playground/player.py` + dashboard Player pane): a human joins
  the same world through the same tool contract — the dashboard serves the live
  humanoid skeleton (23 bodies) and agent state over `/api/player/state`.
* **RL** (`playground/rl/`): the authoritative locomotion environment
  contract (14-action / 43-dimensional observation `rl_env/3` — v2 plus the
  commanded target velocity, `docs/RL_ENVIRONMENT.md`), one consolidated
  reward pipeline with a measured anti-exploit structure, a deterministic
  env-side curriculum, and (B3) a genuine actor-critic PPO trainer trained
  only from real MuJoCo rollouts (`playground/rl/ppo.py`,
  `scripts/rl_ppo.py train`, seeded checkpoints in `experiments/rl/`). The
  scripted gait remains a benchmark, never a learned policy.

---

## 5. Architecture

```
            ┌──────────────────────── agents/adapters.py ──────────────────────┐
  AI brain  │ OpenAICompat (OpenAI, LM Studio, Ollama Cloud, vLLM…) · Anthropic  │  ← provider-neutral
            │ Policy/scripted · Random · External (HTTP)                        │
            └──────────────┬────────────────────────────────────────────────────┘
                           │ tool calls (JSON)                ▲ structured results
            ┌──────────────▼──────────── interface/ ──────────────────────────┐
            │ schema.py        65 tools, JSON-schema, argument validation      │
            │ environment.py   single entry point: validate → execute →        │
            │                  log action + events (SHA-256 chain) → answer    │
            │                  + lifecycle status on every result              │
            │ physical.py      observation / movement / objects / measurement  │
            └───┬───────────────────┬───────────────────┬─────────────────────┘
                │                   │                   │
   ┌────────────▼────────┐ ┌────────▼───────────┐ ┌─────▼───────────────────┐
   │ world/ (MuJoCo)     │ │ skills/            │ │ memory/store.py         │
   │ spec.py island +    │ │ SkillSpec catalog  │ │ per-agent files,        │
   │   humanoid MJCF     │ │ SkillsLibrary +    │ │ evidence-gated status,  │
   │ physics.py PD joints│ │ AgentSkillAdapter  │ │ audit trail, ablations  │
   │  gait API, recovery,│ │ (16 skills →       │ └─────────────────────────┘
   │  carry pin, throw,  │ │  existing impls)   │ ┌───── integrity/ ────────┐
   │  climb, catch,      │ └────────────────────┘ │ evidence.py claims vs   │
   │  flights, frames    │ ┌────────────────────┐ │  logged measurements    │
   │  checkpoints        │ │ manipulation.py    │ │ claims.py towers, spire,│
   │  hidden world rule  │ │ 2-link reach IK +  │ │  juggling… measured     │
   └─────────────────────┘ │ contact-gated grip │ │  from physics, never    │
                           └────────────────────┘ │  from the agent         │
   ┌──────────── session/ ──────────────┐ └───────┬─────────────────────────┘
   │ runner.py  brief → act → save →    │ ┌───────▼─────────────────────────┐
   │            reset; integrity report │ │ storage/db.py (SQLite):         │
   │ manager.py 13 agents, seq/conc     │ │ sessions, actions, events,      │
   │ benchmark  Group A/B, ablation     │ │ frames, messages, records,      │
   └────────────────────────────────────┘ │ checkpoints, benchmarks (SHA-256│
   ┌──────── rl/ + combat/ + arena/ ────┐ │  append-only chain)             │
   │ 14-DoF env · projectiles · matches │ └─────────────────────────────────┘
   └────────────────────────────────────┘ ┌─── dashboard/ (FastAPI + three.js) ─┐
                                          │ 3D humanoid viewer, replay, memory  │
                                          │ editor, analytics, Player pane,     │
                                          │ 13-agent compare, ML tab, HTTP API  │
                                          └─────────────────────────────────────┘
```

### Key design decisions

* **Time only moves while an action runs.** Sessions are replayable bit for bit
  from the action log; checkpoints restore state exactly, including flights in
  progress (tested).
* **MuJoCo is authoritative.** No script or agent path writes qpos/qvel. Movement
  emerges from contact, force caps and PD control. Skill/composition layers can
  only *drive* the existing controllers — never bypass them.
* **Nothing about the physics is told to the agent.** Tool descriptions say what
  a tool does, never what will happen. The default exploration prompt is
  `free_agency_v3` (strictly non-directive; prompt recorded per session with a
  hash). Tests check the prompt for leaks.
* **The agent cannot certify itself.** Memory statuses (`hypothesis → probable →
  confirmed → disproven`) are granted by evidence checks against the engine's
  action log. Removed records are tombstoned, never deleted. The hidden
  **world rule** is published only in `docs/RESEARCHER_NOTES.md`.
* **Lifecycle honesty everywhere** (Phase 12): every result says
  `completed / partial / failed`; chains mark interrupted steps; movement error
  payloads carry what was actually measured (`distance_travelled`,
  `off_course`, `blocked_by`, remaining distances).

---

## 6. Project directory

```
ai-playground/
  config/default.yaml        every experimental parameter (hashed per session)
  playground/
    config.py                config loading, validation, hashing, agent roster
    world/spec.py            island spec, humanoid MJCF generation
    world/physics.py         MuJoCo: bodies, PD joints, gait API, recovery, carry,
                             throw/climb/catch, flights, frames, checkpoints
    locomotion.py            GaitController (walk/run/stand/crouch/recover)
    manipulation.py          2-link reach IK + contact-gated grasp/push/pull
    skills/                  SkillSpec, library (registry/invoke), 16-skill
                             catalogue, agent adapter, capability protocol
    interface/schema.py      65-tool catalogue (OpenAI + Anthropic formats)
    interface/physical.py    tool wrappers + skill/run_chain surface
    interface/environment.py entry point, lifecycle statuses, programs, sandbox
    memory/store.py          memory files, evidence gates, audit, ablations
    integrity/               evidence + independent achievement verifiers
    combat/                  projectiles, impacts, equipment
    arena/                   matches, scoring, modes
    rl/                      env (14 DoF), rewards + anti-exploit, curriculum
    player.py                human player controller
    dashboard/               FastAPI server + three.js console + Player API
    ml/                      dataset builders, trainers, controllers (agent nn_*)
    session/                 runner, manager, benchmark, model_compare
    agents/                  adapters (OpenAI compat, Anthropic, policy, external)
    cli.py / demo.py
  tests/                     547-test suite: HVE (Happy/Validate/Error), physics,
                             locomotion, manipulation, skills, RL, integrity, UI… 
  scripts/                   health_check.py, engine_probe.py, profile_step.py,
                             trace_session.py, audit_agents.py
  docs/                      OPERATING.md · TOOLS.md · RESEARCHER_NOTES.md ·
                             PHYSICS_EXTENSION (+CONTRACT) · AGENT_PHYSICAL_
                             INTEGRATION.md · BUGLOG_2026-11-17 (+_PHASE12) ·
                             ML_LAB.md · ML_INTERFACE.md · GATEWAY.md
  run_tests.sh               full suite + health check
```

---

## 7. Installation

Requirements: Python 3.11+ (3.13 used here). No GPU required.

```bash
git clone https://github.com/s63jp/ai-playground.git
cd ai-playground
uv venv --python 3.13 .venv          # or: python -m venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m playground init      # experiments/ db + 13 agents
bash run_tests.sh                                # suite + health check
```

Dependencies: `mujoco numpy fastapi uvicorn httpx pyyaml pytest`. That is the whole list.

---

## 8. Workflows

`PY=.venv/Scripts/python.exe` (POSIX shells: `.venv/bin/python`)

**Look at it**

```bash
$PY -m playground serve          # dashboard + HTTP API → http://127.0.0.1:8765/
$PY -m playground demo           # end-to-end scripted demonstration (~10 s)
```

**Run agents**

```bash
$PY -m playground explore --agent agent_001 --adapter lmstudio            # local model
$PY -m playground explore --agent agent_001 --adapter ollama_cloud --model kimi-k3
$PY -m playground explore --agent agent_001 --adapter claude               # ANTHROPIC_API_KEY
$PY -m playground run-agents --sessions 3 --concurrency 4 --adapter lmstudio
```

A run with an unreachable or unauthenticated endpoint surfaces as an **adapter
error, never a fake success**.

**Read what agents know**

```bash
$PY -m playground memory show agent_001
$PY -m playground memory file agent_001 handbook
$PY -m playground messages <session_id>      # full transcript + actions
$PY -m playground verify-log <session_id>    # event hash-chain audit
```

**Compare models fairly** (identical conditions, profiled)

```bash
$PY -m playground models                     # what an openai_compat server offers
$PY -m playground compare-models --adapter lmstudio --models q1 q2
```

**Test the memory hypothesis** (the lab's core question)

```bash
$PY -m playground benchmark --group-a agent_001,agent_002,agent_003 --adapter lmstudio
$PY -m playground ablate agent_001 --remove D-001 --tasks spire_landing,throw_height
```

**Physical / skills surface** — what Phase 12 gave agents:

```bash
$PY -m pytest tests/test_agent_physics_integration.py -q   # lifecycle statuses + tools
$PY -m pytest tests/test_locomotion.py tests/test_manipulation.py -q
.venv/Scripts/python.exe scripts/demo_humanoid_walk.py      # live walk trace
```

**Neural controllers** (offline or agent-driven; cpu / xpu / auto)

```bash
$PY -m playground ml status
$PY -m playground ml generate --dataset throws --agent agent_001
$PY -m playground ml train --dataset throws --device xpu
$PY -m playground datasets                  # what the lab can build
```

---

## 9. Verification gates

The project holds itself to a hard verification gate:

1. **HVE suite** — `tests/test_hve.py` (Happy / Validate / Error paths) plus a
   smoke suite; new or changed behaviour gets coverage **first**.
2. **Real runs shown** — physical claims are backed by measurements (displacement,
   event streams, joint angles), never by a function returning.
3. **Independent sign-off** — `run_tests.sh` runs the whole suite then
   `scripts/health_check.py`; current state: **547 passed, 3 skipped** +
   `{"healthy": true}`.
4. **Dashboard boot smoke** — server boots, `/api/health` ok, tools catalogue
   served (65), humanoid skeleton (23 bodies) on the Player API; all Phase-12
   tools verified present over HTTP.

---

## 10. Honest status

* **Real models drove this.** LM Studio with Qwen3.5-9B ran exploration sessions
  end to end during development. Adapters for Claude, OpenAI and Ollama Cloud are
  in-repo; credential problems surface as adapter errors.
* **The scripted reference agent is a hand-written policy** used for tests, the
  demo and pipeline validation. Its "discoveries" are real measurements through
  the tool interface and its later-session behaviour depends on its memory files
  — but its *curiosity* is authored. Research conclusions need real-model runs.
* **Learning-from-memory is the open question, not a claim.** The ML lab and the
  evidence machinery demonstrate trained neural controllers, and the
  benchmark/ablation pipeline measures whether experienced agents beat fresh
  ones — but no LLM agent has yet completed train → deploy → verify on its own.
* **The rig has honest limits, documented, not hidden**: no in-place turn
  (turning rides the stride); long walks drift (measured, reported); deep leg
  crumples stay wedged; jump is a payload rocket, not a stride leap. See
  `docs/BUGLOG_2026-11-17.md` and `docs/BUGLOG_2026-11-17_PHASE12.md` for the
  measured bugs and fixes behind the current surface.
* A large 13-agent × 60-minute study with a real model is an operational run you
  start — it is not pre-recorded here.

---

## 11. Documentation

| doc | what it covers |
|---|---|
| [docs/OPERATING.md](docs/OPERATING.md) | every command, HTTP API for external agents, config, recovery |
| [docs/TOOLS.md](docs/TOOLS.md) | the tool catalogue as agents see it |
| [docs/PHYSICS_EXTENSION.md](docs/PHYSICS_EXTENSION.md) | phase tracker P1–P12 with verification commands |
| [docs/PHYSICS_EXTENSION_CONTRACT.md](docs/PHYSICS_EXTENSION_CONTRACT.md) | frozen interfaces + ownership from the physics fan-out |
| [docs/AGENT_PHYSICAL_INTEGRATION.md](docs/AGENT_PHYSICAL_INTEGRATION.md) | Phase 12: architecture, contract, new tool surface |
| [docs/BUGLOG_2026-11-17.md](docs/BUGLOG_2026-11-17.md) | measured gait/recovery findings behind the walk |
| [docs/BUGLOG_2026-11-17_PHASE12.md](docs/BUGLOG_2026-11-17_PHASE12.md) | agent-integration bugfixes with live evidence |
| [docs/ML_LAB.md](docs/ML_LAB.md) / [docs/ML_INTERFACE.md](docs/ML_INTERFACE.md) | machine-learning lab + agent `nn_*` tools |
| [docs/GATEWAY.md](docs/GATEWAY.md) | the model-gateway HTTP surface |
| [docs/RESEARCHER_NOTES.md](docs/RESEARCHER_NOTES.md) | the deliberately hidden world rule (researcher eyes) |

## License

MIT — see [LICENSE](LICENSE).
