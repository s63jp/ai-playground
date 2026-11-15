# AI Experimental Playground

[![tests](https://github.com/s63jp/ai-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/s63jp/ai-playground/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A laboratory where autonomous AI agents live on a small simulated 3D island, handle objects, discover
how the physics behaves, set their own goals, build things and keep written memory across sessions.
It is designed to answer one question with evidence: **does an agent's written experience actually
make it better at things later?**

It is inspired by the MIT "Eko" autonomous-exploration experiment. The simulator is independent of any AI
provider. Claude, GPT models, Qwen, Hermes, local LM Studio/Ollama models, fine-tunes, or a human with
`curl` all drive the same body through the same structured tool interface.

---

## 1. Architecture

```
            ┌──────────────────────── agents/adapters.py ───────────────────────┐
  AI brain  │ OpenAICompat (OpenAI, LM Studio, Ollama, vLLM…) · Anthropic ·      │  ← provider-neutral
            │ Policy/scripted · Random baseline · External (HTTP)                │
            └──────────────┬─────────────────────────────────────────────────────┘
                           │ tool calls (JSON)                ▲ structured results
            ┌──────────────▼──────────── interface/ ─────────────────────────────┐
            │ schema.py   40 tools, JSON-schema, argument validation              │
            │ environment.py  the single entry point: validate → execute →        │
            │               log action + physics events (hash-chained) → answer   │
            │ physical.py observation / movement / objects / measurement          │
            └───┬─────────────────────┬──────────────────────┬────────────────────┘
                │                     │                      │
   ┌────────────▼───────┐  ┌──────────▼──────────┐  ┌────────▼──────────────────┐
   │ world/ (MuJoCo)    │  │ memory/store.py      │  │ integrity/               │
   │ spec.py  island    │  │ per-agent files,     │  │ evidence.py: experiment  │
   │ physics.py body,   │  │ evidence-gated       │  │  claims vs engine values │
   │  carry, throw,     │  │ status, audit trail, │  │ claims.py: towers, spire,│
   │  climb, events,    │  │ ablation copies      │  │  juggling… measured from │
   │  flights, frames,  │  └──────────────────────┘  │  physics, never the agent│
   │  hidden world rule │                            └──────────────────────────┘
   └────────────────────┘
   ┌─────────────── session/ ───────────────┐  ┌──── storage/db.py (SQLite) ────────┐
   │ runner.py   brief → act → save → reset │  │ sessions, actions, events (append- │
   │ manager.py  13 agents, seq/concurrent  │  │ only + SHA-256 chain), frames,     │
   │ benchmark.py Group A/B, ablation       │  │ messages, records, checkpoints,    │
   └────────────────────────────────────────┘  │ benchmark results, memory audit    │
   ┌──── analysis/behaviour.py ─────┐          └────────────────────────────────────┘
   │ 4 behaviour categories from    │  ┌── dashboard/ (FastAPI + three.js) ─────────────┐
   │ recorded actions only          │  │ 3D viewer + replay, activity, memory editor,    │
   └────────────────────────────────┘  │ analytics, 13-agent comparison, benchmarks, API │
                                        └─────────────────────────────────────────────────┘
```

### Why MuJoCo and not the installed engines

Phase 1 found Unreal Engine 5.8, Godot 4.7.2 and Blender 5.2.1 on this machine, plus an Intel Arc A770 with no
CUDA. The deciding requirements were deterministic physics, headless operation for many agents, exact
replay and session recovery. MuJoCo 4.0.1 (one `pip install`, no GPU needed) was measured in
`scripts/engine_probe.py`:

| check | measured |
|---|---|
| bitwise determinism, two identical runs | equal SHA-256 of full state |
| 5-block tower + 50 kg body on top, 10 s | worst drift 0.24 mm |
| contact normal force under the body | 490.50 N (weight 490.50 N) |
| jump apex vs analytic ballistics | 0.8115 m vs 0.8155 m |
| speed with the island loaded | ~20× real time per agent (one CPU core) |

Unreal and Godot would add a renderer, an editor process and non-deterministic stepping without improving
the simulation. They remain an option for a nicer visual client later. The dashboard renders the island in
the browser with three.js, from recorded frames.

### Key design decisions

* **Time only moves while an action runs.** A session is fully reproducible from its action log.
  Checkpoints restore bit for bit (tested), including flights in progress.
* **The body** is a non-rotating box on three slide joints with a force-limited walking controller.
  Everything it does obeys Newton's third law: carried mass is added to the body, a throw recoils the body,
  and pushing off a loose object pushes that object.
* **Nothing about the physics is told to the agent.** Tool descriptions say what a tool does, never what
  will happen. The exploration prompt is the personality text from the brief verbatim plus how the tools
  and memory work (`agents/prompts.py`). A test checks the prompt for leaks.
* **Prompt versioning:** The default exploration prompt is `free_agency_v2` (researcher-supplied, Nov 2026).
  The original brief personality `brief_v1` is selectable via `agents.exploration_prompt` in config. Every
  session records `prompt_id` and `system_prompt_hash` so results from different prompts are never mixed.
* **Discoverable mechanics** that emerge from the physics: velocity at release adds to a throw (jump-throws
  go higher, running throws go further), carried mass lowers jumps, throws recoil the body, stacked blocks
  act as steps, the terrace needs a step to reach, objects can be caught in flight. One deliberately hidden
  **world rule** is described only in `docs/RESEARCHER_NOTES.md`.
* **The agent cannot certify itself.** Memory statuses are granted by an evidence check against the engine's
  action log, and achievements are verified from the world state and the physics event stream.

---

## 2. Project directory

```
ai-playground/
  config/default.yaml          every experimental parameter (hashed into each session record)
  playground/
    config.py                  config loading, validation, hashing
    world/spec.py              the island: objects, dimensions, MJCF generation
    world/physics.py           MuJoCo wrapper: body, carry, throw, climb, catch, events, flights, frames,
                               checkpoints, hidden rule, researcher-only scene setup
    world/mathutil.py          directions, quaternions
    interface/schema.py        the 40-tool catalogue (OpenAI + Anthropic formats) and argument validation
    interface/physical.py      observation / movement / object / measurement tools
    interface/environment.py   the single entry point; logging; programs; sandbox; memory tools; claims
    memory/store.py            agent memory files, evidence-gated status, admin ops, ablation copies
    integrity/evidence.py      checks experiment records against logged measurements; flight plausibility
    integrity/claims.py        independent achievement verifiers (tower, object_on, juggle, …)
    analysis/behaviour.py      the four behaviour categories
    session/runner.py          session lifecycle, reset, recovery, integrity report
    session/manager.py         many agents, sequential or concurrent
    session/benchmark.py       benchmark tasks + verifiers, Group A/B, ablation
    agents/adapters.py         provider adapters
    agents/prompts.py          prompts (personality text verbatim)
    agents/scripted.py         hand-written reference policy (NOT an AI; for tests and the demo)
    dashboard/server.py        FastAPI: external-agent API, researcher API, dashboard
    dashboard/static/          dashboard (three.js vendored, works offline)
    demo.py                    the end-to-end demonstration
    cli.py                     `playground …` commands
  tests/                       conftest.py, test_hve.py (Happy/Validate/Error), test_smoke.py,
                               test_regression_extra.py (prompt recording, determinism, etc.)
  scripts/                     health_check.py, engine_probe.py, trace_session.py, audit_agents.py
  experiments/                 created at run time: playground.db, agents/agent_001 … agent_013, ml/
  docs/                        OPERATING.md, RESEARCHER_NOTES.md, TOOLS.md, ML_LAB.md, ML_INTERFACE.md
  run_tests.sh
```

---

## 3. Installation

Requirements: Python 3.11+ (3.13 used here). No GPU required.

```bash
cd <repo>
uv venv --python 3.13 .venv          # or: python -m venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m playground init      # creates experiments/playground.db + 13 agent folders
bash run_tests.sh                                # test suite + health check
```

Dependencies: `mujoco numpy fastapi uvicorn httpx pyyaml pytest`. That is the whole list.

---

## 4. Quick start

```bash
PY=.venv/Scripts/python.exe

# Launch the world viewer / dashboard (read-only; never interferes with agents)
$PY -m playground serve                     # http://127.0.0.1:8765/

# The demonstration (separate workspace experiments/demo; takes ~10 s)
$PY -m playground demo

# One 60-minute free exploration session with a real model
$PY -m playground explore --agent agent_001 --adapter lmstudio          # LM Studio
$PY -m playground explore --agent agent_001 --adapter claude            # needs ANTHROPIC_API_KEY
$PY -m playground explore --agent agent_001 --adapter openai --model gpt-5   # needs OPENAI_API_KEY
$PY -m playground explore --agent agent_001 --adapter ollama_cloud --model kimi-k3   # needs OLLAMA_API_KEY
$PY -m playground explore --agent agent_001 --adapter lmstudio_unsloth4b  # researcher's Unsloth Qwen 4B

# All 13 agents, 3 sessions each, 4 at a time
$PY -m playground run-agents --sessions 3 --concurrency 4 --adapter lmstudio

# Inspect memory
$PY -m playground memory show agent_001
$PY -m playground memory file agent_001 handbook

# Benchmarks
$PY -m playground benchmark --group-a agent_001,agent_002,agent_003 --adapter lmstudio
$PY -m playground ablate agent_001 --remove D-001 --tasks spire_landing,throw_height --adapter lmstudio
```

`docs/OPERATING.md` covers every command, the HTTP API for external agents, configuration, recovery,
and how to read the results. `docs/ML_LAB.md` documents the machine learning infrastructure for
neural controllers. Intel XPU notes are in `reports/XPU_DIAGNOSIS.md`.

---

## 5. Honest status

See **Known limitations** in `docs/OPERATING.md`. In short:

* The **scripted reference agent** is a hand-written policy used for tests, the demonstration and pipeline
  validation. Its "discoveries" are real measurements made through the tool interface, and its later-session
  behaviour depends on what its memory files contain. Its *curiosity* is authored, though, so it says
  nothing about AI behaviour. Research conclusions need runs with real models.
* Real-model runs need a reachable endpoint. During development, LM Studio with Qwen3.5-9B was verified driving
  sessions end to end. The **lmstudio_unsloth4b** adapter (researcher's custom Unsloth Qwen 4B) was explored
  but recorded no experiments during the evaluation period. Ollama Cloud returned HTTP 402 (no credits) and
  was reported as an adapter error, not a fake success.
* **Learning-from-memory not yet demonstrated:** While the ML lab infrastructure supports agent-driven training,
  no LLM-controlled agent has successfully completed the train→deploy→verify loop.
* A full 13-agent × 60-minute study with a real model is an operational run you start; it is not
  pre-recorded here.
