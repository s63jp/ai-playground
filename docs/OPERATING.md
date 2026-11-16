# Operating manual

All commands are run from the project folder with the project's Python:

```bash
cd /path/to/ai-playground      # wherever you cloned the repo
PY=.venv/Scripts/python.exe
```

`--config FILE` (before the subcommand) merges a YAML file over `config/default.yaml`, e.g. to point at a
different workspace (`_root: D:/study_2`) or change the session length. The SHA-256 of the physics-relevant
configuration is stored with every session.

---

## 1. Launch the world and watch it

```bash
$PY -m playground serve               # http://127.0.0.1:8765/   (API docs at /docs)
```

The dashboard only reads the database, so it can be opened, closed or refreshed while agents are running in
another process without affecting them.

| Tab | Shows |
|---|---|
| **Watch & replay** | 3D island (orbit with the mouse), agent body (blue, white cone on top), every object, recorded throw trajectories (dashed), a replay scrubber with play speed. Beside it: the agent's own words, recent actions, the self-selected objectives with their independent verification, experiments with their evidence verdicts, and an *integrity* table (claimed vs verified). "follow live" jumps to whatever session is running. |
| **Memory** | Every memory item with researcher controls (**challenge**, **mark unverified**, **remove**), the immutable audit trail, and the raw files (handbook.md, diary.md, mistakes.md, unfinished_goals.md, experiments.json, programs/). |
| **Analytics** | Session timeline and a per-session table: actions, failures, experiments passing/failing evidence checks, verified/rejected claims, handbook state, the four behaviour categories. |
| **Compare agents** | All 13 agents side by side: discoveries by status, memory quality (share of experiments consistent with the engine), mistakes, programs, verified claims, behaviour totals, benchmark completion, plus everything each agent currently believes. |
| **Benchmarks** | Every Group A/B and ablation run, per task and group. |

Replay is available for every session ever run (frames at 20 Hz are stored with periodic keyframes).

---

## 2. Connect an AI agent

### 2a. Built-in adapters (the simulator calls the model)

Adapters are defined in `config/default.yaml` under `adapters:`. Secrets are never stored in the config:
`api_key_env` names an environment variable.

| name | type | default model | needs |
|---|---|---|---|
| `lmstudio` | openai_compat | `qwen/qwen3.5-9b` at `http://127.0.0.1:1234/v1` | LM Studio server running with that model loaded |
| `lmstudio_unsloth4b` | openai_compat | `unslothqen 4b` at `http://127.0.0.1:1234/v1` | researcher's custom Unsloth Qwen 4B Q8_0 |
| `ollama_cloud` | openai_compat | `kimi-k3` | `OLLAMA_API_KEY` with credit |
| `openai` | openai_compat | `gpt-5` | `OPENAI_API_KEY` |
| `claude` | anthropic | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `scripted` | reference policy | – | nothing (not an AI) |
| `random` | random actions | – | nothing (baseline) |

Any OpenAI-compatible server works (vLLM, llama.cpp `--api`, Ollama local at `http://127.0.0.1:11434/v1`,
OpenRouter, a fine-tune behind a proxy): add an entry with `type: openai_compat`, `base_url`, `model`, and
optionally `api_key_env`, `temperature`, `max_tokens`, `timeout`, `max_history_chars`, `extra_body`.
`--model` overrides the model on the command line.

LM Studio from the command line:
```bash
~/.lmstudio/bin/lms server start
~/.lmstudio/bin/lms load qwen/qwen3.5-9b --gpu max --context-length 32768 -y
```

### 2b. External agents over HTTP (the agent calls the simulator)

For Hermes, custom agent loops, or anything that wants to stay in control of its own conversation.
Start `playground serve`, then:

```bash
# start a session: returns system_prompt, opening_message (the memory briefing) and the tool schemas
curl -s -X POST localhost:8765/api/agent/sessions -H 'Content-Type: application/json' \
     -d '{"agent_id":"hermes_001","model":"kimi-k3"}'
# -> {"session_id": 7, "system_prompt": "...", "opening_message": "...", "tools": [...], ...}

# act (optionally record what the agent said with "say")
curl -s -X POST localhost:8765/api/agent/sessions/7/call -H 'Content-Type: application/json' \
     -d '{"tool":"jump","args":{"wait":"landing"},"say":"Let me see how high I can jump."}'

# finish: saves memory, snapshots it into session_history/, discards the world (reset)
curl -s -X POST localhost:8765/api/agent/sessions/7/finish -H 'Content-Type: application/json' -d '{}'
```

Benchmark tasks can be run externally too: `{"agent_id": "...", "mode": "benchmark", "task_id":
"spire_landing"}`. On `finish` the server verifies the task itself and returns `task_result`.

`GET /api/tools` returns the tool catalogue. Every tool result is
`{"success": bool, "action": name, "action_index": n, ...}`. Failures carry
`{"error": {"code", "message"}, "details": {...}}`.

---

## 3. Start a 60-minute exploration session

```bash
$PY -m playground explore --agent agent_001 --adapter lmstudio
```

* The default length is 60 minutes of wall-clock time (`session.duration_minutes`). `--minutes 15`
  shortens it. `--max-steps N` adds a cap on agent actions, which gives reproducible budgets across
  models of different speeds.
* The agent receives the personality instruction verbatim, a description of its situation and tools, and
  its own memory files read back from disk. There is no task, no score and no reward. The default
  exploration prompt is `free_agency_v2` (researcher-supplied, Nov 2026); the original brief personality
  `brief_v1` is selectable via `agents.exploration_prompt`. Every session records the `prompt_id` and
  `system_prompt_hash` it ran with.
* Five minutes before the end (or 8 actions before a step cap) the agent is told time is running out. When
  time is up, physical tools stop working and the agent gets 12 more calls to write to its memory.
* The agent can stop at any time with `end_session`. A model that replies three times without acting is
  treated as having stopped.
* At the end, everything is saved, the memory files are copied to
  `session_history/session_NNN/` together with `final_world_state.json` and `summary.json`, and the world
  is discarded. The next session builds a fresh island from the specification, and the summary records the
  state hash it will start from.

**Recovery.** If the model endpoint fails or the process is interrupted, the session is marked
`interrupted`. Resume it with:
```bash
$PY -m playground explore --agent agent_001 --adapter lmstudio --resume <session_id>
```
The world is restored from the last checkpoint (every 10 actions) and the remaining logged physical
actions are replayed deterministically. The resume report checks the rebuilt state hash against the one
logged at the time (`state_matches_log`). The model is told the session was resumed.

**Orphans.** Every session-starting command (`explore`, `run-agents`, `benchmark`, `ablate`, `serve`)
reconciles the database at start: sessions still marked `running` from a dead process are marked
`interrupted` (the resumable state) before anything new begins, so a killed run can never leave a
permanent "running" ghost in the record.

---

## 4. Run 13 agents

```bash
$PY -m playground run-agents --sessions 3 --concurrency 1 --adapter lmstudio   # sequential
$PY -m playground run-agents --sessions 3 --concurrency 4 --adapter claude     # 4 at a time
$PY -m playground run-agents --agents agent_001,agent_005 --sessions 2 --adapter lmstudio
```

Each agent has its own memory directory, its own sessions and its own freshly built island. Nothing is
shared during exploration. Every agent gets the same starting island and the same personality text.
Concurrency is only a scheduling choice: each agent's physics runs on its own world object, and a single
local GPU model will just serialise the requests. Use `--concurrency 1` for LM Studio and higher values
for hosted APIs.

---

## 5. Inspect persistent memories

```
experiments/agents/agent_001/
  diary.md  handbook.md  mistakes.md  unfinished_goals.md     rendered, human-readable
  discoveries.json  experiments.json  state.json              canonical data
  programs/*.json                                              saved programs
  session_history/session_001/ …                               snapshot after each session + final world state
  audit.log                                                    every researcher intervention
```

```bash
$PY -m playground memory show agent_001            # all items with ids and statuses
$PY -m playground memory file agent_001 handbook   # print a rendered file
```

**Epistemic status.** The handbook has four sections: *verified*, *probable*, *untested hypotheses* and
*disproven*. The agent asks for a status, and the system grants the one its evidence supports:

* An experiment record cites `action_index` numbers. Its claimed measurements are compared with the values
  the engine returned for those actions. The verdict is `consistent` (numbers match), `grounded` (valid
  physical actions but no matching number), `contradicted` (a number disagrees), `invalid_references`, or
  `implausible`.
* `confirmed` needs **two independent experiments with engine-matched measurements** (citing the same
  actions twice counts once). `probable` needs one checked supporting experiment. With no checked evidence
  an entry stays a *hypothesis*, whatever the agent asked for. `disproven` needs a checked contradicting
  experiment.

**False beliefs and memory quality (Phase 9).**
```bash
$PY -m playground memory challenge agent_001 D-002 --reason "Contradicts agent_004's measurements"
$PY -m playground memory unverify  agent_001 D-002 --reason "re-test from scratch"
$PY -m playground memory remove    agent_001 D-002 --reason "ablation"
```
* **challenge** flags the entry in the handbook ("CHALLENGED … re-test before relying on it"). It clears when
  a new checked experiment settles it.
* **unverify** demotes the entry to *hypothesis*. Evidence recorded before the reset no longer counts.
* **remove** deletes the item. Removing an experiment re-evaluates every discovery that cited it.

Every operation is recorded in `audit.log` and in the immutable `memory_audit` table. The same controls
are in the dashboard's Memory tab.

---

## 6. Benchmarks and ablation

```bash
# Group A = agents with exploration memory, Group B = fresh agents (empty memory), identical tasks
$PY -m playground benchmark --group-a agent_001,agent_002,agent_003 --adapter lmstudio --repeats 2

# Ablation: the same experienced agent, full memory vs memory with chosen items removed
$PY -m playground ablate agent_001 --remove D-001 --tasks spire_landing,throw_height --adapter lmstudio
```

| task id | task | verified by |
|---|---|---|
| `five_towers` | build as many 5-block towers as possible | number of 5+ block columns at rest |
| `spire_landing` | a block on top of THE SPIRE without climbing | block at rest touching the spire top; body never on the spire |
| `generate_blocks` | make an additional block exist | an `object_appeared` physics event |
| `throw_height` | throw to 6.5 m (±0.25) | recorded flight peaks |
| `limited_structure` | tallest structure from 4 blocks | tallest resting multi-object structure |
| `complete_construction` | finish a 3-block column left on the island | that column extended to 5+ |

Each result records completion, score/accuracy, simulated time, actions, attempts, failed actions,
resources (objects handled, lost, generated), experiments, and memory use (relevant handbook entries at
the start, memory reads, programs from earlier sessions run). Benchmarks always run on **copies** of
memory (`experiments/benchmark_memory/run_NNNN/`), so an agent's real memory is never changed by a
benchmark (tested). The comparison reports differences in both directions and makes no significance claim.

**Task budgets are per-task, not from the CLI:** `benchmark` and `ablate` ignore `--max-steps` /
`--minutes`, because every task defines its own step limit and wall-clock ceiling (e.g. `five_towers`
140 actions, `spire_climbing` 80 actions + 30 min) so tasks are comparable across groups. The CLI
accepts the `--max-steps` flag on these commands but it does nothing there — it is only meaningful for
`explore` and `run-agents`.

---

## 7. Integrity: what each record means

| question | where |
|---|---|
| What the agent actually did | `actions` table: every call, arguments, full result, success/failure (append-only) |
| What the physics engine recorded | `events` table, `source='physics'`: flights, collisions, landings, appearances (append-only, SHA-256 hash chain) |
| What the agent claims | `records` of kind `claim`, `experiment`, `discovery`, `goal`, its messages |
| What its memory says | the files in its memory directory, and their audit trail |
| What was independently verified | claim verifiers, evidence verdicts, benchmark `task_result`, `integrity_flag` records |

```bash
$PY -m playground verify-log 12      # recompute session 12's hash chain
```
`GET /api/sessions/{id}/integrity` returns the full claimed-vs-verified report for a session.

---

## 8. Configuration reference (selected)

| key | default | meaning |
|---|---|---|
| `session.duration_minutes` | 60 | wall-clock length of a session |
| `session.max_steps` | null | optional cap on agent tool calls |
| `memory.confirm_min_experiments` | 2 | measured experiments needed for *confirmed* |
| `agents.count` | 13 | number of agents created by `init` / `run-agents` |
| `rules.ring_generation.enabled` | true | the hidden world rule (see RESEARCHER_NOTES.md) |
| `agent_body.*` | | mass, jump force, reach, carry and climb limits, maximum throw force |
| `world.*` | | island, platforms, spire, blocks, rock, ring |
| `database` | experiments/playground.db | SQLite file |

### Prompt size and per-call latency

A local model's per-call time grows with the request: by ~80 actions an auto-probed 262k
window can let history grow past 30k tokens (several minutes per call on XPU). Mitigations,
all recorded in the session's config hash:

* **`context_tokens: 32768`** (pinned on the local adapters) bounds the whole request:
  the history trim drops oldest exchanges first instead of growing into the window.
* Dropped history is not lost: each dropped message is compressed into a one-line digest
  carried inside the "earlier turns" marker (tool names, outcomes, agent's own words —
  deterministic, no model-generated summary). The full record stays in the database.
* **`playground messages <session_id>`** dumps any session's complete transcript
  (messages + every action with arguments and full results); `--out FILE` writes JSONL.

---

## 9. Tests

```bash
bash run_tests.sh           # ALL test files + health check
bash run_tests.sh --fast    # skip GPU/XPU tests and long XPU checks in health_check
```

The test suite runs:
- `tests/test_hve.py` - Happy/Validate/Error physics and interface tests
- `tests/test_smoke.py` - full system smoke tests (sessions, API, CLI)
- `tests/test_regression_extra.py` - prompt recording, determinism, collision debouncing
- `scripts/health_check.py` - environment and dependency verification

---

## 10. ML Lab

The ML lab provides infrastructure for training neural controllers from recorded sessions.
See [ML_LAB.md](ML_LAB.md) for the full researcher guide.

Key commands:
```bash
$PY -m playground ml devices              # show available compute devices
$PY -m playground ml generate --spec ...  # generate training data
$PY -m playground ml train --spec ...     # train a model
$PY -m playground ml list                 # list registered models
```

Intel XPU notes are in `reports/XPU_DIAGNOSIS.md`.

---

## 11. Known limitations

* **The scripted agent is not an AI.** It exists to exercise and validate the laboratory. Its later-session
  improvement is real in the narrow sense that it is caused by the contents of its memory files (removing
  one entry changes its behaviour, shown in `playground demo`), but its strategies were written by hand.
  Conclusions about AI behaviour require real-model runs.
* **No real-model study has been run yet.** Real-model connectivity was verified with LM Studio and
  Qwen3.5-9B driving an exploration session through the tool interface. The `lmstudio_unsloth4b` adapter
  (researcher's Unsloth Qwen 4B) was explored but recorded no experiments during evaluation. Ollama Cloud
  currently rejects requests with HTTP 402 (no credits), and no OpenAI or Anthropic key was exercised here,
  although both adapters are tested against protocol-faithful fake endpoints.
* **Learning-from-memory not yet demonstrated.** While the ML lab infrastructure supports agent-driven
  training, no LLM-controlled agent has successfully completed the train→deploy→verify loop.
* **Body model.** The body is a non-rotating box without limbs. "Climb" is a mantle onto ledges up to
  1.05 m above the feet. It cannot balance, fall over or be injured. The held object is attached rigidly.
* **Behaviour classification** is rule-based on recorded actions plus a few keyword cues. It measures
  structure (goals set, meanings assigned, varied repetition, activity outside any goal), not motives.
  "Activity for its own sake" means *no recorded purpose*, which is not the same as *no purpose*.
* **Time.** Sessions are bounded by wall-clock minutes (the brief's 60) and optionally by an action cap.
  Simulated time only advances during actions. A slow model therefore gets fewer actions in 60 minutes
  than a fast one, so use `--max-steps` for cross-model comparisons.
* **Evidence checks** match numbers in the agent's experiment record against the cited actions' results
  (2% / 2 cm tolerance). A derived quantity (e.g. a difference the agent calculated) is reported as
  "not found" rather than judged.
* **Statistics.** The benchmark summary reports means and completion rates per group. No significance tests
  are run, and with small groups none would be meaningful.
* The dashboard is a monitoring tool for one researcher on localhost. It has no authentication; do not
  expose port 8765 to a network.
