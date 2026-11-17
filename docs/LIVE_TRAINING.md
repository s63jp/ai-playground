# Live Neural Training mode (M6-C foundations)

An interactive experience in which a genuinely untrained neural humanoid
learns to walk in front of the user: the user creates a fresh network, picks
the walking objective, starts training, watches the actual simulation attempt
the task, sees performance change as weights update, and can pause, resume,
stop and save the trained brain.

Status: **foundations complete on `feature/live-neural-training`** (this
branch, based on the B2 commit `cbac1cd`). The real-trainer wiring becomes
active the moment the B3 commit (ppo.py / checkpoint.py / eval.py) lands on
main — nothing else in this layer needs to change (see §7).

## 0. Two surfaces, one engine

**Primary: the main dashboard.** `register_rl_live_routes(app, cfg)`
(`playground/rl_live/rest.py`) mounts everything on the existing
`playground serve` FastAPI app (the gateway/player-router mounting
convention), and the dashboard gains a **"Live neural training"** tab
(`static/js/rl_live_tab.js`) beside "RL environment" — same three.js vendor
set, `OrbitControls` and `HumanoidView` the other agent-run viewers use.

**Optional: a standalone sidecar** (`python -m playground.rl_live.server`,
port 8766) for when you want an isolated window without the whole dashboard.
Both surfaces drive the SAME session/bus/telemetry modules; the sidecar is a
convenience, not a second engine.

## 1. Honest-display contract

What the viewer shows is the actual training simulation, streamed from the
training process itself:

- **The rig on screen is the one being trained.** Frames are read straight
  from the LocoEnv's MuJoCo data (`d.xpos` / `d.xquat` — the solver's own
  kinematics) at every 20 ms decision boundary, decimated to ~30 Hz.
- **No scripted gait, no demonstration puppet, no preview simulation.** The
  tap never simulates anything; it only reads.
- **Actions shown are the policy's own actions**; rewards/contacts/fall flags
  come from the env's own `step` return and `info`.
- Telemetry records are the trainer's own per-iteration records — the exact
  same schema the B3 run writes to `progress.jsonl`.

## 2. Architecture (one process, one thread per concern)

```
┌───────────────────────────── rl_live sidecar (:8766, 127.0.0.1 only) ───────┐
│ session thread:  LocoEnv → FrameTapEnv (read-only) → PPOTrainer (the B3     │
│                  algorithm, unmodified; pause/stop via should_stop;         │
│                  telemetry via on_iteration)                                │
│ tap → frames.py  → bus (30 Hz latest-wins fanout) → SSE → browser           │
│ on_iteration     → run dir progress.jsonl + SSE iteration events            │
│ B3 refresher     → read-only tail of experiments/rl/b3_run1/progress.jsonl  │
│ HTTP: /rl_live.html + /static/* (reuses dashboard vendor three.js,          │
│       OrbitControls, HumanoidView) + JSON control/status endpoints          │
└──────────────────────────────────────────────────────────────────────────────┘
```

Key pieces (`playground/rl_live/`):

| module | role |
|---|---|
| `frames.py` | read-only extraction: scene spec (bodies/geoms/joints) + per-decision frames (body poses, joints, root, action, reward components, contacts, fall/fault). Names only — ids are re-derived whenever the world instance changes (LocoEnv rebuilds it every reset). |
| `tap.py` | `FrameTapEnv` proxies `reset`/`step`/`episode_summary` to the real env and packs frames. A failing viewer can never break training (sink errors are isolated and counted). |
| `session.py` | state machine `idle → starting → running ⇄ paused → stopped \| error`; fresh Actor/Critic per start; own `progress.jsonl` + `run_config.json` + checkpoints under `experiments/rl_live/<name>/`. |
| `interop.py` | the single seam to the B3 trainer modules (lazy imports; clear `InteropUnavailable` behaviour while B3 is not on the tree). |
| `bus.py` | SSE fanout; frames latest-wins (a slow viewer shows the freshest pose, never a stale backlog). |
| `rest.py` | the FastAPI mount for the main dashboard server (`register_rl_live_routes(app, cfg)`) — same session/bus/B3 objects as the sidecar, exposed under `/api/rl_live/*` on :8765. |
| `server.py` | stdlib HTTP+SSE sidecar (optional isolated surface); localhost-only by policy; traversal-guarded statics; the only writes go to the session's own run dir. |

Dashboard UI files (small, additive diffs in shared files so the tab fits
the existing shell):

- `dashboard/server.py`: +2 lines after the player-router mount (import +
  `register_rl_live_routes(app, cfg)`).
- `dashboard/static/index.html`: the `live` nav button, the `#tab-live`
  section, one `<script type="module">` tag.
- `dashboard/static/app.js`: one `PAGE_META` entry, one loader-map entry.
- `dashboard/static/js/rl_live_tab.js`: the tab module (own three.js scene
  instance, no changes to the island viewer code paths).

## 3. Running it

Dashboard (primary):
```bash
cd <repo>
.venv/Scripts/python.exe -m playground serve
# → http://127.0.0.1:8765/#live   (the "Live neural training" tab)
```

Optional standalone sidecar (isolated window on its own port):
```bash
.venv/Scripts/python.exe -m playground.rl_live.server
# → http://127.0.0.1:8766/rl_live.html
```

Sidecar flags: `--port` (8766), `--frame-hz` (30; MuJoCo gives 50),
`--out-root` (default `experiments/rl_live`), `--b3-progress` / `--no-b3`.
On the dashboard mount the same knobs come from the `rl_live:` config block
(`out_root`, `b3_progress`, `no_b3`) with the same defaults.

A live training session uses the same one-thread CPU budget conventions as
`scripts/rl_ppo.py train` (`torch.set_num_threads(1)`); do not run long live
sessions and the B3 run at full tilt at the same time on a shared machine if
you need the last few percent of throughput.

## 4. API

| endpoint | method | meaning |
|---|---|---|
| `/api/rl_live/status` | GET | session state, counters, last iteration record, checkpoints |
| `/api/rl_live/objectives` | GET | objective registry (walking) + stage ladder + trainer availability |
| `/api/rl_live/spec` | GET | cached scene spec (after a start), else `{"available": false}` |
| `/api/rl_live/stream` | GET (SSE) | `frame`, `reset`, `episode_end`, `iteration`, `stage_transition`, `status`, `saved`, `spec`, `b3_iteration` |
| `/api/rl_live/telemetry.json?limit=N` | GET | recent iteration records (own run) |
| `/api/rl_b3/telemetry?limit=N` | GET | **read-only** tail of the running B3 experiment |
| `/api/rl_live/start` | POST | `{objective, name?, seed?, total_steps?, ppo_overrides?}` |
| `/api/rl_live/pause` / `resume` / `stop` | POST | control the session |
| `/api/rl_live/save` | POST | `{tag}` — checkpoint; allowed only while paused/stopped |

All errors are `{"ok": false, "error": {"code", "message"}}` (HTTP 4xx),
matching the project's error convention.

## 5. Session semantics

- **Fresh brain per start**: a new `TanhGaussianActor`/`Critic` is built
  inside `interop.build_session_trainer` exactly like `scripts/rl_ppo.py train`
  does (same seeding discipline).
- **Pause** rides `PPOTrainer.should_stop`; it applies at the current rollout
  boundary — at most one 2048-decision rollout (~8 s) later. The UI says so.
  The trainer object (Adam state, RNG, counters, curriculum) is preserved, so
  resume continues exactly where training stopped.
- **Stop** also waits for the boundary, then writes `ckpt_final.json`.
- **Save** writes through `playground/rl/checkpoint.py` (the B3 format with
  contract fingerprints) — only while paused/stopped, so torch state is never
  read mid-update.
- Progress records go to `experiments/rl_live/<name>/progress.jsonl`, the
  same schema as the B3 file, so any tooling written for B3 telemetry reads
  this file unchanged.

## 6. Why the B3 run is safe while this exists

- The branch is a git worktree from `cbac1cd`; the B3 session's working tree
  is physically unreachable from this branch's file operations.
- The sidecar never opens B3 paths for writing. The only B3 access is
  `read_records()`/`ProgressTail.read_new()` on its `progress.jsonl`.
- Live training only starts when a user asks for it via API/CLI; nothing in
  this module launches training on import, and no second trainer exists here
  — `PPOTrainer` is the trainer.
- The default `--out-root` is a new directory (`experiments/rl_live/`); run
  names are single path segments (validated).

## 7. What lands with the B3 commit

`interop.py` imports `playground/rl/ppo.py` and `playground/rl/checkpoint.py`
lazily. While they are absent (this branch as saved), the session reports
`trainer_available: false`, refuses to start, and the viewer still works as
telemetry + live-frame plumbing. After B3 merges to main:

1. `git rebase main` on this branch (no conflicts expected: this branch only
   adds `playground/rl_live/` and its tests/docs).
2. `tests/test_rl_live_integration_b3.py` (skipped while the modules are
   absent) becomes runnable automatically and exercises a real short session.
3. Optionally port `objectives` beyond walking only after measured need;
   stages 4–6 of the ladder stay reserved (`env_ready=False`) exactly as the
   B2/B3 contract declares.

## 8. Protocol payloads (stability surface)

- **frame**: `{t:"frame", episode, decision, sim_t, bodies:{name:{p:[x,y,z],
  q:[w,x,y,z]}}, joints:{name:{q,v}}, root:{pos,vel}, up_z, action[14],
  reward, components{...}, effort_j, contacts{left,right}, fallen, faulted,
  terminated, truncated, target_velocity[3], dt_window}` — quaternion order
  is MuJoCo's `(w, x, y, z)` everywhere.
- **iteration**: the trainer's own record (iteration, total_valid_steps,
  stage, sigma, losses/KL/clip, roll_* aggregates) — byte-compatible with the
  B3 `progress.jsonl` lines.
- **scene spec**: `{spec_version, bodies:[{name, rig, parent, geoms:[{type in
  (plane|sphere|capsule|cylinder|box|ellipsoid), size, pos, quat, rgba}],
  ...}], joints, policy_joints, meshes_skipped, ...}` — mesh geoms are
  counted, not rendered (documented limitation).

## 9. Known limitations (honest list)

- Mesh geoms do not render (the humanoid rig is primitives, so the rig is
  accurate; island-level meshes would appear as omitted).
- Pause granularity is the rollout, not the decision (algorithm-respecting).
- One live session per sidecar process; objective registry offers walking
  only, per the mission scoping.
- `AgentHandle.agent_skeleton()` is not used for rig rendering (it is the
  box-agent/limb layout); the geom-accurate spec replaces it, with
  `HumanoidView` retained as the no-spec fallback.
