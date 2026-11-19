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

## 0. One server, one engine

**Everything runs on the main dashboard server.** `playground serve` builds one
FastAPI app and mounts both neural surfaces on it (the gateway/player-router
mounting convention):

- `register_rl_live_routes(app, cfg)` (`playground/rl_live/rest.py`) → the
  **"Live neural training"** tab (`static/js/rl_live_tab.js`), under
  `/api/rl_live/*`.
- `register_rl_playback_routes(app, cfg)` (`playground/rl_live/rest_playback.py`)
  → the **"Checkpoint playback"** tab (`static/playback.html`), under
  `/api/rl_playback/*` with its own Bus/SSE so training and playback never
  collide.

Both drive the SAME session/bus/telemetry/playback modules. There is **no
separate sidecar process and no second server** — the earlier standalone
`:8766`/`:8770` surfaces have been consolidated into these dashboard mounts.
The viewer reuses the same three.js vendor set, `OrbitControls`, and both the
primitive `HumanoidView` (V1 fallback) and the Blender-FBX `createHumanoidV2`
(V2, default) rig renderers.

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
┌─────────────────────── dashboard server (:8765, `playground serve`) ────────┐
│ session thread:  LocoEnv → FrameTapEnv (read-only) → PPOTrainer (the B3     │
│                  algorithm, unmodified; pause/stop via should_stop;         │
│                  telemetry via on_iteration)                                │
│ tap → frames.py  → bus (30 Hz latest-wins fanout) → SSE → browser           │
│ on_iteration     → run dir progress.jsonl + SSE iteration events            │
│ B3 refresher     → read-only tail of experiments/rl/b3_run1/progress.jsonl  │
│ HTTP: /static/* (dashboard shell, vendor three.js, OrbitControls,          │
│       HumanoidView, humanoid_v2 + FBX assets) + /api/rl_live/* and          │
│       /api/rl_playback/* JSON control/status/SSE endpoints                  │
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
| `rest.py` | the FastAPI mount for live training on the dashboard server (`register_rl_live_routes(app, cfg)`), exposed under `/api/rl_live/*` on :8765. |
| `rest_playback.py` | the FastAPI mount for checkpoint playback on the same server (`register_rl_playback_routes(app, cfg)`), under `/api/rl_playback/*` with its own Bus/SSE. |
| `playback.py` | `PlaybackSession`: a saved, verified checkpoint drives the real MuJoCo sim through its own inference — no PPO updates, no scripted motion (see §10). |

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

One server, both surfaces:
```bash
cd <repo>
.venv/Scripts/python.exe -m playground serve
# → http://127.0.0.1:8765/#live       (the "Live neural training" tab)
# → http://127.0.0.1:8765/#playback   (the "Checkpoint playback" tab)
```

Knobs come from the `rl_live:` config block (`out_root`, `b3_progress`,
`no_b3`, `frame_hz`) with the same defaults the mounts were built with
(`out_root` → `experiments/rl_live`, frame_hz 30). Checkpoint playback is
started per-checkpoint from its tab (or `POST /api/rl_playback/start`); its
run artifacts land under `experiments/rl_playback/<name>/`.

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
- The live mount never opens B3 paths for writing. The only B3 access is
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

- The humanoid rig renders as the retargeted Blender FBX mesh (V2, the
  default; `?rig=v1` or any V2 error falls back to the primitive rig).
  Island-level (non-rig) MuJoCo meshes are still counted, not rendered.
- Pause granularity is the rollout, not the decision (algorithm-respecting).
- One live session per server; the objective registry offers walking plus the
  M7-1a S1 individual-limb ladder.
- `AgentHandle.agent_skeleton()` is not used for rig rendering (it is the
  box-agent/limb layout); the geom-accurate spec replaces it, with
  `HumanoidView` retained as the no-spec fallback.

## 10. Checkpoint playback (read-only demo surface)

`playground.rl_live.playback.PlaybackSession` replays a saved, verified
checkpoint through the real MuJoCo simulation and streams it through the same
SSE pipeline. Nothing here trains. It is served by the **same dashboard
server** via `register_rl_playback_routes` (`rest_playback.py`) on the
`/api/rl_playback/*` family with its own Bus — the **"Checkpoint playback"**
dashboard tab — so no separate process or port is involved.

    # start the dashboard, open the "Checkpoint playback" tab, pick a
    # checkpoint + stage + inference mode and press start, or equivalently:
    POST /api/rl_playback/start
      {"experiment": "<run>", "checkpoint": "ckpt_final.json",
       "stage": "sustained", "inference": "deterministic", "speed": 1.0}
    # (a checkpoint outside experiments/rl_live may be given as {"path": ...}
    #  under an allowed training-output root)

- **The rig is the checkpoint's own era.** Before any world is built, the
  sidecar extracts the historical asset from git (`e7e81ac:assets/
  humanoid_mujoco.xml` — the B3 era), refuses the bytes unless SHA-256
  matches the recorded value, copies it into the run dir and points
  `playground.world.spec.HUMANOID_ASSET_PATH` at the copy (a process-local
  seam; the default path and every other process keep the current asset).
  Every episode's compiled model is re-checked against the era's head
  inertial before a decision runs.
- **The checkpoint is loaded through THE `rl/checkpoint.py` loader** (tamper
  hash + obs/action contract fingerprint); the optimiser state is discarded
  on load: there is no import of `PPOTrainer` and no update path of any
  kind. The actor's parameter digest (every float32, log_std included) is
  recomputed after every episode and shown in the viewer; any change would
  abort playback.
- **Inference modes are explicit and labelled**: `deterministic`
  (a = tanh(mu(s)) — the recorded evaluation convention, the primary demo)
  and `stochastic` (z = mu + sigma*eps with sigma exactly as trained). A
  switch applies at the next episode start (`mode_next` until then); the
  sampling RNG is re-seeded per episode so stochastic replay is reproducible
  per seed.
- **Controls**: pause / resume / reset-episode (replays the current seed
  from its start; also works while paused, applied on resume) / speed
  0.25-4.0 x (wall-clock pacing of the real 20 ms decisions) / mode switch.
- **Records**: `experiments/rl_playback/<name>/episodes.jsonl` per episode
  (duration, net + axis displacement, falls, MuJoCo warning names,
  action-hash) and `playback_info.json` (identity: checkpoint sha, seed,
  dims, rig fingerprint, actor digests). The checkpoint's own run dir is
  only ever READ (writes verified absent by test).
- **Proof hooks**: the deterministic band against the recorded
  10-episode held-out evaluation is asserted bit-exact by
  `tests/test_rl_playback.py::test_playback_rig_reproduces_the_recorded_heldout_eval`,
  and the playback HTTP mount is covered by `tests/test_rl_playback_rest.py`.
  Playback shares the dashboard server with live training but keeps its own
  Bus/SSE, `/api/rl_playback/*` routes and `experiments/rl_playback/` run dir,
  so the two never collide.
