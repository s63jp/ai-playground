# PHYSICS EXTENSION — FROZEN MULTI-AGENT CONTRACT (Nov 2026)

This is the **binding contract** for the parallel AI Playground physics-extension build.
Read it fully before writing anything. Your package brief adds detail on top; this file
wins wherever the two disagree.

## 0. Ground rules (every package)

- Project root: this repository checkout. Python: `./.venv/Scripts/python.exe`.
- pytest 9 + `-q` prints **no final count line**. Pass/fail = exit code; quote the `[ 100% ]`
  progress line from your runs as evidence.
- **No git commands at all** (no commit/stash/checkout). The parent handles git.
- **Exclusive file ownership**: you may only create/edit files listed under your package.
  Need a change in someone else's file (config, schema, server, another package)? Put it in
  your report as an explicit **PATCH REQUEST** with exact old_text/new_string snippets.
- **Shared parent-owned files you must never edit**: `config/default.yaml`,
  `playground/config.py`, `playground/dashboard/server.py`, `playground/interface/schema.py`,
  `playground/interface/physical.py`, `requirements.txt`, anything under `playground/session/`,
  `playground/agents/`, `playground/memory/`, `playground/storage/`, `playground/gateway/`,
  `run_tests.sh`, `scripts/health_check.py` (owned by P10), all `tests/` not in your list.
- **New agent-facing tools**: do NOT register them yourself. Export from your module:
  - `TOOLS_EXT: list[dict]` — tool definitions in the exact shape produced by `_tool()` in
    `playground/interface/schema.py` (read it for the shape; import `_tool` if importable).
  - `TOOL_HANDLERS: dict[str, callable]` — name → `def handler(tools, **kwargs) -> dict`
    where `tools` is the `PhysicalTools` instance (has `.w` = world and all existing wrappers).
  The parent wires both into schema.py/physical.py. Include the exact dicts in your report.
- **Config**: never edit `config/default.yaml`. Put defaults as module constants and request a
  parent patch. If you need a section in default.yaml, provide the exact YAML snippet.
- **No new hard dependencies** without a patch request. Torch is already in the venv (use CPU in
  tests: `torch.set_num_threads(1)` friendly; do not use GPU in tests).
- **No ports/servers**: never bind ports or start the dashboard from tests. Test servers
  through `TestClient` / in-process fixtures like existing tests do.
- **Machine-independent tests**: derive all repo paths from `__file__` (hardcoded
  a hardcoded absolute machine path breaks CI and is a regression test target).
- **Free-will contract for agent-facing text** (user mandate): no tool description, notice or
  result string may steer agents toward actions or objectives. Facts and "your choice" framings
  only. If you add a tool description, follow existing schema.py wording style.
- **Determinism in physics asserts**: drive each hinge with a **fresh MjData** per hinge; use
  generous margins (6–40x) so results hold on CI's newer MuJoCo.
- **Do not weaken or skip existing tests.** If an existing test conflicts with contract-required
  behaviour, report it as a CONTRACT ISSUE instead of editing it — unless the test file is in
  your ownership (then update it and say so).
- Never fake results. If blocked, say so in the report with the real error output.

## 1. Exclusive ownership table

| # | Package | Owned files (exclusive) |
|---|---------|------------------------|
| P1 | locomotion-core | `playground/world/physics.py`, `playground/world/spec.py`, `assets/humanoid_mujoco.xml`, `assets/HUMANOID_ASSET_README.md`, `playground/locomotion.py` (new), `scripts/demo_humanoid_walk.py`, `tests/test_humanoid_world.py`, `tests/test_humanoid_pd_walk.py`, `tests/test_locomotion.py` (new) |
| P2 | rl-core | `playground/rl/**` (new package), `tests/test_rl_core.py` (new) |
| P3 | skills-library | `playground/skills/**` (new package), `tests/test_skills.py` (new) |
| P4 | manipulation | `playground/manipulation.py` (new), `tests/test_manipulation.py` (new) |
| P5 | projectiles-impacts | `playground/combat/projectiles.py`, `playground/combat/impacts.py`, `playground/combat/__init__.py` (new), `tests/test_projectiles.py` (new), `playground/dashboard/static/js/splat.js` (new), `playground/dashboard/static/js/impact_fx.js` (new) |
| P6 | equipment | `playground/combat/equipment.py` (new), `tests/test_equipment.py` (new) |
| P7 | arena | `playground/arena/**` (new package), `tests/test_arena.py` (new) |
| P8 | human-player | `playground/player.py` (new), `playground/dashboard/player_api.py` (new), `playground/dashboard/static/js/player.js` (new), `tests/test_player.py` (new) |
| P9 | visualisation | `playground/dashboard/static/app.js`, `playground/dashboard/static/index.html`, `playground/dashboard/static/style.css`, `playground/dashboard/static/js/humanoid_view.js` (new), `tests/test_ui_humanoid.py` (new) |
| P10 | qa/perf/docs | `scripts/profile_step.py`, `tests/test_performance.py` (new), `tests/test_mode_integration.py` (new), `docs/PHYSICS_EXTENSION.md` (new), `scripts/health_check.py` |
| Parent | everything else | shared wiring: schema.py/physical.py registration, server.py routers, config/default.yaml, requirements, final commit, cross-package integration |

`static/js/splat.js`, `static/js/impact_fx.js` (P5) and `static/js/player.js` (P8) are new
files; P9 owns all **existing** static assets and `index.html` script tags (P8 sends its
`<script>` requirement as a patch request, parent applies).

## 2. Frozen world API (P1 must implement; every package consumes)

P1 removes the `if "hip" in joint: self.d.qpos[self.aq] += 0.4 * self.dt` teleport hack and the
duplicate second PD loop in `_step` (torque is currently applied twice). All locomotion runs
through `qfrc_applied`/`ctrl` only. Forward displacement must come from foot–ground contact
physics. The box-agent path must keep working exactly as today (all existing tests pass).

New/changed world methods (exact signatures):

```python
def register_step_callback(self, fn: Callable[["PhysicsWorld"], None]) -> None
    # fn(world) runs once per physics step AFTER d.qfrc_applied[:] = 0.0 and after the
    # internal PD/gait application, BEFORE mj_step. Callbacks run in registration order.
    # (Existing _pd bookkeeping stays; callback list survives get_state/set_state? — no:
    # callbacks are Python-level, do not serialize.)

def set_gait(self, mode: str, speed: float = 1.0, heading: float | None = None) -> dict
    # mode in {"stand","walk","run","stop","crouch"}; "stop" halts and keeps balance.
    # Internal GaitController (playground/locomotion.py) drives hip/knee/ankle/spine PD
    # targets each step. Returns {"mode":..., "speed":..., "heading":...}. Emits "gait_change".
    # heading: radians, world-frame yaw target; None = keep current.

def stand_up(self) -> dict
    # Recovery from prone/supine/ fallen poses using joint targets only.
    # Emits "recovery_started" then "recovery_complete" (or "recovery_failed" after 6 s sim time).

def agent_fallen(self) -> bool
    # Deterministic: torso height below threshold OR torso up-vector tilt beyond threshold.
    # Must be False for a healthy standing humanoid, True after the body settles on the ground.

def agent_skeleton(self) -> list[dict]
    # [{"name": body_name, "pos": [x,y,z], "quat": [w,x,y,z]}, ...] for every humanoid body
    # in kinematic tree order (torso first). Box agent returns the single agent body entry.
    # Consumed by P9 rendering and by the dashboard /api/state (parent patches server).

def spawn_object(self, oid: str, pos, quat=None) -> dict
    # Move an EXISTING movable object (must be in self.specs with matching qadr) to pos,
    # zero velocity, optional quat. Emits "object_moved". Raises KeyError for unknown oid.
    # (No new bodies post-build — MuJoCo limitation. P5 projectile pool = pre-declared pooled
    # objects in cfg["objects"], see P5 brief.)

def set_object_velocity(self, oid: str, velocity, angular=None) -> dict
    # qvel write for a movable object's slide/free dofs. Emits "object_launched".
```

Also in P1 scope: `tests/test_humanoid_pd_walk.py` currently depends on the teleport hack for
its displacement assertion — rewrite it to assert real contact-driven displacement from
`set_gait("walk")` (generous margin, e.g. ≥ 0.15 m over ≥ 8 s sim time, and upright at end).

## 3. Frozen cross-package interfaces

### P2 RL consumes world
> **M6-B B2 supersession note (2026-11-17; contract extended at B3).** The
> authoritative RL environment contract is now `docs/RL_ENVIRONMENT.md`
> (contract version `rl_env/3` — the v2 40-dim observation plus the appended
> commanded-velocity slice, one measured reward pipeline in
> `playground/rl/rewards.py`, one curriculum in
> `playground/rl/curriculum.py`). The pre-M6 text below is kept for
> history; its reward-component list (`energy_penalty`, `joint_limit`) and its
> `train(cfg)`/checkpoint artifact requirements describe the REMOVED broken pre-M6
> trainer, not the current system.
> **M7-0a supersession note (2026-11-18).** The authoritative contract is
> now `rl_env` v4: 68-dim observation / 15-action (the fixed 43-dim v3
> prefix preserved forever, the v4 command block riding after it), with the
> read-only legacy-v3 mode for historical lineages; the M7-0b loading law
> adds the op-tag checkpoint loading (fresh/resume/continue_fork/
> transfer_fork) and the declared-prefix surgery loader. Same pointer:
> `docs/RL_ENVIRONMENT.md`.
`LocoEnv(cfg)` wraps a `PhysicsWorld` directly (build via `PhysicsWorld(load_config())` with
`agent_body.type: "humanoid"`). OBS: fixed-size float32 vector (humanoid: 14 joint positions +
14 joint velocities + torso up-vector(3) + torso linear vel(3) + upright margin(1) — document
the layout in a module docstring; expose `obs_dim`). ACTION: 14 normalized joint targets
[-1,1] → radians via joint range scaling; applied via `set_joint_target` (PD gains from cfg).
Termination: `world.agent_fallen()` or step limit. Reward: config-driven components
(`velocity_tracking`, `upright`, `alive`, `energy_penalty`, `smoothness`, `joint_limit`) with
coefficients from a module DEFAULTS dict; anti-exploit guards **required**: fall termination,
no reward for oscillating in place when velocity tracking is on (reward is tracking error-based
not distance-only), energy penalty, and a check in tests that a policy doing nothing
(alive-only standing) scores strictly less on the walking curriculum reward than a moving one.

Artifacts: `experiments/rl/<run_id>/{config.json, metrics.jsonl, checkpoints/step_NNNNNN.pt}`.
`train(cfg)` must accept `{max_steps, max_minutes, checkpoint_every, curriculum_phase, seed}`
and terminate on budget. Checkpoint API must roundtrip reproducibly (same policy → same obs →
same action; deterministic test).

Curriculum (curriculum.py): configurable phase list; phase fields
`{name, gravity_scale, upright_assist, max_steps, reward_overrides}` — assistance implemented
**in the env wrapper** (per-step corrective force on torso capped by config, gravity via
MuJoCo model opt override before stepping), NOT by moving qpos. Tests for phase progression +
reproducibility. Training smoke test must finish < 60 s CPU.

### P3 Skills consumes everything via one adapter
```python
# playground/skills/adapter.py
class CapabilityAdapter(Protocol):
    def set_gait(mode, speed=1.0, heading=None) -> dict
    def stand_up() -> dict
    def agent_fallen() -> bool
    def reach(object_id) -> dict          # P4
    def grasp(object_id) -> dict          # P4
    def release() -> dict                 # P4
    def pick_up(object_id) -> dict        # existing world
    def throw(object_id, direction, force) -> dict   # existing world
```
`WorldCapabilities(world, manipulation=None)` builds itself from a world + optional P4
controller. Missing capability → skill marked `unavailable` in registry, never crashes.
`SkillSpec` frozen fields: `id, name, category, inputs: dict, outputs: dict, preconditions:
dict, success_criteria: dict, failure_conditions: dict, controller: dict, status, version:
str, metrics: dict`. `SkillsLibrary(root: Path)` — `register(spec)`, `get(id)`,
`list(), save()/load()` (JSON per skill under root), `invoke(skill_id, adapter, params) ->
{"skill": id, "status": "succeeded|failed|unavailable", "detail": ...}`. Catalog of the
spec's 16 initial skills ships in `catalog.py`; each maps to real adapter calls (gait/reach/
grasp/throw) where the capability exists; no fake success. Export `TOOLS_EXT` +
`TOOL_HANDLERS` with one `invoke_skill` tool (`skill: str, params: object`).

### P4 Manipulation consumes P1 world only
`ManipulationController(world)`: `reach(object_id)` — arm-joint PD targets toward the object
via simple Jacobian-free IK heuristic on shoulder/elbow; `grasp(object_id)` — only succeeds
if a contact between an agent limb geometry and the object exists; grip is maintained by a
per-step corrective wrench on the object toward the palm site (registered via
`world.register_step_callback`), **never by teleporting**: the object must keep physical
displacement continuity (test: object travels hand-to-target continuously, its max per-step
displacement bounded). `release()`, `push/pull` (forces via `xfrc_applied` on the object,
bounded by cfg). If the humanoid XML lacks hand geom/site entities, DO NOT edit P1's assets —
report a PATCH REQUEST for P1 with exact XML. `TOOLS_EXT`: `reach/ grasp/ release`.

### P5 Projectiles consumes P1 world
`playground/combat/projectiles.py`:
```python
class ProjectileSystem:
    def __init__(self, world, pool_ids: list[str])   # pool_ids: pooled movable objects in cfg
    def launch(self, origin, direction, speed: float, cfg: dict | None = None) -> str  # proj_id
    def update(self) -> list[dict]   # call once per step; returns NEW impact dicts:
    # {"proj_id", "pos":[x,y,z], "normal":[x,y,z], "angle_deg", "surface": str, "speed", "t"}
    def explosion(self, pos, radius: float, force: float, particle_count: int = 64) -> dict
```
Launch = `world.spawn_object(pool_id, origin)` + `world.set_object_velocity(...)`; impact =
contact detection with relative speed threshold; explosion = radial impulse on bodies within
radius (`d.xfrc_applied`, magnitude scaled by distance, Fibonacci-sphere particle directions
in the emitted event — use `math` golden-ratio sampling, seeded). Emits events
`"projectile_impact"` and `"explosion"` (payload = the dicts above + `particle_dirs`).
Client contract: `static/js/splat.js` exports `SplatRenderer` (init(canvas-ish scene hooks) →
`addSplatter(pos, normal, angleDeg, colour)` → geometry built by raycast-conforming decal
meshes merged into one BufferGeometry batch per colour, no flat floating textures, budget ≤
a few draw calls for thousands of decals) and `static/js/impact_fx.js` exports
`ImpactFX(explosionEvent)` particle burst using an InstancedMesh with configurable counts. They
integrate via events consumed in P9's `app.js` (P9 adds small event-dispatch glue; both
packages state the exact event payload they expect — it is this contract's dicts).
`TOOLS_EXT`: `throw_projectile (direction, speed)`, `explode?` — only if physically triggered
by agent actions; no agent tool for `explosion` unless the spec requires it (it doesn't; leave
explosion to arena/equipment).

### P6 Equipment consumes P5
`playground/combat/equipment.py`: `EQUIPMENT: dict[str, dict]` data-driven specs
(mass, grip: `one_hand|two_hand|dual`, hands_required, projectile cfg, accuracy_spread, rate,
ammo_capacity, reload_steps). `EquipmentItem(world, equip_id)` with `can_grip(left,right)`,
`grip()/ungrip()`, `fire(direction) -> {"fired": bool, "ammo": n, "proj_ids": [...]}` calling
**only** the frozen launcher callable `(origin, direction, spread_deg) -> list[str]` injected
at construction (P5's ProjectileSystem adapted; P6 defines the adapter glue file-local).
`TOOLS_EXT`: `equip (equipment_id)`, `fire (direction, strength?)`, `reload`.
No equipment-specific logic in the humanoid controller (contract §18.9).

### P7 Arena consumes existing session system (read-only)
`playground/arena/match.py`: `ArenaMatch(cfg)` — cfg:
`{teams: [{name, agents: [agent_cfgs], equipment, size}], rules: {score_target, duration_s,
friendly_fire}, arena: {obstacles, cover}}`. `ArenaMatch.plan()` → validated world/agent plan;
`play()` executes against the existing session orchestration (READ `playground/session/` and
`playground/agents/` first — they are parent-owned; integrate with them via their public APIs
only, e.g. launching sessions, reading events; if the required public API is missing, report
a PATCH REQUEST instead of editing). `MatchResult`: per-team/agent stats (hits, accuracy,
distance, falls), winner, duration. Victory conditions configurable; hit detection via P5
impact events when equipment is in play. Post-match analysis written to SQLite through the
existing storage module's public functions only.
`playground/arena/modes.py`: `build_world_cfg(mode)` with modes `duel`, `team`, `unequal`
(different team sizes, capability/equipment variance). Tests with fake/stub agents (no LLM).
`TOOLS_EXT`: none (arena is dashboard/config driven).

### P8 Human player consumes P1 hook
`playground/player.py`:
```python
class PlayerController:
    def __init__(self, world, cfg: dict | None = None)
    def apply_input(self, state: dict) -> dict   # {move:[x,y], yaw, pitch, jump: bool, run: bool}
    # registers via world.register_step_callback; converts input into slide-joint qfrc /
    # velocity targets each step (same physics as agents: no teleporting; jump = vertical
    # impulse, capped); player body = the agent body itself.
    def state(self) -> dict   # {pos, vel, yaw, pitch, grounded}
```
`playground/dashboard/player_api.py` — `APIRouter` with `POST /api/player/input`,
`GET /api/player/state`, `POST /api/player/join` (attach to the live world without resetting
agents — patch request to parent for the world-registry accessor if none is exported).
`static/js/player.js` — pointer-lock mouse look, WASD, shift-run, space-jump, camera
first/third person toggle `V`, POSTs input at ≥ 20 Hz; all UI text factual (free-will rules do
not apply to the human player UI, but no forced objectives there either). Patch requests to
parent: include router in server.py; `<script>` include in index.html goes to P9.

### P9 Visualisation consumes P1 world state
`app.js` must keep `TestUiContract` green (`tests/test_ml_engine_generators.py`) — run
`node --check app.js` after edits AND that test. Render the humanoid from skeleton data in
`/api/state` (parent patches server.py to add `"skeleton": world.agent_skeleton()` — send the
patch request; P1 guarantees the API). Limb meshes: capsule proxies from skeleton pos/quat;
fallback to existing box when skeleton absent. New module `static/js/humanoid_view.js` holds
the view logic (export `HumanoidView`); app.js wires it. Debug overlays (toggle, default off):
joint axes, collision proxies, ground contacts, gait target markers, RL reward HUD (reads
optional `/api/state["rl_debug"]` — optional field, absent = hidden). Team colours + status
indicators: extend existing agent rendering data only, no invented endpoints. Wire P5's
`SplatRenderer`/`ImpactFX` event consumption (payloads = contract dicts). `player.js`
integration: script tag in index.html (P8 requested) + camera-mode plumbing.
DO NOT break existing tabs/loaders (`showTab()` mapping is load-bearing).

### P10 QA/perf/docs
- `scripts/profile_step.py`: extend to benchmark: box vs humanoid step time (median/p95 over
  ≥ 200 steps), scaling with object count, memory RSS. Print JSON to stdout; also write
  `experiments/rl/../per reports` → no: write to `experiments/perf/<timestamp>_step.json`
  (gitignored dir is fine).
- `tests/test_performance.py`: humanoid step budget (generous CI margin: median < 25 ms),
  4x speed = 4x simulated seconds for fixed wall step count (sim time scaling contract),
  projectile pool launch throughput smoke.
- `tests/test_mode_integration.py`: cfg `simulation.mode` in {playground, training, arena,
  interactive} — assert default.yaml documents the key (parent adds it; test reads the
  parent-applied YAML via `load_config()` and validates dispatch table wiring points:
  importable entry symbols `playground.arena.ArenaMatch`, `playground.rl.curriculum`,
  `playground.player.PlayerController`; if a symbol is missing at your runtime, xfail-skip
  with reason, do NOT fail the suite for other packages' latency).
- `scripts/health_check.py`: extend: import-check new subsystems, report module status + skip
  cleanly when experiments/ absent.
- `docs/PHYSICS_EXTENSION.md`: phase-status tracker skeleton (Phase 0-11 from the user spec)
  with a "Status evidence" table the parent fills at integration; write your own package's
  section (deliverables + how to verify).

## 4. Integration gate (parent, not children)

1. `git diff --stat` reviewed against the ownership table (violations rejected).
2. Children's patch requests and config snippets applied.
3. TOOLS_EXT/TOOL_HANDLERS registered in schema.py/physical.py; server routers included.
4. Full suite `./.venv/Scripts/python.exe -m pytest tests/ -q` green + health check run.
5. Dashboard smoke: server starts, `/api/health` ok, humanoid skeleton in `/api/state`.
6. ONE commit by parent.

## 5. Report format (every child, final message)

```
PACKAGE: <P# name>
CHANGED FILES: <list>
ROOT CAUSES/DECISIONS: <bullets w/ evidence>
TESTS ADDED/UPDATED: <files + what they cover>
REAL TEST OUTPUT: <pasted [ 100% ] lines + exit codes>
PATCH REQUESTS: <exact old_string/new_string snippets, target file>
CONFIG SNIPPETS: <exact YAML for default.yaml>
CONTRACT ISSUES: <anything ambiguous/broken in this contract>
LIMITATIONS: <honest list>
```
