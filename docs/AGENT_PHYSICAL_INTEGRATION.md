# AGENT PHYSICAL INTEGRATION — Contract (Phase 12)

Integrates the completed physical-intelligence extension (commit `cd68da2`) into the
existing autonomous agents of AI Playground. **Extension, not replacement**: every
physical behaviour must ride the existing implementation. Nothing here introduces a
second locomotion system, teleports, qpos/qvel writes, or scripted objectives.

Status values: `landed` / `partial` / `pending`, filled against test evidence only.

## Audit findings (verified 2026-11-17, live probe `probe_agent_integration.py`)

1. **Tool surface vs capability gap.** `PhysicalTools` (playground/interface/physical.py)
   exposes observation, `move/walk_to/climb/jump/set_joint_target/wait`, object verbs
   (`pick_up/put_down/throw/place/rotate_object/catch/push/pull/slide/nudge/kick/roll/
   stack_on/build_row`) and measurement tools — but the extension's gait/recovery/
   manipulation/skill layers were NOT agent-facing: no `run_to`, `turn_to`, `stop_moving`,
   standing, falling evidence, recovery, approach/reach, no skills access.
2. **Layer 5 gap.** `Environment._call` returns `success: true/false` + measured body, but
   there is no uniform lifecycle `status` vocabulary (completed / partial / failed /
   interrupted). Partial outcomes exist only as `ActionError` details
   (e.g. blocked walk: `distance_travelled`, `objects_disturbed` are already measured and
   logged — probe P5).
3. **No in-place turn exists in the rig.** `set_gait('stand', heading=X)` holds zero
   waist (probe P1/P2: torso yaw 0.0 deg, waist qpos 0.002 rad after 8 s). The heading
   servo exists only in the walk cycle (`GaitController._gait_targets` → waist yaw =
   heading − 90 rad, feet twist in place against ground friction). Standing never falls in
   8 s (z 0.802). => `turn_to` MUST ride a short real walk at the requested heading; a
   stand-mode heading servo would be a new locomotion implementation (forbidden).
4. **Bug found: ManipulationController.reach passes DEGREES into `set_joint_target`
   (radians)** (probe P4: shoulder target 15.0 slammed the joint to 1.759 rad, its limit
   ~100 deg). This is a bug fix, permitted; the controller's geometry otherwise stands.
5. **Recovery is measurable**: `GaitController.get_state()['recovery_done']` + events
   `recovery_started` / `recovery_complete` (0.5 s sustained upright) / `recovery_failed`
   (6 s timeout).
6. **Skills library exists but is unwired**: 16 `SkillSpec`s (catalog), `SkillsLibrary`
   (registry/invoke/status convention), `WorldCapabilities` capability adapter. Agents
   cannot call any of it today.
7. Speed semantics measured: `move(speed=…)` is the ONLY speed/heading pathway for
   humanoids (`run`/`walk` are the same controller, different requested speed; the sim
   yields what it yields and the payload reports measured m/s).

## Frozen decisions

### D1 — Control architecture (Layer 1–5)

- L1 LLM reasoning = existing session loop (`session/runner.run_session` + adapters).
  Unchanged.
- L2 Skill interface = `run_skill(name, **params)` + `run_chain(steps)` + composite
  `carry_to`, dispatched via `SkillsLibrary` against a new `AgentSkillAdapter`
  (playground/skills/agent_adapter.py) built on existing world/locomotion/manipulation
  APIs. Composability: `approach_object → reach_for → pick_up → carry → place`.
- L3 Physical controllers = `PhysicsWorld.move/set_gait/jump/climb/catch/…`,
  `GaitController`, `ManipulationController` (units fixed), recovery. No duplicates.
- L4 MuJoCo authoritative. No qpos/qvel writes anywhere in the new surface.
- L5 Feedback = uniform `status` on every `Environment._call` result:
  `completed` / `partial` / `failed` (+ `interrupted` reserved for chain aborts).
  Success is never reported because a function returned: movement/object results carry
  measured evidence (distance_travelled, arrival flags, held state, recovery events).

### D2 — Status vocabulary (additive, in `Environment._call`)

| code | condition |
|------|-----------|
| `completed` | success result (all categories) |
| `partial` | `ActionError` with measurable progress in details (`distance_travelled` present, or a caught/moved/rotated partial) |
| `failed` | `ToolError`, any `ActionError` without progress evidence |
| `interrupted` | run_chain ledger: steps after a failure are marked interrupted |

`requested`/`executing` are visible to the session/dashboards via the existing append-only
action log (`t_start`/`t_end`, args/result sha, streamed frames); agent calls are
synchronous by design. This is documented, not hidden.

### D3 — New agent-facing tools (schema + PhysicalTools wrappers)

| tool | category | rides |
|------|----------|-------|
| `run_to(x, y, speed)` | movement | `walk_to` at higher speed (existing gait only) |
| `turn_to(heading)` | movement | short real walk at heading via `move`; measured facing + drift; no fake servo |
| `stop_moving()` | movement | gait `stop` config + brake; measured velocity |
| `stand()` | movement | gait `stand` (neutral posture hold) |
| `stand_up()` | movement | `GaitController.begin_recovery` + measured recovery event |
| `approach_object(object_id, stop_short)` | movement | object position → `walk_to` offset |
| `reach_for(object_id)` | object | `ManipulationController.reach` (units fixed) + measured hand distance |
| `carry_to(object_id, x, y)` | object | pick_up → walk_to → place composite with per-step ledger |
| `run_skill(skill, **params)` | skill | `SkillsLibrary` + `AgentSkillAdapter` (16 skills) |
| `run_chain(steps)` | skill | ordered skills, per-step ledger, abort on failure, remainder interrupted |

`inspect_nearby_objects` = existing `look_around` (no duplicate tool; audit rule D0).

### D4 — Observation addition

`PhysicsWorld.agent_state()` gains `"fallen"` (measured `agent_fallen()`) so fall state is
first-class observation. Humanoid rig facts reach agents through observation, not through
tool descriptions (established convention).

### D5 — Ownership

- Phase owner edits: `interface/schema.py`, `interface/physical.py`, `interface/environment.py`,
  `skills/agent_adapter.py` (new), `skills/library.py` (additive `result` key),
  `ManipulationController` (units fix), `world/physics.py` (`agent_state` additive key).
- Tests: `tests/test_agent_physics_integration.py` (HVE: Happy/Validate/Error) written
  BEFORE behaviour; machine-independent.
- Docs: this file + BUGLOG entry (reach units bug; no-in-place-turn finding) +
  PHYSICS_EXTENSION tracker row (Phase 12).

### D6 — Free-will

Tool descriptions state capabilities and argument semantics only — never objectives,
never how-to guidance, no promises about physical outcomes. The island's autonomy
contract is untouched: agents choose; the surface only reports what physically happened.
