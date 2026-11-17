# PHYSICS EXTENSION — Phase Tracker

Implements "AI Playground — Advanced Physical Intelligence & Multi-Agent Simulation"
(extension of the existing autonomous playground; the original behaviour is preserved).

Every phase below lists its scope and the real command that verifies it. Status values:
`landed` (code + tests on the branch), `partial`, `pending`. The status column is filled
against evidence (test output), never against intent.

| Phase | Scope | Verification command | Status |
|-------|-------|----------------------|--------|
| 0 | Architecture investigation + contract | `docs/PHYSICS_EXTENSION_CONTRACT.md` (frozen interfaces, ownership) | landed |
| 1 | Articulated humanoid body + joint physics (MuJoCo MJCF, hinge set, motors) | `./.venv/Scripts/python.exe -m pytest tests/test_humanoid_world.py tests/test_humanoid_pd_walk.py -q` | landed |
| 2 | Smooth motor control: single PD application point, no teleport hacks | grep-free evidence: `pytest tests/test_humanoid_pd_walk.py -q`; code comment at the single PD loop in `playground/world/physics.py::_step` | landed |
| 3 | Walking / balance / recovery via contact physics (`playground/locomotion.py`, world gait API) | `pytest tests/test_locomotion.py -q` (displacement ≥ 0.15 m / 8 s, stand-up, heading) | landed — walk 0.80-0.96 m/8 s z-stable, heading 92 deg, ROLL/CATCH/EXIT recovery measured; see BUGLOG sec 7-8 |
| 4 | Reinforcement learning + curriculum (`playground/rl/`: env, rewards, curriculum, train, checkpoints) | `pytest tests/test_rl_core.py -q` (real-world smoke green; anti-exploit gated on Phase 3 gait) | landed — anti-exploit real test green with locomotion-grade env gains |
| 5 | Running / jumping / advanced locomotion | same as Phase 3 suite (speed scaling, heading rotation, crouch) | landed — displacement-based speed scaling monotone, heading +/-30 deg; jump = force-rocket payload (rig is a rigid column; documented stride cap) |
| 6 | Physical object manipulation (`playground/manipulation.py`: reach/grasp/push/pull, no teleport) | `pytest tests/test_manipulation.py -q` | landed (audit at gate) |
| 7 | Projectiles + impacts + explosions (`playground/combat/projectiles.py`, `impacts.py`, splat/impact FX) | `pytest tests/test_projectiles.py -q` | in verification |
| 8 | Equipment + throwable objects (`playground/combat/equipment.py` registry, grip, ammo, spread) | `pytest tests/test_equipment.py -q` | landed |
| 9 | Multi-agent arena (`playground/arena/`: match, scoring, modes, reports) | `pytest tests/test_arena.py -q` | landed (stub-agent level) |
| 10 | Human player integration (`playground/player.py`, `dashboard/player_api.py`, `static/js/player.js`) | `pytest tests/test_player.py -q` | landed (audit at gate) |
| 11 | Performance optimisation + full integration | `pytest tests/test_performance.py tests/test_mode_integration.py -q`; `python scripts/profile_step.py --benchmark`; `python scripts/health_check.py` | landed |

## Package verification table

| Package | Deliverables | Acceptance evidence |
|---------|--------------|---------------------|
| P1 locomotion-core | world API (set_gait/stand_up/agent_fallen/agent_skeleton/spawn_object/set_object_velocity/register_step_callback), GaitController, no teleport hacks, single PD path | `pytest tests/test_locomotion.py tests/test_humanoid_world.py tests/test_humanoid_pd_walk.py -q` + `scripts/demo_humanoid_walk.py` trace |
| P2 rl-core | `playground/rl/` env/rewards/curriculum/train + checkpoints + anti-exploit rewards | `pytest tests/test_rl_core.py -q` |
| P3 skills-library | `playground/skills/` SkillSpec/adapter/library/catalog(16) | `pytest tests/test_skills.py -q` |
| P4 manipulation | `playground/manipulation.py` contact-gated grasp, displacement continuity | `pytest tests/test_manipulation.py -q` |
| P5 projectiles | `playground/combat/projectiles.py`, `impacts.py`, `static/js/splat.js`, `static/js/impact_fx.js` | `pytest tests/test_projectiles.py -q` + `node --check` |
| P6 equipment | `playground/combat/equipment.py` registry + frozen launcher signature | `pytest tests/test_equipment.py -q` |
| P7 arena | `playground/arena/` match/scoring/modes/report export | `pytest tests/test_arena.py -q` |
| P8 human-player | `playground/player.py`, `dashboard/player_api.py`, `static/js/player.js` | `pytest tests/test_player.py -q` |
| P9 visualisation | `static/js/humanoid_view.js` + app.js/index.html wiring, debug overlays | `pytest tests/test_ui_humanoid.py tests/test_ml_engine_generators.py -q` + `node --check` |
| P10 qa/perf/docs | `scripts/profile_step.py --benchmark`, `tests/test_performance.py`, `tests/test_mode_integration.py`, `health_check.py` extension, this tracker | commands in Phase 11 row |

## Integration gate (parent)

1. `git diff --stat` reviewed against the contract ownership table.
2. Child patch requests applied (tools registration in `schema.py`/`physical.py`, server router include).
3. Full suite green + `scripts/health_check.py` run.
4. Dashboard smoke: server boots, `/api/health` ok, `skeleton` present in `/api/state`.
5. ONE commit by the parent at the end.
