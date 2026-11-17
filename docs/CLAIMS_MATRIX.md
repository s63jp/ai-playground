# CLAIMS MATRIX — implementation baseline (mission §2)

Mission: `docs/specifications/claims-remediation-mission.md` ("Make AI Experimental Playground
Deliver Everything Its README Claims"). This file classifies every mission/README capability
against actual implementation, tests, runtime evidence and integration state. Classifications:

1 = Working and verified  ·  2 = Implemented but unreliable  ·  3 = Partially implemented
4 = Present as scaffolding only  ·  5 = Missing entirely

Statuses grounded in: code inspection (`playground/**`), the test suite
(`run_tests.sh`, baseline run 2026-11-17, numbers below), `docs/PHYSICS_EXTENSION.md`,
`docs/AGENT_PHYSICAL_INTEGRATION.md`, `docs/BUGLOG_2026-11-17*.md`. "Verify" = real
simulator run shown, not a function returning.

## Baseline (branch `feature/claims-remediation` @ 0348f95)

- Working tree clean at audit time (only untracked: MUJOCO_LOG.TXT).
- Full suite `bash run_tests.sh` (2026-11-17 12:0x, 294.70 s): **549 passed, 1 skipped,
  0 failed** (550 total).
- `scripts/health_check.py`: `{"healthy": true, "failed": []}` — imports ok (mujoco
  4.0.1), physics config 7f2ec2696a17, determinism `4881acec8425` twice, 23.3× real
  time, hash chain ok, 18 adapters, LM Studio up (19 models), physics extension
  modules 10/10 importable.

## Matrix

### Body & movement (mission §1.1)

| # | Claim | Class | Evidence / gap |
|---|-------|-------|----------------|
| 1 | Stand and maintain balance | 1 | `tests/test_locomotion.py` (stand 8 s, z 0.802 no fall); GaitController posture servos; BUGLOG sec 3 |
| 2 | Walk and run, stable controlled locomotion | 1 | contact-driven walk measured 0.80–0.96 m/8 s, z held 0.79, `tests/test_humanoid_pd_walk.py`, `tests/test_locomotion.py`; walk lab suite `scripts/humanoid_walk_lab.py --suite all` |
| 3 | Accelerate / decelerate / change direction | 1 | displacement-based speed scaling monotone (Phase 5); brake path `stop_moving`; measured m/s in payload |
| 4 | Turn towards targets | 1 | `turn_to` rides a short real walk arc; ±30° heading verified. Honest rig limit published: no in-place turn servo (waist twist only within stride) |
| 5 | Jump and land | 1 | `jump()` blocked until landing, analytic-ballistics match 0.8115 vs 0.8155 m; landing skill. Documented cap: stride-jump impossible on rigid column — payload rocket only |
| 6 | Fall naturally | 1 | fold-sit at ~0.42 m; no scripted fall animation; `agent_fallen()` fires on 0.3 s sustained condition |
| 7 | Detect when fallen | 1 | `agent_fallen()` + `world_state` `fallen` field, measured |
| 8 | Recover through physical actuation | 1 | ROLL→CATCH→EXIT joint schedule via PD; `recovery_complete` only after 0.5 s sustained upright. Honest limit: deep crumples stay wedged (armless rig) |
| 9 | Navigate uneven terrain and obstacles | 2 | climb exists (steps/spire), walk verdicts measured (`blocked`/deflected/`completed`, objects shoved/rolled); no ground-height variance beyond island geometry yet |

### Object interaction (mission §1.2)

| # | Claim | Class | Evidence / gap |
|---|-------|-------|----------------|
| 10 | Reach for objects | 1 | 2-link IK + measured joint tracking + measured shortfall; Phase-12 units bug fixed |
| 11 | Grasp and release | 1 | contact-gated grip `tests/test_manipulation.py` |
| 12 | Lift and carry | 1 | chest hold-pin; load added to body mass, turns with figure |
| 13 | Push and pull | 1 | push/pull/slide/nudge/kick/roll verbs, Newton's-third-law paths |
| 14 | Throw with controlled force and direction | 1 | release-velocity physics; recoil measured; throw_height benchmark task |
| 15 | Catch moving objects | 1 | `catch()` measured interception |
| 16 | Objects of different size/weight/properties | 1 | 12 blocks + rock (rolls) + mass-dependent behaviour (carry, throw recoil, tower drift) |

### Combat & equipment (mission §1.3) — the big gap

| # | Claim | Class | Evidence / gap |
|---|-------|-------|----------------|
| 17 | Multiple independently controlled humanoids in one world | **5** | `PhysicsWorld` owns exactly one `agent_body` (cfg key, single rig in `world/spec.py` MJCF); player/RL/arena all assume the single agent. **Root dependency for 18–28.** |
| 18 | Equipment that agents physically pick up, hold, carry, use | 4 | `combat/equipment.py` registry (mass/grip/ammo/spread) is real data + real fire/reload math, but grip = boolean flags — no MuJoCo attachment; `equip/fire/reload` agent tools are placeholder stubs returning `{"status":"ok"}`; absent from the 65-tool schema. Only the grenade `throw` delegates to real `world.throw` |
| 19 | Melee weapons, shields | 5 | registry contains paintball guns + grenade only; no melee spec, no shield/parry mechanics |
| 20 | Actual physical attacks and defensive actions | 3 | projectile launches + impacts are real (P5); kick/push usable offensively; no melee strike, block or dodge semantics |
| 21 | Projectile-based equipment | 2 | ProjectileSystem real: launches by initial velocity, contact impacts, splat FX; equipment launcher signature frozen; but no agent-facing surface beyond world-level |
| 22 | Detect real impacts through MuJoCo contacts | 2 | `projectile_impact` events real; no contact-force damage mapping on agent bodies (single body today) |
| 23 | Calculate damage from physical interactions + equipment | **5** | no damage model anywhere in `playground/**` |
| 24 | Health, incapacitation, match termination | **5** | no health state; scorer has victory-by-hits only (stub) |
| 25 | Agents make independent tactical decisions in combat | 4 | LLM loop + tools exist generally; no arena session mode exists to place an agent in one (steered design: arena solo + arena coop prompt variants, see M5) |
| 26 | Duels and team-based encounters | 4 | `arena/` ArenaMatch/Scorer/modes with validated plans — but play() is injected-callable orchestration tested with fake worlds/sessions only; scorer maps proj_id→agent by its own admission |
| 27 | Record and replay matches | 3 | per-session replay real (frames + three.js + checkpoints); match-format report JSON has `timeline: []` placeholder |

### Intelligence & learning (mission §1.4)

| # | Claim | Class | Evidence / gap |
|---|-------|-------|----------------|
| 28 | LLM agents make independent high-level decisions | 1 | adapters (OpenAI-compat/Anthropic/policy/external) + session loop; real models ran exploration end-to-end; provider errors surface as adapter errors |
| 29 | Physical controllers handle high-frequency movement/balance | 1 | PD/gait/manipulation under the hood; LLM never in the control loop (D1 architecture) |
| 30 | RL can train and improve physical behaviours | 2 | env (14 DoF), rewards + anti-exploit (regression-tested), curriculum, checkpoints, CLI all present; full train → measured-improvement run is operational work, not yet recorded |
| 31 | Reusable physical skills developed and executed | 2 | 16-skill catalogue + `run_skill`/`run_chain` with ledger (executed); *development* of skills is RL-adjacent and open |
| 32 | Agents retain individual written memories across sessions | 1 | MemoryStore: diary/handbook with evidence-gated statuses, tombstones, ablation support; per-agent isolation proven |
| 33 | Previous experiences influence later decisions | 2 | handbook digests ride the prompt; continuation sessions carry open questions; benchmark Group A/B measures experienced vs fresh |
| 34 | Experiments measure whether memory/learning improve performance | 2 | benchmark/ablate/compare-models pipeline + leaderboards exist and run; README honest status: no LLM train→deploy→verify loop completed yet |

### Simulation & evidence (mission §1.5)

| # | Claim | Class | Evidence / gap |
|---|-------|-------|----------------|
| 35 | Preserve 3D island + free exploration | 1 | default mode `playground` unchanged; `simulation.mode` documented, default test |
| 36 | Multiple agents sharing a world | **5** | same root gap as 17; today multiple agents run in *separate* worlds (manager.py: "Agents never share anything") |
| 37 | Preserve individual agent identities and memories | 1 | 13-agent roster, per-agent memory dirs + sessions |
| 38 | Record actions and physical events | 1 | action log + events, per-tool physics events |
| 39 | SHA-256-chained logs | 1 | append-only chain + `verify-log` CLI + `db.verify_chain` |
| 40 | Replay and measurable experiment results | 1 (sessions) / 3 (matches) | deterministic replay (bitwise determinism, checkpoints incl. mid-flight); match replay absent until 27 closes |
| 41 | Provider independence + structured tool interface | 1 | 65-tool catalogue in `interface/schema.py` (dynamically enumerated, no hardcoded counts in new code) |

## Independent verifiers / cross-checks used

- `scripts/health_check.py` — module import + physics determinism + extension modules.
- `scripts/humanoid_walk_lab.py --suite all` — standing/walking evidence matrix (live run).
- `scripts/engine_probe.py`, `scripts/profile_step.py --benchmark` — README island numbers.
- `tests/test_mode_integration.py` — mode keys documented, integration points importable.
- DB: `experiments/playground.db` — sessions/actions/events/frames, hash chain.

## Dependency-ordered sequence (mission §3)

```
M1 verify physics integrity  ──► M2 verify interaction  ──► M3 multi-agent world
                                                            │
                                              M4 equipment & combat
                                                            │
                                              M5 arena orchestration (solo/coop prompts)
                                                            │
                                              M6 learning integration
```

- **M1/M2**: largely *landed* on this branch (above, classes 1–2) — audit verifies with real
  runs; gaps only where class 2/3 (terrain variety #9, damage #23).
- **M3 multi-agent world** — root gap (#17, #36). Extends `world/spec.py` MJCF generation to
  N agent bodies + per-agent actuators/observations/gait controllers; single-agent API
  preserved (default agent alias). Per-agent identity, tool surface, memory unchanged.
- **M4 equipment & combat** — needs M3 for attacker/target; real attachment (weld/pin),
  registry extended with melee + shield specs, damage from measured impacts, health/
  incapacitation/match-termination, agent tools added to the schema (count grows
  dynamically — no hardcoded counts).
- **M5 arena orchestration** — needs M3+M4. Production factories for `ArenaMatch`
  (world_factory → multi-agent PhysicsWorld; session_launcher → `run_session`
  mode='arena'); **separate session prompts for arena solo and arena coop** (steered
  2026-11-17: new registry entries in `agents/prompts.py`, recorded+hashed per session,
  free-will leak tests extended); `playground arena` CLI; report JSON with real timeline +
  DB persistence + replay.
- **M6 learning integration** — benchmark/ablate extended with arena score as an outcome;
  demonstrable train→deploy→verify loop (RL checkpoint → scripted/nn controller in a
  session → measured improvement) and memory-improvement evidence runs.

## Milestone 0 (this audit): what must not be mistaken for working

1. Arena (`arena/match.py`) self-describes as deliberately lightweight: production factories
   "will pass real factories" — they did not arrive. Scorer: "tests will populate it
   manually", "cannot reliably map proj_id → agent".
2. Equipment agent tools are inline no-op stubs (`{"status": "ok"}`) — never registered in
   the real catalogue; the file's own comment says so.
3. `simulation.mode` values are documented but nothing dispatches on the value.
4. Existing 547-suite numbers include stub-level unit tests (arena/equipment) that do not
   constitute integration evidence for 17–27.

## M3 progress (2026-11-17, in flight — see session handoff summary)

Commit range 828a159..(M3 WIP). Multi-agent world architecture landed:

- `world/spec.py`: N-body MJCF via `agent_rigs()` + `_prefix_names()` (per-agent prefixed
  names e.g. `agent_002:left_hip`); legacy single-rig output byte-identical (verified vs
  `git show HEAD:` build_mjcf output). **BUG FIX shipped**: the merged humanoid joint
  default (`limited="true" range="-90 90"`) clamped the root slide joints to ±1.5708 m —
  the agent could never move farther than 1.57 m from spawn on ANY axis (pre-existing at
  HEAD, hidden because every test walk < 1 m). Slides now explicitly `limited="false"`.
- `world/physics.py`: `AgentRuntime` per-agent state; `_ScopedModel`+`AgentView` scoping
  (the proven GaitController/ManipulationController code rides unchanged through views);
  `agent=` kwarg across the agent-facing API (move/jump/climb/wait/catch/pick_up/throw/
  impulse/place/put_down/rotate/setup_agent/set_gait/stand_up/agent_*/feet/total_mass);
  per-rt `_step`/`_post_step`; multi-aware collisions (objects name agent ids; contact
  normal added in multi), `agents_report()` (measured other-agent perception),
  `held_by_any()`, exclusive holds (`object_taken` naming the holder), get/set_state with
  per-agent gait checkpoint (GaitController.get_state/set_state exist), legacy-compat
  delegation properties (incl. `_gait_controller`→`gait` alias).
- `playground/interface/physical.py`: `PhysicalTools(world, agent_id=...)` binds via
  AgentView; `look_around` gains `agents_nearby` (multi only). `environment.py`
  `Environment(world=...)` world-injection seam + session-agent tool binding.
- Real MuJoCo contact filter verified working (minimal model: agent-agent push-apart).

`tests/test_multiagent_world.py` (16 tests): 15 green, 1 failing —
`test_e_agent_agent_contact_attribution` (walker-vs-standing-defender contact not yet
detected at 14 s); first task on resume. Targeted green: locomotion/pd_walk/world suites
still pass on the refactored engine.

Remaining M3: E contact test, full-suite run, `scripts/profile_multiagent.py` numbers
(1/2/4 humanoids), `git diff --check`, CLAIMS_MATRIX update, milestone commit.
