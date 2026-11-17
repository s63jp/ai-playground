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

### Combat & equipment (mission §1.3) — M4 landed for rows 17–24; M5 landed rows 25–27

| # | Claim | Class | Evidence / gap |
|---|-------|-------|----------------|
| 17 | Multiple independently controlled humanoids in one world | 1 | M3 CLOSED (2026-11-17): `world/spec.py` N-body MJCF with per-agent prefixed names, `AgentRuntime` + `_ScopedModel`/`AgentView` scoping, `agent=` API; `tests/test_multiagent_world.py` 10/10; `scripts/profile_multiagent.py --agents 1 2 4` 2473/1604.6/818.3 steps/s — full evidence in "M3 — CLOSED" below |
| 18 | Equipment that agents physically pick up, hold, carry, use | 1 | M4: `combat/equipment.py` registry grew `combat_class`/`damage_scale`/`dims_m`/`shape` (7 items: 4 paintball guns, grenade, training baton, paint shield); `world/spec.py` builds REAL equipment + latent pellet-pool bodies from `cfg["combat"]` (no-combat worlds byte-identical to pre-M4); holds ride the real carry pin (`CombatRuntime.hold` → `world.pick_up`); fire/reload run through the real ProjectileSystem launcher; 7 combat tools in the shipped catalogue (`interface/schema.py` "combat" category) delegated by `PhysicalTools` → `combat_*` world methods; tests: `test_combat_world.py` 7, `test_equipment.py` 5, `test_skills.py` 5 |
| 19 | Melee weapons, shields | 1 | M4: training baton (melee, `charge_speed`) + paint shield (`guard_height`/`guard_forward`, absorb); strike = PD straight-arm thrust with the hand-pinned weapon's velocity written from the measured arm-tip interval (`_pin_held(measure=True)` — a kinematic hold alone is invisible to collisions); shield stays collidable while equipped (`_set_collidable` kept ON) and absorbs in the damage path; `defend("guard")` sets the guard pose; `test_combat_world.py` |
| 20 | Actual physical attacks and defensive actions | 1 | M4: `attack` = measured arm-swing strike (baton impulse ≈ 11.8 N·s; scale 0.8 ⇒ ≈9.4 damage/swing); `defend(stance)` = guard-pose rewrite of the carry-pin height/forward + shield absorb; honest limit: no dodge verb — mobility verbs (move/push) remain the movement options; `test_combat_world.py`, `test_combat_damage.py` |
| 21 | Projectile-based equipment | 1 | M4: `fire` parks nothing — pool pellets launch at the held launcher's measured muzzle via `spawn_object`+`set_object_velocity` into the real ProjectileSystem (`_unpark`+active.add); honest refusals: empty pellet pool, rate limit, reload window; grenade = real `world.throw`; `test_combat_world.py`, `test_projectiles.py` |
| 22 | Detect real impacts through MuJoCo contacts | 1 | M4: damage is read FROM MuJoCo contact events as the measured collision impulse (N·s); per-agent attribution names the exact struck body part from the contact geom pair (`_agent_parts` enrichment); the holder's own weapon pair is refused (solver-weld artifact, measured no-damage); `test_combat_damage.py` |
| 23 | Calculate damage from physical interactions + equipment | 1 | M4: damage = measured impulse × item `damage_scale` × part multiplier (head/neck 1.5, torso 1.0, hip 0.9, limbs 0.6) × shield absorb; floor `damage_min_impulse` 0.8 N·s (walk-in contacts stay below by physics — measured), cap 95; measured constants recorded in BUGLOG §10; `test_combat_damage.py` |
| 24 | Health, incapacitation, match termination | 1 | M4: per-agent health consumes the measured damage; incapacitation is PHYSICAL — real `put_down` + `incapacitate_rig` (gait None, that agent's PD entries deleted → gravity crumple, `set_gait` refuses them); `match_ended` emitted with the surviving winner when a team is eliminated (`_check_match`); `test_combat_match.py` 7 |
| 25 | Agents make independent tactical decisions in combat | 2 | M5: arena session modes exist and run end-to-end — `run_session(mode='arena_solo'\|'arena_coop')` with dedicated prompt registry entries (`ARENA_SOLO_SYSTEM`/`ARENA_COOP_SYSTEM`, never reused from exploration/benchmark), prompt id + system-prompt hash recorded per session, `run_agents(mode=...)` passthrough; recorded matches drove real combatants through the real combat tools (equip/look_around/fire) with per-shot measured aiming; scripted combatants so far — no LLM-driven arena match recorded yet (that is the remaining honest gap) |
| 26 | Duels and team-based encounters | 1 | M5: production factories (`arena/factories.py`, no stubs) build the REAL multi-agent combat world from the match cfg (`build_match_setup` → `PhysicsWorld` + `setup_agent_pose` + stand stance), translate the real combat events into the scorer with proven proj_id→(agent, team) attribution, and drive one recorded session per combatant on the match thread; `run_arena_match` wires plan+factories+scorer end to end with the runtime's measured KO arbitration; `playground arena` CLI (direct + subprocess verified); scored duel recorded 2026-11-17 (#120/#121: 8 shots → 8 measured torso hits, 8.9–9.6 damage each, per-agent stats); team-based: 2v2 worlds build, launch and record (coop modes + `agents_nearby` perception) — recorded coop *combat* runs and LLM combatants are the documented next work |
| 27 | Record and replay matches | 1 | M5: `ArenaMatch` records the real match-event timeline (every translated event, elapsed-stamped); `export_match_report` writes it (+ end_reason/sessions) — the `timeline: []` placeholder is gone; the dashboard lists arena **runs** (per-run grouping), opens any combatant session through the ordinary replay player (frames + three.js + scrubber: the #120/#121 duel replays at 800 frames), with `/api/arena` serving the run view; friendly mode labels (`1v1 duel` / `team match`) in the UI |

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
| 36 | Multiple agents sharing a world | **1** | M3 CLOSED (2026-11-17): `PhysicsWorld` accepts an agents list and builds the multi-agent MJCF; the runtime data plane still isolates identities/memories per agent (manager.py unchanged for separate worlds) — see "M3 — CLOSED" below |
| 37 | Preserve individual agent identities and memories | 1 | 13-agent roster, per-agent memory dirs + sessions |
| 38 | Record actions and physical events | 1 | action log + events, per-tool physics events |
| 39 | SHA-256-chained logs | 1 | append-only chain + `verify-log` CLI + `db.verify_chain` |
| 40 | Replay and measurable experiment results | 1 | deterministic replay (bitwise determinism, checkpoints incl. mid-flight); match replay via the recorded event timeline + per-session frame streams (M5) |
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

## M3 — CLOSED (2026-11-17, evidence-backed)

Root gaps #17 and #36 addressed. Multi-agent world architecture landed in
commit range 828a159..M3:

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
  delegation properties (incl. `_gait_controller`→`gait`, `ac`→`spec`, `agent_bid`→`bid`,
  `agent_gid`→`gid` aliases).
- `playground/interface/physical.py`: `PhysicalTools(world, agent_id=...)` binds via
  AgentView; `look_around` gains `agents_nearby` (multi only). `environment.py`
  `Environment(world=...)` world-injection seam + session-agent tool binding.
- Real MuJoCo contact filter verified working (minimal model: agent-agent push-apart).

**Measured evidence (M3 close):**

- `tests/test_multiagent_world.py`: **10/10 green** — spawn, mappings, independent
  walks, stop/continue, manipulation+attribution, isolation, legacy surface,
  cross-agent holds, determinism, agent-agent contact attribution. Contact fact
  (measured per-step): walking into a standing defender registers first MuJoCo
  agent-agent contact at ~0.41 m centre gap (t≈5.75 s from 0.72 m at walk speed
  1.0); a stroll never crosses the impact-event threshold, so the physical link is
  proven directly (min_gap + contact_seen) and the event branch is kept for real
  impacts (M4).
- Full-suite gate caught and fixed four integration regressions on the way
  (box-rig `agent_fallen` crash, three missing legacy delegation aliases, the
  `mj_setConst`-after-`mj_setState` state clobber, and a premature gap-based test
  break) — details and probe evidence in `BUGLOG_2026-11-17.md` §9. Final gate:
  **557 passed, 3 skipped**, three consecutive full-suite greens.
- Explicit Gate 1: `pytest tests/test_hve.py tests/test_smoke.py -v` → **83 passed**;
  `health_check.py` → `{"healthy": true, "failed": []}` (determinism PASS, 12.2× real time).
- Multi-agent speed (measured, `scripts/profile_multiagent.py --agents 1 2 4`,
  steady-state walking humanoids): **1 agent 2473 steps/s (4.9× real time),
  2 agents 1604.6 (3.2×), 4 agents 818.3 (1.6×)**; every agent upright (z 0.793)
  through the run, max |qacc| 49.4 (`experiments/multiagent_perf.json`).

**Still honest limits (documented, not hidden):** long straight walks drift
course (waist-twist vs foot friction — report-don't-raise); walk-into contact
stays under the event threshold by design; equipment/combat landed in M4
(below); arena factories are still the lightweight ones (M5).

## M4 — CLOSED (2026-11-17 late, evidence-backed)

Combat/equipment rows 18–24 (root rows 17/36 closed with M3, above) landed on
top of `0f8b9ee` in the in-flight commit `cbb5c3d` and are closed by this
milestone:

- `combat/equipment.py`: registry grew `combat_class` (ranged/throwable/melee/
  shield), `damage_scale` (damage points per measured impulse N·s), `dims_m` +
  `shape` for real bodies, `guard_height`/`guard_forward` (shield),
  `charge_speed` (melee); per-class `validate_equipment`; `EquipmentItem`
  gained `object_id` (the real equipped body) + `ammo_capacity`.
- `world/spec.py`: `cfg["combat"]` builds REAL equipment bodies + a latent
  parked pellet pool; a world WITHOUT `cfg["combat"]` is byte-identical to
  pre-M4 (default island untouched).
- `combat/runtime.py` (new): CombatRuntime rides the existing step callback —
  holds via the real carry pin; damage = measured collision impulse × item
  scale × part multiplier (head/neck 1.5, torso 1.0, hip 0.9, limbs 0.6) ×
  shield absorb, floor 0.8 N·s, cap 95; `fire` through the real
  ProjectileSystem pool; `attack` = PD straight-arm thrust with the hand-
  pinned weapon's measured velocity; `defend` = guard pose; health 0 → real
  `put_down` + `incapacitate_rig` (gait None, PD entries deleted → gravity
  crumple); `match_ended` with winner on elimination.
- `world/physics.py`: `setup_agent_pose` (honest facings), combat_* world
  methods + AgentView delegates, shields kept collidable while equipped,
  `incapacitate_rig`, `set_gait` refuses incapacitated agents, `_agent_parts`
  names the exact struck body part from the contact geom.
- `interface/schema.py` + `interface/physical.py`: 7 combat tools
  (equip/unequip/combat_state/fire/reload/attack/defend) registered in the
  shipped catalogue with facts-only text, delegated through the views.

**Measured evidence (M4 close):**

- Targeted 8 suites (combat world 7 / damage 6 / match 7 = the 20 contract
  tests, + equipment 5, projectiles 3, multiagent 10, arena 5, skills 5):
  **48/48 green**.
- Full suite TWICE consecutively: **579 passed, 1 skipped, 0 failed** (580
  collected, exit 0) in both runs — M3 baseline 557+3=560 plus the 20 M4
  tests; two formerly environment-skipped tests now run under M4's code.
- Explicit Gate 1: `pytest tests/test_hve.py tests/test_smoke.py -v` →
  **83 passed** in 48.97 s.
- `scripts/health_check.py` → `{"healthy": true, "failed": []}`; determinism
  PASS (hashes equal), physics speed 12.7× real time.
- `scripts/profile_multiagent.py --agents 1 2 4` → **2679.3 / 1661.9 / 878.9
  steps/s** (5.4× / 3.3× / 1.8× real time) — equal-or-better than the
  M3-close numbers (2473 / 1604.6 / 818.3), i.e. the M4 step-hook cost on
  non-combat worlds is unmeasurable (`combat is None` guard + 2 dict
  reads/step); all agents upright (z 0.793), max |qacc| 49.4.
- Physics pitfalls and measured damage constants:
  `BUGLOG_2026-11-17.md` §10.

**Still honest limits (M4, documented, not hidden):** no dodge verb (mobility
verbs remain the movement options); the arena is still library-only —
`ArenaMatch.play()` takes caller-injected factories, and the M5 factory's
input contract is exactly the `cfg["combat"]` shape built by `world/spec.py`;
no live arena/session run exists yet, so no "versus" capability claim is made
until M5 lands it.

## M5 — CLOSED (2026-11-17 late, evidence-backed)

M5 (arena orchestration) closed rows 25–27: production factories + arena
session modes + real scoring/replay, with the first recorded scored duel.
Delivered, verified by live runs (not stubs — the first worker attempt produced
a stub world and was rejected and rebuilt on the real engine):

- `arena/factories.py` (new): `build_match_setup` translates a match cfg into
  the real multi-agent combat world config (`cfg["combat"]`: equipment items
  within reach, latent pellet pool); `production_world_factory` builds the REAL
  `PhysicsWorld`, orients every combatant via `setup_agent_pose` and sets the
  `stand` combat stance; `ArenaWorldAdapter` steps the engine and translates the
  real events (`pellet_launched`/`agent_damaged`/`agent_incapacitated`/
  `match_ended`) into the scorer's impact contract with real proj_id→(agent,
  team) attribution (melee via the wielder of the striking item); the session
  launcher records every combatant through `run_session(mode='arena_solo'\
|'arena_coop')` on the SHARED world (`run_session(world=...)`), fans the
  world's events/frames out to all participating sessions, and drives one
  honest adapter turn per combatant between physics ticks (match-owned single
  thread — sessions never step physics concurrently); `run_arena_match` wires
  plan+factories+scorer end to end, augments the result with the runtime's
  measured KO arbitration (`end_reason`), and finalizes each session record.
- `playground/cli.py`: `playground arena match.json [--adapter --model --db
  --out --root]` — verified by direct call AND a real subprocess.
- `session/runner.py`: `world=` shared-world injection; `arena_solo`/
  `arena_coop` prompt dispatch (registry entries in `agents/prompts.py`,
  recorded prompt_id + system-prompt hash per session); `session/manager.py`
  `run_agents(mode=...)`.
- `arena/scoring.py`: proj_id→(shooter_agent, shooter_team) registry; per-agent
  `hits_dealt`/`hits_taken` (the 'cannot map proj_id→agent' hack and its
  apologia are gone); `arena/match.py`: REAL match-event timeline in
  `play()`/`step()` and in `export_match_report` (the `timeline: []`
  placeholder is gone), plus `end_reason`/`sessions` in the report.
- Dashboard: `/api/arena` matches grouped per RUN (15 s window), arena selector
  in the Watch tab, per-combatant rigs from the shared-world frames (friendly
  `1v1 duel`/`team match` labels), retry on boot races.
- `interface/physical.py`: `look_around` entries now carry measured `position`
  (the honest aiming surface for combatants).

**Measured duel (sessions #120/#121, recorded live 2026-11-17):** scripted
combatants equip, `look_around` (opponent + held muzzle in one observation),
solve LOS to the opposing torso centre, fire — **8 shots → 8 impacts (4 each
way), 8.9–9.6 damage per hit, per-agent stats credited, both rigs upright the
full 20 s** (final z 0.799; the 400→800-frame record replays in the dashboard).

**Gate (M5 close):** see §11 gate evidence in `BUGLOG_2026-11-17.md`.

**Still honest limits (M5, documented, not hidden):** combatants were scripted
(PolicyAdapter) — no LLM-driven arena match recorded yet; team-vs-team coop
runs recorded only with holding combatants (no live coop combat); the arena
world cfg passthrough of match `arena` geometry is spawn/layout keys only;
`look_around`'s new `position` field is additive and the bearing/distance
fields remain for older consumers.

## M5.5 — CLOSED (2026-11-17 evening, evidence-backed): arena visual polish + dashboard run modes

Presentation and integration milestone over M5 (no redesign; no physics changes):
every new pixel is driven by recorded data, and Run & Models now launches the
modes the backend actually runs.

- **Records:** every arena session now stores its match manifest —
  `arena/factories.py` passes `arena_context.match` {name, mode, teams,
  items(id→EQUIPMENT id), pellets} through `run_session`, which records it in
  the session `config_json` (`arena` key — additive; free-play sessions
  unchanged); `/api/sessions/{sid}` surfaces it as `arena_context`.
- **Run modes:** `GET /api/run/modes` serves the four backend-genuinely-supported
  modes (exploration / benchmark / arena_solo / arena_coop — grounded in
  runner dispatch + `session/benchmark.run_task` + `arena/factories.mode_for`)
  and the real EQUIPMENT registry (dims/class/grip/projectile radius). The
  dashboard's Run & Models card grew a mode selector with per-mode fields:
  free-play (as before), benchmark task runs (new `POST /api/run/benchmark`,
  one `run_task` session as a run job), and arena matches (new
  `POST /api/run/arena` → `run_arena_match` on the run-jobs thread with
  per-combatant adapter+model, team names, equipment, duration/pellets/
  score-target/friendly-fire, all validated server-side against the same
  factories the CLI uses). Submit → job poll → result banner → the match
  appears in the Watch & Replay arena selector. Refusals (bad mode/teams/
  equipment/layout mismatch/unknown adapter or task) are 400s before any job
  starts.
- **Visuals (all presentation over recorded physics — no invented motion):**
  `dashboard/static/js/combatfx.js`: paintball markers built from the real
  EQUIPMENT dims (receiver/barrel/hopper/grip/trigger guard + team-colour
  accents; variants for baton/shield/grenade/dual pistols), positioned by the
  item body's recorded pose each frame (held at the real carry pin — measured
  z 0.84 m / 0.30 m forward; dropped where the body rests); paintballs as
  glossy team-coloured spheres from the recorded `pellet_launched`
  attribution (shooter→team→colour); per-pellet trails drawn from recorded
  frame poses only (index-accurate on scrub, 64-sample window); ground paint
  decals from recorded pellet collision events (throttled per pellet,
  floor-height contacts); short-lived impact particles from recorded
  `agent_damaged` contact positions in the shooter's colour. Extra combatant
  rigs take their team colour from the manifest. Equipment stays a plain box
  only for recordings with neither manifest nor equip evidence (documented
  legacy gap, see BUGLOG §12c).

**Recorded evidence (this milestone):** duel #122/#123 (driven, measured aim,
13 shots→13 hits, report `experiments/arena_paint_demo_report.json`) and dashboard-
launched duel #124/#125 (scripted combatants end-to-end through
`POST /api/run/arena` → job `15c2a1d1` → Watch & Replay); match #126/#127
re-verified post-fix. Replay of #122/#127 renders markers at the body poses,
attribution-coloured pellets (14), 16 trail lines, and paint decals at the
recorded contact points (browser-verified with the live server).

**Gate (M5.5 close):** full suite ×2 consecutive: run 1 `run_tests.sh`
**612 passed / 0 failed / 3 environmental skips of 615** (11 new
`test_arena_dashboard_m55.py` tests + 604 prior; skips named: LM Studio
offline ×2, dataset absent — unchanged from M5) with
`health_check {"healthy": true, "failed": []}`; run 2 pytest full exit 0
(3 skips, all dots); explicit HVE+smoke **83 passed in 52.05 s**;
`git diff --check` clean.
