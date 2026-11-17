# M6-B Stage B1 — Physical Safety Foundation (evidence report)

**Date:** 2026-11-17 · **Stage:** B1 of M6-B (Neural Brain Foundation) · **Baseline:** `f4bf8bf` (M6-A audit HEAD)

## 1. Corrected assumption: asset XML ctrlranges are NOT this rig's torque limits

The M6-A audit proposed capping per-joint torque at the asset XML's motor `ctrlrange`
values (hips ±30, knees ±24, spine ±30, shoulders ±15, elbows ±8, head ±3 N·m),
assuming "the asset's motor ctrlranges ... prove the intended physical magnitude scale".
That assumption is **wrong for this controller stack** and is corrected here:

- The compiled humanoid model has **no actuators** (`m.nu == 0`): the XML motors are
  never instantiated, so the ctrlranges are unreachable hardware specification, not
  runtime limits.
- There is **no gear or transmission**: the PD loop (`PhysicsWorld._step`) writes
  generalized force directly into `d.qfrc_applied[dof]` — torque in N·m on each hinge
  dof, force in N on the root slide dofs (which are bounded separately by the
  `walk_force` traction limit).
- The existing PD controllers command their own much larger scale, which **measurement
  confirms** (raw envelope, caps disabled during measurement,
  `ai-playground-audit/probe_torque_results.json`):

| Scenario (scripted controller) | ankle | hip | knee | waist_yaw | spine | head | shoulder/elbow | foot_yaw |
|---|---|---|---|---|---|---|---|---|
| stand (10 s) | 0 | 0.1 | 0.2 | 0 | 0 | 0 | 0 | 0 |
| walk 1.0 m/s | **315–327** | 42–52 | 51–67 | **41.3** | 6.6 | 7.2 | 2.3/1.3 | 3.5 |
| run 2.0 m/s | **316** | 51 | 67 | 38.8 | 9.2 | 10.6 | 2.6/0.9 | 3.0 |
| turn 45° / 120° at 1.0 m/s | 305–314 | — | 63–73 | ≤ 23.7 | — | — | — | — |
| recovery (deep-fold pose, kp 80) | 5.5 | **310** | 243 | 0 | 5.3 | 7.9 | **8.6** | 0 |
| recovery (canonical fold, kp 150 — test_locomotion pose) | — | ≤ 400 | **452.3** | — | **41.9** | 7.9 | ≤ 8.6 | — |
| M4 melee baton strike (swing raise kp 60/48) | — | — | — | — | — | — | **124.3 / 166.3** | — |

Probes sample every physics step (via `register_step_callback` so controller-internal
stepping loops are included — an earlier probe that sampled only after each attack
returned under-measured the combat swing as "17 N·m"; corrected above). The combat
envelope lives in `ai-playground-audit/combat_envelope_probe.json`.

## 2. Configured ceilings (config/default.yaml → `world.safety`)

Caps sit **above every measured scripted raw peak** (the small-torque families get a
modest absolute value; the high-torque families keep ~1.2–1.43× margin) and far below
the runaway scale demonstrated in the audit (~10⁶ N·m):

| Joint family | Configured cap (N·m) | Measured legitimate peak | Margin | Basis |
|---|---|---|---|---|
| elbow | 220 | 166.3 (combat strike raise/strike transients) | 1.32× | combat swing, per-step sampled |
| knee | 570 | 452.3 (recovery catch, kp 250) | 1.26× | canonical recovery |
| shoulder | 160 | 124.3 (combat strike) | 1.29× | combat swing, per-step sampled |
| ankle | 400 | 326.7 (walk stance, kp 600) | 1.22× | |
| hip | 400 | 310.0 (recovery, kp 200) | 1.29× | |
| default | 600 | — | — | fallback for unlisted joints (hip-class bound) |
| waist_yaw | 120 | 41.3 | 2.9× | low-torque family: modest absolute bound |
| spine_L2 | 60 | 41.9 | 1.43× | |
| head | 20 | 10.6 | 1.9× | |
| foot_yaw | 20 | 3.5 | 5.7× | |

Note: with the final caps nothing legitimately scripted clips (verified by regression
tests, below). Two earlier draft ceilings DID clip legitimate demand and were corrected
from measurement, not raised blindly: knee 320 (raised to 570 — the canonical recovery's
catch stage demands 452.3 N·m, measured); shoulder 40 / elbow 30 (raised to 160 / 220 —
the combat strike's raise/strike transients demand 124.3 / 166.3 N·m, per-step sampled).
In both cases the clipping measurably altered controller behaviour (recovery catch
dynamics; strike self-contact impulse 15.92 vs 17.70 N·s clean), which is what the
regressions caught.

## 2b. Full-suite regression incident during B1 (root cause + fix)

The first B1 gate run failed `test_combat_match::test_b_melee_attack_needs_a_melee_weapon
_and_respects_rate_limit`: the melee attacker was KO'd by its own weapon during the FIRST
strike. Root cause chain, traced from the safety log:

1. **First divergence — `clamp_target: true` (an incorrect default, since corrected):**
   the combat raise commands the elbow to **2.35 rad by design**, beyond the model's
   ±π/2 (1.5708) limit — the controller relies on MuJoCo's joint-limit stop as part of
   its measured dynamics. My initial default rewrote that target every raise step
   (~step 150 onward: `joint_target_clamped`, joint `agent_001:left_elbow`, requested
   1.577→1.65+ → clamped 1.5708, source "unspecified"), changing the contact-force
   profile of the whole raise.
2. **Secondary echo (the misleading 868/1674 N·m readings):** once the raise trajectory
   diverged, contact chaos amplified the PD error terms — the recorded saturation
   `max_raw` of 868 (shoulder) / 1673 (elbow) were demands of the DIVERGED trajectory,
   not the legitimate swing. With the default corrected (`clamp_target: false`), the
   same clean sequence measures a legitimate envelope of only 124.3 / 166.3 N·m.
3. **Residual divergence — torque caps on the swing transients:** the elbow cap (30 at
   the time) clipped the clean strike's transient at step 226 (t=0.452, raw −59.67 vs
   limit 30.0, joint `agent_001:left_elbow`, source "unspecified"), again altering the
   contact trajectory (self-contact impulse 15.92 vs 17.70 N·s with caps fully off).
   Final caps (160/220) clear the measured envelope, and the capped run now reproduces
   the caps-off run's contacts **exactly** (bit-identical physics; determinism verified
   by repeated runs).

Fix: `clamp_target` is a configurable mode defaulting OFF (safety does not depend on
it — the torque caps bound the force of any finite target and non-finite values are
rejected), and the shoulder/elbow caps were re-derived from per-step-sampled
measurements of the clean combat envelope. Verified: combat suite green repeatedly;
B1 regressions re-run green; walking/running/recovery envelope probes re-run.

## 2c. Pristine-checkpoint cross-verification (standalone)

Beyond the in-suite regressions, B1 physics was compared against the actual pristine
`f4bf8bf` checkout in an isolated git worktree, running the M4 melee scenario verbatim
(`ai-playground-audit/pristine_compare.py`; every leg executed under `python -I` with
only that tree's root on `sys.path`) — this is what the in-suite capped-vs-caps-off
regression alone cannot prove, namely that the B1 code changes do not alter the combat
physics that `f4bf8bf` shipped:

| Leg | Code under test | Contact impulses (N·s, attack1 @ t = 0.47) | Saturation / warnings |
|---|---|---|---|
| pristine native | f4bf8bf worktree (no safety wiring exists) | 17.7047 (to agent_001) / 11.8241 (agent_002) | — (no surface) |
| B1 safety-forced-off | B1 repo, ceilings 1e9, clamp off | 17.7047 / 11.8241 | { } / { } |
| B1 native defaults | B1 repo, clamp_target off, caps 160/220 | 17.7047 / 11.8241 | { } / { } |

All three legs are **bit-identical** (raw full-precision floats compared, not the
rounded probe values) and match `ai-playground-audit/combat_envelope_probe.json`
"clean"; no leg incapacitates either agent; `attack3` registers `attacked` in all
legs. This proves the B1 wiring is numerically inert when inactive *and* that the
shipped defaults do not clip the legitimate swing. Evidence:
`ai-playground-audit/pristine_compare_results.txt`; the worktree was removed after
the run.

## 3. Safety controls implemented

All controls live at the world's single joint-force hook (`PhysicsWorld.set_joint_target`
→ `_step` PD loop), so every caller is protected — scripted controllers, `set_joint_target`
tool (LLM/HTTP), and the RL env all get the identical contract:

1. **Input validation** (`physics.py::set_joint_target`): non-finite `joint`, `position`,
   `kp`, `kd` (NaN, ±Inf, or non-numeric — numpy scalars excluded from the "string" trap)
   raise `ActionError("invalid_joint_target", ...)`; nothing is stored; invalid values are
   never silently converted. Negative gains clamp to 0 (they would be positive feedback).
2. **Gain ceilings** (`kp_ceiling` 1000 / `kd_ceiling` 100): beyond-ceiling gains clamp
   with a `joint_gains_clamped` event. Bounded gains + torque cap ⇒ bounded force.
3. **Target saturation** (`clamp_target`): finite targets beyond a limited joint's real
   range saturate into `jnt_range` (actuator-ctrlrange semantics) with a
   `joint_target_clamped` event. Unknown joints keep the legacy step-time drop (box-agent
   worlds unaffected).
4. **Per-joint torque caps** in the PD loop, applied to the actual written torque before
   `d.qfrc_applied[dofadr] += torque` (hinge dofs only): `joint_torque_saturated` events
   with per-source attribution; saturation stats record **frequency** (events + saturation
   steps), **duration** (`duration_s` = steps × dt) and **magnitude** (`max_raw`, N·m).
5. **Command-source attribution**: `set_joint_target(..., source=...)` labels the
   demanding controller. The RL env tags `"neural_policy"`, the manual tool `"tool"`,
   untagged internal callers remain `"unspecified"` — so scripted and neural demands are
   distinguishable in the saturation record (`safety_stats().torque_saturation[j].sources`).
   Neural-policy demands bounded by the env's own gain schedule (~60/25/35/25/15/10 kp) can
   never exceed ~95 N·m even at full output — an order of magnitude below any cap.
6. **MuJoCo warning watchdog** (`_check_physics_warnings`, run after every `mj_step`): the
   seven `mjWARN_*` counters (incl. `mjWARN_BADQACC`) are diffed each step; new warnings
   emit a structured `physics_warning` event attributed to the owning agent(s) and the
   commanding source(s) of recently-commanded joints (degrades honestly to `["none"]` if
   no source can be identified, e.g. direct-`_pd` writes). Previously this failure mode was
   silent: MuJoCo suppressed the affected step's physics with no observable trace.
7. **Environment boundary guards** (`rl/env.py::LocoEnv.step`): non-finite action vectors
   or wrong shapes are rejected (`ActionError("non_finite_action" / "invalid_action_shape")`)
   before any mapping; finite out-of-range actions are clipped into the documented
   encoding boundary [-1, 1] with per-episode clip statistics
   (`LocoEnv.action_clipping()` — call count, clipped-call count, element count, max
   elements in one call, and the clipping rate so excessive clipping is detectable).
   After every step, new MuJoCo warnings or a non-finite observation mark the transition
   **FAULTED** (`info['faulted']`, zero reward, `done=True`) — invalid physics must not
   enter training data (enforced by contract; the B3 trainer consumes it).
8. **Tool/HTTP boundary** (`interface/schema.py::normalise_args`): JSON admits NaN/Infinity
   literals; number-typed arguments are now rejected with a structured
   `ToolError("invalid_argument")` before any tool body runs. `Environment.call` surfaces
   world-level `ActionError` as `{"success": False, "error": {code, message}}` as before
   (verified by test). The tool wrapper now forwards omitted gains to the world defaults
   (kp 15 / kd 1.5) instead of a silent zero-gain (force-less) registration
   (M6-A audit defect #10), and tags `source="tool"`.
9. **`CATEGORIES`** in `interface/schema.py` now includes `"control"` and `"skill"`
   (both used by real tools; M6-A audit finding #11 — stale category enumeration).
10. **Observability API**: `PhysicsWorld.safety_stats()` (counters: rejects, target/gain
    clamps, per-joint saturation with sources, per-name warning counts),
    `physics_warnings()`, `safety_log()` (bounded structured event log,
    recording-independent so RL training worlds without frame recording still expose it).

## 4. Tests (tests/test_safety_b1.py — 29 tests, all passing)

Rejection: NaN/±Inf position/kp/kd each rejected with `invalid_joint_target`, nothing
stored, no warnings; invalid joint name rejected. Saturation: 1e30 target saturates into
jnt_range with `joint_target_clamped`; kp=1e6/kd=1e4 clamp to ceilings with
`joint_gains_clamped`; negative gains clamp to 0. Torque caps: resolution order
(exact/left-right-family/head_*/default/multi-agent prefix) asserted; **property test**
(1000 steps of a hostile random policy over-range targets, over-ceiling gains) keeps every
|qfrc_applied| ≤ cap with zero physics warnings, saturation observed and recorded with
per-source attribution. Env boundary: NaN/±Inf/wrong-shape actions rejected with structured
codes before any physical effect; poisoned-weights network (NaN output) rejected at the
same guard; finite out-of-range actions clipped into [-1,1] with the clip rate observable.
BADQACC: a rogue API command (clamps/caps disabled in cfg, kp=1e6 target=1e30 — the audit's
demonstrated failure) surfaces `mjWARN_BADQACC` in `physics_warning` events attributed to
agent `agent_001` and source `bypass_probe`; a direct `_pd` write degrades attribution
honestly to `["none"]`; the env flags the post-warning transition FAULTED (done, zero
reward, `info['physics_warnings']`). Tool path: NaN through `Environment.call` returns a
structured error and never reaches the world; JSON `NaN` literal rejected by the schema
validator; bare tool calls take world-default gains and tag `source="tool"`.
Regressions: scripted walk displacement **bit-identical** (|Δd| < 1e-6 m over 8 s) with
caps on vs off and nothing saturated; canonical collapse+recovery completes (z ≥ 0.55,
sustained posture hold, zero clipping); recovery after a corrected target runs clean.

## 5. Physical probes (ai-playground-audit/)

- `probe_torque.py` → `probe_torque_results.json`: raw per-joint force envelope across
  stand / walk / run / 45°+120° turning / both recovery poses, with the force-path
  verification (`nu=0`, direct `d.qfrc_applied` write, dt 0.002), run with caps disabled.
- Interactive probes: BADQACC triggers at 1e30-target scale; recovery outcome z=0.802
  with caps on; z-threshold bounce behavior of the debounced `agent_fallen` flag
  documented in the recovery test.

## 6. Outstanding safety limitations (honest)

1. **Slide-dof forces are not torque-capped** (root traction, bounded by `walk_force`
   300 N config; jump 2200 N config) — existing config bounds, out of the PD route's scope.
2. `set_object_velocity` is still unguarded (M6-A D1.4; setup/admin scope, not a session
   tool) — untouched in B1; flagged for a later stage.
3. `d.warning` covers the seven MuJoCo warning types; per-**dof** attribution inside a
   multi-agent world is coarse (agent-level attribution only).
4. A caller holding a direct reference `_step`/`_pd` bypasses validation by construction —
   the watchdog makes even that path observable (tests demonstrate attribution degrading
   to `["none"]`), but it cannot be prevented at the API level.
5. The gain ceilings (kp 1000 / kd 100) are engineering bounds, not measurements: any
   scripted gain up to 600 kp is admitted, and the torque caps bound the force either way.

## 7. B1 gate result (final)

- **Full-suite regression**: 644 collected → **643 passed, 1 skipped** (`test_ml_training.py:320`
  real-dataset unavailable; the LM Studio integration tests ran and passed this time).
  Genuine exit code **0** — `ai-playground-audit/fullsuite_b1_rerun.txt`. (Counts verified
  via `--collect-only` because this run's final one-line pytest summary did not flush to the
  redirect; the arithmetic is exact: pristine baseline 615 collected = 612 passed + 3 skips,
  plus the 29 new B1 tests, with the two LM Studio skips promoted to passes.)
- **Pristine cross-verification** (§2c): bit-identical combat contacts across
  pristine-native / B1-safety-off / B1-defaults legs — 17.7047 / 11.8241 N·s.
- **In-suite closures**: combat determinism ×3, scripted walk bit-identical under caps with
  zero saturation, canonical collapse+recovery completes unclipped, tool/env boundary
  guards against non-finite commands (all in `tests/test_safety_b1.py` + `tests/test_combat_match.py`).

B1 gate **closed** on 2026-11-17; next stage B2 (RL environment) awaits approval.
