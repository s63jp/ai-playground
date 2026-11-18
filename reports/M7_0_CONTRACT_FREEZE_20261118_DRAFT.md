# M7-0 — Contract Freeze Specification (DRAFT — pending Jamie review/approval)

**Date:** 2026-11-18 · **Status:** DRAFT. **No implementation has begun; no env/trainer code has been touched.** This document is the reviewable object for the M7-0 approval gate, per Jamie's decision ("review the actual design document and probe evidence, then approve M7-0 before touching the training environment").

It freezes, concretely, the six things Jamie requires before limb training begins:

1. the final observation and action contracts for the foundational curriculum (§1–2);
2. which joints each limb-control task can command (§3);
3. how individual limb capability is measured (§4);
4. how balance and coordinated standing are objectively assessed (§5);
5. what constitutes passing each capability gate (§6);
6. how visual inspection supports the numerical evidence (§7).

Plus the three-operation loading law (§8) and the M7-0 execution plan (§9). All geometry/normalisation constants are **anchored to measured values** (probe `reports/M7_AUDIT_PROBE.json`) — none invented. Where a pass band must be calibrated against behaviour that does not exist yet, the freeze defines the **metric, protocol and calibration row**; the number itself is set at the gate from measured rows + stated margin (the project's measured-numbers rule).

## 1. Observation contract v4 (`rl_env/4`) — one freeze for the whole foundational curriculum

Design law: **append-only over v3; v3 (43 dims) remains a strict prefix forever.** One fixed-width command block serves every foundational stage — no per-stage contract bumps, no second freeze.

| Slice | Name | Dims | Encoding / frozen scaling |
|---|---|---|---|
| `[0:14]` … `[40:43]` | **v3 unchanged** | 43 | bit-for-bit as documented (`docs/RL_ENVIRONMENT.md`); `[40:43]` stays the world-frame velocity command v\* |
| `[43:58]` | `limb_pose_target` | 15 | per-joint normalized pose target, **same encoding as the action mapping**: `n ∈ [−1,1]` ↔ world target `lo + (n+1)/2·(hi−lo)`; `0` = neutral range midpoint (0 rad every hinge). Order = action joint order (§2). |
| `[58:60]` | `com_shift` | 2 | commanded COM shift from the **measured standing-anchor** (−0.0793, 0.089) m: x `n·1` ↔ ±0.070 m; y ↔ ±0.040 m. Anchored to the measured foot hull (x edges −0.2143/+0.0557, y 0.0179–0.1979): every commanded bound keeps ≥ 3.1 cm static margin (`M7_AUDIT_PROBE.json`). |
| `[60:63]` | `waypoint` | 3 | (ux, uy, d̂): unit direction to the target **block in the agent torso frame** (heading-invariant) + `d̂ = distance / d_ref`; `d_ref` measured from the arena bounds at M7-0 probe time (P3). |
| `[63:64]` | `stop_request` | 1 | {0, 1}: "park near the waypoint now" — arms the stop-tolerance behaviour. |
| `[64:68]` | `cmd_mask` | 4 | {0,1} × (limb, balance, waypoint, stop) in that fixed order. **The mask is the stage/task selector in-obs** (no task one-hot — the standing key decision). |

`obs_dim = 68`, dtype float32, contract version 4.

**Mask semantics (the disambiguation rule):** mask bit 0 ⇒ that channel group is INACTIVE: the env ignores it (no reward consumes it) and the policy reads zeros. Mask bit 1 ⇒ active: values are the actual command; a 0 inside an ACTIVE limb group is a real command ("go to neutral"), not "no instruction". Each stage's phase data fixes which mask bits are on (per-stage table below) — the curriculum stays the single scheduler, so identical calls give identical episodes (B2 determinism preserved).

**Stage → interpretation table (the foundational ladder):**

| Stage | limb | balance | waypoint | stop | velocity `[40:43]` | reward additions (via phase `reward_coeffs` + new keyed components) |
|---|---|---|---|---|---|---|
| S1 limb-control | 1 | 0 | 0 | 0 | zero | `limb_tracking` on; progress structurally 0 (v\*=0) |
| S2 balance | 0→1 (coupling sub-gate turns limb back on) | 1 | 0 | 0 | zero | `com_tracking` on |
| S3 standing | 0 | 0 | 0 | 0 | zero | v3 standing behaviour; gates only |
| S4 disturbance | 0 | 0 | 0 | 0 | zero | env-side impulse hook (§9), no reward change |
| S5 gait | 0 | 0 | 0 | 0 | **ladder speeds / backwards / ramps** | v3 terms; waist genuinely commanded (§2) |
| S6 waypoint | 0 | 0 | 1 | 1 | zero (subordinated) | `approach_progress` + `stop_tolerance`; `velocity_tracking` coefficient set 0 per-phase (the mechanism already supports per-phase coefficients) |

## 2. Action contract v4 — 15 joints, frozen once

**Recommendation (decision D1): freeze `action_dim = 15` now**, = the 14 legacy joints **in unchanged order** + `waist_yaw` appended at index 14 (append-only action rule). Waist becomes a genuinely commanded joint from M7-0 onward.

Why now rather than a bump at M7-4 (the alternative):

- Jamie's requirement is the FINAL contract for the foundational curriculum — S5/S6 need the waist for turning/orienting, so 15 *is* the final contract; freezing it avoids a mid-learning contract change (which would force a surgery exercise on a live lineage exactly when training evidence is most at risk).
- S1–S3 behaviour is unchanged in kind: waist commands sit at neutral (0), the physics path is the same PD-through-`set_joint_target` the posture PD uses today — the standing equilibrium carries over (gate N2 below measures the delta instead of assuming it away).
- Every v4 lineage shares one contract ⇒ every later fork is a continue-fork (cheap); v3→v4 remains the SINGLE cross-contract transfer pair, exercised once, well tested.

Honest costs recorded: the effort measurement set grows from 14 to 15 dofs (standing delta ≈ 0 — waist velocity ≈ 0 at equilibrium — but the neutral gate N2 MEASURES the bound, no hand-wave); smoothness now spans 15 dims (near-constant in S1–S3); waist gains/caps are frozen contract values and must be **measured first** (probe P1).

**Frozen waist parameters (pending P1):** initial gains = the current env posture PD's (kp 50.0, kd 5.0 — `env.py:171`); P1 measures hold quality/saturation at these gains and sweep-informed caps; final numbers recorded in the freeze annex before any training. Passive hinges remain exactly `left_foot_yaw`/`right_foot_yaw` (env_posture).

## 3. Which joints each limb-control task can command (S1 task grammar)

Limb groups over the 15 action joints:

| Group | Joints (action indices) |
|---|---|
| torso | spine_L2 [0] |
| left_arm | left_shoulder [1], left_elbow [2] |
| right_arm | right_shoulder [3], right_elbow [4] |
| head | head_pitch [5], head_yaw [6], head_roll [7] |
| left_leg | left_hip [8], left_knee [9], left_ankle [10] |
| right_leg | right_hip [11], right_knee [12], right_ankle [13] |
| waist | waist_yaw [14] — S1 commands it to HOLD neutral (steadiness is a posture gate); real turning arrives at S5 |

Per-task active subset = the group under test; every other joint's limb channel shows its **neutral rest target** (0), so the whole 15-vector is always a complete, self-consistent desired pose. The env's limb task defines which subset the `limb_tracking` reward consumes; the gate requires the INACTIVE joints to stay within their rest band too (no thrashing elsewhere while one limb obeys).

S1 task forms (each commanded pose held T_hold, repeated R times, seeded spawns):
- arm tasks: shoulder flexion targets × elbow targets (incl. hold combinations);
- leg tasks: hip/knee/ankle excursions **within the both-feet-contact upright discipline** — full single-leg weight transfer is S2 territory by stage separation (Jamie's rule: prove limb control first; stance transfer needs balance machinery that does not exist until S2 passes);
- head/torso tasks: pitch/yaw/roll targets incl. holds;
- waist: neutral-hold under S1 (gate, not a commanded-motion task until S5).

**Rig honesty (recorded):** all these asks live within the current single-axis sagittal hinges ±π/2. "Raise arm sideways" (abduction), wrist DOFs etc. are **not** commandable — a rig extension is a separate measured decision, flagged in the M7 audit §6, interacting with B7 anthropometry.

## 4. How individual limb capability is measured (S1 protocol)

Per limb group, per commanded pose, over R repeats × seeded spawn set, measured **from engine state** (never from policy internals):

1. `reach_error_norm` — normalized joint space error at hold-end and hold-mean, per joint and per group (mean / p95 / max).
2. `hold_variance` — per-joint error variance across the hold window (steady tracking, not oscillation).
3. `repeatability` — per-joint error std ACROSS the R repeats of the same command.
4. **Discipline terms (the "not by flopping the body" exclusion):** chest-aware up_z ≥ band; torso up_z ≥ band; COM inside the support hull (measured per window); falls = 0; faulted windows = 0; inactive-group deviation ≤ band.
5. `effort_mean_w` ≤ band (no violent compliance), action clipping rate logged and bounded.

**Calibration rows (measured, not invented):** probe P2 measures, per joint, the PD-achievable steady error and variance for the same hold targets through the honest `set_joint_target` path with the frozen gains (pure actuator mechanics, no policy). Gate bands = P2 reference + stated margin + the untrained-policy baseline row. No band is asserted before its calibration row exists.

## 5. How balance and standing are objectively assessed (S2/S3)

S2 (balance/weight control), all from engine measurements:
- `com_tracking_error` — COM offset vs commanded `com_shift`, mean/p95 over the hold (COM from `body_mass ⊗ xipos` — the probe-measured channel);
- single-leg stance: **stance fraction** from the observed per-foot support forces `[38:40]` vs the commanded stance; hold duration; falls = 0;
- **coupling sub-gate (the S1→S2 bridge):** S1 limb sequences issued WHILE balance terms hold — limb targets active, balance commands active simultaneously; posture/upright bands maintained (this is the measured proof that limb skills exist before they are combined — Jamie's separation made testable);
- posture bands verbatim from the B4 contract (head |rel pitch| ≤ 0.35/0.60 rad; head world-up ≥ 0.9/0.8; chest |tilt| ≤ 0.35/0.60 rad).

S3 (coordinated standing):
- stand from the existing jittered spawn box; hold the full 8-s horizon (400 decisions); drift ≤ (that lineage's own zero-action row + margin) — the standing-drift precedent rows exist (b4_posture1 +0.176 m / b4_head1 +0.4455 m);
- the 4,800-window standing equilibrium row as the reference (2.4972/window, 0 falls, chest_upz ≡ 1.0);
- COM excursions stay within the static-margin envelope; posture bands held; effort ≤ zero-action band + margin; 0 falls, 0 faulted windows.

## 6. Gate form (what "pass" means)

Every M7 gate report states, per stage:
1. the metric set above, computed over the fixed protocol (held-out seed base 97,010,000, disjoint, deterministic eval plane + the 40-ep stochastic replay rows for trajectory tables);
2. its **calibration rows** (P2 reference, untrained baseline, zero-action control, scripted gait fixtures where applicable) — numbers set as measured-reference + stated margin, never a priori;
3. the full-suite regression (`run_tests.sh`, ×2, exit 0) — necessary, never sufficient;
4. the **three-operation tag** of the producing policy (§8) and the lineage it came from;
5. the visual-inspection annex (§7) bound to the numeric rows.

A gate passes only with: all numeric bands met × the regression green × the visual annex reviewed × no unexplained discrepancy between screenshots and numbers.

## 7. How visual inspection supports the numerical evidence

Reusing the proven playback machinery (cda7aba) — read-only, deterministic:
- every gated eval plane gets a playback session (`playground.rl_live.playback`); screenshots via headless Chrome/CDP at fixed episode fractions (0/25/50/75/100 %) from two fixed cameras (side sagittal, front) — the exact method that verified the walking figure and the geometry fixes;
- screenshot artifacts filed under `reports/m7/<gate>/viewer/` and **annotated to the numeric rows** (frame ↔ decision index ↔ metric row);
- live-training observation stays on the read-only dashboard tab (surfaces last — no new UI in M7-0);
- rule: screenshots support numbers; numbers stand alone; any visible disagreement between them stops the gate.

## 8. The three-operation law (architectural, never interchangeable)

| Operation | What it is | Contract | Optimiser/σ/counters | Required evidence |
|---|---|---|---|---|
| **Resume** | continue the SAME run in the SAME run dir | unchanged | restored in full | restoration test (weights, Adam, RNG, curriculum stage); continuation telemetry |
| **Continue-fork** | new run dir branching from a saved brain | SAME (v4↔v4) | preserved (Adam moments, σ, step/episode counters continue; run-local bookkeeping restarts) | fork-start restoration + **parent SHA unchanged after fork start** + eval-before rows + gate evals under the same protocol |
| **Transfer-fork** | adapt to changed task/contract | may differ (e.g. v3→v4) | ported where shapes match, fresh where new; σ re-initialised to `log_std_init` (decision D5; alternatives measured later, never silently) | fingerprint subset checks + **surgery round-trip proof** (fork-moment outputs on the shared slice identical to the parent, float tolerance) + parent-under-child-task baseline + parent-protocol forgetting eval after training + child-protocol gate evals |

Enforcement: the run launcher records the operation tag in run metadata (and `save_checkpoint` carries it via `extra_meta` — no format bump); every eval/trajectory row names its tag. The machinery REFUSES to attempt surgery without a `transfer_fork` tag and refuses continue-fork when fingerprints differ.

**Transfer surgery spec (v3→v4 pair, the only cross-contract pair):** obs — port the trunk first-layer rows for `[0:43]` onto the new (64×68) weight; **zero-init rows [43:68]** (decision D3) so the fork-moment behaviour is exactly the parent's — the round-trip test asserts this equality. Actor output — port columns `[0:14]`, zero-init column 14 (waist). Critic — same row rule. Hidden widths unchanged ((64,64)) with the meta extension recording widths (additive field, default (64,64) for existing checkpoints).

## 9. M7-0 execution plan (after approval — nothing below has started)

**Pre-implementation probes (read-only, available today through the honest world API):**
- **P1 waist capacity probe** — hold at neutral + commanded-target sweep via `set_joint_target` at candidate gains; record torque, drift, saturation attribution → freezes waist gains/caps;
- **P2 per-joint PD tracking reference** — steady error/variance for representative hold targets on all 15 joints → the S1 calibration rows;
- **P3 arena bounds + target-block inventory** — `world.boundaries()`, positions/sizes of the 20 in-scene blocks → freezes `d_ref` and waypoint normalisation;
- **P4 support-hull re-verification** — extends `M7_AUDIT_PROBE.json` with hull-based margins (the x-gap between feet exists; the COM margin basis is the convex hull, stated explicitly).

**Implementation (one reviewable change set, gated in two commits):**
- **M7-0a — env contract v4:** command block + mask + `set_phase` keys (per-stage table), measured constant annex, contract fingerprint updates, meta extension; tests: append-only layout test, per-stage mask semantics, standing-equivalence neutral gate N1 (v3 fixtures on the 43-slice preserved) and N2 (standing equilibrium delta bounded by the measured waist-epsilon with stated margin over ≥ 4,800 windows), full suite ×2;
- **M7-0b — loading law:** op tags in the run launcher + `extra_meta`, continue-fork, transfer-fork surgery + round-trip tests (both directions of the failure mode: surgery refused on non-transfer tag; round-trip equality proven), full suite ×2.

B3 / b4_head1 / b4_posture1: untouched; they register later (M7-6) as read-only comparison lineages.

## 10. Decisions awaiting Jamie at the M7-0 approval gate

- **D1** action 15 with waist live from M7-0 (recommended; alternative = 14 now + bump at M7-4) — §2;
- **D2** command-bound constants: `com_shift` ±0.070 m x / ±0.040 m y, agent-frame waypoint, frozen `d_ref` from P3 — §1;
- **D3** transfer surgery new-channel init = zeros (exact fork-moment behavioural identity) — §8;
- **D4** S6 first gate uses an ENV-DESIGNATED target block in-observation; agent-side "identify which block" (multiple candidates) is a later, separately gated upgrade — recorded boundary, not smuggled scope;
- **D5** transfer-fork σ re-initialisation (proposed default) — §8;
- **D6** S4 impulse magnitudes: derived at M7-3 from the scripted-controller envelope + B1 caps (not frozen now) — noted.
