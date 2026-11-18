# M7-0 — Contract Freeze Specification (DRAFT v2 — post-probe revision, pending Jamie review/approval)

**Date:** 2026-11-18 · **Status:** DRAFT v2 (supersedes the v1 text pushed at commit `1ffc396`; same review/approval gate, nothing implemented). **No implementation has begun; no env/trainer code has been touched.**

v1 (the published draft) was reviewed against the actual source line-by-line and completed with four measured probes, all read-only, all run 2026-11-18 from scratch `LocoEnv`s through the honest world API:

- **P1** `ai-playground-audit/m7_p1_waist_capacity.py` → `m7_p1_waist_capacity.json` — waist gains/cap/turn envelope;
- **P2** `ai-playground-audit/m7_p2_pd_reference.py` → `m7_p2_pd_reference.json` — per-joint PD tracking reference (the S1 calibration rows);
- **P3** `ai-playground-audit/m7_p3_arena_blocks.py` → `m7_p3_arena_blocks.json` — arena bounds, block inventory, waypoint normalisation;
- **P4** `ai-playground-audit/m7_p4_support_hull.py` → `m7_p4_support_hull.json` — hull, settled COM, margins, stance baseline.

Probe outputs are the **confirmed measurements** (§0); the contract slices and gate bands in §1–§8 are **proposed design** anchored to them. Decisions D1–D6 are resolved in §10 with a blocking column. Corrections to v1 are marked **[CORRECTED]** or **[AMENDED]** inline.

## 0. What was verified against the actual build (confirmed measurements)

### 0.1 The v3 contract (source: `rl/env.py`, `rl/rewards.py`, `rl/curriculum.py`, `rl/checkpoint.py`, `rl/eval.py`)

- Obs `float32[43]` exact layout verified (machine-readable sections in `M7_AUDIT_PROBE.json` / `observation_layout()`): `[0:14]` joint pos `(q−lo)/(hi−lo)·2−1`; `[14:28]` joint vel /6.0; `[28:31]` torso up (torso body xmat col 2); `[31:34]` **root slide** velocity /5.0; `[34:35]` up_z; `[35:38]` torso fwd (xmat col 1 — the rig's local +Y chest axis); `[38:40]` per-foot support force /1000 (contact-force path: `mj_contactForce`, |nz|>0.7, self-pairs excluded — `env.py:491-517`); `[40:43]` commanded v\* world /5.0.
- Action 14 → PD via `set_joint_target`; mapping `target = lo + (a+1)/2·(hi−lo)` (`env.py:368`); **nu = 0 in the live model** — the PD torque via `qfrc_applied` is the only force path (`world/physics.py:983-1009`); the asset XML's `<motor>` entries are not the live actuation story.
- All 14 action ranges are symmetric ±π/2 ⇒ **action 0 ⇒ target 0 rad exactly**; the same is true for waist_yaw (±π). This makes mask-neutral and N2-equivalence claims exact rather than approximate.
- Reward: 8 keyed components (`COMPONENT_KEYS`), per-phase coefficient rewrite in `set_phase` (`rewards.py` unknown-key `ValueError`). New keyed components ride by extending `DEFAULTS` with config-default 0.0 — the b4_posture1 bit-identical pattern. `heading_progress_raw` returns 0 at zero v\* (structurally), so S1/S2/S3 progress is 0 by code, not by tuning.
- Curriculum: `Stage` records + `set_phase` non-cumulative rewrite; reserved stages (`disturbance/terrain/navigation`) declared `env_ready=False`, and the sequence **holds** at a stage whose successor is reserved (`curriculum.py:166-170`). `[40:43]`-per-phase `velocity_tracking = 0` is supported (S6 subordination).
- Loader: `rl_checkpoint/1` + SHA-256 tamper check; strict fingerprint diff (obs/action dims, section slices, joint order, mapping + joints + kp) — raising `CheckpointError`. **Loader gap found [CORRECTED]:** the diff does **not** compare `kd` values **or joint ranges** — a range/kd change is silently loadable today. M7-0b extends the diff to ranges + kd (small additive change; part of the loading law, §8).
- Checkpoint meta carries obs/action dims, joint order, PD gains, `log_std` per-dim vector, optimizer, torch RNG, counters; `extra_meta` passes through (`checkpoint.py:133`). Hidden widths are **not** in meta — meta extension (additive, default (64,64)) stands per §8.
- Policy: `TanhGaussianActor`, trunk tanh MLP (64,64), per-dim `log_std` vector, `log_std_init = 0.0` ⇒ **σ_init = 1.0** (NOT the 0.1 figure sometimes quoted — 0.1 is the reward tracking kernel σ). Parent lineages end at mean log_std ≈ −0.23…−0.28 (σ ≈ 0.75–0.80), per-dim learned values vary.
- Eval: held-out base 97 010 000, deterministic mean-action, zero-action control, scripted gait benchmark via `set_gait("walk", speed, heading)`, upright-vs-fell displacement split (`eval.py`) — the §6 gate harness as drafted.

### 0.2 The rig (P1/P2 + model inspection)

- Joint inventory: 14 policy hinges (asset order, §2) + 3 posture-PD hinges today — `waist_yaw (50,5)`, `left/right_foot_yaw (8,1)` (`env.py:170-173`). **v1 wording "passive hinges remain left/right_foot_yaw" is the v4 change, now stated as such:** waist moves from `env_posture` to the policy channel; foot_yaw stays conventional.
- **`waist_yaw` is joint index 3** — the agent root body's 4th dof (slides agent_x/y/z = 0-2), axis `(0,0,1)`, range ±π — the rig's only rotational root dof; the torso is an articulated child so the dof carries ~1 kg·m² effective inertia (`world/spec.py:207-223`). Live model: `nu=0`, `njnt=42`, `nq=174`, `nv=152` (20 freejoint blocks + ring + rock).
- **[CORRECTED] model-order enumeration cannot produce the v4 order:** with waist no longer excluded, model joint order puts waist **first** (index 3). M7-0a must enumerate by an explicit ordered NAME list (the frozen 14-name v3 order + `"waist_yaw"`), never by model order. Written into the implementation contract (§2 law).
- B1 torque caps verified live from config (`world/safety.torque_limits`): default 600; spine_L2 60, **waist_yaw 120**, shoulder 160, elbow 220, hip 400, knee 570, ankle 400, head 20, foot_yaw 20; gain ceilings kp 1000 / kd 100 (candidate (50,5) far below ceilings). Saturation is attributed per joint and per source (`neural_policy` vs `env_posture` vs probe tags) — the attribution seam P1/P2 rely on is live.

### 0.3 P1 — waist capacity (measured)

- **Neutral hold (24 s, window row A):** waist |qpos| max **1.6e−11 rad**, |torque| max 3.6e−9 N·m — the standing equilibrium at neutral is torque-free; the v3 standing behaviour is waist-neutral today. N2's expected delta ≈ 0 is now a measured structural expectation, re-measured over ≥4,800 windows at M7-0a.
- **Standing commanded targets (P2 rows, fresh resets):** a friction dead-band — target ±0.35 rad → steady err 0.29; ±0.70 → 0.57; ±1.00 → 0.66 (torque ≤ 29.5 N·m); ±2.00 → 0.14 (the body slips and tracks). P1B's sequential −1.5 rad case saturated the 120 N·m cap (raw demand exceeded cap; attribution logged) and fell. **Consequence (design):** S1's waist task is the neutral-hold gate only; commanded standing yaw tasks are NOT reliable capability demonstrations at (50,5) — real turns are a gait-coupled skill (next).
- **Scripted turn envelope (P1C, `set_gait` path):** straight h=90 max 3.85 N·m; 90°→180° turn transient max 29.3, sustained 22.4; 90°→0° max 37.6; straight-with-180°-twist (h=270) max 47.5 / sustained 34.8. All ≥ **2.5× under the 120 N·m cap**, 0 falls, headings reached. The measured turn envelope bounds S5's waist demand; the S5 turn sub-gate validates against it at M7-4 launch.
- **[AMENDED reading]** v1's "gains = the current posture PD's (50,5)" is verbatim true (`env.py:171` is exactly the waist entry) — the *claim that needs the gate* is not the gains but the CHANNEL MOVE (env_posture → policy), measured by N2.

### 0.4 P2 — per-joint PD reference (the S1 calibration rows; fresh resets, seed 20261117 + 3-seed repeats)

Achievable steady |err| (rad) / sustained |torque| (N·m) at the representative targets (worst over ±0.35 and ±0.70 cases):

| joint | err | torque | note |
|---|---|---|---|
| spine_L2 | 0.061 | 2.3 | |
| shoulders | 0.065 | 1.6 | |
| elbows | 0.118 | 0.65 | |
| head_pitch/yaw/roll | 0.090–0.096 | ≤0.9 | head torque capacity 20 N·m, huge headroom |
| hips | 0.195 | 10.1 | |
| knees | 0.149 | 8.5 | |
| **ankles** | **0.344** | 8.6 | **ground-pinned under load:** qpos stays ≈ 0 — commanded loaded-ankle poses do not track; ankle tasks are swing/COM-shift dependent (see §3) |
| waist_yaw | 0.29 (±0.35) → 0.66 (±1.0) | ≤29.5 | friction dead-band (P1B/P2) |

All 15×6 hold cases completed upright: **0 falls, chest_up_z ≥ 0.949 (≈1.0 except spine cases)**, torso up ≥ 0.95, per-window error variance 0.002–0.08 rad², cross-seed error spread identical to 4 decimals (jitter-robust attractors, ±0.15 m spawn box). **S1 consequence:** leg tasks are holdable at ±0.35/±0.70 (hip/knee) inside the two-feet discipline; ankle holds under double-support are pinned — the ankle target channel is exercised meaningfully only with COM moved toward that foot (S2 pairing) or in later stage variants; the gate bands are P2 rows + stated margin.

### 0.5 P3 — arena and targets (measured)

- `world.boundaries()`: start (0,0,0) at island centre; walkable ±12.0 m both axes; corner distance 16.97 m; ground z 0; boundary walls are agent-collidable (`world/spec.py:306-307`).
- Block inventory: 20 cylinders (`block_01…20`, geom size r=0.25, half-h=0.25 → 0.5 m diameter × 0.5 m), ring (16 capsule segments), rock (sphere r=0.3) — all parked 90–115 m off-island by default; `setup_place` repositions any of them (verified in `M7_AUDIT_PROBE.json`). **Nav targets are placed, never defaulted.**
- **`d_ref` frozen [AMENDED — no longer "at M7-0 probe time"]:** `d_ref = 4.0 m` = the measured reachable approach distance of the standard 400-window horizon at the sustained 0.5 m/s command (400 × 0.02 s × 0.5 = 4.0). Target sampling (S6): `setup_place`d within **[0.5, 3.5] m** of the episode spawn, inside the measured ±12 m island; `d̂ = dist/4.0 ≤ 0.875`.

### 0.6 P4 — support hull, COM anchor, stance baseline (measured, settled equilibrium)

- Foot hull (seed 20261117): x [−0.2143, +0.0557], y [0.0179, 0.1979] — the two feet share their y-span exactly, so **the double-support convex hull IS the bounding rectangle** (x-gap −0.1243…−0.0343 lies inside the hull; stated explicitly). Single-support hulls are the individual foot rectangles; per-seed hull translations exactly track spawn jitter.
- **Full-body COM at the settled equilibrium: (−0.07926, 0.08871, 0.89944) m** (47.05 kg incl. the 10 kg root column) — the root column's inertial point coincides in x,y with the 37.05 kg subtree COM, so **v1's anchor (−0.0793, 0.089) survives on the full body**; COM z (0.899) is fixed across seeds and uncommanded.
- `com_shift` bounds ±0.070/±0.040 from the full-body anchor — commanded-bound corners stay inside the double-support hull with margins **x ±6.50 cm, y 3.08 / 6.92 cm** (min 3.08 cm; v1 said ≥3.1 — restate ≥3.0 measured).
- Single-support reach: minimum |com_shift| to move the COM over either foot = **0.045 m** (within the ±0.070 bound); at the same-side command extreme the COM stays inside the stance foot with **2.5 cm margin**; the opposite-side pairing (e.g. stance left + full-right shift) is geometrically outside the stance hull — the **S2 command-pairing rule** is therefore required, not optional (§5).
- Stance-fraction calibration row: left/right share **0.500/0.500** at equilibrium (`[38:40]` forces) — the S2 baseline. Waist qpos after 4 s settle: 0.0 exactly, all seeds.

## 1. Observation contract v4 (`rl_env/4`) — confirmed layout, one freeze for the whole foundational curriculum

Append-only over v3; v3 (43 dims) remains a strict prefix forever. `obs_dim = 68`, dtype float32. Slice table unchanged from v1 and verified:

| Slice | Name | Dims | Encoding / frozen scaling |
|---|---|---|---|
| `[0:43]` | **v3 unchanged** | 43 | bit-for-bit as verified in §0.1 |
| `[43:58]` | `limb_pose_target` | 15 | per-joint normalized target, the action mapping constants: `n ∈ [−1,1] ↔ lo + (n+1)/2·(hi−lo)`; 0 = neutral (0 rad every joint — exact for all 15, symmetric ranges). Order = §2 joint order. |
| `[58:60]` | `com_shift` | 2 | commanded COM shift (x, y) from the **settled full-body COM anchor (−0.0793, 0.0887) m**, ±0.070 m x / ±0.040 m y (margins §0.6). COM z not commanded. |
| `[60:63]` | `waypoint` | 3 | (ux, uy, d̂): unit direction in the **yaw-only heading frame** = torso xmat col-1 horizontal projection normalized (fwd_h) and its left-hand perpendicular (left_h) — robust to chest pitch/roll, heading-invariant; `d̂ = dist / 4.0` (P3). ux,uy ∈ [−1,1] unitless. |
| `[63:64]` | `stop_request` | 1 | {0,1}: park near the waypoint (stop-tolerance band ≥ block radius 0.25 m, band set at M7-5 from measured stop rows). |
| `[64:68]` | `cmd_mask` | 4 | {0,1} × (limb, balance, waypoint, stop), fixed order. No task one-hot (standing decision kept). |

**Mask semantics [AMENDED — the disambiguation rule, now precise]:**
- mask 0 ⇒ the group's command values are zeroed in obs and **no reward consumes the group** (limb_tracking / com_tracking / approach / stop off).
- mask 1 ⇒ live: an active 0-valued limb target is a real "hold neutral" command (indistinguishable from inactive only by the mask bit).
- **The action space is NEVER masked** — all 15 dofs always map through (`env.py:366-371`-style path). v1's "the env ignores it" is hereby pinned to obs/reward semantics only: pinning inactive limb ACTIONS to neutral is impossible without breaking S5/S6 (the legs are limb channels; walking needs them live under limb-mask 0). "Inactive commands do not influence the environment" holds because every non-limb command group's only influence path IS the reward, and inactive limb command VALUES are zeroed in obs; the policy's body authority is never gated. Verified against source; no other interpretation survives S5.
- Per-stage mask bits ride phase data (`set_phase` extension, M7-0a) — the curriculum stays the single scheduler; identical calls give identical episodes.

**Stage → interpretation table (the foundational ladder):**

| Stage | limb | balance | waypoint | stop | velocity `[40:43]` | reward additions |
|---|---|---|---|---|---|---|
| S1 limb-control | 1 | 0 | 0 | 0 | zero | `limb_tracking`; progress structurally 0 |
| S2 balance | 0→1 (coupling sub-gate) | 1 | 0 | 0 | zero | `com_tracking` |
| S3 standing | 0 | 0 | 0 | 0 | zero | v3 standing behaviour; gates only |
| S4 disturbance | 0 | 0 | 0 | 0 | zero (M7-3 redefines; note: the RESERVED `disturbance` phase currently carries v\* [0.5,0,0] (+X) — flagged for redefinition, it has never been trained) | none (env-side hook) |
| S5 gait | 0 | 0 | 0 | 0 | **ladder speeds / backwards / ramps / DIRECTIONS** | v3 terms; waist genuinely commanded |
| S6 waypoint | 0 | 0 | 1 | 1 | zero (subordinated) | `approach_progress` + `stop_tolerance`; `velocity_tracking` coefficient 0 per phase (supported) |

**S5 turning rides v\* [RESOLVED — closes v1's turn-command gap]:** v\* is world-frame; the heading is observable via `torso_fwd [35:38]`. Commanding v\* in a direction other than +Y is a legitimate turn command: the honest way to earn progress + tracking is to yaw the chest axis onto the commanded direction with the now-live waist, exactly the mechanism the scripted controller demonstrates in P1C. No new command group is needed; no waypoint-group activation at S5 (skill-before-combine preserved: turning is proved at S5, navigation combining turn+approach only at S6). Evidence of the turn = measured angle between torso_fwd and the v\* heading → 0, upright, under the P1C envelope.

**Missing-information resolutions [the freeze's answers]:**
- **Stance is NOT a command.** No commanded-stance channel exists in v4 and none is added: single-support is commanded through `com_shift` toward the stance foot (reach 0.045 m ≤ 0.070 ✓ P4) — `com_shift` is the single balance command; stance fraction is a GATE metric (measured `[38:40]`, baseline 0.5/0.5).
- **Orientation-at-stop is a gate metric**, not a command channel: chest axis vs target direction measured at stop (S6), possible because the waypoint frame + live waist make orientation reachable.
- **Command schedules** (S1 pose sequences, S2 shift sequences, S6 placements) are stage-defined, seeded, deterministic episode data — the env exposes command setters consumed at reset/stage boundaries; identical calls give identical episodes (B2 determinism, generalized). The S6 target placement rides `setup_place` (verified seam) inside `reset(seed)`.
- COM z intentionally not commanded; head "look-at" intentionally not a v4 command (head targets ride limb commands).

## 2. Action contract v4 — 15 joints, frozen once (decision D1: ACCEPT)

**Freeze `action_dim = 15` = the 14 legacy joints in unchanged order + `waist_yaw` at index 14.**

- **Implementation law [CORRECTED]:** the joint order is the explicit frozen NAME list `[spine_L2, left_shoulder, left_elbow, right_shoulder, right_elbow, head_pitch, head_yaw, head_roll, left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle, waist_yaw]` — the model-order enumeration must NOT be used for v4 (waist is model joint index 3; naive enumeration puts it first and breaks the append-only obs prefix). The name list goes into the contract fingerprint (the loader diff already checks joint order exactly).
- Waist becomes a genuinely commanded joint from M7-0 onward: measured basis P1/P2 (§0.3). Passive hinges remain exactly `left_foot_yaw`/`right_foot_yaw` (env_posture), as v1 said.
- Gains frozen **(kp 50.0, kd 5.0)** from `env.py:171` (the current posture values), measured at P1/P2: neutral hold torque-free; turn envelope ≥2.5× under the B1 cap 120 N·m. Cap unchanged (no B1 config change needed).
- N2 (M7-0a) measures the effort-set growth 14→15 (`_dofadr` grows from `joint_names`, `env.py:175`) and the standing-equivalence delta over ≥4,800 windows — the measured bound replaces the v1 "≈ 0" claim; P1A's zero-torque neutral row is the structural prior.

## 3. Which joints each limb task can command (S1 task grammar)

Groups unchanged from v1 (torso; left/right arm; head; left/right leg; waist = neutral-hold). Inactive groups show neutral targets; every inactive joint must respect its rest band (gate term).

Measured refinements over v1 (from P2): ankle tasks, **ground-pinned under load** — S1's ankle targets are restricted to the small-amplitude holdable set inside double-support (or exercised under S2's COM-shifted pairing); hip/knee ±0.35/±0.70 holds are holdable (P2 6/6). Waist: neutral-hold gate only (P1B/P2 dead-band). Rig honesty unchanged: sagittal single-axis ±π/2; no abduction/wrist; extension is a separate measured decision (M7 audit §6, B7 interaction).

## 4. How limb capability is measured (S1 protocol)

The v1 metric set stands (reach/hold-variance/repeatability/discipline/effort). **The calibration rows now exist (P2)**: per joint, the PD-achievable steady error/variance/torque through the exact action path (14) and the posture path (waist). Gate bands = P2 row + stated margin + the untrained-policy baseline row, set at M7-1 launch — never a priori. Band derivation example (stated form, not yet a number): arm hold band = P2 err_max (0.065–0.118) × margin (≥1.5) rounded; ankle band uses the ground-pinned reality (bands derive from the achievable set, not the commanded ideal).

## 5. How balance and standing are objectively assessed (S2/S3) [AMENDED with the pairing rule]

- `com_tracking` vs commanded `com_shift` (mean/p95 over hold; COM from `body_mass ⊗ xipos` full-body, engine-measured).
- **Command-pairing rule (new, required):** during single-support sub-tasks the S2 sampler commands `com_shift` only within the stance-consistent subset (min 0.045 m toward the stance foot; the same-side ±0.070 extreme stays inside with 2.5 cm measured margin); the gate checks the COM against the **current measured support hull** (single foot under single support, not the double-support hull). A commanded shift and a stance that together put the COM outside the support hull is a sampler defect, not a policy failure — the gate separates these.
- Stance fraction from `[38:40]` (baseline 0.5/0.5 measured); hold duration; falls = 0.
- Coupling sub-gate (S1→S2 bridge): limb mask + balance mask live simultaneously; posture/upright bands maintained.
- Posture bands verbatim from the B4 contract (head |rel pitch| ≤ 0.35/0.60; head world-up ≥ 0.9/0.8; chest |tilt| ≤ 0.35/0.60).
- S3: the v1 set stands — 8-s horizon hold, drift ≤ own zero-action row + margin (precedent rows: b4_posture1 +0.176 / b4_head1 +0.4455), 4,800-window standing equilibrium reference (2.4972/window, 0 falls, chest_upz ≡ 1.0 — re-measured under v4 by N2), COM excursions inside the static-margin envelope, 0 falls / 0 faulted.
- **"COM position ≠ active balance" is enforced structurally:** S2/S3 pass on COM TRACKING under commanded shifts and stance transitions (dynamic demands), never on static COM position; the true balance proof is S4 (disturbance recovery, D6) — S4 is not bypassable for "balance capability" claims.

## 6. Gate form

Unchanged from v1 (metric set over the fixed held-out protocol 97,010,000; calibration rows; full-suite regression ×2 exit 0; operation tag + lineage; visual annex). Calibration rows now concrete: P2 (limb), P4 (balance/stance), P1 (waist/S5 envelope), plus zero-action and gait fixtures.

## 7. Visual inspection

Unchanged from v1 (playback/CDP two-camera annex bound to rows; surfaces last).

## 8. The three-operation law

Table as v1, with two amendments:

- **[AMENDED] D5 — σ policy:** transfer-fork **preserves the parent's per-dim `log_std` on the shared action slice [0:14]** and appends `log_std[14] = log_std_init` (0.0 ⇒ σ 1.0). Verified basis: `log_std` is a per-dim learned vector (`ppo.py:140`); re-initialising all 15 to σ 1.0 would silently RAISE σ on every proven skill (parents end σ ≈ 0.75–0.80 mean, learned per-dim) — a larger behavioural jump than the zero-init μ surgery the design is careful to avoid. With preserved σ, the fork-moment **action DISTRIBUTION on the shared slice is identical to the parent's** — the round-trip test asserts this distribution-level identity (deterministic outputs AND log_std equality), which is strictly stronger than the v1 mean-only statement.
- **[New] Loader diff extension:** `_fingerprint_diff` gains joint ranges + kd (verified gap §0.1) — part of M7-0b, additive, no format bump.
- Everything else stands: continue-fork refuses on fingerprint differences; surgery requires the transfer_fork op tag (launcher-recorded into run metadata + `extra_meta`; verified passthrough `checkpoint.py:133`); surgery specifics (trunk row port + zero-init [43:68], μ-head column port + zero-init col 14, critic same rows, hidden-width meta additive) verified against the actor/critic architecture.

## 9. M7-0 execution plan — after approval

Pre-implementation probes P1–P4 **complete** (§0, this session). Remaining plan:

- **M7-0a env contract v4** — name-ordered 15-joint enumeration (§2 law); command block + mask + set_phase keys; new keyed reward components (`limb_tracking`, `com_tracking`, `approach_progress`, `stop_tolerance`) with config-default 0.0 (bit-identical fixtures); loader-fingerprint constants append to fixtures as measured v4 constants (com anchor, d_ref, sampling bounds); tests: append-only layout, per-stage mask semantics, **N1** (v3 fixtures on the 43-slice bit-preserved), **N2** (standing equivalence ≥4,800 windows with waist live — expected ≈0 from P1A), full suite ×2.
- **M7-0b loading law** — op tags; continue-fork; transfer surgery + distribution-level round-trip tests; fingerprint diff extension; full suite ×2.
- B3 / b4_head1 / b4_posture1 untouched; register later (M7-6) read-only.

## 10. Decision table

| | Decision | Status | Evidence | Required tests | Blocks implementation? |
|---|---|---|---|---|---|
| **D1** | action 15, waist live from M7-0, gains (50,5), cap 120 unchanged | **ACCEPTED** | P1A (neutral ≈ zero torque), P1C (turn envelope ≥2.5× headroom, 0 falls), P2 (dead-band ⇒ S1 waist = neutral hold), `nu=0` path verified, B1 cap present | N1 + N2; S5 turn-amplitude validation vs P1C envelope at M7-4 | **Yes (freeze)** |
| **D2** | `com_shift` ±0.070 x / ±0.040 y + waypoint frame + `d_ref` 4.0 + sampling [0.5, 3.5] m + stop band basis (block r 0.25) | **AMENDED → ACCEPTED** (adds the S2 stance-pairing rule; v1's "≥3.1 cm" restated ≥3.0 measured 3.08 cm; d_ref and sampling frozen from P3 instead of deferred) | P4 margins (x 6.50, y 3.08/6.92, single-support 2.5 cm, reach 0.045); P3 island half 12.0, corner 16.97, horizon-reach 4.0 | N2; S2 protocol tests; nav-sampler determinism test (M7-5) | **Yes (freeze)** |
| **D3** | surgery new channels zero-init | **ACCEPTED** | verified actor/critic shapes (TanhGaussianActor (64,64), trunk/μ-head/log_std param lists) | round-trip test (now distribution-level incl. log_std) | **Yes (freeze)** |
| **D4** | env-designated target for the first nav gate | **ACCEPTED** | P3 (`setup_place` proven; blocks parked off-island ⇒ placement mandatory) | nav-stage tests at M7-5 | No (M7-5 scope) |
| **D5** | σ transfer policy | **AMENDED → preserved-σ slice + appended init** | per-dim `log_std` vector; `log_std_init` 0.0 (σ 1.0) vs parent learned σ ≈ 0.75–0.80 mean; re-init would silently move every skill's noise level | distribution round-trip (deterministic outputs + log_std equality on the shared slice) | **Yes (freeze)** |
| **D6** | S4 impulse magnitudes derived at M7-3 from the scripted envelope + B1 caps | **ACCEPTED (defer as drafted)** | disturbance seams verified (`impulse_object`, `set_object_velocity`, `get_state/set_state`); caps table verified (§0.2); the reserved phase's current v\* [0.5,0,0] flagged for M7-3 redefinition | disturbance-hook tests at M7-3 | No (M7-3) |

**Unresolved measurements (explicit, none a freeze blocker):** (1) N2's standing-equivalence bound value (≥4,800 windows, M7-0a); (2) gate band numbers derived from P2/P4 rows with stated margins at each stage launch; (3) S5 turn-command amplitude set vs the P1C envelope (M7-4 launch); (4) stop-tolerance band from measured rows (M7-5 launch); (5) disturbance magnitude classes (M7-3).
