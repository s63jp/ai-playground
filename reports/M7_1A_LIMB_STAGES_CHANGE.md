# M7-1a — Individual limb-control stages (S1): implementation spec (written BEFORE code)

**Date:** 2026-11-18 · **Basis:** Jamie's M7-1 opening directive (accept M7-0b; M7-1 tightly scoped to individual limb control, head/neck AND trunk posture included, **no walking curriculum**); the approved M7-0 freeze v2 (S1 grammar §3, capability protocol §4, P2 calibration rows); measured evidence `ai-playground-audit/m7_p2_pd_reference.json`. All M7-0a seams verified against source before this spec was written: `LocoEnv.set_commands` (loud validation, v3-refusal), `LocoEnv.set_phase` (`reward_coeffs` partial rewrite + `commands` key, non-cumulative), `_command_block_obs` (masked groups emit zeros), `rewards.limb_tracking_penalty` (−w·mean over the ACTIVE group's joints, mask-gated), `Curriculum(stages=…)` (custom ladders accepted; DEFAULT_STAGES untouched).

## 1. Scope contract (this stage does, and does not)

- **DOES:** a separate M7 limb-control ladder `M7_S1_STAGES` for the M7 lineages only — standing as the base, then ONE stage per individual limb group (left arm, right arm, head/neck, torso/Trunk (spine_L2), left leg, right leg), each training that group's tracked holds under a measured schedule while the body stands. Per Jamie's scope: head and trunk are IN; the ladder's eval carries the B4 posture bands throughout.
- **DOES NOT:** walking curriculum (no controlled/sustained/turn stages — the M7 limb ladder's `target_velocity` is zero **explicitly** in every phase, never the stale `[0.5,0,0]` default); no waypoint/balance-command/stop stages (M7-2/M7-3); no transfer surgery use (M7-5); no .aipg/registry (M7-6); no change to `DEFAULT_STAGES` (the B2/B3/B4 fixtures keep their indices AND their behavior bit-identically — the ladder is a NEW tuple consumed by explicit run-launch selection).

## 2. The ladder `M7_S1_STAGES` (indices, phases)

| # | id | limb_group | Commands (per-episode schedule) | rationale (measured) |
|---|----|-----------|----------------------------------|----------------------|
| 0 | `standing` | — (reused, the EXISTING Stage object) | none | base: stand-first + the N2 4,800-window standing reference row applies unchanged |
| 1 | `m7_s1_left_arm` | left_arm | hold targets for shoulder+elbow | P2: err ≤ 0.065/0.118 rad, torque ≤ 1.56 N·m |
| 2 | `m7_s1_right_arm` | right_arm | same, mirrored | P2 identical rows |
| 3 | `m7_s1_head` | head | hold targets for pitch/yaw/roll | P2: err ≤ 0.096 rad, torque ≤ 0.88 N·m (20 N·m cap) |
| 4 | `m7_s1_torso` | torso | hold targets for spine_L2 | P2: err ≤ 0.061 rad, torque ≤ 2.27 N·m; the trunk-posture ask |
| 5 | `m7_s1_left_leg` | left_leg | hold targets hip+ (knee); **ankle stays neutral** | P2: err 0.194/0.149 rad; 8.5–10.1 N·m, largest loads |
| 6 | `m7_s1_right_leg` | right_leg | same, mirrored | |
| 7 | `m7_s1_all` | all (15) | every joint commanded; **the waist is pinned at neutral 0 in every window — the P1B neutral-hold gate, its tracking error measured as a gate, not trained** | the ladder's terminal stage (runs forever), the S1 capability summary row |

Rationale for the order: ascending measured physical load (torque headroom), head/neck first per Jamie's inclusion note. **The waist is NEVER commanded in S1** (P1B/P2 dead-band ⇒ neutral-hold only): every schedule leaves `waist_yaw` at 0 rad normalized 0.0, and the stage-7 "all" group still excludes the waist (targets zero ⇒ neutral). Duration: `hold_windows` segments × schedule length ≥ one episode of 400 windows; the terminal stage's `duration_decisions` runs forever (the sustained-stage convention).

Phase keys per limb stage: `{"gravity_scale": 1.0, "upright_assist": 0.6, "max_steps": 400, "target_velocity": [0.0, 0.0, 0.0], "reward_coeffs": {"limb_tracking": <w after the probe>}, "commands": {"mask": (1,0,0,0), "limb_group": <g>}, "command_schedule": {...spec §3...}}`. `upright_assist` stays at the standing value (0.6) at launch — the B4-era standing row is the reference baseline; any drop is a measured, separately-recorded decision, never tuning mid-stage.

## 3. The command schedule (the ONLY new env seam)

`command_schedule` (phase key, non-cumulative: absent ⇒ no schedule, inactive commands, exactly like the M7-0a `commands` rule):

```
{"group": <LIMB_GROUPS name>,
 "targets_rad": [per-joint target lists — the measured P2 holdable sets, VERBATIM:
     ±0.35, ±0.70 rad for every non-waist joint (waist always excluded)],
 "hold_windows": 120,        # one hold segment
 "transition_windows": 30}   # ramp between holds (the P2 50-window hold + settle 15 pattern, scaled to episodes)
```

Semantics: at `reset(seed)` the env deterministically builds the episode's hold sequence from the spec (seeded numpy RNG; identical seeds ⇒ identical episodes — the B2 determinism contract generalized); inactive joints sit at neutral (0 rad). At each segment boundary the env itself calls `set_commands(limb_targets=…)` (the engine owns the schedule so the trainer stays untouched); `transition_windows` linearly interpolate to the next hold (the command block carries the interpolated target in the same window the reward tracks it). Loud validation on `set_phase('command_schedule')`: unknown group refused; per-joint targets must be inside the joint's `range_rad` AND inside the measured P2 holdable set for that joint ({0, ±0.35, ±0.70} rad; the waist never accept a non-zero target; ankle targets REFUSED for leg stages (P2: ground-pinned — the measured 0.344 rad tracking error is a COM-shift (M7-2) finding, not an S1 task); `hold_windows`/`transition_windows` positive with `hold+transition ≤ max_steps`).

Per-episode determinism detail: the segment ORDER is seeded from the reset seed (a fixed rotation by default — `targets_rad` in listed order cycling; a shuffled variant rides a `shuffle: true` spec flag with the seed) — recorded so replay/eval can regenerate identical commands.

## 4. Reward for S1 (measured coefficient; the probe fixes w)

`limb_tracking` = −w · mean over the ACTIVE group's joints of (measured − commanded)² in normalized joint units (M7-0a, unchanged code). The base components (progress/velocity/effort/alive/upright chest-aware/effort/smoothness/stability/fall) keep their standing-stage values. The coefficient w is NOT picked a priori: it comes from the **untrained-baseline probe** (audit script, read-only, no training): a fresh policy AND the zero-action control replay the same schedules per group; the probe records the per-group window magnitudes (bad-tracking err² vs P2-reference err²) and w is fixed so the tracking gradient between the untrained row and the P2 row is ≥ 0.05/window with the standing-equilibrium total kept intact. The probe output is durable evidence under `reports/`.

## 5. Capability measurement (the S1 gate, per freeze §4 + measured bands)

At each limb stage's checkpoint: matched held-out deterministic eval (seed base 97,010,000, 12 episodes, same seeds as B3/B4 rows) per group with:
1. **Per-joint tracking band**: steady abs-err ≤ the P2 measured row × margin (≥2: e.g. left_arm ≤ 0.13/0.24 rad; head ≤ 0.19; torso ≤ 0.13; legs ≤ 0.29/0.39; waist stays inside the neutral dead-band ±0.29 measured). Bands re-asserted per stage from the P2 row + margin at launch, never a priori.
2. **Inactive rest bands**: every non-group joint stays within its measured zero-action variance band (the probe's row).
3. **Posture bands verbatim B4**: head |rel pitch| ≤ 0.35/0.60, head world-up ≥ 0.9/0.8, chest |tilt| ≤ 0.35/0.60 — measured over whole episodes, stochastic-as-trained.
4. **0 falls / 0 faulted windows** in the eval plane; σ as trained; the standing-drift row (b4_posture1 +0.176 m the reference band family).
5. Visual annex: playback CDP screenshots bound to the numeric rows (screenshots support, never substitute).

## 6. Implementation plan (files + tests)

- `playground/rl/curriculum.py`: `M7_S1_STAGES` tuple + the schedule-spec data (the P2-verbatim target sets are CONSTANTS in the module with provenance comments); DEFAULT_STAGES untouched.
- `playground/rl/env.py`: `set_phase` gains validated `command_schedule`; reset() consumes it (seeded per-episode sequence); step() applies segment boundaries + transition interpolation through set_commands.
- `scripts/rl_ppo.py`: `--stages m7_s1` selects the M7 ladder for a run (explicit, loud: the M7-1 lineage records `stages: m7_s1` in config.json/result.json; default stays the B4 ladder).
- Tests (new `tests/test_rl_m7_s1_stages.py`): ladder ids/indices; phase validation (unknown group, ankle-in-leg refused, waist-refused, stale-default guard: every M7 phase carries explicit zero target_velocity); schedule determinism (same seed ⇒ identical hold sequences; different seeds differ); env-side boundary application (command block changes at the declared window, interpolated ramp values observed); limb_tracking gates (active group only; inactive-joint neutral targets in obs); DEFAULT_STAGES fixture identity (standing/controlled/sustained byte-identical reward paths under the M7 module import); ladder-forever terminal convention.
- The untrained-baseline + w probe (audit dir, read-only): per-group magnitude rows → the w fix + gate bands recorded in the stage record before any training.

## 7. Training-run shape (after the M7-1a gate; NOT in this change)

A fresh v4 lineage (`--operation fresh`, op-tag law, own run dir under experiments/rl/m7_s1_run1) on the M7 ladder; per-checkpoint pulls per the standing Jamie protocol (matched seeds, durable JSONs per checkpoint, trajectory table). Continue/transfer forks stay untouched (M7-5's exercise). **The run launches only after Jamie sees the M7-1a gate report** (the B3/B4 precedent: runs are contract-approved, never auto-started).

## 8. What stays open (explicit)

w (§4 probe), the exact per-joint gate-band margins (§5, from the probe), the terminal stage's duration semantics if the S1-all row proves degenerate with the per-group rows. M7-2 (balance/stance pairing), M7-3 (disturbance hook + the stale v\* correction), M7-4 (gait ladder + waist unlock), M7-5 (nav + the first transfer-fork lineage), M7-6 (registry + .aipg) — untouched here.
