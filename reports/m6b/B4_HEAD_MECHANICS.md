# M6-B — Stage B4 (first gate): the head-mechanics correction isolated

**Scope.** Stage B4 is reward + curriculum refinement. Jamie's B4 contract
(2026-11-18, 8 items) separated its first committed layer as a controlled
PHYSICAL correction — the head inertial fix — trained alone under the
unchanged B3 reward + curriculum, so the two failure sources this stage exists
to address are never confounded. This document is the gate record for that
correction (`b4_head1`); the posture-ask reward lineage (`b4_posture1`) is the
next B4 layer with its own gate.

Companion evidence:

- `reports/HEAD_POSTURE_DIAG_20261118.md` — the 8-question root-cause probe
- `reports/B4_HEAD_FIX_20261118.md` — the full experiment log (contract,
  baselines, per-checkpoint pulls, evals, resolutions)
- `reports/B4_POSTURE_CKPT*.json` — durable checkpoint-posture copies
- `ai-playground-audit/` — all probes (`b4_*`, `probe_head_diag*`) with JSONs

## The correction

`assets/humanoid_mujoco.xml` head body gains an explicit `<inertial
pos="0 0 0.13" mass="1.8" diaginertia="0.010 0.011 0.009"/>`. Root cause
(measured, `probe_head_diag_results.json`): the geom-derived head was
6.3706 kg with COM +0.26 m above the three neck hinges — a 16.25 N·m gravity
moment against a kp=8 neck whose full-range capacity is 12.57 N·m. The
upright equilibrium was mathematically unstable; the head pinned at its
−90° mechanical limit under neutral targets. The upright reward measured only
the pelvis up-vector (the slide-root pelvis cannot pitch ⇒ `upright_fraction ≡
1.000` always) and the stability term was dead — the collapse mode was
invisible to the objective. The fix decouples mass from visual geometry:
compiled 1.8 kg @ +0.13 m ⇒ 2.2955 N·m vs 12.5664 N·m PD capacity (5×
margin); B1's 20 N·m head cap is untroubled; obs/action contracts, all other
bodies, gains and geoms unchanged; obs layout still 43-dim (`rl_env/3`).

Old checkpoints load unchanged (checkpoint contract fingerprints cover the
env/reward/curriculum code, not the asset). **Cross-E5 lineage note: B3
numbers on the old rig are historical evidence, not comparable physics.**

## Run record — `experiments/rl/b4_head1/`

- Seed 20261117, 1.2 M steps (b3_run1's exact prefix), ladder standing
  200k → controlled 500k (recovered from b3_run1's progress transition at
  200,704 — the live config default 400k would NOT have matched), checkpoints
  every 200k boundary.
- Paused intentionally at 497,664 (Jamie's 7-point pause contract, ~02:28);
  resumed 2026-11-18 from the SHA-verified `ckpt_000401408` (load line:
  `resumed from …: step 401408` / `curriculum restored to stage 'controlled'`).
- Completed at the first 2,048 boundary ≥ 1.2 M = **1,200,128 steps
  (it. 586, wall 2,727 s post-resume)**, exit 0.
- **`result.json`: faulted_windows_total = 0, rollout_rejections = 0 across
  the entire run.** b3_run1 recorded 133 faulted windows (DOF-9 warning
  family) in 1.36 M on the old rig: the head fix eliminated training-time
  physics failures outright. Standing-stage falls dropped 161 → 83 (−48 %)
  at matched ladders.

## Checkpoint trajectory (every checkpoint pulled and compared)

40-episode stochastic replay probes, seeds 990100+, save-time stage — the
same protocol as all prior rows (scripts `b4_posture_probe.py`, durable
copies `reports/B4_POSTURE_CKPT*.json`):

| ckpt | stage | completion | falls (fwd/bwd) | axis m/ep | head up ≥0.9 | chest ≤0.35 |
|---|---|---|---|---|---|---|
| 200,704 | controlled | 1/40 | 39 (7/33) | −0.21 | 16.5 % | 97.9 % |
| 401,408 | controlled | 18/40 | 22 (40/0) | +1.04 | 52.0 % | 95.7 % |
| 602,112 | controlled | 10/40 | 30 (23/17) | +0.21 | **68.3 %** | 69.5 % |
| 802,816 | sustained | 21/40 | 19 (21/19) | +0.04 | 3.2 % | 2.8 % |
| 1,003,520 | sustained | 27/40 | 13 (28/12) | +0.36 | 3.1 % | 1.9 % |
| final 1,200,128 | sustained | 29/40 | 11 (29/11) | +0.37 | 2.0 % | 1.7 % |

Findings carried into the next layer:

- The head-up regime built during controlled (peaked **68.3 %**) did not
  survive the stage switch: at sustained it collapses to ~3 % and stays
  (chest_end −1.51 rad at the end). Nothing in the reward asks for posture,
  so the faster command re-folds the chest — the diagnosis's mechanism,
  now measured per-checkpoint on the fixed rig.
- Sustained walking recovers monotonically (completion 52.5 → 72.5 %, axis
  +0.04 → +0.37 m/ep, reward 358 → 519) but plateaus far below the
  B3-1.2M-weights mechanism row on the identical rig/protocol (95 %
  complete, +1.86 m/ep): the fast folded-chest gait is what the velocity
  kernel pays for. This is the measured motivation for the posture-ask
  lineage.
- Backward-collapse mode: eliminated by 401,408 (33/39 → 0/40 backward) and
  stays gone (final 11 falls, all forward/mixed, net displacement positive).

## Contract step 8 — matched held-out evals (seeds 97010000+)

| row | axis m/ep | speed m/s | falls | faulted | complete | reward |
|---|---|---|---|---|---|---|
| **B4 ckpt_final sustained, 12 ep** | **+0.6197** | 0.105 | 0 | 0 | **12/12** | 675.2 |
| B3 ckpt 1.204M sustained, 12 ep (old rig) | +0.8949 | 0.480 | 0 | — | 0/12 (faults) | 300.8 |
| B3 ckpt_final 2.0M sustained, 10 ep (old rig) | +2.9426 | 0.463 | 0 | 0 | 10/10 | 781.7 |
| **B4 ckpt_final standing, 12 ep** | — | 0.061 | 0 | 0 | **12/12** (drifts +0.4455 m) | 976.8 |
| B3 1.204M standing (old rig) | — | — | 12/12 falls | — | — | — |

Fixed-rig baselines (fresh +0.0211, zero −0.0001, gait h=0 0.7301, gait h=90
+0.7057). Honest reading: first lineage to complete every eval episode with
zero physics warnings (B3's completion limitation is gone), standing falls
inverted to full survival — but the policy now drifts under a standing
command instead of falling (standing still is un-asked-for), and walks slower
than both B3 policies because upright locomotion earns poor tracking reward
against the fold. σ ended 0.751 — the anneal is unconverged at this budget
(inherited limitation from B3's early stop at 1.2 M of the planned 2 M).

## Preserved-B3 expectations — Jamie's resolutions (2026-11-18)

Gate precursor on the fixed-rig tree: 3 failed / 747 passed / 3 skipped
(`ai-playground-audit/b4_fullsuite_gate_pre.txt`) — all three traced to
recorded consequences of the changed head inertia. Resolutions applied:

1. **Gait fixture re-measured** (tests `test_rl_env_b2` h=0 → 0.730 ± 0.05;
   `test_rl_ppo_integration_b3` h=90 → 0.706 ± 0.05, B4 notes in docstrings;
   old anchors preserved as historical evidence).
2. **Hostile-policy QACC: head-only gain cap in the test rng** — HEAD kp
   clamped to 400 (50× training kp, ~5× below the measured threshold
   crossing; `b4_hostile_qacc_repro.py`), rng stream unchanged, zero-warning
   assertion retained unchanged.
3. **Combat strike premise re-recorded** — caps correctly engage on both arm
   caps (raw ~953/~744 N·m vs pre-B4 ~166/~124); strike lands, holder never
   KO'd, opponent impulse unchanged within 0.02 %, self impulse 182.6 →
   143.2 N·s (cap-bound), attribution recorded, zero warnings both ways
   (`b4_combat_rerecord_probe.py` → JSON).

No safety assertion was weakened anywhere: cap enforcement, saturation
attribution, rejection, and zero-warning semantics all still assert pass in
their own right; what changed is only the recorded physics of the corrected
asset.

## Gate

| run | suite | health |
|---|---|---|
| precursor (pre-decision) | 3 failed / 747 passed (expected fails #1/#2/#3), 3 skipped | healthy |
| gate run 1 (final tree) | **750 passed / 0 failed / 3 skipped, exit 0** (`b4_fullsuite_gate1.txt`) | healthy |
| gate run 2 (final tree) | **750 passed / 0 failed / 3 skipped, exit 0** (`b4_fullsuite_gate2.txt`) | healthy |

Re-scope battery (`b4_rescope_battery.txt`): the three touched test files,
66 tests, exit 0.

## Commit

Single stage commit: the asset line, the three test re-scopes, and all B4
evidence reports (HEAD_POSTURE_DIAG, B4_HEAD_FIX log, mechanism-baseline and
per-checkpoint posture JSONs, pause log, this stage record). Viewer files
(rotateX geometry fix) are a separate commit by design.

## Outstanding limitations

- The policy's standing command drifts +0.4455 m (survives, does not stand);
  σ 0.751 unconverged anneal; sustained walk at 0.105 m/s is slow; all three
  are the measured case FOR the posture-ask / reward layer (`b4_posture1`),
  which is B4 proper and starts next per Jamie's sequencing decision.
- The foot-in-shin self-contact asset quirk (B2 deferral) remains unfixed,
  re-flagged with mechanism-row evidence; any future fix is its own
  re-baselining lineage.
- `main` was ahead of origin by 1 (`d1f6d8d`) before this gate commit; push
  timing is Jamie's call.
