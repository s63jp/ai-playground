# M7-1a — S1 limb-control calibration gate (probe result + launch decision)

**Date:** 2026-11-19 · **Evidence:** `reports/m7_s1_probe.py` →
`reports/M7_1A_S1_PROBE.json` (read-only probe; no training, no model writes).
**Basis:** `reports/M7_1A_LIMB_STAGES_CHANGE.md` §4 (fix the `limb_tracking`
coefficient `w` from measured baselines) and §5 (capability gate bands).

## 1. What the probe did

Per S1 limb group it replayed the stage's `command_schedule` with the
`limb_tracking` coefficient pinned to **1.0**, so each window's reward
component equals exactly `−mean_err²` over the active group in normalized
joint units — the same units the trained reward uses. Two "bad tracking"
controls (**zero-action** PD-hold and a **fresh 256×256 TanhGaussianActor**
deterministic mean action) were contrasted against the **P2 PD-reference**
"good tracking" row (`ai-playground-audit/m7_p2_pd_reference.json`, converted
to normalized err²). Seed base 97,010,000.

## 2. Measured rows (normalized err², per window, steady holds)

| group | bad err² (zero) | bad err² (untrained 256×256) | P2 good err² | separation | base rwd/window |
|---|---|---|---|---|---|
| left_arm | 0.1359 | 0.1341 | 0.0037 | 0.1322 | 2.50 |
| right_arm | 0.1529 | 0.1525 | 0.0037 | 0.1492 | 2.50 |
| head | 0.1376 | 0.1399 | 0.0035 | 0.1342 | 2.50 |
| torso | 0.1071 | 0.1073 | 0.0015 | 0.1056 | 2.50 |
| left_leg | 0.0906 | 0.0908 | 0.0241 | **0.0665** | 2.50 |
| right_leg | 0.0906 | 0.0910 | 0.0241 | **0.0665** | 2.50 |
| all | 0.1047 | 0.1049 | 0.0120 | 0.0927 | 2.50 |

The untrained 256×256 policy's mean action ≈ zero-action across every group
(its final-layer init is near-zero), so both are genuine "cannot track yet"
baselines. The legs show the smallest separation because their P2
good-tracking error is the largest (0.194/0.149 rad — the heaviest load).

## 3. The measured coefficient window

- **Signal floor** `w ≥ 0.05 / min(separation) = 0.05 / 0.0665 = 0.752`
  (binding group: `left_leg`). Below this the reward cannot separate good
  from bad leg tracking by the required 0.05/window.
- **Equilibrium ceiling** `w ≤ 0.05 · base/window / max(P2 err²)
  = 0.05 · 2.50 / 0.0241 = 5.181` (binding group: `left_leg`). Above this the
  residual penalty when tracking *well* exceeds 5% of the standing reward.
- **Feasible window: [0.75, 5.18].** The provisional **w = 2.0 sits inside
  it** (the geometric-mean pick, √(0.75·5.18)=1.97, also rounds to 2.0).

**Decision: `limb_tracking` coefficient `w = 2.0` is measured-confirmed.**
It is not at either edge, so it is robust to small measurement error. No
change to the shipped value was needed — the probe validated it.

## 4. Capability gate bands (recorded, report §5)

Per-joint steady abs-err bands = P2 measured abs-err × margin 2 (full table in
`M7_1A_S1_PROBE.json` `groups.*.gate_bands_abs_err_rad`). Examples: head
pitch/yaw/roll ≤ 0.192/0.181/0.180 rad; left hip/knee ≤ 0.389/0.297 rad.
Posture bands verbatim B4: head |rel pitch| ≤ 0.35/0.60, head world-up ≥
0.9/0.8, chest |tilt| ≤ 0.35/0.60.

**Known conservatism:** the pinned joints (ankles, waist) are commanded to
neutral in S1, but the probe currently records their band from the P2
*tracked*-error row (e.g. left_ankle 0.688 rad) rather than an inactive-rest
variance band. These loose bounds are non-binding for the commanded joints and
will be tightened from measured zero-action variance when the first real S1
held-out eval runs (§5.2 inactive-rest bands).

## 5. Launch gate — STILL CLOSED pending authorization

`M7_S1_PROVISIONAL` remains **True**: `scripts/rl_ppo.py train --stages m7_s1`
is still refused. Per the M7-1a report §7, the run launches only after review
of this gate report. No individual-limb training has been started.

To open the gate and launch (a deliberate, separate step):
1. Set `M7_S1_PROVISIONAL = False` in `playground/rl/curriculum.py` (and update
   `tests/test_rl_m7_s1_stages.py::test_launcher_refuses_provisional_m7_ladder`
   to assert the gate mechanism via a monkeypatched flag rather than the
   shipped value — so gate coverage is retained, not weakened).
2. Launch the fresh v4 lineage per §7 (`--operation fresh --stages m7_s1
   --hidden 256,256`, own run dir under `experiments/rl/m7_s1_run1`).
