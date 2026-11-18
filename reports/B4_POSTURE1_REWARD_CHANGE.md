# B4 posture-ask reward change — exact equation documentation (pre-implementation)

Per Jamie's 2026-11-18 decision: proceed with `b4_posture1` using a
**chest-aware upright reward as the ONLY reward change**; document the exact
reward equation and baseline contribution before implementation. Companion
measurements in `ai-playground-audit/b4_chest_upz_*.json` (probes:
`b4_chest_upz_dist.py`, old code — no repo change needed for any number here).

## 1. The current reward, exactly

`playground/rl/rewards.py` is the single authority (`compute_reward`, weights
`DEFAULTS`). Per decision window (20 ms at decimation 10), with
`up_z` = pelvis/torso `xmat` col 2, z component, world frame (range [−1, 1],
`env.py` builds it and `obs[34]` is the same measurement):

| component | equation (exact) | measured per-window contribution, b4_head1 sustained final era |
|---|---|---|
| progress | `8.0 * min(max(0, (p_after − p_before) · ĥ), |v*|·0.02)` | mean 6.02/episode (≈ 0.0165/window) |
| velocity_tracking | `1.0 * exp(−‖v − v*‖² / 0.1)` | ≈ 52.0/episode ≈ 0.134/window (vs kernel max 1.0/window) |
| upright | `0.5 * up_z` | 173.9/episode ≈ 0.449/window (max 0.5) |
| alive | `+1.0 if up_z > 0.4 else −1.0` | 347.8/episode ≈ +0.90/window (never fully +1 — some windows dipped ≤ 0.4) |
| effort | `−0.005 * J_window (policy 14 dofs)` | −9.37/episode |
| smoothness | `−0.1 * ‖a − a_prev‖²` (shaping) | −35.8/episode |
| stability | `−0.2 * var(up_z, 25 windows)` | **0.0 — mathematically dead** (pelvis cannot pitch: var ≡ 0) |
| fall | `−5.0` once, if terminated by fall | −1.0/episode |
| total | sum | ≈ 533.5/episode stochastic (σ 0.75), ≈ 1.38/window |

Termination is separate from reward: `agent_fallen()` (pelvis up-vector + root
height) is the FALL TRUTH and is unchanged by this contract; FAULTED windows
(physics suppressed) stay zero-reward + excluded.

## 2. The proposed change, exactly

One new ENGINE MEASUREMENT, two component rewrites, nothing else:

- New `MeasuredStep.chest_up_z` — `spine_L2` body `xmat` col 2, z component
  (world frame). The `spine_L2` body is the chest: it carries the `g:chest`
  box geom and the whole arms+head subtree; its hinge pitch IS the chest tilt
  the posture probes measure. Identical measurement channel to the probes'
  head world-up (`xmat` col 2 z), range [−1, 1], +1 at spawn.
- `posture ≡ min(up_z, chest_up_z)`
- upright: `0.5 * posture` (was `0.5 * up_z`)
- alive: `+1.0 if posture > 0.4 else −1.0` (was on `up_z`)

STABILITY UNCHANGED (stays on torso up_z — its repoint is a second, deferred
change). Head-up term NOT added (deferred). Fall rule unchanged. All other
components byte-identical. Observation/action contracts unchanged (obs stays
43-dim `rl_env/3`; policy dofs already see `spine_L2` position at obs, so the
asked-for posture is OBSERVED — a learnable ask). B1 safety semantics
untouched.

## 3. chest_up_z range, sign, normalisation (contract item, measured)

Same physical channel and scale as `up_z` by construction — dimensionless
cosine in [−1, 1], +1 upright, negative below horizontal — so it composes with
the existing weights without renormalisation. Measured on b4_head1 replays
(12 ep each, seeds 990100+; `b4_chest_upz_*.json`):

| regime | mean | min | ≥0.9 | ≥0.4 | reward delta/window (upright + alive) |
|---|---|---|---|---|---|
| sustained, ckpt_final (folded failure) | **0.115** | −0.009 | 2.2 % | 5.6 % | **−2.33 total** (−0.44 upright, −1.89 alive) |
| controlled, 401k-era (posture built) | — (602,112: 0.954) | 0.784 | 89.2 % | 100 % | **−0.023 total** (alive never flips) |
| standing, ckpt_final (drift regime) | 0.827 | 0.774 | 7.2 % | 100 % | −0.087 total (alive never flips) |

Interpretation: the change is a near no-op for an upright-chest policy
(−2 %/window at the 602k head-up era) and a deliberate ~2.3/window penalty
exactly where the measured failure lives — comparable to the entire gross
reward rate (~1.4-2.5/window), so folding stops being reward-competitive. In
the sustained-fold regime alive flips to −1 in ~94 % of windows because
`chest_up_z < 0.4` there; this is the pressure that makes the fold expensive
at every window, not just at episode end.

## 4. Standing invariance (analytic + measured)

At the standing equilibrium (spine 0, torso up): `posture = min(1, 1) = 1` ⇒
upright = 0.5, alive = +1 — **bit-identical to the current rule**. B2's
standing-equilibrium arithmetic therefore survives unchanged. The standing
probe above (−0.087/window) is not a standing-equilibrium cost: it lands on
windows where the b4_head1 policy DRIFTS by walking with a tilted chest —
i.e. the new rule mildly prices the standing-command drift that the acceptance
list targets, without touching a true standing behaviour.

## 5. Consequences and frozen items

- Frozen per contract: no mechanical/asset/curriculum/obs/action/PPO/safety
  changes; `b4_head1` untouched and remains the immutable comparison baseline;
  historical B3/B4 evidence untouched.
- Checkpoint fingerprints cover observation/action/contract version/dims ONLY
  (not the reward code), so **old checkpoints remain loadable after this
  change**; the lineage boundary is the run dir (`experiments/rl/b4_posture1/`
  vs `b4_head1` vs `b3_run1`), the same convention B3 established. Documented
  risk: an old checkpoint loaded under the new reward would train/eval with
  chest-aware values — never done in this stage; all pre-change numbers here
  come from frozen artifacts.
- The `stability` repoint and any head-up term are explicitly deferred as
  separate future changes (Jamie: one reward change at a time, so causality
  is attributable).

## 6. Implementation footprint

`playground/rl/rewards.py` (MeasuredStep field with neutral default 1.0 so
existing unit fixtures keep old behaviour; `compute_reward` min-rule on
upright/alive + docstrings), `playground/rl/env.py` (chest body id cached at
build; `chest_up_z` measured in step() and passed to MeasuredStep),
`docs/RL_ENVIRONMENT.md` §3 (the authority doc's upright/alive lines + a B4
note), `tests/test_rl_core.py` (min-rule unit tests: fold reduces upright and
flips alive; default keeps old behaviour; standing invariance). No other file
changes.

## 7. Acceptance protocol (unchanged discipline, Jamie's list)

Run `b4_posture1`: seed 20261117, 1.2 M steps, ladder standing 200k →
controlled 500k (b3/b4 ladder), checkpoints every 200k boundary. EVERY
checkpoint pulled and probed as it lands (40-ep replay protocol, plus the
chest-upz distribution rows above as new baseline comparators). At budget:
matched held-out evals (sustained 12 ep + standing 12 ep, seeds 97010000+,
`--episodes 12`, run-dir outputs) + fixed-rig baselines; acceptance measured on
Jamie's list: chest-up/head-up percentages, sustained completion and falls,
forward displacement and speed, standing-command drift, reward components and
policy entropy, physics faults and rollout rejections. Gate: full suite ×2 +
health, commit with evidence, report.
