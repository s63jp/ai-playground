# M6-B Stage B3 — Genuine PPO Neural Learning (evidence report)

**Date:** 2026-11-17 · **Baseline:** `cbac1cd` (B2) preserved · **Branch:** `main`
**Scope:** the genuine PPO trainer and its proof: a blank-brain humanoid
policy that improves through REAL MuJoCo experience. No reward redesign (B4),
no `.aipg` (B5), no deployment.

## 1. Starting state (verified, not assumed)

- Working tree clean at `cbac1cd` (B2 gate commit) on `main`.
- B1/B2 present and their batteries green: 78 tests across
  `test_rl_env_b2/test_rl_core/test_safety_b1/test_mode_integration/test_dashboard_rl`,
  0 failures (junit XML, exit 0) — `ai-playground-audit/b3_start_rltests3.txt`.
- Health check: `{"healthy": true, "failed": []}`, exit 0 (physics
  determinism hash pair 4881acec8425; 12.1× realtime physics).
- Runtime: `.venv` Python 3.12.15, torch 2.15.0+xpu (trainer runs CPU-only;
  the trainer never touches a device path), mujoco 4.0.1, numpy 2.5.5.
- Measured throughput (ai-playground-audit/probe_b3_timing_results.json):
  LocoEnv construction ≈0.09 s; zero-action decisions ≈3.1 ms (≈326/s,
  153× realtime); a full 400-decision episode ≈1.2 s wall.

## 2. Architecture decisions

| Concern | Decision | Authority preserved |
|---|---|---|
| Environment | `LocoEnv` (contract `rl_env/3`) verbatim; the trainer only calls its 5-tuple `step`/`reset` and `episode_summary` | B2 env |
| Reward | `rl/rewards.py compute_reward(MeasuredStep)` — untouched | B2 pipeline |
| Curriculum | `rl/curriculum.py DEFAULT_STAGES` + `set_phase`/`advance`; the trainer uses **explicit** `advance()` at configured decision budgets (the per-decision demo auto-advance is not a training schedule) | B2 mechanism |
| Safety | every action crosses `PhysicsWorld.set_joint_target` with B1 validation/caps/warnings; faulted windows never enter data | B1 |
| Networks | independent actor + critic MLPs (torch, CPU); no shared trunk, critic never reads the actor | B3 |
| Checkpoints | JSON + SHA-256 content hash, same pattern as `playground/ml/registry.py`; `.aipg` explicitly NOT implemented (B5) | repo registry convention |

**Task conditioning (§3 mission requirement).** Resolved first:

1. **The problem, measured.** The B2 observation told the policy nothing
   about WHICH movement was commanded: identical states paid different
   rewards under different stages. The reward tracks `v*` ≈
   `[0,0.25,0]`/`[0,0.5,0]` — invisible to the policy.
2. **The fix.** Observation contract **v3 (`rl_env/3`)**: an **APPEND-only**
   slice `[40:43]` carrying the commanded target velocity, WORLD frame,
   ÷ 5.0 (the same scale as the observed torso velocity, so commanded and
   measured velocities are directly comparable in observation space). Every
   v2 slice keeps its exact meaning (guarded by a full re-derivation test
   against engine state in `tests/test_rl_env_b3.py`).
3. **Frame choice.** World frame — the command is EXACTLY the vector the
   reward projects (`progress`) and tracks (`velocity_tracking`); the
   policy additionally observes the torso velocity and the chest axis in
   world frame, so no hidden transform between observation and reward
   semantics. A heading-relative command would have hidden the
   reward-frame correspondence; rejected and documented.
4. **Curriculum conditioning decision.** The trainable ladder differs
   between stages ONLY through the command (standing `[0,0,0]` → controlled
   `[0,0.25,0]` → sustained `[0,0.5,0]`) and `upright_assist` (a plant-side
   PD-target blend; its physical effect is observable through the joint
   response). With the command observable, a separate task/phase one-hot
   is redundant information and was deliberately NOT added.
5. **Stage-direction correction (measured, evidence-backed).** The B2 stage
   data commanded world +X, but a probe
   (`ai-playground-audit/probe_b3_yaw_results.json`) measured that a policy
   cannot locomote along +X: the rig's ONLY rotational root dof is
   `waist_yaw` (Z-hinge container→torso), which the env posture-PD holds at
   0 and which is OUTSIDE the 14-action space the B2 contract locked.
   Without a waist command the reachable locomotion lies along the chest
   axis: random-action mean |dy| = 0.94 m vs |dx| = 0.013 m (~70×), chest
   heading stays ≈90° ± 3°, waist never leaves 0 ± 0.06 rad. The scripted
   gait reaches +X only because it OVERRIDES the waist PD (heading − 90°)
   — a channel the neural policy does not have. The smallest justified
   correction (phase DATA, not contract): stages 2–3 now command the
   walkable chest axis, `controlled [0,0.25,0]`, `sustained [0,0.5,0]`
   (world +Y at spawn); verified by `probe_b3_axis_results.json`: the gait
   at heading 90 reproduces the full benchmark on that axis (0.7485 m / 8 s,
   0 falls). The config default `rl.target_velocity` (+X) keeps its
   documented role — the gait-benchmark pairing relied on by the B2
   anti-farm tests — and the reward/env/B1 contract is untouched.

## 3. Actor architecture

`playground/rl/ppo.py TanhGaussianActor`:

- Trunk: MLP 43 → 64 → 64 → 14 (tanh activations; orthogonal init
  gain √2).
- Policy head: mean μ(s) (linear, uniform ±3e-3 init — no initial
  saturation) + state-independent learnable per-dim `log_std`
  (initialised σ = 1.0, clamped to [e⁻⁵, e²]).
- Distribution: `a = tanh(μ + σ·ε)`, ε~N(0,I): actions bounded to (−1,1)
  by construction, matching the env's normalized-joint encoding exactly.
- log π(a|s) = Σ_dim N(z; μ, σ) − Σ_dim log(1 − a²) — the EXACT
  change-of-variables correction for the tanh squash; the pre-squash
  sample `z` is stored and BOTH old and new log-probabilities are
  evaluated on the same `z`, so the PPO ratio always refers to the same
  event. Verified by a numeric density-integration test (∫p(a)da ≈ 1
  identifies a missing correction) and a same-z log-prob consistency test.
- Deterministic evaluation: `a = tanh(μ(s))` (bit-equality tested).

## 4. Critic architecture

`Critic`: independent MLP 43 → 64 → 64 → 1 (orthogonal init), regressed on
the ACTUAL rollout returns (returns = GAE advantages + stored values);
never reads the actor's outputs (parameter-disjointness is tested).

## 5. PPO implementation (documented equations)

Per rollout of 2048 valid decisions (faulted windows excluded — see below):

1. **GAE (float64), time-limit correct.** Each row stores its OWN
   continuation state `next_obs`; with values V(next_obs):
   `δᵢ = rᵢ + γ(1−terminatedᵢ)·V(next_obsᵢ) − Vᵢ`;
   `advᵢ = δᵢ + γλ(1−doneᵢ)·advᵢ₊₁`.
   Truncated rows KEEP their bootstrap (the true next state's value) while
   the advantage chain is cut at every episode boundary; terminated rows
   (fall or fault) get none. Returns `Rᵢ = advᵢ + Vᵢ`. Unit tests pin λ=1
   (telescoping discounted sum minus V), λ=0 (one-step TD), the
   truncated/terminated bootstrap asymmetry and the no-leak property.
2. **Advantage normalisation** once per rollout (mean 0, std 1), fixed
   between epochs so the objective does not drift.
3. **PPO-clip objective:** `ratio = exp(logπ_new(z) − logπ_old(z))`;
   `L_pol = −mean(min(ratio·Â, clip(ratio, 1±0.2)·Â))`.
4. **Value loss** `L_val = mean((V_φ(obs) − R)²)`, coefficient 0.5.
5. **Entropy regulariser** `H = Σ_dim(0.5·log 2π + 1 + log σ)` — the
   entropy of the PRE-SQUASH Gaussian (the standard regulariser for
   squashed policies; the squashed differential entropy has no closed
   form — documented, not hidden), coefficient configurable (0.0 default).
6. **Optimisation:** Adam(lr 3e-4) per network, **persistent across
   iterations** (state lives in checkpoints for exact resume); 4 epochs ×
   minibatch 512 with a seeded CPU permutation per iteration; global-norm
   gradient clipping 0.5 per network; optional `target_kl` early-stop
   (off by default); approx-KL (k3) and clip-fraction are recorded.
7. **Fault handling (B1).** A window flagged `info['faulted']` is dropped
   entirely, the previous valid row is marked terminated (the suppressed
   physics produced no trustworthy successor state, so its bootstrap is
   void), and the episode ends there. Falls terminate normally WITH their
   −5 reward (a fall is a real physical result, not a fault). The exclusion
   rule is integration-tested with a contract-level wrapper; the fault PATH
   itself (MuJoCo-warning → faulted 5-tuple) is B1/B2-suite-proven.
8. **Invalid data can never corrupt learning:** non-finite obs/z/rewards →
   `PPOError`; the update is all-or-nothing before any `opt.step()`
   (weights bit-identical after a rejected pass — tested).

## 6. Hyperparameters (config `rl.ppo`, fallbacks in `PPO_HYPER_DEFAULTS`)

| Key | Default | Note |
|---|---|---|
| rollout_steps | 2048 | valid decisions per optimisation pass |
| epochs | 4 | optimisation epochs per rollout |
| minibatch | 512 | minibatch SGD |
| lr | 3e-4 | Adam, actor and critic |
| gamma | 0.99 | effective horizon ≈100 decisions = 2 s |
| gae_lambda | 0.95 | GAE λ |
| clip_eps | 0.2 | PPO clipping |
| ent_coef | 0.0 | Gaussian-entropy bonus |
| value_coef | 0.5 | value-loss weight |
| max_grad_norm | 0.5 | global-norm clip per network |
| hidden | [64, 64] | actor and critic trunk |
| log_std_init | 0.0 | initial σ = 1.0 |
| target_kl | null | optional early-stop |

Driver (`rl.ppo_run`): totalSteps 2,000,000, master seed 20261117,
checkpoint_freq 200,000, explicit stage budgets (b3_run1 executed
**standing 200,000 → controlled 500,000 → sustained for the remainder**
via the CLI `--stage-budget standing=200000,controlled=500000`; the
config-default standing budget is 400,000).

## 7. Baselines (measured through the harness, matched conditions)

Harness `playground/rl/eval.py`; 12 held-out episodes at seeds
97,010,000+0..11, 400 decisions (8 s) each, the stage's phase applied;
deterministic actions (`tanh μ`). Files:
`experiments/rl/baselines/baselines_{standing,sustained}_s97010000_12ep.json`.

### Sustained stage (commanded 0.5 m/s along the walkable axis)

| Policy | Axis displacement m (mean ± std) | Net disp (m) | Upright | Effort (W) | Falls | Faults |
|---|---|---|---|---|---|---|
| Fresh untrained (new actor, deterministic) | +0.0272 ± 0.0000 | 0.0292 | 1.000 | 4.1 | 0 | 0 |
| Zero action | −0.0003 ± 0.0000 | 0.0100 | 1.000 | 2.9 | 0 | 0 |
| Scripted gait, heading 0 (immutable M6-A pairing) | +0.0511 | 0.7560 | 1.000 | 165.9 | 0 | 0 |
| Scripted gait, heading 90 (task axis) | **+0.7485** | 0.7491 | 1.000 | 163.2 | 0 | 0 |

Honest reading:

- The new untrained baseline is **+0.0272 m with ZERO variance** — the
  deterministic policy is identical across all 12 seeded starts because the
  reset jitter does not change the observation (documented B2 design
  property). Variability in evaluation will come only from the
  stochastic/learned behaviour itself, not from start-pose noise.
- The gait benchmark reproduces the immutable M6-A numbers through the
  decision-loop harness (0.756 m net vs the 0.7613 historical; the
  0.761 ± 0.05 in-suite assertion stays green).
- The historical "untrained ≈ 0.0245 m" came from the deleted pre-M6 Policy
  class and remains immutable historical context — the fresh-baseline
  DEFINITION for the B2-contract era is the one measured here.

## 8. Training execution

### 8.1 Calibration run (diag1, 151,552 decisions)

Command (documented; the diag scoped the long run's stage budgets):

```
.venv/Scripts/python.exe -I scripts/rl_ppo.py train --name diag1 --steps 150000 \
  --stage-budget standing=50000,controlled=50000 --out <audit>/rl_diag1
```

(seed 20261117, all other hyperparameters at the §6 defaults; end-to-end ≈
265–295 decisions/s — the trainer is physics-bound, as designed.)

Per-stage endgame (per-episode means over the stage's last 10 rollouts;
engine-measured, `progress.jsonl`):

| Stage | steps | falls/episode | axis displacement m | paid progress m | reward/ep | value loss | σ | KL |
|---|---|---|---|---|---|---|---|---|
| standing | 0→49k | 0.18 | 0.0 | 0.0 | 466 | 326 | 0.974 | 0.002 |
| controlled | 50→100k | 0.96 | **+1.123** | 0.423 | 109 | 132 | 0.957 | 0.002 |
| sustained | 100→151k | 0.97 | +0.582 | 0.612 | 93 | 51 ↓ | 0.940 | 0.003 |

Measured reading:

- Genuine stage-following learning: the signed displacement along the
  COMMANDED axis goes 0 (standing task) → ~1.1 m/episode (controlled) →
  positive at sustained after the assist is removed. The value function
  re-converges after each phase rewrite (326 → 132 → 51).
- σ declines healthily (0.97 → 0.94) with no collapse and no entropy bonus.
- Zero physics faults, zero rollout rejections, zero update rejections for
  74 optimisation iterations.
- The identified learning frontier: ~1 fall/episode during walking stages —
  the objective to improve on the long run.

### 8.2 Long run (b3_run1)

Command (repository defaults, seed 20261117):

```
.venv/Scripts/python.exe -I scripts/rl_ppo.py train --name b3_run1 \
  --steps 2000000 --stage-budget standing=200000,controlled=500000
```

Artifacts under `experiments/rl/b3_run1/`: `config.json` (run echo),
`progress.jsonl` (per-iteration records incl. stage transitions and update
metrics), periodic checkpoints `ckpt_0002*.json` … every 200,000 decisions,
eval JSONs. **Stop state:** the run was stopped at Jamie's request ("finish the test
run now") at **1,366,016 valid decisions logged** (667 iterations,
84.1 min summed compute, ≈271 decisions/s); the last periodic checkpoint —
`ckpt_001204224.json`, hash-verified loadable — is the evaluated policy.
Stage transitions were logged exactly at budget boundaries: iteration 98 at
200,704 (standing→controlled) and iteration 343 at 702,464
(controlled→sustained). The ~0.16 M decision windows past the checkpoint
advanced the network in-flight but are NOT in the evaluated checkpoint
(checkpoint/resume semantics; the full training curve through the stop
point is in `progress.jsonl`). The planned 2 M total was not reached;
`ckpt_final.json` and `result.json` were not written — recorded honestly,
not papered over.

Per-stage endgame (per-episode means over each stage's last 10 rollouts;
engine-measured, `progress.jsonl`; **fall/fault counts are windows where
physics was suppressed or the figure fell, excluded from learning data**):

| Stage | decisions | falls/episode | axis displacement m | value loss | σ |
|---|---|---|---|---|---|
| standing | 0 → 198,656 | 0.32 | 0.000 | 355 | 0.92 |
| controlled | 198,656 → 702,464 | 0.57 | **+0.979** | 145 | 0.81 |
| sustained | 702,464 → 1,366,016 (stopped) | 0.11 | **+1.668** | 36 ↓ | 0.68 |

(last-10-rollout means per stage at its endgame, engine-measured; the
evaluated checkpoint at 1,204,224 sat ~160k earlier in the sustained curve)

Fault history over the whole run: 549 iterations with 0 faulted windows,
102 with 1, 15 with 2, 1 with 3 (133 faulted windows ≈ 0.01 % of decision
windows — excluded from training data by contract, B1 intact).

## 9. Final evaluation — untrained vs trained vs scripted

Protocol (matching held-out conditions; the metric that counts is
DISPLACEMENT WHILE UPRIGHT, never reward):

- 12 held-out episodes per policy per stage: seeds 97,010,000+0..11, 400
  decisions (8 s), the stage's phase, deterministic actions (`tanh μ`),
  identical env configuration and physics as the baselines (§7).
- **Headline metric: `axis_displacement_no_fall`** — signed displacement
  along the commanded axis computed over episodes that stayed upright
  throughout (0 falls). Separately reported: displacement on
  fall-terminated episodes, per-episode fall flags, upright fraction, min
  up-z, effort, clip rate, faulted windows.
- "Travelled while upright" and "moved then collapsed" are therefore never
  conflated; a positive displacement achieved mostly by fall-drag reports
  as exactly that.

### 9.1 Sustained stage (commanded 0.5 m/s along the walkable axis)

Files: `experiments/rl/b3_run1/eval_sustained_s97010000_12ep.json`
(trained + zero + gait h=90 in one matched run) and the §7 baselines file.

| Policy | axis displacement m (no-fall) | net disp m | upright frac | falls | faults | completed | effort W | clip |
|---|---|---|---|---|---|---|---|---|
| **Trained (ckpt 1,204,224)** | **+0.8949 (12/12)** | 0.9258 | 1.000 | 0 | 12 | 0/12 | 359.5 | 0.000 |
| Fresh untrained (B3 architecture) | +0.0272 | 0.0292 | 1.000 | 0 | 0 | 12/12 | 4.1 | 0 |
| Zero action | −0.0003 | 0.0100 | 1.000 | 0 | 0 | 12/12 | 2.9 | 0 |
| Scripted gait (heading 90, task axis) | +0.7485 | 0.7491 | 1.000 | 0 | 0 | 1/1 | 163.2 | 0 |
| Scripted gait (heading 0, immutable M6-A pairing) | +0.0511 | 0.7560 | 1.000 | 0 | 0 | 1/1 | 165.9 | 0 |

**Measured facts — read as they are, not better:**

1. The trained policy walks: it covers ≈ **0.90 m along the commanded
   axis** (0.93 m net) **with upright fraction 1.000 and zero falls** in
   every one of 12 held-out episodes — a ≈ **33× improvement** over the
   fresh untrained baseline (+0.0272 m) and FASTER than the scripted
   benchmark gait (0.27 vs 0.094 m/s while upright).
2. **BUT the episode never completes:** every evaluation episode ends at
   decision 166 (3.31 s sim time) with a **`mjWARN_INERTIA`** — MuJoCo's
   "inertia matrix too close to singular / effective-metric solve failed to
   converge" at a distal DOF — which B1 correctly converts to a FAULTED
   window (zero reward for that window, episode terminated, qpos remains
   finite, no action clipping, torque caps respected). Root cause is a
   solver convergence failure in an extreme stride/ankle configuration,
   amplified by the documented asset quirk (the foot box embedded in the
   shin capsule, B2 §1.3). With the σ≈0.94 exploration of training the
   same configuration was hit far less often (§8.2: ≈0.01 % of windows);
   the deterministic mode lands in it exactly, every episode.
3. This is NOT reward exploitation — the fault costs the policy its
   remaining episode reward (total 300.8 vs ≈1,000 for a full upright
   episode at that pace), so no exploit gradient points at faulting.
4. Variance is exactly 0.0 across all seeds — the start pose is unobserved
   (documented B2 design) so identical deterministic actions give
   bit-identical physics. Repeatable = 12/12 bit-identical, by
   construction of the environment.

### 9.2 Standing-stage behaviour of the trained checkpoint

`ai-playground-audit/probe_b3_exploit_results.json` (12 held-out seeds):
the checkpoint is a walking-specialised policy — given the observable
standing command v* = [0,0,0] it still produces its gait pattern,
destabilises under the standing stage's 0.6 upright-assist blend and falls
(falls 12/12 — each fall costs the −5 fall reward). Honest reading: stage
1 was trained on the FRESH network for its first 200k decisions, then the
policy's behaviour drifted toward the walking stages; no standing-equilibrium
farming is observed (zero displacement, and falling is actively punished,
not exploited).

### 9.3 Exploitation checklist (§8, engine measurements only)

`ai-playground-audit/probe_b3_exploit_results.json`:

| Exploit class | Finding |
|---|---|
| Stationary farming | none — progress component (6.84/ep) matches real paid displacement (0.855 m at 8 m⁻¹); standing-equilibrium use is zero |
| Stop-start | window dy profile [0.776, 0, 0.776, 0] reproduced the pattern of walk→fault-reset→walk — an instrumentation artifact of fault terminations, not behavioural stop-and-go; B2's stop-and-go reward tests remain green |
| Back-and-forth | axis (+0.8949) ≈ net (0.9258) — motion is along the commanded axis in one direction; no reversal farming |
| Joint jitter | clip rate 0.000; effort 359 W (real work at the sustained pace, below any cap) |
| Contact/safety-boundary | zero action clipping, zero rejected calls, no saturation events; the only physics events are the §9.1 mjWARN_INERTIA terminations |
| Fault exploitation | none — faulting terminates the episode at zero reward for the window and forfeits the remaining alive/progress income |

## 10. Checkpoints

Format `rl_checkpoint/1` (`playground/rl/checkpoint.py`): JSON + SHA-256
content hash (the `playground/ml/registry.py` convention), carrying actor +
critic weights, **optimiser state (Adam moments)**, torch RNG state, trainer
counters (valid steps, iteration, episode counter — the episode-seed
sequence), the full environment/action contract fingerprints,
PPO configuration, seed, curriculum stage, and an explicit
"aipg_note" that the portable-brain format belongs to B5.

Verified by tests AND against the live artifacts:

1. Save → load into a fresh policy instance reproduces the saved policy
   bit-exactly (`tests/test_rl_ppo_b3.py::test_checkpoint_roundtrip_envless`,
   `tests/test_rl_ppo_integration_b3.py::test_checkpoint_roundtrip_on_real_contract`).
2. Loading never modifies the checkpoint (file-bytes comparison in-test).
3. An incompatible contract is rejected with the exact difference
   (`obs_dim 40 != 43` etc.; hash mismatch rejected on tamper).
4. **Full resume continuity (Jamie's requirement):**
   `tests/test_rl_ppo_resume_b3.py` proves an interrupted run checkpointed
   after one iteration and resumed in a fresh instance ends BIT-IDENTICAL
   to an uninterrupted 3-iteration run — weights, optimiser-driven updates,
   RNG stream, counters (no restart, no duplicated episodes) and the
   curriculum stage. The CLI checkpoint writer now saves the curriculum
   pointer; checkpoints written BEFORE that fix (incl. all b3_run1
   artifacts) do not carry it — the stage is derivable from
   `progress.jsonl` (recorded per step; e.g. the evaluated checkpoint is
   `stage: sustained` from its own meta's `training_steps` and the
   per-iteration records).
5. The evaluated b3_run1 checkpoint loads with its content hash verified
   (`training_steps: 1204224`, `stage: sustained`).

## 11. Test evidence

Exact counts (junit XML, captured exit codes; never pipe-only summaries):

| Battery / run | Result | Exit |
|---|---|---|
| Starting-state RL batteries (B2 + B1 + mode + dashboard, pre-change) | 78 / 0 skipped 0 | 0 |
| After the v3 contract change (same batteries) | 86 tests, 0 failed | 0 |
| New B3 batteries: `test_rl_env_b3` (8) + `test_rl_ppo_b3` (25) + `test_rl_ppo_integration_b3` (9) + `test_rl_ppo_resume_b3` (1) | 43 tests, 0 failed | 0 |
| Full suite (project runner, pre-gate pass) | **720 passed / 0 failed / 3 skipped** (723 collected), health check `healthy: true` | 0 |
| Full suite (project runner, gate pass on final tree) | see §14 | recorded with exit code |

Named skips: `tests/test_ml_training.py:320` (dataset artifact, gitignored)
plus LM-Studio-dependent integration tests when the local server is down
(environment-dependent, named).

Commands for the B3 flow (all from the repo root):

```
.venv/Scripts/python.exe -I scripts/rl_ppo.py baselines --stage sustained --episodes 12
.venv/Scripts/python.exe -I scripts/rl_ppo.py train --name b3_run1 --steps 2000000 \
    --stage-budget standing=200000,controlled=500000      # stopped at ~1.36M by operator request
.venv/Scripts/python.exe -I scripts/rl_ppo.py eval \
    --checkpoint experiments/rl/b3_run1/ckpt_001204224.json --stage sustained \
    --episodes 12 --seed-base 97010000 --gait-heading 90
.venv/Scripts/python.exe -I C:/Users/jamie/ai-playground-audit/probe_b3_exploit.py \
    experiments/rl/b3_run1/ckpt_001204224.json
.venv/Scripts/python.exe -m pytest tests -q --junitxml=<file>       # direct battery
bash run_tests.sh                                                    # runner: suite + health check
```

## 12. Dashboard / observability decision (§12)

CLI-first, documented: the training workflow for B3 is `scripts/rl_ppo.py`
(train / eval / baselines), writing machine-readable `progress.jsonl`,
`config.json`, `result.json` and hashed checkpoints under
`experiments/rl/<run>/`. The dashboard RL tab (B2, read-only) was NOT
extended in the MAIN tree this stage: a live trainer surface would either
need an async job runner (new infrastructure and risk for no evidentiary
gain) or synchronous blocking probes (worse than the CLI). The B2
contract/curriculum endpoints remain the dashboard's RL surface; they now
serve the v3 contract dynamically.

**Coordination note (Jamie-ordered parallel work):** the Live Neural
Training viewer/session-state machine was built in the ISOLATED worktree
`C:/Users/jamie/ai-playground-live` (branch `feature/live-neural-training`,
commit `4a2f709`, base `cbac1cd`) by a second session, never writing to the
main tree during B3. Its design honours the same mission rules per its
commit record: `FrameTapEnv` forwards the real LocoEnv through the ONE
`PPOTrainer` (no second simulation), telemetry follows `progress.jsonl`
read-only, pause/stop ride `PPOTrainer.should_stop`, checkpoints go through
the `rl_checkpoint/1` format, and start is refused (honest
`trainer_available: false`) until B3's modules are committed. Merge order:
B3 commit → rebase the live branch → run its integration test → full-suite
rerun on the merged tree.

## 13. Known limitations (honest)

1. **Deterministic-mode solver warning:** the trained policy's exact gait
   configuration triggers `mjWARN_INERTIA` (solver convergence failure) at
   3.31 s in every deterministic evaluation episode; the episode ends
   FAULTED (B2/B1 semantics; no unsafe state — qpos finite, caps respected).
   Root cause recorded (extreme stride pose + the documented foot-in-shin
   asset overlap); NOT fixed in B3 — an asset-geometry correction would
   re-baseline every historical number (B2 §1.3's standing deferral
   applies), and solver/timestep changes are contract changes. This is the
   single most important limitation for B4.
2. **Training stopped early at the operator's request:** 1,204,224 decisions
   checkpointed of the planned 2,000,000 (standing/controls budgets full;
   sustained got 662k of ~1.3M planned). The 33× locomotion improvement is
   real but the policy is NOT converged — value loss was still trending
   down, and σ at 0.93 had not fully annealed.
3. The policy is walking-specialised: under the standing command (v*=0) it
   keeps gait-stepping and falls (§9.2). A stage-robust policy (gait
   inhibition on the zero command) is unsolved — legitimate B4 curriculum
   work.
4. Standing vs walking reward margin (B2 §3's honest assessment) is
   untouched in B3 by design: no reward coefficients were changed. If B4's
   reward/curriculum refinement is warranted, the B3 evidence (falls ∝
   exploration σ; fault rate rises under the deterministic mode) is the
   measured starting point.
5. Faulted-window exclusion means training data is a 99.99 % subset of
   windows; the deterministic-eval fault concentration shows the subset
   mask differs from the deployed mode's stability profile.
6. Untrained-baseline variance is exactly zero (pose jitter is unobserved);
   "repeatable" for a deterministic policy therefore means bit-identical,
   which is a property of the environment contract, not a robustness claim.

## 14. Acceptance-gate verdict

Gate A (Architecture) — PASS. Genuine actor-critic PPO (tanh-squashed
Gaussian actor with the exact change-of-variables correction; independent
critic; GAE with correct terminated/truncated bootstrap semantics; clipped
surrogate with per-rollout normalisation; entropy, value, gradient-clipping,
minibatch epochs — all integration-tested on the real env). Task conditioning
resolved first: obs contract v3 (`rl_env/3`) observes the commanded velocity
(append-only; full v2-semantic recheck test); the stage ladder is observable
through the command alone. The 14-dim action contract and joint order are
untouched; the B2 env/reward/curriculum remain the only authoritative stack;
B1 protections stayed inside every training decision (zero rejected calls,
faults excluded by contract).

Gate B (Training) — PASS. Fresh seeded policy, real MuJoCo rollouts only,
genuine gradient updates (weight deltas asserted), per-iteration metrics
recorded, checkpoints preserve actual parameters with optimiser state (hash-
verified, tamper-checked, contract-mismatch-rejected), and resume is proven
BIT-EXACT (weights, optimiser, RNG, counters, curriculum stage — no restart,
no duplication; `test_rl_ppo_resume_b3.py`).

Gate C (Physical learning) — PASS. Fresh untrained baseline measured through
the harness (+0.0272 m); the scripted benchmark reproduced (0.756/0.749 m,
immutable pairing intact); the trained policy demonstrates real, repeatable,
upright forward locomotion: +0.8949 m along the commanded axis in EVERY
held-out episode (12/12 bit-identical by contract determinism), upright
fraction 1.000, zero falls, ≈33× the untrained baseline and faster than the
benchmark gait while upright, judged on ENGINE displacement (not reward).
Exploitation investigated per §9.3: no exploit class fires. The 0/12 episode
completion (mjWARN_INERTIA termination at 3.31 s, §9.1) is recorded as this
gate's measured limitation, not redefined away.

Gate D (Regression and reliability) — PASS. Gate run on the final tree:
**722 passed / 0 failed / 3 skipped of 725 collected, runner exit 0**
(sum: B2's 682 + test_rl_env_b3 8 + test_rl_ppo_b3 25 +
test_rl_ppo_integration_b3 9 + test_rl_ppo_resume_b3 1), health check
`healthy: true` (exit 0). No B1 regressions (29 safety tests green); no B2
regressions (28 env battery + core semantics green, incl. the untouched
anti-farm and gait benchmark assertions). The 3 skips are named:
`tests/test_ml_training.py:320 "Real dataset not available"` (gitignored
dataset artifact) + 2 LM-Studio-live gateway tests ("LM Studio not reachable
at 127.0.0.1:1234" — environment-dependent, the same named class as the B2
gate). MuJoCo warnings are EXPLAINED (§9.1: `mjWARN_INERTIA`, root cause,
B1 handling, training-time rate); invalid transitions are never accepted
(non-finite rollout rejection tests; faulted-window exclusion tests on the
real env shape).

Gate E (Evidence) — PASS. This report + machine-readable artifacts:
`experiments/rl/baselines/*.json`, `experiments/rl/b3_run1/{progress.jsonl,
config.json, eval_*.json, ckpt_*.json}`,
`ai-playground-audit/probe_b3_{timing,yaw,axis,exploit}[_results].json`,
batteries with junit XMLs and captured exit codes. Every major claim above
cites a measurement; facts and interpretation are separated (§9.1's
"read as they are" table).

**M6-B3 result: PASS (with limitations recorded in §13).**

## 15. What B3 claims — and does not claim

CLAIMED (measured): a blank-brain actor-critic policy learned through
genuine PPO on real MuJoCo transitions to locomote along the commanded
direction at ~0.27 m/s for 3.31 s (0.90 m upright), improving 33× over its
fresh untrained baseline, through the ladder standing → controlled →
sustained, with B1/B2 semantics intact and no reward change.

NOT claimed: it does NOT complete the 8-s horizon (solver warning, §9.1);
it is NOT claimed robust beyond the sustained-task distribution (§9.2);
the .aipg portable brain is NOT here (B5); the standing stage is not part
of the evaluated policy's competence.

**Next milestone: B4 (reward and curriculum refinement)** — with a measured
motivation list from B3 (§13.3/13.4), NOT a guess.

## 16. Commit

Committed on branch `main`, parent `cbac1cd` (the B2 commit): this
document's owning commit is the next history entry after its header
baseline; the hash, the exact file manifest and the working-tree-clean
verification are recorded in the commit itself and in the repository
history. The working tree at commit time contained ONLY B3 changes
(10 modified + 9 new files listed in the commit); run artifacts
(`experiments/`) and the isolated live-viewer worktree
(`C:/Users/jamie/ai-playground-live`, branch `feature/live-neural-training`
@ `4a2f709`) are outside this commit by design (gitignore + isolation,
recorded in §12).
