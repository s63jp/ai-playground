# Protected comparison baselines (B3, B4)

**Status:** protected, read-only reference brains. Never overwrite, retrain in place, convert, or delete.
**Location:** `baselines/` at the repository root (tracked by Git, not ignored).
**Machine-readable record:** [`manifest.json`](manifest.json). Every file is listed under its brain, with a repository-relative path, role, size, and SHA-256.
**Copied:** 2026-11-20 (M6-B0). Original source: the `experiments/rl/` folder of the second working copy, which lives outside this repository (read-only, unmodified). A verified copy is kept in the 2026-11-20 preservation backup, also outside this repository. Provenance identifiers are in `manifest.json`.

---

## 1. Identities

| Baseline | Final checkpoint | Content SHA-256 (stored inside the file) | File SHA-256 | Steps | Stage at save |
|---|---|---|---|---|---|
| **B3** `b3_run1` | `b3_run1/ckpt_final.json` | `b6d68a3525082d24e7c775cafc2d04459e633f92d5b879c0a38f9c2c6f6b3382` | `7c307aee249920de1273bf5aec14c10acf5c2d9927918054459266702f311f9a` | 2,000,896 | sustained |
| **B4 head** `b4_head1` | `b4_head1/ckpt_final.json` | `0dcd782595d0f49f08b068eb6e302cfc1da1de05e9dcb23e8521bba87445ca7c` | `1072edeaa4d3fc4a9e9381bc755f7ff61da91acfe6f2aa83db2a93628c6a435c` | 1,200,128 | sustained |
| **B4 posture** `b4_posture1` | `b4_posture1/ckpt_final.json` | `cea2794313a532bbb871c64b8a3749ca32cb0dbe062ca7c501d2734b0734ca8a` | `e81702ec17be904e9454c20bb751e15a211884ee3246f5d9b489c9cc943a71f3` | 1,200,128 | sustained |

All three: format `rl_checkpoint/1`, policy class `TanhGaussianActor`, `contract_version` **3**, `obs_dim` **43**, `action_dim` **14**.

### Files belonging to each brain

| Brain | Files | Folder |
|---|---|---|
| B3 `b3_run1` | 6: `ckpt_final.json`, `config.json`, `result.json`, `progress.jsonl`, `eval_sustained_s97010000_10ep.json`, `eval_sustained_s97010000_12ep.json` | `b3_run1/` |
| B4 head `b4_head1` | 7: `ckpt_final.json`, `config.json`, `result.json`, `progress.jsonl`, `eval_standing_s97010000_12ep.json`, `eval_sustained_s97010000_12ep.json`, `baselines_sustained.json/baselines_sustained_s97010000_10ep.json` | `b4_head1/` (subfolder kept as in source) |
| B4 posture `b4_posture1` | 7: `ckpt_final.json`, `config.json`, `result.json`, `progress.jsonl`, `eval_standing_s97010000_12ep.json`, `eval_sustained_s97010000_12ep.json`, `baselines_sustained_s97010000_10ep.json` | `b4_posture1/` |

Role of each file: `ckpt_final.json` is the loadable brain. `config.json` is the training configuration. `result.json` is the run outcome. `progress.jsonl` is the training log. The `eval_*` and `baselines_*` files are held-out evaluations. Only `ckpt_final.json` is needed to load a brain; the rest document it.

## 2. Original training runs

Source folders (second checkout, read-only), all seed `20261117`:

- **B3, `experiments/rl/b3_run1/`:** PPO, 2,000,896 valid steps (977 iterations), 164 faulted windows, 0 rollout rejections. Its config records `started_from_checkpoint` = `b3_run1/ckpt_001204224.json`. That intermediate checkpoint is **not** part of this baseline; the lineage is recorded only.
- **B4 head, `experiments/rl/b4_head1/`:** 1,200,128 steps (586 iterations), 0 faulted windows, 0 rollout rejections. Resumed from `ckpt_000401408.json` (intermediate, not copied). Stage budgets: standing 200k, controlled 500k.
- **B4 posture, `experiments/rl/b4_posture1/`:** 1,200,128 steps (586 iterations), 0 faulted windows. Trained **from scratch** (`started_from_checkpoint: null`) with the posture-ask reward.

Evaluation evidence (held-out seeds `97010000+`) is copied alongside each final: `eval_*`, `baselines_*`, `progress.jsonl`, `result.json`, `config.json`.

Not copied, deliberately: the intermediate `ckpt_0002…–ckpt_0018…` checkpoints and the walk GIFs. They remain in the source folder and the preservation backup.

## 3. Physics differences: B3 vs B4

| | **B3 (`b3_run1`)** | **B4 (`b4_head1`, `b4_posture1`)** |
|---|---|---|
| Rig | **Old rig:** head `inertial` derived from geometry | **Fixed rig:** `assets/humanoid_mujoco.xml` head gains an explicit `<inertial pos="0 0 0.13" mass="1.8" …>` |
| Head mass / COM | 6.3706 kg, COM +0.26 m above the neck hinges | 1.8 kg, COM +0.13 m above the neck hinges |
| Head gravity moment vs neck PD capacity | 16.25 N·m demand vs 12.57 N·m capacity (unstable upright equilibrium) | 2.30 N·m vs 12.57 N·m (≈5× margin) |
| Observed effect | Head pinned at −90° under neutral targets; upright reward blind to it | Head collapse removed |
| Faulted windows in training | 164 | 0 |

Unchanged between them: observation and action contracts, all other bodies, gains, and geoms. B1's 20 N·m head cap is unaffected.

**Comparison rule:** B3 numbers come from the old rig and are historical evidence, not comparable physics, against B4 or any fixed-rig run. Any B3-vs-B4 or B3-vs-new comparison must state the rig.

## 4. Observation and action contracts (`rl_env/3`)

**Observation (43 floats)**, from the checkpoint's `meta.contract.observation`:

| Section | Slice | Unit / encoding |
|---|---|---|
| `joint_pos` | [0, 14) | normalised, `(2q − lo − hi)/(hi − lo)` ∈ [−1, 1] |
| `joint_vel` | [14, 28) | rad/s |
| `torso_up` | [28, 31) | unit vector |
| `torso_lin_vel` | [31, 34) | m/s |
| `upright_margin` | [34, 35) | raw |
| `torso_forward` | [35, 38) | unit vector |
| `foot_ground_force` | [38, 40) | N |
| `command` | [40, 43) | m/s, world frame, ÷5 (appended in v3) |

**Action (14 floats)**, `joint_targets_normalized`: `target = lo + (a + 1)·0.5·(hi − lo)` per joint. The joint list is in each checkpoint's `meta.contract.action.joints`.

**Deterministic inference:** `a = tanh(mu(s))`.

## 5. Legacy `contract_version = 3` requirement

- The **default environment on this checkout is contract version 4** (`rl_env/4`: 68 observations, 15 actions). This is unchanged.
- `rl/checkpoint.py load_checkpoint(path, env=…)` compares the stored contract with the live env and **refuses** a version mismatch. A v4 env rejects these baselines with `contract_version 3 != 4`.
- To load and evaluate them, construct the env in the read-only legacy mode:

  ```python
  from playground.rl.env import LocoEnv
  from playground.rl.checkpoint import load_checkpoint

  env = LocoEnv(cfg, contract_version=3)   # read-only legacy evaluation mode (M7-0a)
  ck = load_checkpoint("baselines/b4_head1/ckpt_final.json", env=env)
  ```

- Converting these baselines to `rl_env/4` or to a new observation contract is **out of scope** and is not done here.

## 6. Role: protected comparison baselines

- These are the reference brains for M6 comparisons: B3 (PPO learning on the old rig), B4 head (head-mass fix, controlled → sustained curriculum), and B4 posture (posture-ask reward, from scratch).
- **Never overwrite.** New runs go in new folders. A comparison run records the baseline it is measured against by `content_sha256`.
- **Never retrain in place.** Continuations start a new run folder and record `started_from_checkpoint` against the baseline hash.
- Evaluation uses the stored seed protocols (for example `sustained_s97010000_10ep`, `standing_s97010000_12ep`) so results are reproducible.
- **Code must not write here.** The CI test `tests/test_protected_baselines.py` enforces this.

## 7. Git tracking and integrity

- `.gitattributes` contains `baselines/** -text`. Git stores these files byte-for-byte, so CRLF/LF conversion (`core.autocrlf`) cannot change their hashes on Windows or CI. Textual diffs still work.
- `baselines/` is not in `.gitignore`. Git tracks it as ordinary files (about 7 MB total; no LFS).
- CI (`tests/test_protected_baselines.py`, runs on Ubuntu, no physics or XPU marker) checks:
  - the SHA-256 of every file against `manifest.json`;
  - each `ckpt_final.json`'s internal `content_sha256`, recomputed exactly as `rl/checkpoint.py` computes it;
  - that the files on disk match the manifest exactly (no missing or extra files);
  - that no code under `playground/` or `scripts/` references `baselines/` as a write target.
- `manifest.json` is not self-hashed (that would be circular); its integrity rests on Git history and the CI check.

## 8. Verification record (M6-B0)

- Original sources hashed before copying, copied with exclusive create (no overwrite), then re-hashed. All 20 files passed; the source was unchanged after copying.
- The copies in `baselines/` were produced from the verified `experiments/baselines/` copies, then checked against the original second-checkout hashes.
- Each final checkpoint's internal content hash was recomputed from the copy: match.
