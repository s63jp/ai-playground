# M7-1a — Individual-limb (S1) training system: operator guide

**Date:** 2026-11-19 · **Status:** complete, tested, verified end-to-end in the
live UI. Builds on the M7-1a limb ladder + calibration (reports/
M7_1A_LIMB_STAGES_CHANGE.md, M7_1A_S1_GATE.md) and the 256×256 architecture
scaling. The B3/B4 walking ladder and the 2M experiment are untouched.

## What you can do now

Select an individual limb, pick a neural architecture, train it live while
watching the rig and telemetry, then evaluate its held-out capability against
recorded gate bands and save its brain — all from the live-training page, the
sidecar HTTP API, or the CLI.

## 1. Launch gate (satisfied by evidence, not bypassed)

`playground.rl.curriculum.m7_s1_launch_block_reason()` returns `None` only
while the shipped `limb_tracking` coefficient `w` sits inside the measured
feasible window `[0.75, 5.18]` (reports/M7_1A_S1_PROBE.json). Every launch
path — the CLI `--stages m7_s1` and the live session — checks it, so an
uncalibrated coefficient refuses training. w=2.0 is in-window, so the gate is
satisfied. This is distinct from the **mastery** gate below.

## 2. Training

- **Live page** (`playground.rl_live.server`, or the dashboard mount):
  objective dropdown lists the 7 S1 limbs (left/right arm, head, torso,
  left/right leg, all). Pick an architecture (64×64 / 128×128 / 256×256 /
  256×256×256 / custom), seed, steps, lr, and the standing warm-up, then Start.
  The session builds the 2-stage ladder (standing base → the selected limb)
  and trains through the ONE real `PPOTrainer`. Pause/resume/stop/save work as
  before; the 3D viewer shows the actual trained rig.
- **CLI:** `scripts/rl_ppo.py train --stages m7_s1 --hidden 256,256
  --stage-budget standing=N ...` trains the full 7-stage S1 ladder.
- `--hidden` selects per-run actor/critic widths without editing YAML;
  checkpoints restore their OWN recorded architecture.

## 3. Capability evaluation — the HONEST mastery gate

`playground.rl.s1_eval.evaluate_s1_limb(checkpoint, stage_id, ...)` runs
held-out deterministic episodes and measures per-joint steady tracking abs-err
(rad), pinned-joint rest, chest posture, and falls, comparing each against the
recorded gate bands (P2 × margin 2). A limb is reported **mastered only** when
every commanded joint is within band, posture holds (chest-up ≥ 0.9), and
there are zero falls — never on step count or reward.

- **Live page:** "Evaluate capability" (enabled when paused/stopped) renders a
  per-joint err-vs-band table and the verdict.
- **API:** `POST /api/rl_live/evaluate {experiment, checkpoint, stage_id,
  episodes}`; refused while training is actively running.
- **CLI:** `scripts/rl_ppo.py eval-s1 --checkpoint C --stage m7_s1_head
  [--require-mastered]`.

## 4. Per-experiment checkpoint management

Each run has its own dir under `experiments/rl_live/<name>` with
`run_config.json` (records `limb_stage_id`, architecture, budgets),
`progress.jsonl`, `ckpt_*.json`, and `s1_eval_<stage>.json`. List them with
`GET /api/rl_live/experiments`; read a stored verdict with
`GET /api/rl_live/experiment/<name>/eval/<file>`.

## 5. Verified end-to-end (2026-11-19)

Driven in the live browser UI: selected the head limb, 64×64, trained 4000
steps (stage transition standing→m7_s1_head observed, `limb_tracking` reward
live at −45.7), stopped+saved `ckpt_final.json`, then evaluated — verdict
**NOT YET** (head pitch/yaw/roll err ~0.50–0.55 rad vs ~0.18–0.19 bands,
posture ✓, 0 falls, 12 episodes). The honest gate correctly withholds mastery
from an undertrained limb. A longer production run is the operator's to launch.

## 6. Preserved

DEFAULT_STAGES (B2/B3/B4) byte-identical; the 2M `m7_256_2m_run` experiment
untouched (gitignored, on disk); the walking objective and the single PPO
trainer unchanged.
