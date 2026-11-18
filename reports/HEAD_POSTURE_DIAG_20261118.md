# Watcher-run head/posture diagnosis — 2026-11-18

Session `watch_alongside_b3` (stopped cleanly by its Stop path at 653,312 steps /
iteration 319, controlled stage, seed 20261118 — nothing re-trained, nothing
reset; all probing below ran as a separate read-only process against its own
physics instance). The 585,728/iteration-286 reading on the panel was a
mid-run snapshot of the same session.

Evidence: `ai-playground-audit/probe_head_diag.py` + `probe_patchcheck*.py`
probes, results in `ai-playground-audit/probe_head_diag_results.json`. Code
refs are `playground/rl/...` on main @ d1f6d8d.

## The eight questions

1. **Does the upright reward measure torso orientation only? — YES, and worse:
   it measures the one part of the rig that cannot tip.** `upright_reward`
   consumes only `up_z` = torso `xmat` column 2 (`rewards.py:101-104`,
   `env.py:528-531,537`); `agent_fallen()` is the same torso up-vector plus a
   ROOT-HEIGHT rule (`world/physics.py:1111-1141`). The "torso" body is the
   pelvis box of a rigid slide-root column with no pitch dof (physics.py's own
   comment: "the humanoid cannot tip … no pitch dof"). Hence
   `upright_fraction ≡ 1.000` in every telemetry record — the chest+head+arms
   subtree hangs off the `spine_L2` hinge entirely unmeasured.

2. **Are head/neck orientation in the observations? — YES.** The obs contract
   (`env.py:509-548`) includes `head_pitch/yaw/roll` positions AND velocities
   (joint_order slices 5-7 pos / 12-14 vel), plus the chest is visible via
   torso-relative joint states. Seeing the head was never the problem.

3. **Do the neck/head joints have sufficient motor control? — NO; this is the
   core defect.** The head body is a 6.37 kg sphere whose COM sits 0.26 m above
   the three head hinges (asset XML `head` body: sphere
   `pos="0 0 0.26" size="0.115"`, no explicit inertial → ~6.4 kg from default
   density). Measured on the compiled model: gravity moment `m·g·d = 16.25
   N·m`, neck PD `kp=8, kd=1, joint damping 1.5, range ±1.5708 rad`
   (`env.py:162`, world `<default>`). The PD's maximum restoring torque at full
   range is 12.57 N·m < 16.25 N·m, and the destabilising gravity gradient
   (16.25 N·m/rad) exceeds the PD stiffness (8 N·m/rad) — the upright
   equilibrium is mathematically unstable and no static target inside the
   action map can hold the head near vertical. Probe-verified: with all head
   targets 0, the head runs to its −90° mechanical limit within seconds
   (chin down; both with and without the standing-stage assist). Same probe
   with a physically realistic head (1.8 kg @ 0.13 m) holds **exactly
   0.0000 rad** — the gain was never the problem, the head mass/lever was.
   (The B1 head torque cap of 20 N·m is not binding.)

4. **Is head pitch rewarded or penalised? — NEITHER.** No reward component
   contains any head measurement; `COMPONENT_KEYS` = progress, tracking,
   upright (torso), alive (torso), effort, smoothness, stability (torso
   up-vector variance), fall (`rewards.py:57-58,164-177`).

5. **Do backward falls correlate with forward head pitch? — Not within the
   653k policy itself; both directions exist and both trace to the unholdable
   neck.** Replay of the checkpoint policy (40 stochastic episodes, controlled
   phase, sampled like training): 40/40 end in HEIGHT COLLAPSE (leg fold —
   the pelvis never tips, `lean_at_end ≈ 0`), 28/40 backward vs 12/40
   forward. In the backward episodes the upper body hangs BACKWARD: head pitch
   +0.75 rad relative to a chest leaned +0.93 rad (world ≈ 95°), head COM on
   average 13.7 cm behind the neck hinge. The chin-forwards hang Jamie watched
   earlier in the run is the OTHER equilibrium of the same instability (the
   neutral-target pin at −90°). Which way the head droops follows the policy's
   own (unrewarded) head targets; the collapse follows the lean.

6. **Does the COM leave the support polygon? — YES.** Feet are 4.5×9 cm boxes.
   With a drooped head/chest the standing figure's COM sits outside the foot
   rectangle (static margin −4.5 cm, Part A); during replay walking the
   median worst-case margin is −9.4 cm and at the fall onset −4.6 cm
   (backward) / −7.7 cm (forward). The rig stands on root traction + leg-PD
   torque rather than balance, so the negative margins do not topple it
   statically — but they mean the legs carry a large persistent moment arm
   that folds under load at gait speed.

7. **Does the forward-locomotion reward encourage unstable positioning? — It
   does not reward instability, but it cannot see it and barely punishes
   falling.** Posture-free terms (alive ≈ 1.45, upright 0.5, tracking
   0.37-0.51 per window in the live records) make up ~2.3 of the ~2.45 gross
   per-window rate; the one-shot fall penalty (−5) equals about 2-3 windows of
   gross reward. Nothing in the objective mentions chest or head pose, so
   "walk backward while the upper body hangs back, collapse at the end" is not
   selected against — backward episodes in the replay actually out-earn
   forward ones (295 vs 249 avg reward; ~206 vs 161 windows), which matches the
   live telemetry's negative axis displacement (−0.89…−0.22 m in the final
   records). Backward-drift-then-collapse is the current local optimum.

8. **Is the initial posture correctly configured? — Spawn yes; maintainable
   no.** Spawn places every hinge at 0 (upright) with ±0.15 m x/y jitter
   (`env.py:294-329`) — a correct standing pose (probe: torso up 1.000, root
   0.802 m). But under neutral targets the neck immediately droops (see 3) and
   the chest slumps to −0.371 rad; the "initial posture" is correct but not
   statically maintainable on the head/neck, and only marginally on the spine
   (kp 35 vs an upper-body gravity moment of roughly 21 N·m/rad once the head
   loads it).

## Unified causal chain

Unholdable 6.4 kg/0.26 m head on a kp=8 neck → the head lives far off
vertical in one of two gravity-dominated equilibria (chin-pin forward at the
−90° limit when targets sit near 0; backward hang when the policy's head
channels drift positive — by 653k they command near-full-backward targets) →
the chest follows (spine +0.5…+1.0 rad) → COM displaced outside the 4.5×9 cm
feet → walking load carried by torque instead of balance → legs fold at gait
speed, mostly backward → the reward never sees any of it (pelvis-only
upright ≡ 1.0, dead stability term, −5 fall penalty) and the collapse mode is
reward-competitive → it persists.

A secondary observed mode: one replay episode slid 18.3 m at ~2.3 m/s
(runaway traction slide) — outside these eight questions but worth noting.

## Recommended smallest changes for a subsequent experiment

Do NOT change: gravity, leg/spine PD gains, the action space, the obs
contract (the head is already observed), episode length, or the PPO
configuration. Also honour "no reward/checkpoint changes to the stopped run"
— all of the below is for a NEW run.

1. **Asset fix (one line, physics-only).** Decouple the head's mass from its
   visual on `assets/humanoid_mujoco.xml` head body by adding an explicit
   inertial, e.g. `<inertial pos="0 0 0.13" mass="1.8"
   diaginertia="0.010 0.011 0.009"/>` (realistic human head+neck is roughly
   5 % body mass). Verified in-memory: the same kp=8 neck then holds the head
   at exactly 0.0 rad under neutral targets (droop ≤ ~0.3 rad, 5× torque
   margin, B1 head cap untroubled). Caveat: comparable to the deferred
   foot-in-shin asset quirk — gait-benchmark/eval numbers from b3_run1 are
   not apples-to-apples on a changed rig; treat the next run as a new
   lineage (new run name), old checkpoints still load (contract fingerprints
   are unchanged) but cross-evals are not comparable.

2. **Reward term so the (now achievable) posture is asked for.** The 653k
   policy's backward-chest/backward-head habit was measured to persist under
   fix 1 alone (27/40 backward collapses unchanged) — mass fixing is a
   capability change, not a behaviour change. Smallest honest addition:
   measure the CHEST axis in the posture terms so the hanging-chest regime
   stops being free (feed `min(torso_up_z, chest_up_z)` — or a separate small
   `posture` component — into `upright`/`alive`), and optionally a small
   head-up term since it is now holdable. One new measured field in
   `MeasuredStep` plus one component keeps the pipeline shape.

3. **Repoint the dead stability term (optional).** `stability` is the variance
   of the PELVIS up-vector (≡ 0.000 always — the pelvis cannot pitch). The
   same term on the chest axis gives the balance pressure a live channel
   without new engine measurements.

Sequencing: 1 first (or 1+2 together in one next experiment); 2 without 1
would ask for a posture the rig cannot hold; 1 without 2 shows capability
gains only after re-training.

## Status of the training layer

- `watch_alongside_b3` is untouched (ckpt_final.json = 653,312 steps, hash
  verified by the loader during probing; progress.jsonl/run_config.json never
  written by the probes).
- The dashboard (8765) + sidecar (8766) processes from last night are still
  up with the stopped session loaded; no trainer is running since 01:19.
- Pending from last session (deferred, not started): viewer
  resume-from-checkpoint support; the CDP-verified viewer rig `rotateX` fix
  is still uncommitted in the working tree.

Follow-up experiment: `reports/B4_HEAD_FIX_20261118.md` — the B4 controlled
lineage, including the preserved mechanism-only baseline (B3 1.204M weights
replayed on the fixed head: 95 % completion vs 0 % on the old rig, physical
correction isolated from learning).
