# M6-B — Stage B4 (second gate): the posture-ask reward lineage (`b4_posture1`)

**One reward change, per Jamie's single-change contract:** upright/alive
consume `min(torso_up_z, chest_up_z)` (chest-aware posture). Stability term
unchanged; no head-up term; no mechanical/asset/curriculum/obs/action/PPO
changes. Change documentation (exact equations + measured baseline
contributions BEFORE implementation, per Jamie's requirement):
`reports/B4_POSTURE1_REWARD_CHANGE.md`. Implementation: `rewards.py`
(`chest_up_z` field, default 1.0 keeps all pre-change fixtures bit-identical),
`env.py` (chest measured from the `spine_L2` body, engine measurement only —
obs contract untouched), `docs/RL_ENVIRONMENT.md` §3, unit test. Battery at
implementation: 73 tests exit 0.

**Standing invariance on real behaviour (Jamie's scrutiny, point 1, verified
not assumed):** zero-action standing under the new code, 4,800 real windows —
`chest_up_z ≡ 1.0` exactly, old-rule offline recompute delta ≡ 0 on 0.0 % of
windows, reward 2.4972/window identical under both rules, 0 falls, net
0.01 m (`B4P1_STANDING_CHECK_zeroaction.json`).

## Run record

Seed 20261117, 1.2 M steps, ladder standing 200k → controlled 500k — the
b3/b4_head1 ladder exactly, for row-level comparability. Completed at
1,200,128 (it. 586, wall 4,424.8 s), exit 0. **`result.json`:
faulted_windows_total = 0, rollout_rejections = 0 across the entire run.**
Every checkpoint pulled and probed as it landed (40-ep stochastic replays,
seeds 990100+, save-time stage) plus matched-window telemetry aggregation
(hardened parser semantics verified against raw records).

## Per-checkpoint trajectory (40-ep replays; falls/complete = falls/completed)

| ckpt | stage | falls/complete | fwd/bwd | axis m/ep | head ≥0.9 | chest up ≥0.9 (windows) | rule cost/window |
|---|---|---|---|---|---|---|---|
| 200,704 | controlled | 34/6 | 9/31 | −0.65 | 22.2 % | 99.2 % (min 0.811) | −0.007 |
| 401,408 | controlled | 17/23 | 35/5 | +1.16 | 54.6 % | 95.6 % | −0.013 |
| 602,112 | controlled | 21/19 | 39/1 | +1.83 | 63.1 % | 95.1 % | −0.014 |
| 802,816 | sustained | 40/0 | 39/1 | +1.74 | 57.7 % | 71.3 % | −0.040 |
| 1,003,520 | sustained | 40/0 | 40/0 | +1.70 | 62.3 % | **82.6 %** (mean 0.949) | −0.025 |
| final | sustained | 29/11 | 36/3 | +1.86 | 4.7 % | 56.7 % (mean 0.891) | −0.058 |

b4_head1 same-point anchors (same protocol): 200k 39/1 axis −0.21;
401k 22/18 axis +1.04; 602k 30/10 axis +0.21; 802k 19/21 with head 3.2 %
/ chest ~3 % (the collapse point); 1M 13/27 axis +0.36; final 11/29.

Key findings:

1. **The motivating failure did not return.** head1's head-up regime peaked
   in controlled (68.3 %) and collapsed to ~3 % the moment the sustained
   switch arrived (chest folding to −1.5 rad by 1 M). posture1's chest regime
   survived the identical transition: 71.3 % ≥0.9 at 802k, **82.6 % at 1 M**
   — and at final the DEEP fold never exists: chest ≥0.5 in 99.7 %/≥0.4 in
   99.8 % of final windows, at a rule cost the policy pays (−0.058/window)
   and still walks. head1's final chest world-up ≥0.9 was ~2-3 %.
2. **The alive flip never became the dominant force**: 0.1-0.0 % of windows —
   the fold was PRE-EMPTED (upright gradient 0.481→0.472/window) rather than
   punished after the fact. Jamie's scrutiny point 2 answered: the training
   signal did not need its heaviest lever, and movement was not discouraged —
   progress/tracking per window are the highest of all three lineages
   through the entire run.
3. **Transition shock, then speed-stability**: posture1's sustained replay
   rows show 40/40 falls at 802k/1M (episodes ~2.7 s, walking ABOVE the 0.5
   m/s command — a speed-stability frontier, not a fold), improving to 29/40
   by the final checkpoint. The matched EVAL plane (below) is clean.

## Contract step 8 — matched held-out evals (seeds 97010000+, saved+copied)

| row | axis m/ep | speed m/s | falls | faulted | complete | reward | effort W |
|---|---|---|---|---|---|---|---|
| **B4-posture1 ckpt_final sustained, 12 ep** | **+3.5194** | **0.485** | 0 | 0 | **12/12** | 746.0 | 618 |
| B3 ckpt_final 2.0M sustained, 10 ep (old rig) | +2.9426 | 0.463 | 0 | 0 | 10/10 | 781.7 | — |
| B4 head1 ckpt_final sustained, 12 ep | +0.6197 | 0.105 | 0 | 0 | 12/12 | 675.2 | 48.2 |
| **B4-posture1 ckpt_final standing, 12 ep** | — | 0.058 | 0 | 0 | **12/12**, drift **+0.1758 m** | 936.7 | 173.4 |
| B4 head1 standing, 12 ep | — | 0.061 | 0 | 0 | 12/12, drift +0.4455 | 976.8 | 20.6 |

Fixed-rig baselines re-run (identical to head1-era: fresh +0.0211, zero
−0.0001, gait h=0 0.7301, gait h=90 +0.7057 — rig unchanged, confirming the
comparability chain).

## Acceptance list (Jamie's), answered

1. **Chest-up percentages**: mid-sustained 82.6 % ≥0.9; final 56.7 % ≥0.9 /
   82.0 % ≥0.8 with the alive floor held (≥0.4 at 99.8 %, ≥0.5 at 99.7 %);
   deep-fold regime permanently gone (head1 final: ~2-3 % ≥0.9).
2. **Sustained completion and falls**: 12/12, 0 falls, 0 faulted (matched
   eval); stochastic-probe falls declined 40→40→29 with episodes lengthening
   135→293 windows (speed-stability learning in progress).
3. **Forward displacement and speed**: **+3.52 m/ep at 0.485 m/s — the
   fastest held-out walker of any lineage** (5.7× head1's +0.62 @ 0.105;
   beats B3's 2.0M best +2.94 @ 0.463 on the old rig, with zero fault-outs).
4. **Standing-command drift**: +0.176 m vs head1's +0.4455 (−61 %); 0 falls
   both. Still drifts (standing still not directly asked; reward 936.7 vs
   zero-action 998.9 — in-place activity at 173 W is the standing-effort cost).
5. **Reward components + entropy**: upright/window 0.472 (chest-aware cost
   live but small); alive/window ≈ +1.0 (flip never dominant); σ ended 0.802
   (head1 0.751) — anneal still unconverged at the 1.2 M budget (inherited).
6. **Physics faults + rejections**: 0 and 0 across the whole run (all three
   lineages on the fixed rig now train fault-free).

## Honest trade-offs and open levers

- **Head-up collapsed at the end of sustained** (62.3 % @ 1M → 4.7 % final):
  the head is UNREWARDED by contract (deferred), and the policy traded head
  posture for gait speed exactly as the diagnosis predicted an un-asked-for
  head would. The deferred head-up term and the stability-term repoint remain
  the next single-change levers, each to be contracted and measured
  separately per Jamie's one-change rule.
- **Late-sustained chest uprightness slipped** (82.6 % ≥0.9 @ 1M → 56.7 %
  final) in exchange for the fast gait; the deep fold stayed impossible.
- Standing still is still un-asked-for (drift, in-place effort).
- σ anneal still unconverged at the fixed 1.2 M budget.

## Gate

| run | suite | health |
|---|---|---|
| gate run 1 (post-change tree) | **751 passed / 0 failed / 3 skipped, exit 0** (`b4p1_fullsuite_gate1.txt`) | healthy |
| gate run 2 | **751 passed / 0 failed / 3 skipped, exit 0** (`b4p1_fullsuite_gate2.txt`) | healthy |

751 = 750 (head-fix gate tree) + the new chest-aware min-rule unit test.
Commit-scope note (Jamie): the gravity-roadmap document
(`B4_GRAVITY_CURRICULUM_DESIGN.md`) is deliberately committed SEPARATELY from
this experiment (next-milestone scope, not posture1 evidence).

## Next (pending Jamie's acceptance decisions)

The single-change contract is complete and measurable either way; the follow
up levers (stability repoint, head-up term, budget/σ extension) and the
anthropometric + gravity-curriculum milestone (Jamie's decided structure in
`reports/B4_GRAVITY_CURRICULUM_DESIGN.md`) await his sequencing.
