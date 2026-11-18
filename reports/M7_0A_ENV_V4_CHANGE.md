# M7-0a — Environment contract v4: the ONE change set, documented BEFORE implementation

**Date:** 2026-11-18 · **Status:** implementation spec (per Jamie's approval of M7-0 in principle, conditioned on following `M7_0_CONTRACT_FREEZE_20261118_DRAFT.md` v2 exactly). Written before code, per the b4_posture1 rule. **Nothing in this stage touches PPO, reward coefficients in use, the curriculum ladder, physics, or any run artifact.**

## 1. What M7-0a is (and is not)

ONE reviewable change set (the v2 freeze §9): the env speaks contract `rl_env/4` by default — obs 68 (43-bit-preserved prefix + the 25-dim command block), action 15 (name-ordered, waist appended), command state + mask, four new KEYED reward components with config-default 0.0, contract fingerprints updated, playback made contract-driven for read-only legacy brains. **Not in this stage:** stage ladders using the commands (M7-1+), samplers, target placement (M7-5), the surgical loading law (M7-0b), trainer changes (never).

## 2. Contract switch (the design that keeps legacy lineages intact)

- `LocoEnv` gains an **instance contract version** from `rl.contract_version` (absent ⇒ 4). Allowed: {3, 4}; anything else raises.
- **v4 (default):** obs 68 / action 15; posture-PD joints = left/right_foot_yaw only; the waist joins the policy gains table (50, 5).
- **v3 (legacy, read-only evaluation surfaces):** obs 43 / action 14; **bit-identical to the current tree** — same joint enumeration result (verified equal to model enumeration), same gains/posture table, same obs code path executed verbatim. `set_commands` raises on a v3-mode env (that contract has no command block).
- The frozen **name order** (v2 §2 law) is the single source of truth for v4:
  `[spine_L2, left_shoulder, left_elbow, right_shoulder, right_elbow, head_pitch, head_yaw, head_roll, left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle, waist_yaw]`
  A construction test asserts the 14-name prefix equals the model-order enumeration on the current rig (drift detector for the model itself); the waist is never taken from model order (it is joint index 3).
- Class attributes `LocoEnv.obs_dim/action_dim/contract_version` remain as **class-level defaults = 68/15/4** (symbolic uses follow); v3-mode instances override at instance level.

## 3. Env changes (itemized)

1. Gains: v4 `_gains` += `waist_yaw: (50.0, 5.0)`; `_posture_gains` = foot_yaws (8,1) only. v3 mode keeps today's tables exactly.
2. Obs: build v3 identically first (same code, no branches inside); v4 concatenates the command block:
   - `[43:58]` limb_pose_target — the commanded normalized targets × mask bit 0; inactive ⇒ zeros;
   - `[58:60]` com_shift — **normalized n ∈ [−1,1]** (n × 0.070 m x / n × 0.040 m y from the anchor); inactive ⇒ zeros;
   - `[60:63]` waypoint (ux, uy, d̂) × mask; `[63:64]` stop × mask; `[64:68]` cmd_mask (limb, balance, waypoint, stop), fixed order.
3. Command state + `set_commands(mask=..., limb_targets=..., com_shift=..., waypoint=..., stop=..., limb_group=...)` with loud validation (shape/range/finiteness; `limb_group` ∈ the 6 groups + "all", default "all"). `set_phase` gains the optional `commands` key (non-cumulative: absent ⇒ zero commands, mask all-off); `commands` in a v3-mode phase raises.
4. Effort `_dofadr` follows `joint_names` (14 v3 / 15 v4) — the measured growth N2 records.
5. `observation_layout()/action_layout()` version-conditional dicts; the v3 dict stays byte-comparable with the current fixture content (the `.aipg` manifest reproduces exactly this schema).
6. COM: v4 envs compute the full-body COM (all rig bodies incl. the root column — the P4 method) each window and pass measurements to rewards; v3 mode computes nothing new (the components are inert there).
7. Playback (`rl_live/playback.py`): a cheap pre-read of the checkpoint's `meta.contract` (json only — the strict loader still performs hash/tamper + fingerprint verification) selects `contract_version` for the playback env, then the full strict load runs unchanged. Effect: a recorded v3 brain is replayed on a v3-mode env (bit-identical to today), a recorded v4 brain on v4 — the brain's contract decides the sim. No training path changes.

## 4. Reward equations (documented before implementation, Jamie's rule)

`rewards.py` DEFAULTS gains four CONFIG-DEFAULT-0.0 keys: `limb_tracking`, `com_tracking`, `approach_progress`, `stop_tolerance`. `COMPONENT_KEYS` grows 8 → 12. Frozen constants live in `rewards.py` (imported by env): `COM_ANCHOR = (−0.07926, 0.08871)` m (P4 settled full-body), `COM_SHIFT_SCALE = (0.070, 0.040)` m, `WAYPOINT_D_REF = 4.0` m (P3), `STOP_BAND_DEFAULT_M = 0.5` m (config key `stop_band_m`, re-measured at M7-5).

With MeasuredStep gaining engine-measured fields (defaults keep every hand-built fixture bit-identical):
- `commands`: dict (mask 4-bools, limb_targets[15] normalized, com_shift[2] normalized, waypoint[3], stop 0/1) — default: all zeros / all-off;
- `joint_pos_norm`: the current 15 normalized joint positions (v4 envs; `None` default);
- `com_xy`: the measured full-body COM xy (v4; `None` default);
- `target_point_xyz`: the env's target-block point (v4; **`None` default — M7-0a has no placement machinery; approach/stop are computed only when a target point exists**);
- `dist_to_target_before/after` derived in env.step.

Exact equations (all gate on `commands["mask"]` bits — mask off ⇒ the term contributes nothing, no coefficient scaling can resurrect it):

```
limb_tracking      = −c.limb_tracking · mean_{j ∈ limb_group} (p_j − t_j)²
                     # p_j = joint_pos_norm[j] (measured); t_j = limb_targets[j];
                     # group "all" = the 15; per-group = the §3 group members.
                     # Mask off OR (joint_pos_norm is None) ⇒ 0.0.
com_tracking       = −c.com_tracking · ‖ (com_xy − COM_ANCHOR) − n_shift ⊙ COM_SHIFT_SCALE ‖²
                     # metres²; n_shift = com_shift normalized command.
                     # Mask off OR (com_xy is None) ⇒ 0.0.
approach_progress  = +c.approach_progress · max(0, d_before − d_after)
                     # metres per window toward target_point; Mask(waypoint) off
                     # OR (target_point_xyz is None) ⇒ 0.0.
stop_tolerance     = −c.stop_tolerance · max(0, d_after − stop_band_m) · stop
                     # stop mask bit AND stop command AND target point required;
                     # otherwise 0.0.
```

**Baseline contribution (the invariance argument):** every DEFAULT coefficient is 0.0 **and** every existing stage's mask is all-off ⇒ every existing fixture and run computes the identical reward as the current tree, window for window (the b4_posture1 default-field pattern). Test-visible change: `info["components"]` and episode summaries gain four keys valued 0.0 — tests asserting exact totals are unaffected; tests asserting exact component-key sets are updated (enumerated in §7). No existing reward behavior changes until M7-1+/M7-2+ stages request coefficients — one-change-per-lineage discipline preserved.

## 5. Loader (this stage: unchanged; M7-0b: MANDATORY acceptance extension)

Per Jamie's mid-turn directive (2026-11-18): the M7-0b fingerprint-diff extension (joint **ranges + kd** alongside the existing dims/slices/order/mapping/kp) is a **mandatory transfer-loader acceptance requirement**, not an optional hardening — two brains whose fingerprints pass on dims alone but differ in control mechanics are not compatible and must never silently load. Recorded in the freeze doc; M7-0b implementation without it cannot pass its gate.

## 6. Environment-designated disturbance velocity (recorded)

The reserved `disturbance` stage's phase currently carries the stale `target_velocity [0.5, 0.0, 0.0]` (+X, pre-axis-lock). Per Jamie's directive: **M7-3's design MUST correct it** (redefined v\* for the recovery task, measured basis) before the stage is ever `env_ready`. Untouched in M7-0a (the stage is reserved; nothing trains into it).

## 7. Test changes (enumerated, with reason)

| Test | Change | Reason |
|---|---|---|
| `test_rl_core.py:25` | `(43,)` → `(LocoEnv.obs_dim,) == (68,)` | the live default contract |
| `test_rl_env_b2.py:439` | `14/3` → `15/4` **and** asserts the v3-mode env still reads `14/3` | freeze §2 (dual-contract) |
| `test_rl_env_b3.py:32-33` | same pattern (default 68/4; v3 mode 43/3) | freeze §2 |
| `test_dashboard_rl.py:26-34` | served layout `4/68/15` | dashboard serves the LIVE contract; asserts the v4 sections exist |
| `test_rl_ppo_integration_b3.py:227` | expected message string made computed (`!= {env.obs_dim}`) | the test is about v2-fingerprint rejection, not the literal 43 |
| `test_rl_live_integration_b3.py:52` | acceptance set `(40, 43)` → `(40, 43, 68)` | live sessions save v4 metas now |
| `test_safety_b1.py::test_env_rejects_wrong_action_dimensions` | wrong-shape list refreshed: `(14,)` `(16,)` `(15,1)` wrong, `(15,)` valid; rejected count 3 → 4 | M7-0a shape semantics |
| `test_safety_b1.py::test_env_flags_faulted_after_numerical_warning` | the rogue-PD joint moves `waist_yaw` → `left_foot_yaw` (the passive hinge): under v4 the waist is ENV-COMMANDED and a rogue waist write is overwritten by the action path (the new contract working); the semantic (a rogue write on a non-action joint is detected → FAULTED) is preserved | M7-0a seam semantics |
| new: `test_rl_contract_v4.py` | append-only N1 (bit-identity of `[0:43]` between v3/v4 modes under identical seeds/phases/actions); action mapping/name-order/ranges/gains; command-block obs semantics (mask zeros vs active values); command setter validation; reward-component mask gating + zero-default fixture-identity; `set_commands` rejection on v3 mode | the new contract's own tests |

No behavioural fixture (gait 0.7301/h90 0.7057, standing invariance, B1 zero-warning hostile, B2 anti-farm, restoration tests) has its expectation weakened; each must pass **unmodified** — that is N1's meaning. The hostile-policy test drives `PhysicsWorld` directly with an explicit 17-joint list including `waist_yaw` — unaffected by the env's action count.

## 8. The M7-0a acceptance gate (maps 1:1 to the six approval conditions)

1. **43 dims unchanged** → N1 pytest (bit-equality over a standing + hostile sweep) + v3-mode layout equality.
2. **Waist as action 15, name-ordered** → construction/mapping pytest (name order, ranges ±π, 0 → 0 rad, gains (50,5), saturation source tag `neural_policy`).
3. **No unintended neutral-waist movement** → N2 (below) waist row + P1's zero-torque neutral row as prior.
4. **Standing equivalence ≥ 4,800 windows** → N2 probe, corrected protocol (see addendum): 12 × 400-window standing episodes in LOCKSTEP (legacy + v4), obs `[0:43]` bit-identity per window, the reward difference exactly the designed effort-set term, 0 falls / 0 faulted, per-window reward ≈ the measured 2.4972 standing row. Durable JSON under `reports/`; pytest carries a 400-window smoke for the suite.
5. **v3 fixtures/tests valid** → full suite ×2 exit 0 with only the §7 enumerated (freeze-justified) updates; all behavioural numbers untouched.
6. **No leakage into historical lineages** → playback v3 regression green (b3_run1 replay bit-exact as before: same eval-identity checks), no writes to any run dir (mtime/SHA discipline as at every stage).

### Addendum (N2 v2, measured 2026-11-18)

The first N2 run exposed two facts that sharpened the protocol (both recorded honestly in `m7_0a_n2_standing_equivalence.json`):

1. **The continuous (single-reset) 4,800-window standing run is NOT the B4 standing row**: zero-action standing drifts slowly and eventually falls beyond the 400-window episode horizon (measured: ~1.23 m drift, first fall ≈ window 3,450 of the continuous run, per-window reward halved by the fallen suffix). The measured B4 row (2.4972/window, 0 falls) is a per-episode standing equilibrium — N2 v2 therefore runs the gate as 12 × 400-window episodes (same protocol family as the historical row). The healthy prefix of the continuous run reproduces 2.4972/window exactly.
2. **The reward equality gate is the designed effort relation, not bit-equality**: v4's effort measurement set grew 14 → 15 dofs, so `r4 − r3 == −effort_weight · (eff4 − eff3)` EXACTLY (measured: violations 0, delta ≈ 1e−12 J overall); the observed-state prefix stays BIT-identical throughout. The first run's "mismatches" were exactly this relation at ULP scale during the continuous run's tumble.

Stopped after the gate for reporting; M7-0b (loading law) is a separate change set.
