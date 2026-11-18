# M6-B — The RL Environment Contract (`rl_env/3`)

**Stage B2 document, extended at B3 for observation contract v3 (task
conditioning) and the curriculum stage-direction correction. Scope:
environment infrastructure ONLY — observation, action, reward, termination,
curriculum, lifecycle. Not covered here: the RL training algorithm (B3 —
see `reports/m6b/B3_PPO_LEARNING.md`), the reward programme tuning (B4),
`.aipg` brain files (B5), agent deployment.**

One environment, one reward pipeline, one curriculum mechanism:

| Concern | Authority |
|---|---|
| Environment | `playground/rl/env.py` — `LocoEnv` (contract version 2) |
| Reward | `playground/rl/rewards.py` — `compute_reward(MeasuredStep) -> (total, components)` |
| Curriculum | `playground/rl/curriculum.py` — `Curriculum` + `DEFAULT_STAGES`, applied through `LocoEnv.set_phase` |

The broken pre-M6 trainer (`playground/rl/train.py`) and the duplicate
curriculum class it coexisted with are **removed** (B2). Nothing in the
repository runs gradient training yet — see [What does NOT exist yet](#what-does-not-exist-yet).

`LocoEnv.observation_layout()` / `LocoEnv.action_layout()` return the
machine-readable schema that the (future) `.aipg` manifest must reproduce
byte-for-byte. The config block `rl:` in `config/default.yaml` mirrors the
numeric defaults used by this contract; every key has a matching fallback
constant in the code.

---

## 1. Observation contract (float32[43], layout v3)

v3 is an **APPEND-only** extension of v2 (and v2 of v1): every v2 section
keeps its slice and meaning. v3 adds slice 40–43, the commanded target
velocity ("task conditioning"; see §1.5). Section table from
`observation_layout()` (slices are `[start, end)`):

| # | Slice | Field | Units / frame | Meaning |
|---|---|---|---|---|
| 1 | 0–14 | `joint_pos` | normalised `(2q − lo − hi)/(hi − lo)` | hinge angle of the 14 policy joints, in JOINT ORDER below; measured from `d.qpos` |
| 2 | 14–28 | `joint_vel` | rad/s ÷ 6.0 | `d.qvel` of the same 14 dofs |
| 3 | 28–31 | `torso_up` | unit vector, WORLD frame | torso body `xmat` column 2 |
| 4 | 31–34 | `torso_lin_vel` | m/s ÷ 5.0, WORLD frame | root slide-joint (`agent_x/y/z`) velocity (`agent_vel()`) |
| 5 | 34–35 | `upright_margin` | raw | torso up-z (`torso_up[2]`) |
| 6 | 35–38 | `torso_forward` | unit vector, WORLD frame | torso `xmat` **column 1** — the rig's chest/heading axis (the rig faces its local +Y; measured during the scripted walk, column 1 rotates to `(cos h, sin h)` for heading h) |
| 7 | 38–40 | `foot_ground_force` | N ÷ 1000 (left, right) | ground-support normal force under each foot: contacts whose counterpart body is **outside this rig** (self-pairs are excluded — see §1.3) and with \|contact normal z\| > 0.7 (the same support rule `PhysicsWorld._post_step` uses for `grounded`); forces summed via `mj_contactForce`, snapshot at the decision boundary |
| 8 | 40–43 | `command` | m/s ÷ 5.0, WORLD frame (v3, B3) | the commanded target velocity `v*` that the reward tracks — the movement objective itself; same scale as `torso_lin_vel` so commanded and measured velocities are directly comparable. Changed at every `set_phase`/`set_target_velocity`, deterministic |

**Joint order** (model order, probe-verified; constant for a rig revision):

```
spine_L2, left_shoulder, left_elbow, right_shoulder, right_elbow,
head_pitch, head_yaw, head_roll, left_hip, left_knee, left_ankle,
right_hip, right_knee, right_ankle
```

Model joint ranges (all limited; read live from the compiled model):
±1.5708 rad for the 14 policy joints (waist_yaw and the two foot_yaw
hinges — the 3 passive posture joints — are ±3.1416 but are NOT observation
fields; their state is indirectly observable through torso/limb response).

**Coordinate frames.** World frame: +x east, +y north, +z up (config
header). Torso up/forward are expressed in this world frame via `d.xmat`
(never quaternions). Joint angles are internal hinge coordinates; the
sagittal limbs are X-hinges (positive hip = leg forward along the chest
axis, negative knee = heel-back flexion — the gait-controller convention,
`locomotion.py`).

**Normalisation.** Fixed per-field scales, documented above (6.0 rad/s
covers the measured walk envelope 4.83 rad/s with 1.24× margin; 5.0 m/s
covers the 4.0 m/s `max_speed` cap with 1.25× margin). There is **no
running-statistics standardisation** and **no observation clipping** —
values may exceed the ±1 band when physics is pushed beyond the scripting
envelope; the network sees the honest numbers.

**Invalid/unavailable measurements.** None are synthesised. If the engine
state itself goes non-finite (or a MuJoCo warning fired during the window),
the transition is FAULTED and the observation is zeroed before return
(§5); faulted transitions must not enter training data.

**Deliberately NOT observed:** absolute root position (the task is
velocity- and posture-conditioned; a seeded reset moves the world start
pose without changing the first observation — this is a documented design
property, verified by test), other agents, held objects, terrain shape.
`layout_version` 4 would add such fields APPEND-only (v3 followed exactly
this rule by appending the command).

### 1.5 Task conditioning (added at B3, layout v3)

Before B3 the observation did not tell the policy WHICH movement it was
being asked to perform — identical physical states could require different
actions depending on the commanded objective, which was hidden in
`env.target_velocity`. v3 fixes this by observing the commanded target
velocity itself (slice 40–43, WORLD frame, ÷ 5.0 like `torso_lin_vel`):

- **Frame.** Commands stay WORLD-frame (`v*` is exactly the vector
  `compute_reward`'s `velocity_tracking` and `progress` project against).
  The policy also observes the torso velocity and the chest axis in world
  frame, so commanded-vs-achieved velocity is directly comparable in
  observation space with no hidden transform.
- **Curriculum conditioning.** The trainable ladder differs between stages
  ONLY through the command (`standing [0,0,0]` → `controlled [0,0.25,0]` →
  `sustained [0,0.5,0]`) and `upright_assist` (a plant-side blending of the
  policy's own PD targets toward neutral — its physical effect is observable
  through the joint response, and it is not part of the task definition).
  With the command observed, a separate task/phase identifier is redundant
  and was deliberately NOT added.
- **Stage-direction correction (measured, B3).** The B2 stage data commanded
  `+X`, but the probe `ai-playground-audit/probe_b3_yaw.py` measured that a
  policy cannot locomote along +X: the rig's ONLY rotational root dof is
  `waist_yaw` (a Z-hinge between the container and the torso) which the env
  posture-PD holds at 0 and which is OUTSIDE the 14-action space; without a
  waist command the reachable locomotion is along the chest axis —
  random-action mean |dy| = 0.94 m vs |dx| = 0.013 m, chest heading stays
  ≈90° ± 3°, waist never leaves 0 ± 0.06 rad. The scripted gait reaches +X
  only because it OVERRIDES the waist PD (heading − 90°). Stages 2–3
  therefore command the walkable chest axis (world +Y at spawn): the gait
  at heading 90 reproduces the full benchmark on that axis (0.749 m / 8 s,
  `probe_b3_axis.py`). This is a phase-DATA correction; the reward, the
  env mechanism and the B1 safety stack are untouched, and the config
  default `rl.target_velocity` (+X) keeps its documented role: the bare-env
  task pairing matching the heading-0 gait benchmark and its B2 anti-farm
  tests (`tests/test_rl_env_b2.py`).

### 1.3 Discovered physical quirk (documented, not fixed at B2)

The shipped asset embeds each foot box partly inside its shin capsule
(`g:left_foot` box at the ankle overlaps `g:left_shin` capsule radius
0.04): a persistent internal contact pair exists every step (measured:
constant dist −0.0400, efc normal-force ≈ 7.2·10⁵ N, constant across the
whole 8-s walk — it cancels internally through the ankle hinge). Two
consequences, both measured:

1. The gait controller's own `_foot_planted` reads "planted" for BOTH feet
   at 100% of sampled steps during a real 0.76-m walk — the contact-driven
   gain switching effectively rides the stance path (this is the shipped
   behaviour that produced all historical baselines).
2. An "any-contact" observation boolean would be constant (uninformative);
   the contract therefore observes the non-self support-FORCE rule above,
   which carries real stance loading (300–1000 N stance).

Fixing the asset geometry would change all historical baselines — deferred
with evidence; see `reports/m6b/B2_ENVIRONMENT.md`.

### 1.4 Finiteness and determinism (verified)

- Observations are finite on every probe run; any non-finite reading
  becomes a FAULTED window (never a raw return).
- Two `LocoEnv` instances constructed identically produce bit-identical
  observation sequences for identical action scripts (test + probe).

## 2. Action contract (14 normalized joints)

| Property | Value |
|---|---|
| Dimension | 14 (`action_layout()['action_dim']`) |
| Encoding | normalized joint-position targets, [-1, 1] |
| Mapping | `target = lo + (a + 1) * 0.5 * (hi − lo)` with lo/hi = the joint's REAL compiled range |
| Joints | the 14 in §1 joint order (includes the 3 head dofs — kept per the M6-B design decision) |
| PD gains | per-joint (kp/kd): hips 60/5, knees 60/5, ankles 25/1, spine 35/4, shoulders 25/3, elbows 10/2, **head 8/1** (explicit since B2; was the world default 15/1.5) |
| Decimation | one policy decision per `action_repeat = 10` physics substeps ⇒ **20 ms policy period**; the PD tracks every 2 ms (500 Hz) — during the window the mapped targets persist (natural hold; the world is untouched) |
| Actuation | position-setpoint PD through `PhysicsWorld.set_joint_target(..., source="neural_policy")` → `d.qfrc_applied`; the policy never touches qpos/qvel |
| Bypasses | B1 safety unchanged: non-finite/wrong-shape actions REJECTED (`ActionError`), finite out-of-range actions CLIPPED to [-1,1] with per-episode clip statistics; NaN-weighted networks are caught at the same guard |

**Conventional-PD assistance (policy-external).** Three passive posture
hinges — `waist_yaw`, `left_foot_yaw`, `right_foot_yaw` — are held by the
env with fixed conventional PD (target 0 rad; waist 50/5, foot_yaw 8/1 —
the same magnitudes the gait controller uses for these families) tagged
`source="env_posture"`. They are outside the action space and cannot be
commanded by a policy. Optional additional assistance:
`upright_assist` (0..1, curriculum-controlled) blends the policy's targets
toward neutral on spine + hips + knees through the SAME PD path — a
curriculum knob, never a teleport, never a qpos write.

Gains are part of the action contract (they shape the physical meaning of
a normalized command) and are carried in `action_layout()`, from where the
B5 `.aipg` manifest must reproduce them.

## 3. Reward and termination

`step()` returns **(obs, reward, terminated, truncated, info)** — the v2
protocol (documented migration from the v1 4-tuple). Every
reward component is built from engine measurements recorded in the
`MeasuredStep` (positions before/after the window, torso velocity, up-z,
measured effort J, recent up-z window) and summed ONCE, in
`rewards.compute_reward`. `info['components']` carries the full breakdown
per step; `episode_summary()` accumulates it per episode.

| Component | Weight (config `rl.reward`) | Definition (measured) | Anti-exploit property |
|---|---|---|---|
| `progress` | 8.0 m⁻¹ | metres advanced along the commanded heading THIS window (`(p−p_prev)·ĥ`, ĥ = v*/‖v*‖; negative/sideways → 0), CLAMPED at ‖v*‖·window_s | pays displacement, not velocity sampling; stop-and-go cannot out-earn a walk covering more metres; round trips pay the half rate; target speed caps the payout |
| `velocity_tracking` | 1.0, σ=0.1 | exp(−‖v−v*‖²/σ) on the FULL 3-D world velocity | sideways/backwards motion scores below standing with the same target (v1 semantics, kept) |
| `upright` | 0.5 | **chest-aware** per window: `min(torso up-z, chest up-z)` — the chest measure is the `spine_L2` body's world up-vector z (`xmat` col 2, same channel/scale as torso). B4 posture ask (b4_posture1, one change per Jamie's decision): pelvis up alone is blind to a chest fold because the slide-root pelvis has no pitch dof — measured b4_head1 sustained fold: torso ≈ 1.0 while chest mean 0.115 | integral = chest-aware upright duration; standing (spine 0) is bit-identical `min(1,1)=1` |
| `alive` | ±1.0 (margin 0.4) | same chest-aware posture above 0.4 → +1, else −1 | fold states (chest_up_z < 0.4) pay −1 every window, not just at termination |
| `effort` | −0.005 J⁻¹ | measured mechanical work Σ\|qfrc_applied·qvel\|·dt over the 14 policy dofs | flail costs real joules (scripted walk ≈ 3.3 J/window vs standing ≈ 0.04); the normalized command magnitude is NOT what is penalised |
| `smoothness` | −0.1 | ‖a−a_prev‖² on the clipped-encoded actions | control chatter shaping |
| `stability` | −0.2 | variance of up-z over the last 25 decisions | body jitter penalty (B4 note: mathematically ≡ 0 on this rig — the pelvis cannot pitch; repointing it at the chest axis is a deliberately SEPARATE future change, deferred to keep reward causality attributable) |
| `fall` | −5.0 once | emitted at a fall-terminated window only | falling is worse than slow progress |

**Honest assessment (not "this reward is good because the numbers look
reasonable"):** measured on the real model, sustained scripted walking
(0.795 m/s average during the ramped 8-s episode) earns ≈ +1.7/decision
against the standing equilibrium ≈ 1.58 — the margin is thin, driven by
the tight σ=0.1 kernel (the gait's average speed 0.095–0.10 m/s sits far
from the 0.5 m/s target). The standing local optimum still exists; what
progress adds is a per-metre term standing can never collect. The v2
design keeps the standing stage (stage 1) a legitimate equilibrium for
THAT phase only (target velocity [0,0,0] ⇒ tracking pays 1.0 for matched
stillness), and stage 2-3 raise the target. A σ sweep is explicitly
deferred until B3 gives a learning signal (audit §14.3 recommendation,
kept for baseline comparability).

**Termination vs truncation.** `terminated` = physical end: fall
(`agent_fallen()` debounced truth, `info['fallen']`) or FAULTED window
(`info['faulted']`, MuJoCo warning list in `info['physics_warnings']`).
`truncated` = exhausted `max_steps` decisions (default 400 = 8 s at the
20-ms decision period; `rl.max_episode_steps`). A fall and a budget
expiry in the same window report **terminated** (fall wins). FAULTED
windows carry zero reward and must be excluded from training data
(`info['faulted']` is the exclusion flag; `episode_summary()` counts them
as `faulted_windows`). There is **no survival bonus** for reaching the
horizon.

**Safety-fault handling (B1, preserved).** The world-level validation and
observation stack (input rejection, gain ceilings, per-joint torque caps
with source attribution, MuJoCo warning watchdog) is untouched; the env
adds the action-boundary guards. Fault windows are identifiable and
excluded; invalid physics steps never become successful transitions.

## 4. Curriculum

`Curriculum` + `DEFAULT_STAGES` (see module). Stage transition semantics:

- **Application** is only through `LocoEnv.set_phase`, whose semantics are
  non-cumulative BY CONSTRUCTION: gravity rewrites from the captured base
  (present-key) or stays (absent); `upright_assist`, `target_velocity` and
  `reward_coeffs` are ALWAYS rewritten (absent ⇒ env/config defaults;
  `reward_coeffs` partials apply over the frozen config defaults). The
  pre-M6 defect — gravity multiplied cumulatively every application — is
  structurally unreachable (guarded by test).
- **Transitions** are explicit and deterministic: `Curriculum.step()`
  returns True exactly on a transition; `select()/advance()` move the
  pointer explicitly; the same decision sequence gives the same
  progression (no RNG). Durations are decision counts (standing 400,
  controlled 1200, sustained 1600).
- **Stages** (mission ladder): `standing` → `controlled` → `sustained` are
  trainable today; `disturbance`, `terrain`, `navigation` are **declared
  but reserved** (`env_ready=False`): advancing into them raises with the
  reason until their env-side hooks exist. No stage pretends.

| Stage | target v* | upright_assist | Purpose |
|---|---|---|---|
| `standing` | [0, 0, 0] | 0.6 | balance; tracking pays matched stillness |
| `controlled` | [0, 0.25, 0] | 0.3 | slow walking along the walkable chest axis (§1.5) with assistance |
| `sustained` | [0, 0.5, 0] | 0.0 | the honest walking task on that axis |
| `disturbance` | — | — | RESERVED (needs a measured disturbance hook) |
| `terrain` | — | — | RESERVED (needs terrain spawn phases) |
| `navigation` | — | — | RESERVED (needs heading/waypoint sampling) |

## 5. Episode lifecycle, reset, reproducibility

- `reset(seed=None)`: rebuilds `PhysicsWorld` (project convention — no
  state leakage), re-applies the current curriculum phase state to the new
  world, re-registers PD (policy joints at 0 + posture hinges), zeroes all
  episode accounting (boundary stats, metrics, upright window, warn
  baseline).
- **Seeded initial conditions:** `reset(seed=k)` draws ONLY the start
  position from a uniform ±0.15 m box (`rl.start_pose_jitter`) via the
  world's own `setup_agent_pose` (initial-configuration class, as used by
  benchmark setup). The default spawn is deterministic; joints start
  neutral; root velocity is zeroed by the world's own spawn reset. No
  mid-episode qpos/qvel writes exist in this environment — all motion is
  the MuJoCo solver's response to PD commands.
- **Determinism (verified):** same seed → same start pose → identical
  episode metrics for identical action scripts (rel 1e-9); same-script
  consecutive episodes on one env measure identically after reset;
  `step_count`/clip counters zero between episodes; obs/action dims are
  constant for the contract version.
- **Metrics:** `episode_summary()` reports steps, sim time, path length,
  net displacement, mean speed, REWARDED and raw heading progress,
  upright fraction, min up-z, effort (J and mean W), the per-component
  reward sums, reward total, falls, faulted windows, truncation point,
  action-clipping stats, seed, and target velocity.

## 6. Baselines (M6-A measurements — immutable, re-measured for reference)

Historical references from M6-A (unchanged, never rewritten):

| Controller | Distance / 8 s |
|---|---|
| Zero action | ≈ 0.0004 m |
| Untrained policy (record's 22,926-param MLP, deterministic policy) | ≈ 0.0245 m |
| Scripted gait, speed 1.0 | ≈ 0.7613 m |

B2 re-measures them THROUGH the new env contract (`reports/m6b/
B2_ENVIRONMENT.md`): the scripted gait reproduces ≈ 0.76 m through the
decimated loop (asserted in-test to ±0.05), and zero-action remains ≈
0.001 m. The **untrained-policy** number's network class no longer exists
(the broken trainer was removed); the eval harness (later M6-B stage) will
define its untrained baseline explicitly. The scripted gait remains a
BENCHMARK, never labelled learned.

## 7. What now exists (B3) and what does NOT exist yet

- **The RL training algorithm EXISTS at B3:** a genuine actor-critic PPO
  trainer (`playground/rl/ppo.py`), documented in
  `reports/m6b/B3_PPO_LEARNING.md` and §7-equation form in the module
  docstring. It trains only from real MuJoCo rollouts through this
  contract, with seeded checkpoints under `experiments/rl/`
  (`playground/rl/checkpoint.py`; NOT the `.aipg` format) and the matched
  evaluation harness `playground/rl/eval.py` (CLI `scripts/rl_ppo.py`).
- **No portable brain files / `.aipg`** (B5); the manifest will carry
  exactly `observation_layout()`/`action_layout()`.
- **No deployment path for a neural policy into an agent** (adapter work;
  later stage).
- **No .aipg-, Studio- or dashboard-training surface claims.** The script
  CLI is the training workflow for B3; the dashboard RL tab stays
  read-only for the environment contract.
- **No neural walking claim of any kind beyond what the measured B3
  evidence records.** Scripted gait ≠ learned; the trained policy's
  quality is whatever `reports/m6b/B3_PPO_LEARNING.md` measured.

## 8. Known limitations (documented honestly)

1. The standing equilibrium survives under steady-state arithmetic (§3);
   curriculum stage-1 intentionally uses it as ITS equilibrium. Expect the
   σ sweep / progress-weight tuning to matter at B3.
2. Progress pays the half rate on round trips (per-window non-negative
   projection); a policy that reverses slowly keeps close to half rate —
   the velocity kernel is the return-leg penalty. A stricter episode-level
   heading integral is a planned B4 knob if B3's policies exploit it.
3. Effort is measured on the 14 policy dofs only; the conventional PD
   posture work is environment cost. Effort pairing is post-step
   (forward-Euler), an O(dt) approximation of instantaneous power.
4. Contact observation is the non-self support-force snapshot at the
   decision boundary (not a per-substep integral).
5. `agent_fallen` is debounced 0.3 s; a collapse window inside that settle
   is paid as normal physics until the debounce clears (audited and kept
   as the truth signal; a faster margin-based early-termination knob is a
   possible B3 addition with documented hysteresis).
6. The asset's foot-inside-shin overlap (§1.3) is evidenced and deferred.
