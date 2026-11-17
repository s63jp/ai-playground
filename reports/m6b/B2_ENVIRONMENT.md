# M6-B Stage B2 — Consolidated RL Environment (evidence report)

**Date:** 2026-11-17 · **Baseline:** `d90a54d` (B1) preserved · **Branch:** `main`
**Scope:** environment infrastructure only — one authoritative env, reward, curriculum,
lifecycle. No trainer, no PPO, no learning claims (B3+).

## 1. Re-audit result (current source verified, not assumed)

The M6-A audit (§2.2, §16, §17) was re-verified against the current source tree:

| Audit finding | Status at B2 re-audit | Action taken |
|---|---|---|
| `rl/train.py` `compute_reward` has no `return` → train() crashes at step 0 | CONFIRMED still true | trainer DELETED (documented migration) |
| `train.py` regresses policy-output mean to reward scalar (not RL), γ unused | CONFIRMED | same deletion |
| `rl/curriculum.py` `Curriculum.progress` uses non-existent `PhysicsWorld.opt` attr; cumulative gravity re-multiply | CONFIRMED | file rewritten (§4); mechanism replaced |
| Env obs = raw 35-dim v1; reward reads obs slices; no decimation; no contact/heading obs | CONFIRMED | contract v2 (§2-§3) |
| M6-A baselines reproduce (walk 0.7613 m / 8 s; zero action ≈ 0.0004 m horizontal; untrained 0.0245 m) | RE-MEASURED pre-implementation through the v1 env: walk 0.761347 m — bit-faithful | preserved immutable (§6) |
| Scripted-walk runs `set_gait` → gait overrides env targets per substep (ordering) | CONFIRMED preserved | decimation keeps env-mapping-then-substeps ordering: gait re-applies targets every substep (benchmark invariance measured, §6) |

Two findings recorded during the re-audit probes that the audit did NOT contain:

1. **Foot–shin self-collision quirk (measured, deferred):** the shipped asset embeds
   each foot box partly inside its shin capsule; a persistent internal contact pair
   exists at every step (constant dist −0.0400, efc normal ≈ 7.2·10⁵ N, unchanged
   across the whole walk; `g:left_foot` box `pos="0 0.02 -0.03"` overlaps the
   `g:left_shin` capsule radius 0.04 at the ankle). Consequences: the gait's
   `_foot_planted` reads "planted" for BOTH feet at 100% of steps of a real walk, and
   an any-contact observation boolean would be constant. The v2 contract therefore
   observes non-self ground-SUPPORT FORCE (§3). Fixing the asset would change every
   historical baseline → deferred with evidence; flagged for a later stage.
2. **`max_steps` semantic (v1)** counted raw 2-ms steps (2-s episodes at default 1000),
   not decisions. v2 counts decisions (400 = 8 s default), documented in the
   contract and config.

## 2. What changed (files)

| File | Change |
|---|---|
| `playground/rl/env.py` | rewritten as the authoritative env: obs v2 (40-dim, normalised), decimation (20 ms), measured-input reward handoff, 5-tuple protocol, seeded reset, posture-PD hinges, episode metrics, machine-readable layouts; B1 guards preserved verbatim |
| `playground/rl/rewards.py` | the ONE reward pipeline: `MeasuredStep` (engine measurements) + `compute_reward → (total, components)`; unit-broken `joint_limit` dropped; `velocity_tracking/upright/alive/smoothness` semantics kept; new measured `progress/effort/stability/fall` |
| `playground/rl/curriculum.py` | rewritten: deterministic explicit stage sequencer (`DEFAULT_STAGES` 1-6, stages 4-6 declared RESERVED with `env_ready=false`), applies ONLY through `env.set_phase` |
| `playground/rl/train.py` | DELETED (broken duplicate trainer + duplicate reward; see migration note below) |
| `config/default.yaml` | new `rl:` block (contract version, decimation, horizon, normalisation scales, reward weights, jitter) |
| `scripts/health_check.py` | module list: `playground.rl.train` → `playground.rl.{rewards,curriculum}` |
| `tests/test_mode_integration.py` | integration seam updated: `LocoEnv` + `Curriculum` callable (was `train(cfg)`) |
| `tests/test_rl_core.py` | migrated to the v2 contract (component semantics on measured inputs) |
| `tests/test_safety_b1.py` | 5 mechanical migrations: 5-tuple unpacking; poisoned-NN test uses a local minimal torch module (the deleted `Policy` class is replaced inside the test — the B1 contract under test is unchanged) |
| `tests/test_rl_env_b2.py` | NEW: 26-test B2 gate battery |
| `tests/test_dashboard_rl.py` | NEW: 5 dashboard endpoint tests |
| `playground/dashboard/server.py` | +3 read-only endpoints: `/api/rl/contract`, `/api/rl/curriculum`, `/api/rl/probe` (bounded, single-flight) |
| `playground/dashboard/static/index.html` + `app.js` | new "RL environment" tab: contract tables, curriculum table, one bounded measured probe leg with honest labelling (no trainer, scripted gait ≠ learned) |
| `docs/RL_ENVIRONMENT.md` | NEW: the authoritative contract document (obs/action schema tables, reward + termination semantics, curriculum, lifecycle, limitations) |
| `README.md` | RL bullet truth-aligned (no checkpoints/CLI/trainer claims) |

**Migration of `rl/train.py` removal (documented):** the module was reachable only by
import-checks (`scripts/health_check.py`, `tests/test_mode_integration.py:39-40`) —
never executed by any session, CLI, dashboard or test-with-execution (audit §2.2 §16.7).
Its reward copy disagreed with the env's (6 components incl. unit-broken `joint_limit`
vs the env's 4). Removing it leaves `playground.rl.train` unimportable by design; both
import sites were migrated to the real integration seam. B3 introduces the genuine
trainer (`rl/ppo.py` per the audit plan).

## 3. Observation contract (v2, 40-dim) — highlights

Full table in `docs/RL_ENVIRONMENT.md`; machine-readable via
`LocoEnv.observation_layout()` (the .aipg manifest source). Measured during the
re-audit probe and carried into the contract:

- chest/heading axis = torso `xmat` **column 1** (local +Y in world; during a
  heading-h walk it rotates to `(cos h, sin h)` — probe-verified).
- joint-velocity scale 6.0 rad/s: measured walk envelope knee 4.83 rad/s (×1.24
  margin).
- foot contact = non-self ground-support force (‖nz‖>0.7), measured during the walk
  as 0–≈2.0 kN (left), 0–≈5.9 kN (right, landing transients), means ≈0.25 kN — a
  real stance signal where the any-contact boolean would be constant (quirk §1).
- absolute root position is deliberately NOT observed (documented; seeded reset
  moves the world pose without changing the first observation).
- determinism: two identically-constructed envs → bit-identical obs sequences
  (`max_obs_abs_diff = 0.0`), in-test and in-probe.

## 4. Action contract (14) — highlights

Mapping `lo + (a+1)·0.5·(hi−lo)` into real ranges; PD at 500 Hz; policy at 50 Hz
(`action_repeat=10`, 20 ms — audit recommendation; "natural hold" between decisions).
Head joints stay in the action set (design decision kept) with explicit gains 8/1
(were the world default 15/1.5). Passive hinges (`waist_yaw`, both `foot_yaw`) are
conventional PD registered by the env (50/5, 8/1 — gait-consistent magnitudes),
tagged `source="env_posture"`, outside the policy's reach. B1 enforcement verified
intact by the migrated B1 suite (§7).

## 5. Reward — measured behaviour (probe tables)

All numbers from `ai-playground-audit/probe_b2_env_results.json` (real MuJoCo,
400 decisions = 8 s, seed 20261117 where noted; components = per-episode sums).

| Profile | net disp (m) | path (m) | progress raw / PAID (m) | effort (W) | reward total | falls/faults |
|---|---|---|---|---|---|---|
| zero action | 0.0100 (≈ standing z-settle; horizontal ≈ 0.0004 as historical) | 0.021 | 0.000 / 0.000 | 2.0 | 632.7 | 0 / 0 |
| sustained walk 1.0 m/s | 0.756 | 0.934 | 0.787 / 0.787 | 165.9 | 685.4 | 0 / 0 |
| bursty walk (2 s on / 2 s off ×2) | 0.524 | 1.508 | 0.290 / 0.290 | 68.0 | 641.7 | 0 / 0 |
| round trip (2 s +X, 6 s −X) | 0.552 | 1.135 | 0.199 / 0.199 (half-rate behaviour: return leg paid ~0) | 163.9 | 625.7 | 0 / 0 |

Reading (anti-farm, measured):

- **Stop-and-go cannot out-earn distance:** the bursty profile travelled MORE path
  (1.508 m) yet earned less progress (0.290 vs 0.787 m) and a lower total (641.7 vs
  685.4) than the sustained walk — the v1 velocity-sampling exploit (a duty-cycler
  beating standing without travelling, audit D3.3) is structurally dead.
- **Round trips pay the half rate:** 1.135 m of travel on a return walk paid 0.199 m
  of progress (forward legs only); repeated movement cannot inflate the progress term
  beyond the straight-line rate.
- **Standing farm:** settled standing pays alive 1.0 + upright 0.5 + tracking
  exp(−‖v*‖²/σ) ≈ 0.082/decision ≈ 1.58/decision — the standing equilibrium is
  DOCUMENTED as still existing (honest); it cannot collect progress. Its intentional
  use is stage 1 of the curriculum (target [0,0,0] makes matched stillness the
  legitimate optimum of that phase).
- **Velocity jiggle:** action chatter at the encoding extremes (velocity noise, no net
  travel) pays zero progress; unit+real tests cover it.

Reward weights were set from measured envelopes, not vibes: progress 8.0 m⁻¹ (the
gait's 0.787 m of real progress pays ≈ 6.3/episode); effort −0.005 J⁻¹ (walking ~3.3
J/decision → ≈ −0.017/decision — a ≈6% shave on the tracking gain, an order below the
progress gain, and standing pays ≈ 0.0004); σ stays 0.1 for baseline comparability with
the σ-sweep explicitly deferred to B3 (audit §14.3). **The honest standing-vs-walking
arithmetic:** sustained scripted walk ≈ +1.71/decision vs standing 1.58 — a thin margin;
the reward CAN pay walking (2.5 at exact target speed) but the shipped gait's average
0.095 m/s sits far from the 0.5 target. No reward-quality claim beyond this is made;
the training-time success metric stays engine-measured displacement (HVE rule).

Note the clamp: no leg in the probe ever hit the per-window progress cap
(|v*|·dt = 0.01 m) because the shipped gait tops at ≈0.10 m/s; the clamp is exercised
by unit tests with out-of-window displacements (the anti-teleport bound).

## 6. Baselines (immutable, re-measured through the new env for reference)

| Measurement | Historical (M6-A, immutable) | Through B2 env | Note |
|---|---|---|---|
| Scripted gait, 8 s | 0.7613 m | 0.756–0.761 m (test asserts 0.761±0.05; probe 0.7560 with a jittered start pose; unjittered ≈ exact) | decimation keeps per-substep gait ordering → physics unchanged |
| Zero action, 8 s | 0.0004 m | 0.0004 m horizontal (3-D net 0.0100 includes the standing z-settle) | posture-PD addition unchanged the standing result within rounding |
| Untrained policy | 0.0245 m | not re-measured — its network class (deleted `Policy`) no longer exists; the eval harness defines its untrained baseline explicitly in a later stage | honest: no silent redefinition |
| Standing reward/step | 1.582/dec | 1.579–1.582/dec (with v2 component arithmetic; effort 0.0004 deducted) | component sums observable |

## 7. Curriculum (measured)

Probe run (random 0.3-scale actions): `standing` (400 decisions) → transition →
`controlled` (1200) → `sustained`; gravity constant −9.81 across all phase
applications (the pre-M6 cumulative-multiply defect is structurally unreachable);
transitions exactly at the budget boundaries; `snapshot()` records the full ladder.
Reserved stages raise a clear error on `select`/`advance` (in-suite tests). Curriculum
state persists across `env.reset()` (per-design, documented) and is re-applied to each
rebuilt world.

## 8. Episode lifecycle (measured)

- `reset(seed)`: same seed → same start pose (rel 1e-12) → identical episode metrics
  for identical action scripts (rel 1e-9); different seeds → different poses within
  the ±0.15 m jitter box; unseeded reset deterministic (bit-equal obs).
- Episode isolation: counters zeroed by reset (`steps`, clip stats, faulted windows,
  metrics); consecutive same-script episodes measure identically; no world state
  leaks (fresh `PhysicsWorld` per episode, project convention).
- Truncation at exactly `max_steps` decisions; terminated-vs-truncated split
  (fall/fault → `terminated`; budget → `truncated`).
- FAULTED windows: terminated + zero reward + `info['faulted']`+
  `physics_warnings`, counted in `episode_summary()['faulted_windows']` (B1
  semantics on the v2 protocol, both unit-proven and exercised by the migrated B1
  suite).

## 9. Tests

- `tests/test_rl_env_b2.py` — 28 tests (obs/action contract, mapping, passive hinges,
  reward components, coefficient-phase semantics, farming resistance on real physics,
  truncation, fault propagation, fall termination with the penalty in components,
  reset/seed reproducibility, episode isolation, curriculum transitions/non-cumulativity/
  reserved-stage errors, gait benchmark horizon, decimation hold).
- `tests/test_rl_core.py` — 10 tests (migrated core semantics + real anti-exploit).
- `tests/test_safety_b1.py` — 29 tests (migrated mechanically; all B1 contracts green).
- `tests/test_dashboard_rl.py` — 5 tests (contract, curriculum, bounded probes, rejects).
- `tests/test_mode_integration.py` — migrated seam.
- **Full suite (gate):** see §11.

## 10. Dashboard (added at B2 close)

Read-only "RL environment" tab (`#rl`) + 3 endpoints. It exposes exactly what is
true today: the v2 contract, the 6-stage ladder (3 ready, 3 reserved), and ONE
bounded measured probe leg (zero-action or scripted-gait; server-bounded ≤ 10 s,
single-flight). Verified live over real HTTP: contract served (v2/40/14), curriculum
table, gait leg 0.153 m / 2 s, zero leg clean, invalid inputs → 400. The tab states on
its face that no trainer exists and scripted-gait legs are benchmarks, not learned
policies. Node syntax check on `app.js` passed; all element ids cross-checked.

Deferred intentionally (later M6-B stages): no RL training UI, no .aipg import/export,
no brain assignment — the Studio belongs to M6-C.

## 11. B2 gate result

Gate run on the frozen tree after the dashboard step, verified twice:

- **Full suite (project runner `bash run_tests.sh`): Passed 681 / Failed 0 /
  Skipped 1 — Total 682, runner exit code 0** — `ai-playground-audit/fullsuite_b2_summary.txt`.
- **Full suite (direct `pytest tests -q -rs`): exit 0**, collection **682 tests**;
  the sole skip is named — `tests/test_ml_training.py:320 "Real dataset not
  available"` (the gitignored dataset artifact from an earlier phase); the two LM
  Studio integration tests RAN and PASSED this time.
- Arithmetic: B1 gate 644 collected + 28 (new B2 battery) + 5 (dashboard RL) +
  5 (net test_rl_core migration delta, 5→10) = **682**.
- Health check: included in the runner's run (exit 0 overall).

- Probes executed: `probe_b2_audit.py` (re-audit facts: joint order/ranges, forward
  axis, velocity/effort envelopes, determinism, walk reproduction),
  `probe_b2_env.py` (profiles table, contacts, curriculum, contract dump). Results in
  `ai-playground-audit/probe_b2_audit_results.json` and `probe_b2_env_results.json`.

## 12. Remaining limitations (honest)

1. No learning signal yet — the standing equilibrium persists in the reward
   arithmetic and nothing in this stage trains (B3).
2. progress pays the half rate on round trips and the per-window clamp only bounds
   sub-window bursts; episode-level heading integrals stay a B4 knob if exploited.
3. effort is a post-step forward-Euler pairing over the 14 policy dofs (O(dt)
   approximation); posture-hinge work is environment cost, not the policy's book.
4. contact observation is a decision-boundary snapshot of non-self support force.
5. `agent_fallen` is debounced 0.3 s — the honest truth signal kept from the audit;
   a faster margin-based early-termination knob is deferred to B3 with hysteresis
   documented.
6. The asset's foot-inside-shin overlap (§1, measured) is deferred: fixing it would
   alter every historical baseline. Evidence recorded; a future stage may correct the
   geometry with a full re-baseline.
7. The dashboard probe is synchronous (bounded ≤ 10 s sim) — a job-style async runner
   would belong with the B3 trainer surface.

B2 gate **closes** on 2026-11-17. Next: B3 (PPO trainer) — STOPPED, awaiting explicit
approval; no B3 work has been started.
