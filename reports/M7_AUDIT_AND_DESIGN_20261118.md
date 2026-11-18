# M7 — Foundational Movement, Gait & Transfer Learning: Audit + Design

**Date:** 2026-11-18 · **Status:** proposal (audit + design exercise per Jamie's brief — **no trainer or env code changed in this task**) · **Evidence:** `reports/M7_AUDIT_PROBE.json` (read-only probe `ai-playground-audit/m7_audit_probe.py`, one scratch `LocoEnv`, no repo writes beyond this report pair) · Repo at `main` = `cda7aba`, tree clean apart from these two files.

## 0. Scope and standing rules (from the brief)

- First M7 task = **audit and design exercise**, not trainer changes. Nothing below changes PPO, the env contract, or any run artifact.
- **B3 / B4 (b4_head1) / B4 posture1 (b4_posture1) remain untouched** as historical comparison brains. In M7 terms: they may be *read* (for transfer parentage and comparison rows), never retrained in place.
- Working rules per `[[ai-playground-workflow]]`: measured evidence only, gate-then-report at each sub-milestone, surfaces (dashboard etc.) last.

## 1. Audit — what exists today (verified against source + probe)

### 1.1 One env, one reward, one curriculum (`rl/env.py`, `rl/rewards.py`, `rl/curriculum.py`)

Verified by reading source and by probe (probe rebuilt a default env):

- **Contract `rl_env/3`:** obs `float32[43]` in 8 documented sections (`[0:14]` joint pos, `[14:28]` joint vel, `[28:31]` torso up, `[31:34]` torso vel, `[34:35]` upright margin, `[35:38]` torso forward, `[38:40]` per-foot support force, `[40:43]` **commanded target velocity v\*, world frame**); action `float32[14]` normalized joint targets → PD through `PhysicsWorld.set_joint_target`, 20 ms decisions, 500 Hz PD. Obs v3 is an **append-only** extension of v2 (which was append-only over v1) — the precedent M7's command block must follow.
- **Reward:** one pipeline, 8 components, per-window coefficients from `DEFAULTS ∪ phase.reward_coeffs` with an unknown-key `ValueError`. Chest-aware upright since b4_posture1: `min(torso_up_z, chest_up_z)` on upright/alive; progress clamped to |v\*|·window; everything engine-measured.
- **Curriculum mechanism:** frozen `Stage` records (`id/name/description/duration_decisions/phase/env_ready`) + deterministic explicit transitions (`select/advance/step`), phases applied ONLY through `LocoEnv.set_phase` (non-cumulative rewrite: gravity from captured base, targets/rewards rewritten from sources of truth). Current mission ladder: standing(400) → controlled(1200) → sustained(1600); **disturbance / terrain / navigation declared RESERVED with `env_ready=False`** — exactly M7 stages 4/6 already have home slots in the mechanism, they only need their env hooks.
- **Task conditioning already exists:** the B3 command slice `[40:43]` is a world-frame velocity command *observed by the policy* — M7's "commanded limb targets / waypoints / weight shifts" are a generalization of a pattern the contract already uses, not a new idea bolted on.
- **Trainer stage flow:** `PPOTrainer` advances the curriculum explicitly via `Curriculum.advance()` at configured `stage_budgets` (`rl/ppo.py:719-732`); final stage runs forever. New M7 ladders ride this unchanged.

### 1.2 Checkpoints, lineage, resume (`rl/checkpoint.py`, `rl_live/playback.py`)

- **`rl_checkpoint/1`:** JSON with SHA-256 content hash (tamper check refuses hand-edits), full contract fingerprints (obs section slices + joint order + action mapping/joint names/**PD gains** + dims), actor/critic weights, **optimizer states, torch RNG, trainer counters** (step/episode/iteration/stage-steps). Load is **strict**: any obs-dim, joint-order, mapping or gain mismatch raises `CheckpointError` — incompatible loads fail loudly, never silently mis-fire.
- **Resume is proven** (B3-era restoration test: weights, optimiser, RNG, counters AND curriculum stage; exercised again for real at b4_head1's pause→resume; loader line + `result.json` verified).
- **Lineage discipline already operational:** one run dir per lineage (`b3_run1`, `b4_head1`, `b4_posture1`), writes only via explicit `--out`, SHA-256 baselines on lineage artifacts, read-only checkpoint playback (`cda7aba`, rig SHA verified every start). The *convention* for immutable parents and versioned children is therefore 80% in place — what's missing is a registry that makes it a **first-class, queryable structure** rather than a convention.
- `save_checkpoint(..., extra_meta=...)` exists: parent pointers, lineage tags and eval-before records can ride **without a format bump**.

### 1.3 Evaluation (harness + probe protocol)

- **Matched-condition harness** (`rl/eval.py`): same phase/horizon/seed-base for every policy compared; held-out seed base 97,010,000 (disjoint from training space); deterministic policy eval, zero-action control, **scripted gait benchmark** (`set_gait walk/speed/heading`); metric dict with mean/std/median/min/max plus the **upright-vs-displacement breakdown** (Jamie's B3 demand — "travelled while upright" separated from "moved then collapsed").
- **Per-checkpoint replay protocol** (in practice, not in the harness): 40-ep stochastic replays at seeds 990100+, save-time stage, run at every checkpoint of b4_head1/b4_posture1 — the trajectory-table system keeps rows comparable across budgets and lineages.
- **Forgetting measurement already exists in embryo:** every sustained-trained brain has been re-evaled under the *standing* protocol (eval_standing). Its result is exactly a forgetting measurement: b4_head1 standing drift +0.4455 m scoring below zero-action; b4_posture1 +0.176 m (−61%). M7's "evaluate before and after retraining" formalizes this existing practice into a gate.

### 1.4 Physical seams (probe `M7_AUDIT_PROBE.json`)

- **Rig inventory (measured):** 14 policy hinges — `spine_L2, l/r_shoulder, l/r_elbow, head_pitch/yaw/roll, l/r_hip, l/r_knee, l/r_ankle`, all single-axis sagittal (X) ±π/2 — plus **3 passive** hinges held by env posture PD: `waist_yaw ±π`, `left/right_foot_yaw ±π`. All limb joints are **1-DOF**: arms have flexion only (no abduction), legs hip/knee/ankle pitch only. Head is the only 3-DOF chain (pitch/yaw/roll).
- **Masses:** policy-bearing subtree 37.05 kg (15 bodies), + the 10 kg root slide column (`agent` container) = **47.05 kg total**, matching the B7 mass audit. Proportions already audited (legs 33.7%, arms 11.3%, pelvis 14.9% ≈ human; chest 15% and head 3.8% under human; the 10 kg root column has no human analogue).
- **COM is directly measurable from engine state** — standing COM measured at (−0.079, 0.089, 0.936) m, inside the foot support region (support y 0.018–0.198 m). M7-S2 weight-shift tasks need no new model work: COM vs support polygon is `d.xipos ⊗ m.body_mass` arithmetic the probes already do.
- **`waist_yaw` is the rig's ONLY rotational root dof and it is currently PASSIVE** (env posture PD). This is *why* locomotion is locked to the spawn chest axis (world +Y) and why turning is impossible in the current contract. **M7 stage 5 (turning) and stage 6 (targets in different directions) require converting `waist_yaw` into a policy-action joint** — an action-contract change (14→15, append at the END of the joint order). B4-head lesson applies: waist capacity must be measured against turn loads BEFORE the unlock (it carries the 51 kg slide-column figure's whole yaw demand).
- **The scene already carries 20 freejoint blocks** (`j:block_01…20`) + ring + rock, and `setup_place` repositions an existing block (probe: block_01 (−2.5,−3.0) → (0.5,1.5,0.05)). **M7 stage 6 needs no new spawn machinery** — only target designation, target *observation*, and reward.
- **Disturbance force paths exist** in the engine: `impulse_object`, `set_object_velocity`, the `mj_applyFT`/`qfrc_applied` paths inside `PhysicsWorld._step` (combat), plus `get_state`/`set_state` snapshots for deterministic push replays. A stage-4 hook is new code but has engine precedent to model from — bounded seeded magnitude classes, per-window attribution, safety-log events (the B1 pattern), never unbounded forces.
- **Scripted gait controller** (`locomotion.py`): stand/walk/run modes, speed + heading, scheduled recovery with `recovery_complete` emitted **only after** the measured fallen-condition clears. It is M7's benchmark fixture for stage 4/5 gates and the *measured source* of push magnitudes.

## 2. Gap analysis — Jamie's six stages vs today

| Stage (Jamie's ask) | Exists today | Gap (env / obs / reward / contract impact) |
|---|---|---|
| **M7-S1 Individual limb control** | Joint pos/vel observed; PD joint targets ARE the action space; reward takes new components; assist-blend knob exists | (a) **Limb-target commands** — nothing commands a limb pose today (command slice is velocity-only); (b) **limb-tracking reward component** (error-to-commanded-pose, grouped joints); (c) gate probes: per-limb reach windows, repeatable trajectories. *Rig honesty:* "raise/lower/bend/extend" lives within the current single-axis hinges (sagittal plane); **no abduction, no wrist** — sideways arm raising needs a rig change (separate measured decision, interacts with B7 anthropometry). Gate must reject "limb moves by flopping the body": upright/COM/posture already measurable. |
| **M7-S2 Balance & weight control** | COM measurable (probe); foot support forces observed `[38:40]`; chest-aware posture reward live; upright_assist knob; stability component | (a) **weight-shift commands** (obs append: commanded COM offset/stance fraction); (b) **COM-tracking reward while upright**; (c) single-leg-support probes from existing contact channels. |
| **M7-S3 Coordinated standing** | M6 standing stage exists; standing equilibrium measured over 4,800 real windows (2.4972/window, 0 falls, chest_upz ≡ 1); posture band metrics defined (B4 contract); jittered spawn box exists | Mostly **gate definition** (head/chest/COM bands × duration × effort) — the standing machinery is the best-covered ask. |
| **M7-S4 Recovery & stability** | `disturbance` stage RESERVED in the mechanism (env_ready=False); scripted recovery schedule with measured completion; engine force paths; state snapshots for deterministic replays; B1 attribution | (a) **env-side disturbance hook** (bounded, seeded, attributed impulse/force classes on the agent, magnitudes sourced from the scripted envelope — never invented); (b) **recovery definition** (upright hold T seconds post-push, standing position re-attained); (c) gate: recovery rate over direction × magnitude grid. |
| **M7-S5 Walking gaits** | Sustained walk conditioned on observed v\* (b4_posture1: +3.5194 m/ep @ 0.485 m/s, 12/12, 0 faults); gait fixtures (h0 0.7301, h90 0.7057) as immutable comparisons; foot forces + positions observable | (a) **command ladders** for speeds / backwards / start-stop (v\* ramps 0↔0.5) — curriculum *data*, machinery unchanged; (b) **the gait phase cycle itself must come from commands+gates, never from scripting** (replaying the scripted gait is not learning — same principle as gait-as-benchmark-not-policy): sub-stages pay measured foot-alternation/placement metrics, phase-indexed foot-target *shaping* only as measured last resort; (c) **turning requires the waist_yaw action unlock** (contract 14→15) with its own capacity-then-neutral gate. |
| **M7-S6 Block-target navigation** | `navigation` stage RESERVED; 20 in-scene blocks + proven `setup_place`; world-frame + agent-frame conditioning possible; scripted move/turn/recovery as benchmark | (a) **target observation** (obs append: agent-relative target vector + distance, normalized); (b) **target sampling contract** — seeded per-episode placement via setup mechanics inside `reset` (B2 determinism: no mid-episode moves in the first gate; moving targets = a later, separately-gated variant); (c) **reward**: approach-progress-to-target + stop tolerance; (d) gate: stop-distance distribution over random directions/bearings, orientation at stop, no collisions. "Adjust as the target moves" = explicitly its own sub-gate, not smuggled into this one. |

**Cross-stage:** all obs extensions must be **append-only** (v3 stays a strict prefix); all action extensions **append at the end** of the joint order; reward components keyed and coefficient-gated so each stage's ladder sets weights through `set_phase` like today. One env, one pipeline — stages are data + measured gates, not forked trainers.

## 3. Brain reuse & retraining — the second pillar

### 3.1 Brain registry (immutable parents, versioned children)

- A **brain** = a frozen `rl_checkpoint/1` file + a **manifest**: id/version, `parent_sha256`, lineage tags, contract snapshot (the layouts already fingerprinted), rig SHA, stage/ladder record, and the **eval-before** rows taken at registration.
- Registry home proposal: tracked `brains/` at repo root (write-once; SHA verified on every load — the tamper check already does this). Run dirs stay gitignored; registry brains are durable copies the way `reports/B4P1_*.json` are.
- **"Retraining must not destroy the original brain"** is enforced by three existing mechanisms + one new one: SHA tamper check, write-once registry policy, the run-dir lineage convention, and the new **surgery loader** below. b3/b4/b4_posture1 become the first registered lineages (read-only).

### 3.2 The transfer machinery — three distinct operations (Jamie's decision, 2026-11-18)

Jamie's ruling: the brain-loading machinery must distinguish **three operations, never presented as interchangeable**, each with its own evidence and evaluation:

1. **Resume** — continue the SAME training run, restoring its full state. (Exists: `--resume`, proven by the restoration test + b4 pause→resume.) Evidence: restoration test + continuation telemetry.
2. **Continue-fork** — branch from a saved brain into a new run dir with a **compatible architecture and preserved training state** (weights, optimiser, σ, counters); no contract change. Evidence: fork-start restoration + parent-immutability check (SHA unchanged) + eval-before rows + same-protocol gate evals.
3. **Transfer-fork** — adapt a saved brain to a **changed task or contract**, with an explicit, tested mapping of compatible observations and actions (the surgery loader below). Evidence: surgery round-trip proof + parent under child task (baseline) + parent-protocol forgetting eval after + child-protocol gate evals.

Every run launcher invocation records its operation tag (`resume | continue_fork | transfer_fork`) in run metadata, and every gate/eval report names the operation that produced its policy. The surgery specifics follow:

`load_checkpoint` is strict today — every mismatch raises. Transfer needs a **documented second path**, used only when the parent's contract is a declared subset:

- **Obs surgery:** parent fingerprint must match the child's layout as a *prefix* (append-only discipline makes this check exact: same section slices, same joint order, same gains). Port the trunk's first-layer rows for the shared slice; zero-init (or orthogonal-init, decided + measured at M7-0) the new command-block channels; copy mu/critic heads and deeper rows verbatim.
- **Action surgery:** parent action joints must be an exact *prefix* of the child's (new joints appended at the end). Copy the output layer's existing columns; zero-init appended columns.
- **Round-trip proof required at implementation:** a child built from a parent must reproduce the parent's outputs on the shared obs/action slice to float tolerance before any training — the same standard as the playback seam's bit-exactness check.
- When contracts match exactly (same rig, same command block — e.g. a walking brain → gait-sub-stage): plain resume into a **new run dir** (exists, proven).
- Widths: checkpoint meta carries obs/action dims but **not hidden widths** — extend policy meta at M7 (additive field, default (64,64) for old checkpoints) so surgery validates architecture, not just dims.

### 3.3 Evaluation before/after; catastrophic forgetting

- **Per child lineage:** parent evaluated under the child's protocol *before* training (the "eval-before" row) and both parent-protocol and child-protocol evals re-run at each gate. Gate = child ≥ parent on the new task **and** child ≥ a *measured* band of the parent's own held-out numbers on the parent task (the band comes from data — e.g. today's standing-drift rows — never declared a priori).
- **If a forgetting gate fails:** mitigation options (longer budget / σ schedule; interleaved parent-task rehearsal windows) are **separate measured experiments**, not silent curriculum edits — same rule that made B4 changes legal.
- **Composition:** today's actor is one tanh-Gaussian MLP (64,64); "skills" are curriculum (ladders), not modules. The scripted skills library is the scripted analogue. M7's recommendation: prove weight-transfer composition (b4 walking brain → navigation brain) first — cheapest, most certain. Neural skill modularity (multi-head/mixture) is a later measured experiment, gated on the forgetting data.

### 3.4 B5 (.aipg) folds in

The manifest requirement ("reproduces exactly `observation_layout()`/`action_layout()`") is already satisfied by construction — `.aipg` export becomes a registry *producer*, not a new design effort.

## 4. Proposed sequencing (each sub-milestone gates before the next)

- **M7-0 Contract v4 freeze:** command-block spec (exact slices/normalisation/determinism/masking), append-only promise, surgery-loader spec, hidden-width meta; **neutral gates**: v3 fixtures bit-identical through the new path with the command block inactive (the b4_posture1 standing-invariance pattern), plus surgery round-trip test on synthetic prefixes.
- **M7-1 Limb-control stage family** (S1): new lineage, scratch training; gate = commanded-limb reach + upright hold + COM discipline. **M7-1 and M7-2 are strictly separate** (Jamie's decision): the M7-1 gate must pass and close before any M7-2 lineage starts — individual limb control is proven before skills are expected to combine into stable standing.
- **M7-2 Balance/weight-shift + standing gates** (S2, S3): ladder S2→S3; gate = COM tracking under shift commands × posture bands.
- **M7-3 Disturbance hook + recovery** (S4): hook lands first with measured magnitude classes + B1 attribution; gate = recovery grid.
- **M7-4 Gait ladder + waist unlock** (S5): waist capacity measured THEN unlocked (its own neutral gate — head-mechanics lesson); gate = speed ladder + backwards + start-stop + turning fixtures vs scripted benchmarks.
- **M7-5 Navigation** (S6): target obs + sampling + reward; **the first full brain-lineage exercise** — one fork *continues* the M7-4 walking brain (same contract), another *transfers* it via surgery or fresh-branch comparison; forgetting gate recorded.
- **M7-6 Registry + .aipg** (B5 folded): write-once registry, produce `.aipg` from registry brains; playback viewer gains lineage browsing (surface work — last, per the ordering rule).

## 5. Relations to pending/decided items

- **M6-B7 (anthropometry + gravity):** stays a separate milestone with its decided structure (Phase A at 1 g → gravity ladder → 1 g final eval with the direct-1g control). Natural insertion either after M7-2 (balance gates give the posture baseline to protect anthropometric changes against) or after M7-5; gravity needs no contract change and could also ride later M7 stages as a curriculum variant. Jamie's sequencing call.
- **M6 open issues** (gait-fixture drift note, hostile-policy qacc scope, combat-cap premise): unaffected, remain deferred per the B4 close.
- **B6 (demonstration):** the playback machinery is reusable for every M7 lineage unchanged.

## 6. Honest risks / limitations

1. **Rig DOF:** S1's "raise/lower/bend/extend" is satisfiable sagittally; "raise sideways" is not (no abduction). Rig extension = mechanical change = its own measured gate, and interacts with B7's anthropometric corrections. Decision deferred to Jamie.
2. **Turning loads:** waist unlock shifts yaw demand onto one hinge carrying the slide-column figure — capacity-vs-load measurement is a hard prerequisite (the head-mechanics lesson, directly).
3. **Capacity & wall-clock:** obs 43 → ~60+ (command block ≈ 20 dims: limb targets 14, balance 2, target 3, stop flag 1) on a (64,64) CPU MLP; M6 runs took ~4,400 s at 1.2 M steps on the *current* dims. Capacity and run-time must be measured at M7-1, not assumed. Hidden widths (64,64) may need widening — measured, per-stage.
4. **σ annealing** was still unconverged at 1.2 M (0.75–0.80) on every M6 lineage: S5/S6 ladders likely need longer budgets; set budgets from measured σ/completion, not round numbers.
5. **Command-block width frozen once:** one fixed-width block with zero-masking per mode (instead of per-stage contract bumps) keeps v3 a strict prefix forever — the design constraint M7-0 must lock.
6. **Moving targets** ("adjust direction as the target moves") are its own later gate; the first navigation gate is static random targets.
7. **Forgetting bands** are measured per lineage; no threshold is declared in advance.
8. **Reward shaping vs physics teaching:** the b4_posture1 + gravity-roadmap lesson cuts both ways — M7 asks reward only where no command/gate can ask it (e.g. COM tracking has no other signal), and every new component needs the documented-equation + baseline-contribution treatment before implementation.

## 7. What this task did NOT change

No code, no trainer, no env, no existing run dirs, no evidence files: this report + `M7_AUDIT_PROBE.json` (from a scratch world) are the only artifacts. B3/b4_head1/b4_posture1 untouched. Next step is Jamie's sequencing call (M7-0 or adjustments to the S-ladder / registry design / B7 insertion).
