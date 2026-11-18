# M7-0b — The loading law: op tags, forks, transfer surgery — implementation spec (written BEFORE code)

**Date:** 2026-11-18 · **Basis:** the approved M7-0 freeze v2 §8 + Jamie's directives (the kd/ranges fingerprint extension is a **mandatory transfer-loader acceptance requirement**, not optional hardening). Keep `9889ffc` as the completed M7-0a checkpoint; this is the next change set. No training is started in this stage; B3 / b4_head1 / b4_posture1 untouched.

## 1. The three operations (launcher-enforced, never interchangeable)

| Op | Mechanism | Contract | Optimiser/σ/counters | Evidence recorded |
|---|---|---|---|---|
| `fresh` | new lineage, no brain load | live | fresh | op tag only |
| `resume` | continue the SAME run (same run dir) | exact (strict loader) | restored in full (Adam, torch RNG, counters, curriculum stage) — the existing resume path, unchanged | continuation telemetry |
| `continue_fork` | branch from a saved brain into a NEW run dir | exact (strict loader raises on the slightest difference) | restored in full; run-local bookkeeping (progress/config/result) fresh | fork-start restoration + **parent SHA re-verified at the first checkpoint** + parent SHA in every checkpoint |
| `transfer_fork` | adapt a brain whose contract is a **declared prefix subset** of the live contract | subset (surgery loader only) | fresh optimisers (new shapes); σ = parent per-dim values on the shared slice + appended dim at `log_std_init` (D5-amended); counters restart at 0 (a new child lineage) | surgery record (parent sha, ported rows, appended entries) + round-trip proof + parent SHA in every checkpoint |

**Launcher refusals (loud, before any training):** `resume` whose target dir differs from the parent checkpoint's dir (that ambiguity IS a fork — refuse with guidance); `continue_fork`/`transfer_fork` writing into the parent's run dir (parent-lineage pollution — refuse); `continue_fork`/`transfer_fork` without `--fork-from`; `transfer_fork` whose parent is not a declared prefix subset (TransferError → CLI exit 4); `fresh` given `--resume`/`--fork-from`. **The surgery path is unreachable without the `transfer_fork` tag** and continue-fork refuses on any fingerprint difference (the extended strict check).

## 2. The MANDATORY loader fingerprint extension (`checkpoint._fingerprint_diff`)

The diff is extended from (dims, section slices, joint order, mapping + joint names + kp) to ALSO compare:

1. **kd values** per action joint (`action_layout()['joints'][i]['kd']`);
2. **joint ranges** per action joint (`range_rad`) AND the observation-side `joint_ranges_rad` dict;
3. **`contract_version`** explicitly.

Two brains whose dims/order match but whose control mechanics differ (kd, ranges) are NOT compatible and can no longer load — matching Jamie's acceptance requirement verbatim.

**Backward-compatibility verified against the historical lineages (2026-11-18, read-only):** b3_run1 / b4_head1 / b4_posture1 `ckpt_final.json` already carry per-joint `kp`/`kd`/`range_rad` AND obs `joint_order`/`joint_ranges_rad` (the B2-era layout included them), so the extended diff is a pure tightening for every existing brain — nothing old stops loading. Dims and `contract_version` are compared **only when both sides carry them**: the envless loader path now derives `contract_version` from the layouts' `layout_version` stamps (before this fix a version disagreement could not reach the diff on that path); real loads always fill dims + version, so the leniency exists only for hand-built diff-test dicts.

## 3. Surgery loader (`playground/rl/transfer.py`, the documented second path)

`read` + **hash/tamper verification of the FILE itself** (format tag + `content_sha256`, the strict integrity path) but the fingerprint check is REPLACED by a **declared prefix subset check**:

- stored contract_version < live (no downgrade);
- stored observation sections EXACTLY equal the live sections' first-k entries (names + slices); obs/action dims = the stored layouts' dims; obs `joint_order` equal; action `joints` equal as an exact prefix of the live action list **including kp, kd AND range_rad**; mapping string equal; normalisation keys present and equal.

Port (v3→v4 the currently declared pair; the mechanism is generic prefix-subset):

- **Actor:** `trunk.0.weight` — the OBS growth is the weight's INPUT (column) dim: copy columns `[0:obs_parent]`, **zero-init columns `[obs_parent:obs_child]`**; deeper trunk layers + biases copied; `mu_head.weight` is `nn.Linear(hidden → action_dim)` so the APPENDED action is the weight's ROW dim — copy the first `act_parent` rows, **zero the appended rows** (the first cut ported columns — the wrong direction — and the round-trip proof caught it before any fork ran); `mu_head.bias` same row rule; `log_std` — **copy per-dim `[0:act_parent]`** (D5-amended: the parent's learned σ is preserved, not re-initialised) + append one entry = `log_std_init`. `mu_head.*` and `log_std` are ported ONLY by these explicit blocks (the generic loop never touches them).
- **Critic:** input-layer rows same rule; everything else copied.
- **Round-trip proof** (run inside the surgery, results in the surgery record): on a random probe obs batch — child deterministic outputs on the shared slice == parent's to 1e−6 (the appended waist column is zero ⇒ μ_waist = 0 ⇒ deterministic waist action = tanh(0) = 0), log_std exact-equal on the shared slice, and the child critic value == parent's to 1e−5. Distribution-level: the fork-moment ACTION DISTRIBUTION on the shared slice is the parent's.
- Hidden widths: taken from the parent policy meta when present, else (64,64) — the meta extension itself is recorded in `policy.hidden` when the launcher saves v4 checkpoints (additive; old checkpoints default). The derivation is `in-dims[1:] + last out-dim` — the first cut (`widths[1:]`) dropped the LAST hidden width, turning (64,64) into (64,) and would have made the surgery rebuild a one-hidden-layer child for two-hidden-layer M7-era parents; caught by the tests, fixed in both `checkpoint._actor_hidden` and `transfer._actor_hidden_widths`.

**Findings recorded by the M7-0b test run (2026-11-18, all fixed before the gate):** the round-trip probe is row-by-row (`deterministic_action` is a single-row API), and the probe seeds its own RNG but RESTORES the caller's torch RNG state on the way out so a launcher transfer_fork keeps its training stream; the critic's port mutates the live parameters directly (its `state_dict()` returns references — the actor's returns clones, which is why the actor needs the closing `load_state_dict(sd)`); the surgery record additionally verifies the observation `normalisation` block as a prefix (spec'd here, was unimplemented in the first cut).

## 4. Launcher wiring (`scripts/rl_ppo.py train`)

`--operation {fresh, resume, continue_fork, transfer_fork}` + `--fork-from PATH`. Defaults: `--resume` present ⇒ resume (unchanged behaviour for the historical pause/resume contract); else fresh. Every checkpoint's `extra_meta` and the run's `config.json`/`result.json` carry `operation` and, for forks, `run_parent {checkpoint, sha256, run, obs_dim, action_dim, contract_version}`. Transfer-fork runs stand up the trainer from the surgery actor/critic with curriculum standing (stage 0) and `total_steps` counted fresh; continue-fork uses the exact resume restore into the new dir.

## 5. Tests (`tests/test_rl_m7_loading_law.py`)

1. Extended fingerprint diff: two fingerprints differing ONLY in kd → CheckpointError; ONLY in a joint range → CheckpointError; differing `contract_version` → CheckpointError; identical → loads.
2. Surgery round-trip: a v3-mode env parent (random actor/critic, strict-saved) transferred into a v4 env: shared-slice outputs equal (1e−6), log_std exact on `[0:14]`, appended `log_std[14] == log_std_init`, waist μ-column zero ⇒ deterministic waist action 0, critic value equal (1e−5).
3. Surgery refusal: a parent whose stored sections are not a prefix (v2-style) → TransferError; a stored contract_version ≥ live → refuse (TransferError); a fingerprint-mismatch continue-fork attempt (a v3 parent under continue_fork on a v4 env) → CheckpointError (the extended strict check, incl. the kd/ranges additions). Both failure directions are proven at the LAUNCHER (see item 4): surgery refuses the non-prefix parent on the transfer path; the strict loader refuses the same parent on the continue path.
4. Launcher law (all through the REAL `cmd_train` arg flow, tiny 512-step single-iteration budgets): `continue_fork`/`transfer_fork` require `--fork-from` (PPOError); an unknown operation name is refused; `fresh` given `--resume`/`--fork-from` is refused; a fork writing into the parent's run dir is refused; a fork target that already holds a run is refused; `resume` into a different dir is refused; a fresh run's checkpoints/result carry `operation: fresh`; a continue-fork's checkpoints carry `operation: continue_fork` + parent sha + preserved counters; **the parent file is byte-unchanged at every checkpoint** (sha compared); a continue-fork from a v3 parent under a live v4 env is REFUSED by the strict loader (CheckpointError, nothing written — the both-directions proof); transfer_fork through the launcher records the pair (3, 4), the round-trip proof, `log_std_appended`, starts at stage 0 with fresh counters; **the parent-immutability watchdog FIRES** (monkeypatched divergent parent sha → PPOError "PARENT LINEAGE CHANGED" at the checkpoint write, no poisoned checkpoint, no result.json, progress written).
5. M7-0a companion rides this commit: `test_policy_path_overwrites_rogue_waist_write` (the B1 inverse semantic,Jamie's scrutiny point 1 — the structural waist-move regression).

## 6. What this stage does NOT do

No training run starts (forks are gated at the launcher; the gates prove the machinery, not a trained brain); no curriculum ladder (M7-1); no .aipg/registry (M7-6); historical lineages stay read-only. After the gate: full suite ×2 + commit, report, await Jamie.
