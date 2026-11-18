# M6-B / M7-0b — THE LOADING LAW (op tags, forks, transfer surgery)

**Date:** 2026-11-18 · **Basis:** approved M7-0 freeze v2 §8 + Jamie's directives (the kd/joint-ranges/contract-version fingerprint extension is a **mandatory acceptance requirement**; op tags are never interchangeable; keep `9889ffc` as the completed M7-0a checkpoint). Change spec written BEFORE code: `reports/M7_0B_LOADING_LAW_CHANGE.md` (implemented to the spec, with the corrections below recorded as the spec's as-implemented notes).

## 1. What this stage implements

**The three loading operations** (`scripts/rl_ppo.py train --operation {fresh, resume, continue_fork, transfer_fork}` + `--fork-from`), resolved and refused BEFORE any training (`_resolve_operation`):

| Op | Mechanism | Contract | Optimiser/σ/counters | Recorded |
|---|---|---|---|---|
| `fresh` | new lineage, no brain load | live (v4, 68/15) | fresh | op tag in every ckpt + config.json + result.json |
| `resume` | same run, same dir (unchanged historical pause/resume path) | strict loader | restored in full | `resumed from …: step N` |
| `continue_fork` | exact restore into a NEW dir (strict loader — refuses ANY difference) | exact | restored in full; run-local bookkeeping fresh | fork record + parent sha in EVERY checkpoint |
| `transfer_fork` | declared prefix-subset surgery (the ONLY path to `transfer.surgery_transfer`) | subset (version strictly below live, sections/order/ranges/normalisation/kp/kd/range_rad as exact prefix) | fresh optimisers; σ preserved per-dim on the shared slice + appended at `log_std_init` (D5-amended); counters restart at 0; curriculum stage 0 | surgery record (pair, parent sha, ported sites, zero-init list, round-trip proof) |

**Refusals (loud, pre-training):** fork op without `--fork-from`; unknown op; `fresh` with parent pointers; a fork writing into the parent's run dir; a fork target already holding a run; `resume` into a different dir (that ambiguity IS a fork — refused with guidance); transfer on a non-prefix parent (TransferError, exits 4 in the CLI); the strict loader refusing ANY fingerprint difference on resume/continue_fork.

**The MANDATORY fingerprint extension** (`checkpoint._fingerprint_diff`): the strict diff now also covers **kd per action joint**, **range_rad per action joint**, the **obs joint_ranges_rad dict**, and **contract_version**; every refusal names the field (`action joint i kd differs (…)`). Two brains with matching dims but differing control mechanics can no longer load.

**The parent-immutability watchdog:** every fork re-reads the parent file's SHA-256 at EVERY checkpoint write and aborts (PPOError "PARENT LINEAGE CHANGED") before any poisoned checkpoint lands.

## 2. Findings from making the first cut test-proven (all fixed before the gate)

The pause left transfer.py/launcher flow untested with three known test bugs; the test run surfaced, and this stage fixed, seven more — each is a defect that would have shipped silently:

1. **`pc_sd` was an undefined name** in the critic port loop (first execution path — the launcher could never have run transfer_fork).
2. **The surgery's mu_head port was the WRONG DIRECTION**: `mu_head` is `nn.Linear(hidden → action_dim)`, so the appended action is the weight's ROW dim; the first cut ported columns (the input dim) and crashed the round-trip. Fixed to row-port; the change doc's wording corrected ("columns" → rows for the output head, columns for the input growth).
3. **The generic weight loop crashed on 1-D `log_std`** (indexed `c_shape[1]` on a `(15,)` tuple) and row-ported `mu_head` uselessly before the explicit block; both keys now ported ONLY by the explicit law blocks.
4. **`_actor_hidden` dropped the LAST hidden width** (`(64, 64)` → `(64,)`): the surgery would have rebuilt a one-hidden-layer child for two-hidden-layer M7-era parents. Fixed in both `checkpoint._actor_hidden` and `transfer._actor_hidden_widths` (`in-dims[1:] + last out-dim`).
5. **The envless loader could not see `contract_version` at all** — a version disagreement never reached the diff on the fingerprint-dict path. Fixed by deriving it from the layouts' `layout_version` stamps (save + load). Dims and version compare **only when both sides carry them**; the strict loader always fills them, so the leniency touches only hand-built diff-test dicts.
6. **`deterministic_action` is a single-row API** — the round-trip probe now runs row-by-row (the first cut flattened a `(16, dim)` batch into one garbage row).
7. **The internal probe reseeds the global torch RNG** — now save/restore around the probe so a launcher transfer_fork keeps its training stream; the critic's port mutates the live parameters directly while the actor's `state_dict()` returns clones (documented in-module; the actor needs the closing `load_state_dict`).

**Backward compatibility measured, not assumed:** b3_run1 / b4_head1 / b4_posture1 checkpoints already record per-joint `kp`/`kd`/`range_rad` + obs `joint_order`/`joint_ranges_rad` (B2-era layout), `policy.hidden` absent is handled (widths derived from the weights) — the extension is a pure tightening; every historical brain still loads strict (suite regression below).

## 3. Tests executed + actual results

- `tests/test_rl_m7_loading_law.py` (9 tests): fingerprint extension (kd/range/version all refuse, identical loads); field-named refusals; v3→v4 surgery round-trip (non-vacuous: parent σ `linspace(-1.2, 0.4)` NOT all-zeros, preserved exactly, waist σ = `log_std_init`, appended waist action exact 0, μ-column rows zero-recorded, hidden [64, 64]); surgery refusals (non-prefix sections, equal version, shared-slice kd + range mechanics, normalisation drift); launcher refusals (6 cases); fresh tags; continue-fork through the real launcher (tags, parent byte-unchanged, counters CONTINUED not restarted); **both failure directions**: continue_fork refuses the v3 parent loudly (strict loader) AND the kd-only fingerprint refusal (unit), surgery unreachable without the transfer tag; transfer-fork through the real launcher into a v4 child (pair [3, 4], round-trip passed, stage 0, fresh counters); **watchdog FIRES** (divergent parent sha ⇒ PPOError at the write, no poisoned ckpt_final, no result.json, progress written).
  → **`ai-playground-audit/m7b_battery_loaderlaw.txt` exit 0 (9 passed)**.
- Batteries: `tests/test_rl_contract_v4.py + test_rl_core.py + tests/test_safety_b1.py` (incl. the B1 inverse-semantics companion `test_policy_path_overwrites_rogue_waist_write`, which RIDES this commit) → **`ai-playground-audit/m7b_battery_core_b1.txt` exit 0 (55 passed)**.
- Full suite (SINGLE gate at Jamie's instruction 2026-11-18 — "skip gate 2 if it's just a repeat of gate 1"): **792 collected = 779 (M7-0a gate) + 9 new loading-law tests + 3 skipped + 1 B1 companion — 789 passed / 0 failed / 3 skipped, exit 0** (`ai-playground-audit/m7b_fullsuite_gate1.txt`; the pytest one-line summary again did not flush to the redirect — the documented gotcha — so the counts are confirmed by the 792-test `--collect-only` total (789 + 3 skipped = 792 ✓, the dot rows sum to 792 with 3 skipped) plus exit 0). Health: `{"healthy": true, "failed": []}` exit 0 (`ai-playground-audit/m7b_health.txt`). Lineage SHAs re-verified byte-identical AFTER the gate (all five: 7c307aee / 20b182f4 / c066bd55 / 1072edea / e81702ec).

## 4. Probes performed

- Read-only contract-field audit of the three historical lineage finals (the backward-compatibility check above).
- Lineage no-write check (SHA-256, this stage's runs vs recorded baselines; pre-gate AND post-gate): b3_run1 `ckpt_final` `7c307aee…` ✓ unchanged; b4_head1 `ckpt_000200704` `20b182f4…` ✓, `ckpt_000401408` `c066bd55…` ✓ (the baseline `c066bd55` is the 401,408 milestone ckpt — recorded as the pause-era tamper baseline, not the final), and its `ckpt_final` `1072edea…` recorded NOW as this gate's working baseline (no prior sha recorded; the M7-0a-era check keyed on the milestone file); b4_posture1 `ckpt_final` `e81702ec…` ✓ unchanged.
- The four tiny launcher runs in the tests exercise the REAL cmd_train flow (fresh → continue_fork, fresh v3-saved → transfer_fork) — the fork branches and watchdog ran end-to-end for the first time.

## 5. Outstanding limitations

- The surgery was proven against a STRICT-SAVED SYNTHETIC v3 parent (the same save path the lineages use, same new-format fields). No historical checkpoint (b3_run1/b4_head1/b4_posture1, all v3) was surgery-loaded in this gate — by design that first happens at the first real transfer-fork lineage (M7-5 nav per the freeze, or earlier if sequencing says otherwise); the machinery is version-tested, not brain-tested.
- The externally locked file `experiments/rl/b4_head1/baselines_sustained.json` (read denied to this session by an unknown process's open handle; ACL grants jamie FullControl; pre-existing, mtime 2026-11-18 11:06) — not readable in the no-write sweep; it is outside every write surface of this stage and untouched. Stopping whichever process holds it is Jamie's call. The idle duplicate playback process (PID 7368, yesterday's leftover, no port) is also still up — same standing status.
- No M7-1 training started (Jamie's rule): this stage gates LOADING LAW only; limbs come next and await approval.
