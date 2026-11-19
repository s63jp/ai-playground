# ML lab interface contract + work-package ownership (parent-owned, frozen for the sub-agent round)

Every sub-agent builds against THIS document. If something here is wrong or missing, do not invent
a different interface: implement what is written, and list the problem under "contract issues" in your
final report. The parent integrates and resolves.

Baseline commit: `e5d4872`. Python: `.venv/Scripts/python.exe` (3.13). Project root:
the local clone of this repository. Shell is git-bash on Windows.

## 1. File ownership

Only edit files you own. Read anything. Need a change in someone else's file? Put the exact patch
(unified diff or precise description) under "patch requests" in your final report. Never edit it yourself.

| Owner | Files (exclusive write access) |
|---|---|
| WP1 XPU | `playground/ml/nets.py`; `scripts/xpu_diagnostics.py`, `scripts/xpu_stress.py`, `scripts/gpu_load_lmstudio.py`; may DELETE `scripts/spike_torch.py`, `scripts/spike_xpu_bisect.py`, `scripts/spike_xpu_bisect2.py`, `scripts/xpu_attribute.py`, `scripts/xpu_attribute2.py` after folding anything useful into the two kept scripts; new `tests/test_ml_training.py`; new `reports/XPU_DIAGNOSIS.md`; anything under `reports/xpu_*` |
| WP2 Memory | `playground/memory/store.py`, `playground/session/runner.py`, `playground/session/manager.py`; new `tests/test_memory_persistence.py`; new `reports/MEMORY_PERSISTENCE.md` |
| WP3 NN | `playground/ml/data.py`, `features.py`, `evaluate.py`, `controller.py`, `registry.py`, `__init__.py`, new `playground/ml/agent_tools.py`; `playground/interface/environment.py`, `playground/interface/schema.py`; `playground/integrity/evidence.py` (only if needed for the rule in 3.4); `playground/agents/adapters.py` (only to add the `neural` adapter type); `playground/cli.py`; new `tests/test_ml_lab.py` |
| WP4 Dashboard | everything under `playground/dashboard/` (server.py, static/*); new `tests/test_dashboard_ml.py` |
| WP5 QA/docs | `README.md`, `docs/OPERATING.md`, new `docs/ML_LAB.md`, `run_tests.sh`, `scripts/health_check.py`; all other `scripts/*` NOT listed under WP1 (consolidate/delete one-off probes); new `tests/test_regression_extra.py` |
| Parent only | `config/default.yaml`, `requirements*.txt`, `pyproject.toml`, `playground/agents/prompts.py`, `playground/world/*`, `tests/test_hve.py`, `tests/test_smoke.py`, `tests/conftest.py`, this file, git |

Config changes: do NOT edit `config/default.yaml`. Put the YAML snippet you need in your report; code
must work with sensible in-code defaults until the parent merges it (read with `cfg.get(...)`).

## 2. Rules for every sub-agent

- No git commits, no `git checkout/reset/stash/clean`, no branch changes. `git diff`/`git status` are fine.
- Other agents are editing other files at the same time. If an existing test fails in code you do not own,
  re-run once; if it still fails, report it, don't fix it.
- Only WP1 uses the GPU (`device="xpu"`). Everyone else trains with `device="cpu"` in code paths they
  exercise and in tests, so the contention experiments are not disturbed.
- Don't touch `experiments/` data except under a temp dir or a clearly named subfolder you create
  (`experiments/_wp<N>_scratch/`), which you delete at the end. Tests use tmp dirs (see `tests/conftest.py`).
- Don't start/stop the dashboard on port 8765 (in use). Use another free port if you need your own server.
- "Done" = your new tests pass + `tests/test_hve.py tests/test_smoke.py` still pass, with the real pytest
  output pasted in your final report. Never claim a result you did not run.

## 3. ML interface

### 3.1 Existing core (keep signatures)
- `playground.ml.MLError(code, message, **details)`; `playground.ml.torch_status() -> dict`.
- `nets.validate_spec(spec) -> spec`: `{"name", "inputs": [col...], "target": col, "hidden": [64, 64]
  (default), "activation": "relu", "dropout": 0.0}`; columns from `features.INPUT_COLUMNS` / `TARGET_COLUMNS`.
- `nets.validate_train(cfg) -> cfg`: keys `epochs, batch_size, lr, weight_decay, seed, device
  ("auto"|"cpu"|"xpu"), patience, val_fraction, test_fraction`.
- `nets.train(spec, rows, train_cfg) -> result` with keys at least: `spec, train_config, device, state_dict
  (lists), normaliser, metrics{test, baseline_mean, baseline_linear, beats_mean_baseline,
  beats_linear_baseline}, history[{epoch, train_loss, val_loss}], best_epoch, epochs_run, seconds,
  parameters, checks{gradients_finite, weight_change_l2, first_train_loss, best_val_loss}, split,
  test_examples`. WP1 may ADD keys, must not remove/rename.
- `nets.Predictor(model_dict).predict(row) / .predict_rows(rows)` (CPU inference). A model dict is what
  `Registry.load_model` returns (it contains spec, state_dict, normaliser).
- `nets.summarise(result)`.
- `data.generate_throw_dataset(cfg, n, seed, ranges=None) -> {"manifest", "rows"}`;
  `data.dataset_from_session_events(db, session_ids) -> {"manifest", "rows"}`; `save_dataset / load_dataset`.
- `evaluate.fresh_engine_test(cfg, model, n, seed)`; `evaluate.closed_loop_height(cfg, model, targets, ...)`.
- `registry.Registry(cfg, root=None)`: `save_dataset(name, ds)`, `load_dataset(name)`, `list_datasets()`,
  `dataset_info()`, `save_model(owner, result, dataset, extra=None) -> {owner,name,version,path,content_sha256}`,
  `load_model(owner, name, version=None)` (verifies content hash, raises `model_tampered`),
  `list_models(owner=None)`, `attach(owner, name, version, key, value)`.

### 3.2 New in WP3 (registry.py) — used by WP2, WP4, WP5
- `registry.researcher_registry(cfg) -> Registry` = `Registry(cfg)` (root `<experiments>/ml/`).
- `registry.agent_registry(cfg, agent_id, memory_dir: Path | None = None) -> Registry` with root
  `<memory_root>/<agent_id>/nn/` (i.e. `<agent memory folder>/nn/datasets`, `/nn/models/<agent_id>/<name>/v<k>.json`).
  Agent-built networks therefore live INSIDE the agent's memory folder: they persist across sessions,
  are copied by `MemoryStore.copy_to` (ablation/benchmark copies) and are agent-isolated.
- `registry.all_models(cfg) -> list[dict]`: researcher models + every agent's models, each with an added
  `"scope": "researcher" | "agent"` and `"agent_id"`.

### 3.3 Agent-facing tools (WP3: schema.py + environment.py + ml/agent_tools.py)
New tool category `"learning"` (NOT a physical category). `program_ok=False` for all. Names and args:

| Tool | Args | Result (success) |
|---|---|---|
| `nn_list` | — | `{datasets: [...], models: [{name, version, target, inputs, test_rmse, linear_rmse, beats_linear}], columns: {inputs: {...}, targets: {...}}}` |
| `nn_make_dataset` | `name`, `source`: `"this_session"` \| `"all_my_sessions"` | `{name, rows, columns, note}`. Rows come ONLY from throws this agent actually made (engine flight records in the DB for this agent's sessions). Never generated for it. |
| `nn_train` | `dataset`, `name`, `inputs[]`, `target`, optional `hidden[]`, `epochs` (≤ 2000), `activation` | `{name, version, test: {...}, baseline_linear: {...}, beats_linear_baseline, epochs_run, parameters, seconds}`; CPU only; refuses < 20 usable rows with `not_enough_data` |
| `nn_predict` | `model`, `inputs{}` (or `rows[]` ≤ 50), optional `version` | `{prediction(s), model, version, "is_prediction": true, "note": "a prediction, not a measurement"}` |
| `nn_evaluate` | `model`, optional `version` | held-out metrics + `{"fresh_check": ...}` predictions vs the agent's own throws recorded AFTER the model was trained (if any) |

Rules: all learning-tool results carry `"is_prediction": true` where they contain predicted numbers;
`record_experiment` evidence checks must NOT accept learning-tool actions as measurements (they are not in
the physical categories, so this should already hold — WP3 verifies with a test). Models are saved via
`agent_registry(cfg, agent_id, memory.dir)`. Training counts against the session's wall clock.

### 3.4 Memory integration (WP2)
- `MemoryStore.briefing()` lists the agent's networks (name, version, target, inputs, test RMSE vs
  linear) under `----- nn/ (networks you trained; run nn_list / nn_predict) -----` when `<dir>/nn/models`
  has any; read the JSON files directly (don't import torch).
- `MemoryStore.close_session` snapshot includes `nn/` (model JSONs only).
- `MemoryStore.stats()` gains `"networks": <count>`.
- `remove_item` supports ids of the form `NN-<name>` (removes that network's folder, audited), so the
  ablation tooling can remove a network like any other memory item. `all_items()` lists them with
  `kind: "network"`.

### 3.5 CLI (WP3, playground/cli.py) — subcommand `ml`
`playground ml status` · `ml generate --name N --n 1500 --seed 7` · `ml train --dataset N --name M
--inputs a,b,c --target max_height [--hidden 64,64] [--epochs 400] [--device auto|cpu|xpu]` ·
`ml list [--owner X]` · `ml evaluate --owner researcher --name M [--version K] [--fresh 200]
[--closed-loop]` · `ml controller --name M` (runs the `throw_height` benchmark task with the neural
controller and prints the verified result). All print JSON.

### 3.6 HTTP API (WP4, playground/dashboard/server.py) — read-mostly, researcher side
- `GET /api/ml/status` → `torch_status()`
- `GET /api/ml/datasets` → researcher `dataset_info()`
- `GET /api/ml/models` → `all_models(cfg)` (no weights)
- `GET /api/ml/models/{scope}/{owner}/{name}?version=K` → model without `state_dict` (history, metrics,
  test_examples, checks, closed_loop if attached)
- `POST /api/ml/train` `{dataset, spec, train: {...}}` → `{job_id}`; trains in a background thread
  (one at a time; 409 if busy); researcher registry only.
- `GET /api/ml/jobs/{job_id}` → `{status: queued|running|done|error, result_summary|error}`
- `POST /api/ml/evaluate` `{owner, name, version, kind: "fresh"|"closed_loop", n}` → job id, same jobs API.
Dashboard tab "Neural networks": status (torch/XPU), datasets, model table (researcher + agents, scope
column), model detail (loss curve train/val, metrics vs baselines, test examples, checks, closed-loop
results), train form, evaluate buttons. Agent-authored text escaped with `esc()`.

## 4. Integration gate (parent, after all five finish)
1. Collect each report's changed-file list; `git diff --stat` must show no edits outside ownership.
2. Review every diff; apply patch requests + config snippets.
3. Full suite (`bash run_tests.sh` + every `tests/test_*.py`) passes.
4. CPU and XPU training both run; XPU integrated `nets.train` run while LM Studio is generating.
5. Checkpoint save → reload → identical predictions.
6. Demo still passes; dashboard tabs load.
7. One reviewed commit. The 13-agent experiment comes after this, with a capable model loaded.
