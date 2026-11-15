"""HVE coverage for the Nov 2026 dashboard/session fixes.

Covers three things the user reported broken:
  1. Behaviour datasets built from the recorded log returned 0 rows (build_dataset read
     `args_json`/`result_json`, but Database.actions() returns parsed `args`/`result` dicts).
  2. The NN training form only offered throw columns, so every generated dataset could not be trained on
     (/api/ml/columns now serves dataset-specific vocabularies).
  3. The dashboard had no stop-session control (/api/sessions/{sid}/stop + /api/run/jobs/{id}/stop and the
     per-session stop signal in the runner).
Every fixture workspace is temporary: nothing touches the real experiments/ database.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import playground.session.runner as runner_mod
from playground.agents import make_adapter
from playground.dashboard import create_app
from playground.ml.datasets import DATASETS, build_dataset
from playground.ml.features import INPUT_COLUMNS, TARGET_COLUMNS
from playground.ml.registry import Registry
from playground.session.runner import (_register_session_stop, _unregister_session_stop,
                                       is_session_stop_requested, request_stop_session, run_session)

# --------------------------------------------------------------------------- helpers


def _climb_action(idx: int, ok: bool = True) -> dict:
    """A realistic climb action record, shaped exactly like Database.actions() returns them."""
    result = {"action": "climb", "action_index": idx, "sim_time": 1.0 + idx, "success": ok,
              "ledge_height": 1.0 if ok else None, "start_position": [-4.2855, -3.28, 0.8],
              "started_airborne": False, "height_gained": 1.0 if ok else None}
    if not ok:
        result = {"action": "climb", "action_index": idx, "sim_time": 1.0 + idx, "success": ok,
                  "error": "cannot_climb"}
    return {"idx": idx, "tool": "climb", "category": "movement", "args": {"direction": 90},
            "result": result, "success": 1 if ok else 0, "error_code": None, "source": "agent", "turn": 1}


def _jump_action(idx: int) -> dict:
    result = {"action": "jump", "action_index": idx, "sim_time": 5.0 + idx, "success": True,
              "total_mass": 80.0, "takeoff_feet_height": 0.8, "apex_above_takeoff": 0.766,
              "air_time": 0.842, "max_feet_height": 1.57, "impact_speed": 3.88}
    return {"idx": idx, "tool": "jump", "category": "movement", "args": {},
            "result": result, "success": 1, "error_code": None, "source": "agent", "turn": 1}


def _move_action(idx: int) -> dict:
    result = {"action": "move", "action_index": idx, "sim_time": 9.0 + idx, "success": True,
              "start_position": [0.0, 0.0, 0.8], "end_position": [0.0, 3.03, 0.8],
              "distance_travelled": 3.03, "height_change": 0.0, "elapsed": 1.95,
              "velocity": [0.0, 1.55, 0.0], "still_moving": False}
    return {"idx": idx, "tool": "move", "category": "movement", "args": {"direction": 90, "distance": 3},
            "result": result, "success": 1, "error_code": None, "source": "agent", "turn": 2}


# --------------------------------------------------------------------------- 1. dataset rows from the log


class TestBuildDatasetFromDatabase:
    def test_rows_coming_from_database_actions_are_not_empty(self, cfg, db):
        """Regression: the dashboard passes rows straight from db.actions(), whose keys are parsed
        `args`/`result` dicts. Before the fix this path produced 0 rows even with good records."""
        sid = db.create_session(agent_id="agent_t", mode="exploration", session_number=1, adapter="test")
        db.append_action(sid, **_climb_action(1), t_start=1.0, t_end=2.0, wall_ts=100.0)
        db.append_action(sid, **_jump_action(2), t_start=3.0, t_end=4.0, wall_ts=101.0)
        db.append_action(sid, **_move_action(3), t_start=5.0, t_end=6.0, wall_ts=102.0)
        actions = db.actions(sid)
        assert actions, "fixture produced no actions"
        for name in ("climb", "jump", "locomotion"):
            ds = build_dataset(actions, name)
            assert ds["rows"], f"{name} returned 0 rows from real database records"
        # climb target column is derivable: the vocabulary comes from the manifest
        ds = build_dataset(actions, "climb")
        assert ds["manifest"]["kind"] == "climb"
        assert ds["manifest"]["inputs"] and ds["manifest"]["targets"]

    def test_failed_action_rows_are_skipped_not_crashing(self, cfg, db):
        sid = db.create_session(agent_id="agent_t", mode="exploration", session_number=1, adapter="test")
        db.append_action(sid, **_climb_action(1, ok=False))
        ds = build_dataset(db.actions(sid), "climb")
        assert ds["rows"] == []

    def test_raw_json_records_still_work(self):
        """Direct CLI/scripts use records with raw `args_json`/`result_json` strings; keep that path."""
        actions = [{"idx": 0, "tool": "climb", "args": {"direction": 90},
                    "result": {"action": "climb", "success": True, "ledge_height": 1.0,
                               "start_position": [0, 0, 0], "started_airborne": False,
                               "height_gained": 1.0}}]
        ds = build_dataset(actions, "climb")
        assert len(ds["rows"]) == 1

    def test_malformed_json_records_are_skipped(self):
        actions = [{"tool": "climb", "args_json": "{not json", "result_json": "{alsonot"}]
        ds = build_dataset(actions, "climb")
        assert ds["rows"] == []

    def test_unknown_dataset_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_dataset([], "not_a_real_dataset")

    def test_every_registered_dataset_builds_some_rows_from_mixed_record(self, cfg, db):
        """The 'most generations return 0 rows' complaint: every catalogue dataset must be able to mine
        rows from a log that contains its tool. Datasets keyed to tools absent from the fixture log
        (manipulation/placement/affordance) are validated for shape instead of row count."""
        sid = db.create_session(agent_id="agent_t", mode="exploration", session_number=1, adapter="test")
        db.append_action(sid, **_climb_action(1))
        db.append_action(sid, **_jump_action(2))
        db.append_action(sid, **_move_action(3))
        actions = db.actions(sid)
        tools_logged = {a["tool"] for a in actions}
        for name, s in DATASETS.items():
            ds = build_dataset(actions, name)
            if set(s.tools) & tools_logged:
                assert ds["rows"], f"dataset {name} ignores actions for {s.tools}"
            else:
                assert ds["rows"] == [], f"{name} produced rows without its tools being logged"


# --------------------------------------------------------------------------- 2. dataset-specific columns


@pytest.fixture
def client(cfg, db):
    return TestClient(create_app(cfg, db))


class TestDatasetSpecificColumns:
    def test_default_columns_are_the_throw_vocabulary(self, client):
        r = client.get("/api/ml/columns")
        assert r.status_code == 200
        data = r.json()
        assert set(data["inputs"]) == set(INPUT_COLUMNS)
        assert set(data["targets"]) == set(TARGET_COLUMNS)

    def test_behaviour_dataset_serves_its_own_vocabulary(self, cfg, client):
        reg = Registry(cfg)
        actions = [_climb_action(i) for i in range(3)]
        ds = build_dataset(actions, "climb")
        reg.save_dataset("climb_cols_test", ds)
        r = client.get("/api/ml/columns", params={"dataset": "climb_cols_test"})
        assert r.status_code == 200
        data = r.json()
        assert data["dataset"] == "climb_cols_test"
        # The climb row layout, not the throw layout
        assert "ledge_height" in data["inputs"] or "ledge_height" in data["targets"]
        assert "force" not in data["inputs"], "throw column leaked into behaviour dataset vocabulary"
        inputs, targets = data["inputs"], data["targets"]
        rows = ds["rows"]
        # Every manifest key exists in the served vocabulary
        assert set(next(iter(rows)).keys()) <= set(inputs) | set(targets)

    def test_unknown_dataset_returns_400(self, client):
        r = client.get("/api/ml/columns", params={"dataset": "no_such_dataset"})
        assert r.status_code == 400

    def test_throw_dataset_falls_back_to_throw_vocab(self, cfg, client):
        from playground.ml.data import generate_throw_dataset
        reg = Registry(cfg)
        reg.save_dataset("throw_cols_fallback", generate_throw_dataset(cfg, 40, seed=1))
        r = client.get("/api/ml/columns", params={"dataset": "throw_cols_fallback"})
        assert r.status_code == 200
        data = r.json()
        assert data["dataset"] == "throw_cols_fallback"
        assert set(data["inputs"]) == set(INPUT_COLUMNS)

    def test_build_behaviour_dataset_via_dashboard_job_has_rows(self, cfg, db, client):
        """End-to-end: the same endpoint the dashboard's behaviour-builder form hits now yields rows."""
        sid = db.create_session(agent_id="agent_t", mode="exploration", session_number=1, adapter="test")
        db.append_action(sid, **_climb_action(1))
        db.append_action(sid, **_climb_action(2))
        r = client.post("/api/ml/build-behaviour", json={"name": "climb_job_test", "kind": "climb"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        summary = wait_ml_job(client, job_id)
        assert summary["rows"] > 0, "dashboard behaviour build still produces 0 rows"
        cols = client.get("/api/ml/columns", params={"dataset": "climb_job_test"})
        assert cols.status_code == 200
        assert cols.json()["targets"]


def wait_ml_job(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/ml/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            assert j["status"] == "done", j.get("error")
            return j["result_summary"]
        time.sleep(0.05)
    raise AssertionError("ML job did not finish")


# --------------------------------------------------------------------------- 3. stop session


class TestStopSignalRegistry:
    def test_request_stop_sets_flag_for_registered_session(self):
        _register_session_stop(4242)
        try:
            assert is_session_stop_requested(4242) is False
            assert request_stop_session(4242) is True
            assert is_session_stop_requested(4242) is True
        finally:
            _unregister_session_stop(4242)

    def test_request_stop_for_unknown_session_returns_false(self):
        _unregister_session_stop(987654)
        assert request_stop_session(987654) is False


class _SlowAdapter:
    """Adapter that keeps calling look_around forever, so the runner loop keeps going until stopped."""

    name, model, driven_externally = "slow_test", "test", False
    calls = 0

    def begin(self, system, tool_names):
        pass

    def step(self, user_text, tool_results):
        _SlowAdapter.calls += 1
        time.sleep(0.02)
        from playground.agents.adapters import ToolCall, Turn
        return Turn(tool_calls=[ToolCall(f"c{_SlowAdapter.calls}", "look_around", {})])

    def describe(self):
        return {"adapter": self.name, "model": self.model}


class TestStopRunningSession:
    @staticmethod
    def _running_sid(db):
        rows = db.q("SELECT id FROM sessions WHERE agent_id='agent_t' AND status='running'")
        return rows[0]["id"] if rows else None

    @classmethod
    def _wait_registered(cls, db, timeout: float = 15.0) -> int:
        """The runner registers the stop signal right after creating the session row; wait for that
        moment so the stop request cannot race the (very short) startup window."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            sid = cls._running_sid(db)
            if sid is not None and sid in runner_mod._session_stop_events:
                return sid
            time.sleep(0.01)
        raise AssertionError("test session never started")

    def test_run_session_stops_with_stopped_by_user(self, cfg, db):
        result: dict = {}

        def target():
            result["s"] = run_session(cfg, db, "agent_t", _SlowAdapter(), mode="exploration")

        th = threading.Thread(target=target, daemon=True)
        th.start()
        sid = self._wait_registered(db)
        assert request_stop_session(sid) is True
        th.join(timeout=60)
        assert not th.is_alive()
        s = result["s"]
        assert s["end_reason"] == "stopped_by_user", s["end_reason"]
        assert s["error"] is None
        assert db.session(sid)["status"] == "completed"

    def test_stop_endpoint_on_running_session(self, cfg, db):
        app = create_app(cfg, db)
        client = TestClient(app)
        result: dict = {}

        def target():
            result["s"] = run_session(cfg, db, "agent_t", _SlowAdapter(), mode="exploration")

        th = threading.Thread(target=target, daemon=True)
        th.start()
        sid = self._wait_registered(db)
        r = client.post(f"/api/sessions/{sid}/stop")
        assert r.status_code == 200
        assert r.json()["ok"] is True and r.json()["stopped"] is True
        th.join(timeout=60)
        assert result["s"]["end_reason"] == "stopped_by_user"

    def test_stop_endpoint_unknown_session_404(self, cfg, db):
        client = TestClient(create_app(cfg, db))
        assert client.post("/api/sessions/999999/stop").status_code == 404

    def test_stop_endpoint_finished_session_409(self, cfg, db):
        cfg["session"]["max_steps"] = 5
        s = run_session(cfg, db, "agent_t", make_adapter(cfg, "scripted"), mode="exploration")
        client = TestClient(create_app(cfg, db))
        r = client.post(f"/api/sessions/{s['session_id']}/stop")
        assert r.status_code == 409

    def test_run_job_stop_request(self, client):
        r = client.post("/api/run/jobs/does-not-exist/stop")
        assert r.status_code == 404


class TestExternalSessionStop:
    def test_stop_finishes_an_externally_driven_session(self, cfg, db):
        client = TestClient(create_app(cfg, db))
        r = client.post("/api/agent/sessions", json={"agent_id": "external_t", "mode": "exploration"})
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        stop = client.post(f"/api/sessions/{sid}/stop")
        assert stop.status_code == 200
        assert stop.json()["stopped"] is True
        assert "external" in stop.json()["note"]
        row = db.session(sid)
        assert row["status"] == "completed"
        assert row["end_reason"] == "stopped_by_user"


# --------------------------------------------------------------------------- 4. run-form config knobs


class TestRunConfigKnobs:
    def test_explore_run_honours_max_steps_and_concurrency(self, cfg, db):
        cfg["session"]["max_steps"] = 5
        client = TestClient(create_app(cfg, db))
        r = client.post("/api/run/explore", json={
            "agent_id": "agent_t", "adapter": "scripted", "sessions": 2,
            "max_steps": 5, "concurrency": 2, "max_continuations": 0})
        assert r.status_code == 200, r.text
        job = run_job_wait(client, r.json()["job_id"])
        assert job["result"]["sessions_run"] == 2
        # max_steps caps *tool calls* (turns may batch several calls); budget exhaustion ends the
        # session with end_reason "step_limit" after a short wrap-up. Compare with the conftest
        # default (400): the sessions here run single-digit tool calls, proving the knob reached the loop.
        for p in job["progress"]:
            assert p["end_reason"] == "step_limit", f"expected step_limit, got {p['end_reason']} — full: {p!r}"
            assert p["actions"] <= 20, f"session ran {p['actions']} actions despite max_steps=5"

    def test_explore_rejects_unknown_adapter(self, client):
        r = client.post("/api/run/explore", json={"agent_id": "agent_t", "adapter": "nope"})
        assert r.status_code == 400

    def test_running_job_stops_mid_run(self, cfg, db, monkeypatch):
        """A stop mid-run must signal the in-flight session and spawn nothing more."""
        from playground.agents.adapters import Adapter

        app = create_app(cfg, db)
        client = TestClient(app)

        class SlowAdapter(Adapter):
            name, model, driven_externally = "slow_job", "test", False

            def begin(self, system, tool_names):
                pass

            def step(self, user_text, tool_results):
                self.n = getattr(self, "n", 0) + 1
                time.sleep(0.05)
                from playground.agents.adapters import ToolCall, Turn
                return Turn(tool_calls=[ToolCall(f"c{self.n}", "look_around", {})])

            def describe(self):
                return {"adapter": self.name, "model": self.model}

        monkeypatch.setattr("playground.agents.adapters.make_adapter",
                            lambda cfg_, name, model=None: SlowAdapter())
        # The endpoint rejects unknown adapter names before reaching the factory; register a stub so
        # the patched factory is what actually builds the adapter.
        cfg.setdefault("adapters", {})["slow_job"] = {"type": "openai_compat", "base_url": "http://127.0.0.1:1"}
        r = client.post("/api/run/explore", json={
            "agent_id": "agent_t", "adapter": "slow_job", "sessions": 3, "concurrency": 1,
            "max_continuations": 0})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]

        # Wait until the first in-flight session exists, then ask the job to stop.
        deadline = time.time() + 15
        while time.time() < deadline:
            rows = db.q("SELECT id FROM sessions WHERE agent_id='agent_t' AND status='running'")
            if rows and rows[0]["id"] in runner_mod._session_stop_events:
                break
            time.sleep(0.02)
        assert client.post(f"/api/run/jobs/{job_id}/stop").json()["ok"] is True
        job = run_job_wait(client, job_id)
        assert job["result"]["stopped"] is True
        assert job["result"]["sessions_run"] < 3, "job ran all sessions despite the stop request"
        for p in job["progress"]:
            assert p["end_reason"] == "stopped_by_user", p


class TestPanicKillSwitch:
    """One call stops every running session (in-process + external) and every running run job."""

    def test_panic_stops_in_process_and_external_sessions_and_run_jobs(self, cfg, db, monkeypatch):
        from playground.agents.adapters import Adapter
        app = create_app(cfg, db)
        client = TestClient(app)
        results: list[dict] = []

        # 1) an in-process loop session running on a thread
        def target():
            results.append(run_session(cfg, db, "agent_panic", _SlowAdapter(), mode="exploration"))
        t1 = threading.Thread(target=target, daemon=True)
        t1.start()

        # 2) a run job with a slow adapter
        class SlowJobAdapter(Adapter):
            name, model, driven_externally = "slow_job", "test", False
            calls = 0
            def begin(self, system, tool_names):
                pass
            def step(self, user_text, tool_results):
                SlowJobAdapter.calls += 1
                time.sleep(0.03)
                from playground.agents.adapters import ToolCall, Turn
                return Turn(tool_calls=[ToolCall(f"c{SlowJobAdapter.calls}", "look_around", {})])
            def describe(self):
                return {"adapter": self.name, "model": self.model}

        monkeypatch.setattr("playground.agents.adapters.make_adapter",
                            lambda cfg_, name, model=None: SlowJobAdapter())
        cfg.setdefault("adapters", {})["slow_job"] = {"type": "openai_compat", "base_url": "http://127.0.0.1:1"}
        deadline = time.time() + 15
        while time.time() < deadline:
            rows = db.q("SELECT id FROM sessions WHERE agent_id='agent_panic' AND status='running'")
            if rows and rows[0]["id"] in runner_mod._session_stop_events:
                break
            time.sleep(0.02)
        assert rows, "in-process session never started"
        job = client.post("/api/run/explore", json={"agent_id": "agent_panic2", "adapter": "slow_job",
                                                    "sessions": 3, "concurrency": 1}).json()["job_id"]

        # 3) an externally driven session
        ext = client.post("/api/agent/sessions", json={"agent_id": "agent_panic3", "mode": "exploration"}).json()

        # PANIC
        r = client.post("/api/panic")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["sessions_signalled"] >= 2, body     # the in-process session + the external one
        assert body["external_finished"] == 1
        assert body["run_jobs_stopping"] == 1

        # The run job winds down (stops spawning, stops in-flight sessions).
        deadline = time.time() + 60
        while time.time() < deadline:
            j = client.get(f"/api/run/jobs/{job}").json()
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert j["status"] == "done", j
        assert j["result"]["stopped"] is True
        assert j["result"]["sessions_run"] < 3

        t1.join(timeout=60)
        assert not t1.is_alive()
        assert results[0]["end_reason"] == "stopped_by_user"
        # The external session was finished cleanly.
        assert db.session(ext["session_id"])["status"] == "completed"
        assert db.session(ext["session_id"])["end_reason"] == "stopped_by_user"

    def test_panic_with_nothing_running_is_a_clean_noop(self, client):
        r = client.post("/api/panic")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["sessions_signalled"] == 0 and body["run_jobs_stopping"] == 0


class TestIncrementalDatasets:
    """After each run the log-derived datasets rebuild; the UI can see what changed."""

    def test_refresh_log_datasets_rebuilds_and_flags_changes(self, cfg, db):
        from playground.ml import datasets as ds_mod
        from playground.ml.registry import Registry
        reg = Registry(cfg)
        # Seed a behaviour dataset from two climb actions, plus a throw dataset that must be untouched.
        sid = db.create_session(agent_id="agent_t", mode="exploration", session_number=1, adapter="test")
        db.append_action(sid, **_climb_action(1))
        actions = db.actions(sid)
        reg.save_dataset("climb", build_dataset(actions, "climb"))
        from playground.ml.data import generate_throw_dataset
        reg.save_dataset("throw_seed", generate_throw_dataset(cfg, 6, seed=3))

        out = ds_mod.refresh_log_datasets(db, reg)
        by_name = {d["name"]: d for d in out}
        assert "climb" in by_name and "throw_seed" not in by_name, "throw datasets must not be refreshed"
        assert by_name["climb"]["rows"] == 1

        # A new session logs more actions; refresh must grow the dataset and flag the change.
        sid2 = db.create_session(agent_id="agent_t", mode="exploration", session_number=2, adapter="test")
        db.append_action(sid2, **_climb_action(1))
        out2 = ds_mod.refresh_log_datasets(db, reg)
        by2 = {d["name"]: d for d in out2}
        assert by2["climb"]["rows"] == 2
        assert by2["climb"]["changed"] is True
        assert reg.load_dataset("climb")["manifest"]["updated_at"] > 0

        # Refreshing again with no new actions: same rows, no 'changed' flag.
        out3 = ds_mod.refresh_log_datasets(db, reg)
        by3 = {d["name"]: d for d in out3}
        assert by3["climb"]["rows"] == 2 and by3["climb"]["changed"] is False

    def test_run_job_triggers_auto_dataset_refresh(self, cfg, db):
        from playground.ml.registry import Registry
        cfg["session"]["max_steps"] = 5
        reg = Registry(cfg)
        actions = [_climb_action(i) for i in (1, 2)]
        reg.save_dataset("climb", build_dataset(actions, "climb"))
        client = TestClient(create_app(cfg, db))
        r = client.post("/api/run/explore", json={"agent_id": "agent_t", "adapter": "scripted",
                                                  "sessions": 1, "max_steps": 5})
        job = run_job_wait(client, r.json()["job_id"])
        deadline = time.time() + 30
        refresh = None
        while time.time() < deadline:
            j = client.get(f"/api/run/jobs/{r.json()['job_id']}").json()
            refresh = j.get("dataset_refresh")
            if refresh:
                break
            time.sleep(0.1)
        assert refresh, "run job did not report a dataset refresh"
        names = [d["name"] for d in refresh if "name" in d]
        assert "climb" in names


class TestDatasetQualityPanel:
    """One GET answers rows/age, log coverage, class balance, warnings and lineage for a dataset."""

    def test_quality_reports_balances_warnings_and_lineage(self, cfg, db, client):
        import hashlib as hl
        import json as jl
        reg = Registry(cfg)
        sid = db.create_session(agent_id="agent_t", mode="exploration", session_number=1, adapter="test")
        for i in (1, 2, 3):                                   # all failed=False → skewed success
            db.append_action(sid, **_climb_action(i))
        ds = build_dataset(db.actions(sid), "climb")
        ds["manifest"]["rows_sha256"] = hl.sha256(jl.dumps(ds["rows"], sort_keys=True).encode()).hexdigest()
        ds["manifest"]["updated_at"] = time.time() - 120     # 2 minutes old
        reg.save_dataset("climb_q", ds)
        r = client.get("/api/ml/datasets/climb_q/quality")
        assert r.status_code == 200
        q = r.json()
        assert q["rows"] == 3
        assert q["coverage"] == {"climb": 3}
        assert q["balance"]["column"] == "success" and q["balance"]["class_1"] == 3 and q["balance"]["class_0"] == 0
        assert any("20-row" in w for w in q["warnings"]), q["warnings"]
        assert any("skew" in w for w in q["warnings"])
        assert q["rows_sha256"] and len(q["rows_sha256"]) == 64
        assert q["stale"] is False and 1 <= q["age_minutes"] <= 3

    def test_quality_zero_rows_dataset_warns_hard(self, cfg, db, client):
        reg = Registry(cfg)
        reg.save_dataset("empty_ds", build_dataset([], "climb"))
        q = client.get("/api/ml/datasets/empty_ds/quality").json()
        assert q["rows"] == 0
        assert any(w.startswith("0 rows") for w in q["warnings"])
        assert q["balance"] is None

    def test_quality_unknown_dataset_400(self, client):
        assert client.get("/api/ml/datasets/nope/quality").status_code == 400

    def test_quality_of_throw_dataset(self, cfg, db, client):
        from playground.ml.data import generate_throw_dataset
        reg = Registry(cfg)
        reg.save_dataset("throw_q", generate_throw_dataset(cfg, 40, seed=1))
        q = client.get("/api/ml/datasets/throw_q/quality").json()
        assert q["kind"] is None and q["coverage"] == {}          # throw datasets carry no kind/tools
        assert 30 <= q["rows"] <= 40                              # engine variance: failed throws are filtered


def run_job_wait(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/run/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            assert j["status"] == "done", j.get("error")
            return j
        time.sleep(0.1)
    raise AssertionError("run job did not finish")
