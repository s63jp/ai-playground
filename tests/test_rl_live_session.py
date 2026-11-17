"""The live session state machine, driven by injected fakes (torch-free).

The fakes exercise the session's contract with the trainer: pause/stop ride a
`should_stop` callback exactly like the real PPOTrainer's; checkpoints go
through an injected saver exactly like the real B3 saver.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from playground.rl_live import interop
from playground.rl_live.session import LiveBrainSession, trainer_steps


# ------------------------------------------------------------------- fakes
class FakeTrainer:
    def __init__(self, target, on_iteration, should_stop):
        self.total_valid_steps = 0
        self.iteration = 0
        self._target = target
        self.on_iteration = on_iteration
        self.should_stop = should_stop
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        while self.total_valid_steps < self._target:
            if self.should_stop():
                return {"stopped": True, "reason": "should_stop"}
            time.sleep(0.01)
            self.iteration += 1
            self.total_valid_steps += 10
            stage = "standing" if self.total_valid_steps <= 30 else "controlled"
            self.on_iteration({
                "iteration": self.iteration,
                "total_valid_steps": self.total_valid_steps,
                "stage": stage,
                "stage_transition_to": ("controlled" if stage == "controlled"
                                        and self.iteration == 4 else None),
                "sigma": 0.99, "mean_reward": 1.5,
                "roll_net_displacement_mean": 0.5,
                "roll_upright_fraction_mean": 1.0,
            })
        return {"stopped": False}

    def rng_state(self):
        return {"total_valid_steps": self.total_valid_steps}


def fake_saver(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fake": True, "kwargs": str(sorted(kwargs))}),
                    encoding="utf-8")
    return {"path": str(path), "content_sha256": "x" * 64,
            "bytes": path.stat().st_size}


def make_session(tmp_path, events=None):
    def factory(env, curriculum, *, name, seed, total_steps, stage_budgets,
                ppo_overrides, on_iteration, should_stop):
        return {"trainer": FakeTrainer(total_steps, on_iteration, should_stop),
                "actor": None, "critic": None,
                "ppo_cfg_dict": {"lr": 3e-4}}

    session = LiveBrainSession(
        trainer_factory=factory, saver=fake_saver,
        broadcaster=(events.append if events is not None else None),
        root=tmp_path / "rl_live")
    return session


def wait_state(sess, want, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sess.state == want:
            return True
        time.sleep(0.02)
    return False


def wait_until(pred, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------------ tests
def test_unavailable_tree_surfaces_clearly(tmp_path, monkeypatch):
    """Post-B3-merge the modules exist, so unavailability is FORCED through
    the availability probe (still a real production state when torch is
    missing); the honest error-surfacing contract stays covered."""
    monkeypatch.setattr("playground.rl_live.interop.trainer_available",
                        lambda: False)
    s = LiveBrainSession(root=tmp_path / "rl_live")
    assert s.state == "error"
    st = s.status()["session"]
    assert "B3" in st["error"]
    r = s.start()
    assert r["ok"] is False and r["error"]["code"] == "trainer_unavailable"
    assert s.status()["trainer_available"] is False


def test_full_state_machine(tmp_path):
    s = make_session(tmp_path)
    assert s.state == "idle"
    r = s.start(name="walk_a", objective="walking", seed=5, total_steps=10_000)
    assert r["ok"] is True
    assert wait_state(s, "running")
    assert wait_until(lambda: len(_iter_recs(s)) >= 3)
    # running: save refused, double start refused
    assert s.save("during")["error"]["code"] == "save_requires_paused"
    r2 = s.start(name="other")
    assert r2["ok"] is False and r2["error"]["code"] == "busy"
    # pause applies at the boundary (fake: immediately)
    assert s.pause()["ok"] is True
    assert wait_state(s, "paused")
    n_when_paused = trainer_steps({"trainer": s._bundle["trainer"]})
    # save while paused succeeds and lands in the run dir
    r = s.save(tag="peek")
    assert r["ok"] is True
    ck = Path(r["checkpoint"]["path"])
    assert ck.exists() and ck.name == "ckpt_peek.json"
    assert ck.parent == Path(s._run_dir)
    # resume continues from the same counter
    assert s.resume()["ok"] is True
    assert wait_state(s, "running")
    assert wait_until(lambda: trainer_steps(
        {"trainer": s._bundle["trainer"]}) > n_when_paused)
    # stop → final checkpoint + stopped
    assert s.stop()["ok"] is True
    assert wait_state(s, "stopped")
    assert (Path(s._run_dir) / "ckpt_final.json").exists()
    # progress.jsonl + run_config.json written
    prog = (Path(s._run_dir) / "progress.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(l) for l in prog.splitlines() if l.strip()]
    assert len(lines) >= 3
    assert lines[0]["stage"] == "standing"
    assert "controlled" in {l["stage"] for l in lines}
    rcfg = json.loads((Path(s._run_dir) / "run_config.json").read_text())
    assert rcfg["name"] == "walk_a" and rcfg["objective"] == "walking"
    # a finished session can start a NEW session
    r = s.start(name="walk_b", seed=6, total_steps=5_000)
    assert r["ok"] is True
    assert wait_state(s, "running")
    s.stop()
    assert wait_state(s, "stopped")


def _iter_recs(s):
    return s.recent_iterations(50)


def test_start_validation(tmp_path):
    s = make_session(tmp_path)
    assert s.start(objective="climbing")["error"]["code"] == "unknown_objective"
    assert s.start(seed=-1)["error"]["code"] == "invalid_config"
    assert s.start(total_steps=0)["error"]["code"] == "invalid_config"
    assert s.start(name="a/b")["error"]["code"] == "invalid_config"
    assert s.start(ppo_overrides={"mystery": 1})["error"]["code"] == \
        "invalid_config"
    assert s.state == "idle"                    # nothing was launched


def test_save_without_session_and_bad_tag(tmp_path):
    s = make_session(tmp_path)
    assert s.save()["error"]["code"] == "nothing_to_save"
    s.start(name="walk_c", total_steps=5_000)
    assert wait_state(s, "running")
    s.pause()
    assert wait_state(s, "paused")
    assert s.save(tag="x/y")["error"]["code"] == "invalid_config"
    s.stop()
    assert wait_state(s, "stopped")
