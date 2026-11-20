"""S1 individual-limb pipeline: capability eval (honest mastery gate), live
session S1 objectives + launch-gate enforcement, and the per-limb ladder.
Complements tests/test_rl_m7_s1_stages.py (ladder/env/schedule)."""
import time

import pytest

from playground.rl.env import LocoEnv
from playground.rl.ppo import TanhGaussianActor
from playground.rl import s1_eval
from playground.rl_live.session import LiveBrainSession, OBJECTIVES


# ----------------------------------------------------- capability evaluation
def _fresh_policy():
    e = LocoEnv()
    return TanhGaussianActor(e.obs_dim, e.action_dim, hidden=(64, 64), seed=1)


def test_s1_eval_untrained_is_not_mastered():
    """The honest mastery gate: an untrained policy cannot track the commanded
    limb, so it is reported NOT mastered — never on step count, only on
    measured tracking against the recorded bands."""
    v = s1_eval.evaluate_s1_limb(None, "m7_s1_head", episodes=1,
                                 policy=_fresh_policy())
    assert v["mastered"] is False
    assert set(v["commanded_joints"]) == {"head_pitch", "head_yaw", "head_roll"}
    # every commanded joint carries a measured err and a band, and fails wide
    for n in v["commanded_joints"]:
        row = v["joints"][n]
        assert row["band_rad"] > 0
        assert row["steady_abs_err_rad"] > row["band_rad"]
        assert row["pass"] is False
    assert "chest_up_z_min" in v["posture"]


def test_s1_eval_pins_and_bands():
    """Leg stages pin the ankle (checked as neutral-hold, not tracking) and the
    gate bands come from the recorded calibration probe."""
    v = s1_eval.evaluate_s1_limb(None, "m7_s1_left_leg", episodes=1,
                                 policy=_fresh_policy())
    assert "left_ankle" in v["pinned_joints"]
    assert v["joints"]["left_ankle"]["kind"] == "pinned_neutral"
    bands = s1_eval.load_gate_bands("m7_s1_head")
    assert bands["head_pitch"] > 0


def test_s1_eval_rejects_bad_stage():
    with pytest.raises(ValueError):
        s1_eval.evaluate_s1_limb(None, "standing", policy=_fresh_policy())
    with pytest.raises(ValueError):
        s1_eval.evaluate_s1_limb(None, "walking", policy=_fresh_policy())


# ------------------------------------------------- live session S1 objectives
class _FakeTrainer:
    def __init__(self, target, on_iteration, should_stop):
        self.total_valid_steps = 0
        self.iteration = 0
        self._target = target
        self.on_iteration = on_iteration
        self.should_stop = should_stop

    def train(self):
        while self.total_valid_steps < self._target:
            if self.should_stop():
                return
            time.sleep(0.005)
            self.iteration += 1
            self.total_valid_steps += 10
            self.on_iteration({"iteration": self.iteration,
                               "total_valid_steps": self.total_valid_steps,
                               "stage": "m7_s1_head",
                               "roll_comp_limb_tracking_mean": -1.0})

    def rng_state(self):
        return {"total_valid_steps": self.total_valid_steps}


def _fake_saver(path, **kw):
    from pathlib import Path
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fake": True}), encoding="utf-8")
    return {"path": str(p), "content_sha256": "x" * 64, "bytes": p.stat().st_size}


def _make_session(tmp_path):
    def factory(env, curriculum, *, name, seed, total_steps, stage_budgets,
                ppo_overrides, on_iteration, should_stop):
        # the real limb ladder env/curriculum was built by the session
        assert [s.id for s in curriculum.stages] == ["standing", "m7_s1_head"]
        return {"trainer": _FakeTrainer(total_steps, on_iteration, should_stop),
                "actor": None, "critic": None, "ppo_cfg_dict": {"hidden": [64, 64]}}
    return LiveBrainSession(trainer_factory=factory, saver=_fake_saver,
                            root=tmp_path / "rl_live")


def test_s1_objectives_registered():
    assert OBJECTIVES["m7_s1_head"]["s1"] is True
    assert OBJECTIVES["m7_s1_head"]["limb_stage_id"] == "m7_s1_head"
    assert OBJECTIVES["walking"]["s1"] is False


def test_s1_start_blocked_when_uncalibrated(tmp_path, monkeypatch):
    """The live start path enforces the calibration launch gate: an
    out-of-window coefficient refuses the S1 objective before any thread."""
    import playground.rl.curriculum as cur
    monkeypatch.setattr(cur, "_M7_S1_W", 0.1)
    sess = _make_session(tmp_path)
    r = sess.start(objective="m7_s1_head", total_steps=100, name="blocked")
    assert r["ok"] is False and r["error"]["code"] == "s1_gate_blocked"
    assert sess.state == "idle"


def test_s1_start_builds_limb_ladder(tmp_path):
    """With the gate satisfied (shipped w=2.0), an S1 start builds the
    standing+limb ladder and records the limb stage in the run config."""
    import json
    sess = _make_session(tmp_path)
    r = sess.start(objective="m7_s1_head", total_steps=30, name="s1run",
                   stage_budgets={"standing": 10})
    assert r["ok"] is True
    t0 = time.time()
    while sess.state not in ("stopped", "error") and time.time() - t0 < 10:
        time.sleep(0.03)
    assert sess.state == "stopped", sess._error
    assert sess.config.limb_stage_id == "m7_s1_head"
    assert sess.config.s1 is True
    rc = json.loads((sess._run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert rc["limb_stage_id"] == "m7_s1_head"
