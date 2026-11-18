"""Live-frame extraction against the real LocoEnv (B2 tree).

These run the real MuJoCo world — the only honest way to test that frames
carry the actual state. Torch is not needed here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from playground.rl.env import LocoEnv
from playground.rl_live.frames import FrameIndex, pack_frame, scene_spec
from playground.rl_live.tap import FrameTapEnv, RecordingSink


def make_env(tmp_path):
    return LocoEnv(cfg={"_root": str(tmp_path), "agent_body": {"type": "humanoid"}})


def test_scene_spec_shape(tmp_path):
    env = make_env(tmp_path)
    spec = scene_spec(env)
    assert spec["spec_version"] == 1
    names = {b["name"] for b in spec["bodies"]}
    assert "torso" in names and "left_foot" in names and "right_foot" in names
    rig = [b for b in spec["bodies"] if b["rig"]]
    assert len(rig) >= 10                       # the humanoid rig
    assert spec["policy_joints"] == list(env.joint_names)
    assert len(spec["policy_joints"]) == 15   # M7-0a v4
    for body in spec["bodies"]:
        for g in body["geoms"]:
            assert g["type"] in ("plane", "sphere", "capsule", "cylinder",
                                 "box", "ellipsoid")
            assert len(g["rgba"]) == 4
        assert "parent" in body
    assert isinstance(spec["meshes_skipped"], int)
    json.dumps(spec)                            # must be viewer-serialisable


def test_pack_frame_through_tap(tmp_path):
    env = make_env(tmp_path)
    sink = RecordingSink()
    tap = FrameTapEnv(env, sink=sink)
    obs = tap.reset(seed=3)
    assert obs.shape == (env.obs_dim,)
    a = np.zeros(env.action_dim, dtype=np.float32)
    for _ in range(3):
        tap.step(a)
    kinds = [m["t"] for m in sink.messages]
    assert kinds[0] == "reset"
    assert kinds.count("frame") == 3
    assert "episode_end" not in kinds           # 3 decisions, no episode end
    f = sink.messages[2]
    assert f["episode"] == 1 and f["decision"] == 2
    assert set(f["bodies"])
    assert set(f["joints"]) == set(env.joint_names)
    assert f["action"] == [0.0] * env.action_dim
    assert np.isfinite(f["root"]["pos"]).all()
    assert -1.0 <= f["up_z"] <= 1.0
    assert isinstance(f["reward"], float)
    assert set(f["contacts"]) == {"left", "right"}
    assert f["dt_window"] == pytest.approx(env.action_repeat * float(
        env.world.m.opt.timestep), rel=1e-6)
    # drive to the episode cap: one truncated episode_end, then an explicit
    # new episode (the TAP itself never auto-resets — the trainer does that)
    for _ in range(env.max_steps):
        tap.step(a)
    ends = [m for m in sink.messages if m["t"] == "episode_end"]
    assert len(ends) == 1 and ends[0]["truncated"] is True
    tap.reset(seed=4)
    assert tap.episode_count == 2
    assert sink.messages[-1]["t"] == "reset"


def test_tap_is_invisible_to_determinism(tmp_path):
    """Tap-wrapped stepping produces the exact same trajectories as direct
    stepping (the tap does not touch the env), even with a hostile sink."""
    a = np.zeros(15, dtype=np.float32)   # M7-0a v4 action
    obs1 = make_env(tmp_path).reset(seed=7)     # separate worlds, same seed
    env1 = make_env(tmp_path)
    env1.reset(seed=7)
    direct = []
    for _ in range(3):
        o, r, term, trunc, info = env1.step(a)
        direct.append((r, o.copy(), bool(info.get("faulted"))))

    class Angrier:
        def __init__(self): self.n = 0
        def __call__(self, msg): self.n += 1; raise RuntimeError("viewer died")

    env2 = make_env(tmp_path)
    tap = FrameTapEnv(env2, sink=Angrier())
    obs2 = tap.reset(seed=7)
    tap_results = [tap.step(a) for _ in range(3)]
    assert obs2.shape == obs1.shape
    np.testing.assert_array_equal(obs2, obs1)
    for (rew_d, obs_d, fault_d), (o_t, rew_t, term_t, trunc_t, info_t) in \
            zip(direct, tap_results):
        assert rew_t == rew_d
        assert bool(info_t.get("faulted")) == fault_d
        np.testing.assert_array_equal(o_t, obs_d)
    assert tap.dropped_frames > 0                # the failures were isolated
    assert tap.live_frames == 0                  # nothing sent, nothing broken


def test_frame_json_budget(tmp_path):
    env = make_env(tmp_path)
    idx = FrameIndex(env)
    env.reset(seed=11)
    _, r, term, trunc, info = env.step(np.zeros(15, dtype=np.float32))
    frame = pack_frame(env, action=np.zeros(15), reward=r, terminated=term,
                       truncated=trunc, info=info, episode=1, decision=1,
                       index=idx)
    blob = json.dumps(frame)
    assert len(blob) < 12_000                    # per-frame JSON budget
    assert frame["v"] == 1
