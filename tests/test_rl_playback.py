"""Checkpoint playback (playground/rl_live/playback.py): the B3-era rig seam,
the verified checkpoint load, the streaming session, the explicit inference
modes, and the read-only guarantees (no PPO code path, no writes to the B3
run dir). Nothing here trains anything.

The heavy test is the recorded-held-out band check (10 real MuJoCo episodes,
THE recorded-evaluation proof); everything else runs on a 6-decision fast
phase.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "experiments" / "rl" / "b3_run1" / "ckpt_final.json"
RECORDED = (REPO / "experiments" / "rl" / "b3_run1" /
            "eval_sustained_s97010000_10ep.json")

FAST_PHASE = {"gravity_scale": 1.0, "upright_assist": 0.0, "max_steps": 6,
              "target_velocity": [0.0, 0.5, 0.0]}


class RecordingSink:
    """Collects every tap message (frames, resets, episode events)."""

    def __init__(self):
        self.messages: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, msg: dict) -> None:
        with self.lock:
            self.messages.append(msg)


def kinds(sink: RecordingSink) -> dict:
    out: dict = {}
    with sink.lock:
        for m in sink.messages:
            key = m.get("t", "?")
            out[key] = out.get(key, 0) + 1
    return out


def wait_state(sess, want, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sess.state == want:
            return True
        time.sleep(0.02)
    return False


def make_session(tmp_path, *, inference="deterministic", env_overrides=None,
                 name=None, action_repeat=1, seed_base=9_900_000, speed=1.0):
    from playground.rl_live.playback import PlaybackSession
    env_overrides = dict(env_overrides or {})
    env_overrides.setdefault("action_repeat", action_repeat)
    return PlaybackSession(
        checkpoint=str(CKPT), stage="fast_phase", phase=dict(FAST_PHASE),
        inference=inference, root=tmp_path / "pb_runs", name=name or "pb_test",
        env_overrides=env_overrides, seed_base=seed_base, speed=speed)


def wait_records(sess, want, timeout=25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(sess._episode_records) >= want:
            return True
        time.sleep(0.05)
    return False


def stopsess(sess):
    try:
        sess.playback_control("stop")
        wait_state(sess, "stopped", 10)
    except Exception:
        pass


def wire_sink(sess, sink: RecordingSink):
    """Wire the session the way the server's App does: the tap's frames AND
    the broadcast events (spec/status) both land in the sink."""
    sess._frame_sink = sink
    sess._broadcast = lambda kind, payload: sink({"t": kind, **payload})


@pytest.fixture(scope="module")
def need_ckpt():
    if not CKPT.exists():
        pytest.skip("b3_run1/ckpt_final.json not present on this tree")
    return CKPT


# ------------------------------------------------------- the B3-era rig seam
def test_extract_b3_rig_xml_verifies_the_recorded_sha(tmp_path):
    from playground.rl_live import playback as pb
    rig = pb.extract_b3_rig_xml(tmp_path / "rig")
    assert rig["sha256"] == pb.B3_ASSET_SHA256
    got = hashlib.sha256(Path(rig["path"]).read_bytes()).hexdigest()
    assert got == pb.B3_ASSET_SHA256


def test_extract_refuses_a_hash_mismatch(monkeypatch, tmp_path):
    from playground.rl_live import playback as pb
    monkeypatch.setattr(pb, "B3_ASSET_SHA256", "0" * 64)
    with pytest.raises(pb.PlaybackError, match="hash mismatch"):
        pb.extract_b3_rig_xml(tmp_path / "rig")


def test_verify_head_inertial_rejects_the_b4_head():
    from playground.rl_live import playback as pb
    from playground.config import load_config
    from playground.rl.env import LocoEnv
    cfg = load_config(None, overrides={})
    rl = dict(cfg.get("rl") or {})
    env = LocoEnv(cfg={"agent_body": {"type": "humanoid"}, **rl})
    bid = int(env.world.m.body("head").id)
    assert float(env.world.m.body_mass[bid]) != pb.B3_HEAD["mass"], \
        "the default asset must still be the B4-corrected head"
    with pytest.raises(pb.PlaybackError, match="NOT the B3-era head"):
        pb.verify_head_inertial(env.world.m)


# ------------------------------------------------------------- the session
def test_session_builds_and_verifies_the_b3_rig(need_ckpt, tmp_path):
    from playground.rl_live import playback as pb
    from playground.world import spec as world_spec
    sess = make_session(tmp_path)
    assert sess.start()["ok"] is True
    try:
        assert wait_state(sess, "running", 25), \
            f"state={sess.state} err={sess._error}"
        idt = sess.status()["session"]["identity"]
        rig = idt["rig"]
        assert rig["verified"] is True
        assert rig["sha256"] == pb.B3_ASSET_SHA256
        assert rig["compiled_head_mass_kg"] == pb.B3_HEAD["mass"]
        assert rig["compiled_head_ipos"] == list(pb.B3_HEAD["ipos"])
        assert rig["compiled_head_inertia"] == list(pb.B3_HEAD["inertia"])
        assert Path(str(world_spec.HUMANOID_ASSET_PATH)).name == \
            "humanoid_mujoco_b3_era.xml"
        assert Path(str(world_spec.HUMANOID_ASSET_PATH)).resolve() != \
            (REPO / "assets" / "humanoid_mujoco.xml").resolve()
        st = sess.status()["session"]
        assert st["config"]["inference"] == "deterministic"
        assert st["config"]["checkpoint"] == str(CKPT)
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)


def test_playback_streams_frames_and_records_episodes(need_ckpt, tmp_path):
    sink = RecordingSink()
    sess = make_session(tmp_path, name="pb_frames")
    wire_sink(sess, sink)                       # tap frames + broadcast events
    sess.start()
    try:
        assert wait_state(sess, "running", 25), sess._error
        assert wait_records(sess, 3, 30), (len(sess._episode_records),
                                           sess._error)
        ks = kinds(sink)
        assert ks.get("frame", 0) > 0, "the viewer feed carried no frames"
        assert ks.get("reset", 0) >= 3
        spec = [m for m in sink.messages if m.get("spec_version")]
        assert spec, "no scene spec broadcast (the rig could never be built)"
        assert any(b.get("rig") for b in spec[0]["bodies"])
        s0 = sess.status()["session"]
        assert s0["identity"]["actor_unmodified"] is True
        assert s0["identity"]["actor_sha256"] == \
            s0["identity"]["actor_sha256_start"]
        rec = s0["last_episode"]
        for key in ("falls", "faulted_windows", "physics_warnings",
                    "axis_displacement_m", "net_displacement_m",
                    "sim_time_s", "mode", "seed", "actor_sha256"):
            assert key in rec, key
        assert "NOT comparable" in rec["reward_total_note"]
        info = json.loads((tmp_path / "pb_runs" / "pb_frames"
                           / "playback_info.json").read_text(encoding="utf-8"))
        assert info["identity"]["content_sha256"]
        assert info["identity"]["rig"]["verified"] is True
        assert info["identity"]["actor_sha256_start"] == \
            s0["identity"]["actor_sha256_start"]
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)


def test_playback_controls_pause_resume_reset_speed(need_ckpt, tmp_path):
    sink = RecordingSink()
    sess = make_session(tmp_path, name="pb_controls", action_repeat=100,
                        speed=0.5)
    wire_sink(sess, sink)
    sess.start()
    try:
        assert wait_state(sess, "running", 25), sess._error
        assert wait_records(sess, 1, 25), sess._error
        sess.playback_control("pause")
        assert wait_state(sess, "paused", 5)
        frames_at_pause = kinds(sink).get("frame", 0)
        time.sleep(0.6)
        n1 = kinds(sink).get("frame", 0)
        # at most one in-flight decision can still land after the pause
        assert n1 - frames_at_pause <= 2, "frames kept flowing while paused"
        time.sleep(0.6)
        assert kinds(sink).get("frame", 0) == n1, \
            "frames kept flowing while the playback was paused"
        r = sess.playback_control("resume")
        assert r["ok"] is True and sess.status()["session"]["state"] == "running"
        deadline = time.time() + 20
        while time.time() < deadline and \
                not any(rec.get("reset_requested")
                        for rec in sess._episode_records):
            sess.playback_control("reset")    # presses can land inside the
            time.sleep(0.3)                   # setup window and be consumed
        assert any(rec.get("reset_requested") for rec in sess._episode_records), \
            [(r2.get("seed"), r2.get("reset_requested"))
             for r2 in sess._episode_records]
        r = sess.playback_control("speed", {"speed": 2.5})
        assert r["ok"] is True and sess.status()["session"]["speed"] == 2.5
        assert sess.playback_control("speed", {"speed": 9.0})["ok"] is False
        assert sess.playback_control("nonsense")["ok"] is False
        assert wait_records(sess, 4, 40)
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)


def test_inference_modes_are_explicit_and_labelled(need_ckpt, tmp_path):
    sess = make_session(tmp_path, inference="deterministic", name="pb_mode")
    sess.start()
    try:
        assert wait_state(sess, "running", 25)
        assert sess.status()["session"]["mode"] == "deterministic"
        r = sess.playback_control("mode", {"mode": "stochastic"})
        assert r["ok"] is True and r["mode_next"] == "stochastic"
        # the running episode stays labelled until the boundary
        assert sess.status()["session"]["mode"] == "deterministic"
        deadline = time.time() + 25
        while time.time() < deadline:
            newst = sess.status()["session"]
            if newst["mode"] == "stochastic" and newst["mode_next"] is None                     and newst["last_episode"]                     and newst["last_episode"]["mode"] == "stochastic":
                break
            time.sleep(0.05)
        newst = sess.status()["session"]
        assert newst["mode"] == "stochastic" and newst["mode_next"] is None
        assert newst["last_episode"]["mode"] == "stochastic"
        assert sess.playback_control("mode", {"mode": "quantum"})["ok"] is False
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)


def test_modes_and_replays(need_ckpt, tmp_path):
    """A parked reset discards the in-flight episode and replays its OWN seed
    from the start (never advances the seed); deterministic and stochastic
    inference are labelled per episode and the mode matters: the same seed
    under stochastic sampling (sigma as trained) produces a different episode
    than the deterministic mean action. The deterministic same-seed identity
    itself is proven in test_deterministic_same_seed_replays_identically.
    (Deterministic replays are the recorded evaluation convention; the
    mode comparisons below use the as-trained sigma.)"""
    det_rows = {}
    sink = RecordingSink()
    # slow episodes (0.4 s decisions, ~2.4 s per episode) so the parked
    # mid-episode reset is deterministic, not a timing race
    sess = make_session(tmp_path, inference="deterministic", name="pb_det",
                        action_repeat=100, speed=0.5)
    wire_sink(sess, sink)
    sess.start()
    try:
        assert wait_state(sess, "running", 25)
        assert wait_records(sess, 1, 25)
        first = dict(sess._episode_records[0])   # a full ep0 (6 decisions)
        det_rows[first["seed"]] = first["net_displacement_m"]

        def park_mid_episode_and_reset():
            """Park mid-episode (verified), press reset, resume; returns
            (partial_row, replay_row) drawn only from rows appended AFTER the
            press, with bounded retries when a press is consumed at a fresh
            episode's start (nothing measured -> skipped, replayed)."""
            for _attempt in range(5):
                deadline = time.time() + 10
                while time.time() < deadline and sess._tap._decision < 1:
                    time.sleep(0.05)
                if sess._tap._decision < 1:
                    continue
                sess.playback_control("pause")
                if not wait_state(sess, "paused", 5):
                    continue
                if sess._tap._decision < 1:      # parked at a fresh episode
                    sess.playback_control("resume")
                    continue
                n0 = len(sess._episode_records)
                sess.playback_control("reset")
                sess.playback_control("resume")
                deadline = time.time() + 25
                while time.time() < deadline:
                    rows_ = list(sess._episode_records)[n0:]
                    partial = next((r for r in rows_
                                    if r.get("reset_requested")), None)
                    if partial is not None:
                        idx = rows_.index(partial)
                        replay = next((r for r in rows_[idx + 1:]
                                       if r["seed"] == partial["seed"]
                                       and not r.get("reset_requested")
                                       and r["steps"] == 6), None)
                        if replay is not None:
                            return partial, replay
                    time.sleep(0.05)
            raise AssertionError("parked reset never produced "
                                 "partial+replay rows: "
                                 f"{sess._episode_records} err={sess._error}")

        partial1, full1 = park_mid_episode_and_reset()
        assert 0 < partial1["steps"] < 6
        assert partial1["reset_requested"] is True
        # the parked reset discards the in-flight episode and replays THAT
        # SAME seed from its start (never advances the seed)
        assert full1["steps"] == 6 and full1["seed"] == partial1["seed"]
        assert full1["reset_requested"] is False
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)

    sto_rows = {}
    sess = make_session(tmp_path, inference="stochastic", name="pb_sto",
                        action_repeat=100, speed=0.5)
    sess.start()
    try:
        assert wait_state(sess, "running", 25)
        assert wait_records(sess, 2, 25)
        rows_ = sess._episode_records[:2]
        sto_rows[rows_[0]["seed"]] = rows_[0]["net_displacement_m"]
        # (a) different mode, same seed: sampled episode differs from mean
        assert rows_[0]["seed"] in det_rows, (rows_[0]["seed"], det_rows)
        assert sto_rows[rows_[0]["seed"]] != det_rows[rows_[0]["seed"]]
        # (b) the two stochastic episodes (different sampling seeds) differ
        assert rows_[1]["seed"] != rows_[0]["seed"]
        assert rows_[1]["net_displacement_m"] != rows_[0]["net_displacement_m"]
        assert all(r2["mode"] == "stochastic" for r2 in rows_)
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)


def test_rig_seam_restored_and_default_rig_untouched_after_stop(
        need_ckpt, tmp_path):
    """The rig seam is a PROCESS-GLOBAL constant: if a session leaked the
    B3-era path, every later humanoid build in the same process would run on
    the wrong asset (this exact leak broke the B1 combat test's cap-engagement
    expectation in the full suite). Both the seam constant and a fresh
    default-rig world must be back to normal once the session ends."""
    from playground.world import spec as world_spec
    before = str(world_spec.HUMANOID_ASSET_PATH)
    sess = make_session(tmp_path, name="pb_seam_restore")
    assert sess.start()["ok"] is True
    try:
        assert wait_state(sess, "running", 25)
        assert wait_records(sess, 2, 30)
        assert str(world_spec.HUMANOID_ASSET_PATH) != before
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)
    assert str(world_spec.HUMANOID_ASSET_PATH) == before, \
        "the B3-era rig seam leaked past the session's end"

    # the interaction proof: a fresh default-rig world AFTER playback must
    # compile the current (B4) head again
    from playground.config import load_config
    from playground.rl.env import LocoEnv
    cfg = load_config(None, overrides={})
    rl = dict(cfg.get("rl") or {})
    env = LocoEnv(cfg={"agent_body": {"type": "humanoid"}, **rl})
    bid = int(env.world.m.body("head").id)
    assert float(env.world.m.body_mass[bid]) == 1.8, \
        "a build after playback did not use the default asset"


# ------------------------------------------------- the read-only guarantees
def test_deterministic_same_seed_replays_identically(need_ckpt):
    """Race-free proof of the deterministic replay identity: two full
    episodes of the SAME seed under the mean action reproduce the identical
    trajectory (same rig, same loader, THE deterministic convention)."""
    import torch
    torch.set_num_threads(1)
    from playground.config import load_config
    from playground.rl.curriculum import DEFAULT_STAGES
    from playground.rl.env import LocoEnv
    from playground.rl.checkpoint import load_checkpoint
    from playground.rl_live import playback as pb

    rig = pb.extract_b3_rig_xml(REPO / "experiments" / "rl_playback"
                                / "_pytest_rig")
    from playground.world import spec as world_spec
    prev = str(world_spec.HUMANOID_ASSET_PATH)
    world_spec.HUMANOID_ASSET_PATH = rig["path"]
    try:
        cfg = load_config(None, overrides={})
        rl = dict(cfg.get("rl") or {})
        # M7-0a: pin the recorded brain's contract (same helper the playback
        # session uses) — the v3 lineage replays on the v3 legacy mode.
        _cv = pb.read_checkpoint_contract(CKPT)
        if _cv in (3, 4):
            rl["contract_version"] = _cv
        env = LocoEnv(cfg={"agent_body": {"type": "humanoid"}, "rl": rl})
        env.set_phase(dict(DEFAULT_STAGES[2].phase))
        env.set_phase({"gravity_scale": 1.0, "upright_assist": 0.0,
                       "max_steps": 6, "target_velocity": [0.0, 0.5, 0.0]})
        ck = load_checkpoint(str(CKPT), env=env)
        actor = ck["actor"]
        nets = []
        for _ in range(2):
            obs = env.reset(seed=9_900_123)
            steps = 0
            while steps < 6:
                a = actor.deterministic_action(obs)
                obs, _r, term, trunc, info = env.step(a)
                steps += 1
                if term or trunc:
                    break
            s = env.episode_summary()
            nets.append((s["net_displacement_m"], s["final_position"]))
    finally:
        world_spec.HUMANOID_ASSET_PATH = prev
    assert nets[0] == nets[1], "deterministic same-seed replay diverged"


def test_no_ppo_path_and_weights_never_change(need_ckpt, tmp_path):
    from playground.rl_live import playback as pb
    # nothing in the module's namespace can train: no trainer, no optimizers
    assert not any("PPOTrainer" in k for k in vars(pb))
    assert not any("optimizer" in k.lower() for k in vars(pb))
    sess = make_session(tmp_path, name="pb_weights")
    sess.start()
    try:
        assert wait_state(sess, "running", 25)
        assert wait_records(sess, 3, 30)
        idt = sess.status()["session"]["identity"]
        assert idt["actor_unmodified"] is True
        assert idt["actor_sha256"] == idt["actor_sha256_start"]
        assert all(r2["actor_sha256"] == idt["actor_sha256_start"]
                   for r2 in sess._episode_records)
        assert sess._tap._env is not None
        # optimiser state is dropped at load: the session retains none
        assert not hasattr(sess._actor, "_optimizer")
        assert getattr(sess, "_optimizer_actor", None) is None
    finally:
        stopsess(sess)
        wait_state(sess, "stopped", 10)
        idt = sess.status()["session"]["identity"]
        assert idt.get("actor_unmodified") is True
        assert idt.get("actor_sha256_end") == idt["actor_sha256_start"]


def test_playback_never_writes_b3_run1(need_ckpt, tmp_path):
    b3 = REPO / "experiments" / "rl" / "b3_run1"

    def snap(root: Path):
        out = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                st = p.stat()
                out[str(p)] = (st.st_size, st.st_mtime_ns)
        return out

    before = snap(b3)
    sess = make_session(tmp_path, name="pb_ro")
    sess.start()
    try:
        assert wait_state(sess, "running", 25)
        assert wait_records(sess, 2, 25)
    finally:
        stopsess(sess)
    after = snap(b3)
    assert after == before, "the playback session wrote to b3_run1"
