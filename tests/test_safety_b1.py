"""M6-B stage B1 — physical safety foundation.

Covers the B1 safety matrix:
- NaN / ±Infinity / invalid target, kp, kd are REJECTED at the world boundary with
  structured errors, never stored, physics stays finite.
- Extremely large finite targets are saturated into the joint's real range; excessive
  gains are clamped to configured ceilings; both are observable (events + counters).
- Per-joint torque caps bound the ACTUAL force written to d.qfrc_applied, even for a
  hostile controller, and MuJoCo numerical warnings stay at zero under capped control.
- BADQACC (silent physics suppression) surfaces as a physics_warning event when a
  rogue caller bypasses validation, and the env marks the transition FAULTED.
- The env rejects non-finite/wrong-shaped neural actions before any physical effect
  and records finite out-of-range CLIPPING frequency.
- The tool boundary returns structured errors and never reaches the world.
- Regression: scripted walking is unchanged under the caps; recovery still completes;
  recovery after a corrected target works.

Cap values are NOT invented: they sit above the measured scripted operating envelope
(reports/m6b/B1_SAFETY.md, probe_torque_results.json: ankle 326.7, hip 310, knee 243.4
N·m peaks across stand/walk/run/turn/recovery) and below the ~1e6 N·m runaway scale.
"""
import math

import numpy as np
import pytest

from playground.config import load_config
from playground.interface.schema import ToolError, normalise_args
from playground.world import PhysicsWorld
from playground.world.physics import ActionError
from playground.rl.env import LocoEnv


def humanoid_cfg(**over):
    return load_config(overrides={"agent_body": {"type": "humanoid"}, **over})


@pytest.fixture()
def hw():  # humanoid world, default safety ON
    return PhysicsWorld(humanoid_cfg())


def _badqacc(w):
    return w.physics_warnings().get("mjWARN_BADQACC", 0)


# ---------------------------------------------------------------- rejection
@pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_target_rejected(hw, val):
    with pytest.raises(ActionError) as e:
        hw.set_joint_target("left_hip", val, kp=60.0, kd=5.0)
    assert e.value.code == "invalid_joint_target"
    assert "left_hip" not in hw._pd
    assert hw.physics_warnings() == {}
    # world state untouched and finite; a corrected command afterwards works
    hw.set_joint_target("left_hip", 0.1, kp=60.0, kd=5.0)
    hw.step(1)
    assert "left_hip" in hw._pd
    assert _badqacc(hw) == 0


@pytest.mark.parametrize("param,val", [("kp", float("nan")), ("kp", float("inf")),
                                       ("kd", float("-inf")), ("kd", float("nan")),
                                       ("position", float("inf"))])
def test_non_finite_gains_rejected(hw, param, val):
    args = {"position": 0.0, "kp": 60.0, "kd": 5.0}
    args[param] = val
    with pytest.raises(ActionError):
        hw.set_joint_target("left_hip", **args)
    assert "left_hip" not in hw._pd
    assert hw.safety_stats()["target_rejects"] >= 1


def test_invalid_joint_name_rejected(hw):
    with pytest.raises(ActionError):
        hw.set_joint_target(None, 0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------- saturation
def test_huge_finite_target_torque_capped():
    """1e30 target (the audit's BADQACC trigger) can never reach the solver at
    dangerous force: the per-joint torque caps bound the written force regardless
    of the target. Default config passes the huge finite target through unclamped
    (documented, observable; MuJoCo's own limit stops govern the joint); enabling
    `clamp_target` saturates it into jnt_range instead."""
    w = PhysicsWorld(humanoid_cfg())  # default: caps ON, target saturation OFF
    w.set_joint_target("left_hip", 1e30, kp=60.0, kd=5.0)
    assert w._pd["left_hip"]["target"] == 1e30  # passed through, not silently rewritten
    dof = int(w.m.jnt_dofadr[w.m.joint("left_hip").id])
    w.step(1)
    assert abs(float(w.d.qfrc_applied[dof])) <= 400.0 + 1e-6  # hip cap bounds the force
    assert w.physics_warnings() == {}
    assert not np.isnan(w.d.qpos).any()

    sat = humanoid_cfg(**{"world": {"safety": {"clamp_target": True}}})
    w2 = PhysicsWorld(sat)
    w2.set_joint_target("left_hip", 1e30, kp=60.0, kd=5.0)
    lo, hi = (float(w2.m.jnt_range[w2.m.joint("left_hip").id][0]),
              float(w2.m.jnt_range[w2.m.joint("left_hip").id][1]))
    assert lo - 1e-9 <= w2._pd["left_hip"]["target"] <= hi + 1e-9
    dof2 = int(w2.m.jnt_dofadr[w2.m.joint("left_hip").id])
    w2.step(1)
    assert abs(float(w2.d.qfrc_applied[dof2])) <= 400.0 + 1e-6
    kinds = {e["type"] for e in w2.safety_log()}
    assert "joint_target_clamped" in kinds


def test_excessive_gains_clamped(hw):
    hw.set_joint_target("left_hip", 1.5, kp=1e6, kd=1e4)
    assert hw._pd["left_hip"]["kp"] == 1000.0   # configured ceiling
    assert hw._pd["left_hip"]["kd"] == 100.0
    assert hw.safety_stats()["gain_clamps"] >= 1
    dof = int(hw.m.jnt_dofadr[hw.m.joint("left_hip").id])
    hw.step(1)
    assert abs(float(hw.d.qfrc_applied[dof])) <= 400.0 + 1e-6


def test_negative_gains_clamped_to_zero(hw):
    """A negative gain would be positive feedback; it clamps to 0, never rejects
    (reject-only is reserved for non-finite values)."""
    hw.set_joint_target("left_hip", 0.0, kp=-5.0, kd=-2.0)
    assert hw._pd["left_hip"]["kp"] == 0.0
    assert hw._pd["left_hip"]["kd"] == 0.0


def test_torque_cap_resolution():
    w = PhysicsWorld(humanoid_cfg())
    assert w._torque_cap("left_hip") == 400.0          # left_/right_ family
    assert w._torque_cap("spine_L2") == 60.0           # exact
    assert w._torque_cap("head_pitch") == 20.0         # head_* family
    assert w._torque_cap("left_foot_yaw") == 20.0
    assert w._torque_cap("waist_yaw") == 120.0
    assert w._torque_cap("left_shoulder") == 160.0     # above the 124.3 N·m combat swing peak
    assert w._torque_cap("left_elbow") == 220.0        # above the 166.3 N·m combat swing peak
    assert w._torque_cap("left_knee") == 570.0         # above the measured catch-stage peak
    assert w._torque_cap("some_unlisted") == 600.0     # default
    assert w._torque_cap("agent_002:left_knee") == 570.0  # multi-agent prefix


def test_torque_cap_property_random_hostile_policy():
    """A hostile controller (random over-range targets, over-ceiling gains) for 1000
    steps can never move |qfrc_applied| past the per-joint cap, and zero numerical
    warnings accumulate. Saturation must be OBSERVED and RECORDED with per-source
    attribution."""
    w = PhysicsWorld(humanoid_cfg())
    names = ["spine_L2", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "head_pitch", "head_yaw", "head_roll", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle",
             "waist_yaw", "left_foot_yaw", "right_foot_yaw"]
    dof = {n: int(w.m.jnt_dofadr[w.m.joint(n).id]) for n in names}
    # B4 rig (Jamie decision 2026-11-18, head-only gain cap in the test rng):
    # the fixed head's hinge inertia is 0.0404 kg·m², ~13x below the old rig's.
    # A hostile HEAD command at extreme kp can throw qacc past MuJoCo's
    # huge-value threshold before any cap matters (measured repro
    # ai-playground-audit/b4_hostile_qacc_repro.py: kp=2240, kd=+253 fired
    # qacc ~8.4e4 rad/s² vs ~6.5e3 on the old rig, transient, self-recovering;
    # training head gains are kp=8, four orders of magnitude below, unreachable
    # by any policy). The hostile stream therefore clamps HEAD kp at 400 —
    # 50x the training gain, ~5x below the measured crossing — while every
    # other joint stays fully hostile (kp to 4000, over-range targets, negative
    # kd). The zero-warning assertion below is RETAINED unchanged: head kp=400
    # with the same kd draws produces zero warnings (verified, rng stream
    # unchanged — the clamp is applied after sampling, no extra draws).
    head_kp_ceiling = 400.0
    rng = np.random.default_rng(20261117)
    for step in range(1000):
        if step % 7 == 0:
            n = names[int(rng.integers(len(names)))]
            lo, hi = (float(w.m.jnt_range[w.m.joint(n).id][0]),
                      float(w.m.jnt_range[w.m.joint(n).id][1]))
            kp = float(rng.uniform(-50.0, 4000.0))
            if n.startswith("head"):
                kp = min(kp, head_kp_ceiling)
            w.set_joint_target(n, float(rng.uniform(lo - 3.0, hi + 3.0)),
                               kp=kp,
                               kd=float(rng.uniform(-10.0, 500.0)), source="hostile_policy")
        w.step(1)
        for n, d in dof.items():
            t = abs(float(w.d.qfrc_applied[d]))
            cap = w._torque_cap(n)
            if cap is not None:
                assert t <= cap + 1e-6, f"{n}: {t} > cap {cap} at step {step}"
    assert w.physics_warnings() == {}
    sat = w.safety_stats()["torque_saturation"]
    assert sat, "a hostile policy should have saturated at least one joint cap"
    for jname, s in sat.items():
        assert s["steps_saturated"] >= s["events"]
        assert s["max_raw"] >= w._torque_cap(jname)
        assert "hostile_policy" in s["sources"]
    # scripted demands stay distinguishable from the hostile source
    assert any("hostile_policy" in s["sources"] for s in sat.values())


# ---------------------------------------------------------------- env boundary
def test_env_rejects_non_finite_actions():
    env = LocoEnv(humanoid_cfg())
    env.reset()
    for bad in (float("nan"), float("inf"), float("-inf")):
        a = np.zeros(LocoEnv.action_dim, dtype=np.float32)
        a[3] = bad
        with pytest.raises(ActionError) as e:
            env.step(a)
        assert e.value.code == "non_finite_action"
        assert "left_knee" not in env.world._pd or env.world._pd["left_knee"]["target"] == 0.0
    assert env.clip_stats["rejected_calls"] == 3
    assert env.world.physics_warnings() == {}
    # a corrected action steps cleanly afterwards
    obs, r, term, trunc, info = env.step(np.zeros(LocoEnv.action_dim, dtype=np.float32))
    assert obs.shape == (LocoEnv.obs_dim,) and not info.get("faulted")


def test_nan_neural_network_output_rejected():
    """A network whose output is NaN (poisoned weights) hits the guard: rejected with
    a structured error before any physical effect. Note: this rig's policy head ends
    in tanh, which maps infinite pre-activations to the finite boundary ±1 — a
    tanh-output policy can emit NaN but not ±Inf; the ±Inf case is exercised directly
    on the boundary (an upstream head that bypasses tanh, e.g. an exploded linear
    output layer)."""
    torch = pytest.importorskip("torch")

    class _TinyPolicy(torch.nn.Module):
        """Minimal poisoned-weights stand-in (the old rl.train.Policy is gone:
        B2 deleted the broken pseudo-trainer; the guard's contract is what is
        tested here, not that class)."""

        def __init__(self, obs_dim: int, act_dim: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 16), torch.nn.Tanh(),
                torch.nn.Linear(16, act_dim), torch.nn.Tanh())

        def forward(self, x):
            return self.net(x)

    env = LocoEnv(humanoid_cfg())
    env.reset()
    policy = _TinyPolicy(LocoEnv.obs_dim, LocoEnv.action_dim)
    with torch.no_grad():
        policy.net[-2].weight[0, 0] = float("nan")     # NaN output ([-1] is the output Tanh)
    with pytest.raises(ActionError):
        env.step(policy(torch.from_numpy(env._get_obs()).float().unsqueeze(0)).detach()
                 .squeeze(0).numpy())
    # ±Inf from a (hypothetical) untanh'd controller head
    inf_action = np.zeros(LocoEnv.action_dim, dtype=np.float32)
    inf_action[2] = float("inf")
    inf_action[5] = float("-inf")
    with pytest.raises(ActionError) as e:
        env.step(inf_action)
    assert e.value.code == "non_finite_action"
    assert env.world.physics_warnings() == {}
    assert not np.isnan(env.world.d.qpos).any()
    assert env.clip_stats["rejected_calls"] == 2  # one NaN call + one ±Inf call


def test_env_rejects_wrong_action_dimensions():
    env = LocoEnv(humanoid_cfg())
    env.reset()
    for shape in ((10,), (15,), (14, 1)):
        with pytest.raises(ActionError) as e:
            env.step(np.zeros(shape, dtype=np.float32))
        assert e.value.code == "invalid_action_shape"
    assert env.clip_stats["rejected_calls"] == 3


def test_env_clips_and_counts_out_of_range_actions():
    env = LocoEnv(humanoid_cfg())
    env.reset()
    a = np.zeros(LocoEnv.action_dim, dtype=np.float32)
    a[0] = 2.5
    a[1] = -3.0
    obs, r, term, trunc, info = env.step(a)
    assert not info.get("faulted")
    cs = env.action_clipping()
    assert cs["elements_clipped"] == 2
    assert cs["calls_with_clipping"] == 1
    assert cs["max_elements_in_call"] == 2
    # clipped to the encoding boundary: each over-range joint target == its range limit
    for i in (0, 1):
        name = env.joint_names[i]
        low, high = env.joint_range[name]
        bound = high if i == 0 else low
        assert env.world._pd[name]["target"] == pytest.approx(bound, abs=1e-6)
    # clipping rate is recorded and declines once the policy behaves
    for _ in range(2):
        obs, r, term, trunc, info = env.step(np.zeros(LocoEnv.action_dim, dtype=np.float32))
    assert env.action_clipping()["clipping_rate"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------- BADQACC watchdog
def test_badqacc_watchdog_detects_and_attributes():
    """A rogue caller that registers a BADQACC-scale command through the world API
    with clamps/caps disabled (the M6-A audit's demonstrated failure: kp=1e6, target
    1e30) is no longer silent: the watchdog surfaces mjWARN_BADQACC as a structured
    event attributed to the owning agent and the commanding source. Attribution
    degrades honestly to ['none'] when a caller bypassed the API entirely."""
    cfg = humanoid_cfg(**{"world": {"safety": {"enforce_torque_limits": False,
                                               "clamp_target": False,
                                               "kp_ceiling": 1e9, "kd_ceiling": 1e9}}})
    w = PhysicsWorld(cfg)
    w.set_joint_target("left_hip", 1e30, kp=1e6, kd=0.0, source="bypass_probe")
    w.step(1)
    warns = w.physics_warnings()
    assert warns, "BADQACC-scale command must produce a numerical warning"
    assert "mjWARN_BADQACC" in warns
    kinds = [e for e in w.safety_log() if e["type"] == "physics_warning"]
    assert kinds and kinds[-1]["agents"] == ["agent_001"]  # attribution: the rig
    assert kinds[-1]["sources"] == ["bypass_probe"]

    w2 = PhysicsWorld(cfg)
    w2._pd["left_hip"] = {"target": 1e30, "kp": 1e6, "kd": 0.0, "source": "rogue"}
    w2.step(1)
    log2 = [e for e in w2.safety_log() if e["type"] == "physics_warning"]
    assert log2 and log2[-1]["agents"] == ["none"]


def test_env_flags_faulted_after_numerical_warning():
    """The env marks the transition FAULTED after MuJoCo numerically suppressed a
    step (rogue direct-PD write on a joint the env does not command, mid-episode),
    so it cannot enter training data."""
    cfg = humanoid_cfg(**{"world": {"safety": {"enforce_torque_limits": False,
                                               "clamp_target": False,
                                               "kp_ceiling": 1e9, "kd_ceiling": 1e9}}})
    env = LocoEnv(cfg)
    env.reset()
    env.world._pd["waist_yaw"] = {"target": 1e30, "kp": 1e6, "kd": 0.0, "source": "rogue"}
    obs, r, term, trunc, info = env.step(np.zeros(LocoEnv.action_dim, dtype=np.float32))
    assert term and info.get("faulted") is True
    assert "mjWARN_BADQACC" in info["physics_warnings"]
    assert env.faulted()
    assert r == 0.0  # faulted transition carries a zero training reward


# ---------------------------------------------------------------- observability
def test_safety_surface_observable():
    """Counters rise per control kind; the clamp events land in the bounded log."""
    hw = PhysicsWorld(humanoid_cfg(**{"world": {"safety": {"clamp_target": True}}}))
    with pytest.raises(ActionError):
        hw.set_joint_target("left_hip", float("nan"))
    hw.set_joint_target("left_hip", 1e6, kp=1e6)
    stats = hw.safety_stats()
    assert stats["target_rejects"] >= 1
    assert stats["target_clamps"] >= 1
    assert stats["gain_clamps"] >= 1
    for e in hw.safety_log():
        assert {"step", "t", "type"} <= set(e)
    # sources are recorded on the commands themselves
    hw.set_joint_target("left_knee", 0.0, source="neural_policy")
    assert hw._pd["left_knee"]["source"] == "neural_policy"


# ---------------------------------------------------------------- regressions
def test_combat_swing_unchanged_under_caps():
    """Regression RE-RECORDED for the B4 rig (2026-11-18, Jamie decision): the
    M4 melee strike remains an existing legitimate controller, but on the B4
    rig (head inertial fix) the strike's raw swing transient now crosses BOTH
    arm caps of the striking arm — measured raw peaks ~953 N·m (left_elbow) and
    ~744 N·m (left_shoulder) versus the pre-B4 premise of ~124/~166 N·m that
    cleared them (probe: ai-playground-audit/b4_combat_rerecord_probe.py, JSON
    b4_combat_rerecord_probe.json; failing premise documented in
    b4_fullsuite_gate_pre.txt). The caps correctly bind during the transient,
    so the capped run is no longer bit-identical to the uncapped one. The
    honest contract is the measured re-record: the strike still lands, the
    holder is not KO'd, the OPPONENT impulse is unchanged within the measured
    0.02 %, the SELF impulse drops by the measured cap-bound amount
    (182.6 -> 143.2 N·s at t=0.47), saturation attribution is recorded, and
    ZERO physics warnings accumulate both ways."""
    import sys
    sys.path.insert(0, "tests")
    import test_combat_match as tcm
    from playground.world.physics import PhysicsWorld

    def run(safety_off):
        cfg = tcm._combat_cfg()
        if safety_off:
            cfg["world"]["safety"] = {"enforce_torque_limits": False, "clamp_target": False,
                                      "kp_ceiling": 1e9, "kd_ceiling": 1e9}
        w = PhysicsWorld(cfg)
        w._agents["agent_001"]._climb = None
        tools = tcm._tools(w, "agent_001")
        tools.equip("baton_01")
        out1 = tools.attack([0.0, 1.0, 0.0])
        tools.attack([0.0, 1.0, 0.0])
        for _ in range(int(1.5 / w.dt)):
            w.step(1)
        out3 = tools.attack([0.0, 1.0, 0.0])
        return (out1["contacts"], out3["attacked"], w._agents["agent_001"].incapacitated,
                w.safety_stats()["torque_saturation"], w.physics_warnings())

    c_off = run(True)
    c_on = run(False)

    def impulse(contacts, target):
        vals = [c["impulse"] for c in contacts
                if list(c["with"]) == [target] and abs(c["t"] - 0.47) < 0.01]
        assert len(vals) == 1, f"expected exactly one {target} contact record at t=0.47: {vals}"
        return vals[0]

    assert c_on[1] == c_off[1] is True, "the strike must land under caps"
    assert c_on[2] is False, "the strike must not KO its own holder"
    assert c_off[3] == {}, "caps-off run must saturate nothing"
    sat = c_on[3]
    assert set(sat) == {"agent_001:left_elbow", "agent_001:left_shoulder"}, \
        f"cap engagement diverged from the B4 re-record: {sorted(sat)}"
    for jname, s in sat.items():
        assert s["events"] >= 1
        assert s["max_raw"] > 160.0, f"{jname}: raw demand never crossed the 160 N·m cap"
        assert s["sources"], f"{jname}: saturation must carry attribution"
    assert c_on[4] == {} and c_off[4] == {}, "no physics warnings under caps or without them"
    assert impulse(c_on[0], "agent_002") == \
        pytest.approx(impulse(c_off[0], "agent_002"), rel=5e-4), \
        "the opponent impulse must be unchanged within the measured 0.02 %"
    a1_off = impulse(c_off[0], "agent_001")
    a1_on = impulse(c_on[0], "agent_001")
    assert a1_off == pytest.approx(182.6, abs=3.0) and a1_on == pytest.approx(143.2, abs=3.0), \
        f"strike re-record drifted: caps-on {a1_on}, caps-off {a1_off}"
def test_scripted_walk_unchanged_under_caps():
    """Audit regression (d): with the caps ENABLED (defaults) a scripted walk must be
    bit-identical in displacement to the SAME world with caps disabled: every scripted
    demand sits below the measured caps, so no torque is ever clamped."""
    base = humanoid_cfg()
    off = load_config(overrides={"agent_body": {"type": "humanoid"},
                                 "world": {"safety": {"enforce_torque_limits": False,
                                                      "clamp_target": False}}})

    def walk_world(cfg):
        w = PhysicsWorld(cfg)
        w.set_gait("walk", speed=1.0)
        start = w.agent_pos().copy()
        t0 = w.time
        while w.time - t0 < 8.0:
            w.step(1)
        d = float(np.hypot(w.agent_pos()[0] - start[0], w.agent_pos()[1] - start[1]))
        return d, w

    d_on, w_on = walk_world(base)
    d_off, w_off = walk_world(off)
    assert d_on > 0.3, f"walk displacement regression: {d_on:.3f} m in 8 s"
    assert abs(d_on - d_off) < 1e-6, f"caps changed scripted walk: {d_on} vs {d_off}"
    # nothing saturated during the honest walk
    assert w_on.safety_stats()["torque_saturation"] == {}


def test_scripted_recovery_still_works_under_caps():
    """Regression: collapse + stand_up recovery completes with the caps enabled.
    Collapse uses the repo's canonical fold (test_locomotion: knee -1.1 / hip 0.9 at
    kp 150 after a stand phase). The collapse assert is on the MEASURED pose (the
    debounced agent_fallen flag flaps near the height threshold while the folded rig
    bounces); the recovery assert is the sustained standing outcome. With the knee
    cap above the measured 452 N·m catch-stage peak, NOTHING should saturate."""
    w = PhysicsWorld(humanoid_cfg())
    w.set_gait('stand')
    for _ in range(int(5.0 / w.dt)):
        w.step(1)
    w._gait_controller = None
    fold = [("left_knee", -1.1), ("right_knee", -1.1), ("left_hip", 0.9), ("right_hip", 0.9)]
    for _ in range(int(4.0 / w.dt)):
        for j, t in fold:
            w.set_joint_target(j, t, kp=150.0, kd=6.0)
        w.step(1)
    assert float(w.agent_pos()[2]) < 0.35, "collapse premise failed (still standing)"
    w.stand_up()
    for _ in range(int(9.0 / w.dt)):
        w.step(1)
    assert not w.agent_fallen(), f"agent did not stand up (z={w.agent_pos()[2]:.3f})"
    assert w.agent_pos()[2] >= 0.55, f"recovery plateau too low: z={w.agent_pos()[2]:.3f}"
    for _ in range(int(1.5 / w.dt)):  # posture hold keeps it up
        w.step(1)
    assert not w.agent_fallen(), "agent sagged back down after recovery"
    assert w.safety_stats()["torque_saturation"] == {}, \
        "scripted recovery must not clip legitimately under the configured caps"


def test_recovery_after_corrected_target(hw):
    """After a rejected non-finite command the world accepts a corrected finite
    command and runs clean (no suppression, finite state)."""
    with pytest.raises(ActionError):
        hw.set_joint_target("left_knee", float("inf"))  # type: ignore[arg-type]
    hw.set_joint_target("left_knee", -0.2, kp=60.0, kd=5.0)
    hw.step(25)
    assert hw.physics_warnings() == {}
    assert np.isfinite(hw.d.qpos).all() and np.isfinite(hw.d.qvel).all()


# ---------------------------------------------------------------- tool boundary
def test_tool_boundary_rejects_non_finite(make_env):
    """Audit regression (e): through Environment.call (the LLM/HTTP path) a NaN
    position returns a STRUCTURED error and never reaches the world (the world's PD
    dict for that joint is unchanged)."""
    env = make_env()
    before = dict(env.world._pd)
    res = env.call("set_joint_target", {"joint": "left_hip",
                                        "position": float("nan"), "kp": 60.0, "kd": 5.0})
    assert res["success"] is False
    assert res["error"]["code"] in ("invalid_joint_target", "invalid_argument")
    assert res["status"] in ("failed", "partial")
    assert env.world._pd == before


def test_tool_boundary_rejects_json_nan_literal(make_env):
    """JSON admits NaN literals: the schema validator must catch them before the tool
    body runs."""
    env = make_env()
    res = env.call("set_joint_target", '{"joint": "left_hip", "position": NaN}')
    assert res["success"] is False


def test_normalise_args_finiteness():
    for bad in (float("nan"), float("-inf")):
        with pytest.raises(ToolError) as e:
            normalise_args("set_joint_target",
                           {"joint": "left_hip", "position": 0.0, "kp": bad})
        assert e.value.code == "invalid_argument"


def test_tool_gains_default_to_world_defaults():
    """Audit defect #10: a bare tool call used to register a zero-gain (force-less)
    PD entry; it now takes the world defaults (15/1.5) and is tagged source='tool'."""
    import mujoco
    from playground.interface.physical import PhysicalTools
    w = PhysicsWorld(humanoid_cfg())
    joint = next(mujoco.mj_id2name(w.m, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(w.m.njnt)
                 if int(w.m.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_HINGE))
    tools = PhysicalTools(w)
    res = tools.set_joint_target(joint, 0.0)
    assert res["kp"] is None  # ack reports None (world default applied), not a fake 0
    assert w._pd[joint]["kp"] == 15.0
    assert w._pd[joint]["kd"] == 1.5
    assert w._pd[joint]["source"] == "tool"
