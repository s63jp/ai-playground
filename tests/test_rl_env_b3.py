"""M6-B stage B3 — observation contract v3 (task conditioning) battery.

B3 resolved the task-conditioning problem: the policy must OBSERVE what
movement it is being asked to perform. v3 appends the commanded target
velocity (the exact quantity the reward tracks), WORLD frame, scaled by the
torso-velocity scale. These tests pin:

- the append-only relationship to v2 (every v2 slice unchanged, semantics
  recomputed from engine state by the documented formulas),
- command observability: values, scale, determinism, finiteness,
- curriculum conditioning: the stage ladder is visible through the command
  alone (no task-id input needed), including the B3 measured stage data
  (stages command the walkable chest axis, world +Y at spawn).

Reference evidence: ai-playground-audit/probe_b3_yaw.py (policy-reachable
locomotion = chest axis; waist_yaw is the rig's only rotational root dof and
is held by the env posture PD, outside the 14-action space).
"""
import numpy as np
import pytest

from playground.rl.curriculum import DEFAULT_STAGES, Curriculum
from playground.rl.env import LocoEnv


def _zero(env):
    return np.zeros(env.action_dim, dtype=np.float32)


# ------------------------------------------------------------ v3 append-only
def test_obs_v3_contract_shape_and_version():
    # M7-0a: the v3 CONTRACT persists as the read-only legacy mode; the
    # default is v4 with the v3 43 bit-preserved as a strict prefix.
    assert LocoEnv.obs_dim == 68
    assert LocoEnv.contract_version == 4
    env = LocoEnv()
    layout = env.observation_layout()
    assert layout["layout_version"] == 4 and layout["dim"] == 68
    legacy = LocoEnv(cfg={"rl": {"contract_version": 3}})
    ll = legacy.observation_layout()
    assert ll["layout_version"] == 3 and ll["dim"] == 43
    assert layout["command_frame"] == "world"
    # v4: the last section is the command block mask; the v3-era command
    # section keeps its slice at [40, 43]; the legacy layout still ends at it
    assert layout["sections"][-1]["name"] == "cmd_mask"         and layout["sections"][-1]["slice"] == [64, 68]
    assert layout["sections"][7]["name"] == "command"         and layout["sections"][7]["slice"] == [40, 43]
    assert ll["sections"][-1]["name"] == "command"         and ll["sections"][-1]["slice"] == [40, 43]


def test_obs_v3_is_append_only_full_semantic_recheck():
    """Recompute every documented v2 field from engine state and compare with
    the emitted observation: changing any v2 slice would fail here."""
    env = LocoEnv()
    env.reset(seed=7)
    for _ in range(12):
        obs, *_ = env.step((_zero(env) + 0.2).astype(np.float32))
    m = env.world.m
    d = env.world.d
    # joint sections by the documented formulas
    jp, jv = [], []
    for name in env._obs_joint_names:   # the OBS layout is the fixed 14
        j = m.joint(name)
        lo, hi = env.joint_range[name]
        span = (hi - lo) if (hi - lo) != 0 else 1.0
        q = float(d.qpos[int(j.qposadr[0])])
        v = float(d.qvel[int(j.dofadr[0])])
        jp.append((q - lo) / span * 2.0 - 1.0)
        jv.append(v / 6.0)
    torso_bid = m.body("torso").id
    mat3 = np.asarray(d.xmat[torso_bid], dtype=float).reshape(3, 3)
    vel = env.world.agent_vel().astype(float)
    cl, cr = env._foot_contacts_n()
    expect = np.array(
        jp + jv
        + [float(x) for x in mat3[:, 2]]
        + [float(x) / 5.0 for x in vel]
        + [float(mat3[2, 2])]
        + [float(x) for x in mat3[:, 1]]
        + [cl / 1000.0, cr / 1000.0], dtype=np.float32)
    assert obs.shape == (68,)
    assert np.allclose(obs[0:40], expect, atol=1e-5), \
        "a v2 slice changed its meaning under v4 (append-only)"
    # the v3 append is exactly the commanded velocity / scale (unchanged)
    assert np.allclose(obs[40:43],
                       np.asarray(env.target_velocity, dtype=np.float32) / np.float32(5.0),
                       atol=1e-6)
    # and the v4 block reads zeros with no commands (M7-0a)
    assert np.all(obs[43:68] == 0.0)


# ------------------------------------------------------------ command observability
def test_command_obs_matches_set_target_velocity():
    env = LocoEnv()
    obs = env.reset()
    tv = np.array([-0.4, 0.6, 0.1])
    env.set_target_velocity(tv)
    obs, *_ = env.step(_zero(env))
    assert np.allclose(obs[40:43], tv / 5.0, atol=1e-6)
    assert np.allclose(obs[31:34], env.world.agent_vel() / 5.0, atol=1e-6)


def test_command_obs_finite_and_bounded_by_default_task_set():
    env = LocoEnv()
    for _ in range(5):
        obs, *_ = env.step(_zero(env))
        assert np.all(np.isfinite(obs[40:43]))
    # a large-but-finite command stays finite in the observation
    env.set_target_velocity([4.0, -4.0, 0.0])
    obs, *_ = env.step(_zero(env))
    assert np.all(np.isfinite(obs[40:43]))
    assert np.allclose(obs[40:43], [0.8, -0.8, 0.0], atol=1e-6)
    with pytest.raises(ValueError):
        env.set_target_velocity([float("nan"), 0, 0])


def test_command_deterministic_across_constructions_and_seeded_resets():
    a = LocoEnv()
    b = LocoEnv()
    oa, ob = a.reset(), b.reset()
    assert np.array_equal(oa, ob), "command included: identical construction -> bit-identical obs"
    for _ in range(10):
        oa2, *_ = a.step(_zero(a))
        ob2, *_ = b.step(_zero(b))
        assert np.array_equal(oa2, ob2)
    # a seeded reset must not change the command (it only jitters the pose)
    oc = a.reset(seed=123)
    assert np.allclose(oc[40:43], a.target_velocity / 5.0, atol=1e-6)


# ------------------------------------------------------------ curriculum conditioning
def test_curriculum_stages_visible_through_command_alone():
    """The trainable ladder (standing/controlled/sustained) differs ONLY by
    target velocity (+ upright_assist, a plant knob) and the ladder is fully
    observable in obs[40:43] -> no separate task/phase id input is needed."""
    env = LocoEnv()
    env.reset()
    c = Curriculum(env)
    obs, *_ = env.step(_zero(env))
    assert np.allclose(obs[40:43], [0.0, 0.0, 0.0], atol=1e-6)
    c.select("controlled")
    obs, *_ = env.step(_zero(env))
    assert np.allclose(obs[40:43], [0.0, 0.05, 0.0], atol=1e-6)
    c.select("sustained")
    obs, *_ = env.step(_zero(env))
    assert np.allclose(obs[40:43], [0.0, 0.1, 0.0], atol=1e-6)


def test_curriculum_stage_data_commands_walkable_axis():
    """B3 measured correction: stages 2/3 command world +Y (the spawn-facing
    chest axis). A policy without waist control locomotes along that axis
    (probe_b3_yaw: mean |dy| 0.94 m vs |dx| 0.013 m); the scripted gait walks
    the same axis equally well (probe_b3_axis: heading 90 -> 0.749 m / 8 s).
    Stage phases are DATA; the reward/env contract is untouched."""
    ph = [dict(s.phase) for s in DEFAULT_STAGES[:3]]
    assert ph[0]["target_velocity"] == pytest.approx([0.0, 0.0, 0.0])
    assert ph[1]["target_velocity"] == pytest.approx([0.0, 0.25, 0.0])
    assert ph[2]["target_velocity"] == pytest.approx([0.0, 0.5, 0.0])
    # the other stage parameters are unchanged from the B2 gate state
    assert ph[0]["upright_assist"] == pytest.approx(0.6)
    assert ph[1]["upright_assist"] == pytest.approx(0.3)
    assert ph[2]["upright_assist"] == pytest.approx(0.0)
    assert all(p.get("gravity_scale") == 1.0 for p in ph)
    assert all(p["max_steps"] == 400 for p in ph)


def test_command_obs_survives_phase_application_and_step():
    """set_phase rewrites the command (non-cumulative semantics) and the
    observation tracks the CURRENT command every decision."""
    env = LocoEnv()
    env.reset()
    env.set_phase({"target_velocity": [0.0, 0.5, 0.0], "upright_assist": 0.0})
    for _ in range(3):
        obs, *_ = env.step(_zero(env))
        assert np.allclose(obs[40:43], [0.0, 0.1, 0.0], atol=1e-6)
    env.set_phase({})  # reset to config defaults (documented semantics)
    obs, *_ = env.step(_zero(env))
    assert np.allclose(obs[40:43], [0.1, 0.0, 0.0], atol=1e-6)
