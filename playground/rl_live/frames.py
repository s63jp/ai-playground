"""Live-frame extraction from the authoritative LocoEnv (M6-C foundations).

Read-only: every value here is measured from the live MuJoCo model/data that
the trainer is stepping (`env.world.m` / `env.world.d`), or comes from the
env's own step return/`info`. NOTHING in this module touches physics state.

Two payloads:

  scene_spec(env)     one-episode-static description of the world: bodies,
                      geoms (plane/sphere/capsule/cylinder/box/ellipsoid),
                      joint list. The viewer builds meshes from it once and
                      then only applies per-frame body poses. Mesh geoms are
                      skipped (counted, not rendered).
  pack_frame(...)     one decision boundary: every rig body's world pose
                      (d.xpos / d.xquat — MuJoCo's own forward kinematics),
                      the 14 policy joints (q, qvel), the action the policy
                      just applied, the reward measured by the ONE pipeline,
                      foot support contacts, fall / fault flags and the
                      commanded target velocity. These are the actual training
                      env and the actual policy — never a separate simulation.

The world is rebuilt every episode (LocoEnv.reset), so a FrameIndex must be
re-resolved on world identity change; body GEOMETRY is unchanged (same cfg)
but ids are not, hence a frame always carries NAMES, never ids.

Quaternions are MuJoCo's (w, x, y, z) order everywhere.
"""

from __future__ import annotations

import numpy as np

SPEC_VERSION = 1
FRAME_VERSION = 1

MAX_POLICY_JOINTS = 14


# ------------------------------------------------------------ small helpers
def _r(v, nd=4):
    return round(float(v), nd)


def _rlist(seq, nd=4):
    return [_r(v, nd) for v in seq]


def _mujoco():
    import mujoco
    return mujoco


def _geom_type_names():
    mj = _mujoco()
    names = ("plane", "sphere", "capsule", "ellipsoid", "cylinder", "box")
    return {int(getattr(mj.mjtGeom, "mjGEOM_" + name.upper())): name
            for name in names}


# ------------------------------------------------------------- rig identity
def agent_container_id(env) -> int:
    """The rig's container body id (mirrors LocoEnv._collect_bodies)."""
    m = env.world.m
    try:
        return int(env.world._ag(None).bid)
    except Exception:
        return int(m.body("agent").id)


def rig_body_ids(env) -> set[int]:
    """Every body whose parent chain reaches the agent container, including
    the world (id 0 is worldbody and never in the rig)."""
    m = env.world.m
    pid = np.asarray(m.body_parentid, dtype=int)
    container = agent_container_id(env)
    rig = set()
    for i in range(m.nbody):
        bid = int(i)
        while bid != 0:
            if bid == container:
                rig.add(int(i))
                break
            bid = int(pid[bid])
    return rig


def rig_body_names(env) -> list[str]:
    """Ordered rig body names (model order — stable across episodes)."""
    m = env.world.m
    rig = rig_body_ids(env)
    out = []
    for i in range(m.nbody):
        if i in rig:
            name = m.body(i).name
            if isinstance(name, (bytes, bytearray)):
                name = name.decode()
            out.append(str(name))
    return out


def _name_of(m, handle) -> str:
    try:
        n = handle.name
    except Exception:
        return ""
    return str(n.decode() if isinstance(n, (bytes, bytearray)) else n)


# --------------------------------------------------------------- scene spec
def scene_spec(env) -> dict:
    """Build the viewer's scene description from the live model.

    Deterministic for a given cfg; independent of episode state except that
    root/agent positions come from the current episode (frames supersede it).
    """
    m = env.world.m
    d = env.world.d
    gtypes = _geom_type_names()
    rig = rig_body_ids(env)
    meshes_skipped = 0
    geoms_skipped = 0
    bodies = []
    for bid in range(m.nbody):
        geoms = []
        for gid in range(int(m.body_geomadr[bid]),
                         int(m.body_geomadr[bid]) + int(m.body_geomnum[bid])):
            tcode = int(m.geom_type[gid])
            tname = gtypes.get(tcode)
            if tname is None:
                if tcode == int(getattr(_mujoco().mjtGeom, "mjGEOM_MESH", 7)):
                    meshes_skipped += 1
                else:
                    geoms_skipped += 1
                continue
            size = np.asarray(m.geom_size[gid], dtype=float)
            if tname == "plane":
                # An infinite plane carries zeros; render a bounded floor.
                size = size if size[0] > 0 else np.array([15.0, 15.0, 0.0])
            geoms.append({
                "type": tname,
                "size": _rlist(size, 4),
                "pos": _rlist(np.asarray(m.geom_pos[gid], dtype=float), 4),
                "quat": _rlist(np.asarray(m.geom_quat[gid], dtype=float), 5),
                "rgba": _rlist(np.asarray(m.geom_rgba[gid], dtype=float), 3),
                "group": int(m.geom_group[gid]),
            })
        name = _name_of(m, m.body(bid))
        body = {
            "name": name,
            "rig": bid in rig,
            "parent": _name_of(m, m.body(int(m.body_parentid[bid]))) if bid != 0 else None,
            "pos": _rlist(np.asarray(d.xpos[bid], dtype=float), 4),
            "quat": _rlist(np.asarray(d.xquat[bid], dtype=float), 5),
            "geoms": geoms,
        }
        bodies.append(body)
    joints = []
    try:
        hinge = int(_mujoco().mjtJoint.mjJNT_HINGE)
        slide = int(_mujoco().mjtJoint.mjJNT_SLIDE)
    except Exception:
        hinge, slide = 3, 2
    for jid in range(m.njnt):
        jtype = int(m.jnt_type[jid])
        if jtype not in (hinge, slide):
            continue
        name = _name_of(m, m.joint(jid))
        if not name:
            continue
        limited = bool(int(m.jnt_limited[jid]))
        rng = np.asarray(m.jnt_range[jid], dtype=float)
        joints.append({
            "name": name,
            "type": "hinge" if jtype == hinge else "slide",
            "limited": limited,
            "range": _rlist(rng, 4) if limited else None,
        })
    policy_joints = []
    if hasattr(env, "joint_names"):
        policy_joints = [str(n) for n in env.joint_names][:MAX_POLICY_JOINTS]
    return {
        "spec_version": SPEC_VERSION,
        "sim": {"dt": _r(m.opt.timestep, 6),
                "gravity": _rlist(np.asarray(m.opt.gravity, dtype=float), 3)},
        "world_size_hint": 30.0,
        "bodies": bodies,
        "joints": joints,
        "policy_joints": policy_joints,
        "meshes_skipped": meshes_skipped,
        "geoms_skipped": geoms_skipped,
        "notes": [
            "quaternion order (w, x, y, z)",
            "mesh geoms are listed in meshes_skipped, not rendered",
            "body poses per frame come from d.xpos/xquat at the decision boundary",
        ],
    }


# --------------------------------------------------------------- frame pack
class FrameIndex:
    """Resolved, name-keyed access into the CURRENT world instance.

    PhysicsWorld is rebuilt on every episode reset, so indices are re-derived
    whenever `env.world` changes identity. Body ids are internal; payloads
    carry names only.
    """

    def __init__(self, env):
        self.env = env
        self._world = None
        self._rig_names = []
        self._torso_bid = None
        self._joint_idx = []          # (name, qposadr, dofadr)

    def ensure(self):
        if self._world is self.env.world:
            return
        m = self.env.world.m
        self._world = self.env.world
        self._rig_names = rig_body_names(self.env)
        try:
            self._torso_bid = int(m.body("torso").id)
        except Exception:
            self._torso_bid = None
        self._joint_idx = []
        for name in getattr(self.env, "joint_names", [])[:MAX_POLICY_JOINTS]:
            try:
                j = m.joint(name)
                self._joint_idx.append((name, int(j.qposadr[0]), int(j.dofadr[0])))
            except Exception:
                continue

    def body_dict(self) -> dict:
        """{name: {"p": [x,y,z], "q": [w,x,y,z]}} for all rig bodies."""
        d = self.env.world.d
        m = self.env.world.m
        pid = np.asarray(m.body_parentid, dtype=int)
        rig = rig_body_ids(self.env)
        out = {}
        for bid, name in zip((i for i in range(m.nbody) if i in rig), self._rig_names):
            out[name] = {
                "p": _rlist(np.asarray(d.xpos[bid], dtype=float), 4),
                "q": _rlist(np.asarray(d.xquat[bid], dtype=float), 5),
            }
        # zip pairing discipline: _rig_names was derived from the same rig set
        # on THIS world instance (ensure()), so lengths must agree.
        return out

    def joints(self) -> dict:
        d = self.env.world.d
        out = {}
        for name, qadr, vadr in self._joint_idx:
            out[name] = {"q": _r(float(d.qpos[qadr]), 4),
                         "v": _r(float(d.qvel[vadr]), 4)}
        return out

    def up_z(self) -> float:
        if self._torso_bid is None:
            return _r(0.0, 4)
        m3 = np.asarray(self.env.world.d.xmat[self._torso_bid],
                        dtype=float).reshape(3, 3)
        return _r(m3[2, 2], 4)


def pack_frame(env, *, action, reward, terminated, truncated, info,
               episode, decision, index: FrameIndex) -> dict:
    """One decision boundary, from the state env.step just produced."""
    index.ensure()
    m = env.world.m
    d = env.world.d
    info = info or {}
    contacts = info.get("contacts") or (0.0, 0.0)
    comps = {k: _r(v, 4) for k, v in (info.get("components") or {}).items()}
    return {
        "t": "frame",
        "v": FRAME_VERSION,
        "episode": int(episode),
        "decision": int(decision),
        "sim_t": _r(env.world.time, 4),
        "bodies": index.body_dict(),
        "joints": index.joints(),
        "root": {"pos": _rlist(np.asarray(env.world.agent_pos(), dtype=float), 4),
                 "vel": _rlist(np.asarray(env.world.agent_vel(), dtype=float), 4)},
        "up_z": index.up_z(),
        "action": _rlist(np.asarray(action, dtype=float), 4),
        "reward": _r(float(reward), 5),
        "components": comps,
        "effort_j": _r(float(info.get("effort_j", 0.0)), 4),
        "contacts": {"left": _r(contacts[0], 2), "right": _r(contacts[1], 2)},
        "fallen": bool(info.get("fallen", False)),
        "faulted": bool(info.get("faulted", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "target_velocity": _rlist(np.asarray(env.target_velocity, dtype=float), 4),
        "dt_window": _r(env.action_repeat * float(m.opt.timestep), 5),
    }


def pack_reset_event(env, *, episode, seed) -> dict:
    pos = np.asarray(env.world.agent_pos(), dtype=float)
    return {"t": "reset", "episode": int(episode), "seed": (None if seed is None else int(seed)),
            "start_pos": _rlist(pos, 4), "sim_t": _r(env.world.time, 4)}


def pack_episode_event(env, *, episode, terminated, truncated) -> dict:
    """Episode end: the env's own measured summary only (no derivation)."""
    s = env.episode_summary()
    keep = ("steps", "sim_time_s", "net_displacement_m", "path_len_m",
            "mean_speed_m_s", "upright_fraction", "min_up_z",
            "effort_total_j", "effort_mean_w", "reward_total", "falls",
            "faulted_windows", "truncated_at", "components")
    slim = {k: s.get(k) for k in keep}
    slim["t"] = "episode_end"
    slim["episode"] = int(episode)
    slim["terminated"] = bool(terminated)
    slim["truncated"] = bool(truncated)
    slim["reason"] = ("fault" if slim.get("faulted_windows") else
                      "fall" if slim.get("falls") else
                      "time" if truncated else "other")
    slim["start_position"] = s.get("start_position")
    slim["final_position"] = s.get("final_position")
    return slim
