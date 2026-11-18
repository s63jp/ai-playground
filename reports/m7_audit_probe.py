"""M7 audit probe (2026-11-18, read-only) — verify the facts the M7 design rests on.

No training, no repo writes; builds ONE default LocoEnv locally and reports:
  1. contract dims + joint inventory (policy + passive + scene-only)
  2. agent-subtree mass + standing COM vs foot support region (stage-2/3 measurement seam)
  3. waist_yaw presence/state in the scene model (the turning blocker)
  4. block-target spawn seam (setup_place -> obj_pos on a fresh scratch world)
  5. disturbance-path existence checks (engine force-application API)
"""
import json
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\jamie\ai-playground")

from playground.rl.env import LocoEnv  # noqa: E402

out = {"probe": "m7_audit_probe", "date": "2026-11-18"}

env = LocoEnv()
obs = env.reset(seed=20261117)

al = env.action_layout()
ol = env.observation_layout()
out["contract"] = {
    "contract_version": env.contract_version,
    "obs_dim": int(env.obs_dim),
    "action_dim": int(env.action_dim),
    "joint_order": list(env.joint_names),
    "joint_ranges_rad": ol["joint_ranges_rad"],
    "gains_kp": {j["name"]: j["kp"] for j in al["joints"]},
    "passive_pd": [p["name"] for p in al["passive_conventional_pd"]],
    "obs_sections": [(s["name"], s["slice"]) for s in ol["sections"]],
}
out["obs_observed_len"] = int(len(obs))

m, d = env.world.m, env.world.d

# full scene hinge inventory incl. passive + the agent root slides
all_joints = {}
for i in range(m.njnt):
    name = m.joint(i).name or f"jnt{i}"
    jtype = int(m.jnt_type[i])
    tname = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jtype, str(jtype))
    limited = bool(int(m.jnt_limited[i]))
    all_joints[name] = {"type": tname,
                        "range": ([float(m.jnt_range[i][0]), float(m.jnt_range[i][1])]
                                  if limited else None)}
out["scene_joints"] = all_joints

# agent-subtree mass + COM at the standing spawn (stage 2/3 seam: is COM
# measurable in-model, and where does it sit vs the feet?)
body_ids = set()
container = int(env.world._ag(None).bid)
for i in range(m.nbody):
    bid = i
    while bid != 0:
        if bid == container:
            body_ids.add(i)
            break
        bid = int(m.body_parentid[bid])
mass = 0.0
com = np.zeros(3)
for bid in body_ids:
    if bid == container:
        continue
    bmass = float(m.body_mass[bid])
    mass += bmass
    com += bmass * np.asarray(d.xipos[bid], dtype=float)
com /= max(mass, 1e-9)
out["agent_subtree"] = {
    "n_bodies": len(body_ids) - 1,
    "total_mass_kg": round(mass, 3),
    "standing_com_xyz_m": [round(float(x), 4) for x in com],
}

# foot support extents from the foot geoms (static margin at spawn)
foot_rows = []
for side in ("left", "right"):
    g = m.geom(f"g:{side}_foot")
    body_xipos = np.asarray(d.xipos[int(m.geom_bodyid[g.id])], dtype=float)
    sz = np.asarray(m.geom_size[g.id], dtype=float)
    gpos = np.asarray(d.geom_xpos[g.id], dtype=float)
    foot_rows.append({"side": side, "half_size_xyz": [round(float(x), 4) for x in sz],
                      "center_xy": [round(float(gpos[0]), 4), round(float(gpos[1]), 4)]})
out["feet"] = foot_rows
y_lo = min(r["center_xy"][1] - r["half_size_xyz"][1] for r in foot_rows)
y_hi = max(r["center_xy"][1] + r["half_size_xyz"][1] for r in foot_rows)
out["static_margin"] = {
    "support_y_range_m": [round(y_lo, 4), round(y_hi, 4)],
    "com_y_m": round(float(com[1]), 4),
    "com_inside_feet": bool(y_lo <= float(com[1]) <= y_hi),
}

# stage-4 seam: engine force paths that exist (existence only, not exercised)
out["disturbance_seams"] = {
    "impulse_object_world_api": hasattr(env.world, "impulse_object"),
    "set_object_velocity": hasattr(env.world, "set_object_velocity"),
    "set_joint_target": hasattr(env.world, "set_joint_target"),
    "state_snapshot": hasattr(env.world, "get_state") and hasattr(env.world, "set_state"),
}

# stage-6 seam: the scene ALREADY carries 20 freejoint blocks (j:block_01..20) +
# ring + rock; setup_place repositions an EXISTING object (verified here on
# block_01 — no new spawn machinery needed, proven on THIS scratch world)
try:
    before = np.asarray(env.world.obj_pos("block_01"), dtype=float)
    env.world.setup_place("block_01",
                          position=[0.5, 1.5, 0.05],
                          rotation_euler_degrees=(0.0, 0.0, 0.0))
    pos = np.asarray(env.world.obj_pos("block_01"), dtype=float)
    out["block_target_seam"] = {
        "works": bool(np.linalg.norm(pos - np.asarray([0.5, 1.5, 0.05])) < 0.2),
        "moved_from_xy_m": [round(float(before[0]), 4), round(float(before[1]), 4)],
        "placed_xyz_m": [round(float(x), 4) for x in pos],
    }
except Exception as exc:  # pragma: no cover
    out["block_target_seam"] = {"works": False, "error": repr(exc)}

out["eval_constants"] = {"seed_base": 97_010_000, "default_episodes": 10,
                         "default_max_steps": 400}

path = r"C:\Users\jamie\ai-playground\reports\M7_AUDIT_PROBE.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
