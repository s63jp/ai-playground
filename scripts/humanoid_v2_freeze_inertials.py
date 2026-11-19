"""Capture the legacy humanoid model's per-body inertials so visual mesh geoms
can be attached WITHOUT changing mass/inertia (MuJoCo derives mass from geoms
when a body has no explicit <inertial>).

Writes assets/meshes/humanoid_v2/inertials.json
Run: .venv/Scripts/python.exe scripts/humanoid_v2_freeze_inertials.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)

import hashlib

import mujoco
from playground.config import load_config
from playground.world.spec import HUMANOID_ASSET_PATH
from playground.world.physics import PhysicsWorld

# Which bodies need freezing is DISCOVERED, not hardcoded: a body whose mass is
# geom-derived (no explicit <inertial> in the asset) changes when visual meshes
# are attached. Hardcoding this list is how a stale table silently re-imposed
# the pre-correction head mass.
def _geom_derived_bodies(m, mf) -> list[str]:
    import mujoco as _mj
    with open(HUMANOID_ASSET_PATH, "r", encoding="utf-8") as fh:
        asset = fh.read()
    out = []
    for body in sorted(mf["bodies"]):
        bid = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            continue
        # does the asset give this body its own <inertial>?
        i = asset.find('<body name="%s"' % body)
        explicit = False
        if i >= 0:
            nxt = asset.find("<body", i + 1)
            seg = asset[i: nxt if nxt > 0 else len(asset)]
            explicit = "<inertial" in seg
        if not explicit:
            out.append(body)
    return out

OUT = os.path.join(REPO, "assets", "meshes", "humanoid_v2", "inertials.json")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="hv2in_")
    # NOTE: build the LEGACY rig (no visual) - that is the mass we must preserve.
    cfg = load_config(overrides={"_root": tmp, "agent_body": {"type": "humanoid"}})
    w = PhysicsWorld(cfg)
    m = w.m
    with open(os.path.join(os.path.dirname(OUT), "manifest.json"), encoding="utf-8") as fh:
        mf = json.load(fh)
    targets = _geom_derived_bodies(m, mf)
    print("geom-derived bodies needing a frozen inertial: %d -> %s"
          % (len(targets), targets))
    table = {}
    for name in targets:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, name
        q = m.body_iquat[bid]
        table[name] = {
            "mass": float(m.body_mass[bid]),
            "pos": [float(v) for v in m.body_ipos[bid]],
            "quat": [float(v) for v in q],            # (w, x, y, z)
            "diaginertia": [float(v) for v in m.body_inertia[bid]],
            "geomnum": int(m.body_geomnum[bid]),
        }
        print(f"{name:15s} mass={table[name]['mass']:9.6f} "
              f"ipos={table[name]['pos']} quat={table[name]['quat']} "
              f"I={table[name]['diaginertia']}")
    with open(HUMANOID_ASSET_PATH, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    doc = {
        "generator": "scripts/humanoid_v2_freeze_inertials.py",
        "purpose": "freeze legacy mass/inertia when humanoid_v2 visual meshes are attached",
        "source_model": "build_mjcf(cfg with agent_body.type=humanoid) at qpos0",
        # Ties this table to the exact rig it was measured from. spec.py refuses
        # to use it if the asset changes, so a future head/limb mass correction
        # cannot be silently undone by a stale freeze.
        "humanoid_asset_sha256": digest,
        "bodies": table,
    }
    print("humanoid_mujoco.xml sha256:", digest)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
