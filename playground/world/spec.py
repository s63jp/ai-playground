"""World specification: every object on the island and the MJCF that realises it.

The spec is the single source of truth for the *initial* configuration, so a
session reset is simply "build the world from the spec again".
"""
from __future__ import annotations

import math, os, re
from dataclasses import dataclass, field, asdict
from typing import Any

PARK_ORIGIN = (60.0, 60.0, -40.0)    # where latent / lost bodies are parked (no collisions, no gravity)


@dataclass
class ObjectSpec:
    id: str
    name: str
    kind: str                      # ground | platform | spire | block | rock | ring
    shape: str                     # box | sphere | torus
    movable: bool
    position: tuple[float, float, float]          # centre, metres
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)   # quaternion w,x,y,z
    dimensions: dict[str, float] = field(default_factory=dict)
    mass: float | None = None      # kg; None for immovable structures
    colour: str = "#888888"
    friction: float = 0.9
    unique: bool = False
    latent: bool = False           # exists in the engine but not (yet) in the world
    properties: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict:
        d = asdict(self)
        d.pop("latent")
        d["position"] = list(self.position)
        d["rotation"] = list(self.rotation)
        return d


def build_specs(cfg: dict) -> list[ObjectSpec]:
    w = cfg["world"]
    half = float(w["island"]["half_extent"])
    specs: list[ObjectSpec] = [
        ObjectSpec(
            id="ground", name="Island Ground", kind="ground", shape="box", movable=False,
            position=(0.0, 0.0, -0.5), dimensions={"x": 2 * half, "y": 2 * half, "z": 1.0},
            colour="#c9c4a8", friction=0.9, properties={"top_height": 0.0},
        )
    ]
    for p in w["platforms"]:
        h = float(p["height"])
        specs.append(ObjectSpec(
            id=p["id"], name=p["name"], kind="platform", shape="box", movable=False,
            position=(float(p["center"][0]), float(p["center"][1]), h / 2),
            dimensions={"x": float(p["size"][0]), "y": float(p["size"][1]), "z": h},
            colour=p.get("colour", "#999999"), friction=0.9, properties={"top_height": h},
        ))
    s = w["spire"]
    specs.append(ObjectSpec(
        id=s["id"], name=s["name"], kind="spire", shape="box", movable=False,
        position=(float(s["center"][0]), float(s["center"][1]), float(s["height"]) / 2),
        dimensions={"x": float(s["size"][0]), "y": float(s["size"][1]), "z": float(s["height"])},
        colour=s.get("colour", "#dddddd"), friction=0.9, unique=True,
        properties={"top_height": float(s["height"])},
    ))
    b = w["blocks"]
    size = float(b["size"])
    total = int(b["count"]) + int(b.get("reserve", 0))
    for i in range(total):
        latent = i >= int(b["count"])
        if latent:
            k = i - int(b["count"])
            pos = (PARK_ORIGIN[0] + 2.0 * k, PARK_ORIGIN[1], PARK_ORIGIN[2])
        else:
            pos = (float(b["origin"][0]) + (i % 6) * float(b["spacing"]),
                   float(b["origin"][1]) - (i // 6) * float(b["spacing"]), size / 2)
        specs.append(ObjectSpec(
            id=f"block_{i + 1:02d}", name=f"Building Block {i + 1}", kind="block", shape="box",
            movable=True, position=pos, dimensions={"x": size, "y": size, "z": size},
            mass=float(b["mass"]), colour=b["colours"][i % len(b["colours"])],
            friction=float(b["friction"]), latent=latent,
        ))
    r = w["rock"]
    specs.append(ObjectSpec(
        id=r["id"], name=r["name"], kind="rock", shape="sphere", movable=True,
        position=(float(r["position"][0]), float(r["position"][1]), float(r["radius"])),
        dimensions={"radius": float(r["radius"]), "diameter": 2 * float(r["radius"])},
        mass=float(r["mass"]), colour=r.get("colour", "#666666"), friction=float(r["friction"]),
        unique=True,
    ))
    g = w["ring"]
    specs.append(ObjectSpec(
        id=g["id"], name=g["name"], kind="ring", shape="torus", movable=True,
        position=(float(g["position"][0]), float(g["position"][1]), float(g["tube_radius"])),
        dimensions={"major_radius": float(g["major_radius"]), "tube_radius": float(g["tube_radius"]),
                    "outer_diameter": 2 * (float(g["major_radius"]) + float(g["tube_radius"])),
                    "inner_diameter": 2 * (float(g["major_radius"]) - float(g["tube_radius"]))},
        mass=float(g["mass"]), colour=g.get("colour", "#f2c14e"), friction=float(g["friction"]),
        unique=True, properties={"segments": int(g.get("segments", 16))},
    ))
    return specs


def _extract_full_body(xml: str, body_name: str) -> str:
    """Return the full <body name=body_name> ... </body> block, including nested bodies."""
    start_tag = f'<body name="{body_name}"'
    start = xml.find(start_tag)
    if start == -1:
        return ''
    # Find end of opening tag
    open_end = xml.find('>', start)
    if open_end == -1:
        return ''
    idx = open_end + 1
    depth = 1
    while idx < len(xml) and depth > 0:
        next_open = xml.find('<body', idx)
        next_close = xml.find('</body>', idx)
        if next_close == -1:
            break
        if next_open != -1 and next_open < next_close:
            depth += 1
            idx = next_open + 1
        else:
            depth -= 1
            idx = next_close + len('</body>')
    return xml[start:idx]

def _f(*vals: float) -> str:
    return " ".join(f"{v:.6g}" for v in vals)







def build_mjcf(cfg: dict, specs: list[ObjectSpec]) -> str:
    """Generate the MuJoCo model.

    Collision groups (contype / conaffinity bit masks):
      bit 0 - ordinary solids (ground, structures, movable objects, the agent body)
      bit 1 - world-boundary walls, which only the agent body collides with
    Ray-cast groups (geom group): 0 world + active objects, 1 agent, 2 hidden, 3 boundary.
    """
    w = cfg["world"]
    a = cfg["agent_body"]
    half = float(w["island"]["half_extent"])
    hw = float(a["width"]) / 2
    hh = float(a["height"]) / 2
    out: list[str] = []
    out.append('<mujoco model="ai_playground">')
    out.append('  <compiler angle="radian" autolimits="true"/>')
    out.append(f'  <option timestep="{w["timestep"]}" gravity="0 0 {-float(w["gravity"])}" cone="elliptic" '
               'impratio="10" integrator="implicitfast"/>')
    # If humanoid, include its asset block at top level before default.
    if a.get('type', 'box') == 'humanoid':
        hum_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'humanoid_mujoco.xml'))
        with open(hum_path, 'r', encoding='utf-8') as f_h:
            hum_xml = f_h.read()
        asset_match = re.search(r'<asset>.*?</asset>', hum_xml, re.DOTALL)
        if asset_match:
            out.append(asset_match.group(0))
    out.append('  <default>')
    out.append('    <geom condim="4" friction="0.9 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001" contype="1" conaffinity="1" group="0"/>')
    out.append('  </default>')
    out.append('  <worldbody>')



    for s in specs:
        if s.movable:
            continue
        d = s.dimensions
        out.append(f'    <geom name="g:{s.id}" type="box" pos="{_f(*s.position)}" '
                   f'size="{_f(d["x"] / 2, d["y"] / 2, d["z"] / 2)}" friction="{_f(s.friction, 0.005, 0.0001)}"/>')
    # Boundary walls: invisible, collide with the agent body only.
    t, wall_h = 0.5, 40.0
    for name, pos, size in (
        ("east", (half + t, 0, wall_h / 2 - 5), (t, half + 2 * t, wall_h / 2)),
        ("west", (-half - t, 0, wall_h / 2 - 5), (t, half + 2 * t, wall_h / 2)),
        ("north", (0, half + t, wall_h / 2 - 5), (half + 2 * t, t, wall_h / 2)),
        ("south", (0, -half - t, wall_h / 2 - 5), (half + 2 * t, t, wall_h / 2)),
    ):
        out.append(f'    <geom name="g:boundary:{name}" type="box" pos="{_f(*pos)}" size="{_f(*size)}" '
                   'contype="2" conaffinity="0" group="3" condim="1"/>')
    # The agent body: an upright box on three slide joints. Frictionless with priority so
    # that all traction comes from the walking controller (see physics.py).
    sx, sy = float(w["start"][0]), float(w["start"][1])
    out.append(f'    <body name="agent" pos="{_f(sx, sy, hh + 0.002)}">')
    out.append('      <joint name="agent_x" type="slide" axis="1 0 0"/>')
    out.append('      <joint name="agent_y" type="slide" axis="0 1 0"/>')
    out.append('      <joint name="agent_z" type="slide" axis="0 0 1"/>')
    # Insert humanoid torso body (with limbs)
    if a.get("type", "box") == "humanoid":
        hum_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "humanoid_mujoco.xml")
        with open(hum_path, "r", encoding="utf-8") as f_h:
            hum_xml = f_h.read()
        # torso body (with limbs)
        torso_block = _extract_full_body(hum_xml, "torso")
        if torso_block:
            out.append(torso_block)
    else:
        out.append(f'    <geom name="g:agent" type="box" size="{_f(hw, hw, hh)}" mass="{a["mass"]}" '
+               'friction="0 0 0" priority="1" condim="1" contype="1" conaffinity="3" group="1"/>')

    # close the agent body
    out.append('    </body>')
    # movable objects
    for s in specs:
        if not s.movable:
            continue
        grp = 2 if s.latent else 0
        col = ' contype="0" conaffinity="0"' if s.latent else ""
        out.append(f'    <body name="{s.id}" pos="{_f(*s.position)}" quat="{_f(*s.rotation)}">')
        out.append(f'      <freejoint name="j:{s.id}"/>')
        fr = _f(s.friction, 0.005, 0.0001)
        if s.shape == "box":
            d = s.dimensions
            out.append(f'      <geom name="g:{s.id}" type="box" size="{_f(d["x"] / 2, d["y"] / 2, d["z"] / 2)}" '
+                       f'mass="{s.mass}" friction="{fr}" group="{grp}"{col}/>')
        elif s.shape == "sphere":
            out.append(f'      <geom name="g:{s.id}" type="sphere" size="{_f(s.dimensions["radius"])}" '
+                       f'mass="{s.mass}" condim="6" friction="{_f(s.friction, 0.02, 0.02)}" group="{grp}"/>')
        elif s.shape == "torus":
            n = int(s.properties.get("segments", 16))
            R, r = s.dimensions["major_radius"], s.dimensions["tube_radius"]
            for k in range(n):
                a0 = 2 * math.pi * k / n
                a1 = 2 * math.pi * (k + 1) / n
                out.append(
                    f'      <geom name="g:{s.id}:{k:02d}" type="capsule" size="{_f(r)}" '
+                    f'fromto="{_f(R * math.cos(a0), R * math.sin(a0), 0, R * math.cos(a1), R * math.sin(a1), 0)}" '
+                    f'mass="{s.mass / n:.6g}" friction="{fr}" group="{grp}"/>')
        out.append('    </body>')
    out.append('  </worldbody>')
    out.append('</mujoco>')
    return "\n".join(out)
