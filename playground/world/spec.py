"""World specification: every object on the island and the MJCF that realises it.

The spec is the single source of truth for the *initial* configuration, so a
session reset is simply "build the world from the spec again".
"""
from __future__ import annotations

import hashlib, json, math, os, re
from dataclasses import dataclass, field, asdict
from typing import Any

PARK_ORIGIN = (60.0, 60.0, -40.0)    # where latent / lost bodies are parked (no collisions, no gravity)

# The humanoid asset source. This is a module-level constant (not a config
# key) so the DEFAULT behaviour of every build is byte-identical to the
# previous inline path expressions; checkpoint playback ( rl_live.playback )
# points it at a SHA-verified copy of a historical asset for demos and never
# touches the default. All three humanoid reads in this module go through it.
HUMANOID_ASSET_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "humanoid_mujoco.xml"))


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
    # Combat (M4): equipment bodies are real movable objects; pellets are a
    # latent parked pool the launcher materialises in flight. A world without
    # a cfg["combat"] section is byte-identical to the pre-M4 island.
    from ..combat.equipment import EQUIPMENT
    combat = cfg.get("combat") or {}
    for entry in combat.get("items") or []:
        item_id = str(entry.get("item"))
        if item_id not in EQUIPMENT:
            raise ValueError(f"combat item '{entry.get('id')}' references unknown equipment '{item_id}'")
        es = EQUIPMENT[item_id]
        dx, dy, dz = (float(v) for v in es["dims_m"])
        shape = es.get("shape", "box")
        if shape == "sphere":
            r = dz / 2.0
            dims = {"radius": r, "diameter": dz}
            pos = (float(entry["at"][0]), float(entry["at"][1]), r)
        else:
            dims = {"x": dx, "y": dy, "z": dz}
            pos = (float(entry["at"][0]), float(entry["at"][1]), dz / 2.0)
        specs.append(ObjectSpec(
            id=str(entry.get("id")), name=es["name"], kind="equipment", shape=shape,
            movable=True, position=pos, dimensions=dims, mass=float(es["mass_kg"]),
            colour=entry.get("colour", "#e07020"), unique=True,
            properties={"equipment": item_id},
        ))
    pellets_n = int(combat.get("pellets", 0))
    pr = float(combat.get("pellet_radius", 0.02))
    pm = float(combat.get("pellet_mass", 0.06))
    for i in range(pellets_n):
        specs.append(ObjectSpec(
            id=f"pellet_{i + 1:02d}", name=f"Paint Pellet {i + 1}", kind="pellet",
            shape="sphere", movable=True,
            position=(PARK_ORIGIN[0] + 2.0 * (i + 1), PARK_ORIGIN[1] + 2.0, PARK_ORIGIN[2]),
            dimensions={"radius": pr, "diameter": 2 * pr}, mass=pm,
            colour="#d22b2b", latent=True, unique=True,
            properties={"pellet": True},
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







def _hv2_dir() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "meshes", "humanoid_v2"))


_HV2_CACHE: dict | None = None


def _hv2_data() -> dict:
    """manifest + frozen inertials for the humanoid_v2 visual set (cached).

    The frozen inertials are tied to a FINGERPRINT of the humanoid asset. If
    assets/humanoid_mujoco.xml changes (as it did when the head mass/COM was
    corrected in the B4 head-mechanics fix), a stale table would silently
    re-impose the old masses, so a mismatch raises instead.
    """
    global _HV2_CACHE
    if _HV2_CACHE is None:
        base = _hv2_dir()
        with open(os.path.join(base, "manifest.json"), "r", encoding="utf-8") as fh:
            mf = json.load(fh)
        with open(os.path.join(base, "inertials.json"), "r", encoding="utf-8") as fh:
            inert = json.load(fh)
        with open(HUMANOID_ASSET_PATH, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        recorded = inert.get("humanoid_asset_sha256")
        if recorded and recorded != digest:
            raise AssertionError(
                "humanoid_v2: inertials.json was frozen against "
                "humanoid_mujoco.xml %s but the asset is now %s. Re-run "
                "scripts/humanoid_v2_freeze_inertials.py - using the stale "
                "table would silently restore the old body masses."
                % (recorded[:12], digest[:12]))
        _HV2_CACHE = {"mf": mf, "inert": inert, "base": base}
    return _HV2_CACHE


def _hv2_on(ac: dict) -> bool:
    """Opt-in: only agent_body.visual == 'humanoid_v2' changes generated XML."""
    return ac.get("visual") == "humanoid_v2"


def _hv2_mesh_assets() -> str:
    """<compiler meshdir> + <asset> entries for the humanoid_v2 visual set."""
    parts = sorted(_hv2_data()["mf"]["parts"])
    meshdir = _hv2_dir().replace("\\", "/")
    rows = "\n".join('    <mesh name="hv2_%s" file="%s.obj"/>' % (p, p) for p in parts)
    return '  <compiler meshdir="%s"/>\n  <asset>\n%s\n  </asset>' % (meshdir, rows)


def _hv2_visual_inject(block: str, ac: dict) -> str:
    """Attach humanoid_v2 visual meshes to a humanoid torso block.

    Two edits per target body:
      * an explicit <inertial> from inertials.json for bodies whose mass is
        geom-derived - MuJoCo recomputes mass from geoms, so adding 35 meshes
        would otherwise change those bodies' mass and invalidate training.
      * a non-colliding mesh <geom> (contype/conaffinity 0, group 1, density 0).
    """
    if not _hv2_on(ac):
        return block
    d = _hv2_data()
    mf = d["mf"]
    bodies = d["inert"]["bodies"]
    # manifest schema: bodies = {rig body -> [part names]}, parts[p]["rgba"]
    by_body = {b: sorted(items) for b, items in mf["bodies"].items()}

    for body, items in sorted(by_body.items()):
        anchor = '<body name="%s"' % body
        idx = block.find(anchor)
        if idx < 0:
            raise AssertionError(
                "humanoid_v2: body '%s' missing from the rig block" % body)
        close = block.index(">", idx) + 1
        ins = []
        fr = bodies.get(body)
        if fr is not None:
            ins.append('<inertial pos="%s" quat="%s" mass="%.17g" diaginertia="%s"/>'
                       % (" ".join("%.17g" % v for v in fr["pos"]),
                          " ".join("%.17g" % v for v in fr["quat"]),
                          fr["mass"],
                          " ".join("%.17g" % v for v in fr["diaginertia"])))
        for part in items:
            rgba = " ".join("%.4g" % v for v in mf["parts"][part]["rgba"])
            ins.append('<geom name="g:hv2:%s" type="mesh" mesh="hv2_%s" rgba="%s" '
                       'contype="0" conaffinity="0" group="1" density="0"/>'
                       % (part, part, rgba))
        block = block[:close] + "".join(ins) + block[close:]
    return block


def _humanoid_torso_block(ac: dict) -> str:
    """Extract the asset torso body and apply the world's rig contract edits.

    Shared by single-agent (legacy) and multi-agent generation; the caller
    then prefixes names per agent. Returns the torso <body> block WITHOUT any
    per-agent naming, so this helper's output for a name-passthrough call is
    byte-identical to the legacy inline generation.
    """
    hum_path = HUMANOID_ASSET_PATH
    with open(hum_path, "r", encoding="utf-8") as f_h:
        hum_xml = f_h.read()
    # torso body (with limbs)
    torso_block = _extract_full_body(hum_xml, "torso")
    if torso_block:
        # The asset defines the figure in its own frame (ground at z=0,
        # torso centre at 1.15). Nested under the agent body — whose
        # origin sits at hh+0.002 — the torso must be offset to t = 1.15
        # - hh - 0.002 = 0.348 so the feet land at the agent root height.
        # Without this the figure floated 0.8 m above the ground and every
        # spawn began with a 0.8 m free-fall impact (the QACC blow-ups).
        old = '<body name="torso" pos="0 0 1.15">'
        assert old in torso_block, (
            "humanoid asset torso anchor line changed; update the humanoid "
            "torso offset in playground/world/spec.py")
        # waist_yaw: the rig's only rotational root dof. The head/limb axes
        # can't re-aim a walk by themselves — with the root on 3 slide joints
        # the legs can only sweep along the torso's local +X, so heading
        # control needs the whole upper figure to yaw (Nov 17 heading test).
        # Full ±180 deg (radians for the world compiler) so every compass
        # heading is reachable; damped 2.0, posture-held by the gait PD.
        new = ('<body name="torso" pos="0 0 0.348">'
               '<joint name="waist_yaw" type="hinge" axis="0 0 1" '
               'limited="true" range="-3.1416 3.1416" damping="2.0"/>')
        torso_block = torso_block.replace(old, new, 1)
        # With the waist joint the torso is an articulated child, so the
        # agent container no longer carries welded mass and MuJoCo would
        # refuse (or, with a token mass, destabilise: a near-massless
        # parent makes the waist dof's effective inertia collapse and the
        # solver explodes at mjMINVAL scales). Give it pelvis-class mass
        # and inertia: the slides then carry a ~51 kg figure, and the
        # waist dof keeps a healthy ~1+ kg m^2 about its axis.
        torso_block = ('<inertial pos="0 0 0" mass="10.0" diaginertia="1.0 1.2 1.0"/>'
                       + torso_block)
        # Locomotion pairing experiment (cfg-switchable). When
        # agent_body.limb_friction_engine is true, limb geoms get the
        # frictionless, normal-only contact pairing the engine pairs with
        # the root-traction drive (same as the box agent base geom; real
        # foot friction under the root-locked rig converts the traction's
        # tipping moment into leg-joint fold instead of walking). When
        # false (the current default — the humanoid walks CONTACT-DRIVEN
        # and needs foot grip), the asset's original friction stays.
        # humanoid_v2 visual meshes (opt-in). Injected here so BOTH the
        # single-agent and multi-agent (_prefix_names) paths carry them.
        torso_block = _hv2_visual_inject(torso_block, ac)
        if bool(ac.get("limb_friction_engine", False)):
            # priority="-1": contact parameters come from the geom with
            # the LOWER priority, and the floor specs compile at priority
            # 0 — with priority="1" the frictionless pair params never
            # reached the foot-floor contact and the fold/sink persisted.
            torso_block = re.sub(
                r'<geom (?![^>]*friction=)',
                '<geom condim="1" friction="0 0 0" priority="-1" ',
                torso_block)
    return torso_block


def _prefix_names(xml_block: str, prefix: str) -> str:
    """Rewrite every name="..." attribute with a <prefix><name> form.

    Used to stamp N copies of the humanoid rig into one model. The prefix is
    the agent's own id (e.g. "agent_002:"), so scoped names are both unique
    and self-describing in events/replays.
    """
    if not prefix:
        return xml_block
    return re.sub(r'name="([^"]+)"', lambda m: f'name="{prefix}{m.group(1)}"', xml_block)


def _multi_spec_start(w: dict, spec_a: dict, idx: int) -> list[float]:
    """Per-agent spawn (x, y): its own start, an offset from the world start, or a
    1.2 m row along +y (deterministic, no initial body overlap)."""
    sx, sy = float(w["start"][0]), float(w["start"][1])
    off = spec_a.get("start_offset")
    if "start" in spec_a:
        return [float(spec_a["start"][0]), float(spec_a["start"][1])]
    if off is not None:
        return [sx + float(off[0]), sy + float(off[1])]
    return [sx, sy + 1.2 * idx]


def agent_rigs(cfg: dict) -> dict[str, dict]:
    """Per-agent rig plan shared by MJCF generation (build_mjcf) and the physics
    runtime (PhysicsWorld.__init__): for every agent its merged body spec, its
    name map (asset/rig name -> model name) and its spawn position.

    With no cfg["agent_bodies"], returns the legacy single rig: agent_001 with an
    identity name map and the world's own start — the pre-M3 behaviour exactly.
    """
    w = cfg["world"]
    a = cfg["agent_body"]
    multi = cfg.get("agent_bodies")
    rigs: dict[str, dict] = {}
    if multi:
        for idx, (aid, override) in enumerate(multi.items()):
            spec_a = {**a, **(override or {})}
            prefix = f"{aid}:"
            nm = {"agent": prefix + "agent", "agent_x": prefix + "agent_x",
                  "agent_y": prefix + "agent_y", "agent_z": prefix + "agent_z",
                  "g:agent": f"g:agent:{aid}"}
            if spec_a.get("type", "box") == "humanoid":
                block = _humanoid_torso_block(spec_a)
                for name in re.findall(r'name="([^"]+)"', block):
                    nm.setdefault(name, prefix + name)
            rigs[aid] = {"prefix": prefix, "name_map": nm,
                         "start": _multi_spec_start(w, spec_a, idx), "spec": spec_a}
    else:
        sx, sy = float(w["start"][0]), float(w["start"][1])
        rigs["agent_001"] = {"prefix": "", "name_map": {}, "start": [sx, sy], "spec": dict(a)}
    return rigs


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
        hum_path = HUMANOID_ASSET_PATH
        with open(hum_path, 'r', encoding='utf-8') as f_h:
            hum_xml = f_h.read()
        asset_match = re.search(r'<asset>.*?</asset>', hum_xml, re.DOTALL)
        if asset_match:
            out.append(asset_match.group(0))
        # humanoid_v2 visual meshes: a second <compiler meshdir>/<asset> pair
        # (MuJoCo merges repeated sections) so the opt-in leaves the asset's
        # own block untouched.
        if _hv2_on(a):
            out.append(_hv2_mesh_assets())
        # Include the humanoid's <default> block too. It carries the asset's
        # own rig contract for joints (damping="1.5" limited="true" range
        # "-90 90"). The previous integration copied only <asset>, so every
        # hinge compiled UNLIMITED and UNDAMPED — the ankle could spin 9.8 rad
        # and slam the solver (qacc -> 5e9) instead of standing (Nov 17 lab).
        default_match = re.search(r'<default>.*?</default>', hum_xml, re.DOTALL)
        if default_match:
            # The asset is authored under its own degree-based compiler, but
            # the world compiles in RADIANS: copying range="-90 90" verbatim
            # compiled as ±90 rad (±5153°) — effectively unlimited joints
            # (lab evidence: limfrac 0.02 on a knee reaching 1.93 rad, the
            # ankle able to spin 9.78 rad). Rewrite the documented ±90°
            # contract into radians before inclusion.
            d0 = default_match.group(0)
            assert 'range="-90 90"' in d0, (
                "humanoid joint range anchor changed; update the radian "
                "transform in playground/world/spec.py")
            out.append(d0.replace('range="-90 90"', 'range="-1.5708 1.5708"'))
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
    multi_agents = cfg.get("agent_bodies")  # {agent_id: per-agent spec overrides}
    if multi_agents:
        # ---- multi-agent world: one independently named rig per agent ----
        # Every name inside an agent's rig is prefixed with the agent's own id
        # ("agent_002:left_hip"), so joint/actuator/geom address mappings are
        # per-agent by construction and never collide. A world with no
        # "agent_bodies" key keeps the legacy single-rig naming below
        # byte-identically.
        for idx, (aid, rig) in enumerate(agent_rigs(cfg).items()):
            spec_a = rig["spec"]
            hw_a = float(spec_a["width"]) / 2
            hh_a = float(spec_a["height"]) / 2
            ax0, ay0 = rig["start"]
            P = rig["prefix"]
            out.append(f'    <body name="{P}agent" pos="{_f(ax0, ay0, hh_a + 0.002)}">')
            # limited="false": the humanoid asset's joint default (limited=true,
            # range -90..90 — parsed as radians by the radian compiler) must not
            # clamp the root slides to a 1.57 m pen around the spawn point.
            out.append(f'      <joint name="{P}agent_x" type="slide" axis="1 0 0" limited="false" range="-1e5 1e5"/>')
            out.append(f'      <joint name="{P}agent_y" type="slide" axis="0 1 0" limited="false" range="-1e5 1e5"/>')
            out.append(f'      <joint name="{P}agent_z" type="slide" axis="0 0 1" limited="false" range="-1e5 1e5"/>')
            if spec_a.get("type", "box") == "humanoid":
                torso = _prefix_names(_humanoid_torso_block(spec_a), P)
                indent = re.sub(r'\n', '\n      ', torso)
                out.append('      ' + indent)
            else:
                out.append(f'      <geom name="g:agent:{aid}" type="box" size="{_f(hw_a, hw_a, hh_a)}" '
                           f'mass="{spec_a["mass"]}" '
                           'friction="0 0 0" priority="1" condim="1" contype="1" conaffinity="3" group="1"/>')
            out.append('    </body>')
    else:
        out.append(f'    <body name="agent" pos="{_f(sx, sy, hh + 0.002)}">')
        # See the multi-agent branch: the slides must be explicitly unlimited —
        # the merged humanoid joint default clamped them to ±1.57 m (radian
        # compiler, degrees-parsed range), silently penning the agent.
        out.append('      <joint name="agent_x" type="slide" axis="1 0 0" limited="false" range="-1e5 1e5"/>')
        out.append('      <joint name="agent_y" type="slide" axis="0 1 0" limited="false" range="-1e5 1e5"/>')
        out.append('      <joint name="agent_z" type="slide" axis="0 0 1" limited="false" range="-1e5 1e5"/>')
        # Insert humanoid torso body (with limbs)
        if a.get("type", "box") == "humanoid":
            hum_path = HUMANOID_ASSET_PATH
            with open(hum_path, "r", encoding="utf-8") as f_h:
                hum_xml = f_h.read()
            # torso body (with limbs)
            torso_block = _extract_full_body(hum_xml, "torso")
            if torso_block:
                # The asset defines the figure in its own frame (ground at z=0,
                # torso centre at 1.15). Nested under the agent body — whose
                # origin sits at hh+0.002 — the torso must be offset to t = 1.15
                # - hh - 0.002 = 0.348 so the feet land at the agent root height.
                # Without this the figure floated 0.8 m above the ground and every
                # spawn began with a 0.8 m free-fall impact (the QACC blow-ups).
                old = '<body name="torso" pos="0 0 1.15">'
                assert old in torso_block, (
                    "humanoid asset torso anchor line changed; update the humanoid "
                    "torso offset in playground/world/spec.py")
                # waist_yaw: the rig's only rotational root dof. The head/limb axes
                # can't re-aim a walk by themselves — with the root on 3 slide joints
                # the legs can only sweep along the torso's local +X, so heading
                # control needs the whole upper figure to yaw (Nov 17 heading test).
                # Full ±180 deg (radians for the world compiler) so every compass
                # heading is reachable; damped 2.0, posture-held by the gait PD.
                new = ('<body name="torso" pos="0 0 0.348">'
                       '<joint name="waist_yaw" type="hinge" axis="0 0 1" '
                       'limited="true" range="-3.1416 3.1416" damping="2.0"/>')
                torso_block = torso_block.replace(old, new, 1)
                # humanoid_v2 visual meshes (opt-in). The single-agent path keeps
                # its own inline copy of this block, so the injection has to
                # happen here too — injecting only in _humanoid_torso_block()
                # silently produced a mesh-less model (nmesh 35, ngeom unchanged).
                torso_block = _hv2_visual_inject(torso_block, a)
                # With the waist joint the torso is an articulated child, so the
                # agent container no longer carries welded mass and MuJoCo would
                # refuse (or, with a token mass, destabilise: a near-massless
                # parent makes the waist dof's effective inertia collapse and the
                # solver explodes at mjMINVAL scales). Give it pelvis-class mass
                # and inertia: the slides then carry a ~51 kg figure, and the
                # waist dof keeps a healthy ~1+ kg m^2 about its axis.
                out.append('      <inertial pos="0 0 0" mass="10.0" diaginertia="1.0 1.2 1.0"/>')
                # Locomotion pairing experiment (cfg-switchable). When
                # agent_body.limb_friction_engine is true, limb geoms get the
                # frictionless, normal-only contact pairing the engine pairs with
                # the root-traction drive (same as the box agent base geom; real
                # foot friction under the root-locked rig converts the traction's
                # tipping moment into leg-joint fold instead of walking). When
                # false (the current default — the humanoid walks CONTACT-DRIVEN
                # and needs foot grip), the asset's original friction stays.
                if bool(a.get("limb_friction_engine", False)):
                    # priority="-1": contact parameters come from the geom with
                    # the LOWER priority, and the floor specs compile at priority
                    # 0 — with priority="1" the frictionless pair params never
                    # reached the foot-floor contact and the fold/sink persisted.
                    torso_block = re.sub(
                        r'<geom (?![^>]*friction=)',
                        '<geom condim="1" friction="0 0 0" priority="-1" ',
                        torso_block)
                out.append(torso_block)
        else:
            out.append(f'    <geom name="g:agent" type="box" size="{_f(hw, hw, hh)}" mass="{a["mass"]}" '
                       'friction="0 0 0" priority="1" condim="1" contype="1" conaffinity="3" group="1"/>')
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
