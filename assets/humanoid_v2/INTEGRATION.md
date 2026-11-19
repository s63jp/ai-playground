# humanoid_v2 — visual integration into the MuJoCo rig

Status: **integrated, opt-in, physics-neutral**. Based on `origin/main`
(v4 RL contract, post-B4 head correction).

## What this is

The 35-part Blender robot is attached as **visual geometry only** to the
existing MuJoCo rig. The Blender armature is **not** used as a physics
skeleton — the existing bodies, joints, limits and actuators remain
authoritative.

| compiled physics | legacy | humanoid_v2 |
|---|---:|---:|
| `nq` / `nv` / `nu` / `njnt` / `nbody` | 174 / 152 / 0 / 42 / 39 | **identical** |
| `ngeom` | 60 | 95 (+35 visual) |
| `nmesh` | 0 | 35 |
| colliding geoms | 52 | **52** |
| total body mass | 161.050253 kg | **161.050253 kg** (delta 0.0) |
| per-body mass / inertia / ipos | — | **bit-identical** |
| 300 steps from `qpos0` | — | `qpos`/`qvel`/`qacc` delta **0.000e+00**, contacts 91 = 91 |

Reproduce: `python scripts/humanoid_v2_physics_diff.py` (non-zero exit if any
physics quantity moves).

## Opt-in

Only `agent_body.visual == "humanoid_v2"` changes the generated XML
(`_hv2_on`). Any other value — including a typo — yields the legacy model,
covered by a negative test.

## The mass trap, and the stale-freeze trap behind it

MuJoCo derives a body's mass/inertia from its geoms when no explicit
`<inertial>` is present, so attaching 35 mesh geoms silently changes the mass of
every geom-derived body. The fix is to freeze the legacy inertials to
`assets/meshes/humanoid_v2/inertials.json` and inject them as explicit
`<inertial>` tags.

That fix has a sharp edge. **A frozen table is only valid for the rig it was
measured from.** An earlier version of this work was frozen against a rig whose
head was 6.37 kg at COM +0.26 m. The B4 head-mechanics fix later replaced that
with **1.8 kg at +0.13 m** (the old figure produced a 16.25 N·m gravity moment
against a 12.57 N·m neck capacity — the head could not be held up). Injecting
the stale table on top of the corrected rig would have silently restored the
broken head.

Two guards now make that impossible:

1. **The freeze is discovered, not hardcoded.** `_geom_derived_bodies()` parses
   the asset and freezes only bodies that genuinely lack an explicit
   `<inertial>`. On the current rig that is **11 bodies — the head is correctly
   excluded**, because the asset now gives it one.
2. **The table is fingerprinted.** `inertials.json` records the SHA-256 of
   `assets/humanoid_mujoco.xml`, and `_hv2_data()` refuses to load a table whose
   fingerprint no longer matches, naming the script to re-run.

## RL contract — v4, unchanged

The live contract is **68 observations / 15 actions**
(`obs_dim = 43 + 15 + 2 + 3 + 1 + 4`), with **`waist_yaw` in the action space at
index 14 — action 15**; only the foot yaws stay passive. v3 legacy mode keeps
43/14 and excludes the waist.

The integration adds **no actuators** and does not touch the contract. Proven
two ways:

- direct assertions on `LocoEnv.obs_dim` / `action_dim` / `_EXCLUDED` and on
  `joint_names[14] == "waist_yaw"`;
- a checkpoint round-trip: `save_checkpoint`/`load_checkpoint` verify obs/action
  dims, section slices, joint order, mapping and PD gains, and a checkpoint
  saved against the **legacy** world loads against the **visual** world. One
  contract, which is what "visual only" means.

## Coordinates

Blender Z-up matches MuJoCo Z-up; meshes are exported in body-local frames.
Figure height 1.755 m, soles at `z = 0` (<1e-4).

### Pitfall: reading mesh vertices back

MuJoCo stores `mesh_vert` bbox-centred and **already in the geom frame**, with
`geom_pos = mesh_pos` and `geom_quat = mesh_quat`. The only correct transform:

```python
world = mesh_vert @ geom_xmat.reshape(3, 3).T + geom_xpos
```

Adding `mesh_pos` or re-applying `mesh_quat` double-transforms any part whose
`geom_quat != I` (feet, chest): axes scramble, the measured height inflates to
~2.05 m and connectivity reports ~9 components instead of 1.

### Pitfall: two torso-block paths

`spec.py` builds the humanoid torso in **two** places — `_humanoid_torso_block()`
(multi-agent) and an inline copy in `build_mjcf()` (single-agent). Injecting into
only one produces a model with `nmesh 35` but `ngeom` unchanged: the meshes load
as assets and are never instantiated, and a mass-equality check then passes
*trivially*. Both paths are patched.

## Tests — `tests/test_humanoid_v2_integration.py` (16 passed)

**Happy** — model loads; all 35 meshes present by name; files on disk with a
recorded source SHA-256.

**Validate** — bodies/joints/collision geoms intact; mass & inertia
bit-identical; v4 contract 68/15 with waist at action 15 and no added actuators;
height 1.70–1.80 m with soles on the ground; vertex-level L/R mirror symmetry;
all 35 parts form **one** connected component; 300-step dynamics identical;
visual geoms take zero contacts; articulation moves the head and left-chain
visuals (>1e-4) while the right chain holds (<1e-9) and `qpos0` restores the
exact rest pose; checkpoint round-trip across both worlds.

**Error** — flag off leaves XML untouched (`ngeom == 60`); unknown `visual`
value fails closed; a bad `meshdir` makes MuJoCo refuse to load rather than
silently yielding a mesh-less model.

## Renders

`assets/humanoid_v2/renders/` via `scripts/humanoid_v2_render.py`:
`hv2_front` · `hv2_side` · `hv2_three_quarter` · `hv2_head_detail` ·
`hv2_articulated` · `legacy_front`.

Rest pose shows arms at the sides, not the asset's A-pose — that is the existing
rig's `qpos0`, deliberately untouched.

## Not included (deliberately)

- **No dashboard viewer changes.** `rl_live` already provides checkpoint
  playback in the dashboard; wiring the robot meshes into that path is a
  follow-up, not a parallel player.
- **No neural-drive demo.** Proving the shell is driven by a policy needs a real
  PPO brain; the machine has no trained checkpoint under `experiments/`, and
  `playground/rl/train.py` (an older toy loop) no longer exists on main.

## Reproduce

```bash
./.venv/Scripts/python.exe -m pytest tests/test_humanoid_v2_integration.py -v
./.venv/Scripts/python.exe scripts/humanoid_v2_physics_diff.py
./.venv/Scripts/python.exe scripts/humanoid_v2_freeze_inertials.py   # after any rig change
./.venv/Scripts/python.exe scripts/humanoid_v2_render.py
```
