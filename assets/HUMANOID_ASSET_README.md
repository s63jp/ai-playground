# Humanoid Asset for AI Playground (MuJoCo)

## Status: VERIFIED WORKING (Nov 2026)

`assets/humanoid_mujoco.xml` loads and simulates cleanly under mujoco 4.0.1: 14 bodies / 14 named
joints / 12 actuators, 250 steps stable, motors drive their joints. Covered by
`tests/test_humanoid_asset.py` (runs in the HVE suite) and by
`scripts/test_humanoid_integration.py` (standalone run).

## Corrections made to the original export

The exported XML as first shipped **did not load** — mujoco rejected it on the first actuator:
1. **Unnamed joints**: the actuators referenced `joint="torso/spine_L2"` but the joint elements had no
   `name` attributes. Every joint is now named and actuators reference bare joint names.
2. **Ball joint driven by three motors**: the head was one `<joint type="ball"/>` with three motors
   aimed at it. Replaced with three named hinges (`head_pitch` / `head_yaw` / `head_roll`) composed on
   one body (MuJoCo composes multiple joints per body; intermediate bodies must own geoms to have mass).
3. **Layout lying along -Y**: limbs extended sideways (legs/arms along -Y and a `-0.5` root offset).
   The figure now stands along +Z: legs below the torso, arms hanging at the sides, head above, feet on
   a ground plane — a 1.6 m standing figure matching the island's agent body (0.44 × 1.6 m box).
4. **Wrong absolute path**: `meshdir="<home>/Projects/<other-project>/assets/meshes/"`
   pointed at a different project. Removed (the asset uses no mesh files) — the XML is portable.
5. **Joint damping** (1.5) added so a neutral ctrl holds a stable rest pose instead of flailing.


## Joint Hierarchy (14 joints total)

```
RootSpine (welded to world)
├── L2_Spine → L3_Spine → L4_Spine
│   └── Pelvis
│       ├── LeftShoulder → LeftElbow (upper arm + forearm)
│       ├── RightShoulder → RightElbow (upper arm + forearm)  
│       ├── LeftHip → LeftKnee (thigh + shin)
│       └── RightHip → RightKnee (thigh + shin)
│   └── Neck → Head
```

## Physics Properties

- **Torso**: 7.0 kg mass, realistic dimensions
- **Limbs**: density=1000 kg/m³ (human-scale materials)
- **Ground**: friction=1.0 for stable simulation
- **Joints**: All hinge joints with -90° to +90° range

## Usage in MuJoCo Python API

```python
import mujoco

# Load the model
model = mujoco.MjModel.from_xml_path('humanoid_mujoco.xml')
data = mujoco.MjData(model)

# Run simulation for 1 second at 500 Hz
mujoco.mj_step(model, data, nsteps=500)

# Control joints via actuator IDs
actuator_ids = [model.actuator_name2id(name) for name in model.actuator_names]
for i in range(len(actuator_ids)):
    data.ctrl[actuator_ids[i]] = 0.0  # Reset to neutral position

mujoco.mj_forward(model, data)
```

## Usage in Your AI Playground Project

The humanoid asset is optimized for:
- Reinforcement learning (RL) training
- Physics-based character control
- Motion generation and animation
- Multi-agent simulations

## Customization Tips

1. **Edit meshes**: Open `humanoid_rigged.blend` in Blender to modify geometry
2. **Adjust physics**: Edit `humanoid_mujoco.xml` to tune mass/friction values  
3. **Add actuators**: Modify the `<actuator>` section for custom control schemes
4. **Change materials**: Update material properties in XML or Blender

## MuJoCo Export Format Notes

This asset uses MuJoCo's native MJCF format which:
- Supports hierarchical body structures
- Includes mass and inertia tensors for realistic dynamics
- Provides actuator definitions for AI control
- Is compatible with LeRobot, Stable Baselines3, and other RL frameworks

## Next Steps

1. Test the model in MuJoCo to verify physics behavior
2. Integrate into your AI playground's environment class
3. Add reward functions for desired behaviors (walking, balancing, etc.)
4. Train policies using reinforcement learning algorithms

---
Created with Blender 5.2.1 LTS | Optimized for MuJoCo Physics Simulation
