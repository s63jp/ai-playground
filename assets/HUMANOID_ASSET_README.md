# Humanoid Asset for AI Playground (MuJoCo)

## Files Created

1. **humanoid_rigged.blend** - Blender source file with rigged humanoid
2. **humanoid_mujoco.xml** - MuJoCo physics simulation model

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
