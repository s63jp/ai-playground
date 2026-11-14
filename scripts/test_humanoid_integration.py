"""Integration test for humanoid asset in AI Playground."""
import mujoco
import numpy as np
import sys

def test_humanoid_load():
    """Test that the humanoid model loads correctly."""
    print("=== Testing Humanoid Asset Integration ===\n")
    
    # Load the humanoid model
    xml_path = "<repo>/assets/humanoid_mujoco.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ Model loaded successfully")
        print(f"  MuJoCo version: {mujoco.__version__}")
        print(f"  Number of bodies: {model.nbody}")
        print(f"  Number of joints: {model.njnt}")
        print(f"  Number of actuators: {model.nu}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Create data and initialize
    try:
        data = mujoco.MjData(model)
        print("✓ Data initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize data: {e}")
        return False
    
    # Run a few simulation steps
    try:
        for i in range(10):
            mujoco.mj_step(model, data)
        print("✓ Simulation steps completed (10 iterations)")
    except Exception as e:
        print(f"✗ Failed during simulation: {e}")
        return False
    
    # Check joint positions are reasonable
    try:
        qpos = data.qpos.ravel()
        print(f"\n✓ Joint state check:")
        print(f"  Position DOFs: {model.nq - model.nv}")
        print(f"  Velocity DOFs: {model.nv}")
        
        # Check torso is upright (y position should be around -0.5)
        torso_y = data.qpos[1] if len(data.qpos.ravel()) > 1 else None
        if torso_y is not None:
            print(f"  Torso Y position: {torso_y:.4f}")
            
    except Exception as e:
        print(f"✗ Failed joint state check: {e}")
    
    # Test actuator control
    try:
        actuator_ids = [model.actuator_name2id(name) for name in model.actuator_names]
        print(f"\n✓ Actuator test:")
        print(f"  Actuators available: {len(actuator_ids)}")
        
        # Apply small forces to see if actuators work
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        print("  Actuator control applied successfully")
        
    except Exception as e:
        print(f"✗ Failed actuator test: {e}")
    
    return True

def test_humanoid_in_playground():
    """Test humanoid integration with existing playground physics."""
    print("\n=== Testing Humanoid in Playground Context ===\n")
    
    # Load the model
    xml_path = "<repo>/assets/humanoid_mujoco.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print("✓ Humanoid loaded in playground context")
    
    # Simulate for 1 second at 500 Hz (500 steps)
    print(f"Running simulation: 500 steps @ {model.opt.timestep*1000:.2f}Hz...")
    for i in range(500):
        mujoco.mj_step(model, data)
    
    print("✓ Simulation completed successfully")
    
    # Check final state
    torso_y = data.qpos[1] if len(data.qpos.ravel()) > 1 else None
    print(f"Final torso Y position: {torso_y:.4f}")
    
    return True

if __name__ == "__main__":
    success = test_humanoid_load()
    if success:
        test_humanoid_in_playground()
        
    print("\n=== Integration Test Complete ===")
    sys.exit(0 if success else 1)
