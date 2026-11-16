import bpy
import xml.etree.ElementTree as ET

def export_to_mujoco():
    """Export the humanoid rig to MuJoCo MJCF format."""
    
    print("=== Exporting Humanoid to MuJoCo Format ===")
    
    tree = ET.Element("mujoco")
    tree.set("compiler", "mesh")
    tree.set("size", "nbody 20 njoint 14 nsite 11")
    
    # Add visual elements
    vis_elem = ET.SubElement(tree, "visual")
    ET.SubElement(vis_elem, "globaloffset").set("fromcamera", "true")
    
    # Add asset directory for meshes
    asset_elem = ET.SubElement(tree, "asset")
    
    # Torso mesh and body with mass properties
    torso_body = ET.SubElement(vis_elem, "body", name="torso")
    ET.SubElement(torso_body, "geom").set(
        type="box", size=(0.3, 0.4, 0.25), density=1000, friction=0.95
    )
    
    # Add body mass properties (MuJoCo needs these)
    ET.SubElement(torso_body, "inert").set(
        mass="7.0", pos="0 -0.2 0"
    )
    
    # Left arm chain
    left_arm_body = ET.SubElement(vis_elem, "body", name="left_shoulder")
    ET.SubElement(left_arm_body, "joint").set(
        type="hinge", axis="0 1 0", pos="0.25 -0.38 0"
    )
    ET.SubElement(left_arm_body, "geom").set(
        type="capsule", size=0.04, fromto="(0.25,-0.38,0) (0.25,-0.79,0)", density=1000
    )
    
    left_forearm_body = ET.SubElement(vis_elem, "body", name="left_elbow")
    ET.SubElement(left_forearm_body, "joint").set(
        type="hinge", axis="0 1 0", pos="0.25 -0.65 0"
    )
    ET.SubElement(left_forearm_body, "geom").set(
        type="capsule", size=0.035, fromto="(0.25,-0.79,0) (0.25,-0.92,0)", density=1000
    )
    
    # Right arm chain
    right_arm_body = ET.SubElement(vis_elem, "body", name="right_shoulder")
    ET.SubElement(right_arm_body, "joint").set(
        type="hinge", axis="0 1 0", pos="-0.25 -0.38 0"
    )
    ET.SubElement(right_arm_body, "geom").set(
        type="capsule", size=0.04, fromto="(-0.25,-0.38,0) (-0.25,-0.79,0)", density=1000
    )
    
    right_forearm_body = ET.SubElement(vis_elem, "body", name="right_elbow")
    ET.SubElement(right_forearm_body, "joint").set(
        type="hinge", axis="0 1 0", pos="-0.25 -0.65 0"
    )
    ET.SubElement(right_forearm_body, "geom").set(
        type="capsule", size=0.035, fromto="(-0.25,-0.79,0) (-0.25,-0.92,0)", density=1000
    )
    
    # Left leg chain (hip joint at pelvis level)
    left_thigh_body = ET.SubElement(vis_elem, "body", name="left_hip")
    ET.SubElement(left_thigh_body, "joint").set(
        type="hinge", axis="0 1 0", pos="0.22 -0.48 0"
    )
    ET.SubElement(left_thigh_body, "geom").set(
        type="capsule", size=0.05, fromto="(0.22,-0.48,0) (0.22,-0.78,0)", density=1000
    )
    
    left_shin_body = ET.SubElement(vis_elem, "body", name="left_knee")
    ET.SubElement(left_shin_body, "joint").set(
        type="hinge", axis="0 1 0", pos="0.22 -0.75 0"
    )
    ET.SubElement(left_shin_body, "geom").set(
        type="capsule", size=0.04, fromto="(0.22,-0.78,0) (0.22,-1.08,0)", density=1000
    )
    
    # Right leg chain
    right_thigh_body = ET.SubElement(vis_elem, "body", name="right_hip")
    ET.SubElement(right_thigh_body, "joint").set(
        type="hinge", axis="0 1 0", pos="-0.22 -0.48 0"
    )
    ET.SubElement(right_thigh_body, "geom").set(
        type="capsule", size=0.05, fromto="(-0.22,-0.48,0) (-0.22,-0.78,0)", density=1000
    )
    
    right_shin_body = ET.SubElement(vis_elem, "body", name="right_knee")
    ET.SubElement(right_shin_body, "joint").set(
        type="hinge", axis="0 1 0", pos="-0.22 -0.75 0"
    )
    ET.SubElement(right_shin_body, "geom").set(
        type="capsule", size=0.04, fromto="(-0.22,-0.78,0) (-0.22,-1.08,0)", density=1000
    )
    
    # Head/body connection (ball joint for full rotation)
    head_body = ET.SubElement(vis_elem, "body", name="head")
    ET.SubElement(head_body, "joint").set(
        type="ball", pos="0 -0.25 0"
    )
    ET.SubElement(head_body, "geom").set(
        type="sphere", size=0.18, density=1000
    )
    
    # Ground plane for simulation
    ground = ET.SubElement(vis_elem, "body")
    ET.SubElement(ground, "geom").set(
        type="plane", size=(10, 10), friction=1.0
    )
    
    # Export to XML file
    xml_str = ET.tostring(tree, encoding='unicode')

    import os
    try:                # Blender sets __file__ for `blender --python` runs
        _HERE = os.path.dirname(os.path.abspath(__file__))
    except NameError:   # pasted into the text editor: fall back to cwd
        _HERE = os.getcwd()
    out_dir = os.environ.get("ASSET_OUT_DIR") or os.path.normpath(os.path.join(_HERE, "..", "assets"))
    os.makedirs(out_dir, exist_ok=True)
    xml_path = os.path.join(out_dir, "humanoid_mujoco.xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)
    
    print(f"✓ Generated MuJoCo XML export to:")
    print(f"  {xml_path}")

if __name__ == "__main__":
    export_to_mujoco()
