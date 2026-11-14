import bpy
import math

# Continue building the humanoid rig
torso_body = bpy.data.objects["Torso"]

def create_bone(name, location, rotation, parent=None):
    """Create a bone with proper hierarchy."""
    bone = bpy.data.bones.new(name, type='BONE')
    if parent:
        parent.bones.append(bone)
    else:
        bpy.context.scene.skeleton.bones.link(bone)
    
    # Set bone transform (position and rotation at rest pose)
    bone.head = location[0], location[1], location[2]
    bone.tail = location[0]+0.01, location[1]+0.01, location[2]+0.01
    
    return bone

# Create skeleton hierarchy for MuJoCo-compatible rigging
root_spine_bone = create_bone("RootSpine", (0, 0, 0), (0, 0, 0))

L2_bone = create_bone("L2_Spine", (0, -0.15, 0), (math.radians(-30), math.radians(0), math.radians(0)), root_spine_bone)
L3_bone = create_bone("L3_Spine", (0, -0.18, 0), (math.radians(-25), math.radians(0), math.radians(0)), L2_bone)
L4_bone = create_bone("L4_Spine", (0, -0.20, 0), (math.radians(-20), math.radians(0), math.radians(0)), L3_bone)

pelvis_bone = create_bone("Pelvis", (0, -0.45, 0), (math.radians(10), math.radians(-5), math.radians(0)), L4_bone)

print("Spine and pelvis bones created...")

# === LEFT ARM HIERARCHY ===
shoulder_L_bone = create_bone("LeftShoulder", (0.25, -0.38, 0), (math.radians(0), math.radians(170), math.radians(0)), pelvis_bone)
elbow_L_bone = create_bone("LeftElbow", (0.25, -0.65, 0), (math.radians(-45), math.radians(0), math.radians(0)), shoulder_L_bone)

# Upper arm mesh
upper_arm_obj = bpy.data.objects.new("UpperArm_Left", None)
upper_arm_mesh = bpy.data.meshes.new("UpperArm_Mesh")
upper_arm_mesh.from_pydata(
    vertices=[
        (0.25, -0.65, 0), (0.18, -0.73, 0.04), (0.18, -0.73, -0.04),
        (0.25, -0.79, 0), (0.18, -0.73, -0.04), (0.18, -0.73, 0.04)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
upper_arm_obj.data = upper_arm_mesh
upper_arm_obj.rotation_mode = 'QUATERNION'
bpy.context.collection.objects.link(upper_arm_obj)
upper_arm_obj.parent = shoulder_L_bone
upper_arm_obj.location = (0, 0, 0.15)

# Lower arm mesh
lower_arm_obj = bpy.data.objects.new("LowerArm_Left", None)
lower_arm_mesh = bpy.data.meshes.new("LowerArm_Mesh")
lower_arm_mesh.from_pydata(
    vertices=[
        (0.25, -0.79, 0), (0.18, -0.86, 0.03), (0.18, -0.86, -0.03),
        (0.25, -0.92, 0), (0.18, -0.86, -0.03), (0.18, -0.86, 0.03)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
lower_arm_obj.data = lower_arm_mesh
lower_arm_obj.rotation_mode = 'QUATERNION'
bpy.context.collection.objects.link(lower_arm_obj)
lower_arm_obj.parent = elbow_L_bone
lower_arm_obj.location = (0, 0, 0.15)

print("Left arm created...")

# === RIGHT ARM HIERARCHY ===
shoulder_R_bone = create_bone("RightShoulder", (-0.25, -0.38, 0), (math.radians(0), math.radians(-170), math.radians(0)), pelvis_bone)
elbow_R_bone = create_bone("RightElbow", (-0.25, -0.65, 0), (math.radians(-45), math.radians(0), math.radians(0)), shoulder_R_bone)

upper_arm_obj_R = bpy.data.objects.new("UpperArm_Right", None)
upper_arm_mesh_R = bpy.data.meshes.new("UpperArm_Mesh_R")
upper_arm_mesh_R.from_pydata(
    vertices=[(-0.25, -0.65, 0), (-0.18, -0.73, 0.04), (-0.18, -0.73, -0.04),
              (-0.25, -0.79, 0), (-0.18, -0.73, -0.04), (-0.18, -0.73, 0.04)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
upper_arm_obj_R.data = upper_arm_mesh_R
bpy.context.collection.objects.link(upper_arm_obj_R)
upper_arm_obj_R.parent = shoulder_R_bone
upper_arm_obj_R.location = (0, 0, 0.15)

lower_arm_obj_R = bpy.data.objects.new("LowerArm_Right", None)
lower_arm_mesh_R = bpy.data.meshes.new("LowerArm_Mesh_R")
lower_arm_mesh_R.from_pydata(
    vertices=[(-0.25, -0.79, 0), (-0.18, -0.86, 0.03), (-0.18, -0.86, -0.03),
              (-0.25, -0.92, 0), (-0.18, -0.86, -0.03), (-0.18, -0.86, 0.03)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
lower_arm_obj_R.data = lower_arm_mesh_R
bpy.context.collection.objects.link(lower_arm_obj_R)
lower_arm_obj_R.parent = elbow_R_bone
lower_arm_obj_R.location = (0, 0, 0.15)

print("Right arm created...")

# === LEFT LEG HIERARCHY ===
hip_L_bone = create_bone("LeftHip", (0.22, -0.48, 0), (math.radians(0), math.radians(-90), math.radians(0)), pelvis_bone)
knee_L_bone = create_bone("LeftKnee", (0.22, -0.75, 0), (math.radians(-160), math.radians(0), math.radians(0)), hip_L_bone)

upper_leg_obj = bpy.data.objects.new("UpperLeg_Left", None)
upper_leg_mesh = bpy.data.meshes.new("UpperLeg_Mesh")
upper_leg_mesh.from_pydata(
    vertices=[
        (0.22, -0.48, 0), (0.15, -0.63, 0.05), (0.15, -0.63, -0.05),
        (0.22, -0.78, 0), (0.15, -0.63, -0.05), (0.15, -0.63, 0.05)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
upper_leg_obj.data = upper_leg_mesh
bpy.context.collection.objects.link(upper_leg_obj)
upper_leg_obj.parent = hip_L_bone
upper_leg_obj.location = (0, 0, 0.18)

lower_leg_obj = bpy.data.objects.new("LowerLeg_Left", None)
lower_leg_mesh = bpy.data.meshes.new("LowerLeg_Mesh")
lower_leg_mesh.from_pydata(
    vertices=[
        (0.22, -0.78, 0), (0.15, -0.93, 0.04), (0.15, -0.93, -0.04),
        (0.22, -1.08, 0), (0.15, -0.93, -0.04), (0.15, -0.93, 0.04)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
lower_leg_obj.data = lower_leg_mesh
bpy.context.collection.objects.link(lower_leg_obj)
lower_leg_obj.parent = knee_L_bone
lower_leg_obj.location = (0, 0, 0.18)

print("Left leg created...")

# === RIGHT LEG HIERARCHY ===
hip_R_bone = create_bone("RightHip", (-0.22, -0.48, 0), (math.radians(0), math.radians(-90), math.radians(0)), pelvis_bone)
knee_R_bone = create_bone("RightKnee", (-0.22, -0.75, 0), (math.radians(-160), math.radians(0), math.radians(0)), hip_R_bone)

upper_leg_obj_R = bpy.data.objects.new("UpperLeg_Right", None)
upper_leg_mesh_R = bpy.data.meshes.new("UpperLeg_Mesh_R")
upper_leg_mesh_R.from_pydata(
    vertices=[(-0.22, -0.48, 0), (-0.15, -0.63, 0.05), (-0.15, -0.63, -0.05),
              (-0.22, -0.78, 0), (-0.15, -0.63, -0.05), (-0.15, -0.63, 0.05)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
upper_leg_obj_R.data = upper_leg_mesh_R
bpy.context.collection.objects.link(upper_leg_obj_R)
upper_leg_obj_R.parent = hip_R_bone
upper_leg_obj_R.location = (0, 0, 0.18)

lower_leg_obj_R = bpy.data.objects.new("LowerLeg_Right", None)
lower_leg_mesh_R = bpy.data.meshes.new("LowerLeg_Mesh_R")
lower_leg_mesh_R.from_pydata(
    vertices=[(-0.22, -0.78, 0), (-0.15, -0.93, 0.04), (-0.15, -0.93, -0.04),
              (-0.22, -1.08, 0), (-0.15, -0.93, -0.04), (-0.15, -0.93, 0.04)
    ],
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
    faces=[]
)
lower_leg_obj_R.data = lower_leg_mesh_R
bpy.context.collection.objects.link(lower_leg_obj_R)
lower_leg_obj_R.parent = knee_R_bone
lower_leg_obj_R.location = (0, 0, 0.18)

print("Right leg created...")

# === HEAD HIERARCHY ===
neck_bone = create_bone("Neck", (0, -0.25, 0), (math.radians(0), math.radians(-45), math.radians(0)), L3_bone)

head_obj = bpy.data.objects.new("Head", None)
head_mesh = bpy.data.meshes.new("Head_Mesh")
head_vertices = [
    (0, -0.25, 0), (-0.18, -0.43, 0.12), (0.18, -0.43, 0.12),
    (0, -0.61, 0), (-0.18, -0.43, -0.12), (0.18, -0.43, -0.12)
]
head_edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]
head_mesh.from_pydata(head_vertices, head_edges, [])
head_obj.data = head_mesh
bpy.context.collection.objects.link(head_obj)
head_obj.parent = neck_bone
head_obj.location = (0, 0, 0.18)

print("Head created...")

# === SAVE AS BLEND ===
import os
os.makedirs("<home>/Projects/<other-project>/assets", exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath="<home>/Projects/<other-project>/assets/humanoid_rigged.blend")
print(f"Saved rigged humanoid to: <home>/Projects/<other-project>/assets/humanoid_rigged.blend")
