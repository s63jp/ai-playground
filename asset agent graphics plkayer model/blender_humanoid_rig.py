import bpy
import math

def create_humanoid_for_mujoco():
    """Create a rigged humanoid asset optimized for MuJoCo physics simulation."""
    
    # Clean scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    print("Cleaning scene...")
    
    # === MATERIALS ===
    skin_mat = bpy.data.materials.new(name="Skin_PBR")
    skin_mat.use_nodes = True
    bsdf = skin_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (0.7, 0.65, 0.6)
    bsdf.inputs['Subsurface'].default_value = 0.3
    bsdf.inputs['Roughness'].default_value = 0.4
    
    fabric_mat = bpy.data.materials.new(name="Fabric_PBR")
    fabric_mat.use_nodes = True
    bsdf_fabric = fabric_mat.node_tree.nodes["Principled BSDF"]
    bsdf_fabric.inputs['Base Color'].default_value = (0.3, 0.25, 0.2)
    bsdf_fabric.inputs['Roughness'].default_value = 0.6
    
    print("Materials created...")
    
    # === BODY PARTS HIERARCHY ===
    
    # Torso/Main Body
    torso_obj = bpy.data.objects.new("Torso", None)
    mesh_data = bpy.data.meshes.new("Torso_Mesh")
    mesh_data.from_pydata(
        vertices=[
            (0, 0.15, 0), (-0.2, -0.3, 0.15), (0.2, -0.3, 0.15),
            (0, -0.45, 0), (-0.2, -0.3, -0.15), (0.2, -0.3, -0.15)
        ],
        edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
        faces=[]
    )
    mesh_data.calc_loop_lengths()
    torso_obj.data = mesh_data
    torso_obj.rotation_mode = 'QUATERNION'
    bpy.context.collection.objects.link(torso_obj)
    
    # Apply scale and position
    torso_obj.scale = (1.0, 1.0, 1.0)
    torso_obj.location = (0, 0, 0.5)
    torso_body = torso_obj
    
    torso_body.data.materials.append(skin_mat)
    
    print("Torso created...")
