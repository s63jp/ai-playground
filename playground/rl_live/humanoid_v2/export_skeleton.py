import bpy, json
from mathutils import Matrix

OUT = "C:/Users/jamie/ai-playground/playground/rl_live/humanoid_v2/blender_skeleton.json"
arm = bpy.data.objects["humanoid_rig"]
assert arm.type == 'ARMATURE'

def m4(M):
    return [[round(float(M[r][c]), 6) for c in range(4)] for r in range(4)]

data = {
    "source": "humanoid-robot/assets/humanoid_robot_rigged.blend",
    "blender_version": bpy.app.version_string,
    "armature": arm.name,
    "armature_matrix_world": m4(arm.matrix_world),
    "unit_scale": bpy.context.scene.unit_settings.scale_length,
    "bones": [], "meshes": [], "other_objects": [],
}
for b in arm.data.bones:
    data["bones"].append({
        "name": b.name,
        "parent": b.parent.name if b.parent else None,
        "matrix_local": m4(b.matrix_local),          # armature-space rest
        "head_local": [round(float(v), 6) for v in b.head_local],
        "tail_local": [round(float(v), 6) for v in b.tail_local],
    })

for o in bpy.data.objects:
    if o.type != 'MESH':
        data["other_objects"].append({"name": o.name, "type": o.type})
        continue
    groups = [g.name for g in o.vertex_groups]
    dominant, secondary_verts, unweighted, max_second = {}, 0, 0, 0.0
    for v in o.data.vertices:
        ws = sorted(((o.vertex_groups[g.group].name, g.weight) for g in v.groups),
                    key=lambda t: -t[1])
        if not ws:
            unweighted += 1
            continue
        dominant[ws[0][0]] = dominant.get(ws[0][0], 0) + 1
        if len(ws) > 1 and ws[1][1] > 0.01:
            secondary_verts += 1
            max_second = max(max_second, ws[1][1])
    data["meshes"].append({
        "name": o.name,
        "parent": o.parent.name if o.parent else None,
        "parent_type": o.parent_type,
        "matrix_world": m4(o.matrix_world),
        "n_verts": len(o.data.vertices),
        "armature_modifiers": [m.object.name if m.object else None
                               for m in o.modifiers if m.type == 'ARMATURE'],
        "dominant_bone_counts": dominant,
        "unweighted_verts": unweighted,
        "verts_with_secondary_weight_gt_0.01": secondary_verts,
        "max_secondary_weight": round(max_second, 4),
        "n_vertex_groups": len(groups),
    })

with open(OUT, "w") as f:
    json.dump(data, f, indent=1)
print("WROTE", OUT, "bones", len(data["bones"]), "meshes", len(data["meshes"]))
