import bpy, json, sys
# Golden per-mesh world-space bounds for the viewer check (run headless in Blender).
OUT = "C:/Users/jamie/ai-playground/playground/dashboard/static/assets/humanoid_v2/blender_mesh_bounds.json"
res = {}
for o in bpy.data.objects:
    if o.type != "MESH":
        continue
    if not any(m.type == "ARMATURE" for m in o.modifiers):
        continue
    mw = o.matrix_world
    pts = [mw @ v.co for v in o.data.vertices]
    mn = [min(p[i] for p in pts) for i in range(3)]
    mx = [max(p[i] for p in pts) for i in range(3)]
    cen = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
    res[o.name] = {"n_verts": len(pts), "min": mn, "max": mx, "centroid": cen}
json.dump({"source": "humanoid_robot_rigged.blend (Blender 5.2.1, world space, metres)",
           "meshes": res}, open(OUT, "w"), indent=1)
print("MESHES", len(res))
print("WROTE", OUT)
