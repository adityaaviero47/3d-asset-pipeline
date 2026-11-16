"""Headless Blender operations: import, audit, cleanup, decimate, export.

Runs inside the `bpy` module (Blender as a Python package), so no GUI is needed.
"""
import math
import bpy  # must come first: registers bmesh
import bmesh
import numpy as np


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def mesh_arrays(obj, world=True):
    """Return (verts Nx3, tris Mx3) of the evaluated mesh (modifiers applied)."""
    dg = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(dg)
    me = eo.to_mesh()
    me.calc_loop_triangles()
    v = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    t = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", t)
    t = t.reshape(-1, 3)
    if world:
        M = np.array(obj.matrix_world)
        v = v @ M[:3, :3].T + M[:3, 3]
    eo.to_mesh_clear()
    return v, t


def audit_object(obj, merge_dist=1e-5):
    """Count the problems a 3D artist would otherwise fix by hand."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    loose_verts = sum(1 for v in bm.verts if not v.link_edges)
    loose_edges = sum(1 for e in bm.edges if not e.link_faces)
    zero_faces = sum(1 for f in bm.faces if f.calc_area() < 1e-12)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold and e.link_faces)
    ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
    # duplicate vertices: how many would "merge by distance" remove
    before = len(bm.verts)
    dup = bmesh.ops.find_doubles(bm, verts=bm.verts, dist=merge_dist)
    dup_verts = len(dup["targetmap"])
    bm.free()
    s = obj.matrix_world.to_scale()
    unapplied = int(any(abs(c - 1.0) > 1e-4 for c in s)
                    or any(abs(r) > 1e-4 for r in obj.matrix_world.to_euler()))
    neg_scale = int(s.x * s.y * s.z < 0)
    return {
        "verts": before, "tris": sum(len(p.vertices) - 2 for p in me.polygons),
        "loose_verts": loose_verts, "loose_edges": loose_edges,
        "zero_area_faces": zero_faces, "non_manifold_edges": non_manifold,
        "ngons": ngons, "duplicate_verts": dup_verts,
        "unapplied_transform": unapplied, "negative_scale": neg_scale,
        "no_uv": int(len(me.uv_layers) == 0),
        "no_material": int(len(me.materials) == 0 or all(m is None for m in me.materials)),
    }


def cleanup_object(obj, merge_dist=1e-5):
    """Fix what the audit found, without changing the visible surface."""
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    if obj.data.users > 1:              # instanced mesh: make single-user first
        obj.data = obj.data.copy()
    if obj.parent is not None:          # bake inherited rotation/scale into the object
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=merge_dist)
    loose = [v for v in bm.verts if not v.link_edges]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    loose_e = [e for e in bm.edges if not e.link_faces]
    bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def purge_unused():
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def set_decimate(obj, ratio):
    mod = obj.modifiers.get("LOD") or obj.modifiers.new("LOD", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = float(ratio)
    mod.use_collapse_triangulate = True
    return mod


def clear_decimate(obj):
    mod = obj.modifiers.get("LOD")
    if mod:
        obj.modifiers.remove(mod)


def export_glb(path, objects):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in objects:
        o.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True, export_texcoords=True,
        export_normals=True, export_materials="EXPORT")


def safe_name(s):
    out = "".join(c if c.isalnum() else "_" for c in s).strip("_")
    return out or "mesh"
