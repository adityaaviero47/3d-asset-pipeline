"""Render an exported GLB to PNG with Blender's Workbench engine (headless, CPU).

Usage: python render_preview.py <in.glb> <out.png> [wireframe]
"""
import sys

import bpy
from mathutils import Vector


def main(glb, png, wire=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    centre, size = (lo + hi) / 2, (hi - lo).length
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.scene.collection.objects.link(cam)
    cam.location = centre + Vector((0.75, -0.95, 0.45)) * size
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_start = size * 0.01   # tiny assets (a 2 cm part) sit inside the default 0.1 m
    cam.data.clip_end = size * 10
    scn = bpy.context.scene
    scn.camera = cam
    scn.render.engine = "BLENDER_WORKBENCH"
    scn.display.shading.light = "STUDIO"
    scn.display.shading.color_type = "MATERIAL"
    scn.display.shading.show_object_outline = True
    if wire:
        for o in objs:
            m = o.modifiers.new("w", "WIREFRAME")
            m.thickness = size * 0.0006
            m.use_replace = False
    scn.render.resolution_x, scn.render.resolution_y = 900, 600
    scn.render.filepath = png
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3)
