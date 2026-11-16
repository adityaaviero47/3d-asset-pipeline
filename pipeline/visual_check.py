"""Stage 4: check what a viewer actually sees, not just the geometry.

The surface-error metric ignores shading, UVs and textures. This stage renders
every LOD from 6 fixed views (Blender Workbench, headless) and compares each
LOD against LOD0 in image space:
  silhouette IoU  overlap of the object masks (shape on screen)
  SSIM            structural similarity of the shaded, textured renders,
                  over the object's bounding box at its own brightness range
An LOD passes when both clear the thresholds below on every view, rendered at
the largest screen size the LOD is actually shown at (its LODGroup switch
point). Full-size scores are kept in the report as a stricter reference.

Usage: python visual_check.py <lods_dir> <out.json>
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image
from skimage.metrics import structural_similarity

RES = 400
# largest on-screen height (fraction of view) at which each LOD is shown;
# matches the LODGroup switch points in unity/Editor/LodSetImporter.cs
SWITCH = {"LOD1": 0.50, "LOD2": 0.20}
VIEWS = [(az, el) for el in (20, 50) for az in (30, 150, 270)]   # 6 views
IOU_MIN, SSIM_MIN = 0.97, 0.90


def load(glb):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return (lo + hi) / 2, (hi - lo).length


def render_views(objs, centre, size, mode, tmp, res=RES):
    scn = bpy.context.scene
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scn.collection.objects.link(cam)
    cam.data.clip_start = size * 0.01   # tiny assets (a 2 cm part) sit inside the default 0.1 m
    cam.data.clip_end = size * 10
    scn.camera = cam
    scn.render.engine = "BLENDER_WORKBENCH"
    scn.render.resolution_x = scn.render.resolution_y = res
    sh = scn.display.shading
    sh.light = "FLAT" if mode == "mask" else "STUDIO"
    sh.color_type = "SINGLE" if mode == "mask" else "TEXTURE"
    sh.single_color = (1, 1, 1)
    scn.world = scn.world or bpy.data.worlds.new("w")
    sh.background_type = "VIEWPORT"
    sh.background_color = (0, 0, 0)
    scn.render.film_transparent = False
    imgs = []
    for k, (az, el) in enumerate(VIEWS):
        a, e = math.radians(az), math.radians(el)
        d = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        cam.location = centre + d * size * 1.4
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
        path = f"{tmp}/{mode}_{k}.png"
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        imgs.append(np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255)
    return imgs


def object_ssim(s0, s1, mask, pad=4):
    """SSIM over the object's bounding box only, scaled to its own brightness
    range, so a dark object on a black background is not scored as a match."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 1.0
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    c0, c1 = s0[y0:y1, x0:x1], s1[y0:y1, x0:x1]
    rng = max(float(c0.max() - c0.min()), 1e-3)
    win = min(7, (min(c0.shape) // 2) * 2 - 1)
    return float(structural_similarity(c0, c1, data_range=rng, win_size=max(win, 3)))


def check_asset(lods_dir, asset):
    """Render and score LOD1 and LOD2 of one asset against its LOD0."""
    lods_dir = Path(lods_dir)
    lod0 = lods_dir / f"{asset}_LOD0.glb"
    with tempfile.TemporaryDirectory() as tmp:
        centre, size = bounds(load(lod0))          # same cameras for every LOD
        row = {"asset": asset}
        ref = {}
        lod0 = load(lod0)
        for lod, frac in SWITCH.items():          # LOD0 references at each check size
            for tag, res in (("full", RES), ("switch", int(RES * frac))):
                key = (lod, tag)
                ref[key] = (render_views(lod0, centre, size, "mask", tmp, res),
                            render_views(lod0, centre, size, "shaded", tmp, res))
        for lod, frac in SWITCH.items():
            objs = load(lods_dir / f"{asset}_{lod}.glb")
            row[lod] = {}
            for tag, res in (("full", RES), ("switch", int(RES * frac))):
                masks = render_views(objs, centre, size, "mask", tmp, res)
                shaded = render_views(objs, centre, size, "shaded", tmp, res)
                r_masks, r_shaded = ref[(lod, tag)]
                ious, ssims = [], []
                for m0, m1, s0, s1 in zip(r_masks, masks, r_shaded, shaded):
                    a, b = m0 > 0.05, m1 > 0.05
                    ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
                    ssims.append(object_ssim(s0, s1, a | b))
                row[lod][tag] = {"res": res, "iou_min": min(ious), "iou_mean": float(np.mean(ious)),
                                 "ssim_min": min(ssims), "ssim_mean": float(np.mean(ssims)),
                                 "pass": min(ious) >= IOU_MIN and min(ssims) >= SSIM_MIN}
            row[lod]["pass"] = row[lod]["switch"]["pass"]
    return row


def main(lods_dir, out_json):
    results = []
    for lod0 in sorted(Path(lods_dir).glob("*_LOD0.glb")):
        results.append(check_asset(lods_dir, lod0.name[:-len("_LOD0.glb")]))
        print(json.dumps(results[-1]), flush=True)
    Path(out_json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
