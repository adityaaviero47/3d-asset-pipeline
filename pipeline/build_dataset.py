"""Stage 1: audit and clean every asset, then sweep decimation ratios per mesh.

For every mesh with enough triangles, it decimates at each ratio in RATIOS and
measures the surface error against the cleaned original. The smallest ratio
that stays within the error budget is the label the ML model learns to predict.

Usage: python build_dataset.py <assets_dir> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import blender_ops as B
import geometry as G

RATIOS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_TRIS = 500


def main(assets_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audits, rows = [], []
    for glb in sorted(Path(assets_dir).glob("*/glTF-Binary/*.glb")):
        asset = glb.stem
        B.reset_scene()
        t0 = time.time()
        objs = B.import_glb(glb)
        before = {o.name: B.audit_object(o) for o in objs}
        for o in objs:
            B.cleanup_object(o)
        B.purge_unused()
        after = {o.name: B.audit_object(o) for o in objs}
        clean_s = time.time() - t0
        audits.append({"asset": asset, "meshes": len(objs), "clean_seconds": clean_s,
                       "before": before, "after": after})
        print(f"{asset:28s} meshes={len(objs):3d} clean={clean_s:5.1f}s", flush=True)

        for o in objs:
            v0, t0_ = B.mesh_arrays(o)
            if len(t0_) < MIN_TRIS:
                continue
            feats = G.mesh_features(v0, t0_)
            sweep = []
            for r in RATIOS:
                if r == 1.0:
                    sweep.append((r, len(t0_), 0.0, 0.0))
                    continue
                B.set_decimate(o, r)
                v1, t1 = B.mesh_arrays(o)
                p95, mx = G.surface_error(v0, t0_, v1, t1)
                sweep.append((r, len(t1), p95, mx))
            B.clear_decimate(o)
            rows.append({"asset": asset, "mesh": o.name, "tris": int(len(t0_)),
                         **feats, "sweep": sweep})
    (out / "audit.json").write_text(json.dumps(audits, indent=1))
    (out / "dataset.json").write_text(json.dumps(rows))
    print("meshes in dataset:", len(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
