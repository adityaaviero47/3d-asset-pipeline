"""Stage 3: the production run. Raw GLB in -> clean, engine-ready LODs out.

For every asset: import -> audit -> cleanup -> per mesh, the ML model picks the
LOD1 / LOD2 decimation ratios -> verify each LOD's surface error once (step up
if over budget) -> export <asset>_LOD0/1/2.glb (+Y up, metres, transforms
applied) -> validate every file with the Khronos glTF Validator -> report.

Textures are capped per LOD (LOD1 1024 px, LOD2 512 px), since textures, not
triangles, dominate file size and GPU memory on XR headsets.

Usage: python run_pipeline.py <assets_dir> <dataset.json,...> <out_dir> <validator.mjs>
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import blender_ops as B
import geometry as G
from train_eval import FEATURES, label, load

LODS = {"LOD1": 0.0005, "LOD2": 0.002}     # error budgets (fraction of bbox diagonal)
TEX_CAP = {"LOD1": 1024, "LOD2": 512}      # longest texture side per LOD (pixels)
GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_TRIS = 500


def fit_models(rows):
    X = np.array([[r[f] for f in FEATURES] for r in rows])
    models = {}
    for lod, budget in LODS.items():
        y = np.log([label(r["sweep"], budget) for r in rows])
        m = GradientBoostingRegressor(loss="quantile", alpha=0.6, n_estimators=300,
                                      max_depth=3, learning_rate=0.05, subsample=0.8,
                                      random_state=0).fit(X, y)
        models[lod] = m
    return models


def snap_up(r):
    return next(g for g in GRID if g >= r - 1e-9)


def process_asset(glb, models, out, validator, bump=None):
    """Run one asset end to end. bump[lod] raises every mesh's ratio by that
    many grid steps (used when the on-screen check fails)."""
    bump = bump or {}
    out = Path(out)
    asset = glb.stem
    t_start = time.time()
    B.reset_scene()
    objs = B.import_glb(glb)
    before = [B.audit_object(o) for o in objs]
    for o in objs:
        B.cleanup_object(o)
    B.purge_unused()
    after = [B.audit_object(o) for o in objs]
    base = {o.name: B.mesh_arrays(o) for o in objs}
    lod_tris = {"LOD0": int(sum(len(t) for _, t in base.values()))}
    evals = 0
    files = {}
    B.export_glb(out / f"{asset}_LOD0.glb", objs)
    files["LOD0"] = out / f"{asset}_LOD0.glb"
    for lod, budget in LODS.items():
        total = 0
        for o in objs:
            v0, t0 = base[o.name]
            if len(t0) < MIN_TRIS:
                B.clear_decimate(o)
                total += len(t0)
                continue
            f = G.mesh_features(v0, t0)
            r = snap_up(float(np.exp(models[lod].predict([[f[k] for k in FEATURES]])[0])))
            r = GRID[min(GRID.index(r) + bump.get(lod, 0), len(GRID) - 1)]
            while r < 1.0:
                B.set_decimate(o, r)
                v1, t1 = B.mesh_arrays(o)
                evals += 1
                if G.surface_error(v0, t0, v1, t1)[0] <= budget:
                    break
                r = GRID[GRID.index(r) + 1]
            if r >= 1.0:
                B.clear_decimate(o)
                total += len(t0)
            else:
                total += len(t1)
        lod_tris[lod] = int(total)
        B.cap_textures(TEX_CAP[lod])
        path = out / f"{asset}_{lod}.glb"
        B.export_glb(path, objs)
        files[lod] = path
    for o in objs:
        B.clear_decimate(o)
    secs = time.time() - t_start
    v = json.loads(subprocess.run(["node", validator, *map(str, files.values())],
                                  capture_output=True, text=True, check=True).stdout)
    issues = lambda a: sum(x[k] for x in a for k in
                           ("loose_verts", "loose_edges", "zero_area_faces",
                            "duplicate_verts", "unapplied_transform", "negative_scale"))
    return {
        "asset": asset, "meshes": len(objs), "seconds": round(secs, 2),
        "fixable_issues_before": issues(before), "fixable_issues_after": issues(after),
        "audit_before": {k: sum(x[k] for x in before) for k in before[0]} if before else {},
        "audit_after": {k: sum(x[k] for x in after) for k in after[0]} if after else {},
        "tris": lod_tris, "measure_runs": evals,
        "validator_errors": {k: v[str(p)]["errors"] for k, p in files.items()},
        "validator_warnings": {k: v[str(p)]["warnings"] for k, p in files.items()},
        "bytes": {k: p.stat().st_size for k, p in files.items()},
        "bump": dict(bump),
    }


def main(assets_dir, dataset_paths, out_dir, validator):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = fit_models(load(dataset_paths.split(",")))
    report = []
    for glb in sorted(Path(assets_dir).glob("*/glTF-Binary/*.glb")):
        report.append(process_asset(glb, models, out, validator))
        print(json.dumps(report[-1]), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
