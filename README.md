# 3D Asset Automation Pipeline: Blender Cleanup, ML-Driven LODs, Engine-Ready glTF

Raw 3D models rarely drop straight into a game engine. They come with
duplicate vertices, degenerate faces, rotations and negative scales baked into
parent nodes, and far too many triangles for real-time use. Artists fix this by
hand, mesh by mesh, and then guess decimation ratios for each level of detail
(LOD), checking each guess by eye.

This pipeline does all of that headlessly with Blender's Python API (`bpy`),
and replaces the guessing with a machine-learning model that predicts how far
each mesh can be simplified before its surface visibly drifts.

![LOD0 vs LOD2](docs/lod_compare.png)

*ToyCar (Khronos sample asset): cleaned LOD0 at 103,793 triangles vs LOD2 at
10,377 (10%). Some shading artefacts are visible on the door panel and glass at
LOD2, which is the far-distance level.*

## What it does

```
raw .glb ─► audit ─► cleanup ─► ML picks LOD ratios ─► verify error ─► export LOD0/1/2 .glb ─► Khronos glTF Validator
```

1. **Audit** every mesh: duplicate vertices, loose geometry, zero-area faces,
   non-manifold edges, unapplied or inherited transforms, negative scale,
   missing UVs or materials.
2. **Cleanup** without changing the visible surface: unparent with transforms
   kept, apply rotation and scale, merge duplicates, dissolve degenerate faces,
   delete loose geometry, recalculate normals, purge unused data.
3. **Pick decimation ratios with ML.** For each mesh, a gradient-boosting model
   reads 12 shape features (triangle count, density, sharp-edge fraction,
   dihedral angles, boundary edges, planarity, aspect ratios…) and predicts the
   smallest ratio that keeps the surface error within budget.
4. **Verify** each LOD once against the cleaned mesh: two-sided
   point-to-surface distance (95th percentile, as a fraction of the bounding-box
   diagonal). If it is over budget, step up one ratio and re-check.
5. **Export** `<asset>_LOD0/1/2.glb` (+Y up, metres, modifiers applied) and
   **validate** every file against the glTF 2.0 spec with the official Khronos
   glTF Validator.

LOD budgets: LOD1 ≤ 0.05% of the bounding-box diagonal, LOD2 ≤ 0.2%.

## Data

36 real production-style assets from the
[Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
(cars, a truck, props, furniture, a city block): **392 meshes, 2.75M
triangles**. The ML dataset is the 196 meshes with 500+ triangles, each swept
across 11 decimation ratios with the error measured at every step.

## Results

### Cleanup (all 36 assets)

| Defect | Before | After |
|---|---:|---:|
| Duplicate vertices | 643,777 | 30 |
| Unapplied / inherited transforms | 304 | 0 |
| Negative scale | 13 | 0 |
| Zero-area faces | 843 | 242 |

### ML vs baselines (LOD1 budget, tested on unseen assets)

5-fold cross-validation grouped by asset, so every number below comes from
assets the model never saw in training.

| Policy | Meshes within budget | Triangles kept | Decimate + measure runs per mesh |
|---|---:|---:|---:|
| Fixed 50% ratio (typical batch script) | 67.3% | 50.0% | 0 |
| Fixed 20% ratio | 34.2% | 20.0% | 0 |
| Binary search over ratios (exact) | 100% | 19.8% | 3.75 |
| **ML predict + verify** | **100%** | **24.0%** | **1.39** |

- ML needs **63% fewer** decimate-and-measure runs than binary search while
  keeping every mesh within budget and cutting 76% of triangles (search: 80%).
- The ML guess passes on the **first try for 77%** of meshes.
- A fixed 50% ratio, what a simple script would do, breaks the error budget on
  **one mesh in three**.
- At the looser LOD2 budget the pattern holds: ML keeps 11.4% of triangles in
  1.36 runs per mesh vs 8.8% in 3.81 for search.

### Production run

| | |
|---|---|
| Files exported | 108 (36 assets × LOD0/1/2) |
| Khronos glTF Validator errors | **0** |
| LOD1 / LOD2 triangles | 24.9% / 12.2% of LOD0 |
| Time | 9.1 s per asset on a 2-core CPU (329 s for all 36) |

Full per-asset numbers: [`results/pipeline_report.json`](results/pipeline_report.json),
[`results/ml_vs_baselines.json`](results/ml_vs_baselines.json).

## Engine compatibility

Output is glTF 2.0 binary (`.glb`) with +Y up and metre units, the conventions
Unity (via glTFast) and Unreal Engine 5 (via Interchange) import natively. Every
file passes the Khronos validator with zero errors. The files have **not** been
imported into Unity or Unreal as part of this repo; that is the next step.

## Limitations

- Error is geometric only. It does not measure UV stretching or shading
  artefacts, which do show up at LOD2 on some parts (see the image above).
- 242 of 843 zero-area faces survive cleanup: Blender's degenerate-dissolve
  does not remove all of them. Not investigated yet.
- Cleanup flattens the node hierarchy (meshes are unparented to bake their
  transforms). Animated and skinned assets need a different path.
- Texture size dominates file size, so LODs barely shrink the `.glb` files.
  Texture downscaling is not done yet.

## Run it

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt   # bpy = Blender 5.3 as a module
(cd tools && npm install)                                            # Khronos glTF Validator

# assets: sparse-clone Models/*/glTF-Binary from KhronosGroup/glTF-Sample-Assets into data/
cd pipeline
python build_dataset.py ../data ../out           # audit, cleanup, ratio sweep  (~20 min, 2 cores)
python train_eval.py ../out/dataset.json         # ML vs baselines, grouped CV
python run_pipeline.py ../data ../out/dataset.json ../out/lods ../tools/validate.mjs
python render_preview.py ../out/lods/ToyCar_LOD2.glb preview.png
cd .. && python -m pytest -q tests               # metric sanity tests
```

## Layout

```
pipeline/blender_ops.py     import, audit, cleanup, decimate, export (bpy)
pipeline/geometry.py        surface-error metric and 12 mesh shape features
pipeline/build_dataset.py   stage 1: audit + cleanup + ratio sweep per mesh
pipeline/train_eval.py      stage 2: ML model vs fixed-ratio and search baselines
pipeline/run_pipeline.py    stage 3: production LOD export + validation + report
pipeline/render_preview.py  headless Workbench render of any .glb
tools/validate.mjs          Khronos glTF Validator wrapper
tests/                      metric tests (self-error is zero, scale-free, monotonic)
```
