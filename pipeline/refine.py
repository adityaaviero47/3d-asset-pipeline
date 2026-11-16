"""Stage 5: close the loop between the geometry metric and what is on screen.

For every LOD that fails the on-screen check (visual_check.py), raise that
asset's decimation ratios one grid step, re-export, re-validate and re-check,
up to MAX_ROUNDS times. LODs that still fail are flagged for an artist.

Usage: python refine.py <assets_dir> <dataset.json,...> <lods_dir> <validator.mjs> <visual.json,...>
"""
import json
import sys
from pathlib import Path

import run_pipeline as P
import visual_check as V
from train_eval import load

MAX_ROUNDS = 4


def main(assets_dir, dataset_paths, lods_dir, validator, visual_paths):
    models = P.fit_models(load(dataset_paths.split(",")))
    visual = {r["asset"]: r for p in visual_paths.split(",") for r in json.load(open(p))}
    reports = {r["asset"]: r for r in json.load(open(Path(lods_dir) / "report.json"))}
    sources = {g.stem: g for g in Path(assets_dir).glob("*/glTF-Binary/*.glb")}
    log = []
    for asset, row in sorted(visual.items()):
        if asset not in sources:
            continue
        failing = [lod for lod in ("LOD1", "LOD2") if not row[lod]["pass"]]
        if not failing:
            continue
        bump = {lod: 0 for lod in failing}
        first = {lod: row[lod]["switch"]["ssim_min"] for lod in failing}
        for rnd in range(1, MAX_ROUNDS + 1):
            for lod in failing:
                bump[lod] += 1
            rep = P.process_asset(sources[asset], models, lods_dir, validator, bump)
            row = V.check_asset(lods_dir, asset)
            failing = [lod for lod in failing if not row[lod]["pass"]]
            if not failing:
                break
        reports[asset] = rep
        visual[asset] = row
        entry = {"asset": asset, "rounds": rnd, "bump": rep["bump"], "still_failing": failing,
                 "ssim_before": first,
                 "ssim_after": {lod: row[lod]["switch"]["ssim_min"] for lod in first},
                 "tris": rep["tris"]}
        log.append(entry)
        print(json.dumps(entry), flush=True)
    Path(lods_dir, "report.json").write_text(json.dumps(list(reports.values()), indent=1))
    Path(lods_dir, "visual.json").write_text(json.dumps(list(visual.values()), indent=1))
    Path(lods_dir, "refine_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
