"""Cross-check Unity's import against the pipeline's own report.

Reads Unity's unity_import_report.csv and results/pipeline_report.json and
checks, for every asset and LOD, that Unity imported it, that no material is
missing, that its size matches LOD0, and that Unity's triangle count matches
the pipeline's.

Usage: python unity/compare_with_pipeline.py <unity_import_report.csv> results/pipeline_report.json
"""
import csv
import json
import sys


def main(csv_path, report_path):
    unity = {(r["asset"], r["lod"]): r for r in csv.DictReader(open(csv_path))}
    pipe = {r["asset"]: r for r in json.load(open(report_path))}
    checked = imported = tri_match = no_missing = same_size = 0
    for asset, r in sorted(pipe.items()):
        for lod, tris in r["tris"].items():
            checked += 1
            u = unity.get((asset, lod))
            if not u or u["imported"] != "yes":
                print(f"NOT IMPORTED  {asset} {lod}")
                continue
            imported += 1
            ut = int(u["triangles"])
            if abs(ut - tris) <= max(2, 0.001 * tris):
                tri_match += 1
            else:
                print(f"TRIS DIFFER   {asset} {lod}: unity {ut} vs pipeline {tris}")
            if int(u["missing_materials"]) == 0:
                no_missing += 1
            else:
                print(f"MISSING MAT   {asset} {lod}: {u['missing_materials']}")
            if u["bounds_diff_vs_lod0"] == "" or float(u["bounds_diff_vs_lod0"]) <= 0.01:
                same_size += 1
            else:
                print(f"SIZE CHANGED  {asset} {lod}: {u['bounds_diff_vs_lod0']}")
    print(f"\nfiles checked: {checked}")
    print(f"imported:            {imported}/{checked}")
    print(f"triangles match:     {tri_match}/{imported}")
    print(f"no missing material: {no_missing}/{imported}")
    print(f"same size as LOD0:   {same_size}/{imported}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
