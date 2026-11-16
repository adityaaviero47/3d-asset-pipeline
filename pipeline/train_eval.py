"""Stage 2: learn the decimation ratio from mesh shape, test on unseen assets.

Label: the smallest swept ratio whose surface error (95th percentile, as a
fraction of the bounding-box diagonal) stays within BUDGET.

Policies compared on every mesh, with models always tested on assets they
never saw in training (GroupKFold by asset):
  fixed-r    : one ratio for every mesh (what a batch script does)
  search     : binary search over the ratio grid (exact, but several decimate +
               measure runs per mesh)
  ml         : predict the ratio, decimate once, measure once; if over budget,
               step up the grid until it passes
Usage: python train_eval.py <dataset.json> [<dataset.json> ...]
"""
import json
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from geometry import FEATURES

BUDGET = 0.0005          # 0.05% of the bounding-box diagonal
RATIOS = None


def load(paths):
    rows = []
    for p in paths:
        rows += json.load(open(p))
    return rows


def label(sweep, budget):
    ok = [r for r, _, p95, _ in sweep if p95 <= budget]
    return min(ok)


def err_at(sweep, r):
    for rr, tris, p95, _ in sweep:
        if abs(rr - r) < 1e-9:
            return p95, tris
    raise KeyError(r)


def binary_search_evals(sweep, budget):
    """Evaluations a binary search over the grid needs (1.0 is free: no change)."""
    grid = [s[0] for s in sweep]
    lo, hi, evals = 0, len(grid) - 1, 0      # hi is known to pass (ratio 1.0)
    while lo < hi:
        mid = (lo + hi) // 2
        evals += 1
        if err_at(sweep, grid[mid])[0] <= budget:
            hi = mid
        else:
            lo = mid + 1
    return grid[hi], evals


def ml_policy(sweep, pred, budget):
    grid = [s[0] for s in sweep]
    i = next((k for k, r in enumerate(grid) if r >= pred - 1e-9), len(grid) - 1)
    evals = 0
    while grid[i] < 1.0:
        evals += 1
        if err_at(sweep, grid[i])[0] <= budget:
            return grid[i], evals
        i += 1
    return 1.0, evals


def main(paths, budget=BUDGET, quiet=False):
    rows = load(paths)
    X = np.array([[r[f] for f in FEATURES] for r in rows])
    y = np.log(np.array([label(r["sweep"], budget) for r in rows]))
    groups = np.array([r["asset"] for r in rows])
    tris = np.array([r["tris"] for r in rows], dtype=float)
    pred = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = GradientBoostingRegressor(loss="quantile", alpha=0.6, n_estimators=300,
                                      max_depth=3, learning_rate=0.05, subsample=0.8,
                                      random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = np.exp(m.predict(X[te]))

    res = {}
    def summarise(name, chosen, evals, passed):
        kept = np.array([err_at(r["sweep"], c)[1] for r, c in zip(rows, chosen)])
        res[name] = {"within_budget": float(np.mean(passed)),
                     "tris_kept": float(kept.sum() / tris.sum()),
                     "evals_per_mesh": float(np.mean(evals))}

    oracle = [label(r["sweep"], budget) for r in rows]
    summarise("oracle", oracle, [0] * len(rows), [True] * len(rows))
    for fr in (0.5, 0.3, 0.2):
        ch = [fr] * len(rows)
        summarise(f"fixed-{fr}", ch, [0] * len(rows),
                  [err_at(r["sweep"], fr)[0] <= budget for r in rows])
    bs = [binary_search_evals(r["sweep"], budget) for r in rows]
    summarise("search", [b[0] for b in bs], [b[1] for b in bs], [True] * len(rows))
    ml = [ml_policy(r["sweep"], p, budget) for r, p in zip(rows, pred)]
    first_try = [err_at(r["sweep"], next(g for g in [s[0] for s in r["sweep"]] if g >= p - 1e-9))[0] <= budget
                 if p < 1 else True for r, p in zip(rows, pred)]
    summarise("ml", [m[0] for m in ml], [m[1] for m in ml], [True] * len(rows))
    res["ml"]["first_try_pass"] = float(np.mean(first_try))
    res["ml"]["mean_abs_log_err"] = float(np.mean(np.abs(np.log(pred) - y)))
    res["n_meshes"] = len(rows)
    res["n_assets"] = int(len(set(groups)))
    res["n_tris"] = int(tris.sum())
    res["budget"] = budget
    if not quiet:
        print(json.dumps(res, indent=1))
    return res, rows, pred


if __name__ == "__main__":
    main(sys.argv[1:])
