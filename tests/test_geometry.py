"""Sanity tests for the error metric and features (no Blender needed)."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import geometry as G  # noqa: E402


def sphere(sub, scale=1.0):
    m = trimesh.creation.icosphere(subdivisions=sub)
    return m.vertices * scale, m.faces


def test_self_error_is_zero():
    v, t = sphere(4)
    assert G.surface_error(v, t, v, t)[0] < 1e-9


def test_error_is_scale_free():
    # a 1 cm part and a 10 m part must score the same
    v0, t0 = sphere(5, 0.01)
    v1, t1 = sphere(2, 0.01)
    small = G.surface_error(v0, t0, v1, t1)[0]
    v0, t0 = sphere(5, 10.0)
    v1, t1 = sphere(2, 10.0)
    big = G.surface_error(v0, t0, v1, t1)[0]
    assert abs(small - big) < 0.1 * big


def test_coarser_mesh_has_more_error():
    v0, t0 = sphere(5)
    errs = [G.surface_error(v0, t0, *sphere(s))[0] for s in (4, 3, 2, 1)]
    assert all(a < b for a, b in zip(errs, errs[1:]))


def test_features_are_finite():
    v, t = sphere(3)
    f = G.mesh_features(v, t)
    assert set(f) == set(G.FEATURES)
    assert all(np.isfinite(x) for x in f.values())
