"""Pure NumPy/SciPy geometry: surface error between two meshes and mesh features."""
import numpy as np


def _areas(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    return 0.5 * np.linalg.norm(n, axis=1), n


def sample_surface(v, t, n, rng):
    area, _ = _areas(v, t)
    if area.sum() <= 0:
        return v[rng.integers(0, len(v), n)]
    idx = rng.choice(len(t), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = v[t[idx, 0]], v[t[idx, 1]], v[t[idx, 2]]
    return (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c


def diag(v):
    return float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0


def surface_error(v0, t0, v1, t1, n=8000, seed=0):
    """Two-sided point-to-surface distance, as a fraction of the original's
    bounding-box diagonal (exact distance to the nearest triangle, via trimesh).

    Returns (p95, max) over points sampled on both surfaces.
    """
    import trimesh
    rng = np.random.default_rng(seed)
    if len(t1) == 0:
        return 1.0, 1.0
    # work in unit scale: trimesh tolerances misbehave on centimetre-sized parts
    c, s = v0.mean(0), diag(v0)
    v0, v1 = (v0 - c) / s, (v1 - c) / s
    m0 = trimesh.Trimesh(v0, t0, process=False)
    m1 = trimesh.Trimesh(v1, t1, process=False)
    p0 = sample_surface(v0, t0, n, rng)
    p1 = sample_surface(v1, t1, n, rng)
    _, d01, _ = trimesh.proximity.closest_point(m1, p0)
    _, d10, _ = trimesh.proximity.closest_point(m0, p1)
    d = np.concatenate([d01, d10])
    return float(np.percentile(d, 95)), float(d.max())


FEATURES = ["log_tris", "log_verts", "area_norm", "aspect_mid", "aspect_min",
            "sharp_frac", "mean_dihedral", "boundary_frac", "area_cv",
            "valence_std", "planarity", "tri_density"]


def mesh_features(v, t):
    """Shape descriptors that tell how much a mesh can be simplified."""
    area, nrm = _areas(v, t)
    keep = area > 1e-14
    t, area, nrm = t[keep], area[keep], nrm[keep]
    fn = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    d = diag(v)
    ext = np.sort(v.max(0) - v.min(0))[::-1] + 1e-12
    # edge -> faces adjacency
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    f_of_e = np.tile(np.arange(len(t)), 3)
    key = e[:, 0] * (len(v) + 1) + e[:, 1]
    order = np.argsort(key)
    key, f_of_e = key[order], f_of_e[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    pair = counts == 2
    fa, fb = f_of_e[start[pair]], f_of_e[start[pair] + 1]
    cosang = np.clip((fn[fa] * fn[fb]).sum(1), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    valence = np.bincount(t.ravel(), minlength=len(v))
    valence = valence[valence > 0]
    centered = v - v.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(centered.T) + 1e-18 * np.eye(3)))
    return {
        "log_tris": np.log10(len(t) + 1),
        "log_verts": np.log10(len(v) + 1),
        "area_norm": float(area.sum() / d ** 2),
        "aspect_mid": float(ext[1] / ext[0]),
        "aspect_min": float(ext[2] / ext[0]),
        "sharp_frac": float((ang > 30).mean()) if len(ang) else 0.0,
        "mean_dihedral": float(ang.mean()) if len(ang) else 0.0,
        "boundary_frac": float((counts == 1).mean()),
        "area_cv": float(area.std() / (area.mean() + 1e-18)),
        "valence_std": float(valence.std()),
        "planarity": float(ev[0] / (ev[2] + 1e-18)),
        "tri_density": float(np.log10(len(t) / (area.sum() / d ** 2 + 1e-12) + 1)),
    }
