"""Float64 references for the collision distance primitives in kernels/collision.cuh."""

from __future__ import annotations

import numpy as np


def sphere_sphere(c, r, o):
    return np.linalg.norm(c - o[:3]) - (r + o[3])


def sphere_capsule(c, r, cap):
    a, b, cr = cap[:3], cap[3:6], cap[6]
    v = b - a
    t = 0.0 if v @ v <= 1e-12 else np.clip((c - a) @ v / (v @ v), 0.0, 1.0)
    return np.linalg.norm(c - (a + t * v)) - (r + cr)


def box_sdf_local(p, hl):
    q = np.abs(p) - hl
    return np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0)


def sphere_box(c, r, box):
    d = c - box[:3]
    axes = box[3:12].reshape(3, 3)
    return box_sdf_local(axes @ d, box[12:15]) - r


def sphere_halfspace(c, r, hs):
    return (c - hs[3:6]) @ hs[:3] - r


PRIMITIVES = {"spheres": sphere_sphere, "capsules": sphere_capsule, "boxes": sphere_box,
              "halfspaces": sphere_halfspace}


def colldist_from_sdf(d, margin):
    """The smooth signed clearance the trajopt costs penalize (<= 0; 0 beyond the margin)."""
    d = min(d, margin)
    val = d - 0.5 * margin if d < 0 else -0.5 / (margin + 1e-6) * (d - margin) ** 2
    return min(val, 0.0)


def link_spheres(robot, q, model):
    """World centres and radii of a SelfCollision model's spheres, per link: list of
    (centres (k, 3), radii (k,))."""
    from . import kinematics as K
    T = K.frame_poses(robot, q)
    out = []
    for n in range(len(model.link_start) - 1):
        j = int(model.link_joint[n])
        frame = T[j] if j >= 0 else np.array([1, 0, 0, 0, 0, 0, 0.0])
        sph = np.asarray(model.sph[model.link_start[n]:model.link_start[n + 1]], float)
        c = np.array([K.quat_rotate(frame[:4], s[:3]) + frame[4:] for s in sph]).reshape(-1, 3)
        out.append((c, sph[:, 3]))
    return out


def self_pair_distances(robot, q, model):
    ls = link_spheres(robot, q, model)
    d = []
    for i, j in zip(model.pair_i, model.pair_j):
        (ci, ri), (cj, rj) = ls[i], ls[j]
        d.append(min((np.linalg.norm(a - b) - x - y for a, x in zip(ci, ri) for b, y in zip(cj, rj)),
                     default=np.inf))
    return np.array(d)


def world_link_distances(robot, q, model, world):
    ls = link_spheres(robot, q, model)
    obs = [(fn, o) for kind, fn in PRIMITIVES.items()
           for o in ([] if getattr(world, kind) is None else np.asarray(getattr(world, kind), float))]
    return np.array([[min((fn(c, r, o) for c, r in zip(*ls[n])), default=np.inf)
                      for fn, o in obs] for n in range(len(ls))])


def esdf_trilinear(grid, origin, voxel, p):
    """Edge-clamped trilinear sample of a C-order (nx, ny, nz) grid."""
    dims = np.array(grid.shape)
    idx = np.clip((np.asarray(p) - origin) / voxel, 0, dims - 1 - 1e-6)
    i0 = np.floor(idx).astype(int)
    f = idx - i0
    i1 = np.minimum(i0 + 1, dims - 1)
    val = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) *
                     (f[2] if dz else 1 - f[2]))
                val += w * grid[(i1 if dx else i0)[0], (i1 if dy else i0)[1], (i1 if dz else i0)[2]]
    return val


def esdf_link_distances(robot, q, model, grid, origin, voxel):
    return np.array([min((esdf_trilinear(grid, origin, voxel, c) - r for c, r in zip(cs, rs)),
                         default=np.inf) for cs, rs in link_spheres(robot, q, model)])
