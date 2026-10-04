"""Fused FK + collision kernels vs the oracle: self pair distances, the lowest sphere point,
per-link distances to world primitives and to an ESDF grid, and every Jacobian against
central differences of the float64 oracle.

Every output is a min over spheres, which is not differentiable where two spheres tie; the
Jacobians are therefore checked at random configurations (axis-aligned ones -- the zero pose,
single joints at a limit -- put collinear link spheres at equal height and create ties)."""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion._build import SelfCollision
from grim.motion.ik import World
from grim.motion.reference import collision as R

from .conftest import assert_close_scaled, config_samples, load_robot, requires_gpu
from .traj_cases import sphere_model

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

ROBOTS = ["fr3", "iiwa14", "g1"]


def self_model(robot) -> SelfCollision:
    """The trajopt sphere rows as a CSR link model (a link per row that has spheres)."""
    tc = sphere_model(robot)
    rows = [r for r in range(len(tc.sphere_rad)) if (tc.sphere_rad[r] > 0).any()]
    sph, start = [], [0]
    for r in rows:
        live = tc.sphere_rad[r] > 0
        sph += [np.r_[o, rad] for o, rad in zip(tc.sphere_off[r][live], tc.sphere_rad[r][live])]
        start.append(len(sph))
    link = {r: i for i, r in enumerate(rows)}
    joint = [r if r < robot.n_joints else -1 for r in rows]
    pi = np.array([link[a] for a in tc.pair_i], np.int32)
    pj = np.array([link[b] for b in tc.pair_j], np.int32)
    return SelfCollision(np.array(sph), np.array(start, np.int32), np.array(joint, np.int32), pi, pj)


WORLD = World(spheres=np.array([[0.3, 0.1, 0.5, 0.1], [-0.2, 0.3, 0.8, 0.05]]),
              capsules=np.array([[0.2, -0.3, 0.2, 0.4, -0.3, 0.9, 0.04]]),
              boxes=np.array([[0.5, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.1, 0.2, 0.05]]),
              halfspaces=np.array([[0, 0, 1, 0, 0, -0.02]]))


def _fd(fn, q, eps=1e-6):
    cols = []
    for a in range(len(q)):
        dq = np.zeros_like(q)
        dq[a] = eps
        cols.append((fn(q + dq) - fn(q - dq)) / (2 * eps))
    return np.stack(cols, -1)


@pytest.mark.parametrize("name", ROBOTS)
def test_self_and_world(name):
    from grim.motion import collision
    robot = load_robot(name)
    model = self_model(robot)
    c = collision.build(robot, model, WORLD.counts())
    q = config_samples(robot, n_random=8)
    d, z = (np.asarray(x) for x in collision.self_distances(c, q))
    w = np.asarray(collision.world_distances(c, q, WORLD))
    for b in range(len(q)):
        assert_close_scaled(d[b], R.self_pair_distances(robot, q[b], model), 1e-5, f"{name} self")
        lowest = min((cs[:, 2] - rs).min() for cs, rs in R.link_spheres(robot, q[b], model))
        assert abs(z[b] - lowest) < 1e-5, f"{name} min_z"
        assert_close_scaled(w[b], R.world_link_distances(robot, q[b], model, WORLD), 1e-5,
                            f"{name} world")


@pytest.mark.parametrize("name", ROBOTS[:2])
def test_jacobians(name):
    from grim.motion import collision
    robot = load_robot(name)
    model = self_model(robot)
    c = collision.build(robot, model, WORLD.counts())
    q = config_samples(robot, n_random=4, seed=5)[-4:]
    _, dj, _, zj = (np.asarray(x) for x in collision.self_distances(c, q, jacobian=True))
    _, wj = (np.asarray(x) for x in collision.world_distances(c, q, WORLD, jacobian=True))
    for b in range(len(q)):
        fd_self = _fd(lambda x: R.self_pair_distances(robot, x, model), q[b])
        assert_close_scaled(dj[b], fd_self, 1e-3, f"{name} self jacobian b={b}")
        fd_world = _fd(lambda x: R.world_link_distances(robot, x, model, WORLD), q[b])
        assert_close_scaled(wj[b], fd_world, 1e-3, f"{name} world jacobian b={b}")
        fd_z = _fd(lambda x: np.array(min((cs[:, 2] - rs).min()
                                          for cs, rs in R.link_spheres(robot, x, model))), q[b])
        assert_close_scaled(zj[b], fd_z, 1e-3, f"{name} min_z jacobian b={b}")


@pytest.mark.parametrize("name", ROBOTS[:2])
def test_esdf(name):
    from grim.motion import collision
    robot = load_robot(name)
    model = self_model(robot)
    c = collision.build(robot, model)
    origin, voxel, n = np.array([-1.0, -1.0, -0.5]), 0.05, 40
    axes = [origin[k] + voxel * np.arange(n) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    grid = np.minimum(np.sqrt((X - 0.3) ** 2 + (Y - 0.1) ** 2 + (Z - 0.5) ** 2) - 0.15, Z + 0.02)
    q = config_samples(robot, n_random=6, seed=5)[-6:]
    out = np.asarray(collision.esdf_distances(c, q, grid, origin, voxel))
    for b in range(len(q)):
        assert_close_scaled(out[b], R.esdf_link_distances(robot, q[b], model, grid, origin, voxel),
                            1e-5, f"{name} esdf")
    _, jac = (np.asarray(x) for x in collision.esdf_distances(c, q, grid, origin, voxel, jacobian=True))
    fd = _fd(lambda x: R.esdf_link_distances(robot, x, model, grid, origin, voxel), q[0], eps=1e-5)
    assert_close_scaled(jac[0], fd, 2e-2, f"{name} esdf jacobian")
