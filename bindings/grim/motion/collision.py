"""Fused FK + collision queries of a robot, built per robot (``kernels/collision/
fused_collision.cu``): self-collision pair distances, per-link distances to world primitives
or to an ESDF voxel grid, each optionally with its Jacobian. One thread per configuration;
FK never leaves the thread.

The robot's collision model is a :class:`~grim.motion._build.SelfCollision` (link-local
spheres in CSR runs per link, the joint frame posing each link, and the self pairs); its
pair table is used by the self query and its spheres by every query. World obstacle COUNTS
are part of the build; their poses (and the ESDF grid) are per call.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from ._build import SelfCollision
from .ik import World
from .robot import MotionRobot

SYMBOLS = ("FusedSelfCollisionFfi", "FusedWorldCollisionFfi", "FusedWorldEsdfFfi",
           "FusedSelfCollisionJacFfi", "FusedWorldCollisionJacFfi", "FusedWorldEsdfJacFfi")


@dataclass(frozen=True)
class Checker:
    """A built collision kernel set; hashable (a valid static argument under jax.jit)."""
    names: tuple[str, ...]
    n_q: int
    n_links: int
    n_pairs: int
    world_counts: tuple[int, int, int, int]


def build(robot: MotionRobot, model: SelfCollision, world_counts=(0, 0, 0, 0),
          traced: bool = False) -> Checker:
    prob = _build.Problem(self_collision=model,
                          world_counts=tuple(int(n) for n in world_counts))
    names = _build.target("collision/fused_collision", SYMBOLS, robot, prob, traced)
    return Checker(names, robot.n_act, len(model.link_start) - 1, len(model.pair_i),
                   prob.world_counts)


def self_distances(c: Checker, q, jacobian: bool = False):
    """Per self pair, the min signed distance over its spheres ``(B, P)``; and the lowest
    sphere point ``min(z - r)`` ``(B,)``. With ``jacobian``: ``(d, dd/dq, z, dz/dq)``."""
    q = jnp.asarray(q, jnp.float32)
    B = q.shape[0]
    f = jax.ShapeDtypeStruct
    if not jacobian:
        return jax.ffi.ffi_call(c.names[0], (f((B, c.n_pairs), jnp.float32),
                                             f((B,), jnp.float32)))(q)
    return jax.ffi.ffi_call(c.names[3], (f((B, c.n_pairs), jnp.float32),
                                         f((B, c.n_pairs, c.n_q), jnp.float32),
                                         f((B,), jnp.float32), f((B, c.n_q), jnp.float32)))(q)


def world_distances(c: Checker, q, world: World, jacobian: bool = False):
    """Per (link, obstacle), the min signed distance over the link's spheres
    ``(B, N_links, M)``, obstacles ordered spheres, capsules, boxes, halfspaces."""
    if world.counts() != c.world_counts:
        raise ValueError(f"world counts {world.counts()} differ from the build {c.world_counts}")
    q = jnp.asarray(q, jnp.float32)
    B, M = q.shape[0], sum(c.world_counts)
    f = jax.ShapeDtypeStruct
    if not jacobian:
        return jax.ffi.ffi_call(c.names[1], f((B, c.n_links, M), jnp.float32))(q, *world.arrays())
    return jax.ffi.ffi_call(c.names[4], (f((B, c.n_links, M), jnp.float32),
                                         f((B, c.n_links, M, c.n_q), jnp.float32)))(
        q, *world.arrays())


def esdf_distances(c: Checker, q, grid, origin, voxel: float, jacobian: bool = False):
    """Per link, the min over its spheres of (trilinear ESDF at the centre - radius)
    ``(B, N_links)`` (+inf for links without spheres). ``grid`` (nx, ny, nz) C order,
    ``origin`` the world position of voxel (0, 0, 0)'s centre, ``voxel`` the edge length."""
    q = jnp.asarray(q, jnp.float32)
    B = q.shape[0]
    f = jax.ShapeDtypeStruct
    args = (q, jnp.asarray(grid, jnp.float32), jnp.asarray(origin, jnp.float32).reshape(3))
    if not jacobian:
        return jax.ffi.ffi_call(c.names[2], f((B, c.n_links), jnp.float32))(
            *args, voxel=np.float32(voxel))
    return jax.ffi.ffi_call(c.names[5], (f((B, c.n_links), jnp.float32),
                                         f((B, c.n_links, c.n_q), jnp.float32)))(
        *args, voxel=np.float32(voxel))
