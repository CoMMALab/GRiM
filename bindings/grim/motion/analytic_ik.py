"""Closed-form IK for the 7-DOF "spherical shoulder, intersecting axes 5-6, offset axis 7"
family (Franka Panda / FR3), built per robot (``kernels/ik/analytic_ik.cu``).

The kernel sweeps the redundancy parameter q7 and, for each value, solves the remaining six
joints with Paden-Kahan subproblems (8 branches). :func:`arm_geometry` reads the home-pose
geometry the kernel needs off the robot's own kinematics and checks the structural
preconditions; it is baked into the build.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from .reference import kinematics as K
from .robot import MotionRobot

CONCURRENT_TOL = 1e-6     # max distance [m] from the common point to any axis
GEOM_N_SCALARS = 95


@dataclass(frozen=True)
class ArmGeometry:
    axes: np.ndarray        # (7, 3) unit world axes at q = 0
    points: np.ndarray      # (7, 3) a point on each axis
    shoulder: np.ndarray    # (3,) where axes 1-2-3 meet
    wrist: np.ndarray       # (3,) where axes 5-6 meet
    m_home: np.ndarray      # (4, 4) end-effector pose at q = 0
    lower: np.ndarray       # (7,)
    upper: np.ndarray       # (7,)
    act: np.ndarray         # (7,) the chain's actuated indices (q order)

    @property
    def cos_alpha(self) -> float:
        return float(self.axes[4] @ self.axes[5])


def _common_point(points, dirs):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, d in zip(points, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ p
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    res = max(np.linalg.norm((x - p) - d * (d @ (x - p))) for p, d in zip(points, dirs))
    return x, res


def arm_geometry(robot: MotionRobot, ee_joint: int) -> ArmGeometry:
    """Home-pose geometry of the chain ending at joint frame ``ee_joint``."""
    T = K.frame_poses(robot, np.zeros(robot.n_act))
    chain = [j for j in reversed(robot.chain(ee_joint))
             if robot.act_idx[j] != -1 and np.linalg.norm(robot.twists[j, 3:]) > 0]
    if len(chain) != 7:
        raise ValueError(f"analytic IK needs a 7-revolute chain; the chain to "
                         f"{robot.joint_names[ee_joint]!r} has {len(chain)}")
    dirs = np.stack([K.quat_rotate(T[j, :4], robot.twists[j, 3:] /
                                   np.linalg.norm(robot.twists[j, 3:])) for j in chain])
    pts = np.stack([T[j, 4:] for j in chain])
    shoulder, r_s = _common_point(pts[0:3], dirs[0:3])
    wrist, r_w = _common_point(pts[4:6], dirs[4:6])
    _, r_w3 = _common_point(pts[4:7], dirs[4:7])
    if r_s > CONCURRENT_TOL or r_w > CONCURRENT_TOL or r_w3 <= CONCURRENT_TOL:
        raise ValueError("analytic IK: the chain is not in the spherical-shoulder / "
                         "intersecting-5-6 / offset-7 family (shoulder residual "
                         f"{r_s:.1e} m, wrist 5-6 {r_w:.1e} m, wrist 5-6-7 {r_w3:.1e} m)")
    w, x, y, z = T[ee_joint, :4]
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    M[:3, 3] = T[ee_joint, 4:]
    act = np.array([robot.act_idx[j] for j in chain])
    return ArmGeometry(dirs, pts, shoulder, wrist, M, robot.lower[act], robot.upper[act], act)


def pack_geometry(g: ArmGeometry) -> np.ndarray:
    """Flatten to the kernel's ``struct ArmGeom`` field order."""
    blob = np.concatenate([g.axes.ravel(), g.points.ravel(), g.shoulder, g.wrist,
                           g.m_home.ravel(), np.linalg.inv(g.m_home).ravel(), [g.cos_alpha],
                           g.lower, g.upper]).astype(np.float64)
    assert blob.size == GEOM_N_SCALARS, blob.size
    return blob


def _geometry_header(g: ArmGeometry) -> str:
    vals = ", ".join(f"{v:.17e}" for v in pack_geometry(g))
    return (f"namespace grim::robot {{\n"
            f"__device__ __constant__ double kArmGeom[{GEOM_N_SCALARS}] = {{{vals}}};\n}}\n")


def analytic_ik(robot: MotionRobot, targets, ee_joint: int, q7_samples=None, *,
                previous_q=None, spheres_home=None, sphere_joint=None, self_pairs=None,
                world=None, respect_limits: bool = True, err_tol: float = 1e-4,
                margin: float = 0.005):
    """Best branch per target -> ``(q (B, 7), err (B,), found (B,), clearance (B,))``.

    ``targets`` (B, 4, 4) end-effector poses. ``q7_samples`` (S,) redundancy values (default
    32 across joint 7's range). With ``previous_q`` (B, 7) the valid branch closest to it is
    returned instead of the lowest-error one. ``q`` is in the chain's joint order
    (``arm_geometry(...).act`` gives the actuated indices). Collision: ``spheres_home``
    (N, 4) spheres at q = 0 in the world frame, ``sphere_joint`` (N,) the chain joint
    (0-6, or -1 for the base) moving each, ``self_pairs`` (P, 2), ``world`` a
    :class:`grim.motion.ik.World`.
    """
    from .ik import World
    g = arm_geometry(robot, ee_joint)
    header = _build.robot_header(robot, _build.Problem(ee_joints=(int(ee_joint),)))
    so = _build.build("ik/analytic_ik", header + _geometry_header(g), 1, robot.n_joints, 1)
    (name,) = _build.register(so, ("AnalyticIkFfi",))
    targets = jnp.asarray(targets, jnp.float32).reshape(-1, 4, 4)
    B = targets.shape[0]
    q7 = jnp.linspace(g.lower[6], g.upper[6], 32) if q7_samples is None else q7_samples
    prev = jnp.zeros((B, 7)) if previous_q is None else previous_q
    world = world or World()
    return jax.ffi.ffi_call(name, (jax.ShapeDtypeStruct((B, 7), jnp.float32),
                                   jax.ShapeDtypeStruct((B,), jnp.float32),
                                   jax.ShapeDtypeStruct((B,), jnp.int32),
                                   jax.ShapeDtypeStruct((B,), jnp.float32)),
                            vmap_method="sequential")(
        targets, jnp.asarray(q7, jnp.float32).reshape(-1),
        jnp.asarray(prev, jnp.float32).reshape(B, 7),
        jnp.asarray(np.zeros((0, 4)) if spheres_home is None else spheres_home, jnp.float32),
        jnp.asarray(np.zeros(0) if sphere_joint is None else sphere_joint, jnp.int32),
        jnp.asarray(np.zeros((0, 2)) if self_pairs is None else self_pairs, jnp.int32).reshape(-1, 2),
        *world.arrays(), respect_limits=np.int64(respect_limits),
        use_prev=np.int64(previous_q is not None), err_tol=np.float32(err_tol),
        margin=np.float32(margin))
