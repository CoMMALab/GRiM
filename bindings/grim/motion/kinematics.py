"""Batched kinematics of a :class:`MotionRobot` on the GPU (``kernels/kinematics.cu``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from .robot import MotionRobot


def kinematics(robot: MotionRobot, q, targets=None, ee_joints: tuple[int, ...] = (),
               *, chain_only: bool = False, traced: bool = False,
               runtime_rot_joint: int = -1, rot=None):
    """Frame poses, pose residuals and geometric Jacobians at configurations ``q``.

    Args:
        q: (B, n_q) actuated configurations (``robot.actuated_names`` order).
        targets: (B, n_ee, 7) target poses ``[qw, qx, qy, qz, x, y, z]`` (identity if None).
        ee_joints: joint frames whose residual/Jacobian to return.
        chain_only: Jacobian columns over the end-effector chains only (see ``Problem``).
        traced: use cricket's straight-line FK instead of the baked tables.
        runtime_rot_joint, rot: that joint's parent rotation (wxyz) supplied per call.

    Returns:
        ``(T, r, J)``: (B, n_joints, 7), (B, 6 n_ee), (B, 6 n_ee, n_solved).
    """
    problem = _build.Problem(ee_joints=tuple(int(e) for e in ee_joints), chain_only=chain_only,
                             runtime_rot_joint=int(runtime_rot_joint))
    if (runtime_rot_joint >= 0) != (rot is not None):
        raise ValueError("pass rot exactly when runtime_rot_joint >= 0")
    lead = (jnp.asarray(rot, jnp.float32).reshape(4),) if rot is not None else ()
    n_solved = len(_build.solved_joints(robot, problem))
    (name,) = _build.target("kinematics", ("KinematicsFfi",), robot, problem, traced)
    q = jnp.asarray(q, jnp.float32)
    B, n_ee = q.shape[0], len(problem.ee_joints)
    if targets is None:
        targets = jnp.broadcast_to(jnp.array([1, 0, 0, 0, 0, 0, 0], jnp.float32), (B, n_ee, 7))
    return jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, robot.n_joints, 7), jnp.float32),
        jax.ShapeDtypeStruct((B, 6 * n_ee), jnp.float32),
        jax.ShapeDtypeStruct((B, 6 * n_ee, n_solved), jnp.float32),
    ))(*lead, q, jnp.asarray(targets, jnp.float32))


def solved_columns(robot: MotionRobot, ee_joints, chain_only: bool) -> np.ndarray:
    """Actuated indices of the Jacobian columns :func:`kinematics` returns."""
    return np.array(_build.solved_joints(
        robot, _build.Problem(ee_joints=tuple(ee_joints), chain_only=chain_only)), int)
