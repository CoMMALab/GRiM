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
        traced: use mantis's straight-line FK instead of the baked tables.
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


def sphere_kinematics(robot: MotionRobot, q, sphere_joint, sphere_centres, *,
                      traced: bool = False, threads: int = 128):
    """World sphere centres (B, S, 3), fused into one FK kernel.

    Static geometry is in joint-frame coordinates; joint -1 denotes the world.
    Output preserves sphere order, including duplicates. Both backends specialize
    to the requested spheres and their ancestors. Threads controls configurations
    per block, not work per configuration.
    """
    from . import _sphere_fk
    joints = np.asarray(sphere_joint)
    centres = np.asarray(sphere_centres, np.float32)
    if joints.ndim != 1 or centres.shape != (len(joints), 3):
        raise ValueError("expected sphere_joint (S,) and sphere_centres (S, 3)")
    if not np.issubdtype(joints.dtype, np.integer) or np.any((joints < -1) | (joints >= robot.n_joints)):
        raise ValueError("sphere joints must be integer indices in [-1, n_joints)")
    if not 1 <= threads <= 1024:
        raise ValueError("threads must be in [1, 1024]")
    q = jnp.asarray(q, jnp.float32)
    if q.ndim != 2 or q.shape[1] != robot.n_act:
        raise ValueError("q must have shape (batch, n_act)")
    if not len(joints):
        return jnp.empty((q.shape[0], 0, 3), jnp.float32)
    name = _sphere_fk.target(robot, joints, centres, traced)
    return jax.ffi.ffi_call(name, jax.ShapeDtypeStruct(
        (q.shape[0], len(joints), 3), jnp.float32))(q, threads=np.int32(threads))
