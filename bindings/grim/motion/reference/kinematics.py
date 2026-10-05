"""Float64 numpy reference for the motion kernels' kinematics.

Layer 1 of GRiM's motion verification: these functions are checked against Pinocchio
(``test/motion/test_reference_kinematics.py``) and the CUDA kernels are checked against
these. They reproduce the kernels' *definitions*, not just their intent:

* :func:`frame_poses` -- every joint frame, ``[qw, qx, qy, qz, x, y, z]``;
* :func:`pose_residual` -- ``[p - p*, log(q q*^-1)]`` with the quaternion sign fixed to
  ``w >= 0``;
* :func:`pose_jacobian` -- the world-frame GEOMETRIC Jacobian at the end-effector point
  (linear rows, then angular). This is what every IK solver differentiates with; it is the
  exact derivative of the position rows but only the small-error derivative of the
  orientation rows, by design.
"""

from __future__ import annotations

import numpy as np

from ..robot import MotionRobot


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_rotate(q, v):
    w, u = q[0], np.asarray(q[1:])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


def se3_compose(a, b):
    return np.r_[quat_mul(a[:4], b[:4]), quat_rotate(a[:4], b[4:]) + a[4:]]


def se3_exp(tangent):
    v, w = tangent[:3], tangent[3:]
    th2 = w @ w
    if th2 < 1e-12:
        return np.r_[1.0, 0.0, 0.0, 0.0, v]
    th = np.sqrt(th2)
    q = np.r_[np.cos(th / 2), np.sin(th / 2) / th * w]
    A, B, C = np.sin(th) / th, (1 - np.cos(th)) / th2, (th - np.sin(th)) / (th2 * th)
    t = A * v + B * np.cross(w, v) + C * (w * (w @ v) - th2 * v)
    return np.r_[q, t]


def full_q(robot: MotionRobot, q: np.ndarray) -> np.ndarray:
    """Every joint's value (fixed joints 0, mimic joints mul * q_src + off)."""
    out = np.zeros(robot.n_joints)
    for j in range(robot.n_joints):
        src = robot.mimic_act_idx[j] if robot.mimic_act_idx[j] != -1 else robot.act_idx[j]
        out[j] = (0.0 if src == -1 else q[src]) * robot.mimic_mul[j] + robot.mimic_off[j]
    return out


def frame_poses(robot: MotionRobot, q: np.ndarray) -> np.ndarray:
    """(n_joints, 7) world pose of every joint frame."""
    qj = full_q(robot, np.asarray(q, np.float64))
    T = np.zeros((robot.n_joints, 7))
    for j in robot.topo_inv:
        T_pc = se3_compose(robot.parent_tf[j].astype(np.float64),
                           se3_exp(robot.twists[j].astype(np.float64) * qj[j]))
        p = robot.parent_idx[j]
        T[j] = T_pc if p == -1 else se3_compose(T[p], T_pc)
    return T


def pose_residual(pose, target):
    pose, target = np.asarray(pose, np.float64), np.asarray(target, np.float64)
    q_err = quat_mul(pose[:4], target[:4] * np.array([1, -1, -1, -1]))
    if q_err[0] < 0:
        q_err = -q_err
    s = np.linalg.norm(q_err[1:])
    rot = q_err[1:] * (2 * np.arctan2(s, q_err[0]) / s) if s > 1e-6 else 2 * q_err[1:]
    return np.r_[pose[4:] - target[4:], rot]


def pose_jacobian(robot: MotionRobot, q: np.ndarray, ee_joint: int) -> np.ndarray:
    """(6, n_act) geometric Jacobian of joint frame ``ee_joint`` (mimic joints folded in)."""
    T = frame_poses(robot, q)
    p_ee = T[ee_joint, 4:]
    J = np.zeros((6, robot.n_act))
    for j in robot.chain(ee_joint):
        a = robot.act_idx[j] if robot.act_idx[j] != -1 else robot.mimic_act_idx[j]
        tw = robot.twists[j].astype(np.float64)
        if a == -1 or not tw.any():
            continue
        if tw[3:] @ tw[3:] > 1e-12:
            z = quat_rotate(T[j, :4], tw[3:] / np.linalg.norm(tw[3:]))
            col = np.r_[np.cross(z, p_ee - T[j, 4:]), z]
        else:
            col = np.r_[quat_rotate(T[j, :4], tw[:3] / np.linalg.norm(tw[:3])), np.zeros(3)]
        J[:, a] += robot.mimic_mul[j] * col
    return J
