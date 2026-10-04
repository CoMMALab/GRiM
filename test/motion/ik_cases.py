"""IK problem generation shared by the solver tests (Principle 1 for IK)."""

from __future__ import annotations

import numpy as np

from grim.motion import MotionRobot
from grim.motion.reference import kinematics as K


def reachable_targets(robot: MotionRobot, ee_joints, n: int, seed: int = 0):
    """Targets from FK of random in-limit configurations (so a solution exists), including
    configurations at a joint limit and at the zero pose, where arms are often singular."""
    rng = np.random.default_rng(seed)
    lo, hi = robot.lower, robot.upper
    qs = rng.uniform(lo, hi, size=(n, robot.n_act))
    qs[0] = 0.0
    if n > 1:
        qs[1] = np.where(rng.random(robot.n_act) < 0.5, lo, hi)
    T = np.stack([K.frame_poses(robot, q)[list(ee_joints)] for q in qs])
    return T, qs


def seeds(robot: MotionRobot, n_problems: int, n_seeds: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    s = rng.uniform(robot.lower, robot.upper, size=(n_problems, n_seeds, robot.n_act))
    s[:, 0] = 0.5 * (robot.lower + robot.upper)
    return s


def pose_errors(robot: MotionRobot, q, targets, ee_joints):
    """(max position error [m], max rotation error [rad]) over end-effectors, per config."""
    T = K.frame_poses(robot, q)
    r = np.stack([K.pose_residual(T[e], targets[k]) for k, e in enumerate(ee_joints)])
    return np.linalg.norm(r[:, :3], axis=1).max(), np.linalg.norm(r[:, 3:], axis=1).max()
