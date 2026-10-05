"""Trajectory problems shared by the trajopt tests."""

from __future__ import annotations

import numpy as np

from grim.motion import MotionRobot
from grim.motion._build import TrajCollision
from grim.motion.ik import World
from grim.motion.reference import kinematics as K


def sphere_model(robot: MotionRobot, per_row: int = 2, radius: float = 0.05) -> TrajCollision:
    """A simple collision model: spheres along each moving link (between consecutive joint
    origins), self pairs between non-adjacent rows."""
    T = K.frame_poses(robot, np.zeros(robot.n_act))
    R = robot.n_joints + 1
    off = np.zeros((R, per_row, 3))
    rad = -np.ones((R, per_row))
    for j in range(robot.n_joints):
        kids = [c for c in range(robot.n_joints) if robot.parent_idx[c] == j]
        if not kids:
            continue
        # Child origin in joint j's frame.
        w, x, y, z = T[j, :4]
        Rj = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        tip = Rj.T @ (T[kids[0], 4:] - T[j, 4:])
        for s in range(per_row):
            off[j, s] = tip * (s + 0.5) / per_row
            rad[j, s] = radius
    rows = [r for r in range(R) if (rad[r] > 0).any()]
    adjacent = {(int(robot.parent_idx[j]), j) for j in range(robot.n_joints)}
    pairs = [(a, b) for i, a in enumerate(rows) for b in rows[i + 2:]
             if (a, b) not in adjacent and (b, a) not in adjacent]
    pi, pj = (np.array([p[k] for p in pairs], np.int32) for k in (0, 1))
    return TrajCollision(off, rad, pi, pj)


def straight_lines(robot: MotionRobot, B: int, T: int, seed: int = 0, noise: float = 0.05):
    """B noisy straight-line trajectories between two in-limit configurations."""
    rng = np.random.default_rng(seed)
    mid = 0.5 * (robot.lower + robot.upper)
    span = 0.25 * (robot.upper - robot.lower)
    start, goal = mid - span, mid + span
    s = np.linspace(0, 1, T)[:, None]
    base = start + s * (goal - start)
    x = base[None] + rng.normal(scale=noise, size=(B, T, robot.n_act))
    x[:, 0], x[:, -1] = start, goal
    return np.clip(x, robot.lower, robot.upper), start, goal


def obstacle_world(robot: MotionRobot, start, goal) -> World:
    """A sphere near the end-effector's midpoint, plus a floor."""
    leaf = max(range(robot.n_joints), key=lambda j: len(robot.chain(j)))
    p = K.frame_poses(robot, 0.5 * (start + goal))[leaf, 4:]
    return World(spheres=np.array([[p[0] + 0.1, p[1], p[2], 0.08]]),
                 halfspaces=np.array([[0, 0, 1, 0, 0, -0.05]]))
