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


def collision_case(robot: MotionRobot, n_per_joint: int = 2, seed: int = 0):
    """A synthetic collision setup touching every collision code path: ``n_per_joint`` spheres
    in each moving joint frame (world collision), the same spheres grouped per joint as
    self-collision links with every pair of links at least two joints apart, and a world of
    each obstacle kind placed around the arm's reach. Returns ``(CollisionModel, World)``."""
    from grim.motion._build import SelfCollision
    from grim.motion.ik import CollisionModel, World
    rng = np.random.default_rng(seed)
    joints = [j for j in range(robot.n_joints) if robot.act_idx[j] != -1]
    sph, sj = [], []
    for j in joints:
        for _ in range(n_per_joint):
            sph.append(np.r_[rng.normal(0.0, 0.04, 3), rng.uniform(0.03, 0.06)])
            sj.append(j)
    sph, sj = np.array(sph), np.array(sj, np.int32)
    depth = {j: len(robot.chain(j)) for j in joints}
    pi, pj = [], []
    for a in range(len(joints)):
        for b in range(a + 1, len(joints)):
            ja, jb = joints[a], joints[b]
            if ja not in robot.chain(jb) and jb not in robot.chain(ja) or abs(depth[ja] - depth[jb]) >= 2:
                pi.append(a)
                pj.append(b)
    sc = SelfCollision(sph, np.arange(0, len(sph) + 1, n_per_joint, dtype=np.int32),
                       np.array(joints, np.int32), np.array(pi, np.int32), np.array(pj, np.int32))
    T = K.frame_poses(robot, 0.5 * (robot.lower + robot.upper))
    reach = T[joints, 4:]
    c = reach.mean(0)
    span = max(float(np.linalg.norm(reach - c, axis=1).max()), 0.2)
    world = World(spheres=np.array([[*(c + span * rng.normal(0, 0.5, 3)), 0.1],
                                    [*(c + span * rng.normal(0, 0.5, 3)), 0.08]]),
                  capsules=np.array([[*(c + [0.3 * span, 0, 0]), *(c + [0.3 * span, 0.4 * span, 0.2]), 0.05]]),
                  boxes=np.array([[*(c + [0, -0.4 * span, 0]), 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.1, 0.05, 0.2]]),
                  halfspaces=np.array([[0, 0, 1, *(reach[:, :2].mean(0)), float(reach[:, 2].min()) - 0.05]]))
    return CollisionModel(sph, sj, sc), world
