"""Float64 references for the trajectory-optimization kernels' costs."""

from __future__ import annotations

import numpy as np

from ..robot import MotionRobot
from . import collision as C
from . import kinematics as K

ACC_STENCIL = np.array([-1, 16, -30, 16, -1]) / 12.0


def accelerations(traj):
    """(T-4, n) five-point second differences."""
    T = len(traj)
    return np.stack([ACC_STENCIL @ traj[t:t + 5] for t in range(T - 4)]) if T >= 5 else \
        np.zeros((0, traj.shape[1]))


def smoothness(traj, w_smooth, w_acc, w_jerk):
    a = accelerations(traj)
    j = a[1:] - a[:-1]
    return w_smooth * (w_acc * np.sum(a ** 2) + w_jerk * np.sum(j ** 2))


def limit_penalty(traj, lower, upper, w_limits):
    v = np.maximum(traj - upper, 0) + np.maximum(lower - traj, 0)
    return w_limits * np.sum(v ** 2)


def row_spheres(robot: MotionRobot, q, tc):
    """World centres (R, S, 3) and radii (R, S) of the trajectory collision model at q."""
    T = K.frame_poses(robot, q)
    frames = np.concatenate([T, [[1, 0, 0, 0, 0, 0, 0]]])     # row n_joints: the world
    off, rad = np.asarray(tc.sphere_off, float), np.asarray(tc.sphere_rad, float)
    centres = np.stack([[K.quat_rotate(frames[r, :4], off[r, s]) + frames[r, 4:]
                         for s in range(off.shape[1])] for r in range(off.shape[0])])
    return centres, rad


def collision_cost(robot, q, tc, world, margin, w):
    """Self pairs and (row, obstacle) pairs: hard min over spheres, then the clearance map."""
    c, rad = row_spheres(robot, q, tc)
    live = rad >= 0
    cost = 0.0
    for i, j in zip(tc.pair_i, tc.pair_j):
        ds = [np.linalg.norm(c[i, a] - c[j, b]) - rad[i, a] - rad[j, b]
              for a in np.nonzero(live[i])[0] for b in np.nonzero(live[j])[0]]
        if ds:
            cost -= min(C.colldist_from_sdf(min(ds), margin), 0.0) * w
    for kind, fn in C.PRIMITIVES.items():
        obstacles = getattr(world, kind)
        if obstacles is None:
            continue
        for o in np.asarray(obstacles, float):
            for r in range(len(rad)):
                ds = [fn(c[r, s], rad[r, s], o) for s in np.nonzero(live[r])[0]]
                if ds:
                    cost -= min(C.colldist_from_sdf(min(ds), margin), 0.0) * w
    return cost


def sco_cost(robot, traj, tc, world, *, lower, upper, w_smooth, w_acc, w_jerk, w_limits,
             w_collision_max, collision_margin):
    """The nonlinear cost ``kernels/trajopt/sco_trajopt.cu`` reports per trajectory."""
    traj = np.asarray(traj, float)
    return (smoothness(traj, w_smooth, w_acc, w_jerk)
            + limit_penalty(traj, lower, upper, w_limits)
            + sum(collision_cost(robot, q, tc, world, collision_margin, w_collision_max)
                  for q in traj))


def hinge_collision_cost(robot, q, tc, world, margin):
    """Sum over self pairs and (row, obstacle) pairs of max(0, margin - d)^2, d the hard
    min over the pair's spheres (STOMP / CHOMP / LS)."""
    c, rad = row_spheres(robot, q, tc)
    live = rad >= 0
    cost = 0.0
    for i, j in zip(tc.pair_i, tc.pair_j):
        ds = [np.linalg.norm(c[i, a] - c[j, b]) - rad[i, a] - rad[j, b]
              for a in np.nonzero(live[i])[0] for b in np.nonzero(live[j])[0]]
        if ds:
            cost += max(margin - min(ds), 0.0) ** 2
    for kind, fn in C.PRIMITIVES.items():
        obstacles = getattr(world, kind)
        if obstacles is None:
            continue
        for o in np.asarray(obstacles, float):
            for r in range(len(rad)):
                ds = [fn(c[r, s], rad[r, s], o) for s in np.nonzero(live[r])[0]]
                if ds:
                    cost += max(margin - min(ds), 0.0) ** 2
    return cost


def stomp_cost(robot, traj, tc, world, *, lower, upper, w_smooth, w_acc, w_jerk, w_limits,
               w_collision_max, collision_margin):
    """The cost ``kernels/trajopt/stomp_trajopt.cu`` reports per trajectory."""
    traj = np.asarray(traj, float)
    return (smoothness(traj, w_smooth, w_acc, w_jerk)
            + limit_penalty(traj, lower, upper, w_limits)
            + w_collision_max * sum(hinge_collision_cost(robot, q, tc, world, collision_margin)
                                    for q in traj))


def smooth_min(ds, tau):
    """-tau log sum exp(-d / tau); +1e10 for an empty group (as the kernels do)."""
    if not ds:
        return 1e10
    v = -np.asarray(ds, float) / tau
    m = v.max()
    return -tau * (np.log(np.exp(v - m).sum()) + m)


def group_distances(robot, q, tc, world, tau):
    """The 5 smooth-min distance groups: self, then world spheres/capsules/boxes/halfspaces;
    each pair contributes the hard min over its spheres."""
    c, rad = row_spheres(robot, q, tc)
    live = rad >= 0
    groups = [[]]
    for i, j in zip(tc.pair_i, tc.pair_j):
        ds = [np.linalg.norm(c[i, a] - c[j, b]) - rad[i, a] - rad[j, b]
              for a in np.nonzero(live[i])[0] for b in np.nonzero(live[j])[0]]
        if ds:
            groups[0].append(min(ds))
    for kind, fn in C.PRIMITIVES.items():
        groups.append([])
        obstacles = getattr(world, kind)
        if obstacles is None:
            continue
        for r in range(len(rad)):
            for o in np.asarray(obstacles, float):
                ds = [fn(c[r, s], rad[r, s], o) for s in np.nonzero(live[r])[0]]
                if ds:
                    groups[-1].append(min(ds))
    return [smooth_min(g, tau) for g in groups]


def ls_cost(robot, traj, tc, world, *, lower, upper, w_smooth, w_acc, w_jerk, w_limits,
            w_collision_max, collision_margin, smooth_min_temperature=0.05):
    """The cost ``kernels/trajopt/ls_trajopt.cu`` reports: 3-point accelerations and jerks,
    limits, and a hinge on each smooth-min distance group."""
    traj = np.asarray(traj, float)
    acc = traj[2:] - 2 * traj[1:-1] + traj[:-2]
    jerk = acc[1:] - acc[:-1]
    coll = sum(max(collision_margin - d, 0.0) ** 2
               for q in traj for d in group_distances(robot, q, tc, world, smooth_min_temperature))
    return (w_smooth * (w_acc * np.sum(acc ** 2) + w_jerk * np.sum(jerk ** 2))
            + limit_penalty(traj, lower, upper, w_limits) + w_collision_max * coll)
