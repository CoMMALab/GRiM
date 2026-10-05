"""Float64 references for the IK solver kernels.

Each function transcribes its kernel's algorithm step for step (same weights, scaling,
damping schedule, trust region, line search and accept rule), so a CUDA solve can be
compared to it iteration by iteration. Collision terms are not modelled here; collision
behaviour is checked through solution certificates instead.
"""

from __future__ import annotations

import numpy as np

from ..robot import MotionRobot
from . import kinematics as K

LS_ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.025)


def stacked_residual_jacobian(robot: MotionRobot, q, targets, ee_joints, cols):
    r = np.concatenate([K.pose_residual(K.frame_poses(robot, q)[e], targets[k])
                        for k, e in enumerate(ee_joints)])
    J = np.concatenate([K.pose_jacobian(robot, q, e) for e in ee_joints])[:, cols]
    return r, J


def trust_radius(r: np.ndarray) -> float:
    rp = max(np.linalg.norm(r[6 * k:6 * k + 3]) for k in range(len(r) // 6))
    ro = max(np.linalg.norm(r[6 * k + 3:6 * k + 6]) for k in range(len(r) // 6))
    if rp > 1e-2 or ro > 0.6:
        return 0.38
    if rp > 1e-3 or ro > 0.25:
        return 0.22
    if rp > 2e-4 or ro > 0.08:
        return 0.12
    return 0.05


def ls_ik(robot: MotionRobot, seed, targets, ee_joints, cols, *, lower, upper, fixed_mask,
          max_iter, pos_weight, ori_weight, lambda_init, eps_pos, eps_ori):
    """One seed of ``kernels/ik/ls_ik.cu``; returns (best q, best weighted squared error).

    ``cols`` are the solved actuated indices (the build's ``solved_joints``); the others
    stay at the seed.
    """
    W = np.array([pos_weight] * 3 + [ori_weight] * 3, np.float64)
    q = np.asarray(seed, np.float64).copy()
    lo, hi = np.asarray(lower, np.float64)[cols], np.asarray(upper, np.float64)[cols]
    fixed = np.asarray(fixed_mask)[cols] != 0
    n_ee = len(ee_joints)
    Wr = np.tile(W, n_ee)

    def err(x):
        r, _ = stacked_residual_jacobian(robot, x, targets, ee_joints, cols)
        return float(np.sum((r * Wr) ** 2)), r

    best_q, (best_err, _) = q.copy(), err(q)
    lam = lambda_init
    for _ in range(max_iter):
        r, J = stacked_residual_jacobian(robot, q, targets, ee_joints, cols)
        if all(np.linalg.norm(r[6 * k:6 * k + 3]) < eps_pos and
               np.linalg.norm(r[6 * k + 3:6 * k + 6]) < eps_ori for k in range(n_ee)):
            break
        fw = r * Wr
        J = J * Wr[:, None]
        curr = float(fw @ fw)
        scale = np.sqrt((J ** 2).sum(0)) + 1e-8
        Js = J / scale
        A = Js.T @ Js + lam * np.eye(len(cols))
        rhs = -Js.T @ fw
        A[fixed, :] = 0
        A[:, fixed] = 0
        A[fixed, fixed] = 1
        rhs[fixed] = 0
        try:
            L = np.linalg.cholesky(A)
            delta = np.linalg.solve(L.T, np.linalg.solve(L, rhs)) / scale
        except np.linalg.LinAlgError:
            delta = np.zeros(len(cols))
        R = trust_radius(r)
        n = np.linalg.norm(delta)
        if n > R:
            delta *= R / (n + 1e-18)
        trials = []
        for a in LS_ALPHAS:
            x = q.copy()
            x[cols] = np.clip(q[cols] + a * delta, lo, hi)
            trials.append((err(x)[0], x))
        k = int(np.argmin([t[0] for t in trials]))      # lowest index wins a tie
        e_k, x_k = trials[k]
        if e_k < curr * (1 - 1e-4):
            q = x_k
            lam = max(lam * 0.5, 1e-10)
        else:
            lam = min(lam * 3.0, 1e6)
        if e_k < best_err:
            best_err, best_q = e_k, x_k.copy()
    return best_q, best_err
