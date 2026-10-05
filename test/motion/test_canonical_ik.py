"""Canonical IK: certified by the KKT conditions of argmin 1/2|q - q_ref|^2 s.t. r(q) = 0,
evaluated by the oracle -- the result still reaches the target, and q* - q_ref has no
component in the Jacobian's null space (the canonical point is unique there)."""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import kinematics as K

from .conftest import ee_sets, load_robot, requires_gpu
from .ik_cases import reachable_targets

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]


@pytest.mark.parametrize("name", ["iiwa14", "fr3"])
def test_canonical_ik_kkt(name):
    from grim.motion.ik import canonical_ik
    robot = load_robot(name)
    ees = ee_sets(robot)[0]
    targets, q = reachable_targets(robot, ees, 16)
    rng = np.random.default_rng(3)
    q_ref = np.clip(q + rng.normal(scale=0.3, size=q.shape), robot.lower, robot.upper)
    q_star, iters = (np.asarray(x) for x in canonical_ik(robot, q, q_ref, targets, ees,
                                                           max_iters=2000))
    for p in range(len(q)):
        T = K.frame_poses(robot, q_star[p])
        r = np.concatenate([K.pose_residual(T[e], targets[p, k]) for k, e in enumerate(ees)])
        assert np.linalg.norm(r) < 1e-3, f"{name} p={p}: left the manifold, |r|={np.linalg.norm(r):.2e}"
        assert np.linalg.norm(q_star[p] - q_ref[p]) <= np.linalg.norm(q[p] - q_ref[p]) + 1e-5
        J = np.concatenate([K.pose_jacobian(robot, q_star[p], e) for e in ees])
        N = np.eye(robot.n_act) - np.linalg.pinv(J) @ J      # null-space projector
        d = q_star[p] - q_ref[p]
        assert np.linalg.norm(N @ d) < 1e-2 * max(np.linalg.norm(d), 1e-3), (
            f"{name} p={p}: stationarity violated, null-space residual {np.linalg.norm(N @ d):.2e}")
