"""LS-IK (Levenberg-Marquardt): 0, 1 and 2 iterations match the float64 transcription in
grim.motion.reference.ik, seed by seed. The solver-agnostic checks are in
test_ik_solvers.py."""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import ik as ref

from .conftest import assert_close_scaled, ee_sets, load_robot, requires_gpu
from .ik_cases import reachable_targets, seeds

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

SETTINGS = dict(pos_weight=50.0, ori_weight=10.0, lambda_init=5e-3, eps_pos=1e-4, eps_ori=1e-3)
CASES = [("fr3", 0), ("iiwa14", 0), ("baxter", 1), ("g1", 1)]       # (robot, ee set)


def _solve(robot, ees, seeds_, targets, **kw):
    from grim.motion.ik import ls_ik
    q, e = ls_ik(robot, seeds_, targets, ees, **SETTINGS, **kw)
    return np.asarray(q), np.asarray(e)


def _case(name, which, n_problems=6, n_seeds=8):
    robot = load_robot(name)
    ees = ee_sets(robot)[which]
    targets, _ = reachable_targets(robot, ees, n_problems)
    return robot, ees, seeds(robot, n_problems, n_seeds), targets


@pytest.mark.parametrize("name,which", CASES)
@pytest.mark.parametrize("iters", [0, 1, 2])
def test_iterations_match_reference(name, which, iters):
    from grim.motion.kinematics import solved_columns
    robot, ees, s, targets = _case(name, which, n_problems=3, n_seeds=4)
    q, e = _solve(robot, ees, s, targets, max_iter=iters)
    cols = solved_columns(robot, ees, chain_only=True)
    for p in range(s.shape[0]):
        for k in range(s.shape[1]):
            q_ref, e_ref = ref.ls_ik(robot, s[p, k], targets[p], ees, cols, lower=robot.lower,
                                     upper=robot.upper, fixed_mask=np.zeros(robot.n_act),
                                     max_iter=iters, **SETTINGS)
            assert_close_scaled(q[p, k], q_ref, 1e-4, f"{name} p={p} s={k} q")
            assert_close_scaled(e[p, k], e_ref, 1e-3, f"{name} p={p} s={k} err")
