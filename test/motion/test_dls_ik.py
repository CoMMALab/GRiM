"""DLS-IK (Levenberg-Marquardt, dual form): 0, 1 and 2 iterations match the float64
transcription in grim.motion.reference.ik, seed by seed. The solver-agnostic checks are in
test_ik_solvers.py."""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import ik as ref

from .conftest import assert_close_scaled, crocket_available, ee_sets, load_robot, requires_gpu
from .ik_cases import reachable_targets, seeds

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

SETTINGS = dict(pos_weight=50.0, ori_weight=10.0, lambda_init=5e-3, eps_pos=1e-4, eps_ori=1e-3)
CASES = [("fr3", 0), ("iiwa14", 0), ("baxter", 0), ("baxter", 1), ("g1", 0), ("g1", 1)]  # (robot, ee set)
TRACED = [False, pytest.param(True, marks=pytest.mark.skipif(
    not crocket_available(), reason="crocket not installed"))]


@pytest.mark.parametrize("traced", TRACED, ids=["tables", "traced"])
@pytest.mark.parametrize("name,which", CASES)
@pytest.mark.parametrize("iters", [0, 1, 2])
def test_iterations_match_reference(name, which, iters, traced):
    """traced=True on a one-end-effector set runs crocket's fused step (GRIM_FUSED_DLS)."""
    from grim.motion.ik import dls_ik
    from grim.motion.kinematics import solved_columns
    robot = load_robot(name)
    ees = ee_sets(robot)[which]
    targets, _ = reachable_targets(robot, ees, 3)
    s = seeds(robot, 3, 4)
    q, e = (np.asarray(x) for x in dls_ik(robot, s, targets, ees, max_iter=iters, traced=traced,
                                                   **SETTINGS))
    cols = solved_columns(robot, ees, chain_only=True)
    for p in range(s.shape[0]):
        for k in range(s.shape[1]):
            q_ref, e_ref = ref.dls_ik(robot, s[p, k], targets[p], ees, cols, lower=robot.lower,
                                      upper=robot.upper, fixed_mask=np.zeros(robot.n_act),
                                      max_iter=iters, **SETTINGS)
            assert_close_scaled(q[p, k], q_ref, 1e-4, f"{name} p={p} s={k} q")
            assert_close_scaled(e[p, k], e_ref, 1e-3, f"{name} p={p} s={k} err")


def test_collision_is_rejected():
    from grim.motion import ik
    robot = load_robot("fr3")
    ees = ee_sets(robot)[0]
    t = ik.build("dls", robot, ees)
    targets, _ = reachable_targets(robot, ees, 1)
    with pytest.raises(Exception, match="no collision terms"):
        np.asarray(ik.run(t, seeds(robot, 1, 2), targets, enable_collision=True, max_iter=1,
                          **SETTINGS)[0])


@pytest.mark.skipif(not crocket_available(), reason="crocket not installed")
def test_fused_build_rejects_what_it_bakes():
    from grim.motion import ik
    robot = load_robot("fr3")
    ees = ee_sets(robot)[0]
    t = ik.build("dls", robot, ees, traced=True, fused_dls_ori_ratio=0.2)
    targets, _ = reachable_targets(robot, ees, 1)
    s = seeds(robot, 1, 2)
    with pytest.raises(ValueError, match="bakes ori_weight"):
        ik.run(t, s, targets, **{**SETTINGS, "ori_weight": 20.0})
    with pytest.raises(ValueError, match="bakes the URDF limits"):
        ik.run(t, s, targets, lower=robot.lower, **SETTINGS)
