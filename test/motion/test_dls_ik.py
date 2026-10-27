"""DLS-IK (Levenberg-Marquardt, dual form): 0, 1 and 2 iterations match the float64
transcription in grim.motion.reference.ik, seed by seed. The solver-agnostic checks are in
test_ik_solvers.py."""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import ik as ref

from .conftest import assert_close_scaled, ee_sets, load_robot, requires_gpu
from .ik_cases import reachable_targets, seeds

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

SETTINGS = dict(pos_weight=50.0, ori_weight=10.0, lambda_init=5e-3, eps_pos=1e-4, eps_ori=1e-3)
CASES = [("fr3", 0), ("iiwa14", 0), ("baxter", 0), ("baxter", 1), ("g1", 0), ("g1", 1)]  # (robot, ee set)
# (backend, rtol): the double build agrees with the float64 transcription far tighter; 1e-6
# leaves room for the float32 seeds and targets the launcher passes.
BACKENDS = [("crocket-f32", 1e-4), ("crocket-f64", 1e-6)]


@pytest.mark.parametrize("backend,rtol", BACKENDS)
@pytest.mark.parametrize("name,which", CASES)
@pytest.mark.parametrize("iters", [0, 1, 2])
def test_iterations_match_reference(name, which, iters, backend, rtol):
    from grim.motion.ik import dls_ik
    from grim.motion.kinematics import solved_columns
    robot = load_robot(name)
    ees = ee_sets(robot)[which]
    targets, _ = reachable_targets(robot, ees, 3)
    s = seeds(robot, 3, 4)
    q, e = (np.asarray(x) for x in dls_ik(robot, s, targets, ees, max_iter=iters, backend=backend,
                                          **SETTINGS))
    cols = solved_columns(robot, ees, chain_only=True)
    for p in range(s.shape[0]):
        for k in range(s.shape[1]):
            q_ref, e_ref = ref.dls_ik(robot, s[p, k], targets[p], ees, cols, lower=robot.lower,
                                      upper=robot.upper, fixed_mask=np.zeros(robot.n_act),
                                      max_iter=iters, **SETTINGS)
            assert_close_scaled(q[p, k], q_ref, rtol, f"{backend} {name} p={p} s={k} q")
            assert_close_scaled(e[p, k], e_ref, 10 * rtol, f"{backend} {name} p={p} s={k} err")


def test_collision_is_rejected():
    from grim.motion import ik
    robot = load_robot("fr3")
    ees = ee_sets(robot)[0]
    t = ik.build("dls", robot, ees)
    targets, _ = reachable_targets(robot, ees, 1)
    with pytest.raises(Exception, match="no collision terms"):
        np.asarray(ik.run(t, seeds(robot, 1, 2), targets, enable_collision=True, max_iter=1,
                          **SETTINGS)[0])

