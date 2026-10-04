"""Layer 2: the motion kernels' kinematics vs the numpy oracle, on every asset robot.

Every solver reads the robot through robot.cuh, so this is the primitive gate the solver
tests build on: frame poses, pose residuals and geometric Jacobians, for one and several
end-effectors, over all joints and over the end-effector chains alone, from the baked
tables and (when cricket is installed) from the traced straight-line FK.
"""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import kinematics as K

from .conftest import (ROBOTS, assert_close_scaled, config_samples, cricket_available, ee_sets,
                       load_robot, requires_gpu)

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

RTOL = 2e-5   # float32 kernels vs float64 oracle
CONTINUOUS = {"fetch", "gen3"}   # cricket cannot trace continuous joints


def _targets(robot, q, ees, rng):
    # A target near each EE's pose at a perturbed configuration: residuals are O(0.1-1),
    # including near-pi rotation errors, rather than ~0.
    q2 = q + rng.normal(scale=0.8, size=q.shape)
    out = np.zeros((len(q), len(ees), 7))
    for b in range(len(q)):
        T = K.frame_poses(robot, q2[b])
        out[b] = T[list(ees)]
    return out


@pytest.mark.parametrize("traced", [False, pytest.param(True, marks=pytest.mark.skipif(
    not cricket_available(), reason="cricket not installed"))], ids=["tables", "traced"])
@pytest.mark.parametrize("chain_only", [False, True], ids=["all", "chain"])
@pytest.mark.parametrize("name", ROBOTS)
def test_kinematics_match_reference(name, chain_only, traced):
    from grim.motion.kinematics import kinematics, solved_columns

    robot = load_robot(name)
    if traced and name in CONTINUOUS:
        with pytest.raises(ValueError, match="continuous"):
            kinematics(robot, np.zeros((1, robot.n_act)), None, ee_sets(robot)[0], traced=True)
        return
    q = config_samples(robot)
    rng = np.random.default_rng(1)
    for ees in ee_sets(robot):
        tgt = _targets(robot, q, ees, rng)
        T, r, J = (np.asarray(x) for x in kinematics(robot, q, tgt, ees,
                                                      chain_only=chain_only, traced=traced))
        cols = solved_columns(robot, ees, chain_only)
        for b in range(len(q)):
            Tw = K.frame_poses(robot, q[b])
            # Quaternion sign is a gauge: compare rotations via the residual against itself.
            assert_close_scaled(T[b, :, 4:], Tw[:, 4:], RTOL, f"{name} b={b} positions")
            for j in range(robot.n_joints):
                assert np.allclose(K.pose_residual(T[b, j], Tw[j]), 0, atol=1e-5), \
                    f"{name} b={b} frame {robot.joint_names[j]} rotation"
            r_ref = np.concatenate([K.pose_residual(Tw[e], tgt[b, k]) for k, e in enumerate(ees)])
            J_ref = np.concatenate([K.pose_jacobian(robot, q[b], e) for e in ees])[:, cols]
            # Rotation residuals near pi are ill-conditioned in float32: compare where the
            # rotation error is below 3 rad.
            ok = np.repeat([np.linalg.norm(r_ref[6 * k + 3:6 * k + 6]) < 3.0
                            for k in range(len(ees))], 6)
            assert_close_scaled(r[b][ok], r_ref[ok], 1e-4, f"{name} b={b} residual")
            assert_close_scaled(J[b], J_ref, RTOL, f"{name} b={b} jacobian")


@pytest.mark.parametrize("name", ["fr3", "iiwa14"])
def test_chain_only_solves_fewer_columns(name):
    from grim.motion.kinematics import solved_columns

    robot = load_robot(name)
    for ees in ee_sets(robot):
        cols = solved_columns(robot, ees, chain_only=True)
        want = sorted({int(robot.act_idx[j] if robot.act_idx[j] != -1 else robot.mimic_act_idx[j])
                       for e in ees for j in robot.chain(e)} - {-1})
        assert list(cols) == want
