"""Analytic IK (closed form, fr3): every branch it reports as found is certified by the FK
oracle, limits hold, near-every reachable target is found, and runs are deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import kinematics as K

from .conftest import load_robot, requires_gpu

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]


def _matrix(pose):
    w, x, y, z = pose[:4]
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    M[:3, 3] = pose[4:]
    return M


def test_analytic_ik_fr3():
    from grim.motion.analytic_ik import analytic_ik, arm_geometry
    robot = load_robot("fr3")
    ee = robot.joint_index("fr3_joint7")
    g = arm_geometry(robot, ee)
    rng = np.random.default_rng(0)
    qs = rng.uniform(robot.lower, robot.upper, size=(64, robot.n_act))
    poses = np.stack([K.frame_poses(robot, q)[ee] for q in qs])
    targets = np.stack([_matrix(p) for p in poses])
    q, err, found, _ = (np.asarray(x) for x in analytic_ik(robot, targets, ee,
                                                            q7_samples=np.linspace(g.lower[6], g.upper[6], 64)))
    assert found.mean() >= 0.95, f"found {found.mean():.0%} of reachable targets"
    for b in np.nonzero(found)[0]:
        full = np.zeros(robot.n_act)
        full[g.act] = q[b]
        r = K.pose_residual(K.frame_poses(robot, full)[ee], poses[b])
        assert np.linalg.norm(r[:3]) < 1e-3 and np.linalg.norm(r[3:]) < 1e-3, (b, r)
        assert np.all(q[b] >= g.lower - 1e-5) and np.all(q[b] <= g.upper + 1e-5)
    again = [np.asarray(x) for x in analytic_ik(robot, targets, ee,
                                                q7_samples=np.linspace(g.lower[6], g.upper[6], 64))]
    assert all(np.array_equal(a, b) for a, b in zip(again, (q, err, found)))


def test_analytic_ik_rejects_other_families():
    from grim.motion.analytic_ik import arm_geometry
    robot = load_robot("iiwa14")     # spherical wrist (SRS), not the offset-wrist family
    with pytest.raises(ValueError, match="family"):
        arm_geometry(robot, robot.joint_index("iiwa_joint_7"))
