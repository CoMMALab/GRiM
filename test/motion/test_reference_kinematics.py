"""Layer 1: the motion oracle's kinematics vs Pinocchio, float64.

The motion kernels are validated against ``grim.motion.reference``; this file is what makes
that reference trustworthy. Every joint frame of every asset robot is compared at the
zero pose, at both limit corners and at random configurations, and the geometric Jacobian is
compared to Pinocchio's ``LOCAL_WORLD_ALIGNED`` frame Jacobian.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from grim.motion import MotionRobot
from grim.motion.reference import kinematics as K

pytestmark = pytest.mark.pinocchio_equivalence

ASSETS = Path(__file__).resolve().parents[2] / "config" / "robot_assets"
ROBOTS = sorted(p.stem for p in ASSETS.glob("*.urdf"))


def _samples(robot: MotionRobot, n_random: int = 6) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    lo, hi = robot.lower.astype(np.float64), robot.upper.astype(np.float64)
    return [np.zeros(robot.n_act), lo, hi, *(rng.uniform(lo, hi) for _ in range(n_random))]


def _pin_q(model, robot: MotionRobot, q: np.ndarray, qfull: np.ndarray) -> np.ndarray:
    out = pin.neutral(model)
    for jid in range(1, model.njoints):
        name = model.names[jid]
        v = qfull[robot.joint_index(name)]
        i, nq = model.idx_qs[jid], model.nqs[jid]
        if nq == 2:            # continuous: (cos, sin)
            out[i:i + 2] = np.cos(v), np.sin(v)
        else:
            out[i] = v
    return out


def _frame(model, name):
    # A joint frame, never a same-named link frame (baxter has both).
    for kind in (pin.FrameType.JOINT, pin.FrameType.FIXED_JOINT):
        if model.existFrame(name, kind):
            return model.getFrameId(name, kind)
    raise KeyError(name)


@pytest.fixture(scope="module", params=ROBOTS)
def pair(request):
    xml = (ASSETS / f"{request.param}.urdf").read_text()
    return MotionRobot.from_urdf(xml), pin.buildModelFromXML(xml)


def test_frame_poses_match_pinocchio(pair):
    robot, model = pair
    data = model.createData()
    for q in _samples(robot):
        T = K.frame_poses(robot, q)
        pin.framesForwardKinematics(model, data, _pin_q(model, robot, q, K.full_q(robot, q)))
        for j, name in enumerate(robot.joint_names):
            M = data.oMf[_frame(model, name)]
            assert np.allclose(T[j, 4:], M.translation, atol=1e-10), name
            R = pin.Quaternion(*T[j, :4]).toRotationMatrix()   # (w, x, y, z)
            assert np.allclose(R, M.rotation, atol=1e-10), name


def test_pose_jacobian_matches_pinocchio(pair):
    robot, model = pair
    data = model.createData()
    # Pinocchio's tangent columns, mapped to actuated q (mimic joints are separate DOFs there).
    for q in _samples(robot, 3):
        qp = _pin_q(model, robot, q, K.full_q(robot, q))
        pin.computeJointJacobians(model, data, qp)
        pin.framesForwardKinematics(model, data, qp)
        for ee in (j for j in range(robot.n_joints) if robot.act_idx[j] != -1):
            name = robot.joint_names[ee]
            Jp = pin.getFrameJacobian(model, data, _frame(model, name),
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            want = np.zeros((6, robot.n_act))
            for jid in range(1, model.njoints):
                j = robot.joint_index(model.names[jid])
                a = robot.act_idx[j] if robot.act_idx[j] != -1 else robot.mimic_act_idx[j]
                if a != -1:
                    want[:, a] += robot.mimic_mul[j] * Jp[:, model.idx_vs[jid]]
            assert np.allclose(K.pose_jacobian(robot, q, ee), want, atol=1e-9), name


def test_pose_residual_is_zero_at_target_and_sign_invariant():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pose = np.r_[q, rng.normal(size=3)]
        assert np.allclose(K.pose_residual(pose, pose), 0, atol=1e-12)
        flipped = np.r_[-q, pose[4:]]
        assert np.allclose(K.pose_residual(pose, flipped), 0, atol=1e-12)
