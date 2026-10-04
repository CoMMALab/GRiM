"""Runtime parent rotation (``runtime_rot_joint``): a build that takes one joint's parent rotation
per call must behave exactly like a build with that rotation baked in.

The oracle is the reference evaluated on a copy of the robot whose table carries the rotation.
Two launches with different rotations inside one jit check that each launch's upload is ordered
before its own kernel and does not leak into the other.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from grim.motion.reference import kinematics as K
from grim.motion.reference import trajopt as ref_traj

from .conftest import assert_close_scaled, config_samples, load_robot, requires_gpu
from .ik_cases import pose_errors, reachable_targets, seeds
from .traj_cases import obstacle_world, sphere_model, straight_lines

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

JOINT = 0   # the root joint: every frame below it moves with the rotation


def _rotations(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _with_rotation(robot, rot):
    pt = np.array(robot.parent_tf, copy=True)
    pt[JOINT, :4] = rot
    return dataclasses.replace(robot, parent_tf=pt)


def test_kinematics_and_launch_ordering():
    import jax
    from grim.motion.kinematics import kinematics

    robot = load_robot("iiwa14")
    q = config_samples(robot)
    r1, r2 = _rotations(2)

    @jax.jit
    def two(q):
        return (kinematics(robot, q, runtime_rot_joint=JOINT, rot=r1)[0],
                kinematics(robot, q, runtime_rot_joint=JOINT, rot=r2)[0])

    for T, rot in zip(two(q), (r1, r2)):
        T = np.asarray(T)
        mod = _with_rotation(robot, rot)
        for b in range(len(q)):
            Tw = K.frame_poses(mod, q[b])
            assert_close_scaled(T[b, :, 4:], Tw[:, 4:], 1e-5, f"b={b} positions")
            for j in range(robot.n_joints):
                assert np.allclose(K.pose_residual(T[b, j], Tw[j]), 0, atol=1e-5)


def test_ls_ik_solves_the_rotated_problem():
    from grim.motion import ik

    robot = load_robot("iiwa14")
    mod = _with_rotation(robot, _rotations(1, seed=3)[0])
    ees = (robot.n_joints - 1,)
    targets, _ = reachable_targets(mod, ees, 6)
    s = seeds(robot, 6, 8)
    t = ik.build("ls", robot, ees, runtime_rot_joint=JOINT)
    q, e = (np.asarray(x) for x in ik.run(t, s, targets, rot=mod.parent_tf[JOINT, :4],
                                          max_iter=100, pos_weight=50.0, ori_weight=10.0,
                                          lambda_init=5e-3, eps_pos=1e-4, eps_ori=1e-3,
                                          collision_weight=1e4, collision_margin=0.02))
    best = np.argmin(e, axis=1)
    for p in range(6):
        pos, rot = pose_errors(mod, q[p, best[p]], targets[p], ees)
        assert pos < 1e-3 and rot < 1e-2, (p, pos, rot)


def test_trajopt_zero_iteration_cost_uses_the_rotation():
    from grim.motion import trajopt

    robot = load_robot("iiwa14")
    mod = _with_rotation(robot, _rotations(1, seed=5)[0])
    x, start, goal = straight_lines(robot, 4, 24)
    tc, world = sphere_model(robot), obstacle_world(mod, start, goal)
    w = dict(w_smooth=1.0, w_acc=0.5, w_jerk=0.1, w_limits=1.0, w_collision_max=100.0,
             collision_margin=0.01)
    _, c = trajopt.sco_trajopt(robot, x, collision=tc, world=world, start=start, goal=goal,
                               n_outer_iters=0, runtime_rot_joint=JOINT,
                               rot=mod.parent_tf[JOINT, :4], **w)
    want = np.array([ref_traj.sco_cost(mod, x[b], tc, world, lower=robot.lower,
                                       upper=robot.upper, **w) for b in range(len(x))])
    assert want.max() > 1.0, "test scene must have active collision terms"
    assert_close_scaled(np.asarray(c), want, 1e-4, "rotated initial cost")


def test_traced_build_refuses_a_runtime_rotation():
    from grim.motion import ik

    robot = load_robot("iiwa14")
    with pytest.raises(ValueError, match="baked tables"):
        ik.build("ls", robot, (robot.n_joints - 1,), traced=True, runtime_rot_joint=JOINT)
