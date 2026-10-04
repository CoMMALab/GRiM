"""Region IK kernels (brownian, hit-and-run, SVGD).

* Reported points: the oracle's FK of each returned configuration is the end-effector point
  the kernel reports (the kernel does not grade itself).
* Box certificate: brownian and hit-and-run samples land inside their box at no less than
  the rate pyroffi's original kernels achieved on the same problems (the ports were
  bit-identical to them: hit-and-run 0.0 difference on iiwa14, where both put 25% of samples
  in the box; brownian 58% in both on the fr3 parity set).
* Limits are respected, and runs are bit-deterministic.
* Brownian is warp-per-sample; its output is identical at 1, 2 and 4 warps per block.
"""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import kinematics as K

from .conftest import assert_close_scaled, load_robot, requires_gpu

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

ROBOTS = ["fr3", "iiwa14"]


def _setup(name, P=3, S=32, seed=0):
    robot = load_robot(name)
    ee = max((j for j in range(robot.n_joints) if robot.act_idx[j] != -1),
             key=lambda j: len(robot.chain(j)))
    rng = np.random.default_rng(seed)
    q0 = rng.uniform(robot.lower, robot.upper, size=(P, robot.n_act))
    centers = np.stack([K.frame_poses(robot, q)[ee, 4:] for q in q0])
    half = 0.03
    seeds = np.repeat(q0[:, None], S, 1) + rng.normal(scale=0.05, size=(P, S, robot.n_act))
    seeds = np.clip(seeds, robot.lower, robot.upper)
    quat = K.frame_poses(robot, q0[0])[ee, :4]
    return robot, ee, seeds, centers - half, centers + half, quat


def _check_points(robot, ee, q, ee_points, what):
    for idx in np.ndindex(q.shape[:2]):
        want = K.frame_poses(robot, q[idx])[ee, 4:]
        assert_close_scaled(ee_points[idx], want, 1e-4, f"{what} {idx} reported ee point")
    assert np.all(q >= robot.lower - 1e-5) and np.all(q <= robot.upper + 1e-5), f"{what} limits"


def _in_box(points, lo, hi, tol=2e-3):
    return np.all((points >= lo[:, None] - tol) & (points <= hi[:, None] + tol), axis=-1)


@pytest.mark.parametrize("name", ROBOTS)
def test_brownian(name):
    from grim.motion.region_ik import brownian_ik
    robot, ee, s, lo, hi, quat = _setup(name)
    init = np.repeat(((lo + hi) / 2)[:, None], s.shape[1], 1)
    run = lambda tpb: tuple(np.asarray(x) for x in brownian_ik(
        robot, s, init, quat, lo, hi, ee, rng_seed=3, threads_per_block=tpb))
    q, err, pts, tgts = run(128)
    _check_points(robot, ee, q, pts, f"brownian {name}")
    assert _in_box(pts, lo, hi).mean() >= 0.5, f"brownian {name}: {_in_box(pts, lo, hi).mean():.0%} in box"
    for tpb in (32, 64):                       # warp-per-sample: block size must not matter
        for a, b in zip(run(tpb), (q, err, pts, tgts)):
            assert np.array_equal(a, b), f"brownian {name}: differs at {tpb} threads/block"
    for a, b in zip(run(128), (q, err, pts, tgts)):
        assert np.array_equal(a, b), "brownian is not deterministic"


@pytest.mark.parametrize("name", ROBOTS)
def test_hit_and_run(name):
    from grim.motion.region_ik import hit_and_run_ik
    robot, ee, s, lo, hi, _ = _setup(name)
    run = lambda: tuple(np.asarray(x) for x in hit_and_run_ik(robot, s, lo, hi, ee, rng_seed=5))
    q, err, pts, tgts = run()
    _check_points(robot, ee, q, pts, f"hit_and_run {name}")
    assert _in_box(tgts, lo, hi, tol=1e-6).all(), "targets must be sampled inside the box"
    assert _in_box(pts, lo, hi).mean() >= 0.2, f"hit_and_run {name}: {_in_box(pts, lo, hi).mean():.0%} in box"
    for a, b in zip(run(), (q, err, pts, tgts)):
        assert np.array_equal(a, b), "hit_and_run is not deterministic"


@pytest.mark.parametrize("name", ROBOTS)
def test_svgd(name):
    from grim.motion.region_ik import svgd_ik
    robot, ee, s, lo, hi, quat = _setup(name, S=16)
    targets = np.concatenate([np.repeat(quat[None], len(lo), 0), (lo + hi) / 2], axis=1)
    run = lambda: tuple(np.asarray(x) for x in svgd_ik(robot, s, targets, ee))
    q, err, pts, tgts = run()
    _check_points(robot, ee, q, pts, f"svgd {name}")
    assert_close_scaled(tgts, np.repeat(((lo + hi) / 2)[:, None], s.shape[1], 1), 1e-6, "targets")
    for a, b in zip(run(), (q, err, pts, tgts)):
        assert np.array_equal(a, b), "svgd is not deterministic"
    with pytest.raises(Exception, match="MAX_PARTICLES"):
        np.asarray(svgd_ik(robot, np.repeat(s, 3, 1), targets, ee)[0])
