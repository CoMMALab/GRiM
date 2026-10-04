"""Trajectory optimizers: reference cost, endpoint, limit, determinism and batch checks.

* Zero iterations: the reported cost equals the float64 oracle's cost of the initial
  trajectory -- smoothness, limits and collision through the robot's own sphere rows. This
  is the check that catches a sphere posed by the wrong frame (the earlier sphere-row bug).
* Endpoints stay pinned exactly; limit violations stay small (limits are a soft penalty); the
  reported cost is the
  oracle's cost of the returned trajectory (it does not grade itself) and is no worse than
  the initial cost.
* Runs are bit-deterministic and a trajectory's result does not depend on its batch slot.
"""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import trajopt as ref

from .conftest import assert_close_scaled, load_robot, requires_gpu
from .traj_cases import obstacle_world, sphere_model, straight_lines

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

ROBOTS = ["fr3", "iiwa14"]
OPTIMIZERS = ["sco", "stomp", "chomp", "ls"]
W = dict(w_smooth=1.0, w_acc=0.5, w_jerk=0.1, w_limits=1.0, w_collision_max=100.0,
         collision_margin=0.01)
ZERO = {"sco": dict(n_outer_iters=0), "stomp": dict(n_iters=0), "chomp": dict(n_iters=0),
        "ls": dict(n_outer_iters=0)}
COST = {"sco": ref.sco_cost, "stomp": ref.stomp_cost, "chomp": ref.stomp_cost,
        "ls": ref.ls_cost}


def _run(kind, robot, x, tc, world, start, goal, **kw):
    from grim.motion import trajopt
    fn = {"sco": trajopt.sco_trajopt, "stomp": trajopt.stomp_trajopt,
          "chomp": trajopt.chomp_trajopt, "ls": trajopt.ls_trajopt}[kind]
    t, c = fn(robot, x, collision=tc, world=world, start=start, goal=goal, **W, **kw)
    return np.asarray(t), np.asarray(c)


def _setup(name, B=4, T=24):
    robot = load_robot(name)
    x, start, goal = straight_lines(robot, B, T)
    return robot, x, start, goal, sphere_model(robot), obstacle_world(robot, start, goal)


def _oracle(kind, robot, traj, tc, world):
    return COST[kind](robot, traj, tc, world, lower=robot.lower, upper=robot.upper, **W)


@pytest.mark.parametrize("kind", OPTIMIZERS)
@pytest.mark.parametrize("name", ROBOTS)
def test_zero_iterations_cost_matches_reference(kind, name):
    robot, x, start, goal, tc, world = _setup(name)
    t, c = _run(kind, robot, x, tc, world, start, goal, **ZERO[kind])
    assert_close_scaled(t, x, 1e-6, "trajectory moved with zero iterations")
    want = np.array([_oracle(kind, robot, x[b], tc, world) for b in range(len(x))])
    assert want.max() > 1.0, "test scene must have active collision terms"
    assert_close_scaled(c, want, 1e-4, f"{kind} {name} initial cost")


@pytest.mark.parametrize("kind", OPTIMIZERS)
@pytest.mark.parametrize("name", ROBOTS)
def test_solution(kind, name):
    robot, x, start, goal, tc, world = _setup(name)
    t, c = _run(kind, robot, x, tc, world, start, goal)
    assert np.array_equal(t[:, 0], np.broadcast_to(start.astype(np.float32), t[:, 0].shape))
    assert np.array_equal(t[:, -1], np.broadcast_to(goal.astype(np.float32), t[:, -1].shape))
    # Limits are a soft penalty (in the oracle cost below), not a constraint: bound the
    # violation instead of forbidding it. STOMP's sampled candidates overshoot by up to
    # 0.076 rad here, identically in pyroffi's original kernel (parity to 2e-7).
    viol = np.maximum(t - robot.upper, 0) + np.maximum(robot.lower - t, 0)
    assert viol.max() < 0.1, f"{kind} {name}: limit violation {viol.max():.3f} rad"
    want = np.array([_oracle(kind, robot, t[b], tc, world) for b in range(len(t))])
    assert_close_scaled(c, want, 1e-3, f"{kind} {name} reported cost")
    init = np.array([_oracle(kind, robot, x[b], tc, world) for b in range(len(x))])
    assert np.all(want <= init * (1 + 1e-5) + 1e-6), f"{kind} {name}: cost increased {init} -> {want}"


@pytest.mark.parametrize("kind", OPTIMIZERS)
def test_deterministic(kind):
    robot, x, start, goal, tc, world = _setup("fr3")
    a = _run(kind, robot, x, tc, world, start, goal)
    b = _run(kind, robot, x, tc, world, start, goal)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


@pytest.mark.parametrize("kind", ["sco", "chomp", "ls"])
def test_batch_invariant(kind):
    """Trajectories are independent blocks. (STOMP keys its noise on the batch slot.)"""
    robot, x, start, goal, tc, world = _setup("fr3")
    a = _run(kind, robot, x, tc, world, start, goal)
    r = _run(kind, robot, x[::-1], tc, world, start, goal)
    assert np.array_equal(r[0][::-1], a[0]) and np.array_equal(r[1][::-1], a[1])
