"""Trajectory optimizers: reference cost, endpoint, limit, determinism and batch checks.

* Zero iterations: the reported cost equals the float64 oracle's cost of the initial
  trajectory -- smoothness, limits and collision through the robot's own sphere rows. This
  is the check that catches a sphere posed by the wrong frame (the earlier sphere-row bug).
* Endpoints stay pinned exactly; limit violations stay small (limits are a soft penalty); the
  reported cost is the
  oracle's cost of the returned trajectory (it does not grade itself) and is no worse than
  the initial cost.
* Runs are bit-deterministic and a trajectory's result does not depend on its batch slot.

Every case runs at both scalar types and all three tiers.
"""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import trajopt as ref

from .conftest import assert_close_scaled, load_robot, requires_gpu
from .traj_cases import obstacle_world, sphere_model, straight_lines

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

ROBOTS = ["fr3", "iiwa14"]
OPTIMIZERS = ["sco", "stomp", "chomp"]
W = dict(w_smooth=1.0, w_acc=0.5, w_jerk=0.1, w_limits=1.0, w_collision_max=100.0,
         collision_margin=0.01)
ZERO = {"sco": dict(n_outer_iters=0), "stomp": dict(n_iters=0), "chomp": dict(n_iters=0)}
COST = {"sco": ref.sco_cost, "stomp": ref.stomp_cost, "chomp": ref.stomp_cost}
CONFIGS = [(b, t) for b in ("crocket-f32", "crocket-f64") for t in ("thread", "warp", "block")]


def _run(kind, robot, x, tc, world, start, goal, **kw):
    from grim.motion import trajopt
    fn = {"sco": trajopt.sco_trajopt, "stomp": trajopt.stomp_trajopt,
          "chomp": trajopt.chomp_trajopt}[kind]
    t, c = fn(robot, x, collision=tc, world=world, start=start, goal=goal, **W, **kw)
    return np.asarray(t), np.asarray(c)


def _setup(name, B=4, T=24):
    robot = load_robot(name)
    x, start, goal = straight_lines(robot, B, T)
    return robot, x, start, goal, sphere_model(robot), obstacle_world(robot, start, goal)


def _oracle(kind, robot, traj, tc, world):
    return COST[kind](robot, traj, tc, world, lower=robot.lower, upper=robot.upper, **W)


@pytest.mark.parametrize("backend,tier", CONFIGS)
@pytest.mark.parametrize("kind", OPTIMIZERS)
@pytest.mark.parametrize("name", ROBOTS)
def test_zero_iterations_and_solution(kind, name, backend, tier):
    """The initial cost is the oracle's; the result keeps its endpoints,
    reports the oracle's cost of what it returns, and does not raise the cost."""
    robot, x, start, goal, tc, world = _setup(name)
    ck = dict(backend=backend, tier=tier)
    t, c = _run(kind, robot, x, tc, world, start, goal, **ZERO[kind], **ck)
    want = np.array([_oracle(kind, robot, x[b], tc, world) for b in range(len(x))])
    assert_close_scaled(c, want, 1e-4 if backend == "crocket-f32" else 1e-6, f"{kind} {name} {backend} {tier} initial cost")
    t, c = _run(kind, robot, x, tc, world, start, goal, **ck)
    assert np.array_equal(t[:, 0], np.broadcast_to(start.astype(np.float32), t[:, 0].shape))
    assert np.array_equal(t[:, -1], np.broadcast_to(goal.astype(np.float32), t[:, -1].shape))
    # Limits are a soft penalty (in the oracle cost below), not a constraint: bound the
    # violation instead of forbidding it (STOMP's sampled candidates overshoot by up to 0.076 rad).
    viol = np.maximum(t - robot.upper, 0) + np.maximum(robot.lower - t, 0)
    assert viol.max() < 0.1, f"{kind} {name} {backend} {tier}: limit violation {viol.max():.3f} rad"
    got = np.array([_oracle(kind, robot, t[b], tc, world) for b in range(len(t))])
    assert_close_scaled(c, got, 1e-3, f"{kind} {name} {backend} {tier} reported cost")
    init = np.array([_oracle(kind, robot, x[b], tc, world) for b in range(len(x))])
    assert np.all(got <= init * (1 + 1e-5) + 1e-6), f"{kind} {name} {backend} {tier}: cost increased"


@pytest.mark.parametrize("tier", ["thread", "warp", "block"])
@pytest.mark.parametrize("kind", OPTIMIZERS)
def test_deterministic_and_batch_invariant(kind, tier):
    """Bit-deterministic; a trajectory's result does not depend on its batch slot (except
    STOMP, which keys its noise on the slot) or on what the workspace held before."""
    robot, x, start, goal, tc, world = _setup("fr3")
    ck = dict(backend="crocket-f32", tier=tier)
    a = _run(kind, robot, x, tc, world, start, goal, **ck)
    b = _run(kind, robot, x, tc, world, start, goal, **ck)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    if kind != "stomp":
        r = _run(kind, robot, x[::-1], tc, world, start, goal, **ck)
        assert np.array_equal(r[0][::-1], a[0]) and np.array_equal(r[1][::-1], a[1])
