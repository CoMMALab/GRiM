"""Checks every IK solver kernel must pass (see test/motion/TESTING.md).

* Zero iterations: the solver returns the seed and the oracle's merit for it.
* Thread-count invariance (Principle 2): the block tier is bit-identical at 32, 64, 128 and a
  random non-multiple-of-32 block size.
* Tier agreement: thread, warp and block tiers agree to round-off after one iteration.
* Batch invariance: a problem's answer does not depend on its slot or the batch size.
* Certificates: on reachable targets the oracle confirms the returned pose error, the
  joint limits, and the solver's reported error.
Solver-specific reference iterations live in test_<solver>.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import ik as ref

from .conftest import assert_close_scaled, ee_sets, load_robot, requires_gpu
from .ik_cases import pose_errors, reachable_targets, seeds

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

RANDOM_THREADS = int(np.random.default_rng().integers(33, 255)) | 1   # odd: never a multiple of 32
CASES = [("fr3", 0), ("iiwa14", 0), ("baxter", 1), ("g1", 1)]       # (robot, ee set)
W = dict(pos_weight=50.0, ori_weight=10.0)


def _hjcd_noise(s, iters):
    """Stall-kick noise derived from the seed values, so it permutes with the seeds."""
    k = np.arange(max(iters, 1))[None, None, :, None]
    return 0.5 * np.sin(1e3 * s[:, :, None, :] + 17.0 * k)


def _solve(kind, robot, ees, s, t, iters, **kw):
    from grim.motion import ik
    if kind == "ls":
        out = ik.ls_ik(robot, s, t, ees, max_iter=iters, **W, **kw)
    elif kind == "sqp":
        out = ik.sqp_ik(robot, s, t, ees, max_iter=iters, **W, **kw)
    elif kind == "hjcd":
        out = ik.hjcd_ik_lm(robot, s, t, ees, _hjcd_noise(s, iters), max_iter=iters, **kw)
    elif kind == "hjcd_coarse":
        out = ik.hjcd_ik_coarse(robot, s, t, ees, k_max=iters, **kw)
    elif kind == "mppi":
        n_mppi = min(iters, 4)
        out = ik.mppi_ik(robot, s, t, ees, rng_seed=7, n_mppi_iters=n_mppi,
                         n_lbfgs_iters=iters - n_mppi, **W, **kw)
    return tuple(np.asarray(x) for x in out)


SOLVERS = ["ls", "sqp", "hjcd", "hjcd_coarse", "mppi"]
TIERED = ["ls", "sqp", "hjcd"]                 # one seed per thread / warp / block
SLOT_INVARIANT = ["ls", "sqp", "hjcd", "hjcd_coarse"]   # mppi keys its noise on the slot
WEIGHTED = {"ls": True, "sqp": True, "hjcd": False, "hjcd_coarse": False, "mppi": True}
# Minimum solve rate on reachable targets (certificate test); coarse search is phase 1 only.
# hjcd here is phase 2 alone (no coarse search): on the fr3 set pyroffi's original kernel
# solved 92.5% and this port 87.5%, the gap being iteration chaos after ~10 steps (the two
# agree to 1e-5 through 10 iterations; see docs/source/user_guide/concepts/motion.rst).
SOLVE_RATE = {"ls": 0.95, "sqp": 0.95, "hjcd": 0.85, "mppi": 0.9}


def _case(name, which, n_problems=6, n_seeds=8):
    robot = load_robot(name)
    ees = ee_sets(robot)[which]
    targets, _ = reachable_targets(robot, ees, n_problems)
    return robot, ees, seeds(robot, n_problems, n_seeds), targets


def _merit(kind, robot, ees, q, target):
    """The oracle's value of the error a solver reports (weighted or plain squared)."""
    from grim.motion.kinematics import solved_columns
    w = W if WEIGHTED[kind] else dict(pos_weight=1.0, ori_weight=1.0)
    _, e = ref.ls_ik(robot, q, target, ees, solved_columns(robot, ees, True),
                     lower=robot.lower, upper=robot.upper, fixed_mask=np.zeros(robot.n_act),
                     max_iter=0, lambda_init=1e-3, eps_pos=0, eps_ori=0, **w)
    return e


@pytest.mark.parametrize("kind", SOLVERS)
@pytest.mark.parametrize("name,which", CASES)
def test_zero_iterations_return_seed_and_merit(kind, name, which):
    robot, ees, s, t = _case(name, which, 3, 4)
    q, e = _solve(kind, robot, ees, s, t, 0)[:2]
    assert_close_scaled(q, s, 1e-6, f"{kind} {name} q")
    want = np.array([[_merit(kind, robot, ees, s[p, k], t[p]) for k in range(s.shape[1])]
                     for p in range(s.shape[0])])
    assert_close_scaled(e, want, 1e-4, f"{kind} {name} merit")


@pytest.mark.parametrize("kind", TIERED)
@pytest.mark.parametrize("name,which", CASES)
def test_block_tier_is_thread_count_invariant(kind, name, which):
    robot, ees, s, t = _case(name, which)
    want = _solve(kind, robot, ees, s, t, 20, tier="block", block_threads=32)
    for threads in (64, 128, RANDOM_THREADS):
        got = _solve(kind, robot, ees, s, t, 20, tier="block", block_threads=threads)
        for a, b in zip(got, want):
            assert np.array_equal(a, b), f"{kind} {name}: block tier at {threads} threads differs"


@pytest.mark.parametrize("kind", TIERED)
@pytest.mark.parametrize("name,which", CASES)
def test_tiers_agree_to_roundoff(kind, name, which):
    """The tiers run the same step with differently-ordered Cholesky factorizations (GLASS
    thread / warp / block potrf), so one step agrees to float32 round-off; later iterates
    drift chaotically near convergence, which the certificate test covers."""
    robot, ees, s, t = _case(name, which)
    want = _solve(kind, robot, ees, s, t, 1, tier="thread")
    for tier in ("warp", "block"):
        got = _solve(kind, robot, ees, s, t, 1, tier=tier)
        assert_close_scaled(got[0], want[0], 1e-6, f"{kind} {name} {tier} q")
        assert_close_scaled(got[1], want[1], 1e-5, f"{kind} {name} {tier} err")


@pytest.mark.parametrize("kind", SLOT_INVARIANT)
@pytest.mark.parametrize("name,which", CASES[:2])
def test_batch_position_invariance(kind, name, which):
    robot, ees, s, t = _case(name, which)
    want = _solve(kind, robot, ees, s, t, 20)[:2]
    rev = slice(None, None, -1)
    got = _solve(kind, robot, ees, s[rev, rev], t[rev], 20)[:2]
    for a, b in zip(got, want):
        assert np.array_equal(a[rev, rev], b)
    one = _solve(kind, robot, ees, s[2:3], t[2:3], 20)[:2]
    for a, b in zip(one, want):
        assert np.array_equal(a[0], b[2])


@pytest.mark.parametrize("kind", SOLVERS)
@pytest.mark.parametrize("name,which", CASES[:2])
def test_deterministic(kind, name, which):
    """Run-to-run bit determinism (no atomics or races decide an output)."""
    robot, ees, s, t = _case(name, which)
    a, b = _solve(kind, robot, ees, s, t, 20), _solve(kind, robot, ees, s, t, 20)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("kind", list(SOLVE_RATE))
@pytest.mark.parametrize("name,which", [("fr3", 0), ("iiwa14", 0), ("baxter", 1)])
def test_reachable_targets_are_solved(kind, name, which):
    robot, ees, s, t = _case(name, which, n_problems=40, n_seeds=16)
    out = _solve(kind, robot, ees, s, t, 100)
    q, e = out[:2]
    assert np.all(q >= robot.lower - 1e-6) and np.all(q <= robot.upper + 1e-6), "limits violated"
    best = q[np.arange(len(q)), e.argmin(1)]
    ok = [all(np.array(pose_errors(robot, best[p], t[p], ees)) < (1e-3, 1e-2))
          for p in range(len(q))]
    assert np.mean(ok) >= SOLVE_RATE[kind], f"{kind} {name}: solved {np.mean(ok):.0%}"
    for p in range(len(q)):   # the reported error is the oracle's, not self-graded
        assert_close_scaled(e[p].min(), _merit(kind, robot, ees, best[p], t[p]), 1e-3,
                            f"{kind} {name} p={p} reported error")
