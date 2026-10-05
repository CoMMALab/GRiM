"""Shared fixtures for the motion equivalence suite (see test/motion/TESTING.md)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from grim.motion import MotionRobot

ASSETS = Path(__file__).resolve().parents[2] / "config" / "robot_assets"
ROBOTS = sorted(p.stem for p in ASSETS.glob("*.urdf"))


def gpu_available() -> bool:
    if shutil.which("nvcc") is None:
        return False
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def cricket_available() -> bool:
    try:
        import cricket  # noqa: F401
        return True
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(not gpu_available(), reason="needs a CUDA GPU, jax[cuda] and nvcc")


def load_robot(name: str) -> MotionRobot:
    return MotionRobot.from_urdf(str(ASSETS / f"{name}.urdf"))


def config_samples(robot: MotionRobot, n_random: int = 32, seed: int = 0) -> np.ndarray:
    """Principle 1 for kinematics: the zero pose, both limit corners, each joint alone at a
    limit (axis-aligned extremes), and uniform random configurations."""
    rng = np.random.default_rng(seed)
    lo, hi = robot.lower, robot.upper
    rows = [np.zeros(robot.n_act), lo, hi]
    for a in range(robot.n_act):
        for v in (lo[a], hi[a]):
            x = np.zeros(robot.n_act)
            x[a] = v
            rows.append(x)
    rows += list(rng.uniform(lo, hi, size=(n_random, robot.n_act)))
    return np.asarray(rows)


def ee_sets(robot: MotionRobot) -> list[tuple[int, ...]]:
    """One end-effector at the deepest actuated frame, and every leaf frame at once."""
    depth = {j: len(robot.chain(j)) for j in range(robot.n_joints)}
    deepest = max((j for j in range(robot.n_joints) if robot.act_idx[j] != -1), key=depth.get)
    parents = set(int(p) for p in robot.parent_idx)
    leaves = tuple(j for j in range(robot.n_joints) if j not in parents)
    return [(deepest,), leaves[:4]] if len(leaves) > 1 else [(deepest,)]


def assert_close_scaled(actual, expected, rtol: float, what: str = ""):
    """Principle 3: floor atol at rtol * max|expected|, so a structurally-zero entry carrying
    float32 round-off of the array's scale passes while an O(scale) error fails."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert np.all(np.isfinite(actual)), f"{what}: non-finite output"
    if actual.size == 0:
        return
    atol = rtol * max(np.abs(expected).max(initial=0.0), 1.0)
    err = np.abs(actual - expected)
    worst = np.unravel_index(np.argmax(err - rtol * np.abs(expected)), err.shape)
    assert np.all(err <= atol + rtol * np.abs(expected)), (
        f"{what}: max err {err[worst]:.3e} at {worst} (got {actual[worst]:.6g}, want "
        f"{expected[worst]:.6g}, atol {atol:.1e})")


@pytest.fixture(scope="session", autouse=True)
def _jax_gpu_memory():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
