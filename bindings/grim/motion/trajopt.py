"""Trajectory optimization on the GPU, built per robot.

Every optimizer takes initial trajectories ``(B, T, n_q)``, joint limits, endpoints (``(n_q,)``
shared or ``(B, n_q)`` per trajectory) and the world, and returns the optimized trajectories
with a cost per trajectory. Collision uses a :class:`~grim.motion._build.TrajCollision` model,
traced by crocket with its analytic gradient (:mod:`grim.motion._crocket_traj`); ``backend``
picks the scalar type, ``"crocket-f32"`` (default) or ``"crocket-f64"``, and ``tier`` gives one
trajectory a thread, a warp or a block of ``block_threads`` lanes. ``runtime_rot_joint`` >= 0
takes that joint's parent rotation per call as ``rot`` (wxyz): the reference rotation of a
floating base's chart.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from ._build import TrajCollision
from .ik import World
from .robot import MotionRobot

SCO_MAX_G, SCO_MAX_M = 5, 8      # kernels/trajopt_crocket/sco_trajopt.cu
BACKENDS = {"crocket-f32": "float", "crocket-f64": "double"}   # backend -> scalar type
TIERS = {"thread": 0, "warp": 1, "block": 2}


def _ck_target(kernel: str, symbol: str, robot: MotionRobot, prob, backend: str) -> str:
    """Build (or load) a trajopt kernel: the robot header plus the traced collision terms of
    :mod:`grim.motion._crocket_traj`."""
    from . import _crocket_traj
    header = _build.robot_header(robot, prob) + _crocket_traj.source(
        robot, prob.traj_collision, prob.runtime_rot_joint, prob.world_counts, BACKENDS[backend])
    flags = (f"-DGRIM_CK_REAL={BACKENDS[backend]}",
             *((f"-DGRIM_RUNTIME_ROT_JOINT={prob.runtime_rot_joint}",)
               if prob.runtime_rot_joint >= 0 else ()))
    so = _build.build(f"trajopt_crocket/{kernel}", header, robot.n_act, robot.n_joints, 0, flags)
    return _build.register(so, (symbol,))[0]


def _ck_check(backend: str, tier: str, block_threads: int):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}: one of {sorted(BACKENDS)}")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}: one of {sorted(TIERS)}")
    if not 32 <= block_threads <= 1024 or block_threads % 32:
        raise ValueError(f"block_threads={block_threads}: a multiple of 32 in 32..1024")


def _block_threads(T: int) -> int:
    """The default block size: the next power of two >= T, in 32..256."""
    b = 32
    while b < T and b < 256:
        b *= 2
    return b


def _rot_operand(runtime_rot_joint: int, rot) -> tuple:
    if (runtime_rot_joint >= 0) != (rot is not None):
        raise ValueError("pass rot exactly when runtime_rot_joint >= 0")
    return (jnp.asarray(rot, jnp.float32).reshape(4),) if rot is not None else ()


def _prep(robot, init_trajs, lower, upper, start, goal):
    """Inputs as float32, with every trajectory's endpoints pinned to start / goal."""
    x = jnp.asarray(init_trajs, jnp.float32)
    lo = jnp.asarray(robot.lower if lower is None else lower, jnp.float32)
    hi = jnp.asarray(robot.upper if upper is None else upper, jnp.float32)
    s = jnp.asarray(x[0, 0] if start is None else start, jnp.float32)
    g = jnp.asarray(x[0, -1] if goal is None else goal, jnp.float32)
    x = x.at[:, 0].set(jnp.broadcast_to(s, x[:, 0].shape))
    x = x.at[:, -1].set(jnp.broadcast_to(g, x[:, -1].shape))
    return x, lo, hi, s, g


def sco_trajopt(robot: MotionRobot, init_trajs, *, collision: TrajCollision | None = None,
                world: World = World(), lower=None, upper=None, start=None, goal=None,
                n_outer_iters: int = 10, n_inner_iters: int = 30, m_lbfgs: int = 6,
                w_smooth: float = 1.0, w_acc: float = 0.5, w_jerk: float = 0.1,
                w_limits: float = 1.0, w_trust: float = 0.5, w_collision: float = 1.0,
                w_collision_max: float = 100.0, penalty_scale: float = 3.0,
                collision_margin: float = 0.01, smooth_min_temperature: float = 0.05,
                runtime_rot_joint: int = -1, rot=None, backend: str = "crocket-f32",
                tier: str = "block", block_threads: int | None = None):
    """Sequential convex optimization (``kernels/trajopt_crocket/sco_trajopt.cu``) ->
    ``(trajs, costs)``.

    Each outer iteration linearizes the smooth-min collision distances about the current
    trajectory and minimizes the convexified cost (smoothness, limits, linearized collision
    hinge, trust region) with L-BFGS; the collision weight grows by ``penalty_scale`` up to
    ``w_collision_max``. ``costs`` is the true (nonlinear) cost of the result.
    ``block_threads`` defaults to the next power of two >= T in 32..256.
    """
    if m_lbfgs > SCO_MAX_M:
        raise ValueError(f"m_lbfgs must be <= {SCO_MAX_M}")
    prob = _build.Problem(traj_collision=collision, world_counts=world.counts(),
                          runtime_rot_joint=runtime_rot_joint)
    x, lo, hi, s, g = _prep(robot, init_trajs, lower, upper, start, goal)
    B, T, n = x.shape
    bt = block_threads or _block_threads(T)
    _ck_check(backend, tier, bt)
    name = _ck_target("sco_trajopt", "CkScoTrajoptFfi", robot, prob, backend)
    size = (B * (6 * T * n + T * SCO_MAX_G + T * SCO_MAX_G * n + 2 * m_lbfgs * T * n)
            * (2 if backend.endswith("f64") else 1))
    trajs, costs, _ = jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, T, n), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.float32),
        jax.ShapeDtypeStruct((size,), jnp.float32)))(
        *_rot_operand(runtime_rot_joint, rot), x, *world.arrays(), lo, hi, s, g,
        n_outer_iters=np.int64(n_outer_iters), n_inner_iters=np.int64(n_inner_iters),
        m_lbfgs=np.int64(m_lbfgs), w_smooth=np.float32(w_smooth), w_acc=np.float32(w_acc),
        w_jerk=np.float32(w_jerk), w_limits=np.float32(w_limits), w_trust=np.float32(w_trust),
        w_collision=np.float32(w_collision), w_collision_max=np.float32(w_collision_max),
        penalty_scale=np.float32(penalty_scale), collision_margin=np.float32(collision_margin),
        smooth_min_temperature=np.float32(smooth_min_temperature),
        tier=np.int64(TIERS[tier]), block_threads=np.int64(bt))
    return trajs, costs


def stomp_trajopt(robot: MotionRobot, init_trajs, *, collision: TrajCollision | None = None,
                  world: World = World(), lower=None, upper=None, start=None, goal=None,
                  n_iters: int = 50, n_samples: int = 128, noise_scale: float = 0.05,
                  temperature: float = 0.1, step_size: float = 0.3, w_smooth: float = 1.0,
                  w_acc: float = 0.5, w_jerk: float = 0.1, w_limits: float = 1.0,
                  w_collision: float = 10.0, w_collision_max: float = 100.0,
                  collision_penalty_scale: float = 1.05, collision_margin: float = 0.01,
                  rng_seed: int = 0, runtime_rot_joint: int = -1, rot=None,
                  backend: str = "crocket-f32",
                  tier: str = "block", block_threads: int | None = None):
    """STOMP: sample smooth noise around each trajectory, softmax-weight the samples by cost
    and step toward their weighted mean (``kernels/trajopt_crocket/stomp_trajopt.cu``) ->
    ``(trajs, costs)``; the best sampled trajectory is returned, scored with
    ``w_collision_max``. At most 512 samples and 64 timesteps. ``backend`` / ``tier`` /
    ``block_threads`` as in :func:`sco_trajopt`; the tier is that of the sample evaluation, one
    sample per thread, warp or block."""
    prob = _build.Problem(traj_collision=collision, world_counts=world.counts(),
                          runtime_rot_joint=runtime_rot_joint)
    x, lo, hi, s, g = _prep(robot, init_trajs, lower, upper, start, goal)
    B, T, n = x.shape
    bt = block_threads or _block_threads(T)
    _ck_check(backend, tier, bt)
    name = _ck_target("stomp_trajopt", "CkStompTrajoptFfi", robot, prob, backend)
    size = B * (2 * T * n + 1 + 2 * n_samples) * (2 if backend.endswith("f64") else 1)
    trajs, costs, _ = jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, T, n), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.float32),
        jax.ShapeDtypeStruct((size,), jnp.float32)))(
        *_rot_operand(runtime_rot_joint, rot), x, *world.arrays(), lo, hi, s, g,
        n_iters=np.int64(n_iters), n_samples=np.int64(n_samples),
        noise_scale=np.float32(noise_scale), temperature=np.float32(temperature),
        step_size=np.float32(step_size), w_smooth=np.float32(w_smooth),
        w_acc=np.float32(w_acc), w_jerk=np.float32(w_jerk), w_limits=np.float32(w_limits),
        w_collision=np.float32(w_collision), w_collision_max=np.float32(w_collision_max),
        collision_penalty_scale=np.float32(collision_penalty_scale),
        collision_margin=np.float32(collision_margin), rng_seed=np.int64(rng_seed),
        tier=np.int64(TIERS[tier]), block_threads=np.int64(bt))
    return trajs, costs


def chomp_trajopt(robot: MotionRobot, init_trajs, *, collision: TrajCollision | None = None,
                  world: World = World(), lower=None, upper=None, start=None, goal=None,
                  n_iters: int = 100, step_size: float = 0.05, w_smooth: float = 1.0,
                  w_acc: float = 0.5, w_jerk: float = 0.1, w_limits: float = 1.0,
                  w_collision: float = 3.0, w_collision_max: float = 50.0,
                  collision_penalty_scale: float = 1.05, collision_margin: float = 0.01,
                  use_covariant_update: bool = True, smoothness_reg: float = 1e-3,
                  grad_clip_norm: float = 10.0, max_delta_per_step: float = 0.05,
                  early_stop_patience: int = 15, min_cost_improve: float = 1e-5,
                  runtime_rot_joint: int = -1, rot=None, backend: str = "crocket-f32",
                  tier: str = "block", block_threads: int = 64):
    """CHOMP: covariant (smoothness-metric) gradient descent with a line search
    (``kernels/trajopt_crocket/chomp_trajopt.cu``) -> ``(trajs, costs)``, scored with
    ``w_collision_max``. At most 64 timesteps."""
    prob = _build.Problem(traj_collision=collision, world_counts=world.counts(),
                          runtime_rot_joint=runtime_rot_joint)
    _ck_check(backend, tier, block_threads)
    name = _ck_target("chomp_trajopt", "CkChompTrajoptFfi", robot, prob, backend)
    x, lo, hi, s, g = _prep(robot, init_trajs, lower, upper, start, goal)
    B, T, n = x.shape
    size = B * (3 * T * n + 5 * T) * (2 if backend.endswith("f64") else 1)
    trajs, costs, _ = jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, T, n), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.float32),
        jax.ShapeDtypeStruct((size,), jnp.float32)))(
        *_rot_operand(runtime_rot_joint, rot), x, *world.arrays(), lo, hi, s, g,
        n_iters=np.int64(n_iters), step_size=np.float32(step_size),
        w_smooth=np.float32(w_smooth), w_acc=np.float32(w_acc), w_jerk=np.float32(w_jerk),
        w_limits=np.float32(w_limits), w_collision=np.float32(w_collision),
        w_collision_max=np.float32(w_collision_max),
        collision_penalty_scale=np.float32(collision_penalty_scale),
        collision_margin=np.float32(collision_margin),
        use_covariant_update=np.int64(use_covariant_update),
        smoothness_reg=np.float32(smoothness_reg), grad_clip_norm=np.float32(grad_clip_norm),
        max_delta_per_step=np.float32(max_delta_per_step),
        early_stop_patience=np.int64(early_stop_patience),
        min_cost_improve=np.float32(min_cost_improve),
        tier=np.int64(TIERS[tier]), block_threads=np.int64(block_threads))
    return trajs, costs
