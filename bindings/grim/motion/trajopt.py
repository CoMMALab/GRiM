"""Trajectory optimization on the GPU, built per robot.

Every optimizer takes initial trajectories ``(B, T, n_q)``, joint limits, endpoints (``(n_q,)``
shared or ``(B, n_q)`` per trajectory) and the world, and returns the optimized trajectories
with a cost per trajectory. Collision uses a :class:`~grim.motion._build.TrajCollision` model.
``traced=True`` builds against cricket's straight-line FK instead of the baked tables.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from ._build import TrajCollision
from .ik import World
from .robot import MotionRobot

SCO_MAX_G, SCO_MAX_M = 5, 8      # kernels/trajopt/sco_trajopt.cu


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
                fd_eps: float = 1e-4, traced: bool = False):
    """Sequential convex optimization (``kernels/trajopt/sco_trajopt.cu``) -> ``(trajs, costs)``.

    Each outer iteration linearizes the smooth-min collision distances about the current
    trajectory and minimizes the convexified cost (smoothness, limits, linearized collision
    hinge, trust region) with L-BFGS; the collision weight grows by ``penalty_scale`` up to
    ``w_collision_max``. ``costs`` is the true (nonlinear) cost of the result.
    """
    if m_lbfgs > SCO_MAX_M:
        raise ValueError(f"m_lbfgs must be <= {SCO_MAX_M}")
    prob = _build.Problem(traj_collision=collision, world_counts=world.counts())
    (name,) = _build.target("trajopt/sco_trajopt", ("ScoTrajoptFfi",), robot, prob,
                              traced)
    x, lo, hi, s, g = _prep(robot, init_trajs, lower, upper, start, goal)
    B, T, n = x.shape
    stride = 4 * T * n + T * SCO_MAX_G * n + 2 * m_lbfgs * T * n + 2 * SCO_MAX_M \
        + T * (robot.n_joints + 1) * 7
    trajs, costs, _ = jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, T, n), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.float32),
        jax.ShapeDtypeStruct((B * stride,), jnp.float32)))(
        x, *world.arrays(), lo, hi, s, g,
        n_outer_iters=np.int64(n_outer_iters), n_inner_iters=np.int64(n_inner_iters),
        m_lbfgs=np.int64(m_lbfgs), w_smooth=np.float32(w_smooth), w_acc=np.float32(w_acc),
        w_jerk=np.float32(w_jerk), w_limits=np.float32(w_limits), w_trust=np.float32(w_trust),
        w_collision=np.float32(w_collision), w_collision_max=np.float32(w_collision_max),
        penalty_scale=np.float32(penalty_scale), collision_margin=np.float32(collision_margin),
        smooth_min_temperature=np.float32(smooth_min_temperature), fd_eps=np.float32(fd_eps))
    return trajs, costs


def stomp_trajopt(robot: MotionRobot, init_trajs, *, collision: TrajCollision | None = None,
                  world: World = World(), lower=None, upper=None, start=None, goal=None,
                  n_iters: int = 50, n_samples: int = 128, noise_scale: float = 0.05,
                  temperature: float = 0.1, step_size: float = 0.3, w_smooth: float = 1.0,
                  w_acc: float = 0.5, w_jerk: float = 0.1, w_limits: float = 1.0,
                  w_collision: float = 10.0, w_collision_max: float = 100.0,
                  collision_penalty_scale: float = 1.05, collision_margin: float = 0.01,
                  rng_seed: int = 0, traced: bool = False):
    """STOMP: sample smooth noise around each trajectory, softmax-weight the samples by cost
    and step toward their weighted mean (``kernels/trajopt/stomp_trajopt.cu``) ->
    ``(trajs, costs)``; the best sampled trajectory is returned, scored with
    ``w_collision_max``. At most 512 samples and 64 timesteps."""
    prob = _build.Problem(traj_collision=collision, world_counts=world.counts())
    (name,) = _build.target("trajopt/stomp_trajopt", ("StompTrajoptFfi",), robot, prob,
                              traced)
    x, lo, hi, s, g = _prep(robot, init_trajs, lower, upper, start, goal)
    B, T, n = x.shape
    trajs, costs, _ = jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, T, n), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.float32),
        jax.ShapeDtypeStruct((B * T * n + B + 2 * B * n_samples,), jnp.float32)))(
        x, *world.arrays(), lo, hi, s, g,
        n_iters=np.int64(n_iters), n_samples=np.int64(n_samples),
        noise_scale=np.float32(noise_scale), temperature=np.float32(temperature),
        step_size=np.float32(step_size), w_smooth=np.float32(w_smooth), w_acc=np.float32(w_acc),
        w_jerk=np.float32(w_jerk), w_limits=np.float32(w_limits),
        w_collision=np.float32(w_collision), w_collision_max=np.float32(w_collision_max),
        collision_penalty_scale=np.float32(collision_penalty_scale),
        collision_margin=np.float32(collision_margin), rng_seed=np.int64(rng_seed))
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
                  fd_eps: float = 1e-4, traced: bool = False):
    """CHOMP: covariant (smoothness-metric) gradient descent with a line search
    (``kernels/trajopt/chomp_trajopt.cu``) -> ``(trajs, costs)``, scored with
    ``w_collision_max``. At most 64 timesteps."""
    prob = _build.Problem(traj_collision=collision, world_counts=world.counts())
    (name,) = _build.target("trajopt/chomp_trajopt", ("ChompTrajoptFfi",), robot, prob,
                              traced)
    x, lo, hi, s, g = _prep(robot, init_trajs, lower, upper, start, goal)
    B, T, n = x.shape
    trajs, costs, _ = jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, T, n), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.float32),
        jax.ShapeDtypeStruct((B * 3 * T * n,), jnp.float32)))(
        x, *world.arrays(), lo, hi, s, g,
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
        min_cost_improve=np.float32(min_cost_improve), fd_eps=np.float32(fd_eps))
    return trajs, costs


LST_G = 5      # kernels/trajopt/ls_trajopt.cu


def ls_trajopt(robot: MotionRobot, init_trajs, *, collision: TrajCollision | None = None,
               world: World = World(), lower=None, upper=None, start=None, goal=None,
               n_outer_iters: int = 8, n_ls_iters: int = 8, lambda_init: float = 5e-3,
               w_smooth: float = 1.0, w_acc: float = 0.6, w_jerk: float = 0.2,
               w_limits: float = 1.0, w_trust: float = 0.5, w_endpoint: float = 100.0,
               w_collision: float = 1.0, w_collision_max: float = 100.0,
               penalty_scale: float = 3.0, collision_margin: float = 0.01,
               smooth_min_temperature: float = 0.05, max_delta_per_step: float = 0.1,
               fd_eps: float = 1e-4, traced: bool = False):
    """Least-squares trajopt: each outer step linearizes the smooth-min collision groups and
    runs diagonal Gauss-Newton / LM on the stacked residuals, one thread per trajectory
    (``kernels/trajopt/ls_trajopt.cu``) -> ``(trajs, costs)``, scored with
    ``w_collision_max``."""
    prob = _build.Problem(traj_collision=collision, world_counts=world.counts())
    (name,) = _build.target("trajopt/ls_trajopt", ("LsTrajoptFfi",), robot, prob,
                              traced)
    x, lo, hi, s, g = _prep(robot, init_trajs, lower, upper, start, goal)
    B, T, n = x.shape
    m = (5 * T - 3) * n + T * LST_G
    stride = T * n + T * LST_G + T * LST_G * n + 2 * m + 2 * T * n + T * (robot.n_joints + 1) * 7
    trajs, costs, _ = jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((B, T, n), jnp.float32), jax.ShapeDtypeStruct((B,), jnp.float32),
        jax.ShapeDtypeStruct((B * stride,), jnp.float32)))(
        x, *world.arrays(), lo, hi, s, g,
        n_outer_iters=np.int64(n_outer_iters), n_ls_iters=np.int64(n_ls_iters),
        lambda_init=np.float32(lambda_init), w_smooth=np.float32(w_smooth),
        w_acc=np.float32(w_acc), w_jerk=np.float32(w_jerk), w_limits=np.float32(w_limits),
        w_trust=np.float32(w_trust), w_endpoint=np.float32(w_endpoint),
        w_collision=np.float32(w_collision), w_collision_max=np.float32(w_collision_max),
        penalty_scale=np.float32(penalty_scale), collision_margin=np.float32(collision_margin),
        smooth_min_temperature=np.float32(smooth_min_temperature),
        max_delta_per_step=np.float32(max_delta_per_step), fd_eps=np.float32(fd_eps))
    return trajs, costs
