"""Region IK: configurations whose end-effector lands inside a box (or matches a pose
distribution), built per robot.

All three solvers target ONE end-effector joint frame and solve every actuated joint. Each
returns ``(q, err, ee_points, target_points)``: per sample the configuration, its error, the
end-effector position reached and the target point it was driven toward.

``brownian_ik`` and ``hit_and_run_ik`` are warp-per-sample (the warp is the unit of work,
so there are no tiers) and are built with ``--use_fast_math`` like their originals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from .robot import MotionRobot

_FAST_MATH = ("--use_fast_math",)


def _limits(robot, lower, upper, fixed_mask):
    return (jnp.asarray(robot.lower if lower is None else lower, jnp.float32),
            jnp.asarray(robot.upper if upper is None else upper, jnp.float32),
            jnp.asarray(np.zeros(robot.n_act) if fixed_mask is None else fixed_mask, jnp.int32))


def _outs(P, S, n_q):
    return (jax.ShapeDtypeStruct((P, S, n_q), jnp.float32),
            jax.ShapeDtypeStruct((P, S), jnp.float32),
            jax.ShapeDtypeStruct((P, S, 3), jnp.float32),
            jax.ShapeDtypeStruct((P, S, 3), jnp.float32))


def _target(kernel, symbol, robot, ee_joint, flags=()):
    problem = _build.Problem(ee_joints=(int(ee_joint),))
    return _build.target(kernel, (symbol,), robot, problem, extra_flags=flags)[0]


def brownian_ik(robot: MotionRobot, seeds, init_points, target_quat, box_mins, box_maxs,
                ee_joint: int, *, rng_seed: int = 0, lower=None, upper=None, fixed_mask=None,
                max_iter: int = 20, pos_weight: float = 50.0, ori_weight: float = 0.0,
                lambda_init: float = 5e-3, eps_pos: float = 1e-4, noise_std: float = 0.02,
                n_brownian_steps: int = 100, fk_check_freq: int = 5,
                threads_per_block: int = 128):
    """Brownian walk in the null space of the end-effector box constraint
    (``kernels/region_ik/brownian_ik.cu``). ``init_points`` (P, S, 3) start points,
    ``target_quat`` (4,) the orientation to hold, ``box_mins``/``box_maxs`` (P, 3)."""
    name = _target("region_ik/brownian_ik", "BrownianMotionIkFfi", robot, ee_joint, _FAST_MATH)
    seeds = jnp.asarray(seeds, jnp.float32)
    P, S, _ = seeds.shape
    return jax.ffi.ffi_call(name, _outs(P, S, robot.n_act))(
        seeds, jnp.asarray(init_points, jnp.float32), jnp.asarray(target_quat, jnp.float32),
        jnp.asarray(box_mins, jnp.float32), jnp.asarray(box_maxs, jnp.float32),
        *_limits(robot, lower, upper, fixed_mask), jnp.asarray([rng_seed], jnp.int32),
        max_iter=np.int64(max_iter), pos_weight=np.float32(pos_weight),
        ori_weight=np.float32(ori_weight), lambda_init=np.float32(lambda_init),
        eps_pos=np.float32(eps_pos), noise_std=np.float32(noise_std),
        n_brownian_steps=np.int64(n_brownian_steps), fk_check_freq=np.int64(fk_check_freq),
        threads_per_block=np.int64(threads_per_block))


def hit_and_run_ik(robot: MotionRobot, seeds, box_mins, box_maxs, ee_joint: int, *,
                   rng_seed: int = 0, lower=None, upper=None, fixed_mask=None,
                   max_iter: int = 20, n_iterations: int = 100, pos_weight: float = 50.0,
                   ori_weight: float = 0.0, lambda_init: float = 5e-3, eps_pos: float = 1e-4,
                   eps_ori: float = 1e-4, noise_std: float = 0.02):
    """Hit-and-run sampling of box targets, each solved by Gauss-Newton
    (``kernels/region_ik/hit_and_run_ik.cu``)."""
    name = _target("region_ik/hit_and_run_ik", "HitAndRunIkFfi", robot, ee_joint, _FAST_MATH)
    seeds = jnp.asarray(seeds, jnp.float32)
    P, S, _ = seeds.shape
    return jax.ffi.ffi_call(name, _outs(P, S, robot.n_act))(
        seeds, jnp.asarray(box_mins, jnp.float32), jnp.asarray(box_maxs, jnp.float32),
        *_limits(robot, lower, upper, fixed_mask), jnp.asarray([rng_seed], jnp.int32),
        max_iter=np.int64(max_iter), n_iterations=np.int64(n_iterations),
        pos_weight=np.float32(pos_weight), ori_weight=np.float32(ori_weight),
        lambda_init=np.float32(lambda_init), eps_pos=np.float32(eps_pos),
        eps_ori=np.float32(eps_ori), noise_std=np.float32(noise_std))


def svgd_ik(robot: MotionRobot, seeds, targets, ee_joint: int, *, lower=None, upper=None,
            fixed_mask=None, n_iters: int = 50, bandwidth: float = 0.1,
            step_size: float = 0.05):
    """Stein variational gradient descent over at most 32 particles per problem
    (``kernels/region_ik/svgd_ik.cu``); ``targets`` (P, 7)."""
    name = _target("region_ik/svgd_ik", "SvgdRegionIkFfi", robot, ee_joint)
    seeds = jnp.asarray(seeds, jnp.float32)
    P, S, _ = seeds.shape
    return jax.ffi.ffi_call(name, _outs(P, S, robot.n_act))(
        seeds, jnp.asarray(targets, jnp.float32).reshape(P, 1, 7),
        *_limits(robot, lower, upper, fixed_mask), n_iters=np.int64(n_iters),
        bandwidth=np.float32(bandwidth), step_size=np.float32(step_size))
