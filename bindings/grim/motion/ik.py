"""Multi-seed IK solvers on the GPU, built per robot.

Every solver takes plain arrays: seeds ``(P, S, n_q)`` in ``robot.actuated_names`` order and
targets ``(P, n_ee, 7)`` ``[qw, qx, qy, qz, x, y, z]`` for the joint frames ``ee_joints``, and
returns at least the best configuration and weighted squared error per seed.

A solve is two steps, which callers working under ``jax.jit`` use separately:

* :func:`build` compiles (or loads from cache) the kernel for the problem's STRUCTURE --
  robot, end-effector frames, collision model, obstacle counts -- and returns a hashable
  :class:`Target`. It needs concrete arrays, so it runs outside any trace.
* :func:`run` launches a target on the per-call VALUES -- seeds, targets, obstacle poses,
  limits, solver settings -- and is safe inside ``jax.jit``.

The named functions (:func:`ls_ik`, :func:`sqp_ik`, ...) do both.

``tier`` picks how many threads own one seed: ``"thread"`` (one; best at large batch),
``"warp"`` or ``"block"`` (cooperative; best at small batch / high DOF). ``block_threads``
is the block tier's block size.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from ._build import SelfCollision
from .robot import MotionRobot

TIERS = {"thread": 0, "warp": 1, "block": 2}


@dataclass(frozen=True)
class World:
    """World obstacles; each kind is an (M, width) array.

    spheres ``[x, y, z, r]``; capsules ``[ax, ay, az, bx, by, bz, r]``; boxes ``[center(3),
    axis_x(3), axis_y(3), axis_z(3), half_extents(3)]``; halfspaces ``[normal(3), point(3)]``.
    """
    spheres: np.ndarray | None = None
    capsules: np.ndarray | None = None
    boxes: np.ndarray | None = None
    halfspaces: np.ndarray | None = None

    def arrays(self) -> tuple[jax.Array, ...]:
        return tuple(jnp.zeros((0, _build.WORLD_WIDTH[k]), jnp.float32) if a is None
                     else jnp.asarray(a, jnp.float32).reshape(-1, _build.WORLD_WIDTH[k])
                     for k, a in zip(_build.WORLD_KINDS,
                                     (self.spheres, self.capsules, self.boxes, self.halfspaces)))

    def counts(self) -> tuple[int, int, int, int]:
        return tuple(int(a.shape[0]) for a in self.arrays())


@dataclass(frozen=True)
class CollisionModel:
    """The robot's collision geometry: spheres for world collision, in their joint frames,
    and (optionally) the self-collision model."""
    robot_spheres: np.ndarray | None = None        # (S, 4) xyz + radius, joint frame
    robot_sphere_joint: np.ndarray | None = None   # (S,)
    self_collision: SelfCollision | None = None


# kernel -> (source, FFI symbol, tiered, solves every joint)
KERNELS = {
    "ls": ("ik/ls_ik", "LsIkFfi", True, False),
    "sqp": ("ik/sqp_ik", "SqpIkFfi", True, False),
    "hjcd_coarse": ("ik/hjcd_ik", "HjcdIkCoarseFfi", False, False),
    "hjcd_lm": ("ik/hjcd_ik", "HjcdIkLmFfi", True, False),
    # Off-chain joints still receive noise, so a chain-only build is not equivalent.
    "mppi": ("ik/mppi_ik", "MppiIkFfi", False, True),
    "canonical": ("ik/canonical_ik", "CanonicalIkFfi", False, True),
}


@dataclass(frozen=True)
class Target:
    """A built IK kernel. Hashable, so it can be a static argument of a jitted function."""
    kernel: str
    name: str                       # registered FFI target
    n_q: int
    n_ee: int
    world_counts: tuple[int, int, int, int]
    has_robot_spheres: bool
    lower: tuple[float, ...]
    upper: tuple[float, ...]


def problem(robot: MotionRobot, ee_joints, collision: CollisionModel | None = None,
            world_counts=(0, 0, 0, 0), chain_only: bool = True) -> _build.Problem:
    c = collision or CollisionModel()
    has_spheres = c.robot_spheres is not None and len(c.robot_spheres) > 0
    sc = c.self_collision
    return _build.Problem(
        ee_joints=tuple(int(e) for e in ee_joints),
        robot_spheres=np.asarray(c.robot_spheres, np.float32) if has_spheres else None,
        robot_sphere_joint=np.asarray(c.robot_sphere_joint, np.int32) if has_spheres else None,
        self_collision=sc if sc is not None and len(sc.pair_i) > 0 else None,
        world_counts=tuple(int(n) for n in world_counts), chain_only=chain_only)


def build(kernel: str, robot: MotionRobot, ee_joints, *, collision: CollisionModel | None = None,
          world_counts=(0, 0, 0, 0), chain_only: bool = True, traced: bool = False) -> Target:
    """Compile (or load) ``kernel`` for this robot and problem structure."""
    source, symbol, _, all_joints = KERNELS[kernel]
    prob = problem(robot, ee_joints, collision, world_counts, chain_only and not all_joints)
    (name,) = _build.target(source, (symbol,), robot, prob, traced)
    return Target(kernel, name, robot.n_act, len(prob.ee_joints), prob.world_counts,
                  prob.robot_spheres is not None, tuple(map(float, robot.lower)),
                  tuple(map(float, robot.upper)))


def run(target: Target, seeds, targets, *, world: World = World(), lower=None, upper=None,
        fixed_mask=None, enable_collision: bool | None = None, tier: str = "thread",
        block_threads: int = 64, noise=None, rng_seed: int = 0, **settings):
    """Launch a built IK kernel. ``settings`` are the kernel's solver settings (see the named
    functions); ``noise`` is hjcd_lm's stall-kick noise and ``rng_seed`` mppi's seed."""
    _, _, tiered, _ = KERNELS[target.kernel]
    if world.counts() != target.world_counts:
        raise ValueError(f"world has obstacle counts {world.counts()}; the target was built "
                         f"for {target.world_counts}")
    seeds = jnp.asarray(seeds, jnp.float32)
    P, S, _ = seeds.shape
    lo = jnp.asarray(target.lower if lower is None else lower, jnp.float32)
    hi = jnp.asarray(target.upper if upper is None else upper, jnp.float32)
    fm = jnp.asarray(np.zeros(target.n_q) if fixed_mask is None else fixed_mask, jnp.int32)
    if enable_collision is None:
        enable_collision = target.has_robot_spheres and sum(target.world_counts) > 0
    outs = [jax.ShapeDtypeStruct((P, S, target.n_q), jnp.float32),
            jax.ShapeDtypeStruct((P, S), jnp.float32)]
    after_seeds, after_limits = [], []
    if target.kernel == "sqp":
        outs.append(jax.ShapeDtypeStruct((P, S), jnp.int32))
    elif target.kernel == "hjcd_lm":
        after_seeds.append(jnp.asarray(noise, jnp.float32))
        outs.append(jax.ShapeDtypeStruct((P,), jnp.int32))
    elif target.kernel == "mppi":
        after_limits.append(jnp.asarray([rng_seed], jnp.int32))
    attrs = {k: (np.int64(v) if isinstance(v, (bool, int, np.integer)) else np.float32(v))
             for k, v in settings.items()}
    if tiered:
        attrs.update(tier=np.int64(TIERS[tier]), block_threads=np.int64(block_threads))
    return jax.ffi.ffi_call(target.name, tuple(outs))(
        seeds, *after_seeds, jnp.asarray(targets, jnp.float32).reshape(P, target.n_ee, 7),
        *world.arrays(), lo, hi, fm, *after_limits, **attrs,
        enable_collision=np.int64(bool(enable_collision)))


def _solve(kernel, robot, seeds, targets, ee_joints, *, world, collision, chain_only, traced,
           **kw):
    t = build(kernel, robot, ee_joints, collision=collision, world_counts=world.counts(),
              chain_only=chain_only, traced=traced)
    return run(t, seeds, targets, world=world, **kw)


def ls_ik(robot: MotionRobot, seeds, targets, ee_joints, *, lower=None, upper=None,
          fixed_mask=None, world: World = World(), collision: CollisionModel | None = None,
          max_iter: int = 60, pos_weight: float = 50.0, ori_weight: float = 10.0,
          lambda_init: float = 5e-3, eps_pos: float = 1e-4, eps_ori: float = 1e-3,
          collision_weight: float = 1e4, collision_margin: float = 0.02,
          tier: str = "thread", block_threads: int = 64, traced: bool = False,
          chain_only: bool = True):
    """Levenberg-Marquardt from every seed (``kernels/ik/ls_ik.cu``) -> ``(q, err)``.

    World collision applies when ``collision`` has robot spheres and ``world`` obstacles;
    self-collision when ``collision.self_collision`` has pairs. Both enter the step (as
    Gauss-Newton rows) and the merit, weighted by ``collision_weight``.
    """
    return _solve("ls", robot, seeds, targets, ee_joints, world=world, collision=collision,
                  chain_only=chain_only, traced=traced, lower=lower, upper=upper,
                  fixed_mask=fixed_mask, tier=tier, block_threads=block_threads,
                  max_iter=max_iter, pos_weight=pos_weight, ori_weight=ori_weight,
                  lambda_init=lambda_init, eps_pos=eps_pos, eps_ori=eps_ori,
                  collision_weight=collision_weight, collision_margin=collision_margin)


def sqp_ik(robot: MotionRobot, seeds, targets, ee_joints, *, lower=None, upper=None,
           fixed_mask=None, world: World = World(), collision: CollisionModel | None = None,
           max_iter: int = 60, n_inner_iters: int = 10, pos_weight: float = 50.0,
           ori_weight: float = 10.0, lambda_init: float = 5e-3, eps_pos: float = 1e-4,
           eps_ori: float = 1e-3, collision_weight: float = 1e4,
           collision_margin: float = 0.02, tier: str = "thread", block_threads: int = 64,
           traced: bool = False, chain_only: bool = True):
    """SQP with joint limits and collision as hard constraints (``kernels/ik/sqp_ik.cu``)
    -> ``(q, err, feasible)``; ``feasible`` is 1 where the returned configuration satisfies
    the collision constraint (always 1 without collision)."""
    return _solve("sqp", robot, seeds, targets, ee_joints, world=world, collision=collision,
                  chain_only=chain_only, traced=traced, lower=lower, upper=upper,
                  fixed_mask=fixed_mask, tier=tier, block_threads=block_threads,
                  max_iter=max_iter, n_inner_iters=n_inner_iters, pos_weight=pos_weight,
                  ori_weight=ori_weight, lambda_init=lambda_init, eps_pos=eps_pos,
                  eps_ori=eps_ori, collision_weight=collision_weight,
                  collision_margin=collision_margin)


def hjcd_ik_coarse(robot: MotionRobot, seeds, targets, ee_joints, *, lower=None, upper=None,
                   fixed_mask=None, world: World = World(),
                   collision: CollisionModel | None = None, k_max: int = 20,
                   collision_weight: float = 1e4, collision_margin: float = 0.02,
                   traced: bool = False, chain_only: bool = True):
    """HJCD phase 1: greedy coordinate descent, one thread per seed
    (``kernels/ik/hjcd_ik.cu``) -> ``(q, err)``."""
    return _solve("hjcd_coarse", robot, seeds, targets, ee_joints, world=world,
                  collision=collision, chain_only=chain_only, traced=traced, lower=lower,
                  upper=upper, fixed_mask=fixed_mask, k_max=k_max,
                  collision_weight=collision_weight, collision_margin=collision_margin)


def hjcd_ik_lm(robot: MotionRobot, seeds, targets, ee_joints, noise, *, lower=None,
               upper=None, fixed_mask=None, world: World = World(),
               collision: CollisionModel | None = None, max_iter: int = 60,
               stall_patience: int = 6, lambda_init: float = 5e-3,
               limit_prior_weight: float = 1e-4, kick_scale: float = 0.02,
               eps_pos: float = 1e-4, eps_ori: float = 1e-3, early_stop: bool = False,
               collision_weight: float = 1e4, collision_margin: float = 0.02,
               tier: str = "thread", block_threads: int = 64, traced: bool = False,
               chain_only: bool = True):
    """HJCD phase 2: Levenberg-Marquardt with a joint-limit prior and stall kicks
    (``kernels/ik/hjcd_ik.cu``) -> ``(q, err, stopped)``.

    ``noise`` (P, S, max_iter, n_q) are the stall kicks, drawn by the caller so the solve is
    deterministic. ``early_stop`` stops a problem's seeds once one has converged: faster,
    but then only each problem's best seed is meaningful (the rest depend on scheduling).
    """
    return _solve("hjcd_lm", robot, seeds, targets, ee_joints, world=world, collision=collision,
                  chain_only=chain_only, traced=traced, lower=lower, upper=upper,
                  fixed_mask=fixed_mask, tier=tier, block_threads=block_threads, noise=noise,
                  max_iter=max_iter, stall_patience=stall_patience, early_stop=early_stop,
                  lambda_init=lambda_init, limit_prior_weight=limit_prior_weight,
                  kick_scale=kick_scale, eps_pos=eps_pos, eps_ori=eps_ori,
                  collision_weight=collision_weight, collision_margin=collision_margin)


def mppi_ik(robot: MotionRobot, seeds, targets, ee_joints, *, rng_seed: int = 0, lower=None,
            upper=None, fixed_mask=None, world: World = World(),
            collision: CollisionModel | None = None, n_particles: int = 16,
            n_mppi_iters: int = 4, n_lbfgs_iters: int = 30, m_lbfgs: int = 5,
            sigma: float = 0.1, mppi_temperature: float = 0.05, pos_weight: float = 50.0,
            ori_weight: float = 10.0, eps_pos: float = 1e-4, eps_ori: float = 1e-3,
            collision_weight: float = 1e4, collision_margin: float = 0.02,
            traced: bool = False):
    """MPPI particle search then L-BFGS, one thread per seed (``kernels/ik/mppi_ik.cu``)
    -> ``(q, err)``. The noise stream is keyed on (rng_seed, problem, seed slot), so a
    solve is deterministic but a seed's result depends on its slot."""
    return _solve("mppi", robot, seeds, targets, ee_joints, world=world, collision=collision,
                  chain_only=False, traced=traced, lower=lower, upper=upper,
                  fixed_mask=fixed_mask, rng_seed=rng_seed, n_particles=n_particles,
                  n_mppi_iters=n_mppi_iters, n_lbfgs_iters=n_lbfgs_iters, m_lbfgs=m_lbfgs,
                  sigma=sigma, mppi_temperature=mppi_temperature, pos_weight=pos_weight,
                  ori_weight=ori_weight, eps_pos=eps_pos, eps_ori=eps_ori,
                  collision_weight=collision_weight, collision_margin=collision_margin)


def build_canonical(robot: MotionRobot, ee_joints, *, collision: CollisionModel | None = None,
                    world_counts=(0, 0, 0, 0)) -> Target:
    return build("canonical", robot, ee_joints, collision=collision, world_counts=world_counts)


def run_canonical(target: Target, q, q_ref, targets, *, world: World = World(),
                  max_iters: int = 200, step: float = 0.1, tol: float = 1e-5,
                  damping: float = 1e-6, collision_margin: float = 0.0):
    q = jnp.asarray(q, jnp.float32)
    P = q.shape[0]
    return jax.ffi.ffi_call(target.name, (jax.ShapeDtypeStruct((P, target.n_q), jnp.float32),
                                          jax.ShapeDtypeStruct((P,), jnp.int32)))(
        q, jnp.asarray(q_ref, jnp.float32),
        jnp.asarray(targets, jnp.float32).reshape(P, target.n_ee, 7), *world.arrays(),
        max_iters=np.int64(max_iters), step=np.float32(step), tol=np.float32(tol),
        damping=np.float32(damping), collision_margin=np.float32(collision_margin))


def canonical_ik(robot: MotionRobot, q, q_ref, targets, ee_joints, *, world: World = World(),
                 collision: CollisionModel | None = None, **kw):
    """Slide IK solutions ``q`` (P, n_q) along their self-motion manifolds to the point
    nearest ``q_ref`` (``kernels/ik/canonical_ik.cu``) -> ``(q*, iterations)``.

    Solves ``argmin 1/2 |q - q_ref|^2 s.t. r(q) = 0`` by damped Gauss-Newton, which pins a
    redundant arm's solution to one point so its derivative is well defined. With collision
    the step never lowers the clearance the input had. ``kw``: max_iters, step, tol,
    damping, collision_margin.
    """
    t = build_canonical(robot, ee_joints, collision=collision, world_counts=world.counts())
    return run_canonical(t, q, q_ref, targets, world=world, **kw)
