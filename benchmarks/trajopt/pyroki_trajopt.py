"""pyroki's trajectory optimization (external/pyroki at the pinned commit,
examples/pyroki_snippets/_trajopt.py ``solve_trajopt``), from given start and goal
configurations: the example's step 1 (IK for the endpoint poses) is dropped, steps 2-4 are its
lines unchanged except that the safety margin is ``MARGIN`` (see there), the start / end
constraints take their configuration as a cost argument instead of closing over it, and the world-collision residual is defined once at module
level (the example's per-call functions make every problem a new jit trace, so timing would
measure compilation: ~20 s of the example's ~23 s per call).

pyroki has no cylinder primitive: ``Cylinder`` and its capsule distance are added here through
pyroki's own registry (``collision._collision.COLLISION_FUNCTIONS``), with the same two-step
closest-point refinement pyroki's ``capsule_box`` uses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from pyroki.collision import _collision, _utils
from pyroki.collision._geometry import Capsule, CollGeom


@jdc.pytree_dataclass
class Cylinder(CollGeom):
    """Solid cylinder along its local z: size = [radius, half-height]."""

    def _create_one_mesh(self, index):  # pragma: no cover - visualization only
        raise NotImplementedError


def _point_to_cylinder_sdf(p, radius, half):
    rho = jnp.linalg.norm(p[..., :2] + 1e-12, axis=-1)
    q = jnp.stack([rho - radius, jnp.abs(p[..., 2]) - half], axis=-1)
    return jnp.linalg.norm(jnp.maximum(q, 0.0) + 1e-12, axis=-1) * (q.max(-1) > 0) + jnp.minimum(q.max(-1), 0.0)


def capsule_cylinder(capsule: Capsule, cyl: Cylinder):
    """As pyroki's capsule_box: refine the segment point twice, then the solid's SDF."""
    off = capsule.axis * capsule.height[..., None] / 2
    pos = capsule.pose.translation()
    p1 = cyl.pose.inverse().apply(pos + off)
    p2 = cyl.pose.inverse().apply(pos - off)
    radius, half = cyl.size[..., 0], cyl.size[..., 1]
    pt = _utils.closest_segment_point(p1, p2, jnp.zeros_like(p1))
    rho = jnp.linalg.norm(pt[..., :2] + 1e-12, axis=-1, keepdims=True)
    clamped = jnp.concatenate([pt[..., :2] * jnp.minimum(1.0, radius[..., None] / rho),
                               jnp.clip(pt[..., 2:], -half[..., None], half[..., None])], axis=-1)
    pt = _utils.closest_segment_point(p1, p2, clamped)
    return _point_to_cylinder_sdf(pt, radius, half) - capsule.radius


_collision.COLLISION_FUNCTIONS[(Capsule, Cylinder)] = capsule_cylinder


def sphere_cylinder(sphere: pk.collision.Sphere, cyl: Cylinder):
    """The solid cylinder's SDF at the sphere centre, minus its radius (the validity check's spheres)."""
    p = cyl.pose.inverse().apply(sphere.pose.translation())
    return _point_to_cylinder_sdf(p, cyl.size[..., 0], cyl.size[..., 1]) - sphere.radius


_collision.COLLISION_FUNCTIONS[(pk.collision.Sphere, Cylinder)] = sphere_cylinder


def _wxyz(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()[..., [3, 0, 1, 2]]


def world_geoms(p) -> list[CollGeom]:
    """The problem's obstacles as pyroki geometry: one batched Box, one batched Cylinder."""
    out = []
    if len(p["boxes"]):
        b = onp.asarray(p["boxes"])
        R = b[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1)   # columns: the box axes
        out.append(pk.collision.Box.from_extent(2 * b[:, 12:15], b[:, :3], _wxyz(R)))
    if len(p["cylinders"]):
        c = onp.asarray(p["cylinders"])
        z = onp.array([0.0, 0.0, 1.0])
        from scipy.spatial.transform import Rotation
        R = onp.stack([Rotation.align_vectors(a[None], z[None])[0].as_matrix() for a in c[:, 3:6]])
        pose = jaxlie.SE3(jnp.asarray(onp.concatenate([_wxyz(R), c[:, :3]], axis=1)))
        out.append(Cylinder(pose=pose, size=jnp.asarray(c[:, 6:8])))
    return out


# The example's 0.05 m safety margin, here the shared mbm.MARGIN (0.01 m): MBM's goals are grasps closer than 5 cm to
# obstacles, where a 5 cm clearance constraint and the goal constraint conflict and the solver
# trades the goal away (goals then miss by 3e-3 to 2e-2 rad, so nothing meets the 1e-3 rad
# endpoint test, even problems the straight seed already solves). 1 cm is GRiM's default margin
# and TrajOpt's default margin buffer (0 margin + 1 cm); cuRobo activates at 0.25 cm, Drake at 0.
MARGIN = __import__("mbm").MARGIN


@jaxls.Cost.factory(kind="constraint_eq_zero", name="start_pose_constraint")
def start_pose_constraint(vals: jaxls.VarValues, var, cfg) -> jax.Array:
    return (vals[var] - cfg).flatten()


@jaxls.Cost.factory(kind="constraint_eq_zero", name="end_pose_constraint")
def end_pose_constraint(vals: jaxls.VarValues, var, cfg) -> jax.Array:
    return (vals[var] - cfg).flatten()


def compute_world_coll_residual(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll_obj: pk.collision.CollGeom,
    prev_traj_vars: jaxls.Var[jax.Array],
    curr_traj_vars: jaxls.Var[jax.Array],
):
    """The example's nested function, unchanged, at module level (it closes over nothing)."""
    coll = robot_coll.get_swept_capsules(
        robot, vals[prev_traj_vars], vals[curr_traj_vars]
    )
    dist = pk.collision.collide(
        coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1))
    )  # >0 means no collision
    return dist.flatten() - MARGIN  # safety margin


def solve(robot, robot_coll, world_coll, start_cfg, end_cfg, timesteps, dt, host=True, max_iters=None,
          w_smooth=1.0, w_self=5.0, init_traj=None, return_cost=False, verbose=True, linear_solver="conjugate_gradient",
          snap_every=0):
    """examples/pyroki_snippets/_trajopt.py solve_trajopt, steps 2-4. ``host=False`` returns the
    device array, so the whole solve can sit under ``jax.jit`` (compiled once per shape).
    ``init_traj``: the start (default: the example's straight line); ``return_cost``: also return the
    final cost (squared residual norm), to rank starts by.
    ``snap_every`` > 0 (debug; needs external_patches/jaxls_snapshots.py applied): also the solution
    and its cost after every k-th outer step, ``(traj, cost, snap_trajs (S, T, n), snap_costs (S,))``.
    ``w_smooth`` / ``w_self``: the example's smoothness and self-collision weights (its values by default)."""
    start_cfg, end_cfg = jnp.asarray(start_cfg), jnp.asarray(end_cfg)
    # 2. Initialize the trajectory through linearly interpolating the start and end poses.
    if init_traj is None:
        init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # 3. Optimize the trajectory.
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)  # Add batch dimension.

    # --- Soft costs ---
    costs: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([w_smooth])[None],
        ),
        pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
            robot.joint_var_cls(jnp.arange(4, timesteps)),
            robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
            dt,
            jnp.array([0.1])[None],
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            traj_vars,
            0.02,
            w_self,
        ),
    ]

    # --- Constraints (augmented Lagrangian penalties) ---
    # Joint limits.
    costs.append(pk.costs.limit_constraint(robot, traj_vars))

    # Start / end pose constraints.
    costs.append(start_pose_constraint(robot.joint_var_cls(jnp.arange(0, 2)), start_cfg[None]))
    costs.append(end_pose_constraint(robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)), end_cfg[None]))

    # Velocity limits.
    costs.append(
        pk.costs.limit_velocity_constraint(
            robot,
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            dt,
        )
    )

    # World collision avoidance using swept volumes (compute_world_coll_residual, module level).
    for world_coll_obj in world_coll:
        costs.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot,
                    robot_coll,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                kind="constraint_geq_zero",
                name="world Collision (sweep)",
            )
        )

    # 4. Solve the optimization problem with augmented Lagrangian for constraints.
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[traj_vars]).analyze()
    kw = dict(initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
              verbose=verbose, linear_solver=linear_solver,
              # the benchmark's iteration budget (jaxls's default, 100, when None)
              **({} if max_iters is None else {"termination": jaxls.TerminationConfig(max_iterations=max_iters)}))
    cost = lambda v: jnp.sum(problem.compute_residual_vector(v) ** 2)
    if snap_every:   # debug: external_patches/jaxls_snapshots.py, applied
        with jaxls.record_snapshots(snap_every):
            solution, summary = problem.solve(**kw, return_summary=True)
        snaps = summary.snapshots
        return solution[traj_vars], cost(solution), jax.vmap(lambda v: v[traj_vars])(snaps), jax.vmap(cost)(snaps)
    solution = problem.solve(**kw)
    traj = onp.array(solution[traj_vars]) if host else solution[traj_vars]
    if return_cost:
        return traj, cost(solution)
    return traj
