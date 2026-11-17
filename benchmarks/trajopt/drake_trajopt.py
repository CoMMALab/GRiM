"""Drake ``KinematicTrajectoryOptimization`` (1.51.1 wheel) on the shared problem (mbm.py).

Drake has no packaged collision-avoiding trajopt; this follows the canonical pattern of Tedrake's
Robotic Manipulation course (book/trajectories/kinematic_trajectory_optimization.ipynb,
``trajopt_shelves_demo``, manipulation@dd2bc0f) with its settings: a cubic B-spline (order 4) of
``CONTROL_POINTS`` control points, duration and path-length costs, the plant's position and velocity
limits, duration in [0.5, 5] s, the endpoints as path constraints (joint values here, not gripper
positions), and zero end velocities. As the demo, it solves once without collisions, then adds
``MinimumDistanceLowerBoundConstraint`` (bound mbm.MARGIN, influence 0.1 m) at
``COLLISION_SAMPLES`` path points and solves again from the first solution, with Drake's default
solver for the program (SNOPT in the wheel). The first solve starts from the straight line. Robot and world: the shared sphere
URDF (tesseract_trajopt.robot_files), MBM's boxes and cylinders as native Drake shapes, self
pairs outside the shared list filtered out of collision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mbm  # noqa: E402

CONTROL_POINTS = 10
COLLISION_SAMPLES = 25
DURATION = (0.5, 5.0)
INFLUENCE = 0.1


def build(p: dict, urdf: Path):
    """(diagram, plant, scene_graph) of the robot and problem ``p``'s obstacles."""
    from pydrake.geometry import Box, CollisionFilterDeclaration, Cylinder, GeometrySet
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.multibody.plant import CoulombFriction
    from pydrake.planning import RobotDiagramBuilder
    b = RobotDiagramBuilder(time_step=0.0)
    plant, sg = b.plant(), b.scene_graph()
    model = b.parser().AddModels(str(urdf))[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(mbm.BASE, model))
    fr = CoulombFriction(0.5, 0.5)
    for i, box in enumerate(p["boxes"]):
        X = RigidTransform(RotationMatrix(box[3:12].reshape(3, 3).T), box[:3])
        plant.RegisterCollisionGeometry(plant.world_body(), X, Box(*(2 * box[12:15])), f"box{i}", fr)
    for i, c in enumerate(p["cylinders"]):
        z = c[3:6] / np.linalg.norm(c[3:6])
        x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
        x /= np.linalg.norm(x)
        X = RigidTransform(RotationMatrix(np.stack([x, np.cross(z, x), z], 1)), c[:3])
        plant.RegisterCollisionGeometry(plant.world_body(), X, Cylinder(c[6], 2 * c[7]), f"cyl{i}", fr)
    plant.Finalize()
    diagram = b.Build()
    names = [plant.get_joint(j).name() for j in plant.GetJointIndices(model) if plant.get_joint(j).num_positions()]
    if names != mbm.JOINTS:
        raise ValueError(f"Drake joint order {names} differs from the URDF's")
    # Self collision: only the shared pairs (Drake filters parent-child pairs already; filter the rest).
    keep = {tuple(sorted(x)) for x in mbm.self_pairs()}
    links = [plant.get_body(i).name() for i in plant.GetBodyIndices(model)]
    geoms = {l: plant.GetCollisionGeometriesForBody(plant.GetBodyByName(l, model)) for l in links}
    mgr = sg.collision_filter_manager()
    for i, a in enumerate(links):
        for b_ in links[i + 1:]:
            if tuple(sorted((a, b_))) not in keep and geoms[a] and geoms[b_]:
                mgr.Apply(CollisionFilterDeclaration().ExcludeBetween(GeometrySet(geoms[a]), GeometrySet(geoms[b_])))
    return diagram, plant


def problem(diagram, plant, p: dict):
    """The KinematicTrajectoryOptimization of ``p``, with the straight-line seed as its guess."""
    from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint
    from pydrake.planning import KinematicTrajectoryOptimization
    from pydrake.trajectories import BsplineTrajectory
    ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(ctx)
    n = plant.num_positions()
    q0, qg = np.asarray(p["start"], float), np.asarray(p["goal"], float)
    trajopt = KinematicTrajectoryOptimization(n, CONTROL_POINTS, spline_order=4)
    prog = trajopt.get_mutable_prog()
    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
    trajopt.AddDurationConstraint(*DURATION)
    trajopt.AddPathPositionConstraint(q0, q0, 0)
    trajopt.AddPathPositionConstraint(qg, qg, 1)
    trajopt.AddPathVelocityConstraint(np.zeros(n), np.zeros(n), 0)
    trajopt.AddPathVelocityConstraint(np.zeros(n), np.zeros(n), 1)
    seed = BsplineTrajectory(trajopt.basis(),
                             [(q0 + (qg - q0) * t).reshape(-1, 1) for t in np.linspace(0, 1, CONTROL_POINTS)])
    trajopt.SetInitialGuess(seed)
    coll = MinimumDistanceLowerBoundConstraint(plant, mbm.MARGIN, plant_ctx, None, INFLUENCE)
    return trajopt, prog, [(coll, s) for s in np.linspace(0, 1, COLLISION_SAMPLES)]


def solve(trajopt, prog, collision, max_iters: int | None = None, samples: int = 100) -> tuple[np.ndarray, bool]:
    """The demo's two solves: without collisions, then with ``collision`` (problem()'s constraint
    and path points) from that solution. ``max_iters``: SNOPT's major-iteration limit of each solve
    (the benchmark's budget); None: SNOPT's default."""
    from pydrake.solvers import SnoptSolver, Solve, SolverOptions
    opts = SolverOptions()
    if max_iters is not None:
        opts.SetOption(SnoptSolver().solver_id(), "Major iterations limit", int(max_iters))
    result = Solve(prog, solver_options=opts)
    trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))
    for c, s in collision:
        trajopt.AddPathPositionConstraint(c, s)
    result = Solve(prog, solver_options=opts)
    traj = trajopt.ReconstructTrajectory(result)
    ts = np.linspace(traj.start_time(), traj.end_time(), samples)
    return np.array([traj.value(t).ravel() for t in ts]), bool(result.is_success())
