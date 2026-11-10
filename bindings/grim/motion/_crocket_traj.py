"""crocket-traced builds of the trajectory optimizers (``kernels/trajopt_crocket/``).

SCO, CHOMP and STOMP read the robot only through their collision terms over a
:class:`TrajCollision` model: every sphere of every row, posed by its joint frame, or by the root
link for row ``n_joints``. crocket traces those terms (``traj_collision``, ``traj_smooth_min``)
for one waypoint, with their gradients by a world-wrench backward pass, in ``float`` or
``double`` (``-DGRIM_CK_REAL=...``). Smoothness, limits and the solver loops stay CUDA in the
same scalar type.

The traces are crocket's thread variant at every tier: the cooperative tiers spread a
trajectory's timesteps (and samples, line-search trials) over lanes, each lane evaluating its own
waypoint (see kernels/tier_kernel.cuh, FK INDEPENDENCE).
"""

from __future__ import annotations

import numpy as np

from ._build import TrajCollision
from ._crocket_ik import _matrix, _model, _table, _wxyz
from .robot import MotionRobot


def source(robot: MotionRobot, tc: TrajCollision | None, runtime_rot_joint: int = -1,
           world_counts=(0, 0, 0, 0), real: str = "float") -> str:
    """The traced collision terms of ``tc`` (:func:`_analytic_source`) and the gather / scatter
    between the configuration's order and crocket's, namespace ``grim::tk``."""
    model = _model(robot)
    root = sorted(model.root_links)[0]
    if tc is None:
        off, rad = np.zeros((robot.n_joints + 1, 1, 3)), -np.ones((robot.n_joints + 1, 1))
        pi = pj = np.zeros(0, int)
    else:
        off, rad = np.asarray(tc.sphere_off, float), np.asarray(tc.sphere_rad, float)
        pi, pj = np.asarray(tc.pair_i, int), np.asarray(tc.pair_j, int)
    R, S = rad.shape
    rot_pre, rot_opt = [], {}
    if runtime_rot_joint >= 0:   # as in _crocket_ik.source
        name = robot.joint_names[runtime_rot_joint]
        if name not in model.joint_names:
            raise ValueError(f"runtime_rot_joint {name!r} is not a movable joint: crocket folds "
                             "fixed joints into the movable joints below them")
        jc = model.joint_names.index(name)
        jt = model.joints[jc]
        Rc = (jt.E[0] + jt.E[1] if jt.kind == "revolute" else jt.E[0]).T
        rot_pre = _wxyz(Rc @ _matrix(robot.parent_tf[runtime_rot_joint, :4]).T)
        rot_opt = {"rot_joint": jc}
    rows = []
    for r in range(R):
        name = robot.joint_names[r] if r < robot.n_joints else root
        rows.append((name, tuple((*map(float, off[r, s]), float(rad[r, s]))
                                 for s in range(S) if rad[r, s] >= 0)))
    traced = _analytic_source(model, tuple(rows), tuple(zip(map(int, pi), map(int, pj))),
                              world_counts, rot_opt, real)
    gather = "".join(f"    q[{i}] = T(cfg[{robot.actuated_names.index(n)}]);\n"
                     for i, n in enumerate(model.joint_names))
    return (f"{traced}\n"
            "namespace grim::tk {\n"
            f"constexpr int NQ = {model.n_q};\n"
            f"constexpr int NROT = {len(rot_pre)};\n"
            + _table("double", "kRotPre", rot_pre)
            + "template <class T, class U>\n"
            "static __device__ __forceinline__ void gather(const U* __restrict__ cfg, T* __restrict__ q)\n"
            f"{{\n{gather}}}\n"
            "/** A gradient in crocket's q columns, scattered into the configuration's order. */\n"
            "template <class T>\n"
            "static __device__ __forceinline__ void scatter(const T* __restrict__ g, T* __restrict__ out)\n"
            "{\n" + "".join(f"    out[{robot.actuated_names.index(n)}] = g[{i}];\n"
                            for i, n in enumerate(model.joint_names)) + "}\n"
            "}  // namespace grim::tk\n")


# name -> (algorithm, options) of the traced functions (kernels/trajopt_crocket/tk.cuh)
_ANALYTIC = {"ck_hinge_grad": ("traj_collision", {"mode": "hinge"}),
             "ck_hinge": ("traj_collision", {"mode": "hinge", "gradient": False}),
             "ck_sdf": ("traj_collision", {"mode": "sdf", "gradient": False}),
             "ck_smin_jac": ("traj_smooth_min", {})}


def _analytic_source(model, rows, pairs, world_counts, rot_opt, real: str) -> str:
    """crocket's ``traj_collision`` / ``traj_smooth_min`` for this model: one waypoint's
    collision cost (or SCO's smooth-min distances) and its gradient by crocket's world-wrench
    backward pass. Each is the thread variant at the statement budget crocket's register guard (``harness.fit``) found spill-free."""
    import dataclasses

    import crocket
    from crocket import cuda, harness

    counts = dict(zip(("n_spheres", "n_capsules", "n_boxes", "n_halfspaces"), map(int, world_counts)))
    dtype = np.float32 if real == "float" else np.float64
    sections = []
    for name, (alg, extra) in _ANALYTIC.items():
        p = crocket.trace(model, alg, rows=rows, pairs=pairs, **counts, **rot_opt, **extra)
        p = dataclasses.replace(p, name=name)
        _, budget = harness.fit(p, cuda.Variant("thread"), dtype)
        sections.append(cuda.emit(p, [cuda.Variant("thread")], budget))
    return cuda.header(model, sections, "robot_grad").replace("#pragma once", "")
