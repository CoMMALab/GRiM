"""mantis-traced builds of the trajectory optimizers (``kernels/trajopt_mantis/``).

SCO, CHOMP and STOMP read the robot only through their collision terms over a
:class:`TrajCollision` model: every sphere of every row, posed by its joint frame, or by the root
link for row ``n_joints``. mantis traces the robot half of those terms for one waypoint: the
sphere centres (``traj_spheres``) and the world-wrench backward pass from per-sphere gradients
to ``dcost/dq`` (``traj_backprop``), in ``float`` or ``double`` (``-DGRIM_CK_REAL=...``). The
world is a runtime argument: the distances to it (and between self pairs), the costs and the
per-sphere gradients are CUDA loops in kernels/trajopt_mantis/tk.cuh, so one build serves any
obstacles. Smoothness, limits and the solver loops stay CUDA in the same scalar type.

The traces are mantis's thread variant at every tier: the cooperative tiers spread a
trajectory's timesteps (and samples, line-search trials) over lanes, each lane evaluating its own
waypoint (see kernels/tier_kernel.cuh, FK INDEPENDENCE).
"""

from __future__ import annotations

import numpy as np

from ._build import TrajCollision
from ._mantis_ik import _matrix, _model, _table, _wxyz
from .robot import MotionRobot


def source(robot: MotionRobot, tc: TrajCollision | None, runtime_rot_joint: int = -1,
           real: str = "float") -> str:
    """The traced robot half of ``tc``'s collision terms (:func:`_analytic_source`), the sphere
    tables tk.cuh loops over, and the gather / scatter between the configuration's order and
    mantis's, namespace ``grim::tk``."""
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
    if runtime_rot_joint >= 0:   # as in _mantis_ik.source
        name = robot.joint_names[runtime_rot_joint]
        if name not in model.joint_names:
            raise ValueError(f"runtime_rot_joint {name!r} is not a movable joint: mantis folds "
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
    traced = _analytic_source(model, tuple(rows), tuple(zip(map(int, pi), map(int, pj))), rot_opt, real)
    # Live spheres in traj_spheres' order (row-major, padding skipped): radius, and each row's run.
    live_rad = [s[3] for _, sph in rows for s in sph]
    row_start = np.cumsum([0] + [len(sph) for _, sph in rows]).tolist()
    gather = "".join(f"    q[{i}] = T(cfg[{robot.actuated_names.index(n)}]);\n"
                     for i, n in enumerate(model.joint_names))
    return (f"{traced}\n"
            "namespace grim::tk {\n"
            f"constexpr int NQ = {model.n_q};\n"
            f"constexpr int NROT = {len(rot_pre)};\n"
            f"constexpr int SL = {len(live_rad)};\n"
            + _table("double", "kSphRad", live_rad) + _table("int", "kRowStart", row_start)
            + _table("double", "kRotPre", rot_pre)
            + "template <class T, class U>\n"
            "static __device__ __forceinline__ void gather(const U* __restrict__ cfg, T* __restrict__ q)\n"
            f"{{\n{gather}}}\n"
            "/** A gradient in mantis's q columns, scattered into the configuration's order. */\n"
            "template <class T>\n"
            "static __device__ __forceinline__ void scatter(const T* __restrict__ g, T* __restrict__ out)\n"
            "{\n" + "".join(f"    out[{robot.actuated_names.index(n)}] = g[{i}];\n"
                            for i, n in enumerate(model.joint_names)) + "}\n"
            "}  // namespace grim::tk\n")


# name -> (algorithm, options) of the traced functions (kernels/trajopt_mantis/tk.cuh). The self
# terms depend on the robot alone, so they stay whole traces (mantis's traj_collision /
# traj_smooth_min with no world: straight-line, in registers); the world terms are tk.cuh loops
# over the runtime obstacles around the sphere centres (traj_spheres) and the backward pass
# (traj_backprop).
_ANALYTIC = {"ck_spheres": ("traj_spheres", {}), "ck_backprop": ("traj_backprop", {}),
             "ck_self_hinge_grad": ("traj_collision", {"mode": "hinge", "pairs": True}),
             "ck_self_hinge": ("traj_collision", {"mode": "hinge", "gradient": False, "pairs": True}),
             "ck_self_sdf": ("traj_collision", {"mode": "sdf", "gradient": False, "pairs": True}),
             "ck_self_smin": ("traj_smooth_min", {"pairs": True})}


def _analytic_source(model, rows, pairs, rot_opt, real: str) -> str:
    """The traced functions of ``_ANALYTIC`` for this model: one waypoint's sphere centres,
    ``dcost/dq`` from per-sphere gradients by mantis's world-wrench backward pass, and the self
    terms' cost (and gradient, or SCO's smooth-min group and its Jacobian).
    Each is the thread variant at the statement budget mantis's register guard
    (``harness.fit``) found spill-free."""
    import dataclasses

    import mantis
    from mantis import cuda, harness

    dtype = np.float32 if real == "float" else np.float64
    sections = []
    for name, (alg, extra) in _ANALYTIC.items():
        extra = dict(extra)
        if extra.pop("pairs", False):
            extra["pairs"] = pairs
        p = mantis.trace(model, alg, rows=rows, **rot_opt, **extra)
        p = dataclasses.replace(p, name=name)
        _, budget = harness.fit(p, cuda.Variant("thread"), dtype)
        sections.append(cuda.emit(p, [cuda.Variant("thread")], budget))
    return cuda.header(model, sections, "robot_grad").replace("#pragma once", "")
