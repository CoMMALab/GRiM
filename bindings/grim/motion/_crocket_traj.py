"""crocket-traced builds of the trajectory optimizers (``kernels/trajopt_crocket/``).

SCO, CHOMP and STOMP read the robot only through the world centres of their collision spheres
(``TrajCollision``): every sphere of every row, posed by its joint frame, or by the root link for
row ``n_joints``. crocket traces exactly that, ``ik_geometry_value`` with no target frames and
the spheres as ``wspheres``, so a configuration's whole kinematics is one straight-line call in
``float`` or ``double`` (``-DGRIM_CK_REAL=...``). Distances, finite differences and the solver
loops stay CUDA in the same scalar type.

The trace is crocket's thread variant at every tier: the cooperative tiers spread a trajectory's
timesteps (and perturbations, samples, line-search trials) over lanes, each lane posing its own
configuration, so the FK stays one lane's work (see kernels/tier_kernel.cuh, FK INDEPENDENCE).
"""

from __future__ import annotations

import numpy as np

from ._build import TrajCollision
from ._crocket_ik import _constant, _matrix, _model, _table, _wxyz
from .robot import MotionRobot


def source(robot: MotionRobot, tc: TrajCollision | None, runtime_rot_joint: int = -1,
           analytic: bool = False, world_counts=(0, 0, 0, 0), real: str = "float") -> str:
    """The traced sphere centres of ``tc`` and their tables, namespace ``grim::tk``. With
    ``analytic``, also the traced collision costs with their analytic gradients
    (:func:`_analytic_source`)."""
    import crocket
    model = _model(robot)
    root = sorted(model.root_links)[0]
    if tc is None:
        off, rad = np.zeros((robot.n_joints + 1, 1, 3)), -np.ones((robot.n_joints + 1, 1))
        pi = pj = np.zeros(0, int)
    else:
        off, rad = np.asarray(tc.sphere_off, float), np.asarray(tc.sphere_rad, float)
        pi, pj = np.asarray(tc.pair_i, int), np.asarray(tc.pair_j, int)
    R, S = rad.shape
    idx = -np.ones(R * S, int)
    spheres, radii = [], []
    for r in range(R):
        for s in range(S):
            if rad[r, s] < 0:
                continue
            idx[r * S + s] = len(spheres)
            spheres.append((robot.joint_names[r] if r < robot.n_joints else root,
                            tuple(float(x) for x in off[r, s])))
            radii.append(rad[r, s])
    opts = {"frames": (), "cols": (), "wspheres": tuple(spheres), "sspheres": ()}
    rot_pre = []
    if runtime_rot_joint >= 0:   # as in _crocket_ik.source
        name = robot.joint_names[runtime_rot_joint]
        if name not in model.joint_names:
            raise ValueError(f"runtime_rot_joint {name!r} is not a movable joint: crocket folds "
                             "fixed joints into the movable joints below them")
        jc = model.joint_names.index(name)
        jt = model.joints[jc]
        Rc = (jt.E[0] + jt.E[1] if jt.kind == "revolute" else jt.E[0]).T
        rot_pre = _wxyz(Rc @ _matrix(robot.parent_tf[runtime_rot_joint, :4]).T)
        opts["rot_joint"] = jc
    traced = crocket.generate(model, {"ik_geometry_value": opts},
                              variants=[crocket.Variant("thread")], name="robot")
    if analytic:
        rows = []
        for r in range(R):
            name = robot.joint_names[r] if r < robot.n_joints else root
            rows.append((name, tuple((*map(float, off[r, s]), float(rad[r, s]))
                                     for s in range(S) if rad[r, s] >= 0)))
        rot_opt = {"rot_joint": opts["rot_joint"]} if "rot_joint" in opts else {}
        traced += _analytic_source(model, tuple(rows), tuple(zip(map(int, pi), map(int, pj))),
                                   world_counts, rot_opt, real)
    gather = "".join(f"    q[{i}] = T(cfg[{robot.actuated_names.index(n)}]);\n"
                     for i, n in enumerate(model.joint_names))
    return (f"{traced}\n"
            "namespace grim::tk {\n"
            "namespace tr = crocket::robots::robot;\n"
            f"constexpr int NQ = {model.n_q};\n"
            f"constexpr int NSPH = {len(spheres)};\n"
            f"constexpr int NROT = {len(rot_pre)};\n"
            + _table("double", "kRotPre", rot_pre)
            + _table("int", "kSph", idx)
            + _distance_source(idx, radii, pi, pj, R, S)
            + "template <class T, class U>\n"
            "static __device__ __forceinline__ void gather(const U* __restrict__ cfg, T* __restrict__ q)\n"
            f"{{\n{gather}}}\n"
            "/** A gradient in crocket's q columns, scattered into the configuration's order. */\n"
            "template <class T>\n"
            "static __device__ __forceinline__ void scatter(const T* __restrict__ g, T* __restrict__ out)\n"
            "{\n" + "".join(f"    out[{robot.actuated_names.index(n)}] = g[{i}];\n"
                            for i, n in enumerate(model.joint_names)) + "}\n"
            "}  // namespace grim::tk\n")


def _distance_source(idx, radii, pi, pj, R: int, S: int) -> str:
    """Per-pair and per-row hard-min distances with literal sphere indices and radii. Relying on
    nvcc to unroll table loops instead fails on large models (g1: 324 pairs) and rebuilds the
    radius table on the stack at every use."""
    sph = lambda row: [(int(a), f"R({radii[a]:.17e})") for a in idx[row * S:(row + 1) * S] if a >= 0]  # noqa: E731
    def body(terms):
        return "".join(f"            {{ const R v = {t}; if (v < m) m = v; }}\n" for t in terms)
    pairs = "".join(
        f"        case {p}:\n"
        + body(f"d(cw + {3 * a}, {ra}, cw + {3 * b}, {rb})" for a, ra in sph(pi[p]) for b, rb in sph(pj[p]))
        + "            break;\n" for p in range(len(pi)))
    rows = "".join(
        f"        case {j}:\n" + body(f"d(cw + {3 * a}, {ra})" for a, ra in sph(j)) + "            break;\n"
        for j in range(R))
    return ("/** Hard min of d over the spheres of self pair p (1e10 when a row has none). */\n"
            "template <class R, class D>\n"
            "static __device__ __forceinline__ R self_min_gen(const R* cw, int p, D d)\n"
            "{\n    R m = R(1e10);\n    switch (p) {\n" + pairs + "    }\n    return m;\n}\n"
            "/** Hard min of d over row j's spheres (1e10 when the row has none). */\n"
            "template <class R, class D>\n"
            "static __device__ __forceinline__ R row_min_gen(const R* cw, int j, D d)\n"
            "{\n    R m = R(1e10);\n    switch (j) {\n" + rows + "    }\n    return m;\n}\n")


# name -> (algorithm, options) of the analytic build's traced functions (kernels/trajopt_crocket/tk.cuh)
_ANALYTIC = {"ck_hinge_grad": ("traj_collision", {"mode": "hinge"}),
             "ck_hinge": ("traj_collision", {"mode": "hinge", "gradient": False}),
             "ck_sdf": ("traj_collision", {"mode": "sdf", "gradient": False}),
             "ck_smin_jac": ("traj_smooth_min", {}),
             "ck_smin": ("traj_smooth_min", {"gradient": False})}


def _analytic_source(model, rows, pairs, world_counts, rot_opt, real: str) -> str:
    """crocket's ``traj_collision`` / ``traj_smooth_min`` for this model: one waypoint's
    collision cost (or SCO's smooth-min distances) and its gradient by crocket's world-wrench
    backward pass, replacing the kernels' central differences. Each is the thread variant at the
    statement budget crocket's register guard (``harness.fit``) found spill-free."""
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
