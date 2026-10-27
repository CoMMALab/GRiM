"""crocket-traced builds of the IK kernels (``kernels/ik_crocket/``).

A crocket IK build runs the same solver as its hand-written counterpart in ``kernels/ik/`` and
binds the same FFI signature, but everything it reads about the robot comes from one traced
crocket algorithm, ``ik_geometry``, specialised to the build's end-effectors, solved columns
and collision spheres:

* the stacked pose residual and Jacobian of every end-effector;
* the world centres of the world-collision and self-collision spheres;
* every joint's world twist, from which a collision gradient is a dot product.

Distances to the runtime obstacles, the active set and the solver loop stay plain CUDA,
templated on the scalar type: a build is ``float`` or ``double`` end to end
(``-DGRIM_CK_REAL=...``). :func:`source` writes the traced header and the tables between
crocket's joint order and GRiM's; it is appended to ``grim_robot_gen.cuh``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .robot import MotionRobot


def _model(robot: MotionRobot):
    import crocket
    from ._build import kinematic_urdf
    with tempfile.TemporaryDirectory(prefix="grim_ck_") as tmp:
        urdf = Path(tmp) / "robot.urdf"
        urdf.write_text(kinematic_urdf(robot.urdf_xml))
        model = crocket.Robot(urdf, name="robot")
    if sorted(model.joint_names) != sorted(robot.actuated_names):
        raise ValueError(f"crocket's joints {model.joint_names} do not match the robot's actuated "
                         f"joints {list(robot.actuated_names)} (continuous / planar / floating joints "
                         "are not supported by crocket builds)")
    return model


def _table(ctype: str, name: str, values) -> str:
    """A compile-time table as a constexpr accessor ``name(i)``: readable in device code and
    folded wherever ``i`` is a constant (the kernels' loops over spheres and joints unroll)."""
    flat = np.asarray(values).reshape(-1)
    vals = ([f"{float(v):.17e}" for v in flat] if ctype == "double" else [str(int(v)) for v in flat]) \
        or ["0"]
    return (f"static __host__ __device__ __forceinline__ constexpr {ctype} {name}(int i)\n"
            f"{{\n    constexpr {ctype} t[{len(vals)}] = {{{', '.join(vals)}}};\n    return t[i];\n}}\n")


def _constant(ctype: str, name: str, values) -> str:
    flat = np.asarray(values).reshape(-1)
    vals = ([f"{float(v):.17e}" for v in flat] if ctype == "double" else [str(int(v)) for v in flat]) or ["0"]
    return f"__device__ __constant__ {ctype} {name}[{len(vals)}] = {{{', '.join(vals)}}};\n"


def source(robot: MotionRobot, problem, solved: list[int], frozen: list[int]) -> str:
    """The traced ``ik_geometry`` of ``problem`` and its tables, namespace ``grim::ck``."""
    import crocket
    from ._build import SelfCollision
    model = _model(robot)
    col_of = {robot.actuated_names.index(n): i for i, n in enumerate(model.joint_names)}
    root = sorted(model.root_links)[0]

    def frame(j: int) -> str:
        return root if j < 0 else robot.joint_names[j]

    def chain(name: str) -> list[int]:
        out, j = [], model.frame(name).parent
        while j >= 0:
            out.append(j)
            j = model.joints[j].parent
        return out

    sph = np.zeros((0, 4)) if problem.robot_spheres is None else np.asarray(problem.robot_spheres)
    sph_joint = np.zeros(0, int) if problem.robot_sphere_joint is None else \
        np.asarray(problem.robot_sphere_joint, int)
    sc = problem.self_collision or SelfCollision.none()
    self_frame = []                                   # self sphere -> its link's joint frame
    for link in range(len(sc.link_start) - 1):
        self_frame += [int(sc.link_joint[link])] * int(sc.link_start[link + 1] - sc.link_start[link])
    # A world sphere with no joint frame is skipped, as kernels/ik/*.cu do (`jidx < 0`).
    ws_idx = [i for i, j in enumerate(sph_joint) if 0 <= j < robot.n_joints]
    ws = tuple((frame(int(sph_joint[i])), tuple(float(x) for x in sph[i, :3])) for i in ws_idx)
    ss = tuple((frame(j), tuple(float(x) for x in s[:3])) for s, j in zip(np.asarray(sc.sph), self_frame))
    frames = tuple(robot.joint_names[e] for e in problem.ee_joints)
    cols = tuple(col_of[a] for a in solved)
    opts = {"frames": frames, "cols": cols, "wspheres": ws, "sspheres": ss}
    traced = crocket.generate(model, {"ik_geometry": opts, "ik_geometry_value": opts},
                              variants=[crocket.Variant("thread")], name="robot")

    nj = len(model.joints)
    joint_col = [solved.index(robot.actuated_names.index(model.joint_names[j.q_index]))
                 if robot.actuated_names.index(model.joint_names[j.q_index]) in solved else -1
                 for j in model.joints]
    ws_chain = np.zeros((max(len(ws), 1), nj), int)
    for i, (name, _) in enumerate(ws):
        ws_chain[i, chain(name)] = 1
    ss_chain = np.zeros((max(len(ss), 1), nj), int)
    for i, (name, _) in enumerate(ss):
        ss_chain[i, chain(name)] = 1
    src = {a: f"cfg[{solved.index(a)}]" if a in solved else f"frz[{frozen.index(a)}]"
           for a in range(robot.n_act)}
    gather = "".join(f"    q[{i}] = {src[robot.actuated_names.index(n)]};\n"
                     for i, n in enumerate(model.joint_names))
    return (f"{traced}\n"
            "namespace grim::ck {\n"
            "namespace tr = crocket::robots::robot;\n"
            f"constexpr int NQ = {model.n_q};\n"
            f"constexpr int NJ = {nj};\n"
            f"constexpr int NWS = {len(ws)};\n"
            f"constexpr int NSS = {len(ss)};\n"
            + _table("int", "kJointCol", joint_col)
            # Read with runtime sphere indices: __constant__, where a constexpr table would be
            # copied onto the stack at every call.
            + _constant("double", "kWsRad", [sph[i, 3] for i in ws_idx])
            + _constant("double", "kSsRad", np.asarray(sc.sph)[:, 3] if len(sc.sph) else [])
            + _constant("signed char", "kWsChain", ws_chain)
            + _constant("signed char", "kSsChain", ss_chain)
            + "template <class T>\n"
            "static __device__ __forceinline__ void gather(const T* __restrict__ cfg,\n"
            "    const T* __restrict__ frz, T* __restrict__ q)\n"
            f"{{\n{gather}    (void)frz;\n}}\n"
            "}  // namespace grim::ck\n")
