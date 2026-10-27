"""crocket-traced builds of the IK kernels (``kernels/ik_crocket/``).

These are GRiM's IK kernels. Each runs the solver of the hand-written kernel it replaced (tag
``pre-crocket-ik``; a local copy may sit in the untracked ``legacy/``) and binds the same FFI
signature, but everything it reads about the robot comes from one traced
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

BLOCK_THREADS = 64   # default lanes per seed at the block tier (crocket's block variant is built for it)


def _model(robot: MotionRobot):
    import crocket
    from ._build import kinematic_urdf
    with tempfile.TemporaryDirectory(prefix="grim_ck_") as tmp:
        urdf = Path(tmp) / "robot.urdf"
        urdf.write_text(kinematic_urdf(robot.urdf_xml))
        model = crocket.Robot(urdf, name="robot")
    if sorted(model.joint_names) != sorted(robot.actuated_names):
        raise ValueError(f"crocket's joints {model.joint_names} do not match the robot's actuated "
                         f"joints {list(robot.actuated_names)} (planar / floating joints "
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


def _matrix(wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _wxyz(R) -> np.ndarray:
    """Unit quaternion (w >= 0) of rotation matrix ``R``, Shepperd's method."""
    t = np.trace(R)
    if t > 0:
        w = np.sqrt(1 + t) / 2
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        v = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) / 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / (4 * v)
        q[1 + i], q[1 + j], q[1 + k] = v, (R[j, i] + R[i, j]) / (4 * v), (R[k, i] + R[i, k]) / (4 * v)
    q = np.asarray(q)
    return q if q[0] >= 0 else -q


def source(robot: MotionRobot, problem, solved: list[int], frozen: list[int],
           block_threads: int = BLOCK_THREADS) -> str:
    """The traced ``ik_geometry`` of ``problem`` and its tables, namespace ``grim::ck``; the block
    tier runs ``block_threads`` lanes per seed."""
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
    # A world sphere with no joint frame is skipped (`jidx < 0` in the hand-written kernels).
    ws_idx = [i for i, j in enumerate(sph_joint) if 0 <= j < robot.n_joints]
    ws = tuple((frame(int(sph_joint[i])), tuple(float(x) for x in sph[i, :3])) for i in ws_idx)
    ss = tuple((frame(j), tuple(float(x) for x in s[:3])) for s, j in zip(np.asarray(sc.sph), self_frame))
    frames = tuple(robot.joint_names[e] for e in problem.ee_joints)
    cols = tuple(col_of[a] for a in solved)
    opts = {"frames": frames, "cols": cols, "wspheres": ws, "sspheres": ss}
    rot_pre = []
    if problem.runtime_rot_joint >= 0:
        # GRiM's per-call rotation replaces joint j's URDF origin rotation R0; crocket's replaces
        # the whole parent-to-joint rotation at q = 0, Rc = Rpre R0 (Rpre: fixed joints folded in
        # before it). The kernel hands crocket q(Rpre) * rot, rot being GRiM's operand.
        name = robot.joint_names[problem.runtime_rot_joint]
        if name not in model.joint_names:
            raise ValueError(f"runtime_rot_joint {name!r} is not a movable joint: crocket folds "
                             "fixed joints into the movable joints below them")
        jc = model.joint_names.index(name)
        jt = model.joints[jc]
        Rc = (jt.E[0] + jt.E[1] if jt.kind == "revolute" else jt.E[0]).T
        rot_pre = _wxyz(Rc @ _matrix(robot.parent_tf[problem.runtime_rot_joint, :4]).T)
        opts["rot_joint"] = jc
    # Thread, warp and block (block_threads lanes) from the same trace: the kernels compile at
    # every tier and call the variant of theirs.
    traced = crocket.generate(model, {"ik_geometry": opts, "ik_geometry_value": opts},
                              variants=[crocket.Variant("thread"), crocket.Variant("warp", units=32),
                                        crocket.Variant("block", units=block_threads)], name="robot")

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
            f"constexpr int BLOCK_THREADS = {block_threads};\n"
            f"constexpr int NROT = {len(rot_pre)};\n"   # 4 with a runtime parent rotation
            + _table("double", "kRotPre", rot_pre)
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
