"""Per-robot builds of the motion kernels.

A motion kernel is compiled for one :class:`~grim.motion.MotionRobot` and one problem
structure. :func:`robot_header` writes everything that build depends on into
``grim_robot_gen.cuh`` (see ``kernels/robot.cuh`` for what the kernels see):

* the kinematic tables and sizes, as compile-time constants;
* which actuated joints are solved and which ride along frozen from the seed;
* the end-effector frames and their ancestor chains;
* the collision tables (robot spheres, self-collision pairs) and the world obstacle counts;
* optionally (``traced=True``) crocket's straight-line FK of the same robot.

:func:`build` nvcc's a kernel source against that header into a shared library, cached on
disk under a key over every input, and :func:`register` exposes its XLA FFI handlers to JAX.
The first build of a robot takes tens of seconds (minutes when traced); later processes
load the cached library.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .robot import MotionRobot

KERNELS_DIR = Path(__file__).parent / "kernels"
_REPO = Path(__file__).resolve().parents[3]
GLASS_DIR = Path(os.environ.get("GRIM_GLASS_DIR", _REPO / "external" / "GRiD" / "external" / "GLASS"))

WORLD_KINDS = ("spheres", "capsules", "boxes", "halfspaces")
# Row width of each world obstacle kind, as the kernels read it.
WORLD_WIDTH = {"spheres": 4, "capsules": 7, "boxes": 15, "halfspaces": 6}


@dataclass(frozen=True)
class SelfCollision:
    """Self-collision model: link-local spheres in CSR runs per link, and active link pairs.

    ``sph`` (K, 4) xyz+radius in the link frame; ``link_start`` (L+1) CSR offsets into ``sph``;
    ``link_joint`` (L) the joint frame posing each link; ``pair_i``/``pair_j`` (P) the link pairs
    to test (adjacency-filtered, e.g. from an SRDF).
    """
    sph: np.ndarray
    link_start: np.ndarray
    link_joint: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray

    @staticmethod
    def none() -> "SelfCollision":
        return SelfCollision(np.zeros((0, 4)), np.zeros(1, np.int32), np.zeros(0, np.int32),
                             np.zeros(0, np.int32), np.zeros(0, np.int32))


@dataclass(frozen=True)
class TrajCollision:
    """Collision model of the trajectory optimizers.

    ``sphere_off`` (R, S, 3) / ``sphere_rad`` (R, S): spheres of row ``r`` live in joint frame
    ``r`` for ``r < n_joints`` and in the world frame for ``r == n_joints`` (a root link);
    ``R = n_joints + 1`` and a radius < 0 pads a row. ``pair_i``/``pair_j`` (P,) are the row
    pairs checked for self-collision.
    """
    sphere_off: np.ndarray
    sphere_rad: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray


@dataclass(frozen=True)
class Problem:
    """The structure one build is specialized to (everything but the per-call values).

    ``ee_joints`` are the joint frames whose poses are targeted (IK; empty for trajopt).
    ``robot_spheres`` (S, 4) / ``robot_sphere_joint`` (S,) are the world-collision spheres in
    their joint frames. ``world_counts`` is the number of world obstacles of each kind in
    ``WORLD_KINDS``: the obstacle SET is part of the build, their poses are runtime inputs.
    ``chain_only`` solves over the end-effector chains alone (only valid without collision:
    joints off the chains then have zero gradient and can never move).
    ``runtime_rot_joint`` >= 0 makes that joint's parent rotation a per-call input (the
    reference rotation of a floating base's chart) instead of a baked constant; the build then
    takes it as the first operand, a wxyz ``(4,)`` float32.
    ``fused_dls_ori_ratio`` (traced, one end-effector) also traces crocket's whole DLS step
    ``ik_lm_step`` at position weight 1 and this orientation weight, with the URDF's limits
    baked in (``GRIM_FUSED_DLS``; see kernels/ik/dls_ik.cu).
    """
    ee_joints: tuple[int, ...] = ()
    robot_spheres: np.ndarray | None = None
    robot_sphere_joint: np.ndarray | None = None
    self_collision: SelfCollision | None = None
    world_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    chain_only: bool = False
    traj_collision: TrajCollision | None = None
    runtime_rot_joint: int = -1
    fused_dls_ori_ratio: float | None = None


def _constant(ctype: str, name: str, values) -> str:
    values = np.asarray(values).reshape(-1)
    fmt = (lambda v: f"{float(v):.9e}f") if ctype == "float" else (lambda v: str(int(v)))
    return (f"__device__ __constant__ {ctype} {name}[{max(values.size, 1)}] = "
            f"{{{', '.join(fmt(v) for v in values) or '0'}}};\n")


def _index_fn(name: str, values) -> str:
    return (f"static __host__ __device__ __forceinline__ constexpr int {name}(int i)\n"
            f"{{\n    constexpr int t[{max(len(values), 1)}] = {{{', '.join(map(str, values)) or '0'}}};\n"
            f"    return t[i];\n}}\n")


def solved_joints(robot: MotionRobot, problem: Problem) -> list[int]:
    """Actuated indices (q order) a build optimizes."""
    has_collision = (problem.robot_spheres is not None and len(problem.robot_spheres) > 0) or \
        (problem.self_collision is not None and len(problem.self_collision.pair_i) > 0)
    if not problem.chain_only or has_collision or not problem.ee_joints:
        return list(range(robot.n_act))
    on_chain = set()
    for e in problem.ee_joints:
        for j in robot.chain(e):
            a = robot.act_idx[j] if robot.act_idx[j] != -1 else robot.mimic_act_idx[j]
            if a != -1:
                on_chain.add(int(a))
    return sorted(on_chain)


def robot_header(robot: MotionRobot, problem: Problem, traced: bool = False) -> str:
    solved = solved_joints(robot, problem)
    frozen = [a for a in range(robot.n_act) if a not in solved]
    ee = list(problem.ee_joints)
    ancestors = np.zeros((max(len(ee), 1), robot.n_joints), np.int32)
    for k, e in enumerate(ee):
        ancestors[k, robot.chain(e)] = 1
    sph = np.zeros((0, 4)) if problem.robot_spheres is None else np.asarray(problem.robot_spheres)
    sph_joint = np.zeros(0, np.int32) if problem.robot_sphere_joint is None else \
        np.asarray(problem.robot_sphere_joint)
    sc = problem.self_collision or SelfCollision.none()
    c = _constant

    out = ["#pragma once\n"]
    if traced:
        out.append(_traced_source(robot, solved, frozen, ee, problem.fused_dls_ori_ratio))
    out.append(
        "namespace grim::robot {\n"
        f"constexpr int n_frames = {robot.n_joints};\n"
        f"constexpr int n_q = {robot.n_act};\n"
        f"constexpr int n_solved = {len(solved)};\n"
        f"constexpr int n_frozen = {len(frozen)};\n"
        f"constexpr int n_ee = {len(ee)};\n"
        + c("float", "kTwists", robot.twists) + c("float", "kParentTf", robot.parent_tf)
        + c("int", "kParentIdx", robot.parent_idx) + c("int", "kActIdx", robot.act_idx)
        + c("float", "kMimicMul", robot.mimic_mul) + c("float", "kMimicOff", robot.mimic_off)
        + c("int", "kMimicActIdx", robot.mimic_act_idx) + c("int", "kTopoInv", robot.topo_inv)
        + c("int", "kAncestor", ancestors)
        + _index_fn("solved_idx", solved) + _index_fn("frozen_idx", frozen)
        + _index_fn("ee_joint", ee)
        + f"constexpr int n_robot_spheres = {len(sph_joint)};\n"
        + c("float", "kRobotSpheres", sph) + c("int", "kRobotSphereJoint", sph_joint)
        + f"constexpr int n_self_pairs = {len(sc.pair_i)};\n"
        + f"constexpr int n_self_links = {max(len(sc.link_start) - 1, 0)};\n"
        + c("float", "kSelfSph", sc.sph) + c("int", "kSelfLinkStart", sc.link_start)
        + c("int", "kSelfLinkJoint", sc.link_joint)
        + c("int", "kSelfPairI", sc.pair_i) + c("int", "kSelfPairJ", sc.pair_j)
        + "".join(f"constexpr int n_world_{k} = {int(n)};\n"
                  for k, n in zip(WORLD_KINDS, problem.world_counts))
        + _traj_tables(robot, problem.traj_collision)
        + "}  // namespace grim::robot\n")
    return "".join(out)


def _traj_tables(robot: MotionRobot, tc: TrajCollision | None) -> str:
    if tc is None:
        off, rad = np.zeros((robot.n_joints + 1, 1, 3)), -np.ones((robot.n_joints + 1, 1))
        pi = pj = np.zeros(0, np.int32)
    else:
        off, rad, pi, pj = (np.asarray(x) for x in (tc.sphere_off, tc.sphere_rad,
                                                    tc.pair_i, tc.pair_j))
    if off.shape[0] != robot.n_joints + 1:
        raise ValueError(f"TrajCollision has {off.shape[0]} rows; the robot needs "
                         f"n_joints + 1 = {robot.n_joints + 1}")
    return (f"constexpr int traj_N = {off.shape[0]};\nconstexpr int traj_S = {off.shape[1]};\n"
            f"constexpr int traj_P = {len(pi)};\n"
            + _constant("float", "kTrajSphereOff", off) + _constant("float", "kTrajSphereRad", rad)
            + _constant("int", "kTrajPairI", pi) + _constant("int", "kTrajPairJ", pj))


def kinematic_urdf(xml: str) -> str:
    """``xml`` without <visual>/<collision> geometry: tracing needs only the kinematic tree and
    inertias, and loading meshes would make it depend on files a URDF merely points to."""
    import re
    return re.sub(r"<(visual|collision)\b.*?</\1>|<(visual|collision)\b[^>]*/>", "", xml, flags=re.S)


def _traced_source(robot: MotionRobot, solved: list[int], frozen: list[int], ee: list[int],
                   fused_dls_ori_ratio: float | None = None) -> str:
    """crocket's straight-line FK (and with ``fused_dls_ori_ratio`` its DLS step), plus the maps
    between its joint order and ours."""
    try:
        import crocket
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("traced=True needs crocket (see install/motion_install.sh)") from exc
    # crocket traces the pose and Jacobian of ONE end-effector frame (any joint frame, fixed
    # ones included); with several end-effectors the build traces frame poses only
    # (FRAMES_ONLY) and each residual and Jacobian comes from those frames.
    target = int(ee[0]) if ee else deepest_actuated_joint(robot)
    with tempfile.TemporaryDirectory(prefix="grim_traced_") as tmp:
        urdf = Path(tmp) / "robot.urdf"
        urdf.write_text(kinematic_urdf(robot.urdf_xml))
        try:
            model = crocket.Robot(urdf, name="robot")
        except ValueError as exc:
            if "Unsupported joint type" in str(exc):
                raise ValueError("traced builds support revolute, prismatic and fixed joints "
                                 f"only (no continuous / planar / floating): {exc}") from exc
            raise
    names = model.joint_names
    if sorted(names) != sorted(robot.actuated_names):
        raise ValueError(f"crocket's joints {names} do not match the robot's actuated joints "
                         f"{list(robot.actuated_names)} (continuous/planar/floating joints are "
                         "not supported by the traced build)")
    algorithms = {}
    if fused_dls_ori_ratio is not None:
        if len(ee) != 1:
            raise ValueError("the fused DLS step traces one end-effector frame")
        algorithms["ik_lm_step"] = {"frame": robot.joint_names[ee[0]], "pos_weight": 1.0,
                                    "ori_weight": float(fused_dls_ori_ratio)}
    source = crocket.generate(model, algorithms, variants=[crocket.Variant("thread")], name="robot",
                              kinematics={
        "frames": list(robot.joint_names), "end_effector": robot.joint_names[target]})
    src = {a: f"cfg[{solved.index(a)}]" if a in solved else f"frz[{frozen.index(a)}]"
           for a in range(robot.n_act)}
    gather = "".join(f"    q[{i}] = {src[robot.actuated_names.index(n)]};\n"
                     for i, n in enumerate(names))
    scatter = "".join(
        f"    J[{row * len(solved) + solved.index(robot.actuated_names.index(n))}] = "
        f"Jt[{row * len(names) + i}];\n"
        for row in range(6) for i, n in enumerate(names)
        if robot.actuated_names.index(n) in solved)
    unscatter = "".join(f"    cfg[{solved.index(robot.actuated_names.index(n))}] = q[{i}];\n"
                        for i, n in enumerate(names) if robot.actuated_names.index(n) in solved)
    return (f"{source}\n#define GRIM_TRACED_KINEMATICS 1\n"
            + ("#define GRIM_TRACED_FRAMES_ONLY 1\n" if len(ee) > 1 else "")
            + ("#define GRIM_FUSED_DLS 1\n" if algorithms else "")
            + "namespace grim::robot {\n"
            "namespace traced = crocket::robots::robot;\n"
            "static __device__ __forceinline__ void gather_q(const float* __restrict__ cfg,\n"
            "    const float* __restrict__ frz, float* __restrict__ q)\n"
            f"{{\n{gather}    (void)frz;\n}}\n"
            "static __device__ __forceinline__ void scatter_cfg(const float* __restrict__ q,\n"
            "    float* __restrict__ cfg)\n"
            f"{{\n{unscatter}}}\n"
            "static __device__ __forceinline__ void scatter_jacobian(const float* __restrict__ Jt,\n"
            "    float* __restrict__ J)\n"
            f"{{\n{scatter}}}\n"
            "}  // namespace grim::robot\n")


def deepest_actuated_joint(robot: MotionRobot) -> int:
    return max((j for j in range(robot.n_joints) if robot.act_idx[j] != -1),
               key=lambda j: len(robot.chain(j)))


def cache_root() -> Path:
    env = os.environ.get("GRIM_MOTION_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "grim" / "motion"


def gpu_arch() -> str:
    return os.environ.get("GRIM_MOTION_ARCH", "-arch=native")


def compile_kernel(kernel: str, generated: dict[str, str], flags: tuple[str, ...] = ()) -> Path:
    """nvcc ``kernels/<kernel>.cu`` with the ``generated`` headers (name -> text) next to it.

    Cached on disk under a key over the kernel sources, GLASS, the generated headers and the
    flags, so each structure compiles once; concurrent builds of the same key are safe.
    """
    source = KERNELS_DIR / f"{kernel}.cu"
    flags = ["-O3", "-std=c++17", gpu_arch(), "--shared", "--compiler-options", "-fPIC",
             *flags]
    import jaxlib
    xla_include = Path(jaxlib.__file__).parent / "include"
    inputs = sorted(KERNELS_DIR.rglob("*.cu*")) + sorted(GLASS_DIR.rglob("*.cuh"))
    key = hashlib.sha1("\x00".join(
        [kernel, *(f"{k}\x01{v}" for k, v in sorted(generated.items())), " ".join(flags),
         str(xla_include),
         *(f"{p.relative_to(p.anchor)}\x01{p.read_text()}" for p in inputs)]).encode()).hexdigest()
    out_dir = cache_root() / key
    so = out_dir / f"{Path(kernel).name}.so"
    if so.is_file():
        return so

    cache_root().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root()))
    try:
        for name, text in generated.items():
            (tmp / name).write_text(text)
        cmd = ["nvcc", *flags, f"-I{tmp}", f"-I{KERNELS_DIR}", f"-I{GLASS_DIR}",
               f"-I{xla_include}", "-o", str(tmp / so.name), str(source)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            lines = result.stderr.splitlines()
            errors = [ln for i, ln in enumerate(lines)
                      if "error" in ln or any("error" in x for x in lines[max(i - 2, 0):i])]
            raise RuntimeError(f"nvcc failed to build {kernel}:\n" +
                               "\n".join(errors or lines[-40:]))
        try:
            tmp.rename(out_dir)
        except OSError:   # another process finished the same build first
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return so


def build(kernel: str, header: str, n_solve: int, n_frames: int, n_ee: int,
          extra_flags: tuple[str, ...] = ()) -> Path:
    """Compile a robot kernel against its ``grim_robot_gen.cuh`` (see :func:`robot_header`)."""
    return compile_kernel(kernel, {"grim_robot_gen.cuh": header},
                   (f"-DMAX_JOINTS={n_frames}", f"-DMAX_ACT={max(n_solve, 1)}",
                    f"-DMAX_EE={max(n_ee, 1)}", f"-DGRIM_SOLVE_N_BUCKETS(X)=X({max(n_solve, 1)})",
                    *extra_flags))


LIBRARIES: dict = {}   # FFI target name -> its loaded library (for host-side setters)


@lru_cache(maxsize=None)
def register(so: Path, symbols: tuple[str, ...]) -> tuple[str, ...]:
    """Register each FFI handler in ``so`` as a CUDA custom-call target; returns the names."""
    import jax
    lib = ctypes.CDLL(str(so))
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    tag = so.parent.name[:16]
    names = []
    for symbol in symbols:
        capsule = capsule_new(ctypes.cast(getattr(lib, symbol), ctypes.c_void_p),
                              b"xla._CUSTOM_CALL_TARGET", None)
        names.append(f"grim_{symbol}_{tag}")
        LIBRARIES[names[-1]] = lib
        jax.ffi.register_ffi_target(names[-1], capsule, platform="CUDA")
    return tuple(names)


def target(kernel: str, symbols: tuple[str, ...], robot: MotionRobot, problem: Problem,
           traced: bool = False, extra_flags: tuple[str, ...] = (),
           crocket_real: str | None = None) -> tuple[str, ...]:
    """Build (or load) ``kernel`` for ``robot``/``problem`` and return its FFI target names.
    ``crocket_real`` ("float" / "double") appends the crocket IK geometry (see _crocket_ik)."""
    header = robot_header(robot, problem, traced)
    solved = solved_joints(robot, problem)
    n_solve = len(solved)
    if crocket_real is not None:
        from . import _crocket_ik
        header += _crocket_ik.source(robot, problem, solved,
                                     [a for a in range(robot.n_act) if a not in solved])
        extra_flags = (*extra_flags, f"-DGRIM_CK_REAL={crocket_real}")
    if problem.runtime_rot_joint >= 0:
        if traced:
            raise ValueError("a runtime parent rotation needs the baked tables: crocket's "
                             "traced FK folds every transform into constants")
        extra_flags = (*extra_flags, f"-DGRIM_RUNTIME_ROT_JOINT={problem.runtime_rot_joint}")
    so = build(kernel, header, n_solve, robot.n_joints, len(problem.ee_joints), extra_flags)
    return register(so, symbols)
