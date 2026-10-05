"""Rigid-body dynamics from cricket traces, built per robot (``kernels/dynamics/
traced_dynamics.cu``).

cricket traces Pinocchio's algorithms for one robot into straight-line code; GRiM compiles
each op at two tiers, complementing GRiM's own block-cooperative dynamics kernels:

* ``thread``: one configuration per thread (wins at large batch);
* ``block``: the same trace scheduled by ``cricket.tiered`` (warp-split): 32 configurations
  per block, the block's warps splitting each dataflow level.

======== ======================================= =================================
op       computes                                 inputs -> output
======== ======================================= =================================
id       inverse dynamics (RNEA)                  q, qd, qdd -> tau
crba     mass matrix (CRBA)                       q -> M
fd       forward dynamics (ABA)                   q, qd, tau -> qdd
fd_crba  forward dynamics as M qdd = tau - b      q, qd, tau -> qdd (GLASS Cholesky)
id_du    inverse-dynamics derivatives             q, qd, qdd -> [dtau/dq | dtau/dqd]
======== ======================================= =================================

Arguments and results are float32 in the robot's actuated joint order. Large derivative
traces can be miscompiled by ptxas at -O2/-O3 (CUDA 13.3, G1's id_du); every build is
validated by the test suite, and ``ptxas_opt`` lowers ptxas' level for such robots.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from .robot import MotionRobot

CONTRACT_VERSION = 2
BLOCK_WARPS = 8
_SMEM_LIMIT = 96 * 1024
# op -> (cricket GenOptions flag, inputs per joint, outputs as a function of n)
_TRACED = {
    "id": ("inverse_dynamics", 3, lambda n: n),
    "crba": ("mass_matrix", 1, lambda n: n * n),
    "fd": ("forward_dynamics", 3, lambda n: n),
    "id_du": ("inverse_dynamics_derivatives", 3, lambda n: 2 * n * n),
}
OPS = ("id", "crba", "fd", "fd_crba", "id_du")
# cricket's ABA crashes on models with mimic joints and its derivative tracer refuses them;
# forward dynamics on such robots goes through CRBA + Cholesky (fd_crba).
MIMIC_OPS = ("id", "crba", "fd_crba")


def _needed_traces(ops) -> set[str]:
    need = {o for o in ops if o in _TRACED}
    if "fd_crba" in ops:
        need |= {"id", "crba"}
    return need


def _cricket_traces(robot: MotionRobot, ops) -> tuple[dict, list[str]]:
    import cricket

    with tempfile.TemporaryDirectory(prefix="grim_dynamics_") as tmp:
        urdf, srdf = Path(tmp) / "robot.urdf", Path(tmp) / "robot.srdf"
        urdf.write_text(_build.kinematic_urdf(robot.urdf_xml))
        srdf.write_text(_build._EMPTY_SRDF)
        # Name the end-effector: cricket's own pick of a distal link can crash.
        ee = _build._child_link(robot, _build.deepest_actuated_joint(robot))
        opts = cricket.GenOptions(urdf=urdf, srdf=srdf, end_effector=ee, language="cuda",
                                  data={"name": "Robot"})
        need = _needed_traces(ops)
        for op, (flag, *_) in _TRACED.items():
            setattr(opts, flag, op in need)
        try:
            gen = cricket.generate_robot_source(opts)
        except RuntimeError as exc:
            if "Unsupported joint type" in str(exc):
                raise ValueError("traced dynamics supports revolute, prismatic and fixed joints "
                                 f"only (no continuous / planar / floating): {exc}") from exc
            raise
    return gen.data, list(gen.data["joint_names"])


def _thread_fn(name: str, code: str) -> str:
    n_v = max([int(v) for v in re.findall(r"\bv\[(\d+)\]", code)] + [0]) + 1
    body = re.sub(r"\by\[(\d+)\]", r"y[\1 * ys]", code)
    return (f"__device__ __forceinline__ void {name}_thread(const float* x, float* y, int ys)\n"
            f"{{\n    float v[{n_v}];\n{body}\n}}\n")


def _block_fn(name: str, code: str) -> tuple[str, int]:
    from cricket import tiered

    sched = tiered.schedule(code, BLOCK_WARPS)
    src = tiered.emit(sched, f"{name}_block", "warp", y_stride="ys")
    return src.replace("float* s)", "float* s, int ys)", 1), tiered.smem_floats(sched, "warp")


def contract_header(data: dict, n: int, ops: tuple[str, ...]) -> str:
    """Every op in ``ops`` at thread and block tier, in cricket's joint order."""
    out = ["#pragma once\n// GRiM traced dynamics, generated. Do not edit.\n",
           '#include "glass.cuh"\n',
           f"namespace grim::traced {{\nconstexpr int contract_version = {CONTRACT_VERSION};\n"
           f"constexpr int n_q = {n};\n}}\n",
           "namespace grim::traced::ops {\n"]
    scratch, entries = {}, []
    for op in ("id", "crba", "fd", "id_du"):
        if op not in ops and not (op in ("id", "crba") and "fd_crba" in ops):
            continue
        flag, per_joint, n_out = _TRACED[op]
        code = data[f"{flag}_code"]
        block, floats = _block_fn(op, code)
        scratch[op] = floats
        out += [_thread_fn(op, code), block]
        if op in ops:
            entries.append((op, per_joint * n, n_out(n)))
    if "fd_crba" in ops:
        # Forward dynamics as M(q) qdd = tau - b(q, qd), b = RNEA(q, qd, 0), by GLASS Cholesky.
        m_off = scratch["crba"] + scratch["id"]
        b_off = m_off + 32 * n * n
        scratch["fd_crba"] = b_off + 32 * n
        out.append(f"""__device__ __forceinline__ void fd_crba_solve(float* M, float* b)
{{
    int fail = 0;
    glass::thread::potrf<float, {n}, true>(M, &fail);
    glass::thread::potrs<float, {n}>(M, b);
}}
__device__ __forceinline__ void fd_crba_thread(const float* x, float* y, int ys)
{{
    float M[{n * n}], xb[{3 * n}], b[{n}];
    crba_thread(x, M, 1);
    for (int i = 0; i < {2 * n}; ++i) xb[i] = x[i];
    for (int i = 0; i < {n}; ++i) xb[{2 * n} + i] = 0.f;
    id_thread(xb, b, 1);
    for (int i = 0; i < {n}; ++i) b[i] = x[{2 * n} + i] - b[i];
    fd_crba_solve(M, b);
    for (int i = 0; i < {n}; ++i) y[i * ys] = b[i];
}}
// Mass matrix and bias by the warp-split tier into per-lane scratch, then warp 0's lanes each
// solve their own configuration.
__device__ __forceinline__ void fd_crba_block(int rank, const float* x, float* y, float* s, int ys)
{{
    const int lane = threadIdx.x & 31;
    float xb[{3 * n}];
    for (int i = 0; i < {2 * n}; ++i) xb[i] = x[i];
    for (int i = 0; i < {n}; ++i) xb[{2 * n} + i] = 0.f;
    crba_block(rank, x, s + {m_off} + lane, s, 32);
    id_block(rank, xb, s + {b_off} + lane, s + {scratch["crba"]}, 32);
    if (rank != 0) return;
    float M[{n * n}], b[{n}];
    for (int i = 0; i < {n * n}; ++i) M[i] = s[{m_off} + i * 32 + lane];
    for (int i = 0; i < {n}; ++i) b[i] = x[{2 * n} + i] - s[{b_off} + i * 32 + lane];
    fd_crba_solve(M, b);
    for (int i = 0; i < {n}; ++i) y[i * ys] = b[i];
}}
""")
        entries.append(("fd_crba", 3 * n, n))
    for op, floats in scratch.items():
        out.append(f"constexpr int {op}_scratch = {floats};\n"
                   f"constexpr bool {op}_smem = {'true' if floats * 4 <= _SMEM_LIMIT else 'false'};\n"
                   f"constexpr int {op}_warps = {BLOCK_WARPS};\n")
    out.append("}  // namespace grim::traced::ops\n#define GRIM_CONTRACT_OPS(X) "
               + " ".join(f"X({op}, {nin}, {nout})" for op, nin, nout in entries) + "\n")
    return "".join(out)


class TracedDynamics:
    """Dynamics of one robot at the thread and warp-split tiers.

    Methods take ``(B, n_q)`` arrays (actuated order) and ``tier`` ``"thread"`` or
    ``"block"``. Gravity is cricket's (Pinocchio's) ``-9.81`` along world z.
    """

    def __init__(self, robot: MotionRobot, ops: tuple[str, ...] | None = None,
                 ptxas_opt: int | None = None):
        has_mimic = bool((robot.mimic_act_idx != -1).any())
        if ops is None:
            ops = MIMIC_OPS if has_mimic else OPS
        if has_mimic and set(ops) - set(MIMIC_OPS):
            raise ValueError(f"ops {sorted(set(ops) - set(MIMIC_OPS))} are not supported for "
                             "robots with mimic joints (cricket); use fd via_crba=True")
        self.actuated = list(robot.actuated_names)
        self.n_q = n = robot.n_act
        data, names = _cricket_traces(robot, ops)
        if sorted(names) != sorted(self.actuated):
            raise ValueError(f"cricket joints {names} do not match actuated joints {self.actuated}")
        self._to_gen = np.array([self.actuated.index(nm) for nm in names])   # generator <- ours
        self._from_gen = np.argsort(self._to_gen)
        self.ops = ops
        flags = (f"-Xptxas=-O{ptxas_opt}",) if ptxas_opt is not None else ()
        so = _build.compile_kernel("dynamics/traced_dynamics",
                                   {"traced_dynamics_gen.cuh": contract_header(data, n, ops)}, flags)
        symbols = tuple(f"Contract_{op}_{tier}" for op in ops for tier in ("thread", "block"))
        self._targets = dict(zip(((op, tier) for op in ops for tier in ("thread", "block")),
                                 _build.register(so, symbols)))

    def inverse_dynamics(self, q, qd, qdd, tier: str = "thread"):
        return self._run("id", tier, q, qd, qdd)

    def mass_matrix(self, q, tier: str = "thread"):
        return self._run("crba", tier, q)

    def forward_dynamics(self, q, qd, tau, tier: str = "thread", via_crba: bool = False):
        return self._run("fd_crba" if via_crba else "fd", tier, q, qd, tau)

    def inverse_dynamics_gradient(self, q, qd, qdd, tier: str = "thread"):
        """``[dtau/dq | dtau/dqd]``, shape ``(B, n, 2n)``, rows the output joint."""
        return self._run("id_du", tier, q, qd, qdd)

    def _run(self, op: str, tier: str, *args):
        n = self.n_q
        args = [jnp.asarray(a, jnp.float32).reshape(-1, n) for a in args]
        batch = args[0].shape[0]
        packed = jnp.concatenate([a[:, self._to_gen] for a in args], axis=1)
        n_out = {"id": n, "fd": n, "fd_crba": n, "crba": n * n, "id_du": 2 * n * n}[op]
        res = jax.ffi.ffi_call(self._targets[op, tier],
                               jax.ShapeDtypeStruct((n_out, batch), jnp.float32))(packed).T
        g = self._from_gen
        if op in ("id", "fd", "fd_crba"):
            return res[:, g]
        if op == "crba":
            M = jnp.swapaxes(res.reshape(batch, n, n), 1, 2)   # column-major -> [row, col]
            return M[:, g][:, :, g]
        D = jnp.swapaxes(res.reshape(batch, 2, n, n), 2, 3)[:, :, g][:, :, :, g]
        return jnp.concatenate([D[:, 0], D[:, 1]], axis=-1)
