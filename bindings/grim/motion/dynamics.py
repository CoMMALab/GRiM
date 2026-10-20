"""Rigid-body dynamics from crocket traces, built per robot (``kernels/dynamics/
traced_dynamics.cu``).

crocket traces RNEA, CRBA, ABA and the analytical RNEA derivatives for one robot into
straight-line code with the robot's constants folded; GRiM compiles each op at two tiers,
complementing GRiM's own block-cooperative dynamics kernels:

* ``thread``: one configuration per thread (wins at large batch);
* ``block``: crocket's ``split`` tier: 32 configurations per block, the block's warps
  splitting each stage of the kinematic tree.

======== ======================================= =================================
op       computes                                 inputs -> output
======== ======================================= =================================
id       inverse dynamics (RNEA)                  q, qd, qdd -> tau
crba     mass matrix (CRBA)                       q -> M
fd       forward dynamics (ABA)                   q, qd, tau -> qdd
fd_crba  forward dynamics as M qdd = tau - b      q, qd, tau -> qdd (GLASS Cholesky)
id_du    inverse-dynamics derivatives             q, qd, qdd -> [dtau/dq | dtau/dqd]
======== ======================================= =================================

Arguments and results are float32 in the robot's actuated joint order. crocket splits
functions over its statement budget into ``__noinline__`` stages, the guard against ptxas
miscompiling very large straight-line traces; every build is still validated by the test
suite, and ``ptxas_opt`` lowers ptxas' level if one ever needs it.
"""

from __future__ import annotations

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
# op -> (crocket algorithm, inputs per joint, outputs as a function of n)
_TRACED = {
    "id": ("rnea", 3, lambda n: n),
    "crba": ("crba", 1, lambda n: n * n),
    "fd": ("aba", 3, lambda n: n),
    "id_du": ("rnea_derivatives", 3, lambda n: 2 * n * n),
}
OPS = ("id", "crba", "fd", "fd_crba", "id_du")
# ABA is not defined on mimic-coupled trees and Pinocchio's derivatives refuse mimic joints;
# forward dynamics on such robots goes through CRBA + Cholesky (fd_crba).
MIMIC_OPS = ("id", "crba", "fd_crba")


def _needed_traces(ops) -> set[str]:
    need = {o for o in ops if o in _TRACED}
    if "fd_crba" in ops:
        need |= {"id", "crba"}
    return need


def _crocket_traces(robot: MotionRobot, ops):
    import crocket

    with tempfile.TemporaryDirectory(prefix="grim_dynamics_") as tmp:
        urdf = Path(tmp) / "robot.urdf"
        urdf.write_text(_build.kinematic_urdf(robot.urdf_xml))
        try:
            model = crocket.Robot(urdf, name="robot")
        except ValueError as exc:
            if "Unsupported joint type" in str(exc):
                raise ValueError("traced dynamics supports revolute, prismatic and fixed joints "
                                 f"only (no continuous / planar / floating): {exc}") from exc
            raise
    programs = {op: crocket.trace(model, _TRACED[op][0]) for op in _needed_traces(ops)}
    return model, programs


def _wrappers(op: str, alg: str, n: int) -> str:
    """The contract's ``<op>_thread(x, y, ys)`` / ``<op>_block(rank, x, y, s, ys)`` over crocket's
    thread and split tiers; ``x`` packs the inputs, the outputs are consecutive in ``y``."""
    ins = ", ".join(f"x + {k * n}" for k in range(_TRACED[op][1]))
    outs = "y, y + %d * ys" % (n * n) if op == "id_du" else "y"
    return (f"__device__ __forceinline__ void {op}_thread(const float* x, float* y, int ys)\n"
            f"{{\n    R::{alg}_thread<float>({ins}, {outs}, ys);\n}}\n"
            f"__device__ __forceinline__ void {op}_block(int rank, const float* x, float* y, float* s, int ys)\n"
            f"{{\n    (void)rank;  // the split tier ranks by warp itself\n"
            f"    R::{alg}_split<float>({ins}, {outs}, s, ys);\n}}\n")


def contract_header(model, programs: dict, n: int, ops: tuple[str, ...]) -> str:
    """Every op in ``ops`` at thread and block tier, in crocket's joint order."""
    import crocket

    variants = [crocket.Variant("thread"), crocket.Variant("split", BLOCK_WARPS)]
    sections = [crocket.cuda.emit(programs[op], variants) for op in ("id", "crba", "fd", "id_du")
                if op in programs]
    out = ["// GRiM traced dynamics, generated. Do not edit.\n",
           crocket.cuda.header(model, sections, name="robot"),
           '#include "glass.cuh"\n',
           f"namespace grim::traced {{\nconstexpr int contract_version = {CONTRACT_VERSION};\n"
           f"constexpr int n_q = {n};\n}}\n",
           "namespace grim::traced::ops {\nnamespace R = crocket::robots::robot;\n"]
    scratch, entries = {}, []
    for op in ("id", "crba", "fd", "id_du"):
        if op not in programs:
            continue
        alg, per_joint, n_out = _TRACED[op]
        sc = crocket.schedule.build(programs[op], variants[1].units, variants[1].threshold)
        scratch[op] = crocket.cuda.smem(sc, "split")
        out.append(_wrappers(op, alg, n))
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
// Mass matrix and bias by the split tier into per-lane scratch, then warp 0's lanes each
// solve their own configuration.
__device__ __forceinline__ void fd_crba_block(int rank, const float* x, float* y, float* s, int ys)
{{
    const int lane = threadIdx.x & 31;
    float xb[{3 * n}];
    for (int i = 0; i < {2 * n}; ++i) xb[i] = x[i];
    for (int i = 0; i < {n}; ++i) xb[{2 * n} + i] = 0.f;
    crba_block(rank, x, s + {m_off} + lane, s, 32);
    id_block(rank, xb, s + {b_off} + lane, s + {scratch["crba"]}, 32);
    __syncthreads();  // M and b were written by every warp; crocket's tiers end without a barrier
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
    ``"block"``. Gravity is Pinocchio's ``-9.81`` along world z.
    """

    def __init__(self, robot: MotionRobot, ops: tuple[str, ...] | None = None,
                 ptxas_opt: int | None = None):
        has_mimic = bool((robot.mimic_act_idx != -1).any())
        if ops is None:
            ops = MIMIC_OPS if has_mimic else OPS
        if has_mimic and set(ops) - set(MIMIC_OPS):
            raise ValueError(f"ops {sorted(set(ops) - set(MIMIC_OPS))} are not supported for "
                             "robots with mimic joints; use fd via_crba=True")
        self.actuated = list(robot.actuated_names)
        self.n_q = n = robot.n_act
        model, programs = _crocket_traces(robot, ops)
        names = model.joint_names
        if sorted(names) != sorted(self.actuated):
            raise ValueError(f"crocket joints {names} do not match actuated joints {self.actuated}")
        self._to_gen = np.array([self.actuated.index(nm) for nm in names])   # generator <- ours
        self._from_gen = np.argsort(self._to_gen)
        self.ops = ops
        flags = (f"-Xptxas=-O{ptxas_opt}",) if ptxas_opt is not None else ()
        so = _build.compile_kernel("dynamics/traced_dynamics",
                                   {"traced_dynamics_gen.cuh": contract_header(model, programs, n, ops)}, flags)
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
