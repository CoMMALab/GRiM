"""Rigid-body dynamics from crocket traces, built per robot (``kernels/dynamics/
ck_dynamics.cu``).

crocket traces RNEA, CRBA, ABA and the analytical RNEA derivatives for one robot into
straight-line code with the robot's constants folded; GRiM compiles each at crocket's thread,
warp and block variants (one configuration per thread, warp or block), in float or double.

======== ======================================= =================================
op       computes                                 inputs -> output
======== ======================================= =================================
id       inverse dynamics (RNEA)                  q, qd, qdd -> tau
crba     mass matrix (CRBA)                       q -> M
fd       forward dynamics (ABA)                   q, qd, tau -> qdd
fd_crba  forward dynamics as M qdd = tau - b      q, qd, tau -> qdd (Cholesky)
id_du    inverse-dynamics derivatives             q, qd, qdd -> [dtau/dq | dtau/dqd]
minv     inverse mass matrix                      q -> M^-1
fd_du    forward-dynamics derivatives             q, qd, tau -> [dqdd/dq | dqdd/dqd]
======== ======================================= =================================

Arguments and results are float32 in the robot's actuated joint order. crocket splits
functions over its statement budget into ``__noinline__`` stages, the guard against ptxas
miscompiling very large straight-line traces; every build is still validated by the test
suite. ABA and Pinocchio's derivatives are not defined on mimic-coupled trees, so mimic robots
get ``CK_MIMIC_OPS`` only.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import _build
from .robot import MotionRobot

# op -> (kernel enum, crocket algorithms it calls)
_CK_OPS = {"id": ("ID", ("rnea",)), "crba": ("CRBA", ("crba",)), "fd": ("FD", ("aba",)),
           "fd_crba": ("FD_CRBA", ("crba", "rnea")), "id_du": ("ID_DU", ("rnea_derivatives",)),
           "minv": ("MINV", ("crba",)),
           "fd_du": ("FD_DU", ("aba", "rnea_derivatives", "crba"))}
CK_OPS = tuple(_CK_OPS)
CK_MIMIC_OPS = ("id", "crba", "fd_crba", "minv")
CK_BACKENDS = {"crocket-f32": "float", "crocket-f64": "double"}
CK_TIERS = {"thread": 0, "warp": 1, "block": 2}


class CrocketDynamics:
    """Dynamics of one robot from crocket traces (``kernels/dynamics/ck_dynamics.cu``), in float
    or double (``backend``), one configuration per thread, warp or block (``tier``): every op of
    GRiD's generated dynamics but its second-order tensors (``idsva_so``).

    Methods take ``(B, n_q)`` arrays (actuated order) and return float32; gravity is
    Pinocchio's ``-9.81`` along world z. ``fd_du`` is GRiD's ``fd_grad``: ``[dqdd/dq | dqdd/dqd]``
    at ``qdd = fd(q, qd, tau)``.
    """

    def __init__(self, robot: MotionRobot, backend: str = "crocket-f32",
                 ops: tuple[str, ...] | None = None, block_threads: int = 64):
        import crocket
        if backend not in CK_BACKENDS:
            raise ValueError(f"unknown backend {backend!r}: one of {sorted(CK_BACKENDS)}")
        has_mimic = bool((robot.mimic_act_idx != -1).any())
        ops = ops or (CK_MIMIC_OPS if has_mimic else CK_OPS)
        if has_mimic and set(ops) - set(CK_MIMIC_OPS):
            raise ValueError(f"ops {sorted(set(ops) - set(CK_MIMIC_OPS))} are not supported for "
                             "robots with mimic joints")
        self.actuated = list(robot.actuated_names)
        self.n_q = n = robot.n_act
        with tempfile.TemporaryDirectory(prefix="grim_dynamics_") as tmp:
            urdf = Path(tmp) / "robot.urdf"
            urdf.write_text(_build.kinematic_urdf(robot.urdf_xml))
            model = crocket.Robot(urdf, name="robot")
        if sorted(model.joint_names) != sorted(self.actuated):
            raise ValueError(f"crocket joints {model.joint_names} do not match actuated joints "
                             f"{self.actuated}")
        self._to_gen = np.array([self.actuated.index(nm) for nm in model.joint_names])
        self._from_gen = np.argsort(self._to_gen)
        algs = sorted({a for op in ops for a in _CK_OPS[op][1]})
        src = crocket.generate(model, {a: {} for a in algs}, name="robot", variants=[
            crocket.Variant("thread"), crocket.Variant("warp", units=32),
            crocket.Variant("block", units=block_threads)])
        smem = " ".join(f"R::{a}_warp_smem, R::{a}_block_smem," for a in algs)
        header = (f"{src}\nnamespace grim::dyn {{\nnamespace R = crocket::robots::robot;\n"
                  f"constexpr int N = {n};\nconstexpr int BLOCK_THREADS = {block_threads};\n"
                  f"constexpr int SMEM_ALL[] = {{{smem}}};\n"
                  "constexpr int smem_max() { int m = 1; for (int v : SMEM_ALL) m = v > m ? v : m; return m; }\n"
                  "constexpr int SMEM = smem_max();\n}\n"
                  + "".join(f"#define GRIM_CK_HAS_{a}\n" for a in algs) +
                  "#define GRIM_CK_DYN_OPS(X) " + " ".join(f"X({op}, {_CK_OPS[op][0]})" for op in ops)
                  + "\n")
        so = _build.compile_kernel("dynamics/ck_dynamics", {"ck_dynamics_gen.cuh": header},
                                   (f"-DGRIM_CK_REAL={CK_BACKENDS[backend]}",))
        self.ops = ops
        self._targets = dict(zip(ops, _build.register(so, tuple(f"CkDyn_{op}" for op in ops))))

    def __call__(self, op: str, *args, tier: str = "thread"):
        n = self.n_q
        args = [jnp.asarray(a, jnp.float32).reshape(-1, n) for a in args]
        batch = args[0].shape[0]
        packed = jnp.concatenate([a[:, self._to_gen] for a in args], axis=1)
        n_out = {"id": n, "fd": n, "fd_crba": n, "crba": n * n, "minv": n * n,
                 "id_du": 2 * n * n, "fd_du": 2 * n * n}[op]
        res = jax.ffi.ffi_call(self._targets[op], jax.ShapeDtypeStruct((n_out, batch), jnp.float32))(
            packed, tier=np.int64(CK_TIERS[tier])).T
        g = self._from_gen
        if op in ("id", "fd", "fd_crba"):
            return res[:, g]
        if op in ("crba", "minv"):
            M = jnp.swapaxes(res.reshape(batch, n, n), 1, 2)   # column-major -> [row, col]
            return M[:, g][:, :, g]
        D = jnp.swapaxes(res.reshape(batch, 2, n, n), 2, 3)[:, :, g][:, :, :, g]
        return jnp.concatenate([D[:, 0], D[:, 1]], axis=-1)
