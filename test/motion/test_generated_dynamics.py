"""Any-batch JAX handlers around a GRiM-generated header (``grim.motion.generated_dynamics``).

The kernels are GRiM's own generated dynamics; what this module adds is the handler layer
(packing, scratch workspace, output layouts). So every op is checked against the float64
RBDReference oracle on a header generated from the same parsed model: values directly, and
the gradient ops against central differences of the oracle. Also: determinism, a batch larger
than grim.jax's default max_batch, and the runtime inertia table (baseline reproduces the
baked kernels; a change moves the result; a reset restores it bit for bit).
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import numpy as np
import pytest

from .conftest import ASSETS, assert_close_scaled, requires_gpu

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

NAME = "iiwa14"
B = 300          # above grim.jax's default max_batch (256)


def _parse():
    from URDFParser import URDFParser
    return URDFParser().parse(str(ASSETS / f"{NAME}.urdf"))


def _header(runtime_inertia=False):
    from grim_codegen.GRiMCodeGenerator import GRiMCodeGenerator
    gen = GRiMCodeGenerator(_parse(), False, False, FILE_NAMESPACE="grim")
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        cwd = os.getcwd()
        try:
            os.chdir(d)
            gen.gen_all_code(runtime_inertia=runtime_inertia)
        finally:
            os.chdir(cwd)
        return open(os.path.join(d, "grim.cuh")).read()


@pytest.fixture(scope="module")
def built():
    from grim.motion.generated_dynamics import build
    targets, so = build(_header())
    return targets


def _states(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, (B, n)).astype(np.float32) for _ in range(3)]


def _call(target, out_shape, *args, gravity=-9.81):
    import jax
    import jax.numpy as jnp
    attrs = {} if gravity is None else {"gravity": np.float32(gravity)}
    return np.asarray(jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(out_shape, jnp.float32))(
        *(jnp.asarray(a) for a in args), **attrs))


def _oracle():
    from RBDReference import RBDReference
    return RBDReference(_parse())


def test_values_match_the_oracle(built):
    ref = _oracle()
    n = len(_parse().get_joints_ordered_by_id())
    q, qd, x = _states(n)
    tau = _call(built["id"], (B, n), q, qd, x)
    qdd = _call(built["fd"], (B, n), q, qd, x)
    M = np.swapaxes(_call(built["crba"], (B, n, n), q), 1, 2)
    Mi_u = np.tril(_call(built["minv"], (B, n, n), q, gravity=None))  # [col, row], row <= col
    Mi = np.swapaxes(Mi_u + np.triu(np.swapaxes(Mi_u, 1, 2), 1), 1, 2)
    for b in range(0, B, 37):
        d = [a[b].astype(np.float64) for a in (q, qd, x)]
        assert_close_scaled(tau[b], ref.inverse_dynamics(*d)[0], 1e-4, f"id b={b}")
        assert_close_scaled(qdd[b], ref.forward_dynamics(*d), 1e-3, f"fd b={b}")
        assert_close_scaled(M[b], ref.crba(d[0]), 1e-4, f"crba b={b}")
        assert_close_scaled(Mi[b], ref.minv(d[0]), 1e-3, f"minv b={b}")


def test_gradients_match_oracle_differences(built):
    ref = _oracle()
    n = len(_parse().get_joints_ordered_by_id())
    q, qd, x = _states(n, seed=1)
    G = {op: np.swapaxes(_call(built[op], (B, 2 * n, n), q, qd, x), 1, 2)
         for op in ("id_grad", "fd_grad")}
    f = {"id_grad": lambda *a: ref.inverse_dynamics(*a)[0], "fd_grad": ref.forward_dynamics}
    h = 1e-6
    for b in range(0, B, 61):
        d = [a[b].astype(np.float64) for a in (q, qd, x)]
        for op, fn in f.items():
            want = np.zeros((n, 2 * n))
            for k in range(2):           # d/dq, then d/dqd
                for j in range(n):
                    p, m = [v.copy() for v in d], [v.copy() for v in d]
                    p[k][j] += h
                    m[k][j] -= h
                    want[:, k * n + j] = (fn(*p) - fn(*m)) / (2 * h)
            assert_close_scaled(G[op][b], want, 1e-3, f"{op} b={b}")


def test_deterministic(built):
    n = len(_parse().get_joints_ordered_by_id())
    q, qd, x = _states(n, seed=2)
    for op in ("id", "fd", "id_grad"):
        shape = (B, n) if op in ("id", "fd") else (B, 2 * n, n)
        a = _call(built[op], shape, q, qd, x)
        assert np.array_equal(a, _call(built[op], shape, q, qd, x)), op


def test_runtime_inertia_table(built):
    import ctypes

    from grim.motion.generated_dynamics import build
    targets, so = build(_header(runtime_inertia=True), runtime_inertia=True)
    lib = ctypes.CDLL(str(so))
    lib.DynSetInertiaParams.argtypes = [ctypes.POINTER(ctypes.c_float)]
    n = len(_parse().get_joints_ordered_by_id())
    assert lib.DynInertiaParamsSize() == 10 * n
    base = np.ascontiguousarray(
        np.asarray(_parse().get_inertia_params_ordered_by_id()[1:], np.float32).reshape(-1))
    upload = lambda p: lib.DynSetInertiaParams(p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    q, qd, x = _states(n, seed=3)
    baked = _call(built["id"], (B, n), q, qd, x)
    upload(base)
    first = _call(targets["id"], (B, n), q, qd, x)
    assert_close_scaled(first, baked, 1e-5, "runtime table at the URDF's values")
    heavier = base.copy()
    heavier[10 * (n - 1)] *= 2.0          # double the last link's mass
    upload(heavier)
    assert np.abs(_call(targets["id"], (B, n), q, qd, x) - first).max() > 1e-3
    upload(base)
    assert np.array_equal(_call(targets["id"], (B, n), q, qd, x), first)
