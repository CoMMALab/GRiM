"""Compiled least-squares solvers on hand-written problem headers (no problem compiler needed).

* Dense LM / augmented Lagrangian at every tier and precision: a constrained Rosenbrock
  problem. LM is a local method, so the certificate is local optimality: the oracle lists
  the local minimizers of the cost restricted to the equality line (plus the inequality
  bound when it is active) by a fine 1-D search, and the solve must land on one of them.
  Runs are bit-deterministic.
* Banded LM (trajectory structure, one block per problem): a linear chain least-squares
  problem with one fixed variable, against numpy's exact least-squares solution;
  bit-deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

# min 100 (x1 - x0^2)^2 + (1 - x0)^2   s.t.  x0 + x1 = p0,  x0 <= p1
DENSE = r"""
namespace grim::costs_gen {
constexpr int n_x = 2, n_r = 4, scratch_size = 0;
constexpr bool has_constraints = true;
__host__ __device__ constexpr int row_kind(int i) { return i < 2 ? 0 : (i == 2 ? 1 : 2); }
__device__ inline void residual(const real* x, const io_t* p, real* r, real*, int, int)
{
    r[0] = real(10) * (x[1] - x[0] * x[0]);
    r[1] = real(1) - x[0];
    r[2] = x[0] + x[1] - real(p[0]);
    r[3] = x[0] - real(p[1]);
}
__device__ inline void residual_jacobian(const real* x, const io_t* p, real* r, real* J, real* s,
                                         int rank, int size)
{
    residual(x, p, r, s, rank, size);
    J[0] = real(-20) * x[0]; J[1] = real(10);
    J[2] = real(-1);         J[3] = real(0);
    J[4] = real(1);          J[5] = real(1);
    J[6] = real(1);          J[7] = real(0);
}
}  // namespace grim::costs_gen
"""

SETTINGS = dict(max_iters=200, al_iters=20, lambda_initial=1e-3, tolerance=1e-10,
                al_tolerance=1e-7, grad_tolerance=1e-9, grad_start=10, param_tolerance=1e-10)


def _attrs(**kw):
    s = dict(SETTINGS, **kw)
    return {k: (np.int32(v) if isinstance(v, int) else np.float32(v)) for k, v in s.items()}


def _run(target, x0, params, n_x, ws_bytes, dt):
    import jax
    import jax.numpy as jnp
    B = x0.shape[0]
    return [np.asarray(v) for v in jax.ffi.ffi_call(target, (
        jax.ShapeDtypeStruct((B, n_x), dt), jax.ShapeDtypeStruct((B,), dt),
        jax.ShapeDtypeStruct((B,), dt), jax.ShapeDtypeStruct((B, ws_bytes), jnp.uint8)))(
        jnp.asarray(x0, dt), jnp.asarray(params, dt), **_attrs())]


def _dense_local_minima(c, u):
    """Local minimizers of f(x0) = 100 (c - x0 - x0^2)^2 + (1 - x0)^2 on x0 <= u (the
    feasible set of the equality line), including the bound if f decreases into it."""
    x0 = np.linspace(-4, u, 4_000_001)
    f = 100 * ((c - x0) - x0 ** 2) ** 2 + (1 - x0) ** 2
    interior = np.nonzero((f[1:-1] < f[:-2]) & (f[1:-1] <= f[2:]))[0] + 1
    mins = [x0[k] for k in interior] + ([u] if f[-1] < f[-2] else [])
    return [np.array([m, c - m]) for m in mins]


@pytest.mark.parametrize("tier", ["thread", "warp", "block"])
@pytest.mark.parametrize("real", ["float", "double"])
def test_dense_constrained_rosenbrock(tier, real):
    import jax.numpy as jnp
    from grim.motion.costs import dense_solver
    target = dense_solver(DENSE, tier, real=real, io="float")
    rng = np.random.default_rng(0)
    params = np.stack([rng.uniform(0.5, 2.0, 16), rng.uniform(0.2, 1.5, 16)], 1)
    x0 = rng.uniform(-1.5, 1.5, size=(16, 2))
    x, cost, viol = _run(target, x0, params, 2, 8, jnp.float32)[:3]
    tol = 2e-3 if real == "float" else 1e-4
    for b in range(16):
        dist = min(np.abs(x[b] - m).max() for m in _dense_local_minima(*params[b]))
        assert dist < tol, (b, x[b], _dense_local_minima(*params[b]), params[b])
        # float32 compute floors the violation near 1.2e-3 (penalties reach 1e7).
        assert viol[b] < (2e-3 if real == "float" else 1e-4)
    again = _run(target, x0, params, 2, 8, jnp.float32)
    assert np.array_equal(again[0], x) and np.array_equal(again[1], cost)


def _chain_header(n, fixed_var):
    """x_{i+1} - x_i - p_i = 0 (cost rows, instances of group 0 over (x_i, x_{i+1})) and
    x_0 - p_{n-1} (group 1), with variable `fixed_var` held at its initial value."""
    vtab = [v for i in range(n - 1) for v in (i, i + 1)] + [0]
    ptab = list(range(n - 1)) + [n - 1]
    arr = lambda xs: ", ".join(map(str, xs))
    fixed = [1 if j == fixed_var else 0 for j in range(n)]
    return f"""
namespace grim::costs_gen {{
constexpr int n_x = {n}, n_r = {n}, band = 1, n_groups = 2;
constexpr bool has_constraints = false;
__device__ constexpr int row_kind[{n}] = {{{arr([0] * n)}}};
__device__ constexpr int fixed[{n}] = {{{arr(fixed)}}};
__device__ constexpr int vtab[{len(vtab)}] = {{{arr(vtab)}}};
__device__ constexpr int ptab[{len(ptab)}] = {{{arr(ptab)}}};
__device__ constexpr int g_nloc[2] = {{2, 1}};
__device__ constexpr int g_nploc[2] = {{1, 1}};
__device__ constexpr int g_nr[2] = {{1, 1}};
__device__ constexpr int g_vtab0[2] = {{0, {2 * (n - 1)}}};
__device__ constexpr int g_ptab0[2] = {{0, {n - 1}}};
__device__ constexpr int g_row0[2] = {{0, {n - 1}}};
__device__ constexpr int g_j0[2] = {{0, {2 * (n - 1)}}};
__device__ constexpr int g_inst_prefix[2] = {{0, {n - 1}}};
__device__ constexpr int g_col_prefix[2] = {{0, {2 * (n - 1)}}};
constexpr int n_inst_total = {n}, n_col_total = {2 * (n - 1) + 1};
constexpr int max_nloc = 2, max_nploc = 1, max_nr = 1, j_size = {2 * (n - 1) + 1};
__device__ inline void stage_res(int g, const real* xl, const real* pl, real* r)
{{
    r[0] = g == 0 ? xl[1] - xl[0] - pl[0] : xl[0] - pl[0];
}}
__device__ inline void stage_jvp(int g, const real* xl, const real* pl, const real* t, real* r,
                                 real* jt)
{{
    stage_res(g, xl, pl, r);
    jt[0] = g == 0 ? t[1] - t[0] : t[0];
}}
}}  // namespace grim::costs_gen
"""


def test_banded_chain():
    import jax.numpy as jnp
    from grim.motion.costs import banded_solver
    n, B, fixed_var = 40, 6, 17
    target, ws = banded_solver(_chain_header(n, fixed_var), real="double", io="float")
    rng = np.random.default_rng(1)
    params = rng.normal(size=(B, n))
    x0 = rng.normal(size=(B, n))
    x, cost, _ = _run(target, x0, params, n, ws, jnp.float32)[:3]
    for b in range(B):
        # Rows: x_{i+1} - x_i = p_i (i < n-1) and x_0 = p_{n-1}; column `fixed_var` held at x0.
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i], A[i, i + 1] = -1, 1
        A[n - 1, 0] = 1
        rhs = params[b] - A[:, fixed_var] * np.float32(x0[b, fixed_var])
        free = [j for j in range(n) if j != fixed_var]
        want = np.full(n, np.float32(x0[b, fixed_var]), float)
        want[free] = np.linalg.lstsq(A[:, free], rhs, rcond=None)[0]
        assert x[b, fixed_var] == np.float32(x0[b, fixed_var])
        assert np.abs(x[b] - want).max() < 1e-4, np.abs(x[b] - want).max()
    again = _run(target, x0, params, n, ws, jnp.float32)
    assert np.array_equal(again[0], x) and np.array_equal(again[1], cost), "not deterministic"


POINTWISE = r"""
#pragma once
namespace grim::pointwise_gen {
constexpr int n_in = 2, n_out = 3, scratch_size = 0;
__device__ __forceinline__ void pointwise(const real* __restrict__ x, real* __restrict__ o,
                                          real* __restrict__, const int)
{
    o[0] = x[0] * x[1];
    o[1] = sin(x[0]) + x[1];
    o[2] = x[0] - x[1];
}
}  // namespace grim::pointwise_gen
"""


def test_pointwise():
    import jax
    import jax.numpy as jnp
    from grim.motion.costs import pointwise
    target = pointwise(POINTWISE)
    x = np.random.default_rng(3).normal(size=(37, 2))
    with jax.enable_x64(True):
        o = np.asarray(jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((37, 3), jnp.float64))(
            jnp.asarray(x)))
    want = np.stack([x[:, 0] * x[:, 1], np.sin(x[:, 0]) + x[:, 1], x[:, 0] - x[:, 1]], 1)
    assert np.allclose(o, want, rtol=1e-14, atol=1e-14)
