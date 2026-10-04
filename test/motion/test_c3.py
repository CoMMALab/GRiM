"""C3+ inner-solve kernels (float64) vs numpy.

* Projection onto complementarity (pa >= 0, pb >= 0, pa pb = 0, nearest in the w_eta metric)
  and the box clamp, elementwise.
* Riccati: the multiple-shooting LQ subproblem (dx_0 = 0, dx_{k+1} = A dx_k + Bt w_k + c_k,
  Gauss-Newton stage costs plus the ADMM consensus rows) against the dense solution of the
  same QP, eliminated onto the inputs w.
The PCG / ADMM / PGS / merit operators are covered by pyroffi's tests against its JAX C3
reference (benchmarks/c3_push/c3.py); a numpy oracle for them is still to be written.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

NX, NU, NL, NT = 3, 1, 2, 4
M = NU + NL


def _x64():
    import jax
    return jax.enable_x64(True)


def test_projection_and_box():
    import jax
    import jax.numpy as jnp
    from grim.motion.costs import c3
    t, _ = c3(NX, NU, NL, NT)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    w_eta = 2.5
    with _x64():
        pa, pb = (np.asarray(v) for v in jax.ffi.ffi_call(t["C3ProjectFfi"], (
            jax.ShapeDtypeStruct((200,), jnp.float64), jax.ShapeDtypeStruct((200,), jnp.float64)))(
            jnp.asarray(a), jnp.asarray(b), w_eta=np.float64(w_eta)))
        lo, hi = -0.3 * np.ones(5), 0.4 * np.ones(5)
        x = rng.normal(size=(40, 5))
        box = np.asarray(jax.ffi.ffi_call(t["C3BoxFfi"], jax.ShapeDtypeStruct(x.shape, jnp.float64))(
            jnp.asarray(x), jnp.asarray(lo), jnp.asarray(hi)))
    assert np.all(pa >= 0) and np.all(pb >= 0) and np.all(pa * pb == 0)
    cost = lambda u, v: (a - u) ** 2 + w_eta * (b - v) ** 2
    best = np.minimum(cost(np.maximum(a, 0), 0), cost(0, np.maximum(b, 0)))
    assert np.allclose(cost(pa, pb), best, atol=1e-14)
    assert np.array_equal(box, np.clip(x, lo, hi))


def _lq(rng, B, N):
    lin = dict(A=rng.normal(scale=0.4, size=(B, N, NX, NX)) + np.eye(NX),
               Bt=rng.normal(size=(B, N, NX, M)), E=rng.normal(size=(B, N, NL, NX)),
               Ht=rng.normal(size=(B, N, NL, M)), Eta=rng.normal(size=(B, N, NL)),
               Lam=rng.normal(size=(B, N, NL)), c=rng.normal(scale=0.1, size=(B, N, NX)),
               Jx=rng.normal(size=(B, N, NT, NX)), Ju=rng.normal(size=(B, N, NT, M)),
               r=rng.normal(size=(B, N, NT)))
    return lin


def _dense_lq(L, b, N, s2, s2w, tl, te):
    """Minimize sum_k 1/2 s_k^T G_k s_k + g_k^T s_k, s_k = [dx_k | w_k], over w with dx from
    the dynamics, by eliminating dx = S w + d and solving the normal equations."""
    nW = N * M
    S = np.zeros((N, NX, nW))
    d = np.zeros((N, NX))
    for k in range(N - 1):
        A, Bt = L["A"][b, k], L["Bt"][b, k]
        S[k + 1] = A @ S[k]
        S[k + 1][:, k * M:(k + 1) * M] += Bt
        d[k + 1] = A @ d[k] + L["c"][b, k]
    H, h = np.zeros((nW, nW)), np.zeros(nW)
    for k in range(N):
        J = np.hstack([L["Jx"][b, k], L["Ju"][b, k]])
        E = np.hstack([L["E"][b, k], L["Ht"][b, k]])
        G = J.T @ J + s2w * E.T @ E
        G[NX + NU:, NX + NU:] += s2 * np.eye(NL)
        g = J.T @ L["r"][b, k] + s2w * E.T @ (L["Eta"][b, k] + te[b, k])
        g[NX + NU:] += s2 * (L["Lam"][b, k] + tl[b, k])
        P = np.zeros((NX + M, nW))
        P[:NX] = S[k]
        P[NX:, k * M:(k + 1) * M] = np.eye(M)
        q = np.r_[d[k], np.zeros(M)]
        H += P.T @ G @ P
        h += P.T @ (G @ q + g)
    w = np.linalg.solve(H, -h).reshape(N, M)
    dx = np.einsum("kiw,w->ki", S, w.reshape(-1)) + d
    return dx, w


def test_riccati_matches_dense_qp():
    import jax
    import jax.numpy as jnp
    from grim.motion.costs import c3
    t, lib = c3(NX, NU, NL, NT)
    B, N, rho, w_eta = 3, 6, 4.0, 0.7
    rng = np.random.default_rng(2)
    L = _lq(rng, B, N)
    tl, te = rng.normal(size=(B, N, NL)), rng.normal(size=(B, N, NL))
    ws = lib.c3_workspace_doubles(N, 0, 0) * B
    with _x64():
        dX, W, _ = (np.asarray(v) for v in jax.ffi.ffi_call(t["C3RiccatiFfi"], (
            jax.ShapeDtypeStruct((B, N, NX), jnp.float64), jax.ShapeDtypeStruct((B, N, M), jnp.float64),
            jax.ShapeDtypeStruct((ws,), jnp.float64)))(
            *(jnp.asarray(L[k]) for k in ("A", "Bt", "E", "Ht", "Eta", "Lam", "c", "Jx", "Ju", "r")),
            jnp.asarray(tl), jnp.asarray(te), rho=np.float64(rho), w_eta=np.float64(w_eta)))
    s2 = 0.5 * rho
    for b in range(B):
        dx_ref, w_ref = _dense_lq(L, b, N, s2, s2 * w_eta, tl, te)
        assert np.allclose(dX[b], dx_ref, atol=1e-7), np.abs(dX[b] - dx_ref).max()
        assert np.allclose(W[b], w_ref, atol=1e-7), np.abs(W[b] - w_ref).max()
