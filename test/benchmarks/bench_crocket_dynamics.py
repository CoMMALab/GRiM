"""Every dynamics op GRiD generates (``grim.motion.generated_dynamics``, block-cooperative) against
crocket (``grim.motion.dynamics.CrocketDynamics``) in float and double at every tier (one
configuration per thread / warp / block).

Both sides are jitted JAX FFI calls on the same random float32 states. Per (robot, op, batch):

* time: ms per call, median of REPS runs of K back-to-back calls (a run lasts about MIN_RUN_S);
* err: max |out - Pinocchio| / max |Pinocchio| over the first SCORE_MAX states (float64 oracle).

GRiD's ``fd`` (through its M^-1) is the baseline of both crocket ``fd`` (ABA) and ``fd_crba``
(CRBA + Cholesky); ``id_du`` / ``fd_du`` are GRiD's ``id_grad`` / ``fd_grad``. ``idsva_so``
(second-order) has no crocket counterpart and is not run.

    python test/benchmarks/bench_crocket_dynamics.py <robot> [--out results/dynamics_<robot>.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from bench_traced_vs_generated import ASSETS, grid_header  # noqa: E402
from motion.test_traced_dynamics import _pin  # noqa: E402

from grim.motion import MotionRobot, generated_dynamics  # noqa: E402
from grim.motion.dynamics import CrocketDynamics  # noqa: E402

BATCHES = (128, 16384)
REPS, MIN_RUN_S = 5, 0.2
SCORE_MAX = 16
# crocket op -> GRiD op
GRID_OP = {"id": "id", "crba": "crba", "fd": "fd", "fd_crba": "fd", "id_du": "id_grad",
           "minv": "minv", "fd_du": "fd_grad"}


def timed(fn, *args) -> float:
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    k = max(1, int(MIN_RUN_S / max(time.perf_counter() - t0, 1e-6)))
    runs = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(k):
            out = fn(*args)
        jax.block_until_ready(out)
        runs.append((time.perf_counter() - t0) / k)
    return 1e3 * float(np.median(runs))


def pinocchio_ref(model, data, iq, iv, op, q, qd, x):
    qp, vp, ap = np.zeros(model.nq), np.zeros(model.nv), np.zeros(model.nv)
    qp[iq], vp[iv], ap[iv] = q, qd, x
    ix = np.ix_(iv, iv)
    if op == "id":
        return pin.rnea(model, data, qp, vp, ap)[iv]
    M = pin.crba(model, data, qp)
    M = np.triu(M) + np.triu(M, 1).T
    if op == "crba":
        return M[ix]
    if op == "minv":
        return np.linalg.inv(M[ix])
    if op == "fd_crba":   # M^-1 (tau - b): also defined with mimic joints, unlike ABA
        return np.linalg.solve(M[ix], x - pin.rnea(model, data, qp, vp, 0 * ap)[iv])
    if op == "fd":
        return pin.aba(model, data, qp, vp, ap)[iv]
    if op == "id_du":
        dq, dv, _ = pin.computeRNEADerivatives(model, data, qp, vp, ap)
        return np.hstack([dq[ix], dv[ix]])
    pin.computeABADerivatives(model, data, qp, vp, ap)
    return np.hstack([data.ddq_dq[ix], data.ddq_dv[ix]])


def grid_call(targets, op, n, B, to_gen, from_gen):
    """GRiD's op as (B, ...) in the actuated order, shaped like CrocketDynamics' output. GRiD's
    joint order is its URDF parser's, which is crocket's (``to_gen`` / ``from_gen``)."""
    gop = GRID_OP[op]
    shape = {"id": (B, n), "fd": (B, n), "crba": (B, n, n), "minv": (B, n, n),
             "id_grad": (B, 2 * n, n), "fd_grad": (B, 2 * n, n)}[gop]
    call = jax.ffi.ffi_call(targets[gop], jax.ShapeDtypeStruct(shape, jnp.float32))

    g = from_gen

    def f(*a):
        kw = {} if gop == "minv" else {"gravity": np.float32(-9.81)}
        y = call(*(v[:, to_gen] for v in a), **kw)
        if gop in ("crba", "minv"):   # [col, row]
            y = jnp.swapaxes(y, 1, 2)
            if gop == "minv":       # upper triangle only
                y = jnp.triu(y) + jnp.swapaxes(jnp.triu(y, 1), 1, 2)
            return y[:, g][:, :, g]
        if gop in ("id_grad", "fd_grad"):   # (2n cols, n rows) -> (n, 2n)
            y = jnp.swapaxes(y, 1, 2)[:, g]
            return jnp.concatenate([y[:, :, :n][:, :, g], y[:, :, n:][:, :, g]], axis=-1)
        return y[:, g]
    return f


def bench(name: str) -> list[dict]:
    urdf = ASSETS / f"{name}.urdf"
    robot = MotionRobot.from_urdf(str(urdf))
    n = robot.n_act
    model, data, iq, iv = _pin(robot)
    grid, _ = generated_dynamics.build(grid_header(urdf))
    ck = {b: CrocketDynamics(robot, b) for b in ("crocket-f32", "crocket-f64")}
    rows = []
    for B in BATCHES:
        rng = np.random.default_rng(0)
        q = rng.uniform(robot.lower, robot.upper, (B, n))
        qd, x = rng.uniform(-1, 1, (B, n)), rng.uniform(-5, 5, (B, n))
        args32 = [jnp.asarray(a, jnp.float32) for a in (q, qd, x)]
        q32, qd32, x32 = (np.asarray(a, float) for a in args32)
        for op in ck["crocket-f32"].ops:
            want = np.array([pinocchio_ref(model, data, iq, iv, op, q32[b], qd32[b], x32[b])
                             for b in range(SCORE_MAX)])
            scale = max(np.abs(want).max(), 1e-12)
            a = args32[:1] if op in ("crba", "minv") else args32
            d0 = ck["crocket-f32"]
            cfgs = [("grid", "block", jax.jit(grid_call(grid, op, n, B, d0._to_gen, d0._from_gen)))]
            cfgs += [(b, t, jax.jit(lambda *v, d=d, op=op, t=t: d(op, *v, tier=t)))
                     for b, d in ck.items() for t in ("thread", "warp", "block")]
            for backend, tier, fn in cfgs:
                row = dict(robot=name, n_q=n, op=op, batch=B, backend=backend, tier=tier)
                try:
                    ms = timed(fn, *a)
                    got = np.asarray(fn(*a))[:SCORE_MAX]
                except Exception as exc:   # noqa: BLE001 - an op a tier cannot run is a result
                    rows.append({**row, "error": str(exc)[:300]})
                    print(f"{name} {op} B={B} {backend} {tier}: {str(exc)[:120]}", flush=True)
                    continue
                err = float(np.abs(got - want).max() / scale)
                rows.append({**row, "ms": ms, "err": err})
                print(f"{name:7s} n={n:2d} {op:8s} B={B:6d} {backend:11s} {tier:6s} {ms:9.3f} ms  "
                      f"err {err:.1e}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("robot")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = bench(args.robot)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
