"""crocket-traced dynamics (``grim.motion.dynamics``, thread and warp-split tiers) against
GRiD's generated block-cooperative kernels (``grim.motion.generated_dynamics``).

Both sides are JAX FFI calls on the same random float32 states, so call overhead matches;
times are the median of ``REPS`` jitted calls. Correctness is the test suite's job
(test/motion/test_traced_dynamics.py, test_generated_dynamics.py), not this script's.

    python test/benchmarks/bench_traced_vs_generated.py [robot ...] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from grim.motion import MotionRobot
from grim.motion import generated_dynamics
from grim.motion.dynamics import TracedDynamics

ASSETS = Path(__file__).resolve().parents[2] / "external" / "GRiD" / "grid_assets" / "robots"
BATCHES = (1, 16, 128, 1024, 16384)
REPS = 50
# traced op -> (generated op, generated output shape as a function of n)
PAIRS = {"id": ("id", lambda n: (n,)), "crba": ("crba", lambda n: (n, n)),
         "fd": ("fd", lambda n: (n,)), "fd_crba": ("fd", lambda n: (n,)),
         "id_du": ("id_grad", lambda n: (2 * n, n))}


def grid_header(urdf: Path) -> str:
    from grid_codegen import GRiDCodeGenerator
    from URDFParser import URDFParser
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        cwd = os.getcwd()
        try:
            os.chdir(d)
            GRiDCodeGenerator(URDFParser().parse(str(urdf)), False, False).gen_all_code()
        finally:
            os.chdir(cwd)
        return (Path(d) / "grid.cuh").read_text()


def median_ms(fn, *args) -> float:
    f = jax.jit(fn)
    f(*args).block_until_ready()
    times = []
    for _ in range(REPS):
        t = time.perf_counter()
        f(*args).block_until_ready()
        times.append(time.perf_counter() - t)
    return 1e3 * float(np.median(times))


def bench(name: str) -> list[dict]:
    robot = MotionRobot.from_urdf(str(ASSETS / f"{name}.urdf"))
    traced = TracedDynamics(robot)
    generated, _ = generated_dynamics.build(grid_header(ASSETS / f"{name}.urdf"))
    n = traced.n_q
    traced_fns = {"id": traced.inverse_dynamics, "crba": traced.mass_matrix,
                  "fd": traced.forward_dynamics,
                  "fd_crba": lambda q, qd, u, tier: traced.forward_dynamics(q, qd, u, tier, via_crba=True),
                  "id_du": traced.inverse_dynamics_gradient}
    rows = []
    for batch in BATCHES:
        rng = np.random.default_rng(0)
        q, qd, x = (jnp.asarray(rng.uniform(-1, 1, (batch, n)), jnp.float32) for _ in range(3))
        for op in traced.ops:
            gen_op, shape = PAIRS[op]
            gen_args = (q,) if gen_op == "crba" else (q, qd, x)
            call = jax.ffi.ffi_call(generated[gen_op], jax.ShapeDtypeStruct((batch, *shape(n)), jnp.float32))
            row = {"robot": name, "n": n, "batch": batch, "op": op,
                   "generated_ms": median_ms(lambda *a: call(*a, gravity=np.float32(-9.81)), *gen_args)}
            for tier in ("thread", "block"):
                fn = traced_fns[op]
                row[f"{tier}_ms"] = median_ms(lambda *a, fn=fn, tier=tier: fn(*a, tier=tier), *gen_args)
            best = min(row["thread_ms"], row["block_ms"])
            row["speedup"] = row["generated_ms"] / best
            rows.append(row)
            print(f"{name:8s} n={n:2d} B={batch:6d} {op:8s} GRiD {row['generated_ms']:8.3f} ms  "
                  f"thread {row['thread_ms']:8.3f}  block {row['block_ms']:8.3f}  "
                  f"traced/GRiD {row['speedup']:5.2f}x", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("robots", nargs="*", default=["iiwa14", "g1"])
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = [r for name in args.robots for r in bench(name)]
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
