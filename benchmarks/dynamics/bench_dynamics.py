"""Dynamics: inverse dynamics (RNEA), mass matrix (CRBA), forward dynamics (ABA) and the inverse
dynamics derivatives d tau / d(q, qd), batched, gravity -9.81 z, on GRiD's release protocol.

Every cell runs GRiD's release worker (``external/grid/test/benchmarks/release/worker.py``)
through ``grid_protocol.py``: independent float64 Pinocchio oracle (strict policy), warmups plus a
sustained warm period, ``ITERATIONS`` timed samples, post-timing and repeatability checks, and
``REPEATS`` separate processes per cell, as GRiD's ``collect.py`` runs them. Accuracy policy
``fp32-fd-warnings``, as GRiD's release table uses it (``timing_night.sh``): a float32 fd cell
outside the strict entrywise gate but within relative L2 1e-3 is kept as ``accuracy_warning``.

* ``grid_cuda``  GRiD's generated kernels through GRiD's C++ bridge (``grid_rbd``, external/grid)
* ``grid_jax``   the same kernels through ``grid_rbd``'s JAX FFI surface
* ``mantis``    mantis-traced kernels (GRiM's ``MantisDynamics``) through JAX, fastest tier per batch
* ``mantis_cuda`` the same kernel through a C bridge (``mantis_bridge.cu``) timed like GRiD's
                 kernel bridge: no Python or JAX in the timed path
* ``mantis_cuda_fwd_ad`` / ``mantis_cuda_rev_ad``  id_du only: the C bridge with id_du from mantis's
                 ahead-of-time forward- / reverse-mode AD of the RNEA trace (``MantisDynamics(id_du=...)``)
                 in place of its analytical RNEA derivatives
* ``frax``       frax (external/frax) under ``jax.jit(jax.vmap(...))``; id_du is ``jax.jacfwd``
* ``pinocchio``  Pinocchio (external/pinocchio, built with CppADCodeGen) through GRiD's codegen
                 bridge; CPU, persistent thread pool, best thread count per batch
* ``rbdl``       RBDL (external/rbdl) through ``rbdl_bridge.cpp``, GRiD's CPU pool; float64; no id_du

Times per call: ``resident`` (inputs and outputs on the device; GPU methods) and
``host`` (host arrays in and out). GRiD's batches plus 1 and 16384.

    python dynamics/bench_dynamics.py <prepare|grid_cuda|grid_jax|mantis|mantis_cuda[_fwd_ad|_rev_ad]|frax|pinocchio|rbdl> [robot ...]

Run in the ``grid-bench`` env (see RUNNING.md for the environment variables).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

GRID_BATCHES = (16, 32, 64, 128, 256, 1024)  # GRiD's release batches
EXTRA_BATCHES = (1, 16384)
BATCHES = tuple(sorted(GRID_BATCHES + EXTRA_BATCHES))
OPS = {"id": "inverse_dynamics", "crba": "crba", "fd": "forward_dynamics", "id_du": "inverse_dynamics_gradient"}
METHODS = ("grid_cuda", "grid_jax", "mantis", "mantis_cuda", "mantis_cuda_fwd_ad", "mantis_cuda_rev_ad", "frax",
           "pinocchio", "rbdl")
ID_DU_ONLY = ("mantis_cuda_fwd_ad", "mantis_cuda_rev_ad")   # differ from mantis_cuda only in id_du
UNAVAILABLE = {("rbdl", "id_du"): "RBDL has no inverse dynamics derivatives"}
REPEATS, ITERATIONS, WARMUPS = 3, 30, 5  # GRiD collect.py's defaults
if os.environ.get("BENCH_FAST"):   # run_all.sh --fast: one process per cell, four batches
    GRID_BATCHES, REPEATS, ITERATIONS, WARMUPS = (128, 1024), 1, 10, 2
    BATCHES = tuple(sorted(GRID_BATCHES + EXTRA_BATCHES))
CPU_THREADS = int(os.environ.get("BENCH_CPU_THREADS", os.cpu_count() or 1))


def prepare(robot: str) -> None:
    import pinocchio as pin
    urdf = common.MODEL_URDF[robot]
    m = common.model(urdf)
    d = m.createData()
    rng = np.random.default_rng(3)
    q = common.configs(urdf, max(BATCHES), seed=3)
    qd, qdd = (rng.uniform(-1, 1, q.shape).astype(np.float32) for _ in range(2))
    tau = np.array([pin.rnea(m, d, *(v[b].astype(np.float64) for v in (q, qd, qdd))) for b in range(len(q))],
                   np.float32)
    path = common.inputs_path("dynamics", robot)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, joint_names=np.array(common.joint_names(urdf)), q=q, qd=qd, qdd=qdd, tau=tau)


def _median(cells, key):
    xs = [c[key]["median_us"] for c in cells if key in c]
    return statistics.median(xs) / 1e3 if xs else None


def run(method: str, robot: str, op: str) -> list[dict]:
    """GRiD's worker, ``REPEATS`` processes; one row per batch (medians over repeats, in ms)."""
    if (method, op) in UNAVAILABLE:
        return [{"batch": B, "op": op, "ms": None, "status": "unavailable", "note": UNAVAILABLE[method, op]}
                for B in BATCHES]
    caps, crashes = [], {}
    with tempfile.TemporaryDirectory(prefix="dyn_") as tmp:
        for repeat in range(REPEATS):
            # GRiD's batches in one process, as collect.py runs them; each extra batch in its own,
            # so a native crash (GRiD's bridge segfaults when its workspace does not fit) only
            # loses that cell.
            for group in (GRID_BATCHES, *((b,) for b in EXTRA_BATCHES)):
                out = Path(tmp) / f"r{repeat}-{group[0]}.json"
                cmd = [sys.executable, str(HERE / "grid_protocol.py"), "--robot", robot, "--backend", method,
                       "--operation", OPS[op], "--batches", *map(str, group), "--iterations", str(ITERATIONS),
                       "--warmups", str(WARMUPS), "--accuracy-policy", "fp32-fd-warnings", "--cpu-threads", str(CPU_THREADS), "--output", str(out)]
                log = subprocess.run(cmd, capture_output=True, text=True)
                if out.exists():
                    caps.append(json.loads(out.read_text()))
                else:
                    oom = "out of memory" in log.stderr or "RESOURCE_EXHAUSTED" in log.stderr
                    for b in group:
                        crashes[b] = "OOM" if oom else f"worker died (rc {log.returncode}): {log.stderr[-200:]}"
    rows = []
    for B in BATCHES:
        cells = [c for cap in caps for c in cap["cells"] if c["batch"] == B]
        ok = [c for c in cells if c.get("comparison_eligible")]
        if B in crashes or len(ok) < len(cells) or not ok:
            reasons = sorted({c.get("reason", c.get("status", "?")) for c in cells} - {None})
            if B in crashes:
                reasons.insert(0, crashes[B])
            oom = any("OOM" in r or "out of memory" in r.lower() or "RESOURCE_EXHAUSTED" in r for r in reasons)
            rows.append({"batch": B, "op": op, "ms": None, "status": "oom" if oom else "failed",
                         "note": "; ".join(reasons)[:300]})
            continue
        resident, host = _median(ok, "resident"), _median(ok, "host_to_host")
        err = max(b["relative_l2"] for c in ok for b in c["oracle_agreement"]["blocks"])
        a = ok[0]["adapter"]
        variant = a.get("tier") or (f"{ok[0].get('selected_threads')} threads" if "selected_threads" in ok[0] else "")
        rows.append({"batch": B, "op": op, "ms": resident if resident is not None else host,
                     "resident_ms": resident, "host_ms": host, "variant": variant, "err": err,
                     "dtype": a.get("dtype"), "status": "accuracy_warning" if any(c["status"] == "accuracy_warning" for c in ok) else "validated", "repeats": len(ok)})
    return rows


def main():
    method, robots = sys.argv[1], sys.argv[2:] or list(common.ROBOTS)
    if method == "prepare":
        for robot in robots:
            prepare(robot)
        return
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}: one of {METHODS}")
    rows = []
    for robot in robots:
        for op in (("id_du",) if method in ID_DU_ONLY else OPS):
            for row in run(method, robot, op):
                rows.append({"method": method, "robot": robot, **row})
                t = (f"{row['ms']:10.4f} ms ({row['variant']}, host {row['host_ms']:.4f})  rel_l2 {row['err']:.1e}"
                     if row["ms"] is not None else f"{row['status']}: {row.get('note', '')}")
                print(f"{method:9s} {robot:7s} B={row['batch']:6d} {op:6s} {t}", flush=True)
            common.save(f"dynamics_{method}_{'_'.join(robots)}", rows)


if __name__ == "__main__":
    main()
