# Open tasks: GRiM

Things that have been found but not fixed yet. Add an entry when you find one, and delete it
when it's fixed (the fix commit is the record). Format: **title**: symptom / evidence / where /
suggested fix. Items marked _(verify)_ were seeded from earlier session notes and haven't been
re-checked against current code.

## Bugs

- **GLASS fix not in upstream GRiD**: the block-tier trsv/potrs fix for `__restrict__` callers
  is merged in GLASS (`9a24f37`), and `kernels/ik_mantis/ck.cuh` relies on it, but upstream
  GRiD still pins GLASS `9e57178`, and we can't push to GRiD. Until GRiD bumps it, check GLASS
  out by hand (`git -C external/GRiD/external/GLASS checkout 9a24f37`; see `AGENTS.md`).
  When GRiD's pointer moves, bump `external/GRiD` and delete the manual-checkout notes here, in
  `AGENTS.md`, and in pyroffi's `AGENTS.md` / `OPEN_TASKS.md`.

- **`collision.cuh` world normals are still finite differences**: `world_prim_grad_dir`
  central-differences `world_prim_dist` at h = 1e-4 in float32, while the IK kernels use the
  closed-form `n_prim` (`kernels/ik_mantis/ck.cuh`, checked against `reference/collision.py`'s
  `NORMALS`). Fix: move `n_prim` somewhere both include and use it here.
- **g1 trajopt cooperative tiers spill**: the warp/block tiers of `kernels/trajopt_mantis` are
  not chunked by `harness.fit` (only the thread trace is), so g1 warp still spills 232 B at budget
  125 and `fit` raises (`test/benchmarks/TRAJOPT_DYNAMICS_RESULTS.md`, analytic section). Fix:
  run the register guard per cooperative variant too, then add g1 to `test_trajopt.py`'s robots.
- **Trajopt analytic gradients have no direct check**: `test_trajopt.py` checks the 0-iteration
  cost against the float64 oracle and that solves lower it, but nothing compares
  `ck_hinge_grad` / `ck_smin_jac` against a float64 derivative of `reference/trajopt.py`. Fix: a
  per-waypoint FFI test hook, compared to central differences of the oracle in float64.

## Features

- **Full f32 / f64 support**: `backend="mantis-f64"` is double inside the kernel only; every
  launcher still takes and returns float32 (IK, trajopt, `MantisDynamics`), and pyroffi exposes
  no backend choice. Fix: float64 I/O end to end when the inputs are float64, and a `backend`
  option through pyroffi's solver configs.

## Performance

- **pyroffi_grid CRBA per-column ID** _(verify)_: hand-rolled kernel is 3.5–12× slower than
  GRiD's generated CRBA. Fix: call the `grid::crba` host wrapper.

## Housekeeping

- **Trajopt benchmark needs a rewrite**: `test/benchmarks/bench_mantis_trajopt.py` still runs
  `backend="grim"` and the FD prototype backends, both removed (tag `pre-crocket-trajopt`), and its
  success metric was faulty (0.00 on fr3/baxter/g1 for every backend).

- **cricket/mantis prototype branches** _(verify)_: tiers / sco-grad / multi-ee are local and
  unmerged. The tier rename breaks pyroffi.

- **C3+ (`kernels/trajopt/c3.cu`) is still hand-written**: deferred from the mantis migration.
