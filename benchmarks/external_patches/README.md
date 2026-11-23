# external_patches

Patches to the pinned clones in `benchmarks/external/` (`clone.sh`) that expose internals the
libraries don't. Apply after cloning; the benchmarks themselves never need them.

- **jaxls** (pyroki's solver; `brentyi/jaxls` at `50a58be`, the commit pip installed into the `grim`
  env): `jaxls-record-iterates.patch` adds `jaxls.record_iterates()`, which records the variable
  values after every outer LM iteration (same trace-time switch as its `record_iteration_times()`).
  Used by `anneal_trajopt.py` for the convergence viewer (`view_trajopt.py`). Put the patched source
  first on `PYTHONPATH`; the installed copy stays untouched.

  ```bash
  git clone https://github.com/brentyi/jaxls.git benchmarks/external/jaxls
  git -C benchmarks/external/jaxls checkout 50a58be88c5ef74532f09e3f55268b4f02c490e3
  git -C benchmarks/external/jaxls apply ../../external_patches/jaxls-record-iterates.patch
  ```

## Debug capture for success-vs-time curves

Per-iteration capture of every trajopt method, so success can be graded along a solve from one
recorded run instead of a ladder of iteration budgets. Recording adds work: time with it off, record
in a separate pass.

```bash
bash benchmarks/external_patches/apply.sh          # jaxls copy + the cuRobo clone patch
bash benchmarks/external_patches/apply.sh revert
```

| method | how it records | one record |
|---|---|---|
| GRiM SCO / CHOMP / STOMP | `snap_every=k` on `grim.motion.trajopt.*` (GRiM's own kernels, no patch): written into a device output buffer inside the kernel | after every k-th iteration (SCO: inner step), what the kernel would return there and its cost; bit-identical to a run stopped at that budget (`test/motion/test_trajopt.py::test_debug_snapshots`) |
| pyroki (jaxls) | `jaxls_snapshots.py` writes a patched copy of the installed jaxls to `external/jaxls_snapshots` (put it first on `PYTHONPATH`): `jaxls.record_snapshots(k)` stacks the solution into `SolveSummary.snapshots` in the while-loop carry (device, no callback); `pyroki_trajopt.solve(snap_every=k)` | the solution a run with `max_iterations = k (s+1)` returns (jaxls counts LM tries). jaxls on the GPU is not bit-deterministic, so a recorded run is an equally valid run, not a copy of the timed one |
| cuRobo v2 | its own `store_debug` recorder; `curobo_debug_best.patch` adds `best_action` / `best_cost` (what it returns: `return_best_action`); `curobo_debug.py` attaches it and turns knots into positions. Build the planner with `use_cuda_graph=False` for the recorded pass | every L-BFGS iteration of every optimize() in the plan (trajopt, then finetune, whose cost weights differ), all seeds |
| Drake (SNOPT) | `drake_trajopt.solve(record=[])` (Drake's visualization callback, no patch) and `drake_trajopt.iterates()` | every SNOPT function evaluation, line-search trials included (Drake exposes no per-major-iteration hook), both solves, timestamped |
| TrajOpt (Tesseract) | `trajopt_harness` with a tenth argument (sco's optimizer callback, no patch) | the joint values at the start of every SQP iteration |

`jaxls_snapshots.py` checks its anchors and refuses anything but upstream 50a58be. It is
independent of `jaxls-record-iterates.patch` (a host callback per iteration, for the viewer).
