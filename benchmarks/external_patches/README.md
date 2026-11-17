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

No patch is needed for the others: cuRobo records iterates with its own `store_debug=True`,
TrajOpt through the solver profile's `callbacks` (`trajopt_harness.cpp`'s `iterates` argument),
and GRiM and Drake (a wheel) by truncated solves.
