# CLAUDE.md — agent & contributor onboarding

Orientation for an AI agent (or a new human) working in this repo. This file is tracked and
authoritative for conventions.

## What GRiM is

The motion-generation layer on top of GRiD: `bindings/grim/motion/` (IK, region IK, trajectory
optimization, fused collision, compiled least squares, C3+, traced dynamics). GRiD itself is the
`external/GRiD` submodule, used unmodified: it generates the per-robot rigid-body dynamics
(`grid_codegen` → `grid.cuh`, `grid::` namespace) and brings the GLASS / RBDReference /
URDFParser peers and the sample robots (`external/GRiD/config/robot_assets/`). Motion kernels use
their own `grim::` namespace; only `kernels/dynamics/generated_dynamics.cu` calls GRiD's generated
kernels. cricket (`external/cricket`) is the Pinocchio trace compiler behind `traced=True`.

**Do not edit `external/`.** A change GRiM needs in GRiD goes upstream to A2R-Lab/GRiD (cricket:
saiccoumar/cricket), then the submodule pointer is bumped and the motion suite re-run.

## Layout

- `bindings/grim/__init__.py` — puts the `external/GRiD` checkout on `sys.path` when GRiD is not
  installed (`GRIM_GRID_PATH` overrides the location).
- `bindings/grim/motion/` — kernels (`kernels/*.cu`, hand-written, GLASS-form thread / warp /
  block tiers) built per robot by `_build.py` (robot tables baked as `__constant__`, optional
  cricket traces), their JAX launchers, and the float64 oracle in `reference/`.
- `test/motion/` — read `test/motion/TESTING.md` before touching a kernel.
- `install/motion_install.sh` — conda-env install: cricket, GRiD and GRiM editable, jax[cuda13].

## Test

```bash
python -m pytest test/motion -q                       # needs a GPU, nvcc, jax[cuda]; cricket for traced cases
python -m pytest test/motion -m pinocchio_equivalence  # oracle vs Pinocchio (CPU)
```

## Durable engineering conventions

- **One problem never spans blocks.** Thread and warp tiers pack many problems into a block;
  independent per-column or elementwise work may spread across blocks because it never
  communicates.
- **Thread-count invariant** — a kernel's output must be identical at 1 / 32 / any thread count,
  and tier / batch / run-to-run invariant. Test it.
- **Two surfaces** — every kernel is validated against the float64 numpy oracle, which is
  validated against Pinocchio. Keep that invariant.
- **Fix, don't guard** — no `xfail`/`skip`/defensive guards; fix the root cause.
- **Physics**: gravity `-9.81`; Pinocchio is authoritative.
- **Validate every traced build.** Very large straight-line cricket traces have been miscompiled
  silently by ptxas at -O2/-O3.
- **Verify yourself** before committing; don't trust a subagent's "done".
- **Git**: short single-line commit messages, no Co-Authored-By footer; path-scoped `git add`
  (never `-A`/`.`); submodules committed/pushed before the parent pointer bump; push only when asked.
