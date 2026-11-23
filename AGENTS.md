# AGENTS.md: GRiM

Orientation for an AI agent (or a new human) working in this repo. This file is tracked and
authoritative for conventions.

## What GRiM is

The motion-generation layer on top of GRiD: `bindings/grim/motion/` (IK, region IK, trajectory
optimization, fused collision, C3+, mantis dynamics). The generic compiled least-squares
solvers live in the standalone `cudals` package, not here. GRiD itself is the
`external/GRiD` submodule, used unmodified: it generates the per-robot rigid-body dynamics
(`grid_codegen` → `grid.cuh`, `grid::` namespace) and brings the GLASS / RBDReference /
URDFParser peers and the sample robots (`external/GRiD/config/robot_assets/`). Motion kernels use
their own `grim::` namespace; only `kernels/dynamics/generated_dynamics.cu` calls GRiD's generated
kernels. mantis (`external/mantis`) is the trace compiler behind every IK kernel
(`kernels/ik_mantis`) and trajectory optimizer (`kernels/trajopt_mantis`), `traced=True` and
`grim.motion.dynamics` (`MantisDynamics`). `generated_dynamics` still wraps GRiD for what
mantis lacks: floating base, runtime inertia and the second-order `idsva_so`.

**Do not edit `external/`.** A change GRiM needs in GRiD goes upstream to A2R-Lab/GRiD, then the
submodule pointer is bumped and the motion suite re-run.

**Manual GLASS checkout (temporary).** The mantis IK block tier needs GLASS's restricted-caller
fix (GLASS `9a24f37`), but upstream GRiD still pins GLASS `9e57178`. After
`git submodule update --init --recursive`, run:

```bash
git -C external/GRiD/external/GLASS checkout 9a24f37
```

Without it, block-tier SQP with collision silently diverges from the thread tier. Remove this
note once GRiD bumps GLASS (tracked in `OPEN_TASKS.md`).

## Layout

- `bindings/grim/__init__.py` — puts the `external/GRiD` checkout on `sys.path` when GRiD is not
  installed (`GRIM_GRID_PATH` overrides the location).
- `bindings/grim/motion/` — kernels (`kernels/*.cu`, hand-written, GLASS-form thread / warp /
  block tiers) built per robot by `_build.py` (robot tables baked as `__constant__`, optional
  mantis traces), their JAX launchers, and the float64 oracle in `reference/`. IK is
  `kernels/ik_mantis` only: the solver loop in CUDA, everything it reads about the robot traced
  by mantis's `ik_geometry`, float or double (`backend="mantis-f32"` / `"mantis-f64"`). The
  hand-written IK kernels it replaced are at tag `pre-crocket-ik` (a local copy may sit in the
  untracked `legacy/`). Trajopt (SCO, STOMP, CHOMP) is `kernels/trajopt_mantis` only, the same
  way: mantis traces the collision terms with their analytic gradients
  (`grim.motion._mantis_traj`). The hand-written trajopt kernels, and the least-squares one
  (now a cudals problem in pyroffi), are at tag `pre-crocket-trajopt`.
- `test/motion/` — read `test/motion/TESTING.md` before touching a kernel.
- `benchmarks/` — the suite against external libraries (pRRTC, cuRobo, pyroki, GRiD, frax,
  Pinocchio, RBDL, TrajOpt, Drake): FK + collision, IK, dynamics, trajopt, timing GRiM's kernels
  built on mantis traces; moved here from mantis. `bash benchmarks/run_all.sh` (which calls
  `run_kinematics.sh`, `run_dynamics.sh`, `run_trajopt.sh`; shared setup in `bench_env.sh`);
  setup from a fresh clone in `benchmarks/RUNNING.md`.
- `install/motion_install.sh` — conda-env install: mantis, GRiD and GRiM editable, jax[cuda13].

## Test

```bash
python -m pytest test/motion -q                       # needs a GPU, nvcc, jax[cuda] and mantis
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
- **Validate every traced build.** Very large straight-line mantis traces have been miscompiled
  silently by ptxas at -O2/-O3.
- **Verify yourself** before committing; don't trust a subagent's "done".
- **Git**: short single-line commit messages, no Co-Authored-By footer; path-scoped `git add`
  (never `-A`/`.`); submodules committed/pushed before the parent pointer bump; push only when asked.
