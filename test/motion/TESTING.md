# Motion-kernel verification

How GRiM's motion kernels (IK, region IK, trajectory optimization, fused collision, compiled
least squares, C3+, traced dynamics) are tested, and why. It applies GRiD's equivalence
strategy (`external/GRiD/test/TESTING_STRATEGY.md`) to iterative solvers. Read that first; this file covers
what changes when a kernel is an optimizer rather than a closed-form algorithm.

## Two surfaces, three levels

GRiD's invariant is a float64 numpy oracle validated against Pinocchio, and CUDA validated
against the oracle. The motion oracle is `bindings/grim/motion/reference/`:

| module | what | validated against |
|---|---|---|
| `kinematics` | frame poses, pose residual, geometric Jacobian | Pinocchio (`test_reference_kinematics.py`, every asset robot, 1e-10) |
| `collision` | primitive distances, clearance map, link/pair minima, ESDF trilinear sampling | closed form |
| `ik` | LS-IK, transcribed step for step | via `kinematics` |
| `trajopt` | the four trajectory optimizers' reported costs | via `kinematics` + `collision` |

A final-answer comparison is weak for an iterative solver: it converges from many states, and
float32 round-off makes iterates drift chaotically after a few steps. So kernels are checked at
three levels:

1. **Primitives.** Everything a solver reads about the robot goes through `kernels/robot.cuh`
   (FK, residual, Jacobian) and `kernels/collision.cuh`. `test_kinematics_equivalence.py`
   checks them on every asset robot, for one and several end-effectors, over all joints and
   over the end-effector chains alone, from baked tables and from cricket traces.
2. **Steps.** Zero, one and two iterations against the transcribed reference (LS-IK), or the
   zero-iteration merit or cost (every IK solver and trajectory optimizer). The
   zero-iteration cost check catches a mis-posed collision sphere: the earlier sphere-row bug,
   where stock and traced kernels shared the error, would have failed it immediately.
3. **Certificates.** The oracle grades the solver's output, so no solver grades itself:
   - pose error on reachable targets;
   - reported error = the oracle's error of the returned configuration;
   - joint limits;
   - KKT stationarity (canonical IK);
   - box membership (region IK);
   - local optimality (compiled least squares);
   - the dense-QP solution (C3 Riccati).

Floors on success rates are measured, never guessed. Where a method itself misses targets, the
floor cites what pyroffi's original kernel achieved on the same problems (the parity gate
below).

## Invariances (GRiD Principle 2, adapted)

- **Thread count.** A tiered kernel's block tier must be bit-identical at 32, 64, 128 and a
  random odd block size. A missing barrier shows up at more than one warp, not at 32.
- **Tiers.** Thread, warp and block tiers run the same algorithm with differently ordered
  Cholesky factorizations (GLASS thread / warp / block `potrf`). One step therefore agrees to
  round-off; later iterates are covered by certificates.
- **Batch position.** A problem's answer may not depend on its slot or on the batch size.
  This catches a cooperative loop writing a thread-local array (the SQP `H_s` trap). It does
  not apply to MPPI and STOMP, which key their noise on the slot by design.
- **Determinism.** Every kernel is run twice and must give bit-identical outputs. No atomics
  or races may decide an output. The banded least-squares assembly was rewritten as
  owner-computes for this, and HJCD's cross-seed early stop is opt-in (`early_stop`).

## Samples (GRiD Principle 1, adapted)

`conftest.config_samples` covers:
- the zero pose and both limit corners;
- each joint alone at each limit;
- uniform random configurations.

IK targets come from FK of random configurations (`ik_cases`), including a zero-pose
(singular) and a limit-corner target. Dynamics adds high-velocity and high-acceleration
states. Jacobian checks use random configurations only, because axis-aligned ones create
sphere ties where a min is not differentiable.

## Tolerances (Principle 3)

`conftest.assert_close_scaled` floors `atol` at `rtol * max|expected|`. Each float32-specific
widening carries its measurement in a comment. Examples: float32 AL violation floors near
1.2e-3, and STOMP's soft-limit overshoot is 0.076 rad in both pyroffi's original and the port.

## Every build is a test subject

Kernels are compiled per robot and per problem structure, so a defect can live in one build
only. ptxas silently miscompiled G1's traced inverse-dynamics gradient at -O2/-O3. The suite
therefore builds every (robot, kernel, collision/structure, tier, dtype) cell it tests from
scratch through `grim.motion._build` (content-keyed cache), and nothing validates at runtime.

## The migration parity gate (one-time)

When a kernel moved from pyroffi, it was run on identical inputs against pyroffi's original
`.so`, with results recorded in `docs/source/user_guide/concepts/motion.rst`:
- At 1-10 iterations: agreement to float32 round-off.
- At convergence: equal solve rates, equal box rates, or equal costs.

That gate is not part of this suite, because GRiM does not depend on pyroffi.

## Adding a motion kernel

1. Read the robot only through `robot.cuh` / `collision.cuh` (or add the primitive there,
   with its oracle and a level-1 test).
2. Add or extend the oracle in `reference/`.
3. Cover levels 2 and 3, the invariances that apply, and determinism.
4. Route launches through a configurable block size if the kernel is tiered.
