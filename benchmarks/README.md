# Benchmarks

FK, IK and dynamics of mantis against the libraries it competes with, on three robots: Franka
Panda (7 DOF), Rethink Baxter (14/15 DOF, two arms) and Unitree G1 with hands (43 DOF tree).

```bash
bash benchmarks/run_all.sh        # picks a free GPU, runs everything, rewrites the tables below
bash benchmarks/run_kinematics.sh # or one block: run_kinematics.sh (FK, IK), run_dynamics.sh, run_trajopt.sh
```

Setup, running a single benchmark, and the raw result files: see [RUNNING.md](RUNNING.md).

## Findings

- **FK sphere output:** GRiM and mantis now fuse FK and sphere placement, pruning unused
  branches. At B = 16384, mantis's JAX path takes 0.47 / 0.61 / 0.60 ms on
  Panda / Baxter / G1, versus 0.47 / 1.56 / 2.03 ms in the previous pose-output
  implementation. G1 is now faster than pRRTC (0.65 ms) and cuRobo (0.77 ms);
  Panda and Baxter still trail those baselines. Specialization aligns the output
  workload; it does not make every launch strategy equally fast.
- **IK:** traced, GRiM's own LS algorithm at GRiM's precision takes 14-24 ms at 1024 x 8 seeds,
  against GRiM's 70-207 ms (4.9-8.7x) and pyroki's 0.8-5.6 s, with GRiM's success in every
  cell. At 1 x 8 the gain is 1.5x on Baxter and g1, and GRiM's block tier is faster on Panda.
  pyroki solves slightly more (98-100%).
- **Dynamics:** at B >= 1024 mantis leads GRiD by up to 62x (inverse dynamics at B = 16384).
  At B <= 128 GRiD is mostly faster, most of all on g1 derivatives. GRiD runs out of memory on
  g1 at B = 16384, and so does frax's autodiff derivative. At large batch frax is as fast as GRiD
  or faster for ID and CRBA.
- **Lesson:** trace the whole computation. Swapping mantis's FK into hand-written kernels
  gained at most 3x; tracing the whole IK step gained up to 8.7x at equal algorithm and precision,
  and the dynamics up to 62x.

## Protocol

Every method gets the same float32 inputs, cached once in `results/inputs/` (so methods in
other conda envs read identical arrays), and is timed the same way (`common.amortized_ms`):
50 back-to-back asynchronous calls, one sync, divided by 50; the median of 9 such runs after
3 warm-up calls. That is per-call throughput time with inputs already on the GPU, measured the
with host wall-clock timing for JAX, PyTorch and the C bridge, and CUDA events
for pRRTC. The C bridge removes Python/JAX dispatch but still includes C launch overhead. All runs use one GPU (RTX A5000).

Every output is checked against Pinocchio in float64 on 65 rows: the first 32, 32 spread over
the largest batch, and the last row of every batch. JAX runs with
`JAX_DEFAULT_MATMUL_PRECISION=highest`: JAX's default TF32 matmuls on Ampere gave frax
relative errors of 1e-3 to 4e-2; true float32 costs it about 5%.

Where a library offers several kernels for the same job, the fastest per cell is reported and
named: GRiM IK's thread / warp / block tiers, mantis dynamics' thread / warp-split tiers.
A call whose memory does not fit on the GPU at some batch is reported as OOM, not chunked.

## Robots

| | FK | IK and dynamics |
|---|---|---|
| panda | `robots/panda.urdf` (pRRTC's spherized panda, 59 spheres) | same file |
| baxter | `robots/baxter_spherized.urdf` (pRRTC's, 75 spheres, 14 DOF) | `robots/baxter.urdf` (GRiD's, 15 DOF incl. head pan) |
| g1 | `robots/g1.urdf` (pyroffi-grim's spherized G1 with hands, 68 spheres, 43 DOF) | same file |

pRRTC's spherized baxter has no inertias, so IK and dynamics use GRiD's. The panda file drops
one `<mimic>` tag on a *fixed* finger joint (meaningless there; pyroki refuses it).

## Methods

**FK** computes the world centre of every collision sphere, `(B, S, 3)`.
pRRTC, cuRobo, GRiM and mantis emit spheres directly from their kernels; pyroki
computes frame poses and places spheres in the same JIT.

- **pRRTC** `fk<Robot>` from its robot headers, launched as `pRRTC.cu` launches it (64 threads,
  16 configurations per block, shared-memory output), by `fk/prrtc/harness.cu`. Panda and Baxter
  are pRRTC's shipped headers. G1 is ported with pRRTC's documented flow (README "Adding a
  Robot"): cricket's `gpu-cc-early-exit` branch (CoMMALab/cricket@98582c3) on the spherized G1,
  configs in `fk/prrtc/cricket_config/`, output `fk/prrtc/g1.cuh` and the robot struct
  `g1_struct.hh`. Only the main (fine-sphere) kernel is generated; FK uses no approx kernel.
  Building that cricket branch against current conda-forge needed four fixes: C++17 (Pinocchio
  3.9), no `Boost::system` (header-only since Boost 1.89), `fmt/format.h` (fmt 11), and a signed
  `joint_id_to_dof` (the universe joint's -1 was emitted as 18446744073709551615). With them,
  regenerating Baxter reproduces pRRTC's shipped `fk<Baxter>` and its tables exactly. CUDA 13
  removed `cudaDeviceProp::memoryClockRate`, read only by a pRRTC print helper; the harness
  build aliases it.
- **cuRobo** (v0.8 dev) `Kinematics.compute_kinematics` with the URDF's own spheres (not
  cuRobo's yml spheres) and one tool frame. cuRobo drops joints that move no sphere.
- **pyroki** `Robot.forward_kinematics`.
- **GRiD/GRiM** `grim.motion.kinematics.sphere_kinematics(traced=False)`: specialized
  baked-table FK over only the requested spheres' ancestor joints, fused with placement.
- **mantis** the same call with `traced=True`: traces sphere centres directly, allowing
  elimination of unused frames and intermediate poses. Both use one thread per configuration,
  128 threads per block, and write only sphere centres; no pose buffer or second launch.
  The C bridge replays that identical single kernel. The general pose-output API remains
  available but is not used in these FK rows. Results identify this as `fused-spheres-v1`.

**Fused FK + collision** is deprecated: it checked robot spheres against oriented boxes only, with
no self-collision and no other primitive type, so it did not measure the full collision task. Its
scripts (`fk/bench_fkcc.py`, `fk/mantis_fkcc.py`) and results are in git history.

**IK**: one end-effector (`panda_grasptarget`, `right_gripper`, `right_hand_palm_link`), P pose
targets from FK of random in-limit configurations, 8 random seeds each, position weight 50,
orientation weight 10, at most 60 iterations. Success: some seed within 1 mm and 0.01 rad, inside
joint limits, by Pinocchio.

- **GRiM LS IK, GRiD tables**: `grim.motion.ik` `ls` kernel (Levenberg-Marquardt), `traced=False`.
- **mantis-traced GRiM LS IK** (`ik/mantis_ik_ls.py`): GRiM's `ls` algorithm and precision,
  traced, so the row isolates tracing. `ik_ls_system` (FK, chain Jacobian, weighted residual) is
  traced in float, `ik_ls_solve` (Jacobi column scaling, the chain's n x n normal equations
  `(Js^T Js + λI) d = -Js^T r`, unrolled Cholesky) in double, as GRiM splits them; the
  line-search merit `ik_pose_cost` is traced in float. The loop around them copies `ls_ik.cu`:
  convergence test, trust-region radii, the five line-search step sizes clamped to the joint
  limits, accept below `cost × (1 - 1e-4)` (λ / 2, else λ × 3), all-time best returned. The step
  matches GRiM's float64 reference to 1e-10, and success matches GRiM's in every cell. One seed
  per thread. A raw CUDA launch timed like the pRRTC harness, so it pays no JAX dispatch (about
  0.05 ms, small against these times). Not shown: the same build all-float is a further
  4.7-5.3x faster at 1024 x 8 with no loss of success, and mantis's own 6x6 `ik_lm_step`
  (McVAMP's inner LM form) about 2x beyond that, 0.1-1.6 points less success.
- **pyroki + jaxls**: pyroki's own IK formulation (`pose_cost_analytic_jac` + `limit_constraint`,
  dense-Cholesky LM, as in its `_solve_ik` example), `vmap`ped over problems × seeds.

**Dynamics**: ID (RNEA), mass matrix (CRBA), FD, and ∂ID/∂(q, q̇); random in-limit q, q̇ and q̈
in [-1, 1], FD fed the ID torques; gravity -9.81 z. Error is `max |x - ref| / max(1, max |ref|)`.

- **GRiD**: GRiD's generated kernels (`grid_codegen` header via `grim.motion.generated_dynamics`).
- **mantis**: `grim.motion.dynamics.TracedDynamics`; FD is ABA.
- **frax**: `jax.jit(jax.vmap(...))` of `rnea`, `crba`, `forward_dynamics`. frax has no
  analytical derivatives; its ∂ID is `jax.jacfwd` of its RNEA.

## Results

<!-- tables -->
### FK: world centres of every collision sphere

Spheres per robot: panda 59, baxter 75, g1 68. Time per call in ms (lower is better).

**B = 1**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.076 | 0.094 | 0.155 |
| cuRobo | 0.136 | 0.135 | 0.134 |
| pyroki | 0.271 | 0.824 | 0.979 |
| GRiD/GRiM (baked tables) | 0.035 | 0.068 | 0.159 |
| mantis (traced) | 0.037 | 0.037 | 0.066 |
| GRiD/GRiM (baked tables), C bridge | 0.026 | 0.064 | 0.155 |
| mantis (traced), C bridge | 0.009 | 0.014 | 0.062 |

**B = 128**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.075 | 0.093 | 0.152 |
| cuRobo | 0.137 | 0.137 | 0.136 |
| pyroki | 0.352 | 1.114 | 1.332 |
| GRiD/GRiM (baked tables) | 0.049 | 0.092 | 0.197 |
| mantis (traced) | 0.033 | 0.040 | 0.102 |
| GRiD/GRiM (baked tables), C bridge | 0.044 | 0.088 | 0.193 |
| mantis (traced), C bridge | 0.027 | 0.036 | 0.097 |

**B = 1024**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.075 | 0.093 | 0.153 |
| cuRobo | 0.139 | 0.137 | 0.136 |
| pyroki | 0.343 | 1.058 | 1.271 |
| GRiD/GRiM (baked tables) | 0.052 | 0.089 | 0.175 |
| mantis (traced) | 0.036 | 0.040 | 0.096 |
| GRiD/GRiM (baked tables), C bridge | 0.043 | 0.084 | 0.171 |
| mantis (traced), C bridge | 0.027 | 0.036 | 0.092 |

**B = 16384**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.346 | 0.501 | 0.653 |
| cuRobo | 0.148 | 0.493 | 0.771 |
| pyroki | 1.528 | 4.883 | 5.915 |
| GRiD/GRiM (baked tables) | 0.089 | 0.234 | 0.630 |
| mantis (traced) | 0.059 | 0.112 | 0.305 |
| GRiD/GRiM (baked tables), C bridge | 0.084 | 0.228 | 0.623 |
| mantis (traced), C bridge | 0.056 | 0.108 | 0.301 |

**Max sphere-centre error vs Pinocchio float64 (m), over all batches**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 8.4e-07 | 1.3e-06 | 1.1e-06 |
| cuRobo | 2.0e-07 | 3.1e-07 | 1.6e-07 |
| pyroki | 2.8e-07 | 4.9e-07 | 3.3e-07 |
| GRiD/GRiM (baked tables) | 2.9e-07 | 6.1e-07 | 3.2e-07 |
| mantis (traced) | 2.2e-07 | 2.7e-07 | 1.9e-07 |
| GRiD/GRiM (baked tables), C bridge | 2.9e-07 | 6.1e-07 | 3.2e-07 |
| mantis (traced), C bridge | 2.2e-07 | 2.7e-07 | 1.9e-07 |

### Global IK: one end-effector pose from random seeds

GRiM: 32 seeds, 60 iterations max, fastest tier; pyroki and cuRobo as their own benchmarks configure them. Time per call in ms, then success rate (position < 1 mm, orientation < 0.01 rad, within the URDF limits; Pinocchio float64).

**1 problem**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK | 2.79 (100.0%, thread) | 3.05 (100.0%, thread) | 4.34 (100.0%, warp) |
| GRiM DLS IK | 0.88 (100.0%, thread) | 0.92 (100.0%, thread) | 1.88 (100.0%, thread) |
| GRiM SQP IK | 11.10 (100.0%, thread) | 10.88 (100.0%, thread) | 31.03 (100.0%, warp) |
| GRiM HJCD IK (coarse CD -> LM) | 2.94 (100.0%, thread) | 2.96 (100.0%, thread) | 5.21 (100.0%, block) |
| GRiM MPPI IK (MPPI -> L-BFGS) | 2.92 (100.0%, thread) | 3.81 (100.0%, block) | 10.77 (100.0%, thread) |
| GRiM LS IK, C bridge | 2.78 (100.0%, thread) | 3.04 (100.0%, thread) | 4.34 (100.0%, warp) |
| GRiM DLS IK, C bridge | 0.87 (100.0%, thread) | 0.91 (100.0%, thread) | 1.87 (100.0%, thread) |
| GRiM SQP IK, C bridge | 11.09 (100.0%, thread) | 10.86 (100.0%, thread) | 31.02 (100.0%, warp) |
| GRiM HJCD IK, C bridge | 2.93 (100.0%, thread) | 2.95 (100.0%, thread) | 5.20 (100.0%, block) |
| GRiM MPPI IK, C bridge | 2.91 (100.0%, thread) | 3.80 (100.0%, block) | 10.77 (100.0%, thread) |
| pyroki IK-Beam (analytic Jacobian; own 64->4 seeds) | 19.79 (100.0%, ik-beam 64->4) | 48.91 (100.0%, ik-beam 64->4) | 55.44 (100.0%, ik-beam 64->4) |
| cuRobo v2 IKSolver (LM-seeded L-BFGS; own seeds) | 4.45 (100.0%, lbfgs+lm-seed) | 4.46 (100.0%, lbfgs+lm-seed) | 4.56 (100.0%, lbfgs+lm-seed) |

**128 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK | 2.83 (100.0%, thread) | 2.93 (100.0%, thread) | 6.81 (100.0%, thread) |
| GRiM DLS IK | 0.89 (100.0%, thread) | 0.92 (100.0%, thread) | 1.97 (100.0%, thread) |
| GRiM SQP IK | 11.09 (100.0%, thread) | 10.97 (100.0%, thread) | 44.89 (100.0%, thread) |
| GRiM HJCD IK (coarse CD -> LM) | 2.92 (96.9%, thread) | 2.96 (91.4%, thread) | 6.57 (99.2%, thread) |
| GRiM MPPI IK (MPPI -> L-BFGS) | 2.92 (100.0%, thread) | 4.70 (100.0%, thread) | 13.12 (100.0%, thread) |
| GRiM LS IK, C bridge | 2.82 (100.0%, thread) | 2.92 (100.0%, thread) | 6.80 (100.0%, thread) |
| GRiM DLS IK, C bridge | 0.88 (100.0%, thread) | 0.91 (100.0%, thread) | 1.96 (100.0%, thread) |
| GRiM SQP IK, C bridge | 11.08 (100.0%, thread) | 10.96 (100.0%, thread) | 44.83 (100.0%, thread) |
| GRiM HJCD IK, C bridge | 2.91 (96.9%, thread) | 2.95 (91.4%, thread) | 6.56 (99.2%, thread) |
| GRiM MPPI IK, C bridge | 2.91 (100.0%, thread) | 4.69 (100.0%, thread) | 13.06 (100.0%, thread) |
| pyroki IK-Beam (analytic Jacobian; own 64->4 seeds) | 62.24 (100.0%, ik-beam 64->4) | 179.05 (99.2%, ik-beam 64->4) | 333.84 (100.0%, ik-beam 64->4) |
| cuRobo v2 IKSolver (LM-seeded L-BFGS; own seeds) | 8.72 (100.0%, lbfgs+lm-seed) | 8.87 (100.0%, lbfgs+lm-seed) | 10.07 (100.0%, lbfgs+lm-seed) |

**1024 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK | 7.20 (99.9%, thread) | 7.34 (99.9%, thread) | 18.36 (100.0%, thread) |
| GRiM DLS IK | 2.42 (99.9%, thread) | 2.48 (99.9%, thread) | 6.24 (100.0%, thread) |
| GRiM SQP IK | 25.74 (100.0%, thread) | 25.73 (100.0%, thread) | 132.71 (100.0%, thread) |
| GRiM HJCD IK (coarse CD -> LM) | 5.69 (97.2%, thread) | 7.57 (93.8%, thread) | 17.88 (98.8%, thread) |
| GRiM MPPI IK (MPPI -> L-BFGS) | 7.08 (100.0%, thread) | 15.39 (99.9%, thread) | 46.98 (100.0%, thread) |
| GRiM LS IK, C bridge | 7.18 (99.9%, thread) | 7.35 (99.9%, thread) | 19.11 (100.0%, thread) |
| GRiM DLS IK, C bridge | 2.56 (99.9%, thread) | 2.60 (99.9%, thread) | 6.52 (100.0%, thread) |
| GRiM SQP IK, C bridge | 25.98 (100.0%, thread) | 26.06 (100.0%, thread) | 133.00 (100.0%, thread) |
| GRiM HJCD IK, C bridge | 5.62 (97.2%, thread) | 7.60 (93.8%, thread) | 18.24 (98.8%, thread) |
| GRiM MPPI IK, C bridge | 7.19 (100.0%, thread) | 15.96 (99.9%, thread) | 48.37 (100.0%, thread) |
| pyroki IK-Beam (analytic Jacobian; own 64->4 seeds) | 294.20 (100.0%, ik-beam 64->4) | 1097.39 (99.6%, ik-beam 64->4) | 8883.43 (99.9%, ik-beam 64->4) |
| cuRobo v2 IKSolver (LM-seeded L-BFGS; own seeds) | 42.00 (100.0%, lbfgs+lm-seed) | 43.29 (100.0%, lbfgs+lm-seed) | 52.77 (100.0%, lbfgs+lm-seed) |

### Supplementary: mantis IK steps

mantis's traced IK steps in the benchmark's own minimal CUDA loops (one seed per thread, 32 seeds, 60 iterations), across precisions. A profile of the trace, not a comparison with the solvers above.

**1 problem**

| method | panda | baxter | g1 |
|---|---|---|---|
| mantis `ik_lm_step` (6x6 LM) | 1.54 (100.0%, thread) | 1.67 (100.0%, thread) | 2.29 (100.0%, thread) |
| mantis `ik_ls_step`, float | 3.13 (100.0%, f32) | 3.23 (100.0%, f32) | 4.68 (100.0%, f32) |
| mantis `ik_ls_step`, float system / double solve | 10.22 (100.0%, mixed) | 10.38 (100.0%, mixed) | 16.67 (100.0%, mixed) |
| mantis `ik_ls_step`, double | 24.49 (100.0%, f64) | 25.37 (100.0%, f64) | 36.38 (100.0%, f64) |
| mantis SQP step, float | 19.07 (100.0%, f32) | 18.88 (100.0%, f32) | 35.34 (100.0%, f32) |
| mantis SQP step, float FK / double QP | 84.56 (100.0%, mixed) | 85.65 (100.0%, mixed) | 163.96 (100.0%, mixed) |
| mantis SQP step, double | 104.68 (100.0%, f64) | 104.56 (100.0%, f64) | 190.79 (100.0%, f64) |

**128 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| mantis `ik_lm_step` (6x6 LM) | 1.52 (100.0%, thread) | 1.68 (100.0%, thread) | 2.38 (100.0%, thread) |
| mantis `ik_ls_step`, float | 3.08 (100.0%, f32) | 3.14 (99.2%, f32) | 4.51 (100.0%, f32) |
| mantis `ik_ls_step`, float system / double solve | 10.90 (100.0%, mixed) | 11.02 (100.0%, mixed) | 17.80 (100.0%, mixed) |
| mantis `ik_ls_step`, double | 29.24 (100.0%, f64) | 30.26 (100.0%, f64) | 42.62 (100.0%, f64) |
| mantis SQP step, float | 18.94 (100.0%, f32) | 18.90 (100.0%, f32) | 34.81 (100.0%, f32) |
| mantis SQP step, float FK / double QP | 89.93 (100.0%, mixed) | 91.83 (100.0%, mixed) | 172.34 (100.0%, mixed) |
| mantis SQP step, double | 114.08 (100.0%, f64) | 114.77 (100.0%, f64) | 205.27 (100.0%, f64) |

**1024 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| mantis `ik_lm_step` (6x6 LM) | 3.62 (100.0%, thread) | 2.92 (99.9%, thread) | 6.56 (100.0%, thread) |
| mantis `ik_ls_step`, float | 5.68 (99.9%, f32) | 5.80 (99.8%, f32) | 11.85 (100.0%, f32) |
| mantis `ik_ls_step`, float system / double solve | 50.93 (100.0%, mixed) | 50.04 (99.7%, mixed) | 81.81 (100.0%, mixed) |
| mantis `ik_ls_step`, double | 188.90 (100.0%, f64) | 197.38 (99.7%, f64) | 267.94 (100.0%, f64) |
| mantis SQP step, float | 44.03 (100.0%, f32) | 43.98 (100.0%, f32) | 77.37 (100.0%, f32) |
| mantis SQP step, float FK / double QP | 448.61 (100.0%, mixed) | 448.30 (100.0%, mixed) | 840.72 (100.0%, mixed) |
| mantis SQP step, double | 614.33 (100.0%, f64) | 618.88 (100.0%, f64) | 1062.57 (100.0%, f64) |

### Dynamics: GRiD's release protocol

Time per call in ms, median over 3 processes of 30 samples each (GRiD's `collect.py` defaults), every cell checked against an independent float64 Pinocchio oracle under GRiD's release-table policy (`fp32-fd-warnings`): †: a float32 FD cell past the strict entrywise gate but within relative L2 1e-3, kept with a warning as GRiD keeps it. GPU methods are timed device-resident (inputs and outputs on the GPU); the CPU libraries take host arrays, so theirs is the host call, as in GRiD's Figure 2b. B = 1 and 16384 are additions to GRiD's batches. n/a: the library has no such operation; OOM: the call's workspace does not fit in the GPU's 24 GB. mantis, GRiD (JAX FFI) and frax are called through JAX, so their resident time includes JAX dispatch (about 0.1 ms, which dominates below B = 1024); GRiD's C++ bridge times its kernel launch directly.

**B = 1**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.021 | 0.089 |
| ID (RNEA) · mantis (JAX) | 0.136 | 0.154 | 0.239 |
| ID (RNEA) · GRiD (C++ bridge) | 0.034 | 0.053 | 0.127 |
| ID (RNEA) · GRiD (JAX FFI) | 0.182 | 0.223 | 0.302 |
| ID (RNEA) · frax | 0.254 | 0.401 | 0.819 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.003 | 0.004 | 0.006 |
| ID (RNEA) · RBDL (CPU, float64) | 0.004 | 0.004 | 0.007 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.198 |
| M (CRBA) · mantis (JAX) | 0.135 | 0.127 | 0.349 |
| M (CRBA) · GRiD (C++ bridge) | 0.032 | 0.051 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.160 | 0.199 | 0.286 |
| M (CRBA) · frax | 0.230 | 0.404 | 0.829 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.004 | 0.005 | 0.009 |
| M (CRBA) · RBDL (CPU, float64) | 0.004 | 0.005 | 0.009 |
| FD · mantis (C bridge) | 0.016 | 0.050 | 0.183 |
| FD · mantis (JAX) | 0.127 | 0.191 | 0.329 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.262 |
| FD · GRiD (JAX FFI) | 0.227 | 0.264 | 0.452 |
| FD · frax | 0.369 | 0.510 | 1.026† |
| FD · Pinocchio codegen (CPU) | 0.004 | 0.006 | 0.016 |
| FD · RBDL (CPU, float64) | 0.005 | 0.006 | 0.013 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.076 | 0.170 | 1.260 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.226 | 0.321 | 1.431 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.405 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.218 | 0.244 | 0.596 |
| ∂ID/∂(q, q̇) · frax | 0.427 | 0.689 | 2.020 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.006 | 0.009 | 0.026 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.021 | 0.084 |
| ID (RNEA) · mantis (JAX) | 0.140 | 0.137 | 0.240 |
| ID (RNEA) · GRiD (C++ bridge) | 0.033 | 0.053 | 0.126 |
| ID (RNEA) · GRiD (JAX FFI) | 0.189 | 0.220 | 0.293 |
| ID (RNEA) · frax | 0.262 | 0.404 | 0.853 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.008 | 0.015 | 0.036 |
| ID (RNEA) · RBDL (CPU, float64) | 0.014 | 0.026 | 0.068 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.199 |
| M (CRBA) · mantis (JAX) | 0.168 | 0.136 | 0.342 |
| M (CRBA) · GRiD (C++ bridge) | 0.032 | 0.050 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.143 | 0.194 | 0.288 |
| M (CRBA) · frax | 0.242 | 0.419 | 0.856 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.011 | 0.023 | 0.085 |
| M (CRBA) · RBDL (CPU, float64) | 0.014 | 0.026 | 0.100 |
| FD · mantis (C bridge) | 0.015 | 0.051 | 0.183 |
| FD · mantis (JAX) | 0.166 | 0.204 | 0.341 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.258 |
| FD · GRiD (JAX FFI) | 0.215 | 0.257 | 0.434 |
| FD · frax | 0.366 | 0.517 | 1.185† |
| FD · Pinocchio codegen (CPU) | 0.019 | 0.039 | 0.146 |
| FD · RBDL (CPU, float64) | 0.027 | 0.052 | 0.132 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.076 | 0.166 | 1.267 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.229 | 0.319 | 1.454 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.403 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.223 | 0.262 | 0.600 |
| ∂ID/∂(q, q̇) · frax | 0.499 | 0.892 | 3.045 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.033 | 0.079 | 0.172 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 32**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.021 | 0.085 |
| ID (RNEA) · mantis (JAX) | 0.151 | 0.147 | 0.232 |
| ID (RNEA) · GRiD (C++ bridge) | 0.033 | 0.053 | 0.126 |
| ID (RNEA) · GRiD (JAX FFI) | 0.202 | 0.226 | 0.307 |
| ID (RNEA) · frax | 0.269 | 0.407 | 0.864 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.013 | 0.026 | 0.069 |
| ID (RNEA) · RBDL (CPU, float64) | 0.024 | 0.048 | 0.127 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.202 |
| M (CRBA) · mantis (JAX) | 0.125 | 0.133 | 0.362 |
| M (CRBA) · GRiD (C++ bridge) | 0.032 | 0.050 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.145 | 0.193 | 0.294 |
| M (CRBA) · frax | 0.248 | 0.425 | 0.856 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.018 | 0.042 | 0.144 |
| M (CRBA) · RBDL (CPU, float64) | 0.025 | 0.048 | 0.129 |
| FD · mantis (C bridge) | 0.015 | 0.051 | 0.184 |
| FD · mantis (JAX) | 0.141 | 0.199 | 0.335 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.258 |
| FD · GRiD (JAX FFI) | 0.217 | 0.258 | 0.447 |
| FD · frax | 0.365 | 0.524 | 1.209† |
| FD · Pinocchio codegen (CPU) | 0.035 | 0.075 | 0.181 |
| FD · RBDL (CPU, float64) | 0.051 | 0.102 | 0.142 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.076 | 0.167 | 1.295 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.224 | 0.319 | 1.483 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.403 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.220 | 0.250 | 0.598 |
| ∂ID/∂(q, q̇) · frax | 0.596 | 0.999 | 3.791 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.063 | 0.144 | 0.295 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 64**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.022 | 0.085 |
| ID (RNEA) · mantis (JAX) | 0.146 | 0.169 | 0.235 |
| ID (RNEA) · GRiD (C++ bridge) | 0.033 | 0.053 | 0.125 |
| ID (RNEA) · GRiD (JAX FFI) | 0.204 | 0.231 | 0.306 |
| ID (RNEA) · frax | 0.265 | 0.413 | 0.948 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.022 | 0.047 | 0.134 |
| ID (RNEA) · RBDL (CPU, float64) | 0.046 | 0.093 | 0.142 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.241 |
| M (CRBA) · mantis (JAX) | 0.130 | 0.129 | 0.403 |
| M (CRBA) · GRiD (C++ bridge) | 0.031 | 0.050 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.153 | 0.193 | 0.299 |
| M (CRBA) · frax | 0.251 | 0.419 | 0.888 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.032 | 0.080 | 0.169 |
| M (CRBA) · RBDL (CPU, float64) | 0.047 | 0.094 | 0.147 |
| FD · mantis (C bridge) | 0.016 | 0.053 | 0.196 |
| FD · mantis (JAX) | 0.123 | 0.204 | 0.350 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.256 |
| FD · GRiD (JAX FFI) | 0.223 | 0.266 | 0.430 |
| FD · frax | 0.367 | 0.538† | 1.283† |
| FD · Pinocchio codegen (CPU) | 0.066 | 0.142 | 0.273 |
| FD · RBDL (CPU, float64) | 0.100 | 0.133 | 0.173 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.078 | 0.170 | 1.352 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.243 | 0.334 | 1.582 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.409 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.225 | 0.261 | 0.639 |
| ∂ID/∂(q, q̇) · frax | 0.596 | 1.179 | 5.515 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.124 | 0.160 | 0.502 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 128**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.024 | 0.131 |
| ID (RNEA) · mantis (JAX) | 0.133 | 0.138 | 0.283 |
| ID (RNEA) · GRiD (C++ bridge) | 0.036 | 0.061 | 0.154 |
| ID (RNEA) · GRiD (JAX FFI) | 0.194 | 0.229 | 0.345 |
| ID (RNEA) · frax | 0.278 | 0.465 | 0.997 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.041 | 0.091 | 0.162 |
| ID (RNEA) · RBDL (CPU, float64) | 0.089 | 0.125 | 0.166 |
| M (CRBA) · mantis (C bridge) | 0.013 | 0.024 | 0.286 |
| M (CRBA) · mantis (JAX) | 0.130 | 0.180 | 0.471 |
| M (CRBA) · GRiD (C++ bridge) | 0.034 | 0.058 | 0.167 |
| M (CRBA) · GRiD (JAX FFI) | 0.182 | 0.201 | 0.322 |
| M (CRBA) · frax | 0.244 | 0.442 | 0.916 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.060 | 0.148 | 0.203 |
| M (CRBA) · RBDL (CPU, float64) | 0.092 | 0.126 | 0.215 |
| FD · mantis (C bridge) | 0.016 | 0.059 | 0.233 |
| FD · mantis (JAX) | 0.138 | 0.204 | 0.400 |
| FD · GRiD (C++ bridge) | 0.057 | 0.096 | 0.290 |
| FD · GRiD (JAX FFI) | 0.228 | 0.272 | 0.471 |
| FD · frax | 0.375 | 0.585† | 1.366† |
| FD · Pinocchio codegen (CPU) | 0.129 | 0.165 | 0.458 |
| FD · RBDL (CPU, float64) | 0.135 | 0.162 | 0.314 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.092 | 0.195 | 1.873 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.239 | 0.362 | 2.129 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.055 | 0.099 | 0.806 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.240 | 0.283 | 1.051 |
| ∂ID/∂(q, q̇) · frax | 0.671 | 1.502 | 8.246 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.162 | 0.207 | 0.841 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 256**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.024 | 0.131 |
| ID (RNEA) · mantis (JAX) | 0.144 | 0.186 | 0.283 |
| ID (RNEA) · GRiD (C++ bridge) | 0.044 | 0.081 | 0.221 |
| ID (RNEA) · GRiD (JAX FFI) | 0.219 | 0.260 | 0.404 |
| ID (RNEA) · frax | 0.317 | 0.499 | 1.077 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.080 | 0.145 | 0.204 |
| ID (RNEA) · RBDL (CPU, float64) | 0.128 | 0.142 | 0.274 |
| M (CRBA) · mantis (C bridge) | 0.013 | 0.024 | 0.288 |
| M (CRBA) · mantis (JAX) | 0.126 | 0.168 | 0.504 |
| M (CRBA) · GRiD (C++ bridge) | 0.063 | 0.078 | 0.387 |
| M (CRBA) · GRiD (JAX FFI) | 0.206 | 0.223 | 0.559 |
| M (CRBA) · frax | 0.276 | 0.469 | 0.999 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.117 | 0.165 | 0.319 |
| M (CRBA) · RBDL (CPU, float64) | 0.126 | 0.153 | 0.360 |
| FD · mantis (C bridge) | 0.016 | 0.059 | 0.234 |
| FD · mantis (JAX) | 0.153 | 0.201 | 0.394 |
| FD · GRiD (C++ bridge) | 0.066 | 0.118 | 0.613 |
| FD · GRiD (JAX FFI) | 0.235 | 0.293 | 0.812 |
| FD · frax | 0.426 | 0.621† | 1.520† |
| FD · Pinocchio codegen (CPU) | 0.150 | 0.198 | 0.730 |
| FD · RBDL (CPU, float64) | 0.159 | 0.275 | 0.512 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.092 | 0.260 | 1.955 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.236 | 0.426 | 2.256 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.070 | 0.140 | 1.597 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.242 | 0.334 | 1.897 |
| ∂ID/∂(q, q̇) · frax | 0.839 | 2.203 | 14.677 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.175 | 0.432 | 0.946 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 1024**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.024 | 0.124 |
| ID (RNEA) · mantis (JAX) | 0.136 | 0.138 | 0.279 |
| ID (RNEA) · GRiD (C++ bridge) | 0.143 | 0.222 | 0.792 |
| ID (RNEA) · GRiD (JAX FFI) | 0.331 | 0.414 | 0.993 |
| ID (RNEA) · frax | 0.400 | 0.623 | 1.408 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.154 | 0.280 | 0.529 |
| ID (RNEA) · RBDL (CPU, float64) | 0.218 | 0.330 | 0.699 |
| M (CRBA) · mantis (C bridge) | 0.013 | 0.024 | 0.333 |
| M (CRBA) · mantis (JAX) | 0.129 | 0.172 | 0.660 |
| M (CRBA) · GRiD (C++ bridge) | 0.168 | 0.225 | 1.138 |
| M (CRBA) · GRiD (JAX FFI) | 0.315 | 0.379 | 1.332 |
| M (CRBA) · frax | 0.324 | 0.597 | 1.437 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.183 | 0.367 | 0.824 |
| M (CRBA) · RBDL (CPU, float64) | 0.244 | 0.376 | 0.952 |
| FD · mantis (C bridge) | 0.016 | 0.059 | 0.224 |
| FD · mantis (JAX) | 0.132 | 0.202 | 0.394 |
| FD · GRiD (C++ bridge) | 0.227 | 0.340 | 1.933 |
| FD · GRiD (JAX FFI) | 0.419 | 0.530 | 2.187 |
| FD · frax | 0.535 | 0.850† | 2.506† |
| FD · Pinocchio codegen (CPU) | 0.318 | 0.608 | 1.626† |
| FD · RBDL (CPU, float64) | 0.360 | 0.640 | 1.341 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.087 | 0.297 | 1.909 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.247 | 0.499 | 2.482 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.228 | 0.451 | 6.315 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.430 | 0.672 | 6.857 |
| ∂ID/∂(q, q̇) · frax | 1.648 | 6.662 | 52.577 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.487 | 0.795 | 3.344 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16384**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.016 | 0.029 | 0.186 |
| ID (RNEA) · mantis (JAX) | 0.136 | 0.192 | 0.395 |
| ID (RNEA) · GRiD (C++ bridge) | 1.745 | 2.608 | OOM |
| ID (RNEA) · GRiD (JAX FFI) | 2.155 | 3.048 | 11.397 |
| ID (RNEA) · frax | 1.398 | 2.897 | 9.891 |
| ID (RNEA) · Pinocchio codegen (CPU) | 1.017 | 1.675 | 4.956 |
| ID (RNEA) · RBDL (CPU, float64) | 1.761 | 3.529 | 8.072 |
| M (CRBA) · mantis (C bridge) | 0.020 | 0.056 | 0.783 |
| M (CRBA) · mantis (JAX) | 0.188 | 0.451 | 11.779 |
| M (CRBA) · GRiD (C++ bridge) | 2.041 | 2.924 | OOM |
| M (CRBA) · GRiD (JAX FFI) | 2.327 | 3.212 | 17.099 |
| M (CRBA) · frax | 1.295 | 3.052 | 10.757 |
| M (CRBA) · Pinocchio codegen (CPU) | 1.101 | 2.854 | 14.210 |
| M (CRBA) · RBDL (CPU, float64) | 1.919 | 3.611 | 16.232 |
| FD · mantis (C bridge) | 0.023 | 0.077 | 0.424 |
| FD · mantis (JAX) | 0.167 | 0.248 | 0.639 |
| FD · GRiD (C++ bridge) | 3.162 | 4.502 | OOM |
| FD · GRiD (JAX FFI) | 3.571 | 4.992 | 29.366 |
| FD · frax | 2.419 | 5.258† | 25.107† |
| FD · Pinocchio codegen (CPU) | 2.301 | 5.138 | 20.577† |
| FD · RBDL (CPU, float64) | 3.575 | 7.182 | 16.706 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.157 | 0.526 | 3.883 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.436 | 1.424 | 26.796 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 3.350 | 6.190 | OOM |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 3.798 | 6.969 | OOM |
| ∂ID/∂(q, q̇) · frax | 19.690 | 98.408 | OOM |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 4.319 | 11.192 | 44.760 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

#### Host to host (host arrays in and out, including transfers)


**B = 1**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.026 | 0.099 |
| ID (RNEA) · mantis (JAX) | 0.656 | 0.673 | 0.774 |
| ID (RNEA) · GRiD (C++ bridge) | 0.045 | 0.065 | 0.139 |
| ID (RNEA) · GRiD (JAX FFI) | 0.738 | 0.733 | 0.857 |
| ID (RNEA) · frax | 0.733 | 0.916 | 1.282 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.003 | 0.004 | 0.006 |
| ID (RNEA) · RBDL (CPU, float64) | 0.004 | 0.004 | 0.007 |
| M (CRBA) · mantis (C bridge) | 0.017 | 0.026 | 0.206 |
| M (CRBA) · mantis (JAX) | 0.455 | 0.459 | 0.646 |
| M (CRBA) · GRiD (C++ bridge) | 0.039 | 0.058 | 0.147 |
| M (CRBA) · GRiD (JAX FFI) | 0.483 | 0.533 | 0.653 |
| M (CRBA) · frax | 0.757 | 0.846 | 1.223 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.004 | 0.005 | 0.009 |
| M (CRBA) · RBDL (CPU, float64) | 0.004 | 0.005 | 0.009 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.189 |
| FD · mantis (JAX) | 0.673 | 0.707 | 0.850 |
| FD · GRiD (C++ bridge) | 0.064 | 0.099 | 0.273 |
| FD · GRiD (JAX FFI) | 0.808 | 0.773 | 0.965 |
| FD · frax | 0.869 | 1.011 | 1.456† |
| FD · Pinocchio codegen (CPU) | 0.004 | 0.006 | 0.016 |
| FD · RBDL (CPU, float64) | 0.005 | 0.006 | 0.013 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.181 | 1.265 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.741 | 0.846 | 2.004 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.061 | 0.098 | 0.418 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.728 | 0.800 | 1.093 |
| ∂ID/∂(q, q̇) · frax | 0.942 | 1.158 | 2.301 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.006 | 0.009 | 0.026 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.026 | 0.091 |
| ID (RNEA) · mantis (JAX) | 0.659 | 0.679 | 0.783 |
| ID (RNEA) · GRiD (C++ bridge) | 0.045 | 0.065 | 0.138 |
| ID (RNEA) · GRiD (JAX FFI) | 0.685 | 0.731 | 0.899 |
| ID (RNEA) · frax | 0.758 | 0.935 | 1.312 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.008 | 0.015 | 0.036 |
| ID (RNEA) · RBDL (CPU, float64) | 0.014 | 0.026 | 0.068 |
| M (CRBA) · mantis (C bridge) | 0.017 | 0.025 | 0.209 |
| M (CRBA) · mantis (JAX) | 0.462 | 0.451 | 0.646 |
| M (CRBA) · GRiD (C++ bridge) | 0.039 | 0.058 | 0.150 |
| M (CRBA) · GRiD (JAX FFI) | 0.476 | 0.515 | 0.596 |
| M (CRBA) · frax | 0.704 | 0.895 | 1.252 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.011 | 0.023 | 0.085 |
| M (CRBA) · RBDL (CPU, float64) | 0.014 | 0.026 | 0.100 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.190 |
| FD · mantis (JAX) | 0.682 | 0.798 | 0.853 |
| FD · GRiD (C++ bridge) | 0.064 | 0.099 | 0.269 |
| FD · GRiD (JAX FFI) | 0.733 | 0.749 | 0.951 |
| FD · frax | 0.836 | 1.004 | 1.688† |
| FD · Pinocchio codegen (CPU) | 0.019 | 0.039 | 0.146 |
| FD · RBDL (CPU, float64) | 0.027 | 0.052 | 0.132 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.175 | 1.284 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.769 | 0.802 | 2.040 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.061 | 0.099 | 0.423 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.751 | 0.766 | 1.171 |
| ∂ID/∂(q, q̇) · frax | 1.002 | 1.314 | 3.215 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.033 | 0.079 | 0.172 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 32**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.025 | 0.092 |
| ID (RNEA) · mantis (JAX) | 0.665 | 0.683 | 0.749 |
| ID (RNEA) · GRiD (C++ bridge) | 0.046 | 0.065 | 0.139 |
| ID (RNEA) · GRiD (JAX FFI) | 0.744 | 0.739 | 0.849 |
| ID (RNEA) · frax | 0.788 | 0.919 | 1.390 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.013 | 0.026 | 0.069 |
| ID (RNEA) · RBDL (CPU, float64) | 0.024 | 0.048 | 0.127 |
| M (CRBA) · mantis (C bridge) | 0.016 | 0.026 | 0.214 |
| M (CRBA) · mantis (JAX) | 0.475 | 0.459 | 0.693 |
| M (CRBA) · GRiD (C++ bridge) | 0.039 | 0.057 | 0.155 |
| M (CRBA) · GRiD (JAX FFI) | 0.469 | 0.495 | 0.629 |
| M (CRBA) · frax | 0.764 | 0.896 | 1.276 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.018 | 0.042 | 0.144 |
| M (CRBA) · RBDL (CPU, float64) | 0.025 | 0.048 | 0.129 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.190 |
| FD · mantis (JAX) | 0.664 | 0.700 | 0.891 |
| FD · GRiD (C++ bridge) | 0.064 | 0.099 | 0.270 |
| FD · GRiD (JAX FFI) | 0.732 | 0.815 | 0.958 |
| FD · frax | 0.880 | 1.048 | 1.668† |
| FD · Pinocchio codegen (CPU) | 0.035 | 0.075 | 0.181 |
| FD · RBDL (CPU, float64) | 0.051 | 0.102 | 0.142 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.175 | 1.323 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.752 | 0.814 | 2.115 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.061 | 0.099 | 0.433 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.782 | 0.775 | 1.178 |
| ∂ID/∂(q, q̇) · frax | 1.100 | 1.446 | 3.989 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.063 | 0.144 | 0.295 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 64**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.016 | 0.033 | 0.099 |
| ID (RNEA) · mantis (JAX) | 0.651 | 0.676 | 0.752 |
| ID (RNEA) · GRiD (C++ bridge) | 0.046 | 0.066 | 0.138 |
| ID (RNEA) · GRiD (JAX FFI) | 0.756 | 0.739 | 0.800 |
| ID (RNEA) · frax | 0.800 | 0.885 | 1.406 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.022 | 0.047 | 0.134 |
| ID (RNEA) · RBDL (CPU, float64) | 0.046 | 0.093 | 0.142 |
| M (CRBA) · mantis (C bridge) | 0.016 | 0.027 | 0.264 |
| M (CRBA) · mantis (JAX) | 0.480 | 0.506 | 0.773 |
| M (CRBA) · GRiD (C++ bridge) | 0.039 | 0.059 | 0.164 |
| M (CRBA) · GRiD (JAX FFI) | 0.506 | 0.543 | 0.614 |
| M (CRBA) · frax | 0.715 | 0.895 | 1.292 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.032 | 0.080 | 0.169 |
| M (CRBA) · RBDL (CPU, float64) | 0.047 | 0.094 | 0.147 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.215 |
| FD · mantis (JAX) | 0.675 | 0.692 | 0.884 |
| FD · GRiD (C++ bridge) | 0.064 | 0.100 | 0.269 |
| FD · GRiD (JAX FFI) | 0.747 | 0.799 | 0.980 |
| FD · frax | 0.925 | 1.095† | 1.748† |
| FD · Pinocchio codegen (CPU) | 0.066 | 0.142 | 0.273 |
| FD · RBDL (CPU, float64) | 0.100 | 0.133 | 0.173 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.177 | 1.405 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.765 | 0.862 | 2.236 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.061 | 0.102 | 0.458 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.756 | 0.829 | 1.206 |
| ∂ID/∂(q, q̇) · frax | 1.086 | 1.600 | 5.699 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.124 | 0.160 | 0.502 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 128**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.016 | 0.034 | 0.148 |
| ID (RNEA) · mantis (JAX) | 0.710 | 0.679 | 0.797 |
| ID (RNEA) · GRiD (C++ bridge) | 0.048 | 0.074 | 0.169 |
| ID (RNEA) · GRiD (JAX FFI) | 0.771 | 0.785 | 0.898 |
| ID (RNEA) · frax | 0.808 | 0.906 | 1.428 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.041 | 0.091 | 0.162 |
| ID (RNEA) · RBDL (CPU, float64) | 0.089 | 0.125 | 0.166 |
| M (CRBA) · mantis (C bridge) | 0.017 | 0.037 | 0.332 |
| M (CRBA) · mantis (JAX) | 0.455 | 0.511 | 0.883 |
| M (CRBA) · GRiD (C++ bridge) | 0.041 | 0.070 | 0.219 |
| M (CRBA) · GRiD (JAX FFI) | 0.523 | 0.534 | 0.759 |
| M (CRBA) · frax | 0.752 | 0.899 | 1.426 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.060 | 0.148 | 0.203 |
| M (CRBA) · RBDL (CPU, float64) | 0.092 | 0.126 | 0.215 |
| FD · mantis (C bridge) | 0.025 | 0.067 | 0.249 |
| FD · mantis (JAX) | 0.661 | 0.694 | 0.929 |
| FD · GRiD (C++ bridge) | 0.069 | 0.109 | 0.305 |
| FD · GRiD (JAX FFI) | 0.761 | 0.800 | 1.021 |
| FD · frax | 0.908 | 1.130† | 1.838† |
| FD · Pinocchio codegen (CPU) | 0.129 | 0.165 | 0.458 |
| FD · RBDL (CPU, float64) | 0.135 | 0.162 | 0.314 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.100 | 0.214 | 1.783 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.771 | 0.899 | 2.981 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.069 | 0.121 | 0.894 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.798 | 0.838 | 1.820 |
| ∂ID/∂(q, q̇) · frax | 1.154 | 1.987 | 8.763 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.162 | 0.207 | 0.841 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 256**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.042 | 0.151 |
| ID (RNEA) · mantis (JAX) | 0.691 | 0.660 | 0.804 |
| ID (RNEA) · GRiD (C++ bridge) | 0.057 | 0.094 | 0.242 |
| ID (RNEA) · GRiD (JAX FFI) | 0.766 | 0.802 | 0.966 |
| ID (RNEA) · frax | 0.847 | 1.020 | 1.474 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.080 | 0.145 | 0.204 |
| ID (RNEA) · RBDL (CPU, float64) | 0.128 | 0.142 | 0.274 |
| M (CRBA) · mantis (C bridge) | 0.018 | 0.042 | 0.377 |
| M (CRBA) · mantis (JAX) | 0.457 | 0.449 | 1.027 |
| M (CRBA) · GRiD (C++ bridge) | 0.073 | 0.102 | 0.476 |
| M (CRBA) · GRiD (JAX FFI) | 0.577 | 0.534 | 1.073 |
| M (CRBA) · frax | 0.761 | 0.925 | 1.594 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.117 | 0.165 | 0.319 |
| M (CRBA) · RBDL (CPU, float64) | 0.126 | 0.153 | 0.360 |
| FD · mantis (C bridge) | 0.026 | 0.075 | 0.249 |
| FD · mantis (JAX) | 0.678 | 0.764 | 0.973 |
| FD · GRiD (C++ bridge) | 0.077 | 0.130 | 0.633 |
| FD · GRiD (JAX FFI) | 0.752 | 0.828 | 1.368 |
| FD · frax | 0.961 | 1.145† | 2.003† |
| FD · Pinocchio codegen (CPU) | 0.150 | 0.198 | 0.730 |
| FD · RBDL (CPU, float64) | 0.159 | 0.275 | 0.512 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.102 | 0.296 | 2.132 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.759 | 1.000 | 3.222 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.087 | 0.171 | 1.765 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.771 | 0.873 | 2.896 |
| ∂ID/∂(q, q̇) · frax | 1.354 | 2.773 | 15.419 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.175 | 0.432 | 0.946 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 1024**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.027 | 0.044 | 0.164 |
| ID (RNEA) · mantis (JAX) | 0.703 | 0.797 | 1.125 |
| ID (RNEA) · GRiD (C++ bridge) | 0.160 | 0.246 | 0.839 |
| ID (RNEA) · GRiD (JAX FFI) | 0.891 | 0.995 | 1.552 |
| ID (RNEA) · frax | 0.949 | 1.106 | 1.926 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.154 | 0.280 | 0.529 |
| ID (RNEA) · RBDL (CPU, float64) | 0.218 | 0.330 | 0.699 |
| M (CRBA) · mantis (C bridge) | 0.032 | 0.076 | 0.657 |
| M (CRBA) · mantis (JAX) | 0.507 | 0.620 | 1.780 |
| M (CRBA) · GRiD (C++ bridge) | 0.192 | 0.278 | 1.458 |
| M (CRBA) · GRiD (JAX FFI) | 0.647 | 0.805 | 2.580 |
| M (CRBA) · frax | 0.795 | 1.221 | 2.976 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.183 | 0.367 | 0.824 |
| M (CRBA) · RBDL (CPU, float64) | 0.244 | 0.376 | 0.952 |
| FD · mantis (C bridge) | 0.035 | 0.077 | 0.262 |
| FD · mantis (JAX) | 0.751 | 0.819 | 1.256 |
| FD · GRiD (C++ bridge) | 0.241 | 0.359 | 1.982 |
| FD · GRiD (JAX FFI) | 0.975 | 1.118 | 2.824 |
| FD · frax | 1.069 | 1.391† | 3.034† |
| FD · Pinocchio codegen (CPU) | 0.318 | 0.608 | 1.626† |
| FD · RBDL (CPU, float64) | 0.360 | 0.640 | 1.341 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.115 | 0.377 | 2.554 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.781 | 1.200 | 4.879 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.260 | 0.545 | 6.941 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.945 | 1.399 | 8.901 |
| ∂ID/∂(q, q̇) · frax | 2.143 | 7.210 | 53.983 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.487 | 0.795 | 3.344 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16384**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.101 | 0.187 | 0.632 |
| ID (RNEA) · mantis (JAX) | 0.963 | 1.669 | 6.350 |
| ID (RNEA) · GRiD (C++ bridge) | 1.846 | 2.806 | OOM |
| ID (RNEA) · GRiD (JAX FFI) | 2.888 | 4.002 | 13.039 |
| ID (RNEA) · frax | 1.992 | 3.710 | 10.840 |
| ID (RNEA) · Pinocchio codegen (CPU) | 1.017 | 1.675 | 4.956 |
| ID (RNEA) · RBDL (CPU, float64) | 1.761 | 3.529 | 8.072 |
| M (CRBA) · mantis (C bridge) | 0.166 | 0.711 | 5.753 |
| M (CRBA) · mantis (JAX) | 0.881 | 2.538 | 76.009 |
| M (CRBA) · GRiD (C++ bridge) | 2.224 | 3.609 | OOM |
| M (CRBA) · GRiD (JAX FFI) | 3.013 | 5.026 | 137.336 |
| M (CRBA) · frax | 2.204 | 4.824 | 125.583 |
| M (CRBA) · Pinocchio codegen (CPU) | 1.101 | 2.854 | 14.210 |
| M (CRBA) · RBDL (CPU, float64) | 1.919 | 3.611 | 16.232 |
| FD · mantis (C bridge) | 0.110 | 0.240 | 0.875 |
| FD · mantis (JAX) | 0.962 | 1.622 | 6.496 |
| FD · GRiD (C++ bridge) | 3.243 | 4.659 | OOM |
| FD · GRiD (JAX FFI) | 4.383 | 5.882 | 31.534 |
| FD · frax | 2.955 | 6.020† | 26.618† |
| FD · Pinocchio codegen (CPU) | 2.301 | 5.138 | 20.577† |
| FD · RBDL (CPU, float64) | 3.575 | 7.182 | 16.706 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.485 | 1.852 | 13.624 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 1.849 | 6.069 | 253.090 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 3.619 | 7.322 | OOM |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 5.144 | 10.764 | OOM |
| ∂ID/∂(q, q̇) · frax | 20.687 | 100.577 | OOM |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 4.319 | 11.192 | 44.760 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

### Kinematic trajectory optimization: MotionBenchMaker (Panda), 250 iterations

800 MBM problems, seed 0, one problem per call, 32 waypoints (`bash run_trajopt.sh`, `bench_trajopt.py --trace`). GRiM (CHOMP, STOMP, SCO), pyroki and cuRobo start from the same 64 starts per problem (the straight line and 63 sine-bump perturbations of it, one batch; pyroki runs 6 iterations on all 64, then the rest on its best 4, its paper's IK beam); TrajOpt and Drake (CPU) from the straight line. Every method runs once to 250 of its own iterations and reports, every 10, the answer it would return if stopped there (its own pick over its starts, checked with its own collision code), graded in float64 on the shared sphere model (collision-free including dense interpolation, endpoints within 1e-3 rad, trajopt/mbm.py) and stamped with the wall time reached. Every method runs its tuned parameters, frozen on the MπNets tuning set (RUNNING.md, "Trajopt tuning"): GRiM STOMP, TrajOpt and Drake (marked tuned); the others kept their defaults.

| method | 250 iterations | clock |
|---|---|---|
| GRiM CHOMP / STOMP / SCO | kernel iterations (SCO: 25 inner x 10 outer L-BFGS steps) | kernel snapshots (`snap_every`); time split evenly by iteration within the one launch, plus that checkpoint's own check |
| pyroki | jaxls LM steps (6 + 244) | jaxls snapshots (external_patches/jaxls_snapshots.py); each beam phase timed, split evenly by iteration |
| cuRobo v2 ‡ | L-BFGS steps (225 trajopt + 25 time-optimal finetune), one attempt, no RRT | its debug recorder, a timestamp per iteration (CUDA graphs off) |
| TrajOpt (Tesseract) | SQP iterations | the harness's per-iteration iterates and times |
| Drake (SNOPT) | major iterations, per solve | every 10th function evaluation (its bundled SNOPT reports no major iterations) |

‡ cuRobo uses its own benchmark's 2.5 mm collision activation distance; every other method the shared 1 cm margin.

Table: each method's result at 250 iterations. Path length and smoothness: medians over the problems that method solved; time: median per problem.

![success over time](results/trajopt_trace_success.png)

![path quality over time](results/trajopt_trace_quality.png)

![Pareto](results/trajopt_trace_pareto.png)

| method | success % | path length (rad) | smoothness | median time (ms) |
|---|---|---|---|---|
| GRiM CHOMP (64 starts) | 94.1 | 4.037 | 0.0016 | 940.0 |
| GRiM STOMP (64 starts) (tuned) | 91.5 | 4.582 | 0.0027 | 5591.7 |
| GRiM SCO (64 starts) | 82.6 | 4.845 | 0.2654 | 116.4 |
| pyroki (jaxls, augmented Lagrangian, 64 starts) | 84.2 | 3.634 | 0.0289 | 14658.9 |
| cuRobo v2 MotionPlanner (64 starts, no RRT) ‡ | 67.4 | 3.903 | 0.0025 | 956.8 |
| TrajOpt (Tesseract, CPU) (tuned) | 60.9 | 3.638 | 0.0025 | 2188.5 |
| Drake KinematicTrajectoryOptimization (SNOPT, CPU) (tuned) | 57.4 | 3.711 | 0.0003 | 1833.1 |
<!-- /tables -->
