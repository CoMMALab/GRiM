# Benchmarks

FK, IK and dynamics of mantis against the libraries it competes with, on three robots: Franka
Panda (7 DOF), Rethink Baxter (14/15 DOF, two arms) and Unitree G1 with hands (43 DOF tree).

```bash
bash benchmarks/run_all.sh        # picks a free GPU, runs everything, rewrites the tables below
```

Setup, running a single benchmark, and the raw result files: see [RUNNING.md](RUNNING.md).

## Findings

- **FK alone:** mantis gains nothing over GRiM's baked tables (each runs inside the same JAX
  sphere placement). pRRTC and cuRobo are fastest at B = 16384 (0.19-0.65 ms); pyroki is
  8-10x slower.
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
same way for JAX, PyTorch and raw CUDA (the pRRTC harness runs the identical loop with CUDA
events). All runs use one GPU (RTX A5000).

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

**FK** computes the world centre of every collision sphere, `(B, S, 3)`: the one output all five
produce. pRRTC and cuRobo emit spheres from their kernels; pyroki, GRiM and mantis compute frame
poses and place the spheres in the same jit (identical placement code for all three).

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
- **GRiD/GRiM** `grim.motion.kinematics(traced=False)`: GRiM's FK kernel over GRiD-style baked
  tables, poses only (no end-effector Jacobian).
- **mantis** the same call with `traced=True`: mantis's straight-line FK.

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
| cuRobo | 0.144 | 0.139 | 0.140 |
| pyroki | 0.271 | 0.820 | 0.974 |
| GRiD/GRiM (baked tables) | 0.046 | 0.105 | 0.166 |
| mantis (traced) | 0.045 | 0.066 | 0.130 |
| GRiD/GRiM (baked tables), C bridge | 0.040 | 0.101 | 0.163 |
| mantis (traced), C bridge | 0.023 | 0.063 | 0.126 |

**B = 128**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.075 | 0.093 | 0.152 |
| cuRobo | 0.139 | 0.139 | 0.143 |
| pyroki | 0.352 | 1.112 | 1.329 |
| GRiD/GRiM (baked tables) | 0.096 | 0.290 | 0.408 |
| mantis (traced) | 0.104 | 0.370 | 0.505 |
| GRiD/GRiM (baked tables), C bridge | 0.091 | 0.285 | 0.394 |
| mantis (traced), C bridge | 0.098 | 0.364 | 0.490 |

**B = 1024**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.075 | 0.093 | 0.153 |
| cuRobo | 0.142 | 0.143 | 0.142 |
| pyroki | 0.344 | 1.058 | 1.270 |
| GRiD/GRiM (baked tables) | 0.108 | 0.323 | 0.425 |
| mantis (traced) | 0.117 | 0.406 | 0.531 |
| GRiD/GRiM (baked tables), C bridge | 0.098 | 0.292 | 0.399 |
| mantis (traced), C bridge | 0.105 | 0.372 | 0.504 |

**B = 16384**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 0.346 | 0.501 | 0.653 |
| cuRobo | 0.145 | 0.498 | 0.772 |
| pyroki | 1.532 | 4.885 | 5.961 |
| GRiD/GRiM (baked tables) | 0.466 | 1.346 | 1.760 |
| mantis (traced) | 0.469 | 1.506 | 1.967 |
| GRiD/GRiM (baked tables), C bridge | 0.376 | 1.067 | 1.282 |
| mantis (traced), C bridge | 0.380 | 1.226 | 1.487 |

**Max sphere-centre error vs Pinocchio float64 (m), over all batches**

| method | panda | baxter | g1 |
|---|---|---|---|
| pRRTC `fk<Robot>` | 8.4e-07 | 1.3e-06 | 1.1e-06 |
| cuRobo | 2.0e-07 | 3.1e-07 | 1.6e-07 |
| pyroki | 2.8e-07 | 4.9e-07 | 3.3e-07 |
| GRiD/GRiM (baked tables) | 2.9e-07 | 6.1e-07 | 3.2e-07 |
| mantis (traced) | 2.0e-07 | 2.4e-07 | 1.7e-07 |
| GRiD/GRiM (baked tables), C bridge | 2.9e-07 | 6.1e-07 | 3.2e-07 |
| mantis (traced), C bridge | 2.0e-07 | 2.4e-07 | 1.4e-07 |

### Global IK: one end-effector pose from random seeds

GRiM: 8 seeds, 60 iterations max, fastest tier; pyroki, cuRobo and frax as their own benchmarks configure them (frax runs on the CPU). Time per call in ms, then success rate (position < 1 mm, orientation < 0.01 rad, within the URDF limits; Pinocchio float64).

**1 problem**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK | 2.71 (100.0%, thread) | 2.99 (100.0%, thread) | 4.35 (100.0%, warp) |
| GRiM DLS IK | 0.88 (100.0%, thread) | 0.91 (100.0%, thread) | 1.87 (100.0%, thread) |
| GRiM SQP IK | 10.86 (100.0%, thread) | 10.46 (100.0%, thread) | 29.80 (100.0%, warp) |
| GRiM HJCD IK (coarse CD -> LM) | 2.89 (100.0%, thread) | 2.82 (100.0%, thread) | 5.21 (100.0%, block) |
| GRiM MPPI IK (MPPI -> L-BFGS) | 2.84 (100.0%, thread) | 3.82 (100.0%, block) | 10.59 (100.0%, thread) |
| GRiM LS IK, C bridge | 2.70 (100.0%, thread) | 2.98 (100.0%, thread) | 4.34 (100.0%, warp) |
| GRiM DLS IK, C bridge | 0.87 (100.0%, thread) | 0.90 (100.0%, thread) | 1.86 (100.0%, thread) |
| GRiM SQP IK, C bridge | 10.85 (100.0%, thread) | 10.44 (100.0%, thread) | 29.79 (100.0%, warp) |
| GRiM HJCD IK, C bridge | 2.88 (100.0%, thread) | 2.81 (100.0%, thread) | 5.20 (100.0%, block) |
| GRiM MPPI IK, C bridge | 2.83 (100.0%, thread) | 3.80 (100.0%, block) | 10.58 (100.0%, thread) |
| pyroki IK-Beam (analytic Jacobian; own 64->4 seeds) | 19.44 (100.0%, ik-beam 64->4) | 49.47 (100.0%, ik-beam 64->4) | 52.71 (100.0%, ik-beam 64->4) |
| cuRobo v2 IKSolver (LM-seeded L-BFGS; own seeds) | 4.52 (100.0%, lbfgs+lm-seed) | 4.51 (100.0%, lbfgs+lm-seed) | 4.69 (100.0%, lbfgs+lm-seed) |
| frax GlobalIKSolver (multi-start diff-IK SQP, ElastiQP; 32 starts; CPU float64) | — | — | — |

**128 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK | 2.88 (97.7%, thread) | 3.04 (97.7%, thread) | 6.88 (99.2%, thread) |
| GRiM DLS IK | 0.89 (95.3%, thread) | 0.93 (94.5%, thread) | 1.88 (97.7%, thread) |
| GRiM SQP IK | 11.10 (99.2%, thread) | 11.11 (99.2%, thread) | 44.90 (100.0%, thread) |
| GRiM HJCD IK (coarse CD -> LM) | 2.94 (90.6%, thread) | 2.97 (84.4%, thread) | 6.49 (97.7%, thread) |
| GRiM MPPI IK (MPPI -> L-BFGS) | 2.94 (98.4%, thread) | 4.50 (96.1%, thread) | 10.96 (100.0%, thread) |
| GRiM LS IK, C bridge | 2.87 (97.7%, thread) | 3.03 (97.7%, thread) | 6.87 (99.2%, thread) |
| GRiM DLS IK, C bridge | 0.88 (95.3%, thread) | 0.92 (94.5%, thread) | 1.87 (97.7%, thread) |
| GRiM SQP IK, C bridge | 11.09 (99.2%, thread) | 11.10 (99.2%, thread) | 44.88 (100.0%, thread) |
| GRiM HJCD IK, C bridge | 2.93 (90.6%, thread) | 2.96 (84.4%, thread) | 6.48 (97.7%, thread) |
| GRiM MPPI IK, C bridge | 2.93 (98.4%, thread) | 4.49 (96.1%, thread) | 10.96 (100.0%, thread) |
| pyroki IK-Beam (analytic Jacobian; own 64->4 seeds) | 62.55 (100.0%, ik-beam 64->4) | 179.94 (99.2%, ik-beam 64->4) | 336.58 (100.0%, ik-beam 64->4) |
| cuRobo v2 IKSolver (LM-seeded L-BFGS; own seeds) | 8.81 (100.0%, lbfgs+lm-seed) | 8.95 (100.0%, lbfgs+lm-seed) | 10.13 (100.0%, lbfgs+lm-seed) |
| frax GlobalIKSolver (multi-start diff-IK SQP, ElastiQP; 32 starts; CPU float64) | — | — | — |

**1024 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| GRiM LS IK | 2.81 (96.7%, thread) | 2.93 (96.2%, thread) | 7.18 (98.3%, thread) |
| GRiM DLS IK | 0.90 (94.4%, thread) | 0.93 (96.0%, thread) | 2.23 (98.3%, thread) |
| GRiM SQP IK | 11.43 (99.3%, thread) | 11.40 (99.3%, thread) | 46.43 (100.0%, thread) |
| GRiM HJCD IK (coarse CD -> LM) | 2.95 (90.2%, thread) | 3.01 (86.6%, thread) | 7.08 (96.4%, thread) |
| GRiM MPPI IK (MPPI -> L-BFGS) | 3.07 (97.8%, thread) | 5.29 (97.5%, thread) | 16.76 (99.8%, thread) |
| GRiM LS IK, C bridge | 2.87 (96.7%, thread) | 2.92 (96.2%, thread) | 7.17 (98.3%, thread) |
| GRiM DLS IK, C bridge | 0.92 (94.4%, thread) | 0.92 (96.0%, thread) | 2.22 (98.3%, thread) |
| GRiM SQP IK, C bridge | 11.42 (99.3%, thread) | 11.39 (99.3%, thread) | 46.41 (100.0%, thread) |
| GRiM HJCD IK, C bridge | 2.94 (90.2%, thread) | 2.99 (86.6%, thread) | 7.07 (96.4%, thread) |
| GRiM MPPI IK, C bridge | 3.05 (97.8%, thread) | 5.28 (97.5%, thread) | 16.87 (99.8%, thread) |
| pyroki IK-Beam (analytic Jacobian; own 64->4 seeds) | 297.20 (100.0%, ik-beam 64->4) | 1104.34 (99.6%, ik-beam 64->4) | 2523.29 (99.9%, ik-beam 64->4) |
| cuRobo v2 IKSolver (LM-seeded L-BFGS; own seeds) | 79.05 (100.0%, lbfgs+lm-seed) | 43.73 (100.0%, lbfgs+lm-seed) | 52.72 (100.0%, lbfgs+lm-seed) |
| frax GlobalIKSolver (multi-start diff-IK SQP, ElastiQP; 32 starts; CPU float64) | — | — | — |

### Supplementary: mantis IK steps

mantis's traced IK steps in the benchmark's own minimal CUDA loops (one seed per thread, 8 seeds, 60 iterations), across precisions. A profile of the trace, not a comparison with the solvers above.

**1 problem**

| method | panda | baxter | g1 |
|---|---|---|---|
| mantis `ik_lm_step` (6x6 LM) | 1.54 (100.0%, thread) | 1.66 (100.0%, thread) | 2.16 (100.0%, thread) |
| mantis `ik_ls_step`, float | 2.98 (100.0%, f32) | 3.20 (100.0%, f32) | 4.25 (100.0%, f32) |
| mantis `ik_ls_step`, float system / double solve | 10.19 (100.0%, mixed) | 10.36 (100.0%, mixed) | 16.64 (100.0%, mixed) |
| mantis `ik_ls_step`, double | 24.42 (100.0%, f64) | 25.31 (100.0%, f64) | 36.33 (100.0%, f64) |
| mantis SQP step, float | 18.70 (100.0%, f32) | 18.47 (100.0%, f32) | 34.29 (100.0%, f32) |
| mantis SQP step, float FK / double QP | 84.11 (100.0%, mixed) | 83.33 (100.0%, mixed) | 161.50 (100.0%, mixed) |
| mantis SQP step, double | 103.98 (100.0%, f64) | 102.10 (100.0%, f64) | 188.06 (100.0%, f64) |

**128 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| mantis `ik_lm_step` (6x6 LM) | 1.53 (96.1%, thread) | 1.67 (95.3%, thread) | 2.30 (97.7%, thread) |
| mantis `ik_ls_step`, float | 3.21 (96.9%, f32) | 3.30 (96.9%, f32) | 4.67 (99.2%, f32) |
| mantis `ik_ls_step`, float system / double solve | 10.24 (97.7%, mixed) | 10.34 (97.7%, mixed) | 16.47 (99.2%, mixed) |
| mantis `ik_ls_step`, double | 24.51 (97.7%, f64) | 25.38 (97.7%, f64) | 36.37 (99.2%, f64) |
| mantis SQP step, float | 19.10 (99.2%, f32) | 19.12 (99.2%, f32) | 34.93 (100.0%, f32) |
| mantis SQP step, float FK / double QP | 84.60 (99.2%, mixed) | 86.33 (99.2%, mixed) | 163.55 (100.0%, mixed) |
| mantis SQP step, double | 104.81 (99.2%, f64) | 105.39 (99.2%, f64) | 191.12 (100.0%, f64) |

**1024 problems**

| method | panda | baxter | g1 |
|---|---|---|---|
| mantis `ik_lm_step` (6x6 LM) | 1.57 (94.6%, thread) | 1.83 (95.5%, thread) | 2.80 (98.4%, thread) |
| mantis `ik_ls_step`, float | 3.06 (96.3%, f32) | 3.15 (96.2%, f32) | 4.52 (98.3%, f32) |
| mantis `ik_ls_step`, float system / double solve | 14.29 (96.2%, mixed) | 14.35 (95.8%, mixed) | 23.81 (98.5%, mixed) |
| mantis `ik_ls_step`, double | 47.55 (96.2%, f64) | 49.66 (95.8%, f64) | 67.49 (98.5%, f64) |
| mantis SQP step, float | 19.00 (99.3%, f32) | 19.00 (99.2%, f32) | 35.01 (100.0%, f32) |
| mantis SQP step, float FK / double QP | 117.97 (99.3%, mixed) | 120.12 (99.4%, mixed) | 223.19 (100.0%, mixed) |
| mantis SQP step, double | 158.42 (99.3%, f64) | 159.64 (99.4%, f64) | 275.20 (100.0%, f64) |

### Dynamics: GRiD's release protocol

Time per call in ms, median over 3 processes of 30 samples each (GRiD's `collect.py` defaults), every cell checked against an independent float64 Pinocchio oracle under GRiD's release-table policy (`fp32-fd-warnings`): †: a float32 FD cell past the strict entrywise gate but within relative L2 1e-3, kept with a warning as GRiD keeps it. GPU methods are timed device-resident (inputs and outputs on the GPU); the CPU libraries take host arrays, so theirs is the host call, as in GRiD's Figure 2b. B = 1 and 16384 are additions to GRiD's batches. n/a: the library has no such operation; OOM: the call's workspace does not fit in the GPU's 24 GB. mantis, GRiD (JAX FFI) and frax are called through JAX, so their resident time includes JAX dispatch (about 0.1 ms, which dominates below B = 1024); GRiD's C++ bridge times its kernel launch directly.

**B = 1**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.021 | 0.089 |
| ID (RNEA) · mantis (JAX) | 0.154 | 0.181 | 0.256 |
| ID (RNEA) · GRiD (C++ bridge) | 0.034 | 0.053 | 0.127 |
| ID (RNEA) · GRiD (JAX FFI) | 0.206 | 0.229 | 0.321 |
| ID (RNEA) · frax | 0.254 | 0.408 | 0.837 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.004 | 0.004 | 0.005 |
| ID (RNEA) · RBDL (CPU, float64) | 0.004 | 0.004 | 0.007 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.198 |
| M (CRBA) · mantis (JAX) | 0.143 | 0.135 | 0.347 |
| M (CRBA) · GRiD (C++ bridge) | 0.032 | 0.051 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.160 | 0.189 | 0.290 |
| M (CRBA) · frax | 0.246 | 0.424 | 0.854 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.004 | 0.005 | 0.009 |
| M (CRBA) · RBDL (CPU, float64) | 0.004 | 0.005 | 0.009 |
| FD · mantis (C bridge) | 0.016 | 0.051 | 0.183 |
| FD · mantis (JAX) | 0.158 | 0.212 | 0.340 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.263 |
| FD · GRiD (JAX FFI) | 0.229 | 0.260 | 0.444 |
| FD · frax | 0.367 | 0.526 | 1.026† |
| FD · Pinocchio codegen (CPU) | 0.004 | 0.006 | 0.015 |
| FD · RBDL (CPU, float64) | 0.005 | 0.006 | 0.013 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.076 | 0.170 | 1.259 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.243 | 0.328 | 1.467 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.405 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.244 | 0.267 | 0.589 |
| ∂ID/∂(q, q̇) · frax | 0.441 | 0.722 | 2.009 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.006 | 0.009 | 0.028 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.021 | 0.084 |
| ID (RNEA) · mantis (JAX) | 0.170 | 0.138 | 0.272 |
| ID (RNEA) · GRiD (C++ bridge) | 0.033 | 0.053 | 0.126 |
| ID (RNEA) · GRiD (JAX FFI) | 0.164 | 0.232 | 0.313 |
| ID (RNEA) · frax | 0.270 | 0.416 | 0.855 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.008 | 0.014 | 0.036 |
| ID (RNEA) · RBDL (CPU, float64) | 0.013 | 0.025 | 0.067 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.199 |
| M (CRBA) · mantis (JAX) | 0.143 | 0.177 | 0.383 |
| M (CRBA) · GRiD (C++ bridge) | 0.032 | 0.050 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.149 | 0.203 | 0.289 |
| M (CRBA) · frax | 0.265 | 0.453 | 0.910 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.010 | 0.022 | 0.083 |
| M (CRBA) · RBDL (CPU, float64) | 0.014 | 0.025 | 0.101 |
| FD · mantis (C bridge) | 0.015 | 0.051 | 0.183 |
| FD · mantis (JAX) | 0.149 | 0.229 | 0.350 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.258 |
| FD · GRiD (JAX FFI) | 0.235 | 0.260 | 0.449 |
| FD · frax | 0.370 | 0.535 | 1.181† |
| FD · Pinocchio codegen (CPU) | 0.019 | 0.039 | 0.142 |
| FD · RBDL (CPU, float64) | 0.026 | 0.050 | 0.118 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.076 | 0.166 | 1.268 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.234 | 0.323 | 1.460 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.403 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.236 | 0.256 | 0.625 |
| ∂ID/∂(q, q̇) · frax | 0.511 | 0.902 | 3.021 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.033 | 0.077 | 0.176 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 32**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.021 | 0.085 |
| ID (RNEA) · mantis (JAX) | 0.168 | 0.148 | 0.237 |
| ID (RNEA) · GRiD (C++ bridge) | 0.033 | 0.053 | 0.126 |
| ID (RNEA) · GRiD (JAX FFI) | 0.213 | 0.241 | 0.316 |
| ID (RNEA) · frax | 0.281 | 0.420 | 0.874 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.012 | 0.025 | 0.068 |
| ID (RNEA) · RBDL (CPU, float64) | 0.024 | 0.047 | 0.126 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.202 |
| M (CRBA) · mantis (JAX) | 0.148 | 0.160 | 0.371 |
| M (CRBA) · GRiD (C++ bridge) | 0.032 | 0.050 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.141 | 0.204 | 0.304 |
| M (CRBA) · frax | 0.262 | 0.453 | 0.870 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.017 | 0.041 | 0.134 |
| M (CRBA) · RBDL (CPU, float64) | 0.025 | 0.048 | 0.125 |
| FD · mantis (C bridge) | 0.016 | 0.051 | 0.184 |
| FD · mantis (JAX) | 0.175 | 0.211 | 0.351 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.258 |
| FD · GRiD (JAX FFI) | 0.234 | 0.272 | 0.448 |
| FD · frax | 0.391 | 0.550 | 1.221† |
| FD · Pinocchio codegen (CPU) | 0.034 | 0.073 | 0.155 |
| FD · RBDL (CPU, float64) | 0.049 | 0.098 | 0.141 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.076 | 0.167 | 1.297 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.240 | 0.350 | 1.499 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.404 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.228 | 0.264 | 0.601 |
| ∂ID/∂(q, q̇) · frax | 0.630 | 1.004 | 3.860 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.062 | 0.138 | 0.207 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 64**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.022 | 0.086 |
| ID (RNEA) · mantis (JAX) | 0.169 | 0.173 | 0.257 |
| ID (RNEA) · GRiD (C++ bridge) | 0.033 | 0.053 | 0.126 |
| ID (RNEA) · GRiD (JAX FFI) | 0.205 | 0.237 | 0.312 |
| ID (RNEA) · frax | 0.279 | 0.436 | 0.961 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.022 | 0.046 | 0.129 |
| ID (RNEA) · RBDL (CPU, float64) | 0.045 | 0.091 | 0.129 |
| M (CRBA) · mantis (C bridge) | 0.012 | 0.020 | 0.240 |
| M (CRBA) · mantis (JAX) | 0.146 | 0.151 | 0.421 |
| M (CRBA) · GRiD (C++ bridge) | 0.032 | 0.050 | 0.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.161 | 0.199 | 0.300 |
| M (CRBA) · frax | 0.272 | 0.459 | 0.887 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.031 | 0.079 | 0.160 |
| M (CRBA) · RBDL (CPU, float64) | 0.047 | 0.093 | 0.150 |
| FD · mantis (C bridge) | 0.016 | 0.053 | 0.196 |
| FD · mantis (JAX) | 0.155 | 0.217 | 0.361 |
| FD · GRiD (C++ bridge) | 0.053 | 0.088 | 0.256 |
| FD · GRiD (JAX FFI) | 0.231 | 0.269 | 0.443 |
| FD · frax | 0.391 | 0.555† | 1.315† |
| FD · Pinocchio codegen (CPU) | 0.064 | 0.136 | 0.226 |
| FD · RBDL (CPU, float64) | 0.096 | 0.126 | 0.199 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.078 | 0.170 | 1.353 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.267 | 0.326 | 1.582 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.049 | 0.086 | 0.409 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.241 | 0.276 | 0.626 |
| ∂ID/∂(q, q̇) · frax | 0.616 | 1.274 | 5.444 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.120 | 0.149 | 0.394 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 128**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.024 | 0.131 |
| ID (RNEA) · mantis (JAX) | 0.176 | 0.165 | 0.319 |
| ID (RNEA) · GRiD (C++ bridge) | 0.037 | 0.061 | 0.154 |
| ID (RNEA) · GRiD (JAX FFI) | 0.211 | 0.252 | 0.339 |
| ID (RNEA) · frax | 0.287 | 0.486 | 0.995 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.040 | 0.089 | 0.156 |
| ID (RNEA) · RBDL (CPU, float64) | 0.087 | 0.131 | 0.149 |
| M (CRBA) · mantis (C bridge) | 0.013 | 0.024 | 0.286 |
| M (CRBA) · mantis (JAX) | 0.141 | 0.160 | 0.483 |
| M (CRBA) · GRiD (C++ bridge) | 0.034 | 0.058 | 0.167 |
| M (CRBA) · GRiD (JAX FFI) | 0.133 | 0.204 | 0.345 |
| M (CRBA) · frax | 0.275 | 0.466 | 0.953 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.058 | 0.128 | 0.193 |
| M (CRBA) · RBDL (CPU, float64) | 0.091 | 0.122 | 0.230 |
| FD · mantis (C bridge) | 0.016 | 0.059 | 0.233 |
| FD · mantis (JAX) | 0.156 | 0.205 | 0.401 |
| FD · GRiD (C++ bridge) | 0.057 | 0.097 | 0.290 |
| FD · GRiD (JAX FFI) | 0.243 | 0.298 | 0.493 |
| FD · frax | 0.387 | 0.605† | 1.387† |
| FD · Pinocchio codegen (CPU) | 0.124 | 0.150 | 0.289 |
| FD · RBDL (CPU, float64) | 0.121 | 0.146 | 0.300 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.092 | 0.195 | 1.817 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.261 | 0.363 | 2.146 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.055 | 0.099 | 0.807 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.237 | 0.284 | 1.058 |
| ∂ID/∂(q, q̇) · frax | 0.706 | 1.499 | 8.245 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.139 | 0.198 | 0.672 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 256**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.012 | 0.024 | 0.131 |
| ID (RNEA) · mantis (JAX) | 0.149 | 0.150 | 0.307 |
| ID (RNEA) · GRiD (C++ bridge) | 0.045 | 0.081 | 0.221 |
| ID (RNEA) · GRiD (JAX FFI) | 0.232 | 0.275 | 0.409 |
| ID (RNEA) · frax | 0.336 | 0.503 | 1.084 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.077 | 0.127 | 0.194 |
| ID (RNEA) · RBDL (CPU, float64) | 0.119 | 0.133 | 0.247 |
| M (CRBA) · mantis (C bridge) | 0.013 | 0.024 | 0.288 |
| M (CRBA) · mantis (JAX) | 0.175 | 0.144 | 0.510 |
| M (CRBA) · GRiD (C++ bridge) | 0.064 | 0.078 | 0.387 |
| M (CRBA) · GRiD (JAX FFI) | 0.217 | 0.236 | 0.562 |
| M (CRBA) · frax | 0.280 | 0.483 | 1.027 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.113 | 0.150 | 0.348 |
| M (CRBA) · RBDL (CPU, float64) | 0.120 | 0.146 | 0.348 |
| FD · mantis (C bridge) | 0.016 | 0.059 | 0.234 |
| FD · mantis (JAX) | 0.152 | 0.221 | 0.398 |
| FD · GRiD (C++ bridge) | 0.066 | 0.119 | 0.613 |
| FD · GRiD (JAX FFI) | 0.255 | 0.308 | 0.825 |
| FD · frax | 0.441 | 0.628† | 1.519† |
| FD · Pinocchio codegen (CPU) | 0.143 | 0.192 | 0.643 |
| FD · RBDL (CPU, float64) | 0.138 | 0.236 | 0.375 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.092 | 0.260 | 1.958 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.257 | 0.443 | 2.254 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.071 | 0.141 | 1.600 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.283 | 0.345 | 1.898 |
| ∂ID/∂(q, q̇) · frax | 0.874 | 2.231 | 14.836 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.178 | 0.260 | 0.928 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 1024**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.013 | 0.024 | 0.124 |
| ID (RNEA) · mantis (JAX) | 0.163 | 0.151 | 0.281 |
| ID (RNEA) · GRiD (C++ bridge) | 0.143 | 0.223 | 0.791 |
| ID (RNEA) · GRiD (JAX FFI) | 0.339 | 0.424 | 0.999 |
| ID (RNEA) · frax | 0.396 | 0.635 | 1.425 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.141 | 0.210 | 0.384 |
| ID (RNEA) · RBDL (CPU, float64) | 0.163 | 0.313 | 0.593 |
| M (CRBA) · mantis (C bridge) | 0.013 | 0.024 | 0.334 |
| M (CRBA) · mantis (JAX) | 0.129 | 0.189 | 0.664 |
| M (CRBA) · GRiD (C++ bridge) | 0.168 | 0.226 | 1.139 |
| M (CRBA) · GRiD (JAX FFI) | 0.329 | 0.391 | 1.356 |
| M (CRBA) · frax | 0.332 | 0.616 | 1.471 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.158 | 0.321 | 0.779 |
| M (CRBA) · RBDL (CPU, float64) | 0.214 | 0.308 | 1.020 |
| FD · mantis (C bridge) | 0.016 | 0.059 | 0.225 |
| FD · mantis (JAX) | 0.168 | 0.216 | 0.409 |
| FD · GRiD (C++ bridge) | 0.227 | 0.341 | 1.933 |
| FD · GRiD (JAX FFI) | 0.428 | 0.549 | 2.167 |
| FD · frax | 0.558 | 0.855† | 2.530† |
| FD · Pinocchio codegen (CPU) | 0.298 | 0.447 | 1.552† |
| FD · RBDL (CPU, float64) | 0.323 | 0.511 | 1.303 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.087 | 0.296 | 1.911 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.260 | 0.529 | 2.463 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.229 | 0.451 | 6.323 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.435 | 0.696 | 6.814 |
| ∂ID/∂(q, q̇) · frax | 1.668 | 6.740 | 52.557 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.467 | 0.737 | 3.194 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16384**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.016 | 0.029 | 0.186 |
| ID (RNEA) · mantis (JAX) | 0.169 | 0.211 | 0.426 |
| ID (RNEA) · GRiD (C++ bridge) | 1.745 | 2.608 | OOM |
| ID (RNEA) · GRiD (JAX FFI) | 2.172 | 3.055 | 11.406 |
| ID (RNEA) · frax | 1.437 | 2.976 | 9.899 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.743 | 1.610 | 4.740 |
| ID (RNEA) · RBDL (CPU, float64) | 1.497 | 3.390 | 8.019 |
| M (CRBA) · mantis (C bridge) | 0.020 | 0.056 | 0.772 |
| M (CRBA) · mantis (JAX) | 0.217 | 0.451 | 11.781 |
| M (CRBA) · GRiD (C++ bridge) | 2.042 | 2.925 | OOM |
| M (CRBA) · GRiD (JAX FFI) | 2.330 | 3.218 | 17.098 |
| M (CRBA) · frax | 1.346 | 3.057 | 11.008 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.984 | 2.651 | 12.605 |
| M (CRBA) · RBDL (CPU, float64) | 1.543 | 3.342 | 15.116 |
| FD · mantis (C bridge) | 0.023 | 0.077 | 0.425 |
| FD · mantis (JAX) | 0.154 | 0.250 | 0.652 |
| FD · GRiD (C++ bridge) | 3.162 | 4.505 | OOM |
| FD · GRiD (JAX FFI) | 3.605 | 5.003 | 29.377 |
| FD · frax | 2.483 | 5.287† | 25.611† |
| FD · Pinocchio codegen (CPU) | 2.093 | 4.969 | 18.745† |
| FD · RBDL (CPU, float64) | 3.301 | 6.610 | 16.254 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.154 | 0.529 | 4.013 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.438 | 1.424 | 26.792 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 3.263 | 6.304 | OOM |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 3.868 | 7.034 | OOM |
| ∂ID/∂(q, q̇) · frax | 19.930 | 97.997 | OOM |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 3.945 | 10.207 | 42.835 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

#### Host to host (host arrays in and out, including transfers)


**B = 1**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.026 | 0.099 |
| ID (RNEA) · mantis (JAX) | 0.798 | 0.766 | 0.835 |
| ID (RNEA) · GRiD (C++ bridge) | 0.046 | 0.066 | 0.139 |
| ID (RNEA) · GRiD (JAX FFI) | 0.767 | 0.798 | 0.896 |
| ID (RNEA) · frax | 0.807 | 0.966 | 1.309 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.004 | 0.004 | 0.005 |
| ID (RNEA) · RBDL (CPU, float64) | 0.004 | 0.004 | 0.007 |
| M (CRBA) · mantis (C bridge) | 0.017 | 0.026 | 0.206 |
| M (CRBA) · mantis (JAX) | 0.491 | 0.509 | 0.711 |
| M (CRBA) · GRiD (C++ bridge) | 0.040 | 0.058 | 0.147 |
| M (CRBA) · GRiD (JAX FFI) | 0.487 | 0.572 | 0.639 |
| M (CRBA) · frax | 0.796 | 0.947 | 1.327 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.004 | 0.005 | 0.009 |
| M (CRBA) · RBDL (CPU, float64) | 0.004 | 0.005 | 0.009 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.189 |
| FD · mantis (JAX) | 0.702 | 0.788 | 0.929 |
| FD · GRiD (C++ bridge) | 0.064 | 0.099 | 0.274 |
| FD · GRiD (JAX FFI) | 0.790 | 0.794 | 1.003 |
| FD · frax | 0.877 | 1.083 | 1.557† |
| FD · Pinocchio codegen (CPU) | 0.004 | 0.006 | 0.015 |
| FD · RBDL (CPU, float64) | 0.005 | 0.006 | 0.013 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.181 | 1.265 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.794 | 0.872 | 2.044 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.061 | 0.099 | 0.419 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.771 | 0.831 | 1.144 |
| ∂ID/∂(q, q̇) · frax | 0.990 | 1.197 | 2.274 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.006 | 0.009 | 0.028 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.026 | 0.092 |
| ID (RNEA) · mantis (JAX) | 0.767 | 0.744 | 0.830 |
| ID (RNEA) · GRiD (C++ bridge) | 0.046 | 0.066 | 0.140 |
| ID (RNEA) · GRiD (JAX FFI) | 0.738 | 0.836 | 0.842 |
| ID (RNEA) · frax | 0.822 | 0.986 | 1.376 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.008 | 0.014 | 0.036 |
| ID (RNEA) · RBDL (CPU, float64) | 0.013 | 0.025 | 0.067 |
| M (CRBA) · mantis (C bridge) | 0.017 | 0.025 | 0.210 |
| M (CRBA) · mantis (JAX) | 0.522 | 0.554 | 0.742 |
| M (CRBA) · GRiD (C++ bridge) | 0.039 | 0.058 | 0.150 |
| M (CRBA) · GRiD (JAX FFI) | 0.516 | 0.567 | 0.657 |
| M (CRBA) · frax | 0.769 | 0.929 | 1.329 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.010 | 0.022 | 0.083 |
| M (CRBA) · RBDL (CPU, float64) | 0.014 | 0.025 | 0.101 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.190 |
| FD · mantis (JAX) | 0.774 | 0.747 | 0.869 |
| FD · GRiD (C++ bridge) | 0.064 | 0.099 | 0.270 |
| FD · GRiD (JAX FFI) | 0.844 | 0.806 | 1.017 |
| FD · frax | 0.949 | 1.108 | 1.672† |
| FD · Pinocchio codegen (CPU) | 0.019 | 0.039 | 0.142 |
| FD · RBDL (CPU, float64) | 0.026 | 0.050 | 0.118 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.175 | 1.284 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.798 | 0.876 | 2.065 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.061 | 0.100 | 0.424 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.819 | 0.814 | 1.191 |
| ∂ID/∂(q, q̇) · frax | 1.060 | 1.409 | 3.191 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.033 | 0.077 | 0.176 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 32**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.025 | 0.092 |
| ID (RNEA) · mantis (JAX) | 0.717 | 0.762 | 0.835 |
| ID (RNEA) · GRiD (C++ bridge) | 0.046 | 0.066 | 0.140 |
| ID (RNEA) · GRiD (JAX FFI) | 0.754 | 0.819 | 0.878 |
| ID (RNEA) · frax | 0.806 | 0.994 | 1.374 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.012 | 0.025 | 0.068 |
| ID (RNEA) · RBDL (CPU, float64) | 0.024 | 0.047 | 0.126 |
| M (CRBA) · mantis (C bridge) | 0.016 | 0.026 | 0.215 |
| M (CRBA) · mantis (JAX) | 0.539 | 0.518 | 0.720 |
| M (CRBA) · GRiD (C++ bridge) | 0.040 | 0.058 | 0.155 |
| M (CRBA) · GRiD (JAX FFI) | 0.553 | 0.555 | 0.731 |
| M (CRBA) · frax | 0.819 | 0.964 | 1.336 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.017 | 0.041 | 0.134 |
| M (CRBA) · RBDL (CPU, float64) | 0.025 | 0.048 | 0.125 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.190 |
| FD · mantis (JAX) | 0.689 | 0.817 | 0.983 |
| FD · GRiD (C++ bridge) | 0.064 | 0.100 | 0.271 |
| FD · GRiD (JAX FFI) | 0.793 | 0.820 | 1.003 |
| FD · frax | 0.915 | 1.153 | 1.726† |
| FD · Pinocchio codegen (CPU) | 0.034 | 0.073 | 0.155 |
| FD · RBDL (CPU, float64) | 0.049 | 0.098 | 0.141 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.175 | 1.324 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.822 | 0.910 | 2.173 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.062 | 0.099 | 0.435 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.767 | 0.811 | 1.227 |
| ∂ID/∂(q, q̇) · frax | 1.165 | 1.498 | 4.000 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.062 | 0.138 | 0.207 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 64**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.016 | 0.028 | 0.099 |
| ID (RNEA) · mantis (JAX) | 0.737 | 0.801 | 0.883 |
| ID (RNEA) · GRiD (C++ bridge) | 0.046 | 0.066 | 0.139 |
| ID (RNEA) · GRiD (JAX FFI) | 0.769 | 0.804 | 0.887 |
| ID (RNEA) · frax | 0.857 | 0.977 | 1.465 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.022 | 0.046 | 0.129 |
| ID (RNEA) · RBDL (CPU, float64) | 0.045 | 0.091 | 0.129 |
| M (CRBA) · mantis (C bridge) | 0.016 | 0.027 | 0.264 |
| M (CRBA) · mantis (JAX) | 0.520 | 0.499 | 0.824 |
| M (CRBA) · GRiD (C++ bridge) | 0.039 | 0.059 | 0.164 |
| M (CRBA) · GRiD (JAX FFI) | 0.551 | 0.570 | 0.674 |
| M (CRBA) · frax | 0.822 | 1.013 | 1.334 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.031 | 0.079 | 0.160 |
| M (CRBA) · RBDL (CPU, float64) | 0.047 | 0.093 | 0.150 |
| FD · mantis (C bridge) | 0.026 | 0.058 | 0.215 |
| FD · mantis (JAX) | 0.786 | 0.812 | 0.999 |
| FD · GRiD (C++ bridge) | 0.064 | 0.100 | 0.270 |
| FD · GRiD (JAX FFI) | 0.819 | 0.832 | 1.045 |
| FD · frax | 0.995 | 1.122† | 1.886† |
| FD · Pinocchio codegen (CPU) | 0.064 | 0.136 | 0.226 |
| FD · RBDL (CPU, float64) | 0.096 | 0.126 | 0.199 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.083 | 0.178 | 1.405 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.813 | 0.930 | 2.333 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.061 | 0.102 | 0.459 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.835 | 0.867 | 1.291 |
| ∂ID/∂(q, q̇) · frax | 1.197 | 1.660 | 5.704 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.120 | 0.149 | 0.394 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 128**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.016 | 0.034 | 0.148 |
| ID (RNEA) · mantis (JAX) | 0.750 | 0.731 | 0.911 |
| ID (RNEA) · GRiD (C++ bridge) | 0.048 | 0.074 | 0.169 |
| ID (RNEA) · GRiD (JAX FFI) | 0.796 | 0.814 | 0.898 |
| ID (RNEA) · frax | 0.854 | 1.039 | 1.551 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.040 | 0.089 | 0.156 |
| ID (RNEA) · RBDL (CPU, float64) | 0.087 | 0.131 | 0.149 |
| M (CRBA) · mantis (C bridge) | 0.017 | 0.037 | 0.332 |
| M (CRBA) · mantis (JAX) | 0.558 | 0.498 | 0.972 |
| M (CRBA) · GRiD (C++ bridge) | 0.042 | 0.070 | 0.220 |
| M (CRBA) · GRiD (JAX FFI) | 0.556 | 0.576 | 0.801 |
| M (CRBA) · frax | 0.810 | 0.977 | 1.544 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.058 | 0.128 | 0.193 |
| M (CRBA) · RBDL (CPU, float64) | 0.091 | 0.122 | 0.230 |
| FD · mantis (C bridge) | 0.026 | 0.067 | 0.249 |
| FD · mantis (JAX) | 0.839 | 0.775 | 1.006 |
| FD · GRiD (C++ bridge) | 0.069 | 0.110 | 0.306 |
| FD · GRiD (JAX FFI) | 0.799 | 0.851 | 1.026 |
| FD · frax | 0.969 | 1.180† | 1.935† |
| FD · Pinocchio codegen (CPU) | 0.124 | 0.150 | 0.289 |
| FD · RBDL (CPU, float64) | 0.121 | 0.146 | 0.300 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.100 | 0.215 | 1.776 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.825 | 0.940 | 3.035 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.069 | 0.121 | 0.895 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.829 | 0.900 | 1.859 |
| ∂ID/∂(q, q̇) · frax | 1.213 | 1.994 | 8.775 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.139 | 0.198 | 0.672 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 256**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.017 | 0.042 | 0.151 |
| ID (RNEA) · mantis (JAX) | 0.775 | 0.766 | 0.930 |
| ID (RNEA) · GRiD (C++ bridge) | 0.058 | 0.095 | 0.242 |
| ID (RNEA) · GRiD (JAX FFI) | 0.818 | 0.837 | 1.009 |
| ID (RNEA) · frax | 0.861 | 1.091 | 1.552 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.077 | 0.127 | 0.194 |
| ID (RNEA) · RBDL (CPU, float64) | 0.119 | 0.133 | 0.247 |
| M (CRBA) · mantis (C bridge) | 0.018 | 0.042 | 0.377 |
| M (CRBA) · mantis (JAX) | 0.569 | 0.586 | 1.070 |
| M (CRBA) · GRiD (C++ bridge) | 0.073 | 0.102 | 0.477 |
| M (CRBA) · GRiD (JAX FFI) | 0.616 | 0.603 | 1.115 |
| M (CRBA) · frax | 0.813 | 1.054 | 1.688 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.113 | 0.150 | 0.348 |
| M (CRBA) · RBDL (CPU, float64) | 0.120 | 0.146 | 0.348 |
| FD · mantis (C bridge) | 0.026 | 0.075 | 0.250 |
| FD · mantis (JAX) | 0.722 | 0.843 | 1.035 |
| FD · GRiD (C++ bridge) | 0.079 | 0.131 | 0.634 |
| FD · GRiD (JAX FFI) | 0.834 | 0.846 | 1.421 |
| FD · frax | 0.982 | 1.272† | 2.013† |
| FD · Pinocchio codegen (CPU) | 0.143 | 0.192 | 0.643 |
| FD · RBDL (CPU, float64) | 0.138 | 0.236 | 0.375 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.103 | 0.296 | 2.133 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.824 | 1.069 | 3.385 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.088 | 0.172 | 1.768 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 0.832 | 0.975 | 2.954 |
| ∂ID/∂(q, q̇) · frax | 1.417 | 2.783 | 15.534 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.178 | 0.260 | 0.928 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 1024**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.027 | 0.045 | 0.164 |
| ID (RNEA) · mantis (JAX) | 0.780 | 0.767 | 1.267 |
| ID (RNEA) · GRiD (C++ bridge) | 0.161 | 0.247 | 0.839 |
| ID (RNEA) · GRiD (JAX FFI) | 0.906 | 1.014 | 1.631 |
| ID (RNEA) · frax | 1.001 | 1.271 | 2.032 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.141 | 0.210 | 0.384 |
| ID (RNEA) · RBDL (CPU, float64) | 0.163 | 0.313 | 0.593 |
| M (CRBA) · mantis (C bridge) | 0.033 | 0.076 | 0.660 |
| M (CRBA) · mantis (JAX) | 0.573 | 0.683 | 1.967 |
| M (CRBA) · GRiD (C++ bridge) | 0.193 | 0.278 | 1.456 |
| M (CRBA) · GRiD (JAX FFI) | 0.723 | 0.882 | 2.644 |
| M (CRBA) · frax | 0.889 | 1.246 | 2.964 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.158 | 0.321 | 0.779 |
| M (CRBA) · RBDL (CPU, float64) | 0.214 | 0.308 | 1.020 |
| FD · mantis (C bridge) | 0.035 | 0.078 | 0.262 |
| FD · mantis (JAX) | 0.732 | 0.867 | 1.304 |
| FD · GRiD (C++ bridge) | 0.242 | 0.362 | 2.012 |
| FD · GRiD (JAX FFI) | 1.058 | 1.161 | 2.957 |
| FD · frax | 1.143 | 1.490† | 3.122† |
| FD · Pinocchio codegen (CPU) | 0.298 | 0.447 | 1.552† |
| FD · RBDL (CPU, float64) | 0.323 | 0.511 | 1.303 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.119 | 0.377 | 2.554 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 0.915 | 1.444 | 4.952 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 0.261 | 0.545 | 6.950 |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 1.039 | 1.502 | 9.088 |
| ∂ID/∂(q, q̇) · frax | 2.165 | 7.348 | 53.808 |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 0.467 | 0.737 | 3.194 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |

**B = 16384**

| op · method | panda | baxter | g1 |
|---|---|---|---|
| ID (RNEA) · mantis (C bridge) | 0.101 | 0.187 | 0.634 |
| ID (RNEA) · mantis (JAX) | 1.121 | 2.023 | 6.842 |
| ID (RNEA) · GRiD (C++ bridge) | 1.850 | 2.807 | OOM |
| ID (RNEA) · GRiD (JAX FFI) | 2.974 | 4.223 | 13.623 |
| ID (RNEA) · frax | 2.145 | 3.842 | 11.419 |
| ID (RNEA) · Pinocchio codegen (CPU) | 0.743 | 1.610 | 4.740 |
| ID (RNEA) · RBDL (CPU, float64) | 1.497 | 3.390 | 8.019 |
| M (CRBA) · mantis (C bridge) | 0.166 | 0.712 | 5.730 |
| M (CRBA) · mantis (JAX) | 1.017 | 2.624 | 81.316 |
| M (CRBA) · GRiD (C++ bridge) | 2.225 | 3.611 | OOM |
| M (CRBA) · GRiD (JAX FFI) | 3.181 | 5.211 | 138.182 |
| M (CRBA) · frax | 2.219 | 5.199 | 126.947 |
| M (CRBA) · Pinocchio codegen (CPU) | 0.984 | 2.651 | 12.605 |
| M (CRBA) · RBDL (CPU, float64) | 1.543 | 3.342 | 15.116 |
| FD · mantis (C bridge) | 0.110 | 0.240 | 0.876 |
| FD · mantis (JAX) | 1.062 | 1.726 | 6.984 |
| FD · GRiD (C++ bridge) | 3.245 | 4.667 | OOM |
| FD · GRiD (JAX FFI) | 4.363 | 6.256 | 31.773 |
| FD · frax | 3.136 | 6.239† | 27.186† |
| FD · Pinocchio codegen (CPU) | 2.093 | 4.969 | 18.745† |
| FD · RBDL (CPU, float64) | 3.301 | 6.610 | 16.254 |
| ∂ID/∂(q, q̇) · mantis (C bridge) | 0.486 | 1.859 | 13.622 |
| ∂ID/∂(q, q̇) · mantis (JAX) | 2.241 | 8.145 | 186.533 |
| ∂ID/∂(q, q̇) · GRiD (C++ bridge) | 3.593 | 7.362 | OOM |
| ∂ID/∂(q, q̇) · GRiD (JAX FFI) | 5.234 | 10.394 | OOM |
| ∂ID/∂(q, q̇) · frax | 21.226 | 100.605 | OOM |
| ∂ID/∂(q, q̇) · Pinocchio codegen (CPU) | 3.945 | 10.207 | 42.835 |
| ∂ID/∂(q, q̇) · RBDL (CPU, float64) | n/a | n/a | n/a |
<!-- /tables -->
