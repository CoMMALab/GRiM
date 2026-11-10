# Trajopt and dynamics kernels: crocket vs the hand-written kernels

Prototype only, in the `pyroffi-grim-crocket` clone. The defaults are unchanged: stock SCO, CHOMP
and STOMP stay GRiM's default backend (`backend="grim"`), and GRiD stays the dynamics backend.
crocket is opt-in through `backend="crocket-f32" | "crocket-f64"` and `CrocketDynamics`.

**Setup.** RTX A5000, `test/benchmarks/bench_crocket_trajopt.py` and `bench_crocket_dynamics.py`.
Trajopt: T = 32, B ∈ {64, 1024} straight-line problems, self-collision only (`self`) or with
`obstacle_world` (`world`). Dynamics: B ∈ {128, 16384} random states. Every crocket kernel is
built at the thread, warp and block (64 threads) tiers, in float32 (`f32`) and float64 (`f64`),
with float32 I/O. Stock GRiM trajopt runs at its own launch shape (SCO and STOMP block, CHOMP
warp); GRiD dynamics is its generated block kernels.

**Selection.** Each trajopt cell is the fastest configuration among those whose success rate is
within 2 points of the cell's best, at the largest batch stock reached within 20 s/call
(B=1024 unless marked). Success is the fraction of the first 64 trajectories that are
collision-free by the float64 oracle (margin 0) and within the joint limits (1e-3). Each dynamics
cell is the fastest configuration at B=16384, with its max relative error against Pinocchio
(float64) over 16 states. Speedup is against the stock kernel.

**Parity.**
- SCO: the cost at 0 iterations matches the float64 oracle (f32 ≤ 4e-7 relative, f64 exact to the
  float32 cast). The final cost matches stock to 1e-7; trajectories differ from stock by ≤ 2e-2
  (fr3) and ≤ 5e-4 (iiwa14).
- CHOMP: the cost at 0 iterations matches the oracle. Trajectories drift from stock by up to 5e-2
  (f32) because the float32 finite-difference hinge gradient is noisy. A float64 numpy
  reimplementation of one iteration puts stock and crocket-f32 both ~1e-1 from it, and
  crocket-f64 within 1.4–3e-3, so f64 tracks the true algorithm and the f32 drift is shared noise.
  Success rates match stock in every cell.
- STOMP: matches stock to ≤ 1.2e-5 at every backend and tier.
- Dynamics: f32 ≤ 4e-6 and f64 ≤ 1e-7 relative error vs Pinocchio at every op and tier
  (`test/motion/test_crocket_dynamics.py`: iiwa14, fr3 with mimic, gen3).

The success rates of 0.00 on fr3, baxter and g1 hold for stock too: those straight-line problems
are not solvable at T = 32 with these weights, so those rows compare speed and parity only.

## Trajopt: best configuration per cell

| robot | scenario | SCO | CHOMP | STOMP |
|---|---|---|---|---|
| fr3 | self | **GRiM block** 536.0 ms<br>stock best · crocket 1.5× slower · s 0.00 | **f32 block** 503.4 ms (B=64)<br>×4.8 · s 0.00 | **GRiM block** 461.0 ms (B=64)<br>stock best · crocket 3.1× slower · s 0.00 |
| fr3 | world | **GRiM block** 567.7 ms<br>stock best · crocket 1.6× slower · s 0.00 | **f32 block** 549.5 ms (B=64)<br>×5.3 · s 0.00 | **GRiM block** 654.8 ms (B=64)<br>stock best · crocket 2.5× slower · s 0.00 |
| iiwa14 | self | **f32 warp** 157.0 ms<br>×1.8 · s 1.00 | **f32 block** 92.3 ms (B=64)<br>×16.8 · s 1.00 | **f32 warp** 1.09 s<br>×3.8 · s 1.00 |
| iiwa14 | world | **f32 warp** 160.8 ms<br>×1.8 · s 1.00 | **f32 block** 96.9 ms (B=64)<br>×19.3 · s 0.94 | **f32 warp** 1.16 s<br>×4.6 · s 0.25 |
| baxter | self | **GRiM block** 4.39 s<br>stock best · crocket 3.9× slower · s 0.00 | **f32 block** 10.58 s (B=64)<br>×1.1 · s 0.00 | **GRiM block** 1.61 s (B=64)<br>stock best · crocket 13.3× slower · s 0.00 |
| baxter | world | **GRiM block** 4.60 s<br>stock best · crocket 3.9× slower · s 0.00 | **f32 block** 11.03 s (B=64)<br>×1.2 · s 0.00 | **GRiM block** 1.69 s (B=64)<br>stock best · crocket 13.3× slower · s 0.00 |
| g1 | self | **GRiM block** 1.26 s (B=64)<br>stock best · crocket 2.7× slower · s 0.00 | **GRiM warp** 12.26 s (B=64)<br>stock best · crocket 1.6× slower · s 0.00 | **GRiM block** 1.70 s (B=64)<br>stock best · crocket 21.5× slower · s 0.00 |
| g1 | world | **GRiM block** 1.33 s (B=64)<br>stock best · crocket 2.6× slower · s 0.00 | **GRiM warp** 13.13 s (B=64)<br>stock best · crocket 1.4× slower · s 0.00 | **GRiM block** 1.81 s (B=64)<br>stock best · crocket 20.9× slower · s 0.00 |

Trajopt wins: GRiM block 12, f32 block 6, f32 warp 4, GRiM warp 2 (24 cells); speedups where crocket wins: ×1.1, ×1.2, ×1.8, ×1.8, ×3.8, ×4.6, ×4.8, ×5.3, ×16.8, ×19.3

## Dynamics: best configuration per cell (vs GRiD)

| robot | n | id | crba | fd | fd_crba | minv | id_du | fd_du |
|---|---|---|---|---|---|---|---|---|
| fr3 | 8 | **f32 thread** 53 µs<br>×54.2 · err 2e-07 | **f32 thread** 59 µs<br>×146.1 · err 3e-07 | — | **f32 thread** 55 µs<br>×256.8 · err 1e-07 | **f32 thread** 137 µs<br>×76.4 · err 2e-07 | — | — |
| iiwa14 | 7 | **f32 thread** 71 µs<br>×44.4 · err 2e-07 | **f32 thread** 125 µs<br>×30.2 · err 9e-08 | **f32 thread** 102 µs<br>×59.2 · err 4e-08 | **f32 thread** 100 µs<br>×80.0 · err 1e-07 | **f32 thread** 255 µs<br>×20.0 · err 2e-07 | **f32 thread** 733 µs<br>×8.2 · err 2e-07 | **f32 thread** 1.42 ms<br>×8.1 · err 8e-07 |
| baxter | 15 | **f32 thread** 163 µs<br>×42.0 · err 2e-07 | **f32 thread** 981 µs<br>×12.4 · err 2e-07 | **f32 thread** 504 µs<br>×32.4 · err 9e-08 | **f32 thread** 700 µs<br>×27.3 · err 2e-07 | **f32 thread** 3.32 ms<br>×3.3 · err 2e-07 | **f32 thread** 4.09 ms<br>×4.4 · err 2e-07 | **f32 thread** 7.12 ms<br>×4.6 · err 5e-07 |
| g1 | 29 | **f32 thread** 300 µs<br>×33.1 · err 2e-07 | **f32 thread** 4.74 ms<br>×2.5 · err 1e-07 | **f32 thread** 935 µs<br>×24.0 · err 2e-07 | **f32 thread** 3.58 ms<br>×6.3 · err 2e-06 | **GRiD block** 14.8 ms<br>stock best · crocket 1.4× slower · err 3e-07 | **f32 thread** 12.9 ms<br>×2.8 · err 2e-07 | **f32 thread** 44.0 ms<br>×1.6 · err 2e-06 |

Dynamics wins: f32 thread 24, GRiD block 1

## Stock baselines

| robot | scenario | SCO | CHOMP | STOMP |
|---|---|---|---|---|
| fr3 | self | block 536.0 ms (B=1024) · s 0.00 | warp 2.39 s (B=64) · s 0.00 | block 9.59 s (B=1024) · s 0.00 |
| fr3 | world | block 567.7 ms (B=1024) · s 0.00 | warp 2.89 s (B=64) · s 0.00 | block 10.61 s (B=1024) · s 0.00 |
| iiwa14 | self | block 278.4 ms (B=1024) · s 1.00 | warp 1.55 s (B=64) · s 1.00 | block 4.13 s (B=1024) · s 1.00 |
| iiwa14 | world | block 294.1 ms (B=1024) · s 1.00 | warp 1.87 s (B=64) · s 0.94 | block 5.27 s (B=1024) · s 0.25 |
| baxter | self | block 4.39 s (B=1024) · s 0.00 | warp 11.53 s (B=64) · s 0.00 | block 1.61 s (B=64) · s 0.00 |
| baxter | world | block 4.60 s (B=1024) · s 0.00 | warp 12.75 s (B=64) · s 0.00 | block 1.69 s (B=64) · s 0.00 |
| g1 | self | block 1.26 s (B=64) · s 0.00 | warp 12.26 s (B=64) · s 0.00 | block 1.70 s (B=64) · s 0.00 |
| g1 | world | block 1.33 s (B=64) · s 0.00 | warp 13.13 s (B=64) · s 0.00 | block 1.81 s (B=64) · s 0.00 |

| robot | id | crba | fd | fd_crba | minv | id_du | fd_du |
|---|---|---|---|---|---|---|---|
| fr3 | 2.85 ms · err 2e-07 | 8.62 ms · err 2e-07 | — | 14.0 ms · err 3e-07 | 10.5 ms · err 3e-07 | — | — |
| iiwa14 | 3.13 ms · err 1e-07 | 3.76 ms · err 9e-08 | 6.03 ms · err 8e-08 | 8.01 ms · err 8e-08 | 5.09 ms · err 9e-08 | 6.01 ms · err 2e-07 | 11.5 ms · err 9e-07 |
| baxter | 6.87 ms · err 2e-07 | 12.1 ms · err 2e-07 | 16.3 ms · err 1e-07 | 19.1 ms · err 1e-07 | 11.0 ms · err 7e-08 | 18.0 ms · err 2e-07 | 32.5 ms · err 4e-07 |
| g1 | 9.92 ms · err 2e-07 | 11.8 ms · err 1e-07 | 22.4 ms · err 2e-07 | 22.4 ms · err 2e-07 | 14.8 ms · err 3e-07 | 35.7 ms · err 1e-07 | 71.3 ms · err 1e-06 |

## Findings

- **Dynamics:** crocket f32 thread beats GRiD in 24 of the 25 measured op cells. The speedup is ×1.6–257, largest
  on small robots. The one loss is `minv` on g1 (n = 29), where the per-thread Cholesky and
  column solves outgrow the registers. fr3 has no `fd` / `id_du` / `fd_du`, because ABA is
  undefined on mimic joints; use `fd_crba`. GRiD's `idsva_so` (second order) has no crocket
  counterpart yet.
- **Sphere centres in registers.** The collision-pair and radius tables are compile-time constants
  and the distance loops are fully unrolled (`tk.cuh`), so the traced sphere centres stay in
  registers instead of local memory. On iiwa14 the STOMP evaluation went from 7.2 ms to 0.9 ms per
  iteration (stock: 3.1 ms), with identical results. The tables above are after this change.
- **CHOMP:** crocket wins on fr3, iiwa14 and baxter (×1.1–19). The stock kernel is one 32-thread
  block per trajectory and ignores `tier` (nsys: 1873 ms/call stock against 171 ms crocket on
  iiwa14 with obstacles, B=64). It loses on g1 (1.4–1.6× slower).
- **SCO and STOMP:** crocket wins only on iiwa14 (SCO ×1.8, STOMP ×3.8–4.6). It is slower on fr3
  (SCO 1.5×, STOMP 2.5–3.1×), baxter (3.9×, 13×) and g1 (2.7×, 21×). The gap grows with the number
  of spheres and pairs. The full unroll likely bloats code and spills on large sphere models (g1
  STOMP got slower after it: about 21 s to 36 s per call). This has not been profiled. The next
  step is to stage each lane's centres in shared memory instead of unrolling.
- **Thread tier for trajopt:** never competitive (seconds per call, and 120–280 s on g1),
  because one thread runs the whole solver. Warp and block are the only useful trajopt tiers.
- **f64:** costs 1.5–5× more than f32 at the same tier. It is worth using only for CHOMP, where
  it tracks the true float64 algorithm.

**Caveats.** The trajopt sweep ran on GPU 3 while other users' jobs ran on GPUs 2–3 for part of
it. Stock and crocket rows were timed back to back under the same conditions. Dynamics fr3 and
g1 were rerun on an idle GPU 0; under contention, g1's GRiD kernels had run out of workspace at
B=16384.

## Analytic collision gradients (`crocket-grad-f32`), 2026-11-09

`crocket-grad-*` replaces the prototype's collision value/gradient with crocket-traced kernels
(`traj_collision`, `traj_smooth_min`): sphere/capsule/box/halfspace distances and an RNEA-style
backward pass for dcost/dq, GRiD style, no finite differences. Thread-tier traces are chunked by
`harness.fit` until ptxas reports zero spills. Supersedes the SCO/STOMP findings above.

Median ms per call, B=64 obstacles, 3 seeds, idle GPUs 1–3 (one robot set per GPU, all backends
back to back). Prototype and analytic list the faster of warp/block (block in every case but
iiwa14/fr3 STOMP, where warp ≈ block).

| robot | solver | stock | prototype `crocket-f32` | analytic `crocket-grad-f32` | vs stock | vs prototype |
|---|---|---|---|---|---|---|
| iiwa14 | SCO | 61.0 | 43.7 | **38.4** | 1.6× | 1.14× |
| | CHOMP | 1876 | 97.5 | **85.3** | 22× | 1.14× |
| | STOMP | 301 | **75.2** | 77.0 | 3.9× | 0.98× |
| fr3 | SCO | 97.2 | 69.0 | **57.4** | 1.7× | 1.2× |
| | CHOMP | 2949 | 137.5 | **102.7** | 29× | 1.3× |
| | STOMP | 243 | 115.5 | **92.3** | 2.6× | 1.25× |
| baxter | SCO | 512 | 287 | **84.8** | 6.0× | 3.4× |
| | CHOMP | 12473 | 1424 | **313.5** | 40× | 4.5× |
| | STOMP | 1595 | 885 | **652.9** | 2.4× | 1.35× |
| g1 | SCO | 1333 | 964 | **216** | 6.2× | 4.5× |
| | CHOMP | 14354 | 2936 | **345** | 42× | 8.5× |
| | STOMP | 1785 | 1565 | **740** | 2.4× | 2.1× |

- **Quality:** STOMP and SCO final costs match stock. CHOMP final cost is lower with the
  analytic gradient (iiwa14 0.13 vs 0.20 stock; success 0.98 vs 0.97). Why CHOMP's cost differs
  is not yet explained. Success is 0 on fr3/baxter/g1 for every backend at this setup, so those
  rows compare speed and cost only.
- **Tiers:** the thread tier (one trajectory per thread) takes 0.8–19 s. Warp/block, with one
  waypoint per lane, are the useful tiers. Splitting one waypoint across a warp or block is
  5–10× slower than one waypoint per lane (per-waypoint kernel, B=2048: iiwa14 thread 182 µs vs
  warp 935 µs; baxter 1485 vs 11632 µs), because the trace is mostly serial dependency chains.
- **Register guard:** at the default budget the thread trace spills (iiwa14 856 B, g1 12320 B);
  `fit` reaches zero spills at budget 1000 (iiwa14, fr3), 250 (baxter), 125 (g1). The cooperative
  tiers are not yet chunked: g1 warp still spills 232 B at budget 125 and `fit` raises.
