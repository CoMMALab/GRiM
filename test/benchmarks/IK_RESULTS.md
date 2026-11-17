# IK kernels: mantis vs the hand-written kernels

The best backend and tier for every robot and IK method, measured before GRiM's IK moved to
mantis-only kernels (`kernels/ik_mantis`). The hand-written kernels they replaced are at tag
`pre-crocket-ik`.

**Setup.** RTX A5000, CUDA 13.3, `test/benchmarks/bench_mantis_ik.py`. B = 16384 problems × 8
seeds. Canonical runs one seed per problem for 200 iterations. Analytic runs 64 q7 samples per
target on fr3 only. Every mantis kernel was built at the thread, warp and block (64 threads)
tiers in float32 (`f32`) and float64 (`f64`). Stock GRiM (`GRiM`) ran at every tier it supported.

**Selection.** Each cell is the fastest configuration among those whose success rate is within
2 points of the cell's best. The batch is 16384, or the largest batch every configuration
reached. Success is the fraction of problems whose best seed reaches the target within
1 mm / 0.01 rad by the float64 oracle, inside the limits and (collision) collision-free.
Speedup is against stock GRiM's own fastest tier. `~×` means stock GRiM was too slow to run at
that batch within 20 s/call, so its time is extrapolated linearly from its largest measured
batch (always ≥ 1024 × 8, where the GPU is saturated).

Wins: mantis f32 thread 73, stock GRiM thread 3, mantis f32 block 2 (78 cells).
The median speedup where mantis wins is ×13.

## Best configuration per cell

| robot | targets | LS | DLS | SQP | HJCD coarse | HJCD LM | MPPI | Canonical | Analytic |
|---|---|---|---|---|---|---|---|---|---|
| fr3 | pose | **f32 thread** 43.2 ms<br>×30.5 · s 0.91 | **f32 thread** 13.6 ms<br>×8.0 · s 0.68 | **f32 thread** 280.9 ms<br>×21.9 · s 0.99 | **f32 thread** 1.53 ms<br>×22.9 · s 0.00 | **f32 thread** 42.4 ms<br>×32.4 · s 0.92 | **f32 thread** 39.9 ms<br>×7.7 · s 0.47 | **f32 thread** 2.94 ms<br>×5.5 · s 1.00 | **f32 block** 31.5 ms<br>×18.6 · s 0.98 |
| fr3 | multi | **f32 thread** 14.9 ms<br>×26.4 · s 1.00 | **GRiM thread** 589.8 ms<br>stock best · s 1.00 | **f32 thread** 22.3 ms<br>×27.0 · s 1.00 | **f32 thread** 1.44 ms<br>×24.7 · s 1.00 | **f32 thread** 15.1 ms<br>×25.7 · s 1.00 | **f32 thread** 21.4 ms<br>×17.0 · s 1.00 | — | — |
| fr3 | collision | **f32 thread** 578.3 ms<br>×7.2 · s 0.09 | — | **f32 thread** 1.77 s<br>×8.0 · s 0.17 | **f32 thread** 3.04 ms<br>×13.0 · s 0.00 | **f32 thread** 317.2 ms<br>×9.9 · s 0.06 | **f32 thread** 251.0 ms<br>×7.0 · s 0.03 | **f32 thread** 88.0 ms<br>×1.7 · s 1.00 | **f32 block** 42.0 ms<br>×15.2 · s 0.98 |
| iiwa14 | pose | **f32 thread** 25.2 ms<br>×47.0 · s 0.99 | **f32 thread** 12.5 ms<br>×5.1 · s 0.99 | **f32 thread** 98.1 ms<br>×40.3 · s 1.00 | **f32 thread** 1.63 ms<br>×13.0 · s 0.00 | **f32 thread** 18.8 ms<br>×55.2 · s 1.00 | **f32 thread** 20.1 ms<br>×8.3 · s 0.98 | **f32 thread** 2.74 ms<br>×3.2 · s 1.00 | — |
| iiwa14 | multi | **f32 thread** 30.5 ms<br>×52.8 · s 0.99 | **f32 thread** 143.0 ms<br>×1.1 · s 0.99 | **f32 thread** 106.6 ms<br>×39.7 · s 1.00 | **f32 thread** 2.12 ms<br>×19.9 · s 0.00 | **f32 thread** 26.6 ms<br>×62.6 · s 0.99 | **f32 thread** 25.1 ms<br>×10.6 · s 0.97 | — | — |
| iiwa14 | collision | **f32 thread** 384.0 ms<br>×9.0 · s 0.69 | — | **f32 thread** 1.06 s<br>×10.2 · s 0.70 | **f32 thread** 3.64 ms<br>×6.6 · s 0.00 | **f32 thread** 239.8 ms<br>×8.7 · s 0.45 | **f32 thread** 184.0 ms<br>×6.2 · s 0.66 | **f32 thread** 87.1 ms<br>×1.6 · s 1.00 | — |
| baxter | pose | **f32 thread** 25.1 ms<br>×52.6 · s 0.96 | **f32 thread** 12.7 ms<br>×15.2 · s 0.96 | **f32 thread** 106.0 ms<br>×41.4 · s 0.99 | **f32 thread** 1.55 ms<br>×41.6 · s 0.00 | **f32 thread** 20.8 ms<br>×65.7 · s 0.87 | **f32 thread** 45.8 ms<br>×15.1 · s 0.93 | **f32 thread** 3.16 ms<br>×9.5 · s 1.00 | — |
| baxter | multi | **f32 thread** 2.37 ms<br>×57.9 · s 1.00 | **GRiM thread** 9.36 ms<br>stock best · s 1.00 | **f32 thread** 3.04 ms<br>×56.6 · s 1.00 | **f32 thread** 1.70 ms<br>×39.7 · s 1.00 | **f32 thread** 12.9 ms<br>×79.0 · s 1.00 | **f32 thread** 23.1 ms<br>×14.2 · s 1.00 | — | — |
| baxter | collision | **f32 thread** 2.53 s<br>×7.5 · s 0.00 | — | **f32 thread** 5.20 s<br>~×7.1 · s 0.00 | **f32 thread** 7.16 ms<br>×11.5 · s 0.00 | **f32 thread** 1.18 s<br>×8.2 · s 0.00 | **f32 thread** 816.7 ms<br>×6.4 · s 0.00 | **f32 thread** 400.4 ms<br>×1.4 · s 1.00 | — |
| g1 | pose | **f32 thread** 76.0 ms<br>×30.8 · s 0.99 | **f32 thread** 42.4 ms<br>×5.8 · s 0.99 | **f32 thread** 489.8 ms<br>×17.9 · s 1.00 | **f32 thread** 2.71 ms<br>×27.7 · s 0.01 | **f32 thread** 69.9 ms<br>×50.6 · s 0.98 | **f32 thread** 84.6 ms<br>×12.8 · s 0.99 | **f32 thread** 9.05 ms<br>×4.4 · s 1.00 | — |
| g1 | multi | **f32 thread** 509.6 ms<br>×24.8 · s 0.70 | **GRiM thread** 1.47 s<br>stock best · s 0.80 | **f32 thread** 1.78 s<br>~×15.2 · s 0.85 | **f32 thread** 16.5 ms<br>×7.2 · s 0.00 | **f32 thread** 476.6 ms<br>×26.6 · s 0.01 | **f32 thread** 147.5 ms<br>×7.9 · s 0.37 | — | — |
| g1 | collision | **f32 thread** 6.83 s (B=4096)<br>~×5.4 · s 0.00 | — | **f32 thread** 9.24 s (B=4096)<br>~×4.5 · s 0.00 | **f32 thread** 41.6 ms<br>×3.2 · s 0.00 | **f32 thread** 5.69 s<br>~×6.0 · s 0.00 | **f32 thread** 3.55 s<br>×4.3 · s 0.00 | **f32 thread** 664.7 ms<br>×1.1 · s 1.00 | — |

## Stock GRiM's fastest tier (the speedup baseline)

| robot | targets | LS | DLS | SQP | HJCD coarse | HJCD LM | MPPI | Canonical | Analytic |
|---|---|---|---|---|---|---|---|---|---|
| fr3 | pose | thread 1.32 s | thread 108.5 ms | thread 6.15 s | thread 35.1 ms | thread 1.37 s | thread 308.3 ms | thread 16.1 ms | block 585.7 ms |
| fr3 | multi | thread 393.2 ms | thread 589.8 ms | thread 603.2 ms | thread 35.7 ms | thread 388.3 ms | thread 363.8 ms | — | — |
| fr3 | collision | thread 4.13 s | — | thread 14.16 s | thread 39.6 ms | thread 3.13 s | thread 1.75 s | thread 150.5 ms | block 636.9 ms |
| iiwa14 | pose | thread 1.18 s | thread 63.2 ms | thread 3.96 s | thread 21.3 ms | thread 1.04 s | thread 166.6 ms | thread 8.70 ms | — |
| iiwa14 | multi | thread 1.61 s | thread 155.0 ms | thread 4.23 s | thread 42.3 ms | thread 1.66 s | thread 264.6 ms | — | — |
| iiwa14 | collision | thread 3.47 s | — | thread 10.79 s | thread 24.2 ms | thread 2.09 s | thread 1.13 s | thread 143.0 ms | — |
| baxter | pose | thread 1.32 s | thread 192.1 ms | thread 4.38 s | thread 64.6 ms | thread 1.36 s | thread 690.8 ms | thread 29.9 ms | — |
| baxter | multi | thread 137.0 ms | thread 9.36 ms | thread 172.2 ms | thread 67.5 ms | thread 1.02 s | thread 329.4 ms | — | — |
| baxter | collision | thread 18.92 s | — | thread ~37.04 s | thread 82.3 ms | thread 9.75 s | thread 5.19 s | thread 541.6 ms | — |
| g1 | pose | thread 2.34 s | thread 247.8 ms | thread 8.78 s | thread 74.9 ms | thread 3.54 s | thread 1.08 s | thread 39.9 ms | — |
| g1 | multi | thread 12.64 s | thread 1.47 s | thread ~27.03 s | thread 118.9 ms | thread 12.70 s | thread 1.17 s | — | — |
| g1 | collision | warp ~36.94 s | — | thread ~41.85 s | thread 131.7 ms | thread ~33.93 s | thread 15.32 s | thread 754.5 ms | — |

## Notes

- **Thread tier wins at large batch.** One seed per thread with the per-seed work in
  registers beats cooperative warps and blocks everywhere except analytic IK, where the block
  tier wins. Warp wins no cell; its closest was g1 collision LS, 2.5× behind thread.
- **Multi-end-effector DLS** was the one method stock GRiM kept: mantis f32 thread is 1.3–1.8×
  slower on fr3, baxter and g1, and slightly faster on iiwa14 (143 vs 155 ms). GRiM now runs
  the mantis kernel there too.
- **HJCD coarse** is only the sampling phase; its success is near 0 by design.
- **Collision success is low for every backend, stock included:** ≤ 0.17 on fr3, 0 on baxter
  and g1. Joints that move collision spheres but no end-effector get column scale 1e-8, which
  leaves the system ill-conditioned. mantis matches stock success; it doesn't fix this.
- **g1 collision HJCD LM, mantis f32 block** was not run. ptxas (CUDA 13.3) miscompiles it: in
  the traced `ik_geometry_block_s2`, one lane's `switch(threadIdx.x)` arm reuses uniform
  register UR4 while another arm stores through it (illegal shared address 0x40000008, also at
  `-Xptxas -O1`). f64 and the other tiers are correct.
- **Stock GRiM block SQP with collision** solved through GLASS `block::potrs`, which is wrong
  through `__restrict__` callers (fix on GLASS branch `fix-block-restrict-stale`). It was never
  stock's fastest tier, so no cell depends on it.

## Reproducing

```bash
python test/benchmarks/bench_mantis_ik.py <robot> --out test/benchmarks/results/tiers_<robot>.json
```

The raw rows behind these tables (including the stock GRiM rows, backend `"grim"`) are
`test/benchmarks/results/tiers_*.json`. The `*_rerun.json` files replace the cells that the
first run under-measured. The script now runs mantis only.
