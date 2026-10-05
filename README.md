# GRiM

A GPU-accelerated library for rigid-body **motion generation**: robot dynamics, kinematics and
collisions with analytical derivatives and Hessians, plus inverse kinematics, trajectory
optimization and compiled least-squares solvers built per robot.

GRiM builds on [GRiD](https://github.com/A2R-Lab/GRiD), which it includes unmodified as the
`external/GRiD` submodule: GRiD generates the per-robot rigid-body dynamics (`grid_codegen`,
`grid.cuh`, the `grid_rbd` numpy/JAX/torch wrapper) and GRiM adds the motion layer
(`grim.motion`, below), held to GRiD's verification standard. For dynamics, kinematics and
their derivatives, use GRiD directly and read its README.

## Motion generation (`grim.motion`)

Every kernel is compiled for one robot (and one collision / problem structure) on first use and
cached on disk; there is no runtime robot description, so loops are sized by the robot and there
is no DOF ceiling.

| module | kernels |
|---|---|
| `grim.motion.ik` | Levenberg-Marquardt, SQP with hard limits and collision constraints, HJCD (coordinate descent + LM), MPPI + L-BFGS, canonical (redundancy-resolved) IK; thread / warp / block tiers |
| `grim.motion.analytic_ik` | closed-form IK for the Panda / FR3 family |
| `grim.motion.region_ik` | Brownian, hit-and-run and SVGD sampling of end-effector regions |
| `grim.motion.trajopt` | SCO, STOMP, CHOMP and least-squares trajectory optimization |
| `grim.motion.collision` | fused FK + self / world-primitive / ESDF distances and their Jacobians |
| `grim.motion.costs` | compiled least squares (dense, banded) and the C3+ contact solver numerics, for problem compilers such as pyroffi |
| `grim.motion.dynamics` | cricket-traced dynamics (RNEA, CRBA, ABA, RNEA derivatives) at thread and warp-split tiers |
| `grim.motion.kinematics` | batched FK, pose residuals and Jacobians |

Verification follows GRiD's two-surface rule: a float64 numpy oracle
(`grim.motion.reference`, validated against Pinocchio) and the CUDA kernels validated against
it at three levels (primitives, solver steps, certificates), with thread-count, tier, batch and
determinism invariances. See [test/motion/TESTING.md](test/motion/TESTING.md).

![The GRiM package ecosystem: a user's URDF goes through URDFParser to the code generator (built on GLASS) and RBDReference, producing CUDA C++ with NumPy, JAX, and PyTorch wrappers; benchmarks and tests, backed by pytest-gpu-proof and external oracles, produce validated outputs and performance benchmarks.](docs/imgs/GRiM.png)

## Layout

| path | what |
|---|---|
| `bindings/grim/motion/` | motion kernels (`kernels/*.cu`, GLASS-form thread / warp / block tiers), their per-robot builder `_build.py`, JAX launchers, and the float64 oracle in `reference/` |
| `test/motion/` | the motion test suite; see [TESTING.md](test/motion/TESTING.md) |
| `external/GRiD` | GRiD (submodule), which brings `GLASS`, `RBDReference`, `URDFParser` and the sample robots in `config/robot_assets/` |
| `external/cricket` | cricket (submodule), the Pinocchio trace compiler behind `traced=True` and `grim.motion.dynamics` |

## Install

```shell
git clone --recursive https://github.com/commalab/GRiM.git   # or: git submodule update --init --recursive
conda activate <env>                                          # cricket builds against conda-forge Pinocchio
bash install/motion_install.sh                                # cricket + grim (editable) + jax[cuda13]
python -m pytest test/motion -q
```

`import grim` puts the `external/GRiD` checkout on `sys.path` when GRiD is not installed
(`GRIM_GRID_PATH` points it elsewhere), so a source checkout runs without installing GRiD.
