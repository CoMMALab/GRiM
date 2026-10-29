# GRiM

GRiM is a GPU library for robot motion generation. It solves inverse kinematics, optimizes
trajectories, and checks collisions on CUDA, with analytical derivatives throughout.

GRiM is built on [GRiD](https://github.com/A2R-Lab/GRiD), which generates robot-specific
rigid-body dynamics kernels. GRiD is included unmodified as a submodule. If you only need
dynamics, kinematics, or their derivatives, use GRiD directly.

![The GRiM package ecosystem: a user's URDF goes through URDFParser to the code generator (built on GLASS) and RBDReference, producing CUDA C++ with NumPy, JAX, and PyTorch wrappers; benchmarks and tests, backed by pytest-gpu-proof and external oracles, produce validated outputs and performance benchmarks.](docs/imgs/GRiM.png)

## What's in it

Kernels are compiled for your robot the first time you use them, then cached on disk. Because
each build is specialized to one robot, there is no limit on the number of joints.

| module | what it does |
|---|---|
| `grim.motion.ik` | Inverse kinematics: Levenberg-Marquardt, SQP with joint limits and collision constraints, coordinate descent, MPPI + L-BFGS, and redundancy-resolved IK |
| `grim.motion.analytic_ik` | Closed-form IK for the Franka Panda and FR3 |
| `grim.motion.region_ik` | Sampling joint configurations whose end effector lies in a target region |
| `grim.motion.trajopt` | Trajectory optimization: SCO, STOMP, CHOMP, and least squares |
| `grim.motion.collision` | Self- and world-collision distances and their Jacobians |
| `grim.motion.costs` | Contact (C3+) numerics for contact-aware trajectory optimization |
| `grim.motion.dynamics` | Inverse and forward dynamics, mass matrix, and dynamics derivatives, compiled by [crocket](https://github.com/commalab/crocket) |
| `grim.motion.kinematics` | Batched forward kinematics, pose errors, and Jacobians |

Each kernel is tested against a float64 NumPy reference implementation, which is itself tested
against [Pinocchio](https://github.com/stack-of-tasks/pinocchio). See
[test/motion/TESTING.md](test/motion/TESTING.md).

## Install

You need an NVIDIA GPU, the CUDA toolkit (`nvcc`), and a conda environment. Pinocchio from
conda-forge is needed for some tests.

```shell
git clone --recursive https://github.com/commalab/GRiM.git
conda activate <env>
bash install/motion_install.sh   # installs crocket, GRiM, and jax[cuda13]
python -m pytest test/motion -q
```
