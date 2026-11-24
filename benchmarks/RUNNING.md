# Running the benchmarks

How to set up, run the benchmarks, and find the numbers. For what is measured and why, see
[README.md](README.md).

## Quick start: from a fresh clone

```bash
# 1. GRiM (GLASS pin: temporary, block-tier IK needs its fix; see ../AGENTS.md)
git clone --recursive git@github.com:commalab/GRiM.git && cd GRiM
git -C external/GRiD/external/GLASS checkout 9a24f37
conda create -n grim python=3.12 -y && conda activate grim
bash install/motion_install.sh               # GRiD, mantis, GRiM editable, jax[cuda13], Pinocchio
python -m pytest test/motion -q              # optional: kernels build and match the oracle

# 2. Baselines (pinned clones) and the patches trajopt's trace mode needs
bash benchmarks/external/clone.sh
bash benchmarks/external_patches/apply.sh    # pyroki jaxls snapshots, cuRobo recorder

# 3. The other envs (section 1 below): curobo, grid-bench + dyn-bench, tesseract-src (build the
#    TrajOpt harness), tesseract-bench, drake-bench. Kinematics needs grim + curobo, dynamics the
#    GRiD pair, trajopt all of them.

# 4. Run, on an idle GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
GPU=0 bash benchmarks/run_all.sh --fast      # minutes: every path, into results/fast, README untouched
GPU=0 nohup bash benchmarks/run_all.sh > benchmarks/results/run_all.log 2>&1 &   # hours
```

`nvcc` must be on `PATH`; the kernels compile for the GPU you're on (`MANTIS_ARCH` for another,
section 1). Results land in `benchmarks/results/` and `run_all.sh` ends by rewriting the tables in
README.md; section 2 has the per-block scripts and running on a second machine.

## 1. One-time setup

**Clone the baselines** (pRRTC, cuRobo, pyroki, frax for dynamics) at the commits we pin:

```bash
bash benchmarks/external/clone.sh
```

They go to `benchmarks/external/<name>`, which git ignores.

**Conda environments.** Never use the base env.

| env | used for | what it needs |
|---|---|---|
| `grim` | GRiM, mantis, pyroki, pRRTC, frax dynamics, grading | JAX (CUDA), Pinocchio, GRiM's `install/motion_install.sh` |
| `curobo` | cuRobo FK, IK and trajopt | PyTorch, and `warp-lang==1.19.0` (the pinned cuRobo calls `wp.kernel(module_options=...)`, which 1.12 lacks). cuRobo is loaded from the clone, so it needs no install. Its own motion benchmark (the calibration) also needs `pip install pin tabulate pandas seaborn` (numpy 2.4.6; robometrics' stale `numpy<2` pin is harmless). |
| `tesseract-bench` | TrajOpt (trajopt benchmark) | `pip install tesseract-robotics==0.35.0 numpy pyyaml`; set `TESSERACT_RESOURCE_PATH=benchmarks/external/tesseract` |
| `drake-bench` | Drake (trajopt benchmark) | `pip install drake==1.51.1 numpy pyyaml` |
| `tesseract-src` | building TrajOpt and its harness from `external/` (trajopt benchmark) | see "TrajOpt build" below |
| `grid-bench` | dynamics (GRiD's release protocol: GRiD, mantis, frax, Pinocchio, RBDL) | see below |
| `dyn-bench` | C++ builds of Pinocchio (with CppADCodeGen) and RBDL from `external/` | see below |

**Dynamics setup.** The dynamics benchmark runs GRiD's own release benchmark worker, so it
needs GRiD installed the way GRiD's docs install it, and C++ builds of Pinocchio and RBDL:

```bash
E=~/miniconda3/envs/dyn-bench
conda create -n dyn-bench -c conda-forge python=3.12 cmake ninja cxx-compiler pkg-config eigen \
    boost-cpp urdfdom console_bridge cppad cppadcodegen numpy
# conda's CppADCodeGen ships no pkg-config file; GRiD's Pinocchio bridge finds codegen through one
printf 'prefix=%s\nName: cppadcg\nDescription: CppADCodeGen\nVersion: 2.5.0\nCflags: -I${prefix}/include\nLibs: -L${prefix}/lib -ldl\n' \
    "$E" > "$E/lib/pkgconfig/cppadcg.pc"
PATH=$E/bin:$PATH cmake -S external/pinocchio -B /tmp/pin-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$E -DCMAKE_PREFIX_PATH=$E -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_TESTING=OFF \
    -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_WITH_CODEGEN_SUPPORT=ON -DBUILD_WITH_AUTODIFF_SUPPORT=ON
PATH=$E/bin:$PATH cmake --build /tmp/pin-build -j 12 && cmake --install /tmp/pin-build
PATH=$E/bin:$PATH cmake -S external/rbdl -B /tmp/rbdl-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$E -DRBDL_BUILD_ADDON_URDFREADER=ON -DRBDL_USE_ROS_URDF_LIBRARY=OFF \
    -DRBDL_BUILD_ADDON_BENCHMARK=ON
PATH=$E/bin:$PATH cmake --build /tmp/rbdl-build && cmake --install /tmp/rbdl-build

conda create -n grid-bench python=3.12
grep -v '^pin<4' external/grid/install/requirements-dev.txt > /tmp/grid-req.txt
~/miniconda3/envs/grid-bench/bin/pip install -e "external/grid[all]" "jax[cuda12]" pin==3.9.0 \
    "cmeel-tinyxml2<11" -r /tmp/grid-req.txt
~/miniconda3/envs/grid-bench/bin/pip install --no-deps -e external/frax
```

**TrajOpt build.** The trajopt benchmark times TrajOpt through `trajopt/trajopt_harness.cpp`, a C++
harness over the Tesseract stack built from the pinned clones (`external/clone.sh`), so that the SQP
iteration limit and the margin can be set (the `tesseract-robotics` wheel can't). The wheel in
`tesseract-bench` still writes the robot files and checks the harness: both give identical
trajectories at the same settings.

```bash
conda create -n tesseract-src -c conda-forge python=3.12 cmake ninja pkg-config c-compiler cxx-compiler \
    gcc_linux-64=12 gxx_linux-64=12 colcon-common-extensions eigen boost-cpp console_bridge tinyxml2 \
    yaml-cpp jsoncpp assimp octomap bullet-cpp flann orocos-kdl cereal "pcl>=1.13"
E=~/miniconda3/envs/tesseract-src
mkdir -p external/tesseract_ws/src && cd external/tesseract_ws/src
for r in boost_plugin_loader ros_industrial_cmake_boilerplate tesseract tesseract_planning trajopt; do ln -s ../../$r .; done
cd .. && PATH=$E/bin:$PATH CMAKE_PREFIX_PATH=$E colcon build --merge-install --packages-up-to tesseract_planning \
    --packages-ignore trajopt_ifopt trajopt_sqp qpoases vhacd --cmake-args -DCMAKE_BUILD_TYPE=Release -GNinja \
    "-DCMAKE_CXX_FLAGS=-include stdexcept" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTING=OFF \
    -DTESSERACT_BUILD_TEST_SUITE=OFF -DTESSERACT_BUILD_FCL=OFF -DTESSERACT_BUILD_VHACD=OFF -DTESSERACT_BUILD_IKFAST=OFF \
    -DTESSERACT_BUILD_OPW=OFF -DTESSERACT_BUILD_UR=OFF -DTESSERACT_BUILD_OMPL=OFF -DTESSERACT_BUILD_DESCARTES=OFF \
    -DTESSERACT_BUILD_TRAJOPT_IFOPT=OFF -DTESSERACT_BUILD_TASK_COMPOSER_PLANNING=OFF \
    -DTESSERACT_BUILD_TASK_COMPOSER_TASKFLOW=OFF -DTESSERACT_BUILD_RUCKIG=OFF -DTESSERACT_ENABLE_EXAMPLES=OFF \
    -DBUILD_IPOPT=OFF -DBUILD_SNOPT=OFF -DTRAJOPT_BUILD_IFOPT=OFF
cd ../../trajopt && PATH=$E/bin:$PATH cmake -S harness -B ../external/tesseract_ws/build_harness -GNinja \
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=-include stdexcept" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH="$PWD/../external/tesseract_ws/install;$E"
PATH=$E/bin:$PATH cmake --build ../external/tesseract_ws/build_harness
```

`-include stdexcept` works around missing includes that newer compilers reject; the optional
planners (OMPL, Descartes, IFOPT, the task composer) are not needed for the benchmark.
`bench_trajopt.py trajopt` runs the harness with `install/lib` and the env's `lib` on
`LD_LIBRARY_PATH`. Rebuild only the harness (`cmake --build` above) after editing it.

`pin` from pip is only the Python oracle GRiD's worker validates against; the timed Pinocchio is
the C++ build above, found through `PKG_CONFIG_PATH`. Don't put `dyn-bench/bin` first on `PATH`
when timing: Pinocchio's codegen then cannot run the system C compiler it JIT-compiles with.

`nvcc` must be on `PATH`. The CUDA harnesses compile for the GPU you're on. To target a different
one, set `MANTIS_ARCH`, for example `MANTIS_ARCH=-arch=sm_120` for Blackwell, which needs CUDA
12.8 or later.

## 2. Run everything

```bash
bash benchmarks/run_all.sh          # picks an idle GPU itself; refuses if every GPU is busy
GPU=2 bash benchmarks/run_all.sh    # or choose one
ONLY=trajopt bash benchmarks/run_all.sh   # only some blocks: kinematics, dynamics, trajopt (comma-separated), then the tables
```

`run_all.sh` calls one script per block, each runnable alone with the same flags and `GPU=`
(shared setup in `bench_env.sh`); `python tables.py` rebuilds the tables afterwards:

| script | what it runs |
|---|---|
| `run_kinematics.sh` | FK and global IK, every method, panda / baxter / g1 |
| `run_dynamics.sh` | dynamics on GRiD's release protocol (GPU methods, then Pinocchio and RBDL on the CPU) |
| `run_trajopt.sh` | MBM trajopt, trace mode: each method once to 250 iterations, a checkpoint every 10 (needs `external_patches/apply.sh`), every method with its tuned parameters (`--tuned`; defaults where tuning kept them); `python tables.py trace` for its table and figures. `SERIAL=1` runs the CPU chain after the GPU one instead of alongside it |

On another machine (e.g. a Jetson AGX Thor), keep its results apart and pin the clocks first:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks      # Jetson: fixed clocks; note the power mode with the results
MANTIS_ARCH=-arch=sm_110 GPU=0 SERIAL=1 BENCH_RESULTS=results/thor bash benchmarks/run_all.sh
```

Overnight, detached from the terminal (survives logging out):

```bash
GPU=0 nohup bash benchmarks/run_all.sh > benchmarks/results/run_all.log 2>&1 &
tail -f benchmarks/results/run_all.log      # progress; per-run logs in benchmarks/results/logs/
cat benchmarks/results/trajopt_failed.log   # trajopt runs that failed (the rest carry on)
```

The trajopt block runs two chains at once, GPU methods and CPU methods (TrajOpt, Drake), about
7 h on an RTX A5000 for 800 problems (pyroki ~4 h, STOMP ~2 h of it). Each run saves
`results/trajopt_*_trace250_s0.json` (a graded row per checkpoint) and `_q.npz` (trajectories, regradable with `bench_trajopt.py score`). A failed run can be
redone alone with its command from the log, then `python tables.py` rebuilds every table.

This creates the shared inputs, runs every method on all three robots, and rewrites the tables in
`README.md`. A full run takes hours.

## 3. Run one benchmark

Run from `benchmarks/`. These need the same environment variables `run_all.sh` sets: copy the
`export` lines from the top of that script, or use:

```bash
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/../bindings:$PWD/external/pyroki/src:$PWD/../external/mantis/python:$PWD/external/frax:$PWD/fk"
```

Check `nvidia-smi` first and pick an idle GPU.

Every script takes `<method> [robot ...]`. Robots are `panda`, `baxter` and `g1`; leave them out to
run all three.

```bash
python ik/bench_ik.py prepare                 # once: shared seeds and targets (also fk/, dynamics/)
python ik/bench_ik.py grim_ls panda           # one method, one robot
python fk/bench_fk.py mantis
python dynamics/bench_dynamics.py grid g1
```

| benchmark | script | methods |
|---|---|---|
| FK | `fk/bench_fk.py` | `prrtc pyroki grim mantis curobo grim_cuda mantis_cuda` |
| Global IK | `ik/bench_ik.py` | `grim_ls grim_dls grim_sqp grim_hjcd grim_mppi pyroki curobo`, and `grim_<kernel>_cuda` |
| mantis IK steps (supplementary) | `ik/bench_ik.py` | `mantis mantis_ls_{f32,mixed,f64} mantis_sqp_{f32,mixed,f64}` |
| Kinematic trajopt (MBM, Panda) | `trajopt/bench_trajopt.py` | `grim_chomp grim_stomp grim_sco pyroki curobo trajopt drake`; flags `--iters N`, `--tuned`, `--tune`, `--params JSON` (see "Trajopt tuning") |
| Dynamics | `dynamics/bench_dynamics.py` | `grid_cuda grid_jax mantis mantis_cuda frax pinocchio rbdl` |

**`_cuda` methods** run the same GRiM (or mantis) kernels as the JAX row of the same name, but
from C, through `bridge/grim_bridge.py`: it records the JAX method's FFI calls once, compiles each
GRiM kernel file unchanged into a bridge library, and replays the calls from C with no Python or
JAX in the timed loop (dynamics: `dynamics/mantis_bridge.cu`, timed like GRiD's own kernel
bridge). Every bridge run must reproduce the JAX row's output exactly, or the cell fails. The
first run of each compiles a second copy of every kernel, so it takes about as long again.

**Methods that need a different env:**

- **cuRobo:** run it in the `curobo` env with the clone on the path. That env has no Pinocchio, so
  grade the IK result afterwards in `grim`:
  ```bash
  PYTHONPATH="$PWD/external/curobo:$PWD/external/curobo/benchmark:$PYTHONPATH" \
    ~/miniconda3/envs/curobo/bin/python ik/bench_ik.py curobo
  JAX_PLATFORMS=cpu python ik/bench_ik.py score curobo
  ```
- **Dynamics:** `grid-bench` env, one capture at a time on a quiet machine (GRiD's rule):
  ```bash
  E=~/miniconda3/envs/dyn-bench
  PYTHONPATH="$PWD/../bindings:$PWD/../external/mantis/python" PKG_CONFIG_PATH="$E/lib/pkgconfig" \
  LD_LIBRARY_PATH="$E/lib" ~/miniconda3/envs/grid-bench/bin/python dynamics/bench_dynamics.py pinocchio panda
  ```
  Each cell runs GRiD's release worker three times (`dynamics/grid_protocol.py`), validates
  against a float64 Pinocchio oracle and keeps the median. One process can take one GPU method
  minutes to compile per robot on first use.

### Trajopt tuning

Each library first runs as its own reference setup does, then gets the same small tuning budget,
so no method (GRiM included) is tuned more than the others:

1. **Calibration.** cuRobo's own `benchmark/motion_plan_benchmark.py --kpi`, unmodified,
   reproduces its published success (2583/2590, 99.73%, against 99.69% in its docs; 38 ms mean
   planning on an RTX A5000 in Python, against 4.46 ms for the C++ runtime on an RTX 5090). Drake
   follows the Manipulation course's `trajopt_shelves_demo` (two solves, the plant's limits);
   TrajOpt runs Tesseract's default profiles, or (`"profile": "example"`) its pick-and-place
   example's free-space collision profile.
2. **Tuning set.** `--tune` swaps MBM for 60 MπNets problems (`mbm.TUNE_PER_SCENE`, the first 5
   feasible of each of the 12 categories in robometrics' `mpinets_set.yaml`), disjoint from MBM.
3. **Grid.** 2-4 points per method at 100 iterations, defaults included, through
   `--params '{"key": value}'` (each run's file name carries its parameters):

   | method | grid | chosen | tuning-set success, default -> chosen |
   |---|---|---|---|
   | grim_chomp | `w_collision_max` 50/200/1000 x `step_size` 0.05/0.1 | default | 0.700 |
   | grim_stomp | `w_collision_max` 100/1000 x `step_size` 0.3/0.6 | `w_collision_max` 1000 | 0.483 -> 0.567 |
   | grim_sco | `w_collision_max` 100/1000 x `penalty_scale` 3/10 | default | 0.650 |
   | pyroki | `w_smooth` 1/0.1 x `w_self` 5/20 | default | 0.850 |
   | trajopt | `profile` default/example x `coeff` default/50 | `profile` example | 0.717 -> 0.783 |
   | drake | `CONTROL_POINTS` 10/16 x `COLLISION_SAMPLES` 25/50 | `COLLISION_SAMPLES` 50 | 0.767 -> 0.783 |
   | curobo | `num_seeds` 1/4, own seeds, activation 0.01/0.0025, check interval 25/50 | default | 0.383 (all 0.367-0.383) |

   The winner is the highest success, then the lowest median time. It is frozen in
   `bench_trajopt.TUNED` and applied with `--tuned`. 60 problems resolve about 2 points of
   success, so the small differences (Drake, pyroki) are within noise.
4. **cuRobo** runs as designed: `MotionPlanner.plan_cspace` with its own benchmark's planner
   settings (4 trajopt seeds, RRT fallback, up to 20 attempts, 2.5 mm activation), on the shared
   URDF limits and sphere model. It has no iteration budget, so it is one point on the
   success-vs-time figure. Its trajectory optimizer alone (`solve_cspace` from the shared seed, the
   tuning grid above) solves only 38% of the tuning set; its own validator agrees (40%).

### Trajopt statistics

Every method repeats over seeds 0-4 (`--seed`; GRiM STOMP's and cuRobo's RNG; the others are
deterministic, apart from pyroki's run-to-run variation), so each cell aggregates 800 x 5
(problem, seed) runs. `tables.py` reports success and median time with 95% intervals from a
bootstrap over problems (a problem's seeds resampled together), the success-vs-time figure
(`results/trajopt_success_vs_time.png`), exact McNemar tests on (problem, seed) pairs against the
most successful method at 100 iterations (Holm-corrected), and path quality only on the pairs
every method solved. `python tables.py traj <suffix>` prints the section for a partial run (for
example `_table_pick_panda_n1`) without touching the README.

Every method's time is its solve only: the world, the seed and the problem (GRiM's world, pyroki's
geometry, cuRobo's `update_world`, Drake's program, TrajOpt's SQP problem) are built before the
timer starts. For TrajOpt the harness splits `TrajOptMotionPlanner::solve` line for line at its
`opt->optimize()`, which alone is timed (same trajectories as `planner.solve`).

Each finished cell prints a line, for example:

```
grim_ls  panda   P= 1024x8     2.820 ms (thread)  success 0.967  pos 6.94e-04 mm  ori 1.48e-06 rad
```

## 4. Where the results are

**Quick look in the terminal:** pivot tables (method × batch, fastest marked) straight from the
saved results, with no rerun:

```bash
python report.py              # all benchmarks, all robots
python report.py dyn g1       # filter by benchmark (fk ik dyn) and/or robot
```

**Tables:** the "Results" section of [README.md](README.md). To rebuild it from the saved results
without rerunning anything:

```bash
python tables.py
```

**Raw data:** one JSON file per run in `results/`, named `<benchmark>_<method>_<robots>.json`, for
example `results/ik_grim_ls_panda_baxter_g1.json`. Each file is a list of rows:

| benchmark | fields per row |
|---|---|
| `fk` | `method robot batch ms max_err` |
| `ik` | `method robot batch seeds ms tier success pos_mm ori_rad` |
| `dynamics` | `method robot batch op ms resident_ms host_ms variant err dtype status repeats` |

- `ms` is the time per call, averaged over back-to-back calls after warm-up.
- `success` is the fraction of problems Pinocchio accepts: position within 1 mm, orientation
  within 0.01 rad, and joints inside the URDF limits.
- `results/inputs/` holds the shared seeds, targets and configurations, so every method sees the
  same problems.
- An `ik_*_q.npz` holds raw IK solutions saved for grading in another env (`bench_ik.py score`).

Load them in Python:

```python
import json, glob
rows = [r for f in glob.glob("results/ik_*.json") for r in json.load(open(f))]
```

Rerunning a method on the same robots overwrites its file. When a benchmark has several files for
the same method, `tables.py` uses the first in name order, so delete any partial file (for example
`ik_grim_ls_panda.json`) once a full three-robot run exists.
