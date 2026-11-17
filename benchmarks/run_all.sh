#!/usr/bin/env bash
# Every benchmark, every method, on one free GPU; then the tables in README.md.
#   bash benchmarks/run_all.sh            # GPU picked automatically (refuses if all are busy)
#   GPU=2 bash benchmarks/run_all.sh
#   ONLY=trajopt bash benchmarks/run_all.sh      # some of fk, ik, dynamics, trajopt (comma-separated), then the tables
#   ONLY=fk,ik bash benchmarks/run_all.sh
#   bash benchmarks/run_all.sh --gpu-only        # skip the CPU-only methods (frax IK, Pinocchio and RBDL
#                                                 # dynamics, TrajOpt and Drake): run them once per CPU, not per GPU
#   bash benchmarks/run_all.sh --fast            # quick probe into results/fast (README untouched): dynamics 1 repeat
#                                                 # at B = 1, 128, 1024, 16384; trajopt seed 0, first problem per scene, 100 iters
# Envs: grim (JAX: GRiM, mantis, pyroki, frax, Pinocchio), curobo (PyTorch; cuRobo v2 from the clone). pRRTC is built
# from PRRTC_DIR (default external/prrtc) with nvcc. Baselines: bash external/clone.sh.
set -euo pipefail
GPU_ONLY=0 FAST=0
for arg in "$@"; do
  case "${arg}" in
    --gpu-only) GPU_ONLY=1 ;;
    --fast) FAST=1 ;;
    *) echo "unknown argument '${arg}' (--gpu-only, --fast)"; exit 1 ;;
  esac
done
cpu() { [[ "${GPU_ONLY}" == 0 ]]; }   # cpu: run the CPU-only methods (not with --gpu-only)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRIM="$(cd "${HERE}/.." && pwd)"
CONDA="${CONDA_ROOT:-$HOME/miniconda3}/envs"
JAXPY="${CONDA}/grim/bin/python"
TORCHPY="${CONDA}/curobo/bin/python"
EXT="${HERE}/external"
FRAX_DIR="${FRAX_DIR:-${EXT}/frax}"
export PRRTC_DIR="${PRRTC_DIR:-${EXT}/prrtc}"

if [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F', ' '$2 < 500 && $3 < 5 {print $1; exit}')
  [[ -n "${GPU}" ]] || { echo "all GPUs busy:"; nvidia-smi; exit 1; }
fi
echo "GPU ${GPU}"
# True float32 matmuls: JAX's default TF32 on Ampere costs frax 1e-3..4e-2 relative error.
export CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONPATH="${GRIM}/bindings:${EXT}/pyroki/src:${GRIM}/external/mantis/python:${FRAX_DIR}:${HERE}/fk"
cd "${HERE}"
SEEDS="0 1 2 3 4" BUDGETS="25 50 100 200" LIMIT=()
if [[ "${FAST}" == 1 ]]; then
  export BENCH_FAST=1 BENCH_RESULTS=results/fast
  SEEDS=0 BUDGETS=100 LIMIT=(--limit 1)
fi
# cuRobo v2 is pure Python (kernels compile at runtime): loaded from the clone, not an install.
TORCHPATH="${EXT}/curobo:${EXT}/curobo/benchmark:${PYTHONPATH}"

want() { [[ -z "${ONLY:-}" || ",${ONLY}," == *",$1,"* ]]; }   # want <block>: unset ONLY runs every block
OB="${ONLY:-}"; for b in ${OB//,/ }; do
  [[ " fk ik dynamics trajopt " == *" $b "* ]] || { echo "ONLY: unknown block '$b' (fk, ik, dynamics, trajopt)"; exit 1; }
done

if want fk; then
JAX_PLATFORMS=cpu "${JAXPY}" fk/bench_fk.py prepare
for m in prrtc pyroki grim mantis grim_cuda mantis_cuda; do "${JAXPY}" fk/bench_fk.py "$m"; done
PYTHONPATH="${TORCHPATH}" "${TORCHPY}" fk/bench_fk.py curobo
fi

if want ik; then
JAX_PLATFORMS=cpu "${JAXPY}" ik/bench_ik.py prepare
for m in grim_ls grim_dls grim_sqp grim_hjcd grim_mppi grim_ls_cuda grim_dls_cuda grim_sqp_cuda grim_hjcd_cuda grim_mppi_cuda pyroki mantis mantis_ls_f32 mantis_ls_mixed mantis_ls_f64 mantis_sqp_f32 mantis_sqp_mixed mantis_sqp_f64; do "${JAXPY}" ik/bench_ik.py "$m"; done
PYTHONPATH="${TORCHPATH}" "${TORCHPY}" ik/bench_ik.py curobo
JAX_PLATFORMS=cpu "${JAXPY}" ik/bench_ik.py score curobo  # no Pinocchio in the curobo env
# frax GlobalIK: ElastiQP is CPU/float64-only; frax's recommended CPU settings (frax-bench env: pip install -e "external/frax[ik]" pin)
cpu && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED" \
  "${CONDA}/frax-bench/bin/python" ik/bench_ik.py frax
fi

if want dynamics; then
# Dynamics on GRiD's release protocol (dynamics/grid_protocol.py), grid-bench env: GRiD, Pinocchio and
# RBDL from external/ (Pinocchio and RBDL built into dyn-bench; see RUNNING.md). One capture at a time.
JAX_PLATFORMS=cpu "${CONDA}/grid-bench/bin/python" dynamics/bench_dynamics.py prepare
GRIDPY="${CONDA}/grid-bench/bin/python"; DYN="${CONDA}/dyn-bench"
DYN_METHODS="grid_cuda grid_jax mantis mantis_cuda frax"
cpu && DYN_METHODS+=" pinocchio rbdl"
for m in ${DYN_METHODS}; do
  PYTHONPATH="${GRIM}/bindings:${GRIM}/external/mantis/python" PKG_CONFIG_PATH="${DYN}/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
    LD_LIBRARY_PATH="${DYN}/lib" JAX_DEFAULT_MATMUL_PRECISION=highest "${GRIDPY}" dynamics/bench_dynamics.py "$m"
done
fi

if want trajopt; then
# Kinematic trajopt on MotionBenchMaker (trajopt/), 5 seeds, at each iteration budget: GRiM and pyroki
# in grim, TrajOpt in tesseract-bench and Drake in drake-bench (both graded afterwards in grim). Every
# method with its defaults, then --tuned for the methods bench_trajopt.TUNED changes. cuRobo runs its
# MotionPlanner as designed (no iteration budget): once per seed, in curobo. Two chains run at once,
# the GPU methods and the CPU ones (TrajOpt, Drake); a failed run is logged to results/trajopt_failed.log
# and the night goes on. Per-run output: results/logs/.
JAX_PLATFORMS=cpu "${JAXPY}" trajopt/bench_trajopt.py prepare
R="${BENCH_RESULTS:-results}"; mkdir -p "$R/logs"; : > "$R/trajopt_failed.log"
run() {   # run <log name> <command...>
  local log="$R/logs/$1.log"; shift
  "$@" > "${log}" 2>&1 || echo "$(date +%T) FAILED (see ${log}): $*" | tee -a "$R/trajopt_failed.log"
}
gpu_chain() {
  for s in ${SEEDS}; do
    run "curobo_s$s" env PYTHONPATH="${TORCHPATH}" "${TORCHPY}" trajopt/bench_trajopt.py curobo "${LIMIT[@]}" --seed "$s"
    for n in ${BUDGETS}; do
      for m in grim_chomp grim_stomp grim_sco pyroki; do run "${m}_i${n}_s$s" "${JAXPY}" trajopt/bench_trajopt.py "$m" --iters "$n" "${LIMIT[@]}" --seed "$s"; done
      run "grim_stomp_tuned_i${n}_s$s" "${JAXPY}" trajopt/bench_trajopt.py grim_stomp --iters "$n" --tuned "${LIMIT[@]}" --seed "$s"
    done
  done
}
cpu_chain() {
  for s in ${SEEDS}; do
    for n in ${BUDGETS}; do
      for t in "" --tuned; do
        run "trajopt${t:+_tuned}_i${n}_s$s" env TESSERACT_RESOURCE_PATH="${EXT}/tesseract" "${CONDA}/tesseract-bench/bin/python" trajopt/bench_trajopt.py trajopt --iters "$n" $t "${LIMIT[@]}" --seed "$s"
        run "drake${t:+_tuned}_i${n}_s$s" "${CONDA}/drake-bench/bin/python" trajopt/bench_trajopt.py drake --iters "$n" $t "${LIMIT[@]}" --seed "$s"
        for m in trajopt drake; do run "score_${m}${t:+_tuned}_i${n}_s$s" env JAX_PLATFORMS=cpu "${JAXPY}" trajopt/bench_trajopt.py score "$m" --iters "$n" $t "${LIMIT[@]}" --seed "$s"; done
      done
    done
  done
}
gpu_chain & { cpu && cpu_chain; } & wait
echo "trajopt done; failures: $(wc -l < "$R/trajopt_failed.log")"
fi

"${JAXPY}" tables.py
