# Shared setup for run_kinematics.sh, run_dynamics.sh, run_trajopt.sh (sourced with the script's "$@").
# Flags: --gpu-only (skip the CPU-only methods: Pinocchio and RBDL dynamics, TrajOpt and Drake; run them
# once per CPU, not per GPU), --cpu-only (only those; no GPU used), --time-budget=DUR (kill any single
# method run after DUR, timeout(1) durations; logged as over budget, the rest go on), --fast (quick probe
# into results/fast). GPU=<idx> picks the device; unset, the first idle one (refuses if all are busy).
set -euo pipefail
GPU_ONLY=0 CPU_ONLY=0 FAST=0 BUDGET=""
for arg in "$@"; do
  case "${arg}" in
    --gpu-only) GPU_ONLY=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --fast) FAST=1 ;;
    --time-budget=*) BUDGET="${arg#*=}" ;;
    *) echo "unknown argument '${arg}' (--gpu-only, --cpu-only, --fast, --time-budget=DUR)"; exit 1 ;;
  esac
done
[[ "${GPU_ONLY}${CPU_ONLY}" != 11 ]] || { echo "--gpu-only and --cpu-only exclude each other"; exit 1; }
cpu() { [[ "${GPU_ONLY}" == 0 ]]; }   # cpu: run the CPU-only methods (not with --gpu-only)
gpu() { [[ "${CPU_ONLY}" == 0 ]]; }   # gpu: run the GPU methods (not with --cpu-only)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRIM="$(cd "${HERE}/.." && pwd)"
CONDA="${CONDA_ROOT:-$HOME/miniconda3}/envs"
JAXPY="${CONDA}/grim/bin/python"
TORCHPY="${CONDA}/curobo/bin/python"
EXT="${HERE}/external"
FRAX_DIR="${FRAX_DIR:-${EXT}/frax}"
export PRRTC_DIR="${PRRTC_DIR:-${EXT}/prrtc}"
# budget <command...>: run it under --time-budget; a run over budget is logged and skipped, not fatal.
budget() {
  [[ -n "${BUDGET}" ]] || { "$@"; return; }
  local rc=0; timeout --kill-after=60 "${BUDGET}" "$@" || rc=$?
  if [[ "${rc}" == 124 || "${rc}" == 137 ]]; then
    echo "$(date +%T) OVER BUDGET (${BUDGET}): $*" | tee -a "${HERE}/${BENCH_RESULTS:-results}/over_budget.log" >&2
    return 0
  fi
  return "${rc}"
}
if ! gpu; then
  GPU=""   # --cpu-only: hide the GPUs
elif [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F', ' '$2 < 500 && $3 < 5 {print $1; exit}')
  [[ -n "${GPU}" ]] || { echo "all GPUs busy:"; nvidia-smi; exit 1; }
fi
export GPU
echo "GPU ${GPU:-none}"
# True float32 matmuls: JAX's default TF32 on Ampere costs frax 1e-3..4e-2 relative error.
export CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONPATH="${GRIM}/bindings:${EXT}/pyroki/src:${GRIM}/external/mantis/python:${FRAX_DIR}:${HERE}/fk"
cd "${HERE}"
if [[ "${FAST}" == 1 ]]; then
  export BENCH_FAST=1 BENCH_RESULTS=results/fast
fi
# cuRobo v2 is pure Python (kernels compile at runtime): loaded from the clone, not an install.
TORCHPATH="${EXT}/curobo:${EXT}/curobo/benchmark:${PYTHONPATH}"
