#!/usr/bin/env bash
# Every benchmark, every method, on one free GPU; then the tables in README.md. Each block is its own
# script, runnable alone with the same flags: run_kinematics.sh (FK, IK), run_dynamics.sh, run_trajopt.sh.
#   bash benchmarks/run_all.sh            # GPU picked automatically (refuses if all are busy)
#   GPU=2 bash benchmarks/run_all.sh
#   ONLY=trajopt bash benchmarks/run_all.sh      # some of kinematics, dynamics, trajopt (comma-separated), then the tables
#   bash benchmarks/run_all.sh --gpu-only        # skip the CPU-only methods (Pinocchio and RBDL
#                                                 # dynamics, TrajOpt and Drake): run them once per CPU, not per GPU
#   bash benchmarks/run_all.sh --cpu-only        # only those CPU-only methods (no GPU used)
#   bash benchmarks/run_all.sh --time-budget=2h  # kill any single method run after 2h (timeout(1) durations)
#   bash benchmarks/run_all.sh --fast            # quick probe into results/fast (README untouched)
#   BENCH_RESULTS=results/thor bash benchmarks/run_all.sh   # another machine's results, kept apart
# Envs and setup: RUNNING.md. Baselines: bash external/clone.sh.
source "$(dirname "${BASH_SOURCE[0]}")/bench_env.sh" "$@"   # checks the flags, picks (and exports) the GPU
want() { [[ -z "${ONLY:-}" || ",${ONLY}," == *",$1,"* ]]; }   # want <block>: unset ONLY runs every block
OB="${ONLY:-}"; for b in ${OB//,/ }; do
  [[ " kinematics dynamics trajopt " == *" $b "* ]] || { echo "ONLY: unknown block '$b' (kinematics, dynamics, trajopt)"; exit 1; }
done
for b in kinematics dynamics trajopt; do
  if want "$b"; then bash "${HERE}/run_${b}.sh" "$@"; fi
done
"${JAXPY}" tables.py
