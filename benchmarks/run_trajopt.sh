#!/usr/bin/env bash
# Kinematic trajopt on MotionBenchMaker (trajopt/), trace mode: every method once to 250 iterations, a
# checkpoint every 10 (bench_trajopt.py TRACE), seed 0, defaults then --tuned where bench_trajopt.TUNED
# changes something. GRiM and pyroki in grim, cuRobo in curobo, TrajOpt in tesseract-bench and Drake in
# drake-bench (the last three graded afterwards in grim). Needs external_patches/apply.sh (pyroki's
# jaxls snapshots, cuRobo's recorder). Two chains run at once, the GPU methods and the CPU ones
# (SERIAL=1: one after the other, e.g. on a Jetson, where they share memory and cores); a failed run is
# logged to results/trajopt_failed.log and the rest go on. Per-run output: results/logs/.
# Flags: bench_env.sh (--fast: the first problem per scene). Tables: python tables.py trace.
source "$(dirname "${BASH_SOURCE[0]}")/bench_env.sh" "$@"
LIMIT=(); [[ "${FAST}" == 1 ]] && LIMIT=(--limit 1)
F=(--trace "${LIMIT[@]}")
JAX_PLATFORMS=cpu "${JAXPY}" trajopt/bench_trajopt.py prepare
R="${BENCH_RESULTS:-results}"; mkdir -p "$R/logs"; : > "$R/trajopt_failed.log"
run() {   # run <log name> <command...>
  local log="$R/logs/$1.log"; shift
  budget "$@" > "${log}" 2>&1 || echo "$(date +%T) FAILED (see ${log}): $*" | tee -a "$R/trajopt_failed.log"
}
gpu_chain() {
  for m in grim_sco grim_chomp grim_stomp; do run "${m}_trace" "${JAXPY}" trajopt/bench_trajopt.py "$m" "${F[@]}"; done
  run grim_stomp_tuned_trace "${JAXPY}" trajopt/bench_trajopt.py grim_stomp --tuned "${F[@]}"
  run curobo_trace env PYTHONPATH="${TORCHPATH}" "${TORCHPY}" trajopt/bench_trajopt.py curobo "${F[@]}"
  run score_curobo_trace env JAX_PLATFORMS=cpu "${JAXPY}" trajopt/bench_trajopt.py score curobo "${F[@]}"
  run pyroki_trace env PYTHONPATH="${EXT}/jaxls_snapshots:${PYTHONPATH}" "${JAXPY}" trajopt/bench_trajopt.py pyroki "${F[@]}"   # longest (~4 h)
}
cpu_chain() {
  for t in "" --tuned; do
    run "trajopt${t:+_tuned}_trace" env TESSERACT_RESOURCE_PATH="${EXT}/tesseract" "${CONDA}/tesseract-bench/bin/python" trajopt/bench_trajopt.py trajopt $t "${F[@]}"
    run "drake${t:+_tuned}_trace" "${CONDA}/drake-bench/bin/python" trajopt/bench_trajopt.py drake $t "${F[@]}"
    for m in trajopt drake; do run "score_${m}${t:+_tuned}_trace" env JAX_PLATFORMS=cpu "${JAXPY}" trajopt/bench_trajopt.py score "$m" $t "${F[@]}"; done
  done
}
if [[ -n "${SERIAL:-}" ]]; then
  { gpu && gpu_chain; } || true; { cpu && cpu_chain; } || true
else
  { gpu && gpu_chain; } & { cpu && cpu_chain; } & wait
fi
echo "trajopt done; failures: $(wc -l < "$R/trajopt_failed.log")"
