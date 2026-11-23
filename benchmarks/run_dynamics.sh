#!/usr/bin/env bash
# Dynamics on GRiD's release protocol (dynamics/grid_protocol.py), grid-bench env: GRiD, Pinocchio and
# RBDL from external/ (Pinocchio and RBDL built into dyn-bench; see RUNNING.md). One capture at a time,
# on a quiet machine. Flags: bench_env.sh. Tables: python tables.py.
source "$(dirname "${BASH_SOURCE[0]}")/bench_env.sh" "$@"
GRIDPY="${CONDA}/grid-bench/bin/python"; DYN="${CONDA}/dyn-bench"
JAX_PLATFORMS=cpu "${GRIDPY}" dynamics/bench_dynamics.py prepare
DYN_METHODS=""
gpu && DYN_METHODS="grid_cuda grid_jax mantis mantis_cuda frax"
cpu && DYN_METHODS+=" pinocchio rbdl"
for m in ${DYN_METHODS}; do
  PYTHONPATH="${GRIM}/bindings:${GRIM}/external/mantis/python" PKG_CONFIG_PATH="${DYN}/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
    LD_LIBRARY_PATH="${DYN}/lib" budget "${GRIDPY}" dynamics/bench_dynamics.py "$m"
done
