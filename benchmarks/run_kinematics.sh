#!/usr/bin/env bash
# FK and global IK, every method, on panda, baxter and g1 (flags: bench_env.sh). Tables: python tables.py.
source "$(dirname "${BASH_SOURCE[0]}")/bench_env.sh" "$@"
JAX_PLATFORMS=cpu "${JAXPY}" fk/bench_fk.py prepare
JAX_PLATFORMS=cpu "${JAXPY}" ik/bench_ik.py prepare
gpu || exit 0   # every kinematics method uses the GPU
for m in prrtc pyroki grim mantis grim_cuda mantis_cuda; do budget "${JAXPY}" fk/bench_fk.py "$m"; done
PYTHONPATH="${TORCHPATH}" budget "${TORCHPY}" fk/bench_fk.py curobo
for m in grim_ls grim_dls grim_sqp grim_hjcd grim_mppi grim_ls_cuda grim_dls_cuda grim_sqp_cuda grim_hjcd_cuda grim_mppi_cuda pyroki mantis mantis_ls_f32 mantis_ls_mixed mantis_ls_f64 mantis_sqp_f32 mantis_sqp_mixed mantis_sqp_f64; do
  budget "${JAXPY}" ik/bench_ik.py "$m"
done
PYTHONPATH="${TORCHPATH}" budget "${TORCHPY}" ik/bench_ik.py curobo
JAX_PLATFORMS=cpu "${JAXPY}" ik/bench_ik.py score curobo  # no Pinocchio in the curobo env
