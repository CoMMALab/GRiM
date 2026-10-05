#!/usr/bin/env bash
# Motion extras: cricket's Python extension, for traced kinematics (grim.motion, traced=True)
# and traced dynamics (grim.motion.dynamics). cricket builds against conda-forge Pinocchio and
# CppADCodeGen, so run this inside a conda env (e.g. `conda activate grim`), not the pip venv.
#   CRICKET_DIR=/path/to/cricket bash install/motion_install.sh
# cricket: github.com/saiccoumar/cricket, branch cuda_backend-v2 (the CUDA language backend).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${CONDA_PREFIX:?activate the target conda env first}"
: "${CRICKET_DIR:?set CRICKET_DIR to a cricket checkout (branch cuda_backend-v2)}"

conda install -c conda-forge --solver=libmamba -y \
  pinocchio cppad eigen cgal-cpp nlohmann_json fmt cxx-compiler ninja pkg-config patch \
  nanobind scikit-build-core
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCRICKET_BUILD_PYTHON=ON"
pip install -e "${CRICKET_DIR}" --no-build-isolation
pip install -e "${REPO_ROOT}" "jax[cuda13]" pytest
echo "Verify: python -m pytest test/motion -q"
