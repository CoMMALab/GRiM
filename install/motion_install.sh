#!/usr/bin/env bash
# GRiM install: GRiD (external/GRiD), cricket's Python extension, for traced kinematics (grim.motion, traced=True)
# and traced dynamics (grim.motion.dynamics). cricket builds against conda-forge Pinocchio and
# CppADCodeGen, so run this inside a conda env (e.g. `conda activate grim`), not the pip venv.
#   bash install/motion_install.sh
# cricket is the external/cricket submodule (github.com/saiccoumar/cricket, branch
# cuda_backend-v3: the CUDA language backend). CRICKET_DIR overrides it with another checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${CONDA_PREFIX:?activate the target conda env first}"
CRICKET_DIR="${CRICKET_DIR:-${REPO_ROOT}/external/cricket}"
if [[ ! -f "${REPO_ROOT}/external/GRiD/grid_codegen/__init__.py" ]]; then
  git -C "${REPO_ROOT}" submodule update --init --recursive external/GRiD
fi
if [[ ! -f "${CRICKET_DIR}/pyproject.toml" ]]; then
  git -C "${REPO_ROOT}" submodule update --init --recursive external/cricket
fi

conda install -c conda-forge --solver=libmamba -y \
  pinocchio cppad eigen cgal-cpp nlohmann_json fmt cxx-compiler ninja pkg-config patch \
  nanobind scikit-build-core
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCRICKET_BUILD_PYTHON=ON"
pip install -e "${CRICKET_DIR}" --no-build-isolation
pip install -e "${REPO_ROOT}/external/GRiD" -e "${REPO_ROOT}" "jax[cuda13]" pytest
echo "Verify: python -m pytest test/motion -q"
