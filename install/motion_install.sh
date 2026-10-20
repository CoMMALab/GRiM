#!/usr/bin/env bash
# GRiM install: GRiD (external/GRiD) and crocket (external/crocket, pure Python), for traced
# kinematics (grim.motion, traced=True) and traced dynamics (grim.motion.dynamics). Pinocchio
# (conda-forge) is only the tests' second oracle, so run this inside a conda env
# (e.g. `conda activate grim`), not the pip venv.
#   bash install/motion_install.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${CONDA_PREFIX:?activate the target conda env first}"
if [[ ! -f "${REPO_ROOT}/external/GRiD/grid_codegen/__init__.py" ]]; then
  git -C "${REPO_ROOT}" submodule update --init --recursive external/GRiD
fi

conda install -c conda-forge --solver=libmamba -y pinocchio
pip install -e "${REPO_ROOT}/external/crocket"
pip install -e "${REPO_ROOT}/external/GRiD" -e "${REPO_ROOT}" "jax[cuda13]" pytest
echo "Verify: python -m pytest test/motion -q"
