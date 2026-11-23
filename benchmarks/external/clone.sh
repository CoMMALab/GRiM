#!/usr/bin/env bash
# Clone every baseline the benchmarks compare against into benchmarks/external/<name>, at a pinned
# commit (blank = default branch HEAD; pin it once measured). Clones are gitignored, not submodules,
# so `git submodule update --recursive` in GRiM / pyroffi never pulls them.
#   bash benchmarks/external/clone.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
while read -r name url commit; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  [[ -d "${name}" ]] || git clone "${url}" "${name}"
  [[ -n "${commit:-}" ]] && git -C "${name}" checkout -q "${commit}"
  git -C "${name}" submodule update -q --init --recursive
  echo "${name} $(git -C "${name}" rev-parse --short HEAD)"
done <<'REPOS'
prrtc   https://github.com/commalab/prrtc.git      6a516f7
curobo  https://github.com/NVlabs/curobo.git       3a0ab73
pyroki  https://github.com/chungmin99/pyroki.git   388e43e
frax    git@github.com:StanfordASL/frax.git        f5e87ab
grid    https://github.com/A2R-Lab/GRiD.git        712df1c
pinocchio https://github.com/stack-of-tasks/pinocchio.git v3.9.0
rbdl    https://github.com/rbdl/rbdl.git           v3.4.0
robometrics https://github.com/fishbotics/robometrics.git 81e3d1d
trajopt https://github.com/tesseract-robotics/trajopt.git 0.35.0
tesseract_python https://github.com/tesseract-robotics/tesseract_python.git 0.35.0
tesseract https://github.com/tesseract-robotics/tesseract.git 0.35.0
tesseract_planning https://github.com/tesseract-robotics/tesseract_planning.git 0.35.0
ros_industrial_cmake_boilerplate https://github.com/ros-industrial/ros_industrial_cmake_boilerplate.git 0.7.4
boost_plugin_loader https://github.com/tesseract-robotics/boost_plugin_loader.git 0.4.3
ruckig https://github.com/Levi-Armstrong/ruckig.git cpack-v0.9.2
drake   https://github.com/RobotLocomotion/drake.git v1.51.1
REPOS
