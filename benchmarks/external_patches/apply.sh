#!/bin/bash
# Debug (per-iteration capture) patches for the external trajopt baselines; see README.md here.
#   bash benchmarks/external_patches/apply.sh [revert]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="${HERE}/../external"
PY="${CONDA_ROOT:-$HOME/miniconda3}/envs/grim/bin/python"
if [[ "${1:-}" == revert ]]; then
  "${PY}" "${HERE}/jaxls_snapshots.py" revert
  git -C "${EXT}/curobo" apply -R "${HERE}/curobo_debug_best.patch"
  echo "reverted"; exit 0
fi
"${PY}" "${HERE}/jaxls_snapshots.py" apply   # a patched copy in external/jaxls_snapshots (PYTHONPATH it)
git -C "${EXT}/curobo" apply --check "${HERE}/curobo_debug_best.patch" 2>/dev/null \
  && git -C "${EXT}/curobo" apply "${HERE}/curobo_debug_best.patch" \
  || echo "curobo: patch already applied (or the clone moved): $(git -C "${EXT}/curobo" diff --stat | tail -1)"
echo "applied"
