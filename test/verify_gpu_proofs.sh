#!/usr/bin/env bash
# Verify GRiM's GPU-proof tiers, CPU-only. Each submodule's receipt is checked against its own
# signer at the commit GRiM pins; GRiM's receipt covers the motion suite and fingerprints the
# submodule pointers, so a bump needs a fresh GRiM receipt too.
#   bash test/verify_gpu_proofs.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu-proof verify --repo external/GRiD --receipt external/GRiD/gpu-proof.json \
  --policy external/GRiD/test/gpu-proof-policy.yaml --expected-skips external/GRiD/test/gpu-proof-expected-skips.txt
gpu-proof verify --repo external/cricket2 --receipt external/cricket2/gpu-proof.json \
  --policy external/cricket2/tests/gpu-proof-policy.yaml
gpu-proof verify --repo . --receipt gpu-proof.json --policy test/gpu-proof-policy.yaml
