"""Single source of truth for the GENERATION-TIME environment knobs (2026-09-24).

Every ``os.environ`` read in grim_codegen that changes the emitted header bytes is
listed here, and every header cache — the bindings store (bindings/grim/_cache.py),
the benchmark header cache (test/benchmarks/baselines/grid/run.py) and the CUDA
equivalence harness (test/cuda_equivalents/cuda_harness.py) — folds
``generation_env()`` into its key. The bug class this closes: each cache hand-listed
a different subset, so a knob flipped for an A/B (GRIM_FDSVA_SO_MINV_TILE) was served
the OTHER variant's cached header and the A/B compared a header to itself.
test/test_generation_env_knobs.py asserts the list matches the reads in the tree and
that every cache references it.
"""
from __future__ import annotations

import os

GENERATION_ENV_KNOBS = (
    "GRIM_CUDA_TARGET_SHARED_MEM_BYTES",
    "GRIM_CUDA_TARGET_LITE_SHARED_MEM_BYTES",
    "GRIM_CUDA_SHARED_MEM_TYPE_SIZE_BYTES",
    "GRIM_NO_LICM_BARRIER",
    "GRIM_FDSVA_SO_MINV_TILE",
    "GRIM_GLASS_REVISION",
    "GRIM_ENABLE_MUJOCO_KERNELS",
)


# Bindings-only knobs: read by bindings/grim (never by grim_codegen), folded into the
# per-robot .so key next to the generation knobs. GRIM_CXX_STD forces the wrapper C++
# standard (default: what the installed torch's ATen requires — c++20 from torch 2.14 —
# else c++17).
BINDINGS_ENV_KNOBS = (
    "GRIM_CXX_STD",
)


def generation_env() -> dict[str, str | None]:
    """The current value of every generation-time knob (None = unset), for cache keys."""
    return {k: os.environ.get(k) for k in GENERATION_ENV_KNOBS}
