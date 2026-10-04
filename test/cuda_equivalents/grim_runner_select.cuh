#ifndef GRIM_RUNNER_SELECT_CUH
#define GRIM_RUNNER_SELECT_CUH
//
// Per-algorithm COMPILE selection for the correctness runners (split, not monolith).
//
// The harness compiles ONE algorithm per translation unit: it passes
//   -DGRIM_RUN_SPLIT  -DRUN_<ALGO>=1
// for the single selected algo. In that mode GRIM_RUN_DEFAULT is 0, so every
// other RUN_<ALGO> defaults to 0 and only the selected algo's launch/host/dump
// blocks compile and run. A build break (or missing codegen dependency) in algo Y
// can therefore never void validation of algo X: X's TU never references Y.
// This retires the "one TU compiles every algorithm" coverage void (Bug A,
// 2026-06-17) where a crba_inner build break masked a forward_dynamics VALUE bug.
//
// With NO -DGRIM_RUN_SPLIT (default; local all-in-one runs) GRIM_RUN_DEFAULT is 1,
// so every RUN_<ALGO> defaults to 1 and the runner builds its full algo set exactly
// as before — back-compat, byte-identical behavior.
//
// Each runner, right after `#include "grim.cuh"`, includes this header and then
// declares a default for each algorithm token it owns:
//     #include "grim_runner_select.cuh"
//     #ifndef RUN_INVERSE_DYNAMICS
//     #  define RUN_INVERSE_DYNAMICS GRIM_RUN_DEFAULT
//     #endif
// and wraps that algo's blocks in `#if RUN_INVERSE_DYNAMICS ... #endif`.
//
#ifdef GRIM_RUN_SPLIT
#  define GRIM_RUN_DEFAULT 0
#else
#  define GRIM_RUN_DEFAULT 1
#endif

#endif // GRIM_RUNNER_SELECT_CUH
