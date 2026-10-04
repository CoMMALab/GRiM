/**
 * Levenberg-Marquardt / augmented Lagrangian over a compiled least-squares problem, at one
 * GLASS tier.
 *
 * `cost_gen.cuh` is written per problem structure (and per tier variant) by a problem
 * compiler (pyroffi.costs is one; grim.motion.costs documents the contract). It provides
 * tier-templated device functions
 *
 *   residual(x, p, r, scratch, rank, size)
 *   residual_jacobian(x, p, r, J, scratch, rank, size)     J: (n_r, n_x) row-major
 *
 * written in GLASS's form: straight-line code every lane of a group runs redundantly, plus
 * loops `for (i = rank; i < n; i += size)` that the group splits, each followed by
 * GRIM_COST_SYNC(). Each residual row has a kind:
 *
 *   0  cost              minimise r^2
 *   1  equality          r == 0      (augmented Lagrangian)
 *   2  inequality        r <= 0      (augmented Lagrangian; >= 0 rows arrive negated)
 *
 * With multipliers `lam` and penalty `rho` the merit is
 *
 *   f(x) = sum_cost r^2 + sum_con (rho / 2) t^2,   t = r + lam / rho   (t = max(t, 0) for <=)
 *
 * and its Gauss-Newton model is H = sum w J J^T, g = sum w (r + s) J, with w = 1, s = 0 on
 * cost rows and w = rho / 2 (0 on an inactive inequality), s = lam / rho on constraint rows.
 *
 * Tiers (GRIM_COST_TIER, one per build): one problem per thread (Thread), per warp
 * (Warp), or per block (Block). Per-problem state lives in registers/local memory at the
 * thread tier and in shared memory at the cooperative tiers. Scalar control flow (LM
 * damping, acceptance, AL updates) is computed redundantly and identically by every lane,
 * so every branch is group-uniform.
 */

#include "glass_solve.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>

// One build per tier: the generated code is plain functions (templates of this size are
// pathologically slow in nvcc's front end), and syncs through this macro.
#if GRIM_COST_TIER == 0
#define GRIM_COST_SYNC() ((void)0)
#elif GRIM_COST_TIER == 1
#define GRIM_COST_SYNC() __syncwarp()
#else
#define GRIM_COST_SYNC() __syncthreads()
#endif

// Precision (one build per choice): `real` is what the solver computes in, `io_t` what the
// FFI buffers hold. Double compute behind float32 buffers is how a float64 solve runs when
// JAX itself is not in x64 mode.
#ifndef GRIM_COST_REAL
#define GRIM_COST_REAL float
#endif
#ifndef GRIM_COST_IO
#define GRIM_COST_IO float
#endif
namespace grim::costs_gen {
using real = GRIM_COST_REAL;
using io_t = GRIM_COST_IO;
}

#include "cost_gen.cuh"

namespace ffi = xla::ffi;
namespace G = grim::costs_gen;
using G::real;
using G::io_t;
using grim::Tier;

constexpr Tier TIER = static_cast<Tier>(GRIM_COST_TIER);
constexpr int NX = G::n_x;
constexpr int NR = G::n_r;
constexpr int NS = G::scratch_size > 0 ? G::scratch_size : 1;

// Per-problem state: x, xt, dx, g (NX each), H (NX*NX), r, rt, lam, w, s (NR each),
// J (NR*NX). The generated code's scratch (NS per problem) lives in a global workspace: it
// can be far larger than shared memory, and one problem's slice stays in L1/L2.
constexpr int SLOT_FLOATS = 4 * NX + NX * NX + 5 * NR + NR * NX;
constexpr int SHARED_BUDGET = 48 * 1024;
constexpr int slots_that_fit(int want)
{
    while (want > 1 && want * (SLOT_FLOATS * (int)sizeof(real) + 16) > SHARED_BUDGET) want /= 2;
    return want;
}
constexpr int WARPS_PER_BLOCK = slots_that_fit(8);
constexpr int BLOCK_TPB = 128;
constexpr int THREAD_TPB = 64;
constexpr int SLOTS = TIER == Tier::Warp ? WARPS_PER_BLOCK : 1;
constexpr int TPB = TIER == Tier::Thread ? THREAD_TPB
                  : TIER == Tier::Warp   ? 32 * WARPS_PER_BLOCK : BLOCK_TPB;

static_assert(TIER == Tier::Block || NX <= grim::TIER_CHOICE_MAX_N,
              "grim cost_lm: the thread and warp tiers support up to 32 variables; "
              "use the block tier");
#ifdef GRIM_COST_JVP
static_assert(TIER == Tier::Thread || NX <= TPB, "one Jacobian column per lane");
#endif
static_assert(TIER == Tier::Thread || SLOT_FLOATS * (int)sizeof(real) + 16 <= SHARED_BUDGET,
              "grim cost_lm: per-problem state exceeds the shared-memory budget at this tier");

// Termination as jaxls: relative cost change, gradient (inf-norm, from grad_start on) and
// parameter step, whichever comes first.
struct CostLmSettings {
    int max_iters;
    int al_iters;
    float lambda_initial;
    float tolerance;       // relative cost change
    float al_tolerance;
    float grad_tolerance;
    int grad_start;
    float param_tolerance;
};

static __device__ __forceinline__ real merit(const real* r, const real* lam, real rho)
{
    real f = real(0.0);
    #pragma unroll
    for (int i = 0; i < NR; ++i) {
        const int k = G::row_kind(i);
        if (k == 0) {
            f += r[i] * r[i];
        } else {
            real t = r[i] + lam[i] / rho;
            if (k == 2) t = fmax(t, real(0.0));
            f += real(0.5) * rho * t * t;
        }
    }
    return f;
}

__global__ void __launch_bounds__(TPB)
cost_lm_kernel(const io_t* __restrict__ x0, const io_t* __restrict__ params, int param_stride,
               io_t* __restrict__ x_out, io_t* __restrict__ cost_out,
               io_t* __restrict__ viol_out, real* __restrict__ workspace, int n_problems,
               CostLmSettings st)
{
    // Compile-time constants at the thread tier, so every per-row loop below unrolls and the
    // solver state stays in registers.
    const int rank = TIER == Tier::Thread ? 0 : TIER == Tier::Warp ? (int)(threadIdx.x & 31u)
                                                                    : (int)threadIdx.x;
    const int size = TIER == Tier::Thread ? 1 : TIER == Tier::Warp ? 32 : (int)blockDim.x;
    const int slot = TIER == Tier::Warp ? (int)(threadIdx.x >> 5) : 0;
    const int b = TIER == Tier::Thread ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                : TIER == Tier::Warp   ? (int)(blockIdx.x * WARPS_PER_BLOCK + slot)
                                       : (int)blockIdx.x;
    if (b >= n_problems) return;  // group-uniform: a whole thread/warp/block retires together

#if GRIM_COST_TIER == 0
    // Separate small arrays, indexed by constants after unrolling: registers, not local memory.
    real x[NX], xt[NX], dx[NX], g[NX], H[NX * NX], r[NR], rt[NR], lam[NR], w[NR], s[NR];
#ifdef GRIM_COST_NORMAL_EQUATIONS
    real* J = nullptr;
#else
    real J[NR * NX];
#endif
    int fail_slot = 0;
    int* s_fail = &fail_slot;
#else
    __shared__ real smem[SLOTS][SLOT_FLOATS];
    __shared__ int sfail[SLOTS];
    int* s_fail = &sfail[slot];
    real* x = smem[slot];
    real* xt = x + NX;
    real* dx = xt + NX;
    real* g = dx + NX;
    real* H = g + NX;
    real* r = H + NX * NX;
    real* rt = r + NR;
    real* lam = rt + NR;
    real* w = lam + NR;
    real* s = w + NR;
    real* J = s + NR;
#endif
    real* scratch = workspace + (size_t)b * NS;
    const io_t* p = params + (size_t)b * param_stride;
    const auto sync = [] { GRIM_COST_SYNC(); };

    #pragma unroll

    for (int j = rank; j < NX; j += size) x[j] = (real)x0[(size_t)b * NX + j];
    #pragma unroll
    for (int i = rank; i < NR; i += size) lam[i] = real(0.0);
    sync();
    G::residual(x, p, r, scratch, rank, size);
    sync();

    // ALGENCAN-style initial penalty, as jaxls: rho = 10 max(1, |f|) / max(1, c^2 / 2).
    real rho = real(1.0);
    if constexpr (G::has_constraints) {
        real f0 = real(0.0), c2 = real(0.0);
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            const int k = G::row_kind(i);
            if (k == 0) f0 += r[i] * r[i];
            else { const real v = (k == 2) ? fmax(r[i], real(0.0)) : r[i]; c2 += v * v; }
        }
        rho = real(10.0) * fmax(real(1.0), fabs(f0)) / fmax(real(1.0), real(0.5) * c2);
    }

    // One LM loop with one iteration budget, as jaxls: the multipliers are updated whenever
    // the inner solve has converged, not by restarting a full solve per AL round (which, on a
    // constraint that cannot be met, would spend al_iters x max_iters iterations).
    real prev_viol = INFINITY;
    int al_updates = 0;
    real mu = st.lambda_initial;
    real f = merit(r, lam, rho);
    {
        for (int it = 0; it < st.max_iters; ++it) {
            #pragma unroll
            for (int i = rank; i < NR; i += size) {
                const int k = G::row_kind(i);
                if (k == 0) { w[i] = real(1.0); s[i] = real(0.0); continue; }
                s[i] = lam[i] / rho;
                w[i] = ((k == 1) || (r[i] + s[i] > real(0.0))) ? real(0.5) * rho : real(0.0);
            }
#ifdef GRIM_COST_NORMAL_EQUATIONS
            // Scalar variant: H and g emitted symbolically (Jacobian sparsity folded away).
            G::normal_equations(x, p, w, s, r, H, g, scratch, rank, size);
            #pragma unroll
            for (int a = rank; a < NX; a += size) {
                H[a * NX + a] += mu;
                dx[a] = -g[a];
            }
#elif defined(GRIM_COST_JVP)
            // Column-parallel forward mode: lane j computes J[:, j] as the directional
            // derivative along e_j, running the same straight-line code as every other lane.
            if (rank < NX) {
                real t[NX], rr[NR], jt[NR];
#pragma unroll
                for (int k = 0; k < NX; ++k) t[k] = (k == rank) ? real(1.0) : real(0.0);
                G::residual_jvp(x, p, t, rr, jt);
                for (int i = 0; i < NR; ++i) J[i * NX + rank] = jt[i];
            }
            sync();
#ifdef GRIM_COST_SPARSE_NE
            // Structurally sparse assembly: only the rows that touch both columns of an entry.
            for (int e = rank; e < NX * NX; e += size) H[e] = real(0.0);
            sync();
            for (int k = rank; k < G::h_nnz; k += size) {
                const int a = G::h_a[k], c = G::h_c[k];
                real acc = real(0.0);
                for (int q = G::h_ptr[k]; q < G::h_ptr[k + 1]; ++q) {
                    const int i = G::h_rows[q];
                    acc += w[i] * J[i * NX + a] * J[i * NX + c];
                }
                H[a * NX + c] = acc + (a == c ? mu : real(0.0));
                H[c * NX + a] = acc + (a == c ? mu : real(0.0));
            }
            for (int a = rank; a < NX; a += size) {
                real acc = real(0.0);
                for (int q = G::g_ptr[a]; q < G::g_ptr[a + 1]; ++q) {
                    const int i = G::g_rows[q];
                    acc += w[i] * (r[i] + s[i]) * J[i * NX + a];
                }
                dx[a] = -acc;
            }
#else
            for (int e = rank; e < NX * NX; e += size) {
                const int a = e / NX, c = e % NX;
                real acc = real(0.0);
                for (int i = 0; i < NR; ++i) acc += w[i] * J[i * NX + a] * J[i * NX + c];
                H[e] = acc + (a == c ? mu : real(0.0));
            }
            for (int a = rank; a < NX; a += size) {
                real acc = real(0.0);
                for (int i = 0; i < NR; ++i) acc += w[i] * (r[i] + s[i]) * J[i * NX + a];
                dx[a] = -acc;
            }
#endif
#else
            G::residual_jacobian(x, p, r, J, scratch, rank, size);  // syncs (w, s too)
            #pragma unroll
            for (int e = rank; e < NX * NX; e += size) {
                const int a = e / NX, c = e % NX;
                real acc = real(0.0);
                #pragma unroll
                for (int i = 0; i < NR; ++i) acc += w[i] * J[i * NX + a] * J[i * NX + c];
                H[e] = acc + (a == c ? mu : real(0.0));
            }
            #pragma unroll
            for (int a = rank; a < NX; a += size) {
                real acc = real(0.0);
                #pragma unroll
                for (int i = 0; i < NR; ++i) acc += w[i] * (r[i] + s[i]) * J[i * NX + a];
                dx[a] = -acc;
            }
#endif
            sync();
            real gmax = real(0.0);  // dx holds -g here
            #pragma unroll
            for (int j = 0; j < NX; ++j) gmax = fmax(gmax, fabs(dx[j]));
            const bool grad_done = it >= st.grad_start && gmax < st.grad_tolerance;
            sync();
            const bool ok = grim::tier_posv<TIER, real, NX>(H, dx, s_fail);
            sync();
            real step2 = real(0.0), x2 = real(0.0);
            #pragma unroll
            for (int j = 0; j < NX; ++j) {
                step2 += dx[j] * dx[j];
                x2 += x[j] * x[j];
            }
            #pragma unroll
            for (int j = rank; j < NX; j += size) xt[j] = x[j] + dx[j];
            sync();
            G::residual(xt, p, rt, scratch, rank, size);
            sync();
            const real ft = merit(rt, lam, rho);
            bool inner_done;
            if (ok && ft < f) {
                #pragma unroll
                for (int j = rank; j < NX; j += size) x[j] = xt[j];
                #pragma unroll
                for (int i = rank; i < NR; i += size) r[i] = rt[i];
                sync();
                const real rel = (f - ft) / fmax(f, real(1e-30));
                f = ft;
                mu = fmax(mu * real(0.25), real(1e-10));
                inner_done = rel < st.tolerance ||
                             sqrt(step2) < st.param_tolerance * (sqrt(x2) + st.param_tolerance);
            } else {
                mu *= real(4.0);
                inner_done = mu > real(1e10);
            }
            inner_done = inner_done || grad_done;
            if (!inner_done) continue;
            if constexpr (!G::has_constraints) break;
            if (al_updates++ >= st.al_iters) break;
            real viol = real(0.0);
            #pragma unroll
            for (int i = 0; i < NR; ++i) {
                const int k = G::row_kind(i);
                if (k == 1) viol = fmax(viol, fabs(r[i]));
                if (k == 2) viol = fmax(viol, r[i]);
            }
            #pragma unroll
            for (int i = rank; i < NR; i += size) {
                const int k = G::row_kind(i);
                if (k == 1) lam[i] = fmin(fmax(lam[i] + rho * r[i], real(-1e7)), real(1e7));
                if (k == 2) lam[i] = fmin(fmax(lam[i] + rho * r[i], real(0.0)), real(1e7));
            }
            sync();
            if (viol < st.al_tolerance) break;
            if (viol > real(0.5) * prev_viol) rho = fmin(rho * real(4.0), real(1e7));
            prev_viol = viol;
            mu = st.lambda_initial;
            f = merit(r, lam, rho);  // the merit changed with lam / rho
        }
    }

    #pragma unroll

    for (int j = rank; j < NX; j += size) x_out[(size_t)b * NX + j] = (io_t)x[j];
    if (rank == 0) {
        real cost = real(0.0), viol = real(0.0);
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            const int k = G::row_kind(i);
            if (k == 0) cost += r[i] * r[i];
            if (k == 1) viol = fmax(viol, fabs(r[i]));
            if (k == 2) viol = fmax(viol, r[i]);
        }
        cost_out[b] = (io_t)cost;
        viol_out[b] = (io_t)viol;
    }
}

#if defined(GRIM_COST_IO_F64)
#define GRIM_COST_IO_DT ffi::DataType::F64
#else
#define GRIM_COST_IO_DT ffi::DataType::F32
#endif

static ffi::Error CostLmImpl(cudaStream_t stream, ffi::Buffer<GRIM_COST_IO_DT> x0,
                             ffi::Buffer<GRIM_COST_IO_DT> params,
                             ffi::ResultBuffer<GRIM_COST_IO_DT> x_out,
                             ffi::ResultBuffer<GRIM_COST_IO_DT> cost_out,
                             ffi::ResultBuffer<GRIM_COST_IO_DT> viol_out,
                             ffi::ResultBuffer<ffi::DataType::U8> workspace,
                             int32_t max_iters, int32_t al_iters, float lambda_initial,
                             float tolerance, float al_tolerance, float grad_tolerance,
                             int32_t grad_start, float param_tolerance)
{
    const int n_problems = static_cast<int>(x0.dimensions()[0]);
    const int param_stride = params.dimensions()[0] == 1 ? 0 : static_cast<int>(params.dimensions()[1]);
    const CostLmSettings st{max_iters, al_iters, lambda_initial, tolerance, al_tolerance,
                            grad_tolerance, grad_start, param_tolerance};
    const int per_block = TIER == Tier::Thread ? THREAD_TPB : (TIER == Tier::Warp ? WARPS_PER_BLOCK : 1);
    const int blocks = (n_problems + per_block - 1) / per_block;
    cost_lm_kernel<<<blocks, TPB, 0, stream>>>(
        x0.typed_data(), params.typed_data(), param_stride, x_out->typed_data(),
        cost_out->typed_data(), viol_out->typed_data(),
        reinterpret_cast<real*>(workspace->typed_data()), n_problems, st);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CostLmFfi, CostLmImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRIM_COST_IO_DT>>()   // x0      (B, n_x)
        .Arg<ffi::Buffer<GRIM_COST_IO_DT>>()   // params  (B | 1, max(n_p, 1))
        .Ret<ffi::Buffer<GRIM_COST_IO_DT>>()   // x       (B, n_x)
        .Ret<ffi::Buffer<GRIM_COST_IO_DT>>()   // cost    (B,)  sum of squared cost rows
        .Ret<ffi::Buffer<GRIM_COST_IO_DT>>()   // viol    (B,)  max constraint violation
        .Ret<ffi::Buffer<ffi::DataType::U8>>()    // workspace (B, scratch * sizeof(real)) bytes
        .Attr<int32_t>("max_iters")
        .Attr<int32_t>("al_iters")
        .Attr<float>("lambda_initial")
        .Attr<float>("tolerance")
        .Attr<float>("al_tolerance")
        .Attr<float>("grad_tolerance")
        .Attr<int32_t>("grad_start")
        .Attr<float>("param_tolerance"));
