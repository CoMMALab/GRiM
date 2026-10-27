/**
 * Analytic IK on crocket-traced closed form: the hand-written analytic_ik.cu (tag pre-crocket-ik)'s subproblem
 * decomposition of the spherical-shoulder / intersecting-5-6 / offset-7 family, traced by
 * crocket with the arm geometry folded in (`analytic_ik_candidates`: all eight branches of one
 * q7 value and their pose errors; `analytic_ik_chain`: the screw products that place the
 * home-frame collision spheres). In `real` (GRIM_CK_REAL) throughout.
 *
 * One target per thread, warp or block (`tier`); lanes stride over the q7 samples, each scoring
 * its eight branches exactly as the hand-written kernel does (finite, error below err_tol, inside
 * the limits, clearance above margin, closest to prev_cfg when given), then the group's argmin
 * picks the winner. The FFI signature is AnalyticIkFfi's plus `tier`.
 */

#include "grim_robot_gen.cuh"

#include "xla/ffi/api/ffi.h"

#include <cfloat>
#include <cmath>

namespace ffi = xla::ffi;

#ifndef GRIM_CK_REAL
#define GRIM_CK_REAL double
#endif
using real = GRIM_CK_REAL;
namespace tr = crocket::robots::robot;

#define N_JOINT 7
#define CK_AN_WARPS 4      // targets per block at the warp tier
#define CK_AN_BLOCK 64     // lanes per target at the block tier
#define CK_AN_SLOTS 128    // shared reduction slots per block: max(CK_AN_WARPS * 32, CK_AN_BLOCK)

__device__ __forceinline__ real ck_d_sphere(const real* c, real rr, const float* o)
{
    const real dx = c[0] - o[0], dy = c[1] - o[1], dz = c[2] - o[2];
    return sqrt(dx*dx + dy*dy + dz*dz) - (rr + real(o[3]));
}

__device__ __forceinline__ real ck_d_capsule(const real* c, real rr, const float* o)
{
    const real vx = real(o[3]) - o[0], vy = real(o[4]) - o[1], vz = real(o[5]) - o[2];
    const real len2 = vx*vx + vy*vy + vz*vz;
    real t = 0;
    if (len2 > real(1e-12)) {
        t = ((c[0] - o[0]) * vx + (c[1] - o[1]) * vy + (c[2] - o[2]) * vz) / len2;
        t = fmax(real(0), fmin(real(1), t));
    }
    const real dx = c[0] - (o[0] + t * vx), dy = c[1] - (o[1] + t * vy), dz = c[2] - (o[2] + t * vz);
    return sqrt(dx*dx + dy*dy + dz*dz) - (rr + real(o[6]));
}

__device__ __forceinline__ real ck_d_box(const real* c, real rr, const float* o)
{
    const real dx = c[0] - o[0], dy = c[1] - o[1], dz = c[2] - o[2];
    real q[3];
    for (int k = 0; k < 3; ++k)
        q[k] = fabs(dx * o[3 + 3*k] + dy * o[4 + 3*k] + dz * o[5 + 3*k]) - real(o[12 + k]);
    const real m0 = fmax(q[0], real(0)), m1 = fmax(q[1], real(0)), m2 = fmax(q[2], real(0));
    return sqrt(m0*m0 + m1*m1 + m2*m2) + fmin(fmax(fmax(q[0], q[1]), q[2]), real(0)) - rr;
}

__device__ __forceinline__ real ck_d_halfspace(const real* c, real rr, const float* o)
{
    return (c[0] - o[3]) * o[0] + (c[1] - o[4]) * o[1] + (c[2] - o[5]) * o[2] - rr;
}

// TIER 0: one target per thread (the thread scans every q7 sample); 1: one per warp; 2: one per
// block of CK_AN_BLOCK threads. Lanes stride over the q7 samples, then the group's argmin.
template <int TIER>
__global__ void ck_analytic_ik_kernel(
    const float* __restrict__ targets, const float* __restrict__ q7_samples,
    const float* __restrict__ prev_cfg, const float* __restrict__ spheres_home,
    const int* __restrict__ sphere_joint, const int* __restrict__ self_pairs,
    const float* __restrict__ world_spheres, const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes, const float* __restrict__ world_halfspaces,
    int n_sph, int n_pairs, int n_ws, int n_wc, int n_wb, int n_wh,
    int n_q7, int batch, int respect_limits, int use_prev, float err_tol, float margin,
    float* __restrict__ q_out, float* __restrict__ err_out, int* __restrict__ found_out,
    float* __restrict__ clear_out)
{
    __shared__ real s_score[CK_AN_SLOTS];
    __shared__ int s_idx[CK_AN_SLOTS];
    __shared__ float s_clear[CK_AN_SLOTS];
    __shared__ real s_best[CK_AN_SLOTS * N_JOINT];
    __shared__ real s_err[CK_AN_SLOTS];

    const int L = TIER == 0 ? 1 : TIER == 1 ? 32 : CK_AN_BLOCK;
    const int rank = TIER == 0 ? 0 : TIER == 1 ? (int)(threadIdx.x & 31u) : (int)threadIdx.x;
    const int base = TIER == 1 ? (int)(threadIdx.x >> 5) * 32 : 0;   // this group's reduction slots
    const int b = TIER == 0 ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                : TIER == 1 ? (int)(blockIdx.x * CK_AN_WARPS + (threadIdx.x >> 5)) : (int)blockIdx.x;
    if (b >= batch) return;   // group-uniform
    auto gsync = [&]() { if (TIER == 1) __syncwarp(); if (TIER == 2) __syncthreads(); };
    real T[16];
    for (int i = 0; i < 16; ++i) T[i] = targets[b * 16 + i];
    auto lim_lo = [](int i) { return real(grim::robot::kArmGeom[81 + i]); };
    auto lim_hi = [](int i) { return real(grim::robot::kArmGeom[88 + i]); };

    real best_score = real(DBL_MAX), best_err = real(DBL_MAX);
    float best_clear = 1e30f;
    real best_q[N_JOINT] = {0, 0, 0, 0, 0, 0, 0};

    for (int is = rank; is < n_q7; is += L) {
        const real q7 = q7_samples[is];
        real Q[8 * N_JOINT], E[8];
        tr::analytic_ik_candidates_thread<real>(T, &q7, Q, E, 1);
        for (int br = 0; br < 8; ++br) {
            const real* q = Q + br * N_JOINT;
            bool finite = true;
            for (int i = 0; i < N_JOINT; ++i) finite = finite && isfinite((double)q[i]);
            const real err = finite ? E[br] : real(DBL_MAX);
            bool in_lim = true;
            if (respect_limits)
                for (int i = 0; i < N_JOINT; ++i)
                    in_lim = in_lim && q[i] >= lim_lo(i) - real(1e-6) && q[i] <= lim_hi(i) + real(1e-6);
            const bool ok = finite && err < real(err_tol) && in_lim;

            float clr = 1e30f;
            bool collision_free = true;
            if (n_sph > 0) {
                if (ok) {
                    real chain[8 * 12];
                    tr::analytic_ik_chain_thread<real>(q, chain, 1);
                    real best = real(1e30);
                    // Spheres placed by the screw product of the last joint moving each.
                    auto place = [&](int k, real* c) {
                        const real* Tk = chain + 12 * sphere_joint[k];
                        const float* p = spheres_home + 4 * k;
                        for (int r3 = 0; r3 < 3; ++r3)
                            c[r3] = Tk[4 * r3] * p[0] + Tk[4 * r3 + 1] * p[1] + Tk[4 * r3 + 2] * p[2] + Tk[4 * r3 + 3];
                    };
                    for (int k = 0; k < n_sph; ++k) {
                        real c[3];
                        place(k, c);
                        const real rr = spheres_home[4 * k + 3];
                        for (int m = 0; m < n_ws; ++m) best = fmin(best, ck_d_sphere(c, rr, world_spheres + 4 * m));
                        for (int m = 0; m < n_wc; ++m) best = fmin(best, ck_d_capsule(c, rr, world_capsules + 7 * m));
                        for (int m = 0; m < n_wb; ++m) best = fmin(best, ck_d_box(c, rr, world_boxes + 15 * m));
                        for (int m = 0; m < n_wh; ++m) best = fmin(best, ck_d_halfspace(c, rr, world_halfspaces + 6 * m));
                    }
                    for (int pp = 0; pp < n_pairs; ++pp) {
                        const int ia = self_pairs[2 * pp], ib = self_pairs[2 * pp + 1];
                        real ca[3], cb[3];
                        place(ia, ca);
                        place(ib, cb);
                        const real dx = ca[0] - cb[0], dy = ca[1] - cb[1], dz = ca[2] - cb[2];
                        best = fmin(best, sqrt(dx*dx + dy*dy + dz*dz) - real(spheres_home[4 * ia + 3])
                                              - real(spheres_home[4 * ib + 3]));
                    }
                    clr = (float)best;
                }
                collision_free = clr > margin;
            }
            real score;
            if (use_prev) {
                real d = 0;
                for (int i = 0; i < N_JOINT; ++i) { const real dd = q[i] - real(prev_cfg[b * N_JOINT + i]); d += dd * dd; }
                score = sqrt(d);
            } else {
                score = err;
            }
            if (!collision_free) score += real(1e3);
            if (!ok) score += real(1e6);
            if (score < best_score) {
                best_score = score;
                best_err = err;
                best_clear = clr;
                for (int i = 0; i < N_JOINT; ++i) best_q[i] = q[i];
            }
        }
    }

    auto write = [&](const real* q, real e, float clr) {
        bool lim = true;
        for (int i = 0; i < N_JOINT; ++i) {
            q_out[b * N_JOINT + i] = (float)q[i];
            if (respect_limits) lim = lim && q[i] >= lim_lo(i) - real(1e-6) && q[i] <= lim_hi(i) + real(1e-6);
        }
        err_out[b] = (float)e;
        found_out[b] = (e < real(err_tol) && lim) ? 1 : 0;
        clear_out[b] = clr;
    };
    if constexpr (TIER == 0) {
        write(best_q, best_err, best_clear);
    } else {
        const int me = base + rank;
        s_score[me] = best_score;
        s_idx[me] = rank;
        s_clear[me] = best_clear;
        s_err[me] = best_err;
        for (int i = 0; i < N_JOINT; ++i) s_best[me * N_JOINT + i] = best_q[i];
        gsync();
        for (int stride = L / 2; stride > 0; stride >>= 1) {   // lowest lane wins a tie
            if (rank < stride && s_score[me + stride] < s_score[me]) {
                s_score[me] = s_score[me + stride];
                s_idx[me] = s_idx[me + stride];
            }
            gsync();
        }
        if (rank == 0) {
            const int win = base + s_idx[base];
            write(s_best + win * N_JOINT, s_err[win], s_clear[win]);
        }
    }
}

static ffi::Error CkAnalyticIkImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> targets, ffi::Buffer<ffi::DataType::F32> q7_samples,
    ffi::Buffer<ffi::DataType::F32> prev_cfg, ffi::Buffer<ffi::DataType::F32> spheres_home,
    ffi::Buffer<ffi::DataType::S32> sphere_joint, ffi::Buffer<ffi::DataType::S32> self_pairs,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    int64_t respect_limits, int64_t use_prev, float err_tol, float margin, int64_t tier,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> q_out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> err_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> found_out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> clear_out)
{
    const auto tdims = targets.dimensions();
    if (tdims.size() != 3 || tdims[1] != 4 || tdims[2] != 4)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "analytic_ik (crocket): targets must be [B, 4, 4]");
    const int batch = static_cast<int>(tdims[0]);
    if (batch == 0) return ffi::Error::Success();
#define CK_AN_ARGS                                                                                  \
        targets.typed_data(), q7_samples.typed_data(), prev_cfg.typed_data(), spheres_home.typed_data(),    \
        sphere_joint.typed_data(), self_pairs.typed_data(), world_spheres.typed_data(),                     \
        world_capsules.typed_data(), world_boxes.typed_data(), world_halfspaces.typed_data(),               \
        static_cast<int>(spheres_home.dimensions()[0]), static_cast<int>(self_pairs.dimensions()[0]),       \
        static_cast<int>(world_spheres.dimensions()[0]), static_cast<int>(world_capsules.dimensions()[0]),  \
        static_cast<int>(world_boxes.dimensions()[0]), static_cast<int>(world_halfspaces.dimensions()[0]),  \
        static_cast<int>(q7_samples.dimensions()[0]), batch, static_cast<int>(respect_limits),              \
        static_cast<int>(use_prev), err_tol, margin, q_out->typed_data(), err_out->typed_data(),            \
        found_out->typed_data(), clear_out->typed_data()
    if (tier == 1)      ck_analytic_ik_kernel<1><<<(batch + CK_AN_WARPS - 1) / CK_AN_WARPS, CK_AN_WARPS * 32, 0, stream>>>(CK_AN_ARGS);
    else if (tier == 2) ck_analytic_ik_kernel<2><<<batch, CK_AN_BLOCK, 0, stream>>>(CK_AN_ARGS);
    else                ck_analytic_ik_kernel<0><<<(batch + 31) / 32, 32, 0, stream>>>(CK_AN_ARGS);
#undef CK_AN_ARGS
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success()
                            : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AnalyticIkFfi, CkAnalyticIkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // targets
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q7_samples
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // prev_cfg
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // spheres_home
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // sphere_joint
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pairs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Attr<int64_t>("respect_limits")
        .Attr<int64_t>("use_prev")
        .Attr<float>("err_tol")
        .Attr<float>("margin")
        .Attr<int64_t>("tier")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());
