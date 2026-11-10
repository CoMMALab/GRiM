/**
 * CHOMP, in `real`, one trajectory per thread, warp or block (`tier`): smoothness + limits
 * gradient, crocket's traced collision cost and analytic gradient, gradient clipping, covariant
 * direction (CG on the smoothness metric, one DOF per lane), a four-candidate line search, early
 * stop. The hand-written kernel this replaced is at tag pre-crocket-trajopt.
 *
 * Lanes take the independent configuration evaluations (each timestep's cost and gradient; each
 * (candidate, timestep) of the line search) into the workspace;
 * after a barrier every lane forms the same totals in the old kernel's serial order, so the
 * control flow (best step, stall count, stop) is group-uniform without broadcasts.
 */

#include "trajopt_crocket/tk.cuh"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>

namespace ffi = xla::ffi;
using namespace grim::tk;

#define CHOMP_MAX_T 128
constexpr int LS_N = 4;

__host__ __device__ inline size_t chomp_stride(int T) { return 3 * (size_t)T * NA + (size_t)(1 + LS_N) * T; }

/** Smoothness + limits at x(s, d), the old kernel's order: acc windows, jerk windows, limits. */
template <class X>
__device__ __forceinline__ real chomp_smooth_limits(int T, X x, const float* lo, const float* hi,
                                                    real w_smooth, real w_acc, real w_jerk, real w_limits)
{
    real c = 0;
    for (int t = 0; t < T - 4; ++t)
        for (int d = 0; d < NA; ++d) {
            real a = 0;
            for (int k = 0; k < 5; ++k) a += st(k) * x(t + k, d);
            c += w_smooth * w_acc * a * a;
        }
    for (int t = 0; t < T - 5; ++t)
        for (int d = 0; d < NA; ++d) {
            real a0 = 0, a1 = 0;
            for (int k = 0; k < 5; ++k) { a0 += st(k) * x(t + k, d); a1 += st(k) * x(t + 1 + k, d); }
            const real j = a1 - a0;
            c += w_smooth * w_jerk * j * j;
        }
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < NA; ++d) {
            const real v = limit_viol(x(t, d), lo[d], hi[d]);
            c += w_limits * v * v;
        }
    return c;
}

__device__ __forceinline__ real metric_at(const real* x, int i, int n, real reg)
{
    real y = 6 * x[i];
    if (i - 1 >= 0) y += -4 * x[i - 1];
    if (i + 1 < n) y += -4 * x[i + 1];
    if (i - 2 >= 0) y += x[i - 2];
    if (i + 2 < n) y += x[i + 2];
    return y + reg * x[i];
}

/** CG on the pentadiagonal smoothness metric (the old kernel's chomp_metric_cg_solve). */
__device__ __forceinline__ void metric_cg(const real* b, real* x, int n, real reg)
{
    real r[CHOMP_MAX_T], p[CHOMP_MAX_T], Ap[CHOMP_MAX_T];
    real rs = 0;
    for (int i = 0; i < n; ++i) { x[i] = 0; r[i] = b[i]; p[i] = r[i]; rs += r[i] * r[i]; }
    if (rs < real(1e-20)) return;
    for (int it = 0; it < (n < 24 ? n : 24); ++it) {
        real pAp = 0;
        for (int i = 0; i < n; ++i) { Ap[i] = metric_at(p, i, n, reg); pAp += p[i] * Ap[i]; }
        if (fabs(pAp) < real(1e-20)) break;
        const real a = rs / pAp;
        real rn = 0;
        for (int i = 0; i < n; ++i) { x[i] += a * p[i]; r[i] -= a * Ap[i]; rn += r[i] * r[i]; }
        if (rn < real(1e-12)) break;
        const real beta = rn / (rs + real(1e-20));
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rs = rn;
    }
}

template <grim::Tier TIER>
__global__ void ck_chomp_trajopt_kernel(
    const float* __restrict__ init, World world, const float* __restrict__ lower,
    const float* __restrict__ upper, const float* __restrict__ start, const float* __restrict__ goal,
    int start_stride, int goal_stride, float* __restrict__ out_trajs, float* __restrict__ out_costs,
    real* __restrict__ workspace, int B, int T, int n_iters, real step_size, real w_smooth, real w_acc,
    real w_jerk, real w_limits, real w_collision, real w_collision_max, real penalty_scale, real margin,
    int covariant, real reg, real clip, real max_delta, int patience, real min_improve)
{
    const Grp<TIER> grp;
    const int b = group_index<TIER>();
    if (b >= B) return;   // group-uniform
    const int n = T * NA;
    real* traj = workspace + (size_t)b * chomp_stride(T);
    real* grad = traj + n;
    real* dir = grad + n;
    real* cb = dir + n;          // [T] each timestep's collision cost
    real* cl = cb + T;           // [LS_N * T] each candidate's, then the final trajectory's
    const float* st0 = start + (size_t)b * start_stride;
    const float* gl0 = goal + (size_t)b * goal_stride;

    for (int i = grp.rank; i < n; i += grp.size) {
        const int t = i / NA, d = i % NA;
        traj[i] = t == 0 ? st0[d] : t == T - 1 ? gl0[d] : init[(size_t)b * n + i];
    }
    grp.sync();
    const real ALPHA[LS_N] = {1, real(0.5), real(0.25), 0};
    auto at = [&](int s, int d) { return traj[s * NA + d]; };
    real w_coll = w_collision;
    int stall = 0;

    for (int it = 0; it < n_iters; ++it) {
        // Smoothness + limits gradient, one element per lane.
        for (int i = grp.rank; i < n; i += grp.size) {
            const int t = i / NA, d = i % NA;
            const real q = traj[i];
            const real v = limit_viol(q, lower[d], upper[d]);
            grad[i] = w_smooth * smooth_grad_at(t, d, T, at, w_acc, w_jerk)
                    + 2 * w_limits * v * (q > upper[d] ? real(1) : q < lower[d] ? real(-1) : real(0));
        }
        grp.sync();
        // Collision: each timestep's cost and its analytic gradient, one timestep per lane.
        for (int t = grp.rank; t < T; t += grp.size) {
            real g[NA];
            cb[t] = hinge_cost_grad(traj + t * NA, world, margin, g);
            for (int d = 0; d < NA; ++d) grad[t * NA + d] += w_coll * g[d];
        }
        grp.sync();
        real curr = chomp_smooth_limits(T, at, lower, upper, w_smooth, w_acc, w_jerk, w_limits);
        for (int t = 0; t < T; ++t) curr += w_coll * cb[t];

        if (grp.leader()) {
            for (int d = 0; d < NA; ++d) { grad[d] = 0; grad[(T - 1) * NA + d] = 0; }
            if (clip > 0) {
                real gg = 0;
                for (int i = NA; i < (T - 1) * NA; ++i) gg += grad[i] * grad[i];
                const real gn = sqrt(gg);
                if (gn > clip) {
                    const real s = clip / (gn + real(1e-12));
                    for (int i = NA; i < (T - 1) * NA; ++i) grad[i] *= s;
                }
            }
        }
        grp.sync();
        const int Tin = T > 2 ? T - 2 : 0;
        for (int d = grp.rank; d < NA; d += grp.size) {
            dir[d] = 0;
            dir[(T - 1) * NA + d] = 0;
            if (covariant && Tin > 0) {
                real g[CHOMP_MAX_T], z[CHOMP_MAX_T];
                for (int i = 0; i < Tin; ++i) g[i] = grad[(i + 1) * NA + d];
                metric_cg(g, z, Tin, reg);
                for (int i = 0; i < Tin; ++i) dir[(i + 1) * NA + d] = -z[i];
            } else {
                for (int t = 1; t < T - 1; ++t) dir[t * NA + d] = -grad[t * NA + d];
            }
        }
        grp.sync();

        // Line search: every (candidate, timestep) collision cost, then the totals.
        for (int i = grp.rank; i < LS_N * T; i += grp.size) {
            const int a = i / T, t = i % T;
            real q[NA];
            for (int d = 0; d < NA; ++d) q[d] = traj[t * NA + d] + step_size * ALPHA[a] * dir[t * NA + d];
            cl[i] = hinge_cost(q, world, margin);
        }
        grp.sync();
        real best = curr, best_alpha = 0;
        for (int a = 0; a < LS_N; ++a) {
            const real alpha = step_size * ALPHA[a];
            auto x = [&](int s, int d) { return traj[s * NA + d] + alpha * dir[s * NA + d]; };
            real c = chomp_smooth_limits(T, x, lower, upper, w_smooth, w_acc, w_jerk, w_limits);
            for (int t = 0; t < T; ++t) c += w_coll * cl[a * T + t];
            if (c < best) { best = c; best_alpha = alpha; }
        }
        grp.sync();   // every lane has read traj / dir
        if (best_alpha > 0)
            for (int i = NA + grp.rank; i < (T - 1) * NA; i += grp.size)
                traj[i] += fmax(-max_delta, fmin(max_delta, best_alpha * dir[i]));
        stall = best < curr - min_improve ? 0 : stall + 1;
        w_coll = fmin(w_coll * penalty_scale, w_collision_max);
        grp.sync();
        if (stall >= patience) break;
    }

    for (int t = grp.rank; t < T; t += grp.size) cl[t] = hinge_cost(traj + t * NA, world, margin);
    grp.sync();
    for (int i = grp.rank; i < n; i += grp.size) out_trajs[(size_t)b * n + i] = (float)traj[i];
    if (grp.leader()) {
        real c = chomp_smooth_limits(T, at, lower, upper, w_smooth, w_acc, w_jerk, w_limits);
        for (int t = 0; t < T; ++t) c += w_collision_max * cl[t];
        out_costs[b] = (float)c;
    }
}

static ffi::Error CkChompTrajoptImpl(
    cudaStream_t stream, GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> init, ffi::Buffer<ffi::DataType::F32> ws,
    ffi::Buffer<ffi::DataType::F32> wc, ffi::Buffer<ffi::DataType::F32> wb,
    ffi::Buffer<ffi::DataType::F32> wh, ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper, ffi::Buffer<ffi::DataType::F32> start,
    ffi::Buffer<ffi::DataType::F32> goal,
    int64_t n_iters, float step_size, float w_smooth, float w_acc, float w_jerk, float w_limits,
    float w_collision, float w_collision_max, float penalty_scale, float margin, int64_t covariant,
    float reg, float clip, float max_delta, int64_t patience, float min_improve,
    int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_trajs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_costs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> workspace)
{
    GRIM_ROT_UPLOAD(stream);
    const int B = (int)init.dimensions()[0], T = (int)init.dimensions()[1];
    if ((int)init.dimensions()[2] != NA || ws.dimensions()[0] != n_ws || wc.dimensions()[0] != n_wc ||
        wb.dimensions()[0] != n_wb || wh.dimensions()[0] != n_wh)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "chomp_trajopt (crocket): launch does not match the robot and obstacle "
                          "counts of this build");
    if (T > CHOMP_MAX_T)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "chomp_trajopt (crocket): T > 128");
    if (workspace->element_count() * sizeof(float) < (size_t)B * chomp_stride(T) * sizeof(real))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "chomp_trajopt (crocket): workspace too small");
    if (B == 0) return ffi::Error::Success();
    const size_t ep = (size_t)B * NA;
    const int ss = start.element_count() >= ep ? NA : 0, gs = goal.element_count() >= ep ? NA : 0;
    const World w{ws.typed_data(), wc.typed_data(), wb.typed_data(), wh.typed_data()};
    auto smem = [](grim::Tier) { return (size_t)0; };
    TK_LAUNCH(ck_chomp_trajopt_kernel, tier, B, (int)block_threads, smem, stream,
              init.typed_data(), w, lower.typed_data(), upper.typed_data(), start.typed_data(),
              goal.typed_data(), ss, gs, out_trajs->typed_data(), out_costs->typed_data(),
              reinterpret_cast<real*>(workspace->typed_data()), B, T, (int)n_iters, step_size,
              w_smooth, w_acc, w_jerk, w_limits, w_collision, w_collision_max, penalty_scale, margin,
              (int)covariant, reg, clip, max_delta, (int)patience, min_improve);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success() : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CkChompTrajoptFfi, CkChompTrajoptImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        GRIM_ROT_BIND
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // init_trajs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // start
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // goal
        .Attr<int64_t>("n_iters")
        .Attr<float>("step_size")
        .Attr<float>("w_smooth")
        .Attr<float>("w_acc")
        .Attr<float>("w_jerk")
        .Attr<float>("w_limits")
        .Attr<float>("w_collision")
        .Attr<float>("w_collision_max")
        .Attr<float>("collision_penalty_scale")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("use_covariant_update")
        .Attr<float>("smoothness_reg")
        .Attr<float>("grad_clip_norm")
        .Attr<float>("max_delta_per_step")
        .Attr<int64_t>("early_stop_patience")
        .Attr<float>("min_cost_improve")
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());
