/**
 * SCO trajectory optimization, in `real`, one trajectory per thread, warp or block (`tier`).
 * Each outer iteration linearizes the smooth-min collision distances about the trajectory
 * (crocket's traced distances and their analytic Jacobian), then L-BFGS minimizes smoothness +
 * limits + the linearized hinge + a trust region, with a five-step line search. Lanes own
 * timesteps (t = rank, rank + size, ...); the scalars every lane needs (L-BFGS's rho and alpha)
 * come out of group reductions, so each lane keeps its own copy. Every buffer of the workspace
 * is `real`. The hand-written kernel this replaced is at tag pre-crocket-trajopt.
 */

#include "trajopt_crocket/tk.cuh"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>

namespace ffi = xla::ffi;
using namespace grim::tk;

#define SCO_MAX_M 8

/** Reals of one trajectory's workspace. */
__host__ __device__ inline size_t sco_stride(int T, int m)
{
    const size_t n = (size_t)T * NA;
    return 6 * n + (size_t)T * G + (size_t)T * G * NA + 2 * (size_t)m * n;
}

struct ScoW {
    real *traj, *qk, *dk, *grad, *dir, *best, *gprev, *J, *s, *y;
};

/** Inner cost of timestep t at x = traj + alpha * dir; with `grad`, also its gradient row. */
__device__ __forceinline__ real sco_inner(const ScoW& w, int t, int T, real alpha, bool with_grad,
                                          const float* lo, const float* hi, real w_smooth, real w_acc,
                                          real w_jerk, real w_limits, real w_trust, real w_coll, real margin)
{
    // alpha == 0 never reads dir: it is unset before the first direction.
    auto x = [&](int s, int d) { return alpha == 0 ? w.traj[s * NA + d] : w.traj[s * NA + d] + alpha * w.dir[s * NA + d]; };
    real cost = smooth_cost_at(t, T, x, w_smooth, w_acc, w_jerk);
    real g[NA];
    if (with_grad)
        for (int d = 0; d < NA; ++d) g[d] = w_smooth * smooth_grad_at(t, d, T, x, w_acc, w_jerk);
    for (int d = 0; d < NA; ++d) {
        const real q = x(t, d);
        const real v = limit_viol(q, lo[d], hi[d]);
        cost += w_limits * v * v;
        if (with_grad) g[d] += 2 * w_limits * v * (q > hi[d] ? real(1) : q < lo[d] ? real(-1) : real(0));
    }
    for (int k = 0; k < G; ++k) {
        const real* Jk = w.J + (t * G + k) * NA;
        real dl = w.dk[t * G + k];
        for (int d = 0; d < NA; ++d) dl += Jk[d] * (x(t, d) - w.qk[t * NA + d]);
        const real v = fmax(real(0), margin - dl);
        cost += w_coll * v * v;
        if (with_grad)
            for (int d = 0; d < NA; ++d) g[d] += -2 * w_coll * v * Jk[d];
    }
    for (int d = 0; d < NA; ++d) {
        const real dl = x(t, d) - w.qk[t * NA + d];
        cost += w_trust * dl * dl;
        if (with_grad) g[d] += 2 * w_trust * dl;
    }
    if (with_grad)
        for (int d = 0; d < NA; ++d) w.grad[t * NA + d] = (t == 0 || t == T - 1) ? real(0) : g[d];
    return cost;
}

template <grim::Tier TIER>
__global__ void ck_sco_trajopt_kernel(
    const float* __restrict__ init, World world, const float* __restrict__ lower,
    const float* __restrict__ upper, const float* __restrict__ start, const float* __restrict__ goal,
    int start_stride, int goal_stride, float* __restrict__ out_trajs, float* __restrict__ out_costs,
    real* __restrict__ workspace, int B, int T, int n_outer, int n_inner, int m_lbfgs,
    real w_smooth, real w_acc, real w_jerk, real w_limits, real w_trust, real w_collision,
    real w_collision_max, real penalty_scale, real margin, real tau)
{
    const Grp<TIER> grp;
    const int b = group_index<TIER>();
    extern __shared__ __align__(16) unsigned char tk_smem_[];
    real* red = reinterpret_cast<real*>(tk_smem_);
    if (b >= B) return;   // group-uniform
    const int n = T * NA;
    real* base = workspace + (size_t)b * sco_stride(T, m_lbfgs);
    ScoW w;
    w.traj = base; w.qk = w.traj + n; w.grad = w.qk + n; w.dir = w.grad + n; w.best = w.dir + n;
    w.gprev = w.best + n; w.dk = w.gprev + n; w.J = w.dk + T * G; w.s = w.J + T * G * NA;
    w.y = w.s + m_lbfgs * n;
    const float* st0 = start + (size_t)b * start_stride;
    const float* gl0 = goal + (size_t)b * goal_stride;

    auto pin = [&]() {
        for (int t = grp.rank; t < T; t += grp.size)
            if (t == 0 || t == T - 1)
                for (int d = 0; d < NA; ++d) w.traj[t * NA + d] = t == 0 ? st0[d] : gl0[d];
    };
    for (int i = grp.rank; i < n; i += grp.size) w.traj[i] = init[(size_t)b * n + i];
    grp.sync();
    pin();
    grp.sync();

    auto sum = [&](real v) { return reduce<TIER>(grp, red, v); };
    auto inner_cost = [&](real alpha, bool with_grad, real w_coll) {
        real c = 0;
        for (int t = grp.rank; t < T; t += grp.size)
            c += sco_inner(w, t, T, alpha, with_grad, lower, upper, w_smooth, w_acc, w_jerk, w_limits,
                           w_trust, w_coll, margin);
        return sum(c);
    };

    real w_coll = w_collision;
    for (int outer = 0; outer < n_outer; ++outer) {
        // Linearize: smooth-min distances and their analytic Jacobians, one timestep per lane.
        for (int t = grp.rank; t < T; t += grp.size) {
            real* q = w.qk + t * NA;
            for (int d = 0; d < NA; ++d) q[d] = w.traj[t * NA + d];
            real Jt[G * NA];
            smooth_min_jac(q, world, tau, w.dk + t * G, Jt);
            for (int k = 0; k < G; ++k)
                for (int d = 0; d < NA; ++d) w.J[(t * G + k) * NA + d] = Jt[k * NA + d];
        }
        grp.sync();

        // L-BFGS. rho / alpha are reduction results, identical on every lane: kept per lane.
        real rho[SCO_MAX_M], ah[SCO_MAX_M];
        int m_used = 0, newest = 0;
        real best = inner_cost(0, true, w_coll);
        for (int t = grp.rank; t < T; t += grp.size)
            for (int d = 0; d < NA; ++d) {
                const int i = t * NA + d;
                w.best[i] = w.traj[i];
                w.gprev[i] = w.grad[i];
            }
        for (int it = 0; it < n_inner; ++it) {
            grp.sync();
            const real cost = inner_cost(0, true, w_coll);
            if (it > 0) {
                real sy = 0, yy = 0;
                for (int t = grp.rank; t < T; t += grp.size)
                    for (int d = 0; d < NA; ++d) {
                        const int i = t * NA + d;
                        const real yi = w.grad[i] - w.gprev[i];
                        sy += w.dir[i] * yi;
                        yy += yi * yi;
                    }
                sy = sum(sy);
                yy = sum(yy);
                if (sy > real(1e-10) * yy + real(1e-30)) {
                    const int nn = (newest + 1) % m_lbfgs;
                    for (int t = grp.rank; t < T; t += grp.size)
                        for (int d = 0; d < NA; ++d) {
                            const int i = t * NA + d;
                            w.s[nn * n + i] = w.dir[i];
                            w.y[nn * n + i] = w.grad[i] - w.gprev[i];
                        }
                    rho[nn] = real(1) / (sy + real(1e-30));
                    newest = nn;
                    if (m_used < m_lbfgs) ++m_used;
                }
            }
            // Two-loop recursion; every vector op stays on the lane's own timesteps.
            auto own = [&](auto f) {
                for (int t = grp.rank; t < T; t += grp.size)
                    for (int d = 0; d < NA; ++d) f(t * NA + d);
            };
            own([&](int i) { w.gprev[i] = w.grad[i]; w.dir[i] = -w.grad[i]; });
            if (m_used == 0) {
                real gg = 0;
                own([&](int i) { gg += w.grad[i] * w.grad[i]; });
                const real inv = real(1) / (sqrt(sum(gg)) + real(1e-18));
                own([&](int i) { w.dir[i] *= inv; });
            } else {
                for (int k = 0; k < m_used; ++k) {
                    const int j = (newest - k + m_lbfgs) % m_lbfgs;
                    real l = 0;
                    own([&](int i) { l += w.s[j * n + i] * w.dir[i]; });
                    ah[k] = sum(l) * rho[j];
                    own([&](int i) { w.dir[i] -= ah[k] * w.y[j * n + i]; });
                }
                real sy = 0, yy = 0;
                own([&](int i) {
                    sy += w.s[newest * n + i] * w.y[newest * n + i];
                    yy += w.y[newest * n + i] * w.y[newest * n + i];
                });
                const real gamma = sum(sy) / (sum(yy) + real(1e-18));
                own([&](int i) { w.dir[i] *= gamma; });
                for (int k = m_used - 1; k >= 0; --k) {
                    const int j = (newest - k + m_lbfgs) % m_lbfgs;
                    real l = 0;
                    own([&](int i) { l += w.y[j * n + i] * w.dir[i]; });
                    const real coeff = ah[k] - sum(l) * rho[j];
                    own([&](int i) { w.dir[i] += w.s[j * n + i] * coeff; });
                }
            }
            for (int t = grp.rank; t < T; t += grp.size)
                if (t == 0 || t == T - 1)
                    for (int d = 0; d < NA; ++d) w.dir[t * NA + d] = 0;
            grp.sync();

            const real LS_A[5] = {1, real(0.5), real(0.25), real(0.1), real(0.025)};
            real tc[5];
            for (int a = 0; a < 5; ++a) tc[a] = inner_cost(LS_A[a], false, w_coll);
            const real suff = cost * (1 - real(1e-4));
            int ba = 0;
            real bl = tc[0];
            for (int a = 1; a < 5; ++a) if (tc[a] < bl) { bl = tc[a]; ba = a; }
            for (int a = 0; a < 5; ++a) if (tc[a] < suff) { ba = a; bl = tc[a]; break; }
            own([&](int i) {
                const real step = LS_A[ba] * w.dir[i];
                w.traj[i] += step;
                w.dir[i] = step;
            });
            if (bl < best) {
                own([&](int i) { w.best[i] = w.traj[i]; });
                best = bl;
            }
        }
        for (int t = grp.rank; t < T; t += grp.size)
            for (int d = 0; d < NA; ++d) w.traj[t * NA + d] = w.best[t * NA + d];
        pin();
        grp.sync();
        w_coll = fmin(w_coll * penalty_scale, w_collision_max);
    }

    // True (nonlinear) cost of the result.
    real c = 0;
    for (int t = grp.rank; t < T; t += grp.size) {
        auto x = [&](int s, int d) { return w.traj[s * NA + d]; };
        c += w_collision_max * sdf_cost(w.traj + t * NA, world, margin);
        c += smooth_cost_at(t, T, x, w_smooth, w_acc, w_jerk);
        for (int d = 0; d < NA; ++d) {
            const real v = limit_viol(x(t, d), lower[d], upper[d]);
            c += w_limits * v * v;
        }
    }
    c = sum(c);
    for (int i = grp.rank; i < n; i += grp.size) out_trajs[(size_t)b * n + i] = (float)w.traj[i];
    if (grp.leader()) out_costs[b] = (float)c;
}

static ffi::Error CkScoTrajoptImpl(
    cudaStream_t stream, GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> init, ffi::Buffer<ffi::DataType::F32> ws,
    ffi::Buffer<ffi::DataType::F32> wc, ffi::Buffer<ffi::DataType::F32> wb,
    ffi::Buffer<ffi::DataType::F32> wh, ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper, ffi::Buffer<ffi::DataType::F32> start,
    ffi::Buffer<ffi::DataType::F32> goal,
    int64_t n_outer, int64_t n_inner, int64_t m_lbfgs, float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust, float w_collision, float w_collision_max, float penalty_scale,
    float margin, float tau, int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_trajs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_costs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> workspace)
{
    GRIM_ROT_UPLOAD(stream);
    const int B = (int)init.dimensions()[0], T = (int)init.dimensions()[1];
    if ((int)init.dimensions()[2] != NA || ws.dimensions()[0] != n_ws || wc.dimensions()[0] != n_wc ||
        wb.dimensions()[0] != n_wb || wh.dimensions()[0] != n_wh)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "sco_trajopt (crocket): launch does not match the robot and obstacle "
                          "counts of this build");
    if (m_lbfgs < 1 || m_lbfgs > SCO_MAX_M)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "sco_trajopt (crocket): m_lbfgs not in 1..8");
    if (workspace->element_count() * sizeof(float) < (size_t)B * sco_stride(T, (int)m_lbfgs) * sizeof(real))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "sco_trajopt (crocket): workspace too small");
    if (B == 0) return ffi::Error::Success();
    const size_t ep = (size_t)B * NA;
    const int ss = start.element_count() >= ep ? NA : 0, gs = goal.element_count() >= ep ? NA : 0;
    const World w{ws.typed_data(), wc.typed_data(), wb.typed_data(), wh.typed_data()};
    auto smem = [&](grim::Tier t) { return t == grim::Tier::Block ? (size_t)block_threads * sizeof(real) : 0; };
    TK_LAUNCH(ck_sco_trajopt_kernel, tier, B, (int)block_threads, smem, stream,
              init.typed_data(), w, lower.typed_data(), upper.typed_data(), start.typed_data(),
              goal.typed_data(), ss, gs, out_trajs->typed_data(), out_costs->typed_data(),
              reinterpret_cast<real*>(workspace->typed_data()), B, T, (int)n_outer, (int)n_inner,
              (int)m_lbfgs, w_smooth, w_acc, w_jerk, w_limits, w_trust, w_collision, w_collision_max,
              penalty_scale, margin, tau);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success() : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CkScoTrajoptFfi, CkScoTrajoptImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        GRIM_ROT_BIND
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // init_trajs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // start
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // goal
        .Attr<int64_t>("n_outer_iters")
        .Attr<int64_t>("n_inner_iters")
        .Attr<int64_t>("m_lbfgs")
        .Attr<float>("w_smooth")
        .Attr<float>("w_acc")
        .Attr<float>("w_jerk")
        .Attr<float>("w_limits")
        .Attr<float>("w_trust")
        .Attr<float>("w_collision")
        .Attr<float>("w_collision_max")
        .Attr<float>("penalty_scale")
        .Attr<float>("collision_margin")
        .Attr<float>("smooth_min_temperature")
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_trajs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_costs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // workspace (real)
);
