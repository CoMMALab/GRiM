/**
 * What the trajectory optimizers share: crocket's traced collision terms and their analytic
 * gradients, the smoothness and limit terms, and the tier scaffolding, all in `real`
 * (GRIM_CK_REAL, float or double). See grim.motion._crocket_traj. Inputs and outputs stay
 * float32; every buffer in between is `real`.
 *
 * A trajectory (SCO, CHOMP) or a sample (STOMP's evaluation) is owned by one group: a thread,
 * a warp, or a block. Lanes stride over timesteps / perturbations / trials in GLASS form
 * (`for (i = rank; i < n; i += size)`), each posing its own configurations with crocket's
 * thread variant; group reductions are a fixed tree, so a result is independent of scheduling.
 */
#pragma once

#include "robot.cuh"
#include "glass_solve.cuh"

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifndef GRIM_CK_REAL
#define GRIM_CK_REAL float
#endif

namespace grim::tk {

using real = GRIM_CK_REAL;

constexpr int N = grim::robot::traj_N, S = grim::robot::traj_S, P = grim::robot::traj_P;
constexpr int n_ws = grim::robot::n_world_spheres, n_wc = grim::robot::n_world_capsules,
              n_wb = grim::robot::n_world_boxes,   n_wh = grim::robot::n_world_halfspaces;
constexpr int NA = grim::robot::n_q;
constexpr int G = 5;   // SCO's smooth-min groups: self, spheres, capsules, boxes, halfspaces

struct World {
    const float* spheres; const float* capsules; const float* boxes; const float* halfspaces;
};

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

template <grim::Tier TIER>
struct Grp {
    int rank, size;
    __device__ Grp()
        : rank(TIER == grim::Tier::Thread ? 0 : TIER == grim::Tier::Warp ? (int)(threadIdx.x & 31u) : (int)threadIdx.x),
          size(TIER == grim::Tier::Thread ? 1 : TIER == grim::Tier::Warp ? 32 : (int)blockDim.x) {}
    __device__ void sync() const
    {
        if constexpr (TIER == grim::Tier::Warp) __syncwarp();
        if constexpr (TIER == grim::Tier::Block) __syncthreads();
    }
    __device__ bool leader() const { return rank == 0; }
};

constexpr int WARPS = 4;      // warp tier: groups per block
constexpr int THREADS = 32;   // thread tier: groups per block

/** The group's index in the grid (group-uniform), and its slot in the block. */
template <grim::Tier TIER>
__device__ __forceinline__ int group_index()
{
    if constexpr (TIER == grim::Tier::Thread) return blockIdx.x * blockDim.x + threadIdx.x;
    else if constexpr (TIER == grim::Tier::Warp) return blockIdx.x * WARPS + (threadIdx.x >> 5);
    else return blockIdx.x;
}

template <grim::Tier TIER>
__device__ __forceinline__ int group_slot()
{
    return TIER == grim::Tier::Warp ? (int)(threadIdx.x >> 5) : 0;
}

/** Blocks and threads for `n` groups; `block_threads` lanes at the block tier. */
template <grim::Tier TIER>
inline dim3 grid_of(int n) { return TIER == grim::Tier::Thread ? (n + THREADS - 1) / THREADS : TIER == grim::Tier::Warp ? (n + WARPS - 1) / WARPS : n; }
template <grim::Tier TIER>
inline int threads_of(int block_threads) { return TIER == grim::Tier::Thread ? THREADS : TIER == grim::Tier::Warp ? 32 * WARPS : block_threads; }

/** Sum (or min) of each lane's `v`, identical on every lane: a fixed-order tree. `red` is the
 *  group's scratch of `size` reals (block tier only). */
template <grim::Tier TIER, bool MIN = false>
__device__ __forceinline__ real reduce(const Grp<TIER>& g, real* red, real v)
{
    auto op = [](real a, real b) { return MIN ? (b < a ? b : a) : a + b; };
    if constexpr (TIER == grim::Tier::Thread) {
        (void)red;
        return v;
    } else if constexpr (TIER == grim::Tier::Warp) {
        (void)red;
        for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_down_sync(0xffffffffu, v, o));
        return __shfl_sync(0xffffffffu, v, 0);
    } else {
        const int p2 = 1 << (31 - __clz(g.size));   // largest power of two <= size
        red[g.rank] = v;
        g.sync();
        if (g.rank < g.size - p2) red[g.rank] = op(red[g.rank], red[g.rank + p2]);
        g.sync();
        for (int o = p2 / 2; o > 0; o >>= 1) {
            if (g.rank < o) red[g.rank] = op(red[g.rank], red[g.rank + o]);
            g.sync();
        }
        const real out = red[0];
        g.sync();
        return out;
    }
}

/** Launch KERNEL<TIER> over `n` groups at the runtime tier `t` (0 thread, 1 warp, 2 block),
 *  with `smem(TIER)` bytes of dynamic shared memory. */
#define TK_LAUNCH(KERNEL, t, n, block_threads, SMEM, stream, ...)                                \
    do {                                                                                         \
        auto tk_go_ = [&](auto tier_tag) {                                                       \
            constexpr grim::Tier T_ = decltype(tier_tag)::value;                                \
            const size_t sm_ = (SMEM)(T_);                                                       \
            if (sm_ > 48 * 1024)                                                                 \
                cudaFuncSetAttribute(KERNEL<T_>, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                     (int)sm_);                                                  \
            KERNEL<T_><<<grim::tk::grid_of<T_>(n), grim::tk::threads_of<T_>(block_threads), sm_, \
                         (stream)>>>(__VA_ARGS__);                                               \
        };                                                                                       \
        if ((t) == 1) tk_go_(std::integral_constant<grim::Tier, grim::Tier::Warp>{});            \
        else if ((t) == 2) tk_go_(std::integral_constant<grim::Tier, grim::Tier::Block>{});      \
        else tk_go_(std::integral_constant<grim::Tier, grim::Tier::Thread>{});                   \
    } while (0)

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/** The launch's parent rotation as crocket's rot_joint wants it (see _crocket_ik.source). */
__device__ __forceinline__ void load_rot(real* rot)
{
#ifdef GRIM_RUNTIME_ROT_JOINT
    const float* g = grim::robot::kParentTf + GRIM_RUNTIME_ROT_JOINT * 7;
    const real a0 = kRotPre(0), a1 = kRotPre(1), a2 = kRotPre(2), a3 = kRotPre(3);
    const real b0 = g[0], b1 = g[1], b2 = g[2], b3 = g[3];
    rot[0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
    rot[1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
    rot[2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1;
    rot[3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0;
#else
    (void)rot;
#endif
}

// The collision terms (grim.motion._crocket_traj): crocket's traj_collision / traj_smooth_min,
// one waypoint each, gradients by its world-wrench backward pass. Values come from the same
// traces, so a cost and its gradient always agree.
namespace tg = crocket::robots::robot_grad;

struct Args {
    real q[NQ > 0 ? NQ : 1], rot[NROT > 0 ? NROT : 1];
    real ws[4 * n_ws + 1], wc[7 * n_wc + 1], wb[15 * n_wb + 1], wh[6 * n_wh + 1], prm[2];
    template <class U>
    __device__ __forceinline__ Args(const U* cfg, const World& w, real margin, real tau)
    {
        gather<real>(cfg, q);
        load_rot(rot);
        for (int i = 0; i < 4 * n_ws; ++i) ws[i] = w.spheres[i];
        for (int i = 0; i < 7 * n_wc; ++i) wc[i] = w.capsules[i];
        for (int i = 0; i < 15 * n_wb; ++i) wb[i] = w.boxes[i];
        for (int i = 0; i < 6 * n_wh; ++i) wh[i] = w.halfspaces[i];
        prm[0] = margin;
        prm[1] = tau;
    }
};

/** CHOMP / STOMP: sum of squared hinges max(0, margin - d) at `cfg`. */
template <class U>
__device__ __forceinline__ real hinge_cost(const U* cfg, const World& w, real margin)
{
    Args a(cfg, w, margin, real(0));
    real c, g[1];
    tg::ck_hinge_thread<real>(a.q, a.ws, a.wc, a.wb, a.wh, a.prm, a.rot, &c, g, 1);
    return c;
}

/** The hinge cost and its gradient in `cfg`'s order (NA). */
template <class U>
__device__ __forceinline__ real hinge_cost_grad(const U* cfg, const World& w, real margin, real* grad)
{
    Args a(cfg, w, margin, real(0));
    real c, g[NQ > 0 ? NQ : 1];
    tg::ck_hinge_grad_thread<real>(a.q, a.ws, a.wc, a.wb, a.wh, a.prm, a.rot, &c, g, 1);
    for (int d = 0; d < NA; ++d) grad[d] = 0;
    scatter<real>(g, grad);
    return c;
}

/** SCO's true collision cost at `cfg`: -sum min(colldist_from_sdf(d), 0). */
template <class U>
__device__ __forceinline__ real sdf_cost(const U* cfg, const World& w, real margin)
{
    Args a(cfg, w, margin, real(0));
    real c, g[1];
    tg::ck_sdf_thread<real>(a.q, a.ws, a.wc, a.wb, a.wh, a.prm, a.rot, &c, g, 1);
    return c;
}

/** The smooth-min distances and their Jacobian, row-major G x NA in `cfg`'s order. */
template <class U>
__device__ __forceinline__ void smooth_min_jac(const U* cfg, const World& w, real tau, real* out, real* jac)
{
    Args a(cfg, w, real(0), tau);
    real J[G * (NQ > 0 ? NQ : 1)];
    tg::ck_smin_jac_thread<real>(a.q, a.ws, a.wc, a.wb, a.wh, a.prm, a.rot, out, J, 1);
    for (int k = 0; k < G; ++k) {
        for (int d = 0; d < NA; ++d) jac[k * NA + d] = 0;
        scatter<real>(J + k * NQ, jac + k * NA);
    }
}

// ---------------------------------------------------------------------------
// Smoothness (5-point acceleration stencil and its first difference) and limits
// ---------------------------------------------------------------------------

__device__ __forceinline__ real st(int k)
{
    return k == 0 || k == 4 ? real(-1) / 12 : k == 1 || k == 3 ? real(16) / 12 : real(-30) / 12;
}

/** Timestep t's owned acceleration and jerk windows of x(s, d) (x: functor). */
template <class X>
__device__ __forceinline__ real smooth_cost_at(int t, int T, X x, real w_smooth, real w_acc, real w_jerk)
{
    real c = 0;
    if (t < T - 4)
        for (int d = 0; d < NA; ++d) {
            real a = 0;
            for (int k = 0; k < 5; ++k) a += st(k) * x(t + k, d);
            c += w_smooth * w_acc * a * a;
        }
    if (t < T - 5)
        for (int d = 0; d < NA; ++d) {
            real a0 = 0, a1 = 0;
            for (int k = 0; k < 5; ++k) { a0 += st(k) * x(t + k, d); a1 += st(k) * x(t + 1 + k, d); }
            const real j = a1 - a0;
            c += w_smooth * w_jerk * j * j;
        }
    return c;
}

/** d(smoothness)/d x(s, d) (without the w_smooth factor). */
template <class X>
__device__ __forceinline__ real smooth_grad_at(int s, int d, int T, X x, real w_acc, real w_jerk)
{
    const int n_acc = T >= 5 ? T - 4 : 0, n_jerk = T >= 6 ? T - 5 : 0;
    real g = 0;
    for (int tt = s >= 4 ? s - 4 : 0; tt <= (s < n_acc ? s : n_acc - 1); ++tt) {
        real a = 0;
        for (int k = 0; k < 5; ++k) a += st(k) * x(tt + k, d);
        g += 2 * w_acc * a * st(s - tt);
    }
    for (int tt = s >= 5 ? s - 5 : 0; tt <= (s < n_jerk ? s : n_jerk - 1); ++tt) {
        real a0 = 0, a1 = 0;
        for (int k = 0; k < 5; ++k) { a0 += st(k) * x(tt + k, d); a1 += st(k) * x(tt + 1 + k, d); }
        const int k1 = s - tt - 1, k2 = s - tt;
        const real c1 = k1 >= 0 && k1 <= 4 ? st(k1) : real(0);
        const real c2 = k2 >= 0 && k2 <= 4 ? st(k2) : real(0);
        g += 2 * w_jerk * (a1 - a0) * (c1 - c2);
    }
    return g;
}

__device__ __forceinline__ real limit_viol(real q, float lo, float hi)
{
    return fmax(real(0), q - hi) + fmax(real(0), real(lo) - q);
}

}  // namespace grim::tk
