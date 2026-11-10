/**
 * What the crocket trajectory optimizers share: the traced sphere centres, the collision and
 * smoothness terms over them, and the tier scaffolding, all in `real` (GRIM_CK_REAL, float or
 * double). See grim.motion._crocket_traj.
 *
 * Each term reproduces the kernel it replaced (kernels/trajopt/) in `real`: same distances,
 * same per-pair hard min, same smooth-min, same stencils, same summation order where the old
 * kernel summed serially. Inputs and outputs stay float32; every buffer in between is `real`.
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
constexpr int NSPHB = NSPH > 0 ? NSPH : 1;
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

/** World centre of every collision sphere at configuration `cfg` (NA, float or real). */
template <class U>
__device__ __forceinline__ void centres(const U* __restrict__ cfg, real* __restrict__ cw)
{
    real q[NQ > 0 ? NQ : 1], rot[NROT > 0 ? NROT : 1], r[1];
    gather<real>(cfg, q);
    load_rot(rot);
    tr::ik_geometry_value_thread<real>(q, nullptr, rot, r, cw, nullptr, 1);
}

// Distances (kernels/collision.cuh, in `real`).
__device__ __forceinline__ real d_sph(const real* a, real ra, const real* b, real rb)
{
    const real dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return sqrt(dx*dx + dy*dy + dz*dz) - (ra + rb);
}

__device__ __forceinline__ real d_sphere(const real* c, real rr, const float* o)
{
    const real dx = c[0] - o[0], dy = c[1] - o[1], dz = c[2] - o[2];
    return sqrt(dx*dx + dy*dy + dz*dz) - (rr + real(o[3]));
}

__device__ __forceinline__ real d_capsule(const real* c, real rr, const float* o)
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

__device__ __forceinline__ real d_box(const real* c, real rr, const float* o)
{
    const real dx = c[0] - o[0], dy = c[1] - o[1], dz = c[2] - o[2];
    real q[3];
    for (int k = 0; k < 3; ++k)
        q[k] = fabs(dx * o[3 + 3*k] + dy * o[4 + 3*k] + dz * o[5 + 3*k]) - real(o[12 + k]);
    const real m0 = fmax(q[0], real(0)), m1 = fmax(q[1], real(0)), m2 = fmax(q[2], real(0));
    return sqrt(m0*m0 + m1*m1 + m2*m2) + fmin(fmax(fmax(q[0], q[1]), q[2]), real(0)) - rr;
}

__device__ __forceinline__ real d_halfspace(const real* c, real rr, const float* o)
{
    return (c[0] - o[3]) * o[0] + (c[1] - o[4]) * o[1] + (c[2] - o[5]) * o[2] - rr;
}

/** collision.cuh's colldist_from_sdf in `real`. */
__device__ __forceinline__ real colldist_from_sdf(real d, real margin)
{
    d = fmin(d, margin);
    if (d < real(0)) return d - real(0.5) * margin;
    const real t = d - margin;
    return real(-0.5) / (margin + real(1e-6)) * t * t;
}

/** Hard min over the spheres of self pair p (1e10 when a row has none). */
__device__ __forceinline__ real self_min(const real* cw, int p)
{
    return self_min_gen<real>(cw, p, [](const real* a, real ra, const real* b, real rb) { return d_sph(a, ra, b, rb); });
}

/** Hard min over row j's spheres against obstacle `o` of `kind` (1e10 when the row has none). */
template <int KIND>
__device__ __forceinline__ real row_min(const real* cw, int j, const float* o)
{
    return row_min_gen<real>(cw, j, [o](const real* c, real r) {
        return KIND == 0 ? d_sphere(c, r, o) : KIND == 1 ? d_capsule(c, r, o)
             : KIND == 2 ? d_box(c, r, o) : d_halfspace(c, r, o);
    });
}

/** f(group, hard-min distance) over every self pair, then row-major over (row, obstacle) with
 *  obstacle kinds in WORLD_KINDS order: the old kernels' order. */
template <class F>
__device__ __forceinline__ void for_each_distance(const real* cw, const World& w, F f)
{
#pragma unroll
    for (int p = 0; p < P; ++p) {   // unrolled, the generated switches fold: cw stays in registers
        const real m = self_min(cw, p);
        if (m < real(1e9)) f(0, m);
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < n_ws; ++k) { const real m = row_min<0>(cw, j, w.spheres + 4 * k); if (m < real(1e9)) f(1, m); }
        for (int k = 0; k < n_wc; ++k) { const real m = row_min<1>(cw, j, w.capsules + 7 * k); if (m < real(1e9)) f(2, m); }
        for (int k = 0; k < n_wb; ++k) { const real m = row_min<2>(cw, j, w.boxes + 15 * k); if (m < real(1e9)) f(3, m); }
        for (int k = 0; k < n_wh; ++k) { const real m = row_min<3>(cw, j, w.halfspaces + 6 * k); if (m < real(1e9)) f(4, m); }
    }
}

#ifdef GRIM_CK_ANALYTIC
// The traced collision terms (grim.motion._crocket_traj._analytic_source): crocket's
// traj_collision / traj_smooth_min, one waypoint each, gradients by its world-wrench backward
// pass instead of central differences. Values come from the same traces, so a cost and its
// gradient always agree.
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

/** SCO's smooth-min distance of each group at `cfg` (1e10 for an empty group). */
template <class U>
__device__ __forceinline__ void smooth_min_dists(const U* cfg, const World& w, real tau, real* out)
{
    Args a(cfg, w, real(0), tau);
    real jac[1];
    tg::ck_smin_thread<real>(a.q, a.ws, a.wc, a.wb, a.wh, a.prm, a.rot, out, jac, 1);
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
#else
/** CHOMP / STOMP: sum of squared hinges max(0, margin - d) at `cfg`. */
template <class U>
__device__ __forceinline__ real hinge_cost(const U* cfg, const World& w, real margin)
{
    real cw[3 * NSPHB];
    centres(cfg, cw);
    real cost = 0;
    for_each_distance(cw, w, [&](int, real d) {
        const real v = fmax(real(0), margin - d);
        cost += v * v;
    });
    return cost;
}

/** SCO's true collision cost at `cfg`: -sum min(colldist_from_sdf(d), 0). */
template <class U>
__device__ __forceinline__ real sdf_cost(const U* cfg, const World& w, real margin)
{
    real cw[3 * NSPHB];
    centres(cfg, cw);
    real cost = 0;
    for_each_distance(cw, w, [&](int, real d) { cost -= fmin(colldist_from_sdf(d, margin), real(0)); });
    return cost;
}

/** SCO's smooth-min distance of each group at `cfg` (1e10 for an empty group). */
template <class U>
__device__ __forceinline__ void smooth_min_dists(const U* cfg, const World& w, real tau, real* out)
{
    real cw[3 * NSPHB];
    centres(cfg, cw);
    real mx[G], se[G];
    for (int g = 0; g < G; ++g) { mx[g] = real(-1e30); se[g] = 0; }
    for_each_distance(cw, w, [&](int g, real d) {
        const real v = -d / tau;
        if (v > mx[g]) { se[g] *= exp(mx[g] - v); mx[g] = v; }
        se[g] += exp(v - mx[g]);
    });
    for (int g = 0; g < G; ++g) out[g] = se[g] <= real(0) ? real(1e10) : -tau * (log(se[g]) + mx[g]);
}

#endif  // GRIM_CK_ANALYTIC

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
