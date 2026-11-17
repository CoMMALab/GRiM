/**
 * What the trajectory optimizers share: mantis's traced collision terms and their analytic
 * gradients, the smoothness and limit terms, and the tier scaffolding, all in `real`
 * (GRIM_CK_REAL, float or double). See grim.motion._mantis_traj. Inputs and outputs stay
 * float32; every buffer in between is `real`.
 *
 * A trajectory (SCO, CHOMP) or a sample (STOMP's evaluation) is owned by one group: a thread,
 * a warp, or a block. Lanes stride over timesteps / perturbations / trials in GLASS form
 * (`for (i = rank; i < n; i += size)`), each posing its own configurations with mantis's
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
constexpr int NA = grim::robot::n_q;
constexpr int G = 6;   // SCO's smooth-min groups: self, spheres, capsules, boxes, halfspaces, cylinders

/** The world of one call: obstacle buffers and their counts, both runtime (no rebuild per world). */
struct World {
    const float* spheres; const float* capsules; const float* boxes; const float* halfspaces;
    const float* cylinders;
    int ns, nc, nb, nh, ny;
};

// The World of an FFI call, from its five obstacle buffers ws, wc, wb, wh, wy.
#define TK_WORLD                                                                               \
    grim::tk::World{ws.typed_data(), wc.typed_data(), wb.typed_data(), wh.typed_data(),        \
                    wy.typed_data(), (int)ws.dimensions()[0], (int)wc.dimensions()[0],         \
                    (int)wb.dimensions()[0], (int)wh.dimensions()[0], (int)wy.dimensions()[0]}

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

/** The launch's parent rotation as mantis's rot_joint wants it (see _mantis_ik.source). */
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

// The collision terms (grim.motion._mantis_traj). The self terms depend on the robot alone and
// are whole mantis traces (ck_self_*: straight-line, in registers). The world is a runtime
// argument: its terms are loops here, row by row (each row's sphere centres in registers once,
// the runtime obstacles inside), in mantis's term order and with its hard-min tie-break (the
// first minimum), each distance's gradient mantis's closed form (algorithms._obstacle); their
// per-sphere gradients reach dcost/dq through mantis's backward pass (ck_backprop).
namespace tg = mantis::robots::robot_grad;

constexpr int SLB = SL > 0 ? SL : 1;
constexpr int SMAX = S > 0 ? S : 1;   // spheres per row (TrajCollision's padded width)

/** One waypoint: mantis's q (and rot) and every live sphere's world centre. */
struct Pose {
    real q[NQ > 0 ? NQ : 1], rot[NROT > 0 ? NROT : 1], c[3 * SLB];
    template <class U>
    __device__ __forceinline__ explicit Pose(const U* cfg)
    {
        gather<real>(cfg, q);
        load_rot(rot);
        tg::ck_spheres_thread<real>(q, rot, c, 1);
    }
};

/** The self-term traces take the (empty) world inputs of traj_collision's signature. */
struct SelfArgs {
    real none[1] = {0}, prm[2];
    __device__ SelfArgs(real margin, real tau) : prm{margin, tau} {}
};
#define TK_SELF_IN(x, a) (x).q, (a).none, (a).none, (a).none, (a).none, (a).none, (a).prm, (x).rot

__device__ __forceinline__ real tk_dot(const real* a, const real* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/** Signed distance of sphere (c, r) to obstacle o of `kind` (1 sphere .. 5 cylinder), and its
 *  gradient in c (mantis's algorithms._obstacle). */
__device__ __forceinline__ real prim_dg(int kind, const real* c, real r, const float* o, real* g)
{
    if (kind == 1 || kind == 2) {   // sphere; capsule: distance to the clamped segment point
        real p[3] = {real(o[0]), real(o[1]), real(o[2])};
        real rad = kind == 1 ? real(o[3]) : real(o[6]);
        if (kind == 2) {
            const real ax[3] = {real(o[3]) - p[0], real(o[4]) - p[1], real(o[5]) - p[2]};
            const real len2 = tk_dot(ax, ax);
            const real ok = len2 > real(1e-12) ? real(1) : real(0);
            const real v[3] = {c[0] - p[0], c[1] - p[1], c[2] - p[2]};
            const real t = fmin(fmax(tk_dot(v, ax) / (len2 + (real(1) - ok)), real(0)), real(1)) * ok;
            for (int i = 0; i < 3; ++i) p[i] += t * ax[i];
        }
        const real v[3] = {c[0] - p[0], c[1] - p[1], c[2] - p[2]};
        const real n = sqrt(tk_dot(v, v));
        for (int i = 0; i < 3; ++i) g[i] = v[i] / n;
        return n - (rad + r);
    }
    if (kind == 3) {   // box: centre, 3 axes, half-extents
        const real v[3] = {c[0] - real(o[0]), c[1] - real(o[1]), c[2] - real(o[2])};
        real qk[3], m[3], sg[3];
        for (int k = 0; k < 3; ++k) {
            const real pr = v[0] * real(o[3 + 3 * k]) + v[1] * real(o[4 + 3 * k]) + v[2] * real(o[5 + 3 * k]);
            sg[k] = pr < real(0) ? real(-1) : real(1);
            qk[k] = fabs(pr) - real(o[12 + k]);
            m[k] = fmax(qk[k], real(0));
        }
        const real m2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
        const real out = m2 > real(0) ? real(1) : real(0);
        const real nm = sqrt(m2);
        const int kmax = qk[1] > qk[0] ? (qk[2] > qk[1] ? 2 : 1) : (qk[2] > qk[0] ? 2 : 0);
        for (int i = 0; i < 3; ++i) g[i] = 0;
        for (int k = 0; k < 3; ++k) {
            const real coef = out * m[k] / (nm + (real(1) - out)) + (real(1) - out) * (k == kmax ? real(1) : real(0));
            for (int i = 0; i < 3; ++i) g[i] += coef * sg[k] * real(o[3 + 3 * k + i]);
        }
        return nm + fmin(qk[kmax], real(0)) - r;
    }
    if (kind == 4) {   // halfspace: normal, a point on the plane
        for (int i = 0; i < 3; ++i) g[i] = real(o[i]);
        return (c[0] - real(o[3])) * real(o[0]) + (c[1] - real(o[4])) * real(o[1]) + (c[2] - real(o[5])) * real(o[2]) - r;
    }
    // solid cylinder: centre, unit axis, radius, half-height (the box formula over radial, axial)
    const real v[3] = {c[0] - real(o[0]), c[1] - real(o[1]), c[2] - real(o[2])};
    const real ax[3] = {real(o[3]), real(o[4]), real(o[5])};
    const real a = tk_dot(v, ax);
    const real w[3] = {v[0] - a * ax[0], v[1] - a * ax[1], v[2] - a * ax[2]};
    const real rho2 = tk_dot(w, w);
    const real ok = rho2 > real(1e-24) ? real(1) : real(0);
    const real rn = sqrt(rho2 + (real(1) - ok));
    const real rho = rn * ok;
    const real sg = a < real(0) ? real(-1) : real(1);
    const real q1 = rho - real(o[6]), q2 = fabs(a) - real(o[7]);
    const real m1 = fmax(q1, real(0)), m2 = fmax(q2, real(0));
    const real mm = m1 * m1 + m2 * m2;
    const real out = mm > real(0) ? real(1) : real(0);
    const real nm = sqrt(mm);
    const bool axial = q2 > q1;   // inside: the face of the larger excess (radial on a tie)
    const real cr = out * m1 / (nm + (real(1) - out)) + (real(1) - out) * (axial ? real(0) : real(1));
    const real cz = out * m2 / (nm + (real(1) - out)) + (real(1) - out) * (axial ? real(1) : real(0));
    for (int i = 0; i < 3; ++i) g[i] = cr * w[i] / rn + cz * sg * ax[i];
    return nm + fmin(axial ? q2 : q1, real(0)) - r;
}

/** Every world term at one waypoint, in mantis's order (row by row, obstacles by kind):
 *  coef = f(group, d), and when `gs` is given, gs[sphere] += coef * dd/dc for the argmin sphere. */
template <class F>
__device__ __forceinline__ void for_each_world_term(const Pose& x, const World& w, real* gs, F f)
{
    const float* base[5] = {w.spheres, w.capsules, w.boxes, w.halfspaces, w.cylinders};
    const int count[5] = {w.ns, w.nc, w.nb, w.nh, w.ny};
    constexpr int width[5] = {4, 7, 15, 6, 8};
#pragma unroll
    for (int r = 0; r < N; ++r) {
        const int s0 = kRowStart(r), ns = kRowStart(r + 1) - s0;
        if (ns == 0) continue;
        real cr[3 * SMAX];   // this row's centres, once
#pragma unroll
        for (int i = 0; i < 3 * SMAX; ++i) cr[i] = i < 3 * ns ? x.c[3 * s0 + i] : real(0);
#pragma unroll
        for (int k = 0; k < 5; ++k)
            for (int ob = 0; ob < count[k]; ++ob) {
                float o[15];
#pragma unroll
                for (int i = 0; i < width[k]; ++i) o[i] = base[k][(size_t)ob * width[k] + i];
                real best = real(1e30), gb[3] = {0, 0, 0};
                int sa = 0;
#pragma unroll
                for (int s = 0; s < SMAX; ++s) {
                    if (s >= ns) break;
                    real g[3];
                    const real d = prim_dg(k + 1, cr + 3 * s, real(kSphRad(s0 + s)), o, g);
                    if (s == 0 || d < best) { best = d; sa = s; gb[0] = g[0]; gb[1] = g[1]; gb[2] = g[2]; }
                }
                const real coef = f(k + 1, best);
                if (gs != nullptr && coef != real(0)) {
#pragma unroll
                    for (int s = 0; s < SMAX; ++s)
                        if (s < ns && s == sa)
                            for (int i = 0; i < 3; ++i) gs[3 * (s0 + s) + i] += coef * gb[i];
                }
            }
    }
}

/** dcost/dq in the configuration's order (NA) from per-sphere gradients gs (3 SL), added to grad. */
__device__ __forceinline__ void backprop_add(const Pose& x, const real* gs, real* grad)
{
    real g[NQ > 0 ? NQ : 1], out[NA];
    tg::ck_backprop_thread<real>(x.q, gs, x.rot, g, 1);
    for (int d = 0; d < NA; ++d) out[d] = 0;
    scatter<real>(g, out);
    for (int d = 0; d < NA; ++d) grad[d] += out[d];
}

/** CHOMP / STOMP: sum of squared hinges max(0, margin - d) at `cfg`. */
template <class U>
__device__ __forceinline__ real hinge_cost(const U* cfg, const World& w, real margin)
{
    const Pose x(cfg);
    const SelfArgs a(margin, real(0));
    real cost, g[1];
    tg::ck_self_hinge_thread<real>(TK_SELF_IN(x, a), &cost, g, 1);
    for_each_world_term(x, w, nullptr, [&](int, real d) {
        const real v = fmax(margin - d, real(0));
        cost += v * v;
        return real(0);
    });
    return cost;
}

/** The hinge cost and its gradient in `cfg`'s order (NA). */
template <class U>
__device__ __forceinline__ real hinge_cost_grad(const U* cfg, const World& w, real margin, real* grad)
{
    const Pose x(cfg);
    const SelfArgs a(margin, real(0));
    real cost, gq[NQ > 0 ? NQ : 1], gs[3 * SLB];
    tg::ck_self_hinge_grad_thread<real>(TK_SELF_IN(x, a), &cost, gq, 1);
    for (int d = 0; d < NA; ++d) grad[d] = 0;
    scatter<real>(gq, grad);
    for (int i = 0; i < 3 * SL; ++i) gs[i] = 0;
    for_each_world_term(x, w, gs, [&](int, real d) {
        const real v = fmax(margin - d, real(0));
        cost += v * v;
        return real(-2) * v;
    });
    backprop_add(x, gs, grad);
    return cost;
}

/** SCO's true collision cost at `cfg`: -sum min(colldist_from_sdf(d), 0). */
template <class U>
__device__ __forceinline__ real sdf_cost(const U* cfg, const World& w, real margin)
{
    const Pose x(cfg);
    const SelfArgs a(margin, real(0));
    const real inv = real(0.5) / (margin + real(1e-6));
    real cost, g[1];
    tg::ck_self_sdf_thread<real>(TK_SELF_IN(x, a), &cost, g, 1);
    for_each_world_term(x, w, nullptr, [&](int, real d) {
        if (d < real(0)) cost += real(0.5) * margin - d;
        else if (d < margin) cost += inv * (d - margin) * (d - margin);
        return real(0);
    });
    return cost;
}

/** The smooth-min distances and their Jacobian, row-major G x NA in `cfg`'s order: the self
 *  group from mantis's trace, the world groups from the loops. */
template <class U>
__device__ __forceinline__ void smooth_min_jac(const U* cfg, const World& w, real tau, real* out, real* jac)
{
    const Pose x(cfg);
    const SelfArgs a(real(0), tau);
    {
        real d[G], J[G * (NQ > 0 ? NQ : 1)];
        tg::ck_self_smin_thread<real>(TK_SELF_IN(x, a), d, J, 1);
        out[0] = d[0];
        for (int k = 0; k < NA; ++k) jac[k] = 0;
        scatter<real>(J, jac);
    }
    real mx[G], se[G];
    bool any[G];
    for (int k = 1; k < G; ++k) { mx[k] = real(-1e30); se[k] = 0; any[k] = false; }
    for_each_world_term(x, w, nullptr, [&](int k, real d) { mx[k] = fmax(mx[k], -d / tau); any[k] = true; return real(0); });
    for_each_world_term(x, w, nullptr, [&](int k, real d) { se[k] += exp(-d / tau - mx[k]); return real(0); });
    real gs[3 * SLB];
    for (int k = 1; k < G; ++k) {
        for (int d = 0; d < NA; ++d) jac[k * NA + d] = 0;
        if (!any[k]) {
            out[k] = real(1e10);
            continue;
        }
        out[k] = -tau * (log(se[k]) + mx[k]);
        for (int i = 0; i < 3 * SL; ++i) gs[i] = 0;
        for_each_world_term(x, w, gs, [&](int kk, real d) { return kk == k ? exp(-d / tau - mx[k]) / se[k] : real(0); });
        backprop_add(x, gs, jac + k * NA);
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
