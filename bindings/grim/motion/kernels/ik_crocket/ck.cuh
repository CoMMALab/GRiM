/**
 * What every crocket IK kernel shares: the traced robot geometry, the collision terms over it,
 * and the dense solve, all templated on the scalar type `real` (GRIM_CK_REAL, float or double).
 *
 * The robot is read through crocket's `ik_geometry` (grim.motion._crocket_ik), one traced call
 * per configuration: stacked pose residual r (6 per end-effector) and Jacobian J (row-major
 * M x NC), world centres of the world-collision spheres (cw) and self-collision spheres (cs),
 * and each joint's world twist tw = [w, v_o] (times its mimic scale). A sphere centre c then
 * moves with v_o + w x c per unit of its joint's column, so a collision gradient is a sum of
 * dot products over the sphere's chain (kWsChain / kSsChain). `ik_geometry_value` is the same
 * without J and tw, for line-search trials.
 *
 * The collision terms reproduce kernels/collision.cuh exactly (same distances, same active
 * test, same self-pair argmin, same central-difference world gradient), in `real`. The world
 * obstacle counts are the build's; the obstacles themselves are per call.
 *
 * Kernels run one seed per thread over the flat (problem, seed) index, so small seed counts
 * still fill blocks.
 */
#pragma once

#include "robot.cuh"
#include "glass_solve.cuh"

#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#ifndef GRIM_CK_REAL
#define GRIM_CK_REAL float
#endif

namespace grim::ck {

using real = GRIM_CK_REAL;

constexpr int NC = grim::robot::n_solved;
constexpr int NE = grim::robot::n_ee;
constexpr int M  = 6 * NE;
constexpr int NF = grim::robot::n_frozen > 0 ? grim::robot::n_frozen : 1;
constexpr int NWSB = NWS > 0 ? NWS : 1;
constexpr int NSSB = NSS > 0 ? NSS : 1;
constexpr int n_ws = grim::robot::n_world_spheres, n_wc = grim::robot::n_world_capsules,
              n_wb = grim::robot::n_world_boxes,   n_wh = grim::robot::n_world_halfspaces;
constexpr bool has_world_geom = NWS > 0 && (n_ws + n_wc + n_wb + n_wh) > 0;
constexpr bool has_self = grim::robot::n_self_pairs > 0;
constexpr int THREADS = 32;   // thread tier: seeds per block
constexpr int CK_MAX_EVAL = 32;  // independent evaluations held at once (trials, particles)

/** Everything one configuration's traced geometry call produces. */
struct Geom {
    real r[M];
    real J[M * NC];
    real cw[3 * NWSB];
    real cs[3 * NSSB];
    real tw[NWS + NSS > 0 ? 6 * NJ : 1];   // twists: only collision gradients read them
};

/** A trial's geometry: residual and sphere centres only. */
struct GeomV {
    real r[M];
    real cw[3 * NWSB];
    real cs[3 * NSSB];
};

/** Per-seed state: the solved variables' configuration and the frozen joints carried along. */
struct Seed {
    real frz[NF];
    real target[7 * NE];
    real rot[NROT > 0 ? NROT : 1];   // crocket's per-call parent rotation (runtime_rot_joint builds)
};

/** sd.rot = kRotPre * the launch's rotation (GRIM_ROT_UPLOAD wrote it into kParentTf), as
 *  crocket's rot_joint wants it: see _crocket_ik.source. */
__device__ __forceinline__ void load_rot(Seed& sd)
{
#ifdef GRIM_RUNTIME_ROT_JOINT
    const float* g = grim::robot::kParentTf + GRIM_RUNTIME_ROT_JOINT * 7;
    const real a0 = kRotPre(0), a1 = kRotPre(1), a2 = kRotPre(2), a3 = kRotPre(3);
    const real b0 = g[0], b1 = g[1], b2 = g[2], b3 = g[3];
    sd.rot[0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
    sd.rot[1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
    sd.rot[2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1;
    sd.rot[3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0;
#else
    (void)sd;
#endif
}

__device__ __forceinline__ void load_seed(const float* __restrict__ seeds, const float* __restrict__ targets,
                                          int gs, int problem, real* __restrict__ cfg, Seed& sd)
{
    const float* row = seeds + (size_t)gs * grim::robot::n_q;
    for (int i = 0; i < NC; ++i) cfg[i] = row[grim::robot::solved_idx(i)];
    for (int k = 0; k < grim::robot::n_frozen; ++k) sd.frz[k] = row[grim::robot::frozen_idx(k)];
    for (int i = 0; i < 7 * NE; ++i) sd.target[i] = targets[(size_t)problem * 7 * NE + i];
    load_rot(sd);
}

__device__ __forceinline__ void store(const real* __restrict__ cfg, const Seed& sd,
                                      float* __restrict__ out, int gs)
{
    float* row = out + (size_t)gs * grim::robot::n_q;
    for (int i = 0; i < NC; ++i) row[grim::robot::solved_idx(i)] = (float)cfg[i];
    for (int k = 0; k < grim::robot::n_frozen; ++k) row[grim::robot::frozen_idx(k)] = (float)sd.frz[k];
}

/** Residual, Jacobian, sphere centres and twists at `cfg`. */
__device__ __forceinline__ void geometry(const real* __restrict__ cfg, const Seed& sd, Geom& g)
{
    real q[NQ];
    gather<real>(cfg, sd.frz, q);
    tr::ik_geometry_thread<real>(q, sd.target, sd.rot, g.r, g.J, g.cw, g.cs, g.tw, 1);
}

/** Residual and sphere centres at `cfg` (no Jacobian): a trial's merit. One lane, any tier. */
template <class G>
__device__ __forceinline__ void geometry_value(const real* __restrict__ cfg, const Seed& sd, G& g)
{
    real q[NQ];
    gather<real>(cfg, sd.frz, q);
    tr::ik_geometry_value_thread<real>(q, sd.target, sd.rot, g.r, g.cw, g.cs, 1);
}

__device__ __forceinline__ real nrm3(const real* v) { return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
__device__ __forceinline__ real clamp(real x, real lo, real hi) { return fmax(lo, fmin(hi, x)); }

/** Every end-effector inside (eps_pos, eps_ori). */
__device__ __forceinline__ bool converged(const real* r, real eps_pos, real eps_ori)
{
    for (int e = 0; e < NE; ++e)
        if (nrm3(r + 6 * e) >= eps_pos || nrm3(r + 6 * e + 3) >= eps_ori) return false;
    return true;
}

/** GRiM's trust-region radius from the largest end-effector errors. */
__device__ __forceinline__ real trust_radius(const real* r)
{
    real mp = 0, mo = 0;
    for (int e = 0; e < NE; ++e) { mp = fmax(mp, nrm3(r + 6 * e)); mo = fmax(mo, nrm3(r + 6 * e + 3)); }
    if (mp > real(1e-2) || mo > real(0.6))  return real(0.38);
    if (mp > real(1e-3) || mo > real(0.25)) return real(0.22);
    if (mp > real(2e-4) || mo > real(0.08)) return real(0.12);
    return real(0.05);
}

__device__ __forceinline__ void clip_to_radius(real* d, real R)
{
    real n = 0;
    for (int a = 0; a < NC; ++a) n += d[a] * d[a];
    n = sqrt(n);
    if (n > R) { const real s = R / (n + real(1e-18)); for (int a = 0; a < NC; ++a) d[a] *= s; }
}

// ---------------------------------------------------------------------------
// Distances (collision.cuh, in `real`)
// ---------------------------------------------------------------------------

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

/** The world obstacles of one call (counts fixed by the build). */
struct World {
    const float* spheres; const float* capsules; const float* boxes; const float* halfspaces;
};

__device__ __forceinline__ real d_prim(const real* c, real rr, int kind, const float* o)
{
    switch (kind) {
        case 0:  return d_sphere(c, rr, o);
        case 1:  return d_capsule(c, rr, o);
        case 2:  return d_box(c, rr, o);
        default: return d_halfspace(c, rr, o);
    }
}

template <class F>
__device__ __forceinline__ void for_each_prim(const World& w, F f)
{
    for (int m = 0; m < n_ws; ++m) f(0, w.spheres + 4 * m);
    for (int m = 0; m < n_wc; ++m) f(1, w.capsules + 7 * m);
    for (int m = 0; m < n_wb; ++m) f(2, w.boxes + 15 * m);
    for (int m = 0; m < n_wh; ++m) f(3, w.halfspaces + 6 * m);
}

__device__ __forceinline__ real d_sph_sph(const real* a, real ra, const real* b, real rb)
{
    const real dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return sqrt(dx*dx + dy*dy + dz*dz) - (ra + rb);
}

/** Minimum distance over one self-collision link pair, and its argmin sphere pair. */
__device__ __forceinline__ real self_pair(const real* cs, int p, int& ia, int& ib)
{
    const int li = grim::robot::kSelfPairI[p], lj = grim::robot::kSelfPairJ[p];
    real best = real(1e9);
    ia = ib = -1;
    for (int a = grim::robot::kSelfLinkStart[li]; a < grim::robot::kSelfLinkStart[li + 1]; ++a)
        for (int b = grim::robot::kSelfLinkStart[lj]; b < grim::robot::kSelfLinkStart[lj + 1]; ++b) {
            const real d = d_sph_sph(cs + 3 * a, real(kSsRad[a]), cs + 3 * b, real(kSsRad[b]));
            if (d < best) { best = d; ia = a; ib = b; }
        }
    return best;
}

/** Unweighted collision penalty: sum over active constraints of (d - margin)^2 (one lane). */
template <class G>
__device__ __forceinline__ real collision_raw(const G& g, const World& w, bool want_world, real margin)
{
    real pen = 0;
    if constexpr (has_world_geom) {
        if (want_world)
            for (int i = 0; i < NWS; ++i)
                for_each_prim(w, [&](int kind, const float* o) {
                    const real d = d_prim(g.cw + 3 * i, real(kWsRad[i]), kind, o);
                    if (d < margin) pen += (d - margin) * (d - margin);
                });
    }
    if constexpr (has_self) {
        for (int p = 0; p < grim::robot::n_self_pairs; ++p) {
            int ia, ib;
            const real d = self_pair(g.cs, p, ia, ib);
            if (d < margin) pen += (d - margin) * (d - margin);
        }
    }
    return pen;
}

/** Minimum clearance (distance - margin) over every pair, for canonical IK's guard. */
template <class G>
__device__ __forceinline__ real min_clearance(const G& g, const World& w, real margin)
{
    real best = real(1e9);
    if constexpr (has_self) {
        for (int p = 0; p < grim::robot::n_self_pairs; ++p) {
            int ia, ib;
            best = fmin(best, self_pair(g.cs, p, ia, ib) - margin);
        }
    }
    if constexpr (has_world_geom) {
        for (int i = 0; i < NWS; ++i)
            for_each_prim(w, [&](int kind, const float* o) {
                best = fmin(best, d_prim(g.cw + 3 * i, real(kWsRad[i]), kind, o) - margin);
            });
    }
    return best;
}

/** g[col] += sign * u . (v_o + w x pt) over the joints on sphere `s`'s chain (WORLD: kWsChain,
 *  else kSsChain). */
template <bool WORLD>
__device__ __forceinline__ void point_grad(const Geom& g, int s, const real* pt,
                                           const real* u, real sign, real* grad)
{
    #pragma unroll
    for (int j = 0; j < NJ; ++j) {
        if (!(WORLD ? kWsChain[NJ * s + j] : kSsChain[NJ * s + j]) || kJointCol(j) < 0) continue;
        const real* t = g.tw + 6 * j;
        const real vx = t[3] + t[1] * pt[2] - t[2] * pt[1];
        const real vy = t[4] + t[2] * pt[0] - t[0] * pt[2];
        const real vz = t[5] + t[0] * pt[1] - t[1] * pt[0];
        grad[kJointCol(j)] += sign * (u[0] * vx + u[1] * vy + u[2] * vz);
    }
}

/**
 * Every active collision constraint as `f(grad, viol)`, grad = dd/dcfg (NC), viol = margin - d > 0:
 * self pairs first, then each world sphere against each obstacle, as collision_gauss_newton_terms.
 */
template <class F>
__device__ __forceinline__ void collision_terms(const Geom& g, const World& w, bool want_world,
                                                real margin, F f)
{
    real grad[NC > 0 ? NC : 1];
    if constexpr (has_self) {
        for (int p = 0; p < grim::robot::n_self_pairs; ++p) {
            int ia, ib;
            const real d = self_pair(g.cs, p, ia, ib);
            if (d >= margin) continue;
            const real* ci = g.cs + 3 * ia;
            const real* cj = g.cs + 3 * ib;
            const real ux = ci[0] - cj[0], uy = ci[1] - cj[1], uz = ci[2] - cj[2];
            const real n = sqrt(ux*ux + uy*uy + uz*uz);
            if (n < real(1e-9)) continue;
            const real u[3] = {ux / n, uy / n, uz / n};
            for (int a = 0; a < NC; ++a) grad[a] = 0;
            point_grad<false>(g, ia, ci, u, real(1), grad);
            point_grad<false>(g, ib, cj, u, real(-1), grad);
            f(grad, margin - d);
        }
    }
    if constexpr (has_world_geom) {
        if (!want_world) return;
        for (int i = 0; i < NWS; ++i) {
            const real* c = g.cw + 3 * i;
            const real rr = real(kWsRad[i]);
            for_each_prim(w, [&](int kind, const float* o) {
                const real d = d_prim(c, rr, kind, o);
                if (d >= margin) return;
                // Central differences on the distance, as world_prim_grad_dir.
                const real h = real(1e-4);
                real u[3], cp[3] = {c[0], c[1], c[2]};
                for (int ax = 0; ax < 3; ++ax) {
                    cp[ax] = c[ax] + h;
                    const real dp = d_prim(cp, rr, kind, o);
                    cp[ax] = c[ax] - h;
                    const real dm = d_prim(cp, rr, kind, o);
                    cp[ax] = c[ax];
                    u[ax] = (dp - dm) * (real(0.5) / h);
                }
                const real n2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
                if (n2 < real(1e-12)) return;
                const real inv = real(1) / sqrt(n2);
                u[0] *= inv; u[1] *= inv; u[2] *= inv;
                for (int a = 0; a < NC; ++a) grad[a] = 0;
                point_grad<true>(g, i, c, u, real(1), grad);
                f(grad, margin - d);
            });
        }
    }
}

/** Weighted pose error |W r|^2 with W = (pos_weight x3, ori_weight x3) per end-effector. */
__device__ __forceinline__ real weighted_err(const real* r, real wp, real wo)
{
    real e = 0;
    for (int k = 0; k < M; ++k) { const real v = r[k] * ((k % 6) < 3 ? wp : wo); e += v * v; }
    return e;
}

/** SPD solve A x = b in place (A destroyed), N = NC; on a non-PD pivot b is zeroed. */
template <int N>
__device__ __forceinline__ bool posv(real* A, real* b)
{
    int fail = 0;
    glass::thread::potrf<real, (uint32_t)N, /*CHECK=*/true>(A, &fail);
    if (fail) { for (int i = 0; i < N; ++i) b[i] = 0; return false; }
    glass::thread::potrs<real, (uint32_t)N>(A, b);
    return true;
}


// ---------------------------------------------------------------------------
// Tiers: one kernel body at thread / warp / block
// ---------------------------------------------------------------------------
// A seed (canonical: a problem; analytic: a target) is owned by one thread, one warp (32
// lanes) or one block (BLOCK_THREADS lanes). Every loop that can split is written
// `for (i = G.rank; i < n; i += G.size)`, which at the thread tier (rank 0, size 1) is the
// sequential loop, so the thread tier compiles to the one-seed-per-thread kernel. State that
// every lane needs (the configuration, damping, best error) is kept per lane and updated
// identically from shared data; per-seed buffers (Work) live in registers / local memory at
// the thread tier and in shared memory, one slot per seed, at the cooperative tiers.
//
// The traced geometry is crocket's variant of the same tier (thread / warp / block). Line-search
// trials, particles and candidates are independent: lanes take one each with crocket's thread
// variant. Collision constraints are taken LANES at a time (rounds), their rows written to shared
// memory and summed in a fixed order, so results do not depend on scheduling.

template <grim::Tier TIER>
__host__ __device__ constexpr int lanes() { return TIER == grim::Tier::Thread ? 1 : TIER == grim::Tier::Warp ? 32 : BLOCK_THREADS; }

template <grim::Tier TIER>
struct Grp {
    int rank, size;
    __device__ Grp()
        : rank(TIER == grim::Tier::Thread ? 0 : TIER == grim::Tier::Warp ? (int)(threadIdx.x & 31u) : (int)threadIdx.x),
          size(lanes<TIER>()) {}
    __device__ void sync() const
    {
        if constexpr (TIER == grim::Tier::Warp) __syncwarp();
        if constexpr (TIER == grim::Tier::Block) __syncthreads();
#ifdef GRIM_CK_SKEW_NS
        // Test builds (ik.build(debug_skew_ns=...)): the upper lanes leave every barrier late, so
        // a phase missing its barrier reads what the lower lanes already rewrote, every run.
        if constexpr (TIER != grim::Tier::Thread)
            if (rank >= (TIER == grim::Tier::Warp ? 16 : 32)) __nanosleep(GRIM_CK_SKEW_NS);
#endif
    }
    __device__ bool leader() const { return rank == 0; }
};

constexpr int DN = NC > M ? NC : M;                     // dense system size (LS/SQP: NC, DLS/canonical: M)
__host__ __device__ constexpr int cmax(int a, int b) { return a > b ? a : b; }

/** Per-seed buffers. N: the factored system's size (LS/SQP/HJCD: NC; DLS/canonical: M). H: keep the
 *  system before factoring (SQP's active-set / ADMM re-solves). GV: a cooperative value call. */
struct NoGV {};
template <grim::Tier TIER, int N = NC, bool WITH_H = false, bool WITH_GV = false>
struct Work {
    static constexpr int L = lanes<TIER>();
    static constexpr int SCR = TIER == grim::Tier::Warp
        ? cmax(cmax(tr::ik_geometry_warp_smem, tr::ik_geometry_value_warp_smem), 1)
        : TIER == grim::Tier::Block
        ? cmax(cmax(tr::ik_geometry_block_smem, tr::ik_geometry_value_block_smem), 1) : 1;
    static constexpr bool coll = has_self || has_world_geom;
    Geom g;                                               // the iteration's geometry
    std::conditional_t<WITH_GV, GeomV, NoGV> gv;
    real A[N * N];                                        // the factored system
    real H[WITH_H ? NC * NC : 1];
    real b[N];
    real x[N];
    real gbuf[coll ? L * NC : 1];                         // a collision round's gradients
    real vbuf[coll ? L : 1];                              //   and violations
    real red[TIER == grim::Tier::Block ? L : 1];          // block reductions
    real err[CK_MAX_EVAL];                                // independent evaluations
    real scr[SCR];                                        // crocket's cooperative scratch
    int flag;
};

/** Seeds per block at the warp tier: as many warps as their Work slots fit in 40 KB of shared. */
template <class W>
__host__ __device__ constexpr int warps_per_block() { return sizeof(W) * 8 <= 40960 ? 8 : sizeof(W) * 4 <= 40960 ? 4
                                       : sizeof(W) * 2 <= 40960 ? 2 : 1; }

/** The seed this group owns (group-uniform), and its Work slot within the block. */
template <grim::Tier TIER>
__device__ __forceinline__ int seed_index(int wpb)
{
    if constexpr (TIER == grim::Tier::Thread) return blockIdx.x * blockDim.x + threadIdx.x;
    else if constexpr (TIER == grim::Tier::Warp) return blockIdx.x * wpb + (threadIdx.x >> 5);
    else return blockIdx.x;
}

/** This seed's Work: thread-local at the thread tier; a slot of the dynamic shared memory the
 *  launch sizes (launch_shape) at the cooperative tiers. */
template <class W, bool LOCAL>
struct Local { W w; };
template <class W>
struct Local<W, false> {};

template <grim::Tier TIER, class W>
__device__ __forceinline__ W& work_slot(Local<W, TIER == grim::Tier::Thread>& l, unsigned char* smem)
{
    if constexpr (TIER == grim::Tier::Thread) { (void)smem; return l.w; }
    else return reinterpret_cast<W*>(smem)[TIER == grim::Tier::Warp ? (threadIdx.x >> 5) : 0];
}

/** Declare `w`, this seed's Work of type W (a Work<TIER, ...>). */
#define CK_WORK(TIER, W, w)                                                    \
    Local<W, (TIER) == grim::Tier::Thread> w##_local_;                         \
    extern __shared__ __align__(16) unsigned char ck_smem_[];                  \
    W& w = work_slot<TIER, W>(w##_local_, ck_smem_)

/** Grid, block and dynamic shared memory for `n` seeds (problems / targets) at TIER. */
struct Shape { int blocks, threads, wpb; size_t smem; };

template <grim::Tier TIER, class W>
inline Shape launch_shape(int n)
{
    if constexpr (TIER == grim::Tier::Thread) return {(n + THREADS - 1) / THREADS, THREADS, 1, 0};
    else if constexpr (TIER == grim::Tier::Warp) {
        constexpr int wpb = warps_per_block<W>();
        return {(n + wpb - 1) / wpb, wpb * 32, wpb, wpb * sizeof(W)};
    } else return {n, BLOCK_THREADS, 1, sizeof(W)};
}

/** Launch KERNEL<TIER> over `n` seeds at the runtime tier `t` (0 thread, 1 warp, 2 block), its
 *  Work being WORK<TIER>. The kernel takes the warp tier's seeds-per-block as its first argument. */
#define CK_LAUNCH(KERNEL, WORK, t, n, stream, ...)                                             \
    do {                                                                                       \
        auto ck_go_ = [&](auto tier_tag) {                                                     \
            constexpr grim::Tier T_ = decltype(tier_tag)::value;                              \
            const grim::ck::Shape sh_ = grim::ck::launch_shape<T_, WORK<T_>>(n);                \
            if (sh_.smem > 48 * 1024)                                                          \
                cudaFuncSetAttribute(KERNEL<T_>, cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                                     (int)sh_.smem);                                           \
            KERNEL<T_><<<sh_.blocks, sh_.threads, sh_.smem, (stream)>>>(sh_.wpb, __VA_ARGS__); \
        };                                                                                     \
        if ((t) == 1) ck_go_(std::integral_constant<grim::Tier, grim::Tier::Warp>{});          \
        else if ((t) == 2) ck_go_(std::integral_constant<grim::Tier, grim::Tier::Block>{});    \
        else ck_go_(std::integral_constant<grim::Tier, grim::Tier::Thread>{});                 \
    } while (0)

/** Residual, Jacobian, centres and twists at `cfg` into w.g, by crocket's variant of TIER. */
template <grim::Tier TIER, class W>
__device__ __forceinline__ void geometry(const Grp<TIER>& G, const real* __restrict__ cfg, const Seed& sd, W& w)
{
    G.sync();   // every lane is done reading the previous w.g / scratch
    real q[NQ];
    gather<real>(cfg, sd.frz, q);
    if constexpr (TIER == grim::Tier::Thread) tr::ik_geometry_thread<real>(q, sd.target, sd.rot, w.g.r, w.g.J, w.g.cw, w.g.cs, w.g.tw, 1);
    else if constexpr (TIER == grim::Tier::Warp) tr::ik_geometry_warp<real>(q, sd.target, sd.rot, w.g.r, w.g.J, w.g.cw, w.g.cs, w.g.tw, w.scr, 1);
    else tr::ik_geometry_block<real>(q, sd.target, sd.rot, w.g.r, w.g.J, w.g.cw, w.g.cs, w.g.tw, w.scr, 1);
    G.sync();
}

/** Residual and centres at `cfg` into w.gv, by crocket's variant of TIER. */
template <grim::Tier TIER, class W>
__device__ __forceinline__ void geometry_value(const Grp<TIER>& G, const real* __restrict__ cfg, const Seed& sd, W& w)
{
    G.sync();
    real q[NQ];
    gather<real>(cfg, sd.frz, q);
    if constexpr (TIER == grim::Tier::Thread) tr::ik_geometry_value_thread<real>(q, sd.target, sd.rot, w.gv.r, w.gv.cw, w.gv.cs, 1);
    else if constexpr (TIER == grim::Tier::Warp) tr::ik_geometry_value_warp<real>(q, sd.target, sd.rot, w.gv.r, w.gv.cw, w.gv.cs, w.scr, 1);
    else tr::ik_geometry_value_block<real>(q, sd.target, sd.rot, w.gv.r, w.gv.cw, w.gv.cs, w.scr, 1);
    G.sync();
}

__host__ __device__ constexpr int pow2_floor(int n) { return n < 2 ? 1 : 2 * pow2_floor(n / 2); }

/** Sum of each lane's `v`, identical on every lane (fixed-order tree, then a broadcast). */
template <grim::Tier TIER, class W>
__device__ __forceinline__ real reduce_sum(const Grp<TIER>& G, W& w, real v)
{
    if constexpr (TIER == grim::Tier::Thread) {
        return v;
    } else if constexpr (TIER == grim::Tier::Warp) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        return __shfl_sync(0xffffffffu, v, 0);
    } else {
        constexpr int P2 = pow2_floor(BLOCK_THREADS);
        w.red[G.rank] = v;
        G.sync();
        if constexpr (P2 < BLOCK_THREADS) {   // fold the lanes past the power of two in first
            if (G.rank < BLOCK_THREADS - P2) w.red[G.rank] += w.red[G.rank + P2];
            G.sync();
        }
        for (int o = P2 / 2; o > 0; o >>= 1) {
            if (G.rank < o) w.red[G.rank] += w.red[G.rank + o];
            G.sync();
        }
        const real out = w.red[0];
        G.sync();
        return out;
    }
}

/** Flat collision constraint index: self pairs first, then world sphere-major over obstacles. */
constexpr int NPRIM = n_ws + n_wc + n_wb + n_wh;
constexpr int N_CON = (has_self ? grim::robot::n_self_pairs : 0) + (has_world_geom ? NWS * NPRIM : 0);

__device__ __forceinline__ void prim_of(const World& w, int m, int& kind, const float*& o)
{
    if (m < n_ws) { kind = 0; o = w.spheres + 4 * m; return; }
    m -= n_ws;
    if (m < n_wc) { kind = 1; o = w.capsules + 7 * m; return; }
    m -= n_wc;
    if (m < n_wb) { kind = 2; o = w.boxes + 15 * m; return; }
    m -= n_wb;
    kind = 3; o = w.halfspaces + 6 * m;
}

/** Constraint `c`'s distance (and, with `grad`, dd/dcfg), on one lane; false when out of play. */
template <class Gm>
__device__ __forceinline__ bool constraint(const Gm& g, const World& world, bool want_world, int c,
                                           real& d, real* grad, real margin)
{
    constexpr int P = has_self ? grim::robot::n_self_pairs : 0;
    if (c < P) {
        if constexpr (has_self) {
            int ia, ib;
            d = self_pair(g.cs, c, ia, ib);
            if (grad == nullptr || d >= margin) return true;
            const real* ci = g.cs + 3 * ia;
            const real* cj = g.cs + 3 * ib;
            const real ux = ci[0] - cj[0], uy = ci[1] - cj[1], uz = ci[2] - cj[2];
            const real n = sqrt(ux*ux + uy*uy + uz*uz);
            if (n < real(1e-9)) { d = margin; return false; }   // skipped, as the serial walk
            const real u[3] = {ux / n, uy / n, uz / n};
            for (int a = 0; a < NC; ++a) grad[a] = 0;
            if constexpr (std::is_same_v<Gm, Geom>) {
                point_grad<false>(g, ia, ci, u, real(1), grad);
                point_grad<false>(g, ib, cj, u, real(-1), grad);
            }
        }
        return true;
    }
    if constexpr (has_world_geom) {
        if (!want_world) return false;
        c -= P;
        const int i = c / NPRIM;
        int kind;
        const float* o;
        prim_of(world, c % NPRIM, kind, o);
        const real* cc = g.cw + 3 * i;
        const real rr = real(kWsRad[i]);
        d = d_prim(cc, rr, kind, o);
        if (grad == nullptr || d >= margin) return true;
        const real h = real(1e-4);
        real u[3], cp[3] = {cc[0], cc[1], cc[2]};
        for (int ax = 0; ax < 3; ++ax) {
            cp[ax] = cc[ax] + h;
            const real dp = d_prim(cp, rr, kind, o);
            cp[ax] = cc[ax] - h;
            const real dm = d_prim(cp, rr, kind, o);
            cp[ax] = cc[ax];
            u[ax] = (dp - dm) * (real(0.5) / h);
        }
        const real n2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
        if (n2 < real(1e-12)) { d = margin; return false; }
        const real inv = real(1) / sqrt(n2);
        u[0] *= inv; u[1] *= inv; u[2] *= inv;
        for (int a = 0; a < NC; ++a) grad[a] = 0;
        if constexpr (std::is_same_v<Gm, Geom>) point_grad<true>(g, i, cc, u, real(1), grad);
        return true;
    }
    return false;
}

/** Unweighted penalty of w.g (or w.gv): constraints split over lanes, a fixed-order sum. */
template <grim::Tier TIER, class W, class Gm>
__device__ __forceinline__ real collision_raw_coop(const Grp<TIER>& G, W& w, const Gm& g,
                                                    const World& world, bool want_world, real margin)
{
    if constexpr (TIER == grim::Tier::Thread) {   // the nested walk, kinds resolved at compile time
        (void)G; (void)w;
        return collision_raw(g, world, want_world, margin);
    }
    real pen = 0;
    for (int c = G.rank; c < N_CON; c += G.size) {
        real d;
        if (constraint(g, world, want_world, c, d, (real*)nullptr, margin) && d < margin) pen += (d - margin) * (d - margin);
    }
    return reduce_sum(G, w, pen);
}

/**
 * The active constraints of w.g, a round of `lanes` at a time: each lane writes its constraint's
 * gradient (divided by `sc` unless null) to w.gbuf[rank] and violation to w.vbuf[rank]
 * (0: inactive), then every lane calls
 * `f(n)` with the round's n slots in constraint order. Same constraints, order and values as
 * collision_terms.
 */
template <grim::Tier TIER, class W, class F>
__device__ __forceinline__ void collision_rounds(const Grp<TIER>& G, W& w, const World& world,
                                                 bool want_world, real margin, const real* sc, F f)
{
    if constexpr (TIER == grim::Tier::Thread) {   // collision_terms' walk: one active row at a time
        collision_terms(w.g, world, want_world, margin, [&](real* gg, real viol) {
            for (int a = 0; a < NC; ++a) w.gbuf[a] = sc != nullptr ? gg[a] / sc[a] : gg[a];
            w.vbuf[0] = viol;
            f(1);
        });
        return;
    }
    for (int base = 0; base < N_CON; base += G.size) {
        const int c = base + G.rank;
        real d = margin;
        real* gr = w.gbuf + G.rank * NC;
        const bool live = c < N_CON && constraint(w.g, world, want_world, c, d, gr, margin) && d < margin;
        if (live && sc != nullptr)   // into the solve's column-scaled space, once per row
            for (int a = 0; a < NC; ++a) gr[a] /= sc[a];
        w.vbuf[G.rank] = live ? margin - d : real(0);
        G.sync();
        const int n = N_CON - base < G.size ? N_CON - base : G.size;
        bool any = false;   // most rounds are clear of every margin: skip their row updates
        for (int k = 0; k < n; ++k) any |= w.vbuf[k] > real(0);
        if (any) f(n);
        G.sync();
    }
}

/**
 * Add a collision round's Gauss-Newton terms, A += cw g g^T and b += cwb viol g over its live
 * slots (w.gbuf / w.vbuf), A being NC x NC with leading dimension lda. The thread tier's round is
 * one row: a direct rank-1 update. The cooperative tiers split A's entries over lanes and sum the
 * slots in order.
 */
template <grim::Tier TIER, class W>
__device__ __forceinline__ void gn_accumulate(const Grp<TIER>& G, W& w, int n, real* __restrict__ A, int lda,
                                              real* __restrict__ b, real cw, real cwb)
{
    if constexpr (TIER == grim::Tier::Thread) {
        (void)G; (void)n;
        // The row in registers: A and w.gbuf share the Work struct, and without the copy every
        // A update would reload the row (65% of the kernel's stalls on g1 with collision).
        real g[NC];
        for (int a = 0; a < NC; ++a) g[a] = w.gbuf[a];
        const real v = w.vbuf[0];
        for (int i = 0; i < NC; ++i) {
            const real gi = cw * g[i];
            for (int j = 0; j < NC; ++j) A[i * lda + j] += gi * g[j];
            if (b != nullptr) b[i] += cwb * v * g[i];
        }
    } else {
        for (int i = G.rank; i < NC; i += G.size) {   // a row per lane
            for (int j = 0; j < NC; ++j) {
                real acc = 0;
                for (int s = 0; s < n; ++s)
                    if (w.vbuf[s] > real(0)) acc += w.gbuf[s * NC + i] * w.gbuf[s * NC + j];
                A[i * lda + j] += cw * acc;
            }
            if (b != nullptr) {
                real acc = 0;
                for (int s = 0; s < n; ++s)
                    if (w.vbuf[s] > real(0)) acc += w.vbuf[s] * w.gbuf[s * NC + i];
                b[i] += cwb * acc;
            }
        }
    }
}

/**
 * Jacobi-scale the assembled step system in place, at TIER: sc[a] = sqrt(A[a][a] + 1e-6 max A[b][b])
 * over every term (pose and collision alike), A <- D^-1 A D^-1 + lam I and b <- D^-1 b. Scaling
 * from the pose Jacobian alone leaves a joint that moves collision spheres but no end-effector at
 * sc ~ 0, its collision rows blown up by 1/sc. The floor keeps a roundoff-level column (lam is
 * relative in scaled space) from taking a full step. sc comes back identical on every lane.
 */
template <grim::Tier TIER>
__device__ __forceinline__ void jacobi_scale(const Grp<TIER>& G, real* __restrict__ A, real* __restrict__ b,
                                             real lam, real* __restrict__ sc)
{
    real dmax = 0;
    for (int a = 0; a < NC; ++a) dmax = fmax(dmax, A[a * NC + a]);
    for (int a = 0; a < NC; ++a) sc[a] = sqrt(A[a * NC + a] + real(1e-6) * dmax) + real(1e-8);
    G.sync();   // every lane has read the diagonal before rows are rescaled in place
    for (int i = G.rank; i < NC; i += G.size) {
        for (int j = 0; j < NC; ++j) A[i * NC + j] = A[i * NC + j] / (sc[i] * sc[j]) + (i == j ? lam : real(0));
        b[i] /= sc[i];
    }
    G.sync();
}

/** Factor w.A (N x N) in place for repeated solves (potrs_tier), at TIER; false if not PD. */
template <grim::Tier TIER, int N, class W>
__device__ __forceinline__ bool potrf_tier(const Grp<TIER>& G, W& w)
{
    G.sync();
    if constexpr (TIER == grim::Tier::Thread) {
        int fail = 0;
        glass::thread::potrf<real, (uint32_t)N, /*CHECK=*/true>(w.A, &fail);
        return fail == 0;
    } else {
        if (G.leader()) w.flag = 0;
        G.sync();
        grim::tier_potrf<TIER, real, (uint32_t)N>(w.A, &w.flag);
        G.sync();
        return w.flag == 0;
    }
}

/** Solve w.A x = w.b in place from potrf_tier's factor, at TIER. */
template <grim::Tier TIER, int N, class W>
__device__ __forceinline__ void potrs_tier(const Grp<TIER>& G, W& w)
{
    G.sync();
    if constexpr (TIER == grim::Tier::Thread) glass::thread::potrs<real, (uint32_t)N>(w.A, w.b);
    else grim::tier_potrs<TIER, real, (uint32_t)N>(w.A, w.b);
    G.sync();
}

/** SPD solve of the N x N system w.A x = w.b in place, at TIER (w.b zeroed if not PD). */
template <grim::Tier TIER, int N, class W>
__device__ __forceinline__ void posv_tier(const Grp<TIER>& G, W& w)
{
    G.sync();
    if constexpr (TIER == grim::Tier::Thread) {
        posv<N>(w.A, w.b);
    } else {
        if (G.leader()) w.flag = 0;
        G.sync();
        grim::tier_posv<TIER, real, (uint32_t)N>(w.A, w.b, &w.flag);
    }
    G.sync();
}

}  // namespace grim::ck

// The block tier's lanes per seed are fixed by the build (crocket's block variant is generated
// for them): the launch must ask for the same.
#define GRIM_CK_CHECK_BLOCK_THREADS(name, block_threads)                                      \
    if ((block_threads) != grim::ck::BLOCK_THREADS)                                            \
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,                                     \
                          name ": block_threads does not match this build's (pass it to build)")

// The shared FFI epilogue: the launch's CUDA status, naming block_threads when the block tier's
// kernel needs more registers than a block of that many lanes has.
#define GRIM_CK_LAUNCH_STATUS(name, tier)                                                     \
    const cudaError_t err = cudaGetLastError();                                                \
    if (err == cudaErrorLaunchOutOfResources && (tier) == 2)                                   \
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,                                     \
                          std::string(name) + ": block_threads=" +                             \
                          std::to_string(grim::ck::BLOCK_THREADS) + " lanes per seed need more " \
                          "registers than a block has for this kernel; build with a smaller " \
                          "block_threads");                                                    \
    return err == cudaSuccess ? ffi::Error::Success()                                          \
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err))

// The shared FFI preamble: the launch must match the build's robot and obstacle counts.
#define GRIM_CK_CHECK_WORLD(name, seeds_dim2, ee_dim)                                         \
    if ((seeds_dim2) != grim::robot::n_q || (ee_dim) != grim::robot::n_ee ||                   \
        world_spheres.dimensions()[0] != grim::robot::n_world_spheres ||                       \
        world_capsules.dimensions()[0] != grim::robot::n_world_capsules ||                     \
        world_boxes.dimensions()[0] != grim::robot::n_world_boxes ||                           \
        world_halfspaces.dimensions()[0] != grim::robot::n_world_halfspaces)                   \
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,                                     \
                          name ": launch does not match the robot and obstacle counts of this build")
