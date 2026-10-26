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
constexpr int THREADS = 32;

/** Everything one configuration's traced geometry call produces. */
struct Geom {
    real r[M];
    real J[M * NC];
    real cw[3 * NWSB];
    real cs[3 * NSSB];
    real tw[6 * NJ];
};

/** Per-seed state: the solved variables' configuration and the frozen joints carried along. */
struct Seed {
    real frz[NF];
    real target[7 * NE];
};

__device__ __forceinline__ void load_seed(const float* __restrict__ seeds, const float* __restrict__ targets,
                                          int gs, int problem, real* __restrict__ cfg, Seed& sd)
{
    const float* row = seeds + (size_t)gs * grim::robot::n_q;
    for (int i = 0; i < NC; ++i) cfg[i] = row[grim::robot::solved_idx(i)];
    for (int k = 0; k < grim::robot::n_frozen; ++k) sd.frz[k] = row[grim::robot::frozen_idx(k)];
    for (int i = 0; i < 7 * NE; ++i) sd.target[i] = targets[(size_t)problem * 7 * NE + i];
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
    tr::ik_geometry_thread<real>(q, sd.target, g.r, g.J, g.cw, g.cs, g.tw, 1);
}

/** Residual and sphere centres at `cfg` (no Jacobian): a trial's merit. */
__device__ __forceinline__ void geometry_value(const real* __restrict__ cfg, const Seed& sd, Geom& g)
{
    real q[NQ];
    gather<real>(cfg, sd.frz, q);
    tr::ik_geometry_value_thread<real>(q, sd.target, g.r, g.cw, g.cs, 1);
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
            const real d = d_sph_sph(cs + 3 * a, real(kSsRad(a)), cs + 3 * b, real(kSsRad(b)));
            if (d < best) { best = d; ia = a; ib = b; }
        }
    return best;
}

/** Unweighted collision penalty: sum over active constraints of (d - margin)^2. */
__device__ __forceinline__ real collision_raw(const Geom& g, const World& w, bool want_world, real margin)
{
    real pen = 0;
    if constexpr (has_world_geom) {
        if (want_world)
            for (int i = 0; i < NWS; ++i)
                for_each_prim(w, [&](int kind, const float* o) {
                    const real d = d_prim(g.cw + 3 * i, real(kWsRad(i)), kind, o);
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
__device__ __forceinline__ real min_clearance(const Geom& g, const World& w, real margin)
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
                best = fmin(best, d_prim(g.cw + 3 * i, real(kWsRad(i)), kind, o) - margin);
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
        if (!(WORLD ? kWsChain(NJ * s + j) : kSsChain(NJ * s + j)) || kJointCol(j) < 0) continue;
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
            const real rr = real(kWsRad(i));
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

}  // namespace grim::ck

// The shared FFI preamble: the launch must match the build's robot and obstacle counts.
#define GRIM_CK_CHECK_WORLD(name, seeds_dim2, ee_dim)                                         \
    if ((seeds_dim2) != grim::robot::n_q || (ee_dim) != grim::robot::n_ee ||                   \
        world_spheres.dimensions()[0] != grim::robot::n_world_spheres ||                       \
        world_capsules.dimensions()[0] != grim::robot::n_world_capsules ||                     \
        world_boxes.dimensions()[0] != grim::robot::n_world_boxes ||                           \
        world_halfspaces.dimensions()[0] != grim::robot::n_world_halfspaces)                   \
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,                                     \
                          name ": launch does not match the robot and obstacle counts of this build")
