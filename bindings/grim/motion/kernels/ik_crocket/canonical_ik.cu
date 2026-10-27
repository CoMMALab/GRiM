/**
 * Canonical IK on crocket-traced geometry: the hand-written canonical_ik.cu (tag pre-crocket-ik)'s damped Gauss-Newton on
 * argmin 1/2 |q - q_ref|^2 s.t. r(q) = 0, in `real`:
 *
 *     dq = -J^+ r + step * (I - J^+ J)(q_ref - q),   J^+ = J^T (J J^T + damping I)^-1,
 *
 * with the same clamped-pivot Cholesky, and the same clearance guard (the null-space part is
 * halved up to six times so the step never lowers the minimum clearance the input had, and the
 * last configuration at or above that floor is returned). One problem per thread, warp or block
 * (`tier`); the FFI signature is CanonicalIkFfi's plus `tier` / `block_threads`.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

__device__ __forceinline__ void canon_chol(real* __restrict__ A)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j <= i; ++j) {
            real s = A[i * M + j];
            for (int k = 0; k < j; ++k) s -= A[i * M + k] * A[j * M + k];
            A[i * M + j] = i == j ? sqrt(fmax(s, real(1e-12))) : s / A[j * M + j];
        }
}

__device__ __forceinline__ void canon_solve(const real* __restrict__ L, real* __restrict__ x)
{
    for (int i = 0; i < M; ++i) {
        real s = x[i];
        for (int k = 0; k < i; ++k) s -= L[i * M + k] * x[k];
        x[i] = s / L[i * M + i];
    }
    for (int i = M - 1; i >= 0; --i) {
        real s = x[i];
        for (int k = i + 1; k < M; ++k) s -= L[k * M + i] * x[k];
        x[i] = s / L[i * M + i];
    }
}

template <grim::Tier TIER>
using CanonWork = Work<TIER, M, false, /*WITH_GV=*/true>;

template <grim::Tier TIER>
__global__ void ck_canonical_ik_kernel(
    int wpb, const float* __restrict__ cfgs, const float* __restrict__ cfg_refs, const float* __restrict__ target_Ts,
    World world, int n_problems, int max_iters, real step, real tol, real damping, real margin,
    float* __restrict__ out_q, int* __restrict__ out_iters)
{
    const int p = seed_index<TIER>(wpb);
    if (p >= n_problems) return;
    const Grp<TIER> G;
    CK_WORK(TIER, CanonWork<TIER>, w);
    Seed sd;
    real q[NC], qref[NC];
    for (int a = 0; a < NC; ++a) {
        q[a] = cfgs[(size_t)p * NC + a];
        qref[a] = cfg_refs[(size_t)p * NC + a];
    }
    for (int i = 0; i < 7 * NE; ++i) sd.target[i] = target_Ts[(size_t)p * 7 * NE + i];
    load_rot(sd);

    constexpr bool guard = has_self || has_world_geom;
    // The clearance of w.gv, computed on the leader and shared.
    auto clearance = [&]() {
        G.sync();
        if (G.leader()) w.err[CK_MAX_EVAL - 1] = min_clearance(w.gv, world, margin);
        G.sync();
        const real c = w.err[CK_MAX_EVAL - 1];
        G.sync();
        return c;
    };
    real q_feas[NC], c_floor = 0;
    bool have_feas = false;
    if constexpr (guard) {
        geometry_value(G, q, sd, w);
        c_floor = clearance();
        for (int a = 0; a < NC; ++a) q_feas[a] = q[a];
        have_feas = true;
    }

    int used = max_iters;
    for (int it = 0; it < max_iters; ++it) {
        geometry(G, q, sd, w);
        // G = J J^T + damping I, a row per lane; the clamped-pivot factor on the leader.
        for (int i = G.rank; i < M; i += G.size) {
            real ji[NC];
            for (int a = 0; a < NC; ++a) ji[a] = w.g.J[i * NC + a];
            for (int j = 0; j < M; ++j) {
                real acc = i == j ? damping : real(0);
                for (int a = 0; a < NC; ++a) acc += ji[a] * w.g.J[j * NC + a];
                w.A[i * M + j] = acc;
            }
        }
        G.sync();
        if (G.leader()) canon_chol(w.A);
        G.sync();
        real y[M], Jw[M], wv[NC];
        for (int i = 0; i < M; ++i) y[i] = w.g.r[i];
        canon_solve(w.A, y);
        for (int a = 0; a < NC; ++a) wv[a] = qref[a] - q[a];
        for (int i = 0; i < M; ++i) {
            real acc = 0;
            for (int a = 0; a < NC; ++a) acc += w.g.J[i * NC + a] * wv[a];
            Jw[i] = acc;
        }
        canon_solve(w.A, Jw);
        real dp[NC], dn[NC];
        for (int a = 0; a < NC; ++a) {
            real x = 0, z = 0;
            for (int i = 0; i < M; ++i) { x -= w.g.J[i * NC + a] * y[i]; z -= w.g.J[i * NC + a] * Jw[i]; }
            dp[a] = x;
            dn[a] = z + wv[a];
        }
        real scale = 1;
        if constexpr (guard) {
            // The six halvings are independent candidates: one per lane, the first that keeps the
            // clearance floor wins (none: the null-space step is dropped).
            auto clear_at = [&](int t) {
                real q_try[NC];
                const real sc_t = real(1) / real(1 << t);
                for (int a = 0; a < NC; ++a) q_try[a] = q[a] + dp[a] + step * sc_t * dn[a];
                GeomV gt;
                grim::ck::geometry_value(q_try, sd, gt);
                return min_clearance(gt, world, margin);
            };
            int pick = 6;
            if constexpr (TIER == grim::Tier::Thread) {
                for (int t = 0; t < 6; ++t) if (clear_at(t) >= c_floor) { pick = t; break; }
            } else {
                for (int t = G.rank; t < 6; t += G.size) w.err[t] = clear_at(t);
                G.sync();
                for (int t = 0; t < 6; ++t) if (w.err[t] >= c_floor) { pick = t; break; }
                G.sync();
            }
            scale = pick < 6 ? real(1) / real(1 << pick) : real(0);
        }
        real step_sq = 0;
        for (int a = 0; a < NC; ++a) {
            const real d = dp[a] + step * scale * dn[a];
            q[a] += d;
            step_sq += d * d;
        }
        if constexpr (guard) {
            geometry_value(G, q, sd, w);
            if (clearance() >= c_floor) {
                for (int a = 0; a < NC; ++a) q_feas[a] = q[a];
                have_feas = true;
            }
        }
        if (step_sq < tol * tol) { used = it + 1; break; }
    }
    if constexpr (guard) {
        if (have_feas) {
            geometry_value(G, q, sd, w);
            if (clearance() < c_floor)
                for (int a = 0; a < NC; ++a) q[a] = q_feas[a];
        }
    }
    if (G.leader()) {
        for (int a = 0; a < NC; ++a) out_q[(size_t)p * NC + a] = (float)q[a];
        out_iters[p] = used;
    }
}

static ffi::Error CkCanonicalIkImpl(
    cudaStream_t stream,
    GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> cfgs, ffi::Buffer<ffi::DataType::F32> cfg_refs,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    int64_t max_iters, float step, float tol, float damping, float collision_margin,
    int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_q, ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_iters)
{
    GRIM_ROT_UPLOAD(stream);
    GRIM_CK_CHECK_BLOCK_THREADS("canonical_ik (crocket)", block_threads);
    static_assert(grim::robot::n_frozen == 0, "canonical IK solves every joint");
    const int n_problems = static_cast<int>(cfgs.dimensions()[0]);
    GRIM_CK_CHECK_WORLD("canonical_ik (crocket)", cfgs.dimensions()[1],
                        target_Ts.dimensions()[target_Ts.dimensions().size() - 2]);
    if (n_problems == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    CK_LAUNCH(ck_canonical_ik_kernel, CanonWork, tier, n_problems, stream,
        cfgs.typed_data(), cfg_refs.typed_data(), target_Ts.typed_data(), world, n_problems,
        static_cast<int>(max_iters), step, tol, damping, collision_margin,
        out_q->typed_data(), out_iters->typed_data());
    GRIM_CK_LAUNCH_STATUS("canonical_ik (crocket)", tier);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CanonicalIkFfi, CkCanonicalIkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        GRIM_ROT_BIND
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfgs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfg_refs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Attr<int64_t>("max_iters")
        .Attr<float>("step")
        .Attr<float>("tol")
        .Attr<float>("damping")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>());
