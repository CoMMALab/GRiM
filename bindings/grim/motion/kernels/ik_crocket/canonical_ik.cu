/**
 * Canonical IK on crocket-traced geometry: kernels/ik/canonical_ik.cu's damped Gauss-Newton on
 * argmin 1/2 |q - q_ref|^2 s.t. r(q) = 0, in `real`:
 *
 *     dq = -J^+ r + step * (I - J^+ J)(q_ref - q),   J^+ = J^T (J J^T + damping I)^-1,
 *
 * with the same clamped-pivot Cholesky, and the same clearance guard (the null-space part is
 * halved up to six times so the step never lowers the minimum clearance the input had, and the
 * last configuration at or above that floor is returned). One problem per thread; same FFI
 * signature as CanonicalIkFfi.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

__device__ __forceinline__ void canon_chol(real* A)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j <= i; ++j) {
            real s = A[i * M + j];
            for (int k = 0; k < j; ++k) s -= A[i * M + k] * A[j * M + k];
            A[i * M + j] = i == j ? sqrt(fmax(s, real(1e-12))) : s / A[j * M + j];
        }
}

__device__ __forceinline__ void canon_solve(const real* L, real* x)
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

__global__ void ck_canonical_ik_kernel(
    const float* __restrict__ cfgs, const float* __restrict__ cfg_refs, const float* __restrict__ target_Ts,
    World world, int n_problems, int max_iters, real step, real tol, real damping, real margin,
    float* __restrict__ out_q, int* __restrict__ out_iters)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_problems) return;
    Seed sd;
    real q[NC], qref[NC];
    for (int a = 0; a < NC; ++a) {
        q[a] = cfgs[(size_t)p * NC + a];
        qref[a] = cfg_refs[(size_t)p * NC + a];
    }
    for (int i = 0; i < 7 * NE; ++i) sd.target[i] = target_Ts[(size_t)p * 7 * NE + i];

    constexpr bool guard = has_self || has_world_geom;
    Geom g;
    real q_feas[NC], c_floor = 0;
    bool have_feas = false;
    if constexpr (guard) {
        geometry_value(q, sd, g);
        c_floor = min_clearance(g, world, margin);
        for (int a = 0; a < NC; ++a) q_feas[a] = q[a];
        have_feas = true;
    }

    int used = max_iters;
    for (int it = 0; it < max_iters; ++it) {
        geometry(q, sd, g);
        real G[M * M], y[M], Jw[M], w[NC];
        for (int i = 0; i < M; ++i)
            for (int j = 0; j <= i; ++j) {
                real s = 0;
                for (int a = 0; a < NC; ++a) s += g.J[i * NC + a] * g.J[j * NC + a];
                if (i == j) s += damping;
                G[i * M + j] = G[j * M + i] = s;
            }
        canon_chol(G);
        for (int i = 0; i < M; ++i) y[i] = g.r[i];
        canon_solve(G, y);
        for (int a = 0; a < NC; ++a) w[a] = qref[a] - q[a];
        for (int i = 0; i < M; ++i) {
            real s = 0;
            for (int a = 0; a < NC; ++a) s += g.J[i * NC + a] * w[a];
            Jw[i] = s;
        }
        canon_solve(G, Jw);
        real dp[NC], dn[NC];
        for (int a = 0; a < NC; ++a) {
            real x = 0, z = 0;
            for (int i = 0; i < M; ++i) { x -= g.J[i * NC + a] * y[i]; z -= g.J[i * NC + a] * Jw[i]; }
            dp[a] = x;
            dn[a] = z + w[a];
        }
        real scale = 1;
        if constexpr (guard) {
            for (int t = 0; t < 6; ++t) {
                real q_try[NC];
                for (int a = 0; a < NC; ++a) q_try[a] = q[a] + dp[a] + step * scale * dn[a];
                Geom gt;
                geometry_value(q_try, sd, gt);
                if (min_clearance(gt, world, margin) >= c_floor) break;
                scale *= real(0.5);
                if (t == 5) scale = 0;
            }
        }
        real step_sq = 0;
        for (int a = 0; a < NC; ++a) {
            const real d = dp[a] + step * scale * dn[a];
            q[a] += d;
            step_sq += d * d;
        }
        if constexpr (guard) {
            Geom gt;
            geometry_value(q, sd, gt);
            if (min_clearance(gt, world, margin) >= c_floor) {
                for (int a = 0; a < NC; ++a) q_feas[a] = q[a];
                have_feas = true;
            }
        }
        if (step_sq < tol * tol) { used = it + 1; break; }
    }
    if constexpr (guard) {
        if (have_feas) {
            geometry_value(q, sd, g);
            if (min_clearance(g, world, margin) < c_floor)
                for (int a = 0; a < NC; ++a) q[a] = q_feas[a];
        }
    }
    for (int a = 0; a < NC; ++a) out_q[(size_t)p * NC + a] = (float)q[a];
    out_iters[p] = used;
}

static ffi::Error CkCanonicalIkImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfgs, ffi::Buffer<ffi::DataType::F32> cfg_refs,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    int64_t max_iters, float step, float tol, float damping, float collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_q, ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_iters)
{
    static_assert(grim::robot::n_frozen == 0, "canonical IK solves every joint");
    const int n_problems = static_cast<int>(cfgs.dimensions()[0]);
    GRIM_CK_CHECK_WORLD("canonical_ik (crocket)", cfgs.dimensions()[1],
                        target_Ts.dimensions()[target_Ts.dimensions().size() - 2]);
    if (n_problems == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    ck_canonical_ik_kernel<<<(n_problems + 63) / 64, 64, 0, stream>>>(
        cfgs.typed_data(), cfg_refs.typed_data(), target_Ts.typed_data(), world, n_problems,
        static_cast<int>(max_iters), step, tol, damping, collision_margin,
        out_q->typed_data(), out_iters->typed_data());
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CanonicalIkFfi, CkCanonicalIkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>());
