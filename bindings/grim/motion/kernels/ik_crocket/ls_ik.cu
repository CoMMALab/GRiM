/**
 * LS-IK on crocket-traced geometry: kernels/ik/ls_ik.cu's Levenberg-Marquardt, step for step
 * (weighted residual, Jacobi column scaling, Gauss-Newton collision rows, fixed joints, the
 * trust-region radii, the five clamped trial steps, accept below curr * (1 - 1e-4), all-time
 * best tracking), in `real` throughout. One seed per thread; same FFI signature as LsIkFfi.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

__global__ void ck_ls_ik_kernel(
    const float* __restrict__ seeds, const float* __restrict__ target_Ts, World world,
    const float* __restrict__ lower, const float* __restrict__ upper, const int* __restrict__ fixed_mask,
    float* __restrict__ out, float* __restrict__ out_err,
    int n_problems, int n_seeds, int max_iter, int enable_collision,
    real wp, real wo, real lambda_init, real eps_pos, real eps_ori, real cw_weight, real margin)
{
    const int gs = blockIdx.x * blockDim.x + threadIdx.x;
    if (gs >= n_problems * n_seeds) return;
    Seed sd;
    real cfg[NC], best_cfg[NC], lo[NC], hi[NC];
    bool fixed[NC];
    load_seed(seeds, target_Ts, gs, gs / n_seeds, cfg, sd);
    for (int a = 0; a < NC; ++a) {
        const int src = grim::robot::solved_idx(a);
        lo[a] = lower[src]; hi[a] = upper[src]; fixed[a] = fixed_mask[src] != 0;
        best_cfg[a] = cfg[a];
    }
    const bool want_world = enable_collision && NWS > 0;
    const bool want_coll = want_world || has_self;

    Geom g;
    geometry(cfg, sd, g);
    real best_err = weighted_err(g.r, wp, wo);
    if (want_coll) best_err += cw_weight * collision_raw(g, world, want_world, margin);
    real lam = lambda_init;

    for (int iter = 0; iter < max_iter; ++iter) {
        geometry(cfg, sd, g);
        if (converged(g.r, eps_pos, eps_ori) &&
            (!want_coll || cw_weight * collision_raw(g, world, want_world, margin) <= real(1e-12)))
            break;

        real fw[M];
        for (int k = 0; k < M; ++k) {
            const real w = (k % 6) < 3 ? wp : wo;
            fw[k] = g.r[k] * w;
            for (int a = 0; a < NC; ++a) g.J[k * NC + a] *= w;
        }
        real curr_err = 0;
        for (int k = 0; k < M; ++k) curr_err += fw[k] * fw[k];
        if (want_coll) curr_err += cw_weight * collision_raw(g, world, want_world, margin);

        real sc[NC];
        for (int a = 0; a < NC; ++a) {
            real s = 0;
            for (int k = 0; k < M; ++k) s += g.J[k * NC + a] * g.J[k * NC + a];
            sc[a] = sqrt(s) + real(1e-8);
        }
        for (int k = 0; k < M; ++k)
            for (int a = 0; a < NC; ++a) g.J[k * NC + a] /= sc[a];

        real A[NC * NC], b[NC];
        for (int i = 0; i < NC; ++i) {
            for (int j = 0; j < NC; ++j) {
                real acc = 0;
                for (int k = 0; k < M; ++k) acc += g.J[k * NC + i] * g.J[k * NC + j];
                A[i * NC + j] = acc + (i == j ? lam : real(0));
            }
            real rb = 0;
            for (int k = 0; k < M; ++k) rb += g.J[k * NC + i] * fw[k];
            b[i] = -rb;
        }
        if (want_coll)
            collision_terms(g, world, want_world, margin, [&](real* gg, real viol) {
                for (int a = 0; a < NC; ++a) gg[a] /= sc[a];
                for (int i = 0; i < NC; ++i) {
                    for (int j = 0; j < NC; ++j) A[i * NC + j] += cw_weight * gg[i] * gg[j];
                    b[i] += cw_weight * viol * gg[i];
                }
            });
        for (int a = 0; a < NC; ++a) {
            if (!fixed[a]) continue;
            for (int j = 0; j < NC; ++j) A[a * NC + j] = A[j * NC + a] = 0;
            A[a * NC + a] = 1;
            b[a] = 0;
        }
        posv<NC>(A, b);

        real delta[NC];
        for (int a = 0; a < NC; ++a) delta[a] = b[a] / sc[a];
        clip_to_radius(delta, trust_radius(g.r));

        const real alphas[5] = {1, 0.5, 0.25, 0.1, 0.025};
        real best_a_err = real(1e30);
        int best_a = 0;
        for (int ai = 0; ai < 5; ++ai) {
            real trial[NC];
            for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[ai] * delta[a], lo[a], hi[a]);
            Geom gt;
            geometry_value(trial, sd, gt);
            real e = weighted_err(gt.r, wp, wo);
            if (want_coll) e += cw_weight * collision_raw(gt, world, want_world, margin);
            if (e < best_a_err) { best_a_err = e; best_a = ai; }
        }
        real trial[NC];
        for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[best_a] * delta[a], lo[a], hi[a]);
        if (best_a_err < curr_err * real(1 - 1e-4)) {
            for (int a = 0; a < NC; ++a) cfg[a] = trial[a];
            lam = fmax(lam * real(0.5), real(1e-10));
        } else {
            lam = fmin(lam * real(3), real(1e6));
        }
        if (best_a_err < best_err) {
            best_err = best_a_err;
            for (int a = 0; a < NC; ++a) best_cfg[a] = trial[a];
        }
    }
    store(best_cfg, sd, out, gs);
    out_err[gs] = (float)best_err;
}

static ffi::Error CkLsIkImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds, ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower, ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter, float pos_weight, float ori_weight, float lambda_init, float eps_pos,
    float eps_ori, int64_t enable_collision, float collision_weight, float collision_margin,
    int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    (void)tier; (void)block_threads;
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("ls_ik (crocket)", seeds.dimensions()[2], target_Ts.dimensions()[1]);
    const int n = n_problems * n_seeds;
    if (n == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    ck_ls_ik_kernel<<<(n + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        seeds.typed_data(), target_Ts.typed_data(), world, lower.typed_data(), upper.typed_data(),
        fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(), n_problems, n_seeds,
        static_cast<int>(max_iter), static_cast<int>(enable_collision), pos_weight, ori_weight,
        lambda_init, eps_pos, eps_ori, collision_weight, collision_margin);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LsIkFfi, CkLsIkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // fixed_mask
        .Attr<int64_t>("max_iter")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("lambda_init")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());
