/**
 * HJCD-IK on crocket-traced geometry: kernels/ik/hjcd_ik.cu's two phases, in `real` throughout.
 *
 *   Coarse: greedy coordinate descent, the joint with the largest predicted reduction of the
 *           adaptively weighted error per step (orientation off while position >= 1 mm).
 *   LM:     adaptive weights, Jacobi scaling, a soft joint-limit prior, the trust-region radii,
 *           the five trials with an early exit on sufficient decrease, stall kicks from the
 *           caller's noise, all-time best tracking, and the opt-in cross-seed early stop.
 *
 * Collision enters the merit only (no step terms), as in the hand-written kernel. Same FFI
 * signatures as HjcdIkCoarseFfi / HjcdIkLmFfi.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

/** w[0:3] = 1, w[3:6] = clamp(|r_pos| / |r_ori|, 0.05, 1). */
__device__ __forceinline__ void adaptive_weights(const real* r, real* w)
{
    const real s = clamp((nrm3(r) + real(1e-8)) / (nrm3(r + 3) + real(1e-8)), real(0.05), real(1));
    w[0] = w[1] = w[2] = 1;
    w[3] = w[4] = w[5] = s;
}

__global__ void ck_hjcd_coarse_kernel(
    const float* __restrict__ seeds, const float* __restrict__ target_Ts, World world,
    const float* __restrict__ lower, const float* __restrict__ upper, const int* __restrict__ fixed_mask,
    float* __restrict__ out, float* __restrict__ out_err,
    int n_problems, int n_seeds, int k_max, int enable_collision, real cw_weight, real margin)
{
    const int gs = blockIdx.x * blockDim.x + threadIdx.x;
    if (gs >= n_problems * n_seeds) return;
    Seed sd;
    real cfg[NC], lo[NC], hi[NC];
    bool fixed[NC];
    load_seed(seeds, target_Ts, gs, gs / n_seeds, cfg, sd);
    for (int a = 0; a < NC; ++a) {
        const int src = grim::robot::solved_idx(a);
        lo[a] = lower[src]; hi[a] = upper[src]; fixed[a] = fixed_mask[src] != 0;
    }
    Geom g;
    for (int iter = 0; iter < k_max; ++iter) {
        geometry(cfg, sd, g);
        real w[M];
        for (int e = 0; e < NE; ++e) {
            adaptive_weights(g.r + 6 * e, w + 6 * e);
            if (nrm3(g.r + 6 * e) >= real(1e-3)) w[6 * e + 3] = w[6 * e + 4] = w[6 * e + 5] = 0;
        }
        int best = -1;
        real best_impr = -1, best_dot = 0, best_sq = 1;
        for (int a = 0; a < NC; ++a) {
            real dot = 0, sq = 0;
            for (int k = 0; k < M; ++k) {
                const real jw = g.J[k * NC + a] * w[k];
                dot += jw * g.r[k] * w[k];
                sq += jw * jw;
            }
            sq += real(1e-8);
            if (fixed[a]) continue;
            const real impr = dot * dot / sq;
            if (impr > best_impr) { best_impr = impr; best = a; best_dot = dot; best_sq = sq; }
        }
        if (best < 0) break;
        cfg[best] = clamp(cfg[best] - best_dot / best_sq, lo[best], hi[best]);
    }
    geometry_value(cfg, sd, g);
    real err = 0;
    for (int k = 0; k < M; ++k) err += g.r[k] * g.r[k];
    const bool want_world = enable_collision && NWS > 0;
    if (want_world || has_self) err += cw_weight * collision_raw(g, world, want_world, margin);
    out_err[gs] = (float)err;
    store(cfg, sd, out, gs);
}

__global__ void ck_hjcd_lm_kernel(
    const float* __restrict__ seeds, const float* __restrict__ noise, const float* __restrict__ target_Ts,
    World world, const float* __restrict__ lower, const float* __restrict__ upper,
    const int* __restrict__ fixed_mask, float* __restrict__ out, float* __restrict__ out_err,
    int* __restrict__ stop_flag, int n_problems, int n_seeds, int max_iter, int early_stop,
    real lambda_init, real limit_prior_weight, real kick_scale, real eps_pos, real eps_ori,
    int stall_patience, int enable_collision, real cw_weight, real margin)
{
    const int gs = blockIdx.x * blockDim.x + threadIdx.x;
    if (gs >= n_problems * n_seeds) return;
    const int p = gs / n_seeds;
    Seed sd;
    real cfg[NC], best_cfg[NC], lo[NC], hi[NC], mid[NC], half[NC];
    bool fixed[NC];
    load_seed(seeds, target_Ts, gs, p, cfg, sd);
    for (int a = 0; a < NC; ++a) {
        const int src = grim::robot::solved_idx(a);
        lo[a] = lower[src]; hi[a] = upper[src]; fixed[a] = fixed_mask[src] != 0;
        best_cfg[a] = cfg[a];
        const bool bounded = isfinite(lo[a]) && isfinite(hi[a]);
        mid[a] = bounded ? (lo[a] + hi[a]) * real(0.5) : real(0);
        half[a] = bounded ? (hi[a] - lo[a]) * real(0.5) + real(1e-8) : real(INFINITY);
    }
    const bool want_world = enable_collision && NWS > 0;
    const bool want_coll = want_world || has_self;
    auto pen = [&](const Geom& gg) { return want_coll ? cw_weight * collision_raw(gg, world, want_world, margin) : real(0); };

    Geom g;
    geometry(cfg, sd, g);
    real best_err = pen(g);
    for (int k = 0; k < M; ++k) best_err += g.r[k] * g.r[k];
    real lam = lambda_init;
    int stall = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (early_stop && *(volatile int*)(stop_flag + p) != 0) break;
        geometry(cfg, sd, g);
        const real cpen = pen(g);
        real curr_err = cpen;
        for (int k = 0; k < M; ++k) curr_err += g.r[k] * g.r[k];
        if (converged(g.r, eps_pos, eps_ori) && (!want_coll || cpen <= real(1e-12))) {
            if (early_stop) { atomicExch(stop_flag + p, 1); __threadfence(); }
            break;
        }

        real w[M], fw[M];
        for (int e = 0; e < NE; ++e) adaptive_weights(g.r + 6 * e, w + 6 * e);
        for (int k = 0; k < M; ++k) {
            fw[k] = g.r[k] * w[k];
            for (int a = 0; a < NC; ++a) g.J[k * NC + a] *= w[k];
        }
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
                A[i * NC + j] = acc;
            }
            real rb = 0;
            for (int k = 0; k < M; ++k) rb += g.J[k * NC + i] * fw[k];
            const real d_raw = limit_prior_weight / (half[i] * half[i]);
            A[i * NC + i] += lam + d_raw / (sc[i] * sc[i]);
            b[i] = -(rb + d_raw * (cfg[i] - mid[i]) / sc[i]);
        }
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
        const real suff = curr_err * real(1 - 1e-4);
        real best_a_err = real(1e30);
        int best_a = 0;
        for (int ai = 0; ai < 5; ++ai) {
            real trial[NC];
            for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[ai] * delta[a], lo[a], hi[a]);
            Geom gt;
            geometry_value(trial, sd, gt);
            real e = pen(gt);
            for (int k = 0; k < M; ++k) e += gt.r[k] * gt.r[k];
            if (e < best_a_err) {
                best_a_err = e; best_a = ai;
                if (e < suff) break;
            }
        }
        real trial[NC];
        for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[best_a] * delta[a], lo[a], hi[a]);
        if (best_a_err < suff) {
            for (int a = 0; a < NC; ++a) cfg[a] = trial[a];
            lam = fmax(lam * real(0.5), real(1e-10));
            stall = 0;
        } else {
            lam = fmin(lam * real(3), real(1e6));
            ++stall;
        }
        if (best_a_err < best_err) {
            best_err = best_a_err;
            for (int a = 0; a < NC; ++a) best_cfg[a] = trial[a];
        }
        if (stall >= stall_patience) {
            const float* kick = noise + ((size_t)gs * max_iter + iter) * NC;
            for (int a = 0; a < NC; ++a)
                if (!fixed[a]) cfg[a] = clamp(cfg[a] + real(kick[a]) * kick_scale, lo[a], hi[a]);
            lam = lambda_init;
            stall = 0;
        }
    }
    store(best_cfg, sd, out, gs);
    out_err[gs] = (float)best_err;
}

static ffi::Error CkHjcdCoarseImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds, ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower, ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t k_max, int64_t enable_collision, float collision_weight, float collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("hjcd_ik (crocket)", seeds.dimensions()[2],
                        target_Ts.dimensions()[target_Ts.dimensions().size() - 2]);
    const int n = n_problems * n_seeds;
    if (n == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    ck_hjcd_coarse_kernel<<<(n + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        seeds.typed_data(), target_Ts.typed_data(), world, lower.typed_data(), upper.typed_data(),
        fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(), n_problems, n_seeds,
        static_cast<int>(k_max), static_cast<int>(enable_collision), collision_weight, collision_margin);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}

static ffi::Error CkHjcdLmImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds, ffi::Buffer<ffi::DataType::F32> noise,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower, ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter, int64_t stall_patience, int64_t early_stop, float lambda_init,
    float limit_prior_weight, float kick_scale, float eps_pos, float eps_ori,
    int64_t enable_collision, float collision_weight, float collision_margin,
    int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> stop_flag)
{
    (void)tier; (void)block_threads;
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("hjcd_ik (crocket)", seeds.dimensions()[2],
                        target_Ts.dimensions()[target_Ts.dimensions().size() - 2]);
    if (n_problems == 0 || n_seeds == 0) return ffi::Error::Success();
    cudaMemsetAsync(stop_flag->typed_data(), 0, n_problems * sizeof(int), stream);
    const int n = n_problems * n_seeds;
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    ck_hjcd_lm_kernel<<<(n + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        seeds.typed_data(), noise.typed_data(), target_Ts.typed_data(), world, lower.typed_data(),
        upper.typed_data(), fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(),
        stop_flag->typed_data(), n_problems, n_seeds, static_cast<int>(max_iter),
        static_cast<int>(early_stop), lambda_init, limit_prior_weight, kick_scale, eps_pos, eps_ori,
        static_cast<int>(stall_patience), static_cast<int>(enable_collision), collision_weight,
        collision_margin);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HjcdIkCoarseFfi, CkHjcdCoarseImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("k_max")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HjcdIkLmFfi, CkHjcdLmImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // noise
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("max_iter")
        .Attr<int64_t>("stall_patience")
        .Attr<int64_t>("early_stop")
        .Attr<float>("lambda_init")
        .Attr<float>("limit_prior_weight")
        .Attr<float>("kick_scale")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>());
