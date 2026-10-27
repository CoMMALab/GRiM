/**
 * HJCD-IK on crocket-traced geometry: kernels/ik/hjcd_ik.cu's two phases, in `real` throughout.
 *
 *   Coarse: greedy coordinate descent, the joint with the largest predicted reduction of the
 *           adaptively weighted error per step (orientation off while position >= 1 mm).
 *   LM:     adaptive weights, Jacobi scaling, a soft joint-limit prior, the trust-region radii,
 *           the five trials with an early exit on sufficient decrease, stall kicks from the
 *           caller's noise, all-time best tracking, and the opt-in cross-seed early stop.
 *
 * Collision enters the merit only (no step terms), as in the hand-written kernel. One seed per
 * thread, warp or block (`tier`); FFI signatures as HjcdIkCoarseFfi (plus `tier` /
 * `block_threads`) and HjcdIkLmFfi.
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

template <grim::Tier TIER>
using HjcdCoarseWork = Work<TIER, 1, false, /*WITH_GV=*/true>;

template <grim::Tier TIER>
__global__ void ck_hjcd_coarse_kernel(
    int wpb, const float* __restrict__ seeds, const float* __restrict__ target_Ts, World world,
    const float* __restrict__ lower, const float* __restrict__ upper, const int* __restrict__ fixed_mask,
    float* __restrict__ out, float* __restrict__ out_err,
    int n_problems, int n_seeds, int k_max, int enable_collision, real cw_weight, real margin)
{
    const int gs = seed_index<TIER>(wpb);
    if (gs >= n_problems * n_seeds) return;
    const Grp<TIER> G;
    CK_WORK(TIER, HjcdCoarseWork<TIER>, w);
    Seed sd;
    real cfg[NC], lo[NC], hi[NC];
    bool fixed[NC];
    load_seed(seeds, target_Ts, gs, gs / n_seeds, cfg, sd);
    for (int a = 0; a < NC; ++a) {
        const int src = grim::robot::solved_idx(a);
        lo[a] = lower[src]; hi[a] = upper[src]; fixed[a] = fixed_mask[src] != 0;
    }
    for (int iter = 0; iter < k_max; ++iter) {
        geometry(G, cfg, sd, w);
        real wt[M];
        for (int e = 0; e < NE; ++e) {
            adaptive_weights(w.g.r + 6 * e, wt + 6 * e);
            if (nrm3(w.g.r + 6 * e) >= real(1e-3)) wt[6 * e + 3] = wt[6 * e + 4] = wt[6 * e + 5] = 0;
        }
        // The joint with the largest predicted reduction (lowest index on ties): every lane scans
        // all NC columns of the shared Jacobian, so they agree.
        int best = -1;
        real best_impr = -1, best_dot = 0, best_sq = 1;
        for (int a = 0; a < NC; ++a) {
            real dot = 0, sq = 0;
            for (int k = 0; k < M; ++k) {
                const real jw = w.g.J[k * NC + a] * wt[k];
                dot += jw * w.g.r[k] * wt[k];
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
    geometry_value(G, cfg, sd, w);
    real err = 0;
    for (int k = 0; k < M; ++k) err += w.gv.r[k] * w.gv.r[k];
    const bool want_world = enable_collision && NWS > 0;
    if (want_world || has_self) err += cw_weight * collision_raw_coop(G, w, w.gv, world, want_world, margin);
    if (G.leader()) {
        out_err[gs] = (float)err;
        store(cfg, sd, out, gs);
    }
}

template <grim::Tier TIER>
using HjcdLmWork = Work<TIER>;

template <grim::Tier TIER>
__global__ void ck_hjcd_lm_kernel(
    int wpb, const float* __restrict__ seeds, const float* __restrict__ noise, const float* __restrict__ target_Ts,
    World world, const float* __restrict__ lower, const float* __restrict__ upper,
    const int* __restrict__ fixed_mask, float* __restrict__ out, float* __restrict__ out_err,
    int* __restrict__ stop_flag, int n_problems, int n_seeds, int max_iter, int early_stop,
    real lambda_init, real limit_prior_weight, real kick_scale, real eps_pos, real eps_ori,
    int stall_patience, int enable_collision, real cw_weight, real margin)
{
    const int gs = seed_index<TIER>(wpb);
    if (gs >= n_problems * n_seeds) return;
    const Grp<TIER> G;
    CK_WORK(TIER, HjcdLmWork<TIER>, w);
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

    geometry(G, cfg, sd, w);
    real best_err = want_coll ? cw_weight * collision_raw_coop(G, w, w.g, world, want_world, margin) : real(0);
    for (int k = 0; k < M; ++k) best_err += w.g.r[k] * w.g.r[k];
    real lam = lambda_init;
    int stall = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (early_stop) {   // read once per group so every lane takes the same branch
            G.sync();
            if (G.leader()) w.flag = *(volatile int*)(stop_flag + p);
            G.sync();
            const bool stop = w.flag != 0;
            G.sync();
            if (stop) break;
        }
        geometry(G, cfg, sd, w);
        const real cpen = want_coll ? cw_weight * collision_raw_coop(G, w, w.g, world, want_world, margin) : real(0);
        real curr_err = cpen;
        for (int k = 0; k < M; ++k) curr_err += w.g.r[k] * w.g.r[k];
        if (converged(w.g.r, eps_pos, eps_ori) && (!want_coll || cpen <= real(1e-12))) {
            if (early_stop && G.leader()) { atomicExch(stop_flag + p, 1); __threadfence(); }
            break;
        }

        real wt[M], fw[M];
        for (int e = 0; e < NE; ++e) adaptive_weights(w.g.r + 6 * e, wt + 6 * e);
        for (int k = 0; k < M; ++k) fw[k] = w.g.r[k] * wt[k];
        real sc[NC];
        for (int a = 0; a < NC; ++a) {
            real s2 = 0;
            for (int k = 0; k < M; ++k) { const real v = w.g.J[k * NC + a] * wt[k]; s2 += v * v; }
            sc[a] = sqrt(s2) + real(1e-8);
        }
        for (int k = G.rank; k < M; k += G.size)
            for (int a = 0; a < NC; ++a) w.g.J[k * NC + a] *= wt[k] / sc[a];
        G.sync();
        for (int i = G.rank; i < NC; i += G.size) {
            real ji[M];
            for (int k = 0; k < M; ++k) ji[k] = w.g.J[k * NC + i];
            for (int j = 0; j < NC; ++j) {
                real acc = 0;
                for (int k = 0; k < M; ++k) acc += ji[k] * w.g.J[k * NC + j];
                w.A[i * NC + j] = acc;
            }
            real rb = 0;
            for (int k = 0; k < M; ++k) rb += ji[k] * fw[k];
            const real d_raw = limit_prior_weight / (half[i] * half[i]);
            w.A[i * NC + i] += lam + d_raw / (sc[i] * sc[i]);
            w.b[i] = -(rb + d_raw * (cfg[i] - mid[i]) / sc[i]);
        }
        G.sync();
        for (int a = G.rank; a < NC; a += G.size) {
            if (!fixed[a]) continue;
            for (int j = 0; j < NC; ++j) w.A[a * NC + j] = w.A[j * NC + a] = 0;
            w.A[a * NC + a] = 1;
            w.b[a] = 0;
        }
        posv_tier<TIER, NC>(G, w);
        real delta[NC];
        for (int a = 0; a < NC; ++a) delta[a] = w.b[a] / sc[a];
        clip_to_radius(delta, trust_radius(w.g.r));

        // Trials: early exit on sufficient decrease on the one lane of the thread tier; all five in
        // parallel elsewhere, then the same first-sufficient rule applied in order.
        const real alphas[5] = {1, 0.5, 0.25, 0.1, 0.025};
        const real suff = curr_err * real(1 - 1e-4);
        auto eval = [&](int ai) {
            real trial[NC];
            for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[ai] * delta[a], lo[a], hi[a]);
            GeomV gt;
            grim::ck::geometry_value(trial, sd, gt);
            real e = want_coll ? cw_weight * collision_raw(gt, world, want_world, margin) : real(0);
            for (int k = 0; k < M; ++k) e += gt.r[k] * gt.r[k];
            return e;
        };
        real best_a_err = real(1e30);
        int best_a = 0;
        if constexpr (TIER == grim::Tier::Thread) {
            for (int ai = 0; ai < 5; ++ai) {
                const real e = eval(ai);
                if (e < best_a_err) { best_a_err = e; best_a = ai; if (e < suff) break; }
            }
        } else {
            for (int ai = G.rank; ai < 5; ai += G.size) w.err[ai] = eval(ai);
            G.sync();
            for (int ai = 0; ai < 5; ++ai) {
                const real e = w.err[ai];
                if (e < best_a_err) { best_a_err = e; best_a = ai; if (e < suff) break; }
            }
            G.sync();
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
    if (G.leader()) {
        store(best_cfg, sd, out, gs);
        out_err[gs] = (float)best_err;
    }
}

static ffi::Error CkHjcdCoarseImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds, ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower, ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t k_max, int64_t enable_collision, float collision_weight, float collision_margin,
    int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    (void)block_threads;
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("hjcd_ik (crocket)", seeds.dimensions()[2],
                        target_Ts.dimensions()[target_Ts.dimensions().size() - 2]);
    const int n = n_problems * n_seeds;
    if (n == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    CK_LAUNCH(ck_hjcd_coarse_kernel, HjcdCoarseWork, tier, n, stream,
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
    (void)block_threads;
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("hjcd_ik (crocket)", seeds.dimensions()[2],
                        target_Ts.dimensions()[target_Ts.dimensions().size() - 2]);
    if (n_problems == 0 || n_seeds == 0) return ffi::Error::Success();
    cudaMemsetAsync(stop_flag->typed_data(), 0, n_problems * sizeof(int), stream);
    const int n = n_problems * n_seeds;
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    CK_LAUNCH(ck_hjcd_lm_kernel, HjcdLmWork, tier, n, stream,
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
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
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
