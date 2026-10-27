/**
 * LS-IK on crocket-traced geometry: the hand-written ls_ik.cu (tag pre-crocket-ik)'s Levenberg-Marquardt, step for step
 * (weighted residual, Gauss-Newton collision rows, Jacobi scaling of the assembled system
 * (jacobi_scale: the hand-written kernel scaled by the pose Jacobian alone), fixed joints, the
 * trust-region radii, the five clamped trial steps, accept below curr * (1 - 1e-4), all-time
 * best tracking), in `real` throughout. One seed per thread, warp or block (`tier`); same FFI
 * signature as LsIkFfi.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

template <grim::Tier TIER>
using LsWork = Work<TIER>;

template <grim::Tier TIER>
__global__ void ck_ls_ik_kernel(
    int wpb, const float* __restrict__ seeds, const float* __restrict__ target_Ts, World world,
    const float* __restrict__ lower, const float* __restrict__ upper, const int* __restrict__ fixed_mask,
    float* __restrict__ out, float* __restrict__ out_err,
    int n_problems, int n_seeds, int max_iter, int enable_collision,
    real wp, real wo, real lambda_init, real eps_pos, real eps_ori, real cw_weight, real margin)
{
    const int gs = seed_index<TIER>(wpb);
    if (gs >= n_problems * n_seeds) return;   // group-uniform
    const Grp<TIER> G;
    CK_WORK(TIER, LsWork<TIER>, w);
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

    geometry(G, cfg, sd, w);
    real best_err = weighted_err(w.g.r, wp, wo);
    if (want_coll) best_err += cw_weight * collision_raw_coop(G, w, w.g, world, want_world, margin);
    real lam = lambda_init;

    for (int iter = 0; iter < max_iter; ++iter) {
        geometry(G, cfg, sd, w);
        const real pen = want_coll ? collision_raw_coop(G, w, w.g, world, want_world, margin) : real(0);
        if (converged(w.g.r, eps_pos, eps_ori) && (!want_coll || cw_weight * pen <= real(1e-12))) break;

        real fw[M];
        real curr_err = cw_weight * pen;
        for (int k = 0; k < M; ++k) { fw[k] = w.g.r[k] * ((k % 6) < 3 ? wp : wo); curr_err += fw[k] * fw[k]; }
        G.sync();   // every lane is done with J before rows are weighted in place
        // Weighted Jacobian Jw = W J in place.
        for (int k = G.rank; k < M; k += G.size) {
            const real wk = (k % 6) < 3 ? wp : wo;
            for (int a = 0; a < NC; ++a) w.g.J[k * NC + a] *= wk;
        }
        G.sync();
        // Rows of A = Jw^T Jw and b = -Jw^T W r, a row per lane (all rows on one lane at the
        // thread tier, column i held in registers across the row); scaled once complete.
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
            w.b[i] = -rb;
        }
        if (want_coll)
            collision_rounds(G, w, world, want_world, margin, (const real*)nullptr,
                             [&](int n) { gn_accumulate(G, w, n, w.A, NC, w.b, cw_weight, cw_weight); });
        G.sync();
        real sc[NC];
        jacobi_scale(G, w.A, w.b, lam, sc);
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

        // The five trials, one per lane (all of them on the one lane at the thread tier).
        const real alphas[5] = {1, 0.5, 0.25, 0.1, 0.025};
        for (int ai = G.rank; ai < 5; ai += G.size) {
            real trial[NC];
            for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[ai] * delta[a], lo[a], hi[a]);
            GeomV gt;
            grim::ck::geometry_value(trial, sd, gt);
            real e = weighted_err(gt.r, wp, wo);
            if (want_coll) e += cw_weight * collision_raw(gt, world, want_world, margin);
            w.err[ai] = e;
        }
        G.sync();
        real best_a_err = real(1e30);
        int best_a = 0;
        for (int ai = 0; ai < 5; ++ai)
            if (w.err[ai] < best_a_err) { best_a_err = w.err[ai]; best_a = ai; }
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
    if (G.leader()) {
        store(best_cfg, sd, out, gs);
        out_err[gs] = (float)best_err;
    }
}

static ffi::Error CkLsIkImpl(
    cudaStream_t stream,
    GRIM_ROT_PARAM
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
    GRIM_ROT_UPLOAD(stream);
    GRIM_CK_CHECK_BLOCK_THREADS("ls_ik (crocket)", block_threads);
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("ls_ik (crocket)", seeds.dimensions()[2], target_Ts.dimensions()[1]);
    const int n = n_problems * n_seeds;
    if (n == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    CK_LAUNCH(ck_ls_ik_kernel, LsWork, tier, n, stream,
        seeds.typed_data(), target_Ts.typed_data(), world, lower.typed_data(), upper.typed_data(),
        fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(), n_problems, n_seeds,
        static_cast<int>(max_iter), static_cast<int>(enable_collision), pos_weight, ori_weight,
        lambda_init, eps_pos, eps_ori, collision_weight, collision_margin);
    GRIM_CK_LAUNCH_STATUS("ls_ik (crocket)", tier);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LsIkFfi, CkLsIkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        GRIM_ROT_BIND
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
