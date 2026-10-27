/**
 * DLS-IK on crocket-traced geometry: the hand-written dls_ik.cu (tag pre-crocket-ik)'s dual-form Levenberg-Marquardt step
 * d = -Jw^T (Jw Jw^T + lam I)^-1 W r (an M x M solve, M = 6 * n_ee), the clamped trial accepted
 * when the weighted error drops (lam / 3) else rejected (lam * 4), in `real` throughout. No
 * collision terms, as the hand-written kernel. One seed per thread, warp or block (`tier`); the
 * FFI signature is DlsIkFfi's plus `tier` / `block_threads`.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

template <grim::Tier TIER>
using DlsWork = Work<TIER, M, false, /*WITH_GV=*/true>;

template <grim::Tier TIER>
__global__ void ck_dls_ik_kernel(
    int wpb, const float* __restrict__ seeds, const float* __restrict__ target_Ts,
    const float* __restrict__ lower, const float* __restrict__ upper, const int* __restrict__ fixed_mask,
    float* __restrict__ out, float* __restrict__ out_err,
    int n_problems, int n_seeds, int max_iter, real wp, real wo, real lambda_init, real eps_pos, real eps_ori)
{
    const int gs = seed_index<TIER>(wpb);
    if (gs >= n_problems * n_seeds) return;
    const Grp<TIER> G;
    CK_WORK(TIER, DlsWork<TIER>, w);
    Seed sd;
    real cfg[NC], lo[NC], hi[NC];
    bool fixed[NC];
    load_seed(seeds, target_Ts, gs, gs / n_seeds, cfg, sd);
    for (int a = 0; a < NC; ++a) {
        const int src = grim::robot::solved_idx(a);
        lo[a] = lower[src]; hi[a] = upper[src]; fixed[a] = fixed_mask[src] != 0;
    }

    geometry(G, cfg, sd, w);
    real err = weighted_err(w.g.r, wp, wo);
    bool jac_current = true;   // w.g.J belongs to cfg; false after an accepted step
    real lam = lambda_init;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (!jac_current) { geometry(G, cfg, sd, w); jac_current = true; }
        if (converged(w.g.r, eps_pos, eps_ori)) break;

        // A = Jw Jw^T + lam I (M x M), a row per lane; Jw = W J with fixed columns zeroed.
        real wk[M];
        for (int k = 0; k < M; ++k) wk[k] = (k % 6) < 3 ? wp : wo;
        G.sync();
        for (int i = G.rank; i < M; i += G.size) {
            real ji[NC];
            for (int a = 0; a < NC; ++a) ji[a] = fixed[a] ? real(0) : w.g.J[i * NC + a] * wk[i];
            for (int k = 0; k < M; ++k) {
                real acc = i == k ? lam : real(0);
                for (int a = 0; a < NC; ++a) acc += ji[a] * (fixed[a] ? real(0) : w.g.J[k * NC + a] * wk[k]);
                w.A[i * M + k] = acc;
            }
            w.b[i] = w.g.r[i] * wk[i];
        }
        posv_tier<TIER, M>(G, w);

        real trial[NC];
        for (int a = 0; a < NC; ++a) {
            real d = 0;
            if (!fixed[a])
                for (int k = 0; k < M; ++k) d -= w.g.J[k * NC + a] * wk[k] * w.b[k];
            trial[a] = clamp(cfg[a] + d, lo[a], hi[a]);
        }
        geometry_value(G, trial, sd, w);
        const real e_trial = weighted_err(w.gv.r, wp, wo);
        if (e_trial < err) {
            for (int a = 0; a < NC; ++a) cfg[a] = trial[a];
            err = e_trial;
            jac_current = false;
            lam = fmax(lam / real(3), real(1e-7));
        } else {
            lam = fmin(lam * real(4), real(1e7));
        }
    }
    if (G.leader()) {
        store(cfg, sd, out, gs);
        out_err[gs] = (float)err;
    }
}

static ffi::Error CkDlsIkImpl(
    cudaStream_t stream,
    GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> seeds, ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower, ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter, float pos_weight, float ori_weight, float lambda_init, float eps_pos,
    float eps_ori, int64_t enable_collision, int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    GRIM_ROT_UPLOAD(stream);
    (void)block_threads;   // the block tier runs BLOCK_THREADS lanes (crocket's block variant)
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    if (seeds.dimensions()[2] != grim::robot::n_q || target_Ts.dimensions()[1] != grim::robot::n_ee)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "dls_ik (crocket): launch does not match the robot of this build");
    if (enable_collision || world_spheres.dimensions()[0] || world_capsules.dimensions()[0] ||
        world_boxes.dimensions()[0] || world_halfspaces.dimensions()[0])
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "dls_ik has no collision terms; use ls_ik or sqp_ik");
    const int n = n_problems * n_seeds;
    if (n == 0) return ffi::Error::Success();
    CK_LAUNCH(ck_dls_ik_kernel, DlsWork, tier, n, stream,
        seeds.typed_data(), target_Ts.typed_data(), lower.typed_data(), upper.typed_data(),
        fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(), n_problems, n_seeds,
        static_cast<int>(max_iter), pos_weight, ori_weight, lambda_init, eps_pos, eps_ori);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DlsIkFfi, CkDlsIkImpl,
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
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());
