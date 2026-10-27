/**
 * MPPI + L-BFGS IK on crocket-traced geometry: the hand-written mppi_ik.cu (tag pre-crocket-ik) step for step, in `real`
 * except the noise, which is drawn by the same float xorshift / Box-Muller stream (keyed on
 * rng_seed, problem and seed slot) so both precisions and the hand-written kernel see the same
 * particles. MPPI with RNG replay, then (unless MPPI already converged) L-BFGS with the
 * trust-region radii and an early-exit line search; collision in the merit and, for L-BFGS, the
 * Gauss-Newton gradient. Same FFI signature as MppiIkFfi.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

#define CK_MAX_LBFGS_M 8

template <grim::Tier TIER>
using MppiWork = Work<TIER, 1, false, /*WITH_GV=*/true>;

/** `n` draws of xorshift32: where the stream stands after n / 2 normals. */
__device__ __forceinline__ void rng_skip(uint32_t& state, int n)
{
    for (int i = 0; i < n; ++i) xorshift32(state);
}

template <grim::Tier TIER>
__global__ void ck_mppi_ik_kernel(
    int wpb, const float* __restrict__ seeds, const float* __restrict__ target_Ts, World world,
    const float* __restrict__ lower, const float* __restrict__ upper, const int* __restrict__ fixed_mask,
    const int* __restrict__ rng_seed_ptr, float* __restrict__ out, float* __restrict__ out_err,
    int n_problems, int n_seeds, int n_particles, int n_mppi_iters, int n_lbfgs_iters, int m_lbfgs,
    int enable_collision, real cw_weight, real margin, real sigma, real temperature,
    real wp, real wo, real eps_pos, real eps_ori)
{
    const int gs = seed_index<TIER>(wpb);
    if (gs >= n_problems * n_seeds) return;
    const Grp<TIER> G;
    CK_WORK(TIER, MppiWork<TIER>, w);
    const int p = gs / n_seeds, s = gs % n_seeds;
    Seed sd;
    real cfg[NC], best_cfg[NC], lo[NC], hi[NC];
    bool fixed[NC];
    int n_free = 0;
    load_seed(seeds, target_Ts, gs, p, cfg, sd);
    for (int a = 0; a < NC; ++a) {
        const int src = grim::robot::solved_idx(a);
        lo[a] = lower[src]; hi[a] = upper[src]; fixed[a] = fixed_mask[src] != 0;
        best_cfg[a] = cfg[a];
        n_free += !fixed[a];
    }
    const bool want_world = enable_collision && NWS > 0;
    const bool want_coll = want_world || has_self;
    auto merit_coop = [&]() {   // of w.gv
        real e = weighted_err(w.gv.r, wp, wo);
        if (want_coll) e += cw_weight * collision_raw_coop(G, w, w.gv, world, want_world, margin);
        return e;
    };

    geometry_value(G, cfg, sd, w);
    real best_err = merit_coop();

    uint32_t rng = (uint32_t)(*rng_seed_ptr) ^ (uint32_t)(s * 0x9e3779b9u) ^ (uint32_t)(p * 0x6c62272eu);
    xorshift32(rng); xorshift32(rng); xorshift32(rng);
    const real inv_temp = real(1) / fmax(temperature, real(1e-8));
    const int np = n_particles < CK_MAX_EVAL ? n_particles : CK_MAX_EVAL;
    const int per_particle = 2 * n_free;   // xorshift draws one particle's noise takes

    for (int it = 0; it < n_mppi_iters; ++it) {
        const uint32_t checkpoint = rng;
        // Particles, one per lane: each lane jumps the stream to its particle.
        uint32_t lane_rng = checkpoint;
        rng_skip(lane_rng, per_particle * G.rank);
        for (int k = G.rank; k < np; k += G.size) {
            real trial[NC];
            for (int a = 0; a < NC; ++a) {
                const real n = fixed[a] ? real(0) : real(rng_normal(lane_rng)) * sigma;
                trial[a] = clamp(cfg[a] + n, lo[a], hi[a]);
            }
            GeomV gt;
            grim::ck::geometry_value(trial, sd, gt);
            real e = weighted_err(gt.r, wp, wo);
            if (want_coll) e += cw_weight * collision_raw(gt, world, want_world, margin);
            w.err[k] = e;
            rng_skip(lane_rng, per_particle * (G.size - 1));
        }
        G.sync();
        real costs[CK_MAX_EVAL];
        real min_cost = real(1e30);
        for (int k = 0; k < np; ++k) { costs[k] = w.err[k]; min_cost = fmin(min_cost, costs[k]); }
        real total = 0;
        for (int k = 0; k < np; ++k) { costs[k] = exp(-(costs[k] - min_cost) * inv_temp); total += costs[k]; }
        const real inv_total = real(1) / fmax(total, real(1e-20));
        uint32_t replay = checkpoint;
        real delta[NC];
        for (int a = 0; a < NC; ++a) delta[a] = 0;
        for (int k = 0; k < np; ++k) {
            const real wk = costs[k] * inv_total;
            for (int a = 0; a < NC; ++a) delta[a] += wk * (fixed[a] ? real(0) : real(rng_normal(replay)) * sigma);
        }
        rng = replay;   // where the particles left the stream
        for (int a = 0; a < NC; ++a) cfg[a] = clamp(cfg[a] + delta[a], lo[a], hi[a]);
        geometry_value(G, cfg, sd, w);
        const real e = merit_coop();
        if (e < best_err) { best_err = e; for (int a = 0; a < NC; ++a) best_cfg[a] = cfg[a]; }
    }

    geometry_value(G, best_cfg, sd, w);
    bool done = converged(w.gv.r, eps_pos, eps_ori);
    if (enable_collision || has_self) {
        const real pen = collision_raw_coop(G, w, w.gv, world, want_world, margin);
        done = done && cw_weight * pen <= real(1e-12);
    }

    if (!done) {
        for (int a = 0; a < NC; ++a) cfg[a] = best_cfg[a];
        const int mm = m_lbfgs < CK_MAX_LBFGS_M ? m_lbfgs : CK_MAX_LBFGS_M;
        real s_buf[CK_MAX_LBFGS_M * NC], y_buf[CK_MAX_LBFGS_M * NC], rho[CK_MAX_LBFGS_M], alph[CK_MAX_LBFGS_M];
        real grad[NC], g_prev[NC], cfg_prev[NC], dir[NC];
        int m_used = 0, newest = -1;
        for (int i = 0; i < CK_MAX_LBFGS_M * NC; ++i) s_buf[i] = y_buf[i] = 0;
        for (int i = 0; i < CK_MAX_LBFGS_M; ++i) rho[i] = 0;
        for (int a = 0; a < NC; ++a) { g_prev[a] = 0; cfg_prev[a] = cfg[a]; }

        for (int iter = 0; iter < n_lbfgs_iters; ++iter) {
            geometry(G, cfg, sd, w);
            const real pen = want_coll ? cw_weight * collision_raw_coop(G, w, w.g, world, want_world, margin) : real(0);
            if (converged(w.g.r, eps_pos, eps_ori) && (!want_coll || pen <= real(1e-12))) break;
            real fw[M];
            real curr_err = pen;
            for (int k = 0; k < M; ++k) { fw[k] = w.g.r[k] * ((k % 6) < 3 ? wp : wo); curr_err += fw[k] * fw[k]; }
            for (int a = 0; a < NC; ++a) {
                real acc = 0;
                for (int k = 0; k < M; ++k) acc += w.g.J[k * NC + a] * ((k % 6) < 3 ? wp : wo) * fw[k];
                grad[a] = fixed[a] ? real(0) : acc;
            }
            if (want_coll)
                collision_rounds(G, w, world, want_world, margin, (const real*)nullptr, [&](int n) {
                    for (int s2 = 0; s2 < n; ++s2) {
                        const real viol = w.vbuf[s2];
                        if (!(viol > real(0))) continue;
                        for (int a = 0; a < NC; ++a)
                            if (!fixed[a]) grad[a] -= cw_weight * viol * w.gbuf[s2 * NC + a];
                    }
                });
            if (iter > 0) {
                real sy = 0, yy = 0, sk[NC], yk[NC];
                for (int a = 0; a < NC; ++a) {
                    sk[a] = cfg[a] - cfg_prev[a]; yk[a] = grad[a] - g_prev[a];
                    sy += sk[a] * yk[a]; yy += yk[a] * yk[a];
                }
                if (sy > real(1e-10) * yy + real(1e-30)) {
                    newest = (newest + 1) % mm;
                    for (int a = 0; a < NC; ++a) { s_buf[newest * NC + a] = sk[a]; y_buf[newest * NC + a] = yk[a]; }
                    rho[newest] = real(1) / sy;
                    if (m_used < mm) ++m_used;
                }
            }
            for (int a = 0; a < NC; ++a) { cfg_prev[a] = cfg[a]; g_prev[a] = grad[a]; }
            if (m_used > 0) {
                for (int a = 0; a < NC; ++a) dir[a] = -grad[a];
                for (int i = 0; i < m_used; ++i) {
                    const int idx = (newest - i + mm) % mm;
                    real al = 0;
                    for (int a = 0; a < NC; ++a) al += rho[idx] * s_buf[idx * NC + a] * dir[a];
                    alph[i] = al;
                    for (int a = 0; a < NC; ++a) dir[a] -= al * y_buf[idx * NC + a];
                }
                real sy = 0, yy = 0;
                for (int a = 0; a < NC; ++a) {
                    sy += s_buf[newest * NC + a] * y_buf[newest * NC + a];
                    yy += y_buf[newest * NC + a] * y_buf[newest * NC + a];
                }
                const real gamma = yy > real(1e-20) ? fmax(real(1e-8), fmin(real(1), sy / yy)) : real(1);
                for (int a = 0; a < NC; ++a) dir[a] *= gamma;
                for (int i = m_used - 1; i >= 0; --i) {
                    const int idx = (newest - i + mm) % mm;
                    real be = 0;
                    for (int a = 0; a < NC; ++a) be += rho[idx] * y_buf[idx * NC + a] * dir[a];
                    for (int a = 0; a < NC; ++a) dir[a] += s_buf[idx * NC + a] * (alph[i] - be);
                }
            } else {
                real gn = 0;
                for (int a = 0; a < NC; ++a) gn += grad[a] * grad[a];
                gn = sqrt(gn) + real(1e-18);
                for (int a = 0; a < NC; ++a) dir[a] = -grad[a] / gn;
            }
            for (int a = 0; a < NC; ++a) if (fixed[a]) dir[a] = 0;
            clip_to_radius(dir, trust_radius(w.g.r));

            const real alphas[5] = {1, 0.5, 0.25, 0.1, 0.025};
            const real suff = curr_err * real(1 - 1e-4);
            auto eval = [&](int ai) {
                real trial[NC];
                for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[ai] * dir[a], lo[a], hi[a]);
                GeomV gt;
                grim::ck::geometry_value(trial, sd, gt);
                real e = weighted_err(gt.r, wp, wo);
                if (want_coll) e += cw_weight * collision_raw(gt, world, want_world, margin);
                return e;
            };
            real best_a_err = real(1e30);
            int best_a = 0;
            if constexpr (TIER == grim::Tier::Thread) {
                for (int ai = 0; ai < 5; ++ai) {
                    const real e = eval(ai);
                    if (e < best_a_err) { best_a_err = e; best_a = ai; }
                    if (e < suff) break;
                }
            } else {
                for (int ai = G.rank; ai < 5; ai += G.size) w.err[ai] = eval(ai);
                G.sync();
                for (int ai = 0; ai < 5; ++ai) {
                    const real e = w.err[ai];
                    if (e < best_a_err) { best_a_err = e; best_a = ai; }
                    if (e < suff) break;
                }
                G.sync();
            }
            for (int a = 0; a < NC; ++a) cfg[a] = clamp(cfg[a] + alphas[best_a] * dir[a], lo[a], hi[a]);
            if (best_a_err < best_err) { best_err = best_a_err; for (int a = 0; a < NC; ++a) best_cfg[a] = cfg[a]; }
        }
    }
    if (G.leader()) {
        store(best_cfg, sd, out, gs);
        out_err[gs] = (float)best_err;
    }
}

static ffi::Error CkMppiIkImpl(
    cudaStream_t stream,
    GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> seeds, ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower, ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask, ffi::Buffer<ffi::DataType::S32> rng_seed_buf,
    int64_t n_particles, int64_t n_mppi_iters, int64_t n_lbfgs_iters, int64_t m_lbfgs,
    float sigma, float mppi_temperature, float pos_weight, float ori_weight, float eps_pos,
    float eps_ori, int64_t enable_collision, float collision_weight, float collision_margin,
    int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    GRIM_ROT_UPLOAD(stream);
    (void)block_threads;
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("mppi_ik (crocket)", seeds.dimensions()[2],
                        target_Ts.dimensions()[target_Ts.dimensions().size() - 2]);
    const int n = n_problems * n_seeds;
    if (n == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    CK_LAUNCH(ck_mppi_ik_kernel, MppiWork, tier, n, stream,
        seeds.typed_data(), target_Ts.typed_data(), world, lower.typed_data(), upper.typed_data(),
        fixed_mask.typed_data(), rng_seed_buf.typed_data(), out->typed_data(), out_err->typed_data(),
        n_problems, n_seeds, static_cast<int>(n_particles), static_cast<int>(n_mppi_iters),
        static_cast<int>(n_lbfgs_iters), static_cast<int>(m_lbfgs), static_cast<int>(enable_collision),
        collision_weight, collision_margin, sigma, mppi_temperature, pos_weight, ori_weight, eps_pos, eps_ori);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MppiIkFfi, CkMppiIkImpl,
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
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // rng_seed
        .Attr<int64_t>("n_particles")
        .Attr<int64_t>("n_mppi_iters")
        .Attr<int64_t>("n_lbfgs_iters")
        .Attr<int64_t>("m_lbfgs")
        .Attr<float>("sigma")
        .Attr<float>("mppi_temperature")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());
