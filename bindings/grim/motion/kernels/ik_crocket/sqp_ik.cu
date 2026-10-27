/**
 * SQP-IK on crocket-traced geometry: the hand-written sqp_ik.cu (tag pre-crocket-ik) step for step, in `real` throughout.
 *
 * Each iteration builds the Jacobi-scaled QP (H = Js^T Js + lam I plus the Gauss-Newton
 * collision terms, g = Js^T W r), takes the Newton step clamped to the scaled joint box and
 * projected onto the linearised collision rows (POCS). Without collision rows, n_inner_iters
 * active-set re-solves hold bound joints and re-solve the rest; with them, the QP
 * min 1/2 p'Hp + g'p s.t. box and the (at most GRIM_MAX_COLL_CON most violated) rows is solved
 * by OSQP-style ADMM on one factorisation, then projected exactly. Trust-region clip, the five
 * clamped trials, accept below curr * (1 - 1e-4), and a filter that keeps the best feasible
 * configuration ahead of any infeasible one. One seed per thread, warp or block (`tier`); same
 * FFI signature as SqpIkFfi.
 */

#include "ik_crocket/ck.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace grim::ck;

#define GRIM_MAX_COLL_CON     4
#define GRIM_COLL_POCS_SWEEPS 8
#define GRIM_ADMM_MAX_ITERS   25
#define GRIM_ADMM_CHECK_EVERY 5

template <grim::Tier TIER>
using SqpWork = Work<TIER, NC, /*WITH_H=*/true>;

template <grim::Tier TIER>
__global__ void ck_sqp_ik_kernel(
    int wpb, const float* __restrict__ seeds, const float* __restrict__ target_Ts, World world,
    const float* __restrict__ lower, const float* __restrict__ upper, const int* __restrict__ fixed_mask,
    float* __restrict__ out, float* __restrict__ out_err, int* __restrict__ out_feasible,
    int n_problems, int n_seeds, int max_iter, int n_inner_iters, int enable_collision,
    real cw_weight, real margin, real wp, real wo, real lambda_init, real eps_pos, real eps_ori)
{
    const int gs = seed_index<TIER>(wpb);
    if (gs >= n_problems * n_seeds) return;
    const Grp<TIER> G;
    CK_WORK(TIER, SqpWork<TIER>, w);
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
    real best_viol = 0;
    if (want_coll) {
        best_viol = collision_raw_coop(G, w, w.g, world, want_world, margin);
        best_err += cw_weight * best_viol;
    }
    real lam = lambda_init;

    for (int iter = 0; iter < max_iter; ++iter) {
        geometry(G, cfg, sd, w);
        const real pen = want_coll ? collision_raw_coop(G, w, w.g, world, want_world, margin) : real(0);
        if (converged(w.g.r, eps_pos, eps_ori) && (!want_coll || cw_weight * pen <= real(1e-12))) break;

        real fw[M];
        real curr_err = cw_weight * pen;
        for (int k = 0; k < M; ++k) { fw[k] = w.g.r[k] * ((k % 6) < 3 ? wp : wo); curr_err += fw[k] * fw[k]; }
        real sc[NC];
        for (int a = 0; a < NC; ++a) {
            real s2 = 0;
            for (int k = 0; k < M; ++k) { const real v = w.g.J[k * NC + a] * ((k % 6) < 3 ? wp : wo); s2 += v * v; }
            sc[a] = sqrt(s2) + real(1e-8);
        }
        for (int k = G.rank; k < M; k += G.size) {
            const real wk = (k % 6) < 3 ? wp : wo;
            for (int a = 0; a < NC; ++a) w.g.J[k * NC + a] *= wk / sc[a];
        }
        G.sync();
        // H = Js^T Js + lam I and g = Js^T W r (w.x), a row per lane.
        for (int i = G.rank; i < NC; i += G.size) {
            real ji[M];
            for (int k = 0; k < M; ++k) ji[k] = w.g.J[k * NC + i];
            for (int j = 0; j < NC; ++j) {
                real acc = 0;
                for (int k = 0; k < M; ++k) acc += ji[k] * w.g.J[k * NC + j];
                w.H[i * NC + j] = acc + (i == j ? lam : real(0));
            }
            real gb = 0;
            for (int k = 0; k < M; ++k) gb += ji[k] * fw[k];
            w.x[i] = gb;
        }
        // Collision: Gauss-Newton terms into H / g, and the most violated rows (scaled space)
        // grad(d)' p >= margin - d kept, identically on every lane, for the hard stage.
        real coll_A[GRIM_MAX_COLL_CON][NC], coll_b[GRIM_MAX_COLL_CON];
        int n_coll = 0;
        if (want_coll)
            collision_rounds(G, w, world, want_world, margin, sc, [&](int n) {
                for (int s2 = 0; s2 < n; ++s2) {
                    const real viol = w.vbuf[s2];
                    if (!(viol > real(0))) continue;
                    int slot = n_coll;
                    if (n_coll < GRIM_MAX_COLL_CON) {
                        ++n_coll;
                    } else {
                        int worst = -1;
                        real smallest = viol;
                        for (int c = 0; c < GRIM_MAX_COLL_CON; ++c)
                            if (coll_b[c] < smallest) { smallest = coll_b[c]; worst = c; }
                        slot = worst;
                    }
                    if (slot >= 0) {
                        for (int a = 0; a < NC; ++a) coll_A[slot][a] = w.gbuf[s2 * NC + a];
                        coll_b[slot] = viol;
                    }
                }
                gn_accumulate(G, w, n, w.H, NC, w.x, cw_weight, -cw_weight);
            });
        G.sync();
        real gq[NC];
        for (int a = 0; a < NC; ++a) gq[a] = w.x[a];

        real lb[NC], ub[NC];
        for (int a = 0; a < NC; ++a) {
            lb[a] = fixed[a] ? real(0) : (lo[a] - cfg[a]) * sc[a];
            ub[a] = fixed[a] ? real(0) : (hi[a] - cfg[a]) * sc[a];
        }
        auto project_feasible = [&](real* p) {
            for (int sweep = 0; sweep < GRIM_COLL_POCS_SWEEPS; ++sweep) {
                bool feasible = true;
                for (int c = 0; c < n_coll; ++c) {
                    real ap = 0, nn = 0;
                    for (int a = 0; a < NC; ++a) { ap += coll_A[c][a] * p[a]; nn += coll_A[c][a] * coll_A[c][a]; }
                    const real viol = coll_b[c] - ap;
                    if (viol <= real(1e-7) || nn < real(1e-12)) continue;
                    feasible = false;
                    const real st = viol / nn;
                    for (int a = 0; a < NC; ++a) p[a] += st * coll_A[c][a];
                }
                for (int a = 0; a < NC; ++a) p[a] = clamp(p[a], lb[a], ub[a]);
                if (feasible) break;
            }
        };
        auto fix_rows = [&](bool rhs) {
            for (int a = G.rank; a < NC; a += G.size) {
                if (!fixed[a]) continue;
                for (int j = 0; j < NC; ++j) w.A[a * NC + j] = w.A[j * NC + a] = 0;
                w.A[a * NC + a] = 1;
                if (rhs) w.b[a] = 0;
            }
        };

        // Newton step.
        for (int i = G.rank; i < NC; i += G.size) {
            for (int j = 0; j < NC; ++j) w.A[i * NC + j] = w.H[i * NC + j];
            w.b[i] = -gq[i];
        }
        G.sync();
        fix_rows(true);
        posv_tier<TIER, NC>(G, w);
        real p[NC];
        for (int a = 0; a < NC; ++a) p[a] = clamp(w.b[a], lb[a], ub[a]);
        project_feasible(p);

        // Box-only active-set refinement.
        for (int k = 0; n_coll == 0 && k < n_inner_iters; ++k) {
            real active[NC], pb[NC];
            for (int a = 0; a < NC; ++a) {
                active[a] = (p[a] <= lb[a] + real(1e-8) || p[a] >= ub[a] - real(1e-8)) ? real(1) : real(0);
                pb[a] = clamp(p[a], lb[a], ub[a]) * active[a];
            }
            G.sync();   // every lane has read the previous solution out of w.b
            for (int a = G.rank; a < NC; a += G.size) {
                real adj = 0;
                for (int b2 = 0; b2 < NC; ++b2) adj += w.H[a * NC + b2] * pb[b2];
                w.b[a] = -(gq[a] + adj) * (real(1) - active[a]);
                for (int b2 = 0; b2 < NC; ++b2)
                    w.A[a * NC + b2] = (active[a] > real(0.5) || active[b2] > real(0.5)) ? real(0) : w.H[a * NC + b2];
                if (active[a] > real(0.5)) w.A[a * NC + a] = 1;
            }
            posv_tier<TIER, NC>(G, w);
            for (int a = 0; a < NC; ++a) p[a] = active[a] > real(0.5) ? pb[a] : clamp(w.b[a], lb[a], ub[a]);
            project_feasible(p);
        }

        // Hard-constrained QP by ADMM (OSQP indirect form): one factorisation, many solves.
        if (n_coll > 0) {
            const real sigma = real(1e-6), rho = real(1), alpha = real(1.6);
            G.sync();
            for (int i = G.rank; i < NC; i += G.size)
                for (int j = 0; j < NC; ++j) {
                    real m = w.H[i * NC + j];
                    for (int c = 0; c < n_coll; ++c) m += rho * coll_A[c][i] * coll_A[c][j];
                    if (i == j) m += sigma + rho;
                    w.A[i * NC + j] = m;
                }
            G.sync();
            fix_rows(false);
            const bool factored = potrf_tier<TIER, NC>(G, w);
            if (factored) {
                real x[NC], zb[NC], yb[NC], zc[GRIM_MAX_COLL_CON], yc[GRIM_MAX_COLL_CON];
                for (int a = 0; a < NC; ++a) { x[a] = p[a]; zb[a] = clamp(x[a], lb[a], ub[a]); yb[a] = 0; }
                for (int c = 0; c < n_coll; ++c) {
                    real ax = 0;
                    for (int a = 0; a < NC; ++a) ax += coll_A[c][a] * x[a];
                    zc[c] = fmax(ax, coll_b[c]);
                    yc[c] = 0;
                }
                for (int it = 0; it < GRIM_ADMM_MAX_ITERS; ++it) {
                    for (int a = G.rank; a < NC; a += G.size) {
                        real v = sigma * x[a] - gq[a] + (rho * zb[a] - yb[a]);
                        for (int c = 0; c < n_coll; ++c) v += coll_A[c][a] * (rho * zc[c] - yc[c]);
                        w.b[a] = fixed[a] ? real(0) : v;
                    }
                    potrs_tier<TIER, NC>(G, w);
                    for (int a = 0; a < NC; ++a) x[a] = w.b[a];
                    G.sync();   // w.b is rebuilt next sweep
                    for (int a = 0; a < NC; ++a) {
                        const real ax = alpha * x[a] + (real(1) - alpha) * zb[a];
                        const real zn = clamp(ax + yb[a] / rho, lb[a], ub[a]);
                        yb[a] += rho * (ax - zn);
                        zb[a] = zn;
                    }
                    for (int c = 0; c < n_coll; ++c) {
                        real ax = 0;
                        for (int a = 0; a < NC; ++a) ax += coll_A[c][a] * x[a];
                        ax = alpha * ax + (real(1) - alpha) * zc[c];
                        const real zn = fmax(ax + yc[c] / rho, coll_b[c]);
                        yc[c] += rho * (ax - zn);
                        zc[c] = zn;
                    }
                    if ((it + 1) % GRIM_ADMM_CHECK_EVERY == 0) {
                        real r_prim = 0, s_prim = 0, r_dual = 0, s_dual = 0;
                        for (int a = 0; a < NC; ++a) {
                            r_prim = fmax(r_prim, fabs(x[a] - zb[a]));
                            s_prim = fmax(s_prim, fmax(fabs(x[a]), fabs(zb[a])));
                        }
                        for (int c = 0; c < n_coll; ++c) {
                            real ax = 0;
                            for (int a = 0; a < NC; ++a) ax += coll_A[c][a] * x[a];
                            r_prim = fmax(r_prim, fabs(ax - zc[c]));
                            s_prim = fmax(s_prim, fmax(fabs(ax), fabs(zc[c])));
                        }
                        for (int a = 0; a < NC; ++a) {
                            real hx = 0, aty = yb[a];
                            for (int b2 = 0; b2 < NC; ++b2) hx += w.H[a * NC + b2] * x[b2];
                            for (int c = 0; c < n_coll; ++c) aty += coll_A[c][a] * yc[c];
                            r_dual = fmax(r_dual, fabs(hx + gq[a] + aty));
                            s_dual = fmax(s_dual, fmax(fabs(hx), fmax(fabs(gq[a]), fabs(aty))));
                        }
                        if (r_prim <= real(1e-4) + real(1e-3) * s_prim && r_dual <= real(1e-4) + real(1e-3) * s_dual)
                            break;
                    }
                }
                for (int a = 0; a < NC; ++a) p[a] = x[a];
            }
            project_feasible(p);
        }

        real delta[NC];
        for (int a = 0; a < NC; ++a) delta[a] = p[a] / sc[a];
        clip_to_radius(delta, trust_radius(w.g.r));

        // The five trials, one per lane; the winner's violation for the filter.
        const real alphas[5] = {1, 0.5, 0.25, 0.1, 0.025};
        for (int ai = G.rank; ai < 5; ai += G.size) {
            real trial[NC];
            for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[ai] * delta[a], lo[a], hi[a]);
            GeomV gt;
            grim::ck::geometry_value(trial, sd, gt);
            real e = weighted_err(gt.r, wp, wo);
            real viol = 0;
            if (want_coll) { viol = collision_raw(gt, world, want_world, margin); e += cw_weight * viol; }
            w.err[ai] = e;
            w.err[8 + ai] = viol;
        }
        G.sync();
        real best_a_err = real(1e30);
        int best_a = 0;
        for (int ai = 0; ai < 5; ++ai)
            if (w.err[ai] < best_a_err) { best_a_err = w.err[ai]; best_a = ai; }
        const real trial_viol = w.err[8 + best_a];
        G.sync();
        real trial[NC];
        for (int a = 0; a < NC; ++a) trial[a] = clamp(cfg[a] + alphas[best_a] * delta[a], lo[a], hi[a]);
        if (best_a_err < curr_err * real(1 - 1e-4)) {
            for (int a = 0; a < NC; ++a) cfg[a] = trial[a];
            lam = fmax(lam * real(0.5), real(1e-10));
        } else {
            lam = fmin(lam * real(3), real(1e6));
        }
        // Filter: feasibility first, then pose error (the restoration phase folded in).
        const bool trial_feas = trial_viol <= real(0), best_feas = best_viol <= real(0);
        const bool take = trial_feas != best_feas ? trial_feas
                        : trial_feas ? best_a_err < best_err : trial_viol < best_viol;
        if (take) {
            best_err = best_a_err;
            best_viol = trial_viol;
            for (int a = 0; a < NC; ++a) best_cfg[a] = trial[a];
        }
    }
    if (G.leader()) {
        store(best_cfg, sd, out, gs);
        out_err[gs] = (float)best_err;
        out_feasible[gs] = best_viol > real(0) ? 0 : 1;
    }
}

static ffi::Error CkSqpIkImpl(
    cudaStream_t stream,
    GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> seeds, ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres, ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes, ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower, ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter, int64_t n_inner_iters, float pos_weight, float ori_weight, float lambda_init,
    float eps_pos, float eps_ori, int64_t enable_collision, float collision_weight,
    float collision_margin, int64_t tier, int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_feasible)
{
    GRIM_ROT_UPLOAD(stream);
    (void)block_threads;   // the block tier runs BLOCK_THREADS lanes (crocket's block variant)
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds = static_cast<int>(seeds.dimensions()[1]);
    GRIM_CK_CHECK_WORLD("sqp_ik (crocket)", seeds.dimensions()[2], target_Ts.dimensions()[1]);
    const int n = n_problems * n_seeds;
    if (n == 0) return ffi::Error::Success();
    const World world{world_spheres.typed_data(), world_capsules.typed_data(),
                      world_boxes.typed_data(), world_halfspaces.typed_data()};
    CK_LAUNCH(ck_sqp_ik_kernel, SqpWork, tier, n, stream,
        seeds.typed_data(), target_Ts.typed_data(), world, lower.typed_data(), upper.typed_data(),
        fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(), out_feasible->typed_data(),
        n_problems, n_seeds, static_cast<int>(max_iter), static_cast<int>(n_inner_iters),
        static_cast<int>(enable_collision), collision_weight, collision_margin, pos_weight,
        ori_weight, lambda_init, eps_pos, eps_ori);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SqpIkFfi, CkSqpIkImpl,
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
        .Attr<int64_t>("n_inner_iters")
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>());
