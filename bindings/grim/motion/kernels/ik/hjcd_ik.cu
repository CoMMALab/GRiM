/**
 * HJCD-IK CUDA kernel with XLA FFI binding.
 *
 * Implements the two-phase HJCD-IK algorithm in CUDA:
 *
 *   Phase 1 (Coarse):  Greedy coordinate-descent — selects the single
 *                      joint with maximum predicted error reduction per step.
 *   Phase 2 (Refine):  Levenberg-Marquardt — column-scaled normal equations
 *                      with joint-limit prior, line search, stall kicks.
 *
 * Multi-EE support: stacked residuals and Jacobians for all EEs simultaneously.
 * FK is called once per LM iteration; each EE reads from the FK result.
 *
 * Built per robot (grim.motion._build): kinematics and collision tables are
 * compile-time constants, read through robot.cuh.
 *
 * Numerical stability:
 *   - FK and Jacobian in float32.
 *   - Normal-equation matrix and Cholesky solve in float64.
 *   - All kernel launches are associated with the caller's CUDA stream so
 *     there are no implicit device synchronisations.

 */

#include "robot.cuh"
#include "glass_solve.cuh"
#include "tier_kernel.cuh"
#include "collision.cuh"

#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Math helpers (IK-specific)
// ---------------------------------------------------------------------------

/** Dot product of two 3-vectors. */
__device__ __forceinline__
float dot3(const float* __restrict__ a, const float* __restrict__ b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// ---------------------------------------------------------------------------
// Adaptive weighting (matches JAX _adaptive_weights)
// ---------------------------------------------------------------------------

/**
 * Compute per-residual adaptive weights.
 * w[0:3] = 1, w[3:6] = clamp(pos_err / ori_err, 0.05, 1.0).
 */
__device__ void adaptive_weights(const float* __restrict__ r, float* __restrict__ w)
{
    const float pos_err = norm3(r)     + 1e-8f;
    const float ori_err = norm3(r + 3) + 1e-8f;
    const float ori_scale = clampf(pos_err / ori_err, 0.05f, 1.0f);
    w[0] = 1.0f; w[1] = 1.0f; w[2] = 1.0f;
    w[3] = ori_scale; w[4] = ori_scale; w[5] = ori_scale;
}

// ---------------------------------------------------------------------------
// Coarse greedy coordinate-descent kernel (Phase 1)
// ---------------------------------------------------------------------------

/**
 * One CUDA thread per seed.  Runs k_max greedy CD steps.
 *
 * Multi-EE: stacked residuals and Jacobians; per-EE adaptive weights;
 * CD selects best joint based on all EEs combined.
 *
 * @param seeds          (n_problems, n_seeds, n_act)
 * @param target_Ts      (n_problems, n_ee, 7)          target poses
 * @param lower          (n_act,) lower limits
 * @param upper          (n_act,) upper limits
 * @param fixed_mask     (n_act,) int32; 1 = fixed, 0 = free
 * @param out            (n_problems, n_seeds, n_act)  output configurations
 * @param n_ee           number of end-effectors
 */

__global__
void hjcd_ik_coarse_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ target_Ts,       // (n_problems, n_ee, 7)
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    int n_problems, int n_seeds,
    int k_max, int enable_collision, float collision_weight, float collision_margin)
{
    GRIM_IK_DIMS_PROLOGUE();

    // ── Shared memory: robot parameters loaded once per block ───────────────
    __shared__ float s_target_Ts   [MAX_EE * 7];
    __shared__ float s_lower   [MAX_ACT];
    __shared__ float s_upper   [MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];

    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[GRIM_ACT_SRC(i)];
        s_upper[i]      = upper[GRIM_ACT_SRC(i)];
        s_fixed_mask[i] = fixed_mask[GRIM_ACT_SRC(i)];
    }
    const int p = blockIdx.y;
    for (int i = threadIdx.x; i < n_ee * 7; i += blockDim.x)
        s_target_Ts[i] = target_Ts[p * n_ee * 7 + i];
    __syncthreads();

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p * n_seeds + s;

    // Local configuration (thread-private).
    float cfg[MAX_ACT];
    GRIM_LOAD_SEED(cfg);

    // Scratch for FK world transforms and stacked Jacobian.
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    for (int iter = 0; iter < k_max; iter++) {
        GRIM_RES_JAC(cfg);

        // Per-EE adaptive weights with orientation gating (pos < 1mm).
        float w[6 * MAX_EE];
        for (int ee = 0; ee < n_ee; ee++) {
            float w_ee[6];
            adaptive_weights(r + ee*6, w_ee);
            const float pe = norm3(r + ee*6);
            if (pe >= 1e-3f) { w_ee[3] = 0.0f; w_ee[4] = 0.0f; w_ee[5] = 0.0f; }
            for (int k = 0; k < 6; k++) w[ee*6+k] = w_ee[k];
        }

        // fw = r * w (stacked), apply weights in-place to J rows.
        float fw[6 * MAX_EE];
        for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * w[k];
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k * n_act + a] *= w[k];

        // Per-joint: JwTfw[a] = Jw[:,a]^T fw,  Jw_normsq[a] = ||Jw[:,a]||^2.
        float JwTfw[MAX_ACT], Jw_normsq[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float s_dot = 0.0f, s_sq = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) {
                const float jwa = J[k * n_act + a];
                s_dot += jwa * fw[k];
                s_sq  += jwa * jwa;
            }
            JwTfw[a]     = s_dot;
            Jw_normsq[a] = s_sq + 1e-8f;
        }

        // Find best joint.
        int best = -1;
        float best_impr = -1.0f;
        for (int a = 0; a < n_act; a++) {
            if (s_fixed_mask[a]) continue;
            const float impr = JwTfw[a] * JwTfw[a] / Jw_normsq[a];
            if (impr > best_impr) { best_impr = impr; best = a; }
        }
        if (best < 0) break;

        // Apply step for best joint only.
        const float step = -JwTfw[best] / Jw_normsq[best];
        cfg[best] = clampf(cfg[best] + step, s_lower[best], s_upper[best]);
    }

    // Compute final unweighted error for scoring (sum over all EEs).
    GRIM_RES(cfg, r);
    float final_err = 0.0f;
    for (int k = 0; k < 6 * n_ee; k++) final_err += r[k] * r[k];

    // Self-collision is independent of world geometry, so it must not sit behind
    // `enable_collision` (which tracks *world* obstacles); an obstacle-free
    // problem is exactly where a folded solution slips through unnoticed.
    const bool want_world_coarse = enable_collision && n_robot_spheres > 0;
    const bool want_self_coarse  = n_self_pairs > 0;
    if (want_world_coarse || want_self_coarse) {
        GRIM_FK_ALL(cfg, T_world);

        float pen = 0.0f;
        // Own guard: self-collision alone must not switch on world penalties.
        for (int i = 0; want_world_coarse && i < n_robot_spheres; i++) {
            const int jidx = robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= n_joints) continue;

            const float* sp = robot_spheres_local + i * 4;
            float local_p[3] = {sp[0], sp[1], sp[2]};
            float world_p[3];
            apply_se3_point(T_world + jidx * 7, local_p, world_p);
            const float rr = sp[3];

            for (int m = 0; m < n_world_spheres; m++) {
                const float* o = world_spheres + m * 4;
                const float d = sphere_sphere_dist(world_p[0], world_p[1], world_p[2], rr,
                                                   o[0], o[1], o[2], o[3]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_capsules; m++) {
                const float* o = world_capsules + m * 7;
                const float d = sphere_capsule_dist(world_p[0], world_p[1], world_p[2], rr,
                                                    o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_boxes; m++) {
                const float* o = world_boxes + m * 15;
                const float d = sphere_box_dist(world_p[0], world_p[1], world_p[2], rr,
                                                o[0], o[1], o[2],
                                                o[3], o[4], o[5],
                                                o[6], o[7], o[8],
                                                o[9], o[10], o[11],
                                                o[12], o[13], o[14]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_halfspaces; m++) {
                const float* o = world_halfspaces + m * 6;
                const float d = sphere_halfspace_dist(world_p[0], world_p[1], world_p[2], rr,
                                                      o[0], o[1], o[2], o[3], o[4], o[5]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
        }
        if (want_self_coarse) {
            pen += self_collision_penalty(
                T_world, self_sph_local, self_link_start, self_link_joint,
                self_pair_i, self_pair_j, n_self_pairs, collision_margin);
        }
        final_err += collision_weight * pen;
    }
    out_err[gs] = final_err;

    // Write output.
    GRIM_STORE(cfg);
}

// ---------------------------------------------------------------------------
// LM refinement kernel (Phase 2)
// ---------------------------------------------------------------------------

/**
 * One CUDA thread per refinement seed.  Runs max_iter LM iterations with:
 *   - Per-EE adaptive pos/ori row weighting.
 *   - Jacobi column scaling.
 *   - Soft joint-limit prior.
 *   - Line search over {1, 0.5, 0.25, 0.1, 0.025} step multipliers.
 *   - Stall detection with random kicks.
 *   - Best-config tracking.
 *   - Multi-EE: convergence requires ALL EEs satisfied.
 *
 * @param seeds         (n_problems, n_seeds, n_act)
 * @param noise         (n_problems, n_seeds, max_iter, n_act)  kick noise
 * @param target_Ts     (n_problems, n_ee, 7)          target poses
 */
// Launch shape per tier (LM is register/local-memory heavy, so blocks stay modest).
#define GRIM_HJCD_THREAD_TPB 32

// Trial step sizes in the LM line search; also sizes the warp/block reduction buffer.
#define N_HJCD_ALPHAS 5

// Tiered: one seed per thread / warp / block. See _tier_kernel.cuh for the pattern —
// every cooperative loop is `for (i = rank; i < n; i += size)`, which collapses to the
// original sequential code at Tier::Thread.
//
// Unlike ls_ik, this kernel's line search EARLY-EXITS on sufficient descent, so the
// tiers genuinely differ there rather than sharing one loop; see the line-search
// comment below.
template <grim::Tier TIER, uint32_t N>
__global__
void hjcd_ik_lm_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ noise,         // (n_problems, n_seeds, max_iter, n_act)
    const float* __restrict__ target_Ts,       // (n_problems, n_ee, 7)
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    int*         __restrict__ stop_flag,
    int n_problems, int n_seeds, int max_iter, int early_stop,
    float lambda_init, float limit_prior_weight, float kick_scale,
    float eps_pos, float eps_ori, int stall_patience,
    int enable_collision, float collision_weight, float collision_margin)
{
    GRIM_IK_DIMS_PROLOGUE();

    // ── Shared memory: robot parameters loaded once per block ───────────────
    __shared__ float s_target_Ts   [MAX_EE * 7];
    __shared__ float s_lower   [MAX_ACT];
    __shared__ float s_upper   [MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];

    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[GRIM_ACT_SRC(i)];
        s_upper[i]      = upper[GRIM_ACT_SRC(i)];
        s_fixed_mask[i] = fixed_mask[GRIM_ACT_SRC(i)];
    }
    const int p = blockIdx.y;
    for (int i = threadIdx.x; i < n_ee * 7; i += blockDim.x)
        s_target_Ts[i] = target_Ts[p * n_ee * 7 + i];
    __syncthreads();

    GRIM_TIER_GROUP_VARS(TIER);
    const int s = GRIM_TIER_SEED_INDEX(TIER);
    if (s >= n_seeds) return;   // group-uniform: whole thread/warp/block retires
    const int gs = p * n_seeds + s;

    // Per-seed normal equations. Thread tier: thread-local so nvcc can promote it.
    // Warp/block: shared, since every cooperating lane reads it.
    constexpr int SLOTS  = GRIM_TIER_SLOTS(TIER, N);
    constexpr int SMEM_N = GRIM_TIER_SMEM_N(TIER, N);
    __shared__ double sh_A   [SLOTS][SMEM_N * SMEM_N];
    __shared__ double sh_rhs [SLOTS][SMEM_N];
    __shared__ int    sh_fail[SLOTS];
    __shared__ int    sh_stop[SLOTS];   // early_stop flag, read once per group
    __shared__ float  sh_ls  [SLOTS][N_HJCD_ALPHAS];
    const int slot = GRIM_TIER_SLOT(TIER);

    // Per-lane state. At the warp/block tiers every lane of the group runs the same
    // FK/Jacobian on the same seed, so these stay identical across the group.
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    // Load initial config.
    GRIM_LOAD_SEED(cfg);
    for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];

    // Joint-limit mid / half-range for prior.
    float mid[MAX_ACT], half_range[MAX_ACT];
    for (int a = 0; a < n_act; a++) {
        // An unbounded coordinate (infinite limit, e.g. a floating base) gets no prior.
        const bool bounded = isfinite(s_lower[a]) && isfinite(s_upper[a]);
        mid[a]        = bounded ? (s_lower[a] + s_upper[a]) * 0.5f : 0.0f;
        half_range[a] = bounded ? (s_upper[a] - s_lower[a]) * 0.5f + 1e-8f : INFINITY;
    }

    // Compute initial unweighted squared error (sum over all EEs).
    GRIM_RES_JAC(cfg);
    float best_err = 0.0f;
    for (int k = 0; k < 6 * n_ee; k++) best_err += r[k] * r[k];

    // Hoisted out of the merit lambda: the convergence test below needs them.
    const bool want_self  = n_self_pairs > 0;
    const bool want_world = enable_collision && n_robot_spheres > 0;

    auto collision_penalty = [&](const float* cfg_eval, float* T_eval) {
        // See the coarse kernel above: self-collision is not gated on
        // `enable_collision`, which tracks world obstacles only.
        if (!want_world && !want_self) return 0.0f;

        GRIM_FK_ALL(cfg_eval, T_eval);

        float pen = 0.0f;
        // Own guard: self-collision alone must not switch on world penalties.
        for (int i = 0; want_world && i < n_robot_spheres; i++) {
            const int jidx = robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= n_joints) continue;

            const float* sp = robot_spheres_local + i * 4;
            float local_p[3] = {sp[0], sp[1], sp[2]};
            float world_p[3];
            apply_se3_point(T_eval + jidx * 7, local_p, world_p);
            const float rr = sp[3];

            for (int m = 0; m < n_world_spheres; m++) {
                const float* o = world_spheres + m * 4;
                const float d = sphere_sphere_dist(world_p[0], world_p[1], world_p[2], rr,
                                                   o[0], o[1], o[2], o[3]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_capsules; m++) {
                const float* o = world_capsules + m * 7;
                const float d = sphere_capsule_dist(world_p[0], world_p[1], world_p[2], rr,
                                                    o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_boxes; m++) {
                const float* o = world_boxes + m * 15;
                const float d = sphere_box_dist(world_p[0], world_p[1], world_p[2], rr,
                                                o[0], o[1], o[2],
                                                o[3], o[4], o[5],
                                                o[6], o[7], o[8],
                                                o[9], o[10], o[11],
                                                o[12], o[13], o[14]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_halfspaces; m++) {
                const float* o = world_halfspaces + m * 6;
                const float d = sphere_halfspace_dist(world_p[0], world_p[1], world_p[2], rr,
                                                      o[0], o[1], o[2], o[3], o[4], o[5]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
        }

        // Self-collision. This solver checked the robot against the WORLD but
        // never against itself, so a returned configuration could have the arm
        // folded through its own links and still report collision-free. On the
        // Panda, 6.5% of random in-limit configurations self-collide.
        //
        // Shares `self_collision_penalty` with the fused collision kernel so
        // both evaluate identical geometry. `self_sph_local` travels with the
        // tables because `link_start` indexes IT, not `robot_spheres_local`.
        // n_self_pairs == 0 disables the whole thing, and that is the default --
        // existing callers are unaffected until they pass a pair table, which
        // must be SRDF-filtered (without an SRDF the spherized model treats
        // adjacent links as permanently overlapping and rejects everything).
        if (want_self) {
            pen += self_collision_penalty(
                T_eval, self_sph_local, self_link_start, self_link_joint,
                self_pair_i, self_pair_j, n_self_pairs, collision_margin);
        }
        return collision_weight * pen;
    };

    best_err += collision_penalty(cfg, T_world);

    float lam   = lambda_init;
    int   stall = 0;
    bool  done  = false;

    for (int iter = 0; iter < max_iter; iter++) {
        if (done) break;
        // Opt-in: stop once another seed of this problem has converged. Faster, but the
        // other seeds' outputs then depend on scheduling (only the problem's best is
        // meaningful), so it is off by default. The flag is read by ONE lane and shared,
        // so a cooperative group never splits around a barrier.
        if (early_stop) {
            bool stop;
            if constexpr (TIER == grim::Tier::Thread) {
                stop = *(volatile int*)(stop_flag + p) != 0;
            } else {
                if (leader) sh_stop[slot] = *(volatile int*)(stop_flag + p);
                group_sync();
                stop = sh_stop[slot] != 0;
                group_sync();
            }
            if (stop) break;
        }

        // ── Jacobian + residual ──────────────────────────────────────────
        GRIM_RES_JAC(cfg);

        // Unweighted current error (sum over all EEs).
        float curr_err = 0.0f;
        for (int k = 0; k < 6 * n_ee; k++) curr_err += r[k] * r[k];
        curr_err += collision_penalty(cfg, T_world);

        // Early exit check: ALL EEs must converge.
        {
            bool all_conv = true;
            for (int ee = 0; ee < n_ee; ee++) {
                float r_pos = norm3(r + ee*6);
                float r_ori = norm3(r + ee*6 + 3);
                if (r_pos >= eps_pos || r_ori >= eps_ori) { all_conv = false; break; }
            }
            // Pose convergence is not convergence while a collision constraint
            // is active: the arm can sit exactly on target and folded through
            // itself. Open loop means running until everything being solved for
            // has converged, not until the pose has.
            if (all_conv && (want_self || want_world))
                all_conv = collision_penalty(cfg, T_world) <= 1e-12f;
            if (all_conv) {
                done = true;
                if (early_stop && leader) {
                    atomicExch(stop_flag + p, 1);
                    __threadfence();
                }
                break;
            }
        }

        // ── Per-EE adaptive row weighting ────────────────────────────────
        float w[6 * MAX_EE];
        for (int ee = 0; ee < n_ee; ee++) {
            float w_ee[6];
            adaptive_weights(r + ee*6, w_ee);
            for (int k = 0; k < 6; k++) w[ee*6+k] = w_ee[k];
        }
        float fw[6 * MAX_EE];
        for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * w[k];
        // Apply weights in-place to J rows.
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k * n_act + a] *= w[k];

        // ── Jacobi column scaling ─────────────────────────────────────────
        float col_scale[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float sq = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) { float v = J[k*n_act+a]; sq += v*v; }
            col_scale[a] = sqrtf(sq) + 1e-8f;
        }
        // Scale J in-place → Js (reuse J buffer).
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k * n_act + a] /= col_scale[a];

        // ── Normal equations (float64 for numerical stability) ───────────
        // Assembled at stride N (the compile-time bucket), not n_act, so the GLASS
        // solve reads the buffer directly with no serial repack.
        double  A_local[(TIER == grim::Tier::Thread) ? N * N : 1];
        double  rhs_local[(TIER == grim::Tier::Thread) ? N : 1];
        double* A_s   = (TIER == grim::Tier::Thread) ? A_local   : sh_A[slot];
        double* rhs_s = (TIER == grim::Tier::Thread) ? rhs_local : sh_rhs[slot];

        // Every lane of the group holds an identical J, so the (i,j) entries of
        // A = J^T J distribute with no communication. This is the O(n_act^2 * 6*n_ee)
        // inner product — the heaviest step after the FKs.
        for (int idx = rank; idx < n_act * n_act; idx += size) {
            const int i = idx / n_act, j = idx % n_act;
            double acc = 0.0;
            for (int k = 0; k < 6 * n_ee; k++)
                acc += (double)J[k*n_act+i] * (double)J[k*n_act+j];
            A_s[i*(int)N + j] = acc;
        }
        for (int i = rank; i < n_act; i += size) {
            double rb = 0.0;
            for (int k = 0; k < 6 * n_ee; k++)
                rb += (double)J[k*n_act+i] * (double)fw[k];
            rhs_s[i] = rb;
        }
        group_sync();

        // Joint-limit prior + LM damping (in scaled space). Diagonal-only, so each
        // lane owns its own `a` with no overlap.
        for (int a = rank; a < n_act; a += size) {
            const double D_prior_raw  = (double)limit_prior_weight /
                                        ((double)half_range[a] * (double)half_range[a]);
            const double cs2          = (double)col_scale[a] * (double)col_scale[a];
            const double D_prior_s    = D_prior_raw / cs2;
            const double g_prior_s    = D_prior_raw * (double)(cfg[a] - mid[a])
                                        / (double)col_scale[a];
            A_s[a*(int)N + a] += (double)lam + D_prior_s;
            rhs_s[a]           = -(rhs_s[a] + g_prior_s);
        }
        group_sync();

        // Mask fixed joints. Lane `a` owns row a and column a; two masked lanes
        // a != a' overlap only at A[a][a'] and A[a'][a], where both write 0 — same
        // value, benign. No lane but `a` writes the diagonal A[a][a].
        for (int a = rank; a < n_act; a += size) {
            if (!s_fixed_mask[a]) continue;
            for (int j = 0; j < n_act; j++)
                A_s[a*(int)N + j] = A_s[j*(int)N + a] = 0.0;
            A_s[a*(int)N + a] = 1.0;
            rhs_s[a] = 0.0;
        }
        // Identity-pad [n_act, N) — disjoint from the masking above (top-left block
        // vs the tail), so no barrier is needed between them.
        grim::pad_tail_identity<double, N>(rank, size, n_act, A_s, rhs_s);
        group_sync();

        // Solve (overwrites A_s and rhs_s; solution in rhs_s).
        grim::tier_posv<TIER, double, N>(A_s, rhs_s, &sh_fail[slot]);
        group_sync();

        // Unscale: delta_a = p_s[a] / col_scale[a].
        float delta[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            delta[a] = (float)rhs_s[a] / col_scale[a];

        // Trust-region step clipping: use MAX error across all EEs.
        {
            float max_p = 0.0f, max_o = 0.0f;
            for (int ee = 0; ee < n_ee; ee++) {
                max_p = fmaxf(max_p, norm3(r + ee*6));
                max_o = fmaxf(max_o, norm3(r + ee*6 + 3));
            }
            float R;
            if      (max_p > 1e-2f || max_o > 0.6f)  R = 0.38f;
            else if (max_p > 1e-3f || max_o > 0.25f) R = 0.22f;
            else if (max_p > 2e-4f || max_o > 0.08f) R = 0.12f;
            else                                       R = 0.05f;
            float dnorm = 0.0f;
            for (int a = 0; a < n_act; a++) dnorm += delta[a]*delta[a];
            dnorm = sqrtf(dnorm);
            if (dnorm > R) {
                const float scale = R / (dnorm + 1e-18f);
                for (int a = 0; a < n_act; a++) delta[a] *= scale;
            }
        }

        // ── Line search: candidates with unweighted error ─────────────────
        // This search EARLY-EXITS: it scans alphas in order, tracks the running best,
        // and stops at the first new best that also achieves sufficient descent. The
        // common case therefore costs ONE FK, not five — so the tiers must differ:
        //
        //   Thread: keep the sequential scan. The early exit saves real work here and
        //           there are no idle lanes to spend on speculation.
        //   Warp/Block: evaluate every alpha in parallel (one per lane, on lanes that
        //           would otherwise idle), then REPLAY the identical scan over the
        //           precomputed errors. Each trial's error is independent of the scan,
        //           so the replay reproduces the early exit's choice EXACTLY, including
        //           which alpha wins a tie. Speculation costs nothing: those lanes have
        //           no other work, and the wall time is one FK either way.
        const float alphas[N_HJCD_ALPHAS] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
        const float suff_thresh = curr_err * (1.0f - 1e-4f);
        float best_alpha_err  = 1e30f;
        int   best_alpha_idx  = 0;

        // One trial evaluation, shared by both paths so they cannot drift apart.
        auto eval_alpha = [&](int ai) -> float {
            float cfg_trial[MAX_ACT];
            float r_trial[6 * MAX_EE];
            for (int a = 0; a < n_act; a++)
                cfg_trial[a] = clampf(cfg[a] + alphas[ai] * delta[a],
                                      s_lower[a], s_upper[a]);
            GRIM_RES(cfg_trial, r_trial);
            float e = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) e += r_trial[k] * r_trial[k];
            return e + collision_penalty(cfg_trial, T_world);
        };

        if constexpr (TIER == grim::Tier::Thread) {
            for (int ai = 0; ai < N_HJCD_ALPHAS; ai++) {
                const float e = eval_alpha(ai);
                if (e < best_alpha_err) {
                    best_alpha_err = e;
                    best_alpha_idx = ai;
                    if (e < suff_thresh) break;
                }
            }
        } else {
            for (int ai = rank; ai < N_HJCD_ALPHAS; ai += size)
                sh_ls[slot][ai] = eval_alpha(ai);
            group_sync();
            // Identical scan over the precomputed errors: same winner, same tie-break,
            // and every lane reaches the same answer, so no broadcast is needed.
            for (int ai = 0; ai < N_HJCD_ALPHAS; ai++) {
                const float e = sh_ls[slot][ai];
                if (e < best_alpha_err) {
                    best_alpha_err = e;
                    best_alpha_idx = ai;
                    if (e < suff_thresh) break;
                }
            }
        }

        float trial_cfg[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            trial_cfg[a] = clampf(cfg[a] + alphas[best_alpha_idx] * delta[a],
                                  s_lower[a], s_upper[a]);

        // Accept / reject.
        const bool improved = best_alpha_err < curr_err * (1.0f - 1e-4f);
        if (improved) {
            for (int a = 0; a < n_act; a++) cfg[a] = trial_cfg[a];
            lam = fmaxf(lam * 0.5f, 1e-10f);
            stall = 0;
        } else {
            lam = fminf(lam * 3.0f, 1e6f);
            stall++;
        }

        // ── Track all-time best ─────────────────────────────────────────
        if (best_alpha_err < best_err) {
            best_err = best_alpha_err;
            for (int a = 0; a < n_act; a++) best_cfg[a] = trial_cfg[a];
        }

        // ── Stall kick ──────────────────────────────────────────────────
        if (stall >= stall_patience) {
            const float* kick_noise = noise + ((p * n_seeds + s) * max_iter + iter) * n_act;
            for (int a = 0; a < n_act; a++) {
                if (s_fixed_mask[a]) continue;
                cfg[a] = clampf(cfg[a] + kick_noise[a] * kick_scale,
                                s_lower[a], s_upper[a]);
            }
            lam   = lambda_init;
            stall = 0;
        }
    }

    // Write best-seen config and error.
    // At the warp/block tiers every lane of the group ran the same FK on the same seed
    // and read the same solved rhs_s, so all lanes hold bit-identical state here. The
    // leader guard avoids a redundant same-value write race, not a disagreement.
    if (leader) {
        GRIM_STORE(best_cfg);
        out_err[gs] = best_err;
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handlers
// ---------------------------------------------------------------------------

static ffi::Error HjcdIkCoarseImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds,
    ffi::Buffer<ffi::DataType::F32> target_Ts,       // (n_problems, n_ee, 7)
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t k_max,
    int64_t enable_collision,
    float   collision_weight,
    float   collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    // The build fixes the robot and the obstacle counts; reject a launch that disagrees.
    if (seeds.dimensions()[2] != grim::robot::n_q ||
        target_Ts.dimensions()[target_Ts.dimensions().size() - 2] != grim::robot::n_ee ||
        world_spheres.dimensions()[0] != grim::robot::n_world_spheres ||
        world_capsules.dimensions()[0] != grim::robot::n_world_capsules ||
        world_boxes.dimensions()[0] != grim::robot::n_world_boxes ||
        world_halfspaces.dimensions()[0] != grim::robot::n_world_halfspaces)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "hjcd_ik: launch does not match the robot and obstacle counts of this build");
    if (n_problems == 0 || n_seeds == 0) return ffi::Error::Success();
    constexpr int THREADS_MAX = 128;
    const int threads  = n_seeds < THREADS_MAX ? n_seeds : THREADS_MAX;
    const int blocks_x = (n_seeds + threads - 1) / threads;

    hjcd_ik_coarse_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
        seeds.typed_data(),
        target_Ts.typed_data(),
        world_spheres.typed_data(),
        world_capsules.typed_data(),
        world_boxes.typed_data(),
        world_halfspaces.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        out->typed_data(),
        out_err->typed_data(),
        n_problems, n_seeds, static_cast<int>(k_max),
        static_cast<int>(enable_collision),
        collision_weight,
        collision_margin);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

static ffi::Error HjcdIkLmImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds,
    ffi::Buffer<ffi::DataType::F32> noise,
    ffi::Buffer<ffi::DataType::F32> target_Ts,       // (n_problems, n_ee, 7)
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter,
    int64_t stall_patience,
    int64_t early_stop,
    float   lambda_init,
    float   limit_prior_weight,
    float   kick_scale,
    float   eps_pos,
    float   eps_ori,
    int64_t enable_collision,
    float   collision_weight,
    float   collision_margin,
    int64_t tier_attr,
    int64_t block_threads,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> stop_flag)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    // The build fixes the robot and the obstacle counts; reject a launch that disagrees.
    if (seeds.dimensions()[2] != grim::robot::n_q ||
        target_Ts.dimensions()[target_Ts.dimensions().size() - 2] != grim::robot::n_ee ||
        world_spheres.dimensions()[0] != grim::robot::n_world_spheres ||
        world_capsules.dimensions()[0] != grim::robot::n_world_capsules ||
        world_boxes.dimensions()[0] != grim::robot::n_world_boxes ||
        world_halfspaces.dimensions()[0] != grim::robot::n_world_halfspaces)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "hjcd_ik: launch does not match the robot and obstacle counts of this build");
    if (n_problems == 0 || n_seeds == 0) return ffi::Error::Success();
    const int bucket = grim::solve_bucket(grim::robot::n_solved);
    const grim::Tier tier = static_cast<grim::Tier>(tier_attr);

    // Zero per-problem stop flags before kernel launch.
    cudaMemsetAsync(stop_flag->typed_data(), 0, n_problems * sizeof(int), stream);

#define GRIM_HJCD_ARGS                                                      \
        seeds.typed_data(), noise.typed_data(), \
        target_Ts.typed_data(), \
        world_spheres.typed_data(),       \
        world_capsules.typed_data(), world_boxes.typed_data(),                 \
        world_halfspaces.typed_data(),                                         \
        lower.typed_data(), upper.typed_data(),                                \
        fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(),     \
        stop_flag->typed_data(),                                               \
        n_problems, n_seeds, \
        static_cast<int>(max_iter), static_cast<int>(early_stop),              \
        lambda_init, limit_prior_weight, kick_scale,                           \
        eps_pos, eps_ori,                                                      \
        static_cast<int>(stall_patience),                                      \
        static_cast<int>(enable_collision),                                    \
        collision_weight, collision_margin

    GRIM_TIER_DISPATCH(hjcd_ik_lm_kernel, bucket, tier, n_problems, n_seeds,
                          GRIM_HJCD_THREAD_TPB, static_cast<int>(block_threads), stream,
                          GRIM_HJCD_ARGS);
#undef GRIM_HJCD_ARGS

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Handler registration
// ---------------------------------------------------------------------------

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HjcdIkCoarseFfi, HjcdIkCoarseImpl,
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out_err

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HjcdIkLmFfi, HjcdIkLmImpl,
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // out
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // out_err
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // stop_flag
