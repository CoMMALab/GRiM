/**
 * Damped least-squares IK (DLS): Levenberg-Marquardt in its dual form, with XLA FFI binding.
 *
 * The step is d = -Jw^T (Jw Jw^T + lam I)^-1 W r: the same direction as LS-IK's primal
 * (Jw^T Jw + lam I)^-1 Jw^T W r without its column scaling, but solved as an m x m system,
 * m = 6 * n_ee, whatever the joint count. Globalization is LM's alone: the clamped trial
 * q + d is accepted when the weighted error drops (lam / 3), else rejected (lam * 4) and the
 * step re-solved from the same q. No line search, no trust-region radius, no best-so-far
 * tracking (acceptance is monotone, so the current q is the best seen). This is crocket's
 * ik_lm_step loop (McVAMP's inner LM form) on GRiM's residual and Jacobian.
 *
 * One seed per thread, threads over the flat (problem, seed) index so small seed counts still
 * fill blocks; each thread reads its own target. Everything in float32: the m x m Gram matrix is damped and small,
 * and the solve rate matches LS-IK's double solve (see test_ik_solvers.py).
 *
 * No collision terms: a collision row per active constraint would grow m with the scene.
 * Use ls_ik or sqp_ik for collision-aware IK.
 *
 * Built per robot (grim.motion._build): kinematics tables are compile-time constants, read
 * through robot.cuh. grim.motion.reference.ik.dls_ik is the float64 transcription.
 *
 * GRIM_FUSED_DLS (traced, one end-effector) replaces the step with crocket's traced
 * ik_lm_step: FK, Jacobian, residual, the 6x6 solve and the limit projection as straight-line
 * register code. Its orientation log is crocket's smooth one (equal to robot.cuh's to float32
 * round-off away from pi), and its joint limits and weight ratio are baked in.
 */

#include "robot.cuh"

#include "xla/ffi/api/ffi.h"

#include <cmath>

namespace ffi = xla::ffi;

#define GRIM_DLS_IK_TPB 64

/**
 * @param seeds        (n_problems, n_seeds, n_q)    initial configurations
 * @param target_Ts    (n_problems, n_ee, 7)          target poses
 * @param lower/upper  (n_q,)                         joint limits
 * @param fixed_mask   (n_q,) int32                   1 = frozen joint
 * @param out          (n_problems, n_seeds, n_q)     final configurations
 * @param out_err      (n_problems, n_seeds)           their weighted squared errors
 */
__global__
void dls_ik_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ target_Ts,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    int   n_problems, int n_seeds, int max_iter,
    float pos_weight, float ori_weight, float lambda_init,
    float eps_pos, float eps_ori)
{
    GRIM_IK_DIMS_PROLOGUE();
    constexpr int M = 6 * n_ee;

    __shared__ float s_lower     [MAX_ACT];
    __shared__ float s_upper     [MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];

    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[GRIM_ACT_SRC(i)];
        s_upper[i]      = upper[GRIM_ACT_SRC(i)];
        s_fixed_mask[i] = fixed_mask[GRIM_ACT_SRC(i)];
    }
    __syncthreads();

    const int gs = blockIdx.x * blockDim.x + threadIdx.x;
    if (gs >= n_problems * n_seeds) return;
    // Named for the GRIM_RES* macros, which read the targets from `s_target_Ts`.
    float s_target_Ts[MAX_EE * 7];
    for (int i = 0; i < n_ee * 7; i++) s_target_Ts[i] = target_Ts[(gs / n_seeds) * n_ee * 7 + i];

    float cfg[MAX_ACT];
    GRIM_LOAD_SEED(cfg);

#ifdef GRIM_FUSED_DLS
    // crocket's ik_lm_step, traced at weights (1, ori_weight / pos_weight) with the URDF limits:
    // scaling every weight by w scales J J^T and r by w, so the step at damping lam / w^2 and
    // the accept decisions are this kernel's; the reported error is the step's cost * w^2.
    // Each buffer is [q_next (n_q), cost, pos_err2, ori_err2] of the step evaluated at its input.
    // The trial is evaluated at the damping it gets if accepted (lam / 3), so after an accept
    // its q_next is already the next step: one step evaluation per iteration, at the same lam
    // as the table path. (Evaluating it at the old lam lags the damping by one iteration.)
    {
        namespace tr = grim::robot::traced;
        constexpr int N = tr::n_q;
        const float w2 = pos_weight * pos_weight;
        float q[N], buf_a[N + 3], buf_b[N + 3];
        float *cur = buf_a, *nxt = buf_b;
        grim::robot::gather_q(cfg, frz, q);
        float lam = lambda_init / w2;
        auto step = [&](const float* x, float l, float* y) {
            tr::ik_lm_step_thread<float>(x, s_target_Ts, &l, y, y + N, y + N + 1, y + N + 2, 1);
        };
        step(q, lam, cur);
        for (int iter = 0; iter < max_iter; iter++) {
            if (cur[N + 1] < eps_pos * eps_pos && cur[N + 2] < eps_ori * eps_ori) break;
            const float lam_acc = fmaxf(lam / 3.0f, 1e-7f / w2);
            step(cur, lam_acc, nxt);
            if (nxt[N] < cur[N]) {
                for (int i = 0; i < N; i++) q[i] = cur[i];
                float* t = cur; cur = nxt; nxt = t;
                lam = lam_acc;
            } else {
                lam = fminf(lam * 4.0f, 1e7f / w2);
                step(q, lam, cur);
            }
        }
        grim::robot::scatter_cfg(q, cfg);
        GRIM_STORE(cfg);
        out_err[gs] = cur[N] * w2;
        return;
    }
#endif

    float W[6] = {pos_weight, pos_weight, pos_weight, ori_weight, ori_weight, ori_weight};
    float r[6 * MAX_EE], r_trial[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    auto weighted_err = [&](const float* rr) {
        float e = 0.0f;
        for (int k = 0; k < M; k++) { const float v = rr[k] * W[k % 6]; e += v * v; }
        return e;
    };

    GRIM_RES_JAC(cfg);
    float err = weighted_err(r);
    bool jac_current = true;    // J belongs to cfg; false after an accepted step
    float lam = lambda_init;

    for (int iter = 0; iter < max_iter; iter++) {
        if (!jac_current) { GRIM_RES_JAC(cfg); jac_current = true; }

        bool conv = true;
        for (int ee = 0; ee < n_ee; ee++)
            if (norm3(r + ee*6) >= eps_pos || norm3(r + ee*6 + 3) >= eps_ori) { conv = false; break; }
        if (conv) break;

        // Weighted rows; frozen columns zeroed, so the step leaves them at the seed.
        float Jw[6 * MAX_EE * MAX_ACT], fw[6 * MAX_EE];
        for (int k = 0; k < M; k++) {
            fw[k] = r[k] * W[k % 6];
            for (int a = 0; a < n_act; a++)
                Jw[k*n_act+a] = s_fixed_mask[a] ? 0.0f : J[k*n_act+a] * W[k % 6];
        }

        // L L^T = Jw Jw^T + lam I (lower triangle), then z = (L L^T)^-1 fw.
        float L[6 * MAX_EE * 6 * MAX_EE], z[6 * MAX_EE];
        bool ok = true;
        for (int i = 0; i < M && ok; i++) {
            for (int k = 0; k <= i; k++) {
                float acc = (i == k) ? lam : 0.0f;
                for (int a = 0; a < n_act; a++) acc += Jw[i*n_act+a] * Jw[k*n_act+a];
                for (int j = 0; j < k; j++) acc -= L[i*M+j] * L[k*M+j];
                if (i == k) {
                    if (!(acc > 0.0f)) { ok = false; break; }
                    L[i*M+i] = sqrtf(acc);
                } else {
                    L[i*M+k] = acc / L[k*M+k];
                }
            }
        }
        if (ok) {
            for (int i = 0; i < M; i++) {
                float acc = fw[i];
                for (int j = 0; j < i; j++) acc -= L[i*M+j] * z[j];
                z[i] = acc / L[i*M+i];
            }
            for (int i = M - 1; i >= 0; i--) {
                float acc = z[i];
                for (int j = i + 1; j < M; j++) acc -= L[j*M+i] * z[j];
                z[i] = acc / L[i*M+i];
            }
        } else {
            for (int i = 0; i < M; i++) z[i] = 0.0f;
        }

        float trial[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float d = 0.0f;
            for (int k = 0; k < M; k++) d -= Jw[k*n_act+a] * z[k];
            trial[a] = clampf(cfg[a] + d, s_lower[a], s_upper[a]);
        }
        GRIM_RES(trial, r_trial);
        const float e_trial = weighted_err(r_trial);

        if (e_trial < err) {
            for (int a = 0; a < n_act; a++) cfg[a] = trial[a];
            for (int k = 0; k < M; k++) r[k] = r_trial[k];
            err = e_trial;
            jac_current = false;
            lam = fmaxf(lam / 3.0f, 1e-7f);
        } else {
            lam = fminf(lam * 4.0f, 1e7f);
        }
    }

    GRIM_STORE(cfg);
    out_err[gs] = err;
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error DlsIkImpl(
    cudaStream_t stream,
    GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> seeds,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter,
    float   pos_weight,
    float   ori_weight,
    float   lambda_init,
    float   eps_pos,
    float   eps_ori,
    int64_t enable_collision,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    GRIM_ROT_UPLOAD(stream);
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    if (seeds.dimensions()[2] != grim::robot::n_q || target_Ts.dimensions()[1] != grim::robot::n_ee)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "dls_ik: launch does not match the robot of this build");
    if (enable_collision || world_spheres.dimensions()[0] || world_capsules.dimensions()[0] ||
        world_boxes.dimensions()[0] || world_halfspaces.dimensions()[0])
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "dls_ik has no collision terms; use ls_ik or sqp_ik");
    if (n_problems == 0 || n_seeds == 0) return ffi::Error::Success();

    const int n = n_problems * n_seeds;
    const int threads = n < GRIM_DLS_IK_TPB ? n : GRIM_DLS_IK_TPB;
    dls_ik_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        seeds.typed_data(), target_Ts.typed_data(), lower.typed_data(), upper.typed_data(),
        fixed_mask.typed_data(), out->typed_data(), out_err->typed_data(),
        n_problems, n_seeds, static_cast<int>(max_iter),
        pos_weight, ori_weight, lambda_init, eps_pos, eps_ori);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DlsIkFfi, DlsIkImpl,
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out cfgs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out errors
);
