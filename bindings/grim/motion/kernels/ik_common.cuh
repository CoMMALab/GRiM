/**
 * Shared IK helpers for CUDA kernels.
 *
 * Provides:
 *   - xorshift32, rng_normal        (fast GPU PRNG, Box–Muller)
 *   - cross3, norm3, clampf
 *   - lbfgs_two_loop                (Nocedal two-loop L-BFGS recursion)
 *   - chol_solve                    (float32/float64 Cholesky solve)
 *   - pose_residual                 (the IK pose error; robot.cuh builds on it)
 */

#pragma once

#include "fk.cuh"

#include <cmath>
#include <cstdint>

// MAX_JOINTS / MAX_ACT / MAX_EE are the robot's own sizes, defined by the per-robot build.
#ifndef MAX_PARTICLES
#define MAX_PARTICLES 32
#endif

#ifndef MAX_LBFGS_M
#define MAX_LBFGS_M 8
#endif

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

// xorshift32 fast PRNG (GPU-friendly, no libcurand dependency).
static __device__ __forceinline__
uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Box–Muller transform: returns one standard-normal sample, advances state twice.
static __device__ __forceinline__
float rng_normal(uint32_t& state)
{
    // Two uniform samples in (0, 1] via bit-masking to 24-bit mantissa.
    const float u1 = fmaxf(1e-7f, (float)(xorshift32(state) >> 8) * (1.0f / 16777216.0f));
    const float u2 =              (float)(xorshift32(state) >> 8) * (1.0f / 16777216.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static __device__ __forceinline__
void cross3(const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static __device__ __forceinline__
float norm3(const float* __restrict__ v)
{
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static __device__ __forceinline__
float clampf(float x, float lo, float hi)
{
    return fmaxf(lo, fminf(hi, x));
}

// ---------------------------------------------------------------------------
// L-BFGS two-loop recursion (float32, in-place on p)
// ---------------------------------------------------------------------------

/**
 * Compute the L-BFGS descent direction  p = H_k^{-1} (-g)  using the
 * Nocedal two-loop recursion.
 *
 * Initialises q = -g, applies m history pairs (newest → oldest), scales by
 * the Shanno–Kettler γ, then applies the second loop (oldest → newest).
 * Result stored in p[].
 *
 * @param g           gradient at current point (n_act,)
 * @param s_buf       circular buffer of s = Δq vectors  (m_max, n_act)
 * @param y_buf       circular buffer of y = Δg vectors  (m_max, n_act)
 * @param rho_buf     rho = 1/(y^T s) scalars            (m_max,)
 * @param alpha_buf   scratch space                       (m_max,)
 * @param n_act       number of active DOF
 * @param m_max       buffer capacity (≤ MAX_LBFGS_M)
 * @param m_used      number of valid pairs stored        (0 → m_max)
 * @param newest      index of the most recently stored pair (-1 if m_used==0)
 * @param p           output: descent direction (n_act,)
 */
static __device__ void lbfgs_two_loop(
    const float* __restrict__ g,
    const float* __restrict__ s_buf,
    const float* __restrict__ y_buf,
    const float* __restrict__ rho_buf,
    float*       __restrict__ alpha_buf,
    int n_act, int m_max, int m_used, int newest,
    float* __restrict__ p)
{
    // q = -g
    for (int a = 0; a < n_act; a++) p[a] = -g[a];

    if (m_used == 0) return;   // No history → steepest descent.

    // First loop: newest → oldest
    for (int i = 0; i < m_used; i++) {
        const int idx        = (newest - i + m_max) % m_max;
        const float* s_i     = s_buf + idx * n_act;
        const float* y_i     = y_buf + idx * n_act;
        const float  rho_i   = rho_buf[idx];
        float alpha_i = 0.0f;
        for (int a = 0; a < n_act; a++) alpha_i += rho_i * s_i[a] * p[a];
        alpha_buf[i] = alpha_i;
        for (int a = 0; a < n_act; a++) p[a] -= alpha_i * y_i[a];
    }

    // Initial Hessian: H_0 = γ I  (Shanno–Kettler scaling using newest pair).
    {
        const float* s_k = s_buf + newest * n_act;
        const float* y_k = y_buf + newest * n_act;
        float sy = 0.0f, yy = 0.0f;
        for (int a = 0; a < n_act; a++) { sy += s_k[a] * y_k[a]; yy += y_k[a] * y_k[a]; }
        const float gamma = (yy > 1e-20f) ? fmaxf(1e-8f, fminf(1.0f, sy / yy)) : 1.0f;
        for (int a = 0; a < n_act; a++) p[a] *= gamma;
    }

    // Second loop: oldest → newest
    for (int i = m_used - 1; i >= 0; i--) {
        const int idx      = (newest - i + m_max) % m_max;
        const float* s_i   = s_buf + idx * n_act;
        const float* y_i   = y_buf + idx * n_act;
        const float  rho_i = rho_buf[idx];
        float beta = 0.0f;
        for (int a = 0; a < n_act; a++) beta += rho_i * y_i[a] * p[a];
        const float coeff = alpha_buf[i] - beta;
        for (int a = 0; a < n_act; a++) p[a] += s_i[a] * coeff;
    }
    // p = H_k^{-1} (-g) = descent direction.
}

// ---------------------------------------------------------------------------
// Cholesky solver (sequential, in-place)
// ---------------------------------------------------------------------------

static __device__ bool chol_solve(float* __restrict__ A,
                                  float* __restrict__ b,
                                  int n)
{
    for (int k = 0; k < n; k++) {
        float s = A[k*n + k];
        for (int p = 0; p < k; p++) { float lkp = A[k*n+p]; s -= lkp*lkp; }
        if (s <= 0.0f) {
            for (int i = 0; i < n; i++) b[i] = 0.0f;
            return false;
        }
        float lkk = sqrtf(s);
        A[k*n + k] = lkk;
        for (int i = k+1; i < n; i++) {
            float t = A[i*n + k];
            for (int p = 0; p < k; p++) t -= A[i*n+p] * A[k*n+p];
            A[i*n + k] = t / lkk;
        }
        for (int j = k+1; j < n; j++) A[k*n + j] = 0.0f;
    }

    // Forward substitution: L y = b
    float y[MAX_ACT];
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int p = 0; p < i; p++) s -= A[i*n+p] * y[p];
        y[i] = s / A[i*n + i];
    }

    // Backward substitution: L^T x = y
    for (int i = n-1; i >= 0; i--) {
        float s = y[i];
        for (int p = i+1; p < n; p++) s -= A[p*n + i] * b[p];
        b[i] = s / A[i*n + i];
    }
    return true;
}

static __device__ bool chol_solve(double* __restrict__ A,
                                  double* __restrict__ b,
                                  int n)
{
    for (int k = 0; k < n; k++) {
        double s = A[k*n + k];
        for (int p = 0; p < k; p++) { double lkp = A[k*n+p]; s -= lkp*lkp; }
        if (s <= 0.0) {
            for (int i = 0; i < n; i++) b[i] = 0.0;
            return false;
        }
        double lkk = sqrt(s);
        A[k*n + k] = lkk;
        for (int i = k+1; i < n; i++) {
            double t = A[i*n + k];
            for (int p = 0; p < k; p++) t -= A[i*n+p] * A[k*n+p];
            A[i*n + k] = t / lkk;
        }
        for (int j = k+1; j < n; j++) A[k*n + j] = 0.0;
    }

    // Forward substitution: L y = b
    double y[MAX_ACT];
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int p = 0; p < i; p++) s -= A[i*n+p] * y[p];
        y[i] = s / A[i*n + i];
    }

    // Backward substitution: L^T x = y
    for (int i = n-1; i >= 0; i--) {
        double s = y[i];
        for (int p = i+1; p < n; p++) s -= A[p*n + i] * b[p];
        b[i] = s / A[i*n + i];
    }
    return true;
}

// ---------------------------------------------------------------------------
// IK residual and geometric Jacobian
// ---------------------------------------------------------------------------

// r = [p_ee - p_tgt, log(q_ee * q_tgt^-1)] for poses stored as [w, x, y, z, tx, ty, tz].
static __device__ __forceinline__ void pose_residual(
    const float* __restrict__ T_ee,
    const float* __restrict__ target_T,
    float*       __restrict__ r)
{
    r[0] = T_ee[4] - target_T[4];
    r[1] = T_ee[5] - target_T[5];
    r[2] = T_ee[6] - target_T[6];

    const float q_tgt_inv[4] = { target_T[0], -target_T[1], -target_T[2], -target_T[3] };
    float q_err[4];
    quat_mul(T_ee, q_tgt_inv, q_err);
    if (q_err[0] < 0.0f) {
        q_err[0] = -q_err[0]; q_err[1] = -q_err[1];
        q_err[2] = -q_err[2]; q_err[3] = -q_err[3];
    }
    const float sin_half = sqrtf(q_err[1]*q_err[1] + q_err[2]*q_err[2] + q_err[3]*q_err[3]);
    if (sin_half > 1e-6f) {
        const float inv_sin = 2.0f * atan2f(sin_half, q_err[0]) / sin_half;
        r[3] = q_err[1] * inv_sin;
        r[4] = q_err[2] * inv_sin;
        r[5] = q_err[3] * inv_sin;
    } else {
        r[3] = 2.0f * q_err[1];
        r[4] = 2.0f * q_err[2];
        r[5] = 2.0f * q_err[3];
    }
}
