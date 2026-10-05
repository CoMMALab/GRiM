/**
 * Levenberg-Marquardt / augmented Lagrangian over a compiled least-squares problem whose
 * normal equations are BANDED: trajectories, where every cost couples a few neighbouring
 * knots. All per-problem state lives in a global workspace.
 *
 * `cost_gen.cuh` (a problem compiler's banded variant, e.g. pyroffi.costs) describes the problem as cost
 * GROUPS, each a batch of identical instances (one per knot, segment, ...) sharing one
 * lane-private residual / JVP over the instance's local variables:
 *
 *   stage_res(g, xl, pl, r)            r:  that instance's rows
 *   stage_jvp(g, xl, pl, t, r, jt)     jt: their derivative along local tangent t
 *
 * with index tables mapping each instance's local variables (vtab) and parameters (ptab)
 * to the global vectors. Row kinds and the merit are as in _cost_lm_kernel.cu.
 *
 * Launch: ONE block per problem (GRiM's single-block rule: a problem never spans blocks).
 * The block's threads split the embarrassingly parallel phases (Jacobian over (instance,
 * column), residual over instances, the scatter-add of each instance's J^T W J into the
 * lower band of H) and run the banded Cholesky, the step and the LM / AL control; phases
 * are separated by block barriers, and a finished problem's block exits. Variables in
 * `fixed` keep their initial value (their rows/columns of H are replaced by the identity).
 */

#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstdio>

#define GRIM_COST_SYNC() ((void)0)   // generated functions are lane-private

#ifndef GRIM_COST_REAL
#define GRIM_COST_REAL double
#endif
#ifndef GRIM_COST_IO
#define GRIM_COST_IO double
#endif
namespace grim::costs_gen {
using real = GRIM_COST_REAL;
using io_t = GRIM_COST_IO;
}

#include "cost_gen.cuh"

namespace ffi = xla::ffi;
namespace G = grim::costs_gen;
using G::io_t;
using G::real;

constexpr int NX = G::n_x;
constexpr int NR = G::n_r;
constexpr int BW = G::band;            // H[i][j] == 0 for |i - j| > BW
constexpr int TPB = 256;
// Workspace (reals per problem): state | x, xt, dx, g | H band (BW+1) x NX | r, rt, lam, w, s | J
constexpr size_t N_STATE = 16;
constexpr size_t OFF_X = N_STATE;
constexpr size_t OFF_H = OFF_X + 4 * (size_t)NX;
constexpr size_t OFF_R = OFF_H + (size_t)(BW + 1) * NX;
constexpr size_t OFF_J = OFF_R + 5 * (size_t)NR;
// Variable -> (instance, local slot) index for the deterministic assembly: CSR offsets
// (NX + 1) then items (n_col_total), as ints, padded to whole reals.
constexpr size_t OFF_INV = OFF_J + (size_t)G::j_size;
constexpr size_t N_INV_INTS = (size_t)NX + 1 + (size_t)G::n_col_total;
constexpr size_t WS = OFF_INV + (N_INV_INTS * sizeof(int) + sizeof(real) - 1) / sizeof(real);
// State slots (reals): scalars written by thread 0, read by the block after a barrier.
enum { S_RHO, S_MU, S_F, S_PREV_VIOL, S_DONE, S_AL_UPDATES };

#ifdef GRIM_COST_PROFILE
#define PROF_T(i) do { __syncthreads(); if (threadIdx.x == 0 && blockIdx.x == 0) { long long _t = clock64(); prof[i] += _t - prof_last; prof_last = _t; } } while (0)
#else
#define PROF_T(i) __syncthreads()
#endif

struct CostLmSettings {
    int max_iters;
    int al_iters;
    float lambda_initial;
    float tolerance;
    float al_tolerance;
    float grad_tolerance;
    int grad_start;
    float param_tolerance;
};

__device__ __forceinline__ int group_of(int item, const int* prefix, int* local)
{
    int g = 0;
    while (g + 1 < G::n_groups && item >= prefix[g + 1]) ++g;
    *local = item - prefix[g];
    return g;
}

// Block-wide reductions (every lane gets the result).
__device__ real block_sum(real v, real* sh)
{
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = TPB / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const real out = sh[0];
    __syncthreads();
    return out;
}

__device__ real block_max(real v, real* sh)
{
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = TPB / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const real out = sh[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ void gather(int g, int i, const real* x, const io_t* p, real* xl, real* pl)
{
    const int nl = G::g_nloc[g], np = G::g_nploc[g];
    const int* vt = G::vtab + G::g_vtab0[g] + i * nl;
    const int* pt = G::ptab + G::g_ptab0[g] + i * np;
    for (int m = 0; m < nl; ++m) xl[m] = x[vt[m]];
    for (int m = 0; m < np; ++m) pl[m] = (real)p[pt[m]];
}

// Lane `lane` of `stride` lanes working on one problem.
__device__ void eval_residual(const real* x, const io_t* p, real* r_out, int lane, int stride)
{
    for (int item = lane; item < G::n_inst_total; item += stride) {
        int i;
        const int g = group_of(item, G::g_inst_prefix, &i);
        real xl[G::max_nloc], pl[G::max_nploc > 0 ? G::max_nploc : 1], rr[G::max_nr];
        gather(g, i, x, p, xl, pl);
        G::stage_res(g, xl, pl, rr);
        const int nr = G::g_nr[g];
        real* dst = r_out + G::g_row0[g] + i * nr;
        for (int k = 0; k < nr; ++k) dst[k] = rr[k];
    }
}

// J[g][i] (nr x nloc, row-major): over (instance, local column).
__device__ void eval_jacobian(const real* x, const io_t* p, real* J, int lane, int stride)
{
    for (int item = lane; item < G::n_col_total; item += stride) {
        int rest;
        const int g = group_of(item, G::g_col_prefix, &rest);
        const int nl = G::g_nloc[g], nr = G::g_nr[g];
        const int i = rest / nl, c = rest % nl;
        real* dst = J + G::g_j0[g] + (size_t)i * nr * nl + c;
        if (G::fixed[G::vtab[G::g_vtab0[g] + i * nl + c]]) {
            for (int k = 0; k < nr; ++k) dst[k * nl] = real(0);
            continue;
        }
        real xl[G::max_nloc], pl[G::max_nploc > 0 ? G::max_nploc : 1], t[G::max_nloc];
        real rr[G::max_nr], jt[G::max_nr];
        gather(g, i, x, p, xl, pl);
        for (int m = 0; m < nl; ++m) t[m] = (m == c) ? real(1) : real(0);
        G::stage_jvp(g, xl, pl, t, rr, jt);
        for (int k = 0; k < nr; ++k) dst[k * nl] = jt[k];
    }
}

// H (lower band, zeroed beforehand) += sum_i J_i^T W J_i, grad += J^T W (r + s).
// Index every (instance, local slot) item by the global variable it touches, in item order.
// Built once per launch by one thread (O(n_col_total)), so the order is fixed.
__device__ void build_inverse(int* ptr, int* items)
{
    if (threadIdx.x != 0) return;
    for (int j = 0; j <= NX; ++j) ptr[j] = 0;
    for (int item = 0; item < G::n_col_total; ++item) {
        int rest;
        const int g = group_of(item, G::g_col_prefix, &rest);
        const int nl = G::g_nloc[g];
        ++ptr[G::vtab[G::g_vtab0[g] + (rest / nl) * nl + rest % nl] + 1];
    }
    for (int j = 0; j < NX; ++j) ptr[j + 1] += ptr[j];
    for (int item = 0; item < G::n_col_total; ++item) {
        int rest;
        const int g = group_of(item, G::g_col_prefix, &rest);
        const int nl = G::g_nloc[g];
        const int v = G::vtab[G::g_vtab0[g] + (rest / nl) * nl + rest % nl];
        int k = ptr[v];
        while (items[k] != -1) ++k;     // first free slot of v's run
        items[k] = item;
    }
}

// Gauss-Newton gradient and lower band of H. Each variable `ga` is owned by one thread, which
// sums its contributions in item order: no atomics, so the result is run-to-run identical
// (GRiM's fixed-order reduction rule).
__device__ void assemble(const real* J, const real* r, const real* w, const real* s, real* H,
                         real* grad, const int* inv_ptr, const int* inv_items, int lane, int stride)
{
    for (int ga = lane; ga < NX; ga += stride) {
        if (G::fixed[ga]) continue;
        real gacc = real(0);
        for (int q = inv_ptr[ga]; q < inv_ptr[ga + 1]; ++q) {
            int rest;
            const int item = inv_items[q];
            const int g = group_of(item, G::g_col_prefix, &rest);
            const int nl = G::g_nloc[g], nr = G::g_nr[g];
            const int i = rest / nl, a = rest % nl;
            const int* vt = G::vtab + G::g_vtab0[g] + i * nl;
            const real* Ji = J + G::g_j0[g] + (size_t)i * nr * nl;
            const int row0 = G::g_row0[g] + i * nr;
            for (int k = 0; k < nr; ++k) gacc += w[row0 + k] * (r[row0 + k] + s[row0 + k]) * Ji[k * nl + a];
            for (int c = 0; c < nl; ++c) {
                const int gc = vt[c];
                if (gc < ga || G::fixed[gc]) continue;
                real h = real(0);
                for (int k = 0; k < nr; ++k) h += w[row0 + k] * Ji[k * nl + a] * Ji[k * nl + c];
                H[(size_t)(gc - ga) * NX + ga] += h;
            }
        }
        grad[ga] = gacc;
    }
}

// In-place banded Cholesky of the lower band (one block): H = L L^T. False if not PD.
__device__ bool band_potrf(real* H, int* s_fail)
{
    if (threadIdx.x == 0) *s_fail = 0;
    __syncthreads();
    for (int j = 0; j < NX; ++j) {
        if (threadIdx.x == 0) {
            const real d = H[j];
            if (!(d > real(0))) *s_fail = 1;
            H[j] = sqrt(fmax(d, real(1e-300)));
        }
        __syncthreads();
        if (*s_fail) return false;
        const real ljj = H[j];
        const int m = min(BW, NX - 1 - j);
        for (int d = 1 + threadIdx.x; d <= m; d += TPB) H[(size_t)d * NX + j] /= ljj;
        __syncthreads();
        for (int e = threadIdx.x; e < m * m; e += TPB) {
            const int d1 = e / m + 1, d2 = e % m + 1;
            if (d2 < d1) continue;
            H[(size_t)(d2 - d1) * NX + j + d1] -= H[(size_t)d2 * NX + j] * H[(size_t)d1 * NX + j];
        }
        __syncthreads();
    }
    return true;
}

// Solve L L^T x = b in place (one block).
__device__ void band_potrs(const real* L, real* b)
{
    for (int j = 0; j < NX; ++j) {
        if (threadIdx.x == 0) b[j] /= L[j];
        __syncthreads();
        const int m = min(BW, NX - 1 - j);
        for (int d = 1 + threadIdx.x; d <= m; d += TPB) b[j + d] -= L[(size_t)d * NX + j] * b[j];
        __syncthreads();
    }
    for (int j = NX - 1; j >= 0; --j) {
        if (threadIdx.x == 0) {
            const int m = min(BW, NX - 1 - j);
            real acc = b[j];
            for (int d = 1; d <= m; ++d) acc -= L[(size_t)d * NX + j] * b[j + d];
            b[j] = acc / L[j];
        }
        __syncthreads();
    }
}

__device__ real merit(const real* r, const real* lam, real rho, real* sh)
{
    real f = real(0);
    for (int i = threadIdx.x; i < NR; i += TPB) {
        const int k = G::row_kind[i];
        if (k == 0) {
            f += r[i] * r[i];
        } else {
            real t = r[i] + lam[i] / rho;
            if (k == 2) t = fmax(t, real(0));
            f += real(0.5) * rho * t * t;
        }
    }
    return block_sum(f, sh);
}

__global__ void __launch_bounds__(TPB)
cost_lm_banded_kernel(const io_t* __restrict__ x0, const io_t* __restrict__ params, int param_stride,
                      io_t* __restrict__ x_out, io_t* __restrict__ cost_out,
                      io_t* __restrict__ viol_out, real* __restrict__ workspace, int n_problems,
                      CostLmSettings st)
{
    __shared__ real sh[TPB];
    __shared__ int s_fail;
    const int b = blockIdx.x;                  // problem (one block each)
    const bool active_block = b < n_problems;  // always true: the grid is n_problems blocks
    const bool leader = active_block;
    const int tid = threadIdx.x;
    const int lane = tid, stride = TPB;
    real* ws = workspace + (size_t)(active_block ? b : 0) * WS;
    real* state = ws;
    real *x = ws + OFF_X, *xt = x + NX, *dx = x + 2 * NX, *grad = x + 3 * NX, *H = ws + OFF_H;
    real *r = ws + OFF_R, *rt = r + NR, *lam = rt + NR, *w = lam + NR, *s = w + NR, *J = ws + OFF_J;
    int* inv_ptr = reinterpret_cast<int*>(ws + OFF_INV);
    int* inv_items = inv_ptr + NX + 1;
    const io_t* p = params + (size_t)(active_block ? b : 0) * param_stride;
#ifdef GRIM_COST_PROFILE
    long long prof[6] = {0, 0, 0, 0, 0, 0}, prof_last = clock64();
#endif

    if (active_block) {
        for (int j = lane; j < NX; j += stride) x[j] = (real)x0[(size_t)b * NX + j];
        for (int i = lane; i < NR; i += stride) lam[i] = real(0);
    }
    __syncthreads();
    for (int k = lane; k < G::n_col_total; k += stride) inv_items[k] = -1;
    __syncthreads();
    build_inverse(inv_ptr, inv_items);
    if (active_block) eval_residual(x, p, r, lane, stride);
    __syncthreads();
    if (leader) {
        real rho = real(1);
        if constexpr (G::has_constraints) {
            real f0 = real(0), c2 = real(0);
            for (int i = tid; i < NR; i += TPB) {
                const int k = G::row_kind[i];
                if (k == 0) f0 += r[i] * r[i];
                else { const real v = (k == 2) ? fmax(r[i], real(0)) : r[i]; c2 += v * v; }
            }
            f0 = block_sum(f0, sh);
            c2 = block_sum(c2, sh);
            rho = real(10) * fmax(real(1), fabs(f0)) / fmax(real(1), real(0.5) * c2);
        }
        const real f = merit(r, lam, rho, sh);
        if (tid == 0) {
            state[S_RHO] = rho;
            state[S_MU] = st.lambda_initial;
            state[S_F] = f;
            state[S_PREV_VIOL] = INFINITY;
            state[S_DONE] = real(0);
            state[S_AL_UPDATES] = real(0);
        }
    }
    __syncthreads();

    for (int it = 0; it < st.max_iters; ++it) {
        // Read after the previous iteration's trailing barrier: block-uniform.
        const bool live = active_block && ((volatile real*)state)[S_DONE] == real(0);
        if (!live) break;
        if (live) {
            const real rho = state[S_RHO];
            for (int i = lane; i < NR; i += stride) {
                const int k = G::row_kind[i];
                if (k == 0) { w[i] = real(1); s[i] = real(0); continue; }
                s[i] = lam[i] / rho;
                w[i] = ((k == 1) || (r[i] + s[i] > real(0))) ? real(0.5) * rho : real(0);
            }
            for (int e = lane; e < (BW + 1) * NX; e += stride) H[e] = real(0);
            for (int j = lane; j < NX; j += stride) grad[j] = real(0);
            eval_jacobian(x, p, J, lane, stride);
        }
        PROF_T(0);
        if (live) assemble(J, r, w, s, H, grad, inv_ptr, inv_items, lane, stride);
        PROF_T(1);
        if (live && leader) {
            const real mu = state[S_MU];
            real gl = real(0);
            for (int j = tid; j < NX; j += TPB) {
                if (G::fixed[j]) { H[j] = real(1); dx[j] = real(0); continue; }
                H[j] += mu;
                dx[j] = -grad[j];
                gl = fmax(gl, fabs(grad[j]));
            }
            const real gmax = block_max(gl, sh);
            const bool ok = band_potrf(H, &s_fail);
            if (ok) band_potrs(H, dx);
            real s2 = real(0), q2 = real(0);
            for (int j = tid; j < NX; j += TPB) {
                if (!ok) dx[j] = real(0);
                s2 += dx[j] * dx[j];
                q2 += x[j] * x[j];
                xt[j] = x[j] + dx[j];
            }
            const real step2 = block_sum(s2, sh), x2 = block_sum(q2, sh);
            if (tid == 0) {   // stash for the acceptance test after the trial residual
                state[8] = gmax;
                state[9] = ok ? real(1) : real(0);
                state[10] = step2;
                state[11] = x2;
            }
        }
        PROF_T(2);
        if (live) eval_residual(xt, p, rt, lane, stride);
        PROF_T(3);
        if (live && leader) {
            // Every thread loads the scalars before thread 0 may overwrite them.
            real rho = state[S_RHO], mu = state[S_MU], f = state[S_F];
            const real gmax = state[8], step2 = state[10], x2 = state[11];
            const real prev_viol = state[S_PREV_VIOL], al_updates = state[S_AL_UPDATES];
            const bool ok = state[9] != real(0);
            real new_prev_viol = prev_viol;
            __syncthreads();
            const bool grad_done = it >= st.grad_start && gmax < st.grad_tolerance;
            const real ft = merit(rt, lam, rho, sh);
            bool inner_done, done = false;
            if (ok && ft < f) {
                for (int j = tid; j < NX; j += TPB) x[j] = xt[j];
                for (int i = tid; i < NR; i += TPB) r[i] = rt[i];
                __syncthreads();
                const real rel = (f - ft) / fmax(f, real(1e-30));
                f = ft;
                mu = fmax(mu * real(0.25), real(1e-10));
                inner_done = rel < st.tolerance ||
                             sqrt(step2) < st.param_tolerance * (sqrt(x2) + st.param_tolerance);
            } else {
                mu *= real(4);
                inner_done = mu > real(1e10);
            }
            inner_done = inner_done || grad_done;
            if (inner_done) {
                if constexpr (!G::has_constraints) {
                    done = true;
                } else if (al_updates >= st.al_iters) {
                    done = true;
                } else {
                    real vl = real(0);
                    for (int i = tid; i < NR; i += TPB) {
                        const int k = G::row_kind[i];
                        if (k == 1) vl = fmax(vl, fabs(r[i]));
                        if (k == 2) vl = fmax(vl, r[i]);
                    }
                    const real viol = block_max(vl, sh);
                    for (int i = tid; i < NR; i += TPB) {
                        const int k = G::row_kind[i];
                        if (k == 1) lam[i] = fmin(fmax(lam[i] + rho * r[i], real(-1e7)), real(1e7));
                        if (k == 2) lam[i] = fmin(fmax(lam[i] + rho * r[i], real(0)), real(1e7));
                    }
                    __syncthreads();
                    if (viol < st.al_tolerance) {
                        done = true;
                    } else {
                        if (viol > real(0.5) * prev_viol) rho = fmin(rho * real(4), real(1e7));
                        new_prev_viol = viol;
                        mu = st.lambda_initial;
                        f = merit(r, lam, rho, sh);
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                state[S_AL_UPDATES] = al_updates + (inner_done ? real(1) : real(0));
                state[S_PREV_VIOL] = new_prev_viol;
                state[S_RHO] = rho;
                state[S_MU] = mu;
                state[S_F] = f;
                state[S_DONE] = done ? real(1) : real(0);
                __threadfence();
            }
        }
        PROF_T(4);
    }

#ifdef GRIM_COST_PROFILE
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("banded profile (Mcycles): jac %.1f assemble %.1f solve %.1f residual %.1f control %.1f\n",
               prof[0] * 1e-6, prof[1] * 1e-6, prof[2] * 1e-6, prof[3] * 1e-6, prof[4] * 1e-6);
#endif
    if (leader) {
        for (int j = tid; j < NX; j += TPB) x_out[(size_t)b * NX + j] = (io_t)x[j];
        real c = real(0), v = real(0);
        for (int i = tid; i < NR; i += TPB) {
            const int k = G::row_kind[i];
            if (k == 0) c += r[i] * r[i];
            if (k == 1) v = fmax(v, fabs(r[i]));
            if (k == 2) v = fmax(v, r[i]);
        }
        c = block_sum(c, sh);
        v = block_max(v, sh);
        if (tid == 0) {
            cost_out[b] = (io_t)c;
            viol_out[b] = (io_t)v;
        }
    }
}

#if defined(GRIM_COST_IO_F64)
#define GRIM_COST_IO_DT ffi::DataType::F64
#else
#define GRIM_COST_IO_DT ffi::DataType::F32
#endif

static ffi::Error CostLmBandedImpl(cudaStream_t stream, ffi::Buffer<GRIM_COST_IO_DT> x0,
                                   ffi::Buffer<GRIM_COST_IO_DT> params,
                                   ffi::ResultBuffer<GRIM_COST_IO_DT> x_out,
                                   ffi::ResultBuffer<GRIM_COST_IO_DT> cost_out,
                                   ffi::ResultBuffer<GRIM_COST_IO_DT> viol_out,
                                   ffi::ResultBuffer<ffi::DataType::U8> workspace,
                                   int32_t max_iters, int32_t al_iters, float lambda_initial,
                                   float tolerance, float al_tolerance, float grad_tolerance,
                                   int32_t grad_start, float param_tolerance)
{
    const int n_problems = static_cast<int>(x0.dimensions()[0]);
    const int param_stride = params.dimensions()[0] == 1 ? 0 : static_cast<int>(params.dimensions()[1]);
    CostLmSettings st{max_iters, al_iters, lambda_initial, tolerance, al_tolerance,
                      grad_tolerance, grad_start, param_tolerance};
    if (n_problems > 0)
        cost_lm_banded_kernel<<<n_problems, TPB, 0, stream>>>(
            x0.typed_data(), params.typed_data(), param_stride, x_out->typed_data(),
            cost_out->typed_data(), viol_out->typed_data(),
            reinterpret_cast<real*>(workspace->typed_data()), n_problems, st);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CostLmFfi, CostLmBandedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRIM_COST_IO_DT>>()   // x0      (B, n_x)
        .Arg<ffi::Buffer<GRIM_COST_IO_DT>>()   // params  (B | 1, max(n_p, 1))
        .Ret<ffi::Buffer<GRIM_COST_IO_DT>>()   // x       (B, n_x)
        .Ret<ffi::Buffer<GRIM_COST_IO_DT>>()   // cost    (B,)
        .Ret<ffi::Buffer<GRIM_COST_IO_DT>>()   // viol    (B,)
        .Ret<ffi::Buffer<ffi::DataType::U8>>()    // workspace (B, WS * sizeof(real)) bytes
        .Attr<int32_t>("max_iters")
        .Attr<int32_t>("al_iters")
        .Attr<float>("lambda_initial")
        .Attr<float>("tolerance")
        .Attr<float>("al_tolerance")
        .Attr<float>("grad_tolerance")
        .Attr<int32_t>("grad_start")
        .Attr<float>("param_tolerance"));

// Bytes of workspace one problem needs (callers size the FFI workspace output with it).
extern "C" size_t grim_cost_lm_banded_workspace_bytes() { return WS * sizeof(real); }
