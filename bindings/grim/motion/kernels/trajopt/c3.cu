/**
 * Tier-4 (contact-implicit trajopt) numerics in float64: the problem-independent inner
 * solve of multiple-shooting C3+ (see benchmarks/c3_push/c3.py, the reference oracle).
 *
 * Sizes are compile-time (one build per problem shape, cached by _c3_cuda.py):
 *   C3_NX  state dim, C3_NU control dim, C3_NL complementarity pairs per knot,
 *   C3_NT  rows of the task residual per knot (task + trust-region rows).
 * The horizon N and the batch B are runtime.
 *
 * Per knot k the caller hands the linearization c3.py builds in JAX:
 *   A (NX,NX)  Bt (NX,M)  E (NL,NX)  Ht (NL,M)  Eta (NL)  Lam (NL)  c (NX) defects
 *   Jx (NT,NX) Ju (NT,M)  r (NT)     task residual rows and their Jacobians wrt (dx_k, w_k),
 * with M = NU + NL, w_k = [du_k | dlam_k]. The ADMM consensus rows are added here:
 *   s (Lam + dlam + tl),  s sqrt(w_eta) (Eta + E dx + Ht w + te),  s = sqrt(rho / 2).
 * Wave 2 fills (Jx, Ju, r, A, Bt, E, Ht, Eta) from emitted device functions; this file never
 * sees problem physics.
 *
 * Matrices are row-major. Symmetric factors go through GLASS's thread-scope Cholesky
 * (glass::thread::potrf/potrs/trsv, column-major lower: identical storage for a full
 * symmetric matrix). Parallelism is GATO-style: one CUDA block per problem, threads strided
 * over knots / segments; only the Riccati sweep and the coarse solve are single-thread.
 */

#include "glass.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstdint>

namespace ffi = xla::ffi;
using T = double;

#ifndef C3_NX
#define C3_NX 10
#endif
#ifndef C3_NU
#define C3_NU 2
#endif
#ifndef C3_NL
#define C3_NL 10
#endif
#ifndef C3_NT
#define C3_NT 26
#endif
constexpr int NX = C3_NX, NU = C3_NU, NL = C3_NL, NT = C3_NT;
constexpr int M = NU + NL, NS = NX + M;
constexpr int TPB = 128;

// ─────────────────────────────────────────────────────────────── small dense helpers ──
template <int R, int C>
__device__ __forceinline__ void mv(const T* A, const T* x, T* y, T alpha = 1, T beta = 0)
{   // y = alpha A x + beta y
    for (int i = 0; i < R; ++i) {
        T acc = 0;
        for (int j = 0; j < C; ++j) acc += A[i * C + j] * x[j];
        y[i] = beta == T(0) ? alpha * acc : alpha * acc + beta * y[i];  // y may be uninitialised
    }
}
template <int R, int C>
__device__ __forceinline__ void mtv(const T* A, const T* x, T* y, T alpha = 1, T beta = 0)
{   // y = alpha A^T x + beta y
    for (int j = 0; j < C; ++j) {
        T acc = 0;
        for (int i = 0; i < R; ++i) acc += A[i * C + j] * x[i];
        y[j] = beta == T(0) ? alpha * acc : alpha * acc + beta * y[j];
    }
}
template <int N>
__device__ __forceinline__ void chol(T* A) { glass::thread::potrf<T, N>(A); }
template <int N>
__device__ __forceinline__ void cho_solve(const T* L, T* b) { glass::thread::potrs<T, N>(L, b); }
template <int N>
__device__ __forceinline__ void tri_lower(const T* L, T* x) { glass::thread::trsv<T, N>(L, x); }
template <int N>
__device__ __forceinline__ void tri_lower_t(const T* L, T* x)
{
    glass::thread::trsv<T, N, glass::FillMode::Lower, glass::Diag::NonUnit, true>(L, x);
}

// ────────────────────────────────────────────────────────────── the linearization ──
struct Lin {
    const T *A, *Bt, *E, *Ht, *Eta, *Lam, *c, *Jx, *Ju, *r;
};
__device__ __forceinline__ Lin lin_of(const Lin& g, int b, int N)
{
    const size_t n = N;
    return Lin{g.A + b * n * NX * NX, g.Bt + b * n * NX * M, g.E + b * n * NL * NX,
               g.Ht + b * n * NL * M, g.Eta + b * n * NL, g.Lam + b * n * NL, g.c + b * n * NX,
               g.Jx + b * n * NT * NX, g.Ju + b * n * NT * M, g.r + b * n * NT};
}

// Gauss-Newton Hessian of stage k over s = [dx | w]: J^T J with the consensus rows.
__device__ void stage_hess(const Lin& L, int k, T s2, T s2w, T* G /*NS*NS*/)
{
    const T* Jx = L.Jx + (size_t)k * NT * NX;
    const T* Ju = L.Ju + (size_t)k * NT * M;
    const T* E = L.E + (size_t)k * NL * NX;
    const T* Ht = L.Ht + (size_t)k * NL * M;
    auto jrow = [&](int r, int i) { return i < NX ? Jx[r * NX + i] : Ju[r * M + i - NX]; };
    auto erow = [&](int r, int i) { return i < NX ? E[r * NX + i] : Ht[r * M + i - NX]; };
    for (int i = 0; i < NS; ++i)
        for (int j = i; j < NS; ++j) {
            T acc = 0;
            for (int r = 0; r < NT; ++r) acc += jrow(r, i) * jrow(r, j);
            T ae = 0;
            for (int r = 0; r < NL; ++r) ae += erow(r, i) * erow(r, j);
            acc += s2w * ae;
            if (i == j && i >= NX + NU) acc += s2;          // s (lam + dlam) rows
            G[i * NS + j] = acc;
            G[j * NS + i] = acc;
        }
}

// Gradient J^T r at s = 0 for consensus targets (tl, te) (null = zero).
__device__ void stage_grad(const Lin& L, int k, T s2, T s2w, const T* tl, const T* te, T* g)
{
    const T* Jx = L.Jx + (size_t)k * NT * NX;
    const T* Ju = L.Ju + (size_t)k * NT * M;
    const T* r = L.r + (size_t)k * NT;
    const T* E = L.E + (size_t)k * NL * NX;
    const T* Ht = L.Ht + (size_t)k * NL * M;
    T e[NL];
    for (int l = 0; l < NL; ++l) e[l] = L.Eta[k * NL + l] + (te ? te[k * NL + l] : T(0));
    mtv<NT, NX>(Jx, r, g);
    mtv<NT, M>(Ju, r, g + NX);
    mtv<NL, NX>(E, e, g, s2w, 1);
    mtv<NL, M>(Ht, e, g + NX, s2w, 1);
    for (int l = 0; l < NL; ++l) g[NX + NU + l] += s2 * (L.Lam[k * NL + l] + (tl ? tl[k * NL + l] : T(0)));
}

// ───────────────────────────────────────────────────────────────────── Riccati ──
// c3.riccati_factor / riccati_solve: stage costs folded into the stage that produces the
// state, no terminal term, dx_0 = 0, dx_{k+1} = A dx + Bt w + c. One thread, N sequential.
struct RicWork {
    T *Lq, *K, *Pn, *kff;   // (N,M,M) chol(Quu), (N,M,NX), (N,NX,NX) P_{k+1}, (N,M)
    T *idg;                 // (N,M) reciprocal diagonal of chol(Quu)
    T *Qinv;                // (N,M,M) Quu^-1: the solve's feedforward is then a parallel matvec
    T *Gall, *gall;         // stage Hessians (N,NS,NS) / gradients (N,NS), hoisted out of the sweep
};

__device__ void ric_factor(const Lin& L, int N, T s2, T s2w, const RicWork& w, T reg = 1e-9)
{
    T P[NX * NX] = {};
    T G[NS * NS], PA[NX * NX], PB[NX * M], Quu[M * M], Qux[M * NX], Pnew[NX * NX];
    for (int k = N - 1; k >= 0; --k) {
        for (int i = 0; i < NX * NX; ++i) w.Pn[(size_t)k * NX * NX + i] = P[i];
        stage_hess(L, k, s2, s2w, G);
        const T* a = L.A + (size_t)k * NX * NX;
        const T* bt = L.Bt + (size_t)k * NX * M;
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NX; ++j) {
                T acc = 0;
                for (int l = 0; l < NX; ++l) acc += P[i * NX + l] * a[l * NX + j];
                PA[i * NX + j] = acc;
            }
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < M; ++j) {
                T acc = 0;
                for (int l = 0; l < NX; ++l) acc += P[i * NX + l] * bt[l * M + j];
                PB[i * M + j] = acc;
            }
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j) {
                T acc = G[(NX + i) * NS + NX + j] + (i == j ? reg : T(0));
                for (int l = 0; l < NX; ++l) acc += bt[l * M + i] * PB[l * M + j];
                Quu[i * M + j] = acc;
            }
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < NX; ++j) {
                T acc = G[j * NS + NX + i];                       // S^T
                for (int l = 0; l < NX; ++l) acc += bt[l * M + i] * PA[l * NX + j];
                Qux[i * NX + j] = acc;
            }
        chol<M>(Quu);
        T* Lk = w.Lq + (size_t)k * M * M;
        for (int i = 0; i < M * M; ++i) Lk[i] = Quu[i];
        T* Kk = w.K + (size_t)k * M * NX;
        T col[M];
        for (int j = 0; j < NX; ++j) {
            for (int i = 0; i < M; ++i) col[i] = Qux[i * NX + j];
            cho_solve<M>(Quu, col);
            for (int i = 0; i < M; ++i) Kk[i * NX + j] = -col[i];
        }
        // P = Qxx + Qux^T K,  Qxx = Q + A^T P A
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NX; ++j) {
                T acc = G[i * NS + j];
                for (int l = 0; l < NX; ++l) acc += a[l * NX + i] * PA[l * NX + j];
                for (int l = 0; l < M; ++l) acc += Qux[l * NX + i] * Kk[l * NX + j];
                Pnew[i * NX + j] = acc;
            }
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NX; ++j) P[i * NX + j] = 0.5 * (Pnew[i * NX + j] + Pnew[j * NX + i]);
    }
}

__device__ void ric_solve(const Lin& L, int N, T s2, T s2w, const T* tl, const T* te,
                          const RicWork& w, T* dX, T* W)
{
    T p[NX] = {}, pc[NX], g[NS], qu[M], pn[NX];
    for (int k = N - 1; k >= 0; --k) {
        const T* a = L.A + (size_t)k * NX * NX;
        const T* bt = L.Bt + (size_t)k * NX * M;
        const T* Pk = w.Pn + (size_t)k * NX * NX;
        for (int i = 0; i < NX; ++i) pc[i] = p[i];
        mv<NX, NX>(Pk, L.c + (size_t)k * NX, pc, 1, 1);
        stage_grad(L, k, s2, s2w, tl, te, g);
        for (int i = 0; i < M; ++i) qu[i] = g[NX + i];
        mtv<NX, M>(bt, pc, qu, 1, 1);
        T* kf = w.kff + (size_t)k * M;
        for (int i = 0; i < M; ++i) kf[i] = qu[i];
        cho_solve<M>(w.Lq + (size_t)k * M * M, kf);
        for (int i = 0; i < M; ++i) kf[i] = -kf[i];
        for (int i = 0; i < NX; ++i) pn[i] = g[i];
        mtv<NX, NX>(a, pc, pn, 1, 1);
        mtv<M, NX>(w.K + (size_t)k * M * NX, qu, pn, 1, 1);
        for (int i = 0; i < NX; ++i) p[i] = pn[i];
    }
    T dx[NX] = {}, wk[M], nx_[NX];
    for (int k = 0; k < N; ++k) {
        const T* a = L.A + (size_t)k * NX * NX;
        const T* bt = L.Bt + (size_t)k * NX * M;
        for (int i = 0; i < M; ++i) wk[i] = w.kff[(size_t)k * M + i];
        mv<M, NX>(w.K + (size_t)k * M * NX, dx, wk, 1, 1);
        for (int i = 0; i < NX; ++i) dX[(size_t)k * NX + i] = dx[i];
        for (int i = 0; i < M; ++i) W[(size_t)k * M + i] = wk[i];
        for (int i = 0; i < NX; ++i) nx_[i] = L.c[(size_t)k * NX + i];
        mv<NX, NX>(a, dx, nx_, 1, 1);
        mv<NX, M>(bt, wk, nx_, 1, 1);
        for (int i = 0; i < NX; ++i) dx[i] = nx_[i];
    }
}

// Cholesky of a full symmetric M x M matrix in shared memory, cooperatively, into GLASS's
// column-major-lower layout (L(i,j) at A[j*M+i]) so glass::thread::potrs still reads it.
// FP64 sqrt/div are slow software sequences on GA10x (one 12x12 thread-scope potrf cost
// ~59k cycles), so only M of each happen and the solves below multiply by idg instead.
__device__ void chol_coop(T* A, T* idg)
{
    const int t = threadIdx.x, nt = blockDim.x;
    for (int j = 0; j < M; ++j) {
        if (t == 0) { const T d = sqrt(A[j * M + j]); A[j * M + j] = d; idg[j] = T(1) / d; }
        __syncthreads();
        for (int i = j + 1 + t; i < M; i += nt) A[j * M + i] *= idg[j];
        __syncthreads();
        const int m = M - j - 1;
        for (int e = t; e < m * m; e += nt) {
            const int r = j + 1 + e / m, c = j + 1 + e % m;
            if (c <= r) A[c * M + r] -= A[j * M + r] * A[j * M + c];
        }
        __syncthreads();
    }
}
// x <- (L L^T)^-1 x with L from chol_coop (division-free).
__device__ __forceinline__ void cho_solve_idg(const T* A, const T* idg, T* x)
{
    for (int i = 0; i < M; ++i) {
        T acc = x[i];
        for (int j = 0; j < i; ++j) acc -= A[j * M + i] * x[j];
        x[i] = acc * idg[i];
    }
    for (int i = M - 1; i >= 0; --i) {
        T acc = x[i];
        for (int j = i + 1; j < M; ++j) acc -= A[i * M + j] * x[j];
        x[i] = acc * idg[i];
    }
}

// Block-cooperative Riccati: the same recursion with every matrix entry of a step spread
// over the block's threads (the knot loop stays sequential). Shared scratch per block.
struct RicShared {
    T P[NX * NX], G[NS * NS], PA[NX * NX], PB[NX * M], Quu[M * M], Qux[M * NX], Pnew[NX * NX], idg[M];
    T p[NX], pc[NX], g[NS], qu[M], kf[M], dx[NX], wk[M];
};

__device__ __forceinline__ T stage_hess_ij(const Lin& L, int k, T s2, T s2w, int i, int j)
{
    const T* Jx = L.Jx + (size_t)k * NT * NX;
    const T* Ju = L.Ju + (size_t)k * NT * M;
    const T* E = L.E + (size_t)k * NL * NX;
    const T* Ht = L.Ht + (size_t)k * NL * M;
    T acc = 0, ae = 0;
    for (int r = 0; r < NT; ++r)
        acc += (i < NX ? Jx[r * NX + i] : Ju[r * M + i - NX]) * (j < NX ? Jx[r * NX + j] : Ju[r * M + j - NX]);
    for (int r = 0; r < NL; ++r)
        ae += (i < NX ? E[r * NX + i] : Ht[r * M + i - NX]) * (j < NX ? E[r * NX + j] : Ht[r * M + j - NX]);
    acc += s2w * ae;
    if (i == j && i >= NX + NU) acc += s2;
    return acc;
}

__device__ __forceinline__ T stage_grad_i(const Lin& L, int k, T s2, T s2w, const T* tl, const T* te, int i)
{
    const T* Jx = L.Jx + (size_t)k * NT * NX;
    const T* Ju = L.Ju + (size_t)k * NT * M;
    const T* r = L.r + (size_t)k * NT;
    const T* E = L.E + (size_t)k * NL * NX;
    const T* Ht = L.Ht + (size_t)k * NL * M;
    T acc = 0;
    for (int q = 0; q < NT; ++q) acc += (i < NX ? Jx[q * NX + i] : Ju[q * M + i - NX]) * r[q];
    T ae = 0;
    for (int l = 0; l < NL; ++l)
        ae += (i < NX ? E[l * NX + i] : Ht[l * M + i - NX]) *
              (L.Eta[k * NL + l] + (te ? te[k * NL + l] : T(0)));
    acc += s2w * ae;
    if (i >= NX + NU) acc += s2 * (L.Lam[k * NL + i - NX - NU] + (tl ? tl[k * NL + i - NX - NU] : T(0)));
    return acc;
}

__device__ void ric_factor_coop(const Lin& L, int N, T s2, T s2w, const RicWork& w, RicShared& S,
                                T reg = 1e-9)
{
    const int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < NX * NX; i += nt) S.P[i] = 0;
    for (int e = t; e < N * NS * NS; e += nt)        // every stage Hessian, in parallel
        w.Gall[e] = stage_hess_ij(L, e / (NS * NS), s2, s2w, (e / NS) % NS, e % NS);
    __syncthreads();
    for (int k = N - 1; k >= 0; --k) {
        const T* a = L.A + (size_t)k * NX * NX;
        const T* bt = L.Bt + (size_t)k * NX * M;
        for (int i = t; i < NX * NX; i += nt) w.Pn[(size_t)k * NX * NX + i] = S.P[i];
        for (int e = t; e < NS * NS; e += nt) S.G[e] = w.Gall[(size_t)k * NS * NS + e];
        for (int e = t; e < NX * NX; e += nt) {
            const int i = e / NX, j = e % NX;
            T acc = 0;
            for (int l = 0; l < NX; ++l) acc += S.P[i * NX + l] * a[l * NX + j];
            S.PA[e] = acc;
        }
        for (int e = t; e < NX * M; e += nt) {
            const int i = e / M, j = e % M;
            T acc = 0;
            for (int l = 0; l < NX; ++l) acc += S.P[i * NX + l] * bt[l * M + j];
            S.PB[e] = acc;
        }
        __syncthreads();
        for (int e = t; e < M * M; e += nt) {
            const int i = e / M, j = e % M;
            T acc = S.G[(NX + i) * NS + NX + j] + (i == j ? reg : T(0));
            for (int l = 0; l < NX; ++l) acc += bt[l * M + i] * S.PB[l * M + j];
            S.Quu[e] = acc;
        }
        for (int e = t; e < M * NX; e += nt) {
            const int i = e / NX, j = e % NX;
            T acc = S.G[j * NS + NX + i];
            for (int l = 0; l < NX; ++l) acc += bt[l * M + i] * S.PA[l * NX + j];
            S.Qux[e] = acc;
        }
        __syncthreads();
        chol_coop(S.Quu, S.idg);
        T* Lk = w.Lq + (size_t)k * M * M;
        for (int i = t; i < M * M; i += nt) Lk[i] = S.Quu[i];
        for (int i = t; i < M; i += nt) w.idg[(size_t)k * M + i] = S.idg[i];
        T* Kk = w.K + (size_t)k * M * NX;
        if (t < NX) {
            T col[M];
            for (int i = 0; i < M; ++i) col[i] = S.Qux[i * NX + t];
            cho_solve_idg(S.Quu, S.idg, col);
            for (int i = 0; i < M; ++i) Kk[i * NX + t] = -col[i];
        } else if (t < NX + M) {
            const int j = t - NX;
            T col[M];
            for (int i = 0; i < M; ++i) col[i] = (i == j);
            cho_solve_idg(S.Quu, S.idg, col);
            for (int i = 0; i < M; ++i) w.Qinv[(size_t)k * M * M + i * M + j] = col[i];
        }
        __syncthreads();
        for (int e = t; e < NX * NX; e += nt) {
            const int i = e / NX, j = e % NX;
            T acc = S.G[i * NS + j];
            for (int l = 0; l < NX; ++l) acc += a[l * NX + i] * S.PA[l * NX + j];
            for (int l = 0; l < M; ++l) acc += S.Qux[l * NX + i] * Kk[l * NX + j];
            S.Pnew[e] = acc;
        }
        __syncthreads();
        for (int e = t; e < NX * NX; e += nt) S.P[e] = 0.5 * (S.Pnew[e] + S.Pnew[(e % NX) * NX + e / NX]);
        __syncthreads();
    }
}

__device__ void ric_solve_coop(const Lin& L, int N, T s2, T s2w, const T* tl, const T* te,
                               const RicWork& w, RicShared& S, T* dX, T* W)
{
    const int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < NX; i += nt) S.p[i] = 0;
    for (int e = t; e < N * NS; e += nt) w.gall[e] = stage_grad_i(L, e / NS, s2, s2w, tl, te, e % NS);
    __syncthreads();
    for (int k = N - 1; k >= 0; --k) {
        const T* a = L.A + (size_t)k * NX * NX;
        const T* bt = L.Bt + (size_t)k * NX * M;
        const T* Pk = w.Pn + (size_t)k * NX * NX;
        const T* ck = L.c + (size_t)k * NX;
        for (int i = t; i < NX; i += nt) {
            T acc = S.p[i];
            for (int j = 0; j < NX; ++j) acc += Pk[i * NX + j] * ck[j];
            S.pc[i] = acc;
        }
        for (int i = t; i < NS; i += nt) S.g[i] = w.gall[(size_t)k * NS + i];
        __syncthreads();
        for (int i = t; i < M; i += nt) {
            T acc = S.g[NX + i];
            for (int l = 0; l < NX; ++l) acc += bt[l * M + i] * S.pc[l];
            S.qu[i] = acc;
        }
        __syncthreads();
        const T* Qi = w.Qinv + (size_t)k * M * M;
        for (int i = t; i < M; i += nt) {
            T acc = 0;
            for (int j = 0; j < M; ++j) acc += Qi[i * M + j] * S.qu[j];
            S.kf[i] = acc;
        }
        const T* Kk = w.K + (size_t)k * M * NX;
        for (int i = t; i < NX; i += nt) {
            T acc = S.g[i];
            for (int l = 0; l < NX; ++l) acc += a[l * NX + i] * S.pc[l];
            for (int l = 0; l < M; ++l) acc += Kk[l * NX + i] * S.qu[l];
            S.Pnew[i] = acc;
        }
        __syncthreads();
        for (int i = t; i < M; i += nt) w.kff[(size_t)k * M + i] = -S.kf[i];
        for (int i = t; i < NX; i += nt) S.p[i] = S.Pnew[i];
        __syncthreads();
    }
    for (int i = t; i < NX; i += nt) S.dx[i] = 0;
    __syncthreads();
    for (int k = 0; k < N; ++k) {
        const T* a = L.A + (size_t)k * NX * NX;
        const T* bt = L.Bt + (size_t)k * NX * M;
        const T* Kk = w.K + (size_t)k * M * NX;
        for (int i = t; i < M; i += nt) {
            T acc = w.kff[(size_t)k * M + i];
            for (int j = 0; j < NX; ++j) acc += Kk[i * NX + j] * S.dx[j];
            S.wk[i] = acc;
            W[(size_t)k * M + i] = acc;
        }
        for (int i = t; i < NX; i += nt) dX[(size_t)k * NX + i] = S.dx[i];
        __syncthreads();
        for (int i = t; i < NX; i += nt) {
            T acc = L.c[(size_t)k * NX + i];
            for (int j = 0; j < NX; ++j) acc += a[i * NX + j] * S.dx[j];
            for (int j = 0; j < M; ++j) acc += bt[i * M + j] * S.wk[j];
            S.Pnew[i] = acc;
        }
        __syncthreads();
        for (int i = t; i < NX; i += nt) S.dx[i] = S.Pnew[i];
        __syncthreads();
    }
}

// (L1, E1) at knot k from a QP solution: Lam + dlam, Eta + E dx + Ht w.
__device__ void lam_eta_at(const Lin& L, int k, const T* dX, const T* W, T* L1, T* E1)
{
    const T* w = W + (size_t)k * M;
    for (int l = 0; l < NL; ++l) L1[l] = L.Lam[k * NL + l] + w[NU + l];
    for (int l = 0; l < NL; ++l) E1[l] = L.Eta[k * NL + l];
    mv<NL, NX>(L.E + (size_t)k * NL * NX, dX + (size_t)k * NX, E1, 1, 1);
    mv<NL, M>(L.Ht + (size_t)k * NL * M, w, E1, 1, 1);
}

// ─────────────────────────────────────────────────────────────────── projection ──
// c3.project: nearest point of {a>=0, b>=0, a*b=0}, eta distances weighted by w_eta.
__device__ __forceinline__ void project_pair(T a, T b, T w_eta, T& pa, T& pb)
{
    const T a0 = fmax(a, T(0)), b0 = fmax(b, T(0));
    const bool keep = (a - a0) * (a - a0) + w_eta * b * b <= a * a + w_eta * (b - b0) * (b - b0);
    pa = keep ? a0 : T(0);
    pb = keep ? T(0) : b0;
}
__device__ __forceinline__ T box_clamp(T x, T lo, T hi) { return fmin(fmax(x, lo), hi); }

// ───────────────────────────────────────────────────────── block reduction helper ──
__device__ T block_sum(T v, T* sh)
{   // warp shuffles, then one partial per warp: two block barriers instead of log2(TPB) + 2
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    T out = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) out += sh[i];
    return out;
}

// ──────────────────────────────────────────────── Schur-complement PCG (GATO-style) ──
// c3._schur_pcg_qp with proximal centering, plus the two-level preconditioner:
//   M^-1 = sum_i R_i^T S_i^-1 R_i + R_0^T (R_0 S R_0^T)^-1 R_0
// segments of L knots (block-Thomas, one thread per segment) and a piecewise-constant
// coarse space (block-tridiagonal over segments, one thread). precond: 0 block-Jacobi,
// 1 two-level.
struct PcgWork {
    T *H, *sig, *Sjj, *Slo;          // (N,NS,NS) (N) (N,NX,NX) S_{j,j-1}
    T *Pf, *Pe;                      // per-knot chol of Sjj (Jacobi) / segment factors F, E
    T *Cf, *Ce;                      // coarse factors (nseg,NX,NX) each
    T *Hg, *v, *res, *z, *d, *Sd, *tmp, *uc;
    T *Di, *Wf, *CDi, *CW, *ty;      // explicit block-LDL^T inverses / multipliers; C^T v scratch
};

// Inverse of an NX x NX SPD matrix from its GLASS Cholesky factor (thread scope).
__device__ void inv_from_chol(const T* F, T* out)
{
    T col[NX];
    for (int j = 0; j < NX; ++j) {
        for (int i = 0; i < NX; ++i) col[i] = (i == j);
        cho_solve<NX>(F, col);
        for (int i = 0; i < NX; ++i) out[i * NX + j] = col[i];
    }
}
// C = A B for NX x NX row-major.
__device__ __forceinline__ void mm_nx(const T* A, const T* B, T* C)
{
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NX; ++j) {
            T acc = 0;
            for (int l = 0; l < NX; ++l) acc += A[i * NX + l] * B[l * NX + j];
            C[i * NX + j] = acc;
        }
}

__device__ void schur_setup(const Lin& L, int N, T s2, T s2w, T reg, int precond, int seg,
                            const PcgWork& w)
{
    T G[NS * NS];
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        stage_hess(L, k, s2, s2w, G);
        T tr = 0;
        for (int i = 0; i < NS; ++i) tr += G[i * NS + i];
        const T sg = reg * tr / NS + reg;
        w.sig[k] = sg;
        for (int i = 0; i < NS; ++i) G[i * NS + i] += sg;
        chol<NS>(G);
        T* Hk = w.H + (size_t)k * NS * NS;
        T col[NS];
        for (int j = 0; j < NS; ++j) {
            for (int i = 0; i < NS; ++i) col[i] = (i == j);
            cho_solve<NS>(G, col);
            for (int i = 0; i < NS; ++i) Hk[i * NS + j] = col[i];
        }
    }
    __syncthreads();
    // Sjj = H_j[:NX,:NX] + F_{j-1} H_{j-1} F_{j-1}^T,  S_{j,j-1} = -F_{j-1} H_{j-1}[:, :NX]
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const T* Hj = w.H + (size_t)j * NS * NS;
        T* S = w.Sjj + (size_t)j * NX * NX;
        for (int a = 0; a < NX; ++a)
            for (int b = 0; b < NX; ++b) S[a * NX + b] = Hj[a * NS + b];
        if (j > 0) {
            const T* Hp = w.H + (size_t)(j - 1) * NS * NS;
            const T* A = L.A + (size_t)(j - 1) * NX * NX;
            const T* Bt = L.Bt + (size_t)(j - 1) * NX * M;
            auto F = [&](int a, int i) { return i < NX ? A[a * NX + i] : Bt[a * M + i - NX]; };
            T FH[NX * NS];
            for (int a = 0; a < NX; ++a)
                for (int i = 0; i < NS; ++i) {
                    T acc = 0;
                    for (int l = 0; l < NS; ++l) acc += F(a, l) * Hp[l * NS + i];
                    FH[a * NS + i] = acc;
                }
            T* Sl = w.Slo + (size_t)j * NX * NX;
            for (int a = 0; a < NX; ++a)
                for (int b = 0; b < NX; ++b) {
                    T acc = 0;
                    for (int l = 0; l < NS; ++l) acc += FH[a * NS + l] * F(b, l);
                    S[a * NX + b] += acc;
                    Sl[a * NX + b] = -FH[a * NS + b];
                }
        }
    }
    __syncthreads();
    if (precond == 0) {
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            T* P = w.Pf + (size_t)j * NX * NX;
            for (int i = 0; i < NX * NX; ++i) P[i] = w.Sjj[(size_t)j * NX * NX + i];
            chol<NX>(P);
            inv_from_chol(P, w.Di + (size_t)j * NX * NX);
        }
        __syncthreads();
        return;
    }
    // Two-level: segment block-Thomas factors F_k = chol(D~_k), E_k = S_{k,k-1} F_{k-1}^-T.
    const int nseg = (N + seg - 1) / seg;
    for (int s = threadIdx.x; s < nseg; s += blockDim.x) {
        const int a0 = s * seg, b0 = min(a0 + seg, N);
        for (int k = a0; k < b0; ++k) {
            T* F = w.Pf + (size_t)k * NX * NX;
            for (int i = 0; i < NX * NX; ++i) F[i] = w.Sjj[(size_t)k * NX * NX + i];
            if (k > a0) {
                T* E = w.Pe + (size_t)k * NX * NX;
                const T* Fp = w.Pf + (size_t)(k - 1) * NX * NX;
                for (int i = 0; i < NX; ++i) {                  // E row i = F_{k-1}^-1 Slo row i
                    T row[NX];
                    for (int j = 0; j < NX; ++j) row[j] = w.Slo[(size_t)k * NX * NX + i * NX + j];
                    tri_lower<NX>(Fp, row);
                    for (int j = 0; j < NX; ++j) E[i * NX + j] = row[j];
                }
                for (int i = 0; i < NX; ++i)
                    for (int j = 0; j < NX; ++j) {
                        T acc = 0;
                        for (int l = 0; l < NX; ++l) acc += E[i * NX + l] * E[j * NX + l];
                        F[i * NX + j] -= acc;
                    }
            }
            chol<NX>(F);
        }
    }
    // Coarse R_0 S R_0^T: diagonal = sum of every block inside the segment, sub-diagonal =
    // the one coupling block S_{first(i), last(i-1)}.
    if (threadIdx.x == 0) {
        for (int s = 0; s < nseg; ++s) {
            const int a0 = s * seg, b0 = min(a0 + seg, N);
            T* D = w.Cf + (size_t)s * NX * NX;
            for (int i = 0; i < NX * NX; ++i) D[i] = 0;
            for (int k = a0; k < b0; ++k)
                for (int i = 0; i < NX; ++i)
                    for (int j = 0; j < NX; ++j) {
                        D[i * NX + j] += w.Sjj[(size_t)k * NX * NX + i * NX + j];
                        if (k > a0)
                            D[i * NX + j] += w.Slo[(size_t)k * NX * NX + i * NX + j] +
                                             w.Slo[(size_t)k * NX * NX + j * NX + i];
                    }
            if (s > 0) {
                T* E = w.Ce + (size_t)s * NX * NX;
                const T* Fp = w.Cf + (size_t)(s - 1) * NX * NX;
                for (int i = 0; i < NX; ++i) {
                    T row[NX];
                    for (int j = 0; j < NX; ++j) row[j] = w.Slo[(size_t)a0 * NX * NX + i * NX + j];
                    tri_lower<NX>(Fp, row);
                    for (int j = 0; j < NX; ++j) E[i * NX + j] = row[j];
                }
                for (int i = 0; i < NX; ++i)
                    for (int j = 0; j < NX; ++j) {
                        T acc = 0;
                        for (int l = 0; l < NX; ++l) acc += E[i * NX + l] * E[j * NX + l];
                        D[i * NX + j] -= acc;
                    }
            }
            chol<NX>(D);
        }
    }
    __syncthreads();
    // Explicit form of both block-LDL^T factorizations, so the preconditioner is matvecs:
    //   Di_k = Dt_k^-1,  W_k = S_{k,k-1} Di_{k-1}  (and the same per coarse block).
    for (int k = threadIdx.x; k < N; k += blockDim.x) inv_from_chol(w.Pf + (size_t)k * NX * NX, w.Di + (size_t)k * NX * NX);
    for (int c = threadIdx.x; c < nseg; c += blockDim.x) inv_from_chol(w.Cf + (size_t)c * NX * NX, w.CDi + (size_t)c * NX * NX);
    __syncthreads();
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        if (k % seg) mm_nx(w.Slo + (size_t)k * NX * NX, w.Di + (size_t)(k - 1) * NX * NX, w.Wf + (size_t)k * NX * NX);
    for (int c = 1 + threadIdx.x; c < nseg; c += blockDim.x)
        mm_nx(w.Slo + (size_t)c * seg * NX * NX, w.CDi + (size_t)(c - 1) * NX * NX, w.CW + (size_t)c * NX * NX);
    __syncthreads();
}

// Cooperative block-tridiagonal solve of `nch` independent chains of `len` blocks over
// `nblk` blocks (block b in chain b / len), explicit LDL^T form: y_b -= W_b y_{b-1} forward,
// x_b = Di_b (y_b - S_{b+1,b}^T x_{b+1}) backward. S_{b+1,b} = Slo[(b+1) * stride].
// Threads cover (chain, row); depth ~3 * len barriers.
__device__ void thomas_coop(int nch, int len, int nblk, const T* Di, const T* Wf, const T* Slo,
                            int stride, T* y, T* x)
{
    const int work = nch * NX;
    for (int i = 1; i < len; ++i) {
        for (int e = threadIdx.x; e < work; e += blockDim.x) {
            const int c = e / NX, row = e % NX, b = c * len + i;
            if (b >= nblk) continue;
            const T* Wb = Wf + (size_t)b * NX * NX + row * NX;
            const T* yp = y + (size_t)(b - 1) * NX;
            T acc = 0;
            for (int l = 0; l < NX; ++l) acc += Wb[l] * yp[l];
            y[(size_t)b * NX + row] -= acc;
        }
        __syncthreads();
    }
    for (int i = len - 1; i >= 0; --i) {
        if (i + 1 < len) {                    // y_b -= S_{b+1,b}^T x_{b+1}, in place per row
            for (int e = threadIdx.x; e < work; e += blockDim.x) {
                const int c = e / NX, row = e % NX, b = c * len + i;
                if (b + 1 >= nblk) continue;
                const T* Su = Slo + (size_t)(b + 1) * stride * NX * NX;
                const T* xn = x + (size_t)(b + 1) * NX;
                T acc = 0;
                for (int m = 0; m < NX; ++m) acc += Su[m * NX + row] * xn[m];
                y[(size_t)b * NX + row] -= acc;
            }
            __syncthreads();
        }
        for (int e = threadIdx.x; e < work; e += blockDim.x) {
            const int c = e / NX, row = e % NX, b = c * len + i;
            if (b >= nblk) continue;
            const T* Db = Di + (size_t)b * NX * NX + row * NX;
            const T* yb = y + (size_t)b * NX;
            T acc = 0;
            for (int l = 0; l < NX; ++l) acc += Db[l] * yb[l];
            x[(size_t)b * NX + row] = acc;
        }
        __syncthreads();
    }
}

// Block-Thomas solve over blocks [a0, b0) of factors (F, E), in place on x (knot-major).
__device__ void thomas_solve(const T* Ff, const T* Ee, int a0, int b0, T* x)
{
    for (int k = a0; k < b0; ++k) {
        T* xk = x + (size_t)k * NX;
        if (k > a0) mv<NX, NX>(Ee + (size_t)k * NX * NX, xk - NX, xk, -1, 1);
        tri_lower<NX>(Ff + (size_t)k * NX * NX, xk);
    }
    for (int k = b0 - 1; k >= a0; --k) {
        T* xk = x + (size_t)k * NX;
        if (k + 1 < b0) mtv<NX, NX>(Ee + (size_t)(k + 1) * NX * NX, xk + NX, xk, -1, 1);
        tri_lower_t<NX>(Ff + (size_t)k * NX * NX, xk);
    }
}

__device__ void apply_precond(int N, int precond, int seg, const PcgWork& w, const T* r, T* z)
{
    if (precond == 0) {
        for (int e = threadIdx.x; e < N * NX; e += blockDim.x) {
            const int j = e / NX, row = e % NX;
            const T* D = w.Di + (size_t)j * NX * NX + row * NX;
            const T* rj = r + (size_t)j * NX;
            T acc = 0;
            for (int l = 0; l < NX; ++l) acc += D[l] * rj[l];
            z[e] = acc;
        }
        __syncthreads();
        return;
    }
    const int nseg = (N + seg - 1) / seg;
    // y (forward work) lives in w.tmp; the coarse rhs / solution in uc / uc + nseg*NX.
    T* y = w.tmp;
    T* uy = w.uc;
    T* ux = w.uc + (size_t)nseg * NX;
    for (int e = threadIdx.x; e < N * NX; e += blockDim.x) y[e] = r[e];
    for (int e = threadIdx.x; e < nseg * NX; e += blockDim.x) {
        const int c = e / NX, row = e % NX;
        T acc = 0;
        for (int k = c * seg; k < min((c + 1) * seg, N); ++k) acc += r[(size_t)k * NX + row];
        uy[e] = acc;
    }
    __syncthreads();
    thomas_coop(nseg, seg, N, w.Di, w.Wf, w.Slo, 1, y, z);
    thomas_coop(1, nseg, nseg, w.CDi, w.CW, w.Slo, seg, uy, ux);
    for (int e = threadIdx.x; e < N * NX; e += blockDim.x)
        z[e] += ux[(size_t)(e / NX / seg) * NX + e % NX];
    __syncthreads();
}

// S v = C H C^T v over (knot, row) work items: y = C^T v, t = H y, out = C t.
__device__ void schur_mv(const Lin& L, int N, const PcgWork& w, const T* v, T* out)
{
    for (int e = threadIdx.x; e < N * NS; e += blockDim.x) {
        const int j = e / NS, i = e % NS;
        T acc = i < NX ? v[(size_t)j * NX + i] : T(0);
        if (j < N - 1) {                                    // - F_j^T v_{j+1}
            const T* vn = v + (size_t)(j + 1) * NX;
            for (int l = 0; l < NX; ++l)
                acc -= (i < NX ? L.A[(size_t)j * NX * NX + l * NX + i]
                               : L.Bt[(size_t)j * NX * M + l * M + i - NX]) * vn[l];
        }
        w.ty[e] = acc;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < N * NS; e += blockDim.x) {
        const int j = e / NS, i = e % NS;
        const T* Hr = w.H + (size_t)j * NS * NS + i * NS;
        const T* y = w.ty + (size_t)j * NS;
        T acc = 0;
        for (int l = 0; l < NS; ++l) acc += Hr[l] * y[l];
        w.tmp[e] = acc;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < N * NX; e += blockDim.x) {
        const int j = e / NX, i = e % NX;
        T acc = w.tmp[(size_t)j * NS + i];
        if (j > 0) {
            const T* zp = w.tmp + (size_t)(j - 1) * NS;
            const T* Ar = L.A + (size_t)(j - 1) * NX * NX + i * NX;
            const T* Br = L.Bt + (size_t)(j - 1) * NX * M + i * M;
            for (int l = 0; l < NX; ++l) acc -= Ar[l] * zp[l];
            for (int l = 0; l < M; ++l) acc -= Br[l] * zp[NX + l];
        }
        out[e] = acc;
    }
    __syncthreads();
}

__device__ T dotv(int n, const T* a, const T* b, T* sh)
{
    T acc = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += a[i] * b[i];
    return block_sum(acc, sh);
}

// One QP solve: s = Hg - H C^T v after `iters` PCG iterations on S v = C Hg - b (warm v).
// Returns the iterations run (tol > 0 stops early on ||res|| <= tol ||rhs||).
__device__ int schur_qp(const Lin& L, int N, T s2, T s2w, const T* tl, const T* te,
                        const T* s_prev, int iters, T tol, int precond, int seg,
                        const PcgWork& w, T* dX, T* W, T* sh)
{
    const int n = N * NX;
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        T g[NS];
        stage_grad(L, k, s2, s2w, tl, te, g);
        for (int i = 0; i < NS; ++i) g[i] = -g[i];
        if (s_prev)
            for (int i = 0; i < NS; ++i) g[i] += w.sig[k] * s_prev[(size_t)k * NS + i];
        mv<NS, NS>(w.H + (size_t)k * NS * NS, g, w.Hg + (size_t)k * NS);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < N; j += blockDim.x) {       // rhs = C Hg - b  -> d (temp)
        T* o = w.d + (size_t)j * NX;
        for (int i = 0; i < NX; ++i) o[i] = w.Hg[(size_t)j * NS + i];
        if (j > 0) {
            const T* hp = w.Hg + (size_t)(j - 1) * NS;
            mv<NX, NX>(L.A + (size_t)(j - 1) * NX * NX, hp, o, -1, 1);
            mv<NX, M>(L.Bt + (size_t)(j - 1) * NX * M, hp + NX, o, -1, 1);
            for (int i = 0; i < NX; ++i) o[i] -= L.c[(size_t)(j - 1) * NX + i];
        }
    }
    __syncthreads();
    schur_mv(L, N, w, w.v, w.Sd);
    for (int i = threadIdx.x; i < n; i += blockDim.x) w.res[i] = w.d[i] - w.Sd[i];
    const T rhs2 = dotv(n, w.d, w.d, sh);
    apply_precond(N, precond, seg, w, w.res, w.z);
    for (int i = threadIdx.x; i < n; i += blockDim.x) w.d[i] = w.z[i];
    T rz = dotv(n, w.res, w.z, sh);
    int it = 0;
    for (; it < iters; ++it) {
        if (tol > 0 && dotv(n, w.res, w.res, sh) <= tol * tol * rhs2) break;
        schur_mv(L, N, w, w.d, w.Sd);
        const T a = rz / (dotv(n, w.d, w.Sd, sh) + 1e-30);
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            w.v[i] += a * w.d[i];
            w.res[i] -= a * w.Sd[i];
        }
        __syncthreads();
        apply_precond(N, precond, seg, w, w.res, w.z);
        const T rz2 = dotv(n, w.res, w.z, sh);
        const T beta = rz2 / (rz + 1e-30);
        for (int i = threadIdx.x; i < n; i += blockDim.x) w.d[i] = w.z[i] + beta * w.d[i];
        __syncthreads();
        rz = rz2;
    }
    for (int j = threadIdx.x; j < N; j += blockDim.x) {       // C^T v, then s = Hg - H C^T v
        T y[NS], hy[NS];
        for (int i = 0; i < NS; ++i) y[i] = i < NX ? w.v[(size_t)j * NX + i] : T(0);
        if (j < N - 1) {
            const T* vn = w.v + (size_t)(j + 1) * NX;
            mtv<NX, NX>(L.A + (size_t)j * NX * NX, vn, y, -1, 1);
            mtv<NX, M>(L.Bt + (size_t)j * NX * M, vn, y + NX, -1, 1);
        }
        mv<NS, NS>(w.H + (size_t)j * NS * NS, y, hy);
        for (int i = 0; i < NX; ++i) dX[(size_t)j * NX + i] = w.Hg[(size_t)j * NS + i] - hy[i];
        for (int i = 0; i < M; ++i) W[(size_t)j * M + i] = w.Hg[(size_t)j * NS + NX + i] - hy[NX + i];
    }
    __syncthreads();
    return it;
}

// ─────────────────────────────────────────────────────────────── workspace layout ──
// Private scratch of one explicit-operator column: kff (N,M), unit targets tl/te (N,NL)
// each, and the solve's dX (N,NX), W (N,M).
__host__ __device__ inline size_t col_doubles(int N) { return (size_t)N * (2 * M + 2 * NL + NX); }

struct Sizes {
    int N, nseg;
    size_t ric, pcg, expl, total;
};
__host__ __device__ inline Sizes sizes_of(int N, int seg, int backend)
{
    Sizes s;
    s.N = N;
    s.nseg = seg > 0 ? (N + seg - 1) / seg : N;
    s.ric = (size_t)N * (2 * M * M + M * NX + NX * NX + 2 * M);
    s.pcg = (size_t)N * (NS * NS + 1 + 4 * NX * NX) + 2 * (size_t)s.nseg * NX * NX +
            (size_t)N * (NS + 5 * NX + NS) + 2 * (size_t)s.nseg * NX +
            2 * (size_t)N * NX * NX + 2 * (size_t)s.nseg * NX * NX + (size_t)N * NS;
    const size_t nt = 2 * (size_t)N * NL;
    s.expl = backend == 1 ? nt * nt + nt + (nt + 1) * col_doubles(N) : 0;
    s.total = s.ric + s.pcg + s.expl + 4 * (size_t)N * NL + (size_t)N * (NX + M + NS);
    return s;
}
struct Work {
    RicWork ric;
    PcgWork pcg;
    T *G, *y0, *cols;        // explicit operator; per-column private scratch
    T *tl, *te, *tln, *ten;  // consensus targets (last used) and the z-step's (L1, E1)
    T *dX, *W, *sp;          // current QP solution; proximal centre [dx | w] per knot
};
__device__ Work work_of(T* ws, const Sizes& s)
{
    const int N = s.N;
    Work w;
    T* p = ws;
    w.ric.Lq = p; p += (size_t)N * M * M;
    w.ric.K = p; p += (size_t)N * M * NX;
    w.ric.Pn = p; p += (size_t)N * NX * NX;
    w.ric.kff = p; p += (size_t)N * M;
    w.ric.idg = p; p += (size_t)N * M;
    w.ric.Qinv = p; p += (size_t)N * M * M;
    w.pcg.H = p; p += (size_t)N * NS * NS;
    w.pcg.sig = p; p += N;
    w.pcg.Sjj = p; p += (size_t)N * NX * NX;
    w.pcg.Slo = p; p += (size_t)N * NX * NX;
    w.pcg.Pf = p; p += (size_t)N * NX * NX;
    w.pcg.Pe = p; p += (size_t)N * NX * NX;
    w.pcg.Cf = p; p += (size_t)s.nseg * NX * NX;
    w.pcg.Ce = p; p += (size_t)s.nseg * NX * NX;
    w.pcg.Hg = p; p += (size_t)N * NS;
    w.pcg.v = p; p += (size_t)N * NX;
    w.pcg.res = p; p += (size_t)N * NX;
    w.pcg.z = p; p += (size_t)N * NX;
    w.pcg.d = p; p += (size_t)N * NX;
    w.pcg.Sd = p; p += (size_t)N * NX;
    w.pcg.tmp = p; p += (size_t)N * NS;
    w.pcg.uc = p; p += 2 * (size_t)s.nseg * NX;
    w.pcg.Di = p; p += (size_t)N * NX * NX;
    w.pcg.Wf = p; p += (size_t)N * NX * NX;
    w.pcg.CDi = p; p += (size_t)s.nseg * NX * NX;
    w.pcg.CW = p; p += (size_t)s.nseg * NX * NX;
    w.pcg.ty = p; p += (size_t)N * NS;
    w.ric.Gall = w.pcg.H;   // the PCG region is unused while Riccati runs (backends exclusive)
    w.ric.gall = w.pcg.tmp;
    const size_t nt = 2 * (size_t)N * NL;
    w.G = p; p += s.expl ? nt * nt : 0;
    w.y0 = p; p += s.expl ? nt : 0;
    w.cols = p; p += s.expl ? (nt + 1) * col_doubles(N) : 0;
    w.tl = p; p += (size_t)N * NL;
    w.te = p; p += (size_t)N * NL;
    w.tln = p; p += (size_t)N * NL;
    w.ten = p; p += (size_t)N * NL;
    w.dX = p; p += (size_t)N * NX;
    w.W = p; p += (size_t)N * M;
    w.sp = p; p += (size_t)N * NS;
    return w;
}

// ────────────────────────────────────────────────────────────────────── kernels ──
__global__ void project_kernel(const T* a, const T* b, T* pa, T* pb, size_t n, T w_eta)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        project_pair(a[i], b[i], w_eta, pa[i], pb[i]);
}

__global__ void box_kernel(const T* x, const T* lo, const T* hi, T* out, size_t n, size_t per)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = box_clamp(x[i], lo[i % per], hi[i % per]);
}

// One thread per problem: factor + solve for given targets.
__global__ void riccati_kernel(Lin g, const T* tl, const T* te, T* dX, T* W, T* ws, int B, int N,
                               T s2, T s2w, Sizes s)
{
    __shared__ RicShared S;
    const int b = blockIdx.x;
    const Lin L = lin_of(g, b, N);
    Work w = work_of(ws + b * s.total, s);
    ric_factor_coop(L, N, s2, s2w, w.ric, S);
    ric_solve_coop(L, N, s2, s2w, tl + (size_t)b * N * NL, te + (size_t)b * N * NL, w.ric, S,
                   dX + (size_t)b * N * NX, W + (size_t)b * N * M);
}

// One block per problem: Schur-PCG QP solve.
__global__ void pcg_kernel(Lin g, const T* tl, const T* te, const T* s_prev, const T* v0,
                           T* dX, T* W, T* v_out, int* iters_out, T* ws, int N, T s2, T s2w,
                           T reg, int iters, T tol, int precond, int seg, Sizes s)
{
    __shared__ T sh[TPB];
    const int b = blockIdx.x;
    const Lin L = lin_of(g, b, N);
    Work w = work_of(ws + b * s.total, s);
    schur_setup(L, N, s2, s2w, reg, precond, seg, w.pcg);
    for (int i = threadIdx.x; i < N * NX; i += blockDim.x) w.pcg.v[i] = v0[(size_t)b * N * NX + i];
    __syncthreads();
    const int it = schur_qp(L, N, s2, s2w, tl + (size_t)b * N * NL, te + (size_t)b * N * NL,
                            s_prev ? s_prev + (size_t)b * N * NS : nullptr, iters, tol, precond,
                            seg, w.pcg, dX + (size_t)b * N * NX, W + (size_t)b * N * M, sh);
    for (int i = threadIdx.x; i < N * NX; i += blockDim.x) v_out[(size_t)b * N * NX + i] = w.pcg.v[i];
    if (threadIdx.x == 0) iters_out[b] = it;
}

// Explicit operator: column j of G is (L1, E1)(e_j) - (L1, E1)(0); one thread per column,
// column -1 computes y0 = (L1, E1)(0). Every column reuses the Riccati factor already in
// the workspace (only the right-hand side differs) and owns its kff/target/solution scratch.
__global__ void explicit_columns_kernel(Lin g, T* ws, int N, T s2, T s2w, Sizes s)
{
    const int b = blockIdx.y;
    const int nt = 2 * N * NL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x - 1;
    if (j >= nt) return;
    const Lin L = lin_of(g, b, N);
    Work w = work_of(ws + b * s.total, s);
    T* col = w.cols + (size_t)(j + 1) * col_doubles(N);
    RicWork rw = w.ric;
    rw.kff = col;
    T* tl = col + (size_t)N * M;
    T* te = tl + (size_t)N * NL;
    T* dX = te + (size_t)N * NL;
    T* W = dX + (size_t)N * NX;
    for (int i = 0; i < N * NL; ++i) { tl[i] = 0; te[i] = 0; }
    if (j >= 0) (j < N * NL ? tl[j] : te[j - N * NL]) = 1;
    ric_solve(L, N, s2, s2w, tl, te, rw, dX, W);
    T L1[NL], E1[NL];
    for (int k = 0; k < N; ++k) {
        lam_eta_at(L, k, dX, W, L1, E1);
        for (int l = 0; l < NL; ++l) {
            const int rl = k * NL + l, re = N * NL + k * NL + l;
            if (j < 0) { w.y0[rl] = L1[l]; w.y0[re] = E1[l]; }
            else { w.G[(size_t)rl * nt + j] = L1[l]; w.G[(size_t)re * nt + j] = E1[l]; }
        }
    }
}
// After the columns: G[:, j] -= y0 (one thread per row).
__global__ void explicit_center_kernel(T* ws, int N, Sizes s)
{
    const int b = blockIdx.y;
    const int nt = 2 * N * NL;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nt) return;
    Work w = work_of(ws + b * s.total, s);
    for (int j = 0; j < nt; ++j) w.G[(size_t)row * nt + j] -= w.y0[row];
}
__global__ void ric_factor_kernel(Lin g, T* ws, int B, int N, T s2, T s2w, Sizes s)
{
    __shared__ RicShared S;
    const int b = blockIdx.x;
    const Lin L = lin_of(g, b, N);
    Work w = work_of(ws + b * s.total, s);
    ric_factor_coop(L, N, s2, s2w, w.ric, S);
}

// The ADMM inner loop (c3._c3_inner semantics) over one QP backend, one block per problem.
// backend: 0 Riccati, 1 explicit operator (factor + G built by the kernels above), 2 PCG.
__global__ void admm_kernel(Lin g, const T* dl0, const T* de0, const T* wl0, const T* we0,
                            T* dX_out, T* W_out, T* dl_out, T* de_out, T* wl_out, T* we_out,
                            int* pcg_iters, T* ws, int N, T rho, T w_eta, int n_admm, int backend,
                            T reg, int iters, T tol, int precond, int seg, int prox, Sizes s,
                            T scale)
{
    __shared__ T sh[TPB];
    const int b = blockIdx.x;
    const Lin L = lin_of(g, b, N);
    Work w = work_of(ws + b * s.total, s);
    const T s2 = 0.5 * rho, s2w = s2 * w_eta;
    const size_t off = (size_t)b * N * NL;
    T* dl = dl_out + off; T* de = de_out + off; T* wl = wl_out + off; T* we = we_out + off;
    for (int i = threadIdx.x; i < N * NL; i += blockDim.x) {
        dl[i] = dl0[off + i]; de[i] = de0[off + i]; wl[i] = wl0[off + i]; we[i] = we0[off + i];
    }
    __shared__ RicShared RS;
    if (backend == 0) ric_factor_coop(L, N, s2, s2w, w.ric, RS);
    if (backend == 2) {
        schur_setup(L, N, s2, s2w, reg, precond, seg, w.pcg);
        for (int i = threadIdx.x; i < N * NX; i += blockDim.x) w.pcg.v[i] = 0;
        for (int i = threadIdx.x; i < N * NX; i += blockDim.x) w.dX[i] = 0;
        for (int i = threadIdx.x; i < N * M; i += blockDim.x) w.W[i] = 0;
    }
    __syncthreads();
    const int nt = 2 * N * NL;
    int it_total = 0;
    for (int a = 0; a < n_admm; ++a) {
        for (int i = threadIdx.x; i < N * NL; i += blockDim.x) {
            w.tl[i] = wl[i] - dl[i];
            w.te[i] = we[i] - de[i];
        }
        __syncthreads();
        // z-step -> (L1, E1) into (tln, ten)
        if (backend == 1) {
            for (int row = threadIdx.x; row < nt; row += blockDim.x) {
                const T* Gr = w.G + (size_t)row * nt;
                T acc = w.y0[row];
                for (int j = 0; j < N * NL; ++j) acc += Gr[j] * w.tl[j];
                for (int j = 0; j < N * NL; ++j) acc += Gr[N * NL + j] * w.te[j];
                (row < N * NL ? w.tln[row] : w.ten[row - N * NL]) = acc;
            }
        } else {
            if (backend == 0) {
                ric_solve_coop(L, N, s2, s2w, w.tl, w.te, w.ric, RS, w.dX, w.W);
            } else {
                // Proximal centre: the previous ADMM iterate's (dX, W), packed [dx | w].
                for (int k = threadIdx.x; k < N; k += blockDim.x) {
                    for (int i = 0; i < NX; ++i) w.sp[(size_t)k * NS + i] = w.dX[(size_t)k * NX + i];
                    for (int i = 0; i < M; ++i) w.sp[(size_t)k * NS + NX + i] = w.W[(size_t)k * M + i];
                }
                __syncthreads();
                it_total += schur_qp(L, N, s2, s2w, w.tl, w.te, prox ? w.sp : nullptr, iters, tol,
                                     precond, seg, w.pcg, w.dX, w.W, sh);
            }
            __syncthreads();
            for (int k = threadIdx.x; k < N; k += blockDim.x)
                lam_eta_at(L, k, w.dX, w.W, w.tln + (size_t)k * NL, w.ten + (size_t)k * NL);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < N * NL; i += blockDim.x) {
            const T L1 = w.tln[i], E1 = w.ten[i];
            T pa, pb;
            project_pair(L1 + wl[i], E1 + we[i], w_eta, pa, pb);
            dl[i] = pa; de[i] = pb;
            wl[i] += L1 - pa;
            we[i] += E1 - pb;
        }
        __syncthreads();
    }
    // Final QP solution: the last iteration's z-step (exact re-solve for Riccati/explicit).
    if (backend != 2) {
        ric_solve_coop(L, N, s2, s2w, w.tl, w.te, w.ric, RS, w.dX, w.W);
    }
    for (int i = threadIdx.x; i < N * NX; i += blockDim.x) dX_out[(size_t)b * N * NX + i] = w.dX[i];
    for (int i = threadIdx.x; i < N * M; i += blockDim.x) W_out[(size_t)b * N * M + i] = w.W[i];
    for (int i = threadIdx.x; i < N * NL; i += blockDim.x) { wl[i] *= scale; we[i] *= scale; }
    if (threadIdx.x == 0) pcg_iters[b] = it_total;
}

// Per-knot PGS LCP, one thread per (problem, knot): 0 <= lam _|_ F lam + c >= 0.
__global__ void pgs_kernel(const T* F, const T* c, const T* lam0, T* lam, size_t n, int sweeps)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const T* Fi = F + i * NL * NL;
    const T* ci = c + i * NL;
    T l[NL];
    for (int a = 0; a < NL; ++a) l[a] = lam0[i * NL + a];
    for (int s = 0; s < sweeps; ++s)
        for (int a = 0; a < NL; ++a) {
            T acc = 0;
            for (int b = 0; b < NL; ++b) acc += Fi[a * NL + b] * l[b];
            l[a] = fmax(T(0), l[a] - (acc + ci[a]) / Fi[a * NL + a]);
        }
    for (int a = 0; a < NL; ++a) lam[i * NL + a] = l[a];
}

// ms_c3_solve's step selection: M = cost + mu * sum|d| per alpha, first argmin, ok = M < M_cur.
__global__ void merit_kernel(const T* cd, const T* cd_cur, const T* mu, int* idx, int* ok, int B, int na)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int best = 0;
    T bm = INFINITY;
    for (int a = 0; a < na; ++a) {
        const T m = cd[(size_t)(b * na + a) * 2] + mu[b] * cd[(size_t)(b * na + a) * 2 + 1];
        if (m < bm) { bm = m; best = a; }
    }
    idx[b] = best;
    ok[b] = bm < cd_cur[b * 2] + mu[b] * cd_cur[b * 2 + 1];
}

// ───────────────────────────────────────────────────────────────── FFI handlers ──
using F64 = ffi::Buffer<ffi::DataType::F64>;
using RF64 = ffi::ResultBuffer<ffi::DataType::F64>;
using RI32 = ffi::ResultBuffer<ffi::DataType::S32>;

static ffi::Error cuda_status()
{
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}
// Workspace doubles per problem, for the Python side to size the workspace output
// (the one source of truth for the layout above).
extern "C" size_t c3_workspace_doubles(int N, int seg, int backend)
{
    return sizes_of(N, seg, backend).total;
}
static T* workspace(RF64& ws, size_t need)
{
    return ws->element_count() >= need ? ws->typed_data() : nullptr;
}
static Lin lin_from(const F64& A, const F64& Bt, const F64& E, const F64& Ht, const F64& Eta,
                    const F64& Lam, const F64& c, const F64& Jx, const F64& Ju, const F64& r)
{
    return Lin{A.typed_data(), Bt.typed_data(), E.typed_data(), Ht.typed_data(), Eta.typed_data(),
               Lam.typed_data(), c.typed_data(), Jx.typed_data(), Ju.typed_data(), r.typed_data()};
}
#define LIN_ARGS F64 A, F64 Bt, F64 E, F64 Ht, F64 Eta, F64 Lam, F64 c, F64 Jx, F64 Ju, F64 r
#define LIN_BIND                                                                            \
    .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>() \
        .Arg<F64>().Arg<F64>()

static ffi::Error ProjectImpl(cudaStream_t st, F64 a, F64 b, RF64 pa, RF64 pb, double w_eta)
{
    const size_t n = a.element_count();
    project_kernel<<<(n + 255) / 256, 256, 0, st>>>(a.typed_data(), b.typed_data(), pa->typed_data(),
                                                    pb->typed_data(), n, w_eta);
    return cuda_status();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(C3ProjectFfi, ProjectImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<F64>().Arg<F64>().Ret<F64>()
        .Ret<F64>().Attr<double>("w_eta"));

static ffi::Error BoxImpl(cudaStream_t st, F64 x, F64 lo, F64 hi, RF64 out)
{
    const size_t n = x.element_count();
    box_kernel<<<(n + 255) / 256, 256, 0, st>>>(x.typed_data(), lo.typed_data(), hi.typed_data(),
                                                out->typed_data(), n, lo.element_count());
    return cuda_status();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(C3BoxFfi, BoxImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<F64>().Arg<F64>().Arg<F64>()
        .Ret<F64>());

static ffi::Error RiccatiImpl(cudaStream_t st, LIN_ARGS, F64 tl, F64 te, RF64 dX, RF64 W, RF64 wsb,
                              double rho, double w_eta)
{
    const int B = static_cast<int>(A.dimensions()[0]), N = static_cast<int>(A.dimensions()[1]);
    const Sizes sz = sizes_of(N, 0, 0);
    T* ws = workspace(wsb, sz.total * B);
    if (!ws) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "c3 riccati: workspace too small");
    const T s2 = 0.5 * rho;
    riccati_kernel<<<B, TPB, 0, st>>>(lin_from(A, Bt, E, Ht, Eta, Lam, c, Jx, Ju, r),
        tl.typed_data(), te.typed_data(), dX->typed_data(), W->typed_data(), ws, B, N, s2, s2 * w_eta, sz);
    return cuda_status();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(C3RiccatiFfi, RiccatiImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() LIN_BIND
        .Arg<F64>().Arg<F64>().Ret<F64>().Ret<F64>().Ret<F64>().Attr<double>("rho").Attr<double>("w_eta"));

static ffi::Error PcgImpl(cudaStream_t st, LIN_ARGS, F64 tl, F64 te, F64 s_prev, F64 v0, RF64 dX,
                          RF64 W, RF64 v, RI32 iters_out, RF64 wsb, double rho,
                          double w_eta, double reg, int32_t iters, double tol, int32_t precond,
                          int32_t seg, int32_t prox)
{
    const int B = static_cast<int>(A.dimensions()[0]), N = static_cast<int>(A.dimensions()[1]);
    const int sg = seg > 0 ? seg : N;
    const Sizes sz = sizes_of(N, sg, 2);
    T* ws = workspace(wsb, sz.total * B);
    if (!ws) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "c3 pcg: workspace too small");
    const T s2 = 0.5 * rho;
    pcg_kernel<<<B, TPB, 0, st>>>(lin_from(A, Bt, E, Ht, Eta, Lam, c, Jx, Ju, r), tl.typed_data(),
        te.typed_data(), prox ? s_prev.typed_data() : nullptr, v0.typed_data(), dX->typed_data(),
        W->typed_data(), v->typed_data(), iters_out->typed_data(), ws, N, s2, s2 * w_eta, reg, iters,
        tol, precond, sg, sz);
    return cuda_status();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(C3PcgFfi, PcgImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() LIN_BIND
        .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Ret<F64>().Ret<F64>().Ret<F64>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>().Ret<F64>().Attr<double>("rho").Attr<double>("w_eta")
        .Attr<double>("reg").Attr<int32_t>("iters").Attr<double>("tol").Attr<int32_t>("precond")
        .Attr<int32_t>("seg").Attr<int32_t>("prox"));

static ffi::Error AdmmImpl(cudaStream_t st, LIN_ARGS, F64 dl, F64 de, F64 wl, F64 we, RF64 dX,
                           RF64 W, RF64 dl2, RF64 de2, RF64 wl2, RF64 we2, RI32 it, RF64 wsb, double rho, double rho_scale, double rho_max, double w_eta,
                           int32_t n_admm, int32_t backend, double reg, int32_t iters, double tol,
                           int32_t precond, int32_t seg, int32_t prox)
{
    const int B = static_cast<int>(A.dimensions()[0]), N = static_cast<int>(A.dimensions()[1]);
    const int sg = seg > 0 ? seg : N;
    const Sizes sz = sizes_of(N, sg, backend);
    T* ws = workspace(wsb, sz.total * B);
    if (!ws) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "c3 admm: workspace too small");
    const Lin L = lin_from(A, Bt, E, Ht, Eta, Lam, c, Jx, Ju, r);
    const T s2 = 0.5 * rho;
    if (backend == 1) {
        ric_factor_kernel<<<B, TPB, 0, st>>>(L, ws, B, N, s2, s2 * w_eta, sz);
        const int nt = 2 * N * NL;
        dim3 grid((nt + 1 + 63) / 64, B);
        explicit_columns_kernel<<<grid, 64, 0, st>>>(L, ws, N, s2, s2 * w_eta, sz);
        dim3 grid2((nt + 127) / 128, B);
        explicit_center_kernel<<<grid2, 128, 0, st>>>(ws, N, sz);
    }
    const T rho1 = fmin(rho * rho_scale, rho_max);
    admm_kernel<<<B, TPB, 0, st>>>(L, dl.typed_data(), de.typed_data(), wl.typed_data(), we.typed_data(),
        dX->typed_data(), W->typed_data(), dl2->typed_data(), de2->typed_data(), wl2->typed_data(),
        we2->typed_data(), it->typed_data(), ws, N, rho, w_eta, n_admm, backend, reg, iters, tol,
        precond, sg, prox, sz, rho / rho1);
    return cuda_status();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(C3AdmmFfi, AdmmImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() LIN_BIND
        .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Ret<F64>().Ret<F64>().Ret<F64>().Ret<F64>()
        .Ret<F64>().Ret<F64>().Ret<ffi::Buffer<ffi::DataType::S32>>().Ret<F64>().Attr<double>("rho")
        .Attr<double>("rho_scale").Attr<double>("rho_max").Attr<double>("w_eta")
        .Attr<int32_t>("n_admm").Attr<int32_t>("backend").Attr<double>("reg").Attr<int32_t>("iters")
        .Attr<double>("tol").Attr<int32_t>("precond").Attr<int32_t>("seg").Attr<int32_t>("prox"));

static ffi::Error PgsImpl(cudaStream_t st, F64 F, F64 c, F64 lam0, RF64 lam, int32_t sweeps)
{
    const size_t n = c.element_count() / NL;
    pgs_kernel<<<(n + 127) / 128, 128, 0, st>>>(F.typed_data(), c.typed_data(), lam0.typed_data(),
                                                lam->typed_data(), n, sweeps);
    return cuda_status();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(C3PgsFfi, PgsImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<F64>().Arg<F64>().Arg<F64>()
        .Ret<F64>().Attr<int32_t>("sweeps"));

static ffi::Error MeritImpl(cudaStream_t st, F64 cd, F64 cd_cur, F64 mu, RI32 idx, RI32 ok)
{
    const int B = static_cast<int>(cd.dimensions()[0]), na = static_cast<int>(cd.dimensions()[1]);
    merit_kernel<<<(B + 127) / 128, 128, 0, st>>>(cd.typed_data(), cd_cur.typed_data(), mu.typed_data(),
                                                  idx->typed_data(), ok->typed_data(), B, na);
    return cuda_status();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(C3MeritFfi, MeritImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<F64>().Arg<F64>().Arg<F64>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>().Ret<ffi::Buffer<ffi::DataType::S32>>());
