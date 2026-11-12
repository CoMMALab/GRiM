// crocket dynamics in `real` (GRIM_CK_REAL, float or double), one configuration per thread,
// warp or block: crocket's thread / warp / block variant of each traced algorithm, and the
// dense solves of the composite ops in the same scalar type. Compiled per robot by
// grim.motion.dynamics.CrocketDynamics against "ck_dynamics_gen.cuh", which holds crocket's
// rnea / crba / aba / rnea_derivatives and, in grim::dyn, N (joints), SMEM (the largest
// cooperative scratch, reals) and BLOCK_THREADS.
//
// Ops (inputs -> outputs, matrices column-major, joint order crocket's):
//   id       q, qd, qdd -> tau
//   crba     q          -> M
//   fd       q, qd, tau -> qdd                               (ABA)
//   fd_crba  q, qd, tau -> qdd = M^-1 (tau - rnea(q, qd, 0))  (Cholesky)
//   id_du    q, qd, qdd -> [dtau/dq | dtau/dqd]
//   minv     q          -> M^-1
//   fd_du    q, qd, tau -> [dqdd/dq | dqdd/dqd] = -M^-1 [dtau/dq | dtau/dqd] at qdd = fd
//
// Buffers: input (B, NIN) float32 row-major; output (NOUT, B) float32, coalesced stores.

#include <string>
#include <type_traits>

#include "glass_solve.cuh"
#include "ck_dynamics_gen.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace R = crocket::robots::robot;
using grim::dyn::N;
using real = GRIM_CK_REAL;

enum Op { ID, CRBA, FD, FD_CRBA, ID_DU, MINV, FD_DU };
__host__ __device__ constexpr int nin(Op o) { return o == CRBA || o == MINV ? N : 3 * N; }
__host__ __device__ constexpr int nout(Op o) { return o == CRBA || o == MINV ? N * N : o == ID_DU || o == FD_DU ? 2 * N * N : N; }
// Reals of one configuration's work: outputs, then temporaries.
__host__ __device__ constexpr int nwork(Op o) { return o == FD_CRBA ? N + N * N + 2 * N : o == MINV ? 2 * N * N : o == FD_DU ? 3 * N * N + N : nout(o); }

template <grim::Tier TIER>
struct Grp {
    int rank, size;
    __device__ Grp() : rank(TIER == grim::Tier::Thread ? 0 : TIER == grim::Tier::Warp ? (int)(threadIdx.x & 31u) : (int)threadIdx.x),
                       size(TIER == grim::Tier::Thread ? 1 : TIER == grim::Tier::Warp ? 32 : (int)blockDim.x) {}
    __device__ void sync() const
    {
        if constexpr (TIER == grim::Tier::Warp) __syncwarp();
        if constexpr (TIER == grim::Tier::Block) __syncthreads();
    }
};

constexpr int WARPS = 4;   // warp tier: at most this many configurations per block
constexpr size_t SMEM_MAX = 99 * 1024;

/** The block's crocket tables for OP at TIER, ahead of the configurations' slots. */
template <grim::Tier TIER, Op OP>
__host__ __device__ constexpr int tab_total()
{
    return TIER == grim::Tier::Warp ? grim::dyn::tab_warp_total<real>(OP)
         : TIER == grim::Tier::Block ? grim::dyn::tab_block_total<real>(OP) : 0;
}
__device__ __forceinline__ real* dyn_tables()
{
    extern __shared__ __align__(16) unsigned char dyn_smem_[];
    return reinterpret_cast<real*>(dyn_smem_);
}

// crocket's variant of TIER; the cooperative ones end without a barrier, so sync after. Their
// tables (one region per algorithm) are filled by the group first: identical writes from every
// group of the block, so no group waits on another.
#define CK_CALL(ALG, ...)                                                                       \
    do {                                                                                        \
        if constexpr (TIER == grim::Tier::Thread) R::ALG##_thread<real>(__VA_ARGS__, 1);        \
        else if constexpr (TIER == grim::Tier::Warp) {                                          \
            real* tb_ = dyn_tables() + grim::dyn::tab_warp_##ALG<real>(OP);                     \
            R::ALG##_warp_load_tables<real>(tb_, g.rank, g.size);                               \
            g.sync();                                                                           \
            R::ALG##_warp<real>(__VA_ARGS__, s, tb_, 1);                                        \
        } else {                                                                                \
            real* tb_ = dyn_tables() + grim::dyn::tab_block_##ALG<real>(OP);                    \
            R::ALG##_block_load_tables<real>(tb_, g.rank, g.size);                              \
            g.sync();                                                                           \
            R::ALG##_block<real>(__VA_ARGS__, s, tb_, 1);                                       \
        }                                                                                       \
        g.sync();                                                                               \
    } while (0)

/** In-place lower Cholesky of the column-major N x N `A` (one lane). */
__device__ __forceinline__ void chol(real* A)
{
    for (int j = 0; j < N; ++j) {
        real d = A[j + j * N];
        for (int k = 0; k < j; ++k) d -= A[j + k * N] * A[j + k * N];
        d = sqrt(d);
        A[j + j * N] = d;
        for (int i = j + 1; i < N; ++i) {
            real v = A[i + j * N];
            for (int k = 0; k < j; ++k) v -= A[i + k * N] * A[j + k * N];
            A[i + j * N] = v / d;
        }
    }
}

/** b <- (L L^T)^-1 b. */
__device__ __forceinline__ void chol_solve(const real* L, real* b)
{
    for (int i = 0; i < N; ++i) {
        real v = b[i];
        for (int k = 0; k < i; ++k) v -= L[i + k * N] * b[k];
        b[i] = v / L[i + i * N];
    }
    for (int i = N - 1; i >= 0; --i) {
        real v = b[i];
        for (int k = i + 1; k < N; ++k) v -= L[k + i * N] * b[k];
        b[i] = v / L[i + i * N];
    }
}

template <grim::Tier TIER, Op OP>
__device__ __forceinline__ void run_op(const Grp<TIER>& g, const real* x, real* y, real* s)
{
    if constexpr (OP == ID) CK_CALL(rnea, x, x + N, x + 2 * N, y);
    else if constexpr (OP == CRBA) CK_CALL(crba, x, y);
    // aba / rnea_derivatives are generated only when requested (never for mimic robots)
#ifdef GRIM_CK_HAS_aba
    else if constexpr (OP == FD) CK_CALL(aba, x, x + N, x + 2 * N, y);
#endif
#ifdef GRIM_CK_HAS_rnea_derivatives
    else if constexpr (OP == ID_DU) CK_CALL(rnea_derivatives, x, x + N, x + 2 * N, y, y + N * N);
#endif
    else if constexpr (OP == FD_CRBA) {
        real* M = y + N;
        real* zero = M + N * N;
        real* bias = zero + N;
        for (int i = g.rank; i < N; i += g.size) zero[i] = 0;
        g.sync();
        CK_CALL(crba, x, M);
        CK_CALL(rnea, x, x + N, zero, bias);
        if (g.rank == 0) {
            for (int i = 0; i < N; ++i) y[i] = x[2 * N + i] - bias[i];
            chol(M);
            chol_solve(M, y);
        }
    } else if constexpr (OP == MINV) {
        real* M = y + N * N;
        CK_CALL(crba, x, M);
        if (g.rank == 0) chol(M);
        g.sync();
        for (int j = g.rank; j < N; j += g.size) {   // column j of M^-1
            for (int i = 0; i < N; ++i) y[i + j * N] = i == j;
            chol_solve(M, y + j * N);
        }
    }
#if defined(GRIM_CK_HAS_aba) && defined(GRIM_CK_HAS_rnea_derivatives)
    else if constexpr (OP == FD_DU) {
        real* M = y + 2 * N * N;
        real* qdd = M + N * N;
        CK_CALL(aba, x, x + N, x + 2 * N, qdd);
        CK_CALL(rnea_derivatives, x, x + N, qdd, y, y + N * N);
        CK_CALL(crba, x, M);
        if (g.rank == 0) chol(M);
        g.sync();
        for (int j = g.rank; j < 2 * N; j += g.size) {
            real* c = y + j * N;
            for (int i = 0; i < N; ++i) c[i] = -c[i];
            chol_solve(M, c);
        }
    }
#endif
}

template <grim::Tier TIER, Op OP>
__global__ void dyn_kernel(const float* __restrict__ in, float* __restrict__ out, int batch, int wpb)
{
    const Grp<TIER> g;
    const int b = TIER == grim::Tier::Thread ? blockIdx.x * blockDim.x + threadIdx.x
                : TIER == grim::Tier::Warp ? blockIdx.x * wpb + (threadIdx.x >> 5) : blockIdx.x;
    if (b >= batch) return;   // group-uniform
    constexpr int NI = nin(OP), NW = nwork(OP), SLOT = NI + NW + grim::dyn::SMEM;
    real* x;
    real* y;
    real* s;
    real lx[TIER == grim::Tier::Thread ? NI : 1], ly[TIER == grim::Tier::Thread ? NW : 1];
    if constexpr (TIER == grim::Tier::Thread) {
        x = lx; y = ly; s = nullptr;
    } else {
        x = dyn_tables() + tab_total<TIER, OP>() + (TIER == grim::Tier::Warp ? (threadIdx.x >> 5) : 0) * SLOT;
        y = x + NI;
        s = y + NW;
    }
    for (int i = g.rank; i < NI; i += g.size) x[i] = in[(size_t)b * NI + i];
    g.sync();
    run_op<TIER, OP>(g, x, y, s);
    g.sync();
    for (int i = g.rank; i < nout(OP); i += g.size) out[(size_t)i * batch + b] = (float)y[i];
}

template <Op OP>
ffi::Error Run(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> in,
               ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, int64_t tier)
{
    if (in.dimensions().size() != 2 || in.dimensions()[1] != nin(OP))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "crocket dynamics: input is not (B, NIN)");
    const int batch = (int)in.dimensions()[0];
    if (batch == 0) return ffi::Error::Success();
    constexpr size_t slot = sizeof(real) * (nin(OP) + nwork(OP) + grim::dyn::SMEM);
    constexpr size_t tab_w = sizeof(real) * tab_total<grim::Tier::Warp, OP>(), tab_b = sizeof(real) * tab_total<grim::Tier::Block, OP>();
    if (tier != 0 && slot + (tier == 1 ? tab_w : tab_b) > SMEM_MAX)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "crocket dynamics: one configuration's work exceeds shared memory at the "
                          "cooperative tiers; use the thread tier");
    if (tier == 0) {
        dyn_kernel<grim::Tier::Thread, OP><<<(batch + 127) / 128, 128, 0, stream>>>(in.typed_data(), out->typed_data(), batch, 1);
    } else if (tier == 1) {
        const int fit = (int)((SMEM_MAX - tab_w) / slot);
        const int wpb = fit < WARPS ? fit : WARPS;
        cudaFuncSetAttribute(dyn_kernel<grim::Tier::Warp, OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(tab_w + wpb * slot));
        dyn_kernel<grim::Tier::Warp, OP><<<(batch + wpb - 1) / wpb, 32 * wpb, tab_w + wpb * slot, stream>>>(
            in.typed_data(), out->typed_data(), batch, wpb);
    } else {
        cudaFuncSetAttribute(dyn_kernel<grim::Tier::Block, OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(tab_b + slot));
        dyn_kernel<grim::Tier::Block, OP><<<batch, grim::dyn::BLOCK_THREADS, tab_b + slot, stream>>>(
            in.typed_data(), out->typed_data(), batch, 1);
    }
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success()
                            : ffi::Error(ffi::ErrorCode::kInternal, std::string("crocket dynamics: ") + cudaGetErrorString(e));
}

#define CK_DYN_HANDLER(NAME, OP)                                                               \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, Run<OP>,                                               \
        ffi::Ffi::Bind()                                                                       \
            .Ctx<ffi::PlatformStream<cudaStream_t>>()                                          \
            .Arg<ffi::Buffer<ffi::DataType::F32>>()                                            \
            .Ret<ffi::Buffer<ffi::DataType::F32>>()                                            \
            .Attr<int64_t>("tier"));

#define CK_DYN_OP(NAME, OP) CK_DYN_HANDLER(CkDyn_##NAME, OP)
GRIM_CK_DYN_OPS(CK_DYN_OP)
