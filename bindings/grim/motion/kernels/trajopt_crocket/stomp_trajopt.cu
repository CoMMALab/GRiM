/**
 * STOMP over crocket's traced collision cost, in `real` (the hand-written kernel this replaced
 * is at tag pre-crocket-trajopt): per iteration, evaluate K smooth-noise samples around each
 * trajectory, softmax-weight them by cost, step toward the weighted noise, and keep the best
 * sample seen; the result is that best sample, scored with w_collision_max.
 *
 * The sample evaluation, the one kernel that poses the robot, runs one sample per thread, warp or
 * block (`tier`), lanes over timesteps. The cooperative tiers stage the sample in shared memory as
 * the stock kernel does; the thread tier regenerates each noise value where it reads it. The
 * noise itself is the stock kernel's, in float, so a sample is the same in both precisions.
 * Softmax, update and best-tracking read no kinematics and keep their stock shapes, in `real`.
 */

#include "trajopt_crocket/tk.cuh"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>

namespace ffi = xla::ffi;
using namespace grim::tk;

#define STOMP_MAX_T 64
#define STOMP_MAX_K 512

static __device__ __forceinline__ float stomp_noise(uint64_t seed, int iter, int k, int t, int d, float scale)
{
    uint32_t s = (uint32_t)(seed ^ ((uint32_t)iter * 2654435761u) ^ ((uint32_t)k * 2246822519u)
                            ^ ((uint32_t)t * 3266489917u) ^ ((uint32_t)d * 1664525u));
    if (s == 0u) s = 1u;
    return rng_normal(s) * scale;
}

static __device__ __constant__ float FIR[7] = {0.07015f, 0.13107f, 0.19071f, 0.21608f, 0.19071f, 0.13107f, 0.07015f};

/** The stock kernel's smooth noise of sample k at (t, d): 0 for k = 0 and at the endpoints. */
static __device__ __forceinline__ float fir_noise(uint64_t seed, int iter, int k, int t, int d, int T, float scale)
{
    if (k == 0 || t == 0 || t >= T - 1) return 0.0f;
    const float sc = scale * 2.4721f;
    float e = 0.0f;
    for (int j = -3; j <= 3; ++j) {
        const int tj = t + j;
        e += FIR[j + 3] * ((tj > 0 && tj < T - 1) ? stomp_noise(seed, iter, k, tj, d, sc) : 0.0f);
    }
    return e;
}

__device__ __forceinline__ uint64_t traj_seed(uint64_t base, int b) { return base ^ ((uint64_t)b * 6364136223846793005ULL); }

/** Cost of sample (b, k) of `traj` into costs[b * K + k] (and out_f[b] when given): one sample
 *  per group, lanes over timesteps. */
template <grim::Tier TIER>
__global__ void ck_stomp_eval_kernel(
    const real* __restrict__ traj, real* __restrict__ costs, float* __restrict__ out_f, int B, int K, int T,
    uint64_t base_seed, int iter, float noise_scale, World world, const float* __restrict__ lower,
    const float* __restrict__ upper, real w_smooth, real w_acc, real w_jerk, real w_limits, real w_coll,
    real margin)
{
    const Grp<TIER> grp;
    const int gi = group_index<TIER>();
    extern __shared__ __align__(16) unsigned char tk_smem_[];
    if (gi >= B * K) return;   // group-uniform
    const int b = gi / K, k = gi % K, n = T * NA;
    const real* tb = traj + (size_t)b * n;
    const uint64_t seed = traj_seed(base_seed, b);
    real* red = reinterpret_cast<real*>(tk_smem_);
    real* sq = red + (TIER == grim::Tier::Block ? grp.size : 0) + group_slot<TIER>() * n;
    if constexpr (TIER != grim::Tier::Thread) {
        for (int i = grp.rank; i < n; i += grp.size)
            sq[i] = tb[i] + (real)fir_noise(seed, iter, k, i / NA, i % NA, T, noise_scale);
        grp.sync();
    }
    auto x = [&](int s, int d) -> real {
        if constexpr (TIER == grim::Tier::Thread) return tb[s * NA + d] + (real)fir_noise(seed, iter, k, s, d, T, noise_scale);
        else return sq[s * NA + d];
    };
    real c = 0;
    for (int t = grp.rank; t < T; t += grp.size) {
        c += smooth_cost_at(t, T, x, w_smooth, w_acc, w_jerk);
        real q[NA];
        for (int d = 0; d < NA; ++d) {
            q[d] = x(t, d);
            const real v = limit_viol(q[d], lower[d], upper[d]);
            c += w_limits * v * v;
        }
        c += w_coll * hinge_cost(q, world, margin);
    }
    c = reduce<TIER>(grp, red, c);
    if (grp.leader()) {
        costs[(size_t)b * K + k] = c;
        if (out_f) out_f[b] = (float)c;
    }
}

/** Fixed-order block tree over blockDim.x (a power of two) lanes; MIN keeps the argmin too. */
template <bool MIN>
__device__ __forceinline__ real block_tree(real v, real* sv, int* si, int& arg)
{
    const int k = threadIdx.x;
    sv[k] = v;
    if (MIN) si[k] = k;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (k < s) {
            if (MIN) { if (sv[k + s] < sv[k]) { sv[k] = sv[k + s]; si[k] = si[k + s]; } }
            else sv[k] += sv[k + s];
        }
        __syncthreads();
    }
    const real out = sv[0];
    if (MIN) arg = si[0];
    __syncthreads();
    return out;
}

/** Softmax weights of each trajectory's K sample costs (one block per trajectory). */
__global__ void ck_stomp_softmax_kernel(const real* __restrict__ costs, real* __restrict__ weights, int K, real temperature)
{
    extern __shared__ __align__(16) unsigned char sm_[];
    real* sv = reinterpret_cast<real*>(sm_);
    int* si = reinterpret_cast<int*>(sv + blockDim.x);
    const int b = blockIdx.x, k = threadIdx.x;
    int arg;
    const real c = k < K ? costs[(size_t)b * K + k] : real(1e30);
    const real mn = block_tree<true>(c, sv, si, arg);
    const real sh = k < K ? c - mn : real(0);
    const real mean = block_tree<false>(sh, sv, si, arg) / K;
    const real df = k < K ? sh - mean : real(0);
    const real sd = sqrt(block_tree<false>(df * df, sv, si, arg) / K);
    const real beta = fmax(sd, real(1e-6)) * temperature;
    const real w = k < K ? exp(-sh / (beta + real(1e-18))) : real(0);
    const real sw = block_tree<false>(w, sv, si, arg);
    if (k < K) weights[(size_t)b * K + k] = w / (sw + real(1e-30));
}

/** traj += step * (weighted noise), clamped to the limits (one block per interior timestep). */
__global__ void ck_stomp_update_kernel(real* __restrict__ traj, const real* __restrict__ weights,
                                       const float* __restrict__ lower, const float* __restrict__ upper,
                                       uint64_t base_seed, int iter, float noise_scale, int K, int T, real step)
{
    extern __shared__ __align__(16) unsigned char sm_[];
    real* sv = reinterpret_cast<real*>(sm_);
    const int b = blockIdx.x, t = blockIdx.y + 1, k = threadIdx.x;
    const uint64_t seed = traj_seed(base_seed, b);
    const real w = k < K ? weights[(size_t)b * K + k] : real(0);
    int arg;
    for (int d = 0; d < NA; ++d) {
        const real nv = k < K ? (real)fir_noise(seed, iter, k, t, d, T, noise_scale) : real(0);
        const real dl = block_tree<false>(w * nv, sv, nullptr, arg);
        if (k == 0) {
            real& v = traj[(size_t)b * T * NA + t * NA + d];
            v = fmax(real(lower[d]), fmin(real(upper[d]), v + step * dl));
        }
    }
}

/** best <- this iteration's best sample of `traj` when it beats best_cost. */
__global__ void ck_stomp_track_best_kernel(const real* __restrict__ traj, const real* __restrict__ costs,
                                           real* __restrict__ best, real* __restrict__ best_cost,
                                           uint64_t base_seed, int iter, float noise_scale, int K, int T)
{
    extern __shared__ __align__(16) unsigned char sm_[];
    real* sv = reinterpret_cast<real*>(sm_);
    int* si = reinterpret_cast<int*>(sv + blockDim.x);
    const int b = blockIdx.x, k = threadIdx.x;
    int kb;
    const real c = block_tree<true>(k < K ? costs[(size_t)b * K + k] : real(1e30), sv, si, kb);
    if (c < best_cost[b]) {   // block-uniform
        const uint64_t seed = traj_seed(base_seed, b);
        for (int i = k; i < T * NA; i += blockDim.x)
            best[(size_t)b * T * NA + i] = traj[(size_t)b * T * NA + i]
                + (real)fir_noise(seed, iter, kb, i / NA, i % NA, T, noise_scale);
        __syncthreads();
        if (k == 0) best_cost[b] = c;
    }
}

__global__ void ck_stomp_init_kernel(const float* __restrict__ init, real* __restrict__ traj, real* __restrict__ best,
                                     real* __restrict__ best_cost, int B, int n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)B * n) traj[i] = best[i] = init[i];
    if (i < (size_t)B) best_cost[i] = real(1e30);
}

__global__ void ck_stomp_out_kernel(const real* __restrict__ best, float* __restrict__ out, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)best[i];
}

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static ffi::Error CkStompTrajoptImpl(
    cudaStream_t stream, GRIM_ROT_PARAM
    ffi::Buffer<ffi::DataType::F32> init, ffi::Buffer<ffi::DataType::F32> ws,
    ffi::Buffer<ffi::DataType::F32> wc, ffi::Buffer<ffi::DataType::F32> wb,
    ffi::Buffer<ffi::DataType::F32> wh, ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper, ffi::Buffer<ffi::DataType::F32> start,
    ffi::Buffer<ffi::DataType::F32> goal,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_trajs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_costs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> workspace,
    int64_t n_iters, int64_t n_samples, float noise_scale, float temperature, float step_size,
    float w_smooth, float w_acc, float w_jerk, float w_limits, float w_collision, float w_collision_max,
    float penalty_scale, float margin, int64_t rng_seed, int64_t tier, int64_t block_threads)
{
    GRIM_ROT_UPLOAD(stream);
    (void)start; (void)goal;   // the launcher pins them into init_trajs, as for the stock kernel
    const int B = (int)init.dimensions()[0], T = (int)init.dimensions()[1], K = (int)n_samples;
    if ((int)init.dimensions()[2] != NA || ws.dimensions()[0] != n_ws || wc.dimensions()[0] != n_wc ||
        wb.dimensions()[0] != n_wb || wh.dimensions()[0] != n_wh)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "stomp_trajopt (crocket): launch does not match the robot and obstacle "
                          "counts of this build");
    if (K < 1 || K > STOMP_MAX_K || T > STOMP_MAX_T)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "stomp_trajopt (crocket): n_samples > 512 or T > 64");
    const size_t n = (size_t)T * NA;
    if (workspace->element_count() * sizeof(float) < ((size_t)B * (2 * n + 1 + 2 * K)) * sizeof(real))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "stomp_trajopt (crocket): workspace too small");
    if (B == 0 || T == 0) return ffi::Error::Success();

    real* traj = reinterpret_cast<real*>(workspace->typed_data());
    real* best = traj + B * n;
    real* best_cost = best + B * n;
    real* costs = best_cost + B;
    real* weights = costs + (size_t)B * K;
    const World w{ws.typed_data(), wc.typed_data(), wb.typed_data(), wh.typed_data()};
    const float* lo = lower.typed_data();
    const float* hi = upper.typed_data();
    const uint64_t seed = (uint64_t)rng_seed;

    ck_stomp_init_kernel<<<(unsigned)((B * n + 255) / 256), 256, 0, stream>>>(init.typed_data(), traj, best, best_cost, B, (int)n);
    const int Kb = next_pow2(K);
    auto smem = [&](grim::Tier t) {
        return t == grim::Tier::Block ? ((size_t)block_threads + n) * sizeof(real)
             : t == grim::Tier::Warp ? (size_t)WARPS * n * sizeof(real) : (size_t)0;
    };
    real w_coll = w_collision;
    for (int it = 0; it < (int)n_iters; ++it) {
        TK_LAUNCH(ck_stomp_eval_kernel, tier, B * K, (int)block_threads, smem, stream,
                  traj, costs, nullptr, B, K, T, seed, it, noise_scale, w, lo, hi, w_smooth, w_acc, w_jerk,
                  w_limits, w_coll, margin);
        ck_stomp_softmax_kernel<<<B, Kb, Kb * (sizeof(real) + sizeof(int)), stream>>>(costs, weights, K, temperature);
        if (T > 2)
            ck_stomp_update_kernel<<<dim3(B, T - 2), Kb, Kb * sizeof(real), stream>>>(
                traj, weights, lo, hi, seed, it, noise_scale, K, T, step_size);
        ck_stomp_track_best_kernel<<<B, Kb, Kb * (sizeof(real) + sizeof(int)), stream>>>(
            traj, costs, best, best_cost, seed, it, noise_scale, K, T);
        w_coll = fmin(w_coll * penalty_scale, (real)w_collision_max);
    }
    // The best sample, scored without noise (sample 0) at w_collision_max.
    TK_LAUNCH(ck_stomp_eval_kernel, tier, B, (int)block_threads, smem, stream,
              best, costs, out_costs->typed_data(), B, 1, T, seed, 0, noise_scale, w, lo, hi, w_smooth,
              w_acc, w_jerk, w_limits, w_collision_max, margin);
    ck_stomp_out_kernel<<<(unsigned)((B * n + 255) / 256), 256, 0, stream>>>(best, out_trajs->typed_data(), B * n);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success() : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CkStompTrajoptFfi, CkStompTrajoptImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        GRIM_ROT_BIND
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // init_trajs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // start
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // goal
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_trajs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_costs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // workspace (real)
        .Attr<int64_t>("n_iters")
        .Attr<int64_t>("n_samples")
        .Attr<float>("noise_scale")
        .Attr<float>("temperature")
        .Attr<float>("step_size")
        .Attr<float>("w_smooth")
        .Attr<float>("w_acc")
        .Attr<float>("w_jerk")
        .Attr<float>("w_limits")
        .Attr<float>("w_collision")
        .Attr<float>("w_collision_max")
        .Attr<float>("collision_penalty_scale")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("rng_seed")
        .Attr<int64_t>("tier")
        .Attr<int64_t>("block_threads"));
