// Each thread owns a configuration; only sphere centres leave the kernel. Centres are staged in
// shared memory (odd per-thread stride: no bank conflicts) and written back by the whole block,
// so the (B, S, 3) store is coalesced; direct per-thread stores were ~10x slower at large B.
#include "robot.cuh"
#include <xla/ffi/api/ffi.h>
namespace ffi = xla::ffi;

constexpr int kRow = grim::robot::sphere_count * 3, kStride = kRow | 1;
constexpr int kMaxThreads = (48 * 1024 / (kStride * 4)) / 32 * 32;
static_assert(kMaxThreads >= 32, "sphere count too large for shared-memory staging");

__global__ void sphere_fk_kernel(const float* q, float* out, int batch) {
    extern __shared__ float stage[];
    int first = blockIdx.x * blockDim.x, b = first + threadIdx.x;
    if (b < batch) grim::robot::sphere_centres(q + (size_t)b*grim::robot::n_q, stage + threadIdx.x*kStride);
    __syncthreads();
    int n = min((int)blockDim.x, batch - first) * kRow;
    float* dst = out + (size_t)first * kRow;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = stage[(i / kRow) * kStride + i % kRow];
}

static ffi::Error SphereFkImpl(cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> q,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, int threads) {
    if (threads < 1 || threads > 1024)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "threads must be in [1, 1024]");
    if (q.dimensions().size() != 2 || q.dimensions()[1] != grim::robot::n_q)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "sphere FK expects (batch, n_q)");
    int batch = q.dimensions()[0];
    threads = threads < kMaxThreads ? threads : kMaxThreads;
    if (batch && grim::robot::sphere_count)
        sphere_fk_kernel<<<(batch+threads-1)/threads, threads, threads*kStride*sizeof(float), stream>>>(
            q.typed_data(), out->typed_data(), batch);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(SphereFkFfi, SphereFkImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>().Attr<int>("threads"));
