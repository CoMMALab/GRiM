// Each thread owns a configuration; only sphere centres leave the kernel.
#include "robot.cuh"
#include <xla/ffi/api/ffi.h>
namespace ffi = xla::ffi;

__global__ void sphere_fk_kernel(const float* q, float* out, int batch) {
    using namespace grim::robot;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) sphere_centres(q + (size_t)b*n_q, out + (size_t)b*sphere_count*3);
}

static ffi::Error SphereFkImpl(cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> q,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, int threads) {
    if (threads < 1 || threads > 1024)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "threads must be in [1, 1024]");
    if (q.dimensions().size() != 2 || q.dimensions()[1] != grim::robot::n_q)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "sphere FK expects (batch, n_q)");
    int batch = q.dimensions()[0];
    if (batch && grim::robot::sphere_count)
        sphere_fk_kernel<<<(batch+threads-1)/threads, threads, 0, stream>>>(
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
