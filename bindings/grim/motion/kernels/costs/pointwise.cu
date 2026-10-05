/**
 * One generated function applied row-wise, one thread per row: O[i] = f(X[i]).
 *
 * `pointwise_gen.cuh` is written by a problem compiler (e.g. pyroffi.costs._pointwise, from a jaxpr) (fully scalarized,
 * thread-tier form). It defines n_in, n_out, scratch_size, the constant tables and
 *
 *   pointwise(x, o, scratch, rank)
 */

#include "xla/ffi/api/ffi.h"

#include <cmath>

#define GRIM_COST_SYNC() ((void)0)
using real = double;

#include "pointwise_gen.cuh"

namespace ffi = xla::ffi;
namespace gen = grim::pointwise_gen;

__global__ void pointwise_kernel(const real* __restrict__ X, real* __restrict__ O, size_t n)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    real scratch[gen::scratch_size > 0 ? gen::scratch_size : 1];
    gen::pointwise(X + i * gen::n_in, O + i * gen::n_out, scratch, 0);
}

static ffi::Error PointwiseImpl(cudaStream_t st, ffi::Buffer<ffi::F64> X, ffi::ResultBuffer<ffi::F64> O)
{
    const size_t n = X.element_count() / gen::n_in;
    if (n) pointwise_kernel<<<(n + 63) / 64, 64, 0, st>>>(X.typed_data(), O->typed_data(), n);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success()
                            : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(PointwiseFfi, PointwiseImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());
