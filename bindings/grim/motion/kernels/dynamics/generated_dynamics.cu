// XLA FFI handlers for a robot's GRiM-generated dynamics kernels (grim.cuh), for JAX callers
// that need any batch size: workspace comes from XLA's scratch allocator per call, and the
// robot model is cached per device. (grim.jax's generated handles instead copy through
// buffers preallocated at the build's max_batch.) Built by grim.motion.generated_dynamics.
//
// The generated __global__ kernels are launched directly on the XLA stream; the __host__
// wrappers are bypassed because they do their own transfers and synchronization. The launch
// contract copied from them:
//   * dynamic shared memory is sized by <ALGO>_DYNAMIC_SHARED_MEM_BYTES<T>();
//   * every kernel except inverse_dynamics takes a global spill workspace sized
//     GRIM_WORKSPACE_BYTES_PER_TIMESTEP * GRIM_WORKSPACE_SLOTS per timestep;
//   * the dynamics kernels take a `T *d_f_ext` buffer, passed as nullptr here.
//
// Batch mapping: the batch dimension B is the kernels' NUM_TIMESTEPS. Inputs arrive as
// separate (B, NUM_POS) buffers (velocity-like ones zero-padded past NUM_VEL) and are
// interleaved into the [q | qd (| u/qdd)] per-timestep layout by small pack kernels.
// Build flags: GRIM_GEN_DYN_FLOATING_BASE (free-flyer root), GRIM_GEN_DYN_RUNTIME_INERTIA
// (exports DynInertiaParamsSize / DynSetInertiaParams for a header generated with
// runtime_inertia=True).

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "grim.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using T = float;

constexpr int kNq = grim::NUM_JOINTS;

// ---------------------------------------------------------------------------
// Per-device robot model cache (mirrors the FK kernel's ModelCache pattern).
// ---------------------------------------------------------------------------

template <typename Kernel>
void AllowSmem(Kernel kernel, size_t bytes) {
  if (bytes > 48 * 1024)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
}

grim::robotModel<T>* GetRobotModel() {
  static std::mutex mu;
  static std::unordered_map<int, grim::robotModel<T>*> models;
  int device = -1;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(mu);
  auto it = models.find(device);
  if (it != models.end()) return it->second;
  grim::robotModel<T>* model = grim::init_robotModel<T>();
  // Allow the gradient kernels to exceed the default dynamic shared memory
  // limit on large robots (replicates grim::init_grid's cudaFuncSetAttribute
  // calls for the kernel instantiations used here). The overload set is
  // disambiguated by taking the address through an exactly-typed pointer.
  void (*id_du_kern)(T*, unsigned char*, const T*, int, const T*, T*,
                     const grim::robotModel<T>*, const T, const int) =
      &grim::inverse_dynamics_gradient_kernel<T>;
  void (*fd_du_kern)(T*, unsigned char*, const T*, int, T*,
                     const grim::robotModel<T>*, const T, const int) =
      &grim::forward_dynamics_gradient_kernel<T>;
  cudaFuncSetAttribute(id_du_kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       grim::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>());
  cudaFuncSetAttribute(fd_du_kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       grim::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>());
  // A free-flyer's arenas can pass the default 48 KB too; opt every kernel in.
  AllowSmem(&grim::forward_dynamics_kernel<T>,
            grim::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>());
  AllowSmem(&grim::minv_kernel<T>, grim::MINV_DYNAMIC_SHARED_MEM_BYTES<T>());
  AllowSmem(&grim::crba_kernel<T>, grim::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>());
  models[device] = model;
  return model;
}

// ---------------------------------------------------------------------------
// Pack kernels: (B, n) x k separate buffers -> interleaved stride-(k*n).
// ---------------------------------------------------------------------------

__global__ void Pack2Kernel(T* dst, const T* a, const T* b, int n, int batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch * n) return;
  int k = idx / n, i = idx % n;
  dst[k * 2 * n + i] = a[idx];
  dst[k * 2 * n + n + i] = b[idx];
}

__global__ void Pack3Kernel(T* dst, const T* a, const T* b, const T* c, int n,
                            int batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch * n) return;
  int k = idx / n, i = idx % n;
  dst[k * 3 * n + i] = a[idx];
  dst[k * 3 * n + n + i] = b[idx];
  dst[k * 3 * n + 2 * n + i] = c[idx];
}

inline dim3 LaunchDims(int batch) {
  // The GRiD kernels use a grid-stride loop over timesteps.
  return dim3(static_cast<unsigned>(batch < 65535 ? batch : 65535), 1, 1);
}

inline dim3 ThreadDims() {
  return dim3(static_cast<unsigned>(grim::MAX_PERF_LEVEL_THREADS), 1, 1);
}

inline int PackBlocks(int total) { return (total + 255) / 256; }

ffi::Error CudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string(what) + ": " + cudaGetErrorString(err));
  }
  return ffi::Error::Success();
}

ffi::Error CheckDims(int64_t batch, int64_t n, int64_t elems) {
  if (n != kNq || elems != batch * n) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "buffer shape mismatch against NUM_JOINTS");
  }
  return ffi::Error::Success();
}

// Call-scoped device scratch, sourced from XLA's own allocator.
//
// These used to live on private `cudaMallocAsync`/`cudaFreeAsync` calls on
// the XLA stream. That put them in a separate memory pool from JAX/XLA, so
// with XLA preallocating the device up front the FFI mallocs competed for
// the sliver XLA left behind and eventually OOM'd -- the same failure mode
// already found and fixed for the collision kernels (see the comment above
// `scratch_alloc_void` in _robogpu_collision_host.cu). Instead we draw them
// from `ffi::ScratchAllocator`, which allocates stream-ordered from XLA's
// device allocator and reclaims everything when the handler returns.
struct ScratchBuffer {
  T* ptr = nullptr;
  ffi::Error err = ffi::Error::Success();
  ScratchBuffer(ffi::ScratchAllocator& scratch, size_t count) {
    auto p = scratch.Allocate(count * sizeof(T), alignof(T));
    if (!p.has_value()) {
      err = ffi::Error(ffi::ErrorCode::kResourceExhausted,
                       "scratch.Allocate(q_qd) failed");
      return;
    }
    ptr = reinterpret_cast<T*>(*p);
  }
};

// The global spill workspace every non-inverse_dynamics kernel now takes.
// Sized exactly as grim::init_gridData does. It is scratch: the kernels only
// use it to spill what does not fit in the shared arena at the chosen resource
// tier, so a stream-ordered per-call allocation is correct (and at TIER_SHARED
// several kernels never touch it at all).
struct WorkspaceBuffer {
  unsigned char* ptr = nullptr;
  ffi::Error err = ffi::Error::Success();
  WorkspaceBuffer(ffi::ScratchAllocator& scratch, int64_t batch) {
    const size_t bytes = grim::GRIM_WORKSPACE_BYTES_PER_TIMESTEP<T>() *
                         GRIM_WORKSPACE_SLOTS * static_cast<size_t>(batch);
    if (bytes == 0) return;
    auto p = scratch.Allocate(bytes, alignof(std::max_align_t));
    if (!p.has_value()) {
      err = ffi::Error(ffi::ErrorCode::kResourceExhausted,
                       "scratch.Allocate(workspace) failed");
      return;
    }
    ptr = reinterpret_cast<unsigned char*>(*p);
  }
};

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

// Inverse dynamics: (q, qd, qdd) -> joint torques c.  All (B, n).
ffi::Error DynIdImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                      float gravity,
                      ffi::Buffer<ffi::DataType::F32> q,
                      ffi::Buffer<ffi::DataType::F32> qd,
                      ffi::Buffer<ffi::DataType::F32> qdd,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> c) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grim::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd(scratch, 2 * kNq * batch);
  if (q_qd.err.failure()) return q_qd.err;
  Pack2Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd.ptr, q.typed_data(), qd.typed_data(), kNq, batch);
  grim::inverse_dynamics_kernel<T>
      <<<LaunchDims(batch), ThreadDims(),
         grim::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          c->typed_data(), q_qd.ptr, 2 * kNq, qdd.typed_data(),
          /*d_f_ext=*/nullptr, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "inverse_dynamics_kernel");
}

// Forward dynamics: (q, qd, u) -> joint accelerations qdd.  All (B, n).
ffi::Error DynFdImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                      float gravity,
                      ffi::Buffer<ffi::DataType::F32> q,
                      ffi::Buffer<ffi::DataType::F32> qd,
                      ffi::Buffer<ffi::DataType::F32> u,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> qdd) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grim::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_u(scratch, 3 * kNq * batch);
  if (q_qd_u.err.failure()) return q_qd_u.err;
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_u.ptr, q.typed_data(), qd.typed_data(), u.typed_data(), kNq, batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grim::forward_dynamics_kernel<T>
      <<<LaunchDims(batch), ThreadDims(),
         grim::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          qdd->typed_data(), ws.ptr, q_qd_u.ptr, 3 * kNq, /*d_f_ext=*/nullptr,
          model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "forward_dynamics_kernel");
}

// Direct Minv: q -> inverse mass matrix, (B, n, n), SYMMETRIC_UPPER filled.
ffi::Error DynMinvImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                        ffi::Buffer<ffi::DataType::F32> q,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> minv) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grim::robotModel<T>* model = GetRobotModel();

  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grim::minv_kernel<T>
      <<<LaunchDims(batch), ThreadDims(),
         grim::MINV_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          minv->typed_data(), ws.ptr, q.typed_data(), kNq, model, batch);
  return CudaCheck(cudaGetLastError(), "minv_kernel");
}

// Mass matrix M(q): q -> (B, n, n), [t, col, row] fully populated.
// Mass matrix M(q) via GRiD's own generated CRBA kernel (BFS-parallel
// composite-inertia accumulation), rather than the old n-column ID sweep.
// crba_kernel still takes a [q|qd] interleaved input (stride 2n) and a
// gravity scalar even though M(q) does not depend on either; qd is packed
// as zero here to match ID/FD's packing convention with no extra kernel.
ffi::Error DynCrbaImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                        float gravity,
                        ffi::Buffer<ffi::DataType::F32> q,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> m) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grim::robotModel<T>* model = GetRobotModel();

  ScratchBuffer qd_zero(scratch, kNq * batch);
  if (qd_zero.err.failure()) return qd_zero.err;
  if (auto err = CudaCheck(
          cudaMemsetAsync(qd_zero.ptr, 0, kNq * batch * sizeof(T), stream),
          "crba qd memset");
      err.failure())
    return err;
  ScratchBuffer q_qd(scratch, 2 * kNq * batch);
  if (q_qd.err.failure()) return q_qd.err;
  Pack2Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd.ptr, q.typed_data(), qd_zero.ptr, kNq, batch);

  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grim::crba_kernel<T><<<LaunchDims(batch), ThreadDims(),
                        grim::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
      m->typed_data(), ws.ptr, q_qd.ptr, 2 * kNq, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "crba_kernel");
}

// Analytic inverse dynamics gradient: (q, qd, qdd) -> dc/d[q,qd],
// (B, 2n, n) with column-major n x 2n per timestep ([dq block | dqd block]).
ffi::Error DynIdGradImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                          float gravity,
                          ffi::Buffer<ffi::DataType::F32> q,
                          ffi::Buffer<ffi::DataType::F32> qd,
                          ffi::Buffer<ffi::DataType::F32> qdd,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> dc_du) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grim::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd(scratch, 2 * kNq * batch);
  if (q_qd.err.failure()) return q_qd.err;
  Pack2Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd.ptr, q.typed_data(), qd.typed_data(), kNq, batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grim::inverse_dynamics_gradient_kernel<T>
      <<<LaunchDims(batch), ThreadDims(),
         grim::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
         stream>>>(dc_du->typed_data(), ws.ptr, q_qd.ptr, 2 * kNq,
                   qdd.typed_data(), /*d_f_ext=*/nullptr, model, gravity,
                   batch);
  return CudaCheck(cudaGetLastError(), "inverse_dynamics_gradient_kernel");
}

// Analytic forward dynamics gradient: (q, qd, u) -> dqdd/d[q,qd],
// (B, 2n, n) with column-major n x 2n per timestep.
ffi::Error DynFdGradImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                          float gravity,
                          ffi::Buffer<ffi::DataType::F32> q,
                          ffi::Buffer<ffi::DataType::F32> qd,
                          ffi::Buffer<ffi::DataType::F32> u,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> df_du) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
  grim::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_u(scratch, 3 * kNq * batch);
  if (q_qd_u.err.failure()) return q_qd_u.err;
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_u.ptr, q.typed_data(), qd.typed_data(), u.typed_data(), kNq, batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grim::forward_dynamics_gradient_kernel<T>
      <<<LaunchDims(batch), ThreadDims(),
         grim::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
         stream>>>(df_du->typed_data(), ws.ptr, q_qd_u.ptr, 3 * kNq,
                   /*d_f_ext=*/nullptr, model, gravity, batch);
  return CudaCheck(cudaGetLastError(), "forward_dynamics_gradient_kernel");
}

// Second-order inverse dynamics (GRiD idsva_so, body-frame fixed-base
// dispatch): (q, qd, qdd) -> 4 flattened (n,n,n) tensors per timestep,
// concatenated as [d2tau_dq | d2tau_dqd | d2tau_cross | dM_dq], each n^3
// floats (SECOND_ORDER_TENSOR_SIZE = 4*n^3 total). Same [q|qd|qdd] packing
// as DynFdGradImpl (stride 3n = Q_QD_U_STRIDE).
ffi::Error DynIdsvaSoImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                           float gravity,
                           ffi::Buffer<ffi::DataType::F32> q,
                           ffi::Buffer<ffi::DataType::F32> qd,
                           ffi::Buffer<ffi::DataType::F32> qdd,
                           ffi::Result<ffi::Buffer<ffi::DataType::F32>> out) {
  const int64_t batch = q.dimensions()[0];
  const int64_t n = q.dimensions()[1];
  if (auto err = CheckDims(batch, n, q.element_count()); err.failure())
    return err;
#ifdef GRIM_GEN_DYN_FLOATING_BASE
  // GRiD emits no body-frame idsva_so for a free-flyer; the Python side refuses second
  // derivatives through a floating base before reaching here.
  (void)stream; (void)scratch; (void)gravity; (void)qd; (void)qdd; (void)out;
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "idsva_so: no body-frame second-order kernel for a floating base");
#else
  grim::robotModel<T>* model = GetRobotModel();

  ScratchBuffer q_qd_qdd(scratch, 3 * kNq * batch);
  if (q_qd_qdd.err.failure()) return q_qd_qdd.err;
  Pack3Kernel<<<PackBlocks(batch * kNq), 256, 0, stream>>>(
      q_qd_qdd.ptr, q.typed_data(), qd.typed_data(), qdd.typed_data(), kNq,
      batch);
  WorkspaceBuffer ws(scratch, batch);
  if (ws.err.failure()) return ws.err;
  grim::idsva_so_body_frame_kernel<T>
      <<<LaunchDims(batch), ThreadDims(),
         grim::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
          out->typed_data(), ws.ptr, q_qd_qdd.ptr, 3 * kNq, model, gravity,
          batch);
  return CudaCheck(cudaGetLastError(), "idsva_so_body_frame_kernel");
#endif
}

}  // namespace

#ifdef GRIM_GEN_DYN_RUNTIME_INERTIA
// Runtime-mutable inertia table (a header generated with runtime_inertia=True).
//
// These are NOT FFI handlers and are deliberately not stream-ordered:
// set_inertia_params is a blocking host->device memcpy into device-resident *model* state,
// which is not a traceable JAX value. Call it between launches, never under a trace.
extern "C" int DynInertiaParamsSize() { return 10 * grim::NUM_JOINTS; }

extern "C" void DynSetInertiaParams(const float* h_params) {
  grim::robotModel<T>* model = GetRobotModel();
  cudaDeviceSynchronize();  // the table is read by any in-flight kernel launch
  grim::set_inertia_params<T>(model, h_params);
}
#endif  // GRIM_GEN_DYN_RUNTIME_INERTIA

XLA_FFI_DEFINE_HANDLER_SYMBOL(DynIdFfi, DynIdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DynFdFfi, DynFdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // u
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DynMinvFfi, DynMinvImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DynCrbaFfi, DynCrbaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DynIdGradFfi, DynIdGradImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DynFdGradFfi, DynFdGradImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // u
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DynIdsvaSoFfi, DynIdsvaSoImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<float>("gravity")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qd
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // qdd
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>());
