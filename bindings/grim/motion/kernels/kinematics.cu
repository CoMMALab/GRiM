/**
 * Batched kinematics of the built robot: every joint frame, and the IK pose residual and
 * geometric Jacobian of its end-effectors.
 *
 * This is the IK counterpart of GRiD's separate gradient kernels: the solvers mutate their
 * in-loop J (pose weights, column scaling) and evaluate it at the iterate, not the winner,
 * so derivatives of a solution come from this kernel at the returned configuration. It is
 * also the primitive the motion equivalence suite checks against the numpy oracle -- every
 * solver reads the robot through exactly these functions (robot.cuh).
 *
 * The residual is [p - p*, log(q q*^-1)] (world-frame position difference + quaternion
 * error), NOT the SE(3) local log map; callers differentiating through it must use the same
 * convention.
 */

#include "robot.cuh"

#include <xla/ffi/api/ffi.h>
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

// One thread per configuration: one FK sweep plus a Jacobian fill, no cooperation.
__global__ void kinematics_kernel(
    const float* __restrict__ q,         // (B, n_q)
    const float* __restrict__ targets,   // (B, n_ee, 7)
    int n_batch,
    float* __restrict__ out_T,           // (B, n_frames, 7)
    float* __restrict__ out_r,           // (B, 6 n_ee)
    float* __restrict__ out_J)           // (B, 6 n_ee, n_solved)
{
    using namespace grim::robot;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) return;

    float cfg[n_solved > 0 ? n_solved : 1], frz[frz_len];
    load_state(q + (size_t)b * n_q, cfg, frz);

    float T[n_frames * 7];
    frame_poses(cfg, frz, T);
    for (int k = 0; k < n_frames * 7; ++k) out_T[(size_t)b * n_frames * 7 + k] = T[k];

    if constexpr (n_ee > 0) {
        float r[6 * n_ee], J[6 * n_ee * (n_solved > 0 ? n_solved : 1)];
        residual_and_jacobian(cfg, frz, targets + (size_t)b * n_ee * 7, r, J);
        for (int k = 0; k < 6 * n_ee; ++k) out_r[(size_t)b * 6 * n_ee + k] = r[k];
        for (int k = 0; k < 6 * n_ee * n_solved; ++k)
            out_J[(size_t)b * 6 * n_ee * n_solved + k] = J[k];
    }
}

static ffi::Error KinematicsImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> q,
    ffi::Buffer<ffi::DataType::F32> targets,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_T,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_r,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_J)
{
    const int n_batch = static_cast<int>(q.dimensions()[0]);
    if (q.dimensions()[1] != grim::robot::n_q)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "kinematics: q width does not match the robot this kernel was built for");
    if (n_batch == 0) return ffi::Error::Success();
    const int block = 128;
    kinematics_kernel<<<(n_batch + block - 1) / block, block, 0, stream>>>(
        q.typed_data(), targets.typed_data(), n_batch,
        out_T->typed_data(), out_r->typed_data(), out_J->typed_data());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KinematicsFfi, KinematicsImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // q
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // targets
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // frame poses
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // residual
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // jacobian
