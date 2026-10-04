/**
 * The robot a motion kernel was built for: one device API over the per-robot header.
 *
 * Every GRiM motion kernel is compiled for ONE robot. `grim_robot_gen.cuh` is written by
 * grim.motion._build and holds, in namespace grim::robot:
 *
 *   n_frames, n_q, n_solved, n_frozen, n_ee      compile-time sizes
 *   kTwists, kParentTf, kParentIdx, kActIdx,     the kinematic tables (one row per URDF
 *   kMimicMul, kMimicOff, kMimicActIdx, kTopoInv joint), as __constant__ arrays
 *   ee_joint(e), kAncestor                       end-effector frames and their chains
 *   solved_idx(i), frozen_idx(k)                 solved variable / frozen slot -> q index
 *   collision tables + world obstacle counts     (see grim.motion._build)
 *
 * and, when built with GRIM_TRACED_KINEMATICS, cricket's straight-line FK of the same robot
 * (namespace `traced`: frame_poses, and for one end-effector ee_pose_jacobian unless
 * GRIM_TRACED_FRAMES_ONLY) with the maps gather_q()/scatter_jacobian() between cricket's joint
 * order and ours.
 *
 * SOLVED vs FROZEN: a kernel optimizes `n_solved` variables (`cfg`) and carries the other
 * `n_frozen` actuated joints unchanged from the seed (`frz`). They differ only for an IK
 * build without collision, which solves over the end-effector chains alone; everywhere
 * else n_solved == n_q and frz is empty.
 *
 * All poses are [qw, qx, qy, qz, x, y, z]; T_world is (n_frames, 7) in joint order; J is
 * (6 * n_ee, n_solved) row-major. grim.motion.reference.kinematics is the float64 oracle
 * for every function here.
 */
#pragma once

#include "ik_common.cuh"
#include "grim_robot_gen.cuh"

namespace grim::robot {

constexpr int frz_len = n_frozen > 0 ? n_frozen : 1;   // size of a kernel's frz[] array

/** Every actuated joint (q order) from the solved variables and the frozen joints. */
static __device__ __forceinline__ void full_q(
    const float* __restrict__ cfg, const float* __restrict__ frz, float* __restrict__ q)
{
    for (int i = 0; i < n_solved; ++i) q[solved_idx(i)] = cfg[i];
    for (int k = 0; k < n_frozen; ++k) q[frozen_idx(k)] = frz[k];
    (void)frz;
}

/** One seed/output row (every actuated joint) <-> solved variables + frozen joints. */
static __device__ __forceinline__ void load_state(
    const float* __restrict__ row, float* __restrict__ cfg, float* __restrict__ frz)
{
    for (int i = 0; i < n_solved; ++i) cfg[i] = row[solved_idx(i)];
    for (int k = 0; k < n_frozen; ++k) frz[k] = row[frozen_idx(k)];
    (void)frz;
}

static __device__ __forceinline__ void store_state(
    const float* __restrict__ cfg, const float* __restrict__ frz, float* __restrict__ row)
{
    for (int i = 0; i < n_solved; ++i) row[solved_idx(i)] = cfg[i];
    for (int k = 0; k < n_frozen; ++k) row[frozen_idx(k)] = frz[k];
    (void)frz;
}

/** World pose of every joint frame. */
static __device__ __forceinline__ void frame_poses(
    const float* __restrict__ cfg, const float* __restrict__ frz, float* __restrict__ T_world)
{
#ifdef GRIM_TRACED_KINEMATICS
    float q[traced::n_q];
    gather_q(cfg, frz, q);
    traced::frame_poses(q, T_world);
#else
    float q[n_q > 0 ? n_q : 1];
    full_q(cfg, frz, q);
    fk_single(q, kTwists, kParentTf, kParentIdx, kActIdx, kMimicMul, kMimicOff,
              kMimicActIdx, kTopoInv, T_world, n_frames, n_q);
#endif
}

/** Stacked pose residuals r (6 * n_ee) against `targets` (n_ee, 7). */
static __device__ __forceinline__ void residual(
    const float* __restrict__ cfg, const float* __restrict__ frz,
    const float* __restrict__ targets, float* __restrict__ r)
{
#if defined(GRIM_TRACED_KINEMATICS) && !defined(GRIM_TRACED_FRAMES_ONLY)
    if constexpr (n_ee == 1) {
        float q[traced::n_q], pose[7];
        gather_q(cfg, frz, q);
        traced::ee_pose(q, pose);
        pose_residual(pose, targets, r);
        return;
    }
#endif
    float T[n_frames * 7];
    frame_poses(cfg, frz, T);
    for (int e = 0; e < n_ee; ++e) pose_residual(T + ee_joint(e) * 7, targets + e * 7, r + e * 6);
}

/**
 * Stacked residuals and geometric Jacobian. Column a of J is d(pose)/d(cfg[a]): linear
 * rows are the EE-point velocity, angular rows the frame's angular velocity, both in the
 * world frame, with mimic joints folded into the joint they follow.
 */
static __device__ __forceinline__ void residual_and_jacobian(
    const float* __restrict__ cfg, const float* __restrict__ frz,
    const float* __restrict__ targets, float* __restrict__ r, float* __restrict__ J)
{
#if defined(GRIM_TRACED_KINEMATICS) && !defined(GRIM_TRACED_FRAMES_ONLY)
    if constexpr (n_ee == 1) {
        float q[traced::n_q], pose[7], Jt[6 * traced::n_q];
        gather_q(cfg, frz, q);
        traced::ee_pose_jacobian(q, pose, Jt);
        pose_residual(pose, targets, r);
        scatter_jacobian(Jt, J);
        return;
    }
#endif
    float T[n_frames * 7];
    frame_poses(cfg, frz, T);
    for (int i = 0; i < 6 * n_ee * n_solved; ++i) J[i] = 0.0f;
    // Solved-variable column of actuated joint a, or -1 when it is frozen.
    auto col = [](int a) {
        for (int i = 0; i < n_solved; ++i) if (solved_idx(i) == a) return i;
        return -1;
    };
    for (int e = 0; e < n_ee; ++e) {
        const float* T_ee = T + ee_joint(e) * 7;
        pose_residual(T_ee, targets + e * 7, r + e * 6);
        float* Je = J + e * 6 * n_solved;
        for (int j = 0; j < n_frames; ++j) {
            if (!kAncestor[e * n_frames + j]) continue;
            const int a = kActIdx[j] >= 0 ? kActIdx[j] : kMimicActIdx[j];
            const int c = a >= 0 ? col(a) : -1;
            if (c < 0) continue;
            const float* tw = kTwists + j * 6;
            const float* Tj = T + j * 7;
            const float ang2 = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
            const float lin2 = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];
            float lin[3], ang[3] = {0.0f, 0.0f, 0.0f};
            if (ang2 > 1e-12f) {
                const float s = rsqrtf(ang2);
                const float ax[3] = {tw[3]*s, tw[4]*s, tw[5]*s};
                quat_rotate(Tj, ax, ang);
                const float arm[3] = {T_ee[4]-Tj[4], T_ee[5]-Tj[5], T_ee[6]-Tj[6]};
                cross3(ang, arm, lin);
            } else if (lin2 > 1e-12f) {
                const float s = rsqrtf(lin2);
                const float ax[3] = {tw[0]*s, tw[1]*s, tw[2]*s};
                quat_rotate(Tj, ax, lin);
            } else {
                continue;
            }
            const float m = kMimicMul[j];
            Je[0*n_solved + c] += m * lin[0];
            Je[1*n_solved + c] += m * lin[1];
            Je[2*n_solved + c] += m * lin[2];
            Je[3*n_solved + c] += m * ang[0];
            Je[4*n_solved + c] += m * ang[1];
            Je[5*n_solved + c] += m * ang[2];
        }
    }
}

}  // namespace grim::robot

// ---------------------------------------------------------------------------
// Kernel-side shorthands
// ---------------------------------------------------------------------------
// Kernels open with GRIM_IK_DIMS_PROLOGUE(), which names the robot's sizes and its baked
// collision tables as locals, and use the macros below against their own `cfg`, `frz`,
// `s_target_Ts`, `r`, `J`, `seeds`, `out` and `gs`. GRIM_RES / GRIM_RES_JAC do NOT leave
// FK in T_world; call GRIM_FK_ALL before reading it.
#define GRIM_IK_DIMS_PROLOGUE()                                                           \
    constexpr int n_joints = grim::robot::n_frames;                                       \
    constexpr int n_act    = grim::robot::n_solved;                                       \
    constexpr int n_ee     = grim::robot::n_ee;                                           \
    constexpr int n_full   = grim::robot::n_q;                                            \
    constexpr int n_robot_spheres    = grim::robot::n_robot_spheres;                      \
    constexpr int n_self_pairs       = grim::robot::n_self_pairs;                         \
    constexpr int n_world_spheres    = grim::robot::n_world_spheres;                      \
    constexpr int n_world_capsules   = grim::robot::n_world_capsules;                     \
    constexpr int n_world_boxes      = grim::robot::n_world_boxes;                        \
    constexpr int n_world_halfspaces = grim::robot::n_world_halfspaces;                   \
    const float* robot_spheres_local    = grim::robot::kRobotSpheres;                     \
    const int*   robot_sphere_joint_idx = grim::robot::kRobotSphereJoint;                 \
    const float* self_sph_local  = grim::robot::kSelfSph;                                 \
    const int*   self_link_start = grim::robot::kSelfLinkStart;                           \
    const int*   self_link_joint = grim::robot::kSelfLinkJoint;                           \
    const int*   self_pair_i     = grim::robot::kSelfPairI;                               \
    const int*   self_pair_j     = grim::robot::kSelfPairJ;                               \
    (void)n_joints; (void)n_act; (void)n_ee; (void)n_full; (void)n_robot_spheres;         \
    (void)n_self_pairs; (void)n_world_spheres; (void)n_world_capsules;                    \
    (void)n_world_boxes; (void)n_world_halfspaces; (void)robot_spheres_local;             \
    (void)robot_sphere_joint_idx; (void)self_sph_local; (void)self_link_start;            \
    (void)self_link_joint; (void)self_pair_i; (void)self_pair_j
#define GRIM_ACT_SRC(i)      grim::robot::solved_idx(i)
#define GRIM_FK_ALL(q, T)    grim::robot::frame_poses((q), frz, (T))
#define GRIM_RES_JAC(q)      grim::robot::residual_and_jacobian((q), frz, s_target_Ts, r, J)
#define GRIM_RES(q, rr)      grim::robot::residual((q), frz, s_target_Ts, (rr))
#define GRIM_LOAD_SEED(cfg)                                                               \
    float frz[grim::robot::frz_len];                                                      \
    grim::robot::load_state(seeds + gs * n_full, (cfg), frz)
#define GRIM_STORE(cfg)      grim::robot::store_state((cfg), frz, out + gs * n_full)

namespace grim::robot {
/** Device address of a baked __constant__ table, for kernels that take table pointers. */
template <typename T, size_t N>
static inline const T* device_ptr(const T (&symbol)[N])
{
    void* p = nullptr;
    cudaGetSymbolAddress(&p, symbol);
    return static_cast<const T*>(p);
}
}  // namespace grim::robot

// Runtime parent rotation of one joint (a floating base's chart reference). A build made with
// -DGRIM_RUNTIME_ROT_JOINT=j takes one extra operand, right after the stream: the wxyz rotation
// of joint j's parent transform for this launch. The handler writes it into the baked table on
// the launch stream before launching, so the copy is ordered before this kernel and after any
// earlier launch. Other builds compile exactly as before.
#ifdef GRIM_RUNTIME_ROT_JOINT
#define GRIM_ROT_PARAM ffi::Buffer<ffi::DataType::F32> runtime_rot,
#define GRIM_ROT_BIND .Arg<ffi::Buffer<ffi::DataType::F32>>()   // runtime parent rotation
#define GRIM_ROT_UPLOAD(stream)                                                            \
    cudaMemcpyToSymbolAsync(grim::robot::kParentTf, runtime_rot.typed_data(),              \
                            4 * sizeof(float), GRIM_RUNTIME_ROT_JOINT * 7 * sizeof(float),  \
                            cudaMemcpyDeviceToDevice, (stream))
#else
#define GRIM_ROT_PARAM
#define GRIM_ROT_BIND
#define GRIM_ROT_UPLOAD(stream) ((void)0)
#endif
