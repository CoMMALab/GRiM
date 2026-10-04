/*
 * Canonicalisation: slide an IK solution along the self-motion manifold to the
 * point nearest a reference configuration.
 *
 *     q* = argmin 1/2 ||q - q_ref||^2   subject to   r(q, t) = 0
 *
 * This is what makes IK derivatives correct on a redundant arm. r = 0 alone
 * does not determine q* (a 7-DOF arm with 6 task constraints leaves a 1-D
 * solution curve), so differentiating it answers a question the solver did not
 * ask -- pinv silently supplies the missing information by picking the
 * minimum-norm tangent. Pinning q* to a definite point of that curve makes the
 * KKT sensitivity exact (pyroffi's optimization_engines/_canonical_ik.py drives it).
 *
 * Why this is a kernel and not JAX. The JAX loop cost ~4 ms PER ITERATION of
 * XLA dispatch overhead -- 1615 ms at B=64, 13.7 s at B=1024, i.e. 61-149x the
 * IK solve it was correcting. The arithmetic is trivial: each iteration is one
 * FK + task Jacobian + a small SPD solve, which is exactly what the IK solve
 * kernels already do many times over in 26 ms. The overhead was the whole cost.
 *
 * The iteration is Gauss-Newton on the constrained problem:
 *
 *     dq = -J^+ r  +  step * (I - J^+ J) (q_ref - q)
 *
 * The first term restores r = 0; the second walks toward q_ref along the null
 * space WITHOUT moving the end-effector. `step` must be damped: at step = 1
 * (the exact Gauss-Newton step) the iteration diverges on manifold curvature --
 * measured ||q - q_ref|| growing 6.0 -> 34.9 -- because a full step leaves the
 * linearisation's region of validity. Larger trust-region variants were tried
 * and also failed to converge; the small damped step is required, which is
 * precisely why the iteration count is high and why it belongs in CUDA.
 *
 * PRECISION. This runs in float32, because the shared FK/Jacobian helper is
 * float32. That converges to |r| ~ 1e-5, not the ~1e-16 the derivative rule
 * wants, so the Python side finishes with a few float64 Newton steps. Bulk here,
 * polish there.
 *
 * J is the GEOMETRIC Jacobian (the helper's orientation rows are the angular
 * Jacobian, not d(log R)/dq). That is harmless here: the two coincide wherever
 * r = 0, so the FIXED POINT is unchanged, and only the path taken to reach it
 * differs slightly.
 */

#include "robot.cuh"
#include "collision.cuh"

#include <xla/ffi/api/ffi.h>
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

#define CANON_MAX_RES (6 * MAX_EE)

// Cholesky of a small SPD matrix, in place, lower triangular. Serial and
// per-thread on purpose: m is 6 for a single end-effector, so a cooperative
// GLASS factorisation would cost more in synchronisation than it saves. (GLASS
// is also vendored and must not be modified.)
__device__ __forceinline__ void canon_chol(float* A, int m)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[i * m + j];
            for (int k = 0; k < j; k++) s -= A[i * m + k] * A[j * m + k];
            if (i == j) {
                A[i * m + j] = sqrtf(fmaxf(s, 1e-12f));
            } else {
                A[i * m + j] = s / A[j * m + j];
            }
        }
    }
}

// Solve (L L^T) x = b in place on x, given the factor from canon_chol.
__device__ __forceinline__ void canon_chol_solve(const float* L, float* x, int m)
{
    for (int i = 0; i < m; i++) {
        float s = x[i];
        for (int k = 0; k < i; k++) s -= L[i * m + k] * x[k];
        x[i] = s / L[i * m + i];
    }
    for (int i = m - 1; i >= 0; i--) {
        float s = x[i];
        for (int k = i + 1; k < m; k++) s -= L[k * m + i] * x[k];
        x[i] = s / L[i * m + i];
    }
}

__global__ void canonical_ik_kernel(
    const float* __restrict__ cfgs,        // (n_problems, n_act)
    const float* __restrict__ cfg_refs,    // (n_problems, n_act)
    const float* __restrict__ target_Ts,   // (n_problems, n_ee, 7)
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    int n_problems,
    int max_iters, float step, float tol, float damping, float collision_margin,
    float* __restrict__ out_q,             // (n_problems, n_act)
    int*   __restrict__ out_iters)         // (n_problems,)
{
    GRIM_IK_DIMS_PROLOGUE();
    float frz[grim::robot::frz_len];   // every joint is solved: nothing frozen
    (void)frz;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_problems) return;

    const int m = 6 * n_ee;

    float q[MAX_ACT], qref[MAX_ACT], w[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[CANON_MAX_RES], J[CANON_MAX_RES * MAX_ACT];
    float G[CANON_MAX_RES * CANON_MAX_RES], y[CANON_MAX_RES];

    for (int a = 0; a < n_act; a++) {
        q[a]    = cfgs[(size_t)p * n_act + a];
        qref[a] = cfg_refs[(size_t)p * n_act + a];
    }

    // Minimum signed clearance over every active constraint, margin included.
    // Feasible is >= 0. This is the SAME geometry the IK solvers enforce, so a
    // configuration this call accepts is one they would also call collision
    // free -- the two cannot drift apart into disagreeing about feasibility.
    auto min_clearance = [&](const float* Tw) -> float {
        float best = 1e9f;
        if (n_self_pairs > 0) {
            best = fminf(best, self_collision_min_dist(
                Tw, self_sph_local, self_link_start, self_link_joint,
                self_pair_i, self_pair_j, n_self_pairs) - collision_margin);
        }
        for (int i = 0; i < n_robot_spheres; i++) {
            const int jidx = robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= n_joints) continue;
            const float* sp = robot_spheres_local + i * 4;
            const float local_p[3] = {sp[0], sp[1], sp[2]};
            float c[3];
            apply_se3_point(Tw + (size_t)jidx * 7, local_p, c);
            const float rr = sp[3];
            for (int k = 0; k < n_world_spheres; k++)
                best = fminf(best, world_prim_dist(c, rr, kWorldSphere, world_spheres + k * 4) - collision_margin);
            for (int k = 0; k < n_world_capsules; k++)
                best = fminf(best, world_prim_dist(c, rr, kWorldCapsule, world_capsules + k * 7) - collision_margin);
            for (int k = 0; k < n_world_boxes; k++)
                best = fminf(best, world_prim_dist(c, rr, kWorldBox, world_boxes + k * 15) - collision_margin);
            for (int k = 0; k < n_world_halfspaces; k++)
                best = fminf(best, world_prim_dist(c, rr, kWorldHalfspace, world_halfspaces + k * 6) - collision_margin);
        }
        return best;
    };

    const bool guard = (n_self_pairs > 0) || (n_robot_spheres > 0 &&
        (n_world_spheres + n_world_capsules + n_world_boxes + n_world_halfspaces) > 0);

    float q_try[MAX_ACT], T_try[MAX_JOINTS * 7];

    // Last iterate known to be feasible, so canonicalisation can never hand
    // back something WORSE than what it was given. The null-space step is
    // backtracked for feasibility, but the pose term is not -- it must be free
    // to restore r = 0 -- so a problem that starts infeasible, or one whose
    // pose correction crosses an obstacle, could otherwise drift out of the
    // feasible set the solver had reached.
    // Feasibility is judged RELATIVE TO THE INPUT, not absolutely. A solve that
    // ends with clearance inside the margin band -- common for a spherized model
    // whose links nearly touch -- is "infeasible" by an absolute test, so an
    // absolute guard never records a fallback and silently walks unprotected.
    // That is what left ls 4/32 self-colliding. The rule that always holds:
    // never hand back LESS clearance than we were given.
    float q_feas[MAX_ACT];
    bool have_feas = false;
    float c_floor = 0.0f;
    if (guard) {
        grim::robot::frame_poses(q, frz, T_try);
        const float c0 = min_clearance(T_try);
        // NEVER DECREASE clearance -- not merely "stay feasible". An absolute
        // test depends on this kernel and the caller's collision model agreeing
        // exactly, and they do not: with a 0.02 m standoff requested, an
        // absolute guard walked to 0.0155 m by its own measure while the
        // checker scored it short. Holding clearance monotone makes the guard
        // independent of that disagreement -- canonicalisation can improve
        // clearance or hold it, never spend it.
        c_floor = c0;
        for (int a = 0; a < n_act; a++) q_feas[a] = q[a];
        have_feas = true;
    }

    int used = max_iters;
    for (int it = 0; it < max_iters; it++) {
        grim::robot::residual_and_jacobian(q, frz, target_Ts + (size_t)p * n_ee * 7, r, J);

        // G = J J^T + damping I, then factor once and reuse for both solves.
        // Forming the m x m normal matrix (m = 6) rather than an SVD is the
        // point: pinv-by-SVD was measured to be the wrong tool here.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                float s = 0.0f;
                for (int a = 0; a < n_act; a++) s += J[i * n_act + a] * J[j * n_act + a];
                if (i == j) s += damping;
                G[i * m + j] = s;
                G[j * m + i] = s;
            }
        }
        canon_chol(G, m);

        // Pose term: dp = -J^T (G^-1 r)
        for (int i = 0; i < m; i++) y[i] = r[i];
        canon_chol_solve(G, y, m);

        // Null-space term: w = q_ref - q, then dn = w - J^T (G^-1 (J w)),
        // which is (I - J^+ J) w without ever forming the n x n projector.
        for (int a = 0; a < n_act; a++) w[a] = qref[a] - q[a];
        float Jw[CANON_MAX_RES];
        for (int i = 0; i < m; i++) {
            float s = 0.0f;
            for (int a = 0; a < n_act; a++) s += J[i * n_act + a] * w[a];
            Jw[i] = s;
        }
        canon_chol_solve(G, Jw, m);

        float dp_v[MAX_ACT], dn_v[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float dp = 0.0f, dn = 0.0f;
            for (int i = 0; i < m; i++) {
                dp -= J[i * n_act + a] * y[i];
                dn -= J[i * n_act + a] * Jw[i];
            }
            dn_v[a] = dn + w[a];
            dp_v[a] = dp;
        }

        // COLLISION-AWARE STEP. The walk moves only in the task null space, so
        // the end-effector pose is preserved exactly -- but the null space of
        // the POSE constraint is not the null space of the COLLISION
        // constraints. Self-motion swings the elbow and wrist, which is exactly
        // what clearance depends on, so an unguarded walk slides into geometry
        // the solver had avoided (measured: world collisions 0.0% -> 6.2% on
        // the cuRobo parity set).
        //
        // The null-space component is therefore backtracked until it keeps the
        // configuration feasible. The pose term is never scaled -- giving up
        // pose to gain clearance is the solver's job, not this one's -- and if
        // no fraction is admissible the null-space move is simply dropped for
        // this iteration, leaving q where it already was: feasible.
        float scale = 1.0f;
        if (guard) {
            for (int t = 0; t < 6; t++) {
                for (int a = 0; a < n_act; a++)
                    q_try[a] = q[a] + dp_v[a] + step * scale * dn_v[a];
                grim::robot::frame_poses(q_try, frz, T_try);
                if (min_clearance(T_try) >= c_floor) break;
                scale *= 0.5f;
                if (t == 5) scale = 0.0f;
            }
        }

        float step_sq = 0.0f;
        for (int a = 0; a < n_act; a++) {
            const float d = dp_v[a] + step * scale * dn_v[a];
            q[a] += d;
            step_sq += d * d;
        }

        if (guard) {
            grim::robot::frame_poses(q, frz, T_try);
            if (min_clearance(T_try) >= c_floor) {
                for (int a = 0; a < n_act; a++) q_feas[a] = q[a];
                have_feas = true;
            }
        }

        // Early exit. The JAX version ran a FIXED count and so both wasted work
        // on easy problems and silently failed to converge on hard ones at large
        // batch (|r| degraded to 1.8e-2 at B=1024).
        if (step_sq < tol * tol) { used = it + 1; break; }
    }

    // Return the walked point only if it is feasible; otherwise the last
    // feasible one seen. Never a configuration this kernel knows is in
    // collision -- that would trade the suite's hard guarantee for a gradient.
    if (guard && have_feas) {
        grim::robot::frame_poses(q, frz, T_try);
        if (min_clearance(T_try) < c_floor)
            for (int a = 0; a < n_act; a++) q[a] = q_feas[a];
    }

    for (int a = 0; a < n_act; a++) out_q[(size_t)p * n_act + a] = q[a];
    out_iters[p] = used;
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error CanonicalIkImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfgs,
    ffi::Buffer<ffi::DataType::F32> cfg_refs,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    int64_t max_iters,
    float   step,
    float   tol,
    float   damping,
    float   collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_q,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_iters)
{
    const int n_problems = static_cast<int>(cfgs.dimensions()[0]);
    if (cfgs.dimensions()[1] != grim::robot::n_q ||
        target_Ts.dimensions()[target_Ts.dimensions().size() - 2] != grim::robot::n_ee ||
        world_spheres.dimensions()[0] != grim::robot::n_world_spheres ||
        world_capsules.dimensions()[0] != grim::robot::n_world_capsules ||
        world_boxes.dimensions()[0] != grim::robot::n_world_boxes ||
        world_halfspaces.dimensions()[0] != grim::robot::n_world_halfspaces)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "canonical_ik: launch does not match the robot and obstacle counts of this build");
    if (n_problems == 0) return ffi::Error::Success();

    const int block = 64;
    const int grid  = (n_problems + block - 1) / block;

    canonical_ik_kernel<<<grid, block, 0, stream>>>(
        cfgs.typed_data(), cfg_refs.typed_data(),
        target_Ts.typed_data(),
        world_spheres.typed_data(), world_capsules.typed_data(),
        world_boxes.typed_data(), world_halfspaces.typed_data(),
        n_problems, static_cast<int>(max_iters), step, tol, damping, collision_margin,
        out_q->typed_data(), out_iters->typed_data());

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CanonicalIkFfi, CanonicalIkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfgs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfg_refs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Attr<int64_t>("max_iters")
        .Attr<float>("step")
        .Attr<float>("tol")
        .Attr<float>("damping")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_q
        .Ret<ffi::Buffer<ffi::DataType::S32>>()); // out_iters
