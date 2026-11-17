// RBDL behind GRiD's release CPU-baseline protocol: the same persistent pool (GRiD's
// test/benchmarks/release/release_pool.h, included unchanged), one independent model per thread,
// contiguous slices with slice 0 on the caller, the same C ABI shape as GRiD's pin_bridge.cpp.
// RBDL is double only: float32 inputs are widened per sample inside the call.
// Ops (GRiD's operation indices): 0 inverse_dynamics (RNEA), 4 forward_dynamics (ABA),
// 8 crba (CompositeRigidBodyAlgorithm, full symmetric matrix).
#include "release_pool.h"
#include <rbdl/rbdl.h>
#include <rbdl/addons/urdfreader/urdfreader.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace RigidBodyDynamics;
using namespace RigidBodyDynamics::Math;

static std::string rbdl_error;

struct Context {
    Model model;
    VectorNd q, qd, t, out;
    MatrixNd M;
    int op;
    Context(const char *urdf, int op_) : op(op_) {
        if (!Addons::URDFReadFromFile(urdf, &model, false, false))
            throw std::runtime_error(std::string("RBDL could not read ") + urdf);
        model.gravity = Vector3d(0., 0., -9.81);
        const int n = model.dof_count;
        q = qd = t = out = VectorNd::Zero(n);
        M = MatrixNd::Zero(n, n);
    }
};

struct RbdlPool {
    std::vector<std::unique_ptr<Context>> contexts;
    std::unique_ptr<ReleasePool> pool;
};

extern "C" const char *rbdl_release_error() { return rbdl_error.c_str(); }

extern "C" void *rbdl_release_pool_create(const char *urdf, int op, int threads) {
    if (op != 0 && op != 4 && op != 8) { rbdl_error = "unsupported operation"; return nullptr; }
    if (threads < 1) { rbdl_error = "pool needs at least one thread"; return nullptr; }
    try {
        auto *p = new RbdlPool;
        for (int k = 0; k < threads; ++k) p->contexts.emplace_back(new Context(urdf, op));
        p->pool.reset(new ReleasePool((std::size_t)threads - 1));
        return p;
    } catch (const std::exception &e) { rbdl_error = e.what(); return nullptr; }
}
extern "C" void rbdl_release_pool_close(void *ptr) { delete static_cast<RbdlPool *>(ptr); }
extern "C" int rbdl_release_dof(void *ptr) { return static_cast<RbdlPool *>(ptr)->contexts[0]->model.dof_count; }

// Name of the URDF child link that drives q index ``i`` (RBDL keeps link names, not joint names).
extern "C" const char *rbdl_release_q_link(void *ptr, int i) {
    static std::string name;
    Model &m = static_cast<RbdlPool *>(ptr)->contexts[0]->model;
    for (unsigned b = 1; b < m.mBodies.size(); ++b)
        if ((int)m.mJoints[b].q_index == i && m.mJoints[b].mDoFCount == 1) { name = m.GetBodyName(b); return name.c_str(); }
    name.clear();
    return name.c_str();
}

static void eval_range(Context &c, const float *qs, const float *vs, const float *ts, int start, int stop,
                       double *output) {
    const int n = c.model.dof_count;
    for (int s = start; s < stop; ++s) {
        for (int j = 0; j < n; ++j) {
            c.q[j] = qs[(size_t)s * n + j];
            c.qd[j] = vs[(size_t)s * n + j];
            c.t[j] = ts[(size_t)s * n + j];
        }
        if (c.op == 0) {
            InverseDynamics(c.model, c.q, c.qd, c.t, c.out);
            for (int j = 0; j < n; ++j) output[(size_t)s * n + j] = c.out[j];
        } else if (c.op == 4) {
            ForwardDynamics(c.model, c.q, c.qd, c.t, c.out);
            for (int j = 0; j < n; ++j) output[(size_t)s * n + j] = c.out[j];
        } else {
            // H zeroed once at construction, as RBDL's benchmark does: CRBA writes every tree
            // entry and never touches the rest.
            CompositeRigidBodyAlgorithm(c.model, c.q, c.M, true);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) output[(size_t)s * n * n + i * n + j] = c.M(i, j);
        }
    }
}

extern "C" int rbdl_release_pool_eval(void *ptr, const float *qs, const float *vs, const float *ts, int batch,
                                      double *output, int active) {
    if (!ptr || !qs || !vs || !ts || !output || batch < 1 || active < 1) return -1;
    auto &p = *static_cast<RbdlPool *>(ptr);
    const int count = std::min<int>(active, std::min<int>((int)p.contexts.size(), batch));
    std::vector<int> rcs((size_t)count, 0);
    std::vector<std::string> errors((size_t)count);
    p.pool->run((std::size_t)count, [&](std::size_t slot) {
        const int start = (int)((long long)batch * (long long)slot / count);
        const int stop = (int)((long long)batch * (long long)(slot + 1) / count);
        try { eval_range(*p.contexts[slot], qs, vs, ts, start, stop, output); }
        catch (const std::exception &e) { rcs[slot] = -3; errors[slot] = e.what(); }
    });
    for (int k = 0; k < count; ++k)
        if (rcs[(size_t)k]) { rbdl_error = errors[(size_t)k]; return rcs[(size_t)k]; }
    return 0;
}
