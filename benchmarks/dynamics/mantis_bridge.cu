// mantis dynamics behind a C ABI, timed the way GRiD's release kernel bridge times GRiD
// (external/grid/test/benchmarks/release/kernel_bridge.cu, grid_kernel_time): host-clock samples
// of one launch + cudaDeviceSynchronize ("compute", inputs and outputs resident on the device) and
// of pinned H2D + launch + D2H + sync ("with memory"), after a time-based warm-up.
//
// The kernel is GRiM's, unchanged: kernels/dynamics/ck_dynamics.cu is included below, built against
// the same generated header and flags grim.motion.dynamics.MantisDynamics compiles it with
// (dynamics/grid_protocol.py captures them). launch() is Run<OP>'s launch configuration from that
// file without the XLA FFI wrapper.
#include "dynamics/ck_dynamics.cu"

#include <chrono>
#include <cstring>

namespace {

double now_us()
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <Op OP>
int launch(int tier, const float* in, float* out, int batch)
{
    constexpr size_t slot = sizeof(real) * (nin(OP) + nwork(OP) + grim::dyn::SMEM);
    constexpr size_t tab_w = sizeof(real) * tab_total<grim::Tier::Warp, OP>(), tab_b = sizeof(real) * tab_total<grim::Tier::Block, OP>();
    if (tier != 0 && slot + (tier == 1 ? tab_w : tab_b) > SMEM_MAX) return -10;   // GRiM refuses this tier
    if (tier == 0) {
        dyn_kernel<grim::Tier::Thread, OP><<<(batch + 127) / 128, 128>>>(in, out, batch, 1);
    } else if (tier == 1) {
        const int fit = (int)((SMEM_MAX - tab_w) / slot);
        const int wpb = fit < WARPS ? fit : WARPS;
        cudaFuncSetAttribute(dyn_kernel<grim::Tier::Warp, OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(tab_w + wpb * slot));
        dyn_kernel<grim::Tier::Warp, OP><<<(batch + wpb - 1) / wpb, 32 * wpb, tab_w + wpb * slot>>>(in, out, batch, wpb);
    } else {
        cudaFuncSetAttribute(dyn_kernel<grim::Tier::Block, OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(tab_b + slot));
        dyn_kernel<grim::Tier::Block, OP><<<batch, grim::dyn::BLOCK_THREADS, tab_b + slot>>>(in, out, batch, 1);
    }
    return 0;
}

int launch_op(int op, int tier, const float* in, float* out, int batch)
{
    switch (op) {
#define CK_CASE(NAME, OP) case OP: return launch<OP>(tier, in, out, batch);
        GRIM_CK_DYN_OPS(CK_CASE)
#undef CK_CASE
    }
    return -2;
}

int sizes(int op, int* ni, int* no)
{
    switch (op) {
#define CK_CASE(NAME, OP) case OP: *ni = nin(OP); *no = nout(OP); return 0;
        GRIM_CK_DYN_OPS(CK_CASE)
#undef CK_CASE
    }
    return -2;
}

struct Ctx {
    int op, tier, batch, ni, no;
    float *d_in, *d_out, *h_in, *h_out;   // h_*: pinned, as GRiD's gridData host buffers
};

int check(cudaError_t e) { return e == cudaSuccess ? 0 : 100 + (int)e; }

int compute(Ctx& c)
{
    if (int rc = launch_op(c.op, c.tier, c.d_in, c.d_out, c.batch)) return rc;
    return check(cudaDeviceSynchronize());
}

int with_mem(Ctx& c)
{
    cudaMemcpyAsync(c.d_in, c.h_in, sizeof(float) * c.batch * c.ni, cudaMemcpyHostToDevice);
    if (int rc = launch_op(c.op, c.tier, c.d_in, c.d_out, c.batch)) return rc;
    cudaMemcpyAsync(c.h_out, c.d_out, sizeof(float) * c.batch * c.no, cudaMemcpyDeviceToHost);
    return check(cudaDeviceSynchronize());
}

}  // namespace

extern "C" void ck_close(void* h)
{
    if (!h) return;
    Ctx* c = static_cast<Ctx*>(h);
    cudaFree(c->d_in); cudaFree(c->d_out); cudaFreeHost(c->h_in); cudaFreeHost(c->h_out);
    delete c;
}

// op: ck_dynamics.cu's Op enum; tier 0 thread, 1 warp, 2 block. in: (batch, nin) row-major, staged
// to the device here. Returns null and sets *rc on failure (-10: GRiM refuses the tier).
extern "C" void* ck_create(int op, int tier, const float* in, int batch, int* rc)
{
    Ctx* c = new Ctx{op, tier, batch, 0, 0, nullptr, nullptr, nullptr, nullptr};
    if ((*rc = sizes(op, &c->ni, &c->no))) { delete c; return nullptr; }
    const size_t bi = sizeof(float) * batch * c->ni, bo = sizeof(float) * batch * c->no;
    if ((*rc = check(cudaMalloc(&c->d_in, bi))) || (*rc = check(cudaMalloc(&c->d_out, bo)))
        || (*rc = check(cudaMallocHost(&c->h_in, bi))) || (*rc = check(cudaMallocHost(&c->h_out, bo)))) {
        ck_close(c);
        return nullptr;
    }
    std::memcpy(c->h_in, in, bi);
    if ((*rc = check(cudaMemcpy(c->d_in, c->h_in, bi, cudaMemcpyHostToDevice))) || (*rc = compute(*c))) {
        ck_close(c);
        return nullptr;
    }
    return c;
}

// One call each way; out: (nout, batch), the kernel's layout.
extern "C" int ck_host(void* h, float* out)
{
    Ctx& c = *static_cast<Ctx*>(h);
    if (int rc = with_mem(c)) return rc;
    std::memcpy(out, c.h_out, sizeof(float) * c.batch * c.no);
    return 0;
}

extern "C" int ck_resident(void* h, float* out)
{
    Ctx& c = *static_cast<Ctx*>(h);
    if (int rc = compute(c)) return rc;
    return check(cudaMemcpy(out, c.d_out, sizeof(float) * c.batch * c.no, cudaMemcpyDeviceToHost));
}

// grid_kernel_time's loops: warm-up (at least `warmups`, until `warm_seconds`), then `iterations`
// host-clock samples of with_mem, then of compute.
extern "C" int ck_time(void* h, double warm_seconds, int warmups, int iterations, double* with_mem_us, double* compute_us)
{
    Ctx& c = *static_cast<Ctx*>(h);
    const double warm_start = now_us();
    int done = 0;
    do {
        if (int rc = with_mem(c)) return rc;
        ++done;
    } while (done < warmups || now_us() - warm_start < warm_seconds * 1e6);
    for (int i = 0; i < iterations; ++i) {
        const double t0 = now_us();
        with_mem(c);
        with_mem_us[i] = now_us() - t0;
    }
    for (int i = 0; i < warmups; ++i) compute(c);
    for (int i = 0; i < iterations; ++i) {
        const double t0 = now_us();
        compute(c);
        compute_us[i] = now_us() - t0;
    }
    return check(cudaGetLastError());
}
