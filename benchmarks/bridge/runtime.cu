// Runtime of the GRiM C bridge (bridge/grim_bridge.py): device memory, a recorded program of steps
// (GRiM FFI implementations through grim_invoke.cuh, or the steps below), and its timing loop.
// Built with -cudart shared, as every bridge library, so all share one CUDA runtime.
//
// The steps below are the JAX code our benchmarks wrap around a GRiM FFI call, ported line for line;
// every bridge run is checked against the JAX row's output (grim_bridge.Program.check).
#include <chrono>
#include <cstdint>
#include <cstdio>

#include "xla/ffi/api/c_api.h"

using StepFn = int (*)(cudaStream_t, XLA_FFI_Buffer*, const double*);
struct Step {
    StepFn fn;
    XLA_FFI_Buffer* bufs;
    const double* attrs;
};

static int check(cudaError_t e)
{
    if (e == cudaSuccess) return 0;
    std::fprintf(stderr, "grim bridge: %s\n", cudaGetErrorString(e));
    return 100 + (int)e;
}

extern "C" void* gb_malloc(size_t bytes)
{
    void* p = nullptr;
    return cudaMalloc(&p, bytes ? bytes : 1) == cudaSuccess ? p : nullptr;
}
extern "C" void gb_free(void* p) { cudaFree(p); }
extern "C" int gb_h2d(void* d, const void* h, size_t n) { return check(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); }
extern "C" int gb_d2h(void* h, const void* d, size_t n) { return check(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost)); }

extern "C" int gb_run(const Step* steps, int n)
{
    for (int i = 0; i < n; ++i)
        if (int rc = steps[i].fn(0, steps[i].bufs, steps[i].attrs)) return rc;
    return check(cudaGetLastError());
}

// common.amortized_ms in C: `warmup` synchronized runs, then per rep `k` back-to-back runs and one
// cudaDeviceSynchronize on the host clock; ms[r] is that rep's time per run.
extern "C" int gb_time(const Step* steps, int n, int k, int reps, int warmup, double* ms)
{
    for (int i = 0; i < warmup; ++i) {
        if (int rc = gb_run(steps, n)) return rc;
        if (int rc = check(cudaDeviceSynchronize())) return rc;
    }
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < k; ++i)
            if (int rc = gb_run(steps, n)) return rc;
        if (int rc = check(cudaDeviceSynchronize())) return rc;
        ms[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / k;
    }
    return 0;
}

// fk/bench_fk.py _place: sphere centres from (B, F, 7) wxyz+xyz frame poses.
// b: poses (B, F, 7), idx (S,) int32 (-1: fixed in the world), centres (S, 3), out (B, S, 3).
__global__ void place_kernel(const float* poses, const int* idx, const float* c, float* out, int B, int F, int S)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * S) return;
    const int b = i / S, s = i % S, j = idx[s];
    const float cx = c[3 * s], cy = c[3 * s + 1], cz = c[3 * s + 2];
    float* o = out + 3 * (size_t)i;
    if (j < 0) { o[0] = cx; o[1] = cy; o[2] = cz; return; }
    const float* p = poses + ((size_t)b * F + j) * 7;
    const float w = p[0], vx = p[1], vy = p[2], vz = p[3];
    const float ux = vy * cz - vz * cy, uy = vz * cx - vx * cz, uz = vx * cy - vy * cx;   // u = v x c
    const float wx = vy * uz - vz * uy, wy = vz * ux - vx * uz, wz = vx * uy - vy * ux;   // v x u
    o[0] = cx + 2 * (w * ux + wx) + p[4];
    o[1] = cy + 2 * (w * uy + wy) + p[5];
    o[2] = cz + 2 * (w * uz + wz) + p[6];
}

extern "C" int gb_place(cudaStream_t s, XLA_FFI_Buffer* b, const double*)
{
    const int B = (int)b[0].dims[0], F = (int)b[0].dims[1], S = (int)b[1].dims[0];
    place_kernel<<<(B * S + 127) / 128, 128, 0, s>>>((const float*)b[0].data, (const int*)b[1].data,
                                                      (const float*)b[2].data, (float*)b[3].data, B, F, S);
    return 0;
}
