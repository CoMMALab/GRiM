// Focused full-state momentum contract; no unrelated kernel instantiations.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include "grim.cuh"
#ifndef TEST_SCALAR
#define TEST_SCALAR double
#endif
#ifndef TEST_TIER
#define TEST_TIER grim::TIER_MINIMAL
#endif
using T = TEST_SCALAR;
constexpr int NX = 2 * grim::NUM_VEL;

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n'; std::exit(2);
    }
}

template<bool MJX>
void run(const std::vector<T>& input, int threads) {
    auto* model = grim::init_robotModel<T>();
    T *in = nullptr, *out = nullptr;
    unsigned char *workspace = nullptr;
    const int size = 1 + NX + NX*NX;
    check(cudaMalloc(&in, input.size()*sizeof(T)));
    check(cudaMalloc(&out, size*sizeof(T)));
    check(cudaMalloc(&workspace, grim::GRIM_WORKSPACE_BYTES_PER_TIMESTEP<T>()));
    check(cudaMemset(workspace, 0, grim::GRIM_WORKSPACE_BYTES_PER_TIMESTEP<T>()));
    check(cudaMemcpy(in, input.data(), input.size()*sizeof(T), cudaMemcpyHostToDevice));
    const auto bytes = grim::DCCRBA_DYNAMIC_SHARED_MEM_BYTES<T, TEST_TIER>();
    check(cudaFuncSetAttribute(grim_plant::momentum_cost_kernel<T, MJX, TEST_TIER>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    grim_plant::momentum_cost_kernel<T, MJX, TEST_TIER><<<1, threads, bytes>>>(
        out, out+1, out+1+NX, workspace,
        in, in+grim::NUM_POS, in+grim::NUM_POS+grim::NUM_VEL,
        in+grim::NUM_POS+grim::NUM_VEL+6, model, 1);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    std::vector<T> result(size);
    check(cudaMemcpy(result.data(), out, size*sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << std::setprecision(17);
    for (auto value : result) std::cout << value << ' ';
    std::cout << '\n';
    check(cudaFree(in)); check(cudaFree(out)); check(cudaFree(workspace));
    grim::close_grim<T, grim::GRIM_DATA_ALL>(nullptr, model, nullptr);
}

int main(int argc, char** argv) {
    std::vector<T> input(grim::NUM_POS + grim::NUM_VEL + 12);
    for (auto& v : input) { double value; if (!(std::cin >> value)) return 3; v = T(value); }
    int threads = argc > 2 ? std::atoi(argv[2]) : 32;
    if (argc > 1 && std::atoi(argv[1])) run<true>(input, threads);
    else run<false>(input, threads);
}
