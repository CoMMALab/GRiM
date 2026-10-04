// Test runner for the CUDA joint-torque regressor kernel.
//
// Mirrors `cuda_idsva_so_world_frame_smoke_runner.cu` and invokes
// `inverse_dynamics_regressor` (host launcher). R2: the regressor output d_Y is
// now a grimData field (hd_data->d_Y, nv x 10*NUM_BODIES); the host copies it back
// into hd_data->h_Y, which this runner reads directly. Used by `test_cuda_regressor`
// to validate the CUDA emission against `RBDReference.inverse_dynamics_regressor`
// (the verified numpy reference) and the identity Y @ pi == inverse_dynamics(q,qd,qdd).

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grim.cuh"

#ifndef GRIM_CUDA_REGRESSOR_TEST_THREADS
#define GRIM_CUDA_REGRESSOR_TEST_THREADS 64
#endif

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double value;
        if (!(std::cin >> value)) {
            std::cerr << "Failed to read input value " << i << std::endl;
            std::exit(2);
        }
        dst[i] = static_cast<T>(value);
    }
}

template <typename T>
void print_flat(const std::string &name, const T *data, int rows, int cols) {
    std::cout << "BEGIN " << name << " " << rows << " " << cols << "\n";
    std::cout << std::setprecision(10);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c) std::cout << " ";
            std::cout << static_cast<double>(data[r * cols + c]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}

template <typename T>
int run() {
    // Unified gravity convention: GRiM and the RBDReference oracle both use -9.81.
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    const int _req_threads = GRIM_CUDA_REGRESSOR_TEST_THREADS;
    const int _nthreads = _req_threads < grim::MAX_PERF_LEVEL_THREADS ? _req_threads : grim::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(_nthreads, 1, 1);

    const int nv = grim::NUM_VEL;
    const int nb = grim::NUM_BODIES;
    const int Y_rows = nv;
    const int Y_cols = 10 * nb;
    const int Y_size = Y_rows * Y_cols;

    cudaStream_t *streams = grim::init_grim<T>();
    grim::robotModel<T> *d_robot_model = grim::init_robotModel<T>();
    grim::grimData<T> *hd_data = grim::init_grimData<T, 1>();

    // q|qd|qdd into the q_qd_u host buffer. Canonical layout (Q_QD_U_STRIDE == 3*NUM_POS):
    // each field gets a NUM_POS(=nq)-wide slot -> q@0, qd@nq, qdd@2*nq (the kernel reads
    // s_qdd at 2*NUM_POS). Fixed-base nq==nv makes this byte-identical to the old nv-based
    // qdd@nq+nv; FLOATING nq>nv needs the nq-based 2*NUM_POS offset (else qdd is read shifted
    // by nq-nv -> wrong base acceleration -> globally wrong regressor).
    read_vector(hd_data->h_q_qd_u, grim::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grim::NUM_POS], grim::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[2 * grim::NUM_POS], grim::NUM_VEL);

    // R2: regressor output lives in grimData (hd_data->d_Y / hd_data->h_Y); the
    // host launcher copies device->host into hd_data->h_Y.
    grim::inverse_dynamics_regressor<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    const T *h_Y = hd_data->h_Y;

    int first_bad = -1;
    for (int i = 0; i < Y_size; ++i) {
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_Y[i]))) first_bad = i;
    }
    if (first_bad >= 0) {
        std::cerr << "Non-finite regressor output at " << first_bad << std::endl;
    }

    T config[5];
    config[0] = static_cast<T>(grim::NUM_POS);
    config[1] = static_cast<T>(grim::NUM_VEL);
    config[2] = static_cast<T>(grim::NUM_BODIES);
    config[3] = static_cast<T>(grim::NUM_JOINTS);
    config[4] = static_cast<T>(grim::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>());

    print_flat("regressor_config", config, 1, 5);
    print_flat("regressor", h_Y, Y_rows, Y_cols);

    grim::close_grim<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main() {
    return run<float>();
}
