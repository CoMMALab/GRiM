/***
Built + run by examples/codegen/print_grim.py (which auto-detects the GPU arch):
  nvcc -std=c++11 -o printGRiM.exe printGRiM.cu -gencode arch=compute_<ARCH>,code=sm_<ARCH>
(<ARCH> e.g. 120 for sm_120). Dumps every generated kernel's output for a given grim.cuh.
***/

#include <random>
#include <algorithm>
#include "grim.cuh"
#define RANDOM_MEAN 0
#define RANDOM_STDEV 1
std::default_random_engine randEng(1337); // fixed seed
std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv
template <typename T>
T getRand(){return static_cast<T>(randDist(randEng));}

template <typename T>
__host__
void test(){
    T gravity = static_cast<T>(-9.81);  // signed gravitational accel (matches RBDReference/codegen convention)
    dim3 dimms(grim::MAX_PERF_LEVEL_THREADS,1,1);
    cudaStream_t *streams = grim::init_grim<T>();
    grim::robotModel<T> *d_robotModel = grim::init_robotModel<T>();
    grim::grimData<T> *hd_data = grim::init_grimData<T,1>();
    
    // load q,qd,u
    for(int j = 0; j < grim::NUM_JOINTS; j++){
        hd_data->h_q_qd_u[j] = getRand<double>(); 
        hd_data->h_q_qd_u[j+grim::NUM_JOINTS] = getRand<double>(); 
        hd_data->h_q_qd_u[j+2*grim::NUM_JOINTS] = getRand<double>();
    }
    gpuErrchk(cudaMemcpy(hd_data->d_q_qd_u,hd_data->h_q_qd_u,3*grim::NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    printf("q,qd,u\n");
    printMat<T,1,grim::NUM_JOINTS>(hd_data->h_q_qd_u,1);
    printMat<T,1,grim::NUM_JOINTS>(&hd_data->h_q_qd_u[grim::NUM_JOINTS],1);
    printMat<T,1,grim::NUM_JOINTS>(&hd_data->h_q_qd_u[2*grim::NUM_JOINTS],1);

    printf("c via inverse dynamics\n");
    grim::inverse_dynamics<T,false,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,grim::NUM_JOINTS>(hd_data->h_c,1);

    printf("Minv via direct minv\n");
    grim::minv<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    printMat<T,grim::NUM_JOINTS,grim::NUM_JOINTS>(hd_data->h_Minv,grim::NUM_JOINTS);

    printf("qdd via forward dynamics\n");
    grim::forward_dynamics<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,grim::NUM_JOINTS>(hd_data->h_qdd,1);

    printf("qdd via aba\n");
    grim::aba<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,grim::NUM_JOINTS>(hd_data->h_qdd,1);

    printf("M via crba\n");
    grim::crba<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,grim::NUM_JOINTS,grim::NUM_JOINTS>(hd_data->h_M,grim::NUM_JOINTS);

    grim::inverse_dynamics_gradient<T,true,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printf("dc_dq\n");
    printMat<T,grim::NUM_JOINTS,grim::NUM_JOINTS>(hd_data->h_dc_du,grim::NUM_JOINTS);
    printf("dc_dqd\n");
    printMat<T,grim::NUM_JOINTS,grim::NUM_JOINTS>(&hd_data->h_dc_du[grim::NUM_JOINTS*grim::NUM_JOINTS],grim::NUM_JOINTS);

    grim::forward_dynamics_gradient<T,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printf("df_dq\n");
    printMat<T,grim::NUM_JOINTS,grim::NUM_JOINTS>(hd_data->h_df_du,grim::NUM_JOINTS);
    printf("df_dqd\n");
    printMat<T,grim::NUM_JOINTS,grim::NUM_JOINTS>(&hd_data->h_df_du[grim::NUM_JOINTS*grim::NUM_JOINTS],grim::NUM_JOINTS);

    printf("end_effector_pose\n");
    grim::end_effector_pose<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,6*grim::NUM_EES>(hd_data->h_end_effector_pose,1);

    // printf("end_effector_pose - for panda_grasptarget_hand\n");
    // grim::end_effector_pose_panda_grasptarget_hand<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    // printMat<T,1,6*grim::NUM_EES>(hd_data->h_end_effector_pose,1);

    printf("end_effector_pose_gradient (d/dv tangent, pinocchio convention; 6 x NUM_VEL per ee)\n");
    grim::end_effector_pose_gradient<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    for(int ee=0; ee < grim::NUM_EES; ee++){
        printf("end_effector_pose_gradient[%d]\n",ee);
        printMat<T,6,grim::NUM_VEL>(&hd_data->h_end_effector_pose_gradient[ee*6*grim::NUM_VEL],6);
    }

    printf("end_effector_pose_hessian\n");
    grim::end_effector_pose_hessian<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    for(int ee=0; ee < grim::NUM_EES; ee++){
        printf("end_effector_pose_gradient[%d]\n",ee);
        printMat<T,6,grim::NUM_VEL>(&hd_data->h_end_effector_pose_gradient[ee*6*grim::NUM_VEL],6);
        printf("end_effector_pose_hessian[%d]\n",ee);
        for (int i=0; i < 6; i++){
            int offset = ee*6*grim::NUM_VEL*grim::NUM_VEL + i*grim::NUM_VEL*grim::NUM_VEL;
            printf("[%d]\n",i); printMat<T,grim::NUM_VEL,grim::NUM_VEL>(&hd_data->h_end_effector_pose_hessian[offset],grim::NUM_VEL);
        }
    }
    grim::close_grim<T>(streams,d_robotModel,hd_data);
}

int main(void){
    test<float>(); return 0;
}