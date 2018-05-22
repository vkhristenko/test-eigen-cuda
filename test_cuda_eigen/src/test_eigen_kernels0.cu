#include "test_cuda_eigen/interface/test_eigen_kernels0.h"

//
// vector addition for eigen
//
__global__ void cu_eigen_vadd(Eigen::Vector3d* va,
                         Eigen::Vector3d* vb,
                         Eigen::Vector3d* vc) {
    int id = blockIdx.x;
    vc[id] = va[id] + vb[id];
}

void eigen_vector_add(Eigen::Vector3d* va, 
                      Eigen::Vector3d* vb, 
                      Eigen::Vector3d* vc, int const n) {
    cu_eigen_vadd<<<n, 1>>>(va, vb, vc);
}

//
// vector dot product with eigen
//
__global__ void cu_eigen_vdot(Eigen::Vector3d* va,
                         Eigen::Vector3d* vb,
                         Eigen::Vector3d::value_type* vc) {
    int id = blockIdx.x;
    vc[id] = va[id].dot(vb[id]);
}

void eigen_vector_dot(Eigen::Vector3d* va, 
                      Eigen::Vector3d* vb, 
                      Eigen::Vector3d::value_type* vc, int const n) {
    cu_eigen_vdot<<<n, 1>>>(va, vb, vc);
}

//
// matrix addition with eigen
//
__global__ void cu_eigen_madd(Matrix10x10* ma, 
                              Matrix10x10* mb, 
                              Matrix10x10* mc) {
    int id = blockIdx.x;
    mc[id] = ma[id] + mb[id];
}

void eigen_matrix_add(Matrix10x10* ma,
                      Matrix10x10* mb,
                      Matrix10x10* mc, int const n) {
    printf("starting a kernel\n");
    cu_eigen_madd<<<n, 1>>>(ma, mb, mc);
    printf("finished a kernel\n");
}

//
// various matrix tests
//
__global__ void cu_eigen_mtests(Matrix10x10 *min,
                                Matrix10x10 *mout) {
    int id = blockIdx.x;
    // test transposition
    mout[id] = min[id].transpose();
    mout[id].row(9) = Eigen::Matrix<Matrix10x10::value_type, 1, 10>::Zero();

    mout[id].llt();
//    Eigen::LLT<Matrix10x10> llt = mout[id].llt();
}

void eigen_matrix_tests(Matrix10x10 *min,
                        Matrix10x10 *mout,
                        int const n) {
    cu_eigen_mtests<<<n, 1>>>(min, mout);
}
