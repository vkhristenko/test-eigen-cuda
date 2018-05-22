#include "test_cuda_eigen/interface/test_kernels0.h"
#include <Eigen/Dense>

//
// simple vector addition example
//
__global__ void cu_vector_add(int* a, int* b, int* c) {
    int id = blockIdx.x;
    c[id] = a[id] + b[id];
}

void vector_add(int *a, int* b, int* c, int const n) {
    cu_vector_add<<<n, 1>>>(a, b, c);
}
