#include <iostream>

#ifdef USE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "test_cuda_eigen/interface/test_eigen_kernels0.h"
#endif

int main() {
    std::cout << "hello world" << std::endl;

#ifdef USE_CUDA
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    int constexpr n = 8;
    Eigen::Vector3d a[n], b[n], c[n];
    Eigen::Vector3d *d_a, *d_b, *d_c;
    Eigen::Vector3d::value_type dot_out[n];
    Eigen::Vector3d::value_type* d_dot_out;


    for (auto i=0; i<n; i++) { // populates vectors with values 
        a[i] << i, i+1, i+2;
        b[i] << i, i+1, i+2;
    }

    // alloc on the device
    cudaMalloc((void**)&d_a, n * sizeof(Eigen::Vector3d) );
    cudaMalloc((void**)&d_b, n*sizeof(Eigen::Vector3d));
    cudaMalloc((void**)&d_c, n*sizeof(Eigen::Vector3d));
    cudaMalloc((void**)&d_dot_out, n*sizeof(Eigen::Vector3d::value_type));

    // transfer to the device
    cudaMemcpy(d_a, a, n*sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice);
    
    // run the kernel
    eigen_vector_mult(d_a, d_b, d_c, n);
    eigen_vector_dot(d_a, d_b, d_dot_out, n);

    cudaMemcpy(c, d_c, n*sizeof(Eigen::Vector3d), cudaMemcpyDeviceToHost);
    cudaMemcpy(dot_out, d_dot_out, n*sizeof(Eigen::Vector3d::value_type),
        cudaMemcpyDeviceToHost);

    for (auto i=0; i<n; i++) 
        if (i%2 == 0) {
            std::cout << "c[" << i << "]" << std::endl
                << c[i] << std::endl;

            std::cout << "-------------" << std::endl;
            std::cout << "dot_out[" << i << "] = " << dot_out[i] << std::endl;
        }

    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_c );
    cudaFree(d_dot_out);

#endif // USE_CUDA
}
