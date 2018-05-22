#include <iostream>

#ifdef USE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "test_cuda_eigen/interface/test_eigen_kernels0.h"
#endif

#include <Eigen/Dense>

int main() {
    std::cout << "hello world" << std::endl;

#ifdef USE_CUDA
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    int constexpr n = 100;
    Matrix10x10 a[n], b[n], c[n], d[n];
    Matrix10x10 *d_a, *d_b, *d_c, *d_d;

    for (auto i=0; i<n; i++) {
        a[i] = Matrix10x10::Random(nrows, ncols) * 100;
        b[i] = Matrix10x10::Random(nrows, ncols) * 100;

        if (i%10 == 0) {
            std::cout << "a[" << i << "] =" << std::endl
                << a[i] << std::endl;
            std::cout << "b[" << i << "] =" << std::endl
                << b[i] << std::endl;
            std::cout << "a[" << i << "] + b[" << i << "] = " << std::endl
                << a[i] + b[i] << std::endl;
        }
    }

    // alloc on the device
    cudaMalloc((void**)&d_a, n*sizeof(Matrix10x10));
    cudaMalloc((void**)&d_b, n*sizeof(Matrix10x10));
    cudaMalloc((void**)&d_c, n*sizeof(Matrix10x10));
    cudaMalloc((void**)&d_d, n*sizeof(Matrix10x10));

    // transfer to the device
    cudaMemcpy(d_a, a, n*sizeof(Matrix10x10), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(Matrix10x10), cudaMemcpyHostToDevice);
    
    // run the kernel
    eigen_matrix_add(d_a, d_b, d_c, n);

    // want to see errors from this kernel
    eigen_matrix_tests(d_a, d_d, n);

    cudaDeviceSynchronize();
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "cuda error!" << std::endl
            << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(c, d_c, n*sizeof(Matrix10x10), cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d_d, n*sizeof(Matrix10x10), cudaMemcpyDeviceToHost);

    for (auto i=0; i<n; i++) 
        if (i%10 == 0) {
            std::cout << "c[" << i << "]" << std::endl
                << c[i] << std::endl;

            std::cout << "d[" << i << "] = "  << std::endl
                << d[i] << std::endl;
        }

    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_c );
    cudaFree(d_d);

#endif // USE_CUDA
}
