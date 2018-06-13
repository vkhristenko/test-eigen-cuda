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

    int constexpr n = 8;
    Matrix8x8 a[n], b[n], c[n], d[n];
    Matrix8x8 *d_a, *d_b, *d_c, *d_d;

    for (auto i=0; i<n; i++) {
        a[i] = Matrix8x8::Random(nrows, ncols) * 8;
        b[i] = Matrix10x10::Random(nrows, ncols) * 8;

        if (i%2 == 0) {
            std::cout << "a[" << i << "] =" << std::endl
                << a[i] << std::endl;
            std::cout << "b[" << i << "] =" << std::endl
                << b[i] << std::endl;
            std::cout << "a[" << i << "] + b[" << i << "] = " << std::endl
                << a[i] + b[i] << std::endl;
        }
    }

    // alloc on the device
    cudaMalloc((void**)&d_a, n*sizeof(Matrix8x8));
    cudaMalloc((void**)&d_b, n*sizeof(Matrix8x8));
    cudaMalloc((void**)&d_c, n*sizeof(Matrix8x8));
    cudaMalloc((void**)&d_d, n*sizeof(Matrix8x8));

    // transfer to the device
    cudaMemcpy(d_a, a, n*sizeof(Matrix8x8), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(Matrix8x8), cudaMemcpyHostToDevice);
    
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

    cudaMemcpy(c, d_c, n*sizeof(Matrix8x8), cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d_d, n*sizeof(Matrix8x8), cudaMemcpyDeviceToHost);

    for (auto i=0; i<n; i++) {
      if (i%2 == 0) {
        std::cout << "c[" << i << "]" << std::endl
            << c[i] << std::endl;

        std::cout << "d[" << i << "] = "  << std::endl
            << d[i] << std::endl;
      }
    }

    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_c );
    cudaFree(d_d);

#endif // USE_CUDA
}
