#include <iostream>

#ifdef USE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "test_cuda_eigen/interface/test_kernels0.h"
#endif

int main() {
    std::cout << "hello world" << std::endl;

#ifdef USE_CUDA
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    int constexpr n = 1000;
    int a[n], b[n], c[n];
    int *d_a, *d_b, *d_c;

    for (auto i=0; i<n; i++) {
        a[i] = i;
        b[i] = i;
    }

    // alloc on the device
    cudaMalloc((void**)&d_a, n * sizeof(int) );
    cudaMalloc((void**)&d_b, n*sizeof(int));
    cudaMalloc((void**)&d_c, n*sizeof(int));

    // transfer to the device
    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);
    
    // run the kernel
    vector_add(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

    for (auto i=0; i<n; i++) 
        if (i%100 == 0)
            std::cout << "c[" << i << "] = " << c[i] << std::endl;

    cudaFree( d_a );
    cudaFree( d_a );
    cudaFree( d_a );

#endif // USE_CUDA
}
