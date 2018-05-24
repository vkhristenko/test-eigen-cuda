#include <iostream>

#ifdef USE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "test_cuda_eigen/interface/test_eigen_kernels0.h"
#endif

/// Compute the Cholesky decomposition (A = L*L^T) on GPU
int main() {

#ifdef USE_CUDA
    // Allocate CPU memory for a 100 ten-by-ten matrices
    const int sz = 10;
    Matrix10x10 a, l;
    Matrix10x10 *d_a, *d_l;

    // Create a 100 Hermitian positive definite matrices
    a = Matrix10x10::Random(sz, sz);
    a = 0.5*(a + a.transpose());
    a = a + a.cols()*Matrix10x10::Identity();

    // Allocate GPU memory
    cudaMalloc((void**)&d_a, sizeof(Matrix10x10));
    cudaMalloc((void**)&d_l, sizeof(Matrix10x10));

    // Transfer to the GPU
    cudaMemcpy(d_a, &a, sizeof(Matrix10x10), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, &l, sizeof(Matrix10x10), cudaMemcpyHostToDevice);
    
    // Call the kernel wrapper with the input and output matrices
    eigen_matrix_cholesky_decomposition(d_a, d_l);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "cuda error!" << std::endl
            << cudaGetErrorString(err) << std::endl;
    }

    // Transfer back from the GPU
    cudaMemcpy(&l, d_l, sizeof(Matrix10x10), cudaMemcpyDeviceToHost);

    // Check the result with CPU values
    DecompLLT decomp;
    decomp.compute(a);
    Matrix10x10 l_cpu = decomp.matrixLLT();
    if (!(l.isApprox(l_cpu))) {
      std::cout << "Test failed" << std::endl;
      std::cout << l_cpu << std::endl << std::endl;
      std::cout << l << std::endl;
      return 1;
    }

    std::cout << "Tests passed!" << std::endl;

    cudaFree(d_a);
    cudaFree(d_l);
#endif  // USE_CUDA

  return 0;
}
