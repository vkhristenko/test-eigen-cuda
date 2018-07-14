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
    auto lambda_1in1out = [n](std::function<void(Matrix10x10*, Matrix10x10*, int)> kernel_launch,
                                            std::function<Matrix10x10(Matrix10x10&)> transform,
                                            std::string && name) {
        std::cout << "test " << name << " has started" << std::endl;
        Matrix10x10 a[n], b[n];
        Matrix10x10 *d_a, *d_b;
        cudaMalloc((void**)&d_a, n * sizeof(Matrix10x10));
        cudaMalloc((void**)&d_b, n * sizeof(Matrix10x10));

        // randomize a matrix
        for (int i=0; i<n; i++)
            a[i] = Matrix10x10::Random(nrows, ncols) * 100;

        cudaMemcpy(d_a, a, n * sizeof(Matrix10x10), cudaMemcpyHostToDevice);

        kernel_launch(d_a, d_b, n);

        cudaDeviceSynchronize();
        cudaError err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "cuda eror!" << std::endl
                      << cudaGetErrorString(err) << std::endl;
            std::cout << "test " << name << " failed" << std::endl;
//            assert(false);
        }

        cudaMemcpy(b, d_b, n * sizeof(Matrix10x10), cudaMemcpyDeviceToHost);

        // check
        auto sum = 0;
        for (auto i=0; i<n; i++) {
            auto m = transform(a[i]);
            if (b[i].isApprox(m)) {
#ifdef DEBUG
                std::cout << "case " << i 
                          << m
                          << std::endl
                          << "--------------------------------------------------------"
                          << std::endl
                          << b[i]
                          << std::endl;
#endif
                sum++;
            } else {
                std::cout << "case " << i
                          << std::endl
                          << m
                          << std::endl
                          << "--------------------------------------------------------"
                          << std::endl
                          << b[i]
                          << std::endl;
            }
        }

        cudaFree(d_a);
        cudaFree(d_b);

        if (sum == n)
            std::cout << "test " + name + " passed" << std::endl;
        else {
            std::cout << "test " <<  name << " did not pass" << std::endl;
  //          assert(sum == n);
        }
    };

    lambda_1in1out(
        [](Matrix10x10* a, Matrix10x10* b, int n) { eigen_optest_0(a, b, n); },
        [](Matrix10x10& m) -> Matrix10x10 { return m.llt().matrixLLT(); },
        std::string("cholesky")
    );

    lambda_1in1out(
        [](Matrix10x10* a, Matrix10x10 *b, int n) { eigen_optest_1(a, b, n); },
        [](Matrix10x10& m) -> Matrix10x10 { return m.llt().matrixL().solve(m); },
        std::string("matrixL")
    );
    lambda_1in1out(
        [](Matrix10x10* a, Matrix10x10 *b, int n) { eigen_optest_2(a, b, n); },
        [](Matrix10x10& m) -> Matrix10x10 { return m.ldlt().solve(m); },
        std::string("LDLT")
    );
    lambda_1in1out(
        [](Matrix10x10* a, Matrix10x10 *b, int n) { eigen_optest_3(a, b, n); },
        [](Matrix10x10& m) -> Matrix10x10 { return m.colPivHouseholderQr().solve(m); },
        std::string("QR")
    );
#endif // USE_CUDA
}
