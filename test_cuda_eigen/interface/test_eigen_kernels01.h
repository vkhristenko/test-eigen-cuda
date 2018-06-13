#ifndef execs_test_cuda_eigen_interface_test_eigen_kernels01_h
#define execs_test_cuda_eigen_interface_test_eigen_kernels01_h

#include <Eigen/Cholesky>
#include <Eigen/Dense>

int constexpr nrows = 8;
int constexpr ncols = 8;
using Matrix8x8 = Eigen::Matrix<double, nrows, ncols>;
using DecompLLT = Eigen::LLT<Matrix8x8>;

void eigen_vector_add(Eigen::Vector3d*, Eigen::Vector3d*, Eigen::Vector3d*, int const);

void eigen_vector_dot(Eigen::Vector3d*, Eigen::Vector3d*, Eigen::Vector3d::value_type*,
    int const);

void eigen_matrix_cholesky_decomposition(Matrix8x8* ma, Matrix8x8* ml);

void eigen_matrix_add(Matrix8x8*, Matrix8x8*, Matrix8x8*, int const);

void eigen_matrix_tests(Matrix8x8*, Matrix8x8*, int const);

#endif // execs_test_cuda_interface_test_kernels0_h
