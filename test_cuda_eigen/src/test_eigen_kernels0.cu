#include "test_cuda_eigen/interface/test_eigen_kernels0.h"
#include <cusolverDn.h>
#include <Eigen/Dense>

static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
	switch (error)
	{
	case CUSOLVER_STATUS_SUCCESS:
		return "CUSOLVER_SUCCESS";

	case CUSOLVER_STATUS_NOT_INITIALIZED:
		return "CUSOLVER_STATUS_NOT_INITIALIZED";

	case CUSOLVER_STATUS_ALLOC_FAILED:
		return "CUSOLVER_STATUS_ALLOC_FAILED";

	case CUSOLVER_STATUS_INVALID_VALUE:
		return "CUSOLVER_STATUS_INVALID_VALUE";

	case CUSOLVER_STATUS_ARCH_MISMATCH:
		return "CUSOLVER_STATUS_ARCH_MISMATCH";

	case CUSOLVER_STATUS_EXECUTION_FAILED:
		return "CUSOLVER_STATUS_EXECUTION_FAILED";

	case CUSOLVER_STATUS_INTERNAL_ERROR:
		return "CUSOLVER_STATUS_INTERNAL_ERROR";

	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	}

	return "<unknown>";
}

inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
{
	if (CUSOLVER_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSOLVE error in file '%s', line %d, error: %s \nterminating!\n", __FILE__, __LINE__, \
			_cusolverGetErrorEnum(err)); \
			assert(0); \
	}
}

extern "C" void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

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

/* vector multiplication for eigen */

__global__ void cu_eigen_vmult(Eigen::Vector3d* va, 
							Eigen::Vector3d* vb,
							Eigen::Vector3d* vc) {
	int id = blockIdx.x;
	vc[id] = va[id] * vb[id];
}

void eigen_vector_mult(Eigen::Vector3d* va,
					Eigen::Vector3d* vb,
					Eigen::Vector3d* vc, int const n) {
	cu_eigen_vmult<<<n, 1>>>(va, vb, vc);
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
    auto llt  = min[id].llt();
    mout[id] = llt.matrixLLT();
}

__global__ void cu_eigen_optest_0(Matrix10x10 *in, Matrix10x10 *out) {
    int idx = blockIdx.x;
    auto llt = in[idx].llt();
    out[idx] = llt.matrixLLT();
}

__global__ void cu_eigen_optest_1(Matrix10x10 *in, Matrix10x10 *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("111\n");
    /*
    Eigen::Matrix2f A, b;
    A << 2, -1, -1, 3;
    b << 1,2,3,1;
    auto llt = A.llt().matrixL().solve(b);
    Eigen::Matrix2f tmp = llt;
    */

    if (idx < n)
        out[idx] = in[idx].llt().matrixL().solve(in[idx]);

//    Matrix10x10 x = llt;
//    llt.xxx();
//    printf("222\n");
//    out[idx] = llt;
//    printf("333\n");
}

__global__ void cu_eigen_optest_2(Matrix10x10 *in, Matrix10x10 *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
         out[idx] = in[idx].ldlt().solve(in[idx]);
    }
}

__global__ void cu_eigen_optest_3(Matrix10x10 *in, Matrix10x10 *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
         out[idx] = in[idx].colPivHouseholderQr().solve(in[idx]);
    }
}

void eigen_optest_1(Matrix10x10 *in, Matrix10x10 *out, int n) {
    cu_eigen_optest_1<<<n, 1>>>(in, out, n);
}

void eigen_optest_0(Matrix10x10 *in, Matrix10x10 *out, int n) {
    cu_eigen_optest_0<<<n, 1>>>(in, out);
}

void eigen_optest_2(Matrix10x10 *in, Matrix10x10 *out, int n) {
    cu_eigen_optest_2<<<n,1>>>(in, out, n);
}

void eigen_optest_3(Matrix10x10 *in, Matrix10x10 *out, int n) {
    cu_eigen_optest_3<<<n,1>>>(in, out, n);
}

void eigen_matrix_tests(Matrix10x10 *min,
                        Matrix10x10 *mout,
                        int const n) {
    cu_eigen_mtests<<<n, 1>>>(min, mout);
}

__global__ void map_to_eigen_matrix(double* data, Matrix10x10* m, int size) {
  *m = Eigen::Map<Matrix10x10>(data, size, size);
}

void eigen_matrix_cholesky_decomposition(Matrix10x10* ma, Matrix10x10* ml) {
  cusolverDnHandle_t handle = NULL;
  cusolveSafeCall(cusolverDnCreate(&handle));

  // use lower triangle of matrix in cholesky decomposition
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  double* raw_ptr = ma->data();

  int work_size = 0;
  cusolveSafeCall(cusolverDnDpotrf_bufferSize(handle, uplo, ma->cols(), raw_ptr, ma->rows(), &work_size));
  double *work; cudaMalloc(&work, work_size * sizeof(double));

  cusolveSafeCall(cusolverDnDpotrf(handle, uplo, ma->cols(), raw_ptr, ma->rows(), work, work_size, NULL));
    map_to_eigen_matrix<<<1,1>>>(raw_ptr, ml, ma->rows());
  // this doesn't work because Eigen::Map will assume raw_ptr is a CPU address
  // *ml = Eigen::Map<Matrix3x3>(raw_ptr, m, m);
}
