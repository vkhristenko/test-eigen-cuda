cmake_minimum_required(VERSION 3.8.0)

# project name
project(test-eigen-cuda)

# cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS} SYSTEM)
cuda_include_directories(${CUDA_INCLUDE_DIRS})
find_package(NVVM)

# eigen
message (EIGEN_HOME = ${EIGEN_HOME})
include_directories(${EIGEN_HOME} SYSTEM)
if (CUDA_FOUND)
    cuda_include_directories(${EIGEN_HOME})
endif()
