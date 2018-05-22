cmake_minimum_required(VERSION 3.8.0)

# project name
project(test-eigen-cuda)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

# cuda
find_package(CUDA)
add_definitions(-DUSE_CUDA)
include_directories(${CUDA_INCLUDE_DIRS} SYSTEM)
cuda_include_directories(${CUDA_INCLUDE_DIRS})
find_package(NVVM)

# eigen
message (EIGEN_HOME = ${EIGEN_HOME})
include_directories(${EIGEN_HOME} SYSTEM)
if (CUDA_FOUND)
    cuda_include_directories(${EIGEN_HOME})
endif()

# always follow the source tree for inclusions ( a la cmssw)
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(test_cuda_eigen)
