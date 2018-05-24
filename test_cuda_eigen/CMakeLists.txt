message("compiling test_eigen_cuda")

#or files with .cu extension
if (CUDA_FOUND)
  file (GLOB CUSOURCES src/*.cu)
  # silence Eigen warnings
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
  cuda_add_library(test_cuda_eigen_kernels SHARED ${CUSOURCES})
  target_link_libraries(test_cuda_eigen_kernels)
  set(MY_CUDA_LIBS test_cuda_eigen_kernels ${CUDA_cusolver_LIBRARY})
endif()

# src directory should contain only source files with main entry point
file (GLOB SOURCES src/*.cpp)
foreach (SOURCE ${SOURCES})
    string(REPLACE ".cpp" "" EXEC ${SOURCE})
    get_filename_component(EXEC ${EXEC} NAME)
    message("adding executable ${EXEC}")
    add_executable(test_cuda_eigen_${EXEC} ${SOURCE})
    target_link_libraries(test_cuda_eigen_${EXEC} ${CUDA_LIBRARIES} ${MY_CUDA_LIBS})
endforeach(SOURCE)
