### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_61,code=sm_61
    )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Specify include directories
include_directories(
    kernels
    utility
    )

# Specify library paths
#link_directories(
#    /opt/foobar/lib
#    /opt/joestuff/lib
#    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
  parallel_reduction
  src/parallel_reduction.cu
  kernels/parallel_reduction.cuh
)
cuda_add_executable(
  parallel_histogram
  src/parallel_histogram.cu
  kernels/parallel_histogram.cuh
)
cuda_add_executable(
  parallel_scan_inclusive
  src/parallel_scan_inclusive.cu
  kernels/parallel_scan.cuh
)
cuda_add_executable(
  parallel_scan_exclusive
  src/parallel_scan_exclusive.cu
  kernels/parallel_scan.cuh
)

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries(
#    hellocuda
#    -lfoobar
#    -ljoestuff
#)
