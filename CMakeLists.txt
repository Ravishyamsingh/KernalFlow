cmake_minimum_required(VERSION 3.18)
project(KernelFlow LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Find packages
find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIB cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect CUDA source files
file(GLOB KERNEL_SOURCES kernels/*.cu)
file(GLOB LAYER_SOURCES layers/*.cu)
set(RL_SOURCES rl/environment.cu rl/replay_buffer.cu)

# Core library from all CUDA sources
add_library(kernelflow_core STATIC
    ${KERNEL_SOURCES}
    ${LAYER_SOURCES}
    ${RL_SOURCES}
    src/stream_engine.cu
)
set_target_properties(kernelflow_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kernelflow_core
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${CUDNN_LIB}
)

# Main executable
add_executable(kernelflow_main src/tensor.cpp src/agent.cpp src/model.cu)
set_target_properties(kernelflow_main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)
set_source_files_properties(src/tensor.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(kernelflow_main kernelflow_core)

# Benchmark executables
add_executable(bench_gemm benchmark/bench_gemm.cu)
set_target_properties(bench_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bench_gemm kernelflow_core)

add_executable(bench_inference benchmark/bench_inference.cu src/tensor.cpp src/model.cu)
set_target_properties(bench_inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(src/tensor.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(bench_inference kernelflow_core)

# Test executables
add_executable(test_tensor tests/test_tensor.cu src/tensor.cpp)
set_target_properties(test_tensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(src/tensor.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_tensor CUDA::cudart)

add_executable(test_model tests/test_model.cu src/tensor.cpp src/model.cu)
set_target_properties(test_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_model kernelflow_core)

# DQN training executable
add_executable(dqn_train rl/dqn.cu src/tensor.cpp src/model.cu src/stream_engine.cu)
set_target_properties(dqn_train PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(src/tensor.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(dqn_train kernelflow_core)