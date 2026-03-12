cmake_minimum_required(VERSION 3.18)

# IMPORTANT: Set CUDA flags BEFORE project() to allow VS 2026 compilation
# VS 2026 is not officially supported by CUDA 12.8, but works with this flag
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(KernelFlow LANGUAGES CXX CUDA)

# Visual Studio 2026 (v18) compatibility notes:
# - CUDA 12.8 officially supports VS 2017-2022
# - VS 2026 works with -allow-unsupported-compiler flag
# - If build fails, consider installing VS 2022 or building through VS IDE
if(MSVC)
    message(STATUS "Detected MSVC version: ${MSVC_VERSION}")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif()

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# CUDA architectures (adjust based on your GPU)
# 89 = Ada (RTX 40 series), 86 = Ampere (RTX 30 series), 75 = Turing (RTX 20 series)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
endif()

# Find packages
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit directory: ${CUDAToolkit_LIBRARY_DIR}")

# Try to find cuDNN (optional, not required for basic functionality)
find_library(CUDNN_LIB cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)
if(CUDNN_LIB)
    message(STATUS "Found cuDNN: ${CUDNN_LIB}")
else()
    message(WARNING "cuDNN not found - some features may be unavailable")
    set(CUDNN_LIB "")
endif()

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
    src/tensor.cpp
)
set_target_properties(kernelflow_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(src/tensor.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(kernelflow_core
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)
# Link cuDNN only if found
if(CUDNN_LIB)
    target_link_libraries(kernelflow_core ${CUDNN_LIB})
endif()

# Main executable
add_executable(kernelflow_main src/agent.cpp src/model.cu)
set_target_properties(kernelflow_main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)
set_source_files_properties(src/agent.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(kernelflow_main kernelflow_core)

# Benchmark executables
add_executable(bench_gemm benchmark/bench_gemm.cu)
set_target_properties(bench_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bench_gemm kernelflow_core)

add_executable(bench_inference benchmark/bench_inference.cu src/model.cu)
set_target_properties(bench_inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bench_inference kernelflow_core)

# Test executables
add_executable(test_tensor tests/test_tensor.cu)
set_target_properties(test_tensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_tensor kernelflow_core)

add_executable(test_model tests/test_model.cu src/model.cu)
set_target_properties(test_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_model kernelflow_core)

# DQN training executable
add_executable(dqn_train rl/dqn.cu src/model.cu)
set_target_properties(dqn_train PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(dqn_train kernelflow_core)