#pragma once

#include "tensor.hpp"

// ============================================================================
// GEMM Kernels (kernels/gemm.cu)
// ============================================================================

// Naive matrix multiply — one thread per output element, global memory only
__global__ void naive_gemm_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K);

// Tiled matrix multiply — shared memory tiling with TILE_SIZE=32
__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K);

// Host wrapper — launches tiled GEMM with proper grid/block dims
// A[M×K] * B[K×N] = C[M×N], all device pointers, row-major
void launch_gemm(float* A, float* B, float* C, int M, int N, int K);

// ============================================================================
// Activation Kernels (kernels/activations.cu)
// ============================================================================

// GPU fill kernel (used by Tensor::fill)
__global__ void fill_kernel(float* data, float value, int n);

// Element-wise activations — operate in-place on device memory
__global__ void relu_kernel(float* data, int n);
__global__ void sigmoid_kernel(float* data, int n);
__global__ void tanh_kernel(float* data, int n);

// Softmax — one block per row, numerically stable (subtract max)
// data is [batch_size × row_size], row-major
__global__ void softmax_kernel(float* data, int row_size);

// Host wrappers — launch kernels with correct grid/block dimensions
void launch_relu(float* data, int n);
void launch_sigmoid(float* data, int n);
void launch_tanh(float* data, int n);
void launch_softmax(float* data, int row_size, int batch_size);

// ============================================================================
// Reduction Kernels (kernels/reduction.cu)
// ============================================================================

// Block-level sum reduction — output[blockIdx.x] = partial sum
__global__ void reduction_sum_kernel(const float* input, float* output, int n);

// Block-level max reduction — output[blockIdx.x] = partial max
__global__ void reduction_max_kernel(const float* input, float* output, int n);

// Host wrappers — two-phase reduction, return scalar result on CPU
float launch_sum(float* data, int n);
float launch_mean(float* data, int n);
float launch_max(float* data, int n);

// ============================================================================
// Conv2D Kernel (kernels/conv2d.cu)
// ============================================================================
// (To be implemented in Phase 2)

// ============================================================================
// BatchNorm Kernel (kernels/batchnorm.cu)
// ============================================================================
// (To be implemented in Phase 2)