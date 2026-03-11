// ============================================================================
// KernelFlow — Activation Function Kernels
// ============================================================================
//
// Element-wise activation functions for neural network layers.
// All operate in-place on GPU memory.
//
// Kernels:
//   1. relu_kernel     — f(x) = max(0, x)
//   2. sigmoid_kernel  — f(x) = 1 / (1 + exp(-x))
//   3. tanh_kernel     — f(x) = tanh(x)
//   4. softmax_kernel  — numerically stable softmax per row
//
// ============================================================================

#include "kernels.hpp"
#include <cfloat>

// ============================================================================
// 1. ReLU Kernel
// ============================================================================
//
// Rectified Linear Unit: clamps negative values to zero.
// Most popular activation in modern deep learning — cheap and effective.
//
//   f(x) = max(0, x)
//
// Each thread processes one element. Simple but extremely fast because
// there's no expensive math — just a comparison.
// ============================================================================
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// ============================================================================
// 2. Sigmoid Kernel
// ============================================================================
//
// Logistic sigmoid: squashes any real number into (0, 1).
// Used in output layers for binary classification and in gating mechanisms.
//
//   f(x) = 1 / (1 + exp(-x))
//
// For numerical stability with large negative x, expf(-x) → ∞ could
// overflow, but in practice float exp saturates gracefully. The formula
// is inherently stable in this direction.
// ============================================================================
__global__ void sigmoid_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

// ============================================================================
// 3. Tanh Kernel
// ============================================================================
//
// Hyperbolic tangent: squashes any real number into (-1, 1).
// Zero-centered unlike sigmoid, sometimes preferred in hidden layers.
//
//   f(x) = tanh(x)
//
// CUDA provides a fast intrinsic tanhf().
// ============================================================================
__global__ void tanh_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = tanhf(data[idx]);
    }
}

// ============================================================================
// 4. Softmax Kernel (Numerically Stable, Per-Row)
// ============================================================================
//
// Converts a row of raw scores (logits) into a probability distribution.
// Used in the output layer for multi-class classification and in DQN
// to convert Q-values into action probabilities.
//
// Naive formula:   softmax(x_i) = exp(x_i) / sum(exp(x_j))
//   Problem: if x_i = 1000, exp(1000) = INF → overflow!
//
// Stable formula:  softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
//   Subtracting max ensures the largest exponent is exp(0) = 1.
//   Mathematically identical, numerically safe.
//
// This kernel processes ONE ROW per block:
//   - blockIdx.x  = row index (batch element)
//   - threadIdx.x = column index within the row
//   - row_size    = number of columns (classes/actions)
//
// For rows larger than blockDim.x, each thread handles multiple elements
// via a stride loop.
//
// Steps:
//   1. Find row max using parallel reduction in shared memory
//   2. Compute exp(x - max) for each element, store back in-place
//   3. Sum all exp values using parallel reduction in shared memory
//   4. Divide each element by the sum
//
// ============================================================================

// Max threads per block for softmax — must be power of 2 for reduction
#define SOFTMAX_THREADS 256

__global__ void softmax_kernel(float* data, int row_size) {
    // Each block handles one row (one sample in the batch)
    int row = blockIdx.x;
    float* row_data = data + row * row_size;

    int tid = threadIdx.x;

    // Shared memory: used for parallel reductions (max and sum)
    __shared__ float sdata[SOFTMAX_THREADS];

    // ================================================================
    // Step 1: Find the maximum value in this row
    //
    // Each thread finds the max across its stride-assigned elements,
    // then we reduce across threads in shared memory.
    // ================================================================
    float local_max = -FLT_MAX;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_data[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Parallel reduction to find global max across all threads
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];  // broadcast: all threads read the result

    // ================================================================
    // Step 2: Compute exp(x_i - max) for each element (in-place)
    // Step 3: Compute the sum of all exp values
    //
    // We do both in one pass: compute exp, write it back, and
    // accumulate the local sum simultaneously.
    // ================================================================
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = expf(row_data[i] - row_max);
        row_data[i] = val;      // store exp value in-place
        local_sum += val;        // accumulate for sum
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Parallel reduction to find global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];  // broadcast: all threads read the result

    // ================================================================
    // Step 4: Normalize — divide each exp value by the sum
    //
    // After this, row_data[i] ∈ (0, 1) and sum(row_data) = 1.0
    // ================================================================
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_data[i] /= row_sum;
    }
}

// ============================================================================
// Host Wrapper: launch_relu
// ============================================================================
void launch_relu(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(data, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ============================================================================
// Host Wrapper: launch_sigmoid
// ============================================================================
void launch_sigmoid(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sigmoid_kernel<<<blocks, threads>>>(data, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ============================================================================
// Host Wrapper: launch_tanh
// ============================================================================
void launch_tanh(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    tanh_kernel<<<blocks, threads>>>(data, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ============================================================================
// Host Wrapper: launch_softmax
// ============================================================================
//
// Parameters:
//   data       — device pointer to [batch_size × row_size] matrix, row-major
//   row_size   — number of columns (classes/actions per sample)
//   batch_size — number of rows (samples in the batch)
//
// Launches one block per row. Each block has SOFTMAX_THREADS threads.
// ============================================================================
void launch_softmax(float* data, int row_size, int batch_size) {
    int threads = SOFTMAX_THREADS;
    int blocks = batch_size;  // one block per row

    softmax_kernel<<<blocks, threads>>>(data, row_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}