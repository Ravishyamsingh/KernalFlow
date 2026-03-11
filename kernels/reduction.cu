// ============================================================================
// KernelFlow — Parallel Reduction Kernels
// ============================================================================
//
// Reduction = collapsing an array of N values into a single scalar.
//   sum:  [3, 1, 4, 1, 5] → 14
//   max:  [3, 1, 4, 1, 5] → 5
//   mean: [3, 1, 4, 1, 5] → 2.8
//
// WHY PARALLEL REDUCTION?
//   A sequential loop takes O(N) time. On a GPU with thousands of threads,
//   parallel reduction takes O(N/P + log P) time where P = number of threads.
//   For N = 1M elements: CPU ~1ms, GPU ~0.05ms.
//
// ALGORITHM (Two-Phase):
//   Phase 1 — Each block reduces its chunk to a single value.
//             gridDim.x blocks produce gridDim.x partial results.
//   Phase 2 — A single block reduces the partial results to the final scalar.
//
// WITHIN EACH BLOCK:
//   1. Each thread loads & accumulates elements via stride loop
//   2. Store into shared memory
//   3. Tree reduction in shared memory (halving active threads each step)
//   4. Final warp uses __shfl_down_sync (no shared memory needed, no __syncthreads)
//   5. Thread 0 writes the block's result to global output
//
// ============================================================================

#include "kernels.hpp"
#include <cfloat>

#define REDUCTION_THREADS 256

// ============================================================================
// Warp-Level Reduction: __shfl_down_sync explained
// ============================================================================
//
// A "warp" is 32 threads that execute in lockstep on the same SIMD unit.
// Within a warp, threads can directly read each other's registers using
// shuffle instructions — no shared memory needed, no synchronization needed.
//
// __shfl_down_sync(mask, val, delta):
//   Thread i receives the value of thread (i + delta) within the warp.
//
//   Example with 8 threads (simplified), reducing sum:
//
//     Thread:     T0   T1   T2   T3   T4   T5   T6   T7
//     Values:     [3]  [1]  [4]  [1]  [5]  [9]  [2]  [6]
//
//     shfl_down by 4:  T0 += T4, T1 += T5, T2 += T6, T3 += T7
//     Values:     [8]  [10] [6]  [7]   -    -    -    -
//
//     shfl_down by 2:  T0 += T2, T1 += T3
//     Values:     [14] [17]  -    -    -    -    -    -
//
//     shfl_down by 1:  T0 += T1
//     Values:     [31]  -    -    -    -    -    -    -
//
//   Final result in T0. No shared memory, no sync — pure register ops.
//   0xffffffff mask means all 32 threads participate.
//
// ============================================================================

// Warp-level sum reduction (last 32 threads, no __syncthreads needed)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ============================================================================
// 1. Sum Reduction Kernel
// ============================================================================
//
// Each block reduces its assigned chunk of the input to a single sum.
// Output: out[blockIdx.x] = sum of this block's elements.
//
// Handles arbitrary array sizes via stride loop in the load phase.
// ============================================================================
__global__ void reduction_sum_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[REDUCTION_THREADS];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Load phase: accumulate multiple elements per thread via stride ---
    // This handles arrays larger than gridDim * blockDim and ensures
    // all elements are covered.
    float sum = 0.0f;
    for (int i = global_id; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // --- Tree reduction in shared memory ---
    // Halve active threads each step. Stop at 32 (one warp).
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // --- Final warp reduction using shuffle (no sync needed) ---
    if (tid < 32) {
        // Load from shared memory into register for warp shuffle
        float val = sdata[tid] + sdata[tid + 32];
        val = warp_reduce_sum(val);

        // Thread 0 writes this block's result
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// 2. Max Reduction Kernel
// ============================================================================
//
// Same structure as sum, but uses fmaxf instead of addition.
// ============================================================================
__global__ void reduction_max_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[REDUCTION_THREADS];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Load phase: find max across stride-assigned elements ---
    float local_max = -FLT_MAX;
    for (int i = global_id; i < n; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // --- Tree reduction in shared memory ---
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // --- Final warp reduction ---
    if (tid < 32) {
        float val = fmaxf(sdata[tid], sdata[tid + 32]);
        val = warp_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// Host Wrapper: launch_sum
// ============================================================================
//
// Two-phase reduction:
//   Phase 1: N elements → num_blocks partial sums
//   Phase 2: num_blocks partial sums → 1 final sum
//
// Returns the scalar result on CPU.
// ============================================================================
float launch_sum(float* data, int n) {
    int threads = REDUCTION_THREADS;
    int blocks = (n + threads - 1) / threads;
    // Cap blocks to avoid excessive overhead for phase 2
    if (blocks > 1024) blocks = 1024;

    // Phase 1: reduce N elements → `blocks` partial sums
    float* d_partial = nullptr;
    CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));

    reduction_sum_kernel<<<blocks, threads>>>(data, d_partial, n);
    CHECK_CUDA(cudaGetLastError());

    // Phase 2: reduce `blocks` partial sums → 1 final sum
    float* d_result = nullptr;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    reduction_sum_kernel<<<1, threads>>>(d_partial, d_result, blocks);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy scalar result back to CPU
    float result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float),
                           cudaMemcpyDeviceToHost));

    cudaFree(d_partial);
    cudaFree(d_result);

    return result;
}

// ============================================================================
// Host Wrapper: launch_mean
// ============================================================================
//
// Computes mean = sum / n using the sum reduction.
// ============================================================================
float launch_mean(float* data, int n) {
    float sum = launch_sum(data, n);
    return sum / static_cast<float>(n);
}

// ============================================================================
// Host Wrapper: launch_max
// ============================================================================
//
// Two-phase max reduction, same pattern as launch_sum.
// ============================================================================
float launch_max(float* data, int n) {
    int threads = REDUCTION_THREADS;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    // Phase 1: reduce N elements → `blocks` partial maxes
    float* d_partial = nullptr;
    CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));

    reduction_max_kernel<<<blocks, threads>>>(data, d_partial, n);
    CHECK_CUDA(cudaGetLastError());

    // Phase 2: reduce `blocks` partial maxes → 1 final max
    float* d_result = nullptr;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    reduction_max_kernel<<<1, threads>>>(d_partial, d_result, blocks);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy scalar result back to CPU
    float result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float),
                           cudaMemcpyDeviceToHost));

    cudaFree(d_partial);
    cudaFree(d_result);

    return result;
}