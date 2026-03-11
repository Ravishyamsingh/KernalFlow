// ============================================================================
// KernelFlow — Linear (Fully Connected) Layer
// ============================================================================
//
// Implements: output = input @ weights^T + bias
//
//   input:   [batch, in_features]
//   weights: [out_features, in_features]  (row-major)
//   bias:    [out_features]
//   output:  [batch, out_features]
//
// The forward pass uses a fused GEMM-transpose kernel that computes
// C = A * B^T directly, avoiding an explicit transpose of the weight matrix.
// Bias is added via a separate lightweight kernel.
//
// Weight initialization uses Xavier uniform via curand on GPU.
//
// ============================================================================

#include "layers.hpp"
#include "kernels.hpp"
#include <curand_kernel.h>
#include <cmath>

#define TILE_SIZE 32

// ============================================================================
// Xavier Initialization Kernel
// ============================================================================
//
// Xavier uniform initialization sets weights to random values in the range:
//   [-limit, +limit]  where  limit = sqrt(6 / (fan_in + fan_out))
//
// This keeps the variance of activations and gradients roughly equal across
// layers, preventing vanishing/exploding signals in deep networks.
//
// Each thread initializes its own curandState with a unique seed derived
// from thread index + a base seed, then generates one uniform random value.
//
// ============================================================================
__global__ void xavier_init_kernel(float* data, int n, int fan_in, int fan_out,
                                   unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Initialize per-thread random state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Xavier uniform range
        float limit = sqrtf(6.0f / static_cast<float>(fan_in + fan_out));

        // Generate uniform in [-limit, +limit]
        float val = curand_uniform(&state);       // (0, 1]
        data[idx] = val * 2.0f * limit - limit;   // map to [-limit, +limit]
    }
}

// ============================================================================
// Tiled GEMM with Transposed B: C = A * B^T
// ============================================================================
//
// Standard GEMM computes: C[m][n] = sum_k A[m][k] * B[k][n]
// This kernel computes:   C[m][n] = sum_k A[m][k] * B[n][k]   (B transposed)
//
// The only difference from the regular tiled GEMM is how B is indexed
// when loading tiles into shared memory:
//   Regular:    B[t*TILE + ty][col]    → B_data[(t*TILE + ty) * N + col]
//   TransposeB: B[col][t*TILE + ty]    → B_data[col * K + (t*TILE + ty)]
//
// where col = blockIdx.x * TILE + threadIdx.x is the output column,
// which corresponds to the ROW of B (since B is transposed).
//
// Matrix dimensions:
//   A is [M × K]  (input: [batch × in_features])
//   B is [N × K]  (weights: [out_features × in_features], NOT transposed in memory)
//   C is [M × N]  (output: [batch × out_features])
//
// ============================================================================
__global__ void tiled_gemm_transB_kernel(const float* A, const float* B,
                                         float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // output row (batch dim)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // output col (out_features dim)

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A: A[row][t*TILE + tx]
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B^T: B^T[t*TILE + ty][col] = B[col][t*TILE + ty]
        // B is [N × K] row-major, so B[col][k] = B_data[col * K + k]
        int b_k = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_k < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[col * K + b_k];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Bias Addition Kernel
// ============================================================================
//
// Adds bias[j] to every row of a [batch × out_features] matrix:
//   output[i][j] += bias[j]
//
// Each thread handles one element.
// ============================================================================
__global__ void bias_add_kernel(float* output, const float* bias,
                                int batch, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features;

    if (idx < total) {
        int col = idx % out_features;
        output[idx] += bias[col];
    }
}

// ============================================================================
// LinearLayer Implementation
// ============================================================================

// Constructor: allocate weights and bias, initialize with Xavier uniform
LinearLayer::LinearLayer(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features)
{
    // Allocate weight tensor [out_features × in_features]
    weights_ = new Tensor({out_features, in_features});

    // Allocate bias tensor [out_features]
    bias_ = new Tensor({out_features});

    // Xavier initialization on GPU for weights
    int n = out_features * in_features;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Use a seed based on pointer address for reasonable uniqueness
    unsigned long long seed = reinterpret_cast<unsigned long long>(weights_->gpu_ptr()) ^ 42ULL;

    xavier_init_kernel<<<blocks, threads>>>(
        weights_->gpu_ptr(), n, in_features, out_features, seed
    );
    CHECK_CUDA(cudaGetLastError());

    // Zero-initialize bias on GPU
    bias_->fill(0.0f);

    CHECK_CUDA(cudaDeviceSynchronize());
}

// Destructor: free weight and bias tensors
LinearLayer::~LinearLayer() {
    delete weights_;
    delete bias_;
}

// Forward pass: output = input @ weights^T + bias
Tensor LinearLayer::forward(Tensor& input) {
    auto shape = input.get_shape();
    int batch = shape[0];
    int in_dim = shape[1];

    // Allocate output tensor [batch × out_features]
    Tensor output({batch, out_features_});

    // --- GEMM: output = input @ weights^T ---
    // input:   [batch × in_features]       (A: M=batch, K=in_features)
    // weights: [out_features × in_features] (B: N=out_features, K=in_features)
    // output:  [batch × out_features]       (C: M=batch, N=out_features)
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_features_ + TILE_SIZE - 1) / TILE_SIZE,
              (batch + TILE_SIZE - 1) / TILE_SIZE);

    tiled_gemm_transB_kernel<<<grid, block>>>(
        input.gpu_ptr(), weights_->gpu_ptr(), output.gpu_ptr(),
        batch, out_features_, in_dim
    );
    CHECK_CUDA(cudaGetLastError());

    // --- Bias addition: output[i][j] += bias[j] ---
    int total = batch * out_features_;
    int threads = 256;
    int blocks_bias = (total + threads - 1) / threads;

    bias_add_kernel<<<blocks_bias, threads>>>(
        output.gpu_ptr(), bias_->gpu_ptr(), batch, out_features_
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return output;
}

// Load pretrained weights and bias from CPU arrays to GPU
void LinearLayer::load_weights(const float* w, const float* b) {
    // Copy weight data: CPU → host buffer → GPU
    // w is expected in [out_features × in_features] row-major layout
    std::memcpy(weights_->cpu_ptr(), w,
                out_features_ * in_features_ * sizeof(float));
    weights_->to_gpu();

    // Copy bias data
    std::memcpy(bias_->cpu_ptr(), b, out_features_ * sizeof(float));
    bias_->to_gpu();
}

// Get weight and bias tensors (for inspection / serialization)
Tensor* LinearLayer::get_weights() { return weights_; }
Tensor* LinearLayer::get_bias() { return bias_; }