#pragma once

#include "tensor.hpp"
#include <cstring>

// ============================================================================
// LinearLayer — Fully Connected Layer
// ============================================================================
//
// Computes: output = input @ weights^T + bias
//
//   input shape:   [batch, in_features]
//   weights shape: [out_features, in_features]
//   bias shape:    [out_features]
//   output shape:  [batch, out_features]
//
// Uses a tiled GEMM kernel with transposed B for the matrix multiply,
// then adds bias via a separate kernel.
//
// ============================================================================

// CUDA kernels defined in layers/linear.cu
__global__ void xavier_init_kernel(float* data, int n, int fan_in, int fan_out,
                                   unsigned long long seed);
__global__ void tiled_gemm_transB_kernel(const float* A, const float* B,
                                         float* C, int M, int N, int K);
__global__ void bias_add_kernel(float* output, const float* bias,
                                int batch, int out_features);

class LinearLayer {
private:
    Tensor* weights_;       // [out_features, in_features]
    Tensor* bias_;          // [out_features]
    int in_features_;
    int out_features_;

public:
    // Construct with Xavier-initialized weights and zero bias
    LinearLayer(int in_features, int out_features);

    // No copies
    LinearLayer(const LinearLayer&) = delete;
    LinearLayer& operator=(const LinearLayer&) = delete;

    // Forward pass: input[batch, in] → output[batch, out]
    Tensor forward(Tensor& input);

    // Load pretrained weights from CPU arrays
    // w: [out_features × in_features] row-major, b: [out_features]
    void load_weights(const float* w, const float* b);

    // Accessors for weights and bias tensors
    Tensor* get_weights();
    Tensor* get_bias();

    int get_in_features() const { return in_features_; }
    int get_out_features() const { return out_features_; }

    ~LinearLayer();
};

// ============================================================================
// Conv2DLayer (layers/conv2d.cu)
// ============================================================================
// (To be implemented)

// ============================================================================
// BatchNormLayer (layers/batchnorm.cu)
// ============================================================================
// (To be implemented)