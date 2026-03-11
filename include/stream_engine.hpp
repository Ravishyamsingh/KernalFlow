#pragma once

#include "tensor.hpp"
#include "model.hpp"
#include <cuda_runtime.h>

// ============================================================================
// StreamEngine — Multi-stream inference engine
// ============================================================================
//
// Splits a large batch across N_STREAMS CUDA streams to overlap:
//   - Host→Device memcpy per sub-batch (if needed)
//   - Kernel launches for GEMM, bias, activations per sub-batch
//   - Independent sub-batches execute concurrently on the GPU
//
// Usage:
//   StreamEngine engine(model_ptr);
//   Tensor out = engine.streamed_forward(input);   // 4-stream pipelined
//   Tensor out = engine.forward(input);             // single-stream baseline
//
// ============================================================================

#define N_STREAMS 4

class StreamEngine {
private:
    cudaStream_t streams_[N_STREAMS];
    Sequential*  model_;   // non-owning pointer — caller owns the model

public:
    StreamEngine(Sequential* model);
    ~StreamEngine();

    StreamEngine(const StreamEngine&) = delete;
    StreamEngine& operator=(const StreamEngine&) = delete;

    // Single-stream baseline forward (delegates to model->forward)
    Tensor forward(Tensor& input);

    // Multi-stream forward: splits batch into N_STREAMS groups, runs
    // each sub-batch on a separate CUDA stream, synchronizes, merges output.
    Tensor streamed_forward(Tensor& input);

    // Benchmark: compare single-stream vs multi-stream for n_iters forward passes
    // Prints timing results and speedup.
    void benchmark(int batch_size, int input_dim, int n_iters = 1000);

    // Access raw stream handle
    cudaStream_t get_stream(int idx) { return streams_[idx]; }
};

// ============================================================================
// GraphEngine — CUDA Graph-based inference engine
// ============================================================================
//
// WHY CUDA GRAPHS?
//
// Every CUDA kernel launch incurs CPU-side overhead:
//   1. Driver validates kernel parameters
//   2. Launch command is serialized into the GPU command buffer
//   3. The command is sent to the GPU over PCIe / NVLink
//
// For small, fast kernels (like our DQN network: 4→128→64→2), the GPU
// finishes each kernel before the CPU can even submit the next one.
// This means the GPU sits IDLE between kernels, waiting for the CPU.
//
// CUDA Graphs solve this by capturing the entire kernel sequence once,
// then replaying it with a SINGLE launch command. The GPU receives the
// full pipeline as one atomic unit — no inter-kernel CPU round-trips.
//
// Typical improvement: 2-5x for inference pipelines with many small kernels.
//
// Workflow:
//   1. capture_forward()  — record all kernel launches into a cudaGraph_t
//   2. cudaGraphInstantiate() — compile the graph into an executable
//   3. execute_graph()    — copy input, launch entire graph, copy output
//
// Constraints:
//   - All GPU memory addresses must remain fixed between capture and replay
//   - No dynamic allocations (cudaMalloc) inside the captured region
//   - No host-device synchronization inside the captured region
//   - Kernel grid/block dimensions must stay constant
//
// This is why we pre-allocate ALL intermediate buffers before capture,
// and launch raw kernels (not the layer wrappers that call cudaMalloc).
//
// ============================================================================

class GraphEngine {
private:
    cudaStream_t    capture_stream_;   // Stream used for capture and replay
    cudaGraph_t     graph_;            // The captured graph
    cudaGraphExec_t graph_exec_;       // The instantiated (compiled) graph
    bool            captured_;         // Whether a graph has been captured

    // Pre-allocated fixed-address GPU buffers for the captured pipeline.
    // These MUST remain at the same addresses between capture and replay.
    float* d_input_buf_;     // [batch × input_dim]       — graph input
    float* d_h0_buf_;        // [batch × hidden0]         — after Linear0
    float* d_h1_buf_;        // [batch × hidden1]         — after Linear2
    float* d_output_buf_;    // [batch × output_dim]      — graph output

    // Captured dimensions (must match at replay time)
    int batch_size_;
    int input_dim_;
    int output_dim_;

public:
    GraphEngine();
    ~GraphEngine();

    GraphEngine(const GraphEngine&) = delete;
    GraphEngine& operator=(const GraphEngine&) = delete;

    // Capture the forward pass of a Sequential model into a CUDA graph.
    // sample_input must have the exact batch size that will be used at replay.
    // The model architecture must be: Linear, ReLU, Linear, ReLU, Linear.
    void capture_forward(Sequential& model, Tensor& sample_input);

    // Execute the captured graph with new input data.
    // Input must match the batch_size and input_dim used during capture.
    Tensor execute_graph(Tensor& input);

    // Benchmark: compare normal kernel launches vs graph replay.
    // Runs n_iters inferences with each method and prints comparison.
    void compare_graph_vs_normal(Sequential& model, int batch_size = 512,
                                 int input_dim = 4, int n_iters = 10000);

    bool is_captured() const { return captured_; }
};
