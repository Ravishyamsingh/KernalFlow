// ============================================================================
// KernelFlow — StreamEngine: Multi-Stream Inference
// ============================================================================
//
// Overlaps multiple sub-batch forward passes on separate CUDA streams.
//
// The key insight: existing layer forward() calls use the default stream
// and call cudaDeviceSynchronize(). To get true overlap, we must launch
// the raw kernels directly on specific streams, bypassing the wrappers.
//
// For the DQN network (Linear→ReLU→Linear→ReLU→Linear):
//   For each stream:
//     1. AsyncMemcpy sub-batch input into scratch buffer on stream[i]
//     2. tiled_gemm_transB_kernel on stream[i]
//     3. bias_add_kernel on stream[i]
//     4. relu_kernel on stream[i]
//     5. (repeat for next linear+relu)
//     6. Final linear (no activation)
//     7. AsyncMemcpy output from scratch into merged output on stream[i]
//   Synchronize all streams.
//
// ============================================================================

#include "stream_engine.hpp"
#include "layers.hpp"
#include <cstdio>

#define TILE_SIZE 32

// ============================================================================
// StreamEngine — Constructor / Destructor
// ============================================================================

StreamEngine::StreamEngine(Sequential* model) : model_(model) {
    for (int i = 0; i < N_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams_[i]));
    }
    printf("[StreamEngine] Created %d CUDA streams\n", N_STREAMS);
}

StreamEngine::~StreamEngine() {
    for (int i = 0; i < N_STREAMS; ++i) {
        cudaStreamDestroy(streams_[i]);
    }
}

// ============================================================================
// Single-stream baseline forward
// ============================================================================
Tensor StreamEngine::forward(Tensor& input) {
    return model_->forward(input);
}

// ============================================================================
// Helper: run one linear layer's kernels on a specific stream
//
// Performs: output = input @ weights^T + bias
// All pointers are device pointers; kernels are launched on `stream`.
// No synchronization is performed — the caller handles it.
// ============================================================================
static void linear_forward_on_stream(
    const float* d_input, float* d_output,
    const float* d_weights, const float* d_bias,
    int batch, int in_features, int out_features,
    cudaStream_t stream)
{
    // GEMM: output = input @ weights^T
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_features + TILE_SIZE - 1) / TILE_SIZE,
              (batch        + TILE_SIZE - 1) / TILE_SIZE);

    tiled_gemm_transB_kernel<<<grid, block, 0, stream>>>(
        d_input, d_weights, d_output, batch, out_features, in_features
    );

    // Bias add
    int total = batch * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bias_add_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_bias, batch, out_features
    );
}

// ============================================================================
// Helper: run ReLU on a specific stream
// ============================================================================
static void relu_on_stream(float* data, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

// ============================================================================
// streamed_forward — Multi-stream pipelined forward pass
//
// Splits the input batch [total_batch × input_dim] into N_STREAMS sub-batches.
// Each sub-batch is processed independently on its own CUDA stream:
//   stream 0: rows [0,              sub_batch)
//   stream 1: rows [sub_batch,      2*sub_batch)
//   stream 2: rows [2*sub_batch,    3*sub_batch)
//   stream 3: rows [3*sub_batch,    total_batch)
//
// The network layers are extracted from the Sequential model:
//   Layer 0: LinearLayerAdapter(4 → 128)
//   Layer 1: ReLU
//   Layer 2: LinearLayerAdapter(128 → 64)
//   Layer 3: ReLU
//   Layer 4: LinearLayerAdapter(64 → 2)
//
// Per-stream scratch buffers are allocated for intermediate activations.
// Results are written directly into the correct offset of the output tensor.
// ============================================================================
Tensor StreamEngine::streamed_forward(Tensor& input) {
    auto shape = input.get_shape();
    int total_batch = shape[0];
    int input_dim   = shape[1];

    // Extract linear layers from the model
    // Expected architecture: Linear, ReLU, Linear, ReLU, Linear
    LinearLayer* lin0 = static_cast<LinearLayerAdapter*>(model_->get_layer(0))->get();
    LinearLayer* lin1 = static_cast<LinearLayerAdapter*>(model_->get_layer(2))->get();
    LinearLayer* lin2 = static_cast<LinearLayerAdapter*>(model_->get_layer(4))->get();

    int h0 = lin0->get_out_features();  // 128
    int h1 = lin1->get_out_features();  //  64
    int out_dim = lin2->get_out_features(); // 2

    // Allocate output tensor [total_batch × out_dim]
    Tensor output({total_batch, out_dim});

    // Per-stream scratch buffers for intermediate activations
    float* d_scratch_h0[N_STREAMS];  // [sub_batch × 128]
    float* d_scratch_h1[N_STREAMS];  // [sub_batch × 64]

    int sub_batch_base = total_batch / N_STREAMS;
    int remainder = total_batch % N_STREAMS;

    // Compute the max sub-batch size (last stream may get extras)
    int max_sub = sub_batch_base + (remainder > 0 ? 1 : 0);

    // Allocate scratch per stream
    for (int i = 0; i < N_STREAMS; ++i) {
        CHECK_CUDA(cudaMalloc(&d_scratch_h0[i], max_sub * h0 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_scratch_h1[i], max_sub * h1 * sizeof(float)));
    }

    // Launch sub-batch processing on each stream
    int offset = 0;
    for (int s = 0; s < N_STREAMS; ++s) {
        // Distribute remainder across first few streams
        int sub_batch = sub_batch_base + (s < remainder ? 1 : 0);
        if (sub_batch == 0) continue;

        cudaStream_t stream = streams_[s];

        // Pointers into the input and output at this sub-batch's offset
        float* d_in  = input.gpu_ptr()  + offset * input_dim;
        float* d_out = output.gpu_ptr() + offset * out_dim;

        // Layer 0: Linear(input_dim → h0) on stream
        linear_forward_on_stream(
            d_in, d_scratch_h0[s],
            lin0->get_weights()->gpu_ptr(), lin0->get_bias()->gpu_ptr(),
            sub_batch, input_dim, h0, stream
        );

        // Layer 1: ReLU in-place on scratch_h0
        relu_on_stream(d_scratch_h0[s], sub_batch * h0, stream);

        // Layer 2: Linear(h0 → h1) on stream
        linear_forward_on_stream(
            d_scratch_h0[s], d_scratch_h1[s],
            lin1->get_weights()->gpu_ptr(), lin1->get_bias()->gpu_ptr(),
            sub_batch, h0, h1, stream
        );

        // Layer 3: ReLU in-place on scratch_h1
        relu_on_stream(d_scratch_h1[s], sub_batch * h1, stream);

        // Layer 4: Linear(h1 → out_dim) → write directly into output
        linear_forward_on_stream(
            d_scratch_h1[s], d_out,
            lin2->get_weights()->gpu_ptr(), lin2->get_bias()->gpu_ptr(),
            sub_batch, h1, out_dim, stream
        );

        offset += sub_batch;
    }

    // Synchronize all streams
    for (int s = 0; s < N_STREAMS; ++s) {
        CHECK_CUDA(cudaStreamSynchronize(streams_[s]));
    }

    // Free scratch
    for (int i = 0; i < N_STREAMS; ++i) {
        cudaFree(d_scratch_h0[i]);
        cudaFree(d_scratch_h1[i]);
    }

    return output;
}

// ============================================================================
// benchmark — Compare single-stream vs multi-stream forward pass
//
// Times n_iters forward passes with and without streams.
// Uses cudaEvent for accurate GPU timing.
// ============================================================================
void StreamEngine::benchmark(int batch_size, int input_dim, int n_iters) {
    printf("\n");
    printf("============================================================\n");
    printf("  StreamEngine Benchmark\n");
    printf("  Batch: %d | Input: %d | Iterations: %d | Streams: %d\n",
           batch_size, input_dim, n_iters, N_STREAMS);
    printf("============================================================\n\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Create test input
    Tensor test_input({batch_size, input_dim});
    test_input.fill(0.1f);

    // ---- Warm-up ----
    for (int i = 0; i < 10; ++i) {
        Tensor warmup_in({batch_size, input_dim});
        CHECK_CUDA(cudaMemcpy(warmup_in.gpu_ptr(), test_input.gpu_ptr(),
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        Tensor out = model_->forward(warmup_in);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- Time single-stream (baseline) ----
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < n_iters; ++i) {
        Tensor in_copy({batch_size, input_dim});
        CHECK_CUDA(cudaMemcpy(in_copy.gpu_ptr(), test_input.gpu_ptr(),
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        Tensor out = forward(in_copy);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_single_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_single_ms, start, stop));

    printf("  Single-stream: %8.2f ms total  (%6.3f ms/iter)\n",
           time_single_ms, time_single_ms / n_iters);

    // ---- Time multi-stream ----
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < n_iters; ++i) {
        Tensor in_copy({batch_size, input_dim});
        CHECK_CUDA(cudaMemcpy(in_copy.gpu_ptr(), test_input.gpu_ptr(),
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        Tensor out = streamed_forward(in_copy);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_multi_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_multi_ms, start, stop));

    printf("  Multi-stream:  %8.2f ms total  (%6.3f ms/iter)\n",
           time_multi_ms, time_multi_ms / n_iters);

    // ---- Speedup ----
    float speedup = time_single_ms / time_multi_ms;
    printf("\n  Streams speedup: %.2fx\n", speedup);

    printf("============================================================\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
//
//  G R A P H   E N G I N E
//
// ============================================================================
//
// CUDA Graphs eliminate per-kernel CPU launch overhead by recording the
// entire inference pipeline once, then replaying it as a single GPU command.
//
// Normal inference flow (CPU-bound):
//
//   CPU: launch GEMM0 ──wait── launch bias0 ──wait── launch ReLU0 ──wait── ...
//   GPU:              [GEMM0]            [bias0]             [ReLU0]
//                  ^^^idle^^^        ^^^idle^^^          ^^^idle^^^
//
// With CUDA Graphs (GPU-bound):
//
//   CPU: cudaGraphLaunch() ─────────────────────── done
//   GPU: [GEMM0][bias0][ReLU0][GEMM1][bias1][ReLU1][GEMM2][bias2]
//              ^^^ no idle gaps — GPU runs the full pipeline without CPU ^^^
//
// For small networks like DQN (4→128→64→2), each kernel takes ~10-50μs
// but CPU launch overhead is ~5-15μs per kernel. With 8 kernel launches
// (3 GEMMs + 3 bias adds + 2 ReLUs), that's 40-120μs of launch overhead
// per inference — often more than the actual compute time!
//
// CUDA Graphs reduce this to a single ~5μs launch for the entire pipeline.
//
// ============================================================================

// ============================================================================
// GraphEngine Constructor / Destructor
// ============================================================================

GraphEngine::GraphEngine()
    : graph_(nullptr), graph_exec_(nullptr), captured_(false),
      d_input_buf_(nullptr), d_h0_buf_(nullptr),
      d_h1_buf_(nullptr), d_output_buf_(nullptr),
      batch_size_(0), input_dim_(0), output_dim_(0)
{
    CHECK_CUDA(cudaStreamCreate(&capture_stream_));
    printf("[GraphEngine] Initialized — ready for graph capture\n");
}

GraphEngine::~GraphEngine() {
    // Destroy the graph executable and graph if they exist
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_)      cudaGraphDestroy(graph_);

    // Free pre-allocated buffers
    if (d_input_buf_)  cudaFree(d_input_buf_);
    if (d_h0_buf_)     cudaFree(d_h0_buf_);
    if (d_h1_buf_)     cudaFree(d_h1_buf_);
    if (d_output_buf_) cudaFree(d_output_buf_);

    cudaStreamDestroy(capture_stream_);
}

// ============================================================================
// capture_forward — Record the inference pipeline into a CUDA graph
//
// Steps:
//   1. Extract layer weights from the Sequential model
//   2. Pre-allocate all intermediate buffers at FIXED addresses
//   3. Copy sample input into the fixed input buffer
//   4. Begin stream capture (cudaStreamBeginCapture)
//   5. Launch all kernels on the capture stream — these are NOT executed,
//      they are only RECORDED into the graph
//   6. End stream capture (cudaStreamEndCapture) → produces cudaGraph_t
//   7. Instantiate the graph (cudaGraphInstantiate) → produces cudaGraphExec_t
//
// After this, execute_graph() can replay the entire pipeline
// with a single cudaGraphLaunch() call.
//
// IMPORTANT: The model architecture must be Linear→ReLU→Linear→ReLU→Linear.
// The kernel parameters (grid, block, pointers) are baked into the graph.
// Only the INPUT DATA can change between replays (via cudaMemcpy before launch).
// ============================================================================
void GraphEngine::capture_forward(Sequential& model, Tensor& sample_input) {
    auto shape = sample_input.get_shape();
    batch_size_ = shape[0];
    input_dim_  = shape[1];

    // Extract the three linear layers
    LinearLayer* lin0 = static_cast<LinearLayerAdapter*>(model.get_layer(0))->get();
    LinearLayer* lin1 = static_cast<LinearLayerAdapter*>(model.get_layer(2))->get();
    LinearLayer* lin2 = static_cast<LinearLayerAdapter*>(model.get_layer(4))->get();

    int h0 = lin0->get_out_features();       // 128
    int h1 = lin1->get_out_features();       //  64
    output_dim_ = lin2->get_out_features();  //   2

    // ------------------------------------------------------------------
    // Step 1: Pre-allocate fixed-address buffers
    //
    // These buffers will be used during BOTH capture and replay.
    // cudaMalloc returns deterministic addresses that won't change,
    // so the kernel arguments baked into the graph remain valid.
    // ------------------------------------------------------------------
    if (d_input_buf_)  cudaFree(d_input_buf_);
    if (d_h0_buf_)     cudaFree(d_h0_buf_);
    if (d_h1_buf_)     cudaFree(d_h1_buf_);
    if (d_output_buf_) cudaFree(d_output_buf_);

    CHECK_CUDA(cudaMalloc(&d_input_buf_,  batch_size_ * input_dim_  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_h0_buf_,     batch_size_ * h0          * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_h1_buf_,     batch_size_ * h1          * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_buf_, batch_size_ * output_dim_ * sizeof(float)));

    // ------------------------------------------------------------------
    // Step 2: Copy sample input into the fixed buffer
    //
    // The graph needs valid data during capture for correct kernel
    // parameter validation. The actual values don't matter — they'll
    // be overwritten before each replay.
    // ------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_input_buf_, sample_input.gpu_ptr(),
                          batch_size_ * input_dim_ * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Step 3: Destroy any previously captured graph
    // ------------------------------------------------------------------
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (graph_)      { cudaGraphDestroy(graph_);          graph_ = nullptr; }

    // ------------------------------------------------------------------
    // Step 4: Begin stream capture
    //
    // After this call, ALL CUDA operations on capture_stream_ are
    // recorded into a graph instead of being executed. This includes
    // kernel launches, memcpys, and synchronization primitives.
    //
    // cudaStreamCaptureModeGlobal: captures work from ALL threads that
    // enqueue onto this stream (safest mode).
    // ------------------------------------------------------------------
    CHECK_CUDA(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));

    // ------------------------------------------------------------------
    // Step 5: Launch the full inference pipeline on the capture stream
    //
    // These kernel launches are NOT executed — they are only recorded.
    // We use the raw kernels (not layer wrappers) because the wrappers
    // call cudaDeviceSynchronize(), which is illegal during capture.
    // ------------------------------------------------------------------

    // Layer 0: Linear(input_dim → h0)
    linear_forward_on_stream(
        d_input_buf_, d_h0_buf_,
        lin0->get_weights()->gpu_ptr(), lin0->get_bias()->gpu_ptr(),
        batch_size_, input_dim_, h0, capture_stream_
    );

    // Layer 1: ReLU in-place on h0_buf
    relu_on_stream(d_h0_buf_, batch_size_ * h0, capture_stream_);

    // Layer 2: Linear(h0 → h1)
    linear_forward_on_stream(
        d_h0_buf_, d_h1_buf_,
        lin1->get_weights()->gpu_ptr(), lin1->get_bias()->gpu_ptr(),
        batch_size_, h0, h1, capture_stream_
    );

    // Layer 3: ReLU in-place on h1_buf
    relu_on_stream(d_h1_buf_, batch_size_ * h1, capture_stream_);

    // Layer 4: Linear(h1 → output_dim)
    linear_forward_on_stream(
        d_h1_buf_, d_output_buf_,
        lin2->get_weights()->gpu_ptr(), lin2->get_bias()->gpu_ptr(),
        batch_size_, h1, output_dim_, capture_stream_
    );

    // ------------------------------------------------------------------
    // Step 6: End stream capture → produces the graph
    // ------------------------------------------------------------------
    CHECK_CUDA(cudaStreamEndCapture(capture_stream_, &graph_));

    // ------------------------------------------------------------------
    // Step 7: Instantiate (compile) the graph
    //
    // This step validates the graph, resolves dependencies, and produces
    // an optimized executable. The GPU driver may fuse adjacent kernels
    // or reorder independent operations for better throughput.
    // ------------------------------------------------------------------
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

    captured_ = true;

    // Count the nodes in the graph for diagnostics
    size_t num_nodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph_, nullptr, &num_nodes));
    printf("[GraphEngine] Captured graph with %zu nodes (batch=%d, %d→...→%d)\n",
           num_nodes, batch_size_, input_dim_, output_dim_);
}

// ============================================================================
// execute_graph — Replay the captured pipeline with new input data
//
// Steps:
//   1. Copy new input into the fixed input buffer (d_input_buf_)
//   2. Launch the entire captured graph with ONE API call
//   3. Copy output from the fixed output buffer (d_output_buf_) into a Tensor
//
// The graph replays ALL kernels in the recorded order without any
// per-kernel CPU launch overhead. The GPU receives the full pipeline
// as a single command and executes it back-to-back.
//
// This is the key performance advantage: instead of 8 separate kernel
// launches (each with ~5-15μs CPU overhead), we have 1 graph launch
// (~5μs total), saving 35-105μs per inference.
// ============================================================================
Tensor GraphEngine::execute_graph(Tensor& input) {
    // Validate dimensions match capture
    auto shape = input.get_shape();
    int batch = shape[0];
    int dim   = shape[1];

    if (batch != batch_size_ || dim != input_dim_) {
        fprintf(stderr, "[GraphEngine] ERROR: input shape [%d, %d] does not "
                "match captured shape [%d, %d]\n",
                batch, dim, batch_size_, input_dim_);
        std::exit(EXIT_FAILURE);
    }

    // Step 1: Copy new input data into the fixed graph input buffer.
    //
    // This is the ONLY memory operation needed before replay — the graph
    // will read from d_input_buf_ and write through d_h0_buf_, d_h1_buf_
    // to d_output_buf_ using the same addresses as during capture.
    CHECK_CUDA(cudaMemcpyAsync(d_input_buf_, input.gpu_ptr(),
                               batch_size_ * input_dim_ * sizeof(float),
                               cudaMemcpyDeviceToDevice, capture_stream_));

    // Step 2: Launch the entire graph — ONE API call replaces 8 kernel launches.
    //
    // cudaGraphLaunch enqueues the entire captured pipeline onto the stream.
    // The GPU will execute all 8 kernels (3 GEMMs + 3 bias adds + 2 ReLUs)
    // back-to-back without waiting for CPU dispatch between them.
    CHECK_CUDA(cudaGraphLaunch(graph_exec_, capture_stream_));

    // Step 3: Synchronize and copy output to a new Tensor
    CHECK_CUDA(cudaStreamSynchronize(capture_stream_));

    Tensor output({batch_size_, output_dim_});
    CHECK_CUDA(cudaMemcpy(output.gpu_ptr(), d_output_buf_,
                          batch_size_ * output_dim_ * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    return output;
}

// ============================================================================
// compare_graph_vs_normal — Benchmark normal launches vs graph replay
//
// Measures the end-to-end latency of n_iters forward passes using:
//   1. Normal kernel launches (model.forward() with per-kernel dispatch)
//   2. CUDA Graph replay (single cudaGraphLaunch per inference)
//
// The speedup is most visible for:
//   - Small networks with many layers (high launch-to-compute ratio)
//   - Small batch sizes (kernels finish fast, CPU becomes bottleneck)
//   - Repeated inference with the same batch size (e.g., RL action selection)
//
// For our DQN network (4→128→64→2 with batch=512):
//   - 8 kernel launches per inference
//   - Each kernel runs ~10-50μs on GPU
//   - CPU launch overhead: ~5-15μs × 8 = 40-120μs per inference
//   - Graph launch overhead: ~5μs total per inference
//   - Expected speedup: 1.5-3x depending on GPU
// ============================================================================
void GraphEngine::compare_graph_vs_normal(Sequential& model, int batch_size,
                                          int input_dim, int n_iters) {
    printf("\n");
    printf("============================================================\n");
    printf("  GraphEngine Benchmark: Normal vs CUDA Graph\n");
    printf("  Batch: %d | Input: %d | Iterations: %d\n",
           batch_size, input_dim, n_iters);
    printf("============================================================\n\n");

    // Create test input
    Tensor test_input({batch_size, input_dim});
    test_input.fill(0.5f);

    // Capture the graph if not already done
    if (!captured_ || batch_size != batch_size_ || input_dim != input_dim_) {
        capture_forward(model, test_input);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ---- Warm-up (both paths) ----
    for (int i = 0; i < 50; ++i) {
        Tensor warmup_in({batch_size, input_dim});
        CHECK_CUDA(cudaMemcpy(warmup_in.gpu_ptr(), test_input.gpu_ptr(),
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        Tensor out1 = model.forward(warmup_in);

        Tensor warmup_in2({batch_size, input_dim});
        CHECK_CUDA(cudaMemcpy(warmup_in2.gpu_ptr(), test_input.gpu_ptr(),
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        Tensor out2 = execute_graph(warmup_in2);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- Time normal kernel launches ----
    //
    // Each iteration calls model.forward(), which dispatches 8 separate
    // kernel launches from the CPU. The GPU idles between kernels while
    // the CPU prepares the next launch.
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < n_iters; ++i) {
        Tensor in_copy({batch_size, input_dim});
        CHECK_CUDA(cudaMemcpy(in_copy.gpu_ptr(), test_input.gpu_ptr(),
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        Tensor out = model.forward(in_copy);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_normal_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_normal_ms, start, stop));
    float avg_normal = time_normal_ms / n_iters;

    // ---- Time graph replay ----
    //
    // Each iteration calls execute_graph(), which:
    //   1. Copies input into the fixed buffer (~2μs)
    //   2. Launches the ENTIRE pipeline with ONE cudaGraphLaunch (~5μs)
    //   3. Synchronizes
    //
    // The GPU executes all 8 kernels back-to-back with zero idle gaps
    // between them, because the driver has the full dependency graph.
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < n_iters; ++i) {
        Tensor in_copy({batch_size, input_dim});
        CHECK_CUDA(cudaMemcpy(in_copy.gpu_ptr(), test_input.gpu_ptr(),
                              batch_size * input_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        Tensor out = execute_graph(in_copy);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_graph_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_graph_ms, start, stop));
    float avg_graph = time_graph_ms / n_iters;

    // ---- Results ----
    float speedup = time_normal_ms / time_graph_ms;

    printf("  Normal launch: %8.2f ms total  (%.4f ms avg per inference)\n",
           time_normal_ms, avg_normal);
    printf("  Graph replay:  %8.2f ms total  (%.4f ms avg per inference)\n",
           time_graph_ms, avg_graph);
    printf("\n  Speedup: %.2fx\n", speedup);
    printf("\n  Analysis:\n");
    printf("    Per-inference CPU overhead saved: %.4f ms\n",
           avg_normal - avg_graph);
    printf("    Over %d inferences: %.2f ms saved\n",
           n_iters, time_normal_ms - time_graph_ms);
    printf("============================================================\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}
