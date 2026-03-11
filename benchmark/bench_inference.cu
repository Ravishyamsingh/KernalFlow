// ============================================================================
// KernelFlow — Complete Inference Benchmark Suite
// ============================================================================
//
// Four comprehensive benchmarks covering every layer of the KernelFlow stack:
//
//   BENCHMARK 1: GEMM Performance        — Naive vs Tiled vs cuBLAS
//   BENCHMARK 2: Activation Kernels      — CPU vs GPU (ReLU, Softmax)
//   BENCHMARK 3: Full Forward Pass       — Basic vs Streams vs Graphs
//   BENCHMARK 4: RL Training Speed       — Sequential CPU vs Parallel GPU
//
// All results are printed as formatted ASCII tables and saved to
// benchmark/results.csv for offline analysis / Python plotting.
//
// ============================================================================

#include "tensor.hpp"
#include "kernels.hpp"
#include "layers.hpp"
#include "model.hpp"
#include "stream_engine.hpp"
#include "environment.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ── Benchmark parameters ────────────────────────────────────────────────────
#define WARMUP      10
#define GEMM_ITERS  100
#define ACT_SIZE    1048576   // 1M elements
#define ACT_ITERS   200
#define FWD_ITERS   1000
#define RL_STEPS    5000

// ── Forward‑declare CSV helpers ─────────────────────────────────────────────
static FILE* csv_fp = nullptr;

static void csv_open() {
    csv_fp = fopen("benchmark/results.csv", "w");
    if (!csv_fp) {
        printf("WARNING: Could not open benchmark/results.csv for writing\n");
        return;
    }
    fprintf(csv_fp, "benchmark,variant,parameter,time_ms,metric_value,metric_unit\n");
}

static void csv_row(const char* bench, const char* variant,
                    const char* param, double time_ms,
                    double metric, const char* unit) {
    if (!csv_fp) return;
    fprintf(csv_fp, "%s,%s,%s,%.6f,%.4f,%s\n",
            bench, variant, param, time_ms, metric, unit);
}

static void csv_close() {
    if (csv_fp) { fclose(csv_fp); csv_fp = nullptr; }
}

// ── GPU info ────────────────────────────────────────────────────────────────
static void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU : %s  (SM %d.%d, %.1f GB, %d SMs)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.multiProcessorCount);
    printf("\n");
}

// ############################################################################
//
//  BENCHMARK 1 — GEMM Performance
//
// ############################################################################

static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static double gflops(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms * 1e6);
}

static void bench1_gemm() {
    printf("############################################################\n");
    printf("#  BENCHMARK 1 — GEMM Performance                          #\n");
    printf("#  Naive GPU  vs  Tiled GPU (shared mem)  vs  cuBLAS       #\n");
    printf("############################################################\n\n");

    int sizes[] = {256, 512, 1024, 2048};
    int nsizes  = 4;

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("%-8s | %-10s %-10s | %-10s %-10s | %-10s %-10s | %-10s %-10s\n",
           "Size", "CPU ms", "GFLOPS",
           "Naive ms", "GFLOPS",
           "Tiled ms", "GFLOPS",
           "cuBLAS ms", "GFLOPS");
    printf("---------+----------------------+"
           "----------------------+----------------------+"
           "----------------------\n");

    for (int si = 0; si < nsizes; ++si) {
        int M = sizes[si], N = sizes[si], K = sizes[si];

        // ── Host arrays ──
        float* hA = new float[M * K];
        float* hB = new float[K * N];
        float* hC = new float[M * N];
        for (int i = 0; i < M * K; ++i) hA[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < K * N; ++i) hB[i] = (float)rand() / RAND_MAX;

        // ── Device arrays ──
        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

        // ── CPU baseline (skip for 2048) ──
        double cpu_ms = 0, cpu_gf = 0;
        if (M <= 1024) {
            for (int w = 0; w < WARMUP; ++w) cpu_gemm(hA, hB, hC, M, N, K);
            auto st = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < GEMM_ITERS; ++i) cpu_gemm(hA, hB, hC, M, N, K);
            auto en = std::chrono::high_resolution_clock::now();
            cpu_ms = std::chrono::duration<double, std::milli>(en - st).count() / GEMM_ITERS;
            cpu_gf = gflops(M, N, K, cpu_ms);
            csv_row("GEMM", "CPU", std::to_string(M).c_str(), cpu_ms, cpu_gf, "GFLOPS");
        }

        // ── Naive GEMM (skip for M > 1024, very slow) ──
        double naive_ms = 0, naive_gf = 0;
        if (M <= 1024) {
            dim3 nb(32, 32), ng((N + 31) / 32, (M + 31) / 32);
            for (int w = 0; w < WARMUP; ++w) {
                naive_gemm_kernel<<<ng, nb>>>(dA, dB, dC, M, N, K);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int i = 0; i < GEMM_ITERS; ++i)
                naive_gemm_kernel<<<ng, nb>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaEventRecord(t1));
            CHECK_CUDA(cudaEventSynchronize(t1));
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            naive_ms = ms / GEMM_ITERS;
            naive_gf = gflops(M, N, K, naive_ms);
            csv_row("GEMM", "Naive_GPU", std::to_string(M).c_str(), naive_ms, naive_gf, "GFLOPS");
        }

        // ── Tiled GEMM ──
        dim3 tb(32, 32), tg((N + 31) / 32, (M + 31) / 32);
        for (int w = 0; w < WARMUP; ++w)
            tiled_gemm_kernel<<<tg, tb>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < GEMM_ITERS; ++i)
            tiled_gemm_kernel<<<tg, tb>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float tiled_raw; cudaEventElapsedTime(&tiled_raw, t0, t1);
        double tiled_ms = tiled_raw / GEMM_ITERS;
        double tiled_gf = gflops(M, N, K, tiled_ms);
        csv_row("GEMM", "Tiled_GPU", std::to_string(M).c_str(), tiled_ms, tiled_gf, "GFLOPS");

        // ── cuBLAS ──
        float alpha = 1.0f, beta = 0.0f;
        for (int w = 0; w < WARMUP; ++w)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < GEMM_ITERS; ++i)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float cb_raw; cudaEventElapsedTime(&cb_raw, t0, t1);
        double cb_ms = cb_raw / GEMM_ITERS;
        double cb_gf = gflops(M, N, K, cb_ms);
        csv_row("GEMM", "cuBLAS", std::to_string(M).c_str(), cb_ms, cb_gf, "GFLOPS");

        // ── Print row ──
        char szs[16]; snprintf(szs, sizeof(szs), "%dx%d", M, N);
        if (M <= 1024) {
            printf("%-8s | %8.2f   %8.1f   | %8.3f   %8.1f   | %8.3f   %8.1f   | %8.3f   %8.1f\n",
                   szs, cpu_ms, cpu_gf, naive_ms, naive_gf,
                   tiled_ms, tiled_gf, cb_ms, cb_gf);
        } else {
            printf("%-8s | %8s   %8s   | %8s   %8s   | %8.3f   %8.1f   | %8.3f   %8.1f\n",
                   szs, "N/A", "N/A", "N/A", "N/A",
                   tiled_ms, tiled_gf, cb_ms, cb_gf);
        }

        delete[] hA; delete[] hB; delete[] hC;
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        CHECK_CUDA(cudaEventDestroy(t0));
        CHECK_CUDA(cudaEventDestroy(t1));
    }

    cublasDestroy(handle);
    printf("\n");
}

// ############################################################################
//
//  BENCHMARK 2 — Activation Kernels
//
// ############################################################################

// CPU ReLU baseline
static void cpu_relu(float* data, int n) {
    for (int i = 0; i < n; ++i)
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
}

// CPU Softmax baseline (per-row, numerically stable)
static void cpu_softmax(float* data, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = data + r * cols;
        float mx = row[0];
        for (int c = 1; c < cols; ++c) mx = fmaxf(mx, row[c]);
        float s = 0.0f;
        for (int c = 0; c < cols; ++c) { row[c] = expf(row[c] - mx); s += row[c]; }
        for (int c = 0; c < cols; ++c) row[c] /= s;
    }
}

static void bench2_activations() {
    printf("############################################################\n");
    printf("#  BENCHMARK 2 — Activation Kernels                        #\n");
    printf("#  CPU loop  vs  GPU kernel  (ReLU, Softmax)               #\n");
    printf("#  Array size: %d (%.1f M elements)         #\n",
           ACT_SIZE, ACT_SIZE / 1e6);
    printf("############################################################\n\n");

    int n = ACT_SIZE;

    // ── Allocate host + device ──
    float* h_data  = new float[n];
    float* h_back  = new float[n];
    for (int i = 0; i < n; ++i) h_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    memcpy(h_back, h_data, n * sizeof(float));

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    // Bandwidth helper: bytes = n * sizeof(float) read + write
    auto bw_gbs = [&](int elems, double ms) -> double {
        double bytes = 2.0 * elems * sizeof(float); // read + write
        return bytes / (ms * 1e6);                   // ms→s /1e3 then /1e9 → /1e6
    };

    printf("%-12s | %10s %10s | %10s %10s | %10s\n",
           "Kernel", "CPU ms", "CPU GB/s",
           "GPU ms", "GPU GB/s", "Speedup");
    printf("-------------|----------------------|"
           "----------------------|----------\n");

    // ═══════════ ReLU ═══════════
    {
        // CPU
        for (int w = 0; w < WARMUP; ++w) {
            memcpy(h_data, h_back, n * sizeof(float));
            cpu_relu(h_data, n);
        }
        auto st = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ACT_ITERS; ++i) {
            memcpy(h_data, h_back, n * sizeof(float));
            cpu_relu(h_data, n);
        }
        auto en = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(en - st).count() / ACT_ITERS;
        double cpu_bw = bw_gbs(n, cpu_ms);

        // GPU
        CHECK_CUDA(cudaMemcpy(d_data, h_back, n * sizeof(float), cudaMemcpyHostToDevice));
        int thr = 256, blk = (n + thr - 1) / thr;
        for (int w = 0; w < WARMUP; ++w) relu_kernel<<<blk, thr>>>(d_data, n);
        CHECK_CUDA(cudaDeviceSynchronize());

        // reset
        CHECK_CUDA(cudaMemcpy(d_data, h_back, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < ACT_ITERS; ++i)
            relu_kernel<<<blk, thr>>>(d_data, n);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float raw; cudaEventElapsedTime(&raw, t0, t1);
        double gpu_ms = raw / ACT_ITERS;
        double gpu_bw = bw_gbs(n, gpu_ms);

        printf("%-12s | %10.3f %8.1f   | %10.4f %8.1f   | %8.1fx\n",
               "ReLU", cpu_ms, cpu_bw, gpu_ms, gpu_bw, cpu_ms / gpu_ms);

        csv_row("Activation", "CPU_ReLU",  "1M", cpu_ms, cpu_bw, "GB/s");
        csv_row("Activation", "GPU_ReLU",  "1M", gpu_ms, gpu_bw, "GB/s");
    }

    // ═══════════ Softmax ═══════════
    {
        // Treat as 1024 rows × 1024 cols = 1M
        int rows = 1024, cols = 1024;

        // CPU
        for (int w = 0; w < WARMUP; ++w) {
            memcpy(h_data, h_back, n * sizeof(float));
            cpu_softmax(h_data, rows, cols);
        }
        auto st = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ACT_ITERS; ++i) {
            memcpy(h_data, h_back, n * sizeof(float));
            cpu_softmax(h_data, rows, cols);
        }
        auto en = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(en - st).count() / ACT_ITERS;
        double cpu_bw = bw_gbs(n, cpu_ms);

        // GPU
        CHECK_CUDA(cudaMemcpy(d_data, h_back, n * sizeof(float), cudaMemcpyHostToDevice));
        for (int w = 0; w < WARMUP; ++w) {
            softmax_kernel<<<rows, 256>>>(d_data, cols);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(d_data, h_back, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < ACT_ITERS; ++i)
            softmax_kernel<<<rows, 256>>>(d_data, cols);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float raw; cudaEventElapsedTime(&raw, t0, t1);
        double gpu_ms = raw / ACT_ITERS;
        double gpu_bw = bw_gbs(n, gpu_ms);

        printf("%-12s | %10.3f %8.1f   | %10.4f %8.1f   | %8.1fx\n",
               "Softmax", cpu_ms, cpu_bw, gpu_ms, gpu_bw, cpu_ms / gpu_ms);

        csv_row("Activation", "CPU_Softmax", "1M", cpu_ms, cpu_bw, "GB/s");
        csv_row("Activation", "GPU_Softmax", "1M", gpu_ms, gpu_bw, "GB/s");
    }

    delete[] h_data; delete[] h_back;
    cudaFree(d_data);
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    printf("\n");
}

// ############################################################################
//
//  BENCHMARK 3 — Full Forward Pass
//
// ############################################################################

static Sequential* build_dqn_net() {
    Sequential* net = new Sequential();
    net->add(new LinearLayerAdapter(4, 128));
    net->add(new ReLULayer());
    net->add(new LinearLayerAdapter(128, 64));
    net->add(new ReLULayer());
    net->add(new LinearLayerAdapter(64, 2));
    return net;
}

// CPU‑only forward pass (for baseline): simple matrix‑multiply + relu + bias
// weights are stored as [out × in] row‑major
struct CPULinear {
    float* w;   // [out × in]
    float* b;   // [out]
    int in, out;
};

static void cpu_forward(const float* input, float* output, int batch,
                        CPULinear* layers, int n_layers) {
    // Ping-pong buffers
    int max_dim = 128;
    float* buf_a = new float[batch * max_dim];
    float* buf_b = new float[batch * max_dim];
    memcpy(buf_a, input, batch * layers[0].in * sizeof(float));

    for (int L = 0; L < n_layers; ++L) {
        CPULinear& l = layers[L];
        float* src = (L == 0) ? buf_a : ((L % 2 == 0) ? buf_a : buf_b);
        float* dst = (L % 2 == 0) ? buf_b : buf_a;

        // matmul + bias
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < l.out; ++j) {
                float s = l.b[j];
                for (int k = 0; k < l.in; ++k)
                    s += src[i * l.in + k] * l.w[j * l.in + k]; // w is [out×in]
                dst[i * l.out + j] = s;
            }
        }

        // ReLU (except last layer)
        if (L < n_layers - 1) {
            int total = batch * l.out;
            for (int i = 0; i < total; ++i)
                dst[i] = dst[i] > 0.0f ? dst[i] : 0.0f;
        }
    }

    // Copy final result
    int final_dim = layers[n_layers - 1].out;
    float* final_buf = ((n_layers - 1) % 2 == 0) ? buf_b : buf_a;
    memcpy(output, final_buf, batch * final_dim * sizeof(float));

    delete[] buf_a; delete[] buf_b;
}

static void bench3_forward() {
    printf("############################################################\n");
    printf("#  BENCHMARK 3 — Full Forward Pass (DQN Network 4→128→64→2)#\n");
    printf("#  CPU  vs  GPU Basic  vs  GPU+Streams  vs  GPU+Graphs     #\n");
    printf("############################################################\n\n");

    int batches[] = {1, 32, 128, 512};
    int nbatch = 4;
    int input_dim = 4;
    int output_dim = 2;

    // Build GPU model
    Sequential* model = build_dqn_net();
    StreamEngine stream_eng(model);
    GraphEngine graph_eng;

    // Build CPU model with random weights (same dimensions)
    CPULinear cpu_layers[3];
    int dims_in[]  = {4, 128, 64};
    int dims_out[] = {128, 64, 2};
    for (int i = 0; i < 3; ++i) {
        cpu_layers[i].in  = dims_in[i];
        cpu_layers[i].out = dims_out[i];
        cpu_layers[i].w = new float[dims_in[i] * dims_out[i]];
        cpu_layers[i].b = new float[dims_out[i]];
        for (int j = 0; j < dims_in[i] * dims_out[i]; ++j)
            cpu_layers[i].w[j] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
        for (int j = 0; j < dims_out[i]; ++j)
            cpu_layers[i].b[j] = 0.0f;
    }

    printf("%-8s | %10s %12s | %10s %12s | %10s %12s | %10s %12s\n",
           "Batch", "CPU ms", "samples/s",
           "GPU ms", "samples/s",
           "Streams ms", "samples/s",
           "Graph ms", "samples/s");
    printf("---------+-----------------------"
           "+-----------------------"
           "+-----------------------"
           "+-----------------------\n");

    for (int bi = 0; bi < nbatch; ++bi) {
        int B = batches[bi];

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

        // Allocate test data
        float* h_in = new float[B * input_dim];
        float* h_out = new float[B * output_dim];
        for (int i = 0; i < B * input_dim; ++i) h_in[i] = 0.1f;

        Tensor test_input({B, input_dim});
        test_input.fill(0.1f);

        // ── CPU ──
        for (int w = 0; w < WARMUP; ++w) cpu_forward(h_in, h_out, B, cpu_layers, 3);
        auto st = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < FWD_ITERS; ++i) cpu_forward(h_in, h_out, B, cpu_layers, 3);
        auto en = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(en - st).count() / FWD_ITERS;
        double cpu_tput = B / (cpu_ms * 1e-3);

        // ── GPU basic ──
        for (int w = 0; w < WARMUP; ++w) {
            Tensor tmp({B, input_dim});
            CHECK_CUDA(cudaMemcpy(tmp.gpu_ptr(), test_input.gpu_ptr(),
                                  B * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            Tensor out = model->forward(tmp);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < FWD_ITERS; ++i) {
            Tensor tmp({B, input_dim});
            CHECK_CUDA(cudaMemcpy(tmp.gpu_ptr(), test_input.gpu_ptr(),
                                  B * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            Tensor out = model->forward(tmp);
        }
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float raw; cudaEventElapsedTime(&raw, t0, t1);
        double gpu_ms = raw / FWD_ITERS;
        double gpu_tput = B / (gpu_ms * 1e-3);

        // ── GPU + Streams ──
        for (int w = 0; w < WARMUP; ++w) {
            Tensor tmp({B, input_dim});
            CHECK_CUDA(cudaMemcpy(tmp.gpu_ptr(), test_input.gpu_ptr(),
                                  B * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            Tensor out = stream_eng.streamed_forward(tmp);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < FWD_ITERS; ++i) {
            Tensor tmp({B, input_dim});
            CHECK_CUDA(cudaMemcpy(tmp.gpu_ptr(), test_input.gpu_ptr(),
                                  B * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            Tensor out = stream_eng.streamed_forward(tmp);
        }
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&raw, t0, t1);
        double stream_ms = raw / FWD_ITERS;
        double stream_tput = B / (stream_ms * 1e-3);

        // ── GPU + Graph ──
        // Capture at this batch size
        {
            Tensor cap_in({B, input_dim});
            cap_in.fill(0.1f);
            graph_eng.capture_forward(*model, cap_in);
        }

        for (int w = 0; w < WARMUP; ++w) {
            Tensor tmp({B, input_dim});
            CHECK_CUDA(cudaMemcpy(tmp.gpu_ptr(), test_input.gpu_ptr(),
                                  B * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            Tensor out = graph_eng.execute_graph(tmp);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < FWD_ITERS; ++i) {
            Tensor tmp({B, input_dim});
            CHECK_CUDA(cudaMemcpy(tmp.gpu_ptr(), test_input.gpu_ptr(),
                                  B * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            Tensor out = graph_eng.execute_graph(tmp);
        }
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&raw, t0, t1);
        double graph_ms = raw / FWD_ITERS;
        double graph_tput = B / (graph_ms * 1e-3);

        // ── Print row ──
        printf("%-8d | %8.4f %10.0f   | %8.4f %10.0f   | %8.4f %10.0f   | %8.4f %10.0f\n",
               B, cpu_ms, cpu_tput, gpu_ms, gpu_tput,
               stream_ms, stream_tput, graph_ms, graph_tput);

        char bs[16]; snprintf(bs, sizeof(bs), "%d", B);
        csv_row("Forward", "CPU",        bs, cpu_ms,    cpu_tput,    "samples/s");
        csv_row("Forward", "GPU_Basic",  bs, gpu_ms,    gpu_tput,    "samples/s");
        csv_row("Forward", "GPU_Stream", bs, stream_ms, stream_tput, "samples/s");
        csv_row("Forward", "GPU_Graph",  bs, graph_ms,  graph_tput,  "samples/s");

        delete[] h_in; delete[] h_out;
        CHECK_CUDA(cudaEventDestroy(t0));
        CHECK_CUDA(cudaEventDestroy(t1));
    }

    // Cleanup CPU layers
    for (int i = 0; i < 3; ++i) { delete[] cpu_layers[i].w; delete[] cpu_layers[i].b; }
    delete model;
    printf("\n");
}

// ############################################################################
//
//  BENCHMARK 4 — RL Training Speed
//
// ############################################################################

// ---------- CPU CartPole ----------
struct CPUCartPole {
    float state[4]; // cart_pos, cart_vel, pole_angle, pole_vel
    bool done;

    void reset() {
        for (int i = 0; i < 4; ++i) state[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        done = false;
    }

    float step(int action) {
        float force = (action == 1) ? 10.0f : -10.0f;
        float cart_pos = state[0], cart_vel = state[1];
        float angle = state[2],    ang_vel = state[3];

        float cos_a = cosf(angle), sin_a = sinf(angle);
        float total_mass = 1.1f, pole_ml = 0.05f;

        float temp = (force + pole_ml * ang_vel * ang_vel * sin_a) / total_mass;
        float ang_acc = (9.8f * sin_a - cos_a * temp)
                      / (0.5f * (4.0f / 3.0f - 0.1f * cos_a * cos_a / total_mass));
        float cart_acc = temp - pole_ml * ang_acc * cos_a / total_mass;

        cart_vel += 0.02f * cart_acc;
        cart_pos += 0.02f * cart_vel;
        ang_vel  += 0.02f * ang_acc;
        angle    += 0.02f * ang_vel;

        state[0] = cart_pos; state[1] = cart_vel;
        state[2] = angle;    state[3] = ang_vel;

        done = (fabsf(angle) > 0.2094395f || fabsf(cart_pos) > 2.4f);
        return done ? 0.0f : 1.0f;
    }
};

static void bench4_rl() {
    printf("############################################################\n");
    printf("#  BENCHMARK 4 — RL Training Speed                         #\n");
    printf("#  Sequential CPU (1 env)  vs  Parallel GPU (512 envs)     #\n");
    printf("############################################################\n\n");

    int total_steps = RL_STEPS;

    // ── CPU: 1 environment, random actions, timed ──
    {
        CPUCartPole env;
        env.reset();
        int episodes = 0;
        int steps = 0;

        auto st = std::chrono::high_resolution_clock::now();
        while (steps < total_steps) {
            int action = rand() % 2;
            env.step(action);
            steps++;
            if (env.done) { env.reset(); episodes++; }
        }
        auto en = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(en - st).count();
        double cpu_steps_per_sec = total_steps / (cpu_ms * 1e-3);
        double cpu_eps_per_sec   = episodes / (cpu_ms * 1e-3);

        printf("  CPU Sequential (1 env):\n");
        printf("    Steps:          %d\n", total_steps);
        printf("    Episodes:       %d\n", episodes);
        printf("    Time:           %.2f ms\n", cpu_ms);
        printf("    Steps/sec:      %.0f\n", cpu_steps_per_sec);
        printf("    Episodes/sec:   %.0f\n", cpu_eps_per_sec);
        printf("\n");

        csv_row("RL", "CPU_1env", "steps_per_sec", cpu_ms, cpu_steps_per_sec, "steps/s");
        csv_row("RL", "CPU_1env", "episodes_per_sec", cpu_ms, cpu_eps_per_sec, "eps/s");
    }

    // ── GPU: 512 parallel environments ──
    {
        int n_envs = 512;
        ParallelEnv envs(n_envs);

        // Allocate random actions buffer on GPU
        int* d_actions;
        CHECK_CUDA(cudaMalloc(&d_actions, n_envs * sizeof(int)));

        // Copy host‑generated random actions (simple approach)
        int* h_actions = new int[n_envs];

        int gpu_total_steps = 0;
        int gpu_episodes = 0;

        // Need dones on host to count episodes
        int* h_dones = new int[n_envs];

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

        envs.reset();

        CHECK_CUDA(cudaEventRecord(t0));
        while (gpu_total_steps < total_steps) {
            // Generate random actions on host, copy to device
            for (int i = 0; i < n_envs; ++i) h_actions[i] = rand() % 2;
            CHECK_CUDA(cudaMemcpy(d_actions, h_actions, n_envs * sizeof(int),
                                  cudaMemcpyHostToDevice));

            envs.step(d_actions);
            gpu_total_steps += n_envs;

            // Count episodes (environments that terminated)
            CHECK_CUDA(cudaMemcpy(h_dones, envs.get_dones(), n_envs * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_envs; ++i) gpu_episodes += h_dones[i];
        }
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));

        float raw; cudaEventElapsedTime(&raw, t0, t1);
        double gpu_ms = raw;
        double gpu_steps_per_sec = gpu_total_steps / (gpu_ms * 1e-3);
        double gpu_eps_per_sec   = gpu_episodes / (gpu_ms * 1e-3);

        printf("  GPU Parallel (%d envs):\n", n_envs);
        printf("    Steps:          %d\n", gpu_total_steps);
        printf("    Episodes:       %d\n", gpu_episodes);
        printf("    Time:           %.2f ms\n", gpu_ms);
        printf("    Steps/sec:      %.0f\n", gpu_steps_per_sec);
        printf("    Episodes/sec:   %.0f\n", gpu_eps_per_sec);
        printf("\n");

        csv_row("RL", "GPU_512env", "steps_per_sec", gpu_ms, gpu_steps_per_sec, "steps/s");
        csv_row("RL", "GPU_512env", "episodes_per_sec", gpu_ms, gpu_eps_per_sec, "eps/s");

        // ── Comparison ──
        printf("  Comparison:\n");
        printf("    Env throughput speedup: %.1fx  (steps/sec)\n",
               gpu_steps_per_sec / (total_steps / ((double)gpu_ms * 1e-3 + 1e-9)));

        delete[] h_actions; delete[] h_dones;
        cudaFree(d_actions);
        CHECK_CUDA(cudaEventDestroy(t0));
        CHECK_CUDA(cudaEventDestroy(t1));
    }

    printf("\n");
}

// ############################################################################
//
//  MAIN — Run all benchmarks in sequence
//
// ############################################################################

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          KernelFlow — Complete Benchmark Suite          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    print_gpu_info();
    csv_open();

    bench1_gemm();
    bench2_activations();
    bench3_forward();
    bench4_rl();

    csv_close();

    printf("============================================================\n");
    printf("  All benchmarks complete.\n");
    printf("  Raw data saved to: benchmark/results.csv\n");
    printf("============================================================\n\n");

    return 0;
}
