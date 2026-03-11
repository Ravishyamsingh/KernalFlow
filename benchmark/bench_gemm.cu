// ============================================================================
// KernelFlow — GEMM Benchmark
// ============================================================================
//
// Compares three matrix multiplication implementations:
//   1. CPU baseline    — triple nested loop
//   2. Your tiled GEMM — from kernels/gemm.cu (shared memory tiling)
//   3. cuBLAS SGEMM    — NVIDIA's production-optimized library
//
// Tests sizes: 256, 512, 1024, 2048
// Reports: time (ms), GFLOPS, and speedup vs CPU
//
// ============================================================================

#include "kernels.hpp"
#include <cublas_v2.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Number of warmup and timed iterations
#define WARMUP_ITERS 10
#define BENCH_ITERS  100

// ============================================================================
// CPU Baseline: Triple Nested Loop Matrix Multiply
// ============================================================================
// C[M×N] = A[M×K] * B[K×N], row-major
// This is the simplest O(M*N*K) implementation — no SIMD, no blocking.
// ============================================================================
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Benchmark helper: time CPU GEMM using std::chrono (averaged)
// ============================================================================
double bench_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        cpu_gemm(A, B, C, M, N, K);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        cpu_gemm(A, B, C, M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / BENCH_ITERS;
}

// ============================================================================
// Benchmark helper: time your tiled GEMM using cudaEvent_t (averaged)
// ============================================================================
double bench_tiled_gemm(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(total_ms) / BENCH_ITERS;
}

// ============================================================================
// Benchmark helper: time cuBLAS SGEMM using cudaEvent_t (averaged)
// ============================================================================
//
// cuBLAS uses COLUMN-MAJOR layout by default. To multiply row-major
// matrices without transposing, we use the identity:
//
//   C = A * B  (row-major)
//   is equivalent to
//   C^T = B^T * A^T  (column-major)
//
// Since a row-major matrix is the column-major transpose, we pass:
//   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//               N, M, K,           ← dimensions swapped
//               &alpha, d_B, N,    ← "A" in cuBLAS is our B
//               d_A, K,            ← "B" in cuBLAS is our A
//               &beta, d_C, N)     ← result written as C^T → row-major C
//
// ============================================================================
double bench_cublas(cublasHandle_t handle, float* d_A, float* d_B, float* d_C,
                    int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha, d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha, d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(total_ms) / BENCH_ITERS;
}

// ============================================================================
// GFLOPS calculation
// ============================================================================
// Matrix multiply does 2*M*N*K floating-point operations:
//   M*N*K multiplications + M*N*K additions = 2*M*N*K FLOPs
// ============================================================================
double compute_gflops(int M, int N, int K, double ms) {
    double flops = 2.0 * M * N * K;
    return flops / (ms * 1e6);  // ms → seconds: /1e3, then /1e9 for giga = /1e6
}

// ============================================================================
// Main — Run benchmarks across all sizes and print table
// ============================================================================
int main() {
    // Matrix sizes to test (square: M = N = K = size)
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("============================================================\n");
    printf("  KernelFlow GEMM Benchmark\n");
    printf("  GPU: %s\n", prop.name);
    printf("  Warmup: %d iterations | Timed: %d iterations\n",
           WARMUP_ITERS, BENCH_ITERS);
    printf("============================================================\n\n");

    // Table header
    printf("%-10s | %10s | %10s | %13s | %10s | %10s | %10s | %7s\n",
           "Size", "CPU (ms)", "CPU GFLOPS", "Tiled GPU(ms)",
           "GPU GFLOPS", "cuBLAS(ms)", "cBL GFLOPS", "Speedup");
    printf("-----------|------------|------------|---------------|");
    printf("------------|------------|------------|--------\n");

    for (int s = 0; s < num_sizes; ++s) {
        int M = sizes[s];
        int N = sizes[s];
        int K = sizes[s];
        int total = M * K;  // elements in A (and B, C are similar)

        // Allocate host memory and fill with random values
        float* h_A = new float[M * K];
        float* h_B = new float[K * N];
        float* h_C = new float[M * N];

        for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

        // Allocate device memory and copy data
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

        // --- Benchmark CPU (skip 2048 — too slow) ---
        double cpu_ms = 0.0;
        double cpu_gflops = 0.0;
        if (M <= 1024) {
            cpu_ms = bench_cpu(h_A, h_B, h_C, M, N, K);
            cpu_gflops = compute_gflops(M, N, K, cpu_ms);
        }

        // --- Benchmark your tiled GEMM ---
        double gpu_ms = bench_tiled_gemm(d_A, d_B, d_C, M, N, K);
        double gpu_gflops = compute_gflops(M, N, K, gpu_ms);

        // --- Benchmark cuBLAS ---
        double cublas_ms = bench_cublas(cublas_handle, d_A, d_B, d_C, M, N, K);
        double cublas_gflops = compute_gflops(M, N, K, cublas_ms);

        // --- Print row ---
        char size_str[32];
        snprintf(size_str, sizeof(size_str), "%dx%d", M, N);

        if (M <= 1024) {
            double speedup = cpu_ms / gpu_ms;
            printf("%-10s | %10.2f | %10.1f | %13.3f | %10.1f | %10.3f | %10.1f | %6.1fx\n",
                   size_str, cpu_ms, cpu_gflops, gpu_ms, gpu_gflops,
                   cublas_ms, cublas_gflops, speedup);
        } else {
            // CPU too slow for 2048+ — skip and show N/A
            printf("%-10s | %10s | %10s | %13.3f | %10.1f | %10.3f | %10.1f | %7s\n",
                   size_str, "N/A", "N/A", gpu_ms, gpu_gflops,
                   cublas_ms, cublas_gflops, "N/A");
        }

        // Cleanup
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    printf("\n");

    // Cleanup cuBLAS
    cublasDestroy(cublas_handle);

    printf("Benchmark complete.\n");
    return 0;
}