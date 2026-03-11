// ============================================================================
// KernelFlow — GEMM (General Matrix Multiply) Kernels
// ============================================================================
//
// Computes: C = A * B
//   A is [M x K]
//   B is [K x N]
//   C is [M x N]
//
// All matrices are stored in row-major order:
//   element (row, col) of matrix with N columns => index = row * N + col
//
// Three implementations provided:
//   1. naive_gemm_kernel  — one thread per output element, global memory only
//   2. tiled_gemm_kernel  — shared memory tiling for memory bandwidth reuse
//   3. launch_gemm        — host wrapper that launches the tiled kernel
//
// ============================================================================

#include "tensor.hpp"

#define TILE_SIZE 32

// ============================================================================
// VERSION 1: Naive GEMM Kernel
// ============================================================================
//
// Algorithm:
//   Each thread computes exactly one element C[row][col] by iterating
//   over the shared K dimension and accumulating the dot product:
//
//     C[row][col] = sum_{k=0}^{K-1} A[row][k] * B[k][col]
//
// Performance problem:
//   Every thread reads K elements from A and K elements from B from
//   GLOBAL memory. For an M×N output, that's M*N*K global reads total.
//   Global memory bandwidth (~900 GB/s on modern GPUs) becomes the
//   bottleneck — the compute units sit idle waiting for data.
//
// ============================================================================
__global__ void naive_gemm_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// VERSION 2: Tiled GEMM Kernel (Shared Memory)
// ============================================================================
//
// KEY INSIGHT — Why tiling works:
//
//   In the naive version, if two threads in the same row compute C[row][0]
//   and C[row][1], they BOTH read the entire row A[row][0..K-1] from
//   global memory independently. That's redundant bandwidth waste.
//
//   Tiling fixes this by loading a TILE_SIZE × TILE_SIZE sub-block of A
//   and B into fast shared memory (~100x faster than global). Then all
//   threads in the block reuse that cached data for their partial sums.
//
// HOW IT WORKS:
//
//   The K dimension is split into tiles of width TILE_SIZE:
//
//   A [M×K]              B [K×N]              C [M×N]
//   ┌──┬──┬──┬──┐       ┌──┬──┬──┬──┐       ┌──┬──┬──┬──┐
//   │  │  │  │  │       │  │▓▓│  │  │       │  │▓▓│  │  │
//   │  │  │  │  │       │  │▓▓│  │  │       │  │▓▓│  │  │
//   ├──┼──┼──┼──┤   *   ├──┼──┼──┼──┤   =   ├──┼──┼──┼──┤
//   │▓▓│▓▓│▓▓│▓▓│       │  │▓▓│  │  │       │  │▓▓│  │  │
//   │▓▓│▓▓│▓▓│▓▓│       │  │▓▓│  │  │       │  │▓▓│  │  │
//   └──┴──┴──┴──┘       └──┴──┴──┴──┘       └──┴──┴──┴──┘
//    ▓ = data needed      ▓ = data needed     ▓ = output tile
//    (full row tiles)     (full col tiles)     (one tile)
//
//   For each tile index t (0, 1, ..., ceil(K/TILE_SIZE)-1):
//     1. Cooperatively load tile of A [TILE_SIZE × TILE_SIZE] into shared mem
//     2. Cooperatively load tile of B [TILE_SIZE × TILE_SIZE] into shared mem
//     3. __syncthreads() — make sure all loads are visible
//     4. Each thread accumulates TILE_SIZE partial products
//     5. __syncthreads() — make sure all reads are done before next load
//
//   Data reuse ratio: each element loaded from global memory is used by
//   TILE_SIZE threads instead of 1 → ~TILE_SIZE× bandwidth reduction.
//
// ============================================================================
__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    // Shared memory tiles — each block loads one tile of A and one tile of B
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // Global row/col this thread is responsible for in the output matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulator for the dot product (stays in register — fastest memory)
    float sum = 0.0f;

    // Number of tiles needed to cover the K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // ----------------------------------------------------------------
        // Step 1: Cooperatively load tile of A into shared memory
        //
        //   Thread (ty, tx) loads A[row][t*TILE_SIZE + tx]
        //   This is one element of the current A tile.
        //   Boundary check: row < M and column < K
        // ----------------------------------------------------------------
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;  // zero-pad out-of-bounds
        }

        // ----------------------------------------------------------------
        // Step 2: Cooperatively load tile of B into shared memory
        //
        //   Thread (ty, tx) loads B[t*TILE_SIZE + ty][col]
        //   This is one element of the current B tile.
        //   Boundary check: row index < K and col < N
        // ----------------------------------------------------------------
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;  // zero-pad out-of-bounds
        }

        // ----------------------------------------------------------------
        // Step 3: Synchronize — all threads must finish loading before
        //         any thread starts reading from shared memory
        // ----------------------------------------------------------------
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 4: Compute partial dot product for this tile
        //
        //   Each thread multiplies one row of tile_A with one column of
        //   tile_B, accumulating TILE_SIZE products into sum.
        //   This is where the shared memory reuse pays off — tile_A and
        //   tile_B are read ~TILE_SIZE times each but loaded only once.
        // ----------------------------------------------------------------
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // ----------------------------------------------------------------
        // Step 5: Synchronize — all threads must finish reading before
        //         the next iteration overwrites the shared memory tiles
        // ----------------------------------------------------------------
        __syncthreads();
    }

    // Write the final accumulated value to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// VERSION 3: Host Wrapper
// ============================================================================
//
// Launches tiled_gemm_kernel with the correct 2D grid and block dimensions.
//
// Parameters:
//   A — device pointer to matrix A [M × K], row-major
//   B — device pointer to matrix B [K × N], row-major
//   C — device pointer to output C [M × N], row-major
//   M, N, K — matrix dimensions
//
// ============================================================================
void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    // Block size: TILE_SIZE × TILE_SIZE threads (32×32 = 1024 threads max)
    dim3 block(TILE_SIZE, TILE_SIZE);

    // Grid size: enough blocks to cover the full M×N output matrix
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "GEMM launched: [" << M << "x" << K << "] * ["
              << K << "x" << N << "] = [" << M << "x" << N << "]" << std::endl;

    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}