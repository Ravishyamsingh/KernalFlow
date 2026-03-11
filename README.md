<p align="center">
  <strong>KernelFlow</strong><br>
  <em>A GPU-Accelerated Deep Learning Inference Engine with Reinforcement Learning</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900?style=flat-square&logo=nvidia" alt="CUDA 12">
  <img src="https://img.shields.io/badge/C++-17-00599C?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/CMake-3.18+-064F8C?style=flat-square&logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/cuBLAS-integrated-76b900?style=flat-square" alt="cuBLAS">
</p>

---

KernelFlow is a from-scratch GPU inference engine written in C++17 and CUDA.
It implements custom GEMM kernels, neural network layers, CUDA Streams,
CUDA Graphs, and a fully GPU-parallel DQN agent — all without using
high-level frameworks like PyTorch or TensorFlow.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        KernelFlow Architecture                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐   │
│   │  DQN Agent   │   │  Benchmark   │   │   Visualization       │   │
│   │  (rl/)       │   │  Suite       │   │   (Python)            │   │
│   └──────┬───────┘   └──────┬───────┘   └───────────────────────┘   │
│          │                  │                                        │
│   ┌──────┴──────────────────┴───────────────────────────────┐       │
│   │              Execution Engine (src/)                     │       │
│   │    ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │       │
│   │    │   CUDA      │  │    CUDA      │  │  Sequential  │  │       │
│   │    │   Streams   │  │    Graphs    │  │  Model       │  │       │
│   │    │   (4-way)   │  │  (captured)  │  │  Forward     │  │       │
│   │    └─────────────┘  └──────────────┘  └─────────────┘  │       │
│   └──────────────────────────┬──────────────────────────────┘       │
│                              │                                       │
│   ┌──────────────────────────┴──────────────────────────────┐       │
│   │                  Layer System (layers/)                   │       │
│   │    ┌──────────┐  ┌──────────┐  ┌──────────┐            │       │
│   │    │  Linear  │  │   ReLU   │  │ Softmax  │            │       │
│   │    │ (tiled)  │  │(in-place)│  │(per-row) │            │       │
│   │    └──────────┘  └──────────┘  └──────────┘            │       │
│   └──────────────────────────┬──────────────────────────────┘       │
│                              │                                       │
│   ┌──────────────────────────┴──────────────────────────────┐       │
│   │                  CUDA Kernels (kernels/)                  │       │
│   │  ┌────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │       │
│   │  │ Naive  │ │   Tiled    │ │Activations │ │Reduction │ │       │
│   │  │ GEMM   │ │   GEMM     │ │ReLU/Sig/   │ │Sum/Max/  │ │       │
│   │  │        │ │ (shmem 32) │ │Tanh/Smax   │ │Mean      │ │       │
│   │  └────────┘ └────────────┘ └────────────┘ └──────────┘ │       │
│   └──────────────────────────┬──────────────────────────────┘       │
│                              │                                       │
│   ┌──────────────────────────┴──────────────────────────────┐       │
│   │           Tensor Runtime (include/tensor.hpp)             │       │
│   │         GPU Memory Management · Host ↔ Device Transfer    │       │
│   └──────────────────────────────────────────────────────────┘       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component         | Technology                     | Purpose                              |
|-------------------|--------------------------------|--------------------------------------|
| **Language**      | C++17                          | Core engine & layer system           |
| **GPU**           | CUDA 12.x                      | Custom kernels, memory management    |
| **Math**          | cuBLAS                         | Reference GEMM for benchmarking      |
| **Random**        | cuRAND                         | Xavier initialization, ε-greedy      |
| **Build**         | CMake 3.18+                    | Cross-platform CUDA compilation      |
| **Visualization** | Python (matplotlib, pandas)    | Benchmark dashboards, reward plots   |
| **Architecture**  | SM 6.0 – 8.6                   | Pascal through Ampere GPUs           |

---

## Prerequisites

| Requirement       | Minimum Version | Notes                                |
|-------------------|-----------------|--------------------------------------|
| NVIDIA GPU        | SM 6.0+         | Pascal (GTX 1060) or newer           |
| CUDA Toolkit      | 12.0            | Includes nvcc, cuBLAS, cuRAND        |
| cuDNN             | 8.x             | Linked but reserved for extensions   |
| CMake             | 3.18            | Required for CUDA language support   |
| C++ Compiler      | GCC 9+ / MSVC 19.20+ | C++17 support required          |
| Python 3          | 3.8+            | Only for visualization scripts       |
| Python packages   | matplotlib, pandas, numpy | `pip install matplotlib pandas numpy` |

---

## Build Instructions

```bash
# Clone and build
cd KernelFlow
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

All executables are placed in the `build/` directory.

---

## How to Run

### DQN Training

```bash
./dqn_train
```

Trains a DQN agent on 512 parallel CartPole environments using GPU-accelerated
inference with CUDA Streams. Produces `rewards.csv` for plotting.

### GEMM Benchmark

```bash
./bench_gemm
```

Compares CPU, naive GPU, tiled GPU (shared memory), and cuBLAS GEMM at
matrix sizes 128–2048. Reports time (ms) and throughput (GFLOPS).

### Full Benchmark Suite

```bash
./bench_inference
```

Runs all four benchmarks (GEMM, Activations, Forward Pass, RL Training) and
saves raw data to `benchmark/results.csv`.

### Unit Tests

```bash
./test_tensor          # Tensor allocation, transfer, fill
./test_model           # Sequential model forward pass
```

### Visualization

```bash
# Reward curve (after training)
python viz/reward_plot.py --input rewards.csv

# Benchmark dashboard (after bench_inference)
python benchmark/plot_results.py --input benchmark/results.csv
```

Outputs: `reward_curve.png` and `benchmark_dashboard.png`.

---

## Expected Benchmark Results

Results will vary by GPU. Representative numbers on an RTX 3080:

### GEMM Performance (GFLOPS)

| Size      | Naive GPU | Tiled GPU | cuBLAS  |
|-----------|-----------|-----------|---------|
| 256×256   | ~50       | ~180      | ~800    |
| 512×512   | ~80       | ~350      | ~2500   |
| 1024×1024 | ~100      | ~500      | ~5000   |
| 2048×2048 | —         | ~600      | ~8000   |

### Forward Pass Latency (ms) — DQN Network (4→128→64→2)

| Batch | CPU     | GPU Basic | GPU+Streams | GPU+Graphs |
|-------|---------|-----------|-------------|------------|
| 1     | ~0.003  | ~0.040    | ~0.035      | ~0.010     |
| 32    | ~0.060  | ~0.045    | ~0.038      | ~0.012     |
| 128   | ~0.230  | ~0.050    | ~0.042      | ~0.015     |
| 512   | ~0.900  | ~0.065    | ~0.050      | ~0.020     |

> **Key insight**: CUDA Graphs eliminate kernel launch overhead, delivering
> the largest speedup at small batch sizes where launch cost dominates compute.

### RL Training Throughput

| Method                | Steps/sec   |
|-----------------------|-------------|
| CPU Sequential (1 env)| ~50,000     |
| GPU Parallel (512 env)| ~5,000,000+ |

---

## Project Structure

```
KernelFlow/
├── include/                    # Header files
│   ├── tensor.hpp              #   Tensor class + CHECK_CUDA macro
│   ├── kernels.hpp             #   GEMM, activation, reduction declarations
│   ├── layers.hpp              #   LinearLayer (tiled GEMM + bias)
│   ├── model.hpp               #   Sequential model, ReLU/Softmax layers
│   ├── stream_engine.hpp       #   StreamEngine (4 streams) + GraphEngine
│   ├── environment.hpp         #   ParallelEnv (512 CartPole envs)
│   ├── replay_buffer.hpp       #   GPU replay buffer
│   └── agent.hpp               #   Agent interface
│
├── src/                        # Core implementations
│   ├── tensor.cpp              #   Tensor memory management
│   ├── model.cu                #   Sequential forward pass
│   ├── stream_engine.cu        #   Multi-stream & graph execution
│   └── agent.cpp               #   Agent entry point
│
├── kernels/                    # CUDA kernel implementations
│   ├── gemm.cu                 #   Naive + tiled GEMM (TILE_SIZE=32)
│   ├── activations.cu          #   ReLU, Sigmoid, Tanh, Softmax kernels
│   ├── reduction.cu            #   Sum, Max, Mean parallel reductions
│   ├── conv2d.cu               #   (reserved)
│   └── batchnorm.cu            #   (reserved)
│
├── layers/                     # Layer forward-pass implementations
│   ├── linear.cu               #   Tiled GEMM with transposed weights
│   ├── conv2d.cu               #   (reserved)
│   └── batchnorm.cu            #   (reserved)
│
├── rl/                         # Reinforcement learning
│   ├── dqn.cu                  #   DQN agent: ε-greedy, Huber loss, target net
│   ├── environment.cu          #   GPU CartPole (semi-implicit Euler)
│   └── replay_buffer.cu        #   GPU replay buffer with batch sampling
│
├── benchmark/                  # Performance benchmarks
│   ├── bench_gemm.cu           #   GEMM: CPU vs Tiled vs cuBLAS
│   ├── bench_inference.cu      #   Full 4-benchmark suite → results.csv
│   └── plot_results.py         #   Dashboard visualization
│
├── tests/                      # Unit tests
│   ├── test_tensor.cu          #   Tensor allocation & transfer tests
│   └── test_model.cu           #   Model forward-pass tests
│
├── viz/                        # Visualization
│   ├── reward_plot.py          #   Training reward + epsilon curves
│   └── kernel_profile.py       #   (reserved)
│
├── CMakeLists.txt              # Build configuration
├── overview.md                 # Project overview document
└── README.md                   # This file
```

---

## What This Project Demonstrates

### GPU & Parallel Programming
- Custom CUDA kernels for GEMM, activations, and reductions
- Shared memory tiling (32×32) for memory-coalesced matrix multiplication
- Thread-block cooperative Softmax with shared-memory reduction
- Parallel environment simulation (512 CartPole instances)

### CUDA Optimization Techniques
- **CUDA Streams**: 4-stream concurrent kernel execution for pipelined inference
- **CUDA Graphs**: Captured execution graphs that eliminate per-launch overhead
- **Memory coalescing**: Row-major tiled GEMM with bank-conflict-free shared memory access
- **cuBLAS integration**: B^T × A^T trick for row-major matrices in column-major cuBLAS

### Deep Learning Systems
- Layer abstraction with virtual dispatch (`Layer` → `LinearLayer`, `ReLU`, `Softmax`)
- Adapter pattern (`LinearLayerAdapter`) for composing heterogeneous layer types
- Xavier weight initialization via cuRAND
- Sequential model container with forward-pass chaining

### Reinforcement Learning
- Deep Q-Network (DQN) with experience replay and target network
- Epsilon-greedy exploration with exponential decay
- Huber loss for stable gradient updates
- Fully GPU-resident training pipeline (no host↔device round-trips in the critical path)

### Software Engineering
- Modular architecture: kernels → layers → model → agent
- CMake build system with separable CUDA compilation
- Benchmark suite with CSV export and automated visualization
- Professional data visualization with matplotlib

---

## References

1. Mnih, V. et al. *"Human-level control through deep reinforcement learning."*
   Nature 518, 529–533 (2015).
   [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)

2. NVIDIA. *CUDA C++ Programming Guide.*
   [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

3. Kirk, D.B. & Hwu, W.W. *Programming Massively Parallel Processors:
   A Hands-on Approach.* 4th Edition, Morgan Kaufmann (2022).

4. NVIDIA. *cuBLAS Library Documentation.*
   [docs.nvidia.com/cuda/cublas](https://docs.nvidia.com/cuda/cublas/)

5. NVIDIA. *CUDA Graphs Documentation.*
   [docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

---

<sub>Built as a college project demonstrating GPU programming, parallel computing, and reinforcement learning.</sub>
