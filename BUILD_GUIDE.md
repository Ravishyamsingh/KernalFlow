# KernelFlow - Build & Run Guide

## 🎯 Quick Start (Recommended)

### Build in Visual Studio 2026 IDE (Easiest Method)

1. **Open the project:**
   - File → Open → Folder
   - Select: `C:\Users\princ\source\repos\KernalFlow`

2. **Configure CMake:**
   - VS will automatically detect CMakeLists.txt
   - Wait for "CMake generation finished" in the Output window
   - If errors occur, see Troubleshooting section below

3. **Build:**
   - Build → Build All (or press `Ctrl+Shift+B`)
   - Wait for compilation to complete

4. **Run:**
   - Select executable from the toolbar dropdown (e.g., `bench_gemm.exe`)
   - Debug → Start Without Debugging (`Ctrl+F5`)

---

## 🔨 Build via Command Line

### Option 1: Using PowerShell Script (Automated)

```powershell
.\build.ps1
```

This script will:
- Check CUDA and CMake installations
- Clean build directory
- Configure CMake
- Build the project
- Show available executables

### Option 2: Manual CMake Build

```powershell
# Clean
Remove-Item -Path out\build\x64-Debug -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path out\build\x64-Debug -Force

# Configure
cmake -S . -B out\build\x64-Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build out\build\x64-Debug --config Debug
```

---

## 🚀 Running the Project

After successful build, you'll have these executables in `out\build\x64-Debug\`:

### 1. **GEMM Benchmark** (Recommended to start)
```powershell
.\out\build\x64-Debug\bench_gemm.exe
```
**What it does:** Compares your custom GEMM kernel vs cuBLAS on matrices of size 256, 512, 1024, 2048
**Output:** Timing (ms), GFLOPS, and speedup metrics

### 2. **Tensor Tests**
```powershell
.\out\build\x64-Debug\test_tensor.exe
```
**What it does:** Unit tests for tensor operations (allocation, transfer, math)

### 3. **Model Tests**
```powershell
.\out\build\x64-Debug\test_model.exe
```
**What it does:** Tests neural network layers and forward passes

### 4. **Inference Benchmark**
```powershell
.\out\build\x64-Debug\bench_inference.exe
```
**What it does:** Benchmarks model inference with CUDA streams and graphs

### 5. **DQN Training**
```powershell
.\out\build\x64-Debug\dqn_train.exe
```
**What it does:** Trains a Deep Q-Network agent (reinforcement learning)

### 6. **Main Application**
```powershell
.\out\build\x64-Debug\kernelflow_main.exe
```
**What it does:** Main entry point (check `src/` for implementation)

---

## ⚠️ Troubleshooting

### Issue 1: "unsupported Microsoft Visual Studio version"

**Cause:** CUDA 12.8 officially supports VS 2017-2022, but not VS 2026 yet.

**Solutions:**
1. **Recommended:** Install Visual Studio 2022 Community alongside VS 2026
   - Download: https://visualstudio.microsoft.com/vs/older-downloads/
   - Install "Desktop development with C++" workload
   - CUDA will automatically detect and use VS 2022

2. **Use the flag:** Already added to `CMakeLists.txt`:
   ```cmake
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
   ```

3. **Build through VS IDE:** Open the folder in VS 2026 and build directly (most reliable)

### Issue 2: "cudafe++ died with status 0xC0000005"

**Cause:** Access violation in CUDA compiler with VS 2026.

**Solution:** Use Visual Studio IDE to build instead of command line:
- File → Open → Folder → Select project
- Build → Build All

### Issue 3: "cuDNN not found"

**Status:** This is a WARNING, not an error. cuDNN is optional.

**If needed:**
```powershell
# Download cuDNN from NVIDIA (requires free account)
# Extract to: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\
```

### Issue 4: CMake configuration fails

**Solution:** Clean and reconfigure:
```powershell
Remove-Item -Path out -Recurse -Force
cmake -S . -B out\build\x64-Debug -G "Visual Studio 18 2026" -A x64
```

### Issue 5: "Ninja not found"

**Solution:** Use Visual Studio generator instead:
```powershell
cmake -S . -B out\build\x64-Debug -G "Visual Studio 18 2026" -A x64
cmake --build out\build\x64-Debug --config Debug
```

---

## 🔧 CMake Configuration Options

### Change CUDA Architecture (GPU Compute Capability)

Edit `CMakeLists.txt` line 18:

```cmake
# Current (supports RTX 20-40 series)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# For older GPUs (GTX 1060+):
set(CMAKE_CUDA_ARCHITECTURES 60 70 75)

# For latest GPUs only:
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090
```

### Build Release (Optimized) Instead of Debug

```powershell
cmake -S . -B out\build\x64-Release -DCMAKE_BUILD_TYPE=Release
cmake --build out\build\x64-Release --config Release
```

---

## 📊 Visualizing Results

After running benchmarks, you can visualize the results:

### Plot GEMM Performance
```powershell
python benchmark/plot_results.py
```

### Kernel Profiling
```powershell
python viz/kernel_profile.py
```

### Training Rewards (after DQN training)
```powershell
python viz/reward_plot.py
```

---

## 🎓 Understanding the Code

### Key Files to Study

1. **`kernels/gemm.cu`** (currently open)
   - Naive GEMM implementation
   - Tiled GEMM with shared memory
   - Great starting point to understand GPU optimization

2. **`include/tensor.hpp`**
   - Tensor data structure
   - GPU memory management

3. **`layers/linear.cu`**
   - Fully connected layer implementation
   - Uses your GEMM kernel

4. **`rl/dqn.cu`**
   - Deep Q-Network training loop
   - Experience replay buffer

### Architecture Flow

```
User Code
    ↓
Model API (model.hpp)
    ↓
Layers (layers/*.cu)
    ↓
Kernels (kernels/*.cu) ← Your GEMM code is here
    ↓
CUDA Runtime / cuBLAS
    ↓
GPU Hardware
```

---

## 🔍 Checking Your GPU

```powershell
# Check CUDA devices
nvidia-smi

# Get detailed GPU info
nvcc --version
```

---

## 📝 Next Steps

1. **Start simple:** Run `bench_gemm.exe` to see your GEMM kernel in action
2. **Read the code:** Study `kernels/gemm.cu` comments (very detailed!)
3. **Experiment:** Modify `TILE_SIZE` in `gemm.cu` and rebuild
4. **Profile:** Use Nsight Compute to profile your kernels
5. **Extend:** Add new kernels or layers

---

## 💡 Tips

- **Fast iteration:** Build only what changed:
  ```powershell
  cmake --build out\build\x64-Debug --target bench_gemm
  ```

- **Verbose build:** See actual compiler commands:
  ```powershell
  cmake --build out\build\x64-Debug --verbose
  ```

- **Clean build:** When things go wrong:
  ```powershell
  cmake --build out\build\x64-Debug --target clean
  cmake --build out\build\x64-Debug
  ```

---

## 📧 Common Issues Summary

| Symptom | Solution |
|---------|----------|
| VS 2026 not supported | Install VS 2022 or build via VS IDE |
| cuDNN warning | Ignore (optional dependency) |
| Access violation | Use VS IDE instead of command line |
| Slow build | Use Ninja generator: `-G Ninja` |
| Wrong GPU arch | Edit `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt |

---

**Happy GPU Computing! 🚀**

For issues, check: https://github.com/Ravishyamsingh/KernalFlow/issues
