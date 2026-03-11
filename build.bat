@echo off
REM ============================================================================
REM KernelFlow Build Script for Visual Studio 2026 + CUDA 12.8
REM ============================================================================

echo.
echo ========================================
echo  KernelFlow CUDA Project Builder
echo ========================================
echo.

REM Clean build directory
if exist out\build\x64-Debug (
    echo Cleaning old build directory...
    rmdir /s /q out\build\x64-Debug 2>nul
)
mkdir out\build\x64-Debug

echo.
echo Configuring with CMake...
echo.

REM Configure with CMake using Ninja generator
cmake -S . -B out\build\x64-Debug ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" ^
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo.
    echo Possible solutions:
    echo 1. Install Visual Studio 2022 (recommended for CUDA 12.8)
    echo 2. Update CUDA Toolkit to a version that supports VS 2026
    echo 3. Try building in Visual Studio directly
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Building project...
echo ========================================
echo.

cmake --build out\build\x64-Debug --config Debug

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Build completed successfully!
echo ========================================
echo.
echo Executables are in: out\build\x64-Debug\
echo.
echo Available executables:
echo   - test_tensor.exe       (Tensor unit tests)
echo   - test_model.exe        (Model unit tests)
echo   - bench_gemm.exe        (GEMM benchmarking)
echo   - bench_inference.exe   (Inference benchmarking)
echo   - dqn_train.exe         (DQN training)
echo   - kernelflow_main.exe   (Main application)
echo.
pause
