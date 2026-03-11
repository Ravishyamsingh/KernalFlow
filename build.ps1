# ============================================================================
# KernelFlow Build Script (PowerShell) for Visual Studio 2026 + CUDA 12.8
# ============================================================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " KernelFlow CUDA Project Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check CUDA installation
Write-Host "[INFO] Checking CUDA installation..." -ForegroundColor Yellow
try {
    $cudaVersion = & nvcc --version 2>&1 | Select-String "release"
    Write-Host "[OK] $cudaVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] CUDA Toolkit not found! Please install CUDA 12.x" -ForegroundColor Red
    exit 1
}

# Check CMake
Write-Host "[INFO] Checking CMake installation..." -ForegroundColor Yellow
try {
    $cmakeVersion = & cmake --version 2>&1 | Select-Object -First 1
    Write-Host "[OK] $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] CMake not found! Please install CMake 3.18+" -ForegroundColor Red
    exit 1
}

# Clean build directory
if (Test-Path "out\build\x64-Debug") {
    Write-Host "[INFO] Cleaning old build directory..." -ForegroundColor Yellow
    Remove-Item -Path "out\build\x64-Debug" -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path "out\build\x64-Debug" -Force | Out-Null

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Configuring with CMake..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Try to configure - first attempt with Developer Command Prompt
$env:Path = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;" + $env:Path

$configSuccess = $false

# Attempt 1: Use Ninja generator (fastest)
Write-Host "[ATTEMPT 1] Configuring with Ninja generator..." -ForegroundColor Yellow
$result = cmake -S . -B out\build\x64-Debug `
    -G "Ninja" `
    -DCMAKE_BUILD_TYPE=Debug `
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" `
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89" 2>&1

if ($LASTEXITCODE -eq 0) {
    $configSuccess = $true
    Write-Host "[OK] Configuration successful with Ninja!" -ForegroundColor Green
}

# Attempt 2: If Ninja fails, try Visual Studio generator
if (-not $configSuccess) {
    Write-Host ""
    Write-Host "[ATTEMPT 2] Trying Visual Studio generator..." -ForegroundColor Yellow
    
    Remove-Item -Path "out\build\x64-Debug" -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path "out\build\x64-Debug" -Force | Out-Null
    
    $result = cmake -S . -B out\build\x64-Debug `
        -G "Visual Studio 18 2026" `
        -A x64 `
        -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" `
        -DCMAKE_CUDA_ARCHITECTURES="75;86;89" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        $configSuccess = $true
        Write-Host "[OK] Configuration successful with VS generator!" -ForegroundColor Green
    }
}

if (-not $configSuccess) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " Configuration Failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "CUDA 12.8 does not officially support Visual Studio 2026." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Recommended solutions:" -ForegroundColor Cyan
    Write-Host "1. Install Visual Studio 2022 Community (free)" -ForegroundColor White
    Write-Host "   Download: https://visualstudio.microsoft.com/vs/older-downloads/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Wait for CUDA 12.9+ which may support VS 2026" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Build using Visual Studio IDE directly:" -ForegroundColor White
    Write-Host "   - Open folder in VS 2026" -ForegroundColor Gray
    Write-Host "   - Let it configure CMake automatically" -ForegroundColor Gray
    Write-Host "   - Build > Build All" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Building project..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cmake --build out\build\x64-Debug --config Debug

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Build completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Executables are in: out\build\x64-Debug\" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available executables:" -ForegroundColor Yellow
Write-Host "  test_tensor.exe       " -NoNewline -ForegroundColor White; Write-Host "(Tensor unit tests)" -ForegroundColor Gray
Write-Host "  test_model.exe        " -NoNewline -ForegroundColor White; Write-Host "(Model unit tests)" -ForegroundColor Gray
Write-Host "  bench_gemm.exe        " -NoNewline -ForegroundColor White; Write-Host "(GEMM benchmarking)" -ForegroundColor Gray
Write-Host "  bench_inference.exe   " -NoNewline -ForegroundColor White; Write-Host "(Inference benchmarking)" -ForegroundColor Gray
Write-Host "  dqn_train.exe         " -NoNewline -ForegroundColor White; Write-Host "(DQN training)" -ForegroundColor Gray
Write-Host "  kernelflow_main.exe   " -NoNewline -ForegroundColor White; Write-Host "(Main application)" -ForegroundColor Gray
Write-Host ""
Write-Host "To run GEMM benchmark: " -NoNewline -ForegroundColor Cyan
Write-Host ".\out\build\x64-Debug\bench_gemm.exe" -ForegroundColor White
Write-Host ""
