# ============================================================================
# KernelFlow Quick Run Script
# ============================================================================
# Usage: .\run.ps1 [executable_name]
# Example: .\run.ps1 bench_gemm
# ============================================================================

param(
    [string]$Target = "bench_gemm"
)

$BuildDir = "out\build\x64-Debug"
$ExePath = Join-Path $BuildDir "$Target.exe"

# Available executables
$Executables = @{
    "bench_gemm" = "GEMM Benchmark (compares naive, tiled, cuBLAS)"
    "test_tensor" = "Tensor unit tests"
    "test_model" = "Model unit tests"
    "bench_inference" = "Inference benchmark"
    "dqn_train" = "DQN training"
    "kernelflow_main" = "Main application"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " KernelFlow Quick Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if executable exists
if (-not (Test-Path $ExePath)) {
    Write-Host "[ERROR] Executable not found: $ExePath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Available executables:" -ForegroundColor Yellow
    foreach ($key in $Executables.Keys | Sort-Object) {
        Write-Host "  $key" -NoNewline -ForegroundColor White
        Write-Host " - $($Executables[$key])" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "Usage: .\run.ps1 <executable_name>" -ForegroundColor Cyan
    Write-Host "Example: .\run.ps1 bench_gemm" -ForegroundColor Gray
    Write-Host ""
    
    # Check if project is built
    if (-not (Test-Path $BuildDir)) {
        Write-Host "[INFO] Project not built yet. Run: .\build.ps1" -ForegroundColor Yellow
    }
    
    exit 1
}

Write-Host "[INFO] Running: $Target" -ForegroundColor Green
if ($Executables.ContainsKey($Target)) {
    Write-Host "[INFO] Description: $($Executables[$Target])" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "----------------------------------------" -ForegroundColor DarkGray
Write-Host ""

# Run the executable
& $ExePath

Write-Host ""
Write-Host "----------------------------------------" -ForegroundColor DarkGray
Write-Host ""
Write-Host "[INFO] Execution completed" -ForegroundColor Green
Write-Host ""
