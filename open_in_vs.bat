@echo off
echo Opening KernelFlow in Visual Studio 2026...
echo.
echo Once VS opens:
echo 1. Wait for CMake to finish configuring (check Output window)
echo 2. Build ^> Build All (Ctrl+Shift+B)
echo 3. Run benchmark: Select "bench_gemm.exe" from dropdown, press Ctrl+F5
echo.
start "" "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\devenv.exe" "%~dp0"
