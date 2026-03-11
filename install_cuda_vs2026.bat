@echo off
REM ============================================================================
REM Install CUDA Build Tools for Visual Studio 2026
REM Must be run as Administrator!
REM ============================================================================

echo.
echo Installing CUDA 12.8 build tools for Visual Studio 2026...
echo.

set SRC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\visual_studio_integration\MSBuildExtensions
set DST=C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v180\BuildCustomizations

if not exist "%SRC%\CUDA 12.8.props" (
    echo ERROR: CUDA files not found at:
    echo %SRC%
    echo.
    echo Please make sure CUDA 12.8 is installed properly.
    pause
    exit /b 1
)

if not exist "%DST%" (
    echo ERROR: Visual Studio folder not found at:
    echo %DST%
    echo.
    echo Please make sure Visual Studio 2026 is installed.
    pause
    exit /b 1
)

echo Copying CUDA build customization files...
echo From: %SRC%
echo To:   %DST%
echo.

copy /Y "%SRC%\CUDA 12.8.props" "%DST%\"
copy /Y "%SRC%\CUDA 12.8.targets" "%DST%\"
copy /Y "%SRC%\CUDA 12.8.xml" "%DST%\"
copy /Y "%SRC%\Nvda.Build.CudaTasks.v12.8.dll" "%DST%\"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! CUDA build tools installed for VS 2026.
    echo You can now build CUDA projects in Visual Studio.
    echo.
) else (
    echo.
    echo ERROR: Failed to copy files.
    echo Please run this script as Administrator!
    echo Right-click ^> Run as administrator
    echo.
)

pause
