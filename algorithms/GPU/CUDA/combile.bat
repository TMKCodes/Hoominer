@echo off
setlocal EnableDelayedExpansion

rem ===========================================================================
rem Configuration
rem ===========================================================================
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "OUTPUT_DIR=C:\Users\tonil\Repositories\Hoominer\releases\hoominer\cubins"
set "SRC=hoohash.cu"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

rem ===========================================================================
rem Initialize Visual Studio environment
rem ===========================================================================
echo Initializing Visual Studio 2022 x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL
if errorlevel 1 (
    echo *** ERROR: vcvars64.bat failed ***
    pause & exit /b 1
)

echo.
echo Building strict IEEE-754 FP64 cubins (sm_75 to sm_120)...
echo.

rem ===========================================================================
rem Build loop - 100% safe version (no carets inside variables!)
rem ===========================================================================
for %%A in (75 80 86 89 90 100 120) do (
    set "ARCH=sm_%%A"
    echo [%%A] Building !ARCH! ...

    "%CUDA_PATH%\bin\nvcc.exe" ^
        -cubin ^
        -arch=!ARCH! ^
        -O3 ^
        -lineinfo ^
        -fmad=false ^
        -prec-div=true ^
        -prec-sqrt=true ^
        -ftz=false ^
        -Xptxas --fmad=false ^
        -diag-suppress=68 ^
        -I"%CUDA_PATH%\include" ^
        "%SRC%" ^
        -o "%OUTPUT_DIR%\hoohash_!ARCH!.cubin"

    if errorlevel 1 (
        echo *** FAILED !ARCH! ***
        pause
        exit /b 1
    ) else (
        echo     Success: hoohash_!ARCH!.cubin created
        echo.
    )
)

echo ==================================================
echo All 7 cubins built successfully!
echo Strict IEEE-754 FP64 compliance on every architecture
echo ==================================================
echo.
endlocal
pause