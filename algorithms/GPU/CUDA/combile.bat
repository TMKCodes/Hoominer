@echo off
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64
set OUTPUT_DIR=C:\Users\tonil\Repositories\Hoominer\releases\hoominer\cubins
mkdir "%OUTPUT_DIR%"
for %%a in (sm_50 sm_52 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120) do (
    nvcc -cubin -arch=%%a -fmad=false -prec-div=true -expt-relaxed-constexpr ^
         -ccbin "%MSVC_PATH%" -I"%CUDA_PATH%\include" hoohash.cu -o "%OUTPUT_DIR%\hoohash_%%a.cubin"
)