@echo off
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64
set OUTPUT_DIR=C:\Users\tonil\Repositories\Hoominer\releases\hoominer\cubins
mkdir "%OUTPUT_DIR%"
for %%a in (50 52 60 61 70 75 80 86 89 90 100 120) do (
    nvcc -cubin -arch=compute_%%a -code=sm_%%a -fmad=false ^
         -ccbin "%MSVC_PATH%" -I"%CUDA_PATH%\include" hoohash.cu -o "%OUTPUT_DIR%\hoohash_sm_%%a.cubin"
)