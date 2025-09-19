@echo off

nvcc -cubin -arch=sm_50 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm50.cubin
nvcc -cubin -arch=sm_52 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm52.cubin
nvcc -cubin -arch=sm_60 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm60.cubin
nvcc -cubin -arch=sm_61 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm61.cubin
nvcc -cubin -arch=sm_70 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm70.cubin
nvcc -cubin -arch=sm_75 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm75.cubin
nvcc -cubin -arch=sm_80 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm80.cubin
nvcc -cubin -arch=sm_89 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm89.cubin
nvcc -cubin -arch=sm_90 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm90.cubin
nvcc -cubin -arch=sm_100 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm100.cubin
nvcc -cubin -arch=sm_120 -O3 -Xptxas -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" hoohash.cu -o C:\Users\tonil\Repositories\Hoominer\cubins\hoohash_sm120.cubin
