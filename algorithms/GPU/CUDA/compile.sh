#!/bin/bash

# Compile hoohash.cu for supported architectures with proper include paths
# Output to the same directory as the executable
mkdir ../../../build/cubins
nvcc -arch=sm_50 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm50.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_52 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm52.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_60 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm60.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_61 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm61.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_70 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm70.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_75 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm75.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_80 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm80.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_86 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm86.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_89 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm89.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_90 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm90.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr
nvcc -arch=sm_100 -cubin hoohash.cu -o ../../../build/cubins/hoohash_sm100.cubin -I/usr/local/cuda/include -I/usr/include -fmad=false -prec-div=true -expt-relaxed-constexpr