#!/bin/bash
mkdir -p ../../../releases/hoominer/cubins
for arch in sm_50 sm_52 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120; do
    nvcc -arch=$arch -cubin hoohash.cu -o ../../../releases/hoominer/cubins/hoohash_${arch}.cubin \
         -I/usr/local/cuda/include -I/usr/include \
         -fmad=false -prec-div=true -expt-relaxed-constexpr
done