#!/bin/bash
mkdir -p ../../../releases/hoominer/cubins
for arch in 50 52 60 61 70 75 80 86 89 90 100 120; do
    nvcc -arch=compute_$arch -code=sm_$arch  -cubin hoohash.cu -o ../../../releases/hoominer/cubins/hoohash_sm_${arch}.cubin \
         -I/usr/local/cuda/include -I/usr/include \
         -fmad=false -prec-div=true -expt-relaxed-constexpr
done

# -ccbin=/usr/bin/g++13 