#!/bin/bash

make clean && make
cd ./algorithms/GPU/CUDA
./compile.sh
cd ../../..
cp ./build/hoominer ./releases/hoominer/hoominer
chmod +x ./releases/hoominer/hoominer
