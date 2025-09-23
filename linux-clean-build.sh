#!/bin/bash

make clean && make
cd ./algorithms/GPU/CUDA
./compile.sh
cd ../../..
cp ./build/hoominer ./releases/hoominer/hoominer
cp -r ./build/cubins ./releases/hoominer/cubins
chmod +x /releases/hoominer/hoominer
