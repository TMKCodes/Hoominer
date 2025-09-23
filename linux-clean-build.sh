#!/bin/bash

make clean && make
cd ./algorithms/GPU/CUDA
./compile.sh
cd ../../..
cp ./build/hoominer ./releases/hoominer/hoominer
chmod +x ./releases/hoominer/hoominer
rsync -avz --progress ./releases/hoominer/ tonto@192.168.122.1:/mnt/st3/tonto/Hoosat-Repositories/HTN/hoominer/releases/hoominer/