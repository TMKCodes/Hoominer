#!/bin/bash
# Runs Hoominer with configured parameters

. h-manifest.conf

# Source miner configuration
source /hive/miners/custom/hoominer/hoominer.conf

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME.log")" && touch "$CUSTOM_LOG_BASENAME.log"

# Build command-line arguments
CMD="./hoominer"
[ -n "$STRATUM_URL" ] && CMD="$CMD --stratum $STRATUM_URL"
[ -n "$WAL" ] && CMD="$CMD --user $WAL"
[ -n "$PASS" ] && CMD="$CMD --pass $PASS"
[ -n "$GPU_IDS" ] && CMD="$CMD --gpu-ids $GPU_IDS"
[ "$DISABLE_CPU" == "1" ] && CMD="$CMD --disable-cpu"
[ "$DISABLE_GPU" == "1" ] && CMD="$CMD --disable-gpu"
[ "$DISABLE_OPENCL" == "1" ] && CMD="$CMD --disable-opencl"
[ "$DISABLE_CUDA" == "1" ] && CMD="$CMD --disable-cuda"
[ -n "$CPU_THREADS" ] && CMD="$CMD --cpu-threads $CPU_THREADS"
[ -n "$EXTRA_CONFIG" ] && CMD="$CMD $EXTRA_CONFIG"

# Set working directory and execute
cd /hive/miners/custom/hoominer
exec $CMD >> $CUSTOM_LOG_BASENAME.log 2>&1