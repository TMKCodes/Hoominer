#!/bin/bash
# Generates Hoominer configuration file

# Source HiveOS rig and wallet configurations
[[ ! -e $RIG_CONF ]] && echo -e "${RED}No rig config $RIG_CONF${NOCOLOR}" && exit 1
[[ ! -e $WALLET_CONF ]] && echo -e "${RED}No wallet config $WALLET_CONF${NOCOLOR}" && exit 1
. $RIG_CONF
. $WALLET_CONF

# Define configuration file path
MINER_CONFIG="/hive/miners/custom/hoominer/hoominer.conf"
mkfile_from_symlink $MINER_CONFIG

# Generate configuration
conf="# Hoominer configuration
STRATUM_URL=\"$CUSTOM_URL\"
WAL=\"$WAL\"
PASS=\"x\"
GPU_IDS=\"$CUSTOM_USER_CONFIG\"
DISABLE_CPU=1
DISABLE_GPU=0
DISABLE_OPENCL=0
DISABLE_CUDA=0
CPU_THREADS=\"\"
EXTRA_CONFIG=\"\""

# Write configuration to file
echo -e "$conf" > $MINER_CONFIG