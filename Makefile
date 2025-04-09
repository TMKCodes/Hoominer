# Define the compiler and flags
CC = gcc
CFLAGS = -fPIC -g -Wall -Wextra -DTEST
LDFLAGS = -shared

# Build and output directories
BUILD_DIR = build

# Output library name (local, can be deprecated)
TARGET_LIB = $(BUILD_DIR)/lib-hoohash.so

# Miner target
MINER_SRC = miner.c
MINER_OBJ = $(BUILD_DIR)/miner.o
MINER_BIN = $(BUILD_DIR)/miner

# Default rule
all: hoohash $(MINER_BIN)

# Call into subdirectory to build hoohash
.PHONY: hoohash hoohash-clean
hoohash:
	$(MAKE) -C algorithms/hoohash

hoohash-clean:
	$(MAKE) -C algorithms/hoohash clean

# Create build directory if it does not exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile miner.c to object file
$(MINER_OBJ): $(MINER_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link the miner binary
$(MINER_BIN): $(MINER_OBJ)
	$(CC) -o $@ $(MINER_OBJ) \
		algorithms/hoohash/build/hoohash.o \
		algorithms/hoohash/build/bigint.o \
		-lm -lblake3 -lgmp -ljson-c

# Clean everything including hoohash
clean: hoohash-clean
	rm -rf $(BUILD_DIR)