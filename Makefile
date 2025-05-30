# Define the compiler and flags
CC = gcc
CFLAGS = -fPIC -g -Wall -Wextra -DTEST
LDFLAGS =

# Build and output directories
BUILD_DIR = build

# Output static library
TARGET_LIB = $(BUILD_DIR)/lib-hoohash.a

# Miner target
MINER_SRC = hoominer.c
MINER_OBJ = $(BUILD_DIR)/hoominer.o
MINER_BIN = $(BUILD_DIR)/hoominer

# Default rule
all: hoohash $(MINER_BIN)

# Call into subdirectory to build hoohash static lib
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
	$(CC) $(CFLAGS) -Ialgorithms/blake3/c -c $< -o $@

# Link the miner binary against static lib-hoohash.a and blake3.a
$(MINER_BIN): $(MINER_OBJ) | $(BUILD_DIR)
	$(CC) -o $@ $(MINER_OBJ) \
		algorithms/hoohash/build/lib-hoohash.a \
		algorithms/blake3/c/build/libblake3.a \
		-lm -lgmp -ljson-c
	chmod +x $@

# Clean everything including hoohash
clean: hoohash-clean
	rm -rf $(BUILD_DIR)
