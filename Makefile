# Define the compiler and flags
CC = gcc
CFLAGS = -fPIC -g -O0 -Wall -Wextra -DTEST -DDEBUG
LDFLAGS = -lOpenCL

# Build and output directories
BUILD_DIR = build

# Output static library
TARGET_LIB = $(BUILD_DIR)/lib-hoohash.a

# Find all .c source files under .src/
SRC_DIR = src/
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# Miner binary target
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

# Compile all source files from .src/ to build/*.o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Ialgorithms/blake3/c -c $< -o $@

# Link the miner binary against static libs and all objects
$(MINER_BIN): $(OBJS) | $(BUILD_DIR)
	$(CC) -o $@ $(OBJS) \
		algorithms/hoohash/build/lib-hoohash.a \
		algorithms/blake3/c/build/libblake3.a \
		-lm -lgmp -ljson-c -lOpenCL
	chmod +x $@

# Clean everything including hoohash
clean: hoohash-clean
	rm -rf $(BUILD_DIR)

