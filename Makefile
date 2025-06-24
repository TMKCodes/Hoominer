# Compiler
NVCC = nvcc

# Flags
CFLAGS = -Xcompiler "-fPIC -g -O0 -Wall -Wextra -DTEST -DDEBUG"
INCLUDES = -Ialgorithms/blake3/c -I/opt/cuda/include
NVCCFLAGS = $(CFLAGS) $(INCLUDES)
LDFLAGS = -lm -lgmp -ljson-c -lOpenCL -L/opt/cuda/lib64 -lcuda -lcudart -lnvrtc

# Directories
SRC_DIR = src
BUILD_DIR = build

# Output binary
MINER_BIN = $(BUILD_DIR)/hoominer

# Source files
C_SRCS = $(wildcard $(SRC_DIR)/*.c)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)

ALL_SRCS = $(C_SRCS) $(CU_SRCS)

OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SRCS)) \
       $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.cu.o, $(CU_SRCS))

# Default rule
all: hoohash $(MINER_BIN)

# Build hoohash static lib
.PHONY: hoohash hoohash-clean
hoohash:
	$(MAKE) -C algorithms/hoohash

hoohash-clean:
	$(MAKE) -C algorithms/hoohash clean

# Ensure build dir exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile .c files with nvcc
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile .cu files with nvcc
$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link final binary
$(MINER_BIN): $(OBJS) | $(BUILD_DIR)
	$(NVCC) -o $@ $(OBJS) \
		algorithms/hoohash/build/lib-hoohash.a \
		algorithms/blake3/c/build/libblake3.a \
		$(LDFLAGS)
	chmod +x $@

# Clean everything
clean: hoohash-clean
	rm -rf $(BUILD_DIR)
