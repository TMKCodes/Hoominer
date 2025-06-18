# Compiler
CC = gcc

# Source files
SRC1 = opencl-compiler.c
SRC2 = test-blake3-kernel.c
SRC3 = test-hoohash-kernel.c

# Output binaries
BIN1 = opencl-compiler
BIN2 = test-blake3-kernel
BIN3 = test-hoohash-kernel

# Compiler flags
CFLAGS = -Wall -O2

# Platform-specific linker flags for OpenCL
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LDFLAGS = -lOpenCL
endif
ifeq ($(UNAME_S),Darwin)
    LDFLAGS = -framework OpenCL
endif
ifeq ($(OS),Windows_NT)
    LDFLAGS = -lOpenCL
    BIN1 := opencl-compiler.exe
    BIN2 := test-blake3-kernel.exe
    BIN3 := test-hoohash-kernel.exe
endif

# Default target builds both
all: $(BIN1) $(BIN2) $(BIN3)

# Build first binary
$(BIN1): $(SRC1)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Build second binary
$(BIN2): $(SRC2)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Build thid binary
$(BIN3): $(SRC3)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(BIN1) $(BIN2)
