# Compiler
NVCC = nvcc

# Flags
CFLAGS = -Xcompiler "-fPIC -g -O0 -Wall -Wextra -DTEST -DDEBUG"
INCLUDES = -Ialgorithms/blake3/c -I/opt/cuda/include -Iexternal/libmicrohttpd/build/include -Iexternal/json-c/ -I/external/libpciaccess/include/
NVCCFLAGS = $(CFLAGS) $(INCLUDES)
LDFLAGS = -lcudart_static -lm -lgmp -lOpenCL -L/opt/cuda/lib64 -lcuda -lcudart -lnvidia-ml -lssl -lcrypto
NVCCFLAGS = --compiler-options "-static-libgcc -static-libstdc++"
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

# Build static libs
.PHONY: hoohash hoohash-clean
hoohash:
	$(MAKE) -C algorithms/hoohash

hoohash-clean:
	$(MAKE) -C algorithms/hoohash clean

# .PHONY: json-c json-c-clean

# json-c:
# 	cd external/json-c && mkdir -p build && cd build && cmake .. -DBUILD_SHARED_LIBS=OFF  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make &&  cd ../../..
# json-c-clean:
# 	rm -rf external/json-c/build external/json-c/install

# .PHONY: microhttpd

# microhttpd:
#   cd external/libmicrohttpd && ./autogen.sh && ./configure --enable-static --disable-shared --prefix=/usr/local --disable-https && make && sudo make install && cd ../..

# .PHONY: pciaccess pciaccess-clean

# pciaccess:
# 	meson setup --reconfigure ./external/libpciaccess/build ./external/libpciaccess --default-library=static --buildtype=release
# 	meson compile -C ./external/libpciaccess/build
#   meson install -C ./external/libpciaccess/build

# pciaccess-clean:
# 	rm -rf external/libpciaccess/build external/libpciaccess/install

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
		/usr/local/lib/libmicrohttpd.a \
		external/json-c/build/libjson-c.a \
		/external/libpciaccess/install/lib/libpciaccess.a \
		$(LDFLAGS)
	chmod +x $@

# Clean everything
clean: hoohash-clean
	rm -rf $(BUILD_DIR)
