## blake3

# cmake -S c -B c/build "-DCMAKE_INSTALL_PREFIX=/usr/local"
# sudo cmake --build c/build --target install

# .PHONY: json-c json-c-clean

# json-c:
# 	cd external/json-c && mkdir -p build && cd build && cmake .. -DBUILD_SHARED_LIBS=OFF  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make && sudo make install && cd ../../..
# json-c-clean:
# 	rm -rf external/json-c/build external/json-c/install

# .PHONY: microhttpd

# microhttpd:
#   cd external/libmicrohttpd && ./autogen.sh && ./configure --enable-static --disable-shared --prefix=/usr/local --disable-https && make && sudo make install && cd ../..

# .PHONY: pciaccess pciaccess-clean

# pciaccess:
# 	meson setup --reconfigure ./external/libpciaccess/build ./external/libpciaccess --default-library=static --buildtype=release --prefix=/usr/local
# 	meson compile -C ./external/libpciaccess/build
#   meson install -C ./external/libpciaccess/build

# pciaccess-clean:
# 	rm -rf external/libpciaccess/build external/libpciaccess/install

# Compiler
NVCC = nvcc

# Flags
# Set STATIC=0 for a dynamically-linked build (useful for Valgrind/ASan and systems
# that don't have static archives for all deps).
STATIC ?= 1

# If STATIC=1 is requested but required static archives are missing, automatically
# fall back to a dynamic link to keep builds working on typical distros.
STATIC_EFFECTIVE := $(STATIC)

SSL_STATIC_CANDIDATES = /usr/lib/x86_64-linux-gnu/libssl.a /usr/lib/libssl.a /usr/local/lib/libssl.a
CRYPTO_STATIC_CANDIDATES = /usr/lib/x86_64-linux-gnu/libcrypto.a /usr/lib/libcrypto.a /usr/local/lib/libcrypto.a
GMP_STATIC_CANDIDATES = /usr/lib/x86_64-linux-gnu/libgmp.a /usr/lib/libgmp.a /usr/local/lib/libgmp.a
PCIACCESS_STATIC_CANDIDATES = /usr/local/lib/x86_64-linux-gnu/libpciaccess.a /usr/local/lib/libpciaccess.a /usr/lib/x86_64-linux-gnu/libpciaccess.a /usr/lib/libpciaccess.a

SSL_STATIC = $(firstword $(wildcard $(SSL_STATIC_CANDIDATES)))
CRYPTO_STATIC = $(firstword $(wildcard $(CRYPTO_STATIC_CANDIDATES)))
GMP_STATIC = $(firstword $(wildcard $(GMP_STATIC_CANDIDATES)))
PCIACCESS_STATIC = $(firstword $(wildcard $(PCIACCESS_STATIC_CANDIDATES)))

ifeq ($(STATIC),1)
ifeq ($(SSL_STATIC),)
STATIC_EFFECTIVE := 0
endif
ifeq ($(CRYPTO_STATIC),)
STATIC_EFFECTIVE := 0
endif
ifeq ($(GMP_STATIC),)
STATIC_EFFECTIVE := 0
endif
ifeq ($(PCIACCESS_STATIC),)
STATIC_EFFECTIVE := 0
endif
ifeq ($(STATIC_EFFECTIVE),0)
$(warning STATIC=1 requested but one or more static archives are missing; falling back to STATIC=0. Install libssl-dev/libgmp-dev and static pciaccess to enable full static builds.)
endif
endif

CFLAGS = -Xcompiler "-fPIC -g -O0 -Wall -Wextra -fcommon"
INCLUDES = -Ialgorithms/blake3/c -I/opt/cuda/include -I/usr/local/include -I/usr/include
NVCCFLAGS = $(CFLAGS) $(INCLUDES)
LDFLAGS = -lcudart_static -lm -lOpenCL

ifeq ($(STATIC),1)
	# Keep this block for backwards compatibility, but gate actual static linking
	# on STATIC_EFFECTIVE below.
else
PCIACCESS_LIB = -lpciaccess
endif

ifeq ($(STATIC_EFFECTIVE),1)
CFLAGS += -Xcompiler "-static-libgcc -static-libstdc++"
NVCCFLAGS += --linker-options "-static"
SSL_LIB = $(SSL_STATIC)
CRYPTO_LIB = $(CRYPTO_STATIC)
GMP_LIB = $(GMP_STATIC)
PCIACCESS_LIB = $(PCIACCESS_STATIC)
else
SSL_LIB = -lssl
CRYPTO_LIB = -lcrypto
GMP_LIB = -lgmp
PCIACCESS_LIB = -lpciaccess
endif

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
		/usr/local/lib/libblake3.a \
		/usr/local/lib/libmicrohttpd.a \
		/usr/local/lib/libjson-c.a \
		$(SSL_LIB) \
		$(CRYPTO_LIB) \
		$(GMP_LIB) \
		$(PCIACCESS_LIB) \
		$(LDFLAGS)
	chmod +x $@

# Clean everything
clean: hoohash-clean
	rm -rf $(BUILD_DIR)