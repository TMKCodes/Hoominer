#ifndef CUDA_H
#define CUDA_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef _WIN32
#include <dlfcn.h> // For dynamic loading on Linux
#include <fcntl.h>
#include <unistd.h>
#else
#include <windows.h>
#endif
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

// Constants
#define DOMAIN_HASH_SIZE 32
#define RANDOM_TYPE_XOSHIRO 1
#define RANDOM_TYPE_LEAN 0
#define PRINTF_BUFFER_SIZE (1024 * 1024) // 1MB printf buffer

  // Forward declarations
  typedef struct CudaResult CudaResult;
  typedef struct CudaResources CudaResources;
  typedef struct cudaDeviceProp cudaDeviceProp;

  // Struct holding results from CUDA kernel
  struct CudaResult
  {
    uint64_t nonce;
    unsigned char hash[32];
  };

  // Struct holding CUDA resources and device state
  struct CudaResources
  {
    cudaDeviceProp device_prop;
    cudaStream_t stream;
    CUmodule module;
    CUfunction kernel;
    unsigned char *previous_header;
    int64_t *timestamp;
    double *matrix;
    unsigned char *target;
    char *printf_buffer;       // Device printf buffer
    size_t optimal_block_size; // Optimal threads per block
    size_t optimal_grid_size;  // Optimal blocks in grid
    int device_id;             // Store device ID for cudaSetDevice
    char device_name[256];
    unsigned int pci_bus_id;
    int usegpu;
    CudaResult *result;
    uint64_t *nonces_processed;
  };

  // Function declarations
  CudaResources *initialize_all_cuda_gpus(unsigned int *device_count, unsigned int selected_gpus[256], int selected_gpus_num);
  bool load_cuda_kernel_binary(CudaResources *resource, const char *cubin_filename, const char *kernel_name, int work_multiplier);
  void cleanup_cuda_resources(CudaResources *resource);
  void cleanup_all_cuda_gpus(CudaResources *resources, unsigned int device_count);
  cudaError_t run_cuda_hoohash_kernel(CudaResources *resource, unsigned char *previous_header, unsigned char *target, double matrix[64][64],
                                      int64_t timestamp, uint64_t start_nonce, CudaResult *result);
  cudaError_t retrieve_kernel_printf(CudaResources *resource);

#ifdef __cplusplus
}
#endif

#endif // CUDA_H