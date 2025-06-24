#ifndef CUDA_H
#define CUDA_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

// Constants
#define DOMAIN_HASH_SIZE 32
#define RANDOM_TYPE_XOSHIRO 1
#define RANDOM_TYPE_LEAN 0
#define MAX_ITERATIONS 18446744073709551615UL

  // Forward declarations
  typedef struct CudaResult CudaResult;
  typedef struct CudaResources CudaResources;
  typedef struct cudaDeviceProp cudaDeviceProp;

  // Struct holding results from CUDA kernel
  struct CudaResult
  {
    unsigned long long nonce;
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
    unsigned long *timestamp;
    double *matrix;
    unsigned char *target;
    unsigned long long *random_state;
    CudaResult *result;
    unsigned long long *h_random_state; // Host-side random state
    size_t max_grid_size;
    size_t max_block_size;
    int device_id; // Store device ID for cudaSetDevice
    char device_name[256];
  };

  // Function declarations
  CudaResources *initialize_selected_cuda_gpus(unsigned int *device_indices, unsigned int num_selected, unsigned int *device_count);
  CudaResources *initialize_all_cuda_gpus(unsigned int *device_count);
  cudaError_t compile_cuda_kernel_from_xxd_header(CudaResources *resource, const char *source, size_t source_len, const char *kernel_name, const char **required_extensions, size_t num_required_extensions);
  cudaError_t load_cuda_kernel_binary(CudaResources *resource, const char *cubin_filename, const char *kernel_name);
  void cleanup_cuda_resources(CudaResources *resource);
  void cleanup_all_cuda_gpus(CudaResources *resources, unsigned int device_count);
  cudaError_t run_cuda_hoohash_kernel(CudaResources *resource, unsigned char *previous_header, unsigned char *target, double matrix[64][64],
                                      unsigned long timestamp, unsigned long nonce_mask, unsigned long nonce_fixed, CudaResult *result);

#ifdef __cplusplus
}
#endif

#endif // CUDA_H