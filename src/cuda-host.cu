#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#include "cuda-host.h"

static uint64_t splitmix64(uint64_t *state)
{
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static cudaError_t create_xoshiro_random_state(CudaResources *resource)
{
  size_t total_bytes = resource->max_block_size * 4 * sizeof(unsigned long long);
  resource->h_random_state = (unsigned long long *)malloc(total_bytes);
  if (resource->h_random_state == NULL)
  {
    fprintf(stderr, "Random state allocation failed for %s\n", resource->device_name);
    return cudaErrorMemoryAllocation;
  }

  uint64_t seed_base[2];
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd >= 0)
  {
    if (read(fd, seed_base, sizeof(seed_base)) != sizeof(seed_base))
    {
      fprintf(stderr, "Warning: Failed to read full entropy from /dev/urandom\n");
      seed_base[0] = (uint64_t)time(NULL);
      seed_base[1] = (uint64_t)clock();
    }
    close(fd);
  }
  else
  {
    fprintf(stderr, "Warning: /dev/urandom unavailable, using fallback seed\n");
    seed_base[0] = (uint64_t)time(NULL);
    seed_base[1] = (uint64_t)clock();
  }

  for (size_t i = 0; i < resource->max_block_size; i++)
  {
    uint64_t state = seed_base[0] ^ (i * 0x9e3779b97f4a7c15ULL);
    state ^= seed_base[1];

    resource->h_random_state[i * 4 + 0] = splitmix64(&state);
    resource->h_random_state[i * 4 + 1] = splitmix64(&state);
    resource->h_random_state[i * 4 + 2] = splitmix64(&state);
    resource->h_random_state[i * 4 + 3] = splitmix64(&state);

    if (resource->h_random_state[i * 4 + 0] == 0 && resource->h_random_state[i * 4 + 1] == 0 &&
        resource->h_random_state[i * 4 + 2] == 0 && resource->h_random_state[i * 4 + 3] == 0)
    {
      resource->h_random_state[i * 4 + 0] = 0x9e3779b97f4a7c15ULL;
    }
  }

  return cudaSuccess;
}

CudaResources *initialize_selected_cuda_gpus(unsigned int *device_indices, unsigned int num_selected, unsigned int *device_count)
{
  cudaError_t err = cudaSuccess;
  int num_devices;

  *device_count = 0;
  if (!device_indices || num_selected == 0)
  {
    fprintf(stderr, "Invalid input: No device indices\n");
    return NULL;
  }

  err = cudaGetDeviceCount(&num_devices);
  if (err != cudaSuccess || num_devices == 0)
  {
    fprintf(stderr, "No CUDA devices: %s\n", cudaGetErrorString(err));
    return NULL;
  }

  for (unsigned int i = 0; i < num_selected; i++)
  {
    if (device_indices[i] >= (unsigned int)num_devices)
    {
      fprintf(stderr, "Invalid device index %u: Only %u devices\n", device_indices[i], num_devices);
      return NULL;
    }
  }

  CudaResources *res = (CudaResources *)calloc(num_selected, sizeof(CudaResources));
  if (!res)
  {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (unsigned int i = 0; i < num_selected; i++)
  {
    unsigned int idx = device_indices[i];
    err = cudaSetDevice(idx);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Set device %u failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }

    err = cudaGetDeviceProperties(&res[i].device_prop, idx);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u properties query failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }
    strncpy(res[i].device_name, res[i].device_prop.name, sizeof(res[i].device_name) - 1);
    res[i].device_id = idx; // Set device_id

    if (res[i].device_prop.computeMode == cudaComputeModeProhibited || res[i].device_prop.major < 2)
    {
      fprintf(stderr, "Device %u (%s) lacks sufficient compute capability\n", idx, res[i].device_name);
      goto cleanup;
    }

    err = cudaStreamCreate(&res[i].stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u stream creation failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }

    res[i].max_grid_size = res[i].device_prop.maxGridSize[0];
    res[i].max_block_size = res[i].device_prop.maxThreadsPerBlock;
    printf("Max grid size: %ld\n", res[i].max_grid_size);
    printf("Max block size: %ld\n", res[i].max_block_size);

    // Allocate device memory with individual error checks
    err = cudaMalloc(&res[i].previous_header, DOMAIN_HASH_SIZE);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u previous_header allocation failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].timestamp, sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u timestamp allocation failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].matrix, 64 * 64 * sizeof(double));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u matrix allocation failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].target, DOMAIN_HASH_SIZE);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u target allocation failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].result, sizeof(CudaResult));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u result allocation failed: %s\n", idx, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc((void **)&res[i].random_state, res[i].max_block_size * 4 * sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "cudaMalloc failed for random state: %s\n", cudaGetErrorString(err));
      goto cleanup;
    }

    err = create_xoshiro_random_state(&res[i]);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Random state initialization failed for %s: %s\n", res[i].device_name, cudaGetErrorString(err));
      goto cleanup;
    }

    printf("Initialized GPU %u: %s\n", idx, res[i].device_name);
  }

  *device_count = num_selected;
  return res;

cleanup:
  for (unsigned int j = 0; j < num_selected; j++)
  {
    cudaFree(res[j].previous_header);
    cudaFree(res[j].timestamp);
    cudaFree(res[j].matrix);
    cudaFree(res[j].target);
    cudaFree(res[j].result);
    cudaFree(res[j].random_state);
    free(res[j].h_random_state);
    cudaStreamDestroy(res[j].stream);
  }
  free(res);
  return NULL;
}

CudaResources *initialize_all_cuda_gpus(unsigned int *device_count)
{
  cudaError_t err = cudaSuccess;
  int num_devices;

  *device_count = 0;
  err = cudaGetDeviceCount(&num_devices);
  if (err != cudaSuccess || num_devices == 0)
  {
    fprintf(stderr, "No CUDA devices: %s\n", cudaGetErrorString(err));
    return NULL;
  }

  CudaResources *res = (CudaResources *)calloc(num_devices, sizeof(CudaResources));
  if (!res)
  {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (unsigned int i = 0; i < (unsigned int)num_devices; i++)
  {
    err = cudaSetDevice(i);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Set device %u failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }

    err = cudaGetDeviceProperties(&res[i].device_prop, i);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u properties query failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }
    strncpy(res[i].device_name, res[i].device_prop.name, sizeof(res[i].device_name) - 1);
    res[i].device_id = i; // Set device_id

    if (res[i].device_prop.computeMode == cudaComputeModeProhibited || res[i].device_prop.major < 2)
    {
      fprintf(stderr, "Device %u (%s) lacks sufficient compute capability\n", i, res[i].device_name);
      goto cleanup;
    }

    err = cudaStreamCreate(&res[i].stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u stream creation failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }

    res[i].max_grid_size = res[i].device_prop.maxGridSize[0];
    res[i].max_block_size = res[i].device_prop.maxThreadsPerBlock;
    printf("Max grid size: %ld\n", res[i].max_grid_size);
    printf("Max block size: %ld\n", res[i].max_block_size);

    // Allocate device memory with individual error checks
    err = cudaMalloc(&res[i].previous_header, DOMAIN_HASH_SIZE);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u previous_header allocation failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].timestamp, sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u timestamp allocation failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].matrix, 64 * 64 * sizeof(double));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u matrix allocation failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].target, DOMAIN_HASH_SIZE);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u target allocation failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].result, sizeof(CudaResult));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u result allocation failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].random_state, res[i].max_block_size * 4 * sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u random_state allocation failed: %s\n", i, cudaGetErrorString(err));
      goto cleanup;
    }

    err = create_xoshiro_random_state(&res[i]);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Random state initialization failed for %s: %s\n", res[i].device_name, cudaGetErrorString(err));
      goto cleanup;
    }

    printf("Initialized GPU %u: %s\n", i, res[i].device_name);
  }

  *device_count = num_devices;
  return res;

cleanup:
  for (unsigned int j = 0; j < (unsigned int)num_devices; j++)
  {
    cudaFree(res[j].previous_header);
    cudaFree(res[j].timestamp);
    cudaFree(res[j].matrix);
    cudaFree(res[j].target);
    cudaFree(res[j].result);
    cudaFree(res[j].random_state);
    free(res[j].h_random_state);
    cudaStreamDestroy(res[j].stream);
  }
  free(res);
  return NULL;
}

cudaError_t compile_cuda_kernel_from_xxd_header(CudaResources *resource, const char *source, size_t source_len, const char *kernel_name, const char **required_extensions, size_t num_required_extensions)
{
  cudaError_t err;
  CUresult cu_err;
  nvrtcProgram prog;
  char *ptx = NULL;
  size_t ptx_size;

  if (!source || source_len == 0 || !kernel_name)
  {
    fprintf(stderr, "Invalid source or kernel name for %s\n", resource->device_name);
    return cudaErrorInvalidValue;
  }

  err = cudaSetDevice(resource->device_id);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Set device failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  // Check required extensions
  for (size_t i = 0; i < num_required_extensions; i++)
  {
    if (strcmp(required_extensions[i], "double_precision") == 0 && resource->device_prop.major < 2)
    {
      fprintf(stderr, "Device %s does not support double precision\n", resource->device_name);
      return cudaErrorInvalidDevice;
    }
  }

  // Create NVRTC program
  nvrtcResult nvrtc_err = nvrtcCreateProgram(&prog, (const char *)source, "hoohash.cu", 0, NULL, NULL);
  if (nvrtc_err != NVRTC_SUCCESS)
  {
    fprintf(stderr, "Failed to create NVRTC program for %s: %s\n", resource->device_name, nvrtcGetErrorString(nvrtc_err));
    return cudaErrorInvalidSource;
  }

  // Set compilation options based on device compute capability
  char arch_option[32];
  snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%d%d", resource->device_prop.major, resource->device_prop.minor);
  const char *cuda_include_path = getenv("CUDA_INCLUDE_PATH") ?: "/usr/local/cuda/include";
  const char *std_include_path = getenv("STD_INCLUDE_PATH") ?: "/usr/include";
  char cuda_include_option[256];
  char std_include_option[256];
  snprintf(cuda_include_option, sizeof(cuda_include_option), "-I%s", cuda_include_path);
  snprintf(std_include_option, sizeof(std_include_option), "-I%s", std_include_path);
  const char *opts[] = {
      arch_option,
      "--use_fast_math",
      "--std=c++17",
      cuda_include_option,
      std_include_option};

  // Compile the program
  nvrtc_err = nvrtcCompileProgram(prog, 4, opts); // Update to 4 options
  if (nvrtc_err != NVRTC_SUCCESS)
  {
    fprintf(stderr, "NVRTC compilation failed for %s: %s\n", resource->device_name, nvrtcGetErrorString(nvrtc_err));
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    char *log = (char *)malloc(log_size);
    if (log)
    {
      nvrtcGetProgramLog(prog, log);
      fprintf(stderr, "Compilation log:\n%s\n", log);
      free(log);
    }
    nvrtcDestroyProgram(&prog);
    return cudaErrorInvalidSource;
  }

  // Get PTX
  nvrtc_err = nvrtcGetPTXSize(prog, &ptx_size);
  if (nvrtc_err != NVRTC_SUCCESS)
  {
    fprintf(stderr, "Failed to get PTX size for %s: %s\n", resource->device_name, nvrtcGetErrorString(nvrtc_err));
    nvrtcDestroyProgram(&prog);
    return cudaErrorInvalidPtx;
  }

  ptx = (char *)malloc(ptx_size);
  if (!ptx)
  {
    fprintf(stderr, "Memory allocation failed for PTX\n");
    nvrtcDestroyProgram(&prog);
    return cudaErrorMemoryAllocation;
  }

  nvrtc_err = nvrtcGetPTX(prog, ptx);
  if (nvrtc_err != NVRTC_SUCCESS)
  {
    fprintf(stderr, "Failed to get PTX for %s: %s\n", resource->device_name, nvrtcGetErrorString(nvrtc_err));
    free(ptx);
    nvrtcDestroyProgram(&prog);
    return cudaErrorInvalidPtx;
  }

  nvrtcDestroyProgram(&prog);

  // Load PTX into CUDA module
  cu_err = cuModuleLoadData(&resource->module, ptx);
  free(ptx);
  if (cu_err != CUDA_SUCCESS)
  {
    const char *err_str;
    cuGetErrorString(cu_err, &err_str);
    fprintf(stderr, "Module load failed for %s: %s\n", resource->device_name, err_str);
    return cudaErrorInvalidPtx;
  }

  cu_err = cuModuleGetFunction(&resource->kernel, resource->module, kernel_name);
  if (cu_err != CUDA_SUCCESS)
  {
    const char *err_str;
    cuGetErrorString(cu_err, &err_str);
    fprintf(stderr, "Kernel %s creation failed for %s: %s\n", kernel_name, resource->device_name, err_str);
    cuModuleUnload(resource->module);
    return cudaErrorInvalidKernelImage;
  }

  printf("Kernel %s compiled for %s\n", kernel_name, resource->device_name);
  return cudaSuccess;
}

cudaError_t load_cuda_kernel_binary(CudaResources *resource, const char *cubin_filename, const char *kernel_name)
{
  cudaError_t err;
  CUresult cu_err;

  FILE *file = fopen(cubin_filename, "rb");
  if (!file)
  {
    fprintf(stderr, "Failed to open %s: %s\n", cubin_filename, strerror(errno));
    return cudaErrorInvalidValue;
  }

  fseek(file, 0, SEEK_END);
  size_t binary_size = ftell(file);
  rewind(file);
  char *binary = (char *)malloc(binary_size);
  if (!binary || fread(binary, 1, binary_size, file) != binary_size)
  {
    fprintf(stderr, "Failed to read %s\n", cubin_filename);
    free(binary);
    fclose(file);
    return cudaErrorInvalidValue;
  }
  fclose(file);

  err = cudaSetDevice(resource->device_id);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Set device failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    free(binary);
    return err;
  }

  cu_err = cuModuleLoadData(&resource->module, binary);
  free(binary);
  if (cu_err != CUDA_SUCCESS)
  {
    const char *err_str;
    cuGetErrorString(cu_err, &err_str);
    fprintf(stderr, "Module load failed for %s: %s\n", resource->device_name, err_str);
    return cudaErrorInvalidPtx;
  }

  cu_err = cuModuleGetFunction(&resource->kernel, resource->module, kernel_name);
  if (cu_err != CUDA_SUCCESS)
  {
    const char *err_str;
    cuGetErrorString(cu_err, &err_str);
    fprintf(stderr, "Kernel %s creation failed for %s: %s\n", kernel_name, resource->device_name, err_str);
    cuModuleUnload(resource->module);
    return cudaErrorInvalidKernelImage;
  }

  printf("Kernel %s loaded for %s\n", kernel_name, resource->device_name);
  return cudaSuccess;
}

cudaError_t run_cuda_hoohash_kernel(CudaResources *resource, unsigned char *previous_header, unsigned char *target, double matrix[64][64],
                                    unsigned long timestamp, unsigned long nonce_mask, unsigned long nonce_fixed, CudaResult *result)
{
  cudaError_t err;

  if (!resource || !previous_header || !target || !matrix || !result)
  {
    fprintf(stderr, "Invalid input pointers for %s\n", resource ? resource->device_name : "unknown");
    return cudaErrorInvalidValue;
  }

  err = cudaSetDevice(resource->device_id);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Set device failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  // Async data transfers
  err = cudaMemcpyAsync(resource->previous_header, previous_header, DOMAIN_HASH_SIZE, cudaMemcpyHostToDevice, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Memory copy to previous_header failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  err = cudaMemcpyAsync(resource->timestamp, &timestamp, sizeof(unsigned long), cudaMemcpyHostToDevice, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Memory copy to timestamp failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  err = cudaMemcpyAsync(resource->matrix, matrix, 64 * 64 * sizeof(double), cudaMemcpyHostToDevice, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Memory copy to matrix failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  err = cudaMemcpyAsync(resource->target, target, DOMAIN_HASH_SIZE, cudaMemcpyHostToDevice, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Memory copy to target failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  err = cudaMemcpyAsync(resource->random_state, resource->h_random_state,
                        resource->max_block_size * 4 * sizeof(unsigned long long),
                        cudaMemcpyHostToDevice, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Memory copy to random_state failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  CudaResult init_result = {0};
  err = cudaMemcpyAsync(resource->result, &init_result, sizeof(CudaResult), cudaMemcpyHostToDevice, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Result buffer initialization failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  unsigned long random_type = RANDOM_TYPE_LEAN;
  void *args[] = {
      &nonce_mask,
      &nonce_fixed,
      &resource->previous_header,
      &resource->timestamp,
      &resource->matrix,
      &resource->target,
      &random_type,
      &resource->random_state,
      &resource->result};

  CUresult cu_err = cuLaunchKernel(resource->kernel,
                                   resource->max_grid_size, 1, 1,
                                   resource->max_block_size, 1, 1,
                                   0,
                                   resource->stream,
                                   args,
                                   NULL);

  // Check kernel launch error
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Kernel launch failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  // Copy result back to host
  err = cudaMemcpyAsync(result, resource->result, sizeof(CudaResult), cudaMemcpyDeviceToHost, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Result copy failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  // Wait for all operations to complete
  err = cudaStreamSynchronize(resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Stream synchronization failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  return cudaSuccess;
}

void cleanup_cuda_resources(CudaResources *resource)
{
  if (!resource)
    return;

  cudaSetDevice(resource->device_id);
  free(resource->h_random_state);
  cudaFree(resource->previous_header);
  cudaFree(resource->timestamp);
  cudaFree(resource->matrix);
  cudaFree(resource->target);
  cudaFree(resource->result);
  cudaFree(resource->random_state);
  cuModuleUnload(resource->module);
  cudaStreamDestroy(resource->stream);
}

void cleanup_all_cuda_gpus(CudaResources *resources, unsigned int device_count)
{
  if (!resources)
    return;
  for (unsigned int i = 0; i < device_count; i++)
  {
    cleanup_cuda_resources(&resources[i]);
  }
  free(resources);
}