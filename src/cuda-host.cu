#include "cuda-host.h"

// Function to calculate optimal grid and block dimensions
static void calculate_optimal_dimensions(CudaResources *resource)
{
  const int warp_size = 32;
  int sm_count = resource->device_prop.multiProcessorCount;
  int max_threads_per_block = resource->device_prop.maxThreadsPerBlock;
  int max_threads_per_sm = resource->device_prop.maxThreadsPerMultiProcessor;

  // Start with a typical block size
  int threads_per_block = 256;
  int max_active_blocks_per_sm = 0;

  // Estimate optimal blocks per SM
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_per_sm, resource->kernel, threads_per_block, 0);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Occupancy calc failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    max_active_blocks_per_sm = 2;
  }

  // Choose target blocks per SM
  int target_blocks_per_sm = max_active_blocks_per_sm >= 2 ? max_active_blocks_per_sm : 2;

  // Adjust threads per block to stay within per-SM thread limit
  int max_possible_threads = max_threads_per_sm / target_blocks_per_sm;
  threads_per_block = (threads_per_block > max_possible_threads) ? max_possible_threads : threads_per_block;
  threads_per_block = (threads_per_block / warp_size) * warp_size;

  // Clamp to device limits
  if (threads_per_block > max_threads_per_block)
    threads_per_block = max_threads_per_block - (max_threads_per_block % warp_size);
  if (threads_per_block < warp_size)
    threads_per_block = warp_size;

  // Compute total grid size
  int grid_size = sm_count * target_blocks_per_sm;
  if (grid_size > resource->device_prop.maxGridSize[0])
    grid_size = resource->device_prop.maxGridSize[0];

  resource->optimal_block_size = threads_per_block;
  resource->optimal_grid_size = grid_size;

  printf("Calculated for %s: block_size=%d, grid_size=%d\n",
         resource->device_name, threads_per_block, grid_size);
}

int compare_pci_bus_id(const void *a, const void *b)
{
  const CudaResources *ra = (const CudaResources *)a;
  const CudaResources *rb = (const CudaResources *)b;
  return (int)(ra->pci_bus_id - rb->pci_bus_id);
}

static uint64_t splitmix64(uint64_t *state)
{
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static cudaError_t create_xoshiro_random_state(CudaResources *resource)
{
  size_t total_bytes = 4 * sizeof(unsigned long long);
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

  uint64_t state = seed_base[0] ^ seed_base[1];
  resource->h_random_state[0] = splitmix64(&state);
  resource->h_random_state[1] = splitmix64(&state);
  resource->h_random_state[2] = splitmix64(&state);
  resource->h_random_state[3] = splitmix64(&state);

  if (resource->h_random_state[0] == 0 && resource->h_random_state[1] == 0 &&
      resource->h_random_state[2] == 0 && resource->h_random_state[3] == 0)
  {
    resource->h_random_state[0] = 0x9e3779b97f4a7c15ULL;
    resource->h_random_state[1] = 0x9e3779b97f4a7c15ULL;
    resource->h_random_state[2] = 0x9e3779b97f4a7c15ULL;
    resource->h_random_state[3] = 0x9e3779b97f4a7c15ULL;
  }

  return cudaSuccess;
}

CudaResources *initialize_all_cuda_gpus(unsigned int *device_count, unsigned int selected_gpus[256], int selected_gpus_num)
{
  cudaError_t err = cudaSuccess;
  int num_devices, devices_found;

  *device_count = 0;
  devices_found = 0;
  err = cudaGetDeviceCount(&num_devices);
  if (err != cudaSuccess || num_devices == 0)
  {
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
    res[i].device_id = i;
    res[i].pci_bus_id = res[i].device_prop.pciBusID; // Store PCI-BUS-ID

    if (res[i].device_prop.computeMode == cudaComputeModeProhibited || res[i].device_prop.major < 2)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u, %s) lacks sufficient compute capability\n",
              i, res[i].pci_bus_id, res[i].device_name);
      goto cleanup;
    }
    if (selected_gpus_num > 0)
    {
      int found = 0;
      for (int x = 0; x < selected_gpus_num; x++)
      {
        if (res[i].pci_bus_id == (unsigned int)selected_gpus[x])
        {
          found = 1;
          devices_found++;
          printf("Using device %d", res[i].pci_bus_id);
          break;
        }
      }
      if (found == 0)
      {
        printf("Skipped using device %d", res[i].pci_bus_id);
        continue; // skip the GPU since it was not specified.
      }
    }

    err = cudaStreamCreate(&res[i].stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) stream creation failed: %s\n",
              i, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }

    err = cudaMalloc(&res[i].previous_header, DOMAIN_HASH_SIZE);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) previous_header allocation failed: %s\n",
              i, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].timestamp, sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) timestamp allocation failed: %s\n",
              i, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].matrix, 64 * 64 * sizeof(double));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) matrix allocation failed: %s\n",
              i, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].target, DOMAIN_HASH_SIZE);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) target allocation failed: %s\n",
              i, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].result, sizeof(CudaResult));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) result allocation failed: %s\n",
              i, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }
    err = cudaMalloc(&res[i].random_state, 4 * sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) random_state allocation failed: %s\n",
              i, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }

    err = create_xoshiro_random_state(&res[i]);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Random state initialization failed for %s (PCI-BUS-ID: %u): %s\n",
              res[i].device_name, res[i].pci_bus_id, cudaGetErrorString(err));
      goto cleanup;
    }
  }
  qsort(res, devices_found, sizeof(CudaResources), compare_pci_bus_id);
  *device_count = devices_found;
  return res;

cleanup:
  for (unsigned int j = 0; j < (unsigned int)devices_found; j++)
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

  // Calculate optimal grid and block dimensions after kernel is loaded
  calculate_optimal_dimensions(resource);

  printf("Kernel %s loaded for %s\n", kernel_name, resource->device_name);
  return cudaSuccess;
}

// cudaError_t retrieve_kernel_printf(CudaResources *resource)
// {
//   cudaError_t err;
//   char *h_printf_buffer = (char *)malloc(PRINTF_BUFFER_SIZE);
//   if (!h_printf_buffer)
//   {
//     fprintf(stderr, "Host printf buffer allocation failed for %s\n", resource->device_name);
//     return cudaErrorMemoryAllocation;
//   }

//   err = cudaMemcpyAsync(h_printf_buffer, resource->printf_buffer, PRINTF_BUFFER_SIZE, cudaMemcpyDeviceToHost, resource->stream);
//   if (err != cudaSuccess)
//   {
//     fprintf(stderr, "Printf buffer copy failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
//     free(h_printf_buffer);
//     return err;
//   }

//   err = cudaStreamSynchronize(resource->stream);
//   if (err != cudaSuccess)
//   {
//     fprintf(stderr, "Stream synchronization failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
//     free(h_printf_buffer);
//     return err;
//   }

//   h_printf_buffer[PRINTF_BUFFER_SIZE - 1] = '\0';
//   if (strlen(h_printf_buffer) > 0)
//   {
//     printf("Kernel printf output from %s:\n%s\n", resource->device_name, h_printf_buffer);
//   }

//   free(h_printf_buffer);
//   return cudaSuccess;
// }

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

  // err = cudaMemsetAsync(resource->printf_buffer, 0, PRINTF_BUFFER_SIZE, resource->stream);
  // if (err != cudaSuccess)
  // {
  //   fprintf(stderr, "Printf buffer reset failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
  //   return err;
  // }

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
                        4 * sizeof(unsigned long long),
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
                                   resource->optimal_grid_size, 1, 1,
                                   resource->optimal_block_size, 1, 1,
                                   0,
                                   resource->stream,
                                   args,
                                   NULL);
  if (cu_err != CUDA_SUCCESS)
  {
    const char *err_str;
    cuGetErrorString(cu_err, &err_str);
    fprintf(stderr, "Kernel launch failed for %s: %s\n", resource->device_name, err_str);
    return cudaErrorLaunchFailure;
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Kernel launch error for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  err = cudaMemcpyAsync(result, resource->result, sizeof(CudaResult), cudaMemcpyDeviceToHost, resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Result copy failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  err = cudaStreamSynchronize(resource->stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Stream synchronization failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
    return err;
  }

  // err = retrieve_kernel_printf(resource);
  // if (err != cudaSuccess)
  // {
  //   fprintf(stderr, "Failed to retrieve kernel printf for %s: %s\n", resource->device_name, cudaGetErrorString(err));
  //   return err;
  // }

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
  // cudaFree(resource->printf_buffer);
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