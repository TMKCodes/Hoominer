

#include "opencl-host.h"
#include "platform_compat.h"

cl_int calculate_work_sizes(StratumContext *ctx, OpenCLResources *resource)
{
  // Query work group sizes
  cl_int err;
  err = clGetDeviceInfo(resource->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &resource->max_work_group_size, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Work group query failed for %s: %d\n", resource->device_name, err);
    clReleaseKernel(resource->kernel);
    clReleaseProgram(resource->program);
    return err;
  }
  err = clGetKernelWorkGroupInfo(resource->kernel, resource->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                 sizeof(size_t), &resource->preferred_multiple, NULL);
  if (err != CL_SUCCESS)
  {
    resource->preferred_multiple = 64; // Fallback
  }
  cl_uint compute_units;
  clGetDeviceInfo(resource->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
  resource->max_global_work_size = compute_units * resource->max_work_group_size * ctx->config->gpu_work_multiplier;

  printf("Max local work size: %zd\n", resource->max_work_group_size);
  printf("Max global work size: %zd\n", resource->max_global_work_size);
  return CL_SUCCESS;
}

static cl_uint get_pci_bus_id_from_libpciaccess(cl_uint device_idx)
{
#ifdef _WIN32
  // libpciaccess is not available on Windows; return 0 to indicate unknown
  (void)device_idx;
  return 0;
#else
  struct pci_device_iterator *iter = NULL;
  struct pci_device *dev = NULL;
  cl_uint pci_bus_id = 0;
  int current_idx = -1;
  int ret;

  // Initialize libpciaccess
  ret = pci_system_init();
  if (ret != 0)
  {
    fprintf(stderr, "Failed to initialize libpciaccess: %d\n", ret);
    return 0;
  }

  // Create iterator for PCI devices
  iter = pci_slot_match_iterator_create(NULL);
  if (!iter)
  {
    fprintf(stderr, "Failed to create libpciaccess iterator\n");
    pci_system_cleanup();
    return 0;
  }

  // Iterate through PCI devices
  for (dev = pci_device_next(iter); dev; dev = pci_device_next(iter))
  {
    // Filter for AMD devices (vendor ID 0x1002)
    if (dev->vendor_id == 0x1002)
    {
      // Check for VGA or Display controller (class 0x03XX)
      if ((dev->device_class & 0xFF00) == 0x0300)
      {
        current_idx++;
        if (current_idx == (int)device_idx)
        {
          // Found the matching device
          pci_bus_id = dev->bus;
          break;
        }
      }
    }
  }

  if (pci_bus_id == 0)
  {
    fprintf(stderr, "Could not find PCI-BUS-ID for device %u via libpciaccess\n", device_idx);
  }

  // Cleanup
  pci_iterator_destroy(iter);
  pci_system_cleanup();
  return pci_bus_id;
#endif
}

int compare_pci_bus_id(const void *a, const void *b)
{
  const OpenCLResources *ra = (const OpenCLResources *)a;
  const OpenCLResources *rb = (const OpenCLResources *)b;
  return (int)(ra->pci_bus_id - rb->pci_bus_id);
}

OpenCLResources *initalize_all_opencl_gpus(StratumContext *ctx, cl_uint *device_count)
{
  cl_platform_id platform;
  cl_device_id *devices;
  cl_uint num_platforms, num_devices, non_nvidia_count, devices_found;
  cl_int err;

  *device_count = 0;
  err = clGetPlatformIDs(1, &platform, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0)
  {
    fprintf(stderr, "No platforms: %d\n", err);
    return NULL;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0)
  {
    fprintf(stderr, "No GPUs: %d\n", err);
    return NULL;
  }

  devices = malloc(num_devices * sizeof(cl_device_id));
  if (!devices)
  {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Device query failed: %d\n", err);
    free(devices);
    return NULL;
  }

  // Count non-NVIDIA devices
  cl_uint *non_nvidia_indices = malloc(num_devices * sizeof(cl_uint));
  if (!non_nvidia_indices)
  {
    fprintf(stderr, "Memory allocation failed\n");
    free(devices);
    return NULL;
  }
  non_nvidia_count = 0;
  for (cl_uint i = 0; i < num_devices; i++)
  {
    char vendor[128];
    err = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Vendor query failed for device %u: %d\n", i, err);
      free(non_nvidia_indices);
      free(devices);
      return NULL;
    }
    if (strstr(vendor, "NVIDIA") == NULL)
    {
      non_nvidia_indices[non_nvidia_count++] = i;
    }
  }

  if (non_nvidia_count == 0)
  {
    free(non_nvidia_indices);
    free(devices);
    return NULL;
  }

  // Allocate memory for the maximum possible number of valid devices
  OpenCLResources *res = calloc(non_nvidia_count, sizeof(OpenCLResources));
  if (!res)
  {
    fprintf(stderr, "Memory allocation failed\n");
    free(non_nvidia_indices);
    free(devices);
    return NULL;
  }
  devices_found = 0;
  for (cl_uint i = 0; i < non_nvidia_count; i++)
  {
    cl_uint idx = non_nvidia_indices[i];
    char vendor[128];
    char device_name[128];
    err = clGetDeviceInfo(devices[idx], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u name query failed: %d\n", idx, err);
      goto cleanup;
    }

    // Retrieve vendor for PCI-BUS-ID logic
    err = clGetDeviceInfo(devices[idx], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (%s) vendor query failed: %d\n", idx, device_name, err);
      goto cleanup;
    }

    // Retrieve PCI-BUS-ID for AMD devices
    cl_uint pci_bus_id = 0;
    char extensions[2048];
    if (strstr(vendor, "AMD") || strstr(vendor, "Advanced Micro Devices"))
    {
      err = clGetDeviceInfo(devices[idx], CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
      if (err == CL_SUCCESS && strstr(extensions, "cl_amd_device_topology"))
      {
        err = clGetDeviceInfo(devices[idx], 0x4037 /* CL_DEVICE_TOPOLOGY_AMD */, sizeof(cl_uint), &pci_bus_id, NULL);
      }
      else
      {
        err = CL_INVALID_VALUE; // Extension not supported
      }
    }
    else
    {
      err = CL_INVALID_VALUE; // No known extension for other vendors
    }

    if (err != CL_SUCCESS)
    {
      pci_bus_id = get_pci_bus_id_from_libpciaccess(idx);
    }

    // Check if the device is in the selected_gpus list (if any)
    if (ctx->config->selected_gpus_num > 0)
    {
      int found = 0;
      for (int x = 0; x < ctx->config->selected_gpus_num; x++)
      {
        if (pci_bus_id == (cl_uint)ctx->config->selected_gpus[x])
        {
          found = 1;
          break;
        }
      }
      if (found == 0)
      {
        continue; // Skip the GPU since it was not specified
      }
    }
    else
    {
      printf("Using device %u (%s, PCI-BUS-ID: %u)\n", idx, device_name, pci_bus_id);
    }

    // Populate the res array only for valid devices
    res[devices_found].device = devices[idx];
    strncpy(res[devices_found].device_name, device_name, sizeof(res[devices_found].device_name) - 1);
    res[devices_found].pci_bus_id = pci_bus_id;

    err = clGetDeviceInfo(res[devices_found].device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    if (err != CL_SUCCESS || !strstr(extensions, "cl_khr_fp64"))
    {
      fprintf(stderr, "Device %u (%s, PCI-BUS-ID: %u) lacks cl_khr_fp64: %d\n",
              idx, res[devices_found].device_name, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }

    res[devices_found].context = clCreateContext(NULL, 1, &res[devices_found].device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) context failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }

    // Check if out-of-order queue is supported
    cl_command_queue_properties device_queue_props = 0;
    err = clGetDeviceInfo(res[devices_found].device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &device_queue_props, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) queue properties query failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }

    cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES,
        (device_queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0,
        0};

    res[devices_found].queue = clCreateCommandQueueWithProperties(res[devices_found].context, res[devices_found].device, properties, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) creation failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      clReleaseContext(res[devices_found].context);
      goto cleanup;
    }

    // Initialize buffers
    res[devices_found].previous_header_buf = clCreateBuffer(res[devices_found].context, CL_MEM_READ_ONLY, DOMAIN_HASH_SIZE, NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) buffer creation failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }

    res[devices_found].timestamp_buf = clCreateBuffer(res[devices_found].context, CL_MEM_READ_ONLY, sizeof(cl_long), NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) buffer creation failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }

    res[devices_found].matrix_buf = clCreateBuffer(res[devices_found].context, CL_MEM_READ_ONLY, 64 * 64 * sizeof(double), NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) buffer creation failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }

    res[devices_found].target_buf = clCreateBuffer(res[devices_found].context, CL_MEM_READ_ONLY, DOMAIN_HASH_SIZE, NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) buffer creation failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }

    res[devices_found].result_buf = clCreateBuffer(res[devices_found].context, CL_MEM_READ_WRITE, sizeof(OpenCLResult), NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u (PCI-BUS-ID: %u) buffer creation failed: %d\n", idx, res[devices_found].pci_bus_id, err);
      goto cleanup;
    }
    devices_found++;
  }

  if (devices_found == 0)
  {
    free(non_nvidia_indices);
    free(devices);
    free(res);
    return NULL;
  }

  // Reallocate to the exact number of devices found
  res = (OpenCLResources *)realloc(res, devices_found * sizeof(OpenCLResources));
  if (!res)
  {
    fprintf(stderr, "Memory reallocation failed\n");
    goto cleanup;
  }

  qsort(res, devices_found, sizeof(OpenCLResources), compare_pci_bus_id);

  free(non_nvidia_indices);
  free(devices);
  *device_count = devices_found;
  return res;

cleanup:
  for (cl_uint j = 0; j < devices_found; j++)
  {
    if (res[j].previous_header_buf)
      clReleaseMemObject(res[j].previous_header_buf);
    if (res[j].timestamp_buf)
      clReleaseMemObject(res[j].timestamp_buf);
    if (res[j].matrix_buf)
      clReleaseMemObject(res[j].matrix_buf);
    if (res[j].target_buf)
      clReleaseMemObject(res[j].target_buf);
    if (res[j].result_buf)
      clReleaseMemObject(res[j].result_buf);
    if (res[j].queue)
      clReleaseCommandQueue(res[j].queue);
    if (res[j].context)
      clReleaseContext(res[j].context);
  }
  free(non_nvidia_indices);
  free(devices);
  free(res);
  return NULL;
}

cl_int compile_opencl_kernel_from_xxd_header(StratumContext *ctx, OpenCLResources *resource, const unsigned char *kernel, unsigned int kernel_length, const char *kernel_name, const char **required_extensions, size_t num_required_extensions)
{
  cl_int err;
  char *source = malloc(kernel_length + 1);
  if (!source)
  {
    fprintf(stderr, "Failed to allocate memory for kernel source\n");
    return CL_OUT_OF_HOST_MEMORY;
  }
  memcpy(source, kernel, kernel_length);
  source[kernel_length] = '\0';
  const char *src_ptr = source;
  size_t source_len = (size_t)kernel_length;

  // printf("Kernel source size: %u bytes\n", kernel_length);
  printf("Attempting to create kernel: %s\n", kernel_name);

  // Query device capabilities
  char version[128];
  char extensions[2048];
  clGetDeviceInfo(resource->device, CL_DEVICE_OPENCL_C_VERSION, sizeof(version), version, NULL);
  clGetDeviceInfo(resource->device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
  if (ctx->config->debug == 1)
  {
    printf("Device OpenCL C version: %s\n", version);
    printf("Device extensions: %s\n", extensions);
  }

  // Check required extensions
  for (size_t i = 0; i < num_required_extensions; i++)
  {
    if (!strstr(extensions, required_extensions[i]))
    {
      fprintf(stderr, "Device %s does not support required extension: %s\n", resource->device_name, required_extensions[i]);
      free(source);
      return CL_INVALID_DEVICE; // Return CL_INVALID_DEVICE for missing extension
    }
  }

  // Create program
  resource->program = clCreateProgramWithSource(resource->context, 1, &src_ptr, &source_len, &err);
  free(source);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Program creation failed for %s: %d\n", resource->device_name, err);
    return err;
  }

  if (!resource->program)
  {
    fprintf(stderr, "Invalid program for %s\n", resource->device_name);
    return CL_INVALID_VALUE;
  }

  if (!resource->device)
  {
    fprintf(stderr, "Invalid device for %s\n", resource->device_name);
    return CL_INVALID_VALUE;
  }

  // Compile program
  // const char *build_options = "-cl-opt-disable"; // No optimizations.
  char build_options[512];
  if (ctx->config->build_options)
  {
    snprintf(build_options, sizeof(build_options), "%s", ctx->config->build_options);
  }
  else
  {
    snprintf(build_options, sizeof(build_options), "-O%d", ctx->config->opencl_optimization_level);
  }
  err = clBuildProgram(resource->program, 1, &resource->device, build_options, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    char log[4096];
    clGetProgramBuildInfo(resource->program, resource->device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "Build failed for %s with options '%s': %s\n", resource->device_name, build_options, log);
    clReleaseProgram(resource->program);
    return err;
  }

  // List available kernels to debug CL_INVALID_KERNEL_NAME
  cl_uint num_kernels;
  err = clCreateKernelsInProgram(resource->program, 0, NULL, &num_kernels);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to query kernels: %d\n", err);
  }
  else
  {
    cl_kernel *kernels = malloc(sizeof(cl_kernel) * num_kernels);
    err = clCreateKernelsInProgram(resource->program, num_kernels, kernels, NULL);
    if (err == CL_SUCCESS)
    {
      for (cl_uint i = 0; i < num_kernels; i++)
      {
        char kernel_name_buf[256];
        clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, sizeof(kernel_name_buf), kernel_name_buf, NULL);
        // printf("Kernel %u: %s\n", i, kernel_name_buf);
        clReleaseKernel(kernels[i]);
      }
    }
    free(kernels);
  }

  // Create kernel
  resource->kernel = clCreateKernel(resource->program, kernel_name, &err);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Kernel creation failed for %s: %d\n", resource->device_name, err);
    clReleaseProgram(resource->program);
    return err;
  }

  err = calculate_work_sizes(ctx, resource);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Calculate worksizes failed for %s: %d\n", resource->device_name, err);
    clReleaseProgram(resource->program);
    return err;
  }

  printf("Kernel %s compiled for %s\n", kernel_name, resource->device_name);
  return CL_SUCCESS;
}

cl_int load_opencl_kernel_binary(StratumContext *ctx, OpenCLResources *resource, const char *binary_filename, const char *kernel_name)
{
  cl_int err;
  FILE *file = fopen(binary_filename, "rb");
  if (!file)
  {
    fprintf(stderr, "Failed to open %s: %s\n", binary_filename, strerror(errno));
    return CL_INVALID_VALUE;
  }

  fseek(file, 0, SEEK_END);
  size_t binary_size = ftell(file);
  rewind(file);
  unsigned char *binary = malloc(binary_size);
  if (!binary || fread(binary, 1, binary_size, file) != binary_size)
  {
    fprintf(stderr, "Failed to read %s\n", binary_filename);
    fclose(file);
    free(binary);
    return CL_INVALID_VALUE;
  }
  fclose(file);

  resource->program = clCreateProgramWithBinary(resource->context, 1, &resource->device, &binary_size, (const unsigned char **)&binary, NULL, &err);
  free(binary);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Program creation failed for %s: %d\n", resource->device_name, err);
    return err;
  }

  err = clBuildProgram(resource->program, 1, &resource->device, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    char log[2048];
    clGetProgramBuildInfo(resource->program, resource->device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "Build failed for %s: %s\n", resource->device_name, log);
    clReleaseProgram(resource->program);
    return err;
  }

  resource->kernel = clCreateKernel(resource->program, kernel_name, &err);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Kernel creation failed for %s: %d\n", resource->device_name, err);
    clReleaseProgram(resource->program);
    return err;
  }
  err = calculate_work_sizes(ctx, resource);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Calculate worksizes failed for %s: %d\n", resource->device_name, err);
    clReleaseProgram(resource->program);
    return err;
  }
  printf("Kernel %s loaded for %s\n", kernel_name, resource->device_name);
  return CL_SUCCESS;
}

cl_int run_opencl_hoohash_kernel(OpenCLResources *resource, int threadindex,
                                 cl_ulong global_work_size, cl_ulong local_work_size,
                                 unsigned char *previous_header, unsigned char *target,
                                 double matrix[64][64], int64_t timestamp,
                                 uint64_t start_nonce, OpenCLResult *result)
{
  cl_int err = CL_SUCCESS;
  cl_event write_events[5] = {NULL, NULL, NULL, NULL, NULL};
  cl_event kernel_event = NULL;
  cl_event read_result_event = NULL;

  // Validate inputs
  if (!resource || !previous_header || !target || !matrix || !result)
  {
    fprintf(stderr, "Invalid input pointers for %s\n",
            resource ? resource->device_name : "unknown");
    return CL_INVALID_VALUE;
  }

  // Flatten matrix into row-major 1D array to guarantee layout
  const size_t MAT_DIM = 64;
  const size_t MAT_COUNT = MAT_DIM * MAT_DIM;
  const size_t MAT_BYTES = MAT_COUNT * sizeof(double);
  double *flat_matrix = (double *)malloc(MAT_BYTES);
  if (!flat_matrix)
  {
    fprintf(stderr, "Failed to allocate host buffer for flattened matrix\n");
    return CL_OUT_OF_HOST_MEMORY;
  }
  for (size_t r = 0; r < MAT_DIM; ++r)
  {
    for (size_t c = 0; c < MAT_DIM; ++c)
    {
      flat_matrix[r * MAT_DIM + c] = matrix[r][c];
    }
  }

  // Write to persistent buffers asynchronously
  err = clEnqueueWriteBuffer(resource->queue, resource->previous_header_buf,
                             CL_FALSE, 0, DOMAIN_HASH_SIZE, previous_header,
                             0, NULL, &write_events[0]);
  err |= clEnqueueWriteBuffer(resource->queue, resource->timestamp_buf,
                              CL_FALSE, 0, sizeof(int64_t), &timestamp,
                              0, NULL, &write_events[1]);
  err |= clEnqueueWriteBuffer(resource->queue, resource->matrix_buf,
                              CL_FALSE, 0, MAT_BYTES, flat_matrix,
                              0, NULL, &write_events[2]);
  err |= clEnqueueWriteBuffer(resource->queue, resource->target_buf,
                              CL_FALSE, 0, DOMAIN_HASH_SIZE, target,
                              0, NULL, &write_events[3]);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Buffer write failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Initialize result buffer
  OpenCLResult init_result = {0};
  err = clEnqueueWriteBuffer(resource->queue, resource->result_buf, CL_FALSE,
                             0, sizeof(OpenCLResult), &init_result,
                             0, NULL, &write_events[4]);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Result buffer write failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Set kernel arguments (assumes kernel signature expects matrix as __global const double *matrix)
  err = clSetKernelArg(resource->kernel, 0, sizeof(cl_ulong), &local_work_size);
  err |= clSetKernelArg(resource->kernel, 1, sizeof(uint64_t), &start_nonce);
  err |= clSetKernelArg(resource->kernel, 2, sizeof(cl_mem), &resource->previous_header_buf);
  err |= clSetKernelArg(resource->kernel, 3, sizeof(cl_mem), &resource->timestamp_buf);
  err |= clSetKernelArg(resource->kernel, 4, sizeof(cl_mem), &resource->matrix_buf);
  err |= clSetKernelArg(resource->kernel, 5, sizeof(cl_mem), &resource->target_buf);
  err |= clSetKernelArg(resource->kernel, 6, sizeof(cl_mem), &resource->result_buf);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Arg setting failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Execute kernel asynchronously, waiting for all writes to finish
  err = clEnqueueNDRangeKernel(resource->queue, resource->kernel, 1, NULL,
                               (const size_t *)&global_work_size,
                               (const size_t *)&local_work_size,
                               5, write_events, &kernel_event);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Kernel execution failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Read result asynchronously (wait on kernel_event)
  err = clEnqueueReadBuffer(resource->queue, resource->result_buf, CL_FALSE,
                            0, sizeof(OpenCLResult), result, 1,
                            &kernel_event, &read_result_event);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Read failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Wait for completion
  {
    cl_event read_events[1] = {read_result_event};
    err = clWaitForEvents(1, read_events);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Event wait failed for %s: %d\n", resource->device_name, err);
    }
  }

cleanup:
  // Free host buffer
  if (flat_matrix)
    free(flat_matrix);

  // Clean up events
  for (int i = 0; i < 5; i++)
  {
    if (write_events[i])
    {
      clReleaseEvent(write_events[i]);
      write_events[i] = NULL;
    }
  }
  if (kernel_event)
  {
    clReleaseEvent(kernel_event);
    kernel_event = NULL;
  }
  if (read_result_event)
  {
    clReleaseEvent(read_result_event);
    read_result_event = NULL;
  }

  return err;
}

void cleanup_opencl_resources(OpenCLResources *resource)
{
  if (!resource)
    return;

  clReleaseMemObject(resource->previous_header_buf);
  clReleaseMemObject(resource->timestamp_buf);
  clReleaseMemObject(resource->matrix_buf);
  clReleaseMemObject(resource->target_buf);
  clReleaseMemObject(resource->result_buf);
  clReleaseKernel(resource->kernel);
  clReleaseProgram(resource->program);
  clReleaseCommandQueue(resource->queue);
  clReleaseContext(resource->context);
}

void cleanup_all_opencl_gpus(OpenCLResources *resources, cl_uint device_count)
{
  if (!resources)
    return;
  for (cl_uint i = 0; i < device_count; i++)
  {
    cleanup_opencl_resources(&resources[i]);
  }
  free(resources);
}
