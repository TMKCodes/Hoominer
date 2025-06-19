#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#include "opencl.h"

OpenCLResources *initialize_selected_gpus(cl_uint *device_indices, cl_uint num_selected, cl_uint *device_count)
{
  cl_platform_id platform;
  cl_device_id *devices;
  cl_uint num_platforms, num_devices;
  cl_int err;

  *device_count = 0;
  if (!device_indices || num_selected == 0)
  {
    fprintf(stderr, "Invalid input: No device indices\n");
    return NULL;
  }

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

  for (cl_uint i = 0; i < num_selected; i++)
  {
    if (device_indices[i] >= num_devices)
    {
      fprintf(stderr, "Invalid device index %u: Only %u devices\n", device_indices[i], num_devices);
      return NULL;
    }
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

  OpenCLResources *res = calloc(num_selected, sizeof(OpenCLResources));
  if (!res)
  {
    fprintf(stderr, "Memory allocation failed\n");
    free(devices);
    return NULL;
  }

  for (cl_uint i = 0; i < num_selected; i++)
  {
    cl_uint idx = device_indices[i];
    res[i].device = devices[idx];
    err = clGetDeviceInfo(res[i].device, CL_DEVICE_NAME, sizeof(res[i].device_name), res[i].device_name, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u name query failed: %d\n", idx, err);
      goto cleanup;
    }

    char extensions[1024];
    err = clGetDeviceInfo(res[i].device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    if (err != CL_SUCCESS || !strstr(extensions, "cl_khr_fp64"))
    {
      fprintf(stderr, "Device %u (%s) lacks cl_khr_fp64: %d\n", idx, res[i].device_name, err);
      goto cleanup;
    }

    res[i].context = clCreateContext(NULL, 1, &res[i].device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u context failed: %d\n", idx, err);
      goto cleanup;
    }

    // Check if out-of-order queue is supported
    cl_command_queue_properties queue_properties = 0;
    cl_command_queue_properties device_queue_props;
    err = clGetDeviceInfo(res[i].device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &device_queue_props, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u queue properties query failed: %d\n", idx, err);
      goto cleanup;
    }
    if (device_queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    {
      queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }

    res[i].queue = clCreateCommandQueue(res[i].context, res[i].device, queue_properties, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u queue failed: %d\n", idx, err);
      clReleaseContext(res[i].context);
      goto cleanup;
    }

    // Initialize persistent buffers
    res[i].previous_header_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, DOMAIN_HASH_SIZE, NULL, &err);
    res[i].timestamp_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, sizeof(cl_long), NULL, &err);
    res[i].matrix_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, 64 * 64 * sizeof(double), NULL, &err);
    res[i].target_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, DOMAIN_HASH_SIZE, NULL, &err);
    res[i].result_buf = clCreateBuffer(res[i].context, CL_MEM_READ_WRITE, sizeof(OpenCLResult), NULL, &err);
    res[i].random_state_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, MAX_GLOBAL_SIZE * sizeof(cl_ulong4), NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u buffer creation failed: %d\n", idx, err);
      goto cleanup;
    }

    printf("Initialized GPU %u: %s\n", idx, res[i].device_name);
  }

  free(devices);
  *device_count = num_selected;
  return res;

cleanup:
  for (cl_uint j = 0; j < num_selected; j++)
  {
    clReleaseMemObject(res[j].previous_header_buf);
    clReleaseMemObject(res[j].timestamp_buf);
    clReleaseMemObject(res[j].matrix_buf);
    clReleaseMemObject(res[j].target_buf);
    clReleaseMemObject(res[j].result_buf);
    clReleaseMemObject(res[j].random_state_buf);
    clReleaseCommandQueue(res[j].queue);
    clReleaseContext(res[j].context);
  }
  free(devices);
  free(res);
  return NULL;
}

OpenCLResources *initialize_all_gpus(cl_uint *device_count)
{
  cl_platform_id platform;
  cl_device_id *devices;
  cl_uint num_platforms, num_devices;
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

  OpenCLResources *res = calloc(num_devices, sizeof(OpenCLResources));
  if (!res)
  {
    fprintf(stderr, "Memory allocation failed\n");
    free(devices);
    return NULL;
  }

  for (cl_uint i = 0; i < num_devices; i++)
  {
    res[i].device = devices[i];
    err = clGetDeviceInfo(res[i].device, CL_DEVICE_NAME, sizeof(res[i].device_name), res[i].device_name, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u name query failed: %d\n", i, err);
      goto cleanup;
    }

    char extensions[1024];
    err = clGetDeviceInfo(res[i].device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    if (err != CL_SUCCESS || !strstr(extensions, "cl_khr_fp64"))
    {
      fprintf(stderr, "Device %u (%s) lacks cl_khr_fp64: %d\n", i, res[i].device_name, err);
      goto cleanup;
    }

    res[i].context = clCreateContext(NULL, 1, &res[i].device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u context failed: %d\n", i, err);
      goto cleanup;
    }

    // Check if out-of-order queue is supported
    cl_command_queue_properties queue_properties = 0;
    cl_command_queue_properties device_queue_props;
    err = clGetDeviceInfo(res[i].device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &device_queue_props, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u queue properties query failed: %d\n", i, err);
      goto cleanup;
    }
    if (device_queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    {
      queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }

    res[i].queue = clCreateCommandQueue(res[i].context, res[i].device, queue_properties, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u queue failed: %d\n", i, err);
      clReleaseContext(res[i].context);
      goto cleanup;
    }

    // Initialize persistent buffers
    res[i].previous_header_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, DOMAIN_HASH_SIZE, NULL, &err);
    res[i].timestamp_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, sizeof(cl_long), NULL, &err);
    res[i].matrix_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, 64 * 64 * sizeof(double), NULL, &err);
    res[i].target_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, DOMAIN_HASH_SIZE, NULL, &err);
    res[i].result_buf = clCreateBuffer(res[i].context, CL_MEM_READ_WRITE, sizeof(OpenCLResult), NULL, &err);
    res[i].random_state_buf = clCreateBuffer(res[i].context, CL_MEM_READ_ONLY, MAX_GLOBAL_SIZE * sizeof(cl_ulong4), NULL, &err);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Device %u buffer creation failed: %d\n", i, err);
      goto cleanup;
    }

    printf("Initialized GPU %u: %s\n", i, res[i].device_name);
  }

  free(devices);
  *device_count = num_devices;
  return res;

cleanup:
  for (cl_uint j = 0; j < num_devices; j++)
  {
    clReleaseMemObject(res[j].previous_header_buf);
    clReleaseMemObject(res[j].timestamp_buf);
    clReleaseMemObject(res[j].matrix_buf);
    clReleaseMemObject(res[j].target_buf);
    clReleaseMemObject(res[j].result_buf);
    clReleaseMemObject(res[j].random_state_buf);
    clReleaseCommandQueue(res[j].queue);
    clReleaseContext(res[j].context);
  }
  free(devices);
  free(res);
  return NULL;
}

cl_int load_kernel_binary(OpenCLResources *resource, const char *binary_filename, const char *kernel_name)
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

  // Query work group sizes
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
  resource->max_global_work_size = compute_units * resource->max_work_group_size * 10;
  printf("Max local work size: %ld\n", resource->max_work_group_size);
  printf("Max global work size: %ld\n", resource->max_global_work_size);

  // Generate random state
  if (posix_memalign((void **)&resource->random_state, 64, resource->max_global_work_size * sizeof(cl_ulong4)) != 0)
  {
    fprintf(stderr, "Random state allocation failed for %s\n", resource->device_name);
    err = CL_OUT_OF_HOST_MEMORY;
  }

  // Copy same state to all entries
  for (size_t i = 1; i < resource->max_global_work_size; i++)
  {

    unsigned int seed = (unsigned int)time(NULL);

    // Use rand_r to generate initial entropy
    unsigned int entropy = rand_r(&seed);
    seed ^= entropy; // add more variability to seed

    // Fill random_state[0]
    uint64_t r1 = ((uint64_t)rand_r(&seed) << 32) | rand_r(&seed);
    uint64_t r2 = ((uint64_t)rand_r(&seed) << 32) | rand_r(&seed);

    resource->random_state[i].s[0] = r1;
    resource->random_state[i].s[1] = r2;
    resource->random_state[i].s[2] = ((uint64_t)rand_r(&seed) << 32) | rand_r(&seed);
    resource->random_state[i].s[3] = ((uint64_t)rand_r(&seed) << 32) | rand_r(&seed);
  }

  printf("Kernel %s loaded for %s\n", kernel_name, resource->device_name);
  return CL_SUCCESS;
}

cl_int run_hoohash_kernel(OpenCLResources *resource, cl_ulong local_size, cl_ulong global_size,
                          unsigned char *previous_header, unsigned char *target, double matrix[64][64],
                          unsigned long timestamp, cl_ulong nonce_mask, cl_ulong nonce_fixed, OpenCLResult *result)
{
  cl_int err;
  cl_event write_events[5], kernel_event, read_event;

  // Validate inputs
  if (!resource || !previous_header || !target || !matrix || !result)
  {
    fprintf(stderr, "Invalid input pointers for %s\n", resource ? resource->device_name : "unknown");
    return CL_INVALID_VALUE;
  }

  // Optimize work group size using stored values
  local_size = (local_size / resource->preferred_multiple) * resource->preferred_multiple;
  local_size = (local_size > resource->max_work_group_size) ? resource->max_work_group_size : local_size;
  global_size = (global_size / local_size) * local_size;
  // global_size = (global_size > resource->max_global_work_size) ? resource->max_global_work_size : global_size;

  // Write to persistent buffers asynchronously
  err = clEnqueueWriteBuffer(resource->queue, resource->previous_header_buf, CL_FALSE, 0, DOMAIN_HASH_SIZE, previous_header, 0, NULL, &write_events[0]);
  err |= clEnqueueWriteBuffer(resource->queue, resource->timestamp_buf, CL_FALSE, 0, sizeof(cl_long), &timestamp, 0, NULL, &write_events[1]);
  err |= clEnqueueWriteBuffer(resource->queue, resource->matrix_buf, CL_FALSE, 0, 64 * 64 * sizeof(double), matrix, 0, NULL, &write_events[2]);
  err |= clEnqueueWriteBuffer(resource->queue, resource->target_buf, CL_FALSE, 0, DOMAIN_HASH_SIZE, target, 0, NULL, &write_events[3]);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Buffer write failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Initialize result buffer
  OpenCLResult init_result = {0};
  err = clEnqueueWriteBuffer(resource->queue, resource->result_buf, CL_FALSE, 0, sizeof(OpenCLResult), &init_result, 0, NULL, &write_events[4]);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Result buffer write failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  err = clEnqueueWriteBuffer(resource->queue, resource->random_state_buf, CL_FALSE, 0, global_size * sizeof(cl_ulong4), resource->random_state, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Random state buffer write failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Set kernel arguments
  cl_ulong random_type = RANDOM_TYPE_LEAN; // RANDOM_TYPE_LEAN OR RANDOM_TYPE_XOSHIRO
  err = clSetKernelArg(resource->kernel, 0, sizeof(cl_ulong), &local_size);
  err |= clSetKernelArg(resource->kernel, 1, sizeof(cl_ulong), &nonce_mask);
  err |= clSetKernelArg(resource->kernel, 2, sizeof(cl_ulong), &nonce_fixed);
  err |= clSetKernelArg(resource->kernel, 3, sizeof(cl_mem), &resource->previous_header_buf);
  err |= clSetKernelArg(resource->kernel, 4, sizeof(cl_mem), &resource->timestamp_buf);
  err |= clSetKernelArg(resource->kernel, 5, sizeof(cl_mem), &resource->matrix_buf);
  err |= clSetKernelArg(resource->kernel, 6, sizeof(cl_mem), &resource->target_buf);
  err |= clSetKernelArg(resource->kernel, 7, sizeof(cl_ulong), &random_type);
  err |= clSetKernelArg(resource->kernel, 8, sizeof(cl_mem), &resource->random_state_buf);
  err |= clSetKernelArg(resource->kernel, 9, sizeof(cl_mem), &resource->result_buf);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Arg setting failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Execute kernel asynchronously
  err = clEnqueueNDRangeKernel(resource->queue, resource->kernel, 1, NULL, &global_size, &local_size, 5, write_events, &kernel_event);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Kernel execution failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Read result asynchronously
  err = clEnqueueReadBuffer(resource->queue, resource->result_buf, CL_FALSE, 0, sizeof(OpenCLResult), result, 1, &kernel_event, &read_event);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Read failed for %s: %d\n", resource->device_name, err);
    goto cleanup;
  }

  // Wait for completion
  err = clWaitForEvents(1, &read_event);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Event wait failed for %s: %d\n", resource->device_name, err);
  }

cleanup:
  // Clean up events
  for (int i = 0; i < 5; i++)
    if (write_events[i])
      clReleaseEvent(write_events[i]);
  if (kernel_event)
    clReleaseEvent(kernel_event);
  if (read_event)
    clReleaseEvent(read_event);
  return err;
}

void cleanup_opencl_resources(OpenCLResources *resource)
{
  if (!resource)
    return;

  free(resource->random_state);
  clReleaseMemObject(resource->previous_header_buf);
  clReleaseMemObject(resource->timestamp_buf);
  clReleaseMemObject(resource->matrix_buf);
  clReleaseMemObject(resource->target_buf);
  clReleaseMemObject(resource->result_buf);
  clReleaseMemObject(resource->random_state_buf);
  clReleaseKernel(resource->kernel);
  clReleaseProgram(resource->program);
  clReleaseCommandQueue(resource->queue);
  clReleaseContext(resource->context);
}

void cleanup_all_gpus(OpenCLResources *resources, cl_uint device_count)
{
  if (!resources)
    return;
  for (cl_uint i = 0; i < device_count; i++)
  {
    cleanup_opencl_resources(&resources[i]);
  }
  free(resources);
}