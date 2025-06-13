#include <CL/cl.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char *loadBinary(const char *filename, size_t *length)
{
  FILE *file = fopen(filename, "rb");
  if (!file)
  {
    perror("Failed to open binary file");
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  *length = ftell(file);
  rewind(file);

  unsigned char *binary = (unsigned char *)malloc(*length);
  if (!binary)
  {
    fclose(file);
    perror("Failed to allocate memory for binary");
    return NULL;
  }

  fread(binary, 1, *length, file);
  fclose(file);
  return binary;
}

void printHash(unsigned char *hash, size_t len)
{
  for (size_t i = 0; i < len; i++)
  {
    printf("%02x", hash[i]);
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    printf("Usage: %s kernel.bin kernel_func hash\n", argv[0]);
    return 1;
  }

  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  // Initialize OpenCL
  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error getting platform: %d\n", err);
    return 1;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error getting device: %d\n", err);
    return 1;
  }

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating context: %d\n", err);
    return 1;
  }

  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating command queue: %d\n", err);
    clReleaseContext(context);
    return 1;
  }

  // Load pre-compiled binary
  size_t binary_size;
  unsigned char *binary = loadBinary(argv[1], &binary_size);
  if (!binary)
  {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }

  // Create program from binary
  const unsigned char *binary_ptr = binary;
  program = clCreateProgramWithBinary(context, 1, &device, &binary_size, &binary_ptr, NULL, &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating program from binary: %d\n", err);
    free(binary);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }

  // Build program (required to link binary)
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    char buffer[4096];
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &log_size);
    printf("Build failed:\n%.*s\n", (int)log_size, buffer);
    free(binary);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }

  // Create kernel
  kernel = clCreateKernel(program, argv[2], &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating kernel: %d\n", err);
    free(binary);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }

  clock_t start, end;
  double cpu_time_used;
  start = clock();

  for (int i = 0; i < 100000; i++)
  {
    // Prepare test data
    char input_str[64];
    snprintf(input_str, sizeof(input_str), "Hello, Hoosat Network! %d", i);

    size_t input_len = strlen(input_str);
    size_t out_len = 32;

    unsigned char *input = (unsigned char *)malloc(input_len);
    unsigned char *output = (unsigned char *)malloc(out_len);
    if (!input || !output)
    {
      printf("Error allocating memory for input/output\n");
      free(input);
      free(output);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      return 1;
    }
    memcpy(input, input_str, input_len);

    // Create buffers
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      input_len, input, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating input buffer: %d\n", err);
      free(input);
      free(output);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      return 1;
    }

    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       out_len, NULL, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating output buffer: %d\n", err);
      clReleaseMemObject(input_buf);
      free(input);
      free(output);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      return 1;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_ulong), &input_len);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_ulong), &out_len);
    if (err != CL_SUCCESS)
    {
      printf("Error setting kernel arguments: %d\n", err);
      clReleaseMemObject(input_buf);
      clReleaseMemObject(output_buf);
      free(input);
      free(output);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      return 1;
    }

    // Execute kernel
    size_t global_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("Error enqueuing kernel: %d\n", err);
      clReleaseMemObject(input_buf);
      clReleaseMemObject(output_buf);
      free(input);
      free(output);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      return 1;
    }

    // Read output
    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, out_len, output, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("Error reading output buffer: %d\n", err);
      clReleaseMemObject(input_buf);
      clReleaseMemObject(output_buf);
      free(input);
      free(output);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      return 1;
    }

    // Print result
    printf("Input: %s\n", input_str);
    printf("BLAKE3 hash: ");
    printHash(output, out_len);

    // Cleanup
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    free(input);
    free(output);
  }
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Execution time: %f seconds\n", cpu_time_used);

  // Free binary after the loop
  free(binary);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}