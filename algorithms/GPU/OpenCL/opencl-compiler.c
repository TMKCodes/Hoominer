#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *loadSource(const char *filename, size_t *length)
{
  FILE *file = fopen(filename, "r");
  if (!file)
  {
    perror("Failed to open file");
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  *length = ftell(file);
  rewind(file);

  char *source = (char *)malloc(*length + 1);
  if (!source)
  {
    fclose(file);
    perror("Failed to allocate memory for source");
    return NULL;
  }

  fread(source, 1, *length, file);
  source[*length] = '\0';
  fclose(file);
  return source;
}

int main(int argc, char **argv)
{
  if (argc < 4)
  {
    printf("Usage: %s kernel.cl kernel_function kernel.bin\n", argv[0]);
    return 1;
  }

  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_program program;

  // Get platform
  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error getting platform: %d\n", err);
    return 1;
  }

  // Get device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error getting device: %d\n", err);
    return 1;
  }

  // Create context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating context: %d\n", err);
    return 1;
  }

  // Load kernel source
  size_t src_len;
  char *src = loadSource(argv[1], &src_len);
  if (!src)
  {
    clReleaseContext(context);
    return 1;
  }

  // Create and build program
  program = clCreateProgramWithSource(context, 1, (const char **)&src, &src_len, &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating program: %d\n", err);
    free(src);
    clReleaseContext(context);
    return 1;
  }

  // Build program
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    char buffer[4096];
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &log_size);
    printf("Build failed:\n%.*s\n", (int)log_size, buffer);
    clReleaseProgram(program);
    free(src);
    clReleaseContext(context);
    return 1;
  }

  // Create kernel
  cl_kernel kernel = clCreateKernel(program, argv[2], &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating kernel: %d\n", err);
    clReleaseProgram(program);
    free(src);
    clReleaseContext(context);
    return 1;
  }

  // Create command queue
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS)
  {
    printf("Error creating command queue: %d\n", err);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(src);
    clReleaseContext(context);
    return 1;
  }

  // Save the binary
  size_t binary_size;
  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error getting binary size: %d\n", err);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(src);
    clReleaseContext(context);
    return 1;
  }

  unsigned char *binary = (unsigned char *)malloc(binary_size);
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error getting binary: %d\n", err);
    free(binary);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(src);
    clReleaseContext(context);
    return 1;
  }

  // Save binary to file
  FILE *fp = fopen(argv[3], "wb");
  if (fp)
  {
    fwrite(binary, 1, binary_size, fp);
    fclose(fp);
    printf("Binary saved to %s\n", argv[3]);
  }
  else
  {
    printf("Failed to save binary\n");
  }

  // Cleanup
  free(binary);
  clReleaseCommandQueue(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  free(src);
  clReleaseContext(context);

  printf("Build succeeded!\n");
  return 0;
}