#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DOMAIN_HASH_SIZE 32
#define RANDOM_TYPE_XOSHIRO 1
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

void printMatrixSummary(double matrix[64][64])
{
  printf("Matrix (first 3x3 elements):\n");
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      printf("%.2f ", matrix[i][j]);
    }
    printf("\n");
  }
}

typedef struct
{
  uint64_t s0;
  uint64_t s1;
  uint64_t s2;
  uint64_t s3;
} xoshiro_state;

xoshiro_state xoshiro_init(const uint8_t *bytes)
{
  xoshiro_state state;
  state.s0 = *(uint64_t *)(&bytes[0]);
  state.s1 = *(uint64_t *)(&bytes[8]);
  state.s2 = *(uint64_t *)(&bytes[16]);
  state.s3 = *(uint64_t *)(&bytes[24]);
  return state;
}

uint64_t rotl64(const uint64_t x, int k)
{
  return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro_gen(xoshiro_state *x)
{
  uint64_t res = rotl64(x->s0 + x->s3, 23) + x->s0;
  uint64_t t = x->s1 << 17;
  x->s2 ^= x->s0;
  x->s3 ^= x->s1;
  x->s1 ^= x->s2;
  x->s0 ^= x->s3;
  x->s2 ^= t;
  x->s3 = rotl64(x->s3, 45);
  return res;
}

void generateHoohashMatrix(uint8_t *hash, double mat[64][64])
{
  xoshiro_state state = xoshiro_init(hash);
  double normalize = 1000000.0;
  for (int i = 0; i < 64; i++)
  {
    for (int j = 0; j < 64; j++)
    {
      uint64_t val = xoshiro_gen(&state);
      uint32_t lower_4_bytes = val & 0xFFFFFFFF;
      mat[i][j] = (double)lower_4_bytes / (double)UINT32_MAX * normalize;
    }
  }
}

typedef struct
{
  cl_ulong nonce;
  cl_uchar hash[32];
  cl_uchar first_pass[32];
  cl_uchar vector[64];
  cl_double product[64];
} Result;

int compare_target(uint8_t *hash, uint8_t *target)
{
  uint8_t reversed_hash[DOMAIN_HASH_SIZE];
  for (int i = 0; i < DOMAIN_HASH_SIZE; i++)
  {
    reversed_hash[i] = hash[DOMAIN_HASH_SIZE - 1 - i];
  }
  for (size_t i = 0; i < DOMAIN_HASH_SIZE; i++)
  {
    if (reversed_hash[i] > target[i])
      return 1;
    if (reversed_hash[i] < target[i])
      return -1;
  }
  return 0;
}

int run_kernel(cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel,
               cl_ulong local_size, cl_ulong global_size, unsigned char *testname, unsigned char *previous_header,
               unsigned char *target, double matrix[64][64], unsigned long timestamp,
               cl_ulong nonce_mask, cl_ulong nonce_fixed, cl_device_id device)
{
  cl_int err;
  // Query max work group size
  size_t max_work_group_size;
  err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error querying max work group size: %d\n", err);
    return 1;
  }
  if (local_size > max_work_group_size)
  {
    local_size = max_work_group_size;
  }
  if (global_size % local_size != 0)
  {
    global_size = ((global_size / local_size) + 1) * local_size;
  }

  // Check double-precision support
  char extensions[1024];
  err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error querying device extensions: %d\n", err);
    return 1;
  }
  if (strstr(extensions, "cl_khr_fp64") == NULL)
  {
    printf("Error: Device does not support cl_khr_fp64 extension\n");
    return 1;
  }

  // Print input values
  printf("Test Name: %s\n", testname);
  printf("Local Size: %lu\n", local_size);
  printf("Global Size: %lu\n", global_size);
  printf("Timestamp: %lu\n", timestamp);
  printf("Previous Header: ");
  printHash(previous_header, DOMAIN_HASH_SIZE);
  printf("Target: ");
  printHash(target, DOMAIN_HASH_SIZE);
  printf("Nonce Mask: %016lx\n", nonce_mask);
  printf("Nonce Fixed: %016lx\n", nonce_fixed);
  printf("Random Type: %u\n", RANDOM_TYPE_XOSHIRO);
  printMatrixSummary(matrix);

  cl_ulong iteration = 0;
  Result result = {0, {0}, {0}, {0}, {0}};
  cl_ulong random_type = RANDOM_TYPE_XOSHIRO;
  cl_long timestamp_cl = (cl_long)timestamp;

  while (iteration < 18446744073709551615UL)
  {
    clock_t iter_start = clock();

    // Allocate random_state on heap with unique seed per iteration
    cl_ulong4 *random_state = (cl_ulong4 *)malloc(global_size * sizeof(cl_ulong4));
    if (!random_state)
    {
      printf("Error allocating random_state\n");
      return 1;
    }
    for (size_t i = 0; i < global_size; i++)
    {
      uint8_t seed[32];
      for (int j = 0; j < 32; j++)
      {
        seed[j] = previous_header[j] ^ ((i + iteration) & 0xFF);
      }
      random_state[i] = (cl_ulong4){
          *(uint64_t *)(&seed[0]),
          *(uint64_t *)(&seed[8]),
          *(uint64_t *)(&seed[16]),
          *(uint64_t *)(&seed[24])};
    }

    // Create buffers
    cl_mem previous_header_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                DOMAIN_HASH_SIZE, previous_header, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating previous_header buffer: %d\n", err);
      free(random_state);
      return 1;
    }
    cl_mem timestamp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(cl_long), &timestamp_cl, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating timestamp buffer: %d\n", err);
      clReleaseMemObject(previous_header_buf);
      free(random_state);
      return 1;
    }
    cl_mem matrix_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       64 * 64 * sizeof(double), matrix, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating matrix buffer: %d\n", err);
      clReleaseMemObject(previous_header_buf);
      clReleaseMemObject(timestamp_buf);
      free(random_state);
      return 1;
    }
    cl_mem target_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       DOMAIN_HASH_SIZE, target, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating target buffer: %d\n", err);
      clReleaseMemObject(previous_header_buf);
      clReleaseMemObject(timestamp_buf);
      clReleaseMemObject(matrix_buf);
      free(random_state);
      return 1;
    }
    cl_mem random_state_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                             sizeof(cl_ulong4) * global_size, random_state, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating random_state buffer: %d\n", err);
      clReleaseMemObject(previous_header_buf);
      clReleaseMemObject(timestamp_buf);
      clReleaseMemObject(matrix_buf);
      clReleaseMemObject(target_buf);
      free(random_state);
      return 1;
    }
    cl_mem result_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(Result), &result, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating result buffer: %d\n", err);
      clReleaseMemObject(previous_header_buf);
      clReleaseMemObject(timestamp_buf);
      clReleaseMemObject(matrix_buf);
      clReleaseMemObject(target_buf);
      clReleaseMemObject(random_state_buf);
      free(random_state);
      return 1;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &local_size);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_ulong), &nonce_mask);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_ulong), &nonce_fixed);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &previous_header_buf);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &timestamp_buf);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &matrix_buf);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &target_buf);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_ulong), &random_type);
    err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &random_state_buf);
    err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &result_buf);
    err |= clSetKernelArg(kernel, 10, sizeof(cl_ulong), &global_size);
    if (err != CL_SUCCESS)
    {
      printf("Error setting kernel arguments: %d\n", err);
      goto cleanup;
    }

    // Execute kernel
    size_t local_size_arg = local_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size_arg, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("Error enqueuing kernel: %d\n", err);
      goto cleanup;
    }

    // Read output
    err = clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0, sizeof(Result), &result, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("Error reading result buffer: %d\n", err);
      goto cleanup;
    }

    // Calculate and display hashes per second
    clock_t iter_end = clock();
    double iter_time = ((double)(iter_end - iter_start)) / CLOCKS_PER_SEC;
    double hashes_per_second = iter_time > 0 ? (double)global_size / iter_time : 0;
    printf("Iteration %lu: Hash rate: %.2f Mh/s for %s\n",
           iteration, hashes_per_second / 1e6, testname);

    // Check if valid PoW hash is found
    if (result.nonce != 0)
    {
      unsigned char hash_bytes[DOMAIN_HASH_SIZE] = {0};
      memcpy(hash_bytes, &result.hash, DOMAIN_HASH_SIZE);
      unsigned char first_pass[DOMAIN_HASH_SIZE] = {0};
      memcpy(first_pass, &result.first_pass, DOMAIN_HASH_SIZE);
      if (compare_target(hash_bytes, target) <= 0)
      {
        printf("\nValid PoW Hash Found for %s:\n", testname);
        printf("Timestamp: %lu\n", timestamp);
        printf("Previous Header: ");
        printHash(previous_header, DOMAIN_HASH_SIZE);
        printf("Target: ");
        printHash(target, DOMAIN_HASH_SIZE);
        printf("First pass: ");
        printHash(first_pass, DOMAIN_HASH_SIZE);
        printf("Final Nonce: %lu\n", result.nonce);
        printf("Final Hash: ");
        printHash(hash_bytes, DOMAIN_HASH_SIZE);
        printf("Hash rate: %.2f Gh/s\n", hashes_per_second / 1e9);
        goto cleanup;
      }
      else
      {
        printf("Invalid hash found for %s, retrying...\n", testname);
        result.nonce = 0;
      }
    }

    // Cleanup for next iteration
    clReleaseMemObject(previous_header_buf);
    clReleaseMemObject(timestamp_buf);
    clReleaseMemObject(matrix_buf);
    clReleaseMemObject(target_buf);
    clReleaseMemObject(random_state_buf);
    clReleaseMemObject(result_buf);
    free(random_state);
    iteration++;
    continue;

  cleanup:
    printf("Cleaning up!\n");
    clReleaseMemObject(previous_header_buf);
    clReleaseMemObject(timestamp_buf);
    clReleaseMemObject(matrix_buf);
    clReleaseMemObject(target_buf);
    clReleaseMemObject(random_state_buf);
    clReleaseMemObject(result_buf);
    free(random_state);
    if (err != CL_SUCCESS)
      return -1;
    if (iteration >= 18446744073709551615UL && result.nonce == 0)
    {
      printf("Failed to find valid hash for %s after %lu iterations\n", testname, 18446744073709551615UL);
      timestamp = timestamp + 1;
      iteration = 0;
      continue;
    }
    return 0;
  }
  return 0;
}

int main(int argc, char **argv)
{
  cl_uint num_platforms = 0;
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

  char extensions[1024];
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
  if (strstr(extensions, "cl_khr_fp64") == NULL)
  {
    printf("Error: Device does not support cl_khr_fp64 extension\n");
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
  unsigned char *binary = loadBinary("kernel.bin", &binary_size);
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

  // Build program
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
  kernel = clCreateKernel(program, "Hoohash_hash", &err);
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
  cl_ulong local_size = 64;
  cl_ulong global_size = 65536; // Increased for Test 2 to search more nonces

  // Test cases
  cl_ulong nonce_mask = 0xFFFFFFFFFFFFFFFFUL;
  cl_ulong nonce_fixed = 0;
  cl_long timestamp1 = 1725374568455;
  cl_long timestamp2 = 1725374568234;
  cl_long timestamp3 = 1725374569324;
  unsigned char previous_header1[DOMAIN_HASH_SIZE] = {0xa4, 0x9d, 0xbc, 0x7d, 0x44, 0xae, 0x83, 0x25, 0x38, 0x23, 0x59, 0x2f, 0xd3, 0x88, 0xf2, 0x19, 0xf3, 0xcb, 0x83, 0x63, 0x9d, 0x54, 0xc9, 0xe4, 0xc3, 0x15, 0x4d, 0xb3, 0x6f, 0x2b, 0x51, 0x57};
  unsigned char previous_header2[DOMAIN_HASH_SIZE] = {0xb7, 0xc8, 0xf4, 0x3d, 0x8a, 0x99, 0xae, 0xcd, 0xd3, 0x79, 0x12, 0xc9, 0xad, 0x4f, 0x2e, 0x51, 0xc8, 0x00, 0x9f, 0x7c, 0xe1, 0xcd, 0xf6, 0xe3, 0xbe, 0x27, 0x67, 0x97, 0x2c, 0xc6, 0x8a, 0x1c};
  unsigned char previous_header3[DOMAIN_HASH_SIZE] = {0xb7, 0xc8, 0xf4, 0x3d, 0x8a, 0x99, 0xae, 0xcd, 0xd3, 0x79, 0x12, 0xc9, 0xad, 0x4f, 0x2e, 0x51, 0xc8, 0x00, 0x9f, 0x7c, 0xe1, 0xcd, 0xf6, 0xe3, 0xbe, 0x27, 0x67, 0x97, 0x2c, 0xc6, 0x8a, 0x1c};
  ;
  unsigned char target1[DOMAIN_HASH_SIZE] = {
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  unsigned char target2[DOMAIN_HASH_SIZE] = {
      0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  unsigned char target3[DOMAIN_HASH_SIZE] = {
      0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  double matrix1[64][64];
  generateHoohashMatrix(previous_header1, matrix1);
  double matrix2[64][64];
  generateHoohashMatrix(previous_header2, matrix2);
  double matrix3[64][64];
  generateHoohashMatrix(previous_header3, matrix3);

  printf("Test 1: First previous header, first target\n");
  if (run_kernel(context, queue, program, kernel, local_size, global_size, "Test 1", previous_header1, target1, matrix1, timestamp1, nonce_mask, nonce_fixed, device) != 0)
  {
    printf("Test 1 failed\n");
  }
  printf("\nTest 2: Second previous header, second target\n");
  if (run_kernel(context, queue, program, kernel, local_size, global_size, "Test 2", previous_header2, target2, matrix2, timestamp2, nonce_mask, nonce_fixed, device) != 0)
  {
    printf("Test 2 failed\n");
  }
  printf("\nTest 3: Second previous header, second target\n");
  if (run_kernel(context, queue, program, kernel, local_size, global_size, "Test 3", previous_header3, target3, matrix3, timestamp3, nonce_mask, nonce_fixed, device) != 0)
  {
    printf("Test 3 failed\n");
  }

  // Cleanup OpenCL resources
  free(binary);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("\nTotal execution time: %f seconds\n", cpu_time_used);

  return 0;
}