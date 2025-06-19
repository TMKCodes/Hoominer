#ifndef OPENCL_H
#define OPENCL_H
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#define DOMAIN_HASH_SIZE 32
#define RANDOM_TYPE_XOSHIRO 1
#define RANDOM_TYPE_LEAN 0
#define MAX_GLOBAL_SIZE 524288
#define MAX_ITERATIONS 18446744073709551615UL

typedef struct OpenCLResult OpenCLResult;
typedef struct OpenCLResources OpenCLResources;

struct OpenCLResult
{
  cl_ulong nonce;
  cl_uchar hash[32];
  cl_uchar first_pass[32];
  cl_uchar vector[64];
  cl_double product[64];
};

// Structure for OpenCL resources
struct OpenCLResources
{
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue2;
  cl_mem previous_header_buf;
  cl_mem timestamp_buf;
  cl_mem matrix_buf;
  cl_mem target_buf;
  cl_mem random_state_buf;
  cl_mem result_buf;
  size_t max_work_group_size;
  size_t preferred_multiple;
  size_t max_global_work_size;
  cl_ulong4 *random_state;
  char device_name[256];
};

OpenCLResources *initialize_selected_gpus(cl_uint *device_indices, cl_uint num_selected, cl_uint *device_count);
OpenCLResources *initialize_all_gpus(cl_uint *device_count);
cl_int load_kernel_binary(OpenCLResources *resource, const char *binary_filename, const char *kernel_name);
void cleanup_opencl_resources(OpenCLResources *resource);
void cleanup_all_gpus(OpenCLResources *resources, cl_uint device_count);

// Move to hoohash-miner?
cl_int run_hoohash_kernel(OpenCLResources *resource, cl_ulong local_size, cl_ulong global_size, unsigned char *previous_header,
                          unsigned char *target, double matrix[64][64], unsigned long timestamp, cl_ulong nonce_mask,
                          cl_ulong nonce_fixed, OpenCLResult *result);
#endif