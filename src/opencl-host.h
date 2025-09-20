#ifndef OPENCL_HOST_H
#define OPENCL_HOST_H
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <pciaccess.h>
#endif
#include "stratum.h"

#define DOMAIN_HASH_SIZE 32
#define RANDOM_TYPE_XOSHIRO 1
#define RANDOM_TYPE_LEAN 0
#define MAX_GLOBAL_SIZE 524288

typedef struct OpenCLResult OpenCLResult;
typedef struct OpenCLResources OpenCLResources;
typedef struct StratumContext StratumContext;

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
  cl_mem nonces_buf;
  size_t max_work_group_size;
  size_t preferred_multiple;
  size_t max_global_work_size;
  cl_ulong4 random_state;
  char device_name[256];
  cl_uint pci_bus_id;
};

OpenCLResources *initalize_all_opencl_gpus(StratumContext *ctx, cl_uint *device_count);
cl_int compile_opencl_kernel_from_xxd_header(StratumContext *ctx, OpenCLResources *resource, const unsigned char *kernel, unsigned int kernel_length, const char *kernel_name, const char **required_extensions, size_t num_required_extensions);
cl_int load_opencl_kernel_binary(StratumContext *ctx, OpenCLResources *resource, const char *binary_filename, const char *kernel_name);
void cleanup_opencl_resources(OpenCLResources *resource);
void cleanup_all_opencl_gpus(OpenCLResources *resources, cl_uint device_count);

// Move to hoohash-miner?
cl_int run_opencl_hoohash_kernel(OpenCLResources *resource, int threadindex, cl_ulong global_work_size, cl_ulong local_work_size,
                                 unsigned char *previous_header, unsigned char *target, double matrix[64][64],
                                 int64_t timestamp, uint64_t start_nonce, OpenCLResult *result);
#endif