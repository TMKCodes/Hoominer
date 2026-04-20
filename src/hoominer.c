#define CL_TARGET_OPENCL_VERSION 200
#include <stdio.h>
#include <signal.h>
#include <pthread.h>
#include <stdbool.h>
#ifndef _WIN32
#include <libgen.h>
#endif
#include <time.h> // Added for time tracking
#include <blake3.h>
#include "config.h"
#include "opencl-host.h"
#include "cuda-host.h"
#include "stratum.h"
#include "reporting.h"
#include "miner-hoohash.h"
#include "miner-pepepow.h"
#include "hoohash_cl.h"
#include "api.h"
#include "platform_compat.h"

static void apply_opencl_cache_env(const struct HoominerConfig *config)
{
  if (!config || !config->disable_opencl_cache)
    return;

#ifdef _WIN32
  _putenv_s("CL_CACHE_DISABLE", "1");
  _putenv_s("POCL_CACHE_DISABLE", "1");
  _putenv_s("POCL_CACHE", "0");
  _putenv_s("AMD_OCL_CACHE_DISABLE", "1");
#else
  setenv("CL_CACHE_DISABLE", "1", 1);
  setenv("CL_CACHE_DIR", "/tmp/hoominer_cl_cache", 1);
  setenv("POCL_CACHE_DISABLE", "1", 1);
  setenv("POCL_CACHE", "0", 1);
  setenv("POCL_CACHE_DIR", "/tmp/hoominer_pocl_cache", 1);
  setenv("AMD_OCL_CACHE_DISABLE", "1", 1);
  setenv("OCL_CACHE_DISABLE", "1", 1);
#endif
  printf("OpenCL cache disabled (best-effort via environment variables)\n");
}

#ifdef _WIN32
#include <windows.h>
int get_cpu_threads()
{
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return sysInfo.dwNumberOfProcessors;
}
#elif __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <csignal>
int get_cpu_threads()
{
  int ncpu;
  size_t len = sizeof(ncpu);
  sysctlbyname("hw.logicalcpu", &ncpu, &len, NULL, 0);
  return ncpu;
}
#else
int get_cpu_threads()
{
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 4;
}
#endif

#define HASH_SIZE 32

char *exe_path = NULL;
StratumContext *ctx = NULL;

static void cleanup_stratum_connection(StratumContext *ctx)
{
  if (!ctx)
    return;

  // Ask threads to exit their loops.
  ctx->running = false;

  // Join (do not cancel) so threads can run their own cleanup sections.
  if (ctx->recv_thread_created)
  {
    pthread_join(ctx->recv_thread, NULL);
    ctx->recv_thread_created = 0;
  }

  if (ctx->hd && ctx->hd->display_thread_created)
  {
    pthread_join(ctx->hd->display_thread, NULL);
    ctx->hd->display_thread_created = 0;
  }

  // Mining threads also use ctx->running; join them here to avoid leaking
  // joinable thread resources across reconnect cycles.
  if (ctx->ms)
  {
    if (ctx->ms->mining_cpu_threads)
    {
      for (int i = 0; i < ctx->ms->num_cpu_threads; i++)
      {
        pthread_join(ctx->ms->mining_cpu_threads[i], NULL);
      }
      free(ctx->ms->mining_cpu_threads);
      ctx->ms->mining_cpu_threads = NULL;
      ctx->ms->num_cpu_threads = 0;
    }
    if (ctx->ms->mining_opencl_threads)
    {
      for (int i = 0; i < ctx->ms->num_opencl_threads; i++)
      {
        pthread_join(ctx->ms->mining_opencl_threads[i], NULL);
      }
      free(ctx->ms->mining_opencl_threads);
      ctx->ms->mining_opencl_threads = NULL;
      ctx->ms->num_opencl_threads = 0;
    }
    if (ctx->ms->mining_cuda_threads)
    {
      for (int i = 0; i < ctx->ms->num_cuda_threads; i++)
      {
        pthread_join(ctx->ms->mining_cuda_threads[i], NULL);
      }
      free(ctx->ms->mining_cuda_threads);
      ctx->ms->mining_cuda_threads = NULL;
      ctx->ms->num_cuda_threads = 0;
    }

    // Clear job queue to prevent memory leaks on reconnection
    pthread_mutex_lock(&ctx->ms->job_queue.queue_mutex);
    for (int i = 0; i < JOB_QUEUE_SIZE; i++)
    {
      if (ctx->ms->job_queue.jobs[i].job_id)
      {
        free(ctx->ms->job_queue.jobs[i].job_id);
        ctx->ms->job_queue.jobs[i].job_id = NULL;
      }
      ctx->ms->job_queue.jobs[i].running = 0;
      ctx->ms->job_queue.jobs[i].completed = 0;
    }
    ctx->ms->job_queue.head = 0;
    ctx->ms->job_queue.tail = 0;
    ctx->ms->new_job_available = 0;
    pthread_mutex_unlock(&ctx->ms->job_queue.queue_mutex);

    // Clear extranonce on reconnection to avoid stale data
    if (ctx->ms->extranonce)
    {
      free(ctx->ms->extranonce);
      ctx->ms->extranonce = NULL;
    }
    ctx->ms->extranonce2_size = 0;
    memset(ctx->ms->current_en2, 0, sizeof(ctx->ms->current_en2));
  }

  // As a safety net, close SSL/socket here too (thread normally does SSL cleanup).
  if (ctx->ssl)
  {
    SSL_shutdown(ctx->ssl);
    SSL_free(ctx->ssl);
    ctx->ssl = NULL;
  }
  if (ctx->ssl_ctx)
  {
    SSL_CTX_free(ctx->ssl_ctx);
    ctx->ssl_ctx = NULL;
  }
  if (ctx->sockfd >= 0)
  {
    socket_close_portable(ctx->sockfd);
    ctx->sockfd = -1;
  }
}
static void self_test_hash_endianness(void)
{
#ifdef _WIN32
  printf("[self-test] Verifying header endianness on Windows...\n");
  uint64_t ids[4] = {0x0102030405060708ULL, 0x1112131415161718ULL, 0x2122232425262728ULL, 0x3132333435363738ULL};
  uint8_t header_le[32] = {0};
  uint8_t header_memcpy[32] = {0};
  // Variant A: current little-endian staging
  smallJobHeader(ids, header_le);
  // Variant B: memcpy without swapping
  memcpy(header_memcpy + 0, &ids[0], 8);
  memcpy(header_memcpy + 8, &ids[1], 8);
  memcpy(header_memcpy + 16, &ids[2], 8);
  memcpy(header_memcpy + 24, &ids[3], 8);

  State s1 = {0};
  memcpy(s1.PrevHeader, header_le, 32);
  s1.Timestamp = 0;
  s1.Nonce = 0;
  generateHoohashMatrix(s1.PrevHeader, s1.mat);
  uint8_t out1[32] = {0};
  CalculateProofOfWorkValue(&s1, out1);

  State s2 = {0};
  memcpy(s2.PrevHeader, header_memcpy, 32);
  s2.Timestamp = 0;
  s2.Nonce = 0;
  generateHoohashMatrix(s2.PrevHeader, s2.mat);
  uint8_t out2[32] = {0};
  CalculateProofOfWorkValue(&s2, out2);

  printf("[self-test] Variant A (LE swap) header: ");
  print_hex("", header_le, 32);
  printf("[self-test] Variant A hash: ");
  print_hex("", out1, 32);
  printf("[self-test] Variant B (memcpy) header: ");
  print_hex("", header_memcpy, 32);
  printf("[self-test] Variant B hash: ");
  print_hex("", out2, 32);
  if (memcmp(out1, out2, 32) != 0)
  {
    printf("[self-test] Hashes differ. If pool accepts only one variant, we will align header staging accordingly.\n");
  }
#endif
}

static void self_test_from_main_vectors(void)
{
  // Two vectors adapted from algorithms/hoohash/main_test.c
  {
    printf("[self-test] --------------------------------------------------------------\n");
    State state = (State){0};
    uint8_t result[DOMAIN_HASH_SIZE] = {0};
    uint8_t PrevHeader[DOMAIN_HASH_SIZE] = {
        0xa4, 0x9d, 0xbc, 0x7d, 0x44, 0xae, 0x83, 0x25,
        0x38, 0x23, 0x59, 0x2f, 0xd3, 0x88, 0xf2, 0x19,
        0xf3, 0xcb, 0x83, 0x63, 0x9d, 0x54, 0xc9, 0xe4,
        0xc3, 0x15, 0x4d, 0xb3, 0x6f, 0x2b, 0x51, 0x57};
    memcpy(state.PrevHeader, PrevHeader, DOMAIN_HASH_SIZE);
    state.Timestamp = 1725374568455ULL;
    state.Nonce = 7598630810654817703ULL;
    generateHoohashMatrix(state.PrevHeader, state.mat);
    CalculateProofOfWorkValue(&state, result);
    printf("[self-test] Vector1 PrevHeader: ");
    print_hex("", PrevHeader, DOMAIN_HASH_SIZE);
    printf("[self-test] Vector1 Timestamp: %lu, Nonce: %lu\n", state.Timestamp, state.Nonce);
    printf("[self-test] Vector1 Output: ");
    print_hex("", result, DOMAIN_HASH_SIZE);
  }
  {
    printf("[self-test] --------------------------------------------------------------\n");
    State state = (State){0};
    uint8_t result[DOMAIN_HASH_SIZE] = {0};
    uint8_t PrevHeader[DOMAIN_HASH_SIZE] = {
        0xb7, 0xc8, 0xf4, 0x3d, 0x8a, 0x99, 0xae, 0xcd,
        0xd3, 0x79, 0x12, 0xc9, 0xad, 0x4f, 0x2e, 0x51,
        0xc8, 0x00, 0x9f, 0x7c, 0xe1, 0xcd, 0xf6, 0xe3,
        0xbe, 0x27, 0x67, 0x97, 0x2c, 0xc6, 0x8a, 0x1c};
    memcpy(state.PrevHeader, PrevHeader, DOMAIN_HASH_SIZE);
    state.Timestamp = 1725374568234ULL;
    state.Nonce = 14449448288038978941ULL;
    generateHoohashMatrix(state.PrevHeader, state.mat);
    CalculateProofOfWorkValue(&state, result);
    printf("[self-test] Vector2 PrevHeader: ");
    print_hex("", PrevHeader, DOMAIN_HASH_SIZE);
    printf("[self-test] Vector2 Timestamp: %lu, Nonce: %lu\n", state.Timestamp, state.Nonce);
    printf("[self-test] Vector2 Output: ");
    print_hex("", result, DOMAIN_HASH_SIZE);
  }
}

/*
 * PEPEPOW self-test: exercise hoohashv110_compute() (the PEPEPOW hash
 * function) against a synthetic 80-byte header.  The test exercises the
 * complete code path (BLAKE3 first-pass, matrix generation from masked
 * header, matrix multiplication) and prints the output so that it can be
 * compared against a reference implementation.
 */
static void self_test_pepepow(void)
{
  printf("[pepepow-self-test] --------------------------------------------------\n");
  printf("[pepepow-self-test] Testing PEPEPOW (hoohashv110) hash function\n");

  /* Construct a deterministic 80-byte test header where each byte equals
   * its index.  Bytes [76..79] (nonce field) are explicitly zeroed so the
   * matrix seed is computed for nonce=0 as the mining loop would do. */
  uint8_t test_header[PEPEPOW_HEADER_SIZE] = {0};
  for (int i = 0; i < PEPEPOW_NONCE_OFFSET; i++)
    test_header[i] = (uint8_t)(i & 0xFF);
  /* test_header[76..79] remain 0 (nonce-zeroed template). */

  /* Build the nonce-masked header and derive the matrix seed. */
  uint8_t masked[PEPEPOW_HEADER_SIZE];
  memcpy(masked, test_header, PEPEPOW_HEADER_SIZE);
  /* nonce bytes [76..79] are already 0 */

  blake3_hasher hasher;
  uint8_t matrix_seed[DOMAIN_HASH_SIZE];
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, masked, PEPEPOW_HEADER_SIZE);
  blake3_hasher_finalize(&hasher, matrix_seed, DOMAIN_HASH_SIZE);

  double mat[64][64];
  generateHoohashMatrix(matrix_seed, mat);

  /* Test with nonce = 1: write as big-endian at offset 76 */
  uint8_t hdr_n1[PEPEPOW_HEADER_SIZE];
  memcpy(hdr_n1, test_header, PEPEPOW_HEADER_SIZE);
  /* BE nonce=1: bytes [0x00, 0x00, 0x00, 0x01] */
  hdr_n1[76] = 0x00;
  hdr_n1[77] = 0x00;
  hdr_n1[78] = 0x00;
  hdr_n1[79] = 0x01;

  /* Compute first-pass hash and final PoW hash using hoohashv110 logic. */
  uint8_t first_pass[DOMAIN_HASH_SIZE];
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, hdr_n1, PEPEPOW_HEADER_SIZE);
  blake3_hasher_finalize(&hasher, first_pass, DOMAIN_HASH_SIZE);

  uint8_t output[DOMAIN_HASH_SIZE];
  HoohashMatrixMultiplication(mat, first_pass, output, 1ULL);

  printf("[pepepow-self-test] Header (first 16 bytes): ");
  print_hex("", test_header, 16);
  printf("[pepepow-self-test] MatrixSeed: ");
  print_hex("", matrix_seed, DOMAIN_HASH_SIZE);
  printf("[pepepow-self-test] FirstPass (nonce=1): ");
  print_hex("", first_pass, DOMAIN_HASH_SIZE);
  printf("[pepepow-self-test] Output (nonce=1): ");
  print_hex("", output, DOMAIN_HASH_SIZE);
  printf("[pepepow-self-test] --------------------------------------------------\n");
}

void cleanup(int sig)
{
  printf("Cleanup initiated due to signal %d\n", sig);

  if (ctx == NULL)
  {
    free(exe_path);
    fflush(stdout);
    exit(sig == SIGINT || sig == SIGTERM ? EXIT_FAILURE : EXIT_SUCCESS);
  }

  // Stop and join connection/display threads
  cleanup_stratum_connection(ctx);

  if (ctx->ms)
  {
    // Clear job queue
    // printf("Cleaning jobs\n");
    pthread_mutex_destroy(&ctx->ms->job_queue.queue_mutex);
    pthread_cond_destroy(&ctx->ms->job_queue.queue_cond);
    for (int i = 0; i < JOB_QUEUE_SIZE; i++)
    {
      if (ctx->ms->job_queue.jobs[i].job_id)
      {
        free(ctx->ms->job_queue.jobs[i].job_id);
        ctx->ms->job_queue.jobs[i].job_id = NULL;
      }
      ctx->ms->job_queue.jobs[i].running = 0;
      ctx->ms->job_queue.jobs[i].completed = 0;
    }
    ctx->ms->job_queue.head = 0;
    ctx->ms->job_queue.tail = 0;

    for (int i = 0; i < JOB_QUEUE_SIZE; i++)
    {
      if (ctx->ms->job_queue.jobs[i].job_id)
      {
        free(ctx->ms->job_queue.jobs[i].job_id);
        ctx->ms->job_queue.jobs[i].job_id = NULL;
      }
    }

    // Clean up other mining state resources
    if (ctx->ms->global_target)
    {
      free(ctx->ms->global_target);
      ctx->ms->global_target = NULL;
    }
    if (ctx->ms->extranonce)
    {
      free(ctx->ms->extranonce);
      ctx->ms->extranonce = NULL;
    }
    pthread_mutex_destroy(&ctx->ms->job_mutex);
    pthread_mutex_destroy(&ctx->ms->target_mutex);
    free(ctx->ms);
    ctx->ms = NULL;
  }

  // Clean up OpenCL resources
  if (ctx->opencl_resources)
  {
    for (cl_uint i = 0; i < ctx->opencl_device_count; i++)
    {
      if (ctx->opencl_resources[i].previous_header_buf)
        clReleaseMemObject(ctx->opencl_resources[i].previous_header_buf);
      if (ctx->opencl_resources[i].timestamp_buf)
        clReleaseMemObject(ctx->opencl_resources[i].timestamp_buf);
      if (ctx->opencl_resources[i].matrix_buf)
        clReleaseMemObject(ctx->opencl_resources[i].matrix_buf);
      if (ctx->opencl_resources[i].target_buf)
        clReleaseMemObject(ctx->opencl_resources[i].target_buf);
      if (ctx->opencl_resources[i].result_buf)
        clReleaseMemObject(ctx->opencl_resources[i].result_buf);
      if (ctx->opencl_resources[i].pepepow_header_buf)
        clReleaseMemObject(ctx->opencl_resources[i].pepepow_header_buf);
      if (ctx->opencl_resources[i].random_state_buf)
        clReleaseMemObject(ctx->opencl_resources[i].random_state_buf);
      if (ctx->opencl_resources[i].kernel)
        clReleaseKernel(ctx->opencl_resources[i].kernel);
      if (ctx->opencl_resources[i].program)
        clReleaseProgram(ctx->opencl_resources[i].program);
      if (ctx->opencl_resources[i].queue)
        clReleaseCommandQueue(ctx->opencl_resources[i].queue);
      if (ctx->opencl_resources[i].context)
        clReleaseContext(ctx->opencl_resources[i].context);
    }
    free(ctx->opencl_resources);
    ctx->opencl_resources = NULL;
    ctx->opencl_device_count = 0;
  }

  // Clean up CUDA resources
  cleanup_all_cuda_gpus(ctx->cuda_resources, ctx->cuda_device_count);

  // Clean up hashrate display
  if (ctx->hd)
  {
    free_hashrate_display(ctx->hd);
    ctx->hd = NULL;
  }

  // Close socket
  if (ctx->sockfd >= 0)
  {
    close(ctx->sockfd);
    ctx->sockfd = -1;
  }

  // Free config and its members
  if (ctx->config)
  {
    // Free stratum pool IPs
    for (int i = 0; i < ctx->config->stratum_urls_num; i++)
    {
      if (ctx->config->stratum_urls[i].pool_ip)
      {
        free(ctx->config->stratum_urls[i].pool_ip);
        ctx->config->stratum_urls[i].pool_ip = NULL;
      }
    }
    // Free build options if allocated
    if (ctx->config->build_options)
    {
      free(ctx->config->build_options);
      ctx->config->build_options = NULL;
    }
    free(ctx->config);
    ctx->config = NULL;
  }

  // Free context
  free(ctx);
  ctx = NULL;

  // Free global resources
  free(exe_path);
  exe_path = NULL;

  fflush(stdout);
  exit(sig == SIGINT || sig == SIGTERM ? EXIT_FAILURE : EXIT_SUCCESS);
}

int initialize_mining(StratumContext *ctx, const char *username, const char *algorithm, char *exe_dir)
{
  ctx->worker = username;
  ctx->ms->num_cpu_threads = get_cpu_threads();
  if (ctx->config->cpu_threads > 0)
    ctx->ms->num_cpu_threads = ctx->config->cpu_threads;
  // printf("CPU threads %d\n", ctx->ms->num_cpu_threads);
  ctx->ms->num_opencl_threads = 0;
  ctx->ms->num_cuda_threads = 0;

  if (ctx->config->disable_gpu == false)
  {
    if (ctx->config->disable_opencl == false)
    {
      ctx->opencl_resources = initalize_all_opencl_gpus(ctx, &ctx->opencl_device_count);
      if (ctx->config->list_gpus == false)
      {
        printf("OpenCL devices found: %d\n", ctx->opencl_device_count);
        if (strcmp(algorithm, "hoohash") == 0)
        {
          for (cl_uint i = 0; i < ctx->opencl_device_count; i++)
          {
            const char *required_extensions[] = {"cl_khr_fp64"};
            size_t num_required_extensions = 1;
            cl_int compile_kernel_error = compile_opencl_kernel_from_xxd_header(ctx, &ctx->opencl_resources[i], Hoohash_cl, Hoohash_cl_len, "Hoohash_hash", required_extensions, num_required_extensions);
            if (compile_kernel_error != CL_SUCCESS)
            {
              printf("Failed to initialize OpenCL kernels, Error code %d.\n", compile_kernel_error);
              return -1;
            }
            printf("Loaded OpenCL Hoohash kernel for device %u\n", ctx->cpu_device_count + i);
          }
        }
        else if (strcmp(algorithm, "pepepow") == 0)
        {
          for (cl_uint i = 0; i < ctx->opencl_device_count; i++)
          {
            const char *required_extensions[] = {"cl_khr_fp64"};
            size_t num_required_extensions = 1;
            cl_int compile_kernel_error = compile_opencl_kernel_from_xxd_header(ctx, &ctx->opencl_resources[i], Hoohash_cl, Hoohash_cl_len, "Pepepow_hash", required_extensions, num_required_extensions);
            if (compile_kernel_error != CL_SUCCESS)
            {
              printf("Failed to initialize OpenCL Pepepow kernel, Error code %d.\n", compile_kernel_error);
              return -1;
            }
            printf("Loaded OpenCL Pepepow kernel for device %u\n", ctx->cpu_device_count + i);
          }
        }
      }
    }
    /* CUDA is not supported for PEPEPOW. */
    if (ctx->config->disable_cuda == false &&
        strcmp(algorithm, "pepepow") != 0)
    {
      ctx->cuda_resources = initialize_all_cuda_gpus(&ctx->cuda_device_count, ctx->config->selected_gpus, ctx->config->selected_gpus_num);
      if (ctx->cuda_resources != NULL && ctx->config->list_gpus == false)
      {
        printf("CUDA devices found: %d\n", ctx->cuda_device_count);
        if (strcmp(algorithm, "hoohash") == 0)
        {
          for (cl_uint i = 0; i < ctx->cuda_device_count; i++)
          {
            char cubin_filename[128];
            int major = ctx->cuda_resources[i].device_prop.major;
            int minor = ctx->cuda_resources[i].device_prop.minor;
            printf("Device %u: %s, Compute capability %d.%d\n", i, ctx->cuda_resources[i].device_name, major, minor);
            int arch_code = major * 10 + minor;
            snprintf(cubin_filename, sizeof(cubin_filename), "%s/cubins/hoohash_sm_%d%d.cubin", exe_dir, major, minor);
            printf("Looking for cubin file: %s\n", cubin_filename);
            FILE *file_check = fopen(cubin_filename, "rb");
            if (!file_check)
            {
              printf("Error: Cannot open %s: %s\n", cubin_filename, strerror(errno));
              return -1;
            }
            fclose(file_check);
            int supported_archs[] = {50, 52, 60, 61, 70, 75, 80, 86, 89, 90, 100, 110, 120};
            int supported = 0;
            for (long unsigned int j = 0; j < sizeof(supported_archs) / sizeof(supported_archs[0]); j++)
            {
              if (arch_code == supported_archs[j])
              {
                supported = 1;
                break;
              }
            }
            if (!supported)
            {
              printf("Unsupported compute capability %d.%d for device %u\n", major, minor, i);
              return -1;
            }
            bool compile_kernel_error = load_cuda_kernel_binary(&ctx->cuda_resources[i], cubin_filename, "Hoohash_hash", ctx->config->gpu_work_multiplier);
            if (!compile_kernel_error)
            {
              printf("Failed to load CUDA kernel binary %s for device %u\n", cubin_filename, i);
              return -1;
            }
            printf("Loaded CUDA Hoohash kernel %s for device %u\n", cubin_filename, i);
          }
        }
      }
    }
  }
  return 0;
}

void initialize_reporting_devices(StratumContext *ctx)
{
  if (ctx->config->disable_cpu == false)
  {
    ReportingDevice *cpu_reporting_device = init_reporting_device(0, "CPU");
    add_reporting_device(ctx->hd, cpu_reporting_device);
  }
  if (ctx->config->disable_gpu == false)
  {
    if (ctx->config->disable_opencl == false)
    {
      for (unsigned int i = 0; i < ctx->opencl_device_count; i++)
      {
        char device_name[64];
        snprintf(device_name, sizeof(device_name), "GPU[BUS_ID: %d]", ctx->opencl_resources[i].pci_bus_id);
        ReportingDevice *opencl_reporting_device = init_reporting_device(ctx->cpu_device_count + i, device_name);
        add_reporting_device(ctx->hd, opencl_reporting_device);
      }
    }
    if (ctx->config->disable_cuda == false)
    {
      for (unsigned int i = 0; i < ctx->cuda_device_count; i++)
      {
        char device_name[64];
        snprintf(device_name, sizeof(device_name), "GPU[BUS_ID: %d]", ctx->cuda_resources[i].pci_bus_id);
        ReportingDevice *cuda_reporting_device = init_reporting_device(ctx->cpu_device_count + ctx->opencl_device_count + i, device_name);
        add_reporting_device(ctx->hd, cuda_reporting_device);
      }
    }
  }
}

int main(int argc, char **argv)
{
  signal(SIGINT, cleanup);
  signal(SIGTERM, cleanup);
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);
#endif

  struct HoominerConfig *config = malloc(sizeof(struct HoominerConfig));
  config->username = "user";
  config->password = "x";
  config->algorithm = "hoohash";

  // Initialize context
  ctx = init_stratum_context();
  if (!ctx)
  {
    printf("Failed to allocate StratumContext\n");
    free(config);
    return 1;
  }
  ctx->config = config;
  ctx->version = "0.3.4";
  printf("Welcome to Hoominer v%s\n", ctx->version);

  // Parse arguments
  parse_args(argc, argv, config);
  // Run a quick endianness self-test on Windows to aid debugging
  self_test_hash_endianness();
  if (config->debug == 1)
  {
    if (strcmp(config->algorithm, "pepepow") == 0)
    {
      printf("[self-test] Running PEPEPOW test vectors...\n");
      self_test_pepepow();
    }
    else
    {
      printf("[self-test] Running hoohash test vectors...\n");
      self_test_from_main_vectors();
    }
  }

  if (config->list_gpus == false)
  {
    if (!config->username)
    {
      printf("--username required.\n");
      cleanup(1);
      return 1;
    }
  }

  ctx->ms = init_mining_state();
  if (!ctx->ms)
  {
    printf("Failed to initialize mining state\n");
    cleanup(1);
    return 1;
  }
  ctx->cpu_device_count = 0;
  if (ctx->config->disable_cpu == 0)
  {
    ctx->cpu_device_count = 1;
  }
  int display_devices_length = ctx->cpu_device_count + ctx->opencl_device_count + ctx->cuda_device_count;
  ctx->hd = init_hashrate_display(display_devices_length);
  if (!ctx->hd)
  {
    printf("Failed to initialize hashrate display\n");
    cleanup(1);
    return 1;
  }
  // printf("Initialized Hashrate calculation for %d\n", display_devices_length);

  // Get executable's directory
  char exe_dir_buf[1024] = {0};
#ifdef _WIN32
  get_exe_dir(exe_dir_buf, sizeof(exe_dir_buf));
  char *exe_dir = exe_dir_buf;
#else
  exe_path = strdup(argv[0]);
  if (!exe_path)
  {
    fprintf(stderr, "Failed to allocate memory for executable path\n");
    cleanup(1);
    return 1;
  }
  char *exe_dir = dirname(exe_path);
#endif
  printf("Executable directory: %s\n", exe_dir);

  // Initialize mining resources
  apply_opencl_cache_env(config);
  if (initialize_mining(ctx, config->username, config->algorithm, exe_dir) != 0)
  {
    printf("Failed to initialize mining resources.\n");
    cleanup(1);
    return 1;
  }

  if (config->list_gpus == true)
  {
    list_gpus(ctx);
    cleanup(1);
    return 1;
  }

  initialize_reporting_devices(ctx);

  struct MHD_Daemon *daemon = NULL;
  if (config->api_enabled)
  {
    daemon = start_api(ctx, config);
  }

  // Main loop with reconnection logic and timeout
  time_t reconnect_start_time = 0;
  const int RECONNECT_TIMEOUT = 30; // 30 seconds timeout

  while (true)
  {
    ctx->running = true;
    if (start_stratum_connection(ctx, config) == 0)
    {
      // Connection successful, reset reconnect timer
      reconnect_start_time = 0;
      // Run until disconnection
      while (ctx->running)
      {
        sleep_ms(100); // Check running status periodically
      }
    }

    // Disconnection detected or initialization failed
    printf("Disconnected from stratum server. Reconnecting...\n");
    cleanup_stratum_connection(ctx);
    if (reconnect_start_time == 0)
    {
      reconnect_start_time = time(NULL);
    }
    else if (time(NULL) - reconnect_start_time >= RECONNECT_TIMEOUT)
    {
      printf("Reconnection timeout after %d seconds. Terminating miner.\n", RECONNECT_TIMEOUT);
      cleanup(1);
      return 1;
    }

    printf("Reconnecting in 0.1 second...\n");
    sleep_ms(100);
  }
  if (daemon)
  {
    stop_api(daemon);
  }
  cleanup(0); // Unreachable due to infinite loop, kept for completeness
  return 0;
}