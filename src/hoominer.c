#define CL_TARGET_OPENCL_VERSION 200
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <libgen.h>
#include <pthread.h>
#include <stdbool.h>
#include <cuda.h>
#include "opencl.h"
#include "cuda-host.h"
#include "stratum.h"
#include "reporting.h"
#include "hoohash-miner.h"

#include "hoohash_cl.h"

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

char *pool_ip = NULL;
int pool_port = 5555;
char *exe_path = NULL;

// Utility Functions

void parse_args(int argc, char **argv, char **pool_ip, int *pool_port, const char **username, const char **password,
                const char **algorithm, int *disable_cpu, int *disable_gpu, int *threads)
{
  *disable_cpu = 0;
  *disable_gpu = 0;
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--user") && i + 1 < argc)
      *username = argv[++i];
    else if (!strcmp(argv[i], "--pass") && i + 1 < argc)
      *password = argv[++i];
    else if (!strcmp(argv[i], "--algorithm") && i + 1 < argc)
      *algorithm = argv[++i];
    else if (!strcmp(argv[i], "--disable-cpu"))
      *disable_cpu = 1;
    else if (!strcmp(argv[i], "--disable-gpu"))
      *disable_gpu = 1;
    else if (!strcmp(argv[i], "--cpu-threads") && i + 1 < argc)
      *threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stratum") && i + 1 < argc)
    {
      const char *stratum_url = argv[++i];
      if (strncmp(stratum_url, "stratum+tcp://", 14) != 0)
      {
        fprintf(stderr, "Invalid stratum URL format\n");
        exit(1);
      }

      const char *url_part = stratum_url + 14;
      char *url = malloc(strlen(url_part) + 1);
      if (!url)
      {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
      }
      strcpy(url, url_part);

      char *colon = strchr(url, ':');
      if (!colon)
      {
        free(url);
        fprintf(stderr, "Stratum URL missing port\n");
        exit(1);
      }
      *colon = '\0';

      *pool_ip = malloc(strlen(url) + 1);
      if (!*pool_ip)
      {
        free(url);
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
      }
      strcpy(*pool_ip, url);

      *pool_port = atoi(colon + 1);
      free(url);
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("Usage: %s [--stratum <stratum+tcp://domain:port>] [--user <user>] [--pass <pass>] [--disable-cpu] [--cpu-threads <n>]\n", argv[0]);
      exit(0);
    }
  }
}

StratumContext *ctx = NULL;

void cleanup(int sig)
{
  printf("Cleanup initiated due to signal %d\n", sig);
  fflush(stdout);

  if (ctx != NULL)
  {
    ctx->running = false; // Signal threads to stop

    // Cancel and join all mining threads
    if (ctx->ms)
    {
      cleanup_mining_threads(ctx->ms);
    }

    // Clean up remaining resources
    cleanup_stratum_context(ctx);
    ctx = NULL;
  }

  free(exe_path);
  free(pool_ip);

  if (sig == SIGINT || sig == SIGTERM)
  {
    exit(EXIT_FAILURE);
  }
  else
  {
    exit(EXIT_SUCCESS);
  }
}

int main(int argc, char **argv)
{
  signal(SIGINT, cleanup);
  signal(SIGPIPE, SIG_IGN);

  const char *username = "user";
  const char *password = "x";
  const char *algorithm = "hoohash";

  ctx = init_stratum_context();
  if (!ctx)
  {
    printf("Failed to allocate StratumContext\n");
    return 1;
  }
  ctx->hd = init_hashrate_display();
  ctx->ms = init_mining_state();

  ctx->ms->num_cpu_threads = get_cpu_threads();
  parse_args(argc, argv, &pool_ip, &pool_port, &username, &password, &algorithm, &ctx->disable_cpu, &ctx->disable_gpu, &ctx->ms->num_cpu_threads);
  if (!pool_ip)
  {
    printf("--stratum required, could not parse ip of the pool from the stratum address.\n");
    cleanup_stratum_context(ctx);
    return 1;
  }

  if (!username)
  {
    printf("--username required.\n");
    cleanup_stratum_context(ctx);
    return 1;
  }

  // Get executable's directory
  exe_path = strdup(argv[0]);
  if (!exe_path)
  {
    fprintf(stderr, "Failed to allocate memory for executable path\n");
    free(pool_ip);
    cleanup_stratum_context(ctx);
    return 1;
  }
  char *exe_dir = dirname(exe_path);

  ctx->worker = username;
  if (ctx->disable_gpu == 0)
  {
    ctx->opencl_resources = initalize_all_opencl_gpus(&ctx->opencl_device_count);
    printf("OpenCL devices found %d\n", ctx->opencl_device_count);
    if (strcmp(algorithm, "hoohash") == 0)
    {
      for (cl_uint i = 0; i < ctx->opencl_device_count; i++)
      {
        const char *required_extensions[] = {"cl_khr_fp64"};
        size_t num_required_extensions = 1;

        cl_int compile_kernel_error = compile_opencl_kernel_from_xxd_header(&ctx->opencl_resources[i], Hoohash_cl, Hoohash_cl_len, "Hoohash_hash", required_extensions, num_required_extensions);
        if (compile_kernel_error != CL_SUCCESS)
        {
          printf("Failed to initialize OpenCL kernels, Error code %d.\n", compile_kernel_error);
          break;
        }
        printf("Loaded OpenCL Hoohash kernel for device %u\n", i);
      }
    }
    ctx->cuda_resources = initialize_all_cuda_gpus(&ctx->cuda_device_count);
    printf("CUDA devices found %d\n", ctx->cuda_device_count);
    if (strcmp(algorithm, "hoohash") == 0)
    {
      for (cl_uint i = 0; i < ctx->cuda_device_count; i++)
      {
        char cubin_filename[128];
        int major = ctx->cuda_resources[i].device_prop.major;
        int minor = ctx->cuda_resources[i].device_prop.minor;
        printf("Device %u: %s, Compute capability %d.%d\n", i, ctx->cuda_resources[i].device_name, major, minor);

        // Map compute capability to .cubin filename
        int arch_code = major * 10 + minor;
        snprintf(cubin_filename, sizeof(cubin_filename), "%s/cubins/hoohash_sm%d%d.cubin", exe_dir, major, minor);

        // Check if the .cubin file exists
        FILE *file_check = fopen(cubin_filename, "rb");
        if (!file_check)
        {
          printf("Error: Cannot open %s: %s\n", cubin_filename, strerror(errno));
          continue;
        }
        fclose(file_check);

        // List of supported architectures
        int supported_archs[] = {50, 52, 60, 61, 70, 75, 80, 86, 89, 90, 100};
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
          continue;
        }

        // Load the .cubin file
        cudaError_t compile_kernel_error = load_cuda_kernel_binary(&ctx->cuda_resources[i], cubin_filename, "Hoohash_hash");
        if (compile_kernel_error != cudaSuccess)
        {
          const char *err_str;
          cuGetErrorString((CUresult)compile_kernel_error, &err_str);
          printf("Failed to load CUDA kernel binary %s for device %u: Error code %d (%s)\n",
                 cubin_filename, i, compile_kernel_error, err_str ? err_str : "Unknown");
          continue;
        }
        printf("Loaded CUDA Hoohash kernel %s for device %u\n", cubin_filename, i);
      }
    }
  }

  pthread_t recv_thread, display_thread;
  while (ctx->running)
  {
    ctx->sockfd = connect_to_stratum_server(pool_ip, pool_port);
    if (ctx->sockfd < 0)
    {
      printf("Failed to connect to stratum server. Retrying in 5 seconds...\n");
      sleep(5);
      continue;
    }

    if (stratum_subscribe(ctx->sockfd) < 0 || stratum_authenticate(ctx->sockfd, username, password) < 0)
    {
      printf("Stratum initialization failed. Retrying in 5 seconds...\n");
      close(ctx->sockfd);
      ctx->sockfd = -1;
      sleep(5);
      continue;
    }

    if (pthread_create(&display_thread, NULL, hashrate_display_thread, ctx) != 0 ||
        pthread_create(&recv_thread, NULL, stratum_receive_thread, ctx) != 0)
    {
      printf("Failed to create display and receive threads.\n");
      close(ctx->sockfd);
      ctx->sockfd = -1;
      cleanup_stratum_context(ctx);
      return 1;
    }

    if (start_mining_threads(ctx, ctx->ms) != 0)
    {
      printf("Failed to start mining threads.\n");
      pthread_cancel(recv_thread);
      pthread_cancel(display_thread);
      pthread_join(recv_thread, NULL);
      pthread_join(display_thread, NULL);
      close(ctx->sockfd);
      ctx->sockfd = -1;
      cleanup_stratum_context(ctx);
      return 1;
    }

    // Wait for threads to complete
    pthread_join(recv_thread, NULL);
    pthread_join(display_thread, NULL);
  }

  cleanup(0);
  return 0;
}