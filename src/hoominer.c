
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <pthread.h>
#include <stdbool.h>
#include "opencl.h"
#include "stratum.h"
#include "reporting.h"
#include "hoohash-miner.h"

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
  printf("Cleanup because of %d signal", sig);
  if (ctx->hd != NULL)
  {
    cleanup_hashrate_display(ctx);
  }
  if (ctx->ms != NULL)
  {
    if (ctx->ms->job != NULL)
    {
      cleanup_job(ctx->ms);
    }
    cleanup_mining_threads(ctx->ms);
    cleanup_mining_state(ctx->ms);
  }
  cleanup_stratum_context(ctx);
}

int main(int argc, char **argv)
{
  signal(SIGINT, cleanup);
  signal(SIGPIPE, SIG_IGN);

  char *pool_ip = NULL;
  int pool_port = 5555;
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
    return 1;
  }

  if (!username)
  {
    printf("--username required.\n");
    return 1;
  }
  ctx->worker = username;
  if (ctx->disable_gpu == 0)
  {
    ctx->opencl_resources = initialize_all_gpus(&ctx->opencl_device_count);
    printf("OpenCL devices found %d\n", ctx->opencl_device_count);
    if (!ctx->opencl_resources || ctx->opencl_device_count == 0)
    {
      fprintf(stderr, "Failed to initialize OpenCL GPUs, not yet testing for CUDA GPUs.\n");
      return 1;
    }
    // ctx->opencl_resources = opencl_resources;
    const char *opencl_kernel_filename = NULL;
    const char *opencl_kernel_name = NULL;
    if (strcmp(algorithm, "hoohash") == 0)
    {
      opencl_kernel_filename = "./build/hoohash.bin"; // TODO: change for release
      opencl_kernel_name = "Hoohash_hash";
    }
    if (opencl_kernel_filename != NULL && opencl_kernel_name != NULL)
    {
      for (cl_uint i = 0; i < ctx->opencl_device_count; i++)
      {
        cl_int load_kernel_error = load_kernel_binary(&ctx->opencl_resources[i], opencl_kernel_filename, opencl_kernel_name);
        if (load_kernel_error != CL_SUCCESS)
        {
          printf("Failed to initialize OpenCL kernels, not yet testing for CUDA GPUs. Error code %d.\n", load_kernel_error);

          break;
        }
      }
    }
  }

  while (ctx->running)
  {
    ctx->sockfd = connect_to_stratum_server(pool_ip, pool_port);
    if (ctx->sockfd < 0)
    {
      printf("Failed to connect to stratum server. Retrying in 5 seconds...\n");
      free(pool_ip);
      cleanup(1);
      sleep(5);
      continue;
    }

    if (stratum_subscribe(ctx->sockfd) < 0 || stratum_authenticate(ctx->sockfd, username, password) < 0)
    {
      printf("Stratum initialization failed. Retrying in 5 seconds...\n");
      close(ctx->sockfd);
      free(pool_ip);
      cleanup(2);
      sleep(5);
      continue;
    }

    pthread_t recv_thread, display_thread;
    if (pthread_create(&display_thread, NULL, hashrate_display_thread, ctx) != 0 ||
        pthread_create(&recv_thread, NULL, stratum_receive_thread, ctx) != 0)
    {
      printf("Failed to create threads. Retrying in 5 seconds...\n");
      pthread_cancel(display_thread);
      pthread_join(display_thread, NULL);
      pthread_cancel(recv_thread);
      pthread_join(recv_thread, NULL);
      close(ctx->sockfd);
      free(pool_ip);
      cleanup(3);
      sleep(5);
      continue;
    }

    if (start_mining_threads(ctx, ctx->ms) != 0)
    {
      printf("Failed to start mining threads. Retrying in 5 seconds...\n");
      pthread_cancel(recv_thread);
      pthread_cancel(display_thread);
      pthread_join(recv_thread, NULL);
      pthread_join(display_thread, NULL);
      close(ctx->sockfd);
      free(pool_ip);
      cleanup(4);
      sleep(5);
      continue;
    }

    while (ctx->running)
      sleep(1);

    pthread_cancel(recv_thread);
    pthread_cancel(display_thread);
    pthread_join(recv_thread, NULL);
    pthread_join(display_thread, NULL);
    close(ctx->sockfd);
    free(ctx);
  }

  free(pool_ip);
  cleanup(0);
  return 0;
}