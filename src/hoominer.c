#define CL_TARGET_OPENCL_VERSION 200
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <libgen.h>
#include <pthread.h>
#include <stdbool.h>
#include <time.h> // Added for time tracking
#include "config.h"
#include "opencl-host.h"
#include "cuda-host.h"
#include "stratum.h"
#include "reporting.h"
#include "miner-hoohash.h"
#include "hoohash_cl.h"
#include "api.h"

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

void cleanup(int sig)
{
  printf("Cleanup initiated due to signal %d\n", sig);

  if (ctx == NULL)
  {
    free(exe_path);
    fflush(stdout);
    exit(sig == SIGINT || sig == SIGTERM ? EXIT_FAILURE : EXIT_SUCCESS);
  }

  // Signal all threads to stop
  ctx->running = false;

  // Cancel and join threads
  if (ctx->hd && ctx->hd->display_thread)
  {
    pthread_cancel(ctx->hd->display_thread);
    pthread_join(ctx->hd->display_thread, NULL);
  }

  if (ctx->recv_thread)
  {
    pthread_cancel(ctx->recv_thread);
    pthread_join(ctx->recv_thread, NULL);
  }

  if (ctx->ms)
  {
    // Stop mining threads
    if (ctx->ms->mining_cpu_threads)
    {
      for (int i = 0; i < ctx->ms->num_cpu_threads; i++)
      {
        if (ctx->ms->mining_cpu_threads[i])
        {
          pthread_cancel(ctx->ms->mining_cpu_threads[i]);
          pthread_join(ctx->ms->mining_cpu_threads[i], NULL);
        }
      }
      free(ctx->ms->mining_cpu_threads);
      ctx->ms->mining_cpu_threads = NULL;
    }

    if (ctx->ms->mining_opencl_threads)
    {
      for (int i = 0; i < ctx->ms->num_opencl_threads; i++)
      {
        if (ctx->ms->mining_opencl_threads[i])
        {
          pthread_cancel(ctx->ms->mining_opencl_threads[i]);
          pthread_join(ctx->ms->mining_opencl_threads[i], NULL);
        }
      }
      free(ctx->ms->mining_opencl_threads);
      ctx->ms->mining_opencl_threads = NULL;
    }

    if (ctx->ms->mining_cuda_threads)
    {
      for (int i = 0; i < ctx->ms->num_cuda_threads; i++)
      {
        if (ctx->ms->mining_cuda_threads[i])
        {
          pthread_cancel(ctx->ms->mining_cuda_threads[i]);
          pthread_join(ctx->ms->mining_cuda_threads[i], NULL);
        }
      }
      free(ctx->ms->mining_cuda_threads);
      ctx->ms->mining_cuda_threads = NULL;
    }

    // Clean up job queueif (ctx->hd && ctx->hd->display_thread)
    {
      // printf("Cleaning display thread\n");
      pthread_cancel(ctx->hd->display_thread);
      pthread_join(ctx->hd->display_thread, NULL);
      ctx->hd->display_thread = 0;
    }
    if (ctx->recv_thread)
    {
      // printf("Cleaning receive thread\n");
      pthread_cancel(ctx->recv_thread);
      pthread_join(ctx->recv_thread, NULL);
      ctx->recv_thread = 0;
    }
    if (ctx->ms && ctx->ms->mining_cpu_threads)
    {
      // printf("Cleaning CPU threads\n");
      for (int i = 0; i < ctx->ms->num_cpu_threads; i++)
      {
        if (ctx->ms->mining_cpu_threads[i])
        {
          pthread_cancel(ctx->ms->mining_cpu_threads[i]);
          pthread_join(ctx->ms->mining_cpu_threads[i], NULL);
          ctx->ms->mining_cpu_threads[i] = 0;
        }
      }
      free(ctx->ms->mining_cpu_threads);
      ctx->ms->mining_cpu_threads = NULL;
    }
    if (ctx->ms && ctx->ms->mining_opencl_threads)
    {
      // printf("Cleaning OpenCL threads\n");
      for (int i = 0; i < ctx->ms->num_opencl_threads; i++)
      {
        if (ctx->ms->mining_opencl_threads[i])
        {
          pthread_cancel(ctx->ms->mining_opencl_threads[i]);
          pthread_join(ctx->ms->mining_opencl_threads[i], NULL);
          ctx->ms->mining_opencl_threads[i] = 0;
        }
      }
      free(ctx->ms->mining_opencl_threads);
      ctx->ms->mining_opencl_threads = NULL;
    }
    if (ctx->ms && ctx->ms->mining_cuda_threads)
    {
      // printf("Cleaning CUDA threads\n");
      for (int i = 0; i < ctx->ms->num_cuda_threads; i++)
      {
        if (ctx->ms->mining_cuda_threads[i])
        {
          pthread_cancel(ctx->ms->mining_cuda_threads[i]);
          pthread_join(ctx->ms->mining_cuda_threads[i], NULL);
          ctx->ms->mining_cuda_threads[i] = 0;
        }
      }
      free(ctx->ms->mining_cuda_threads);
      ctx->ms->mining_cuda_threads = NULL;
    }
    if (ctx->sockfd >= 0)
    {
      // printf("Cleaning sockfd\n");
      close(ctx->sockfd);
      ctx->sockfd = -1;
    }

    // Clear job queue
    if (ctx->ms)
    {
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
    }

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
    pthread_mutex_destroy(&ctx->hd->hashrate_mutex);
    free(ctx->hd);
    ctx->hd = NULL;
  }

  // Close socket
  if (ctx->sockfd >= 0)
  {
    close(ctx->sockfd);
    ctx->sockfd = -1;
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
      printf("OpenCL devices found: %d\n", ctx->opencl_device_count);
      if (ctx->config->list_gpus == false)
      {
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
      }
    }
    if (ctx->config->disable_cuda == false)
    {
      ctx->cuda_resources = initialize_all_cuda_gpus(&ctx->cuda_device_count, ctx->config->selected_gpus, ctx->config->selected_gpus_num);
      printf("CUDA devices found: %d\n", ctx->cuda_device_count);
      if (ctx->config->list_gpus == false)
      {
        if (strcmp(algorithm, "hoohash") == 0)
        {
          for (cl_uint i = 0; i < ctx->cuda_device_count; i++)
          {
            char cubin_filename[128];
            int major = ctx->cuda_resources[i].device_prop.major;
            int minor = ctx->cuda_resources[i].device_prop.minor;
            printf("Device %u: %s, Compute capability %d.%d\n", i, ctx->cuda_resources[i].device_name, major, minor);
            int arch_code = major * 10 + minor;
            snprintf(cubin_filename, sizeof(cubin_filename), "%s/cubins/hoohash_sm%d%d.cubin", exe_dir, major, minor);
            FILE *file_check = fopen(cubin_filename, "rb");
            if (!file_check)
            {
              printf("Error: Cannot open %s: %s\n", cubin_filename, strerror(errno));
              continue;
            }
            fclose(file_check);
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
            bool compile_kernel_error = load_cuda_kernel_binary(&ctx->cuda_resources[i], cubin_filename, "Hoohash_hash");
            if (compile_kernel_error != cudaSuccess)
            {
              printf("Failed to load CUDA kernel binary %s for device %u\n",
                     cubin_filename, i, compile_kernel_error);
              continue;
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
        char *name_copy = strdup(device_name);
        // ReportingDevice *opencl_reporting_device = init_reporting_device(ctx->cpu_device_count + i, ctx->opencl_resources[i].device_name);
        ReportingDevice *opencl_reporting_device = init_reporting_device(ctx->cpu_device_count + i, name_copy);
        add_reporting_device(ctx->hd, opencl_reporting_device);
      }
    }
    if (ctx->config->disable_cuda == false)
    {
      for (unsigned int i = 0; i < ctx->cuda_device_count; i++)
      {
        char device_name[64];
        snprintf(device_name, sizeof(device_name), "GPU[BUS_ID: %d]", ctx->cuda_resources[i].pci_bus_id);
        char *name_copy = strdup(device_name);
        // ReportingDevice *cuda_reporting_device = init_reporting_device(ctx->cpu_device_count + ctx->opencl_device_count + i, ctx->cuda_resources[i].device_name);
        ReportingDevice *cuda_reporting_device = init_reporting_device(ctx->cpu_device_count + ctx->opencl_device_count + i, name_copy);
        add_reporting_device(ctx->hd, cuda_reporting_device);
      }
    }
  }
}

int main(int argc, char **argv)
{
  signal(SIGINT, cleanup);
  signal(SIGTERM, cleanup);
  signal(SIGPIPE, SIG_IGN);

  struct HoominerConfig *config = malloc(sizeof(struct HoominerConfig));
  config->username = "user";
  config->password = "x";
  config->algorithm = "hoohash";

  // Initialize context
  ctx = init_stratum_context();
  if (!ctx)
  {
    printf("Failed to allocate StratumContext\n");
    return 1;
  }
  ctx->config = config;
  ctx->version = "0.3.0";
  printf("Welcome to Hoominer v%s\n", ctx->version);

  // Parse arguments
  parse_args(argc, argv, config);

  if (config->list_gpus == false)
  {
    if (!config->pool_ip)
    {
      printf("--stratum required, could not parse ip of the pool from the stratum address.\n");
      cleanup(1);
      return 1;
    }
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
  exe_path = strdup(argv[0]);
  if (!exe_path)
  {
    fprintf(stderr, "Failed to allocate memory for executable path\n");
    cleanup(1);
    return 1;
  }
  char *exe_dir = dirname(exe_path);

  // Initialize mining resources
  if (initialize_mining(ctx, config->username, config->algorithm, exe_dir) != 0)
  {
    printf("Failed to initialize mining resources.\n");
    cleanup(1);
    return 1;
  }

  if (config->list_gpus == true)
  {
    list_gpus(ctx);
    return 1;
  }

  initialize_reporting_devices(ctx);

  struct MHD_Daemon *daemon = start_api(ctx);

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
        sleep(0.1); // Check running status periodically
      }
    }

    // Disconnection detected or initialization failed
    printf("Disconnected from stratum server. Reconnecting...\n");
    ctx->running = false;

    // Clean up threads and socket
    if (ctx->hd && ctx->hd->display_thread)
    {
      pthread_cancel(ctx->hd->display_thread);
      pthread_join(ctx->hd->display_thread, NULL);
      ctx->hd->display_thread = 0;
    }
    if (ctx->recv_thread)
    {
      pthread_cancel(ctx->recv_thread);
      pthread_join(ctx->recv_thread, NULL);
      ctx->recv_thread = 0;
    }
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
    sleep(1);
  }
  stop_api(daemon);
  cleanup(0); // Unreachable due to infinite loop, kept for completeness
  return 0;
}