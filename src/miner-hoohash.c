#include "miner-hoohash.h"
#include <time.h>
#include <inttypes.h>
#include "platform_compat.h"

#ifdef _WIN32
#include <windows.h>
// Simple CLOCK_MONOTONIC shim using QPC
static void clock_gettime_monotonic(struct timespec *ts)
{
  static LARGE_INTEGER freq = {0};
  LARGE_INTEGER now;
  if (!freq.QuadPart)
  {
    QueryPerformanceFrequency(&freq);
  }
  QueryPerformanceCounter(&now);
  long double seconds = (long double)now.QuadPart / (long double)freq.QuadPart;
  ts->tv_sec = (time_t)seconds;
  ts->tv_nsec = (long)((seconds - ts->tv_sec) * 1e9);
}
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#define clock_gettime(id, ts) clock_gettime_monotonic(ts)

// endian helpers for Windows
static inline uint64_t __byte_swap_u64(uint64_t x)
{
  return ((x & 0x00000000000000FFULL) << 56) |
         ((x & 0x000000000000FF00ULL) << 40) |
         ((x & 0x0000000000FF0000ULL) << 24) |
         ((x & 0x00000000FF000000ULL) << 8) |
         ((x & 0x000000FF00000000ULL) >> 8) |
         ((x & 0x0000FF0000000000ULL) >> 24) |
         ((x & 0x00FF000000000000ULL) >> 40) |
         ((x & 0xFF00000000000000ULL) >> 56);
}
static inline uint64_t htole64(uint64_t x)
{
  // Windows is little-endian; keep x
  return x;
}
static inline uint64_t le64toh(uint64_t x)
{
  return x;
}
#endif

MiningState *init_mining_state()
{
  MiningState *state = calloc(1, sizeof(MiningState));
  if (!state)
  {
    fprintf(stderr, "Failed to allocate memory for MiningState\n");
    exit(1);
  }
  state->num_cpu_threads = 0;
  state->num_opencl_threads = 0;
  state->num_cuda_threads = 0;
  state->global_target = NULL;
  state->extranonce = NULL;
  state->job = NULL;
  state->mining_cpu_threads = NULL;
  state->mining_opencl_threads = NULL;
  state->mining_cuda_threads = NULL;
  state->new_job_available = 0;
  pthread_mutex_init(&state->job_mutex, NULL);
  pthread_mutex_init(&state->target_mutex, NULL);

  state->job_queue.head = 0;
  state->job_queue.tail = 0;
  pthread_mutex_init(&state->job_queue.queue_mutex, NULL);
  pthread_cond_init(&state->job_queue.queue_cond, NULL);
  for (int i = 0; i < JOB_QUEUE_SIZE; i++)
  {
    state->job_queue.jobs[i].job_id = NULL;
    state->job_queue.jobs[i].running = 0;
    state->job_queue.jobs[i].completed = 0;
  }

  return state;
}

void uint64_to_little_endian(uint64_t value, uint8_t *buffer)
{
  value = htole64(value);
  memcpy(buffer, &value, sizeof(uint64_t));
}

uint64_t little_endian_to_uint64(const uint8_t *buffer)
{
  uint64_t value;
  memcpy(&value, buffer, sizeof(uint64_t));
  return le64toh(value);
}

void smallJobHeader(const uint64_t *ids, uint8_t *headerData)
{
  for (size_t i = 0; i < 4; i++)
  {
    uint64_to_little_endian(ids[i], headerData + i * 8);
  }
}

int hex_to_bytes(const char *hex, uint8_t *bytes, size_t len)
{
  if (strlen(hex) != len * 2)
    return -1;
  for (size_t i = 0; i < len; i++)
  {
    if (sscanf(hex + i * 2, "%2hhx", &bytes[i]) != 1)
      return -1;
  }
  return 0;
}

void print_hex(const char *label, const uint8_t *data, size_t len)
{
  printf("%s: 0x", label);
  for (size_t i = 0; i < len; i++)
    printf("%02x", data[i]);
  printf("\n");
}

int submit_mining_solution(int sockfd, const char *worker, const char *job_id, uint64_t nonce, uint8_t *hash, MiningState *ms, StratumContext *ctx, int reporting_index)
{
  pthread_mutex_lock(&ms->job_queue.queue_mutex);
  for (int i = 0; i < JOB_QUEUE_SIZE; i++)
  {
    int idx = (ms->job_queue.head + i) % JOB_QUEUE_SIZE;
    if (ms->job_queue.jobs[idx].job_id && strcmp(ms->job_queue.jobs[idx].job_id, job_id) == 0)
    {
      ms->job_queue.jobs[idx].completed = 1;
      ms->job_queue.jobs[idx].running = 0;
      if (idx == ms->job_queue.head)
        ms->job_queue.head = (ms->job_queue.head + 1) % JOB_QUEUE_SIZE;
      break;
    }
  }
  pthread_cond_broadcast(&ms->job_queue.queue_cond);
  pthread_mutex_unlock(&ms->job_queue.queue_mutex);

  json_object *req = json_object_new_object();
  json_object_object_add(req, "id", json_object_new_int(1));
  json_object_object_add(req, "method", json_object_new_string("mining.submit"));

  json_object *params = json_object_new_array();
  json_object_array_add(params, json_object_new_string(worker));
  json_object_array_add(params, json_object_new_string(job_id));
  char nonce_hex[20];
  snprintf(nonce_hex, sizeof(nonce_hex), "%016" PRIx64, nonce);
  json_object_array_add(params, json_object_new_string(nonce_hex));
  char *hash_hex = encodeHex(hash, DOMAIN_HASH_SIZE);
  json_object_array_add(params, json_object_new_string(hash_hex));
  json_object_object_add(req, "params", params);

  printf("Solution found, Nonce: %" PRIu64 ", PoW hash: %s\n", (uint64_t)nonce, hash_hex);

  const char *msg = json_object_to_json_string_ext(req, JSON_C_TO_STRING_PLAIN);
  if (!msg)
  {
    free(hash_hex);
    json_object_put(req);
    return -1;
  }

  static char submission_buffer[4096];
  // struct timespec start, end;
  // clock_gettimeCLOCK_MONOTONIC, &start);
  snprintf(submission_buffer, sizeof(submission_buffer), "%s\n", msg);
  int ret = send(sockfd, submission_buffer, strlen(submission_buffer), 0);
  // clock_gettime(CLOCK_MONOTONIC, &end);
  // double latency = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  // printf("Share submission latency: %.3f ms\n", latency * 1000);

  free(hash_hex);
  json_object_put(req);
  enqueue_int_fifo(&ctx->mining_submit_fifo, reporting_index);
  return ret < 0 ? -1 : 0;
}

static int get_current_job(MiningState *ms, QueuedJob *job, char **current_job_id)
{
  pthread_mutex_lock(&ms->job_queue.queue_mutex);
  if (ms->job_queue.head == ms->job_queue.tail)
  {
    // Queue is empty
    pthread_mutex_unlock(&ms->job_queue.queue_mutex);
    return 0;
  }
  // Get the most recent job (at tail - 1)
  int tail_idx = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
  *job = ms->job_queue.jobs[tail_idx];
  if (*current_job_id == NULL || strcmp(*current_job_id, job->job_id) != 0)
  {
    free(*current_job_id);
    *current_job_id = strdup(job->job_id);
    if (!*current_job_id)
    {
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);
      fprintf(stderr, "Memory allocation failed for job_id\n");
      return 0;
    }
  }
  int job_valid = job->running && !job->completed;
  pthread_mutex_unlock(&ms->job_queue.queue_mutex);
  return job_valid;
}

void *mining_cpu_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  const int thread_index = mt->threadIndex;
  free(mt);
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;
  int reporting_index = 0;
  if (reporting_index >= ctx->hd->device_count)
  {
    fprintf(stderr, "CPU reporting index %d exceeds device count %d\n", reporting_index, ctx->hd->device_count);
    return NULL;
  }
  ReportingDevice *cpu_reporting_device = ctx->hd->devices[reporting_index];

  uint64_t nonce = (uint64_t)thread_index;
  uint64_t step = ms->num_cpu_threads;

  while (ctx->running)
  {

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      sleep_ms(100);
      continue;
    }

    memcpy(state.PrevHeader, current_job.header, DOMAIN_HASH_SIZE);
    state.Timestamp = current_job.timestamp;
    memcpy(state.mat, current_job.matrix, sizeof(double) * 64 * 64);
    struct timespec start_time, end_time;
    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &start_time);
      printf("Starting job %s\n", current_job.job_id);
    }

    int nonces_processed_for_job = 0;

    while (ctx->running)
    {

      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      // Check if the current job is still valid
      int job_valid = 0;
      int tail_idx = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
      if (ms->job_queue.head != ms->job_queue.tail &&
          ms->job_queue.jobs[tail_idx].job_id &&
          strcmp(ms->job_queue.jobs[tail_idx].job_id, current_job_id) == 0 &&
          ms->job_queue.jobs[tail_idx].running &&
          !ms->job_queue.jobs[tail_idx].completed)
      {
        job_valid = 1;
      }
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      state.Nonce = nonce;
      uint8_t result[DOMAIN_HASH_SIZE];
      CalculateProofOfWorkValue(&state, result);

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      cpu_reporting_device->nonces_processed++;
      nonces_processed_for_job++;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      pthread_mutex_lock(&ms->target_mutex);
      int meets_target = compare_target(result, ms->global_target, DOMAIN_HASH_SIZE);
      pthread_mutex_unlock(&ms->target_mutex);

      if (meets_target <= 0)
      {
        // struct timespec now;
        // clock_gettime(CLOCK_MONOTONIC, &now);
        // uint64_t current_time_ms = now.tv_sec * 1000ULL + now.tv_nsec / 1000000ULL;
        // if (current_job.timestamp * 1000ULL + JOB_MAX_AGE > current_time_ms)
        // {
        if (ctx->config->debug == 1)
          printf("Submitting solution for job %s", current_job_id);
        submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, state.Nonce, result, ms, ctx, reporting_index);
        current_job.completed = 1;
        current_job.running = 0;
        break;
        // }
      }
      nonce += step;
    }
    current_job.running = 0;
    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &end_time);
      long elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000L +
                        (end_time.tv_nsec - start_time.tv_nsec);
      double elapsed_ms = elapsed_ns / 1e6;
      printf("Job %s: runtime: %.3f ms, and nonces processed %d\n", current_job_id, elapsed_ms, nonces_processed_for_job);
    }
  }

  free(current_job_id);
  return NULL;
}

void *mining_opencl_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  const int thread_index = mt->threadIndex;
  free(mt);
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;
  cl_ulong local_work_size = ctx->opencl_resources[thread_index].max_work_group_size;
  cl_ulong global_work_size = ctx->opencl_resources[thread_index].max_global_work_size;
  uint64_t random_base = rand() & 0x3FFFF;
  // Prevent integer overflow in multiplication
  cl_ulong thread_work_size = (cl_ulong)thread_index * global_work_size;
  if (thread_work_size > UINT64_MAX / random_base)
  {
    thread_work_size = UINT64_MAX / (random_base + 1);
  }
  unsigned long long start_nonce = random_base * thread_work_size;
  if (ms->extranonce != NULL)
  {
    cl_ulong extranonce_val = strtoull(ms->extranonce, NULL, 10);
    start_nonce = (extranonce_val << 32) | start_nonce;
  }
  int reporting_index = ctx->cpu_device_count + thread_index;
  if (reporting_index >= ctx->hd->device_count)
  {
    fprintf(stderr, "OpenCL reporting index %d exceeds device count %d\n", reporting_index, ctx->hd->device_count);
    return NULL;
  }
  ReportingDevice *opencl_reporting_device = ctx->hd->devices[reporting_index];

  while (ctx->running)
  {

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      sleep_ms(100);
      continue;
    }

    memcpy(state.PrevHeader, current_job.header, DOMAIN_HASH_SIZE);
    state.Timestamp = current_job.timestamp;
    memcpy(state.mat, current_job.matrix, sizeof(double) * 64 * 64);
    struct timespec start_time, end_time;
    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &start_time);
      printf("Starting job %s\n", current_job_id);
    }
    cl_ulong nonces_processed_for_job = 0;

    while (ctx->running)
    {

      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      // Check if the current job is still valid
      int job_valid = 0;
      int tail_idx = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
      if (ms->job_queue.head != ms->job_queue.tail &&
          ms->job_queue.jobs[tail_idx].job_id &&
          strcmp(ms->job_queue.jobs[tail_idx].job_id, current_job_id) == 0 &&
          ms->job_queue.jobs[tail_idx].running &&
          !ms->job_queue.jobs[tail_idx].completed)
      {
        job_valid = 1;
      }
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      OpenCLResult result = {0};
      cl_int status = run_opencl_hoohash_kernel(&ctx->opencl_resources[thread_index], global_work_size,
                                                local_work_size, state.PrevHeader, ms->global_target, state.mat, state.Timestamp, start_nonce, &result);

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      opencl_reporting_device->nonces_processed += global_work_size;
      nonces_processed_for_job += global_work_size;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      if (status != CL_SUCCESS)
      {
        if (status == -54)
        {
          fprintf(stderr, "Device %d: Kernel execution failed: CL_INVALID_WORK_GROUP_SIZE. Reducing local work size by half.\n", thread_index);
          local_work_size /= 2;
          break;
        }
        else
        {
          fprintf(stderr, "Device %d: Kernel execution failed: %d\n", thread_index, status);
          start_nonce += global_work_size;
          break;
        }
      }

      if (result.nonce != 0)
      {
        pthread_mutex_lock(&ms->target_mutex);
        int meets_target = compare_target(result.hash, ms->global_target, DOMAIN_HASH_SIZE);
        pthread_mutex_unlock(&ms->target_mutex);

        if (meets_target <= 0)
        {
          // uint8_t cpu_result[DOMAIN_HASH_SIZE];
          // state.Nonce = result.nonce;
          // CalculateProofOfWorkValue(&state, cpu_result);
          // if (memcmp(cpu_result, result.hash, DOMAIN_HASH_SIZE) != 0) {
          //     printf("Warning: Mismatch between CPU and GPU result for nonce %" PRIu64 "\n", result.nonce);
          //     print_hex("CPU Result", cpu_result, DOMAIN_HASH_SIZE);
          //     print_hex("GPU Result", result.hash, DOMAIN_HASH_SIZE);
          // } else {
          //   struct timespec now;
          //   clock_gettime(CLOCK_MONOTONIC, &now);
          //   uint64_t current_time_ms = now.tv_sec * 1000ULL + now.tv_nsec / 1000000ULL;
          //   if (current_job.timestamp * 1000ULL + JOB_MAX_AGE > current_time_ms)
          //   {
          if (ctx->config->debug == 1)
            printf("Submitting solution for job %s", current_job_id);
          submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, result.nonce, result.hash, ms, ctx, reporting_index);
          current_job.completed = 1;
          current_job.running = 0;
          break;
          //   }
          // }
        }
      }
      start_nonce += global_work_size;
    }
    current_job.running = 0;
    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &end_time);
      long elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000L +
                        (end_time.tv_nsec - start_time.tv_nsec);
      double elapsed_ms = elapsed_ns / 1e6;
      printf("Job %s: runtime: %.3f ms, and nonces processed %lu\n", current_job_id, elapsed_ms, nonces_processed_for_job);
    }
  }

  free(current_job_id);
  return NULL;
}

void *mining_cuda_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  const int thread_index = mt->threadIndex;
  free(mt);
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;

  // Validate thread index and CUDA resources
  if (!ctx || !ctx->cuda_resources || thread_index >= (int)ctx->cuda_device_count)
  {
    fprintf(stderr, "Invalid CUDA thread parameters: ctx=%p, threadIndex=%d, device_count=%u\n",
            ctx, thread_index, ctx ? ctx->cuda_device_count : 0);
    return NULL;
  }

  // Prevent integer overflow in multiplication
  size_t grid_size = ctx->cuda_resources[thread_index].optimal_grid_size;
  size_t block_size = ctx->cuda_resources[thread_index].optimal_block_size;
  if (grid_size > UINT64_MAX / block_size)
  {
    grid_size = UINT64_MAX / (block_size + 1);
  }
  unsigned long long hashes_per_cuda_call = grid_size * block_size;

  // Prevent overflow in thread multiplication
  if (thread_index > 0 && hashes_per_cuda_call > UINT64_MAX / (uint64_t)thread_index)
  {
    hashes_per_cuda_call = UINT64_MAX / ((uint64_t)thread_index + 1);
  }
  unsigned long long start_nonce = (uint64_t)thread_index * hashes_per_cuda_call;
  if (ms->extranonce != NULL)
  {
    uint64_t extranonce_val = strtoull(ms->extranonce, NULL, 10);
    start_nonce = (extranonce_val << 32) | ((uint64_t)thread_index * hashes_per_cuda_call);
  }
  int reporting_index = ctx->cpu_device_count + ctx->opencl_device_count + thread_index;
  if (reporting_index >= ctx->hd->device_count || reporting_index < 0)
  {
    fprintf(stderr, "CUDA reporting index %d exceeds device count %d or is negative\n", reporting_index, ctx->hd->device_count);
    return NULL;
  }

  // Additional validation for reporting device
  if (!ctx->hd->devices || !ctx->hd->devices[reporting_index])
  {
    fprintf(stderr, "Invalid reporting device at index %d\n", reporting_index);
    return NULL;
  }

  ReportingDevice *cuda_reporting_device = ctx->hd->devices[reporting_index];
  cuda_reporting_device->nonces_processed = 0;

  while (ctx->running)
  {

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      sleep_ms(100);
      continue;
    }

    memcpy(state.PrevHeader, current_job.header, DOMAIN_HASH_SIZE);
    state.Timestamp = current_job.timestamp;
    memcpy(state.mat, current_job.matrix, sizeof(double) * 64 * 64);
    struct timespec start_time, end_time;
    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &start_time);
      printf("Starting job %s\n", current_job_id);
    }
    int nonces_processed_for_job = 0;

    while (ctx->running)
    {
      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      // Check if the current job is still valid
      int job_valid = 0;
      int tail_idx = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
      if (ms->job_queue.head != ms->job_queue.tail &&
          ms->job_queue.jobs[tail_idx].job_id &&
          strcmp(ms->job_queue.jobs[tail_idx].job_id, current_job_id) == 0 &&
          ms->job_queue.jobs[tail_idx].running &&
          !ms->job_queue.jobs[tail_idx].completed)
      {
        job_valid = 1;
      }
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      CudaResult result = {0};
      int error = run_cuda_hoohash_kernel(&ctx->cuda_resources[thread_index],
                                          state.PrevHeader, ms->global_target, state.mat, (long long)state.Timestamp,
                                          start_nonce, &result);

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      cuda_reporting_device->nonces_processed += hashes_per_cuda_call;
      nonces_processed_for_job += hashes_per_cuda_call;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      if (error != 0)
      {
        fprintf(stderr, "Device %d: Kernel execution failed: %d\n", thread_index, error);
        start_nonce += hashes_per_cuda_call;
        break;
      }

      if (result.nonce != 0)
      {
        pthread_mutex_lock(&ms->target_mutex);
        int meets_target = compare_target(result.hash, ms->global_target, DOMAIN_HASH_SIZE);
        pthread_mutex_unlock(&ms->target_mutex);

        if (meets_target <= 0)
        {
          uint8_t cpu_result[DOMAIN_HASH_SIZE];
          state.Nonce = result.nonce;
          CalculateProofOfWorkValue(&state, cpu_result);
          if (memcmp(cpu_result, result.hash, DOMAIN_HASH_SIZE) != 0)
          {
            printf("Warning: Mismatch between CPU and GPU result for nonce %" PRIu64 "\n", result.nonce);
            print_hex("CPU Result", cpu_result, DOMAIN_HASH_SIZE);
            print_hex("GPU Result", result.hash, DOMAIN_HASH_SIZE);
          }
          else
          {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            uint64_t current_time_ms = now.tv_sec * 1000ULL + now.tv_nsec / 1000000ULL;
            if (current_job.timestamp * 1000ULL + JOB_MAX_AGE > current_time_ms)
            {
              if (ctx->config->debug == 1)
                printf("Submitting solution for job %s", current_job_id);
              submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, result.nonce, result.hash, ms, ctx, reporting_index);
              current_job.completed = 1;
              current_job.running = 0;
              break;
            }
          }
        }
      }

      start_nonce += hashes_per_cuda_call;
    }
    current_job.running = 0;

    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &end_time);
      long elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000L +
                        (end_time.tv_nsec - start_time.tv_nsec);
      double elapsed_ms = elapsed_ns / 1e6;
      printf("Job %s: runtime: %.3f ms, and nonces processed %d\n", current_job_id, elapsed_ms, nonces_processed_for_job);
    }
  }

  free(current_job_id);
  return NULL;
}

int start_mining_threads(StratumContext *ctx, MiningState *ms)
{
  if (ctx->config->disable_cpu == false)
  {
    ms->mining_cpu_threads = malloc(ms->num_cpu_threads * sizeof(pthread_t));
    if (!ms->mining_cpu_threads)
    {
      printf("start_mining_threads: Failed to allocate mining_threads\n");
      return 1;
    }
    if (ctx->config->debug == 1)
      printf("Starting %d CPU threads\n", ms->num_cpu_threads);
    for (int i = 0; i < ms->num_cpu_threads; i++)
    {
      MiningThread *mt = malloc(sizeof(MiningThread));
      if (!mt)
      {
        printf("start_mining_threads: Failed to allocate MiningThread\n");
        return 1;
      }
      mt->threadIndex = i;
      mt->ctx = ctx;
      if (pthread_create(&ms->mining_cpu_threads[i], NULL, mining_cpu_thread, mt) != 0)
      {
        printf("start_mining_threads: Failed to create thread %d\n", i);
        free(mt);
        return 1;
      }
    }
  }
  if (ctx->config->disable_gpu == false)
  {
    if (ctx->config->disable_opencl == false)
    {
      ms->num_opencl_threads = ctx->opencl_device_count;
      ms->mining_opencl_threads = malloc(ms->num_opencl_threads * sizeof(pthread_t));
      if (!ms->mining_opencl_threads)
      {
        printf("start_mining_threads: Failed to allocate mining_threads\n");
        return 1;
      }
      if (ctx->config->debug == 1)
        printf("Starting %d OpenCL threads\n", ms->num_opencl_threads);
      for (int i = 0; i < ms->num_opencl_threads; i++)
      {
        MiningThread *mt = malloc(sizeof(MiningThread));
        if (!mt)
        {
          printf("start_mining_threads: Failed to allocate MiningThread\n");
          return 1;
        }
        mt->threadIndex = i;
        mt->ctx = ctx;
        if (pthread_create(&ms->mining_opencl_threads[i], NULL, mining_opencl_thread, mt))
        {
          printf("start_mining_threads: Failed to create thread %d\n", i);
          free(mt);
          return 1;
        }
      }
    }
    if (ctx->config->disable_cuda == false)
    {
      ms->num_cuda_threads = ctx->cuda_device_count;
      ms->mining_cuda_threads = malloc(ms->num_cuda_threads * sizeof(pthread_t));
      if (!ms->mining_cuda_threads)
      {
        printf("start_mining_threads: Failed to allocate mining_threads\n");
        return 1;
      }
      if (ctx->config->debug == 1)
        printf("Starting %d CUDA threads\n", ms->num_cuda_threads);
      for (int i = 0; i < ms->num_cuda_threads; i++)
      {
        MiningThread *mt = malloc(sizeof(MiningThread));
        if (!mt)
        {
          printf("start_mining_threads: Failed to allocate MiningThread\n");
          return 1;
        }
        mt->threadIndex = i;
        mt->ctx = ctx;
        if (pthread_create(&ms->mining_cuda_threads[i], NULL, mining_cuda_thread, mt))
        {
          printf("start_mining_threads: Failed to create thread %d\n", i);
          free(mt);
          return 1;
        }
      }
    }
  }
  return 0;
}
