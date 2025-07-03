#include "miner-hoohash.h"
#include <time.h>

MiningState *init_mining_state()
{
  MiningState *state = malloc(sizeof(MiningState));
  if (!state)
  {
    fprintf(stderr, "Failed to allocate memory for MiningState\n");
    exit(1);
  }
  state->num_cpu_threads = 0;
  state->num_opencl_threads = 0;
  state->num_cuda_threads = 0;
  state->global_target = NULL;
  state->job = NULL;
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

  // printf("Submitting job %s, age: %ld seconds\n", job_id, time(NULL) - ms->job_queue.jobs[ms->job_queue.head].timestamp);

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
  while (ms->job_queue.head == ms->job_queue.tail || ms->job_queue.jobs[ms->job_queue.head].completed)
  {
    pthread_cond_wait(&ms->job_queue.queue_cond, &ms->job_queue.queue_mutex);
    if (ms->job_queue.head == ms->job_queue.tail && !ms->job_queue.jobs[ms->job_queue.head].running)
    {
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);
      return 0;
    }
  }

  *job = ms->job_queue.jobs[ms->job_queue.head];
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
  pthread_mutex_unlock(&ms->job_queue.queue_mutex);
  return job->running && !job->completed;
}

void *mining_cpu_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;
  int reporting_index = 0;
  ReportingDevice *cpu_reporting_device = ctx->hd->devices[reporting_index];

  uint64_t nonce = mt->threadIndex;
  uint64_t step = ms->num_cpu_threads;

  while (ctx->running)
  {

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      usleep(100);
      continue;
    }

    memcpy(state.PrevHeader, current_job.header, DOMAIN_HASH_SIZE);
    state.Timestamp = current_job.timestamp;
    memcpy(state.mat, current_job.matrix, sizeof(double) * 64 * 64);

    while (ctx->running)
    {

      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      int job_valid = ms->job_queue.head != ms->job_queue.tail &&
                      ms->job_queue.jobs[ms->job_queue.head].running &&
                      !ms->job_queue.jobs[ms->job_queue.head].completed &&
                      strcmp(ms->job_queue.jobs[ms->job_queue.head].job_id, current_job_id) == 0;
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      state.Nonce = nonce;
      uint8_t result[DOMAIN_HASH_SIZE];
      CalculateProofOfWorkValue(&state, result);

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      cpu_reporting_device->nonces_processed++;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      pthread_mutex_lock(&ms->target_mutex);
      int meets_target = compare_target(result, ms->global_target, DOMAIN_HASH_SIZE);
      pthread_mutex_unlock(&ms->target_mutex);

      if (meets_target <= 0)
      {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        uint64_t current_time_ms = now.tv_sec * 1000ULL + now.tv_nsec / 1000000ULL;
        if (current_job.timestamp * 1000ULL + JOB_MAX_AGE > current_time_ms)
        {
          submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, state.Nonce, result, ms, ctx, reporting_index);
          break;
        }
      }
      nonce += step;
    }
  }

  free(current_job_id);
  return NULL;
}

void *mining_opencl_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;
  cl_ulong local_work_size = ctx->opencl_resources[mt->threadIndex].max_work_group_size;
  cl_ulong global_work_size = ctx->opencl_resources[mt->threadIndex].max_global_work_size;
  cl_ulong nonce_mask = 0xFFFFFFFFFFFFFFFFULL;
  cl_ulong nonce_fixed = (cl_ulong)mt->threadIndex * global_work_size;
  int reporting_index = ctx->cpu_device_count + mt->threadIndex;
  ReportingDevice *opencl_reporting_device = ctx->hd->devices[reporting_index];

  while (ctx->running)
  {

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      usleep(100);
      continue;
    }

    memcpy(state.PrevHeader, current_job.header, DOMAIN_HASH_SIZE);
    state.Timestamp = current_job.timestamp;
    memcpy(state.mat, current_job.matrix, sizeof(double) * 64 * 64);

    while (ctx->running)
    {

      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      int job_valid = ms->job_queue.head != ms->job_queue.tail &&
                      ms->job_queue.jobs[ms->job_queue.head].running &&
                      !ms->job_queue.jobs[ms->job_queue.head].completed &&
                      strcmp(ms->job_queue.jobs[ms->job_queue.head].job_id, current_job_id) == 0;
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      OpenCLResult result = {0};
      cl_int status = run_opencl_hoohash_kernel(&ctx->opencl_resources[mt->threadIndex], global_work_size, local_work_size, state.PrevHeader, ms->global_target, state.mat, state.Timestamp, nonce_mask, nonce_fixed, &result);

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      opencl_reporting_device->nonces_processed += global_work_size;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      if (status != CL_SUCCESS)
      {
        fprintf(stderr, "Device %d: Kernel execution failed: %d\n", mt->threadIndex, status);
        nonce_fixed += global_work_size;
        break;
      }

      if (result.nonce != 0)
      {

        pthread_mutex_lock(&ms->target_mutex);
        int meets_target = compare_target(result.hash, ms->global_target, DOMAIN_HASH_SIZE);
        pthread_mutex_unlock(&ms->target_mutex);

        if (meets_target <= 0)
        {
          struct timespec now;
          clock_gettime(CLOCK_MONOTONIC, &now);
          uint64_t current_time_ms = now.tv_sec * 1000ULL + now.tv_nsec / 1000000ULL;
          if (current_job.timestamp * 1000ULL + JOB_MAX_AGE > current_time_ms)
          {
            submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, result.nonce, result.hash, ms, ctx, reporting_index);
            break;
          }
        }
      }

      nonce_fixed += global_work_size;
    }
  }

  free(current_job_id);
  return NULL;
}

void *mining_cuda_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;
  unsigned long nonce_mask = 0xFFFFFFFFFFFFFFFFULL;
  unsigned long hashes_per_cuda_call = ctx->cuda_resources[mt->threadIndex].optimal_grid_size * ctx->cuda_resources[mt->threadIndex].optimal_block_size;
  unsigned long nonce_fixed = (unsigned long)mt->threadIndex * hashes_per_cuda_call;
  int reporting_index = ctx->cpu_device_count + ctx->opencl_device_count + mt->threadIndex;
  ReportingDevice *cuda_reporting_device = ctx->hd->devices[reporting_index];

  while (ctx->running)
  {

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      usleep(100);
      continue;
    }

    memcpy(state.PrevHeader, current_job.header, DOMAIN_HASH_SIZE);
    state.Timestamp = current_job.timestamp;
    memcpy(state.mat, current_job.matrix, sizeof(double) * 64 * 64);

    while (ctx->running)
    {
      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      int job_valid = ms->job_queue.head != ms->job_queue.tail &&
                      ms->job_queue.jobs[ms->job_queue.head].running &&
                      !ms->job_queue.jobs[ms->job_queue.head].completed &&
                      strcmp(ms->job_queue.jobs[ms->job_queue.head].job_id, current_job_id) == 0;
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      CudaResult result = {0};
      int error = run_cuda_hoohash_kernel(&ctx->cuda_resources[mt->threadIndex],
                                          state.PrevHeader, ms->global_target, state.mat, state.Timestamp,
                                          nonce_mask, nonce_fixed, &result);

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      cuda_reporting_device->nonces_processed += hashes_per_cuda_call;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      if (error != 0)
      {
        fprintf(stderr, "Device %d: Kernel execution failed: %d\n", mt->threadIndex, error);
        nonce_fixed += hashes_per_cuda_call;
        break;
      }

      if (result.nonce != 0)
      {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        uint64_t current_time_ms = now.tv_sec * 1000ULL + now.tv_nsec / 1000000ULL;
        if (current_job.timestamp * 1000ULL + JOB_MAX_AGE > current_time_ms)
        {
          submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, result.nonce, result.hash, ms, ctx, reporting_index);
          break;
        }
      }

      nonce_fixed += hashes_per_cuda_call;
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
