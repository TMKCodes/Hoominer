#include "hoohash-miner.h"
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
  if (ctx->disable_cpu == 0)
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
  if (ctx->disable_gpu == 0)
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
  return 0;
}
void process_stratum_message(json_object *message, StratumContext *ctx, MiningState *ms)
{
  if (!message)
  {
    printf("process_stratum_message: Null message received\n");
    return;
  }

  json_object *method_obj;
  if (json_object_object_get_ex(message, "method", &method_obj) && json_object_is_type(method_obj, json_type_string))
  {
    const char *method_str = json_object_get_string(method_obj);
    if (!strcmp(method_str, "mining.set_difficulty"))
    {
      json_object *params;
      if (json_object_object_get_ex(message, "params", &params) && json_object_is_type(params, json_type_array))
      {
        json_object *diff = json_object_array_get_idx(params, 0);
        if (json_object_is_type(diff, json_type_double) || json_object_is_type(diff, json_type_int))
        {
          pthread_mutex_lock(&ms->target_mutex);
          if (ms->global_target)
            free(ms->global_target);
          double difficulty = json_object_get_double(diff);
          // printf("Received difficulty: %.6f\n", difficulty);
          ms->global_target = target_from_pool_difficulty(difficulty, DOMAIN_HASH_SIZE);
          pthread_mutex_unlock(&ms->target_mutex);
        }
      }
    }
    else if (!strcmp(method_str, "mining.notify"))
    {
      json_object *params;
      if (!json_object_object_get_ex(message, "params", &params) || !json_object_is_type(params, json_type_array))
      {
        printf("mining.notify: params missing or not an array\n");
        return;
      }

      json_object *job_id = json_object_array_get_idx(params, 0);
      json_object *header_item = json_object_array_get_idx(params, 1);
      json_object *time_param = json_object_array_get_idx(params, 2);

      if (!json_object_is_type(job_id, json_type_string) || !header_item || !time_param ||
          (!json_object_is_type(time_param, json_type_int) && !json_object_is_type(time_param, json_type_double)))
      {
        printf("mining.notify: Invalid parameters\n");
        return;
      }

      uint8_t header[DOMAIN_HASH_SIZE] = {0};
      uint64_t timestamp_int;

      if (json_object_is_type(time_param, json_type_int))
        timestamp_int = json_object_get_uint64(time_param);
      else
        timestamp_int = (uint64_t)json_object_get_double(time_param);

      if (json_object_is_type(header_item, json_type_array))
      {
        if (json_object_array_length(header_item) != 4)
        {
          printf("Invalid header array size: %zu\n", json_object_array_length(header_item));
          return;
        }
        uint64_t hash_elements[4] = {0};
        for (int i = 0; i < 4; i++)
        {
          json_object *item = json_object_array_get_idx(header_item, i);
          if (json_object_is_type(item, json_type_int))
          {
            hash_elements[i] = json_object_get_uint64(item);
          }
          else if (json_object_is_type(item, json_type_string))
          {
            if (sscanf(json_object_get_string(item), "%" SCNx64, &hash_elements[i]) != 1)
            {
              return;
            }
          }
          else
          {
            printf("Invalid header element at index %d\n", i);
            return;
          }
        }
        smallJobHeader(hash_elements, header);
      }
      else if (json_object_is_type(header_item, json_type_string))
      {
        const char *hex_str = json_object_get_string(header_item);
        if (strlen(hex_str) != 64)
        {
          printf("Invalid hex string length: %lu\n", strlen(hex_str));
          return;
        }
        if (hex_to_bytes(hex_str, header, DOMAIN_HASH_SIZE) != 0)
        {
          printf("Failed to parse hex header: %s\n", hex_str);
          return;
        }
        print_hex("Parsed Hex Header", header, DOMAIN_HASH_SIZE);
      }
      else
      {
        printf("Invalid header format in mining.notify\n");
        return;
      }

      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      // Clear outdated jobs
      while (ms->job_queue.head != ms->job_queue.tail &&
             ms->job_queue.jobs[ms->job_queue.head].timestamp < (uint64_t)time(NULL) - JOB_MAX_AGE)
      {
        free(ms->job_queue.jobs[ms->job_queue.head].job_id);
        ms->job_queue.jobs[ms->job_queue.head].job_id = NULL;
        ms->job_queue.head = (ms->job_queue.head + 1) % JOB_QUEUE_SIZE;
      }

      int next_tail = (ms->job_queue.tail + 1) % JOB_QUEUE_SIZE;
      if (next_tail == ms->job_queue.head)
      {
        free(ms->job_queue.jobs[ms->job_queue.head].job_id);
        ms->job_queue.jobs[ms->job_queue.head].job_id = NULL;
        ms->job_queue.head = (ms->job_queue.head + 1) % JOB_QUEUE_SIZE;
      }

      QueuedJob *new_job = &ms->job_queue.jobs[ms->job_queue.tail];
      const char *jid = json_object_get_string(job_id);
      if (jid)
      {
        new_job->job_id = strdup(jid);
        if (new_job->job_id)
        {
          memcpy(new_job->header, header, DOMAIN_HASH_SIZE);
          new_job->timestamp = timestamp_int;
          new_job->running = 1;
          new_job->completed = 0;
          generateHoohashMatrix(header, new_job->matrix);
          ms->job_queue.tail = next_tail;
          ms->new_job_available = 1; // Signal new job
          pthread_cond_broadcast(&ms->job_queue.queue_cond);
        }
        else
        {
          printf("Failed to allocate memory for job ID\n");
        }
      }
      else
      {
        printf("Job ID string is NULL\n");
      }
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);
    }
  }
  else
  {
    json_object *result;
    json_object *error;
    int device_index;
    int devices = ctx->cpu_device_count + ctx->opencl_device_count + ctx->cuda_device_count;
    dequeue_int_fifo(&ctx->mining_submit_fifo, &device_index);
    if (devices >= device_index)
    {
      ReportingDevice *device = ctx->hd->devices[device_index];
      if (json_object_object_get_ex(message, "error", &error))
      {
        if (!json_object_is_type(error, json_type_null))
        {
          if (json_object_is_type(error, json_type_array))
          {
            json_object *code = json_object_array_get_idx(error, 0);
            // json_object *msg = json_object_array_get_idx(error, 1);
            int err_code = json_object_get_int(code);
            // const char *err_msg = json_object_get_string(msg);
            if (err_code == 21)
            {
              // printf("Stale share detected (job not found): %s\n", err_msg);
              pthread_mutex_lock(&ms->job_queue.queue_mutex);
              device->stales++;
              pthread_mutex_unlock(&ms->job_queue.queue_mutex);
            }
            else if (err_code == 20)
            {
              // printf("Stale share detected (duplicate): %s\n", err_msg);
              pthread_mutex_lock(&ms->job_queue.queue_mutex);
              device->stales++;
              pthread_mutex_unlock(&ms->job_queue.queue_mutex);
            }
            else
            {
              const char *result_str = json_object_to_json_string(message);
              printf("Error: %s\n", result_str);
              pthread_mutex_lock(&ms->job_queue.queue_mutex);
              device->rejected++;
              pthread_mutex_unlock(&ms->job_queue.queue_mutex);
            }
          }
        }
      }
      if (json_object_object_get_ex(message, "result", &result))
      {
        if (json_object_is_type(result, json_type_boolean))
        {
          pthread_mutex_lock(&ms->job_queue.queue_mutex);
          if (json_object_get_boolean(result))
            device->accepted++;
          else
            device->rejected++;
          pthread_mutex_unlock(&ms->job_queue.queue_mutex);
        }
      }
    }
  }
}