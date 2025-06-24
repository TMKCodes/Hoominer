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

void cleanup_mining_state(MiningState *state)
{
  if (state)
  {
    pthread_mutex_lock(&state->job_queue.queue_mutex);
    for (int i = 0; i < JOB_QUEUE_SIZE; i++)
    {
      if (state->job_queue.jobs[i].job_id)
      {
        free(state->job_queue.jobs[i].job_id);
        state->job_queue.jobs[i].job_id = NULL;
      }
    }
    pthread_mutex_unlock(&state->job_queue.queue_mutex);
    pthread_mutex_destroy(&state->job_queue.queue_mutex);
    pthread_cond_destroy(&state->job_queue.queue_cond);

    if (state->global_target)
      free(state->global_target);

    pthread_mutex_destroy(&state->job_mutex);
    pthread_mutex_destroy(&state->target_mutex);
    free(state);
  }
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

int submit_mining_solution(int sockfd, const char *worker, const char *job_id, uint64_t nonce, uint8_t *hash, MiningState *ms)
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

  const char *msg = json_object_to_json_string_ext(req, JSON_C_TO_STRING_PLAIN);
  if (!msg)
  {
    free(hash_hex);
    json_object_put(req);
    return -1;
  }

  size_t len = strlen(msg);
  char *msg_with_newline = malloc(len + 2);
  if (!msg_with_newline)
  {
    free(hash_hex);
    json_object_put(req);
    return -1;
  }

  strcpy(msg_with_newline, msg);
  msg_with_newline[len] = '\n';
  msg_with_newline[len + 1] = '\0';

  int ret = send(sockfd, msg_with_newline, len + 1, 0);
  free(msg_with_newline);
  free(hash_hex);
  json_object_put(req);
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
  State state = {0};
  char *current_job_id = NULL;
  StratumContext *ctx = mt->ctx;
  MiningState *ms = ctx->ms;
  QueuedJob current_job = {0};

  while (ctx->running)
  {
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      usleep(100000);
      continue;
    }

    memcpy(state.PrevHeader, current_job.header, DOMAIN_HASH_SIZE);
    state.Timestamp = current_job.timestamp;
    memcpy(state.mat, current_job.matrix, sizeof(double) * 64 * 64);

    uint64_t nonce = mt->threadIndex;
    uint64_t step = ms->num_cpu_threads;

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

      pthread_mutex_lock(&ms->target_mutex);
      int meets_target = compare_target(result, ms->global_target, DOMAIN_HASH_SIZE);
      pthread_mutex_unlock(&ms->target_mutex);

      pthread_mutex_lock(&mt->ctx->hd->hashrate_mutex);
      mt->ctx->hd->nonces_processed++;
      pthread_mutex_unlock(&mt->ctx->hd->hashrate_mutex);

      if (meets_target <= 0)
      {
        submit_mining_solution(mt->ctx->sockfd, ctx->worker, current_job_id, nonce, result, ms);
        break;
      }

      nonce += step;
    }
  }

  free(current_job_id);
  free(mt);
  return NULL;
}

void *mining_opencl_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;
  uint8_t *local_target = malloc(DOMAIN_HASH_SIZE);
  if (!local_target)
  {
    fprintf(stderr, "Device %d: Failed to allocate local_target\n", mt->threadIndex);
    exit(1);
  }
  cl_ulong local_size = ctx->opencl_resources->max_work_group_size;
  cl_ulong global_size = ctx->opencl_resources->max_global_work_size;
  cl_ulong nonce_mask = 0xFFFFFFFFFFFFFFFFULL;
  cl_ulong nonce_fixed = (cl_ulong)mt->threadIndex * global_size;

  while (ctx->running)
  {
    pthread_mutex_lock(&ms->target_mutex);
    if (ms->global_target)
      memcpy(local_target, ms->global_target, DOMAIN_HASH_SIZE);
    else
      memset(local_target, 0, DOMAIN_HASH_SIZE);
    pthread_mutex_unlock(&ms->target_mutex);

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      usleep(100000);
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
      cl_int status = run_opencl_hoohash_kernel(&ctx->opencl_resources[mt->threadIndex], local_size, global_size,
                                                state.PrevHeader, local_target, state.mat, state.Timestamp,
                                                nonce_mask, nonce_fixed, &result);
      if (status != CL_SUCCESS)
      {
        fprintf(stderr, "Device %d: Kernel execution failed: %d\n", mt->threadIndex, status);
        nonce_fixed += global_size;
        break;
      }

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      ctx->hd->nonces_processed += global_size;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      if (result.nonce != 0)
      {
        pthread_mutex_lock(&ms->target_mutex);
        int meets_target = compare_target(result.hash, ms->global_target, DOMAIN_HASH_SIZE);
        pthread_mutex_unlock(&ms->target_mutex);

        if (meets_target <= 0)
        {
          submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, result.nonce, result.hash, ms);
          break;
        }
      }

      nonce_fixed += global_size;
    }
  }

  free(current_job_id);
  free(local_target);
  return NULL;
}

void *mining_cuda_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  MiningState *ms = ctx->ms;
  State state = {0};
  char *current_job_id = NULL;
  uint8_t *local_target = malloc(DOMAIN_HASH_SIZE);
  if (!local_target)
  {
    fprintf(stderr, "Device %d: Failed to allocate local_target\n", mt->threadIndex);
    exit(1);
  }
  cl_ulong global_size = ctx->cuda_resources->max_block_size;
  cl_ulong nonce_mask = 0xFFFFFFFFFFFFFFFFULL;
  cl_ulong nonce_fixed = (cl_ulong)mt->threadIndex * global_size;

  while (ctx->running)
  {
    pthread_mutex_lock(&ms->target_mutex);
    if (ms->global_target)
      memcpy(local_target, ms->global_target, DOMAIN_HASH_SIZE);
    else
      memset(local_target, 0, DOMAIN_HASH_SIZE);
    pthread_mutex_unlock(&ms->target_mutex);

    QueuedJob current_job = {0};
    if (!get_current_job(ms, &current_job, &current_job_id))
    {
      usleep(100000);
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
      cl_int status = run_cuda_hoohash_kernel(&ctx->cuda_resources[mt->threadIndex],
                                              state.PrevHeader, local_target, state.mat, state.Timestamp,
                                              nonce_mask, nonce_fixed, &result);
      if (status != CL_SUCCESS)
      {
        fprintf(stderr, "Device %d: Kernel execution failed: %d\n", mt->threadIndex, status);
        nonce_fixed += global_size;
        break;
      }

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      ctx->hd->nonces_processed += global_size;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      if (result.nonce != 0)
      {
        printf("Nonce: %llu, PoW Hash: %s\n", result.nonce, encodeHex(result.hash, 32));
        pthread_mutex_lock(&ms->target_mutex);
        int meets_target = compare_target(result.hash, ms->global_target, DOMAIN_HASH_SIZE);
        pthread_mutex_unlock(&ms->target_mutex);

        if (meets_target <= 0)
        {
          submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, result.nonce, result.hash, ms);
          break;
        }
      }

      nonce_fixed += global_size;
    }
  }

  free(current_job_id);
  free(local_target);
  return NULL;
}

void cleanup_job(MiningState *ms)
{
  pthread_mutex_lock(&ms->job_queue.queue_mutex);
  for (int i = 0; i < JOB_QUEUE_SIZE; i++)
  {
    if (ms->job_queue.jobs[i].job_id)
    {
      free(ms->job_queue.jobs[i].job_id);
      ms->job_queue.jobs[i].job_id = NULL;
      ms->job_queue.jobs[i].running = 0;
      ms->job_queue.jobs[i].completed = 0;
    }
  }
  ms->job_queue.head = 0;
  ms->job_queue.tail = 0;
  pthread_mutex_unlock(&ms->job_queue.queue_mutex);
}

void cleanup_mining_threads(MiningState *ms)
{
  if (ms->mining_cpu_threads != NULL)
  {
    for (int i = 0; i < ms->num_cpu_threads; i++)
    {
      pthread_cancel(ms->mining_cpu_threads[i]);
      pthread_join(ms->mining_cpu_threads[i], NULL);
    }
    free(ms->mining_cpu_threads);
    ms->mining_cpu_threads = NULL;
  }
  if (ms->mining_opencl_threads != NULL)
  {
    for (int i = 0; i < ms->num_opencl_threads; i++)
    {
      pthread_cancel(ms->mining_opencl_threads[i]);
      pthread_join(ms->mining_opencl_threads[i], NULL);
    }
    free(ms->mining_opencl_threads);
    ms->mining_opencl_threads = NULL;
  }
  if (ms->mining_cuda_threads != NULL)
  {
    for (int i = 0; i < ms->num_cuda_threads; i++)
    {
      pthread_cancel(ms->mining_cuda_threads[i]);
      pthread_join(ms->mining_cuda_threads[i], NULL);
    }
    free(ms->mining_cuda_threads);
    ms->mining_cuda_threads = NULL;
  }
}

int start_mining_threads(StratumContext *ctx, MiningState *ms)
{
  ctx->hd->accepted = 0;
  ctx->hd->rejected = 0;
  ctx->hd->nonces_processed = 0;
  if (!ctx->disable_cpu)
  {
    ms->mining_cpu_threads = malloc(ms->num_cpu_threads * sizeof(pthread_t));
    if (!ms->mining_cpu_threads)
    {
      printf("start_mining_threads: Failed to allocate mining_threads\n");
      cleanup_job(ms);
      return 1;
    }
    for (int i = 0; i < ms->num_cpu_threads; i++)
    {
      MiningThread *mt = malloc(sizeof(MiningThread));
      if (!mt)
      {
        printf("start_mining_threads: Failed to allocate MiningThread\n");
        cleanup_mining_threads(ms);
        return 1;
      }
      mt->threadIndex = i;
      mt->ctx = ctx;
      if (pthread_create(&ms->mining_cpu_threads[i], NULL, mining_cpu_thread, mt) != 0)
      {
        printf("start_mining_threads: Failed to create thread %d\n", i);
        free(mt);
        cleanup_mining_threads(ms);
        return 1;
      }
    }
  }
  if (!ctx->disable_gpu)
  {
    ms->num_opencl_threads = ctx->opencl_device_count;
    ms->mining_opencl_threads = malloc(ms->num_opencl_threads * sizeof(pthread_t));
    if (!ms->mining_opencl_threads)
    {
      printf("start_mining_threads: Failed to allocate mining_threads\n");
      cleanup_job(ms);
      return 1;
    }
    for (int i = 0; i < ms->num_opencl_threads; i++)
    {
      MiningThread *mt = malloc(sizeof(MiningThread));
      if (!mt)
      {
        printf("start_mining_threads: Failed to allocate MiningThread\n");
        cleanup_mining_threads(ms);
        return 1;
      }
      mt->threadIndex = i;
      mt->ctx = ctx;
      if (pthread_create(&ms->mining_opencl_threads[i], NULL, mining_opencl_thread, mt))
      {
        printf("start_mining_threads: Failed to create thread %d\n", i);
        free(mt);
        cleanup_mining_threads(ms);
        return 1;
      }
    }
    ms->num_cuda_threads = ctx->cuda_device_count;
    ms->mining_cuda_threads = malloc(ms->num_cuda_threads * sizeof(pthread_t));
    if (!ms->mining_cuda_threads)
    {
      printf("start_mining_threads: Failed to allocate mining_threads\n");
      cleanup_job(ms);
      return 1;
    }
    for (int i = 0; i < ms->num_cuda_threads; i++)
    {
      MiningThread *mt = malloc(sizeof(MiningThread));
      if (!mt)
      {
        printf("start_mining_threads: Failed to allocate MiningThread\n");
        cleanup_mining_threads(ms);
        return 1;
      }
      mt->threadIndex = i;
      mt->ctx = ctx;
      if (pthread_create(&ms->mining_cuda_threads[i], NULL, mining_cuda_thread, mt))
      {
        printf("start_mining_threads: Failed to create thread %d\n", i);
        free(mt);
        cleanup_mining_threads(ms);
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
      json_object *time = json_object_array_get_idx(params, 2);

      if (!json_object_is_type(job_id, json_type_string) || !header_item || !time ||
          (!json_object_is_type(time, json_type_int) && !json_object_is_type(time, json_type_double)))
      {
        printf("mining.notify: Invalid parameters\n");
        return;
      }

      uint8_t header[DOMAIN_HASH_SIZE] = {0};
      uint64_t timestamp_int;

      if (json_object_is_type(time, json_type_int))
        timestamp_int = json_object_get_uint64(time);
      else
        timestamp_int = (uint64_t)json_object_get_double(time);

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
      int next_tail = (ms->job_queue.tail + 1) % JOB_QUEUE_SIZE;
      if (next_tail == ms->job_queue.head)
      {
        free(ms->job_queue.jobs[ms->job_queue.head].job_id);
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
    if (json_object_object_get_ex(message, "result", &result))
    {
      if (json_object_is_type(result, json_type_boolean))
      {
        pthread_mutex_lock(&ms->job_queue.queue_mutex);
        if (json_object_get_boolean(result))
          ctx->hd->accepted++;
        else
          ctx->hd->rejected++;
        pthread_mutex_unlock(&ms->job_queue.queue_mutex);
      }
    }
  }
}