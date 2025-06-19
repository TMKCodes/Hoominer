#include "hoohash-miner.h"

MiningState *init_mining_state()
{
  MiningState *state = malloc(sizeof(MiningState));
  if (!state)
  {
    fprintf(stderr, "Failed to allocate memory for MiningState\n");
    exit(1);
  }
  state->num_cpu_threads = 0;
  state->global_target = NULL;
  state->job = malloc(sizeof(MiningJob)); // Allocate job
  if (!state->job)
  {
    fprintf(stderr, "Failed to allocate memory for MiningJob\n");
    free(state);
    exit(1);
  }
  memset(state->job, 0, sizeof(MiningJob)); // Initialize job fields
  pthread_mutex_init(&state->job_mutex, NULL);
  state->mining_cpu_threads = 0;
  state->mining_opencl_threads = 0;
  pthread_mutex_init(&state->target_mutex, NULL);
  return state;
}

void cleanup_mining_state(MiningState *state)
{
  if (state)
  {
    free(state);
    state = NULL;
  }
}

void uint64_to_little_endian(uint64_t value, uint8_t *buffer)
{
  value = htole64(value); // Convert to little-endian
  memcpy(buffer, &value, sizeof(uint64_t));
}

uint64_t little_endian_to_uint64(const uint8_t *buffer)
{
  uint64_t value;
  memcpy(&value, buffer, sizeof(uint64_t));
  return le64toh(value); // Convert from little-endian
}

void smallJobHeader(const uint64_t *ids, uint8_t *headerData)
{
  for (int i = 0; i < 4; i++)
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

int submit_mining_solution(int sockfd, const char *worker, const char *job_id, uint64_t nonce, uint8_t *hash)
{
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
  // printf("Submitted solution: %s\n", msg_with_newline);
  free(msg_with_newline);
  free(hash_hex);
  json_object_put(req);
  return ret < 0 ? -1 : 0;
}

void *mining_cpu_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  State state = {0};
  char *current_job_id = NULL;
  StratumContext *ctx = mt->ctx;
  MiningState *ms = ctx->ms;

  while (ctx->running)
  {
    pthread_mutex_lock(&ms->job_mutex);
    int is_running = ms->job->running;
    char *job_id = NULL;
    if (ms->job->job_id)
    {
      job_id = malloc(strlen(ms->job->job_id) + 1);
      if (!job_id)
      {
        fprintf(stderr, "Memory allocation failed\n");
        pthread_mutex_unlock(&ms->job_mutex);
        exit(1);
      }
      strcpy(job_id, ms->job->job_id);
    }
    if (job_id && is_running)
    {
      memcpy(state.PrevHeader, ms->job->header, DOMAIN_HASH_SIZE);
      state.Timestamp = ms->job->timestamp;
    }
    pthread_mutex_unlock(&ms->job_mutex);

    if (!job_id || !is_running)
    {
      free(job_id);
      usleep(100000);
      continue;
    }

    if (current_job_id && strcmp(current_job_id, job_id) != 0)
    {
      free(current_job_id);
      current_job_id = NULL;
    }

    if (!current_job_id)
    {
      current_job_id = job_id;
      job_id = NULL;
    }
    else
    {
      free(job_id);
      job_id = NULL;
    }

    // print_hex("Mining PrevHeader", state.PrevHeader, DOMAIN_HASH_SIZE);
    uint64_t nonce = mt->threadIndex;
    uint64_t step = ms->num_cpu_threads;
    generateHoohashMatrix(state.PrevHeader, state.mat);

    while (ms->job->running)
    {
      pthread_mutex_lock(&ms->job_mutex);
      int job_changed = ms->job->job_id == NULL || current_job_id == NULL || strcmp(ms->job->job_id, current_job_id) != 0;
      int still_running = ms->job->running;
      pthread_mutex_unlock(&ms->job_mutex);

      if (job_changed || !still_running)
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
        submit_mining_solution(mt->ctx->sockfd, ctx->worker, current_job_id, nonce, result);

        // Wait for a new job before continuing mining
        while (ctx->running)
        {
          pthread_mutex_lock(&ms->job_mutex);
          int job_changed = ms->job->job_id == NULL || current_job_id == NULL ||
                            strcmp(ms->job->job_id, current_job_id) != 0;
          int still_running = ms->job->running;
          pthread_mutex_unlock(&ms->job_mutex);

          if (job_changed && still_running)
          {
            // New job received, update current_job_id and break to continue mining
            free(current_job_id);
            current_job_id = NULL;
            nonce = 0;
            break;
          }
          usleep(100000); // Sleep 100ms to avoid busy-waiting
        }
        continue; // Skip to outer loop to pick up new job
      }
      if (meets_target <= 0)
      {
        submit_mining_solution(mt->ctx->sockfd, ctx->worker, current_job_id, nonce, result);

        // Wait for a new job before continuing mining
        while (ctx->running)
        {
          pthread_mutex_lock(&ms->job_mutex);
          int job_changed = ms->job->job_id == NULL || current_job_id == NULL ||
                            strcmp(ms->job->job_id, current_job_id) != 0;
          pthread_mutex_unlock(&ms->job_mutex);

          if (job_changed)
          {
            break;
          }
        }
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
  char job_id_buffer[256] = {0};                    // Reusable buffer for job_id
  uint8_t *local_target = malloc(DOMAIN_HASH_SIZE); // Local copy of global_target
  if (!local_target)
  {
    fprintf(stderr, "Device %d: Failed to allocate local_target\n", mt->threadIndex);
    exit(1);
  }

  // OpenCL kernel work sizes, optimized for GPU
  cl_ulong local_size = ctx->opencl_resources->max_work_group_size;
  cl_ulong global_size = ctx->opencl_resources->max_global_work_size;
  cl_ulong nonce_mask = 0xFFFFFFFFFFFFFFFFULL;
  cl_ulong nonce_fixed = 0;

  while (ctx->running)
  {
    // Check job validity once per outer loop
    pthread_mutex_lock(&ms->job_mutex);
    int is_running = ms->job->running;
    if (ms->job->job_id)
    {
      strncpy(job_id_buffer, ms->job->job_id, sizeof(job_id_buffer) - 1);
      job_id_buffer[sizeof(job_id_buffer) - 1] = '\0';
    }
    else
    {
      job_id_buffer[0] = '\0';
    }
    if (is_running && job_id_buffer[0])
    {
      memcpy(state.PrevHeader, ms->job->header, DOMAIN_HASH_SIZE);
      state.Timestamp = ms->job->timestamp;
    }
    pthread_mutex_unlock(&ms->job_mutex);

    // Copy global_target under mutex
    pthread_mutex_lock(&ms->target_mutex);
    if (ms->global_target)
    {
      memcpy(local_target, ms->global_target, DOMAIN_HASH_SIZE);
    }
    else
    {
      memset(local_target, 0, DOMAIN_HASH_SIZE); // Fallback if target is NULL
    }
    pthread_mutex_unlock(&ms->target_mutex);

    if (!is_running || !job_id_buffer[0])
    {
      usleep(100000); // 100ms sleep
      continue;
    }

    // Update current job ID
    if (current_job_id && strcmp(current_job_id, job_id_buffer) != 0)
    {
      free(current_job_id);
      current_job_id = NULL;
      nonce_fixed = (cl_ulong)mt->threadIndex * global_size;
    }
    if (!current_job_id)
    {
      current_job_id = strdup(job_id_buffer);
      if (!current_job_id)
      {
        fprintf(stderr, "Device %d: Memory allocation failed for current_job_id\n", mt->threadIndex);
        free(local_target);
        exit(1);
      }
    }

    // Generate hoohash matrix
    generateHoohashMatrix(state.PrevHeader, state.mat);

    // Process nonces until job changes
    OpenCLResult result = {0};
    while (ms->job->running)
    {
      // Execute kernel asynchronously
      cl_int status = run_hoohash_kernel(&ctx->opencl_resources[mt->threadIndex], local_size, global_size,
                                         state.PrevHeader, local_target, state.mat, state.Timestamp,
                                         nonce_mask, nonce_fixed, &result);
      if (status != CL_SUCCESS)
      {
        fprintf(stderr, "Device %d: Kernel execution failed: %d\n", mt->threadIndex, status);
        nonce_fixed += global_size;
        break;
      }

      // Update nonces processed
      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      ctx->hd->nonces_processed += global_size;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      // Process result
      if (result.nonce != 0)
      {
        pthread_mutex_lock(&ms->target_mutex);
        int meets_target = compare_target(result.hash, ms->global_target, DOMAIN_HASH_SIZE);
        pthread_mutex_unlock(&ms->target_mutex);

        if (meets_target <= 0)
        {
          submit_mining_solution(ctx->sockfd, ctx->worker, current_job_id, result.nonce, result.hash);

          // Wait for a new job before continuing mining
          while (ctx->running)
          {
            pthread_mutex_lock(&ms->job_mutex);
            int job_changed = ms->job->job_id == NULL || current_job_id == NULL ||
                              strcmp(ms->job->job_id, current_job_id) != 0;
            pthread_mutex_unlock(&ms->job_mutex);
            if (job_changed)
            {
              break;
            }
          }
          break; // Exit inner loop to pick up new job
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
  if (ms->job)
  {
    pthread_mutex_destroy(&ms->job_mutex);
    ms->job->job_id = NULL;
    ms->job->running = 0;
    free(ms->job);
    ms->job = NULL;
  }
}

void cleanup_mining_threads(MiningState *ms)
{
  if (ms->mining_cpu_threads)
  {
    for (int i = 0; i < ms->num_cpu_threads; i++)
    {
      pthread_cancel(ms->mining_cpu_threads[i]);
      pthread_join(ms->mining_cpu_threads[i], NULL);
    }
    free(ms->mining_cpu_threads);
    ms->mining_cpu_threads = NULL;
  }
  if (ms->mining_opencl_threads)
  {
    for (int i = 0; i < 1; i++)
    {
      pthread_cancel(ms->mining_opencl_threads[i]);
      pthread_join(ms->mining_opencl_threads[i], NULL);
    }
    free(ms->mining_opencl_threads);
    ms->mining_opencl_threads = NULL;
  }
}

int start_mining_threads(StratumContext *ctx, MiningState *ms)
{

  if (ctx->disable_cpu == false)
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
  if (ctx->disable_gpu == false)
  {
    ms->mining_opencl_threads = malloc(ctx->opencl_device_count * sizeof(pthread_t));
    if (!ms->mining_opencl_threads)
    {
      printf("start_mining_threads: Failed to allocate mining_threads\n");
      cleanup_job(ms);
      return 1;
    }
    for (unsigned int i = 0; i < ctx->opencl_device_count; i++)
    {
      MiningThread *mt = malloc(sizeof(MiningThread));
      if (!mt)
      {
        printf("start_mining_threads: Failed to allocate MiningThread\n");
        cleanup_mining_threads(ms);
        return 1;
      }
      mt->threadIndex = 0;
      mt->ctx = ctx;
      if (pthread_create(&ms->mining_opencl_threads[i], NULL, mining_opencl_thread, mt))
      {
        printf("start_mining_threads: Failed to create thread %d\n", 0);
        free(mt);
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

  // const char *raw_msg = json_object_to_json_string_ext(message, JSON_C_TO_STRING_PLAIN);
  // if (raw_msg)
  //   printf("Received message: %s\n", raw_msg);

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
        // Target value : 0001869ffffffffff00000000000000000000000000000000000000000000000 Target number : 2695994666715063587147843047034415327626875491774906241713145745919442944
        // Difficulty set : diff = 0.000010 Target value : 0001869ffffffffff00000000000000000000000000000000000000000000000 Target number : 2695994666715063587147843047034415327626875491774906241713145745919442944

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
              // printf("Failed to parse header element %d: %s\n", i, json_object_get_string(item));
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
        // print_hex("Parsed Array Header", header, DOMAIN_HASH_SIZE);
      }
      else if (json_object_is_type(header_item, json_type_string))
      {
        const char *hex_str = json_object_get_string(header_item);
        if (strlen(hex_str) != 64)
        {
          printf("Invalid hex string length: %lu, expected 64\n", strlen(hex_str));
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

      pthread_mutex_lock(&ms->job_mutex);
      free(ms->job->job_id);

      const char *jid = json_object_get_string(job_id);
      if (jid)
      {
        ms->job->job_id = malloc(strlen(jid) + 1);
        if (ms->job->job_id)
        {
          strcpy(ms->job->job_id, jid);
          memcpy(ms->job->header, header, DOMAIN_HASH_SIZE);
          ms->job->timestamp = timestamp_int;
          ms->job->running = 1;
          // print_hex("Stored Job Header", job->header, DOMAIN_HASH_SIZE);
        }
        else
        {
          printf("Failed to allocate memory for job ID\n");
        }
      }
      else
      {
        ms->job->job_id = NULL;
        printf("Job ID string is NULL\n");
      }

      pthread_mutex_unlock(&ms->job_mutex);
    }
  }
  else
  {
    if (ms->job->running)
    {
      json_object *result;
      if (json_object_object_get_ex(message, "result", &result))
      {
        if (json_object_is_type(result, json_type_boolean) && json_object_get_boolean(result))
        {
          pthread_mutex_lock(&ms->job_mutex);
          ms->job->running = 0;
          pthread_mutex_unlock(&ms->job_mutex);
          ctx->hd->accepted++;
        }
        else
        {
          ctx->hd->rejected++;
          fprintf(stderr, "Job rejected. Full result: %s\n", json_object_to_json_string(message));
        }
      }
    }
  }
}
