#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <json-c/json.h>
#include <inttypes.h>
#include <sys/select.h>
#include <math.h>
#include <stdint.h>
#include <signal.h>
#include <gmp.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "algorithms/hoohash/hoohash.h"
#include <endian.h>
#ifdef __CUDA__
#include <cuda_runtime.h>
typedef struct
{
  int device_id;
  char name[256];
} GPUDevice;

GPUDevice *initialize_gpu_devices(int *device_count)
{
  cudaError_t err = cudaGetDeviceCount(device_count);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA initialization failed: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  if (*device_count == 0)
  {
    fprintf(stderr, "No CUDA-compatible devices found.\n");
    exit(1);
  }

  printf("Found %d CUDA-compatible device(s).\n", *device_count);

  GPUDevice *devices = malloc(*device_count * sizeof(GPUDevice));
  if (!devices)
  {
    fprintf(stderr, "Memory allocation failed.\n");
    exit(1);
  }

  for (int i = 0; i < *device_count; i++)
  {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, i);
    devices[i].device_id = i;
    strncpy(devices[i].name, device_prop.name, sizeof(devices[i].name) - 1);
    devices[i].name[sizeof(devices[i].name) - 1] = '\0';
    printf("Device %d: %s\n", i, devices[i].name);
  }

  return devices;
}

#elif __OPENCL_CL_H
#include <CL/cl.h>
typedef struct
{
  cl_device_id device_id;
  char name[256];
} GPUDevice;

GPUDevice *initialize_gpu_devices(int *device_count)
{
  cl_platform_id platform_id = NULL;
  cl_device_id *device_ids = NULL;
  cl_uint num_platforms, num_devices;
  cl_int err;

  // Get the number of OpenCL platforms
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0)
  {
    fprintf(stderr, "OpenCL initialization failed: No platforms found.\n");
    exit(1);
  }

  printf("Found %u OpenCL platform(s).\n", num_platforms);

  // Get the first platform
  err = clGetPlatformIDs(1, &platform_id, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to get OpenCL platform.\n");
    exit(1);
  }

  // Get the number of devices for the platform
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0)
  {
    fprintf(stderr, "No OpenCL GPU devices found.\n");
    exit(1);
  }

  printf("Found %u OpenCL GPU device(s).\n", num_devices);

  device_ids = malloc(num_devices * sizeof(cl_device_id));
  if (!device_ids)
  {
    fprintf(stderr, "Memory allocation failed.\n");
    exit(1);
  }

  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, device_ids, NULL);
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to get OpenCL devices.\n");
    free(device_ids);
    exit(1);
  }

  GPUDevice *devices = malloc(num_devices * sizeof(GPUDevice));
  if (!devices)
  {
    fprintf(stderr, "Memory allocation failed.\n");
    free(device_ids);
    exit(1);
  }

  for (cl_uint i = 0; i < num_devices; i++)
  {
    devices[i].device_id = device_ids[i];
    err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(devices[i].name), devices[i].name, NULL);
    if (err != CL_SUCCESS)
    {
      fprintf(stderr, "Failed to get device name.\n");
      free(device_ids);
      free(devices);
      exit(1);
    }
    printf("Device %u: %s\n", i, devices[i].name);
  }

  free(device_ids);
  *device_count = num_devices;
  return devices;
}

#else
typedef struct
{
  int device_id;
  char name[256];
} GPUDevice;

GPUDevice *initialize_gpu_devices(int *device_count)
{
  printf("No GPU support available. Falling back to CPU.\n");
  *device_count = 0;
  return NULL;
}
#endif

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
int get_cpu_threads()
{
  int ncpu;
  size_t len = sizeof(ncpu);
  sysctlbyname("hw.logicalcpu", &ncpu, &len, NULL, 0);
  return ncpu;
}
#else
#include <unistd.h>
int get_cpu_threads()
{
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 4;
}
#endif

#define BUFFER_SIZE 8192
#define HASH_SIZE 32
#define DOMAIN_HASH_SIZE 32

typedef struct
{
  char *job_id;
  uint8_t header[DOMAIN_HASH_SIZE];
  uint64_t timestamp;
  volatile int running;
  int thread_index;
} MiningJob;

typedef struct
{
  volatile int running;
  int sockfd;
} StratumContext;

typedef struct
{
  int sockfd;
  int threadIndex;
} MiningThread;

MiningJob job = {0};
pthread_mutex_t job_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t *mining_threads = NULL;
int num_threads = 0;
uint8_t *global_target = NULL;
pthread_mutex_t target_mutex = PTHREAD_MUTEX_INITIALIZER;

volatile uint64_t nonces_processed = 0;
volatile uint64_t cpu_accepted = 0;
volatile uint64_t cpu_rejected = 0;
pthread_mutex_t hashrate_mutex = PTHREAD_MUTEX_INITIALIZER;

volatile sig_atomic_t program_running = 1;

// Utility Functions

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

double difficulty_from_target(uint8_t *target)
{
  mpz_t target_val, max_val, quotient;
  mpz_init(target_val);
  mpz_init(max_val);
  mpz_init(quotient);

  mpz_import(target_val, DOMAIN_HASH_SIZE, 1, sizeof(uint8_t), 0, 0, target);
  mpz_set_str(max_val, "00000000FFFF0000000000000000000000000000000000000000000000000000", 16);

  if (mpz_cmp_ui(target_val, 0) == 0)
  {
    mpz_clear(target_val);
    mpz_clear(max_val);
    mpz_clear(quotient);
    return 0.0;
  }

  mpz_tdiv_q(quotient, max_val, target_val);
  double difficulty = mpz_get_d(quotient);

  mpz_clear(target_val);
  mpz_clear(max_val);
  mpz_clear(quotient);
  return difficulty;
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

// uint8_t *target_from_pool_difficulty(double difficulty)
// {
//   uint8_t *target_bytes = calloc(DOMAIN_HASH_SIZE, sizeof(uint8_t));
//   // Hardcode stratum target for testing
//   const char *stratum_target = "0001869ffffffffff00000000000000000000000000000000000000000000000";
//   hex_to_bytes(stratum_target, target_bytes, DOMAIN_HASH_SIZE);
//   printf("Using hardcoded stratum target: 0x%s\n", stratum_target);
//   return target_bytes;
// }

uint8_t *target_from_pool_difficulty(double difficulty)
{
  if (difficulty <= 0)
    return NULL;

  mpz_t max_target, target;
  mpf_t diff_mpf, max_mpf;

  mpz_init(max_target);
  mpz_init(target);
  mpf_init2(diff_mpf, 512);
  mpf_init2(max_mpf, 512);

  mpz_set_str(max_target, "00000000FFFF0000000000000000000000000000000000000000000000000000", 16);
  mpf_set_d(diff_mpf, difficulty);
  mpf_set_z(max_mpf, max_target);
  mpf_div(diff_mpf, max_mpf, diff_mpf);
  mpf_floor(diff_mpf, diff_mpf);
  mpz_set_f(target, diff_mpf);

  uint8_t *target_bytes = calloc(DOMAIN_HASH_SIZE, sizeof(uint8_t));
  if (!target_bytes)
    goto cleanup;

  size_t count;
  mpz_export(target_bytes, &count, 1, sizeof(uint8_t), 0, 0, target);
  if (count < DOMAIN_HASH_SIZE)
  {
    memmove(target_bytes + (DOMAIN_HASH_SIZE - count), target_bytes, count);
    memset(target_bytes, 0, DOMAIN_HASH_SIZE - count);
  }
  else if (count > DOMAIN_HASH_SIZE)
  {
    memmove(target_bytes, target_bytes + (count - DOMAIN_HASH_SIZE), DOMAIN_HASH_SIZE);
  }

  // printf("Computed Target: 0x");
  // for (int i = 0; i < DOMAIN_HASH_SIZE; i++)
  //   printf("%02x", target_bytes[i]);
  // printf("\nComputed Difficulty: %.6f\n", difficulty_from_target(target_bytes));

cleanup:
  mpz_clear(max_target);
  mpz_clear(target);
  mpf_clear(diff_mpf);
  mpf_clear(max_mpf);
  return target_bytes;
}

int compare_target(uint8_t *hash, uint8_t *target)
{
  uint8_t reversed_hash[DOMAIN_HASH_SIZE];
  for (int i = 0; i < DOMAIN_HASH_SIZE; i++)
  {
    reversed_hash[i] = hash[DOMAIN_HASH_SIZE - 1 - i];
  }
  for (size_t i = 0; i < DOMAIN_HASH_SIZE; i++)
  {
    if (reversed_hash[i] > target[i])
      return 1;
    if (reversed_hash[i] < target[i])
      return -1;
  }
  return 0;
}

void smallJobHeader(const uint64_t *ids, uint8_t *headerData)
{
  for (int i = 0; i < 4; i++)
  {
    uint64_to_little_endian(ids[i], headerData + i * 8);
  }
}

void print_hex(const char *label, const uint8_t *data, size_t len)
{
  printf("%s: 0x", label);
  for (size_t i = 0; i < len; i++)
    printf("%02x", data[i]);
  printf("\n");
}

void reverse_string_in_place(char *str)
{
  if (str == NULL)
  {
    return;
  }

  size_t len = strlen(str);
  if (len <= 1)
  {
    return;
  }

  for (size_t i = 0; i < len / 2; i++)
  {
    char temp = str[i];
    str[i] = str[len - 1 - i];
    str[len - 1 - i] = temp;
  }
}

void reverse_uint8_array(uint8_t *arr, size_t length)
{
  if (arr == NULL || length <= 1)
  {
    return;
  }

  size_t left = 0;
  size_t right = length - 1;

  while (left < right)
  {
    uint8_t temp = arr[left];
    arr[left] = arr[right];
    arr[right] = temp;

    left++;
    right--;
  }
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

void *hashrate_display_thread(void *arg)
{
  while (program_running)
  {
    sleep(1);
    pthread_mutex_lock(&hashrate_mutex);
    double hashrate = nonces_processed;
    nonces_processed = 0;
    pthread_mutex_unlock(&hashrate_mutex);

    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    char time_str[9];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm);

    printf("[%-6s] ==============================================================================\n", time_str);
    printf("[%-6s] |[hoohash]\t\t\t| Accepted shares \t| Rejected shares \t|\n", time_str);
    printf("[%-6s] ", time_str);
    if (hashrate > 1000000000000)
    {
      printf("|CPU : %.2f TH/s\t\t|  ", hashrate);
    }
    else if (hashrate > 1000000000)
    {
      printf("|CPU : %.2f GH/s\t\t|  ", hashrate / 1000.0);
    }
    else if (hashrate > 1000000)
    {
      printf("|CPU : %.2f MH/s\t\t|  ", hashrate / 1000.0);
    }
    else if (hashrate > 1000)
    {
      printf("|CPU : %.2f KH/s\t\t|  ", hashrate / 1000.0);
    }
    else
    {
      printf("|CPU : %.2f H/s\t\t|  ", hashrate);
    }
    if (cpu_accepted > 1000000000000)
    {
      printf("%ld \t| ", cpu_accepted);
    }
    else if (cpu_accepted > 1000)
    {
      printf("%ld \t\t| ", cpu_accepted);
    }
    else
    {
      printf("%ld \t\t\t| ", cpu_accepted);
    }
    if (cpu_rejected > 1000000000000)
    {
      printf("%ld \t|\n", cpu_rejected);
    }
    else if (cpu_rejected > 1000)
    {
      printf("%ld \t\t|\n", cpu_rejected);
    }
    else
    {
      printf("%ld \t\t\t|\n", cpu_rejected);
    }
    printf("[%-6s] ==============================================================================\n", time_str);
  }
  return NULL;
}

uint8_t global_smallest_hash[DOMAIN_HASH_SIZE] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
pthread_mutex_t smallest_hash_mutex = PTHREAD_MUTEX_INITIALIZER;

void *mining_thread(void *arg)
{
  MiningThread *mt = (MiningThread *)arg;
  State state = {0};
  char *current_job_id = NULL;

  while (program_running)
  {
    pthread_mutex_lock(&job_mutex);
    int is_running = job.running;
    char *job_id = NULL;
    if (job.job_id)
    {
      job_id = malloc(strlen(job.job_id) + 1);
      if (!job_id)
      {
        fprintf(stderr, "Memory allocation failed\n");
        pthread_mutex_unlock(&job_mutex);
        exit(1);
      }
      strcpy(job_id, job.job_id);
    }
    if (job_id && is_running)
    {
      memcpy(state.PrevHeader, job.header, DOMAIN_HASH_SIZE);
      state.Timestamp = job.timestamp;
    }
    pthread_mutex_unlock(&job_mutex);

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
    uint64_t step = num_threads;
    generateHoohashMatrix(state.PrevHeader, state.mat);

    while (job.running)
    {
      pthread_mutex_lock(&job_mutex);
      int job_changed = job.job_id == NULL || current_job_id == NULL || strcmp(job.job_id, current_job_id) != 0;
      int still_running = job.running;
      pthread_mutex_unlock(&job_mutex);

      if (job_changed || !still_running)
        break;

      state.Nonce = nonce;
      uint8_t result[DOMAIN_HASH_SIZE];
      CalculateProofOfWorkValue(&state, result);

      pthread_mutex_lock(&target_mutex);
      int meets_target = compare_target(result, global_target) <= 0;
      pthread_mutex_unlock(&target_mutex);

      if (meets_target)
      {
        submit_mining_solution(mt->sockfd, "worker", current_job_id, nonce, result);
        usleep(500000);
      }

      pthread_mutex_lock(&hashrate_mutex);
      nonces_processed++;
      pthread_mutex_unlock(&hashrate_mutex);

      nonce += step;
    }
  }

  free(current_job_id);
  free(mt);
  return NULL;
}

void cleanup_job()
{
  pthread_mutex_lock(&job_mutex);
  free(job.job_id);
  job.job_id = NULL;
  job.running = 0;
  pthread_mutex_unlock(&job_mutex);
}

void cleanup_mining_threads()
{
  if (mining_threads)
  {
    for (int i = 0; i < num_threads; i++)
    {
      pthread_cancel(mining_threads[i]);
      pthread_join(mining_threads[i], NULL);
    }
    free(mining_threads);
    mining_threads = NULL;
  }
}

int start_mining_threads(StratumContext *ctx)
{
  mining_threads = malloc(num_threads * sizeof(pthread_t));
  if (!mining_threads)
  {
    printf("start_mining_threads: Failed to allocate mining_threads\n");
    cleanup_job();
    return 1;
  }

  for (int i = 0; i < num_threads; i++)
  {
    MiningThread *mt = malloc(sizeof(MiningThread));
    if (!mt)
    {
      printf("start_mining_threads: Failed to allocate MiningThread\n");
      cleanup_mining_threads();
      return 1;
    }
    mt->threadIndex = i;
    mt->sockfd = ctx->sockfd;
    if (pthread_create(&mining_threads[i], NULL, mining_thread, mt) != 0)
    {
      printf("start_mining_threads: Failed to create thread %d\n", i);
      free(mt);
      cleanup_mining_threads();
      return 1;
    }
  }
  return 0;
}

void process_stratum_message(int sockfd, json_object *message)
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
          pthread_mutex_lock(&target_mutex);
          if (global_target)
            free(global_target);
          double difficulty = json_object_get_double(diff);
          global_target = target_from_pool_difficulty(difficulty);
          pthread_mutex_unlock(&target_mutex);
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

      pthread_mutex_lock(&job_mutex);
      free(job.job_id);

      const char *jid = json_object_get_string(job_id);
      if (jid)
      {
        job.job_id = malloc(strlen(jid) + 1);
        if (job.job_id)
        {
          strcpy(job.job_id, jid);
          memcpy(job.header, header, DOMAIN_HASH_SIZE);
          job.timestamp = timestamp_int;
          job.running = 1;
          // print_hex("Stored Job Header", job.header, DOMAIN_HASH_SIZE);
        }
        else
        {
          printf("Failed to allocate memory for job ID\n");
        }
      }
      else
      {
        job.job_id = NULL;
        printf("Job ID string is NULL\n");
      }

      pthread_mutex_unlock(&job_mutex);
    }
  }
  else
  {
    if (job.running)
    {
      json_object *result;
      if (json_object_object_get_ex(message, "result", &result))
      {
        if (json_object_is_type(result, json_type_boolean) && json_object_get_boolean(result))
        {
          pthread_mutex_lock(&job_mutex);
          job.running = 0;
          pthread_mutex_unlock(&job_mutex);
          cpu_accepted++;
        }
        else
        {
          cpu_rejected++;
        }
      }
    }
  }
}

void *stratum_receive_thread(void *arg)
{
  StratumContext *ctx = (StratumContext *)arg;
  char buffer[BUFFER_SIZE];
  char json_buffer[BUFFER_SIZE * 2] = {0};
  size_t json_len = 0;

  while (program_running && ctx->running)
  {
    fd_set fds;
    struct timeval tv = {1, 0};
    FD_ZERO(&fds);
    FD_SET(ctx->sockfd, &fds);

    if (select(ctx->sockfd + 1, &fds, NULL, NULL, &tv) <= 0)
      continue;

    int bytes = recv(ctx->sockfd, buffer, BUFFER_SIZE - 1, 0);
    if (bytes <= 0)
    {
      printf("stratum_receive_thread: Connection lost or error (%d)\n", bytes);
      ctx->running = 0;
      break;
    }

    buffer[bytes] = '\0';
    if (json_len + bytes >= sizeof(json_buffer))
    {
      printf("stratum_receive_thread: Buffer overflow, resetting\n");
      json_len = 0;
      continue;
    }

    memcpy(json_buffer + json_len, buffer, bytes);
    json_len += bytes;

    char *start = json_buffer;
    char *end;
    while ((end = strchr(start, '\n')))
    {
      *end = '\0';
      json_object *msg = json_tokener_parse(start);
      if (msg)
      {
        process_stratum_message(ctx->sockfd, msg);
        json_object_put(msg);
      }
      else
        printf("stratum_receive_thread: Failed to parse JSON: %s\n", start);
      json_len -= (end - start + 1);
      start = end + 1;
    }
    if (json_len > 0)
      memmove(json_buffer, start, json_len);
  }
  return NULL;
}

int stratum_subscribe(int sockfd)
{
  json_object *req = json_object_new_object();
  json_object_object_add(req, "id", json_object_new_int(1));
  json_object_object_add(req, "method", json_object_new_string("mining.subscribe"));
  json_object *params = json_object_new_array();
  json_object_array_add(params, json_object_new_string("Hoominer/0.0.0"));
  json_object_object_add(req, "params", params);

  const char *msg = json_object_to_json_string_ext(req, JSON_C_TO_STRING_PLAIN);
  if (!msg)
  {
    json_object_put(req);
    return -1;
  }
  size_t len = strlen(msg);
  char *msg_with_newline = malloc(len + 2);
  if (!msg_with_newline)
  {
    json_object_put(req);
    return -1;
  }

  strcpy(msg_with_newline, msg);
  msg_with_newline[len] = '\n';
  msg_with_newline[len + 1] = '\0';

  int ret = send(sockfd, msg_with_newline, len + 1, 0);
  free(msg_with_newline);
  json_object_put(req);
  return ret < 0 ? -1 : 0;
}

int stratum_authenticate(int sockfd, const char *username, const char *password)
{
  json_object *req = json_object_new_object();
  json_object_object_add(req, "id", json_object_new_int(1));
  json_object_object_add(req, "method", json_object_new_string("mining.authorize"));
  json_object *params = json_object_new_array();
  json_object_array_add(params, json_object_new_string(username));
  json_object_array_add(params, json_object_new_string(password));
  json_object_object_add(req, "params", params);

  const char *msg = json_object_to_json_string_ext(req, JSON_C_TO_STRING_PLAIN);
  if (!msg)
  {
    json_object_put(req);
    return -1;
  }
  size_t len = strlen(msg);
  char *msg_with_newline = malloc(len + 2);
  if (!msg_with_newline)
  {
    json_object_put(req);
    return -1;
  }

  strcpy(msg_with_newline, msg);
  msg_with_newline[len] = '\n';
  msg_with_newline[len + 1] = '\0';

  int ret = send(sockfd, msg_with_newline, len + 1, 0);
  free(msg_with_newline);
  json_object_put(req);
  return ret < 0 ? -1 : 0;
}

int connect_to_stratum_server(const char *hostname, int port)
{
  struct addrinfo hints = {0}, *res, *p;
  int sockfd;
  char port_str[6];
  snprintf(port_str, sizeof(port_str), "%d", port);

  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(hostname, port_str, &hints, &res) != 0)
  {
    perror("getaddrinfo error");
    return -1;
  }

  for (p = res; p != NULL; p = p->ai_next)
  {
    sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sockfd < 0)
      continue;

    if (connect(sockfd, p->ai_addr, p->ai_addrlen) == 0)
    {
      char ip_str[INET_ADDRSTRLEN];
      struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
      inet_ntop(AF_INET, &(ipv4->sin_addr), ip_str, INET_ADDRSTRLEN);
      printf("Connected to %s (resolved IP: %s)\n", hostname, ip_str);
      break;
    }
    close(sockfd);
  }

  freeaddrinfo(res);
  if (p == NULL)
  {
    perror("Failed to connect");
    return -1;
  }
  return sockfd;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void parse_args(int argc, char **argv, char **pool_ip, int *pool_port, const char **username, const char **password, int *threads)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--user") && i + 1 < argc)
      *username = argv[++i];
    else if (!strcmp(argv[i], "--pass") && i + 1 < argc)
      *password = argv[++i];
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
      printf("Usage: %s [--stratum <stratum+tcp://domain:port>] [--user <user>] [--pass <pass>] [--cpu-threads <n>]\n", argv[0]);
      exit(0);
    }
  }
}

void cleanup(int sig)
{
  cleanup_mining_threads();
  cleanup_job();
  pthread_mutex_lock(&target_mutex);
  if (global_target)
  {
    free(global_target);
    global_target = NULL;
  }
  pthread_mutex_unlock(&target_mutex);
  program_running = 0;
}

int main(int argc, char **argv)
{
  signal(SIGINT, cleanup);
  signal(SIGPIPE, SIG_IGN);

  char *pool_ip = NULL;
  int pool_port = 5555;
  const char *username = "user";
  const char *password = "x";
  num_threads = get_cpu_threads() * 2;

  int device_count = 0;
  GPUDevice *devices = initialize_gpu_devices(&device_count);

  if (devices)
  {
    for (int i = 0; i < device_count; i++)
    {
      printf("Device %d: %s\n", devices[i].device_id, devices[i].name);
    }
    free(devices);
  }

  parse_args(argc, argv, &pool_ip, &pool_port, &username, &password, &num_threads);
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

  while (program_running)
  {
    StratumContext *ctx = malloc(sizeof(StratumContext));
    if (!ctx)
    {
      printf("Failed to allocate StratumContext\n");
      free(pool_ip);
      return 1;
    }
    ctx->running = 1;
    ctx->sockfd = connect_to_stratum_server(pool_ip, pool_port);
    if (ctx->sockfd < 0)
    {
      printf("Failed to connect to stratum server. Retrying in 5 seconds...\n");
      free(pool_ip);
      free(ctx);
      sleep(5);
      continue;
    }

    if (stratum_subscribe(ctx->sockfd) < 0 || stratum_authenticate(ctx->sockfd, username, password) < 0)
    {
      printf("Stratum initialization failed. Retrying in 5 seconds...\n");
      close(ctx->sockfd);
      free(pool_ip);
      free(ctx);
      sleep(5);
      continue;
    }

    pthread_t recv_thread, display_thread;
    if (pthread_create(&display_thread, NULL, hashrate_display_thread, NULL) != 0 ||
        pthread_create(&recv_thread, NULL, stratum_receive_thread, ctx) != 0)
    {
      printf("Failed to create threads. Retrying in 5 seconds...\n");
      pthread_cancel(display_thread);
      pthread_join(display_thread, NULL);
      pthread_cancel(recv_thread);
      pthread_join(recv_thread, NULL);
      close(ctx->sockfd);
      free(pool_ip);
      free(ctx);
      sleep(5);
      continue;
    }

    if (start_mining_threads(ctx) != 0)
    {
      printf("Failed to start mining threads. Retrying in 5 seconds...\n");
      pthread_cancel(recv_thread);
      pthread_cancel(display_thread);
      pthread_join(recv_thread, NULL);
      pthread_join(display_thread, NULL);
      close(ctx->sockfd);
      free(pool_ip);
      free(ctx);
      sleep(5);
      continue;
    }

    while (program_running && ctx->running)
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