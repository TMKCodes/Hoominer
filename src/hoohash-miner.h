#ifndef HOOHASH_MINER_H
#define HOOHASH_MINER_H
#include <json-c/json.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <signal.h>
#include <endian.h>
#include <time.h>
#include "stratum.h"
#include "target.h"
#include "opencl.h"
#include "reporting.h"
#include "../algorithms/hoohash/hoohash.h"

typedef struct StratumContext StratumContext;
typedef struct MiningState MiningState;
typedef struct MiningJob MiningJob;
typedef struct MiningThread MiningThread;
typedef struct DeviceThread DeviceThread;
typedef struct QueuedJob QueuedJob;
typedef struct QueuedSolution QueuedSolution;
typedef struct SolutionQueue SolutionQueue;
typedef struct JobQueue JobQueue;

#define JOB_QUEUE_SIZE 4
struct QueuedJob
{
  char *job_id;
  uint8_t header[DOMAIN_HASH_SIZE];
  uint64_t timestamp;
  double matrix[64][64]; // Cache Hoohash matrix
  int running;
};

struct JobQueue
{
  QueuedJob jobs[JOB_QUEUE_SIZE];
  int head, tail;
  pthread_mutex_t queue_mutex;
  pthread_cond_t queue_cond;
};

// Solution queue for asynchronous submission
#define SOLUTION_QUEUE_SIZE 16
struct QueuedSolution
{
  char *worker;
  char *job_id;
  uint64_t nonce;
  uint8_t hash[DOMAIN_HASH_SIZE];
};

struct SolutionQueue
{
  QueuedSolution solutions[SOLUTION_QUEUE_SIZE];
  int head, tail;
  pthread_mutex_t queue_mutex;
  pthread_cond_t queue_cond;
  int sockfd;
};
struct DeviceThread
{
  StratumContext *ctx;
  OpenCLResources *res;
  cl_ulong nonce_base;
  cl_ulong nonce_mask;
  cl_ulong global_size;
  cl_ulong local_size;
  cl_uint index;
  State *state;
  char *current_job_id;
};

struct MiningJob
{
  char *job_id;
  uint8_t header[DOMAIN_HASH_SIZE];
  uint64_t timestamp;
  volatile int running;
  int thread_index;
};

struct MiningState
{
  int num_cpu_threads;
  uint8_t *global_target;
  MiningJob *job;
  pthread_mutex_t job_mutex;
  pthread_t *mining_cpu_threads;
  pthread_t *mining_opencl_threads;
  pthread_mutex_t target_mutex;
};

struct MiningThread
{
  int threadIndex;
  StratumContext *ctx;
};

MiningState *init_mining_state();
void cleanup_mining_state(MiningState *state);
void uint64_to_little_endian(uint64_t value, uint8_t *buffer);
uint64_t little_endian_to_uint64(const uint8_t *buffer);
void smallJobHeader(const uint64_t *ids, uint8_t *headerData);
int hex_to_bytes(const char *hex, uint8_t *bytes, size_t len);
void print_hex(const char *label, const uint8_t *data, size_t len);
int submit_mining_solution(int sockfd, const char *worker, const char *job_id, uint64_t nonce, uint8_t *hash);
void *mining_cpu_thread(void *arg);
void cleanup_job(MiningState *ms);
void cleanup_mining_threads(MiningState *ms);
int start_mining_threads(StratumContext *ctx, MiningState *ms);
void process_stratum_message(json_object *message, StratumContext *ctx, MiningState *ms);

#endif