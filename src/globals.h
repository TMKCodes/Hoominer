#include <stdint.h>
#include <pthread.h>
#include <signal.h>

#define DOMAIN_HASH_SIZE 32

// Forward declarations
typedef struct HashrateDisplay HashrateDisplay;
typedef struct MiningState MiningState;
typedef struct MiningJob MiningJob;
typedef struct StratumContext StratumContext;
typedef struct MiningThread MiningThread;

// Struct definitions
struct HashrateDisplay
{
  uint64_t nonces_processed;
  uint64_t cpu_accepted;
  uint64_t cpu_rejected;
  pthread_mutex_t hashrate_mutex;
  sig_atomic_t running;
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
  MiningJob job;
  pthread_mutex_t job_mutex;
  pthread_t *mining_threads;
  pthread_mutex_t target_mutex;
};

struct StratumContext
{
  volatile int running;
  int sockfd;
  HashrateDisplay *hd;
  MiningState *ms;
};

struct MiningThread
{
  int threadIndex;
  StratumContext *ctx;
};
