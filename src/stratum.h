#define _POSIX_C_SOURCE 200809L
#ifndef STRATUM_H
#define STRATUM_H
#include <assert.h>
#include <json-c/json.h>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>
#include <endian.h>
#include <time.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <stdbool.h>
#include <string.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include "config.h"
#include "datatypes.h"
#include "reporting.h"
#include "miner-hoohash.h"

#define BUFFER_SIZE 16384

typedef struct StratumContext StratumContext;
typedef struct OpenCLResources OpenCLResources;
typedef struct HashrateDisplay HashrateDisplay;
typedef struct MiningState MiningState;
typedef struct CudaResources CudaResources;
typedef struct HoominerConfig HoominerConfig;

struct StratumContext
{
  char *version;
  volatile int running;
  int sockfd;
  SSL *ssl;         // SSL connection object
  SSL_CTX *ssl_ctx; // SSL context
  const char *worker;
  int disable_cpu;
  int disable_gpu;
  unsigned int cpu_device_count;
  unsigned int opencl_device_count;
  unsigned int cuda_device_count;
  OpenCLResources *opencl_resources;
  CudaResources *cuda_resources;
  HashrateDisplay *hd;
  MiningState *ms;
  pthread_t recv_thread;
  IntFifo mining_submit_fifo;
  HoominerConfig *config;
};

StratumContext *init_stratum_context();
void *stratum_receive_thread(void *arg);
int start_stratum_connection(StratumContext *ctx, HoominerConfig *config);
int stratum_subscribe(int sockfd, StratumContext *ctx, SSL *ssl);
int stratum_authenticate(int sockfd, const char *username, const char *password, SSL *ssl);
int connect_to_stratum_server(const char *hostname, int port);
int init_ssl_connection(StratumContext *ctx);
void process_stratum_message(json_object *message, StratumContext *ctx, MiningState *ms);
void free_stratum_context(StratumContext *ctx);

#endif